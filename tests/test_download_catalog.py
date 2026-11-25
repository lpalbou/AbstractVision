"""`abstractvision download` and the model listings: package selection, the files a package needs,
gated access, disk space, the download plan against the cache, schedule adapters, cache state and
memory modes.

No network and no weights: the Hugging Face client is a fake that serves the repositories' real file
names (sizes included), writes huggingface_hub's cache layout, and refuses access the way the Hub does.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from abstractvision import model_downloads as md  # noqa: E402

GIB = 1024**3

try:
    import huggingface_hub  # noqa: F401

    _HUB = True
except ImportError:  # the base CI job installs no optional dependency
    _HUB = False
# The fake Hub stands in for huggingface_hub's functions, so the module itself must be importable.
_needs_hub = unittest.skipUnless(_HUB, "huggingface_hub is not installed")

# AbstractFramework/minimax-h3-8bit as the Hub lists it (sizes rounded; the names are the real ones).
H3_8BIT_FILES = {
    ".gitattributes": 1519,
    "LICENSE": 17604,
    "NOTICE": 604,
    "README.md": 8868,
    "audio_scheduler/scheduler_config.json": 96,
    "audio_vae/0.safetensors": 605_251_263,
    "audio_vae/config.json": 2271,
    "audio_vae/model.safetensors.index.json": 61234,
    "model_index.json": 2936,
    "modular_model_index.json": 2935,
    "scheduler/scheduler_config.json": 500,
    "text_encoder/0.safetensors": 27_900_000_000,
    "text_encoder/config.json": 4000,
    "tokenizer/tokenizer.json": 11_000_000,
    "transformer/0.safetensors": 35_270_000_000,
    "transformer/config.json": 546,
    "transformer/model.safetensors.index.json": 88782,
    "vae/0.safetensors": 5_200_000_000,
    "vae/config.json": 900,
}

# AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit: the A14B models carry a second expert.
WAN_A14B_FILES = {
    "model_index.json": 498,
    "transformer/0.safetensors": 2_126_914_156,
    "transformer/model.safetensors.index.json": 98380,
    "transformer_2/0.safetensors": 2_126_914_156,
    "transformer_2/model.safetensors.index.json": 98380,
    "vae/diffusion_pytorch_model.safetensors": 507_000_000,
}


def _etag(index, name):
    return f"sha{index:04d}" if name.endswith(".safetensors") else f"blob{index:04d}"


def _siblings(files):
    return [
        SimpleNamespace(
            rfilename=name,
            size=size,
            blob_id=f"blob{i:04d}",
            lfs=SimpleNamespace(sha256=f"sha{i:04d}") if name.endswith(".safetensors") else None,
        )
        for i, (name, size) in enumerate(sorted(files.items()))
    ]


class _FakeHub:
    """Stands in for huggingface_hub and writes its cache layout: `blobs/<etag>`, the snapshot links
    to them, `refs/main`, and `blobs/<etag>.incomplete` while a file is in flight.

    `fetched` lists the files it fetched (a file whose snapshot link or blob exists is not fetched,
    as in huggingface_hub). `gated` refuses downloads (403); `listing_error` is raised by the
    listing; `interrupt_after` fetches that many files, leaves a partial blob and raises
    KeyboardInterrupt; `drop` names files the download leaves out."""

    SHA = "c0ffee"

    def __init__(
        self, files, *, gated=False, hub_gated=False, listing_error=None, interrupt_after=None, drop=(), contents=None
    ):
        self.files = dict(files)
        self.gated = gated
        self.hub_gated = hub_gated
        self.listing_error = listing_error
        self.interrupt_after = interrupt_after
        self.drop = set(drop)
        self.contents = dict(contents or {})
        self.downloads = []
        self.fetched = []
        self.listings = 0

    def api(self, *args, **kwargs):
        hub = self

        class _Api:
            def model_info(self, repo_id, revision=None, files_metadata=False, token=None):
                hub.listings += 1
                if hub.listing_error is not None:
                    raise hub.listing_error
                return SimpleNamespace(siblings=_siblings(hub.files), sha=hub.SHA, gated=hub.hub_gated)

            def list_repo_files(self, repo_id, token=None, **kwargs):
                if hub.listing_error is not None:
                    raise hub.listing_error
                return sorted(hub.files)

        return _Api()

    def content(self, name):
        if name in self.contents:
            return self.contents[name]
        return "{}" if name.endswith(".json") else "x"

    def hf_hub_download(self, repo_id, filename, *, cache_dir=None, **kwargs):
        # What huggingface_hub does: the file lands in the cache (blob, snapshot link, refs/main).
        return str(self._store(Path(cache_dir or md.default_hf_cache_root()), repo_id, filename))

    def snapshot_download(self, *, repo_id, cache_dir, allow_patterns=None, ignore_patterns=None, **kwargs):
        from huggingface_hub.utils import filter_repo_objects

        self.downloads.append({"repo_id": repo_id, "allow_patterns": allow_patterns, "token": kwargs.get("token")})
        if self.gated:
            error = RuntimeError(f"403 Client Error. Cannot access gated repo for url https://huggingface.co/{repo_id}.")
            error.response = SimpleNamespace(status_code=403)
            raise error
        repo_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
        names = filter_repo_objects(list(self.files), allow_patterns=allow_patterns, ignore_patterns=ignore_patterns)
        for name in names:
            if name not in self.drop:
                self._store(Path(cache_dir), repo_id, name)
        return str(repo_dir / "snapshots" / self.SHA)

    def _store(self, cache_dir, repo_id, name):
        repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / "main").write_text(self.SHA)
        pointer = repo_dir / "snapshots" / self.SHA / name
        if pointer.exists():
            return pointer
        index = sorted(self.files).index(name)
        blob = repo_dir / "blobs" / _etag(index, name)
        blob.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists():
            partial = blob.with_name(blob.name + ".incomplete")
            if self.interrupt_after is not None and len(self.fetched) >= self.interrupt_after:
                partial.write_bytes(b"\0" * 7)
                raise KeyboardInterrupt
            self.fetched.append(name)
            blob.write_text(self.content(name), encoding="utf-8")
            if partial.exists():
                partial.unlink()  # the library renames the finished `.incomplete` file to the blob
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.symlink_to(os.path.relpath(blob, pointer.parent))
        return pointer

    def session(self):
        hub = self

        class _Session:
            def get(self, url, headers=None, follow_redirects=False):
                import httpx

                name = url.split("/resolve/", 1)[1].split("/", 1)[1]
                return httpx.Response(200, text=hub.content(name), request=httpx.Request("GET", url))

        return _Session()


class _Env(contextlib.ExitStack):
    """A private HF cache, an Apple Silicon host, a fake hub and no cached schedule adapters."""

    def __init__(self, hub, *, free_bytes=4096 * GIB):
        super().__init__()
        self.hub = hub
        self.free_bytes = free_bytes

    def __enter__(self):
        super().__enter__()
        self.cache = Path(self.enter_context(tempfile.TemporaryDirectory()))
        self.enter_context(patch.dict("os.environ", {"HF_HUB_CACHE": str(self.cache)}, clear=True))
        self.enter_context(patch.object(md, "local_model_profile", return_value="apple-silicon"))
        self.enter_context(patch("huggingface_hub.HfApi", side_effect=self.hub.api))
        self.enter_context(patch("huggingface_hub.snapshot_download", side_effect=self.hub.snapshot_download))
        self.enter_context(patch("huggingface_hub.hf_hub_download", side_effect=self.hub.hf_hub_download))
        self.enter_context(patch("huggingface_hub.utils.get_session", side_effect=self.hub.session))
        self.enter_context(
            patch.object(md.shutil, "disk_usage", return_value=SimpleNamespace(free=self.free_bytes))
        )
        self.enter_context(patch("sys.stdin.isatty", return_value=False))
        self.adapters = []
        self.enter_context(
            patch(
                "abstractvision.backends.mflux.resolve_mlx_gen_schedule_adapter",
                side_effect=self._adapter,
            )
        )
        return self

    def repo_dir(self, repo_id):
        return self.cache / f"models--{repo_id.replace('/', '--')}"

    def cached(self, repo_id, names):
        """Put files into the cache as a finished huggingface_hub download leaves them."""

        for name in names:
            self.hub._store(self.cache, repo_id, name)
        self.hub.fetched.clear()

    def written(self):
        return sorted(str(p.relative_to(self.cache)) for p in self.cache.rglob("*") if not p.is_dir())

    def _adapter(self, schedule, *, download=False):
        self.adapters.append((schedule, download))
        if schedule == "minimax-h3":
            return None
        return ("hf:lightx2v/Minimax-h3-Turbo/turbo.safetensors", "/cache/turbo.safetensors" if download else None)


def _run_cli(argv):
    from abstractvision.cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue()


def _run_cli_exit(argv):
    """(exit code, stdout, stderr) of a command that ends with SystemExit."""

    from abstractvision.cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(argv)
        except SystemExit as e:
            code = e.code
    if isinstance(code, str):
        err.write(code)
        code = 1
    return code, out.getvalue(), err.getvalue()


@_needs_hub
class TestPackageFiles(unittest.TestCase):
    def _selected(self, repo_id, files):
        from huggingface_hub.utils import filter_repo_objects

        preset = md.find_model_preset(repo_id, target="mlx", engine="mlx-gen", require_8bit=False)
        return set(filter_repo_objects(list(files), allow_patterns=list(preset.allow_patterns)))

    def test_minimax_h3_packages_download_the_soundtrack_decoder_and_the_notice(self):
        for repo_id in (
            "AbstractFramework/minimax-h3-8bit",
            "AbstractFramework/minimax-h3-bf16",
            "AbstractFramework/minimax-h3-4bit",
        ):
            selected = self._selected(repo_id, H3_8BIT_FILES)
            self.assertIn("audio_vae/0.safetensors", selected, repo_id)
            self.assertIn("audio_scheduler/scheduler_config.json", selected, repo_id)
            self.assertIn("NOTICE", selected, repo_id)
            self.assertNotIn(".gitattributes", selected, repo_id)

    def test_wan_a14b_packages_download_the_second_expert(self):
        for repo_id in (
            "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
            "AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        ):
            self.assertIn("transformer_2/0.safetensors", self._selected(repo_id, WAN_A14B_FILES), repo_id)

    def test_mlx_gen_presets_cover_every_pattern_mlx_gen_downloads(self):
        try:
            from mflux.cli import mlx_gen
        except Exception:
            self.skipTest("mlx-gen is not installed")
        from huggingface_hub.utils import filter_repo_objects

        from abstractvision.backends.mflux import _base_model_for_preset

        checked = 0
        for preset in md.model_presets(target="mlx", engine="mlx-gen", include_non_8bit=True):
            try:
                config = mlx_gen._model_config(preset.repo_id, base_model=_base_model_for_preset(None, preset.repo_id, preset))
                patterns = mlx_gen._download_patterns(config, config.model_name if config else preset.repo_id)
            except Exception:
                continue
            for pattern in patterns or ():
                # A file name each mlx-gen pattern selects must also be one the preset selects.
                example = pattern.replace("**", "x").replace("*", "x")
                with self.subTest(repo=preset.repo_id, pattern=pattern):
                    self.assertTrue(
                        list(filter_repo_objects([example], allow_patterns=list(preset.allow_patterns))),
                        f"{preset.repo_id}: {example!r} ({pattern}) is not downloaded",
                    )
            checked += 1
        self.assertGreater(checked, 20)


class TestPackageSelection(unittest.TestCase):
    def setUp(self):
        self.profile = patch.object(md, "local_model_profile", return_value="apple-silicon")
        self.profile.start()
        self.addCleanup(self.profile.stop)

    def test_the_model_id_selects_the_recommended_package(self):
        for name in ("MiniMaxAI/MiniMax-H3", "minimax-h3"):
            self.assertEqual(md.find_model_preset(name).repo_id, "AbstractFramework/minimax-h3-8bit")

    def test_bits_select_a_package_by_width(self):
        self.assertEqual(md.find_model_preset("MiniMaxAI/MiniMax-H3", bits=4).repo_id, "AbstractFramework/minimax-h3-4bit")
        self.assertEqual(md.find_model_preset("MiniMaxAI/MiniMax-H3", bits=16).repo_id, "AbstractFramework/minimax-h3-bf16")
        self.assertEqual(md.find_model_preset("MiniMaxAI/MiniMax-H3", bits=8).repo_id, "AbstractFramework/minimax-h3-8bit")
        with self.assertRaisesRegex(ValueError, "No 2-bit package .*minimax-h3-4bit"):
            md.find_model_preset("MiniMaxAI/MiniMax-H3", bits=2)

    def test_package_repo_ids_select_themselves_and_conflicting_bits_are_refused(self):
        for repo_id, bits in (
            ("AbstractFramework/minimax-h3-8bit", 8),
            ("AbstractFramework/minimax-h3-bf16", 16),
            ("minimax-h3-4bit", 4),
        ):
            self.assertEqual(md.find_model_preset(repo_id).quantization_bits, bits)
        with self.assertRaisesRegex(ValueError, "is the 8-bit package"):
            md.find_model_preset("AbstractFramework/minimax-h3-8bit", bits=4)

    def test_the_source_snapshot_is_an_explicit_non_quantized_choice(self):
        preset = md.find_model_preset("MiniMaxAI/MiniMax-H3", require_8bit=False)
        self.assertEqual((preset.repo_id, preset.source), ("MiniMaxAI/MiniMax-H3", "official"))

    def test_other_hosts_get_the_recommended_package_too(self):
        with patch.object(md, "local_model_profile", return_value="cpu"):
            self.assertEqual(md.find_model_preset("MiniMaxAI/MiniMax-H3").repo_id, "AbstractFramework/minimax-h3-8bit")

    def test_models_without_a_recommendation_stay_exact(self):
        with self.assertRaisesRegex(ValueError, "names several MLX-Gen packages"):
            md.find_model_preset("black-forest-labs/FLUX.2-klein-4B", target="mlx", engine="mlx-gen")
        preset = md.find_model_preset("black-forest-labs/FLUX.2-klein-4B", target="mlx", engine="mlx-gen", bits=4)
        self.assertEqual(preset.repo_id, "AbstractFramework/flux.2-klein-4b-4bit")


class TestCatalogAccessFacts(unittest.TestCase):
    def test_minimax_h3_access_licence_and_recommendation(self):
        presets = {p.repo_id: p for p in md.model_presets(target="mlx", engine="mlx-gen", include_non_8bit=True)}
        for repo_id in ("AbstractFramework/minimax-h3-8bit", "AbstractFramework/minimax-h3-bf16", "AbstractFramework/minimax-h3-4bit"):
            self.assertTrue(presets[repo_id].gated, repo_id)
            self.assertEqual(presets[repo_id].license_name, "MiniMax H3 Community License")
        self.assertFalse(presets["MiniMaxAI/MiniMax-H3"].gated)
        self.assertEqual([r for r, p in presets.items() if p.recommended], ["AbstractFramework/minimax-h3-8bit"])

    def test_every_preset_of_a_gated_catalog_repository_is_gated(self):
        from abstractvision.model_capabilities import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        gated = {dl.repo_id for mid in reg.list_models() for dl in reg.get(mid).downloads if dl.gated}
        self.assertIn("black-forest-labs/FLUX.1-dev", gated)
        presets = md.model_presets(target="auto", include_non_8bit=True, include_all_targets=True)
        self.assertEqual({p.repo_id for p in presets if p.gated}, gated & {p.repo_id for p in presets})


def _repo_not_found(status=401):
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError

    url = "https://huggingface.co/api/models/no-such-org/no-such-vision-model"
    return RepositoryNotFoundError(
        f"{status} Client Error.", response=httpx.Response(status, request=httpx.Request("GET", url))
    )


# Green-Sky/SD-Turbo-GGUF as the Hub lists it (sizes exact, 2026-11-25).
SD_TURBO_GGUF_FILES = {
    ".gitattributes": 1687,
    "README.md": 619,
    "old/sd_turbo-q8_0.gguf": 3_368_976_896,
    "sd_turbo-f16-q8_0.gguf": 2_023_745_376,
    "sd_turbo-f16.gguf": 2_612_748_512,
}

# stabilityai/sd-turbo (a Diffusers pipeline): its model_index.json names five components.
SD_TURBO_DIFFUSERS_FILES = {
    "README.md": 3000,
    "model_index.json": 616,
    "scheduler/scheduler_config.json": 500,
    "sd_turbo.safetensors": 5_200_000_000,
    "text_encoder/config.json": 600,
    "text_encoder/model.fp16.safetensors": 680_000_000,
    "text_encoder/model.safetensors": 1_360_000_000,
    "tokenizer/vocab.json": 1_000_000,
    "unet/config.json": 1500,
    "unet/diffusion_pytorch_model.fp16.safetensors": 1_730_000_000,
    "unet/diffusion_pytorch_model.safetensors": 3_460_000_000,
    "vae/config.json": 600,
    "vae/diffusion_pytorch_model.fp16.safetensors": 167_000_000,
    "vae/diffusion_pytorch_model.safetensors": 335_000_000,
}
SD_TURBO_INDEX = json.dumps(
    {
        "_class_name": "StableDiffusionPipeline",
        "feature_extractor": [None, None],
        "scheduler": ["diffusers", "EulerDiscreteScheduler"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
)
WAN_A14B_INDEX = json.dumps(
    {
        "_class_name": "WanPipeline",
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "transformer_2": ["diffusers", "WanTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
    }
)
WAN_A14B = "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"


@_needs_hub
class TestDownloadCommand(unittest.TestCase):
    def test_gated_refusal_names_the_package_page_and_the_licence(self):
        hub = _FakeHub(H3_8BIT_FILES, gated=True)
        with _Env(hub):
            with self.assertRaises(SystemExit) as ctx:
                _run_cli(["download", "MiniMaxAI/MiniMax-H3"])
        message = str(ctx.exception)
        self.assertIn("https://huggingface.co/AbstractFramework/minimax-h3-8bit", message)
        self.assertIn("MiniMax H3 Community License", message)
        self.assertIn("the repository is gated", message)
        self.assertIn("hf auth login", message)
        self.assertNotIn("Traceback", message)
        self.assertEqual(len(hub.downloads), 1)

    def test_a_download_that_does_not_fit_is_refused_before_it_starts(self):
        files = dict(H3_8BIT_FILES, **{"transformer/0.safetensors": 115 * GIB})
        hub = _FakeHub(files)
        with _Env(hub, free_bytes=50 * GIB):
            with self.assertRaises(SystemExit) as ctx:
                _run_cli(["download", "MiniMaxAI/MiniMax-H3", "--bits", "16"])
        message = str(ctx.exception)
        self.assertIn("Not enough free disk space to download AbstractFramework/minimax-h3-bf16", message)
        self.assertIn("files to fetch take 146.4 GiB", message)
        self.assertIn("50.0 GiB is free", message)
        self.assertEqual(hub.downloads, [])

    def test_open_repositories_say_so_in_download_and_dry_run(self):
        for argv in (["--dry-run"], []):
            with _Env(_FakeHub(WAN_A14B_FILES, contents={"model_index.json": WAN_A14B_INDEX})):
                rc, out = _run_cli(["download", WAN_A14B, *argv])
            self.assertEqual(rc, 0, argv)
            self.assertIn("access: open\n", out, argv)

    def test_a_download_missing_files_says_how_to_fetch_them(self):
        hub = _FakeHub(H3_8BIT_FILES, drop={"audio_vae/0.safetensors"})
        with _Env(hub):
            with self.assertRaises(SystemExit) as ctx:
                _run_cli(["download", "AbstractFramework/minimax-h3-8bit"])
        self.assertIn("is incomplete: 1 of", str(ctx.exception))
        self.assertIn("audio_vae/0.safetensors", str(ctx.exception))
        self.assertIn("Run the same command again to fetch them", str(ctx.exception))

    def test_a_turbo_schedule_also_fetches_its_adapter(self):
        hub = _FakeHub(H3_8BIT_FILES)
        with _Env(hub) as env:
            rc, out = _run_cli(["download", "MiniMaxAI/MiniMax-H3", "--base-model", "minimax-h3-turbo-544p"])
        self.assertEqual(rc, 0)
        self.assertEqual(hub.downloads[0]["repo_id"], "AbstractFramework/minimax-h3-8bit")
        self.assertIn(("minimax-h3-turbo-544p", True), env.adapters)
        self.assertIn("adapter (minimax-h3-turbo-544p): /cache/turbo.safetensors", out)
        self.assertIn("access: gated; accept the MiniMax H3 Community License", out)

    def test_without_a_schedule_the_adapters_are_named_not_fetched(self):
        hub = _FakeHub(H3_8BIT_FILES)
        with _Env(hub) as env:
            rc, out = _run_cli(["download", "AbstractFramework/minimax-h3-8bit"])
        self.assertEqual(rc, 0)
        self.assertNotIn(True, [download for _schedule, download in env.adapters])
        self.assertIn(
            "`abstractvision download AbstractFramework/minimax-h3-8bit --base-model minimax-h3-turbo-544p`", out
        )

    def test_a_schedule_of_another_model_is_refused(self):
        with _Env(_FakeHub(H3_8BIT_FILES)):
            with self.assertRaisesRegex(SystemExit, "not a schedule of AbstractFramework/minimax-h3-8bit"):
                _run_cli(["download", "AbstractFramework/minimax-h3-8bit", "--base-model", "flux2-klein-4b"])

    def test_a_catalog_model_is_never_fetched_as_an_unfiltered_snapshot(self):
        hub = _FakeHub(H3_8BIT_FILES)
        with _Env(hub), patch("abstractvision.cli.repository_preset", side_effect=AssertionError("snapshot")):
            with self.assertRaisesRegex(SystemExit, "No 2-bit package"):
                _run_cli(["download", "MiniMaxAI/MiniMax-H3", "--bits", "2"])
            with self.assertRaisesRegex(SystemExit, "names several MLX-Gen packages"):
                _run_cli(["download", "black-forest-labs/FLUX.2-klein-4B"])
        self.assertEqual(hub.downloads, [])

    def test_dry_run_reports_the_plan_and_downloads_nothing(self):
        hub = _FakeHub(H3_8BIT_FILES)
        with _Env(hub, free_bytes=20 * GIB) as env:
            rc, out = _run_cli(["download", "MiniMaxAI/MiniMax-H3", "--dry-run", "--base-model", "minimax-h3-turbo"])
            self.assertEqual(env.written(), [])
        self.assertEqual(rc, 0)
        self.assertEqual(hub.downloads, [])
        self.assertIn("selected: AbstractFramework/minimax-h3-8bit", out)
        self.assertIn("Not enough free disk space", out)
        self.assertIn("adapter (minimax-h3-turbo): hf:lightx2v/Minimax-h3-Turbo/turbo.safetensors (would be fetched)", out)
        self.assertEqual(
            out.rstrip().splitlines()[-1],
            "dry run: nothing was downloaded; the download would not fit (64.2 GiB to fetch, 20.0 GiB free)",
        )

    def test_a_listing_without_sizes_sizes_nothing_and_fetches_nothing(self):
        files = dict(H3_8BIT_FILES, **{"audio_vae/0.safetensors": None})  # the listing gives no size
        with _Env(_FakeHub(files)):
            rc, out = _run_cli(["download", "MiniMaxAI/MiniMax-H3", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIn("gives no size for audio_vae/0.safetensors, so the download cannot be sized", out)
        self.assertIn("if HF_ENDPOINT points at a mirror, point it at a server that lists file sizes", out)
        self.assertIn("dry run: nothing was downloaded; without the Hub listing nothing was sized", out)
        hub = _FakeHub(files)
        with _Env(hub):
            with self.assertRaisesRegex(SystemExit, "Nothing was downloaded for AbstractFramework/minimax-h3-8bit"):
                _run_cli(["download", "MiniMaxAI/MiniMax-H3"])
        self.assertEqual(hub.downloads, [])

    def test_repo_snapshot_accepts_a_string_cache_dir(self):
        hub = _FakeHub({"model.safetensors": 10, "config.json": 2})
        with _Env(hub) as env:
            path = md.download_hf_repo_snapshot("some-org/some-model", cache_dir=str(env.cache))
            self.assertTrue((path / "model.safetensors").is_file())


@_needs_hub
class TestDownloadPlanAgainstTheCache(unittest.TestCase):
    """D1: a file is complete in the cache or fetched whole; nothing is promised about partial files."""

    def _partial(self, env, repo_id, name, size):
        index = sorted(env.hub.files).index(name)
        blob = env.repo_dir(repo_id) / "blobs" / (_etag(index, name) + ".incomplete")
        blob.parent.mkdir(parents=True, exist_ok=True)
        with open(blob, "wb") as fh:
            fh.truncate(size)

    def test_a_partial_file_is_fetched_whole_whatever_the_transport_left(self):
        # Both huggingface_hub transports keep an interrupted file as `blobs/<etag>.incomplete`: plain HTTP
        # with the bytes it received (it may continue from them), Xet with whatever it last wrote (it starts
        # the file over). The plan treats both alike: the file is to fetch, at its full size.
        repo = "AbstractFramework/minimax-h3-8bit"
        outputs = []
        for partial_bytes in (GIB, 0):
            hub = _FakeHub(H3_8BIT_FILES)
            with _Env(hub) as env:
                env.cached(repo, ["vae/0.safetensors"])
                self._partial(env, repo, "transformer/0.safetensors", partial_bytes)
                rc, out = _run_cli(["download", repo])
            self.assertEqual(rc, 0)
            self.assertIn("transformer/0.safetensors", hub.fetched)
            self.assertNotIn("vae/0.safetensors", hub.fetched)
            self.assertNotIn("resum", out)
            outputs.append([line for line in out.splitlines() if line.startswith("cache:")])
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], ["cache: 1 of 18 files complete (4.8 GiB); 17 files to fetch (59.4 GiB)"])

    def test_the_disk_check_counts_the_files_to_fetch_at_their_full_size(self):
        repo = "AbstractFramework/minimax-h3-8bit"
        hub = _FakeHub(H3_8BIT_FILES)
        # 59.4 GiB to fetch; a 30 GiB partial transformer file is on disk but does not count as fetched.
        with _Env(hub, free_bytes=40 * GIB) as env:
            env.cached(repo, ["vae/0.safetensors"])
            self._partial(env, repo, "transformer/0.safetensors", 30 * GIB)
            with self.assertRaisesRegex(SystemExit, "the 17 files to fetch take 59.4 GiB and 40.0 GiB is free"):
                _run_cli(["download", repo])
        self.assertEqual(hub.downloads, [])

    def test_an_interrupted_download_exits_130_and_says_what_stays(self):
        hub = _FakeHub(SD_TURBO_GGUF_FILES, interrupt_after=1)
        with _Env(hub):
            code, out, err = _run_cli_exit(["download", "Green-Sky/SD-Turbo-GGUF", "--dry-run"])
            self.assertEqual(code, 0)
            with patch("abstractvision.backends.stable_diffusion_cpp._require_sd_cli", return_value="sd-cli"):
                code, out, err = _run_cli_exit(["download", "Green-Sky/SD-Turbo-GGUF"])
            self.assertEqual(code, 130)
            self.assertIn("The files that completed stay in the Hugging Face cache", err)
            self.assertIn("the same command continues with the rest", err)
            self.assertNotIn("resum", err)
            # The next run: one file complete, the interrupted one to fetch whole.
            _rc, out = _run_cli(["download", "Green-Sky/SD-Turbo-GGUF", "--dry-run"])
        self.assertIn("cache: 1 of 2 files complete (619 B); 1 file to fetch (1.9 GiB)", out)


@_needs_hub
class TestDownloadDecidesFromThePlan(unittest.TestCase):
    """D2 and D3: `download` always plans from the Hub listing and fetches exactly what is missing."""

    def test_a_second_download_says_everything_is_present_and_fetches_nothing(self):
        hub = _FakeHub(SD_TURBO_GGUF_FILES)
        with _Env(hub), patch("abstractvision.backends.stable_diffusion_cpp._require_sd_cli", return_value="sd-cli"):
            _run_cli(["download", "Green-Sky/SD-Turbo-GGUF"])
            self.assertEqual(hub.fetched, ["README.md", "sd_turbo-f16-q8_0.gguf"])
            rc, out = _run_cli(["download", "Green-Sky/SD-Turbo-GGUF"])
        self.assertEqual(rc, 0)
        self.assertIn("download size: 1.9 GiB in 2 files", out)
        self.assertIn("cache: all 2 files already present; nothing to fetch", out)
        self.assertEqual(len(hub.downloads), 1)  # the second run requested nothing
        self.assertEqual(hub.fetched, ["README.md", "sd_turbo-f16-q8_0.gguf"])

    def test_a_snapshot_missing_files_the_package_selects_is_completed(self):
        # A Wan 2.2 A14B package as 0.3.36 left it: everything but the low-noise expert (transformer_2/).
        hub = _FakeHub(WAN_A14B_FILES, contents={"model_index.json": WAN_A14B_INDEX})
        with _Env(hub) as env:
            env.cached(WAN_A14B, [n for n in WAN_A14B_FILES if not n.startswith("transformer_2/")])
            from abstractvision.model_inventory import preset_status

            preset = md.find_model_preset(WAN_A14B, target="mlx", engine="mlx-gen", require_8bit=False)
            before = preset_status(preset)
            rc, out = _run_cli(["download", WAN_A14B])
            after = preset_status(preset)
        self.assertEqual(before["cache"], "incomplete")
        self.assertIn("model_index.json names transformer_2, not in the snapshot", before["cache_gaps"])
        self.assertEqual(rc, 0)
        self.assertIn("cache: 4 of 6 files complete (2.5 GiB); 2 files to fetch (2.0 GiB)", out)
        self.assertEqual(sorted(hub.fetched), ["transformer_2/0.safetensors", "transformer_2/model.safetensors.index.json"])
        self.assertEqual(after["cache"], "cached")

    def test_without_the_listing_a_cached_snapshot_is_reported_as_it_is(self):
        import httpx

        repo = "AbstractFramework/minimax-h3-8bit"
        hub = _FakeHub(H3_8BIT_FILES)
        with _Env(hub) as env:
            env.cached(repo, list(H3_8BIT_FILES))
            hub.listing_error = httpx.ConnectError("[Errno 8] nodename nor servname provided")
            rc, out = _run_cli(["download", repo])
        self.assertEqual(rc, 0)
        self.assertIn(
            "hub: no usable listing of AbstractFramework/minimax-h3-8bit: the Hugging Face Hub could not be reached "
            "at https://huggingface.co (ConnectError:",
            out,
        )
        self.assertIn("it was not compared with the Hub, so it may lack files the package selects", out)
        self.assertNotIn("already present", out)
        self.assertEqual(hub.downloads, [])

    def test_without_the_listing_an_incomplete_or_absent_snapshot_stops(self):
        import httpx

        repo = "AbstractFramework/minimax-h3-8bit"
        for cached in ([], ["model_index.json", "README.md"]):
            hub = _FakeHub(H3_8BIT_FILES, listing_error=httpx.ConnectError("offline"))
            with _Env(hub) as env:
                env.cached(repo, cached)
                with self.assertRaises(SystemExit) as ctx:
                    _run_cli(["download", repo])
            message = str(ctx.exception)
            self.assertIn(
                "Nothing was downloaded for AbstractFramework/minimax-h3-8bit: the Hugging Face Hub could not be reached",
                message,
            )
            self.assertIn("no weight file" if cached else "The cache holds nothing of it", message)


@_needs_hub
class TestLegacyModelDir(unittest.TestCase):
    """`--model-dir` / `model_dir` (0.3.36): a legacy `<model_dir>/<local_dir_name>` tree is moved into the cache
    as 0.3.36 did; that snapshot is reported as not compared with the Hub, and nothing is fetched on top."""

    REPO = "AbstractFramework/minimax-h3-8bit"

    def _tree(self, root):
        preset = md.find_model_preset(self.REPO)
        tree = Path(root) / preset.local_dir_name
        (tree / "transformer").mkdir(parents=True)
        (tree / "transformer" / "0.safetensors").write_bytes(b"w" * 64)
        (tree / "model_index.json").write_text("{}", encoding="utf-8")
        return tree, (tree / "transformer" / "0.safetensors").stat().st_ino

    def test_a_legacy_tree_is_moved_into_the_cache_and_nothing_is_fetched(self):
        hub = _FakeHub(H3_8BIT_FILES)
        with _Env(hub) as env, tempfile.TemporaryDirectory() as legacy:
            tree, inode = self._tree(legacy)
            code, dry, _err = _run_cli_exit(["download", self.REPO, "--model-dir", legacy, "--dry-run"])
            self.assertEqual(code, 0)
            self.assertTrue(tree.is_dir())  # the dry run moves nothing
            self.assertEqual(env.written(), [])
            rc, out = _run_cli(["download", self.REPO, "--model-dir", legacy])
            snapshot = Path(out.split("snapshot_dir: ", 1)[1].splitlines()[0])
            moved = snapshot / "transformer" / "0.safetensors"
            self.assertFalse(tree.exists())
            self.assertEqual(moved.stat().st_ino, inode)  # moved, as 0.3.36 did: no second copy of the bytes
            self.assertEqual([p for p in env.written() if "blobs" in p], [])
        self.assertIn(f"the legacy directory {tree} (2 files, 66 B) would be imported", dry)
        self.assertEqual(rc, 0)
        self.assertIn(f"cache: imported {self.REPO} from the legacy directory {tree}", out)
        self.assertIn("it was not compared with the Hub, so it may lack files the package selects; nothing was fetched", out)
        self.assertEqual((hub.downloads, hub.listings), ([], 0))  # no fetch, and no Hub request at all

    def test_without_a_legacy_tree_model_dir_changes_nothing(self):
        hub = _FakeHub(H3_8BIT_FILES)
        with _Env(hub), tempfile.TemporaryDirectory() as legacy:
            rc, out = _run_cli(["download", self.REPO, "--model-dir", legacy])
        self.assertEqual(rc, 0)
        self.assertIn("cache: 0 of 18 files complete (0 B); 18 files to fetch", out)
        self.assertEqual(len(hub.fetched), 18)

    def test_a_legacy_tree_of_a_package_already_cached_is_removed_as_before(self):
        hub = _FakeHub(H3_8BIT_FILES)
        with _Env(hub) as env, tempfile.TemporaryDirectory() as legacy:
            env.cached(self.REPO, list(H3_8BIT_FILES))
            tree, _inode = self._tree(legacy)
            rc, out = _run_cli(["download", self.REPO, "--model-dir", legacy])
            self.assertFalse(tree.exists())
        self.assertEqual(rc, 0)
        self.assertIn(f"legacy: removed {tree}; the cache already holds a snapshot of {self.REPO}", out)
        self.assertIn("cache: all 18 files already present; nothing to fetch", out)
        self.assertEqual(hub.fetched, [])


@_needs_hub
class TestGenerationWithAllowDownloadOffline(unittest.TestCase):
    """MLX-Gen generation with `allow_download=True` reaches `download_hf_repo_snapshot` only for a package the
    cache does not hold: offline, a cached package runs as in 0.3.36 (no Hub request), and a missing one fails
    saying why nothing was downloaded."""

    REPO = "AbstractFramework/flux.2-klein-4b-4bit"
    FILES = {
        "README.md": 10,
        "text_encoder/config.json": 2,
        "text_encoder/0.safetensors": 5,
        "tokenizer/tokenizer.json": 2,
        "transformer/config.json": 2,
        "transformer/0.safetensors": 5,
        "vae/config.json": 2,
        "vae/0.safetensors": 5,
    }

    def _resolve(self, env):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        return MFluxVisionBackend(config=MFluxBackendConfig(model=self.REPO, allow_download=True))._resolve_model()

    def test_a_cached_package_resolves_offline_without_any_hub_request(self):
        hub = _FakeHub(self.FILES, listing_error=AssertionError("the listing was requested"))
        with _Env(hub) as env, patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            env.cached(self.REPO, list(self.FILES))
            path, base = self._resolve(env)
        self.assertEqual(Path(path), env.repo_dir(self.REPO) / "snapshots" / _FakeHub.SHA)
        self.assertEqual(base, "flux2-klein-4b")
        self.assertEqual((hub.listings, hub.downloads), (0, []))  # no Hub request of any kind

    def test_offline_with_nothing_cached_says_why_nothing_was_downloaded(self):
        import httpx
        from huggingface_hub.utils import OfflineModeIsEnabled

        from abstractvision.errors import OptionalDependencyMissingError

        for error, expected in (
            (OfflineModeIsEnabled("offline"), "offline mode is on (HF_HUB_OFFLINE), so the Hugging Face Hub was not contacted"),
            (httpx.ConnectError("[Errno 8] nodename nor servname provided"), "the Hugging Face Hub could not be reached at https://huggingface.co"),
        ):
            hub = _FakeHub(self.FILES, listing_error=error)
            with _Env(hub) as env:
                with self.assertRaises(OptionalDependencyMissingError) as ctx:
                    self._resolve(env)
                self.assertEqual(env.written(), [])
            self.assertIn(f"Nothing was downloaded for {self.REPO}: {expected}", str(ctx.exception))
            self.assertEqual(hub.downloads, [])


@_needs_hub
class TestDownloadReportsThePresetItRuns(unittest.TestCase):
    def test_a_gguf_package_names_its_own_target_and_engine(self):
        with _Env(_FakeHub(SD_TURBO_GGUF_FILES)):
            rc, out = _run_cli(["download", "Green-Sky/SD-Turbo-GGUF", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIn("target: gguf; engine: stable-diffusion.cpp; bits: 8", out)
        self.assertNotIn("target: mlx", out)
        self.assertNotIn("for mlx", out)


@_needs_hub
class TestDiffusersPlanWritesNothing(unittest.TestCase):
    """D5: the Diffusers selection reads model_index.json into memory; a refusal leaves no file."""

    def test_a_refused_diffusers_download_leaves_the_cache_empty(self):
        hub = _FakeHub(SD_TURBO_DIFFUSERS_FILES, contents={"model_index.json": SD_TURBO_INDEX})
        with _Env(hub, free_bytes=1 * GIB) as env:
            with self.assertRaisesRegex(SystemExit, "Not enough free disk space"):
                _run_cli(["download", "stabilityai/sd-turbo", "--provider", "diffusers"])
            self.assertEqual(env.written(), [])
        self.assertEqual(hub.downloads, [])

    def test_the_dry_run_and_the_download_select_the_same_files(self):
        hub = _FakeHub(SD_TURBO_DIFFUSERS_FILES, contents={"model_index.json": SD_TURBO_INDEX})
        with _Env(hub) as env:
            _rc, dry = _run_cli(["download", "stabilityai/sd-turbo", "--provider", "diffusers", "--dry-run"])
            self.assertEqual(env.written(), [])
            _rc, out = _run_cli(["download", "stabilityai/sd-turbo", "--provider", "diffusers"])
        planned = "download size: 2.4 GiB in 10 files"
        self.assertIn(planned, dry)
        self.assertIn(planned, out)
        self.assertEqual(len(hub.fetched), 10)
        self.assertNotIn("sd_turbo.safetensors", hub.fetched)
        self.assertNotIn("unet/diffusion_pytorch_model.safetensors", hub.fetched)  # the fp16 set on Apple silicon


@_needs_hub
class TestRepositoriesOutsideTheCatalog(unittest.TestCase):
    """D6: an `owner/name` the catalog does not list goes through the same plan; a 401 says what it can mean."""

    def test_an_unknown_repository_401_is_not_called_gated(self):
        hub = _FakeHub({}, listing_error=_repo_not_found(401))
        with _Env(hub) as env:
            code, out, err = _run_cli_exit(["download", "no-such-org/no-such-vision-model"])
            self.assertEqual(env.written(), [])
        self.assertEqual(code, 1)
        self.assertIn("(401)", err)
        self.assertIn("does not exist, is private, or is gated, and does not say which", err)
        self.assertNotIn("accept the model terms", err)
        self.assertNotIn("the repository is gated", err)
        self.assertEqual(hub.downloads, [])

    def test_a_repository_outside_the_catalog_is_planned_sized_and_checked(self):
        files = {"config.json": 2000, "model.safetensors": 8 * GIB}
        hub = _FakeHub(files)
        with _Env(hub, free_bytes=4 * GIB):
            code, out, err = _run_cli_exit(["download", "some-org/some-model"])
        self.assertEqual(code, 1)
        self.assertIn("catalog: not in the AbstractVision catalog", out)
        self.assertIn("access: open", out)
        self.assertIn("download size: 8.0 GiB in 2 files", out)
        self.assertIn("Not enough free disk space to download some-org/some-model", err)
        self.assertEqual(hub.downloads, [])
        hub = _FakeHub(files, hub_gated="manual")
        with _Env(hub):
            rc, out = _run_cli(["download", "some-org/some-model"])
        self.assertEqual(rc, 0)
        self.assertIn("access: gated; accept its terms at https://huggingface.co/some-org/some-model", out)
        self.assertEqual(sorted(hub.fetched), ["config.json", "model.safetensors"])

    def test_an_unknown_plain_name_points_to_the_catalog(self):
        with _Env(_FakeHub({})):
            code, out, err = _run_cli_exit(["download", "no-such-vision-model"])
        self.assertEqual(code, 1)
        self.assertIn("'no-such-vision-model' is not in the AbstractVision catalog", err)
        self.assertIn("`abstractvision catalog`", err)
        self.assertLess(len(err), 400)

    def test_a_catalog_model_id_with_several_packages_is_not_called_unpublished(self):
        with _Env(_FakeHub({})):
            code, _out, err = _run_cli_exit(["download", "baidu/ERNIE-Image-Turbo"])
        self.assertEqual(code, 1)
        self.assertNotIn("not an exact published model id", err)
        self.assertIn("names several MLX-Gen packages and the catalog recommends none of them", err)
        self.assertIn("AbstractFramework/ernie-image-turbo-4bit", err)


class TestGenerationNeverDownloads(unittest.TestCase):
    def test_the_refusal_names_the_catalog_licence_and_points_to_the_explicit_download(self):
        from abstractvision.backends.mflux import _explicit_download_note

        note = _explicit_download_note("AbstractFramework/minimax-h3-8bit", "minimax-h3")
        self.assertIn("MiniMaxAI/MiniMax-H3 weights are never downloaded during generation", note)
        self.assertIn("(MiniMax H3 Community License)", note)
        self.assertIn("`abstractvision download AbstractFramework/minimax-h3-8bit`", note)
        self.assertEqual(_explicit_download_note("AbstractFramework/flux.2-klein-4b-8bit", "flux2-klein-4b"), "")
        try:
            import mflux.models.common.config  # noqa: F401
        except Exception:
            return
        note = _explicit_download_note("AbstractFramework/minimax-h3-8bit", "minimax-h3-turbo-544p")
        self.assertIn(
            "`abstractvision download AbstractFramework/minimax-h3-8bit --base-model minimax-h3-turbo-544p`", note
        )

    def test_schedule_adapters_come_from_mlx_gen(self):
        try:
            import mflux.models.common.config  # noqa: F401
        except Exception:
            self.skipTest("mlx-gen is not installed")
        from abstractvision.backends.mflux import _mlx_gen_schedule_adapter

        self.assertIsNone(_mlx_gen_schedule_adapter("minimax-h3"))
        self.assertIsNone(_mlx_gen_schedule_adapter("flux2-klein-4b"))
        for schedule in ("minimax-h3-turbo", "minimax-h3-turbo-544p"):
            self.assertTrue(str(_mlx_gen_schedule_adapter(schedule)).startswith("hf:lightx2v/Minimax-h3-Turbo/"))
        with self.assertRaisesRegex(ValueError, "Unknown MLX-Gen base model"):
            _mlx_gen_schedule_adapter("not-a-model")


class TestListings(unittest.TestCase):
    ROWS = (
        {
            "public_task": "text-to-video",
            "measured_runs": [
                {"footprint_bytes": 40 * GIB, "outcome": "completed", "quantize": 8, "width": 1344, "height": 768, "frames": 124, "steps": 8, "low_ram": False, "stream_blocks": False},
                {"footprint_bytes": 37 * GIB, "outcome": "completed", "quantize": 8, "width": 960, "height": 544, "frames": 124, "steps": 8, "low_ram": False, "stream_blocks": False},
                {"footprint_bytes": 26 * GIB, "outcome": "completed", "quantize": 8, "width": 960, "height": 544, "frames": 124, "steps": 8, "low_ram": True, "stream_blocks": False},
                {"footprint_bytes": 11 * GIB, "outcome": "completed", "quantize": 8, "width": 960, "height": 544, "frames": 124, "steps": 8, "low_ram": True, "stream_blocks": True},
                {"footprint_bytes": 99 * GIB, "outcome": "killed", "quantize": 4, "width": 960, "height": 544, "frames": 124, "steps": 8, "low_ram": False, "stream_blocks": False},
            ],
        },
    )

    def setUp(self):
        from abstractvision import model_inventory as inv

        self.inv = inv
        for cached in (inv._mlx_gen_memory_modes, inv._mlx_gen_rows):
            cached.cache_clear()
            self.addCleanup(cached.cache_clear)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.cache = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.dict("os.environ", {"HF_HUB_CACHE": str(self.cache)}, clear=True))
        self.stack.enter_context(patch.object(inv, "framework_hf_cache_roots", return_value=[]))

    # mlx-gen's table, under its own mode names; the bf16 package is its None width.
    SMALLEST = {None: {"default": 96, "low_ram": 64, "stream_blocks": 32}, 8: {"default": 48, "low_ram": 36, "stream_blocks": 24}, 4: {"low_ram": 32}}

    def _mlx_gen(self, available=True):
        backend = SimpleNamespace(
            supported_memory_modes=lambda: ("default", "low", "lowest"),
            describe_routes=lambda model=None: list(self.ROWS),
        )
        self.backend_configs = []

        def build(repo_id, base_model=None):
            self.backend_configs.append((repo_id, base_model))
            return backend if available else None

        self.stack.enter_context(patch.object(self.inv, "_mlx_gen_backend", side_effect=build))
        engine = SimpleNamespace(MINIMAX_H3_SMALLEST_TESTED_MACHINE=self.SMALLEST)
        import_module = importlib.import_module

        def engine_module(name, *args, **kwargs):
            return engine if name == "mflux.task_inference" else import_module(name, *args, **kwargs)

        self.stack.enter_context(patch.object(self.inv.importlib, "import_module", side_effect=engine_module))

    def _install(self, repo_id, size):
        repo_dir = self.cache / f"models--{repo_id.replace('/', '--')}"
        snapshot = repo_dir / "snapshots" / "abc"
        (snapshot / "transformer").mkdir(parents=True)
        with open(snapshot / "transformer" / "0.safetensors", "wb") as fh:
            fh.truncate(size)
        (repo_dir / "refs").mkdir()
        (repo_dir / "refs" / "main").write_text("abc")

    def test_show_model_json_reports_cache_state_modes_and_measured_peaks(self):
        self._mlx_gen()
        self._install("AbstractFramework/minimax-h3-8bit", 3 * GIB)
        rc, out = _run_cli(["show-model", "MiniMaxAI/MiniMax-H3", "--json"])
        self.assertEqual(rc, 0)
        model = json.loads(out)
        downloads = {d["repo_id"]: d for d in model["downloads"]}
        package = downloads["AbstractFramework/minimax-h3-8bit"]
        self.assertEqual(package["cache"], "cached")
        self.assertEqual(package["cache_bytes"], 3 * GIB)
        self.assertTrue(package["gated"])
        self.assertTrue(package["recommended"])
        self.assertEqual(package["memory_modes"], ["default", "low", "lowest"])
        # The 8-bit package's completed runs, labelled by mode: never another width's.
        self.assertEqual(
            [(p["task"], p["memory_mode"], p["width"], p["height"], p["peak_bytes"]) for p in package["memory_peaks"]],
            [
                ("text_to_video", "default", 1344, 768, 40 * GIB),
                ("text_to_video", "default", 960, 544, 37 * GIB),
                ("text_to_video", "low", 960, 544, 26 * GIB),
                ("text_to_video", "lowest", 960, 544, 11 * GIB),
            ],
        )
        self.assertEqual(package["memory_peaks"], self.inv.memory_peaks("AbstractFramework/minimax-h3-8bit"))
        self.assertEqual(package["smallest_tested_memory_gb"], {"default": 48, "low": 36, "lowest": 24})
        self.assertEqual(downloads["AbstractFramework/minimax-h3-4bit"]["cache"], "absent")
        self.assertNotIn("memory_peaks", downloads["AbstractFramework/minimax-h3-4bit"])
        self.assertNotIn("memory_peaks", downloads["MiniMaxAI/MiniMax-H3"])

    def test_show_model_text_without_mlx_gen_says_not_available(self):
        self._mlx_gen(available=False)
        rc, out = _run_cli(["show-model", "MiniMaxAI/MiniMax-H3"])
        self.assertEqual(rc, 0)
        self.assertIn("memory modes on this host: not available on this host", out)
        self.assertIn("access: gated (accept the MiniMax H3 Community License", out)
        self.assertIn("access: open", out)
        self.assertNotIn("measured by mlx-gen", out)

    def test_show_model_text_with_mlx_gen_shows_peaks_and_the_smallest_tested_memory(self):
        self._mlx_gen()
        rc, out = _run_cli(["show-model", "MiniMaxAI/MiniMax-H3"])
        self.assertIn("memory (low, measured by mlx-gen): peak 26.0 GiB at 960x544, 124 frames, 8 steps (text_to_video)", out)
        self.assertIn("smallest memory mlx-gen tested: default 48 GB, low 36 GB, lowest 24 GB", out)

    def test_memory_peaks_are_one_package_width_and_schedule(self):
        self._mlx_gen()
        peaks = self.inv.memory_peaks("AbstractFramework/minimax-h3-8bit", base_model="h3-turbo")
        self.assertEqual(self.backend_configs, [("AbstractFramework/minimax-h3-8bit", "minimax-h3-turbo")])
        self.assertEqual(
            peaks[2],
            {
                "task": "text_to_video",
                "memory_mode": "low",
                "width": 960,
                "height": 544,
                "frames": 124,
                "steps": 8,
                "peak_bytes": 26 * GIB,
                "machine_total_ram_bytes": None,
            },
        )
        json.dumps(peaks)
        self.assertEqual(self.inv.memory_peaks("AbstractFramework/minimax-h3-4bit"), [])  # its only run was killed
        self.assertEqual(self.inv.memory_peaks("MiniMaxAI/MiniMax-H3"), [])  # a source snapshot, not a package
        self.assertEqual(self.inv.memory_peaks("someone/not-in-the-catalog"), [])
        with self.assertRaisesRegex(ValueError, "not a schedule of AbstractFramework/minimax-h3-8bit"):
            self.inv.memory_peaks("AbstractFramework/minimax-h3-8bit", base_model="flux2-klein-4b")

    def test_memory_peaks_are_empty_without_the_engine(self):
        self._mlx_gen(available=False)
        self.assertEqual(self.inv.memory_peaks("AbstractFramework/minimax-h3-8bit"), [])

    def test_the_smallest_tested_memory_is_read_for_the_package_width(self):
        self._mlx_gen()
        smallest = self.inv._mlx_gen_smallest_tested_memory_gb
        self.assertEqual(smallest("AbstractFramework/minimax-h3-bf16"), {"default": 96, "low": 64, "lowest": 32})
        self.assertEqual(smallest("AbstractFramework/minimax-h3-4bit"), {"low": 32})
        self.assertEqual(smallest("AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"), {})  # not MiniMax-H3

    def test_the_plugin_relays_memory_peaks(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        self._mlx_gen()
        cap = _AbstractVisionCapability(SimpleNamespace(config={}))
        self.assertEqual(
            cap.memory_peaks("AbstractFramework/minimax-h3-8bit", base_model="minimax-h3-turbo"),
            self.inv.memory_peaks("AbstractFramework/minimax-h3-8bit", base_model="minimax-h3-turbo"),
        )
        self.assertEqual(len(cap.memory_peaks("AbstractFramework/minimax-h3-8bit")), 4)

    def test_catalog_marks_installed_packages_and_lists_memory_modes(self):
        self._mlx_gen()
        self._install("AbstractFramework/minimax-h3-4bit", 2 * GIB)
        with patch("abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"):
            rc, out = _run_cli(["catalog", "--provider", "mlx-gen", "--task", "text_to_video", "--json"])
        self.assertEqual(rc, 0)
        rows = {d["repo_id"]: d for m in json.loads(out) for d in m["downloads"]}
        self.assertEqual(rows["AbstractFramework/minimax-h3-4bit"]["cache"], "cached")
        self.assertEqual(rows["AbstractFramework/minimax-h3-4bit"]["cache_bytes"], 2 * GIB)
        self.assertEqual(rows["AbstractFramework/minimax-h3-8bit"]["cache"], "absent")
        self.assertTrue(rows["AbstractFramework/minimax-h3-8bit"]["recommended"])
        self.assertEqual(rows["AbstractFramework/minimax-h3-8bit"]["memory_modes"], ["default", "low", "lowest"])

    @_needs_hub
    def test_the_local_look_names_what_it_can_see_is_missing(self):
        from abstractvision.model_inventory import local_cache_state

        hub = _FakeHub(SD_TURBO_DIFFUSERS_FILES, contents={"model_index.json": SD_TURBO_INDEX})
        sd_turbo = md.find_model_preset("stabilityai/sd-turbo", target="diffusers", engine="diffusers", require_8bit=False)
        gguf = md.find_model_preset("Green-Sky/SD-Turbo-GGUF", target="gguf", engine="stable-diffusion.cpp")
        with _Env(hub) as env:
            # What the 0.3.37 release candidate left after a refused Diffusers download: model_index.json only.
            env.cached("stabilityai/sd-turbo", ["model_index.json"])
            stray = local_cache_state(sd_turbo, extra_roots=())
            # An interrupted GGUF download: the README complete, the model a partial blob.
            env.hub.files = dict(SD_TURBO_GGUF_FILES)
            env.cached("Green-Sky/SD-Turbo-GGUF", ["README.md"])
            index = sorted(SD_TURBO_GGUF_FILES).index("sd_turbo-f16-q8_0.gguf")
            partial = env.repo_dir("Green-Sky/SD-Turbo-GGUF") / "blobs" / (_etag(index, "x.gguf") + ".incomplete")
            partial.write_bytes(b"\0" * 9)
            interrupted = local_cache_state(gguf, extra_roots=())
            partial.unlink()
            env.cached("Green-Sky/SD-Turbo-GGUF", ["sd_turbo-f16-q8_0.gguf"])
            complete = local_cache_state(gguf, extra_roots=())
        self.assertEqual(stray.state, "incomplete")
        self.assertEqual(stray.gaps, ("no weight file",))
        self.assertEqual(interrupted.state, "incomplete")
        self.assertIn("partial files of an interrupted download (.incomplete)", interrupted.gaps)
        self.assertEqual(complete.state, "cached")
        self.assertEqual(complete.files, 2)

    @_needs_hub
    def test_a_component_the_patterns_do_not_name_is_not_a_gap(self):
        from abstractvision.model_inventory import local_cache_state

        index = json.dumps({"_class_name": "HiDreamImagePipeline", "text_encoder_4": ["transformers", "LlamaForCausalLM"],
                            "transformer": ["diffusers", "HiDreamImageTransformer2DModel"]})
        files = {"model_index.json": 900, "transformer/config.json": 500, "transformer/model.safetensors": 10}
        preset = md.find_model_preset("HiDream-ai/HiDream-I1-Full", target="diffusers", engine="diffusers", require_8bit=False)
        with _Env(_FakeHub(files, contents={"model_index.json": index})) as env:
            env.cached("HiDream-ai/HiDream-I1-Full", list(files))
            state = local_cache_state(preset, extra_roots=())
        self.assertEqual((state.state, state.gaps), ("cached", ()))

    def test_show_model_text_says_cached_or_incomplete_never_installed(self):
        self._mlx_gen(available=False)
        self._install("AbstractFramework/minimax-h3-8bit", 3 * GIB)
        partial = self.cache / "models--AbstractFramework--minimax-h3-4bit" / "blobs" / "abc.incomplete"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b"\0")
        rc, out = _run_cli(["show-model", "MiniMaxAI/MiniMax-H3"])
        self.assertEqual(rc, 0)
        self.assertIn("cache: cached (3.0 GiB)", out)
        self.assertIn("cache: incomplete (files of an unfinished download but no snapshot", out)
        self.assertIn("cache: not cached", out)
        self.assertNotIn("installed", out)

    def test_models_json_lists_each_model_with_its_downloads(self):
        self._mlx_gen(available=False)
        rc, out = _run_cli(["models", "--json"])
        self.assertEqual(rc, 0)
        models = {m["model_id"]: m for m in json.loads(out)}
        h3 = models["MiniMaxAI/MiniMax-H3"]
        self.assertEqual(h3["license_name"], "MiniMax H3 Community License")
        self.assertEqual(h3["tasks"], ["image_to_video", "text_to_video"])
        self.assertIsNone(h3["downloads"][0]["memory_modes"])
        rc, plain = _run_cli(["models"])
        self.assertIn("MiniMaxAI/MiniMax-H3\n", plain)


class TestAccessError(unittest.TestCase):
    def test_message_names_the_licence_and_keeps_one_upstream_line(self):
        error = md.HuggingFaceAccessError(
            "AbstractFramework/minimax-h3-8bit",
            raw_message="403 Client Error.\n\nCannot access gated repo.\nAccess restricted.",
            status_code=403,
            license_name="MiniMax H3 Community License",
            gated=True,
        )
        text = str(error)
        self.assertIn("accept the MiniMax H3 Community License", text)
        self.assertIn("https://huggingface.co/AbstractFramework/minimax-h3-8bit", text)
        self.assertIn("Upstream error: 403 Client Error.", text)
        self.assertNotIn("Access restricted", text)

    def test_a_401_for_a_repository_not_marked_gated_says_what_it_can_mean(self):
        text = str(md.HuggingFaceAccessError("no-such-org/no-such-vision-model", status_code=401))
        self.assertIn("does not exist, is private, or is gated, and does not say which", text)
        self.assertIn("Check the name at https://huggingface.co/no-such-org/no-such-vision-model", text)
        self.assertNotIn("the repository is gated,", text)


if __name__ == "__main__":
    unittest.main()
