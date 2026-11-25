"""`abstractvision download` and the model listings: package selection, the files a package needs,
gated access, disk space, resumed downloads, schedule adapters, cache state and memory modes.

No network and no weights: the Hugging Face client is a fake that serves the repositories' real file
names (sizes included) and refuses gated downloads the way the Hub does.
"""

from __future__ import annotations

import contextlib
import io
import json
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
    """Stands in for huggingface_hub: the listing is public, gated files are refused (403)."""

    def __init__(self, files, *, gated=False, interrupt=False, drop=()):
        self.files = dict(files)
        self.gated = gated
        self.interrupt = interrupt
        self.drop = set(drop)
        self.downloads = []

    def api(self, *args, **kwargs):
        hub = self

        class _Api:
            def model_info(self, repo_id, revision=None, files_metadata=False, token=None):
                return SimpleNamespace(siblings=_siblings(hub.files))

        return _Api()

    def snapshot_download(self, *, repo_id, cache_dir, allow_patterns=None, ignore_patterns=None, **kwargs):
        from huggingface_hub.utils import filter_repo_objects

        self.downloads.append({"repo_id": repo_id, "allow_patterns": allow_patterns, "token": kwargs.get("token")})
        if self.gated:
            error = RuntimeError(
                f"403 Client Error. Cannot access gated repo for url https://huggingface.co/{repo_id}.\n"
                "Access to model is restricted and you are not in the authorized list."
            )
            error.response = SimpleNamespace(status_code=403)
            raise error
        if self.interrupt:
            raise KeyboardInterrupt
        repo_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
        snapshot = repo_dir / "snapshots" / "abc"
        for partial in (repo_dir / "blobs").glob("*.incomplete"):
            partial.unlink()  # the download completes the partial blobs
        for name in filter_repo_objects(list(self.files), allow_patterns=allow_patterns, ignore_patterns=ignore_patterns):
            if name in self.drop:
                continue
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}" if name.endswith(".json") else "x", encoding="utf-8")
        return str(snapshot)


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
        with self.assertRaisesRegex(ValueError, "not an exact published model id"):
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


class TestDownloadCommand(unittest.TestCase):
    def test_gated_refusal_names_the_package_page_and_the_licence(self):
        hub = _FakeHub(H3_8BIT_FILES, gated=True)
        with _Env(hub):
            with self.assertRaises(SystemExit) as ctx:
                _run_cli(["download", "MiniMaxAI/MiniMax-H3"])
        message = str(ctx.exception)
        self.assertIn("https://huggingface.co/AbstractFramework/minimax-h3-8bit", message)
        self.assertIn("MiniMax H3 Community License", message)
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
        self.assertIn("it is 146.4 GiB", message)
        self.assertIn("50.0 GiB is free", message)
        self.assertEqual(hub.downloads, [])

    def test_a_partial_download_is_resumed_and_counted(self):
        hub = _FakeHub(H3_8BIT_FILES)
        with _Env(hub) as env:
            blobs = env.cache / "models--AbstractFramework--minimax-h3-8bit" / "blobs"
            blobs.mkdir(parents=True)
            names = sorted(H3_8BIT_FILES)
            sha = {name: f"sha{i:04d}" for i, name in enumerate(names)}
            (blobs / sha["vae/0.safetensors"]).write_bytes(b"\0" * 1024)
            with open(blobs / sha["vae/0.safetensors"], "r+b") as fh:
                fh.truncate(H3_8BIT_FILES["vae/0.safetensors"])
            with open(blobs / (sha["transformer/0.safetensors"] + ".incomplete"), "wb") as fh:
                fh.truncate(GIB)
            rc, out = _run_cli(["download", "AbstractFramework/minimax-h3-8bit"])
        self.assertEqual(rc, 0)
        self.assertIn("cache: resuming, 5.8 GiB already present", out)
        self.assertIn("snapshot_dir:", out)

    def test_a_download_missing_files_is_reported_with_how_to_resume(self):
        hub = _FakeHub(H3_8BIT_FILES, drop={"audio_vae/0.safetensors"})
        with _Env(hub):
            with self.assertRaises(SystemExit) as ctx:
                _run_cli(["download", "AbstractFramework/minimax-h3-8bit"])
        self.assertIn("is incomplete: 1 of", str(ctx.exception))
        self.assertIn("audio_vae/0.safetensors", str(ctx.exception))
        self.assertIn("Run the same command again to resume", str(ctx.exception))

    def test_an_interrupted_download_says_how_to_resume(self):
        hub = _FakeHub(H3_8BIT_FILES, interrupt=True)
        with _Env(hub):
            with self.assertRaises(SystemExit) as ctx:
                _run_cli(["download", "AbstractFramework/minimax-h3-8bit"])
        self.assertIn("interrupted", str(ctx.exception))
        self.assertIn("run the same command again to resume", str(ctx.exception))

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
        with _Env(hub), patch("abstractvision.cli.download_hf_repo_snapshot", side_effect=AssertionError("snapshot")):
            with self.assertRaisesRegex(SystemExit, "No 2-bit package"):
                _run_cli(["download", "MiniMaxAI/MiniMax-H3", "--bits", "2"])
            with self.assertRaisesRegex(SystemExit, "not an exact published model id"):
                _run_cli(["download", "black-forest-labs/FLUX.2-klein-4B"])
        self.assertEqual(hub.downloads, [])

    def test_dry_run_reports_the_plan_and_downloads_nothing(self):
        hub = _FakeHub(H3_8BIT_FILES)
        with _Env(hub, free_bytes=20 * GIB):
            rc, out = _run_cli(["download", "MiniMaxAI/MiniMax-H3", "--dry-run", "--base-model", "minimax-h3-turbo"])
        self.assertEqual(rc, 0)
        self.assertEqual(hub.downloads, [])
        self.assertIn("selected: AbstractFramework/minimax-h3-8bit", out)
        self.assertIn("Not enough free disk space", out)
        self.assertIn("adapter (minimax-h3-turbo): hf:lightx2v/Minimax-h3-Turbo/turbo.safetensors (would be fetched)", out)
        self.assertIn("dry run: nothing was downloaded", out)

    def test_an_unknown_size_says_the_disk_was_not_checked(self):
        files = dict(H3_8BIT_FILES, **{"audio_vae/0.safetensors": None})  # the listing gives no size
        for argv in (["--dry-run"], []):
            with _Env(_FakeHub(files)):
                rc, out = _run_cli(["download", "MiniMaxAI/MiniMax-H3", *argv])
            self.assertEqual(rc, 0, argv)
            self.assertIn("disk: not checked, because the download size is unknown", out, argv)
            self.assertNotIn("enough free space", out, argv)
        with _Env(_FakeHub(H3_8BIT_FILES)):
            _rc, out = _run_cli(["download", "MiniMaxAI/MiniMax-H3", "--dry-run"])
        self.assertIn("disk: enough free space", out)

    def test_repo_snapshot_accepts_a_string_cache_dir(self):
        hub = _FakeHub({"model.safetensors": 10, "config.json": 2})
        with _Env(hub) as env:
            path = md.download_hf_repo_snapshot("some-org/some-model", cache_dir=str(env.cache))
            self.assertTrue((path / "model.safetensors").is_file())


class TestGenerationNeverDownloads(unittest.TestCase):
    def test_the_refusal_points_to_the_explicit_download(self):
        from abstractvision.backends.mflux import _explicit_download_note

        note = _explicit_download_note("minimax-h3-turbo-544p")
        self.assertIn("never downloaded during generation", note)
        self.assertIn(
            "abstractvision download AbstractFramework/minimax-h3-8bit --base-model minimax-h3-turbo-544p", note
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

    def _mlx_gen(self, available=True):
        backend = SimpleNamespace(
            supported_memory_modes=lambda: ("default", "low", "lowest"),
            describe_routes=lambda model=None: list(self.ROWS),
        )
        self.stack.enter_context(patch.object(self.inv, "_mlx_gen_backend", return_value=backend if available else None))
        self.stack.enter_context(
            patch.object(
                self.inv,
                "_mlx_gen_smallest_tested_machines",
                return_value={8: {"default": 48, "low_ram": 36, "stream_blocks": 24}},
            )
        )

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
        self.assertTrue(package["installed"])
        self.assertEqual(package["installed_bytes"], 3 * GIB)
        self.assertTrue(package["gated"])
        self.assertTrue(package["recommended"])
        self.assertEqual(package["memory_modes"], ["default", "low", "lowest"])
        measured = package["memory_measurements"]
        self.assertEqual(measured["default"]["peak_bytes"], 37 * GIB)
        self.assertEqual((measured["default"]["width"], measured["default"]["height"]), (960, 544))
        self.assertEqual(measured["low"]["smallest_tested_machine_gb"], 36)
        self.assertEqual(measured["lowest"]["peak_bytes"], 11 * GIB)
        self.assertFalse(downloads["AbstractFramework/minimax-h3-4bit"]["installed"])
        self.assertNotIn("memory_measurements", downloads["AbstractFramework/minimax-h3-4bit"])
        self.assertNotIn("memory_measurements", downloads["MiniMaxAI/MiniMax-H3"])

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
        self.assertIn(
            "memory (low, measured by mlx-gen): peak 26.0 GiB at 960x544, 124 frames, 8 steps; smallest tested memory: 36 GB",
            out,
        )

    def test_catalog_marks_installed_packages_and_lists_memory_modes(self):
        self._mlx_gen()
        self._install("AbstractFramework/minimax-h3-4bit", 2 * GIB)
        with patch("abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"):
            rc, out = _run_cli(["catalog", "--provider", "mlx-gen", "--task", "text_to_video", "--json"])
        self.assertEqual(rc, 0)
        rows = {d["repo_id"]: d for m in json.loads(out) for d in m["downloads"]}
        self.assertTrue(rows["AbstractFramework/minimax-h3-4bit"]["installed"])
        self.assertEqual(rows["AbstractFramework/minimax-h3-4bit"]["installed_bytes"], 2 * GIB)
        self.assertFalse(rows["AbstractFramework/minimax-h3-8bit"]["installed"])
        self.assertTrue(rows["AbstractFramework/minimax-h3-8bit"]["recommended"])
        self.assertEqual(rows["AbstractFramework/minimax-h3-8bit"]["memory_modes"], ["default", "low", "lowest"])

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
        )
        text = str(error)
        self.assertIn("accept the MiniMax H3 Community License", text)
        self.assertIn("https://huggingface.co/AbstractFramework/minimax-h3-8bit", text)
        self.assertIn("Upstream error: 403 Client Error.", text)
        self.assertNotIn("Access restricted", text)


if __name__ == "__main__":
    unittest.main()
