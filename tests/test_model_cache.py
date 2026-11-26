import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestModelCache(unittest.TestCase):
    def test_default_hf_cache_root_uses_hf_home(self):
        from abstractvision.model_cache import default_hf_cache_root

        with patch.dict("os.environ", {"HF_HOME": "/tmp/hf-home"}, clear=True):
            self.assertEqual(default_hf_cache_root(), Path("/tmp/hf-home/hub"))

    def test_migrates_legacy_tree_into_hf_snapshot(self):
        from abstractvision.model_cache import ensure_hf_repo_snapshot, hf_repo_dir, resolve_hf_repo_snapshot

        repo_id = "AbstractFramework/flux.2-klein-4b-8bit"

        with tempfile.TemporaryDirectory() as legacy_td, tempfile.TemporaryDirectory() as cache_td:
            legacy_root = Path(legacy_td)
            source_dir = legacy_root / "flux2-klein-4b-mlx-8bit"
            (source_dir / "transformer").mkdir(parents=True)
            (source_dir / "transformer" / "0.safetensors").write_bytes(b"x")

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                snapshot = ensure_hf_repo_snapshot(
                    repo_id,
                    source_dir=source_dir,
                    cache_dir=cache_td,
                    cleanup_source=True,
                )

                self.assertIsNotNone(snapshot)
                self.assertTrue(snapshot.is_dir())
                self.assertFalse(source_dir.exists())
                self.assertTrue((snapshot / "transformer" / "0.safetensors").is_file())

                repo_dir = hf_repo_dir(repo_id, cache_dir=cache_td)
                self.assertEqual((repo_dir / "refs" / "main").read_text(encoding="utf-8").strip(), snapshot.name)
                self.assertEqual(resolve_hf_repo_snapshot(repo_id, cache_dir=cache_td), snapshot)

    def test_rejects_incomplete_snapshot_without_weights(self):
        from abstractvision.model_cache import cached_hf_model_sources, hf_snapshot_is_usable, incomplete_hf_model_sources, resolve_hf_repo_snapshot

        repo_id = "baidu/ERNIE-Image-Turbo"

        with tempfile.TemporaryDirectory() as cache_td:
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                snap = (
                    Path(cache_td)
                    / "models--baidu--ERNIE-Image-Turbo"
                    / "snapshots"
                    / "abc123"
                )
                (snap / ".cache" / "huggingface" / "download" / "transformer").mkdir(parents=True, exist_ok=True)
                (snap / "model_index.json").write_text("{}", encoding="utf-8")
                (snap / "transformer" / "config.json").parent.mkdir(parents=True, exist_ok=True)
                (snap / "transformer" / "config.json").write_text("{}", encoding="utf-8")
                (
                    snap
                    / ".cache"
                    / "huggingface"
                    / "download"
                    / "transformer"
                    / "weights.123.incomplete"
                ).write_bytes(b"x")
                refs = snap.parents[1] / "refs"
                refs.mkdir(parents=True, exist_ok=True)
                (refs / "main").write_text("abc123", encoding="utf-8")

                self.assertFalse(
                    hf_snapshot_is_usable(snap, required_files=("model_index.json",), require_weight_files=True)
                )
                self.assertEqual(
                    cached_hf_model_sources(
                        repo_id,
                        cache_dir=cache_td,
                        required_files=("model_index.json",),
                        require_weight_files=True,
                    ),
                    [],
                )
                self.assertEqual(
                    incomplete_hf_model_sources(
                        repo_id,
                        cache_dir=cache_td,
                        required_files=("model_index.json",),
                        require_weight_files=True,
                    ),
                    ["configured cache"],
                )
                self.assertIsNone(
                    resolve_hf_repo_snapshot(
                        repo_id,
                        cache_dir=cache_td,
                        required_files=("model_index.json",),
                        require_weight_files=True,
                    )
                )

    def test_rejects_sharded_snapshot_with_missing_indexed_weights(self):
        from abstractvision.model_cache import (
            cached_hf_model_sources,
            hf_snapshot_is_usable,
            hf_snapshot_missing_indexed_weight_files,
            incomplete_hf_model_sources,
            resolve_hf_repo_snapshot,
        )

        repo_id = "black-forest-labs/FLUX.2-klein-9B"

        with tempfile.TemporaryDirectory() as cache_td:
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                snap = (
                    Path(cache_td)
                    / "models--black-forest-labs--FLUX.2-klein-9B"
                    / "snapshots"
                    / "abc123"
                )
                snap.mkdir(parents=True, exist_ok=True)
                (snap / "model_index.json").write_text("{}", encoding="utf-8")
                (snap / "transformer").mkdir(parents=True, exist_ok=True)
                (snap / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
                    '{"weight_map":{"layer.0":"diffusion_pytorch_model-00001-of-00002.safetensors","layer.1":"diffusion_pytorch_model-00002-of-00002.safetensors"}}',
                    encoding="utf-8",
                )
                (snap / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(b"x")
                refs = snap.parents[1] / "refs"
                refs.mkdir(parents=True, exist_ok=True)
                (refs / "main").write_text("abc123", encoding="utf-8")

                self.assertEqual(
                    hf_snapshot_missing_indexed_weight_files(snap),
                    ["transformer/diffusion_pytorch_model-00002-of-00002.safetensors"],
                )
                self.assertFalse(
                    hf_snapshot_is_usable(snap, required_files=("model_index.json",), require_weight_files=True)
                )
                self.assertEqual(
                    cached_hf_model_sources(
                        repo_id,
                        cache_dir=cache_td,
                        required_files=("model_index.json",),
                        require_weight_files=True,
                    ),
                    [],
                )
                self.assertEqual(
                    incomplete_hf_model_sources(
                        repo_id,
                        cache_dir=cache_td,
                        required_files=("model_index.json",),
                        require_weight_files=True,
                    ),
                    ["configured cache"],
                )
                self.assertIsNone(
                    resolve_hf_repo_snapshot(
                        repo_id,
                        cache_dir=cache_td,
                        required_files=("model_index.json",),
                        require_weight_files=True,
                    )
                )

    def test_rejects_snapshot_with_incomplete_repo_blobs(self):
        from abstractvision.model_cache import cached_hf_model_sources, hf_snapshot_is_usable, incomplete_hf_model_sources, resolve_hf_repo_snapshot

        repo_id = "Qwen/Qwen-Image-Edit-2511"

        with tempfile.TemporaryDirectory() as cache_td:
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                repo_dir = Path(cache_td) / "models--Qwen--Qwen-Image-Edit-2511"
                snap = repo_dir / "snapshots" / "abc123"
                (snap / "model_index.json").parent.mkdir(parents=True, exist_ok=True)
                (snap / "model_index.json").write_text("{}", encoding="utf-8")
                (snap / "vae").mkdir(parents=True, exist_ok=True)
                (snap / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
                (repo_dir / "blobs").mkdir(parents=True, exist_ok=True)
                (repo_dir / "blobs" / "partial-weight.incomplete").write_bytes(b"x")
                refs = repo_dir / "refs"
                refs.mkdir(parents=True, exist_ok=True)
                (refs / "main").write_text("abc123", encoding="utf-8")

                self.assertFalse(
                    hf_snapshot_is_usable(snap, required_files=("model_index.json",), require_weight_files=True)
                )
                self.assertEqual(
                    cached_hf_model_sources(
                        repo_id,
                        cache_dir=cache_td,
                        required_files=("model_index.json",),
                        require_weight_files=True,
                    ),
                    [],
                )
                self.assertEqual(
                    incomplete_hf_model_sources(
                        repo_id,
                        cache_dir=cache_td,
                        required_files=("model_index.json",),
                        require_weight_files=True,
                    ),
                    ["configured cache"],
                )
                self.assertIsNone(
                    resolve_hf_repo_snapshot(
                        repo_id,
                        cache_dir=cache_td,
                        required_files=("model_index.json",),
                        require_weight_files=True,
                    )
                )

    def test_marks_lock_only_repo_as_incomplete(self):
        from abstractvision.model_cache import incomplete_hf_model_sources

        repo_id = "example-org/Test-Partial-Model"

        with tempfile.TemporaryDirectory() as cache_td:
            lock_dir = Path(cache_td) / ".locks" / "models--example-org--Test-Partial-Model"
            lock_dir.mkdir(parents=True, exist_ok=True)

            self.assertEqual(
                incomplete_hf_model_sources(
                    repo_id,
                    cache_dir=cache_td,
                    required_files=("model_index.json",),
                    require_weight_files=True,
                ),
                ["configured cache"],
            )


if __name__ == "__main__":
    unittest.main()


class TestCacheRefs(unittest.TestCase):
    def test_a_nested_ref_is_written_and_a_name_leaving_refs_is_refused(self):
        from abstractvision.model_cache import write_hf_ref

        with tempfile.TemporaryDirectory() as td:
            repo_dir = Path(td) / "models--org--repo"
            write_hf_ref(repo_dir, "refs/pr/1", "c0ffee")  # a pull-request revision, as huggingface_hub writes it
            write_hf_ref(repo_dir, "main", "c0ffee")
            self.assertEqual((repo_dir / "refs" / "refs" / "pr" / "1").read_text(), "c0ffee")
            for name in ("../../escape", "/etc/x", "a/../../b", "a//b", ".", "..\\escape"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    write_hf_ref(repo_dir, name, "c0ffee")
            self.assertEqual(sorted(str(p.relative_to(td)) for p in Path(td).rglob("*") if p.is_file()),
                             ["models--org--repo/refs/main", "models--org--repo/refs/refs/pr/1"])


class TestHubClientCacheIsRead(unittest.TestCase):
    """A model cached where huggingface_hub looks (its own constant, in a fresh process) is found, whichever
    variable locates that cache; the root downloads write to stays."""

    _CHILD = (
        "import json\n"
        "from abstractvision.backends.huggingface_diffusers import (\n"
        "    HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend)\n"
        "from huggingface_hub import constants\n"
        "from abstractvision.model_cache import default_hf_cache_root\n"
        "b = HuggingFaceDiffusersVisionBackend(config=HuggingFaceDiffusersBackendConfig(model_id='some/model', device='cpu'))\n"
        "print(json.dumps({'found': str(b._resolve_any_snapshot_dir()), 'write_root': str(default_hf_cache_root()),\n"
        "                  'hub_cache': constants.HF_HUB_CACHE}))\n"
    )

    def setUp(self):
        import importlib.util

        if importlib.util.find_spec("huggingface_hub") is None:
            self.skipTest("huggingface_hub is not installed")

    def test_hf_hub_cache(self):
        self._check("HF_HUB_CACHE", lambda root: root / "hub-cache", lambda root, home: root / "hub-cache")

    def test_huggingface_hub_cache(self):
        self._check("HUGGINGFACE_HUB_CACHE", lambda root: root / "legacy-hub", self._home_cache)

    def test_hf_home(self):
        self._check("HF_HOME", lambda root: root / "hf-home", lambda root, home: root / "hf-home" / "hub")

    def test_xdg_cache_home(self):
        self._check("XDG_CACHE_HOME", lambda root: root / "xdg", self._home_cache)

    def _check(self, variable, value, expected_write_root):
        import json
        import os
        import subprocess
        import sys

        import abstractvision

        with tempfile.TemporaryDirectory() as td:
            root, home = Path(td), Path(td) / "home"
            home.mkdir()
            hub = {
                "HF_HUB_CACHE": root / "hub-cache",
                "HUGGINGFACE_HUB_CACHE": root / "legacy-hub",
                "HF_HOME": root / "hf-home" / "hub",
                "XDG_CACHE_HOME": root / "xdg" / "huggingface" / "hub",
            }[variable]
            snapshot = hub / "models--some--model" / "snapshots" / ("0" * 40)
            (snapshot / "unet").mkdir(parents=True)
            (snapshot / "model_index.json").write_text('{"_class_name": "StableDiffusionPipeline"}')
            (snapshot / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 16)
            (hub / "models--some--model" / "refs").mkdir()
            (hub / "models--some--model" / "refs" / "main").write_text(snapshot.name)
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(home),
                "HF_HUB_OFFLINE": "1",
                "PYTHONPATH": str(Path(abstractvision.__file__).resolve().parents[1]),
                variable: str(value(root)),
            }
            proc = subprocess.run(
                [sys.executable, "-c", self._CHILD], env=env, cwd=td, capture_output=True, text=True, check=False
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            seen = json.loads(proc.stdout.strip().splitlines()[-1])
            self.assertEqual(Path(seen["hub_cache"]).resolve(), hub.resolve())
            self.assertEqual(Path(seen["found"]).resolve(), snapshot.resolve())
            self.assertEqual(Path(seen["write_root"]).resolve(), expected_write_root(root, home).resolve())

    @staticmethod
    def _home_cache(root, home):
        return home / ".cache" / "huggingface" / "hub"
