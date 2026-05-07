import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestPlaygroundServer(unittest.TestCase):
    def test_normalizes_raw_and_provider_model_ids(self):
        from abstractvision.playground_server import normalize_model_id_for_backend

        self.assertEqual(
            normalize_model_id_for_backend("runwayml/stable-diffusion-v1-5"),
            ("diffusers", "runwayml/stable-diffusion-v1-5"),
        )
        self.assertEqual(
            normalize_model_id_for_backend("diffusers/runwayml/stable-diffusion-v1-5"),
            ("diffusers", "runwayml/stable-diffusion-v1-5"),
        )
        self.assertEqual(normalize_model_id_for_backend("diffusers/default")[0], "diffusers")
        self.assertEqual(normalize_model_id_for_backend("sdcpp/default"), ("sdcpp", None))
        self.assertEqual(
            normalize_model_id_for_backend("openai-compatible/dall-e-3"), ("openai", "dall-e-3")
        )

    def test_loads_raw_huggingface_model_id_without_provider_prefix(self):
        import abstractvision.backends.huggingface_diffusers as hf_backend
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        seen = {}

        class FakeBackend:
            def __init__(self, *, config):
                seen["config"] = config

        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))

        with patch.object(hf_backend, "HuggingFaceDiffusersVisionBackend", FakeBackend):
            out = state.load_model("runwayml/stable-diffusion-v1-5")

        self.assertTrue(out["ok"])
        self.assertEqual(seen["config"].model_id, "runwayml/stable-diffusion-v1-5")
        self.assertFalse(seen["config"].allow_download)
        self.assertEqual(out["active"]["model_id"], "runwayml/stable-diffusion-v1-5")

    def test_lists_cached_huggingface_registry_models(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            snap = cache / "models--runwayml--stable-diffusion-v1-5" / "snapshots" / "abc123"
            snap.mkdir(parents=True)

            state = PlaygroundState(
                PlaygroundServerConfig(
                    diffusers_cache_dir=str(cache),
                    diffusers_allow_download=False,
                    default_model_id="",
                )
            )
            out = state.list_models()

        models = {m["id"]: m for m in out["models"]}
        self.assertIn("runwayml/stable-diffusion-v1-5", models)
        self.assertTrue(models["runwayml/stable-diffusion-v1-5"]["cached"])
        self.assertIn("configured cache", models["runwayml/stable-diffusion-v1-5"]["cached_in"])

    def test_default_config_does_not_select_backend_without_backend_env(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with patch.dict("os.environ", {}, clear=True):
            cfg = PlaygroundServerConfig()
            state = PlaygroundState(cfg)
            out = state.list_models()

        self.assertEqual(cfg.backend_kind, "")
        self.assertEqual(cfg.default_model_id, "")
        self.assertIsNone(out["active"])

    def test_default_config_lists_remote_when_base_url_is_set(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with patch.dict("os.environ", {"ABSTRACTVISION_BASE_URL": "http://localhost:1234/v1"}, clear=True):
            cfg = PlaygroundServerConfig()
            state = PlaygroundState(cfg)
            out = state.list_models()

        self.assertEqual(cfg.backend_kind, "openai")
        models = {m["id"]: m for m in out["models"]}
        self.assertIn("openai-compatible/default", models)
        self.assertEqual(models["openai-compatible/default"]["backend"], "openai")

    def test_generation_job_uses_active_backend_and_returns_b64_json(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState
        from abstractvision.types import GeneratedAsset

        class FakeBackend:
            def generate_image_with_progress(self, request, progress_callback=None):
                if progress_callback:
                    progress_callback(1, request.steps)
                return GeneratedAsset(
                    media_type="image",
                    data=b"\x89PNG\r\n\x1a\nfake",
                    mime_type="image/png",
                    metadata={"prompt": request.prompt},
                )

        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))
        state._active_backend = FakeBackend()
        state._active_backend_kind = "diffusers"
        state._active_model_id = "runwayml/stable-diffusion-v1-5"
        job = state.start_image_generation_job({"prompt": "hello", "steps": 1})

        snap = None
        for _ in range(100):
            snap = state.get_job(job["job_id"])
            if snap["state"] == "succeeded":
                break
            time.sleep(0.01)

        self.assertIsNotNone(snap)
        self.assertEqual(snap["state"], "succeeded")
        self.assertIn("b64_json", snap["result"]["data"][0])
        self.assertEqual(snap["progress"]["step"], 1)

    def test_generation_job_uses_backend_snapshot_if_active_model_changes(self):
        import abstractvision.playground_server as playground_server
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState
        from abstractvision.types import GeneratedAsset

        class FakeBackend:
            def __init__(self, name):
                self.name = name
                self.unloaded = False

            def generate_image_with_progress(self, request, progress_callback=None):
                return GeneratedAsset(
                    media_type="image",
                    data=(b"\x89PNG\r\n\x1a\n" + self.name.encode("ascii")),
                    mime_type="image/png",
                    metadata={"backend": self.name},
                )

            def unload(self):
                self.unloaded = True

        targets = []

        class DeferredThread:
            def __init__(self, *, target, name, daemon):
                self.target = target

            def start(self):
                targets.append(self.target)

        first = FakeBackend("first")
        second = FakeBackend("second")
        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))
        state._active_backend = first
        state._active_backend_kind = "diffusers"
        state._active_model_id = "first-model"

        with patch.object(playground_server.threading, "Thread", DeferredThread):
            job = state.start_image_generation_job({"prompt": "hello", "steps": 1})

        state._active_backend = second
        state._active_backend_kind = "openai"
        state._active_model_id = "second-model"
        targets[0]()
        snap = state.get_job(job["job_id"])

        self.assertEqual(snap["state"], "succeeded")
        self.assertEqual(snap["result"]["data"][0]["metadata"]["backend"], "first")
        self.assertFalse(first.unloaded)

    def test_cli_has_playground_command(self):
        from abstractvision.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["playground", "--host", "127.0.0.1", "--port", "8999"])
        self.assertEqual(args.cmd, "playground")
        self.assertEqual(args.port, 8999)
        self.assertTrue(callable(args._fn))


if __name__ == "__main__":
    unittest.main()
