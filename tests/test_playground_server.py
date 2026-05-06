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

    def test_cli_has_playground_command(self):
        from abstractvision.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["playground", "--host", "127.0.0.1", "--port", "8999"])
        self.assertEqual(args.cmd, "playground")
        self.assertEqual(args.port, 8999)
        self.assertTrue(callable(args._fn))


if __name__ == "__main__":
    unittest.main()
