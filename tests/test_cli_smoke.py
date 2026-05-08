import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestCliSmoke(unittest.TestCase):
    def test_models_lists_known_id(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["models"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("Qwen/Qwen-Image-2512", out)
        self.assertIn("black-forest-labs/FLUX.2-klein-4B", out)

    def test_tasks_lists_text_to_image(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["tasks"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("text_to_image", out)

    def test_show_model_prints_tasks_section(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["show-model", "zai-org/GLM-Image"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("zai-org/GLM-Image", out)
        self.assertIn("tasks:", out)

    def test_provider_models_lists_openai_compatible_catalog(self):
        from abstractvision.cli import main

        class _Resp:
            headers = {}

            def read(self):
                return json.dumps({"data": [{"id": "provider/image-model"}]}).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        seen = {}

        def fake_urlopen(req, timeout=0):
            seen["url"] = req.full_url
            seen["method"] = req.get_method()
            return _Resp()

        buf = io.StringIO()
        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            with contextlib.redirect_stdout(buf):
                rc = main(["provider-models", "--base-url", "http://localhost:1234/v1"])

        self.assertEqual(rc, 0)
        self.assertEqual(seen, {"url": "http://localhost:1234/v1/models", "method": "GET"})
        self.assertIn("provider/image-model", buf.getvalue())

    def test_provider_models_openai_uses_default_catalog(self):
        from abstractvision.cli import main

        class _Resp:
            headers = {}

            def read(self):
                return json.dumps(
                    {
                        "data": [
                            {"id": "gpt-4.1"},
                            {"id": "gpt-image-1"},
                            {"id": "dall-e-3"},
                        ]
                    }
                ).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        seen = {}

        def fake_urlopen(req, timeout=0):
            seen["url"] = req.full_url
            seen["method"] = req.get_method()
            seen["auth"] = req.headers.get("Authorization")
            return _Resp()

        buf = io.StringIO()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
                with contextlib.redirect_stdout(buf):
                    rc = main(["provider-models", "--openai", "--task", "text_to_image"])

        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertEqual(
            seen,
            {
                "url": "https://api.openai.com/v1/models",
                "method": "GET",
                "auth": "Bearer sk-test",
            },
        )
        self.assertIn("gpt-image-1", out)
        self.assertIn("dall-e-3", out)
        self.assertNotIn("gpt-4.1", out)

    def test_repl_help_prioritizes_small_local_examples(self):
        from abstractvision.cli import _repl_help

        out = _repl_help()
        self.assertIn("runwayml/stable-diffusion-v1-5", out)
        self.assertIn("cache-only by default", out)
        self.assertIn("black-forest-labs/FLUX.2-klein-4B", out)
        self.assertIn("/backend sdcpp <model.gguf|model.safetensors> [sd_cli_path]", out)
        self.assertIn("/provider-models", out)
        self.assertIn("--negative-prompt", out)
        self.assertNotIn("FLUX.2-klein-9B", out)
        self.assertNotIn("--negative ...", out)

    def test_repl_state_starts_unconfigured_without_backend_env(self):
        from abstractvision.cli import DEFAULT_DIFFUSERS_DEVICE, _ReplState

        with patch.dict("os.environ", {}, clear=True):
            state = _ReplState()

        self.assertEqual(state.backend_kind, "")
        self.assertIsNone(state.model_id)
        self.assertEqual(state.diffusers_device, DEFAULT_DIFFUSERS_DEVICE)
        self.assertFalse(state.diffusers_allow_download)
        self.assertEqual(state.defaults["t2i"]["width"], 512)
        self.assertEqual(state.defaults["t2i"]["height"], 512)

    def test_repl_state_defaults_to_openai_when_base_url_is_configured(self):
        from abstractvision.cli import _ReplState

        with patch.dict("os.environ", {"ABSTRACTVISION_BASE_URL": "http://localhost:1234/v1"}, clear=True):
            state = _ReplState()

        self.assertEqual(state.backend_kind, "openai")
        self.assertEqual(state.base_url, "http://localhost:1234/v1")
        self.assertIsNone(state.model_id)

    def test_repl_state_openai_override_does_not_inherit_diffusers_model(self):
        from abstractvision.cli import _ReplState

        with patch.dict("os.environ", {"ABSTRACTVISION_BACKEND": "openai"}, clear=True):
            state = _ReplState()

        self.assertEqual(state.backend_kind, "openai")
        self.assertIsNone(state.model_id)

    def test_repl_accepts_sdcpp_single_model_backend(self):
        from abstractvision.cli import main

        commands = iter(
            [
                "/backend sdcpp /models/sd-v1-5.gguf /opt/sd-cli",
                "/config",
                "/exit",
            ]
        )
        buf = io.StringIO()
        with patch("builtins.input", side_effect=lambda _prompt: next(commands)):
            with contextlib.redirect_stdout(buf):
                rc = main(["repl"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn('"backend_kind": "sdcpp"', out)
        self.assertIn('"sdcpp_bin": "/opt/sd-cli"', out)
        self.assertIn('"sdcpp_model": "/models/sd-v1-5.gguf"', out)
        self.assertIn('"sdcpp_diffusion_model": null', out)
        self.assertIn('"model_id": null', out)


if __name__ == "__main__":
    unittest.main()
