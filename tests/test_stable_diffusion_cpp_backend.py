import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X9+QAAAABJRU5ErkJggg=="
)


class TestStableDiffusionCppVisionBackend(unittest.TestCase):
    def test_generate_image_builds_cmd_and_reads_output(self):
        from abstractvision.backends.stable_diffusion_cpp import StableDiffusionCppBackendConfig, StableDiffusionCppVisionBackend
        from abstractvision.types import ImageGenerationRequest

        backend = StableDiffusionCppVisionBackend(
            config=StableDiffusionCppBackendConfig(
                sd_cli_path="sd-cli",
                diffusion_model="model.gguf",
                vae="vae.safetensors",
                llm="llm.gguf",
            )
        )

        def fake_run(cmd, check, stdout, stderr, cwd, timeout):
            out_path = cmd[cmd.index("--output") + 1]
            Path(out_path).write_bytes(PNG_1X1)

        with patch("abstractvision.backends.stable_diffusion_cpp._require_sd_cli", return_value="/usr/bin/sd-cli"):
            with patch("abstractvision.backends.stable_diffusion_cpp.subprocess.run", side_effect=fake_run) as run_mock:
                asset = backend.generate_image(
                    ImageGenerationRequest(
                        prompt="hello",
                        negative_prompt="nope",
                        width=64,
                        height=32,
                        steps=12,
                        guidance_scale=2.5,
                        seed=123,
                        extra={"sampling_method": "euler", "offload_to_cpu": True},
                    )
                )

        self.assertEqual(asset.media_type, "image")
        self.assertTrue(asset.data.startswith(b"\x89PNG\r\n\x1a\n"))

        cmd = run_mock.call_args[0][0]
        self.assertIn("--diffusion-model", cmd)
        self.assertIn("model.gguf", cmd)
        self.assertIn("--vae", cmd)
        self.assertIn("vae.safetensors", cmd)
        self.assertIn("--llm", cmd)
        self.assertIn("llm.gguf", cmd)

        self.assertIn("--prompt", cmd)
        self.assertIn("hello", cmd)
        self.assertIn("--negative-prompt", cmd)
        self.assertIn("nope", cmd)

        self.assertIn("--width", cmd)
        self.assertIn("64", cmd)
        self.assertIn("--height", cmd)
        self.assertIn("32", cmd)
        self.assertIn("--steps", cmd)
        self.assertIn("12", cmd)
        self.assertIn("--cfg-scale", cmd)
        self.assertIn("2.5", cmd)
        self.assertIn("--seed", cmd)
        self.assertIn("123", cmd)

        # extra -> flags
        self.assertIn("--sampling-method", cmd)
        self.assertIn("euler", cmd)
        self.assertIn("--offload-to-cpu", cmd)

    def test_edit_image_passes_init_and_mask(self):
        from abstractvision.backends.stable_diffusion_cpp import StableDiffusionCppBackendConfig, StableDiffusionCppVisionBackend
        from abstractvision.types import ImageEditRequest

        backend = StableDiffusionCppVisionBackend(
            config=StableDiffusionCppBackendConfig(
                sd_cli_path="sd-cli",
                model="full_model.safetensors",
            )
        )

        def fake_run(cmd, check, stdout, stderr, cwd, timeout):
            out_path = cmd[cmd.index("--output") + 1]
            Path(out_path).write_bytes(PNG_1X1)

        with patch("abstractvision.backends.stable_diffusion_cpp._require_sd_cli", return_value="/usr/bin/sd-cli"):
            with patch("abstractvision.backends.stable_diffusion_cpp.subprocess.run", side_effect=fake_run) as run_mock:
                asset = backend.edit_image(
                    ImageEditRequest(
                        prompt="edit",
                        image=PNG_1X1,
                        mask=PNG_1X1,
                        steps=5,
                        seed=7,
                        extra={"strength": 0.5},
                    )
                )

        self.assertEqual(asset.mime_type, "image/png")
        self.assertTrue(asset.data.startswith(b"\x89PNG\r\n\x1a\n"))

        cmd = run_mock.call_args[0][0]
        self.assertIn("--init-img", cmd)
        self.assertIn("--mask", cmd)
        self.assertIn("--strength", cmd)
        self.assertIn("0.5", cmd)

    def test_qwen_image_requires_components(self):
        from abstractvision.backends.stable_diffusion_cpp import StableDiffusionCppBackendConfig, StableDiffusionCppVisionBackend
        from abstractvision.errors import OptionalDependencyMissingError
        from abstractvision.types import ImageGenerationRequest

        backend = StableDiffusionCppVisionBackend(
            config=StableDiffusionCppBackendConfig(sd_cli_path="sd-cli", diffusion_model="model.gguf")
        )

        with patch("abstractvision.backends.stable_diffusion_cpp._try_read_gguf_architecture", return_value="qwen_image"):
            with self.assertRaises(OptionalDependencyMissingError):
                backend.generate_image(ImageGenerationRequest(prompt="hello"))


if __name__ == "__main__":
    unittest.main()
