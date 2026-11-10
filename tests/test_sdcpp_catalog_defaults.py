"""stable-diffusion.cpp runs at the catalog's steps and guidance when the request sets none.

Measured on a Quadro RTX 5000 (sd-cli Vulkan, 2026-11-10): GGUF generations ran sd-cli's own
defaults (20 steps, cfg 7) because the backend passed `--steps` / `--cfg-scale` only when the
request set them: FLUX.2 [klein] 4B ran 20 steps instead of its catalog 4, Z-Image-Turbo 20 instead
of 9. Dropping `_with_catalog_defaults` turns this RED. No sd-cli: the command is captured.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from abstractvision.backends.stable_diffusion_cpp import StableDiffusionCppBackendConfig, StableDiffusionCppVisionBackend
from abstractvision.types import ImageGenerationRequest

_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _cmd(model_id, request):
    backend = StableDiffusionCppVisionBackend(config=StableDiffusionCppBackendConfig(
        sd_cli_path="sd-cli", diffusion_model="model-Q8_0.gguf", vae="vae.safetensors", llm="llm.gguf",
        capabilities_model_id=model_id))

    def fake_run(cmd, check, stdout, stderr, cwd, timeout=None):
        Path(cmd[cmd.index("--output") + 1]).write_bytes(_PNG)

    with patch("abstractvision.backends.stable_diffusion_cpp._require_sd_cli", return_value="/usr/bin/sd-cli"), \
         patch("abstractvision.backends.stable_diffusion_cpp.subprocess.run", side_effect=fake_run) as run:
        backend.generate_image(request)
    return run.call_args[0][0]


def _opt(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


class SdcppCatalogDefaultsTests(unittest.TestCase):
    def test_flux2_klein_runs_four_steps_without_cfg(self):
        cmd = _cmd("black-forest-labs/FLUX.2-klein-4B", ImageGenerationRequest(prompt="fox"))
        self.assertEqual(_opt(cmd, "--steps"), "4")
        self.assertEqual(_opt(cmd, "--cfg-scale"), "1.0")

    def test_z_image_turbo_guidance_zero_becomes_cfg_one(self):
        cmd = _cmd("Tongyi-MAI/Z-Image-Turbo", ImageGenerationRequest(prompt="fox"))
        self.assertEqual(_opt(cmd, "--steps"), "9")
        self.assertEqual(_opt(cmd, "--cfg-scale"), "1.0")

    def test_request_values_win(self):
        cmd = _cmd("black-forest-labs/FLUX.2-klein-4B", ImageGenerationRequest(prompt="fox", steps=7, guidance_scale=2.5))
        self.assertEqual((_opt(cmd, "--steps"), _opt(cmd, "--cfg-scale")), ("7", "2.5"))

    def test_without_a_catalog_model_nothing_is_added(self):
        cmd = _cmd(None, ImageGenerationRequest(prompt="fox"))
        self.assertNotIn("--steps", cmd)
        self.assertNotIn("--cfg-scale", cmd)


if __name__ == "__main__":
    unittest.main()
