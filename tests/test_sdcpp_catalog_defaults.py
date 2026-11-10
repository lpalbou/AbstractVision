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


def _cmd(model_id, request, edit=False):
    backend = StableDiffusionCppVisionBackend(config=StableDiffusionCppBackendConfig(
        sd_cli_path="sd-cli", diffusion_model="model-Q8_0.gguf", vae="vae.safetensors", llm="llm.gguf",
        capabilities_model_id=model_id))

    def fake_run(cmd, check, stdout, stderr, cwd, timeout=None):
        Path(cmd[cmd.index("--output") + 1]).write_bytes(_PNG)

    with patch("abstractvision.backends.stable_diffusion_cpp._require_sd_cli", return_value="/usr/bin/sd-cli"), \
         patch("abstractvision.backends.stable_diffusion_cpp.subprocess.run", side_effect=fake_run) as run:
        if edit:
            backend.edit_image(request)
        else:
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


    def test_guidance_distilled_flux_sends_embedded_guidance_with_cfg_one(self):
        # sd.cpp docs/flux2.md: FLUX.2-dev runs --cfg-scale 1.0; the Diffusers guidance is --guidance.
        cmd = _cmd("black-forest-labs/FLUX.2-dev", ImageGenerationRequest(prompt="fox"))
        self.assertEqual((_opt(cmd, "--guidance"), _opt(cmd, "--cfg-scale")), ("4.0", "1.0"))
        cmd = _cmd("black-forest-labs/FLUX.2-dev", ImageGenerationRequest(prompt="fox", guidance_scale=2.5))
        self.assertEqual((_opt(cmd, "--guidance"), _opt(cmd, "--cfg-scale")), ("2.5", "1.0"))

    def test_klein_base_keeps_real_cfg(self):
        cmd = _cmd("black-forest-labs/FLUX.2-klein-base-4B", ImageGenerationRequest(prompt="fox"))
        self.assertEqual((_opt(cmd, "--cfg-scale"), _opt(cmd, "--guidance")), ("4.0", None))

    def test_qwen_image_edit_defaults_to_the_documented_cfg(self):
        # sd.cpp docs/qwen_image_edit.md: --cfg-scale 2.5; the catalog's 1.0 is Diffusers' unused field.
        from abstractvision.types import ImageEditRequest

        cmd = _cmd("Qwen/Qwen-Image-Edit-2511", ImageEditRequest(prompt="leopard", image=_PNG), edit=True)
        self.assertEqual((_opt(cmd, "--cfg-scale"), _opt(cmd, "--steps")), ("2.5", "15"))
        cmd = _cmd("Qwen/Qwen-Image-Edit-2511", ImageEditRequest(prompt="leopard", image=_PNG, guidance_scale=4.0), edit=True)
        self.assertEqual(_opt(cmd, "--cfg-scale"), "4.0")

if __name__ == "__main__":
    unittest.main()
