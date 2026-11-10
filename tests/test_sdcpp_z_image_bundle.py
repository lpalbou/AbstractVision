"""Z-Image GGUF runs on stable-diffusion.cpp with its text encoder and VAE.

Measured on a Quadro RTX 5000 (Ubuntu 24.04, sd-cli Vulkan build, 2026-11-10): the catalog's
`z-image-turbo` GGUF preset resolved to the diffusion model alone, and sd-cli stopped with
"new_sd_ctx_t failed". With Qwen3-4B (unsloth/Qwen3-4B-GGUF, Q4_K_M) as `--llm` and the repo's
VAE it generated 512x512 (86 s first call, 38 s second). Removing the `z-image-turbo` / `z-image`
bundles turns this RED. No model: the cache holds placeholder files.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from abstractvision.model_cache import import_directory_to_hf_cache
from abstractvision.model_downloads import resolve_sdcpp_model_selection


def _cache(root: Path, src: Path, repo_id: str, files):
    d = src / repo_id.replace("/", "--")
    for rel in files:
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(b"x")
    import_directory_to_hf_cache(d, repo_id=repo_id, cache_dir=str(root), cleanup_source=False)


class SdcppZImageBundleTests(unittest.TestCase):
    def test_z_image_gguf_resolves_its_text_encoder_and_vae(self):
        for key, gguf_repo, gguf, vae_repo in (
            ("z-image-turbo", "unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q8_0.gguf", "Tongyi-MAI/Z-Image-Turbo"),
            ("z-image", "unsloth/Z-Image-GGUF", "z-image-Q8_0.gguf", "Tongyi-MAI/Z-Image"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as cache_td, tempfile.TemporaryDirectory() as src_td:
                root, src = Path(cache_td), Path(src_td)
                _cache(root, src, gguf_repo, [gguf])
                _cache(root, src, vae_repo, ["vae/diffusion_pytorch_model.safetensors"])
                _cache(root, src, "unsloth/Qwen3-4B-GGUF", ["Qwen3-4B-Q4_K_M.gguf"])
                with patch("abstractvision.model_downloads.local_model_profile", return_value="cuda"), \
                     patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                    sel = resolve_sdcpp_model_selection(key, allow_download=False)
                self.assertIsNone(sel.model)
                self.assertTrue(str(sel.diffusion_model).endswith(gguf))
                self.assertTrue(str(sel.vae).endswith("vae/diffusion_pytorch_model.safetensors"))
                self.assertIn("models--" + vae_repo.replace("/", "--"), str(sel.vae))
                self.assertTrue(str(sel.llm).endswith("Qwen3-4B-Q4_K_M.gguf"))


if __name__ == "__main__":
    unittest.main()
