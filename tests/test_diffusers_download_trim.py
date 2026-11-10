"""A Diffusers preset downloads only the files the Diffusers backend loads.

Measured 2026-11-10 (Hugging Face listings): the Diffusers presets downloaded every safetensors
file, so stabilityai/sdxl-turbo took 41.6 GB (root single-file checkpoints plus the fp32 and fp16
sets) for the 6.9 GB the backend reads on a GPU, and black-forest-labs/FLUX.2-klein-4B 23.7 GB
for 16.0 GB (a 7.75 GB root single-file copy of the transformer). Downloading by the preset
patterns again turns this RED. No network: the listings are the repos' real ones.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from abstractvision import model_downloads as md

_SDXL_TURBO = [
    ".gitattributes", "LICENSE.md", "README.md", "image_quality_one_step.png", "model_index.json",
    "scheduler/scheduler_config.json", "sd_xl_turbo_1.0.safetensors", "sd_xl_turbo_1.0_fp16.safetensors",
    "text_encoder/config.json", "text_encoder/model.fp16.safetensors", "text_encoder/model.onnx",
    "text_encoder/model.safetensors", "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors",
    "text_encoder_2/model.onnx", "text_encoder_2/model.onnx_data", "text_encoder_2/model.safetensors",
    "tokenizer/merges.txt", "tokenizer/vocab.json", "tokenizer_2/merges.txt", "tokenizer_2/vocab.json",
    "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.safetensors",
    "unet/model.onnx", "unet/model.onnx_data", "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.safetensors", "vae_decoder/config.json", "vae_decoder/model.onnx",
]
_SDXL_INDEX = {
    "_class_name": "StableDiffusionXLPipeline", "feature_extractor": [None, None], "image_encoder": [None, None],
    "scheduler": ["diffusers", "EulerAncestralDiscreteScheduler"], "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"], "tokenizer": ["transformers", "CLIPTokenizer"],
    "tokenizer_2": ["transformers", "CLIPTokenizer"], "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}
_KLEIN = [
    "LICENSE.md", "README.md", "flux-2-klein-4b.safetensors", "model_index.json", "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors", "text_encoder/model.safetensors.index.json",
    "tokenizer/chat_template.jinja", "tokenizer/tokenizer.json", "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
]
_KLEIN_COMPONENTS = ["scheduler", "text_encoder", "tokenizer", "transformer", "vae"]
_SDXL_COMPONENTS = ["scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet", "vae"]


def _weights(files):
    return sorted(f for f in files if f.endswith((".safetensors", ".onnx", ".onnx_data", ".bin")))


class DiffusersDownloadFilesTests(unittest.TestCase):
    def test_gpu_takes_the_fp16_set_and_no_duplicates(self):
        got = md.diffusers_download_files(_SDXL_TURBO, components=_SDXL_COMPONENTS, variant="fp16")
        self.assertEqual(_weights(got), [
            "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors",
        ])
        self.assertIn("model_index.json", got)
        self.assertIn("unet/config.json", got)
        self.assertIn("tokenizer_2/vocab.json", got)
        self.assertFalse([f for f in got if f.startswith("vae_decoder/")])  # not a model_index component

    def test_processor_takes_the_default_set(self):
        got = md.diffusers_download_files(_SDXL_TURBO, components=_SDXL_COMPONENTS, variant=None)
        self.assertEqual(_weights(got), [
            "text_encoder/model.safetensors", "text_encoder_2/model.safetensors",
            "unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors",
        ])

    def test_sharded_default_set_without_variant_and_no_root_checkpoint(self):
        got = md.diffusers_download_files(_KLEIN, components=_KLEIN_COMPONENTS, variant="fp16")
        self.assertNotIn("flux-2-klein-4b.safetensors", got)
        self.assertEqual(_weights(got), [
            "text_encoder/model-00001-of-00002.safetensors", "text_encoder/model-00002-of-00002.safetensors",
            "transformer/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors",
        ])
        self.assertIn("text_encoder/model.safetensors.index.json", got)
        self.assertIn("tokenizer/chat_template.jinja", got)

    def test_fp16_only_folder_is_kept_on_the_processor(self):
        files = ["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors"]
        got = md.diffusers_download_files(files, components=["unet"], variant=None)
        self.assertIn("unet/diffusion_pytorch_model.fp16.safetensors", got)


class DownloadModelPresetTrimTests(unittest.TestCase):
    def _download(self, all_files, profile="cuda"):
        preset = md.find_model_preset("sdxl-turbo", target="diffusers", engine="diffusers", require_8bit=False)
        captured = {}

        with tempfile.TemporaryDirectory() as td:
            index = Path(td) / "model_index.json"
            index.write_text(json.dumps(_SDXL_INDEX), encoding="utf-8")
            snap = Path(td) / "snap"
            (snap / "unet").mkdir(parents=True)
            (snap / "model_index.json").write_text("{}", encoding="utf-8")
            (snap / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")

            def fake_snapshot_download(**kw):
                captured.update(kw)
                return str(snap)

            api = MagicMock()
            api.list_repo_files.return_value = list(_SDXL_TURBO)
            with patch.object(md, "local_model_profile", return_value=profile), \
                 patch.object(md, "ensure_hf_repo_snapshot", return_value=None), \
                 patch("huggingface_hub.HfApi", return_value=api), \
                 patch("huggingface_hub.hf_hub_download", return_value=str(index)), \
                 patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download):
                md.download_model_preset(preset, all_files=all_files)
        return captured["allow_patterns"], preset

    def test_cuda_download_asks_only_for_the_fp16_set(self):
        allow, _ = self._download(all_files=False)
        self.assertEqual(_weights(allow), [
            "text_encoder/model.fp16.safetensors", "text_encoder_2/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors",
        ])
        self.assertNotIn("sd_xl_turbo_1.0.safetensors", allow)

    def test_all_files_keeps_the_preset_patterns(self):
        allow, preset = self._download(all_files=True)
        self.assertEqual(allow, list(preset.allow_patterns))



class DownloadCliAllFilesTests(unittest.TestCase):
    def test_all_files_flag_reaches_the_download(self):
        import contextlib
        import io

        from abstractvision.cli import main

        seen = []

        def fake_download(preset, **kw):
            seen.append(kw.get("all_files", False))
            return Path("/tmp/hf-cache") / "snapshots" / "abc123"

        with patch("abstractvision.cli.download_model_preset", new=fake_download), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["download", "sdxl-turbo", "--provider", "diffusers"]), 0)
            self.assertEqual(main(["download", "sdxl-turbo", "--provider", "diffusers", "--all-files"]), 0)
        self.assertEqual(seen, [False, True])

if __name__ == "__main__":
    unittest.main()
