import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class _FakeDiffusersOutput:
    def __init__(self, image):
        self.images = [image]


class _FakePipeline:
    def __init__(self, image):
        self._image = image
        self.to_calls = []
        self.calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeDiffusersOutput(self._image)


def _png_bytes(color=(255, 0, 0)):
    from PIL import Image

    img = Image.new("RGB", (4, 4), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestHuggingFaceDiffusersVisionBackend(unittest.TestCase):
    def test_default_torch_dtype_for_devices(self):
        from abstractvision.backends.huggingface_diffusers import _default_torch_dtype_for_device

        import torch

        self.assertEqual(_default_torch_dtype_for_device(torch, "cuda"), torch.float16)
        self.assertEqual(_default_torch_dtype_for_device(torch, "cuda:0"), torch.float16)
        self.assertEqual(_default_torch_dtype_for_device(torch, "mps"), torch.float16)
        self.assertEqual(_default_torch_dtype_for_device(torch, "mps:0"), torch.float16)

    def test_raises_when_mps_device_unavailable(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        fake_t2i_cls = MagicMock()
        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls),
        ):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="mps", allow_download=False)
            )
            with self.assertRaises(ValueError) as ctx:
                backend.generate_image(ImageGenerationRequest(prompt="hello"))
        self.assertIn("mps", str(ctx.exception).lower())

    def test_generate_image_maps_common_params(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        out_img_bytes = _png_bytes()
        from PIL import Image

        fake_image = Image.open(io.BytesIO(out_img_bytes))

        fake_pipe = _FakePipeline(fake_image)
        fake_t2i_cls = MagicMock()
        fake_t2i_cls.from_pretrained.return_value = fake_pipe

        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls),
        ):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="cpu", allow_download=False)
            )
            asset = backend.generate_image(
                ImageGenerationRequest(
                    prompt="hello",
                    negative_prompt="nope",
                    width=64,
                    height=32,
                    steps=12,
                    guidance_scale=7.5,
                    seed=123,
                    extra={"foo": "bar"},
                )
            )

        self.assertEqual(asset.media_type, "image")
        self.assertEqual(asset.mime_type, "image/png")
        self.assertTrue(asset.data.startswith(b"\x89PNG\r\n\x1a\n"))

        # Pipeline load args.
        self.assertTrue(fake_t2i_cls.from_pretrained.called)
        _, kwargs = fake_t2i_cls.from_pretrained.call_args
        self.assertEqual(kwargs.get("local_files_only"), True)
        self.assertEqual(kwargs.get("use_safetensors"), True)

        # Pipeline call kwargs.
        self.assertEqual(len(fake_pipe.calls), 1)
        call = fake_pipe.calls[0]
        self.assertEqual(call.get("prompt"), "hello")
        self.assertEqual(call.get("negative_prompt"), "nope")
        self.assertEqual(call.get("width"), 64)
        self.assertEqual(call.get("height"), 32)
        self.assertEqual(call.get("num_inference_steps"), 12)
        self.assertEqual(call.get("guidance_scale"), 7.5)
        self.assertIn("generator", call)
        self.assertEqual(call.get("foo"), "bar")

    def test_edit_image_uses_inpaint_when_mask_provided(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageEditRequest

        input_img = _png_bytes(color=(0, 255, 0))
        mask_img = _png_bytes(color=(255, 255, 255))
        out_img_bytes = _png_bytes(color=(0, 0, 255))

        from PIL import Image

        fake_image = Image.open(io.BytesIO(out_img_bytes))
        fake_pipe = _FakePipeline(fake_image)

        fake_t2i_cls = MagicMock()
        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()
        fake_inpaint_cls.from_pretrained.return_value = fake_pipe

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls),
        ):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="cpu", allow_download=False)
            )
            asset = backend.edit_image(ImageEditRequest(prompt="edit", image=input_img, mask=mask_img, steps=5, seed=1))

        self.assertEqual(asset.mime_type, "image/png")
        self.assertTrue(asset.data.startswith(b"\x89PNG\r\n\x1a\n"))

        self.assertTrue(fake_inpaint_cls.from_pretrained.called)
        self.assertEqual(len(fake_pipe.calls), 1)
        call = fake_pipe.calls[0]
        self.assertIn("image", call)
        self.assertIn("mask_image", call)
        self.assertEqual(call.get("prompt"), "edit")
        self.assertEqual(call.get("num_inference_steps"), 5)
        self.assertIn("generator", call)


if __name__ == "__main__":
    unittest.main()
