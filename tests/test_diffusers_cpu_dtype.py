"""A Diffusers pipeline on the processor loads every component in float32.

Measured on Linux without a usable GPU (torch 2.15, diffusers 0.42.1, transformers 5.22.2,
2026-11-10): `segmind/tiny-sd` stores float16 weights. With no dtype passed to `from_pretrained`,
Diffusers loaded the UNet and VAE in float32 while Transformers 5 loaded the CLIP text encoder in
the checkpoint's float16, and the first generation failed with "mat1 and mat2 must have the same
dtype, but got Half and Float". Returning `None` again from `_default_torch_dtype_for_device` for
"cpu" turns this RED.
No model, no GPU: the pipeline class is a fake that records what it was loaded with.
"""

from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch

try:
    import torch
    from PIL import Image
except ImportError:  # the base CI job has no torch
    torch = None

from abstractvision.backends import huggingface_diffusers as hd


class _Out:
    def __init__(self, image):
        self.images = [image]


class _Pipe:
    def __init__(self, image):
        self._image = image
        self.dtype = None
        self.device = None

    def to(self, *args, **kwargs):
        self.device = kwargs.get("device", args[0] if args else None)
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]
        return self

    def __call__(self, **kwargs):
        return _Out(self._image)


@unittest.skipIf(torch is None, "torch is not installed")
class DiffusersCpuDtypeTests(unittest.TestCase):
    def test_cpu_default_dtype_is_float32(self):
        self.assertIs(hd._default_torch_dtype_for_device(torch, "cpu"), torch.float32)
        self.assertIs(hd._default_torch_dtype_for_device(torch, "CPU"), torch.float32)
        # Accelerators keep their 16-bit default.
        self.assertIs(hd._default_torch_dtype_for_device(torch, "cuda"), torch.float16)

    def test_cpu_load_passes_float32_to_every_component(self):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color=(200, 30, 30)).save(buf, format="PNG")
        pipe = _Pipe(Image.open(io.BytesIO(buf.getvalue())))
        t2i = MagicMock()
        t2i.from_pretrained.return_value = pipe

        from abstractvision.types import ImageGenerationRequest

        # A processor-only host: device "auto" (what the AbstractCore plugin passes) resolves to cpu.
        with patch.object(hd, "_lazy_import_diffusers", return_value=(MagicMock(), t2i, None, None, "0.0.0")), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.backends.mps, "is_available", return_value=False):
            backend = hd.HuggingFaceDiffusersVisionBackend(
                config=hd.HuggingFaceDiffusersBackendConfig(model_id="some/model", device="auto")
            )
            asset = backend.generate_image(ImageGenerationRequest(prompt="hello", width=64, height=64, steps=2))

        _, kwargs = t2i.from_pretrained.call_args
        self.assertIs(kwargs.get("torch_dtype"), torch.float32)
        self.assertNotIn("variant", kwargs)  # no fp16 checkpoint variant on the processor
        self.assertIs(pipe.dtype, torch.float32)
        self.assertEqual(asset.metadata.get("dtype"), "torch.float32")
        self.assertEqual(backend._effective_device(torch), "cpu")


if __name__ == "__main__":
    unittest.main()
