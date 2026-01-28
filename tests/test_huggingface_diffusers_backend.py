import io
import os
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
        self.lora_loads = []
        self.adapters = None
        self.fused = 0
        self.unfused = 0
        self.unloaded = 0
        self.registered = {}

    def to(self, device):
        self.to_calls.append(device)
        return self

    def register_modules(self, **kwargs):
        self.registered.update(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeDiffusersOutput(self._image)

    def load_lora_weights(self, source: str, adapter_name: str = None, **kwargs):
        self.lora_loads.append({"source": source, "adapter_name": adapter_name, "kwargs": dict(kwargs)})

    def set_adapters(self, names, adapter_weights=None):
        self.adapters = {"names": list(names), "weights": list(adapter_weights) if adapter_weights is not None else None}

    def fuse_lora(self):
        self.fused += 1

    def unfuse_lora(self):
        self.unfused += 1

    def unload_lora_weights(self):
        self.unloaded += 1


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
        self.assertIn(_default_torch_dtype_for_device(torch, "mps"), (torch.bfloat16, torch.float16))
        self.assertIn(_default_torch_dtype_for_device(torch, "mps:0"), (torch.bfloat16, torch.float16))

    def test_maybe_upcasts_vae_to_fp32_on_mps(self):
        from abstractvision.backends.huggingface_diffusers import _maybe_upcast_vae_for_mps

        import torch

        class _FakeVAE:
            dtype = torch.float16

            def __init__(self):
                self.to_kwargs = None

            def to(self, **kwargs):
                self.to_kwargs = dict(kwargs)
                return self

        class _FakePipe:
            def __init__(self):
                self.vae = _FakeVAE()

        pipe = _FakePipe()
        _maybe_upcast_vae_for_mps(torch, pipe, "mps")
        self.assertEqual(pipe.vae.to_kwargs, {"dtype": torch.float32})

    def test_mps_vae_upcast_wraps_encode_decode_inputs(self):
        from abstractvision.backends.huggingface_diffusers import _maybe_upcast_vae_for_mps

        import torch

        class _FakeVAE:
            dtype = torch.float16

            def __init__(self):
                self.encode_seen_dtype = None
                self.decode_seen_dtype = None

            def to(self, **kwargs):
                if "dtype" in kwargs:
                    self.dtype = kwargs["dtype"]
                return self

            def encode(self, x, return_dict=True):
                self.encode_seen_dtype = x.dtype
                if x.dtype != self.dtype:
                    raise RuntimeError("Input type and bias type should be the same")
                return x

            def decode(self, z, return_dict=True, generator=None):
                self.decode_seen_dtype = z.dtype
                if z.dtype != self.dtype:
                    raise RuntimeError("Input type and bias type should be the same")
                return (z,)

        class _FakePipe:
            def __init__(self):
                self.vae = _FakeVAE()

        pipe = _FakePipe()
        _maybe_upcast_vae_for_mps(torch, pipe, "mps")

        pipe.vae.encode(torch.zeros((1, 3, 8, 8), dtype=torch.float16))
        pipe.vae.decode(torch.zeros((1, 4, 8, 8), dtype=torch.float16), return_dict=False)

        self.assertEqual(pipe.vae.dtype, torch.float32)
        self.assertEqual(pipe.vae.encode_seen_dtype, torch.float32)
        self.assertEqual(pipe.vae.decode_seen_dtype, torch.float32)

    def test_raises_when_mps_device_unavailable(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        class _FakeMps:
            @staticmethod
            def is_available():
                return False

        class _FakeBackends:
            mps = _FakeMps()

        class _FakeTorch:
            backends = _FakeBackends()

        fake_diffusion_pipeline_cls = MagicMock()
        fake_t2i_cls = MagicMock()
        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_diffusion_pipeline_cls, fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls, "0.0.0"),
        ), patch("abstractvision.backends.huggingface_diffusers._lazy_import_torch", return_value=_FakeTorch):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="mps")
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
        fake_diffusion_pipeline_cls = MagicMock()
        fake_t2i_cls = MagicMock()
        fake_t2i_cls.from_pretrained.return_value = fake_pipe

        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_diffusion_pipeline_cls, fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls, "0.0.0"),
        ):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="cpu")
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

    def test_offline_mode_sets_hf_env_during_load_and_restores(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        out_img_bytes = _png_bytes()
        from PIL import Image

        fake_image = Image.open(io.BytesIO(out_img_bytes))
        fake_pipe = _FakePipeline(fake_image)

        def _from_pretrained(*_args, **_kwargs):
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
            self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
            self.assertEqual(os.environ.get("DIFFUSERS_OFFLINE"), "1")
            return fake_pipe

        fake_diffusion_pipeline_cls = MagicMock()
        fake_t2i_cls = MagicMock()
        fake_t2i_cls.from_pretrained.side_effect = _from_pretrained
        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()

        old_hf = os.environ.pop("HF_HUB_OFFLINE", None)
        old_tx = os.environ.pop("TRANSFORMERS_OFFLINE", None)
        old_df = os.environ.pop("DIFFUSERS_OFFLINE", None)
        try:
            with patch(
                "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
                return_value=(fake_diffusion_pipeline_cls, fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls, "0.0.0"),
            ):
                backend = HuggingFaceDiffusersVisionBackend(
                    config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="cpu")
                )
                backend.generate_image(ImageGenerationRequest(prompt="hello"))
            self.assertIsNone(os.environ.get("HF_HUB_OFFLINE"))
            self.assertIsNone(os.environ.get("TRANSFORMERS_OFFLINE"))
            self.assertIsNone(os.environ.get("DIFFUSERS_OFFLINE"))
        finally:
            if old_hf is not None:
                os.environ["HF_HUB_OFFLINE"] = old_hf
            if old_tx is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = old_tx
            if old_df is not None:
                os.environ["DIFFUSERS_OFFLINE"] = old_df

    def test_generate_image_does_not_auto_retry_on_invalid_cast_warning(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        out_img_bytes = _png_bytes()
        from PIL import Image

        fake_image = Image.open(io.BytesIO(out_img_bytes))
        fake_pipe = _FakePipeline(fake_image)

        fake_diffusion_pipeline_cls = MagicMock()
        fake_t2i_cls = MagicMock()
        fake_t2i_cls.from_pretrained.return_value = fake_pipe

        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_diffusion_pipeline_cls, fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls, "0.0.0"),
        ):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="cpu")
            )
            with patch.object(backend, "_pipe_call", return_value=(_FakeDiffusersOutput(fake_image), True)), patch.object(
                backend, "_maybe_retry_fp32_on_invalid_output"
            ) as retry:
                asset = backend.generate_image(ImageGenerationRequest(prompt="hello"))

        self.assertEqual(asset.mime_type, "image/png")
        self.assertTrue(asset.metadata.get("had_invalid_cast_warning"))
        self.assertFalse(asset.metadata.get("retried_fp32", False))
        retry.assert_not_called()

    def test_edit_image_uses_inpaint_when_mask_provided(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageEditRequest

        input_img = _png_bytes(color=(0, 255, 0))
        mask_img = _png_bytes(color=(255, 255, 255))
        out_img_bytes = _png_bytes(color=(0, 0, 255))

        from PIL import Image

        fake_image = Image.open(io.BytesIO(out_img_bytes))
        fake_pipe = _FakePipeline(fake_image)

        fake_diffusion_pipeline_cls = MagicMock()
        fake_t2i_cls = MagicMock()
        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()
        fake_inpaint_cls.from_pretrained.return_value = fake_pipe

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_diffusion_pipeline_cls, fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls, "0.0.0"),
        ):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="cpu")
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

    def test_generate_image_applies_loras_from_loras_json_and_caches(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        out_img_bytes = _png_bytes()
        from PIL import Image

        fake_image = Image.open(io.BytesIO(out_img_bytes))
        fake_pipe = _FakePipeline(fake_image)

        # Ensure LoRA loading sees offline env vars.
        def _load_lora_weights(source: str, adapter_name: str = None, **kwargs):
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
            fake_pipe.lora_loads.append({"source": source, "adapter_name": adapter_name, "kwargs": dict(kwargs)})

        fake_pipe.load_lora_weights = _load_lora_weights

        fake_diffusion_pipeline_cls = MagicMock()
        fake_t2i_cls = MagicMock()
        fake_t2i_cls.from_pretrained.return_value = fake_pipe
        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_diffusion_pipeline_cls, fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls, "0.0.0"),
        ):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="cpu")
            )
            req = ImageGenerationRequest(
                prompt="hello",
                extra={
                    "loras_json": '[{"source":"org/lora","scale":0.5},{"source":"org/lora2","weight_name":"x.safetensors"}]'
                },
            )
            a1 = backend.generate_image(req)
            a2 = backend.generate_image(req)

        self.assertEqual(a1.mime_type, "image/png")
        self.assertEqual(a1.metadata.get("lora_signature"), a2.metadata.get("lora_signature"))
        # Should only load once because signature is cached.
        self.assertEqual(len(fake_pipe.lora_loads), 2)
        self.assertEqual(fake_pipe.fused, 1)

    def test_generate_image_applies_rapid_aio_transformer_override(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        out_img_bytes = _png_bytes()
        from PIL import Image

        fake_image = Image.open(io.BytesIO(out_img_bytes))
        fake_pipe = _FakePipeline(fake_image)

        class _FakeTransformer:
            def __init__(self):
                self.to_calls = []

            def to(self, *args, **kwargs):
                self.to_calls.append((args, dict(kwargs)))
                return self

        tr = _FakeTransformer()

        def _from_pretrained(*_args, **_kwargs):
            self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
            self.assertTrue(_kwargs.get("local_files_only"))
            return tr

        fake_diffusion_pipeline_cls = MagicMock()
        fake_t2i_cls = MagicMock()
        fake_t2i_cls.from_pretrained.return_value = fake_pipe
        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_diffusion_pipeline_cls, fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls, "0.0.0"),
        ), patch("diffusers.models.QwenImageTransformer2DModel.from_pretrained", side_effect=_from_pretrained):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="cpu", torch_dtype="float32")
            )
            asset = backend.generate_image(ImageGenerationRequest(prompt="hello", extra={"rapid_aio_repo": "org/rapid"}))

        self.assertEqual(asset.mime_type, "image/png")
        self.assertEqual(asset.metadata.get("rapid_aio_repo"), "org/rapid")
        self.assertIs(fake_pipe.registered.get("transformer"), tr)


if __name__ == "__main__":
    unittest.main()
