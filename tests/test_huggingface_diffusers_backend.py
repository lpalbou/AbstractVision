import io
import os
import sys
import tempfile
import threading
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


class _FakeVideoDiffusersOutput:
    def __init__(self, frames):
        self.frames = [list(frames)]


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
        self.device = None
        self.dtype = None

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args:
            device = args[0]
        if device is not None:
            self.to_calls.append(device)
            self.device = device
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]
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
    @classmethod
    def setUpClass(cls):
        try:
            import PIL  # noqa: F401
            import torch  # noqa: F401
        except Exception as e:
            raise unittest.SkipTest(
                "Diffusers backend unit tests require local test deps; install abstractvision[test] or abstractvision[diffusers]."
            ) from e

    def test_default_torch_dtype_for_devices(self):
        from abstractvision.backends.huggingface_diffusers import _default_torch_dtype_for_device

        import torch

        self.assertEqual(_default_torch_dtype_for_device(torch, "cuda"), torch.float16)
        self.assertEqual(_default_torch_dtype_for_device(torch, "cuda:0"), torch.float16)
        self.assertIn(_default_torch_dtype_for_device(torch, "mps"), (torch.bfloat16, torch.float16))
        self.assertIn(_default_torch_dtype_for_device(torch, "mps:0"), (torch.bfloat16, torch.float16))

    def test_qwen_image_edit_prefers_bf16_on_mps_for_edit_tasks(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

        class _FakeTorch:
            float16 = object()
            bfloat16 = object()

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(
                model_id="Qwen/Qwen-Image-Edit-2511",
                device="mps",
                torch_dtype=None,
            )
        )

        picked = backend._preferred_torch_dtype_for_kind("i2i", "mps", _FakeTorch, _FakeTorch.float16)
        self.assertIs(picked, _FakeTorch.bfloat16)

        picked = backend._preferred_torch_dtype_for_kind("inpaint", "mps", _FakeTorch, _FakeTorch.float16)
        self.assertIs(picked, _FakeTorch.bfloat16)

        backend_explicit = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(
                model_id="Qwen/Qwen-Image-Edit-2511",
                device="mps",
                torch_dtype="float16",
            )
        )
        picked = backend_explicit._preferred_torch_dtype_for_kind("i2i", "mps", _FakeTorch, _FakeTorch.float16)
        self.assertIs(picked, _FakeTorch.float16)

    def test_ensure_pipeline_chat_templates_reads_snapshot_files(self):
        from abstractvision.backends.huggingface_diffusers import _ensure_pipeline_chat_templates

        class _FakeTokenizer:
            chat_template = None

        class _FakePeTokenizer:
            chat_template = None

        class _FakeProcessor:
            chat_template = None

        class _FakePipe:
            def __init__(self):
                self.tokenizer = _FakeTokenizer()
                self.pe_tokenizer = _FakePeTokenizer()
                self.processor = _FakeProcessor()

        with tempfile.TemporaryDirectory() as td:
            snap = Path(td)
            (snap / "tokenizer").mkdir(parents=True, exist_ok=True)
            (snap / "pe_tokenizer").mkdir(parents=True, exist_ok=True)
            (snap / "processor").mkdir(parents=True, exist_ok=True)
            (snap / "tokenizer" / "chat_template.jinja").write_text("tokenizer-template", encoding="utf-8")
            (snap / "pe_tokenizer" / "chat_template.jinja").write_text("pe-tokenizer-template", encoding="utf-8")
            (snap / "processor" / "chat_template.jinja").write_text("processor-template", encoding="utf-8")

            pipe = _FakePipe()
            _ensure_pipeline_chat_templates(pipe, snapshot_dir=snap, model_id="zai-org/GLM-Image")

        self.assertEqual(pipe.tokenizer.chat_template, "tokenizer-template")
        self.assertEqual(pipe.pe_tokenizer.chat_template, "pe-tokenizer-template")
        self.assertEqual(pipe.processor.chat_template, "processor-template")

    def test_ensure_pipeline_chat_templates_applies_fallbacks(self):
        from abstractvision.backends.huggingface_diffusers import _ensure_pipeline_chat_templates

        class _FakeTokenizer:
            chat_template = None

        class _FakePeTokenizer:
            chat_template = None

        class _FakeProcessor:
            chat_template = None

        class _FakePipe:
            def __init__(self, name, *, include_pe_tokenizer: bool = False):
                self.tokenizer = _FakeTokenizer()
                self.processor = _FakeProcessor()
                self.pe_tokenizer = _FakePeTokenizer() if include_pe_tokenizer else None
                self.__class__.__name__ = name

        z_pipe = _FakePipe("ZImagePipeline")
        _ensure_pipeline_chat_templates(z_pipe, snapshot_dir=None, model_id="Tongyi-MAI/Z-Image-Turbo")
        self.assertIsInstance(z_pipe.tokenizer.chat_template, str)
        self.assertIn("<|im_start|>", z_pipe.tokenizer.chat_template)

        glm_pipe = _FakePipe("GlmImagePipeline")
        _ensure_pipeline_chat_templates(glm_pipe, snapshot_dir=None, model_id="zai-org/GLM-Image")
        self.assertIsInstance(glm_pipe.processor.chat_template, str)
        self.assertIn("<|image|>", glm_pipe.processor.chat_template)

        ernie_pipe = _FakePipe("ErnieImagePipeline", include_pe_tokenizer=True)
        _ensure_pipeline_chat_templates(ernie_pipe, snapshot_dir=None, model_id="baidu/ERNIE-Image-Turbo")
        self.assertIsInstance(ernie_pipe.pe_tokenizer.chat_template, str)
        self.assertIn("[SYSTEM_PROMPT]", ernie_pipe.pe_tokenizer.chat_template)

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

    def test_get_capabilities_uses_registry_tasks_for_known_model(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="baidu/ERNIE-Image-Turbo", device="cpu")
        )
        caps = backend.get_capabilities()

        self.assertEqual(set(caps.supported_tasks or []), {"text_to_image"})
        self.assertFalse(caps.supports_mask)

    def test_get_capabilities_for_cogvideox_2b_is_temporarily_disabled(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="zai-org/CogVideoX-2b", device="cpu")
        )
        caps = backend.get_capabilities()

        self.assertEqual(list(caps.supported_tasks or []), [])
        self.assertIsNone(caps.supports_mask)

    def test_get_capabilities_for_glm_image_is_temporarily_disabled(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="zai-org/GLM-Image", device="cpu")
        )
        caps = backend.get_capabilities()

        self.assertEqual(list(caps.supported_tasks or []), [])
        self.assertIsNone(caps.supports_mask)

    def test_normalize_glm_generation_request_uses_registry_defaults(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="zai-org/GLM-Image", device="cpu")
        )
        normalized = backend.normalize_image_generation_request(
            ImageGenerationRequest(prompt="fox", width=513, height=510)
        )

        self.assertEqual(normalized.steps, 50)
        self.assertEqual(normalized.guidance_scale, 1.5)
        self.assertEqual(normalized.width, 544)
        self.assertEqual(normalized.height, 512)

    def test_normalize_glm_generation_request_fills_required_dimensions_when_missing(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="zai-org/GLM-Image", device="cpu")
        )
        normalized = backend.normalize_image_generation_request(ImageGenerationRequest(prompt="fox"))

        self.assertEqual(normalized.steps, 50)
        self.assertEqual(normalized.guidance_scale, 1.5)
        self.assertEqual(normalized.width, 512)
        self.assertEqual(normalized.height, 512)

    def test_normalize_flux2_generation_request_applies_registry_constraints(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="black-forest-labs/FLUX.2-klein-9B", device="cpu")
        )
        normalized = backend.normalize_image_generation_request(
            ImageGenerationRequest(
                prompt="fox",
                negative_prompt="bad anatomy",
                guidance_scale=7.0,
            )
        )

        self.assertIsNone(normalized.negative_prompt)
        self.assertEqual(normalized.guidance_scale, 1.0)

    def test_normalize_cogvideox_video_request_uses_registry_defaults(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import VideoGenerationRequest

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="zai-org/CogVideoX-2b", device="cpu")
        )
        normalized = backend.normalize_video_generation_request(
            VideoGenerationRequest(prompt="fox", width=513, height=511)
        )

        self.assertEqual(normalized.width, 720)
        self.assertEqual(normalized.height, 480)
        self.assertEqual(normalized.fps, 8)
        self.assertEqual(normalized.num_frames, 49)
        self.assertEqual(normalized.steps, 50)
        self.assertEqual(normalized.guidance_scale, 6.0)

    def test_mps_vae_upcast_can_be_disabled_for_video_pipelines(self):
        from abstractvision.backends.huggingface_diffusers import _maybe_upcast_vae_for_mps

        class FakeTorch:
            float16 = "float16"
            float32 = "float32"

        class FakeVAE:
            dtype = FakeTorch.float16

            def __init__(self):
                self.to_calls = []

            def to(self, **kwargs):
                self.to_calls.append(dict(kwargs))

        class FakePipe:
            def __init__(self):
                self.vae = FakeVAE()

        pipe = FakePipe()
        _maybe_upcast_vae_for_mps(FakeTorch, pipe, "mps", allow_fp32_vae=False)
        self.assertEqual(pipe.vae.to_calls, [])

    def test_normalize_glm_edit_request_derives_dimensions_from_input(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageEditRequest
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "input.png"
            Image.new("RGB", (513, 510), color=(12, 34, 56)).save(path, format="PNG")
            image_bytes = path.read_bytes()

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="zai-org/GLM-Image", device="cpu")
        )
        normalized = backend.normalize_image_edit_request(ImageEditRequest(prompt="watercolor", image=image_bytes))

        self.assertEqual(normalized.steps, 15)
        self.assertEqual(normalized.guidance_scale, 1.5)
        self.assertEqual(normalized.extra.get("width"), 544)
        self.assertEqual(normalized.extra.get("height"), 512)

    def test_generate_image_rejects_temporarily_disabled_glm_model(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="zai-org/GLM-Image", device="cpu")
        )

        with self.assertRaisesRegex(Exception, "temporarily disabled"):
            backend.generate_image(ImageGenerationRequest(prompt="fox"))

    def test_edit_image_rejects_temporarily_disabled_glm_model(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageEditRequest

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="zai-org/GLM-Image", device="cpu")
        )

        with self.assertRaisesRegex(Exception, "temporarily disabled"):
            backend.edit_image(ImageEditRequest(prompt="edit", image=_png_bytes(), steps=1, seed=1))

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

    def test_generate_video_rejects_temporarily_disabled_cogvideox_model(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import VideoGenerationRequest

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="zai-org/CogVideoX-2b", device="cpu")
        )

        with self.assertRaisesRegex(Exception, "temporarily disabled"):
            backend.generate_video(VideoGenerationRequest(prompt="hello", steps=12))

    def test_preload_runs_t2i_warmup_once_per_loaded_pipeline(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

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
            backend.preload()
            backend.preload()

        self.assertEqual(fake_t2i_cls.from_pretrained.call_count, 1)
        self.assertEqual(len(fake_pipe.calls), 1)
        warmup = fake_pipe.calls[0]
        self.assertEqual(warmup.get("prompt"), "abstractvision preload warmup")
        self.assertEqual(warmup.get("num_inference_steps"), 1)
        self.assertIn("generator", warmup)
        self.assertNotIn("width", warmup)
        self.assertNotIn("height", warmup)

    def test_preload_warms_up_i2i_only_models_without_loading_t2i(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

        out_img_bytes = _png_bytes()
        from PIL import Image

        fake_image = Image.open(io.BytesIO(out_img_bytes))
        fake_pipe = _FakePipeline(fake_image)

        fake_diffusion_pipeline_cls = MagicMock()
        fake_t2i_cls = MagicMock()
        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()

        # Make i2i load succeed; ensure t2i is never loaded for i2i-only models.
        fake_i2i_cls.from_pretrained.return_value = fake_pipe

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_diffusion_pipeline_cls, fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls, "0.0.0"),
        ):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="Qwen/Qwen-Image-Edit-2511", device="cpu")
            )
            backend.preload()
            backend.preload()

        self.assertEqual(fake_i2i_cls.from_pretrained.call_count, 1)
        self.assertEqual(fake_t2i_cls.from_pretrained.call_count, 0)
        self.assertEqual(len(fake_pipe.calls), 1)

    def test_preload_rejects_temporarily_disabled_glm_model(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="zai-org/GLM-Image", device="cpu")
        )
        with self.assertRaisesRegex(Exception, "temporarily disabled"):
            backend.preload()

    def test_preload_rejects_temporarily_disabled_cogvideox_model(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(model_id="zai-org/CogVideoX-2b", device="cpu")
        )
        with self.assertRaisesRegex(Exception, "temporarily disabled"):
            backend.preload()

    def test_preload_serializes_concurrent_t2i_warmup(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

        out_img_bytes = _png_bytes()
        from PIL import Image

        fake_image = Image.open(io.BytesIO(out_img_bytes))

        class _BlockingPipeline(_FakePipeline):
            def __init__(self, image):
                super().__init__(image)
                self.started = threading.Event()
                self.proceed = threading.Event()
                self.active_calls = 0
                self.max_active_calls = 0

            def __call__(self, **kwargs):
                self.active_calls += 1
                self.max_active_calls = max(self.max_active_calls, self.active_calls)
                self.calls.append(kwargs)
                self.started.set()
                try:
                    self.proceed.wait(timeout=2.0)
                finally:
                    self.active_calls -= 1
                return _FakeDiffusersOutput(self._image)

        fake_pipe = _BlockingPipeline(fake_image)
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
            t1 = threading.Thread(target=backend.preload)
            t2 = threading.Thread(target=backend.preload)
            t1.start()
            self.assertTrue(fake_pipe.started.wait(timeout=1.0))
            t2.start()
            fake_pipe.proceed.set()
            t1.join(timeout=2.0)
            t2.join(timeout=2.0)

        self.assertFalse(t1.is_alive())
        self.assertFalse(t2.is_alive())
        self.assertEqual(len(fake_pipe.calls), 1)
        self.assertEqual(fake_pipe.max_active_calls, 1)

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
                    config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="cpu", allow_download=False)
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

    def test_offline_mode_uses_cached_snapshot_path_and_disables_implicit_token(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        out_img_bytes = _png_bytes()
        from PIL import Image

        fake_image = Image.open(io.BytesIO(out_img_bytes))
        fake_pipe = _FakePipeline(fake_image)

        with tempfile.TemporaryDirectory() as td:
            hf_home = Path(td)
            repo_dir = hf_home / "hub" / "models--runwayml--stable-diffusion-v1-5"
            snap_dir = repo_dir / "snapshots" / "abc123"
            snap_dir.mkdir(parents=True)
            (snap_dir / "model_index.json").write_text("{}", encoding="utf-8")
            (snap_dir / "unet").mkdir(parents=True, exist_ok=True)
            (snap_dir / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
            (repo_dir / "refs").mkdir(parents=True)
            (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")

            def _from_pretrained(model_arg, **_kwargs):
                self.assertEqual(str(model_arg), str(snap_dir))
                self.assertTrue(_kwargs.get("local_files_only"))
                self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
                self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
                self.assertEqual(os.environ.get("DIFFUSERS_OFFLINE"), "1")
                self.assertEqual(os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN"), "1")
                return fake_pipe

            fake_diffusion_pipeline_cls = MagicMock()
            fake_t2i_cls = MagicMock()
            fake_t2i_cls.from_pretrained.side_effect = _from_pretrained
            fake_i2i_cls = MagicMock()
            fake_inpaint_cls = MagicMock()

            with patch.dict("os.environ", {"HF_HOME": str(hf_home)}, clear=False), patch(
                "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
                return_value=(fake_diffusion_pipeline_cls, fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls, "0.0.0"),
            ):
                backend = HuggingFaceDiffusersVisionBackend(
                    config=HuggingFaceDiffusersBackendConfig(
                        model_id="runwayml/stable-diffusion-v1-5",
                        device="cpu",
                    )
                )
                asset = backend.generate_image(ImageGenerationRequest(prompt="hello"))

        self.assertEqual(asset.mime_type, "image/png")
        self.assertTrue(fake_t2i_cls.from_pretrained.called)

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

    def test_text_to_video_dtype_retry_on_mps_does_not_escalate_above_16bit(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

        class _FakeTorch:
            float16 = object()
            float32 = object()
            bfloat16 = object()

        pipe = MagicMock()
        pipe.dtype = _FakeTorch.float16

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(
                model_id="zai-org/CogVideoX-2b",
                device="mps",
                torch_dtype="float16",
                auto_retry_fp32=True,
            )
        )

        with patch("abstractvision.backends.huggingface_diffusers._lazy_import_torch", return_value=_FakeTorch), patch(
            "abstractvision.backends.huggingface_diffusers._move_pipe_to_device"
        ) as move:
            out = backend._maybe_retry_on_dtype_mismatch(
                kind="t2v",
                pipe=pipe,
                kwargs={"prompt": "hello"},
                error=RuntimeError("Input type and bias type should be the same"),
            )

        self.assertIsNone(out)
        move.assert_not_called()

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

        fake_qwen_tr_cls = MagicMock()
        fake_qwen_tr_cls.from_pretrained.side_effect = _from_pretrained

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_diffusion_pipeline_cls, fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls, "0.0.0"),
        ), patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_qwen_image_transformer_2d_model",
            return_value=fake_qwen_tr_cls,
        ):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="cpu", torch_dtype="float32")
            )
            asset = backend.generate_image(ImageGenerationRequest(prompt="hello", extra={"rapid_aio_repo": "org/rapid"}))

        self.assertEqual(asset.mime_type, "image/png")
        self.assertEqual(asset.metadata.get("rapid_aio_repo"), "org/rapid")
        self.assertIs(fake_pipe.registered.get("transformer"), tr)

    def test_auto_device_prefers_cuda_and_uses_fp16_variant(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        out_img_bytes = _png_bytes()
        from PIL import Image

        fake_image = Image.open(io.BytesIO(out_img_bytes))
        fake_pipe = _FakePipeline(fake_image)

        class _FakeCuda:
            @staticmethod
            def is_available():
                return True

        class _FakeMps:
            @staticmethod
            def is_available():
                return True

        class _FakeBackends:
            mps = _FakeMps()

        class _FakeTorch:
            cuda = _FakeCuda()
            backends = _FakeBackends()
            float16 = object()
            float32 = object()
            bfloat16 = object()

        fake_diffusion_pipeline_cls = MagicMock()
        fake_t2i_cls = MagicMock()
        fake_t2i_cls.from_pretrained.return_value = fake_pipe
        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_diffusion_pipeline_cls, fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls, "0.0.0"),
        ), patch("abstractvision.backends.huggingface_diffusers._lazy_import_torch", return_value=_FakeTorch):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="auto")
            )
            asset = backend.generate_image(ImageGenerationRequest(prompt="hello", seed=None))

        self.assertEqual(asset.mime_type, "image/png")

        # Pipeline load args: auto variant should try fp16.
        _, kwargs = fake_t2i_cls.from_pretrained.call_args
        self.assertEqual(kwargs.get("variant"), "fp16")

        # Pipeline moved to cuda (preferred over mps).
        self.assertIn("cuda", fake_pipe.to_calls)

    def test_auto_fp16_variant_falls_back_when_missing(self):
        from abstractvision.backends.huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend
        from abstractvision.types import ImageGenerationRequest

        out_img_bytes = _png_bytes()
        from PIL import Image

        fake_image = Image.open(io.BytesIO(out_img_bytes))
        fake_pipe = _FakePipeline(fake_image)

        class _FakeCuda:
            @staticmethod
            def is_available():
                return True

        class _FakeBackends:
            mps = None

        class _FakeTorch:
            cuda = _FakeCuda()
            backends = _FakeBackends()
            float16 = object()
            float32 = object()
            bfloat16 = object()

        def _from_pretrained(*_args, **kwargs):
            if kwargs.get("variant") == "fp16":
                raise OSError("diffusion_pytorch_model.fp16.safetensors not found")
            return fake_pipe

        fake_diffusion_pipeline_cls = MagicMock()
        fake_t2i_cls = MagicMock()
        fake_t2i_cls.from_pretrained.side_effect = _from_pretrained
        fake_i2i_cls = MagicMock()
        fake_inpaint_cls = MagicMock()

        with patch(
            "abstractvision.backends.huggingface_diffusers._lazy_import_diffusers",
            return_value=(fake_diffusion_pipeline_cls, fake_t2i_cls, fake_i2i_cls, fake_inpaint_cls, "0.0.0"),
        ), patch("abstractvision.backends.huggingface_diffusers._lazy_import_torch", return_value=_FakeTorch):
            backend = HuggingFaceDiffusersVisionBackend(
                config=HuggingFaceDiffusersBackendConfig(model_id="some/model", device="auto")
            )
            asset = backend.generate_image(ImageGenerationRequest(prompt="hello", seed=None))

        self.assertEqual(asset.mime_type, "image/png")
        # First attempt with variant, then fallback without.
        self.assertEqual(fake_t2i_cls.from_pretrained.call_count, 2)
        first_kwargs = fake_t2i_cls.from_pretrained.call_args_list[0].kwargs
        second_kwargs = fake_t2i_cls.from_pretrained.call_args_list[1].kwargs
        self.assertEqual(first_kwargs.get("variant"), "fp16")
        self.assertIsNone(second_kwargs.get("variant"))


if __name__ == "__main__":
    unittest.main()
