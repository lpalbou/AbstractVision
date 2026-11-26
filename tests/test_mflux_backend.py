from __future__ import annotations

import base64
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from _mlx_gen_versions import real_runtime_versions, setUpModule, tearDownModule  # noqa: F401

try:
    import yaml  # noqa: F401

    _YAML_INSTALLED = True
except ImportError:
    _YAML_INSTALLED = False

try:
    import packaging.specifiers  # noqa: F401

    _PACKAGING_INSTALLED = True
except ImportError:
    _PACKAGING_INSTALLED = False

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X9+QAAAABJRU5ErkJggg=="
)


def _solid_png(width, height, color):
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


class _FakeImage:
    def save(self, fp, format=None):  # noqa: A002
        fp.write(PNG_1X1)


class _Generated:
    image = _FakeImage()


class _GeneratedVideo:
    metadata = {
        "task": "text-to-video",
        "frames": 5,
        "fps": 12,
    }

    def save(self, path=None, export_json_metadata=False, overwrite=True):  # noqa: ARG002
        Path(path).write_bytes(b"\x00\x00\x00\x18ftypmp42")

    def _get_metadata(self):
        return dict(self.metadata)


class _FakeProgressEvent:
    def __init__(self, phase, frame, total_frames, step, total_steps, task=None):
        self.phase = phase
        self.frame = frame
        self.total_frames = total_frames
        self.step = step
        self.total_steps = total_steps
        self.task = task
        self.timestep = step

    @property
    def progress(self):
        if self.total_steps:
            return self.step / self.total_steps
        if self.total_frames:
            return self.frame / self.total_frames
        return 0.0

    @property
    def frame_progress(self):
        if self.frame is None or not self.total_frames:
            return None
        return self.frame / self.total_frames


class _FakeCallbackRegistry:
    def __init__(self):
        self.subscriptions = []

    def subscribe_progress(self, callback, *, task=None):
        subscription = (callback, task)
        self.subscriptions.append(subscription)

        def unsubscribe():
            if subscription in self.subscriptions:
                self.subscriptions.remove(subscription)

        return unsubscribe

    def emit(self, event):
        for callback, task in list(self.subscriptions):
            if task is None or task == event.task:
                callback(event)


class _FakeModelConfig:
    @staticmethod
    def from_name(model_name, base_model=None):
        if model_name == "wan2.2-ti2v-5b":
            raise AssertionError("Wan must use the explicit config factory, not short-name inference")
        configs = {
            "flux2-klein-4b": "flux2-4b-config",
            "black-forest-labs/flux.2-klein-4b": "flux2-4b-config",
            "flux2-klein-9b": "flux2-9b-config",
            "black-forest-labs/flux.2-klein-9b": "flux2-9b-config",
            "flux2-klein-base-4b": "flux2-base-4b-config",
            "black-forest-labs/flux.2-klein-base-4b": "flux2-base-4b-config",
            "flux2-klein-base-9b": "flux2-base-9b-config",
            "black-forest-labs/flux.2-klein-base-9b": "flux2-base-9b-config",
            "bonsai-image-ternary": "bonsai-image-ternary-config",
            "prism-ml/bonsai-image-ternary-4b-mlx-2bit": "bonsai-image-ternary-config",
            "z-image": "z-image-config",
            "tongyi-mai/z-image": "z-image-config",
            "z-image-turbo": "z-image-turbo-config",
            "tongyi-mai/z-image-turbo": "z-image-turbo-config",
            "qwen-image": "qwen-image-config",
            "qwen/qwen-image": "qwen-image-config",
            "qwen/qwen-image-2512": "qwen-image-config",
            "qwen-image-edit": "qwen-image-edit-config",
            "qwen-image-edit-2511": "qwen-image-edit-2511-config",
            "qwen/qwen-image-edit-2511": "qwen-image-edit-2511-config",
            "qwen-image-edit-2509": "qwen-image-edit-2509-config",
            "qwen/qwen-image-edit-2509": "qwen-image-edit-2509-config",
            "ernie-image-turbo": "ernie-image-turbo-config",
            "baidu/ernie-image-turbo": "ernie-image-turbo-config",
            "fibo": "fibo-config",
            "briaai/fibo": "fibo-config",
            "fibo-lite": "fibo-lite-config",
            "briaai/fibo-lite": "fibo-lite-config",
            "fibo-edit": "fibo-edit-config",
            "briaai/fibo-edit": "fibo-edit-config",
            "fibo-edit-rmbg": "fibo-edit-rmbg-config",
            "briaai/fibo-edit-rmbg": "fibo-edit-rmbg-config",
            "wan2.2-ti2v-5b": "wan-ti2v-config",
            "wan-ai/wan2.2-ti2v-5b-diffusers": "wan-ti2v-config",
            "wan2.2-t2v-a14b": "wan-t2v-a14b-config",
            "wan-ai/wan2.2-t2v-a14b-diffusers": "wan-t2v-a14b-config",
            "wan2.2-i2v-a14b": "wan-i2v-a14b-config",
            "wan-ai/wan2.2-i2v-a14b-diffusers": "wan-i2v-a14b-config",
            "bytedance-seed/seedvr2-3b": "seedvr2-3b-config",
            "bytedance-seed/seedvr2-7b": "seedvr2-7b-config",
        }
        return configs.get(str(model_name).lower(), str(model_name))

    @staticmethod
    def flux2_klein_4b():
        return "flux2-4b-config"

    @staticmethod
    def flux2_klein_9b():
        return "flux2-9b-config"

    @staticmethod
    def flux2_klein_base_4b():
        return "flux2-base-4b-config"

    @staticmethod
    def flux2_klein_base_9b():
        return "flux2-base-9b-config"

    @staticmethod
    def bonsai_image_ternary():
        return "bonsai-image-ternary-config"

    @staticmethod
    def z_image_turbo():
        return "z-image-turbo-config"

    @staticmethod
    def z_image():
        return "z-image-config"

    @staticmethod
    def qwen_image():
        return "qwen-image-config"

    @staticmethod
    def qwen_image_edit():
        return "qwen-image-edit-config"

    @staticmethod
    def ernie_image_turbo():
        return "ernie-image-turbo-config"

    @staticmethod
    def fibo():
        return "fibo-config"

    @staticmethod
    def fibo_lite():
        return "fibo-lite-config"

    @staticmethod
    def fibo_edit():
        return "fibo-edit-config"

    @staticmethod
    def fibo_edit_rmbg():
        return "fibo-edit-rmbg-config"

    @staticmethod
    def wan2_2_ti2v_5b():
        return "wan-ti2v-config"

    @staticmethod
    def wan2_2_t2v_a14b():
        return "wan-t2v-a14b-config"

    @staticmethod
    def wan2_2_i2v_a14b():
        return "wan-i2v-a14b-config"

    @staticmethod
    def seedvr2_3b():
        return "seedvr2-3b-config"

    @staticmethod
    def seedvr2_7b():
        return "seedvr2-7b-config"


class _FakeModelConfigWithoutWanFactory:
    last_from_name = None

    @staticmethod
    def from_name(model_name, base_model=None):
        _FakeModelConfigWithoutWanFactory.last_from_name = (model_name, base_model)
        return "wan-ti2v-config"


class _FakeModelConfigWithoutWanSupport:
    @staticmethod
    def from_name(model_name, base_model=None):
        raise RuntimeError(f"Cannot infer base_model from {model_name}")


class _FakeFlux2:
    last_init = None
    last_generate = None

    def __init__(self, **kwargs):
        _FakeFlux2.last_init = dict(kwargs)
        self.callbacks = _FakeCallbackRegistry()

    def generate_image(self, **kwargs):
        _FakeFlux2.last_generate = dict(kwargs)
        self.callbacks.emit(_FakeProgressEvent("start", None, None, 0, 4, task="text-to-image"))
        self.callbacks.emit(_FakeProgressEvent("denoise", None, None, 2, 4, task="text-to-image"))
        self.callbacks.emit(_FakeProgressEvent("complete", None, None, 4, 4, task="text-to-image"))
        return _Generated()


class _FakeFlux2Edit:
    last_init = None
    last_generate = None

    def __init__(self, **kwargs):
        _FakeFlux2Edit.last_init = dict(kwargs)
        self.callbacks = _FakeCallbackRegistry()

    def generate_image(self, **kwargs):
        _FakeFlux2Edit.last_generate = dict(kwargs)
        self.callbacks.emit(_FakeProgressEvent("start", None, None, 0, 4, task="image-to-image"))
        self.callbacks.emit(_FakeProgressEvent("denoise", None, None, 2, 4, task="image-to-image"))
        self.callbacks.emit(_FakeProgressEvent("complete", None, None, 4, 4, task="image-to-image"))
        return _Generated()


class _FakeZImage:
    last_init = None
    last_generate = None

    def __init__(self, **kwargs):
        _FakeZImage.last_init = dict(kwargs)

    def generate_image(self, **kwargs):
        _FakeZImage.last_generate = dict(kwargs)
        return _FakeImage()


class _FakeBonsai:
    last_init = None
    last_generate = None

    def __init__(self, **kwargs):
        _FakeBonsai.last_init = dict(kwargs)

    def generate_image(self, **kwargs):
        _FakeBonsai.last_generate = dict(kwargs)
        return _Generated()


class _FakeQwenImage:
    last_init = None
    last_generate = None

    def __init__(self, **kwargs):
        _FakeQwenImage.last_init = dict(kwargs)

    def generate_image(self, **kwargs):
        _FakeQwenImage.last_generate = dict(kwargs)
        return _Generated()


class _FakeQwenImageEdit:
    last_init = None
    last_generate = None

    def __init__(self, **kwargs):
        _FakeQwenImageEdit.last_init = dict(kwargs)

    def generate_image(self, **kwargs):
        _FakeQwenImageEdit.last_generate = dict(kwargs)
        return _Generated()


class _FakeErnieImageTurbo:
    last_init = None
    last_generate = None

    def __init__(self, **kwargs):
        _FakeErnieImageTurbo.last_init = dict(kwargs)

    def generate_image(self, **kwargs):
        _FakeErnieImageTurbo.last_generate = dict(kwargs)
        return _Generated()


class _FakeFIBO:
    last_init = None
    last_generate = None

    def __init__(self, **kwargs):
        _FakeFIBO.last_init = dict(kwargs)

    def generate_image(self, **kwargs):
        _FakeFIBO.last_generate = dict(kwargs)
        return _Generated()


class _FakeFIBOEdit:
    last_init = None
    last_generate = None

    def __init__(self, **kwargs):
        _FakeFIBOEdit.last_init = dict(kwargs)

    def generate_image(self, **kwargs):
        _FakeFIBOEdit.last_generate = dict(kwargs)
        return _Generated()


class _FakeWanVae:
    """Stands in for mlx-gen's Wan VAE: only the decode entry point the backend reroutes."""

    patch_size = 2

    def iter_decode_normalized_latent_slices(self, latents, *, clear_cache_each_slice=False, tile_spatial=False):
        return iter(())


class _FakeWan:
    last_init = None
    last_generate = None
    last_image_path_existed = None
    last_image_size = None
    last_image_corner_pixel = None
    last_image_center_pixel = None
    # Whether the VAE decode entry point was the backend's tiled decode while
    # generate_video ran (mlx-gen resolves it on the instance at decode time).
    last_decode_tiled = None

    def __init__(self, **kwargs):
        _FakeWan.last_init = dict(kwargs)
        self.vae = _FakeWanVae()

    def generate_video(self, **kwargs):
        _FakeWan.last_generate = dict(kwargs)
        _FakeWan.last_decode_tiled = "iter_decode_normalized_latent_slices" in vars(self.vae)
        image_path = kwargs.get("image_path")
        _FakeWan.last_image_path_existed = (
            Path(image_path).exists() if image_path is not None else None
        )
        _FakeWan.last_image_size = None
        _FakeWan.last_image_corner_pixel = None
        _FakeWan.last_image_center_pixel = None
        if image_path is not None and _FakeWan.last_image_path_existed:
            from PIL import Image

            with Image.open(image_path) as img:
                rgb = img.convert("RGB")
                _FakeWan.last_image_size = rgb.size
                _FakeWan.last_image_corner_pixel = rgb.getpixel((0, 0))
                _FakeWan.last_image_center_pixel = rgb.getpixel(
                    (rgb.width // 2, rgb.height // 2)
                )
        progress_callback = kwargs.get("progress_callback")
        if callable(progress_callback):
            progress_callback(_FakeProgressEvent("start", 0, 5, 0, 2))
            progress_callback(_FakeProgressEvent("denoise", 3, 5, 1, 2))
            progress_callback(_FakeProgressEvent("complete", 5, 5, 2, 2))
        return _GeneratedVideo()


class _FakeScaleFactor:
    last_parsed = None

    def __init__(self, value):
        self.value = value

    @classmethod
    def parse(cls, value):
        cls.last_parsed = value
        return cls(value)

    def __str__(self):
        return self.value


class _FakeTilingConfig:
    pass


class _FakeSeedVR2:
    last_init = None
    last_generate = None
    last_image_path_existed = None
    last_instance = None

    def __init__(self, **kwargs):
        _FakeSeedVR2.last_init = dict(kwargs)
        _FakeSeedVR2.last_instance = self
        self.callbacks = _FakeCallbackRegistry()
        self.tiling_config = "unset"

    def generate_image(self, *, seed, image_path, resolution, softness):
        _FakeSeedVR2.last_generate = {
            "seed": seed,
            "image_path": image_path,
            "resolution": resolution,
            "softness": softness,
        }
        _FakeSeedVR2.last_image_path_existed = Path(image_path).exists()
        self.callbacks.emit(_FakeProgressEvent("start", None, None, 0, 1, task="text-to-image"))
        self.callbacks.emit(_FakeProgressEvent("denoise", None, None, 1, 1, task="text-to-image"))
        self.callbacks.emit(_FakeProgressEvent("complete", None, None, 1, 1, task="text-to-image"))
        return _Generated()


class _FakeDownloadRequiredError(FileNotFoundError):
    download_command = "mlxgen download --model AbstractFramework/flux.2-klein-4b-4bit"
    prepare_command = "mlxgen prepare --model AbstractFramework/flux.2-klein-4b-4bit --path ./models/flux2-klein-4b -q 4"


class _ThreadAwareFlux2:
    init_thread = None
    generate_thread = None

    def __init__(self, **kwargs):
        _ThreadAwareFlux2.init_thread = threading.get_ident()

    def generate_image(self, **kwargs):
        _ThreadAwareFlux2.generate_thread = threading.get_ident()
        return _Generated()


class _CountingFlux2:
    init_calls = 0
    generate_calls = []

    def __init__(self, **kwargs):
        _CountingFlux2.init_calls += 1

    def generate_image(self, **kwargs):
        _CountingFlux2.generate_calls.append(dict(kwargs))
        return _Generated()


class _RegisteringCallbackRegistry(_FakeCallbackRegistry):
    def __init__(self):
        super().__init__()
        self.registered = []

    def register(self, callback):
        self.registered.append(callback)


# The MLX cache limit in force before a run (mlx-gen's RAM/8 default on a 64 GiB or larger Mac).
_CAP_IN_FORCE = 8 * 1024**3
_MEMORY_EVENTS = []


class _LowRamFlux2(_FakeFlux2):
    instances = []
    during_generate = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.callbacks = _RegisteringCallbackRegistry()
        self.tiling_config = None
        _LowRamFlux2.instances.append(self)
        _MEMORY_EVENTS.append(("init", None))

    def generate_image(self, **kwargs):
        if _LowRamFlux2.during_generate is not None:
            _LowRamFlux2.during_generate()
        return super().generate_image(**kwargs)


class _LowRamSeedVR2(_FakeSeedVR2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.callbacks = _RegisteringCallbackRegistry()
        _MEMORY_EVENTS.append(("init", None))


class _FakeMemorySaver:
    """Stands in for mlx-gen's MemorySaver (mflux.callbacks.instances.memory_saver)."""

    instances = []

    def __init__(self, model, keep_transformer=True, cache_limit_bytes=1000**3, args=None):
        self.model = model
        self.keep_transformer = keep_transformer
        self.cache_limit_bytes = cache_limit_bytes
        model.tiling_config = "low-ram-tiling"
        _FakeMemorySaver.instances.append(self)


def _route_capabilities(*tasks, block_streaming=False):
    return SimpleNamespace(
        capabilities=tuple(
            SimpleNamespace(public_task=task, supports_block_streaming=block_streaming)
            for task in tasks
        )
    )


class TestMFluxVisionBackend(unittest.TestCase):
    def _lazy_import_return(self, flux_cls=_FakeFlux2, flux_edit_cls=_FakeFlux2Edit, z_cls=_FakeZImage):
        return (_FakeModelConfig, _FakeDownloadRequiredError, flux_cls, flux_edit_cls, z_cls, z_cls)

    def _make_model_dir(self, root: Path, name: str) -> Path:
        model_dir = root / name
        (model_dir / "transformer").mkdir(parents=True)
        (model_dir / "transformer" / "0.safetensors").write_bytes(b"x")
        return model_dir

    def _make_cache_snapshot(self, root: Path, repo_id: str, snapshot_name: str = "abc123") -> Path:
        repo_dir = root / f"models--{repo_id.replace('/', '--')}"
        snap = repo_dir / "snapshots" / snapshot_name
        (snap / "transformer").mkdir(parents=True, exist_ok=True)
        (snap / "transformer" / "0.safetensors").write_bytes(b"x")
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / "main").write_text(snapshot_name, encoding="utf-8")
        return snap

    def _make_adapter_snapshot(
        self,
        root: Path,
        repo_id: str,
        relative_path: str,
        *,
        base_model: str | tuple[str, ...],
        snapshot_name: str = "abc123",
    ) -> Path:
        repo_dir = root / f"models--{repo_id.replace('/', '--')}"
        snap = repo_dir / "snapshots" / snapshot_name
        adapter_path = snap / relative_path
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_bytes(b"adapter")
        base_models = (base_model,) if isinstance(base_model, str) else tuple(base_model)
        frontmatter = ["---"]
        if len(base_models) == 1:
            frontmatter.append(f"base_model: {base_models[0]}")
        else:
            frontmatter.append("base_model:")
            frontmatter.extend([f"  - {value}" for value in base_models])
        frontmatter.extend(["---", "", "# adapter"])
        (snap / "README.md").write_text("\n".join(frontmatter), encoding="utf-8")
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / "main").write_text(snapshot_name, encoding="utf-8")
        return snap

    def test_generate_image_uses_cached_mflux_preset(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            cache_root = Path(cache_td)
            snapshot = self._make_cache_snapshot(
                cache_root, "AbstractFramework/flux.2-klein-4b-8bit"
            )
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-8bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    asset = backend.generate_image(
                        ImageGenerationRequest(
                            prompt="hello",
                            width=64,
                            height=32,
                            steps=4,
                            guidance_scale=1.0,
                            seed=123,
                        )
                    )

        self.assertEqual(asset.media_type, "image")
        self.assertEqual(asset.mime_type, "image/png")
        self.assertTrue(asset.data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(_FakeFlux2.last_init["model_config"], "flux2-4b-config")
        self.assertEqual(_FakeFlux2.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeFlux2.last_generate["prompt"], "hello")
        self.assertEqual(_FakeFlux2.last_generate["width"], 64)
        self.assertEqual(_FakeFlux2.last_generate["height"], 32)
        self.assertEqual(_FakeFlux2.last_generate["num_inference_steps"], 4)
        self.assertEqual(_FakeFlux2.last_generate["guidance"], 1.0)

    def test_generate_image_emits_mlx_gen_progress_events(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest, VideoProgressEvent

        seen = []
        step_seen = []

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-4bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-4bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    backend.generate_image_with_progress(
                        ImageGenerationRequest(
                            prompt="hello",
                            steps=4,
                            extra={"on_progress": seen.append},
                        ),
                        progress_callback=lambda current, total: step_seen.append((current, total)),
                    )

        self.assertEqual([event.phase for event in seen], ["start", "denoise", "complete"])
        self.assertTrue(all(isinstance(event, VideoProgressEvent) for event in seen))
        self.assertEqual([event.task for event in seen], ["text_to_image"] * 3)
        self.assertEqual(seen[-1].step_progress, 1.0)
        self.assertEqual(step_seen, [(0, 4), (2, 4), (4, 4)])

    def test_seedvr2_upscale_routes_cached_model_and_progress(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageUpscaleRequest, VideoProgressEvent

        seen = []
        step_seen = []

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "AbstractFramework/seedvr2-3b-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="seedvr2-3b"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux_seedvr2",
                    return_value=(
                        _FakeModelConfig,
                        _FakeSeedVR2,
                        _FakeScaleFactor,
                        _FakeTilingConfig,
                    ),
                ):
                    asset = backend.upscale_image_with_progress(
                        ImageUpscaleRequest(
                            image=PNG_1X1,
                            scale=2,
                            seed=42,
                            softness=0.25,
                            vae_tiling=True,
                            extra={"on_progress": seen.append},
                        ),
                        progress_callback=lambda current, total: step_seen.append((current, total)),
                    )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(asset.media_type, "image")
        self.assertEqual(asset.metadata["task"], "image_upscale")
        self.assertEqual(asset.metadata["base_model"], "seedvr2-3b")
        self.assertEqual(asset.metadata["quantization_bits"], 8)
        self.assertEqual(asset.metadata["seed"], 42)
        self.assertEqual(asset.metadata["resolution"], "2x")
        self.assertEqual(asset.metadata["scale"], 2)
        self.assertEqual(asset.metadata["softness"], 0.25)
        self.assertTrue(asset.metadata["vae_tiling"])
        self.assertEqual(_FakeSeedVR2.last_init["model_config"], "seedvr2-3b-config")
        self.assertEqual(_FakeSeedVR2.last_init["model_path"], str(snapshot))
        # A prepared package loads as stored; its stored width is recorded.
        self.assertIsNone(_FakeSeedVR2.last_init["quantize"])
        self.assertEqual(_FakeSeedVR2.last_generate["seed"], 42)
        self.assertTrue(_FakeSeedVR2.last_image_path_existed)
        self.assertEqual(str(_FakeSeedVR2.last_generate["resolution"]), "2x")
        self.assertEqual(_FakeScaleFactor.last_parsed, "2x")
        self.assertEqual(_FakeSeedVR2.last_generate["softness"], 0.25)
        self.assertIsInstance(_FakeSeedVR2.last_instance.tiling_config, _FakeTilingConfig)
        self.assertEqual([event.phase for event in seen], ["start", "denoise", "complete"])
        self.assertTrue(all(isinstance(event, VideoProgressEvent) for event in seen))
        self.assertEqual([event.task for event in seen], ["image_upscale"] * 3)
        self.assertEqual(seen[-1].step_progress, 1.0)
        self.assertEqual(step_seen, [(0, 1), (1, 1), (1, 1)])

    def test_seedvr2_exact_canonical_package_ids_resolve_to_runtime_base_models(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageUpscaleRequest

        cases = [
            ("AbstractFramework/seedvr2-3b-8bit", "seedvr2-3b", 8, "seedvr2-3b-config"),
            ("AbstractFramework/seedvr2-3b-4bit", "seedvr2-3b", 4, "seedvr2-3b-config"),
            ("AbstractFramework/seedvr2-7b-8bit", "seedvr2-7b", 8, "seedvr2-7b-config"),
            ("AbstractFramework/seedvr2-7b-4bit", "seedvr2-7b", 4, "seedvr2-7b-config"),
        ]

        with tempfile.TemporaryDirectory() as cache_td:
            snapshots = {
                repo_id: self._make_cache_snapshot(Path(cache_td), repo_id)
                for repo_id, _base, _bits, _config in cases
            }

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux_seedvr2",
                    return_value=(
                        _FakeModelConfig,
                        _FakeSeedVR2,
                        _FakeScaleFactor,
                        _FakeTilingConfig,
                    ),
                ):
                    for repo_id, base_model, bits, config_name in cases:
                        with self.subTest(repo_id=repo_id):
                            backend = MFluxVisionBackend(
                                config=MFluxBackendConfig(model=repo_id)
                            )
                            asset = backend.upscale_image(
                                ImageUpscaleRequest(image=PNG_1X1, scale=2, seed=42)
                            )

                            self.assertTrue(asset.data.startswith(b"\x89PNG"))
                            self.assertEqual(asset.metadata["base_model"], base_model)
                            self.assertEqual(asset.metadata["quantization_bits"], bits)
                            self.assertEqual(asset.metadata["resolution"], "2x")
                            self.assertEqual(asset.metadata["softness"], 0.25)
                            self.assertEqual(_FakeSeedVR2.last_init["model_config"], config_name)
                            self.assertEqual(_FakeSeedVR2.last_init["model_path"], str(snapshots[repo_id]))
                            self.assertIsNone(_FakeSeedVR2.last_init["quantize"])

    def test_seedvr2_preload_uses_upscale_model_without_generation_warmup(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        _FakeSeedVR2.last_generate = None

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/seedvr2-7b-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="seedvr2-7b"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux_seedvr2",
                    return_value=(
                        _FakeModelConfig,
                        _FakeSeedVR2,
                        _FakeScaleFactor,
                        _FakeTilingConfig,
                    ),
                ):
                    backend.preload()

        self.assertEqual(_FakeSeedVR2.last_init["model_config"], "seedvr2-7b-config")
        self.assertIsNone(_FakeSeedVR2.last_init["quantize"])
        self.assertIsNone(_FakeSeedVR2.last_generate)

    def test_seedvr2_local_prepared_folder_infers_7b_base_model(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as td:
            model_dir = self._make_model_dir(Path(td), "seedvr2-7b-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model=str(model_dir)))

            caps = backend.get_capabilities()

        self.assertEqual(caps.supported_tasks, ["image_upscale"])

    def test_seedvr2_provider_catalog_surfaces_local_q8_and_q4_folders(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as model_td, tempfile.TemporaryDirectory() as cache_td:
            root = Path(model_td)
            self._make_model_dir(root, "seedvr2-7b-8bit")
            self._make_model_dir(root, "seedvr2-7b-4bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model_dir=model_td, cache_dir=cache_td)
            )

            models = backend.list_provider_models(task="image_upscale")

        by_id = {str(info.id): info for info in models}
        self.assertIn("AbstractFramework/seedvr2-7b-8bit", by_id)
        self.assertIn("AbstractFramework/seedvr2-7b-4bit", by_id)
        self.assertEqual(by_id["AbstractFramework/seedvr2-7b-8bit"].raw["quantization_bits"], 8)
        self.assertEqual(by_id["AbstractFramework/seedvr2-7b-4bit"].raw["quantization_bits"], 4)

    def test_generate_image_preserves_explicit_mlx_gen_q8_variant(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            cache_root = Path(cache_td)
            q4_snapshot = self._make_cache_snapshot(
                cache_root, "AbstractFramework/flux.2-klein-9b-4bit"
            )
            q8_snapshot = self._make_cache_snapshot(
                cache_root, "AbstractFramework/flux.2-klein-9b-8bit"
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    q4_backend = MFluxVisionBackend(
                        config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-9b-4bit")
                    )
                    q4_backend.generate_image(ImageGenerationRequest(prompt="q4", steps=2, seed=1))
                    self.assertEqual(_FakeFlux2.last_init["model_path"], str(q4_snapshot))

                    q8_backend = MFluxVisionBackend(
                        config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-9b-8bit")
                    )
                    q8_backend.generate_image(ImageGenerationRequest(prompt="q8", steps=2, seed=2))
                    self.assertEqual(_FakeFlux2.last_init["model_path"], str(q8_snapshot))

    def test_generate_image_uses_exact_qwen_2512_q8_repo_id(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest, LoRAAdapterSpec

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/qwen-image-2512-4bit")
            q8_snapshot = self._make_cache_snapshot(
                Path(cache_td), "AbstractFramework/qwen-image-2512-8bit"
            )
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/qwen-image-2512-8bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_qwen",
                        return_value=(_FakeQwenImage, _FakeQwenImageEdit),
                    ):
                        asset = backend.generate_image(
                            ImageGenerationRequest(prompt="q8", steps=2, seed=2)
                        )
                        backend.generate_image(
                            ImageGenerationRequest(
                                prompt="q8-lora",
                                steps=2,
                                seed=3,
                                lora_adapters=(
                                    LoRAAdapterSpec(
                                        source="owner/style-lora:adapter.safetensors",
                                        scale=0.75,
                                    ),
                                ),
                            )
                        )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeQwenImage.last_init["model_path"], str(q8_snapshot))
        self.assertNotIn("quantize", _FakeQwenImage.last_init)
        self.assertEqual(asset.metadata["quantization_bits"], 8)
        self.assertEqual(
            _FakeQwenImage.last_init["lora_paths"],
            ["owner/style-lora:adapter.safetensors"],
        )
        self.assertEqual(_FakeQwenImage.last_init["lora_scales"], [0.75])

    def test_generate_image_resolves_cached_bare_hf_lora_repo_to_local_file(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest, LoRAAdapterSpec

        with tempfile.TemporaryDirectory() as cache_td:
            cache_root = Path(cache_td)
            snapshot = self._make_cache_snapshot(
                cache_root, "AbstractFramework/flux.2-klein-9b-8bit"
            )
            adapter_snapshot = self._make_adapter_snapshot(
                cache_root,
                "example-org/Flux_Klein_9B_Style",
                "Flux Klein - Style v2.safetensors",
                base_model="black-forest-labs/FLUX.2-klein-base-9B",
            )
            expected_adapter = adapter_snapshot / "Flux Klein - Style v2.safetensors"
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-9b-8bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    backend.generate_image(
                        ImageGenerationRequest(
                            prompt="with-lora",
                            steps=2,
                            seed=3,
                            lora_adapters=(
                                LoRAAdapterSpec(
                                    source="example-org/Flux_Klein_9B_Style",
                                    scale=0.75,
                                ),
                            ),
                        )
                    )

        self.assertEqual(_FakeFlux2.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeFlux2.last_init["lora_paths"], [str(expected_adapter)])
        self.assertEqual(_FakeFlux2.last_init["lora_scales"], [0.75])

    def test_generate_image_resolves_cached_hf_lora_handle_to_local_file(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest, LoRAAdapterSpec

        with tempfile.TemporaryDirectory() as cache_td:
            cache_root = Path(cache_td)
            snapshot = self._make_cache_snapshot(
                cache_root, "AbstractFramework/flux.2-klein-9b-8bit"
            )
            adapter_snapshot = self._make_adapter_snapshot(
                cache_root,
                "example-org/Flux_Klein_9B_Style",
                "Flux Klein - Style v2.safetensors",
                base_model="black-forest-labs/FLUX.2-klein-base-9B",
            )
            expected_adapter = adapter_snapshot / "Flux Klein - Style v2.safetensors"
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-9b-8bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    backend.generate_image(
                        ImageGenerationRequest(
                            prompt="with-lora-handle",
                            steps=2,
                            seed=4,
                            lora_adapters=(
                                LoRAAdapterSpec(
                                    source="example-org/Flux_Klein_9B_Style:Flux Klein - Style v2.safetensors",
                                    scale=0.5,
                                ),
                            ),
                        )
                    )

        self.assertEqual(_FakeFlux2.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeFlux2.last_init["lora_paths"], [str(expected_adapter)])
        self.assertEqual(_FakeFlux2.last_init["lora_scales"], [0.5])

    def test_generate_image_rejects_ambiguous_bare_hf_lora_repo(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import OptionalDependencyMissingError
        from abstractvision.types import ImageGenerationRequest, LoRAAdapterSpec

        with tempfile.TemporaryDirectory() as cache_td:
            cache_root = Path(cache_td)
            self._make_cache_snapshot(cache_root, "AbstractFramework/flux.2-klein-9b-8bit")
            self._make_adapter_snapshot(
                cache_root,
                "owner/multi-lora",
                "style-a.safetensors",
                base_model="black-forest-labs/FLUX.2-klein-base-9B",
            )
            self._make_adapter_snapshot(
                cache_root,
                "owner/multi-lora",
                "nested/style-b.safetensors",
                base_model="black-forest-labs/FLUX.2-klein-base-9B",
                snapshot_name="abc123",
            )
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-9b-8bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with self.assertRaisesRegex(
                        OptionalDependencyMissingError, "Select one explicit adapter handle"
                    ) as ctx:
                        backend.generate_image(
                            ImageGenerationRequest(
                                prompt="ambiguous-lora",
                                steps=2,
                                seed=5,
                                lora_adapters=(LoRAAdapterSpec(source="owner/multi-lora"),),
                            )
                        )

        self.assertIn("owner/multi-lora:style-a.safetensors", str(ctx.exception))
        self.assertIn("owner/multi-lora:nested/style-b.safetensors", str(ctx.exception))

    def test_generic_qwen_2512_selector_is_rejected_when_q4_and_q8_exist(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import OptionalDependencyMissingError
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/qwen-image-2512-4bit")
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/qwen-image-2512-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="qwen-image-2512"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with self.assertRaisesRegex(OptionalDependencyMissingError, "ambiguous"):
                    backend.generate_image(ImageGenerationRequest(prompt="q8", steps=2, seed=2))

    def test_explicit_mlx_gen_q8_missing_does_not_fall_back_to_q4(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import OptionalDependencyMissingError
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-9b-4bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-9b-8bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with self.assertRaisesRegex(OptionalDependencyMissingError, "flux.2-klein-9b-8bit"):
                    backend.generate_image(ImageGenerationRequest(prompt="q8", steps=2, seed=2))

    def test_image_to_image_catalog_prefers_dedicated_edit_models(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td:
            root = Path(cache_td)
            self._make_cache_snapshot(root, "AbstractFramework/flux.2-klein-4b-4bit")
            self._make_cache_snapshot(root, "AbstractFramework/qwen-image-edit-2511-8bit")
            self._make_cache_snapshot(root, "AbstractFramework/qwen-image-edit-2511-4bit")
            self._make_cache_snapshot(root, "AbstractFramework/ernie-image-turbo-4bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig())

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                models = backend.list_provider_models(task="image_to_image")

        self.assertGreaterEqual(len(models), 4)
        ids = [m.id for m in models]
        self.assertEqual(ids[0], "AbstractFramework/qwen-image-edit-2511-4bit")
        self.assertEqual(ids[1], "AbstractFramework/qwen-image-edit-2511-8bit")
        self.assertLess(
            ids.index("AbstractFramework/qwen-image-edit-2511-4bit"),
            ids.index("AbstractFramework/flux.2-klein-4b-4bit"),
        )
        self.assertEqual(models[0].raw["catalog_rank"], 0)
        flux_model = models[ids.index("AbstractFramework/flux.2-klein-4b-4bit")]
        self.assertNotIn(
            "guidance_scale",
            flux_model.raw.get("parameter_constraints", {}),
        )

    def test_bonsai_ternary_routes_text_to_image(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(
                Path(cache_td), "prism-ml/bonsai-image-ternary-4B-mlx-2bit"
            )
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(
                    model="prism-ml/bonsai-image-ternary-4B-mlx-2bit"
                )
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_bonsai",
                        return_value=_FakeBonsai,
                    ):
                        asset = backend.generate_image(
                            ImageGenerationRequest(
                                prompt="a small bonsai in a ceramic studio",
                                negative_prompt="blur",
                                width=128,
                                height=128,
                                steps=4,
                                guidance_scale=7.0,
                                seed=123,
                            )
                        )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeBonsai.last_init["model_config"], "bonsai-image-ternary-config")
        self.assertEqual(_FakeBonsai.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeBonsai.last_generate["prompt"], "a small bonsai in a ceramic studio")
        self.assertEqual(_FakeBonsai.last_generate["guidance"], 1.0)
        self.assertNotIn("negative_prompt", _FakeBonsai.last_generate)
        self.assertEqual(asset.metadata["base_model"], "bonsai-image-ternary")
        self.assertEqual(asset.metadata["quantization_bits"], 2)

    def test_z_image_turbo_drops_noop_negative_prompt_and_forces_guidance_off(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(
                Path(cache_td), "AbstractFramework/z-image-turbo-8bit"
            )
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/z-image-turbo-8bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    asset = backend.generate_image(
                        ImageGenerationRequest(prompt="hello", negative_prompt="blur", seed=7)
                    )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeZImage.last_init["model_config"], "z-image-turbo-config")
        self.assertEqual(_FakeZImage.last_init["model_path"], str(snapshot))
        self.assertNotIn("negative_prompt", _FakeZImage.last_generate)
        self.assertEqual(_FakeZImage.last_generate["num_inference_steps"], 9)
        self.assertEqual(_FakeZImage.last_generate["guidance"], 0.0)

    def test_z_image_routes_full_quality_model_and_negative_prompt(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "AbstractFramework/z-image-4bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/z-image-4bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    asset = backend.generate_image(
                        ImageGenerationRequest(prompt="hello", negative_prompt="blur", seed=7)
                    )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeZImage.last_init["model_config"], "z-image-config")
        self.assertEqual(_FakeZImage.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeZImage.last_generate["negative_prompt"], "blur")
        self.assertEqual(_FakeZImage.last_generate["num_inference_steps"], 50)
        self.assertEqual(_FakeZImage.last_generate["guidance"], 3.5)

    def test_ernie_image_turbo_routes_text_to_image(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(
                Path(cache_td), "AbstractFramework/ernie-image-turbo-4bit"
            )
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/ernie-image-turbo-4bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_ernie",
                        return_value=_FakeErnieImageTurbo,
                    ):
                        asset = backend.generate_image(
                            ImageGenerationRequest(prompt="poster", negative_prompt="blur", seed=7)
                        )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeErnieImageTurbo.last_init["model_config"], "ernie-image-turbo-config")
        self.assertEqual(_FakeErnieImageTurbo.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeErnieImageTurbo.last_generate["prompt"], "poster")
        self.assertEqual(_FakeErnieImageTurbo.last_generate["negative_prompt"], "blur")
        self.assertEqual(_FakeErnieImageTurbo.last_generate["num_inference_steps"], 8)
        self.assertEqual(_FakeErnieImageTurbo.last_generate["guidance"], 1.0)

    def test_ernie_image_turbo_routes_image_to_image(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/ernie-image-turbo-4bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/ernie-image-turbo-4bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_ernie",
                        return_value=_FakeErnieImageTurbo,
                    ):
                        asset = backend.edit_image(
                            ImageEditRequest(
                                prompt="make it watercolor",
                                image=PNG_1X1,
                                steps=4,
                                guidance_scale=1.0,
                                seed=8,
                                extra={"strength": 0.35},
                            )
                        )

        self.assertEqual(asset.media_type, "image")
        self.assertEqual(_FakeErnieImageTurbo.last_generate["prompt"], "make it watercolor")
        self.assertEqual(_FakeErnieImageTurbo.last_generate["num_inference_steps"], 4)
        self.assertEqual(_FakeErnieImageTurbo.last_generate["guidance"], 1.0)
        self.assertEqual(_FakeErnieImageTurbo.last_generate["image_strength"], 0.35)
        self.assertTrue(Path(_FakeErnieImageTurbo.last_generate["image_path"]).suffix)

    def test_fibo_routes_text_to_image_and_image_conditioning(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest, ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "briaai/FIBO")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="briaai/FIBO"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_fibo",
                        return_value=(_FakeFIBO, _FakeFIBOEdit),
                    ):
                        text_asset = backend.generate_image(
                            ImageGenerationRequest(prompt="perfume bottle", seed=11)
                        )
                        image_asset = backend.edit_image(
                            ImageEditRequest(
                                prompt="add sunlight",
                                image=PNG_1X1,
                                seed=12,
                                extra={"strength": 0.25},
                            )
                        )

        self.assertEqual(text_asset.media_type, "image")
        self.assertEqual(image_asset.media_type, "image")
        self.assertEqual(_FakeFIBO.last_init["model_config"], "fibo-config")
        self.assertEqual(_FakeFIBO.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeFIBO.last_generate["prompt"], "add sunlight")
        self.assertEqual(_FakeFIBO.last_generate["image_strength"], 0.25)

    def test_fibo_edit_routes_mask_path(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "briaai/Fibo-Edit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="briaai/Fibo-Edit"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_fibo",
                        return_value=(_FakeFIBO, _FakeFIBOEdit),
                    ):
                        asset = backend.edit_image(
                            ImageEditRequest(
                                prompt="remove background",
                                image=PNG_1X1,
                                mask=PNG_1X1,
                                seed=13,
                            )
                        )

        self.assertEqual(asset.media_type, "image")
        self.assertEqual(_FakeFIBOEdit.last_init["model_config"], "fibo-edit-config")
        self.assertEqual(_FakeFIBOEdit.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeFIBOEdit.last_generate["prompt"], "remove background")
        self.assertTrue(str(_FakeFIBOEdit.last_generate["image_path"]).endswith(".png"))
        self.assertTrue(str(_FakeFIBOEdit.last_generate["mask_path"]).endswith(".png"))

    def test_wan_routes_text_to_video(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import LoRAAdapterSpec, VideoGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        base_asset = backend.generate_video(
                            VideoGenerationRequest(
                                prompt="fox",
                                negative_prompt="blur",
                                width=832,
                                height=480,
                                fps=12,
                                num_frames=5,
                                steps=2,
                                guidance_scale=4.5,
                                seed=42,
                                extra={"max_sequence_length": 128},
                            )
                        )
                        base_generate = dict(_FakeWan.last_generate)
                        lora_asset = backend.generate_video(
                            VideoGenerationRequest(
                                prompt="fox",
                                seed=43,
                                lora_adapters=(
                                    LoRAAdapterSpec(
                                        source="owner/wan-lora:video.safetensors",
                                        scale=0.9,
                                        target_role="transformer",
                                    ),
                                ),
                            )
                        )

        self.assertEqual(base_asset.media_type, "video")
        self.assertEqual(base_asset.mime_type, "video/mp4")
        self.assertEqual(base_asset.data, b"\x00\x00\x00\x18ftypmp42")
        self.assertEqual(_FakeWan.last_init["model_config"], "wan-ti2v-config")
        self.assertEqual(_FakeWan.last_init["model_path"], str(snapshot))
        self.assertEqual(base_generate["prompt"], "fox")
        self.assertEqual(base_generate["negative_prompt"], "blur")
        self.assertEqual(base_generate["width"], 832)
        self.assertEqual(base_generate["height"], 480)
        self.assertEqual(base_generate["fps"], 12)
        self.assertEqual(base_generate["num_frames"], 5)
        self.assertEqual(base_generate["num_inference_steps"], 2)
        self.assertEqual(base_generate["guidance"], 4.5)
        self.assertEqual(base_generate["max_sequence_length"], 128)
        self.assertNotIn("image_path", base_generate)
        self.assertEqual(base_asset.metadata["base_model"], "wan2.2-ti2v-5b")
        self.assertEqual(base_asset.metadata["task"], "text_to_video")
        self.assertEqual(
            _FakeWan.last_init["lora_paths"],
            ["owner/wan-lora:video.safetensors"],
        )
        self.assertEqual(_FakeWan.last_init["lora_scales"], [0.9])
        self.assertEqual(_FakeWan.last_init["lora_target_roles"], ["transformer"])
        self.assertEqual(
            lora_asset.metadata["requested_lora_adapters"][0]["target_role"],
            "transformer",
        )

    def test_wan_ti2v_480p_defaults_flow_shift_to_three(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import VideoGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        backend.generate_video(
                            VideoGenerationRequest(
                                prompt="fox",
                                width=832,
                                height=480,
                                num_frames=9,
                                steps=4,
                                seed=42,
                            )
                        )

        self.assertEqual(_FakeWan.last_generate["width"], 832)
        self.assertEqual(_FakeWan.last_generate["height"], 480)
        self.assertEqual(_FakeWan.last_generate["flow_shift"], 3.0)

    def test_wan_ti2v_rejects_undersized_video_requests(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        with self.assertRaises(CapabilityNotSupportedError) as ctx:
                            backend.generate_video(
                                VideoGenerationRequest(
                                    prompt="fox",
                                    width=448,
                                    height=256,
                                    num_frames=9,
                                    steps=4,
                                    seed=42,
                                )
                            )

        self.assertIn("832x480", str(ctx.exception))

    def test_wan_refuses_malformed_values_and_keyframes_like_minimax_h3(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageToVideoRequest, VideoGenerationRequest

        _FakeWan.last_init = None
        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers"))
            # The keyframe cases need a model with an image_to_video route.
            i2v_backend = MFluxVisionBackend(config=MFluxBackendConfig(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers"))
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True), patch(
                "abstractvision.backends.mflux._lazy_import_mflux", return_value=self._lazy_import_return()
            ), patch("abstractvision.backends.mflux._lazy_import_mflux_wan", return_value=_FakeWan):
                for request, error, pattern in (
                    (VideoGenerationRequest(prompt="fox", width="wide", height=480), ValueError, "width"),
                    (VideoGenerationRequest(prompt="fox", width=832.5, height=480), ValueError, "width"),
                    (VideoGenerationRequest(prompt="fox", num_frames="many"), ValueError, "num_frames"),
                    (VideoGenerationRequest(prompt="fox", steps="x"), ValueError, "steps"),
                    (VideoGenerationRequest(prompt="fox", seed="abc"), ValueError, "seed"),
                    (VideoGenerationRequest(prompt="fox", width=-16, height=480), CapabilityNotSupportedError, "width"),
                    (ImageToVideoRequest(image=b"", prompt="fox"), ValueError, "image.*empty"),
                    (ImageToVideoRequest(image=b"garbage", prompt="fox"), ValueError, "image.*decode"),
                ):
                    with self.subTest(request=type(request).__name__, pattern=pattern):
                        run = backend.generate_video if isinstance(request, VideoGenerationRequest) else i2v_backend.image_to_video
                        with self.assertRaisesRegex(error, pattern):
                            run(request)
        self.assertIsNone(_FakeWan.last_init)

    def test_a_failed_preload_warmup_leaves_no_model_resident(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        class _FailingFlux2(_FakeFlux2):
            def generate_image(self, **kwargs):
                raise RuntimeError("warm-up failed")

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-8bit"))
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True), patch(
                "abstractvision.backends.mflux._lazy_import_mflux",
                return_value=self._lazy_import_return(flux_cls=_FailingFlux2),
            ):
                with self.assertRaisesRegex(RuntimeError, "warm-up failed"):
                    backend.preload()
        self.assertFalse(backend.has_loaded_model())
        self.assertIsNone(backend._model_key)

    def test_wan_ti2v_explicit_flow_shift_is_preserved(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import VideoGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        asset = backend.generate_video(
                            VideoGenerationRequest(
                                prompt="fox",
                                width=1280,
                                height=704,
                                num_frames=9,
                                steps=4,
                                seed=42,
                                flow_shift=5.5,
                            )
                        )

        self.assertEqual(_FakeWan.last_generate["flow_shift"], 5.5)
        self.assertEqual(asset.metadata["flow_shift"], 5.5)

    def test_wan_a14b_routes_text_to_video_with_model_defaults(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import VideoGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        asset = backend.generate_video(
                            VideoGenerationRequest(prompt="fox", seed=42)
                        )

        self.assertEqual(_FakeWan.last_init["model_config"], "wan-t2v-a14b-config")
        self.assertEqual(_FakeWan.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeWan.last_generate["width"], 832)
        self.assertEqual(_FakeWan.last_generate["height"], 480)
        self.assertEqual(_FakeWan.last_generate["fps"], 16)
        self.assertEqual(_FakeWan.last_generate["num_frames"], 81)
        self.assertEqual(_FakeWan.last_generate["num_inference_steps"], 40)
        self.assertEqual(_FakeWan.last_generate["guidance"], 4.0)
        self.assertEqual(_FakeWan.last_generate["guidance_2"], 3.0)
        self.assertNotIn("image_path", _FakeWan.last_generate)
        self.assertEqual(asset.metadata["base_model"], "wan2.2-t2v-a14b")
        self.assertEqual(asset.metadata["guidance_2"], 3.0)

    def _run_one_wan_video(self, repo_id, request, *, wan_cls=None):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), repo_id)
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model=repo_id))
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ), patch(
                    "abstractvision.backends.mflux._lazy_import_mflux_wan",
                    return_value=wan_cls or _FakeWan,
                ):
                    asset = backend.generate_video(request)
        return backend, asset

    def test_wan_ti2v_default_canvas_is_832x480_121_frames(self):
        from abstractvision.types import VideoGenerationRequest

        _backend, asset = self._run_one_wan_video(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers", VideoGenerationRequest(prompt="fox", seed=1)
        )

        generate = _FakeWan.last_generate
        self.assertEqual((generate["width"], generate["height"]), (832, 480))
        self.assertEqual(generate["num_frames"], 121)
        self.assertEqual(generate["fps"], 24)
        self.assertEqual(generate["num_inference_steps"], 50)
        # Wan's 480p shift follows the smaller default canvas.
        self.assertEqual(generate["flow_shift"], 3.0)
        self.assertEqual((asset.metadata["width"], asset.metadata["height"]), (832, 480))

    def test_wan_ti2v_caller_can_still_request_720p_canvas(self):
        from abstractvision.types import VideoGenerationRequest

        self._run_one_wan_video(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            VideoGenerationRequest(prompt="fox", width=1280, height=704, num_frames=81, fps=16, steps=30, seed=1),
        )

        generate = _FakeWan.last_generate
        self.assertEqual((generate["width"], generate["height"]), (1280, 704))
        self.assertEqual((generate["num_frames"], generate["fps"], generate["num_inference_steps"]), (81, 16, 30))
        # Above 832x480 the shift is left to mlx-gen's 720p default.
        self.assertNotIn("flow_shift", generate)

    def test_wan_default_canvases_live_in_one_table(self):
        from abstractvision.backends import mflux as mflux_mod

        table = mflux_mod.WAN_VIDEO_DEFAULT_CANVASES
        self.assertEqual(
            set(table),
            {mflux_mod.WAN_TI2V_MODEL_KEY, mflux_mod.WAN_T2V_A14B_MODEL_KEY, mflux_mod.WAN_I2V_A14B_MODEL_KEY},
        )
        for key, canvas in table.items():
            model_def = mflux_mod._MFLUX_MODELS[key]
            self.assertEqual(
                (model_def.default_width, model_def.default_height, model_def.default_frames,
                 model_def.default_fps, model_def.default_steps),
                (canvas.width, canvas.height, canvas.frames, canvas.fps, canvas.steps),
                key,
            )
            self.assertEqual((canvas.frames - 1) % 4, 0, key)

    def test_packaged_capability_registry_follows_wan_default_canvases(self):
        import json
        from importlib import resources

        from abstractvision.backends import mflux as mflux_mod

        data = json.loads(
            resources.files("abstractvision").joinpath("assets/vision_model_capabilities.json").read_text()
        )
        for repo_id, key in (
            ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", mflux_mod.WAN_TI2V_MODEL_KEY),
            ("Wan-AI/Wan2.2-T2V-A14B", mflux_mod.WAN_T2V_A14B_MODEL_KEY),
            ("Wan-AI/Wan2.2-I2V-A14B", mflux_mod.WAN_I2V_A14B_MODEL_KEY),
        ):
            canvas = mflux_mod.WAN_VIDEO_DEFAULT_CANVASES[key]
            for task, spec in data["models"][repo_id]["tasks"].items():
                params = spec["params"]
                self.assertEqual(
                    (params["width"]["default"], params["height"]["default"], params["num_frames"]["default"],
                     params["fps"]["default"], params["steps"]["default"]),
                    (canvas.width, canvas.height, canvas.frames, canvas.fps, canvas.steps),
                    f"{repo_id} {task}",
                )

    def test_wan_video_decodes_tiled_by_default_and_restores_the_vae(self):
        from abstractvision.types import VideoGenerationRequest

        backend, asset = self._run_one_wan_video(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", VideoGenerationRequest(prompt="fox", seed=1)
        )

        self.assertIs(_FakeWan.last_decode_tiled, True)
        self.assertIs(asset.metadata["vae_tiling"], True)
        # The warm A14B instance keeps its VAE, back on mlx-gen's own method.
        self.assertNotIn("iter_decode_normalized_latent_slices", vars(backend._model.vae))

    def test_wan_video_vae_tiling_can_be_switched_off_per_request(self):
        from abstractvision.types import VideoGenerationRequest

        _backend, asset = self._run_one_wan_video(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            VideoGenerationRequest(prompt="fox", seed=1, extra={"vae_tiling": "false"}),
        )

        self.assertIs(_FakeWan.last_decode_tiled, False)
        self.assertIs(asset.metadata["vae_tiling"], False)
        self.assertNotIn("vae_tiling", _FakeWan.last_generate)

    def test_wan_video_rejects_unreadable_vae_tiling_value(self):
        from abstractvision.types import VideoGenerationRequest

        with self.assertRaisesRegex(ValueError, "vae_tiling"):
            self._run_one_wan_video(
                "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                VideoGenerationRequest(prompt="fox", seed=1, extra={"vae_tiling": "sometimes"}),
            )

    def _run_wan_videos_recording_memory_policy(self, repo_id, wan_cls, requests):
        """Run requests through a Wan route, recording cache-limit and model lifecycle events."""
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        events = []
        active_limit = [None]

        def fake_set_limit(limit_bytes):
            previous = active_limit[0] if active_limit[0] is not None else _CAP_IN_FORCE
            active_limit[0] = int(limit_bytes)
            events.append(("set_limit", int(limit_bytes)))
            return previous

        class _RecordingVideo(_GeneratedVideo):
            def save(self, path=None, export_json_metadata=False, overwrite=True):
                events.append(("save", active_limit[0]))
                super().save(path=path, overwrite=overwrite)

        class _RecordingWan(wan_cls):
            instances = 0

            def __init__(self, **kwargs):
                type(self).instances += 1
                events.append(("init", None))
                super().__init__(**kwargs)

            def generate_video(self, **kwargs):
                events.append(("generate", active_limit[0]))
                super().generate_video(**kwargs)
                return _RecordingVideo()

        assets = []
        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), repo_id)
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model=repo_id))
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ), patch(
                    "abstractvision.backends.mflux._lazy_import_mflux_wan",
                    return_value=_RecordingWan,
                ), patch(
                    "abstractvision.backends.mflux._set_mlx_cache_limit",
                    side_effect=fake_set_limit,
                ), patch("abstractvision.backends.mflux._apply_mlx_gen_default_cache_limit"):
                    for request in requests:
                        assets.append(
                            request(backend) if callable(request) else backend.generate_video(request)
                        )
        return backend, assets, events, _RecordingWan

    def test_wan_ti2v_releases_denoiser_before_decode_and_rebuilds_next_request(self):
        from abstractvision.types import VideoGenerationRequest

        backend, assets, events, wan_cls = self._run_wan_videos_recording_memory_policy(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            _FakeWan,
            [
                VideoGenerationRequest(prompt="fox", width=832, height=480, num_frames=5, steps=2, seed=1),
                VideoGenerationRequest(prompt="fox", width=832, height=480, num_frames=5, steps=2, seed=2),
            ],
        )

        self.assertIs(_FakeWan.last_generate.get("release_denoisers_before_decode"), True)
        # mlx-gen cannot reuse a TI2V instance whose denoiser was released:
        # every request must build a fresh one.
        self.assertEqual(wan_cls.instances, 2)
        self.assertIsNone(backend._model)
        self.assertEqual(assets[1].metadata["base_model"], "wan2.2-ti2v-5b")
        self.assertEqual(assets[1].metadata["seed"], 2)

    def test_a_default_wan_run_leaves_the_cache_limit_to_mlx_gen(self):
        from abstractvision.types import VideoGenerationRequest

        _backend, _assets, events, _wan_cls = self._run_wan_videos_recording_memory_policy(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            _FakeWan,
            [VideoGenerationRequest(prompt="fox", width=832, height=480, num_frames=5, steps=2, seed=1)],
        )

        # mlx-gen caps the cache itself at its first weight load (RAM/8, 1-8 GiB) and keeps a
        # cap the host set; a default-mode run never sets one, so it never raises the operator's.
        self.assertEqual([value for kind, value in events if kind == "set_limit"], [])

    def test_wan_a14b_keeps_its_denoisers_between_requests(self):
        from abstractvision.types import VideoGenerationRequest

        backend, _assets, events, wan_cls = self._run_wan_videos_recording_memory_policy(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            _FakeWan,
            [VideoGenerationRequest(prompt="fox", seed=1), VideoGenerationRequest(prompt="fox", seed=2)],
        )

        self.assertNotIn("release_denoisers_before_decode", _FakeWan.last_generate)
        self.assertEqual(wan_cls.instances, 1)
        self.assertIsNotNone(backend._model)
        self.assertNotIn("set_limit", [kind for kind, _value in events])

    def test_wan_ti2v_failed_decode_still_drops_released_model(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import VideoGenerationRequest

        limits = []

        class _FailingVideo(_GeneratedVideo):
            def save(self, path=None, export_json_metadata=False, overwrite=True):
                raise RuntimeError("decode failed")

        class _FailingWan(_FakeWan):
            def generate_video(self, **kwargs):
                super().generate_video(**kwargs)
                return _FailingVideo()

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            )
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ), patch(
                    "abstractvision.backends.mflux._lazy_import_mflux_wan",
                    return_value=_FailingWan,
                ), patch(
                    "abstractvision.backends.mflux._set_mlx_cache_limit",
                    side_effect=lambda value: limits.append(value) or 777,
                ):
                    with self.assertRaisesRegex(RuntimeError, "decode failed"):
                        backend.generate_video(
                            VideoGenerationRequest(
                                prompt="fox", width=832, height=480, num_frames=5, steps=2, seed=1
                            )
                        )

        self.assertIsNone(backend._model)
        self.assertIsNone(backend._model_key)
        self.assertEqual(limits, [])

    def test_wan_a14b_text_to_video_accepts_explicit_guidance_2(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import VideoGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        asset = backend.generate_video(
                            VideoGenerationRequest(prompt="fox", seed=42, guidance_2=2.25)
                        )

        self.assertEqual(_FakeWan.last_generate["guidance_2"], 2.25)
        self.assertEqual(asset.metadata["guidance_2"], 2.25)

    def test_wan_a14b_rejects_non_multiple_of_16_dimensions(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        with self.assertRaises(CapabilityNotSupportedError) as ctx:
                            backend.generate_video(
                                VideoGenerationRequest(
                                    prompt="fox",
                                    width=482,
                                    height=240,
                                    num_frames=9,
                                    steps=4,
                                    seed=42,
                                )
                            )

        self.assertIn("multiples of 16", str(ctx.exception))

    def test_wan_a14b_text_to_video_passes_lora_target_roles(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import LoRAAdapterSpec, VideoGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        asset = backend.generate_video(
                            VideoGenerationRequest(
                                prompt="fox",
                                seed=42,
                                lora_adapters=(
                                    LoRAAdapterSpec(
                                        source="owner/wan-lora:hi.safetensors",
                                        scale=0.8,
                                        target_role="high_noise_transformer",
                                    ),
                                    LoRAAdapterSpec(
                                        source="owner/wan-lora:lo.safetensors",
                                        scale=0.6,
                                        target_role="low_noise_transformer",
                                    ),
                                ),
                            )
                        )

        self.assertEqual(
            _FakeWan.last_init["lora_paths"],
            ["owner/wan-lora:hi.safetensors", "owner/wan-lora:lo.safetensors"],
        )
        self.assertEqual(_FakeWan.last_init["lora_scales"], [0.8, 0.6])
        self.assertEqual(
            _FakeWan.last_init["lora_target_roles"],
            ["high_noise_transformer", "low_noise_transformer"],
        )
        self.assertEqual(len(asset.metadata["requested_lora_adapters"]), 2)

    def test_wan_a14b_routes_image_to_video_with_model_defaults(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageToVideoRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        asset = backend.image_to_video(
                            ImageToVideoRequest(
                                image=_solid_png(32, 32, "blue"),
                                prompt="move",
                                seed=42,
                            )
                        )

        self.assertEqual(_FakeWan.last_init["model_config"], "wan-i2v-a14b-config")
        self.assertEqual(_FakeWan.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeWan.last_generate["width"], 832)
        self.assertEqual(_FakeWan.last_generate["height"], 480)
        self.assertEqual(_FakeWan.last_generate["fps"], 16)
        self.assertEqual(_FakeWan.last_generate["num_frames"], 81)
        self.assertEqual(_FakeWan.last_generate["num_inference_steps"], 40)
        self.assertEqual(_FakeWan.last_generate["guidance"], 3.5)
        self.assertEqual(_FakeWan.last_generate["guidance_2"], 3.5)
        self.assertIn("image_path", _FakeWan.last_generate)
        self.assertEqual(asset.metadata["base_model"], "wan2.2-i2v-a14b")
        self.assertEqual(asset.metadata["guidance_2"], 3.5)

    def test_wan_a14b_image_to_video_accepts_explicit_guidance_2(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageToVideoRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        asset = backend.image_to_video(
                            ImageToVideoRequest(
                                image=_solid_png(32, 32, "blue"),
                                prompt="move",
                                seed=42,
                                guidance_2=2.75,
                            )
                        )

        self.assertEqual(_FakeWan.last_generate["guidance_2"], 2.75)
        self.assertEqual(asset.metadata["guidance_2"], 2.75)

    def test_wan_emits_normalized_video_progress_events(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import VideoGenerationRequest, VideoProgressEvent

        seen = []

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        backend.generate_video(
                            VideoGenerationRequest(
                                prompt="fox",
                                num_frames=5,
                                steps=2,
                                extra={"on_progress": seen.append},
                            )
                        )

        self.assertEqual([event.phase for event in seen], ["start", "denoise", "complete"])
        self.assertTrue(all(isinstance(event, VideoProgressEvent) for event in seen))
        self.assertEqual(seen[1].progress, 0.5)
        self.assertEqual(seen[1].step_progress, 0.5)
        self.assertEqual(seen[1].frame_progress, 0.6)
        self.assertEqual(seen[-1].frame, 5)
        self.assertEqual(seen[-1].total_frames, 5)
        self.assertEqual(seen[-1].step, 2)
        self.assertEqual(seen[-1].total_steps, 2)
        self.assertEqual(seen[-1].progress, 1.0)

    def test_wan_normalizes_missing_progress_from_steps_not_frames(self):
        from abstractvision.backends.mflux import _normalize_video_progress_event

        raw_event = {
            "phase": "denoise",
            "frame": 3,
            "total_frames": 10,
            "step": 2,
            "total_steps": 5,
            "task": "text-to-video",
        }

        event = _normalize_video_progress_event(raw_event)

        self.assertEqual(event.progress, 0.4)
        self.assertEqual(event.step_progress, 0.4)
        self.assertEqual(event.frame_progress, 0.3)

    def test_wan_generate_video_with_progress_uses_step_counts(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import VideoGenerationRequest

        seen = []

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        backend.generate_video_with_progress(
                            VideoGenerationRequest(prompt="fox", num_frames=5, steps=2),
                            progress_callback=lambda current, total: seen.append((current, total)),
                        )

        self.assertEqual(seen, [(0, 2), (1, 2), (2, 2)])

    def test_wan_routes_image_to_video_with_letterboxed_temp_image(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import VideoProgressEvent
        from abstractvision.types import ImageToVideoRequest

        seen = []
        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        asset = backend.image_to_video(
                            ImageToVideoRequest(
                                image=_solid_png(32, 32, (255, 0, 0)),
                                prompt="slow push in",
                                width=832,
                                height=480,
                                fps=8,
                                num_frames=5,
                                steps=1,
                                seed=9,
                                extra={"on_progress": seen.append},
                            )
                        )

        self.assertEqual(asset.media_type, "video")
        self.assertEqual(_FakeWan.last_generate["prompt"], "slow push in")
        self.assertEqual(_FakeWan.last_generate["width"], 832)
        self.assertEqual(_FakeWan.last_generate["height"], 480)
        self.assertEqual(_FakeWan.last_generate["fps"], 8)
        self.assertEqual(_FakeWan.last_generate["num_frames"], 5)
        self.assertEqual(_FakeWan.last_generate["num_inference_steps"], 1)
        self.assertIsNotNone(_FakeWan.last_generate.get("image_path"))
        self.assertTrue(_FakeWan.last_image_path_existed)
        self.assertEqual(_FakeWan.last_image_size, (832, 480))
        self.assertEqual(_FakeWan.last_image_corner_pixel, (0, 0, 0))
        self.assertEqual(_FakeWan.last_image_center_pixel, (255, 0, 0))
        self.assertEqual(asset.metadata["task"], "image_to_video")
        self.assertEqual(asset.metadata["conditioning_image"]["mode"], "letterbox")
        self.assertEqual(asset.metadata["conditioning_image"]["fit_width"], 480)
        self.assertEqual(asset.metadata["conditioning_image"]["fit_height"], 480)
        self.assertEqual(asset.metadata["conditioning_image"]["pad_left"], 176)
        self.assertEqual(asset.metadata["conditioning_image"]["pad_right"], 176)
        self.assertEqual([event.phase for event in seen], ["start", "denoise", "complete"])
        self.assertTrue(all(isinstance(event, VideoProgressEvent) for event in seen))
        self.assertEqual(seen[1].task, "image_to_video")
        self.assertEqual(seen[1].step, 1)
        self.assertEqual(seen[1].total_steps, 2)
        self.assertEqual(seen[1].progress, 0.5)
        self.assertEqual(seen[1].step_progress, 0.5)
        self.assertEqual(seen[1].frame_progress, 0.6)

    def test_wan_config_fallback_uses_registry_id_without_model_def_attribute(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageToVideoRequest

        _FakeModelConfigWithoutWanFactory.last_from_name = None
        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=(
                        _FakeModelConfigWithoutWanFactory,
                        _FakeDownloadRequiredError,
                        _FakeFlux2,
                        _FakeFlux2Edit,
                        _FakeZImage,
                        _FakeZImage,
                    ),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_wan",
                        return_value=_FakeWan,
                    ):
                        asset = backend.image_to_video(
                            ImageToVideoRequest(
                                image=_solid_png(16, 16, (0, 128, 255)),
                                prompt="animate",
                                steps=1,
                            )
                        )

        self.assertEqual(asset.metadata["task"], "image_to_video")
        self.assertEqual(
            _FakeModelConfigWithoutWanFactory.last_from_name,
            ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", None),
        )
        self.assertEqual(_FakeWan.last_init["model_config"], "wan-ti2v-config")

    def test_wan_config_fallback_reports_upgrade_when_installed_mlx_gen_lacks_wan(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import OptionalDependencyMissingError
        from abstractvision.types import ImageToVideoRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=(
                        _FakeModelConfigWithoutWanSupport,
                        _FakeDownloadRequiredError,
                        _FakeFlux2,
                        _FakeFlux2Edit,
                        _FakeZImage,
                        _FakeZImage,
                    ),
                ):
                    with self.assertRaisesRegex(
                        OptionalDependencyMissingError,
                        r"mlx-gen>=0\.40\.1",
                    ):
                        backend.image_to_video(
                            ImageToVideoRequest(
                                image=_solid_png(16, 16, (0, 128, 255)),
                                prompt="animate",
                                steps=1,
                            )
                        )

    def test_z_image_turbo_uses_alternate_cached_repo_for_preset_key(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(
                Path(cache_td), "AbstractFramework/z-image-turbo-8bit"
            )
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/z-image-turbo-8bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    asset = backend.generate_image(
                        ImageGenerationRequest(prompt="hello", negative_prompt="blur", seed=7)
                    )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeZImage.last_init["model_path"], str(snapshot))

    def test_discovers_quarantined_local_mflux_variant(self):
        from abstractvision.backends.mflux import discover_cached_mflux_models

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            local_dir = self._make_model_dir(root, "flux2-klein-9b-mlx-q4")
            with patch(
                "abstractvision.backends.mflux.framework_local_model_roots",
                return_value=[("quarantined local models", root)],
            ):
                with patch.dict(
                    "os.environ", {"HF_HUB_CACHE": str(root / "empty-cache")}, clear=True
                ):
                    with patch(
                        "abstractvision.backends.mflux.framework_hf_cache_roots",
                        return_value=[],
                    ):
                        discovered = discover_cached_mflux_models()

        self.assertIn("flux2-klein-9b", discovered)
        chosen = discovered["flux2-klein-9b"]
        self.assertEqual(chosen.snapshot_dir, local_dir)
        self.assertEqual(chosen.source_label, "quarantined local model dir")

    def test_marks_incompatible_quarantined_hf_snapshot_as_invalid(self):
        from abstractvision.backends.mflux import (
            discover_cached_mflux_models,
            discover_incomplete_mflux_sources,
        )

        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            repo_dir = (
                cache_root
                / "models--AbstractFramework--z-image-turbo-8bit.incompatible.20260515132039"
            )
            snap = repo_dir / "snapshots" / "abc123"
            (snap / "transformer").mkdir(parents=True, exist_ok=True)
            (snap / "transformer" / "0.safetensors").write_bytes(b"x")
            (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
            (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")

            with patch(
                "abstractvision.backends.mflux.framework_local_model_roots",
                return_value=[],
            ):
                with patch.dict(
                    "os.environ", {"HF_HUB_CACHE": str(cache_root / "empty")}, clear=True
                ):
                    with patch(
                        "abstractvision.backends.mflux.framework_hf_cache_roots",
                        return_value=[("quarantined HF cache", cache_root)],
                    ):
                        discovered = discover_cached_mflux_models()
                        invalid = discover_incomplete_mflux_sources()

        self.assertNotIn("z-image-turbo", discovered)
        self.assertIn("z-image-turbo", invalid)
        self.assertIn(
            "incompatible HF cache: quarantined HF cache (AbstractFramework/z-image-turbo-8bit)",
            invalid["z-image-turbo"],
        )

    def test_flux2_drops_unsupported_negative_prompt(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-9b-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-9b-8bit")
            )
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    asset = backend.generate_image(
                        ImageGenerationRequest(prompt="hello", negative_prompt="blur")
                    )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertNotIn("negative_prompt", _FakeFlux2.last_generate)

    def test_flux2_normalizes_steps_and_guidance_scale(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-4bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-4bit")
            )
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    asset = backend.generate_image(
                        ImageGenerationRequest(prompt="hello", steps=1, guidance_scale=7.0)
                    )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeFlux2.last_generate["num_inference_steps"], 2)
        self.assertEqual(_FakeFlux2.last_generate["guidance"], 1.0)

    def test_qwen_lightning_defaults_apply_to_text_to_image_requests(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest, LoRAAdapterSpec

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/qwen-image-2512-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/qwen-image-2512-8bit")
            )
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                normalized = backend.normalize_image_generation_request(
                    ImageGenerationRequest(
                        prompt="hello",
                        lora_adapters=(
                            LoRAAdapterSpec(source="lightx2v/Qwen-Image-2512-Lightning"),
                        ),
                    )
                )

        self.assertEqual(normalized.steps, 4)
        self.assertEqual(normalized.guidance_scale, 1.0)

    def test_qwen_lightning_defaults_apply_to_image_edit_requests(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest, LoRAAdapterSpec

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/qwen-image-edit-2511-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/qwen-image-edit-2511-8bit")
            )
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                normalized = backend.normalize_image_edit_request(
                    ImageEditRequest(
                        prompt="hello",
                        image=PNG_1X1,
                        lora_adapters=(
                            LoRAAdapterSpec(source="lightx2v/Qwen-Image-Edit-2511-Lightning"),
                        ),
                    )
                )

        self.assertEqual(normalized.steps, 4)
        self.assertEqual(normalized.guidance_scale, 1.0)

    def test_a_control_image_is_refused_before_anything_loads(self):
        # MLX-Gen runs structured control through QwenImageControlNet and a ControlNet sidecar
        # repository; this backend loads neither, so the field is refused, typed or in `extra`.
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageGenerationRequest

        _FakeQwenImage.last_init = None
        backend = MFluxVisionBackend(config=MFluxBackendConfig(model="AbstractFramework/qwen-image-8bit"))
        for request, field in (
            (ImageGenerationRequest(prompt="hello", control_image=PNG_1X1, control_strength=0.9), "control_image"),
            (ImageGenerationRequest(prompt="hello", extra={"control_image": PNG_1X1}), "control_image"),
            # The strength alone applies to nothing: refused the same way, not ignored.
            (ImageGenerationRequest(prompt="hello", control_strength=0.9), "control_strength"),
            (ImageGenerationRequest(prompt="hello", extra={"control_strength": 0.9}), "control_strength"),
        ):
            with self.subTest(field=field, extra=sorted(request.extra)):
                with patch.object(MFluxVisionBackend, "_find_configured_model", side_effect=AssertionError("resolved")):
                    with self.assertRaises(CapabilityNotSupportedError) as caught:
                        backend.generate_image(request)
                self.assertEqual((caught.exception.field, caught.exception.supported), (field, ()))
        self.assertIsNone(_FakeQwenImage.last_init)
        self.assertFalse(backend.has_loaded_model())

    def test_qwen_edit_2511_masked_edit_maps_to_mask_path(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest

        _FakeQwenImageEdit.last_generate = None
        fake_capabilities = SimpleNamespace(
            capabilities=(
                SimpleNamespace(public_task="image-to-image", default_for_task=True),
                SimpleNamespace(public_task="image-to-image", supports_mask=True),
            )
        )

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/qwen-image-edit-2511-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/qwen-image-edit-2511-8bit")
            )
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_qwen",
                        return_value=(_FakeQwenImage, _FakeQwenImageEdit),
                    ):
                        with patch(
                            "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                            return_value=(lambda **kwargs: fake_capabilities, None),
                        ):
                            asset = backend.edit_image(
                                ImageEditRequest(
                                    prompt="repair",
                                    image=PNG_1X1,
                                    mask=PNG_1X1,
                                    seed=7,
                                    steps=4,
                                )
                            )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(len(_FakeQwenImageEdit.last_generate["image_paths"]), 1)
        self.assertIn("mask_path", _FakeQwenImageEdit.last_generate)
        self.assertTrue(asset.metadata["mask"])

    def test_wan_lightning_defaults_apply_to_video_requests(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import LoRAAdapterSpec, VideoGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(
                Path(cache_td), "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"
            )
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit")
            )
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                normalized = backend.normalize_video_generation_request(
                    VideoGenerationRequest(
                        prompt="hello",
                        lora_adapters=(
                            LoRAAdapterSpec(
                                source=(
                                    "lightx2v/Wan2.2-Lightning:"
                                    "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors"
                                )
                            ),
                            LoRAAdapterSpec(
                                source=(
                                    "lightx2v/Wan2.2-Lightning:"
                                    "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors"
                                )
                            ),
                        ),
                    )
                )

        self.assertEqual(normalized.steps, 4)
        self.assertEqual(normalized.guidance_scale, 4.0)

    def test_missing_local_model_has_download_hint(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import OptionalDependencyMissingError
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as td:
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(
                    model="AbstractFramework/flux.2-klein-4b-4bit", model_dir=td
                )
            )
            with patch.dict(
                "os.environ", {"HF_HUB_CACHE": str(Path(td) / "empty-cache")}, clear=True
            ):
                with patch(
                    "abstractvision.backends.mflux.framework_hf_cache_roots",
                    return_value=[],
                ):
                    with self.assertRaises(OptionalDependencyMissingError) as ctx:
                        backend.generate_image(ImageGenerationRequest(prompt="hello"))

        self.assertIn(
            "abstractvision download AbstractFramework/flux.2-klein-4b-4bit", str(ctx.exception)
        )

    def test_hf_repo_id_can_resolve_from_cache_without_allow_download(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "org/repo")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="org/repo", base_model="flux2-klein-4b")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    asset = backend.generate_image(ImageGenerationRequest(prompt="hello", seed=2))

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeFlux2.last_init["model_path"], str(snapshot))

    def test_qwen_model_family_uses_qwen_variant(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            local_dir = self._make_model_dir(root, "qwen-image-2512-mlx-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(
                    model=str(local_dir), base_model="qwen-image", model_dir=str(root)
                )
            )

            with patch(
                "abstractvision.backends.mflux._lazy_import_mflux",
                return_value=self._lazy_import_return(),
            ):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux_qwen",
                    return_value=(_FakeQwenImage, _FakeQwenImageEdit),
                ):
                    asset = backend.generate_image(ImageGenerationRequest(prompt="hello", seed=1))

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeQwenImage.last_init["model_config"], "qwen-image-config")
        self.assertEqual(_FakeQwenImage.last_init["model_path"], str(local_dir))
        self.assertEqual(_FakeQwenImage.last_generate["prompt"], "hello")

    def test_qwen_image_capabilities_are_text_to_image_only(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        backend = MFluxVisionBackend(
            config=MFluxBackendConfig(model="AbstractFramework/qwen-image-2512-4bit")
        )
        caps = backend.get_capabilities()

        self.assertEqual(caps.supported_tasks, ["text_to_image"])
        self.assertFalse(caps.supports_mask)

    def test_qwen_base_control_is_not_advertised_although_the_route_row_has_it(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        fake_capabilities = SimpleNamespace(
            capabilities=(
                SimpleNamespace(
                    public_task="text-to-image",
                    default_for_task=True,
                    supports_lora=True,
                ),
                SimpleNamespace(
                    public_task="text-to-image",
                    supports_control_image=True,
                    supports_lora=True,
                ),
            )
        )

        backend = MFluxVisionBackend(
            config=MFluxBackendConfig(model="AbstractFramework/qwen-image-8bit")
        )

        with patch(
            "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
            return_value=(lambda **kwargs: fake_capabilities, None),
        ):
            caps = backend.get_capabilities()

        self.assertEqual(caps.supported_tasks, ["text_to_image"])
        self.assertFalse(caps.supports_mask)
        # The engine's row supports control; this backend does not run that variant.
        self.assertFalse(caps.supports_control_image)

    def test_qwen_image_provider_catalog_does_not_advertise_image_to_image(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/qwen-image-2512-4bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/qwen-image-2512-4bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                models = list(backend.list_provider_models())
                image_edit_models = list(backend.list_provider_models(task="image_to_image"))

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "AbstractFramework/qwen-image-2512-4bit")
        self.assertEqual(tuple(models[0].capabilities), ("text_to_image",))
        self.assertEqual(image_edit_models, [])

    def test_qwen_base_provider_catalog_lists_no_structured_control_params(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        fake_capabilities = SimpleNamespace(
            capabilities=(
                SimpleNamespace(
                    public_task="text-to-image",
                    default_for_task=True,
                    supports_lora=True,
                ),
                SimpleNamespace(
                    public_task="text-to-image",
                    supports_control_image=True,
                    supports_lora=True,
                ),
            )
        )

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/qwen-image-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/qwen-image-8bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                    return_value=(lambda **kwargs: fake_capabilities, None),
                ):
                    models = list(backend.list_provider_models(task="text_to_image"))

        self.assertEqual(len(models), 1)
        params = models[0].raw["task_specs"]["text_to_image"]["params"]
        self.assertNotIn("control_image", params)
        self.assertNotIn("control_strength", params)

    def test_wan_provider_catalog_advertises_video_tasks(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                t2v_models = list(backend.list_provider_models(task="text_to_video"))
                i2v_models = list(backend.list_provider_models(task="image_to_video"))

        self.assertEqual(len(t2v_models), 1)
        self.assertEqual(t2v_models[0].id, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        self.assertEqual(tuple(t2v_models[0].capabilities), ("image_to_video", "text_to_video"))
        self.assertEqual(t2v_models[0].raw["quantization_bits"], 16)
        self.assertEqual(len(i2v_models), 1)
        self.assertEqual(i2v_models[0].id, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

    def test_wan_ti2v_provider_catalog_advertises_prepared_abstractframework_variant(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(
                Path(cache_td), "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"
            )
            backend = MFluxVisionBackend(config=MFluxBackendConfig())

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                t2v_models = list(backend.list_provider_models(task="text_to_video"))
                i2v_models = list(backend.list_provider_models(task="image_to_video"))

        self.assertEqual(
            [m.id for m in t2v_models],
            ["AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"],
        )
        self.assertEqual(tuple(t2v_models[0].capabilities), ("image_to_video", "text_to_video"))
        self.assertEqual(t2v_models[0].raw["base_model"], "wan2.2-ti2v-5b")
        self.assertEqual(t2v_models[0].raw["quantization_bits"], 8)
        self.assertEqual(t2v_models[0].raw["parameter_defaults"]["fps"], 24)
        self.assertEqual(t2v_models[0].raw["parameter_defaults"]["num_frames"], 121)
        self.assertEqual(
            [m.id for m in i2v_models],
            ["AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"],
        )
        self.assertEqual(tuple(i2v_models[0].capabilities), ("image_to_video", "text_to_video"))

    def test_wan_a14b_provider_catalog_advertises_task_specific_models(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
            backend = MFluxVisionBackend(config=MFluxBackendConfig())

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                t2v_models = list(backend.list_provider_models(task="text_to_video"))
                i2v_models = list(backend.list_provider_models(task="image_to_video"))

        self.assertEqual([m.id for m in t2v_models], ["Wan-AI/Wan2.2-T2V-A14B-Diffusers"])
        self.assertEqual(tuple(t2v_models[0].capabilities), ("text_to_video",))
        self.assertEqual(t2v_models[0].raw["parameter_defaults"]["num_frames"], 81)
        self.assertEqual(t2v_models[0].raw["parameter_defaults"]["guidance_2"], 3.0)
        self.assertEqual([m.id for m in i2v_models], ["Wan-AI/Wan2.2-I2V-A14B-Diffusers"])
        self.assertEqual(tuple(i2v_models[0].capabilities), ("image_to_video",))
        self.assertEqual(i2v_models[0].raw["parameter_defaults"]["guidance_2"], 3.5)

    def test_wan_a14b_provider_catalog_advertises_prepared_abstractframework_variants(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(
                Path(cache_td), "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"
            )
            self._make_cache_snapshot(
                Path(cache_td), "AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit"
            )
            backend = MFluxVisionBackend(config=MFluxBackendConfig())

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                t2v_models = list(backend.list_provider_models(task="text_to_video"))
                i2v_models = list(backend.list_provider_models(task="image_to_video"))

        self.assertEqual(
            [m.id for m in t2v_models],
            ["AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"],
        )
        self.assertEqual(tuple(t2v_models[0].capabilities), ("text_to_video",))
        self.assertEqual(t2v_models[0].raw["base_model"], "wan2.2-t2v-a14b")
        self.assertEqual(t2v_models[0].raw["parameter_defaults"]["width"], 832)
        self.assertEqual(t2v_models[0].raw["parameter_defaults"]["height"], 480)
        self.assertEqual(t2v_models[0].raw["parameter_defaults"]["num_frames"], 81)
        self.assertEqual(t2v_models[0].raw["parameter_defaults"]["fps"], 16)
        self.assertEqual(t2v_models[0].raw["parameter_defaults"]["guidance_2"], 3.0)
        self.assertEqual(
            [m.id for m in i2v_models],
            ["AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit"],
        )
        self.assertEqual(tuple(i2v_models[0].capabilities), ("image_to_video",))
        self.assertEqual(i2v_models[0].raw["base_model"], "wan2.2-i2v-a14b")
        self.assertEqual(i2v_models[0].raw["parameter_defaults"]["guidance_2"], 3.5)

    def test_provider_catalog_surfaces_route_level_lora_metadata(self):
        from types import SimpleNamespace

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        fake_capabilities = SimpleNamespace(
            capabilities=(
                SimpleNamespace(
                    public_task="text-to-video",
                    default_for_task=True,
                    supports_lora=True,
                    lora_status="validated",
                    lora_target_roles=("high_noise_transformer", "low_noise_transformer"),
                    lora_validation_profile="wan_a14b_lora_profile",
                ),
            )
        )

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(
                Path(cache_td), "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"
            )
            backend = MFluxVisionBackend(config=MFluxBackendConfig())

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                    return_value=(lambda **kwargs: fake_capabilities, None),
                ):
                    models = list(backend.list_provider_models(task="text_to_video"))

        self.assertEqual(len(models), 1)
        task_spec = models[0].raw["task_specs"]["text_to_video"]
        self.assertTrue(task_spec["supports_lora"])
        self.assertEqual(task_spec["lora_status"], "validated")
        self.assertEqual(
            task_spec["lora_target_roles"],
            ["high_noise_transformer", "low_noise_transformer"],
        )
        self.assertEqual(task_spec["lora_validation_profile"], "wan_a14b_lora_profile")

    @unittest.skipUnless(_YAML_INSTALLED, "PyYAML is not installed: adapter cards are not read")
    def test_provider_adapter_inventory_discovers_ti2v_and_a14b_cached_adapters(self):
        from types import SimpleNamespace

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        def fake_capabilities(*, model=None, **kwargs):
            model_s = str(model or "").lower()
            if "ti2v" in model_s:
                roles = ("transformer",)
                profile = "wan_ti2v5b_q8_hstoric_t2v"
            else:
                roles = ("high_noise_transformer", "low_noise_transformer")
                profile = "wan_a14b_q8_lightning_t2v"
            return SimpleNamespace(
                capabilities=(
                    SimpleNamespace(
                        public_task="text-to-video",
                        default_for_task=True,
                        supports_lora=True,
                        lora_status="validated",
                        lora_target_roles=roles,
                        lora_validation_profile=profile,
                    ),
                )
            )

        with tempfile.TemporaryDirectory() as cache_td:
            cache_root = Path(cache_td)
            self._make_cache_snapshot(
                cache_root, "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"
            )
            self._make_cache_snapshot(
                cache_root, "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"
            )
            self._make_adapter_snapshot(
                cache_root,
                "AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets",
                "HSToric_color_Wan22_5b_LoRA.safetensors",
                base_model="Wan-AI/Wan2.2-TI2V-5B",
            )
            self._make_adapter_snapshot(
                cache_root,
                "lightx2v/Wan2.2-Lightning",
                "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
                base_model=(
                    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                ),
            )
            self._make_adapter_snapshot(
                cache_root,
                "lightx2v/Wan2.2-Lightning",
                "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
                base_model=(
                    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                ),
            )

            backend = MFluxVisionBackend(config=MFluxBackendConfig())

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                    return_value=(fake_capabilities, None),
                ):
                    ti2v_adapters = list(
                        backend.list_provider_adapters(
                            model="AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit",
                            task="text_to_video",
                        )
                    )
                    a14b_adapters = list(
                        backend.list_provider_adapters(
                            model="AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
                            task="text_to_video",
                        )
                    )

        self.assertEqual(len(ti2v_adapters), 1)
        self.assertEqual(
            ti2v_adapters[0].id,
            "AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets:HSToric_color_Wan22_5b_LoRA.safetensors",
        )
        self.assertEqual(
            list(ti2v_adapters[0].compatible_models),
            ["AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"],
        )
        self.assertEqual(list(ti2v_adapters[0].suggested_target_roles), ["transformer"])

        self.assertEqual(len(a14b_adapters), 2)
        roles = {item.id: tuple(item.suggested_target_roles) for item in a14b_adapters}
        self.assertEqual(
            roles[
                "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors"
            ],
            ("high_noise_transformer",),
        )
        self.assertEqual(
            roles[
                "lightx2v/Wan2.2-Lightning:Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors"
            ],
            ("low_noise_transformer",),
        )
        route_details = a14b_adapters[0].raw["compatible_routes"]
        self.assertEqual(route_details[0]["lora_status"], "validated")
        self.assertEqual(route_details[0]["task"], "text_to_video")

    def test_provider_adapter_inventory_uses_known_lightning_defaults_and_skips_fused_models(self):
        from types import SimpleNamespace

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        fake_capabilities = SimpleNamespace(
            capabilities=(
                SimpleNamespace(
                    public_task="image-to-image",
                    default_for_task=True,
                    supports_lora=True,
                    lora_status="validated",
                    lora_target_roles=("transformer",),
                    lora_validation_profile="qwen2511_lightning_profile",
                ),
            )
        )

        with tempfile.TemporaryDirectory() as cache_td:
            cache_root = Path(cache_td)
            self._make_cache_snapshot(
                cache_root, "AbstractFramework/qwen-image-edit-2511-8bit"
            )
            self._make_adapter_snapshot(
                cache_root,
                "lightx2v/Qwen-Image-Edit-2511-Lightning",
                "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                base_model="Qwen/Qwen-Image-Edit-2511",
            )
            self._make_adapter_snapshot(
                cache_root,
                "lightx2v/Qwen-Image-Edit-2511-Lightning",
                "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
                base_model="Qwen/Qwen-Image-Edit-2511",
            )

            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/qwen-image-edit-2511-8bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                    return_value=(lambda **kwargs: fake_capabilities, None),
                ):
                    adapters = list(
                        backend.list_provider_adapters(
                            model="AbstractFramework/qwen-image-edit-2511-8bit",
                            task="image_to_image",
                        )
                    )

        self.assertEqual(
            [item.id for item in adapters],
            [
                "lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
            ],
        )
        self.assertEqual(
            adapters[0].raw["recommended_parameters"],
            {"steps": 4, "guidance_scale": 1.0},
        )
        self.assertEqual(adapters[0].raw["recommended_local_quantization_bits"], [8])
        self.assertIn(
            "https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning#-using-lightning-loras-with-fp8-models",
            adapters[0].raw["documentation_urls"],
        )
        self.assertTrue(adapters[0].raw["compatibility_notes"])

    def test_provider_adapter_inventory_classifies_legacy_qwen_edit_lightning_by_path(self):
        from types import SimpleNamespace

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        fake_capabilities = SimpleNamespace(
            capabilities=(
                SimpleNamespace(
                    public_task="image-to-image",
                    default_for_task=True,
                    supports_lora=True,
                    lora_status="validated",
                    lora_target_roles=("transformer",),
                    lora_validation_profile="qwen_edit_legacy_lightning_profile",
                ),
            )
        )

        with tempfile.TemporaryDirectory() as cache_td:
            cache_root = Path(cache_td)
            self._make_cache_snapshot(cache_root, "AbstractFramework/qwen-image-edit-8bit")
            self._make_adapter_snapshot(
                cache_root,
                "lightx2v/Qwen-Image-Lightning",
                "Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors",
                base_model="Qwen/Qwen-Image",
            )

            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/qwen-image-edit-8bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                    return_value=(lambda **kwargs: fake_capabilities, None),
                ):
                    adapters = list(
                        backend.list_provider_adapters(
                            model="AbstractFramework/qwen-image-edit-8bit",
                            task="image_to_image",
                        )
                    )

        self.assertEqual(len(adapters), 1)
        self.assertEqual(
            adapters[0].raw["compatible_base_model"],
            "qwen-image-edit",
        )
        self.assertEqual(
            adapters[0].raw["recommended_parameters"],
            {"steps": 8, "guidance_scale": 1.0},
        )
        self.assertEqual(adapters[0].raw["recommended_local_quantization_bits"], [8])

    def test_provider_adapter_inventory_skips_full_model_components(self):
        from types import SimpleNamespace

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        fake_capabilities = SimpleNamespace(
            capabilities=(
                SimpleNamespace(
                    public_task="text-to-image",
                    default_for_task=True,
                    supports_lora=True,
                    lora_status="validated",
                    lora_target_roles=("transformer",),
                    lora_validation_profile="ernie_q8_profile",
                ),
            )
        )

        with tempfile.TemporaryDirectory() as cache_td:
            cache_root = Path(cache_td)
            model_snap = self._make_cache_snapshot(
                cache_root, "AbstractFramework/ernie-image-turbo-8bit"
            )
            (model_snap / "README.md").write_text(
                "---\nbase_model: baidu/ERNIE-Image-Turbo\n---\n",
                encoding="utf-8",
            )
            (model_snap / "text_encoder").mkdir(exist_ok=True)
            (model_snap / "text_encoder" / "0.safetensors").write_bytes(b"x")
            (model_snap / "vae").mkdir(exist_ok=True)
            (model_snap / "vae" / "0.safetensors").write_bytes(b"x")
            self._make_adapter_snapshot(
                cache_root,
                "reverentelusarca/ernie-image-elusarca-anime-style-lora",
                "ernie-anime-v1.safetensors",
                base_model="baidu/ERNIE-Image-Turbo",
            )

            backend = MFluxVisionBackend(config=MFluxBackendConfig())

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                    return_value=(lambda **kwargs: fake_capabilities, None),
                ):
                    adapters = list(
                        backend.list_provider_adapters(
                            model="AbstractFramework/ernie-image-turbo-8bit",
                            task="text_to_image",
                        )
                    )

        self.assertEqual(
            [item.id for item in adapters],
            [
                "reverentelusarca/ernie-image-elusarca-anime-style-lora:ernie-anime-v1.safetensors"
            ],
        )

    def test_provider_catalog_exposes_cached_mlx_gen_q4_and_q8_variants(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-4bit")
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-4bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                models = {
                    model.id: model for model in backend.list_provider_models(task="text_to_image")
                }

        self.assertIn("AbstractFramework/flux.2-klein-4b-4bit", models)
        self.assertIn("AbstractFramework/flux.2-klein-4b-8bit", models)
        self.assertEqual(
            models["AbstractFramework/flux.2-klein-4b-4bit"].raw["quantization_bits"], 4
        )
        self.assertEqual(
            models["AbstractFramework/flux.2-klein-4b-4bit"].raw["repo_id"],
            "AbstractFramework/flux.2-klein-4b-4bit",
        )
        self.assertEqual(
            models["AbstractFramework/flux.2-klein-4b-8bit"].raw["quantization_bits"], 8
        )
        self.assertEqual(
            models["AbstractFramework/flux.2-klein-4b-8bit"].raw["repo_id"],
            "AbstractFramework/flux.2-klein-4b-8bit",
        )

    def test_ernie_image_provider_catalog_advertises_text_and_image_to_image(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/ernie-image-turbo-4bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/ernie-image-turbo-4bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                models = list(backend.list_provider_models())
                image_edit_models = list(backend.list_provider_models(task="image_to_image"))

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "AbstractFramework/ernie-image-turbo-4bit")
        self.assertEqual(tuple(models[0].capabilities), ("image_to_image", "text_to_image"))
        self.assertEqual(models[0].raw["quantization_bits"], 4)
        self.assertEqual(len(image_edit_models), 1)
        self.assertEqual(image_edit_models[0].id, "AbstractFramework/ernie-image-turbo-4bit")

    def test_qwen_edit_model_family_routes_image_to_image(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(
                Path(cache_td), "AbstractFramework/qwen-image-edit-2511-4bit"
            )
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/qwen-image-edit-2511-4bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_qwen",
                        return_value=(_FakeQwenImage, _FakeQwenImageEdit),
                    ):
                        asset = backend.edit_image(
                            ImageEditRequest(prompt="watercolor", image=PNG_1X1, seed=123)
                        )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeQwenImageEdit.last_init["model_config"], "qwen-image-edit-2511-config")
        self.assertEqual(_FakeQwenImageEdit.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeQwenImageEdit.last_generate["prompt"], "watercolor")
        self.assertEqual(len(_FakeQwenImageEdit.last_generate["image_paths"]), 1)
        self.assertTrue(_FakeQwenImageEdit.last_generate["image_paths"][0])

    def test_qwen_edit_2509_prefers_exact_model_config_from_name(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(
                Path(cache_td), "AbstractFramework/qwen-image-edit-2509-8bit"
            )
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/qwen-image-edit-2509-8bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_qwen",
                        return_value=(_FakeQwenImage, _FakeQwenImageEdit),
                    ):
                        asset = backend.edit_image(
                            ImageEditRequest(prompt="rotate", image=PNG_1X1, seed=123)
                        )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeQwenImageEdit.last_init["model_config"], "qwen-image-edit-2509-config")
        self.assertEqual(_FakeQwenImageEdit.last_init["model_path"], str(snapshot))

    def test_qwen_edit_accepts_additional_reference_images(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(
                Path(cache_td), "AbstractFramework/qwen-image-edit-2511-4bit"
            )
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/qwen-image-edit-2511-4bit")
            )

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_qwen",
                        return_value=(_FakeQwenImage, _FakeQwenImageEdit),
                    ):
                        asset = backend.edit_image(
                            ImageEditRequest(
                                prompt="combine references",
                                image=PNG_1X1,
                                seed=123,
                                extra={"reference_images": [_solid_png(2, 2, "blue")]},
                            )
                        )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(len(_FakeQwenImageEdit.last_generate["image_paths"]), 2)
        self.assertEqual(asset.metadata["reference_image_count"], 2)
        self.assertEqual(asset.metadata["edit_mode"], "multi_reference")
        for image_path in _FakeQwenImageEdit.last_generate["image_paths"]:
            self.assertFalse(Path(image_path).exists())

    def test_fibo_edit_rejects_additional_reference_images(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageEditRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "briaai/Fibo-Edit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="briaai/Fibo-Edit"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    with patch(
                        "abstractvision.backends.mflux._lazy_import_mflux_fibo",
                        return_value=(_FakeFIBO, _FakeFIBOEdit),
                    ):
                        with self.assertRaises(CapabilityNotSupportedError):
                            backend.edit_image(
                                ImageEditRequest(
                                    prompt="remove background",
                                    image=PNG_1X1,
                                    extra={"reference_images": [PNG_1X1]},
                                )
                            )

    def test_canonical_mlx_gen_exports_alias_compatibility_backend(self):
        from abstractvision.backends import (
            MLXGenBackendConfig,
            MLXGenVisionBackend,
            MFluxBackendConfig,
            MFluxVisionBackend,
        )

        self.assertIs(MLXGenBackendConfig, MFluxBackendConfig)
        self.assertIs(MLXGenVisionBackend, MFluxVisionBackend)

    def test_flux2_klein_capabilities_are_text_to_image_only(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        backend = MFluxVisionBackend(
            config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-4bit")
        )
        caps = backend.get_capabilities()

        self.assertEqual(caps.supported_tasks, ["image_to_image", "text_to_image"])
        self.assertFalse(caps.supports_mask)

    def test_flux2_edit_image_uses_edit_variant_and_input_dimensions(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest

        _FakeFlux2.last_generate = None
        _FakeFlux2Edit.last_generate = None
        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-4bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-4bit")
            )
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    asset = backend.edit_image(
                        ImageEditRequest(
                            prompt="watercolor",
                            image=PNG_1X1,
                            extra={"strength": 0.75},
                            guidance_scale=2.0,
                            seed=123,
                        )
                    )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertIsNone(_FakeFlux2.last_generate)
        self.assertIsNotNone(_FakeFlux2Edit.last_generate)
        self.assertIn("image_paths", _FakeFlux2Edit.last_generate)
        self.assertEqual(len(_FakeFlux2Edit.last_generate["image_paths"]), 1)
        self.assertNotIn("image_strength", _FakeFlux2Edit.last_generate)
        self.assertEqual(_FakeFlux2Edit.last_generate["guidance"], 2.0)
        self.assertEqual(_FakeFlux2Edit.last_generate["width"], 1)
        self.assertEqual(_FakeFlux2Edit.last_generate["height"], 1)
        self.assertNotIn("image_strength", asset.metadata)

    def test_flux2_edit_accepts_additional_reference_images_and_progress(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest, VideoProgressEvent

        seen = []
        _FakeFlux2.last_generate = None
        _FakeFlux2Edit.last_generate = None
        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-9b-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-9b-8bit")
            )
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    asset = backend.edit_image(
                        ImageEditRequest(
                            prompt="combine references",
                            image=PNG_1X1,
                            extra={
                                "reference_images": [_solid_png(2, 2, "blue")],
                                "on_progress": seen.append,
                            },
                            guidance_scale=1.0,
                            seed=123,
                        )
                    )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertIsNone(_FakeFlux2.last_generate)
        self.assertIsNotNone(_FakeFlux2Edit.last_generate)
        self.assertEqual(len(_FakeFlux2Edit.last_generate["image_paths"]), 2)
        self.assertEqual(asset.metadata["reference_image_count"], 2)
        self.assertEqual(asset.metadata["edit_mode"], "multi_reference")
        self.assertEqual([event.phase for event in seen], ["start", "denoise", "complete"])
        self.assertTrue(all(isinstance(event, VideoProgressEvent) for event in seen))
        self.assertEqual([event.task for event in seen], ["image_to_image"] * 3)
        for image_path in _FakeFlux2Edit.last_generate["image_paths"]:
            self.assertFalse(Path(image_path).exists())

    def test_runtime_serializes_model_init_and_generate_on_same_thread(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-4bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-4bit")
            )
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(flux_cls=_ThreadAwareFlux2),
                ):
                    backend.preload()
                    caller_thread = threading.get_ident()
                    result = {}

                    def invoke() -> None:
                        asset = backend.generate_image(ImageGenerationRequest(prompt="hello"))
                        result["asset"] = asset
                        result["caller_thread"] = threading.get_ident()

                    t = threading.Thread(target=invoke)
                    t.start()
                    t.join()

        self.assertTrue(result["asset"].data.startswith(b"\x89PNG"))
        self.assertNotEqual(_ThreadAwareFlux2.init_thread, caller_thread)
        self.assertNotEqual(_ThreadAwareFlux2.generate_thread, result["caller_thread"])
        self.assertEqual(_ThreadAwareFlux2.init_thread, _ThreadAwareFlux2.generate_thread)

    def test_preload_runs_generate_warmup_once_per_loaded_model(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        _CountingFlux2.init_calls = 0
        _CountingFlux2.generate_calls = []

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-4bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(
                    model="AbstractFramework/flux.2-klein-4b-4bit",
                    default_width=320,
                    default_height=192,
                )
            )
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(flux_cls=_CountingFlux2),
                ):
                    backend.preload()
                    backend.preload()

        self.assertEqual(_CountingFlux2.init_calls, 1)
        self.assertEqual(len(_CountingFlux2.generate_calls), 1)
        warmup = _CountingFlux2.generate_calls[0]
        self.assertEqual(warmup["prompt"], "abstractvision preload warmup")
        self.assertEqual(warmup["width"], 320)
        self.assertEqual(warmup["height"], 192)
        self.assertEqual(warmup["num_inference_steps"], 4)
        self.assertEqual(warmup["guidance"], 1.0)
        self.assertEqual(warmup["seed"], 0)


    def _memory_patches(self, stack, cache_td, *, caps, flux_cls=_LowRamFlux2):
        _LowRamFlux2.instances = []
        _LowRamFlux2.during_generate = None
        _FakeMemorySaver.instances = []
        _MEMORY_EVENTS.clear()
        limit = [_CAP_IN_FORCE]

        def fake_set_limit(value):
            previous, limit[0] = limit[0], int(value)
            _MEMORY_EVENTS.append(("limit", int(value)))
            return previous

        stack.enter_context(
            patch.dict("os.environ", {"HF_HUB_CACHE": cache_td, "ABSTRACTVISION_MEMORY_MODE": ""}, clear=True)
        )
        stack.enter_context(
            patch(
                "abstractvision.backends.mflux._lazy_import_mflux",
                return_value=self._lazy_import_return(flux_cls=flux_cls, flux_edit_cls=flux_cls),
            )
        )
        stack.enter_context(
            patch(
                "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                return_value=(lambda **kwargs: caps, None),
            )
        )
        stack.enter_context(
            patch(
                "abstractvision.backends.mflux._lazy_import_mlx_gen_memory_saver",
                return_value=_FakeMemorySaver,
            )
        )
        stack.enter_context(
            patch("abstractvision.backends.mflux._set_mlx_cache_limit", side_effect=fake_set_limit)
        )
        stack.enter_context(patch("abstractvision.backends.mflux._clear_mlx_cache"))
        stack.enter_context(patch("abstractvision.backends.mflux._apply_mlx_gen_default_cache_limit"))
        return limit

    def _run_flux2_with_memory_modes(self, requests, *, block_streaming=False, flux_cls=_LowRamFlux2):
        import contextlib

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        assets = []
        caps = _route_capabilities("text-to-image", "image-to-image", block_streaming=block_streaming)
        with tempfile.TemporaryDirectory() as cache_td, contextlib.ExitStack() as stack:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-8bit")
            )
            self._memory_patches(stack, cache_td, caps=caps, flux_cls=flux_cls)
            self._last_low_ram_backend = backend
            capabilities = backend.get_capabilities()
            for request in requests:
                if callable(request):
                    assets.append(request(backend))
                elif hasattr(request, "image"):
                    assets.append(backend.edit_image(request))
                else:
                    assets.append(backend.generate_image(request))
        return backend, assets, capabilities

    def test_memory_mode_low_runs_image_route_in_mlx_gen_low_ram_mode(self):
        from abstractvision.backends.mflux import LOW_RAM_MLX_CACHE_LIMIT_BYTES
        from abstractvision.types import ImageGenerationRequest

        backend, assets, capabilities = self._run_flux2_with_memory_modes(
            [
                ImageGenerationRequest(prompt="fox", steps=4, seed=1, memory_mode="low"),
                ImageGenerationRequest(prompt="fox", steps=4, seed=2),
            ]
        )

        self.assertEqual(tuple(capabilities.memory_modes), ("default", "low"))
        # As `mlxgen generate --low-ram` (flux2_generate.py): the 1 GB cache is set before the
        # model loads, then one MemorySaver frees the text encoders and, for one seed, the transformer.
        self.assertEqual(
            _MEMORY_EVENTS[:3], [("limit", LOW_RAM_MLX_CACHE_LIMIT_BYTES), ("init", None), ("limit", _CAP_IN_FORCE)]
        )
        self.assertEqual(len(_FakeMemorySaver.instances), 1)
        saver = _FakeMemorySaver.instances[0]
        self.assertFalse(saver.keep_transformer)
        self.assertEqual(saver.cache_limit_bytes, LOW_RAM_MLX_CACHE_LIMIT_BYTES)
        self.assertIs(saver.model, _LowRamFlux2.instances[0])
        self.assertEqual(_LowRamFlux2.instances[0].callbacks.registered, [saver])
        # The released instance is rebuilt for the next (default) request, which runs as before.
        self.assertEqual(len(_LowRamFlux2.instances), 2)
        self.assertEqual(_LowRamFlux2.instances[1].callbacks.registered, [])
        self.assertNotIn("stream_blocks", _FakeFlux2.last_init)
        self.assertEqual(
            {key: assets[0].metadata[key] for key in ("memory_mode", "low_ram", "stream_blocks")},
            {"memory_mode": "low", "low_ram": True, "stream_blocks": False},
        )
        self.assertEqual(assets[1].metadata["memory_mode"], "default")
        self.assertTrue(backend.has_loaded_model())

    def test_memory_mode_lowest_is_refused_where_the_route_cannot_stream_blocks(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageEditRequest

        with self.assertRaisesRegex(CapabilityNotSupportedError, r"memory_mode='lowest'.*supported: default, low"):
            self._run_flux2_with_memory_modes(
                [ImageEditRequest(prompt="edit", image=PNG_1X1, steps=4, seed=1, memory_mode="lowest")]
            )
        self.assertEqual(_LowRamFlux2.instances, [])
        self.assertEqual(_FakeMemorySaver.instances, [])
        self.assertEqual(_MEMORY_EVENTS, [])

    def test_memory_mode_lowest_streams_blocks_where_the_route_publishes_it(self):
        from abstractvision.types import ImageGenerationRequest

        backend, assets, capabilities = self._run_flux2_with_memory_modes(
            [ImageGenerationRequest(prompt="fox", steps=4, seed=1, memory_mode="lowest")],
            block_streaming=True,
        )

        self.assertEqual(tuple(capabilities.memory_modes), ("default", "low", "lowest"))
        self.assertIs(_FakeFlux2.last_init.get("stream_blocks"), True)
        self.assertEqual(len(_FakeMemorySaver.instances), 1)
        self.assertEqual(assets[0].metadata["memory_mode"], "lowest")
        self.assertTrue(assets[0].metadata["stream_blocks"])
        self.assertTrue(assets[0].metadata["low_ram"])
        self.assertFalse(backend.has_loaded_model())

    def test_memory_mode_default_sends_nothing_new(self):
        from abstractvision.types import ImageGenerationRequest

        _backend, assets, _caps = self._run_flux2_with_memory_modes(
            [ImageGenerationRequest(prompt="fox", steps=4, seed=1, memory_mode="default")],
            block_streaming=True,
        )

        self.assertEqual(_FakeMemorySaver.instances, [])
        self.assertEqual(_MEMORY_EVENTS, [("init", None)])
        self.assertNotIn("stream_blocks", _FakeFlux2.last_init)
        # The mode is recorded on every run, "default" included (AbstractCore reads memory_mode_used from it).
        self.assertEqual(
            {key: assets[0].metadata[key] for key in ("memory_mode", "low_ram", "stream_blocks")},
            {"memory_mode": "default", "low_ram": False, "stream_blocks": False},
        )

    def test_memory_mode_low_failed_run_still_drops_the_model_and_restores_the_cache(self):
        from abstractvision.backends.mflux import LOW_RAM_MLX_CACHE_LIMIT_BYTES
        from abstractvision.types import ImageGenerationRequest

        class _FailingLowRamFlux2(_LowRamFlux2):
            def generate_image(self, **kwargs):
                raise RuntimeError("denoise failed")

        with self.assertRaisesRegex(RuntimeError, "denoise failed"):
            self._run_flux2_with_memory_modes(
                [ImageGenerationRequest(prompt="fox", steps=4, memory_mode="low")],
                flux_cls=_FailingLowRamFlux2,
            )
        # The failed instance carried a MemorySaver: it must not be reused.
        self.assertEqual(len(_FakeMemorySaver.instances), 1)
        self.assertEqual(_MEMORY_EVENTS[-1], ("limit", _CAP_IN_FORCE))
        self.assertEqual(_MEMORY_EVENTS[0], ("limit", LOW_RAM_MLX_CACHE_LIMIT_BYTES))
        self.assertFalse(self._last_low_ram_backend.has_loaded_model())

    def test_memory_mode_low_batch_items_are_runs_like_any_other(self):
        from abstractvision import VisionManager

        def batch(backend):
            return VisionManager(backend=backend).generate_image_batch(
                "fox", count=3, seed=1, steps=4, memory_mode="low"
            )

        backend, assets, _caps = self._run_flux2_with_memory_modes([batch])

        # Each item is `mlxgen generate --low-ram` for one seed: its own load, a MemorySaver that
        # frees the text encoders and the transformer, and the instance dropped after it.
        self.assertEqual(len(_LowRamFlux2.instances), 3)
        self.assertEqual([saver.keep_transformer for saver in _FakeMemorySaver.instances], [False] * 3)
        self.assertEqual([model.callbacks.registered for model in _LowRamFlux2.instances],
                         [[saver] for saver in _FakeMemorySaver.instances])
        self.assertEqual(len(assets[0]), 3)
        self.assertFalse(backend.has_loaded_model())

    def test_one_release_rule_for_every_family(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        # (base model, mode) -> the run frees part of the instance, which is then dropped
        cases = {
            ("flux2-klein-4b", "default"): False,
            ("flux2-klein-4b", "low"): True,
            ("fibo", "low"): True,
            ("wan2.2-t2v-a14b", "default"): False,
            ("wan2.2-t2v-a14b", "low"): True,
            ("wan2.2-ti2v-5b", "default"): True,
            ("minimax-h3-turbo-544p", "default"): False,
            ("minimax-h3-turbo-544p", "low"): True,
            ("minimax-h3-turbo-544p", "lowest"): True,
            ("seedvr2-3b", "low"): False,
        }
        for (base_model, mode), expected in cases.items():
            with self.subTest(base_model=base_model, mode=mode):
                backend = MFluxVisionBackend(config=MFluxBackendConfig(model=base_model, base_model=base_model))
                # The plan is built from a checked memory_mode and the resolved model.
                with patch.object(
                    MFluxVisionBackend, "_find_configured_model", return_value=("/m", base_model, None)
                ):
                    plan = backend._memory_plan(mode)
                self.assertEqual(plan.releases_model, expected)

    def test_preload_warmup_ignores_a_process_low_memory_default(self):
        import contextlib

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td, contextlib.ExitStack() as stack:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-8bit"))
            self._memory_patches(stack, cache_td, caps=_route_capabilities("text-to-image"))
            stack.enter_context(patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": "low"}))
            backend.preload()
        self.assertEqual(_FakeMemorySaver.instances, [])
        self.assertTrue(backend.has_loaded_model())

    def test_overlapping_runs_share_one_cache_limit_and_restore_it_last(self):
        import contextlib

        from abstractvision.backends.mflux import (
            LOW_RAM_MLX_CACHE_LIMIT_BYTES,
            MFluxBackendConfig,
            MFluxVisionBackend,
        )
        from abstractvision.types import ImageGenerationRequest, VideoGenerationRequest

        seen = {}
        with tempfile.TemporaryDirectory() as cache_td, contextlib.ExitStack() as stack:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-8bit")
            self._make_cache_snapshot(Path(cache_td), "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
            image_backend = MFluxVisionBackend(config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-8bit"))
            video_backend = MFluxVisionBackend(config=MFluxBackendConfig(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers"))
            limit = self._memory_patches(stack, cache_td, caps=_route_capabilities("text-to-image"))
            stack.enter_context(patch("abstractvision.backends.mflux._lazy_import_mflux_wan", return_value=_FakeWan))

            def runs_inside_the_low_image_run():
                # On another backend thread, a default Wan run (which holds no limit) and a low one
                # start and end while the low-RAM image run holds 1 GB: it stays until both end.
                video_backend.generate_video(VideoGenerationRequest(prompt="fox", seed=1))
                video_backend.generate_video(VideoGenerationRequest(prompt="fox", seed=2, memory_mode="low"))
                seen["after_video"] = limit[0]

            _LowRamFlux2.during_generate = runs_inside_the_low_image_run
            image_backend.generate_image(ImageGenerationRequest(prompt="fox", steps=4, memory_mode="low"))
            seen["after_both"] = limit[0]

        self.assertEqual(seen["after_video"], LOW_RAM_MLX_CACHE_LIMIT_BYTES)
        self.assertEqual(seen["after_both"], _CAP_IN_FORCE)
        limits = [value for kind, value in _MEMORY_EVENTS if kind == "limit"]
        self.assertEqual(limits[-1], _CAP_IN_FORCE)
        self.assertTrue(all(value <= LOW_RAM_MLX_CACHE_LIMIT_BYTES for value in limits[:-1]), limits)

    def test_memory_mode_low_on_wan_a14b_matches_the_single_seed_cli(self):
        from abstractvision.backends.mflux import LOW_RAM_MLX_CACHE_LIMIT_BYTES
        from abstractvision.types import VideoGenerationRequest

        backend, assets, events, _wan_cls = self._run_wan_videos_recording_memory_policy(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            _FakeWan,
            [VideoGenerationRequest(prompt="fox", seed=1, memory_mode="low")],
        )

        # wan_generate.py: release_denoisers_before_decode = low_ram and single_seed (75),
        # release_inactive_denoiser=True on single-seed dual-expert runs (83-89),
        # clear_cache_each_step / clear_cache_each_transformer_block = low_ram (145-149).
        low_kwargs = dict(_FakeWan.last_generate)
        for key in (
            "release_denoisers_before_decode",
            "release_inactive_denoiser",
            "clear_cache_each_step",
            "clear_cache_each_transformer_block",
        ):
            self.assertIs(low_kwargs.get(key), True, key)
        # The 1 GB limit is set before the model loads (wan_generate.py:58, the load is at :62).
        self.assertEqual(events[0], ("set_limit", LOW_RAM_MLX_CACHE_LIMIT_BYTES))
        self.assertLess(events.index(("set_limit", LOW_RAM_MLX_CACHE_LIMIT_BYTES)), events.index(("init", None)))
        self.assertIn(("save", LOW_RAM_MLX_CACHE_LIMIT_BYTES), events)
        self.assertEqual(assets[0].metadata["memory_mode"], "low")
        self.assertFalse(backend.has_loaded_model())

    def test_memory_mode_low_wan_batch_items_each_release(self):
        from abstractvision import VisionManager

        class _BatchWan(_FakeWan):
            calls = []

            def generate_video(self, **kwargs):
                _BatchWan.calls.append(dict(kwargs))
                return super().generate_video(**kwargs)

        _BatchWan.calls = []

        def batch(backend):
            return VisionManager(backend=backend).generate_video_batch(
                "fox", count=2, seed=1, memory_mode="low"
            )

        backend, _assets, _events, wan_cls = self._run_wan_videos_recording_memory_policy(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", _BatchWan, [batch]
        )

        # Each item is `mlxgen generate --low-ram` for one seed (wan_generate.py:75): both release
        # the denoisers (and the inactive expert), and each runs on its own instance.
        for call in _BatchWan.calls:
            self.assertIs(call.get("release_denoisers_before_decode"), True)
            self.assertIs(call.get("release_inactive_denoiser"), True)
            self.assertIs(call.get("clear_cache_each_step"), True)
        self.assertEqual(wan_cls.instances, 2)
        self.assertFalse(backend.has_loaded_model())

    def test_memory_mode_low_on_wan_ti2v_has_no_inactive_expert_to_release(self):
        from abstractvision.types import VideoGenerationRequest

        self._run_wan_videos_recording_memory_policy(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            _FakeWan,
            [VideoGenerationRequest(prompt="fox", width=832, height=480, num_frames=5, steps=2, seed=1, extra={"memory_mode": "low"})],
        )
        self.assertNotIn("release_inactive_denoiser", _FakeWan.last_generate)
        self.assertNotIn("stream_blocks", _FakeWan.last_init)
        self.assertIs(_FakeWan.last_generate.get("clear_cache_each_transformer_block"), True)

    def test_memory_mode_low_on_seedvr2_upscale_is_the_cache_limit_only(self):
        import contextlib

        from abstractvision.backends.mflux import (
            LOW_RAM_MLX_CACHE_LIMIT_BYTES,
            MFluxBackendConfig,
            MFluxVisionBackend,
        )
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageUpscaleRequest

        with tempfile.TemporaryDirectory() as cache_td, contextlib.ExitStack() as stack:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/seedvr2-3b-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="seedvr2-3b"))
            self._memory_patches(stack, cache_td, caps=_route_capabilities("image-upscale", block_streaming=True))
            stack.enter_context(
                patch(
                    "abstractvision.backends.mflux._lazy_import_mflux_seedvr2",
                    return_value=(_FakeModelConfig, _LowRamSeedVR2, _FakeScaleFactor, _FakeTilingConfig),
                )
            )
            self.assertEqual(tuple(backend.supported_memory_modes("image_upscale")), ("default", "low"))
            first = backend.upscale_image(ImageUpscaleRequest(image=PNG_1X1, seed=1, memory_mode="low"))
            instance = _FakeSeedVR2.last_instance
            second = backend.upscale_image(ImageUpscaleRequest(image=PNG_1X1, seed=2, memory_mode="low"))
            with self.assertRaises(CapabilityNotSupportedError):
                backend.upscale_image(ImageUpscaleRequest(image=PNG_1X1, memory_mode="lowest"))

        # seedvr2_upscale.py:1031-1053: images get the 1 GB cache (apply_runtime_memory_options),
        # no MemorySaver, and SeedVR2's own tiling. The instance is reused.
        self.assertEqual(_FakeMemorySaver.instances, [])
        self.assertEqual(instance.tiling_config, "unset")
        self.assertIs(_FakeSeedVR2.last_instance, instance)
        self.assertEqual(
            _MEMORY_EVENTS,
            [
                ("limit", LOW_RAM_MLX_CACHE_LIMIT_BYTES),
                ("init", None),
                ("limit", _CAP_IN_FORCE),
                ("limit", LOW_RAM_MLX_CACHE_LIMIT_BYTES),
                ("limit", _CAP_IN_FORCE),
            ],
        )
        self.assertEqual(first.metadata["memory_mode"], "low")
        self.assertEqual(second.metadata["vae_tiling_policy"], "runtime_default")
        self.assertTrue(backend.has_loaded_model())

    def test_provider_catalog_lists_the_memory_modes_each_model_can_honour(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        for block_streaming, modes in ((False, ["default", "low"]), (True, ["default", "low", "lowest"])):
            caps = _route_capabilities("text-to-image", block_streaming=block_streaming)
            with tempfile.TemporaryDirectory() as cache_td:
                self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-8bit")
                backend = MFluxVisionBackend(
                    config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-8bit")
                )
                with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True), patch(
                    "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                    return_value=(lambda caps=caps, **kwargs: caps, None),
                ):
                    models = [
                        model
                        for model in backend.list_provider_models(task="text_to_image")
                        if model.id == "AbstractFramework/flux.2-klein-4b-8bit"
                    ]

            self.assertEqual(len(models), 1)
            raw = models[0].raw
            self.assertEqual(raw["parameter_defaults"]["memory_mode"], "default")
            # The modes alone say what the model can do; no engine term beside them.
            self.assertEqual(raw["parameter_constraints"]["memory_mode"], {"enum": modes})
            self.assertEqual(raw["parameters"]["constraints"]["memory_mode"]["enum"], modes)

    # --- the resolved model, load-time quantize, preload, entry points ---------------------------

    def _state_patches(self, stack, cache_td, *, caps):
        limit = self._memory_patches(stack, cache_td, caps=caps)
        # Only the test's own cache: no framework root on this machine is scanned.
        stack.enter_context(patch("abstractvision.model_cache.framework_candidate_roots", return_value=[]))
        stack.enter_context(
            patch("abstractvision.backends.mflux._mlx_gen_quantize_choices", return_value=(3, 4, 5, 6, 8))
        )
        stack.enter_context(patch("abstractvision.backends.mflux._lazy_import_mflux_wan", return_value=_FakeWan))
        stack.enter_context(
            patch(
                "abstractvision.backends.mflux._lazy_import_mflux_qwen",
                return_value=(_FakeQwenImage, _FakeQwenImageEdit),
            )
        )
        stack.enter_context(
            patch(
                "abstractvision.backends.mflux._lazy_import_mflux_seedvr2",
                return_value=(_FakeModelConfig, _LowRamSeedVR2, _FakeScaleFactor, _FakeTilingConfig),
            )
        )
        _FakeWan.last_generate = None
        _FakeSeedVR2.last_init = None
        _FakeQwenImageEdit.last_init = None
        _FakeQwenImageEdit.last_generate = None
        return limit

    def test_load_time_quantize_is_recorded_on_every_request_and_after_preload(self):
        import contextlib

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(stack, td, caps=_route_capabilities("text-to-image"))
            source = self._make_model_dir(Path(td), "klein-source")
            config = MFluxBackendConfig(model=str(source), base_model="flux2-klein-4b")
            backend = MFluxVisionBackend(config=config)
            bits = [
                backend.generate_image(
                    ImageGenerationRequest(prompt="fox", steps=2, seed=seed, quantize=8)
                ).metadata["quantization_bits"]
                for seed in range(3)
            ]
            self.assertEqual(bits, [8, 8, 8])
            self.assertEqual(len(_LowRamFlux2.instances), 1)
            self.assertEqual(_FakeFlux2.last_init["quantize"], 8)
            unquantized = backend.generate_image(ImageGenerationRequest(prompt="fox", steps=2, seed=4))
            self.assertIsNone(unquantized.metadata["quantization_bits"])

            preloaded = MFluxVisionBackend(config=config)
            preloaded.preload(quantize=8)
            bits = [
                preloaded.generate_image(
                    ImageGenerationRequest(prompt="fox", steps=2, seed=seed, quantize=8)
                ).metadata["quantization_bits"]
                for seed in range(2)
            ]
            self.assertEqual(bits, [8, 8])
            self.assertEqual(len(_LowRamFlux2.instances), 3)

    def test_the_memory_plan_reads_the_resolved_model_on_a_backend_with_none_configured(self):
        import contextlib

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageUpscaleRequest, VideoGenerationRequest

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(stack, td, caps=_route_capabilities("text-to-video"))
            empty = Path(td) / "no-local-models"
            empty.mkdir()
            self._make_cache_snapshot(Path(td), "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model_dir=str(empty)))
            for seed in range(2):
                with self.subTest(request=seed):
                    backend.generate_video(
                        VideoGenerationRequest(
                            prompt="a paper boat", width=832, height=480, num_frames=5, steps=2, seed=seed,
                            extra={"vae_tiling": False},
                        )
                    )
                    # Every Wan TI2V-5B run releases its denoiser before the decode and drops the model.
                    self.assertTrue(_FakeWan.last_generate.get("release_denoisers_before_decode"))
                    self.assertFalse(backend.has_loaded_model())

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(stack, td, caps=_route_capabilities("image-upscale"))
            empty = Path(td) / "no-local-models"
            empty.mkdir()
            self._make_cache_snapshot(Path(td), "AbstractFramework/seedvr2-3b-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model_dir=str(empty)))
            for seed in range(2):
                backend.upscale_image(ImageUpscaleRequest(image=PNG_1X1, seed=seed, memory_mode="low"))
                # SeedVR2's low mode is the cache limit alone: it keeps its model.
                self.assertTrue(backend.has_loaded_model(), seed)

    def test_preload_of_an_edit_only_model_loads_it_for_edits_without_a_warm_up(self):
        import contextlib

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(stack, td, caps=_route_capabilities("image-to-image"))
            self._make_cache_snapshot(Path(td), "AbstractFramework/qwen-image-edit-2511-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="AbstractFramework/qwen-image-edit-2511-8bit")
            )
            backend.preload()
            self.assertTrue(backend.has_loaded_model())
            self.assertIsNotNone(_FakeQwenImageEdit.last_init)
            self.assertIsNone(_FakeQwenImageEdit.last_generate)
            _FakeQwenImageEdit.last_init = None
            backend.edit_image(ImageEditRequest(prompt="a red door", image=PNG_1X1, steps=2, seed=1))
            self.assertIsNone(_FakeQwenImageEdit.last_init)  # the preloaded instance served the edit
            self.assertIsNotNone(_FakeQwenImageEdit.last_generate)

    def test_preload_in_low_loads_under_the_cache_hold_runs_no_warm_up_and_serves_one_request(self):
        import contextlib

        from abstractvision.backends.mflux import (
            LOW_RAM_MLX_CACHE_LIMIT_BYTES,
            MFluxBackendConfig,
            MFluxVisionBackend,
        )
        from abstractvision.types import ImageGenerationRequest

        for process_mode, preload_mode in (("", "low"), ("low", None)):
            with self.subTest(process_mode=process_mode, preload_mode=preload_mode):
                with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
                    self._state_patches(stack, td, caps=_route_capabilities("text-to-image"))
                    stack.enter_context(patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": process_mode}))
                    self._make_cache_snapshot(Path(td), "AbstractFramework/flux.2-klein-4b-8bit")
                    backend = MFluxVisionBackend(
                        config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-8bit")
                    )
                    _FakeFlux2.last_generate = None
                    backend.preload(memory_mode=preload_mode)
                    # The load ran under the 1 GB hold, restored after it; no generation ran.
                    self.assertEqual(
                        _MEMORY_EVENTS,
                        [("limit", LOW_RAM_MLX_CACHE_LIMIT_BYTES), ("init", None), ("limit", _CAP_IN_FORCE)],
                    )
                    self.assertIsNone(_FakeFlux2.last_generate)
                    self.assertEqual(_FakeMemorySaver.instances, [])
                    self.assertTrue(backend.has_loaded_model())
                    asset = backend.generate_image(ImageGenerationRequest(prompt="fox", steps=2, memory_mode="low"))
                    self.assertEqual(asset.metadata["memory_mode"], "low")
                    self.assertEqual(len(_LowRamFlux2.instances), 1)  # the first low request reused it
                    self.assertFalse(backend.has_loaded_model())  # and dropped it

    def test_load_time_quantize_on_seedvr2_goes_through_the_one_check(self):
        import contextlib

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageUpscaleRequest

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(stack, td, caps=_route_capabilities("image-upscale"))
            self._make_cache_snapshot(Path(td), "AbstractFramework/seedvr2-3b-8bit")
            prepared = MFluxVisionBackend(config=MFluxBackendConfig(model="AbstractFramework/seedvr2-3b-8bit"))
            self.assertEqual(prepared.get_capabilities().quantize_bits, ())
            for run in (
                lambda: prepared.upscale_image(ImageUpscaleRequest(image=PNG_1X1, quantize=4)),
                lambda: prepared.preload(quantize=4),
            ):
                with self.assertRaises(CapabilityNotSupportedError) as caught:
                    run()
                self.assertEqual((caught.exception.field, caught.exception.supported), ("quantize", ()))
            self.assertIsNone(_FakeSeedVR2.last_init)

            source = MFluxVisionBackend(
                config=MFluxBackendConfig(model=str(self._make_model_dir(Path(td), "upscaler")), base_model="seedvr2-3b")
            )
            for run in (
                lambda: source.upscale_image(ImageUpscaleRequest(image=PNG_1X1, quantize=7)),
                lambda: source.preload(quantize=7),
            ):
                with self.assertRaises(CapabilityNotSupportedError) as caught:
                    run()
                self.assertEqual((caught.exception.field, caught.exception.supported), ("quantize", (3, 4, 5, 6, 8)))
            self.assertIsNone(_FakeSeedVR2.last_init)
            asset = source.upscale_image(ImageUpscaleRequest(image=PNG_1X1, quantize=4))
            self.assertEqual((_FakeSeedVR2.last_init["quantize"], asset.metadata["quantization_bits"]), (4, 4))

    def test_with_progress_entry_points_check_a_request_once(self):
        import contextlib

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest, ImageUpscaleRequest

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(stack, td, caps=_route_capabilities("image-upscale"))
            source = MFluxVisionBackend(
                config=MFluxBackendConfig(model=str(self._make_model_dir(Path(td), "upscaler")), base_model="seedvr2-3b")
            )
            checks = []
            original = MFluxVisionBackend._normalize_image_upscale

            def counted(backend, request):
                checks.append(request)
                return original(backend, request)

            stack.enter_context(patch.object(MFluxVisionBackend, "_normalize_image_upscale", counted))
            steps = []
            request = ImageUpscaleRequest(image=PNG_1X1, quantize="none")
            source.upscale_image(request)
            plain = _FakeSeedVR2.last_init["quantize"]
            source.unload()
            source.upscale_image_with_progress(request, progress_callback=lambda *a: steps.append(a))
            # The same result whichever entry point runs it, one check each.
            self.assertEqual((plain, _FakeSeedVR2.last_init["quantize"]), (None, None))
            self.assertEqual(len(checks), 2)
            self.assertTrue(steps)
            # A request a caller already normalized (as VisionManager does) is not checked again.
            normalized = source.normalize_image_upscale_request(ImageUpscaleRequest(image=PNG_1X1))
            source.upscale_image_with_progress(normalized, progress_callback=lambda *a: None)
            self.assertEqual(len(checks), 3)

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(stack, td, caps=_route_capabilities("text-to-image"))
            self._make_cache_snapshot(Path(td), "AbstractFramework/flux.2-klein-4b-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="AbstractFramework/flux.2-klein-4b-8bit"))
            with patch.object(
                MFluxVisionBackend,
                "_normalize_image_generation",
                autospec=True,
                side_effect=MFluxVisionBackend._normalize_image_generation,
            ) as check:
                backend.generate_image_with_progress(
                    ImageGenerationRequest(prompt="fox", steps=2), progress_callback=lambda *a: None
                )
            self.assertEqual(check.call_count, 1)

    def test_unload_drops_the_resolution_and_capabilities_read_it_once(self):
        import contextlib

        from abstractvision.backends import mflux
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        repo = "AbstractFramework/flux.2-klein-4b-8bit"
        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(stack, td, caps=_route_capabilities("text-to-image"))
            first = self._make_cache_snapshot(Path(td), repo, "rev1")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model=repo))
            backend.generate_image(ImageGenerationRequest(prompt="fox", steps=2))
            self.assertEqual(_FakeFlux2.last_init["model_path"], str(first))
            reads = patch.object(mflux, "_is_prepared_mlx_gen_package", wraps=mflux._is_prepared_mlx_gen_package)
            with reads as counted:
                for _ in range(3):
                    backend.get_capabilities()
                backend.generate_image(ImageGenerationRequest(prompt="fox", steps=2))
            self.assertEqual(counted.call_count, 0)

            newer = self._make_cache_snapshot(Path(td), repo, "rev2")
            backend.generate_image(ImageGenerationRequest(prompt="fox", steps=2))
            self.assertEqual(_FakeFlux2.last_init["model_path"], str(first))  # resolved until unload
            backend.unload()
            backend.generate_image(ImageGenerationRequest(prompt="fox", steps=2))
            self.assertEqual(_FakeFlux2.last_init["model_path"], str(newer))

    def _installed(self, **versions):
        # importlib.metadata.version for the given distributions ("mlx_gen" stands for mlx-gen); others absent.
        from importlib import metadata

        installed = {name.replace("_", "-"): version for name, version in versions.items()}

        def version(distribution):
            if distribution not in installed:
                raise metadata.PackageNotFoundError(distribution)
            return installed[distribution]

        return patch("importlib.metadata.version", side_effect=version)

    @unittest.skipUnless(_PACKAGING_INSTALLED, "packaging is not installed (it comes with huggingface_hub)")
    def test_runtime_versions_outside_the_pins_are_refused_before_the_model_is_looked_for(self):
        import contextlib

        from abstractvision.backends import mflux
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import OptionalDependencyMissingError
        from abstractvision.types import ImageGenerationRequest, ImageUpscaleRequest

        self.addCleanup(mflux._require_runtime_versions.cache_clear)
        refused = {
            "mlx-gen too old": ("darwin", {"mlx_gen": "0.38.0", "mlx": "0.32.5"}, "mlx-gen>=0.40.1,<0.41.0", "0.38.0"),
            "mlx-gen a pre-release of the floor": ("linux", {"mlx_gen": "0.40.1rc1"}, "mlx-gen>=0.40.1,<0.41.0", "0.40.1rc1"),
            "mlx-gen too new": ("linux", {"mlx_gen": "0.41.0"}, "mlx-gen>=0.40.1,<0.41.0", "0.41.0"),
            "mlx-gen a pre-release of the ceiling": ("linux", {"mlx_gen": "0.41.0rc1"}, "mlx-gen>=0.40.1,<0.41.0", "0.41.0rc1"),
            "mlx too old on macOS": ("darwin", {"mlx_gen": "0.40.1", "mlx": "0.31.2"}, "mlx>=0.32.4 on macOS", "0.31.2"),
            "mlx a pre-release of the floor": ("darwin", {"mlx_gen": "0.40.1", "mlx": "0.32.4.dev20261101"}, "mlx>=0.32.4 on macOS", "0.32.4.dev20261101"),
        }
        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(stack, td, caps=_route_capabilities("text-to-image"))
            stack.enter_context(patch.object(mflux, "_require_runtime_versions", real_runtime_versions()))
            # Nothing is cached: without the check, these would answer that the model is not cached.
            calls = {
                "generate_image": lambda backend: backend.generate_image(ImageGenerationRequest(prompt="fox")),
                "preload": lambda backend: backend.preload(),
                "upscale_image": lambda backend: backend.upscale_image(ImageUpscaleRequest(image=PNG_1X1)),
            }
            for case, (platform, versions, required, installed) in refused.items():
                for call_name, call in calls.items():
                    with self.subTest(case=case, call=call_name):
                        mflux._require_runtime_versions.cache_clear()
                        model = "AbstractFramework/seedvr2-3b-8bit" if call_name == "upscale_image" else "AbstractFramework/flux.2-klein-4b-8bit"
                        backend = MFluxVisionBackend(config=MFluxBackendConfig(model=model))
                        with patch.object(mflux.sys, "platform", platform), self._installed(**versions):
                            with self.assertRaises(OptionalDependencyMissingError) as caught:
                                call(backend)
                        message = str(caught.exception)
                        self.assertIn(required, message)
                        self.assertIn(f" {installed} is installed", message)
                        self.assertIn('pip install -U "abstractvision[mlx-gen]"', message)
            accepted = {
                "in range": ("darwin", {"mlx_gen": "0.40.1", "mlx": "0.32.4"}),
                "in range, a later patch and its pre-release": ("darwin", {"mlx_gen": "0.40.2rc1", "mlx": "0.33.0"}),
                "old mlx off macOS": ("linux", {"mlx_gen": "0.40.1", "mlx": "0.31.2"}),
                "engine absent": ("darwin", {}),
            }
            for case, (platform, versions) in accepted.items():
                with self.subTest(case=case):
                    mflux._require_runtime_versions.cache_clear()
                    with patch.object(mflux.sys, "platform", platform), self._installed(**versions):
                        self.assertIsNone(mflux._require_runtime_versions())

    def test_a_catalog_read_needs_no_runtime_within_the_pins(self):
        # The schedule's adapter is read from mlx-gen's catalog, which an out-of-range engine still answers.
        import importlib

        from abstractvision.backends import mflux

        stack = patch.object(mflux, "_require_runtime_versions", real_runtime_versions())
        stack.start()
        self.addCleanup(stack.stop)

        try:
            importlib.import_module("mflux.models.common.config")
        except Exception:
            self.skipTest("mlx-gen is not installed")
        self.addCleanup(mflux._require_runtime_versions.cache_clear)
        mflux._require_runtime_versions.cache_clear()
        with patch.object(mflux.sys, "platform", "darwin"), self._installed(mlx_gen="0.38.0", mlx="0.31.2"):
            self.assertTrue(str(mflux._mlx_gen_schedule_adapter("minimax-h3-turbo")).startswith("hf:"))

    def test_the_runtime_bounds_are_the_ones_the_extras_require(self):
        from abstractvision.backends import mflux

        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        block = pyproject[pyproject.index("\nmlx-gen = [") : ]
        block = block[: block.index("\n]")]
        requirements = [line.strip().strip(",").strip('"') for line in block.splitlines()[1:] if line.strip()]
        bounds = {}
        for requirement in requirements:
            specifiers = requirement.split(";", 1)[0]
            name = specifiers.split(">=", 1)[0].strip()
            bounds[name] = {
                operator: value.strip()
                for part in specifiers[len(name):].split(",")
                for operator in (">=", "<")
                if part.strip().startswith(operator) and not part.strip().startswith("<=")
                for value in (part.strip()[len(operator):],)
            }
        self.assertEqual(bounds["mlx-gen"], {">=": mflux.MLX_GEN_VERSION_FLOOR, "<": mflux.MLX_GEN_VERSION_CEILING})
        self.assertEqual(bounds["mlx"], {">=": mflux.MLX_VERSION_FLOOR})

    def test_load_time_quantize_widths_are_the_engines_and_none_without_it(self):
        import sys
        import types

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as td:
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model=str(self._make_model_dir(Path(td), "klein")), base_model="flux2-klein-4b")
            )
            defaults = types.ModuleType("mflux.cli.defaults")
            defaults.defaults = SimpleNamespace(QUANTIZE_CHOICES=[3, 5, 4, 6, 8])
            engine = {"mflux": types.ModuleType("mflux"), "mflux.cli": types.ModuleType("mflux.cli")}
            with patch.object(MFluxVisionBackend, "_route_rows", return_value=[]):
                with patch.dict(sys.modules, {**engine, "mflux.cli.defaults": defaults}):
                    self.assertEqual(backend.quantize_bits(), (3, 4, 5, 6, 8))
                with patch.dict(sys.modules, {**engine, "mflux.cli.defaults": None}):
                    self.assertEqual(backend.quantize_bits(), ())

    def test_a_first_frame_may_be_any_bytes_like_object(self):
        from abstractvision.backends.mflux import _checked_keyframe_size

        png = _solid_png(3, 2, (10, 20, 30))
        for image in (png, bytearray(png), memoryview(png)):
            self.assertEqual(_checked_keyframe_size(image), (3, 2), type(image).__name__)
        for image in ("a.png", None, 3):
            with self.assertRaisesRegex(ValueError, "image must be the image file's bytes"):
                _checked_keyframe_size(image)

    def test_the_out_of_memory_message_reads_as_sentences(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend, _MemoryPlan

        backend = MFluxVisionBackend(config=MFluxBackendConfig())
        with patch.object(MFluxVisionBackend, "supported_memory_modes", return_value=("default", "low")):
            higher = backend._out_of_memory_error(
                MemoryError("needs 61 GB"), _MemoryPlan(mode="default", low_ram=False, stream_blocks=False), task="text_to_video"
            )
            lowest = backend._out_of_memory_error(
                MemoryError("needs 61 GB"), _MemoryPlan(mode="low", low_ram=True, stream_blocks=False), task="text_to_video"
            )
        self.assertEqual(
            str(higher),
            "MLX-Gen cannot fit this text_to_video run in memory with memory_mode='default' (needs 61 GB). "
            "memory_mode='low' lowers the peak, as does a shorter clip or a smaller canvas.",
        )
        self.assertEqual(
            str(lowest),
            "MLX-Gen cannot fit this text_to_video run in memory with memory_mode='low' (needs 61 GB). "
            "This is the lowest memory mode; what lowers the peak is a shorter clip or a smaller canvas.",
        )

    def test_a_boolean_or_non_integral_quantize_is_a_value_error_on_every_family_and_preload(self):
        import contextlib

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import (
            ImageEditRequest,
            ImageGenerationRequest,
            ImageUpscaleRequest,
            VideoGenerationRequest,
        )

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(
                stack, td, caps=_route_capabilities("text-to-image", "image-to-image", "text-to-video", "image-upscale")
            )
            root = Path(td)
            flux = MFluxBackendConfig(model=str(self._make_model_dir(root, "klein")), base_model="flux2-klein-4b")
            wan = MFluxBackendConfig(model=str(self._make_model_dir(root, "wan-src")), base_model="wan2.2-t2v-a14b")
            seedvr2 = MFluxBackendConfig(model=str(self._make_model_dir(root, "upscaler")), base_model="seedvr2-3b")
            runs = {
                "text_to_image": (flux, lambda b, q: b.generate_image(ImageGenerationRequest(prompt="fox", steps=2, quantize=q))),
                "image_to_image": (flux, lambda b, q: b.edit_image(ImageEditRequest(prompt="fox", image=PNG_1X1, steps=2, quantize=q))),
                "text_to_video": (
                    wan,
                    lambda b, q: b.generate_video(
                        VideoGenerationRequest(prompt="fox", num_frames=5, steps=2, quantize=q, extra={"vae_tiling": False})
                    ),
                ),
                "image_upscale": (seedvr2, lambda b, q: b.upscale_image(ImageUpscaleRequest(image=PNG_1X1, quantize=q))),
                "image_upscale (extra)": (
                    seedvr2,
                    lambda b, q: b.upscale_image(ImageUpscaleRequest(image=PNG_1X1, extra={"quantize": q})),
                ),
                "preload (image)": (flux, lambda b, q: b.preload(quantize=q)),
                "preload (upscale)": (seedvr2, lambda b, q: b.preload(quantize=q)),
            }
            for name, (config, run) in runs.items():
                # False is an unquantized spelling on the upscale route, requests and preload alike
                # (`test_upscale_quantize_values_read_as_0_3_36_read_them`).
                upscale = name.startswith("image_upscale") or name == "preload (upscale)"
                values = (True, 8.5, "8.5") if upscale else (True, False, 8.5, "8.5")
                for value in values:
                    with self.subTest(run=name, quantize=value):
                        backend = MFluxVisionBackend(config=config)
                        with self.assertRaisesRegex(ValueError, "quantize must be an integer"):
                            run(backend, value)
                        self.assertFalse(backend.has_loaded_model())
                with self.subTest(run=name, quantize=8.0):
                    backend = MFluxVisionBackend(config=config)
                    run(backend, 8.0)
                    self.assertEqual(backend._model_key.quantize, 8)

    def test_upscale_quantize_values_read_as_0_3_36_read_them(self):
        import contextlib

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageUpscaleRequest

        def published(value):
            # 0.3.36's `MFluxVisionBackend._normalize_upscale_quantize` (commit eea267d), the reference outcome.
            if value is None:
                return None
            text = str(value).strip().lower()
            if text in {"", "0", "none", "false", "off", "no"}:
                return None
            try:
                quantize = int(value)
            except Exception as exc:
                raise ValueError("image_upscale quantize must be 3, 4, 5, 6, 8, or none.") from exc
            if quantize not in {3, 4, 5, 6, 8}:
                raise ValueError("image_upscale quantize must be 3, 4, 5, 6, 8, or none.")
            return quantize

        def outcome(run, value):
            try:
                result = run(value)
            except (ValueError, CapabilityNotSupportedError) as error:
                return type(error).__name__
            return "unquantized" if result is None else result

        table = (
            "", " ", "0", 0, "none", "None", " NONE ", "false", "False", False, "off", "OFF", "no", " No ",
            3, 4, 5, 6, 8, "4", " 8 ", 8.0,
            True, 1, 2, 7, 16, -8, "7", 0.0, 8.5, 3.9, "8.0", "8.5", "true", "yes", "auto", "abc",
        )
        # The intended differences from 0.3.36, and only these:
        intended = {
            # A non-integral number is a ValueError (0.3.36 truncated it to a width).
            8.5: (8, "ValueError"),
            3.9: (3, "ValueError"),
            # An integer that is not one of the engine's widths is refused by the one check every family uses, which
            # names the field and the accepted widths (0.3.36 raised ValueError). The widths are the engine's
            # (`--quantize` choices); with mlx-gen 0.40.1 they are 0.3.36's hard-coded {3, 4, 5, 6, 8}.
            1: ("ValueError", "CapabilityNotSupportedError"),
            2: ("ValueError", "CapabilityNotSupportedError"),
            7: ("ValueError", "CapabilityNotSupportedError"),
            16: ("ValueError", "CapabilityNotSupportedError"),
            -8: ("ValueError", "CapabilityNotSupportedError"),
            "7": ("ValueError", "CapabilityNotSupportedError"),
            0.0: ("ValueError", "CapabilityNotSupportedError"),
        }
        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(stack, td, caps=_route_capabilities("image-upscale"))
            source = MFluxVisionBackend(
                config=MFluxBackendConfig(model=str(self._make_model_dir(Path(td), "upscaler")), base_model="seedvr2-3b")
            )

            def upscale(value):
                source.upscale_image(ImageUpscaleRequest(image=PNG_1X1, quantize=value))
                return _FakeSeedVR2.last_init["quantize"]

            def preload(value):
                # A preload reads `quantize` as an upscale request does, so that request reuses the instance.
                backend = MFluxVisionBackend(config=source._cfg)
                backend.preload(quantize=value)
                return _FakeSeedVR2.last_init["quantize"]

            for value in table:
                with self.subTest(quantize=value):
                    before, now = outcome(published, value), outcome(upscale, value)
                    key = next((k for k in intended if type(k) is type(value) and k == value), None)
                    self.assertEqual((before, now), intended[key] if key is not None else (before, before))
                    self.assertEqual(outcome(preload, value), now)
            # Unset is the model's default: 8 on the source.
            self.assertEqual(upscale(None), 8)
            self.assertEqual(preload(None), 8)

    def test_a_refused_preload_leaves_a_held_model_held(self):
        import contextlib

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(stack, td, caps=_route_capabilities("text-to-image"))
            source = self._make_model_dir(Path(td), "klein")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model=str(source), base_model="flux2-klein-4b"))
            backend.generate_image(ImageGenerationRequest(prompt="fox", steps=2))
            held = backend._model
            for options, error in (
                ({"memory_mode": "lowest"}, CapabilityNotSupportedError),
                ({"memory_mode": "bogus"}, ValueError),
                ({"quantize": True}, ValueError),
                ({"quantize": 7}, CapabilityNotSupportedError),
            ):
                with self.subTest(options=options):
                    with self.assertRaises(error):
                        backend.preload(**options)
                    self.assertIs(backend._model, held)
            self.assertEqual(len(_LowRamFlux2.instances), 1)

            # A preload that fails after reusing the held instance keeps it; one that fails after building an
            # instance drops what it built.
            backend._warmed_model_key = None
            with patch.object(MFluxVisionBackend, "_generate_impl", side_effect=RuntimeError("warm-up failed")):
                with self.assertRaisesRegex(RuntimeError, "warm-up failed"):
                    backend.preload()
                self.assertIs(backend._model, held)
                with self.assertRaisesRegex(RuntimeError, "warm-up failed"):
                    backend.preload(quantize=8)
            self.assertFalse(backend.has_loaded_model())
            self.assertEqual(len(_LowRamFlux2.instances), 2)

    def test_every_numeric_wan_option_is_a_finite_number_typed_or_from_extra(self):
        import contextlib

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import VideoGenerationRequest

        with tempfile.TemporaryDirectory() as td, contextlib.ExitStack() as stack:
            self._state_patches(stack, td, caps=_route_capabilities("text-to-video"))
            source = self._make_model_dir(Path(td), "wan-src")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model=str(source), base_model="wan2.2-t2v-a14b"))
            refused = {
                "guidance_2": {"guidance_2": float("nan")},
                "flow_shift": {"flow_shift": float("inf")},
            }
            refused_extra = {
                "flow_shift": {"flow_shift": "nan"},
                "max_sequence_length": {"max_sequence_length": "long"},
            }
            for field, values in refused.items():
                with self.subTest(typed=field):
                    with self.assertRaisesRegex(ValueError, f"Wan {field} must be"):
                        backend.normalize_video_generation_request(VideoGenerationRequest(prompt="fox", **values))
            for field, extra in refused_extra.items():
                with self.subTest(extra=field):
                    with self.assertRaisesRegex(ValueError, f"Wan {field} must be"):
                        backend.normalize_video_generation_request(VideoGenerationRequest(prompt="fox", extra=extra))
            normalized = backend.normalize_video_generation_request(
                VideoGenerationRequest(prompt="fox", guidance_2="3", extra={"flow_shift": "5", "max_sequence_length": 0})
            )
            self.assertEqual((normalized.guidance_2, normalized.flow_shift), (3.0, 5.0))
            self.assertEqual(normalized.extra["max_sequence_length"], 1)



class TestModelCardFrontMatter(unittest.TestCase):
    @unittest.skipUnless(_YAML_INSTALLED, "PyYAML is not installed")
    def test_a_card_that_starts_with_a_byte_order_mark_is_read(self):
        from abstractvision.backends.mflux import _read_card_frontmatter

        with tempfile.TemporaryDirectory() as td:
            card = Path(td) / "README.md"
            for prefix in (b"", b"\xef\xbb\xbf"):
                with self.subTest(byte_order_mark=bool(prefix)):
                    card.write_bytes(prefix + b"---\nbase_model: org/base\n---\n\n# adapter\n")
                    self.assertEqual(_read_card_frontmatter(card), {"base_model": "org/base"})


if __name__ == "__main__":
    unittest.main()
