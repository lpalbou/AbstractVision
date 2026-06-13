import base64
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

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


class _FakeWan:
    last_init = None
    last_generate = None
    last_image_path_existed = None
    last_image_size = None
    last_image_corner_pixel = None
    last_image_center_pixel = None

    def __init__(self, **kwargs):
        _FakeWan.last_init = dict(kwargs)

    def generate_video(self, **kwargs):
        _FakeWan.last_generate = dict(kwargs)
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
                            quantize=8,
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
        self.assertEqual(_FakeSeedVR2.last_init["quantize"], 8)
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
        self.assertEqual(_FakeWan.last_generate["width"], 1280)
        self.assertEqual(_FakeWan.last_generate["height"], 720)
        self.assertEqual(_FakeWan.last_generate["fps"], 16)
        self.assertEqual(_FakeWan.last_generate["num_frames"], 81)
        self.assertEqual(_FakeWan.last_generate["num_inference_steps"], 40)
        self.assertEqual(_FakeWan.last_generate["guidance"], 4.0)
        self.assertEqual(_FakeWan.last_generate["guidance_2"], 3.0)
        self.assertNotIn("image_path", _FakeWan.last_generate)
        self.assertEqual(asset.metadata["base_model"], "wan2.2-t2v-a14b")
        self.assertEqual(asset.metadata["guidance_2"], 3.0)

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
        self.assertEqual(_FakeWan.last_generate["width"], 1280)
        self.assertEqual(_FakeWan.last_generate["height"], 720)
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
                        r"mlx-gen>=0\.18\.18",
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
        self.assertEqual(t2v_models[0].raw["parameter_defaults"]["width"], 1280)
        self.assertEqual(t2v_models[0].raw["parameter_defaults"]["height"], 720)
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


if __name__ == "__main__":
    unittest.main()
