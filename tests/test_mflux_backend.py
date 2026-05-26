import base64
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X9+QAAAABJRU5ErkJggg=="
)


class _FakeImage:
    def save(self, fp, format=None):  # noqa: A002
        fp.write(PNG_1X1)


class _Generated:
    image = _FakeImage()


class _FakeModelConfig:
    @staticmethod
    def from_name(model_name, base_model=None):
        configs = {
            "flux2-klein-4b": "flux2-4b-config",
            "flux2-klein-9b": "flux2-9b-config",
            "flux2-klein-base-4b": "flux2-base-4b-config",
            "flux2-klein-base-9b": "flux2-base-9b-config",
            "z-image": "z-image-config",
            "z-image-turbo": "z-image-turbo-config",
            "qwen-image": "qwen-image-config",
            "qwen-image-edit": "qwen-image-edit-config",
            "qwen-image-edit-2511": "qwen-image-edit-config",
            "qwen-image-edit-2509": "qwen-image-edit-config",
            "ernie-image-turbo": "ernie-image-turbo-config",
        }
        return configs.get(model_name, str(model_name))

    @staticmethod
    def flux2_klein_4b():
        return "flux2-4b-config"

    @staticmethod
    def flux2_klein_9b():
        return "flux2-9b-config"

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
    def ernie_image_turbo():
        return "ernie-image-turbo-config"


class _FakeFlux2:
    last_init = None
    last_generate = None

    def __init__(self, **kwargs):
        _FakeFlux2.last_init = dict(kwargs)

    def generate_image(self, **kwargs):
        _FakeFlux2.last_generate = dict(kwargs)
        return _Generated()


class _FakeZImage:
    last_init = None
    last_generate = None

    def __init__(self, **kwargs):
        _FakeZImage.last_init = dict(kwargs)

    def generate_image(self, **kwargs):
        _FakeZImage.last_generate = dict(kwargs)
        return _FakeImage()


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
    def _lazy_import_return(self, flux_cls=_FakeFlux2, z_cls=_FakeZImage):
        return (_FakeModelConfig, _FakeDownloadRequiredError, flux_cls, _FakeFlux2, z_cls, z_cls)

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

    def test_generate_image_uses_cached_mflux_preset(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            cache_root = Path(cache_td)
            snapshot = self._make_cache_snapshot(cache_root, "AITRADER/FLUX2-klein-4B-mlx-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-4b"))

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

    def test_generate_image_preserves_explicit_mlx_gen_q8_variant(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            cache_root = Path(cache_td)
            q4_snapshot = self._make_cache_snapshot(cache_root, "AbstractFramework/flux.2-klein-9b-4bit")
            q8_snapshot = self._make_cache_snapshot(cache_root, "AbstractFramework/flux.2-klein-9b-8bit")

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=self._lazy_import_return(),
                ):
                    q4_backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-9b"))
                    q4_backend.generate_image(ImageGenerationRequest(prompt="q4", steps=2, seed=1))
                    self.assertEqual(_FakeFlux2.last_init["model_path"], str(q4_snapshot))

                    q8_backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-9b-q8"))
                    q8_backend.generate_image(ImageGenerationRequest(prompt="q8", steps=2, seed=2))
                    self.assertEqual(_FakeFlux2.last_init["model_path"], str(q8_snapshot))

    def test_explicit_mlx_gen_q8_missing_does_not_fall_back_to_q4(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import OptionalDependencyMissingError
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-9b-4bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-9b-q8"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with self.assertRaisesRegex(OptionalDependencyMissingError, "flux2-klein-9b-q8"):
                    backend.generate_image(ImageGenerationRequest(prompt="q8", steps=2, seed=2))

    def test_z_image_turbo_drops_noop_negative_prompt_and_forces_guidance_off(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "carsenk/z-image-turbo-mflux-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="z-image-turbo"))

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
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="z-image"))

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
            snapshot = self._make_cache_snapshot(Path(cache_td), "AbstractFramework/ernie-image-turbo-4bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="ernie-image-turbo"))

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

    def test_z_image_turbo_uses_alternate_cached_repo_for_preset_key(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "andrevp/Z-Image-Turbo-MLX-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="z-image-turbo"))

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
                with patch.dict("os.environ", {"HF_HUB_CACHE": str(root / "empty-cache")}, clear=True):
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
        from abstractvision.backends.mflux import discover_cached_mflux_models, discover_incomplete_mflux_sources

        with tempfile.TemporaryDirectory() as td:
            cache_root = Path(td)
            repo_dir = (
                cache_root
                / "models--andrevp--Z-Image-Turbo-MLX-8bit.incompatible.20260515132039"
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
                with patch.dict("os.environ", {"HF_HUB_CACHE": str(cache_root / "empty")}, clear=True):
                    with patch(
                        "abstractvision.backends.mflux.framework_hf_cache_roots",
                        return_value=[("quarantined HF cache", cache_root)],
                    ):
                        discovered = discover_cached_mflux_models()
                        invalid = discover_incomplete_mflux_sources()

        self.assertNotIn("z-image-turbo", discovered)
        self.assertIn("z-image-turbo", invalid)
        self.assertIn(
            "incompatible HF cache: quarantined HF cache (andrevp/Z-Image-Turbo-MLX-8bit)",
            invalid["z-image-turbo"],
        )

    def test_flux2_drops_unsupported_negative_prompt(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "deepsweet/FLUX.2-klein-9B-MLX-Q8")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-9b"))
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
            self._make_cache_snapshot(Path(cache_td), "AITRADER/FLUX2-klein-4B-mlx-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-4b"))
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
                config=MFluxBackendConfig(model="flux2-klein-4b", model_dir=td)
            )
            with patch.dict("os.environ", {"HF_HUB_CACHE": str(Path(td) / "empty-cache")}, clear=True):
                with patch(
                    "abstractvision.backends.mflux.framework_hf_cache_roots",
                    return_value=[],
                ):
                    with self.assertRaises(OptionalDependencyMissingError) as ctx:
                        backend.generate_image(ImageGenerationRequest(prompt="hello"))

        self.assertIn("abstractvision download flux2-klein-4b", str(ctx.exception))

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
                config=MFluxBackendConfig(model=str(local_dir), base_model="qwen-image", model_dir=str(root))
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

        backend = MFluxVisionBackend(config=MFluxBackendConfig(model="qwen-image"))
        caps = backend.get_capabilities()

        self.assertEqual(caps.supported_tasks, ["text_to_image"])
        self.assertFalse(caps.supports_mask)

    def test_qwen_image_provider_catalog_does_not_advertise_image_to_image(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/qwen-image-2512-4bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="qwen-image"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                models = list(backend.list_provider_models())
                image_edit_models = list(backend.list_provider_models(task="image_to_image"))

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "qwen-image")
        self.assertEqual(tuple(models[0].capabilities), ("text_to_image",))
        self.assertEqual(image_edit_models, [])

    def test_provider_catalog_exposes_cached_mlx_gen_q4_and_q8_variants(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-4bit")
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/flux.2-klein-4b-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-4b"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                models = {model.id: model for model in backend.list_provider_models(task="text_to_image")}

        self.assertIn("flux2-klein-4b", models)
        self.assertIn("flux2-klein-4b-q8", models)
        self.assertEqual(models["flux2-klein-4b"].raw["quantization_bits"], 4)
        self.assertEqual(models["flux2-klein-4b"].raw["repo_id"], "AbstractFramework/flux.2-klein-4b-4bit")
        self.assertEqual(models["flux2-klein-4b-q8"].raw["quantization_bits"], 8)
        self.assertEqual(models["flux2-klein-4b-q8"].raw["repo_id"], "AbstractFramework/flux.2-klein-4b-8bit")

    def test_ernie_image_provider_catalog_advertises_text_to_image_only(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AbstractFramework/ernie-image-turbo-4bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="ernie-image-turbo"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                models = list(backend.list_provider_models())
                image_edit_models = list(backend.list_provider_models(task="image_to_image"))

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "ernie-image-turbo")
        self.assertEqual(tuple(models[0].capabilities), ("text_to_image",))
        self.assertEqual(models[0].raw["quantization_bits"], 4)
        self.assertEqual(image_edit_models, [])

    def test_qwen_edit_model_family_routes_image_to_image(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "AbstractFramework/qwen-image-edit-2511-4bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="qwen-image-edit-2511"))

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
        self.assertEqual(_FakeQwenImageEdit.last_init["model_config"], "qwen-image-edit-config")
        self.assertEqual(_FakeQwenImageEdit.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeQwenImageEdit.last_generate["prompt"], "watercolor")
        self.assertEqual(len(_FakeQwenImageEdit.last_generate["image_paths"]), 1)
        self.assertTrue(_FakeQwenImageEdit.last_generate["image_paths"][0])

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

        backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-4b"))
        caps = backend.get_capabilities()

        self.assertEqual(caps.supported_tasks, ["image_to_image", "text_to_image"])
        self.assertFalse(caps.supports_mask)

    def test_edit_image_passes_strength_and_input_dimensions(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AITRADER/FLUX2-klein-4B-mlx-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-4b"))
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
                            seed=123,
                        )
                    )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertIn("image_path", _FakeFlux2.last_generate)
        self.assertEqual(_FakeFlux2.last_generate["image_strength"], 0.75)
        self.assertEqual(_FakeFlux2.last_generate["width"], 1)
        self.assertEqual(_FakeFlux2.last_generate["height"], 1)

    def test_runtime_serializes_model_init_and_generate_on_same_thread(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AITRADER/FLUX2-klein-4B-mlx-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-4b"))
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
            self._make_cache_snapshot(Path(cache_td), "AITRADER/FLUX2-klein-4B-mlx-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="flux2-klein-4b", default_width=320, default_height=192)
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
