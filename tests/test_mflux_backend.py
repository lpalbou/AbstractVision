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
    def flux2_klein_4b():
        return "flux2-4b-config"

    @staticmethod
    def flux2_klein_9b():
        return "flux2-9b-config"

    @staticmethod
    def z_image_turbo():
        return "z-image-turbo-config"

    @staticmethod
    def qwen_image():
        return "qwen-image-config"


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
                    return_value=(_FakeModelConfig, _FakeFlux2, _FakeZImage),
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

    def test_z_image_turbo_passes_negative_prompt(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "carsenk/z-image-turbo-mflux-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="z-image-turbo"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=(_FakeModelConfig, _FakeFlux2, _FakeZImage),
                ):
                    asset = backend.generate_image(
                        ImageGenerationRequest(prompt="hello", negative_prompt="blur", seed=7)
                    )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeZImage.last_init["model_config"], "z-image-turbo-config")
        self.assertEqual(_FakeZImage.last_init["model_path"], str(snapshot))
        self.assertEqual(_FakeZImage.last_generate["negative_prompt"], "blur")
        self.assertEqual(_FakeZImage.last_generate["num_inference_steps"], 9)

    def test_z_image_turbo_uses_alternate_cached_repo_for_preset_key(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            snapshot = self._make_cache_snapshot(Path(cache_td), "andrevp/Z-Image-Turbo-MLX-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="z-image-turbo"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=(_FakeModelConfig, _FakeFlux2, _FakeZImage),
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
                    return_value=(_FakeModelConfig, _FakeFlux2, _FakeZImage),
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
                    return_value=(_FakeModelConfig, _FakeFlux2, _FakeZImage),
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
                    return_value=(_FakeModelConfig, _FakeFlux2, _FakeZImage),
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
                return_value=(_FakeModelConfig, _FakeFlux2, _FakeZImage),
            ):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux_qwen",
                    return_value=_FakeQwenImage,
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
            self._make_cache_snapshot(Path(cache_td), "mlx-community/Qwen-Image-2512-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="qwen-image"))

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                models = list(backend.list_provider_models())
                image_edit_models = list(backend.list_provider_models(task="image_to_image"))

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "qwen-image")
        self.assertEqual(tuple(models[0].capabilities), ("text_to_image",))
        self.assertEqual(image_edit_models, [])

    def test_flux2_klein_capabilities_are_text_to_image_only(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-4b"))
        caps = backend.get_capabilities()

        self.assertEqual(caps.supported_tasks, ["text_to_image"])
        self.assertFalse(caps.supports_mask)

    def test_edit_image_is_temporarily_disabled_for_mflux(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageEditRequest

        backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-4b"))
        with self.assertRaisesRegex(Exception, "temporarily disabled"):
            backend.edit_image(ImageEditRequest(prompt="watercolor", image=b"input"))

    def test_runtime_serializes_model_init_and_generate_on_same_thread(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td:
            self._make_cache_snapshot(Path(cache_td), "AITRADER/FLUX2-klein-4B-mlx-8bit")
            backend = MFluxVisionBackend(config=MFluxBackendConfig(model="flux2-klein-4b"))
            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=(_FakeModelConfig, _ThreadAwareFlux2, _FakeZImage),
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
                    return_value=(_FakeModelConfig, _CountingFlux2, _FakeZImage),
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
