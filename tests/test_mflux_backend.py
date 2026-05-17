import base64
import tempfile
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


class TestMFluxVisionBackend(unittest.TestCase):
    def _make_model_dir(self, root: Path, name: str) -> Path:
        model_dir = root / name
        (model_dir / "transformer").mkdir(parents=True)
        (model_dir / "transformer" / "0.safetensors").write_bytes(b"x")
        return model_dir

    def test_generate_image_uses_local_mflux_preset(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            local_dir = self._make_model_dir(root, "flux2-klein-4b-mlx-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="flux2-klein-4b", model_dir=str(root))
            )

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
        self.assertEqual(_FakeFlux2.last_init["model_path"], str(local_dir))
        self.assertEqual(_FakeFlux2.last_generate["prompt"], "hello")
        self.assertEqual(_FakeFlux2.last_generate["width"], 64)
        self.assertEqual(_FakeFlux2.last_generate["height"], 32)
        self.assertEqual(_FakeFlux2.last_generate["num_inference_steps"], 4)
        self.assertEqual(_FakeFlux2.last_generate["guidance"], 1.0)

    def test_z_image_turbo_passes_negative_prompt(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_model_dir(root, "z-image-turbo-mlx-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="z-image-turbo", model_dir=str(root))
            )

            with patch(
                "abstractvision.backends.mflux._lazy_import_mflux",
                return_value=(_FakeModelConfig, _FakeFlux2, _FakeZImage),
            ):
                asset = backend.generate_image(
                    ImageGenerationRequest(prompt="hello", negative_prompt="blur", seed=7)
                )

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeZImage.last_init["model_config"], "z-image-turbo-config")
        self.assertEqual(_FakeZImage.last_generate["negative_prompt"], "blur")
        self.assertEqual(_FakeZImage.last_generate["num_inference_steps"], 9)

    def test_flux2_rejects_negative_prompt(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_model_dir(root, "flux2-klein-9b-mlx-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="flux2-klein-9b", model_dir=str(root))
            )
            with patch(
                "abstractvision.backends.mflux._lazy_import_mflux",
                return_value=(_FakeModelConfig, _FakeFlux2, _FakeZImage),
            ):
                with self.assertRaises(CapabilityNotSupportedError):
                    backend.generate_image(ImageGenerationRequest(prompt="hello", negative_prompt="blur"))

    def test_flux2_rejects_single_step_scheduler_edge_case(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._make_model_dir(root, "flux2-klein-4b-mlx-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="flux2-klein-4b", model_dir=str(root))
            )
            with patch(
                "abstractvision.backends.mflux._lazy_import_mflux",
                return_value=(_FakeModelConfig, _FakeFlux2, _FakeZImage),
            ):
                with self.assertRaises(ValueError) as ctx:
                    backend.generate_image(ImageGenerationRequest(prompt="hello", steps=1))

        self.assertIn("steps >= 2", str(ctx.exception))

    def test_missing_local_model_has_download_hint(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import OptionalDependencyMissingError
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as td:
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="flux2-klein-4b", model_dir=td)
            )
            with self.assertRaises(OptionalDependencyMissingError) as ctx:
                backend.generate_image(ImageGenerationRequest(prompt="hello"))

        self.assertIn("abstractvision download-model flux2-klein-4b", str(ctx.exception))

    def test_hf_repo_id_can_resolve_from_cache_without_allow_download(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.types import ImageGenerationRequest

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            local_dir = self._make_model_dir(root, "flux2-klein-4b-mlx-8bit")
            backend = MFluxVisionBackend(
                config=MFluxBackendConfig(model="org/repo", base_model="flux2-klein-4b", model_dir=str(root))
            )

            with patch(
                "abstractvision.backends.mflux.download_hf_repo_snapshot",
                return_value=local_dir,
            ):
                with patch(
                    "abstractvision.backends.mflux._lazy_import_mflux",
                    return_value=(_FakeModelConfig, _FakeFlux2, _FakeZImage),
                ):
                    asset = backend.generate_image(ImageGenerationRequest(prompt="hello", seed=2))

        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(_FakeFlux2.last_init["model_path"], str(local_dir))

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


if __name__ == "__main__":
    unittest.main()
