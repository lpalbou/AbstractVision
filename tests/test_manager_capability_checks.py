import sys
import unittest
from pathlib import Path

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestVisionManagerCapabilityChecks(unittest.TestCase):
    def test_model_registry_blocks_unsupported_task_before_backend_called(self):
        from abstractvision import VisionManager, VisionModelCapabilitiesRegistry
        from abstractvision.backends import VisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import GeneratedAsset, ImageEditRequest

        class CountingBackend(VisionBackend):
            def __init__(self) -> None:
                self.edit_called = False

            def generate_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
                self.edit_called = True
                return GeneratedAsset(
                    media_type="image", data=b"x", mime_type="image/png", metadata={}
                )

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):  # pragma: no cover
                raise NotImplementedError

            def image_to_video(self, request):  # pragma: no cover
                raise NotImplementedError

        reg = VisionModelCapabilitiesRegistry()
        backend = CountingBackend()
        vm = VisionManager(backend=backend, model_id="Qwen/Qwen-Image-2512", registry=reg)

        with self.assertRaises(CapabilityNotSupportedError):
            vm.edit_image("edit", image=b"img")
        self.assertFalse(backend.edit_called)

    def test_backend_capabilities_block_masked_edits(self):
        from abstractvision import VisionManager, VisionModelCapabilitiesRegistry
        from abstractvision.backends import VisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import GeneratedAsset, ImageEditRequest, VisionBackendCapabilities

        class NoMaskBackend(VisionBackend):
            def get_capabilities(self) -> VisionBackendCapabilities:
                return VisionBackendCapabilities(
                    supported_tasks=["image_to_image"], supports_mask=False
                )

            def generate_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
                return GeneratedAsset(
                    media_type="image", data=b"ok", mime_type="image/png", metadata={}
                )

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):  # pragma: no cover
                raise NotImplementedError

            def image_to_video(self, request):
                seen["i2v"] = request
                return GeneratedAsset(
                    media_type="video", data=b"v", mime_type="video/mp4", metadata={}
                )

        reg = VisionModelCapabilitiesRegistry()
        vm = VisionManager(backend=NoMaskBackend(), model_id="zai-org/GLM-Image", registry=reg)

        with self.assertRaises(CapabilityNotSupportedError):
            vm.edit_image("edit", image=b"img", mask=b"mask")

        # Unmasked edit should pass the capability gate.
        out = vm.edit_image("edit", image=b"img")
        self.assertIsNotNone(out)

    def test_provider_model_listing_delegates_to_backend(self):
        from abstractvision import VisionManager
        from abstractvision.backends import VisionBackend
        from abstractvision.types import GeneratedAsset, ProviderModelInfo

        class CatalogBackend(VisionBackend):
            def list_provider_models(self, *, task=None):
                self.task = task
                return [ProviderModelInfo(id="provider/image-model")]

            def generate_image(self, request):  # pragma: no cover
                return GeneratedAsset(media_type="image", data=b"x", mime_type="image/png")

            def edit_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):  # pragma: no cover
                raise NotImplementedError

            def image_to_video(self, request):  # pragma: no cover
                raise NotImplementedError

        backend = CatalogBackend()
        vm = VisionManager(backend=backend)

        models = vm.list_provider_models(task="text_to_image")

        self.assertEqual([m.id for m in models], ["provider/image-model"])
        self.assertEqual(backend.task, "text_to_image")

    def test_provider_adapter_listing_delegates_to_backend(self):
        from abstractvision import VisionManager
        from abstractvision.backends import VisionBackend
        from abstractvision.types import GeneratedAsset, ProviderAdapterInfo

        class CatalogBackend(VisionBackend):
            def list_provider_adapters(self, *, model=None, task=None):
                self.model = model
                self.task = task
                return [
                    ProviderAdapterInfo(
                        id="owner/example:adapter.safetensors",
                        compatible_models=("AbstractFramework/qwen-image-2512-8bit",),
                        compatible_tasks=("text_to_image",),
                    )
                ]

            def generate_image(self, request):  # pragma: no cover
                return GeneratedAsset(media_type="image", data=b"x", mime_type="image/png")

            def edit_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):  # pragma: no cover
                raise NotImplementedError

            def image_to_video(self, request):  # pragma: no cover
                raise NotImplementedError

        backend = CatalogBackend()
        vm = VisionManager(backend=backend)

        adapters = vm.list_provider_adapters(
            model="AbstractFramework/qwen-image-2512-8bit",
            task="text_to_image",
        )

        self.assertEqual([item.id for item in adapters], ["owner/example:adapter.safetensors"])
        self.assertEqual(backend.model, "AbstractFramework/qwen-image-2512-8bit")
        self.assertEqual(backend.task, "text_to_image")

    def test_generate_image_batch_expands_seed_plan(self):
        from abstractvision import VisionManager
        from abstractvision.backends import VisionBackend
        from abstractvision.types import GeneratedAsset

        seen = []

        class BatchBackend(VisionBackend):
            def generate_image(self, request):
                seen.append(request.seed)
                return GeneratedAsset(media_type="image", data=b"x", mime_type="image/png")

            def edit_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):  # pragma: no cover
                raise NotImplementedError

            def image_to_video(self, request):  # pragma: no cover
                raise NotImplementedError

        vm = VisionManager(backend=BatchBackend())

        outputs = vm.generate_image_batch("hello", count=3, seed=41)

        self.assertEqual(len(outputs), 3)
        self.assertEqual(seen, [41, 42, 43])

    def test_generate_video_batch_accepts_explicit_seed_list(self):
        from abstractvision import VisionManager
        from abstractvision.backends import VisionBackend
        from abstractvision.types import GeneratedAsset

        seen = []

        class BatchBackend(VisionBackend):
            def generate_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def edit_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):
                seen.append(request.seed)
                return GeneratedAsset(media_type="video", data=b"v", mime_type="video/mp4")

            def image_to_video(self, request):  # pragma: no cover
                raise NotImplementedError

        vm = VisionManager(backend=BatchBackend())

        outputs = vm.generate_video_batch("hello", count=2, seeds=(101, 202))

        self.assertEqual(len(outputs), 2)
        self.assertEqual(seen, [101, 202])

    def test_manager_applies_backend_request_normalization(self):
        from abstractvision import VisionManager
        from abstractvision.backends import VisionBackend
        from abstractvision.types import (
            GeneratedAsset,
            ImageEditRequest,
            ImageGenerationRequest,
            ImageUpscaleRequest,
        )

        seen = {}

        class NormalizingBackend(VisionBackend):
            def normalize_image_generation_request(
                self, request: ImageGenerationRequest
            ) -> ImageGenerationRequest:
                return ImageGenerationRequest(
                    prompt=request.prompt,
                    negative_prompt=None,
                    width=request.width,
                    height=request.height,
                    seed=request.seed,
                    steps=2,
                    guidance_scale=1.0,
                    extra=dict(request.extra or {}),
                )

            def normalize_image_edit_request(self, request: ImageEditRequest) -> ImageEditRequest:
                return ImageEditRequest(
                    prompt=request.prompt,
                    image=request.image,
                    mask=request.mask,
                    negative_prompt=None,
                    seed=request.seed,
                    steps=2,
                    guidance_scale=1.0,
                    extra=dict(request.extra or {}),
                )

            def normalize_video_generation_request(self, request):
                from abstractvision.types import VideoGenerationRequest

                return VideoGenerationRequest(
                    prompt=request.prompt,
                    negative_prompt=None,
                    width=request.width,
                    height=request.height,
                    fps=8,
                    num_frames=9,
                    seed=request.seed,
                    steps=2,
                    guidance_scale=1.0,
                    extra=dict(request.extra or {}),
                )

            def normalize_image_to_video_request(self, request):
                from abstractvision.types import ImageToVideoRequest

                return ImageToVideoRequest(
                    image=request.image,
                    prompt=request.prompt,
                    negative_prompt=None,
                    width=request.width,
                    height=request.height,
                    fps=8,
                    num_frames=9,
                    seed=request.seed,
                    steps=2,
                    guidance_scale=1.0,
                    extra=dict(request.extra or {}),
                )

            def normalize_image_upscale_request(
                self, request: ImageUpscaleRequest
            ) -> ImageUpscaleRequest:
                return ImageUpscaleRequest(
                    image=request.image,
                    resolution="2x",
                    scale=2,
                    seed=request.seed,
                    softness=0.25,
                    quantize=8,
                    vae_tiling=None,
                    extra=dict(request.extra or {}),
                )

            def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
                seen["t2i"] = request
                return GeneratedAsset(
                    media_type="image", data=b"x", mime_type="image/png", metadata={}
                )

            def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
                seen["i2i"] = request
                return GeneratedAsset(
                    media_type="image", data=b"x", mime_type="image/png", metadata={}
                )

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):
                seen["t2v"] = request
                return GeneratedAsset(
                    media_type="video", data=b"v", mime_type="video/mp4", metadata={}
                )

            def image_to_video(self, request):
                seen["i2v"] = request
                return GeneratedAsset(
                    media_type="video", data=b"v", mime_type="video/mp4", metadata={}
                )

            def upscale_image(self, request: ImageUpscaleRequest) -> GeneratedAsset:
                seen["upscale"] = request
                return GeneratedAsset(
                    media_type="image", data=b"u", mime_type="image/png", metadata={}
                )

        vm = VisionManager(backend=NormalizingBackend())

        def progress_callback(event):
            return None

        vm.generate_image(
            "hello",
            steps=1,
            guidance_scale=7.0,
            negative_prompt="blur",
            on_progress=progress_callback,
        )
        vm.edit_image(
            "hello",
            image=b"img",
            steps=1,
            guidance_scale=7.0,
            negative_prompt="blur",
            on_progress=progress_callback,
        )
        vm.generate_video(
            "hello",
            steps=1,
            guidance_scale=7.0,
            negative_prompt="blur",
            on_progress=progress_callback,
        )
        vm.image_to_video(
            b"img",
            prompt="hello",
            steps=1,
            guidance_scale=7.0,
            negative_prompt="blur",
            on_progress=progress_callback,
        )
        vm.upscale_image(
            b"img",
            resolution=384,
            softness=0.9,
            quantize=4,
            on_progress=progress_callback,
        )

        self.assertEqual(seen["t2i"].steps, 2)
        self.assertEqual(seen["t2i"].guidance_scale, 1.0)
        self.assertIsNone(seen["t2i"].negative_prompt)
        self.assertIs(seen["t2i"].extra.get("on_progress"), progress_callback)
        self.assertEqual(seen["i2i"].steps, 2)
        self.assertEqual(seen["i2i"].guidance_scale, 1.0)
        self.assertIsNone(seen["i2i"].negative_prompt)
        self.assertIs(seen["i2i"].extra.get("on_progress"), progress_callback)
        self.assertEqual(seen["t2v"].steps, 2)
        self.assertEqual(seen["t2v"].guidance_scale, 1.0)
        self.assertEqual(seen["t2v"].fps, 8)
        self.assertEqual(seen["t2v"].num_frames, 9)
        self.assertIsNone(seen["t2v"].negative_prompt)
        self.assertIs(seen["t2v"].extra.get("on_progress"), progress_callback)
        self.assertEqual(seen["i2v"].steps, 2)
        self.assertEqual(seen["i2v"].guidance_scale, 1.0)
        self.assertEqual(seen["i2v"].fps, 8)
        self.assertEqual(seen["i2v"].num_frames, 9)
        self.assertIsNone(seen["i2v"].negative_prompt)
        self.assertIs(seen["i2v"].extra.get("on_progress"), progress_callback)
        self.assertEqual(seen["upscale"].resolution, "2x")
        self.assertEqual(seen["upscale"].scale, 2)
        self.assertEqual(seen["upscale"].softness, 0.25)
        self.assertEqual(seen["upscale"].quantize, 8)
        self.assertIsNone(seen["upscale"].vae_tiling)
        self.assertIs(seen["upscale"].extra.get("on_progress"), progress_callback)


if __name__ == "__main__":
    unittest.main()
