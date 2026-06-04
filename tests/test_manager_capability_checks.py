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

    def test_manager_applies_backend_request_normalization(self):
        from abstractvision import VisionManager
        from abstractvision.backends import VisionBackend
        from abstractvision.types import GeneratedAsset, ImageEditRequest, ImageGenerationRequest

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


if __name__ == "__main__":
    unittest.main()
