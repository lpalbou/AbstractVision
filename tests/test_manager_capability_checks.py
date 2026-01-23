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
                return GeneratedAsset(media_type="image", data=b"x", mime_type="image/png", metadata={})

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
                return VisionBackendCapabilities(supported_tasks=["image_to_image"], supports_mask=False)

            def generate_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
                return GeneratedAsset(media_type="image", data=b"ok", mime_type="image/png", metadata={})

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):  # pragma: no cover
                raise NotImplementedError

            def image_to_video(self, request):  # pragma: no cover
                raise NotImplementedError

        reg = VisionModelCapabilitiesRegistry()
        vm = VisionManager(backend=NoMaskBackend(), model_id="zai-org/GLM-Image", registry=reg)

        with self.assertRaises(CapabilityNotSupportedError):
            vm.edit_image("edit", image=b"img", mask=b"mask")

        # Unmasked edit should pass the capability gate.
        out = vm.edit_image("edit", image=b"img")
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()

