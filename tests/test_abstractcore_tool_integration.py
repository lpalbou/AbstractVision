import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def _fake_abstractcore_module():
    module = types.ModuleType("abstractcore")

    def tool(**definition_kwargs):
        def decorate(fn):
            fn._tool_definition = types.SimpleNamespace(**definition_kwargs)
            return fn

        return decorate

    module.tool = tool
    return module


class TestAbstractCoreToolIntegration(unittest.TestCase):
    def test_make_vision_tools_and_execute_supported_calls(self):
        from abstractvision import LocalAssetStore, VisionManager, VisionModelCapabilitiesRegistry
        from abstractvision.backends import VisionBackend
        from abstractvision.integrations.abstractcore import make_vision_tools
        from abstractvision.types import GeneratedAsset, ImageEditRequest, ImageGenerationRequest

        class FakeBackend(VisionBackend):
            def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
                return GeneratedAsset(
                    media_type="image",
                    data=f"gen:{request.prompt}".encode("utf-8"),
                    mime_type="image/png",
                    metadata={"task": "text_to_image"},
                )

            def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
                return GeneratedAsset(
                    media_type="image",
                    data=b"edit:" + bytes(request.image),
                    mime_type="image/png",
                    metadata={"task": "image_to_image"},
                )

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):  # pragma: no cover
                raise NotImplementedError

            def image_to_video(self, request):  # pragma: no cover
                raise NotImplementedError

        reg = VisionModelCapabilitiesRegistry()
        model_id = (
            "zai-org/GLM-Image"  # supports text_to_image + image_to_image in the seed registry
        )

        with tempfile.TemporaryDirectory() as td:
            store = LocalAssetStore(td)
            vm = VisionManager(backend=FakeBackend(), store=store)
            with patch.dict(sys.modules, {"abstractcore": _fake_abstractcore_module()}):
                tools = make_vision_tools(vision_manager=vm, model_id=model_id, registry=reg)

            by_name = {t._tool_definition.name: t for t in tools if hasattr(t, "_tool_definition")}
            self.assertIn("vision_text_to_image", by_name)
            self.assertIn("vision_image_to_image", by_name)

            out = by_name["vision_text_to_image"](prompt="hello")
            self.assertIsInstance(out, dict)
            self.assertIn("$artifact", out)
            self.assertEqual(out.get("content_type"), "image/png")

            img_in = store.store_bytes(b"input", content_type="image/png")
            out2 = by_name["vision_image_to_image"](prompt="edit", image_artifact=img_in)
            self.assertIsInstance(out2, dict)
            self.assertIn("$artifact", out2)
            self.assertEqual(out2.get("content_type"), "image/png")

    def test_unsupported_task_raises(self):
        from abstractvision import LocalAssetStore, VisionManager, VisionModelCapabilitiesRegistry
        from abstractvision.backends import VisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.integrations.abstractcore import make_vision_tools
        from abstractvision.types import GeneratedAsset, VideoGenerationRequest

        class FakeBackend(VisionBackend):
            def generate_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def edit_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
                return GeneratedAsset(
                    media_type="video", data=b"v", mime_type="video/mp4", metadata={}
                )

            def image_to_video(self, request):  # pragma: no cover
                raise NotImplementedError

        reg = VisionModelCapabilitiesRegistry()
        model_id = "Qwen/Qwen-Image-2512"  # does NOT support text_to_video

        with tempfile.TemporaryDirectory() as td:
            vm = VisionManager(backend=FakeBackend(), store=LocalAssetStore(td))
            with patch.dict(sys.modules, {"abstractcore": _fake_abstractcore_module()}):
                tools = make_vision_tools(vision_manager=vm, model_id=model_id, registry=reg)
            t2v = next(t for t in tools if t._tool_definition.name == "vision_text_to_video")
            with self.assertRaises(CapabilityNotSupportedError):
                t2v(prompt="make a video")

    def test_tool_integration_reports_missing_abstractcore_without_package_dependency(self):
        from abstractvision.errors import OptionalDependencyMissingError
        from abstractvision.integrations.abstractcore import _require_abstractcore_tool

        with patch.dict(sys.modules, {"abstractcore": None}):
            with self.assertRaises(OptionalDependencyMissingError):
                _require_abstractcore_tool()


if __name__ == "__main__":
    unittest.main()
