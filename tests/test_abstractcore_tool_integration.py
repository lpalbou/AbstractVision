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

    def test_make_vision_tools_prefers_registry_defaults_over_old_hardcoded_values(self):
        from dataclasses import replace

        from abstractvision import LocalAssetStore, VisionManager, VisionModelCapabilitiesRegistry
        from abstractvision.backends import VisionBackend
        from abstractvision.integrations.abstractcore import make_vision_tools
        from abstractvision.types import GeneratedAsset, ImageEditRequest, ImageGenerationRequest

        seen = {}

        class FakeBackend(VisionBackend):
            def normalize_image_generation_request(self, request: ImageGenerationRequest) -> ImageGenerationRequest:
                return replace(request, steps=50, guidance_scale=1.5)

            def normalize_image_edit_request(self, request: ImageEditRequest) -> ImageEditRequest:
                return replace(request, steps=15, guidance_scale=1.5)

            def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
                seen["t2i"] = request
                return GeneratedAsset(media_type="image", data=b"gen", mime_type="image/png", metadata={})

            def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
                seen["i2i"] = request
                return GeneratedAsset(media_type="image", data=b"edit", mime_type="image/png", metadata={})

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):  # pragma: no cover
                raise NotImplementedError

            def image_to_video(self, request):  # pragma: no cover
                raise NotImplementedError

        reg = VisionModelCapabilitiesRegistry()
        model_id = "zai-org/GLM-Image"

        with tempfile.TemporaryDirectory() as td:
            store = LocalAssetStore(td)
            vm = VisionManager(backend=FakeBackend(), store=store)
            with patch.dict(sys.modules, {"abstractcore": _fake_abstractcore_module()}):
                tools = make_vision_tools(vision_manager=vm, model_id=model_id, registry=reg)

            by_name = {t._tool_definition.name: t for t in tools if hasattr(t, "_tool_definition")}
            img_in = store.store_bytes(b"input", content_type="image/png")
            by_name["vision_text_to_image"](prompt="hello")
            by_name["vision_image_to_image"](prompt="edit", image_artifact=img_in)

        self.assertEqual(seen["t2i"].steps, 50)
        self.assertEqual(seen["t2i"].guidance_scale, 1.5)
        self.assertEqual(seen["i2i"].steps, 15)
        self.assertEqual(seen["i2i"].guidance_scale, 1.5)

    def test_make_vision_video_tools_prefer_registry_defaults_over_old_hardcoded_values(self):
        from dataclasses import replace

        from abstractvision import LocalAssetStore, VisionManager, VisionModelCapabilitiesRegistry
        from abstractvision.backends import VisionBackend
        from abstractvision.integrations.abstractcore import make_vision_tools
        from abstractvision.types import GeneratedAsset, ImageToVideoRequest, VideoGenerationRequest

        seen = {}

        class FakeBackend(VisionBackend):
            def normalize_video_generation_request(self, request: VideoGenerationRequest) -> VideoGenerationRequest:
                return replace(request, width=720, height=480, fps=8, num_frames=49, steps=50, guidance_scale=6.0)

            def normalize_image_to_video_request(self, request: ImageToVideoRequest) -> ImageToVideoRequest:
                return replace(request, width=544, height=960, fps=16, num_frames=81, steps=30, guidance_scale=5.0)

            def generate_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def edit_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
                seen["t2v"] = request
                return GeneratedAsset(media_type="video", data=b"video", mime_type="video/mp4", metadata={})

            def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
                seen["i2v"] = request
                return GeneratedAsset(media_type="video", data=b"video", mime_type="video/mp4", metadata={})

        with tempfile.TemporaryDirectory() as td:
            store = LocalAssetStore(td)
            vm = VisionManager(backend=FakeBackend(), store=store)
            reg = VisionModelCapabilitiesRegistry()
            with patch.dict(sys.modules, {"abstractcore": _fake_abstractcore_module()}):
                t2v_tools = make_vision_tools(vision_manager=vm, model_id="zai-org/CogVideoX-2b", registry=reg)
                i2v_tools = make_vision_tools(
                    vision_manager=vm,
                    model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                    registry=reg,
                )

            t2v = next(t for t in t2v_tools if t._tool_definition.name == "vision_text_to_video")
            i2v = next(t for t in i2v_tools if t._tool_definition.name == "vision_image_to_video")
            image_in = store.store_bytes(b"input", content_type="image/png")
            t2v(prompt="make a video")
            i2v(image_artifact=image_in)

        self.assertEqual(seen["t2v"].width, 720)
        self.assertEqual(seen["t2v"].height, 480)
        self.assertEqual(seen["t2v"].fps, 8)
        self.assertEqual(seen["t2v"].num_frames, 49)
        self.assertEqual(seen["t2v"].steps, 50)
        self.assertEqual(seen["t2v"].guidance_scale, 6.0)
        self.assertEqual(seen["i2v"].width, 544)
        self.assertEqual(seen["i2v"].height, 960)
        self.assertEqual(seen["i2v"].fps, 16)
        self.assertEqual(seen["i2v"].num_frames, 81)
        self.assertEqual(seen["i2v"].steps, 30)
        self.assertEqual(seen["i2v"].guidance_scale, 5.0)

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
