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
                t2v_tools = make_vision_tools(
                    vision_manager=vm,
                    model_id="Wan-AI/Wan2.2-T2V-A14B",
                    registry=reg,
                )
                i2v_tools = make_vision_tools(
                    vision_manager=vm,
                    model_id="Wan-AI/Wan2.2-I2V-A14B",
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
        self.assertEqual(seen["t2v"].guidance_2, 3.0)
        self.assertEqual(seen["t2v"].flow_shift, 3.0)
        self.assertEqual(seen["i2v"].width, 544)
        self.assertEqual(seen["i2v"].height, 960)
        self.assertEqual(seen["i2v"].fps, 16)
        self.assertEqual(seen["i2v"].num_frames, 81)
        self.assertEqual(seen["i2v"].steps, 30)
        self.assertEqual(seen["i2v"].guidance_scale, 5.0)
        self.assertEqual(seen["i2v"].guidance_2, 3.5)
        self.assertEqual(seen["i2v"].flow_shift, 3.0)

    def test_make_vision_tools_expose_adapter_catalog_and_batch_lora_surface(self):
        from abstractvision import LocalAssetStore, VisionManager, VisionModelCapabilitiesRegistry
        from abstractvision.backends import VisionBackend
        from abstractvision.integrations.abstractcore import make_vision_tools
        from abstractvision.types import (
            GeneratedAsset,
            ImageEditRequest,
            ImageToVideoRequest,
            ProviderAdapterInfo,
            VideoGenerationRequest,
        )

        seen = {"i2i": [], "t2v": [], "i2v": []}

        class FakeBackend(VisionBackend):
            def list_provider_adapters(self, *, model=None, task=None):
                return [
                    ProviderAdapterInfo(
                        id="AbstractFramework/pencil-style-lora",
                        repo_id="AbstractFramework/pencil-style-lora",
                        compatible_models=(str(model or ""),),
                        compatible_tasks=((str(task),) if task else ("text_to_image", "image_to_image")),
                        suggested_target_roles=("style",),
                        raw={"validated": True},
                    )
                ]

            def generate_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
                seen["i2i"].append(request)
                return GeneratedAsset(media_type="image", data=b"edit", mime_type="image/png", metadata={})

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
                seen["t2v"].append(request)
                return GeneratedAsset(media_type="video", data=b"video", mime_type="video/mp4", metadata={})

            def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
                seen["i2v"].append(request)
                return GeneratedAsset(media_type="video", data=b"video", mime_type="video/mp4", metadata={})

        with tempfile.TemporaryDirectory() as td:
            store = LocalAssetStore(td)
            vm = VisionManager(backend=FakeBackend(), store=store)
            reg = VisionModelCapabilitiesRegistry()
            with patch.dict(sys.modules, {"abstractcore": _fake_abstractcore_module()}):
                i2i_tools = make_vision_tools(
                    vision_manager=vm,
                    model_id="Qwen/Qwen-Image-Edit-2511",
                    registry=reg,
                )
                t2v_tools = make_vision_tools(
                    vision_manager=vm,
                    model_id="Wan-AI/Wan2.2-T2V-A14B",
                    registry=reg,
                )
                i2v_tools = make_vision_tools(
                    vision_manager=vm,
                    model_id="Wan-AI/Wan2.2-I2V-A14B",
                    registry=reg,
                )

            by_name_i2i = {t._tool_definition.name: t for t in i2i_tools if hasattr(t, "_tool_definition")}
            by_name_t2v = {t._tool_definition.name: t for t in t2v_tools if hasattr(t, "_tool_definition")}
            by_name_i2v = {t._tool_definition.name: t for t in i2v_tools if hasattr(t, "_tool_definition")}

            image_in = store.store_bytes(b"input", content_type="image/png")
            ref_one = store.store_bytes(b"style-ref", content_type="image/png")
            ref_two_b64 = "cmVmLXR3bw=="

            adapters = by_name_i2i["vision_list_adapters"](task="image_to_image")
            self.assertEqual(adapters[0]["id"], "AbstractFramework/pencil-style-lora")
            self.assertEqual(adapters[0]["compatible_tasks"], ["image_to_image"])

            i2i_out = by_name_i2i["vision_image_to_image_batch"](
                prompt="edit this",
                image_artifact=image_in,
                count=2,
                seeds=[101, 202],
                reference_images=[ref_one, {"b64": ref_two_b64}],
                lora_adapters=[
                    {"source": "AbstractFramework/pencil-style-lora", "scale": 0.7, "target_role": "style"},
                    {"source": "AbstractFramework/layout-helper-lora", "scale": 0.3, "target_role": "composition"},
                ],
            )
            self.assertEqual(len(i2i_out), 2)

            t2v_out = by_name_t2v["vision_text_to_video_batch"](
                prompt="move slowly",
                count=2,
                seeds=[303, 404],
                guidance_2=3.0,
                flow_shift=4.5,
                lora_adapters=[{"source": "AbstractFramework/cinematic-lora", "scale": 0.8}],
            )
            self.assertEqual(len(t2v_out), 2)

            i2v_out = by_name_i2v["vision_image_to_video_batch"](
                image_artifact=image_in,
                prompt="animate it",
                count=2,
                seeds=[505, 606],
                guidance_2=3.5,
                flow_shift=3.0,
                lora_adapters=[{"source": "AbstractFramework/cinematic-lora", "scale": 0.6}],
            )
            self.assertEqual(len(i2v_out), 2)

        self.assertEqual([request.seed for request in seen["i2i"]], [101, 202])
        self.assertEqual([request.seed for request in seen["t2v"]], [303, 404])
        self.assertEqual([request.seed for request in seen["i2v"]], [505, 606])
        self.assertEqual(len(seen["i2i"][0].lora_adapters), 2)
        self.assertEqual(seen["i2i"][0].lora_adapters[0].source, "AbstractFramework/pencil-style-lora")
        self.assertEqual(seen["i2i"][0].lora_adapters[1].target_role, "composition")
        self.assertEqual(
            seen["i2i"][0].extra["reference_images"],
            [b"style-ref", b"ref-two"],
        )
        self.assertEqual(seen["t2v"][0].flow_shift, 4.5)
        self.assertEqual(seen["t2v"][0].guidance_2, 3.0)
        self.assertEqual(seen["t2v"][0].lora_adapters[0].source, "AbstractFramework/cinematic-lora")
        self.assertEqual(seen["i2v"][0].guidance_2, 3.5)
        self.assertEqual(seen["i2v"][0].flow_shift, 3.0)

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
