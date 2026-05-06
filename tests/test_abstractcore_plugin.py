import importlib
import sys
import unittest
from unittest.mock import patch


class TestAbstractCorePlugin(unittest.TestCase):
    def test_import_abstractvision_is_import_light(self):
        # Importing the package should not eagerly import heavy backend modules.
        sys.modules.pop("abstractvision.backends.huggingface_diffusers", None)
        sys.modules.pop("abstractvision.backends.stable_diffusion_cpp", None)
        import abstractvision

        importlib.reload(abstractvision)

        self.assertNotIn("abstractvision.backends.huggingface_diffusers", sys.modules)
        self.assertNotIn("abstractvision.backends.stable_diffusion_cpp", sys.modules)

    def test_abstractcore_plugin_registers_backend(self):
        from abstractvision.integrations.abstractcore_plugin import register

        calls = {}

        class _Registry:
            def register_vision_backend(self, **kwargs):
                calls.update(kwargs)

        register(_Registry())
        self.assertTrue(calls.get("backend_id"))
        self.assertTrue(callable(calls.get("factory")))
        self.assertIsInstance(calls.get("config_hint"), str)

    def test_abstractcore_plugin_defaults_to_local_diffusers(self):
        import abstractvision.backends.huggingface_diffusers as hf_backend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)
        seen = {}

        class _FakeBackend:
            def __init__(self, *, config):
                seen["config"] = config

            def generate_image(self, request):
                seen["request"] = request
                return GeneratedAsset(
                    media_type="image", data=png, mime_type="image/png", metadata={}
                )

        class _DummyOwner:
            config = {}

        with patch.object(hf_backend, "HuggingFaceDiffusersVisionBackend", _FakeBackend):
            with patch.dict("os.environ", {}, clear=True):
                cap = _AbstractVisionCapability(_DummyOwner())
                out = cap.t2i("a red square", width=64, height=64, steps=2)

        self.assertTrue(out.startswith(b"\x89PNG"))
        cfg = seen["config"]
        self.assertEqual(cfg.model_id, "runwayml/stable-diffusion-v1-5")
        self.assertEqual(cfg.device, "auto")
        self.assertFalse(cfg.allow_download)
        self.assertEqual(seen["request"].width, 64)
        self.assertEqual(seen["request"].height, 64)

    def test_abstractcore_plugin_openai_backend_still_requires_base_url(self):
        from abstractvision.errors import AbstractVisionError
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {"vision_backend": "openai"}

        with patch.dict("os.environ", {}, clear=True):
            cap = _AbstractVisionCapability(_DummyOwner())
            with self.assertRaises(AbstractVisionError) as ctx:
                cap.t2i("hello")

        self.assertIn("Missing vision_base_url", str(ctx.exception))

    def test_abstractcore_plugin_capability_with_injected_backend_bytes_and_artifact(self):
        from abstractvision.backends.base_backend import VisionBackend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import (
            GeneratedAsset,
            ImageEditRequest,
            ImageGenerationRequest,
            ImageToVideoRequest,
            MultiAngleRequest,
            VideoGenerationRequest,
            VisionBackendCapabilities,
        )

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)

        class _StubBackend(VisionBackend):
            def get_capabilities(self):
                return VisionBackendCapabilities(
                    supported_tasks=[
                        "text_to_image",
                        "image_to_image",
                        "text_to_video",
                        "image_to_video",
                    ]
                )

            def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
                return GeneratedAsset(
                    media_type="image",
                    data=png,
                    mime_type="image/png",
                    metadata={"prompt": request.prompt},
                )

            def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
                return GeneratedAsset(
                    media_type="image",
                    data=png,
                    mime_type="image/png",
                    metadata={"prompt": request.prompt},
                )

            def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]:
                return [
                    GeneratedAsset(
                        media_type="image",
                        data=png,
                        mime_type="image/png",
                        metadata={"angle": "front"},
                    )
                ]

            def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
                return GeneratedAsset(
                    media_type="video",
                    data=b"ftyp" + (b"\x00" * 16),
                    mime_type="video/mp4",
                    metadata={},
                )

            def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
                return GeneratedAsset(
                    media_type="video",
                    data=b"ftyp" + (b"\x00" * 16),
                    mime_type="video/mp4",
                    metadata={},
                )

        class _DummyOwner:
            def __init__(self):
                self.config = {"vision_backend_instance": _StubBackend()}

        cap = _AbstractVisionCapability(_DummyOwner())
        out_bytes = cap.t2i("hello")
        self.assertIsInstance(out_bytes, (bytes, bytearray))
        self.assertTrue(out_bytes.startswith(b"\x89PNG"))

        # Artifact mode: use a tiny in-memory store with an AbstractRuntime-like interface.
        class _Meta:
            def __init__(self, artifact_id: str):
                self.artifact_id = artifact_id

        class _Store:
            def __init__(self):
                self._blobs = {}

            def store(
                self,
                content: bytes,
                *,
                content_type: str = "application/octet-stream",
                run_id=None,
                tags=None,
                artifact_id=None,
            ):
                aid = artifact_id or "a1"
                self._blobs[aid] = bytes(content)
                return _Meta(aid)

            def load(self, artifact_id: str):
                b = self._blobs.get(str(artifact_id))
                if b is None:
                    return None

                class _Artifact:
                    def __init__(self, content: bytes):
                        self.content = content

                return _Artifact(b)

        store = _Store()
        out_ref = cap.t2i("hello", artifact_store=store)
        self.assertIsInstance(out_ref, dict)
        self.assertEqual(out_ref.get("$artifact"), "a1")


if __name__ == "__main__":
    unittest.main()
