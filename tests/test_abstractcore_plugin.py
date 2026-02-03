import sys
import unittest


class TestAbstractCorePlugin(unittest.TestCase):
    def test_import_abstractvision_is_import_light(self):
        # Importing the package should not eagerly import heavy backend modules.
        import abstractvision  # noqa: F401

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
                    supported_tasks=["text_to_image", "image_to_image", "text_to_video", "image_to_video"]
                )

            def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={"prompt": request.prompt})

            def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={"prompt": request.prompt})

            def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]:
                return [GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={"angle": "front"})]

            def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
                return GeneratedAsset(media_type="video", data=b"ftyp" + (b"\x00" * 16), mime_type="video/mp4", metadata={})

            def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
                return GeneratedAsset(media_type="video", data=b"ftyp" + (b"\x00" * 16), mime_type="video/mp4", metadata={})

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

