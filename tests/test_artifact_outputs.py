import sys
import tempfile
import unittest
from pathlib import Path

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestArtifactOutputs(unittest.TestCase):
    def test_local_asset_store_roundtrip_and_deterministic_id(self):
        from abstractvision.artifacts import LocalAssetStore, compute_artifact_id, is_artifact_ref

        content = b"fake-png-bytes"
        expected_id = compute_artifact_id(content)

        with tempfile.TemporaryDirectory() as td:
            store = LocalAssetStore(td)
            ref = store.store_bytes(content, content_type="image/png", filename="out.png")

            self.assertTrue(is_artifact_ref(ref))
            self.assertEqual(ref["$artifact"], expected_id)
            self.assertEqual(ref.get("content_type"), "image/png")
            self.assertEqual(ref.get("filename"), "out.png")
            self.assertEqual(ref.get("size_bytes"), len(content))
            self.assertIsInstance(ref.get("sha256"), str)

            # Roundtrip.
            loaded = store.load_bytes(expected_id)
            self.assertEqual(loaded, content)

            # Idempotent store.
            ref2 = store.store_bytes(content, content_type="image/png")
            self.assertEqual(ref2["$artifact"], expected_id)

            # Metadata file exists.
            meta = store.get_metadata(expected_id)
            self.assertIsInstance(meta, dict)
            self.assertEqual(meta.get("artifact_id"), expected_id)

    def test_vision_manager_returns_artifact_ref_when_store_configured(self):
        from abstractvision import LocalAssetStore, VisionManager
        from abstractvision.backends import VisionBackend
        from abstractvision.types import GeneratedAsset, ImageGenerationRequest

        class FakeBackend(VisionBackend):
            def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
                return GeneratedAsset(media_type="image", data=b"img", mime_type="image/png", metadata={"prompt": request.prompt})

            def edit_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):  # pragma: no cover
                raise NotImplementedError

            def image_to_video(self, request):  # pragma: no cover
                raise NotImplementedError

        with tempfile.TemporaryDirectory() as td:
            vm = VisionManager(backend=FakeBackend(), store=LocalAssetStore(td))
            out = vm.generate_image("hello")
            self.assertIsInstance(out, dict)
            self.assertIn("$artifact", out)
            self.assertEqual(out.get("content_type"), "image/png")


if __name__ == "__main__":
    unittest.main()

