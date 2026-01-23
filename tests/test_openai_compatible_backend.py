import base64
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class _FakeHTTPResponse:
    def __init__(self, payload: bytes, *, headers=None):
        self._payload = payload
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestOpenAICompatibleVisionBackend(unittest.TestCase):
    def test_generate_image_b64_json(self):
        from abstractvision.backends.openai_compatible import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend
        from abstractvision.types import ImageGenerationRequest

        png = b"\x89PNG\r\n\x1a\n" + b"abc"
        resp = {"data": [{"b64_json": base64.b64encode(png).decode("ascii")}]}

        def fake_urlopen(req, timeout=0):
            # Basic request shaping sanity.
            self.assertIn("/images/generations", req.full_url)
            body = json.loads(req.data.decode("utf-8"))
            self.assertEqual(body.get("prompt"), "hello")
            return _FakeHTTPResponse(json.dumps(resp).encode("utf-8"))

        cfg = OpenAICompatibleBackendConfig(base_url="http://localhost:1234/v1", api_key="k", model_id="m")
        backend = OpenAICompatibleVisionBackend(config=cfg)

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            out = backend.generate_image(ImageGenerationRequest(prompt="hello"))
        self.assertEqual(out.media_type, "image")
        self.assertEqual(out.mime_type, "image/png")
        self.assertEqual(out.data, png)

    def test_edit_image_multipart_contains_prompt_and_image(self):
        from abstractvision.backends.openai_compatible import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend
        from abstractvision.types import ImageEditRequest

        png = b"\x89PNG\r\n\x1a\n" + b"out"
        resp = {"data": [{"b64_json": base64.b64encode(png).decode("ascii")}]}

        def fake_urlopen(req, timeout=0):
            self.assertIn("/images/edits", req.full_url)
            body = bytes(req.data or b"")
            self.assertIn(b'name="prompt"', body)
            self.assertIn(b"edit it", body)
            self.assertIn(b"input-bytes", body)
            return _FakeHTTPResponse(json.dumps(resp).encode("utf-8"))

        cfg = OpenAICompatibleBackendConfig(base_url="http://localhost:1234/v1", api_key=None, model_id=None)
        backend = OpenAICompatibleVisionBackend(config=cfg)

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            out = backend.edit_image(ImageEditRequest(prompt="edit it", image=b"input-bytes"))
        self.assertEqual(out.media_type, "image")
        self.assertEqual(out.mime_type, "image/png")
        self.assertEqual(out.data, png)

    def test_video_endpoints_are_opt_in(self):
        from abstractvision.backends.openai_compatible import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        cfg = OpenAICompatibleBackendConfig(base_url="http://localhost:1234/v1")
        backend = OpenAICompatibleVisionBackend(config=cfg)
        with self.assertRaises(CapabilityNotSupportedError):
            backend.generate_video(VideoGenerationRequest(prompt="x"))


if __name__ == "__main__":
    unittest.main()

