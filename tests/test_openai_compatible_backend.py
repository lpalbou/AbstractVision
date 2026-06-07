import base64
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

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
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.types import ImageGenerationRequest

        png = b"\x89PNG\r\n\x1a\n" + b"abc"
        resp = {"data": [{"b64_json": base64.b64encode(png).decode("ascii")}]}

        def fake_urlopen(req, timeout=0):
            # Basic request shaping sanity.
            self.assertIn("/images/generations", req.full_url)
            self.assertEqual(req.headers.get("Authorization"), "Bearer k")
            body = json.loads(req.data.decode("utf-8"))
            self.assertEqual(body.get("prompt"), "hello")
            return _FakeHTTPResponse(json.dumps(resp).encode("utf-8"))

        cfg = OpenAICompatibleBackendConfig(
            base_url="http://localhost:1234/v1", api_key="k", model_id="m"
        )
        backend = OpenAICompatibleVisionBackend(config=cfg)

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            out = backend.generate_image(ImageGenerationRequest(prompt="hello"))
        self.assertEqual(out.media_type, "image")
        self.assertEqual(out.mime_type, "image/png")
        self.assertEqual(out.data, png)

    def test_generate_image_uses_custom_path_and_downloads_url_response(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.types import ImageGenerationRequest

        png = b"\x89PNG\r\n\x1a\n" + b"from-url"
        resp = {"data": [{"url": "http://assets.local/out.png"}]}
        seen = {"posts": 0, "gets": 0}

        def fake_urlopen(req, timeout=0):
            if req.full_url == "http://localhost:1234/v1/custom/images":
                seen["posts"] += 1
                return _FakeHTTPResponse(json.dumps(resp).encode("utf-8"))
            if req.full_url == "http://assets.local/out.png":
                seen["gets"] += 1
                return _FakeHTTPResponse(png, headers={"Content-Type": "image/png"})
            raise AssertionError(req.full_url)

        cfg = OpenAICompatibleBackendConfig(
            base_url="http://localhost:1234/v1",
            image_generations_path="/custom/images",
        )
        backend = OpenAICompatibleVisionBackend(config=cfg)

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            out = backend.generate_image(ImageGenerationRequest(prompt="hello"))

        self.assertEqual(out.mime_type, "image/png")
        self.assertEqual(out.metadata.get("source"), "url")
        self.assertEqual(seen, {"posts": 1, "gets": 1})

    def test_generate_image_shapes_real_openai_gpt_image_payload(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.types import ImageGenerationRequest

        png = b"\x89PNG\r\n\x1a\n" + b"abc"
        resp = {"data": [{"b64_json": base64.b64encode(png).decode("ascii")}]}
        seen = {}

        def fake_urlopen(req, timeout=0):
            body = json.loads(req.data.decode("utf-8"))
            seen.update(body)
            return _FakeHTTPResponse(json.dumps(resp).encode("utf-8"))

        cfg = OpenAICompatibleBackendConfig(
            base_url="https://api.openai.com/v1",
            api_key="k",
            model_id="gpt-image-1.5",
        )
        backend = OpenAICompatibleVisionBackend(config=cfg)

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            out = backend.generate_image(
                ImageGenerationRequest(
                    prompt="hello",
                    negative_prompt="no",
                    width=1024,
                    height=1024,
                    steps=5,
                    guidance_scale=2.0,
                    seed=123,
                    extra={"quality": "low"},
                )
            )

        self.assertEqual(out.mime_type, "image/png")
        self.assertEqual(seen.get("model"), "gpt-image-1.5")
        self.assertEqual(seen.get("size"), "1024x1024")
        self.assertEqual(seen.get("quality"), "low")
        self.assertNotIn("response_format", seen)
        self.assertNotIn("negative_prompt", seen)
        self.assertNotIn("width", seen)
        self.assertNotIn("height", seen)
        self.assertNotIn("steps", seen)
        self.assertNotIn("guidance_scale", seen)
        self.assertNotIn("seed", seen)

    def test_edit_image_multipart_contains_prompt_and_image(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
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

        cfg = OpenAICompatibleBackendConfig(
            base_url="http://localhost:1234/v1", api_key=None, model_id=None
        )
        backend = OpenAICompatibleVisionBackend(config=cfg)

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            out = backend.edit_image(ImageEditRequest(prompt="edit it", image=b"input-bytes"))
        self.assertEqual(out.media_type, "image")
        self.assertEqual(out.mime_type, "image/png")
        self.assertEqual(out.data, png)

    def test_edit_image_uses_openai_gpt_image_array_field(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.types import ImageEditRequest

        png = b"\x89PNG\r\n\x1a\n" + b"out"
        resp = {"data": [{"b64_json": base64.b64encode(png).decode("ascii")}]}

        def fake_urlopen(req, timeout=0):
            body = bytes(req.data or b"")
            self.assertIn(b'name="image[]"', body)
            self.assertNotIn(b'name="image"; filename=', body)
            self.assertNotIn(b'name="negative_prompt"', body)
            return _FakeHTTPResponse(json.dumps(resp).encode("utf-8"))

        cfg = OpenAICompatibleBackendConfig(
            base_url="https://api.openai.com/v1",
            api_key="k",
            model_id="gpt-image-1.5",
        )
        backend = OpenAICompatibleVisionBackend(config=cfg)

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            out = backend.edit_image(
                ImageEditRequest(prompt="edit it", image=b"input-bytes", negative_prompt="no")
            )
        self.assertEqual(out.mime_type, "image/png")

    def test_video_endpoints_are_opt_in(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        cfg = OpenAICompatibleBackendConfig(base_url="http://localhost:1234/v1")
        backend = OpenAICompatibleVisionBackend(config=cfg)
        with self.assertRaises(CapabilityNotSupportedError):
            backend.generate_video(VideoGenerationRequest(prompt="x"))

    def test_list_provider_models_calls_models_endpoint_and_parses_entries(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )

        resp = {
            "object": "list",
            "data": [
                {
                    "id": "provider/image-model",
                    "object": "model",
                    "created": 123,
                    "owned_by": "provider",
                    "tasks": {"text_to_image": True},
                },
                "provider/string-model",
            ],
        }

        def fake_urlopen(req, timeout=0):
            self.assertEqual(req.get_method(), "GET")
            self.assertEqual(req.full_url, "http://localhost:1234/v1/models")
            self.assertEqual(req.headers.get("Authorization"), "Bearer k")
            return _FakeHTTPResponse(json.dumps(resp).encode("utf-8"))

        backend = OpenAICompatibleVisionBackend(
            config=OpenAICompatibleBackendConfig(
                base_url="http://localhost:1234/v1",
                api_key="k",
            )
        )

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            models = backend.list_provider_models()

        self.assertEqual([m.id for m in models], ["provider/image-model", "provider/string-model"])
        self.assertEqual(models[0].object, "model")
        self.assertEqual(models[0].created, 123)
        self.assertEqual(models[0].owned_by, "provider")
        self.assertIn("text_to_image", models[0].capabilities)
        self.assertEqual(
            models[1].raw,
            {
                "id": "provider/string-model",
                "provider": "openai-compatible",
                "backend": "openai-compatible",
                "routed_model": "openai-compatible/provider/string-model",
            },
        )

    def test_list_provider_models_filters_official_openai_image_models(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )

        resp = {
            "object": "list",
            "data": [
                {"id": "gpt-4.1", "object": "model"},
                {"id": "gpt-image-1", "object": "model"},
                {"id": "dall-e-3", "object": "model"},
            ],
        }

        def fake_urlopen(req, timeout=0):
            self.assertEqual(req.full_url, "https://api.openai.com/v1/models")
            return _FakeHTTPResponse(json.dumps(resp).encode("utf-8"))

        backend = OpenAICompatibleVisionBackend(
            config=OpenAICompatibleBackendConfig(base_url="https://api.openai.com/v1")
        )

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            models = backend.list_provider_models(task="text_to_image")

        self.assertEqual([m.id for m in models], ["gpt-image-1", "dall-e-3"])

    def test_list_provider_models_keeps_compatible_entries_without_capability_metadata(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )

        resp = {"data": [{"id": "local-image-model"}, {"id": "local-video-model"}]}

        def fake_urlopen(req, timeout=0):
            return _FakeHTTPResponse(json.dumps(resp).encode("utf-8"))

        backend = OpenAICompatibleVisionBackend(
            config=OpenAICompatibleBackendConfig(base_url="http://localhost:1234/v1")
        )

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            models = backend.list_provider_models(task="text_to_image")

        self.assertEqual([m.id for m in models], ["local-image-model", "local-video-model"])

    @unittest.skipUnless(
        os.environ.get("OPENAI_API_KEY")
        and os.environ.get("ABSTRACTVISION_RUN_LIVE_OPENAI_TESTS") == "1",
        "set OPENAI_API_KEY and ABSTRACTVISION_RUN_LIVE_OPENAI_TESTS=1 to run live OpenAI catalog test",
    )
    def test_list_provider_models_default_openai_catalog_live(self):
        from abstractvision import VisionManager
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )

        api_key = os.environ.get("OPENAI_API_KEY")
        backend = OpenAICompatibleVisionBackend(
            config=OpenAICompatibleBackendConfig(
                base_url="https://api.openai.com/v1",
                api_key=api_key,
                timeout_s=30.0,
            )
        )
        manager = VisionManager(backend=backend)

        models = list(manager.list_provider_models(task="text_to_image"))
        model_ids = {m.id for m in models}

        self.assertTrue(model_ids, "OpenAI provider catalog returned no image-capable models")
        self.assertTrue(
            any(mid.startswith(("gpt-image-", "dall-e-")) for mid in model_ids),
            f"Expected at least one OpenAI image model, got: {sorted(model_ids)[:10]}",
        )

    def test_video_capabilities_and_generation_payload(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.types import VideoGenerationRequest

        mp4 = b"\x00\x00\x00\x18ftypmp42" + b"x"
        resp = {"data": [{"b64_json": base64.b64encode(mp4).decode("ascii")}]}
        seen = {}

        def fake_urlopen(req, timeout=0):
            seen["url"] = req.full_url
            seen["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse(json.dumps(resp).encode("utf-8"))

        cfg = OpenAICompatibleBackendConfig(
            base_url="http://localhost:1234/v1",
            model_id="video-model",
            text_to_video_path="/videos/generations",
        )
        backend = OpenAICompatibleVisionBackend(config=cfg)
        self.assertIn("text_to_video", backend.get_capabilities().supported_tasks)

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            out = backend.generate_video(
                VideoGenerationRequest(
                    prompt="move",
                    width=320,
                    height=240,
                    fps=12,
                    num_frames=8,
                    guidance_scale=4.0,
                    guidance_2=2.5,
                )
            )

        self.assertEqual(out.media_type, "video")
        self.assertEqual(out.mime_type, "video/mp4")
        self.assertIn("/videos/generations", seen["url"])
        self.assertEqual(seen["body"].get("model"), "video-model")
        self.assertEqual(seen["body"].get("fps"), 12)
        self.assertEqual(seen["body"].get("num_frames"), 8)
        self.assertEqual(seen["body"].get("guidance_scale"), 4.0)
        self.assertEqual(seen["body"].get("guidance_2"), 2.5)

    def test_image_to_video_json_b64_payload(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.types import ImageToVideoRequest

        mp4 = b"\x00\x00\x00\x18ftypmp42" + b"x"
        resp = {"data": [{"b64_json": base64.b64encode(mp4).decode("ascii")}]}
        seen = {}

        def fake_urlopen(req, timeout=0):
            seen["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse(json.dumps(resp).encode("utf-8"))

        cfg = OpenAICompatibleBackendConfig(
            base_url="http://localhost:1234/v1",
            image_to_video_path="/videos/edits",
            image_to_video_mode="json_b64",
        )
        backend = OpenAICompatibleVisionBackend(config=cfg)

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            out = backend.image_to_video(
                ImageToVideoRequest(image=b"image-bytes", prompt="move", guidance_2=3.5)
            )

        self.assertEqual(out.mime_type, "video/mp4")
        self.assertEqual(base64.b64decode(seen["body"]["image_b64"]), b"image-bytes")
        self.assertEqual(seen["body"].get("prompt"), "move")
        self.assertEqual(seen["body"].get("guidance_2"), 3.5)

    def test_image_to_video_multipart_payload(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.types import ImageToVideoRequest

        mp4 = b"\x00\x00\x00\x18ftypmp42" + b"x"
        resp = {"data": [{"b64_json": base64.b64encode(mp4).decode("ascii")}]}

        def fake_urlopen(req, timeout=0):
            body = bytes(req.data or b"")
            self.assertIn(b'name="image"; filename="image.png"', body)
            self.assertIn(b"image-bytes", body)
            self.assertIn(b'name="prompt"', body)
            self.assertIn(b'name="guidance_2"', body)
            self.assertIn(b"3.5", body)
            return _FakeHTTPResponse(json.dumps(resp).encode("utf-8"))

        cfg = OpenAICompatibleBackendConfig(
            base_url="http://localhost:1234/v1",
            image_to_video_path="/videos/edits",
        )
        backend = OpenAICompatibleVisionBackend(config=cfg)

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            out = backend.image_to_video(
                ImageToVideoRequest(image=b"image-bytes", prompt="move", guidance_2=3.5)
            )

        self.assertEqual(out.mime_type, "video/mp4")

    def test_invalid_response_shape_raises(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.types import ImageGenerationRequest

        def fake_urlopen(req, timeout=0):
            return _FakeHTTPResponse(json.dumps({"data": [{}]}).encode("utf-8"))

        backend = OpenAICompatibleVisionBackend(
            config=OpenAICompatibleBackendConfig(base_url="http://localhost:1234/v1")
        )

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            with self.assertRaises(ValueError) as ctx:
                backend.generate_image(ImageGenerationRequest(prompt="hello"))

        self.assertIn("missing data", str(ctx.exception))

    def test_provider_http_error_has_context(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.types import ImageGenerationRequest

        class _Body:
            def read(self):
                return b'{"error":"bad"}'

            def close(self):
                return None

        def fake_urlopen(req, timeout=0):
            raise HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=_Body(),
            )

        backend = OpenAICompatibleVisionBackend(
            config=OpenAICompatibleBackendConfig(base_url="http://localhost:1234/v1")
        )

        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            with self.assertRaises(RuntimeError) as ctx:
                backend.generate_image(ImageGenerationRequest(prompt="hello"))

        self.assertIn("status=400", str(ctx.exception))
        self.assertIn("bad", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
