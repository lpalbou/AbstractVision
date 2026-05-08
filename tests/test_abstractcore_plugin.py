import importlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestAbstractCorePlugin(unittest.TestCase):
    def test_import_abstractvision_is_import_light(self):
        # Importing the package should not eagerly import heavy backend modules.
        sys.modules.pop("abstractvision.backends.huggingface_diffusers", None)
        sys.modules.pop("abstractvision.backends.stable_diffusion_cpp", None)
        import abstractvision

        importlib.reload(abstractvision)

        self.assertNotIn("abstractvision.backends.huggingface_diffusers", sys.modules)
        self.assertNotIn("abstractvision.backends.stable_diffusion_cpp", sys.modules)

    def test_base_import_and_plugin_registration_do_not_import_heavy_modules_subprocess(self):
        script = r"""
import builtins
import os
import sys

sys.path.insert(0, os.environ["ABSTRACTVISION_SRC"])
blocked = {"torch", "diffusers", "transformers", "PIL", "stable_diffusion_cpp"}
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".", 1)[0]
    if root in blocked:
        raise AssertionError(f"unexpected heavy import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import

import abstractvision
from abstractvision.integrations.abstractcore_plugin import register

calls = []

class Registry:
    def register_vision_backend(self, **kwargs):
        calls.append(dict(kwargs))

register(Registry())
backend_ids = {call["backend_id"] for call in calls}
assert "abstractvision:openai" in backend_ids
assert "abstractvision:openai-compatible" in backend_ids
assert all(callable(call["factory"]) for call in calls)
for name in blocked:
    assert name not in sys.modules, name
print("ok")
"""
        env = dict(os.environ)
        env["ABSTRACTVISION_SRC"] = str(SRC_DIR)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_abstractcore_plugin_registers_backend(self):
        from abstractvision.integrations.abstractcore_plugin import register

        calls = []

        class _Registry:
            def register_vision_backend(self, **kwargs):
                calls.append(dict(kwargs))

        register(_Registry())
        calls_by_id = {c.get("backend_id"): c for c in calls}
        backend_ids = set(calls_by_id)
        self.assertIn("abstractvision:openai", backend_ids)
        self.assertIn("abstractvision:openai-compatible", backend_ids)
        self.assertTrue(all(callable(c.get("factory")) for c in calls))
        self.assertTrue(all(isinstance(c.get("config_hint"), str) for c in calls))
        self.assertEqual(calls_by_id["abstractvision:openai"].get("priority"), 0)
        self.assertEqual(calls_by_id["abstractvision:openai-compatible"].get("priority"), -1)
        self.assertIn("OpenAI HTTP", calls_by_id["abstractvision:openai"]["config_hint"])
        self.assertIn(
            "Compatibility backend id",
            calls_by_id["abstractvision:openai-compatible"]["config_hint"],
        )

        class _DummyOwner:
            config = {}

        openai_cap = calls_by_id["abstractvision:openai"]["factory"](_DummyOwner())
        compatible_cap = calls_by_id["abstractvision:openai-compatible"]["factory"](_DummyOwner())
        self.assertEqual(openai_cap.backend_id, "abstractvision:openai")
        self.assertEqual(compatible_cap.backend_id, "abstractvision:openai-compatible")

    def test_abstractcore_plugin_defaults_to_openai(self):
        import abstractvision.backends.openai_compatible as openai_backend
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

        with patch.object(openai_backend, "OpenAICompatibleVisionBackend", _FakeBackend):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
                cap = _AbstractVisionCapability(_DummyOwner())
                out = cap.t2i("a red square", width=1024, height=1024)

        self.assertTrue(out.startswith(b"\x89PNG"))
        cfg = seen["config"]
        self.assertEqual(cfg.base_url, "https://api.openai.com/v1")
        self.assertEqual(cfg.api_key, "sk-test")
        self.assertEqual(cfg.model_id, "gpt-image-1")
        self.assertEqual(seen["request"].width, 1024)
        self.assertEqual(seen["request"].height, 1024)

    def test_abstractcore_plugin_openai_env_aliases_override_defaults(self):
        import abstractvision.backends.openai_compatible as openai_backend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)
        seen = {}

        class _FakeBackend:
            def __init__(self, *, config):
                seen["config"] = config

            def generate_image(self, request):
                return GeneratedAsset(
                    media_type="image", data=png, mime_type="image/png", metadata={}
                )

        class _DummyOwner:
            config = {}

        env = {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_BASE_URL": "https://proxy.example/v1",
            "OPENAI_IMAGE_MODEL": "gpt-image-custom",
        }
        with patch.object(openai_backend, "OpenAICompatibleVisionBackend", _FakeBackend):
            with patch.dict("os.environ", env, clear=True):
                cap = _AbstractVisionCapability(_DummyOwner())
                out = cap.t2i("a red square", width=1024, height=1024)

        self.assertTrue(out.startswith(b"\x89PNG"))
        cfg = seen["config"]
        self.assertEqual(cfg.base_url, "https://proxy.example/v1")
        self.assertEqual(cfg.api_key, "sk-test")
        self.assertEqual(cfg.model_id, "gpt-image-custom")

    def test_abstractcore_plugin_preserves_explicit_openai_compatible_config(self):
        import abstractvision.backends.openai_compatible as openai_backend
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

        env = {
            "ABSTRACTVISION_BACKEND": "openai-compatible",
            "ABSTRACTVISION_BASE_URL": "http://localhost:1234/v1",
            "ABSTRACTVISION_API_KEY": "local-key",
            "ABSTRACTVISION_MODEL_ID": "local-image",
        }
        with patch.object(openai_backend, "OpenAICompatibleVisionBackend", _FakeBackend):
            with patch.dict("os.environ", env, clear=True):
                cap = _AbstractVisionCapability(_DummyOwner())
                out = cap.t2i("a red square", width=1024, height=1024)

        self.assertTrue(out.startswith(b"\x89PNG"))
        cfg = seen["config"]
        self.assertEqual(cfg.base_url, "http://localhost:1234/v1")
        self.assertEqual(cfg.api_key, "local-key")
        self.assertEqual(cfg.model_id, "local-image")

    def test_abstractcore_plugin_preserves_legacy_base_url_only_config(self):
        import abstractvision.backends.openai_compatible as openai_backend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)
        seen = {}

        class _FakeBackend:
            def __init__(self, *, config):
                seen["config"] = config

            def generate_image(self, request):
                return GeneratedAsset(
                    media_type="image", data=png, mime_type="image/png", metadata={}
                )

        class _DummyOwner:
            config = {}

        env = {
            "ABSTRACTVISION_BASE_URL": "http://localhost:1234/v1",
            "ABSTRACTVISION_API_KEY": "local-key",
        }
        with patch.object(openai_backend, "OpenAICompatibleVisionBackend", _FakeBackend):
            with patch.dict("os.environ", env, clear=True):
                cap = _AbstractVisionCapability(_DummyOwner())
                out = cap.t2i("a red square", width=1024, height=1024)

        self.assertTrue(out.startswith(b"\x89PNG"))
        cfg = seen["config"]
        self.assertEqual(cfg.base_url, "http://localhost:1234/v1")
        self.assertEqual(cfg.api_key, "local-key")
        self.assertIsNone(cfg.model_id)

    def test_abstractcore_plugin_legacy_factory_preserves_compatible_default(self):
        from abstractvision.errors import AbstractVisionError
        from abstractvision.integrations.abstractcore_plugin import register

        calls = {}

        class _Registry:
            def register_vision_backend(self, **kwargs):
                calls[kwargs["backend_id"]] = dict(kwargs)

        class _DummyOwner:
            config = {}

        register(_Registry())
        factory = calls["abstractvision:openai-compatible"]["factory"]
        with patch.dict("os.environ", {}, clear=True):
            cap = factory(_DummyOwner())
            self.assertEqual(cap.backend_id, "abstractvision:openai-compatible")
            with self.assertRaises(AbstractVisionError) as ctx:
                cap.t2i("hello")

        self.assertIn("Missing vision_base_url", str(ctx.exception))

    def test_abstractcore_plugin_can_select_local_diffusers(self):
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
            config = {"vision_backend": "diffusers"}

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

    def test_abstractcore_plugin_can_select_local_sdcpp(self):
        import abstractvision.backends.stable_diffusion_cpp as sdcpp_backend
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
            config = {
                "vision_backend": "sdcpp",
                "vision_sdcpp_model": "/models/sd-v1-5.gguf",
                "vision_sdcpp_bin": "/opt/sd-cli",
            }

        with patch.object(sdcpp_backend, "StableDiffusionCppVisionBackend", _FakeBackend):
            with patch.dict("os.environ", {}, clear=True):
                cap = _AbstractVisionCapability(_DummyOwner())
                out = cap.t2i("a red square", width=64, height=64)

        self.assertTrue(out.startswith(b"\x89PNG"))
        cfg = seen["config"]
        self.assertEqual(cfg.model, "/models/sd-v1-5.gguf")
        self.assertEqual(cfg.sd_cli_path, "/opt/sd-cli")
        self.assertEqual(seen["request"].width, 64)
        self.assertEqual(seen["request"].height, 64)

    def test_abstractcore_plugin_default_openai_backend_requires_api_key(self):
        from abstractvision.errors import AbstractVisionError
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {}

        with patch.dict("os.environ", {}, clear=True):
            cap = _AbstractVisionCapability(_DummyOwner())
            with self.assertRaises(AbstractVisionError) as ctx:
                cap.t2i("hello")

        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_abstractcore_plugin_explicit_compatible_backend_requires_base_url(self):
        from abstractvision.errors import AbstractVisionError
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {}

        with patch.dict("os.environ", {"ABSTRACTVISION_BACKEND": "openai-compatible"}, clear=True):
            cap = _AbstractVisionCapability(_DummyOwner())
            with self.assertRaises(AbstractVisionError) as ctx:
                cap.t2i("hello")

        self.assertIn("Missing vision_base_url", str(ctx.exception))

    def test_abstractcore_plugin_lists_provider_models_from_configured_backend(self):
        import abstractvision.backends.openai_compatible as openai_backend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import ProviderModelInfo

        seen = {}

        class _FakeBackend:
            def __init__(self, *, config):
                seen["config"] = config

            def list_provider_models(self, *, task=None):
                seen["task"] = task
                return [
                    ProviderModelInfo(
                        id="gpt-image-1",
                        object="model",
                        created=123,
                        owned_by="openai",
                        capabilities=("text_to_image",),
                        raw={"id": "gpt-image-1", "provider_note": "x" * 5000},
                    )
                ]

        class _DummyOwner:
            config = {}

        with patch.object(openai_backend, "OpenAICompatibleVisionBackend", _FakeBackend):
            with patch.dict(
                "os.environ",
                {"OPENAI_API_KEY": "sk-test", "ABSTRACTVISION_MODELS_PATH": "/catalog"},
                clear=True,
            ):
                cap = _AbstractVisionCapability(_DummyOwner())
                out = cap.list_provider_models(task="text_to_image")

        json.dumps(out)
        self.assertEqual(seen["task"], "text_to_image")
        self.assertEqual(seen["config"].base_url, "https://api.openai.com/v1")
        self.assertEqual(seen["config"].models_path, "/catalog")
        self.assertEqual(out[0]["id"], "gpt-image-1")
        self.assertEqual(out[0]["capabilities"], ["text_to_image"])
        self.assertTrue(out[0]["raw"]["provider_note"].endswith("...<truncated>"))

    def test_abstractcore_plugin_provider_models_requires_catalog_backend(self):
        from abstractvision.backends.base_backend import VisionBackend
        from abstractvision.errors import AbstractVisionError
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import (
            GeneratedAsset,
            ImageEditRequest,
            ImageGenerationRequest,
            ImageToVideoRequest,
            MultiAngleRequest,
            VideoGenerationRequest,
        )

        class _NoCatalogBackend(VisionBackend):
            def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
                raise NotImplementedError

            def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
                raise NotImplementedError

            def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]:
                raise NotImplementedError

            def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
                raise NotImplementedError

            def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
                raise NotImplementedError

        class _DummyOwner:
            def __init__(self):
                self.config = {"vision_backend_instance": _NoCatalogBackend()}

        cap = _AbstractVisionCapability(_DummyOwner())
        with self.assertRaises(AbstractVisionError) as ctx:
            cap.list_provider_models(task="text_to_image")

        self.assertIn("does not support provider model catalogs", str(ctx.exception))

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
                        "multi_view_image",
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

        out_angles = cap.multi_view_image("hello")
        self.assertEqual(len(out_angles), 1)
        self.assertTrue(out_angles[0].startswith(b"\x89PNG"))

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
