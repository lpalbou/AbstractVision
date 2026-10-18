"""`endpoint:<id>` providers route to that endpoint's OpenAI-compatible API.

Regression (operator report 2026-10-18): an image route set to an
OpenAI-compatible endpoint profile (a relay serving `gpt-image-2`) fell
through to the configured default backend, because the plugin knew one
global `vision_base_url` and no endpoint profiles.
"""

import base64
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from abstractvision.backends.openai_compatible import OpenAICompatibleVisionBackend  # noqa: E402
from abstractvision.errors import AbstractVisionError  # noqa: E402
from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability  # noqa: E402

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
_RELAY = {
    "id": "relay",
    "provider": "openai-compatible",
    "provider_family": "openai-compatible",
    "base_url": "http://127.0.0.1:8317/v1",
    "api_key": "relay-key",
    "enabled": True,
}


class _Owner:
    def __init__(self, resolver=None):
        self.config = {}
        if resolver is not None:
            self.resolve_provider_endpoint_profile = resolver


def _no_core_modules():
    """Fake `abstractcore` modules: no ambient resolver, no config profiles."""
    core = types.ModuleType("abstractcore")
    config = types.ModuleType("abstractcore.config")
    providers = types.ModuleType("abstractcore.providers")
    ctx = types.ModuleType("abstractcore.providers.endpoint_context")
    ctx.current_provider_endpoint_profile_resolver = lambda: None
    config.get_config_manager = lambda: types.SimpleNamespace(resolve_provider_profile=lambda spec: None)
    return {
        "abstractcore": core,
        "abstractcore.config": config,
        "abstractcore.providers": providers,
        "abstractcore.providers.endpoint_context": ctx,
    }


class TestEndpointProfileRouting(unittest.TestCase):
    def _capture_posts(self):
        seen = []

        def fake_post_json(backend, *, path, payload):
            seen.append(("json", backend._cfg.base_url, backend._cfg.api_key, path, dict(payload)))
            return {"data": [{"b64_json": base64.b64encode(_PNG).decode("ascii")}]}

        def fake_post_multipart(backend, *, path, fields, files):
            seen.append(("multipart", backend._cfg.base_url, backend._cfg.api_key, path, dict(fields)))
            return {"data": [{"b64_json": base64.b64encode(_PNG).decode("ascii")}]}

        return seen, fake_post_json, fake_post_multipart

    def test_t2i_and_i2i_go_to_the_resolved_endpoint(self):
        calls = []

        def resolver(spec):
            calls.append(spec)
            return dict(_RELAY) if spec == "endpoint:relay" else None

        seen, post_json, post_multipart = self._capture_posts()
        with patch.dict(sys.modules, _no_core_modules()), patch.dict(
            "os.environ", {"OPENAI_API_KEY": "sk-must-not-leak", "OPENAI_BASE_URL": "https://elsewhere/v1"}, clear=True
        ), patch.object(OpenAICompatibleVisionBackend, "_post_json", post_json), patch.object(
            OpenAICompatibleVisionBackend, "_post_multipart", post_multipart
        ):
            cap = _AbstractVisionCapability(_Owner(resolver))
            out = cap.t2i("a red cube", provider="endpoint:relay", model="gpt-image-2")
            self.assertEqual(bytes(out), _PNG)
            cap.i2i("make it blue", _PNG, provider="endpoint:relay", model="gpt-image-2")

        self.assertEqual(calls[0], "endpoint:relay")
        kind, base_url, api_key, path, payload = seen[0]
        self.assertEqual((kind, base_url, api_key, path), ("json", "http://127.0.0.1:8317/v1", "relay-key", "/images/generations"))
        self.assertEqual(payload["model"], "gpt-image-2")
        self.assertNotIn("base_url", payload)
        self.assertNotIn("api_key", payload)
        kind, base_url, api_key, path, fields = seen[1]
        self.assertEqual((kind, base_url, api_key, path), ("multipart", "http://127.0.0.1:8317/v1", "relay-key", "/images/edits"))
        self.assertEqual(fields["model"], "gpt-image-2")

    def test_binding_reports_a_remote_openai_compatible_backend(self):
        with patch.dict(sys.modules, _no_core_modules()), patch.dict("os.environ", {}, clear=True):
            cap = _AbstractVisionCapability(_Owner(lambda spec: dict(_RELAY)))
            binding = cap._resolve_backend_binding(provider="endpoint:relay", model="gpt-image-2")
            again = cap._resolve_backend_binding(provider="endpoint:relay", model="gpt-image-2")
        self.assertEqual(binding["backend_kind"], "openai-compatible")
        self.assertEqual(binding["provider"], "openai-compatible")
        self.assertEqual(binding["model"], "gpt-image-2")
        self.assertFalse(binding["local_control"])
        self.assertIs(binding["backend"], again["backend"])
        # The cache key carries a digest of the key, never the key.
        self.assertNotIn("relay-key", repr(binding["backend_key"]))

    def test_core_config_profile_resolves_without_a_host_resolver(self):
        modules = _no_core_modules()
        profile = types.SimpleNamespace(private_resolution=lambda: dict(_RELAY, id="cfgrelay"))
        modules["abstractcore.config"].get_config_manager = lambda: types.SimpleNamespace(
            resolve_provider_profile=lambda spec: profile if spec == "endpoint:cfgrelay" else None
        )
        with patch.dict(sys.modules, modules), patch.dict("os.environ", {}, clear=True):
            cap = _AbstractVisionCapability(_Owner())
            binding = cap._resolve_backend_binding(provider="endpoint:cfgrelay", model="gpt-image-2")
        self.assertEqual(binding["backend"]._cfg.base_url, "http://127.0.0.1:8317/v1")

    def test_ambient_resolver_is_consulted(self):
        modules = _no_core_modules()
        modules["abstractcore.providers.endpoint_context"].current_provider_endpoint_profile_resolver = lambda: (
            lambda spec: dict(_RELAY)
        )
        with patch.dict(sys.modules, modules), patch.dict("os.environ", {}, clear=True):
            cap = _AbstractVisionCapability(_Owner())
            binding = cap._resolve_backend_binding(provider="endpoint:relay", model="gpt-image-2")
        self.assertEqual(binding["backend"]._cfg.api_key, "relay-key")

    def test_unknown_profile_fails_loudly(self):
        with patch.dict(sys.modules, _no_core_modules()), patch.dict("os.environ", {}, clear=True):
            cap = _AbstractVisionCapability(_Owner(lambda spec: None))
            with self.assertRaises(AbstractVisionError) as ctx:
                cap.t2i("x", provider="endpoint:missing", model="gpt-image-2")
        self.assertIn("endpoint:missing", str(ctx.exception))

    def test_non_openai_family_is_refused_without_leaking_the_key(self):
        anthropic = dict(_RELAY, provider="anthropic", provider_family="anthropic", api_key="secret-anthropic")
        with patch.dict(sys.modules, _no_core_modules()), patch.dict("os.environ", {}, clear=True):
            cap = _AbstractVisionCapability(_Owner(lambda spec: anthropic))
            with self.assertRaises(AbstractVisionError) as ctx:
                cap.t2i("x", provider="endpoint:claude", model="claude")
        self.assertIn("openai-compatible", str(ctx.exception))
        self.assertNotIn("secret-anthropic", str(ctx.exception))

    def test_keyless_compatible_endpoint_never_borrows_the_openai_key(self):
        keyless = dict(_RELAY, api_key="")
        with patch.dict(sys.modules, _no_core_modules()), patch.dict(
            "os.environ", {"OPENAI_API_KEY": "sk-must-not-leak"}, clear=True
        ):
            cap = _AbstractVisionCapability(_Owner(lambda spec: keyless))
            binding = cap._resolve_backend_binding(provider="endpoint:relay", model="gpt-image-2")
        self.assertIsNone(binding["backend"]._cfg.api_key)

    def test_openai_family_profile_without_base_url_uses_openai(self):
        openai_profile = dict(_RELAY, provider="openai", provider_family="openai", base_url="", api_key="sk-profile")
        with patch.dict(sys.modules, _no_core_modules()), patch.dict("os.environ", {}, clear=True):
            cap = _AbstractVisionCapability(_Owner(lambda spec: openai_profile))
            binding = cap._resolve_backend_binding(provider="endpoint:myopenai", model="gpt-image-1")
        self.assertEqual(binding["backend"]._cfg.base_url, "https://api.openai.com/v1")
        self.assertEqual(binding["backend"]._cfg.api_key, "sk-profile")
        self.assertEqual(binding["backend_kind"], "openai")


if __name__ == "__main__":
    unittest.main()
