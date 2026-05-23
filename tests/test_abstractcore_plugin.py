import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def _make_hf_snapshot(root: Path, repo_id: str, *, snapshot_name: str = "abc123") -> Path:
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    snap = repo_dir / "snapshots" / snapshot_name
    (snap / "transformer").mkdir(parents=True, exist_ok=True)
    (snap / "transformer" / "0.safetensors").write_bytes(b"x")
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(snapshot_name, encoding="utf-8")
    return snap


class TestAbstractCorePlugin(unittest.TestCase):
    def test_import_abstractvision_is_import_light(self):
        # Importing the package should not eagerly import heavy backend modules.
        sys.modules.pop("abstractvision.backends.huggingface_diffusers", None)
        sys.modules.pop("abstractvision.backends.stable_diffusion_cpp", None)
        sys.modules.pop("abstractvision.backends.mflux", None)
        sys.modules.pop("mflux", None)
        import abstractvision

        importlib.reload(abstractvision)

        self.assertNotIn("abstractvision.backends.huggingface_diffusers", sys.modules)
        self.assertNotIn("abstractvision.backends.stable_diffusion_cpp", sys.modules)
        self.assertNotIn("abstractvision.backends.mflux", sys.modules)
        self.assertNotIn("mflux", sys.modules)

    def test_base_import_and_plugin_registration_do_not_import_heavy_modules_subprocess(self):
        script = r"""
import builtins
import os
import sys

sys.path.insert(0, os.environ["ABSTRACTVISION_SRC"])
blocked = {"torch", "diffusers", "transformers", "PIL", "stable_diffusion_cpp", "mflux", "mlx"}
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

    def test_abstractcore_plugin_uses_openai_key_for_official_openai(self):
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

        env = {"OPENAI_API_KEY": "sk-real"}
        with patch.object(openai_backend, "OpenAICompatibleVisionBackend", _FakeBackend):
            with patch.dict("os.environ", env, clear=True):
                cap = _AbstractVisionCapability(_DummyOwner())
                cap.t2i("a red square")

        self.assertEqual(seen["config"].api_key, "sk-real")

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
            "OPENAI_BASE_URL": "http://localhost:1234/v1",
            "OPENAI_API_KEY": "local-key",
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

    def test_abstractcore_plugin_uses_openai_base_url_only_config(self):
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
            "OPENAI_BASE_URL": "http://localhost:1234/v1",
            "OPENAI_API_KEY": "local-key",
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

    def test_abstractcore_plugin_uses_backend_request_normalization(self):
        import abstractvision.backends.huggingface_diffusers as hf_backend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset, ImageGenerationRequest

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)
        seen = {}

        class _FakeBackend:
            def __init__(self, *, config):
                seen["config"] = config

            def normalize_image_generation_request(self, request: ImageGenerationRequest) -> ImageGenerationRequest:
                return ImageGenerationRequest(
                    prompt=request.prompt,
                    negative_prompt=None,
                    width=request.width,
                    height=request.height,
                    seed=request.seed,
                    steps=2,
                    guidance_scale=1.0,
                    extra=dict(request.extra or {}),
                )

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
                out = cap.t2i("a red square", width=64, height=64, steps=1, guidance_scale=7.0, negative_prompt="blur")

        self.assertTrue(out.startswith(b"\x89PNG"))
        self.assertEqual(seen["request"].steps, 2)
        self.assertEqual(seen["request"].guidance_scale, 1.0)
        self.assertIsNone(seen["request"].negative_prompt)

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

    def test_abstractcore_plugin_resolves_cached_sdcpp_bundle_from_model_key(self):
        import abstractvision.backends.stable_diffusion_cpp as sdcpp_backend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.model_cache import import_directory_to_hf_cache
        from abstractvision.types import GeneratedAsset

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)
        seen = {}

        class _FakeBackend:
            def __init__(self, *, config):
                seen["config"] = config

            def generate_image(self, request):
                seen["request"] = request
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={})

        with tempfile.TemporaryDirectory() as cache_td, tempfile.TemporaryDirectory() as src_td:
            cache_root = Path(cache_td)
            src_root = Path(src_td)

            main_dir = src_root / "flux-main"
            main_dir.mkdir(parents=True, exist_ok=True)
            (main_dir / "flux-2-klein-base-4b-Q8_0.gguf").write_bytes(b"GGUF")
            import_directory_to_hf_cache(
                main_dir,
                repo_id="leejet/FLUX.2-klein-base-4B-GGUF",
                cache_dir=str(cache_root),
                cleanup_source=False,
            )

            vae_dir = src_root / "flux-vae"
            (vae_dir / "vae").mkdir(parents=True, exist_ok=True)
            (vae_dir / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"VAE")
            import_directory_to_hf_cache(
                vae_dir,
                repo_id="black-forest-labs/FLUX.2-klein-base-4B",
                cache_dir=str(cache_root),
                cleanup_source=False,
            )

            llm_dir = src_root / "qwen3"
            llm_dir.mkdir(parents=True, exist_ok=True)
            (llm_dir / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"GGUF")
            import_directory_to_hf_cache(
                llm_dir,
                repo_id="unsloth/Qwen3-4B-GGUF",
                cache_dir=str(cache_root),
                cleanup_source=False,
            )

            class _DummyOwner:
                config = {
                    "vision_backend": "sdcpp",
                    "vision_sdcpp_model": "flux2-klein-base-4b",
                    "vision_sdcpp_bin": "/opt/sd-cli",
                }

            with patch.object(sdcpp_backend, "StableDiffusionCppVisionBackend", _FakeBackend):
                with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                    cap = _AbstractVisionCapability(_DummyOwner())
                    out = cap.t2i("a red square", width=64, height=64)

        self.assertTrue(out.startswith(b"\x89PNG"))
        cfg = seen["config"]
        self.assertIsNone(cfg.model)
        self.assertTrue(str(cfg.diffusion_model or "").endswith("flux-2-klein-base-4b-Q8_0.gguf"))
        self.assertTrue(str(cfg.vae or "").endswith("vae/diffusion_pytorch_model.safetensors"))
        self.assertTrue(str(cfg.llm or "").endswith("Qwen3-4B-Q4_K_M.gguf"))
        self.assertEqual(cfg.sd_cli_path, "/opt/sd-cli")

    def test_abstractcore_plugin_can_select_local_mflux(self):
        import abstractvision.backends.mflux as mflux_backend
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
                "vision_backend": "mflux",
                "vision_mflux_model": "flux2-klein-4b",
                "vision_mflux_base_model": "flux2-klein-4b",
            }

        with patch.object(mflux_backend, "MFluxVisionBackend", _FakeBackend):
            with patch.dict("os.environ", {}, clear=True):
                cap = _AbstractVisionCapability(_DummyOwner())
                out = cap.t2i("a red square", width=64, height=64, steps=4)

        self.assertTrue(out.startswith(b"\x89PNG"))
        cfg = seen["config"]
        self.assertEqual(cfg.model, "flux2-klein-4b")
        self.assertEqual(cfg.base_model, "flux2-klein-4b")
        self.assertEqual(seen["request"].width, 64)
        self.assertEqual(seen["request"].height, 64)

    def test_abstractcore_plugin_routes_mflux_model_prefix(self):
        import abstractvision.backends.mflux as mflux_backend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)
        seen = {}

        class _FakeBackend:
            def __init__(self, *, config):
                seen["config"] = config

            def generate_image(self, request):
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={})

        class _DummyOwner:
            config = {}

        with patch.object(mflux_backend, "MFluxVisionBackend", _FakeBackend):
            with patch.dict("os.environ", {}, clear=True):
                cap = _AbstractVisionCapability(_DummyOwner())
                out = cap.t2i("a red square", model="mflux/flux2-klein-9b")

        self.assertTrue(out.startswith(b"\x89PNG"))
        self.assertEqual(seen["config"].model, "flux2-klein-9b")

    def test_abstractcore_plugin_prefers_cached_mflux_for_known_raw_model(self):
        import abstractvision.backends.mflux as mflux_backend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)
        seen = {}

        class _FakeBackend:
            def __init__(self, *, config):
                seen["config"] = config

            def generate_image(self, request):
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={})

        class _DummyOwner:
            config = {}

        with tempfile.TemporaryDirectory() as cache_td:
            _make_hf_snapshot(Path(cache_td), "deepsweet/FLUX.2-klein-9B-MLX-Q8")
            with patch.object(mflux_backend, "MFluxVisionBackend", _FakeBackend):
                with patch("abstractvision.integrations.abstractcore_plugin.sys.platform", "darwin"):
                    with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                        cap = _AbstractVisionCapability(_DummyOwner())
                        out = cap.t2i("a red square", model="black-forest-labs/FLUX.2-klein-9B")

        self.assertTrue(out.startswith(b"\x89PNG"))
        self.assertEqual(seen["config"].model, "flux2-klein-9b")

    def test_abstractcore_plugin_keeps_explicit_diffusers_provider_for_mflux_aliases(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {}

        diffusers_backend = object()
        mflux_backend = object()
        cap = _AbstractVisionCapability(_DummyOwner())

        with patch.object(cap, "_make_diffusers_backend", return_value=diffusers_backend):
            with patch.object(cap, "_make_mflux_backend", return_value=mflux_backend):
                with patch("abstractvision.integrations.abstractcore_plugin._has_local_mflux_preset", return_value=True):
                    with patch("abstractvision.integrations.abstractcore_plugin._is_known_mflux_model_alias", return_value=True):
                        explicit_hf = cap._resolve_backend_binding(
                            provider="huggingface",
                            model="black-forest-labs/FLUX.2-klein-9B",
                        )
                        explicit_diffusers = cap._resolve_backend_binding(
                            provider="diffusers",
                            model="black-forest-labs/FLUX.2-klein-9B",
                        )
                        explicit_mflux = cap._resolve_backend_binding(
                            provider="mflux",
                            model="black-forest-labs/FLUX.2-klein-9B",
                        )
                        inferred = cap._resolve_backend_binding(
                            model="black-forest-labs/FLUX.2-klein-9B",
                        )

        self.assertIs(explicit_hf["backend"], diffusers_backend)
        self.assertEqual(explicit_hf["backend_kind"], "diffusers")
        self.assertEqual(explicit_hf["provider"], "huggingface")
        self.assertEqual(explicit_hf["model"], "black-forest-labs/FLUX.2-klein-9B")
        self.assertEqual(explicit_hf["load_id"], "diffusers/black-forest-labs/FLUX.2-klein-9B")
        self.assertIs(explicit_diffusers["backend"], diffusers_backend)
        self.assertEqual(explicit_diffusers["backend_kind"], "diffusers")
        self.assertIs(explicit_mflux["backend"], mflux_backend)
        self.assertEqual(explicit_mflux["backend_kind"], "mflux")
        self.assertIs(inferred["backend"], mflux_backend)
        self.assertEqual(inferred["backend_kind"], "mflux")

    def test_abstractcore_plugin_i2i_uses_explicit_diffusers_provider_for_mflux_aliases(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset, VisionBackendCapabilities

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)
        seen = {}

        class _DummyOwner:
            config = {}

        class _FakeDiffusersBackend:
            def get_capabilities(self):
                return VisionBackendCapabilities(supported_tasks=["image_to_image"])

            def edit_image(self, request):
                seen["backend"] = "diffusers"
                seen["request"] = request
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={})

        class _FakeMfluxBackend:
            def get_capabilities(self):
                return VisionBackendCapabilities(supported_tasks=["text_to_image"])

            def edit_image(self, request):
                seen["backend"] = "mflux"
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={})

        cap = _AbstractVisionCapability(_DummyOwner())

        with patch.object(cap, "_make_diffusers_backend", return_value=_FakeDiffusersBackend()):
            with patch.object(cap, "_make_mflux_backend", return_value=_FakeMfluxBackend()):
                with patch("abstractvision.integrations.abstractcore_plugin._has_local_mflux_preset", return_value=True):
                    with patch("abstractvision.integrations.abstractcore_plugin._is_known_mflux_model_alias", return_value=True):
                        out = cap.i2i(
                            "make it watercolor",
                            image=png,
                            provider="huggingface",
                            model="black-forest-labs/FLUX.2-klein-9B",
                        )

        self.assertTrue(out.startswith(b"\x89PNG"))
        self.assertEqual(seen["backend"], "diffusers")
        self.assertEqual(seen["request"].prompt, "make it watercolor")

    def test_abstractcore_plugin_canonicalizes_mflux_aliases_for_backend_reuse(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {}

        backend = object()
        cap = _AbstractVisionCapability(_DummyOwner())

        with patch.object(cap, "_make_mflux_backend", return_value=backend) as make_backend:
            first = cap._resolve_backend_binding(
                provider="mflux",
                model="black-forest-labs/FLUX.2-klein-9B",
            )
            second = cap._resolve_backend_binding(
                provider="mflux",
                model="flux2-klein-9b",
            )

        self.assertIs(first["backend"], backend)
        self.assertIs(second["backend"], backend)
        self.assertEqual(first["backend_key"], ("mflux", "flux2-klein-9b"))
        self.assertEqual(second["backend_key"], ("mflux", "flux2-klein-9b"))
        self.assertEqual(first["model"], "flux2-klein-9b")
        self.assertEqual(second["model"], "flux2-klein-9b")
        self.assertEqual(first["load_id"], "mflux/flux2-klein-9b")
        self.assertEqual(second["load_id"], "mflux/flux2-klein-9b")
        self.assertEqual(make_backend.call_count, 1)

    def test_abstractcore_plugin_request_scoped_sdcpp_overrides_configured_backend(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {
                "vision_backend": "diffusers",
                "vision_model_id": "runwayml/stable-diffusion-v1-5",
                "vision_sdcpp_model": "/models/default.gguf",
                "vision_sdcpp_bin": "/opt/sd-cli",
            }

        requested_backend = object()
        configured_backend = object()
        cap = _AbstractVisionCapability(_DummyOwner())

        with patch.object(cap, "_make_sdcpp_backend", return_value=requested_backend) as make_sdcpp:
            with patch.object(cap, "_get_backend", return_value=configured_backend) as get_backend:
                binding = cap._resolve_backend_binding(
                    provider="sdcpp",
                    model="flux2-klein-base-4b",
                )

        self.assertIs(binding["backend"], requested_backend)
        self.assertEqual(binding["backend_kind"], "sdcpp")
        self.assertEqual(binding["provider"], "sdcpp")
        self.assertEqual(binding["model"], "flux2-klein-base-4b")
        self.assertEqual(binding["load_id"], "sdcpp/flux2-klein-base-4b")
        make_sdcpp.assert_called_once_with(model_id="flux2-klein-base-4b")
        get_backend.assert_not_called()

    def test_abstractcore_plugin_reports_mflux_available_from_cache(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {}

        with tempfile.TemporaryDirectory() as cache_td:
            _make_hf_snapshot(Path(cache_td), "deepsweet/FLUX.2-klein-9B-MLX-Q8")
            with patch("abstractvision.integrations.abstractcore_plugin.sys.platform", "darwin"):
                with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                    cap = _AbstractVisionCapability(_DummyOwner())
                    out = cap.available_providers()

        self.assertIn("mflux", out["available_providers"])
        self.assertTrue(out["details"]["mflux"]["weights_present"])

    def test_abstractcore_plugin_routes_mflux_provider_and_stores_artifact(self):
        import abstractvision.backends.mflux as mflux_backend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)
        seen = {}

        class _FakeBackend:
            def __init__(self, *, config):
                seen["config"] = config

            def generate_image(self, request):
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={})

        class _Meta:
            def __init__(self, artifact_id: str):
                self.artifact_id = artifact_id

        class _Store:
            def store(self, content, *, content_type="application/octet-stream", run_id=None, tags=None, artifact_id=None):
                seen["stored"] = {"content": bytes(content), "content_type": content_type, "run_id": run_id, "tags": tags}
                return _Meta(artifact_id or "img-1")

        class _DummyOwner:
            config = {}

        with patch.object(mflux_backend, "MFluxVisionBackend", _FakeBackend):
            with patch.dict("os.environ", {}, clear=True):
                cap = _AbstractVisionCapability(_DummyOwner())
                out = cap.t2i(
                    "a red square",
                    provider="mflux",
                    model="flux2-klein-4b",
                    artifact_store=_Store(),
                    run_id="run-1",
                    tags={"node_id": "n1"},
                )

        self.assertEqual(out.get("$artifact"), "img-1")
        self.assertEqual(seen["config"].model, "flux2-klein-4b")
        self.assertEqual(seen["stored"]["content"], png)
        self.assertEqual(seen["stored"]["run_id"], "run-1")
        self.assertEqual(seen["stored"]["tags"]["node_id"], "n1")
        self.assertEqual(seen["stored"]["tags"]["kind"], "generated_media")

    def test_abstractcore_plugin_unloads_previous_backend_when_switching_models(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)

        class _FakeBackend:
            def __init__(self, name: str):
                self.name = name
                self.unloaded = 0

            def generate_image(self, request):
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={"backend": self.name})

            def unload(self):
                self.unloaded += 1

        class _DummyOwner:
            config = {}

        mflux_backend = _FakeBackend("mflux")
        diffusers_backend = _FakeBackend("diffusers")
        cap = _AbstractVisionCapability(_DummyOwner())

        with patch.object(cap, "_make_mflux_backend", return_value=mflux_backend):
            with patch.object(cap, "_make_diffusers_backend", return_value=diffusers_backend):
                with patch.dict("os.environ", {}, clear=True):
                    first = cap.t2i("a red square", provider="mflux", model="flux2-klein-9b")
                    second = cap.t2i(
                        "a red square",
                        provider="diffusers",
                        model="runwayml/stable-diffusion-v1-5",
                    )

        self.assertTrue(first.startswith(b"\x89PNG"))
        self.assertTrue(second.startswith(b"\x89PNG"))
        self.assertEqual(mflux_backend.unloaded, 1)
        self.assertEqual(diffusers_backend.unloaded, 0)

    def test_abstractcore_plugin_can_preload_list_and_unload_local_models(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {}

        class FakeDiffusersBackend:
            def __init__(self):
                self.preloaded = 0
                self.unloaded = 0

            def preload(self):
                self.preloaded += 1

            def unload(self):
                self.unloaded += 1

        backend = FakeDiffusersBackend()
        cap = _AbstractVisionCapability(_DummyOwner())

        with patch.object(cap, "_make_diffusers_backend", return_value=backend):
            state = cap.load_resident_model(
                {
                    "task": "image_generation",
                    "provider": "diffusers",
                    "model": "runwayml/stable-diffusion-v1-5",
                }
            )
            self.assertEqual(state["task"], "text_to_image")
            self.assertEqual(state["provider"], "huggingface")
            self.assertEqual(state["backend_kind"], "diffusers")
            self.assertEqual(state["load_id"], "diffusers/runwayml/stable-diffusion-v1-5")
            self.assertTrue(state["resident"])
            self.assertEqual(state["state"], "resident")

            loaded = cap.list_loaded_models()
            resident = cap.list_resident_models()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(len(resident), 1)
            self.assertEqual(loaded[0]["source"], "explicit_preload")
            self.assertEqual(cap.list_loaded_models({"provider": "hf"})[0]["backend_kind"], "diffusers")
            self.assertEqual(cap.list_resident_models({"backend": "diffusers"})[0]["provider"], "huggingface")

            out = cap.unload_resident_model(
                {
                    "provider": "diffusers",
                    "model": "runwayml/stable-diffusion-v1-5",
                }
            )
            self.assertEqual(out["state"], "unloaded")
            self.assertFalse(out["resident"])

        self.assertEqual(backend.preloaded, 1)
        self.assertEqual(backend.unloaded, 1)
        self.assertEqual(cap.list_loaded_models(), [])

    def test_abstractcore_plugin_lists_request_warm_models_and_can_unload_them(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)

        class _DummyOwner:
            config = {}

        class FakeDiffusersBackend:
            def __init__(self):
                self.unloaded = 0

            def generate_image(self, request):
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={})

            def edit_image(self, request):
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={})

            def unload(self):
                self.unloaded += 1

        backend = FakeDiffusersBackend()
        cap = _AbstractVisionCapability(_DummyOwner())

        with patch.object(cap, "_make_diffusers_backend", return_value=backend):
            out = cap.t2i(
                "a red square",
                provider="diffusers",
                model="runwayml/stable-diffusion-v1-5",
            )

            self.assertTrue(out.startswith(b"\x89PNG"))
            loaded = cap.list_loaded_models()
            self.assertEqual(len(loaded), 1)
            self.assertFalse(loaded[0]["resident"])
            self.assertEqual(loaded[0]["state"], "active")
            self.assertEqual(loaded[0]["source"], "request")
            self.assertEqual(loaded[0]["tasks"], ["text_to_image"])

            out2 = cap.i2i(
                "edit the square",
                b"input",
                provider="diffusers",
                model="runwayml/stable-diffusion-v1-5",
            )
            self.assertTrue(out2.startswith(b"\x89PNG"))
            loaded_after_edit = cap.list_loaded_models({"task": "i2i"})
            self.assertEqual(len(loaded_after_edit), 1)
            self.assertEqual(loaded_after_edit[0]["tasks"], ["image_to_image", "text_to_image"])
            self.assertEqual(len(cap.list_loaded_models({"task": "text_to_image"})), 1)

            cap.unload_model(
                {
                    "provider": "diffusers",
                    "model": "runwayml/stable-diffusion-v1-5",
                }
            )

        self.assertEqual(backend.unloaded, 1)
        self.assertEqual(cap.list_loaded_models(), [])

    def test_abstractcore_plugin_tracks_text_to_video_request_models(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset

        mp4 = b"ftyp" + (b"\x00" * 16)

        class _DummyOwner:
            config = {}

        class FakeDiffusersBackend:
            def __init__(self):
                self.unloaded = 0

            def generate_video(self, request):
                return GeneratedAsset(media_type="video", data=mp4, mime_type="video/mp4", metadata={})

            def unload(self):
                self.unloaded += 1

        backend = FakeDiffusersBackend()
        cap = _AbstractVisionCapability(_DummyOwner())

        with patch.object(cap, "_make_diffusers_backend", return_value=backend):
            out = cap.t2v(
                "animate the square",
                provider="diffusers",
                model="zai-org/CogVideoX-2b",
            )

            self.assertTrue(out.startswith(b"ftyp"))
            loaded = cap.list_loaded_models({"task": "t2v"})
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["model"], "zai-org/CogVideoX-2b")
            self.assertEqual(loaded[0]["tasks"], ["text_to_video"])

            cap.unload_model(
                {
                    "provider": "diffusers",
                    "model": "zai-org/CogVideoX-2b",
                }
            )

        self.assertEqual(backend.unloaded, 1)
        self.assertEqual(cap.list_loaded_models(), [])

    def test_abstractcore_plugin_resident_backend_survives_model_switches(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset

        png = b"\x89PNG\r\n\x1a\n" + (b"\x00" * 16)

        class _DummyOwner:
            config = {}

        class FakeMFluxBackend:
            def __init__(self):
                self.preloaded = 0
                self.unloaded = 0

            def preload(self):
                self.preloaded += 1

            def generate_image(self, request):
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={"backend": "mflux"})

            def unload(self):
                self.unloaded += 1

        class FakeDiffusersBackend:
            def __init__(self):
                self.unloaded = 0

            def generate_image(self, request):
                return GeneratedAsset(media_type="image", data=png, mime_type="image/png", metadata={"backend": "diffusers"})

            def unload(self):
                self.unloaded += 1

        mflux_backend = FakeMFluxBackend()
        diffusers_backend = FakeDiffusersBackend()
        cap = _AbstractVisionCapability(_DummyOwner())

        with patch.object(cap, "_make_mflux_backend", return_value=mflux_backend):
            with patch.object(cap, "_make_diffusers_backend", return_value=diffusers_backend):
                cap.load_resident_model(
                    {
                        "task": "text_to_image",
                        "provider": "mflux",
                        "model": "flux2-klein-9b",
                    }
                )
                cap.t2i(
                    "a red square",
                    provider="diffusers",
                    model="runwayml/stable-diffusion-v1-5",
                )
                cap.t2i(
                    "a red square",
                    provider="mflux",
                    model="flux2-klein-9b",
                )
                cap.t2i(
                    "a red square",
                    provider="diffusers",
                    model="runwayml/stable-diffusion-v1-5",
                )

        self.assertEqual(mflux_backend.preloaded, 1)
        self.assertEqual(mflux_backend.unloaded, 0)
        self.assertEqual(diffusers_backend.unloaded, 1)

        loaded = cap.list_loaded_models()
        resident = cap.list_resident_models()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(len(resident), 1)
        self.assertEqual(resident[0]["load_id"], "mflux/flux2-klein-9b")
        self.assertTrue(resident[0]["resident"])
        self.assertIn("diffusers/runwayml/stable-diffusion-v1-5", {item["load_id"] for item in loaded})

    def test_abstractcore_plugin_rejects_ambiguous_unload_requests(self):
        from abstractvision.errors import AbstractVisionError
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {}

        class FakeBackend:
            def preload(self):
                return None

        first = FakeBackend()
        second = FakeBackend()
        cap = _AbstractVisionCapability(_DummyOwner())

        with patch.object(cap, "_make_diffusers_backend", side_effect=[first, second]):
            cap.load_resident_model(
                {
                    "task": "text_to_image",
                    "provider": "diffusers",
                    "model": "runwayml/stable-diffusion-v1-5",
                }
            )
            cap.load_resident_model(
                {
                    "task": "text_to_image",
                    "provider": "diffusers",
                    "model": "stabilityai/sdxl-turbo",
                }
            )
            with self.assertRaises(AbstractVisionError) as ctx:
                cap.unload_resident_model({"task": "text_to_image"})

        self.assertIn("Ambiguous unload request", str(ctx.exception))

    def test_abstractcore_plugin_supports_residency_for_injected_local_backend_when_kind_is_configured(self):
        from abstractvision.backends.base_backend import VisionBackend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import (
            GeneratedAsset,
            ImageEditRequest,
            ImageGenerationRequest,
            ImageToVideoRequest,
            MultiAngleRequest,
            VideoGenerationRequest,
        )

        class _InjectedBackend(VisionBackend):
            def __init__(self):
                self.preloaded = 0
                self.unloaded = 0

            def preload(self) -> None:
                self.preloaded += 1

            def unload(self) -> None:
                self.unloaded += 1

            def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
                return GeneratedAsset(media_type="image", data=b"x", mime_type="image/png", metadata={})

            def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
                return GeneratedAsset(media_type="image", data=b"x", mime_type="image/png", metadata={})

            def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]:
                raise NotImplementedError

            def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
                raise NotImplementedError

            def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
                raise NotImplementedError

        backend = _InjectedBackend()

        class _DummyOwner:
            def __init__(self):
                self.config = {
                    "vision_backend_instance": backend,
                    "vision_backend": "diffusers",
                }

        cap = _AbstractVisionCapability(_DummyOwner())
        state = cap.load_resident_model({"task": "text_to_image"})
        self.assertEqual(state["backend_kind"], "diffusers")
        self.assertTrue(state["resident"])
        self.assertEqual(backend.preloaded, 1)

    def test_abstractcore_plugin_rejects_http_backends_for_model_residency(self):
        from abstractvision.errors import AbstractVisionError
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {}

        class FakeOpenAIBackend:
            pass

        cap = _AbstractVisionCapability(_DummyOwner())

        with patch.object(cap, "_make_openai_backend", return_value=FakeOpenAIBackend()):
            with self.assertRaises(AbstractVisionError) as ctx:
                cap.load_resident_model(
                    {
                        "task": "text_to_image",
                        "provider": "openai-compatible",
                        "model": "server/default",
                    }
                )

        self.assertIn("only available for in-process local AbstractVision backends", str(ctx.exception))

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
                {
                    "OPENAI_API_KEY": "sk-test",
                    "ABSTRACTVISION_MODELS_PATH": "/catalog",
                    "ABSTRACTVISION_ASSUME_ONLINE": "1",
                },
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

    def test_abstractcore_plugin_provider_models_does_not_synthesize_openai_default(self):
        from abstractvision.errors import AbstractVisionError
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _Cfg:
            base_url = "https://api.openai.com/v1"

        class _FailingOpenAIBackend:
            _cfg = _Cfg()

            def list_provider_models(self, *, task=None):
                raise RuntimeError("catalog failed")

        class _DummyOwner:
            config = {"vision_backend_instance": _FailingOpenAIBackend()}

        cap = _AbstractVisionCapability(_DummyOwner())
        with self.assertRaises(AbstractVisionError) as ctx:
            cap.list_provider_models(task="text_to_image")

        self.assertIn("catalog failed", str(ctx.exception))
        self.assertNotIn("gpt-image-1", str(ctx.exception))

    def test_abstractcore_plugin_available_providers_excludes_openai_when_offline(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {}

        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "sk-test", "ABSTRACTVISION_ASSUME_OFFLINE": "1"},
            clear=True,
        ):
            cap = _AbstractVisionCapability(_DummyOwner())
            out = cap.available_providers()

        self.assertNotIn("openai", out["available_providers"])
        self.assertFalse(out["details"]["openai"]["reachable"])

    def test_abstractcore_plugin_available_providers_keeps_local_compatible_endpoint(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {}

        with patch.dict(
            "os.environ",
            {"OPENAI_BASE_URL": "http://127.0.0.1:1234/v1", "ABSTRACTVISION_ASSUME_OFFLINE": "1"},
            clear=True,
        ):
            cap = _AbstractVisionCapability(_DummyOwner())
            out = cap.available_providers()

        self.assertIn("openai-compatible", out["available_providers"])
        self.assertTrue(out["details"]["openai-compatible"]["reachable"])

    def test_abstractcore_plugin_provider_models_returns_empty_when_remote_catalog_disabled(self):
        import abstractvision.backends.openai_compatible as openai_backend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _NeverCalledBackend:
            def __init__(self, *, config):
                raise AssertionError("remote catalog backend should not be constructed when offline")

        class _DummyOwner:
            config = {}

        with patch.object(openai_backend, "OpenAICompatibleVisionBackend", _NeverCalledBackend):
            with patch.object(_AbstractVisionCapability, "_make_mflux_backend", side_effect=RuntimeError("disabled")):
                with patch.object(_AbstractVisionCapability, "_make_diffusers_backend", side_effect=RuntimeError("disabled")):
                    with patch.dict(
                        "os.environ",
                        {"OPENAI_API_KEY": "sk-test", "ABSTRACTVISION_ASSUME_OFFLINE": "1"},
                        clear=True,
                    ):
                        cap = _AbstractVisionCapability(_DummyOwner())
                        out = cap.list_provider_models(task="text_to_image")

        self.assertEqual(out, [])

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
