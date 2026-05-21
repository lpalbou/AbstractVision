import contextlib
import io
import json
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


class TestCliSmoke(unittest.TestCase):
    def _make_hf_snapshot(self, root: Path, repo_id: str, *, snapshot_name: str = "abc123") -> Path:
        repo_dir = root / f"models--{repo_id.replace('/', '--')}"
        snap = repo_dir / "snapshots" / snapshot_name
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "model_index.json").write_text("{}", encoding="utf-8")
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (repo_dir / "refs" / "main").write_text(snapshot_name, encoding="utf-8")
        return snap

    def test_models_lists_known_id(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["models"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("Qwen/Qwen-Image-2512", out)
        self.assertIn("black-forest-labs/FLUX.2-klein-4B", out)

    def test_tasks_lists_text_to_image(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["tasks"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("text_to_image", out)

    def test_show_model_prints_tasks_section(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["show-model", "zai-org/GLM-Image"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("zai-org/GLM-Image", out)
        self.assertIn("tasks:", out)

    def test_provider_models_lists_openai_compatible_catalog(self):
        from abstractvision.cli import main

        class _Resp:
            headers = {}

            def read(self):
                return json.dumps({"data": [{"id": "provider/image-model"}]}).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        seen = {}

        def fake_urlopen(req, timeout=0):
            seen["url"] = req.full_url
            seen["method"] = req.get_method()
            return _Resp()

        buf = io.StringIO()
        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            with contextlib.redirect_stdout(buf):
                rc = main(["provider-models", "--base-url", "http://localhost:1234/v1"])

        self.assertEqual(rc, 0)
        self.assertEqual(seen, {"url": "http://localhost:1234/v1/models", "method": "GET"})
        self.assertIn("provider/image-model", buf.getvalue())

    def test_provider_models_openai_uses_default_catalog(self):
        from abstractvision.cli import main

        class _Resp:
            headers = {}

            def read(self):
                return json.dumps(
                    {
                        "data": [
                            {"id": "gpt-4.1"},
                            {"id": "gpt-image-1"},
                            {"id": "dall-e-3"},
                        ]
                    }
                ).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        seen = {}

        def fake_urlopen(req, timeout=0):
            seen["url"] = req.full_url
            seen["method"] = req.get_method()
            seen["auth"] = req.headers.get("Authorization")
            return _Resp()

        buf = io.StringIO()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=True):
            with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
                with contextlib.redirect_stdout(buf):
                    rc = main(["provider-models", "--openai", "--task", "text_to_image"])

        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertEqual(
            seen,
            {
                "url": "https://api.openai.com/v1/models",
                "method": "GET",
                "auth": "Bearer sk-test",
            },
        )
        self.assertIn("gpt-image-1", out)
        self.assertIn("dall-e-3", out)
        self.assertNotIn("gpt-4.1", out)

    def test_repl_help_prioritizes_small_local_examples(self):
        from abstractvision.cli import _repl_help

        out = _repl_help()
        self.assertIn("runwayml/stable-diffusion-v1-5", out)
        self.assertIn("abstractvision model-presets", out)
        self.assertIn("cache-only by default", out)
        self.assertIn("black-forest-labs/FLUX.2-klein-4B", out)
        self.assertIn("/backend sdcpp <model_key|model.gguf|model.safetensors> [sd_cli_path]", out)
        self.assertIn("abstractvision download-model flux2-klein-base-4b --provider sdcpp", out)
        self.assertIn("/provider-models", out)
        self.assertIn("--negative-prompt", out)
        self.assertNotIn("FLUX.2-klein-9B", out)
        self.assertNotIn("--negative ...", out)

    def test_model_presets_lists_8bit_mlx_target(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["model-presets", "--target", "mlx"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("target: mlx", out)
        self.assertIn("provider/engine: any", out)
        self.assertIn("mflux", out)
        self.assertIn("AITRADER/FLUX2-klein-4B-mlx-8bit", out)
        self.assertIn("deepsweet/FLUX.2-klein-9B-MLX-Q8", out)
        self.assertIn("mlx-community/Qwen-Image-2512-8bit", out)
        self.assertIn("carsenk/z-image-turbo-mflux-8bit", out)
        self.assertNotIn("argmaxinc/mlx-stable-diffusion-3-medium", out)

    def test_model_presets_can_filter_by_engine(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["model-presets", "--target", "mlx", "--provider", "mflux"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("provider/engine: mflux", out)
        self.assertIn("flux2-klein-4b", out)
        self.assertNotIn("stable-diffusion-3-medium", out)

    def test_model_presets_rejects_generic_mlx_provider(self):
        from abstractvision.cli import main

        with self.assertRaisesRegex(SystemExit, "generic MLX image backend"):
            main(["model-presets", "--provider", "mlx"])

    def test_model_catalog_lists_8bit_and_fallback_entries(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["model-catalog", "--all-targets"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("policy: recommend 8-bit", out)
        self.assertIn("Qwen/Qwen-Image-2512", out)
        self.assertIn("stable-diffusion-v1-5/stable-diffusion-v1-5", out)
        self.assertIn("baidu/ERNIE-Image", out)
        self.assertIn("stabilityai/stable-diffusion-xl-base-1.0", out)

    def test_model_catalog_task_filter_limits_rows(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["model-catalog", "--task", "image_to_image", "--all-targets"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("black-forest-labs/FLUX.2-klein-4B", out)
        self.assertIn("stable-diffusion-v1-5/stable-diffusion-v1-5", out)
        self.assertNotIn("Tongyi-MAI/Z-Image-Turbo", out)
        self.assertNotIn("Qwen/Qwen-Image-2512", out)

    def test_model_catalog_json_is_parseable(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["model-catalog", "--all-targets", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertIsInstance(payload, list)
        self.assertTrue(any(entry.get("model_id") == "Qwen/Qwen-Image-2512" for entry in payload))

    def test_download_model_refuses_non_8bit_fallback_by_default(self):
        from abstractvision.cli import main

        with self.assertRaises(SystemExit) as ctx:
            main(["download-model", "stable-diffusion"])
        self.assertIn("No 8-bit preset", str(ctx.exception))

    def test_download_model_diffusers_provider_implies_target_and_allows_full_snapshots_by_default(self):
        from abstractvision.cli import main

        calls = {}

        def fake_download(preset, *, model_dir=None, token=None, max_workers=4):
            calls["repo_id"] = preset.repo_id
            calls["target"] = preset.target
            calls["bits"] = preset.quantization_bits
            return Path("/tmp/hf-cache") / f"models--{preset.repo_id.replace('/', '--')}" / "snapshots" / "abc123"

        buf = io.StringIO()
        with patch("abstractvision.cli.download_model_preset", new=fake_download):
            with contextlib.redirect_stdout(buf):
                rc = main(["download-model", "stable-diffusion", "--provider", "diffusers"])

        self.assertEqual(rc, 0)
        self.assertEqual(calls["repo_id"], "stable-diffusion-v1-5/stable-diffusion-v1-5")
        self.assertEqual(calls["target"], "diffusers")
        self.assertEqual(calls["bits"], 16)

    def test_diffusers_provider_uses_cached_snapshot_and_migrates_legacy_tree(self):
        from abstractvision.cli import _build_manager_from_args

        with tempfile.TemporaryDirectory() as legacy_td, tempfile.TemporaryDirectory() as cache_td:
            legacy_root = Path(legacy_td)
            legacy_dir = legacy_root / "qwen-image-2512-diffusers"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dir / "model_index.json").write_text("{}", encoding="utf-8")

            with patch.dict(
                "os.environ",
                {"ABSTRACTVISION_MODEL_DIR": legacy_td, "HF_HUB_CACHE": cache_td},
                clear=True,
            ):
                args = types.SimpleNamespace(
                    store_dir=None,
                    provider="diffusers",
                    backend=None,
                    model="qwen-image",
                    model_id=None,
                    mflux_model=None,
                    mflux_base_model=None,
                    mflux_model_dir=None,
                    mflux_allow_download=False,
                    base_url=None,
                    api_key=None,
                    timeout_s=300.0,
                    models_path=None,
                    images_generations_path="/images/generations",
                    images_edits_path="/images/edits",
                    text_to_video_path=None,
                    image_to_video_path=None,
                    image_to_video_mode="multipart",
                    diffusers_device="cpu",
                    diffusers_torch_dtype=None,
                    diffusers_allow_download=False,
                    diffusers_auto_retry_fp32=True,
                    sdcpp_bin="sd-cli",
                    sdcpp_model=None,
                    sdcpp_diffusion_model=None,
                    sdcpp_vae=None,
                    sdcpp_llm=None,
                    sdcpp_llm_vision=None,
                    sdcpp_extra_args=None,
                    capabilities_model_id=None,
                )

                vm = _build_manager_from_args(args)
                backend = vm.backend
                cfg = getattr(backend, "_cfg", None)
                self.assertIsNotNone(cfg)
                model_id = str(getattr(cfg, "model_id", ""))
                self.assertEqual(model_id, "Qwen/Qwen-Image-2512")
                self.assertTrue((Path(cache_td) / "models--Qwen--Qwen-Image-2512").exists())
                self.assertFalse(legacy_dir.exists())

    def test_sdcpp_provider_resolves_cached_component_bundle_from_model_key(self):
        from abstractvision.cli import _build_manager_from_args
        from abstractvision.model_cache import import_directory_to_hf_cache

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

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                args = types.SimpleNamespace(
                    store_dir=None,
                    provider="sdcpp",
                    backend=None,
                    model="flux2-klein-base-4b",
                    model_id=None,
                    mflux_model=None,
                    mflux_base_model=None,
                    mflux_model_dir=None,
                    mflux_allow_download=False,
                    base_url=None,
                    api_key=None,
                    timeout_s=300.0,
                    models_path=None,
                    images_generations_path="/images/generations",
                    images_edits_path="/images/edits",
                    text_to_video_path=None,
                    image_to_video_path=None,
                    image_to_video_mode="multipart",
                    diffusers_device="cpu",
                    diffusers_torch_dtype=None,
                    diffusers_allow_download=False,
                    diffusers_auto_retry_fp32=True,
                    sdcpp_bin="sd-cli",
                    sdcpp_model=None,
                    sdcpp_diffusion_model=None,
                    sdcpp_vae=None,
                    sdcpp_llm=None,
                    sdcpp_llm_vision=None,
                    sdcpp_extra_args=None,
                    capabilities_model_id=None,
                )

                vm = _build_manager_from_args(args)
                backend = vm.backend
                cfg = getattr(backend, "_cfg", None)
                self.assertIsNotNone(cfg)
                self.assertIsNone(getattr(cfg, "model", None))
                self.assertTrue(str(getattr(cfg, "diffusion_model", "")).endswith("flux-2-klein-base-4b-Q8_0.gguf"))
                self.assertTrue(str(getattr(cfg, "vae", "")).endswith("vae/diffusion_pytorch_model.safetensors"))
                self.assertTrue(str(getattr(cfg, "llm", "")).endswith("Qwen3-4B-Q4_K_M.gguf"))

    def test_download_model_selects_8bit_without_network_when_mocked(self):
        from abstractvision.cli import main

        calls = {}

        def fake_download(preset, *, model_dir=None, token=None, max_workers=4):
            calls["repo_id"] = preset.repo_id
            calls["model_dir"] = model_dir
            calls["max_workers"] = max_workers
            return Path("/tmp/hf-cache") / f"models--{preset.repo_id.replace('/', '--')}" / "snapshots" / "abc123"

        buf = io.StringIO()
        with patch("abstractvision.cli.download_model_preset", new=fake_download):
            with contextlib.redirect_stdout(buf):
                rc = main(
                    [
                        "download-model",
                        "flux2-klein-9b",
                        "--provider",
                        "mflux",
                        "--model-dir",
                        "/tmp/legacy-models",
                        "--max-workers",
                        "2",
                    ]
                )
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertEqual(calls["repo_id"], "deepsweet/FLUX.2-klein-9B-MLX-Q8")
        self.assertEqual(str(calls["model_dir"]), "/tmp/legacy-models")
        self.assertEqual(calls["max_workers"], 2)
        self.assertIn("/tmp/hf-cache/models--deepsweet--FLUX.2-klein-9B-MLX-Q8/snapshots/abc123", out)

    def test_download_model_accepts_repo_id_fallback(self):
        from abstractvision.cli import main

        calls = {}

        def fake_snapshot(repo_id, *, token=None, revision=None, allow_patterns=None, ignore_patterns=None, cache_dir=None, local_files_only=False, max_workers=4):
            calls["repo_id"] = repo_id
            calls["cache_dir"] = cache_dir
            calls["local_files_only"] = local_files_only
            calls["max_workers"] = max_workers
            return Path("/tmp/hf-cache") / "snap"

        buf = io.StringIO()
        with patch.dict("os.environ", {"HF_HUB_CACHE": "/tmp/hf-cache"}, clear=True):
            with patch("abstractvision.cli.download_hf_repo_snapshot", new=fake_snapshot):
                with contextlib.redirect_stdout(buf):
                    rc = main(["download-model", "some-org/some-model", "--max-workers", "2"])

        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertEqual(calls["repo_id"], "some-org/some-model")
        self.assertEqual(calls["cache_dir"], "/tmp/hf-cache")
        self.assertFalse(calls["local_files_only"])
        self.assertEqual(calls["max_workers"], 2)
        self.assertIn("/tmp/hf-cache/snap", out)

    def test_download_model_uses_curated_preset_when_repo_id_is_known(self):
        from abstractvision.cli import main

        calls = {}

        def fake_download(preset, *, model_dir=None, token=None, max_workers=4):
            calls["repo_id"] = preset.repo_id
            calls["local_dir_name"] = preset.local_dir_name
            calls["max_workers"] = max_workers
            return Path("/tmp/hf-cache") / f"models--{preset.repo_id.replace('/', '--')}" / "snapshots" / "abc123"

        buf = io.StringIO()
        with patch("abstractvision.cli.download_model_preset", new=fake_download):
            with contextlib.redirect_stdout(buf):
                rc = main(["download-model", "mlx-community/Qwen-Image-2512-8bit", "--max-workers", "2"])

        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertEqual(calls["repo_id"], "mlx-community/Qwen-Image-2512-8bit")
        self.assertEqual(calls["local_dir_name"], "qwen-image-2512-mlx-8bit")
        self.assertEqual(calls["max_workers"], 2)
        self.assertIn("/tmp/hf-cache/models--mlx-community--Qwen-Image-2512-8bit/snapshots/abc123", out)

    def test_download_model_sdcpp_prints_resolved_bundle_paths(self):
        from abstractvision.cli import main
        from abstractvision.model_downloads import SdcppModelSelection

        def fake_download(preset, *, model_dir=None, token=None, max_workers=4):
            return Path("/tmp/hf-cache") / f"models--{preset.repo_id.replace('/', '--')}" / "snapshots" / "abc123"

        def fake_resolve(name, *, allow_download=False, token=None, max_workers=4):
            self.assertEqual(name, "flux2-klein-base-4b")
            self.assertTrue(allow_download)
            return SdcppModelSelection(
                key="flux2-klein-base-4b",
                repo_id="leejet/FLUX.2-klein-base-4B-GGUF",
                diffusion_model="/tmp/hf-cache/flux-2-klein-base-4b-Q8_0.gguf",
                vae="/tmp/hf-cache/vae/diffusion_pytorch_model.safetensors",
                llm="/tmp/hf-cache/Qwen3-4B-Q4_K_M.gguf",
            )

        buf = io.StringIO()
        with patch("abstractvision.cli.download_model_preset", new=fake_download):
            with patch("abstractvision.cli.resolve_sdcpp_model_selection", new=fake_resolve):
                with contextlib.redirect_stdout(buf):
                    rc = main(["download-model", "flux2-klein-base-4b", "--provider", "sdcpp"])

        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("resolved_diffusion_model: /tmp/hf-cache/flux-2-klein-base-4b-Q8_0.gguf", out)
        self.assertIn("resolved_vae: /tmp/hf-cache/vae/diffusion_pytorch_model.safetensors", out)
        self.assertIn("resolved_llm: /tmp/hf-cache/Qwen3-4B-Q4_K_M.gguf", out)

    def test_download_model_rejects_generic_mlx_engine_prefix(self):
        from abstractvision.cli import main

        with self.assertRaisesRegex(SystemExit, "generic MLX image backend"):
            main(["download-model", "mlx", "flux2-klein-4b"])

    def test_download_model_supports_registry_only_hf_snapshot_preset(self):
        from abstractvision.cli import main

        calls = {}

        def fake_download(preset, *, model_dir=None, token=None, max_workers=4):
            calls["repo_id"] = preset.repo_id
            calls["target"] = preset.target
            calls["engine"] = preset.engine
            calls["bits"] = preset.quantization_bits
            return Path("/tmp/hf-cache") / "models--tencent--HunyuanImage-3.0" / "snapshots" / "abc123"

        buf = io.StringIO()
        with patch("abstractvision.cli.download_model_preset", new=fake_download):
            with contextlib.redirect_stdout(buf):
                rc = main(["download-model", "hunyuanimage-3.0"])

        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertEqual(calls["repo_id"], "tencent/HunyuanImage-3.0")
        self.assertEqual(calls["target"], "hf-snapshot")
        self.assertEqual(calls["engine"], "transformers")
        self.assertEqual(calls["bits"], 16)
        self.assertIn("/tmp/hf-cache/models--tencent--HunyuanImage-3.0/snapshots/abc123", out)

    def test_download_model_surfaces_friendly_hf_access_message(self):
        from abstractvision.cli import main
        from abstractvision.model_downloads import HuggingFaceAccessError

        def fake_download(_preset, *, model_dir=None, token=None, max_workers=4):
            raise HuggingFaceAccessError(
                "stabilityai/stable-diffusion-3.5-medium",
                raw_message="403 Client Error: gated repo",
                status_code=403,
            )

        buf = io.StringIO()
        err = io.StringIO()
        with patch("abstractvision.cli.download_model_preset", new=fake_download):
            with patch("sys.stdin.isatty", return_value=False):
                with contextlib.redirect_stdout(buf):
                    with contextlib.redirect_stderr(err):
                        with self.assertRaises(SystemExit) as ctx:
                            main(["download-model", "sd3.5-medium", "--provider", "diffusers"])

        msg = str(ctx.exception)
        self.assertIn("https://huggingface.co/stabilityai/stable-diffusion-3.5-medium", msg)
        self.assertIn("https://huggingface.co/settings/tokens", msg)
        self.assertIn("hf auth login", msg)

    def test_download_model_can_prompt_for_hf_token_and_retry(self):
        from abstractvision.cli import main
        from abstractvision.model_downloads import HuggingFaceAccessError

        calls = {"count": 0, "tokens": []}

        def fake_download(_preset, *, model_dir=None, token=None, max_workers=4):
            calls["count"] += 1
            calls["tokens"].append(token)
            if calls["count"] == 1:
                raise HuggingFaceAccessError(
                    "stabilityai/stable-diffusion-3.5-medium",
                    raw_message="403 Client Error: gated repo",
                    status_code=403,
                )
            return Path("/tmp/hf-cache/models--stabilityai--stable-diffusion-3.5-medium/snapshots/abc123")

        buf = io.StringIO()
        err = io.StringIO()
        with patch.dict("os.environ", {}, clear=True):
            with patch("abstractvision.cli.download_model_preset", new=fake_download):
                with patch("sys.stdin.isatty", return_value=True):
                    with patch("abstractvision.cli.getpass.getpass", return_value="hf_test_token"):
                        with contextlib.redirect_stdout(buf):
                            with contextlib.redirect_stderr(err):
                                rc = main(["download-model", "sd3.5-medium", "--provider", "diffusers"])

        self.assertEqual(rc, 0)
        self.assertEqual(calls["tokens"], [None, "hf_test_token"])
        self.assertIn("/tmp/hf-cache/models--stabilityai--stable-diffusion-3.5-medium/snapshots/abc123", buf.getvalue())

    def test_repl_state_starts_unconfigured_without_backend_env(self):
        from abstractvision.cli import DEFAULT_DIFFUSERS_DEVICE, _ReplState

        with patch.dict("os.environ", {}, clear=True):
            state = _ReplState()

        self.assertEqual(state.backend_kind, "")
        self.assertIsNone(state.model_id)
        self.assertEqual(state.diffusers_device, DEFAULT_DIFFUSERS_DEVICE)
        self.assertFalse(state.diffusers_allow_download)
        self.assertEqual(state.defaults["t2i"]["width"], 512)
        self.assertEqual(state.defaults["t2i"]["height"], 512)
        self.assertIn("t2v", state.defaults)
        self.assertIsNone(state.defaults["t2v"]["width"])
        self.assertIsNone(state.defaults["t2v"]["fps"])

    def test_repl_state_defaults_to_openai_when_base_url_is_configured(self):
        from abstractvision.cli import _ReplState

        with patch.dict("os.environ", {"OPENAI_BASE_URL": "http://localhost:1234/v1"}, clear=True):
            state = _ReplState()

        self.assertEqual(state.backend_kind, "openai")
        self.assertEqual(state.base_url, "http://localhost:1234/v1")
        self.assertIsNone(state.model_id)

    def test_repl_state_openai_override_does_not_inherit_diffusers_model(self):
        from abstractvision.cli import _ReplState

        with patch.dict("os.environ", {"ABSTRACTVISION_BACKEND": "openai"}, clear=True):
            state = _ReplState()

        self.assertEqual(state.backend_kind, "openai")
        self.assertIsNone(state.model_id)

    def test_repl_accepts_sdcpp_single_model_backend(self):
        from abstractvision.cli import main

        commands = iter(
            [
                "/backend sdcpp /models/sd-v1-5.gguf /opt/sd-cli",
                "/config",
                "/exit",
            ]
        )
        buf = io.StringIO()
        with patch("builtins.input", side_effect=lambda _prompt: next(commands)):
            with contextlib.redirect_stdout(buf):
                rc = main(["cli"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn('"backend_kind": "sdcpp"', out)
        self.assertIn('"sdcpp_bin": "/opt/sd-cli"', out)
        self.assertIn('"sdcpp_model": "/models/sd-v1-5.gguf"', out)
        self.assertIn('"sdcpp_diffusion_model": null', out)
        self.assertIn('"model_id": null', out)

    def test_cli_has_t2v_command(self):
        from abstractvision.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["t2v", "hello", "--num-frames", "9", "--fps", "8"])
        self.assertEqual(args.cmd, "t2v")
        self.assertEqual(args.prompt, "hello")
        self.assertEqual(args.num_frames, 9)
        self.assertEqual(args.fps, 8)
        self.assertTrue(callable(args._fn))

    def test_repl_build_manager_resolves_cached_sdcpp_component_bundle_from_model_key(self):
        from abstractvision.cli import _ReplState, _build_manager_from_state
        from abstractvision.model_cache import import_directory_to_hf_cache

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

            with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                state = _ReplState()
                state.backend_kind = "sdcpp"
                state.sdcpp_bin = "sd-cli"
                state.sdcpp_model = "flux2-klein-base-4b"

                vm = _build_manager_from_state(state)
                backend = vm.backend
                cfg = getattr(backend, "_cfg", None)
                self.assertIsNotNone(cfg)
                self.assertIsNone(getattr(cfg, "model", None))
                self.assertTrue(str(getattr(cfg, "diffusion_model", "")).endswith("flux-2-klein-base-4b-Q8_0.gguf"))
                self.assertTrue(str(getattr(cfg, "vae", "")).endswith("vae/diffusion_pytorch_model.safetensors"))
                self.assertTrue(str(getattr(cfg, "llm", "")).endswith("Qwen3-4B-Q4_K_M.gguf"))


if __name__ == "__main__":
    unittest.main()
