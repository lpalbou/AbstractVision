import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestPlaygroundServer(unittest.TestCase):
    def test_normalizes_raw_and_provider_model_ids(self):
        from abstractvision.playground_server import normalize_model_id_for_backend

        self.assertEqual(
            normalize_model_id_for_backend("runwayml/stable-diffusion-v1-5"),
            ("diffusers", "runwayml/stable-diffusion-v1-5"),
        )
        self.assertEqual(
            normalize_model_id_for_backend("diffusers/runwayml/stable-diffusion-v1-5"),
            ("diffusers", "runwayml/stable-diffusion-v1-5"),
        )
        self.assertEqual(normalize_model_id_for_backend("diffusers/default")[0], "diffusers")
        self.assertEqual(
            normalize_model_id_for_backend("mlx-community/Qwen-Image-2512-8bit"),
            ("diffusers", "mlx-community/Qwen-Image-2512-8bit"),
        )
        self.assertEqual(
            normalize_model_id_for_backend("mflux/AbstractFramework/flux.2-klein-4b-4bit"),
            ("mflux", "AbstractFramework/flux.2-klein-4b-4bit"),
        )
        self.assertEqual(normalize_model_id_for_backend("sdcpp/default"), ("sdcpp", None))
        self.assertEqual(
            normalize_model_id_for_backend("openai-compatible/dall-e-3"), ("openai", "dall-e-3")
        )
        with self.assertRaisesRegex(ValueError, "generic MLX image/video backend"):
            normalize_model_id_for_backend("mlx/flux2-klein-4b")

    def test_loads_raw_huggingface_model_id_without_provider_prefix(self):
        import abstractvision.backends.huggingface_diffusers as hf_backend
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        seen = {}

        class FakeBackend:
            def __init__(self, *, config):
                seen["config"] = config

        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))

        with patch.object(hf_backend, "HuggingFaceDiffusersVisionBackend", FakeBackend):
            with patch(
                "abstractvision.playground_server._local_runtime_available", return_value=True
            ):
                out = state.load_model("runwayml/stable-diffusion-v1-5")

        self.assertTrue(out["ok"])
        self.assertEqual(seen["config"].model_id, "runwayml/stable-diffusion-v1-5")
        self.assertFalse(seen["config"].allow_download)
        self.assertEqual(out["active"]["model_id"], "runwayml/stable-diffusion-v1-5")

    def test_lists_cached_huggingface_registry_models(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            with patch.dict("os.environ", {"HF_HUB_CACHE": td}, clear=True):
                cache = Path(td)
                snap = (
                    cache
                    / "models--stable-diffusion-v1-5--stable-diffusion-v1-5"
                    / "snapshots"
                    / "abc123"
                )
                (snap / "unet").mkdir(parents=True)
                (snap / "model_index.json").write_text("{}", encoding="utf-8")
                (snap / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
                refs = cache / "models--stable-diffusion-v1-5--stable-diffusion-v1-5" / "refs"
                refs.mkdir(parents=True, exist_ok=True)
                (refs / "main").write_text("abc123", encoding="utf-8")

                state = PlaygroundState(
                    PlaygroundServerConfig(
                        diffusers_cache_dir=str(cache),
                        diffusers_allow_download=False,
                        default_model_id="",
                    )
                )
                out = state.list_models()

        models = {m["id"]: m for m in out["models"]}
        self.assertIn("stable-diffusion-v1-5/stable-diffusion-v1-5", models)
        self.assertTrue(models["stable-diffusion-v1-5/stable-diffusion-v1-5"]["cached"])
        self.assertIn(
            "configured cache", models["stable-diffusion-v1-5/stable-diffusion-v1-5"]["cached_in"]
        )

    def test_lists_cached_sdcpp_gguf_preset_as_loadable_image_edit_model(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            gguf_snap = (
                cache / "models--unsloth--Qwen-Image-Edit-2511-GGUF" / "snapshots" / "gguf123"
            )
            gguf_snap.mkdir(parents=True)
            (gguf_snap / "qwen-image-edit-2511-Q8_0.gguf").write_bytes(b"x")
            gguf_refs = cache / "models--unsloth--Qwen-Image-Edit-2511-GGUF" / "refs"
            gguf_refs.mkdir(parents=True, exist_ok=True)
            (gguf_refs / "main").write_text("gguf123", encoding="utf-8")

            with patch("abstractvision.model_downloads.local_model_profile", return_value="cpu"):
                with patch(
                    "abstractvision.playground_server.local_model_profile", return_value="cpu"
                ):
                    with patch(
                        "abstractvision.playground_server._local_runtime_available",
                        return_value=True,
                    ):
                        state = PlaygroundState(
                            PlaygroundServerConfig(
                                diffusers_cache_dir=str(cache),
                                diffusers_allow_download=False,
                                default_model_id="",
                            )
                        )
                        out = state.list_models()

        matching = [m for m in out["models"] if m["load_id"] == "sdcpp/qwen-image-edit-2511-gguf"]
        self.assertEqual(len(matching), 1)
        chosen = matching[0]
        self.assertEqual(chosen["id"], "Qwen/Qwen-Image-Edit-2511")
        self.assertEqual(chosen["backend"], "sdcpp")
        self.assertEqual(chosen["engine"], "stable-diffusion.cpp")
        self.assertEqual(chosen["target"], "gguf")
        self.assertEqual(chosen["bits"], 8)
        self.assertTrue(chosen["cached"])
        self.assertTrue(chosen["loadable"])
        self.assertIn("image_to_image", chosen["tasks"])
        self.assertIn("configured cache", " ".join(chosen["cached_in"]))

    def test_lists_cached_mflux_registry_variants(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            snap = (
                cache / "models--AbstractFramework--flux.2-klein-9b-8bit" / "snapshots" / "abc123"
            )
            (snap / "transformer").mkdir(parents=True)
            (snap / "transformer" / "0.safetensors").write_bytes(b"x")
            refs = cache / "models--AbstractFramework--flux.2-klein-9b-8bit" / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            (refs / "main").write_text("abc123", encoding="utf-8")

            with patch(
                "abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"
            ):
                with patch(
                    "abstractvision.playground_server.local_model_profile",
                    return_value="apple-silicon",
                ):
                    with patch(
                        "abstractvision.playground_server._local_runtime_available",
                        return_value=True,
                    ):
                        state = PlaygroundState(
                            PlaygroundServerConfig(
                                diffusers_cache_dir=str(cache),
                                diffusers_allow_download=False,
                                default_model_id="",
                            )
                        )
                        out = state.list_models()

        mflux_models = [m for m in out["models"] if m["backend"] == "mlx-gen"]
        self.assertTrue(
            any(
                m["load_id"] == "mlx-gen/AbstractFramework/flux.2-klein-9b-8bit"
                for m in mflux_models
            )
        )
        chosen = next(
            m
            for m in mflux_models
            if m["load_id"] == "mlx-gen/AbstractFramework/flux.2-klein-9b-8bit"
        )
        self.assertEqual(chosen["id"], "AbstractFramework/flux.2-klein-9b-8bit")
        self.assertTrue(chosen["cached"])

    def test_lists_cached_alternate_mflux_registry_variants(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            snap = cache / "models--AbstractFramework--z-image-turbo-8bit" / "snapshots" / "abc123"
            (snap / "transformer").mkdir(parents=True)
            (snap / "transformer" / "0.safetensors").write_bytes(b"x")
            refs = cache / "models--AbstractFramework--z-image-turbo-8bit" / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            (refs / "main").write_text("abc123", encoding="utf-8")

            with patch(
                "abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"
            ):
                with patch(
                    "abstractvision.playground_server.local_model_profile",
                    return_value="apple-silicon",
                ):
                    with patch(
                        "abstractvision.playground_server._local_runtime_available",
                        return_value=True,
                    ):
                        state = PlaygroundState(
                            PlaygroundServerConfig(
                                diffusers_cache_dir=str(cache),
                                diffusers_allow_download=False,
                                default_model_id="",
                            )
                        )
                        out = state.list_models()

        chosen = next(
            m
            for m in out["models"]
            if m["load_id"] == "mlx-gen/AbstractFramework/z-image-turbo-8bit"
        )
        self.assertEqual(chosen["id"], "AbstractFramework/z-image-turbo-8bit")
        self.assertTrue(chosen["cached"])
        self.assertTrue(chosen["loadable"])
        self.assertIn("configured cache", " ".join(chosen["cached_in"]))

    def test_lists_quarantined_cached_diffusers_registry_models(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            snap = cache / "models--black-forest-labs--FLUX.2-klein-4B" / "snapshots" / "abc123"
            (snap / "unet").mkdir(parents=True, exist_ok=True)
            (snap / "model_index.json").write_text("{}", encoding="utf-8")
            (snap / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
            refs = cache / "models--black-forest-labs--FLUX.2-klein-4B" / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            (refs / "main").write_text("abc123", encoding="utf-8")

            with patch(
                "abstractvision.playground_server._local_runtime_available", return_value=True
            ):
                state = PlaygroundState(
                    PlaygroundServerConfig(
                        diffusers_cache_dir=str(Path(td) / "empty"),
                        diffusers_allow_download=False,
                        default_model_id="",
                    )
                )
                with patch(
                    "abstractvision.playground_server.framework_hf_cache_roots",
                    return_value=[("quarantined HF cache", cache)],
                ):
                    out = state.list_models()

        chosen = next(
            m
            for m in out["models"]
            if m["id"] == "black-forest-labs/FLUX.2-klein-4B" and m["backend"] == "diffusers"
        )
        self.assertTrue(chosen["cached"])
        self.assertTrue(chosen["loadable"])
        self.assertIn("quarantined HF cache", " ".join(chosen["cached_in"]))

    def test_marks_incomplete_alternate_mflux_cache_as_not_loadable(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            lock_dir = cache / ".locks" / "models--AbstractFramework--z-image-turbo-8bit"
            lock_dir.mkdir(parents=True, exist_ok=True)

            with patch(
                "abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"
            ):
                with patch(
                    "abstractvision.playground_server.local_model_profile",
                    return_value="apple-silicon",
                ):
                    with patch(
                        "abstractvision.playground_server.cached_hf_model_sources", return_value=[]
                    ):
                        state = PlaygroundState(
                            PlaygroundServerConfig(
                                diffusers_cache_dir=str(cache),
                                diffusers_allow_download=False,
                                default_model_id="",
                            )
                        )
                        out = state.list_models()

        chosen = next(
            m
            for m in out["models"]
            if m["load_id"] == "mlx-gen/AbstractFramework/z-image-turbo-8bit"
        )
        self.assertFalse(chosen["cached"])
        self.assertFalse(chosen["loadable"])
        self.assertIn("incomplete HF cache:", chosen["cached_in"][0])

    def test_surfaces_download_only_registry_models_for_playground_catalog(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        state = PlaygroundState(
            PlaygroundServerConfig(
                diffusers_allow_download=False,
                default_model_id="",
            )
        )
        out = state.list_models()

        hunyuan = next(m for m in out["models"] if m["id"] == "tencent/HunyuanImage-3.0")
        self.assertFalse(hunyuan["loadable"])
        self.assertEqual(hunyuan["engine"], "transformers")
        self.assertIn("download only", hunyuan["cached_in"])

    def test_catalog_exposes_task_specs_for_registry_models(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        state = PlaygroundState(
            PlaygroundServerConfig(
                diffusers_allow_download=False,
                default_model_id="",
            )
        )
        out = state.list_models()

        self.assertFalse(any(m["id"] == "zai-org/GLM-Image" for m in out["models"]))

        ernie = next(m for m in out["models"] if m["id"] == "baidu/ERNIE-Image-Turbo")
        self.assertEqual(ernie["tasks"], ["text_to_image"])
        self.assertIn("text_to_image", ernie["task_specs"])
        self.assertNotIn("image_to_image", ernie["task_specs"])

    def test_catalog_hides_temporarily_disabled_cogvideox_2b(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        state = PlaygroundState(
            PlaygroundServerConfig(
                diffusers_allow_download=False,
                default_model_id="",
            )
        )
        out = state.list_models()

        self.assertFalse(any(m["id"] == "zai-org/CogVideoX-2b" for m in out["models"]))

    def test_catalog_surfaces_mflux_flux2_klein_for_image_to_image_and_text_to_image(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState
        from abstractvision.model_downloads import model_presets as real_model_presets

        mflux_flux2 = next(
            preset
            for preset in real_model_presets(
                target="mlx",
                engine="mflux",
                include_non_8bit=True,
                include_all_targets=False,
            )
            if preset.key == "flux2-klein-9b"
        )
        with (
            patch(
                "abstractvision.playground_server.catalog_target_scope",
                return_value={"diffusers", "mlx"},
            ),
            patch(
                "abstractvision.playground_server.model_presets",
                return_value=[mflux_flux2],
            ),
        ):
            state = PlaygroundState(
                PlaygroundServerConfig(
                    diffusers_allow_download=False,
                    default_model_id="",
                )
            )
            out = state.list_models()

        flux2 = next(
            m
            for m in out["models"]
            if m["load_id"] == "mlx-gen/AbstractFramework/flux.2-klein-9b-4bit"
        )
        self.assertEqual(flux2["tasks"], ["image_to_image", "text_to_image"])
        self.assertIn("image_to_image", flux2["task_specs"])

    def test_surface_tasks_for_backend_preserves_registry_tasks_when_diffusers_probe_fails(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        state = PlaygroundState(
            PlaygroundServerConfig(
                diffusers_allow_download=False,
                default_model_id="",
            )
        )

        with patch(
            "abstractvision.backends.huggingface_diffusers.HuggingFaceDiffusersVisionBackend.get_capabilities",
            side_effect=RuntimeError("boom"),
        ):
            tasks, task_specs = state._surface_tasks_for_backend(
                backend="diffusers",
                model_id="zai-org/CogVideoX-2b",
                tasks=["text_to_video"],
                task_specs={"text_to_video": {"params": {"width": {"const": 720}}}},
            )

        self.assertEqual(tasks, ["text_to_video"])
        self.assertIn("text_to_video", task_specs)

    def test_incomplete_cached_diffusers_snapshot_is_not_marked_loadable(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            with patch.dict("os.environ", {"HF_HUB_CACHE": td}, clear=True):
                cache = Path(td)
                snap = cache / "models--baidu--ERNIE-Image-Turbo" / "snapshots" / "abc123"
                (snap / ".cache" / "huggingface" / "download" / "transformer").mkdir(
                    parents=True, exist_ok=True
                )
                (snap / "model_index.json").write_text("{}", encoding="utf-8")
                (snap / "transformer").mkdir(parents=True, exist_ok=True)
                (snap / "transformer" / "config.json").write_text("{}", encoding="utf-8")
                (
                    snap
                    / ".cache"
                    / "huggingface"
                    / "download"
                    / "transformer"
                    / "weights.incomplete"
                ).write_bytes(b"x")
                refs = cache / "models--baidu--ERNIE-Image-Turbo" / "refs"
                refs.mkdir(parents=True, exist_ok=True)
                (refs / "main").write_text("abc123", encoding="utf-8")

                state = PlaygroundState(
                    PlaygroundServerConfig(
                        diffusers_cache_dir=str(cache),
                        diffusers_allow_download=False,
                        default_model_id="",
                    )
                )
                out = state.list_models()

        ernie = next(m for m in out["models"] if m["id"] == "baidu/ERNIE-Image-Turbo")
        self.assertFalse(ernie["cached"])
        self.assertFalse(ernie["loadable"])
        self.assertIn("incomplete HF cache", " ".join(ernie["cached_in"]))

    def test_sharded_diffusers_snapshot_missing_indexed_files_is_not_marked_loadable(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            snap = cache / "models--black-forest-labs--FLUX.2-klein-9B" / "snapshots" / "abc123"
            snap.mkdir(parents=True, exist_ok=True)
            (snap / "model_index.json").write_text("{}", encoding="utf-8")
            (snap / "transformer").mkdir(parents=True, exist_ok=True)
            (snap / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
                '{"weight_map":{"layer.0":"diffusion_pytorch_model-00001-of-00002.safetensors","layer.1":"diffusion_pytorch_model-00002-of-00002.safetensors"}}',
                encoding="utf-8",
            )
            (
                snap / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors"
            ).write_bytes(b"x")
            refs = cache / "models--black-forest-labs--FLUX.2-klein-9B" / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            (refs / "main").write_text("abc123", encoding="utf-8")

            with patch.dict(
                "os.environ", {"HF_HUB_CACHE": str(cache), "HOME": str(cache / "home")}, clear=True
            ):
                state = PlaygroundState(
                    PlaygroundServerConfig(
                        diffusers_cache_dir=str(cache),
                        diffusers_allow_download=False,
                        default_model_id="",
                    )
                )
                out = state.list_models()

        flux = next(
            m
            for m in out["models"]
            if m["id"] == "black-forest-labs/FLUX.2-klein-9B" and m["backend"] == "diffusers"
        )
        self.assertFalse(flux["cached"])
        self.assertFalse(flux["loadable"])
        self.assertIn("incomplete HF cache", " ".join(flux["cached_in"]))

    def test_playground_catalog_hides_apple_only_targets_on_gpu_profiles(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with patch("abstractvision.model_downloads.local_model_profile", return_value="cuda"):
            with patch("abstractvision.playground_server.local_model_profile", return_value="cuda"):
                state = PlaygroundState(
                    PlaygroundServerConfig(
                        diffusers_allow_download=False,
                        default_model_id="",
                    )
                )
                out = state.list_models()

        self.assertEqual(out["platform"], "cuda")
        self.assertTrue(all(m["target"] != "mlx" for m in out["models"]))
        self.assertIn("fp8", out["targets"])

    def test_playground_catalog_hides_gpu_only_targets_on_apple_profiles(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with patch(
            "abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"
        ):
            with patch(
                "abstractvision.playground_server.local_model_profile", return_value="apple-silicon"
            ):
                state = PlaygroundState(
                    PlaygroundServerConfig(
                        diffusers_allow_download=False,
                        default_model_id="",
                    )
                )
                out = state.list_models()

        self.assertEqual(out["platform"], "apple-silicon")
        # Apple Silicon can run GGUF models via stable-diffusion.cpp (Metal); only fp8 is GPU-only.
        self.assertTrue(all(m["target"] != "fp8" for m in out["models"]))
        self.assertIn("mlx", out["targets"])

    def test_playground_marks_mflux_as_not_loadable_when_runtime_is_missing(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            snap = (
                cache / "models--AbstractFramework--flux.2-klein-9b-8bit" / "snapshots" / "abc123"
            )
            (snap / "transformer").mkdir(parents=True)
            (snap / "transformer" / "0.safetensors").write_bytes(b"x")
            refs = cache / "models--AbstractFramework--flux.2-klein-9b-8bit" / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            (refs / "main").write_text("abc123", encoding="utf-8")

            with patch(
                "abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"
            ):
                with patch(
                    "abstractvision.playground_server.local_model_profile",
                    return_value="apple-silicon",
                ):
                    with patch(
                        "abstractvision.playground_server._local_runtime_available",
                        side_effect=lambda backend: backend != "mlx-gen",
                    ):
                        state = PlaygroundState(
                            PlaygroundServerConfig(
                                diffusers_cache_dir=str(cache),
                                diffusers_allow_download=False,
                                default_model_id="",
                            )
                        )
                        out = state.list_models()

        chosen = next(
            m
            for m in out["models"]
            if m["load_id"] == "mlx-gen/AbstractFramework/flux.2-klein-9b-8bit"
        )
        self.assertTrue(chosen["cached"])
        self.assertFalse(chosen["loadable"])
        self.assertIn("mlx-gen runtime missing", " ".join(chosen["cached_in"]))

    def test_default_config_does_not_select_backend_without_backend_env(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with patch.dict("os.environ", {}, clear=True):
            cfg = PlaygroundServerConfig()
            state = PlaygroundState(cfg)
            out = state.list_models()

        self.assertEqual(cfg.backend_kind, "")
        self.assertEqual(cfg.default_model_id, "")
        self.assertIsNone(out["active"])

    def test_default_config_lists_remote_when_base_url_is_set(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with patch.dict("os.environ", {"OPENAI_BASE_URL": "http://localhost:1234/v1"}, clear=True):
            cfg = PlaygroundServerConfig()
            state = PlaygroundState(cfg)
            out = state.list_models()

        self.assertEqual(cfg.backend_kind, "openai")
        models = {m["id"]: m for m in out["models"]}
        self.assertIn("openai-compatible/default", models)
        self.assertEqual(models["openai-compatible/default"]["backend"], "openai")

    def test_generation_job_uses_active_backend_and_returns_b64_json(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState
        from abstractvision.types import GeneratedAsset

        class FakeBackend:
            def generate_image_with_progress(self, request, progress_callback=None):
                if progress_callback:
                    progress_callback(1, request.steps)
                return GeneratedAsset(
                    media_type="image",
                    data=b"\x89PNG\r\n\x1a\nfake",
                    mime_type="image/png",
                    metadata={"prompt": request.prompt},
                )

        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))
        state._active_backend = FakeBackend()
        state._active_backend_kind = "diffusers"
        state._active_model_id = "runwayml/stable-diffusion-v1-5"
        job = state.start_image_generation_job({"prompt": "hello", "steps": 1})

        snap = None
        for _ in range(100):
            snap = state.get_job(job["job_id"])
            if snap["state"] == "succeeded":
                break
            time.sleep(0.01)

        self.assertIsNotNone(snap)
        self.assertEqual(snap["state"], "succeeded")
        self.assertIn("b64_json", snap["result"]["data"][0])
        self.assertEqual(snap["progress"]["step"], 1)

    def test_video_generation_job_uses_active_backend_and_returns_b64_json(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState
        from abstractvision.types import GeneratedAsset

        class FakeBackend:
            def generate_video_with_progress(self, request, progress_callback=None):
                if progress_callback:
                    progress_callback(1, request.steps)
                return GeneratedAsset(
                    media_type="video",
                    data=b"ftyp" + (b"\x00" * 8),
                    mime_type="video/mp4",
                    metadata={"prompt": request.prompt},
                )

        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))
        state._active_backend = FakeBackend()
        state._active_backend_kind = "diffusers"
        state._active_model_id = "zai-org/CogVideoX-2b"
        job = state.start_video_generation_job({"prompt": "hello", "steps": 1})

        snap = None
        for _ in range(100):
            snap = state.get_job(job["job_id"])
            if snap["state"] == "succeeded":
                break
            time.sleep(0.01)

        self.assertIsNotNone(snap)
        self.assertEqual(snap["state"], "succeeded")
        self.assertIn("b64_json", snap["result"]["data"][0])
        self.assertEqual(snap["progress"]["step"], 1)

    def test_generation_job_auto_loads_requested_model_and_normalizes_request(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState
        from abstractvision.types import GeneratedAsset, ImageGenerationRequest

        seen = {}

        class FakeBackend:
            def __init__(self):
                self.unloaded = False
                self.preloaded = False

            def preload(self):
                self.preloaded = True

            def normalize_image_generation_request(
                self, request: ImageGenerationRequest
            ) -> ImageGenerationRequest:
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

            def generate_image_with_progress(self, request, progress_callback=None):
                seen["request"] = request
                if progress_callback:
                    progress_callback(request.steps, request.steps)
                return GeneratedAsset(
                    media_type="image",
                    data=b"\x89PNG\r\n\x1a\nfake",
                    mime_type="image/png",
                    metadata={"prompt": request.prompt},
                )

            def unload(self):
                self.unloaded = True

        backend = FakeBackend()
        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))

        with patch.object(state, "_build_backend", return_value=backend):
            job = state.start_image_generation_job(
                {
                    "prompt": "hello",
                    "model": "mflux/AbstractFramework/flux.2-klein-9b-4bit",
                    "steps": 1,
                    "guidance_scale": 7.0,
                    "negative_prompt": "blur",
                }
            )

        snap = None
        for _ in range(100):
            snap = state.get_job(job["job_id"])
            if snap["state"] == "succeeded":
                break
            time.sleep(0.01)

        self.assertTrue(backend.preloaded)
        self.assertIsNotNone(snap)
        self.assertEqual(snap["state"], "succeeded")
        self.assertEqual(snap["progress"]["total_steps"], 2)
        self.assertEqual(seen["request"].steps, 2)
        self.assertEqual(seen["request"].guidance_scale, 1.0)
        self.assertIsNone(seen["request"].negative_prompt)
        self.assertEqual(
            state.active_snapshot()["model_id"], "mflux/AbstractFramework/flux.2-klein-9b-4bit"
        )

    def test_generation_job_does_not_block_on_requested_model_load(self):
        import abstractvision.playground_server as playground_server
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState
        from abstractvision.types import GeneratedAsset

        class FakeBackend:
            def __init__(self):
                self.preloaded = False

            def preload(self):
                self.preloaded = True

            def generate_image_with_progress(self, request, progress_callback=None):
                if progress_callback:
                    progress_callback(1, request.steps)
                return GeneratedAsset(
                    media_type="image",
                    data=b"\x89PNG\r\n\x1a\nfake",
                    mime_type="image/png",
                    metadata={"prompt": request.prompt},
                )

        targets = []

        class DeferredThread:
            def __init__(self, *, target, name, daemon):
                self.target = target

            def start(self):
                targets.append(self.target)

        backend = FakeBackend()
        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))

        with patch.object(playground_server.threading, "Thread", DeferredThread):
            with patch.object(state, "_build_backend", return_value=backend):
                job = state.start_image_generation_job(
                    {
                        "prompt": "hello",
                        "model": "mflux/AbstractFramework/qwen-image-2512-4bit",
                        "steps": 1,
                    }
                )
                self.assertFalse(backend.preloaded)
                self.assertEqual(job["state"], "queued")
                self.assertIsNone(state.active_snapshot())
                targets[0]()
        snap = state.get_job(job["job_id"])
        self.assertTrue(backend.preloaded)
        self.assertEqual(snap["state"], "succeeded")
        self.assertEqual(
            state.active_snapshot()["model_id"], "mflux/AbstractFramework/qwen-image-2512-4bit"
        )

    def test_load_model_keeps_existing_backend_when_preload_fails(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        class ExistingBackend:
            def __init__(self):
                self.unloaded = False

            def unload(self):
                self.unloaded = True

        class FailingBackend:
            def __init__(self):
                self.unloaded = False

            def preload(self):
                raise RuntimeError("warmup failed")

            def unload(self):
                self.unloaded = True

        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))
        existing = ExistingBackend()
        replacement = FailingBackend()
        state._active_backend = existing
        state._active_backend_kind = "diffusers"
        state._active_model_id = "runwayml/stable-diffusion-v1-5"
        state._active_loaded_at = 123.0

        with patch.object(state, "_build_backend", return_value=replacement):
            with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                state.load_model("mflux/AbstractFramework/flux.2-klein-9b-4bit")

        active = state.active_snapshot()
        self.assertIsNotNone(active)
        self.assertEqual(active["model_id"], "runwayml/stable-diffusion-v1-5")
        self.assertEqual(active["backend"], "diffusers")
        self.assertFalse(existing.unloaded)
        self.assertTrue(replacement.unloaded)

    def test_load_model_can_unload_existing_backend_before_preload(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        events = []

        class ExistingBackend:
            def unload(self):
                events.append("old_unload")

        class ReplacementBackend:
            def preload(self):
                events.append("new_preload")

            def unload(self):
                events.append("new_unload")

        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))
        state._active_backend = ExistingBackend()
        state._active_backend_kind = "diffusers"
        state._active_model_id = "runwayml/stable-diffusion-v1-5"
        state._active_loaded_at = 123.0

        with patch.object(state, "_build_backend", return_value=ReplacementBackend()):
            out = state.load_model(
                "mflux/AbstractFramework/flux.2-klein-9b-4bit", unload_first=True
            )

        self.assertTrue(out["ok"])
        self.assertEqual(events[:2], ["old_unload", "new_preload"])
        self.assertEqual(
            state.active_snapshot()["model_id"], "mflux/AbstractFramework/flux.2-klein-9b-4bit"
        )

    def test_generation_job_uses_backend_snapshot_if_active_model_changes(self):
        import abstractvision.playground_server as playground_server
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState
        from abstractvision.types import GeneratedAsset

        class FakeBackend:
            def __init__(self, name):
                self.name = name
                self.unloaded = False

            def generate_image_with_progress(self, request, progress_callback=None):
                return GeneratedAsset(
                    media_type="image",
                    data=(b"\x89PNG\r\n\x1a\n" + self.name.encode("ascii")),
                    mime_type="image/png",
                    metadata={"backend": self.name},
                )

            def unload(self):
                self.unloaded = True

        targets = []

        class DeferredThread:
            def __init__(self, *, target, name, daemon):
                self.target = target

            def start(self):
                targets.append(self.target)

        first = FakeBackend("first")
        second = FakeBackend("second")
        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))
        state._active_backend = first
        state._active_backend_kind = "diffusers"
        state._active_model_id = "first-model"

        with patch.object(playground_server.threading, "Thread", DeferredThread):
            job = state.start_image_generation_job({"prompt": "hello", "steps": 1})

        state._active_backend = second
        state._active_backend_kind = "openai"
        state._active_model_id = "second-model"
        targets[0]()
        snap = state.get_job(job["job_id"])

        self.assertEqual(snap["state"], "succeeded")
        self.assertEqual(snap["result"]["data"][0]["metadata"]["backend"], "first")
        self.assertFalse(first.unloaded)

    def test_cli_has_playground_command(self):
        from abstractvision.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["playground", "--host", "127.0.0.1", "--port", "8999"])
        self.assertEqual(args.cmd, "playground")
        self.assertEqual(args.port, 8999)
        self.assertTrue(callable(args._fn))


if __name__ == "__main__":
    unittest.main()
