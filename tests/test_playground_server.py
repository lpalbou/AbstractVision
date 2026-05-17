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
        self.assertEqual(normalize_model_id_for_backend("mflux/flux2-klein-4b"), ("mflux", "flux2-klein-4b"))
        self.assertEqual(normalize_model_id_for_backend("sdcpp/default"), ("sdcpp", None))
        self.assertEqual(
            normalize_model_id_for_backend("openai-compatible/dall-e-3"), ("openai", "dall-e-3")
        )

    def test_loads_raw_huggingface_model_id_without_provider_prefix(self):
        import abstractvision.backends.huggingface_diffusers as hf_backend
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        seen = {}

        class FakeBackend:
            def __init__(self, *, config):
                seen["config"] = config

        state = PlaygroundState(PlaygroundServerConfig(default_model_id=""))

        with patch.object(hf_backend, "HuggingFaceDiffusersVisionBackend", FakeBackend):
            with patch("abstractvision.playground_server._local_runtime_available", return_value=True):
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
                snap = cache / "models--runwayml--stable-diffusion-v1-5" / "snapshots" / "abc123"
                (snap / "unet").mkdir(parents=True)
                (snap / "model_index.json").write_text("{}", encoding="utf-8")
                (snap / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
                refs = cache / "models--runwayml--stable-diffusion-v1-5" / "refs"
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
        self.assertIn("runwayml/stable-diffusion-v1-5", models)
        self.assertTrue(models["runwayml/stable-diffusion-v1-5"]["cached"])
        self.assertIn("configured cache", models["runwayml/stable-diffusion-v1-5"]["cached_in"])

    def test_lists_cached_mflux_registry_variants(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            snap = cache / "models--deepsweet--FLUX.2-klein-9B-MLX-Q8" / "snapshots" / "abc123"
            (snap / "transformer").mkdir(parents=True)
            (snap / "transformer" / "0.safetensors").write_bytes(b"x")

            with patch("abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"):
                with patch("abstractvision.playground_server.local_model_profile", return_value="apple-silicon"):
                    with patch("abstractvision.playground_server._local_runtime_available", return_value=True):
                        state = PlaygroundState(
                            PlaygroundServerConfig(
                                diffusers_cache_dir=str(cache),
                                diffusers_allow_download=False,
                                default_model_id="",
                            )
                        )
                        out = state.list_models()

        mflux_models = [m for m in out["models"] if m["backend"] == "mflux"]
        self.assertTrue(any(m["load_id"] == "mflux/flux2-klein-9b" for m in mflux_models))
        chosen = next(m for m in mflux_models if m["load_id"] == "mflux/flux2-klein-9b")
        self.assertEqual(chosen["id"], "black-forest-labs/FLUX.2-klein-9B")
        self.assertTrue(chosen["cached"])

    def test_lists_cached_alternate_mflux_registry_variants(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            snap = cache / "models--andrevp--Z-Image-Turbo-MLX-8bit" / "snapshots" / "abc123"
            (snap / "transformer").mkdir(parents=True)
            (snap / "transformer" / "0.safetensors").write_bytes(b"x")
            refs = cache / "models--andrevp--Z-Image-Turbo-MLX-8bit" / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            (refs / "main").write_text("abc123", encoding="utf-8")

            with patch("abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"):
                with patch("abstractvision.playground_server.local_model_profile", return_value="apple-silicon"):
                    with patch("abstractvision.playground_server._local_runtime_available", return_value=True):
                        state = PlaygroundState(
                            PlaygroundServerConfig(
                                diffusers_cache_dir=str(cache),
                                diffusers_allow_download=False,
                                default_model_id="",
                            )
                        )
                        out = state.list_models()

        chosen = next(m for m in out["models"] if m["load_id"] == "mflux/z-image-turbo")
        self.assertEqual(chosen["id"], "Tongyi-MAI/Z-Image-Turbo")
        self.assertTrue(chosen["cached"])
        self.assertTrue(chosen["loadable"])
        self.assertIn("andrevp/Z-Image-Turbo-MLX-8bit", " ".join(chosen["cached_in"]))

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

            with patch("abstractvision.playground_server._local_runtime_available", return_value=True):
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

        chosen = next(m for m in out["models"] if m["id"] == "black-forest-labs/FLUX.2-klein-4B" and m["backend"] == "diffusers")
        self.assertTrue(chosen["cached"])
        self.assertTrue(chosen["loadable"])
        self.assertIn("quarantined HF cache", " ".join(chosen["cached_in"]))

    def test_marks_incomplete_alternate_mflux_cache_as_not_loadable(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            lock_dir = cache / ".locks" / "models--andrevp--Z-Image-Turbo-MLX-8bit"
            lock_dir.mkdir(parents=True, exist_ok=True)

            with patch("abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"):
                with patch("abstractvision.playground_server.local_model_profile", return_value="apple-silicon"):
                    state = PlaygroundState(
                        PlaygroundServerConfig(
                            diffusers_cache_dir=str(cache),
                            diffusers_allow_download=False,
                            default_model_id="",
                        )
                    )
                    out = state.list_models()

        chosen = next(m for m in out["models"] if m["load_id"] == "mflux/z-image-turbo")
        self.assertFalse(chosen["cached"])
        self.assertFalse(chosen["loadable"])
        self.assertIn("incomplete HF cache:", chosen["cached_in"][0])
        self.assertIn("andrevp/Z-Image-Turbo-MLX-8bit", " ".join(chosen["cached_in"]))

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

        glm = next(m for m in out["models"] if m["id"] == "zai-org/GLM-Image")
        self.assertIn("image_to_image", glm["task_specs"])
        self.assertEqual(glm["task_specs"]["text_to_image"]["params"]["steps"]["default"], 20)
        self.assertEqual(glm["task_specs"]["text_to_image"]["params"]["guidance_scale"]["default"], 1.5)

        ernie = next(m for m in out["models"] if m["id"] == "baidu/ERNIE-Image-Turbo")
        self.assertEqual(ernie["tasks"], ["text_to_image"])
        self.assertNotIn("image_to_image", ernie["task_specs"])

    def test_incomplete_cached_diffusers_snapshot_is_not_marked_loadable(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            with patch.dict("os.environ", {"HF_HUB_CACHE": td}, clear=True):
                cache = Path(td)
                snap = cache / "models--baidu--ERNIE-Image-Turbo" / "snapshots" / "abc123"
                (snap / ".cache" / "huggingface" / "download" / "transformer").mkdir(parents=True, exist_ok=True)
                (snap / "model_index.json").write_text("{}", encoding="utf-8")
                (snap / "transformer").mkdir(parents=True, exist_ok=True)
                (snap / "transformer" / "config.json").write_text("{}", encoding="utf-8")
                (snap / ".cache" / "huggingface" / "download" / "transformer" / "weights.incomplete").write_bytes(b"x")
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
            (snap / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(b"x")
            refs = cache / "models--black-forest-labs--FLUX.2-klein-9B" / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            (refs / "main").write_text("abc123", encoding="utf-8")

            with patch.dict("os.environ", {"HF_HUB_CACHE": str(cache), "HOME": str(cache / "home")}, clear=True):
                state = PlaygroundState(
                    PlaygroundServerConfig(
                        diffusers_cache_dir=str(cache),
                        diffusers_allow_download=False,
                        default_model_id="",
                    )
                )
                out = state.list_models()

        flux = next(m for m in out["models"] if m["id"] == "black-forest-labs/FLUX.2-klein-9B" and m["backend"] == "diffusers")
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

        with patch("abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"):
            with patch("abstractvision.playground_server.local_model_profile", return_value="apple-silicon"):
                state = PlaygroundState(
                    PlaygroundServerConfig(
                        diffusers_allow_download=False,
                        default_model_id="",
                    )
                )
                out = state.list_models()

        self.assertEqual(out["platform"], "apple-silicon")
        self.assertTrue(all(m["target"] not in {"fp8", "gguf"} for m in out["models"]))
        self.assertIn("mlx", out["targets"])

    def test_playground_marks_mflux_as_not_loadable_when_runtime_is_missing(self):
        from abstractvision.playground_server import PlaygroundServerConfig, PlaygroundState

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            snap = cache / "models--deepsweet--FLUX.2-klein-9B-MLX-Q8" / "snapshots" / "abc123"
            (snap / "transformer").mkdir(parents=True)
            (snap / "transformer" / "0.safetensors").write_bytes(b"x")
            refs = cache / "models--deepsweet--FLUX.2-klein-9B-MLX-Q8" / "refs"
            refs.mkdir(parents=True, exist_ok=True)
            (refs / "main").write_text("abc123", encoding="utf-8")

            with patch("abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"):
                with patch("abstractvision.playground_server.local_model_profile", return_value="apple-silicon"):
                    with patch("abstractvision.playground_server._local_runtime_available", side_effect=lambda backend: backend != "mflux"):
                        state = PlaygroundState(
                            PlaygroundServerConfig(
                                diffusers_cache_dir=str(cache),
                                diffusers_allow_download=False,
                                default_model_id="",
                            )
                        )
                        out = state.list_models()

        chosen = next(m for m in out["models"] if m["load_id"] == "mflux/flux2-klein-9b")
        self.assertTrue(chosen["cached"])
        self.assertFalse(chosen["loadable"])
        self.assertIn("mflux runtime missing", " ".join(chosen["cached_in"]))

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
                    "model": "mflux/flux2-klein-9b",
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
        self.assertEqual(state.active_snapshot()["model_id"], "mflux/flux2-klein-9b")

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
                    {"prompt": "hello", "model": "mflux/qwen-image", "steps": 1}
                )
                self.assertFalse(backend.preloaded)
                self.assertEqual(job["state"], "queued")
                self.assertIsNone(state.active_snapshot())
                targets[0]()
        snap = state.get_job(job["job_id"])
        self.assertTrue(backend.preloaded)
        self.assertEqual(snap["state"], "succeeded")
        self.assertEqual(state.active_snapshot()["model_id"], "mflux/qwen-image")

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
