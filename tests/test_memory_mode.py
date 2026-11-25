"""memory_mode: the shared per-request memory setting (types, VisionManager, backends, plugin, CLI)."""

import base64
import contextlib
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X9+QAAAABJRU5ErkJggg=="
)
NO_ENV = {"ABSTRACTVISION_MEMORY_MODE": ""}


def _recording_backend(memory_modes, *, uses_batch_marker=False):
    from abstractvision.backends import VisionBackend
    from abstractvision.types import GeneratedAsset, VisionBackendCapabilities

    class RecordingBackend(VisionBackend):
        def __init__(self) -> None:
            self.uses_batch_marker = uses_batch_marker
            self.requests = []
            self.modes = []

        def get_capabilities(self):
            if memory_modes is None:
                return None
            return VisionBackendCapabilities(memory_modes=memory_modes)

        def supported_memory_modes(self, task=None):
            return tuple(memory_modes) if memory_modes else ("default",)

        def _record(self, request, media_type="image"):
            # As every backend: the mode is resolved and checked before any work.
            _request, mode, _batch = self.resolve_memory_request(request)
            self.requests.append(request)
            self.modes.append(mode)
            return GeneratedAsset(media_type=media_type, data=b"x", mime_type="image/png", metadata={})

        def generate_image(self, request):
            return self._record(request)

        def edit_image(self, request):
            return self._record(request)

        def upscale_image(self, request):
            return self._record(request)

        def generate_angles(self, request):  # pragma: no cover
            raise NotImplementedError

        def generate_video(self, request):
            return self._record(request, media_type="video")

        def image_to_video(self, request):
            return self._record(request, media_type="video")

    return RecordingBackend()


class _FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.headers = {}

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestMemoryModeTypes(unittest.TestCase):
    def test_normalize_accepts_the_three_modes_and_defaults_empty_values(self):
        from abstractvision.types import MEMORY_MODES, normalize_memory_mode

        self.assertEqual(MEMORY_MODES, ("default", "low", "lowest"))
        self.assertEqual(normalize_memory_mode(None), "default")
        self.assertEqual(normalize_memory_mode(""), "default")
        self.assertEqual(normalize_memory_mode(" LOW "), "low")
        self.assertEqual(normalize_memory_mode("lowest"), "lowest")

    def test_normalize_rejects_unknown_modes(self):
        from abstractvision.types import normalize_memory_mode

        for value in ("low_ram", "minimal", "true"):
            with self.assertRaisesRegex(ValueError, "memory_mode must be one of default, low, lowest"):
                normalize_memory_mode(value)

    def test_the_field_wins_then_extra_then_the_environment(self):
        from abstractvision.types import resolve_memory_mode_with_source as resolve

        with patch.dict("os.environ", NO_ENV):
            self.assertEqual(resolve(None), ("default", False))
            self.assertEqual(resolve("low"), ("low", False))
            self.assertEqual(resolve(None, {"memory_mode": "lowest"}), ("lowest", False))
            with self.assertRaisesRegex(ValueError, "disagree"):
                resolve("low", {"memory_mode": "lowest"})
        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": "low"}):
            self.assertEqual(resolve(None), ("low", True))
            self.assertEqual(resolve(None, {"other": 1}), ("low", True))
            self.assertEqual(resolve("default"), ("default", False))
            self.assertEqual(resolve(None, {"memory_mode": "lowest"}), ("lowest", False))
        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": "tiny"}):
            with self.assertRaisesRegex(ValueError, "ABSTRACTVISION_MEMORY_MODE"):
                resolve(None)

    def test_every_generation_request_carries_memory_mode(self):
        from abstractvision import types as av_types

        for cls in (
            av_types.ImageGenerationRequest,
            av_types.ImageEditRequest,
            av_types.ImageUpscaleRequest,
            av_types.VideoGenerationRequest,
            av_types.ImageToVideoRequest,
        ):
            self.assertIn("memory_mode", cls.__dataclass_fields__, cls.__name__)
            self.assertIsNone(cls.__dataclass_fields__["memory_mode"].default)


class TestVisionManagerMemoryMode(unittest.TestCase):
    def setUp(self):
        self._env = patch.dict("os.environ", NO_ENV)
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_low_reaches_a_backend_that_lists_it(self):
        from abstractvision import VisionManager
        from abstractvision.types import MEMORY_MODES

        backend = _recording_backend(MEMORY_MODES)
        vm = VisionManager(backend=backend)
        vm.generate_image("p", memory_mode="low")
        vm.edit_image("p", image=b"img", memory_mode="lowest")
        vm.upscale_image(b"img", memory_mode="low")
        vm.generate_video("p", memory_mode="low")
        vm.image_to_video(b"img", memory_mode="LOW")

        self.assertEqual(backend.modes, ["low", "lowest", "low", "low", "low"])
        # VisionManager carries the value; the backend resolves and checks it.
        self.assertEqual(backend.requests[-1].memory_mode, "LOW")

    def test_extra_memory_mode_reaches_the_backend(self):
        # AbstractCore's HTTP routes forward backend parameters through `extra`.
        from abstractvision import VisionManager
        from abstractvision.types import MEMORY_MODES

        backend = _recording_backend(MEMORY_MODES)
        VisionManager(backend=backend).generate_video(
            "p", extra={"memory_mode": "low", "max_sequence_length": 256}
        )

        self.assertEqual(backend.modes, ["low"])
        request, _mode, _batch = backend.resolve_memory_request(backend.requests[0])
        self.assertEqual((request.memory_mode, request.extra), ("low", {"max_sequence_length": 256}))

    def test_disagreeing_keyword_and_extra_are_refused(self):
        from abstractvision import VisionManager
        from abstractvision.types import MEMORY_MODES

        vm = VisionManager(backend=_recording_backend(MEMORY_MODES))
        with self.assertRaisesRegex(ValueError, "disagree"):
            vm.generate_image("p", memory_mode="low", extra={"memory_mode": "lowest"})

    def test_mode_a_backend_does_not_list_is_refused_before_any_work(self):
        from abstractvision import VisionManager
        from abstractvision.errors import CapabilityNotSupportedError

        backend = _recording_backend(("default", "low"))
        vm = VisionManager(backend=backend)
        with self.assertRaisesRegex(CapabilityNotSupportedError, r"memory_mode='lowest'.*supported: default, low"):
            vm.generate_image("p", memory_mode="lowest")
        with self.assertRaises(CapabilityNotSupportedError):
            vm.generate_video("p", extra={"memory_mode": "lowest"})
        self.assertEqual(backend.requests, [])

        vm.generate_image("p", memory_mode="default")
        self.assertEqual(backend.requests[0].memory_mode, "default")

    def test_backend_without_capabilities_accepts_only_default(self):
        from abstractvision import VisionManager
        from abstractvision.errors import CapabilityNotSupportedError

        backend = _recording_backend(None)
        vm = VisionManager(backend=backend)
        with self.assertRaises(CapabilityNotSupportedError):
            vm.upscale_image(b"img", memory_mode="low")
        vm.upscale_image(b"img")
        self.assertIsNone(backend.requests[0].memory_mode)

    def test_environment_default_is_left_to_the_backend_and_never_refused_early(self):
        from abstractvision import VisionManager
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import MEMORY_MODES

        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": "low"}):
            backend = _recording_backend(MEMORY_MODES)
            VisionManager(backend=backend).generate_image("p")
            # Unset here: the backend applies (and degrades) the process default itself.
            self.assertIsNone(backend.requests[0].memory_mode)
            self.assertEqual(backend.resolve_memory_request(backend.requests[0])[1], "low")
            narrow = _recording_backend(("default",))
            VisionManager(backend=narrow).generate_image("p")
            self.assertEqual(narrow.resolve_memory_request(narrow.requests[0])[1], "default")
            # A value the call names still refuses.
            with self.assertRaises(CapabilityNotSupportedError):
                VisionManager(backend=narrow).generate_image("p", memory_mode="low")

    def test_unknown_mode_is_a_value_error(self):
        from abstractvision import VisionManager
        from abstractvision.types import MEMORY_MODES

        vm = VisionManager(backend=_recording_backend(MEMORY_MODES))
        with self.assertRaises(ValueError):
            vm.generate_image("p", memory_mode="tiny")

    def test_batch_items_carry_one_batch_id_and_mark_the_last(self):
        from abstractvision import VisionManager
        from abstractvision.types import BATCH_EXTRA_KEY, MEMORY_MODES

        backend = _recording_backend(MEMORY_MODES, uses_batch_marker=True)
        VisionManager(backend=backend).generate_image_batch("p", count=3, seed=1, memory_mode="low")

        markers = [request.extra[BATCH_EXTRA_KEY] for request in backend.requests]
        self.assertEqual(len({marker["id"] for marker in markers}), 1)
        self.assertEqual([marker["last"] for marker in markers], [False, False, True])
        self.assertEqual([request.seed for request in backend.requests], [1, 2, 3])

    def test_batch_marker_reaches_only_backends_that_declare_they_use_it(self):
        from abstractvision import VisionManager
        from abstractvision.backends import VisionBackend
        from abstractvision.backends.mflux import MFluxVisionBackend
        from abstractvision.types import BATCH_EXTRA_KEY, GeneratedAsset

        class ThirdPartyBackend(VisionBackend):
            # Written against the VisionBackend interface only: no memory helper, no marker use.
            def __init__(self):
                self.extras = []

            def generate_image(self, request):
                self.extras.append(dict(request.extra or {}))
                return GeneratedAsset(media_type="image", data=b"x", mime_type="image/png", metadata={})

            def edit_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):  # pragma: no cover
                raise NotImplementedError

            def image_to_video(self, request):  # pragma: no cover
                raise NotImplementedError

        backend = ThirdPartyBackend()
        VisionManager(backend=backend).generate_image_batch("p", count=2, extra={"k": 1})
        self.assertEqual(backend.extras, [{"k": 1}, {"k": 1}])
        self.assertFalse(VisionBackend.uses_batch_marker)
        self.assertTrue(MFluxVisionBackend.uses_batch_marker)
        self.assertNotIn(BATCH_EXTRA_KEY, backend.extras[0])


class TestBackendsCheckMemoryModeThemselves(unittest.TestCase):
    """AbstractCore's HTTP server calls backends directly, so each backend enforces the contract."""

    def setUp(self):
        from abstractvision.backends import base_backend

        self._env = patch.dict("os.environ", NO_ENV)
        self._env.start()
        self.addCleanup(self._env.stop)
        base_backend._DEGRADED_PROCESS_MEMORY_MODES_LOGGED.clear()

    def test_the_process_default_fallback_is_logged_once_per_process(self):
        from abstractvision.types import ImageGenerationRequest

        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": "lowest"}):
            with self.assertLogs("abstractvision.backends.base_backend", level="WARNING") as logs:
                for _ in range(3):
                    # A new backend per request, as the AbstractCore plugin and the CLI build them.
                    _recording_backend(None).resolve_memory_request(ImageGenerationRequest(prompt="p"))
        self.assertEqual(len(logs.output), 1)

    def test_base_helper_strips_the_keys_and_refuses_unlisted_modes(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import BATCH_EXTRA_KEY, ImageGenerationRequest

        backend = _recording_backend(None)
        request, mode, batch = backend.resolve_memory_request(
            ImageGenerationRequest(
                prompt="p", extra={"memory_mode": "default", BATCH_EXTRA_KEY: {"id": "b", "last": True}, "k": 1}
            )
        )
        self.assertEqual((mode, batch, request.memory_mode, request.extra), ("default", {"id": "b", "last": True}, "default", {"k": 1}))
        with self.assertRaisesRegex(CapabilityNotSupportedError, "cannot honour memory_mode='low'"):
            backend.resolve_memory_request(ImageGenerationRequest(prompt="p", extra={"memory_mode": "low"}))

    def test_process_default_degrades_to_what_each_backend_can_honour(self):
        from abstractvision.backends import huggingface_diffusers as hd
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageGenerationRequest, highest_memory_mode_at_or_below

        self.assertEqual(highest_memory_mode_at_or_below("lowest", ("default", "low")), "low")
        self.assertEqual(highest_memory_mode_at_or_below("lowest", ("default",)), "default")
        self.assertEqual(highest_memory_mode_at_or_below("low", ("default", "low", "lowest")), "low")

        narrow = _recording_backend(None)  # base backend: default only
        mlx_like = _recording_backend(("default", "low"))
        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": "lowest"}):
            with self.assertLogs("abstractvision.backends.base_backend", level="WARNING") as logs:
                for _ in range(3):
                    _req, mode, _batch = narrow.resolve_memory_request(ImageGenerationRequest(prompt="p"))
                    self.assertEqual(mode, "default")
            self.assertEqual(len(logs.output), 1)  # once per process
            self.assertEqual(mlx_like.resolve_memory_request(ImageGenerationRequest(prompt="p"))[1], "low")
            self.assertEqual(mlx_like.process_memory_mode(), "low")
            # A value on the request keeps refusing.
            with self.assertRaises(CapabilityNotSupportedError):
                mlx_like.resolve_memory_request(ImageGenerationRequest(prompt="p", memory_mode="lowest"))
            with self.assertRaises(CapabilityNotSupportedError):
                narrow.resolve_memory_request(ImageGenerationRequest(prompt="p", extra={"memory_mode": "low"}))

        backend = hd.HuggingFaceDiffusersVisionBackend(config=hd.HuggingFaceDiffusersBackendConfig(model_id="x", device="cpu"))
        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": "low"}), patch.object(
            hd, "_lazy_import_torch", return_value=types.SimpleNamespace()
        ):
            self.assertEqual(backend.resolve_memory_request(ImageGenerationRequest(prompt="p"))[1], "default")
            self.assertEqual(backend._warmup_generation_request().memory_mode, "default")

    def test_openai_compatible_runs_a_process_low_default_as_default(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.types import ImageGenerationRequest

        sent = []

        def fake_urlopen(req, timeout=0):
            sent.append(json.loads(req.data.decode("utf-8")))
            body = {"data": [{"b64_json": base64.b64encode(PNG_1X1).decode("ascii")}]}
            return _FakeHTTPResponse(json.dumps(body).encode("utf-8"))

        backend = OpenAICompatibleVisionBackend(
            config=OpenAICompatibleBackendConfig(base_url="http://localhost:1234/v1", api_key="k", model_id="m")
        )
        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": "low"}), patch(
            "abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen
        ):
            backend.generate_image(ImageGenerationRequest(prompt="p"))
        self.assertEqual(len(sent), 1)
        self.assertNotIn("memory_mode", sent[0])

    def test_openai_compatible_refuses_and_never_sends_the_key_upstream(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import BATCH_EXTRA_KEY, ImageGenerationRequest, VideoGenerationRequest

        sent = []

        def fake_urlopen(req, timeout=0):
            sent.append(json.loads(req.data.decode("utf-8")))
            body = {"data": [{"b64_json": base64.b64encode(PNG_1X1).decode("ascii")}]}
            return _FakeHTTPResponse(json.dumps(body).encode("utf-8"))

        backend = OpenAICompatibleVisionBackend(
            config=OpenAICompatibleBackendConfig(
                base_url="http://localhost:1234/v1", api_key="k", model_id="m", text_to_video_path="/videos"
            )
        )
        self.assertEqual(tuple(backend.get_capabilities().memory_modes), ("default",))
        with patch("abstractvision.backends.openai_compatible.urlopen", new=fake_urlopen):
            for request in (
                ImageGenerationRequest(prompt="p", extra={"memory_mode": "low"}),
                VideoGenerationRequest(prompt="p", memory_mode="lowest"),
            ):
                with self.assertRaises(CapabilityNotSupportedError):
                    getattr(backend, "generate_image" if isinstance(request, ImageGenerationRequest) else "generate_video")(request)
            self.assertEqual(sent, [])
            backend.generate_image(
                ImageGenerationRequest(
                    prompt="p", extra={"memory_mode": "default", BATCH_EXTRA_KEY: {"id": "b", "last": True}}
                )
            )
        self.assertNotIn("memory_mode", sent[0])
        self.assertNotIn(BATCH_EXTRA_KEY, sent[0])

    def test_diffusers_lists_offload_modes_on_cuda_only_and_refuses_elsewhere(self):
        from abstractvision.backends import huggingface_diffusers as hd
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageGenerationRequest

        fake_torch = types.SimpleNamespace()
        for device, expected in (("cuda", ("default", "low", "lowest")), ("cuda:1", ("default", "low", "lowest")), ("cpu", ("default",)), ("mps", ("default",))):
            backend = hd.HuggingFaceDiffusersVisionBackend(
                config=hd.HuggingFaceDiffusersBackendConfig(model_id="runwayml/stable-diffusion-v1-5", device=device)
            )
            with patch.object(hd, "_lazy_import_torch", return_value=fake_torch):
                self.assertEqual(tuple(backend.supported_memory_modes()), expected, device)
                self.assertEqual(tuple(backend.get_capabilities().memory_modes), expected, device)
                if device == "cpu":
                    with patch.object(backend, "_get_or_load_pipeline", side_effect=AssertionError("loaded")):
                        with self.assertRaisesRegex(CapabilityNotSupportedError, "memory_mode='low'"):
                            backend.generate_image(ImageGenerationRequest(prompt="p", extra={"memory_mode": "low"}))
        backend = hd.HuggingFaceDiffusersVisionBackend(config=hd.HuggingFaceDiffusersBackendConfig(model_id="x"))
        with patch.object(hd, "_lazy_import_torch", side_effect=ImportError("no torch")):
            self.assertEqual(tuple(backend.supported_memory_modes()), ("default",))

    def test_warmups_follow_the_process_default_only_where_it_can_be_honoured(self):
        from abstractvision.backends import huggingface_diffusers as hd
        from abstractvision.errors import OptionalDependencyMissingError

        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": "low"}), patch.object(
            hd, "_lazy_import_torch", return_value=types.SimpleNamespace()
        ):
            for device, expected in (("cuda", "low"), ("cpu", "default")):
                backend = hd.HuggingFaceDiffusersVisionBackend(
                    config=hd.HuggingFaceDiffusersBackendConfig(model_id="x", device=device)
                )
                self.assertEqual(backend._warmup_generation_request().memory_mode, expected, device)
                self.assertEqual(backend._warmup_edit_request().memory_mode, expected, device)

        inits = []

        class _StableDiffusion:
            def __init__(self, **kwargs):
                inits.append(kwargs)

        backend = self._sdcpp_backend()
        with patch.dict(sys.modules, {"stable_diffusion_cpp": types.SimpleNamespace(StableDiffusion=_StableDiffusion)}), patch(
            "abstractvision.backends.stable_diffusion_cpp._require_sd_cli",
            side_effect=OptionalDependencyMissingError("no sd-cli"),
        ), patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": "low"}):
            backend.preload()
        self.assertIs(inits[0].get("offload_params_to_cpu"), True)

    def test_diffusers_reloads_a_pipeline_with_less_offload_than_the_request_needs(self):
        from abstractvision.backends import huggingface_diffusers as hd

        class _Pipe:
            pass

        backend = hd.HuggingFaceDiffusersVisionBackend(config=hd.HuggingFaceDiffusersBackendConfig(model_id="x", device="cuda"))
        offloaded = _Pipe()
        offloaded._abstractvision_cpu_offload = True
        offloaded._abstractvision_cpu_offload_mode = "model"
        backend._pipelines["t2i"] = offloaded

        # The configured placement (or a higher one) serves default and low requests as loaded.
        for floor in ("", "model"):
            backend._offload_floor = floor
            self.assertIs(backend._get_or_load_pipeline("t2i"), offloaded)
        # lowest needs sequential offload: the pipeline is dropped and loaded again.
        backend._offload_floor = "sequential"
        with patch.object(hd, "_lazy_import_diffusers", side_effect=RuntimeError("reload")), patch.object(
            hd, "_release_pipeline_adapters"
        ) as released:
            with self.assertRaisesRegex(RuntimeError, "reload"):
                backend._get_or_load_pipeline("t2i")
        self.assertNotIn("t2i", backend._pipelines)
        released.assert_called_once_with(offloaded)
        self.assertEqual(hd._MEMORY_MODE_OFFLOAD_FLOOR, {"default": "", "low": "model", "lowest": "sequential"})

    def test_diffusers_load_raises_the_configured_offload_to_the_request_floor(self):
        import inspect

        from abstractvision.backends import huggingface_diffusers as hd

        load = inspect.getsource(hd.HuggingFaceDiffusersVisionBackend._get_or_load_pipeline)
        decision = load.split("offload, offload_reason = _cuda_offload_decision(", 1)[1]
        # The floor is applied after the configured decision and before any placement.
        self.assertLess(decision.index("_OFFLOAD_RANK.get(offload, 0) < _OFFLOAD_RANK[floor]"), decision.index("enable_sequential_cpu_offload"))
        # A requested offload that fails is refused instead of falling back to less offload.
        self.assertIn('if floor == "sequential":\n                    raise CapabilityNotSupportedError', decision)

    def _sdcpp_backend(self, **cfg):
        from abstractvision.backends.stable_diffusion_cpp import (
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
        )

        return StableDiffusionCppVisionBackend(
            config=StableDiffusionCppBackendConfig(sd_cli_path="sd-cli", diffusion_model="model.gguf", **cfg)
        )

    def test_sdcpp_cli_maps_low_to_offload_to_cpu_and_never_forwards_the_key(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import BATCH_EXTRA_KEY, ImageEditRequest, ImageGenerationRequest

        commands = []

        def fake_run(cmd, check, stdout, stderr, cwd, timeout=None):
            commands.append(list(cmd))
            Path(cmd[cmd.index("--output") + 1]).write_bytes(PNG_1X1)

        backend = self._sdcpp_backend()
        self.assertEqual(tuple(backend.get_capabilities().memory_modes), ("default", "low"))
        with patch("abstractvision.backends.stable_diffusion_cpp._require_sd_cli", return_value="/usr/bin/sd-cli"), patch(
            "abstractvision.backends.stable_diffusion_cpp.subprocess.run", side_effect=fake_run
        ):
            low = backend.generate_image(
                ImageGenerationRequest(prompt="p", extra={"memory_mode": "low", BATCH_EXTRA_KEY: {"id": "b", "last": True}})
            )
            backend.edit_image(ImageEditRequest(prompt="p", image=PNG_1X1, memory_mode="low"))
            backend.generate_image(ImageGenerationRequest(prompt="p"))
            with self.assertRaises(CapabilityNotSupportedError):
                backend.generate_image(ImageGenerationRequest(prompt="p", memory_mode="lowest"))

        self.assertEqual(len(commands), 3)
        for cmd in commands:
            self.assertNotIn("--memory-mode", cmd)
            self.assertFalse(any("abstractvision-batch" in part or "_abstractvision_batch" in part for part in cmd))
        self.assertEqual([cmd.count("--offload-to-cpu") for cmd in commands], [1, 1, 0])
        self.assertEqual(low.metadata["memory_mode"], "low")

    def test_sdcpp_cli_keeps_an_operator_offload_flag_once(self):
        from abstractvision.types import ImageGenerationRequest

        commands = []

        def fake_run(cmd, check, stdout, stderr, cwd, timeout=None):
            commands.append(list(cmd))
            Path(cmd[cmd.index("--output") + 1]).write_bytes(PNG_1X1)

        backend = self._sdcpp_backend(extra_args=("--offload-to-cpu",))
        with patch("abstractvision.backends.stable_diffusion_cpp._require_sd_cli", return_value="/usr/bin/sd-cli"), patch(
            "abstractvision.backends.stable_diffusion_cpp.subprocess.run", side_effect=fake_run
        ):
            backend.generate_image(ImageGenerationRequest(prompt="p", memory_mode="low"))
            backend.generate_image(ImageGenerationRequest(prompt="p"))
        self.assertEqual([cmd.count("--offload-to-cpu") for cmd in commands], [1, 1])

    def test_sdcpp_python_binding_reloads_with_offload_params_to_cpu_for_low(self):
        from abstractvision.errors import OptionalDependencyMissingError
        from abstractvision.types import ImageGenerationRequest

        inits = []

        class _Image:
            def save(self, fp, format=None):  # noqa: A002
                fp.write(PNG_1X1)

        class _StableDiffusion:
            def __init__(self, **kwargs):
                inits.append(kwargs)

            def generate_image(self, prompt, negative_prompt="", width=512, height=512, cfg_scale=7.0, sample_steps=20, seed=42):
                return [_Image()]

        fake_mod = types.SimpleNamespace(__version__="0.0.0", StableDiffusion=_StableDiffusion)
        backend = self._sdcpp_backend()
        with patch.dict(sys.modules, {"stable_diffusion_cpp": fake_mod}), patch(
            "abstractvision.backends.stable_diffusion_cpp._require_sd_cli",
            side_effect=OptionalDependencyMissingError("no sd-cli"),
        ):
            backend.generate_image(ImageGenerationRequest(prompt="p"))
            low = backend.generate_image(ImageGenerationRequest(prompt="p", extra={"memory_mode": "low"}))
            # A loaded offloaded model also serves default requests: the request only raises the level.
            backend.generate_image(ImageGenerationRequest(prompt="p"))

        self.assertEqual(len(inits), 2)
        self.assertNotIn("offload_params_to_cpu", inits[0])
        self.assertIs(inits[1].get("offload_params_to_cpu"), True)
        self.assertNotIn("memory_mode", inits[1])
        self.assertEqual(low.metadata["memory_mode"], "low")


class TestAbstractCorePluginMemoryMode(unittest.TestCase):
    def setUp(self):
        self._env = patch.dict("os.environ", NO_ENV)
        self._env.start()
        self.addCleanup(self._env.stop)

    def _mflux_capability(self, fake_backend_cls):
        import abstractvision.backends.mflux as mflux_backend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {"vision_backend": "mflux", "vision_mflux_model": "AbstractFramework/flux.2-klein-4b-8bit"}

        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(mflux_backend, "MFluxVisionBackend", fake_backend_cls))
        stack.enter_context(patch.dict("os.environ", {}, clear=True))
        self.addCleanup(stack.close)
        return _AbstractVisionCapability(_DummyOwner())

    def test_t2v_memory_mode_reaches_the_mlx_gen_request_field(self):
        from abstractvision.types import MEMORY_MODES, GeneratedAsset, VisionBackendCapabilities

        seen = {}

        class _FakeBackend:
            def __init__(self, *, config):
                pass

            def get_capabilities(self):
                return VisionBackendCapabilities(memory_modes=MEMORY_MODES)

            def generate_video(self, request):
                seen["request"] = request
                return GeneratedAsset(media_type="video", data=b"mp4", mime_type="video/mp4", metadata={})

            def generate_image(self, request):
                seen["request"] = request
                return GeneratedAsset(media_type="image", data=b"png", mime_type="image/png", metadata={})

        cap = self._mflux_capability(_FakeBackend)
        cap.t2v("fox", memory_mode="low", num_frames=5)
        self.assertEqual(seen["request"].memory_mode, "low")
        self.assertNotIn("memory_mode", seen["request"].extra or {})
        # The AbstractCore HTTP route nests backend parameters under `extra`: passed through, the
        # backend resolves it (`resolve_memory_request`).
        cap.t2i("fox", extra={"memory_mode": "lowest"})
        self.assertEqual(seen["request"].extra["memory_mode"], "lowest")

    def test_record_is_dropped_when_the_run_released_the_model(self):
        from abstractvision.types import MEMORY_MODES, GeneratedAsset, VisionBackendCapabilities

        class _FakeBackend:
            loaded = False

            def __init__(self, *, config):
                pass

            def preload(self):
                _FakeBackend.loaded = True

            def has_loaded_model(self):
                return _FakeBackend.loaded

            def get_capabilities(self):
                return VisionBackendCapabilities(memory_modes=MEMORY_MODES)

            def generate_image(self, request):
                _FakeBackend.loaded = request.memory_mode != "low"
                return GeneratedAsset(media_type="image", data=b"png", mime_type="image/png", metadata={})

        cap = self._mflux_capability(_FakeBackend)
        cap.load_resident_model({"provider": "mlx-gen", "model": "AbstractFramework/flux.2-klein-4b-8bit"})
        self.assertEqual(len(cap.list_resident_models()), 1)
        cap.t2i("fox", provider="mlx-gen", model="AbstractFramework/flux.2-klein-4b-8bit", memory_mode="low")
        self.assertEqual(cap.list_loaded_models(), [])
        cap.t2i("fox", provider="mlx-gen", model="AbstractFramework/flux.2-klein-4b-8bit")
        records = cap.list_loaded_models()
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["resident"])

    def test_record_is_dropped_when_a_failed_run_released_the_model(self):
        from abstractvision.types import MEMORY_MODES, VisionBackendCapabilities

        class _FakeBackend:
            loaded = False

            def __init__(self, *, config):
                pass

            def preload(self):
                _FakeBackend.loaded = True

            def has_loaded_model(self):
                return _FakeBackend.loaded

            def get_capabilities(self):
                return VisionBackendCapabilities(memory_modes=MEMORY_MODES)

            def generate_image(self, request):
                _FakeBackend.loaded = False  # a failed low-RAM run drops the instance
                raise RuntimeError("denoise failed")

        cap = self._mflux_capability(_FakeBackend)
        cap.load_resident_model({"provider": "mlx-gen", "model": "AbstractFramework/flux.2-klein-4b-8bit"})
        with self.assertRaisesRegex(RuntimeError, "denoise failed"):
            cap.t2i("fox", provider="mlx-gen", model="AbstractFramework/flux.2-klein-4b-8bit", memory_mode="low")
        self.assertEqual(cap.list_loaded_models(), [])

    def test_plugin_refuses_memory_mode_on_a_provider_that_cannot_honour_it(self):
        from abstractvision.backends import huggingface_diffusers as hd
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _DummyOwner:
            config = {"vision_backend": "diffusers", "vision_model_id": "runwayml/stable-diffusion-v1-5", "vision_device": "cpu"}

        with patch.object(hd, "_lazy_import_torch", return_value=types.SimpleNamespace()), patch.object(
            hd.HuggingFaceDiffusersVisionBackend, "_get_or_load_pipeline", side_effect=AssertionError("loaded")
        ), patch.dict("os.environ", {}, clear=True):
            cap = _AbstractVisionCapability(_DummyOwner())
            with self.assertRaisesRegex(CapabilityNotSupportedError, "memory_mode='low'"):
                cap.t2i("fox", memory_mode="low")

    def test_available_providers_asks_each_backend_without_loading_or_network(self):
        from abstractvision.backends import huggingface_diffusers as hd
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _Owner:
            def __init__(self, config):
                self.config = config

        transport = []

        def fake_urlopen(req, timeout=None):
            transport.append(req.full_url)
            if req.full_url.endswith("/vision/models"):
                entry = {"id": "stand-in/h3", "memory_modes": ["default", "low"]}
                return _FakeHTTPResponse(json.dumps({"models": [entry], "active": entry}).encode())
            return _FakeHTTPResponse(json.dumps({"data": [{"id": "stand-in/h3"}]}).encode())

        with patch.dict("os.environ", {"ABSTRACTVISION_ASSUME_OFFLINE": "1"}, clear=True), patch.object(
            hd, "_lazy_import_torch", return_value=types.SimpleNamespace()
        ), patch.object(hd.HuggingFaceDiffusersVisionBackend, "_get_or_load_pipeline", side_effect=AssertionError("loaded")), patch(
            "abstractvision.backends.openai_compatible.urlopen", side_effect=fake_urlopen
        ):
            # Diffusers: what its configured device can run, whatever accelerator the host has.
            for device, expected in (("cpu", ["default"]), ("cuda", ["default", "low", "lowest"])):
                details = _AbstractVisionCapability(_Owner({"vision_device": device})).available_providers()["details"]
                self.assertEqual(details["huggingface"]["memory_modes"], expected, device)
                self.assertEqual(details["mlx-gen"]["memory_modes"], ["default", "low"])
                self.assertEqual(details["mflux"]["memory_modes"], ["default", "low"])
                self.assertEqual(details["openai"]["memory_modes"], ["default"])
            # OpenAI-compatible: the remote's last discovery answer, else "default"; never a request.
            cap = _AbstractVisionCapability(
                _Owner({"vision_base_url": "http://127.0.0.1:9/v1", "vision_model_id": "stand-in/h3"})
            )
            self.assertEqual(cap.available_providers()["details"]["openai-compatible"]["memory_modes"], ["default"])
            self.assertEqual(transport, [])
            cap.list_provider_models()
            sent = len(transport)
            self.assertEqual(cap.available_providers()["details"]["openai-compatible"]["memory_modes"], ["default", "low"])
            self.assertEqual(len(transport), sent)


# The model-facing tool parameters, as on main before memory modes: memory_mode is not among them.
_TOOL_PARAMETERS_ON_MAIN = {
    "vision_image_to_image": ["prompt", "image_artifact", "image_b64", "mask_artifact", "mask_b64", "negative_prompt", "steps", "guidance_scale", "seed", "reference_images", "lora_adapters"],
    "vision_image_to_image_batch": ["prompt", "image_artifact", "image_b64", "mask_artifact", "mask_b64", "negative_prompt", "steps", "guidance_scale", "seed", "count", "seeds", "reference_images", "lora_adapters"],
    "vision_image_to_video": ["image_artifact", "image_b64", "prompt", "negative_prompt", "width", "height", "fps", "num_frames", "steps", "guidance_scale", "guidance_2", "flow_shift", "seed", "lora_adapters"],
    "vision_image_to_video_batch": ["image_artifact", "image_b64", "prompt", "negative_prompt", "width", "height", "fps", "num_frames", "steps", "guidance_scale", "guidance_2", "flow_shift", "seed", "count", "seeds", "lora_adapters"],
    "vision_image_upscale": ["image_artifact", "image_b64", "scale", "resolution", "softness", "seed", "quantize", "vae_tiling"],
    "vision_list_adapters": ["task"],
    "vision_multi_view_image": ["prompt", "reference_image_artifact", "reference_image_b64", "angles", "negative_prompt", "steps", "guidance_scale", "seed"],
    "vision_text_to_image": ["prompt", "negative_prompt", "width", "height", "steps", "guidance_scale", "seed", "lora_adapters"],
    "vision_text_to_image_batch": ["prompt", "negative_prompt", "width", "height", "steps", "guidance_scale", "seed", "count", "seeds", "lora_adapters"],
    "vision_text_to_video": ["prompt", "negative_prompt", "width", "height", "fps", "num_frames", "steps", "guidance_scale", "guidance_2", "flow_shift", "seed", "lora_adapters"],
    "vision_text_to_video_batch": ["prompt", "negative_prompt", "width", "height", "fps", "num_frames", "steps", "guidance_scale", "guidance_2", "flow_shift", "seed", "count", "seeds", "lora_adapters"],
}


class TestVisionToolsMemoryMode(unittest.TestCase):
    def _tools(self, backend, **kwargs):
        import inspect
        import tempfile

        from abstractvision import VisionManager
        from abstractvision.artifacts import LocalAssetStore
        from abstractvision.integrations.abstractcore import make_vision_tools

        fake_abstractcore = types.ModuleType("abstractcore")
        fake_abstractcore.tool = lambda **_meta: (lambda fn: fn)
        store_dir = tempfile.mkdtemp()
        with patch.dict(sys.modules, {"abstractcore": fake_abstractcore}):
            vm = VisionManager(backend=backend, store=LocalAssetStore(store_dir))
            tools = make_vision_tools(vision_manager=vm, model_id="Qwen/Qwen-Image-2512", **kwargs)
        return {tool.__name__: tool for tool in tools}, inspect

    def test_tool_schemas_are_unchanged_and_carry_no_memory_mode(self):
        tools, inspect = self._tools(None)
        self.assertEqual(
            {name: list(inspect.signature(tool).parameters) for name, tool in tools.items()},
            _TOOL_PARAMETERS_ON_MAIN,
        )

    def test_factory_memory_mode_applies_to_every_tool_call(self):
        from abstractvision.types import MEMORY_MODES

        backend = _recording_backend(MEMORY_MODES)
        with patch.dict("os.environ", NO_ENV):
            tools, _inspect = self._tools(backend, memory_mode="low")
            tools["vision_text_to_image"]("fox")
            tools["vision_text_to_image_batch"]("fox", count=2)
        self.assertEqual([request.memory_mode for request in backend.requests], ["low", "low", "low"])

    def test_factory_refuses_an_unknown_mode_and_the_backend_one_it_does_not_list(self):
        from abstractvision.errors import CapabilityNotSupportedError

        with self.assertRaises(ValueError):
            self._tools(None, memory_mode="tiny")
        backend = _recording_backend(("default",))
        with patch.dict("os.environ", NO_ENV):
            tools, _inspect = self._tools(backend, memory_mode="low")
            with self.assertRaisesRegex(CapabilityNotSupportedError, "memory_mode='low'"):
                tools["vision_text_to_image"]("fox")
        self.assertEqual(backend.requests, [])


class TestCliMemoryMode(unittest.TestCase):
    def test_generation_commands_accept_memory_mode(self):
        from abstractvision.cli import build_parser

        parser = build_parser()
        for argv in (
            ["t2i", "p", "--memory-mode", "low"],
            ["i2i", "p", "--image", "in.png", "--memory-mode", "lowest"],
            ["upscale", "--image", "in.png", "--memory-mode", "low"],
            ["t2v", "p", "--memory-mode", "low"],
            ["i2v", "--image", "in.png", "--memory-mode", "default"],
        ):
            args = parser.parse_args(argv)
            self.assertEqual(args.memory_mode, argv[-1], argv)
        self.assertIsNone(parser.parse_args(["t2i", "p"]).memory_mode)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(["t2i", "p", "--memory-mode", "tiny"])

    def test_t2v_passes_memory_mode_to_the_manager(self):
        from abstractvision import cli

        seen = {}

        class _FakeManager:
            store = None

            def generate_video(self, prompt, **kwargs):
                seen.update(kwargs)
                return {"$artifact": "a"}

        args = cli.build_parser().parse_args(
            ["t2v", "fox", "--memory-mode", "low", "--no-progress", "--provider", "mlx-gen"]
        )
        with patch.object(cli, "_build_manager_from_args", return_value=_FakeManager()), patch.object(
            cli, "_print_json"
        ), patch.object(cli, "_artifact_paths_from_output", return_value=[]):
            self.assertEqual(cli._cmd_t2v(args), 0)
        self.assertEqual(seen["memory_mode"], "low")


if __name__ == "__main__":
    unittest.main()
