"""Typed request options (one definition, one merge with `extra`), loaded-model listings derived
from the backends, the Diffusers offload floor passed per request, refusals that name their
field, and the OpenAI-compatible request bodies."""

import dataclasses
import io
import json
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

NO_ENV = {"ABSTRACTVISION_MEMORY_MODE": ""}
# `extra` keys named like options, for backends that do not list those options.
OPAQUE_EXTRA = {"quantize": "q4_k", "soundscape": "wind", "music": "piano", "generate_audio": "no", "audio_shift": "1.5"}

# Two distinct values per typed option, in the type a request carries.
OPTION_VALUES = {
    "memory_mode": ("low", "lowest"),
    "quantize": (8, 4),
    "soundscape": ("wind over snow", "rain on a roof"),
    "music": ("sparse piano", "slow drums"),
    "generate_audio": (False, True),
    "audio_shift": (1.5, 2.0),
}


def _backend(options=None, modes=("default", "low", "lowest")):
    from abstractvision.backends.base_backend import OPTIONAL_REQUEST_FIELDS, VisionBackend

    class _Backend(VisionBackend):
        def supported_memory_modes(self, task=None):
            return tuple(modes)

        def supported_request_options(self, task=None):
            return frozenset(OPTIONAL_REQUEST_FIELDS if options is None else options)

        def generate_image(self, request):  # pragma: no cover - not called
            raise NotImplementedError

        edit_image = generate_angles = generate_video = image_to_video = generate_image

    return _Backend()


class TestOneDefinitionOfTheOptions(unittest.TestCase):
    def test_capabilities_carry_no_engine_streaming_term(self):
        from abstractvision.types import VisionBackendCapabilities

        names = {f.name for f in dataclasses.fields(VisionBackendCapabilities)}
        self.assertIn("memory_modes", names)
        self.assertNotIn("supports_block_streaming", names)

    def test_the_option_fields_are_fields_of_the_requests_that_carry_them(self):
        from abstractvision import types as av_types
        from abstractvision.backends.base_backend import (
            OPTIONAL_REQUEST_FIELDS,
            REQUEST_OPTION_FIELDS,
            SOUNDTRACK_REQUEST_FIELDS,
        )

        self.assertEqual(REQUEST_OPTION_FIELDS, ("memory_mode", "quantize", *SOUNDTRACK_REQUEST_FIELDS))
        self.assertEqual(OPTIONAL_REQUEST_FIELDS, REQUEST_OPTION_FIELDS[1:])
        self.assertEqual(set(OPTION_VALUES), set(REQUEST_OPTION_FIELDS))
        for cls in (av_types.VideoGenerationRequest, av_types.ImageToVideoRequest):
            self.assertLessEqual(set(REQUEST_OPTION_FIELDS), {f.name for f in dataclasses.fields(cls)}, cls)
        for cls in (av_types.ImageGenerationRequest, av_types.ImageEditRequest, av_types.ImageUpscaleRequest):
            carried = {f.name for f in dataclasses.fields(cls)} & set(REQUEST_OPTION_FIELDS)
            self.assertEqual(carried, {"memory_mode", "quantize"}, cls)


class TestOptionMergedWithExtra(unittest.TestCase):
    def test_every_option_from_its_field_from_extra_both_equal_or_both_different(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        def resolve(backend, typed, extra):
            request = VideoGenerationRequest(prompt="fox", extra={**extra, "k": 1}, **typed)
            return backend.resolve_memory_request(request, task="text_to_video")

        for name, (value, other) in OPTION_VALUES.items():
            # memory_mode is an option of every backend; the others only where the backend lists them.
            for lists in (True, False):
                backend = _backend(options=None if lists else ())
                is_option = lists or name == "memory_mode"
                cases = {
                    "typed only": ({name: value}, {}),
                    "extra only": ({}, {name: value}),
                    "both equal": ({name: value}, {name: value}),
                    "both different": ({name: value}, {name: other}),
                }
                for case, (typed, extra) in cases.items():
                    with self.subTest(option=name, lists=lists, case=case), patch.dict("os.environ", NO_ENV):
                        if is_option and case == "both different":
                            with self.assertRaisesRegex(
                                ValueError, rf"^{name}={value!r} and extra\['{name}'\]={other!r} disagree; pass one\.$"
                            ):
                                resolve(backend, typed, extra)
                        elif is_option:
                            resolved, mode = resolve(backend, typed, extra)
                            self.assertEqual(getattr(resolved, name), value)
                            self.assertEqual(resolved.extra, {"k": 1})
                            self.assertEqual(mode, value if name == "memory_mode" else "default")
                        elif typed:
                            # The typed field is refused where the backend does not list it.
                            with self.assertRaises(CapabilityNotSupportedError) as ctx:
                                resolve(backend, typed, extra)
                            self.assertEqual(ctx.exception.field, name)
                        else:
                            # An `extra` key that is not this backend's option is not interpreted.
                            resolved, _mode = resolve(backend, typed, extra)
                            self.assertIsNone(getattr(resolved, name))
                            self.assertEqual(resolved.extra, {name: value, "k": 1})

    def test_memory_modes_compare_by_their_canonical_name(self):
        from abstractvision.types import ImageGenerationRequest

        with patch.dict("os.environ", NO_ENV):
            request = ImageGenerationRequest(prompt="fox", memory_mode="LOW", extra={"memory_mode": "low"})
            self.assertEqual(_backend().resolve_memory_request(request)[1], "low")

    def test_a_listed_option_the_request_type_does_not_carry_is_refused_from_extra(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageGenerationRequest

        with patch.dict("os.environ", NO_ENV), self.assertRaisesRegex(
            CapabilityNotSupportedError, "does not support soundscape"
        ):
            _backend().resolve_memory_request(
                ImageGenerationRequest(prompt="fox", extra={"soundscape": "wind"}), task="text_to_image"
            )

    def test_backends_that_do_not_list_the_options_get_extra_as_0_3_36_sent_it(self):
        # Captured from AbstractVision 0.3.36 (eea267d) for the same requests: the keys are opaque.
        import base64

        from abstractvision.backends import huggingface_diffusers as hd
        from abstractvision.backends.stable_diffusion_cpp import (
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
        )
        from abstractvision.types import ImageGenerationRequest
        from PIL import Image

        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X9+QAAAABJRU5ErkJggg=="
        )
        calls, commands = [], []

        class _Pipe:
            def to(self, *args, **kwargs):
                return self

            def __call__(self, **kwargs):
                calls.append(sorted(kwargs))
                return types.SimpleNamespace(images=[Image.new("RGB", (4, 4), color=(255, 0, 0))])

        def run(cmd, check, stdout, stderr, cwd, timeout=None):
            out = cmd.index("--output") + 1
            commands.append([*cmd[:out], "<out>", *cmd[out + 1 :]])
            Path(cmd[out]).write_bytes(png)

        t2i = MagicMock()
        t2i.from_pretrained.return_value = _Pipe()
        with patch.dict("os.environ", NO_ENV), patch.object(
            hd, "_lazy_import_diffusers", return_value=(MagicMock(), t2i, MagicMock(), MagicMock(), "0.0.0")
        ):
            hd.HuggingFaceDiffusersVisionBackend(
                config=hd.HuggingFaceDiffusersBackendConfig(model_id="some/model", device="cpu")
            ).generate_image(ImageGenerationRequest(prompt="fox", extra=dict(OPAQUE_EXTRA)))
        with patch.dict("os.environ", NO_ENV), patch(
            "abstractvision.backends.stable_diffusion_cpp.require_sd_cli", return_value="/usr/bin/sd-cli"
        ), patch("abstractvision.backends.stable_diffusion_cpp.subprocess.run", side_effect=run):
            StableDiffusionCppVisionBackend(
                config=StableDiffusionCppBackendConfig(sd_cli_path="sd-cli", diffusion_model="model.gguf")
            ).generate_image(ImageGenerationRequest(prompt="fox", extra=dict(OPAQUE_EXTRA)))

        self.assertEqual(calls, [["audio_shift", "generate_audio", "music", "prompt", "quantize", "soundscape"]])
        self.assertEqual(
            commands,
            [
                [
                    "/usr/bin/sd-cli", "--diffusion-model", "model.gguf", "--output", "<out>", "--prompt", "fox",
                    "--quantize", "q4_k", "--soundscape", "wind", "--music", "piano", "--generate-audio", "no",
                    "--audio-shift", "1.5",
                ]
            ],
        )


class _Backend:
    """A local backend stand-in that says whether it holds its model, as VisionBackend does."""

    def __init__(self):
        self.loaded = False
        self.unloaded = 0
        self.fail = None
        self.release_after_run = False

    def has_loaded_model(self):
        return self.loaded

    def preload(self):
        self.loaded = True
        if self.fail is not None:
            self.loaded = False  # a failed preload drops what it loaded
            raise self.fail

    def unload(self):
        self.unloaded += 1
        self.loaded = False

    def generate_image(self, request):
        from abstractvision.types import GeneratedAsset

        self.loaded = True
        try:
            if self.fail is not None:
                raise self.fail
        finally:
            if self.release_after_run:
                self.loaded = False
        return GeneratedAsset(media_type="image", data=b"png", mime_type="image/png", metadata={})


class TestLoadedModelsAreDerivedFromTheBackends(unittest.TestCase):
    MODEL = "AbstractFramework/flux.2-klein-4b-8bit"

    def setUp(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        self.backend = _Backend()
        self.cap = _AbstractVisionCapability(types.SimpleNamespace(config={}))
        patcher = patch.object(self.cap, "_make_mflux_backend", return_value=self.backend)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _load(self):
        return self.cap.load_resident_model({"provider": "mlx-gen", "model": self.MODEL, "task": "t2i"})

    def _t2i(self):
        return self.cap.t2i("fox", provider="mlx-gen", model=self.MODEL)

    def test_a_model_released_outside_a_plugin_call_is_not_listed(self):
        self._load()
        self.assertEqual(len(self.cap.list_resident_models()), 1)
        self.backend.unload()  # e.g. a caller holding the backend
        self.assertEqual(self.cap.list_loaded_models(), [])
        # The next load starts afresh: a new load, unpinned until pinned again.
        self.assertIs(self._load()["loaded_new"], True)
        self.backend.unload()
        self._t2i()
        (record,) = self.cap.list_loaded_models()
        self.assertEqual((record["resident"], record["source"], record["tasks"]), (False, "request", ["text_to_image"]))

    def test_a_failed_preload_leaves_no_record_and_keeps_a_load_it_did_not_drop(self):
        from abstractvision.errors import GenerationCancelledError

        self.backend.fail = RuntimeError("boom")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            self._load()
        self.assertEqual(self.cap.list_loaded_models(), [])
        self.backend.fail = None
        self._t2i()
        self.backend.preload = lambda: (_ for _ in ()).throw(GenerationCancelledError("stop"))
        with self.assertRaises(GenerationCancelledError):
            self._load()
        (record,) = self.cap.list_loaded_models()  # still held: still listed, still unpinned
        self.assertEqual((record["resident"], record["tasks"]), (False, ["text_to_image"]))

    def test_a_cancelled_run_is_listed_as_the_backend_left_it(self):
        from abstractvision.errors import GenerationCancelledError

        self._load()
        self.backend.fail = GenerationCancelledError("cancelled")
        with self.assertRaises(GenerationCancelledError):
            self._t2i()
        (record,) = self.cap.list_resident_models()  # the run kept the model: still pinned
        self.assertEqual(record["tasks"], ["text_to_image"])
        self.backend.release_after_run = True  # a cancelled low run drops its model
        with self.assertRaises(GenerationCancelledError):
            self._t2i()
        self.assertEqual(self.cap.list_loaded_models(), [])

    def test_a_run_that_releases_its_model_ends_the_load_and_its_pin(self):
        self._load()
        self.backend.release_after_run = True
        self._t2i()
        self.assertEqual(self.cap.list_loaded_models(), [])
        self.backend.release_after_run = False
        self._t2i()  # no listing in between: the load that ended still does not carry over
        (record,) = self.cap.list_loaded_models()
        self.assertEqual((record["resident"], record["state"], record["source"]), (False, "active", "request"))

    def test_unload_releases_the_backend_and_the_listing(self):
        self._load()
        out = self.cap.unload_resident_model({"load_id": f"mlx-gen/{self.MODEL}"})
        self.assertEqual((out["state"], out["loaded"], self.backend.unloaded), ("unloaded", False, 1))
        self.assertEqual(self.cap.list_loaded_models(), [])


class TestBackendsSayWhetherTheyHoldAModel(unittest.TestCase):
    def test_base_default_and_diffusers_answer(self):
        from abstractvision.backends import huggingface_diffusers as hd

        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )

        # A backend that loads nothing in this process (a remote one) holds no model here.
        self.assertIs(_backend().has_loaded_model(), False)
        remote = OpenAICompatibleVisionBackend(config=OpenAICompatibleBackendConfig(base_url="http://127.0.0.1:9/v1"))
        self.assertIs(remote.has_loaded_model(), False)
        backend = hd.HuggingFaceDiffusersVisionBackend(config=hd.HuggingFaceDiffusersBackendConfig(model_id="x"))
        self.assertIs(backend.has_loaded_model(), False)
        backend._pipelines["t2i"] = object()
        self.assertIs(backend.has_loaded_model(), True)
        with patch.object(hd, "_release_pipeline_adapters"), patch.object(hd, "_return_freed_host_memory"), patch.object(
            hd, "_lazy_import_torch", side_effect=ImportError("no torch")
        ):
            backend.unload()
        self.assertIs(backend.has_loaded_model(), False)


class TestStableDiffusionCppSaysWhatItHolds(unittest.TestCase):
    def _cap(self, backend):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        cap = _AbstractVisionCapability(types.SimpleNamespace(config={}))
        patcher = patch.object(cap, "_make_sdcpp_backend", return_value=backend)
        patcher.start()
        self.addCleanup(patcher.stop)
        return cap

    def _backend(self):
        from abstractvision.backends.stable_diffusion_cpp import (
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
        )

        return StableDiffusionCppVisionBackend(
            config=StableDiffusionCppBackendConfig(sd_cli_path="sd-cli", diffusion_model="model.gguf")
        )

    def test_sd_cli_holds_nothing_and_a_failed_first_request_is_not_listed(self):
        backend = self._backend()
        cap = self._cap(backend)
        with patch.dict("os.environ", NO_ENV), patch(
            "abstractvision.backends.stable_diffusion_cpp.require_sd_cli", return_value="/usr/bin/sd-cli"
        ), patch("abstractvision.backends.stable_diffusion_cpp.subprocess.run", side_effect=RuntimeError("sd-cli failed")):
            with self.assertRaisesRegex(RuntimeError, "sd-cli failed"):
                cap.t2i("fox", provider="sdcpp", model="model.gguf")
            self.assertEqual(cap.list_loaded_models(), [])
            cap.load_resident_model({"provider": "sdcpp", "model": "model.gguf"})
            self.assertIs(backend.has_loaded_model(), False)  # sd-cli loads per run
            self.assertEqual(cap.list_loaded_models(), [])

    def test_the_python_binding_holds_its_model_until_unload(self):
        backend = self._backend()
        self.assertIs(backend.has_loaded_model(), False)
        backend._py_model = object()
        self.assertIs(backend.has_loaded_model(), True)
        backend.unload()
        self.assertIs(backend.has_loaded_model(), False)


class _Stop(Exception):
    pass


class _Pipe:
    """Ends a request at its first use of the pipeline after the load."""

    @property
    def dtype(self):
        raise _Stop()


class TestDiffusersOffloadFloorIsPerRequest(unittest.TestCase):
    def setUp(self):
        from abstractvision.backends import huggingface_diffusers as hd

        self.hd = hd
        self.backend = hd.HuggingFaceDiffusersVisionBackend(
            config=hd.HuggingFaceDiffusersBackendConfig(model_id="x", device="cuda")
        )
        self.loads = []
        self.gate = None

        def load(kind, *, memory_mode):
            self.loads.append((kind, memory_mode))
            if self.gate is not None and memory_mode == "lowest":
                self.gate.wait(5)
            return _Pipe()

        for target, kwargs in (
            (hd, {"attribute": "_lazy_import_torch", "return_value": types.SimpleNamespace()}),
            (self.backend, {"attribute": "_get_or_load_pipeline", "side_effect": load}),
            (self.backend, {"attribute": "_preload_pipeline_kind", "return_value": "t2i"}),
            (self.backend, {"attribute": "_is_pipeline_warm", "return_value": True}),
            (self.backend, {"attribute": "_supports_backend_task", "return_value": True}),
            (self.backend, {"attribute": "normalize_image_generation_request", "side_effect": lambda r: r}),
        ):
            patcher = patch.object(target, kwargs.pop("attribute"), **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = patch.dict("os.environ", NO_ENV)
        env.start()
        self.addCleanup(env.stop)

    def _generate(self, mode):
        from abstractvision.types import ImageGenerationRequest

        with self.assertRaises(_Stop):
            self.backend.generate_image(ImageGenerationRequest(prompt="fox", memory_mode=mode))

    def test_each_load_gets_the_mode_of_the_request_it_serves_with_a_preload_between(self):
        self._generate("low")
        self.backend.preload()
        self._generate("lowest")
        self._generate("default")
        self.assertEqual(
            self.loads, [("t2i", "low"), ("t2i", "default"), ("t2i", "lowest"), ("t2i", "default")]
        )
        self.assertFalse(hasattr(self.backend, "_offload_floor"))

    def test_concurrent_requests_load_in_lock_order_each_with_its_own_mode(self):
        self.gate = threading.Event()
        first = threading.Thread(target=self._generate, args=("lowest",))
        first.start()
        for _ in range(500):
            if self.loads:
                break
            threading.Event().wait(0.01)
        self.assertEqual(self.loads, [("t2i", "lowest")])
        second = threading.Thread(target=self._generate, args=("default",))
        preload = threading.Thread(target=self.backend.preload)
        second.start()
        preload.start()
        threading.Event().wait(0.1)
        self.assertEqual(self.loads, [("t2i", "lowest")])  # the others wait for the backend lock
        self.gate.set()
        for thread in (first, second, preload):
            thread.join(5)
        self.assertEqual(self.loads[0], ("t2i", "lowest"))
        self.assertEqual(sorted(self.loads[1:]), [("t2i", "default"), ("t2i", "default")])

    def test_the_real_loader_reuses_or_reloads_from_the_mode_it_is_given(self):
        hd = self.hd
        backend = hd.HuggingFaceDiffusersVisionBackend(config=hd.HuggingFaceDiffusersBackendConfig(model_id="x", device="cuda"))
        offloaded = types.SimpleNamespace(_abstractvision_cpu_offload=True, _abstractvision_cpu_offload_mode="model")
        backend._pipelines["t2i"] = offloaded
        for mode in ("default", "low", "default"):
            self.assertIs(backend._get_or_load_pipeline("t2i", memory_mode=mode), offloaded)
        with patch.object(hd, "_lazy_import_diffusers", side_effect=_Stop()), patch.object(hd, "_release_pipeline_adapters"):
            with self.assertRaises(_Stop):
                backend._get_or_load_pipeline("t2i", memory_mode="lowest")
        self.assertNotIn("t2i", backend._pipelines)


class TestRefusalsNameTheirField(unittest.TestCase):
    def test_the_error_carries_field_and_supported_and_keeps_its_message(self):
        from abstractvision.errors import CapabilityNotSupportedError

        plain = CapabilityNotSupportedError("no")
        self.assertEqual((str(plain), plain.field, plain.supported), ("no", None, None))
        named = CapabilityNotSupportedError("no", field="quantize", supported=[8])
        self.assertEqual((str(named), named.field, named.supported), ("no", "quantize", (8,)))

    def test_backend_refusals_name_the_field_and_what_it_accepts(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageGenerationRequest, VideoGenerationRequest

        with patch.dict("os.environ", NO_ENV):
            for request, field, supported in (
                (ImageGenerationRequest(prompt="fox", memory_mode="lowest"), "memory_mode", ("default", "low")),
                (ImageGenerationRequest(prompt="fox", quantize=8), "quantize", ()),
            ):
                with self.subTest(field=field), self.assertRaises(CapabilityNotSupportedError) as ctx:
                    _backend(options=(), modes=("default", "low")).resolve_memory_request(request)
                self.assertEqual((ctx.exception.field, ctx.exception.supported), (field, supported))
            # A listed option the request type does not carry.
            with self.assertRaises(CapabilityNotSupportedError) as ctx:
                _backend().resolve_memory_request(ImageGenerationRequest(prompt="fox", extra={"soundscape": "w"}))
            self.assertEqual((ctx.exception.field, ctx.exception.supported), ("soundscape", ()))
            with self.assertRaises(CapabilityNotSupportedError) as ctx:
                _backend(options=()).resolve_memory_request(VideoGenerationRequest(prompt="fox", quantize=8, music="piano"))
            self.assertEqual((ctx.exception.field, ctx.exception.supported), (None, None))  # two fields

    def test_a_remote_refusal_keeps_its_field_and_values(self):
        from abstractvision.backends.openai_compatible import _http_error

        body = b'{"error": {"code": "capability_not_supported", "message": "no", "field": "memory_mode", "supported": ["default", "low"]}}'
        error = _http_error(HTTPError("http://x", 422, "Unprocessable", {}, io.BytesIO(body)))
        self.assertEqual((str(error), error.field, error.supported), ("no", "memory_mode", ("default", "low")))

    def test_the_plugin_names_a_misplaced_option(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.integrations.abstractcore_plugin import _refuse_misplaced_typed_options
        from abstractvision.types import ImageGenerationRequest

        with self.assertRaises(CapabilityNotSupportedError) as ctx:
            _refuse_misplaced_typed_options(ImageGenerationRequest, {"music": "piano"}, method="t2i")
        self.assertEqual((ctx.exception.field, ctx.exception.supported), ("music", ()))


class _Response(io.BytesIO):
    headers: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


BOUNDARY = "----abstractvision-" + "0" * 32
EXTRA = {"vae_tiling": False, "quality": "low", "nothing": None, "model": "openai/other-model"}


def _form(*parts):
    out = b""
    for name, value in parts:
        out += f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    for name, content in (("image", "img"), ("mask", "m")):
        out += (
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}.png"\r\n'
            f"Content-Type: image/png\r\n\r\n{content}\r\n"
        ).encode()
    return out + f"--{BOUNDARY}--\r\n".encode()


class TestOpenAICompatibleBodies(unittest.TestCase):
    """`extra` reaches a plain remote as 0.3.36 sent it, and AbstractCore's server nested."""

    NESTED = {"vae_tiling": False, "quality": "low", "nothing": None}

    def setUp(self):
        from abstractvision.backends import openai_compatible as oc

        self.oc = oc
        self.discovery = None  # None: HTTP 404 (a plain remote); an int: that status; a dict: the answer
        self.posts, self.gets = [], []
        self.backend = oc.OpenAICompatibleVisionBackend(
            config=oc.OpenAICompatibleBackendConfig(
                base_url="http://127.0.0.1:9/v1", model_id="image-model", text_to_video_path="/videos/generations"
            )
        )

        def urlopen(req, timeout=None):
            if req.get_method() == "GET":
                self.gets.append(req.full_url)
                if not isinstance(self.discovery, dict):
                    raise HTTPError(req.full_url, self.discovery or 404, "status", {}, io.BytesIO(b"{}"))
                return _Response(json.dumps(self.discovery).encode())
            self.posts.append((req.full_url.rsplit("/v1", 1)[1], req.data))
            return _Response(b'{"data": [{"b64_json": "iVBORw0KGgo="}]}')

        for patcher in (
            patch.object(oc, "urlopen", urlopen),
            patch.object(oc.uuid, "uuid4", return_value=types.SimpleNamespace(hex="0" * 32)),
            patch.dict("os.environ", NO_ENV),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _bodies(self):
        from abstractvision.types import ImageEditRequest, ImageGenerationRequest, VideoGenerationRequest

        self.posts.clear()
        self.backend.generate_image(
            ImageGenerationRequest(
                prompt="fox", negative_prompt="blur", width=64, height=48, seed=3, steps=4,
                guidance_scale=2.5, extra=dict(EXTRA),
            )
        )
        self.backend.edit_image(ImageEditRequest(prompt="fox", image=b"img", mask=b"m", seed=3, steps=4, extra=dict(EXTRA)))
        self.backend.generate_video(
            VideoGenerationRequest(prompt="fox", width=64, height=48, fps=8, num_frames=9, seed=3, extra=dict(EXTRA))
        )
        return dict(self.posts)

    def _typed_request(self):
        from abstractvision.types import ImageGenerationRequest

        self.backend.generate_image(ImageGenerationRequest(prompt="fox", memory_mode="default"))

    def _assert_0_3_36_bodies(self, bodies):
        # Captured from AbstractVision 0.3.36 (eea267d) for the same three requests.
        self.assertEqual(
            bodies,
            {
                "/images/generations": b'{"prompt": "fox", "n": 1, "model": "other-model", "response_format": "b64_json", '
                b'"negative_prompt": "blur", "size": "64x48", "width": 64, "height": 48, "seed": 3, "steps": 4, '
                b'"guidance_scale": 2.5, "vae_tiling": false, "quality": "low", "nothing": null}',
                "/images/edits": _form(
                    ("prompt", "fox"), ("model", "other-model"), ("seed", "3"), ("steps", "4"),
                    ("vae_tiling", "False"), ("quality", "low"),
                ),
                "/videos/generations": b'{"prompt": "fox", "response_format": "b64_json", "n": 1, "model": "other-model", '
                b'"width": 64, "height": 48, "fps": 8, "num_frames": 9, "seed": 3, "vae_tiling": false, "quality": "low", '
                b'"nothing": null}',
            },
        )

    def _assert_nested(self, bodies):
        image = json.loads(bodies["/images/generations"])
        self.assertEqual((image["model"], image["extra"]), ("other-model", self.NESTED))
        self.assertFalse(set(self.NESTED) & set(image))
        video = json.loads(bodies["/videos/generations"])
        self.assertEqual((video["model"], video["extra"]), ("other-model", self.NESTED))
        self.assertEqual(
            bodies["/images/edits"],
            _form(("prompt", "fox"), ("model", "other-model"), ("seed", "3"), ("steps", "4"), ("extra_json", json.dumps(self.NESTED))),
        )

    def test_a_plain_remote_is_asked_once_and_gets_the_0_3_36_body_byte_for_byte(self):
        from abstractvision.types import ImageGenerationRequest

        for _ in range(3):
            self._assert_0_3_36_bodies(self._bodies())
        # Keys named like options are opaque here (0.3.36 body, captured from eea267d).
        self.posts.clear()
        self.backend.generate_image(ImageGenerationRequest(prompt="fox", extra=dict(OPAQUE_EXTRA)))
        self.assertEqual(
            self.posts[0][1],
            b'{"prompt": "fox", "n": 1, "model": "image-model", "response_format": "b64_json", "quantize": "q4_k", '
            b'"soundscape": "wind", "music": "piano", "generate_audio": "no", "audio_shift": "1.5"}',
        )
        self.assertEqual(len(self.gets), 1)

    def test_a_failed_discovery_is_attempted_once_for_extra_and_again_for_a_typed_option(self):
        self.discovery = 503
        for _ in range(3):
            self._assert_0_3_36_bodies(self._bodies())
        self.assertEqual(len(self.gets), 1)
        self.assertIsNone(self.backend._serves_vision_routes)
        self.discovery = {"models": [{"id": "image-model", "memory_modes": ["default"]}]}
        self._typed_request()  # a typed option cannot proceed without knowing: it asks again
        self.assertEqual(len(self.gets), 2)
        self._assert_nested(self._bodies())

    def test_the_first_request_with_extra_to_abstractcore_s_server_is_nested_and_stays_so(self):
        self.discovery = {"models": [{"id": "image-model", "memory_modes": ["default"]}]}
        self._assert_nested(self._bodies())
        self.assertEqual(len(self.gets), 1)
        self.backend._catalog_at -= 3600  # the contents expire; the remote's kind does not
        self.discovery = 503
        self._assert_nested(self._bodies())
        self.assertEqual(len(self.gets), 1)

    def test_a_request_with_neither_extra_nor_an_option_never_asks(self):
        from abstractvision.types import ImageEditRequest, ImageGenerationRequest, VideoGenerationRequest

        self.discovery = {"models": [{"id": "image-model", "memory_modes": ["default"]}]}
        for _ in range(3):
            self.backend.generate_image(ImageGenerationRequest(prompt="fox", width=64, height=48))
            self.backend.edit_image(ImageEditRequest(prompt="fox", image=b"img"))
            self.backend.generate_video(VideoGenerationRequest(prompt="fox", extra={"model": "image-model"}))
        self.assertEqual((len(self.posts), self.gets), (9, []))

    def test_plugin_requests_with_backend_parameters_reach_abstractcore_s_server_nested(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        self.discovery = {"models": [{"id": "image-model", "memory_modes": ["default"]}]}
        cap = _AbstractVisionCapability(types.SimpleNamespace(config={}))
        with patch.object(cap, "_make_openai_backend", return_value=self.backend):
            cap.t2i("fox", provider="openai-compatible", model="image-model", vae_tiling=True)
            cap.i2i("edit", b"img", provider="openai-compatible", model="image-model", strength=0.75)
        (_, image), (_, edit) = self.posts
        self.assertEqual(json.loads(image)["extra"], {"vae_tiling": True})
        self.assertNotIn(b'name="strength"', edit)
        self.assertIn(b'name="extra_json"\r\n\r\n{"strength": 0.75}\r\n', edit)
        self.assertEqual(len(self.gets), 1)


class TestOpenAIHostIsParsed(unittest.TestCase):
    def test_only_the_openai_host_is_openai(self):
        from abstractvision.backends.openai_compatible import looks_like_openai_api

        for url, expected in (
            ("https://api.openai.com/v1", True),
            ("https://API.OpenAI.com:443/v1", True),
            ("https://api.openai.com.evil.example/v1", False),
            ("http://127.0.0.1:9/api.openai.com/v1", False),
            ("http://localhost:8000/v1", False),
            ("", False),
        ):
            with self.subTest(url=url):
                self.assertIs(looks_like_openai_api(url), expected)


if __name__ == "__main__":
    unittest.main()
