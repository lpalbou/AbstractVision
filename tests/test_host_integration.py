"""What a host application embedding AbstractVision relies on: engine metadata, cancellation, load-time
quantize, route rows, the progress-callback contract, the AbstractCore facade, and import hygiene."""

import base64
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X9+QAAAABJRU5ErkJggg=="
)
REPO_ROOT = Path(__file__).resolve().parents[1]


class _HoldsUntilUnload:
    """A local backend stand-in: it holds its model from a preload or a run until `unload`."""

    holds = False

    def has_loaded_model(self):
        return self.holds


class _Image:
    def save(self, fp, format=None):  # noqa: A002
        fp.write(PNG_1X1)


class _Generated:
    image = _Image()

    def _get_metadata(self):
        return {"seed": 7, "steps": 4, "guidance": 1.0, "quantize": None, "lora_paths": None}


class _Registry:
    def __init__(self):
        self.subscribers = []

    def subscribe_progress(self, callback, *, task=None):
        self.subscribers.append(callback)
        return lambda: self.subscribers.remove(callback)


class _FakeFlux2:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.callbacks = _Registry()
        self.runs = 0
        _FakeFlux2.instances.append(self)

    def generate_image(self, **kwargs):
        self.runs += 1
        for step in range(4):
            event = SimpleNamespace(phase="denoise", step=step + 1, total_steps=4, frame=None, total_frames=None, task="text-to-image", timestep=None)
            for callback in list(self.callbacks.subscribers):
                callback(event)
        return _Generated()


class _FakeModelConfig:
    @staticmethod
    def from_name(model_name, base_model=None):
        return "config"


def _flux_rows(**overrides):
    row = dict(public_task="text-to-image", default_for_task=True, supports_block_streaming=False, validated_quantization_bits=(), measured_runs=())
    row.update(overrides)
    return SimpleNamespace(capabilities=(SimpleNamespace(**row),))


class TestMlxGenHostContract(unittest.TestCase):
    def setUp(self):
        _FakeFlux2.instances = []
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        cache = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for repo in ("AbstractFramework/flux.2-klein-4b-8bit", "black-forest-labs/FLUX.2-klein-4B"):
            snap = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc"
            (snap / "transformer").mkdir(parents=True)
            (snap / "transformer" / "0.safetensors").write_bytes(b"x")
            (snap.parent.parent / "refs").mkdir()
            (snap.parent.parent / "refs" / "main").write_text("abc")
        self.stack.enter_context(patch.dict("os.environ", {"HF_HUB_CACHE": str(cache), "ABSTRACTVISION_MEMORY_MODE": ""}, clear=True))
        self.stack.enter_context(
            patch(
                "abstractvision.backends.mflux._lazy_import_mflux",
                return_value=(_FakeModelConfig, FileNotFoundError, _FakeFlux2, _FakeFlux2, object, object),
            )
        )
        # MLX-Gen's `--quantize` choices, which the stand-in runtime does not provide.
        self.stack.enter_context(
            patch("abstractvision.backends.mflux._mlx_gen_quantize_choices", return_value=(3, 4, 5, 6, 8))
        )
        self.rows = _flux_rows()
        self.stack.enter_context(
            patch(
                "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                side_effect=lambda: (lambda **kwargs: self.rows, None),
            )
        )

    def _backend(self, model="AbstractFramework/flux.2-klein-4b-8bit", base_model=None):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        return MFluxVisionBackend(config=MFluxBackendConfig(model=model, base_model=base_model))

    def _source_backend(self):
        source = Path(self.stack.enter_context(tempfile.TemporaryDirectory())) / "FLUX.2-klein-4B"
        (source / "transformer").mkdir(parents=True)
        (source / "transformer" / "0.safetensors").write_bytes(b"x")
        return self._backend(str(source), base_model="flux2-klein-4b")

    def test_engine_metadata_comes_back_with_the_result(self):
        from abstractvision.types import ImageGenerationRequest

        asset = self._backend().generate_image(ImageGenerationRequest(prompt="fox", steps=4, seed=7))
        self.assertEqual(asset.metadata["mlx_gen"]["seed"], 7)
        self.assertEqual(asset.metadata["mlx_gen"]["guidance"], 1.0)

    def test_cancel_event_stops_between_steps_as_a_distinct_outcome_and_keeps_the_model(self):
        from abstractvision.errors import GenerationCancelledError
        from abstractvision.types import ImageGenerationRequest

        cancel = threading.Event()
        steps = []

        def on_progress(event):
            steps.append(event.step)
            if event.step == 2:
                cancel.set()

        backend = self._backend()
        with self.assertRaises(GenerationCancelledError):
            backend.generate_image(
                ImageGenerationRequest(prompt="fox", steps=4, cancel_event=cancel, extra={"on_progress": on_progress})
            )
        self.assertEqual(steps, [1, 2])
        self.assertTrue(backend.has_loaded_model())
        self.assertEqual(backend.get_capabilities().cancellation, "step")
        cancel.clear()
        backend.generate_image(ImageGenerationRequest(prompt="fox", steps=4, cancel_event=cancel))
        self.assertEqual(len(_FakeFlux2.instances), 1)  # same instance served the next request

    def test_cancel_event_without_is_set_is_a_value_error(self):
        from abstractvision.types import ImageGenerationRequest

        with self.assertRaisesRegex(ValueError, "is_set"):
            self._backend().generate_image(ImageGenerationRequest(prompt="fox", cancel_event=object()))

    def test_progress_callback_errors_are_logged_and_the_run_completes(self):
        from abstractvision.types import ImageGenerationRequest

        calls = []

        def broken(current, total):
            calls.append(current)
            raise RuntimeError("host bug")

        with self.assertLogs("abstractvision.backends.mflux", level="ERROR") as logs:
            asset = self._backend().generate_image_with_progress(
                ImageGenerationRequest(prompt="fox", steps=4), progress_callback=broken
            )
        self.assertTrue(asset.data.startswith(b"\x89PNG"))
        self.assertEqual(calls, [1, 2, 3, 4])
        self.assertEqual(len(logs.output), 1)

    def test_load_time_quantize_is_typed_and_validated(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageGenerationRequest

        source = self._source_backend()
        bits = source.get_capabilities().quantize_bits
        self.assertIn(8, bits)
        source.generate_image(ImageGenerationRequest(prompt="fox", quantize=8))
        self.assertEqual(_FakeFlux2.instances[-1].init_kwargs["quantize"], 8)
        with self.assertRaisesRegex(CapabilityNotSupportedError, "accepted"):
            source.generate_image(ImageGenerationRequest(prompt="fox", quantize=7))
        with self.assertRaisesRegex(CapabilityNotSupportedError, "prepared MLX-Gen package"):
            self._backend().generate_image(ImageGenerationRequest(prompt="fox", quantize=4))
        self.rows = _flux_rows(validated_quantization_bits=(8,))
        self.assertEqual(source.quantize_bits(), (8,))

    def test_route_rows_are_available_without_importing_mlx_gen_in_the_host(self):
        from abstractvision import VisionManager

        rows = VisionManager(backend=self._backend()).describe_routes(task="text_to_image")
        self.assertEqual(rows[0]["public_task"], "text-to-image")
        json.dumps(rows)
        self.assertEqual(VisionManager(backend=self._backend()).describe_routes(task="text_to_video"), [])


class TestOtherBackendsRefuseHostOptions(unittest.TestCase):
    def test_quantize_and_cancel_on_other_backends(self):
        from abstractvision.backends.openai_compatible import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError, GenerationCancelledError
        from abstractvision.types import ImageGenerationRequest

        backend = OpenAICompatibleVisionBackend(config=OpenAICompatibleBackendConfig(base_url="http://127.0.0.1:9/v1"))
        with patch("abstractvision.backends.openai_compatible.urlopen", side_effect=AssertionError("sent")):
            with self.assertRaisesRegex(CapabilityNotSupportedError, "quantize"):
                backend.generate_image(ImageGenerationRequest(prompt="p", quantize=8))
            cancelled = threading.Event()
            cancelled.set()
            with self.assertRaises(GenerationCancelledError):
                backend.generate_image(ImageGenerationRequest(prompt="p", cancel_event=cancelled))
        self.assertEqual(backend.get_capabilities().cancellation, "start")
        self.assertEqual(backend.describe_routes(), [])


class TestAbstractCoreFacadeTypedOptions(unittest.TestCase):
    def _capability(self, seen):
        import abstractvision.backends.mflux as mflux_backend
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability
        from abstractvision.types import GeneratedAsset, VisionBackendCapabilities

        class _FakeBackend(_HoldsUntilUnload):
            def __init__(self, *, config):
                pass

            def get_capabilities(self):
                return VisionBackendCapabilities(supports_control_image=True, memory_modes=("default",))

            def generate_image(self, request):
                self.holds = True
                seen["request"] = request
                return GeneratedAsset(media_type="image", data=b"png", mime_type="image/png", metadata={})

            def generate_video(self, request):
                self.holds = True
                seen["request"] = request
                return GeneratedAsset(media_type="video", data=b"mp4", mime_type="video/mp4", metadata={})

        class _Owner:
            config = {"vision_backend": "mflux", "vision_mflux_model": "AbstractFramework/qwen-image-8bit"}

        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(mflux_backend, "MFluxVisionBackend", _FakeBackend))
        stack.enter_context(patch.dict("os.environ", {}, clear=True))
        self.addCleanup(stack.close)
        return _AbstractVisionCapability(_Owner())

    def test_control_image_reaches_the_typed_field(self):
        seen = {}
        cap = self._capability(seen)
        cap.t2i("a cat", control_image=PNG_1X1, control_strength=0.6)
        self.assertEqual(seen["request"].control_image, PNG_1X1)
        self.assertEqual(seen["request"].control_strength, 0.6)
        self.assertNotIn("control_image", seen["request"].extra or {})

    def test_guidance_2_on_an_image_call_is_refused_not_a_type_error(self):
        from abstractvision.errors import CapabilityNotSupportedError

        cap = self._capability({})
        with self.assertRaisesRegex(CapabilityNotSupportedError, "guidance_2"):
            cap.t2i("a cat", guidance_2=3.0)
        with self.assertRaisesRegex(CapabilityNotSupportedError, "soundscape"):
            cap.t2i_batch("a cat", soundscape="wind")

    def test_video_calls_forward_the_soundtrack_options_typed(self):
        seen = {}
        cap = self._capability(seen)
        cap.t2v("fox", soundscape="wind", music="piano", audio_shift=3.0, generate_audio=False, quantize=8)
        request = seen["request"]
        self.assertEqual((request.soundscape, request.music, request.audio_shift, request.generate_audio, request.quantize), ("wind", "piano", 3.0, False, 8))
        self.assertEqual(request.extra or {}, {})

    def test_allow_lists_are_the_request_types_fields(self):
        from abstractvision import types as av_types
        from abstractvision.integrations.abstractcore_plugin import _request_kwarg_names

        for cls in (av_types.ImageGenerationRequest, av_types.ImageEditRequest, av_types.ImageUpscaleRequest, av_types.VideoGenerationRequest):
            names = _request_kwarg_names(cls)
            self.assertTrue(names <= {field for field in cls.__dataclass_fields__}, cls.__name__)
        self.assertNotIn("guidance_2", _request_kwarg_names(av_types.ImageGenerationRequest))
        self.assertIn("control_image", _request_kwarg_names(av_types.ImageGenerationRequest))
        self.assertIn("prompt", _request_kwarg_names(av_types.ImageToVideoRequest))


class TestEmbeddingHygiene(unittest.TestCase):
    def test_import_and_manager_without_abstractcore_write_nothing_and_change_nothing(self):
        code = r"""
import json, logging, os, sys
sys.modules["abstractcore"] = None  # AbstractCore not installed
env = dict(os.environ)
root = (list(logging.getLogger().handlers), logging.getLogger().level)
import abstractvision
from abstractvision import VisionManager
from abstractvision.backends import VisionBackend
from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
import abstractvision.integrations.abstractcore_plugin
import abstractvision.integrations.abstractcore
from abstractvision.types import GeneratedAsset

class Backend(VisionBackend):
    def generate_image(self, request):
        return GeneratedAsset(media_type="image", data=b"x", mime_type="image/png")
    def edit_image(self, request): raise NotImplementedError
    def generate_angles(self, request): raise NotImplementedError
    def generate_video(self, request): raise NotImplementedError
    def image_to_video(self, request): raise NotImplementedError

VisionManager(backend=MFluxVisionBackend(config=MFluxBackendConfig(model="x")))
asset = VisionManager(backend=Backend()).generate_image("p")
print(json.dumps({
    "env_added": sorted(set(os.environ) - set(env)),
    "env_changed": sorted(k for k in env if os.environ.get(k) != env[k]),
    "root_changed": (list(logging.getLogger().handlers), logging.getLogger().level) != root,
    "handlers": sorted(n for n, lg in logging.root.manager.loggerDict.items() if isinstance(lg, logging.Logger) and lg.handlers),
    "heavy": sorted(m for m in ("mflux", "mlx", "torch", "diffusers", "transformers") if sys.modules.get(m) is not None),
    "data": asset.data.decode(),
}))
"""
        with tempfile.TemporaryDirectory() as home:
            env = {key: value for key, value in os.environ.items() if not key.startswith(("HF_", "ABSTRACTVISION_", "XDG_"))}
            env.update(HOME=home, PYTHONPATH=str(REPO_ROOT / "src"))
            out = subprocess.run([sys.executable, "-c", code], env=env, cwd=home, capture_output=True, text=True, check=True)
            report = json.loads(out.stdout.strip().splitlines()[-1])
            written = [str(path) for path in Path(home).rglob("*")]
        self.assertEqual(report, {"env_added": [], "env_changed": [], "root_changed": False, "handlers": [], "heavy": [], "data": "x"})
        self.assertEqual(written, [])


if __name__ == "__main__":
    unittest.main()
