"""MiniMax-H3 (video with a stereo soundtrack) through the MLX-Gen backend, with fakes (no weights)."""

import base64
import contextlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401



def _png(width, height):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (width, height), (200, 120, 40)).save(buf, format="PNG")
    return buf.getvalue()


# A keyframe the backend can decode (it opens every keyframe before any weights load).
PNG_1X1 = _png(16, 9)
MP4 = b"\x00\x00\x00\x18ftypmp42h3"


def _wav_bytes():
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(32000)
        handle.writeframes(b"\x00\x00" * 2 * 32)
    return buf.getvalue()


_WAV_BYTES = _wav_bytes()


def _h3_row(task, **overrides):
    row = dict(
        public_task=task,
        default_for_task=True,
        supports_negative_prompt=False,
        supports_guidance=False,
        supports_lora=True,
        lora_target_roles=("transformer",),
        supports_block_streaming=True,
        generates_audio=True,
        audio_channels=2,
        audio_sample_rate=32000,
        dimension_multiple=32,
        min_frames=124,
        max_frames=345,
        frame_multiple=17,
        frame_remainder=5,
        frame_rounding="up",
        output_fps=24.0,
        default_steps=8,
        default_width=960,
        default_height=544,
        default_frames=124,
        default_flow_shift=12.0,
        default_audio_shift=3.0,
        validated_quantization_bits=(4, 8),
        measured_runs=(SimpleNamespace(footprint_bytes=1, low_ram=True, stream_blocks=True),),
    )
    row.update(overrides)
    return SimpleNamespace(**row)


def _h3_capabilities(**overrides):
    return SimpleNamespace(
        capabilities=(_h3_row("text-to-video", **overrides), _h3_row("image-to-video", **overrides))
    )


class _FakeH3Video:
    def __init__(self, kwargs, *, mux=True, model=None):
        self.kwargs = kwargs
        self.mux = mux
        self.model = model

    def save(self, path=None, export_json_metadata=False, overwrite=True):
        path = Path(path)
        path.write_bytes(MP4)
        init = self.model.init_kwargs if self.model is not None else {}
        metadata = {
            # As MLX-Gen writes them: the model and adapter paths it read, the keyframe path it was given.
            "model": init.get("model_path"),
            "lora_paths": init.get("lora_paths"),
            "image_path": self.kwargs.get("image_path"),
            "video_path": None,
            "seed": self.kwargs["seed"],
            "prompt": "integrated_multimodal_description: " + self.kwargs["prompt"],
            "steps": self.kwargs.get("num_inference_steps", 8),
            "width": self.kwargs.get("width", 960),
            "height": self.kwargs.get("height", 544),
            "frames": 124,
            "fps": 24,
            "requested_frames": self.kwargs.get("num_frames", 124),
            "duration_seconds": 5.167,
            "video_shift": self.kwargs.get("video_shift", 12.0),
            "audio_shift": self.kwargs.get("audio_shift", 3.0),
            "transformer_released": bool(self.kwargs.get("release_transformer")),
            "runtime_memory": {"process_peak_footprint_bytes": 123},
        }
        if init.get("lora_paths"):
            # As MLX-Gen 0.40.1 (LoRAFileReport.to_dict): the paths it was given and resolved.
            metadata["lora_application_reports"] = [
                {"requested_path": p, "resolved_path": p, "scale": 1.0, "role": "transformer", "applied_target_count": 2}
                for p in init["lora_paths"]
            ]
            metadata["lora_applied_file_count"] = len(init["lora_paths"])
        if self.kwargs.get("generate_audio", True):
            metadata.update(
                audio_present=True,
                audio_source="generated",
                audio_channels=2,
                audio_sample_rate=32000,
                audio_duration_seconds=5.1667,
            )
            if self.mux:
                metadata.update(audio_muxed=True, audio_codec="aac", audio_mux_mode="ffmpeg_copy_video_aac_audio")
            else:
                wav = path.with_suffix(".wav")
                wav.write_bytes(_WAV_BYTES)
                metadata.update(audio_muxed=False, audio_mux_reason="boom", audio_sidecar_path=str(wav))
        if export_json_metadata:
            # MLX-Gen's sidecar name (GeneratedVideo.save_metadata): clip.mp4 -> clip.metadata.json.
            path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
        return path


class _FakeMiniMaxH3:
    instances = []
    mux = True
    during_step = None
    # Called with the backend-visible state at construction (see the model-key switch test).
    on_init = None
    streams = True
    # Raised as MLX-Gen's preflights do: "init" from the constructor, "run" from generate_video.
    memory_error = None

    def __init__(self, **kwargs):
        if _FakeMiniMaxH3.on_init is not None:
            _FakeMiniMaxH3.on_init()
        if _FakeMiniMaxH3.memory_error == "init":
            raise MemoryError("MiniMax-H3: the weights need about 70 GiB, and this machine has 64 GiB.")
        self.init_kwargs = dict(kwargs)
        self.block_stream = object() if kwargs.get("stream_blocks") and _FakeMiniMaxH3.streams else None
        self.transformer = object()
        self.transformer_loads = 1
        self.generate_calls = []
        _FakeMiniMaxH3.instances.append(self)

    def generate_video(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        if self.transformer is None:
            # As MLX-Gen: a released transformer is read again by the next run.
            self.transformer = object()
            self.transformer_loads += 1
        if _FakeMiniMaxH3.memory_error == "run":
            raise MemoryError("MiniMax-H3: 1344x768 at 345 frames needs about 140 GiB, so the run cannot fit.")
        progress = kwargs.get("progress_callback")

        def emit(phase, step):
            if progress is not None:
                progress(SimpleNamespace(phase=phase, step=step, total_steps=3, total_frames=124, frame=None, task=None, timestep=None))

        for step in range(3):
            if _FakeMiniMaxH3.during_step is not None:
                _FakeMiniMaxH3.during_step(step)
            emit("denoise", step + 1)
        emit("decode", 3)
        if kwargs.get("release_transformer") and self.block_stream is None:
            self.transformer = None
        if _FakeMiniMaxH3.during_step is not None:
            _FakeMiniMaxH3.during_step("decode")
        emit("generated", 3)
        return _FakeH3Video(kwargs, mux=_FakeMiniMaxH3.mux, model=self)


class _FakeH3Planning:
    """MLX-Gen's pure MiniMax-H3 planning (prompt composition, aspect window, adapter check)."""

    min_aspect_ratio = 1 / 4
    max_aspect_ratio = 4.0
    quantization_bits = (8, 4)

    @staticmethod
    def compose_prompt(prompt, soundscape=None, music=None):
        lines = [line.lstrip() for line in prompt.strip().splitlines()]
        for field, value in (("overall_soundscape", soundscape), ("non_diegetic_music", music)):
            if value and value.strip() and any(line.startswith(f"{field}:") for line in lines):
                raise ValueError(f"The prompt already carries a `{field}:` section.")
        return prompt

    @staticmethod
    def adapter_targets_modulation(path):
        return Path(path).read_bytes() == b"adaln"


class _FakeModelConfig:
    @staticmethod
    def from_name(model_name, base_model=None):
        return ("config", str(model_name), base_model)


_PREPARED_SHARD = b"mlx-gen prepared shard"


class _FakeWeightLoader:
    """MLX-Gen's StreamingWeightLoader.is_prepared: whether a component's shard headers carry its metadata."""

    @staticmethod
    def is_prepared(component):
        return any(shard.read_bytes() == _PREPARED_SHARD for shard in Path(component).glob("*.safetensors"))


class TestMiniMaxH3Backend(unittest.TestCase):
    def setUp(self):
        _FakeMiniMaxH3.instances = []
        _FakeMiniMaxH3.mux = True
        _FakeMiniMaxH3.during_step = None
        _FakeMiniMaxH3.on_init = None
        _FakeMiniMaxH3.streams = True
        _FakeMiniMaxH3.memory_error = None
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.package = self.tmp / "minimax-h3-8bit"
        (self.package / "transformer").mkdir(parents=True)
        (self.package / "transformer" / "0.safetensors").write_bytes(_PREPARED_SHARD)
        self.source = self.tmp / "MiniMax-H3"
        (self.source / "transformer").mkdir(parents=True)
        (self.source / "transformer" / "0.safetensors").write_bytes(b"x")
        self.caps = _h3_capabilities()
        self.stack.enter_context(patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": ""}))
        self.stack.enter_context(
            patch(
                "abstractvision.backends.mflux._lazy_import_mflux",
                return_value=(_FakeModelConfig, FileNotFoundError, object, object, object, object),
            )
        )
        self.stack.enter_context(
            patch("abstractvision.backends.mflux._lazy_import_mflux_minimax_h3", return_value=_FakeMiniMaxH3)
        )
        self.stack.enter_context(
            patch(
                "abstractvision.backends.mflux._lazy_import_mlx_gen_weight_loader",
                return_value=(_FakeWeightLoader, ValueError),
            )
        )
        self.stack.enter_context(
            patch("abstractvision.backends.mflux._lazy_import_mlx_gen_h3_planning", return_value=_FakeH3Planning)
        )
        self.cache_clears = []
        self.stack.enter_context(
            patch(
                "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                side_effect=lambda: (lambda **kwargs: self.caps, None),
            )
        )
        self.limits = []
        self.stack.enter_context(
            patch(
                "abstractvision.backends.mflux._set_mlx_cache_limit",
                side_effect=lambda value: self.limits.append(value) or 8 * 1024**3,
            )
        )
        self.stack.enter_context(patch("abstractvision.backends.mflux._apply_mlx_gen_default_cache_limit"))
        self.stack.enter_context(
            patch("abstractvision.backends.mflux._clear_mlx_cache", side_effect=lambda: self.cache_clears.append(1))
        )

    def _backend(self, path=None, base_model="minimax-h3-turbo-544p"):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        return MFluxVisionBackend(
            config=MFluxBackendConfig(model=str(path or self.package), base_model=base_model)
        )

    def test_text_to_video_passes_the_caller_values_unchanged_and_returns_the_soundtrack(self):
        from abstractvision import VisionManager

        backend = self._backend()
        asset = VisionManager(backend=backend).generate_video(
            "A red fox trots through snow.",
            soundscape="Paws in dry snow.",
            music="Sparse piano.",
            width=640,
            height=352,
            num_frames=124,
            seed=42,
            steps=8,
            flow_shift=12.0,
            audio_shift=3.0,
        )

        model = _FakeMiniMaxH3.instances[0]
        self.assertEqual(model.init_kwargs["model_path"], str(self.package))
        # As `mlxgen generate --model <package> --base-model minimax-h3-turbo-544p`.
        self.assertEqual(model.init_kwargs["model_config"], ("config", str(self.package), "minimax-h3-turbo-544p"))
        self.assertNotIn("stream_blocks", model.init_kwargs)
        self.assertNotIn("quantize", model.init_kwargs)
        call = model.generate_calls[0]
        self.assertEqual(call["prompt"], "A red fox trots through snow.")  # the engine composes
        self.assertEqual(
            {k: call[k] for k in ("seed", "soundscape", "music", "width", "height", "num_frames", "num_inference_steps", "video_shift", "audio_shift", "generate_audio", "release_transformer")},
            {"seed": 42, "soundscape": "Paws in dry snow.", "music": "Sparse piano.", "width": 640, "height": 352, "num_frames": 124, "num_inference_steps": 8, "video_shift": 12.0, "audio_shift": 3.0, "generate_audio": True, "release_transformer": False},
        )
        self.assertNotIn("image_path", call)
        self.assertEqual(asset.data, MP4)
        self.assertEqual(asset.mime_type, "video/mp4")
        self.assertEqual((asset.audio.sample_rate, asset.audio.channels, asset.audio.muxed, asset.audio.codec), (32000, 2, True, "aac"))
        self.assertIsNone(asset.audio.data)
        self.assertEqual(asset.audio_summary()["sample_rate"], 32000)
        self.assertNotIn("audio", asset.metadata)  # the summary is the asset's; a store records it
        self.assertEqual(asset.metadata["prompt"], "integrated_multimodal_description: A red fox trots through snow.")
        self.assertEqual(asset.metadata["mlx_gen"]["runtime_memory"], {"process_peak_footprint_bytes": 123})
        self.assertEqual(asset.metadata["task"], "text_to_video")

    def test_unset_values_stay_unset_so_the_engine_applies_its_schedule(self):
        self._backend().generate_video(
            __import__("abstractvision.types", fromlist=["VideoGenerationRequest"]).VideoGenerationRequest(prompt="fox")
        )
        call = _FakeMiniMaxH3.instances[0].generate_calls[0]
        for name in ("width", "height", "num_frames", "num_inference_steps", "video_shift", "audio_shift", "soundscape", "music"):
            self.assertNotIn(name, call, name)

    def test_first_frame_image_to_video_passes_the_keyframe_as_given(self):
        from abstractvision import VisionManager

        VisionManager(backend=self._backend()).image_to_video(PNG_1X1, prompt="The fox turns.", seed=1)
        call = _FakeMiniMaxH3.instances[0].generate_calls[0]
        keyframe = Path(call["image_path"])
        self.assertFalse(keyframe.exists())  # temporary, removed after the run
        self.assertNotIn("width", call)  # MLX-Gen maps the keyframe onto its own canvas

    def test_requests_the_route_cannot_apply_are_refused_before_loading(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        backend = self._backend()
        for request, pattern in (
            (VideoGenerationRequest(prompt="fox", negative_prompt="blur"), "negative_prompt"),
            (VideoGenerationRequest(prompt="fox", guidance_scale=4.0), "guidance_scale"),
            (VideoGenerationRequest(prompt="fox", fps=16), "fps=16"),
            (VideoGenerationRequest(prompt="fox", extra={"max_sequence_length": 256}), "max_sequence_length"),
        ):
            with self.assertRaisesRegex(CapabilityNotSupportedError, pattern):
                backend.generate_video(request)
        for request, pattern in (
            (VideoGenerationRequest(prompt="fox", num_frames=346), "346 rounds to 362"),
            (VideoGenerationRequest(prompt="fox", width=650, height=352), "multiples of 32"),
        ):
            with self.assertRaisesRegex(CapabilityNotSupportedError, pattern):
                backend.generate_video(request)
        for request, pattern in (
            (VideoGenerationRequest(prompt="fox", width=640), "together"),
            (VideoGenerationRequest(prompt="  "), "needs a prompt"),
        ):
            with self.assertRaisesRegex(ValueError, pattern):
                backend.generate_video(request)
        self.assertEqual(_FakeMiniMaxH3.instances, [])
        # The grid comes from the row: 125 rounds up to 141, as MLX-Gen does.
        backend.generate_video(VideoGenerationRequest(prompt="fox", num_frames=125, fps=24))
        self.assertEqual(_FakeMiniMaxH3.instances[0].generate_calls[0]["num_frames"], 125)

    def test_frame_grid_follows_the_row(self):
        from abstractvision.backends.mflux import _h3_aligned_frames, _h3_frame_counts

        row = vars(_h3_row("text-to-video"))
        self.assertEqual(_h3_frame_counts(row)[:3], (124, 141, 158))
        self.assertEqual(_h3_frame_counts(row)[-1], 345)
        self.assertEqual(len(_h3_frame_counts(row)), 14)
        self.assertEqual((_h3_aligned_frames(125, row), _h3_aligned_frames(124, row)), (141, 124))
        other = dict(row, frame_multiple=8, frame_remainder=1, min_frames=9, max_frames=33)
        self.assertEqual(_h3_frame_counts(other), (9, 17, 25, 33))

    def test_memory_modes_map_to_low_ram_and_block_streaming(self):
        from abstractvision.backends.mflux import LOW_RAM_MLX_CACHE_LIMIT_BYTES
        from abstractvision.types import VideoGenerationRequest

        backend = self._backend()
        caps = backend.get_capabilities()
        self.assertEqual(tuple(caps.memory_modes), ("default", "low", "lowest"))
        self.assertTrue(caps.generates_audio)

        low = backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="low"))
        self.assertIs(_FakeMiniMaxH3.instances[0].generate_calls[0]["release_transformer"], True)
        self.assertEqual(self.limits[0], LOW_RAM_MLX_CACHE_LIMIT_BYTES)
        # The run freed the transformer: the instance is dropped, as on every other family.
        self.assertFalse(backend.has_loaded_model())
        self.assertEqual(low.metadata["memory_mode"], "low")

        lowest = backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="lowest"))
        streamed = _FakeMiniMaxH3.instances[-1]
        self.assertIs(streamed.init_kwargs.get("stream_blocks"), True)
        self.assertIs(streamed.generate_calls[0]["release_transformer"], True)
        self.assertTrue(lowest.metadata["stream_blocks"])
        self.assertFalse(backend.has_loaded_model())

    def test_an_instance_a_low_run_freed_is_never_asked_to_generate_again(self):
        from abstractvision import VisionManager
        from abstractvision.types import VideoGenerationRequest

        backend = self._backend()
        for mode in ("low", "low", "lowest", "low", "default", "default"):
            backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode=mode))
        VisionManager(backend=backend).generate_video_batch("A fox.", count=2, memory_mode="low")
        # Every request after a low or lowest one builds its own instance; the one the default
        # requests loaded freed nothing, so it serves them and the batch's first low item, which
        # frees it. No instance ever reads its transformer a second time.
        self.assertEqual([len(model.generate_calls) for model in _FakeMiniMaxH3.instances], [1, 1, 1, 1, 3, 1])
        self.assertEqual([model.transformer_loads for model in _FakeMiniMaxH3.instances], [1] * 6)
        self.assertFalse(backend.has_loaded_model())

    def test_lowest_with_load_time_quantize_is_refused(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        backend = self._backend(path=self.source)
        with self.assertRaisesRegex(CapabilityNotSupportedError, "streams the stored weights"):
            backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="lowest", quantize=8))
        self.assertEqual(_FakeMiniMaxH3.instances, [])

    def test_a_mode_the_route_does_not_list_is_refused_before_the_model_is_resolved(self):
        from abstractvision import VisionManager
        from abstractvision.backends.mflux import MFluxVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError

        self.caps = _h3_capabilities(supports_block_streaming=False)
        vm = VisionManager(backend=self._backend())
        # Resolving the model may download it (allow_download): the refusal comes first.
        with patch.object(MFluxVisionBackend, "_resolve_model", side_effect=AssertionError("resolved")):
            with self.assertRaisesRegex(CapabilityNotSupportedError, "memory_mode='lowest'"):
                vm.generate_video("A fox.", memory_mode="lowest")
            with self.assertRaisesRegex(CapabilityNotSupportedError, "memory_mode='lowest'"):
                vm.image_to_video(PNG_1X1, prompt="A fox.", extra={"memory_mode": "lowest"})
        self.assertEqual(_FakeMiniMaxH3.instances, [])

    def test_load_time_quantize_is_validated_against_the_row(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        source = self._backend(path=self.source)
        self.assertEqual(source.get_capabilities().quantize_bits, (4, 8))
        asset = source.generate_video(VideoGenerationRequest(prompt="fox", quantize=8))
        self.assertEqual(_FakeMiniMaxH3.instances[0].init_kwargs["quantize"], 8)
        self.assertEqual(asset.metadata["quantization_bits"], 8)
        with self.assertRaisesRegex(CapabilityNotSupportedError, "accepted: 4, 8"):
            source.generate_video(VideoGenerationRequest(prompt="fox", quantize=6))
        with self.assertRaisesRegex(CapabilityNotSupportedError, "prepared MLX-Gen package"):
            self._backend().generate_video(VideoGenerationRequest(prompt="fox", quantize=8))

    def test_refusals_name_no_absolute_local_path(self):
        from abstractvision.types import LoRAAdapterSpec, VideoGenerationRequest

        adaln = self.tmp / "motion-adaln.safetensors"
        adaln.write_bytes(b"adaln")
        missing = self.tmp / "not-there" / "minimax-h3"
        cases = {
            "prepared package with quantize": (self._backend(), VideoGenerationRequest(prompt="fox", quantize=8)),
            "model path that does not exist": (self._backend(path=missing), VideoGenerationRequest(prompt="fox")),
            "lowest with an AdaLN adapter": (
                self._backend(),
                VideoGenerationRequest(prompt="fox", memory_mode="lowest", lora_adapters=(LoRAAdapterSpec(source=str(adaln)),)),
            ),
        }
        for case, (backend, request) in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(Exception) as caught:
                    backend.generate_video(request)
                self.assertNotIn(str(self.tmp), str(caught.exception))
                self.assertNotIn(str(self.tmp.resolve()), str(caught.exception))

    def test_an_unmuxed_soundtrack_is_returned_as_wav_bytes(self):
        from abstractvision.types import VideoGenerationRequest

        _FakeMiniMaxH3.mux = False
        asset = self._backend().generate_video(VideoGenerationRequest(prompt="fox"))
        self.assertFalse(asset.audio.muxed)
        self.assertEqual(asset.audio.data, _WAV_BYTES)
        self.assertEqual(asset.audio.mime_type, "audio/wav")
        self.assertFalse(asset.audio_summary()["muxed"])
        self.assertEqual(asset.audio_summary()["wav"], _WAV_BYTES)

    def test_silent_clip_has_no_audio_track(self):
        from abstractvision.types import VideoGenerationRequest

        asset = self._backend().generate_video(VideoGenerationRequest(prompt="fox", generate_audio=False))
        self.assertIs(_FakeMiniMaxH3.instances[0].generate_calls[0]["generate_audio"], False)
        self.assertIsNone(asset.audio)
        self.assertNotIn("audio", asset.metadata)

    def test_never_downloads_minimax_h3_weights_implicitly(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.errors import OptionalDependencyMissingError
        from abstractvision.types import VideoGenerationRequest

        with tempfile.TemporaryDirectory() as cache_td, patch.dict(
            "os.environ", {"HF_HUB_CACHE": cache_td, "ABSTRACTVISION_MEMORY_MODE": ""}, clear=True
        ), patch("abstractvision.backends.mflux.download_hf_repo_snapshot", side_effect=AssertionError("download")):
            for model in ("AbstractFramework/minimax-h3-8bit", "MiniMaxAI/MiniMax-H3"):
                backend = MFluxVisionBackend(
                    config=MFluxBackendConfig(model=model, base_model="minimax-h3-turbo-544p", allow_download=True)
                )
                with self.assertRaisesRegex(OptionalDependencyMissingError, "never downloaded during generation"):
                    backend.generate_video(VideoGenerationRequest(prompt="fox"))

    def test_cancel_event_stops_between_steps_and_keeps_the_model(self):
        from abstractvision.errors import GenerationCancelledError
        from abstractvision.types import VideoGenerationRequest

        cancel = threading.Event()
        _FakeMiniMaxH3.during_step = lambda step: cancel.set() if step == 1 else None
        backend = self._backend()
        with self.assertRaises(GenerationCancelledError):
            backend.generate_video(VideoGenerationRequest(prompt="fox", cancel_event=cancel))
        self.assertTrue(backend.has_loaded_model())
        # A default-mode run holds no cache limit.
        from abstractvision.backends import mflux

        self.assertEqual(mflux._MLX_CACHE_LIMIT_HOLDS, {})
        self.assertIsNone(mflux._mlx_cache_limit_before_holds)
        self.assertEqual(self.limits, [])
        # A request whose event is already set never starts.
        with self.assertRaises(GenerationCancelledError):
            backend.generate_video(VideoGenerationRequest(prompt="fox", cancel_event=cancel))
        self.assertEqual(len(_FakeMiniMaxH3.instances[0].generate_calls), 1)

    def test_a_raising_progress_callback_is_logged_and_does_not_stop_the_run(self):
        from abstractvision.types import VideoGenerationRequest

        seen = []

        def callback(event):
            seen.append(event.step)
            raise RuntimeError("host bug")

        with self.assertLogs("abstractvision.backends.mflux", level="ERROR") as logs:
            asset = self._backend().generate_video(
                VideoGenerationRequest(prompt="fox", extra={"on_progress": callback})
            )
        self.assertEqual(asset.data, MP4)
        self.assertEqual(seen, [1, 2, 3, 3, 3])  # three steps, the decode, the generated clip
        self.assertEqual(len(logs.output), 1)

    def test_lora_adapters_go_to_the_transformer_role_only(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import LoRAAdapterSpec, VideoGenerationRequest

        backend = self._backend()
        lora = self.tmp / "style.safetensors"
        lora.write_bytes(b"x")
        with self.assertRaisesRegex(CapabilityNotSupportedError, "not high_noise_transformer"):
            backend.generate_video(
                VideoGenerationRequest(
                    prompt="fox", lora_adapters=(LoRAAdapterSpec(source=str(lora), target_role="high_noise_transformer"),)
                )
            )
        backend.generate_video(
            VideoGenerationRequest(prompt="fox", lora_adapters=(LoRAAdapterSpec(source=str(lora), scale=0.5),))
        )
        init = _FakeMiniMaxH3.instances[-1].init_kwargs
        self.assertEqual(init["lora_paths"], [str(lora)])
        self.assertNotIn("lora_target_roles", init)

    def test_describe_routes_returns_json_safe_rows(self):
        rows = self._backend().describe_routes(task="text_to_video")
        self.assertEqual(len(rows), 1)
        json.dumps(rows)
        self.assertEqual(rows[0]["frame_multiple"], 17)
        self.assertEqual(rows[0]["measured_runs"][0]["stream_blocks"], True)
        self.assertEqual(rows[0]["lora_target_roles"], ["transformer"])

    def test_wan_and_other_backends_refuse_soundtrack_options(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend
        from abstractvision.backends.openai_compatible import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        wan = MFluxVisionBackend(config=MFluxBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers"))
        with self.assertRaisesRegex(CapabilityNotSupportedError, "soundscape"):
            wan.generate_video(VideoGenerationRequest(prompt="fox", soundscape="wind"))
        http = OpenAICompatibleVisionBackend(
            config=OpenAICompatibleBackendConfig(base_url="http://127.0.0.1:9/v1", text_to_video_path="/v")
        )
        with self.assertRaisesRegex(CapabilityNotSupportedError, "audio_shift, quantize|quantize"):
            http.generate_video(VideoGenerationRequest(prompt="fox", quantize=8, audio_shift=3.0))

    def test_provider_parameters_come_from_the_row(self):
        from abstractvision.backends.mflux import _MFLUX_MODELS, _mflux_parameter_metadata

        caps = {"text_to_video": [vars(_h3_row("text-to-video"))]}
        metadata = _mflux_parameter_metadata(
            _MFLUX_MODELS["minimax-h3-turbo-544p"], block_streaming=True, route_capabilities=caps
        )
        self.assertEqual(metadata["parameter_defaults"]["steps"], 8)
        self.assertEqual(metadata["parameter_defaults"]["width"], 960)
        self.assertEqual(metadata["parameter_constraints"]["num_frames"]["values"][:2], [124, 141])
        self.assertEqual(metadata["parameter_constraints"]["memory_mode"]["enum"], ["default", "low", "lowest"])


    # --- preload -------------------------------------------------------------------------------

    def test_preload_loads_the_video_model_and_runs_no_image_warmup(self):
        backend = self._backend()
        backend.preload()
        self.assertEqual(len(_FakeMiniMaxH3.instances), 1)
        self.assertEqual(_FakeMiniMaxH3.instances[0].generate_calls, [])
        self.assertTrue(backend.has_loaded_model())
        backend.preload()  # already loaded and warm: nothing more
        self.assertEqual(len(_FakeMiniMaxH3.instances), 1)
        self.assertNotIn("stream_blocks", _FakeMiniMaxH3.instances[0].init_kwargs)

    def test_preload_under_the_lowest_process_mode_streams_blocks_and_the_first_request_reuses_it(self):
        from abstractvision.types import VideoGenerationRequest

        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": "lowest"}):
            backend = self._backend()
            backend.preload()
            self.assertEqual(len(_FakeMiniMaxH3.instances), 1)
            self.assertTrue(_FakeMiniMaxH3.instances[0].init_kwargs.get("stream_blocks"))
            asset = backend.generate_video(VideoGenerationRequest(prompt="A fox."))
        self.assertEqual(len(_FakeMiniMaxH3.instances), 1)  # not loaded again
        self.assertEqual(asset.metadata["memory_mode"], "lowest")

    def test_a_failure_inside_preload_leaves_no_model_resident(self):
        from abstractvision.backends.mflux import MFluxVisionBackend

        backend = self._backend()
        with patch.object(MFluxVisionBackend, "_mark_preloaded", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                backend.preload()
        self.assertEqual(len(_FakeMiniMaxH3.instances), 1)
        self.assertFalse(backend.has_loaded_model())

    def test_plugin_load_resident_model_records_minimax_h3_and_unloads_it(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        owner = SimpleNamespace(
            config={
                "vision_backend": "mflux",
                "vision_mflux_model": str(self.package),
                "vision_mlx_gen_base_model": "minimax-h3-turbo-544p",
            }
        )
        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": ""}, clear=True):
            cap = _AbstractVisionCapability(owner)
            record = cap.load_resident_model({})
            self.assertTrue(record["loaded"])
            self.assertEqual(len(cap.list_resident_models()), 1)
            self.assertEqual(_FakeMiniMaxH3.instances[0].generate_calls, [])
            backend = cap._get_backend()
            self.assertTrue(backend.has_loaded_model())
            unloaded = cap.unload_resident_model({"load_id": record["load_id"]})
            self.assertEqual(unloaded["state"], "unloaded")
            self.assertEqual(cap.list_resident_models(), [])
            self.assertFalse(backend.has_loaded_model())

            # A preload that fails after loading drops what it loaded, and the plugin keeps no record.
            from abstractvision.backends.mflux import MFluxVisionBackend

            with patch.object(MFluxVisionBackend, "_mark_preloaded", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    cap.load_resident_model({})
            self.assertFalse(backend.has_loaded_model())
            self.assertEqual(cap.list_loaded_models(), [])

            # One that fails after reusing the model it already held leaves that model held.
            cap.load_resident_model({})
            held = backend._model
            with patch.object(MFluxVisionBackend, "_mark_preloaded", side_effect=RuntimeError("boom")):
                backend._warmed_model_key = None
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    cap.load_resident_model({})
            self.assertIs(backend._model, held)
            self.assertEqual(len(cap.list_loaded_models()), 1)

    def test_plugin_record_follows_the_instance_a_low_run_drops(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        owner = SimpleNamespace(
            config={
                "vision_backend": "mflux",
                "vision_mflux_model": str(self.package),
                "vision_mlx_gen_base_model": "minimax-h3-turbo-544p",
            }
        )
        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": ""}, clear=True):
            cap = _AbstractVisionCapability(owner)
            cap.t2v("A fox.")
            self.assertEqual(len(cap.list_loaded_models()), 1)
            for call in (lambda: cap.t2v("A fox.", memory_mode="low"), lambda: cap.t2v_batch("A fox.", count=2, memory_mode="low")):
                call()
                self.assertEqual(cap.list_loaded_models(), [])
                self.assertFalse(cap._get_backend().has_loaded_model())
        # One instance for the default request and the first low run, then one per batch item.
        self.assertEqual([model.transformer_loads for model in _FakeMiniMaxH3.instances], [1, 1, 1])

    # --- batches through VisionManager ---------------------------------------------------------

    def test_batch_items_are_runs_like_any_other(self):
        from abstractvision import VisionManager

        for task in ("text_to_video", "image_to_video"):
            for mode in (None, "low"):
                with self.subTest(task=task, memory_mode=mode):
                    _FakeMiniMaxH3.instances = []
                    backend = self._backend()
                    vm = VisionManager(backend=backend)
                    kwargs = {"seeds": [1, 2, 3]} | ({"memory_mode": mode} if mode else {})
                    if task == "text_to_video":
                        assets = vm.generate_video_batch("A fox.", count=3, **kwargs)
                    else:
                        assets = vm.image_to_video_batch(PNG_1X1, prompt="The fox turns.", count=3, **kwargs)
                    self.assertEqual(len(assets), 3)
                    calls = [call for model in _FakeMiniMaxH3.instances for call in model.generate_calls]
                    self.assertEqual([call["seed"] for call in calls], [1, 2, 3])
                    # In low, each item frees the transformer before its decode and its instance is
                    # dropped; in default, one instance serves all three.
                    self.assertEqual([call["release_transformer"] for call in calls], [mode == "low"] * 3)
                    self.assertEqual(len(_FakeMiniMaxH3.instances), 3 if mode == "low" else 1)
                    self.assertEqual({model.transformer_loads for model in _FakeMiniMaxH3.instances}, {1})
                    self.assertEqual(backend.has_loaded_model(), mode is None)
                    self.assertEqual([asset.metadata["task"] for asset in assets], [task] * 3)

    def test_memory_mode_in_extra_is_accepted_on_direct_backend_calls(self):
        from abstractvision.types import ImageToVideoRequest, VideoGenerationRequest

        backend = self._backend()
        backend.generate_video(VideoGenerationRequest(prompt="fox", extra={"memory_mode": "low"}))
        backend.image_to_video(ImageToVideoRequest(image=PNG_1X1, prompt="fox", extra={"memory_mode": "low"}))
        self.assertEqual(
            [call["release_transformer"] for model in _FakeMiniMaxH3.instances for call in model.generate_calls],
            [True, True],
        )

    def test_tools_leave_unset_values_to_the_schedule(self):
        import sys
        import types

        from abstractvision import LocalAssetStore, VisionManager
        from abstractvision.integrations.abstractcore import make_vision_tools

        fake_core = types.ModuleType("abstractcore")
        fake_core.tool = lambda **kw: (lambda fn: fn)
        for base_model in ("minimax-h3-turbo-544p", "minimax-h3"):
            for name, args in (
                ("vision_text_to_video", ("A fox.",)),
                ("vision_text_to_video_batch", ("A fox.",)),
                ("vision_image_to_video", ()),
                ("vision_image_to_video_batch", ()),
            ):
                with self.subTest(base_model=base_model, tool=name):
                    _FakeMiniMaxH3.instances = []
                    vm = VisionManager(
                        backend=self._backend(base_model=base_model), store=LocalAssetStore(self.tmp / "store")
                    )
                    with patch.dict(sys.modules, {"abstractcore": fake_core}):
                        tools = {fn.__name__: fn for fn in make_vision_tools(vision_manager=vm, model_id="MiniMaxAI/MiniMax-H3")}
                    kwargs = {} if args else {"image_b64": base64.b64encode(PNG_1X1).decode(), "prompt": "The fox turns."}
                    tools[name](*args, **kwargs)
                    call = _FakeMiniMaxH3.instances[0].generate_calls[0]
                    # The catalog lists steps, width, height and flow_shift without a default:
                    # MLX-Gen applies the schedule's (8 steps on Turbo, 50 on the base model).
                    for engine_name in ("num_inference_steps", "width", "height", "video_shift"):
                        self.assertNotIn(engine_name, call)

    # --- switching the model key ---------------------------------------------------------------

    def test_a_new_model_key_releases_the_old_instance_before_building_the_new_one(self):
        from abstractvision.types import VideoGenerationRequest

        backend = self._backend()
        seen = []
        _FakeMiniMaxH3.on_init = lambda: seen.append((backend._model is None, len(self.cache_clears)))
        backend.generate_video(VideoGenerationRequest(prompt="fox"))
        clears_after_first = len(self.cache_clears)
        backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="lowest"))
        self.assertEqual(len(_FakeMiniMaxH3.instances), 2)
        self.assertTrue(seen[1][0], "the default instance was still held while the lowest one was built")
        self.assertGreater(seen[1][1], clears_after_first, "the MLX cache was not cleared before the build")
        self.assertIs(_FakeMiniMaxH3.instances[1].init_kwargs.get("stream_blocks"), True)

    def test_lowest_on_weights_that_cannot_stream_is_refused_and_leaves_no_model(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        _FakeMiniMaxH3.streams = False
        backend = self._backend()
        with self.assertRaisesRegex(CapabilityNotSupportedError, "lowest"):
            backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="lowest"))
        self.assertFalse(backend.has_loaded_model())
        self.assertEqual(_FakeMiniMaxH3.instances[0].generate_calls, [])

    # --- request validation before any load ----------------------------------------------------

    def test_bad_values_are_refused_before_loading_naming_the_field(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        backend = self._backend()
        out_of_range = CapabilityNotSupportedError
        cases = (
            (dict(width=0, height=0), out_of_range, "width"),
            (dict(width=-32, height=-32), out_of_range, "width"),
            (dict(width=32, height=4096), out_of_range, "aspect ratio"),
            (dict(width=2048, height=352), out_of_range, "aspect ratio"),
            (dict(width="wide", height=352), ValueError, "width"),
            (dict(width=True, height=352), ValueError, "width"),
            (dict(width=640.5, height=352), ValueError, "width"),
            (dict(num_frames=0), out_of_range, "num_frames"),
            (dict(num_frames=-5), out_of_range, "num_frames"),
            (dict(num_frames=124.7), ValueError, "num_frames"),
            (dict(num_frames="many"), ValueError, "num_frames"),
            (dict(steps=0), out_of_range, "steps"),
            (dict(steps="x"), ValueError, "steps"),
            (dict(steps=2.5), ValueError, "steps"),
            (dict(flow_shift="abc"), ValueError, "flow_shift"),
            (dict(flow_shift=-1.0), out_of_range, "flow_shift"),
            (dict(flow_shift=float("nan")), ValueError, "flow_shift"),
            (dict(audio_shift=0.0), out_of_range, "audio_shift"),
            (dict(audio_shift="loud"), ValueError, "audio_shift"),
            (dict(soundscape=123), ValueError, "soundscape"),
            (dict(music=["a"]), ValueError, "music"),
            (dict(generate_audio="false"), ValueError, "generate_audio"),
            (dict(generate_audio=0), ValueError, "generate_audio"),
            (dict(seed="abc"), ValueError, "seed"),
            (dict(seed=-1), ValueError, "seed"),
            (dict(seed=True), ValueError, "seed"),
            (dict(fps="fast"), ValueError, "fps"),
        )
        for overrides, error, field in cases:
            with self.subTest(**{key: repr(value) for key, value in overrides.items()}):
                with self.assertRaisesRegex(error, field):
                    backend.generate_video(VideoGenerationRequest(prompt="fox", **overrides))
        self.assertEqual(_FakeMiniMaxH3.instances, [])

    def test_accepted_values_reach_the_engine_as_numbers(self):
        from abstractvision.types import VideoGenerationRequest

        self._backend().generate_video(
            VideoGenerationRequest(
                prompt="fox", width=640.0, height=352, num_frames=141.0, steps=8, flow_shift=12, audio_shift=3,
                seed=7, generate_audio=False, fps=24.0,
            )
        )
        call = _FakeMiniMaxH3.instances[0].generate_calls[0]
        self.assertEqual(
            {k: (call[k], type(call[k])) for k in ("width", "num_frames", "num_inference_steps", "video_shift", "audio_shift", "seed")},
            {"width": (640, int), "num_frames": (141, int), "num_inference_steps": (8, int), "video_shift": (12.0, float), "audio_shift": (3.0, float), "seed": (7, int)},
        )
        self.assertIs(call["generate_audio"], False)

    def test_a_prompt_section_and_the_same_field_are_refused_before_loading(self):
        from abstractvision.types import ImageToVideoRequest, VideoGenerationRequest

        backend = self._backend()
        with self.assertRaisesRegex(ValueError, "soundscape.*overall_soundscape"):
            backend.generate_video(
                VideoGenerationRequest(prompt="overall_soundscape: rain\nA fox", soundscape="wind")
            )
        with self.assertRaisesRegex(ValueError, "music.*non_diegetic_music"):
            backend.image_to_video(
                ImageToVideoRequest(image=PNG_1X1, prompt="A fox\nnon_diegetic_music: none", music="piano")
            )
        self.assertEqual(_FakeMiniMaxH3.instances, [])
        # The section alone, or the field alone, is fine.
        backend.generate_video(VideoGenerationRequest(prompt="overall_soundscape: rain\nA fox", music="piano"))

    def test_the_keyframe_is_decoded_before_loading(self):
        from abstractvision.types import ImageToVideoRequest

        backend = self._backend()
        for image, pattern in ((b"", "image.*empty"), (b"garbage", "image.*decode"), ("notbytes", "image.*bytes")):
            with self.subTest(image=image):
                with self.assertRaisesRegex(ValueError, pattern):
                    backend.image_to_video(ImageToVideoRequest(image=image, prompt="fox"))
                # The same check is part of the request's normalization, which VisionManager runs first.
                with self.assertRaisesRegex(ValueError, pattern):
                    backend.normalize_image_to_video_request(ImageToVideoRequest(image=image, prompt="fox"))
        self.assertEqual(_FakeMiniMaxH3.instances, [])

    def test_a_keyframe_outside_the_aspect_window_is_refused_when_the_canvas_is_unset(self):
        from io import BytesIO

        from PIL import Image

        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageToVideoRequest

        buf = BytesIO()
        Image.new("RGB", (100, 10)).save(buf, format="PNG")
        backend = self._backend()
        with self.assertRaisesRegex(CapabilityNotSupportedError, "aspect ratio"):
            backend.image_to_video(ImageToVideoRequest(image=buf.getvalue(), prompt="fox"))
        with self.assertRaisesRegex(CapabilityNotSupportedError, "keyframe's width x height"):
            backend.normalize_image_to_video_request(ImageToVideoRequest(image=buf.getvalue(), prompt="fox"))
        self.assertEqual(_FakeMiniMaxH3.instances, [])

    def test_lowest_refuses_an_adapter_that_trains_the_adaln_projections_before_loading(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import LoRAAdapterSpec, VideoGenerationRequest

        adaln = self.tmp / "motion-adaln.safetensors"
        adaln.write_bytes(b"adaln")
        backend = self._backend()
        with self.assertRaisesRegex(CapabilityNotSupportedError, "motion-adaln.safetensors.*AdaLN"):
            backend.generate_video(
                VideoGenerationRequest(prompt="fox", memory_mode="lowest", lora_adapters=(LoRAAdapterSpec(source=str(adaln)),))
            )
        self.assertEqual(_FakeMiniMaxH3.instances, [])
        # The same adapter is fine where the transformer stays in memory.
        backend.generate_video(
            VideoGenerationRequest(prompt="fox", memory_mode="low", lora_adapters=(LoRAAdapterSpec(source=str(adaln)),))
        )

    def test_planning_comes_from_the_installed_mlx_gen(self):
        import importlib

        from abstractvision.backends import mflux as backend_module

        try:
            importlib.import_module("mflux.models.minimax_h3.variants.minimax_h3")
        except Exception:
            self.skipTest("mlx-gen with MiniMax-H3 is not installed")
        self.stack.close()  # the real planning, not the fake
        planning = backend_module._lazy_import_mlx_gen_h3_planning()
        self.assertLess(planning.min_aspect_ratio, 1.0)
        self.assertGreater(planning.max_aspect_ratio, 1.0)
        with self.assertRaises(ValueError):
            planning.compose_prompt("overall_soundscape: rain\nA fox", "wind", None)
        self.assertTrue(callable(planning.adapter_targets_modulation))
        self.assertEqual(set(planning.quantization_bits), {4, 8})

    # --- the soundtrack on the store, plugin and tool paths --------------------------------------

    def test_audio_summary_is_one_object_for_every_layer(self):
        from abstractvision.types import VideoGenerationRequest

        base = {"present": True, "sample_rate": 32000, "channels": 2, "duration_seconds": 5.1667}
        muxed = self._backend().generate_video(VideoGenerationRequest(prompt="fox"))
        self.assertEqual(muxed.audio_summary(), {**base, "codec": "aac", "muxed": True})
        _FakeMiniMaxH3.mux = False
        loose = self._backend().generate_video(VideoGenerationRequest(prompt="fox"))
        # The WAV MLX-Gen writes is 16-bit PCM; the codec is read from the file itself.
        self.assertEqual(
            loose.audio_summary(), {**base, "codec": "pcm_s16le", "muxed": False, "wav": _WAV_BYTES}
        )
        self.assertEqual(loose.audio_summary(wav={"$artifact": "a" * 32})["wav"], {"$artifact": "a" * 32})
        silent = self._backend().generate_video(VideoGenerationRequest(prompt="fox", generate_audio=False))
        self.assertIsNone(silent.audio_summary())

    def test_an_unmuxed_soundtrack_is_stored_as_a_second_artifact(self):
        from abstractvision import LocalAssetStore, VisionManager

        _FakeMiniMaxH3.mux = False
        store = LocalAssetStore(self.tmp / "store")
        ref = VisionManager(backend=self._backend(), store=store).generate_video("A fox.")
        wav_ref = ref["audio"]["wav"]
        self.assertEqual(wav_ref["content_type"], "audio/wav")
        self.assertEqual(store.load_bytes(wav_ref["$artifact"]), _WAV_BYTES)
        stored = store.get_metadata(ref["$artifact"])["metadata"]
        self.assertEqual(stored["audio"], ref["audio"])
        self.assertEqual(stored["audio"]["muxed"], False)
        self.assertEqual(store.get_metadata(wav_ref["$artifact"])["tags"]["modality"], "audio")

    def test_a_muxed_soundtrack_is_described_in_the_stored_metadata(self):
        from abstractvision import LocalAssetStore, VisionManager

        store = LocalAssetStore(self.tmp / "store")
        ref = VisionManager(backend=self._backend(), store=store).generate_video("A fox.")
        self.assertNotIn("wav", ref["audio"])
        stored = store.get_metadata(ref["$artifact"])["metadata"]
        self.assertEqual(
            stored["audio"],
            {"present": True, "sample_rate": 32000, "channels": 2, "codec": "aac", "muxed": True, "duration_seconds": 5.1667},
        )

    def test_plugin_and_tools_return_the_soundtrack_artifact(self):
        import sys
        import types

        from abstractvision import LocalAssetStore, VisionManager
        from abstractvision.integrations.abstractcore import make_vision_tools
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        _FakeMiniMaxH3.mux = False
        fake_core = types.ModuleType("abstractcore")
        fake_core.tool = lambda **kw: (lambda fn: fn)
        store = LocalAssetStore(self.tmp / "store")
        with patch.dict(sys.modules, {"abstractcore": fake_core}):
            tools = {
                fn.__name__: fn
                for fn in make_vision_tools(
                    vision_manager=VisionManager(backend=self._backend(), store=store), model_id="MiniMaxAI/MiniMax-H3"
                )
            }
        ref = tools["vision_text_to_video"]("A fox.")
        self.assertEqual(store.load_bytes(ref["audio"]["wav"]["$artifact"]), _WAV_BYTES)
        refs = tools["vision_text_to_video_batch"]("A fox.", count=2)
        self.assertTrue(all("$artifact" in item["audio"]["wav"] for item in refs))

        class _RuntimeStore:
            def __init__(self):
                self.blobs = {}

            def store(self, content, *, content_type, run_id=None, tags=None, artifact_id=None):
                artifact_id = f"{len(self.blobs):032x}"
                self.blobs[artifact_id] = (content, content_type)
                return {"artifact_id": artifact_id}

        owner = SimpleNamespace(
            config={
                "vision_backend": "mflux",
                "vision_mflux_model": str(self.package),
                "vision_mlx_gen_base_model": "minimax-h3-turbo-544p",
            }
        )
        runtime_store = _RuntimeStore()
        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": ""}, clear=True):
            cap = _AbstractVisionCapability(owner)
            out = cap.t2v("A fox.", artifact_store=runtime_store)
            wav_id = out["audio"]["wav"]["$artifact"]
            self.assertEqual(runtime_store.blobs[wav_id], (_WAV_BYTES, "audio/wav"))
            self.assertEqual(out["metadata"]["audio"], out["audio"])
            batch = cap.t2v_batch("A fox.", count=2, artifact_store=runtime_store)
            self.assertTrue(all("$artifact" in item["audio"]["wav"] for item in batch))
            # Without an artifact store the plugin returns the MP4 bytes and says the track is not in them.
            with self.assertLogs("abstractvision.integrations.abstractcore_plugin", level="WARNING"):
                self.assertEqual(cap.t2v("A fox."), MP4)

    # --- metadata ------------------------------------------------------------------------------

    def test_metadata_carries_no_local_paths(self):
        from abstractvision import VisionManager
        from abstractvision.types import LoRAAdapterSpec

        _FakeMiniMaxH3.mux = False  # MLX-Gen then records the WAV it wrote next to the clip
        lora = self.tmp / "style.safetensors"
        lora.write_bytes(b"x")
        asset = VisionManager(backend=self._backend()).image_to_video(
            PNG_1X1, prompt="The fox turns.", lora_adapters=(LoRAAdapterSpec(source=str(lora)),)
        )
        text = json.dumps(asset.metadata)
        self.assertNotIn(str(self.tmp), text)
        self.assertNotIn(tempfile.gettempdir(), text)
        self.assertEqual(asset.metadata["model"], "minimax-h3-8bit")
        self.assertEqual(asset.metadata["mlx_gen"]["model"], "minimax-h3-8bit")
        self.assertEqual(asset.metadata["mlx_gen"]["lora_paths"], ["style.safetensors"])
        self.assertEqual(asset.metadata["lora_paths"], ["style.safetensors"])
        for reports in (asset.metadata["lora_application_reports"], asset.metadata["mlx_gen"]["lora_application_reports"]):
            self.assertEqual(
                [(r["requested_path"], r["resolved_path"], r["applied_target_count"]) for r in reports],
                [("style.safetensors", "style.safetensors", 2)],
            )
        for name in ("image_path", "video_path", "audio_sidecar_path"):
            self.assertNotIn(name, asset.metadata["mlx_gen"])

    def test_the_users_text_is_kept_as_written_in_metadata(self):
        from abstractvision.backends.mflux import _public_engine_metadata

        cleaned = _public_engine_metadata(
            {"prompt": "/imagine a fox in snow", "negative_prompt": "/blur", "model": str(self.tmp / "minimax-h3-8bit")}
        )
        self.assertEqual(cleaned, {"prompt": "/imagine a fox in snow", "negative_prompt": "/blur", "model": "minimax-h3-8bit"})

    def test_a_cached_package_is_reported_by_its_repo_id(self):
        from abstractvision.backends.mflux import _public_model_name

        snapshot = self.tmp / "hub" / "models--AbstractFramework--minimax-h3-8bit" / "snapshots" / "abc123"
        self.assertEqual(_public_model_name(str(snapshot)), "AbstractFramework/minimax-h3-8bit")
        self.assertEqual(
            _public_model_name(str(self.tmp / "hub" / "models--lightx2v--Minimax-h3-Turbo" / "snapshots" / "abc" / "turbo.safetensors")),
            "lightx2v/Minimax-h3-Turbo:turbo.safetensors",
        )
        self.assertEqual(_public_model_name("AbstractFramework/minimax-h3-8bit"), "AbstractFramework/minimax-h3-8bit")
        self.assertIsNone(_public_model_name(None))

    # --- failure, cancel and memory ------------------------------------------------------------

    def test_failure_and_cancel_clear_the_mlx_cache(self):
        from abstractvision.errors import GenerationCancelledError
        from abstractvision.types import VideoGenerationRequest

        backend = self._backend()
        backend.preload()
        cancel = threading.Event()
        _FakeMiniMaxH3.during_step = lambda step: cancel.set() if step == 1 else None
        before = len(self.cache_clears)
        with self.assertRaises(GenerationCancelledError):
            backend.generate_video(VideoGenerationRequest(prompt="fox", cancel_event=cancel))
        self.assertGreater(len(self.cache_clears), before)

        def fail(step):
            raise RuntimeError("engine failure")

        _FakeMiniMaxH3.during_step = fail
        before = len(self.cache_clears)
        with self.assertRaisesRegex(RuntimeError, "engine failure"):
            backend.generate_video(VideoGenerationRequest(prompt="fox"))
        self.assertGreater(len(self.cache_clears), before)
        self.assertTrue(backend.has_loaded_model())

    def test_cancel_in_low_mode_drops_the_model_whatever_was_freed(self):
        from abstractvision import VisionManager
        from abstractvision.errors import GenerationCancelledError
        from abstractvision.types import VideoGenerationRequest

        backend = self._backend()
        for when in ("decode", 1):
            cancel = threading.Event()
            _FakeMiniMaxH3.during_step = lambda step, when=when, cancel=cancel: cancel.set() if step == when else None
            with self.assertRaises(GenerationCancelledError):
                backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="low", cancel_event=cancel))
            self.assertFalse(backend.has_loaded_model())
        # A batch item is a run like any other.
        cancel = threading.Event()
        _FakeMiniMaxH3.during_step = lambda step: cancel.set() if step == 1 else None
        with self.assertRaises(GenerationCancelledError):
            VisionManager(backend=backend).generate_video_batch("A fox.", count=2, memory_mode="low", cancel_event=cancel)
        self.assertFalse(backend.has_loaded_model())
        _FakeMiniMaxH3.during_step = None
        backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="low"))
        self.assertEqual(len(_FakeMiniMaxH3.instances), 4)
        self.assertEqual([model.transformer_loads for model in _FakeMiniMaxH3.instances], [1] * 4)

    def test_mlx_gen_memory_errors_become_the_out_of_memory_error_naming_memory_mode(self):
        from abstractvision.errors import GpuOutOfMemoryError
        from abstractvision.types import VideoGenerationRequest

        for where in ("init", "run"):
            with self.subTest(where=where):
                _FakeMiniMaxH3.instances = []
                _FakeMiniMaxH3.memory_error = where
                backend = self._backend()
                with self.assertRaisesRegex(GpuOutOfMemoryError, "memory_mode='default'.*low.*lowest"):
                    backend.generate_video(VideoGenerationRequest(prompt="fox"))
        _FakeMiniMaxH3.memory_error = "run"
        with self.assertRaisesRegex(GpuOutOfMemoryError, "memory_mode='lowest'") as caught:
            self._backend().generate_video(VideoGenerationRequest(prompt="fox", memory_mode="lowest"))
        self.assertIsInstance(caught.exception.__cause__, MemoryError)

    # --- quantize, version floor ---------------------------------------------------------------

    def test_a_boolean_or_non_integral_quantize_is_a_value_error_on_minimax_h3(self):
        from abstractvision.types import VideoGenerationRequest

        for path in (self.package, self.source):
            for value in (True, False, 8.5):
                with self.subTest(path=path.name, quantize=value):
                    backend = self._backend(path=path)
                    with self.assertRaisesRegex(ValueError, "quantize must be an integer"):
                        backend.generate_video(VideoGenerationRequest(prompt="fox", quantize=value))
                    with self.assertRaisesRegex(ValueError, "quantize must be an integer"):
                        backend.preload(quantize=value)
                    self.assertFalse(backend.has_loaded_model())

    def test_quantize_bits_without_rows_lists_no_width_minimax_h3_cannot_take(self):
        self.caps = SimpleNamespace(capabilities=())
        source = self._backend(path=self.source)
        self.assertNotIn(3, source.quantize_bits())
        self.assertNotIn(6, source.quantize_bits())
        self.assertTrue(set(source.quantize_bits()) <= {4, 8})

    def test_the_catalog_states_the_package_floor_for_minimax_h3(self):
        # One floor: the install requirement (mlx-gen>=0.40.3) is what MiniMax-H3 needs at run time.
        from abstractvision.backends import mflux
        from abstractvision.model_capabilities import VisionModelCapabilitiesRegistry

        self.assertEqual(mflux.MLX_GEN_VERSION_FLOOR, "0.40.3")
        spec = VisionModelCapabilitiesRegistry().get("MiniMaxAI/MiniMax-H3")
        for task in ("text_to_video", "image_to_video"):
            self.assertEqual(spec.tasks[task].requires["min_runtime_version"], mflux.MLX_GEN_VERSION_FLOOR)


    # --- the backend class: one rule, one validation, structured refusals, preload -------------

    def test_memory_modes_flags_and_route_tasks_come_from_the_module_rule(self):
        from abstractvision.backends import mflux

        backend = self._backend()
        with patch.object(mflux, "mlx_gen_memory_modes", return_value=("default",)) as modes:
            self.assertEqual(backend.supported_memory_modes(), ("default",))
        self.assertEqual(modes.call_args.kwargs["family"], "minimax-h3")
        streamed_low = mflux.MlxGenMemoryFlags(low_ram=True, stream_blocks=True, engine_mode="low_ram")
        with patch.dict(mflux.MLX_GEN_MEMORY_MODE_FLAGS, {"low": streamed_low}):
            self.assertTrue(backend._memory_plan("low").stream_blocks)
        with patch.object(mflux, "mlx_gen_route_task", return_value="renamed_task"):
            self.assertEqual(set(backend._route_capabilities_for_model("MiniMaxAI/MiniMax-H3")), {"renamed_task"})

    def test_every_run_records_its_memory_mode_default_included(self):
        from abstractvision.types import VideoGenerationRequest

        asset = self._backend().generate_video(VideoGenerationRequest(prompt="fox"))
        self.assertEqual(
            {key: asset.metadata[key] for key in ("memory_mode", "low_ram", "stream_blocks")},
            {"memory_mode": "default", "low_ram": False, "stream_blocks": False},
        )

    def test_refusals_about_one_field_name_it_and_what_it_accepts(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import LoRAAdapterSpec, VideoGenerationRequest

        lora = self.tmp / "motion.safetensors"
        lora.write_bytes(b"x")
        cases = [
            (self._backend(), VideoGenerationRequest(prompt="fox", quantize=8), "quantize", ()),
            (self._backend(path=self.source), VideoGenerationRequest(prompt="fox", quantize=6), "quantize", (4, 8)),
            (self._backend(), VideoGenerationRequest(prompt="fox", width=650, height=352), "width", None),
            (self._backend(), VideoGenerationRequest(prompt="fox", num_frames=2000), "num_frames", (124, 141, 158)),
            (self._backend(), VideoGenerationRequest(prompt="fox", steps=0), "steps", None),
            (self._backend(), VideoGenerationRequest(prompt="fox", fps=30), "fps", (24.0,)),
            (self._backend(), VideoGenerationRequest(prompt="fox", negative_prompt="blur"), "negative_prompt", ()),
            (
                self._backend(),
                VideoGenerationRequest(
                    prompt="fox", lora_adapters=(LoRAAdapterSpec(source=str(lora), target_role="text_encoder"),)
                ),
                "lora_adapters",
                None,
            ),
            # One refusal about several fields names none.
            (self._backend(), VideoGenerationRequest(prompt="fox", fps=30, negative_prompt="blur"), None, None),
        ]
        for backend, request, field, supported in cases:
            with self.subTest(field=field, request=request):
                with self.assertRaises(CapabilityNotSupportedError) as caught:
                    backend.generate_video(request)
                self.assertEqual(caught.exception.field, field)
                if supported is not None:
                    self.assertEqual(tuple(caught.exception.supported)[: len(supported)], supported)
                self.assertNotIn(str(self.tmp), str(caught.exception))
        self.assertEqual(_FakeMiniMaxH3.instances, [])

    def test_a_request_is_normalized_once_on_the_runtime_thread(self):
        import threading

        from abstractvision import VisionManager
        from abstractvision.backends.mflux import MFluxVisionBackend

        backend = self._backend()
        threads, resolved, decoded = [], [], []
        normalize = MFluxVisionBackend._normalize_video
        resolve = MFluxVisionBackend.resolve_memory_request

        def counting_normalize(self, request):
            threads.append(threading.current_thread().name)
            return normalize(self, request)

        def counting_resolve(self, request, *, task=None):
            resolved.append(task)
            return resolve(self, request, task=task)

        from abstractvision.backends import mflux

        checked = mflux._checked_keyframe_size
        with patch.object(MFluxVisionBackend, "_normalize_video", counting_normalize), patch.object(
            MFluxVisionBackend, "resolve_memory_request", counting_resolve
        ), patch.object(mflux, "_checked_keyframe_size", side_effect=lambda image: decoded.append(1) or checked(image)):
            vm = VisionManager(backend=backend)
            vm.generate_video("A fox.")
            vm.image_to_video(PNG_1X1, prompt="The fox turns.")
        # VisionManager's normalize call and the entry point's are one pass, on the backend's thread.
        self.assertEqual(threads, ["abstractvision-mflux-runtime"] * 2)
        self.assertEqual(resolved, ["text_to_video", "image_to_video"])
        self.assertEqual(len(decoded), 1)

    def test_the_configured_model_is_resolved_once(self):
        from abstractvision.backends import mflux
        from abstractvision.types import VideoGenerationRequest

        backend = self._backend()
        scans = []
        discover = mflux.discover_cached_mflux_models
        with patch.object(mflux, "discover_cached_mflux_models", side_effect=lambda **kw: scans.append(1) or discover(**kw)):
            for _ in range(3):
                backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="low"))
        self.assertEqual(len(scans), 1)

    def test_preload_takes_the_request_options_checks_them_first_and_the_first_request_reuses_it(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        backend = self._backend()
        backend.preload(memory_mode="lowest")
        self.assertIs(_FakeMiniMaxH3.instances[0].init_kwargs.get("stream_blocks"), True)
        backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="lowest"))
        self.assertEqual(len(_FakeMiniMaxH3.instances), 1)
        for options, field in (({"quantize": 8}, "quantize"), ({"memory_mode": "tiny"}, None)):
            with self.subTest(options=options):
                fresh = self._backend()
                with self.assertRaises((CapabilityNotSupportedError, ValueError)) as caught:
                    fresh.preload(**options)
                self.assertEqual(getattr(caught.exception, "field", None), field)
                self.assertFalse(fresh.has_loaded_model())
        self.assertEqual(len(_FakeMiniMaxH3.instances), 1)
        source = self._backend(path=self.source)
        source.preload(quantize=8)
        self.assertEqual(_FakeMiniMaxH3.instances[-1].init_kwargs["quantize"], 8)
        source.generate_video(VideoGenerationRequest(prompt="fox", quantize=8))
        self.assertEqual(len(_FakeMiniMaxH3.instances), 2)

    def test_plugin_load_resident_model_passes_memory_mode_and_quantize(self):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        owner = SimpleNamespace(
            config={
                "vision_backend": "mflux",
                "vision_mflux_model": str(self.package),
                "vision_mlx_gen_base_model": "minimax-h3-turbo-544p",
            }
        )
        with patch.dict("os.environ", {"ABSTRACTVISION_MEMORY_MODE": ""}, clear=True):
            cap = _AbstractVisionCapability(owner)
            cap.load_resident_model({"memory_mode": "lowest"})
        self.assertIs(_FakeMiniMaxH3.instances[0].init_kwargs.get("stream_blocks"), True)

    def test_a_backend_without_load_time_quantize_refuses_it_at_preload(self):
        from abstractvision.backends.base_backend import VisionBackend
        from abstractvision.errors import CapabilityNotSupportedError

        class _Plain(VisionBackend):
            def generate_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def edit_image(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_angles(self, request):  # pragma: no cover
                raise NotImplementedError

            def generate_video(self, request):  # pragma: no cover
                raise NotImplementedError

            def image_to_video(self, request):  # pragma: no cover
                raise NotImplementedError

        _Plain().preload()
        with self.assertRaises(CapabilityNotSupportedError) as caught:
            _Plain().preload(quantize=8)
        self.assertEqual(caught.exception.field, "quantize")

    def test_an_image_task_on_minimax_h3_is_refused_before_anything_is_read(self):
        from abstractvision.backends.mflux import MFluxVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageEditRequest, ImageGenerationRequest

        backend = self._backend()
        with patch.object(MFluxVisionBackend, "_resolve_model", side_effect=AssertionError("resolved")):
            with self.assertRaisesRegex(CapabilityNotSupportedError, "no text_to_image route"):
                backend.generate_image(ImageGenerationRequest(prompt="fox"))
            with self.assertRaisesRegex(CapabilityNotSupportedError, "no image_to_image route"):
                backend.edit_image(ImageEditRequest(prompt="fox", image=PNG_1X1))
        self.assertEqual(_FakeMiniMaxH3.instances, [])

    def test_the_default_upscale_quantize_follows_the_prepared_package_check(self):
        from abstractvision.backends import mflux
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        backend = MFluxVisionBackend(config=MFluxBackendConfig(model="seedvr2-3b", base_model="seedvr2-3b"))
        model_path = str(self.tmp)
        with patch.object(MFluxVisionBackend, "_find_configured_model", return_value=(model_path, "seedvr2-3b", 8)):
            for prepared, expected in ((True, None), (False, 8)):
                backend.unload()
                with patch.object(mflux, "_is_prepared_mlx_gen_package", return_value=prepared) as check:
                    self.assertEqual(backend._default_upscale_quantize(), expected)
                    self.assertEqual(backend._default_upscale_quantize(), expected)
                # Read once per resolution.
                check.assert_called_once_with(model_path, "seedvr2-3b")

    def test_a_cancelled_run_hands_the_caller_no_frame_of_the_run(self):
        from abstractvision.errors import GenerationCancelledError
        from abstractvision.types import VideoGenerationRequest

        cancel = threading.Event()
        _FakeMiniMaxH3.during_step = lambda step: cancel.set() if step == 1 else None
        backend = self._backend()
        with self.assertRaises(GenerationCancelledError) as caught:
            backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="low", cancel_event=cancel))
        error = caught.exception
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)
        frames = []
        tb = error.__traceback__
        while tb is not None:
            frames.append(tb.tb_frame.f_code.co_name)
            tb = tb.tb_next
        self.assertFalse({"_generate_h3_run", "_generate_video_run", "_memory_session"} & set(frames), frames)


if __name__ == "__main__":
    unittest.main()
