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

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X9+QAAAABJRU5ErkJggg=="
)
MP4 = b"\x00\x00\x00\x18ftypmp42h3"


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
    def __init__(self, kwargs, *, mux=True):
        self.kwargs = kwargs
        self.mux = mux

    def save(self, path=None, export_json_metadata=False, overwrite=True):
        path = Path(path)
        path.write_bytes(MP4)
        metadata = {
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
                wav.write_bytes(b"RIFF....WAVE")
                metadata.update(audio_muxed=False, audio_mux_reason="boom", audio_sidecar_path=str(wav))
        if export_json_metadata:
            path.with_suffix(".json").write_text(json.dumps(metadata))
        return path


class _FakeMiniMaxH3:
    instances = []
    mux = True
    during_step = None

    def __init__(self, **kwargs):
        self.init_kwargs = dict(kwargs)
        self.block_stream = object() if kwargs.get("stream_blocks") else None
        self.generate_calls = []
        _FakeMiniMaxH3.instances.append(self)

    def generate_video(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        progress = kwargs.get("progress_callback")
        for step in range(3):
            if _FakeMiniMaxH3.during_step is not None:
                _FakeMiniMaxH3.during_step(step)
            if progress is not None:
                progress(SimpleNamespace(phase="denoise", step=step + 1, total_steps=3, total_frames=124, frame=None, task=None, timestep=None))
        return _FakeH3Video(kwargs, mux=_FakeMiniMaxH3.mux)


class _FakeModelConfig:
    @staticmethod
    def from_name(model_name, base_model=None):
        return ("config", str(model_name), base_model)


class TestMiniMaxH3Backend(unittest.TestCase):
    def setUp(self):
        _FakeMiniMaxH3.instances = []
        _FakeMiniMaxH3.mux = True
        _FakeMiniMaxH3.during_step = None
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.package = self.tmp / "minimax-h3-8bit"
        (self.package / "transformer").mkdir(parents=True)
        (self.package / "transformer" / "0.safetensors").write_bytes(b"x")
        (self.package / "README.md").write_text("---\nlibrary_name: mlx-gen\n---\n")
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
                "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                side_effect=lambda: (lambda **kwargs: self.caps, None),
            )
        )
        self.limits = []
        self.stack.enter_context(
            patch(
                "abstractvision.backends.mflux._set_mlx_cache_limit",
                side_effect=lambda value: self.limits.append(value) or 555,
            )
        )
        self.stack.enter_context(patch("abstractvision.backends.mflux._apply_mlx_gen_default_cache_limit"))
        self.stack.enter_context(patch("abstractvision.backends.mflux._clear_mlx_cache"))

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
        self.assertEqual(asset.metadata["audio"]["sample_rate"], 32000)
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
        self.assertTrue(caps.supports_block_streaming)
        self.assertTrue(caps.generates_audio)

        low = backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="low"))
        self.assertIs(_FakeMiniMaxH3.instances[0].generate_calls[0]["release_transformer"], True)
        self.assertEqual(self.limits[0], LOW_RAM_MLX_CACHE_LIMIT_BYTES)
        # The instance stays loaded: MLX-Gen reads the transformer again on the next run.
        self.assertTrue(backend.has_loaded_model())
        self.assertEqual(low.metadata["memory_mode"], "low")

        lowest = backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="lowest"))
        streamed = _FakeMiniMaxH3.instances[-1]
        self.assertIs(streamed.init_kwargs.get("stream_blocks"), True)
        self.assertIs(streamed.generate_calls[0]["release_transformer"], True)
        self.assertTrue(lowest.metadata["stream_blocks"])

    def test_lowest_with_load_time_quantize_is_refused(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoGenerationRequest

        backend = self._backend(path=self.source)
        with self.assertRaisesRegex(CapabilityNotSupportedError, "streams the stored weights"):
            backend.generate_video(VideoGenerationRequest(prompt="fox", memory_mode="lowest", quantize=8))
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

    def test_an_unmuxed_soundtrack_is_returned_as_wav_bytes(self):
        from abstractvision.types import VideoGenerationRequest

        _FakeMiniMaxH3.mux = False
        asset = self._backend().generate_video(VideoGenerationRequest(prompt="fox"))
        self.assertFalse(asset.audio.muxed)
        self.assertEqual(asset.audio.data, b"RIFF....WAVE")
        self.assertEqual(asset.audio.mime_type, "audio/wav")
        self.assertFalse(asset.metadata["audio"]["muxed"])

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
        self.assertEqual(self.limits[-1], 555)  # the cache limit is restored
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
        self.assertEqual(seen, [1, 2, 3])
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


if __name__ == "__main__":
    unittest.main()
