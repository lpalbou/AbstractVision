"""Wan video-to-video on T2V-A14B (0.3.38 step 5): `VideoToVideoRequest`, `video_to_video` on the
backend, `VisionManager.video_to_video` and the plugin's `v2v`, with the stand-in engine classes of
`_mlx_gen_route_fixtures`.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from _mlx_gen_route_fixtures import (
    CLIP,
    EVENTS,
    FLUX2,
    WAN_I2V,
    WAN_T2V,
    WAN_TI2V,
    RouteCase,
    VideoUtil,
    Wan2_2_TI2V,
    needs_planner,
    png,
    strings,
)
from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401

MASK = png(64, 48, color=(255, 255, 255))
BLACK = png(64, 48, color=(0, 0, 0))
# A canvas Wan A14B takes (multiples of 16) and a short clip.
SMALL = {"width": 480, "height": 240, "num_frames": 17, "fps": 16, "steps": 4}


class _Case(RouteCase):
    def v2v(self, model=WAN_T2V, **fields):
        from abstractvision.types import VideoToVideoRequest

        backend = self.backend(model)
        video = fields.pop("video", CLIP)
        return backend, backend.video_to_video(VideoToVideoRequest(prompt="a fox in the snow", video=video, **fields))

    def assert_v2v_refused(self, error, *, field="unset", supported="unset", model=WAN_T2V, **fields):
        from abstractvision.types import VideoToVideoRequest

        backend = self.backend(model)
        video = fields.pop("video", CLIP)
        with self.assertRaises(error) as caught:
            backend.video_to_video(VideoToVideoRequest(prompt="a fox in the snow", video=video, **fields))
        if field != "unset":
            self.assertEqual(caught.exception.field, field)
        if supported != "unset":
            self.assertEqual(caught.exception.supported, supported)
        self.assert_nothing_loaded(backend)
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        return caught.exception


@needs_planner
class TestCalls(_Case):
    def test_the_a14b_call(self):
        _backend, asset = self.v2v(
            video_strength=0.6, mask=MASK, seed=7, guidance_scale=4.5, guidance_2=2.5, flow_shift=3.0,
            negative_prompt="blur", **SMALL,
        )
        name, kwargs, files = self.call()
        self.assertEqual(name, "Wan2_2_TI2V")
        self.assertEqual((files["video_path"]["bytes"], files["video_mask_path"]["bytes"]), (CLIP, MASK))
        self.assertTrue(Path(files["video_path"]["dir"]).name.startswith("abstractvision-mlx-gen-"))
        kwargs.pop("video_path")
        kwargs.pop("video_mask_path")
        self.assertEqual(
            kwargs,
            {
                "seed": 7,
                "prompt": "a fox in the snow",
                "num_inference_steps": 4,
                "height": 240,
                "width": 480,
                "num_frames": 17,
                "fps": 16,
                "guidance": 4.5,
                "negative_prompt": "blur",
                "video_strength": 0.6,
                "guidance_2": 2.5,
                "flow_shift": 3.0,
            },
        )
        self.assertTrue(files["tiled_decode"])
        self.assertEqual((asset.metadata["task"], asset.metadata["video_strength"]), ("video_to_video", 0.6))
        self.assertEqual(asset.mime_type, "video/mp4")
        self.assertEqual(asset.data, CLIP)
        for text in strings(asset.metadata):
            self.assertNotIn(files["video_path"]["dir"], text)
            self.assertFalse(os.path.isabs(text), text)
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_every_optional_value_unset(self):
        _backend, asset = self.v2v()
        name, kwargs, _files = self.call()
        self.assertEqual(name, "Wan2_2_TI2V")
        kwargs.pop("video_path")
        self.assertIsInstance(kwargs.pop("seed"), int)
        # No negative prompt (the model's default applies), no strength (the engine's 0.8).
        self.assertEqual(
            kwargs,
            {
                "prompt": "a fox in the snow",
                "num_inference_steps": 40,
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "fps": 16,
                "guidance": 4.0,
                "guidance_2": 3.0,
            },
        )
        self.assertEqual(asset.metadata["memory_mode"], "default")

    def test_a_path_source_is_read_in_place(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "clip.mp4"
            source.write_bytes(CLIP)
            self.v2v(video=source, seed=1)
            _name, kwargs, _files = self.call()
            self.assertEqual(kwargs["video_path"], str(source))
            self.assertEqual(VideoUtil.inspected, [str(source)])

    def test_the_result_counts_the_engine_s_frames(self):
        _backend, asset = self.v2v(**{**SMALL, "num_frames": 19})
        self.assertEqual(asset.metadata["num_frames"], 17)

    def test_low_ram_releases_the_denoisers_and_drops_the_model(self):
        backend, asset = self.v2v(memory_mode="low", **SMALL)
        _name, kwargs, _files = self.call()
        for name in (
            "release_denoisers_before_decode",
            "release_inactive_denoiser",
            "clear_cache_each_step",
            "clear_cache_each_transformer_block",
        ):
            self.assertIs(kwargs[name], True, name)
        self.assertTrue(asset.metadata["low_ram"])
        self.assertFalse(backend.has_loaded_model())

    def test_vae_tiling_as_on_the_other_wan_routes(self):
        _backend, asset = self.v2v(extra={"vae_tiling": False, "max_sequence_length": 256}, **SMALL)
        _name, kwargs, files = self.call()
        self.assertFalse(files["tiled_decode"])
        self.assertEqual(kwargs["max_sequence_length"], 256)
        self.assertIs(asset.metadata["vae_tiling"], False)

    def test_progress_and_a_cancel_during_the_run(self):
        from abstractvision.errors import GenerationCancelledError

        seen = []
        self.v2v(extra={"on_progress": seen.append}, **SMALL)
        self.assertEqual([event.task for event in seen], ["video_to_video"])
        EVENTS.clear()
        event = threading.Event()
        Wan2_2_TI2V.during_call = event.set
        with self.assertRaises(GenerationCancelledError):
            self.v2v(cancel_event=event, **SMALL)
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_an_engine_error_removes_the_run_directory(self):
        Wan2_2_TI2V.fail_with = RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            self.v2v(**SMALL)
        self.assertEqual(list(self.tmp_root.iterdir()), [])


@needs_planner
class TestRefusals(_Case):
    def test_a_model_without_a_video_to_video_route(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for model in (WAN_TI2V, WAN_I2V, FLUX2):
            with self.subTest(model=model):
                self.assert_v2v_refused(CapabilityNotSupportedError, model=model)

    def test_a_strength_outside_0_1(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for value in (0, 1.5, -1):
            with self.subTest(value=value):
                self.assert_v2v_refused(
                    CapabilityNotSupportedError, field="video_strength", supported=None, video_strength=value
                )
        self.assert_v2v_refused(ValueError, video_strength="strong")

    def test_the_vace_fields_on_a14b(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for fields in (
            {"reference_images": [MASK]},
            {"masked_region_mode": "generate", "mask": MASK},
            {"conditioning_scale": 0.5},
        ):
            name = next(key for key in fields if key != "mask")
            with self.subTest(field=name):
                self.assert_v2v_refused(CapabilityNotSupportedError, field=name, supported=(), **fields)

    def test_a_mask_that_fails_to_decode_or_has_no_white_pixel(self):
        for mask in (b"not a picture", BLACK):
            with self.subTest(white=mask is not BLACK):
                error = self.assert_v2v_refused(ValueError, mask=mask)
                self.assertTrue(str(error).startswith("mask "), error)

    def test_a_clip_too_short_for_the_frames(self):
        from abstractvision.errors import CapabilityNotSupportedError

        VideoUtil.duration_seconds = 0.5
        self.assert_v2v_refused(CapabilityNotSupportedError, field="num_frames", supported=None, **SMALL)
        # Without a duration, the frame count decides.
        VideoUtil.duration_seconds, VideoUtil.frame_count = None, 9
        self.assert_v2v_refused(CapabilityNotSupportedError, field="num_frames", supported=None, **SMALL)

    def test_a_canvas_off_the_multiple_of_16(self):
        from abstractvision.errors import CapabilityNotSupportedError

        self.assert_v2v_refused(CapabilityNotSupportedError, field="width", **{**SMALL, "width": 470})

    def test_an_extra_key_outside_the_allowed_set(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for key in ("solver", "last_image", "context_images"):
            with self.subTest(key=key):
                self.assert_v2v_refused(CapabilityNotSupportedError, field=key, supported=(), extra={key: 1})

    def test_a_source_that_is_not_a_clip(self):
        self.assert_v2v_refused(FileNotFoundError, video=Path("/Users/someone/missing/clip.mp4"))
        self.assert_v2v_refused(ValueError, video=b"")
        self.assert_v2v_refused(ValueError, video=42)

    def test_lowest_memory_mode(self):
        from abstractvision.errors import CapabilityNotSupportedError

        self.assert_v2v_refused(CapabilityNotSupportedError, field="memory_mode", memory_mode="lowest")

    def test_a_cancel_set_before_the_run(self):
        from abstractvision.errors import GenerationCancelledError

        event = threading.Event()
        event.set()
        self.assert_v2v_refused(GenerationCancelledError, cancel_event=event)

    def test_the_other_backends_refuse_the_task(self):
        from abstractvision.backends.stable_diffusion_cpp import (
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
        )
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoToVideoRequest

        backend = StableDiffusionCppVisionBackend(
            config=StableDiffusionCppBackendConfig(sd_cli_path="sd-cli", diffusion_model="model.gguf")
        )
        with self.assertRaisesRegex(CapabilityNotSupportedError, "does not support video_to_video"):
            backend.video_to_video(VideoToVideoRequest(prompt="p", video=CLIP))
        self.assertIs(backend.normalize_video_to_video_request(VideoToVideoRequest(prompt="p", video=CLIP)).video, CLIP)


@needs_planner
class TestPublication(_Case):
    def test_capabilities_and_the_catalog(self):
        backend = self.backend(WAN_T2V)
        self.assertIn("video_to_video", backend.get_capabilities().supported_tasks)
        (model,) = [m for m in backend.list_provider_models(task="video_to_video") if m.id == WAN_T2V]
        params = model.raw["task_specs"]["video_to_video"]["params"]
        self.assertEqual(
            list(params),
            ["prompt", "video", "video_strength", "mask", "negative_prompt", "width", "height", "fps",
             "num_frames", "steps", "guidance_scale", "guidance_2", "flow_shift", "seed"],
        )
        self.assertEqual(model.raw["parameter_constraints"]["num_frames"], {"format": "4n+1"})

    @needs_planner
    def test_the_strength_default_and_range(self):
        backend = self.backend(WAN_T2V)
        (model,) = [m for m in backend.list_provider_models(task="video_to_video") if m.id == WAN_T2V]
        self.assertEqual(model.raw["parameter_defaults"]["video_strength"], 0.8)
        self.assertEqual(
            model.raw["parameter_constraints"]["video_strength"], {"min": 0.0, "min_exclusive": True, "max": 1.0}
        )

    def test_stable_diffusion_cpp_lists_the_catalog_task(self):
        # As it lists every catalog task of a model (CHANGELOG, Changed 16).
        from abstractvision.backends.stable_diffusion_cpp import (
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
        )

        backend = StableDiffusionCppVisionBackend(
            config=StableDiffusionCppBackendConfig(
                sd_cli_path="sd-cli", diffusion_model="model.gguf", capabilities_model_id="Wan-AI/Wan2.2-T2V-A14B"
            )
        )
        self.assertEqual(backend.get_capabilities().supported_tasks, ["text_to_video", "video_to_video"])


@needs_planner
class TestManagerAndFacade(_Case):
    def test_the_manager_stores_a_video(self):
        from abstractvision import VisionManager

        stored = []

        class _Store:
            def store_bytes(self, data, *, content_type, metadata, tags):
                stored.append((content_type, tags))
                return {"$artifact": "a1"}

        VisionManager(backend=self.backend(WAN_T2V), store=_Store()).video_to_video("a fox", CLIP, **SMALL)
        self.assertEqual(
            stored, [("video/mp4", {"kind": "generated_media", "modality": "video", "task": "video_to_video"})]
        )

    def test_the_plugin_v2v(self):
        from abstractvision.errors import CapabilityNotSupportedError

        cap = self.capability(self.backend(WAN_T2V), WAN_T2V)
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "clip.mp4"
            source.write_bytes(CLIP)
            out = cap.v2v("a fox", str(source), mask=MASK, video_strength=0.5, **SMALL)
            _name, kwargs, files = self.call()
        self.assertEqual(out, CLIP)
        self.assertEqual(kwargs["video_path"], str(source))
        self.assertEqual((files["video_mask_path"]["bytes"], kwargs["video_strength"]), (MASK, 0.5))
        EVENTS.clear()
        cap.v2v("a fox", CLIP, **SMALL)
        self.assertEqual(self.call()[2]["video_path"]["bytes"], CLIP)
        with self.assertRaises(FileNotFoundError):
            cap.v2v("a fox", "/Users/someone/missing/clip.mp4")
        with self.assertRaises(CapabilityNotSupportedError) as caught:
            cap.v2v("a fox", CLIP, soundscape="wind")
        self.assertEqual(caught.exception.field, "soundscape")
        loaded = cap.list_loaded_models({"task": "v2v"})
        self.assertEqual([item["model"] for item in loaded], [WAN_T2V])

    def test_the_residency_task_names(self):
        from abstractvision.integrations.abstractcore_plugin import _normalize_residency_task

        for name in ("video_to_video", "video-to-video", "v2v"):
            self.assertEqual(_normalize_residency_task(name), "video_to_video")


@needs_planner
class TestChangedAgainst0337(_Case):
    def test_changed_14_t2v_a14b_is_published_for_video_to_video(self):
        backend = self.backend(WAN_T2V)
        self.assertEqual(backend.get_capabilities().supported_tasks, ["text_to_video", "video_to_video"])
        self.assertIn(WAN_T2V, [m.id for m in backend.list_provider_models(task="video_to_video")])
        catalog = json.loads(
            (Path(__file__).resolve().parents[1] / "src/abstractvision/assets/vision_model_capabilities.json").read_text()
        )
        self.assertIn("video_to_video", catalog["tasks"])


if __name__ == "__main__":
    unittest.main()
