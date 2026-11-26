"""Wan 2.1 VACE 1.3B video-to-video (0.3.38 step 6): `WanVace` with a mask, a masked-region mode,
reference images and a conditioning scale, with the stand-in engine classes of
`_mlx_gen_route_fixtures`.
"""

from __future__ import annotations

import os
import threading
import unittest

from _mlx_gen_route_fixtures import (
    CLIP,
    EVENTS,
    WAN_VACE,
    RouteCase,
    VideoUtil,
    WanVace,
    needs_planner,
    png,
    strings,
)
from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401

MASK = png(64, 48, color=(255, 255, 255))
BLACK = png(64, 48, color=(0, 0, 0))
REF_1 = png(32, 32, color=(200, 20, 20))
REF_2 = png(40, 24, color=(20, 200, 20))
SMALL = {"width": 480, "height": 240, "num_frames": 17, "fps": 16, "steps": 8}


class _Case(RouteCase):
    def v2v(self, **fields):
        from abstractvision.types import VideoToVideoRequest

        backend = self.backend(WAN_VACE)
        return backend, backend.video_to_video(VideoToVideoRequest(prompt="a lantern on a jetty", video=CLIP, **fields))

    def assert_v2v_refused(self, error, *, field="unset", supported="unset", **fields):
        from abstractvision.types import VideoToVideoRequest

        backend = self.backend(WAN_VACE)
        with self.assertRaises(error) as caught:
            backend.video_to_video(VideoToVideoRequest(prompt="a lantern on a jetty", video=CLIP, **fields))
        if field != "unset":
            self.assertEqual(caught.exception.field, field)
        if supported != "unset":
            self.assertEqual(caught.exception.supported, supported)
        self.assert_nothing_loaded(backend)
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        return caught.exception


@needs_planner
class TestCalls(_Case):
    def test_the_vace_call(self):
        _backend, asset = self.v2v(
            mask=MASK, masked_region_mode="repaint", reference_images=[REF_1, REF_2], conditioning_scale=0.7,
            seed=42, negative_prompt="blur", **SMALL,
        )
        name, kwargs, files = self.call()
        self.assertEqual((name, self.inits()), ("WanVace", ["WanVace"]))
        self.assertEqual([item["bytes"] for item in files["reference_image_paths"]], [REF_1, REF_2])
        run_dir = files["video_path"]["dir"]
        self.assertEqual({item["dir"] for item in files["reference_image_paths"]} | {files["video_mask_path"]["dir"]}, {run_dir})
        for key in ("video_path", "video_mask_path", "reference_image_paths"):
            kwargs.pop(key)
        # Never guidance_2: VACE has one transformer.
        self.assertEqual(
            kwargs,
            {
                "seed": 42,
                "prompt": "a lantern on a jetty",
                "num_inference_steps": 8,
                "height": 240,
                "width": 480,
                "num_frames": 17,
                "fps": 16,
                "guidance": 5.0,
                "negative_prompt": "blur",
                "masked_region_mode": "repaint",
                "conditioning_scale": 0.7,
            },
        )
        for text in strings(asset.metadata):
            self.assertNotIn(run_dir, text)
            self.assertFalse(os.path.isabs(text), text)
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_every_optional_value_unset(self):
        self.v2v()
        name, kwargs, _files = self.call()
        self.assertEqual(name, "WanVace")
        kwargs.pop("video_path")
        self.assertIsInstance(kwargs.pop("seed"), int)
        self.assertEqual(
            kwargs,
            {
                "prompt": "a lantern on a jetty",
                "num_inference_steps": 30,
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "fps": 16,
                "guidance": 5.0,
            },
        )

    def test_low_ram_releases_the_denoiser_before_the_decode_and_drops_the_model(self):
        backend, asset = self.v2v(memory_mode="low", **SMALL)
        _name, kwargs, _files = self.call()
        for name in ("release_denoisers_before_decode", "clear_cache_each_step", "clear_cache_each_transformer_block"):
            self.assertIs(kwargs[name], True, name)
        self.assertNotIn("release_inactive_denoiser", kwargs)
        self.assertTrue(asset.metadata["low_ram"])
        self.assertFalse(backend.has_loaded_model())

    def test_a_preload_loads_wan_vace_with_no_warm_up(self):
        backend = self.backend(WAN_VACE)
        backend.preload()
        self.assertEqual(self.inits(), ["WanVace"])
        self.assertEqual([event for event in EVENTS if event[0] == "call"], [])
        self.assertTrue(backend.has_loaded_model())

    def test_an_engine_error_and_a_cancel_remove_the_run_directory(self):
        from abstractvision.errors import GenerationCancelledError

        WanVace.fail_with = RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            self.v2v(reference_images=[REF_1], **SMALL)
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        WanVace.fail_with = None
        event = threading.Event()
        WanVace.during_call = event.set
        with self.assertRaises(GenerationCancelledError):
            self.v2v(reference_images=[REF_1], cancel_event=event, **SMALL)
        self.assertEqual(list(self.tmp_root.iterdir()), [])


@needs_planner
class TestRefusals(_Case):
    def test_the_a14b_fields(self):
        from abstractvision.errors import CapabilityNotSupportedError

        self.assert_v2v_refused(CapabilityNotSupportedError, field="video_strength", supported=(), video_strength=0.5)
        self.assert_v2v_refused(CapabilityNotSupportedError, field="guidance_2", supported=(), guidance_2=3.0)

    def test_the_masked_region_mode(self):
        from abstractvision.errors import CapabilityNotSupportedError

        self.assert_v2v_refused(
            CapabilityNotSupportedError, field="masked_region_mode", supported=("generate", "repaint"),
            masked_region_mode="inpaint", mask=MASK,
        )
        self.assert_v2v_refused(
            CapabilityNotSupportedError, field="masked_region_mode", supported=(), masked_region_mode="generate"
        )

    def test_a_conditioning_scale_that_is_not_a_finite_number(self):
        for value in ("strong", float("nan"), True):
            with self.subTest(value=value):
                self.assert_v2v_refused(ValueError, conditioning_scale=value)

    def test_a_reference_image_that_fails_to_decode(self):
        error = self.assert_v2v_refused(ValueError, reference_images=[b"not a picture"])
        self.assertTrue(str(error).startswith("reference_images "), error)

    def test_the_rows_of_step_5(self):
        from abstractvision.errors import CapabilityNotSupportedError

        self.assert_v2v_refused(ValueError, mask=BLACK)
        self.assert_v2v_refused(CapabilityNotSupportedError, field="width", **{**SMALL, "width": 470})
        self.assert_v2v_refused(CapabilityNotSupportedError, field="solver", supported=(), extra={"solver": "euler"})
        VideoUtil.duration_seconds = 0.5
        self.assert_v2v_refused(CapabilityNotSupportedError, field="num_frames", supported=None, **SMALL)


@needs_planner
class TestPublicationAndFacade(_Case):
    def test_capabilities_parameters_and_catalog(self):
        backend = self.backend(WAN_VACE)
        self.assertEqual(backend.get_capabilities().supported_tasks, ["video_to_video"])
        (model,) = [m for m in backend.list_provider_models(task="video_to_video") if m.id == WAN_VACE]
        defaults, constraints = model.raw["parameter_defaults"], model.raw["parameter_constraints"]
        self.assertEqual(
            (defaults["steps"], defaults["guidance_scale"], defaults["width"], defaults["height"], defaults["num_frames"],
             defaults["fps"]),
            (30, 5.0, 832, 480, 81, 16),
        )
        self.assertNotIn("guidance_2", defaults)
        self.assertEqual(constraints["width"], {"multiple_of": 16})
        params = model.raw["task_specs"]["video_to_video"]["params"]
        self.assertEqual(
            list(params),
            ["prompt", "video", "mask", "reference_images", "masked_region_mode", "conditioning_scale",
             "negative_prompt", "width", "height", "fps", "num_frames", "steps", "guidance_scale", "flow_shift", "seed"],
        )
        self.assertEqual(params["masked_region_mode"]["enum"], ["generate", "repaint"])
        self.assertEqual(params["masked_region_mode"]["default"], "generate")
        self.assertEqual(params["conditioning_scale"]["default"], 1.0)
        self.assertIn("no published maximum", params["reference_images"]["description"])

    def test_the_plugin_v2v(self):
        cap = self.capability(self.backend(WAN_VACE), WAN_VACE)
        cap.v2v("a lantern", CLIP, mask=MASK, masked_region_mode="generate", reference_images=[REF_1], **SMALL)
        name, kwargs, files = self.call()
        self.assertEqual((name, kwargs["masked_region_mode"]), ("WanVace", "generate"))
        self.assertEqual([item["bytes"] for item in files["reference_image_paths"]], [REF_1])


@needs_planner
class TestChangedAgainst0337(_Case):
    def test_changed_14_vace_is_published_for_video_to_video(self):
        backend = self.backend(WAN_VACE)
        self.assertIn("video_to_video", backend.get_capabilities().supported_tasks)
        self.assertIn(WAN_VACE, [m.id for m in backend.list_provider_models(task="video_to_video")])


if __name__ == "__main__":
    unittest.main()
