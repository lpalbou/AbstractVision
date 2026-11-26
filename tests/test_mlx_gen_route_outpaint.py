"""Outpaint and reframe (0.3.38 step 4): MLX-Gen's flux2.outpaint, flux2.reframe, qwen.outpaint and
qwen.reframe routes through the engine's own canvas sessions (`mflux.outpaint`, no weights), with the
stand-in engine classes of `_mlx_gen_route_fixtures`.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from _mlx_gen_route_fixtures import (
    ERNIE,
    EVENTS,
    FIBO_EDIT,
    FLUX2,
    FLUX2_BASE,
    QWEN_EDIT,
    Flux2KleinOutpaint,
    RouteCase,
    needs_planner,
    png,
)
from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401

# The engine's validated outpaint source size; a one-axis box runs one pass, a deep corner two.
SOURCE = png(432, 240)
ONE_PASS = "0,25%,0,25%"
TWO_PASSES = "35%,35%,0,0"
REF = png(32, 32, color=(200, 20, 20))


def _engine_canvas(model, padding, **options):
    """The canvas MLX-Gen's own session builds for the same request: size and digest."""
    import hashlib

    from mflux.outpaint import prepare_outpaint

    with tempfile.TemporaryDirectory() as td:
        source = Path(td) / "source.png"
        source.write_bytes(SOURCE)
        session = prepare_outpaint(source_image=str(source), padding=padding, model=model, workspace=td, **options)
        return (session.width, session.height), hashlib.sha256(Path(session.canvas.canvas_path).read_bytes()).hexdigest()


@needs_planner
class TestCalls(RouteCase):
    def outpaint(self, model, **fields):
        from abstractvision.types import ImageEditRequest

        backend = self.backend(model)
        return backend, backend.edit_image(ImageEditRequest(prompt="a wide harbour", image=SOURCE, **fields))

    def test_flux2_outpaint_call(self):
        _backend, asset = self.outpaint(FLUX2, outpaint_padding=ONE_PASS, seed=5, steps=6, negative_prompt="blur")
        name, kwargs, files = self.call()
        self.assertEqual(name, "Flux2KleinOutpaint")
        self.assertEqual(self.inits(), ["Flux2KleinOutpaint"])
        kwargs.pop("canvas")
        # The session owns the canvas; guidance unset stays unset; the distilled row takes no negative prompt.
        self.assertEqual(kwargs, {"seed": 5, "prompt": "a wide harbour", "num_inference_steps": 6})
        size, digest = _engine_canvas(FLUX2, ONE_PASS)
        self.assertEqual((files["canvas"]["size"], files["canvas"]["digest"]), (size, digest))
        self.assertTrue(Path(files["canvas"]["dir"]).name.startswith("abstractvision-mlx-gen-"))
        self.assertEqual((asset.metadata["width"], asset.metadata["height"]), size)
        self.assertIsInstance(asset.metadata["outpaint_notice"], str)
        self.assertIsInstance(asset.metadata["outpaint_warnings"], list)
        self.assertEqual(asset.metadata["mlx_gen"]["outpaint_passes"], 1)
        self.assertNotIn("outpaint_source_path", asset.metadata["mlx_gen"])
        self.assert_metadata_clean(asset, files["canvas"]["dir"])
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_a_fill_and_its_color_reach_the_engine_canvas(self):
        self.outpaint(FLUX2, outpaint_padding=ONE_PASS, outpaint_fill="solid", outpaint_fill_color="#ff0000", seed=1)
        _name, _kwargs, files = self.call()
        expected = _engine_canvas(FLUX2, ONE_PASS, fill="solid", fill_color=(255, 0, 0), fill_color_explicit=True)
        self.assertEqual((files["canvas"]["size"], files["canvas"]["digest"]), expected)

    def test_the_base_model_takes_the_negative_prompt_and_a_set_guidance(self):
        self.outpaint(FLUX2_BASE, outpaint_padding=ONE_PASS, seed=1, negative_prompt="blur", guidance_scale=3.0)
        _name, kwargs, _files = self.call()
        self.assertEqual((kwargs["negative_prompt"], kwargs["guidance"]), ("blur", 3.0))

    def test_flux2_reframe_call(self):
        _backend, asset = self.outpaint(FLUX2, reframe_padding="0,10%,0,10%", seed=2, steps=4)
        name, kwargs, files = self.call()
        self.assertEqual(name, "Flux2KleinEdit")
        (canvas,) = kwargs.pop("image_paths")
        self.assertTrue(canvas.endswith("reframe_canvas.png"))
        self.assertEqual(
            kwargs,
            {
                "seed": 2,
                "prompt": "a wide harbour",
                "num_inference_steps": 4,
                "width": 528,
                "height": 240,
                "canvas_policy": "exact-resize",
            },
        )
        self.assertNotIn("reframe_source_path", asset.metadata["mlx_gen"])
        self.assert_metadata_clean(asset, files["image_paths"][0]["dir"])

    def test_qwen_outpaint_and_reframe_calls(self):
        self.outpaint(QWEN_EDIT, outpaint_padding=ONE_PASS, seed=3, negative_prompt="blur")
        name, kwargs, files = self.call()
        self.assertEqual(name, "QwenImageEdit")
        self.assertEqual(
            sorted(kwargs),
            sorted(["image_path", "image_paths", "width", "height", "canvas_policy", "mask_path", "seed", "prompt",
                    "num_inference_steps", "negative_prompt"]),
        )
        self.assertEqual((kwargs["width"], kwargs["height"], kwargs["num_inference_steps"]), (656, 240, 20))
        self.assertEqual(files["image_path"]["bytes"], SOURCE)
        EVENTS.clear()
        self.outpaint(QWEN_EDIT, reframe_padding="0,10%,0,10%", seed=3)
        name, kwargs, _files = self.call()
        self.assertEqual(name, "QwenImageEdit")
        self.assertEqual(sorted(kwargs), sorted(["image_paths", "width", "height", "canvas_policy", "seed", "prompt",
                                                 "num_inference_steps"]))

    def test_every_optional_value_unset(self):
        _backend, asset = self.outpaint(FLUX2, outpaint_padding=ONE_PASS)
        name, kwargs, files = self.call()
        self.assertEqual(name, "Flux2KleinOutpaint")
        kwargs.pop("canvas")
        self.assertIsInstance(kwargs.pop("seed"), int)
        self.assertEqual(kwargs, {"prompt": "a wide harbour", "num_inference_steps": 4})
        # The row's default fill (auto) builds the canvas.
        self.assertEqual(files["canvas"]["digest"], _engine_canvas(FLUX2, ONE_PASS)[1])
        self.assertEqual(asset.metadata["memory_mode"], "default")

    def test_a_two_pass_outpaint_counts_its_steps_across_the_passes(self):
        Flux2KleinOutpaint.emit_steps = True
        seen = []
        _backend, asset = self.outpaint(
            FLUX2, outpaint_padding=TWO_PASSES, steps=3, extra={"on_progress": seen.append}
        )
        self.assertEqual([event[1] for event in EVENTS if event[0] == "call"], ["Flux2KleinOutpaint"] * 2)
        steps = [(event.step, event.total_steps) for event in seen]
        self.assertEqual(steps[0], (0, 6))
        self.assertEqual(steps[-1], (6, 6))
        self.assertEqual([step for step, _total in steps], sorted(step for step, _total in steps))
        self.assertEqual(asset.metadata["mlx_gen"]["outpaint_passes"], 2)

    def test_low_ram_runs_a_one_pass_outpaint(self):
        backend, asset = self.outpaint(FLUX2, outpaint_padding=ONE_PASS, memory_mode="low")
        self.assertIn(("register", "MemorySaver", False), EVENTS)
        self.assertTrue(asset.metadata["low_ram"])
        self.assertFalse(backend.has_loaded_model())


@needs_planner
class TestRefusals(RouteCase):
    def refused(self, model, error, **fields):
        from abstractvision.types import ImageEditRequest

        backend = self.backend(model)
        with self.assertRaises(error) as caught:
            backend.edit_image(ImageEditRequest(prompt="a wide harbour", image=SOURCE, **fields))
        self.assert_nothing_loaded(backend)
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        return caught.exception

    def assert_field(self, model, field, supported, **fields):
        from abstractvision.errors import CapabilityNotSupportedError

        error = self.refused(model, CapabilityNotSupportedError, **fields)
        self.assertEqual((error.field, error.supported), (field, supported))

    def test_outpaint_on_a_model_without_an_outpaint_route(self):
        for model in (ERNIE, FIBO_EDIT):
            with self.subTest(model=model):
                self.assert_field(model, "outpaint_padding", (), outpaint_padding=ONE_PASS)

    def test_reframe_on_base_klein(self):
        self.assert_field(FLUX2_BASE, "reframe_padding", (), reframe_padding="0,10%,0,10%")

    def test_a_fill_the_row_does_not_list(self):
        self.assert_field(
            FLUX2, "outpaint_fill", ("auto", "edge", "neutral", "solid", "blur"), outpaint_padding=ONE_PASS,
            outpaint_fill="stretch",
        )

    def test_a_fill_or_a_color_on_the_fixed_fill_row(self):
        self.assert_field(QWEN_EDIT, "outpaint_fill", (), outpaint_padding=ONE_PASS, outpaint_fill="edge")
        self.assert_field(QWEN_EDIT, "outpaint_fill_color", (), outpaint_padding=ONE_PASS, outpaint_fill_color="1,2,3")

    def test_a_color_without_the_solid_fill(self):
        for fill in (None, "edge"):
            with self.subTest(fill=fill):
                self.assert_field(
                    FLUX2, "outpaint_fill_color", (), outpaint_padding=ONE_PASS, outpaint_fill=fill,
                    outpaint_fill_color="1,2,3",
                )

    def test_a_fill_or_a_color_without_an_outpaint(self):
        self.assert_field(FLUX2, "outpaint_fill", (), outpaint_fill="solid")
        self.assert_field(FLUX2, "outpaint_fill_color", (), outpaint_fill_color="1,2,3")

    def test_a_color_the_engine_cannot_read(self):
        for color in ("red", "#ff00", "1,2,300"):
            with self.subTest(color=color):
                error = self.refused(
                    FLUX2, ValueError, outpaint_padding=ONE_PASS, outpaint_fill="solid", outpaint_fill_color=color
                )
                self.assertTrue(str(error).startswith("outpaint_fill_color"), error)

    def test_a_padding_the_engine_rejects(self):
        for name in ("outpaint_padding", "reframe_padding"):
            for padding in ("0,0,0,0", "-5,0,0,0", "1.5%", "wide"):
                with self.subTest(field=name, padding=padding):
                    error = self.refused(FLUX2, ValueError, **{name: padding})
                    self.assertTrue(str(error).startswith(name), error)

    def test_a_canvas_size_in_extra(self):
        for key in ("width", "height"):
            with self.subTest(key=key):
                self.assert_field(FLUX2, "width", (), outpaint_padding=ONE_PASS, extra={key: 512})
                self.assert_field(FLUX2, "width", (), reframe_padding=ONE_PASS, extra={key: 512})

    def test_contradicting_options(self):
        for fields in (
            {"outpaint_padding": ONE_PASS, "reframe_padding": ONE_PASS},
            {"outpaint_padding": ONE_PASS, "mask": SOURCE},
            {"reframe_padding": ONE_PASS, "mask": SOURCE},
        ):
            with self.subTest(fields=sorted(fields)):
                self.refused(FLUX2, ValueError, **fields)

    def test_references_with_an_outpaint_or_a_reframe(self):
        for fields in ({"outpaint_padding": ONE_PASS}, {"reframe_padding": ONE_PASS}):
            for references in ({"reference_images": [REF]}, {"extra": {"images": [REF]}}):
                with self.subTest(fields=list(fields), references=references):
                    self.assert_field(FLUX2, None, None, **fields, **references)

    def test_low_ram_with_a_two_pass_outpaint(self):
        self.assert_field(FLUX2, "memory_mode", ("default",), outpaint_padding=TWO_PASSES, memory_mode="low")

    def test_lowest_memory_mode(self):
        self.assert_field(FLUX2, "memory_mode", ("default", "low"), outpaint_padding=ONE_PASS, memory_mode="lowest")


@needs_planner
class TestRunDirectory(RouteCase):
    def test_it_is_gone_after_an_error_and_a_cancel(self):
        from abstractvision.errors import GenerationCancelledError
        from abstractvision.types import ImageEditRequest

        Flux2KleinOutpaint.fail_with = RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            self.backend(FLUX2).edit_image(ImageEditRequest(prompt="p", image=SOURCE, outpaint_padding=ONE_PASS))
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        Flux2KleinOutpaint.fail_with = None
        event = threading.Event()
        Flux2KleinOutpaint.during_call = event.set
        with self.assertRaises(GenerationCancelledError):
            self.backend(FLUX2).edit_image(
                ImageEditRequest(prompt="p", image=SOURCE, outpaint_padding=TWO_PASSES, cancel_event=event)
            )
        self.assertEqual([event[1] for event in EVENTS if event[0] == "call"], ["Flux2KleinOutpaint"] * 2)
        self.assertEqual(list(self.tmp_root.iterdir()), [])


@needs_planner
class TestPublicationAndFacade(RouteCase):
    def test_the_catalog_lists_the_outpaint_and_reframe_fields(self):
        from abstractvision.model_capabilities import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        # What each model's rows take: reframe on distilled klein only, a fill on klein only.
        fields = {
            "black-forest-labs/FLUX.2-klein-4B": {"outpaint_padding", "outpaint_fill", "outpaint_fill_color", "reframe_padding"},
            "black-forest-labs/FLUX.2-klein-9B": {"outpaint_padding", "outpaint_fill", "outpaint_fill_color", "reframe_padding"},
            "black-forest-labs/FLUX.2-klein-base-4B": {"outpaint_padding", "outpaint_fill", "outpaint_fill_color"},
            "black-forest-labs/FLUX.2-klein-base-9B": {"outpaint_padding", "outpaint_fill", "outpaint_fill_color"},
            "Qwen/Qwen-Image-Edit": {"outpaint_padding", "reframe_padding"},
            "Qwen/Qwen-Image-Edit-2509": {"outpaint_padding", "reframe_padding"},
            "Qwen/Qwen-Image-Edit-2511": {"outpaint_padding", "reframe_padding"},
        }
        canvas = {"outpaint_padding", "outpaint_fill", "outpaint_fill_color", "reframe_padding"}
        for model_id, expected in fields.items():
            with self.subTest(model=model_id):
                params = reg.get(model_id).tasks["image_to_image"].params
                self.assertEqual(set(params) & canvas, expected)
                for name in expected:
                    self.assertIs(params[name]["required"], False)

    def test_the_facade_keywords(self):
        from abstractvision.errors import CapabilityNotSupportedError

        cap = self.capability(self.backend(FLUX2), FLUX2)
        cap.i2i("a wide harbour", image=SOURCE, outpaint_padding=ONE_PASS, outpaint_fill="edge", seed=1)
        self.assertEqual(self.call()[0], "Flux2KleinOutpaint")
        for name in ("outpaint_padding", "outpaint_fill", "outpaint_fill_color", "reframe_padding"):
            with self.subTest(option=name):
                with self.assertRaises(CapabilityNotSupportedError) as caught:
                    cap.t2i("a wide harbour", **{name: "10%"})
                self.assertEqual((caught.exception.field, caught.exception.supported), (name, ()))


class TestChangedAgainst0337(RouteCase):
    def test_changed_9_the_keywords_on_the_other_backends(self):
        self.assert_other_backends_refuse(
            outpaint_padding=ONE_PASS, outpaint_fill="edge", outpaint_fill_color="1,2,3", reframe_padding=ONE_PASS
        )


if __name__ == "__main__":
    unittest.main()
