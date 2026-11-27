"""FLUX.2 masked edits (0.3.38 step 2): MLX-Gen's flux2.inpaint route (`Flux2KleinInpaint`), with
the stand-in engine classes of `_mlx_gen_route_fixtures`.
"""

from __future__ import annotations

import dataclasses
import threading
import unittest
from unittest.mock import patch

from _mlx_gen_route_fixtures import (
    ADAPTER,
    EVENTS,
    FLUX2,
    FLUX2_BASE,
    SOURCE,
    Flux2KleinInpaint,
    RouteCase,
    needs_planner,
    png,
)
from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401

MASK = png(64, 48, color=(255, 255, 255))


def _engine_with(*, base_model=None, supports_lora=None):
    """mlx-gen's own rows and planner, the rows asked with `base_model`, the plan's LoRA flag replaced."""
    from mflux.task_inference import get_model_capabilities, resolve_generation_plan

    def rows(**kwargs):
        return get_model_capabilities(**kwargs, **({"base_model": base_model} if base_model else {}))

    def plan(**kwargs):
        selected = resolve_generation_plan(**kwargs)
        return selected if supports_lora is None else dataclasses.replace(selected, supports_lora=supports_lora)

    return patch("abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference", return_value=(rows, plan))


@needs_planner
class TestMaskedEditCalls(RouteCase):
    def test_flux2_masked_edit_call(self):
        _backend, asset = self.edit(FLUX2, mask=MASK, seed=9, steps=6, negative_prompt="blur")
        name, kwargs, files = self.call()
        self.assertEqual(name, "Flux2KleinInpaint")
        self.assertEqual(self.inits(), ["Flux2KleinInpaint"])
        image_path, mask_path = kwargs.pop("image_path"), kwargs.pop("mask_path")
        # Guidance unset: not passed, the class applies its own default. The distilled row takes no
        # negative prompt.
        self.assertEqual(
            kwargs, {"seed": 9, "prompt": "a red kite", "num_inference_steps": 6, "width": 64, "height": 48}
        )
        self.assertEqual((files["image_path"]["bytes"], files["mask_path"]["bytes"]), (SOURCE, MASK))
        self.assertEqual(files["image_path"]["dir"], files["mask_path"]["dir"])
        self.assertTrue(image_path.endswith("source.png") and mask_path.endswith("mask.png"))
        self.assertTrue(asset.metadata["mask"])
        self.assert_metadata_clean(asset, files["image_path"]["dir"])
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_a_set_guidance_is_passed(self):
        self.edit(FLUX2, mask=MASK, seed=1, guidance_scale=1.0)
        _name, kwargs, _files = self.call()
        self.assertEqual(kwargs["guidance"], 1.0)

    def test_the_base_model_takes_the_negative_prompt_and_its_guidance_unset(self):
        self.edit(FLUX2_BASE, mask=MASK, seed=1, negative_prompt="blur")
        name, kwargs, _files = self.call()
        self.assertEqual(name, "Flux2KleinInpaint")
        self.assertEqual(kwargs["negative_prompt"], "blur")
        self.assertNotIn("guidance", kwargs)
        self.assertEqual(kwargs["num_inference_steps"], 50)

    def test_every_optional_value_unset(self):
        _backend, asset = self.edit(FLUX2, mask=MASK)
        name, kwargs, _files = self.call()
        self.assertEqual(name, "Flux2KleinInpaint")
        self.assertIsInstance(kwargs.pop("seed"), int)
        kwargs.pop("image_path")
        kwargs.pop("mask_path")
        self.assertEqual(kwargs, {"prompt": "a red kite", "num_inference_steps": 4, "width": 64, "height": 48})
        self.assertEqual(asset.metadata["memory_mode"], "default")

    def test_adapters_reach_the_instance_where_the_row_takes_them(self):
        from abstractvision.types import LoRAAdapterSpec

        self.edit(FLUX2, mask=MASK, lora_adapters=(LoRAAdapterSpec(source=ADAPTER, scale=0.8),))
        (init,) = [event[2] for event in EVENTS if event[0] == "init"]
        self.assertEqual((init["lora_paths"], init["lora_scales"]), ([ADAPTER], [0.8]))


@needs_planner
class TestMemory(RouteCase):
    def test_a_default_run_keeps_the_inpaint_instance_and_a_plain_edit_replaces_it(self):
        from abstractvision.types import ImageEditRequest

        backend, _asset = self.edit(FLUX2, mask=MASK)
        self.assertTrue(backend.has_loaded_model())
        self.assertEqual(backend._model_key.variant, "flux2-inpaint")
        held = backend._model
        backend.edit_image(ImageEditRequest(prompt="again", image=SOURCE))
        self.assertEqual(self.inits(), ["Flux2KleinInpaint", "Flux2KleinEdit"])
        self.assertIsNot(backend._model, held)
        self.assertEqual(backend._model_key.variant, "flux2-edit")

    def test_low_ram_frees_the_transformer_and_drops_the_model(self):
        backend, asset = self.edit(FLUX2, mask=MASK, memory_mode="low")
        self.assertIn(("register", "MemorySaver", False), EVENTS)
        self.assertTrue(asset.metadata["low_ram"])
        self.assertFalse(backend.has_loaded_model())


@needs_planner
class TestRefusals(RouteCase):
    def test_a_mask_on_an_explicit_base_identity(self):
        from abstractvision.errors import CapabilityNotSupportedError

        with _engine_with(base_model="flux2-klein-9b"):
            self.assert_refused(FLUX2, CapabilityNotSupportedError, field="mask", supported=(), mask=MASK)

    def test_a_mask_that_fails_to_decode(self):
        for mask in (b"not a picture", b""):
            with self.subTest(mask=mask):
                error = self.assert_refused(FLUX2, ValueError, mask=mask)
                self.assertTrue(str(error).startswith("mask "), error)

    def test_a_mask_with_a_strength(self):
        self.assert_refused(FLUX2, ValueError, mask=MASK, image_strength=0.5)

    def test_adapters_where_the_row_takes_none(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import LoRAAdapterSpec

        with _engine_with(supports_lora=False):
            self.assert_refused(
                FLUX2,
                CapabilityNotSupportedError,
                field="lora_adapters",
                supported=(),
                mask=MASK,
                lora_adapters=(LoRAAdapterSpec(source=ADAPTER),),
            )

    def test_lowest_memory_mode(self):
        from abstractvision.errors import CapabilityNotSupportedError

        self.assert_refused(
            FLUX2, CapabilityNotSupportedError, field="memory_mode", supported=("default", "low"), mask=MASK,
            memory_mode="lowest",
        )


@needs_planner
class TestRunDirectory(RouteCase):
    def test_it_is_gone_after_an_error_and_a_cancel(self):
        from abstractvision.errors import GenerationCancelledError

        Flux2KleinInpaint.fail_with = RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            self.edit(FLUX2, mask=MASK)
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        Flux2KleinInpaint.fail_with = None
        event = threading.Event()
        Flux2KleinInpaint.during_call = event.set
        with self.assertRaises(GenerationCancelledError):
            self.edit(FLUX2, mask=MASK, cancel_event=event)
        self.assertEqual(list(self.tmp_root.iterdir()), [])


@needs_planner
class TestPublicationAndFacade(RouteCase):
    def test_flux2_capabilities_and_catalog(self):
        for model in (FLUX2, FLUX2_BASE):
            with self.subTest(model=model):
                backend = self.backend(model)
                self.assertTrue(backend.get_capabilities().supports_mask)
                params = self.listed(backend, model).raw["task_specs"]["image_to_image"]["params"]
                self.assertIn("mask", params)

    def test_the_facade_and_the_manager_run_the_masked_edit(self):
        from abstractvision import VisionManager

        cap = self.capability(self.backend(FLUX2), FLUX2)
        cap.i2i("a red kite", image=SOURCE, mask=MASK, seed=2)
        self.assertEqual(self.call()[0], "Flux2KleinInpaint")
        EVENTS.clear()
        VisionManager(backend=self.backend(FLUX2)).edit_image("a red kite", image=SOURCE, mask=MASK)
        self.assertEqual(self.call()[0], "Flux2KleinInpaint")


@needs_planner
class TestChangedAgainst0337(RouteCase):
    def test_changed_12_a_mask_on_flux2_runs_the_inpaint_class(self):
        self.edit(FLUX2, mask=MASK)
        self.assertEqual(self.call()[0], "Flux2KleinInpaint")

    def test_changed_14_supports_mask_on_flux2_klein(self):
        self.assertTrue(self.backend("AbstractFramework/flux.2-klein-4b-8bit").get_capabilities().supports_mask)


if __name__ == "__main__":
    unittest.main()
