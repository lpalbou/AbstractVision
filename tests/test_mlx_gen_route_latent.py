"""Latent image-to-image with a strength (0.3.38 step 1): FLUX.2, base Qwen-Image and ERNIE through
MLX-Gen's planner, with stand-in engine classes (`_mlx_gen_route_fixtures`) that record what they are
built with and called with.
"""

from __future__ import annotations

import threading
import unittest
from pathlib import Path

from _mlx_gen_route_fixtures import (
    ADAPTER,
    ERNIE,
    EVENTS,
    FIBO,
    FIBO_EDIT,
    FIBO_LITE,
    FLUX2,
    FLUX2_BASE,
    QWEN,
    QWEN_2512,
    QWEN_EDIT,
    SOURCE,
    QwenImage,
    RouteCase,
    needs_planner,
)
from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401


@needs_planner
class TestLatentCalls(RouteCase):
    def test_flux2_latent_call(self):
        _backend, asset = self.edit(FLUX2, image_strength=0.6, seed=7, steps=6, guidance_scale=3.0, negative_prompt="blur")
        name, kwargs, files = self.call()
        # The distilled weights: guidance forced to 1.0, and the row takes no negative prompt.
        self.assertEqual(name, "Flux2Klein")
        self.assertEqual(self.inits(), ["Flux2Klein"])  # Flux2KleinEdit is not built
        image_path = kwargs.pop("image_path")
        self.assertEqual(
            kwargs,
            {
                "seed": 7,
                "prompt": "a red kite",
                "num_inference_steps": 6,
                "height": 48,
                "width": 64,
                "guidance": 1.0,
                "image_strength": 0.6,
            },
        )
        self.assertEqual(files["image_path"]["bytes"], SOURCE)
        self.assertTrue(Path(files["image_path"]["dir"]).name.startswith("abstractvision-mlx-gen-"))
        self.assertEqual(Path(image_path).parent, Path(files["image_path"]["dir"]))
        self.assertEqual(asset.metadata["image_strength"], 0.6)
        self.assert_metadata_clean(asset, files["image_path"]["dir"])
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_flux2_base_latent_takes_the_negative_prompt_its_row_takes(self):
        self.edit(FLUX2_BASE, image_strength=0.5, seed=1, negative_prompt="blur")
        name, kwargs, _files = self.call()
        self.assertEqual(name, "Flux2Klein")
        kwargs.pop("image_path")
        # Guidance unset: the base model's 1.5, as on its other 0.3.37 routes.
        self.assertEqual(
            kwargs,
            {
                "seed": 1,
                "prompt": "a red kite",
                "num_inference_steps": 50,
                "height": 48,
                "width": 64,
                "guidance": 1.5,
                "negative_prompt": "blur",
                "image_strength": 0.5,
            },
        )

    def test_qwen_latent_call(self):
        self.edit(QWEN, image_strength=0.7, seed=3, steps=12, guidance_scale=2.0, negative_prompt="blur")
        name, kwargs, _files = self.call()
        self.assertEqual(name, "QwenImage")
        self.assertEqual(self.inits(), ["QwenImage"])
        kwargs.pop("image_path")
        self.assertEqual(
            kwargs,
            {
                "seed": 3,
                "prompt": "a red kite",
                "num_inference_steps": 12,
                "height": 48,
                "width": 64,
                "guidance": 2.0,
                "negative_prompt": "blur",
                "image_strength": 0.7,
            },
        )

    def test_every_optional_value_unset_on_base_qwen_runs_latent_at_0_4(self):
        _backend, asset = self.edit(QWEN)
        name, kwargs, files = self.call()
        self.assertEqual(name, "QwenImage")
        kwargs.pop("image_path")
        self.assertIsInstance(kwargs.pop("seed"), int)
        self.assertEqual(
            kwargs,
            {
                "prompt": "a red kite",
                "num_inference_steps": 20,
                "height": 48,
                "width": 64,
                "guidance": 3.5,
                "image_strength": 0.4,
            },
        )
        self.assertEqual((asset.metadata["memory_mode"], asset.metadata["low_ram"]), ("default", False))
        self.assert_metadata_clean(asset, files["image_path"]["dir"])

    def test_ernie_typed_strength_makes_the_0_3_37_call(self):
        self.edit(ERNIE, extra={"strength": 0.35}, seed=5)
        _name, before, _files = self.call()
        EVENTS.clear()
        self.edit(ERNIE, image_strength=0.35, seed=5)
        name, after, _files = self.call()
        self.assertEqual(name, "ErnieImageTurbo")
        before.pop("image_path")
        after.pop("image_path")
        self.assertEqual(after, before)
        self.assertEqual(after["image_strength"], 0.35)

    def test_extra_image_strength_and_the_typed_field_make_the_same_call(self):
        self.edit(QWEN, extra={"image_strength": 0.55}, seed=2)
        _name, from_extra, _files = self.call()
        EVENTS.clear()
        self.edit(QWEN, image_strength=0.55, seed=2)
        _name, typed, _files = self.call()
        from_extra.pop("image_path")
        typed.pop("image_path")
        self.assertEqual(from_extra, typed)

    def test_low_ram_runs_with_the_memory_saver_and_drops_the_model(self):
        backend, asset = self.edit(QWEN, image_strength=0.5, memory_mode="low")
        self.assertEqual(asset.metadata["low_ram"], True)
        self.assertFalse(backend.has_loaded_model())

    def test_a_default_run_keeps_the_instance_and_reuses_it(self):
        from abstractvision.types import ImageEditRequest

        backend, _asset = self.edit(QWEN, image_strength=0.5)
        self.assertTrue(backend.has_loaded_model())
        backend.edit_image(ImageEditRequest(prompt="again", image=SOURCE, image_strength=0.5))
        self.assertEqual(self.inits(), ["QwenImage"])


@needs_planner
class TestLegacyStrength(RouteCase):
    """`extra["strength"]`, the 0.3.37 spelling: its 0.3.37 rule wherever 0.3.37 had the route."""

    def test_strength_alone_on_base_qwen_runs_at_its_value(self):
        self.edit(QWEN, extra={"strength": 0.65})
        _name, kwargs, _files = self.call()
        self.assertEqual(kwargs["image_strength"], 0.65)

    def test_strength_out_of_range_is_clamped_as_in_0_3_37(self):
        for model, name in ((QWEN, "QwenImage"), (ERNIE, "ErnieImageTurbo")):
            with self.subTest(model=model):
                EVENTS.clear()
                self.edit(model, extra={"strength": 1.7})
                called, kwargs, _files = self.call()
                self.assertEqual((called, kwargs["image_strength"]), (name, 1.0))

    def test_strength_on_a_flux2_edit_is_ignored_as_in_0_3_37(self):
        self.edit(FLUX2, extra={"strength": 0.75})
        name, kwargs, _files = self.call()
        self.assertEqual(name, "Flux2KleinEdit")
        self.assertNotIn("image_strength", kwargs)

    def test_the_readme_qwen_image_edit_example_keeps_working(self):
        # README: `abstractvision i2i --model AbstractFramework/qwen-image-edit-2511-4bit --image ./input.png
        # "replace the background with a clean white studio setup" --steps 20 --guidance-scale 2.5 --strength 0.75`
        self.edit(
            QWEN_EDIT,
            prompt="replace the background with a clean white studio setup",
            steps=20,
            guidance_scale=2.5,
            seed=11,
            extra={"strength": 0.75},
        )
        name, kwargs, _files = self.call()
        self.assertEqual(name, "QwenImageEdit")
        self.assertEqual(len(kwargs.pop("image_paths")), 1)
        self.assertEqual(
            kwargs,
            {
                "seed": 11,
                "prompt": "replace the background with a clean white studio setup",
                "num_inference_steps": 20,
                "width": 64,
                "height": 48,
                "guidance": 2.5,
            },
        )

    def test_strength_with_image_strength_raises(self):
        self.assert_refused(QWEN, ValueError, image_strength=0.5, extra={"strength": 0.5})


@needs_planner
class TestRefusals(RouteCase):
    def test_strength_outside_0_1(self):
        for value in (0, 0.0, -0.2, 1.5):
            with self.subTest(value=value):
                error = self.assert_refused(
                    FLUX2, Exception, field="image_strength", supported=None, image_strength=value
                )
                from abstractvision.errors import CapabilityNotSupportedError

                self.assertIsInstance(error, CapabilityNotSupportedError)
                self.assertIn("(0, 1]", str(error))

    def test_strength_not_a_finite_number(self):
        for value in ("strong", float("nan"), float("inf"), True):
            with self.subTest(value=value):
                self.assert_refused(QWEN, ValueError, image_strength=value)

    def test_strength_on_a_model_without_a_latent_route(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for model in (FIBO, FIBO_LITE, FIBO_EDIT, QWEN_EDIT):
            with self.subTest(model=model):
                self.assert_refused(
                    model, CapabilityNotSupportedError, field="image_strength", supported=(), image_strength=0.5
                )

    def test_strength_with_a_mask(self):
        self.assert_refused(FLUX2, ValueError, image_strength=0.5, mask=SOURCE)

    def test_strength_with_adapters_on_a_route_without_lora(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import LoRAAdapterSpec

        adapters = (LoRAAdapterSpec(source=ADAPTER),)
        for model, fields in ((QWEN_2512, {"image_strength": 0.5}), (QWEN_2512, {}), (FLUX2, {"image_strength": 0.5})):
            with self.subTest(model=model, fields=fields):
                self.assert_refused(
                    model,
                    CapabilityNotSupportedError,
                    field="lora_adapters",
                    supported=(),
                    lora_adapters=adapters,
                    **fields,
                )

    def test_lowest_memory_mode(self):
        from abstractvision.errors import CapabilityNotSupportedError

        self.assert_refused(
            QWEN,
            CapabilityNotSupportedError,
            field="memory_mode",
            supported=("default", "low"),
            image_strength=0.5,
            memory_mode="lowest",
        )

    def test_a_cancel_set_before_the_run_loads_nothing(self):
        from abstractvision.errors import GenerationCancelledError

        event = threading.Event()
        event.set()
        self.assert_refused(QWEN, GenerationCancelledError, image_strength=0.5, cancel_event=event)

    def test_a_typed_strength_on_the_other_backends(self):
        self.assert_other_backends_refuse(image_strength=0.5)


@needs_planner
class TestRunDirectory(RouteCase):
    def test_it_is_gone_after_an_error(self):
        QwenImage.fail_with = RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            self.edit(QWEN, image_strength=0.5)
        self.assertTrue(self.call()[2]["image_path"]["exists"])
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_it_is_gone_after_a_cancel(self):
        from abstractvision.errors import GenerationCancelledError

        event = threading.Event()
        QwenImage.during_call = event.set
        with self.assertRaises(GenerationCancelledError):
            self.edit(QWEN, image_strength=0.5, cancel_event=event)
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_a_refused_request_writes_nothing(self):
        from abstractvision.errors import CapabilityNotSupportedError

        self.assert_refused(FIBO_EDIT, CapabilityNotSupportedError, image_strength=0.5)


@needs_planner
class TestPublication(RouteCase):
    def test_base_qwen_capabilities_listing_and_catalog(self):
        backend = self.backend(QWEN)
        caps = backend.get_capabilities()
        self.assertEqual(caps.supported_tasks, ["image_to_image", "text_to_image"])
        self.assertFalse(caps.supports_mask)
        model = self.listed(backend, QWEN)
        self.assertEqual(model.raw["parameter_defaults"]["image_strength"], 0.4)
        self.assertEqual(
            model.raw["parameter_constraints"]["image_strength"], {"min": 0.0, "min_exclusive": True, "max": 1.0}
        )
        spec = model.raw["task_specs"]["image_to_image"]
        self.assertEqual(
            list(spec["params"]),
            ["prompt", "image", "image_strength", "negative_prompt", "steps", "guidance_scale", "seed"],
        )
        self.assertEqual(spec["params"]["image_strength"]["default"], 0.4)
        self.assertEqual(spec["requires"], {"backend": "mlx-gen"})

    def test_a_model_without_a_served_latent_route_publishes_no_strength(self):
        model = self.listed(self.backend(QWEN_EDIT), QWEN_EDIT)
        self.assertNotIn("image_strength", model.raw["parameter_defaults"])
        self.assertNotIn("image_strength", model.raw["parameter_constraints"])


class TestCatalog(unittest.TestCase):
    def test_the_strength_parameter_is_named_image_strength(self):
        from abstractvision.model_capabilities import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        for model_id in (
            "black-forest-labs/FLUX.2-klein-4B",
            "black-forest-labs/FLUX.2-klein-9B",
            "black-forest-labs/FLUX.2-klein-base-4B",
            "black-forest-labs/FLUX.2-klein-base-9B",
            "baidu/ERNIE-Image-Turbo",
            "Qwen/Qwen-Image",
            "Qwen/Qwen-Image-2512",
        ):
            with self.subTest(model=model_id):
                params = reg.get(model_id).tasks["image_to_image"].params
                self.assertFalse(params["image_strength"]["required"])
        ernie = reg.get("baidu/ERNIE-Image-Turbo").tasks["image_to_image"].params
        # The 0.3.37 keyword, described as docs/api.md does: never a second spelling of image_strength.
        self.assertIn("never selects a route", ernie["strength"]["description"])
        self.assertNotIn("alias", ernie["strength"]["description"].lower())


@needs_planner
class TestFacade(RouteCase):
    def test_i2i_takes_image_strength_and_t2i_refuses_it(self):
        from abstractvision.errors import CapabilityNotSupportedError

        cap = self.capability(self.backend(QWEN), QWEN)
        cap.i2i("a red kite", image=SOURCE, image_strength=0.45, seed=4)
        _name, kwargs, _files = self.call()
        self.assertEqual(kwargs["image_strength"], 0.45)
        for method, args in (("t2i", ("a red kite",)), ("t2v", ("a red kite",)), ("i2v", (SOURCE,))):
            with self.subTest(method=method):
                with self.assertRaises(CapabilityNotSupportedError) as caught:
                    getattr(cap, method)(*args, image_strength=0.45)
                self.assertEqual((caught.exception.field, caught.exception.supported), ("image_strength", ()))


@needs_planner
class TestChangedAgainst0337(RouteCase):
    """One test per line of "Changed against 0.3.37" that this step realises (CHANGELOG, Unreleased)."""

    def test_changed_1_strength_out_of_range_is_refused(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for fields in ({"image_strength": 1.5}, {"extra": {"image_strength": 1.5}}, {"extra": {"image_strength": -0.1}}):
            with self.subTest(fields=fields):
                self.assert_refused(QWEN, CapabilityNotSupportedError, field="image_strength", supported=None, **fields)

    def test_changed_2_a_non_numeric_extra_strength_raises(self):
        self.assert_refused(ERNIE, ValueError, extra={"image_strength": "strong"})

    def test_changed_3_extra_image_strength_on_flux2_selects_the_latent_route(self):
        self.edit(FLUX2, extra={"image_strength": 0.6})
        name, kwargs, _files = self.call()
        self.assertEqual((name, kwargs["image_strength"]), ("Flux2Klein", 0.6))

    def test_changed_4_extra_image_strength_on_edit_models_is_refused(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for model in (QWEN_EDIT, FIBO_EDIT):
            with self.subTest(model=model):
                self.assert_refused(
                    model, CapabilityNotSupportedError, field="image_strength", supported=(), extra={"image_strength": 0.5}
                )

    def test_changed_5_extra_image_strength_on_fibo_is_refused_before_any_load(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for model in (FIBO, FIBO_LITE):
            with self.subTest(model=model):
                self.assert_refused(
                    model, CapabilityNotSupportedError, field="image_strength", supported=(), extra={"image_strength": 0.5}
                )

    def test_changed_8_the_field_and_extra_disagree(self):
        self.assert_refused(QWEN, ValueError, image_strength=0.5, extra={"image_strength": 0.6})
        self.assert_refused(QWEN, ValueError, image_strength=0.5, extra={"strength": 0.5})

    def test_changed_9_keywords_on_the_other_backends(self):
        # The typed field is refused by name; `extra` and `strength=` travel as 0.3.37 sent them.
        self.assert_other_backends_refuse(image_strength=0.5)

    def test_changed_11_a_plain_edit_on_base_qwen_runs_latent(self):
        self.edit(QWEN, seed=1)
        name, kwargs, _files = self.call()
        self.assertEqual((name, kwargs["image_strength"]), ("QwenImage", 0.4))

    def test_changed_14_base_qwen_is_published_for_image_to_image(self):
        backend = self.backend(QWEN_2512)
        self.assertIn("image_to_image", backend.get_capabilities().supported_tasks)
        listed = [model.id for model in backend.list_provider_models(task="image_to_image")]
        self.assertIn(QWEN_2512, listed)
        self.assertIn(QWEN, listed)


if __name__ == "__main__":
    unittest.main()
