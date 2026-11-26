"""Z-Image edits and masked edits (0.3.38 step 7): MLX-Gen's z-image.latent and z-image.inpaint
routes (`ZImage`), with the stand-in engine classes of `_mlx_gen_route_fixtures`.
"""

from __future__ import annotations

import threading
import unittest

from _mlx_gen_route_fixtures import (
    EVENTS,
    SOURCE,
    Z_IMAGE,
    Z_IMAGE_TURBO,
    RouteCase,
    ZImage,
    needs_planner,
    png,
)
from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401

MASK = png(64, 48, color=(255, 255, 255))


@needs_planner
class TestCalls(RouteCase):
    def test_the_latent_call(self):
        _backend, asset = self.edit(Z_IMAGE, image_strength=0.6, seed=3, steps=12, negative_prompt="blur")
        name, kwargs, files = self.call()
        self.assertEqual(name, "ZImage")
        kwargs.pop("image_path")
        self.assertEqual(
            kwargs,
            {
                "seed": 3,
                "prompt": "a red kite",
                "num_inference_steps": 12,
                "height": 48,
                "width": 64,
                "guidance": 3.5,
                "negative_prompt": "blur",
                "image_strength": 0.6,
            },
        )
        self.assert_metadata_clean(asset, files["image_path"]["dir"])
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_the_masked_call(self):
        _backend, asset = self.edit(Z_IMAGE_TURBO, mask=MASK, seed=4, negative_prompt="blur")
        name, kwargs, files = self.call()
        self.assertEqual(name, "ZImage")
        self.assertEqual((files["image_path"]["bytes"], files["mask_path"]["bytes"]), (SOURCE, MASK))
        self.assertEqual(files["image_path"]["dir"], files["mask_path"]["dir"])
        kwargs.pop("image_path")
        kwargs.pop("mask_path")
        # Turbo: guidance forced to 0.0; the negative prompt is passed, as the route's row takes one (at
        # guidance 0.0 the engine runs no negative branch); a mask replaces the strength.
        self.assertEqual(
            kwargs,
            {
                "seed": 4,
                "prompt": "a red kite",
                "num_inference_steps": 9,
                "height": 48,
                "width": 64,
                "guidance": 0.0,
                "negative_prompt": "blur",
            },
        )
        self.assertTrue(asset.metadata["mask"])
        self.assertNotIn("image_strength", asset.metadata)
        self.assert_metadata_clean(asset, files["image_path"]["dir"])

    def test_every_optional_value_unset_runs_latent_at_0_4(self):
        self.edit(Z_IMAGE_TURBO)
        name, kwargs, _files = self.call()
        self.assertEqual(name, "ZImage")
        kwargs.pop("image_path")
        self.assertIsInstance(kwargs.pop("seed"), int)
        self.assertEqual(
            kwargs,
            {"prompt": "a red kite", "num_inference_steps": 9, "height": 48, "width": 64, "guidance": 0.0,
             "image_strength": 0.4},
        )

    def test_low_ram_and_the_run_directory(self):
        from abstractvision.errors import GenerationCancelledError

        backend, asset = self.edit(Z_IMAGE_TURBO, mask=MASK, memory_mode="low")
        self.assertIn(("register", "MemorySaver", False), EVENTS)
        self.assertFalse(backend.has_loaded_model())
        ZImage.fail_with = RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            self.edit(Z_IMAGE_TURBO, mask=MASK)
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        ZImage.fail_with = None
        event = threading.Event()
        ZImage.during_call = event.set
        with self.assertRaises(GenerationCancelledError):
            self.edit(Z_IMAGE_TURBO, cancel_event=event)
        self.assertEqual(list(self.tmp_root.iterdir()), [])


@needs_planner
class TestRefusals(RouteCase):
    def test_a_mask_on_base_z_image(self):
        from abstractvision.errors import CapabilityNotSupportedError

        self.assert_refused(Z_IMAGE, CapabilityNotSupportedError, field="mask", supported=(), mask=MASK)

    def test_a_strength_outside_0_1_and_with_a_mask(self):
        from abstractvision.errors import CapabilityNotSupportedError

        self.assert_refused(
            Z_IMAGE_TURBO, CapabilityNotSupportedError, field="image_strength", supported=None, image_strength=1.5
        )
        self.assert_refused(Z_IMAGE_TURBO, ValueError, image_strength=0.5, mask=MASK)

    def test_references_outpaint_and_reframe(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for field, value in (
            ("reference_images", [MASK]),
            ("outpaint_padding", "0,10%,0,10%"),
            ("reframe_padding", "0,10%,0,10%"),
        ):
            with self.subTest(field=field):
                self.assert_refused(Z_IMAGE_TURBO, CapabilityNotSupportedError, field=field, supported=(), **{field: value})


@needs_planner
class TestPublication(RouteCase):
    def test_capabilities_parameters_and_catalog(self):
        turbo, base = self.backend(Z_IMAGE_TURBO), self.backend(Z_IMAGE)
        self.assertEqual(turbo.get_capabilities().supported_tasks, ["image_to_image", "text_to_image"])
        self.assertTrue(turbo.get_capabilities().supports_mask)
        self.assertFalse(base.get_capabilities().supports_mask)
        model = self.listed(turbo, Z_IMAGE_TURBO)
        self.assertEqual(model.raw["parameter_defaults"]["image_strength"], 0.4)
        self.assertEqual(
            model.raw["parameter_constraints"]["image_strength"], {"min": 0.0, "min_exclusive": True, "max": 1.0}
        )
        params = model.raw["task_specs"]["image_to_image"]["params"]
        self.assertEqual(
            list(params),
            ["prompt", "image", "image_strength", "mask", "negative_prompt", "steps", "guidance_scale", "seed"],
        )
        self.assertIs(params["negative_prompt"]["supported"], False)
        self.assertEqual(params["guidance_scale"]["const"], 0.0)
        base_params = self.listed(base, Z_IMAGE).raw["task_specs"]["image_to_image"]["params"]
        self.assertEqual(
            list(base_params), ["prompt", "image", "image_strength", "negative_prompt", "steps", "guidance_scale", "seed"]
        )


class TestCatalogListings(unittest.TestCase):
    def test_stable_diffusion_cpp_lists_the_catalog_task(self):
        # As it lists every catalog task of a model (CHANGELOG, Changed 16).
        from abstractvision.backends.stable_diffusion_cpp import (
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
        )

        for model_id in ("Tongyi-MAI/Z-Image", "Tongyi-MAI/Z-Image-Turbo"):
            backend = StableDiffusionCppVisionBackend(
                config=StableDiffusionCppBackendConfig(
                    sd_cli_path="sd-cli", diffusion_model="model.gguf", capabilities_model_id=model_id
                )
            )
            self.assertEqual(backend.get_capabilities().supported_tasks, ["image_to_image", "text_to_image"])


@needs_planner
class TestChangedAgainst0337(RouteCase):
    def test_changed_11_a_plain_edit_on_z_image_runs_latent(self):
        self.edit(Z_IMAGE)
        name, kwargs, _files = self.call()
        self.assertEqual((name, kwargs["image_strength"]), ("ZImage", 0.4))

    def test_changed_14_z_image_is_published_for_image_to_image(self):
        backend = self.backend(Z_IMAGE_TURBO)
        self.assertTrue(backend.get_capabilities().supports_mask)
        listed = [m.id for m in backend.list_provider_models(task="image_to_image")]
        self.assertIn(Z_IMAGE_TURBO, listed)
        self.assertIn(Z_IMAGE, listed)


if __name__ == "__main__":
    unittest.main()
