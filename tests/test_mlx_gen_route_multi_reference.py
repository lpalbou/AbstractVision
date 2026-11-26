"""Multi-reference edits (0.3.38 step 3): `ImageEditRequest.reference_images` on MLX-Gen's
flux2.multi-reference and qwen.multi-reference routes, with the stand-in engine classes of
`_mlx_gen_route_fixtures`.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from _mlx_gen_route_fixtures import (
    ADAPTER,
    ERNIE,
    EVENTS,
    FLUX2,
    QWEN,
    QWEN_EDIT,
    QWEN_EDIT_2509,
    QWEN_EDIT_PLAIN,
    SOURCE,
    Flux2KleinEdit,
    RouteCase,
    needs_planner,
    png,
)
from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401

REF_1 = png(32, 32, color=(200, 20, 20))
REF_2 = png(80, 40, color=(20, 200, 20))


@needs_planner
class TestMultiReferenceCalls(RouteCase):
    def test_flux2_multi_reference_call(self):
        _backend, asset = self.edit(FLUX2, reference_images=[REF_1, REF_2], seed=4, steps=6)
        name, kwargs, files = self.call()
        self.assertEqual(name, "Flux2KleinEdit")
        self.assertEqual([item["bytes"] for item in files["image_paths"]], [SOURCE, REF_1, REF_2])
        kwargs.pop("image_paths")
        # The first image decides the canvas.
        self.assertEqual(
            kwargs,
            {"seed": 4, "prompt": "a red kite", "num_inference_steps": 6, "width": 64, "height": 48, "guidance": 1.0},
        )
        self.assertEqual((asset.metadata["reference_image_count"], asset.metadata["edit_mode"]), (3, "multi_reference"))
        self.assert_metadata_clean(asset, tempfile.gettempdir())
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_qwen_edit_plus_multi_reference_call(self):
        for model in (QWEN_EDIT, QWEN_EDIT_2509):
            with self.subTest(model=model):
                EVENTS.clear()
                self.edit(model, reference_images=(REF_1,), seed=4)
                name, kwargs, files = self.call()
                self.assertEqual(name, "QwenImageEdit")
                self.assertEqual([item["bytes"] for item in files["image_paths"]], [SOURCE, REF_1])
                kwargs.pop("image_paths")
                self.assertEqual(
                    kwargs,
                    {"seed": 4, "prompt": "a red kite", "num_inference_steps": 20, "width": 64, "height": 48, "guidance": 2.5},
                )

    def test_every_optional_value_unset(self):
        self.edit(FLUX2, reference_images=[REF_1])
        name, kwargs, _files = self.call()
        self.assertEqual(name, "Flux2KleinEdit")
        self.assertIsInstance(kwargs.pop("seed"), int)
        self.assertEqual(len(kwargs.pop("image_paths")), 2)
        self.assertEqual(kwargs, {"prompt": "a red kite", "num_inference_steps": 4, "width": 64, "height": 48, "guidance": 1.0})

    def test_the_0_3_37_flux2_reference_test_with_the_typed_field(self):
        # test_mflux_backend.py::test_flux2_edit_accepts_additional_reference_images_and_progress, typed.
        seen = []
        _backend, asset = self.edit(
            FLUX2,
            prompt="combine references",
            reference_images=[png(2, 2, color=(0, 0, 255))],
            extra={"on_progress": seen.append},
            guidance_scale=1.0,
            seed=123,
        )
        self.assertEqual(self.inits(), ["Flux2KleinEdit"])
        _name, kwargs, _files = self.call()
        self.assertEqual(len(kwargs["image_paths"]), 2)
        self.assertEqual((asset.metadata["reference_image_count"], asset.metadata["edit_mode"]), (2, "multi_reference"))
        self.assertEqual([event.task for event in seen], ["image_to_image"])
        for image_path in kwargs["image_paths"]:
            self.assertFalse(Path(image_path).exists())

    def test_a_path_item_is_read_in_place(self):
        with tempfile.TemporaryDirectory() as td:
            reference = Path(td) / "reference.png"
            reference.write_bytes(REF_1)
            self.edit(FLUX2, reference_images=[reference, str(reference)])
            _name, kwargs, _files = self.call()
            self.assertEqual(kwargs["image_paths"][1:], [str(reference), str(reference)])

    def test_extra_reference_images_and_the_typed_field_make_the_same_call(self):
        self.edit(FLUX2, extra={"reference_images": [REF_1]}, seed=3)
        _name, from_extra, _files = self.call()
        EVENTS.clear()
        self.edit(FLUX2, reference_images=[REF_1], seed=3, extra={"reference_images": (REF_1,)})
        _name, typed, _files = self.call()
        self.assertEqual(len(from_extra.pop("image_paths")), len(typed.pop("image_paths")))
        self.assertEqual(from_extra, typed)


@needs_planner
class TestRefusals(RouteCase):
    def test_a_model_without_a_multi_reference_route(self):
        from abstractvision.errors import CapabilityNotSupportedError

        # (FIBO-Edit publishes no rows: its references run the 0.3.37 branch, which refuses them as in
        # 0.3.37, test_mflux_backend.py::test_fibo_edit_rejects_additional_reference_images.)
        for model in (QWEN_EDIT_PLAIN, ERNIE, QWEN):
            with self.subTest(model=model):
                self.assert_refused(
                    model, CapabilityNotSupportedError, field="reference_images", supported=(), reference_images=[REF_1]
                )

    def test_references_with_a_mask_or_a_strength_go_to_the_planner(self):
        # Every picture reaches the planner, whatever its spelling: no route takes them with a mask or
        # a strength, so the request is refused before any load (never run with the references dropped).
        from abstractvision.errors import CapabilityNotSupportedError

        spellings = (
            {"reference_images": [REF_1]},
            {"extra": {"reference_images": [REF_1]}},
            {"extra": {"images": [REF_1]}},
            {"extra": {"additional_images": REF_1}},
            {"extra": {"image_paths": [REF_1]}},
        )
        for option in ({"mask": REF_1}, {"image_strength": 0.5}):
            for references in spellings:
                with self.subTest(option=list(option), references=references):
                    self.assert_refused(
                        FLUX2, CapabilityNotSupportedError, field=None, supported=None, **option, **references
                    )

    def test_0_3_37_extra_references_run_the_multi_reference_route(self):
        self.edit(FLUX2, extra={"images": [REF_1]}, seed=1)
        name, _kwargs, files = self.call()
        self.assertEqual(name, "Flux2KleinEdit")
        self.assertEqual([item["bytes"] for item in files["image_paths"]], [SOURCE, REF_1])

    def test_the_typed_list_with_a_0_3_37_extra_key(self):
        for key in ("images", "additional_images", "image_paths"):
            with self.subTest(key=key):
                self.assert_refused(FLUX2, ValueError, reference_images=[REF_1], extra={key: [REF_2]})

    def test_the_typed_list_and_extra_reference_images_differ(self):
        self.assert_refused(FLUX2, ValueError, reference_images=[REF_1], extra={"reference_images": [REF_2]})

    def test_adapters_on_a_row_without_lora(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import LoRAAdapterSpec

        self.assert_refused(
            QWEN_EDIT_2509,
            CapabilityNotSupportedError,
            field="lora_adapters",
            supported=(),
            reference_images=[REF_1],
            lora_adapters=(LoRAAdapterSpec(source=ADAPTER),),
        )

    def test_an_item_that_is_not_an_image(self):
        for value in ([42], [None], 42):
            with self.subTest(value=value):
                self.assert_refused(FLUX2, ValueError, reference_images=value)

    def test_one_image_given_alone_is_a_list_of_one(self):
        self.edit(FLUX2, reference_images=REF_1)
        _name, _kwargs, files = self.call()
        self.assertEqual([item["bytes"] for item in files["image_paths"]], [SOURCE, REF_1])

    def test_a_reference_that_fails_to_decode(self):
        error = self.assert_refused(FLUX2, ValueError, reference_images=[b"not a picture"])
        self.assertTrue(str(error).startswith("reference_images "), error)


@needs_planner
class TestFilesAndPublication(RouteCase):
    def test_no_file_is_left_after_an_error_or_a_cancel(self):
        from abstractvision.errors import GenerationCancelledError

        Flux2KleinEdit.fail_with = RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            self.edit(FLUX2, reference_images=[REF_1])
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        Flux2KleinEdit.fail_with = None
        event = threading.Event()
        Flux2KleinEdit.during_call = event.set
        with self.assertRaises(GenerationCancelledError):
            self.edit(FLUX2, reference_images=[REF_1], cancel_event=event)
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_the_rows_and_the_catalog(self):
        backend = self.backend(FLUX2)
        (row,) = [row for row in backend.describe_routes(task="image_to_image") if row["id"] == "flux2.multi-reference"]
        self.assertEqual((row["min_images"], row["max_images"]), (2, None))
        params = self.listed(backend, FLUX2).raw["task_specs"]["image_to_image"]["params"]
        self.assertIn("no published maximum", params["reference_images"]["description"])


@needs_planner
class TestFacade(RouteCase):
    def test_i2i_and_i2i_batch_resolve_each_item(self):
        from abstractvision.errors import CapabilityNotSupportedError

        cap = self.capability(self.backend(FLUX2), FLUX2)
        with tempfile.TemporaryDirectory() as td:
            on_disk = Path(td) / "reference.png"
            on_disk.write_bytes(REF_2)
            cap.i2i("a red kite", image=SOURCE, reference_images=[REF_1, str(on_disk)], seed=1)
        _name, kwargs, files = self.call()
        self.assertEqual([item["bytes"] for item in files["image_paths"]], [SOURCE, REF_1, REF_2])
        EVENTS.clear()
        cap.i2i_batch("a red kite", image=SOURCE, reference_images=[REF_1], seeds=[1, 2], count=2)
        self.assertEqual([len(event[2]["image_paths"]) for event in EVENTS if event[0] == "call"], [2, 2])
        for method, args in (("t2i", ("a red kite",)), ("t2v", ("a red kite",)), ("i2v", (SOURCE,))):
            with self.subTest(method=method):
                with self.assertRaises(CapabilityNotSupportedError) as caught:
                    getattr(cap, method)(*args, reference_images=[REF_1])
                self.assertEqual((caught.exception.field, caught.exception.supported), ("reference_images", ()))


@needs_planner
class TestChangedAgainst0337(RouteCase):
    def test_changed_6_extra_references_on_plain_qwen_image_edit_are_refused(self):
        from abstractvision.errors import CapabilityNotSupportedError

        self.assert_refused(
            QWEN_EDIT_PLAIN, CapabilityNotSupportedError, field="reference_images", extra={"reference_images": [REF_1]}
        )

    def test_changed_7_extra_references_with_an_adapter_on_a_row_without_lora(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import LoRAAdapterSpec

        self.assert_refused(
            QWEN_EDIT_2509,
            CapabilityNotSupportedError,
            field="lora_adapters",
            extra={"reference_images": [REF_1]},
            lora_adapters=(LoRAAdapterSpec(source=ADAPTER),),
        )

    def test_changed_8_the_list_and_extra_disagree(self):
        self.assert_refused(FLUX2, ValueError, reference_images=[REF_1], extra={"reference_images": [REF_2]})
        self.assert_refused(FLUX2, ValueError, reference_images=[REF_1], extra={"images": [REF_2]})

    def test_changed_9_the_keyword_on_the_other_backends(self):
        self.assert_other_backends_refuse(reference_images=[REF_1])


class TestAModelWithoutRows(RouteCase):
    """A folder the engine publishes no rows for: references run their 0.3.37 branch, typed or in
    `extra`; an option 0.3.37 did not have is refused."""

    def test_references_run_the_0_3_37_branch(self):
        from abstractvision.types import ImageEditRequest

        for base_model, name in (("flux2-klein-4b", "Flux2KleinEdit"), ("qwen-image-edit-2511", "QwenImageEdit")):
            for fields in ({"reference_images": [REF_1]}, {"extra": {"reference_images": [REF_1]}}):
                with self.subTest(model=base_model, fields=list(fields)):
                    EVENTS.clear()
                    backend = self.folder_backend(base_model)
                    backend.edit_image(ImageEditRequest(prompt="a red kite", image=SOURCE, seed=1, **fields))
                    called, _kwargs, files = self.call()
                    self.assertEqual(called, name)
                    self.assertEqual([item["bytes"] for item in files["image_paths"]], [SOURCE, REF_1])

    def test_an_option_0_3_37_did_not_have_is_refused(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageEditRequest

        for base_model in ("flux2-klein-4b", "qwen-image-edit-2511"):
            with self.subTest(model=base_model):
                backend = self.folder_backend(base_model)
                with self.assertRaises(CapabilityNotSupportedError) as caught:
                    backend.edit_image(ImageEditRequest(prompt="a red kite", image=SOURCE, image_strength=0.5))
                self.assertEqual((caught.exception.field, caught.exception.supported), ("image_strength", ()))
                self.assertIn("publishes no image_to_image route", str(caught.exception))
                self.assert_nothing_loaded(backend)


class TestOptionIsSet(unittest.TestCase):
    def test_changed_15_an_empty_sequence_is_unset(self):
        from abstractvision.backends.base_backend import option_is_set
        from abstractvision.backends.stable_diffusion_cpp import (
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
        )
        from abstractvision.types import ImageEditRequest

        self.assertEqual([option_is_set(value) for value in ((), [], (b"x",), 0, b"")], [False, False, True, True, True])
        # Every edit request carries the field: unset, it is not refused where it is not an option.
        backend = StableDiffusionCppVisionBackend(
            config=StableDiffusionCppBackendConfig(sd_cli_path="sd-cli", diffusion_model="model.gguf")
        )
        resolved, mode = backend.resolve_memory_request(
            ImageEditRequest(prompt="a red kite", image=SOURCE, extra={"reference_images": [REF_1]}),
            task="image_to_image",
        )
        self.assertEqual((mode, resolved.reference_images, resolved.extra), ("default", (), {"reference_images": [REF_1]}))


if __name__ == "__main__":
    unittest.main()
