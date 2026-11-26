"""Control image on text-to-image (0.3.38 step 8): MLX-Gen's qwen.control route
(`QwenImageControlNet`) on AbstractFramework/qwen-image-8bit, with the stand-in engine classes of
`_mlx_gen_route_fixtures` and a stand-in for the Hugging Face cache lookup of the side weights.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from _mlx_gen_route_fixtures import (
    EVENTS,
    FLUX2,
    QWEN,
    QWEN_2512,
    SOURCE,
    QwenImageControlNet,
    RouteCase,
    needs_planner,
)
from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401

CONTROL = SOURCE
CONTROLNET = "InstantX/Qwen-Image-ControlNet-Union"
CONTROLNET_FILE = "diffusion_pytorch_model.safetensors"


class _Case(RouteCase):
    def setUp(self):
        super().setUp()
        from huggingface_hub.utils import LocalEntryNotFoundError

        self.side_weights = "present"  # or "missing" (not in the cache), "file-absent" (snapshot without it)
        self.lookups = []
        snapshot = Path(tempfile.mkdtemp(dir=self.tmp_root.parent))
        self.addCleanup(lambda: [p.unlink() for p in snapshot.iterdir()])

        def snapshot_download(repo_id, *, allow_patterns, local_files_only):
            self.lookups.append((repo_id, list(allow_patterns), local_files_only))
            if self.side_weights == "missing":
                raise LocalEntryNotFoundError("not cached")
            if self.side_weights == "present":
                (snapshot / CONTROLNET_FILE).write_bytes(b"x")
            return str(snapshot)

        stub = patch(
            "abstractvision.backends.mflux._lazy_import_hf_cache_lookup",
            return_value=(snapshot_download, LocalEntryNotFoundError),
        )
        stub.start()
        self.addCleanup(stub.stop)

    def t2i(self, model=QWEN, **fields):
        from abstractvision.types import ImageGenerationRequest

        backend = self.backend(model)
        return backend, backend.generate_image(ImageGenerationRequest(prompt="a harbour at dawn", **fields))

    def assert_i2i_refuses_the_control_keywords(self):
        from abstractvision.errors import CapabilityNotSupportedError

        cap = self.capability(self.backend(QWEN), QWEN)
        for name, value in (("control_image", CONTROL), ("control_strength", 0.5)):
            with self.subTest(keyword=name):
                with self.assertRaises(CapabilityNotSupportedError) as caught:
                    cap.i2i("a harbour", image=SOURCE, **{name: value})
                self.assertEqual((caught.exception.field, caught.exception.supported), (name, ()))
        self.assertEqual(self.inits(), [])

    def assert_t2i_refused(self, error, *, field="unset", supported="unset", model=QWEN, **fields):
        from abstractvision.types import ImageGenerationRequest

        backend = self.backend(model)
        with self.assertRaises(error) as caught:
            backend.generate_image(ImageGenerationRequest(prompt="a harbour at dawn", **fields))
        if field != "unset":
            self.assertEqual(caught.exception.field, field)
        if supported != "unset":
            self.assertEqual(caught.exception.supported, supported)
        self.assert_nothing_loaded(backend)
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        return caught.exception


@needs_planner
class TestCalls(_Case):
    def test_the_control_call(self):
        _backend, asset = self.t2i(
            control_image=CONTROL, control_strength=0.7, seed=5, steps=12, guidance_scale=4.0, width=512, height=384,
            negative_prompt="blur",
        )
        name, kwargs, files = self.call()
        self.assertEqual(name, "QwenImageControlNet")
        (init,) = [event[2] for event in EVENTS if event[0] == "init"]
        self.assertEqual(init["controlnet_model"], f"{CONTROLNET}:{CONTROLNET_FILE}")
        self.assertEqual(self.lookups, [(CONTROLNET, [CONTROLNET_FILE, "config.json"], True)])
        self.assertEqual(files["controlnet_image_path"]["bytes"], CONTROL)
        kwargs.pop("controlnet_image_path")
        self.assertEqual(
            kwargs,
            {
                "seed": 5,
                "prompt": "a harbour at dawn",
                "controlnet_strength": 0.7,
                "num_inference_steps": 12,
                "height": 384,
                "width": 512,
                "guidance": 4.0,
                "negative_prompt": "blur",
            },
        )
        self.assertEqual(asset.metadata["control_strength"], 0.7)
        self.assert_metadata_clean(asset, files["controlnet_image_path"]["dir"])
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_every_optional_value_unset(self):
        self.t2i(control_image=CONTROL)
        _name, kwargs, _files = self.call()
        kwargs.pop("controlnet_image_path")
        self.assertIsInstance(kwargs.pop("seed"), int)
        # The engine CLI's strength 0.4, the model's guidance 3.5 and steps 20.
        self.assertEqual(
            kwargs,
            {
                "prompt": "a harbour at dawn",
                "controlnet_strength": 0.4,
                "num_inference_steps": 20,
                "height": 1024,
                "width": 1024,
                "guidance": 3.5,
            },
        )

    def test_extra_scheduler_reaches_the_class(self):
        # Review finding 7: as the 0.3.37 image routes pass it.
        self.t2i(control_image=CONTROL, extra={"scheduler": "my.Sched"})
        self.assertEqual(self.call()[1]["scheduler"], "my.Sched")

    def test_a_plain_generation_after_it_rebuilds_qwen_image(self):
        from abstractvision.types import ImageGenerationRequest

        backend, _asset = self.t2i(control_image=CONTROL)
        self.assertTrue(backend._model_key.variant.startswith("qwen-control:"))
        backend.generate_image(ImageGenerationRequest(prompt="again", steps=2))
        self.assertEqual(self.inits(), ["QwenImageControlNet", "QwenImage"])

    def test_low_ram_and_the_run_directory(self):
        from abstractvision.errors import GenerationCancelledError

        backend, asset = self.t2i(control_image=CONTROL, memory_mode="low")
        self.assertIn(("register", "MemorySaver", False), EVENTS)
        self.assertFalse(backend.has_loaded_model())
        QwenImageControlNet.fail_with = RuntimeError("engine failed")
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            self.t2i(control_image=CONTROL)
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        QwenImageControlNet.fail_with = None
        event = threading.Event()
        QwenImageControlNet.during_call = event.set
        with self.assertRaises(GenerationCancelledError):
            self.t2i(control_image=CONTROL, cancel_event=event)
        self.assertEqual(list(self.tmp_root.iterdir()), [])


@needs_planner
class TestRefusals(_Case):
    def test_any_other_model(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for model in (QWEN_2512, FLUX2):
            with self.subTest(model=model):
                self.assert_t2i_refused(
                    CapabilityNotSupportedError, field="control_image", supported=(), model=model, control_image=CONTROL
                )

    def test_side_weights_not_in_the_cache(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for state in ("missing", "file-absent"):
            with self.subTest(state=state):
                self.side_weights = state
                error = self.assert_t2i_refused(
                    CapabilityNotSupportedError, field="control_image", supported=(), control_image=CONTROL
                )
                for part in (CONTROLNET, CONTROLNET_FILE, f"abstractvision download {CONTROLNET}"):
                    self.assertIn(part, str(error))

    def test_a_strength_without_a_control_image_or_not_a_number(self):
        from abstractvision.errors import CapabilityNotSupportedError

        self.assert_t2i_refused(CapabilityNotSupportedError, field="control_strength", supported=(), control_strength=0.5)
        for value in ("strong", float("nan")):
            with self.subTest(value=value):
                self.assert_t2i_refused(ValueError, control_image=CONTROL, control_strength=value)

    def test_a_control_image_that_fails_to_decode(self):
        error = self.assert_t2i_refused(ValueError, control_image=b"not a picture")
        self.assertTrue(str(error).startswith("control_image "), error)

    def test_the_control_keywords_on_i2i(self):
        self.assert_i2i_refuses_the_control_keywords()


@needs_planner
class TestPublicationAndFacade(_Case):
    def test_capabilities_parameters_and_catalog(self):
        backend = self.backend(QWEN)
        self.assertTrue(backend.get_capabilities().supports_control_image)
        self.assertFalse(self.backend(QWEN_2512).get_capabilities().supports_control_image)
        (model,) = [m for m in backend.list_provider_models(task="text_to_image") if m.id == QWEN]
        self.assertEqual(model.raw["parameter_defaults"]["control_strength"], 0.4)
        self.assertNotIn("control_strength", model.raw["parameter_constraints"])
        params = model.raw["task_specs"]["text_to_image"]["params"]
        self.assertIn("control_image", params)
        self.assertEqual(params["control_strength"]["default"], 0.4)

    def test_the_facade_and_the_manager(self):
        from abstractvision import VisionManager

        cap = self.capability(self.backend(QWEN), QWEN)
        cap.t2i("a harbour", control_image=CONTROL, control_strength=0.6, seed=1)
        _name, kwargs, _files = self.call()
        self.assertEqual(kwargs["controlnet_strength"], 0.6)
        EVENTS.clear()
        VisionManager(backend=self.backend(QWEN)).generate_image("a harbour", control_image=CONTROL)
        self.assertEqual(self.call()[0], "QwenImageControlNet")


@needs_planner
class TestChangedAgainst0337(_Case):
    def test_changed_10_control_keywords_on_i2i(self):
        self.assert_i2i_refuses_the_control_keywords()

    def test_changed_13_control_runs_on_qwen_image_8bit(self):
        self.t2i(control_image=CONTROL)
        self.assertEqual(self.call()[0], "QwenImageControlNet")

    def test_changed_14_supports_control_image(self):
        self.assertTrue(self.backend(QWEN).get_capabilities().supports_control_image)


if __name__ == "__main__":
    unittest.main()
