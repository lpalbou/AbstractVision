"""`preload(task=...)` on MLX-Gen: a hint of the task the host is about to run, never a refusal; the
AbstractCore plugin forwards the payload's `task` only to a `preload` that takes it.
"""

from __future__ import annotations

import unittest

from unittest.mock import patch

from _mlx_gen_route_fixtures import (
    CLIP,
    ERNIE,
    EVENTS,
    FIBO,
    FIBO_EDIT,
    FLUX2,
    MODELS,
    QWEN,
    QWEN_EDIT,
    SOURCE,
    WAN_I2V,
    WAN_T2V,
    WAN_TI2V,
    WAN_VACE,
    Z_IMAGE,
    RouteCase,
    needs_planner,
)
from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401


class TestMlxGenPreloadTask(RouteCase):
    @needs_planner
    def test_image_to_image_on_flux2_klein_warms_the_edit_class_a_plain_edit_reuses(self):
        from abstractvision.types import ImageEditRequest

        backend = self.backend(FLUX2)
        backend.preload(task="image_to_image")
        self.assertEqual(self.inits(), ["Flux2KleinEdit"])
        backend.edit_image(ImageEditRequest(prompt="a red kite", image=SOURCE, seed=1))
        self.assertEqual(self.inits(), ["Flux2KleinEdit"])
        # The preload's warm-up edit, then the request's, on the one instance.
        self.assertEqual([event[1] for event in EVENTS if event[0] == "call"], ["Flux2KleinEdit", "Flux2KleinEdit"])

    def test_a_task_the_model_lacks_falls_back_to_the_default_choice(self):
        def classes():
            # Kind and class: a warm-up edit's temporary file names differ from one run to the next.
            return [event[:2] for event in EVENTS]

        for model, task in ((FLUX2, "text_to_video"), (QWEN_EDIT, "text_to_image"), (FLUX2, "not-a-task")):
            with self.subTest(model=model, task=task):
                EVENTS.clear()
                self.backend(model).preload()
                expected = classes()
                EVENTS.clear()
                self.backend(model).preload(task=task)
                self.assertEqual(classes(), expected)

    def test_none_and_text_to_image_preload_as_before_on_every_model_with_text_to_image(self):
        from abstractvision.backends.mflux import (
            _MFLUX_MODELS,
            _PRELOAD_TASK_ORDER,
            MFluxBackendConfig,
            MFluxVisionBackend,
        )

        # The choice: as the previous release made it, the first task of the order the model has.
        for base_model in sorted(_MFLUX_MODELS):
            backend = MFluxVisionBackend(config=MFluxBackendConfig(base_model=base_model))
            tasks = backend._supported_task_names()
            if "text_to_image" not in tasks:
                continue
            with self.subTest(model=base_model):
                before = next(task for task in _PRELOAD_TASK_ORDER if task in tasks)
                self.assertEqual(backend._preload_task(), before)
                self.assertEqual(backend._preload_task("text_to_image"), before)
        # The load itself, through the stand-in engine classes.
        for model in MODELS:
            if "text_to_image" not in self.backend(model)._supported_task_names():
                continue
            with self.subTest(model=model):
                EVENTS.clear()
                self.backend(model).preload()
                expected = list(EVENTS)
                EVENTS.clear()
                self.backend(model).preload(task="text_to_image")
                self.assertEqual(EVENTS, expected)


class _SeedVR2Config:
    @staticmethod
    def from_name(model_name, base_model=None):
        return f"config:{model_name}"

    @staticmethod
    def seedvr2_3b():
        return "config:seedvr2-3b"


class _ScaleFactor:
    @staticmethod
    def parse(value):
        return value


SEEDVR2 = "AbstractFramework/seedvr2-3b-8bit"


class TestPreloadWarmsWhatARunReads(RouteCase):
    """Default mode: an image task runs one warm-up of that task; a video task reads the parts its run holds
    when denoising starts; "low" and "lowest" read nothing ahead."""

    def setUp(self):
        super().setUp()
        from _mlx_gen_route_fixtures import snapshot, stand_in

        self.evaluated = []
        self.snapshots[SEEDVR2] = snapshot(self.snapshots[FLUX2].parents[2], SEEDVR2)
        for stub in (
            patch(
                "abstractvision.backends.mflux._evaluate_parameters",
                side_effect=lambda model, parts: self.evaluated.append((type(model).__name__, parts)),
            ),
            patch(
                "abstractvision.backends.mflux._lazy_import_mflux_seedvr2",
                return_value=(_SeedVR2Config, stand_in("SeedVR2"), _ScaleFactor, object),
            ),
        ):
            stub.start()
            self.addCleanup(stub.stop)

    def calls(self):
        return [event[1] for event in EVENTS if event[0] == "call"]

    # The warm-up canvas side each family must reach: 256, or the smallest side mlx-gen runs it at above 256
    # (cited in mflux._WARMUP_FAMILY_MIN_SIDE). Written out here so a change of the table is a test change.
    FAMILY_SIDE = {"flux2": 256, "qwen": 352, "qwen-edit": 256, "z-image": 256, "ernie-image": 512, "fibo": 352, "fibo-edit": 384}

    def test_text_to_image_warms_with_a_small_ordinary_request(self):
        # The model built, then one generation at the family's warm-up side and 2 steps.
        from abstractvision.types import ImageGenerationRequest

        for model in MODELS:
            backend = self.backend(model)
            if "text_to_image" not in backend._supported_task_names():
                continue
            with self.subTest(model=model):
                side = self.FAMILY_SIDE[backend._resolved_model_def().family]
                EVENTS.clear()
                backend.preload()
                preloaded = list(EVENTS)
                (warmup,) = [event[2] for event in preloaded if event[0] == "call"]
                self.assertEqual(
                    (warmup["width"], warmup["height"], warmup["num_inference_steps"]), (side, side, 2)
                )
                EVENTS.clear()
                model_def = backend._resolved_model_def()
                self.backend(model).generate_image(
                    ImageGenerationRequest(
                        memory_mode="default",
                        prompt="abstractvision preload warmup",
                        width=side,
                        height=side,
                        steps=2,
                        guidance_scale=model_def.default_guidance,
                        seed=0,
                    )
                )
                self.assertEqual(preloaded, EVENTS)
                self.assertEqual(self.evaluated, [])

    def test_the_image_to_image_and_upscale_warm_ups(self):
        from io import BytesIO

        from PIL import Image

        from abstractvision.backends import mflux

        cases = [(FLUX2, "image_to_image"), (QWEN_EDIT, None), (FIBO_EDIT, None), (ERNIE, "image_to_image")]
        if "image_to_image" in self.backend(Z_IMAGE)._supported_task_names():
            cases += [(Z_IMAGE, "image_to_image"), (QWEN, "image_to_image")]
        for model, task in cases:
            with self.subTest(model=model):
                EVENTS.clear()
                backend = self.backend(model)
                side = self.FAMILY_SIDE[backend._resolved_model_def().family]
                backend.preload(task=task)
                (call,) = [event for event in EVENTS if event[0] == "call"]
                facts = call[3].get("image_path") or call[3]["image_paths"][0]
                self.assertEqual(Image.open(BytesIO(facts["bytes"])).size, (side, side))
                self.assertEqual(call[2]["num_inference_steps"], mflux._WARMUP_STEPS)
        EVENTS.clear()
        self.backend(SEEDVR2).preload()
        (call,) = [event for event in EVENTS if event[0] == "call"]
        self.assertEqual(str(call[2]["resolution"]), "1x")
        self.assertEqual(Image.open(BytesIO(call[3]["image_path"]["bytes"])).size, (64, 64))

    def test_the_canvas_is_never_below_a_familys_cited_minimum(self):
        from abstractvision.backends import mflux
        from abstractvision.backends.mflux import _MFLUX_MODELS, MFluxBackendConfig, MFluxVisionBackend

        for base_model, model_def in sorted(_MFLUX_MODELS.items()):
            backend = MFluxVisionBackend(config=MFluxBackendConfig(base_model=base_model))
            for task in ("text_to_image", "image_to_image"):
                if task not in backend._supported_task_names():
                    continue
                with self.subTest(model=base_model, task=task):
                    width, height = backend._warmup_canvas(task)
                    self.assertEqual(width, height)
                    self.assertGreaterEqual(width, mflux._WARMUP_IMAGE_SIDE)
                    self.assertGreaterEqual(width, mflux._WARMUP_FAMILY_MIN_SIDE.get(model_def.family, 0))
                    self.assertEqual(width % 16, 0)

    @needs_planner  # the load-time widths a model accepts are the engine's
    def test_the_warm_up_keeps_the_preloads_quantize(self):
        # A folder of 16-bit weights takes a load-time width: the warm-up edit runs on the preloaded instance.
        backend = self.folder_backend("flux2-klein-9b")
        backend.preload(task="image_to_image", quantize=8)
        inits = [event[2] for event in EVENTS if event[0] == "init"]
        self.assertEqual(([init["quantize"] for init in inits], self.calls()), ([8], ["Flux2KleinEdit"]))

    def test_a_configured_default_canvas_is_kept(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        backend = MFluxVisionBackend(config=MFluxBackendConfig(model=FLUX2, default_width=320, default_height=192))
        backend.preload()
        (warmup,) = [event[2] for event in EVENTS if event[0] == "call"]
        self.assertEqual((warmup["width"], warmup["height"], warmup["num_inference_steps"]), (320, 192, 2))

    @needs_planner
    def test_the_engine_floor_the_steps_read(self):
        from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler

        from abstractvision.backends import mflux

        FlowMatchEulerDiscreteScheduler._validate_num_inference_steps(mflux._WARMUP_STEPS)
        with self.assertRaises(ValueError):
            FlowMatchEulerDiscreteScheduler._validate_num_inference_steps(mflux._WARMUP_STEPS - 1)

    @needs_planner
    def test_image_to_image_runs_one_plain_edit_and_the_first_request_builds_nothing(self):
        from abstractvision.types import ImageEditRequest

        cases = (
            (FLUX2, "image_to_image", "Flux2KleinEdit"),
            (QWEN, "image_to_image", "QwenImage"),
            (Z_IMAGE, "image_to_image", "ZImage"),
            (ERNIE, "image_to_image", "ErnieImageTurbo"),
            (FIBO, "image_to_image", "FIBO"),
            (QWEN_EDIT, None, "QwenImageEdit"),
            (FIBO_EDIT, None, "FIBOEdit"),
        )
        for model, task, cls in cases:
            with self.subTest(model=model):
                EVENTS.clear()
                backend = self.backend(model)
                self.assertIsNone(backend.preload(task=task))
                self.assertEqual((self.inits(), self.calls()), ([cls], [cls]))
                self.assertEqual(list(self.tmp_root.iterdir()), [])
                backend.edit_image(ImageEditRequest(prompt="a red kite", image=SOURCE, seed=1))
                self.assertEqual((self.inits(), self.calls()), ([cls], [cls, cls]))
                self.assertEqual(self.evaluated, [])

    def test_image_upscale_runs_one_upscale(self):
        from abstractvision.types import ImageUpscaleRequest

        backend = self.backend(SEEDVR2)
        backend.preload()
        self.assertEqual((self.inits(), self.calls()), (["SeedVR2"], ["SeedVR2"]))
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        backend.upscale_image(ImageUpscaleRequest(image=SOURCE, seed=1))
        self.assertEqual((self.inits(), self.calls()), (["SeedVR2"], ["SeedVR2", "SeedVR2"]))
        self.assertEqual(self.evaluated, [])

    @needs_planner
    def test_video_tasks_read_the_parts_their_run_holds_and_the_first_request_reads_none(self):
        from abstractvision.types import VideoToVideoRequest

        backend = self.backend(WAN_VACE)
        backend.preload()
        self.assertEqual((self.evaluated, self.calls()), ([("WanVace", ("transformer", "vae"))], []))
        backend.video_to_video(VideoToVideoRequest(prompt="a lantern on a jetty", video=CLIP, seed=1))
        self.assertEqual((len(self.evaluated), self.inits()), (1, ["WanVace"]))

    def test_what_each_video_task_reads(self):
        cases = (
            (WAN_TI2V, None, ("Wan2_2_TI2V", ("transformer",))),
            (WAN_TI2V, "image_to_video", ("Wan2_2_TI2V", ("transformer", "vae"))),
            # A14B: the first expert only; the second is read at the boundary, where the run may free the first.
            (WAN_T2V, None, ("Wan2_2_TI2V", ("transformer",))),
            (WAN_T2V, "video_to_video", ("Wan2_2_TI2V", ("transformer", "vae"))),
            (WAN_I2V, None, ("Wan2_2_TI2V", ("transformer", "vae"))),
        )
        for model, task, expected in cases:
            if task is not None and task not in self.backend(model)._supported_task_names():
                continue  # a task the engine's rows add (skipped in the base job)
            with self.subTest(model=model, task=task):
                self.evaluated.clear()
                EVENTS.clear()
                self.backend(model).preload(task=task)
                self.assertEqual((self.evaluated, self.calls()), ([expected], []))

    def test_minimax_h3_reads_nothing_more(self):
        from abstractvision.backends.mflux import _preload_video_parts

        self.assertEqual(_preload_video_parts("minimax-h3", "text_to_video"), ())
        self.assertEqual(_preload_video_parts("minimax-h3", "image_to_video"), ())

    def test_nothing_ahead_in_low_ram_modes(self):
        for model, task in ((FLUX2, None), (FLUX2, "image_to_image"), (QWEN_EDIT, None), (SEEDVR2, None), (WAN_T2V, None), (WAN_VACE, None)):
            if not self.backend(model)._supported_task_names():
                continue  # VACE without the engine's rows has no task (base job)
            with self.subTest(model=model, task=task):
                self.evaluated.clear()
                EVENTS.clear()
                self.backend(model).preload(memory_mode="low", task=task)
                self.assertEqual((self.evaluated, self.calls()), ([], []))


class TestEvaluateParameters(unittest.TestCase):
    def test_only_the_named_parts(self):
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("needs mlx")
        from abstractvision.backends.mflux import _evaluate_parameters

        class _Part:
            def __init__(self):
                self.weight = mx.ones((4,)) * 2

            def parameters(self):
                return {"weight": self.weight}

        class _Model:
            def __init__(self):
                self.transformer, self.transformer_2, self.vae = _Part(), _Part(), _Part()

        evaluated = []
        with patch.object(mx, "eval", side_effect=lambda trees: evaluated.append(trees)):
            model = _Model()
            _evaluate_parameters(model, ("transformer", "vae"))
            _evaluate_parameters(model, ())
        self.assertEqual(evaluated, [[{"weight": model.transformer.weight}, {"weight": model.vae.weight}]])


class _Recorder:
    """The methods the plugin calls on a backend; `preload` is set per case."""

    def __init__(self):
        self.calls = []

    def has_loaded_model(self):
        return False

    def unload(self):
        return None

    def supported_memory_modes(self, task=None):
        return ["default"]

    def list_provider_models(self, *, task=None):
        return []

    def list_provider_adapters(self, *, model=None, task=None):
        return []


class _NoArguments(_Recorder):
    def preload(self):
        self.calls.append({})


class _MemoryAndQuantize(_Recorder):
    def preload(self, memory_mode=None, quantize=None):
        self.calls.append({"memory_mode": memory_mode, "quantize": quantize})


class _AnyKeyword(_Recorder):
    def preload(self, **kwargs):
        self.calls.append(dict(kwargs))


class TestPluginForwardsTheTask(RouteCase):
    def capability(self, backend, kind):
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _Owner:
            config = {"vision_backend_instance": backend, "vision_backend": kind}

        return _AbstractVisionCapability(_Owner())

    @needs_planner
    def test_to_mlx_gen(self):
        backend = self.backend(FLUX2)
        self.capability(backend, "mlx-gen").load_resident_model({"task": "image_to_image"})
        self.assertEqual(self.inits(), ["Flux2KleinEdit"])

    def test_only_to_a_preload_that_takes_it(self):
        for cls, expected in (
            (_NoArguments, {}),
            (_MemoryAndQuantize, {"memory_mode": "default", "quantize": None}),
            (_AnyKeyword, {"memory_mode": "default", "task": "image_to_image", "warm_up": False}),
        ):
            with self.subTest(backend=cls.__name__):
                backend = cls()
                payload = {"task": "image_to_image", "warm_up": False}
                if cls is not _NoArguments:
                    payload["memory_mode"] = "default"
                record = self.capability(backend, "diffusers").load_resident_model(payload)
                self.assertEqual(backend.calls, [expected])
                # A backend that does not say whether it warmed: the record says nothing either.
                self.assertIsNone(record.get("warmed"))

    def test_warm_up_to_mlx_gen_and_the_record_says_whether_it_warmed(self):
        for warm_up, calls, warmed in ((False, [], False), (True, ["Flux2Klein"], True), (None, ["Flux2Klein"], True)):
            with self.subTest(warm_up=warm_up):
                EVENTS.clear()
                payload = {} if warm_up is None else {"warm_up": warm_up}
                record = self.capability(self.backend(FLUX2), "mlx-gen").load_resident_model(payload)
                self.assertEqual(([e[1] for e in EVENTS if e[0] == "call"], self.inits()), (calls, ["Flux2Klein"]))
                self.assertIs(record["warmed"], warmed)


class TestPreloadWithoutWarmUp(RouteCase):
    """`warm_up=False`: the model is resolved, checked and built; nothing is run or read ahead."""

    def setUp(self):
        super().setUp()
        self.evaluated = []
        stub = patch(
            "abstractvision.backends.mflux._evaluate_parameters",
            side_effect=lambda model, parts: self.evaluated.append((type(model).__name__, parts)),
        )
        stub.start()
        self.addCleanup(stub.stop)

    def events(self, model, **kwargs):
        EVENTS.clear()
        self.evaluated.clear()
        backend = self.backend(model)
        backend.preload(**kwargs)
        return [event[:2] for event in EVENTS], list(self.evaluated), backend.preload_warmed()

    def test_an_image_task(self):
        self.assertEqual(
            self.events(FLUX2, warm_up=False), ([("init", "Flux2Klein")], [], False)
        )
        self.assertEqual(
            self.events(FLUX2, warm_up=True), ([("init", "Flux2Klein"), ("call", "Flux2Klein")], [], True)
        )

    def test_a_video_task(self):
        self.assertEqual(self.events(WAN_T2V, warm_up=False), ([("init", "Wan2_2_TI2V")], [], False))
        self.assertEqual(
            self.events(WAN_T2V, warm_up=True),
            ([("init", "Wan2_2_TI2V")], [("Wan2_2_TI2V", ("transformer",))], True),
        )

    def test_seedvr2(self):
        from _mlx_gen_route_fixtures import snapshot, stand_in

        self.snapshots[SEEDVR2] = snapshot(self.snapshots[FLUX2].parents[2], SEEDVR2)
        with patch(
            "abstractvision.backends.mflux._lazy_import_mflux_seedvr2",
            return_value=(_SeedVR2Config, stand_in("SeedVR2"), _ScaleFactor, object),
        ):
            self.assertEqual(self.events(SEEDVR2, warm_up=False), ([("init", "SeedVR2")], [], False))
            self.assertEqual(
                self.events(SEEDVR2, warm_up=True), ([("init", "SeedVR2"), ("call", "SeedVR2")], [], True)
            )

    def test_low_is_the_same_either_way(self):
        for warm_up in (True, False):
            with self.subTest(warm_up=warm_up):
                self.assertEqual(
                    self.events(FLUX2, memory_mode="low", warm_up=warm_up), ([("init", "Flux2Klein")], [], False)
                )

    def test_a_later_preload_with_warm_up_warms_the_held_model(self):
        backend = self.backend(FLUX2)
        backend.preload(warm_up=False)
        EVENTS.clear()
        backend.preload()
        self.assertEqual([event[:2] for event in EVENTS], [("call", "Flux2Klein")])
        self.assertTrue(backend.preload_warmed())


if __name__ == "__main__":
    unittest.main()
