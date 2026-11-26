"""Fixes from the review of the 0.3.38 routes: one test class per finding, from the reviewer's inputs,
with the stand-in engine classes of `_mlx_gen_route_fixtures`.
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from _mlx_gen_route_fixtures import (
    CLIP,
    ERNIE,
    EVENTS,
    FLUX2,
    FLUX2_BASE,
    QWEN_EDIT,
    QWEN_EDIT_PLAIN,
    SOURCE,
    WAN_I2V,
    WAN_T2V,
    WAN_VACE,
    Z_IMAGE_TURBO,
    RouteCase,
    needs_planner,
    png,
)
from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401

REF = png(32, 32, color=(10, 200, 10))
MASK = png(64, 48, color=(255, 255, 255))


class _PathLike:
    """A path-like that is not a `pathlib.Path` (os.PathLike)."""

    def __init__(self, path):
        self.path = str(path)

    def __fspath__(self):
        return self.path


class TestFinding1RowlessStrength(RouteCase):
    """A model the engine publishes no rows for: image_strength runs 0.3.37's branch on ERNIE only."""

    def test_ernie_runs_the_0_3_37_latent_edit(self):
        from abstractvision.types import ImageEditRequest

        for fields in ({"extra": {"image_strength": 0.7}}, {"image_strength": 0.7}):
            with self.subTest(fields=list(fields)):
                EVENTS.clear()
                backend = self.folder_backend("ernie-image-turbo")
                backend.edit_image(ImageEditRequest(prompt="a red kite", image=SOURCE, seed=1, **fields))
                name, kwargs, _files = self.call()
                self.assertEqual((name, kwargs["image_strength"]), ("ErnieImageTurbo", 0.7))

    def test_ernie_passes_the_negative_prompt_as_0_3_37_did(self):
        from abstractvision.types import ImageEditRequest

        backend = self.folder_backend("ernie-image-turbo")
        backend.edit_image(
            ImageEditRequest(prompt="a red kite", image=SOURCE, seed=1, image_strength=0.7, negative_prompt="blur")
        )
        name, kwargs, _files = self.call()
        self.assertEqual((name, kwargs.get("negative_prompt")), ("ErnieImageTurbo", "blur"))

    def test_any_other_family_refuses_it_before_any_load(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageEditRequest

        for base_model in ("flux2-klein-9b", "fibo"):
            with self.subTest(model=base_model):
                backend = self.folder_backend(base_model)
                with self.assertRaises(CapabilityNotSupportedError) as caught:
                    backend.edit_image(
                        ImageEditRequest(prompt="a red kite", image=SOURCE, extra={"image_strength": 0.7})
                    )
                self.assertEqual((caught.exception.field, caught.exception.supported), ("image_strength", ()))
                self.assert_nothing_loaded(backend)


class TestFinding2RowlessNewTasks(RouteCase):
    """A task this release adds needs the engine's rows: without them the model has its 0.3.37 tasks."""

    def test_image_to_image_on_rowless_z_image_and_base_qwen(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageEditRequest

        for base_model in ("z-image-turbo", "z-image", "qwen-image"):
            for fields in ({}, {"mask": MASK}, {"extra": {"strength": 0.3}}):
                with self.subTest(model=base_model, fields=list(fields)):
                    backend = self.folder_backend(base_model)
                    self.assertEqual(backend.get_capabilities().supported_tasks, ["text_to_image"])
                    with self.assertRaisesRegex(CapabilityNotSupportedError, "has no image_to_image route"):
                        backend.edit_image(ImageEditRequest(prompt="a red kite", image=SOURCE, **fields))
                    self.assert_nothing_loaded(backend)

    def test_video_to_video_on_a_rowless_wan_folder(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoToVideoRequest

        for base_model, tasks in (("wan2.2-t2v-a14b", ["text_to_video"]), ("wan2.1-vace-1.3b", [])):
            with self.subTest(model=base_model):
                backend = self.folder_backend(base_model)
                self.assertEqual(backend.get_capabilities().supported_tasks, tasks)
                with self.assertRaisesRegex(CapabilityNotSupportedError, "has no video_to_video route"):
                    backend.video_to_video(VideoToVideoRequest(prompt="a fox", video=CLIP))
                self.assert_nothing_loaded(backend)


class TestFinding3PluginInputs(RouteCase):
    """The plugin reads what 0.3.37's `extra` path read: any path-like, any bytes-like, artifact refs."""

    def test_the_resolvers(self):
        from abstractvision.integrations.abstractcore_plugin import (
            _resolve_bytes_input,
            _resolve_reference_images,
            _resolve_video_input,
        )

        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "ref.png"
            ref.write_bytes(REF)
            for value in (ref, _PathLike(ref), str(ref), REF, bytearray(REF), memoryview(REF)):
                with self.subTest(value=type(value).__name__):
                    self.assertEqual(_resolve_bytes_input(value, artifact_store=None), REF)
            self.assertEqual(_resolve_reference_images(ref, artifact_store=None), [REF])
            self.assertEqual(_resolve_reference_images(memoryview(REF), artifact_store=None), [REF])
            self.assertEqual(_resolve_reference_images([ref, memoryview(REF)], artifact_store=None), [REF, REF])
            self.assertEqual(_resolve_video_input(_PathLike(ref), artifact_store=None), ref)
            self.assertEqual(_resolve_video_input(memoryview(CLIP), artifact_store=None), CLIP)

    @needs_planner
    def test_i2i_reference_images_given_as_paths(self):
        cap = self.capability(self.backend(FLUX2), FLUX2)
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / "ref.png"
            ref.write_bytes(REF)
            for value in ([ref], ref, [memoryview(REF)]):
                with self.subTest(value=type(value).__name__):
                    EVENTS.clear()
                    cap.i2i("p", SOURCE, reference_images=value, seed=1)
                    name, _kwargs, files = self.call()
                    self.assertEqual(name, "Flux2KleinEdit")
                    self.assertEqual([item["bytes"] for item in files["image_paths"]], [SOURCE, REF])

    def test_single_image_inputs_given_as_paths_and_memoryviews(self):
        cap = self.capability(self.backend(QWEN_EDIT), QWEN_EDIT)
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.png"
            source.write_bytes(SOURCE)
            cap.i2i("p", source, mask=memoryview(MASK), seed=1)
        name, kwargs, _files = self.call()
        self.assertEqual(name, "QwenImageEdit")
        self.assertIn("mask_path", kwargs)


@needs_planner
class TestFinding4ReferencesWithoutAMultiReferenceRoute(RouteCase):
    def test_extra_references_on_models_whose_rows_take_one_image(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for model in (ERNIE, QWEN_EDIT_PLAIN):
            with self.subTest(model=model):
                self.assert_refused(
                    model, CapabilityNotSupportedError, field="reference_images", supported=(),
                    extra={"reference_images": [REF]},
                )


class TestFinding5UnsetPluginKeywords(RouteCase):
    def test_an_unset_edit_keyword_is_not_refused(self):
        cap = self.capability(self.backend(FLUX2), FLUX2)
        for name, value in (("reference_images", []), ("outpaint_padding", ""), ("image_strength", None)):
            with self.subTest(keyword=name):
                EVENTS.clear()
                cap.t2i("a kite", steps=2, **{name: value})
                self.assertEqual(self.call()[0], "Flux2Klein")

    def test_a_set_edit_keyword_is_refused_on_the_other_calls(self):
        from abstractvision.errors import CapabilityNotSupportedError

        cap = self.capability(self.backend(FLUX2), FLUX2)
        for method, args, name, value in (
            ("t2i", ("a kite",), "image_strength", 0.5),
            ("t2v", ("a kite",), "reference_images", [REF]),
            ("i2v", (SOURCE,), "outpaint_padding", "10%"),
            ("upscale_image", (SOURCE,), "reframe_padding", "10%"),
        ):
            with self.subTest(method=method, keyword=name):
                with self.assertRaises(CapabilityNotSupportedError) as caught:
                    getattr(cap, method)(*args, **{name: value})
                self.assertEqual((caught.exception.field, caught.exception.supported), (name, ()))


@needs_planner
class TestFinding6GuidancePair(RouteCase):
    def test_a_guidance_scale_set_alone(self):
        from abstractvision.types import VideoGenerationRequest, VideoToVideoRequest

        canvas = {"width": 480, "height": 240, "num_frames": 17, "guidance_scale": 6.0}
        self.backend(WAN_T2V).generate_video(VideoGenerationRequest(prompt="a fox", **canvas))
        t2v = self.call()[1]
        EVENTS.clear()
        self.backend(WAN_T2V).video_to_video(VideoToVideoRequest(prompt="a fox", video=CLIP, **canvas))
        v2v = self.call()[1]
        # They differ on purpose: video-to-video follows the engine CLI; text-to-video keeps 0.3.37's model default.
        self.assertEqual((t2v["guidance"], t2v["guidance_2"]), (6.0, 3.0))
        self.assertEqual(v2v["guidance"], 6.0)
        self.assertNotIn("guidance_2", v2v)

    def test_unset_and_both_set(self):
        from abstractvision.types import VideoToVideoRequest

        for fields, expected in (({}, (4.0, 3.0)), ({"guidance_scale": 6.0, "guidance_2": 2.0}, (6.0, 2.0))):
            with self.subTest(fields=fields):
                EVENTS.clear()
                backend = self.backend(WAN_T2V)
                request = VideoToVideoRequest(prompt="a fox", video=CLIP, width=480, height=240, num_frames=17, **fields)
                # Normalized again, the request keeps the same pair.
                backend.video_to_video(dataclasses.replace(backend.normalize_video_to_video_request(request), seed=3))
                kwargs = self.call()[1]
                self.assertEqual((kwargs["guidance"], kwargs["guidance_2"]), expected)


@needs_planner
class TestFinding7Scheduler(RouteCase):
    """`extra["scheduler"]` reaches every new route's class, as the 0.3.37 edit branch passes it."""

    def test_the_masked_outpaint_and_reframe_routes(self):
        source = png(432, 240)
        for model, fields in (
            (FLUX2, {"mask": MASK}),
            (FLUX2, {"outpaint_padding": "0,25%,0,25%"}),
            (FLUX2, {"reframe_padding": "0,10%,0,10%"}),
            (QWEN_EDIT, {"outpaint_padding": "0,25%,0,25%"}),
            (QWEN_EDIT, {"reframe_padding": "0,10%,0,10%"}),
        ):
            with self.subTest(model=model, fields=list(fields)):
                from abstractvision.types import ImageEditRequest

                EVENTS.clear()
                self.backend(model).edit_image(
                    ImageEditRequest(prompt="p", image=source, extra={"scheduler": "my.Sched"}, **fields)
                )
                self.assertEqual(self.call()[1]["scheduler"], "my.Sched")


class TestFinding8NormalizedRequestsRunAgain(RouteCase):
    """Normalization accepts its own output: normalize, replace the seed, run, on every video task."""

    def run_twice(self, backend, normalize, run, request):
        normalized = normalize(request)
        run(dataclasses.replace(normalized, seed=2))
        self.assertEqual(self.call()[2] is not None, True)

    def test_text_and_image_to_video(self):
        from abstractvision.types import ImageToVideoRequest, VideoGenerationRequest

        backend = self.backend(WAN_T2V)
        self.run_twice(
            backend, backend.normalize_video_generation_request, backend.generate_video,
            VideoGenerationRequest(prompt="a fox", flow_shift=5.0, width=480, height=240, num_frames=17),
        )
        EVENTS.clear()
        backend = self.backend(WAN_I2V)
        self.run_twice(
            backend, backend.normalize_image_to_video_request, backend.image_to_video,
            ImageToVideoRequest(image=SOURCE, prompt="a fox", flow_shift=5.0, width=480, height=240, num_frames=17),
        )

    @needs_planner
    def test_video_to_video(self):
        from abstractvision.types import VideoToVideoRequest

        for model in (WAN_T2V, WAN_VACE):
            with self.subTest(model=model):
                EVENTS.clear()
                backend = self.backend(model)
                self.run_twice(
                    backend, backend.normalize_video_to_video_request, backend.video_to_video,
                    VideoToVideoRequest(
                        prompt="a fox", video=CLIP, flow_shift=5.0, width=480, height=240, num_frames=17,
                        extra={"max_sequence_length": 256},
                    ),
                )
                self.assertEqual(self.call()[1]["flow_shift"], 5.0)


@needs_planner
class TestFinding9CatalogFollowsTheRows(unittest.TestCase):
    """Every route option a catalog entry lists for an MLX-Gen image task is taken by the engine's rows of
    one of its MLX-Gen packages (entries whose packages have no rows run 0.3.37's paths and are skipped)."""

    def test_the_catalog_lists_only_what_the_rows_take(self):
        from mflux.task_inference import get_model_capabilities

        from abstractvision.backends import mlx_gen_routes
        from abstractvision.model_capabilities import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        options = {option.name: option for option in mlx_gen_routes.ROUTE_OPTIONS}
        checked = 0
        for model_id in reg.list_models():
            spec = reg.get(model_id)
            packages = [d.repo_id for d in spec.downloads if d.engine == "mlx-gen"]
            for task in ("image_to_image", "text_to_image"):
                listed = [name for name in (spec.tasks.get(task).params if task in spec.tasks else {}) if name in options]
                rows = []
                for repo_id in packages:
                    try:
                        capabilities = get_model_capabilities(model=repo_id).capabilities
                    except Exception:
                        continue
                    rows.extend(
                        vars(row) for row in capabilities
                        if row.public_task == task.replace("_", "-") and row.id in mlx_gen_routes.IMAGE_ROUTES
                    )
                if not rows:
                    continue
                for name in listed:
                    with self.subTest(model=model_id, task=task, option=name):
                        flag = options[name].flag
                        taken = any(row.get(flag) for row in rows) if flag else any(
                            mlx_gen_routes.allows_image_count(row, 2) for row in rows
                        )
                        self.assertTrue(taken, f"{model_id} lists {name} for {task}; no served row takes it")
                        checked += 1
        self.assertGreater(checked, 10)


class TestFinding10ControlNet(unittest.TestCase):
    def test_a_control_model_that_is_not_repo_file_is_refused_naming_the_row(self):
        from abstractvision.backends import mlx_gen_routes
        from abstractvision.backends.mflux import MFluxVisionBackend
        from abstractvision.errors import CapabilityNotSupportedError

        for value in ("InstantX/Qwen-Image-ControlNet-Union", ":file.safetensors", None):
            with self.subTest(value=value):
                selected = mlx_gen_routes.SelectedImageRoute(
                    capability_id="qwen.control",
                    route=mlx_gen_routes.IMAGE_ROUTES["qwen.control"],
                    row={},
                    selector="AbstractFramework/qwen-image-8bit",
                    control_model=value,
                )
                with self.assertRaisesRegex(CapabilityNotSupportedError, "qwen.control row") as caught:
                    MFluxVisionBackend._check_controlnet_weights(selected)
                self.assertEqual(caught.exception.field, "control_image")

    def test_the_class_comes_from_the_public_export(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        from abstractvision.backends.mflux import _lazy_import_mflux_qwen_controlnet

        exported = object()
        public = {"mflux.models.qwen.variants": SimpleNamespace(QwenImageControlNet=exported)}
        with patch.dict(sys.modules, public):
            self.assertIs(_lazy_import_mflux_qwen_controlnet(), exported)


class TestFinding11ControlImagePath(unittest.TestCase):
    def test_the_key_is_left_out_like_the_other_temporary_paths(self):
        from abstractvision.backends.mflux import (
            _TEMPORARY_PATH_METADATA_KEYS,
            _public_engine_metadata,
        )

        self.assertIn("controlnet_image_path", _TEMPORARY_PATH_METADATA_KEYS)
        self.assertEqual(_public_engine_metadata({"controlnet_image_path": None, "seed": 1}), {"seed": 1})


class TestFindings12And13ExtraReferences(RouteCase):
    """`extra["reference_images"]` keeps 0.3.37's reading; only what it keeps is counted and passed."""

    @needs_planner
    def test_shapes_0_3_37_read(self):
        def generator():
            yield REF

        for value in ([REF, None], {REF}, generator(), [REF, ""], REF):
            with self.subTest(value=type(value).__name__):
                EVENTS.clear()
                self.edit(FLUX2, extra={"reference_images": value}, seed=1)
                name, kwargs, files = self.call()
                self.assertEqual(name, "Flux2KleinEdit")
                self.assertEqual([item["bytes"] for item in files["image_paths"]], [SOURCE, REF])

    def test_a_blank_item_alone_is_no_reference(self):
        self.edit(QWEN_EDIT_PLAIN, extra={"reference_images": [""]}, seed=1)
        name, kwargs, _files = self.call()
        self.assertEqual((name, len(kwargs["image_paths"])), ("QwenImageEdit", 1))

    def test_the_typed_field_stays_strict(self):
        for value in ([REF, None], [42]):
            with self.subTest(value=value):
                self.assert_refused(QWEN_EDIT, ValueError, reference_images=value)


class TestFinding15OneOptionList(unittest.TestCase):
    def test_the_edit_options_are_the_route_options_0_3_37_lacked_as_typed_fields(self):
        from abstractvision.backends import mlx_gen_routes
        from abstractvision.backends.base_backend import EDIT_REQUEST_FIELDS
        from abstractvision.backends.mflux import _CANVAS_OPTIONS

        names = [option.name for option in mlx_gen_routes.ROUTE_OPTIONS]
        # mask and control_image were typed request fields in 0.3.37 already; the others are 0.3.38's options.
        self.assertEqual(set(EDIT_REQUEST_FIELDS), set(names) - {"mask", "control_image"})
        self.assertEqual(_CANVAS_OPTIONS, ("outpaint_padding", "reframe_padding", "outpaint_fill", "outpaint_fill_color"))
        self.assertEqual(mlx_gen_routes.CANVAS_BOXES, ("outpaint_padding", "reframe_padding"))



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


class TestMlxCacheCap(RouteCase):
    """In every mode, MLX-Gen's default cache cap is applied before any model class is built."""

    def setUp(self):
        super().setUp()
        from unittest.mock import patch

        from _mlx_gen_route_fixtures import snapshot, stand_in

        self.snapshots["AbstractFramework/seedvr2-3b-8bit"] = snapshot(
            self.snapshots[FLUX2].parents[2], "AbstractFramework/seedvr2-3b-8bit"
        )
        self.seedvr2 = stand_in("SeedVR2")
        for name, value in (
            ("_apply_mlx_gen_default_cache_limit", None),
            ("_lazy_import_mflux_seedvr2", (_SeedVR2Config, self.seedvr2, _ScaleFactor, object)),
        ):
            stub = (
                patch(f"abstractvision.backends.mflux.{name}", side_effect=lambda: EVENTS.append(("cap",)))
                if value is None
                else patch(f"abstractvision.backends.mflux.{name}", return_value=value)
            )
            stub.start()
            self.addCleanup(stub.stop)

    def order(self):
        return [event[0] if event[0] == "cap" else f"{event[0]}:{event[1]}" for event in EVENTS if event[0] in ("cap", "init")]

    def test_each_loader_family_in_default_mode(self):
        from abstractvision.types import (
            ImageGenerationRequest,
            ImageUpscaleRequest,
            VideoGenerationRequest,
        )

        runs = (
            (FLUX2, lambda b: b.generate_image(ImageGenerationRequest(prompt="p", steps=2)), "Flux2Klein"),
            (WAN_T2V, lambda b: b.generate_video(VideoGenerationRequest(prompt="p", width=480, height=240, num_frames=17)), "Wan2_2_TI2V"),
            ("AbstractFramework/seedvr2-3b-8bit", lambda b: b.upscale_image(ImageUpscaleRequest(image=SOURCE)), "SeedVR2"),
        )
        for model, run, cls in runs:
            with self.subTest(model=model):
                EVENTS.clear()
                run(self.backend(model))
                self.assertEqual(self.order(), ["cap", f"init:{cls}"])

    def test_a_preload(self):
        self.backend(WAN_T2V).preload()
        self.assertEqual(self.order(), ["cap", "init:Wan2_2_TI2V"])


@needs_planner
class TestMlxCacheCapRule(unittest.TestCase):
    """The cap is MLX-Gen's own once-per-process default: a limit the host set is left alone."""

    def test_a_host_limit_is_kept_and_an_unset_one_gets_the_default(self):
        import mlx.core as mx
        from mflux.utils.runtime_memory import RuntimeMemory

        from abstractvision.backends import mflux as backend_module

        total = RuntimeMemory.total_physical_memory_bytes()
        saved_state = RuntimeMemory._cache_limit_state
        saved_limit = mx.set_cache_limit(total)
        try:
            host = 3 * 1024**3
            RuntimeMemory._cache_limit_state = "unset"
            mx.set_cache_limit(host)
            backend_module._apply_mlx_gen_default_cache_limit()
            self.assertEqual(mx.set_cache_limit(host), host)
            # MLX's untouched library default (about the machine's RAM) gets mlx-gen's default cap.
            RuntimeMemory._cache_limit_state = "unset"
            mx.set_cache_limit(total)
            backend_module._apply_mlx_gen_default_cache_limit()
            self.assertEqual(mx.set_cache_limit(total), RuntimeMemory.default_cache_limit_bytes())
        finally:
            RuntimeMemory._cache_limit_state = saved_state
            mx.set_cache_limit(saved_limit)


class TestCloseOutRowlessPreload(RouteCase):
    """A model with no task has nothing to preload: refused before any load, no text_to_image fallback."""

    def test_a_rowless_vace_folder(self):
        from abstractvision.errors import CapabilityNotSupportedError

        backend = self.folder_backend("wan2.1-vace-1.3b")
        self.assertEqual(backend.get_capabilities().supported_tasks, [])
        with self.assertRaisesRegex(CapabilityNotSupportedError, "publishes no route for this model"):
            backend.preload()
        self.assert_nothing_loaded(backend)

    def test_a_model_with_tasks_keeps_its_preload_task(self):
        from abstractvision.backends.mflux import (
            _MFLUX_MODELS,
            _PRELOAD_TASK_ORDER,
            MFluxBackendConfig,
            MFluxVisionBackend,
        )

        for base_model in sorted(_MFLUX_MODELS):
            with self.subTest(model=base_model):
                backend = MFluxVisionBackend(config=MFluxBackendConfig(base_model=base_model))
                tasks = backend._supported_task_names()
                if not tasks:
                    continue
                expected = next((task for task in _PRELOAD_TASK_ORDER if task in tasks), "text_to_image")
                self.assertIn(expected, tasks)
                self.assertEqual(backend._preload_task(), expected)


class TestCloseOutAdapterTasks(unittest.TestCase):
    """The tasks listed for an adapter follow the rows of the model listed, not the configured model's
    (the adapters of a model other than the configured one are listed with `model=`)."""

    def test_a_z_image_adapter_under_two_configured_models(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        def row(task):
            return SimpleNamespace(public_task=task, default_for_task=True, supports_lora=False)

        def rows_for(z_image_takes_images):
            def get_model_capabilities(*, model, base_model=None):
                text = str(model).lower()
                if "z-image" in text:
                    tasks = ("text-to-image", "image-to-image") if z_image_takes_images else ("text-to-image",)
                elif "qwen-image" in text:
                    tasks = ("text-to-image", "image-to-image")
                else:
                    tasks = ("text-to-image",)
                return SimpleNamespace(capabilities=tuple(row(task) for task in tasks))

            return get_model_capabilities

        with tempfile.TemporaryDirectory() as cache_td:
            repo_dir = Path(cache_td) / "models--example--z-image-turbo-style-lora"
            snap = repo_dir / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            (snap / "style.safetensors").write_bytes(b"adapter")
            (snap / "README.md").write_text("---\nbase_model: Tongyi-MAI/Z-Image-Turbo\n---\n", encoding="utf-8")
            (repo_dir / "refs").mkdir()
            (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
            for z_image_takes_images, expected in ((True, ["image_to_image", "text_to_image"]), (False, ["text_to_image"])):
                for configured in ("AbstractFramework/qwen-image-8bit", "AbstractFramework/flux.2-klein-4b-8bit"):
                    with self.subTest(z_image_rows_take_images=z_image_takes_images, configured=configured):
                        backend = MFluxVisionBackend(config=MFluxBackendConfig(model=configured))
                        with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True), patch(
                            "abstractvision.backends.mflux._lazy_import_mlx_gen_task_inference",
                            return_value=(rows_for(z_image_takes_images), None),
                        ):
                            adapters = list(backend.list_provider_adapters(model="AbstractFramework/z-image-turbo-8bit"))
                        self.assertEqual(len(adapters), 1, adapters)
                        self.assertEqual(sorted(adapters[0].compatible_tasks), expected)


class TestCloseOutTypedReferenceItems(RouteCase):
    """The typed `reference_images` checks its items; on a row-less FLUX.2 folder no decode check comes first."""

    def test_on_a_rowless_flux2_folder(self):
        from abstractvision.types import ImageEditRequest

        for value in ([REF, None], [42], [REF, ""], ["  "]):
            with self.subTest(value=value):
                backend = self.folder_backend("flux2-klein-9b")
                with self.assertRaisesRegex(ValueError, "reference_images items"):
                    backend.edit_image(ImageEditRequest(prompt="a red kite", image=SOURCE, reference_images=value))
                self.assert_nothing_loaded(backend)


@needs_planner
class TestNegativePromptFollowsTheRow(RouteCase):
    """On a model the engine publishes rows for, the route's row decides whether the negative prompt is passed."""

    ROUTES = (
        ("edit", "Flux2KleinEdit", {}),
        ("multi-reference", "Flux2KleinEdit", {"reference_images": [REF]}),
        ("latent", "Flux2Klein", {"image_strength": 0.5}),
        ("text-to-image", "Flux2Klein", None),
    )

    def run_route(self, backend, fields, **guidance):
        from abstractvision.types import ImageEditRequest, ImageGenerationRequest

        common = {"prompt": "a red kite", "negative_prompt": "blur", "seed": 1, **guidance}
        if fields is None:
            backend.generate_image(ImageGenerationRequest(**common))
        else:
            backend.edit_image(ImageEditRequest(image=SOURCE, **common, **fields))
        return self.call()

    def test_flux2_klein_base_takes_it_on_every_route(self):
        for route, class_name, fields in self.ROUTES:
            with self.subTest(route=route):
                EVENTS.clear()
                name, kwargs, _files = self.run_route(self.backend(FLUX2_BASE), fields, guidance_scale=4.0)
                self.assertEqual((name, kwargs.get("negative_prompt")), (class_name, "blur"))

    def test_distilled_flux2_klein_drops_it_silently(self):
        for route, class_name, fields in self.ROUTES:
            with self.subTest(route=route):
                EVENTS.clear()
                name, kwargs, _files = self.run_route(self.backend(FLUX2), fields)
                self.assertEqual(name, class_name)
                self.assertNotIn("negative_prompt", kwargs)

    def test_a_rowless_klein_base_folder_keeps_the_model_definition(self):
        for route, fields in (("edit", {}), ("text-to-image", None)):
            with self.subTest(route=route):
                EVENTS.clear()
                _name, kwargs, _files = self.run_route(
                    self.folder_backend("flux2-klein-base-4b"), fields, guidance_scale=4.0
                )
                self.assertNotIn("negative_prompt", kwargs)


class TestPublishedNegativePrompt(RouteCase):
    """`parameter_constraints["negative_prompt"]` says what the runs do (the rows' rule; without rows, 0.3.37's)."""

    def constraint(self, model, task):
        (listed,) = [m for m in self.backend(model).list_provider_models(task=task) if m.id == model]
        return listed.raw["parameter_constraints"].get("negative_prompt")

    @needs_planner
    def test_models_the_engine_publishes_rows_for(self):
        for model, expected in ((FLUX2_BASE, None), (Z_IMAGE_TURBO, None), (FLUX2, {"supported": False})):
            for task in ("text_to_image", "image_to_image"):
                with self.subTest(model=model, task=task):
                    self.assertEqual(self.constraint(model, task), expected)

    def test_a_rowless_folder_keeps_the_model_definition(self):
        from abstractvision.backends.mflux import _MFLUX_MODELS, _mflux_parameter_metadata

        for base_model in ("flux2-klein-base-4b", "z-image-turbo", "flux2-klein-9b"):
            with self.subTest(model=base_model):
                rows = self.folder_backend(base_model)._current_route_capabilities()
                metadata = _mflux_parameter_metadata(_MFLUX_MODELS[base_model], route_capabilities=rows)
                self.assertEqual(metadata["parameter_constraints"]["negative_prompt"], {"supported": False})

    def test_routes_of_one_task_that_disagree(self):
        from abstractvision.backends.mflux import _MFLUX_MODELS, _mflux_parameter_metadata

        rows = {
            "text_to_image": [{"public_task": "text-to-image", "supports_negative_prompt": True}],
            "image_to_image": [
                {"public_task": "image-to-image", "supports_negative_prompt": True},
                {"public_task": "image-to-image", "supports_negative_prompt": False},
            ],
        }
        constraint = _mflux_parameter_metadata(_MFLUX_MODELS["flux2-klein-base-4b"], route_capabilities=rows)[
            "parameter_constraints"
        ]["negative_prompt"]
        self.assertTrue(constraint["supported"])
        self.assertIn("image_to_image", constraint["description"])


if __name__ == "__main__":
    unittest.main()
