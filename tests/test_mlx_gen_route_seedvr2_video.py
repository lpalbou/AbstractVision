"""SeedVR2 video restoration (0.3.38 step 9): MLX-Gen's plan (`SeedVR2VideoRestorePlan.for_video`) and
`SeedVR2.restore_video_to_path`, with stand-ins that record what they were given.
"""

from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from _mlx_gen_route_fixtures import EVENTS, FLUX2, SOURCE, RouteCase, needs_planner, snapshot
from _mlx_gen_versions import setUpModule, tearDownModule  # noqa: F401

SEEDVR2 = "AbstractFramework/seedvr2-3b-8bit"
CLIP = b"\x00\x00\x00\x18ftypmp42 a synthetic clip"
CACHE_LIMIT = 8 * 1000**3
# The engine's default cap on a 32 GB Mac (RAM / 8), below the plan's 8 GB.
DEFAULT_CAP = 4 * 1024**3


class _ModelConfig:
    @staticmethod
    def from_name(model_name, base_model=None):
        return f"config:{model_name}"


class _ScaleFactor:
    def __init__(self, text):
        self.text = text

    @staticmethod
    def parse(value):
        return _ScaleFactor(str(value))

    def __eq__(self, other):
        return isinstance(other, _ScaleFactor) and other.text == self.text

    def __str__(self):
        return self.text


class _Refused(ValueError):
    def __init__(self, message, *, reason, warnings=()):
        super().__init__(message)
        self.reason = reason
        self.warnings = warnings or (message,)


class _Plan:
    refuse = None  # (reason, message) the next plan raises
    error = None  # a plain exception the next plan raises

    @classmethod
    def for_video(cls, video_path, **kwargs):
        EVENTS.append(("plan", Path(video_path).is_file(), dict(kwargs)))
        if cls.error is not None:
            raise cls.error
        if cls.refuse is not None:
            reason, message = cls.refuse
            raise _Refused(message, reason=reason)
        resolution = kwargs.get("resolution")
        return SimpleNamespace(
            resolution=_ScaleFactor("1x") if resolution is None else resolution,
            start_seconds=kwargs.get("start_seconds", 0.0),
            max_frames=kwargs.get("max_frames"),
            effective_chunk_size=21,
            effective_chunk_overlap=8,
            enforce_memory_budget=True,
            low_ram_required=True,
            keep_transformer=True,
            cache_limit_bytes=CACHE_LIMIT,
        )


class _MemorySaver:
    limit = None  # the case's limit tracker, set by setUp

    def __init__(self, model, keep_transformer=True, cache_limit_bytes=None, args=None):
        self.keep_transformer = keep_transformer
        self.cache_limit_bytes = cache_limit_bytes
        model.tiling_config = "tiled by the saver"
        # As mlx-gen's MemorySaver: it sets the limit it is given.
        _MemorySaver.limit(cache_limit_bytes)


class _Callbacks:
    def __init__(self):
        self.subscribers = []

    def subscribe_progress(self, callback, *, task=None):
        self.subscribers.append(callback)
        return lambda: self.subscribers.remove(callback)

    def register(self, callback):
        EVENTS.append(("register", type(callback).__name__, callback.keep_transformer, callback.cache_limit_bytes))


class SeedVR2:
    during_call = None
    fail_with = None
    limit_in_force = None  # the case's limit reader, set by setUp

    def __init__(self, **kwargs):
        self.callbacks = _Callbacks()
        self.tiling_config = None
        self.transformer = object()
        self.vae = object()
        EVENTS.append(("init", "SeedVR2", dict(kwargs)))
        EVENTS.append(("limit at load", SeedVR2.limit_in_force()))

    def generate_image(self, **kwargs):
        from PIL import Image

        EVENTS.append(("call", "SeedVR2.generate_image", dict(kwargs), {"tiling_config": self.tiling_config}))
        return SimpleNamespace(image=Image.new("RGB", (8, 8)), _get_metadata=lambda: {})

    def restore_video_to_path(self, **kwargs):
        source = Path(kwargs["video_path"])
        EVENTS.append(
            (
                "call",
                "SeedVR2.restore_video_to_path",
                dict(kwargs),
                {"source": source.read_bytes(), "limit": SeedVR2.limit_in_force()},
            )
        )
        if SeedVR2.during_call is not None:
            SeedVR2.during_call()
        for chunk in range(2):
            for callback in list(self.callbacks.subscribers):
                callback({"phase": "denoise", "step": chunk + 1, "total_steps": 2})
        if SeedVR2.fail_with is not None:
            raise SeedVR2.fail_with
        output = Path(kwargs["output_path"])
        output.write_bytes(b"restored " + source.read_bytes())
        sidecar = {
            "model": "/Users/someone/models/seedvr2",
            "video_path": str(source),
            "video_health": {"file": {"source": str(output)}},
            "temporal_chunk_count": 2,
        }
        output.with_suffix(".metadata.json").write_text(json.dumps(sidecar), encoding="utf-8")
        return output


class SeedVR2VideoCase(RouteCase):
    def setUp(self):
        super().setUp()
        self.snapshots[SEEDVR2] = snapshot(self.snapshots[FLUX2].parents[2], SEEDVR2)
        _Plan.refuse = _Plan.error = None
        SeedVR2.during_call = SeedVR2.fail_with = None
        # MLX's cache limit: the engine's default cap in force, and no host limit unless a case sets one.
        self.limit = [DEFAULT_CAP]
        self.host_limit = None

        def set_limit(value):
            previous, self.limit[0] = self.limit[0], int(value)
            return previous

        SeedVR2.limit_in_force = lambda: self.limit[0]
        _MemorySaver.limit = set_limit
        for target, kwargs in (
            ("abstractvision.backends.mflux._lazy_import_mflux_seedvr2", {"return_value": (_ModelConfig, SeedVR2, _ScaleFactor, object)}),
            ("abstractvision.backends.mlx_gen_seedvr2_video.lazy_import_plan", {"return_value": (_Plan, _Refused)}),
            ("abstractvision.backends.mflux._lazy_import_mlx_gen_memory_saver", {"return_value": _MemorySaver}),
            ("abstractvision.backends.mflux._set_mlx_cache_limit", {"side_effect": set_limit}),
            ("abstractvision.backends.mflux._host_managed_mlx_cache_limit", {"side_effect": lambda: self.host_limit}),
        ):
            stub = patch(target, **kwargs)
            stub.start()
            self.addCleanup(stub.stop)
            if target.endswith("_host_managed_mlx_cache_limit"):
                self.host_stub = stub

    def upscale(self, video=CLIP, backend=None, **fields):
        from abstractvision.types import VideoUpscaleRequest

        backend = backend or self.backend(SEEDVR2)
        return backend, backend.upscale_video(VideoUpscaleRequest(video=video, **fields))

    def restore_calls(self):
        return [event for event in EVENTS if event[:2] == ("call", "SeedVR2.restore_video_to_path")]

    def assert_refused_before_load(self, error, *, field="unset", supported="unset", **fields):
        backend = self.backend(SEEDVR2)
        with self.assertRaises(error) as caught:
            self.upscale(backend=backend, **fields)
        if field != "unset":
            self.assertEqual(caught.exception.field, field)
        if supported != "unset":
            self.assertEqual(caught.exception.supported, supported)
        self.assertEqual(self.inits(), [])
        self.assertFalse(backend.has_loaded_model())
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        return caught.exception


class TestTheCall(SeedVR2VideoCase):
    def test_every_value_unset(self):
        _backend, asset = self.upscale(seed=7)
        (plan,) = [event for event in EVENTS if event[0] == "plan"]
        # The plan gets the clip and the model the run loads; every value the request leaves unset is the plan's own.
        (init,) = [event[2] for event in EVENTS if event[0] == "init"]
        self.assertEqual(
            plan[1:],
            (True, {"model_config": init["model_config"], "model_path": init["model_path"], "weights_resident": False}),
        )
        self.assertEqual(init["model_path"], str(self.snapshots[SEEDVR2]))
        (call,) = self.restore_calls()
        kwargs = call[2]
        run_dir = Path(kwargs["video_path"]).parent
        self.assertEqual(call[3]["source"], CLIP)
        self.assertEqual(
            kwargs,
            {
                "seed": 7,
                "video_path": run_dir / "source.mp4",
                "resolution": _ScaleFactor("1x"),
                "start_seconds": 0.0,
                "max_frames": None,
                "output_path": run_dir / "video.mp4",
                "export_json_metadata": True,
                "overwrite": True,
                "temporal_chunk_size": 21,
                "temporal_chunk_overlap": 8,
                "enforce_memory_budget": True,
            },
        )
        self.assertEqual((asset.media_type, asset.mime_type, asset.data), ("video", "video/mp4", b"restored " + CLIP))
        self.assertIsNone(asset.audio)
        self.assertEqual(asset.metadata["task"], "video_upscale")
        self.assertEqual(asset.metadata["mlx_gen"]["temporal_chunk_count"], 2)
        self.assert_metadata_clean(asset, str(run_dir))
        self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_every_value_set(self):
        self.upscale(resolution=720, softness=1.5, start_seconds=1.25, max_frames=17, seed=3)
        (plan,) = [event for event in EVENTS if event[0] == "plan"]
        self.assertEqual(
            {key: plan[2][key] for key in ("resolution", "start_seconds", "max_frames")},
            {"resolution": 720, "start_seconds": 1.25, "max_frames": 17},
        )
        (call,) = self.restore_calls()
        # Softness is clamped to [0, 1] as for images; the plan's values reach the engine.
        self.assertEqual(
            {key: call[2][key] for key in ("resolution", "softness", "start_seconds", "max_frames")},
            {"resolution": 720, "softness": 1.0, "start_seconds": 1.25, "max_frames": 17},
        )

    def test_the_result_is_mp4_whatever_the_source(self):
        # MLX-Gen's writer encodes H.264 for MP4 (its CLI writes video.mp4 for every source); WebM refuses H.264.
        for suffix in (".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", "", None):
            with self.subTest(suffix=suffix):
                EVENTS.clear()
                video = CLIP
                if suffix is not None:
                    video = self.tmp_root.parent / f"clip{suffix}"
                    video.write_bytes(CLIP)
                    self.addCleanup(video.unlink)
                _backend, asset = self.upscale(video=video, seed=1)
                (call,) = self.restore_calls()
                if suffix is not None:
                    self.assertEqual(call[2]["video_path"], video)
                self.assertEqual((call[2]["output_path"].name, asset.mime_type), ("video.mp4", "video/mp4"))
                self.assert_metadata_clean(asset, str(call[2]["output_path"].parent))
                self.assertEqual(list(self.tmp_root.iterdir()), [])

    def test_resolution_is_sent_as_read_and_drop_audio_only_when_set(self):
        for value, sent in (("1x", _ScaleFactor("1x")), (1.0, 1), ("+48", 48), (720, 720)):
            with self.subTest(resolution=value):
                EVENTS.clear()
                self.upscale(resolution=value, seed=1)
                (plan,) = [event for event in EVENTS if event[0] == "plan"]
                self.assertEqual(plan[2]["resolution"], sent)
        for drop_audio in (True, False):
            with self.subTest(drop_audio=drop_audio):
                EVENTS.clear()
                self.upscale(drop_audio=drop_audio, seed=1)
                (call,) = self.restore_calls()
                self.assertIs(call[2]["drop_audio"], drop_audio)
        EVENTS.clear()
        self.upscale(seed=1)
        (call,) = self.restore_calls()
        self.assertNotIn("drop_audio", call[2])

    @needs_planner  # the load-time widths a model accepts are the engine's
    def test_the_quantize_of_a_source_snapshot(self):
        # The 16-bit source loads at 8 bits unless the request names a width, as for an image upscale.
        source = "ByteDance-Seed/SeedVR2-3B"
        self.snapshots[source] = snapshot(self.snapshots[FLUX2].parents[2], source)
        backend = self.backend(source)
        self.upscale(backend=backend, seed=1)
        (init,) = [event[2] for event in EVENTS if event[0] == "init"]
        self.assertEqual(init["quantize"], 8)


class TestMemory(SeedVR2VideoCase):
    def limits_seen(self):
        loads = [event[1] for event in EVENTS if event[0] == "limit at load"]
        savers = [event[3] for event in EVENTS if event[0] == "register"]
        runs = [event[3]["limit"] for event in self.restore_calls()]
        return loads, savers, runs

    def test_the_engine_profile_in_both_modes_and_the_instance_dropped(self):
        for mode in ("default", "low"):
            with self.subTest(mode=mode):
                EVENTS.clear()
                backend, asset = self.upscale(memory_mode=mode, seed=1)
                # No host limit, the default cap below 8 GB: the plan's limit from before the load to the end.
                self.assertEqual(self.limits_seen(), ([CACHE_LIMIT], [CACHE_LIMIT], [CACHE_LIMIT]))
                self.assertEqual(self.limit[0], DEFAULT_CAP)
                self.assertEqual((asset.metadata["memory_mode"], asset.metadata["low_ram"]), (mode, True))
                self.assertFalse(backend.has_loaded_model())

    def test_a_host_limit_is_never_raised(self):
        host = 2 * 1000**3
        self.limit[0] = host
        self.host_limit = host
        self.upscale(seed=1)
        self.assertEqual(self.limits_seen(), ([host], [host], [host]))
        self.assertEqual(self.limit[0], host)

    @needs_planner
    def test_the_engines_answer_decides_what_is_a_host_limit(self):
        # mlx-gen's accessor, through the backend's helper: its default cap (4 GiB here, below 8 GB) is not a
        # host's, so the plan's limit holds; a host's 4 GiB holds throughout.
        from mflux.utils.runtime_memory import RuntimeMemory

        self.host_stub.stop()
        for answer, expected in ((None, CACHE_LIMIT), (DEFAULT_CAP, DEFAULT_CAP)):
            with self.subTest(answer=answer):
                EVENTS.clear()
                with patch.object(RuntimeMemory, "host_managed_cache_limit_bytes", return_value=answer):
                    self.upscale(seed=1)
                self.assertEqual(self.limits_seen(), ([expected], [expected], [expected]))
        self.host_stub.start()

    def test_a_preload_that_read_nothing_is_not_resident_weights(self):
        def resident():
            return [event[2]["weights_resident"] for event in EVENTS if event[0] == "plan"]

        for kwargs, expected in (({"memory_mode": "low"}, False), ({"warm_up": False}, False)):
            with self.subTest(**kwargs):
                EVENTS.clear()
                backend = self.backend(SEEDVR2)
                backend.preload(task="video_upscale", **kwargs)
                self.assertTrue(backend.has_loaded_model())
                self.upscale(backend=backend, seed=1)
                self.assertEqual((resident(), self.inits()), ([expected], ["SeedVR2"]))

    @needs_planner  # the load-time widths a model accepts are the engine's
    def test_the_plan_counts_the_weights_as_loaded_when_the_run_reuses_the_instance(self):
        from abstractvision.types import ImageUpscaleRequest

        def resident():
            return [event[2]["weights_resident"] for event in EVENTS if event[0] == "plan"]

        backend = self.backend(SEEDVR2)
        self.upscale(backend=backend, seed=1)
        self.assertEqual(resident(), [False])
        EVENTS.clear()
        backend.upscale_image(ImageUpscaleRequest(image=SOURCE, seed=1))
        self.upscale(backend=backend, seed=1)
        self.assertEqual((resident(), self.inits()), ([True], ["SeedVR2"]))
        # Another load-time width is another instance: the run releases the held one (a source snapshot, which
        # loads at 8 bits by default).
        source = "ByteDance-Seed/SeedVR2-3B"
        self.snapshots[source] = snapshot(self.snapshots[FLUX2].parents[2], source)
        backend = self.backend(source)
        backend.upscale_image(ImageUpscaleRequest(image=SOURCE, seed=1))
        EVENTS.clear()
        self.upscale(backend=backend, seed=1, quantize=4)
        self.assertEqual(resident(), [False])

    def test_an_image_upscale_after_a_video_run_gets_a_fresh_instance(self):
        from abstractvision.types import ImageUpscaleRequest

        backend, _asset = self.upscale(seed=1)
        backend.upscale_image(ImageUpscaleRequest(image=SOURCE, seed=1))
        self.assertEqual(self.inits(), ["SeedVR2", "SeedVR2"])
        (image_call,) = [event for event in EVENTS if event[:2] == ("call", "SeedVR2.generate_image")]
        self.assertEqual(image_call[3], {"tiling_config": None})

    def test_a_preload_reads_what_the_run_holds_and_the_run_reuses_then_drops_it(self):
        evaluated = []
        with patch(
            "abstractvision.backends.mflux._evaluate_parameters",
            side_effect=lambda model, parts: evaluated.append((type(model).__name__, parts)),
        ):
            backend = self.backend(SEEDVR2)
            backend.preload(task="video_upscale")
            self.assertEqual((evaluated, self.restore_calls()), ([("SeedVR2", ("transformer", "vae"))], []))
            self.upscale(backend=backend, seed=1)
        self.assertEqual(self.inits(), ["SeedVR2"])
        (plan,) = [event for event in EVENTS if event[0] == "plan"]
        self.assertTrue(plan[2]["weights_resident"])
        self.assertFalse(backend.has_loaded_model())

    def test_lowest_is_refused(self):
        self.assert_refused_before_load(Exception, field="memory_mode", memory_mode="lowest")


class TestRefusals(SeedVR2VideoCase):
    def test_the_plans_refusals_before_any_load(self):
        for reason, field in (
            ("scale_above_safe_profile", "resolution"),
            ("beyond_memory_profile", None),
            ("below_temporal_quality_floor", None),
        ):
            with self.subTest(reason=reason):
                from abstractvision.errors import CapabilityNotSupportedError

                _Plan.refuse = (reason, f"engine says {reason}")
                caught = self.assert_refused_before_load(
                    CapabilityNotSupportedError, field=field, supported=None, resolution="2x"
                )
                self.assertEqual(str(caught), f"engine says {reason}")

    def test_an_engine_argument_error_passes_through(self):
        _Plan.error = ValueError("start_seconds must be >= 0")
        caught = self.assert_refused_before_load(ValueError, start_seconds=-1.0)
        self.assertEqual(str(caught), "start_seconds must be >= 0")

    def test_values_of_the_wrong_type_and_unknown_extra_keys(self):
        from abstractvision.errors import CapabilityNotSupportedError

        for fields in (
            {"max_frames": "many"},
            {"start_seconds": "soon"},
            {"softness": "soft"},
            {"drop_audio": "yes"},
        ):
            with self.subTest(fields=fields):
                self.assert_refused_before_load(ValueError, **fields)
        caught = self.assert_refused_before_load(ValueError, resolution="huge")
        self.assertIn("video_upscale resolution", str(caught))
        self.assertIn("'1x'", str(caught))
        for key in ("vae_tiling", "temporal_chunk_size"):
            with self.subTest(key=key):
                self.assert_refused_before_load(
                    CapabilityNotSupportedError, field=key, supported=(), extra={key: 1}
                )
        self.assertEqual([event for event in EVENTS if event[0] == "plan"], [])

    def test_a_clip_the_reader_cannot_decode(self):
        class InvalidDataError(ValueError):
            pass

        InvalidDataError.__module__ = "av.error"

        def unreadable(video_path, **kwargs):
            raise InvalidDataError(f"[Errno 1094995529] Invalid data found when processing input: '{video_path}'")

        with patch.object(_Plan, "for_video", side_effect=unreadable):
            caught = self.assert_refused_before_load(ValueError)
        self.assertTrue(str(caught).startswith("video is not a readable clip"), caught)
        self.assertNotIn("source.mp4", str(caught))
        self.assertNotIn(str(self.tmp_root), str(caught))

    def test_a_model_that_is_not_seedvr2(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoUpscaleRequest

        backend = self.backend(FLUX2)
        with self.assertRaises(CapabilityNotSupportedError):
            backend.upscale_video(VideoUpscaleRequest(video=CLIP))
        self.assertEqual(self.inits(), [])


class TestCancel(SeedVR2VideoCase):
    def test_a_cancel_before_the_run_and_during_it(self):
        from abstractvision.errors import GenerationCancelledError

        cancel = threading.Event()
        cancel.set()
        self.assert_refused_before_load(GenerationCancelledError, cancel_event=cancel)

        cancel = threading.Event()
        SeedVR2.during_call = cancel.set
        backend = self.backend(SEEDVR2)
        with self.assertRaises(GenerationCancelledError):
            self.upscale(backend=backend, cancel_event=cancel, seed=1)
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        self.assertFalse(backend.has_loaded_model())

    def test_an_engine_error_removes_the_directory(self):
        SeedVR2.fail_with = RuntimeError("boom")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.upscale(seed=1)
        self.assertEqual(list(self.tmp_root.iterdir()), [])


class TestPublication(SeedVR2VideoCase):
    def test_tasks_modes_and_catalog(self):
        from abstractvision.model_capabilities import VisionModelCapabilitiesRegistry

        backend = self.backend(SEEDVR2)
        self.assertEqual(backend.get_capabilities().supported_tasks, ["image_upscale", "video_upscale"])
        self.assertEqual(backend.supported_memory_modes("video_upscale"), ("default", "low"))
        registry = VisionModelCapabilitiesRegistry()
        for model in ("ByteDance-Seed/SeedVR2-3B", "ByteDance-Seed/SeedVR2-7B"):
            params = registry.get(model).tasks["video_upscale"].params
            self.assertEqual(params["resolution"]["default"], "1x")
            self.assertEqual(params["resolution"]["max_scale"], 1)
            self.assertEqual((params["softness"]["default"], params["start_seconds"]["default"]), (0.0, 0.0))
            self.assertEqual(params["max_frames"]["minimum"], 1)
        # The image upscale's flat defaults are unchanged; a video listing's are its task's.
        (listed,) = [m for m in backend.list_provider_models(task="image_upscale") if m.id == SEEDVR2]
        self.assertEqual(listed.raw["parameter_defaults"]["resolution"], "2x")
        (listed,) = [m for m in backend.list_provider_models(task="video_upscale") if m.id == SEEDVR2]
        defaults, constraints = listed.raw["parameter_defaults"], listed.raw["parameter_constraints"]
        self.assertEqual(
            {key: defaults.get(key) for key in ("resolution", "softness", "start_seconds", "memory_mode")},
            {"resolution": "1x", "softness": 0.0, "start_seconds": 0.0, "memory_mode": "default"},
        )
        self.assertNotIn("scale", defaults)
        self.assertNotIn("vae_tiling", defaults)
        self.assertEqual(constraints["resolution"], {"max_scale": 1})
        self.assertEqual(constraints["max_frames"], {"min": 1})
        self.assertEqual(constraints["quantize"], {"enum": [3, 4, 5, 6, 8, None]})
        self.assertEqual(constraints["memory_mode"], {"enum": ["default", "low"]})

    def test_the_plugin_method(self):
        from abstractvision.integrations.abstractcore_plugin import _normalize_residency_task

        source = self.tmp_root.parent / "clip.mp4"
        source.write_bytes(CLIP)
        self.addCleanup(source.unlink)
        backend = self.backend(SEEDVR2)
        progress = []
        data = self.capability(backend, SEEDVR2).upscale_video(
            str(source), resolution="1x", max_frames=17, seed=4, on_progress=progress.append
        )
        self.assertEqual(data, b"restored " + CLIP)
        (call,) = self.restore_calls()
        self.assertEqual((call[2]["video_path"], call[2]["max_frames"]), (source, 17))
        self.assertEqual(len(progress), 2)
        self.assertEqual(_normalize_residency_task("video-upscale"), "video_upscale")

    def test_the_plugin_refuses_a_misplaced_option_and_records_the_task(self):
        from abstractvision.errors import CapabilityNotSupportedError

        backend = self.backend(SEEDVR2)
        cap = self.capability(backend, SEEDVR2)
        recorded = []
        with patch.object(cap, "_record_request_run", side_effect=lambda binding, task: recorded.append(task)):
            with self.assertRaises(CapabilityNotSupportedError) as caught:
                cap.upscale_video(CLIP, soundscape="rain")
            self.assertEqual(caught.exception.field, "soundscape")
            cap.upscale_video(CLIP, seed=1)
        self.assertEqual(recorded, ["video_upscale"])

    def test_the_manager_checks_the_task(self):
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.vision_manager import VisionManager

        # The backend's published tasks, then the catalog's for the manager's model.
        with self.assertRaisesRegex(CapabilityNotSupportedError, "video_upscale"):
            VisionManager(backend=self.backend(FLUX2)).upscale_video(CLIP)
        with self.assertRaises(Exception):
            VisionManager(backend=self.backend(SEEDVR2), model_id="black-forest-labs/FLUX.2-klein-4B").upscale_video(CLIP)
        self.assertEqual(self.restore_calls(), [])
        out = VisionManager(backend=self.backend(SEEDVR2), model_id="ByteDance-Seed/SeedVR2-3B").upscale_video(
            CLIP, seed=1
        )
        self.assertEqual(out.data, b"restored " + CLIP)

    def test_other_backends_refuse_the_task(self):
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import VideoUpscaleRequest

        backend = OpenAICompatibleVisionBackend(
            config=OpenAICompatibleBackendConfig(base_url="http://127.0.0.1:9/v1", model_id="m")
        )
        with self.assertRaisesRegex(CapabilityNotSupportedError, "does not support video_upscale"):
            backend.upscale_video(VideoUpscaleRequest(video=CLIP))


if __name__ == "__main__":
    unittest.main()
