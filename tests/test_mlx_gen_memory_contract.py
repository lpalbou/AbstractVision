"""The mlx-gen surface behind memory_mode, checked against the installed package (no weights, no GPU work).

Skipped where mlx-gen is not installed (the base CI job).
"""

import importlib.util
import inspect
import unittest


def _mlx_gen_installed() -> bool:
    try:
        return importlib.util.find_spec("mflux.task_inference") is not None
    except (ImportError, ValueError):
        return False


@unittest.skipUnless(_mlx_gen_installed(), "mlx-gen is not installed")
class TestMlxGenMemoryContract(unittest.TestCase):
    def test_low_ram_is_a_universal_generate_option(self):
        from mflux.task_inference import UNIVERSAL_GENERATE_OPTIONS

        self.assertIn("--low-ram", UNIVERSAL_GENERATE_OPTIONS)

    def test_low_ram_cache_limit_matches_mlx_gen(self):
        from mflux.utils.runtime_memory import RuntimeMemory

        from abstractvision.backends.mflux import LOW_RAM_MLX_CACHE_LIMIT_BYTES

        self.assertEqual(LOW_RAM_MLX_CACHE_LIMIT_BYTES, RuntimeMemory.DEFAULT_LOW_RAM_CACHE_LIMIT_BYTES)

    def test_memory_saver_takes_the_arguments_the_backend_passes(self):
        from abstractvision.backends.mflux import _lazy_import_mlx_gen_memory_saver

        params = inspect.signature(_lazy_import_mlx_gen_memory_saver()).parameters
        for name in ("model", "keep_transformer", "cache_limit_bytes"):
            self.assertIn(name, params)
        # A batch flips keep_transformer on the registered saver before its last item.
        source = inspect.getsource(_lazy_import_mlx_gen_memory_saver())
        self.assertIn("self.keep_transformer = keep_transformer", source)
        self.assertIn("if not self.keep_transformer:", source)

    def test_wan_generate_video_takes_the_low_ram_keywords(self):
        from mflux.models.wan.variants import Wan2_2_TI2V

        params = inspect.signature(Wan2_2_TI2V.generate_video).parameters
        for name in (
            "release_denoisers_before_decode",
            "release_inactive_denoiser",
            "clear_cache_each_step",
            "clear_cache_each_transformer_block",
        ):
            self.assertIn(name, params)

    def test_guard_restores_mlx_gen_default_cap_not_the_library_default(self):
        import mlx.core as mx
        from mflux.utils.runtime_memory import RuntimeMemory

        from abstractvision.backends import mflux as backend_module

        total = RuntimeMemory.total_physical_memory_bytes()
        expected = RuntimeMemory.default_cache_limit_bytes()
        saved_state = RuntimeMemory._cache_limit_state
        saved_limit = mx.set_cache_limit(total)
        low = backend_module.LOW_RAM_MLX_CACHE_LIMIT_BYTES
        try:
            observed = {}
            # A fresh process: MLX's library default (about the machine's RAM), mlx-gen's
            # once-per-process hook not applied yet.
            RuntimeMemory._cache_limit_state = "unset"
            mx.set_cache_limit(total)
            token = backend_module._acquire_mlx_cache_limit(low)
            observed["low_inside"] = mx.set_cache_limit(low)
            backend_module._release_mlx_cache_limit(token)
            observed["low_after"] = mx.set_cache_limit(expected)
            # The next model load leaves mlx-gen's cap in place.
            RuntimeMemory.apply_default_cache_limit_once()
            observed["low_after_next_load"] = mx.set_cache_limit(expected)
            # A cap the operator set below the low-RAM limit is never raised, and is restored.
            operator = low // 2
            mx.set_cache_limit(operator)
            token = backend_module._acquire_mlx_cache_limit(low)
            observed["operator_inside"] = mx.set_cache_limit(operator)
            backend_module._release_mlx_cache_limit(token)
            observed["operator_after"] = mx.set_cache_limit(operator)
            print(f"\ncache limits observed (total RAM {total}): default cap {expected}, {observed}")
            self.assertEqual(observed["low_inside"], low)
            self.assertEqual(observed["low_after"], expected)
            self.assertEqual(observed["low_after_next_load"], expected)
            self.assertLess(observed["low_after"], total)
            self.assertEqual((observed["operator_inside"], observed["operator_after"]), (operator, operator))
        finally:
            RuntimeMemory._cache_limit_state = saved_state
            mx.set_cache_limit(saved_limit)

    def test_route_rows_publish_block_streaming(self):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        backend = MFluxVisionBackend(config=MFluxBackendConfig())
        for selector in (
            "AbstractFramework/flux.2-klein-4b-8bit",
            "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
        ):
            rows = [row for rows in backend._route_capabilities_for_model(selector).values() for row in rows]
            self.assertTrue(rows, selector)
            self.assertTrue(all("supports_block_streaming" in row for row in rows), selector)
            # These routes publish no block streaming, so "lowest" is refused on them; it is
            # accepted for MiniMax-H3 (TestMlxGenMiniMaxH3Contract).
            self.assertFalse(any(row["supports_block_streaming"] for row in rows), selector)
        self.assertEqual(tuple(backend.supported_memory_modes("text_to_image")), ("default", "low"))
        minimax_rows = backend._route_capabilities_for_model("minimax-h3-turbo-544p")
        self.assertTrue(
            all(row["supports_block_streaming"] for rows in minimax_rows.values() for row in rows)
        )


@unittest.skipUnless(_mlx_gen_installed(), "mlx-gen is not installed")
class TestMlxGenMiniMaxH3Contract(unittest.TestCase):
    """The MiniMax-H3 surface the backend drives, checked against the installed mlx-gen."""

    def _backend(self, base_model="minimax-h3-turbo-544p", model="MiniMaxAI/MiniMax-H3"):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        return MFluxVisionBackend(config=MFluxBackendConfig(model=model, base_model=base_model))

    def test_real_rows_make_lowest_honourable_and_follow_the_schedule(self):
        backend = self._backend()
        self.assertEqual(backend.supported_memory_modes("text_to_video"), ("default", "low", "lowest"))
        self.assertEqual(backend.supported_memory_modes("image_to_video"), ("default", "low", "lowest"))
        rows = backend.describe_routes(task="text_to_video")
        self.assertTrue(rows[0]["supports_block_streaming"])
        self.assertTrue(rows[0]["generates_audio"])
        self.assertEqual((rows[0]["default_steps"], rows[0]["default_width"], rows[0]["default_height"]), (8, 960, 544))
        self.assertTrue(any(run["stream_blocks"] for run in rows[0]["measured_runs"]))
        base = self._backend(base_model=None).describe_routes(task="text_to_video")
        self.assertEqual(base[0]["default_steps"], 50)

    def test_frame_grid_matches_mlx_gen(self):
        from mflux.models.minimax_h3.latent_creator.h3_layout import align_num_frames, valid_frame_counts

        from abstractvision.backends.mflux import _h3_aligned_frames, _h3_frame_counts

        row = self._backend().describe_routes(task="text_to_video")[0]
        self.assertEqual(list(_h3_frame_counts(row)), list(valid_frame_counts()))
        for requested in (1, 124, 125, 140, 141, 200, 345):
            self.assertEqual(_h3_aligned_frames(requested, row), align_num_frames(requested), requested)

    def test_constructor_and_generate_video_take_what_the_backend_passes(self):
        from mflux.models.minimax_h3.variants.minimax_h3 import MiniMaxH3

        init = inspect.signature(MiniMaxH3.__init__).parameters
        for name in ("model_config", "model_path", "quantize", "lora_paths", "lora_scales", "stream_blocks"):
            self.assertIn(name, init)
        generate = inspect.signature(MiniMaxH3.generate_video).parameters
        for name in (
            "seed", "prompt", "width", "height", "num_frames", "num_inference_steps", "video_shift",
            "audio_shift", "image_path", "soundscape", "music", "generate_audio", "progress_callback",
            "release_transformer",
        ):
            self.assertIn(name, generate)

    def test_memory_peaks_are_the_package_width_runs_read_without_a_model(self):
        import time
        from unittest.mock import patch

        from mflux.task_inference import MINIMAX_H3_MODES, MINIMAX_H3_SMALLEST_TESTED_MACHINE

        from abstractvision import model_inventory
        from abstractvision.backends.mflux import MLX_GEN_MEMORY_MODE_FLAGS

        never = AssertionError("listing peaks must not build a model")
        target = "abstractvision.backends.mflux"
        with patch(f"{target}._lazy_import_mflux", side_effect=never), patch(
            f"{target}._lazy_import_mflux_minimax_h3", side_effect=never
        ), patch("abstractvision.model_inventory.local_model_profile", return_value="apple-silicon"):
            model_inventory._mlx_gen_rows.cache_clear()
            started = time.perf_counter()
            peaks = {
                repo: model_inventory.memory_peaks(repo)
                for repo in ("AbstractFramework/minimax-h3-8bit", "AbstractFramework/minimax-h3-4bit")
            }
            self.assertLess(time.perf_counter() - started, 1.0)
        runs = self._backend(base_model="minimax-h3-turbo").describe_routes(task="text_to_video")[0]["measured_runs"]
        for repo, width in (("AbstractFramework/minimax-h3-8bit", 8), ("AbstractFramework/minimax-h3-4bit", 4)):
            expected = sorted(
                run["footprint_bytes"] for run in runs if run["quantize"] == width and run["outcome"] == "completed"
            )
            listed = sorted(p["peak_bytes"] for p in peaks[repo] if p["task"] == "text_to_video")
            self.assertEqual(listed, expected, repo)
        self.assertEqual({p["task"] for p in peaks["AbstractFramework/minimax-h3-8bit"]}, {"text_to_video", "image_to_video"})
        # The engine's own names for the modes, as its smallest-tested-machine table uses them.
        self.assertEqual(tuple(flags.engine_mode for flags in MLX_GEN_MEMORY_MODE_FLAGS.values()), MINIMAX_H3_MODES)
        for table in MINIMAX_H3_SMALLEST_TESTED_MACHINE.values():
            self.assertLessEqual(set(table), set(MINIMAX_H3_MODES))

    def test_measured_runs_name_no_schedule_and_are_the_same_on_every_schedule(self):
        # Why `memory_peaks` takes no schedule: nothing in a run tells which schedule it ran with.
        runs = {
            schedule: [row["measured_runs"] for row in self._backend(base_model=schedule).describe_routes()]
            for schedule in ("minimax-h3", "minimax-h3-turbo", "minimax-h3-turbo-544p")
        }
        self.assertEqual(runs["minimax-h3"], runs["minimax-h3-turbo"])
        self.assertEqual(runs["minimax-h3"], runs["minimax-h3-turbo-544p"])
        keys = {key for rows in runs["minimax-h3"] for run in rows for key in run}
        self.assertFalse(keys & {"base_model", "schedule", "model", "turbo_lora", "adapter"}, keys)

    def test_schedule_config_is_the_one_the_cli_builds(self):
        from mflux.models.common.config import ModelConfig

        config = ModelConfig.from_name("AbstractFramework/minimax-h3-8bit", base_model="minimax-h3-turbo-544p")
        overrides = config.transformer_overrides
        self.assertEqual((overrides["default_steps"], overrides["default_width"]), (8, 960))
        self.assertIn("turbo_lora", overrides)


if __name__ == "__main__":
    unittest.main()
