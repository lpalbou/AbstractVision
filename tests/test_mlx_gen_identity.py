"""How the MLX-Gen backend identifies a model, a prepared package, a gated download and the paths
in the engine's metadata: from facts (aliases, cards, shard metadata, catalog fields, exception
types, the engine's own field list), never from words inside a name."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _mlx_gen_installed() -> bool:
    try:
        import mflux.models.common.weights.loading.streaming_weight_loader  # noqa: F401
    except Exception:
        return False
    return True


def _yaml_installed() -> bool:
    try:
        import yaml  # noqa: F401
    except ImportError:
        return False
    return True


_PREPARED_SHARD = b"mlx-gen prepared shard"


class _FakeWeightLoader:
    """MLX-Gen's StreamingWeightLoader.is_prepared, deciding from the shard content."""

    @staticmethod
    def is_prepared(component):
        return any(
            shard.read_bytes() == _PREPARED_SHARD for shard in Path(component).glob("*.safetensors")
        )


def _package(root: Path, name: str, *, prepared: bool = False, card: str = "") -> Path:
    path = root / name
    (path / "transformer").mkdir(parents=True)
    (path / "transformer" / "0.safetensors").write_bytes(_PREPARED_SHARD if prepared else b"x")
    if card:
        (path / "README.md").write_text(card, encoding="utf-8")
    return path


class TestBaseModelIdentification(unittest.TestCase):
    def setUp(self):
        from abstractvision.backends.mflux import _infer_base_model

        self.infer = _infer_base_model
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_minimax_h3_is_identified_exactly_or_by_its_card(self):
        self.assertIsNone(self.infer("someone/minimax-m2-chat-h3lix"))
        self.assertIsNone(self.infer("AbstractFramework/minimax-h3-turbo-lora-pack"))
        self.assertEqual(self.infer("minimax-h3-turbo-544p"), "minimax-h3-turbo-544p")
        self.assertEqual(self.infer("AbstractFramework/minimax-h3-8bit"), "minimax-h3")
        # A Hugging Face cache snapshot is named by its repo id, not by its revision.
        snapshot = "/Users/someone/.cache/huggingface/hub/models--AbstractFramework--minimax-h3-4bit/snapshots/0a1b2c"
        self.assertEqual(self.infer(snapshot), "minimax-h3")

    @unittest.skipUnless(_yaml_installed(), "PyYAML is not installed (it comes with huggingface_hub)")
    def test_a_folder_of_any_name_is_identified_by_its_cards_top_level_base_model(self):
        cards = {
            "string": ("base_model: MiniMaxAI/MiniMax-H3\nlibrary_name: mlx-gen\n", "minimax-h3"),
            "flow list": ("base_model: [MiniMaxAI/MiniMax-H3]\n", "minimax-h3"),
            "block list": ("base_model:\n  - MiniMaxAI/MiniMax-H3\n  - other/model\n", "minimax-h3"),
            "quoted": ("base_model: 'MiniMaxAI/MiniMax-H3'  # the source\n", "minimax-h3"),
            "nested key only": ("model-index:\n  base_model: MiniMaxAI/MiniMax-H3\n", None),
            "not YAML": ("base_model: [MiniMaxAI/MiniMax-H3\n", None),
        }
        for name, (front_matter, expected) in cards.items():
            with self.subTest(card=name):
                package = _package(self.tmp, f"model-{len(name)}-{name[0]}", card=f"---\n{front_matter}---\n# Card\n")
                self.assertEqual(self.infer(str(package)), expected)

    def test_a_parent_directory_never_decides(self):
        self.assertEqual(
            self.infer("/Users/someone/qwen-image-tests/minimax-h3-8bit"), "minimax-h3"
        )
        self.assertEqual(
            self.infer("/Users/someone/fibo-benchmarks/out/flux2-klein-4b"), "flux2-klein-4b"
        )
        # Exact identification of any value comes before name matching of any value.
        self.assertEqual(
            self.infer("/Users/someone/qwen-image-tests/x", "z-image-turbo"), "z-image-turbo"
        )

    def test_names_0_3_36_resolved_by_matching_still_resolve(self):
        self.assertEqual(self.infer("someone/my-qwen-image-finetune"), "qwen-image")
        self.assertEqual(
            self.infer("/Volumes/wan-video/models/z-image-turbo-8bit"), "z-image-turbo"
        )
        self.assertEqual(self.infer("someone/seedvr2-7b-sharp"), "seedvr2-7b")


class TestPreparedPackage(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.cache = self.tmp / "hub"
        self.cache.mkdir()
        stack.enter_context(
            patch.dict(
                "os.environ", {"HF_HUB_CACHE": str(self.cache), "ABSTRACTVISION_MEMORY_MODE": ""}
            )
        )
        stack.enter_context(
            patch("abstractvision.backends.mflux.framework_hf_cache_roots", return_value=[])
        )
        stack.enter_context(
            patch(
                "abstractvision.backends.mflux._lazy_import_mlx_gen_weight_loader",
                return_value=(_FakeWeightLoader, ValueError),
            )
        )
        stack.enter_context(
            patch("abstractvision.backends.mflux.MFluxVisionBackend._route_rows", return_value=[])
        )
        stack.enter_context(
            patch(
                "abstractvision.backends.mflux._mlx_gen_quantize_choices",
                return_value=(3, 4, 5, 6, 8),
            )
        )

    def _backend(self, model, base_model=None):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        return MFluxVisionBackend(
            config=MFluxBackendConfig(model=str(model), base_model=base_model)
        )

    def _assert_agree(self, backend, model_path, *, prepared):
        from abstractvision.errors import CapabilityNotSupportedError

        if prepared:
            self.assertEqual(tuple(backend.get_capabilities().quantize_bits), ())
            with self.assertRaisesRegex(CapabilityNotSupportedError, "prepared MLX-Gen package"):
                backend._checked_load_quantize(8)
        else:
            self.assertEqual(tuple(backend.get_capabilities().quantize_bits), (3, 4, 5, 6, 8))
            self.assertEqual(backend._checked_load_quantize(8), 8)

    def test_a_folder_name_never_makes_a_package_prepared(self):
        from abstractvision.backends.mflux import _is_prepared_mlx_gen_package

        for name in ("seq8", "wan2.2-ti2v-5b-freq8", "klein-q8", "minimax-h3-8bit"):
            source = _package(self.tmp, name, card="---\nlibrary_name: mlx-gen\n---\n")
            self.assertFalse(_is_prepared_mlx_gen_package(source, str(source)), name)
            self._assert_agree(
                self._backend(source, base_model="flux2-klein-4b"), source, prepared=False
            )
        self.assertFalse(
            _is_prepared_mlx_gen_package(
                "/nonexistent/minimax-h3-8bit", "/nonexistent/minimax-h3-8bit"
            )
        )

    def test_the_engine_metadata_makes_a_folder_of_any_name_prepared(self):
        package = _package(self.tmp, "my-model", prepared=True)
        self._assert_agree(
            self._backend(package, base_model="flux2-klein-4b"), package, prepared=True
        )

    def test_the_catalog_says_so_for_its_repositories(self):
        from abstractvision.backends.mflux import _is_prepared_mlx_gen_package

        # Not on disk: the catalog's own fields.
        self.assertTrue(_is_prepared_mlx_gen_package(None, "AbstractFramework/minimax-h3-bf16"))
        self.assertTrue(
            _is_prepared_mlx_gen_package(None, "AbstractFramework/flux.2-klein-4b-8bit")
        )
        self.assertFalse(_is_prepared_mlx_gen_package(None, "MiniMaxAI/MiniMax-H3"))
        self.assertFalse(_is_prepared_mlx_gen_package(None, "Wan-AI/Wan2.2-TI2V-5B-Diffusers"))

    def test_the_2bit_bonsai_snapshot_is_prepared_by_its_catalog_entry(self):
        repo = "prism-ml/bonsai-image-ternary-4B-mlx-2bit"
        snapshot = (
            self.cache / "models--prism-ml--bonsai-image-ternary-4B-mlx-2bit" / "snapshots" / "abc"
        )
        (snapshot / "transformer-packed-mflux").mkdir(parents=True)
        (snapshot / "transformer-packed-mflux" / "diffusion_pytorch_model.safetensors").write_bytes(
            b"x"
        )
        (snapshot / "README.md").write_text("---\nlibrary_name: mlx\n---\n", encoding="utf-8")
        from abstractvision.backends.mflux import _is_prepared_mlx_gen_package

        # The engine reports no MLX-Gen metadata in its shards; the catalog stores it at 2 bits.
        self.assertTrue(_is_prepared_mlx_gen_package(snapshot, None))
        self._assert_agree(self._backend(repo), snapshot, prepared=True)


@unittest.skipUnless(_mlx_gen_installed(), "mlx-gen is not installed")
class TestEngineShardHeader(unittest.TestCase):
    def test_the_engine_reads_the_shard_header(self):
        import numpy as np
        from safetensors.numpy import save_file

        from abstractvision.backends.mflux import _engine_reports_prepared

        with tempfile.TemporaryDirectory() as td:
            for name, metadata, expected in (
                ("prepared", {"mflux_version": "0.40.1", "quantization_level": "8"}, True),
                ("prepared-bf16", {"mflux_version": "0.40.1", "quantization_level": "None"}, True),
                ("source", {"format": "pt"}, False),
                ("unreadable", None, False),
            ):
                component = Path(td) / name / "transformer"
                component.mkdir(parents=True)
                shard = component / "0.safetensors"
                if metadata is None:
                    shard.write_bytes(b"x")
                else:
                    save_file(
                        {"w": np.zeros((2,), dtype=np.float16)}, str(shard), metadata=metadata
                    )
                self.assertIs(_engine_reports_prepared(Path(td) / name), expected, name)


class TestExplicitDownloads(unittest.TestCase):
    """The catalog's per-download `explicit_download` is the only thing generation reads; `gated` is not."""

    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.cache = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(
            patch.dict(
                "os.environ", {"HF_HUB_CACHE": str(self.cache), "ABSTRACTVISION_MEMORY_MODE": ""}
            )
        )
        stack.enter_context(
            patch("abstractvision.backends.mflux.framework_hf_cache_roots", return_value=[])
        )
        self.downloads = []

        def download(repo_id, **kwargs):
            self.downloads.append(repo_id)
            return self.cache / "downloaded" / repo_id

        stack.enter_context(
            patch("abstractvision.backends.mflux.download_hf_repo_snapshot", side_effect=download)
        )

    def _resolve(self, model, base_model=None):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        config = MFluxBackendConfig(model=model, base_model=base_model, allow_download=True)
        return MFluxVisionBackend(config=config)._resolve_model()

    def test_gated_repositories_download_with_allow_download_as_in_0_3_36(self):
        for model, base in (
            ("AbstractFramework/flux.2-klein-9b-8bit", "flux2-klein-9b"),
            ("AbstractFramework/flux.2-klein-base-9b-4bit", "flux2-klein-base-9b"),
            ("AbstractFramework/fibo-8bit", "fibo"),
            ("briaai/FIBO", "fibo"),
            ("briaai/Fibo-Edit", "fibo-edit"),
        ):
            with self.subTest(model=model):
                self.assertEqual(
                    self._resolve(model), (str(self.cache / "downloaded" / model), base)
                )
        self.assertEqual(len(self.downloads), 5)

    def test_a_repository_the_catalog_does_not_list_downloads_as_in_0_3_36(self):
        model = "someone/z-image-turbo-finetune"
        self.assertEqual(
            self._resolve(model), (str(self.cache / "downloaded" / model), "z-image-turbo")
        )

    def test_every_minimax_h3_repository_is_refused_and_the_refusal_names_the_licence(self):
        from abstractvision.errors import OptionalDependencyMissingError

        for model in (
            "AbstractFramework/minimax-h3-8bit",
            "AbstractFramework/minimax-h3-bf16",
            "AbstractFramework/minimax-h3-4bit",
            "MiniMaxAI/MiniMax-H3",
        ):
            with self.subTest(model=model):
                with self.assertRaises(OptionalDependencyMissingError) as caught:
                    self._resolve(model, base_model="minimax-h3")
                message = str(caught.exception)
                self.assertIn("never downloaded during generation", message)
                self.assertIn("MiniMax H3 Community License", message)
                self.assertIn(f"`abstractvision download {model}`", message)
        self.assertEqual(self.downloads, [])

    def test_the_explicit_download_field_is_the_one_source(self):
        from abstractvision.backends import mflux

        def catalog(**flags):
            download = SimpleNamespace(repo_id="AbstractFramework/minimax-h3-8bit", **flags)
            spec = SimpleNamespace(
                model_id="MiniMaxAI/MiniMax-H3", license_name="", downloads=[download]
            )
            return SimpleNamespace(list_models=lambda: [spec.model_id], get=lambda _model_id: spec)

        note = mflux._explicit_download_note
        with patch.object(
            mflux,
            "_mflux_capability_registry",
            return_value=catalog(gated=True, explicit_download=False),
        ):
            self.assertEqual(note("AbstractFramework/minimax-h3-8bit", "minimax-h3"), "")
        with patch.object(
            mflux,
            "_mflux_capability_registry",
            return_value=catalog(gated=False, explicit_download=True),
        ):
            self.assertIn(
                "never downloaded during generation",
                note("AbstractFramework/minimax-h3-8bit", "minimax-h3"),
            )
        # A name that looks like a marked repository is not one.
        self.assertEqual(note("someone/minimax-h3-8bit", "minimax-h3"), "")
        self.assertEqual(note("AbstractFramework/flux.2-klein-9b-8bit", "flux2-klein-9b"), "")


class TestDownloadRequired(unittest.TestCase):
    def test_only_mlx_gens_exception_type_counts(self):
        from abstractvision.backends.mflux import _is_mlx_gen_download_required

        DownloadRequiredError = type("DownloadRequiredError", (FileNotFoundError,), {})
        self.assertFalse(_is_mlx_gen_download_required(DownloadRequiredError("other library")))
        carrying = FileNotFoundError("x")
        carrying.download_command = "something download"
        self.assertFalse(_is_mlx_gen_download_required(carrying))
        self.assertFalse(_is_mlx_gen_download_required(FileNotFoundError("missing")))

    @unittest.skipUnless(_mlx_gen_installed(), "mlx-gen is not installed")
    def test_mlx_gens_download_required_error_is_recognized_and_explained(self):
        from mflux.models.common.download_policy import DownloadRequiredError

        from abstractvision.backends.mflux import (
            _is_mlx_gen_download_required,
            _wrap_mlx_gen_download_required,
        )

        exc = DownloadRequiredError("AbstractFramework/flux.2-klein-4b-8bit")
        self.assertTrue(_is_mlx_gen_download_required(exc))
        self.assertIn(
            "mlxgen download --model AbstractFramework/flux.2-klein-4b-8bit",
            str(_wrap_mlx_gen_download_required(exc)),
        )


# --- metadata paths ------------------------------------------------------------------------------
#
# One sample per route AbstractVision runs, with the key structure mlx-gen 0.40.1 writes (from the
# sidecars of real runs for MiniMax-H3, from the engine's metadata builders for the others) and
# invented values. Every field that holds a path in a real run holds an absolute one here.

_P = "/Users/someone/models"
_LORA_REPORT = {
    "requested_path": f"{_P}/style.safetensors",
    "resolved_path": f"{_P}/style.safetensors",
    "scale": 1.0,
    "role": None,
    "total_key_count": 10,
    "matched_key_count": 10,
    "unmatched_key_count": 0,
    "applied_target_count": 4,
}
_RUNTIME_MEMORY = {
    "phase": "video-save-complete",
    "timestamp": 1.0,
    "pid": 1,
    "platform": "macOS",
    "synchronized": True,
    "mlx_active_memory_bytes": 1,
    "mlx_peak_memory_bytes": 1,
    "mlx_cache_memory_bytes": 1,
    "process_rss_bytes": 1,
    "process_peak_rss_bytes": 1,
    "darwin_physical_footprint_bytes": 1,
    "errors": {},
    "darwin_peak_physical_footprint_bytes": 1,
}
_HEALTH = {
    "frame_count": 9,
    "width": 64,
    "height": 64,
    "fps": 24.0,
    "luma_min": 0.1,
    "luma_max": 0.9,
    "luma_mean": 0.5,
    "mean_temporal_delta": 0.01,
}
_LORA_FIELDS = {
    "lora_paths": [f"{_P}/style.safetensors"],
    "lora_scales": [1.0],
    "lora_application_reports": [dict(_LORA_REPORT)],
    "lora_applied_file_count": 1,
    "lora_applied_target_count": 4,
}
_IMAGE = {  # GeneratedImage._get_metadata
    "metadata_schema_version": 3,
    "mflux_version": "0.40.1",
    "model": f"{_P}/flux.2-klein-4b-8bit",
    "base_model": "flux2-klein-4b",
    "seed": 1,
    "steps": 4,
    "guidance": 1.0,
    "height": 64,
    "width": 64,
    "precision": "mlx.core.bfloat16",
    "quantize": 8,
    "generation_time_seconds": 1.0,
    "created_at": "2026-01-01T00:00:00",
    "image_path": None,
    "image_paths": None,
    "image_strength": None,
    "canvas_policy": None,
    "resize_mode": None,
    "requested_width": None,
    "requested_height": None,
    "source_image_width": None,
    "source_image_height": None,
    "controlnet_image_path": None,
    "controlnet_strength": None,
    "masked_image_path": None,
    "depth_image_path": None,
    "redux_image_paths": None,
    "redux_image_strengths": None,
    "prompt": "/imagine a lighthouse at dusk",
    "negative_prompt": "/blur, C:\\noise",
    "runtime_memory": None,
}
_VIDEO = {  # GeneratedVideo.build_metadata
    "metadata_schema_version": 3,
    "mflux_version": "0.40.1",
    "model": f"{_P}/wan2.2-ti2v-5b",
    "base_model": "wan2.2-ti2v-5b",
    "task": "text-to-video",
    "seed": 1,
    "steps": 8,
    "guidance": 5.0,
    "guidance_2": None,
    "flow_shift": 3.0,
    "solver": None,
    "height": 64,
    "width": 64,
    "requested_height": None,
    "requested_width": None,
    "canvas_policy": None,
    "resize_mode": None,
    "source_image_height": None,
    "source_image_width": None,
    "frames": 9,
    "fps": 24,
    "duration_seconds": 0.375,
    "precision": "mlx.core.bfloat16",
    "quantize": 8,
    "generation_time_seconds": 1.0,
    "created_at": "2026-01-01T00:00:00",
    "image_path": None,
    "video_path": None,
    "prompt": "/imagine a fox",
    "negative_prompt": "/static",
    "lora_paths": None,
    "lora_scales": None,
    "runtime_memory": dict(_RUNTIME_MEMORY),
}
_SAVE = {"save_time_seconds": 0.1, "total_generation_time_seconds": 1.1}
_IMAGE_ROUTE_SAMPLES = {
    "text_to_image": {**_IMAGE, **_LORA_FIELDS},
    "image_to_image (edit, references and mask)": {
        **_IMAGE,
        **_LORA_FIELDS,
        "image_path": f"{_P}/tmp/reference.png",
        "image_paths": [f"{_P}/tmp/reference.png", f"{_P}/second.png"],
        "image_strength": 0.6,
        "masked_image_path": f"{_P}/tmp/mask.png",
        "effective_steps": 12,
        "mask_strength": 1.0,
    },
    "image_upscale (SeedVR2, input carrying an earlier run's metadata)": {
        **_IMAGE,
        "model": f"{_P}/seedvr2-3b-8bit",
        "image_path": f"{_P}/tmp/input.png",
        "resolution": "1024",
        "softness": 0.0,
        "color_correction_mode": "lab",
        "steps_mode": "fixed",
        "seedvr2_checkpoint_variant": "3b",
        "seedvr2_source_layout": "prepared",
        "one_step_residue_pct": 1.5,
        "original_width": 32,
        "original_height": 32,
        "original_seed": 7,
        "original_steps": 4,
        "original_guidance": 1.0,
        "original_model": f"{_P}/flux.2-klein-4b-8bit",
        "original_lora_paths": [f"{_P}/style.safetensors"],
        "original_lora_scales": [1.0],
        "original_lora_application_reports": [dict(_LORA_REPORT)],
        "original_lora_applied_file_count": 1,
        "original_lora_applied_target_count": 4,
        "original_quantize": 8,
    },
}
_VIDEO_ROUTE_SAMPLES = {
    "Wan video (text_to_video / image_to_video)": {
        **_VIDEO,
        **_LORA_FIELDS,
        "task": "image-to-video",
        "image_path": f"{_P}/tmp/keyframe.png",
        "lora_target_roles": ["high_noise_transformer"],
        "released_inactive_denoiser": True,
        "high_noise_reloads": 1,
        "prompt_tokens": 12,
        "prompt_truncated": False,
        "negative_prompt_tokens": 3,
        "negative_prompt_truncated": False,
        "denoising_step_list": [999, 500],
        "video_health": {"file": {"source": f"{_P}/tmp/video.mp4", **_HEALTH}},
        **_SAVE,
    },
    "MiniMax-H3 (soundtrack muxed)": {
        **_VIDEO,
        **_LORA_FIELDS,
        "model": f"{_P}/minimax-h3-8bit",
        "base_model": "minimax-h3-turbo-544p",
        "guidance": None,
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "num_inference_steps": 9,
        "requested_frames": 124,
        "text_tokens": 80,
        "generate_audio": True,
        "modulation_tables": True,
        "transformer_blocks_streamed": False,
        "transformer_released": True,
        "video_health": {
            "frames": {"source": "frames", **_HEALTH},
            "file": {"source": f"{_P}/tmp/video.mp4", **_HEALTH},
        },
        "audio_present": True,
        "audio_source": "generated",
        "audio_channels": 2,
        "audio_sample_rate": 32000,
        "audio_duration_seconds": 0.375,
        "audio_muxed": True,
        "audio_codec": "aac",
        "audio_mux_mode": "ffmpeg_copy_video_aac_audio",
        **_SAVE,
    },
    "MiniMax-H3 (image_to_video, soundtrack left as a WAV)": {
        **_VIDEO,
        "model": f"{_P}/minimax-h3-8bit",
        "task": "image-to-video",
        "image_path": f"{_P}/tmp/keyframe.png",
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "num_inference_steps": 9,
        "requested_frames": 124,
        "text_tokens": 80,
        "generate_audio": True,
        "modulation_tables": False,
        "transformer_blocks_streamed": True,
        "transformer_released": False,
        "video_health": {
            "frames": {"source": "frames", **_HEALTH},
            "file": {"source": f"{_P}/tmp/video.mp4", **_HEALTH},
        },
        "audio_present": True,
        "audio_source": "generated",
        "audio_channels": 2,
        "audio_sample_rate": 32000,
        "audio_duration_seconds": 0.375,
        "audio_muxed": False,
        "audio_mux_reason": "RuntimeError: ffmpeg failed",
        "audio_sidecar_path": f"{_P}/tmp/video.wav",
        **_SAVE,
    },
}
_ROUTE_SAMPLES = {**_IMAGE_ROUTE_SAMPLES, **_VIDEO_ROUTE_SAMPLES}
_USER_TEXT = ("prompt", "negative_prompt")


def _absolute_paths(value, *, top=True):
    # Every string in the metadata that is an absolute path, the user's own text excepted.
    if isinstance(value, str):
        return [value] if os.path.isabs(value) else []
    if isinstance(value, dict):
        return [
            found
            for key, item in value.items()
            if not (top and key in _USER_TEXT)
            for found in _absolute_paths(item, top=False)
        ]
    if isinstance(value, (list, tuple)):
        return [found for item in value for found in _absolute_paths(item, top=False)]
    return []


class TestEngineMetadataPaths(unittest.TestCase):
    def test_no_route_leaves_a_local_path_and_the_users_text_is_kept(self):
        from abstractvision.backends.mflux import (
            _TEMPORARY_PATH_METADATA_KEYS,
            _public_engine_metadata,
        )

        for route, sample in _ROUTE_SAMPLES.items():
            with self.subTest(route=route):
                original = copy.deepcopy(sample)
                cleaned = _public_engine_metadata(sample)
                self.assertEqual(sample, original)  # not modified in place
                self.assertEqual(_absolute_paths(cleaned), [])
                self.assertEqual(cleaned["prompt"], sample["prompt"])
                self.assertEqual(cleaned["negative_prompt"], sample["negative_prompt"])
                for key in _TEMPORARY_PATH_METADATA_KEYS:
                    self.assertNotIn(key, cleaned)
                # Only the path fields change.
                self.assertEqual(json.dumps(cleaned).count(_P), 0)
                for key, value in sample.items():
                    if key not in _TEMPORARY_PATH_METADATA_KEYS and _P not in json.dumps(value):
                        self.assertEqual(cleaned[key], value, key)

        # An edit's image list and its mask are files written for the run: left out, as the source is.
        cleaned = _public_engine_metadata(_ROUTE_SAMPLES["image_to_image (edit, references and mask)"])
        for key in ("image_path", "image_paths", "masked_image_path"):
            self.assertNotIn(key, cleaned)

        cleaned = _public_engine_metadata(_ROUTE_SAMPLES["MiniMax-H3 (soundtrack muxed)"])
        self.assertEqual(cleaned["model"], "minimax-h3-8bit")
        self.assertEqual(
            cleaned["lora_application_reports"][0]["resolved_path"], "style.safetensors"
        )
        self.assertEqual(cleaned["video_health"]["file"]["source"], "video.mp4")
        self.assertEqual(cleaned["video_health"]["frames"]["source"], "frames")

    def test_a_field_the_list_does_not_name_is_kept_as_written(self):
        from abstractvision.backends.mflux import _public_engine_metadata

        # Values are not guessed at: a field is a path field because the engine writes a path there.
        metadata = {
            "model": f"{_P}/fibo-8bit",
            "controlnet_model": "/InstantX/x",
            "notes": ["/a", {"b": "/c"}],
        }
        self.assertEqual(
            _public_engine_metadata(metadata),
            {"model": "fibo-8bit", "controlnet_model": "/InstantX/x", "notes": ["/a", {"b": "/c"}]},
        )

    @unittest.skipUnless(_mlx_gen_installed(), "mlx-gen is not installed")
    def test_the_engine_builders_write_no_field_the_samples_do_not_cover(self):
        # The pinned engine's own builders, called with an absolute path in every path argument: a
        # field they add or rename fails here until it is classified in the samples above.
        import mlx.core as mx
        import PIL.Image
        from mflux.utils.generated_image import GeneratedImage
        from mflux.utils.generated_video import GeneratedVideo

        from abstractvision.backends.mflux import _public_engine_metadata

        config = SimpleNamespace(
            model_name=f"{_P}/model", base_model="base", supports_guidance=True
        )
        image = GeneratedImage(
            image=PIL.Image.new("RGB", (1, 1)),
            model_config=config,
            seed=1,
            prompt="/a prompt",
            steps=1,
            guidance=1.0,
            precision=mx.bfloat16,
            quantization=8,
            generation_time=1.0,
            lora_paths=[f"{_P}/a.safetensors"],
            lora_scales=[1.0],
            image_path=f"{_P}/image.png",
            image_paths=[f"{_P}/image.png"],
            image_strength=0.5,
            masked_image_path=f"{_P}/mask.png",
        )
        with patch.dict("os.environ", {"MFLUX_RUNTIME_MEMORY_TELEMETRY": "0"}):
            video = GeneratedVideo.build_metadata(
                model_config=config,
                seed=1,
                prompt="/a prompt",
                steps=1,
                guidance=1.0,
                guidance_2=None,
                flow_shift=3.0,
                solver=None,
                precision=mx.bfloat16,
                quantization=8,
                generation_time=1.0,
                height=64,
                width=64,
                frame_count=9,
                fps=24,
                image_path=f"{_P}/keyframe.png",
                lora_paths=[f"{_P}/a.safetensors"],
                lora_scales=[1.0],
            )
        image_keys = {key for sample in _IMAGE_ROUTE_SAMPLES.values() for key in sample}
        video_keys = {key for sample in _VIDEO_ROUTE_SAMPLES.values() for key in sample}
        for metadata, covered in ((image._get_metadata(), image_keys), (video, video_keys)):
            self.assertEqual(set(metadata) - covered, set())
            cleaned = _public_engine_metadata(metadata)
            self.assertEqual(_absolute_paths(cleaned), [])
            self.assertEqual(cleaned["prompt"], "/a prompt")


if __name__ == "__main__":
    unittest.main()
