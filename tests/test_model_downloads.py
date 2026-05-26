import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestModelDownloads(unittest.TestCase):
    def test_auto_scope_prefers_apple_targets_and_mlx_gen_qwen(self):
        from abstractvision.model_downloads import catalog_target_scope, find_model_preset

        with patch("abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"):
            with patch("abstractvision.model_downloads.sys.platform", "darwin"):
                self.assertEqual(
                    catalog_target_scope(target="auto", engine=None, include_all_targets=False),
                    ("mlx", "gguf", "diffusers", "hf-snapshot"),
                )
                preset = find_model_preset(
                    "AbstractFramework/qwen-image-2512-4bit",
                    target="auto",
                    engine=None,
                    require_8bit=True,
                )
        self.assertEqual((preset.target, preset.engine), ("mlx", "mlx-gen"))
        self.assertEqual(preset.repo_id, "AbstractFramework/qwen-image-2512-4bit")

    def test_mlx_gen_family_aliases_are_rejected_as_ambiguous(self):
        from abstractvision.model_downloads import find_model_preset

        with self.assertRaisesRegex(ValueError, "not an exact published model id"):
            find_model_preset("qwen-image", target="mlx", engine="mlx-gen", require_8bit=True)

    def test_mlx_gen_catalog_covers_abstractframework_q4_q8_collection(self):
        from abstractvision.model_downloads import model_presets

        expected = {
            "AbstractFramework/flux.2-klein-4b-4bit",
            "AbstractFramework/flux.2-klein-4b-8bit",
            "AbstractFramework/flux.2-klein-9b-4bit",
            "AbstractFramework/flux.2-klein-9b-8bit",
            "AbstractFramework/flux.2-klein-base-4b-4bit",
            "AbstractFramework/flux.2-klein-base-4b-8bit",
            "AbstractFramework/flux.2-klein-base-9b-4bit",
            "AbstractFramework/flux.2-klein-base-9b-8bit",
            "AbstractFramework/qwen-image-4bit",
            "AbstractFramework/qwen-image-8bit",
            "AbstractFramework/qwen-image-2512-4bit",
            "AbstractFramework/qwen-image-2512-8bit",
            "AbstractFramework/qwen-image-edit-4bit",
            "AbstractFramework/qwen-image-edit-8bit",
            "AbstractFramework/qwen-image-edit-2509-4bit",
            "AbstractFramework/qwen-image-edit-2509-8bit",
            "AbstractFramework/qwen-image-edit-2511-4bit",
            "AbstractFramework/qwen-image-edit-2511-8bit",
            "AbstractFramework/z-image-4bit",
            "AbstractFramework/z-image-8bit",
            "AbstractFramework/z-image-turbo-4bit",
            "AbstractFramework/z-image-turbo-8bit",
            "AbstractFramework/ernie-image-turbo-4bit",
            "AbstractFramework/ernie-image-turbo-8bit",
        }
        repos = {
            preset.repo_id
            for preset in model_presets(target="mlx", engine="mlx-gen", include_non_8bit=True)
            if preset.source == "abstractframework-mlx-gen"
        }

        self.assertTrue(expected.issubset(repos))

    def test_mlx_gen_exact_repo_ids_select_q4_and_q8_artifacts(self):
        from abstractvision.model_downloads import find_model_preset

        default_flux = find_model_preset(
            "AbstractFramework/flux.2-klein-4b-4bit",
            target="mlx",
            engine="mlx-gen",
            require_8bit=True,
        )
        quality_flux = find_model_preset(
            "flux.2-klein-4b-8bit",
            target="mlx",
            engine="mlx-gen",
            require_8bit=True,
        )
        quality_qwen_2512 = find_model_preset(
            "AbstractFramework/qwen-image-2512-8bit",
            target="mlx",
            engine="mlx-gen",
            require_8bit=True,
        )
        default_qwen_edit = find_model_preset(
            "AbstractFramework/qwen-image-edit-2511-4bit",
            target="mlx",
            engine="mlx-gen",
            require_8bit=True,
        )
        legacy_qwen_edit = find_model_preset(
            "qwen-image-edit-4bit",
            target="mlx",
            engine="mlx-gen",
            require_8bit=True,
        )
        default_ernie = find_model_preset(
            "AbstractFramework/ernie-image-turbo-4bit",
            target="mlx",
            engine="mlx-gen",
            require_8bit=True,
        )
        quality_ernie = find_model_preset(
            "ernie-image-turbo-8bit",
            target="mlx",
            engine="mlx-gen",
            require_8bit=True,
        )

        self.assertEqual(default_flux.repo_id, "AbstractFramework/flux.2-klein-4b-4bit")
        self.assertEqual(quality_flux.repo_id, "AbstractFramework/flux.2-klein-4b-8bit")
        self.assertEqual(quality_qwen_2512.repo_id, "AbstractFramework/qwen-image-2512-8bit")
        self.assertEqual(default_qwen_edit.repo_id, "AbstractFramework/qwen-image-edit-2511-4bit")
        self.assertEqual(legacy_qwen_edit.repo_id, "AbstractFramework/qwen-image-edit-4bit")
        self.assertEqual(default_ernie.repo_id, "AbstractFramework/ernie-image-turbo-4bit")
        self.assertEqual(quality_ernie.repo_id, "AbstractFramework/ernie-image-turbo-8bit")
        self.assertEqual(default_ernie.local_dir_name, "ernie-image-turbo-mlx-gen-4bit")
        self.assertEqual(quality_ernie.local_dir_name, "ernie-image-turbo-mlx-gen-8bit")

    def test_wan_mlx_gen_preset_is_available_as_non_quantized_video_fallback(self):
        from abstractvision.model_downloads import find_model_preset, model_presets

        presets = model_presets(target="mlx", engine="mlx-gen", include_non_8bit=True)
        wan = next(p for p in presets if p.repo_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

        self.assertEqual(wan.key, "wan2.2-ti2v-5b")
        self.assertEqual(wan.target, "mlx")
        self.assertEqual(wan.engine, "mlx-gen")
        self.assertEqual(wan.quantization_bits, 16)
        self.assertEqual(wan.source, "official")

        selected = find_model_preset("Wan-AI/Wan2.2-TI2V-5B-Diffusers", target="mlx", engine="mlx-gen", require_8bit=False)
        self.assertEqual(selected.repo_id, "Wan-AI/Wan2.2-TI2V-5B-Diffusers")

    def test_fibo_mlx_gen_presets_are_exact_non_quantized_runtime_models(self):
        from abstractvision.model_downloads import find_model_preset, model_presets

        presets = model_presets(target="mlx", engine="mlx-gen", include_non_8bit=True)
        fibo = next(p for p in presets if p.repo_id == "briaai/FIBO")
        fibo_edit = next(p for p in presets if p.repo_id == "briaai/Fibo-Edit")

        self.assertEqual(fibo.key, "fibo")
        self.assertEqual(fibo.target, "mlx")
        self.assertEqual(fibo.engine, "mlx-gen")
        self.assertEqual(fibo.quantization_bits, 16)
        self.assertEqual(fibo.source, "official")
        self.assertEqual(fibo_edit.key, "fibo-edit")

        selected = find_model_preset("briaai/FIBO", target="mlx", engine="mlx-gen", require_8bit=False)
        self.assertEqual(selected.repo_id, "briaai/FIBO")

    def test_sdcpp_gguf_presets_can_be_disabled_on_apple_silicon(self):
        from abstractvision.model_downloads import (
            MacOSGGUFUnsupportedError,
            catalog_target_scope,
            model_presets,
        )

        with patch("abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"):
            with patch("abstractvision.model_downloads.sys.platform", "darwin"):
                with patch.dict("os.environ", {"ABSTRACTVISION_DISABLE_GGUF_ON_MACOS": "1"}, clear=False):
                    self.assertTrue(all(preset.target != "gguf" for preset in model_presets(include_all_targets=True)))
                    with self.assertRaisesRegex(MacOSGGUFUnsupportedError, "disabled on this macOS host"):
                        catalog_target_scope(target="auto", engine="sdcpp", include_all_targets=False)

    def test_auto_scope_prefers_gpu_fp8_before_other_targets(self):
        from abstractvision.model_downloads import catalog_target_scope, find_model_preset

        with patch("abstractvision.model_downloads.local_model_profile", return_value="cuda"):
            self.assertEqual(
                catalog_target_scope(target="auto", engine=None, include_all_targets=False),
                ("fp8", "gguf", "diffusers", "hf-snapshot"),
            )
            preset = find_model_preset("flux2-klein-4b", target="auto", engine=None, require_8bit=True)
        self.assertEqual((preset.target, preset.engine), ("fp8", "diffusers-component"))

    def test_diffusers_presets_keep_chat_templates(self):
        from abstractvision.model_downloads import find_model_preset

        preset = find_model_preset(
            "ernie-image-turbo",
            target="diffusers",
            engine="diffusers",
            require_8bit=False,
        )
        self.assertIn("*.jinja", tuple(preset.allow_patterns or ()))

    def test_qwen_image_edit_handle_resolves_legacy_diffusers_snapshot(self):
        from abstractvision.model_downloads import find_model_preset

        preset = find_model_preset(
            "qwen-image-edit",
            target="diffusers",
            engine="diffusers",
            require_8bit=False,
        )
        self.assertEqual(preset.repo_id, "Qwen/Qwen-Image-Edit")

    def test_qwen_image_edit_handle_resolves_legacy_sdcpp_bundle(self):
        from abstractvision.model_downloads import find_model_preset

        with patch("abstractvision.model_downloads.local_model_profile", return_value="cuda"):
            preset = find_model_preset(
                "qwen-image-edit",
                target="gguf",
                engine="stable-diffusion.cpp",
                require_8bit=True,
            )
        self.assertEqual(preset.repo_id, "unsloth/Qwen-Image-Edit-GGUF")

    def test_qwen_image_edit_2511_handle_resolves_dated_release(self):
        from abstractvision.model_downloads import find_model_preset

        preset = find_model_preset(
            "qwen-image-edit-2511",
            target="diffusers",
            engine="diffusers",
            require_8bit=False,
        )
        self.assertEqual(preset.repo_id, "Qwen/Qwen-Image-Edit-2511")

    def test_qwen_image_edit_2511_gguf_handle_resolves_sdcpp_bundle(self):
        from abstractvision.model_downloads import find_model_preset

        with patch("abstractvision.model_downloads.local_model_profile", return_value="cuda"):
            preset = find_model_preset(
                "qwen-image-edit-2511-gguf",
                target="auto",
                engine="stable-diffusion.cpp",
                require_8bit=True,
            )

        self.assertEqual(preset.repo_id, "unsloth/Qwen-Image-Edit-2511-GGUF")
        self.assertEqual(preset.target, "gguf")
        self.assertEqual(preset.engine, "stable-diffusion.cpp")
        self.assertEqual(preset.upstream_repo_id, "Qwen/Qwen-Image-Edit-2511")

    def test_generic_mlx_engine_is_rejected_and_flux1_mlx_gen_presets_are_not_curated(self):
        from abstractvision.model_downloads import find_model_preset, model_presets, normalize_model_engine

        with self.assertRaisesRegex(ValueError, "generic MLX image/video backend"):
            normalize_model_engine("mlx")
        with self.assertRaisesRegex(ValueError, "generic MLX image/video backend"):
            find_model_preset("mlx/flux2-klein-4b", target="auto", engine=None, require_8bit=True)

        mlx_gen_keys = {
            preset.key
            for preset in model_presets(target="mlx", engine="mflux", include_non_8bit=False)
        }
        self.assertIn("qwen-image-edit-2511", mlx_gen_keys)
        self.assertNotIn("flux1-dev", mlx_gen_keys)
        self.assertNotIn("flux1-schnell", mlx_gen_keys)

        with self.assertRaises(ValueError):
            find_model_preset("flux1-dev", target="mlx", engine="mflux", require_8bit=True)

        preset = find_model_preset("flux1-dev", target="auto", engine=None, require_8bit=False)
        self.assertEqual((preset.target, preset.engine), ("diffusers", "diffusers"))
        self.assertEqual(preset.repo_id, "black-forest-labs/FLUX.1-dev")

    def test_resolve_sdcpp_model_selection_uses_cached_component_bundle(self):
        from abstractvision.model_cache import import_directory_to_hf_cache
        from abstractvision.model_downloads import resolve_sdcpp_model_selection

        with tempfile.TemporaryDirectory() as cache_td, tempfile.TemporaryDirectory() as src_td:
            cache_root = Path(cache_td)
            src_root = Path(src_td)

            main_dir = src_root / "flux-main"
            main_dir.mkdir(parents=True, exist_ok=True)
            (main_dir / "flux-2-klein-base-4b-Q8_0.gguf").write_bytes(b"GGUF")
            import_directory_to_hf_cache(
                main_dir,
                repo_id="leejet/FLUX.2-klein-base-4B-GGUF",
                cache_dir=str(cache_root),
                cleanup_source=False,
            )

            vae_dir = src_root / "flux-vae"
            (vae_dir / "vae").mkdir(parents=True, exist_ok=True)
            (vae_dir / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"VAE")
            import_directory_to_hf_cache(
                vae_dir,
                repo_id="black-forest-labs/FLUX.2-klein-base-4B",
                cache_dir=str(cache_root),
                cleanup_source=False,
            )

            llm_dir = src_root / "qwen3"
            llm_dir.mkdir(parents=True, exist_ok=True)
            (llm_dir / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"GGUF")
            import_directory_to_hf_cache(
                llm_dir,
                repo_id="unsloth/Qwen3-4B-GGUF",
                cache_dir=str(cache_root),
                cleanup_source=False,
            )

            with patch("abstractvision.model_downloads.local_model_profile", return_value="cuda"):
                with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                    selection = resolve_sdcpp_model_selection("flux2-klein-base-4b", allow_download=False)

        self.assertIsNone(selection.model)
        self.assertTrue(str(selection.diffusion_model or "").endswith("flux-2-klein-base-4b-Q8_0.gguf"))
        self.assertTrue(str(selection.vae or "").endswith("vae/diffusion_pytorch_model.safetensors"))
        self.assertTrue(str(selection.llm or "").endswith("Qwen3-4B-Q4_K_M.gguf"))

    def test_resolve_sdcpp_model_selection_reports_missing_companion_files_cleanly(self):
        from abstractvision.model_cache import import_directory_to_hf_cache
        from abstractvision.model_downloads import resolve_sdcpp_model_selection

        with tempfile.TemporaryDirectory() as cache_td, tempfile.TemporaryDirectory() as src_td:
            cache_root = Path(cache_td)
            src_root = Path(src_td)
            main_dir = src_root / "flux-main"
            main_dir.mkdir(parents=True, exist_ok=True)
            (main_dir / "flux-2-klein-base-4b-Q8_0.gguf").write_bytes(b"GGUF")
            import_directory_to_hf_cache(
                main_dir,
                repo_id="leejet/FLUX.2-klein-base-4B-GGUF",
                cache_dir=str(cache_root),
                cleanup_source=False,
            )

            with patch("abstractvision.model_downloads.local_model_profile", return_value="cuda"):
                with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                    with self.assertRaises(RuntimeError) as ctx:
                        resolve_sdcpp_model_selection("flux2-klein-base-4b", allow_download=False)

        self.assertIn("flux2-klein-base-4b", str(ctx.exception))
        self.assertIn("download flux2-klein-base-4b --provider sdcpp", str(ctx.exception))

    def test_resolve_sdcpp_model_selection_supports_legacy_qwen_image_repo_id(self):
        from abstractvision.model_cache import import_directory_to_hf_cache
        from abstractvision.model_downloads import resolve_sdcpp_model_selection

        with tempfile.TemporaryDirectory() as cache_td, tempfile.TemporaryDirectory() as src_td:
            cache_root = Path(cache_td)
            src_root = Path(src_td)

            main_dir = src_root / "qwen-main"
            main_dir.mkdir(parents=True, exist_ok=True)
            (main_dir / "qwen-image-Q8_0.gguf").write_bytes(b"GGUF")
            import_directory_to_hf_cache(
                main_dir,
                repo_id="unsloth/Qwen-Image-GGUF",
                cache_dir=str(cache_root),
                cleanup_source=False,
            )

            comfy_dir = src_root / "qwen-comfy"
            (comfy_dir / "split_files" / "vae").mkdir(parents=True, exist_ok=True)
            (comfy_dir / "split_files" / "vae" / "qwen_image_vae.safetensors").write_bytes(b"VAE")
            (comfy_dir / "split_files" / "text_encoders").mkdir(parents=True, exist_ok=True)
            (comfy_dir / "split_files" / "text_encoders" / "qwen_2.5_vl_7b.safetensors").write_bytes(b"LLM")
            import_directory_to_hf_cache(
                comfy_dir,
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                cache_dir=str(cache_root),
                cleanup_source=False,
            )

            with patch("abstractvision.model_downloads.local_model_profile", return_value="cuda"):
                with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                    selection = resolve_sdcpp_model_selection("unsloth/Qwen-Image-GGUF", allow_download=False)

        self.assertIsNone(selection.model)
        self.assertTrue(str(selection.diffusion_model or "").endswith("qwen-image-Q8_0.gguf"))
        self.assertTrue(str(selection.vae or "").endswith("split_files/vae/qwen_image_vae.safetensors"))
        self.assertTrue(str(selection.llm or "").endswith("split_files/text_encoders/qwen_2.5_vl_7b.safetensors"))

    def test_resolve_sdcpp_model_selection_supports_qwen_image_edit_2509_bundle(self):
        from abstractvision.model_cache import import_directory_to_hf_cache
        from abstractvision.model_downloads import resolve_sdcpp_model_selection

        with tempfile.TemporaryDirectory() as cache_td, tempfile.TemporaryDirectory() as src_td:
            cache_root = Path(cache_td)
            src_root = Path(src_td)

            main_dir = src_root / "qwen-edit-main"
            main_dir.mkdir(parents=True, exist_ok=True)
            (main_dir / "qwen-image-edit-2509-Q8_0.gguf").write_bytes(b"GGUF")
            import_directory_to_hf_cache(
                main_dir,
                repo_id="unsloth/Qwen-Image-Edit-2509-GGUF",
                cache_dir=str(cache_root),
                cleanup_source=False,
            )

            comfy_dir = src_root / "qwen-edit-comfy"
            (comfy_dir / "split_files" / "vae").mkdir(parents=True, exist_ok=True)
            (comfy_dir / "split_files" / "vae" / "qwen_image_vae.safetensors").write_bytes(b"VAE")
            (comfy_dir / "split_files" / "text_encoders").mkdir(parents=True, exist_ok=True)
            (comfy_dir / "split_files" / "text_encoders" / "qwen_2.5_vl_7b.safetensors").write_bytes(b"LLM")
            import_directory_to_hf_cache(
                comfy_dir,
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                cache_dir=str(cache_root),
                cleanup_source=False,
            )

            with patch("abstractvision.model_downloads.local_model_profile", return_value="cuda"):
                with patch.dict("os.environ", {"HF_HUB_CACHE": cache_td}, clear=True):
                    selection = resolve_sdcpp_model_selection("qwen-image-edit-2509", allow_download=False)

        self.assertIsNone(selection.model)
        self.assertTrue(str(selection.diffusion_model or "").endswith("qwen-image-edit-2509-Q8_0.gguf"))
        self.assertTrue(str(selection.vae or "").endswith("split_files/vae/qwen_image_vae.safetensors"))
        self.assertTrue(str(selection.llm or "").endswith("split_files/text_encoders/qwen_2.5_vl_7b.safetensors"))


if __name__ == "__main__":
    unittest.main()
