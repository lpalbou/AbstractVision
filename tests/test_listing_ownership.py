"""One owner per listing fact: public calls across modules, one memory-mode rule and mapping,
catalog notes that are the displayed text, and the engine's rows for MiniMax-H3 defaults."""

from __future__ import annotations

import ast
import contextlib
import io
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src" / "abstractvision"

# Underscore names one module still imports from another, each owned outside this change.
_REMAINING_PRIVATE_IMPORTS = {
    ("backends/openai_compatible.py", "backends.base_backend", "_option_is_set"),
    ("cli.py", "backends.stable_diffusion_cpp", "_require_sd_cli"),
    ("model_inventory.py", "model_cache", "_iter_hf_repo_snapshots"),
}


class TestNoPrivateNamesAcrossModules(unittest.TestCase):
    def test_no_module_imports_another_modules_underscore_name(self):
        found = set()
        for path in sorted(SRC.rglob("*.py")):
            relative = path.relative_to(SRC).as_posix()
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level:
                    package = Path(relative).parent.parts[: len(Path(relative).parent.parts) - (node.level - 1)]
                    module = ".".join((*package, *(node.module or "").split("."))).strip(".")
                elif (node.module or "").startswith("abstractvision."):
                    module = node.module[len("abstractvision.") :]
                else:
                    continue
                found.update((relative, module, alias.name) for alias in node.names if alias.name.startswith("_"))
        self.assertEqual(found, _REMAINING_PRIVATE_IMPORTS)

    def test_the_backend_names_the_base_model_and_the_schedules_of_a_selector(self):
        from abstractvision.backends.mflux import mlx_gen_base_model, mlx_gen_schedules

        self.assertEqual(mlx_gen_base_model("AbstractFramework/minimax-h3-8bit"), "minimax-h3")
        self.assertEqual(mlx_gen_base_model("h3-turbo"), "minimax-h3-turbo")
        self.assertIsNone(mlx_gen_base_model("someone/not-a-model"))
        self.assertEqual(
            mlx_gen_schedules("AbstractFramework/minimax-h3-4bit"),
            ("minimax-h3", "minimax-h3-turbo", "minimax-h3-turbo-544p"),
        )
        self.assertEqual(mlx_gen_schedules("someone/not-a-model"), ())

    def test_one_check_says_whether_a_base_model_is_a_schedule_of_a_package(self):
        from abstractvision import cli
        from abstractvision.backends.mflux import mlx_gen_package_schedule
        from abstractvision.model_downloads import catalog_download_preset

        preset = catalog_download_preset("AbstractFramework/minimax-h3-8bit", target="mlx", engine="mlx-gen")
        self.assertEqual(mlx_gen_package_schedule(preset, "h3-turbo-544p"), "minimax-h3-turbo-544p")
        refusal = (
            "'flux2-klein-4b' is not a schedule of AbstractFramework/minimax-h3-8bit. "
            "Choose one of: minimax-h3, minimax-h3-turbo, minimax-h3-turbo-544p."
        )
        with self.assertRaises(ValueError) as raised:
            mlx_gen_package_schedule(preset, "flux2-klein-4b")
        self.assertEqual(str(raised.exception), refusal)
        with self.assertRaises(SystemExit) as exited:
            cli.main(["download", "AbstractFramework/minimax-h3-8bit", "--base-model", "flux2-klein-4b"])
        self.assertEqual(str(exited.exception), f"--base-model {refusal}")

    def test_the_catalog_lookups_are_public(self):
        from abstractvision.model_downloads import catalog_download_preset, catalog_names_model

        preset = catalog_download_preset("AbstractFramework/minimax-h3-8bit", target="mlx", engine="mlx-gen")
        self.assertEqual(preset.quantization_bits, 8)
        self.assertIsNone(catalog_download_preset("AbstractFramework/minimax-h3-8bit", target="diffusers", engine="diffusers"))
        self.assertTrue(catalog_names_model("MiniMaxAI/MiniMax-H3"))
        self.assertFalse(catalog_names_model("someone/not-a-model"))


class TestOneMemoryModeRule(unittest.TestCase):
    def test_lowest_only_where_a_route_streams_blocks_and_never_on_seedvr2(self):
        from abstractvision.backends.mflux import mlx_gen_memory_modes

        self.assertEqual(mlx_gen_memory_modes(family="minimax-h3", streams_blocks=True), ("default", "low", "lowest"))
        self.assertEqual(mlx_gen_memory_modes(family="flux2", streams_blocks=False), ("default", "low"))
        self.assertEqual(mlx_gen_memory_modes(family="seedvr2-upscale", streams_blocks=True), ("default", "low"))

    def test_the_mapping_reads_both_ways(self):
        from abstractvision.backends.mflux import MLX_GEN_MEMORY_MODE_FLAGS, mlx_gen_memory_mode_of_run
        from abstractvision.types import MEMORY_MODES

        self.assertEqual(tuple(MLX_GEN_MEMORY_MODE_FLAGS), MEMORY_MODES)
        for mode, flags in MLX_GEN_MEMORY_MODE_FLAGS.items():
            run = {"low_ram": flags.low_ram, "stream_blocks": flags.stream_blocks}
            self.assertEqual(mlx_gen_memory_mode_of_run(run), mode)
        self.assertEqual(mlx_gen_memory_mode_of_run({}), "default")
        self.assertIsNone(mlx_gen_memory_mode_of_run({"low_ram": False, "stream_blocks": True}))

    def test_catalog_entries_list_the_modes_of_the_rule(self):
        from abstractvision.backends.mflux import _MFLUX_MODELS, _mflux_parameter_metadata

        for key, streams, expected in (
            ("seedvr2-3b", True, ["default", "low"]),
            ("flux2-klein-4b", False, ["default", "low"]),
            ("flux2-klein-4b", True, ["default", "low", "lowest"]),
            ("minimax-h3", True, ["default", "low", "lowest"]),
        ):
            constraint = _mflux_parameter_metadata(_MFLUX_MODELS[key], block_streaming=streams)["parameter_constraints"][
                "memory_mode"
            ]
            self.assertEqual(constraint, {"enum": expected}, key)


# The Diffusers snapshots whose catalog note `download` printed as this text in 0.3.37's release head.
_CURATED_DIFFUSERS_NOTE = "Official curated Diffusers snapshot (16-bit)."
_CURATED_DIFFUSERS = (
    ("ernie-image", "baidu/ERNIE-Image"),
    ("ernie-image-turbo", "baidu/ERNIE-Image-Turbo"),
    ("flux1-schnell", "black-forest-labs/FLUX.1-schnell"),
    ("flux2-klein-4b", "black-forest-labs/FLUX.2-klein-4B"),
    ("flux2-klein-9b", "black-forest-labs/FLUX.2-klein-9B"),
    ("glm-image", "zai-org/GLM-Image"),
    ("instruct-pix2pix", "timbrooks/instruct-pix2pix"),
    ("qwen-image", "Qwen/Qwen-Image-2512"),
    ("qwen-image-edit-2511", "Qwen/Qwen-Image-Edit-2511"),
    ("sd-turbo", "stabilityai/sd-turbo"),
    ("sd1.4", "CompVis/stable-diffusion-v1-4"),
    ("sd1.5-inpaint", "stable-diffusion-v1-5/stable-diffusion-inpainting"),
    ("sdxl-base", "stabilityai/stable-diffusion-xl-base-1.0"),
    ("sdxl-inpaint", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"),
    ("sdxl-refiner", "stabilityai/stable-diffusion-xl-refiner-1.0"),
    ("sdxl-turbo", "stabilityai/sdxl-turbo"),
    ("stable-diffusion", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
    ("z-image-turbo", "Tongyi-MAI/Z-Image-Turbo"),
)


class TestCatalogNotesAreTheDisplayedText(unittest.TestCase):
    def test_the_curated_diffusers_notes_are_the_text_download_prints(self):
        from abstractvision.model_downloads import _all_presets

        notes = {
            (p.key, p.repo_id): p.notes for p in _all_presets() if (p.target, p.engine) == ("diffusers", "diffusers")
        }
        for key in _CURATED_DIFFUSERS:
            self.assertEqual(notes.get(key), _CURATED_DIFFUSERS_NOTE, key)
        self.assertEqual(sum(note == _CURATED_DIFFUSERS_NOTE for note in notes.values()), len(_CURATED_DIFFUSERS))

    def test_download_prints_the_catalog_note_as_it_is(self):
        from abstractvision import cli
        from abstractvision.model_downloads import find_model_preset

        def with_note(*args, **kwargs):
            return replace(find_model_preset(*args, **kwargs), notes="#FALLBACK: full Diffusers snapshot (not 8-bit). As written.")

        out = io.StringIO()
        with patch.object(cli, "find_model_preset", side_effect=with_note), patch.object(
            cli, "download_model_preset", return_value=Path("/tmp/hf-cache/snapshot")
        ), contextlib.redirect_stdout(out):
            rc = cli.main(["download", "stable-diffusion", "--provider", "diffusers"])
        self.assertEqual(rc, 0)
        self.assertIn("\n#FALLBACK: full Diffusers snapshot (not 8-bit). As written.\n", out.getvalue())


class TestMiniMaxH3DefaultsComeFromTheEngine(unittest.TestCase):
    def test_the_model_table_holds_no_step_count_for_minimax_h3(self):
        from abstractvision.backends.mflux import _MFLUX_MODELS, MINIMAX_H3_MODEL_KEYS

        for key in MINIMAX_H3_MODEL_KEYS:
            self.assertIsNone(_MFLUX_MODELS[key].default_steps, key)
        self.assertTrue(all(d.default_steps for k, d in _MFLUX_MODELS.items() if k not in MINIMAX_H3_MODEL_KEYS))


if __name__ == "__main__":
    unittest.main()
