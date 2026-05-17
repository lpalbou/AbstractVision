import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestModelDownloads(unittest.TestCase):
    def test_auto_scope_prefers_apple_targets_and_mflux_qwen(self):
        from abstractvision.model_downloads import catalog_target_scope, find_model_preset

        with patch("abstractvision.model_downloads.local_model_profile", return_value="apple-silicon"):
            self.assertEqual(
                catalog_target_scope(target="auto", engine=None, include_all_targets=False),
                ("mlx", "diffusers", "hf-snapshot"),
            )
            preset = find_model_preset("qwen-image", target="auto", engine=None, require_8bit=True)
        self.assertEqual((preset.target, preset.engine), ("mlx", "mflux"))

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


if __name__ == "__main__":
    unittest.main()
