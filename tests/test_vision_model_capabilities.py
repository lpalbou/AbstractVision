import unittest
import sys
from pathlib import Path

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


ALL_MODELS = [
    "Qwen/Qwen-Image-2512",
    "Qwen/Qwen-Image-Edit-2511",
    "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
    "Wan-AI/Wan2.2-T2V-A14B",
    "tencent/HunyuanVideo-1.5",
    "genmo/mochi-1-preview",
    "zai-org/GLM-Image",
    "Tongyi-MAI/Z-Image-Turbo",
    "Lightricks/LTX-2",
]


class TestVisionModelCapabilitiesRegistry(unittest.TestCase):
    def test_registry_contains_all_models(self):
        from abstractvision import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        models = set(reg.list_models())
        for mid in ALL_MODELS:
            self.assertIn(mid, models)

    def test_expected_tasks(self):
        from abstractvision import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()

        self.assertTrue(reg.supports("Qwen/Qwen-Image-2512", "text_to_image"))
        self.assertTrue(reg.supports("Tongyi-MAI/Z-Image-Turbo", "text_to_image"))

        self.assertTrue(reg.supports("Qwen/Qwen-Image-Edit-2511", "image_to_image"))
        self.assertTrue(reg.supports("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA", "multi_view_image"))

        self.assertTrue(reg.supports("Wan-AI/Wan2.2-T2V-A14B", "text_to_video"))
        self.assertTrue(reg.supports("tencent/HunyuanVideo-1.5", "text_to_video"))
        self.assertTrue(reg.supports("genmo/mochi-1-preview", "text_to_video"))

        self.assertTrue(reg.supports("Lightricks/LTX-2", "image_to_video"))


if __name__ == "__main__":
    unittest.main()

