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
    "Wan-AI/Wan2.2-T2V-A14B",
    "tencent/HunyuanVideo-1.5",
    "genmo/mochi-1-preview",
    "zai-org/CogVideoX-2b",
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
        self.assertIn("multi_view_image", reg.list_tasks())
        self.assertEqual(reg.models_for_task("multi_view_image"), [])

        self.assertTrue(reg.supports("Wan-AI/Wan2.2-T2V-A14B", "text_to_video"))
        self.assertTrue(reg.supports("tencent/HunyuanVideo-1.5", "text_to_video"))
        self.assertTrue(reg.supports("genmo/mochi-1-preview", "text_to_video"))
        self.assertTrue(reg.supports("zai-org/CogVideoX-2b", "text_to_video"))

        self.assertTrue(reg.supports("Lightricks/LTX-2", "image_to_video"))

    def test_glm_edit_capability_and_defaults_are_explicit(self):
        from abstractvision import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        spec = reg.get("zai-org/GLM-Image")

        self.assertIn("image_to_image", spec.tasks)
        self.assertEqual(spec.tasks["text_to_image"].params["steps"]["default"], 50)
        self.assertEqual(spec.tasks["text_to_image"].params["guidance_scale"]["default"], 1.5)
        self.assertEqual(spec.tasks["text_to_image"].params["width"]["multiple_of"], 32)
        self.assertTrue(spec.tasks["image_to_image"].params["width"]["required"])
        self.assertEqual(spec.tasks["image_to_image"].params["steps"]["default"], 15)


if __name__ == "__main__":
    unittest.main()
