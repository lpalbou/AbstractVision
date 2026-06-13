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
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "prism-ml/bonsai-image-ternary-4B-mlx-2bit",
    "tencent/HunyuanVideo-1.5",
    "genmo/mochi-1-preview",
    "zai-org/CogVideoX-2b",
    "zai-org/GLM-Image",
    "Tongyi-MAI/Z-Image-Turbo",
    "briaai/FIBO",
    "briaai/Fibo-lite",
    "briaai/Fibo-Edit",
    "briaai/Fibo-Edit-RMBG",
    "Lightricks/LTX-2",
    "ByteDance-Seed/SeedVR2-3B",
    "ByteDance-Seed/SeedVR2-7B",
]

ABSTRACTFRAMEWORK_HF_REPOS = [
    "AbstractFramework/ernie-image-turbo-4bit",
    "AbstractFramework/ernie-image-turbo-8bit",
    "AbstractFramework/fibo-4bit",
    "AbstractFramework/fibo-8bit",
    "AbstractFramework/flux.2-klein-4b-4bit",
    "AbstractFramework/flux.2-klein-4b-8bit",
    "AbstractFramework/flux.2-klein-9b-4bit",
    "AbstractFramework/flux.2-klein-9b-8bit",
    "AbstractFramework/flux.2-klein-base-4b-4bit",
    "AbstractFramework/flux.2-klein-base-4b-8bit",
    "AbstractFramework/flux.2-klein-base-9b-4bit",
    "AbstractFramework/flux.2-klein-base-9b-8bit",
    "AbstractFramework/qwen-image-2512-4bit",
    "AbstractFramework/qwen-image-2512-8bit",
    "AbstractFramework/qwen-image-4bit",
    "AbstractFramework/qwen-image-8bit",
    "AbstractFramework/qwen-image-edit-2509-4bit",
    "AbstractFramework/qwen-image-edit-2509-8bit",
    "AbstractFramework/qwen-image-edit-2511-4bit",
    "AbstractFramework/qwen-image-edit-2511-8bit",
    "AbstractFramework/qwen-image-edit-4bit",
    "AbstractFramework/qwen-image-edit-8bit",
    "AbstractFramework/seedvr2-3b-4bit",
    "AbstractFramework/seedvr2-3b-8bit",
    "AbstractFramework/seedvr2-7b-4bit",
    "AbstractFramework/seedvr2-7b-8bit",
    "AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit",
    "AbstractFramework/wan2.2-i2v-a14b-diffusers-bf16",
    "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit",
    "AbstractFramework/wan2.2-t2v-a14b-diffusers-bf16",
    "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit",
    "AbstractFramework/wan2.2-ti2v-5b-diffusers-bf16",
    "AbstractFramework/z-image-4bit",
    "AbstractFramework/z-image-8bit",
    "AbstractFramework/z-image-turbo-4bit",
    "AbstractFramework/z-image-turbo-8bit",
]


class TestVisionModelCapabilitiesRegistry(unittest.TestCase):
    def test_registry_contains_all_models(self):
        from abstractvision import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        models = set(reg.list_models())
        for mid in ALL_MODELS:
            self.assertIn(mid, models)

    def test_registry_downloads_cover_abstractframework_hf_repos(self):
        from abstractvision import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        registered_repos = {
            download.repo_id
            for model_id in reg.list_models()
            for download in reg.get(model_id).downloads
            if download.repo_id.startswith("AbstractFramework/")
        }

        self.assertEqual(
            sorted(set(ABSTRACTFRAMEWORK_HF_REPOS) - registered_repos),
            [],
        )

    def test_expected_tasks(self):
        from abstractvision import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()

        self.assertTrue(reg.supports("Qwen/Qwen-Image-2512", "text_to_image"))
        self.assertTrue(reg.supports("Tongyi-MAI/Z-Image-Turbo", "text_to_image"))
        self.assertTrue(reg.supports("prism-ml/bonsai-image-ternary-4B-mlx-2bit", "text_to_image"))

        self.assertTrue(reg.supports("Qwen/Qwen-Image-Edit-2511", "image_to_image"))
        self.assertTrue(reg.supports("baidu/ERNIE-Image-Turbo", "image_to_image"))
        self.assertTrue(reg.supports("briaai/FIBO", "text_to_image"))
        self.assertTrue(reg.supports("briaai/FIBO", "image_to_image"))
        self.assertTrue(reg.supports("briaai/Fibo-lite", "text_to_image"))
        self.assertTrue(reg.supports("briaai/Fibo-Edit", "image_to_image"))
        self.assertTrue(reg.supports("briaai/Fibo-Edit-RMBG", "image_to_image"))
        self.assertIn("multi_view_image", reg.list_tasks())
        self.assertEqual(reg.models_for_task("multi_view_image"), [])

        self.assertTrue(reg.supports("Wan-AI/Wan2.2-T2V-A14B", "text_to_video"))
        self.assertTrue(reg.supports("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "text_to_video"))
        self.assertTrue(reg.supports("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "image_to_video"))
        self.assertTrue(reg.supports("AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit", "text_to_video"))
        self.assertTrue(reg.supports("AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit", "image_to_video"))
        self.assertTrue(reg.supports("tencent/HunyuanVideo-1.5", "text_to_video"))
        self.assertTrue(reg.supports("genmo/mochi-1-preview", "text_to_video"))
        self.assertTrue(reg.supports("zai-org/CogVideoX-2b", "text_to_video"))

        self.assertTrue(reg.supports("Lightricks/LTX-2", "image_to_video"))
        self.assertTrue(reg.supports("ByteDance-Seed/SeedVR2-3B", "image_upscale"))
        self.assertTrue(reg.supports("ByteDance-Seed/SeedVR2-7B", "image_upscale"))
        self.assertTrue(reg.supports("AbstractFramework/seedvr2-3b-8bit", "image_upscale"))
        self.assertTrue(reg.supports("AbstractFramework/seedvr2-7b-4bit", "image_upscale"))
        self.assertTrue(reg.supports("seedvr2-3b", "image_upscale"))
        self.assertTrue(reg.supports("seedvr2-7b", "image_upscale"))

    def test_seedvr2_upscale_capability_and_defaults_are_explicit(self):
        from abstractvision import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        spec = reg.get("ByteDance-Seed/SeedVR2-3B")
        self.assertIs(reg.get("seedvr2-3b"), spec)
        self.assertIs(reg.get("AbstractFramework/seedvr2-3b-8bit"), spec)
        task = spec.tasks["image_upscale"]

        self.assertEqual(list(task.inputs), ["image"])
        self.assertEqual(list(task.outputs), ["image"])
        self.assertEqual(task.params["resolution"]["default"], "2x")
        self.assertEqual(task.params["scale"]["default"], 2)
        self.assertEqual(task.params["softness"]["default"], 0.25)
        self.assertIsNone(task.params["quantize"]["default"])
        self.assertEqual(task.params["quantize"]["enum"], [3, 4, 5, 6, 8, None])
        self.assertEqual(task.requires["backend"], "mlx-gen")
        self.assertEqual(task.requires["min_runtime_version"], "0.18.13")

        spec_7b = reg.get("ByteDance-Seed/SeedVR2-7B")
        self.assertIs(reg.get("seedvr2-7b"), spec_7b)
        self.assertIs(reg.get("AbstractFramework/seedvr2-7b-8bit"), spec_7b)

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
