import sys
import unittest
from pathlib import Path

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestCapabilityRegistryCoverage(unittest.TestCase):
    def test_schema_version_present(self):
        from abstractvision import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        self.assertTrue(str(reg.schema_version()).strip())

    def test_expected_task_keys_exist(self):
        from abstractvision import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        tasks = set(reg.list_tasks())
        self.assertTrue({"text_to_image", "image_to_image", "multi_view_image", "text_to_video", "image_to_video"}.issubset(tasks))

    def test_every_task_has_at_least_one_model(self):
        from abstractvision import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        for task in reg.list_tasks():
            models = reg.models_for_task(task)
            self.assertGreater(len(models), 0, msg=f"Task {task!r} has no supporting models")

    def test_every_model_declares_at_least_one_task(self):
        from abstractvision import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        for model_id in reg.list_models():
            spec = reg.get(model_id)
            self.assertGreater(len(spec.tasks), 0, msg=f"Model {model_id!r} declares no tasks")


if __name__ == "__main__":
    unittest.main()

