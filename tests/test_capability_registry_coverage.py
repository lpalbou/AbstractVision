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
        placeholder_tasks = {"multi_view_image"}
        for task in reg.list_tasks():
            if task in placeholder_tasks:
                continue
            models = reg.models_for_task(task)
            self.assertGreater(len(models), 0, msg=f"Task {task!r} has no supporting models")

    def test_every_model_declares_at_least_one_task(self):
        from abstractvision import VisionModelCapabilitiesRegistry

        reg = VisionModelCapabilitiesRegistry()
        for model_id in reg.list_models():
            spec = reg.get(model_id)
            self.assertGreater(len(spec.tasks), 0, msg=f"Model {model_id!r} declares no tasks")

    def test_every_curated_preset_has_registry_download_entry(self):
        from abstractvision import VisionModelCapabilitiesRegistry
        from abstractvision.model_downloads import model_presets

        reg = VisionModelCapabilitiesRegistry()
        presets = model_presets(target="auto", engine=None, include_non_8bit=True, include_all_targets=True)

        for preset in presets:
            model_id = str(preset.upstream_repo_id or preset.repo_id)
            spec = reg.get(model_id)
            found = False
            for dl in spec.downloads:
                if (
                    dl.key == preset.key
                    and dl.engine == preset.engine
                    and dl.target == preset.target
                    and dl.repo_id == preset.repo_id
                    and dl.bits == preset.quantization_bits
                ):
                    found = True
                    break
            self.assertTrue(
                found,
                msg=(
                    f"Missing registry download entry for preset: model_id={model_id!r} key={preset.key!r} "
                    f"engine={preset.engine!r} target={preset.target!r} bits={preset.quantization_bits!r} "
                    f"repo_id={preset.repo_id!r}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
