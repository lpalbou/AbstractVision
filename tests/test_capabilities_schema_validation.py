import json
import sys
import unittest
from pathlib import Path

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestCapabilitiesSchemaValidation(unittest.TestCase):
    def test_valid_asset_passes_validation(self):
        from abstractvision.model_capabilities import validate_capabilities_json

        p = REPO_ROOT / "src" / "abstractvision" / "assets" / "vision_model_capabilities.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        validate_capabilities_json(data)

    def test_missing_schema_version(self):
        from abstractvision.model_capabilities import validate_capabilities_json

        with self.assertRaises(ValueError) as ctx:
            validate_capabilities_json({"tasks": {}, "models": {}})
        self.assertIn("schema_version", str(ctx.exception))

    def test_unknown_task_reference_fails(self):
        from abstractvision.model_capabilities import validate_capabilities_json

        data = {
            "schema_version": "1.0",
            "tasks": {"text_to_image": {"description": "t2i"}},
            "models": {
                "m1": {
                    "provider": "huggingface",
                    "license": "apache-2.0",
                    "tasks": {
                        "not_a_task": {
                            "inputs": ["text"],
                            "outputs": ["image"],
                            "params": {"prompt": {"required": True}},
                        }
                    },
                }
            },
        }
        with self.assertRaises(ValueError) as ctx:
            validate_capabilities_json(data)
        msg = str(ctx.exception)
        self.assertIn("unknown task", msg)
        self.assertIn("models", msg)

    def test_param_required_must_be_bool(self):
        from abstractvision.model_capabilities import validate_capabilities_json

        data = {
            "schema_version": "1.0",
            "tasks": {"text_to_image": {"description": "t2i"}},
            "models": {
                "m1": {
                    "provider": "huggingface",
                    "license": "apache-2.0",
                    "tasks": {
                        "text_to_image": {
                            "inputs": ["text"],
                            "outputs": ["image"],
                            "params": {"prompt": {"required": "yes"}},
                        }
                    },
                }
            },
        }
        with self.assertRaises(ValueError) as ctx:
            validate_capabilities_json(data)
        self.assertIn("required", str(ctx.exception))

    def test_base_model_reference_must_exist(self):
        from abstractvision.model_capabilities import validate_capabilities_json

        data = {
            "schema_version": "1.0",
            "tasks": {"multi_view_image": {"description": "mv"}},
            "models": {
                "lora": {
                    "provider": "huggingface",
                    "license": "apache-2.0",
                    "tasks": {
                        "multi_view_image": {
                            "inputs": ["text"],
                            "outputs": ["image[]"],
                            "requires": {"base_model_id": "missing/base"},
                            "params": {"prompt": {"required": True}},
                        }
                    },
                }
            },
        }
        with self.assertRaises(ValueError) as ctx:
            validate_capabilities_json(data)
        self.assertIn("base_model_id", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

