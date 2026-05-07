from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


LOCAL_RUNTIME_PACKAGES = {
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "einops",
    "peft",
    "Pillow",
    "stable-diffusion-cpp-python",
}

DIFFUSERS_RUNTIME_PACKAGES = {
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "einops",
    "peft",
    "Pillow",
}


def _extract_optional_dependency_block(text: str, key: str) -> str:
    marker = f"{key} = ["
    start = text.find(marker)
    if start == -1:
        raise AssertionError(f"Missing optional dependency block: {key}")
    end = text.find("\n]", start)
    if end == -1:
        raise AssertionError(f"Unterminated optional dependency block: {key}")
    return text[start : end + 2]


def _dependency_names(block: str) -> set[str]:
    names = set()
    for match in re.finditer(r'"([^"]+)"', block):
        req = match.group(1)
        name = re.split(r"[<>=!~;,\[]", req, maxsplit=1)[0].strip()
        if name:
            names.add(name)
    return names


class TestPackagingMetadata(unittest.TestCase):
    def test_base_dependencies_are_lightweight(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        base_deps = pyproject.split("[project.optional-dependencies]", 1)[0]

        for package in sorted(LOCAL_RUNTIME_PACKAGES):
            self.assertNotIn(package, base_deps)

    def test_runtime_extras_are_complete(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        diffusers_names = _dependency_names(_extract_optional_dependency_block(pyproject, "diffusers"))
        huggingface_names = _dependency_names(_extract_optional_dependency_block(pyproject, "huggingface"))
        sdcpp_names = _dependency_names(_extract_optional_dependency_block(pyproject, "sdcpp"))
        local_names = _dependency_names(_extract_optional_dependency_block(pyproject, "local"))
        all_names = _dependency_names(_extract_optional_dependency_block(pyproject, "all"))

        self.assertTrue(DIFFUSERS_RUNTIME_PACKAGES.issubset(diffusers_names))
        self.assertTrue(DIFFUSERS_RUNTIME_PACKAGES.issubset(huggingface_names))
        self.assertIn("stable-diffusion-cpp-python", sdcpp_names)
        self.assertIn("Pillow", sdcpp_names)
        self.assertTrue(DIFFUSERS_RUNTIME_PACKAGES.issubset(local_names))
        self.assertIn("stable-diffusion-cpp-python", local_names)
        self.assertTrue(DIFFUSERS_RUNTIME_PACKAGES.issubset(all_names))
        self.assertIn("stable-diffusion-cpp-python", all_names)

    def test_lightweight_marker_extras_and_entry_point_exist(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("openai = []", pyproject)
        self.assertIn("openai-compatible = []", pyproject)
        self.assertIn("abstractcore = []", pyproject)
        self.assertIn('[project.entry-points."abstractcore.capabilities_plugins"]', pyproject)
        self.assertIn(
            'abstractvision = "abstractvision.integrations.abstractcore_plugin:register"',
            pyproject,
        )


if __name__ == "__main__":
    unittest.main()
