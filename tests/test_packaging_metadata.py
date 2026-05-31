import re
import unittest
from pathlib import Path

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
    "mlx-gen",
    "mlx",
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
    line_end = text.find("\n", start)
    if line_end != -1 and text[start:line_end].strip().endswith("[]"):
        return text[start:line_end]
    end = text.find("\n]", start)
    if end == -1:
        raise AssertionError(f"Unterminated optional dependency block: {key}")
    return text[start : end + 2]


def _dependency_requirements(block: str) -> set[str]:
    return {match.group(1) for match in re.finditer(r'"([^"]+)"', block)}


def _optional_dependency_requirements(text: str, key: str) -> set[str]:
    return _dependency_requirements(_extract_optional_dependency_block(text, key))


def _dependency_names(requirements: set[str]) -> set[str]:
    names = set()
    for requirement in requirements:
        name = re.split(r"[<>=!~;,\[]", requirement, maxsplit=1)[0].strip()
        if name:
            names.add(name)
    return names


class TestPackagingMetadata(unittest.TestCase):
    def test_base_dependencies_are_lightweight(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        base_deps = pyproject.split("[project.optional-dependencies]", 1)[0]

        self.assertIn("dependencies = []", base_deps)
        for package in sorted(LOCAL_RUNTIME_PACKAGES):
            self.assertNotIn(package, base_deps)

    def test_runtime_extras_are_complete(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        diffusers_names = _dependency_names(
            _optional_dependency_requirements(pyproject, "diffusers")
        )
        huggingface_names = _dependency_names(
            _optional_dependency_requirements(pyproject, "huggingface")
        )
        sdcpp_names = _dependency_names(_optional_dependency_requirements(pyproject, "sdcpp"))
        mflux_names = _dependency_names(_optional_dependency_requirements(pyproject, "mflux"))
        local_names = _dependency_names(_optional_dependency_requirements(pyproject, "local"))
        apple_names = _dependency_names(_optional_dependency_requirements(pyproject, "apple"))
        gpu_names = _dependency_names(_optional_dependency_requirements(pyproject, "gpu"))
        all_names = _dependency_names(_optional_dependency_requirements(pyproject, "all"))
        all_apple_names = _dependency_names(_optional_dependency_requirements(pyproject, "all-apple"))
        all_gpu_names = _dependency_names(_optional_dependency_requirements(pyproject, "all-gpu"))

        self.assertTrue(DIFFUSERS_RUNTIME_PACKAGES.issubset(diffusers_names))
        self.assertTrue(DIFFUSERS_RUNTIME_PACKAGES.issubset(huggingface_names))
        self.assertIn("stable-diffusion-cpp-python", sdcpp_names)
        self.assertIn("Pillow", sdcpp_names)
        self.assertIn("mlx-gen", mflux_names)
        self.assertTrue(DIFFUSERS_RUNTIME_PACKAGES.issubset(local_names))
        self.assertIn("stable-diffusion-cpp-python", local_names)
        self.assertNotIn("mlx-gen", local_names)
        self.assertTrue(DIFFUSERS_RUNTIME_PACKAGES.issubset(apple_names))
        self.assertIn("stable-diffusion-cpp-python", apple_names)
        self.assertIn("mlx-gen", apple_names)
        self.assertTrue(DIFFUSERS_RUNTIME_PACKAGES.issubset(gpu_names))
        self.assertNotIn("stable-diffusion-cpp-python", gpu_names)
        self.assertNotIn("mlx-gen", gpu_names)
        self.assertTrue(DIFFUSERS_RUNTIME_PACKAGES.issubset(all_names))
        self.assertIn("stable-diffusion-cpp-python", all_names)
        self.assertIn("mlx-gen", all_names)
        self.assertEqual(apple_names, all_apple_names)
        self.assertIn("mlx-gen", all_apple_names)
        self.assertEqual(local_names, all_gpu_names)
        self.assertNotIn("mlx-gen", all_gpu_names)

    def test_runtime_aliases_and_bundles_do_not_drift(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        diffusers = _optional_dependency_requirements(pyproject, "diffusers")
        huggingface = _optional_dependency_requirements(pyproject, "huggingface")
        sdcpp = _optional_dependency_requirements(pyproject, "sdcpp")
        mflux = _optional_dependency_requirements(pyproject, "mflux")
        local = _optional_dependency_requirements(pyproject, "local")
        all_runtime = _optional_dependency_requirements(pyproject, "all")
        apple = _optional_dependency_requirements(pyproject, "apple")
        gpu = _optional_dependency_requirements(pyproject, "gpu")
        all_apple = _optional_dependency_requirements(pyproject, "all-apple")
        all_gpu = _optional_dependency_requirements(pyproject, "all-gpu")
        diffusers_dev = _optional_dependency_requirements(pyproject, "diffusers-dev")
        huggingface_dev = _optional_dependency_requirements(pyproject, "huggingface-dev")

        self.assertEqual(diffusers, huggingface)
        self.assertEqual(diffusers | sdcpp, local)
        self.assertEqual(local | mflux, apple)
        self.assertEqual(diffusers, gpu)
        self.assertEqual(local | mflux, all_runtime)
        self.assertEqual(apple, all_apple)
        self.assertEqual(local, all_gpu)
        self.assertEqual(diffusers_dev, huggingface_dev)

        contributor_only = {
            "pytest",
            "mkdocs",
            "mkdocs-material",
            "build",
            "twine",
            "ruff",
            "black",
            "pre-commit",
        }
        self.assertFalse(contributor_only.intersection(_dependency_names(all_runtime)))

    def test_mlx_gen_extra_uses_current_runtime_floor(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        expected = "mlx-gen>=0.18.8,<0.19.0; platform_system == 'Darwin' and python_version >= '3.10'"

        for extra in ("mlx-gen", "mflux", "apple", "all", "all-apple"):
            self.assertIn(expected, _optional_dependency_requirements(pyproject, extra))

    def test_sdcpp_binding_extras_avoid_known_broken_sdist(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        expected = "stable-diffusion-cpp-python>=0.4.2,<0.4.6"

        for extra in ("sdcpp", "local", "apple", "all", "all-apple", "all-gpu"):
            self.assertIn(expected, _optional_dependency_requirements(pyproject, extra))

        self.assertNotIn(
            "stable-diffusion-cpp-python",
            _dependency_names(_optional_dependency_requirements(pyproject, "gpu")),
        )

    def test_lightweight_marker_extras_and_entry_point_exist(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        for extra in ("openai", "openai-compatible", "abstractcore"):
            self.assertEqual(set(), _optional_dependency_requirements(pyproject, extra))
        self.assertIn('[project.entry-points."abstractcore.capabilities_plugins"]', pyproject)
        self.assertIn(
            'abstractvision = "abstractvision.integrations.abstractcore_plugin:register"',
            pyproject,
        )


if __name__ == "__main__":
    unittest.main()
