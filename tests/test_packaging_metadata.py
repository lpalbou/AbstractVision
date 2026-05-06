from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _extract_optional_dependency_block(text: str, key: str) -> str:
    marker = f"{key} = ["
    start = text.find(marker)
    if start == -1:
        raise AssertionError(f"Missing optional dependency block: {key}")
    end = text.find("\n]", start)
    if end == -1:
        raise AssertionError(f"Unterminated optional dependency block: {key}")
    return text[start : end + 2]


class TestPackagingMetadata(unittest.TestCase):
    def test_sdcpp_binding_is_not_a_base_dependency(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        base_deps = pyproject.split("[project.optional-dependencies]", 1)[0]
        sdcpp_block = _extract_optional_dependency_block(pyproject, "sdcpp")
        local_block = _extract_optional_dependency_block(pyproject, "local")

        self.assertNotIn("stable-diffusion-cpp-python", base_deps)
        self.assertIn("stable-diffusion-cpp-python>=0.4.2", sdcpp_block)
        self.assertIn("stable-diffusion-cpp-python>=0.4.2", local_block)


if __name__ == "__main__":
    unittest.main()
