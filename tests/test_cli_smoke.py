import contextlib
import io
import sys
import unittest
from pathlib import Path

# Ensure `src/` layout is importable when running tests without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class TestCliSmoke(unittest.TestCase):
    def test_models_lists_known_id(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["models"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("Qwen/Qwen-Image-2512", out)

    def test_tasks_lists_text_to_image(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["tasks"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("text_to_image", out)

    def test_show_model_prints_tasks_section(self):
        from abstractvision.cli import main

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main(["show-model", "zai-org/GLM-Image"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("zai-org/GLM-Image", out)
        self.assertIn("tasks:", out)


if __name__ == "__main__":
    unittest.main()

