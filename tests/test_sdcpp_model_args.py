"""sd-cli's `--model-args` list keeps entries the caller passed and adds the backend's own.

Qwen Image Edit 2511 needs `qwen_image_zero_cond_t=true`; stable-diffusion.cpp takes it as a
`--model-args` entry (one comma-separated option), not as a flag of its own.
"""

from __future__ import annotations

import unittest

from abstractvision.backends.stable_diffusion_cpp import _add_sd_cli_model_arg


class SdCliModelArgsTests(unittest.TestCase):
    def test_adds_the_option(self):
        cmd = ["sd-cli", "-r", "in.png"]
        _add_sd_cli_model_arg(cmd, "qwen_image_zero_cond_t", "true")
        self.assertEqual(cmd[-2:], ["--model-args", "qwen_image_zero_cond_t=true"])

    def test_merges_into_an_existing_list_once(self):
        cmd = ["sd-cli", "--model-args", "chroma_use_dit_mask=false"]
        _add_sd_cli_model_arg(cmd, "qwen_image_zero_cond_t", "true")
        _add_sd_cli_model_arg(cmd, "qwen_image_zero_cond_t", "true")
        self.assertEqual(cmd, ["sd-cli", "--model-args", "chroma_use_dit_mask=false,qwen_image_zero_cond_t=true"])

    def test_keeps_the_callers_value(self):
        cmd = ["sd-cli", "--model-args", "qwen_image_zero_cond_t=false"]
        _add_sd_cli_model_arg(cmd, "qwen_image_zero_cond_t", "true")
        self.assertEqual(cmd[-1], "qwen_image_zero_cond_t=false")


    def test_a_valueless_model_args_gets_the_value_in_place(self):
        cmd = ["sd-cli", "--model-args"]
        _add_sd_cli_model_arg(cmd, "qwen_image_zero_cond_t", "true")
        self.assertEqual(cmd, ["sd-cli", "--model-args", "qwen_image_zero_cond_t=true"])
        cmd = ["sd-cli", "--model-args", "--mmap"]
        _add_sd_cli_model_arg(cmd, "qwen_image_zero_cond_t", "true")
        self.assertEqual(cmd, ["sd-cli", "--model-args", "qwen_image_zero_cond_t=true", "--mmap"])

if __name__ == "__main__":
    unittest.main()
