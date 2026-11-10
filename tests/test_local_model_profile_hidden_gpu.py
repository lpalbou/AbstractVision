"""`local_model_profile()` says "cpu" when CUDA_VISIBLE_DEVICES hides every GPU.

Measured on Linux with an NVIDIA GPU (Quadro RTX 5000, 2026-11-10): with
`CUDA_VISIBLE_DEVICES=` torch saw no GPU and the Diffusers backend's device "auto" resolved
to cpu, but `local_model_profile()` still answered "cuda" (it asks `nvidia-smi -L`, which ignores
the variable), so the catalog's auto download target was the FP8 side artifact
(`black-forest-labs/FLUX.2-klein-4b-fp8`). Dropping the CUDA_VISIBLE_DEVICES check turns this RED.
No GPU needed: `nvidia-smi` is simulated.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from unittest.mock import patch

from abstractvision import model_downloads as md


def _nvidia_smi_lists_a_gpu(*args, **kwargs):
    return subprocess.CompletedProcess(args[0], 0, stdout="GPU 0: Quadro RTX 5000 (UUID: GPU-x)\n", stderr="")


class LocalModelProfileHiddenGpuTests(unittest.TestCase):
    def _profile(self, env):
        # The profile is cached per process; never leave a simulated answer behind.
        md.local_model_profile.cache_clear()
        try:
            with patch.object(md.sys, "platform", "linux"), \
                 patch.object(md.platform, "machine", return_value="x86_64"), \
                 patch.object(md.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                 patch.object(md.subprocess, "run", side_effect=_nvidia_smi_lists_a_gpu), \
                 patch.dict(os.environ, env, clear=False):
                if "CUDA_VISIBLE_DEVICES" not in env:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                return md.local_model_profile()
        finally:
            md.local_model_profile.cache_clear()

    def test_visible_gpu_is_cuda(self):
        self.assertEqual(self._profile({}), "cuda")
        self.assertEqual(self._profile({"CUDA_VISIBLE_DEVICES": "0"}), "cuda")

    def test_hidden_gpu_is_cpu(self):
        for value in ("", " ", "-1"):
            with self.subTest(CUDA_VISIBLE_DEVICES=value):
                self.assertEqual(self._profile({"CUDA_VISIBLE_DEVICES": value}), "cpu")


if __name__ == "__main__":
    unittest.main()
