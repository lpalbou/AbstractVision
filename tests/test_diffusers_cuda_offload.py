"""CUDA: a pipeline that does not fit the GPU loads with model CPU offload (framework backlog 0989).

Measured on a 16 GB Quadro RTX 5000: FLUX.2 [klein] 4B in float16 (14.9 GiB of weights) failed with
CUDA out of memory when moved whole to the GPU; with Diffusers' model CPU offload it generated
768x768 in 17 s, 7.8 GiB peak. No GPU is needed here: free memory is simulated.
"""

from __future__ import annotations

import inspect
import unittest
from unittest import mock

try:
    import torch
except ImportError:  # the base CI job has no torch
    torch = None

from abstractvision.backends import huggingface_diffusers as hd

GIB = 1024**3


class _Pipe:
    def __init__(self, sizes_mib):
        # float32 Linear weights of the requested sizes (MiB), one per component.
        self.components = {
            f"c{i}": torch.nn.Linear(int(mib * 1024 * 1024 / 4 / 256), 256, bias=False)
            for i, mib in enumerate(sizes_mib)
        }
        self.components["scheduler"] = object()
        self.offloaded_on = None

    def enable_model_cpu_offload(self, device=None):
        self.offloaded_on = device


def _gpu(free_gib, total_gib):
    return mock.patch.object(
        torch.cuda, "mem_get_info", lambda index=0: (int(free_gib * GIB), int(total_gib * GIB))
    )


@unittest.skipIf(torch is None, "torch is not installed")
class CudaOffloadDecisionTests(unittest.TestCase):
    def test_module_bytes_counts_every_component_at_the_target_dtype(self):
        pipe = _Pipe([64, 32])
        self.assertEqual(hd._pipe_module_bytes(pipe, torch), 96 * 1024 * 1024)
        self.assertEqual(hd._pipe_module_bytes(pipe, torch, torch.float16), 48 * 1024 * 1024)

    def test_pipeline_larger_than_free_vram_is_offloaded(self):
        pipe = _Pipe([64, 64])  # 128 MiB fp32, 64 MiB fp16
        with _gpu(free_gib=1.55, total_gib=2.0):  # reserve 1.5 GiB -> 0.05 GiB (51 MiB) usable
            offload, why = hd._cuda_offload_decision(torch, pipe, "cuda", torch.float16, "auto")
        self.assertTrue(offload)
        self.assertIn("do not fit", why)

    def test_pipeline_that_fits_is_moved_whole(self):
        pipe = _Pipe([64, 64])
        with _gpu(free_gib=14.4, total_gib=15.0):
            self.assertEqual(
                hd._cuda_offload_decision(torch, pipe, "cuda:0", torch.float16, "auto"), (False, "")
            )

    def test_modes_and_non_cuda_devices(self):
        pipe = _Pipe([64])
        with _gpu(free_gib=14.4, total_gib=15.0):
            self.assertIs(hd._cuda_offload_decision(torch, pipe, "cuda", torch.float16, "model")[0], True)
        with _gpu(free_gib=0.1, total_gib=15.0):
            self.assertIs(hd._cuda_offload_decision(torch, pipe, "cuda", torch.float16, "none")[0], False)
            for device in ("cpu", "mps"):
                self.assertIs(hd._cuda_offload_decision(torch, pipe, device, torch.float16, "auto")[0], False)


class CudaOffloadWiringTests(unittest.TestCase):
    def test_config_defaults_to_auto(self):
        self.assertEqual(hd.HuggingFaceDiffusersBackendConfig(model_id="x").cpu_offload, "auto")

    def test_load_path_uses_offload_instead_of_moving_the_pipeline(self):
        source = inspect.getsource(hd.HuggingFaceDiffusersVisionBackend)
        load = source.split("offload, offload_reason = _cuda_offload_decision(", 1)[1]
        self.assertLess(load.index("enable_model_cpu_offload"), load.index("_move_pipe_to_device"))
        self.assertIn("if not offload:", load.split("_move_pipe_to_device", 1)[0])

    def test_offloaded_pipeline_reports_its_execution_device(self):
        class P:
            device = "cpu"
            _execution_device = "cuda:0"

        pipe = P()
        self.assertEqual(hd._pipe_compute_device(pipe), "cpu")
        pipe._abstractvision_cpu_offload = True
        self.assertEqual(hd._pipe_compute_device(pipe), "cuda:0")


if __name__ == "__main__":
    unittest.main()
