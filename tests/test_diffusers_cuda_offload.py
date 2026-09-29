"""CUDA: a pipeline that does not fit the GPU loads with model CPU offload, and with sequential CPU
offload when not even its largest component fits (framework backlog 0989).

Measured on a 16 GB Quadro RTX 5000: FLUX.2 [klein] 4B in float16 (14.9 GiB of weights) failed with
CUDA out of memory when moved whole to the GPU; with Diffusers' model CPU offload it generated
768x768 in 17 s, 7.8 GiB peak. With another process holding 6.5 GiB, sequential CPU offload peaked
near 1.4 GiB (10.9-15.6 s at 4 steps against 13.4-18.5 s with model CPU offload).
No GPU is needed here: free memory is simulated, and the FLUX-sized modules live on torch's "meta"
device (shapes without storage).
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

    def enable_sequential_cpu_offload(self, device=None):
        self.offloaded_on = device


class _FluxKleinPipe(_Pipe):
    """The FLUX.2 [klein] 4B component sizes in float16 (14.9 GiB in all), on the meta device."""

    def __init__(self):
        def module(gib_fp16):
            rows = int(gib_fp16 * GIB / 2 / 4096)  # float16 bytes -> Linear(4096 -> rows) weight
            return torch.nn.Linear(4096, rows, bias=False, device="meta")

        self.components = {
            "text_encoder": module(7.49),  # Qwen3 4B
            "transformer": module(7.25),
            "vae": module(0.16),
            "scheduler": object(),
        }
        self.offloaded_on = None


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
        self.assertEqual(offload, "model")
        self.assertIn("do not fit", why)

    def test_pipeline_that_fits_is_moved_whole(self):
        pipe = _Pipe([64, 64])
        with _gpu(free_gib=14.4, total_gib=15.0):
            self.assertEqual(hd._cuda_offload_decision(torch, pipe, "cuda:0", torch.float16, "auto"), ("", ""))

    # FLUX.2 [klein] 4B on the rehearsal's 16 GB card (14.56 GiB visible): the three placements.

    def test_flux_klein_with_another_model_on_the_gpu_uses_sequential_offload(self):
        # Another process holds 6.5 GiB: 7.77 GiB free - 1.5 GiB reserve = 6.27 GiB usable, below
        # the 7.49 GiB text encoder that model CPU offload would move to the GPU whole.
        with _gpu(free_gib=7.77, total_gib=14.56):
            offload, why = hd._cuda_offload_decision(torch, _FluxKleinPipe(), "cuda", torch.float16, "auto")
        self.assertEqual(offload, "sequential")
        self.assertIn("text_encoder", why)
        self.assertIn("much less GPU memory", why)
        self.assertIn("each step is slower", why)
        self.assertNotIn("much slower", why)

    def test_flux_klein_on_a_free_gpu_uses_model_offload(self):
        # The measured free GPU: 14.2 GiB free, 12.7 GiB usable. The largest component fits, the
        # 14.9 GiB pipeline does not (model CPU offload measured 16 s, 7.8 GiB peak).
        with _gpu(free_gib=14.2, total_gib=14.56):
            offload, why = hd._cuda_offload_decision(torch, _FluxKleinPipe(), "cuda", torch.float16, "auto")
        self.assertEqual(offload, "model")
        self.assertIn("14.9 GiB", why)

    def test_flux_klein_on_a_24_gb_gpu_is_moved_whole(self):
        with _gpu(free_gib=23.0, total_gib=24.0):  # reserve 2.4 GiB -> 20.6 GiB usable
            self.assertEqual(
                hd._cuda_offload_decision(torch, _FluxKleinPipe(), "cuda", torch.float16, "auto"), ("", "")
            )

    def test_without_sequential_offload_support_auto_keeps_model_offload(self):
        pipe = _FluxKleinPipe()
        pipe.enable_sequential_cpu_offload = None
        with _gpu(free_gib=7.77, total_gib=14.56):
            self.assertEqual(hd._cuda_offload_decision(torch, pipe, "cuda", torch.float16, "auto")[0], "model")

    def test_explicit_sequential_without_support_logs_and_uses_model_offload(self):
        pipe = _Pipe([64])
        pipe.enable_sequential_cpu_offload = None
        with _gpu(free_gib=14.4, total_gib=15.0):
            with self.assertLogs(hd.logger, level="WARNING") as logs:
                offload, reason = hd._cuda_offload_decision(torch, pipe, "cuda", torch.float16, "sequential")
        self.assertEqual(offload, "model")
        self.assertIn("not supported", reason)
        self.assertTrue(any("does not support it" in line for line in logs.output), logs.output)
        # Neither offload available: it says so and moves the pipeline whole.
        pipe.enable_model_cpu_offload = None
        with _gpu(free_gib=14.4, total_gib=15.0):
            with self.assertLogs(hd.logger, level="WARNING") as logs:
                self.assertEqual(hd._cuda_offload_decision(torch, pipe, "cuda", torch.float16, "sequential"), ("", ""))
        self.assertTrue(any("the whole pipeline on cuda" in line for line in logs.output), logs.output)

    def test_modes_and_non_cuda_devices(self):
        pipe = _Pipe([64])
        with _gpu(free_gib=14.4, total_gib=15.0):
            self.assertEqual(hd._cuda_offload_decision(torch, pipe, "cuda", torch.float16, "model")[0], "model")
            self.assertEqual(
                hd._cuda_offload_decision(torch, pipe, "cuda", torch.float16, "sequential")[0], "sequential"
            )
        with _gpu(free_gib=0.1, total_gib=15.0):
            # "model" is explicit: it never falls back to sequential offload.
            self.assertEqual(hd._cuda_offload_decision(torch, pipe, "cuda", torch.float16, "model")[0], "model")
            self.assertEqual(hd._cuda_offload_decision(torch, pipe, "cuda", torch.float16, "none")[0], "")
            for device in ("cpu", "mps"):
                for mode in ("auto", "model", "sequential"):
                    self.assertEqual(hd._cuda_offload_decision(torch, pipe, device, torch.float16, mode)[0], "")


class CudaOffloadWiringTests(unittest.TestCase):
    def test_config_defaults_to_auto(self):
        self.assertEqual(hd.HuggingFaceDiffusersBackendConfig(model_id="x").cpu_offload, "auto")

    def test_load_path_uses_offload_instead_of_moving_the_pipeline(self):
        source = inspect.getsource(hd.HuggingFaceDiffusersVisionBackend)
        load = source.split("offload, offload_reason = _cuda_offload_decision(", 1)[1]
        self.assertLess(load.index("enable_model_cpu_offload"), load.index("_move_pipe_to_device"))
        self.assertIn("if not offload:", load.split("_move_pipe_to_device", 1)[0])
        self.assertLess(load.index("enable_sequential_cpu_offload"), load.index("_move_pipe_to_device"))

    def test_result_metadata_reports_the_offload_mode(self):
        class P:
            pass

        pipe = P()
        self.assertEqual(hd._pipe_cpu_offload_mode(pipe), "")
        pipe._abstractvision_cpu_offload = True
        self.assertEqual(hd._pipe_cpu_offload_mode(pipe), "model")
        pipe._abstractvision_cpu_offload_mode = "sequential"
        self.assertEqual(hd._pipe_cpu_offload_mode(pipe), "sequential")
        source = inspect.getsource(hd.HuggingFaceDiffusersVisionBackend)
        self.assertNotIn('meta["cpu_offload"] = "model"', source)
        self.assertEqual(source.count('meta["cpu_offload"] = offload_mode'), 3)

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
