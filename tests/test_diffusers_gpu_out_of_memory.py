"""A CUDA out-of-memory during generation reads as a clear AbstractVision error.

Measured on a 16 GB Quadro RTX 5000 (2026-11-10): Z-Image in bfloat16 with model CPU offload at
1024x1024 failed with torch's raw "CUDA out of memory. Tried to allocate 3.75 GiB ..." traceback.
Removing the out-of-memory branch of `_pipe_call` turns this RED. No GPU: the pipeline raises the
error torch raises.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # the base CI job has no torch
    torch = None

from abstractvision.backends import huggingface_diffusers as hd
from abstractvision.errors import AbstractVisionError, GpuOutOfMemoryError

_TORCH_MESSAGE = (
    "CUDA out of memory. Tried to allocate 3.75 GiB. GPU 0 has a total capacity of 14.56 GiB of which "
    "1.41 GiB is free."
)


class _OomPipe:
    _abstractvision_cpu_offload = True
    _abstractvision_cpu_offload_mode = "model"

    def __call__(self, **kwargs):
        raise torch.OutOfMemoryError(_TORCH_MESSAGE)


@unittest.skipIf(torch is None or not hasattr(torch, "OutOfMemoryError"), "torch with OutOfMemoryError needed")
class DiffusersGpuOutOfMemoryTests(unittest.TestCase):
    def test_out_of_memory_is_a_clear_package_error(self):
        backend = hd.HuggingFaceDiffusersVisionBackend(
            config=hd.HuggingFaceDiffusersBackendConfig(model_id="Tongyi-MAI/Z-Image", device="cpu")
        )
        with self.assertRaises(GpuOutOfMemoryError) as ctx:
            backend._pipe_call(_OomPipe(), {"prompt": "fox", "width": 1024, "height": 1024})
        msg = str(ctx.exception)
        self.assertIsInstance(ctx.exception, AbstractVisionError)
        self.assertIn("Tongyi-MAI/Z-Image at 1024x1024 (model CPU offload)", msg)
        self.assertIn("it needed 3.75 GiB more", msg)
        self.assertIn("cpu_offload='sequential'", msg)
        self.assertIsInstance(ctx.exception.__cause__, torch.OutOfMemoryError)

    def test_the_failed_step_tensors_are_released_before_measuring(self):
        import weakref

        class _Activation:
            pass

        refs = []

        class _HoldingPipe(_OomPipe):
            def __call__(self, **kwargs):
                activation = _Activation()  # stands for the failed step's GPU tensors
                refs.append(weakref.ref(activation))
                raise torch.OutOfMemoryError(_TORCH_MESSAGE)

        backend = hd.HuggingFaceDiffusersVisionBackend(
            config=hd.HuggingFaceDiffusersBackendConfig(model_id="x/y", device="cpu")
        )
        held = None
        try:
            backend._pipe_call(_HoldingPipe(), {"prompt": "fox"})
        except GpuOutOfMemoryError as e:  # not assertRaises: it clears the frames itself
            held = e
        self.assertIsNotNone(held)
        self.assertIsNotNone(held.__cause__.__traceback__)
        # The error (and torch's, with its traceback) is still alive; the failed step's locals are not.
        self.assertIsNone(refs[0]())

    def test_other_errors_pass_through(self):
        class _Boom:
            def __call__(self, **kwargs):
                raise RuntimeError("mat1 and mat2 must have the same dtype")

        backend = hd.HuggingFaceDiffusersVisionBackend(
            config=hd.HuggingFaceDiffusersBackendConfig(model_id="x/y", device="cpu")
        )
        with self.assertRaises(RuntimeError):
            backend._pipe_call(_Boom(), {"prompt": "fox"})


if __name__ == "__main__":
    unittest.main()
