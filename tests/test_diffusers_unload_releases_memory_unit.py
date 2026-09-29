"""A Diffusers unload must actually release the pipeline (framework backlog 0991).

Measured on CUDA (FLUX.2 klein 4B, model CPU offload, 2026-09-29): the process
held 18.8 GB after `unload()` reported success, because the collect inside
`_unload_locked` ran while the loop variable (and the bound `unfuse_lora` / `unload_lora_weights` methods) still pointed at the
last pipeline, so its reference cycles (accelerate offload hooks) survived; the
freed heap was also never handed back to the OS. Releasing adapters inline in
`_unload_locked` again, or dropping its `_return_freed_host_memory()` call,
turns this RED.
No torch, no model.
"""
from __future__ import annotations

import gc
import unittest
import weakref
from unittest.mock import patch

from abstractvision.backends import huggingface_diffusers as hd


class _CyclicPipeline:
    """A pipeline whose only remaining references form a cycle, like the
    module <-> offload-hook cycles accelerate installs."""

    def __init__(self) -> None:
        self.hook = {"module": self}
        self.lora_calls = []

    # Real Diffusers pipelines expose both; the unload calls them, and a bound
    # method left in the unload's frame is a reference to the pipeline.
    def unfuse_lora(self) -> None:
        self.lora_calls.append("unfuse")

    def unload_lora_weights(self) -> None:
        self.lora_calls.append("unload")


def _backend_with(pipes):
    backend = hd.HuggingFaceDiffusersVisionBackend.__new__(hd.HuggingFaceDiffusersVisionBackend)
    backend._pipelines = dict(pipes)
    backend._call_params = {}
    backend._warmed_pipeline_ids = set()
    backend._fused_lora_signature = {}
    backend._rapid_transformer_key = None
    backend._rapid_transformer = None
    return backend


class TestDiffusersUnloadReleasesMemory(unittest.TestCase):
    def test_unload_frees_the_last_pipeline_without_a_later_gc(self) -> None:
        pipes = {"t2i": _CyclicPipeline(), "i2i": _CyclicPipeline()}
        refs = [weakref.ref(p) for p in pipes.values()]
        backend = _backend_with(pipes)
        del pipes

        gc.disable()  # only the unload's own collect may run
        try:
            backend._unload_locked()
            alive = [r() is not None for r in refs]
        finally:
            gc.enable()

        self.assertEqual(alive, [False, False], "a pipeline outlived its unload")

    def test_unload_returns_freed_heap_to_the_os(self) -> None:
        backend = _backend_with({"t2i": _CyclicPipeline()})
        with patch.object(hd, "_return_freed_host_memory") as trim:
            backend._unload_locked()
        trim.assert_called_once_with()

    def test_heap_return_calls_malloc_trim_on_linux_only(self) -> None:
        calls = []

        class _Libc:
            def malloc_trim(self, pad):
                calls.append(pad)
                return 1

        with patch.object(hd.sys, "platform", "linux"), patch("ctypes.CDLL", return_value=_Libc()):
            hd._return_freed_host_memory()
        self.assertEqual(calls, [0])

        with patch.object(hd.sys, "platform", "darwin"), patch("ctypes.CDLL", return_value=_Libc()):
            hd._return_freed_host_memory()
        self.assertEqual(calls, [0])


if __name__ == "__main__":
    unittest.main()
