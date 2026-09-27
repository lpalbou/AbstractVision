"""An mflux unload must return MLX's allocator cache to the OS.

Dropping the weights alone parks them in MLX cache memory, so a process with
zero loaded models can still hold tens of GB of Metal memory. Removing
`mx.clear_cache()` from `MFluxVisionBackend._unload_impl` turns this RED.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch


class TestMFluxUnloadClearsMlxCache(unittest.TestCase):
    def test_unload_returns_mlx_cache_to_the_os(self) -> None:
        calls = {"clear": 0}

        def _clear_cache() -> None:
            calls["clear"] += 1

        core = types.ModuleType("mlx.core")
        core.clear_cache = _clear_cache  # type: ignore[attr-defined]
        pkg = types.ModuleType("mlx")
        pkg.core = core  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"mlx": pkg, "mlx.core": core}):
            from abstractvision.backends.mflux import MFluxVisionBackend

            backend = MFluxVisionBackend.__new__(MFluxVisionBackend)
            backend._model = object()
            backend._model_key = "k"
            backend._warmed_model_key = "k"
            backend._resolved_model_path = "p"
            backend._resolved_base_model = "b"
            backend._resolved_quantization_bits = 8
            backend._runtime_queue = None
            backend._runtime_thread = None
            backend._runtime_thread_id = None

            backend.unload()

        self.assertIsNone(backend._model)
        self.assertEqual(calls["clear"], 1, "unload must clear MLX's allocator cache")


if __name__ == "__main__":
    unittest.main()
