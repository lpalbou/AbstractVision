"""On a CUDA host every auto catalog pick is an artifact an AbstractVision backend loads.

Measured on a Quadro RTX 5000 host (2026-11-10): the CUDA auto target order listed fp8 first, so
`abstractvision download flux2-klein-4b` fetched black-forest-labs/FLUX.2-klein-4b-fp8, a
components-only side artifact that neither the Diffusers nor the stable-diffusion.cpp backend
loads. Moving fp8 last is not enough (downloads are 8-bit by default and FP8 is the only 8-bit
artifact of flux1-dev and flux2-klein-9b-kv), so it is out of the auto order. Putting fp8 back
into `local_catalog_targets()` turns this RED. No network.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from abstractvision import model_downloads as md

# Engines a backend loads (Diffusers pipelines, stable-diffusion.cpp GGUF, MLX-Gen, Transformers snapshots).
_LOADABLE = {"diffusers", "stable-diffusion.cpp", "mlx-gen", "mflux", "transformers"}


class CudaAutoPickIsLoadableTests(unittest.TestCase):
    def test_every_auto_pick_is_loadable(self):
        keys = sorted({p.key for p in md._all_presets() if p.target == "fp8"})
        self.assertIn("flux1-dev", keys)  # the FP8 side artifacts exist in the catalog
        picked = 0
        with patch.object(md, "local_model_profile", return_value="cuda"):
            for key in keys:
                for require_8bit in (True, False):
                    with self.subTest(key=key, require_8bit=require_8bit):
                        try:
                            preset = md.find_model_preset(key, target="auto", engine=None, require_8bit=require_8bit)
                        except ValueError:
                            continue  # no loadable 8-bit artifact: the CLI says so instead of fetching FP8
                        picked += 1
                        self.assertNotEqual(preset.target, "fp8", preset.repo_id)
                        self.assertIn(preset.engine, _LOADABLE, preset.repo_id)
        self.assertGreater(picked, len(keys))  # the check really ran on picks

    def test_fp8_stays_reachable_by_explicit_target(self):
        with patch.object(md, "local_model_profile", return_value="cuda"):
            self.assertNotIn("fp8", md.catalog_target_scope(target="auto", engine=None, include_all_targets=False))
            self.assertIn("fp8", md.catalog_target_scope(target="auto", engine=None, include_all_targets=True))
            explicit = md.find_model_preset("flux1-dev", target="fp8", engine=None, require_8bit=True)
        self.assertEqual((explicit.target, explicit.repo_id), ("fp8", "black-forest-labs/FLUX.1-dev-FP8"))


if __name__ == "__main__":
    unittest.main()
