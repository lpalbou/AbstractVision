"""An 8-bit GGUF preset downloads, and runs, its Q8_0 file only.

Found preparing the NVIDIA (Linux, 16 GB) catalog run, 2026-11-10: the registry-declared GGUF
presets downloaded `*.gguf`, every quantization the repo ships: unsloth/Qwen-Image-2512-GGUF is 15
files and 258 GB (the runtime reads the 22 GB Q8_0 one), unsloth/Z-Image-GGUF 92 GB,
city96/FLUX.2-dev-gguf 347 GB. And the stable-diffusion.cpp selection takes the first file its
patterns match, so for Z-Image and FLUX.2-dev it ran the BF16 file, not the 8-bit one. Going back to
`_default_allow_patterns` in `_all_presets` turns this RED. No network: the file names are the
repos' real listings.
"""

from __future__ import annotations

import fnmatch
import tempfile
import unittest
from pathlib import Path

from abstractvision import model_downloads as md

_QUANTS = ("BF16", "F16", "Q2_K", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_1", "Q4_K_M", "Q4_K_S", "Q5_0", "Q5_1",
           "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0")


def _preset(repo_id):
    return next(p for p in md._all_presets() if p.repo_id == repo_id and p.target == "gguf")


def _matched(preset, files):
    return sorted(f for f in files if any(fnmatch.fnmatch(f, p) for p in preset.allow_patterns) and f.endswith(".gguf"))


class GgufPresetsDownloadOneQuantTests(unittest.TestCase):
    def test_bundle_repo_downloads_only_the_bundle_file(self):
        files = [f"qwen-image-2512-{q}.gguf" for q in _QUANTS]
        self.assertEqual(_matched(_preset("unsloth/Qwen-Image-2512-GGUF"), files), ["qwen-image-2512-Q8_0.gguf"])

    def test_repo_without_bundle_matches_and_selects_only_its_q8_file(self):
        for repo, stem in (("unsloth/Z-Image-GGUF", "z-image"), ("city96/FLUX.2-dev-gguf", "flux2-dev")):
            with self.subTest(repo=repo):
                preset = _preset(repo)
                files = [f"{stem}-{q}.gguf" for q in _QUANTS]
                self.assertEqual(_matched(preset, files), [f"{stem}-Q8_0.gguf"])
                with tempfile.TemporaryDirectory() as td:
                    for name in files:
                        (Path(td) / name).write_bytes(b"GGUF")
                    picked = md._find_snapshot_file(Path(td), md._gguf_patterns_for_preset(preset))
                self.assertEqual(picked.name, f"{stem}-Q8_0.gguf")

    def test_lowercase_q8_names_still_match(self):
        files = ["sd_turbo-f16-q8_0.gguf", "sd_turbo-f16.gguf"]
        self.assertEqual(_matched(_preset("Green-Sky/SD-Turbo-GGUF"), files), ["sd_turbo-f16-q8_0.gguf"])

    def test_download_takes_root_gguf_files_only(self):
        # Green-Sky/SD-Turbo-GGUF's real listing: the hub's fnmatch lets `*q8_0.gguf` match `old/...`.
        from unittest.mock import patch

        try:
            from huggingface_hub.utils import filter_repo_objects
        except ImportError:  # pragma: no cover - huggingface_hub is a test-time dependency here
            self.skipTest("huggingface_hub not installed")
        listing = ["README.md", "old/sd_turbo-q8_0.gguf", "sd_turbo-f16-q8_0.gguf", "sd_turbo-f16.gguf"]
        captured = {}

        def fake_snapshot_download(**kw):
            captured.update(kw)
            raise RuntimeError("stop after the request")

        with patch.object(md, "ensure_hf_repo_snapshot", return_value=None), \
             patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download):
            with self.assertRaises(RuntimeError):
                md.download_model_preset(_preset("Green-Sky/SD-Turbo-GGUF"))
        got = list(filter_repo_objects(listing, allow_patterns=captured["allow_patterns"],
                                       ignore_patterns=captured["ignore_patterns"]))
        self.assertEqual(sorted(got), ["README.md", "sd_turbo-f16-q8_0.gguf"])


if __name__ == "__main__":
    unittest.main()
