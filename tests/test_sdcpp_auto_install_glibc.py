"""The `sd-cli` auto-install picks a build this Linux host can start.

Found on Ubuntu 24.04 (glibc 2.39) with an NVIDIA GPU, 2026-11-10: since stable-diffusion.cpp
master-1044 (2026-11-03) the newest release ships Linux builds for Ubuntu 26.04 only (glibc 2.43),
plus an Ubuntu 24.04 ROCm build. The installer took the newest release's Ubuntu 26.04 Vulkan build
and every GGUF generation failed with "version `GLIBC_2.43' not found". It now walks recent releases
newest first, skips builds for a newer glibc than the host's and ROCm builds, and takes the newest
release that has one (master-1043's Ubuntu 24.04 Vulkan build there). Reverting to the newest
release only, or dropping the glibc check, turns this RED. No network: the release listing is the
real one, trimmed.
"""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from abstractvision.backends import stable_diffusion_cpp as sd

_BASE = "https://github.com/leejet/stable-diffusion.cpp/releases/download"


def _release(tag, *names):
    return {"tag_name": tag, "assets": [{"name": n, "browser_download_url": f"{_BASE}/{tag}/{n}"} for n in names]}


_RELEASES = [
    _release(
        "master-1079-a3c49b1",
        "sd-master-a3c49b1-bin-Darwin-macOS-26.7.2-arm64.zip",
        "sd-master-a3c49b1-bin-Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip",
        "sd-master-a3c49b1-bin-Linux-Ubuntu-26.04-x86_64-vulkan.zip",
        "sd-master-a3c49b1-bin-Linux-Ubuntu-26.04-x86_64.zip",
        "sd-master-a3c49b1-bin-win-cuda12-x64.zip",
    ),
    _release(
        "master-1043-3be2378",
        "sd-master-3be2378-bin-Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip",
        "sd-master-3be2378-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
        "sd-master-3be2378-bin-Linux-Ubuntu-24.04-x86_64.zip",
    ),
]


def _pick(**kw):
    args = dict(plat="linux", machine="x86_64", host_glibc=(2, 39), has_nvidia=True)
    args.update(kw)
    return sd._pick_sd_cli_asset_url(_RELEASES, **args)


class SdCliAutoInstallGlibcTests(unittest.TestCase):
    def test_ubuntu_24_04_with_nvidia_gets_the_newest_24_04_vulkan_build(self):
        self.assertTrue(_pick().endswith("master-1043-3be2378/sd-master-3be2378-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"))

    def test_ubuntu_24_04_without_gpu_gets_the_plain_build(self):
        self.assertTrue(_pick(has_nvidia=False).endswith("sd-master-3be2378-bin-Linux-Ubuntu-24.04-x86_64.zip"))

    def test_ubuntu_26_04_keeps_the_newest_release(self):
        self.assertTrue(_pick(host_glibc=(2, 43)).endswith("sd-master-a3c49b1-bin-Linux-Ubuntu-26.04-x86_64-vulkan.zip"))

    def test_too_old_host_gets_nothing(self):
        self.assertIsNone(_pick(host_glibc=(2, 31)))

    def test_macos_unchanged(self):
        self.assertTrue(_pick(plat="darwin", machine="arm64").endswith("bin-Darwin-macOS-26.7.2-arm64.zip"))

    def test_installer_downloads_the_compatible_build(self):
        requested = []

        def fake_urlopen(req, timeout=None):
            url = getattr(req, "full_url", req)
            requested.append(url)
            if "api.github.com" in url:
                return io.BytesIO(json.dumps(_RELEASES).encode())
            raise OSError("stop after selection")

        import tempfile

        with tempfile.TemporaryDirectory() as td, \
             patch.object(sd.urllib.request, "urlopen", side_effect=fake_urlopen), \
             patch.object(sd.sys, "platform", "linux"), \
             patch.object(sd._platform, "machine", return_value="x86_64"), \
             patch.object(sd._platform, "libc_ver", return_value=("glibc", "2.39")), \
             patch.object(sd.shutil, "which", return_value="/usr/bin/nvidia-smi"):
            self.assertIsNone(sd._try_auto_install_sd_cli(__import__("pathlib").Path(td)))
        self.assertEqual(len(requested), 2)
        self.assertTrue(requested[1].endswith("sd-master-3be2378-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"))


if __name__ == "__main__":
    unittest.main()
