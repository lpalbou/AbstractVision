"""The AbstractCore plugin's keyword handling against 0.3.36, call by call.

Each call of the table runs through `_AbstractVisionCapability` with a backend that records the
request, in its own process. A keyword has the outcome it had in 0.3.36 (delivered with the same
value, as a typed field or in `extra`), except where 0.3.36 raised TypeError (now a typed refusal)
and where a call names a soundtrack option that its request type cannot take (refused since
0.3.37). The 0.3.36 outcomes are recorded in CALLS, produced once from the 0.3.36 release
(eea267d) by:

    mkdir /tmp/v0336 && git archive eea267d src | tar -x -C /tmp/v0336
    cd tests && python -c "import test_plugin_keyword_compatibility as t, pathlib; print(t._run_probe(pathlib.Path('/tmp/v0336/src')))"
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROBE = r'''
import base64, dataclasses, json, sys
from unittest.mock import patch

import abstractvision.backends.mflux as mflux_backend
from abstractvision import types as av_types
from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/6X9+QAAAABJRU5ErkJggg=="
)
seen = {}


class _Backend:
    holds = False

    def __init__(self, *, config):
        pass

    def has_loaded_model(self):
        return self.holds

    def get_capabilities(self):
        return av_types.VisionBackendCapabilities(supports_control_image=True, memory_modes=("default",))

    def _record(self, request, media_type):
        seen["request"] = request
        return av_types.GeneratedAsset(media_type=media_type, data=b"x", mime_type="image/png", metadata={})

    def generate_image(self, request):
        self.holds = True
        return self._record(request, "image")

    def edit_image(self, request):
        self.holds = True
        return self._record(request, "image")

    def upscale_image(self, request):
        self.holds = True
        return self._record(request, "image")

    def generate_video(self, request):
        self.holds = True
        return self._record(request, "video")

    def image_to_video(self, request):
        self.holds = True
        return self._record(request, "video")


class _Owner:
    config = {"vision_backend": "mflux", "vision_mflux_model": "AbstractFramework/qwen-image-8bit"}


POSITIONAL = {"t2i": ("a cat",), "i2i": ("edit", PNG), "upscale_image": (PNG,), "t2v": ("a fox",), "i2v": (PNG,)}
results = []
with patch.object(mflux_backend, "MFluxVisionBackend", _Backend), patch.dict("os.environ", {}, clear=True):
    cap = _AbstractVisionCapability(_Owner())
    for method, kwargs in json.loads(sys.stdin.read()):
        seen.clear()
        try:
            getattr(cap, method)(*POSITIONAL[method], **kwargs)
        except Exception as e:
            results.append({"outcome": type(e).__name__, "message": str(e)})
            continue
        request = seen["request"]
        fields = {field.name for field in dataclasses.fields(request)}
        delivered = {}
        for key in kwargs:
            if key in fields and getattr(request, key) is not None:
                delivered[key] = ["typed", repr(getattr(request, key))]
            elif key in (request.extra or {}):
                delivered[key] = ["extra", repr(request.extra[key])]
            else:
                delivered[key] = ["dropped", None]
        results.append({"outcome": "ok", "delivered": delivered})
print(json.dumps(results))
'''

# (method, keywords, its 0.3.36 outcome: the repr of the value delivered, or the exception raised,
#  what 0.3.37 does differently on purpose, or None)
CALLS = (
    ("t2v", {"vae_tiling": False}, "False", None),
    ("i2v", {"vae_tiling": False}, "False", None),
    ("t2i", {"fps": 24}, "24", None),
    ("t2i", {"resolution": 1024}, "1024", None),
    ("t2i", {"num_frames": 9}, "9", None),
    ("t2i", {"flow_shift": 3.0}, "3.0", None),
    ("t2i", {"scale": 2}, "2", None),
    ("t2i", {"vae_tiling": True}, "True", None),
    ("t2i", {"control_strength": 0.5}, "0.5", None),
    ("t2i", {"quantize": 8}, "8", None),
    ("t2i", {"memory_mode": "default"}, "'default'", None),
    ("t2v", {"control_strength": 0.5}, "0.5", None),
    ("t2v", {"resolution": "2x"}, "'2x'", None),
    ("t2v", {"softness": 0.3}, "0.3", None),
    ("t2v", {"guidance_2": 3.0}, "3.0", None),
    ("i2v", {"resolution": 512}, "512", None),
    ("i2v", {"guidance_2": 3.5}, "3.5", None),
    ("i2i", {"fps": 24}, "24", None),
    ("i2i", {"num_frames": 9}, "9", None),
    ("i2i", {"width": 512}, "512", None),
    ("upscale_image", {"width": 512}, "512", None),
    ("upscale_image", {"steps": 4}, "4", None),
    ("upscale_image", {"guidance_scale": 2.0}, "2.0", None),
    ("upscale_image", {"negative_prompt": "blur"}, "'blur'", None),
    ("upscale_image", {"guidance_2": 3.0}, "3.0", None),
    ("upscale_image", {"fps": 8}, "8", None),
    ("t2v", {"soundscape": "rain"}, "'rain'", None),
    ("t2i", {"guidance_2": 3.0}, "TypeError", "0.3.36 raised TypeError"),
    ("i2i", {"guidance_2": 3.0}, "TypeError", "0.3.36 raised TypeError"),
    ("t2i", {"soundscape": "rain"}, "'rain'", "soundtrack option on an image call"),
    ("t2i", {"generate_audio": False}, "False", "soundtrack option on an image call"),
    ("i2i", {"audio_shift": 3.0}, "3.0", "soundtrack option on an image call"),
    ("upscale_image", {"music": "piano"}, "'piano'", "soundtrack option on an image call"),
)


def _run_probe(src_dir: Path):
    payload = json.dumps([[method, kwargs] for method, kwargs, _before, _deviation in CALLS])
    env = dict(os.environ, PYTHONPATH=str(src_dir))
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "probe.py"
        script.write_text(_PROBE, encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(script)], input=payload, capture_output=True, text=True, env=env, cwd=td, timeout=300
        )
    if done.returncode != 0:
        raise AssertionError(f"probe failed on {src_dir}:\n{done.stderr[-4000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


class TestPluginKeywordCompatibility(unittest.TestCase):
    def test_keywords_keep_their_0_3_36_outcome(self):
        new = _run_probe(REPO_ROOT / "src")
        for (method, kwargs, before, deviation), after in zip(CALLS, new):
            with self.subTest(method=method, **{key: repr(value) for key, value in kwargs.items()}):
                if deviation is None:
                    self.assertEqual(after["outcome"], "ok", after)
                    # The same value reaches the backend, as a typed field or in `extra`.
                    (value,) = [value for _where, value in after["delivered"].values()]
                    self.assertEqual(value, before)
                else:
                    self.assertEqual(before == "TypeError", deviation == "0.3.36 raised TypeError")
                    self.assertEqual(after["outcome"], "CapabilityNotSupportedError")
                    self.assertIn(next(iter(kwargs)), after["message"])
                    self.assertNotIn("it is options", after["message"])


if __name__ == "__main__":
    unittest.main()
