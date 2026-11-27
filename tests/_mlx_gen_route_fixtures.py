"""Stand-in MLX-Gen engine classes and the test case the `test_mlx_gen_route_*.py` modules share.

The stand-ins record what they are built with and called with (`EVENTS`). The planner and its rows
are mlx-gen's own (the pinned release); `needs_planner` skips a test where mlx-gen is not installed.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch


def _planner_installed() -> bool:
    try:
        import mflux.task_inference  # noqa: F401
    except Exception:
        return False
    return True


needs_planner = unittest.skipUnless(_planner_installed(), "mlx-gen's planner is not installed")


def png(width: int, height: int, color=(40, 90, 160)) -> bytes:
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


SOURCE = png(64, 48)
CLIP = b"\x00\x00\x00\x18ftypmp42 a synthetic clip"
FLUX2 = "AbstractFramework/flux.2-klein-9b-8bit"
FLUX2_BASE = "AbstractFramework/flux.2-klein-base-4b-8bit"
QWEN = "AbstractFramework/qwen-image-8bit"
QWEN_2512 = "AbstractFramework/qwen-image-2512-8bit"
ERNIE = "AbstractFramework/ernie-image-turbo-8bit"
QWEN_EDIT = "AbstractFramework/qwen-image-edit-2511-8bit"
QWEN_EDIT_2509 = "AbstractFramework/qwen-image-edit-2509-8bit"
QWEN_EDIT_PLAIN = "AbstractFramework/qwen-image-edit-8bit"
FIBO = "AbstractFramework/fibo-8bit"
FIBO_LITE = "briaai/Fibo-lite"
FIBO_EDIT = "briaai/Fibo-Edit"
WAN_T2V = "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"
WAN_I2V = "AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit"
WAN_TI2V = "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"
WAN_VACE = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
Z_IMAGE_TURBO = "AbstractFramework/z-image-turbo-8bit"
Z_IMAGE = "AbstractFramework/z-image-8bit"
MODELS = (
    FLUX2, FLUX2_BASE, QWEN, QWEN_2512, ERNIE, QWEN_EDIT, QWEN_EDIT_2509, QWEN_EDIT_PLAIN, FIBO, FIBO_LITE, FIBO_EDIT,
    WAN_T2V, WAN_I2V, WAN_TI2V, WAN_VACE, Z_IMAGE_TURBO, Z_IMAGE,
)
ADAPTER = "/Users/someone/adapters/style.safetensors"

# What the stand-ins saw: ("init", class, kwargs) and ("call", class, kwargs, facts about the files).
EVENTS = []


class _Callbacks:
    def __init__(self):
        self.subscribers = []
        self.registered = []

    def subscribe_progress(self, callback, *, task=None):
        self.subscribers.append(callback)
        return lambda: self.subscribers.remove(callback)

    def register(self, callback):
        self.registered.append(callback)
        EVENTS.append(("register", type(callback).__name__, getattr(callback, "keep_transformer", None)))


class _Generated:
    """A generated image as mlx-gen returns it: the picture at the canvas size it was asked for, and
    metadata naming the model and the files it was given (an outpaint session repoints and extends them)."""

    def __init__(self, init, kwargs):
        from PIL import Image

        canvas = kwargs.get("canvas")
        if canvas is not None:
            size = (canvas.target_width, canvas.target_height)
        elif kwargs.get("width") and kwargs.get("height"):
            size = (kwargs["width"], kwargs["height"])
        else:
            size = (4, 4)
        self.image = Image.new("RGB", size, (90, 90, 90))
        self.image_path = str(kwargs.get("image_path")) if kwargs.get("image_path") else None
        self.image_paths = kwargs.get("image_paths")
        self.extra_metadata = None
        self._metadata = {
            "model": init.get("model_path"),
            "lora_paths": init.get("lora_paths"),
            "image_strength": kwargs.get("image_strength"),
            "prompt": kwargs.get("prompt"),
        }

    def _get_metadata(self):
        image_paths = [str(path) for path in self.image_paths] if self.image_paths else None
        return {**self._metadata, "image_path": self.image_path, "image_paths": image_paths, **(self.extra_metadata or {})}


def _file_facts(value):
    path = Path(value)
    return {"exists": path.is_file(), "dir": str(path.parent), "bytes": path.read_bytes() if path.is_file() else None}


def _canvas_facts(canvas):
    # An outpaint canvas (OutpaintCanvas): its size and its file's digest.
    import hashlib

    path = Path(canvas.canvas_path)
    return {
        "size": (canvas.target_width, canvas.target_height),
        "dir": str(path.parent),
        "digest": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def stand_in(name):
    class StandIn:
        fail_with = None  # an exception generate_image raises
        during_call = None  # called inside generate_image, before the progress event
        emit_steps = False  # report each step (start, denoise..., complete) instead of one denoise event

        def __init__(self, **kwargs):
            self.init = dict(kwargs)
            self.callbacks = _Callbacks()
            EVENTS.append(("init", name, dict(kwargs)))

        def generate_image(self, **kwargs):
            files = {
                key: _file_facts(kwargs[key])
                for key in ("image_path", "mask_path", "controlnet_image_path")
                if kwargs.get(key) is not None
            }
            if kwargs.get("image_paths"):
                files["image_paths"] = [_file_facts(path) for path in kwargs["image_paths"]]
            if kwargs.get("canvas") is not None:
                files["canvas"] = _canvas_facts(kwargs["canvas"])
            EVENTS.append(("call", name, dict(kwargs), files))
            if StandIn.during_call is not None:
                StandIn.during_call()
            steps = int(kwargs.get("num_inference_steps") or 2)
            if StandIn.emit_steps:
                events = [("start", 0)] + [("denoise", step) for step in range(1, steps + 1)] + [("complete", steps)]
            else:
                events = [("denoise", 1)]
            for phase, step in events:
                for callback in list(self.callbacks.subscribers):
                    callback({"phase": phase, "step": step, "total_steps": steps if StandIn.emit_steps else 2})
            if StandIn.fail_with is not None:
                raise StandIn.fail_with
            return _Generated(self.init, kwargs)

    StandIn.__name__ = name
    return StandIn


class _WanVae:
    # The decode entry point the backend's tiled decode replaces (wan_vae_tiling).
    patch_size = 2

    def iter_decode_normalized_latent_slices(self, latents, *, clear_cache_each_slice=False, tile_spatial=False):
        return iter(())


class _GeneratedVideo:
    """A generated clip as mlx-gen returns it: save() writes the MP4 and its metadata sidecar."""

    def __init__(self, init, kwargs):
        frames = int(kwargs.get("num_frames") or 81)
        self.metadata = {
            "model": init.get("model_path"),
            "video_path": kwargs.get("video_path"),
            "video_mask_path": kwargs.get("video_mask_path"),
            "reference_image_paths": kwargs.get("reference_image_paths"),
            "frames": (frames - 1) // 4 * 4 + 1,
            "fps": kwargs.get("fps"),
            "prompt": kwargs.get("prompt"),
        }

    def save(self, path=None, export_json_metadata=False, overwrite=True):
        import json

        path = Path(path)
        path.write_bytes(CLIP)
        if export_json_metadata:
            path.with_suffix(".metadata.json").write_text(json.dumps(self.metadata))
        return path


def video_stand_in(name):
    class StandIn:
        fail_with = None
        during_call = None

        def __init__(self, **kwargs):
            self.init = dict(kwargs)
            self.vae = _WanVae()
            self.callbacks = _Callbacks()
            EVENTS.append(("init", name, dict(kwargs)))

        def generate_video(self, **kwargs):
            files = {key: _file_facts(kwargs[key]) for key in ("video_path", "video_mask_path") if kwargs.get(key)}
            if kwargs.get("reference_image_paths"):
                files["reference_image_paths"] = [_file_facts(path) for path in kwargs["reference_image_paths"]]
            files["tiled_decode"] = "iter_decode_normalized_latent_slices" in vars(self.vae)
            EVENTS.append(("call", name, dict(kwargs), files))
            if StandIn.during_call is not None:
                StandIn.during_call()
            progress = kwargs.get("progress_callback")
            if callable(progress):
                progress({"phase": "denoise", "step": 1, "total_steps": 2})
            if StandIn.fail_with is not None:
                raise StandIn.fail_with
            return _GeneratedVideo(self.init, kwargs)

    StandIn.__name__ = name
    return StandIn


class VideoUtil:
    """MLX-Gen's VideoUtil.inspect_video: what a source clip holds (set per test)."""

    frame_count = 121
    duration_seconds = 5.0
    inspected = []

    @classmethod
    def inspect_video(cls, path):
        from types import SimpleNamespace

        cls.inspected.append(str(path))
        return SimpleNamespace(source_frame_count=cls.frame_count, source_duration_seconds=cls.duration_seconds)


class _ModelConfig:
    @staticmethod
    def from_name(model_name, base_model=None):
        return f"config:{model_name}"


class MemorySaver:
    def __init__(self, model, keep_transformer=True, cache_limit_bytes=None, args=None):
        self.keep_transformer = keep_transformer


Flux2Klein = stand_in("Flux2Klein")
Flux2KleinEdit = stand_in("Flux2KleinEdit")
Flux2KleinInpaint = stand_in("Flux2KleinInpaint")
Flux2KleinOutpaint = stand_in("Flux2KleinOutpaint")
QwenImage = stand_in("QwenImage")
QwenImageEdit = stand_in("QwenImageEdit")
QwenImageControlNet = stand_in("QwenImageControlNet")
ErnieImageTurbo = stand_in("ErnieImageTurbo")
FIBO_CLS = stand_in("FIBO")
FIBOEdit = stand_in("FIBOEdit")
ZImage = stand_in("ZImage")
Wan2_2_TI2V = video_stand_in("Wan2_2_TI2V")
WanVace = video_stand_in("WanVace")
STAND_INS = (Wan2_2_TI2V, WanVace, QwenImageControlNet, Flux2Klein, Flux2KleinEdit, Flux2KleinInpaint, Flux2KleinOutpaint, QwenImage, QwenImageEdit, ErnieImageTurbo, FIBO_CLS, FIBOEdit, ZImage)
# The backend's lazy engine imports, each with what the stand-in version returns.
ENGINE_IMPORTS = {
    "_lazy_import_mflux": (_ModelConfig, FileNotFoundError, Flux2Klein, Flux2KleinEdit, ZImage, ZImage),
    "_lazy_import_mflux_qwen": (QwenImage, QwenImageEdit),
    "_lazy_import_mflux_qwen_controlnet": QwenImageControlNet,
    "_lazy_import_mflux_flux2_inpaint": Flux2KleinInpaint,
    "_lazy_import_mflux_flux2_outpaint": Flux2KleinOutpaint,
    "_lazy_import_mflux_ernie": ErnieImageTurbo,
    "_lazy_import_mflux_fibo": (FIBO_CLS, FIBOEdit),
    "_lazy_import_mlx_gen_memory_saver": MemorySaver,
    "_lazy_import_mflux_wan": Wan2_2_TI2V,
    "_lazy_import_mflux_wan_vace": WanVace,
    "_lazy_import_mlx_gen_video_util": VideoUtil,
}


def snapshot(root: Path, repo_id: str) -> Path:
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    snap = repo_dir / "snapshots" / "abc123"
    (snap / "transformer").mkdir(parents=True, exist_ok=True)
    (snap / "transformer" / "0.safetensors").write_bytes(b"x")
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text("abc123", encoding="utf-8")
    return snap


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings(item)


class RouteCase(unittest.TestCase):
    """An HF cache holding MODELS, the stand-in engines, and tempfile's directory observed."""

    def setUp(self):
        EVENTS.clear()
        for cls in STAND_INS:
            cls.fail_with = None
            cls.during_call = None
            cls.emit_steps = False
        VideoUtil.frame_count, VideoUtil.duration_seconds = 121, 5.0
        VideoUtil.inspected.clear()
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        cache = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.snapshots = {model: snapshot(cache, model) for model in MODELS}
        # Where the runs write their temporaries (tempfile's default directory).
        self.tmp_root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch.dict("os.environ", {"HF_HUB_CACHE": str(cache)}, clear=True))
        stack.enter_context(patch.object(tempfile, "tempdir", str(self.tmp_root)))
        for name, value in ENGINE_IMPORTS.items():
            stack.enter_context(patch(f"abstractvision.backends.mflux.{name}", return_value=value))
        stack.enter_context(patch("abstractvision.backends.mflux._set_mlx_cache_limit", return_value=None))
        stack.enter_context(patch("abstractvision.backends.mflux._apply_mlx_gen_default_cache_limit"))

    def backend(self, model):
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        return MFluxVisionBackend(config=MFluxBackendConfig(model=model))

    def folder_backend(self, base_model):
        """A local folder whose name the engine cannot place (so it publishes no rows for it)."""
        from abstractvision.backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        # Inside the test's cache directory, named without any hint of the model.
        cache = self.snapshots[MODELS[0]].parents[2]
        path = cache / f"weights-{len(list(cache.glob('weights-*')))}"
        (path / "transformer").mkdir(parents=True, exist_ok=True)
        (path / "transformer" / "0.safetensors").write_bytes(b"x")
        backend = MFluxVisionBackend(config=MFluxBackendConfig(model=str(path), base_model=base_model))
        self.assertEqual(backend._route_selector_with_rows(), (None, {}))
        return backend

    def edit(self, model, **fields):
        from abstractvision.types import ImageEditRequest

        backend = self.backend(model)
        return backend, backend.edit_image(ImageEditRequest(prompt=fields.pop("prompt", "a red kite"), image=SOURCE, **fields))

    def inits(self):
        return [event[1] for event in EVENTS if event[0] == "init"]

    def call(self):
        calls = [event for event in EVENTS if event[0] == "call"]
        self.assertEqual(len(calls), 1, calls)
        _kind, name, kwargs, files = calls[0]
        return name, kwargs, files

    def listed(self, backend, model_id):
        # The image_to_image listing covers every cached model; this one's entry.
        (model,) = [m for m in backend.list_provider_models(task="image_to_image") if m.id == model_id]
        return model

    def assert_nothing_loaded(self, backend):
        self.assertEqual(self.inits(), [])
        self.assertFalse(backend.has_loaded_model())

    def assert_refused(self, model, error, *, field="unset", supported="unset", **fields):
        from abstractvision.types import ImageEditRequest

        backend = self.backend(model)
        with self.assertRaises(error) as caught:
            backend.edit_image(ImageEditRequest(prompt="a red kite", image=SOURCE, **fields))
        if field != "unset":
            self.assertEqual(caught.exception.field, field)
        if supported != "unset":
            self.assertEqual(caught.exception.supported, supported)
        self.assert_nothing_loaded(backend)
        self.assertEqual(list(self.tmp_root.iterdir()), [])
        return caught.exception

    def assert_metadata_clean(self, asset, run_dir):
        # No string in the metadata names the run's directory or any absolute path.
        for text in strings(asset.metadata):
            self.assertNotIn(run_dir, text)
            self.assertFalse(os.path.isabs(text), text)

    def assert_other_backends_refuse(self, **typed):
        """Diffusers, OpenAI-compatible and stable-diffusion.cpp refuse each typed edit option by name;
        in `extra` the key is not their option and reaches them as 0.3.37 sent it."""
        from abstractvision.backends import huggingface_diffusers as hd
        from abstractvision.backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )
        from abstractvision.backends.stable_diffusion_cpp import (
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
        )
        from abstractvision.errors import CapabilityNotSupportedError
        from abstractvision.types import ImageEditRequest

        backends = {
            "diffusers": hd.HuggingFaceDiffusersVisionBackend(
                config=hd.HuggingFaceDiffusersBackendConfig(model_id="some/model", device="cpu")
            ),
            "openai-compatible": OpenAICompatibleVisionBackend(
                config=OpenAICompatibleBackendConfig(base_url="http://127.0.0.1:9/v1", model_id="image-model")
            ),
            "sdcpp": StableDiffusionCppVisionBackend(
                config=StableDiffusionCppBackendConfig(sd_cli_path="sd-cli", diffusion_model="model.gguf")
            ),
        }
        for kind, backend in backends.items():
            for name, value in typed.items():
                with self.subTest(backend=kind, option=name):
                    with self.assertRaises(CapabilityNotSupportedError) as caught:
                        backend.edit_image(ImageEditRequest(prompt="a red kite", image=SOURCE, **{name: value}))
                    self.assertEqual((caught.exception.field, caught.exception.supported), (name, ()))
                    resolved, _mode = backend.resolve_memory_request(
                        ImageEditRequest(prompt="a red kite", image=SOURCE, extra={name: value, "strength": 0.5}),
                        task="image_to_image",
                    )
                    self.assertEqual(resolved.extra, {name: value, "strength": 0.5})
                    self.assertFalse(getattr(resolved, name))

    def capability(self, backend, model):
        """The AbstractCore plugin's capability, its requests bound to `backend`."""
        from abstractvision.integrations.abstractcore_plugin import _AbstractVisionCapability

        class _Owner:
            config = {}

        cap = _AbstractVisionCapability(_Owner())
        binding = {
            "backend": backend,
            "backend_key": ("mlx-gen", model),
            "backend_kind": "mlx-gen",
            "provider": "mlx-gen",
            "model": model,
            "load_id": f"mlx-gen/{model}",
            "local_control": True,
        }
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(cap, "_resolve_backend_binding", return_value=binding))
        return cap
