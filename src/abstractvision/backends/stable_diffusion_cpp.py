from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..errors import CapabilityNotSupportedError, OptionalDependencyMissingError
from ..types import (
    GeneratedAsset,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageToVideoRequest,
    MultiAngleRequest,
    VideoGenerationRequest,
    VisionBackendCapabilities,
)
from .base_backend import VisionBackend


def _sniff_mime_type(data: bytes) -> str:
    b = bytes(data or b"")
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if b.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "application/octet-stream"


def _sniff_ext(data: bytes) -> str:
    mime = _sniff_mime_type(data)
    if mime == "image/png":
        return ".png"
    if mime == "image/jpeg":
        return ".jpg"
    return ".bin"


def _require_sd_cli(path: str) -> str:
    p = str(path or "").strip()
    if not p:
        raise OptionalDependencyMissingError(
            "stable-diffusion.cpp executable is not configured. "
            "Set sd_cli_path or install `sd-cli` from https://github.com/leejet/stable-diffusion.cpp/releases "
            "(or install `stable-diffusion-cpp-python` to use pip-installable python bindings). "
            "If you intended to run a standard Diffusers model (e.g. 'runwayml/stable-diffusion-v1-5'), use the "
            "Diffusers backend instead."
        )

    # If the user passed a path-like string, validate it exists; otherwise rely on PATH lookup.
    looks_like_path = os.sep in p or (os.altsep and os.altsep in p) or p.startswith(".")
    if looks_like_path:
        if not Path(p).expanduser().exists():
            raise OptionalDependencyMissingError(
                f"stable-diffusion.cpp executable not found at: {p!r}. "
                "Install from https://github.com/leejet/stable-diffusion.cpp/releases or install `stable-diffusion-cpp-python`, "
                "or update sd_cli_path. "
                "If you intended to run a standard Diffusers model (e.g. 'runwayml/stable-diffusion-v1-5'), use the "
                "Diffusers backend instead."
            )
        return p

    resolved = shutil.which(p)
    if not resolved:
        raise OptionalDependencyMissingError(
            f"stable-diffusion.cpp executable not found in PATH: {p!r}. "
            "Install from https://github.com/leejet/stable-diffusion.cpp/releases or install `stable-diffusion-cpp-python`, "
            "or set sd_cli_path. "
            "If you intended to run a standard Diffusers model (e.g. 'runwayml/stable-diffusion-v1-5'), use the "
            "Diffusers backend instead."
        )
    return resolved


def _flatten(xs: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for x in xs:
        if x is None:
            continue
        if isinstance(x, (list, tuple)):
            out.extend(_flatten(x))
            continue
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def _extra_to_cli_args(extra: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for k, v in (extra or {}).items():
        if k is None:
            continue
        key = str(k).strip()
        if not key:
            continue
        if key.startswith("-"):
            # Best-effort: allow advanced users to pass raw flags like "--diffusion-fa".
            flag = key
        else:
            flag = "--" + key.replace("_", "-")
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                args.append(flag)
            continue
        args.extend([flag, str(v)])
    return args


def _parse_sdcpp_extra_args(extra_args: Sequence[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse CLI-style tokens (from config.extra_args) into python-binding kwargs.

    We intentionally only support a small, stable subset of sd-cli flags that map cleanly to
    `stable-diffusion-cpp-python` parameters.
    """

    tokens = [str(t) for t in _flatten(extra_args)]
    flags: Dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            i += 1
            continue
        key = t[2:].strip().replace("-", "_")
        if not key:
            i += 1
            continue

        # bool flag by default; if a value follows and doesn't look like a flag, treat as value.
        value: Any = True
        if i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt and not nxt.startswith("--"):
                value = nxt
                i += 2
            else:
                i += 1
        else:
            i += 1
        flags[key] = value

    init_kwargs: Dict[str, Any] = {}
    default_generate_kwargs: Dict[str, Any] = {}

    def _as_int(v: Any, *, flag: str) -> int:
        try:
            return int(v)
        except Exception as e:
            raise ValueError(f"Invalid value for {flag!r}: expected int, got {v!r}") from e

    def _as_float(v: Any, *, flag: str) -> float:
        try:
            return float(v)
        except Exception as e:
            raise ValueError(f"Invalid value for {flag!r}: expected float, got {v!r}") from e

    for k, v in flags.items():
        if k == "offload_to_cpu" and bool(v):
            init_kwargs["offload_params_to_cpu"] = True
        elif k == "diffusion_fa" and bool(v):
            init_kwargs["diffusion_flash_attn"] = True
        elif k == "flow_shift":
            init_kwargs["flow_shift"] = _as_float(v, flag="--flow-shift")
        elif k == "sampling_method":
            default_generate_kwargs["sample_method"] = str(v)
        elif k == "steps":
            default_generate_kwargs["sample_steps"] = _as_int(v, flag="--steps")
        elif k == "cfg_scale":
            default_generate_kwargs["cfg_scale"] = _as_float(v, flag="--cfg-scale")
        elif k == "seed":
            default_generate_kwargs["seed"] = _as_int(v, flag="--seed")
        elif k == "width":
            default_generate_kwargs["width"] = _as_int(v, flag="--width")
        elif k == "height":
            default_generate_kwargs["height"] = _as_int(v, flag="--height")

    return init_kwargs, default_generate_kwargs


def _extra_to_python_generate_kwargs(extra: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for k, v in (extra or {}).items():
        if k is None or v is None:
            continue
        key = str(k).strip()
        if not key:
            continue
        if key.startswith("-"):
            key = key.lstrip("-")
        key = key.replace("-", "_")

        # Common aliases between sd-cli and stable-diffusion-cpp-python.
        if key == "sampling_method":
            key = "sample_method"
        elif key == "steps":
            key = "sample_steps"
        elif key in {"guidance_scale", "cfg"}:
            key = "cfg_scale"

        out[key] = v

    return out


def _filter_generate_kwargs(model: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys that stable-diffusion-cpp-python does not accept for generate_image()."""

    import inspect

    params = set(inspect.signature(model.generate_image).parameters.keys())
    return {k: v for k, v in kwargs.items() if k in params and v is not None}


def _try_read_gguf_architecture(path: str) -> Optional[str]:
    try:
        import struct

        p = Path(path).expanduser()
        if not p.exists():
            return None
        with p.open("rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            _ver = struct.unpack("<I", f.read(4))[0]
            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            def read_u32() -> int:
                return struct.unpack("<I", f.read(4))[0]

            def read_u64() -> int:
                return struct.unpack("<Q", f.read(8))[0]

            def read_str() -> str:
                n = read_u64()
                return f.read(n).decode("utf-8", errors="replace")

            GGUF_TYPE_STRING = 8
            GGUF_TYPE_ARRAY = 9
            GGUF_TYPE_UINT64 = 10
            GGUF_TYPE_INT64 = 11
            GGUF_TYPE_FLOAT64 = 12

            def skip_value(t: int) -> None:
                # scalar sizes
                if t in (0, 1, 7):
                    f.read(1)
                    return
                if t in (2, 3):
                    f.read(2)
                    return
                if t in (4, 5, 6):
                    f.read(4)
                    return
                if t in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
                    f.read(8)
                    return
                if t == GGUF_TYPE_STRING:
                    n = read_u64()
                    f.read(n)
                    return
                if t == GGUF_TYPE_ARRAY:
                    at = read_u32()
                    n = read_u64()
                    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}.get(at)
                    if size is None:
                        # fallback: give up cleanly (we only need the architecture key).
                        raise ValueError("unsupported gguf array type")
                    f.read(int(n) * int(size))
                    return
                raise ValueError("unsupported gguf value type")

            arch: Optional[str] = None
            for _ in range(int(kv_count)):
                key = read_str()
                t = read_u32()
                if key == "general.architecture" and t == GGUF_TYPE_STRING:
                    arch = read_str()
                else:
                    skip_value(t)
            return arch
    except Exception:
        return None


@dataclass(frozen=True)
class StableDiffusionCppBackendConfig:
    """Config for stable-diffusion.cpp backends.

    This backend is dependency-light by default (stdlib only) and can run via:

    - External executable (`sd-cli`) from stable-diffusion.cpp releases
    - Optional python bindings (pip-installable): `stable-diffusion-cpp-python`

    `StableDiffusionCppVisionBackend` auto-selects:
    - `sd-cli` when available
    - otherwise falls back to python bindings when installed

    External executable:
    https://github.com/leejet/stable-diffusion.cpp

    You can either provide a single `model` (full model), or provide components:
    - diffusion_model (+ optional vae / llm / clip / t5xxl ...)

    For Qwen Image GGUF models, stable-diffusion.cpp expects:
    - diffusion_model (GGUF)
    - vae (safetensors)
    - llm (Qwen2.5-VL text encoder in GGUF)
    """

    sd_cli_path: str = "sd-cli"

    # Single-file full model
    model: Optional[str] = None

    # Component mode
    diffusion_model: Optional[str] = None
    vae: Optional[str] = None
    llm: Optional[str] = None
    llm_vision: Optional[str] = None
    clip_l: Optional[str] = None
    clip_g: Optional[str] = None
    t5xxl: Optional[str] = None

    # Extra args:
    # - CLI mode: forwarded to `sd-cli` (best-effort).
    # - Python mode: a small subset is mapped to python-binding defaults (e.g. --sampling-method, --offload-to-cpu).
    extra_args: Sequence[str] = field(default_factory=tuple)

    # Safety
    timeout_s: float = 60.0 * 60.0  # 1h (image generation can be slow on CPU)
    cwd: Optional[str] = None


class StableDiffusionCppVisionBackend(VisionBackend):
    """Local vision backend that runs stable-diffusion.cpp.

    Supports: text_to_image and image_to_image (including masks when the model supports it).
    """

    def __init__(self, *, config: StableDiffusionCppBackendConfig):
        self._cfg = config
        self._mode: Optional[str] = None  # "cli" | "python"
        self._sd_cli_resolved: Optional[str] = None
        self._py_sd: Any = None
        self._py_model: Any = None
        self._py_init_kwargs: Optional[Dict[str, Any]] = None
        self._py_default_generate_kwargs: Optional[Dict[str, Any]] = None

    def get_capabilities(self) -> VisionBackendCapabilities:
        return VisionBackendCapabilities(
            supported_tasks=["text_to_image", "image_to_image"],
            supports_mask=True,
        )

    def _base_cmd(self) -> List[str]:
        sd_cli = _require_sd_cli(self._cfg.sd_cli_path)
        cmd: List[str] = [sd_cli]

        model = str(self._cfg.model or "").strip()
        diffusion_model = str(self._cfg.diffusion_model or "").strip()
        if model:
            cmd.extend(["--model", model])
        elif diffusion_model:
            cmd.extend(["--diffusion-model", diffusion_model])
        else:
            raise OptionalDependencyMissingError(
                "StableDiffusionCppVisionBackend is not configured. "
                "Set `model` (full model) or `diffusion_model` (component mode)."
            )

        if self._cfg.vae:
            cmd.extend(["--vae", str(self._cfg.vae)])
        if self._cfg.llm:
            cmd.extend(["--llm", str(self._cfg.llm)])
        if self._cfg.llm_vision:
            cmd.extend(["--llm_vision", str(self._cfg.llm_vision)])
        if self._cfg.clip_l:
            cmd.extend(["--clip_l", str(self._cfg.clip_l)])
        if self._cfg.clip_g:
            cmd.extend(["--clip_g", str(self._cfg.clip_g)])
        if self._cfg.t5xxl:
            cmd.extend(["--t5xxl", str(self._cfg.t5xxl)])

        cmd.extend(_flatten(self._cfg.extra_args))
        return cmd

    def _select_mode(self) -> str:
        if self._mode:
            return self._mode

        try:
            self._sd_cli_resolved = _require_sd_cli(self._cfg.sd_cli_path)
            self._mode = "cli"
            return self._mode
        except OptionalDependencyMissingError as cli_error:
            try:
                import stable_diffusion_cpp  # type: ignore
            except Exception as e:
                raise OptionalDependencyMissingError(
                    f"{cli_error} Alternatively, install `stable-diffusion-cpp-python` to use the pip-installable "
                    "stable-diffusion.cpp python bindings."
                ) from e

            self._py_sd = stable_diffusion_cpp
            self._mode = "python"
            return self._mode

    def _ensure_python_model(self) -> Any:
        if self._py_model is not None:
            return self._py_model

        self._select_mode()
        if self._mode != "python":
            raise RuntimeError("Internal error: python model requested while backend is in CLI mode.")

        init_kwargs, default_generate_kwargs = _parse_sdcpp_extra_args(self._cfg.extra_args)
        self._py_init_kwargs = init_kwargs
        self._py_default_generate_kwargs = default_generate_kwargs

        model = str(self._cfg.model or "").strip()
        diffusion_model = str(self._cfg.diffusion_model or "").strip()
        if not model and not diffusion_model:
            raise OptionalDependencyMissingError(
                "StableDiffusionCppVisionBackend is not configured. "
                "Set `model` (full model) or `diffusion_model` (component mode)."
            )

        # stable-diffusion-cpp-python accepts both full model and component paths.
        self._py_model = self._py_sd.StableDiffusion(  # type: ignore[attr-defined]
            model_path=model,
            diffusion_model_path=diffusion_model,
            vae_path=str(self._cfg.vae or ""),
            llm_path=str(self._cfg.llm or ""),
            llm_vision_path=str(self._cfg.llm_vision or ""),
            clip_l_path=str(self._cfg.clip_l or ""),
            clip_g_path=str(self._cfg.clip_g or ""),
            t5xxl_path=str(self._cfg.t5xxl or ""),
            **(self._py_init_kwargs or {}),
        )
        return self._py_model

    def _validate_qwen_image_components(self) -> None:
        diffusion_model = str(self._cfg.diffusion_model or "").strip()
        if not diffusion_model:
            return
        arch = _try_read_gguf_architecture(diffusion_model)
        if arch not in {"qwen_image", "qwen_image_edit"}:
            return
        if not str(self._cfg.vae or "").strip():
            raise OptionalDependencyMissingError("Qwen Image GGUF requires `vae` (e.g. qwen_image_vae.safetensors).")
        if not str(self._cfg.llm or "").strip():
            raise OptionalDependencyMissingError("Qwen Image GGUF requires `llm` (e.g. Qwen2.5-VL-7B-Instruct-*.gguf).")

    def _run(self, cmd: List[str]) -> None:
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self._cfg.cwd) if self._cfg.cwd else None,
                timeout=float(self._cfg.timeout_s),
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"sd-cli timed out after {self._cfg.timeout_s}s") from e
        except subprocess.CalledProcessError as e:
            out = (e.stdout or b"") + b"\n" + (e.stderr or b"")
            msg = out.decode("utf-8", errors="replace")[:4000]
            raise RuntimeError(f"sd-cli failed (exit={e.returncode}). Output:\n{msg}") from e
        except FileNotFoundError as e:
            raise OptionalDependencyMissingError(
                "stable-diffusion.cpp executable not found. "
                "Install `sd-cli` from https://github.com/leejet/stable-diffusion.cpp/releases "
                "or install `stable-diffusion-cpp-python` for pip-installable python bindings, "
                "or set sd_cli_path to the executable path."
            ) from e

    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
        self._validate_qwen_image_components()
        mode = self._select_mode()
        if mode == "cli":
            with tempfile.TemporaryDirectory(prefix="abstractvision-sdcpp-") as td:
                out_path = Path(td) / "output.png"
                cmd = self._base_cmd()
                cmd.extend(["--output", str(out_path)])
                cmd.extend(["--prompt", str(request.prompt)])

                if request.negative_prompt is not None:
                    cmd.extend(["--negative-prompt", str(request.negative_prompt)])
                if request.width is not None:
                    cmd.extend(["--width", str(int(request.width))])
                if request.height is not None:
                    cmd.extend(["--height", str(int(request.height))])
                if request.steps is not None:
                    cmd.extend(["--steps", str(int(request.steps))])
                if request.guidance_scale is not None:
                    cmd.extend(["--cfg-scale", str(float(request.guidance_scale))])
                if request.seed is not None:
                    cmd.extend(["--seed", str(int(request.seed))])

                cmd.extend(_extra_to_cli_args(request.extra))
                self._run(cmd)

                data = out_path.read_bytes()
                mime = _sniff_mime_type(data)
                if not mime.startswith("image/"):
                    raise ValueError("sd-cli produced an unexpected output format (expected an image).")
                return GeneratedAsset(
                    media_type="image",
                    data=data,
                    mime_type=mime,
                    metadata={
                        "source": "stable-diffusion.cpp",
                        "mode": "cli",
                        "sd_cli": str(self._cfg.sd_cli_path),
                        "model": self._cfg.model,
                        "diffusion_model": self._cfg.diffusion_model,
                    },
                )

        model = self._ensure_python_model()
        kwargs = dict(self._py_default_generate_kwargs or {})
        kwargs.update(
            {
                "prompt": str(request.prompt),
                "negative_prompt": str(request.negative_prompt or ""),
            }
        )

        if request.width is not None:
            kwargs["width"] = int(request.width)
        if request.height is not None:
            kwargs["height"] = int(request.height)
        if request.steps is not None:
            kwargs["sample_steps"] = int(request.steps)
        if request.guidance_scale is not None:
            kwargs["cfg_scale"] = float(request.guidance_scale)
        if request.seed is not None:
            kwargs["seed"] = int(request.seed)

        kwargs.update(_extra_to_python_generate_kwargs(request.extra))
        kwargs = _filter_generate_kwargs(model, kwargs)

        images = model.generate_image(**kwargs)
        if not images:
            raise RuntimeError("stable-diffusion.cpp python bindings produced no images.")
        img0 = images[0]
        buf = BytesIO()
        img0.save(buf, format="PNG")
        data = buf.getvalue()
        mime = _sniff_mime_type(data)
        return GeneratedAsset(
            media_type="image",
            data=data,
            mime_type=mime,
            metadata={
                "source": "stable-diffusion.cpp",
                "mode": "python",
                "python_package": getattr(self._py_sd, "__version__", None),
                "model": self._cfg.model,
                "diffusion_model": self._cfg.diffusion_model,
            },
        )

    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
        self._validate_qwen_image_components()
        mode = self._select_mode()
        if mode == "cli":
            with tempfile.TemporaryDirectory(prefix="abstractvision-sdcpp-") as td:
                td_p = Path(td)
                init_ext = _sniff_ext(request.image)
                init_path = td_p / f"init{init_ext}"
                init_path.write_bytes(bytes(request.image))

                mask_path: Optional[Path] = None
                if request.mask is not None:
                    mask_ext = _sniff_ext(request.mask)
                    mask_path = td_p / f"mask{mask_ext}"
                    mask_path.write_bytes(bytes(request.mask))

                out_path = td_p / "output.png"

                cmd = self._base_cmd()
                cmd.extend(["--output", str(out_path)])
                cmd.extend(["--prompt", str(request.prompt)])
                cmd.extend(["--init-img", str(init_path)])
                if mask_path is not None:
                    cmd.extend(["--mask", str(mask_path)])

                if request.negative_prompt is not None:
                    cmd.extend(["--negative-prompt", str(request.negative_prompt)])
                if request.steps is not None:
                    cmd.extend(["--steps", str(int(request.steps))])
                if request.guidance_scale is not None:
                    cmd.extend(["--cfg-scale", str(float(request.guidance_scale))])
                if request.seed is not None:
                    cmd.extend(["--seed", str(int(request.seed))])

                cmd.extend(_extra_to_cli_args(request.extra))
                self._run(cmd)

                data = out_path.read_bytes()
                mime = _sniff_mime_type(data)
                if not mime.startswith("image/"):
                    raise ValueError("sd-cli produced an unexpected output format (expected an image).")
                return GeneratedAsset(
                    media_type="image",
                    data=data,
                    mime_type=mime,
                    metadata={
                        "source": "stable-diffusion.cpp",
                        "mode": "cli",
                        "sd_cli": str(self._cfg.sd_cli_path),
                        "model": self._cfg.model,
                        "diffusion_model": self._cfg.diffusion_model,
                    },
                )

        model = self._ensure_python_model()
        kwargs = dict(self._py_default_generate_kwargs or {})
        kwargs.update(
            {
                "prompt": str(request.prompt),
                "negative_prompt": str(request.negative_prompt or ""),
            }
        )

        from PIL import Image  # pillow is a dependency of stable-diffusion-cpp-python

        init_img = Image.open(BytesIO(bytes(request.image)))
        kwargs["init_image"] = init_img
        if request.mask is not None:
            kwargs["mask_image"] = Image.open(BytesIO(bytes(request.mask)))

        if request.steps is not None:
            kwargs["sample_steps"] = int(request.steps)
        if request.guidance_scale is not None:
            kwargs["cfg_scale"] = float(request.guidance_scale)
        if request.seed is not None:
            kwargs["seed"] = int(request.seed)

        kwargs.update(_extra_to_python_generate_kwargs(request.extra))
        kwargs = _filter_generate_kwargs(model, kwargs)

        images = model.generate_image(**kwargs)
        if not images:
            raise RuntimeError("stable-diffusion.cpp python bindings produced no images.")
        img0 = images[0]
        buf = BytesIO()
        img0.save(buf, format="PNG")
        data = buf.getvalue()
        mime = _sniff_mime_type(data)
        return GeneratedAsset(
            media_type="image",
            data=data,
            mime_type=mime,
            metadata={
                "source": "stable-diffusion.cpp",
                "mode": "python",
                "python_package": getattr(self._py_sd, "__version__", None),
                "model": self._cfg.model,
                "diffusion_model": self._cfg.diffusion_model,
            },
        )

    def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]:
        raise CapabilityNotSupportedError("StableDiffusionCppVisionBackend does not implement multi-view generation.")

    def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
        raise CapabilityNotSupportedError("StableDiffusionCppVisionBackend does not implement text_to_video (phase 2).")

    def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
        raise CapabilityNotSupportedError("StableDiffusionCppVisionBackend does not implement image_to_video (phase 2).")
