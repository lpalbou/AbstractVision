from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
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
            "Set sd_cli_path or install `sd-cli` from https://github.com/leejet/stable-diffusion.cpp/releases"
        )

    # If the user passed a path-like string, validate it exists; otherwise rely on PATH lookup.
    looks_like_path = os.sep in p or (os.altsep and os.altsep in p) or p.startswith(".")
    if looks_like_path:
        if not Path(p).expanduser().exists():
            raise OptionalDependencyMissingError(
                f"stable-diffusion.cpp executable not found at: {p!r}. "
                "Install from https://github.com/leejet/stable-diffusion.cpp/releases or update sd_cli_path."
            )
        return p

    resolved = shutil.which(p)
    if not resolved:
        raise OptionalDependencyMissingError(
            f"stable-diffusion.cpp executable not found in PATH: {p!r}. "
            "Install from https://github.com/leejet/stable-diffusion.cpp/releases or set sd_cli_path."
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
    """Config for stable-diffusion.cpp `sd-cli` subprocess backend.

    This backend is dependency-light (stdlib only) but requires an external executable:
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

    # Extra args always appended to the `sd-cli` invocation (best-effort).
    extra_args: Sequence[str] = field(default_factory=tuple)

    # Safety
    timeout_s: float = 60.0 * 60.0  # 1h (image generation can be slow on CPU)
    cwd: Optional[str] = None


class StableDiffusionCppVisionBackend(VisionBackend):
    """Local vision backend that shells out to stable-diffusion.cpp `sd-cli`.

    Supports: text_to_image and image_to_image (including masks when the model supports it).
    """

    def __init__(self, *, config: StableDiffusionCppBackendConfig):
        self._cfg = config

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
                "or set sd_cli_path to the executable path."
            ) from e

    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
        self._validate_qwen_image_components()
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
                    "sd_cli": str(self._cfg.sd_cli_path),
                    "model": self._cfg.model,
                    "diffusion_model": self._cfg.diffusion_model,
                },
            )

    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
        self._validate_qwen_image_components()
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
                    "sd_cli": str(self._cfg.sd_cli_path),
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
