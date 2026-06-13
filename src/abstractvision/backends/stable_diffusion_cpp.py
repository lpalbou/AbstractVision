from __future__ import annotations

import json
import os
import platform as _platform
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..errors import CapabilityNotSupportedError, OptionalDependencyMissingError
from ..model_capabilities import VisionModelCapabilitiesRegistry
from ..model_cache import cached_hf_model_sources
from ..types import (
    GeneratedAsset,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageToVideoRequest,
    MultiAngleRequest,
    ProviderModelInfo,
    VideoGenerationRequest,
    VisionBackendCapabilities,
)
from .base_backend import VisionBackend

_MACOS_GGUF_DISABLED_ERROR = (
    "GGUF 8-bit stable-diffusion.cpp execution is disabled on this macOS host. "
    "This is controlled by `ABSTRACTVISION_DISABLE_GGUF_ON_MACOS=1`."
)


def _looks_like_gguf_reference(value: Any) -> bool:
    s = str(value or "").strip()
    if not s:
        return False
    name = Path(s).name.lower()
    return Path(s).suffix.lower() == ".gguf" or "gguf" in name


def _raise_if_macos_gguf_requested(*values: Any) -> None:
    if sys.platform != "darwin":
        return
    if not _env_truthy("ABSTRACTVISION_DISABLE_GGUF_ON_MACOS", default=False):
        return
    if any(_looks_like_gguf_reference(value) for value in values):
        raise CapabilityNotSupportedError(_MACOS_GGUF_DISABLED_ERROR)


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
            '(or install `abstractvision[sdcpp]` to use pip-installable python bindings). '
            "If you intended to run a standard Diffusers model (e.g. 'runwayml/stable-diffusion-v1-5'), use the "
            "Diffusers backend instead."
        )

    # If the user passed a path-like string, validate it exists; otherwise rely on PATH lookup.
    looks_like_path = os.sep in p or (os.altsep and os.altsep in p) or p.startswith(".")
    if looks_like_path:
        if not Path(p).expanduser().exists():
            raise OptionalDependencyMissingError(
                f"stable-diffusion.cpp executable not found at: {p!r}. "
                "Install from https://github.com/leejet/stable-diffusion.cpp/releases or install `abstractvision[sdcpp]`, "
                "or update sd_cli_path. "
                "If you intended to run a standard Diffusers model (e.g. 'runwayml/stable-diffusion-v1-5'), use the "
                "Diffusers backend instead."
            )
        return p

    resolved = shutil.which(p)
    if not resolved:
        # Managed install path (no PATH modification required).
        managed_dir = Path.home() / ".abstractvision" / "bin"
        managed = managed_dir / p
        if managed.exists():
            return str(managed)

        # Best-effort: auto-download `sd-cli` for local checkpoint execution without
        # requiring users to compile stable-diffusion.cpp.
        if p in {"sd-cli", "sd-cli.exe"} and _env_truthy("ABSTRACTVISION_SDCPP_AUTO_INSTALL", default=True):
            installed = _try_auto_install_sd_cli(managed_dir)
            if installed is not None:
                return installed
        raise OptionalDependencyMissingError(
            f"stable-diffusion.cpp executable not found in PATH: {p!r}. "
            "Install from https://github.com/leejet/stable-diffusion.cpp/releases or install `abstractvision[sdcpp]`, "
            "or set sd_cli_path. "
            "If you intended to run a standard Diffusers model (e.g. 'runwayml/stable-diffusion-v1-5'), use the "
            "Diffusers backend instead."
        )
    return resolved


def _env_truthy(key: str, *, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return bool(default)
    v = str(raw).strip().lower()
    if v in {"", "0", "false", "no", "off"}:
        return False
    return True


def _is_high_memory_machine(*, min_memory_gb: float = 18.0) -> bool:
    """Best-effort RAM heuristic for backend selection decisions.

    Tests patch this helper directly; keep it stdlib-only and fast.
    """

    try:
        if hasattr(os, "sysconf"):
            page_size = float(os.sysconf("SC_PAGE_SIZE"))
            pages = float(os.sysconf("SC_PHYS_PAGES"))
            total_bytes = page_size * pages
            return total_bytes >= float(min_memory_gb) * 1024.0 * 1024.0 * 1024.0
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if proc.returncode == 0:
                total_bytes = float(str(proc.stdout or "").strip() or "0")
                return total_bytes >= float(min_memory_gb) * 1024.0 * 1024.0 * 1024.0
        except Exception:
            pass
    return False


def _try_auto_install_sd_cli(dest_dir: Path) -> Optional[str]:
    """Download and install `sd-cli` into `~/.abstractvision/bin` (best-effort).

    This targets the stable-diffusion.cpp GitHub releases and avoids requiring a
    compiler toolchain for `stable-diffusion-cpp-python`.
    """

    try:
        plat = sys.platform
        machine = str(_platform.machine() or "").lower()

        exe_name = "sd-cli.exe" if plat.startswith("win") else "sd-cli"
        if plat == "darwin" and machine not in {"arm64", "aarch64"}:
            return None
        if plat == "linux" and machine not in {"x86_64", "amd64", "aarch64", "arm64"}:
            return None

        dest_dir = Path(dest_dir).expanduser().resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_cli = dest_dir / exe_name
        if dest_cli.exists():
            return str(dest_cli)

        # Query latest release metadata.
        api_url = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest"
        req = urllib.request.Request(
            api_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "abstractvision",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            meta = json.loads(r.read().decode("utf-8", errors="replace"))

        assets = meta.get("assets") or []

        def pick_asset_url() -> Optional[str]:
            scored: List[tuple[int, str]] = []
            for a in assets:
                name = str(a.get("name") or "")
                url = str(a.get("browser_download_url") or "")
                if not name or not url:
                    continue
                low = name.lower()
                if not low.endswith(".zip"):
                    continue

                score = 0
                if plat == "darwin":
                    if "bin-darwin" not in low or "arm64" not in low:
                        continue
                    score = 100
                elif plat.startswith("win"):
                    if "bin-win" not in low or "x64" not in low:
                        continue
                    # Prefer CUDA build if present.
                    if "cuda" in low:
                        score += 50
                    if "vulkan" in low:
                        score += 10
                elif plat == "linux":
                    if "bin-linux" not in low:
                        continue
                    if "x86_64" not in low and "arm64" not in low and "aarch64" not in low:
                        continue
                    # Prefer Vulkan build when an NVIDIA runtime is present.
                    if shutil.which("nvidia-smi") and "vulkan" in low:
                        score += 30
                    # Otherwise prefer the plain build.
                    if "vulkan" not in low and "rocm" not in low:
                        score += 5
                else:
                    continue
                scored.append((score, url))
            scored.sort(reverse=True, key=lambda x: x[0])
            return scored[0][1] if scored else None

        selected_url = pick_asset_url()
        if not selected_url:
            return None

        # Download + extract.
        with tempfile.TemporaryDirectory(prefix="abstractvision-sdcpp-bin-") as td:
            td_p = Path(td)
            zip_path = td_p / "sdcpp.zip"
            dl_req = urllib.request.Request(selected_url, headers={"User-Agent": "abstractvision"})
            with urllib.request.urlopen(dl_req, timeout=60) as r, zip_path.open("wb") as f:
                shutil.copyfileobj(r, f)

            extract_dir = td_p / "extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

            cli_path: Optional[Path] = None
            for root, _dirs, files in os.walk(extract_dir):
                if exe_name in files:
                    cli_path = Path(root) / exe_name
                    break
            if cli_path is None or not cli_path.exists():
                return None

            cli_dir = cli_path.parent
            for item in cli_dir.iterdir():
                if item.is_file():
                    target = dest_dir / item.name
                    shutil.copy2(item, target)

        # Make executable + remove quarantine attribute (macOS Gatekeeper).
        if not plat.startswith("win"):
            try:
                dest_cli.chmod(0o755)
            except Exception:
                pass
        if plat == "darwin":
            try:
                subprocess.run(
                    ["xattr", "-rd", "com.apple.quarantine", str(dest_dir)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        return str(dest_cli) if dest_cli.exists() else None
    except Exception:
        return None


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
        elif k == "vae_on_cpu" and bool(v):
            init_kwargs["keep_vae_on_cpu"] = True
        elif k == "clip_on_cpu" and bool(v):
            init_kwargs["keep_clip_on_cpu"] = True
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


def _ensure_ggml_metal_resources() -> None:
    """Ensure ggml's Metal backend can find its shader sources on macOS.

    Some stable-diffusion.cpp builds rely on locating `ggml-metal.metal` via
    the `GGML_METAL_PATH_RESOURCES` environment variable at runtime. AbstractVision
    vendors a copy under `abstractvision/assets/ggml-metal.metal` so users don't
    have to manage the resource path themselves.
    """

    if sys.platform != "darwin":
        return
    if str(os.environ.get("GGML_METAL_PATH_RESOURCES") or "").strip():
        return
    try:
        from importlib import resources as importlib_resources  # py3.9+

        import abstractvision.assets as av_assets

        metal = importlib_resources.files(av_assets).joinpath("ggml-metal.metal")
        with importlib_resources.as_file(metal) as metal_path:
            if metal_path.exists():
                os.environ.setdefault("GGML_METAL_PATH_RESOURCES", str(metal_path.parent))
                return
    except Exception:
        pass

    # #FALLBACK: when running from source trees, resolve relative to this file.
    try:
        assets_dir = Path(__file__).resolve().parents[1] / "assets"
        metal_path = assets_dir / "ggml-metal.metal"
        if metal_path.exists():
            os.environ.setdefault("GGML_METAL_PATH_RESOURCES", str(assets_dir))
    except Exception:
        pass


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


def _infer_gguf_architecture(path: str) -> Optional[str]:
    """Best-effort architecture inference.

    Some community GGUFs (notably Qwen Image Edit) ship with `general.architecture=qwen_image`
    even when they are edit-capable. stable-diffusion.cpp itself selects edit-specific codepaths
    based on model identity, so we also use filename heuristics to distinguish the variants.
    """

    arch = _try_read_gguf_architecture(path)
    if arch != "qwen_image":
        return arch
    name = Path(str(path)).name.lower()
    if "qwen-image-edit" in name or "qwen_image_edit" in name or "image-edit" in name or "image_edit" in name:
        return "qwen_image_edit"
    return arch


def _cmd_has_flag(cmd: Sequence[str], flag: str) -> bool:
    return any(str(x) == flag for x in cmd)


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
    - llm (Qwen2.5-VL text encoder; safetensors or GGUF depending on the variant)
    """

    sd_cli_path: str = "sd-cli"

    # Single-file full model
    model: Optional[str] = None
    capabilities_model_id: Optional[str] = None

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
    timeout_s: Optional[float] = None
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

    def preload(self) -> None:
        self._validate_macos_gguf_supported()
        mode = self._select_mode()
        if mode == "python":
            self._ensure_python_model()

    def unload(self) -> None:
        # Best-effort: drop python-binding model reference so native memory can be reclaimed.
        self._py_model = None
        self._py_init_kwargs = None
        self._py_default_generate_kwargs = None
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    def get_capabilities(self) -> VisionBackendCapabilities:
        self._validate_macos_gguf_supported()
        supported_tasks = ["text_to_image", "image_to_image"]
        capability_model_id = str(self._cfg.capabilities_model_id or "").strip()
        if capability_model_id:
            try:
                reg = VisionModelCapabilitiesRegistry()
                supported_tasks = sorted(str(task_name) for task_name in reg.get(capability_model_id).tasks.keys())
            except Exception:
                pass
        return VisionBackendCapabilities(
            supported_tasks=supported_tasks,
            supports_mask="image_to_image" in set(supported_tasks),
        )

    def _supported_task_names(self, model_id: str) -> List[str]:
        try:
            reg = VisionModelCapabilitiesRegistry()
            return sorted(str(task_name) for task_name in reg.get(str(model_id)).tasks.keys())
        except Exception:
            return ["image_to_image", "text_to_image"]

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        if sys.platform == "darwin" and _env_truthy("ABSTRACTVISION_DISABLE_GGUF_ON_MACOS", default=False):
            return []
        task_s = str(task or "").strip()
        out: List[ProviderModelInfo] = []
        try:
            from ..model_downloads import model_presets
        except Exception:
            return out

        for preset in model_presets(target="gguf", engine="stable-diffusion.cpp", include_non_8bit=False):
            cached_in = cached_hf_model_sources(str(preset.repo_id), require_weight_files=True)
            if not cached_in:
                continue
            model_id = str(preset.upstream_repo_id or preset.repo_id)
            tasks = self._supported_task_names(model_id)
            if task_s and task_s not in tasks:
                continue
            out.append(
                ProviderModelInfo(
                    id=str(preset.key),
                    object="model",
                    owned_by="stable-diffusion.cpp",
                    capabilities=tuple(tasks),
                    raw={
                        "provider": "sdcpp",
                        "backend": "sdcpp",
                        "engine": "stable-diffusion.cpp",
                        "target": str(preset.target),
                        "model": f"sdcpp/{preset.key}",
                        "routed_model": f"sdcpp/{preset.key}",
                        "repo_id": str(preset.repo_id),
                        "upstream_repo_id": str(preset.upstream_repo_id or ""),
                        "download_repo_id": str(preset.repo_id),
                        "quantization_bits": preset.quantization_bits,
                        "local_cached": True,
                        "cached_in": list(cached_in),
                    },
                )
            )
        return out

    def _base_cmd(self) -> List[str]:
        self._validate_macos_gguf_supported()
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
        self._validate_macos_gguf_supported()
        if self._mode:
            return self._mode

        try:
            self._sd_cli_resolved = _require_sd_cli(self._cfg.sd_cli_path)
            self._mode = "cli"
            return self._mode
        except OptionalDependencyMissingError as cli_error:
            try:
                _ensure_ggml_metal_resources()
                import stable_diffusion_cpp  # type: ignore
            except Exception as e:
                raise OptionalDependencyMissingError(
                    f"{cli_error} Alternatively, install `abstractvision[sdcpp]` to use the pip-installable "
                    "stable-diffusion.cpp python bindings."
                ) from e

            self._py_sd = stable_diffusion_cpp
            self._mode = "python"
            return self._mode

    def _validate_macos_gguf_supported(self) -> None:
        _raise_if_macos_gguf_requested(
            self._cfg.model,
            self._cfg.diffusion_model,
            self._cfg.llm,
            self._cfg.llm_vision,
            self._cfg.clip_l,
            self._cfg.clip_g,
            self._cfg.t5xxl,
        )

    def _ensure_python_model(self) -> Any:
        if self._py_model is not None:
            return self._py_model

        self._select_mode()
        if self._mode != "python":
            raise RuntimeError("Internal error: python model requested while backend is in CLI mode.")

        init_kwargs, default_generate_kwargs = _parse_sdcpp_extra_args(self._cfg.extra_args)

        diffusion_model = str(self._cfg.diffusion_model or "").strip()
        if diffusion_model:
            arch = _infer_gguf_architecture(diffusion_model)
            if arch in {"qwen_image", "qwen_image_edit"}:
                # stable-diffusion.cpp docs recommend flow_shift=3 for Qwen Image / Qwen Image Edit.
                init_kwargs.setdefault("flow_shift", 3.0)
                init_kwargs.setdefault("enable_mmap", True)
                # Speed: Qwen Image Edit benefits from flash-attn in the diffusion model when available.
                init_kwargs.setdefault("diffusion_flash_attn", True)
                if sys.platform == "darwin":
                    # #FALLBACK: stable-diffusion.cpp Metal backend does not reliably support
                    # Qwen Image Edit's CLIP/VAE paths; keep them on CPU for correctness.
                    init_kwargs.setdefault("keep_vae_on_cpu", True)
                    init_kwargs.setdefault("keep_clip_on_cpu", True)
                    # Only offload diffusion params on memory-constrained machines.
                    if not _is_high_memory_machine():
                        init_kwargs.setdefault("offload_params_to_cpu", True)
                else:
                    init_kwargs.setdefault("offload_params_to_cpu", True)
            # stable-diffusion.cpp: Qwen Image Edit 2511 requires `--qwen-image-zero-cond-t`
            # (python bindings: qwen_image_zero_cond_t=True) or edit quality degrades significantly.
            if arch == "qwen_image_edit" and "2511" in Path(diffusion_model).name:
                init_kwargs.setdefault("qwen_image_zero_cond_t", True)

        self._py_init_kwargs = init_kwargs
        self._py_default_generate_kwargs = default_generate_kwargs

        model = str(self._cfg.model or "").strip()
        diffusion_model = diffusion_model
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
        arch = _infer_gguf_architecture(diffusion_model)
        if arch not in {"qwen_image", "qwen_image_edit"}:
            return
        if not str(self._cfg.vae or "").strip():
            raise OptionalDependencyMissingError("Qwen Image GGUF requires `vae` (e.g. qwen_image_vae.safetensors).")
        if not str(self._cfg.llm or "").strip():
            raise OptionalDependencyMissingError(
                "Qwen Image GGUF requires `llm` (e.g. qwen_2.5_vl_7b.safetensors or Qwen2.5-VL-7B-Instruct-*.gguf)."
            )
        llm_name = Path(str(self._cfg.llm)).name.lower()
        if "fp8_scaled" in llm_name:
            raise ValueError(
                "Qwen Image stable-diffusion.cpp backend does not support fp8_scaled text encoders "
                "(often produces blank/black outputs). Use `qwen_2.5_vl_7b.safetensors` instead."
            )

    def _run(self, cmd: List[str]) -> None:
        _ensure_ggml_metal_resources()
        stream_output = _env_truthy("ABSTRACTVISION_SDCPP_STREAM_OUTPUT", default=sys.stderr.isatty())
        if stream_output and sys.stderr.isatty():
            try:
                exe = Path(str(cmd[0])).name
            except Exception:
                exe = str(cmd[0])
            print(
                f"[abstractvision] running {exe} (stable-diffusion.cpp). "
                "The Python process may look idle because compute happens in the child process; "
                "this can take several minutes on first load.",
                file=sys.stderr,
                flush=True,
            )
        try:
            if stream_output:
                # Stream sd-cli output to stderr so our stdout stays machine-readable
                # for the CLI JSON payload printed by AbstractVision.
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(self._cfg.cwd) if self._cfg.cwd else None,
                )
                if sys.stderr.isatty():
                    print(f"[abstractvision] {Path(str(cmd[0])).name} pid={proc.pid}", file=sys.stderr, flush=True)

                # Keep a bounded tail for error reporting without unbounded memory growth.
                tail = deque(maxlen=128)  # 128 * 4096 = 512 KiB

                def _pump() -> None:
                    try:
                        if proc.stdout is None:
                            return
                        while True:
                            chunk = proc.stdout.read(4096)
                            if not chunk:
                                break
                            tail.append(chunk)
                            try:
                                sys.stderr.buffer.write(chunk)
                                sys.stderr.buffer.flush()
                            except Exception:
                                pass
                    except Exception:
                        pass

                t = threading.Thread(target=_pump, daemon=True)
                t.start()
                try:
                    rc = proc.wait()
                finally:
                    try:
                        t.join(timeout=1.0)
                    except Exception:
                        pass

                if rc != 0:
                    out = b"".join(tail)
                    raise subprocess.CalledProcessError(rc, cmd, output=out)
            else:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(self._cfg.cwd) if self._cfg.cwd else None,
                )
        except subprocess.CalledProcessError as e:
            out = (getattr(e, "output", None) or getattr(e, "stdout", None) or b"") + b"\n" + (getattr(e, "stderr", None) or b"")
            # Prefer the tail so we keep the most actionable lines (e.g. missing backend ops / abort reason).
            decoded = out.decode("utf-8", errors="replace")
            msg = decoded[-4000:].strip()
            suffix = f" Output:\n{msg}" if msg else ""
            raise RuntimeError(f"sd-cli failed (exit={e.returncode}).{suffix}") from e
        except FileNotFoundError as e:
            raise OptionalDependencyMissingError(
                "stable-diffusion.cpp executable not found. "
                "Install `sd-cli` from https://github.com/leejet/stable-diffusion.cpp/releases "
                "or install `abstractvision[sdcpp]` for pip-installable python bindings, "
                "or set sd_cli_path to the executable path."
            ) from e

    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
        return self.generate_image_with_progress(request, progress_callback=None)

    def _generate_image_python(
        self,
        request: ImageGenerationRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        model = self._ensure_python_model()
        kwargs = dict(self._py_default_generate_kwargs or {})
        kwargs.update(
            {
                "prompt": str(request.prompt),
                "negative_prompt": str(request.negative_prompt or ""),
            }
        )

        if progress_callback is not None:
            zero_based: Dict[str, Optional[bool]] = {"v": None}

            def _pcb(*args: Any, **_kw: Any) -> bool:
                try:
                    step = int(args[0]) if len(args) >= 1 else 0
                    total = int(args[1]) if len(args) >= 2 else None
                    if zero_based["v"] is None:
                        zero_based["v"] = (step == 0)
                    if zero_based["v"]:
                        step = step + 1
                    progress_callback(step, total)
                except Exception:
                    pass
                return True

            kwargs["progress_callback"] = _pcb

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

    def generate_image_with_progress(
        self,
        request: ImageGenerationRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        self._validate_macos_gguf_supported()
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

        return self._generate_image_python(request, progress_callback)

    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
        return self.edit_image_with_progress(request, progress_callback=None)

    def edit_image_with_progress(
        self,
        request: ImageEditRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        self._validate_macos_gguf_supported()
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

                diffusion_model = str(self._cfg.diffusion_model or "").strip()
                arch = _infer_gguf_architecture(diffusion_model) if diffusion_model else None
                qwen_ref_edit = (arch == "qwen_image_edit") and (mask_path is None)

                # Qwen Image Edit uses ref images (not init-img) for the VLM+VAE dual-conditioning path.
                # For mask workflows we still pass init-img so inpaint-style edits can work.
                if qwen_ref_edit:
                    # sd-cli uses `-r` for reference images (Qwen Edit uses ref images, not init-img).
                    cmd.extend(["-r", str(init_path)])
                    if "2511" in Path(diffusion_model).name and "--qwen-image-zero-cond-t" not in cmd:
                        cmd.append("--qwen-image-zero-cond-t")
                    if "--flow-shift" not in cmd:
                        cmd.extend(["--flow-shift", "3"])
                    if "--diffusion-fa" not in cmd:
                        cmd.append("--diffusion-fa")
                    if "--mmap" not in cmd:
                        cmd.append("--mmap")
                    if sys.platform == "darwin" and _is_high_memory_machine():
                        backend_spec = "diffusion=mtl0,clip=cpu,vae=cpu"
                        if "--backend" not in cmd:
                            cmd.extend(["--backend", backend_spec])
                        if "--params-backend" not in cmd:
                            cmd.extend(["--params-backend", backend_spec])
                else:
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

        if progress_callback is not None:
            zero_based: Dict[str, Optional[bool]] = {"v": None}

            def _pcb(*args: Any, **_kw: Any) -> bool:
                try:
                    step = int(args[0]) if len(args) >= 1 else 0
                    total = int(args[1]) if len(args) >= 2 else None
                    if zero_based["v"] is None:
                        zero_based["v"] = (step == 0)
                    if zero_based["v"]:
                        step = step + 1
                    progress_callback(step, total)
                except Exception:
                    pass
                return True

            kwargs["progress_callback"] = _pcb

        from PIL import Image  # pillow is a dependency of stable-diffusion-cpp-python

        init_img = Image.open(BytesIO(bytes(request.image)))
        arch = _infer_gguf_architecture(str(self._cfg.diffusion_model or "").strip()) if self._cfg.diffusion_model else None
        if arch == "qwen_image_edit":
            # Qwen Image Edit uses reference images for the VLM+VAE dual-conditioning path.
            kwargs["ref_images"] = [init_img]
            if request.mask is not None:
                # Best-effort: keep init+mask for inpaint-style edits, while still
                # providing the same image as reference to the Qwen conditioner.
                kwargs["init_image"] = init_img
                kwargs["mask_image"] = Image.open(BytesIO(bytes(request.mask)))
        else:
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
