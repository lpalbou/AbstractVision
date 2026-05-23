from __future__ import annotations

import base64
import importlib.util
import json
import os
import shlex
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from .errors import AbstractVisionError
from .model_capabilities import VisionModelCapabilitiesRegistry
from .model_cache import (
    cached_hf_model_sources,
    default_legacy_model_root,
    framework_hf_cache_roots,
    incomplete_hf_model_sources,
)
from .model_downloads import catalog_target_scope, local_model_profile, model_presets, resolve_sdcpp_model_selection
from .types import GeneratedAsset, ImageEditRequest, ImageGenerationRequest, VideoGenerationRequest

DEFAULT_PLAYGROUND_HOST = "127.0.0.1"
DEFAULT_PLAYGROUND_PORT = 8091
DEFAULT_DIFFUSERS_MODEL_ID = "runwayml/stable-diffusion-v1-5"


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _env_bool(key: str, default: bool = False) -> bool:
    v = _env(key)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _default_backend_kind() -> str:
    explicit = _env("ABSTRACTVISION_PROVIDER") or _env("ABSTRACTVISION_BACKEND")
    if explicit:
        return str(explicit)
    if _env("OPENAI_BASE_URL"):
        return "openai"
    return ""


def _local_runtime_available(backend: Optional[str]) -> bool:
    kind = str(backend or "").strip().lower().replace("_", "-")
    if kind in {"openai", "openai-compatible", "proxy", ""}:
        return True
    if kind in {"huggingface", "hf", "diffusers", "hf-diffusers"}:
        return importlib.util.find_spec("diffusers") is not None and importlib.util.find_spec("torch") is not None
    if kind in {"mflux", "m-flux", "mlx"}:
        return sys.platform == "darwin" and importlib.util.find_spec("mflux") is not None and importlib.util.find_spec("mlx") is not None
    if kind in {"sdcpp", "stable-diffusion.cpp", "stable_diffusion_cpp", "stable-diffusion-cpp"}:
        return importlib.util.find_spec("stable_diffusion_cpp") is not None
    return True


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    s = str(value).strip()
    if not s:
        return None
    return int(s)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    return float(s)


def _redact(text: Any) -> str:
    raw = str(text or "")
    for key in (
        "OPENAI_API_KEY",
        "ABSTRACTCORE_SERVER_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        value = os.environ.get(key)
        if value:
            raw = raw.replace(str(value), "***")
    return raw


def _known_prefix(model_id: str) -> Tuple[Optional[str], str]:
    s = str(model_id or "").strip()
    if not s or "/" not in s:
        return None, s
    head, tail = s.split("/", 1)
    head = head.strip().lower()
    if head in {
        "diffusers",
        "huggingface",
        "hf",
        "mlx",
        "mflux",
        "sdcpp",
        "stable-diffusion.cpp",
        "stable_diffusion_cpp",
        "stable-diffusion-cpp",
        "openai",
        "openai-compatible",
        "openai_compatible",
    }:
        return head, tail.strip()
    return None, s


def _is_default_alias(value: str) -> bool:
    return str(value or "").strip().lower() in {"", "default", "server/default"}


def normalize_model_id_for_backend(model_id: str) -> Tuple[str, Optional[str]]:
    """Return ``(backend_kind, backend_model_id)`` for playground model-load requests.

    AbstractVision's own registry uses raw Hugging Face repo ids, while some
    callers use an explicit backend prefix. The playground accepts both shapes
    so the local UI can stay independent without being brittle.
    """

    s = str(model_id or "").strip()
    if not s:
        raise ValueError("Missing required field: model_id")

    prefix, rest = _known_prefix(s)
    if prefix in {"diffusers", "huggingface", "hf"}:
        return "diffusers", DEFAULT_DIFFUSERS_MODEL_ID if _is_default_alias(rest) else rest
    if prefix == "mlx":
        raise ValueError(
            "AbstractVision does not have a generic MLX image backend yet. "
            "Use `mflux/<preset>` for MFLUX-compatible 8-bit MLX models."
        )
    if prefix == "mflux":
        return "mflux", None if _is_default_alias(rest) else rest
    if prefix in {"sdcpp", "stable-diffusion.cpp", "stable_diffusion_cpp", "stable-diffusion-cpp"}:
        return "sdcpp", None if _is_default_alias(rest) else rest
    if prefix in {"openai", "openai-compatible", "openai_compatible"}:
        return "openai", None if _is_default_alias(rest) else rest

    # Raw registry ids like `runwayml/stable-diffusion-v1-5` are first-class in
    # AbstractVision and map to the local Diffusers backend by default.
    if _is_default_alias(s):
        return "diffusers", DEFAULT_DIFFUSERS_MODEL_ID
    return "diffusers", s


def _hf_cache_roots(extra_cache_dir: Optional[str] = None) -> List[Tuple[str, Path]]:
    roots: List[Tuple[str, Path]] = []

    def add(label: str, value: Optional[str]) -> None:
        if not value:
            return
        p = Path(str(value)).expanduser()
        if not any(existing.resolve() == p.resolve() for _, existing in roots):
            roots.append((label, p))

    add("configured cache", extra_cache_dir)
    add("HF_HUB_CACHE", _env("HF_HUB_CACHE"))
    hf_home = _env("HF_HOME")
    if hf_home:
        add("HF_HOME", str(Path(hf_home).expanduser() / "hub"))
    add("default HF cache", str(Path.home() / ".cache" / "huggingface" / "hub"))
    return roots


def _cached_hf_model_sources(
    model_id: str,
    *,
    cache_dir: Optional[str] = None,
    required_files: Optional[Tuple[str, ...]] = None,
    require_weight_files: bool = False,
) -> List[str]:
    return cached_hf_model_sources(
        model_id,
        cache_dir=cache_dir,
        extra_roots=framework_hf_cache_roots(),
        required_files=required_files,
        require_weight_files=require_weight_files,
    )


def _preset_cache_requirements(*, target: str, engine: str) -> tuple[Tuple[str, ...], bool]:
    t = str(target or "").strip().lower()
    e = str(engine or "").strip().lower()
    if t in {"mlx", "gguf", "fp8", "hf-snapshot"}:
        return tuple(), True
    if t == "diffusers" or e == "diffusers":
        return ("model_index.json",), True
    if e in {"mflux", "stable-diffusion.cpp", "diffusers-component", "transformers"}:
        return tuple(), True
    return tuple(), False


def _format_incomplete_sources(sources: Sequence[str]) -> List[str]:
    out: List[str] = []
    for source in sources:
        text = str(source or "").strip()
        if text:
            out.append(f"incomplete HF cache: {text}")
    return out or ["incomplete HF cache"]


def _detected_mflux_bits(discovered_model: Any) -> Optional[int]:
    probe = " ".join(
        str(value or "")
        for value in (
            getattr(discovered_model, "repo_id", None),
            getattr(getattr(discovered_model, "snapshot_dir", None), "name", ""),
            getattr(discovered_model, "source_detail", None),
        )
    ).strip().lower().replace("_", "-")
    if "q4" in probe or "4bit" in probe:
        return 4
    if "q8" in probe or "8bit" in probe or "fp8" in probe:
        return 8
    return None


def _mflux_variant_label(display_name: str, discovered_model: Any) -> str:
    bits = _detected_mflux_bits(discovered_model)
    text = str(display_name or "")
    if bits is None or bits == 8:
        return text
    if "8-bit" in text:
        return text.replace("8-bit", "Q4" if bits == 4 else f"{bits}-bit")
    return f"{text} [{bits}-bit]"


def _serialize_task_specs(spec: Any) -> Dict[str, Dict[str, Any]]:
    tasks = getattr(spec, "tasks", None)
    if not isinstance(tasks, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for task_name, task_spec in tasks.items():
        out[str(task_name)] = {
            "inputs": list(getattr(task_spec, "inputs", []) or []),
            "outputs": list(getattr(task_spec, "outputs", []) or []),
            "params": dict(getattr(task_spec, "params", {}) or {}),
            "requires": (
                dict(getattr(task_spec, "requires", {}) or {})
                if isinstance(getattr(task_spec, "requires", None), dict)
                else None
            ),
        }
    return out


def _parse_json_bytes(body: bytes) -> Dict[str, Any]:
    if not body:
        return {}
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def _parse_extra_json(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    s = str(value or "").strip()
    if not s:
        return {}
    obj = json.loads(s)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("extra_json must decode to an object")
    return obj


def _parse_multipart(
    content_type: str, body: bytes
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    msg = BytesParser(policy=policy.default).parsebytes(header + body)
    if not msg.is_multipart():
        raise ValueError("Expected multipart/form-data request body")

    fields: Dict[str, str] = {}
    files: Dict[str, Dict[str, Any]] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_param("filename", header="content-disposition")
        if filename is not None:
            files[str(name)] = {
                "filename": str(filename),
                "content_type": part.get_content_type(),
                "content": payload,
            }
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[str(name)] = payload.decode(charset, errors="replace")
    return fields, files


def _asset_to_media_response(
    asset: GeneratedAsset, *, response_format: str = "b64_json"
) -> Dict[str, Any]:
    fmt = str(response_format or "b64_json").strip().lower()
    if fmt != "b64_json":
        raise ValueError(
            "The playground server currently supports response_format='b64_json' only."
        )
    return {
        "created": int(time.time()),
        "data": [
            {
                "b64_json": base64.b64encode(bytes(asset.data)).decode("ascii"),
                "mime_type": asset.mime_type,
                "metadata": dict(asset.metadata or {}),
            }
        ],
    }


def _asset_to_image_response(
    asset: GeneratedAsset, *, response_format: str = "b64_json"
) -> Dict[str, Any]:
    return _asset_to_media_response(asset, response_format=response_format)


def _request_kwargs(payload: Dict[str, Any], *, known: set) -> Dict[str, Any]:
    extra = _parse_extra_json(payload.get("extra"))
    for k, v in payload.items():
        if k not in known and v is not None:
            extra[k] = v
    return extra


@dataclass
class PlaygroundServerConfig:
    host: str = DEFAULT_PLAYGROUND_HOST
    port: int = DEFAULT_PLAYGROUND_PORT
    diffusers_device: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_DIFFUSERS_DEVICE", "auto") or "auto"
    )
    diffusers_torch_dtype: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE")
    )
    diffusers_allow_download: bool = field(
        default_factory=lambda: _env_bool(
            "ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD",
            _env_bool("ABSTRACTVISION_ALLOW_DOWNLOAD", False),
        )
    )
    diffusers_cache_dir: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_DIFFUSERS_CACHE_DIR")
    )
    diffusers_revision: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_DIFFUSERS_REVISION")
    )
    diffusers_variant: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_DIFFUSERS_VARIANT")
    )
    diffusers_auto_retry_fp32: bool = field(
        default_factory=lambda: _env_bool("ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32", True)
    )
    default_model_id: str = field(default_factory=lambda: _env("ABSTRACTVISION_MODEL_ID", "") or "")

    backend_kind: str = field(default_factory=_default_backend_kind)
    openai_base_url: Optional[str] = field(default_factory=lambda: _env("OPENAI_BASE_URL"))
    openai_api_key: Optional[str] = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_timeout_s: float = field(
        default_factory=lambda: float(_env("ABSTRACTVISION_TIMEOUT_S", "300") or "300")
    )
    openai_image_generations_path: str = field(
        default_factory=lambda: _env(
            "ABSTRACTVISION_IMAGES_GENERATIONS_PATH", "/images/generations"
        )
        or "/images/generations"
    )
    openai_image_edits_path: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_IMAGES_EDITS_PATH", "/images/edits")
        or "/images/edits"
    )

    sdcpp_bin: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_SDCPP_BIN", "sd-cli") or "sd-cli"
    )
    sdcpp_model: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_SDCPP_MODEL"))
    sdcpp_diffusion_model: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_SDCPP_DIFFUSION_MODEL")
    )
    sdcpp_vae: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_SDCPP_VAE"))
    sdcpp_llm: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_SDCPP_LLM"))
    sdcpp_llm_vision: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_SDCPP_LLM_VISION")
    )
    sdcpp_extra_args: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_SDCPP_EXTRA_ARGS")
    )
    mflux_model: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_MFLUX_MODEL"))
    mflux_base_model: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_MFLUX_BASE_MODEL"))
    mflux_model_dir: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_MODEL_DIR"))
    mflux_allow_download: bool = field(default_factory=lambda: _env_bool("ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD", False))


@dataclass
class _Job:
    job_id: str
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def snapshot(self, *, include_result: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "job_id": self.job_id,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.progress is not None:
            out["progress"] = dict(self.progress)
        if self.error:
            out["error"] = self.error
        if include_result and self.result is not None:
            out["result"] = self.result
        return out


class PlaygroundState:
    def __init__(self, config: PlaygroundServerConfig):
        self.config = config
        self.registry = VisionModelCapabilitiesRegistry()
        self._active_lock = threading.RLock()
        self._active_backend: Any = None
        self._active_backend_kind: Optional[str] = None
        self._active_model_id: Optional[str] = None
        self._active_loaded_at: Optional[float] = None
        self._backend_refcounts: Dict[int, int] = {}
        self._retired_backends: Dict[int, Any] = {}
        self._jobs_lock = threading.RLock()
        self._jobs: Dict[str, _Job] = {}

    def _task_specs_for_model(self, model_id: str) -> Dict[str, Dict[str, Any]]:
        try:
            spec = self.registry.get(str(model_id))
        except Exception:
            return {}
        return _serialize_task_specs(spec)

    def _surface_tasks_for_backend(
        self,
        *,
        backend: Optional[str],
        model_id: str,
        tasks: List[str],
        task_specs: Dict[str, Dict[str, Any]],
    ) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        backend_kind = str(backend or "").strip().lower()
        if backend_kind in {"huggingface", "hf", "hf-diffusers"}:
            backend_kind = "diffusers"
        allowed = set(str(task) for task in tasks)
        if backend_kind == "diffusers":
            try:
                from .backends.huggingface_diffusers import (
                    HuggingFaceDiffusersBackendConfig,
                    HuggingFaceDiffusersVisionBackend,
                )

                probe = HuggingFaceDiffusersVisionBackend(
                    config=HuggingFaceDiffusersBackendConfig(
                        model_id=str(model_id),
                        device=str(self.config.diffusers_device or "auto"),
                        torch_dtype=self.config.diffusers_torch_dtype,
                        allow_download=bool(self.config.diffusers_allow_download),
                        auto_retry_fp32=bool(self.config.diffusers_auto_retry_fp32),
                        cache_dir=self.config.diffusers_cache_dir,
                        revision=self.config.diffusers_revision,
                        variant=self.config.diffusers_variant,
                    )
                )
                allowed = {str(task) for task in probe.get_capabilities().supported_tasks or []}
            except Exception:
                pass
        elif backend_kind == "mflux":
            try:
                from .backends.mflux import MFluxBackendConfig, MFluxVisionBackend

                probe = MFluxVisionBackend(
                    config=MFluxBackendConfig(
                        model=str(model_id),
                        base_model=str(self.config.mflux_base_model) if self.config.mflux_base_model else None,
                        model_dir=str(self.config.mflux_model_dir) if self.config.mflux_model_dir else None,
                        cache_dir=str(self.config.diffusers_cache_dir) if self.config.diffusers_cache_dir else None,
                        allow_download=bool(self.config.mflux_allow_download),
                    )
                )
                probed = {str(task) for task in probe.get_capabilities().supported_tasks or []}
                allowed = allowed.intersection(probed) if allowed else probed
            except Exception:
                pass
        elif backend_kind == "sdcpp":
            try:
                from .backends.stable_diffusion_cpp import (
                    StableDiffusionCppBackendConfig,
                    StableDiffusionCppVisionBackend,
                )

                probe = StableDiffusionCppVisionBackend(
                    config=StableDiffusionCppBackendConfig(
                        sd_cli_path=str(self.config.sdcpp_bin),
                        model=str(model_id),
                        capabilities_model_id=str(model_id),
                    )
                )
                probed = {str(task) for task in probe.get_capabilities().supported_tasks or []}
                allowed = allowed.intersection(probed) if allowed else probed
            except Exception:
                pass
        filtered_tasks = [str(task) for task in tasks if str(task) in allowed]
        filtered_specs = {
            str(task_name): dict(task_spec)
            for task_name, task_spec in task_specs.items()
            if str(task_name) in allowed
        }
        return filtered_tasks, filtered_specs

    def _same_requested_model(self, requested_model_id: str) -> bool:
        current_model_id = str(self._active_model_id or "").strip()
        requested = str(requested_model_id or "").strip()
        if not current_model_id or not requested:
            return False
        try:
            current_kind, current_backend_model = normalize_model_id_for_backend(current_model_id)
            requested_kind, requested_backend_model = normalize_model_id_for_backend(requested)
        except Exception:
            return current_model_id == requested
        return (current_kind, current_backend_model) == (requested_kind, requested_backend_model)

    def ensure_model_loaded(self, requested_model_id: Optional[str]) -> Dict[str, Any]:
        requested = str(requested_model_id or "").strip()
        if not requested:
            active = self.active_snapshot()
            return {"ok": bool(active), "active": active}
        with self._active_lock:
            if self._same_requested_model(requested):
                return {"ok": True, "active": self.active_snapshot()}
        return self.load_model(requested)

    def active_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._active_lock:
            if not self._active_backend or not self._active_model_id:
                return None
            return {
                "model_id": self._active_model_id,
                "load_id": self._active_model_id,
                "backend": self._active_backend_kind,
                "loaded_at": self._active_loaded_at,
            }

    def list_models(self) -> Dict[str, Any]:
        models: List[Dict[str, Any]] = []
        allow_diffusers_download = bool(self.config.diffusers_allow_download)
        allow_mflux_download = bool(self.config.mflux_allow_download)
        diffusers_runtime_available = _local_runtime_available("diffusers")
        mflux_runtime_available = _local_runtime_available("mflux")
        sdcpp_runtime_available = _local_runtime_available("sdcpp")
        platform_profile = local_model_profile()
        visible_targets = catalog_target_scope(target="auto", engine=None, include_all_targets=False)
        configured = str(self.config.default_model_id or "").strip()
        configured_backend = str(self.config.backend_kind or "").strip().lower()
        if configured_backend in {"openai-compatible", "openai_compatible", "proxy"}:
            configured_backend = "openai"
        elif configured_backend in {"huggingface", "hf", "hf-diffusers"}:
            configured_backend = "diffusers"
        elif configured_backend in {
            "sd-cpp",
            "stable-diffusion.cpp",
            "stable_diffusion_cpp",
            "stable-diffusion-cpp",
        }:
            configured_backend = "sdcpp"
        if not configured and configured_backend == "mflux":
            configured = str(self.config.mflux_model or "").strip()

        seen_load_ids: set[str] = set()
        legacy_root = default_legacy_model_root()
        playground_tasks = {"text_to_image", "image_to_image", "text_to_video", "image_to_video"}
        mflux_cached: Dict[str, Any] = {}
        mflux_invalid: Dict[str, Tuple[str, ...]] = {}
        if "mlx" in visible_targets:
            try:
                from .backends.mflux import (
                    discover_cached_mflux_models,
                    discover_incomplete_mflux_sources,
                )

                mflux_cached = discover_cached_mflux_models(
                    model_dir=str(self.config.mflux_model_dir) if self.config.mflux_model_dir else None,
                    cache_dir=self.config.diffusers_cache_dir,
                )
                mflux_invalid = discover_incomplete_mflux_sources(
                    model_dir=str(self.config.mflux_model_dir) if self.config.mflux_model_dir else None,
                    cache_dir=self.config.diffusers_cache_dir,
                )
            except Exception:
                mflux_cached = {}
                mflux_invalid = {}

        for preset in model_presets(
            target="auto",
            engine=None,
            include_non_8bit=True,
            include_all_targets=False,
        ):
            model_id = str(preset.upstream_repo_id or preset.repo_id)
            try:
                spec = self.registry.get(model_id)
                tasks = sorted(spec.tasks.keys())
                provider = spec.provider
                task_specs = _serialize_task_specs(spec)
            except Exception:
                tasks = ["text_to_image", "image_to_image"]
                provider = "huggingface"
                task_specs = {}

            backend: Optional[str] = None
            load_id: Optional[str] = None
            download_enabled = False
            runtime_available = True
            if preset.engine == "diffusers" and preset.target in {"diffusers", "gguf"}:
                backend = "diffusers"
                load_id = model_id if preset.target == "diffusers" else f"diffusers/{preset.key}"
                runtime_available = diffusers_runtime_available
                download_enabled = runtime_available and allow_diffusers_download
            elif preset.engine == "stable-diffusion.cpp" and preset.target == "gguf":
                backend = "sdcpp"
                load_id = f"sdcpp/{preset.key}"
                runtime_available = sdcpp_runtime_available
                download_enabled = False
            elif preset.engine == "mflux" and preset.target == "mlx":
                backend = "mflux"
                load_id = f"mflux/{preset.key}"
                runtime_available = mflux_runtime_available
                download_enabled = runtime_available and "mlx" in visible_targets and allow_mflux_download
            tasks, task_specs = self._surface_tasks_for_backend(
                backend=backend or preset.engine,
                model_id=model_id,
                tasks=[str(task) for task in tasks],
                task_specs=task_specs,
            )
            if not playground_tasks.intersection(tasks):
                continue

            required_files, require_weight_files = _preset_cache_requirements(
                target=preset.target,
                engine=preset.engine,
            )
            cached_in: List[str]
            invalid_cached_in: Sequence[str]
            fully_cached = False
            variant_label = preset.display_name
            bits = preset.quantization_bits
            if preset.engine == "mflux" and preset.target == "mlx":
                discovered_model = mflux_cached.get(preset.key)
                cached_in = [str(discovered_model.source_detail)] if discovered_model is not None else []
                invalid_cached_in = mflux_invalid.get(preset.key, ())
                fully_cached = bool(cached_in)
                if discovered_model is not None:
                    variant_label = _mflux_variant_label(preset.display_name, discovered_model)
                    bits = _detected_mflux_bits(discovered_model) or bits
            else:
                cached_in = _cached_hf_model_sources(
                    preset.repo_id,
                    cache_dir=self.config.diffusers_cache_dir,
                    required_files=required_files,
                    require_weight_files=require_weight_files,
                )
                base_cached_in: List[str] = []
                if preset.engine == "diffusers" and preset.target == "gguf":
                    base_cached_in = _cached_hf_model_sources(
                        model_id,
                        cache_dir=self.config.diffusers_cache_dir,
                        required_files=("model_index.json",),
                        require_weight_files=True,
                    )
                fully_cached = bool(cached_in) and (
                    preset.engine != "diffusers" or preset.target != "gguf" or bool(base_cached_in)
                )
                invalid_cached_in = incomplete_hf_model_sources(
                    preset.repo_id,
                    cache_dir=self.config.diffusers_cache_dir,
                    extra_roots=framework_hf_cache_roots(),
                    required_files=required_files,
                    require_weight_files=require_weight_files,
                )
                if base_cached_in:
                    cached_in = [*cached_in, *[f"base Diffusers snapshot: {source}" for source in base_cached_in]]
            legacy_dir = legacy_root / preset.local_dir_name
            try:
                if not cached_in and legacy_dir.exists():
                    cached_in = ["legacy model dir"]
                    fully_cached = True
            except Exception:
                pass

            variant_id = str(load_id or f"{preset.engine}:{preset.repo_id}")
            if variant_id in seen_load_ids:
                continue
            seen_load_ids.add(variant_id)

            display_sources = (
                list(cached_in)
                if cached_in
                else (
                    list(invalid_cached_in)
                    if (preset.engine == "mflux" and invalid_cached_in)
                    else (
                        _format_incomplete_sources(invalid_cached_in)
                        if invalid_cached_in
                        else (["download enabled"] if download_enabled else ["download only"])
                    )
                )
            )
            if not runtime_available:
                runtime_text = f"{backend or preset.engine} runtime missing"
                if runtime_text not in display_sources:
                    display_sources.append(runtime_text)
            if (
                preset.engine == "diffusers"
                and preset.target == "gguf"
                and cached_in
                and not fully_cached
                and not download_enabled
                and "base Diffusers snapshot missing" not in display_sources
            ):
                display_sources.append("base Diffusers snapshot missing")

            loadable = bool(load_id) and runtime_available and (fully_cached or download_enabled)
            models.append(
                {
                    "id": model_id,
                    "load_id": load_id,
                    "provider": provider,
                    "backend": backend or preset.engine,
                    "engine": preset.engine,
                    "target": preset.target,
                    "bits": bits,
                    "variant": variant_label,
                    "download_repo_id": preset.repo_id,
                    "tasks": tasks,
                    "task_specs": task_specs,
                    "cached": fully_cached,
                    "cached_in": display_sources,
                    "loadable": loadable,
                }
            )

        if not configured and configured_backend == "openai" and self.config.openai_base_url:
            configured = "openai-compatible/default"

        existing_ids = {
            str(m.get("id") or "")
            for m in models
        }.union({str(m.get("load_id") or "") for m in models})
        if configured and configured not in existing_ids:
            prefix, rest = _known_prefix(configured)
            if prefix in {"openai", "openai-compatible", "openai_compatible"}:
                label = configured
                load_id = configured
                backend = "openai"
                cached_in = ["configured remote"]
            elif configured_backend == "openai":
                rest = configured
                label = f"openai-compatible/{rest}" if rest else "openai-compatible/default"
                load_id = label
                backend = "openai"
                cached_in = ["configured remote"]
            elif configured_backend == "mflux":
                label = configured
                load_id = configured if prefix == "mflux" else f"mflux/{configured}"
                backend = "mflux"
                cached_in = ["configured local preset"]
            else:
                label = configured
                load_id = configured
                backend = "diffusers"
                cached_in = _cached_hf_model_sources(
                    configured,
                    cache_dir=self.config.diffusers_cache_dir,
                    required_files=("model_index.json",),
                    require_weight_files=True,
                )
                if not cached_in and allow_diffusers_download:
                    cached_in = ["download enabled"]
            if cached_in:
                task_specs = self._task_specs_for_model(label)
                tasks = sorted(task_specs.keys()) if task_specs else ["text_to_image", "image_to_image"]
                tasks, task_specs = self._surface_tasks_for_backend(
                    backend=backend,
                    model_id=label,
                    tasks=[str(task) for task in tasks],
                    task_specs=task_specs,
                )
                if playground_tasks.intersection(tasks):
                    runtime_available = _local_runtime_available(backend)
                    display_sources = list(cached_in)
                    if not runtime_available:
                        runtime_text = f"{backend} runtime missing"
                        if runtime_text not in display_sources:
                            display_sources.append(runtime_text)
                    models.append(
                        {
                            "id": label,
                            "load_id": load_id,
                            "provider": "configured",
                            "backend": backend,
                            "engine": backend,
                            "target": "configured",
                            "bits": None,
                            "variant": label,
                            "download_repo_id": None,
                            "tasks": tasks,
                            "task_specs": task_specs,
                            "cached": "cache" in ",".join(cached_in).lower(),
                            "cached_in": display_sources,
                            "loadable": runtime_available,
                        }
                    )

        if configured_backend == "sdcpp":
            models.append(
                {
                    "id": "sdcpp/default",
                    "load_id": "sdcpp/default",
                    "provider": "configured",
                    "backend": "sdcpp",
                    "engine": "sdcpp",
                    "target": "configured",
                    "bits": None,
                    "variant": "sdcpp/default",
                    "download_repo_id": None,
                    "tasks": ["text_to_image", "image_to_image"],
                    "task_specs": {},
                    "cached": True,
                    "cached_in": ["configured local files"]
                    + ([] if sdcpp_runtime_available else ["sdcpp runtime missing"]),
                    "loadable": sdcpp_runtime_available,
                }
            )

        models.sort(
            key=lambda item: (
                0 if item.get("cached") else 1,
                0 if item.get("loadable") else 1,
                str(item.get("backend") or ""),
                str(item.get("id") or ""),
                str(item.get("variant") or ""),
            )
        )
        return {
            "models": models,
            "active": self.active_snapshot(),
            "platform": platform_profile,
            "targets": list(visible_targets),
        }

    def unload_active(self) -> Dict[str, Any]:
        unload_after_lock: Optional[Any] = None
        with self._active_lock:
            backend = self._active_backend
            self._active_backend = None
            self._active_backend_kind = None
            self._active_model_id = None
            self._active_loaded_at = None
            unload_after_lock = self._retire_backend_locked(backend)
        self._unload_backend(unload_after_lock)
        return {"ok": True, "active": None}

    def load_model(self, requested_model_id: str, *, unload_first: bool = False) -> Dict[str, Any]:
        requested = str(requested_model_id or "").strip()
        if not requested:
            raise ValueError("Missing required field: model_id")
        with self._active_lock:
            if self._same_requested_model(requested):
                return {"ok": True, "active": self.active_snapshot()}

        if unload_first:
            unload_before_load: Optional[Any] = None
            with self._active_lock:
                old = self._active_backend
                self._active_backend = None
                self._active_backend_kind = None
                self._active_model_id = None
                self._active_loaded_at = None
                unload_before_load = self._retire_backend_locked(old)
            self._unload_backend(unload_before_load)

        backend_kind, backend_model_id = normalize_model_id_for_backend(requested)

        backend = self._build_backend(backend_kind, backend_model_id)
        preload = getattr(backend, "preload", None)
        try:
            if callable(preload):
                preload()
        except Exception:
            self._unload_backend(backend)
            raise

        unload_after_lock: Optional[Any] = None
        backend_to_unload: Optional[Any] = None
        with self._active_lock:
            if self._same_requested_model(requested):
                backend_to_unload = backend
                out = {"ok": True, "active": self.active_snapshot()}
            else:
                old = self._active_backend
                self._active_backend = backend
                self._active_backend_kind = backend_kind
                self._active_model_id = requested
                self._active_loaded_at = time.time()
                unload_after_lock = self._retire_backend_locked(old)
                out = {"ok": True, "active": self.active_snapshot()}
        self._unload_backend(backend_to_unload)
        self._unload_backend(unload_after_lock)
        return out

    def _unload_backend(self, backend: Optional[Any]) -> None:
        if backend is None:
            return
        unload = getattr(backend, "unload", None)
        if callable(unload):
            unload()

    def _retire_backend_locked(self, backend: Optional[Any]) -> Optional[Any]:
        if backend is None:
            return None
        key = id(backend)
        if self._backend_refcounts.get(key, 0) > 0:
            self._retired_backends[key] = backend
            return None
        self._retired_backends.pop(key, None)
        return backend

    def _acquire_active_backend_snapshot(self) -> Tuple[Any, str, Optional[str]]:
        with self._active_lock:
            backend, model_id = self._active_backend_or_raise()
            key = id(backend)
            self._backend_refcounts[key] = self._backend_refcounts.get(key, 0) + 1
            return backend, model_id, self._active_backend_kind

    def _release_backend_snapshot(self, backend: Any) -> None:
        unload_after_lock: Optional[Any] = None
        with self._active_lock:
            key = id(backend)
            count = self._backend_refcounts.get(key, 0) - 1
            if count > 0:
                self._backend_refcounts[key] = count
                return
            self._backend_refcounts.pop(key, None)
            if self._active_backend is not backend:
                unload_after_lock = self._retired_backends.pop(key, None)
        self._unload_backend(unload_after_lock)

    def _build_backend(self, backend_kind: str, backend_model_id: Optional[str]) -> Any:
        if backend_kind in {"openai-compatible", "openai_compatible", "proxy"}:
            backend_kind = "openai"
        elif backend_kind in {"huggingface", "hf", "hf-diffusers"}:
            backend_kind = "diffusers"
        elif backend_kind in {
            "sd-cpp",
            "stable-diffusion.cpp",
            "stable_diffusion_cpp",
            "stable-diffusion-cpp",
        }:
            backend_kind = "sdcpp"
        elif backend_kind == "m-flux":
            backend_kind = "mflux"

        if not _local_runtime_available(backend_kind):
            raise ValueError(
                f"The local {backend_kind} runtime is not available in this Python environment "
                f"({sys.executable})."
            )

        if backend_kind == "diffusers":
            from .backends.huggingface_diffusers import (
                HuggingFaceDiffusersBackendConfig,
                HuggingFaceDiffusersVisionBackend,
            )

            model_id = (
                backend_model_id or self.config.default_model_id or DEFAULT_DIFFUSERS_MODEL_ID
            )
            cfg = HuggingFaceDiffusersBackendConfig(
                model_id=str(model_id),
                device=str(self.config.diffusers_device or "auto"),
                torch_dtype=self.config.diffusers_torch_dtype,
                allow_download=bool(self.config.diffusers_allow_download),
                auto_retry_fp32=bool(self.config.diffusers_auto_retry_fp32),
                cache_dir=self.config.diffusers_cache_dir,
                revision=self.config.diffusers_revision,
                variant=self.config.diffusers_variant,
            )
            return HuggingFaceDiffusersVisionBackend(config=cfg)

        if backend_kind == "sdcpp":
            from .backends.stable_diffusion_cpp import (
                StableDiffusionCppBackendConfig,
                StableDiffusionCppVisionBackend,
            )

            explicit_model = str(backend_model_id).strip() if backend_model_id else None
            explicit_diffusion_model = None if explicit_model else self.config.sdcpp_diffusion_model
            explicit_vae = None if explicit_model else self.config.sdcpp_vae
            explicit_llm = None if explicit_model else self.config.sdcpp_llm
            explicit_llm_vision = None if explicit_model else self.config.sdcpp_llm_vision
            resolved_sdcpp = None
            if explicit_model and not any((explicit_diffusion_model, explicit_vae, explicit_llm, explicit_llm_vision)):
                candidate_path = Path(str(explicit_model)).expanduser()
                if not candidate_path.exists():
                    try:
                        resolved_sdcpp = resolve_sdcpp_model_selection(str(explicit_model), allow_download=False)
                    except ValueError:
                        resolved_sdcpp = None
            cfg = StableDiffusionCppBackendConfig(
                sd_cli_path=str(self.config.sdcpp_bin),
                model=resolved_sdcpp.model if resolved_sdcpp is not None else (explicit_model or self.config.sdcpp_model),
                capabilities_model_id=(
                    resolved_sdcpp.capabilities_model_id if resolved_sdcpp is not None else explicit_model
                ),
                diffusion_model=(
                    resolved_sdcpp.diffusion_model if resolved_sdcpp is not None else explicit_diffusion_model
                ),
                vae=resolved_sdcpp.vae if resolved_sdcpp is not None else explicit_vae,
                llm=resolved_sdcpp.llm if resolved_sdcpp is not None else explicit_llm,
                llm_vision=resolved_sdcpp.llm_vision if resolved_sdcpp is not None else explicit_llm_vision,
                extra_args=(
                    shlex.split(str(self.config.sdcpp_extra_args))
                    if self.config.sdcpp_extra_args
                    else ()
                ),
            )
            return StableDiffusionCppVisionBackend(config=cfg)

        if backend_kind == "mflux":
            from .backends.mflux import MFluxBackendConfig, MFluxVisionBackend

            cfg = MFluxBackendConfig(
                model=str(backend_model_id or self.config.mflux_model or "") or None,
                base_model=str(self.config.mflux_base_model) if self.config.mflux_base_model else None,
                model_dir=str(self.config.mflux_model_dir) if self.config.mflux_model_dir else None,
                cache_dir=str(self.config.diffusers_cache_dir) if self.config.diffusers_cache_dir else None,
                allow_download=bool(self.config.mflux_allow_download),
            )
            return MFluxVisionBackend(config=cfg)

        if backend_kind == "openai":
            from .backends.openai_compatible import (
                OpenAICompatibleBackendConfig,
                OpenAICompatibleVisionBackend,
            )

            if not self.config.openai_base_url:
                raise ValueError(
                    "OpenAI-compatible backend is not configured. Set OPENAI_BASE_URL, "
                    "or select a local Diffusers/sdcpp model."
                )
            cfg = OpenAICompatibleBackendConfig(
                base_url=str(self.config.openai_base_url),
                api_key=str(self.config.openai_api_key) if self.config.openai_api_key else None,
                model_id=str(backend_model_id) if backend_model_id else None,
                timeout_s=float(self.config.openai_timeout_s),
                image_generations_path=str(self.config.openai_image_generations_path),
                image_edits_path=str(self.config.openai_image_edits_path),
            )
            return OpenAICompatibleVisionBackend(config=cfg)

        raise ValueError(f"Unknown backend kind: {backend_kind!r}")

    def _active_backend_or_raise(self) -> Tuple[Any, str]:
        backend = self._active_backend
        model_id = self._active_model_id
        if backend is None or not model_id:
            raise ValueError("No active model. Select and load a model first.")
        return backend, model_id

    def _require_backend_task_support(self, backend: Any, task: str) -> None:
        try:
            caps = backend.get_capabilities()
        except Exception:
            return
        supported = getattr(caps, "supported_tasks", None) if caps is not None else None
        if supported is not None and str(task) not in {str(item) for item in supported}:
            raise ValueError(f"The selected model does not support {task}.")

    def start_image_generation_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Missing required field: prompt")
        requested_model_id = str(payload.get("model") or "").strip() or None
        if requested_model_id is None and self.active_snapshot() is None:
            raise ValueError("No active model. Select and load a model first.")
        known = {
            "prompt",
            "model",
            "response_format",
            "negative_prompt",
            "width",
            "height",
            "steps",
            "guidance_scale",
            "seed",
            "extra",
        }
        request = ImageGenerationRequest(
            prompt=prompt,
            negative_prompt=(
                str(payload.get("negative_prompt")) if payload.get("negative_prompt") else None
            ),
            width=_to_int(payload.get("width")),
            height=_to_int(payload.get("height")),
            steps=_to_int(payload.get("steps")),
            guidance_scale=_to_float(payload.get("guidance_scale")),
            seed=_to_int(payload.get("seed")),
            extra=_request_kwargs(payload, known=known),
        )
        response_format = str(payload.get("response_format") or "b64_json")
        backend_snapshot: Optional[Any] = None
        if requested_model_id is None:
            backend_snapshot, _model_id, _backend_kind = self._acquire_active_backend_snapshot()

        def run(progress_callback: Callable[[int, Optional[int]], None]) -> Dict[str, Any]:
            backend = backend_snapshot
            if requested_model_id is not None:
                self.ensure_model_loaded(requested_model_id)
                backend, _model_id, _backend_kind = self._acquire_active_backend_snapshot()
            if backend is None:
                raise ValueError("No active model. Select and load a model first.")
            try:
                self._require_backend_task_support(backend, "text_to_image")
                normalized_request = request
                normalize = getattr(backend, "normalize_image_generation_request", None)
                if callable(normalize):
                    normalized_request = normalize(request)
                progress_callback(0, normalized_request.steps)
                asset = backend.generate_image_with_progress(
                    normalized_request, progress_callback=progress_callback
                )
                return _asset_to_media_response(asset, response_format=response_format)
            finally:
                if requested_model_id is not None:
                    self._release_backend_snapshot(backend)

        return self._start_job(run, total_steps=request.steps)

    def start_video_generation_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Missing required field: prompt")
        requested_model_id = str(payload.get("model") or "").strip() or None
        if requested_model_id is None and self.active_snapshot() is None:
            raise ValueError("No active model. Select and load a model first.")
        known = {
            "prompt",
            "model",
            "response_format",
            "negative_prompt",
            "width",
            "height",
            "fps",
            "num_frames",
            "steps",
            "guidance_scale",
            "seed",
            "extra",
        }
        request = VideoGenerationRequest(
            prompt=prompt,
            negative_prompt=(
                str(payload.get("negative_prompt")) if payload.get("negative_prompt") else None
            ),
            width=_to_int(payload.get("width")),
            height=_to_int(payload.get("height")),
            fps=_to_int(payload.get("fps")),
            num_frames=_to_int(payload.get("num_frames")),
            steps=_to_int(payload.get("steps")),
            guidance_scale=_to_float(payload.get("guidance_scale")),
            seed=_to_int(payload.get("seed")),
            extra=_request_kwargs(payload, known=known),
        )
        response_format = str(payload.get("response_format") or "b64_json")
        backend_snapshot: Optional[Any] = None
        if requested_model_id is None:
            backend_snapshot, _model_id, _backend_kind = self._acquire_active_backend_snapshot()

        def run(progress_callback: Callable[[int, Optional[int]], None]) -> Dict[str, Any]:
            backend = backend_snapshot
            if requested_model_id is not None:
                self.ensure_model_loaded(requested_model_id)
                backend, _model_id, _backend_kind = self._acquire_active_backend_snapshot()
            if backend is None:
                raise ValueError("No active model. Select and load a model first.")
            try:
                self._require_backend_task_support(backend, "text_to_video")
                normalized_request = request
                normalize = getattr(backend, "normalize_video_generation_request", None)
                if callable(normalize):
                    normalized_request = normalize(request)
                progress_callback(0, normalized_request.steps)
                generate = getattr(backend, "generate_video_with_progress", None)
                if callable(generate):
                    asset = generate(normalized_request, progress_callback=progress_callback)
                else:
                    asset = backend.generate_video(normalized_request)
                return _asset_to_media_response(asset, response_format=response_format)
            finally:
                if requested_model_id is not None:
                    self._release_backend_snapshot(backend)

        return self._start_job(run, total_steps=request.steps)

    def start_image_edit_job(
        self,
        fields: Dict[str, str],
        files: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = str(fields.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Missing required field: prompt")
        requested_model_id = str(fields.get("model") or "").strip() or None
        if requested_model_id is None and self.active_snapshot() is None:
            raise ValueError("No active model. Select and load a model first.")
        image = files.get("image")
        if not image:
            raise ValueError("Missing required multipart file: image")
        mask = files.get("mask")
        extra = _parse_extra_json(fields.get("extra_json"))
        request = ImageEditRequest(
            prompt=prompt,
            image=bytes(image.get("content") or b""),
            mask=bytes(mask.get("content") or b"") if mask else None,
            negative_prompt=(
                str(fields.get("negative_prompt")) if fields.get("negative_prompt") else None
            ),
            steps=_to_int(fields.get("steps")),
            guidance_scale=_to_float(fields.get("guidance_scale")),
            seed=_to_int(fields.get("seed")),
            extra=extra,
        )
        backend_snapshot: Optional[Any] = None
        if requested_model_id is None:
            backend_snapshot, _model_id, _backend_kind = self._acquire_active_backend_snapshot()

        def run(progress_callback: Callable[[int, Optional[int]], None]) -> Dict[str, Any]:
            backend = backend_snapshot
            if requested_model_id is not None:
                self.ensure_model_loaded(requested_model_id)
                backend, _model_id, _backend_kind = self._acquire_active_backend_snapshot()
            if backend is None:
                raise ValueError("No active model. Select and load a model first.")
            try:
                self._require_backend_task_support(backend, "image_to_image")
                normalized_request = request
                normalize = getattr(backend, "normalize_image_edit_request", None)
                if callable(normalize):
                    normalized_request = normalize(request)
                progress_callback(0, normalized_request.steps)
                asset = backend.edit_image_with_progress(
                    normalized_request, progress_callback=progress_callback
                )
                return _asset_to_media_response(asset, response_format="b64_json")
            finally:
                if requested_model_id is not None:
                    self._release_backend_snapshot(backend)

        return self._start_job(run, total_steps=request.steps)

    def _start_job(
        self,
        fn: Callable[[Callable[[int, Optional[int]], None]], Dict[str, Any]],
        *,
        total_steps: Optional[int],
    ) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        job = _Job(
            job_id=job_id,
            state="queued",
            progress={"step": 0, "total_steps": total_steps} if total_steps is not None else None,
        )
        with self._jobs_lock:
            self._jobs[job_id] = job

        def progress(step: int, total: Optional[int]) -> None:
            with self._jobs_lock:
                current = self._jobs.get(job_id)
                if current is None:
                    return
                current.progress = {"step": int(step), "total_steps": total}
                current.updated_at = time.time()

        def runner() -> None:
            with self._jobs_lock:
                current = self._jobs.get(job_id)
                if current is not None:
                    current.state = "running"
                    current.updated_at = time.time()
            try:
                result = fn(progress)
                with self._jobs_lock:
                    current = self._jobs.get(job_id)
                    if current is not None:
                        current.state = "succeeded"
                        current.result = result
                        current.updated_at = time.time()
            except Exception as e:
                with self._jobs_lock:
                    current = self._jobs.get(job_id)
                    if current is not None:
                        current.state = "failed"
                        current.error = _redact(str(e))
                        current.updated_at = time.time()

        threading.Thread(
            target=runner, name=f"abstractvision-job-{job_id[:8]}", daemon=True
        ).start()
        return job.snapshot(include_result=False)

    def get_job(self, job_id: str, *, consume: bool = False) -> Dict[str, Any]:
        with self._jobs_lock:
            job = self._jobs.get(str(job_id))
            if job is None:
                raise KeyError(str(job_id))
            snap = job.snapshot(include_result=True)
            if consume and job.state in {"succeeded", "failed"}:
                self._jobs.pop(str(job_id), None)
            return snap


def _playground_html_bytes() -> bytes:
    try:
        return (
            resources.files("abstractvision.playground")
            .joinpath("vision_playground.html")
            .read_bytes()
        )
    except Exception:
        pass

    # Repo checkout fallback for editable installs with unusual package-data handling.
    p = Path(__file__).resolve().parents[2] / "playground" / "vision_playground.html"
    if p.is_file():
        return p.read_bytes()
    raise FileNotFoundError(
        "playground/vision_playground.html was not found. Run this command from an AbstractVision checkout."
    )


def _make_handler(state: PlaygroundState) -> type[BaseHTTPRequestHandler]:
    html_cache = _playground_html_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "AbstractVisionPlayground/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

        def _send_json(self, status: int, obj: Dict[str, Any]) -> None:
            body = json.dumps(obj, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, status: int, message: str) -> None:
            self._send_json(status, {"error": {"message": _redact(message), "type": "http_error"}})

        def _send_html(self) -> None:
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(html_cache)))
            self.end_headers()
            self.wfile.write(html_cache)

        def _read_body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or "0")
            return self.rfile.read(n) if n > 0 else b""

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                if path in {"/", "/vision_playground.html", "/playground/vision_playground.html"}:
                    self._send_html()
                    return
                if path == "/v1/models":
                    self._send_json(
                        200,
                        {
                            "object": "list",
                            "data": [{"id": "abstractvision/playground", "object": "model"}],
                        },
                    )
                    return
                if path == "/v1/vision/models":
                    self._send_json(200, state.list_models())
                    return
                prefix = "/v1/vision/jobs/"
                if path.startswith(prefix):
                    job_id = path[len(prefix) :].strip("/")
                    if not job_id:
                        self._send_error_json(404, "Missing job id")
                        return
                    consume = (parse_qs(parsed.query).get("consume") or ["0"])[0] in {
                        "1",
                        "true",
                        "yes",
                    }
                    try:
                        self._send_json(200, state.get_job(job_id, consume=consume))
                    except KeyError:
                        self._send_error_json(404, f"Unknown job id: {job_id}")
                    return
                self._send_error_json(404, f"Not found: {path}")
            except Exception as e:
                traceback.print_exc()
                self._send_error_json(500, str(e))

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                body = self._read_body()
                if path == "/v1/vision/model/load":
                    payload = _parse_json_bytes(body)
                    model_id = str(payload.get("model_id") or payload.get("model") or "").strip()
                    unload_first = bool(payload.get("unload_first"))
                    self._send_json(200, state.load_model(model_id, unload_first=unload_first))
                    return
                if path == "/v1/vision/model/unload":
                    self._send_json(200, state.unload_active())
                    return
                if path == "/v1/vision/jobs/images/generations":
                    payload = _parse_json_bytes(body)
                    self._send_json(200, state.start_image_generation_job(payload))
                    return
                if path == "/v1/vision/jobs/videos/generations":
                    payload = _parse_json_bytes(body)
                    self._send_json(200, state.start_video_generation_job(payload))
                    return
                if path == "/v1/vision/jobs/images/edits":
                    content_type = self.headers.get("Content-Type") or ""
                    fields, files = _parse_multipart(content_type, body)
                    self._send_json(200, state.start_image_edit_job(fields, files))
                    return
                self._send_error_json(404, f"Not found: {path}")
            except json.JSONDecodeError as e:
                self._send_error_json(400, f"Invalid JSON: {e}")
            except AbstractVisionError as e:
                self._send_error_json(400, str(e))
            except ValueError as e:
                self._send_error_json(400, str(e))
            except Exception as e:
                traceback.print_exc()
                self._send_error_json(500, str(e))

    return Handler


def run_playground_server(config: PlaygroundServerConfig) -> int:
    state = PlaygroundState(config)
    handler_cls = _make_handler(state)
    server = ThreadingHTTPServer((config.host, int(config.port)), handler_cls)
    url = f"http://{config.host}:{int(config.port)}/vision_playground.html"
    print("AbstractVision playground")
    print(f"- UI:  {url}")
    print(f"- API: http://{config.host}:{int(config.port)}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0
