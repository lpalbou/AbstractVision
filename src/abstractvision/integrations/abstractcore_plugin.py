from __future__ import annotations

import importlib.util
import ipaddress
import os
import shlex
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union
from urllib.parse import urlparse

from ..artifacts import RuntimeArtifactStoreAdapter, get_artifact_id, is_artifact_ref
from ..backends.base_backend import VisionBackend
from ..errors import AbstractVisionError
from ..types import ProviderModelInfo
from ..vision_manager import VisionManager

_DEFAULT_LOCAL_DIFFUSERS_MODEL_ID = "runwayml/stable-diffusion-v1-5"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_OPENAI_IMAGE_MODEL_ID = "gpt-image-1"
_PROVIDER_RAW_MAX_DEPTH = 4
_PROVIDER_RAW_MAX_ITEMS = 50
_PROVIDER_RAW_MAX_STRING = 4096
_OPENAI_COMPATIBLE_BACKEND_KINDS = {
    "openai_compatible",
    "openai-compatible",
    "proxy",
}
_LOCAL_RESIDENCY_BACKEND_KINDS = {"diffusers", "mflux", "sdcpp"}
_RESIDENCY_TASK_ALIASES = {
    "text_to_image": "text_to_image",
    "text-to-image": "text_to_image",
    "t2i": "text_to_image",
    "image_generation": "text_to_image",
    "image-generation": "text_to_image",
    "image_to_image": "image_to_image",
    "image-to-image": "image_to_image",
    "i2i": "image_to_image",
    "image_edit": "image_to_image",
    "image-edit": "image_to_image",
    "text_to_video": "text_to_video",
    "text-to-video": "text_to_video",
    "t2v": "text_to_video",
    "video_generation": "text_to_video",
    "video-generation": "text_to_video",
    "image_to_video": "image_to_video",
    "image-to-video": "image_to_video",
    "i2v": "image_to_video",
}


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(str(key), None)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _env_first(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        v = _env(str(key))
        if v is not None:
            return v
    return default


def _strip_openai_model_prefixes(value: Any) -> str:
    model = str(value or "").strip()
    while "/" in model:
        head, tail = model.split("/", 1)
        if head.strip().lower().replace("_", "-") in {"openai", "openai-compatible"}:
            model = tail.strip()
            continue
        break
    return model


def _env_bool(key: str, default: bool = False) -> bool:
    v = _env(key)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _owner_cfg(owner: Any, key: str) -> Optional[str]:
    try:
        cfg = getattr(owner, "config", None)
        if isinstance(cfg, dict):
            v = cfg.get(key)
            if v is None:
                return None
            s = str(v).strip()
            return s if s else None
    except Exception:
        return None
    return None


def _owner_cfg_bool(owner: Any, key: str, default: bool = False) -> bool:
    v = _owner_cfg(owner, key)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_openai_api(base_url: Optional[str]) -> bool:
    return "api.openai.com" in str(base_url or "").lower()


def _openai_api_key() -> Optional[str]:
    return _env("OPENAI_API_KEY")


def _internet_override_flags() -> tuple[bool, bool]:
    return (
        _env_bool("ABSTRACTVISION_ASSUME_OFFLINE", False)
        or _env_bool("ABSTRACTCORE_ASSUME_OFFLINE", False),
        _env_bool("ABSTRACTVISION_ASSUME_ONLINE", False)
        or _env_bool("ABSTRACTCORE_ASSUME_ONLINE", False),
    )


def _host_is_local(host: Optional[str]) -> bool:
    host_s = str(host or "").strip().lower()
    if not host_s:
        return False
    if host_s in {"localhost", "127.0.0.1", "::1"} or host_s.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host_s)
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local)


def _catalog_host_port(base_url: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    url = str(base_url or "").strip()
    if not url:
        url = _DEFAULT_OPENAI_BASE_URL
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return None, None
    port = parsed.port
    if port is None:
        port = 443 if str(parsed.scheme or "").lower() == "https" else 80
    return host, int(port)


def _catalog_endpoint_reachable(base_url: Optional[str]) -> bool:
    host, port = _catalog_host_port(base_url)
    if not host or port is None:
        return False
    if _host_is_local(host):
        return True

    assume_offline, assume_online = _internet_override_flags()
    if assume_offline:
        return False
    if assume_online:
        return True
    try:
        with socket.create_connection((host, int(port)), timeout=0.5):
            return True
    except Exception:
        return False


def _remote_provider_catalog_enabled(
    provider_id: Optional[str],
    *,
    base_url: Optional[str],
    api_key: Optional[str],
) -> bool:
    provider = str(provider_id or "").strip().lower().replace("_", "-")
    if provider == "openai":
        if not str(api_key or "").strip():
            return False
        return _catalog_endpoint_reachable(base_url or _DEFAULT_OPENAI_BASE_URL)
    if provider == "openai-compatible":
        if not str(base_url or "").strip():
            return False
        return _catalog_endpoint_reachable(base_url)
    return True


def _backend_catalog_enabled(owner: Any, provider_id: Optional[str], backend: Any) -> bool:
    provider = str(provider_id or "").strip().lower().replace("_", "-")
    if provider not in {"openai", "openai-compatible"}:
        return True
    cfg = getattr(backend, "_cfg", None)
    base_url = getattr(cfg, "base_url", None)
    api_key = getattr(cfg, "api_key", None)
    if api_key is None:
        api_key = _owner_cfg(owner, "vision_api_key") or _openai_api_key()
    return _remote_provider_catalog_enabled(provider, base_url=base_url, api_key=api_key)

def _mflux_weights_present() -> bool:
    """Return True when MFLUX preset weights appear to be downloaded locally."""
    if sys.platform != "darwin":
        return False
    try:
        from ..model_cache import (
            default_hf_cache_root,
            default_legacy_model_root,
            framework_hf_cache_roots,
            resolve_hf_repo_snapshot,
        )
        from ..model_downloads import model_presets
    except Exception:
        return False
    cache_root = str(default_hf_cache_root())
    legacy_root = default_legacy_model_root()
    for preset in model_presets(target="mlx", engine="mflux", include_non_8bit=False):
        local_dir = legacy_root / preset.local_dir_name
        try:
            if resolve_hf_repo_snapshot(
                preset.repo_id,
                cache_dir=cache_root,
                require_weight_files=True,
                extra_roots=framework_hf_cache_roots(),
            ) is not None:
                return True
            if local_dir.is_dir() and any(local_dir.rglob("*.safetensors")):
                return True
        except Exception:
            continue
    return False


def _runtime_installed(provider: str) -> bool | None:
    p = str(provider or "").strip().lower().replace("_", "-")
    if not p:
        return None
    if p in {"openai", "openai-compatible"}:
        return True
    if p in {"huggingface", "diffusers", "hf"}:
        # Diffusers backends require both diffusers and torch.
        return importlib.util.find_spec("diffusers") is not None and importlib.util.find_spec("torch") is not None
    if p in {"mflux", "m-flux"}:
        if sys.platform != "darwin":
            return False
        return importlib.util.find_spec("mflux") is not None and importlib.util.find_spec("mlx") is not None
    if p in {"sdcpp", "stable-diffusion.cpp", "stable-diffusion-cpp", "stable_diffusion_cpp"}:
        return importlib.util.find_spec("stable_diffusion_cpp") is not None
    return None


def _configured_openai_base_url(owner: Any) -> Optional[str]:
    return _owner_cfg(owner, "vision_base_url") or _env("OPENAI_BASE_URL")


def _base_url_implies_openai_compatible(base_url: Any) -> bool:
    return bool(str(base_url or "").strip()) and not _looks_like_openai_api(str(base_url or ""))


def _truncate_string(value: str) -> str:
    if len(value) <= _PROVIDER_RAW_MAX_STRING:
        return value
    # #TRUNCATION: keep provider catalog raw metadata bounded for Core route payloads.
    return value[:_PROVIDER_RAW_MAX_STRING] + "...<truncated>"


def _json_safe_provider_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _truncate_string(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else str(value)
    if isinstance(value, dict):
        if depth >= _PROVIDER_RAW_MAX_DEPTH:
            return {"__truncated__": "max_depth"}
        out: Dict[str, Any] = {}
        for idx, (k, v) in enumerate(value.items()):
            if idx >= _PROVIDER_RAW_MAX_ITEMS:
                # #TRUNCATION: keep provider catalog raw metadata bounded for Core route payloads.
                out["__truncated__"] = f"kept first {_PROVIDER_RAW_MAX_ITEMS} items"
                break
            out[_truncate_string(str(k))] = _json_safe_provider_value(v, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        if depth >= _PROVIDER_RAW_MAX_DEPTH:
            return [{"__truncated__": "max_depth"}]
        out_list: List[Any] = []
        for idx, item in enumerate(value):
            if idx >= _PROVIDER_RAW_MAX_ITEMS:
                # #TRUNCATION: keep provider catalog raw metadata bounded for Core route payloads.
                out_list.append({"__truncated__": f"kept first {_PROVIDER_RAW_MAX_ITEMS} items"})
                break
            out_list.append(_json_safe_provider_value(item, depth=depth + 1))
        return out_list
    return _truncate_string(str(value))


def _provider_model_to_dict(info: ProviderModelInfo) -> Dict[str, Any]:
    raw = _json_safe_provider_value(info.raw if isinstance(info.raw, dict) else {})
    raw_obj = raw if isinstance(raw, dict) else {}
    provider = ""
    for key in ("provider", "provider_id", "provider_name", "backend", "engine_id", "owned_by"):
        value = raw_obj.get(key)
        if isinstance(value, str) and value.strip():
            provider = value.strip()
            break
    if not provider and info.owned_by is not None:
        provider = str(info.owned_by).strip()
    routed_model = ""
    for key in ("routed_model", "routing_model"):
        value = raw_obj.get(key)
        if isinstance(value, str) and value.strip():
            routed_model = value.strip()
            break
    model_value = raw_obj.get("model")
    if not isinstance(model_value, str) or not model_value.strip():
        model_value = routed_model or str(info.id)
    item = {
        "id": str(info.id),
        "model": str(model_value).strip(),
        "provider": provider or None,
        "routed_model": routed_model or None,
        "object": str(info.object) if info.object is not None else None,
        "created": int(info.created) if isinstance(info.created, int) else None,
        "owned_by": str(info.owned_by) if info.owned_by is not None else None,
        "capabilities": [str(c) for c in info.capabilities],
        "raw": raw_obj,
    }
    for key in ("parameter_defaults", "parameter_constraints", "parameters"):
        value = raw_obj.get(key)
        if isinstance(value, dict):
            item[key] = value
    return item


def _backend_supports_provider_catalog(backend: Any) -> bool:
    method = getattr(backend, "list_provider_models", None)
    if not callable(method):
        return False
    if isinstance(backend, VisionBackend):
        impl = getattr(type(backend), "list_provider_models", None)
        return impl is not VisionBackend.list_provider_models
    return True


def _provider_id_for_backend(backend: Any) -> Optional[str]:
    name = type(backend).__name__.lower()
    if "mflux" in name:
        return "mflux"
    if "huggingface" in name or "diffusers" in name:
        return "huggingface"
    if "stable" in name and "diffusion" in name:
        return "sdcpp"
    if "openai" in name:
        base_url = getattr(getattr(backend, "_cfg", None), "base_url", None)
        return "openai" if _looks_like_openai_api(str(base_url or "")) else "openai-compatible"
    return None


def _canonical_backend_kind(value: Any) -> Optional[str]:
    provider = str(value or "").strip().lower().replace("_", "-")
    if not provider:
        return None
    if provider in {"huggingface", "hf", "diffusers", "hf-diffusers"}:
        return "diffusers"
    if provider in {"mflux", "m-flux"}:
        return "mflux"
    if provider in {
        "sdcpp",
        "sd-cpp",
        "stable-diffusion.cpp",
        "stable-diffusion-cpp",
        "stable-diffusion-cpp-python",
        "stable_diffusion_cpp",
    }:
        return "sdcpp"
    if provider in {"openai", "remote"}:
        return "openai"
    if provider in _OPENAI_COMPATIBLE_BACKEND_KINDS:
        return "openai-compatible"
    return provider


def _canonical_provider_for_backend_kind(backend_kind: Optional[str]) -> Optional[str]:
    kind = _canonical_backend_kind(backend_kind)
    if kind == "diffusers":
        return "huggingface"
    return kind


def _canonical_load_id(backend_kind: Optional[str], model_id: Optional[str]) -> str:
    kind = _canonical_backend_kind(backend_kind) or "unknown"
    model = str(model_id or "").strip()
    return f"{kind}/{model}" if model else f"{kind}/default"


def _normalize_residency_task(value: Any) -> Optional[str]:
    raw = str(value or "").strip().lower().replace(" ", "_")
    if not raw:
        return None
    task = _RESIDENCY_TASK_ALIASES.get(raw)
    if task is None:
        raise AbstractVisionError(
            f"Unsupported residency task {value!r}. Use 'text_to_image', 'image_to_image', 'text_to_video', or 'image_to_video'."
        )
    return task


def _has_local_mflux_preset(model_id: str) -> bool:
    if sys.platform != "darwin":
        return False
    try:
        from ..model_cache import (
            default_hf_cache_root,
            default_legacy_model_root,
            framework_hf_cache_roots,
            resolve_hf_repo_snapshot,
        )
        from ..model_downloads import find_model_preset

        preset = find_model_preset(str(model_id), target="mlx", engine="mflux", require_8bit=True)
        if resolve_hf_repo_snapshot(
            preset.repo_id,
            cache_dir=str(default_hf_cache_root()),
            require_weight_files=True,
            extra_roots=framework_hf_cache_roots(),
        ) is not None:
            return True
        local_dir = default_legacy_model_root() / preset.local_dir_name
        return local_dir.exists() and any(local_dir.rglob("*.safetensors"))
    except Exception:
        return False


def _is_known_mflux_model_alias(model_id: str) -> bool:
    if sys.platform != "darwin":
        return False
    try:
        from ..model_downloads import find_model_preset

        find_model_preset(str(model_id), target="mlx", engine="mflux", require_8bit=True)
        return True
    except Exception:
        return False


def _canonical_mflux_model_id(model_id: Optional[str]) -> Optional[str]:
    s = str(model_id or "").strip()
    if not s:
        return None
    try:
        from ..model_downloads import find_model_preset

        return str(find_model_preset(s, target="mlx", engine="mflux", require_8bit=True).key)
    except Exception:
        return s


def _read_bytes_from_path(path: Union[str, Path]) -> bytes:
    p = Path(str(path)).expanduser()
    return p.read_bytes()


def _resolve_bytes_input(value: Union[bytes, Dict[str, Any], str], *, artifact_store: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict):
        if not is_artifact_ref(value):
            raise ValueError("Expected an artifact ref dict like {'$artifact': '...'}")
        if artifact_store is None:
            raise ValueError("artifact_store is required to resolve artifact refs to bytes")
        store = RuntimeArtifactStoreAdapter(artifact_store)
        return store.load_bytes(get_artifact_id(value))
    if isinstance(value, str):
        p = Path(value).expanduser()
        if p.exists() and p.is_file():
            return p.read_bytes()
        raise FileNotFoundError(f"File not found: {value}")
    raise TypeError("Unsupported input type; expected bytes, artifact-ref dict, or file path")


class _AbstractVisionCapability:
    """AbstractCore VisionCapability backed by AbstractVision."""

    backend_id = "abstractvision:openai"
    legacy_backend_id = "abstractvision:openai-compatible"

    def __init__(self, owner: Any, *, backend_id: Optional[str] = None):
        self._owner = owner
        self.backend_id = backend_id or type(self).backend_id
        self._backend = None
        self._routed_backends: Dict[tuple[Any, ...], Any] = {}
        self._state_lock = threading.RLock()
        self._loaded_models: Dict[tuple[Any, ...], Dict[str, Any]] = {}
        self._active_request_backend: Any = None
        self._active_request_backend_key: Optional[tuple[Any, ...]] = None
        self._backend_refcounts: Dict[int, int] = {}
        self._retired_backends: Dict[int, Any] = {}

    def _unload_backend(self, backend: Optional[Any]) -> None:
        if backend is None:
            return
        unload = getattr(backend, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception:
                pass

    def _retire_backend_locked(self, backend: Optional[Any]) -> Optional[Any]:
        if backend is None:
            return None
        backend_id = id(backend)
        if self._backend_refcounts.get(backend_id, 0) > 0:
            self._retired_backends[backend_id] = backend
            return None
        self._retired_backends.pop(backend_id, None)
        return backend

    def _acquire_backend_snapshot(self, backend: Any) -> None:
        with self._state_lock:
            backend_id = id(backend)
            self._backend_refcounts[backend_id] = self._backend_refcounts.get(backend_id, 0) + 1

    def _release_backend_snapshot(self, backend: Any) -> None:
        unload_after_lock: Optional[Any] = None
        with self._state_lock:
            backend_id = id(backend)
            remaining = self._backend_refcounts.get(backend_id, 0) - 1
            if remaining > 0:
                self._backend_refcounts[backend_id] = remaining
                return
            self._backend_refcounts.pop(backend_id, None)
            if self._active_request_backend is not backend:
                unload_after_lock = self._retired_backends.pop(backend_id, None)
        self._unload_backend(unload_after_lock)

    def _resolved_model_for_backend(
        self,
        backend: Any,
        *,
        backend_kind: Optional[str],
        requested_model: Optional[str] = None,
    ) -> Optional[str]:
        requested = str(requested_model or "").strip()
        cfg = getattr(backend, "_cfg", None)
        kind = _canonical_backend_kind(backend_kind)
        if kind == "diffusers":
            value = getattr(cfg, "model_id", None) or requested or _DEFAULT_LOCAL_DIFFUSERS_MODEL_ID
            return str(value).strip() or _DEFAULT_LOCAL_DIFFUSERS_MODEL_ID
        if kind == "mflux":
            value = (
                getattr(cfg, "model", None)
                or requested
                or _owner_cfg(self._owner, "vision_mflux_model")
                or _env("ABSTRACTVISION_MFLUX_MODEL")
                or _owner_cfg(self._owner, "vision_model_id")
                or _env("ABSTRACTVISION_MODEL")
                or _env("ABSTRACTVISION_MODEL_ID")
                or _env("ABSTRACTVISION_DIFFUSERS_MODEL_ID")
            )
            return str(value).strip() if value is not None else None
        if kind == "sdcpp":
            value = getattr(cfg, "model", None) or getattr(cfg, "diffusion_model", None) or requested
            return str(value).strip() if value is not None else None
        if kind in {"openai", "openai-compatible"}:
            value = (
                getattr(cfg, "model_id", None)
                or requested
                or _owner_cfg(self._owner, "vision_model_id")
                or _env("ABSTRACTVISION_MODEL")
                or _env("ABSTRACTVISION_MODEL_ID")
                or _env_first("OPENAI_IMAGE_MODEL_ID", "OPENAI_IMAGE_MODEL")
            )
            return str(value).strip() if value is not None else None
        return requested or None

    def _resolve_backend_binding(self, *, provider: Any = None, model: Any = None) -> Dict[str, Any]:
        provider_id = str(provider or "").strip().lower().replace("_", "-")
        model_id = str(model or "").strip()
        if model_id and "/" in model_id:
            head, tail = model_id.split("/", 1)
            head_id = head.strip().lower().replace("_", "-")
            if head_id == "mflux":
                provider_id = "mflux"
                model_id = tail.strip()
            elif head_id == "mlx":
                raise AbstractVisionError(
                    "AbstractVision does not have a generic MLX image backend yet. "
                    "Use provider/model `mflux/<preset>` for MFLUX-compatible 8-bit MLX models."
                )
            elif head_id in {"huggingface", "hf", "diffusers", "hf-diffusers"}:
                provider_id = "diffusers"
                model_id = tail.strip()
            elif head_id in {"openai", "openai-compatible"}:
                if not provider_id:
                    provider_id = "openai" if head_id == "openai" else "openai-compatible"
                model_id = _strip_openai_model_prefixes(model_id)
        if model_id and not provider_id and _has_local_mflux_preset(model_id):
            provider_id = "mflux"
        if (
            not provider_id
            and model_id.count("/") == 1
            and not model_id.startswith(("./", "../", "/", "~"))
            and "://" not in model_id
        ):
            provider_id = "mflux" if _has_local_mflux_preset(model_id) else "diffusers"
        if provider_id == "mlx":
            raise AbstractVisionError(
                "AbstractVision does not have a generic MLX image backend yet. "
                "Use provider 'mflux' for MFLUX-compatible 8-bit MLX models."
            )
        if provider_id in {"mflux", "m-flux"}:
            model_id = _canonical_mflux_model_id(model_id)

        backend: Any
        backend_key: tuple[Any, ...]
        backend_kind: Optional[str]
        if provider_id in {"mflux", "m-flux"}:
            backend_kind = "mflux"
            backend_key = ("mflux", model_id)
            backend = self._routed_backends.get(backend_key)
            if backend is None:
                backend = self._make_mflux_backend(model_id=model_id or None)
                self._routed_backends[backend_key] = backend
        elif provider_id in {"huggingface", "hf", "diffusers", "hf-diffusers"}:
            backend_kind = "diffusers"
            backend_key = ("diffusers", model_id or _DEFAULT_LOCAL_DIFFUSERS_MODEL_ID)
            backend = self._routed_backends.get(backend_key)
            if backend is None:
                backend = self._make_diffusers_backend(model_id=model_id or None)
                self._routed_backends[backend_key] = backend
        elif provider_id in {
            "sdcpp",
            "sd-cpp",
            "stable-diffusion.cpp",
            "stable-diffusion-cpp",
            "stable_diffusion_cpp",
        }:
            backend_kind = "sdcpp"
            backend_key = ("sdcpp", model_id or "default")
            backend = self._routed_backends.get(backend_key)
            if backend is None:
                backend = self._make_sdcpp_backend(model_id=model_id or None)
                self._routed_backends[backend_key] = backend
        elif provider_id in {"openai", "openai-compatible", "remote", "proxy"}:
            backend_kind = "openai-compatible" if provider_id in {"openai-compatible", "proxy"} else "openai"
            backend_key = ("openai", provider_id, model_id)
            backend = self._routed_backends.get(backend_key)
            if backend is None:
                backend = self._make_openai_backend(model_id=model_id or None, provider_id=provider_id)
                self._routed_backends[backend_key] = backend
        else:
            backend = self._get_backend()
            configured_kind = (
                getattr(backend, "backend_kind", None)
                or getattr(backend, "provider", None)
                or _owner_cfg(self._owner, "vision_backend")
                or _env("ABSTRACTVISION_PROVIDER")
                or _env("ABSTRACTVISION_BACKEND")
            )
            backend_kind = _canonical_backend_kind(_provider_id_for_backend(backend) or configured_kind)
            resolved_model = self._resolved_model_for_backend(
                backend,
                backend_kind=backend_kind,
                requested_model=model_id or None,
            )
            backend_key = ("configured", backend_kind or "unknown", resolved_model or "default")
            return {
                "backend": backend,
                "backend_key": backend_key,
                "backend_kind": backend_kind,
                "provider": _canonical_provider_for_backend_kind(backend_kind),
                "model": resolved_model,
                "load_id": _canonical_load_id(backend_kind, resolved_model),
                "local_control": backend_kind in _LOCAL_RESIDENCY_BACKEND_KINDS,
            }

        resolved_model = self._resolved_model_for_backend(
            backend,
            backend_kind=backend_kind,
            requested_model=model_id or None,
        )
        return {
            "backend": backend,
            "backend_key": backend_key,
            "backend_kind": backend_kind,
            "provider": _canonical_provider_for_backend_kind(backend_kind),
            "model": resolved_model,
            "load_id": _canonical_load_id(backend_kind, resolved_model),
            "local_control": backend_kind in _LOCAL_RESIDENCY_BACKEND_KINDS,
        }

    def _backend_for_request(self, *, provider: Any = None, model: Any = None):
        return self._resolve_backend_binding(provider=provider, model=model)["backend"]

    def _ensure_local_residency_supported(self, binding: Mapping[str, Any]) -> None:
        if bool(binding.get("local_control")):
            return
        raise AbstractVisionError(
            "Model residency control is only available for in-process local AbstractVision backends "
            "('diffusers', 'mflux', and 'sdcpp'). OpenAI-compatible HTTP backends are not controllable "
            "through this plugin, even when they point to localhost."
        )

    def _normalize_loaded_filters(self, filters: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        out = dict(filters or {})
        task = out.get("task")
        if task is not None:
            out["task"] = _normalize_residency_task(task)
        provider = out.get("provider") or out.get("backend") or out.get("backend_kind")
        if provider is not None:
            provider_kind = _canonical_backend_kind(provider)
            out["_provider_kind"] = provider_kind
            out["_provider_name"] = _canonical_provider_for_backend_kind(provider_kind)
        return out

    def _has_loaded_selector(self, filters: Mapping[str, Any]) -> bool:
        return any(
            str(filters.get(key) or "").strip()
            for key in ("provider", "backend", "backend_kind", "model", "load_id", "task")
        )

    def _match_loaded_record(self, record: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
        provider = str(filters.get("_provider_name") or "").strip()
        provider_kind = str(filters.get("_provider_kind") or "").strip()
        model = str(filters.get("model") or "").strip()
        load_id = str(filters.get("load_id") or "").strip()
        state = str(filters.get("state") or "").strip().lower()
        task_filter = filters.get("task")
        resident_filter = filters.get("resident")
        record_provider = str(record.get("provider") or "").strip()
        record_backend_kind = str(record.get("backend_kind") or "").strip()
        if provider and record_provider != provider:
            return False
        if provider_kind and record_backend_kind != provider_kind:
            return False
        if model and str(record.get("model") or "").strip() != model:
            return False
        if load_id and str(record.get("load_id") or "").strip() != load_id:
            return False
        if state and str(record.get("state") or "").strip().lower() != state:
            return False
        if task_filter is not None:
            record_tasks = record.get("tasks")
            if isinstance(record_tasks, list) and record_tasks:
                if str(task_filter) not in {str(item) for item in record_tasks}:
                    return False
            elif str(task_filter) != str(record.get("task") or ""):
                return False
        if resident_filter is not None and bool(record.get("resident")) is not bool(resident_filter):
            return False
        return True

    def _sorted_loaded_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            (dict(item) for item in records),
            key=lambda item: (
                0 if item.get("resident") else 1,
                str(item.get("provider") or ""),
                str(item.get("load_id") or ""),
            ),
        )

    def _find_loaded_backends(self, filters: Mapping[str, Any]) -> List[tuple[tuple[Any, ...], Dict[str, Any], Any]]:
        matches: List[tuple[tuple[Any, ...], Dict[str, Any], Any]] = []
        with self._state_lock:
            for backend_key, record in self._loaded_models.items():
                if not self._match_loaded_record(record, filters):
                    continue
                backend = None
                if self._active_request_backend_key == backend_key:
                    backend = self._active_request_backend
                elif backend_key in self._routed_backends:
                    backend = self._routed_backends.get(backend_key)
                elif backend_key and backend_key[0] == "configured":
                    backend = self._backend
                matches.append((backend_key, dict(record), backend))
        return matches

    def _record_loaded_model(
        self,
        binding: Mapping[str, Any],
        *,
        task: Optional[str],
        resident: bool,
        source: str,
        loaded_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        backend_key = tuple(binding["backend_key"])
        with self._state_lock:
            existing = dict(self._loaded_models.get(backend_key, {}))
            merged_tasks = {
                str(item)
                for item in (existing.get("tasks") or [])
                if str(item).strip()
            }
            if task:
                merged_tasks.add(str(task))
            record = {
                "task": task,
                "tasks": sorted(merged_tasks),
                "provider": binding.get("provider"),
                "model": binding.get("model"),
                "load_id": binding.get("load_id"),
                "backend_kind": binding.get("backend_kind"),
                "scope": "process",
                "state": "resident" if resident else "active",
                "resident": bool(resident),
                "loaded": True,
                "source": str(source),
                "loaded_at": (
                    existing.get("loaded_at")
                    if existing.get("loaded_at") is not None
                    else (loaded_at if loaded_at is not None else now)
                ),
                "last_used_at": now,
                "error": None,
            }
            if existing.get("resident"):
                record["resident"] = True
                record["state"] = "resident"
                record["source"] = "explicit_preload"
            if existing.get("task") and record.get("task") is None:
                record["task"] = existing.get("task")
            if not record["tasks"] and record.get("task"):
                record["tasks"] = [str(record["task"])]
            self._loaded_models[backend_key] = record
            return dict(record)

    def _remove_loaded_model_locked(self, backend_key: tuple[Any, ...]) -> Optional[Dict[str, Any]]:
        existing = self._loaded_models.pop(backend_key, None)
        return dict(existing) if existing is not None else None

    def _activate_request_backend(self, binding: Mapping[str, Any]) -> Any:
        backend = binding["backend"]
        backend_key = tuple(binding["backend_key"])
        unload_after_lock: Optional[Any] = None
        with self._state_lock:
            previous = self._active_request_backend
            previous_key = self._active_request_backend_key
            self._active_request_backend = backend
            self._active_request_backend_key = backend_key
            if previous is None or previous is backend:
                return backend
            previous_record = self._loaded_models.get(previous_key or ())
            if previous_record and bool(previous_record.get("resident")):
                return backend
            self._remove_loaded_model_locked(previous_key or ())
            unload_after_lock = self._retire_backend_locked(previous)
        self._unload_backend(unload_after_lock)
        return backend

    def _make_diffusers_backend(self, *, model_id: Optional[str] = None):
        resolved_model_id = (
            str(model_id).strip()
            if isinstance(model_id, str) and str(model_id).strip()
            else (
                _owner_cfg(self._owner, "vision_model_id")
                or _env("ABSTRACTVISION_DIFFUSERS_MODEL_ID")
                or _env("ABSTRACTVISION_MODEL")
                or _env("ABSTRACTVISION_MODEL_ID")
                or _DEFAULT_LOCAL_DIFFUSERS_MODEL_ID
            )
        )
        device = (
            _owner_cfg(self._owner, "vision_device")
            or _env("ABSTRACTVISION_DIFFUSERS_DEVICE", "auto")
            or "auto"
        )
        torch_dtype = _owner_cfg(self._owner, "vision_torch_dtype") or _env(
            "ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE"
        )
        allow_download = _owner_cfg_bool(
            self._owner,
            "vision_allow_download",
            _env_bool("ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD", False),
        )
        auto_retry_fp32 = _owner_cfg_bool(
            self._owner,
            "vision_auto_retry_fp32",
            _env_bool("ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32", True),
        )

        from ..backends.huggingface_diffusers import (
            HuggingFaceDiffusersBackendConfig,
            HuggingFaceDiffusersVisionBackend,
        )

        cfg = HuggingFaceDiffusersBackendConfig(
            model_id=str(resolved_model_id),
            device=str(device),
            torch_dtype=str(torch_dtype) if torch_dtype else None,
            allow_download=allow_download,
            auto_retry_fp32=auto_retry_fp32,
        )
        return HuggingFaceDiffusersVisionBackend(config=cfg)

    def _make_openai_backend(self, *, model_id: Optional[str] = None, provider_id: Optional[str] = None):
        configured_base_url = _configured_openai_base_url(self._owner)
        requested_provider = str(provider_id or "").strip().lower().replace("_", "-")
        explicit_openai_compatible = requested_provider in {"openai-compatible", "proxy"}

        base_url = configured_base_url
        if not base_url and not explicit_openai_compatible:
            base_url = _DEFAULT_OPENAI_BASE_URL

        owner_api_key = _owner_cfg(self._owner, "vision_api_key")
        if owner_api_key:
            api_key = owner_api_key
        else:
            api_key = _openai_api_key()
        resolved_model_id = (
            str(model_id).strip()
            if isinstance(model_id, str) and str(model_id).strip()
            else (
                _owner_cfg(self._owner, "vision_model_id")
                or _env("ABSTRACTVISION_MODEL")
                or _env("ABSTRACTVISION_MODEL_ID")
            )
        )
        if resolved_model_id:
            resolved_model_id = _strip_openai_model_prefixes(resolved_model_id)
        if not resolved_model_id:
            resolved_model_id = _env_first("OPENAI_IMAGE_MODEL_ID", "OPENAI_IMAGE_MODEL")
        if not resolved_model_id and not explicit_openai_compatible:
            resolved_model_id = _DEFAULT_OPENAI_IMAGE_MODEL_ID
        timeout_s_raw = _owner_cfg(self._owner, "vision_timeout_s") or _env(
            "ABSTRACTVISION_TIMEOUT_S"
        )
        try:
            timeout_s = float(timeout_s_raw) if timeout_s_raw else 300.0
        except Exception:
            timeout_s = 300.0
        if not base_url:
            raise AbstractVisionError(
                "Missing vision_base_url / OPENAI_BASE_URL. "
                "Configure an OpenAI-compatible endpoint or use OPENAI_API_KEY for OpenAI."
            )
        if not explicit_openai_compatible and not api_key:
            raise AbstractVisionError(
                "OpenAI image generation requires OPENAI_API_KEY."
            )

        t2v_path = _owner_cfg(self._owner, "vision_text_to_video_path") or _env(
            "ABSTRACTVISION_TEXT_TO_VIDEO_PATH"
        )
        i2v_path = _owner_cfg(self._owner, "vision_image_to_video_path") or _env(
            "ABSTRACTVISION_IMAGE_TO_VIDEO_PATH"
        )
        i2v_mode = _owner_cfg(self._owner, "vision_image_to_video_mode") or _env(
            "ABSTRACTVISION_IMAGE_TO_VIDEO_MODE", "multipart"
        )
        models_path = _owner_cfg(self._owner, "vision_models_path") or _env(
            "ABSTRACTVISION_MODELS_PATH", "/models"
        )

        from ..backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )

        cfg = OpenAICompatibleBackendConfig(
            base_url=str(base_url),
            api_key=str(api_key) if api_key else None,
            model_id=str(resolved_model_id) if resolved_model_id else None,
            timeout_s=float(timeout_s),
            models_path=str(models_path or "/models"),
            text_to_video_path=str(t2v_path) if t2v_path else None,
            image_to_video_path=str(i2v_path) if i2v_path else None,
            image_to_video_mode=str(i2v_mode or "multipart"),
        )
        return OpenAICompatibleVisionBackend(config=cfg)

    def _make_mflux_backend(self, *, model_id: Optional[str] = None):
        resolved_model = (
            str(model_id).strip()
            if isinstance(model_id, str) and str(model_id).strip()
            else (
                _owner_cfg(self._owner, "vision_mflux_model")
                or _env("ABSTRACTVISION_MFLUX_MODEL")
                or _owner_cfg(self._owner, "vision_model_id")
                or _env("ABSTRACTVISION_MODEL")
                or _env("ABSTRACTVISION_MODEL_ID")
                or _env("ABSTRACTVISION_DIFFUSERS_MODEL_ID")
            )
        )
        resolved_model = _canonical_mflux_model_id(str(resolved_model) if resolved_model else None)
        base_model = _owner_cfg(self._owner, "vision_mflux_base_model") or _env(
            "ABSTRACTVISION_MFLUX_BASE_MODEL"
        )
        model_dir = _owner_cfg(self._owner, "vision_model_dir") or _env("ABSTRACTVISION_MODEL_DIR")
        quantize_raw = _owner_cfg(self._owner, "vision_mflux_quantize") or _env(
            "ABSTRACTVISION_MFLUX_QUANTIZE"
        )
        quantize = int(quantize_raw) if quantize_raw else None
        allow_download = _owner_cfg_bool(
            self._owner,
            "vision_mflux_allow_download",
            _env_bool("ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD", False),
        )

        from ..backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        cfg = MFluxBackendConfig(
            model=str(resolved_model) if resolved_model else None,
            base_model=str(base_model) if base_model else None,
            model_dir=str(model_dir) if model_dir else None,
            quantize=quantize,
            allow_download=allow_download,
        )
        return MFluxVisionBackend(config=cfg)

    def _make_sdcpp_backend(self, *, model_id: Optional[str] = None):
        model = (
            str(model_id).strip()
            if isinstance(model_id, str) and str(model_id).strip()
            else (_owner_cfg(self._owner, "vision_sdcpp_model") or _env("ABSTRACTVISION_SDCPP_MODEL"))
        )
        diffusion_model = _owner_cfg(self._owner, "vision_sdcpp_diffusion_model") or _env(
            "ABSTRACTVISION_SDCPP_DIFFUSION_MODEL"
        )
        if not model and not diffusion_model:
            raise AbstractVisionError(
                "Missing stable-diffusion.cpp model configuration. Set vision_sdcpp_model, "
                "vision_sdcpp_diffusion_model, ABSTRACTVISION_SDCPP_MODEL, or ABSTRACTVISION_SDCPP_DIFFUSION_MODEL."
            )

        from ..backends.stable_diffusion_cpp import (
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
        )
        from ..model_downloads import MacOSGGUFUnsupportedError, resolve_sdcpp_model_selection

        vae = _owner_cfg(self._owner, "vision_sdcpp_vae") or _env("ABSTRACTVISION_SDCPP_VAE")
        llm = _owner_cfg(self._owner, "vision_sdcpp_llm") or _env("ABSTRACTVISION_SDCPP_LLM")
        llm_vision = _owner_cfg(self._owner, "vision_sdcpp_llm_vision") or _env("ABSTRACTVISION_SDCPP_LLM_VISION")
        extra_args = _owner_cfg(self._owner, "vision_sdcpp_extra_args") or _env(
            "ABSTRACTVISION_SDCPP_EXTRA_ARGS"
        )
        resolved_sdcpp = None
        if model and not any((diffusion_model, vae, llm, llm_vision)):
            try:
                model_path = Path(str(model)).expanduser()
            except Exception:
                model_path = None
            if model_path is None or not model_path.exists():
                try:
                    resolved_sdcpp = resolve_sdcpp_model_selection(str(model), allow_download=False)
                except MacOSGGUFUnsupportedError as e:
                    raise AbstractVisionError(str(e)) from e
                except ValueError:
                    resolved_sdcpp = None
                except RuntimeError as e:
                    raise AbstractVisionError(str(e)) from e

        cfg = StableDiffusionCppBackendConfig(
            sd_cli_path=_owner_cfg(self._owner, "vision_sdcpp_bin")
            or _env("ABSTRACTVISION_SDCPP_BIN", "sd-cli")
            or "sd-cli",
            model=resolved_sdcpp.model if resolved_sdcpp is not None else (str(model) if model else None),
            capabilities_model_id=(
                resolved_sdcpp.capabilities_model_id if resolved_sdcpp is not None else (str(model) if model else None)
            ),
            diffusion_model=(
                resolved_sdcpp.diffusion_model
                if resolved_sdcpp is not None
                else (str(diffusion_model) if diffusion_model else None)
            ),
            vae=resolved_sdcpp.vae if resolved_sdcpp is not None else vae,
            llm=resolved_sdcpp.llm if resolved_sdcpp is not None else llm,
            llm_vision=resolved_sdcpp.llm_vision if resolved_sdcpp is not None else llm_vision,
            clip_l=_owner_cfg(self._owner, "vision_sdcpp_clip_l")
            or _env("ABSTRACTVISION_SDCPP_CLIP_L"),
            clip_g=_owner_cfg(self._owner, "vision_sdcpp_clip_g")
            or _env("ABSTRACTVISION_SDCPP_CLIP_G"),
            t5xxl=_owner_cfg(self._owner, "vision_sdcpp_t5xxl")
            or _env("ABSTRACTVISION_SDCPP_T5XXL"),
            extra_args=tuple(shlex.split(str(extra_args))) if extra_args else (),
            timeout_s=float(
                _owner_cfg(self._owner, "vision_timeout_s")
                or _env("ABSTRACTVISION_TIMEOUT_S", "3600")
                or "3600"
            ),
        )
        return StableDiffusionCppVisionBackend(config=cfg)

    def _get_backend(self):
        if self._backend is not None:
            return self._backend

        # Injection hook (useful for tests and advanced embedding).
        try:
            cfg = getattr(self._owner, "config", None)
            if isinstance(cfg, dict):
                inst = cfg.get("vision_backend_instance")
                if inst is not None:
                    self._backend = inst
                    return self._backend
                factory = cfg.get("vision_backend_factory")
                if callable(factory):
                    self._backend = factory(self._owner)
                    return self._backend
        except Exception:
            pass

        # Prefer AbstractCore config keys when present; fall back to standard OpenAI env vars.
        # Hosted OpenAI is the default unless a non-OpenAI base URL or the legacy
        # backend id explicitly selects compatible-endpoint semantics.
        owner_base_url = _owner_cfg(self._owner, "vision_base_url")
        env_base_url = _env("OPENAI_BASE_URL")
        configured_base_url = _configured_openai_base_url(self._owner)
        configured_backend_kind = (
            _owner_cfg(self._owner, "vision_backend")
            or _env("ABSTRACTVISION_PROVIDER")
            or _env("ABSTRACTVISION_BACKEND")
        )
        raw_backend_kind = str(configured_backend_kind or "").strip().lower()
        if not raw_backend_kind:
            if (
                self.backend_id == self.legacy_backend_id
                or bool(owner_base_url)
                or _base_url_implies_openai_compatible(env_base_url)
            ):
                raw_backend_kind = "openai-compatible"
            else:
                raw_backend_kind = "openai"

        explicit_openai_compatible = raw_backend_kind in _OPENAI_COMPATIBLE_BACKEND_KINDS
        backend_kind = raw_backend_kind
        if backend_kind in _OPENAI_COMPATIBLE_BACKEND_KINDS:
            backend_kind = "openai"
        elif backend_kind in {"huggingface", "hf", "hf-diffusers"}:
            backend_kind = "diffusers"
        elif backend_kind in {
            "sd-cpp",
            "stable-diffusion.cpp",
            "stable-diffusion-cpp",
            "stable_diffusion_cpp",
        }:
            backend_kind = "sdcpp"
        elif backend_kind in {"m-flux"}:
            backend_kind = "mflux"

        configured_model_for_auto = (
            _owner_cfg(self._owner, "vision_mflux_model")
            or _env("ABSTRACTVISION_MFLUX_MODEL")
            or _owner_cfg(self._owner, "vision_model_id")
            or _env("ABSTRACTVISION_MODEL")
            or _env("ABSTRACTVISION_MODEL_ID")
            or _env("ABSTRACTVISION_DIFFUSERS_MODEL_ID")
        )
        explicit_remote = configured_base_url and raw_backend_kind in _OPENAI_COMPATIBLE_BACKEND_KINDS
        if (
            backend_kind in {"openai", "diffusers"}
            and not explicit_remote
            and configured_model_for_auto
            and (
                _has_local_mflux_preset(str(configured_model_for_auto))
                or raw_backend_kind in {"mflux", "m-flux"} and _is_known_mflux_model_alias(str(configured_model_for_auto))
            )
            and raw_backend_kind in {"", "huggingface", "hf", "hf-diffusers", "diffusers", "mflux", "m-flux"}
        ):
            backend_kind = "mflux"

        if backend_kind == "diffusers":
            self._backend = self._make_diffusers_backend()
            return self._backend

        if backend_kind == "sdcpp":
            self._backend = self._make_sdcpp_backend()
            return self._backend

        if backend_kind == "mflux":
            self._backend = self._make_mflux_backend()
            return self._backend

        if backend_kind != "openai":
            raise AbstractVisionError(
                f"Unsupported AbstractVision backend for AbstractCore plugin: {backend_kind!r}. "
                "Use 'mflux', 'diffusers', 'sdcpp', 'openai-compatible', or 'openai'."
            )

        base_url = configured_base_url
        if not base_url and not explicit_openai_compatible:
            base_url = _DEFAULT_OPENAI_BASE_URL

        owner_api_key = _owner_cfg(self._owner, "vision_api_key")
        if owner_api_key:
            api_key = owner_api_key
        else:
            api_key = _openai_api_key()
        model_id = (
            _owner_cfg(self._owner, "vision_model_id")
            or _env("ABSTRACTVISION_MODEL")
            or _env("ABSTRACTVISION_MODEL_ID")
        )
        if not model_id:
            model_id = _env_first("OPENAI_IMAGE_MODEL_ID", "OPENAI_IMAGE_MODEL")
        if not model_id and not explicit_openai_compatible:
            model_id = _DEFAULT_OPENAI_IMAGE_MODEL_ID
        timeout_s_raw = _owner_cfg(self._owner, "vision_timeout_s") or _env(
            "ABSTRACTVISION_TIMEOUT_S"
        )
        try:
            timeout_s = float(timeout_s_raw) if timeout_s_raw else 300.0
        except Exception:
            timeout_s = 300.0

        if not base_url:
            raise AbstractVisionError(
                "Missing vision_base_url / OPENAI_BASE_URL. "
                "Configure an OpenAI-compatible endpoint (e.g. http://localhost:8000/v1), "
                "or use ABSTRACTVISION_BACKEND=openai with OPENAI_API_KEY for OpenAI."
            )

        if not explicit_openai_compatible and not api_key:
            raise AbstractVisionError(
                "OpenAI image generation requires OPENAI_API_KEY."
            )

        # Optional video endpoints (not standardized; only enabled when configured).
        t2v_path = _owner_cfg(self._owner, "vision_text_to_video_path") or _env(
            "ABSTRACTVISION_TEXT_TO_VIDEO_PATH"
        )
        i2v_path = _owner_cfg(self._owner, "vision_image_to_video_path") or _env(
            "ABSTRACTVISION_IMAGE_TO_VIDEO_PATH"
        )
        i2v_mode = _owner_cfg(self._owner, "vision_image_to_video_mode") or _env(
            "ABSTRACTVISION_IMAGE_TO_VIDEO_MODE", "multipart"
        )
        models_path = _owner_cfg(self._owner, "vision_models_path") or _env(
            "ABSTRACTVISION_MODELS_PATH", "/models"
        )

        # Import backend module lazily (keeps plugin import-light).
        from ..backends.openai_compatible import (
            OpenAICompatibleBackendConfig,
            OpenAICompatibleVisionBackend,
        )

        cfg = OpenAICompatibleBackendConfig(
            base_url=str(base_url),
            api_key=str(api_key) if api_key else None,
            model_id=str(model_id) if model_id else None,
            timeout_s=float(timeout_s),
            models_path=str(models_path or "/models"),
            text_to_video_path=str(t2v_path) if t2v_path else None,
            image_to_video_path=str(i2v_path) if i2v_path else None,
            image_to_video_mode=str(i2v_mode or "multipart"),
        )
        self._backend = OpenAICompatibleVisionBackend(config=cfg)
        return self._backend

    def _make_manager(self, *, artifact_store: Any) -> VisionManager:
        store = RuntimeArtifactStoreAdapter(artifact_store) if artifact_store is not None else None
        return VisionManager(backend=self._get_backend(), store=store)

    def list_provider_models(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        backends: list[tuple[Any, Optional[str]]] = []
        last_error: Optional[Exception] = None
        injected_backend = False
        configured_base_url = _configured_openai_base_url(self._owner)
        configured_api_key = _owner_cfg(self._owner, "vision_api_key") or _openai_api_key()
        configured_remote_provider = (
            "openai-compatible"
            if _base_url_implies_openai_compatible(configured_base_url)
            else "openai"
        )
        skipped_remote_catalog = False
        attempted_catalog = False
        try:
            owner_cfg = getattr(self._owner, "config", None)
            injected_backend = isinstance(owner_cfg, dict) and (
                owner_cfg.get("vision_backend_instance") is not None
                or owner_cfg.get("vision_backend_factory") is not None
            )
        except Exception:
            injected_backend = False

        try:
            active_backend = self._get_backend()
            backends.append((active_backend, _provider_id_for_backend(active_backend)))
        except Exception:
            pass

        active_provider = backends[0][1] if backends else None

        if not injected_backend and active_provider != "mflux":
            try:
                backends.append((self._make_mflux_backend(), "mflux"))
            except Exception:
                pass

        if not injected_backend and active_provider != "huggingface":
            try:
                backends.append((self._make_diffusers_backend(), "huggingface"))
            except Exception:
                pass

        if (
            not injected_backend
            and active_provider != configured_remote_provider
            and _remote_provider_catalog_enabled(
                configured_remote_provider,
                base_url=configured_base_url,
                api_key=configured_api_key,
            )
        ):
            try:
                backends.append(
                    (
                        self._make_openai_backend(provider_id=configured_remote_provider),
                        configured_remote_provider,
                    )
                )
            except Exception:
                pass

        out: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for idx, (backend, provider) in enumerate(backends):
            if not (injected_backend and idx == 0) and not _backend_catalog_enabled(
                self._owner, provider, backend
            ):
                skipped_remote_catalog = True
                continue
            if not _backend_supports_provider_catalog(backend):
                continue
            attempted_catalog = True
            try:
                models = list(backend.list_provider_models(task=task) or [])
            except Exception as exc:
                last_error = exc
                continue
            for model in models:
                item = _provider_model_to_dict(model)
                if provider and not item.get("provider"):
                    item["provider"] = provider
                model_id = str(item.get("model") or item.get("id") or "").strip()
                provider_id = str(item.get("provider") or provider or "").strip()
                if not model_id:
                    continue
                key = (provider_id, model_id)
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
        if out:
            return out
        if last_error is not None:
            raise AbstractVisionError(str(last_error))
        if not backends:
            return []
        if attempted_catalog or skipped_remote_catalog:
            return []
        raise AbstractVisionError(
            "The selected AbstractVision backend does not support provider model catalogs."
        )

    def available_providers(self, *, task: Optional[str] = None) -> Dict[str, Any]:
        """Return fast provider availability without remote model discovery.

        This method is used by AbstractCore catalog routes and must be robust:
        it should not require outbound HTTP calls (e.g. `/models`) to determine
        provider availability.
        """
        configured_base_url = _configured_openai_base_url(self._owner)
        base_url = str(configured_base_url or "").strip() or None
        api_key = _owner_cfg(self._owner, "vision_api_key") or _openai_api_key()
        openai_available = _remote_provider_catalog_enabled(
            "openai",
            base_url=base_url or _DEFAULT_OPENAI_BASE_URL,
            api_key=api_key,
        )
        compatible_available = _remote_provider_catalog_enabled(
            "openai-compatible",
            base_url=base_url,
            api_key=api_key,
        )

        known = ["openai", "openai-compatible", "huggingface", "mflux", "sdcpp"]
        available: list[str] = []

        if _runtime_installed("huggingface"):
            available.append("huggingface")
        if _runtime_installed("mflux") or _mflux_weights_present():
            available.append("mflux")
        if _runtime_installed("sdcpp"):
            available.append("sdcpp")

        if base_url and _base_url_implies_openai_compatible(base_url):
            if compatible_available:
                available.append("openai-compatible")
        elif openai_available:
            available.append("openai")

        # Keep ordering stable for UIs.
        order = ["openai", "openai-compatible", "huggingface", "mflux", "sdcpp"]
        available_sorted = [provider for provider in order if provider in available]  # provider ids are canonical

        return {
            "task": task,
            "providers": list(known),
            "available_providers": available_sorted,
            "details": {
                provider: {
                    "id": provider,
                    "provider": provider,
                    "installed": _runtime_installed(provider),
                    "weights_present": _mflux_weights_present() if provider == "mflux" else None,
                    "remote": provider in {"openai", "openai-compatible"},
                    "local": provider not in {"openai", "openai-compatible"},
                    "reachable": (
                        openai_available
                        if provider == "openai"
                        else compatible_available
                        if provider == "openai-compatible"
                        else None
                    ),
                }
                for provider in known
            },
            "base_url": base_url,
            "has_api_key": bool(str(api_key or "").strip()),
        }

    def list_available_providers(self, *, task: Optional[str] = None) -> Dict[str, Any]:
        return self.available_providers(task=task)

    def list_available_models(self, *, task: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.list_provider_models(task=task)

    def load_resident_model(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(request or {})
        task = _normalize_residency_task(payload.get("task"))
        provider = payload.get("provider") or payload.get("backend") or payload.get("backend_kind")
        model = payload.get("model") or payload.get("load_id")
        binding = self._resolve_backend_binding(provider=provider, model=model)
        self._ensure_local_residency_supported(binding)
        backend = binding["backend"]
        preload = getattr(backend, "preload", None)
        if callable(preload):
            preload()
        loaded_at = time.time()
        return self._record_loaded_model(
            binding,
            task=task,
            resident=True,
            source="explicit_preload",
            loaded_at=loaded_at,
        )

    def list_loaded_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        filter_map = self._normalize_loaded_filters(filters)
        with self._state_lock:
            records = [dict(item) for item in self._loaded_models.values()]
        filtered = [item for item in records if self._match_loaded_record(item, filter_map)]
        return self._sorted_loaded_records(filtered)

    def list_resident_models(self, filters: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
        filter_map = self._normalize_loaded_filters(filters)
        filter_map["resident"] = True
        return self.list_loaded_models(filter_map)

    def unload_resident_model(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        payload = self._normalize_loaded_filters(request)
        matched = self._find_loaded_backends(payload)
        unload_after_lock: Optional[Any] = None
        removed: Optional[Dict[str, Any]] = None
        if len(matched) > 1:
            raise AbstractVisionError(
                "Ambiguous unload request: more than one loaded local model matched. "
                "Specify `load_id`, or include both provider/backend and model."
            )
        if len(matched) == 1:
            backend_key, matched_record, backend = matched[0]
            with self._state_lock:
                removed = self._remove_loaded_model_locked(backend_key)
                if backend is not None and self._active_request_backend is backend:
                    self._active_request_backend = None
                    self._active_request_backend_key = None
                unload_after_lock = self._retire_backend_locked(backend)
            self._unload_backend(unload_after_lock)
            return {
                "task": matched_record.get("task"),
                "provider": matched_record.get("provider"),
                "model": matched_record.get("model"),
                "load_id": matched_record.get("load_id"),
                "backend_kind": matched_record.get("backend_kind"),
                "scope": "process",
                "state": "unloaded",
                "resident": False,
                "loaded": False,
                "source": matched_record.get("source"),
                "loaded_at": None,
                "last_used_at": matched_record.get("last_used_at"),
                "error": None,
            }

        if not self._has_loaded_selector(payload):
            raise AbstractVisionError(
                "Unload request did not identify a model. Specify `load_id`, or include both "
                "provider/backend and model."
            )

        provider = payload.get("provider") or payload.get("backend") or payload.get("backend_kind")
        model = payload.get("model") or payload.get("load_id")
        binding = self._resolve_backend_binding(provider=provider, model=model)
        self._ensure_local_residency_supported(binding)
        return {
            "task": removed.get("task") if isinstance(removed, dict) else _normalize_residency_task(payload.get("task")),
            "provider": binding.get("provider"),
            "model": binding.get("model"),
            "load_id": binding.get("load_id"),
            "backend_kind": binding.get("backend_kind"),
            "scope": "process",
            "state": "unloaded",
            "resident": False,
            "loaded": False,
            "source": removed.get("source") if isinstance(removed, dict) else None,
            "loaded_at": None,
            "last_used_at": removed.get("last_used_at") if isinstance(removed, dict) else None,
            "error": None,
        }

    # Aliases for Core route adapters that prefer load/list/unload naming.
    def load_model(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return self.load_resident_model(request)

    def unload_model(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        return self.unload_resident_model(request)

    def t2i(self, prompt: str, **kwargs: Any):
        store = kwargs.pop("artifact_store", None)
        run_id = kwargs.pop("run_id", None)
        tags = kwargs.pop("tags", None)
        provider = kwargs.pop("provider", None)
        model = kwargs.pop("model", None)
        binding = self._resolve_backend_binding(provider=provider, model=model)
        backend = self._activate_request_backend(binding)
        allowed_request_keys = {"negative_prompt", "width", "height", "seed", "steps", "guidance_scale", "extra"}
        extra = kwargs.get("extra")
        merged_extra = dict(extra) if isinstance(extra, dict) else {}
        for key in list(kwargs.keys()):
            if key not in allowed_request_keys:
                value = kwargs.pop(key)
                if value is not None:
                    merged_extra[str(key)] = value
        if isinstance(model, str) and model.strip() and "openai" in type(backend).__name__.lower():
            merged_extra["model"] = model.strip()
        if merged_extra:
            kwargs["extra"] = merged_extra
        vm = VisionManager(
            backend=backend,
            store=RuntimeArtifactStoreAdapter(store, run_id=run_id, tags=tags) if store is not None else None,
        )
        self._acquire_backend_snapshot(backend)
        try:
            out = vm.generate_image(str(prompt), **kwargs)
            if binding.get("local_control"):
                self._record_loaded_model(binding, task="text_to_image", resident=False, source="request")
            if isinstance(out, dict):
                return out
            return bytes(getattr(out, "data", b""))
        finally:
            self._release_backend_snapshot(backend)

    def i2i(self, prompt: str, image: Union[bytes, Dict[str, Any], str], **kwargs: Any):
        store = kwargs.pop("artifact_store", None)
        run_id = kwargs.pop("run_id", None)
        tags = kwargs.pop("tags", None)
        provider = kwargs.pop("provider", None)
        model = kwargs.pop("model", None)
        image_b = _resolve_bytes_input(image, artifact_store=store)
        mask = kwargs.pop("mask", None)
        mask_b = None
        if mask is not None:
            mask_b = _resolve_bytes_input(mask, artifact_store=store)
        binding = self._resolve_backend_binding(provider=provider, model=model)
        backend = self._activate_request_backend(binding)
        vm = VisionManager(
            backend=backend,
            store=RuntimeArtifactStoreAdapter(store, run_id=run_id, tags=tags) if store is not None else None,
        )
        self._acquire_backend_snapshot(backend)
        try:
            out = vm.edit_image(str(prompt), image=image_b, mask=mask_b, **kwargs)
            if binding.get("local_control"):
                self._record_loaded_model(binding, task="image_to_image", resident=False, source="request")
            if isinstance(out, dict):
                return out
            return bytes(getattr(out, "data", b""))
        finally:
            self._release_backend_snapshot(backend)

    def multi_view_image(self, prompt: str, **kwargs: Any):
        store = kwargs.pop("artifact_store", None)
        vm = self._make_manager(artifact_store=store)
        out = vm.generate_angles(str(prompt), **kwargs)
        if isinstance(out, list) and all(isinstance(x, dict) for x in out):
            return out
        return [bytes(getattr(asset, "data", b"")) for asset in out]

    def generate_angles(self, prompt: str, **kwargs: Any):
        return self.multi_view_image(prompt, **kwargs)

    def t2v(self, prompt: str, **kwargs: Any):
        store = kwargs.pop("artifact_store", None)
        run_id = kwargs.pop("run_id", None)
        tags = kwargs.pop("tags", None)
        provider = kwargs.pop("provider", None)
        model = kwargs.pop("model", None)
        binding = self._resolve_backend_binding(provider=provider, model=model)
        backend = self._activate_request_backend(binding)
        allowed_request_keys = {
            "negative_prompt",
            "width",
            "height",
            "fps",
            "num_frames",
            "seed",
            "steps",
            "guidance_scale",
            "extra",
        }
        extra = kwargs.get("extra")
        merged_extra = dict(extra) if isinstance(extra, dict) else {}
        for key in list(kwargs.keys()):
            if key not in allowed_request_keys:
                value = kwargs.pop(key)
                if value is not None:
                    merged_extra[str(key)] = value
        if isinstance(model, str) and model.strip() and "openai" in type(backend).__name__.lower():
            merged_extra["model"] = model.strip()
        if merged_extra:
            kwargs["extra"] = merged_extra
        vm = VisionManager(
            backend=backend,
            store=RuntimeArtifactStoreAdapter(store, run_id=run_id, tags=tags) if store is not None else None,
        )
        self._acquire_backend_snapshot(backend)
        try:
            out = vm.generate_video(str(prompt), **kwargs)
            if binding.get("local_control"):
                self._record_loaded_model(binding, task="text_to_video", resident=False, source="request")
            if isinstance(out, dict):
                return out
            return bytes(getattr(out, "data", b""))
        finally:
            self._release_backend_snapshot(backend)

    def i2v(self, image: Union[bytes, Dict[str, Any], str], **kwargs: Any):
        store = kwargs.pop("artifact_store", None)
        run_id = kwargs.pop("run_id", None)
        tags = kwargs.pop("tags", None)
        provider = kwargs.pop("provider", None)
        model = kwargs.pop("model", None)
        image_b = _resolve_bytes_input(image, artifact_store=store)
        binding = self._resolve_backend_binding(provider=provider, model=model)
        backend = self._activate_request_backend(binding)
        vm = VisionManager(
            backend=backend,
            store=RuntimeArtifactStoreAdapter(store, run_id=run_id, tags=tags) if store is not None else None,
        )
        self._acquire_backend_snapshot(backend)
        try:
            out = vm.image_to_video(image=image_b, **kwargs)
            if binding.get("local_control"):
                self._record_loaded_model(binding, task="image_to_video", resident=False, source="request")
            if isinstance(out, dict):
                return out
            return bytes(getattr(out, "data", b""))
        finally:
            self._release_backend_snapshot(backend)


def register(registry: Any) -> None:
    """Register AbstractVision as an AbstractCore capability plugin.

    This function is loaded via the `abstractcore.capabilities_plugins` entry point group.
    """

    def _factory(owner: Any) -> _AbstractVisionCapability:
        return _AbstractVisionCapability(owner, backend_id=_AbstractVisionCapability.backend_id)

    def _legacy_factory(owner: Any) -> _AbstractVisionCapability:
        return _AbstractVisionCapability(
            owner,
            backend_id=_AbstractVisionCapability.legacy_backend_id,
        )

    config_hint = (
        "Default: OpenAI HTTP via https://api.openai.com/v1. Set OPENAI_API_KEY. "
        "Set ABSTRACTVISION_PROVIDER=openai-compatible (alias: ABSTRACTVISION_BACKEND=openai-compatible) "
        "and OPENAI_BASE_URL to target a local/remote compatible /v1 endpoint. "
        "Set ABSTRACTVISION_PROVIDER=mflux|diffusers|sdcpp (alias: ABSTRACTVISION_BACKEND=...) to run local AbstractVision backends."
    )
    legacy_config_hint = (
        "Compatibility backend id: set OPENAI_BASE_URL to a local/remote compatible "
        "/v1 endpoint. New OpenAI configs should use abstractvision:openai or "
        "ABSTRACTVISION_PROVIDER=openai (alias: ABSTRACTVISION_BACKEND=openai) with OPENAI_API_KEY."
    )

    registry.register_vision_backend(
        backend_id=_AbstractVisionCapability.backend_id,
        factory=_factory,
        priority=0,
        description="AbstractVision capability plugin (OpenAI HTTP by default; compatible HTTP, MFLUX, Diffusers, or stable-diffusion.cpp via env/config).",
        config_hint=config_hint,
    )
    registry.register_vision_backend(
        backend_id=_AbstractVisionCapability.legacy_backend_id,
        factory=_legacy_factory,
        priority=-1,
        description="Compatibility backend id for AbstractVision OpenAI-compatible HTTP/local backend selection.",
        config_hint=legacy_config_hint,
    )
