from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    return {
        "id": str(info.id),
        "object": str(info.object) if info.object is not None else None,
        "created": int(info.created) if isinstance(info.created, int) else None,
        "owned_by": str(info.owned_by) if info.owned_by is not None else None,
        "capabilities": [str(c) for c in info.capabilities],
        "raw": _json_safe_provider_value(info.raw if isinstance(info.raw, dict) else {}),
    }


def _backend_supports_provider_catalog(backend: Any) -> bool:
    method = getattr(backend, "list_provider_models", None)
    if not callable(method):
        return False
    if isinstance(backend, VisionBackend):
        impl = getattr(type(backend), "list_provider_models", None)
        return impl is not VisionBackend.list_provider_models
    return True


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

        # Prefer AbstractCore config keys when present; fall back to AbstractVision env vars.
        # Hosted OpenAI is the new default backend id. The legacy backend id and
        # base-url-only env setups retain OpenAI-compatible semantics.
        configured_base_url = _owner_cfg(self._owner, "vision_base_url") or _env(
            "ABSTRACTVISION_BASE_URL"
        )
        configured_backend_kind = _owner_cfg(self._owner, "vision_backend") or _env(
            "ABSTRACTVISION_BACKEND"
        )
        raw_backend_kind = str(configured_backend_kind or "").strip().lower()
        if not raw_backend_kind:
            if self.backend_id == self.legacy_backend_id or configured_base_url:
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

        if backend_kind == "diffusers":
            model_id = (
                _owner_cfg(self._owner, "vision_model_id")
                or _env("ABSTRACTVISION_DIFFUSERS_MODEL_ID")
                or _env("ABSTRACTVISION_MODEL_ID")
                or _DEFAULT_LOCAL_DIFFUSERS_MODEL_ID
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
                model_id=str(model_id),
                device=str(device),
                torch_dtype=str(torch_dtype) if torch_dtype else None,
                allow_download=allow_download,
                auto_retry_fp32=auto_retry_fp32,
            )
            self._backend = HuggingFaceDiffusersVisionBackend(config=cfg)
            return self._backend

        if backend_kind == "sdcpp":
            model = _owner_cfg(self._owner, "vision_sdcpp_model") or _env(
                "ABSTRACTVISION_SDCPP_MODEL"
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

            extra_args = _owner_cfg(self._owner, "vision_sdcpp_extra_args") or _env(
                "ABSTRACTVISION_SDCPP_EXTRA_ARGS"
            )
            cfg = StableDiffusionCppBackendConfig(
                sd_cli_path=_owner_cfg(self._owner, "vision_sdcpp_bin")
                or _env("ABSTRACTVISION_SDCPP_BIN", "sd-cli")
                or "sd-cli",
                model=str(model) if model else None,
                diffusion_model=str(diffusion_model) if diffusion_model else None,
                vae=_owner_cfg(self._owner, "vision_sdcpp_vae") or _env("ABSTRACTVISION_SDCPP_VAE"),
                llm=_owner_cfg(self._owner, "vision_sdcpp_llm") or _env("ABSTRACTVISION_SDCPP_LLM"),
                llm_vision=_owner_cfg(self._owner, "vision_sdcpp_llm_vision")
                or _env("ABSTRACTVISION_SDCPP_LLM_VISION"),
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
            self._backend = StableDiffusionCppVisionBackend(config=cfg)
            return self._backend

        if backend_kind != "openai":
            raise AbstractVisionError(
                f"Unsupported AbstractVision backend for AbstractCore plugin: {backend_kind!r}. "
                "Use 'diffusers', 'sdcpp', 'openai-compatible', or 'openai'."
            )

        base_url = configured_base_url
        if not base_url and not explicit_openai_compatible:
            base_url = _env("OPENAI_BASE_URL", _DEFAULT_OPENAI_BASE_URL)

        api_key = _owner_cfg(self._owner, "vision_api_key") or _env_first(
            "ABSTRACTVISION_API_KEY",
            "OPENAI_API_KEY",
        )
        model_id = _owner_cfg(self._owner, "vision_model_id") or _env("ABSTRACTVISION_MODEL_ID")
        if not model_id and not explicit_openai_compatible:
            model_id = _env_first(
                "OPENAI_IMAGE_MODEL_ID",
                "OPENAI_IMAGE_MODEL",
                default=_DEFAULT_OPENAI_IMAGE_MODEL_ID,
            )
        timeout_s_raw = _owner_cfg(self._owner, "vision_timeout_s") or _env(
            "ABSTRACTVISION_TIMEOUT_S"
        )
        try:
            timeout_s = float(timeout_s_raw) if timeout_s_raw else 300.0
        except Exception:
            timeout_s = 300.0

        if not base_url:
            raise AbstractVisionError(
                "Missing vision_base_url / ABSTRACTVISION_BASE_URL. "
                "Configure an OpenAI-compatible endpoint (e.g. http://localhost:8000/v1), "
                "or use ABSTRACTVISION_BACKEND=openai with OPENAI_API_KEY for OpenAI."
            )

        if _looks_like_openai_api(str(base_url)) and not api_key:
            raise AbstractVisionError(
                "OpenAI image generation requires OPENAI_API_KEY or ABSTRACTVISION_API_KEY."
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
        backend = self._get_backend()
        if not _backend_supports_provider_catalog(backend):
            raise AbstractVisionError(
                "The selected AbstractVision backend does not support provider model catalogs."
            )
        models = backend.list_provider_models(task=task)
        return [_provider_model_to_dict(model) for model in models]

    def t2i(self, prompt: str, **kwargs: Any):
        store = kwargs.pop("artifact_store", None)
        vm = self._make_manager(artifact_store=store)
        out = vm.generate_image(str(prompt), **kwargs)
        if isinstance(out, dict):
            return out
        return bytes(getattr(out, "data", b""))

    def i2i(self, prompt: str, image: Union[bytes, Dict[str, Any], str], **kwargs: Any):
        store = kwargs.pop("artifact_store", None)
        image_b = _resolve_bytes_input(image, artifact_store=store)
        mask = kwargs.pop("mask", None)
        mask_b = None
        if mask is not None:
            mask_b = _resolve_bytes_input(mask, artifact_store=store)
        vm = self._make_manager(artifact_store=store)
        out = vm.edit_image(str(prompt), image=image_b, mask=mask_b, **kwargs)
        if isinstance(out, dict):
            return out
        return bytes(getattr(out, "data", b""))

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
        vm = self._make_manager(artifact_store=store)
        out = vm.generate_video(str(prompt), **kwargs)
        if isinstance(out, dict):
            return out
        return bytes(getattr(out, "data", b""))

    def i2v(self, image: Union[bytes, Dict[str, Any], str], **kwargs: Any):
        store = kwargs.pop("artifact_store", None)
        image_b = _resolve_bytes_input(image, artifact_store=store)
        vm = self._make_manager(artifact_store=store)
        out = vm.image_to_video(image=image_b, **kwargs)
        if isinstance(out, dict):
            return out
        return bytes(getattr(out, "data", b""))


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
        "Default: OpenAI HTTP via https://api.openai.com/v1. Set OPENAI_API_KEY "
        "(or ABSTRACTVISION_API_KEY). Set ABSTRACTVISION_BACKEND=openai-compatible "
        "and ABSTRACTVISION_BASE_URL to target a local/remote compatible /v1 endpoint. "
        "Set ABSTRACTVISION_BACKEND=diffusers or sdcpp to run local AbstractVision backends."
    )
    legacy_config_hint = (
        "Compatibility backend id: set ABSTRACTVISION_BASE_URL to a local/remote compatible "
        "/v1 endpoint. New OpenAI configs should use abstractvision:openai or "
        "ABSTRACTVISION_BACKEND=openai with OPENAI_API_KEY."
    )

    registry.register_vision_backend(
        backend_id=_AbstractVisionCapability.backend_id,
        factory=_factory,
        priority=0,
        description="AbstractVision capability plugin (OpenAI HTTP by default; compatible HTTP, Diffusers, or stable-diffusion.cpp via env/config).",
        config_hint=config_hint,
    )
    registry.register_vision_backend(
        backend_id=_AbstractVisionCapability.legacy_backend_id,
        factory=_legacy_factory,
        priority=-1,
        description="Compatibility backend id for AbstractVision OpenAI-compatible HTTP/local backend selection.",
        config_hint=legacy_config_hint,
    )
