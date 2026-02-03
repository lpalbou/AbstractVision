from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..artifacts import RuntimeArtifactStoreAdapter, is_artifact_ref, get_artifact_id
from ..errors import AbstractVisionError
from ..vision_manager import VisionManager


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(str(key), None)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


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

    backend_id = "abstractvision:openai-compatible"

    def __init__(self, owner: Any):
        self._owner = owner
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
        backend_kind = (_owner_cfg(self._owner, "vision_backend") or _env("ABSTRACTVISION_BACKEND", "openai") or "openai").lower()

        if backend_kind not in {"openai", "openai-compatible"}:
            raise AbstractVisionError(
                "Only the OpenAI-compatible HTTP backend is supported via the AbstractCore plugin (v0). "
                "Set vision_backend='openai' (or ABSTRACTVISION_BACKEND=openai)."
            )

        base_url = _owner_cfg(self._owner, "vision_base_url") or _env("ABSTRACTVISION_BASE_URL")
        api_key = _owner_cfg(self._owner, "vision_api_key") or _env("ABSTRACTVISION_API_KEY")
        model_id = _owner_cfg(self._owner, "vision_model_id") or _env("ABSTRACTVISION_MODEL_ID")
        timeout_s_raw = _owner_cfg(self._owner, "vision_timeout_s") or _env("ABSTRACTVISION_TIMEOUT_S")
        try:
            timeout_s = float(timeout_s_raw) if timeout_s_raw else 300.0
        except Exception:
            timeout_s = 300.0

        if not base_url:
            raise AbstractVisionError(
                "Missing vision_base_url / ABSTRACTVISION_BASE_URL. "
                "Configure an OpenAI-compatible endpoint (e.g. http://localhost:8000/v1)."
            )

        # Optional video endpoints (not standardized; only enabled when configured).
        t2v_path = _owner_cfg(self._owner, "vision_text_to_video_path") or _env("ABSTRACTVISION_TEXT_TO_VIDEO_PATH")
        i2v_path = _owner_cfg(self._owner, "vision_image_to_video_path") or _env("ABSTRACTVISION_IMAGE_TO_VIDEO_PATH")
        i2v_mode = _owner_cfg(self._owner, "vision_image_to_video_mode") or _env("ABSTRACTVISION_IMAGE_TO_VIDEO_MODE", "multipart")

        # Import backend module lazily (keeps plugin import-light).
        from ..backends.openai_compatible import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend

        cfg = OpenAICompatibleBackendConfig(
            base_url=str(base_url),
            api_key=str(api_key) if api_key else None,
            model_id=str(model_id) if model_id else None,
            timeout_s=float(timeout_s),
            text_to_video_path=str(t2v_path) if t2v_path else None,
            image_to_video_path=str(i2v_path) if i2v_path else None,
            image_to_video_mode=str(i2v_mode or "multipart"),
        )
        self._backend = OpenAICompatibleVisionBackend(config=cfg)
        return self._backend

    def _make_manager(self, *, artifact_store: Any) -> VisionManager:
        store = RuntimeArtifactStoreAdapter(artifact_store) if artifact_store is not None else None
        return VisionManager(backend=self._get_backend(), store=store)

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
        return _AbstractVisionCapability(owner)

    config_hint = (
        "Set ABSTRACTVISION_BASE_URL (or pass vision_base_url=...) to point to an OpenAI-compatible /v1 endpoint. "
        "Example: vision_base_url='http://localhost:8000/v1' (AbstractCore Server vision endpoints) or "
        "vision_base_url='http://localhost:1234/v1' (LMStudio/vLLM)."
    )

    registry.register_vision_backend(
        backend_id=_AbstractVisionCapability.backend_id,
        factory=_factory,
        priority=0,
        description="AbstractVision via OpenAI-compatible HTTP backend (env/config-driven).",
        config_hint=config_hint,
    )
