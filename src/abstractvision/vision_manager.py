from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .backends import VisionBackend
from .artifacts import MediaStore
from .errors import BackendNotConfiguredError, CapabilityNotSupportedError
from .model_capabilities import VisionModelCapabilitiesRegistry
from .types import (
    GeneratedAsset,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageToVideoRequest,
    MultiAngleRequest,
    VideoGenerationRequest,
    VisionBackendCapabilities,
)


@dataclass
class VisionManager:
    """High-level orchestrator for generative vision tasks.

    Intentionally thin: delegates execution to the configured backend.
    """

    backend: Optional[VisionBackend] = None
    store: Optional[MediaStore] = None
    model_id: Optional[str] = None
    registry: Optional[VisionModelCapabilitiesRegistry] = None

    def __post_init__(self) -> None:
        if self.model_id and self.registry is None:
            self.registry = VisionModelCapabilitiesRegistry()

    def _require_backend(self) -> VisionBackend:
        if self.backend is None:
            raise BackendNotConfiguredError(
                "No vision backend configured. "
                "Provide a backend to VisionManager(backend=...) before calling generation methods."
            )
        return self.backend

    def _require_model_support(self, task: str) -> None:
        if not self.model_id:
            return
        reg = self.registry or VisionModelCapabilitiesRegistry()
        # Keep a reference so repeated calls don't reload the asset.
        self.registry = reg
        reg.require_support(str(self.model_id), str(task))

    def _backend_caps(self, backend: VisionBackend) -> Optional[VisionBackendCapabilities]:
        try:
            return backend.get_capabilities()
        except Exception:
            return None

    def _require_backend_support(self, backend: VisionBackend, task: str) -> Optional[VisionBackendCapabilities]:
        caps = self._backend_caps(backend)
        if caps is None:
            return None
        if caps.supported_tasks is not None and str(task) not in set([str(t) for t in caps.supported_tasks]):
            raise CapabilityNotSupportedError(f"Backend does not support task '{task}'.")
        return caps

    def _maybe_store(self, asset: GeneratedAsset, *, tags: Optional[Dict[str, str]] = None) -> Union[GeneratedAsset, Dict[str, Any]]:
        if self.store is None:
            return asset
        return self.store.store_bytes(
            asset.data,
            content_type=asset.mime_type,
            metadata=asset.metadata,
            tags=tags,
        )

    def generate_image(self, prompt: str, **kwargs) -> Union[GeneratedAsset, Dict[str, Any]]:
        backend = self._require_backend()
        self._require_model_support("text_to_image")
        self._require_backend_support(backend, "text_to_image")
        asset = backend.generate_image(ImageGenerationRequest(prompt=prompt, **kwargs))
        return self._maybe_store(asset, tags={"kind": "generated_media", "modality": "image", "task": "text_to_image"})

    def edit_image(self, prompt: str, image: bytes, **kwargs) -> Union[GeneratedAsset, Dict[str, Any]]:
        backend = self._require_backend()
        self._require_model_support("image_to_image")
        caps = self._require_backend_support(backend, "image_to_image")
        mask = kwargs.get("mask")
        if mask is not None and caps is not None and caps.supports_mask is False:
            raise CapabilityNotSupportedError("Backend does not support masked image edits (mask parameter).")
        asset = backend.edit_image(ImageEditRequest(prompt=prompt, image=image, **kwargs))
        return self._maybe_store(asset, tags={"kind": "generated_media", "modality": "image", "task": "image_to_image"})

    def generate_angles(self, prompt: str, **kwargs) -> Union[List[GeneratedAsset], List[Dict[str, Any]]]:
        backend = self._require_backend()
        self._require_model_support("multi_view_image")
        self._require_backend_support(backend, "multi_view_image")
        assets = backend.generate_angles(MultiAngleRequest(prompt=prompt, **kwargs))
        if self.store is None:
            return assets
        return [self._maybe_store(a, tags={"kind": "generated_media", "modality": "image", "task": "multi_view_image"}) for a in assets]  # type: ignore[return-value]

    def generate_video(self, prompt: str, **kwargs) -> Union[GeneratedAsset, Dict[str, Any]]:
        backend = self._require_backend()
        self._require_model_support("text_to_video")
        self._require_backend_support(backend, "text_to_video")
        asset = backend.generate_video(VideoGenerationRequest(prompt=prompt, **kwargs))
        return self._maybe_store(asset, tags={"kind": "generated_media", "modality": "video", "task": "text_to_video"})

    def image_to_video(self, image: bytes, **kwargs) -> Union[GeneratedAsset, Dict[str, Any]]:
        backend = self._require_backend()
        self._require_model_support("image_to_video")
        self._require_backend_support(backend, "image_to_video")
        asset = backend.image_to_video(ImageToVideoRequest(image=image, **kwargs))
        return self._maybe_store(asset, tags={"kind": "generated_media", "modality": "video", "task": "image_to_video"})
