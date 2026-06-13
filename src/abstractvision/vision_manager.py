from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from .artifacts import MediaStore
from .backends.base_backend import VisionBackend
from .errors import BackendNotConfiguredError, CapabilityNotSupportedError
from .model_capabilities import VisionModelCapabilitiesRegistry
from .types import (
    GeneratedAsset,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageToVideoRequest,
    ImageUpscaleRequest,
    MultiAngleRequest,
    ProviderAdapterInfo,
    ProviderModelInfo,
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

    def _require_backend_support(
        self, backend: VisionBackend, task: str
    ) -> Optional[VisionBackendCapabilities]:
        caps = self._backend_caps(backend)
        if caps is None:
            return None
        if caps.supported_tasks is not None and str(task) not in {
            str(t) for t in caps.supported_tasks
        }:
            raise CapabilityNotSupportedError(f"Backend does not support task '{task}'.")
        return caps

    def _maybe_store(
        self, asset: GeneratedAsset, *, tags: Optional[Dict[str, str]] = None
    ) -> Union[GeneratedAsset, Dict[str, Any]]:
        if self.store is None:
            return asset
        return self.store.store_bytes(
            asset.data,
            content_type=asset.mime_type,
            metadata=asset.metadata,
            tags=tags,
        )

    def _move_progress_callbacks_to_extra(self, kwargs: Dict[str, Any]) -> None:
        extra = kwargs.get("extra")
        merged_extra = dict(extra) if isinstance(extra, dict) else {}
        for key in ("on_progress", "progress_event_callback", "progress_callback"):
            if key not in kwargs:
                continue
            callback = kwargs.pop(key)
            if callback is not None:
                merged_extra[key] = callback
        if merged_extra:
            kwargs["extra"] = merged_extra

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        """List models advertised by the configured provider backend, if supported."""
        backend = self._require_backend()
        return backend.list_provider_models(task=task)

    def list_provider_adapters(
        self,
        *,
        model: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Sequence[ProviderAdapterInfo]:
        """List adapters advertised or discovered by the configured provider backend."""
        backend = self._require_backend()
        return backend.list_provider_adapters(model=model, task=task)

    def _batch_seeds(
        self,
        *,
        count: int,
        seed: Optional[int] = None,
        seeds: Optional[Sequence[int]] = None,
    ) -> List[Optional[int]]:
        if seeds is not None:
            planned = [int(value) for value in seeds]
            if not planned:
                raise ValueError("Batch generation seeds cannot be empty.")
            if count != len(planned):
                raise ValueError(
                    f"Batch generation count ({count}) must match the number of explicit seeds ({len(planned)})."
                )
            return planned
        if count <= 0:
            raise ValueError("Batch generation count must be >= 1.")
        if count == 1:
            return [int(seed)] if seed is not None else [None]
        if seed is not None:
            base_seed = int(seed)
            return [base_seed + index for index in range(count)]
        rng = random.SystemRandom()
        return [int(rng.randrange(0, 1_000_000_000)) for _ in range(count)]

    def generate_image(self, prompt: str, **kwargs) -> Union[GeneratedAsset, Dict[str, Any]]:
        backend = self._require_backend()
        self._require_model_support("text_to_image")
        self._require_backend_support(backend, "text_to_image")
        self._move_progress_callbacks_to_extra(kwargs)
        request = ImageGenerationRequest(prompt=prompt, **kwargs)
        normalize = getattr(backend, "normalize_image_generation_request", None)
        if callable(normalize):
            request = normalize(request)
        asset = backend.generate_image(request)
        return self._maybe_store(
            asset, tags={"kind": "generated_media", "modality": "image", "task": "text_to_image"}
        )

    def generate_image_batch(
        self,
        prompt: str,
        *,
        count: int = 1,
        seeds: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> List[Union[GeneratedAsset, Dict[str, Any]]]:
        planned_seeds = self._batch_seeds(count=count, seed=kwargs.get("seed"), seeds=seeds)
        out: List[Union[GeneratedAsset, Dict[str, Any]]] = []
        for planned_seed in planned_seeds:
            call_kwargs = dict(kwargs)
            call_kwargs["seed"] = planned_seed
            out.append(self.generate_image(prompt, **call_kwargs))
        return out

    def edit_image(
        self, prompt: str, image: bytes, **kwargs
    ) -> Union[GeneratedAsset, Dict[str, Any]]:
        backend = self._require_backend()
        self._require_model_support("image_to_image")
        caps = self._require_backend_support(backend, "image_to_image")
        mask = kwargs.get("mask")
        if mask is not None and caps is not None and caps.supports_mask is False:
            raise CapabilityNotSupportedError(
                "Backend does not support masked image edits (mask parameter)."
            )
        self._move_progress_callbacks_to_extra(kwargs)
        request = ImageEditRequest(prompt=prompt, image=image, **kwargs)
        normalize = getattr(backend, "normalize_image_edit_request", None)
        if callable(normalize):
            request = normalize(request)
        asset = backend.edit_image(request)
        return self._maybe_store(
            asset, tags={"kind": "generated_media", "modality": "image", "task": "image_to_image"}
        )

    def edit_image_batch(
        self,
        prompt: str,
        image: bytes,
        *,
        count: int = 1,
        seeds: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> List[Union[GeneratedAsset, Dict[str, Any]]]:
        planned_seeds = self._batch_seeds(count=count, seed=kwargs.get("seed"), seeds=seeds)
        out: List[Union[GeneratedAsset, Dict[str, Any]]] = []
        for planned_seed in planned_seeds:
            call_kwargs = dict(kwargs)
            call_kwargs["seed"] = planned_seed
            out.append(self.edit_image(prompt, image=image, **call_kwargs))
        return out

    def upscale_image(self, image: bytes, **kwargs) -> Union[GeneratedAsset, Dict[str, Any]]:
        backend = self._require_backend()
        self._require_model_support("image_upscale")
        self._require_backend_support(backend, "image_upscale")
        self._move_progress_callbacks_to_extra(kwargs)
        request = ImageUpscaleRequest(image=image, **kwargs)
        normalize = getattr(backend, "normalize_image_upscale_request", None)
        if callable(normalize):
            request = normalize(request)
        asset = backend.upscale_image(request)
        return self._maybe_store(
            asset, tags={"kind": "generated_media", "modality": "image", "task": "image_upscale"}
        )

    def generate_angles(
        self, prompt: str, **kwargs
    ) -> Union[List[GeneratedAsset], List[Dict[str, Any]]]:
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
        self._move_progress_callbacks_to_extra(kwargs)
        request = VideoGenerationRequest(prompt=prompt, **kwargs)
        normalize = getattr(backend, "normalize_video_generation_request", None)
        if callable(normalize):
            request = normalize(request)
        asset = backend.generate_video(request)
        return self._maybe_store(
            asset, tags={"kind": "generated_media", "modality": "video", "task": "text_to_video"}
        )

    def generate_video_batch(
        self,
        prompt: str,
        *,
        count: int = 1,
        seeds: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> List[Union[GeneratedAsset, Dict[str, Any]]]:
        planned_seeds = self._batch_seeds(count=count, seed=kwargs.get("seed"), seeds=seeds)
        out: List[Union[GeneratedAsset, Dict[str, Any]]] = []
        for planned_seed in planned_seeds:
            call_kwargs = dict(kwargs)
            call_kwargs["seed"] = planned_seed
            out.append(self.generate_video(prompt, **call_kwargs))
        return out

    def image_to_video(self, image: bytes, **kwargs) -> Union[GeneratedAsset, Dict[str, Any]]:
        backend = self._require_backend()
        self._require_model_support("image_to_video")
        self._require_backend_support(backend, "image_to_video")
        self._move_progress_callbacks_to_extra(kwargs)
        request = ImageToVideoRequest(image=image, **kwargs)
        normalize = getattr(backend, "normalize_image_to_video_request", None)
        if callable(normalize):
            request = normalize(request)
        asset = backend.image_to_video(request)
        return self._maybe_store(
            asset, tags={"kind": "generated_media", "modality": "video", "task": "image_to_video"}
        )

    def image_to_video_batch(
        self,
        image: bytes,
        *,
        count: int = 1,
        seeds: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> List[Union[GeneratedAsset, Dict[str, Any]]]:
        planned_seeds = self._batch_seeds(count=count, seed=kwargs.get("seed"), seeds=seeds)
        out: List[Union[GeneratedAsset, Dict[str, Any]]] = []
        for planned_seed in planned_seeds:
            call_kwargs = dict(kwargs)
            call_kwargs["seed"] = planned_seed
            out.append(self.image_to_video(image, **call_kwargs))
        return out
