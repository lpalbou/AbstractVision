from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple, TypeVar

from ..types import (
    BATCH_EXTRA_KEY,
    MEMORY_MODE_DEFAULT,
    MEMORY_MODE_ENV_VAR,
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
    highest_memory_mode_at_or_below,
    resolve_memory_mode_with_source,
)
from ..errors import CapabilityNotSupportedError, GenerationCancelledError

logger = logging.getLogger(__name__)

_Request = TypeVar("_Request")

# The fallbacks of $ABSTRACTVISION_MEMORY_MODE already logged in this process, as
# (backend type, mode, supported modes): the AbstractCore plugin and the CLI build a backend per
# request, so a flag on the instance would log the same fallback at every request.
_DEGRADED_PROCESS_MEMORY_MODES_LOGGED: set = set()
_DEGRADED_PROCESS_MEMORY_MODES_LOCK = threading.Lock()

# Typed request options a backend must apply or refuse (never ignore).
OPTIONAL_REQUEST_FIELDS: Tuple[str, ...] = (
    "quantize",
    "soundscape",
    "music",
    "generate_audio",
    "audio_shift",
)


def _option_is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def raise_if_cancelled(cancel_event: Any) -> None:
    """Raise GenerationCancelledError when `cancel_event` (anything with is_set()) is set."""
    if cancel_event is None:
        return
    is_set = getattr(cancel_event, "is_set", None)
    if not callable(is_set):
        raise ValueError("cancel_event must provide is_set(), like threading.Event.")
    if is_set():
        raise GenerationCancelledError("The request was cancelled (cancel_event is set).")


class VisionBackend(ABC):
    """Backend interface for generative vision tasks."""

    # True on backends that read VisionManager's batch marker (BATCH_EXTRA_KEY in `extra`);
    # VisionManager sends it to no other backend.
    uses_batch_marker = False

    def supported_memory_modes(self, task: Optional[str] = None) -> Tuple[str, ...]:
        """Memory modes this backend can honour for its configured model and device."""
        _ = task
        return (MEMORY_MODE_DEFAULT,)

    def supported_request_options(self, task: Optional[str] = None) -> FrozenSet[str]:
        """Typed request options this backend applies for `task` (see OPTIONAL_REQUEST_FIELDS).

        A request that sets an option not listed here is refused by `resolve_memory_request`.
        """
        _ = task
        return frozenset()

    def process_memory_mode(self, task: Optional[str] = None) -> str:
        """$ABSTRACTVISION_MEMORY_MODE as this backend runs it (warmups use it too)."""
        mode, from_env = resolve_memory_mode_with_source(None)
        supported = tuple(self.supported_memory_modes(task))
        if from_env and mode not in supported:
            return self._degrade_process_memory_mode(mode, supported)
        return mode

    def _degrade_process_memory_mode(self, mode: str, supported: Sequence[str]) -> str:
        degraded = highest_memory_mode_at_or_below(mode, supported)
        key = (type(self).__name__, mode, tuple(supported))
        with _DEGRADED_PROCESS_MEMORY_MODES_LOCK:
            first = key not in _DEGRADED_PROCESS_MEMORY_MODES_LOGGED
            _DEGRADED_PROCESS_MEMORY_MODES_LOGGED.add(key)
        if first:
            logger.warning(
                "%s cannot honour $%s=%s with its configured model and device; requests that name "
                "no memory_mode run as %r (supported: %s).",
                type(self).__name__,
                MEMORY_MODE_ENV_VAR,
                mode,
                degraded,
                ", ".join(supported),
            )
        return degraded

    def resolve_memory_request(
        self, request: _Request, *, task: Optional[str] = None
    ) -> Tuple[_Request, str, Optional[Dict[str, Any]]]:
        """Resolve, check and strip a request's memory mode; every backend entry point calls it.

        Returns the request with `memory_mode` set and `memory_mode` / the batch marker removed
        from `extra` (so no backend forwards them as engine flags or HTTP fields), the mode, and
        the batch marker set by VisionManager (None outside a batch). A mode the backend does not
        list in `supported_memory_modes(task)` raises CapabilityNotSupportedError before any work
        when the request named it; a mode from $ABSTRACTVISION_MEMORY_MODE (one value for every
        backend of the process) runs as the lowest-memory listed mode that is not below it.
        """
        raw_extra = getattr(request, "extra", None)
        extra = dict(raw_extra) if isinstance(raw_extra, dict) else {}
        mode, from_env = resolve_memory_mode_with_source(getattr(request, "memory_mode", None), extra)
        supported = tuple(self.supported_memory_modes(task))
        if from_env and mode not in supported:
            mode = self._degrade_process_memory_mode(mode, supported)
        if mode not in supported:
            where = f" for {task}" if task else ""
            raise CapabilityNotSupportedError(
                f"{type(self).__name__} cannot honour memory_mode={mode!r}{where} with its configured "
                f"model and device (supported: {', '.join(supported)}). The mode comes from the "
                f"request, or from ${MEMORY_MODE_ENV_VAR} when the request names none."
            )
        unsupported = [
            name
            for name in OPTIONAL_REQUEST_FIELDS
            if _option_is_set(getattr(request, name, None))
            and name not in self.supported_request_options(task)
        ]
        if unsupported:
            where = f" for {task}" if task else ""
            raise CapabilityNotSupportedError(
                f"{type(self).__name__} does not support {', '.join(unsupported)}{where} with its "
                "configured model."
            )
        raise_if_cancelled(getattr(request, "cancel_event", None))
        extra.pop("memory_mode", None)
        batch = extra.pop(BATCH_EXTRA_KEY, None)
        if hasattr(request, "memory_mode"):
            request = replace(request, memory_mode=mode, extra=extra)  # type: ignore[type-var]
        return request, mode, batch if isinstance(batch, dict) else None

    def normalize_image_generation_request(
        self,
        request: ImageGenerationRequest,
    ) -> ImageGenerationRequest:
        """Best-effort request normalization before execution."""
        return request

    def normalize_image_edit_request(
        self,
        request: ImageEditRequest,
    ) -> ImageEditRequest:
        """Best-effort request normalization before execution."""
        return request

    def normalize_video_generation_request(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationRequest:
        """Best-effort request normalization before execution."""
        return request

    def normalize_image_to_video_request(
        self,
        request: ImageToVideoRequest,
    ) -> ImageToVideoRequest:
        """Best-effort request normalization before execution."""
        return request

    def normalize_image_upscale_request(
        self,
        request: ImageUpscaleRequest,
    ) -> ImageUpscaleRequest:
        """Best-effort request normalization before execution."""
        return request

    def generate_image_with_progress(
        self,
        request: ImageGenerationRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        """Generate an image, optionally reporting progress (best-effort)."""
        _ = progress_callback
        return self.generate_image(request)

    def edit_image_with_progress(
        self,
        request: ImageEditRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        """Edit an image, optionally reporting progress (best-effort)."""
        _ = progress_callback
        return self.edit_image(request)

    def generate_video_with_progress(
        self,
        request: VideoGenerationRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        """Generate a video, optionally reporting progress (best-effort)."""
        _ = progress_callback
        return self.generate_video(request)

    def image_to_video_with_progress(
        self,
        request: ImageToVideoRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        """Generate a video from an image, optionally reporting progress (best-effort)."""
        _ = progress_callback
        return self.image_to_video(request)

    def upscale_image_with_progress(
        self,
        request: ImageUpscaleRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        """Upscale an image, optionally reporting progress (best-effort)."""
        _ = progress_callback
        return self.upscale_image(request)

    def describe_routes(
        self, *, model: Optional[str] = None, task: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Engine route rows for `model`/`task`, as JSON; empty when the backend publishes none."""
        _ = model, task
        return []

    def get_capabilities(self) -> Optional[VisionBackendCapabilities]:
        """Return backend-level capability constraints (optional)."""
        return None

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        """Return provider-advertised model entries, when the backend can query them.

        This is explicit provider catalog discovery. Backends must not use it to
        silently select or change the configured model.
        """
        _ = task
        return ()

    def list_provider_adapters(
        self,
        *,
        model: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Sequence[ProviderAdapterInfo]:
        """Return backend-discoverable adapter entries, when supported."""
        _ = model
        _ = task
        return ()

    def preload(self) -> None:
        """Best-effort eager load/prepare; does not guarantee a fully warmed first inference."""
        return None

    def unload(self) -> None:
        """Best-effort: release model weights from memory."""
        return None

    @abstractmethod
    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset: ...

    @abstractmethod
    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset: ...

    @abstractmethod
    def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]: ...

    @abstractmethod
    def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset: ...

    @abstractmethod
    def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset: ...

    def upscale_image(self, request: ImageUpscaleRequest) -> GeneratedAsset:
        _ = request
        raise CapabilityNotSupportedError(
            f"{self.__class__.__name__} does not support image_upscale."
        )
