from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple, TypeVar

from ..types import (
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
    normalize_memory_mode,
    resolve_memory_mode_with_source,
)
from ..errors import CapabilityNotSupportedError, GenerationCancelledError

logger = logging.getLogger(__name__)

_Request = TypeVar("_Request")

# The request type of each task, as the carrier of a preload's options (`_resolve_preload_options`).
_PRELOAD_REQUEST_TYPES: Dict[str, Callable[[], Any]] = {
    "text_to_image": lambda: ImageGenerationRequest(prompt=""),
    "image_to_image": lambda: ImageEditRequest(prompt="", image=b""),
    "image_upscale": lambda: ImageUpscaleRequest(image=b""),
    "text_to_video": lambda: VideoGenerationRequest(prompt=""),
    "image_to_video": lambda: ImageToVideoRequest(image=b""),
}

# The fallbacks of $ABSTRACTVISION_MEMORY_MODE already logged in this process, as
# (backend type, mode, supported modes): the AbstractCore plugin and the CLI build a backend per
# request, so a flag on the instance would log the same fallback at every request.
_DEGRADED_PROCESS_MEMORY_MODES_LOGGED: set = set()
_DEGRADED_PROCESS_MEMORY_MODES_LOCK = threading.Lock()

# The typed request options, defined once. The soundtrack options exist on video requests only.
SOUNDTRACK_REQUEST_FIELDS: Tuple[str, ...] = ("soundscape", "music", "generate_audio", "audio_shift")
# The options a backend must apply or refuse (never ignore); `supported_request_options` lists the
# ones it applies.
OPTIONAL_REQUEST_FIELDS: Tuple[str, ...] = ("quantize", *SOUNDTRACK_REQUEST_FIELDS)
# Every typed request option, which a caller may also pass in `extra`: the memory mode (checked
# against `supported_memory_modes`) and the options above.
REQUEST_OPTION_FIELDS: Tuple[str, ...] = ("memory_mode", *OPTIONAL_REQUEST_FIELDS)


def option_is_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def merge_request_option(name: str, value: Any, extra: Mapping[str, Any]) -> Any:
    """Request option `name`: the typed `value`, else `extra[name]`; ValueError when both differ.

    Memory modes compare by their canonical name ("LOW" and "low" agree).
    """
    from_extra = extra.get(name)
    if not option_is_set(value):
        return from_extra if option_is_set(from_extra) else value
    if option_is_set(from_extra):
        same = (
            normalize_memory_mode(value) == normalize_memory_mode(from_extra)
            if name == "memory_mode"
            else value == from_extra
        )
        if not same:
            raise ValueError(f"{name}={value!r} and extra[{name!r}]={from_extra!r} disagree; pass one.")
    return value


def request_int(name: str, value: Any, *, label: str) -> int:
    """`value` as an int, or ValueError naming the field: ints, integral floats, decimal strings.

    The one rule for an integer request option (a load-time `quantize` included) on every backend that
    reads one; `label` names the backend in the message.
    """
    if isinstance(value, bool):
        raise ValueError(f"{label} {name} must be an integer, got {value!r}.")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} {name} must be an integer, got {value!r}.") from None
    # A number that is not whole (8.5 in any numeric type) is not an integer; a decimal string already failed above.
    if not isinstance(value, str) and value != number:
        raise ValueError(f"{label} {name} must be an integer, got {value!r}.")
    return number


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

    def supported_memory_modes(self, task: Optional[str] = None) -> Tuple[str, ...]:
        """Memory modes this backend can honour for its configured model and device."""
        _ = task
        return (MEMORY_MODE_DEFAULT,)

    def supported_request_options(self, task: Optional[str] = None) -> FrozenSet[str]:
        """Typed request options this backend applies for `task` (see OPTIONAL_REQUEST_FIELDS).

        A request that sets an option not listed here, as a field or in `extra`, is refused by
        `resolve_memory_request`.
        """
        _ = task
        return frozenset()

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
    ) -> Tuple[_Request, str]:
        """Resolve, check and strip a request's typed options; every backend entry point calls it.

        `memory_mode`, and each option the backend lists in `supported_request_options(task)`, is
        taken from its field or from `extra` (`merge_request_option`, which refuses the two
        disagreeing). For any other option an `extra` key is not the option: it stays in `extra`,
        uninterpreted, and reaches the backend as any other `extra` key does. Returns the request
        with the options in their fields and removed from `extra` (so no backend forwards them as
        engine flags or HTTP fields), and the mode. A mode the backend does not list in
        `supported_memory_modes(task)` raises CapabilityNotSupportedError before any work when
        the request named it; a mode from $ABSTRACTVISION_MEMORY_MODE (one value for every
        backend of the process) runs as the lowest-memory listed mode that is not below it. A
        typed option the backend does not list, or a listed one the request type does not carry,
        is refused the same way.
        """
        return self._resolve_request_options(
            request,
            task=task,
            supported=tuple(self.supported_memory_modes(task)),
            listed=self.supported_request_options(task),
        )

    def _resolve_request_options(
        self,
        request: _Request,
        *,
        task: Optional[str],
        supported: Tuple[str, ...],
        listed: FrozenSet[str],
    ) -> Tuple[_Request, str]:
        """`resolve_memory_request` against the given memory modes and options: those of the model
        the request runs on, when a backend answers per model (a remote's entry for the model sent)."""
        raw_extra = getattr(request, "extra", None)
        extra = dict(raw_extra) if isinstance(raw_extra, Mapping) else {}
        options = {}
        for name in REQUEST_OPTION_FIELDS:
            typed = getattr(request, name, None)
            if name == "memory_mode" or name in listed:
                options[name] = merge_request_option(name, typed, extra)
                extra.pop(name, None)
            else:
                options[name] = typed
        mode, from_env = resolve_memory_mode_with_source(options.pop("memory_mode"))
        if from_env and mode not in supported:
            mode = self._degrade_process_memory_mode(mode, supported)
        if mode not in supported:
            where = f" for {task}" if task else ""
            raise CapabilityNotSupportedError(
                f"{type(self).__name__} cannot honour memory_mode={mode!r}{where} with its configured "
                f"model and device (supported: {', '.join(supported)}). The mode comes from the "
                f"request, or from ${MEMORY_MODE_ENV_VAR} when the request names none.",
                field="memory_mode",
                supported=supported,
            )
        unsupported = [
            name
            for name, value in options.items()
            if option_is_set(value) and (not hasattr(request, name) or name not in listed)
        ]
        if unsupported:
            where = f" for {task}" if task else ""
            raise CapabilityNotSupportedError(
                f"{type(self).__name__} does not support {', '.join(unsupported)}{where} with its "
                "configured model.",
                field=unsupported[0] if len(unsupported) == 1 else None,
                supported=() if len(unsupported) == 1 else None,
            )
        raise_if_cancelled(getattr(request, "cancel_event", None))
        if hasattr(request, "memory_mode"):
            carried = {name: value for name, value in options.items() if hasattr(request, name)}
            request = replace(request, memory_mode=mode, extra=extra, **carried)  # type: ignore[type-var]
        return request, mode

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

    def preload(self, memory_mode: Optional[str] = None, quantize: Optional[int] = None) -> None:
        """Best-effort eager load/prepare; does not guarantee a fully warmed first inference.

        `memory_mode` and `quantize` are those of the requests the loaded model is for, checked as
        a request's are (`_resolve_preload_options`) before anything loads. This backend loads
        nothing ahead of a request, so it only checks them.
        """
        if memory_mode is not None or quantize is not None:
            self._resolve_preload_options(memory_mode, quantize, task=None)

    def _resolve_preload_options(
        self, memory_mode: Optional[str], quantize: Optional[int], *, task: Optional[str]
    ) -> Tuple[str, Optional[int]]:
        """The memory mode and load-time quantize a preload for `task` runs with.

        Checked by `resolve_memory_request`, as a request of that task carrying them would be: the
        same refusals, and the process default when `memory_mode` is None.
        """
        carrier = _PRELOAD_REQUEST_TYPES.get(task or "", _PRELOAD_REQUEST_TYPES["text_to_image"])()
        request, mode = self.resolve_memory_request(
            replace(carrier, memory_mode=memory_mode, quantize=quantize), task=task
        )
        return mode, request.quantize

    def unload(self) -> None:
        """Best-effort: release model weights from memory."""
        return None

    def has_loaded_model(self) -> bool:
        """Whether this process holds this backend's model weights in memory now.

        Backends that load models in process answer it: False before a load, after `unload`, and
        after a run or a failed `preload` that released the model. The default, False, is the
        answer of a backend that loads nothing here (a remote one).
        """
        return False

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
