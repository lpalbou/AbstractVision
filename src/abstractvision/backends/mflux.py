from __future__ import annotations

import itertools
import logging
import math
import random
import os
import sys
import tempfile
import queue
import threading
import weakref
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from ..errors import (
    CapabilityNotSupportedError,
    GenerationCancelledError,
    GpuOutOfMemoryError,
    OptionalDependencyMissingError,
)
from ..lora_adapters import (
    adapter_source_for_mlx_gen,
    known_lora_adapter_profile,
    lora_scales_for_backend,
    lora_target_roles_for_backend,
    recommended_lora_request_overrides,
    resolve_request_lora_adapters,
    serialize_lora_adapters,
)
from ..model_capabilities import VisionModelCapabilitiesRegistry
from ..model_downloads import (
    download_command,
    download_hf_repo_snapshot,
    find_model_preset,
    looks_like_hf_repo_id,
    model_presets,
)
from ..model_cache import (
    default_hf_cache_root,
    default_legacy_model_root,
    ensure_hf_repo_snapshot,
    framework_hf_cache_roots,
    framework_local_model_roots,
    hf_cache_roots,
    hf_snapshot_is_usable,
    resolve_hf_repo_snapshot,
)
from ..types import (
    MEMORY_MODE_DEFAULT,
    MEMORY_MODE_LOW,
    MEMORY_MODE_LOWEST,
    MEMORY_MODES,
    GeneratedAsset,
    GeneratedAudioTrack,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageToVideoRequest,
    ImageUpscaleRequest,
    MultiAngleRequest,
    ProviderAdapterInfo,
    ProviderModelInfo,
    VideoProgressEvent,
    VideoGenerationRequest,
    VisionBackendCapabilities,
)
from .base_backend import SOUNDTRACK_REQUEST_FIELDS, VisionBackend, raise_if_cancelled

logger = logging.getLogger(__name__)

MLX_GEN_RUNTIME = "mlx-gen"
MFLUX_PROVIDER = "mflux"
# The mlx-gen releases this AbstractVision runs: >=MLX_GEN_VERSION_FLOOR, <MLX_GEN_VERSION_CEILING.
MLX_GEN_VERSION_FLOOR = "0.40.1"
MLX_GEN_VERSION_CEILING = "0.41.0"
# On macOS: MLX 0.31.0 to 0.32.3 free a 3D convolution's padding buffer before the Metal command that reads it,
# which can end a run with an "invalid resource" error during a decode.
MLX_VERSION_FLOOR = "0.32.4"
WAN_TI2V_MODEL_KEY = "wan2.2-ti2v-5b"
WAN_T2V_A14B_MODEL_KEY = "wan2.2-t2v-a14b"
WAN_I2V_A14B_MODEL_KEY = "wan2.2-i2v-a14b"
SEEDVR2_3B_MODEL_KEY = "seedvr2-3b"
SEEDVR2_7B_MODEL_KEY = "seedvr2-7b"
# MiniMax-H3 (video with a stereo soundtrack). The three keys are MLX-Gen's catalog entries and
# pick the schedule: the base 50-step model, or the Turbo adapters (768p and 544p, 8 steps).
MINIMAX_H3_MODEL_KEY = "minimax-h3"
MINIMAX_H3_TURBO_MODEL_KEY = "minimax-h3-turbo"
MINIMAX_H3_TURBO_544P_MODEL_KEY = "minimax-h3-turbo-544p"
MINIMAX_H3_MODEL_KEYS = (MINIMAX_H3_MODEL_KEY, MINIMAX_H3_TURBO_MODEL_KEY, MINIMAX_H3_TURBO_544P_MODEL_KEY)
MINIMAX_H3_REPO_ID = "MiniMaxAI/MiniMax-H3"
WAN_DEFAULT_GUIDANCE = 5.0


@dataclass(frozen=True)
class WanVideoCanvas:
    """Default generation size of one Wan video model."""

    width: int
    height: int
    frames: int
    fps: int
    steps: int


# Default generation canvas per Wan video model: the ONE place these defaults
# live (the model table, capability discovery and the packaged capability
# registry all follow it). Chosen for speed and memory: a 832x480 canvas has
# about 45% of the tokens of 720p, so each denoising step is much faster and
# the run needs far less memory (measured, mlx-gen 0.38.0, tiled decode:
# TI2V-5B 16.6 GiB at 832x480x121 vs 25.4 GiB at 1280x704x121; A14B 38.4 GiB
# at 832x480x81 vs 53.5 GiB at 1280x720x81). Every request can override
# width, height, num_frames, fps and steps.
WAN_VIDEO_DEFAULT_CANVASES: Dict[str, WanVideoCanvas] = {
    # 5 s at 24 fps. Wan's reference TI2V size is 1280x704; 832x480 is the
    # smallest canvas AbstractVision accepts for it.
    WAN_TI2V_MODEL_KEY: WanVideoCanvas(width=832, height=480, frames=121, fps=24, steps=50),
    # 5 s at 16 fps, Wan's 480p size for the A14B experts.
    WAN_T2V_A14B_MODEL_KEY: WanVideoCanvas(width=832, height=480, frames=81, fps=16, steps=40),
    WAN_I2V_A14B_MODEL_KEY: WanVideoCanvas(width=832, height=480, frames=81, fps=16, steps=40),
}
WAN_TI2V_480P_FLOW_SHIFT = 3.0
WAN_TI2V_PRACTICAL_MIN_WIDTH = 832
WAN_TI2V_PRACTICAL_MIN_HEIGHT = 480
WAN_A14B_PROOF_WIDTH = 480
WAN_A14B_PROOF_HEIGHT = 240
# The MLX cache limit of mlx-gen's low-RAM mode (`--low-ram`,
# RuntimeMemory.DEFAULT_LOW_RAM_CACHE_LIMIT_BYTES), held for the length of a
# memory_mode="low" or "lowest" run, model load included, and restored afterwards;
# never above a limit already in force.
LOW_RAM_MLX_CACHE_LIMIT_BYTES = 1000**3


@dataclass(frozen=True)
class _MemoryPlan:
    """How one request runs for its memory_mode (see `MFluxVisionBackend._memory_plan`).

    "low" is mlx-gen's low-RAM mode (`--low-ram`) as each route's mlx-gen CLI applies
    it for one seed; "lowest" adds block streaming (`--stream-blocks`) and is only accepted
    on routes that publish `supports_block_streaming`. `releases_model` is the one rule for
    every family: the run frees part of the instance, which is then dropped when the run
    ends, whatever its outcome, and the next request loads the model again. A batch item is
    a run like any other.
    """

    mode: str
    low_ram: bool
    stream_blocks: bool
    releases_model: bool = False

    def metadata(self) -> Dict[str, Any]:
        # Recorded on every run, "default" included: AbstractCore reports `memory_mode_used` from it.
        return {
            "memory_mode": self.mode,
            "low_ram": self.low_ram,
            "stream_blocks": self.stream_blocks,
        }


# MLX's cache limit is process-wide while every backend instance runs on its own
# thread. Each low-RAM run holds a token; the tightest held limit applies, never above
# the limit found before the first token (an operator's cap is kept), which is
# restored when the last one ends.
_MLX_CACHE_LIMIT_LOCK = threading.Lock()
_MLX_CACHE_LIMIT_HOLDS: Dict[int, int] = {}
_MLX_CACHE_LIMIT_TOKENS = itertools.count(1)
_mlx_cache_limit_before_holds: Optional[int] = None


def _apply_mlx_gen_default_cache_limit() -> None:
    # mlx-gen caps the MLX cache once per process at its first weight load (RAM/8, 1-8 GiB),
    # unless a limit is already set, which it then treats as host-managed. A hold taken before
    # the first load would hide that default and hand MLX's uncapped library default back on
    # release, so the default is applied first and becomes the limit restored.
    try:
        from mflux.utils.runtime_memory import RuntimeMemory  # type: ignore
    except Exception:
        return
    RuntimeMemory.apply_default_cache_limit_once()


def _acquire_mlx_cache_limit(limit_bytes: int) -> int:
    global _mlx_cache_limit_before_holds
    with _MLX_CACHE_LIMIT_LOCK:
        token = next(_MLX_CACHE_LIMIT_TOKENS)
        first = not _MLX_CACHE_LIMIT_HOLDS
        if first:
            _apply_mlx_gen_default_cache_limit()
        _MLX_CACHE_LIMIT_HOLDS[token] = int(limit_bytes)
        held = min(_MLX_CACHE_LIMIT_HOLDS.values())
        previous = _set_mlx_cache_limit(held)
        if first:
            # MLX's setter returns the limit in force: it is restored at the end, and never raised.
            _mlx_cache_limit_before_holds = previous
        if _tightest_mlx_cache_limit() < held:
            _set_mlx_cache_limit(_tightest_mlx_cache_limit())
        return token


def _tightest_mlx_cache_limit() -> int:
    limits = list(_MLX_CACHE_LIMIT_HOLDS.values())
    if _mlx_cache_limit_before_holds is not None:
        limits.append(_mlx_cache_limit_before_holds)
    return min(limits)


def _held_mlx_cache_limit() -> int:
    with _MLX_CACHE_LIMIT_LOCK:
        return _tightest_mlx_cache_limit() if _MLX_CACHE_LIMIT_HOLDS else LOW_RAM_MLX_CACHE_LIMIT_BYTES


def _release_mlx_cache_limit(token: int) -> None:
    global _mlx_cache_limit_before_holds
    with _MLX_CACHE_LIMIT_LOCK:
        if _MLX_CACHE_LIMIT_HOLDS.pop(token, None) is None:
            return
        if _MLX_CACHE_LIMIT_HOLDS:
            _set_mlx_cache_limit(_tightest_mlx_cache_limit())
        elif _mlx_cache_limit_before_holds is not None:
            _set_mlx_cache_limit(_mlx_cache_limit_before_holds)
            _mlx_cache_limit_before_holds = None


def _set_mlx_cache_limit(limit_bytes: int) -> Optional[int]:
    """Set MLX's allocator cache limit and return the previous one.

    Returns None when this process has no MLX runtime (non-Apple test hosts
    driving a fake Wan model); there is no allocator cache to bound there.
    """
    try:
        import mlx.core as mx  # type: ignore
    except ImportError:
        return None
    return int(mx.set_cache_limit(int(limit_bytes)))


def _wan_vae_tiling_requested(value: Any) -> bool:
    """Resolve the ``vae_tiling`` request option (default on)."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "auto"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"vae_tiling must be true or false (got {value!r}).")


def _install_wan_tiled_decode(model: Any) -> Callable[[], None]:
    from .wan_vae_tiling import install_tiled_decode

    return install_tiled_decode(model.vae)


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx  # type: ignore
    except ImportError:
        return
    mx.clear_cache()


def _route_rows_support_block_streaming(
    route_capabilities: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    return any(
        bool(row.get("supports_block_streaming"))
        for rows in route_capabilities.values()
        for row in rows
    )


@dataclass(frozen=True)
class MlxGenMemoryFlags:
    """How one memory_mode runs on MLX-Gen: the engine's `low_ram` and `stream_blocks` flags, as its
    measured runs record them, and the name mlx-gen's own tables give the mode (`MINIMAX_H3_MODES`)."""

    low_ram: bool
    stream_blocks: bool
    engine_mode: str


# Each memory_mode as MLX-Gen runs it; `mlx_gen_memory_mode_of_run` reads it the other way.
MLX_GEN_MEMORY_MODE_FLAGS: Dict[str, MlxGenMemoryFlags] = {
    MEMORY_MODE_DEFAULT: MlxGenMemoryFlags(low_ram=False, stream_blocks=False, engine_mode="default"),
    MEMORY_MODE_LOW: MlxGenMemoryFlags(low_ram=True, stream_blocks=False, engine_mode="low_ram"),
    MEMORY_MODE_LOWEST: MlxGenMemoryFlags(low_ram=True, stream_blocks=True, engine_mode="stream_blocks"),
}


def mlx_gen_memory_mode_of_run(run: Mapping[str, Any]) -> Optional[str]:
    """The memory_mode of an MLX-Gen measured run (a `measured_runs` entry), from its flags; None
    when its flags are those of no mode."""
    flags = (bool(run.get("low_ram")), bool(run.get("stream_blocks")))
    for mode, mode_flags in MLX_GEN_MEMORY_MODE_FLAGS.items():
        if (mode_flags.low_ram, mode_flags.stream_blocks) == flags:
            return mode
    return None


def mlx_gen_memory_modes(*, family: Optional[str], streams_blocks: bool) -> Tuple[str, ...]:
    """The memory_mode values an MLX-Gen route of `family` honours.

    Every route runs mlx-gen's low-RAM mode ("low"). "lowest" adds block streaming: only on a route
    whose rows publish `supports_block_streaming` (`streams_blocks`), and never on SeedVR2, whose
    constructor takes no stream_blocks."""
    if streams_blocks and family != "seedvr2-upscale":
        return MEMORY_MODES
    return (MEMORY_MODE_DEFAULT, MEMORY_MODE_LOW)


def mlx_gen_route_task(row: Mapping[str, Any]) -> str:
    """The AbstractVision task (`text_to_video`) of an MLX-Gen capability row (`public_task`
    `text-to-video`); "" for a row without one."""
    return str(row.get("public_task") or "").strip().replace("-", "_")


def _lazy_import_mlx_gen_memory_saver() -> Any:
    try:
        from mflux.callbacks.instances.memory_saver import MemorySaver  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            f"memory_mode='low' needs mlx-gen>={MLX_GEN_VERSION_FLOOR} (its low-RAM MemorySaver). "
            'Install/upgrade it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    return MemorySaver

_LORA_METADATA_KEYS = (
    "lora_paths",
    "lora_scales",
    "lora_target_roles",
    "lora_application_reports",
    "lora_applied_file_count",
    "lora_applied_target_count",
)


def _progress_attr(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def _progress_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _progress_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _progress_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_progress_task(value: Any, default: Optional[str] = None) -> Optional[str]:
    raw = str(value or default or "").strip().lower().replace("-", "_")
    return raw or None


def _normalize_video_progress_event(event: Any) -> VideoProgressEvent:
    phase = str(_progress_attr(event, "phase", "running") or "running")
    frame = _progress_optional_int(_progress_attr(event, "frame", None))
    total_frames = _progress_optional_int(_progress_attr(event, "total_frames", None))
    step = _progress_optional_int(_progress_attr(event, "step", None))
    total_steps = _progress_optional_int(_progress_attr(event, "total_steps", None))
    task = _normalize_progress_task(_progress_attr(event, "task", None))
    timestep = _progress_optional_float(_progress_attr(event, "timestep", None))
    step_progress = _progress_optional_float(_progress_attr(event, "step_progress", None))
    event_progress = _progress_optional_float(_progress_attr(event, "progress", None))
    frame_progress = _progress_optional_float(_progress_attr(event, "frame_progress", None))
    if step_progress is None:
        step_progress = event_progress
    if step_progress is None and step is not None and total_steps and total_steps > 0:
        step_progress = max(0.0, min(1.0, float(step) / float(total_steps)))
    if frame_progress is None and frame is not None and total_frames and total_frames > 0:
        frame_progress = max(0.0, min(1.0, float(frame) / float(total_frames)))
    progress = event_progress if event_progress is not None else step_progress
    raw = {
        "phase": phase,
        "frame": frame,
        "total_frames": total_frames,
        "step": step,
        "total_steps": total_steps,
        "progress": progress,
        "step_progress": step_progress,
        "frame_progress": frame_progress,
        "task": task,
        "timestep": timestep,
    }
    return VideoProgressEvent(
        phase=phase,
        frame=frame,
        total_frames=total_frames,
        step=step,
        total_steps=total_steps,
        progress=progress,
        step_progress=step_progress,
        frame_progress=frame_progress,
        task=task,
        timestep=timestep,
        raw=raw,
    )


def _extract_lora_metadata(metadata: Mapping[str, Any] | Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in _LORA_METADATA_KEYS:
        value = metadata.get(key) if isinstance(metadata, dict) else None
        if value not in (None, [], (), {}):
            out[key] = value
    return out


def _pop_progress_callbacks(
    extra: Dict[str, Any],
) -> Tuple[List[Callable[[VideoProgressEvent], None]], Optional[Callable[[int, Optional[int]], None]]]:
    callbacks = [
        callback
        for callback in (
            extra.pop("on_progress", None),
            extra.pop("progress_event_callback", None),
            extra.pop("progress_callback", None),
        )
        if callable(callback)
    ]
    step_progress_callback = extra.pop("_step_progress_callback", None)
    if not callable(step_progress_callback):
        step_progress_callback = None
    return callbacks, step_progress_callback


@dataclass(frozen=True)
class _SavedVideo:
    data: bytes
    engine_metadata: Dict[str, Any]
    audio_wav: Optional[bytes] = None


class _ProgressRelay:
    """Delivers MLX-Gen progress events to a request's callbacks and checks its cancel_event.

    Contract: an exception raised by a host callback is logged (once per run) and does not stop
    the run; a request is stopped only through its `cancel_event`, which raises
    GenerationCancelledError at the next progress event (between denoising steps).
    """

    def __init__(
        self,
        extra: Dict[str, Any],
        *,
        task: str,
        cancel_event: Any = None,
        force_task: bool = False,
    ) -> None:
        self.callbacks, self.step_callback = _pop_progress_callbacks(extra)
        self.task = task
        self.force_task = force_task
        self.cancel_event = cancel_event
        self.cancelled = False
        self._callback_error_logged = False

    @property
    def active(self) -> bool:
        return bool(self.callbacks) or self.step_callback is not None or self.cancel_event is not None

    def __call__(self, raw_event: Any) -> None:
        event = _normalize_video_progress_event(raw_event)
        if self.force_task or event.task is None:
            event = replace(event, task=self.task)
        for callback in self.callbacks:
            self._deliver(callback, event)
        if self.step_callback is not None:
            current = event.step if event.step is not None else (event.frame or 0)
            total = event.total_steps if event.total_steps is not None else event.total_frames
            self._deliver(self.step_callback, current, total)
        self.check_cancelled()

    def check_cancelled(self) -> None:
        try:
            raise_if_cancelled(self.cancel_event)
        except GenerationCancelledError:
            self.cancelled = True
            raise

    def cancelled_error(self, cause: BaseException) -> Optional[GenerationCancelledError]:
        """The error to raise when `cause` ended a run this relay cancelled, else None."""
        if isinstance(cause, GenerationCancelledError):
            return cause
        if self.cancelled:
            error = GenerationCancelledError("The request was cancelled (cancel_event is set).")
            error.__cause__ = cause
            return error
        return None

    def _deliver(self, callback: Callable[..., Any], *args: Any) -> None:
        try:
            callback(*args)
        except Exception:
            if not self._callback_error_logged:
                self._callback_error_logged = True
                logger.exception(
                    "A progress callback raised; the run continues (set the request's cancel_event "
                    "to stop it)."
                )


@dataclass(frozen=True)
class _MFluxModelDef:
    key: str
    config_method: str
    family: str
    # None where the engine's capability rows decide (MiniMax-H3).
    default_steps: Optional[int]
    default_guidance: Optional[float]
    supports_negative_prompt: bool = False
    supports_guidance_override: bool = True
    image_edit_catalog_rank: int = 50
    # Video canvas defaults: only Wan video models set them, from
    # WAN_VIDEO_DEFAULT_CANVASES.
    default_width: Optional[int] = None
    default_height: Optional[int] = None
    default_frames: Optional[int] = None
    default_fps: Optional[int] = None
    default_guidance_2: Optional[float] = None
    default_flow_shift: Optional[float] = None


_MFLUX_MODELS: Dict[str, _MFluxModelDef] = {
    "flux2-klein-4b": _MFluxModelDef(
        key="flux2-klein-4b",
        config_method="flux2_klein_4b",
        family="flux2",
        default_steps=4,
        default_guidance=1.0,
        supports_negative_prompt=False,
        supports_guidance_override=False,
        image_edit_catalog_rank=40,
    ),
    "flux2-klein-9b": _MFluxModelDef(
        key="flux2-klein-9b",
        config_method="flux2_klein_9b",
        family="flux2",
        default_steps=4,
        default_guidance=1.0,
        supports_negative_prompt=False,
        supports_guidance_override=False,
        image_edit_catalog_rank=40,
    ),
    "flux2-klein-base-4b": _MFluxModelDef(
        key="flux2-klein-base-4b",
        config_method="flux2_klein_base_4b",
        family="flux2",
        default_steps=50,
        default_guidance=1.5,
        supports_negative_prompt=False,
        supports_guidance_override=True,
        image_edit_catalog_rank=40,
    ),
    "flux2-klein-base-9b": _MFluxModelDef(
        key="flux2-klein-base-9b",
        config_method="flux2_klein_base_9b",
        family="flux2",
        default_steps=50,
        default_guidance=1.5,
        supports_negative_prompt=False,
        supports_guidance_override=True,
        image_edit_catalog_rank=40,
    ),
    "bonsai-image-ternary": _MFluxModelDef(
        key="bonsai-image-ternary",
        config_method="bonsai_image_ternary",
        family="bonsai",
        default_steps=4,
        default_guidance=1.0,
        supports_negative_prompt=False,
        supports_guidance_override=False,
    ),
    "z-image": _MFluxModelDef(
        key="z-image",
        config_method="z_image",
        family="z-image",
        default_steps=50,
        default_guidance=3.5,
        supports_negative_prompt=True,
        supports_guidance_override=True,
    ),
    "z-image-turbo": _MFluxModelDef(
        key="z-image-turbo",
        config_method="z_image_turbo",
        family="z-image",
        default_steps=9,
        default_guidance=0.0,
        supports_negative_prompt=False,
        supports_guidance_override=False,
    ),
    "qwen-image": _MFluxModelDef(
        key="qwen-image",
        config_method="qwen_image",
        family="qwen",
        default_steps=20,
        default_guidance=3.5,
        supports_negative_prompt=True,
        supports_guidance_override=True,
    ),
    "qwen-image-edit": _MFluxModelDef(
        key="qwen-image-edit",
        config_method="qwen_image_edit",
        family="qwen-edit",
        default_steps=20,
        default_guidance=2.5,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        image_edit_catalog_rank=2,
    ),
    "qwen-image-edit-2511": _MFluxModelDef(
        key="qwen-image-edit-2511",
        config_method="qwen_image_edit",
        family="qwen-edit",
        default_steps=20,
        default_guidance=2.5,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        image_edit_catalog_rank=0,
    ),
    "qwen-image-edit-2509": _MFluxModelDef(
        key="qwen-image-edit-2509",
        config_method="qwen_image_edit",
        family="qwen-edit",
        default_steps=20,
        default_guidance=2.5,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        image_edit_catalog_rank=1,
    ),
    "ernie-image-turbo": _MFluxModelDef(
        key="ernie-image-turbo",
        config_method="ernie_image_turbo",
        family="ernie-image",
        default_steps=8,
        default_guidance=1.0,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        image_edit_catalog_rank=30,
    ),
    "fibo": _MFluxModelDef(
        key="fibo",
        config_method="fibo",
        family="fibo",
        default_steps=50,
        default_guidance=4.0,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        image_edit_catalog_rank=25,
    ),
    "fibo-lite": _MFluxModelDef(
        key="fibo-lite",
        config_method="fibo_lite",
        family="fibo",
        default_steps=8,
        default_guidance=1.0,
        supports_negative_prompt=True,
        supports_guidance_override=False,
        image_edit_catalog_rank=25,
    ),
    "fibo-edit": _MFluxModelDef(
        key="fibo-edit",
        config_method="fibo_edit",
        family="fibo-edit",
        default_steps=50,
        default_guidance=4.0,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        image_edit_catalog_rank=10,
    ),
    "fibo-edit-rmbg": _MFluxModelDef(
        key="fibo-edit-rmbg",
        config_method="fibo_edit_rmbg",
        family="fibo-edit",
        default_steps=10,
        default_guidance=4.0,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        image_edit_catalog_rank=11,
    ),
    SEEDVR2_3B_MODEL_KEY: _MFluxModelDef(
        key=SEEDVR2_3B_MODEL_KEY,
        config_method="seedvr2_3b",
        family="seedvr2-upscale",
        default_steps=1,
        default_guidance=1.0,
        supports_negative_prompt=False,
        supports_guidance_override=False,
    ),
    SEEDVR2_7B_MODEL_KEY: _MFluxModelDef(
        key=SEEDVR2_7B_MODEL_KEY,
        config_method="seedvr2_7b",
        family="seedvr2-upscale",
        default_steps=1,
        default_guidance=1.0,
        supports_negative_prompt=False,
        supports_guidance_override=False,
    ),
    WAN_TI2V_MODEL_KEY: _MFluxModelDef(
        key=WAN_TI2V_MODEL_KEY,
        config_method="wan2_2_ti2v_5b",
        family="wan-video",
        default_steps=WAN_VIDEO_DEFAULT_CANVASES[WAN_TI2V_MODEL_KEY].steps,
        default_guidance=WAN_DEFAULT_GUIDANCE,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        default_width=WAN_VIDEO_DEFAULT_CANVASES[WAN_TI2V_MODEL_KEY].width,
        default_height=WAN_VIDEO_DEFAULT_CANVASES[WAN_TI2V_MODEL_KEY].height,
        default_frames=WAN_VIDEO_DEFAULT_CANVASES[WAN_TI2V_MODEL_KEY].frames,
        default_fps=WAN_VIDEO_DEFAULT_CANVASES[WAN_TI2V_MODEL_KEY].fps,
        # The 480p flow shift for the 832x480 default; canvases above 832x480
        # keep mlx-gen's 720p default (5.0) unless flow_shift is set.
        default_flow_shift=WAN_TI2V_480P_FLOW_SHIFT,
    ),
    WAN_T2V_A14B_MODEL_KEY: _MFluxModelDef(
        key=WAN_T2V_A14B_MODEL_KEY,
        config_method="wan2_2_t2v_a14b",
        family="wan-video",
        default_steps=WAN_VIDEO_DEFAULT_CANVASES[WAN_T2V_A14B_MODEL_KEY].steps,
        default_guidance=4.0,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        default_width=WAN_VIDEO_DEFAULT_CANVASES[WAN_T2V_A14B_MODEL_KEY].width,
        default_height=WAN_VIDEO_DEFAULT_CANVASES[WAN_T2V_A14B_MODEL_KEY].height,
        default_frames=WAN_VIDEO_DEFAULT_CANVASES[WAN_T2V_A14B_MODEL_KEY].frames,
        default_fps=WAN_VIDEO_DEFAULT_CANVASES[WAN_T2V_A14B_MODEL_KEY].fps,
        default_guidance_2=3.0,
        default_flow_shift=3.0,
    ),
    WAN_I2V_A14B_MODEL_KEY: _MFluxModelDef(
        key=WAN_I2V_A14B_MODEL_KEY,
        config_method="wan2_2_i2v_a14b",
        family="wan-video",
        default_steps=WAN_VIDEO_DEFAULT_CANVASES[WAN_I2V_A14B_MODEL_KEY].steps,
        default_guidance=3.5,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        default_width=WAN_VIDEO_DEFAULT_CANVASES[WAN_I2V_A14B_MODEL_KEY].width,
        default_height=WAN_VIDEO_DEFAULT_CANVASES[WAN_I2V_A14B_MODEL_KEY].height,
        default_frames=WAN_VIDEO_DEFAULT_CANVASES[WAN_I2V_A14B_MODEL_KEY].frames,
        default_fps=WAN_VIDEO_DEFAULT_CANVASES[WAN_I2V_A14B_MODEL_KEY].fps,
        default_guidance_2=3.5,
        default_flow_shift=3.0,
    ),
    # MiniMax-H3 defaults (steps, canvas, frames, shifts) come from MLX-Gen's capability rows
    # at run time (`_h3_parameter_metadata`, `_normalize_h3_request_values`).
    **{
        key: _MFluxModelDef(
            key=key,
            config_method="",
            family="minimax-h3",
            default_steps=None,
            default_guidance=None,
            supports_negative_prompt=False,
            supports_guidance_override=False,
        )
        for key in MINIMAX_H3_MODEL_KEYS
    },
}


_KNOWN_MODEL_ALIASES: Dict[str, str] = {
    "black-forest-labs/flux.2-klein-4b": "flux2-klein-4b",
    "abstractframework/flux.2-klein-4b-4bit": "flux2-klein-4b",
    "abstractframework/flux.2-klein-4b-8bit": "flux2-klein-4b",
    "flux2-klein-4b": "flux2-klein-4b",
    "flux-klein-4b": "flux2-klein-4b",
    "klein-4b": "flux2-klein-4b",
    "black-forest-labs/flux.2-klein-9b": "flux2-klein-9b",
    "abstractframework/flux.2-klein-9b-4bit": "flux2-klein-9b",
    "abstractframework/flux.2-klein-9b-8bit": "flux2-klein-9b",
    "flux2-klein-9b": "flux2-klein-9b",
    "flux-klein-9b": "flux2-klein-9b",
    "klein-9b": "flux2-klein-9b",
    "black-forest-labs/flux.2-klein-base-4b": "flux2-klein-base-4b",
    "abstractframework/flux.2-klein-base-4b-4bit": "flux2-klein-base-4b",
    "abstractframework/flux.2-klein-base-4b-8bit": "flux2-klein-base-4b",
    "flux2-klein-base-4b": "flux2-klein-base-4b",
    "flux-klein-base-4b": "flux2-klein-base-4b",
    "klein-base-4b": "flux2-klein-base-4b",
    "black-forest-labs/flux.2-klein-base-9b": "flux2-klein-base-9b",
    "abstractframework/flux.2-klein-base-9b-4bit": "flux2-klein-base-9b",
    "abstractframework/flux.2-klein-base-9b-8bit": "flux2-klein-base-9b",
    "flux2-klein-base-9b": "flux2-klein-base-9b",
    "flux-klein-base-9b": "flux2-klein-base-9b",
    "klein-base-9b": "flux2-klein-base-9b",
    "tongyi-mai/z-image": "z-image",
    "abstractframework/z-image-4bit": "z-image",
    "abstractframework/z-image-8bit": "z-image",
    "z-image": "z-image",
    "zimage": "z-image",
    "tongyi-mai/z-image-turbo": "z-image-turbo",
    "abstractframework/z-image-turbo-4bit": "z-image-turbo",
    "abstractframework/z-image-turbo-8bit": "z-image-turbo",
    "z-image-turbo": "z-image-turbo",
    "zimage-turbo": "z-image-turbo",
    "abstractframework/qwen-image-edit-4bit": "qwen-image-edit",
    "abstractframework/qwen-image-edit-8bit": "qwen-image-edit",
    "qwen/qwen-image-edit": "qwen-image-edit",
    "qwen-image-edit-legacy": "qwen-image-edit",
    "qwen-image-edit-base": "qwen-image-edit",
    "abstractframework/qwen-image-edit-2511-4bit": "qwen-image-edit-2511",
    "abstractframework/qwen-image-edit-2511-8bit": "qwen-image-edit-2511",
    "qwen/qwen-image-edit-2511": "qwen-image-edit-2511",
    "qwen-image-edit": "qwen-image-edit-2511",
    "qwen-image-edit-2511": "qwen-image-edit-2511",
    "abstractframework/qwen-image-edit-2509-4bit": "qwen-image-edit-2509",
    "abstractframework/qwen-image-edit-2509-8bit": "qwen-image-edit-2509",
    "qwen/qwen-image-edit-2509": "qwen-image-edit-2509",
    "qwen-image-edit-2509": "qwen-image-edit-2509",
    "qwen/qwen-image": "qwen-image",
    "qwen/qwen-image-2512": "qwen-image",
    "abstractframework/qwen-image-2512-4bit": "qwen-image",
    "abstractframework/qwen-image-2512-8bit": "qwen-image",
    "abstractframework/qwen-image-4bit": "qwen-image",
    "abstractframework/qwen-image-8bit": "qwen-image",
    "qwen-image": "qwen-image",
    "qwen-image-2512": "qwen-image",
    "baidu/ernie-image-turbo": "ernie-image-turbo",
    "abstractframework/ernie-image-turbo-4bit": "ernie-image-turbo",
    "abstractframework/ernie-image-turbo-8bit": "ernie-image-turbo",
    "ernie-image-turbo": "ernie-image-turbo",
    "ernie-image": "ernie-image-turbo",
    "ernie": "ernie-image-turbo",
    "briaai/fibo": "fibo",
    "abstractframework/fibo-4bit": "fibo",
    "abstractframework/fibo-8bit": "fibo",
    "fibo": "fibo",
    "fibo-4bit": "fibo",
    "fibo-8bit": "fibo",
    "briaai/fibo-lite": "fibo-lite",
    "fibo-lite": "fibo-lite",
    "fibo_lite": "fibo-lite",
    "briaai/fibo-edit": "fibo-edit",
    "fibo-edit": "fibo-edit",
    "fiboedit": "fibo-edit",
    "briaai/fibo-edit-rmbg": "fibo-edit-rmbg",
    "fibo-edit-rmbg": "fibo-edit-rmbg",
    "fiboedit-rmbg": "fibo-edit-rmbg",
    "wan-ai/wan2.2-ti2v-5b-diffusers": WAN_TI2V_MODEL_KEY,
    "abstractframework/wan2.2-ti2v-5b-diffusers-8bit": WAN_TI2V_MODEL_KEY,
    "wan2.2-ti2v-5b-diffusers": WAN_TI2V_MODEL_KEY,
    "wan2.2-ti2v-5b-diffusers-8bit": WAN_TI2V_MODEL_KEY,
    "wan2.2-ti2v-5b": WAN_TI2V_MODEL_KEY,
    "wan2-2-ti2v-5b": WAN_TI2V_MODEL_KEY,
    "wan-ti2v": WAN_TI2V_MODEL_KEY,
    "wan-ai/wan2.2-t2v-a14b": WAN_T2V_A14B_MODEL_KEY,
    "wan-ai/wan2.2-t2v-a14b-diffusers": WAN_T2V_A14B_MODEL_KEY,
    "abstractframework/wan2.2-t2v-a14b-diffusers-8bit": WAN_T2V_A14B_MODEL_KEY,
    "abstractframework/wan2.2-t2v-a14b-diffusers-bf16": WAN_T2V_A14B_MODEL_KEY,
    "wan2.2-t2v-a14b": WAN_T2V_A14B_MODEL_KEY,
    "wan2.2-t2v-a14b-diffusers": WAN_T2V_A14B_MODEL_KEY,
    "wan2.2-t2v-a14b-diffusers-8bit": WAN_T2V_A14B_MODEL_KEY,
    "wan2.2-t2v-a14b-diffusers-bf16": WAN_T2V_A14B_MODEL_KEY,
    "wan2-2-t2v-a14b": WAN_T2V_A14B_MODEL_KEY,
    "wan-t2v-a14b": WAN_T2V_A14B_MODEL_KEY,
    "wan-a14b-t2v": WAN_T2V_A14B_MODEL_KEY,
    "wan-ai/wan2.2-i2v-a14b": WAN_I2V_A14B_MODEL_KEY,
    "wan-ai/wan2.2-i2v-a14b-diffusers": WAN_I2V_A14B_MODEL_KEY,
    "abstractframework/wan2.2-i2v-a14b-diffusers-8bit": WAN_I2V_A14B_MODEL_KEY,
    "abstractframework/wan2.2-i2v-a14b-diffusers-bf16": WAN_I2V_A14B_MODEL_KEY,
    "wan2.2-i2v-a14b": WAN_I2V_A14B_MODEL_KEY,
    "wan2.2-i2v-a14b-diffusers": WAN_I2V_A14B_MODEL_KEY,
    "wan2.2-i2v-a14b-diffusers-8bit": WAN_I2V_A14B_MODEL_KEY,
    "wan2.2-i2v-a14b-diffusers-bf16": WAN_I2V_A14B_MODEL_KEY,
    "wan2-2-i2v-a14b": WAN_I2V_A14B_MODEL_KEY,
    "wan-i2v-a14b": WAN_I2V_A14B_MODEL_KEY,
    "wan-a14b-i2v": WAN_I2V_A14B_MODEL_KEY,
    "wan-video": WAN_TI2V_MODEL_KEY,
    "wan": WAN_TI2V_MODEL_KEY,
    "prism-ml/bonsai-image-ternary-4b-mlx-2bit": "bonsai-image-ternary",
    "bonsai-image-ternary-4b-mlx-2bit": "bonsai-image-ternary",
    "bonsai-image-ternary": "bonsai-image-ternary",
    "bonsai-image-2bit": "bonsai-image-ternary",
    "bonsai-ternary": "bonsai-image-ternary",
    "bonsai-image": "bonsai-image-ternary",
    "bonsai": "bonsai-image-ternary",
    "bytedance-seed/seedvr2-3b": SEEDVR2_3B_MODEL_KEY,
    "bytedance-seed/seedvr2-7b": SEEDVR2_7B_MODEL_KEY,
    "abstractframework/seedvr2-3b-8bit": SEEDVR2_3B_MODEL_KEY,
    "abstractframework/seedvr2-3b-4bit": SEEDVR2_3B_MODEL_KEY,
    "abstractframework/seedvr2-7b-8bit": SEEDVR2_7B_MODEL_KEY,
    "abstractframework/seedvr2-7b-4bit": SEEDVR2_7B_MODEL_KEY,
    "seedvr2": SEEDVR2_3B_MODEL_KEY,
    "seedvr2-3b": SEEDVR2_3B_MODEL_KEY,
    "seedvr2-3b-8bit": SEEDVR2_3B_MODEL_KEY,
    "seedvr2-3b-4bit": SEEDVR2_3B_MODEL_KEY,
    "seedvr2_ema_3b_fp16": SEEDVR2_3B_MODEL_KEY,
    "seedvr2-ema-3b-fp16": SEEDVR2_3B_MODEL_KEY,
    "seedvr2-7b": SEEDVR2_7B_MODEL_KEY,
    "seedvr2-7b-8bit": SEEDVR2_7B_MODEL_KEY,
    "seedvr2-7b-4bit": SEEDVR2_7B_MODEL_KEY,
    "seedvr2_ema_7b_fp16": SEEDVR2_7B_MODEL_KEY,
    "seedvr2-ema-7b-fp16": SEEDVR2_7B_MODEL_KEY,
    # The packages hold the base weights; the schedule comes from base_model (see MINIMAX_H3_*).
    "minimaxai/minimax-h3": MINIMAX_H3_MODEL_KEY,
    "abstractframework/minimax-h3-8bit": MINIMAX_H3_MODEL_KEY,
    "abstractframework/minimax-h3-bf16": MINIMAX_H3_MODEL_KEY,
    "abstractframework/minimax-h3-4bit": MINIMAX_H3_MODEL_KEY,
    "minimax-h3": MINIMAX_H3_MODEL_KEY,
    "minimax-h3-8bit": MINIMAX_H3_MODEL_KEY,
    "minimax-h3-bf16": MINIMAX_H3_MODEL_KEY,
    "minimax-h3-4bit": MINIMAX_H3_MODEL_KEY,
    "minimax-h3-turbo": MINIMAX_H3_TURBO_MODEL_KEY,
    "h3-turbo": MINIMAX_H3_TURBO_MODEL_KEY,
    "minimax-h3-turbo-544p": MINIMAX_H3_TURBO_544P_MODEL_KEY,
    "h3-turbo-544p": MINIMAX_H3_TURBO_544P_MODEL_KEY,
}


_MFLUX_BASE_MODEL_REGISTRY_IDS: Dict[str, str] = {
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "flux2-klein-9b": "black-forest-labs/FLUX.2-klein-9B",
    "flux2-klein-base-4b": "black-forest-labs/FLUX.2-klein-base-4B",
    "flux2-klein-base-9b": "black-forest-labs/FLUX.2-klein-base-9B",
    "bonsai-image-ternary": "prism-ml/bonsai-image-ternary-4B-mlx-2bit",
    "qwen-image": "Qwen/Qwen-Image-2512",
    "qwen-image-edit": "Qwen/Qwen-Image-Edit",
    "qwen-image-edit-2511": "Qwen/Qwen-Image-Edit-2511",
    "qwen-image-edit-2509": "Qwen/Qwen-Image-Edit-2509",
    "z-image": "Tongyi-MAI/Z-Image",
    "z-image-turbo": "Tongyi-MAI/Z-Image-Turbo",
    "ernie-image-turbo": "baidu/ERNIE-Image-Turbo",
    "fibo": "briaai/FIBO",
    "fibo-lite": "briaai/Fibo-lite",
    "fibo-edit": "briaai/Fibo-Edit",
    "fibo-edit-rmbg": "briaai/Fibo-Edit-RMBG",
    SEEDVR2_3B_MODEL_KEY: "ByteDance-Seed/SeedVR2-3B",
    SEEDVR2_7B_MODEL_KEY: "ByteDance-Seed/SeedVR2-7B",
    WAN_TI2V_MODEL_KEY: "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    WAN_T2V_A14B_MODEL_KEY: "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    WAN_I2V_A14B_MODEL_KEY: "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    **{key: MINIMAX_H3_REPO_ID for key in MINIMAX_H3_MODEL_KEYS},
}


_MFLUX_BASE_MODEL_FALLBACK_TASKS: Dict[str, Tuple[str, ...]] = {
    "flux2-klein-4b": ("image_to_image", "text_to_image"),
    "flux2-klein-9b": ("image_to_image", "text_to_image"),
    "flux2-klein-base-4b": ("image_to_image", "text_to_image"),
    "flux2-klein-base-9b": ("image_to_image", "text_to_image"),
    "bonsai-image-ternary": ("text_to_image",),
    "qwen-image": ("text_to_image",),
    "qwen-image-edit": ("image_to_image",),
    "qwen-image-edit-2511": ("image_to_image",),
    "qwen-image-edit-2509": ("image_to_image",),
    "z-image": ("text_to_image",),
    "z-image-turbo": ("text_to_image",),
    "ernie-image-turbo": ("image_to_image", "text_to_image"),
    "fibo": ("image_to_image", "text_to_image"),
    "fibo-lite": ("image_to_image", "text_to_image"),
    "fibo-edit": ("image_to_image",),
    "fibo-edit-rmbg": ("image_to_image",),
    SEEDVR2_3B_MODEL_KEY: ("image_upscale",),
    SEEDVR2_7B_MODEL_KEY: ("image_upscale",),
    WAN_TI2V_MODEL_KEY: ("image_to_video", "text_to_video"),
    WAN_T2V_A14B_MODEL_KEY: ("text_to_video",),
    WAN_I2V_A14B_MODEL_KEY: ("image_to_video",),
    **{key: ("image_to_video", "text_to_video") for key in MINIMAX_H3_MODEL_KEYS},
}


_MFLUX_RUNTIME_ALLOWED_TASKS: Dict[str, Tuple[str, ...]] = {
    "flux2-klein-4b": ("image_to_image", "text_to_image"),
    "flux2-klein-9b": ("image_to_image", "text_to_image"),
    "flux2-klein-base-4b": ("image_to_image", "text_to_image"),
    "flux2-klein-base-9b": ("image_to_image", "text_to_image"),
    "bonsai-image-ternary": ("text_to_image",),
    "qwen-image": ("text_to_image",),
    "qwen-image-edit": ("image_to_image",),
    "qwen-image-edit-2511": ("image_to_image",),
    "qwen-image-edit-2509": ("image_to_image",),
    "z-image": ("text_to_image",),
    "z-image-turbo": ("text_to_image",),
    "ernie-image-turbo": ("image_to_image", "text_to_image"),
    "fibo": ("image_to_image", "text_to_image"),
    "fibo-lite": ("image_to_image", "text_to_image"),
    "fibo-edit": ("image_to_image",),
    "fibo-edit-rmbg": ("image_to_image",),
    SEEDVR2_3B_MODEL_KEY: ("image_upscale",),
    SEEDVR2_7B_MODEL_KEY: ("image_upscale",),
    WAN_TI2V_MODEL_KEY: ("image_to_video", "text_to_video"),
    WAN_T2V_A14B_MODEL_KEY: ("text_to_video",),
    WAN_I2V_A14B_MODEL_KEY: ("image_to_video",),
    **{key: ("image_to_video", "text_to_video") for key in MINIMAX_H3_MODEL_KEYS},
}


def _memory_mode_constraint(memory_modes: Sequence[str]) -> Dict[str, Any]:
    return {"enum": list(memory_modes)}


def _h3_parameter_metadata(row: Mapping[str, Any], *, memory_modes: Sequence[str]) -> Dict[str, Any]:
    # Every number comes from MLX-Gen's capability row for the entry's schedule.
    defaults: Dict[str, Any] = {
        "steps": row.get("default_steps"),
        "width": row.get("default_width"),
        "height": row.get("default_height"),
        "num_frames": row.get("default_frames"),
        "fps": row.get("output_fps"),
        "flow_shift": row.get("default_flow_shift"),
        "audio_shift": row.get("default_audio_shift"),
        "generate_audio": True,
        "memory_mode": MEMORY_MODE_DEFAULT,
    }
    multiple = row.get("dimension_multiple")
    constraints: Dict[str, Any] = {
        "memory_mode": _memory_mode_constraint(memory_modes),
        "width": {"multiple_of": multiple},
        "height": {"multiple_of": multiple},
        "num_frames": {
            "format": f"{row.get('frame_multiple')}n+{row.get('frame_remainder')}",
            "rounding": row.get("frame_rounding"),
            "min": row.get("min_frames"),
            "max": row.get("max_frames"),
            "values": list(_h3_frame_counts(row)),
        },
        "fps": {"const": row.get("output_fps")},
        "negative_prompt": {"supported": False},
        "guidance_scale": {"supported": False},
        "quantize": {"enum": list(row.get("validated_quantization_bits") or ())},
        "audio": {
            "channels": row.get("audio_channels"),
            "sample_rate": row.get("audio_sample_rate"),
        },
    }
    return {
        "parameter_defaults": defaults,
        "parameter_constraints": constraints,
        "parameters": {"defaults": defaults, "constraints": constraints},
    }


def _mflux_parameter_metadata(
    model_def: _MFluxModelDef,
    *,
    block_streaming: bool = False,
    route_capabilities: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
) -> Dict[str, Any]:
    memory_modes = mlx_gen_memory_modes(family=model_def.family, streams_blocks=bool(block_streaming))
    if model_def.family == "minimax-h3":
        rows = [row for task_rows in (route_capabilities or {}).values() for row in task_rows]
        row = next((r for r in rows if mlx_gen_route_task(r) == "text_to_video"), rows[0] if rows else {})
        return _h3_parameter_metadata(row, memory_modes=memory_modes)
    defaults: Dict[str, Any] = {
        "steps": int(model_def.default_steps),
        "memory_mode": MEMORY_MODE_DEFAULT,
    }
    constraints: Dict[str, Any] = {"memory_mode": _memory_mode_constraint(memory_modes)}
    if model_def.default_guidance is not None:
        defaults["guidance_scale"] = float(model_def.default_guidance)
    if model_def.family == "flux2":
        constraints["steps"] = {"min": 2}
    if not model_def.supports_guidance_override and model_def.default_guidance is not None:
        constraints["guidance_scale"] = {"const": float(model_def.default_guidance)}
    if not model_def.supports_negative_prompt:
        constraints["negative_prompt"] = {"supported": False}
    if model_def.family == "wan-video":
        defaults.update(
            {
                "width": model_def.default_width,
                "height": model_def.default_height,
                "fps": model_def.default_fps,
                "num_frames": model_def.default_frames,
            }
        )
        if model_def.default_guidance_2 is not None:
            defaults["guidance_2"] = model_def.default_guidance_2
        if model_def.default_flow_shift is not None:
            defaults["flow_shift"] = model_def.default_flow_shift
        constraints["num_frames"] = {"format": "4n+1"}
        constraints["flow_shift"] = {"min": 0.01}
        if model_def.key == WAN_TI2V_MODEL_KEY:
            constraints["width"] = {
                "multiple_of": 32,
                "practical_visual_min": WAN_TI2V_PRACTICAL_MIN_WIDTH,
            }
            constraints["height"] = {
                "multiple_of": 32,
                "practical_visual_min": WAN_TI2V_PRACTICAL_MIN_HEIGHT,
            }
        else:
            constraints["width"] = {
                "multiple_of": 16,
                "practical_validation_min": WAN_A14B_PROOF_WIDTH,
            }
            constraints["height"] = {
                "multiple_of": 16,
                "practical_validation_min": WAN_A14B_PROOF_HEIGHT,
            }
    if model_def.family == "seedvr2-upscale":
        defaults.update(
            {
                "resolution": "2x",
                "scale": 2,
                "softness": 0.25,
                "quantize": None,
            "vae_tiling": None,
            }
        )
        constraints.update(
            {
                "resolution": {"type": "integer_or_scale_factor"},
                "scale": {"min": 0.01},
                "softness": {"min": 0.0, "max": 1.0},
                "quantize": {"enum": [3, 4, 5, 6, 8, None]},
                "vae_tiling": {"type": "boolean"},
            }
        )
    return {
        "parameter_defaults": defaults,
        "parameter_constraints": constraints,
        "parameters": {
            "defaults": defaults,
            "constraints": constraints,
        },
    }


@dataclass(frozen=True)
class MFluxBackendConfig:
    """Config for the optional MLX-Gen backend.

    MLX-Gen is intentionally optional. The backend uses MLX-Gen's Python API
    in-process and expects local prepared model directories or cached Hugging
    Face snapshots.
    """

    model: Optional[str] = None
    base_model: Optional[str] = None
    model_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    lora_paths: Sequence[str] = field(default_factory=tuple)
    lora_scales: Sequence[float] = field(default_factory=tuple)
    lora_target_roles: Sequence[str] = field(default_factory=tuple)
    allow_download: bool = False
    default_width: int = 1024
    default_height: int = 1024


@lru_cache(maxsize=1)
def _require_runtime_versions() -> None:
    """Refuse an installed mlx-gen outside `>=MLX_GEN_VERSION_FLOOR,<MLX_GEN_VERSION_CEILING`, and on macOS an MLX
    below MLX_VERSION_FLOOR, as pip reads those specifiers (a pre-release of a floor is below it).

    Checked once, before the backend resolves a model to run, so before any answer about the model's files. A
    distribution that is not installed is not refused here: the engine import that follows reports it. The
    `packaging` library comes with huggingface_hub, which every MLX-Gen install has.
    """
    from importlib import metadata

    # (distribution, specifier, where the bound applies)
    bounds = [("mlx-gen", f">={MLX_GEN_VERSION_FLOOR},<{MLX_GEN_VERSION_CEILING}", "")]
    if sys.platform == "darwin":
        bounds.append(("mlx", f">={MLX_VERSION_FLOOR}", " on macOS"))
    for distribution, specifier, where in bounds:
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
        from packaging.specifiers import SpecifierSet

        if not SpecifierSet(specifier).contains(installed, prereleases=True):
            raise OptionalDependencyMissingError(
                f"The MLX-Gen backend needs {distribution}{specifier}{where}; {distribution} {installed} is "
                'installed. Install the matching versions with `pip install -U "abstractvision[mlx-gen]"`.'
            )


def _lazy_import_mflux() -> Tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import mlxgen as _mlxgen  # noqa: F401

        from mflux.models.common.config import ModelConfig  # type: ignore
        from mflux.models.common.download_policy import DownloadRequiredError  # type: ignore
        from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit  # type: ignore
        from mflux.models.z_image import ZImage, ZImageTurbo  # type: ignore
    except Exception as e:
        raise OptionalDependencyMissingError(
            "MLX-Gen backend requires the optional MLX-Gen runtime. "
            'Install it with `pip install "abstractvision[mlx-gen]"` (or the compatibility '
            "`abstractvision[mflux]` extra)."
        ) from e
    return ModelConfig, DownloadRequiredError, Flux2Klein, Flux2KleinEdit, ZImage, ZImageTurbo


def _lazy_import_mlx_gen_task_inference() -> Tuple[Any, Any]:
    try:
        from mflux.task_inference import get_model_capabilities, resolve_generation_plan  # type: ignore
    except Exception as e:
        raise OptionalDependencyMissingError(
            "MLX-Gen capability planning requires a recent MLX-Gen runtime. "
            'Install/upgrade it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    return get_model_capabilities, resolve_generation_plan


def _lazy_import_mflux_qwen() -> Tuple[Any, Any]:
    try:
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage  # type: ignore
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit  # type: ignore
    except Exception as e:
        raise OptionalDependencyMissingError(
            "MLX-Gen Qwen backend requires a recent MLX-Gen runtime. "
            'Install/upgrade it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    return QwenImage, QwenImageEdit


def _lazy_import_mflux_ernie() -> Any:
    try:
        from mflux.models.ernie_image import ErnieImageTurbo  # type: ignore
    except Exception as e:
        raise OptionalDependencyMissingError(
            f"MLX-Gen ERNIE backend requires mlx-gen>={MLX_GEN_VERSION_FLOOR}. "
            'Install/upgrade it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    return ErnieImageTurbo


def _lazy_import_mflux_fibo() -> Tuple[Any, Any]:
    try:
        from mflux.models.fibo.variants.edit import FIBOEdit  # type: ignore
        from mflux.models.fibo.variants.txt2img.fibo import FIBO  # type: ignore
    except Exception as e:
        raise OptionalDependencyMissingError(
            f"MLX-Gen FIBO backend requires mlx-gen>={MLX_GEN_VERSION_FLOOR}. "
            'Install/upgrade it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    return FIBO, FIBOEdit


def _lazy_import_mflux_bonsai() -> Any:
    try:
        from mflux.models.bonsai_image.variants import BonsaiImage  # type: ignore
    except Exception as e:
        raise OptionalDependencyMissingError(
            f"MLX-Gen Bonsai Image generation requires mlx-gen>={MLX_GEN_VERSION_FLOOR}. "
            'Install/upgrade it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    return BonsaiImage


def _lazy_import_mflux_wan() -> Any:
    try:
        from mflux.models.wan.variants import Wan2_2_TI2V  # type: ignore
    except Exception as e:
        raise OptionalDependencyMissingError(
            f"MLX-Gen Wan video generation requires mlx-gen>={MLX_GEN_VERSION_FLOOR}. "
            'Install/upgrade it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    return Wan2_2_TI2V


def _mlx_gen_quantize_choices() -> Tuple[int, ...]:
    try:
        from mflux.cli.defaults import defaults  # type: ignore
    except ImportError as e:
        raise OptionalDependencyMissingError(
            'Load-time quantize needs the MLX-Gen runtime: `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    return tuple(sorted(int(value) for value in defaults.QUANTIZE_CHOICES))


# The catalog `source` of the packages AbstractFramework prepares with MLX-Gen.
_ABSTRACTFRAMEWORK_MLX_GEN_SOURCE = "abstractframework-mlx-gen"


def _is_prepared_mlx_gen_package(location: Any, selector: Any) -> bool:
    """Whether the model is a prepared package, whose stored weights MLX-Gen loads as they are.

    The one answer behind `quantize_bits()` and the load-time `quantize` refusal. `location` is
    the model directory when known, `selector` the configured model. On disk, MLX-Gen's own shard
    metadata decides (`_engine_reports_prepared`); a repository the catalog lists (a cache
    snapshot's repo id, else the selector) decides by its catalog fields. A folder name never does.
    """
    path = Path(str(location)).expanduser() if location else None
    local = path is not None and path.is_dir()
    if local and _engine_reports_prepared(path):
        return True
    names = [_hf_cache_repo_id(path) if local else None]
    if selector and not Path(str(selector)).expanduser().exists():
        names.append(str(selector))
    for name in names:
        preset = _preset_for(name) if name else None
        if preset is not None:
            # Written by `mlxgen prepare` (AbstractFramework packages, 16-bit ones included), or
            # stored below 16 bits (the prepacked 2-bit Bonsai: no MLX-Gen metadata in its shards).
            prepared_source = preset.source == _ABSTRACTFRAMEWORK_MLX_GEN_SOURCE
            return prepared_source or int(preset.quantization_bits or 16) < 16
    return False


def _engine_reports_prepared(path: Path) -> bool:
    # MLX-Gen loads a component as stored when its first shard's safetensors header carries the
    # metadata `mlxgen prepare` writes (StreamingWeightLoader.is_prepared; WeightLoader applies the
    # same test). Only headers are read. A shard the engine cannot read is not a prepared one; the
    # load reports it.
    try:
        StreamingWeightLoader, header_error = _lazy_import_mlx_gen_weight_loader()
    except OptionalDependencyMissingError:
        return False
    for component in sorted(child for child in path.iterdir() if child.is_dir()):
        try:
            if StreamingWeightLoader.is_prepared(component):
                return True
        except (header_error, OSError, ValueError):
            continue
    return False


def _lazy_import_mlx_gen_weight_loader() -> Tuple[Any, Any]:
    try:
        from mflux.models.common.weights.loading.streaming_weight_loader import (  # type: ignore
            StreamingWeightLoader,
        )
        from safetensors import SafetensorError  # type: ignore
    except ImportError as e:
        raise OptionalDependencyMissingError(
            f"Reading MLX-Gen package metadata requires mlx-gen>={MLX_GEN_VERSION_FLOOR}."
        ) from e
    return StreamingWeightLoader, SafetensorError


def _json_safe_route_value(value: Any) -> Any:
    # MLX-Gen rows hold dataclasses (prompt sections, measured runs) and tuples.
    import dataclasses

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe_route_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe_route_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_route_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {str(key): _json_safe_route_value(item) for key, item in vars(value).items()}
    return str(value)


_PROGRESS_EXTRA_KEYS = frozenset(
    {"on_progress", "progress_event_callback", "progress_callback", "_step_progress_callback"}
)
# `extra` keys every route accepts: the progress callbacks, and what `resolve_memory_request`
# reads and strips (the memory mode).
_ROUTING_EXTRA_KEYS = _PROGRESS_EXTRA_KEYS | {"memory_mode"}
# The values an upscale request's `quantize` reads as "load unquantized", compared as `str(value).strip().lower()`
# (so 0 and False too): the upscale route's published spellings. They apply to that route only.
_UPSCALE_UNQUANTIZED_SPELLINGS = frozenset({"", "0", "none", "false", "off", "no"})
_PRELOAD_TASK_ORDER = ("text_to_image", "image_to_image", "text_to_video", "image_to_video", "image_upscale")
# MLX's random key is a 64-bit unsigned seed.
_MAX_SEED = 2**64 - 1


def _request_int(name: str, value: Any, *, label: str) -> int:
    """`value` as an int, or ValueError naming the field: ints, integral floats, decimal strings."""
    if isinstance(value, bool):
        raise ValueError(f"{label} {name} must be an integer, got {value!r}.")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{label} {name} must be an integer, got {value!r}.")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} {name} must be an integer, got {value!r}.") from None


def _request_number(name: str, value: Any, *, label: str) -> float:
    """`value` as a finite float, or ValueError naming the field."""
    if isinstance(value, bool):
        raise ValueError(f"{label} {name} must be a number, got {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} {name} must be a number, got {value!r}.") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} {name} must be a finite number, got {value!r}.")
    return number


def _request_seed(value: Any, *, label: str) -> Optional[int]:
    if value is None:
        return None
    seed = _request_int("seed", value, label=label)
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"{label} seed must lie in [0, 2**64 - 1], got {seed}.")
    return seed


def _checked_keyframe_size(image: Any) -> Tuple[int, int]:
    """Open and decode a conditioning image (any bytes-like object) before any weights load; its (width, height)."""
    try:
        data = memoryview(image).tobytes()
    except TypeError:
        raise ValueError(f"image must be the image file's bytes, got {type(image).__name__}.") from None
    if not data:
        raise ValueError("image is empty (0 bytes).")
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(BytesIO(data)) as img:
            img.load()
            return int(img.width), int(img.height)
    except UnidentifiedImageError:
        raise ValueError("image could not be decoded as a picture: not an image format Pillow reads.") from None
    except (OSError, ValueError, SyntaxError) as e:
        raise ValueError(f"image could not be decoded as a picture: {e}") from None


def _wav_codec(wav: Optional[bytes]) -> Optional[str]:
    """The PCM codec of a WAV file (pcm_s16le for MLX-Gen's), read from its header."""
    if not wav:
        return None
    import wave

    try:
        with wave.open(BytesIO(wav), "rb") as handle:
            bits = int(handle.getsampwidth()) * 8
    except (wave.Error, EOFError):
        return None
    return "pcm_u8" if bits == 8 else f"pcm_s{bits}le"


def _public_model_name(value: Any) -> Optional[str]:
    """What metadata reports for a model or adapter location: never an absolute local path.

    A Hugging Face cache path becomes its repo id (`org/name`, and `org/name:file` for a file in
    it), another local path its last component; a repo id or name is returned unchanged.
    """
    if value is None:
        return None
    text = str(value)
    path = Path(text)
    if not path.is_absolute():
        return text
    parts = path.parts
    for index, part in enumerate(parts):
        if part.startswith("models--") and index + 2 < len(parts) and parts[index + 1] == "snapshots":
            repo_id = part[len("models--") :].replace("--", "/", 1)
            inner = parts[index + 3 :]
            return f"{repo_id}:{'/'.join(inner)}" if inner else repo_id
    return path.name


# Engine metadata keys naming files AbstractVision writes for one run and deletes after it: the source image or
# first frame, an edit's image list (a reference given as a path is listed with them), the mask, the clip and its
# audio sidecar.
_TEMPORARY_PATH_METADATA_KEYS = ("image_path", "image_paths", "masked_image_path", "video_path", "audio_sidecar_path")
# The fields that carry a local path in the metadata mlx-gen 0.40.1 writes on the routes
# AbstractVision runs (GeneratedImage._get_metadata, GeneratedVideo.build_metadata, LoRAFileReport,
# VideoUtil's video_health, and the `original_*` copies an upscale takes from its input's
# metadata), as positions: a key, "[]" for each item of a list, then a key inside it.
_PATH_METADATA_FIELDS = (
    ("model",),
    ("lora_paths", "[]"),
    ("lora_application_reports", "[]", "requested_path"),
    ("lora_application_reports", "[]", "resolved_path"),
    ("original_model",),
    ("original_lora_paths", "[]"),
    ("original_lora_application_reports", "[]", "requested_path"),
    ("original_lora_application_reports", "[]", "resolved_path"),
    ("video_health", "file", "source"),
)


def _public_engine_metadata(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """MLX-Gen's metadata without local paths and without the temporaries AbstractVision wrote.

    Each field of `_PATH_METADATA_FIELDS` is reported through `_public_model_name`; every other
    value is kept as the engine wrote it.
    """
    out: Dict[str, Any] = {
        key: value for key, value in metadata.items() if key not in _TEMPORARY_PATH_METADATA_KEYS
    }
    for position in _PATH_METADATA_FIELDS:
        out = _public_path_field(out, position)
    return out


def _public_path_field(value: Any, position: Tuple[str, ...]) -> Any:
    if not position:
        return _public_model_name(value) if isinstance(value, str) else value
    head, rest = position[0], position[1:]
    if head == "[]":
        if not isinstance(value, (list, tuple)):
            return value
        return [_public_path_field(item, rest) for item in value]
    if isinstance(value, Mapping) and head in value:
        return {**value, head: _public_path_field(value[head], rest)}
    return value


def _public_lora_adapters(adapters: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {**item, "source": _public_model_name(item.get("source"))}
        for item in serialize_lora_adapters(adapters)
    ]


def _h3_aligned_frames(requested: int, row: Mapping[str, Any]) -> int:
    # MLX-Gen rounds a frame count up to its grid (`frame_rounding: up`); the grid is the row's.
    multiple = int(row.get("frame_multiple") or 1)
    remainder = int(row.get("frame_remainder") or 0)
    if requested <= remainder:
        return remainder
    steps = -(-(requested - remainder) // multiple)
    return remainder + steps * multiple


def _h3_frame_counts(row: Mapping[str, Any]) -> Tuple[int, ...]:
    multiple = int(row.get("frame_multiple") or 1)
    lowest = int(row.get("min_frames") or 1)
    highest = int(row.get("max_frames") or lowest)
    return tuple(range(_h3_aligned_frames(lowest, row), highest + 1, multiple))


def _h3_audio_track(engine: Mapping[str, Any], wav: Optional[bytes]) -> Optional[GeneratedAudioTrack]:
    if not engine.get("audio_present"):
        return None
    muxed = bool(engine.get("audio_muxed"))
    return GeneratedAudioTrack(
        sample_rate=int(engine.get("audio_sample_rate") or 0),
        channels=int(engine.get("audio_channels") or 0),
        duration_seconds=float(engine.get("audio_duration_seconds") or 0.0),
        muxed=muxed,
        codec=engine.get("audio_codec") if muxed else _wav_codec(wav),
        data=None if muxed else wav,
        mime_type=None if muxed else ("audio/wav" if wav else None),
    )


def _explicit_download_model(repo_id: str) -> Optional[Any]:
    registry = _mflux_capability_registry()
    return registry.explicit_download_model(repo_id) if registry is not None else None


def _explicit_download_note(repo_id: str, base_model: Optional[str]) -> str:
    """Why generation does not download `repo_id` and the command that does; "" when it may."""
    spec = _explicit_download_model(repo_id)
    if spec is None:
        return ""
    licence = spec.license_name or "its licence"
    return (
        f" {spec.model_id} weights are never downloaded during generation, even with "
        f"allow_download: the catalog marks {repo_id} for explicit download only ({licence}). "
        "Accept the licence (on the repository page while logged in to Hugging Face, for a "
        f"gated repository), then download explicitly: {_download_instruction(repo_id, base_model)}."
    )


def _download_instruction(repo_id: str, base_model: Optional[str] = None) -> str:
    """The `abstractvision download` command that fetches exactly `repo_id` (`download_command`), with
    the schedule when it loads an adapter; the repository page when no form of the command selects it."""
    try:
        adapter = _mlx_gen_schedule_adapter(base_model) if base_model else None
    except (ValueError, OptionalDependencyMissingError):
        adapter = None
    if adapter:
        try:
            return (
                f"`{download_command(repo_id, engine='mlx-gen', base_model=base_model)}` "
                "(which also caches the adapter that schedule loads)"
            )
        except ValueError:
            pass  # not a schedule of this repository: the command without it
    try:
        return f"`{download_command(repo_id, engine='mlx-gen')}`"
    except ValueError:
        return (
            f"no `abstractvision download` command selects {repo_id} on this host; "
            f"fetch it from https://huggingface.co/{repo_id}"
        )


def _mlx_gen_schedule_adapter(base_model: str) -> Optional[str]:
    """The adapter an MLX-Gen schedule attaches on its own (`hf:<repo>/<file>`), or None.

    The MiniMax-H3 Turbo schedules (`minimax-h3-turbo`, `minimax-h3-turbo-544p`) attach a
    lightx2v Turbo LoRA; the value comes from mlx-gen's catalog entry for the schedule."""

    key = _infer_base_model(base_model)
    if not key or key not in _MFLUX_MODELS:
        raise ValueError(
            f"Unknown MLX-Gen base model {base_model!r}. Supported: {', '.join(sorted(_MFLUX_MODELS))}"
        )
    if _MFLUX_MODELS[key].family != "minimax-h3":
        return None
    try:
        from mflux.models.common.config import ModelConfig  # type: ignore
    except ImportError as e:
        raise OptionalDependencyMissingError(
            f"Reading the {key} schedule requires mlx-gen>={MLX_GEN_VERSION_FLOOR}. "
            'Install it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    adapter = ModelConfig.from_name(key).transformer_overrides.get("turbo_lora")
    return str(adapter) if adapter else None


def resolve_mlx_gen_schedule_adapter(base_model: str, *, download: bool = False) -> Optional[Tuple[str, Optional[str]]]:
    """(adapter, local path) for the adapter a schedule attaches; None when it attaches none.

    The path is None when the adapter is not cached. `download=True` fetches it into mlx-gen's
    LoRA cache, as `mlxgen download --base-model <schedule>` does; generation never does."""

    adapter = _mlx_gen_schedule_adapter(base_model)
    if adapter is None:
        return None
    try:
        from mflux.models.common.download_policy import allow_downloads  # type: ignore
        from mflux.models.common.resolution.lora_resolution import LoraResolution  # type: ignore
    except Exception as e:
        raise OptionalDependencyMissingError(
            f"Resolving the {base_model} adapter requires mlx-gen>={MLX_GEN_VERSION_FLOOR}. "
            'Install it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    if download:
        with allow_downloads():
            return adapter, str(LoraResolution.resolve(adapter))
    try:
        return adapter, str(LoraResolution.resolve(adapter))
    except Exception:
        return adapter, None


def _lazy_import_mflux_minimax_h3() -> Any:
    try:
        from mflux.models.minimax_h3.variants.minimax_h3 import MiniMaxH3  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            f"MLX-Gen MiniMax-H3 video generation requires mlx-gen>={MLX_GEN_VERSION_FLOOR}. "
            'Install/upgrade it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    return MiniMaxH3


def _lazy_import_mlx_gen_h3_planning() -> Any:
    """MLX-Gen's own MiniMax-H3 rules a request is checked against before any weights load.

    `compose_prompt` (its verdict on a prompt section given twice), the aspect-ratio window, the
    adapter check for AdaLN projections, and the load-time quantization widths.
    """
    try:
        from mflux.models.minimax_h3.latent_creator import h3_layout  # type: ignore
        from mflux.models.minimax_h3.minimax_h3_initializer import MiniMaxH3Initializer  # type: ignore
        from mflux.models.minimax_h3.variants.minimax_h3 import compose_prompt  # type: ignore
        from mflux.models.minimax_h3.weights.h3_weight_definition import (  # type: ignore
            MiniMaxH3WeightDefinition,
        )
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            f"MLX-Gen MiniMax-H3 video generation requires mlx-gen>={MLX_GEN_VERSION_FLOOR}. "
            'Install/upgrade it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    from types import SimpleNamespace

    return SimpleNamespace(
        compose_prompt=compose_prompt,
        min_aspect_ratio=float(h3_layout.MIN_ASPECT_RATIO),
        max_aspect_ratio=float(h3_layout.MAX_ASPECT_RATIO),
        adapter_targets_modulation=MiniMaxH3Initializer.adapter_targets_modulation,
        quantization_bits=tuple(int(bits) for bits in MiniMaxH3WeightDefinition.QUANTIZATION_BITS),
    )


def _lazy_import_mflux_seedvr2() -> Tuple[Any, Any, Any, Any]:
    try:
        from mflux.models.common.config import ModelConfig  # type: ignore
        from mflux.models.common.vae.tiling_config import TilingConfig  # type: ignore
        from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2  # type: ignore
        from mflux.utils.scale_factor import ScaleFactor  # type: ignore
    except Exception as e:
        raise OptionalDependencyMissingError(
            f"MLX-Gen SeedVR2 image upscaling requires mlx-gen>={MLX_GEN_VERSION_FLOOR}. "
            'Install/upgrade it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    return ModelConfig, SeedVR2, ScaleFactor, TilingConfig


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _infer_bits_from_text(value: Any) -> Optional[int]:
    s = _norm(value)
    if not s:
        return None
    if "2bit" in s or "-q2" in s or s.endswith("q2"):
        return 2
    if "1bit" in s or "-q1" in s or s.endswith("q1"):
        return 1
    if "4bit" in s or "-q4" in s or s.endswith("q4"):
        return 4
    if "8bit" in s or "-q8" in s or "q8-0" in s or s.endswith("q8"):
        return 8
    return None


def _looks_like_path(value: str) -> bool:
    s = str(value or "").strip()
    return s.startswith(("/", "./", "../", "~")) or "\\" in s


def _has_model_files(path: Path) -> bool:
    try:
        return path.exists() and path.is_dir() and any(path.rglob("*.safetensors"))
    except Exception:
        return False


def _has_incomplete_markers(path: Path) -> bool:
    try:
        return any(path.rglob("*.incomplete"))
    except Exception:
        return False


def _is_partial_model_tree(path: Path) -> bool:
    parts = [_norm(part) for part in path.parts]
    if any(".partial" in part or part.endswith("-partial") for part in parts):
        return True
    return _has_incomplete_markers(path)


def _is_incompatible_model_tree(path: Path) -> bool:
    return any(".incompatible" in _norm(part) for part in path.parts)


def _looks_like_mflux_packaged_repo(value: Any) -> bool:
    s = _norm(value)
    return any(
        token in s
        for token in ("mlx-gen", "mflux", "8bit", "q8", "4bit", "q4", "quant", "seedvr2")
    )


def _is_mlx_gen_download_required(exc: BaseException) -> bool:
    # What MLX-Gen raises for weights, tokenizers and adapters it would have to download.
    try:
        from mflux.models.common.download_policy import DownloadRequiredError  # type: ignore
    except ImportError:
        return False
    return isinstance(exc, DownloadRequiredError)


def _wrap_mlx_gen_download_required(exc: BaseException) -> OptionalDependencyMissingError:
    parts = ["MLX-Gen model files are missing; generation does not download files implicitly."]
    message = str(exc).strip()
    if message:
        parts.append(message)
    download_command = str(getattr(exc, "download_command", "") or "").strip()
    prepare_command = str(getattr(exc, "prepare_command", "") or "").strip()
    if download_command:
        parts.append(f"Download first: {download_command}")
    if prepare_command:
        parts.append(f"Prepare a reusable local folder: {prepare_command}")
    return OptionalDependencyMissingError("\n".join(parts))


def _repo_id_from_cache_dir_name(name: str) -> Optional[str]:
    text = str(name or "").strip()
    if not text.startswith("models--"):
        return None
    tail = text[len("models--") :]
    if "--" not in tail:
        return None
    org, repo = tail.split("--", 1)
    if not org or not repo:
        return None
    return f"{org}/{repo}"


def _display_repo_id(repo_id: str) -> str:
    text = str(repo_id or "").strip()
    for token in (".incompatible", ".partial"):
        if token in text:
            text = text.split(token, 1)[0]
    return text


def _cache_root(configured: Optional[str]) -> str:
    return str(Path(configured).expanduser()) if configured else str(default_hf_cache_root())


@dataclass(frozen=True)
class _DiscoveredMFluxModel:
    key: str
    snapshot_dir: Path
    repo_id: Optional[str]
    source_label: str
    source_detail: str


def _mflux_model_roots(model_dir: Optional[str]) -> List[Tuple[str, Path]]:
    roots: List[Tuple[str, Path]] = [
        (
            "configured model dir" if model_dir else "legacy model dir",
            Path(model_dir).expanduser() if model_dir else default_legacy_model_root(),
        )
    ]
    for label, root in framework_local_model_roots():
        duplicate = False
        for _, existing in roots:
            try:
                if existing.resolve() == root.resolve():
                    duplicate = True
                    break
            except Exception:
                if existing == root:
                    duplicate = True
                    break
        if not duplicate:
            roots.append((label, root))
    return roots


def _local_source_label(root_label: str) -> str:
    label = _norm(root_label)
    if "quarantined" in label:
        return "quarantined local model dir"
    if "runtime" in label or "untracked" in label:
        return "local model dir"
    if "configured" in label:
        return "configured model dir"
    return "legacy model dir"


def _candidate_priority(repo_id: Optional[str]) -> int:
    s = _norm(repo_id)
    if not s:
        return 99
    if "seedvr2" in s and ("8bit" in s or "q8" in s):
        return 0
    if "seedvr2" in s and ("4bit" in s or "q4" in s):
        return 1
    if s.startswith("abstractframework/") and ("4bit" in s or "q4" in s):
        return 0
    if s.startswith("abstractframework/") and ("8bit" in s or "q8" in s):
        return 1
    if s.endswith("-mlx-8bit") or "mflux-8bit" in s or "-mlx-q8" in s or "-8bit-mlx" in s:
        return 3
    if "mflux" in s or "mlx" in s:
        return 5
    if "q4" in s or "4bit" in s:
        return 6
    return 7


def _candidate_repo_ids_for_preset(preset: Any) -> Tuple[str, ...]:
    if (
        str(getattr(preset, "target", "") or "").strip().lower() == "mlx"
        and str(getattr(preset, "engine", "") or "").strip().lower().replace("_", "-")
        in {"mlx-gen", "mflux"}
        and str(getattr(preset, "source", "") or "").strip().lower() == _ABSTRACTFRAMEWORK_MLX_GEN_SOURCE
    ):
        repo_id = str(getattr(preset, "repo_id", "") or "").strip()
        return (repo_id,) if looks_like_hf_repo_id(repo_id) else ()
    out: List[str] = []
    for value in (preset.repo_id, *(preset.aliases or ())):
        text = str(value or "").strip()
        if not looks_like_hf_repo_id(text):
            continue
        if text not in out:
            out.append(text)
    return tuple(out)


def _configured_hf_cache_roots(cache_dir: Optional[str]) -> List[Tuple[str, Path]]:
    if cache_dir:
        return [("configured cache", Path(cache_dir).expanduser())]
    return hf_cache_roots(extra_roots=framework_hf_cache_roots())


def _configured_adapter_cache_roots(cache_dir: Optional[str]) -> List[Tuple[str, Path]]:
    roots = list(_configured_hf_cache_roots(cache_dir))
    lora_cache_env = str(Path(os.environ.get("MFLUX_CACHE_DIR", "~/.cache/mflux")).expanduser())
    roots.append(("mlx-gen lora cache", Path(lora_cache_env).expanduser() / "loras"))
    deduped: List[Tuple[str, Path]] = []
    for label, root in roots:
        try:
            resolved = root.expanduser().resolve()
        except Exception:
            resolved = root.expanduser()
        duplicate = False
        for _, existing in deduped:
            try:
                if existing.resolve() == resolved:
                    duplicate = True
                    break
            except Exception:
                if existing == resolved:
                    duplicate = True
                    break
        if not duplicate:
            deduped.append((label, root))
    return deduped


def _adapter_repo_snapshot_dirs(repo_dir: Path) -> List[Path]:
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return []
    out: List[Path] = []
    ref = repo_dir / "refs" / "main"
    try:
        if ref.is_file():
            snap = snapshots_dir / ref.read_text(encoding="utf-8").strip()
            if snap.is_dir():
                out.append(snap)
    except Exception:
        pass
    try:
        for entry in snapshots_dir.iterdir():
            if entry.is_dir() and entry not in out:
                out.append(entry)
    except Exception:
        pass
    return out


def _read_card_frontmatter(readme_path: Path) -> Dict[str, Any]:
    """The top-level keys of a model card's YAML front matter; empty without one, or without PyYAML.

    PyYAML comes with huggingface_hub, which every MLX-Gen install has.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    try:
        lines = readme_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return {}
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end is None:
        return {}
    try:
        payload = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _frontmatter_list(payload: Mapping[str, Any], key: str) -> Tuple[str, ...]:
    value = payload.get(key)
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, (list, tuple)):
        out = [str(item).strip() for item in value if str(item).strip()]
        return tuple(out)
    return ()


def _adapter_base_models_from_card(snapshot_dir: Path) -> Tuple[str, ...]:
    payload = _read_card_frontmatter(snapshot_dir / "README.md")
    base_models = _frontmatter_list(payload, "base_model")
    if base_models:
        return base_models
    return _frontmatter_list(payload, "base_models")


def _snapshot_looks_like_full_model(snapshot_dir: Path) -> bool:
    if (snapshot_dir / "model_index.json").is_file():
        return True
    markers = (
        "transformer",
        "transformer-packed-mflux",
        "unet",
        "vae",
        "scheduler",
        "tokenizer",
        "text_encoder",
        "text_encoder_2",
    )
    return any((snapshot_dir / name).exists() for name in markers)


def _skip_adapter_snapshot(snapshot_dir: Path) -> bool:
    """Return True when a cache snapshot is clearly a runnable model, not an overlay repo.

    Adapter discovery is meant to surface optional overlays only. Prepared or official model
    snapshots often carry README frontmatter that mentions a base model, but their safetensors live
    under required component directories such as ``text_encoder`` or ``vae``. Those files are not
    user-selectable adapters and must not leak into the inventory.
    """

    return _snapshot_looks_like_full_model(snapshot_dir)


def _adapter_candidate_files(snapshot_dir: Path, *, repo_id: Optional[str] = None) -> List[Path]:
    out: List[Path] = []
    try:
        for path in snapshot_dir.rglob("*.safetensors"):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(snapshot_dir)
            except Exception:
                rel = path
            if rel.parts[:1] == (".cache",):
                continue
            if rel.parent.name == "transformer" and path.stem.isdigit():
                continue
            if repo_id:
                profile = known_lora_adapter_profile(f"{repo_id}:{rel.as_posix()}")
                if isinstance(profile, dict) and str(profile.get("artifact_role") or "adapter") != "adapter":
                    continue
            out.append(path)
    except Exception:
        return []
    return sorted(out)


def _adapter_suggested_target_roles(
    relative_path: str,
    allowed_roles: Sequence[str],
) -> Tuple[str, ...]:
    roles = [str(role).strip() for role in allowed_roles if str(role).strip()]
    if not roles:
        return ()
    if len(roles) == 1:
        return tuple(roles)
    name = _norm(relative_path)
    if "high-noise" in name or "high_noise" in name or "highnoise" in name:
        if "high_noise_transformer" in roles:
            return ("high_noise_transformer",)
    if "low-noise" in name or "low_noise" in name or "lownoise" in name:
        if "low_noise_transformer" in roles:
            return ("low_noise_transformer",)
    return tuple(roles)


def _parse_hf_adapter_source(source: str) -> Tuple[Optional[str], Optional[str]]:
    text = str(source or "").strip()
    if not text or _looks_like_path(text):
        return None, None
    if text.startswith("hf:"):
        text = text[3:].strip()
    if ":" in text:
        repo_id, relative_path = text.split(":", 1)
        repo_id = str(repo_id or "").strip()
        relative_path = str(relative_path or "").strip()
        if looks_like_hf_repo_id(repo_id):
            return repo_id, relative_path or None
        return None, None
    if looks_like_hf_repo_id(text):
        return text, None
    return None, None


def _resolve_cached_adapter_file(
    repo_id: str,
    *,
    relative_path: Optional[str],
    cache_dir: Optional[str],
) -> Optional[Path]:
    for _label, root in _configured_hf_cache_roots(cache_dir):
        snapshot_dir = _resolve_snapshot_in_cache_root(repo_id, root)
        if snapshot_dir is None:
            continue
        if relative_path:
            candidate = snapshot_dir / relative_path
            if candidate.is_file():
                return candidate
            candidate_name = Path(relative_path).name
            fallback_matches = [
                path
                for path in _adapter_candidate_files(snapshot_dir, repo_id=repo_id)
                if path.relative_to(snapshot_dir).as_posix() == relative_path or path.name == candidate_name
            ]
            if len(fallback_matches) == 1:
                return fallback_matches[0]
            if fallback_matches:
                break
            continue
        candidates = _adapter_candidate_files(snapshot_dir, repo_id=repo_id)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            break
    if relative_path:
        target_name = Path(relative_path).name
        for cache_label, root in _configured_adapter_cache_roots(cache_dir):
            if "lora cache" not in _norm(cache_label):
                continue
            candidate = root / relative_path
            if candidate.is_file():
                return candidate
            if target_name and target_name != relative_path:
                by_name = root / target_name
                if by_name.is_file():
                    return by_name
    return None


def _cached_adapter_handles(repo_id: str, *, cache_dir: Optional[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for _label, root in _configured_hf_cache_roots(cache_dir):
        snapshot_dir = _resolve_snapshot_in_cache_root(repo_id, root)
        if snapshot_dir is None:
            continue
        for path in _adapter_candidate_files(snapshot_dir, repo_id=repo_id):
            handle = f"{repo_id}:{path.relative_to(snapshot_dir).as_posix()}"
            key = handle.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(handle)
    return out


def _resolve_snapshot_in_cache_root(repo_id: str, root: Path) -> Optional[Path]:
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    candidates: List[Path] = []
    ref = repo_dir / "refs" / "main"
    try:
        if ref.is_file():
            snap = snapshots_dir / ref.read_text(encoding="utf-8").strip()
            if snap.is_dir():
                candidates.append(snap)
    except Exception:
        pass
    try:
        candidates.extend(
            entry for entry in snapshots_dir.iterdir() if entry.is_dir() and entry not in candidates
        )
    except Exception:
        pass
    for snap in candidates:
        if hf_snapshot_is_usable(snap, require_weight_files=True):
            return snap
    return None


def _preset_variant_model_id(preset: Any) -> str:
    """Return the exact prepared model repo id for runnable MLX-Gen variants."""

    return str(getattr(preset, "repo_id", None) or getattr(preset, "key", ""))


def _selection_requires_exact_preset(configured_model: Any, preset: Any) -> bool:
    """Return True when falling back to another quantized variant would be wrong."""

    requested = _norm(configured_model)
    if not requested:
        return False
    if requested == _norm(getattr(preset, "repo_id", None)):
        return True
    if any(token in requested for token in ("q8", "8bit", "q4", "4bit", "legacy")):
        return True
    if int(getattr(preset, "source_priority", 100) or 100) > 30 and requested != _norm(
        getattr(preset, "key", None)
    ):
        return True
    return False


def _mlx_gen_selector_matches_preset(selector: str, preset: Any) -> bool:
    requested = _norm(selector)
    if not requested:
        return False
    aliases = {_norm(alias) for alias in (getattr(preset, "aliases", ()) or ())}
    repo_id = str(getattr(preset, "repo_id", "") or "")
    repo_ids = {_norm(repo_id), _norm(repo_id.rsplit("/", 1)[-1])}
    upstream = str(getattr(preset, "upstream_repo_id", "") or "")
    if upstream:
        repo_ids.add(_norm(upstream))
        repo_ids.add(_norm(upstream.rsplit("/", 1)[-1]))
    return (
        requested == _norm(getattr(preset, "key", ""))
        or requested in aliases
        or requested in repo_ids
    )


def _ambiguous_mlx_gen_selector_choices(selector: Any) -> Tuple[str, ...]:
    requested = str(selector or "").strip()
    if _norm(requested) in {"seedvr2", "seedvr2-3b", "seedvr2-7b"}:
        return ()
    if not requested or _looks_like_path(requested) or looks_like_hf_repo_id(requested):
        requested_n = _norm(requested)
        if "/" in requested:
            for preset in model_presets(
                target="mlx", engine="mlx-gen", include_non_8bit=True, include_all_targets=True
            ):
                if getattr(preset, "source", "") == _ABSTRACTFRAMEWORK_MLX_GEN_SOURCE:
                    continue
                if requested_n == _norm(getattr(preset, "repo_id", None)):
                    return ()
        if any(token in _norm(requested) for token in ("q8", "8bit", "q4", "4bit")):
            return ()
    if any(token in _norm(requested) for token in ("q8", "8bit", "q4", "4bit")):
        return ()
    choices: List[str] = []
    for preset in model_presets(
        target="mlx", engine="mlx-gen", include_non_8bit=True, include_all_targets=True
    ):
        if getattr(preset, "source", "") != _ABSTRACTFRAMEWORK_MLX_GEN_SOURCE:
            continue
        if getattr(preset, "quantization_bits", None) not in {4, 8}:
            continue
        if _mlx_gen_selector_matches_preset(requested, preset):
            repo_id = str(getattr(preset, "repo_id", "") or "")
            if repo_id and repo_id not in choices:
                choices.append(repo_id)
    if len(choices) <= 1:
        return ()
    return tuple(sorted(choices))


def _cached_model_for_exact_preset(
    preset: Any,
    *,
    model_dir: Optional[str],
    cache_dir: Optional[str],
) -> Optional[_DiscoveredMFluxModel]:
    snapshot_dir = _preset_snapshot_dir(preset, model_dir, cache_dir)
    if snapshot_dir is None or not _has_model_files(snapshot_dir):
        return None
    if _is_partial_model_tree(snapshot_dir) or _is_incompatible_model_tree(snapshot_dir):
        return None
    return _DiscoveredMFluxModel(
        key=str(getattr(preset, "key", "")),
        snapshot_dir=snapshot_dir,
        repo_id=str(getattr(preset, "repo_id", "") or "") or None,
        source_label="configured cache" if cache_dir else "default HF cache",
        source_detail=(
            f"{'configured cache' if cache_dir else 'default HF cache'} "
            f"({getattr(preset, 'repo_id', '')})"
        ),
    )


def _discover_cached_legacy_mflux_models(
    model_dir: Optional[str],
) -> Dict[str, _DiscoveredMFluxModel]:
    out: Dict[str, _DiscoveredMFluxModel] = {}
    for root_label, root in _mflux_model_roots(model_dir):
        try:
            entries = list(root.iterdir())
        except Exception:
            continue
        for entry in entries:
            if (
                not entry.is_dir()
                or _is_incompatible_model_tree(entry)
                or _is_partial_model_tree(entry)
            ):
                continue
            if not _has_model_files(entry):
                continue
            base = _infer_base_model(entry.name)
            if base not in _MFLUX_MODELS:
                continue
            if entry.name != str(
                _preset_for(base).local_dir_name if _preset_for(base) else ""
            ) and not _looks_like_mflux_packaged_repo(entry.name):
                continue
            source_label = _local_source_label(root_label)
            candidate = _DiscoveredMFluxModel(
                key=base,
                snapshot_dir=entry,
                repo_id=entry.name,
                source_label=source_label,
                source_detail=f"{source_label} ({entry})",
            )
            chosen = out.get(base)
            candidate_priority = _candidate_priority(candidate.repo_id)
            chosen_priority = (
                _candidate_priority(chosen.repo_id) if chosen is not None else None
            )
            if chosen is None or (
                chosen_priority is not None and candidate_priority < chosen_priority
            ):
                out[base] = candidate
            bits = _infer_bits_from_text(entry.name)
            if bits is not None:
                out.setdefault(f"{base}-{int(bits)}bit", candidate)
    return out


def _discover_cached_hf_mflux_models(cache_dir: Optional[str]) -> Dict[str, _DiscoveredMFluxModel]:
    out: Dict[str, _DiscoveredMFluxModel] = {}
    for label, root in _configured_hf_cache_roots(cache_dir):
        try:
            repo_dirs = list(root.glob("models--*"))
        except Exception:
            continue
        for repo_dir in repo_dirs:
            if _is_incompatible_model_tree(repo_dir):
                continue
            repo_id = _repo_id_from_cache_dir_name(repo_dir.name)
            if not repo_id or not _looks_like_mflux_packaged_repo(repo_id):
                continue
            base = _infer_base_model(repo_id)
            if base not in _MFLUX_MODELS:
                continue
            snap = _resolve_snapshot_in_cache_root(repo_id, root)
            if (
                snap is None
                or not _has_model_files(snap)
                or _is_partial_model_tree(snap)
                or _is_incompatible_model_tree(snap)
            ):
                continue
            candidate = _DiscoveredMFluxModel(
                key=base,
                snapshot_dir=snap,
                repo_id=repo_id,
                source_label=label,
                source_detail=f"{label} ({repo_id})",
            )
            chosen = out.get(base)
            if chosen is None or _candidate_priority(repo_id) < _candidate_priority(chosen.repo_id):
                out[base] = candidate
    return out


def discover_cached_mflux_models(
    *,
    model_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, _DiscoveredMFluxModel]:
    discovered = _discover_cached_legacy_mflux_models(model_dir)

    cache_root = _cache_root(cache_dir)
    extra_cache_roots = [] if cache_dir else framework_hf_cache_roots()
    for preset in model_presets(target="mlx", engine="mlx-gen", include_non_8bit=True):
        for repo_id in _candidate_repo_ids_for_preset(preset):
            snap = (
                _resolve_snapshot_in_cache_root(repo_id, Path(cache_root))
                if cache_dir
                else resolve_hf_repo_snapshot(
                    repo_id,
                    cache_dir=cache_root,
                    require_weight_files=True,
                    extra_roots=extra_cache_roots,
                )
            )
            if (
                snap is None
                or not _has_model_files(snap)
                or _is_partial_model_tree(snap)
                or _is_incompatible_model_tree(snap)
            ):
                continue
            candidate = _DiscoveredMFluxModel(
                key=preset.key,
                snapshot_dir=snap,
                repo_id=repo_id,
                source_label="configured cache" if cache_dir else "default HF cache",
                source_detail=f"{'configured cache' if cache_dir else 'default HF cache'} ({repo_id})",
            )
            chosen = discovered.get(preset.key)
            if chosen is None or _candidate_priority(repo_id) < _candidate_priority(chosen.repo_id):
                discovered[preset.key] = candidate
        if preset.key in discovered:
            continue
    for key, candidate in _discover_cached_hf_mflux_models(cache_dir).items():
        discovered.setdefault(key, candidate)
    return discovered


def discover_incomplete_mflux_sources(
    *,
    model_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, List[str]] = {}
    valid = discover_cached_mflux_models(model_dir=model_dir, cache_dir=cache_dir)

    def add(key: str, detail: str) -> None:
        text = str(detail or "").strip()
        if not text:
            return
        bucket = out.setdefault(key, [])
        if text not in bucket:
            bucket.append(text)

    for preset in model_presets(target="mlx", engine="mlx-gen", include_non_8bit=True):
        if preset.key in valid:
            continue
        for _root_label, root in _mflux_model_roots(model_dir):
            try:
                entries = list(root.iterdir())
            except Exception:
                entries = []
            for entry in entries:
                if not entry.is_dir() or _infer_base_model(entry.name) != preset.key:
                    continue
                if _is_incompatible_model_tree(entry):
                    add(preset.key, f"incompatible local model dir: {entry}")
                    continue
                if _is_partial_model_tree(entry) or not _has_model_files(entry):
                    add(preset.key, f"incomplete local model dir: {entry}")
        for repo_id in _candidate_repo_ids_for_preset(preset):
            for label, root in _configured_hf_cache_roots(cache_dir):
                repo_dir = root / f"models--{repo_id.replace('/', '--')}"
                lock_dir = root / ".locks" / f"models--{repo_id.replace('/', '--')}"
                if repo_dir.exists() or lock_dir.exists():
                    if _is_incompatible_model_tree(repo_dir):
                        add(preset.key, f"incompatible HF cache: {label} ({repo_id})")
                    else:
                        add(preset.key, f"incomplete HF cache: {label} ({repo_id})")
        for label, root in _configured_hf_cache_roots(cache_dir):
            try:
                repo_dirs = list(root.glob("models--*"))
            except Exception:
                repo_dirs = []
            for repo_dir in repo_dirs:
                repo_id = _repo_id_from_cache_dir_name(repo_dir.name)
                if not repo_id or not _looks_like_mflux_packaged_repo(repo_id):
                    continue
                if _infer_base_model(repo_id) != preset.key:
                    continue
                display_repo_id = _display_repo_id(repo_id)
                if _is_incompatible_model_tree(repo_dir):
                    add(preset.key, f"incompatible HF cache: {label} ({display_repo_id})")
                    continue
                if _is_partial_model_tree(repo_dir):
                    add(preset.key, f"incomplete HF cache: {label} ({display_repo_id})")
            try:
                lock_dirs = list((root / ".locks").glob("models--*"))
            except Exception:
                lock_dirs = []
            for lock_dir in lock_dirs:
                repo_id = _repo_id_from_cache_dir_name(lock_dir.name)
                if not repo_id or not _looks_like_mflux_packaged_repo(repo_id):
                    continue
                if _infer_base_model(repo_id) != preset.key:
                    continue
                add(preset.key, f"incomplete HF cache: {label} ({_display_repo_id(repo_id)})")
    return {key: tuple(values) for key, values in out.items()}


def mlx_gen_base_model(*selectors: Any) -> Optional[str]:
    """The MLX-Gen base model (a `base_model` value such as `minimax-h3-turbo`) that `selectors`
    (model ids, repo ids, names or local paths) identify, or None."""
    return _infer_base_model(*selectors)


def mlx_gen_schedules(*selectors: Any) -> Tuple[str, ...]:
    """The schedules (`base_model` values) of the model family `selectors` identify; empty for none.

    `resolve_mlx_gen_schedule_adapter` gives the adapter each one loads."""
    model_def = _MFLUX_MODELS.get(str(_infer_base_model(*selectors) or ""))
    if model_def is None:
        return ()
    return tuple(sorted(key for key, other in _MFLUX_MODELS.items() if other.family == model_def.family))


def mlx_gen_package_schedule(preset: Any, base_model: Any) -> str:
    """`base_model` as the schedule it names, when that is a schedule of the catalog package `preset`;
    ValueError naming the package's schedules otherwise."""
    key = _infer_base_model(base_model)
    choices = mlx_gen_schedules(preset.repo_id, preset.upstream_repo_id, preset.key)
    if key not in choices:
        raise ValueError(
            f"{base_model!r} is not a schedule of {preset.repo_id}."
            + (f" Choose one of: {', '.join(choices)}." if choices else "")
        )
    return str(key)


def _infer_base_model(*values: Any) -> Optional[str]:
    """The MLX-Gen base model `values` (model ids, repo ids, names or local paths) identify.

    Exact identification is tried on every value before anything else: the value as an alias,
    then the repo id of a Hugging Face cache snapshot path or the last path component as an alias,
    then a local package's own card (`base_model`). A parent directory never decides. Name matching
    is the last resort (`_legacy_base_model_match`).
    """
    for identify in (_exact_base_model, _card_base_model, _legacy_base_model_match):
        for value in values:
            key = identify(value)
            if key:
                return key
    return None


def _exact_base_model(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    for name in (text, _hf_cache_repo_id(path), path.name):
        key = _KNOWN_MODEL_ALIASES.get(_norm(name))
        if key:
            return key
    return None


def _hf_cache_repo_id(path: Path) -> Optional[str]:
    # A Hugging Face cache snapshot: <root>/models--<org>--<name>/snapshots/<revision>.
    if path.parent.name != "snapshots":
        return None
    return _repo_id_from_cache_dir_name(path.parent.parent.name)


def _card_base_model(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_dir():
        return None
    for name in _frontmatter_list(_read_card_frontmatter(path / "README.md"), "base_model"):
        key = _KNOWN_MODEL_ALIASES.get(_norm(name))
        if key:
            return key
    return None


def _legacy_base_model_match(value: Any) -> Optional[str]:
    # Name matching over the whole value; MiniMax-H3 is identified only exactly or by its card.
    s = _norm(value)
    if not s:
        return None
    if "qwen" in s and "image-edit" in s:
        if "2509" in s:
            return "qwen-image-edit-2509"
        return "qwen-image-edit-2511"
    if "qwen" in s and "image" in s:
        return "qwen-image"
    if "bonsai" in s:
        return "bonsai-image-ternary"
    if "z-image-turbo" in s or "zimage-turbo" in s:
        return "z-image-turbo"
    if "z-image" in s or "zimage" in s:
        return "z-image"
    if "ernie" in s and "image" in s:
        return "ernie-image-turbo"
    if "fibo" in s and "edit" in s and "rmbg" in s:
        return "fibo-edit-rmbg"
    if "fibo" in s and "edit" in s:
        return "fibo-edit"
    if "fibo" in s and "lite" in s:
        return "fibo-lite"
    if "fibo" in s:
        return "fibo"
    if "seedvr2" in s:
        if "7b" in s:
            return SEEDVR2_7B_MODEL_KEY
        return SEEDVR2_3B_MODEL_KEY
    if "wan" in s and ("t2v-a14b" in s or "t2v_a14b" in s or "t2v-a14" in s):
        return WAN_T2V_A14B_MODEL_KEY
    if "wan" in s and ("i2v-a14b" in s or "i2v_a14b" in s or "i2v-a14" in s):
        return WAN_I2V_A14B_MODEL_KEY
    if (
        s == "wan"
        or "wan2.2-ti2v" in s
        or "wan2-2-ti2v" in s
        or "wan-ti2v" in s
        or "wan-video" in s
    ):
        return WAN_TI2V_MODEL_KEY
    if "klein-base-4b" in s or "kleinbase4b" in s:
        return "flux2-klein-base-4b"
    if "klein-base-9b" in s or "kleinbase9b" in s:
        return "flux2-klein-base-9b"
    if "klein-4b" in s or "klein4b" in s:
        return "flux2-klein-4b"
    if "klein-9b" in s or "klein9b" in s:
        return "flux2-klein-9b"
    return None


def _preset_for(value: Any) -> Any:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return find_model_preset(s, target="mlx", engine="mlx-gen", require_8bit=False)
    except Exception:
        return None


def _base_model_for_preset(configured_base: Optional[str], configured_model: Any, preset: Any) -> str:
    if configured_base:
        return _infer_base_model(configured_base) or str(configured_base)
    inferred = _infer_base_model(
        getattr(preset, "repo_id", None),
        getattr(preset, "upstream_repo_id", None),
        getattr(preset, "key", None),
        configured_model,
    )
    if inferred:
        return inferred
    return str(getattr(preset, "key", "") or "")


@lru_cache(maxsize=1)
def _mflux_capability_registry() -> Optional[VisionModelCapabilitiesRegistry]:
    try:
        return VisionModelCapabilitiesRegistry()
    except Exception:
        return None


def _first_other_engine_preset(value: Any) -> Any:
    s = str(value or "").strip()
    if not s:
        return None
    for target in ("mlx", "gguf", "diffusers", "fp8"):
        try:
            preset = find_model_preset(s, target=target, engine=None, require_8bit=False)
        except Exception:
            continue
        if preset.engine not in {"mflux", "mlx-gen"}:
            return preset
    return None


def _preset_snapshot_dir(
    preset: Any, model_dir: Optional[str], cache_dir: Optional[str]
) -> Optional[Path]:
    legacy_root = Path(model_dir).expanduser() if model_dir else default_legacy_model_root()
    source_dir = legacy_root / preset.local_dir_name
    cache_root = _cache_root(cache_dir)
    extra_cache_roots = framework_hf_cache_roots()
    try:
        return ensure_hf_repo_snapshot(
            preset.repo_id,
            source_dir=source_dir,
            cache_dir=cache_root,
            cleanup_source=True,
            require_weight_files=True,
            extra_roots=extra_cache_roots,
        )
    except Exception:
        return resolve_hf_repo_snapshot(
            preset.repo_id,
            cache_dir=cache_root,
            require_weight_files=True,
            extra_roots=extra_cache_roots,
        )


@dataclass(frozen=True)
class _Resolution:
    """The model a backend's configuration resolved to (see `MFluxVisionBackend._resolve`)."""

    source: Tuple[Any, ...]  # what it was resolved from: configuration and cache locations
    model_path: str
    base_model: str
    stored_bits: Optional[int]  # the width its stored weights carry, when the catalog or its name says
    prepared: bool  # a prepared package: MLX-Gen loads its weights as stored and takes no load-time quantize


class _ModelKey(NamedTuple):
    """What the held instance was built from; a request whose key differs builds another."""

    model_path: str
    base_model: str
    variant: str
    lora_paths: Tuple[str, ...] = ()
    lora_scales: Tuple[float, ...] = ()
    lora_target_roles: Tuple[str, ...] = ()
    stream_blocks: bool = False
    quantize: Optional[int] = None


class MFluxVisionBackend(VisionBackend):
    """Compatibility class for the local MLX-Gen image and video backend."""

    def __init__(self, *, config: MFluxBackendConfig):
        self._cfg = config
        self._model: Any = None
        self._model_key: Optional[_ModelKey] = None
        self._warmed_model_key: Optional[_ModelKey] = None
        self._runtime_lock = threading.Lock()
        self._runtime_queue: Optional[queue.Queue[Any]] = None
        self._runtime_thread: Optional[threading.Thread] = None
        self._runtime_thread_id: Optional[int] = None
        # Requests this backend normalized, by identity: one VisionManager normalized
        # (`normalize_*`) reaches the entry point as that object and is not normalized again.
        self._normalized_requests: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
        # The configured model as last resolved, until `unload()`. See `_resolve_model`.
        self._resolution: Optional[_Resolution] = None

    def _ensure_runtime_thread(self) -> queue.Queue[Any]:
        existing = self._runtime_queue
        worker = self._runtime_thread
        if existing is not None and worker is not None and worker.is_alive():
            return existing
        with self._runtime_lock:
            existing = self._runtime_queue
            worker = self._runtime_thread
            if existing is not None and worker is not None and worker.is_alive():
                return existing

            task_queue: queue.Queue[Any] = queue.Queue()

            def runner() -> None:
                self._runtime_thread_id = threading.get_ident()
                while True:
                    task = task_queue.get()
                    if task is None:
                        break
                    fn, args, kwargs, future = task
                    if future.cancelled():
                        continue
                    cancelled: Optional[GenerationCancelledError] = None
                    try:
                        result = fn(*args, **kwargs)
                    except GenerationCancelledError as e:
                        # A copy that was never raised: no traceback, so no frame of the run (nor
                        # the model it held) reaches the caller, who may keep the exception.
                        cancelled = GenerationCancelledError(*e.args)
                    except BaseException as e:
                        future.set_exception(e)
                        continue
                    else:
                        future.set_result(result)
                        continue
                    # The run's frames are gone with the original exception; return what they held.
                    import gc

                    gc.collect()
                    _clear_mlx_cache()
                    future.set_exception(cancelled)

            worker = threading.Thread(
                target=runner,
                name="abstractvision-mflux-runtime",
                daemon=True,
            )
            worker.start()
            self._runtime_queue = task_queue
            self._runtime_thread = worker
            return task_queue

    def _run_on_runtime_thread(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._runtime_thread_id == threading.get_ident():
            return fn(*args, **kwargs)
        future: Future[Any] = Future()
        self._ensure_runtime_thread().put((fn, args, kwargs, future))
        return future.result()

    def _unload_impl(self) -> None:
        self._model = None
        self._model_key = None
        self._warmed_model_key = None
        # The next request resolves the configuration again (a newer snapshot, one that lost its weights).
        self._resolution = None
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        # The dropped weights stay in MLX's allocator cache, held by the process, until the cache is
        # cleared: clear it here, at the release point.
        try:
            import mlx.core as mx  # type: ignore

            mx.clear_cache()
        except Exception:
            pass

    def _release_model(self) -> None:
        # Forget the held instance and return its memory: when a run freed part of it (see
        # _MemoryPlan.releases_model), and before anything else is loaded when it is replaced
        # (another model key: stream_blocks, quantize, adapters, variant), so two instances are
        # never in memory together. The resolution stays.
        self._model = None
        self._model_key = None
        self._warmed_model_key = None
        import gc

        gc.collect()
        _clear_mlx_cache()

    def has_loaded_model(self) -> bool:
        """Whether a model instance is held; False after a run that released it."""
        return self._model is not None

    @property
    def _resolved_model_path(self) -> Optional[str]:
        return self._resolution.model_path if self._resolution is not None else None

    @property
    def _resolved_base_model(self) -> Optional[str]:
        return self._resolution.base_model if self._resolution is not None else None

    def _quantization_bits(self) -> Optional[int]:
        # The width the held instance runs at: the load-time quantize it was built with, else its stored width.
        if self._model_key is not None and self._model_key.quantize is not None:
            return self._model_key.quantize
        return self._resolution.stored_bits if self._resolution is not None else None

    def unload(self) -> None:
        if self._runtime_queue is None and self._runtime_thread is None:
            self._unload_impl()
            return
        self._run_on_runtime_thread(self._unload_impl)

    def preload(self, memory_mode: Optional[str] = None, quantize: Optional[int] = None) -> None:
        self._run_on_runtime_thread(self._preload_impl, memory_mode, quantize)

    def _capability_model_id(
        self, *, model_id: Optional[str] = None, base_model: Optional[str] = None
    ) -> Optional[str]:
        explicit = str(model_id or "").strip()
        if explicit:
            return explicit
        configured_model = str(self._cfg.model or "").strip()
        preset = _preset_for(configured_model) if configured_model else None
        if preset is not None:
            return str(preset.upstream_repo_id or preset.repo_id)
        resolved_base = str(
            base_model
            or self._resolved_base_model
            or _infer_base_model(self._cfg.base_model, self._cfg.model)
            or ""
        ).strip()
        return _MFLUX_BASE_MODEL_REGISTRY_IDS.get(resolved_base)

    def _supported_task_names(
        self,
        *,
        model_id: Optional[str] = None,
        base_model: Optional[str] = None,
    ) -> List[str]:
        resolved_base = str(
            base_model
            or self._resolved_base_model
            or _infer_base_model(self._cfg.base_model, self._cfg.model)
            or ""
        ).strip()
        capability_model_id = self._capability_model_id(model_id=model_id, base_model=resolved_base)
        allowed = set(_MFLUX_RUNTIME_ALLOWED_TASKS.get(resolved_base, ("text_to_image",)))
        reg = _mflux_capability_registry()
        if reg is not None and capability_model_id:
            try:
                return sorted(
                    str(task_name)
                    for task_name in reg.get(capability_model_id).tasks.keys()
                    if str(task_name) in allowed
                )
            except Exception:
                pass
        fallback = _MFLUX_BASE_MODEL_FALLBACK_TASKS.get(resolved_base, ("text_to_image",))
        return sorted(str(task_name) for task_name in fallback if str(task_name) in allowed)

    def _route_capability_selectors(self) -> Tuple[str, ...]:
        selectors: List[str] = []
        configured_model = str(self._cfg.model or "").strip()
        if configured_model:
            selectors.append(configured_model)
            preset = _preset_for(configured_model)
            if preset is not None:
                selectors.append(_preset_variant_model_id(preset))
        if self._resolved_model_path:
            selectors.append(str(self._resolved_model_path))
        deduped: List[str] = []
        seen: set[str] = set()
        for selector in selectors:
            selector_s = str(selector or "").strip()
            if not selector_s or selector_s in seen:
                continue
            seen.add(selector_s)
            deduped.append(selector_s)
        return tuple(deduped)

    def _current_route_capabilities(self) -> Dict[str, List[Dict[str, Any]]]:
        base_model = _infer_base_model(self._cfg.base_model) or self._resolved_base_model
        # MiniMax-H3 packages hold the base weights; the configured entry picks the schedule,
        # so its rows (defaults, steps) are the entry's, as `--base-model` gives them.
        schedule = base_model if base_model in MINIMAX_H3_MODEL_KEYS else None
        if schedule:
            # The entry's own rows: a catalog repo id (MiniMaxAI/MiniMax-H3) names the base entry.
            capabilities = self._route_capabilities_for_model(schedule)
            if capabilities:
                return capabilities
        for selector in self._route_capability_selectors():
            capabilities = self._route_capabilities_for_model(selector, base_model=schedule)
            if capabilities:
                return capabilities
        return {}

    def _task_spec(
        self,
        task: str,
        *,
        model_id: Optional[str] = None,
        base_model: Optional[str] = None,
    ) -> Optional[Any]:
        resolved_base = str(
            base_model
            or self._resolved_base_model
            or _infer_base_model(self._cfg.base_model, self._cfg.model)
            or ""
        ).strip()
        capability_model_id = self._capability_model_id(model_id=model_id, base_model=resolved_base)
        reg = _mflux_capability_registry()
        if reg is None or not capability_model_id:
            return None
        try:
            return reg.get(capability_model_id).tasks.get(str(task))
        except Exception:
            return None

    def _task_parameter_supported(
        self,
        task: str,
        parameter: str,
        *,
        model_id: Optional[str] = None,
        base_model: Optional[str] = None,
    ) -> Optional[bool]:
        task_spec = self._task_spec(task, model_id=model_id, base_model=base_model)
        if task_spec is None:
            return None
        params = task_spec.params if isinstance(task_spec.params, dict) else {}
        if parameter not in params:
            return False
        value = params.get(parameter)
        if isinstance(value, dict):
            return False if value.get("supported") is False else True
        return True

    def _route_option_support(self, task: str, route_field: str) -> Optional[bool]:
        rows = self._current_route_capabilities().get(str(task), [])
        if not rows:
            return None
        return any(bool(row.get(route_field)) for row in rows)

    def _effective_option_support(
        self,
        task: str,
        *,
        parameter: str,
        route_field: str,
    ) -> Optional[bool]:
        package_support = self._task_parameter_supported(task, parameter)
        runtime_support = self._route_option_support(task, route_field)
        if runtime_support is not None:
            if package_support is False:
                result: Optional[bool] = False
            elif package_support is None:
                result = bool(runtime_support)
            else:
                result = bool(package_support and runtime_support)
        else:
            resolved_base = str(
                self._resolved_base_model
                or _infer_base_model(self._cfg.base_model, self._cfg.model)
                or ""
            ).strip()
            if parameter == "mask":
                if package_support is False:
                    result = False
                else:
                    result = resolved_base in {
                        "qwen-image-edit-2511",
                        "fibo-edit",
                        "fibo-edit-rmbg",
                    }
            else:
                result = package_support

        # Route tables can advertise inpaint-only mask support that the backend
        # still rejects for non-edit families in ``_edit_image_impl``.
        if parameter == "mask" and result:
            model_key = str(
                self._resolved_base_model
                or _infer_base_model(self._cfg.base_model, self._cfg.model)
                or ""
            ).strip()
            model_def = _MFLUX_MODELS.get(model_key)
            if model_def is not None and model_def.family not in {"qwen-edit", "fibo-edit"}:
                return False
        return result

    def _require_route_option_support_if_known(
        self,
        task: str,
        *,
        route_field: str,
        field: str,
        error_message: str,
    ) -> None:
        runtime_support = self._route_option_support(task, route_field)
        if runtime_support is False:
            raise CapabilityNotSupportedError(error_message, field=field, supported=())

    def _route_rows(self, task: Optional[str] = None) -> List[Dict[str, Any]]:
        routes = self._current_route_capabilities()
        if task:
            return list(routes.get(str(task), []))
        return [row for rows in routes.values() for row in rows]

    def describe_routes(
        self, *, model: Optional[str] = None, task: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """MLX-Gen's capability rows (schema 17) for `model` (default: the configured one), as JSON.

        Each row is the engine's own record, unchanged apart from JSON conversion: tasks
        (`public_task`), canvas and frame rules, defaults, LoRA, audio, memory fields
        (`supports_block_streaming`, `measured_runs`, `peak_bytes_*`) and quantization. A host
        builds its controls and memory planning from these without importing mlx-gen.
        """
        if model:
            routes = self._route_capabilities_for_model(str(model))
        else:
            routes = self._current_route_capabilities()
        wanted = str(task or "").strip().replace("-", "_")
        rows = [
            _json_safe_route_value(row)
            for task_name, task_rows in routes.items()
            if not wanted or task_name == wanted
            for row in task_rows
        ]
        return rows

    def supported_request_options(self, task: Optional[str] = None) -> FrozenSet[str]:
        options = {"quantize"}
        if self._configured_family() == "minimax-h3":
            options |= set(SOUNDTRACK_REQUEST_FIELDS)
        return frozenset(options)

    def get_capabilities(self) -> VisionBackendCapabilities:
        routes = self._current_route_capabilities()
        return VisionBackendCapabilities(
            supported_tasks=self._supported_task_names(),
            supports_mask=self._effective_option_support(
                "image_to_image",
                parameter="mask",
                route_field="supports_mask",
            ),
            # MLX-Gen runs structured control through its Qwen ControlNet variant and a ControlNet sidecar
            # repository, neither of which this backend loads.
            supports_control_image=False,
            memory_modes=self.supported_memory_modes(),
            quantize_bits=self.quantize_bits(),
            cancellation="step",
            generates_audio=(
                any(bool(row.get("generates_audio")) for rows in routes.values() for row in rows)
                if routes
                else None
            ),
        )

    def supported_memory_modes(self, task: Optional[str] = None) -> Tuple[str, ...]:
        if task:
            streams = bool(self._route_option_support(task, "supports_block_streaming"))
        else:
            streams = _route_rows_support_block_streaming(self._current_route_capabilities())
        return mlx_gen_memory_modes(family=self._configured_family(), streams_blocks=streams)

    def _normalize_once(self, request: Any, normalize: Callable[[Any], Any], *, task: str) -> Any:
        """`normalize(request)` for `task`, unless `request` is one this backend already normalized."""
        if self._normalized_requests.get(id(request)) is request:
            return request
        self._refuse_a_task_the_model_lacks(task)
        normalized = normalize(request)
        self._normalized_requests[id(normalized)] = normalized
        return normalized

    def _normalized_with_step_progress(
        self,
        request: Any,
        progress_callback: Callable[[int, Optional[int]], None],
        normalize: Callable[[Any], Any],
        *,
        task: str,
    ) -> Any:
        """`request` normalized once for `task`, carrying `progress_callback` for its run.

        The copy holds the same checked values plus a progress callback (routing data no check
        reads), so it is recorded as normalized and its entry point does not check it again.
        """
        normalized = self._normalize_once(request, normalize, task=task)
        carrying = replace(normalized, extra={**(normalized.extra or {}), "_step_progress_callback": progress_callback})
        self._normalized_requests[id(carrying)] = carrying
        return carrying

    def _refuse_a_task_the_model_lacks(self, task: str) -> None:
        # Before anything is read or loaded: the configured model's family has no such route (an
        # image task on MiniMax-H3, a video task on FLUX.2, ...). A model chosen from the cache at
        # the first request is checked by its own normalization.
        model_def = self._configured_model_def()
        if model_def is None:
            return
        tasks = self._supported_task_names()
        if task not in tasks:
            raise CapabilityNotSupportedError(
                f"The configured MLX-Gen model ({model_def.key}) has no {task} route; "
                f"its tasks: {', '.join(tasks) or 'none'}."
            )

    def _memory_plan(self, mode: str) -> _MemoryPlan:
        # `mode` is a checked memory_mode (`resolve_memory_request`); the model is the resolved one.
        flags = MLX_GEN_MEMORY_MODE_FLAGS[mode]
        low_ram = flags.low_ram
        model_def = self._resolved_model_def()
        # The run frees part of the instance: a low-RAM run on every family but SeedVR2 (whose
        # low-RAM mode is the cache limit alone), and every Wan TI2V-5B run (its denoiser,
        # released before the decode, which mlx-gen cannot read again).
        releases_model = (low_ram and model_def.family != "seedvr2-upscale") or model_def.key == WAN_TI2V_MODEL_KEY
        return _MemoryPlan(
            mode=mode,
            low_ram=low_ram,
            stream_blocks=flags.stream_blocks,
            releases_model=releases_model,
        )

    @contextmanager
    def _memory_session(self, plan: _MemoryPlan) -> Iterator[None]:
        """Hold low-RAM mode's MLX cache limit from before the model loads until the run ends.

        When the run ends, an instance the run freed part of is dropped, whatever the outcome.
        """
        token = _acquire_mlx_cache_limit(LOW_RAM_MLX_CACHE_LIMIT_BYTES) if plan.low_ram else None
        try:
            yield
        finally:
            if token is not None:
                _release_mlx_cache_limit(token)
            if plan.releases_model:
                self._release_model()

    def _attach_low_ram_saver(self, model: Any) -> None:
        """Register mlx-gen's low-RAM MemorySaver, as `mlxgen generate --low-ram` does on image routes.

        It tiles the VAE, frees the text encoders before the denoising loop and the transformer
        after it.
        """
        MemorySaver = _lazy_import_mlx_gen_memory_saver()
        # MemorySaver sets the cache limit it is given: the one this run holds.
        model.callbacks.register(
            MemorySaver(model=model, keep_transformer=False, cache_limit_bytes=_held_mlx_cache_limit())
        )

    def _config_lora_adapters(self) -> Tuple[Any, ...]:
        paths = [str(path).strip() for path in (self._cfg.lora_paths or ()) if str(path).strip()]
        if not paths:
            return ()
        scales = [float(value) for value in (self._cfg.lora_scales or ())]
        roles = [str(value).strip() for value in (self._cfg.lora_target_roles or ()) if str(value).strip()]
        out = []
        for index, path in enumerate(paths):
            scale = scales[index] if index < len(scales) else None
            target_role = roles[index] if index < len(roles) else None
            out.append(
                {
                    "source": path,
                    "scale": scale,
                    "target_role": target_role,
                }
            )
        return resolve_request_lora_adapters(out)

    def _effective_request_lora_adapters(self, request: Any) -> Tuple[Any, ...]:
        request_adapters = getattr(request, "lora_adapters", None)
        request_extra = getattr(request, "extra", None)
        effective = resolve_request_lora_adapters(request_adapters, extra=request_extra)
        if effective:
            return effective
        return self._config_lora_adapters()

    def _resolve_mlx_gen_lora_source(self, spec: Any) -> str:
        source = adapter_source_for_mlx_gen(spec)
        repo_id, relative_path = _parse_hf_adapter_source(source)
        if not repo_id:
            return source
        resolved_file = _resolve_cached_adapter_file(
            repo_id,
            relative_path=relative_path,
            cache_dir=self._cfg.cache_dir,
        )
        if resolved_file is not None:
            return str(resolved_file)
        if relative_path:
            return source
        handles = _cached_adapter_handles(repo_id, cache_dir=self._cfg.cache_dir)
        if not handles:
            handles = [
                info.id
                for info in self.list_provider_adapters(model=self._cfg.model or None)
                if str(info.repo_id or "").strip() == repo_id
            ]
        if handles:
            raise OptionalDependencyMissingError(
                f"LoRA repo {repo_id!r} contains multiple adapter files. "
                "Select one explicit adapter handle instead of the bare repo id:\n  - "
                + "\n  - ".join(handles)
            )
        return source

    def _mlx_gen_lora_init_kwargs(self, adapters: Sequence[Any]) -> Dict[str, Any]:
        if not adapters:
            return {}
        lora_paths = [self._resolve_mlx_gen_lora_source(spec) for spec in adapters]
        out: Dict[str, Any] = {"lora_paths": lora_paths}
        lora_scales = lora_scales_for_backend(adapters)
        if lora_scales is not None:
            out["lora_scales"] = lora_scales
        lora_target_roles = lora_target_roles_for_backend(adapters)
        if lora_target_roles is not None:
            out["lora_target_roles"] = lora_target_roles
        return out

    def _generated_image_metadata(self, generated: Any) -> Dict[str, Any]:
        get_metadata = getattr(generated, "_get_metadata", None)
        if not callable(get_metadata):
            return {}
        try:
            metadata = get_metadata()
        except Exception:
            return {}
        return _public_engine_metadata(metadata) if isinstance(metadata, dict) else {}

    def _route_capabilities_for_model(
        self, model_selector: str, *, base_model: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        try:
            get_model_capabilities, _resolve_generation_plan = _lazy_import_mlx_gen_task_inference()
            if base_model:
                capabilities = get_model_capabilities(model=model_selector, base_model=base_model)
            else:
                capabilities = get_model_capabilities(model=model_selector)
        except Exception:
            return {}
        out: Dict[str, List[Dict[str, Any]]] = {}
        for capability in getattr(capabilities, "capabilities", ()) or ():
            raw = getattr(capability, "__dict__", None)
            if not isinstance(raw, dict):
                continue
            task = mlx_gen_route_task(raw)
            if not task:
                continue
            out.setdefault(task, []).append(dict(raw))
        return out

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        # Listing should reflect available local weights even when the optional
        # `mflux` runtime is not installed. Generation will still error until
        # the runtime is present, but catalogs can surface what is already
        # downloaded.
        task_s = str(task or "").strip()
        out = []
        emitted: set[str] = set()
        discovered = discover_cached_mflux_models(
            model_dir=self._cfg.model_dir,
            cache_dir=self._cfg.cache_dir,
        )
        reg = _mflux_capability_registry()
        for preset in model_presets(target="mlx", engine="mlx-gen", include_non_8bit=True):
            model_selector = _preset_variant_model_id(preset)
            if model_selector in emitted:
                continue
            discovered_model = _cached_model_for_exact_preset(
                preset,
                model_dir=self._cfg.model_dir,
                cache_dir=self._cfg.cache_dir,
            )
            is_seedvr2_preset = "seedvr2" in _norm(getattr(preset, "key", "")) or "seedvr2" in _norm(
                getattr(preset, "repo_id", "")
            )
            if discovered_model is None and is_seedvr2_preset:
                local_candidate = discovered.get(preset.key)
                if local_candidate is not None:
                    preset_bits = getattr(preset, "quantization_bits", None)
                    local_bits = _infer_bits_from_text(local_candidate.repo_id) or _infer_bits_from_text(
                        local_candidate.snapshot_dir
                    )
                    if preset_bits is None or local_bits is None or int(preset_bits) == int(local_bits):
                        discovered_model = local_candidate
            if discovered_model is None:
                continue
            base_model = _infer_base_model(preset.repo_id, preset.upstream_repo_id, preset.key)
            tasks = self._supported_task_names(
                model_id=str(preset.upstream_repo_id or preset.repo_id),
                base_model=base_model,
            )
            if task_s and task_s not in tasks:
                continue
            model_def = _MFLUX_MODELS.get(str(base_model or ""))
            schedule = _infer_base_model(self._cfg.base_model)
            route_capabilities = self._route_capabilities_for_model(
                model_selector,
                base_model=schedule
                if model_def is not None
                and model_def.family == "minimax-h3"
                and schedule in MINIMAX_H3_MODEL_KEYS
                else None,
            )
            parameter_metadata = (
                _mflux_parameter_metadata(
                    model_def,
                    block_streaming=_route_rows_support_block_streaming(route_capabilities),
                    route_capabilities=route_capabilities,
                )
                if model_def is not None
                else {}
            )
            task_specs: Dict[str, Dict[str, Any]] = {}
            capability_model_id = self._capability_model_id(
                model_id=str(preset.upstream_repo_id or preset.repo_id),
                base_model=base_model,
            )
            if reg is not None and capability_model_id:
                try:
                    reg_spec = reg.get(capability_model_id)
                except Exception:
                    reg_spec = None
                if reg_spec is not None:
                    task_specs = {
                        str(task_name): {
                            "inputs": list(task_spec.inputs),
                            "outputs": list(task_spec.outputs),
                            "params": dict(task_spec.params),
                            "requires": (
                                dict(task_spec.requires)
                                if isinstance(task_spec.requires, dict)
                                else None
                            ),
                        }
                        for task_name, task_spec in reg_spec.tasks.items()
                        if str(task_name) in set(tasks)
                    }
            for task_name in tasks:
                task_spec = dict(task_specs.get(task_name) or {})
                route_rows = [dict(item) for item in route_capabilities.get(task_name, [])]
                if route_rows:
                    default_route = next(
                        (row for row in route_rows if bool(row.get("default_for_task"))),
                        route_rows[0],
                    )
                    lora_states = {
                        str(row.get("lora_status") or "unsupported")
                        for row in route_rows
                        if bool(row.get("supports_lora"))
                    }
                    task_spec["supports_lora"] = any(
                        bool(row.get("supports_lora")) for row in route_rows
                    )
                    task_spec["lora_status"] = (
                        next(iter(lora_states))
                        if len(lora_states) == 1
                        else default_route.get("lora_status")
                        or ("mixed" if lora_states else "unsupported")
                    )
                    task_spec["lora_target_roles"] = list(
                        dict.fromkeys(
                            str(role)
                            for row in route_rows
                            for role in (row.get("lora_target_roles") or ())
                            if str(role).strip()
                        )
                    )
                    task_spec["lora_validation_profile"] = default_route.get(
                        "lora_validation_profile"
                    )
                    task_spec["route_capabilities"] = route_rows
                if task_spec:
                    task_specs[task_name] = task_spec
            if task_s == "image_to_image" and model_def is not None and model_def.family == "flux2":
                parameter_metadata = {
                    key: (dict(value) if isinstance(value, dict) else value)
                    for key, value in parameter_metadata.items()
                }
                constraints = dict(parameter_metadata.get("parameter_constraints") or {})
                constraints.pop("guidance_scale", None)
                parameter_metadata["parameter_constraints"] = constraints
                parameters = dict(parameter_metadata.get("parameters") or {})
                nested_constraints = dict(parameters.get("constraints") or {})
                nested_constraints.pop("guidance_scale", None)
                parameters["constraints"] = nested_constraints
                parameter_metadata["parameters"] = parameters
            catalog_rank = (
                int(model_def.image_edit_catalog_rank)
                if task_s == "image_to_image" and model_def is not None
                else 50
            )
            out.append(
                ProviderModelInfo(
                    id=model_selector,
                    object="model",
                    owned_by=MLX_GEN_RUNTIME,
                    capabilities=tuple(tasks),
                    raw={
                        "provider": MLX_GEN_RUNTIME,
                        "model": model_selector,
                        "base_model": base_model,
                        "routed_model": f"{MLX_GEN_RUNTIME}/{model_selector}",
                        "legacy_routed_model": f"{MFLUX_PROVIDER}/{model_selector}",
                        "engine": MLX_GEN_RUNTIME,
                        "legacy_engine": MFLUX_PROVIDER,
                        "runtime_package": MLX_GEN_RUNTIME,
                        "runtime_provider": MLX_GEN_RUNTIME,
                        "target": preset.target,
                        "snapshot_dir": str(discovered_model.snapshot_dir),
                        "repo_id": discovered_model.repo_id or preset.repo_id,
                        "upstream_repo_id": preset.upstream_repo_id,
                        "source": preset.source,
                        "quantization_bits": preset.quantization_bits,
                        "catalog_rank": catalog_rank,
                        "cache_source": discovered_model.source_label,
                        "cache_source_detail": discovered_model.source_detail,
                        "task_specs": task_specs,
                        **parameter_metadata,
                    },
                )
            )
            emitted.add(model_selector)
        if task_s == "image_to_image":
            def _sort_key(info: ProviderModelInfo) -> Tuple[int, int, str]:
                raw = info.raw if isinstance(info.raw, dict) else {}
                try:
                    rank = int(raw.get("catalog_rank", 50))
                except Exception:
                    rank = 50
                bits = raw.get("quantization_bits")
                try:
                    bits_i = int(bits) if bits is not None else 16
                except Exception:
                    bits_i = 16
                bit_rank = {4: 0, 8: 1}.get(bits_i, 2)
                return (rank, bit_rank, str(info.id or "").lower())

            out.sort(key=_sort_key)
        return out

    def list_provider_adapters(
        self,
        *,
        model: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Sequence[ProviderAdapterInfo]:
        task_s = str(task or "").strip()
        selected_model = str(model or self._cfg.model or "").strip()
        catalog = list(self.list_provider_models(task=task_s or None))

        def _model_aliases(info: ProviderModelInfo) -> set[str]:
            raw = info.raw if isinstance(info.raw, dict) else {}
            aliases = {
                str(info.id or "").strip(),
                str(raw.get("model") or "").strip(),
                str(raw.get("repo_id") or "").strip(),
                str(raw.get("upstream_repo_id") or "").strip(),
                str(raw.get("base_model") or "").strip(),
            }
            return {_norm(value) for value in aliases if str(value).strip()}

        selected_info = next(
            (
                info
                for info in catalog
                if selected_model and _norm(selected_model) in _model_aliases(info)
            ),
            None,
        )
        selected_raw = selected_info.raw if selected_info and isinstance(selected_info.raw, dict) else {}
        selected_base = str(selected_raw.get("base_model") or "").strip() or _infer_base_model(
            selected_model, self._cfg.base_model
        )

        infos_by_base: Dict[str, List[ProviderModelInfo]] = {}
        for info in catalog:
            raw = info.raw if isinstance(info.raw, dict) else {}
            base_model = str(raw.get("base_model") or "").strip() or _infer_base_model(
                info.id,
                raw.get("repo_id"),
                raw.get("upstream_repo_id"),
            )
            if base_model:
                infos_by_base.setdefault(base_model, []).append(info)

        def _fallback_models_for_base(base_model: str) -> List[str]:
            out: List[str] = []
            for preset in model_presets(target="mlx", engine="mlx-gen", include_non_8bit=True):
                preset_base = _infer_base_model(preset.repo_id, preset.upstream_repo_id, preset.key)
                if preset_base != base_model:
                    continue
                model_id = _preset_variant_model_id(preset)
                if model_id not in out:
                    out.append(model_id)
            return out

        seen: set[str] = set()
        out: List[ProviderAdapterInfo] = []
        for cache_label, root in _configured_adapter_cache_roots(self._cfg.cache_dir):
            try:
                repo_dirs = list(root.glob("models--*"))
            except Exception:
                repo_dirs = []
            for repo_dir in repo_dirs:
                repo_id = _repo_id_from_cache_dir_name(repo_dir.name)
                if not repo_id:
                    continue
                for snapshot_dir in _adapter_repo_snapshot_dirs(repo_dir):
                    if _skip_adapter_snapshot(snapshot_dir):
                        continue
                    base_models = _adapter_base_models_from_card(snapshot_dir)
                    adapter_files = _adapter_candidate_files(snapshot_dir, repo_id=repo_id)
                    if not adapter_files:
                        continue
                    for file_path in adapter_files:
                        relative_path = file_path.relative_to(snapshot_dir).as_posix()
                        handle = f"{repo_id}:{relative_path}"
                        profile = known_lora_adapter_profile(handle)
                        if isinstance(profile, dict) and str(profile.get("artifact_role") or "adapter") != "adapter":
                            continue
                        profile_base_models = tuple(
                            str(item).strip()
                            for item in (profile or {}).get("base_models") or ()
                            if str(item).strip()
                        )
                        effective_base_models = profile_base_models or base_models
                        compatible_base = str(
                            (profile or {}).get("compatible_base_model") or ""
                        ).strip() or _infer_base_model(handle, repo_id, *effective_base_models)
                        if compatible_base is None:
                            continue
                        if selected_base and compatible_base != selected_base:
                            continue
                        if handle.lower() in seen:
                            continue

                        route_infos = (
                            [selected_info]
                            if selected_info is not None
                            else infos_by_base.get(compatible_base, [])
                        )
                        compatible_models: List[str] = []
                        compatible_tasks: List[str] = []
                        role_union: List[str] = []
                        compatible_routes: List[Dict[str, Any]] = []
                        for info in route_infos:
                            if info is None:
                                continue
                            if info.id not in compatible_models:
                                compatible_models.append(info.id)
                            raw = info.raw if isinstance(info.raw, dict) else {}
                            task_specs = raw.get("task_specs") if isinstance(raw.get("task_specs"), dict) else {}
                            for task_name, task_spec in task_specs.items():
                                if task_s and str(task_name) != task_s:
                                    continue
                                if not bool(task_spec.get("supports_lora")):
                                    continue
                                if task_name not in compatible_tasks:
                                    compatible_tasks.append(str(task_name))
                                roles = [
                                    str(role).strip()
                                    for role in (task_spec.get("lora_target_roles") or ())
                                    if str(role).strip()
                                ]
                                for role_s in roles:
                                    if role_s not in role_union:
                                        role_union.append(role_s)
                                compatible_routes.append(
                                    {
                                        "model": str(info.id or ""),
                                        "task": str(task_name),
                                        "supports_lora": True,
                                        "lora_status": str(task_spec.get("lora_status") or ""),
                                        "lora_target_roles": roles,
                                        "lora_validation_profile": str(
                                            task_spec.get("lora_validation_profile") or ""
                                        )
                                        or None,
                                    }
                                )
                        if not compatible_models:
                            compatible_models = _fallback_models_for_base(compatible_base)
                        if not compatible_tasks:
                            compatible_tasks = [
                                task_name
                                for task_name in self._supported_task_names(
                                    model_id=compatible_models[0] if compatible_models else None,
                                    base_model=compatible_base,
                                )
                                if not task_s or task_name == task_s
                            ]
                        if not compatible_tasks:
                            continue

                        suggested_roles = _adapter_suggested_target_roles(relative_path, role_union)
                        out.append(
                            ProviderAdapterInfo(
                                id=handle,
                                repo_id=repo_id,
                                base_models=effective_base_models,
                                compatible_models=tuple(compatible_models),
                                compatible_tasks=tuple(compatible_tasks),
                                suggested_target_roles=suggested_roles,
                                raw={
                                    "provider": MLX_GEN_RUNTIME,
                                    "engine": MLX_GEN_RUNTIME,
                                    "runtime_package": MLX_GEN_RUNTIME,
                                    "repo_id": repo_id,
                                    "snapshot_dir": str(snapshot_dir),
                                    "cache_source": cache_label,
                                    "relative_path": relative_path,
                                    "compatible_base_model": compatible_base,
                                    "compatible_routes": compatible_routes,
                                    **(
                                        {
                                            "adapter_kind": str(profile.get("kind") or ""),
                                            "adapter_family": str(profile.get("family") or ""),
                                            "recommended_parameters": dict(
                                                profile.get("recommended_parameters") or {}
                                            ),
                                            "compatibility_notes": list(
                                                profile.get("compatibility_notes") or ()
                                            ),
                                            "documentation_urls": list(
                                                profile.get("documentation_urls") or ()
                                            ),
                                            "recommended_local_quantization_bits": list(
                                                profile.get("recommended_local_quantization_bits")
                                                or ()
                                            ),
                                            "model_aliases": list(
                                                profile.get("model_aliases") or ()
                                            ),
                                        }
                                        if isinstance(profile, dict)
                                        else {}
                                    ),
                                },
                            )
                        )
                        seen.add(handle.lower())
        out.sort(key=lambda item: (tuple(item.compatible_models), str(item.repo_id or ""), str(item.id or "")))
        return out

    def _resolve_model(self) -> Tuple[str, str]:
        """(model directory, base model) of the configured model; `_resolve` holds the whole record."""
        resolution = self._resolve()
        return resolution.model_path, resolution.base_model

    def _resolve(self) -> _Resolution:
        # Resolved once per configuration and cache location until `unload()`, and again when the
        # resolved directory is gone: finding a model scans the caches, and whether it is a prepared
        # package reads its shard headers.
        source = (self._cfg, str(_cache_root(self._cfg.cache_dir)), tuple(str(r) for r in framework_hf_cache_roots()))
        _require_runtime_versions()
        cached = self._resolution
        if cached is not None and cached.source == source and Path(cached.model_path).exists():
            return cached
        model_path, base_model, stored_bits = self._find_configured_model()
        self._resolution = _Resolution(
            source=source,
            model_path=model_path,
            base_model=base_model,
            stored_bits=stored_bits,
            prepared=_is_prepared_mlx_gen_package(model_path, self._cfg.model),
        )
        return self._resolution

    def _find_configured_model(self) -> Tuple[str, str, Optional[int]]:
        """(model directory, base model, stored width) of the configured model, else of a cached one."""
        configured_model = str(self._cfg.model or "").strip()
        configured_base = _infer_base_model(self._cfg.base_model)
        cache_root = _cache_root(self._cfg.cache_dir)
        extra_cache_roots = framework_hf_cache_roots()
        discovered = discover_cached_mflux_models(
            model_dir=self._cfg.model_dir,
            cache_dir=self._cfg.cache_dir,
        )

        if configured_model:
            ambiguous_choices = _ambiguous_mlx_gen_selector_choices(configured_model)
            if ambiguous_choices:
                choices = ", ".join(ambiguous_choices)
                raise OptionalDependencyMissingError(
                    f"MLX-Gen model selector {configured_model!r} is ambiguous because it maps to multiple "
                    f"prepared AbstractFramework q4/q8 repos. Use an exact model id: {choices}."
                )
            expanded = Path(configured_model).expanduser()
            if expanded.exists():
                base = configured_base or _infer_base_model(configured_model, expanded.name)
                if not base:
                    raise OptionalDependencyMissingError(
                        "Could not infer MLX-Gen base model from local path. "
                        "Set vision_mflux_base_model / ABSTRACTVISION_MFLUX_BASE_MODEL "
                        "to the matching base family for that local path."
                    )
                return str(expanded), base, _infer_bits_from_text(configured_model)

            preset = _preset_for(configured_model)
            if preset is not None:
                discovered_model = _cached_model_for_exact_preset(
                    preset,
                    model_dir=self._cfg.model_dir,
                    cache_dir=cache_root,
                )
                if discovered_model is None and not _selection_requires_exact_preset(
                    configured_model, preset
                ):
                    discovered_model = discovered.get(preset.key)
                snapshot_dir = (
                    discovered_model.snapshot_dir if discovered_model is not None else None
                )
                preset_bits = int(preset.quantization_bits) if preset.quantization_bits is not None else None
                if snapshot_dir is not None and _has_model_files(snapshot_dir):
                    return (
                        str(snapshot_dir),
                        _base_model_for_preset(configured_base, configured_model, preset),
                        preset_bits,
                    )
                explicit_note = _explicit_download_note(
                    preset.repo_id,
                    _base_model_for_preset(configured_base, configured_model, preset),
                )
                if not self._cfg.allow_download or explicit_note:
                    raise OptionalDependencyMissingError(
                        f"MLX-Gen model preset {configured_model!r} is not available in the Hugging Face cache."
                        + (explicit_note or f" Run: {_download_instruction(preset.repo_id)}.")
                    )
                try:
                    downloaded = download_hf_repo_snapshot(
                        preset.repo_id,
                        allow_patterns=list(preset.allow_patterns) or None,
                        token=None,
                        cache_dir=cache_root,
                    )
                except RuntimeError as e:
                    raise OptionalDependencyMissingError(str(e)) from e
                return (
                    str(downloaded),
                    _base_model_for_preset(configured_base, configured_model, preset),
                    preset_bits,
                )

            if _looks_like_path(configured_model):
                raise OptionalDependencyMissingError(
                    f"MLX-Gen model path does not exist: {_public_model_name(configured_model)}"
                )
            if looks_like_hf_repo_id(configured_model):
                cached = resolve_hf_repo_snapshot(
                    configured_model,
                    cache_dir=cache_root,
                    extra_roots=extra_cache_roots,
                )
                if cached is not None:
                    base = configured_base or _infer_base_model(configured_model)
                    if not base:
                        raise OptionalDependencyMissingError(
                            "Could not infer MLX-Gen base model. Set vision_mflux_base_model / "
                            "ABSTRACTVISION_MFLUX_BASE_MODEL."
                        )
                    return str(cached), base, _infer_bits_from_text(configured_model)
                if not self._cfg.allow_download:
                    raise OptionalDependencyMissingError(
                        f"MLX-Gen model repo {configured_model!r} is not cached locally. "
                        "Pre-download it with `abstractvision download <org/name>` "
                        "or set ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD=1 to permit downloads."
                    )
                base = configured_base or _infer_base_model(configured_model)
                if not base:
                    raise OptionalDependencyMissingError(
                        "Could not infer MLX-Gen base model. Set vision_mflux_base_model / "
                        "ABSTRACTVISION_MFLUX_BASE_MODEL."
                    )
                explicit_note = _explicit_download_note(configured_model, base)
                if explicit_note:
                    raise OptionalDependencyMissingError(
                        f"MLX-Gen model repo {configured_model!r} is not cached locally."
                        + explicit_note
                    )
                try:
                    downloaded = download_hf_repo_snapshot(
                        configured_model,
                        cache_dir=cache_root,
                    )
                except RuntimeError as e:
                    raise OptionalDependencyMissingError(str(e)) from e
                return str(downloaded), base, _infer_bits_from_text(configured_model)

            if not self._cfg.allow_download:
                other = _first_other_engine_preset(configured_model)
                if other is not None:
                    raise OptionalDependencyMissingError(
                        f"Model {configured_model!r} maps to a curated preset for engine {other.engine!r} "
                        f"(target={other.target!r}, repo={other.repo_id!r}), not MLX-Gen. "
                        "Use `--provider diffusers` or `--provider sdcpp` as appropriate, or pass an exact "
                        "AbstractFramework MLX-Gen repo id / local path / org/name repo id."
                    )
                raise OptionalDependencyMissingError(
                    f"MLX-Gen model {configured_model!r} is not a known downloaded preset. "
                    "Use a local model path, an exact AbstractFramework MLX-Gen repo id, a Hugging Face repo id already cached "
                    "locally, or set ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD=1 to permit downloads."
                )
            base = configured_base or _infer_base_model(configured_model)
            if not base:
                raise OptionalDependencyMissingError(
                    "Could not infer MLX-Gen base model. Set vision_mflux_base_model / "
                    "ABSTRACTVISION_MFLUX_BASE_MODEL."
                )
            raise OptionalDependencyMissingError(
                f"MLX-Gen model {configured_model!r} is not a known cached preset, path, or Hugging Face repo id."
            )

        for key in (
            "flux2-klein-4b",
            "flux2-klein-9b",
            "bonsai-image-ternary",
            "z-image-turbo",
            "qwen-image",
            "ernie-image-turbo",
            "fibo-lite",
            "fibo",
            SEEDVR2_3B_MODEL_KEY,
            SEEDVR2_7B_MODEL_KEY,
            WAN_TI2V_MODEL_KEY,
            WAN_T2V_A14B_MODEL_KEY,
            WAN_I2V_A14B_MODEL_KEY,
        ):
            discovered_model = discovered.get(key)
            if discovered_model is not None:
                return (
                    str(discovered_model.snapshot_dir),
                    configured_base or key,
                    _infer_bits_from_text(discovered_model.repo_id),
                )

        raise OptionalDependencyMissingError(
            "MLX-Gen backend is not configured and no downloaded MLX-Gen preset was found. "
            "Set vision_mflux_model / ABSTRACTVISION_MFLUX_MODEL or run "
            "`abstractvision download AbstractFramework/flux.2-klein-4b-4bit --provider mlx-gen`."
        )

    def _ensure_model_impl(
        self,
        *,
        edit_variant: bool = False,
        lora_adapters: Sequence[Any] = (),
        stream_blocks: bool = False,
        quantize: Optional[int] = None,
    ) -> Tuple[Any, _MFluxModelDef]:
        model_def = self._resolved_model_def()
        model_path, _base_model = self._resolve_model()
        ModelConfig, _DownloadRequiredError, Flux2Klein, _Flux2KleinEdit, ZImage, ZImageTurbo = (
            _lazy_import_mflux()
        )
        if quantize is not None:
            quantize = self._checked_load_quantize(quantize)
            if stream_blocks:
                raise CapabilityNotSupportedError(
                    "memory_mode='lowest' streams the stored weights from disk, which load-time "
                    "quantize replaces; use a prepared package (or the 16-bit source) without quantize."
                )
        model_variant = "flux2-edit" if edit_variant and model_def.family == "flux2" else "default"
        effective_lora_adapters = tuple(lora_adapters) if lora_adapters else self._config_lora_adapters()
        lora_init_kwargs = self._mlx_gen_lora_init_kwargs(effective_lora_adapters)
        if model_def.family != "wan-video":
            lora_init_kwargs.pop("lora_target_roles", None)
        key = _ModelKey(
            model_path=model_path,
            base_model=model_def.key,
            variant=model_variant,
            lora_paths=tuple(lora_init_kwargs.get("lora_paths") or ()),
            lora_scales=tuple(lora_init_kwargs.get("lora_scales") or ()),
            lora_target_roles=tuple(lora_init_kwargs.get("lora_target_roles") or ()),
            stream_blocks=bool(stream_blocks),
            quantize=quantize,
        )
        if self._model is not None and self._model_key == key:
            return self._model, model_def
        self._release_model()

        model_config = self._model_config_for_definition(ModelConfig, model_def, model_path=model_path)
        if model_def.family == "flux2":
            cls = _Flux2KleinEdit if model_variant == "flux2-edit" else Flux2Klein
        elif model_def.family == "bonsai":
            cls = _lazy_import_mflux_bonsai()
        elif model_def.family == "z-image":
            cls = ZImage if model_def.key == "z-image" else (ZImageTurbo or ZImage)
        elif model_def.family == "qwen":
            QwenImage, _QwenImageEdit = _lazy_import_mflux_qwen()
            cls = QwenImage
        elif model_def.family == "qwen-edit":
            _QwenImage, QwenImageEdit = _lazy_import_mflux_qwen()
            cls = QwenImageEdit
        elif model_def.family == "ernie-image":
            cls = _lazy_import_mflux_ernie()
        elif model_def.family == "fibo":
            FIBO, _FIBOEdit = _lazy_import_mflux_fibo()
            cls = FIBO
        elif model_def.family == "fibo-edit":
            _FIBO, FIBOEdit = _lazy_import_mflux_fibo()
            cls = FIBOEdit
        elif model_def.family == "wan-video":
            cls = _lazy_import_mflux_wan()
        elif model_def.family == "minimax-h3":
            cls = _lazy_import_mflux_minimax_h3()
        else:
            raise OptionalDependencyMissingError(
                f"Unsupported MLX-Gen model family {model_def.family!r}."
            )
        kwargs: Dict[str, Any] = {
            "model_config": model_config,
            "model_path": model_path,
        }
        kwargs.update(lora_init_kwargs)
        if stream_blocks:
            # Only reached when the route publishes supports_block_streaming (see _memory_plan).
            kwargs["stream_blocks"] = True
        if quantize is not None:
            kwargs["quantize"] = int(quantize)
        try:
            self._model = cls(**kwargs)
        except Exception as e:
            if _is_mlx_gen_download_required(e):
                raise _wrap_mlx_gen_download_required(e) from e
            raise
        if stream_blocks and getattr(self._model, "block_stream", True) is None:
            # MLX-Gen keeps the transformer in memory when the weights cannot stream.
            self._release_model()
            raise CapabilityNotSupportedError(
                "memory_mode='lowest' needs weights that stream from disk (a prepared package, or the "
                "16-bit source without quantize); MLX-Gen could not stream these.",
                field="memory_mode", supported=(MEMORY_MODE_DEFAULT, MEMORY_MODE_LOW),
            )
        self._model_key = key
        self._warmed_model_key = None
        return self._model, model_def

    def quantize_bits(self, task: Optional[str] = None) -> Tuple[int, ...]:
        """Load-time `quantize` widths the configured model accepts (empty: refused).

        A route that publishes `validated_quantization_bits` accepts those; other routes accept
        MLX-Gen's `--quantize` choices. A prepared package accepts none, nor does a host without the
        MLX-Gen runtime.
        """
        resolution = self._resolution
        prepared = (
            resolution.prepared
            if resolution is not None
            else _is_prepared_mlx_gen_package(self._cfg.model, self._cfg.model)
        )
        if prepared:
            return ()
        rows = self._route_rows(task)
        validated = sorted(
            {int(bits) for row in rows for bits in (row.get("validated_quantization_bits") or ())}
        )
        if validated:
            return tuple(validated)
        try:
            if self._configured_family() == "minimax-h3":
                # Without rows, MiniMax-H3's own quantization policies (never the generic choices).
                return tuple(sorted(_lazy_import_mlx_gen_h3_planning().quantization_bits))
            return _mlx_gen_quantize_choices()
        except OptionalDependencyMissingError:
            return ()

    def _configured_family(self) -> Optional[str]:
        model_def = self._configured_model_def()
        return model_def.family if model_def is not None else None

    def _configured_model_def(self) -> Optional[_MFluxModelDef]:
        # The configured (or last resolved) model, without resolving it: no cache scan, no download.
        base_model = str(
            self._resolved_base_model or _infer_base_model(self._cfg.base_model, self._cfg.model) or ""
        )
        return _MFLUX_MODELS.get(base_model)

    def _checked_load_quantize(self, quantize: Any) -> int:
        """The one check of a load-time `quantize`, for every family, against the resolved model.

        A bool or a non-integral number is a ValueError naming the field, as for every integer option.
        """
        bits = _request_int("quantize", quantize, label="MLX-Gen")
        resolution = self._resolve()
        if resolution.prepared:
            raise CapabilityNotSupportedError(
                f"{_public_model_name(resolution.model_path)} is a prepared MLX-Gen package whose stored weights are its "
                "quantization; load it without quantize, or quantize the source snapshot.",
                field="quantize", supported=(),
            )
        allowed = self.quantize_bits()
        if bits not in allowed:
            raise CapabilityNotSupportedError(
                f"quantize={bits} is not accepted for this model (accepted: "
                f"{', '.join(str(value) for value in allowed)}).",
                field="quantize", supported=allowed,
            )
        return bits

    def _ensure_seedvr2_model_impl(
        self,
        *,
        quantize: Optional[int],
    ) -> Tuple[Any, _MFluxModelDef]:
        model_def = self._resolved_model_def()
        model_path, _base_model = self._resolve_model()
        if model_def.family != "seedvr2-upscale":
            raise CapabilityNotSupportedError(
                "MLX-Gen image_upscale is implemented for SeedVR2 models today "
                f"(got {model_def.family!r})."
            )
        ModelConfig, SeedVR2, _ScaleFactor, _TilingConfig = _lazy_import_mflux_seedvr2()
        if quantize is not None:
            quantize = self._checked_load_quantize(quantize)
        key = _ModelKey(model_path=model_path, base_model=model_def.key, variant="seedvr2-upscale", quantize=quantize)
        if self._model is not None and self._model_key == key:
            return self._model, model_def
        self._release_model()

        model_config = self._model_config_for_definition(ModelConfig, model_def)
        try:
            self._model = SeedVR2(
                quantize=quantize,
                model_path=model_path,
                model_config=model_config,
            )
        except Exception as e:
            if _is_mlx_gen_download_required(e):
                raise _wrap_mlx_gen_download_required(e) from e
            raise
        self._model_key = key
        self._warmed_model_key = None
        return self._model, model_def

    def _warmup_request(
        self, model_def: _MFluxModelDef, mode: str, quantize: Optional[int]
    ) -> ImageGenerationRequest:
        return ImageGenerationRequest(
            memory_mode=mode,
            prompt="abstractvision preload warmup",
            steps=int(model_def.default_steps),
            guidance_scale=model_def.default_guidance,
            seed=0,
            quantize=quantize,
        )

    def _preload_impl(self, memory_mode: Optional[str], quantize: Optional[int]) -> None:
        # The values are checked as a request of the model's task carrying them is, before the model is
        # resolved (which may download it) or anything loads: a refused preload leaves the backend as it was.
        mode, quantize = self._resolve_preload_options(memory_mode, quantize, task=self._preload_task())
        held = self._model
        try:
            self._preload_model(mode, quantize)
        except BaseException:
            # The instance this preload built is dropped, never left in memory without the caller knowing it
            # is there; one held before the call, which the preload reused, stays.
            if self._model is not None and self._model is not held:
                self._release_model()
            raise

    def _preload_model(self, mode: str, quantize: Optional[int]) -> None:
        model_def = self._resolved_model_def()
        task = self._preload_task()
        plan = self._memory_plan(mode)
        # Loaded as the first request with these values starts, under the same cache hold, into the
        # instance that request reuses. In a mode whose runs drop the model, that request drops it:
        # the preloaded instance serves one request, and no warm-up runs.
        with self._memory_session(replace(plan, releases_model=False)):
            if model_def.family == "seedvr2-upscale":
                self._ensure_seedvr2_model_impl(
                    quantize=quantize if quantize is not None else self._default_upscale_quantize()
                )
            else:
                self._ensure_model_impl(
                    edit_variant=task == "image_to_image", stream_blocks=plan.stream_blocks, quantize=quantize
                )
        if self._model_key is not None and self._warmed_model_key == self._model_key:
            return
        if task != "text_to_image" or plan.releases_model:
            # Only a text-to-image model has a warm-up (one image generation); a video, edit-only or
            # upscale model's first run compiles its own kernels.
            self._mark_preloaded()
            return
        self._generate_impl(self._warmup_request(model_def, mode, quantize))

    def _preload_task(self) -> str:
        # The task whose requests a preload serves: the first of these the model has a route for.
        tasks = self._supported_task_names()
        return next((task for task in _PRELOAD_TASK_ORDER if task in tasks), "text_to_image")

    def _mark_preloaded(self) -> None:
        self._warmed_model_key = self._model_key

    def _resolved_model_def(self) -> _MFluxModelDef:
        _model_path, base_model = self._resolve_model()
        if base_model not in _MFLUX_MODELS:
            raise OptionalDependencyMissingError(
                f"Unsupported MLX-Gen base model {base_model!r}. "
                f"Supported: {', '.join(sorted(_MFLUX_MODELS))}"
            )
        return _MFLUX_MODELS[base_model]

    def _model_config_for_definition(
        self, ModelConfig: Any, model_def: _MFluxModelDef, *, model_path: Optional[str] = None
    ) -> Any:
        if model_def.family == "minimax-h3":
            # As `mlxgen generate --model <package or path> --base-model <entry>`: the entry picks
            # the schedule (and attaches its Turbo adapter) for the weights at model_path.
            return ModelConfig.from_name(str(model_path or MINIMAX_H3_REPO_ID), base_model=model_def.key)
        config_factory = getattr(ModelConfig, model_def.config_method, None)
        from_name = getattr(ModelConfig, "from_name", None)
        registry_id = _MFLUX_BASE_MODEL_REGISTRY_IDS.get(model_def.key, model_def.key)
        from_name_exc: Optional[Exception] = None
        if callable(from_name):
            try:
                return from_name(registry_id)
            except Exception as exc:
                from_name_exc = exc
        if callable(config_factory):
            return config_factory()
        if not callable(from_name):
            raise OptionalDependencyMissingError(
                f"Installed MLX-Gen does not expose ModelConfig.{model_def.config_method}() "
                f"or ModelConfig.from_name(); update mlx-gen for {registry_id}."
            )
        if from_name_exc is not None:
            if model_def.family == "wan-video":
                raise OptionalDependencyMissingError(
                    f"MLX-Gen Wan video generation requires mlx-gen>={MLX_GEN_VERSION_FLOOR}. "
                    'Install/upgrade it with `pip install "abstractvision[mlx-gen]"` '
                    f"for {registry_id}."
                ) from from_name_exc
            if model_def.family == "seedvr2-upscale":
                raise OptionalDependencyMissingError(
                    f"MLX-Gen SeedVR2 image upscaling requires mlx-gen>={MLX_GEN_VERSION_FLOOR}. "
                    'Install/upgrade it with `pip install "abstractvision[mlx-gen]"` '
                    f"for {registry_id}."
                ) from from_name_exc
            raise from_name_exc
        raise OptionalDependencyMissingError(
            f"Could not resolve ModelConfig for {registry_id}."
        )

    def normalize_image_generation_request(
        self,
        request: ImageGenerationRequest,
    ) -> ImageGenerationRequest:
        return self._run_on_runtime_thread(
            self._normalize_once, request, self._normalize_image_generation, task="text_to_image"
        )

    def _normalize_image_generation(self, request: ImageGenerationRequest) -> ImageGenerationRequest:
        for name in ("control_image", "control_strength"):
            if getattr(request, name) is not None or (request.extra or {}).get(name) is not None:
                raise CapabilityNotSupportedError(
                    f"The MLX-Gen backend does not support structured control images ({name}).",
                    field=name,
                    supported=(),
                )
        # The mode and options are checked before the model is resolved, which may download it.
        request, _mode = self.resolve_memory_request(request, task="text_to_image")
        model_def = self._resolved_model_def()
        adapter_defaults = recommended_lora_request_overrides(
            request.lora_adapters,
            extra=request.extra,
            task="text_to_image",
            model=model_def.key,
        )
        steps = (
            int(request.steps)
            if request.steps is not None
            else int(adapter_defaults.get("steps") or model_def.default_steps)
        )
        if model_def.family == "flux2" and steps < 2:
            steps = 2
        guidance = (
            float(request.guidance_scale)
            if request.guidance_scale is not None
            else adapter_defaults.get("guidance_scale", model_def.default_guidance)
        )
        if (
            not model_def.supports_guidance_override
            and model_def.default_guidance is not None
        ):
            guidance = float(model_def.default_guidance)
        negative_prompt = request.negative_prompt
        if negative_prompt and not model_def.supports_negative_prompt:
            negative_prompt = None
        return replace(
            request,
            steps=steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt,
        )

    def normalize_image_edit_request(
        self,
        request: ImageEditRequest,
    ) -> ImageEditRequest:
        return self._run_on_runtime_thread(
            self._normalize_once, request, self._normalize_image_edit, task="image_to_image"
        )

    def _normalize_image_edit(self, request: ImageEditRequest) -> ImageEditRequest:
        # The mode and options are checked before the model is resolved, which may download it.
        request, _mode = self.resolve_memory_request(request, task="image_to_image")
        model_def = self._resolved_model_def()
        adapter_defaults = recommended_lora_request_overrides(
            request.lora_adapters,
            extra=request.extra,
            task="image_to_image",
            model=model_def.key,
        )
        steps = (
            int(request.steps)
            if request.steps is not None
            else int(adapter_defaults.get("steps") or model_def.default_steps)
        )
        if model_def.family == "flux2" and steps < 2:
            steps = 2
        guidance = (
            float(request.guidance_scale)
            if request.guidance_scale is not None
            else adapter_defaults.get("guidance_scale", model_def.default_guidance)
        )
        if (
            not model_def.supports_guidance_override
            and model_def.default_guidance is not None
            and model_def.family != "flux2"
        ):
            guidance = float(model_def.default_guidance)
        negative_prompt = request.negative_prompt
        if negative_prompt and not model_def.supports_negative_prompt:
            negative_prompt = None

        extra = dict(request.extra or {})
        if "image_strength" not in extra and "strength" in extra:
            extra["image_strength"] = extra.get("strength")
        if "image_strength" in extra and extra.get("image_strength") is not None:
            try:
                strength = float(extra.get("image_strength"))
                if strength < 0.0:
                    strength = 0.0
                if strength > 1.0:
                    strength = 1.0
                extra["image_strength"] = strength
            except Exception:
                extra.pop("image_strength", None)
        for key in ("width", "height"):
            if key in extra and extra.get(key) is not None:
                try:
                    extra[key] = int(extra[key])
                except Exception:
                    extra.pop(key, None)
        return replace(
            request,
            steps=steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt,
            extra=extra,
        )

    def _normalize_video_request_values(
        self,
        *,
        request: Union[VideoGenerationRequest, ImageToVideoRequest],
    ) -> Tuple[Union[VideoGenerationRequest, ImageToVideoRequest], Dict[str, Any]]:
        # A first frame is opened and decoded before anything else is checked.
        keyframe = _checked_keyframe_size(request.image) if isinstance(request, ImageToVideoRequest) else None
        # The mode and options are checked before the model is resolved, which may download it.
        request, _mode = self.resolve_memory_request(
            request, task="image_to_video" if isinstance(request, ImageToVideoRequest) else "text_to_video"
        )
        model_def = self._resolved_model_def()
        if model_def.family == "minimax-h3":
            return request, self._normalize_h3_request_values(request=request, keyframe=keyframe)
        if model_def.family != "wan-video":
            raise CapabilityNotSupportedError(
                "MLX-Gen video generation is implemented for Wan and MiniMax-H3 models "
                f"(got {model_def.family!r})."
            )
        extra = dict(request.extra or {})
        task_name = "image_to_video" if isinstance(request, ImageToVideoRequest) else "text_to_video"
        adapter_defaults = recommended_lora_request_overrides(
            request.lora_adapters,
            extra=request.extra,
            task=task_name,
            model=model_def.key,
        )
        label = "Wan"
        steps = (
            _request_int("steps", request.steps, label=label)
            if request.steps is not None
            else int(adapter_defaults.get("steps") or model_def.default_steps)
        )
        if steps < 1:
            steps = 1
        guidance = (
            _request_number("guidance_scale", request.guidance_scale, label=label)
            if request.guidance_scale is not None
            else model_def.default_guidance
        )
        width = (
            _request_int("width", request.width, label=label)
            if request.width is not None
            else int(model_def.default_width)
        )
        height = (
            _request_int("height", request.height, label=label)
            if request.height is not None
            else int(model_def.default_height)
        )
        self._validate_wan_video_dimensions(model_def=model_def, width=width, height=height)
        fps = _request_int("fps", request.fps, label=label) if request.fps is not None else int(model_def.default_fps)
        num_frames = (
            _request_int("num_frames", request.num_frames, label=label)
            if request.num_frames is not None
            else int(model_def.default_frames)
        )
        if fps < 1:
            fps = int(model_def.default_fps)
        if num_frames < 1:
            num_frames = 1
        if extra.get("max_sequence_length") is not None:
            extra["max_sequence_length"] = max(
                1, _request_int("max_sequence_length", extra["max_sequence_length"], label=label)
            )
        guidance_2 = (
            _request_number("guidance_2", request.guidance_2, label=label) if request.guidance_2 is not None else None
        )
        flow_shift_value = request.flow_shift if request.flow_shift is not None else extra.get("flow_shift")
        flow_shift = (
            _request_number("flow_shift", flow_shift_value, label=label) if flow_shift_value is not None else None
        )
        if flow_shift is None and model_def.key == WAN_TI2V_MODEL_KEY:
            longer = max(width, height)
            shorter = min(width, height)
            if shorter <= 480 and longer <= 832:
                flow_shift = WAN_TI2V_480P_FLOW_SHIFT
        if flow_shift is not None:
            flow_shift = float(flow_shift)
            extra["flow_shift"] = flow_shift
        return request, {
            "steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "fps": fps,
            "num_frames": num_frames,
            "flow_shift": flow_shift,
            "guidance_2": guidance_2,
            "seed": _request_seed(request.seed, label=label),
            "extra": extra,
        }

    def _h3_route_row(self, task: str) -> Dict[str, Any]:
        rows = self._route_rows(task)
        if not rows:
            raise CapabilityNotSupportedError(
                f"MLX-Gen publishes no MiniMax-H3 {task} route in this build; "
                f"upgrade mlx-gen to {MLX_GEN_VERSION_FLOOR} or later."
            )
        return next((row for row in rows if row.get("default_for_task")), rows[0])

    def _normalize_h3_request_values(
        self,
        *,
        request: Union[VideoGenerationRequest, ImageToVideoRequest],
        keyframe: Optional[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """Check a MiniMax-H3 request against MLX-Gen's capability row and its own planning rules.

        Runs before any weights load. Values the caller left unset stay unset, so MLX-Gen applies
        the entry's own defaults (steps, canvas, frames, shifts), as `mlxgen generate` does; what
        is set reaches the engine as the type it takes. What the route cannot apply, or a value
        outside what it accepts, is refused with CapabilityNotSupportedError; a value of the wrong
        type, or two fields that contradict each other, with ValueError. Both name the field.
        """
        task = "image_to_video" if isinstance(request, ImageToVideoRequest) else "text_to_video"
        label = "MiniMax-H3"
        row = self._h3_route_row(task)
        refused: Dict[str, str] = {}  # field -> what is refused
        if request.negative_prompt and not row.get("supports_negative_prompt"):
            refused["negative_prompt"] = "negative_prompt"
        if not row.get("supports_guidance"):
            if request.guidance_scale is not None:
                refused["guidance_scale"] = "guidance_scale"
            if request.guidance_2 is not None:
                refused["guidance_2"] = "guidance_2"
        output_fps = row.get("output_fps")
        if request.fps is not None:
            fps = _request_number("fps", request.fps, label=label)
            if output_fps is None or fps != float(output_fps):
                refused["fps"] = f"fps={request.fps} (the route writes {output_fps} fps)"
        extra = dict(request.extra or {})
        unknown = sorted(key for key in extra if key not in _ROUTING_EXTRA_KEYS)
        if unknown:
            refused["extra"] = f"extra keys {', '.join(unknown)}"
        if refused:
            # One refused field is named, with what the route accepts there; several are not.
            only = next(iter(refused)) if len(refused) == 1 else None
            accepted = {"fps": (float(output_fps),) if output_fps is not None else ()}
            raise CapabilityNotSupportedError(
                f"MiniMax-H3 {task} does not support: {'; '.join(refused.values())}.",
                field=only,
                supported=accepted.get(only, ()) if only not in (None, "extra") else None,
            )
        prompt = request.prompt
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"MiniMax-H3 prompt must be a string, got {type(prompt).__name__}.")
        if not str(prompt or "").strip():
            raise ValueError("MiniMax-H3 needs a prompt (the integrated_multimodal_description).")
        texts = {}
        for name in ("soundscape", "music"):
            value = getattr(request, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"MiniMax-H3 {name} must be a string, got {type(value).__name__}.")
            texts[name] = value
        generate_audio = request.generate_audio
        if generate_audio is not None and not isinstance(generate_audio, bool):
            raise ValueError(
                f"MiniMax-H3 generate_audio must be true or false, got {generate_audio!r}."
            )
        width = height = None
        if (request.width is None) != (request.height is None):
            raise ValueError("MiniMax-H3 width and height must be given together, or neither.")
        if request.width is not None:
            width = _request_int("width", request.width, label=label)
            height = _request_int("height", request.height, label=label)
            multiple = int(row.get("dimension_multiple") or 1)
            if width <= 0 or height <= 0 or width % multiple or height % multiple:
                raise CapabilityNotSupportedError(
                    f"MiniMax-H3 width and height must be positive multiples of {multiple}, "
                    f"got {width}x{height}.",
                    field="width",
                )
        num_frames = None
        if request.num_frames is not None:
            num_frames = _request_int("num_frames", request.num_frames, label=label)
            accepted = _h3_frame_counts(row)
            aligned = _h3_aligned_frames(num_frames, row)
            if num_frames < 1 or aligned not in accepted:
                raise CapabilityNotSupportedError(
                    f"MiniMax-H3 num_frames rounds up to the next "
                    f"{row.get('frame_multiple')}n+{row.get('frame_remainder')} count; "
                    f"{num_frames} rounds to {aligned}, outside the accepted counts "
                    f"{', '.join(str(count) for count in accepted)}.",
                    field="num_frames", supported=accepted,
                )
        steps = None
        if request.steps is not None:
            steps = _request_int("steps", request.steps, label=label)
            if steps < 1:
                raise CapabilityNotSupportedError(
                    f"MiniMax-H3 steps must be at least 1, got {steps}.", field="steps"
                )
        shifts = {}
        for name in ("flow_shift", "audio_shift"):
            value = getattr(request, name)
            shifts[name] = None if value is None else _request_number(name, value, label=label)
            if shifts[name] is not None and shifts[name] <= 0:
                # The rectified-flow time shift is a positive factor.
                raise CapabilityNotSupportedError(
                    f"MiniMax-H3 {name} must be positive, got {value!r}.", field=name
                )
        seed = _request_seed(request.seed, label=label)
        planning = _lazy_import_mlx_gen_h3_planning()
        if width is not None:
            self._check_h3_aspect_ratio(width, height, planning=planning, what="width x height")
        elif keyframe is not None:
            # MLX-Gen maps the keyframe's own aspect ratio onto its canvas, within its window.
            self._check_h3_aspect_ratio(*keyframe, planning=planning, what="the keyframe's width x height")
        try:
            # MLX-Gen's own composition, for its verdict on a section given twice.
            planning.compose_prompt(str(prompt), texts["soundscape"], texts["music"])
        except ValueError as e:
            fields = [name for name in ("soundscape", "music") if texts[name] and texts[name].strip()]
            raise ValueError(f"MiniMax-H3 prompt and {' / '.join(fields) or 'prompt'}: {e}") from None
        return {
            "steps": steps,
            "guidance_scale": None,
            "width": width,
            "height": height,
            "fps": request.fps,
            "num_frames": num_frames,
            "flow_shift": shifts["flow_shift"],
            "audio_shift": shifts["audio_shift"],
            "seed": seed,
            "generate_audio": generate_audio,
            "extra": extra,
        }

    @staticmethod
    def _check_h3_aspect_ratio(width: int, height: int, *, planning: Any, what: str) -> None:
        low, high = float(planning.min_aspect_ratio), float(planning.max_aspect_ratio)
        if not low <= width / height <= high:
            raise CapabilityNotSupportedError(
                f"MiniMax-H3 aspect ratio ({what} = {width}x{height}) must lie within "
                f"[{low:g}, {high:g}] (MLX-Gen's window).",
                field="width",
            )

    def _validate_wan_video_dimensions(
        self,
        *,
        model_def: _MFluxModelDef,
        width: int,
        height: int,
    ) -> None:
        if width <= 0 or height <= 0:
            raise CapabilityNotSupportedError(
                "Wan video width and height must be positive integers.", field="width"
            )
        multiple = 32 if model_def.key == WAN_TI2V_MODEL_KEY else 16
        if width % multiple != 0 or height % multiple != 0:
            raise CapabilityNotSupportedError(
                f"{model_def.key} requires width/height multiples of {multiple}. "
                f"Received {width}x{height}.",
                field="width",
            )
        if model_def.key != WAN_TI2V_MODEL_KEY:
            return
        longer = max(width, height)
        shorter = min(width, height)
        if longer < WAN_TI2V_PRACTICAL_MIN_WIDTH or shorter < WAN_TI2V_PRACTICAL_MIN_HEIGHT:
            raise CapabilityNotSupportedError(
                "Wan2.2 TI2V-5B should not be run below 832x480 or 480x832 through AbstractVision. "
                f"Received {width}x{height}. Use the task-specific Wan A14B routes for lower-cost "
                "validation sizes such as 480x240.",
                field="width",
            )

    def normalize_video_generation_request(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationRequest:
        return self._run_on_runtime_thread(self._normalize_once, request, self._normalize_video, task="text_to_video")

    def normalize_image_to_video_request(
        self,
        request: ImageToVideoRequest,
    ) -> ImageToVideoRequest:
        return self._run_on_runtime_thread(self._normalize_once, request, self._normalize_video, task="image_to_video")

    def _normalize_video(
        self, request: Union[VideoGenerationRequest, ImageToVideoRequest]
    ) -> Union[VideoGenerationRequest, ImageToVideoRequest]:
        resolved, values = self._normalize_video_request_values(request=request)
        return replace(resolved, **values)

    def _normalize_upscale_scale_value(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        if text.endswith("x"):
            return text
        try:
            numeric = float(text)
        except Exception:
            return text
        if numeric <= 0:
            raise ValueError("image_upscale scale must be positive.")
        if numeric.is_integer():
            return f"{int(numeric)}x"
        return f"{numeric:g}x"

    def _normalize_upscale_bool(self, value: Any, *, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"", "0", "false", "no", "off", "none"}:
            return False
        if text in {"1", "true", "yes", "on"}:
            return True
        return bool(value)

    def _default_upscale_quantize(self) -> Optional[int]:
        # A prepared package already stores its quantization; the 16-bit source is quantized to 8.
        resolution = self._resolve()
        if getattr(_MFLUX_MODELS.get(resolution.base_model), "family", None) != "seedvr2-upscale":
            return None
        return None if resolution.prepared else 8

    def normalize_image_upscale_request(
        self,
        request: ImageUpscaleRequest,
    ) -> ImageUpscaleRequest:
        return self._run_on_runtime_thread(
            self._normalize_once, request, self._normalize_image_upscale, task="image_upscale"
        )

    def _normalize_image_upscale(self, request: ImageUpscaleRequest) -> ImageUpscaleRequest:
        # The mode and options are checked before the model is resolved, which may download it.
        request, _mode = self.resolve_memory_request(request, task="image_upscale")
        model_def = self._resolved_model_def()
        if model_def.family != "seedvr2-upscale":
            raise CapabilityNotSupportedError(
                "MLX-Gen image_upscale is implemented for SeedVR2 models today "
                f"(got {model_def.family!r})."
            )
        extra = dict(request.extra or {})
        resolution = request.resolution
        if resolution is None and "resolution" in extra:
            resolution = extra.pop("resolution")
        scale = request.scale
        if scale is None and "scale" in extra:
            scale = extra.pop("scale")
        if resolution is None:
            resolution = self._normalize_upscale_scale_value(scale) or "2x"
        softness = request.softness
        if softness is None and "softness" in extra:
            softness = extra.pop("softness")
        softness_f = 0.25 if softness is None else float(softness)
        if softness_f < 0.0:
            softness_f = 0.0
        if softness_f > 1.0:
            softness_f = 1.0
        # Unset: the model's default (8 on the source, none on a prepared package); an unquantized spelling: loaded
        # unquantized; any other value goes as given to the load's one check (`_checked_load_quantize`).
        quantize_raw = request.quantize if request.quantize is not None else extra.pop("quantize", None)
        if quantize_raw is None:
            quantize_value = self._default_upscale_quantize()
        elif str(quantize_raw).strip().lower() in _UPSCALE_UNQUANTIZED_SPELLINGS:
            quantize_value = None
        else:
            quantize_value = quantize_raw
        vae_tiling_raw = request.vae_tiling if request.vae_tiling is not None else extra.pop("vae_tiling", None)
        vae_tiling_value = (
            None if vae_tiling_raw is None else self._normalize_upscale_bool(vae_tiling_raw, default=False)
        )
        return replace(
            request,
            resolution=resolution,
            scale=scale,
            softness=softness_f,
            quantize=quantize_value,
            vae_tiling=vae_tiling_value,
            extra=extra,
        )

    def _coerce_seedvr2_resolution(self, resolution: Any) -> Any:
        if isinstance(resolution, str):
            text = resolution.strip().lower()
            if text.endswith("x"):
                _ModelConfig, _SeedVR2, ScaleFactor, _TilingConfig = _lazy_import_mflux_seedvr2()
                return ScaleFactor.parse(text)
            try:
                return int(text)
            except Exception as exc:
                raise ValueError(
                    "image_upscale resolution must be an integer shortest-edge target "
                    "or a scale factor such as '2x'."
                ) from exc
        if resolution is None:
            _ModelConfig, _SeedVR2, ScaleFactor, _TilingConfig = _lazy_import_mflux_seedvr2()
            return ScaleFactor.parse("2x")
        try:
            target = int(resolution)
        except Exception as exc:
            raise ValueError(
                "image_upscale resolution must be an integer shortest-edge target "
                "or a scale factor such as '2x'."
            ) from exc
        if target <= 0:
            raise ValueError("image_upscale resolution must be positive.")
        return target

    def _sniff_image_suffix(self, image: bytes) -> str:
        if len(image) >= 8 and image[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if len(image) >= 3 and image[:3] == b"\xff\xd8\xff":
            return ".jpg"
        return ".img"

    def _subscribe_progress(
        self,
        model: Any,
        callback: Callable[[Any], None],
        *,
        task: Optional[str],
    ) -> Callable[[], None]:
        callbacks = getattr(model, "callbacks", None)
        subscribe = getattr(callbacks, "subscribe_progress", None)
        if not callable(subscribe):
            return lambda: None
        unsubscribe = subscribe(callback, task=task)
        if callable(unsubscribe):
            return unsubscribe
        return lambda: None

    def _write_temp_image_bytes(self, data: bytes) -> Path:
        suffix = self._sniff_image_suffix(data)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as fp:
            path = Path(fp.name)
            fp.write(data)
        return path

    def _append_reference_image_value(
        self,
        image_paths: List[str],
        temp_paths: List[Path],
        value: Any,
    ) -> None:
        if value is None:
            return
        if isinstance(value, (bytes, bytearray, memoryview)):
            path = self._write_temp_image_bytes(bytes(value))
            temp_paths.append(path)
            image_paths.append(str(path))
            return
        if isinstance(value, Path):
            image_paths.append(str(value.expanduser()))
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                image_paths.append(str(Path(text).expanduser()))
            return
        data = getattr(value, "data", None)
        if isinstance(data, (bytes, bytearray, memoryview)):
            path = self._write_temp_image_bytes(bytes(data))
            temp_paths.append(path)
            image_paths.append(str(path))

    def _extend_reference_image_paths(
        self,
        extra: Dict[str, Any],
        image_paths: List[str],
        temp_paths: List[Path],
    ) -> None:
        for key in ("reference_images", "images", "additional_images", "image_paths"):
            values = extra.pop(key, None)
            if values is None:
                continue
            if isinstance(values, (str, bytes, bytearray, memoryview, Path)):
                iterable = [values]
            else:
                try:
                    iterable = list(values)
                except TypeError:
                    iterable = [values]
            for value in iterable:
                self._append_reference_image_value(image_paths, temp_paths, value)

    def _prepare_i2v_conditioning_image(
        self,
        image_path: Path,
        *,
        width: int,
        height: int,
    ) -> Tuple[Path, Optional[Dict[str, Any]]]:
        if width <= 0 or height <= 0:
            return image_path, None
        try:
            from PIL import Image, ImageOps
        except Exception as exc:  # pragma: no cover - Pillow is a package dependency.
            raise OptionalDependencyMissingError(
                "MLX-Gen image-to-video conditioning requires Pillow to preserve the source "
                "image aspect ratio. Install or upgrade with `pip install \"abstractvision[mlx-gen]\"`."
            ) from exc

        with Image.open(image_path) as source:
            source = ImageOps.exif_transpose(source)
            source_width, source_height = source.size
            if source_width == width and source_height == height:
                return image_path, {
                    "mode": "passthrough",
                    "source_width": source_width,
                    "source_height": source_height,
                    "conditioning_width": width,
                    "conditioning_height": height,
                }

            if source.mode in {"RGBA", "LA"} or (
                source.mode == "P" and "transparency" in source.info
            ):
                rgba = source.convert("RGBA")
                opaque = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
                opaque.alpha_composite(rgba)
                source = opaque.convert("RGB")
            else:
                source = source.convert("RGB")

            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            fitted = ImageOps.contain(source, (width, height), method=resampling)
            pad_left = (width - fitted.width) // 2
            pad_top = (height - fitted.height) // 2
            canvas = Image.new("RGB", (width, height), (0, 0, 0))
            canvas.paste(fitted, (pad_left, pad_top))

            with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as fp:
                prepared_path = Path(fp.name)
                canvas.save(fp, format="PNG")

        return prepared_path, {
            "mode": "letterbox",
            "source_width": source_width,
            "source_height": source_height,
            "conditioning_width": width,
            "conditioning_height": height,
            "fit_width": fitted.width,
            "fit_height": fitted.height,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "pad_right": width - fitted.width - pad_left,
            "pad_bottom": height - fitted.height - pad_top,
            "background": "black",
        }

    def _sniff_image_dimensions(self, image: bytes) -> Optional[Tuple[int, int]]:
        if len(image) >= 24 and image[:8] == b"\x89PNG\r\n\x1a\n" and image[12:16] == b"IHDR":
            try:
                width = int.from_bytes(image[16:20], "big")
                height = int.from_bytes(image[20:24], "big")
                if width > 0 and height > 0:
                    return width, height
            except Exception:
                return None
        if len(image) >= 4 and image[:2] == b"\xff\xd8":
            i = 2
            n = len(image)
            while i + 1 < n:
                if image[i] != 0xFF:
                    i += 1
                    continue
                while i < n and image[i] == 0xFF:
                    i += 1
                if i >= n:
                    break
                marker = image[i]
                i += 1
                if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7 or marker == 0x01:
                    continue
                if i + 1 >= n:
                    break
                seg_len = int.from_bytes(image[i : i + 2], "big")
                if seg_len < 2 or i + seg_len > n:
                    break
                if marker in {
                    0xC0,
                    0xC1,
                    0xC2,
                    0xC3,
                    0xC5,
                    0xC6,
                    0xC7,
                    0xC9,
                    0xCA,
                    0xCB,
                    0xCD,
                    0xCE,
                    0xCF,
                }:
                    try:
                        height = int.from_bytes(image[i + 3 : i + 5], "big")
                        width = int.from_bytes(image[i + 5 : i + 7], "big")
                        if width > 0 and height > 0:
                            return width, height
                    except Exception:
                        return None
                i += seg_len
                if marker == 0xDA:
                    break
        return None

    def _generate_impl(self, request: ImageGenerationRequest) -> GeneratedAsset:
        request = self._normalize_once(request, self._normalize_image_generation, task="text_to_image")
        memory_plan = self._memory_plan(request.memory_mode)
        with self._memory_session(memory_plan):
            return self._generate_run(request, memory_plan)

    def _generate_run(
        self,
        request: ImageGenerationRequest,
        memory_plan: _MemoryPlan,
        *,
        image_path: Optional[Path] = None,
        image_strength: Optional[float] = None,
    ) -> GeneratedAsset:
        # Free for the request `_generate_impl` normalized; an edit's derived request is normalized here.
        task = "image_to_image" if image_path is not None else "text_to_image"
        request = self._normalize_once(request, self._normalize_image_generation, task=task)
        effective_loras = self._effective_request_lora_adapters(request)
        model, model_def = self._ensure_model_impl(
            lora_adapters=effective_loras,
            stream_blocks=memory_plan.stream_blocks,
            quantize=request.quantize,
        )
        extra = dict(request.extra or {})
        seed = int(request.seed) if request.seed is not None else random.randint(0, 1_000_000_000)
        steps = int(request.steps) if request.steps is not None else model_def.default_steps
        width = int(request.width) if request.width is not None else int(self._cfg.default_width)
        height = (
            int(request.height) if request.height is not None else int(self._cfg.default_height)
        )
        guidance = (
            float(request.guidance_scale)
            if request.guidance_scale is not None
            else model_def.default_guidance
        )

        kwargs: Dict[str, Any] = {
            "seed": seed,
            "prompt": str(request.prompt),
            "num_inference_steps": steps,
            "height": height,
            "width": width,
        }
        if guidance is not None:
            kwargs["guidance"] = guidance
        scheduler = extra.pop("scheduler", None)
        if scheduler is not None:
            kwargs["scheduler"] = str(scheduler)
        if request.negative_prompt and model_def.supports_negative_prompt:
            kwargs["negative_prompt"] = str(request.negative_prompt)
        if image_path is not None:
            kwargs["image_path"] = image_path
            kwargs["image_strength"] = float(
                image_strength if image_strength is not None else extra.pop("image_strength", 0.4)
            )
        progress_task = "image-to-image" if image_path is not None else "text-to-image"
        default_task = "image_to_image" if image_path is not None else "text_to_image"
        relay = _ProgressRelay(extra, task=default_task, cancel_event=request.cancel_event)

        if memory_plan.low_ram:
            self._attach_low_ram_saver(model)
        unsubscribe = (
            self._subscribe_progress(model, relay, task=progress_task)
            if relay.active
            else (lambda: None)
        )

        try:
            try:
                relay.check_cancelled()
                generated = model.generate_image(**kwargs)
            except Exception as e:
                cancelled = relay.cancelled_error(e)
                if cancelled is not None:
                    raise cancelled from e
                if _is_mlx_gen_download_required(e):
                    raise _wrap_mlx_gen_download_required(e) from e
                raise
            if self._model_key is not None:
                self._warmed_model_key = self._model_key
            pil_image = getattr(generated, "image", generated)
            mlx_metadata = self._generated_image_metadata(generated)
            buf = BytesIO()
            pil_image.save(buf, format="PNG")
            data = buf.getvalue()
            image_strength_used = kwargs.get("image_strength") if image_path is not None else None
            requested_loras = _public_lora_adapters(effective_loras)
            return GeneratedAsset(
                media_type="image",
                data=data,
                mime_type="image/png",
                metadata={
                    "source": MLX_GEN_RUNTIME,
                    "engine": MLX_GEN_RUNTIME,
                    "legacy_engine": MFLUX_PROVIDER,
                    "runtime_package": MLX_GEN_RUNTIME,
                    "model": _public_model_name(self._resolved_model_path),
                    "base_model": self._resolved_base_model,
                    "quantization_bits": self._quantization_bits(),
                    "seed": seed,
                    "steps": steps,
                    "width": width,
                    "height": height,
                    **({"requested_lora_adapters": requested_loras} if requested_loras else {}),
                    **(_extract_lora_metadata(mlx_metadata) if mlx_metadata else {}),
                    **({"mlx_gen": mlx_metadata} if mlx_metadata else {}),
                    **(
                        {"image_strength": image_strength_used}
                        if image_strength_used is not None
                        else {}
                    ),
                    **memory_plan.metadata(),
                },
            )
        finally:
            unsubscribe()

    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
        return self._run_on_runtime_thread(self._generate_impl, request)

    def generate_image_with_progress(
        self,
        request: ImageGenerationRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        if progress_callback is None:
            return self.generate_image(request)
        return self.generate_image(
            self._run_on_runtime_thread(
                self._normalized_with_step_progress, request, progress_callback, self._normalize_image_generation, task="text_to_image"
            )
        )

    def _edit_image_impl(self, request: ImageEditRequest) -> GeneratedAsset:
        request = self._normalize_once(request, self._normalize_image_edit, task="image_to_image")
        memory_plan = self._memory_plan(request.memory_mode)
        with self._memory_session(memory_plan):
            return self._edit_image_run(request, memory_plan)

    def _edit_image_run(self, request: ImageEditRequest, memory_plan: _MemoryPlan) -> GeneratedAsset:
        effective_loras = self._effective_request_lora_adapters(request)

        _model, model_def = self._ensure_model_impl(
            edit_variant=True,
            lora_adapters=effective_loras,
            stream_blocks=memory_plan.stream_blocks,
            quantize=request.quantize,
        )
        if request.mask is not None:
            self._require_route_option_support_if_known(
                "image_to_image",
                route_field="supports_mask",
                field="mask",
                error_message=(
                    "MLX-Gen masked image edits are not available for the selected model route. "
                    "Inspect `abstractvision show-model <model-id>`; current validated routes are "
                    "`AbstractFramework/qwen-image-edit-2511-8bit`, `briaai/Fibo-Edit`, and "
                    "`briaai/Fibo-Edit-RMBG`."
                ),
            )
            if model_def.family not in {"qwen-edit", "fibo-edit"}:
                raise CapabilityNotSupportedError(
                    "MLX-Gen masked image edits are currently implemented for Qwen Image Edit "
                    f"and FIBO Edit routes, not {model_def.family!r}.",
                    field="mask",
                )
        if model_def.family not in {"flux2", "qwen-edit", "ernie-image", "fibo", "fibo-edit"}:
            raise CapabilityNotSupportedError(
                "MLX-Gen image_to_image is implemented for FLUX.2, Qwen Image Edit, "
                f"ERNIE Image Turbo, and FIBO models today (got {model_def.family!r})."
            )

        extra = dict(request.extra or {})
        width = extra.get("width")
        height = extra.get("height")
        if width is None or height is None:
            sniffed = self._sniff_image_dimensions(request.image)
            if sniffed is not None:
                width, height = sniffed
        strength = extra.get("image_strength")
        if strength is None:
            strength = extra.get("strength")
        image_strength: Optional[float] = None
        if strength is not None:
            try:
                image_strength = float(strength)
            except Exception:
                image_strength = None

        gen_request = ImageGenerationRequest(
            prompt=str(request.prompt),
            negative_prompt=str(request.negative_prompt) if request.negative_prompt else None,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
            seed=int(request.seed) if request.seed is not None else None,
            steps=int(request.steps) if request.steps is not None else None,
            guidance_scale=(
                float(request.guidance_scale) if request.guidance_scale is not None else None
            ),
            lora_adapters=tuple(effective_loras),
            extra=extra,
            memory_mode=memory_plan.mode,
            quantize=request.quantize,
            cancel_event=request.cancel_event,
        )

        suffix = self._sniff_image_suffix(request.image)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as fp:
            tmp_path = Path(fp.name)
            fp.write(request.image)
        tmp_mask_path: Optional[Path] = None
        if request.mask is not None:
            mask_suffix = self._sniff_image_suffix(request.mask)
            with tempfile.NamedTemporaryFile(mode="wb", suffix=mask_suffix, delete=False) as fp:
                tmp_mask_path = Path(fp.name)
                fp.write(request.mask)
        tmp_extra_paths: List[Path] = []
        try:
            if model_def.family in {"flux2", "qwen-edit", "fibo-edit"}:
                seed = (
                    int(request.seed)
                    if request.seed is not None
                    else random.randint(0, 1_000_000_000)
                )
                steps = int(request.steps) if request.steps is not None else model_def.default_steps
                guidance = (
                    float(request.guidance_scale)
                    if request.guidance_scale is not None
                    else model_def.default_guidance
                )
                relay = _ProgressRelay(
                    extra, task="image_to_image", cancel_event=request.cancel_event
                )
                image_paths = [str(tmp_path)]
                self._extend_reference_image_paths(extra, image_paths, tmp_extra_paths)
                if len(image_paths) > 1 and model_def.family not in {"flux2", "qwen-edit"}:
                    raise CapabilityNotSupportedError(
                        "MLX-Gen multi-reference image edits are supported for FLUX.2 and "
                        f"Qwen Image Edit models, not {model_def.family!r}."
                    )
                if tmp_mask_path is not None and len(image_paths) != 1:
                    raise CapabilityNotSupportedError(
                        "MLX-Gen masked Qwen edits require exactly one source image."
                    )
                kwargs: Dict[str, Any] = {
                    "seed": seed,
                    "prompt": str(request.prompt),
                    "num_inference_steps": steps,
                }
                if model_def.family in {"flux2", "qwen-edit"}:
                    kwargs["image_paths"] = image_paths
                    if model_def.family == "qwen-edit" and tmp_mask_path is not None:
                        kwargs["mask_path"] = str(tmp_mask_path)
                else:
                    kwargs["image_path"] = str(tmp_path)
                    if tmp_mask_path is not None:
                        kwargs["mask_path"] = str(tmp_mask_path)
                if width is not None:
                    kwargs["width"] = int(width)
                if height is not None:
                    kwargs["height"] = int(height)
                if guidance is not None:
                    kwargs["guidance"] = guidance
                scheduler = extra.pop("scheduler", None)
                if scheduler is not None:
                    kwargs["scheduler"] = str(scheduler)
                if request.negative_prompt and model_def.supports_negative_prompt:
                    kwargs["negative_prompt"] = str(request.negative_prompt)
                task_name = "image-to-image"

                if memory_plan.low_ram:
                    self._attach_low_ram_saver(_model)
                unsubscribe: Callable[[], None] = lambda: None  # noqa: E731
                try:
                    if relay.active:
                        unsubscribe = self._subscribe_progress(_model, relay, task=task_name)
                    relay.check_cancelled()
                    generated = _model.generate_image(**kwargs)
                except Exception as e:
                    cancelled = relay.cancelled_error(e)
                    if cancelled is not None:
                        raise cancelled from e
                    if _is_mlx_gen_download_required(e):
                        raise _wrap_mlx_gen_download_required(e) from e
                    raise
                finally:
                    unsubscribe()
                pil_image = getattr(generated, "image", generated)
                mlx_metadata = self._generated_image_metadata(generated)
                buf = BytesIO()
                pil_image.save(buf, format="PNG")
                requested_loras = _public_lora_adapters(effective_loras)
                return GeneratedAsset(
                    media_type="image",
                    data=buf.getvalue(),
                    mime_type="image/png",
                    metadata={
                        "source": MLX_GEN_RUNTIME,
                        "engine": MLX_GEN_RUNTIME,
                        "legacy_engine": MFLUX_PROVIDER,
                        "runtime_package": MLX_GEN_RUNTIME,
                        "model": _public_model_name(self._resolved_model_path),
                        "base_model": self._resolved_base_model,
                        "quantization_bits": self._quantization_bits(),
                        "seed": seed,
                        "steps": steps,
                        "reference_image_count": len(image_paths),
                        "edit_mode": "multi_reference" if len(image_paths) > 1 else "edit_reference",
                        **({"requested_lora_adapters": requested_loras} if requested_loras else {}),
                        **(_extract_lora_metadata(mlx_metadata) if mlx_metadata else {}),
                        **({"mlx_gen": mlx_metadata} if mlx_metadata else {}),
                        **({"width": int(width)} if width is not None else {}),
                        **({"height": int(height)} if height is not None else {}),
                        **({"mask": True} if tmp_mask_path is not None else {}),
                        **memory_plan.metadata(),
                    },
                )
            return self._generate_run(
                gen_request,
                memory_plan,
                image_path=tmp_path,
                image_strength=image_strength,
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if tmp_mask_path is not None:
                try:
                    tmp_mask_path.unlink(missing_ok=True)
                except Exception:
                    pass
            for path in tmp_extra_paths:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
        return self._run_on_runtime_thread(self._edit_image_impl, request)

    def edit_image_with_progress(
        self,
        request: ImageEditRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        if progress_callback is None:
            return self.edit_image(request)
        return self.edit_image(
            self._run_on_runtime_thread(
                self._normalized_with_step_progress, request, progress_callback, self._normalize_image_edit, task="image_to_image"
            )
        )

    def _upscale_image_impl(self, request: ImageUpscaleRequest) -> GeneratedAsset:
        request = self._normalize_once(request, self._normalize_image_upscale, task="image_upscale")
        memory_plan = self._memory_plan(request.memory_mode)
        # `mlxgen upscale --low-ram` on images sets the 1 GB MLX cache and nothing else: it
        # registers no MemorySaver and keeps SeedVR2's own tiling. The instance stays loaded.
        with self._memory_session(memory_plan):
            return self._upscale_image_run(request, memory_plan)

    def _upscale_image_run(
        self, request: ImageUpscaleRequest, memory_plan: _MemoryPlan
    ) -> GeneratedAsset:
        model, _model_def = self._ensure_seedvr2_model_impl(quantize=request.quantize)
        _ModelConfig, _SeedVR2, _ScaleFactor, TilingConfig = _lazy_import_mflux_seedvr2()
        try:
            if request.vae_tiling is True:
                setattr(model, "tiling_config", TilingConfig())
            elif request.vae_tiling is False:
                setattr(model, "tiling_config", None)
        except Exception:
            pass

        extra = dict(request.extra or {})
        relay = _ProgressRelay(
            extra, task="image_upscale", cancel_event=request.cancel_event, force_task=True
        )
        seed = int(request.seed) if request.seed is not None else random.randint(0, 1_000_000_000)
        resolution_arg = self._coerce_seedvr2_resolution(request.resolution)
        softness = 0.25 if request.softness is None else float(request.softness)

        tmp_path = self._write_temp_image_bytes(request.image)
        unsubscribe = (
            self._subscribe_progress(model, relay, task=None) if relay.active else (lambda: None)
        )
        try:
            relay.check_cancelled()
            generated = model.generate_image(
                seed=seed,
                image_path=tmp_path,
                resolution=resolution_arg,
                softness=softness,
            )
        except Exception as e:
            cancelled = relay.cancelled_error(e)
            if cancelled is not None:
                raise cancelled from e
            if _is_mlx_gen_download_required(e):
                raise _wrap_mlx_gen_download_required(e) from e
            raise
        finally:
            unsubscribe()
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        if self._model_key is not None:
            self._warmed_model_key = self._model_key
        pil_image = getattr(generated, "image", generated)
        mlx_metadata = self._generated_image_metadata(generated)
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        data = buf.getvalue()
        resolution_meta: Any = (
            str(resolution_arg) if not isinstance(resolution_arg, int) else resolution_arg
        )
        return GeneratedAsset(
            media_type="image",
            data=data,
            mime_type="image/png",
            metadata={
                "source": MLX_GEN_RUNTIME,
                "engine": MLX_GEN_RUNTIME,
                "legacy_engine": MFLUX_PROVIDER,
                "runtime_package": MLX_GEN_RUNTIME,
                "task": "image_upscale",
                "model": _public_model_name(self._resolved_model_path),
                "base_model": self._resolved_base_model,
                "quantization_bits": self._quantization_bits(),
                "seed": seed,
                "steps": 1,
                "resolution": resolution_meta,
                **({"scale": request.scale} if request.scale is not None else {}),
                "softness": softness,
                "vae_tiling": request.vae_tiling,
                "vae_tiling_policy": "explicit" if request.vae_tiling is not None else "runtime_default",
                **({"mlx_gen": mlx_metadata} if mlx_metadata else {}),
                **memory_plan.metadata(),
            },
        )

    def upscale_image(self, request: ImageUpscaleRequest) -> GeneratedAsset:
        return self._run_on_runtime_thread(self._upscale_image_impl, request)

    def upscale_image_with_progress(
        self,
        request: ImageUpscaleRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        if progress_callback is None:
            return self.upscale_image(request)
        return self.upscale_image(
            self._run_on_runtime_thread(
                self._normalized_with_step_progress, request, progress_callback, self._normalize_image_upscale, task="image_upscale"
            )
        )

    def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]:
        raise CapabilityNotSupportedError(
            "MLX-Gen backend does not implement multi-view generation."
        )

    def _save_generated_video(self, generated: Any) -> "_SavedVideo":
        """Write the engine's video (and soundtrack) through its own save, return bytes and metadata.

        `save(..., export_json_metadata=True)` writes MLX-Gen's metadata sidecar, which carries what
        only the save knows (audio mux facts, video health, the process's peak memory); it is the
        engine metadata returned when present, else `_get_metadata()`. A soundtrack MLX-Gen could
        not mux is left next to the clip as a WAV and returned as bytes.
        """
        save = getattr(generated, "save", None)
        if not callable(save):
            raise RuntimeError("MLX-Gen video result does not expose a callable save(...) method.")
        with tempfile.TemporaryDirectory(prefix="abstractvision-video-") as tmp_dir:
            tmp_path = Path(tmp_dir) / "video.mp4"
            saved_path = save(path=tmp_path, export_json_metadata=True, overwrite=True)
            video_path = Path(saved_path) if isinstance(saved_path, (str, Path)) else tmp_path
            if not video_path.is_file():
                video_path = tmp_path
            data = video_path.read_bytes()
            engine_metadata: Dict[str, Any] = {}
            for sidecar in sorted(Path(tmp_dir).glob("*.json")):
                try:
                    import json

                    loaded = json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(loaded, dict):
                    engine_metadata = loaded
                    break
            if engine_metadata:
                engine_metadata = _public_engine_metadata(engine_metadata)
            else:
                engine_metadata = self._generated_video_metadata(generated)
            wav_files = sorted(Path(tmp_dir).glob("*.wav"))
            audio_wav = wav_files[0].read_bytes() if wav_files else None
        return _SavedVideo(data=data, engine_metadata=engine_metadata, audio_wav=audio_wav)

    def _generated_video_metadata(self, generated: Any) -> Dict[str, Any]:
        get_metadata = getattr(generated, "_get_metadata", None)
        if not callable(get_metadata):
            return {}
        try:
            metadata = get_metadata()
        except Exception:
            return {}
        return _public_engine_metadata(metadata) if isinstance(metadata, dict) else {}

    def _generate_video_impl(
        self,
        request: Union[VideoGenerationRequest, ImageToVideoRequest],
        *,
        image_path: Optional[Path] = None,
        conditioning_image_metadata: Optional[Dict[str, Any]] = None,
    ) -> GeneratedAsset:
        task = "text_to_video" if isinstance(request, VideoGenerationRequest) else "image_to_video"
        # Free for an image-to-video request, which `_image_to_video_impl` normalized.
        request = self._normalize_once(request, self._normalize_video, task=task)
        memory_plan = self._memory_plan(request.memory_mode)
        # In low-RAM mode the 1 GB cache limit holds from before the model loads through the decode.
        with self._memory_session(memory_plan):
            return self._generate_video_run(
                request,
                memory_plan,
                task=task,
                image_path=image_path,
                conditioning_image_metadata=conditioning_image_metadata,
            )

    def _generate_video_run(
        self,
        request: Union[VideoGenerationRequest, ImageToVideoRequest],
        memory_plan: _MemoryPlan,
        *,
        task: str,
        image_path: Optional[Path] = None,
        conditioning_image_metadata: Optional[Dict[str, Any]] = None,
    ) -> GeneratedAsset:
        if self._resolved_model_def().family == "minimax-h3":
            return self._generate_h3_run(request, memory_plan, task=task, image_path=image_path)
        effective_loras = self._effective_request_lora_adapters(request)
        try:
            model, model_def = self._ensure_model_impl(
                lora_adapters=effective_loras,
                stream_blocks=memory_plan.stream_blocks,
                quantize=request.quantize,
            )
        except MemoryError as e:
            raise self._out_of_memory_error(e, memory_plan, task=task) from e
        if model_def.family != "wan-video":
            raise CapabilityNotSupportedError(
                f"MLX-Gen {task} is only implemented for Wan video models today (got {model_def.family!r})."
            )

        extra = dict(request.extra or {})
        seed = int(request.seed) if request.seed is not None else random.randint(0, 1_000_000_000)
        steps = int(request.steps) if request.steps is not None else int(model_def.default_steps)
        width = int(request.width) if request.width is not None else int(model_def.default_width)
        height = int(request.height) if request.height is not None else int(model_def.default_height)
        fps = int(request.fps) if request.fps is not None else int(model_def.default_fps)
        num_frames = (
            int(request.num_frames) if request.num_frames is not None else int(model_def.default_frames)
        )
        guidance = (
            float(request.guidance_scale)
            if request.guidance_scale is not None
            else model_def.default_guidance
        )
        max_sequence_length = extra.pop("max_sequence_length", None)
        vae_tiling = _wan_vae_tiling_requested(extra.pop("vae_tiling", None))
        guidance_2 = (
            request.guidance_2
            if request.guidance_2 is not None
            else model_def.default_guidance_2
        )
        flow_shift = request.flow_shift
        relay = _ProgressRelay(extra, task=task, cancel_event=request.cancel_event)

        kwargs: Dict[str, Any] = {
            "seed": seed,
            "prompt": str(request.prompt or ""),
            "num_inference_steps": steps,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "fps": fps,
            "guidance": guidance,
            "negative_prompt": str(request.negative_prompt or ""),
        }
        if image_path is not None:
            kwargs["image_path"] = image_path
        if max_sequence_length is not None:
            kwargs["max_sequence_length"] = int(max_sequence_length)
        if guidance_2 is not None:
            kwargs["guidance_2"] = float(guidance_2)
        if flow_shift is not None:
            kwargs["flow_shift"] = float(flow_shift)
        if relay.active:
            kwargs["progress_callback"] = relay
        # TI2V-5B's single denoiser (9.3 GiB at BF16 runtime) is dead weight while
        # the VAE decodes, and the decode is the run's peak. Releasing it before
        # decode lowered the measured MLX peak from 60.8 to 50.6 GiB at
        # 1280x704x121 (the plan frees it in every mode; the instance is then
        # rebuilt on the next request, about a minute against a run of several).
        # Low-RAM mode is `mlxgen generate --low-ram` on a Wan route for one seed (wan_generate.py):
        # the MLX cache is cleared after every step and transformer block, and the run releases
        # the denoisers before the decode and, on the dual-expert A14B models, the inactive expert.
        if memory_plan.releases_model:
            kwargs["release_denoisers_before_decode"] = True
            if memory_plan.low_ram and model_def.key in {WAN_T2V_A14B_MODEL_KEY, WAN_I2V_A14B_MODEL_KEY}:
                kwargs["release_inactive_denoiser"] = True
        if memory_plan.low_ram:
            kwargs["clear_cache_each_step"] = True
            kwargs["clear_cache_each_transformer_block"] = True
        # The VAE decode is the run's memory peak; decoding in overlapping
        # spatial tiles bounds it by the tile instead of the frame (see
        # wan_vae_tiling). extra={"vae_tiling": False} decodes untiled.
        restore_decode = _install_wan_tiled_decode(model) if vae_tiling else None
        try:
            try:
                relay.check_cancelled()
                generated = model.generate_video(**kwargs)
            except MemoryError as e:
                raise self._out_of_memory_error(e, memory_plan, task=task) from e
            except Exception as e:
                cancelled = relay.cancelled_error(e)
                if cancelled is not None:
                    raise cancelled from e
                if _is_mlx_gen_download_required(e):
                    raise _wrap_mlx_gen_download_required(e) from e
                raise
            # mlx-gen streams the VAE decode into save(), so the decode peak
            # happens here, inside the cache ceiling.
            saved = self._save_generated_video(generated)
            data = saved.data
            mlx_metadata = saved.engine_metadata
        finally:
            if restore_decode is not None:
                restore_decode()
            _clear_mlx_cache()
        if self._model_key is not None:
            self._warmed_model_key = self._model_key
        requested_loras = _public_lora_adapters(effective_loras)
        return GeneratedAsset(
            media_type="video",
            data=data,
            mime_type="video/mp4",
            metadata={
                "source": MLX_GEN_RUNTIME,
                "engine": MLX_GEN_RUNTIME,
                "legacy_engine": MFLUX_PROVIDER,
                "runtime_package": MLX_GEN_RUNTIME,
                "task": task,
                "model": _public_model_name(self._resolved_model_path),
                "base_model": self._resolved_base_model,
                "quantization_bits": self._quantization_bits(),
                "seed": seed,
                "steps": steps,
                "width": width,
                "height": height,
                "fps": fps,
                "num_frames": num_frames,
                "guidance_scale": guidance,
                **({"requested_lora_adapters": requested_loras} if requested_loras else {}),
                **(_extract_lora_metadata(mlx_metadata) if mlx_metadata else {}),
                **({"guidance_2": float(guidance_2)} if guidance_2 is not None else {}),
                **({"flow_shift": float(flow_shift)} if flow_shift is not None else {}),
                "vae_tiling": bool(vae_tiling),
                **(
                    {"conditioning_image": conditioning_image_metadata}
                    if conditioning_image_metadata
                    else {}
                ),
                **({"mlx_gen": mlx_metadata} if mlx_metadata else {}),
                **memory_plan.metadata(),
            },
        )

    def _generate_h3_run(
        self,
        request: Union[VideoGenerationRequest, ImageToVideoRequest],
        memory_plan: _MemoryPlan,
        *,
        task: str,
        image_path: Optional[Path] = None,
    ) -> GeneratedAsset:
        """One MiniMax-H3 clip with its soundtrack, as `mlxgen generate` runs it.

        The prompt is passed unchanged: MLX-Gen composes the structured prompt (and, for a
        keyframe, its picture reference) from `prompt`, `soundscape` and `music`. Low-RAM mode is
        the 1 GB cache (held by the session) plus `release_transformer=True`; the session then
        drops the instance, as for every other family.
        """
        effective_loras = self._effective_request_lora_adapters(request)
        row = self._h3_route_row(task)
        roles = {str(role) for role in (row.get("lora_target_roles") or ())}
        wrong_roles = sorted(
            {str(spec.target_role) for spec in effective_loras if spec.target_role}
            - roles
        )
        if effective_loras and not row.get("supports_lora"):
            raise CapabilityNotSupportedError(
                f"The MiniMax-H3 {task} route takes no LoRA adapters.", field="lora_adapters", supported=()
            )
        if wrong_roles:
            raise CapabilityNotSupportedError(
                f"MiniMax-H3 adapters apply to {', '.join(sorted(roles)) or 'no role'}, "
                f"not {', '.join(wrong_roles)}.",
                field="lora_adapters",
            )
        if memory_plan.stream_blocks:
            self._refuse_h3_adapters_that_keep_the_transformer(effective_loras)
        try:
            model, _model_def = self._ensure_model_impl(
                lora_adapters=effective_loras,
                stream_blocks=memory_plan.stream_blocks,
                quantize=request.quantize,
            )
        except MemoryError as e:
            raise self._out_of_memory_error(e, memory_plan, task=task) from e
        extra = dict(request.extra or {})
        relay = _ProgressRelay(extra, task=task, cancel_event=request.cancel_event)
        seed = int(request.seed) if request.seed is not None else random.randint(0, 1_000_000_000)
        kwargs: Dict[str, Any] = {
            "seed": seed,
            "prompt": str(request.prompt or ""),
            "generate_audio": True if request.generate_audio is None else bool(request.generate_audio),
            "release_transformer": memory_plan.releases_model,
        }
        for name, value in (
            ("width", request.width),
            ("height", request.height),
            ("num_frames", request.num_frames),
            ("num_inference_steps", request.steps),
            ("video_shift", request.flow_shift),
            ("audio_shift", request.audio_shift),
            ("soundscape", request.soundscape),
            ("music", request.music),
        ):
            if value is not None:
                kwargs[name] = value
        if image_path is not None:
            kwargs["image_path"] = str(image_path)
        if relay.active:
            kwargs["progress_callback"] = relay
        try:
            try:
                relay.check_cancelled()
                generated = model.generate_video(**kwargs)
            except MemoryError as e:
                raise self._out_of_memory_error(e, memory_plan, task=task) from e
            except Exception as e:
                cancelled = relay.cancelled_error(e)
                if cancelled is not None:
                    raise cancelled from e
                if _is_mlx_gen_download_required(e):
                    raise _wrap_mlx_gen_download_required(e) from e
                raise
            saved = self._save_generated_video(generated)
        finally:
            # As on the Wan path: whatever the outcome, the run's freed buffers leave the cache.
            _clear_mlx_cache()
        if self._model_key is not None:
            self._warmed_model_key = self._model_key
        engine = saved.engine_metadata
        audio_track = _h3_audio_track(engine, saved.audio_wav)
        requested_loras = _public_lora_adapters(effective_loras)
        return GeneratedAsset(
            media_type="video",
            data=saved.data,
            mime_type="video/mp4",
            audio=audio_track,
            metadata={
                "source": MLX_GEN_RUNTIME,
                "engine": MLX_GEN_RUNTIME,
                "legacy_engine": MFLUX_PROVIDER,
                "runtime_package": MLX_GEN_RUNTIME,
                "task": task,
                "model": _public_model_name(self._resolved_model_path),
                "base_model": self._resolved_base_model,
                "quantization_bits": self._quantization_bits(),
                "seed": seed,
                "steps": engine.get("steps"),
                "width": engine.get("width"),
                "height": engine.get("height"),
                "fps": engine.get("fps"),
                "num_frames": engine.get("frames"),
                "requested_frames": engine.get("requested_frames", request.num_frames),
                "duration_seconds": engine.get("duration_seconds"),
                "video_shift": engine.get("video_shift"),
                "audio_shift": engine.get("audio_shift"),
                "prompt": engine.get("prompt"),
                **({"requested_lora_adapters": requested_loras} if requested_loras else {}),
                **(_extract_lora_metadata(engine) if engine else {}),
                **({"mlx_gen": engine} if engine else {}),
                **memory_plan.metadata(),
            },
        )

    def _refuse_h3_adapters_that_keep_the_transformer(self, adapters: Sequence[Any]) -> None:
        """memory_mode='lowest' streams the blocks; an adapter on their AdaLN projections cannot."""
        paths = self._mlx_gen_lora_init_kwargs(tuple(adapters)).get("lora_paths") or ()
        local = [str(path) for path in paths if Path(str(path)).is_file()]
        if not local:
            return
        planning = _lazy_import_mlx_gen_h3_planning()
        for path in local:
            if planning.adapter_targets_modulation(path):
                raise CapabilityNotSupportedError(
                    f"memory_mode='lowest' streams the transformer blocks from disk, but the adapter "
                    f"{_public_model_name(path)} trains their AdaLN projections, which MLX-Gen must "
                    "then hold in memory with the whole transformer. Run it with memory_mode='low' "
                    "or 'default', or without that adapter."
                )

    def _out_of_memory_error(
        self, error: MemoryError, memory_plan: _MemoryPlan, *, task: str
    ) -> GpuOutOfMemoryError:
        modes = tuple(self.supported_memory_modes(task))
        lower = [mode for mode in modes[modes.index(memory_plan.mode) + 1 :]] if memory_plan.mode in modes else []
        advice = (
            f"memory_mode={' or '.join(repr(mode) for mode in lower)} lowers the peak, as does"
            if lower
            else "This is the lowest memory mode; what lowers the peak is"
        )
        return GpuOutOfMemoryError(
            f"MLX-Gen cannot fit this {task} run in memory with memory_mode={memory_plan.mode!r} ({error}). "
            f"{advice} a shorter clip or a smaller canvas."
        )

    def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
        return self._run_on_runtime_thread(self._generate_video_impl, request)

    def generate_video_with_progress(
        self,
        request: VideoGenerationRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        if progress_callback is None:
            return self.generate_video(request)
        return self.generate_video(
            self._run_on_runtime_thread(
                self._normalized_with_step_progress, request, progress_callback, self._normalize_video, task="text_to_video"
            )
        )

    def _image_to_video_impl(self, request: ImageToVideoRequest) -> GeneratedAsset:
        request = self._normalize_once(request, self._normalize_video, task="image_to_video")
        suffix = self._sniff_image_suffix(request.image)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as fp:
            tmp_path = Path(fp.name)
            fp.write(request.image)
        if self._resolved_model_def().family == "minimax-h3":
            # MiniMax-H3 maps the keyframe onto its own canvas (or the requested one) itself.
            try:
                return self._generate_video_impl(request, image_path=tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)
        conditioning_path = tmp_path
        conditioning_metadata: Optional[Dict[str, Any]] = None
        try:
            conditioning_path, conditioning_metadata = self._prepare_i2v_conditioning_image(
                tmp_path,
                width=int(request.width),
                height=int(request.height),
            )
            return self._generate_video_impl(
                request,
                image_path=conditioning_path,
                conditioning_image_metadata=conditioning_metadata,
            )
        finally:
            if conditioning_path != tmp_path:
                try:
                    conditioning_path.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
        return self._run_on_runtime_thread(self._image_to_video_impl, request)

    def image_to_video_with_progress(
        self,
        request: ImageToVideoRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        if progress_callback is None:
            return self.image_to_video(request)
        return self.image_to_video(
            self._run_on_runtime_thread(
                self._normalized_with_step_progress, request, progress_callback, self._normalize_video, task="image_to_video"
            )
        )


MLXGenBackendConfig = MFluxBackendConfig
MLXGenVisionBackend = MFluxVisionBackend
