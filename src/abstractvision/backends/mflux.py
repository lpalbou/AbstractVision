from __future__ import annotations

import random
import os
import tempfile
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..errors import CapabilityNotSupportedError, OptionalDependencyMissingError
from ..lora_adapters import (
    adapter_source_for_mlx_gen,
    lora_scales_for_backend,
    lora_target_roles_for_backend,
    resolve_request_lora_adapters,
    serialize_lora_adapters,
)
from ..model_capabilities import VisionModelCapabilitiesRegistry
from ..model_downloads import (
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
    GeneratedAsset,
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
from .base_backend import VisionBackend

MLX_GEN_RUNTIME = "mlx-gen"
MFLUX_PROVIDER = "mflux"
MLX_GEN_VERSION_FLOOR = "0.18.18"
WAN_TI2V_MODEL_KEY = "wan2.2-ti2v-5b"
WAN_T2V_A14B_MODEL_KEY = "wan2.2-t2v-a14b"
WAN_I2V_A14B_MODEL_KEY = "wan2.2-i2v-a14b"
SEEDVR2_3B_MODEL_KEY = "seedvr2-3b"
SEEDVR2_7B_MODEL_KEY = "seedvr2-7b"
WAN_DEFAULT_WIDTH = 1280
WAN_DEFAULT_HEIGHT = 704
WAN_DEFAULT_FRAMES = 121
WAN_DEFAULT_STEPS = 50
WAN_DEFAULT_FPS = 24
WAN_DEFAULT_GUIDANCE = 5.0
WAN_TI2V_PRACTICAL_MIN_WIDTH = 832
WAN_TI2V_PRACTICAL_MIN_HEIGHT = 480
WAN_A14B_PROOF_WIDTH = 480
WAN_A14B_PROOF_HEIGHT = 240
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
class _MFluxModelDef:
    key: str
    config_method: str
    family: str
    default_steps: int
    default_guidance: Optional[float]
    supports_negative_prompt: bool = False
    supports_guidance_override: bool = True
    image_edit_catalog_rank: int = 50
    default_width: int = WAN_DEFAULT_WIDTH
    default_height: int = WAN_DEFAULT_HEIGHT
    default_frames: int = WAN_DEFAULT_FRAMES
    default_fps: int = WAN_DEFAULT_FPS
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
        default_steps=WAN_DEFAULT_STEPS,
        default_guidance=WAN_DEFAULT_GUIDANCE,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        default_width=1280,
        default_height=704,
        default_flow_shift=5.0,
    ),
    WAN_T2V_A14B_MODEL_KEY: _MFluxModelDef(
        key=WAN_T2V_A14B_MODEL_KEY,
        config_method="wan2_2_t2v_a14b",
        family="wan-video",
        default_steps=40,
        default_guidance=4.0,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        default_width=1280,
        default_height=720,
        default_frames=81,
        default_fps=16,
        default_guidance_2=3.0,
        default_flow_shift=3.0,
    ),
    WAN_I2V_A14B_MODEL_KEY: _MFluxModelDef(
        key=WAN_I2V_A14B_MODEL_KEY,
        config_method="wan2_2_i2v_a14b",
        family="wan-video",
        default_steps=40,
        default_guidance=3.5,
        supports_negative_prompt=True,
        supports_guidance_override=True,
        default_width=1280,
        default_height=720,
        default_frames=81,
        default_fps=16,
        default_guidance_2=3.5,
        default_flow_shift=3.0,
    ),
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
}


def _mflux_parameter_metadata(model_def: _MFluxModelDef) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {"steps": int(model_def.default_steps)}
    constraints: Dict[str, Any] = {}
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
    return (
        exc.__class__.__name__ == "DownloadRequiredError"
        or hasattr(exc, "download_command")
        or hasattr(exc, "prepare_command")
    )


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
    if "seedvr2" in s and s.startswith("abstractframework/") and ("8bit" in s or "q8" in s):
        return 0
    if "seedvr2" in s and s.startswith("abstractframework/") and ("4bit" in s or "q4" in s):
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
        and str(getattr(preset, "source", "") or "").strip().lower() == "abstractframework-mlx-gen"
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
    try:
        lines = readme_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}
    payload: Dict[str, Any] = {}
    index = 1
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        index += 1
        if stripped == "---":
            break
        if not stripped or ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = str(key or "").strip()
        if not key:
            continue
        text = str(value or "").strip()
        if text:
            payload[key] = text.strip("'\"")
            continue
        items: List[str] = []
        while index < len(lines):
            child = lines[index].strip()
            if child == "---":
                break
            if child.startswith("- "):
                items.append(child[2:].strip().strip("'\""))
                index += 1
                continue
            if child:
                break
            index += 1
        payload[key] = items
    return payload


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


def _adapter_candidate_files(snapshot_dir: Path) -> List[Path]:
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
                if getattr(preset, "source", "") == "abstractframework-mlx-gen":
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
        if getattr(preset, "source", "") != "abstractframework-mlx-gen":
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
                repo_id=None,
                source_label=source_label,
                source_detail=f"{source_label} ({entry})",
            )
            out.setdefault(base, candidate)
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


def _infer_base_model(*values: Any) -> Optional[str]:
    for value in values:
        s = _norm(value)
        if not s:
            continue
        if s in _KNOWN_MODEL_ALIASES:
            return _KNOWN_MODEL_ALIASES[s]
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


class MFluxVisionBackend(VisionBackend):
    """Compatibility class for the local MLX-Gen image and video backend."""

    def __init__(self, *, config: MFluxBackendConfig):
        self._cfg = config
        self._model: Any = None
        self._model_key: Optional[Tuple[Any, ...]] = None
        self._warmed_model_key: Optional[Tuple[Any, ...]] = None
        self._resolved_model_path: Optional[str] = None
        self._resolved_base_model: Optional[str] = None
        self._resolved_quantization_bits: Optional[int] = None
        self._runtime_lock = threading.Lock()
        self._runtime_queue: Optional[queue.Queue[Any]] = None
        self._runtime_thread: Optional[threading.Thread] = None
        self._runtime_thread_id: Optional[int] = None

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
                    try:
                        result = fn(*args, **kwargs)
                    except BaseException as e:
                        future.set_exception(e)
                    else:
                        future.set_result(result)

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
        self._resolved_model_path = None
        self._resolved_base_model = None
        self._resolved_quantization_bits = None
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    def unload(self) -> None:
        if self._runtime_queue is None and self._runtime_thread is None:
            self._unload_impl()
            return
        self._run_on_runtime_thread(self._unload_impl)

    def preload(self) -> None:
        self._run_on_runtime_thread(self._preload_impl)

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

    def get_capabilities(self) -> VisionBackendCapabilities:
        resolved_base = str(
            self._resolved_base_model
            or _infer_base_model(self._cfg.base_model, self._cfg.model)
            or ""
        ).strip()
        return VisionBackendCapabilities(
            supported_tasks=self._supported_task_names(),
            supports_mask=resolved_base in {"fibo-edit", "fibo-edit-rmbg"},
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

    def _mlx_gen_lora_init_kwargs(self, adapters: Sequence[Any]) -> Dict[str, Any]:
        if not adapters:
            return {}
        lora_paths = [adapter_source_for_mlx_gen(spec) for spec in adapters]
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
        return metadata if isinstance(metadata, dict) else {}

    def _route_capabilities_for_model(self, model_selector: str) -> Dict[str, List[Dict[str, Any]]]:
        try:
            get_model_capabilities, _resolve_generation_plan = _lazy_import_mlx_gen_task_inference()
            capabilities = get_model_capabilities(model=model_selector)
        except Exception:
            return {}
        out: Dict[str, List[Dict[str, Any]]] = {}
        for capability in getattr(capabilities, "capabilities", ()) or ():
            raw = getattr(capability, "__dict__", None)
            if not isinstance(raw, dict):
                continue
            task = str(raw.get("public_task") or "").strip().replace("-", "_")
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
            base_model = _infer_base_model(preset.key, preset.repo_id, preset.upstream_repo_id)
            tasks = self._supported_task_names(
                model_id=str(preset.upstream_repo_id or preset.repo_id),
                base_model=base_model,
            )
            if task_s and task_s not in tasks:
                continue
            model_def = _MFLUX_MODELS.get(str(base_model or ""))
            parameter_metadata = (
                _mflux_parameter_metadata(model_def) if model_def is not None else {}
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
            route_capabilities = self._route_capabilities_for_model(model_selector)
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
                preset_base = _infer_base_model(preset.key, preset.repo_id, preset.upstream_repo_id)
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
                    adapter_files = _adapter_candidate_files(snapshot_dir)
                    if not adapter_files:
                        continue
                    for file_path in adapter_files:
                        relative_path = file_path.relative_to(snapshot_dir).as_posix()
                        handle = f"{repo_id}:{relative_path}"
                        compatible_base = _infer_base_model(handle, repo_id, *base_models)
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
                                base_models=base_models,
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
                                },
                            )
                        )
                        seen.add(handle.lower())
        out.sort(key=lambda item: (tuple(item.compatible_models), str(item.repo_id or ""), str(item.id or "")))
        return out

    def _resolve_model(self) -> Tuple[str, str]:
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
                self._resolved_quantization_bits = _infer_bits_from_text(configured_model)
                return str(expanded), base

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
                if snapshot_dir is not None and _has_model_files(snapshot_dir):
                    self._resolved_quantization_bits = (
                        int(preset.quantization_bits)
                        if preset.quantization_bits is not None
                        else None
                    )
                    return str(snapshot_dir), _base_model_for_preset(
                        configured_base, configured_model, preset
                    )
                if not self._cfg.allow_download:
                    raise OptionalDependencyMissingError(
                        f"MLX-Gen model preset {configured_model!r} is not available in the Hugging Face cache. "
                        f"Run: abstractvision download {preset.repo_id} --provider mlx-gen"
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
                self._resolved_quantization_bits = (
                    int(preset.quantization_bits) if preset.quantization_bits is not None else None
                )
                return str(downloaded), _base_model_for_preset(
                    configured_base, configured_model, preset
                )

            if _looks_like_path(configured_model):
                raise OptionalDependencyMissingError(
                    f"MLX-Gen model path does not exist: {configured_model}"
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
                    self._resolved_quantization_bits = _infer_bits_from_text(configured_model)
                    return str(cached), base
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
                try:
                    downloaded = download_hf_repo_snapshot(
                        configured_model,
                        cache_dir=cache_root,
                    )
                except RuntimeError as e:
                    raise OptionalDependencyMissingError(str(e)) from e
                self._resolved_quantization_bits = _infer_bits_from_text(configured_model)
                return str(downloaded), base

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
                self._resolved_quantization_bits = _infer_bits_from_text(discovered_model.repo_id)
                return str(discovered_model.snapshot_dir), configured_base or key

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
    ) -> Tuple[Any, _MFluxModelDef]:
        model_path, base_model = self._resolve_model()
        if base_model not in _MFLUX_MODELS:
            raise OptionalDependencyMissingError(
                f"Unsupported MLX-Gen base model {base_model!r}. "
                f"Supported: {', '.join(sorted(_MFLUX_MODELS))}"
        )
        model_def = _MFLUX_MODELS[base_model]
        model_variant = "flux2-edit" if edit_variant and model_def.family == "flux2" else "default"
        effective_lora_adapters = tuple(lora_adapters) if lora_adapters else self._config_lora_adapters()
        lora_init_kwargs = self._mlx_gen_lora_init_kwargs(effective_lora_adapters)
        if model_def.family != "wan-video":
            lora_init_kwargs.pop("lora_target_roles", None)
        key = (
            model_path,
            base_model,
            model_variant,
            tuple(lora_init_kwargs.get("lora_paths") or ()),
            tuple(lora_init_kwargs.get("lora_scales") or ()),
            tuple(lora_init_kwargs.get("lora_target_roles") or ()),
        )
        if self._model is not None and self._model_key == key:
            return self._model, model_def

        ModelConfig, _DownloadRequiredError, Flux2Klein, _Flux2KleinEdit, ZImage, ZImageTurbo = (
            _lazy_import_mflux()
        )
        model_config = self._model_config_for_definition(ModelConfig, model_def)
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
        else:
            raise OptionalDependencyMissingError(
                f"Unsupported MLX-Gen model family {model_def.family!r}."
            )
        kwargs: Dict[str, Any] = {
            "model_config": model_config,
            "model_path": model_path,
        }
        kwargs.update(lora_init_kwargs)
        try:
            self._model = cls(**kwargs)
        except Exception as e:
            if _is_mlx_gen_download_required(e):
                raise _wrap_mlx_gen_download_required(e) from e
            raise
        self._model_key = key
        self._warmed_model_key = None
        self._resolved_model_path = model_path
        self._resolved_base_model = base_model
        return self._model, model_def

    def _ensure_seedvr2_model_impl(
        self,
        *,
        quantize: Optional[int],
    ) -> Tuple[Any, _MFluxModelDef]:
        model_path, base_model = self._resolve_model()
        if base_model not in _MFLUX_MODELS:
            raise OptionalDependencyMissingError(
                f"Unsupported MLX-Gen base model {base_model!r}. "
                f"Supported: {', '.join(sorted(_MFLUX_MODELS))}"
            )
        model_def = _MFLUX_MODELS[base_model]
        if model_def.family != "seedvr2-upscale":
            raise CapabilityNotSupportedError(
                "MLX-Gen image_upscale is implemented for SeedVR2 models today "
                f"(got {model_def.family!r})."
            )
        key = (model_path, base_model, "seedvr2-upscale", quantize)
        if self._model is not None and self._model_key == key:
            return self._model, model_def

        ModelConfig, SeedVR2, _ScaleFactor, _TilingConfig = _lazy_import_mflux_seedvr2()
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
        self._resolved_model_path = model_path
        self._resolved_base_model = base_model
        self._resolved_quantization_bits = quantize or _infer_bits_from_text(model_path)
        return self._model, model_def

    def _warmup_request(self, model_def: _MFluxModelDef) -> ImageGenerationRequest:
        return ImageGenerationRequest(
            prompt="abstractvision preload warmup",
            steps=int(model_def.default_steps),
            guidance_scale=model_def.default_guidance,
            seed=0,
        )

    def _preload_impl(self) -> None:
        model_def = self._resolved_model_def()
        if model_def.family == "seedvr2-upscale":
            self._ensure_seedvr2_model_impl(quantize=self._default_upscale_quantize())
            self._warmed_model_key = self._model_key
            return
        _model, model_def = self._ensure_model_impl()
        if self._model_key is not None and self._warmed_model_key == self._model_key:
            return
        if model_def.family == "wan-video":
            self._warmed_model_key = self._model_key
            return
        self._generate_impl(self._warmup_request(model_def))

    def _resolved_model_def(self) -> _MFluxModelDef:
        _model_path, base_model = self._resolve_model()
        if base_model not in _MFLUX_MODELS:
            raise OptionalDependencyMissingError(
                f"Unsupported MLX-Gen base model {base_model!r}. "
                f"Supported: {', '.join(sorted(_MFLUX_MODELS))}"
            )
        return _MFLUX_MODELS[base_model]

    def _model_config_for_definition(self, ModelConfig: Any, model_def: _MFluxModelDef) -> Any:
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
        model_def = self._resolved_model_def()
        steps = int(request.steps) if request.steps is not None else int(model_def.default_steps)
        if model_def.family == "flux2" and steps < 2:
            steps = 2
        guidance = (
            float(request.guidance_scale)
            if request.guidance_scale is not None
            else model_def.default_guidance
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
        model_def = self._resolved_model_def()
        steps = int(request.steps) if request.steps is not None else int(model_def.default_steps)
        if model_def.family == "flux2" and steps < 2:
            steps = 2
        guidance = (
            float(request.guidance_scale)
            if request.guidance_scale is not None
            else model_def.default_guidance
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
    ) -> Dict[str, Any]:
        model_def = self._resolved_model_def()
        if model_def.family != "wan-video":
            raise CapabilityNotSupportedError(
                f"MLX-Gen video generation is only implemented for Wan video models today (got {model_def.family!r})."
            )
        extra = dict(request.extra or {})
        steps = int(request.steps) if request.steps is not None else int(model_def.default_steps)
        if steps < 1:
            steps = 1
        guidance = (
            float(request.guidance_scale)
            if request.guidance_scale is not None
            else model_def.default_guidance
        )
        width = int(request.width) if request.width is not None else int(model_def.default_width)
        height = int(request.height) if request.height is not None else int(model_def.default_height)
        self._validate_wan_video_dimensions(model_def=model_def, width=width, height=height)
        fps = int(request.fps) if request.fps is not None else int(model_def.default_fps)
        num_frames = (
            int(request.num_frames) if request.num_frames is not None else int(model_def.default_frames)
        )
        if fps < 1:
            fps = int(model_def.default_fps)
        if num_frames < 1:
            num_frames = 1
        max_sequence_length = extra.get("max_sequence_length")
        if max_sequence_length is not None:
            try:
                extra["max_sequence_length"] = max(1, int(max_sequence_length))
            except Exception:
                extra.pop("max_sequence_length", None)
        flow_shift = request.flow_shift
        if flow_shift is None and extra.get("flow_shift") is not None:
            try:
                flow_shift = float(extra.get("flow_shift"))
            except Exception:
                flow_shift = None
        if flow_shift is None and model_def.key == WAN_TI2V_MODEL_KEY:
            longer = max(width, height)
            shorter = min(width, height)
            if shorter <= 480 and longer <= 832:
                flow_shift = 3.0
        if flow_shift is not None:
            flow_shift = float(flow_shift)
            extra["flow_shift"] = flow_shift
        return {
            "steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
            "fps": fps,
            "num_frames": num_frames,
            "flow_shift": flow_shift,
            "extra": extra,
        }

    def _validate_wan_video_dimensions(
        self,
        *,
        model_def: _MFluxModelDef,
        width: int,
        height: int,
    ) -> None:
        if width <= 0 or height <= 0:
            raise CapabilityNotSupportedError("Wan video width and height must be positive integers.")
        multiple = 32 if model_def.key == WAN_TI2V_MODEL_KEY else 16
        if width % multiple != 0 or height % multiple != 0:
            raise CapabilityNotSupportedError(
                f"{model_def.key} requires width/height multiples of {multiple}. "
                f"Received {width}x{height}."
            )
        if model_def.key != WAN_TI2V_MODEL_KEY:
            return
        longer = max(width, height)
        shorter = min(width, height)
        if longer < WAN_TI2V_PRACTICAL_MIN_WIDTH or shorter < WAN_TI2V_PRACTICAL_MIN_HEIGHT:
            raise CapabilityNotSupportedError(
                "Wan2.2 TI2V-5B should not be run below 832x480 or 480x832 through AbstractVision. "
                f"Received {width}x{height}. Use the task-specific Wan A14B routes for lower-cost "
                "validation sizes such as 480x240."
            )

    def normalize_video_generation_request(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationRequest:
        values = self._normalize_video_request_values(request=request)
        return replace(
            request,
            steps=values["steps"],
            guidance_scale=values["guidance_scale"],
            width=values["width"],
            height=values["height"],
            fps=values["fps"],
            num_frames=values["num_frames"],
            flow_shift=values["flow_shift"],
            extra=values["extra"],
        )

    def normalize_image_to_video_request(
        self,
        request: ImageToVideoRequest,
    ) -> ImageToVideoRequest:
        values = self._normalize_video_request_values(request=request)
        return replace(
            request,
            steps=values["steps"],
            guidance_scale=values["guidance_scale"],
            width=values["width"],
            height=values["height"],
            fps=values["fps"],
            num_frames=values["num_frames"],
            flow_shift=values["flow_shift"],
            extra=values["extra"],
        )

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

    def _normalize_upscale_quantize(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"", "0", "none", "false", "off", "no"}:
            return None
        try:
            quantize = int(value)
        except Exception as exc:
            raise ValueError("image_upscale quantize must be 3, 4, 5, 6, 8, or none.") from exc
        if quantize not in {3, 4, 5, 6, 8}:
            raise ValueError("image_upscale quantize must be 3, 4, 5, 6, 8, or none.")
        return quantize

    def _default_upscale_quantize(self) -> Optional[int]:
        model_def = self._resolved_model_def()
        if model_def.family != "seedvr2-upscale":
            return None
        if self._resolved_quantization_bits in {3, 4, 5, 6, 8}:
            return None
        return 8

    def normalize_image_upscale_request(
        self,
        request: ImageUpscaleRequest,
    ) -> ImageUpscaleRequest:
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
        quantize_raw = request.quantize if request.quantize is not None else extra.pop("quantize", None)
        quantize_value = (
            self._default_upscale_quantize()
            if quantize_raw is None
            else self._normalize_upscale_quantize(quantize_raw)
        )
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
        try:
            unsubscribe = subscribe(callback, task=task) if task is not None else subscribe(callback, task=None)
        except TypeError:
            unsubscribe = subscribe(callback)
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

    def _generate_impl(
        self,
        request: ImageGenerationRequest,
        *,
        image_path: Optional[Path] = None,
        image_strength: Optional[float] = None,
    ) -> GeneratedAsset:
        request = self.normalize_image_generation_request(request)
        effective_loras = self._effective_request_lora_adapters(request)
        model, model_def = self._ensure_model_impl(lora_adapters=effective_loras)
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
        progress_callbacks, step_progress_callback = _pop_progress_callbacks(extra)

        progress_task = "image-to-image" if image_path is not None else "text-to-image"
        default_task = "image_to_image" if image_path is not None else "text_to_image"

        def _progress_bridge(raw_event: Any) -> None:
            event = _normalize_video_progress_event(raw_event)
            if event.task is None:
                event = replace(event, task=default_task)
            for callback in progress_callbacks:
                callback(event)
            if step_progress_callback is not None:
                current = event.step if event.step is not None else 0
                step_progress_callback(current, event.total_steps)

        unsubscribe = (
            self._subscribe_progress(model, _progress_bridge, task=progress_task)
            if progress_callbacks or step_progress_callback is not None
            else (lambda: None)
        )

        try:
            generated = model.generate_image(**kwargs)
        except Exception as e:
            if _is_mlx_gen_download_required(e):
                raise _wrap_mlx_gen_download_required(e) from e
            raise
        finally:
            unsubscribe()
        if self._model_key is not None:
            self._warmed_model_key = self._model_key
        pil_image = getattr(generated, "image", generated)
        mlx_metadata = self._generated_image_metadata(generated)
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        data = buf.getvalue()
        image_strength_used = kwargs.get("image_strength") if image_path is not None else None
        requested_loras = serialize_lora_adapters(effective_loras)
        return GeneratedAsset(
            media_type="image",
            data=data,
            mime_type="image/png",
            metadata={
                "source": MLX_GEN_RUNTIME,
                "engine": MLX_GEN_RUNTIME,
                "legacy_engine": MFLUX_PROVIDER,
                "runtime_package": MLX_GEN_RUNTIME,
                "model": self._resolved_model_path,
                "base_model": self._resolved_base_model,
                "quantization_bits": self._resolved_quantization_bits,
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
            },
        )

    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
        return self._run_on_runtime_thread(self._generate_impl, request)

    def generate_image_with_progress(
        self,
        request: ImageGenerationRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        if progress_callback is None:
            return self.generate_image(request)
        extra = dict(request.extra or {})
        extra["_step_progress_callback"] = progress_callback
        return self.generate_image(replace(request, extra=extra))

    def _edit_image_impl(self, request: ImageEditRequest) -> GeneratedAsset:
        request = self.normalize_image_edit_request(request)
        effective_loras = self._effective_request_lora_adapters(request)

        _model, model_def = self._ensure_model_impl(
            edit_variant=True,
            lora_adapters=effective_loras,
        )
        if request.mask is not None and model_def.family != "fibo-edit":
            raise CapabilityNotSupportedError(
                "MLX-Gen mask edits are currently implemented only for FIBO Edit models."
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
                progress_callbacks, step_progress_callback = _pop_progress_callbacks(extra)
                image_paths = [str(tmp_path)]
                self._extend_reference_image_paths(extra, image_paths, tmp_extra_paths)
                if len(image_paths) > 1 and model_def.family not in {"flux2", "qwen-edit"}:
                    raise CapabilityNotSupportedError(
                        "MLX-Gen multi-reference image edits are supported for FLUX.2 and "
                        f"Qwen Image Edit models, not {model_def.family!r}."
                    )
                kwargs: Dict[str, Any] = {
                    "seed": seed,
                    "prompt": str(request.prompt),
                    "num_inference_steps": steps,
                }
                if model_def.family in {"flux2", "qwen-edit"}:
                    kwargs["image_paths"] = image_paths
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

                def _progress_bridge(raw_event: Any) -> None:
                    event = _normalize_video_progress_event(raw_event)
                    if event.task is None:
                        event = replace(event, task="image_to_image")
                    for callback in progress_callbacks:
                        callback(event)
                    if step_progress_callback is not None:
                        current = event.step if event.step is not None else 0
                        step_progress_callback(current, event.total_steps)

                unsubscribe = (
                    self._subscribe_progress(_model, _progress_bridge, task=task_name)
                    if progress_callbacks or step_progress_callback is not None
                    else (lambda: None)
                )
                try:
                    generated = _model.generate_image(**kwargs)
                except Exception as e:
                    if _is_mlx_gen_download_required(e):
                        raise _wrap_mlx_gen_download_required(e) from e
                    raise
                finally:
                    unsubscribe()
                pil_image = getattr(generated, "image", generated)
                mlx_metadata = self._generated_image_metadata(generated)
                buf = BytesIO()
                pil_image.save(buf, format="PNG")
                requested_loras = serialize_lora_adapters(effective_loras)
                return GeneratedAsset(
                    media_type="image",
                    data=buf.getvalue(),
                    mime_type="image/png",
                    metadata={
                        "source": MLX_GEN_RUNTIME,
                        "engine": MLX_GEN_RUNTIME,
                        "legacy_engine": MFLUX_PROVIDER,
                        "runtime_package": MLX_GEN_RUNTIME,
                        "model": self._resolved_model_path,
                        "base_model": self._resolved_base_model,
                        "quantization_bits": self._resolved_quantization_bits,
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
                    },
                )
            return self._generate_impl(
                gen_request,
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
        extra = dict(request.extra or {})
        extra["_step_progress_callback"] = progress_callback
        return self.edit_image(replace(request, extra=extra))

    def _upscale_image_impl(self, request: ImageUpscaleRequest) -> GeneratedAsset:
        request = self.normalize_image_upscale_request(request)
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
        progress_callbacks, step_progress_callback = _pop_progress_callbacks(extra)
        seed = int(request.seed) if request.seed is not None else random.randint(0, 1_000_000_000)
        resolution_arg = self._coerce_seedvr2_resolution(request.resolution)
        softness = 0.25 if request.softness is None else float(request.softness)

        def _progress_bridge(raw_event: Any) -> None:
            event = _normalize_video_progress_event(raw_event)
            event = replace(event, task="image_upscale")
            for callback in progress_callbacks:
                callback(event)
            if step_progress_callback is not None:
                current = event.step if event.step is not None else 0
                step_progress_callback(current, event.total_steps)

        tmp_path = self._write_temp_image_bytes(request.image)
        unsubscribe = (
            self._subscribe_progress(model, _progress_bridge, task=None)
            if progress_callbacks or step_progress_callback is not None
            else (lambda: None)
        )
        try:
            generated = model.generate_image(
                seed=seed,
                image_path=tmp_path,
                resolution=resolution_arg,
                softness=softness,
            )
        except Exception as e:
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
                "model": self._resolved_model_path,
                "base_model": self._resolved_base_model,
                "quantization_bits": self._resolved_quantization_bits,
                "seed": seed,
                "steps": 1,
                "resolution": resolution_meta,
                **({"scale": request.scale} if request.scale is not None else {}),
                "softness": softness,
                "vae_tiling": request.vae_tiling,
                "vae_tiling_policy": "explicit" if request.vae_tiling is not None else "runtime_default",
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
        extra = dict(request.extra or {})
        extra["_step_progress_callback"] = progress_callback
        return self.upscale_image(replace(request, extra=extra))

    def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]:
        raise CapabilityNotSupportedError(
            "MLX-Gen backend does not implement multi-view generation."
        )

    def _read_generated_video_bytes(self, generated: Any) -> bytes:
        save = getattr(generated, "save", None)
        if not callable(save):
            raise RuntimeError("MLX-Gen video result does not expose a callable save(...) method.")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fp:
            tmp_path = Path(fp.name)
        try:
            try:
                save(path=tmp_path, overwrite=True)
            except TypeError:
                save(tmp_path)
            return tmp_path.read_bytes()
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _generated_video_metadata(self, generated: Any) -> Dict[str, Any]:
        get_metadata = getattr(generated, "_get_metadata", None)
        if not callable(get_metadata):
            return {}
        try:
            metadata = get_metadata()
        except Exception:
            return {}
        return metadata if isinstance(metadata, dict) else {}

    def _generate_video_impl(
        self,
        request: Union[VideoGenerationRequest, ImageToVideoRequest],
        *,
        image_path: Optional[Path] = None,
        conditioning_image_metadata: Optional[Dict[str, Any]] = None,
    ) -> GeneratedAsset:
        if isinstance(request, VideoGenerationRequest):
            request = self.normalize_video_generation_request(request)
            task = "text_to_video"
        else:
            request = self.normalize_image_to_video_request(request)
            task = "image_to_video"
        effective_loras = self._effective_request_lora_adapters(request)
        model, model_def = self._ensure_model_impl(lora_adapters=effective_loras)
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
        guidance_2 = (
            request.guidance_2
            if request.guidance_2 is not None
            else model_def.default_guidance_2
        )
        flow_shift = request.flow_shift
        progress_callbacks, step_progress_callback = _pop_progress_callbacks(extra)

        def _progress_bridge(raw_event: Any) -> None:
            event = _normalize_video_progress_event(raw_event)
            if event.task is None:
                event = replace(event, task=task)
            for callback in progress_callbacks:
                callback(event)
            if step_progress_callback is not None:
                current = event.step if event.step is not None else (event.frame or 0)
                total = event.total_steps if event.total_steps is not None else event.total_frames
                step_progress_callback(current, total)

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
        if progress_callbacks or step_progress_callback is not None:
            kwargs["progress_callback"] = _progress_bridge

        try:
            generated = model.generate_video(**kwargs)
        except Exception as e:
            if _is_mlx_gen_download_required(e):
                raise _wrap_mlx_gen_download_required(e) from e
            raise
        if self._model_key is not None:
            self._warmed_model_key = self._model_key
        data = self._read_generated_video_bytes(generated)
        mlx_metadata = self._generated_video_metadata(generated)
        requested_loras = serialize_lora_adapters(effective_loras)
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
                "model": self._resolved_model_path,
                "base_model": self._resolved_base_model,
                "quantization_bits": self._resolved_quantization_bits,
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
                **(
                    {"conditioning_image": conditioning_image_metadata}
                    if conditioning_image_metadata
                    else {}
                ),
                **({"mlx_gen": mlx_metadata} if mlx_metadata else {}),
            },
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
        extra = dict(request.extra or {})
        extra["_step_progress_callback"] = progress_callback
        return self.generate_video(replace(request, extra=extra))

    def _image_to_video_impl(self, request: ImageToVideoRequest) -> GeneratedAsset:
        request = self.normalize_image_to_video_request(request)
        suffix = self._sniff_image_suffix(request.image)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as fp:
            tmp_path = Path(fp.name)
            fp.write(request.image)
        conditioning_path = tmp_path
        conditioning_metadata: Optional[Dict[str, Any]] = None
        try:
            conditioning_path, conditioning_metadata = self._prepare_i2v_conditioning_image(
                tmp_path,
                width=int(request.width) if request.width is not None else WAN_DEFAULT_WIDTH,
                height=int(request.height) if request.height is not None else WAN_DEFAULT_HEIGHT,
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
        extra = dict(request.extra or {})
        extra["_step_progress_callback"] = progress_callback
        return self.image_to_video(replace(request, extra=extra))


MLXGenBackendConfig = MFluxBackendConfig
MLXGenVisionBackend = MFluxVisionBackend
