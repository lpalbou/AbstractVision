from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union


@dataclass(frozen=True)
class ProviderModelInfo:
    """A model entry returned by a provider catalog endpoint.

    Provider catalogs are runtime metadata from a remote/local service. They do
    not replace the packaged capability registry, which remains the source for
    AbstractVision's known model/task metadata.
    """

    id: str
    object: Optional[str] = None
    created: Optional[int] = None
    owned_by: Optional[str] = None
    capabilities: Sequence[str] = field(default_factory=tuple)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderAdapterInfo:
    """A locally discoverable adapter entry returned by a backend inventory."""

    id: str
    repo_id: Optional[str] = None
    base_models: Sequence[str] = field(default_factory=tuple)
    compatible_models: Sequence[str] = field(default_factory=tuple)
    compatible_tasks: Sequence[str] = field(default_factory=tuple)
    suggested_target_roles: Sequence[str] = field(default_factory=tuple)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisionBackendCapabilities:
    """Backend-level capability constraints (optional; additive).

    This complements the model registry (what a model *can* do) with runtime/backend
    constraints (what a configured backend *will* do).
    """

    supported_tasks: Optional[Sequence[str]] = None
    supports_mask: Optional[bool] = None
    supports_control_image: Optional[bool] = None
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    max_fps: Optional[int] = None
    max_frames: Optional[int] = None


@dataclass(frozen=True)
class VideoProgressEvent:
    """Normalized progress event for local generation backends.

    The name is kept for compatibility with the initial video-only callback
    surface. Image generation backends may also emit this type.
    `progress` is the canonical backend progress fraction. For MLX-Gen image
    and video generation this is denoise-step progress; `frame_progress` is
    additional video context.
    """

    phase: str
    frame: Optional[int] = None
    total_frames: Optional[int] = None
    step: Optional[int] = None
    total_steps: Optional[int] = None
    progress: Optional[float] = None
    step_progress: Optional[float] = None
    frame_progress: Optional[float] = None
    task: Optional[str] = None
    timestep: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoRAAdapterSpec:
    """Package-owned request shape for one LoRA adapter attachment."""

    source: str
    scale: Optional[float] = None
    weight_name: Optional[str] = None
    subfolder: Optional[str] = None
    adapter_name: Optional[str] = None
    target_role: Optional[str] = None


@dataclass(frozen=True)
class ImageGenerationRequest:
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    control_image: Optional[bytes] = None
    control_strength: Optional[float] = None
    lora_adapters: Sequence[LoRAAdapterSpec] = field(default_factory=tuple)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageEditRequest:
    prompt: str
    image: bytes
    mask: Optional[bytes] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    lora_adapters: Sequence[LoRAAdapterSpec] = field(default_factory=tuple)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageUpscaleRequest:
    image: bytes
    resolution: Optional[Union[int, str]] = None
    scale: Optional[Union[int, float, str]] = None
    seed: Optional[int] = None
    softness: Optional[float] = None
    quantize: Optional[int] = None
    vae_tiling: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiAngleRequest:
    prompt: str
    reference_image: Optional[bytes] = None
    angles: Sequence[str] = ("front", "three_quarter", "side", "back")
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoGenerationRequest:
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_2: Optional[float] = None
    flow_shift: Optional[float] = None
    lora_adapters: Sequence[LoRAAdapterSpec] = field(default_factory=tuple)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageToVideoRequest:
    image: bytes
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_2: Optional[float] = None
    flow_shift: Optional[float] = None
    lora_adapters: Sequence[LoRAAdapterSpec] = field(default_factory=tuple)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedAsset:
    """Generic return type for generated media."""

    media_type: str  # "image" | "video"
    data: bytes
    mime_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
