from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass(frozen=True)
class VisionBackendCapabilities:
    """Backend-level capability constraints (optional; additive).

    This complements the model registry (what a model *can* do) with runtime/backend
    constraints (what a configured backend *will* do).
    """

    supported_tasks: Optional[Sequence[str]] = None
    supports_mask: Optional[bool] = None
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    max_fps: Optional[int] = None
    max_frames: Optional[int] = None


@dataclass(frozen=True)
class ImageGenerationRequest:
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
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
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedAsset:
    """Generic return type for generated media."""

    media_type: str  # "image" | "video"
    data: bytes
    mime_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
