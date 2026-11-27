from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# Memory modes a caller can request on a generation call, from the fastest run to the lowest
# memory peak. "default" runs the backend as configured. "low" and "lowest" trade speed for a
# lower peak on backends that implement them; `VisionBackendCapabilities.memory_modes` lists
# what a configured backend can honour, and a mode it does not list is refused, never ignored.
MEMORY_MODE_DEFAULT = "default"
MEMORY_MODE_LOW = "low"
MEMORY_MODE_LOWEST = "lowest"
MEMORY_MODES: Tuple[str, ...] = (MEMORY_MODE_DEFAULT, MEMORY_MODE_LOW, MEMORY_MODE_LOWEST)
# Process default for requests that do not name a mode; a request value wins.
MEMORY_MODE_ENV_VAR = "ABSTRACTVISION_MEMORY_MODE"


def normalize_memory_mode(value: Any) -> str:
    """Return the canonical memory mode for `value` (None and "" mean "default")."""
    if value is None:
        return MEMORY_MODE_DEFAULT
    text = str(value).strip().lower()
    if not text:
        return MEMORY_MODE_DEFAULT
    if text in MEMORY_MODES:
        return text
    raise ValueError(f"memory_mode must be one of {', '.join(MEMORY_MODES)} (got {value!r}).")


def resolve_memory_mode_with_source(value: Any) -> Tuple[str, bool]:
    """The mode a call asks for, and whether it came from $ABSTRACTVISION_MEMORY_MODE.

    `value` (the request's, already merged with its `extra`) wins, then the environment, then
    "default".
    """
    if value is None or not str(value).strip():
        env_value = os.environ.get(MEMORY_MODE_ENV_VAR)
        if env_value is not None and env_value.strip():
            try:
                return normalize_memory_mode(env_value), True
            except ValueError as e:
                raise ValueError(f"{MEMORY_MODE_ENV_VAR}: {e}") from e
    return normalize_memory_mode(value), False


def highest_memory_mode_at_or_below(mode: str, supported: Sequence[str]) -> str:
    """The lowest-memory mode in `supported` that does not go below `mode` ("default" if none)."""
    rank = MEMORY_MODES.index(normalize_memory_mode(mode))
    for candidate in reversed(MEMORY_MODES[: rank + 1]):
        if candidate in supported:
            return candidate
    return MEMORY_MODE_DEFAULT


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
    # Memory modes this backend can honour for its configured model and device (see
    # MEMORY_MODES); None when the backend does not say, which means "default" only.
    memory_modes: Optional[Sequence[str]] = None
    # Load-time quantization widths the configured model accepts (`quantize`); None or empty
    # means the option is refused.
    quantize_bits: Optional[Sequence[int]] = None
    # Whether `cancel_event` is honoured between steps ("step"), or only checked before a run
    # starts ("start").
    cancellation: Optional[str] = None
    # Whether the configured model generates a soundtrack with its videos.
    generates_audio: Optional[bool] = None


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
    memory_mode: Optional[str] = None
    # Load-time quantization of the model's weights, in bits. Backends list what they accept;
    # a value a backend or route cannot apply is refused.
    quantize: Optional[int] = None
    # A threading.Event-like object (anything with is_set()). When it is set, the backend stops
    # at its next progress boundary and raises GenerationCancelledError.
    cancel_event: Optional[Any] = None


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
    memory_mode: Optional[str] = None
    # Load-time quantization of the model's weights, in bits. Backends list what they accept;
    # a value a backend or route cannot apply is refused.
    quantize: Optional[int] = None
    # A threading.Event-like object (anything with is_set()). When it is set, the backend stops
    # at its next progress boundary and raises GenerationCancelledError.
    cancel_event: Optional[Any] = None
    # Latent image-to-image: how far the source is noised before denoising, in (0, 1]. A backend
    # or route without a latent image-to-image route refuses it.
    image_strength: Optional[float] = None
    # Images after `image`, in order (bytes, or a local file's path): a multi-reference edit. A
    # backend or route without one refuses them.
    reference_images: Sequence[Union[bytes, Path, str]] = ()
    # Outpaint: grow the canvas by a box ("top,right,bottom,left", pixels or "N%"), filling the new
    # area; `outpaint_fill` names the conditioning-canvas fill (None: the route's default) and
    # `outpaint_fill_color` its color ("R,G,B" or "#rrggbb", with outpaint_fill="solid").
    # Reframe: grow the canvas by a box and let the model recompose the whole picture.
    outpaint_padding: Optional[str] = None
    outpaint_fill: Optional[str] = None
    outpaint_fill_color: Optional[str] = None
    reframe_padding: Optional[str] = None


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
    memory_mode: Optional[str] = None
    # See ImageGenerationRequest.cancel_event.
    cancel_event: Optional[Any] = None


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
    memory_mode: Optional[str] = None
    # Load-time quantization of the model's weights, in bits. Backends list what they accept;
    # a value a backend or route cannot apply is refused.
    quantize: Optional[int] = None
    # A threading.Event-like object (anything with is_set()). When it is set, the backend stops
    # at its next progress boundary and raises GenerationCancelledError.
    cancel_event: Optional[Any] = None
    # Video with a generated soundtrack (MiniMax-H3): the `overall_soundscape` and
    # `non_diegetic_music` prompt sections, whether to decode the soundtrack, and the audio
    # schedule shift. A backend or route without a soundtrack refuses them.
    soundscape: Optional[str] = None
    music: Optional[str] = None
    generate_audio: Optional[bool] = None
    audio_shift: Optional[float] = None


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
    memory_mode: Optional[str] = None
    # Load-time quantization of the model's weights, in bits. Backends list what they accept;
    # a value a backend or route cannot apply is refused.
    quantize: Optional[int] = None
    # A threading.Event-like object (anything with is_set()). When it is set, the backend stops
    # at its next progress boundary and raises GenerationCancelledError.
    cancel_event: Optional[Any] = None
    # Video with a generated soundtrack (MiniMax-H3): the `overall_soundscape` and
    # `non_diegetic_music` prompt sections, whether to decode the soundtrack, and the audio
    # schedule shift. A backend or route without a soundtrack refuses them.
    soundscape: Optional[str] = None
    music: Optional[str] = None
    generate_audio: Optional[bool] = None
    audio_shift: Optional[float] = None


@dataclass(frozen=True)
class VideoToVideoRequest:
    """A video edited from a source clip (`video`: its bytes, or a local file read in place)."""

    prompt: str
    video: Union[bytes, Path]
    # How far the source is noised, in (0, 1]; None: the engine's default (0.8).
    video_strength: Optional[float] = None
    # One image; white is the region to change.
    mask: Optional[bytes] = None
    # Images after the clip, in order; and how a masked region is treated (Wan VACE only).
    reference_images: Sequence[bytes] = ()
    masked_region_mode: Optional[str] = None
    conditioning_scale: Optional[float] = None
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
    memory_mode: Optional[str] = None
    # Load-time quantization of the model's weights, in bits. Backends list what they accept;
    # a value a backend or route cannot apply is refused.
    quantize: Optional[int] = None
    # A threading.Event-like object (anything with is_set()). When it is set, the backend stops
    # at its next progress boundary and raises GenerationCancelledError.
    cancel_event: Optional[Any] = None


@dataclass(frozen=True)
class VideoUpscaleRequest:
    """A source clip restored (`video`: its bytes, or a local file read in place)."""

    video: Union[bytes, Path]
    # Shortest edge in pixels or a scale such as "1x"; None: "1x".
    resolution: Optional[Union[int, str]] = None
    # In [0, 1]; None: 0.0.
    softness: Optional[float] = None
    seed: Optional[int] = None
    # Where the clip starts, in seconds (>= 0); None: 0.0.
    start_seconds: Optional[float] = None
    # How many frames are restored (>= 1); None: to the clip's end.
    max_frames: Optional[int] = None
    # True: the result is always silent (the engine does not copy the source's soundtrack); None or False: the
    # engine copies it, and fails the run after the restoration when it cannot.
    drop_audio: Optional[bool] = None
    quantize: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    memory_mode: Optional[str] = None
    # See ImageGenerationRequest.cancel_event.
    cancel_event: Optional[Any] = None


@dataclass(frozen=True)
class GeneratedAudioTrack:
    """The soundtrack generated with a video.

    `muxed` says whether the track is inside the asset's MP4 (`codec`: AAC). When the engine
    could not mux it, `data` holds the track as a WAV file instead (`codec`: its PCM format), so
    it is never lost; `VisionManager` stores it as an artifact of its own.
    """

    sample_rate: int
    channels: int
    duration_seconds: float
    muxed: bool
    codec: Optional[str] = None
    data: Optional[bytes] = None
    mime_type: Optional[str] = None


@dataclass(frozen=True)
class GeneratedAsset:
    """Generic return type for generated media."""

    media_type: str  # "image" | "video"
    data: bytes
    mime_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    audio: Optional[GeneratedAudioTrack] = None

    def audio_summary(self, wav: Any = None) -> Optional[Dict[str, Any]]:
        """The soundtrack as every layer reports it (store, plugin, tools, HTTP); None without one.

        `{"present": True, "sample_rate": int, "channels": int, "codec": str, "muxed": bool,
        "duration_seconds": float}`. `codec` is the MP4's audio codec when `muxed`, else the WAV's
        (`pcm_s16le`). Only when the track is not in the MP4, `"wav"` holds it: `wav` when given
        (the stored WAV's artifact ref), else the WAV bytes.
        """
        track = self.audio
        if track is None:
            return None
        summary: Dict[str, Any] = {
            "present": True,
            "sample_rate": int(track.sample_rate),
            "channels": int(track.channels),
            "codec": track.codec,
            "muxed": bool(track.muxed),
            "duration_seconds": float(track.duration_seconds),
        }
        if not track.muxed:
            summary["wav"] = wav if wav is not None else track.data
        return summary
