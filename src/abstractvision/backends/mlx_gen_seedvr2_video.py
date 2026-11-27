"""SeedVR2 video restoration on MLX-Gen: the engine's plan, its refusals and the call that applies it.

AbstractVision keeps no policy of its own here. `SeedVR2VideoRestorePlan.for_video` (mlx-gen 0.40.2) probes
the clip and decides the scale, the chunking, the memory budget and the low-RAM profile, as `mlxgen upscale
--video-path` does; the run applies the plan as the engine's CLI does (`models/seedvr2/cli/seedvr2_upscale.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..errors import CapabilityNotSupportedError, OptionalDependencyMissingError

TASK = "video_upscale"
# The result is MP4 whatever the source's container, as the engine writes it: its CLI names every restored
# video video.mp4 (cli/output_paths.normalize_output_template), its writer encodes H.264 with the MP4 muxer's
# `-movflags +faststart` (utils/video_util.VideoStreamWriter), the audio copy adds AAC with the same flag
# (VideoUtil._build_audio_copy_command), and restore_video_to_path calls its result "a restored MP4".
OUTPUT_NAME = "video.mp4"
MIME_TYPE = "video/mp4"
# The refusal reasons the plan raises, and the request field each one names (None: the clip as a whole).
REFUSAL_FIELDS = {
    "scale_above_safe_profile": "resolution",
    "beyond_memory_profile": None,
    "below_temporal_quality_floor": None,
}


def lazy_import_plan() -> Tuple[Any, Any]:
    try:
        from mflux.models.seedvr2 import SeedVR2VideoRestorePlan, SeedVR2VideoRestoreRefused  # type: ignore
    except Exception as e:
        from .mflux import MLX_GEN_VERSION_FLOOR

        raise OptionalDependencyMissingError(
            f"MLX-Gen SeedVR2 video restoration requires mlx-gen>={MLX_GEN_VERSION_FLOOR}. "
            'Install/upgrade it with `pip install "abstractvision[mlx-gen]"`.'
        ) from e
    return SeedVR2VideoRestorePlan, SeedVR2VideoRestoreRefused


def plan(
    video_path: Path,
    *,
    model_config: Any,
    model_path: Optional[str],
    resolution: Optional[Any],
    start_seconds: Optional[float],
    max_frames: Optional[int],
    weights_resident: bool,
) -> Any:
    """Check point 2: the engine's plan for this clip, before any weights load. A value the request leaves
    unset is the plan's own default (1x, from the clip's start, to its end).

    A refusal of the plan becomes a CapabilityNotSupportedError with the engine's message; an argument
    error stays the engine's ValueError; a clip the engine's reader cannot decode is a ValueError naming no
    temporary path. `weights_resident`: the model's weights are loaded already, so the memory the plan finds
    available is net of them.
    """
    Plan, Refused = lazy_import_plan()
    given = {"start_seconds": start_seconds, "max_frames": max_frames, "resolution": resolution}
    try:
        return Plan.for_video(
            video_path,
            model_config=model_config,
            model_path=model_path,
            weights_resident=weights_resident,
            **{name: value for name, value in given.items() if value is not None},
        )
    except Refused as e:
        raise CapabilityNotSupportedError(
            str(e), field=REFUSAL_FIELDS.get(getattr(e, "reason", None)), supported=None
        ) from None
    except Exception as e:
        # PyAV's errors (av.error.*) are the reader's: the clip, not an argument.
        if not type(e).__module__.startswith("av."):
            raise
        detail = str(e).replace(str(video_path), "video")
        raise ValueError(f"video is not a readable clip: {type(e).__name__}: {detail}") from None


def restore_kwargs(
    restore_plan: Any,
    *,
    seed: int,
    softness: Optional[float],
    drop_audio: Optional[bool],
    video_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """`SeedVR2.restore_video_to_path` keywords, as the engine's CLI passes them from the plan (an unset
    softness or drop_audio is the engine's own default)."""
    return {
        "seed": int(seed),
        "video_path": video_path,
        "resolution": restore_plan.resolution,
        **({"softness": float(softness)} if softness is not None else {}),
        **({"drop_audio": bool(drop_audio)} if drop_audio is not None else {}),
        "start_seconds": restore_plan.start_seconds,
        "max_frames": restore_plan.max_frames,
        "output_path": output_path,
        # The sidecar is the run's engine metadata.
        "export_json_metadata": True,
        "overwrite": True,
        "temporal_chunk_size": restore_plan.effective_chunk_size,
        "temporal_chunk_overlap": restore_plan.effective_chunk_overlap,
        "enforce_memory_budget": restore_plan.enforce_memory_budget,
    }
