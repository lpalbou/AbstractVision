"""What a listing shows about a download preset on this host, read at run time.

`preset_status` answers three questions for one preset without touching the network or loading
weights: is it in the local Hugging Face cache (and how large is it there), which `memory_mode`
values does the engine that serves it honour on this host, and, for MLX-Gen models whose capability
rows publish measured runs, what peak memory each mode reached and the smallest Mac it was tested on.
Every memory figure comes from the installed mlx-gen; nothing is copied into AbstractVision.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .model_cache import framework_hf_cache_roots, incomplete_hf_model_sources, resolve_hf_repo_snapshot
from .model_downloads import VisionModelDownloadPreset, format_bytes, local_model_profile
from .types import MEMORY_MODE_DEFAULT, MEMORY_MODE_LOW, MEMORY_MODE_LOWEST

NOT_AVAILABLE = "not available on this host"

# How each memory_mode runs on MLX-Gen, as the (low_ram, stream_blocks) flags of its measured runs.
_MLX_GEN_MODE_FLAGS: Dict[str, Tuple[bool, bool]] = {
    MEMORY_MODE_DEFAULT: (False, False),
    MEMORY_MODE_LOW: (True, False),
    MEMORY_MODE_LOWEST: (True, True),
}
# The same modes under the names mlx-gen gives them in its smallest-tested-machine table.
_MLX_GEN_MODE_NAMES: Dict[str, str] = {
    MEMORY_MODE_DEFAULT: "default",
    MEMORY_MODE_LOW: "low_ram",
    MEMORY_MODE_LOWEST: "stream_blocks",
}


def installed_snapshot(repo_id: str) -> Optional[Tuple[Path, int]]:
    """The complete cached snapshot of `repo_id` and its size on disk, or None."""

    snapshot = resolve_hf_repo_snapshot(
        str(repo_id), extra_roots=framework_hf_cache_roots(), require_weight_files=True
    )
    if snapshot is None:
        return None
    return snapshot, snapshot_size_bytes(snapshot)


def snapshot_size_bytes(snapshot: Path) -> int:
    """Bytes of the files a snapshot holds (its symlinks resolved, each blob counted once)."""

    seen: set[str] = set()
    total = 0
    for root, dirs, files in os.walk(snapshot):
        dirs[:] = [d for d in dirs if d != ".cache"]
        for name in files:
            try:
                real = Path(root, name).resolve()
                key = str(real)
                if key in seen or not real.is_file():
                    continue
                seen.add(key)
                total += real.stat().st_size
            except Exception:
                continue
    return total


def is_partially_downloaded(repo_id: str) -> bool:
    return bool(
        incomplete_hf_model_sources(
            str(repo_id), extra_roots=framework_hf_cache_roots(), require_weight_files=True
        )
    )


def engine_memory_modes(engine: str, repo_id: str) -> Optional[Tuple[str, ...]]:
    """The memory modes the backend serving `engine` honours for `repo_id` on this host.

    None when that backend cannot run here (its runtime is not installed, or no AbstractVision
    backend loads the artifact)."""

    key = str(engine or "").strip().lower()
    if key in {"mlx-gen", "mflux"}:
        return _mlx_gen_memory_modes(str(repo_id))
    if key == "diffusers":
        return _diffusers_memory_modes()
    if key == "stable-diffusion.cpp":
        return _sdcpp_memory_modes()
    return None


def mlx_gen_memory_measurements(repo_id: str, bits: Optional[int]) -> Dict[str, Dict[str, Any]]:
    """Peak memory per memory_mode for one prepared package, from mlx-gen's capability rows.

    Keyed by memory_mode. Each value holds `peak_bytes` (whole-process footprint), the run's
    `width`, `height`, `frames` and `steps`, and `smallest_tested_machine_gb` when mlx-gen
    publishes the smallest unified memory that mode completed within. The smallest measured
    canvas is reported.
    Empty when mlx-gen is absent or publishes no measurement for the model."""

    rows = _mlx_gen_rows(str(repo_id))
    runs: List[Dict[str, Any]] = []
    for row in rows:
        for run in row.get("measured_runs") or []:
            if isinstance(run, dict) and run not in runs:
                runs.append(run)
    if not runs:
        return {}
    width = int(bits) if bits in (4, 8) else None
    machines = _mlx_gen_smallest_tested_machines(str(repo_id))
    out: Dict[str, Dict[str, Any]] = {}
    for mode, (low_ram, stream_blocks) in _MLX_GEN_MODE_FLAGS.items():
        matching = [
            run
            for run in runs
            if run.get("outcome") == "completed"
            and run.get("quantize") == width
            and bool(run.get("low_ram")) == low_ram
            and bool(run.get("stream_blocks")) == stream_blocks
            and isinstance(run.get("footprint_bytes"), int)
        ]
        if not matching:
            continue
        run = min(matching, key=lambda r: (int(r.get("width") or 0) * int(r.get("height") or 0), r["footprint_bytes"]))
        entry: Dict[str, Any] = {
            "peak_bytes": int(run["footprint_bytes"]),
            "width": run.get("width"),
            "height": run.get("height"),
            "frames": run.get("frames"),
            "steps": run.get("steps"),
        }
        machine = machines.get(width, {}).get(_MLX_GEN_MODE_NAMES[mode]) if machines else None
        if isinstance(machine, int):
            entry["smallest_tested_machine_gb"] = machine
        out[mode] = entry
    return out


def describe_measurement(entry: Dict[str, Any]) -> str:
    text = (
        f"peak {format_bytes(entry.get('peak_bytes'))} at {entry.get('width')}x{entry.get('height')}, "
        f"{entry.get('frames')} frames, {entry.get('steps')} steps"
    )
    if entry.get("smallest_tested_machine_gb"):
        text += f"; smallest tested memory: {entry['smallest_tested_machine_gb']} GB"
    return text


def preset_status(preset: VisionModelDownloadPreset) -> Dict[str, Any]:
    """Cache state, memory modes and MLX-Gen measurements of one preset on this host."""

    installed = installed_snapshot(preset.repo_id)
    modes = engine_memory_modes(preset.engine, preset.repo_id)
    status: Dict[str, Any] = {
        "installed": installed is not None,
        "installed_bytes": installed[1] if installed else None,
        "snapshot_dir": str(installed[0]) if installed else None,
        "partial": installed is None and is_partially_downloaded(preset.repo_id),
        "memory_modes": list(modes) if modes is not None else None,
    }
    # mlx-gen measures its prepared packages, by width; a source snapshot is not one of them.
    if str(preset.source) == "abstractframework-mlx-gen" and modes is not None:
        measurements = mlx_gen_memory_measurements(preset.repo_id, preset.quantization_bits)
        if measurements:
            status["memory_measurements"] = measurements
    return status


def install_label(status: Dict[str, Any]) -> str:
    if status.get("installed"):
        return f"installed ({format_bytes(status.get('installed_bytes'))})"
    if status.get("partial"):
        return "partial download"
    return "not installed"


def memory_modes_label(status: Dict[str, Any]) -> str:
    modes = status.get("memory_modes")
    return ",".join(modes) if modes else NOT_AVAILABLE


@lru_cache(maxsize=None)
def _mlx_gen_memory_modes(repo_id: str) -> Optional[Tuple[str, ...]]:
    backend = _mlx_gen_backend(repo_id)
    if backend is None:
        return None
    try:
        return tuple(backend.supported_memory_modes())
    except Exception:
        return None


@lru_cache(maxsize=None)
def _mlx_gen_rows(repo_id: str) -> Tuple[Dict[str, Any], ...]:
    backend = _mlx_gen_backend(repo_id)
    if backend is None:
        return ()
    try:
        return tuple(row for row in backend.describe_routes(model=repo_id) if isinstance(row, dict))
    except Exception:
        return ()


def _mlx_gen_backend(repo_id: str) -> Any:
    if local_model_profile() != "apple-silicon" or importlib.util.find_spec("mflux") is None:
        return None
    try:
        from .backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        return MFluxVisionBackend(config=MFluxBackendConfig(model=repo_id))
    except Exception:
        return None


def _mlx_gen_smallest_tested_machines(repo_id: str) -> Dict[Optional[int], Dict[str, int]]:
    # mlx-gen publishes this table for MiniMax-H3 only, next to its measured runs.
    try:
        from .backends.mflux import _MFLUX_MODELS, _infer_base_model

        model_def = _MFLUX_MODELS.get(str(_infer_base_model(repo_id) or ""))
        if model_def is None or model_def.family != "minimax-h3":
            return {}
        table = getattr(importlib.import_module("mflux.task_inference"), "MINIMAX_H3_SMALLEST_TESTED_MACHINE", None)
    except Exception:
        return {}
    return table if isinstance(table, dict) else {}


@lru_cache(maxsize=1)
def _diffusers_memory_modes() -> Optional[Tuple[str, ...]]:
    if importlib.util.find_spec("diffusers") is None or importlib.util.find_spec("torch") is None:
        return None
    try:
        from .backends.huggingface_diffusers import (
            HuggingFaceDiffusersBackendConfig,
            HuggingFaceDiffusersVisionBackend,
        )

        backend = HuggingFaceDiffusersVisionBackend(
            config=HuggingFaceDiffusersBackendConfig(
                model_id="catalog",
                device=str(os.environ.get("ABSTRACTVISION_DIFFUSERS_DEVICE") or "auto"),
            )
        )
        return tuple(backend.supported_memory_modes())
    except Exception:
        return None


@lru_cache(maxsize=1)
def _sdcpp_memory_modes() -> Optional[Tuple[str, ...]]:
    binary = str(os.environ.get("ABSTRACTVISION_SDCPP_BIN") or "sd-cli")
    if shutil.which(binary) is None and importlib.util.find_spec("stable_diffusion_cpp") is None:
        return None
    try:
        from .backends.stable_diffusion_cpp import (
            StableDiffusionCppBackendConfig,
            StableDiffusionCppVisionBackend,
        )

        backend = StableDiffusionCppVisionBackend(config=StableDiffusionCppBackendConfig(sd_cli_path=binary))
        return tuple(backend.supported_memory_modes())
    except Exception:
        return None
