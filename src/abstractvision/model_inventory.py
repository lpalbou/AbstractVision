"""What a listing shows about a download preset on this host, read at run time.

`preset_status` answers three questions for one preset without touching the network or loading
weights: what the local Hugging Face cache holds of it (`local_cache_state`), which `memory_mode`
values does the engine that serves it honour on this host, and, for MLX-Gen models whose capability
rows publish measured runs, what peak memory each mode reached and the smallest Mac it was tested on.
Every memory figure comes from the installed mlx-gen; nothing is copied into AbstractVision.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .model_cache import (
    _iter_hf_repo_snapshots,
    framework_hf_cache_roots,
    hf_cache_roots,
    hf_snapshot_has_incomplete_downloads,
    hf_snapshot_has_weight_files,
    hf_snapshot_missing_indexed_weight_files,
)
from .model_downloads import (
    VisionModelDownloadPreset,
    _model_index_components,
    _preset_download_patterns,
    _snapshot_requirements_for_preset,
    format_bytes,
    local_model_profile,
)
from .types import MEMORY_MODE_DEFAULT, MEMORY_MODE_LOW, MEMORY_MODE_LOWEST

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


@dataclass(frozen=True)
class LocalCacheState:
    """What a local look at the Hugging Face cache finds for one preset, without the Hub listing.

    `gaps` are what the look can see is missing: partial `.incomplete` files, snapshot links whose
    content is gone, weight shards an index names, pipeline components `model_index.json` names whose
    folder the preset's patterns name (`components_named_by_patterns`), or no weight file at all. No gap does not prove the snapshot holds every file the package selects:
    only the Hub listing (`download`, `download --dry-run`) can tell that."""

    repo_id: str
    snapshot_dir: Optional[Path]
    files: int
    bytes: int
    gaps: Tuple[str, ...]

    @property
    def state(self) -> str:
        """`cached` (a snapshot without a visible gap), `incomplete`, or `absent`."""

        if self.snapshot_dir is not None and not self.gaps:
            return "cached"
        return "incomplete" if self.gaps else "absent"

    def describe(self) -> str:
        if self.state == "absent":
            return "The cache holds nothing of it."
        where = f" at {self.snapshot_dir}" if self.snapshot_dir is not None else ""
        if self.state == "cached":
            return f"The cache holds a snapshot of it{where} ({self.files} files, {format_bytes(self.bytes)})."
        return f"The cache holds an incomplete snapshot of it{where}: {'; '.join(self.gaps)}."


def local_cache_state(
    preset: VisionModelDownloadPreset, *, extra_roots: Optional[Sequence[Any]] = None
) -> LocalCacheState:
    """The one local look at a preset's cached snapshot (listings, and `download` without the Hub).

    `extra_roots` are cache roots searched besides the configured one (default: the framework's
    runtime caches)."""

    roots = framework_hf_cache_roots() if extra_roots is None else list(extra_roots)
    repo_id = str(preset.repo_id)
    snapshots = _iter_hf_repo_snapshots(repo_id, extra_roots=roots)
    snapshot = snapshots[0][1] if snapshots else None
    if snapshot is None:
        leftovers = _repo_dirs_without_snapshot(repo_id, roots)
        gaps = (f"files of an unfinished download but no snapshot ({leftovers[0]})",) if leftovers else ()
        return LocalCacheState(repo_id=repo_id, snapshot_dir=None, files=0, bytes=0, gaps=gaps)
    files, size = directory_files_and_bytes(snapshot)
    return LocalCacheState(
        repo_id=repo_id, snapshot_dir=snapshot, files=files, bytes=size, gaps=_snapshot_gaps(preset, snapshot)
    )


def preset_status(preset: VisionModelDownloadPreset) -> Dict[str, Any]:
    """Cache state, memory modes and MLX-Gen measurements of one preset on this host."""

    cache = local_cache_state(preset)
    modes = _engine_memory_modes(preset.engine, preset.repo_id)
    status: Dict[str, Any] = {
        "cache": cache.state,
        "cache_bytes": cache.bytes if cache.snapshot_dir is not None else None,
        "cache_gaps": list(cache.gaps),
        "snapshot_dir": str(cache.snapshot_dir) if cache.snapshot_dir is not None else None,
        "memory_modes": list(modes) if modes is not None else None,
    }
    # mlx-gen measures its prepared packages, by width; a source snapshot is not one of them.
    if str(preset.source) == "abstractframework-mlx-gen" and modes is not None:
        measurements = _mlx_gen_memory_measurements(preset.repo_id, preset.quantization_bits)
        if measurements:
            status["memory_measurements"] = measurements
    return status


def _snapshot_gaps(preset: VisionModelDownloadPreset, snapshot: Path) -> Tuple[str, ...]:
    gaps: List[str] = []
    if hf_snapshot_has_incomplete_downloads(snapshot):
        gaps.append("partial files of an interrupted download (.incomplete)")
    broken = _broken_links(snapshot)
    if broken:
        gaps.append(f"{len(broken)} linked files without content (for example {broken[0]})")
    shards = hf_snapshot_missing_indexed_weight_files(snapshot)
    if shards:
        gaps.append(f"{len(shards)} weight files an index names are missing (for example {shards[0]})")
    missing_components = [
        name for name in components_named_by_patterns(preset, snapshot) if not (snapshot / name).is_dir()
    ]
    if missing_components:
        gaps.append(f"model_index.json names {', '.join(missing_components)}, not in the snapshot")
    required_files, require_weight_files = _snapshot_requirements_for_preset(preset)
    for name in required_files:
        if not (snapshot / name).is_file():
            gaps.append(f"no {name}")
    if require_weight_files and not hf_snapshot_has_weight_files(snapshot):
        gaps.append("no weight file")
    return tuple(gaps)


def components_named_by_patterns(preset: VisionModelDownloadPreset, snapshot: Path) -> List[str]:
    """The pipeline components the snapshot's `model_index.json` names whose folder the preset's own
    patterns name (`transformer_2/*`). A component that a broad pattern (`*.json`) would only match is
    left out: a `model_index.json` may name a component the repository does not hold (HiDream-I1's
    `text_encoder_4`, loaded from another repository)."""

    index = snapshot / "model_index.json"
    if not index.is_file():
        return []
    try:
        components = _model_index_components(json.loads(index.read_text(encoding="utf-8")))
    except Exception:
        return []
    allow, _ignore = _preset_download_patterns(preset)
    folders = {str(p).split("/", 1)[0] for p in (allow or ()) if "/" in str(p)}
    return [name for name in components if name in folders]


def _broken_links(snapshot: Path) -> List[str]:
    broken: List[str] = []
    for root, dirs, files in os.walk(snapshot):
        dirs[:] = [d for d in dirs if d != ".cache"]
        for name in files:
            path = Path(root, name)
            if path.is_symlink() and not path.exists():
                broken.append(str(path.relative_to(snapshot)))
    return sorted(broken)


def _repo_dirs_without_snapshot(repo_id: str, roots: Sequence[Any]) -> List[str]:
    name = "models--" + repo_id.replace("/", "--")
    return [str(root / name) for _label, root in hf_cache_roots(extra_roots=roots) if (root / name).exists()]


def directory_files_and_bytes(directory: Path) -> Tuple[int, int]:
    """Files a directory (a snapshot, or a legacy tree) holds and their bytes on disk (symlinks resolved,
    each blob counted once)."""

    seen: set[str] = set()
    total = 0
    for root, dirs, files in os.walk(directory):
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
    return len(seen), total


def _engine_memory_modes(engine: str, repo_id: str) -> Optional[Tuple[str, ...]]:
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


def _mlx_gen_memory_measurements(repo_id: str, bits: Optional[int]) -> Dict[str, Dict[str, Any]]:
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
