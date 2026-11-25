"""What a listing shows about a download preset on this host, read at run time.

`preset_status` answers three questions for one preset without touching the network or loading
weights: what the local Hugging Face cache holds of it (`local_cache_state`), which `memory_mode`
values does the engine that serves it honour on this host, and, for MLX-Gen packages whose capability
rows publish measured runs, what peak memory each mode reached (`memory_peaks`) and the smallest Mac
it was tested on. Every memory figure comes from the installed mlx-gen; nothing is copied into
AbstractVision.
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
    framework_hf_cache_roots,
    hf_cache_roots,
    hf_snapshot_has_incomplete_downloads,
    hf_snapshot_has_weight_files,
    hf_snapshot_missing_indexed_weight_files,
    iter_hf_repo_snapshots,
)
from .model_downloads import (
    VisionModelDownloadPreset,
    catalog_download_preset,
    format_bytes,
    local_model_profile,
    model_index_components,
    preset_download_patterns,
    snapshot_requirements_for_preset,
)


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
    snapshots = iter_hf_repo_snapshots(repo_id, extra_roots=roots)
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
    peaks = memory_peaks(preset.repo_id) if modes is not None else []
    if peaks:
        status["memory_peaks"] = peaks
        smallest = _mlx_gen_smallest_tested_memory_gb(preset.repo_id)
        if smallest:
            status["smallest_tested_memory_gb"] = smallest
    return status


def memory_peaks(repo_id: str, base_model: Optional[str] = None) -> List[Dict[str, Any]]:
    """MLX-Gen's measured peak memory for one catalog package, per task and memory_mode.

    Each row is `{task, memory_mode, width, height, frames, steps, peak_bytes,
    machine_total_ram_bytes}`: a completed run from the `measured_runs` of the engine's capability
    rows for the package (for the `base_model` schedule when given), at the package's stored width.
    `peak_bytes` is the whole-process footprint; `machine_total_ram_bytes` the memory of the Mac it
    ran on. Empty for a repository that is not an MLX-Gen package of the catalog, and when mlx-gen
    is not installed or publishes nothing. No model is built and no weight is read. Raises
    ValueError when `base_model` is not a schedule of the package."""

    preset = catalog_download_preset(str(repo_id), target="mlx", engine="mlx-gen")
    # mlx-gen measures its prepared packages, by width; a source snapshot is not one of them.
    if preset is None or str(preset.source) != "abstractframework-mlx-gen":
        return []
    from .backends.mflux import (
        mlx_gen_memory_mode_of_run,
        mlx_gen_package_schedule,
        mlx_gen_route_task,
    )

    schedule = mlx_gen_package_schedule(preset, base_model) if base_model is not None else None
    width = _engine_quantize(preset.quantization_bits)
    peaks: List[Dict[str, Any]] = []
    for row in _mlx_gen_rows(preset.repo_id, schedule):
        for run in row.get("measured_runs") or []:
            if not isinstance(run, dict) or run.get("outcome") != "completed" or run.get("quantize") != width:
                continue
            mode = mlx_gen_memory_mode_of_run(run)
            if mode is None or not isinstance(run.get("footprint_bytes"), int):
                continue
            peaks.append(
                {
                    "task": mlx_gen_route_task(row),
                    "memory_mode": mode,
                    "width": run.get("width"),
                    "height": run.get("height"),
                    "frames": run.get("frames"),
                    "steps": run.get("steps"),
                    "peak_bytes": run["footprint_bytes"],
                    "machine_total_ram_bytes": run.get("machine_total_ram_bytes"),
                }
            )
    return peaks


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
    required_files, require_weight_files = snapshot_requirements_for_preset(preset)
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
        components = model_index_components(json.loads(index.read_text(encoding="utf-8")))
    except Exception:
        return []
    allow, _ignore = preset_download_patterns(preset)
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
def _mlx_gen_rows(repo_id: str, base_model: Optional[str] = None) -> Tuple[Dict[str, Any], ...]:
    # The rows a backend configured with this package (and schedule) plans with.
    backend = _mlx_gen_backend(repo_id, base_model)
    if backend is None:
        return ()
    try:
        return tuple(row for row in backend.describe_routes() if isinstance(row, dict))
    except Exception:
        return ()


def _mlx_gen_backend(repo_id: str, base_model: Optional[str] = None) -> Any:
    if local_model_profile() != "apple-silicon" or importlib.util.find_spec("mflux") is None:
        return None
    try:
        from .backends.mflux import MFluxBackendConfig, MFluxVisionBackend

        return MFluxVisionBackend(config=MFluxBackendConfig(model=repo_id, base_model=base_model))
    except Exception:
        return None


def _engine_quantize(bits: Optional[int]) -> Optional[int]:
    # A measured run's `quantize`: the package's stored width below 16 bits, None for 16-bit weights.
    return int(bits) if bits is not None and int(bits) < 16 else None


def _mlx_gen_smallest_tested_memory_gb(repo_id: str) -> Dict[str, int]:
    """Per memory_mode, the smallest unified memory (GB) mlx-gen completed a run of this package on.

    mlx-gen publishes this for MiniMax-H3 only, as the module constant
    `mflux.task_inference.MINIMAX_H3_SMALLEST_TESTED_MACHINE` (stored width -> its mode name -> GB)
    rather than in the capability rows; this function is the one place that reads it."""

    preset = catalog_download_preset(str(repo_id), target="mlx", engine="mlx-gen")
    if preset is None:
        return {}
    try:
        from .backends.mflux import (
            MINIMAX_H3_MODEL_KEYS,
            MLX_GEN_MEMORY_MODE_FLAGS,
            mlx_gen_base_model,
        )

        if mlx_gen_base_model(preset.repo_id, preset.upstream_repo_id, preset.key) not in MINIMAX_H3_MODEL_KEYS:
            return {}
        table = getattr(importlib.import_module("mflux.task_inference"), "MINIMAX_H3_SMALLEST_TESTED_MACHINE", None)
    except Exception:
        return {}
    by_engine_mode = table.get(_engine_quantize(preset.quantization_bits)) if isinstance(table, dict) else None
    if not isinstance(by_engine_mode, dict):
        return {}
    return {
        mode: by_engine_mode[flags.engine_mode]
        for mode, flags in MLX_GEN_MEMORY_MODE_FLAGS.items()
        if isinstance(by_engine_mode.get(flags.engine_mode), int)
    }


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
