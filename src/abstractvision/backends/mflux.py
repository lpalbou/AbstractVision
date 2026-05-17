from __future__ import annotations

import random
import tempfile
import importlib.util
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..errors import CapabilityNotSupportedError, OptionalDependencyMissingError
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
    resolve_hf_repo_snapshot,
)
from ..types import (
    GeneratedAsset,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageToVideoRequest,
    MultiAngleRequest,
    ProviderModelInfo,
    VideoGenerationRequest,
    VisionBackendCapabilities,
)
from .base_backend import VisionBackend


@dataclass(frozen=True)
class _MFluxModelDef:
    key: str
    config_method: str
    family: str
    default_steps: int
    default_guidance: Optional[float]
    supports_negative_prompt: bool = False
    supports_guidance_override: bool = True


_MFLUX_MODELS: Dict[str, _MFluxModelDef] = {
    "flux2-klein-4b": _MFluxModelDef(
        key="flux2-klein-4b",
        config_method="flux2_klein_4b",
        family="flux2",
        default_steps=4,
        default_guidance=1.0,
        supports_negative_prompt=False,
        supports_guidance_override=False,
    ),
    "flux2-klein-9b": _MFluxModelDef(
        key="flux2-klein-9b",
        config_method="flux2_klein_9b",
        family="flux2",
        default_steps=4,
        default_guidance=1.0,
        supports_negative_prompt=False,
        supports_guidance_override=False,
    ),
    "z-image-turbo": _MFluxModelDef(
        key="z-image-turbo",
        config_method="z_image_turbo",
        family="z-image",
        default_steps=9,
        default_guidance=None,
        supports_negative_prompt=True,
        supports_guidance_override=True,
    ),
    "qwen-image": _MFluxModelDef(
        key="qwen-image",
        config_method="qwen_image",
        family="qwen",
        default_steps=4,
        default_guidance=4.0,
        supports_negative_prompt=True,
        supports_guidance_override=True,
    ),
}


_KNOWN_MODEL_ALIASES: Dict[str, str] = {
    "black-forest-labs/flux.2-klein-4b": "flux2-klein-4b",
    "aitrader/flux2-klein-4b-mlx-8bit": "flux2-klein-4b",
    "moxin-org/flux.2-klein-4b-8bit-mlx": "flux2-klein-4b",
    "runpod/flux.2-klein-4b-mflux-4bit": "flux2-klein-4b",
    "flux2-klein-4b": "flux2-klein-4b",
    "flux-klein-4b": "flux2-klein-4b",
    "klein-4b": "flux2-klein-4b",
    "black-forest-labs/flux.2-klein-9b": "flux2-klein-9b",
    "deepsweet/flux.2-klein-9b-mlx-q8": "flux2-klein-9b",
    "deepsweet/flux.2-klein-9b-mlx-q4": "flux2-klein-9b",
    "themindstudio/flux2-klein-9b-mlx-4bit": "flux2-klein-9b",
    "flux2-klein-9b": "flux2-klein-9b",
    "flux-klein-9b": "flux2-klein-9b",
    "klein-9b": "flux2-klein-9b",
    "tongyi-mai/z-image-turbo": "z-image-turbo",
    "carsenk/z-image-turbo-mflux-8bit": "z-image-turbo",
    "andrevp/z-image-turbo-mlx": "z-image-turbo",
    "andrevp/z-image-turbo-mlx-8bit": "z-image-turbo",
    "illusion615/z-image-turbo-mlx": "z-image-turbo",
    "z-image-turbo": "z-image-turbo",
    "zimage-turbo": "z-image-turbo",
    "qwen/qwen-image": "qwen-image",
    "qwen/qwen-image-2512": "qwen-image",
    "mlx-community/qwen-image-2512-8bit": "qwen-image",
    "mlx-community/qwen-image-2512-8bit-mlx": "qwen-image",
    "qwen-image": "qwen-image",
    "qwen-image-2512": "qwen-image",
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
    """Config for the optional MFLUX backend.

    MFLUX is Apple/MLX-specific and intentionally optional. The backend uses
    MFLUX's Python API in-process and expects local 8-bit MFLUX-compatible
    model directories by default.
    """

    model: Optional[str] = None
    base_model: Optional[str] = None
    model_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    quantize: Optional[int] = None
    lora_paths: Sequence[str] = field(default_factory=tuple)
    lora_scales: Sequence[float] = field(default_factory=tuple)
    allow_download: bool = False
    default_width: int = 1024
    default_height: int = 1024


def _lazy_import_mflux() -> Tuple[Any, Any, Any]:
    try:
        from mflux.models.common.config import ModelConfig  # type: ignore
        from mflux.models.flux2.variants import Flux2Klein  # type: ignore
        from mflux.models.z_image import ZImage  # type: ignore
    except Exception as e:
        raise OptionalDependencyMissingError(
            "MFLUX backend requires the optional MFLUX runtime. "
            'Install it with `pip install "abstractvision[mflux]"` or '
            '`pip install "abstractvision[all-apple]"` on Apple Silicon.'
        ) from e
    return ModelConfig, Flux2Klein, ZImage


def _lazy_import_mflux_qwen() -> Any:
    try:
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage  # type: ignore
    except Exception as e:
        raise OptionalDependencyMissingError(
            "MFLUX Qwen backend requires a recent MFLUX runtime. "
            'Install/upgrade it with `pip install "abstractvision[mflux]"` (Apple Silicon only).'
        ) from e
    return QwenImage


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


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
    return any(token in s for token in ("mflux", "8bit", "q8", "4bit", "q4", "quant"))


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
        ("configured model dir" if model_dir else "legacy model dir", Path(model_dir).expanduser() if model_dir else default_legacy_model_root())
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
    if s == "mlx-community/qwen-image-2512-8bit":
        return 0
    if s.endswith("-mlx-8bit") or "mflux-8bit" in s or "-mlx-q8" in s or "-8bit-mlx" in s:
        return 1
    if "mlx-community/" in s:
        return 2
    if "mflux" in s or "mlx" in s:
        return 3
    if "q4" in s or "4bit" in s:
        return 4
    return 5


def _candidate_repo_ids_for_preset(preset: Any) -> Tuple[str, ...]:
    out: List[str] = []
    for value in (preset.repo_id, *(preset.aliases or ())):
        text = str(value or "").strip()
        if not looks_like_hf_repo_id(text):
            continue
        if text not in out:
            out.append(text)
    return tuple(out)


def _discover_cached_legacy_mflux_models(model_dir: Optional[str]) -> Dict[str, _DiscoveredMFluxModel]:
    out: Dict[str, _DiscoveredMFluxModel] = {}
    for root_label, root in _mflux_model_roots(model_dir):
        try:
            entries = list(root.iterdir())
        except Exception:
            continue
        for entry in entries:
            if not entry.is_dir() or _is_incompatible_model_tree(entry) or _is_partial_model_tree(entry):
                continue
            if not _has_model_files(entry):
                continue
            base = _infer_base_model(entry.name)
            if base not in _MFLUX_MODELS:
                continue
            if entry.name != str(_preset_for(base).local_dir_name if _preset_for(base) else "") and not _looks_like_mflux_packaged_repo(entry.name):
                continue
            source_label = _local_source_label(root_label)
            out.setdefault(
                base,
                _DiscoveredMFluxModel(
                    key=base,
                    snapshot_dir=entry,
                    repo_id=None,
                    source_label=source_label,
                    source_detail=f"{source_label} ({entry})",
                ),
            )
    return out


def _discover_cached_hf_mflux_models(cache_dir: Optional[str]) -> Dict[str, _DiscoveredMFluxModel]:
    out: Dict[str, _DiscoveredMFluxModel] = {}
    for label, root in hf_cache_roots(cache_dir=cache_dir, extra_roots=framework_hf_cache_roots()):
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
            snap = resolve_hf_repo_snapshot(
                repo_id,
                cache_dir=str(root),
                require_weight_files=True,
                extra_roots=[(label, root)],
            )
            if snap is None or not _has_model_files(snap) or _is_partial_model_tree(snap) or _is_incompatible_model_tree(snap):
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
    extra_cache_roots = framework_hf_cache_roots()
    for preset in model_presets(target="mlx", engine="mflux", include_non_8bit=False):
        for repo_id in _candidate_repo_ids_for_preset(preset):
            snap = resolve_hf_repo_snapshot(
                repo_id,
                cache_dir=cache_root,
                require_weight_files=True,
                extra_roots=extra_cache_roots,
            )
            if snap is None or not _has_model_files(snap) or _is_partial_model_tree(snap) or _is_incompatible_model_tree(snap):
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
    extra_cache_roots = framework_hf_cache_roots()

    def add(key: str, detail: str) -> None:
        text = str(detail or "").strip()
        if not text:
            return
        bucket = out.setdefault(key, [])
        if text not in bucket:
            bucket.append(text)

    for preset in model_presets(target="mlx", engine="mflux", include_non_8bit=False):
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
            for label, root in hf_cache_roots(cache_dir=cache_dir, extra_roots=extra_cache_roots):
                repo_dir = root / f"models--{repo_id.replace('/', '--')}"
                lock_dir = root / ".locks" / f"models--{repo_id.replace('/', '--')}"
                if repo_dir.exists() or lock_dir.exists():
                    if _is_incompatible_model_tree(repo_dir):
                        add(preset.key, f"incompatible HF cache: {label} ({repo_id})")
                    else:
                        add(preset.key, f"incomplete HF cache: {label} ({repo_id})")
        for label, root in hf_cache_roots(cache_dir=cache_dir, extra_roots=extra_cache_roots):
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
        if "qwen" in s and "image" in s:
            return "qwen-image"
        if "z-image" in s or "zimage" in s:
            return "z-image-turbo"
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
        return find_model_preset(s, target="mlx", engine="mflux", require_8bit=True)
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
        if preset.engine != "mflux":
            return preset
    return None


def _preset_snapshot_dir(preset: Any, model_dir: Optional[str], cache_dir: Optional[str]) -> Optional[Path]:
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
    """Local Apple Silicon backend for MFLUX-compatible MLX image models."""

    def __init__(self, *, config: MFluxBackendConfig):
        self._cfg = config
        self._model: Any = None
        self._model_key: Optional[Tuple[Any, ...]] = None
        self._resolved_model_path: Optional[str] = None
        self._resolved_base_model: Optional[str] = None
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
        self._resolved_model_path = None
        self._resolved_base_model = None
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
        self._run_on_runtime_thread(self._ensure_model_impl)

    def get_capabilities(self) -> VisionBackendCapabilities:
        return VisionBackendCapabilities(
            supported_tasks=["text_to_image", "image_to_image"],
            supports_mask=False,
        )

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        # Listing should reflect available local weights even when the optional
        # `mflux` runtime is not installed. Generation will still error until
        # the runtime is present, but catalogs can surface what is already
        # downloaded.
        if task and str(task).strip() not in {"text_to_image", "image_to_image"}:
            return ()
        out = []
        discovered = discover_cached_mflux_models(
            model_dir=self._cfg.model_dir,
            cache_dir=self._cfg.cache_dir,
        )
        for preset in model_presets(target="mlx", engine="mflux", include_non_8bit=False):
            discovered_model = discovered.get(preset.key)
            if discovered_model is None:
                continue
            base_model = _infer_base_model(preset.key, preset.repo_id, preset.upstream_repo_id)
            model_def = _MFLUX_MODELS.get(str(base_model or ""))
            parameter_metadata = _mflux_parameter_metadata(model_def) if model_def is not None else {}
            out.append(
                ProviderModelInfo(
                    id=preset.key,
                    object="model",
                    owned_by="mflux",
                    capabilities=("text_to_image", "image_to_image"),
                    raw={
                        "provider": "mflux",
                        "model": preset.key,
                        "routed_model": f"mflux/{preset.key}",
                        "engine": "mflux",
                        "target": preset.target,
                        "snapshot_dir": str(discovered_model.snapshot_dir),
                        "repo_id": discovered_model.repo_id or preset.repo_id,
                        "upstream_repo_id": preset.upstream_repo_id,
                        "quantization_bits": preset.quantization_bits,
                        "cache_source": discovered_model.source_label,
                        "cache_source_detail": discovered_model.source_detail,
                        **parameter_metadata,
                    },
                )
            )
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
            expanded = Path(configured_model).expanduser()
            if expanded.exists():
                base = configured_base or _infer_base_model(configured_model, expanded.name)
                if not base:
                    raise OptionalDependencyMissingError(
                        "Could not infer MFLUX base model from local path. "
                        "Set vision_mflux_base_model / ABSTRACTVISION_MFLUX_BASE_MODEL "
                        "to flux2-klein-4b, flux2-klein-9b, or z-image-turbo."
                    )
                return str(expanded), base

            preset = _preset_for(configured_model)
            if preset is not None:
                discovered_model = discovered.get(preset.key)
                snapshot_dir = discovered_model.snapshot_dir if discovered_model is not None else _preset_snapshot_dir(
                    preset,
                    self._cfg.model_dir,
                    cache_root,
                )
                if snapshot_dir is not None and _has_model_files(snapshot_dir):
                    return str(snapshot_dir), configured_base or preset.key
                if not self._cfg.allow_download:
                    raise OptionalDependencyMissingError(
                        f"MFLUX model preset {preset.key!r} is not available in the Hugging Face cache. "
                        f"Run: abstractvision download-model {preset.key} --provider mflux"
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
                return str(downloaded), configured_base or preset.key

            if _looks_like_path(configured_model):
                raise OptionalDependencyMissingError(f"MFLUX model path does not exist: {configured_model}")
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
                            "Could not infer MFLUX base model. Set vision_mflux_base_model / "
                            "ABSTRACTVISION_MFLUX_BASE_MODEL."
                        )
                    return str(cached), base
                if not self._cfg.allow_download:
                    raise OptionalDependencyMissingError(
                        f"MFLUX model repo {configured_model!r} is not cached locally. "
                        "Pre-download it with `abstractvision download-model <org/name>` "
                        "or set ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD=1 to permit downloads."
                    )
                base = configured_base or _infer_base_model(configured_model)
                if not base:
                    raise OptionalDependencyMissingError(
                        "Could not infer MFLUX base model. Set vision_mflux_base_model / "
                        "ABSTRACTVISION_MFLUX_BASE_MODEL."
                    )
                try:
                    downloaded = download_hf_repo_snapshot(
                        configured_model,
                        cache_dir=cache_root,
                    )
                except RuntimeError as e:
                    raise OptionalDependencyMissingError(str(e)) from e
                return str(downloaded), base

            if not self._cfg.allow_download:
                other = _first_other_engine_preset(configured_model)
                if other is not None:
                    raise OptionalDependencyMissingError(
                        f"Model {configured_model!r} maps to a curated preset for engine {other.engine!r} "
                        f"(target={other.target!r}, repo={other.repo_id!r}), not MFLUX. "
                        "Use `--provider diffusers` or `--provider sdcpp` as appropriate, or pass an MFLUX preset key "
                        "(flux2-klein-4b, flux2-klein-9b, z-image-turbo, qwen-image) / local path / org/name repo id."
                    )
                raise OptionalDependencyMissingError(
                    f"MFLUX model {configured_model!r} is not a known downloaded preset. "
                    "Use a local model path, a known preset key, a Hugging Face repo id (org/name) already cached "
                    "locally, or set ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD=1 to permit downloads."
                )
            base = configured_base or _infer_base_model(configured_model)
            if not base:
                raise OptionalDependencyMissingError(
                    "Could not infer MFLUX base model. Set vision_mflux_base_model / "
                    "ABSTRACTVISION_MFLUX_BASE_MODEL."
                )
            raise OptionalDependencyMissingError(
                f"MFLUX model {configured_model!r} is not a known cached preset, path, or Hugging Face repo id."
            )

        for key in ("flux2-klein-4b", "flux2-klein-9b", "z-image-turbo", "qwen-image"):
            discovered_model = discovered.get(key)
            if discovered_model is not None:
                return str(discovered_model.snapshot_dir), configured_base or key

        raise OptionalDependencyMissingError(
            "MFLUX backend is not configured and no downloaded MFLUX preset was found. "
            "Set vision_mflux_model / ABSTRACTVISION_MFLUX_MODEL or run "
            "`abstractvision download-model flux2-klein-4b --provider mflux`."
        )

    def _ensure_model_impl(self) -> Tuple[Any, _MFluxModelDef]:
        model_path, base_model = self._resolve_model()
        if base_model not in _MFLUX_MODELS:
            raise OptionalDependencyMissingError(
                f"Unsupported MFLUX base model {base_model!r}. "
                f"Supported: {', '.join(sorted(_MFLUX_MODELS))}"
            )
        model_def = _MFLUX_MODELS[base_model]
        key = (
            model_path,
            base_model,
            self._cfg.quantize,
            tuple(self._cfg.lora_paths or ()),
            tuple(self._cfg.lora_scales or ()),
        )
        if self._model is not None and self._model_key == key:
            return self._model, model_def

        ModelConfig, Flux2Klein, ZImage = _lazy_import_mflux()
        model_config = getattr(ModelConfig, model_def.config_method)()
        if model_def.family == "flux2":
            cls = Flux2Klein
        elif model_def.family == "z-image":
            cls = ZImage
        elif model_def.family == "qwen":
            cls = _lazy_import_mflux_qwen()
        else:
            raise OptionalDependencyMissingError(f"Unsupported MFLUX model family {model_def.family!r}.")
        kwargs: Dict[str, Any] = {
            "model_config": model_config,
            "model_path": model_path,
            "quantize": self._cfg.quantize,
        }
        if self._cfg.lora_paths:
            kwargs["lora_paths"] = list(self._cfg.lora_paths)
        if self._cfg.lora_scales:
            kwargs["lora_scales"] = [float(x) for x in self._cfg.lora_scales]
        self._model = cls(**kwargs)
        self._model_key = key
        self._resolved_model_path = model_path
        self._resolved_base_model = base_model
        return self._model, model_def

    def _resolved_model_def(self) -> _MFluxModelDef:
        _model_path, base_model = self._resolve_model()
        if base_model not in _MFLUX_MODELS:
            raise OptionalDependencyMissingError(
                f"Unsupported MFLUX base model {base_model!r}. "
                f"Supported: {', '.join(sorted(_MFLUX_MODELS))}"
            )
        return _MFLUX_MODELS[base_model]

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
        if not model_def.supports_guidance_override and model_def.default_guidance is not None:
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
        if not model_def.supports_guidance_override and model_def.default_guidance is not None:
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

    def _generate_impl(
        self,
        request: ImageGenerationRequest,
        *,
        image_path: Optional[Path] = None,
        image_strength: Optional[float] = None,
    ) -> GeneratedAsset:
        request = self.normalize_image_generation_request(request)
        model, model_def = self._ensure_model_impl()
        extra = dict(request.extra or {})
        seed = int(request.seed) if request.seed is not None else random.randint(0, 1_000_000_000)
        steps = int(request.steps) if request.steps is not None else model_def.default_steps
        width = int(request.width) if request.width is not None else int(self._cfg.default_width)
        height = int(request.height) if request.height is not None else int(self._cfg.default_height)
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
            kwargs["image_strength"] = float(image_strength if image_strength is not None else extra.pop("image_strength", 0.4))

        generated = model.generate_image(**kwargs)
        pil_image = getattr(generated, "image", generated)
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        data = buf.getvalue()
        return GeneratedAsset(
            media_type="image",
            data=data,
            mime_type="image/png",
            metadata={
                "source": "mflux",
                "engine": "mflux",
                "model": self._resolved_model_path,
                "base_model": self._resolved_base_model,
                "seed": seed,
                "steps": steps,
                "width": width,
                "height": height,
            },
        )

    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
        return self._run_on_runtime_thread(self._generate_impl, request)

    def _edit_image_impl(self, request: ImageEditRequest) -> GeneratedAsset:
        if request.mask is not None:
            raise CapabilityNotSupportedError("MFLUX backend does not support mask-based image editing.")
        with tempfile.TemporaryDirectory(prefix="abstractvision-mflux-") as td:
            image_path = Path(td) / "input.png"
            image_path.write_bytes(bytes(request.image))
            image_strength = None
            if isinstance(request.extra, dict) and request.extra.get("image_strength") is not None:
                image_strength = float(request.extra["image_strength"])
            gen_req = ImageGenerationRequest(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=None,
                height=None,
                seed=request.seed,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                extra=dict(request.extra or {}),
            )
            return self._generate_impl(gen_req, image_path=image_path, image_strength=image_strength)

    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
        return self._run_on_runtime_thread(self._edit_image_impl, request)

    def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]:
        raise CapabilityNotSupportedError("MFLUX backend does not implement multi-view generation.")

    def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
        raise CapabilityNotSupportedError("MFLUX backend does not implement text_to_video.")

    def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
        raise CapabilityNotSupportedError("MFLUX backend does not implement image_to_video.")
