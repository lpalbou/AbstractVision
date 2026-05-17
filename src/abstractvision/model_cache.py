from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

DEFAULT_LEGACY_MODEL_ROOT = Path("~/models")

_LEGACY_MODEL_DIR_ENV_KEYS = ("ABSTRACTVISION_MODEL_DIR", "ABSTRACTVISION_MODELS_DIR")
_LOCAL_MODEL_DIR_ENV_KEYS = (
    *_LEGACY_MODEL_DIR_ENV_KEYS,
    "ABSTRACTCORE_VISION_MODEL_DIR",
    "ABSTRACTCORE_VISION_MODELS_DIR",
)
_HF_CACHE_ENV_KEYS = (
    "ABSTRACTVISION_HF_HUB_CACHE",
    "ABSTRACTCORE_VISION_HF_HUB_CACHE",
    "HF_HUB_CACHE",
    "HF_HUB_CACHE_DIR",
)
_RootEntry = Tuple[str, Path]
_RootSpec = Union[Path, str, Tuple[str, Union[Path, str]]]

_WEIGHT_FILE_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".gguf",
    ".ckpt",
    ".pt",
    ".pth",
    ".onnx",
    ".msgpack",
}


def _expand_path(value: str | Path) -> Path:
    return Path(str(value)).expanduser()


def _repo_dir_name(repo_id: str) -> str:
    return "models--" + str(repo_id).strip().replace("/", "--")


def _dedupe_root_entries(entries: Sequence[_RootEntry]) -> List[_RootEntry]:
    out: List[_RootEntry] = []
    for label, path in entries:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path.expanduser()
        if not resolved.is_dir():
            continue
        duplicate = False
        for _, existing in out:
            try:
                if existing.resolve() == resolved:
                    duplicate = True
                    break
            except Exception:
                if existing == resolved:
                    duplicate = True
                    break
        if not duplicate:
            out.append((label, resolved))
    return out


def framework_candidate_roots() -> List[Path]:
    roots: List[Path] = []
    for candidate in [Path.cwd(), *Path(__file__).resolve().parents]:
        try:
            root = candidate.expanduser().resolve()
        except Exception:
            continue
        if root not in roots:
            roots.append(root)
        if (root / "runtime").is_dir() or (root / "untracked").is_dir():
            parent = root.parent
            if parent not in roots:
                roots.append(parent)
    return roots[:16]


def framework_hf_cache_roots() -> List[_RootEntry]:
    roots: List[_RootEntry] = []
    for root in framework_candidate_roots():
        roots.append(("runtime HF cache", root / "runtime" / "hf-hub"))
        quarantine = root / "runtime" / "model-quarantine"
        try:
            if quarantine.is_dir():
                roots.extend(
                    ("quarantined HF cache", entry / "hf-hub")
                    for entry in quarantine.iterdir()
                    if entry.is_dir()
                )
        except Exception:
            pass
    return _dedupe_root_entries(roots)


def framework_local_model_roots() -> List[_RootEntry]:
    roots: List[_RootEntry] = []
    for key in _LOCAL_MODEL_DIR_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            roots.append((key, _expand_path(value)))
    for root in framework_candidate_roots():
        roots.append(("runtime local models", root / "runtime" / "models" / "abstractvision"))
        roots.append(("untracked local models", root / "untracked" / "models" / "abstractvision"))
        quarantine = root / "runtime" / "model-quarantine"
        try:
            if quarantine.is_dir():
                roots.extend(
                    ("quarantined local models", entry / "models")
                    for entry in quarantine.iterdir()
                    if entry.is_dir()
                )
        except Exception:
            pass
    return _dedupe_root_entries(roots)


def default_legacy_model_root() -> Path:
    """Return the legacy preset root used by older AbstractVision builds."""

    for key in _LEGACY_MODEL_DIR_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return _expand_path(value)
    return DEFAULT_LEGACY_MODEL_ROOT.expanduser()


def default_hf_cache_root() -> Path:
    """Return the primary Hugging Face cache root used by this process."""

    for key in _HF_CACHE_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return _expand_path(value)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return _expand_path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def hf_cache_roots(
    *,
    cache_dir: Optional[str] = None,
    extra_roots: Optional[Sequence[_RootSpec]] = None,
) -> List[Tuple[str, Path]]:
    """Return all cache roots that may contain Hugging Face snapshots.

    The primary cache root is used for downloads and writes; the additional roots
    exist so we can discover caches created by other frontends or older layouts.
    """

    roots: List[Tuple[str, Path]] = []

    def add(label: str, value: Optional[Path | str]) -> None:
        if value is None:
            return
        path = _expand_path(value)
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        for _, existing in roots:
            try:
                if existing.resolve() == resolved:
                    return
            except Exception:
                if existing == path:
                    return
        roots.append((label, path))

    if cache_dir:
        add("configured cache", cache_dir)
    for key in _HF_CACHE_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            add(key, value)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        add("HF_HOME", _expand_path(hf_home) / "hub")
    add("default HF cache", default_hf_cache_root())
    for root in extra_roots or ():
        if isinstance(root, tuple) and len(root) == 2:
            label, value = root
            add(str(label or "extra cache root"), value)
        else:
            add("extra cache root", root)
    return roots


def hf_repo_dir(repo_id: str, *, cache_dir: Optional[str] = None) -> Path:
    """Return the cache directory that stores the given repository snapshots."""

    return hf_cache_roots(cache_dir=cache_dir)[0][1] / _repo_dir_name(repo_id)


def hf_snapshot_has_incomplete_downloads(snapshot_dir: Path | str) -> bool:
    snapshot = _expand_path(snapshot_dir)
    download_dir = snapshot / ".cache" / "huggingface" / "download"
    if not download_dir.is_dir():
        return False
    try:
        for root, _dirs, files in os.walk(download_dir):
            for name in files:
                if str(name).endswith(".incomplete"):
                    return True
    except Exception:
        return False
    return False


def hf_snapshot_has_weight_files(snapshot_dir: Path | str) -> bool:
    snapshot = _expand_path(snapshot_dir)
    if not snapshot.is_dir():
        return False
    try:
        for root, dirs, files in os.walk(snapshot):
            try:
                rel = Path(root).relative_to(snapshot)
            except Exception:
                rel = Path(root)
            if rel.parts[:1] == (".cache",):
                dirs[:] = []
                continue
            for name in files:
                candidate = Path(root) / name
                if Path(name).suffix.lower() in _WEIGHT_FILE_SUFFIXES and candidate.is_file():
                    return True
    except Exception:
        return False
    return False


def hf_snapshot_missing_indexed_weight_files(snapshot_dir: Path | str) -> List[str]:
    snapshot = _expand_path(snapshot_dir)
    if not snapshot.is_dir():
        return []

    missing: List[str] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            rel = path.relative_to(snapshot)
        except Exception:
            rel = path
        text = str(rel)
        if text and text not in seen:
            seen.add(text)
            missing.append(text)

    try:
        for root, dirs, files in os.walk(snapshot):
            try:
                rel = Path(root).relative_to(snapshot)
            except Exception:
                rel = Path(root)
            if rel.parts[:1] == (".cache",):
                dirs[:] = []
                continue
            for name in files:
                if not str(name).endswith(".index.json"):
                    continue
                index_path = Path(root) / name
                if not index_path.is_file():
                    add(index_path)
                    continue
                try:
                    payload = json.loads(index_path.read_text(encoding="utf-8"))
                except Exception:
                    add(index_path)
                    continue
                weight_map = payload.get("weight_map")
                if not isinstance(weight_map, dict):
                    continue
                for rel_name in sorted({str(v).strip() for v in weight_map.values() if str(v).strip()}):
                    target_path = index_path.parent / rel_name
                    if not target_path.is_file():
                        add(target_path)
    except Exception:
        return missing
    return missing


def hf_snapshot_is_usable(
    snapshot_dir: Path | str,
    *,
    required_files: Optional[Sequence[str]] = None,
    require_weight_files: bool = False,
    reject_incomplete: bool = True,
) -> bool:
    snapshot = _expand_path(snapshot_dir)
    if not snapshot.is_dir():
        return False
    if reject_incomplete and hf_snapshot_has_incomplete_downloads(snapshot):
        return False
    for pattern in required_files or ():
        try:
            if not any(path.exists() for path in snapshot.glob(str(pattern))):
                return False
        except Exception:
            return False
    if require_weight_files and not hf_snapshot_has_weight_files(snapshot):
        return False
    if hf_snapshot_missing_indexed_weight_files(snapshot):
        return False
    return True


def _iter_hf_repo_snapshots(
    repo_id: str,
    *,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    extra_roots: Optional[Sequence[_RootSpec]] = None,
) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if "/" not in str(repo_id or ""):
        return out
    repo_dir_name = _repo_dir_name(repo_id)
    for label, root in hf_cache_roots(cache_dir=cache_dir, extra_roots=extra_roots):
        snap = _snapshot_from_repo_dir(root / repo_dir_name, revision)
        if snap is not None:
            out.append((label, snap))
    return out


def cached_hf_model_sources(
    model_id: str,
    *,
    cache_dir: Optional[str] = None,
    extra_roots: Optional[Sequence[_RootSpec]] = None,
    required_files: Optional[Sequence[str]] = None,
    require_weight_files: bool = False,
    reject_incomplete: bool = True,
) -> List[str]:
    """Return cache labels that already contain at least one snapshot for a repo."""

    sources: List[str] = []
    for label, snap in _iter_hf_repo_snapshots(model_id, cache_dir=cache_dir, extra_roots=extra_roots):
        if hf_snapshot_is_usable(
            snap,
            required_files=required_files,
            require_weight_files=require_weight_files,
            reject_incomplete=reject_incomplete,
        ):
            sources.append(label)
    return sources


def incomplete_hf_model_sources(
    model_id: str,
    *,
    cache_dir: Optional[str] = None,
    extra_roots: Optional[Sequence[_RootSpec]] = None,
    required_files: Optional[Sequence[str]] = None,
    require_weight_files: bool = False,
    reject_incomplete: bool = True,
) -> List[str]:
    sources: List[str] = []
    repo_dir_name = _repo_dir_name(model_id)
    for label, root in hf_cache_roots(cache_dir=cache_dir, extra_roots=extra_roots):
        repo_dir = root / repo_dir_name
        lock_dir = root / ".locks" / repo_dir_name
        snap = _snapshot_from_repo_dir(repo_dir, None)
        if snap is None:
            if repo_dir.exists() or lock_dir.exists():
                sources.append(label)
            continue
        if not hf_snapshot_is_usable(
            snap,
            required_files=required_files,
            require_weight_files=require_weight_files,
            reject_incomplete=reject_incomplete,
        ):
            sources.append(label)
    return sources


def _snapshot_from_repo_dir(repo_dir: Path, revision: Optional[str]) -> Optional[Path]:
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return None

    candidates: List[Path] = []
    if revision:
        ref = repo_dir / "refs" / str(revision).strip()
        try:
            if ref.is_file():
                commit = ref.read_text(encoding="utf-8").strip()
                snap_dir = snaps / commit
                if snap_dir.is_dir():
                    return snap_dir
        except Exception:
            pass

    main_ref = repo_dir / "refs" / "main"
    try:
        if main_ref.is_file():
            commit = main_ref.read_text(encoding="utf-8").strip()
            snap_dir = snaps / commit
            if snap_dir.is_dir():
                return snap_dir
    except Exception:
        pass

    try:
        candidates = [entry for entry in snaps.iterdir() if entry.is_dir()]
    except Exception:
        candidates = []
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry.stat().st_mtime)


def resolve_hf_repo_snapshot(
    repo_id: str,
    *,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    extra_roots: Optional[Sequence[_RootSpec]] = None,
    required_files: Optional[Sequence[str]] = None,
    require_weight_files: bool = False,
    reject_incomplete: bool = True,
) -> Optional[Path]:
    """Return the best local snapshot path for a Hugging Face repository."""

    for _, snap in _iter_hf_repo_snapshots(repo_id, cache_dir=cache_dir, revision=revision, extra_roots=extra_roots):
        if hf_snapshot_is_usable(
            snap,
            required_files=required_files,
            require_weight_files=require_weight_files,
            reject_incomplete=reject_incomplete,
        ):
            return snap
    return None


def _legacy_snapshot_name(repo_id: str, source_dir: Path) -> str:
    digest = hashlib.sha1(f"{repo_id}\0{str(source_dir.resolve())}".encode("utf-8")).hexdigest()
    return digest


def _write_ref(repo_dir: Path, ref_name: str, snapshot_name: str) -> None:
    refs_dir = repo_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / ref_name).write_text(snapshot_name, encoding="utf-8")


def _best_effort_remove(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    except Exception:
        pass


def import_directory_to_hf_cache(
    source_dir: Path | str,
    *,
    repo_id: str,
    cache_dir: Optional[str] = None,
    ref_name: str = "main",
    cleanup_source: bool = True,
) -> Path:
    """Move a legacy model directory into a Hugging Face cache snapshot layout."""

    source = _expand_path(source_dir)
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Legacy model directory does not exist: {source}")

    repo_dir = hf_repo_dir(repo_id, cache_dir=cache_dir)
    snapshots_dir = repo_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    snapshot_name = _legacy_snapshot_name(repo_id, source)
    snapshot_dir = snapshots_dir / snapshot_name
    if snapshot_dir.exists():
        _write_ref(repo_dir, ref_name, snapshot_name)
        if cleanup_source and source.resolve() != snapshot_dir.resolve():
            _best_effort_remove(source)
        return snapshot_dir

    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            source.replace(snapshot_dir)
        except OSError:
            shutil.copytree(source, snapshot_dir)
            if cleanup_source:
                _best_effort_remove(source)
    except Exception:
        _best_effort_remove(snapshot_dir)
        raise

    _write_ref(repo_dir, ref_name, snapshot_name)
    return snapshot_dir


def ensure_hf_repo_snapshot(
    repo_id: str,
    *,
    source_dir: Optional[Path | str] = None,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    extra_roots: Optional[Sequence[_RootSpec]] = None,
    ref_name: str = "main",
    cleanup_source: bool = True,
    required_files: Optional[Sequence[str]] = None,
    require_weight_files: bool = False,
    reject_incomplete: bool = True,
) -> Optional[Path]:
    """Resolve a cached snapshot, or import a legacy tree into the HF cache."""

    snap = resolve_hf_repo_snapshot(
        repo_id,
        cache_dir=cache_dir,
        revision=revision,
        extra_roots=extra_roots,
        required_files=required_files,
        require_weight_files=require_weight_files,
        reject_incomplete=reject_incomplete,
    )
    if snap is not None:
        if source_dir is not None and cleanup_source:
            source = _expand_path(source_dir)
            if source.exists() and source.resolve() != snap.resolve():
                _best_effort_remove(source)
        return snap

    if source_dir is None:
        return None
    source = _expand_path(source_dir)
    if not source.exists() or not source.is_dir():
        return None
    return import_directory_to_hf_cache(
        source,
        repo_id=repo_id,
        cache_dir=cache_dir,
        ref_name=ref_name,
        cleanup_source=cleanup_source,
    )
