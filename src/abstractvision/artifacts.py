from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

from .errors import AbstractVisionError


_ARTIFACT_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compute_artifact_id(content: bytes) -> str:
    """Compute a stable content-addressed artifact id (sha256 truncated)."""
    return sha256_hex(content)[:32]


def is_artifact_ref(value: Any) -> bool:
    """Check if a value matches the framework artifact-ref shape."""
    return isinstance(value, dict) and isinstance(value.get("$artifact"), str) and bool(value.get("$artifact"))


def get_artifact_id(ref: Dict[str, Any]) -> str:
    """Extract artifact id from a ref dict."""
    return str(ref["$artifact"])


def make_media_ref(
    artifact_id: str,
    *,
    content_type: Optional[str] = None,
    filename: Optional[str] = None,
    sha256: Optional[str] = None,
    size_bytes: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"$artifact": str(artifact_id)}
    if content_type:
        out["content_type"] = str(content_type)
    if filename:
        out["filename"] = str(filename)
    if sha256:
        out["sha256"] = str(sha256)
    if size_bytes is not None:
        out["size_bytes"] = int(size_bytes)
    if isinstance(metadata, dict) and metadata:
        out["metadata"] = metadata
    return out


class MediaStore(Protocol):
    """Minimal storage interface for generated media outputs (artifact-ref first)."""

    def store_bytes(
        self,
        content: bytes,
        *,
        content_type: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        run_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    def load_bytes(self, artifact_id: str) -> bytes: ...

    def get_metadata(self, artifact_id: str) -> Optional[Dict[str, Any]]: ...


class LocalAssetStore:
    """Local filesystem store for generated assets (standalone mode).

    Writes:
    - content:  <base_dir>/<artifact_id><ext>
    - metadata: <base_dir>/<artifact_id>.meta.json
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        if base_dir is None:
            base_dir = Path.home() / ".abstractvision" / "assets"
        self._base_dir = Path(base_dir).expanduser().resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def _validate_artifact_id(self, artifact_id: str) -> None:
        if not _ARTIFACT_ID_RE.match(str(artifact_id or "")):
            raise ValueError(f"Invalid artifact id: {artifact_id!r}")

    def _meta_path(self, artifact_id: str) -> Path:
        return self._base_dir / f"{artifact_id}.meta.json"

    def _guess_ext(self, content_type: str) -> str:
        ct = str(content_type or "").strip().lower()
        if not ct:
            return ".bin"
        ext = mimetypes.guess_extension(ct) or ""
        # Avoid ambiguous/empty extensions.
        return ext if ext else ".bin"

    def store_bytes(
        self,
        content: bytes,
        *,
        content_type: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        run_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError("content must be bytes")
        content_b = bytes(content)
        content_type = str(content_type or "application/octet-stream")

        sha = sha256_hex(content_b)
        artifact_id = str(artifact_id or compute_artifact_id(content_b))
        self._validate_artifact_id(artifact_id)

        ext = self._guess_ext(content_type)
        content_path = self._base_dir / f"{artifact_id}{ext}"
        meta_path = self._meta_path(artifact_id)

        # Best-effort idempotency: don't rewrite existing blobs.
        if not content_path.exists():
            content_path.write_bytes(content_b)

        meta: Dict[str, Any] = {
            "schema": "abstractvision.asset.v1",
            "artifact_id": artifact_id,
            "content_type": content_type,
            "size_bytes": len(content_b),
            "sha256": sha,
            "created_at": _utc_now_iso(),
            "content_file": content_path.name,
        }
        if filename:
            meta["filename"] = str(filename)
        if run_id:
            meta["run_id"] = str(run_id)
        if isinstance(tags, dict) and tags:
            meta["tags"] = {str(k): str(v) for k, v in tags.items()}
        if isinstance(metadata, dict) and metadata:
            meta["metadata"] = metadata

        # Always refresh meta so UX fields can be updated without rewriting blobs.
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        return make_media_ref(
            artifact_id,
            content_type=content_type,
            filename=str(filename) if filename else None,
            sha256=sha,
            size_bytes=len(content_b),
            metadata=metadata if isinstance(metadata, dict) else None,
        )

    def get_metadata(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        artifact_id = str(artifact_id or "")
        self._validate_artifact_id(artifact_id)
        p = self._meta_path(artifact_id)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def get_content_path(self, artifact_id: str) -> Optional[Path]:
        """Best-effort: return the path to the stored blob for an artifact id (local store only)."""
        artifact_id = str(artifact_id or "")
        self._validate_artifact_id(artifact_id)
        meta = self.get_metadata(artifact_id)
        if isinstance(meta, dict):
            content_file = meta.get("content_file")
            if isinstance(content_file, str) and content_file:
                p = self._base_dir / content_file
                if p.exists():
                    return p

        matches = sorted(self._base_dir.glob(f"{artifact_id}.*"))
        for p in matches:
            if p.name.endswith(".meta.json"):
                continue
            if p.is_file():
                return p
        return None

    def load_bytes(self, artifact_id: str) -> bytes:
        artifact_id = str(artifact_id or "")
        self._validate_artifact_id(artifact_id)
        p = self.get_content_path(artifact_id)
        if p is not None:
            return p.read_bytes()

        # Fallback: locate blob by prefix.
        matches = sorted(self._base_dir.glob(f"{artifact_id}.*"))
        for p in matches:
            if p.name.endswith(".meta.json"):
                continue
            if p.is_file():
                return p.read_bytes()
        raise FileNotFoundError(f"Asset not found: {artifact_id}")


class RuntimeArtifactStoreAdapter:
    """Duck-typed adapter for AbstractRuntime's ArtifactStore (no hard dependency)."""

    def __init__(self, artifact_store: Any):
        self._store = artifact_store

    def store_bytes(
        self,
        content: bytes,
        *,
        content_type: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        run_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        store_fn = getattr(self._store, "store", None)
        if not callable(store_fn):
            raise AbstractVisionError("Provided artifact_store does not have a callable .store(...)")

        content_b = bytes(content)
        content_type = str(content_type or "application/octet-stream")
        sha = sha256_hex(content_b)

        merged_tags: Dict[str, str] = {}
        if isinstance(tags, dict):
            merged_tags.update({str(k): str(v) for k, v in tags.items()})
        if filename and "filename" not in merged_tags:
            merged_tags["filename"] = str(filename)
        if sha and "sha256" not in merged_tags:
            merged_tags["sha256"] = sha

        # Try the full AbstractRuntime signature first; fall back if needed.
        try:
            meta = store_fn(
                content_b,
                content_type=content_type,
                run_id=str(run_id) if run_id else None,
                tags=merged_tags or None,
                artifact_id=str(artifact_id) if artifact_id else None,
            )
        except TypeError:
            meta = store_fn(
                content_b,
                content_type=content_type,
                run_id=str(run_id) if run_id else None,
                tags=merged_tags or None,
            )

        artifact_id_out = None
        if isinstance(meta, dict):
            artifact_id_out = meta.get("artifact_id")
        elif hasattr(meta, "artifact_id"):
            artifact_id_out = getattr(meta, "artifact_id", None)
        if not isinstance(artifact_id_out, str) or not artifact_id_out.strip():
            raise AbstractVisionError("artifact_store.store(...) did not return a usable artifact_id")

        return make_media_ref(
            str(artifact_id_out),
            content_type=content_type,
            filename=str(filename) if filename else None,
            sha256=sha,
            size_bytes=len(content_b),
            metadata=metadata if isinstance(metadata, dict) else None,
        )

    def load_bytes(self, artifact_id: str) -> bytes:
        load_fn = getattr(self._store, "load", None)
        if not callable(load_fn):
            raise AbstractVisionError("Provided artifact_store does not have a callable .load(...)")
        artifact = load_fn(str(artifact_id))
        if artifact is None:
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        if isinstance(artifact, (bytes, bytearray)):
            return bytes(artifact)
        if hasattr(artifact, "content"):
            return bytes(getattr(artifact, "content"))
        raise AbstractVisionError("artifact_store.load(...) returned an unsupported value")

    def get_metadata(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        meta_fn = getattr(self._store, "get_metadata", None)
        if not callable(meta_fn):
            return None
        meta = meta_fn(str(artifact_id))
        if meta is None:
            return None
        if isinstance(meta, dict):
            return meta
        to_dict = getattr(meta, "to_dict", None)
        if callable(to_dict):
            out = to_dict()
            return out if isinstance(out, dict) else None
        if is_dataclass(meta):
            out = asdict(meta)
            return out if isinstance(out, dict) else None
        return None
