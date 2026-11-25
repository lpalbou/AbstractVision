from __future__ import annotations

import glob
import os
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .model_capabilities import VisionModelCapabilitiesRegistry
from .model_cache import (
    default_hf_cache_root,
    default_legacy_model_root,
    ensure_hf_repo_snapshot,
    hf_repo_dir,
    hf_snapshot_is_usable,
    resolve_hf_repo_snapshot,
    write_hf_ref,
)


# The catalog `source` of the packages AbstractFramework prepares with MLX-Gen (`AbstractFramework/*`).
ABSTRACTFRAMEWORK_MLX_GEN_SOURCE = "abstractframework-mlx-gen"

_GENERIC_MLX_BACKEND_ERROR = (
    "AbstractVision does not have a generic MLX image/video backend yet. "
    "Use `--target mlx` to browse MLX artifacts and `--provider mlx-gen` "
    "(legacy alias: `mflux`) for cache-backed MLX-Gen image/video models."
)

_MACOS_GGUF_DISABLED_ERROR = (
    "GGUF 8-bit stable-diffusion.cpp presets are disabled on this macOS host. "
    "This is controlled by `ABSTRACTVISION_DISABLE_GGUF_ON_MACOS=1` (intended for operators who want to "
    "hide the stable-diffusion.cpp catalog entries on Apple Silicon)."
)


class MacOSGGUFUnsupportedError(ValueError):
    """Raised when a macOS host requests a disabled GGUF/stable-diffusion.cpp preset."""


def _env_truthy(key: str, *, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return bool(default)
    v = str(raw).strip().lower()
    if v in {"", "0", "false", "no", "off"}:
        return False
    return True


def _macos_gguf_disabled() -> bool:
    if sys.platform != "darwin":
        return False
    if local_model_profile() != "apple-silicon":
        return False
    return _env_truthy("ABSTRACTVISION_DISABLE_GGUF_ON_MACOS", default=False)


def _raise_macos_gguf_disabled() -> None:
    raise MacOSGGUFUnsupportedError(_MACOS_GGUF_DISABLED_ERROR)


@dataclass(frozen=True)
class VisionModelDownloadPreset:
    key: str
    display_name: str
    repo_id: str
    target: str
    engine: str
    local_dir_name: str
    quantization_bits: Optional[int]
    upstream_repo_id: Optional[str]
    source: str
    aliases: Tuple[str, ...]
    allow_patterns: Tuple[str, ...]
    notes: str = ""
    source_priority: int = 100
    # From the catalog: the repository is gated (its licence must be accepted on its page), the
    # package `download <model id>` picks among several when no `--bits` is given, the licence name.
    gated: bool = False
    recommended: bool = False
    license_name: str = ""
    # From the catalog: generation never fetches this repository, even with allow_download.
    explicit_download: bool = False
    # False for a Hugging Face repository the catalog does not list (`repository_preset`).
    in_catalog: bool = True

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        is_exact_mlx_gen = (
            str(self.target).strip().lower() == "mlx"
            and str(self.engine).strip().lower().replace("_", "-") in {"mlx-gen", "mflux", "m-flux", "mlxgen"}
            and str(self.source).strip().lower() == ABSTRACTFRAMEWORK_MLX_GEN_SOURCE
            and str(self.repo_id).strip().lower().startswith("abstractframework/")
        )
        if is_exact_mlx_gen:
            repo = str(self.repo_id).strip()
            out["aliases"] = [repo, repo.rsplit("/", 1)[-1]]
        else:
            out["aliases"] = list(self.aliases)
        out["allow_patterns"] = list(self.allow_patterns)
        # #FALLBACK: keep the historical key for callers that consumed the
        # first downloader implementation before the engine/target split.
        out["runner"] = self.engine
        return out


# Legacy preset trees used to live under `~/models`. Keep the constant for
# compatibility, but treat it as a migration source rather than the download
# destination.
DEFAULT_MODEL_DIR = Path("~/models")
HF_TOKEN_SETTINGS_URL = "https://huggingface.co/settings/tokens"


_COMMON_MLX_PATTERNS = (
    "model_index.json",
    "*.json",
    "*.md",
    "*.pth",
    "*.pt",
    "LICENSE*",
    "scheduler/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "transformer/*",
    "transformer-packed-mflux/*",
    "vae/*",
    "text_encoder-mlx-4bit/*",
    "assets/*",
    "*.jinja",
    # Wan 2.2 A14B: the low-noise expert.
    "transformer_2/*",
    # MiniMax-H3: the soundtrack decoder and its schedule, and the licence notice.
    "audio_vae/*",
    "audio_scheduler/*",
    "NOTICE*",
)

_COMMON_DIFFUSERS_PATTERNS = (
    "model_index.json",
    "*.json",
    "*.jinja",
    "*.md",
    "*.model",
    "*.txt",
    "*.yaml",
    "LICENSE*",
    "*.safetensors",
)

_DIFFUSERS_PIPELINE_DIRS = (
    "scheduler/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "unet/*",
    "transformer/*",
    "vae/*",
    "safety_checker/*",
    "feature_extractor/*",
    "assets/*",
)

_STABLE_DIFFUSION_DIFFUSERS_PATTERNS = (
    "model_index.json",
    "*.json",
    "*.md",
    "*.txt",
    "*.yaml",
    "LICENSE*",
    "scheduler/*",
    "tokenizer/*",
    "feature_extractor/*",
    "safety_checker/*.json",
    "safety_checker/*.safetensors",
    "text_encoder/*.json",
    "text_encoder/*.safetensors",
    "unet/*.json",
    "unet/*.safetensors",
    "vae/*.json",
    "vae/*.safetensors",
)


class HuggingFaceAccessError(RuntimeError):
    """The Hub refused a repository (HTTP 401, 403 or 404).

    `gated` says the repository is known to be gated (the catalog marks it, or, for a repository
    the catalog does not list, its Hub listing does): the message then names the licence to accept.
    Otherwise the message says what the status code alone tells: the Hub answers 401 to a request
    without valid credentials for a repository that does not exist, is private or is gated, and
    does not say which."""

    def __init__(
        self,
        repo_id: str,
        *,
        raw_message: str = "",
        status_code: Optional[int] = None,
        license_name: str = "",
        gated: bool = False,
    ):
        self.repo_id = str(repo_id)
        self.repo_url = f"https://huggingface.co/{self.repo_id}"
        self.token_url = HF_TOKEN_SETTINGS_URL
        self.raw_message = str(raw_message or "").strip()
        self.status_code = status_code
        self.license_name = str(license_name or "").strip()
        self.gated = bool(gated)
        detail = f" ({status_code})" if status_code else ""
        retry = (
            f"Create or copy a Hugging Face read token at {self.token_url}, then retry with one of:\n"
            "   - `abstractvision download ... --token <HF_TOKEN>`\n"
            "   - `hf auth login` (or `huggingface-cli login`)\n"
            "   - `export HF_TOKEN=...` (or `export HUGGINGFACE_HUB_TOKEN=...`)\n"
        )
        if self.gated:
            terms = f"the {self.license_name}" if self.license_name else "its terms"
            msg = (
                f"Cannot access Hugging Face repo {self.repo_id!r}{detail}: the repository is gated, and this "
                "request has no access to it.\n"
                f"1. Open {self.repo_url} while logged in to Hugging Face and accept {terms} "
                "(or request access).\n"
                f"2. {retry}"
            )
        else:
            why = {
                401: "the Hub gives this answer to a request without valid credentials when the repository "
                "does not exist, is private, or is gated, and does not say which.",
                403: "the Hub refused this account: the repository is private, or gated and this account has "
                "not been granted access, or the token cannot read it.",
                404: "the Hub has no repository by that name that this account can see.",
            }.get(int(status_code or 0), "the Hub refused the request.")
            msg = (
                f"Cannot access Hugging Face repo {self.repo_id!r}{detail}: {why}\n"
                f"1. Check the name at {self.repo_url}.\n"
                "2. If the repository is private or gated, use an account that has access (for a gated one, "
                f"accept its terms on that page). {retry}"
            )
        first_line = next((line.strip() for line in self.raw_message.splitlines() if line.strip()), "")
        if first_line:
            msg += f"Upstream error: {first_line}"
        super().__init__(msg)


class InsufficientDiskSpaceError(RuntimeError):
    """Raised before a download starts when the Hugging Face cache disk cannot hold it."""


@dataclass(frozen=True)
class HubFile:
    """One file a download selects, as the Hub lists it, and what the cache already holds of it.

    `complete` follows huggingface_hub's own test for a cached file (its snapshot link or its blob
    exists): a complete file is never fetched again. `partial` is the size of its `.incomplete` blob
    left by an interrupted download; huggingface_hub writes the file into that same blob, continuing
    it (plain HTTP) or rewriting it from the start (Xet), so its bytes are space the fetch reuses."""

    name: str
    size: int
    complete: bool
    partial: int = 0


# The disk rule every plan applies, stated once in its output.
DISK_RULE = (
    "the files to fetch at their full size, less what their partial (.incomplete) files already hold: "
    "huggingface_hub continues or rewrites each partial file in place"
)


@dataclass(frozen=True)
class HubDownloadPlan:
    """What a download fetches from one repository, measured against the Hugging Face cache before it starts.

    Built from the Hub listing (metadata only) at `revision` (None: the default branch), which named
    `commit`: the fetch takes the files of that commit. `listing_error` says why the listing could
    not be had; the plan then sizes nothing. `gated` is None when neither the catalog nor the listing
    says whether the repository is gated."""

    repo_id: str
    cache_dir: Path
    free_bytes: Optional[int]
    files: Tuple[HubFile, ...] = ()
    commit: Optional[str] = None
    gated: Optional[bool] = None
    license_name: str = ""
    listing_error: Optional[str] = None
    revision: Optional[str] = None

    @property
    def listed(self) -> bool:
        return self.listing_error is None

    @property
    def total_bytes(self) -> Optional[int]:
        return sum(f.size for f in self.files) if self.listed else None

    @property
    def complete_files(self) -> Tuple[HubFile, ...]:
        return tuple(f for f in self.files if f.complete)

    @property
    def fetch_files(self) -> Tuple[HubFile, ...]:
        return tuple(f for f in self.files if not f.complete)

    @property
    def complete_bytes(self) -> int:
        return sum(f.size for f in self.complete_files)

    @property
    def fetch_bytes(self) -> Optional[int]:
        return sum(f.size for f in self.fetch_files) if self.listed else None

    @property
    def partial_bytes(self) -> int:
        return sum(min(f.partial, f.size) for f in self.fetch_files)

    @property
    def required_bytes(self) -> Optional[int]:
        """Free space the fetch needs (`DISK_RULE`)."""

        return self.fetch_bytes - self.partial_bytes if self.listed else None

    def access_line(self) -> str:
        if self.gated is None:
            return "access: unknown (the Hub listing is not available, and the catalog does not list the repository)"
        if not self.gated:
            return "access: open"
        terms = f"the {self.license_name}" if self.license_name else "its terms"
        return (
            f"access: gated; accept {terms} at https://huggingface.co/{self.repo_id} while logged in to "
            "Hugging Face (`hf auth login`, or --token)"
        )

    def cache_line(self) -> str:
        count = len(self.files)
        if not self.fetch_files:
            return f"cache: all {_files(count)} already present; nothing to fetch"
        partial = (
            f", {format_bytes(self.partial_bytes)} of it already in partial files" if self.partial_bytes else ""
        )
        return (
            f"cache: {len(self.complete_files)} of {_files(count)} complete ({format_bytes(self.complete_bytes)}); "
            f"{_files(len(self.fetch_files))} to fetch ({format_bytes(self.fetch_bytes)}{partial})"
        )

    def commit_line(self) -> str:
        named = f"{self.revision!r}" if self.revision else "the default branch"
        return f"commit: {self.commit} ({named} as listed; the files are fetched from this commit)"

    def summary_lines(self) -> List[str]:
        return DownloadPlan(parts=(DownloadPart("package", self),)).summary_lines()

    def disk_space_error(self) -> Optional[str]:
        return DownloadPlan(parts=(DownloadPart("package", self),)).disk_space_error()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "cache_dir": str(self.cache_dir),
            "revision": self.revision,
            "commit": self.commit,
            "gated": self.gated,
            "listing_error": self.listing_error,
            "total_bytes": self.total_bytes,
            "files": len(self.files),
            "complete_files": len(self.complete_files),
            "complete_bytes": self.complete_bytes if self.listed else None,
            "fetch_files": len(self.fetch_files),
            "fetch_bytes": self.fetch_bytes,
            "partial_bytes": self.partial_bytes if self.listed else None,
            "required_bytes": self.required_bytes,
            "free_bytes": self.free_bytes,
        }


@dataclass(frozen=True)
class DownloadPart:
    """One repository a download fetches from: the package itself, or what its engine loads with it
    (a stable-diffusion.cpp bundle's text encoder or VAE, the adapter an MLX-Gen schedule attaches).
    `patterns` are the files it selects, for the local look when the Hub listing is not available."""

    role: str
    plan: HubDownloadPlan
    patterns: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DownloadPlan:
    """Everything `abstractvision download` fetches for one preset, planned before the first byte.

    `parts` are the repositories to fetch from, the package first. `legacy_import` is a legacy
    `~/models/<preset>` tree imported into the cache instead of fetching the package; `legacy_remove`
    one deleted because the cache already holds the package. `base_model` is the MLX-Gen schedule
    asked for, `adapter` the adapter it attaches and `adapter_path` where mlx-gen finds it when it is
    cached (its fetch is then not a part). The sizes add up over the parts, and the disk check
    applies `DISK_RULE` to all of them on each disk they write to."""

    parts: Tuple[DownloadPart, ...]
    legacy_import: Optional[Path] = None
    legacy_remove: Optional[Path] = None
    base_model: Optional[str] = None
    adapter: Optional[str] = None
    adapter_path: Optional[str] = None

    @property
    def package(self) -> Optional[HubDownloadPlan]:
        return next((part.plan for part in self.parts if part.role == "package"), None)

    @property
    def listed(self) -> bool:
        return all(part.plan.listed for part in self.parts)

    def _sum(self, field: str) -> Optional[int]:
        values = [getattr(part.plan, field) for part in self.parts]
        return None if any(v is None for v in values) else sum(values)

    @property
    def total_bytes(self) -> Optional[int]:
        return self._sum("total_bytes")

    @property
    def fetch_bytes(self) -> Optional[int]:
        return self._sum("fetch_bytes")

    @property
    def required_bytes(self) -> Optional[int]:
        return self._sum("required_bytes")

    @property
    def free_bytes(self) -> Optional[int]:
        return next((part.plan.free_bytes for part in self.parts if part.plan.free_bytes is not None), None)

    def summary_lines(self) -> List[str]:
        """The access, size, cache and disk lines a download (or `--dry-run`) prints before it starts:
        the package's, one line per other part, then the totals and the disk check."""

        lines: List[str] = []
        for part in self.parts:
            plan = part.plan
            if part.role == "package":
                lines.append(plan.access_line())
                if not plan.listed:
                    lines.append(
                        f"hub: no usable listing of {plan.repo_id}: {plan.listing_error}. Nothing was sized or "
                        "compared with the cache"
                    )
                elif not plan.files:
                    lines.append(f"files: the Hub lists no file of {plan.repo_id} that this download selects")
                else:
                    lines.append(
                        f"download size: {format_bytes(plan.total_bytes)} in {_files(len(plan.files))}, "
                        f"into {plan.cache_dir}"
                    )
                    lines.append(plan.cache_line())
                    lines.append(plan.commit_line())
                continue
            head = f"{part.role}: {plan.repo_id}; {plan.access_line()}"
            if not plan.listed:
                lines.append(f"{head}; hub: no usable listing: {plan.listing_error}")
            elif not plan.files:
                lines.append(f"{head}; the Hub lists no file of it that this download selects")
            else:
                lines.append(
                    f"{head}; {format_bytes(plan.total_bytes)} in {_files(len(plan.files))}; {plan.cache_line()}; "
                    f"{plan.commit_line()}"
                )
        if self.adapter_path is not None:
            lines.append(f"adapter ({self.base_model}): {self.adapter}, cached at {self.adapter_path}")
        elif self.base_model is not None and self.adapter is None:
            lines.append(f"adapter ({self.base_model}): this schedule loads none")
        listed = [part.plan for part in self.parts if part.plan.listed]
        fetching = [plan for plan in listed if plan.fetch_files]
        if not fetching:
            return lines
        if len(self.parts) > 1:
            lines.append(
                f"total: {format_bytes(sum(p.total_bytes for p in listed))} in "
                f"{_files(sum(len(p.files) for p in listed))}; "
                f"{_files(sum(len(p.fetch_files) for p in listed))} to fetch "
                f"({format_bytes(sum(p.fetch_bytes for p in listed))})"
            )
        need = sum(p.required_bytes for p in listed)
        if self.free_bytes is None:
            lines.append(f"disk: needs {format_bytes(need)} ({DISK_RULE}); not checked, because the free space is unknown")
            return lines
        lines.append(f"free disk space: {format_bytes(self.free_bytes)}")
        verdict = "not enough free space" if self.disk_space_error() else "enough free space"
        lines.append(f"disk: needs {format_bytes(need)} ({DISK_RULE}); {verdict}")
        return lines

    def disk_space_error(self) -> Optional[str]:
        """Why a cache disk cannot hold what the parts writing to it fetch (`DISK_RULE`), or None when
        every disk can (or its free space is unknown)."""

        disks: Dict[Any, List[HubDownloadPlan]] = {}
        for part in self.parts:
            if part.plan.listed:
                disks.setdefault(_disk_of(part.plan.cache_dir), []).append(part.plan)
        for plans in disks.values():
            need = sum(p.required_bytes for p in plans)
            free = next((p.free_bytes for p in plans if p.free_bytes is not None), None)
            if free is None or need <= free:
                continue
            partial = sum(p.partial_bytes for p in plans)
            held = (
                f" ({format_bytes(sum(p.fetch_bytes for p in plans))} less the {format_bytes(partial)} their "
                "partial files already hold)"
                if partial
                else ""
            )
            what = plans[0].repo_id + (
                f" with {', '.join(p.repo_id for p in plans[1:])}" if len(plans) > 1 else ""
            )
            return (
                f"Not enough free disk space to download {what}: the "
                f"{_files(sum(len(p.fetch_files) for p in plans))} to fetch take {format_bytes(need)}{held} and "
                f"{format_bytes(free)} is free at {_existing_parent(plans[0].cache_dir)}. Free some space, or point "
                "HF_HUB_CACHE (or HF_HOME) at a disk with room, then run the same command again."
            )
        return None

    def to_dict(self) -> Dict[str, Any]:
        package = self.package
        return {
            **(package.to_dict() if package is not None else {}),
            "total_bytes": self.total_bytes,
            "fetch_bytes": self.fetch_bytes,
            "required_bytes": self.required_bytes,
            "free_bytes": self.free_bytes,
            "disk_space_error": self.disk_space_error(),
            "parts": [{"role": part.role, **part.plan.to_dict()} for part in self.parts],
            "legacy_import": str(self.legacy_import) if self.legacy_import is not None else None,
            "base_model": self.base_model,
            "adapter": self.adapter,
            "adapter_path": self.adapter_path,
        }


def _disk_of(path: Path) -> Any:
    existing = _existing_parent(path)
    try:
        return os.stat(existing).st_dev
    except OSError:
        return str(existing)


def _files(count: int) -> str:
    return f"{count} file" if count == 1 else f"{count} files"


def format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024.0 or unit == "GiB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} GiB"


_PRESETS: Tuple[VisionModelDownloadPreset, ...] = (
    VisionModelDownloadPreset(
        key="flux2-klein-4b",
        display_name="FLUX.2 klein 4B MLX-Gen q4",
        repo_id="AbstractFramework/flux.2-klein-4b-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="flux2-klein-4b-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-4B",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "flux2-klein-4b",
            "flux-klein-4b",
            "klein-4b",
            "klein4b",
            "flux4b",
            "AbstractFramework/flux.2-klein-4b-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes=(
            "AbstractFramework MLX-Gen prepared q4 folder. This is the default Apple Silicon "
            "recommendation for local memory efficiency."
        ),
        source_priority=20,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-4b",
        display_name="FLUX.2 klein 4B MLX-Gen q8",
        repo_id="AbstractFramework/flux.2-klein-4b-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="flux2-klein-4b-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-4B",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("AbstractFramework/flux.2-klein-4b-8bit",),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared q8 folder. Prefer when output quality is paramount.",
        source_priority=25,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-4b",
        display_name="FLUX.2 klein 4B (Diffusers)",
        repo_id="black-forest-labs/FLUX.2-klein-4B",
        target="diffusers",
        engine="diffusers",
        local_dir_name="flux2-klein-4b-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("flux2-klein-4b", "flux-klein-4b", "klein-4b", "klein4b", "flux4b"),
        allow_patterns=(
            "flux-2-klein-4b.safetensors",
            "model_index.json",
            "*.json",
            "*.md",
            "*.txt",
            "*.yaml",
            "LICENSE*",
            *_DIFFUSERS_PIPELINE_DIRS,
        ),
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=80,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-4b",
        display_name="FLUX.2 klein 4B official FP8",
        repo_id="black-forest-labs/FLUX.2-klein-4b-fp8",
        target="fp8",
        engine="diffusers-component",
        local_dir_name="flux2-klein-4b-fp8",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-4B",
        source="official",
        aliases=("flux2-klein-4b", "flux-klein-4b", "klein-4b", "klein4b", "flux4b"),
        allow_patterns=("*.json", "*.md", "LICENSE*", "*.safetensors"),
        notes=(
            "Official BFL FP8 side artifact. It is not a standalone Diffusers pipeline because "
            "the repo does not include model_index.json. Prefer the MLX preset on Apple Silicon."
        ),
        source_priority=0,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-4b",
        display_name="FLUX.2 klein 4B GGUF Q8_0",
        repo_id="leejet/FLUX.2-klein-4B-GGUF",
        target="gguf",
        engine="stable-diffusion.cpp",
        local_dir_name="flux2-klein-4b-q8_0-gguf",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-4B",
        source="runtime-native-gguf",
        aliases=("flux2-klein-4b", "flux-klein-4b", "klein-4b", "klein4b", "flux4b"),
        allow_patterns=("README.md", "LICENSE*", "flux-2-klein-4b-Q8_0.gguf"),
        notes="Curated Q8_0 GGUF conversion aligned with stable-diffusion.cpp runtime usage.",
        source_priority=40,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-base-4b",
        display_name="FLUX.2 klein base 4B MLX-Gen q4",
        repo_id="AbstractFramework/flux.2-klein-base-4b-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="flux2-klein-base-4b-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-base-4B",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "flux2-klein-base-4b",
            "flux-klein-base-4b",
            "klein-base-4b",
            "kleinbase4b",
            "fluxbase4b",
            "AbstractFramework/flux.2-klein-base-4b-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared q4 folder. Prefer this unless quality is paramount.",
        source_priority=20,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-base-4b",
        display_name="FLUX.2 klein base 4B MLX-Gen q8",
        repo_id="AbstractFramework/flux.2-klein-base-4b-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="flux2-klein-base-4b-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-base-4B",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("AbstractFramework/flux.2-klein-base-4b-8bit",),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared q8 folder. Prefer when output quality is paramount.",
        source_priority=25,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-base-4b",
        display_name="FLUX.2 klein base 4B GGUF Q8_0",
        repo_id="leejet/FLUX.2-klein-base-4B-GGUF",
        target="gguf",
        engine="stable-diffusion.cpp",
        local_dir_name="flux2-klein-base-4b-q8_0-gguf",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-base-4B",
        source="runtime-native-gguf",
        aliases=("flux2-klein-base-4b", "flux-klein-base-4b", "klein-base-4b", "kleinbase4b", "fluxbase4b"),
        allow_patterns=("README.md", "LICENSE*", "flux-2-klein-base-4b-Q8_0.gguf"),
        notes="Q8_0 GGUF conversion aligned with stable-diffusion.cpp runtime usage.",
        source_priority=40,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-9b",
        display_name="FLUX.2 klein 9B MLX-Gen q4",
        repo_id="AbstractFramework/flux.2-klein-9b-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="flux2-klein-9b-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-9B",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "flux2-klein-9b",
            "flux-klein-9b",
            "klein-9b",
            "klein9b",
            "flux9b",
            "AbstractFramework/flux.2-klein-9b-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes=(
            "AbstractFramework MLX-Gen prepared q4 folder. This is the default Apple Silicon "
            "recommendation for local memory efficiency."
        ),
        source_priority=20,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-9b",
        display_name="FLUX.2 klein 9B MLX-Gen q8",
        repo_id="AbstractFramework/flux.2-klein-9b-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="flux2-klein-9b-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-9B",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("AbstractFramework/flux.2-klein-9b-8bit",),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared q8 folder. Prefer when output quality is paramount.",
        source_priority=25,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-9b",
        display_name="FLUX.2 klein 9B (Diffusers)",
        repo_id="black-forest-labs/FLUX.2-klein-9B",
        target="diffusers",
        engine="diffusers",
        local_dir_name="flux2-klein-9b-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("flux2-klein-9b", "flux-klein-9b", "klein-9b", "klein9b", "flux9b"),
        allow_patterns=(
            "flux-2-klein-9b.safetensors",
            "model_index.json",
            "*.json",
            "*.md",
            "*.txt",
            "*.yaml",
            "LICENSE*",
            *_DIFFUSERS_PIPELINE_DIRS,
        ),
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=80,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-9b",
        display_name="FLUX.2 klein 9B official FP8",
        repo_id="black-forest-labs/FLUX.2-klein-9b-fp8",
        target="fp8",
        engine="diffusers-component",
        local_dir_name="flux2-klein-9b-fp8",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-9B",
        source="official",
        aliases=("flux2-klein-9b", "flux-klein-9b", "klein-9b", "klein9b", "flux9b"),
        allow_patterns=("*.json", "*.md", "LICENSE*", "*.safetensors"),
        notes=(
            "Official BFL FP8 side artifact. It is not a standalone Diffusers pipeline because "
            "the repo does not include model_index.json. Prefer the MLX preset on Apple Silicon."
        ),
        source_priority=0,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-9b",
        display_name="FLUX.2 klein 9B GGUF Q8_0",
        repo_id="leejet/FLUX.2-klein-9B-GGUF",
        target="gguf",
        engine="stable-diffusion.cpp",
        local_dir_name="flux2-klein-9b-q8_0-gguf",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-9B",
        source="runtime-native-gguf",
        aliases=("flux2-klein-9b", "flux-klein-9b", "klein-9b", "klein9b", "flux9b"),
        allow_patterns=("README.md", "LICENSE*", "flux-2-klein-9b-Q8_0.gguf"),
        notes="Curated Q8_0 GGUF conversion aligned with stable-diffusion.cpp runtime usage.",
        source_priority=40,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-base-9b",
        display_name="FLUX.2 klein base 9B MLX-Gen q4",
        repo_id="AbstractFramework/flux.2-klein-base-9b-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="flux2-klein-base-9b-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-base-9B",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "flux2-klein-base-9b",
            "flux-klein-base-9b",
            "klein-base-9b",
            "kleinbase9b",
            "fluxbase9b",
            "AbstractFramework/flux.2-klein-base-9b-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared q4 folder. Prefer this unless quality is paramount.",
        source_priority=20,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-base-9b",
        display_name="FLUX.2 klein base 9B MLX-Gen q8",
        repo_id="AbstractFramework/flux.2-klein-base-9b-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="flux2-klein-base-9b-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-base-9B",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("AbstractFramework/flux.2-klein-base-9b-8bit",),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared q8 folder. Prefer when output quality is paramount.",
        source_priority=25,
    ),
    VisionModelDownloadPreset(
        key="flux2-klein-base-9b",
        display_name="FLUX.2 klein base 9B GGUF Q8_0",
        repo_id="leejet/FLUX.2-klein-base-9B-GGUF",
        target="gguf",
        engine="stable-diffusion.cpp",
        local_dir_name="flux2-klein-base-9b-q8_0-gguf",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-base-9B",
        source="runtime-native-gguf",
        aliases=("flux2-klein-base-9b", "flux-klein-base-9b", "klein-base-9b", "kleinbase9b", "fluxbase9b"),
        allow_patterns=("README.md", "LICENSE*", "flux-2-klein-base-9b-Q8_0.gguf"),
        notes="Q8_0 GGUF conversion aligned with stable-diffusion.cpp runtime usage.",
        source_priority=40,
    ),
    VisionModelDownloadPreset(
        key="flux2-dev",
        display_name="FLUX.2 dev (Diffusers)",
        repo_id="black-forest-labs/FLUX.2-dev",
        target="diffusers",
        engine="diffusers",
        local_dir_name="flux2-dev-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("flux2-dev", "flux-2-dev", "flux.2-dev", "black-forest-labs/FLUX.2-dev"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="#FALLBACK: gated on Hugging Face; requires accepting terms + HF token.",
        source_priority=90,
    ),
    VisionModelDownloadPreset(
        key="z-image",
        display_name="Z-Image MLX-Gen q4",
        repo_id="AbstractFramework/z-image-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="z-image-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="Tongyi-MAI/Z-Image",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "z-image",
            "zimage",
            "tongyi-z-image",
            "AbstractFramework/z-image-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared q4 folder. Prefer this unless quality is paramount.",
        source_priority=20,
    ),
    VisionModelDownloadPreset(
        key="z-image",
        display_name="Z-Image MLX-Gen q8",
        repo_id="AbstractFramework/z-image-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="z-image-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="Tongyi-MAI/Z-Image",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("AbstractFramework/z-image-8bit",),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared q8 folder. Prefer when output quality is paramount.",
        source_priority=25,
    ),
    VisionModelDownloadPreset(
        key="z-image-turbo",
        display_name="Z-Image-Turbo MLX-Gen q4",
        repo_id="AbstractFramework/z-image-turbo-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="z-image-turbo-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="Tongyi-MAI/Z-Image-Turbo",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "z-image-turbo",
            "zimage-turbo",
            "tongyi-z-image-turbo",
            "AbstractFramework/z-image-turbo-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes=(
            "AbstractFramework MLX-Gen prepared q4 folder. This is the default Apple Silicon "
            "recommendation for local memory efficiency."
        ),
        source_priority=20,
    ),
    VisionModelDownloadPreset(
        key="z-image-turbo",
        display_name="Z-Image-Turbo MLX-Gen q8",
        repo_id="AbstractFramework/z-image-turbo-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="z-image-turbo-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="Tongyi-MAI/Z-Image-Turbo",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("AbstractFramework/z-image-turbo-8bit",),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared q8 folder. Prefer when output quality is paramount.",
        source_priority=25,
    ),
    VisionModelDownloadPreset(
        key="z-image-turbo",
        display_name="Z-Image-Turbo (Diffusers)",
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        target="diffusers",
        engine="diffusers",
        local_dir_name="z-image-turbo-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("z-image-turbo", "zimage-turbo", "tongyi-z-image-turbo"),
        allow_patterns=(
            "model_index.json",
            "*.json",
            "*.md",
            "*.txt",
            "*.yaml",
            "LICENSE*",
            *_DIFFUSERS_PIPELINE_DIRS,
        ),
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=80,
    ),
    VisionModelDownloadPreset(
        key="z-image-turbo",
        display_name="Z-Image-Turbo GGUF Q8_0",
        repo_id="unsloth/Z-Image-Turbo-GGUF",
        target="gguf",
        engine="stable-diffusion.cpp",
        local_dir_name="z-image-turbo-q8_0-gguf",
        quantization_bits=8,
        upstream_repo_id="Tongyi-MAI/Z-Image-Turbo",
        source="curated-community-gguf",
        aliases=("z-image-turbo", "zimage-turbo", "tongyi-z-image-turbo"),
        allow_patterns=("README.md", "LICENSE*", "z-image-turbo-Q8_0.gguf"),
        notes="Q8_0 GGUF for stable-diffusion.cpp style runtimes.",
        source_priority=40,
    ),
    VisionModelDownloadPreset(
        key="seedvr2-3b",
        display_name="SeedVR2 3B MLX-Gen q8 upscaler",
        repo_id="AbstractFramework/seedvr2-3b-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="seedvr2-3b-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="ByteDance-Seed/SeedVR2-3B",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "seedvr2",
            "seedvr2-3b",
            "seedvr2-3b-8bit",
            "AbstractFramework/seedvr2-3b-8bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes=(
            "Canonical AbstractFramework MLX-Gen prepared SeedVR2 3B 8-bit upscaler. "
            "This is the default upscaler package for local Apple Silicon runs."
        ),
        source_priority=18,
    ),
    VisionModelDownloadPreset(
        key="seedvr2-3b-4bit",
        display_name="SeedVR2 3B MLX-Gen q4 upscaler",
        repo_id="AbstractFramework/seedvr2-3b-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="seedvr2-3b-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="ByteDance-Seed/SeedVR2-3B",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "seedvr2-3b-4bit",
            "AbstractFramework/seedvr2-3b-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="Canonical AbstractFramework MLX-Gen prepared SeedVR2 3B 4-bit upscaler.",
        source_priority=19,
    ),
    VisionModelDownloadPreset(
        key="seedvr2-7b",
        display_name="SeedVR2 7B MLX-Gen q8 upscaler",
        repo_id="AbstractFramework/seedvr2-7b-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="seedvr2-7b-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="ByteDance-Seed/SeedVR2-7B",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "seedvr2-7b",
            "seedvr2-7b-8bit",
            "AbstractFramework/seedvr2-7b-8bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes=(
            "Canonical AbstractFramework MLX-Gen prepared SeedVR2 7B 8-bit upscaler. "
            "Use this larger default-quality package when memory allows."
        ),
        source_priority=18,
    ),
    VisionModelDownloadPreset(
        key="seedvr2-7b-4bit",
        display_name="SeedVR2 7B MLX-Gen q4 upscaler",
        repo_id="AbstractFramework/seedvr2-7b-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="seedvr2-7b-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="ByteDance-Seed/SeedVR2-7B",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "seedvr2-7b-4bit",
            "AbstractFramework/seedvr2-7b-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="Canonical AbstractFramework MLX-Gen prepared SeedVR2 7B 4-bit upscaler.",
        source_priority=19,
    ),
    VisionModelDownloadPreset(
        key="qwen-image",
        display_name="Qwen-Image-2512 MLX-Gen q4",
        repo_id="AbstractFramework/qwen-image-2512-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="qwen-image-2512-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="Qwen/Qwen-Image-2512",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "qwen-image",
            "qwen-image-2512",
            "qwen-image-4bit",
            "qwen-image-2512-4bit",
            "AbstractFramework/qwen-image-2512-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes=(
            "AbstractFramework MLX-Gen prepared Qwen folder. Qwen prepared folders may mix q4/q8 "
            "components; this is still the default local-memory recommendation."
        ),
        source_priority=20,
    ),
    VisionModelDownloadPreset(
        key="qwen-image",
        display_name="Qwen-Image-2512 MLX-Gen q8",
        repo_id="AbstractFramework/qwen-image-2512-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="qwen-image-2512-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="Qwen/Qwen-Image-2512",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("AbstractFramework/qwen-image-2512-8bit",),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared q8 Qwen Image folder. Prefer when output quality is paramount.",
        source_priority=25,
    ),
    VisionModelDownloadPreset(
        key="qwen-image",
        display_name="Qwen-Image legacy MLX-Gen q4",
        repo_id="AbstractFramework/qwen-image-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="qwen-image-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="Qwen/Qwen-Image",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("qwen-image-legacy", "qwen-image-base", "AbstractFramework/qwen-image-4bit"),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared legacy Qwen Image folder; prefer Qwen-Image-2512 for new runs.",
        source_priority=35,
    ),
    VisionModelDownloadPreset(
        key="qwen-image",
        display_name="Qwen-Image legacy MLX-Gen q8",
        repo_id="AbstractFramework/qwen-image-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="qwen-image-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="Qwen/Qwen-Image",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("AbstractFramework/qwen-image-8bit",),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared legacy q8 Qwen Image folder; prefer Qwen-Image-2512 for new runs.",
        source_priority=40,
    ),
    VisionModelDownloadPreset(
        key="qwen-image-edit",
        display_name="Qwen-Image-Edit legacy MLX-Gen q4",
        repo_id="AbstractFramework/qwen-image-edit-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="qwen-image-edit-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="Qwen/Qwen-Image-Edit",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "qwen-image-edit-legacy",
            "qwen-image-edit-base",
            "AbstractFramework/qwen-image-edit-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared legacy Qwen Image Edit folder; prefer Qwen-Image-Edit-2511 for new runs.",
        source_priority=35,
    ),
    VisionModelDownloadPreset(
        key="qwen-image-edit",
        display_name="Qwen-Image-Edit legacy MLX-Gen q8",
        repo_id="AbstractFramework/qwen-image-edit-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="qwen-image-edit-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="Qwen/Qwen-Image-Edit",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("AbstractFramework/qwen-image-edit-8bit",),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared legacy q8 Qwen Image Edit folder; prefer Qwen-Image-Edit-2511 for new runs.",
        source_priority=40,
    ),
    VisionModelDownloadPreset(
        key="qwen-image-edit-2511",
        display_name="Qwen-Image-Edit-2511 MLX-Gen q4",
        repo_id="AbstractFramework/qwen-image-edit-2511-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="qwen-image-edit-2511-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="Qwen/Qwen-Image-Edit-2511",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=(
            "qwen-image-edit",
            "qwen-image-edit-2511",
            "AbstractFramework/qwen-image-edit-2511-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared Qwen Image Edit 2511 folder for image-to-image/edit.",
        source_priority=20,
    ),
    VisionModelDownloadPreset(
        key="qwen-image-edit-2511",
        display_name="Qwen-Image-Edit-2511 MLX-Gen q8",
        repo_id="AbstractFramework/qwen-image-edit-2511-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="qwen-image-edit-2511-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="Qwen/Qwen-Image-Edit-2511",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("AbstractFramework/qwen-image-edit-2511-8bit",),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared q8 Qwen Image Edit 2511 folder. Prefer when output quality is paramount.",
        source_priority=25,
    ),
    VisionModelDownloadPreset(
        key="qwen-image-edit-2509",
        display_name="Qwen-Image-Edit-2509 MLX-Gen q4",
        repo_id="AbstractFramework/qwen-image-edit-2509-4bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="qwen-image-edit-2509-mlx-gen-4bit",
        quantization_bits=4,
        upstream_repo_id="Qwen/Qwen-Image-Edit-2509",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("qwen-image-edit-2509", "AbstractFramework/qwen-image-edit-2509-4bit"),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared Qwen Image Edit 2509 folder for image-to-image/edit.",
        source_priority=25,
    ),
    VisionModelDownloadPreset(
        key="qwen-image-edit-2509",
        display_name="Qwen-Image-Edit-2509 MLX-Gen q8",
        repo_id="AbstractFramework/qwen-image-edit-2509-8bit",
        target="mlx",
        engine="mlx-gen",
        local_dir_name="qwen-image-edit-2509-mlx-gen-8bit",
        quantization_bits=8,
        upstream_repo_id="Qwen/Qwen-Image-Edit-2509",
        source=ABSTRACTFRAMEWORK_MLX_GEN_SOURCE,
        aliases=("AbstractFramework/qwen-image-edit-2509-8bit",),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="AbstractFramework MLX-Gen prepared q8 Qwen Image Edit 2509 folder. Prefer when output quality is paramount.",
        source_priority=30,
    ),
    VisionModelDownloadPreset(
        key="qwen-image",
        display_name="Qwen-Image-2512 (Diffusers)",
        repo_id="Qwen/Qwen-Image-2512",
        target="diffusers",
        engine="diffusers",
        local_dir_name="qwen-image-2512-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("qwen-image", "qwen-image-2512", "qwen-image-diffusers", "qwen-image-2512-diffusers"),
        allow_patterns=(
            "model_index.json",
            "*.json",
            "*.md",
            "*.txt",
            "*.yaml",
            "LICENSE*",
            *_DIFFUSERS_PIPELINE_DIRS,
        ),
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=80,
    ),
    VisionModelDownloadPreset(
        key="qwen-image",
        display_name="Qwen-Image (Diffusers, legacy)",
        repo_id="Qwen/Qwen-Image",
        target="diffusers",
        engine="diffusers",
        local_dir_name="qwen-image-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("qwen-image", "qwen-image-legacy", "qwen-image-diffusers-legacy"),
        allow_patterns=(
            "model_index.json",
            "*.json",
            "*.md",
            "*.txt",
            "*.yaml",
            "LICENSE*",
            *_DIFFUSERS_PIPELINE_DIRS,
        ),
        notes="#FALLBACK: legacy Diffusers snapshot; prefer Qwen-Image-2512 when possible.",
        source_priority=90,
    ),
    VisionModelDownloadPreset(
        key="qwen-image-edit-2511",
        display_name="Qwen-Image-Edit-2511 (Diffusers)",
        repo_id="Qwen/Qwen-Image-Edit-2511",
        target="diffusers",
        engine="diffusers",
        local_dir_name="qwen-image-edit-2511-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("qwen-image-edit-2511", "Qwen/Qwen-Image-Edit-2511"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=80,
    ),
    VisionModelDownloadPreset(
        key="qwen-image-edit-2511-gguf",
        display_name="Qwen-Image-Edit-2511 GGUF Q8_0",
        repo_id="unsloth/Qwen-Image-Edit-2511-GGUF",
        target="gguf",
        engine="stable-diffusion.cpp",
        local_dir_name="qwen-image-edit-2511-gguf-q8_0-gguf",
        quantization_bits=8,
        upstream_repo_id="Qwen/Qwen-Image-Edit-2511",
        source="curated-community-gguf",
        aliases=(
            "qwen-image-edit-2511",
            "qwen-image-edit-2511-gguf",
            "qwen-image-edit-2511-gguf-q8_0",
            "unsloth/Qwen-Image-Edit-2511-GGUF",
            "Qwen/Qwen-Image-Edit-2511",
        ),
        allow_patterns=("README.md", "LICENSE*", "qwen-image-edit-2511-Q8_0.gguf"),
        notes=(
            "Qwen Image Edit 2511 GGUF bundle for stable-diffusion.cpp; requires a separate VAE "
            "safetensors file and a Qwen2.5-VL 7B text encoder (safetensors) at runtime."
        ),
        source_priority=30,
    ),
    VisionModelDownloadPreset(
        key="glm-image",
        display_name="GLM-Image (Diffusers)",
        repo_id="zai-org/GLM-Image",
        target="diffusers",
        engine="diffusers",
        local_dir_name="glm-image-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("glm-image", "glm", "zai-org/GLM-Image"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=90,
    ),
    VisionModelDownloadPreset(
        key="stable-diffusion",
        display_name="Stable Diffusion v1.5 (Diffusers)",
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        target="diffusers",
        engine="diffusers",
        local_dir_name="stable-diffusion-v1-5-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=(
            "stable-diffusion",
            "sd",
            "sd15",
            "stable-diffusion-v1-5",
            "runwayml/stable-diffusion-v1-5",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ),
        allow_patterns=_STABLE_DIFFUSION_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=80,
    ),
    VisionModelDownloadPreset(
        key="sd1.4",
        display_name="Stable Diffusion v1.4 (Diffusers)",
        repo_id="CompVis/stable-diffusion-v1-4",
        target="diffusers",
        engine="diffusers",
        local_dir_name="stable-diffusion-v1-4-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("sd1.4", "sd14", "stable-diffusion-v1-4", "CompVis/stable-diffusion-v1-4"),
        allow_patterns=_STABLE_DIFFUSION_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=85,
    ),
    VisionModelDownloadPreset(
        key="sd1.5-inpaint",
        display_name="Stable Diffusion v1.5 Inpainting (Diffusers)",
        repo_id="stable-diffusion-v1-5/stable-diffusion-inpainting",
        target="diffusers",
        engine="diffusers",
        local_dir_name="stable-diffusion-inpainting-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=(
            "sd1.5-inpaint",
            "sd15-inpaint",
            "stable-diffusion-inpainting",
            "runwayml/stable-diffusion-inpainting",
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
        ),
        allow_patterns=_STABLE_DIFFUSION_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=85,
    ),
    VisionModelDownloadPreset(
        key="instruct-pix2pix",
        display_name="InstructPix2Pix (Diffusers)",
        repo_id="timbrooks/instruct-pix2pix",
        target="diffusers",
        engine="diffusers",
        local_dir_name="instruct-pix2pix-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="community",
        aliases=("instruct-pix2pix", "pix2pix", "timbrooks/instruct-pix2pix"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=95,
    ),
    VisionModelDownloadPreset(
        key="sdxl-base",
        display_name="Stable Diffusion XL base 1.0 (Diffusers)",
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        target="diffusers",
        engine="diffusers",
        local_dir_name="sdxl-base-1.0-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=(
            "sdxl-base",
            "sdxl-base-1.0",
            "sdxl",
            "stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=85,
    ),
    VisionModelDownloadPreset(
        key="sdxl-refiner",
        display_name="Stable Diffusion XL refiner 1.0 (Diffusers)",
        repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        target="diffusers",
        engine="diffusers",
        local_dir_name="sdxl-refiner-1.0-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=(
            "sdxl-refiner",
            "sdxl-refiner-1.0",
            "stable-diffusion-xl-refiner-1.0",
            "stabilityai/stable-diffusion-xl-refiner-1.0",
        ),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=85,
    ),
    VisionModelDownloadPreset(
        key="sdxl-inpaint",
        display_name="Stable Diffusion XL 1.0 Inpainting (Diffusers)",
        repo_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        target="diffusers",
        engine="diffusers",
        local_dir_name="sdxl-inpaint-0.1-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("sdxl-inpaint", "sdxl-inpainting", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=90,
    ),
    VisionModelDownloadPreset(
        key="sdxl-turbo",
        display_name="SDXL Turbo (Diffusers)",
        repo_id="stabilityai/sdxl-turbo",
        target="diffusers",
        engine="diffusers",
        local_dir_name="sdxl-turbo-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("sdxl-turbo", "stabilityai/sdxl-turbo"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=85,
    ),
    VisionModelDownloadPreset(
        key="sd-turbo",
        display_name="Stable Diffusion Turbo (Diffusers)",
        repo_id="stabilityai/sd-turbo",
        target="diffusers",
        engine="diffusers",
        local_dir_name="sd-turbo-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("sd-turbo", "stabilityai/sd-turbo"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=85,
    ),
    VisionModelDownloadPreset(
        key="sd3-medium",
        display_name="Stable Diffusion 3 Medium (Diffusers)",
        repo_id="stabilityai/stable-diffusion-3-medium-diffusers",
        target="diffusers",
        engine="diffusers",
        local_dir_name="sd3-medium-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=(
            "sd3-medium",
            "stable-diffusion-3-medium",
            "stable-diffusion-3-medium-diffusers",
            "stabilityai/stable-diffusion-3-medium-diffusers",
        ),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="#FALLBACK: gated on Hugging Face; requires accepting terms + HF token.",
        source_priority=85,
    ),
    VisionModelDownloadPreset(
        key="sd3.5-medium",
        display_name="Stable Diffusion 3.5 Medium (Diffusers)",
        repo_id="stabilityai/stable-diffusion-3.5-medium",
        target="diffusers",
        engine="diffusers",
        local_dir_name="sd3.5-medium-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=(
            "sd3.5-medium",
            "sd35-medium",
            "stable-diffusion-3.5-medium",
            "stabilityai/stable-diffusion-3.5-medium",
        ),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="#FALLBACK: gated on Hugging Face; requires accepting terms + HF token.",
        source_priority=85,
    ),
    VisionModelDownloadPreset(
        key="sd3.5-large",
        display_name="Stable Diffusion 3.5 Large (Diffusers)",
        repo_id="stabilityai/stable-diffusion-3.5-large",
        target="diffusers",
        engine="diffusers",
        local_dir_name="sd3.5-large-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=(
            "sd3.5-large",
            "sd35-large",
            "stable-diffusion-3.5-large",
            "stabilityai/stable-diffusion-3.5-large",
        ),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="#FALLBACK: gated on Hugging Face; requires accepting terms + HF token.",
        source_priority=85,
    ),
    VisionModelDownloadPreset(
        key="sd3.5-large-turbo",
        display_name="Stable Diffusion 3.5 Large Turbo (Diffusers)",
        repo_id="stabilityai/stable-diffusion-3.5-large-turbo",
        target="diffusers",
        engine="diffusers",
        local_dir_name="sd3.5-large-turbo-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=(
            "sd3.5-large-turbo",
            "sd35-large-turbo",
            "stable-diffusion-3.5-large-turbo",
            "stabilityai/stable-diffusion-3.5-large-turbo",
        ),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="#FALLBACK: gated on Hugging Face; requires accepting terms + HF token.",
        source_priority=85,
    ),
    VisionModelDownloadPreset(
        key="flux1-dev",
        display_name="FLUX.1 dev (Diffusers)",
        repo_id="black-forest-labs/FLUX.1-dev",
        target="diffusers",
        engine="diffusers",
        local_dir_name="flux1-dev-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("flux1-dev", "flux-1-dev", "flux.1-dev", "black-forest-labs/FLUX.1-dev"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="#FALLBACK: gated on Hugging Face; requires accepting terms + HF token.",
        source_priority=85,
    ),
    VisionModelDownloadPreset(
        key="flux1-schnell",
        display_name="FLUX.1 schnell (Diffusers)",
        repo_id="black-forest-labs/FLUX.1-schnell",
        target="diffusers",
        engine="diffusers",
        local_dir_name="flux1-schnell-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=(
            "flux1-schnell",
            "flux-1-schnell",
            "flux.1-schnell",
            "black-forest-labs/FLUX.1-schnell",
        ),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=90,
    ),
    VisionModelDownloadPreset(
        key="ernie-image",
        display_name="ERNIE-Image (Diffusers)",
        repo_id="baidu/ERNIE-Image",
        target="diffusers",
        engine="diffusers",
        local_dir_name="ernie-image-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("ernie", "ernie-image", "baidu/ERNIE-Image"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=90,
    ),
    VisionModelDownloadPreset(
        key="ernie-image-turbo",
        display_name="ERNIE-Image-Turbo (Diffusers)",
        repo_id="baidu/ERNIE-Image-Turbo",
        target="diffusers",
        engine="diffusers",
        local_dir_name="ernie-image-turbo-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("ernie-turbo", "ernie-image-turbo", "baidu/ERNIE-Image-Turbo"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="Official curated Diffusers snapshot (16-bit).",
        source_priority=90,
    ),
    VisionModelDownloadPreset(
        key="playground-v2.5",
        display_name="Playground v2.5 1024px aesthetic (Diffusers)",
        repo_id="playgroundai/playground-v2.5-1024px-aesthetic",
        target="diffusers",
        engine="diffusers",
        local_dir_name="playground-v2.5-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("playground-v2.5", "playground-v2.5-1024", "playgroundai/playground-v2.5-1024px-aesthetic"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="#FALLBACK: official Diffusers snapshot (not 8-bit).",
        source_priority=95,
    ),
)


@dataclass(frozen=True)
class _SdcppBundleComponentSpec:
    role: str
    repo_id: str
    allow_patterns: Tuple[str, ...]
    require_weight_files: bool = True


@dataclass(frozen=True)
class _SdcppBundleSpec:
    mode: str  # "single-file" | "component"
    model_patterns: Tuple[str, ...]
    components: Tuple[_SdcppBundleComponentSpec, ...] = ()


@dataclass(frozen=True)
class SdcppModelSelection:
    key: str
    repo_id: str
    capabilities_model_id: Optional[str] = None
    model: Optional[str] = None
    diffusion_model: Optional[str] = None
    vae: Optional[str] = None
    llm: Optional[str] = None
    llm_vision: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SDCPP_BUNDLES: Dict[str, _SdcppBundleSpec] = {
    "flux2-klein-4b": _SdcppBundleSpec(
        mode="component",
        model_patterns=("flux-2-klein-4b-Q8_0.gguf",),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="black-forest-labs/FLUX.2-klein-4B",
                allow_patterns=("vae/diffusion_pytorch_model.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="unsloth/Qwen3-4B-GGUF",
                allow_patterns=("Qwen3-4B-Q4_K_M.gguf",),
            ),
        ),
    ),
    "flux2-klein-base-4b": _SdcppBundleSpec(
        mode="component",
        model_patterns=("flux-2-klein-base-4b-Q8_0.gguf",),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="black-forest-labs/FLUX.2-klein-base-4B",
                allow_patterns=("vae/diffusion_pytorch_model.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="unsloth/Qwen3-4B-GGUF",
                allow_patterns=("Qwen3-4B-Q4_K_M.gguf",),
            ),
        ),
    ),
    "flux2-klein-9b": _SdcppBundleSpec(
        mode="component",
        model_patterns=("flux-2-klein-9b-Q8_0.gguf",),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="black-forest-labs/FLUX.2-klein-9B",
                allow_patterns=("vae/diffusion_pytorch_model.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="unsloth/Qwen3-8B-GGUF",
                allow_patterns=("Qwen3-8B-Q4_K_M.gguf",),
            ),
        ),
    ),
    "flux2-klein-base-9b": _SdcppBundleSpec(
        mode="component",
        model_patterns=("flux-2-klein-base-9b-Q8_0.gguf",),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="black-forest-labs/FLUX.2-klein-base-9B",
                allow_patterns=("vae/diffusion_pytorch_model.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="unsloth/Qwen3-8B-GGUF",
                allow_patterns=("Qwen3-8B-Q4_K_M.gguf",),
            ),
        ),
    ),
    # Z-Image's GGUF repos hold the diffusion model only; stable-diffusion.cpp also needs its
    # Qwen3-4B text encoder and VAE ("new_sd_ctx_t failed" without them). Measured on a Quadro
    # RTX 5000 (sd-cli Vulkan, 2026-11-10): z-image-turbo-Q8_0 with these companions, 512x512.
    "z-image-turbo": _SdcppBundleSpec(
        mode="component",
        model_patterns=("z-image-turbo-Q8_0.gguf",),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="Tongyi-MAI/Z-Image-Turbo",
                allow_patterns=("vae/diffusion_pytorch_model.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="unsloth/Qwen3-4B-GGUF",
                allow_patterns=("Qwen3-4B-Q4_K_M.gguf",),
            ),
        ),
    ),
    "z-image": _SdcppBundleSpec(
        mode="component",
        model_patterns=("z-image-Q8_0.gguf",),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="Tongyi-MAI/Z-Image",
                allow_patterns=("vae/diffusion_pytorch_model.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="unsloth/Qwen3-4B-GGUF",
                allow_patterns=("Qwen3-4B-Q4_K_M.gguf",),
            ),
        ),
    ),
    "qwen-image": _SdcppBundleSpec(
        mode="component",
        model_patterns=("qwen-image-2512-Q8_0.gguf",),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                allow_patterns=("split_files/vae/qwen_image_vae.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                # NOTE: fp8_scaled variants contain tensors not supported by stable-diffusion.cpp and
                # can lead to blank/black outputs. Prefer the full safetensors encoder.
                allow_patterns=("split_files/text_encoders/qwen_2.5_vl_7b.safetensors",),
            ),
        ),
    ),
    "qwen-image-edit-2509": _SdcppBundleSpec(
        mode="component",
        model_patterns=(
            "qwen-image-edit-2509-Q8_0.gguf",
            "Qwen-Image-Edit-2509-Q8_0.gguf",
            "*2509*Q8_0.gguf",
        ),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                allow_patterns=("split_files/vae/qwen_image_vae.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                # NOTE: fp8_scaled variants contain tensors not supported by stable-diffusion.cpp and
                # can lead to blank/black outputs. Prefer the full safetensors encoder.
                allow_patterns=("split_files/text_encoders/qwen_2.5_vl_7b.safetensors",),
            ),
        ),
    ),
    "qwen-image-edit-2511-gguf": _SdcppBundleSpec(
        mode="component",
        model_patterns=("qwen-image-edit-2511-Q8_0.gguf",),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                allow_patterns=("split_files/vae/qwen_image_vae.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                # NOTE: fp8_scaled variants contain tensors not supported by stable-diffusion.cpp and
                # can lead to blank/black outputs. Prefer the full safetensors encoder.
                allow_patterns=("split_files/text_encoders/qwen_2.5_vl_7b.safetensors",),
            ),
        ),
    ),
}


_SDCPP_BUNDLES_BY_REPO_ID: Dict[str, _SdcppBundleSpec] = {
    "unsloth/qwen-image-gguf": _SdcppBundleSpec(
        mode="component",
        model_patterns=(
            "qwen-image-Q8_0.gguf",
            "Qwen-Image-Q8_0.gguf",
            "qwen-image*q8_0.gguf",
            "Qwen-Image*Q8_0.gguf",
        ),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                allow_patterns=("split_files/vae/qwen_image_vae.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                # NOTE: fp8_scaled variants contain tensors not supported by stable-diffusion.cpp and
                # can lead to blank/black outputs. Prefer the full safetensors encoder.
                allow_patterns=("split_files/text_encoders/qwen_2.5_vl_7b.safetensors",),
            ),
        ),
    ),
    "unsloth/qwen-image-edit-gguf": _SdcppBundleSpec(
        mode="component",
        model_patterns=(
            "qwen-image-edit-Q8_0.gguf",
            "Qwen-Image-Edit-Q8_0.gguf",
            "qwen-image-edit*q8_0.gguf",
            "Qwen-Image-Edit*Q8_0.gguf",
        ),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                allow_patterns=("split_files/vae/qwen_image_vae.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="Comfy-Org/Qwen-Image_ComfyUI",
                # NOTE: fp8_scaled variants contain tensors not supported by stable-diffusion.cpp and
                # can lead to blank/black outputs. Prefer the full safetensors encoder.
                allow_patterns=("split_files/text_encoders/qwen_2.5_vl_7b.safetensors",),
            ),
        ),
    ),
}


def _default_allow_patterns(*, target: str, engine: str) -> Tuple[str, ...]:
    if target == "mlx" or engine in {"mflux", "mlx-gen"}:
        return _COMMON_MLX_PATTERNS
    if target == "fp8" or engine == "diffusers-component":
        return ("*.json", "*.md", "LICENSE*", "*.safetensors")
    if target == "gguf" or engine == "stable-diffusion.cpp":
        return ("README.md", "LICENSE*", "*.gguf")
    if target == "diffusers" or engine == "diffusers":
        return _COMMON_DIFFUSERS_PATTERNS
    # Full Hugging Face snapshots should not constrain filenames by default.
    return ()


def _registry_allow_patterns(
    *, key: str, repo_id: str, target: str, engine: str, bits: Optional[int]
) -> Tuple[str, ...]:
    """Download patterns for a registry-declared artifact.

    An 8-bit GGUF artifact downloads only its Q8_0 file: the GGUF repos ship every quantization
    in one repo (unsloth/Qwen-Image-2512-GGUF holds 15 files, 258 GB; the runtime reads the 22 GB
    Q8_0 one), and the stable-diffusion.cpp selection picks the first file the patterns match,
    which for `*.gguf` was the BF16 file (z-image, FLUX.2-dev), not the 8-bit one. A repo with a
    stable-diffusion.cpp bundle uses the bundle's own file names."""

    if str(target) == "gguf" and bits == 8:
        bundle = _SDCPP_BUNDLES_BY_REPO_ID.get(str(repo_id).strip().lower(), _SDCPP_BUNDLES.get(str(key)))
        quant = tuple(bundle.model_patterns) if bundle is not None else ("*Q8_0.gguf", "*q8_0.gguf")
        return ("README.md", "LICENSE*") + quant
    return _default_allow_patterns(target=target, engine=engine)


def _default_source_priority(*, source: str, target: str, engine: str, bits: Optional[int]) -> int:
    source_s = str(source or "").strip().lower()
    if target == "fp8" and bits == 8:
        return 0
    if source_s == ABSTRACTFRAMEWORK_MLX_GEN_SOURCE and target == "mlx" and bits == 4:
        return 20
    if source_s == ABSTRACTFRAMEWORK_MLX_GEN_SOURCE and target == "mlx" and bits == 8:
        return 25
    if source_s == ABSTRACTFRAMEWORK_MLX_GEN_SOURCE and target == "mlx" and bits == 16:
        return 28
    if target == "mlx" and bits == 4:
        return 30
    if target == "mlx" and bits == 8:
        return 35
    if target == "gguf" and bits == 8:
        return 40
    if source_s == "official":
        return 90
    if "community" in source_s:
        return 95
    return 100


def _slugify_token(value: str) -> str:
    raw = str(value or "").strip().lower()
    chars: List[str] = []
    prev_dash = False
    for ch in raw:
        if ch.isalnum():
            chars.append(ch)
            prev_dash = False
            continue
        if not prev_dash:
            chars.append("-")
            prev_dash = True
    slug = "".join(chars).strip("-")
    return slug or "model"


def _default_local_dir_name(
    *,
    key: str,
    target: str,
    engine: str,
    repo_id: str,
    bits: Optional[int] = None,
) -> str:
    base = _slugify_token(key or repo_id.rsplit("/", 1)[-1])
    suffix = _slugify_token(target or engine)
    if bits is not None and str(target or "").strip().lower() in {"mlx", "gguf", "fp8"}:
        engine_part = _slugify_token(engine or target)
        suffix = f"{engine_part}-{int(bits)}bit"
    if base.endswith(f"-{suffix}") or base == suffix:
        return base
    return f"{base}-{suffix}"


def _default_display_name(*, model_id: str, target: str, engine: str, bits: Optional[int]) -> str:
    stem = str(model_id or "").rsplit("/", 1)[-1].replace("-", " ").strip() or str(model_id)
    if str(model_id).strip() in {"Qwen/Qwen-Image", "Qwen/Qwen-Image-Edit"}:
        stem = f"{stem} (legacy)"
    engine_label = {
        "diffusers": "Diffusers",
        "diffusers-component": "Diffusers component",
        "mflux": "MLX-Gen",
        "mlx-gen": "MLX-Gen",
        "stable-diffusion.cpp": "stable-diffusion.cpp",
        "transformers": "Transformers",
    }.get(str(engine), str(engine or target or "download"))
    if bits is not None:
        return f"{stem} ({engine_label}, {int(bits)}-bit)"
    return f"{stem} ({engine_label})"


def _merge_aliases(*values: str) -> Tuple[str, ...]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        candidates = [str(value or "").strip()]
        if "/" in candidates[0]:
            candidates.append(candidates[0].rsplit("/", 1)[-1])
        for s in candidates:
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
    return tuple(out)


def _is_abstractframework_mlx_gen_preset(preset: VisionModelDownloadPreset) -> bool:
    return (
        str(preset.target).strip().lower() == "mlx"
        and str(preset.engine).strip().lower().replace("_", "-") in {"mlx-gen", "mflux", "m-flux", "mlxgen"}
        and str(preset.source).strip().lower() == ABSTRACTFRAMEWORK_MLX_GEN_SOURCE
        and str(preset.repo_id).strip().lower().startswith("abstractframework/")
    )


def _exact_repo_selectors(preset: VisionModelDownloadPreset) -> set[str]:
    repo = str(preset.repo_id or "").strip().lower()
    if not repo:
        return set()
    return {repo, repo.rsplit("/", 1)[-1]}


@lru_cache(maxsize=1)
def _all_presets() -> Tuple[VisionModelDownloadPreset, ...]:
    out: List[VisionModelDownloadPreset] = list(_PRESETS)
    seen = {(p.key, p.engine, p.target, p.repo_id) for p in out}

    try:
        reg = VisionModelCapabilitiesRegistry()
    except Exception:
        return tuple(sorted(out, key=lambda p: (p.key, p.source_priority, p.repo_id)))

    recommended: set[tuple[str, str, str, str]] = set()
    gated: set[str] = set()
    explicit: set[str] = set()
    license_names: Dict[str, str] = {}
    for model_id in reg.list_models():
        spec = reg.get(model_id)
        for dl in spec.downloads:
            key = (dl.key, dl.engine, dl.target, dl.repo_id)
            if dl.recommended:
                recommended.add(key)
            if dl.gated:
                gated.add(str(dl.repo_id))
            if dl.explicit_download:
                explicit.add(str(dl.repo_id))
            if spec.license_name:
                license_names.setdefault(str(dl.repo_id), spec.license_name)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                VisionModelDownloadPreset(
                    key=str(dl.key),
                    display_name=_default_display_name(
                        model_id=model_id,
                        target=str(dl.target),
                        engine=str(dl.engine),
                        bits=dl.bits,
                    ),
                    repo_id=str(dl.repo_id),
                    target=str(dl.target),
                    engine=str(dl.engine),
                    local_dir_name=_default_local_dir_name(
                        key=str(dl.key),
                        target=str(dl.target),
                        engine=str(dl.engine),
                        repo_id=str(dl.repo_id),
                        bits=dl.bits,
                    ),
                    quantization_bits=dl.bits,
                    upstream_repo_id=None if str(dl.repo_id) == str(model_id) else str(model_id),
                    source=str(dl.source or "official"),
                    aliases=_merge_aliases(str(dl.key), str(model_id), str(dl.repo_id)),
                    allow_patterns=_registry_allow_patterns(
                        key=str(dl.key),
                        repo_id=str(dl.repo_id),
                        target=str(dl.target),
                        engine=str(dl.engine),
                        bits=dl.bits,
                    ),
                    notes=str(dl.notes or spec.notes or ""),
                    source_priority=_default_source_priority(
                        source=str(dl.source or "official"),
                        target=str(dl.target),
                        engine=str(dl.engine),
                        bits=dl.bits,
                    ),
                )
            )

    enriched = [
        replace(
            p,
            gated=p.repo_id in gated,
            explicit_download=p.repo_id in explicit,
            recommended=(p.key, p.engine, p.target, p.repo_id) in recommended,
            license_name=license_names.get(p.repo_id, ""),
        )
        for p in out
    ]
    return tuple(sorted(enriched, key=lambda p: (p.key, p.source_priority, p.repo_id)))


@lru_cache(maxsize=1)
def local_model_profile() -> str:
    machine = str(platform.machine() or "").strip().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "apple-silicon"

    # `nvidia-smi` lists every GPU whatever `CUDA_VISIBLE_DEVICES` says, but an empty value (or
    # "-1") hides them all from CUDA: PyTorch then runs on the processor, so the catalog must
    # not offer GPU artifacts (the Diffusers backend's device "auto" already resolves to cpu).
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    nvidia_smi = shutil.which("nvidia-smi")
    if cuda_visible is not None and cuda_visible.strip() in {"", "-1"}:
        nvidia_smi = None
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [nvidia_smi, "-L"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            if proc.returncode == 0 and "gpu" in str(proc.stdout or "").lower():
                return "cuda"
        except Exception:
            pass

    if sys.platform == "darwin":
        return "macos"
    return "cpu"


def local_catalog_targets() -> Tuple[str, ...]:
    profile = local_model_profile()
    if profile == "apple-silicon":
        # Apple Silicon can run GGUF models via stable-diffusion.cpp (Metal). Operators who
        # want to hide these entries can set `ABSTRACTVISION_DISABLE_GGUF_ON_MACOS=1`.
        return ("mlx", "diffusers", "hf-snapshot") if _macos_gguf_disabled() else ("mlx", "gguf", "diffusers", "hf-snapshot")
    if profile == "cuda":
        # No fp8 in the auto order: those repos are component side artifacts no AbstractVision
        # backend loads (FLUX.2-klein-4b-fp8 is "components only; not a standalone Diffusers
        # pipeline"). Moving it last is not enough: downloads are 8-bit by default, and for
        # flux1-dev or flux2-klein-9b-kv the FP8 repo is the only 8-bit artifact, so it would still
        # be the auto pick. `--target fp8` (or the repo id) still reaches every FP8 artifact.
        return ("gguf", "diffusers", "hf-snapshot")
    return ("diffusers", "hf-snapshot", "gguf")


def catalog_target_scope(
    *,
    target: Optional[str] = "auto",
    engine: Optional[str] = None,
    include_all_targets: bool = False,
) -> Tuple[str, ...]:
    raw_target = str(target or "auto").strip().lower()
    selected_target, selected_engine = resolve_model_target_and_engine(target=target, engine=engine)
    if include_all_targets:
        preferred = list(local_catalog_targets())
        for preset in _all_presets():
            if _macos_gguf_disabled() and preset.target == "gguf":
                continue
            if preset.target not in preferred:
                preferred.append(preset.target)
        return tuple(preferred)
    if raw_target in {"", "auto", "default"} and selected_engine == "diffusers":
        return ("diffusers",) if _macos_gguf_disabled() else ("diffusers", "gguf")
    if raw_target in {"", "auto", "default"} and selected_engine is None:
        return local_catalog_targets()
    return (selected_target,)


def default_model_target() -> str:
    return local_catalog_targets()[0]


def normalize_model_target(target: Optional[str]) -> str:
    value = str(target or "auto").strip().lower()
    if value in {"", "auto", "default"}:
        return default_model_target()
    aliases = {
        "apple": "mlx",
        "mac": "mlx",
        "macos": "mlx",
        "osx": "mlx",
        "metal": "mlx",
        "gpu": "gguf",
        "cuda": "gguf",
        "nvidia": "gguf",
        "stable-diffusion.cpp": "gguf",
        "sdcpp": "gguf",
    }
    return aliases.get(value, value)


def normalize_model_engine(engine: Optional[str]) -> Optional[str]:
    value = str(engine or "").strip().lower()
    if value in {"", "auto", "any", "default", "*"}:
        return None
    if value == "mlx":
        raise ValueError(_GENERIC_MLX_BACKEND_ERROR)
    aliases = {
        "stable-diffusion-cpp": "stable-diffusion.cpp",
        "stable_diffusion_cpp": "stable-diffusion.cpp",
        "sd-cpp": "stable-diffusion.cpp",
        "sdcpp": "stable-diffusion.cpp",
        "gguf": "stable-diffusion.cpp",
        "mflux": "mlx-gen",
        "m-flux": "mlx-gen",
        "metal": "mlx-gen",
        "apple": "mlx-gen",
        "mac": "mlx-gen",
        "macos": "mlx-gen",
        "osx": "mlx-gen",
        "mlx-gen": "mlx-gen",
        "mlxgen": "mlx-gen",
        "mlx_gen": "mlx-gen",
    }
    return aliases.get(value, value)


def resolve_model_target_and_engine(
    *,
    target: Optional[str] = "auto",
    engine: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Resolve a target/engine pair for preset selection.

    When ``target`` is ``auto`` (or empty) and an explicit engine is supplied,
    infer the only compatible artifact target. This keeps UX simple so users can
    specify either ``--target`` or ``--provider/--engine`` without repeating
    themselves (e.g. ``--provider diffusers`` implies ``--target diffusers``).
    """

    raw_target = str(target or "auto").strip().lower()
    selected_engine = normalize_model_engine(engine)
    if raw_target in {"", "auto", "default"} and selected_engine is not None:
        engine_to_target = {
            "mflux": "mlx",
            "mlx-gen": "mlx",
            "diffusers": "diffusers",
            "stable-diffusion.cpp": "gguf",
            "diffusers-component": "fp8",
            "transformers": "hf-snapshot",
        }
        inferred = engine_to_target.get(selected_engine)
        if inferred:
            if inferred == "gguf" and _macos_gguf_disabled():
                _raise_macos_gguf_disabled()
            return inferred, selected_engine
    selected_target = normalize_model_target(target)
    if selected_target == "gguf" and _macos_gguf_disabled():
        _raise_macos_gguf_disabled()
    return selected_target, selected_engine


def model_presets(
    *,
    target: Optional[str] = "auto",
    engine: Optional[str] = None,
    include_non_8bit: bool = False,
    include_all_targets: bool = False,
) -> List[VisionModelDownloadPreset]:
    selected_targets = set(catalog_target_scope(target=target, engine=engine, include_all_targets=include_all_targets))
    selected_engine = resolve_model_target_and_engine(target=target, engine=engine)[1]
    out: List[VisionModelDownloadPreset] = []
    for preset in _all_presets():
        if _macos_gguf_disabled() and preset.target == "gguf":
            continue
        if preset.target not in selected_targets:
            continue
        if selected_engine is not None and preset.engine != selected_engine:
            continue
        if not include_non_8bit and not _is_default_quantized_preset(preset):
            continue
        out.append(preset)
    return sorted(out, key=lambda p: (p.key, p.source_priority, p.repo_id))


def _is_default_quantized_preset(preset: VisionModelDownloadPreset) -> bool:
    bits = preset.quantization_bits
    if preset.target == "mlx" and preset.engine in {"mflux", "mlx-gen"}:
        return bits in {2, 4, 8}
    return bits == 8


def _pick_package(
    name: str, matches: Sequence[VisionModelDownloadPreset], *, bits: Optional[int]
) -> Optional[VisionModelDownloadPreset]:
    """One preset among several packages a model selector matches, or None when it stays ambiguous.

    `bits` names the width; without it the package the catalog marks `recommended` is chosen."""

    if bits is not None:
        width = [p for p in matches if int(p.quantization_bits or 0) == int(bits)]
        if not width:
            available = ", ".join(
                sorted({f"{p.repo_id} ({p.quantization_bits}-bit)" for p in matches})
            )
            raise ValueError(f"No {int(bits)}-bit package for {name!r}. Available: {available}")
        matches = width
    if len({p.repo_id for p in matches}) == 1:
        return sorted(matches, key=lambda p: (p.source_priority, p.repo_id))[0]
    flagged = [p for p in matches if p.recommended]
    if len({p.repo_id for p in flagged}) == 1:
        return flagged[0]
    return None


def find_model_preset(
    name: str,
    *,
    target: Optional[str] = "auto",
    engine: Optional[str] = None,
    require_8bit: bool = True,
    bits: Optional[int] = None,
    download_rule: bool = False,
) -> VisionModelDownloadPreset:
    """The catalog preset `name` selects for a target and engine.

    Across targets the host's order decides. Within the chosen target, several packages select the one
    the catalog recommends; without one, the catalog's own order picks (a curated key such as
    `qwen-image` resolves to its preferred pipeline, as a backend configured with it does).
    `download_rule` applies the `download` command's rule instead: a repository named exactly is that
    repository, and several packages with none recommended raise ValueError listing them."""

    raw_target = str(target or "auto").strip().lower()
    requested = str(name or "").strip().lower()
    if requested.startswith("mlx/"):
        raise ValueError(_GENERIC_MLX_BACKEND_ERROR)
    for prefix in (
        "mflux/",
        "m-flux/",
        "mlx-gen/",
        "mlxgen/",
        "diffusers/",
        "huggingface/",
        "hf/",
        "sdcpp/",
        "sd-cpp/",
        "stable-diffusion-cpp/",
        "stable-diffusion.cpp/",
    ):
        if requested.startswith(prefix):
            requested = requested[len(prefix) :].strip()
            break
    selected_targets = catalog_target_scope(target=target, engine=engine, include_all_targets=False)
    selected_engine = resolve_model_target_and_engine(target=target, engine=engine)[1]
    target_rank = {name: idx for idx, name in enumerate(selected_targets)}
    selected_target_label = ",".join(selected_targets)

    def matches(preset: VisionModelDownloadPreset) -> bool:
        aliases = {a.lower() for a in preset.aliases}
        repo_ids = {preset.repo_id.lower()}
        repo_ids.add(preset.repo_id.rsplit("/", 1)[-1].lower())
        if preset.upstream_repo_id:
            repo_ids.add(preset.upstream_repo_id.lower())
            repo_ids.add(preset.upstream_repo_id.rsplit("/", 1)[-1].lower())
        return requested == preset.key or requested in aliases or requested in repo_ids

    presets = _all_presets()

    if "mlx" in selected_targets and (selected_engine is None or selected_engine in {"mlx-gen", "mflux"}):
        exact_non_abstractframework_matches = [
            p
            for p in presets
            if not _is_abstractframework_mlx_gen_preset(p)
            and p.target in selected_targets
            and (selected_engine is None or p.engine == selected_engine)
            and "/" in requested
            and requested == str(p.repo_id or "").strip().lower()
        ]
        if exact_non_abstractframework_matches and not require_8bit:
            return sorted(
                exact_non_abstractframework_matches,
                key=lambda p: (target_rank.get(p.target, len(target_rank)), p.source_priority, p.repo_id),
            )[0]
        mlx_gen_matches = [
            p
            for p in presets
            if _is_abstractframework_mlx_gen_preset(p)
            and p.target in selected_targets
            and (selected_engine is None or p.engine == selected_engine)
            and matches(p)
        ]
        exact_matches = [p for p in mlx_gen_matches if requested in _exact_repo_selectors(p)]
        if exact_matches:
            chosen = sorted(
                exact_matches,
                key=lambda p: (target_rank.get(p.target, len(target_rank)), p.source_priority, p.repo_id),
            )[0]
            if bits is not None and int(chosen.quantization_bits or 0) != int(bits):
                raise ValueError(
                    f"{chosen.repo_id} is the {chosen.quantization_bits}-bit package; drop --bits or "
                    f"name the {int(bits)}-bit package."
                )
            return chosen
        if mlx_gen_matches and (bits is not None or any(p.recommended for p in mlx_gen_matches)):
            picked = _pick_package(name, mlx_gen_matches, bits=bits)
            if picked is not None:
                return picked
        if mlx_gen_matches:
            repo_choices = {p.repo_id for p in mlx_gen_matches}
            if len(repo_choices) == 1:
                return sorted(
                    mlx_gen_matches,
                    key=lambda p: (target_rank.get(p.target, len(target_rank)), p.source_priority, p.repo_id),
                )[0]
            available = ", ".join(sorted({p.repo_id for p in mlx_gen_matches}))
            source = (
                f"; the source repository {exact_non_abstractframework_matches[0].repo_id}: add --allow-non-8bit"
                if exact_non_abstractframework_matches
                else ""
            )
            raise ValueError(
                f"{name!r} names several MLX-Gen packages and the catalog recommends none of them: "
                f"name one of {available}, or pass --bits{source}."
            )

    def _sort_key(preset: VisionModelDownloadPreset) -> tuple[int, int, str]:
        return (target_rank.get(preset.target, len(target_rank)), preset.source_priority, preset.repo_id)

    def _choose(choices: Sequence[VisionModelDownloadPreset]) -> VisionModelDownloadPreset:
        # A repository named exactly is that repository. Otherwise the host's target order picks the
        # target; within it, several packages select the one the catalog recommends, or stop and are
        # listed, whatever narrowed them.
        ordered = sorted(choices, key=_sort_key)
        exact = [p for p in ordered if p.repo_id.lower() == requested]
        if download_rule and exact:
            return exact[0]
        top = [p for p in ordered if p.target == ordered[0].target]
        repos = sorted({p.repo_id for p in top})
        if len(repos) == 1:
            return ordered[0]
        flagged = [p for p in top if p.recommended]
        if len({p.repo_id for p in flagged}) == 1:
            return flagged[0]
        if not download_rule:
            return ordered[0]
        widths = {p.quantization_bits for p in top}
        raise ValueError(
            f"{name!r} names several {ordered[0].target} packages and the catalog recommends none of them: "
            f"name one of {', '.join(repos)}" + (", or pass --bits." if len(widths) > 1 else ".")
        )

    candidates = [
        p
        for p in presets
        if p.target in selected_targets
        and (selected_engine is None or p.engine == selected_engine)
        and matches(p)
    ]
    if require_8bit and bits is None:
        candidates = [p for p in candidates if _is_default_quantized_preset(p)]
    if bits is not None:
        candidates = [p for p in candidates if int(p.quantization_bits or 0) == int(bits)]
    if candidates:
        return _choose(candidates)

    any_engine_matches = [p for p in presets if matches(p)]
    target_matches = [
        p
        for p in presets
        if p.target in selected_targets and (selected_engine is None or p.engine == selected_engine) and matches(p)
    ]

    if raw_target in {"", "auto", "default"} and selected_engine is None and any_engine_matches:
        # If the caller left target/engine on defaults, pick a sensible fallback
        # when the resolved preset key only maps to one concrete artifact target.
        # This keeps platform-aware catalog defaults while still allowing exact
        # repo ids to resolve to their unique curated preset on non-default hosts.
        possible_targets = sorted({p.target for p in any_engine_matches})
        if len(possible_targets) == 1:
            fallback_target = possible_targets[0]
            fallback = [p for p in any_engine_matches if p.target == fallback_target]
            packages = [p for p in fallback if _is_abstractframework_mlx_gen_preset(p)]
            exact = [p for p in fallback if p.repo_id.lower() == requested]
            if download_rule and not require_8bit and exact and bits is None:
                # As on a host whose targets include it: a repository named exactly is that repository.
                return sorted(exact, key=_sort_key)[0]
            if packages and (bits is not None or any(p.recommended for p in packages)):
                picked = _pick_package(name, packages, bits=bits)
                if picked is not None:
                    return picked
            if require_8bit:
                quantized = [p for p in fallback if _is_default_quantized_preset(p)]
                if quantized:
                    fallback = quantized
            return _choose(fallback)

    if target_matches and bits is None and len({p.repo_id for p in target_matches}) == 1:
        # One repository: a model with a single package selects it, whatever its width.
        return sorted(target_matches, key=_sort_key)[0]
    if target_matches:
        available = ", ".join(sorted({f"{p.target}/{p.engine}:{p.repo_id}" for p in target_matches}))
        engine_msg = f" and engine {selected_engine!r}" if selected_engine else ""
        raise ValueError(
            f"No {'8-bit ' if require_8bit else ''}preset for {name!r} on target {selected_target_label!r}{engine_msg}. "
            f"Available target/repo choices: {available}"
        )
    if any_engine_matches:
        available = ", ".join(sorted({f"{p.target}/{p.engine}:{p.repo_id}" for p in any_engine_matches}))
        if selected_engine:
            raise ValueError(
                f"No {'8-bit ' if require_8bit else ''}preset for {name!r} with engine {selected_engine!r}. "
                f"Available target/engine choices: {available}"
            )
        raise ValueError(
            f"No {'8-bit ' if require_8bit else ''}preset for {name!r} on target {selected_target_label!r}. "
            f"Available target/engine choices: {available}"
        )
    raise ValueError(
        f"{name!r} is not in the AbstractVision catalog. `abstractvision catalog` lists its models "
        "(`--all-targets --all` for every package); a Hugging Face repository outside the catalog is named "
        "`owner/name`."
    )


def catalog_download_preset(repo_id: str, *, target: str, engine: str) -> Optional[VisionModelDownloadPreset]:
    """The download preset for one repository, target and engine, or None."""

    for preset in _all_presets():
        if (preset.repo_id, preset.target, preset.engine) == (str(repo_id), str(target), str(engine)):
            return preset
    return None


def catalog_names_model(name: str) -> bool:
    """Whether `name` is a key, alias, repository or upstream model id of any download preset."""

    requested = str(name or "").strip().lower()
    if not requested:
        return False
    for preset in _all_presets():
        names = {str(preset.key).lower(), str(preset.repo_id).lower(), *(a.lower() for a in preset.aliases)}
        if preset.upstream_repo_id:
            names.add(str(preset.upstream_repo_id).lower())
        if requested in names:
            return True
    return False


def default_download_root() -> Path:
    """Return the legacy preset root used for migration input only."""

    return default_legacy_model_root()


def looks_like_hf_repo_id(value: Any) -> bool:
    """Return True when ``value`` resembles a Hugging Face repo id like ``org/name``.

    This intentionally keeps the heuristic small and conservative so local paths
    (e.g. ``./models`` or ``~/models``) are not misclassified as repo ids.
    """

    s = str(value or "").strip()
    if not s:
        return False
    if s.startswith(("/", "./", "../", "~")):
        return False
    if "\\" in s or " " in s:
        return False
    return s.count("/") == 1 and all(part.strip() for part in s.split("/", 1))


def resolve_hf_token(token: Optional[str] = None) -> Optional[str]:
    explicit = str(token or "").strip()
    if explicit:
        return explicit
    for key in ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"):
        value = str(os.environ.get(key) or "").strip()
        if value:
            return value
    return None


def repository_preset(repo_id: str) -> VisionModelDownloadPreset:
    """A Hugging Face repository the catalog does not list: every file it holds, no engine chosen.

    It downloads through the same plan, size and disk check as a catalog package; its access comes
    from the Hub listing."""

    repo = str(repo_id).strip()
    return VisionModelDownloadPreset(
        key=repo,
        display_name=repo,
        repo_id=repo,
        target="hf-snapshot",
        engine="",
        local_dir_name="",
        quantization_bits=None,
        upstream_repo_id=None,
        # What `download --json` reports as the source of a repository outside the catalog.
        source="huggingface_hub_cache",
        aliases=(),
        allow_patterns=(),
        in_catalog=False,
    )


_MLX_GEN_ENGINE_NAMES = {"mflux", "m-flux", "mlx-gen", "mlxgen"}
_MLX_TARGET_NAMES = {"mlx", "apple", "mac", "macos", "osx", "metal"}


def select_download_preset(
    name: str,
    *,
    target: Optional[str] = "auto",
    engine: Optional[str] = None,
    bits: Optional[int] = None,
    allow_non_8bit: bool = False,
) -> VisionModelDownloadPreset:
    """The preset `abstractvision download <name>` fetches (`--target`, `--provider`, `--bits`,
    `--allow-non-8bit`); ValueError says why there is none.

    A catalog model with a single package selects it; with several packages of one engine, the one the
    catalog recommends, else `bits` picks by width and otherwise the packages are listed. Across
    targets the host's order decides, preferring an 8-bit package unless `allow_non_8bit`; with
    `allow_non_8bit` a repository named exactly selects that repository instead of a package made from it.
    A Diffusers or snapshot target, or an MLX-Gen engine or target named explicitly, does not require 8 bits
    (so `briaai/FIBO --provider mlx-gen` is that source repository, as in 0.3.36). A
    Hugging Face `owner/name` the catalog does not name is the whole repository (`repository_preset`)."""

    raw_target = str(target or "auto").strip().lower()
    raw_engine = str(engine or "auto").strip().lower().replace("_", "-")
    require_8bit = not (
        allow_non_8bit
        or raw_target in {"diffusers", "hf-snapshot", *_MLX_TARGET_NAMES}
        or raw_engine in {"diffusers", "transformers", *_MLX_GEN_ENGINE_NAMES}
    )
    try:
        return find_model_preset(
            name, target=target, engine=engine, require_8bit=require_8bit, bits=bits, download_rule=True
        )
    except ValueError as refusal:
        if bits is None and not catalog_names_model(name) and looks_like_hf_repo_id(name):
            return repository_preset(name)
        raise refusal


def download_command(repo_id: str, *, engine: Optional[str] = None, base_model: Optional[str] = None) -> str:
    """The `abstractvision download` command that fetches exactly `repo_id` on this host (its `engine`
    package when the catalog lists the repository for several), with `--base-model` when given.

    Each form of the command is checked with `select_download_preset`, so the command printed is one
    that selects this repository: a model id that is also a source repository needs `--allow-non-8bit`,
    because the model id alone selects its recommended package. ValueError when no form selects it, when
    the catalog lists the repository but not for `engine` (no command would fetch what that engine loads),
    or when `base_model` is not a schedule of that package. A repository outside the catalog is fetched
    whole, whatever the engine."""

    repo = str(repo_id).strip()
    wanted_engine = normalize_model_engine(engine) if engine else None
    in_catalog = [p for p in _all_presets() if p.repo_id == repo]
    listed = [p for p in in_catalog if wanted_engine is None or p.engine == wanted_engine]
    if in_catalog and not listed:
        engines = ", ".join(sorted({p.engine for p in in_catalog}))
        raise ValueError(f"The catalog lists {repo} for {engines}, not for {wanted_engine}.")
    if base_model is not None:
        if not listed or listed[0].engine != "mlx-gen":
            raise ValueError(f"--base-model names an MLX-Gen schedule; {repo} is not an MLX-Gen package of the catalog.")
        from .backends.mflux import mlx_gen_package_schedule

        base_model = mlx_gen_package_schedule(listed[0], base_model)
    widths = sorted({int(p.quantization_bits) for p in listed if p.quantization_bits in (2, 4, 8, 16)})
    forms: List[Tuple[str, ...]] = [(), ("--allow-non-8bit",), *[("--bits", str(w)) for w in widths]]
    if wanted_engine is not None:
        forms += [("--provider", wanted_engine, *flags) for flags in list(forms)]
    for flags in forms:
        provider = flags[1] if flags[:1] == ("--provider",) else None
        rest = flags[2:] if provider else flags
        bits = int(rest[1]) if rest[:1] == ("--bits",) else None
        try:
            chosen = select_download_preset(
                repo, engine=provider, bits=bits, allow_non_8bit="--allow-non-8bit" in rest
            )
        except ValueError:
            continue
        if chosen.repo_id != repo or (listed and chosen.engine not in {p.engine for p in listed}):
            continue
        schedule = ("--base-model", str(base_model)) if base_model is not None else ()
        return shlex.join(("abstractvision", "download", repo, *flags, *schedule))
    raise ValueError(f"No `abstractvision download` command selects {repo} on this host.")


def _hub_refusal_status(exc: BaseException) -> Optional[int]:
    """The HTTP status when the Hub refused the repository (401, 403, or a repository it does not
    show), else None. Read from the response and the exception type, never from the message."""

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return int(status)
    try:
        # `huggingface_hub.utils` exports it in 0.x and 1.x; `huggingface_hub.errors` only from 0.25.
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        return None
    if isinstance(exc, RepositoryNotFoundError):
        return int(status) if isinstance(status, int) else 404
    return None


def hub_import_problem(error: ImportError) -> str:
    """What is missing when huggingface_hub cannot be imported, from the error's module `name`: the
    library itself, or a module it imports."""

    missing = str(getattr(error, "name", "") or "")
    if not missing or missing.split(".")[0] == "huggingface_hub":
        what = "huggingface_hub is not installed"
    else:
        what = f"huggingface_hub cannot be imported, because the module {missing!r} it imports is not installed"
    return f'{what}; install it with `pip install "abstractvision[models]"`'


def _hub_endpoint() -> str:
    try:
        from huggingface_hub import constants

        return str(constants.ENDPOINT)
    except Exception:
        return "https://huggingface.co"


def _hub_unavailable_reason(
    exc: BaseException, repo_id: str, *, gated: bool, license_name: str, revision: Optional[str] = None
) -> str:
    """Why a Hub request failed and what to do, for a plan without a listing; read from the exception type
    and HTTP status, never from the message. A refusal raises HuggingFaceAccessError.

    huggingface_hub's own types decide: `OfflineModeIsEnabled`, then `HfHubHTTPError` (the Hub answered
    with an error status), then a request that got no answer: the standard library's `OSError` family
    (`ConnectionError`, `TimeoutError`) or any error of the base classes `HfHubHTTPError` derives from,
    which are its transport library's own error base (httpx's in huggingface_hub 1.x, requests' in 0.x)."""

    status = _hub_refusal_status(exc)
    if status is not None:
        raise HuggingFaceAccessError(
            str(repo_id), raw_message=str(exc), status_code=status, gated=gated, license_name=license_name
        ) from exc
    first_line = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
    detail = f"{type(exc).__name__}: {first_line}" if first_line else type(exc).__name__
    try:
        from huggingface_hub.utils import (
            HfHubHTTPError,
            OfflineModeIsEnabled,
            RevisionNotFoundError,
        )
    except ImportError:
        return detail
    if isinstance(exc, OfflineModeIsEnabled):
        return "offline mode is on (HF_HUB_OFFLINE), so the Hugging Face Hub was not contacted; unset HF_HUB_OFFLINE to download"
    if isinstance(exc, RevisionNotFoundError):
        named = f" {revision!r}" if revision else ""
        return f"{repo_id} has no revision{named} on the Hugging Face Hub at {_hub_endpoint()}; name an existing branch, tag or commit"
    if isinstance(exc, HfHubHTTPError):
        code = getattr(getattr(exc, "response", None), "status_code", None)
        return f"the Hugging Face Hub at {_hub_endpoint()} answered HTTP {code}; try again later"
    if isinstance(exc, (OSError, *HfHubHTTPError.__bases__)):
        return f"the Hugging Face Hub could not be reached at {_hub_endpoint()} ({detail}); check the network connection"
    return detail


def _catalog_access(repo_id: str) -> Tuple[Optional[bool], str]:
    """(gated, licence name) of a repository as the catalog gives them; (None, "") when it is not listed."""

    listed = [p for p in _all_presets() if p.repo_id == str(repo_id)]
    if not listed:
        return None, ""
    gated = [p for p in listed if p.gated]
    return (True, gated[0].license_name) if gated else (False, "")


def _preset_access(preset: VisionModelDownloadPreset) -> Tuple[Optional[bool], str]:
    if not preset.in_catalog:
        return None, ""
    return bool(preset.gated), str(preset.license_name or "")


def snapshot_requirements_for_preset(
    preset: VisionModelDownloadPreset,
) -> tuple[Tuple[str, ...], bool]:
    if not preset.in_catalog:
        return tuple(), False
    target = str(preset.target or "").strip().lower()
    engine = str(preset.engine or "").strip().lower()
    if target in {"hf-snapshot", "mlx", "gguf", "fp8"}:
        return tuple(), True
    if target == "diffusers" or engine == "diffusers":
        return ("model_index.json",), True
    if engine in {"transformers", "mflux", "stable-diffusion.cpp", "diffusers-component"}:
        return tuple(), True
    return tuple(), False


def _existing_parent(path: Path) -> Path:
    current = Path(path).expanduser()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _free_bytes(cache_root: Path) -> Optional[int]:
    try:
        return int(shutil.disk_usage(_existing_parent(cache_root)).free)
    except Exception:
        return None


def _hub_json(repo_id: str, filename: str, *, revision: Optional[str], token: Optional[str]) -> Any:
    """A small JSON file of a repository, read from the Hub into memory: the cache is not written.

    The request goes through huggingface_hub's own session, as its `HfApi` calls do, with only the
    arguments both of its session types take (`requests.Session` in 0.x, `httpx.Client` in 1.x); each
    follows redirects as huggingface_hub configures it."""

    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

    response = get_session().get(
        hf_hub_url(str(repo_id), filename, revision=revision),
        headers=build_hf_headers(token=resolve_hf_token(token)),
    )
    hf_raise_for_status(response)
    return response.json()


def model_index_components(index: Any) -> List[str]:
    """The pipeline components a Diffusers `model_index.json` names (entries with a library and class)."""

    if not isinstance(index, dict):
        return []
    return [
        str(k)
        for k, v in index.items()
        if not str(k).startswith("_") and isinstance(v, list) and len(v) == 2 and v[0] is not None
    ]


def _hub_file(repo_dir: Path, commit: str, sibling: Any) -> HubFile:
    # huggingface_hub's own layout (`hf_hub_download`): a file is cached when its snapshot link or its
    # blob exists, and in flight as `blobs/<etag>.incomplete`, the file it writes into.
    lfs = getattr(sibling, "lfs", None)
    etag = str(getattr(lfs, "sha256", "") or getattr(sibling, "blob_id", "") or "")
    blob = repo_dir / "blobs" / etag if etag else None
    complete = (repo_dir / "snapshots" / commit / str(sibling.rfilename)).exists() or (
        blob is not None and blob.exists()
    )
    partial = 0
    if not complete and blob is not None:
        try:
            partial = blob.with_name(blob.name + ".incomplete").stat().st_size
        except OSError:
            partial = 0
    return HubFile(name=str(sibling.rfilename), size=int(sibling.size), complete=complete, partial=partial)


def _plan_hub_download(
    repo_id: str,
    *,
    cache_root: Path,
    token: Optional[str],
    allow_patterns: Optional[Sequence[str]],
    ignore_patterns: Optional[Sequence[str]],
    diffusers_trim: bool,
    gated: Optional[bool],
    license_name: str,
    revision: Optional[str] = None,
) -> HubDownloadPlan:
    """Size a download against the cache before it starts, from the Hub listing (metadata only).

    The files the patterns select (for a Diffusers pipeline with `diffusers_trim`, only those the
    Diffusers backend loads, read from its `model_index.json` without writing to the cache) are
    each complete in the cache or to fetch. A refusal raises `HuggingFaceAccessError`; a listing
    that cannot be had otherwise gives a plan with `listing_error`. `gated` is what the catalog
    says (None: not in the catalog, the listing decides)."""

    repo_id = str(repo_id)
    free_bytes = _free_bytes(cache_root)

    def unlisted(reason: str, access: Optional[bool]) -> HubDownloadPlan:
        return HubDownloadPlan(
            repo_id=repo_id,
            cache_dir=cache_root,
            free_bytes=free_bytes,
            gated=access,
            license_name=license_name,
            listing_error=reason,
            revision=str(revision) if revision else None,
        )

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import filter_repo_objects
    except ImportError as e:
        return unlisted(hub_import_problem(e), gated)
    try:
        info = HfApi().model_info(
            repo_id,
            revision=str(revision) if revision else None,
            files_metadata=True,
            token=resolve_hf_token(token),
        )
    except Exception as e:
        return unlisted(
            _hub_unavailable_reason(e, repo_id, gated=bool(gated), license_name=license_name, revision=revision), gated
        )
    access = gated if gated is not None else bool(getattr(info, "gated", False))
    commit = str(getattr(info, "sha", "") or "")
    selected = list(
        filter_repo_objects(
            items=list(getattr(info, "siblings", None) or []),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
            key=lambda sibling: str(sibling.rfilename),
        )
    )
    if diffusers_trim and any(str(s.rfilename) == "model_index.json" for s in selected):
        try:
            index = _hub_json(repo_id, "model_index.json", revision=commit or None, token=token)
        except Exception as e:
            return unlisted(
                _hub_unavailable_reason(e, repo_id, gated=bool(access), license_name=license_name, revision=commit or None),
                access,
            )
        keep = set(
            diffusers_download_files(
                [str(s.rfilename) for s in selected],
                components=model_index_components(index),
                variant=_diffusers_backend_variant(),
            )
        )
        selected = [s for s in selected if str(s.rfilename) in keep]
    if not commit:
        return unlisted("the listing gives no commit", access)
    unsized = [str(s.rfilename) for s in selected if getattr(s, "size", None) is None]
    if unsized:
        # huggingface.co gives every file's size in this listing (`?blobs=true`); a mirror set with
        # HF_ENDPOINT may not, and a download that cannot be sized is not started.
        return unlisted(
            f"the listing from {_hub_endpoint()} gives no size for {unsized[0]}, so the download cannot be sized; "
            "if HF_ENDPOINT points at a mirror, point it at a server that lists file sizes (huggingface.co does)",
            access,
        )
    repo_dir = hf_repo_dir(repo_id, cache_dir=str(cache_root))
    files = tuple(_hub_file(repo_dir, commit, s) for s in sorted(selected, key=lambda s: str(s.rfilename)))
    return HubDownloadPlan(
        repo_id=repo_id,
        cache_dir=cache_root,
        free_bytes=free_bytes,
        files=files,
        commit=commit,
        gated=access,
        license_name=license_name,
        revision=str(revision) if revision else None,
    )


def _require_disk_space(plan: HubDownloadPlan | DownloadPlan) -> None:
    error = plan.disk_space_error()
    if error is not None:
        raise InsufficientDiskSpaceError(error)


def _require_planned_files(plan: HubDownloadPlan, snapshot: Path) -> None:
    missing = [f.name for f in plan.files if not (snapshot / f.name).is_file()]
    if missing:
        raise RuntimeError(
            f"The download of {plan.repo_id} is incomplete: {len(missing)} of {len(plan.files)} files are "
            f"missing from {snapshot} (for example {missing[0]}). Run the same command again to fetch them; "
            "the files that completed stay in the cache."
        )


def _fetch_plan(plan: HubDownloadPlan, *, token: Optional[str], max_workers: int) -> Path:
    """Fetch the files of a listed plan that the cache lacks, and return the snapshot holding them all.

    The files are those of the commit the listing named (`plan.commit`), so a repository updated since
    the listing cannot mix two commits into one snapshot; the plan's revision (the default branch when
    it named none) is then recorded as that commit in the cache, as huggingface_hub records a branch it
    downloads. Complete files are not fetched again; when every file is already linked in the snapshot,
    nothing is requested at all."""

    if not plan.files:
        raise RuntimeError(f"The Hub lists no file of {plan.repo_id} that the download selects; nothing was fetched.")
    repo_dir = hf_repo_dir(plan.repo_id, cache_dir=str(plan.cache_dir))
    snapshot = repo_dir / "snapshots" / str(plan.commit)
    if not all((snapshot / f.name).exists() for f in plan.files):
        from huggingface_hub import snapshot_download

        try:
            snapshot = Path(
                str(
                    snapshot_download(
                        repo_id=plan.repo_id,
                        repo_type="model",
                        revision=str(plan.commit),
                        allow_patterns=[glob.escape(f.name) for f in plan.files],
                        token=resolve_hf_token(token),
                        cache_dir=str(plan.cache_dir),
                        max_workers=max(1, int(max_workers)),
                    )
                )
            )
        except Exception as e:
            status = _hub_refusal_status(e)
            if status is not None:
                raise HuggingFaceAccessError(
                    plan.repo_id,
                    raw_message=str(e),
                    status_code=status,
                    gated=bool(plan.gated),
                    license_name=plan.license_name,
                ) from e
            raise
    _require_planned_files(plan, snapshot)
    revision = plan.revision or _hub_default_revision()
    if revision != plan.commit:
        write_hf_ref(repo_dir, revision, str(plan.commit))
    return snapshot


def _hub_default_revision() -> str:
    try:
        from huggingface_hub import constants

        return str(constants.DEFAULT_REVISION)
    except Exception:
        return "main"


def download_hf_repo_snapshot(
    repo_id: str,
    *,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    ignore_patterns: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path | str] = None,
    local_files_only: bool = False,
    max_workers: int = 4,
) -> Path:
    """Download (or resolve) a Hugging Face model repo snapshot into the HF cache.

    The download is planned first, as `download_model_preset` does: the files the patterns select
    are listed on the Hub, the cache disk must hold the ones to fetch (`InsufficientDiskSpaceError`),
    and complete files are not fetched again. Without the listing nothing is downloaded
    (`RuntimeError`). `local_files_only=True` resolves the cached snapshot only."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(f"Nothing was downloaded for {repo_id}: {hub_import_problem(e)}.") from e

    cache_root = Path(cache_dir).expanduser() if cache_dir else default_hf_cache_root()
    if local_files_only:
        return Path(
            str(
                snapshot_download(
                    repo_id=str(repo_id),
                    repo_type="model",
                    revision=str(revision) if revision else None,
                    allow_patterns=list(allow_patterns) if allow_patterns else None,
                    ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
                    cache_dir=str(cache_root),
                    local_files_only=True,
                )
            )
        )
    gated, license_name = _catalog_access(str(repo_id))
    plan = _plan_hub_download(
        str(repo_id),
        cache_root=cache_root,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        diffusers_trim=False,
        gated=gated,
        license_name=license_name,
        revision=revision,
    )
    if not plan.listed:
        raise RuntimeError(f"Nothing was downloaded for {repo_id}: {plan.listing_error}.")
    _require_disk_space(plan)
    return _fetch_plan(plan, token=token, max_workers=max_workers)


# Weight formats a Hugging Face repo may ship next to its safetensors. The Diffusers backend
# loads safetensors only (`use_safetensors=True`), from the component folders model_index.json
# names; the others are duplicates it never reads.
_WEIGHT_FILE_SUFFIXES = (
    ".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".onnx", ".onnx_data", ".msgpack", ".h5", ".gguf",
)
_WEIGHT_VARIANTS = ("fp16", "bf16", "fp32", "fp8", "non_ema", "ema")


def _safetensors_variant(name: str) -> Tuple[bool, Optional[str]]:
    """(is a safetensors weight or index file, its Diffusers variant or None)."""

    import re

    base = str(name).rsplit("/", 1)[-1]
    m = re.match(r"^[^.]+\.safetensors\.index(?:\.(?P<v>[A-Za-z0-9_]+))?\.json$", base)
    if m:
        v = m.group("v")
        return True, (v if v in _WEIGHT_VARIANTS else None)
    if not base.endswith(".safetensors"):
        return False, None
    m = re.match(r"^[^.]+\.(?P<v>[A-Za-z0-9_]+)(?:-\d{5}-of-\d{5})?\.safetensors$", base)
    if m and m.group("v") in _WEIGHT_VARIANTS:
        return True, m.group("v")
    return True, None


def diffusers_download_files(
    files: Sequence[str], *, components: Sequence[str], variant: Optional[str]
) -> List[str]:
    """The files of a Diffusers repo the backend loads.

    Root files except weights (a root `*.safetensors` / `.ckpt` is a single-file duplicate of the
    pipeline), and in each `components` folder its non-weight files plus one set of safetensors:
    the `variant` set when the folder ships it (the backend asks Diffusers for variant "fp16" on
    CUDA and MPS), else the default set, else the folder's only variant. `.bin`, `.onnx`,
    `.msgpack` and the other variant sets are never read. Measured: stabilityai/sdxl-turbo is
    41.6 GB with every safetensors and 6.9 GB this way; FLUX.2-klein-4B 23.7 GB and 16.0 GB."""

    wanted = set(str(c) for c in components)
    keep: List[str] = []
    by_dir: Dict[str, List[str]] = {}
    for f in files:
        f = str(f)
        if "/" not in f:
            if not f.lower().endswith(_WEIGHT_FILE_SUFFIXES):
                keep.append(f)
            continue
        folder = f.split("/", 1)[0]
        if folder in wanted:
            by_dir.setdefault(folder, []).append(f)
    for folder in sorted(by_dir):
        sets: Dict[Optional[str], List[str]] = {}
        for f in by_dir[folder]:
            is_safetensors, v = _safetensors_variant(f)
            if is_safetensors:
                sets.setdefault(v, []).append(f)
            elif not f.lower().endswith(_WEIGHT_FILE_SUFFIXES):
                keep.append(f)
        chosen = (sets.get(variant) if variant else None) or sets.get(None) or sets.get("fp16")
        if chosen is None and sets:
            chosen = sets[sorted(sets, key=str)[0]]
        keep.extend(chosen or [])
    return sorted(set(keep))


def _diffusers_backend_variant() -> Optional[str]:
    """The checkpoint variant the Diffusers backend requests on this host: "fp16" where its
    default dtype is float16 (CUDA, Apple MPS), none on the processor (float32)."""

    return "fp16" if local_model_profile() in {"cuda", "apple-silicon"} else None


def preset_download_patterns(
    preset: VisionModelDownloadPreset,
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """(allow, ignore) patterns a preset selects its files with, among those the Hub lists."""

    allow_patterns = list(preset.allow_patterns) or None
    ignore_patterns = None
    if preset.target == "gguf" and not any("/" in str(p) for p in (allow_patterns or ())):
        # A GGUF artifact is a root file (the sd.cpp selection globs the snapshot root), and the
        # hub's fnmatch lets `*` cross `/`: `*q8_0.gguf` also took Green-Sky/SD-Turbo-GGUF's
        # `old/sd_turbo-q8_0.gguf`.
        ignore_patterns = ["*/*"]
    return allow_patterns, ignore_patterns


def plan_model_preset_download(
    preset: VisionModelDownloadPreset,
    *,
    base_model: Optional[str] = None,
    model_dir: Optional[Path] = None,
    token: Optional[str] = None,
    all_files: bool = False,
) -> DownloadPlan:
    """Everything `download_model_preset` fetches for a preset, from the Hub listings, without writing
    anything (`download --dry-run`).

    The parts are the package, then what its engine loads with it: a stable-diffusion.cpp bundle's
    companion repositories (text encoder, VAE), and with `base_model` (an MLX-Gen schedule of the
    package) the adapter it attaches, unless mlx-gen finds it cached already. A legacy tree
    (`legacy_model_tree`) to import replaces the package part. A Diffusers preset selects only the files
    the Diffusers backend loads on this host (`diffusers_download_files`; its `model_index.json` is
    read into memory); `all_files=True` takes every file its patterns match.

    Raises `HuggingFaceAccessError` when the Hub refuses a repository, ValueError when `base_model` is
    not a schedule of the package, and `OptionalDependencyMissingError` when the adapter it attaches
    can only be named by mlx-gen and mlx-gen is not installed: all before anything is fetched. A
    listing that cannot be had otherwise gives a part whose plan has `listing_error`."""

    tree, action = legacy_model_tree(preset, model_dir=model_dir)
    schedule, adapter, adapter_path, adapter_part = _schedule_adapter_part(preset, base_model, token=token)
    parts: List[DownloadPart] = []
    if action != "import":
        parts.append(_package_part(preset, token=token, all_files=all_files))
    parts.extend(_bundle_parts(preset, token=token))
    if adapter_part is not None:
        parts.append(adapter_part)
    return DownloadPlan(
        parts=tuple(parts),
        legacy_import=tree if action == "import" else None,
        legacy_remove=tree if action == "remove" else None,
        base_model=schedule,
        adapter=adapter,
        adapter_path=adapter_path,
    )


def _package_part(preset: VisionModelDownloadPreset, *, token: Optional[str], all_files: bool) -> DownloadPart:
    allow_patterns, ignore_patterns = preset_download_patterns(preset)
    gated, license_name = _preset_access(preset)
    plan = _plan_hub_download(
        preset.repo_id,
        cache_root=default_hf_cache_root(),
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        diffusers_trim=not all_files and preset.target == "diffusers" and preset.engine == "diffusers",
        gated=gated,
        license_name=license_name,
    )
    return DownloadPart("package", plan)


def _sdcpp_bundle(preset: VisionModelDownloadPreset) -> Optional[_SdcppBundleSpec]:
    return _SDCPP_BUNDLES_BY_REPO_ID.get(str(preset.repo_id).strip().lower(), _SDCPP_BUNDLES.get(preset.key))


def _bundle_parts(preset: VisionModelDownloadPreset, *, token: Optional[str]) -> List[DownloadPart]:
    """The companion repositories stable-diffusion.cpp loads with a GGUF diffusion model, one part per
    repository (a repository serving two roles, such as a VAE and a text encoder, is listed once)."""

    bundle = _sdcpp_bundle(preset) if preset.engine == "stable-diffusion.cpp" else None
    if bundle is None or bundle.mode != "component":
        return []
    by_repo: Dict[str, Tuple[List[str], List[str]]] = {}
    for component in bundle.components:
        roles, patterns = by_repo.setdefault(component.repo_id, ([], []))
        roles.append(component.role)
        patterns.extend(component.allow_patterns)
    parts: List[DownloadPart] = []
    for repo_id, (roles, patterns) in by_repo.items():
        gated, license_name = _catalog_access(repo_id)
        plan = _plan_hub_download(
            repo_id,
            cache_root=default_hf_cache_root(),
            token=token,
            allow_patterns=patterns,
            ignore_patterns=None,
            diffusers_trim=False,
            gated=gated,
            license_name=license_name,
        )
        parts.append(DownloadPart(" and ".join(roles), plan, tuple(patterns)))
    return parts


def _schedule_adapter_part(
    preset: VisionModelDownloadPreset, base_model: Optional[str], *, token: Optional[str]
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[DownloadPart]]:
    """(schedule, adapter, path where mlx-gen finds it, part to fetch it) for `download --base-model`.

    The adapter is fetched into huggingface_hub's own cache, one of the two places mlx-gen resolves a
    Hugging Face adapter from (its LoRA cache is the other)."""

    if base_model is None:
        return None, None, None, None
    if not preset.in_catalog or str(preset.engine) != "mlx-gen":
        what = f"a {preset.engine} package" if preset.in_catalog else "not in the AbstractVision catalog"
        raise ValueError(f"--base-model names an MLX-Gen schedule; {preset.repo_id} is {what}.")
    from .backends.mflux import mlx_gen_package_schedule, resolve_mlx_gen_schedule_adapter
    from .errors import OptionalDependencyMissingError

    try:
        schedule = mlx_gen_package_schedule(preset, base_model)
    except ValueError as e:
        raise ValueError(f"--base-model {e}") from None
    try:
        resolved = resolve_mlx_gen_schedule_adapter(schedule)
    except OptionalDependencyMissingError as e:
        raise OptionalDependencyMissingError(
            f"--base-model {schedule}: the adapter a schedule loads is named by mlx-gen, which could not be "
            f"imported here, so it cannot be planned. {e}"
        ) from None
    if resolved is None:
        return schedule, None, None, None
    adapter, path = resolved
    if path:
        return schedule, adapter, str(path), None
    repo_id, filename = _adapter_hub_file(adapter)
    pattern = glob.escape(filename)
    gated, license_name = _catalog_access(repo_id)
    plan = _plan_hub_download(
        repo_id,
        cache_root=_hub_library_cache(),
        token=token,
        allow_patterns=[pattern],
        ignore_patterns=None,
        diffusers_trim=False,
        gated=gated,
        license_name=license_name,
    )
    return schedule, adapter, None, DownloadPart(f"adapter ({schedule})", plan, (pattern,))


def _adapter_hub_file(adapter: str) -> Tuple[str, str]:
    """(repository, file) of an mlx-gen Hugging Face adapter id: `hf:<owner>/<repo>/<file>` or
    `<owner>/<repo>:<file>`, the two forms mlx-gen's LoRA resolution names."""

    text = adapter[len("hf:") :] if adapter.startswith("hf:") else adapter
    if ":" in text:
        repo_id, filename = text.split(":", 1)
    elif text.count("/") == 2:
        repo_id, filename = text.rsplit("/", 1)
    else:
        raise ValueError(f"The adapter {adapter!r} is not a Hugging Face file this command can plan.")
    return repo_id, filename


def _hub_library_cache() -> Path:
    # huggingface_hub's own cache, which mlx-gen reads with its defaults (ABSTRACTVISION_HF_HUB_CACHE
    # does not apply to the engine's lookup).
    try:
        from huggingface_hub import constants

        return Path(str(constants.HF_HUB_CACHE)).expanduser()
    except Exception:
        return default_hf_cache_root()


def legacy_model_tree(
    preset: VisionModelDownloadPreset, *, model_dir: Optional[Path] = None
) -> Tuple[Optional[Path], str]:
    """A legacy `<model_dir>/<preset.local_dir_name>` tree (default `model_dir`: `~/models`, or
    `ABSTRACTVISION_MODEL_DIR`) and what `download_model_preset` does with it: `"import"` (moved into
    the Hugging Face cache as a snapshot), `"remove"` (the cache already holds a usable snapshot of
    the repository, so the tree is deleted), or `(None, "")` when there is no tree."""

    if not preset.local_dir_name:
        return None, ""
    root = Path(model_dir).expanduser() if model_dir is not None else default_legacy_model_root()
    tree = root / preset.local_dir_name
    if not tree.is_dir():
        return None, ""
    required_files, require_weight_files = snapshot_requirements_for_preset(preset)
    cached = resolve_hf_repo_snapshot(
        preset.repo_id,
        cache_dir=str(default_hf_cache_root()),
        required_files=required_files,
        require_weight_files=require_weight_files,
    )
    if cached is not None and cached.resolve() != tree.resolve():
        return tree, "remove"
    return tree, "import"


def download_model_preset(
    preset: VisionModelDownloadPreset,
    *,
    base_model: Optional[str] = None,
    model_dir: Optional[Path] = None,
    token: Optional[str] = None,
    max_workers: int = 4,
    all_files: bool = False,
    report: Optional[Callable[[str], None]] = None,
) -> Path:
    """Download a preset into the Hugging Face cache, as its plan says, and return the package snapshot.

    The whole plan (`plan_model_preset_download`: the package, a stable-diffusion.cpp bundle's
    companions, the adapter of the MLX-Gen schedule `base_model`) is made first, and `report` receives
    its summary. A legacy `<model_dir>/<preset.local_dir_name>` tree (`legacy_model_tree`) is then
    moved into the cache layout and used as the package as it is (`report` says it was not compared
    with the Hub), or deleted when the cache already holds the repository. A cache disk without room
    for the files to fetch raises `InsufficientDiskSpaceError` before anything is written; files
    complete in the cache are not fetched again. A repository the Hub refuses raises
    `HuggingFaceAccessError`. Without a Hub listing nothing is fetched: what a local look finds in the
    cache is used as it is (`report` says it was not compared with the Hub) when it finds every part,
    otherwise `RuntimeError`."""

    plan = plan_model_preset_download(
        preset, base_model=base_model, model_dir=model_dir, token=token, all_files=all_files
    )
    say = report or (lambda _line: None)
    snapshot: Optional[Path] = None
    tree = plan.legacy_import or plan.legacy_remove
    if tree is not None:
        required_files, require_weight_files = snapshot_requirements_for_preset(preset)
        migrated = ensure_hf_repo_snapshot(
            preset.repo_id,
            source_dir=tree,
            cache_dir=str(default_hf_cache_root()),
            cleanup_source=True,
            required_files=required_files,
            require_weight_files=require_weight_files,
        )
        if plan.legacy_import is not None:
            from .model_inventory import local_cache_state

            snapshot = migrated
            state = local_cache_state(preset, extra_roots=())
            say(
                f"cache: imported {preset.repo_id} from the legacy directory {tree} into {snapshot} "
                f"({_files(state.files)}, {format_bytes(state.bytes)}); it was not compared with the Hub, so it "
                "may lack files the package selects; nothing was fetched"
            )
        else:
            say(f"legacy: removed {tree}; the cache already holds a snapshot of {preset.repo_id}")
    for line in plan.summary_lines():
        say(line)
    if not plan.listed:
        return _unlisted_download(preset, plan, snapshot=snapshot, report=say)
    _require_disk_space(plan)
    for part in plan.parts:
        fetched = _fetch_plan(part.plan, token=token, max_workers=max_workers)
        if part.role == "package":
            snapshot = fetched
    if plan.package is not None and snapshot is not None:
        required_files, require_weight_files = snapshot_requirements_for_preset(preset)
        if not hf_snapshot_is_usable(
            snapshot,
            required_files=required_files,
            require_weight_files=require_weight_files,
            reject_incomplete=False,
        ):
            raise RuntimeError(
                f"The download of {preset.repo_id} holds every file the Hub lists for it, but no model weights the "
                f"package needs: {snapshot}. Check that the repository is populated."
            )
    if plan.adapter is not None and plan.adapter_path is None:
        from .backends.mflux import resolve_mlx_gen_schedule_adapter

        resolved = resolve_mlx_gen_schedule_adapter(str(plan.base_model))
        if not resolved or not resolved[1]:
            raise RuntimeError(
                f"The adapter {plan.adapter} was fetched into {_hub_library_cache()}, but mlx-gen does not find it "
                "there; generation with that schedule will not load it."
            )
        say(f"adapter ({plan.base_model}): {resolved[1]}")
    assert snapshot is not None
    return snapshot


def _unlisted_download(
    preset: VisionModelDownloadPreset, plan: DownloadPlan, *, snapshot: Optional[Path], report: Callable[[str], None]
) -> Path:
    """Without the Hub listing of a part nothing is fetched: what the cache holds is used as it is when a
    local look finds every part, never called complete; otherwise RuntimeError says what is missing."""

    from .model_inventory import local_cache_state

    reason = next(part.plan.listing_error for part in plan.parts if not part.plan.listed)
    for part in plan.parts:
        listed = part.plan.listed
        if part.role == "package":
            if listed and not part.plan.fetch_files:
                snapshot = hf_repo_dir(preset.repo_id, cache_dir=str(part.plan.cache_dir)) / "snapshots" / str(
                    part.plan.commit
                )
                continue
            if listed:
                raise RuntimeError(f"Nothing was downloaded for {preset.repo_id}: {reason}.")
            state = local_cache_state(preset, extra_roots=())
            if state.snapshot_dir is None or state.gaps:
                raise RuntimeError(f"Nothing was downloaded for {preset.repo_id}: {reason}. {state.describe()}")
            snapshot = state.snapshot_dir
            report(
                f"cache: a snapshot of {preset.repo_id} is at {state.snapshot_dir} ({_files(state.files)}, "
                f"{format_bytes(state.bytes)}); it was not compared with the Hub, so it may lack files the "
                "package selects"
            )
            continue
        if listed and not part.plan.fetch_files:
            continue
        held = (
            None
            if listed
            else resolve_hf_repo_snapshot(
                part.plan.repo_id,
                cache_dir=str(part.plan.cache_dir),
                required_files=part.patterns,
                require_weight_files=True,
            )
        )
        if held is None:
            raise RuntimeError(
                f"Nothing was downloaded for {preset.repo_id}: its {part.role} {part.plan.repo_id} is not in the cache, "
                f"and {reason}."
            )
        report(f"{part.role}: {part.plan.repo_id} is in the cache at {held}; it was not compared with the Hub")
    assert snapshot is not None
    return snapshot


def _find_snapshot_file(snapshot_dir: Path, patterns: Sequence[str]) -> Optional[Path]:
    snapshot = Path(snapshot_dir).expanduser()
    for pattern in patterns:
        try:
            matches = sorted(path for path in snapshot.glob(str(pattern)) if path.is_file())
        except Exception:
            matches = []
        if matches:
            return matches[0]
    return None


def _gguf_patterns_for_preset(preset: VisionModelDownloadPreset) -> Tuple[str, ...]:
    out = tuple(
        str(pattern)
        for pattern in tuple(preset.allow_patterns or ())
        if ".gguf" in str(pattern).lower()
    )
    return out or ("*.gguf",)


def _sdcpp_bundle_download_hint(repo_id: str) -> str:
    # The command that fetches this GGUF package with its companions (`download_command`).
    try:
        return f"Run `{download_command(repo_id, engine='stable-diffusion.cpp')}` first."
    except ValueError:
        return f"No `abstractvision download` command selects {repo_id} on this host; fetch it from https://huggingface.co/{repo_id}."


def _resolve_cached_sdcpp_companion(
    *,
    requested_name: str,
    package_repo_id: str,
    role: str,
    repo_id: str,
    allow_patterns: Sequence[str],
    allow_download: bool,
    token: Optional[str],
    max_workers: int,
) -> Path:
    cache_root = default_hf_cache_root()
    snapshot = resolve_hf_repo_snapshot(
        repo_id,
        cache_dir=str(cache_root),
        required_files=tuple(allow_patterns),
        require_weight_files=True,
    )
    if snapshot is None:
        if not allow_download:
            raise RuntimeError(
                f"Missing cached stable-diffusion.cpp companion artifact for {requested_name!r}: "
                f"{role} from {repo_id!r}. {_sdcpp_bundle_download_hint(package_repo_id)}"
            )
        snapshot = download_hf_repo_snapshot(
            repo_id,
            token=token,
            allow_patterns=tuple(allow_patterns),
            cache_dir=cache_root,
            max_workers=max(1, int(max_workers)),
        )
    resolved = _find_snapshot_file(snapshot, allow_patterns)
    if resolved is None:
        raise RuntimeError(
            f"Cached stable-diffusion.cpp companion artifact for {requested_name!r} is incomplete: "
            f"{role} from {repo_id!r} is missing one of {tuple(allow_patterns)!r}."
        )
    return resolved


def resolve_sdcpp_model_selection(
    name: str,
    *,
    allow_download: bool = False,
    token: Optional[str] = None,
    max_workers: int = 4,
) -> SdcppModelSelection:
    requested = str(name or "").strip()
    if not requested:
        raise ValueError("Missing stable-diffusion.cpp model selection.")
    if _macos_gguf_disabled():
        _raise_macos_gguf_disabled()

    preset = find_model_preset(
        requested,
        target="gguf",
        engine="stable-diffusion.cpp",
        require_8bit=True,
    )
    bundle = _sdcpp_bundle(preset)
    cache_root = default_hf_cache_root()

    required_files, require_weight_files = snapshot_requirements_for_preset(preset)
    legacy_dir = default_legacy_model_root() / preset.local_dir_name
    snapshot = ensure_hf_repo_snapshot(
        preset.repo_id,
        source_dir=legacy_dir,
        cache_dir=str(cache_root),
        cleanup_source=True,
        required_files=required_files,
        require_weight_files=require_weight_files,
    )
    if snapshot is None:
        if not allow_download:
            raise RuntimeError(
                f"Missing cached stable-diffusion.cpp model artifact for {requested!r} ({preset.repo_id!r}). "
                f"{_sdcpp_bundle_download_hint(preset.repo_id)}"
            )
        snapshot = download_model_preset(
            preset,
            token=token,
            max_workers=max(1, int(max_workers)),
        )

    model_patterns = bundle.model_patterns if bundle is not None else _gguf_patterns_for_preset(preset)
    primary_file = _find_snapshot_file(snapshot, model_patterns)
    if primary_file is None:
        raise RuntimeError(
            f"Cached stable-diffusion.cpp model artifact for {requested!r} is incomplete: "
            f"{preset.repo_id!r} is missing one of {model_patterns!r}."
        )

    if bundle is None or bundle.mode == "single-file":
        return SdcppModelSelection(
            key=preset.key,
            repo_id=preset.repo_id,
            capabilities_model_id=str(preset.upstream_repo_id or preset.repo_id),
            model=str(primary_file),
        )

    component_values: Dict[str, str] = {"diffusion_model": str(primary_file)}
    for component in bundle.components:
        component_values[component.role] = str(
            _resolve_cached_sdcpp_companion(
                requested_name=requested,
                package_repo_id=preset.repo_id,
                role=component.role,
                repo_id=component.repo_id,
                allow_patterns=component.allow_patterns,
                allow_download=allow_download,
                token=token,
                max_workers=max_workers,
            )
        )

    return SdcppModelSelection(
        key=preset.key,
        repo_id=preset.repo_id,
        capabilities_model_id=str(preset.upstream_repo_id or preset.repo_id),
        model=None,
        diffusion_model=component_values.get("diffusion_model"),
        vae=component_values.get("vae"),
        llm=component_values.get("llm"),
        llm_vision=component_values.get("llm_vision"),
    )


def format_model_preset_rows(presets: Sequence[VisionModelDownloadPreset]) -> Iterable[str]:
    rows = [
        (
            p.repo_id,
            p.target,
            p.engine,
            str(p.quantization_bits) if p.quantization_bits is not None else "n/a",
            p.source,
        )
        for p in presets
    ]
    headers = ("model_id", "target", "engine", "bits", "source")
    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    yield fmt.format(*headers)
    yield fmt.format(*("-" * w for w in widths))
    for row in rows:
        yield fmt.format(*row)
