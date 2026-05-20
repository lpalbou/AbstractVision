from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .model_capabilities import VisionModelCapabilitiesRegistry
from .model_cache import (
    default_hf_cache_root,
    default_legacy_model_root,
    ensure_hf_repo_snapshot,
    hf_snapshot_is_usable,
    resolve_hf_repo_snapshot,
)


_GENERIC_MLX_BACKEND_ERROR = (
    "AbstractVision does not have a generic MLX image backend yet. "
    "Use `--target mlx` to browse MLX artifacts and `--provider mflux` "
    "(or `mflux/<preset>`) for MFLUX-compatible 8-bit MLX models."
)


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

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
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
    "*.json",
    "*.md",
    "LICENSE*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "transformer/*",
    "vae/*",
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
    def __init__(self, repo_id: str, *, raw_message: str = "", status_code: Optional[int] = None):
        self.repo_id = str(repo_id)
        self.repo_url = f"https://huggingface.co/{self.repo_id}"
        self.token_url = HF_TOKEN_SETTINGS_URL
        self.raw_message = str(raw_message or "").strip()
        self.status_code = status_code
        detail = f" ({status_code})" if status_code else ""
        msg = (
            f"Cannot access Hugging Face repo {self.repo_id!r}{detail}.\n"
            f"1. Open {self.repo_url} and accept the model terms or request access.\n"
            f"2. Create or copy a Hugging Face read token at {self.token_url}.\n"
            "3. Retry with one of:\n"
            "   - `abstractvision download-model ... --token <HF_TOKEN>`\n"
            "   - `hf auth login` (or `huggingface-cli login`)\n"
            "   - `export HF_TOKEN=...` (or `export HUGGINGFACE_HUB_TOKEN=...`)\n"
        )
        if self.raw_message:
            msg += f"Upstream error: {self.raw_message}"
        super().__init__(msg)


_PRESETS: Tuple[VisionModelDownloadPreset, ...] = (
    VisionModelDownloadPreset(
        key="flux2-klein-4b",
        display_name="FLUX.2 klein 4B MLX 8-bit",
        repo_id="AITRADER/FLUX2-klein-4B-mlx-8bit",
        target="mlx",
        engine="mflux",
        local_dir_name="flux2-klein-4b-mlx-8bit",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-4B",
        source="curated-community-mlx",
        aliases=(
            "flux2-klein-4b",
            "flux-klein-4b",
            "klein-4b",
            "klein4b",
            "flux4b",
            "moxin-org/FLUX.2-klein-4B-8bit-mlx",
            "Runpod/FLUX.2-klein-4B-mflux-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes=(
            "BFL publishes the upstream 4B model and an official FP8 artifact, but not an MLX "
            "8-bit layout. Use this preset on Apple Silicon."
        ),
        source_priority=30,
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
        notes=(
            "#FALLBACK: this is a full Diffusers snapshot (not 8-bit). "
            "FLUX.2 pipeline support may require Diffusers from source (diffusers@main)."
        ),
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
        display_name="FLUX.2 klein 9B MLX 8-bit",
        repo_id="deepsweet/FLUX.2-klein-9B-MLX-Q8",
        target="mlx",
        engine="mflux",
        local_dir_name="flux2-klein-9b-mlx-8bit",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-9B",
        source="curated-community-mlx",
        aliases=(
            "flux2-klein-9b",
            "flux-klein-9b",
            "klein-9b",
            "klein9b",
            "flux9b",
            "deepsweet/FLUX.2-klein-9B-MLX-Q4",
            "themindstudio/flux2-klein-9b-mlx-4bit",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes=(
            "BFL publishes the upstream 9B model and official FP8 artifacts, but not an MLX "
            "8-bit layout. Use this preset on Apple Silicon."
        ),
        source_priority=30,
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
        notes=(
            "#FALLBACK: this is a full Diffusers snapshot (not 8-bit). "
            "The 9B repo is gated on Hugging Face and may require HF_TOKEN."
        ),
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
        key="z-image-turbo",
        display_name="Z-Image-Turbo mflux MLX 8-bit",
        repo_id="carsenk/z-image-turbo-mflux-8bit",
        target="mlx",
        engine="mflux",
        local_dir_name="z-image-turbo-mlx-8bit",
        quantization_bits=8,
        upstream_repo_id="Tongyi-MAI/Z-Image-Turbo",
        source="curated-community-mflux",
        aliases=(
            "z-image-turbo",
            "zimage-turbo",
            "z-image",
            "tongyi-z-image-turbo",
            "andrevp/Z-Image-Turbo-MLX",
            "andrevp/Z-Image-Turbo-MLX-8bit",
            "illusion615/Z-Image-Turbo-MLX",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes=(
            "Tongyi-MAI publishes the upstream full model; this preset keeps Apple Silicon "
            "downloads on an mflux-compatible MLX 8-bit conversion."
        ),
        source_priority=30,
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
        aliases=("z-image-turbo", "zimage-turbo", "z-image", "tongyi-z-image-turbo"),
        allow_patterns=(
            "model_index.json",
            "*.json",
            "*.md",
            "*.txt",
            "*.yaml",
            "LICENSE*",
            *_DIFFUSERS_PIPELINE_DIRS,
        ),
        notes="#FALLBACK: this is a full Diffusers snapshot (not 8-bit).",
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
        aliases=("z-image-turbo", "zimage-turbo", "z-image", "tongyi-z-image-turbo"),
        allow_patterns=("README.md", "LICENSE*", "z-image-turbo-Q8_0.gguf"),
        notes="Q8_0 GGUF for stable-diffusion.cpp style runtimes.",
        source_priority=40,
    ),
    VisionModelDownloadPreset(
        key="qwen-image",
        display_name="Qwen-Image-2512 MLX 8-bit (MFLUX)",
        repo_id="mlx-community/Qwen-Image-2512-8bit",
        target="mlx",
        engine="mflux",
        local_dir_name="qwen-image-2512-mlx-8bit",
        quantization_bits=8,
        upstream_repo_id="Qwen/Qwen-Image-2512",
        source="mlx-community",
        aliases=("qwen-image", "qwen-image-2512", "qwen-image-8bit", "qwen-image-2512-8bit"),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes=(
            "The upstream Qwen Image 2512 repo ships full weights; this preset is a community MLX 8-bit conversion "
            "compatible with the optional MFLUX runtime."
        ),
        source_priority=20,
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
        notes="#FALLBACK: this is a full Diffusers snapshot (not 8-bit).",
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
        key="qwen-image-edit",
        display_name="Qwen-Image-Edit-2511 (Diffusers)",
        repo_id="Qwen/Qwen-Image-Edit-2511",
        target="diffusers",
        engine="diffusers",
        local_dir_name="qwen-image-edit-2511-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("qwen-image-edit", "qwen-image-edit-2511", "Qwen/Qwen-Image-Edit-2511"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
        source_priority=90,
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
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
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
        notes=(
            "#FALLBACK: this is a full Diffusers snapshot (not 8-bit). "
            "Use the Diffusers backend (`--provider diffusers`) to run it."
        ),
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
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
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
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
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
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
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
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
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
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
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
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
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
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
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
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
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
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
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
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
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
        notes="#FALLBACK: full Diffusers snapshot (not 8-bit).",
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
    "qwen-image": _SdcppBundleSpec(
        mode="component",
        model_patterns=("qwen-image-2512-Q8_0.gguf",),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="Qwen/Qwen-Image-2512",
                allow_patterns=("vae/diffusion_pytorch_model.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                allow_patterns=("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",),
            ),
        ),
    ),
    "qwen-image-edit": _SdcppBundleSpec(
        mode="component",
        model_patterns=("qwen-image-edit-2511-Q8_0.gguf",),
        components=(
            _SdcppBundleComponentSpec(
                role="vae",
                repo_id="Qwen/Qwen-Image-Edit-2511",
                allow_patterns=("vae/diffusion_pytorch_model.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                allow_patterns=("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",),
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
                repo_id="Qwen/Qwen-Image-Edit-2509",
                allow_patterns=("vae/diffusion_pytorch_model.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                allow_patterns=("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",),
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
                repo_id="Qwen/Qwen-Image",
                allow_patterns=("vae/diffusion_pytorch_model.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                allow_patterns=("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",),
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
                repo_id="Qwen/Qwen-Image-Edit",
                allow_patterns=("vae/diffusion_pytorch_model.safetensors",),
            ),
            _SdcppBundleComponentSpec(
                role="llm",
                repo_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
                allow_patterns=("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",),
            ),
        ),
    ),
}


def _default_allow_patterns(*, target: str, engine: str) -> Tuple[str, ...]:
    if target == "mlx" or engine == "mflux":
        return _COMMON_MLX_PATTERNS
    if target == "diffusers" or engine == "diffusers":
        return _COMMON_DIFFUSERS_PATTERNS
    if target == "fp8" or engine == "diffusers-component":
        return ("*.json", "*.md", "LICENSE*", "*.safetensors")
    if target == "gguf" or engine == "stable-diffusion.cpp":
        return ("README.md", "LICENSE*", "*.gguf")
    # Full Hugging Face snapshots should not constrain filenames by default.
    return ()


def _default_source_priority(*, source: str, target: str, engine: str, bits: Optional[int]) -> int:
    source_s = str(source or "").strip().lower()
    if target == "fp8" and bits == 8:
        return 0
    if target == "mlx" and bits == 8:
        return 30
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


def _default_local_dir_name(*, key: str, target: str, engine: str, repo_id: str) -> str:
    base = _slugify_token(key or repo_id.rsplit("/", 1)[-1])
    suffix = _slugify_token(target or engine)
    if base.endswith(f"-{suffix}") or base == suffix:
        return base
    return f"{base}-{suffix}"


def _default_display_name(*, model_id: str, target: str, engine: str, bits: Optional[int]) -> str:
    stem = str(model_id or "").rsplit("/", 1)[-1].replace("-", " ").strip() or str(model_id)
    engine_label = {
        "diffusers": "Diffusers",
        "diffusers-component": "Diffusers component",
        "mflux": "MFLUX",
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
        s = str(value or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return tuple(out)


@lru_cache(maxsize=1)
def _all_presets() -> Tuple[VisionModelDownloadPreset, ...]:
    out: List[VisionModelDownloadPreset] = list(_PRESETS)
    seen = {(p.key, p.engine, p.target, p.repo_id) for p in out}

    try:
        reg = VisionModelCapabilitiesRegistry()
    except Exception:
        return tuple(sorted(out, key=lambda p: (p.key, p.source_priority, p.repo_id)))

    for model_id in reg.list_models():
        spec = reg.get(model_id)
        for dl in spec.downloads:
            key = (dl.key, dl.engine, dl.target, dl.repo_id)
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
                    ),
                    quantization_bits=dl.bits,
                    upstream_repo_id=None if str(dl.repo_id) == str(model_id) else str(model_id),
                    source=str(dl.source or "official"),
                    aliases=_merge_aliases(str(dl.key), str(model_id), str(dl.repo_id)),
                    allow_patterns=_default_allow_patterns(target=str(dl.target), engine=str(dl.engine)),
                    notes=str(dl.notes or spec.notes or ""),
                    source_priority=_default_source_priority(
                        source=str(dl.source or "official"),
                        target=str(dl.target),
                        engine=str(dl.engine),
                        bits=dl.bits,
                    ),
                )
            )

    return tuple(sorted(out, key=lambda p: (p.key, p.source_priority, p.repo_id)))


@lru_cache(maxsize=1)
def local_model_profile() -> str:
    machine = str(platform.machine() or "").strip().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "apple-silicon"

    nvidia_smi = shutil.which("nvidia-smi")
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
        return ("mlx", "diffusers", "hf-snapshot")
    if profile == "cuda":
        return ("fp8", "gguf", "diffusers", "hf-snapshot")
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
            if preset.target not in preferred:
                preferred.append(preset.target)
        return tuple(preferred)
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
        "metal": "mflux",
        "apple": "mflux",
        "mac": "mflux",
        "macos": "mflux",
        "osx": "mflux",
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
            "diffusers": "diffusers",
            "stable-diffusion.cpp": "gguf",
            "diffusers-component": "fp8",
            "transformers": "hf-snapshot",
        }
        inferred = engine_to_target.get(selected_engine)
        if inferred:
            return inferred, selected_engine
    return normalize_model_target(target), selected_engine


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
        if preset.target not in selected_targets:
            continue
        if selected_engine is not None and preset.engine != selected_engine:
            continue
        if not include_non_8bit and preset.quantization_bits != 8:
            continue
        out.append(preset)
    return sorted(out, key=lambda p: (p.key, p.source_priority, p.repo_id))


def find_model_preset(
    name: str,
    *,
    target: Optional[str] = "auto",
    engine: Optional[str] = None,
    require_8bit: bool = True,
) -> VisionModelDownloadPreset:
    raw_target = str(target or "auto").strip().lower()
    requested = str(name or "").strip().lower()
    if requested.startswith("mlx/"):
        raise ValueError(_GENERIC_MLX_BACKEND_ERROR)
    for prefix in (
        "mflux/",
        "m-flux/",
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
        if preset.upstream_repo_id:
            repo_ids.add(preset.upstream_repo_id.lower())
        return requested == preset.key or requested in aliases or requested in repo_ids

    presets = _all_presets()

    def _sort_key(preset: VisionModelDownloadPreset) -> tuple[int, int, str]:
        return (target_rank.get(preset.target, len(target_rank)), preset.source_priority, preset.repo_id)

    candidates = [
        p
        for p in presets
        if p.target in selected_targets
        and (selected_engine is None or p.engine == selected_engine)
        and matches(p)
    ]
    if require_8bit:
        candidates = [p for p in candidates if p.quantization_bits == 8]
    if candidates:
        return sorted(candidates, key=_sort_key)[0]

    known = sorted({p.key for p in presets}.union(*(set(p.aliases) for p in presets)))
    any_engine_matches = [p for p in presets if matches(p)]
    target_matches = [
        p
        for p in presets
        if p.target in selected_targets and (selected_engine is None or p.engine == selected_engine) and matches(p)
    ]

    if raw_target in {"", "auto", "default"} and selected_engine is None and any_engine_matches:
        # If the caller left target/engine on defaults, pick a sensible fallback
        # when the resolved preset key only maps to one concrete artifact target.
        # This keeps platform-aware catalog defaults, but still lets explicit repo
        # ids like `mlx-community/...` resolve to their unique curated preset even
        # on non-Apple hosts.
        possible_targets = sorted({p.target for p in any_engine_matches})
        if len(possible_targets) == 1:
            fallback_target = possible_targets[0]
            fallback = [p for p in any_engine_matches if p.target == fallback_target]
            if require_8bit:
                eight_bit = [p for p in fallback if p.quantization_bits == 8]
                if eight_bit:
                    fallback = eight_bit
                elif fallback_target != "hf-snapshot":
                    raise ValueError(
                        f"No 8-bit preset for {name!r}. Available target/engine choices: "
                        + ", ".join(sorted({f"{p.target}/{p.engine}:{p.repo_id}" for p in any_engine_matches}))
                        + ". Use --allow-non-8bit, or select an explicit --target/--provider that has an 8-bit artifact."
                    )
            return sorted(fallback, key=_sort_key)[0]

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
    raise ValueError(f"Unknown vision model preset {name!r}. Known presets: {', '.join(known)}")


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


def _is_hf_access_error(exc: Exception) -> tuple[bool, Optional[int]]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    msg = str(exc).lower()
    if status_code in {401, 403}:
        return True, int(status_code)
    if "gated repo" in msg or "not in the authorized list" in msg or "accept the conditions" in msg:
        return True, int(status_code) if isinstance(status_code, int) else None
    return False, int(status_code) if isinstance(status_code, int) else None


def _snapshot_requirements_for_preset(
    preset: VisionModelDownloadPreset,
) -> tuple[Tuple[str, ...], bool]:
    target = str(preset.target or "").strip().lower()
    engine = str(preset.engine or "").strip().lower()
    if target == "diffusers" or engine == "diffusers":
        return ("model_index.json",), True
    if target in {"hf-snapshot", "mlx", "gguf", "fp8"}:
        return tuple(), True
    if engine in {"transformers", "mflux", "stable-diffusion.cpp", "diffusers-component"}:
        return tuple(), True
    return tuple(), False


def download_hf_repo_snapshot(
    repo_id: str,
    *,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    ignore_patterns: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path] = None,
    local_files_only: bool = False,
    max_workers: int = 4,
) -> Path:
    """Download (or resolve) a Hugging Face model repo snapshot into the HF cache."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "Downloading Hugging Face model snapshots requires `huggingface_hub`. "
            'Install it with `pip install "abstractvision[models]"` or use the `hf download` CLI.'
        ) from e

    try:
        resolved = snapshot_download(
            repo_id=str(repo_id),
            repo_type="model",
            revision=str(revision) if revision else None,
            token=resolve_hf_token(token),
            allow_patterns=list(allow_patterns) if allow_patterns else None,
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
            cache_dir=str(cache_dir.expanduser()) if cache_dir else None,
            local_files_only=bool(local_files_only),
            max_workers=max(1, int(max_workers)),
        )
    except Exception as e:
        is_access_error, status_code = _is_hf_access_error(e)
        if is_access_error:
            raise HuggingFaceAccessError(str(repo_id), raw_message=str(e), status_code=status_code) from e
        raise
    return Path(str(resolved))


def download_model_preset(
    preset: VisionModelDownloadPreset,
    *,
    model_dir: Optional[Path] = None,
    token: Optional[str] = None,
    max_workers: int = 4,
) -> Path:
    """Download a curated preset into the Hugging Face cache.

    Existing `~/models/<preset.local_dir_name>` trees are migrated into the same
    cache layout on first use so older installs do not need a manual re-download.
    """

    required_files, require_weight_files = _snapshot_requirements_for_preset(preset)
    cache_root = default_hf_cache_root()
    legacy_root = Path(model_dir).expanduser() if model_dir is not None else default_legacy_model_root()
    legacy_dir = legacy_root / preset.local_dir_name
    cached = ensure_hf_repo_snapshot(
        preset.repo_id,
        source_dir=legacy_dir,
        cache_dir=str(cache_root),
        cleanup_source=True,
        required_files=required_files,
        require_weight_files=require_weight_files,
    )
    if cached is not None:
        return cached

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "Downloading model presets requires `huggingface_hub`. Install it or use the `hf download` CLI."
        ) from e

    try:
        snapshot = snapshot_download(
            repo_id=preset.repo_id,
            repo_type="model",
            allow_patterns=list(preset.allow_patterns) or None,
            token=resolve_hf_token(token),
            cache_dir=str(cache_root),
            max_workers=max(1, int(max_workers)),
        )
    except Exception as e:
        is_access_error, status_code = _is_hf_access_error(e)
        if is_access_error:
            raise HuggingFaceAccessError(str(preset.repo_id), raw_message=str(e), status_code=status_code) from e
        raise
    resolved = Path(str(snapshot))
    if not hf_snapshot_is_usable(
        resolved,
        required_files=required_files,
        require_weight_files=require_weight_files,
    ):
        raise RuntimeError(
            f"Downloaded snapshot for {preset.repo_id!r} is incomplete or missing model weights: {resolved}\n"
            "This usually means the download was interrupted. Retry the download; if the repo is gated, "
            "accept its terms first and authenticate with a Hugging Face token."
        )
    return resolved


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


def _sdcpp_bundle_download_hint(name: str) -> str:
    return f"abstractvision download-model {name} --provider sdcpp"


def _resolve_cached_sdcpp_companion(
    *,
    requested_name: str,
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
                f"{role} from {repo_id!r}. Run `{_sdcpp_bundle_download_hint(requested_name)}` first."
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

    preset = find_model_preset(
        requested,
        target="gguf",
        engine="stable-diffusion.cpp",
        require_8bit=True,
    )
    bundle = _SDCPP_BUNDLES_BY_REPO_ID.get(str(preset.repo_id).strip().lower(), _SDCPP_BUNDLES.get(preset.key))
    cache_root = default_hf_cache_root()

    required_files, require_weight_files = _snapshot_requirements_for_preset(preset)
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
                f"Run `{_sdcpp_bundle_download_hint(requested)}` first."
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
            model=str(primary_file),
        )

    component_values: Dict[str, str] = {"diffusion_model": str(primary_file)}
    for component in bundle.components:
        component_values[component.role] = str(
            _resolve_cached_sdcpp_companion(
                requested_name=requested,
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
        model=None,
        diffusion_model=component_values.get("diffusion_model"),
        vae=component_values.get("vae"),
        llm=component_values.get("llm"),
        llm_vision=component_values.get("llm_vision"),
    )


def format_model_preset_rows(presets: Sequence[VisionModelDownloadPreset]) -> Iterable[str]:
    rows = [
        (
            p.key,
            p.target,
            p.engine,
            str(p.quantization_bits) if p.quantization_bits is not None else "n/a",
            p.repo_id,
            p.source,
        )
        for p in presets
    ]
    headers = ("key", "target", "engine", "bits", "repo", "source")
    widths = [
        max(len(str(row[i])) for row in [headers, *rows])
        for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    yield fmt.format(*headers)
    yield fmt.format(*("-" * w for w in widths))
    for row in rows:
        yield fmt.format(*row)
