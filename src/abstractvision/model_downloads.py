from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


DEFAULT_MODEL_DIR = Path("~/models")


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
    "*.md",
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
        aliases=("flux2-klein-4b", "flux-klein-4b", "klein-4b", "klein4b", "flux4b"),
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
        repo_id="unsloth/FLUX.2-klein-4B-GGUF",
        target="gguf",
        engine="stable-diffusion.cpp",
        local_dir_name="flux2-klein-4b-q8_0-gguf",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-4B",
        source="curated-community-gguf",
        aliases=("flux2-klein-4b", "flux-klein-4b", "klein-4b", "klein4b", "flux4b"),
        allow_patterns=("README.md", "LICENSE*", "flux-2-klein-4b-Q8_0.gguf"),
        notes="Q8_0 GGUF for stable-diffusion.cpp style runtimes.",
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
        aliases=("flux2-klein-9b", "flux-klein-9b", "klein-9b", "klein9b", "flux9b"),
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
        repo_id="unsloth/FLUX.2-klein-9B-GGUF",
        target="gguf",
        engine="stable-diffusion.cpp",
        local_dir_name="flux2-klein-9b-q8_0-gguf",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.2-klein-9B",
        source="curated-community-gguf",
        aliases=("flux2-klein-9b", "flux-klein-9b", "klein-9b", "klein9b", "flux9b"),
        allow_patterns=("README.md", "LICENSE*", "flux-2-klein-9b-Q8_0.gguf"),
        notes="Q8_0 GGUF for stable-diffusion.cpp style runtimes.",
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
        key="flux1-dev",
        display_name="FLUX.1 dev mflux MLX 8-bit",
        repo_id="dhairyashil/FLUX.1-dev-mflux-8bit",
        target="mlx",
        engine="mflux",
        local_dir_name="flux1-dev-mlx-8bit",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.1-dev",
        source="curated-community-mflux",
        aliases=("flux1-dev", "flux-1-dev", "flux.1-dev", "black-forest-labs/FLUX.1-dev"),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="Community mflux MLX 8-bit conversion for Apple Silicon.",
        source_priority=30,
    ),
    VisionModelDownloadPreset(
        key="flux1-schnell",
        display_name="FLUX.1 schnell mflux MLX 8-bit",
        repo_id="dhairyashil/FLUX.1-schnell-mflux-8bit",
        target="mlx",
        engine="mflux",
        local_dir_name="flux1-schnell-mlx-8bit",
        quantization_bits=8,
        upstream_repo_id="black-forest-labs/FLUX.1-schnell",
        source="curated-community-mflux",
        aliases=(
            "flux1-schnell",
            "flux-1-schnell",
            "flux.1-schnell",
            "black-forest-labs/FLUX.1-schnell",
        ),
        allow_patterns=_COMMON_MLX_PATTERNS,
        notes="Community mflux MLX 8-bit conversion for Apple Silicon.",
        source_priority=30,
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
        aliases=("z-image-turbo", "zimage-turbo", "z-image", "tongyi-z-image-turbo"),
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
        repo_id="runwayml/stable-diffusion-v1-5",
        target="diffusers",
        engine="diffusers",
        local_dir_name="stable-diffusion-v1-5-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="official",
        aliases=("stable-diffusion", "sd", "sd15", "stable-diffusion-v1-5", "runwayml/stable-diffusion-v1-5"),
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
        repo_id="runwayml/stable-diffusion-inpainting",
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
        key="qwen-image-lightning",
        display_name="Qwen-Image Lightning (Diffusers)",
        repo_id="lightx2v/Qwen-Image-Lightning",
        target="diffusers",
        engine="diffusers",
        local_dir_name="qwen-image-lightning-diffusers",
        quantization_bits=16,
        upstream_repo_id=None,
        source="community",
        aliases=("qwen-image-lightning", "lightx2v/Qwen-Image-Lightning"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="#FALLBACK: community Diffusers snapshot (not 8-bit).",
        source_priority=95,
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
        source="community",
        aliases=("playground-v2.5", "playground-v2.5-1024", "playgroundai/playground-v2.5-1024px-aesthetic"),
        allow_patterns=_COMMON_DIFFUSERS_PATTERNS,
        notes="#FALLBACK: community Diffusers snapshot (not 8-bit).",
        source_priority=95,
    ),
)


def default_model_target() -> str:
    if sys.platform == "darwin":
        return "mlx"
    return "gguf"


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
    selected_target, selected_engine = resolve_model_target_and_engine(target=target, engine=engine)
    out: List[VisionModelDownloadPreset] = []
    for preset in _PRESETS:
        if not include_all_targets and preset.target != selected_target:
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
    selected_target, selected_engine = resolve_model_target_and_engine(target=target, engine=engine)

    def matches(preset: VisionModelDownloadPreset) -> bool:
        aliases = {a.lower() for a in preset.aliases}
        repo_ids = {preset.repo_id.lower()}
        if preset.upstream_repo_id:
            repo_ids.add(preset.upstream_repo_id.lower())
        return requested == preset.key or requested in aliases or requested in repo_ids

    candidates = [
        p
        for p in _PRESETS
        if p.target == selected_target
        and (selected_engine is None or p.engine == selected_engine)
        and matches(p)
    ]
    if require_8bit:
        candidates = [p for p in candidates if p.quantization_bits == 8]
    if candidates:
        return sorted(candidates, key=lambda p: (p.source_priority, p.repo_id))[0]

    known = sorted({p.key for p in _PRESETS}.union(*(set(p.aliases) for p in _PRESETS)))
    any_engine_matches = [p for p in _PRESETS if matches(p)]
    target_matches = [
        p
        for p in _PRESETS
        if (selected_engine is None or p.engine == selected_engine) and matches(p)
    ]

    if raw_target in {"", "auto", "default"} and selected_engine is None and any_engine_matches:
        # If the caller left target/engine on defaults, pick a sensible fallback
        # when there is only one possible artifact target for this preset key.
        possible_targets = sorted({p.target for p in any_engine_matches})
        if len(possible_targets) == 1:
            fallback_target = possible_targets[0]
            fallback = [p for p in any_engine_matches if p.target == fallback_target]
            if require_8bit:
                eight_bit = [p for p in fallback if p.quantization_bits == 8]
                if eight_bit:
                    fallback = eight_bit
                else:
                    raise ValueError(
                        f"No 8-bit preset for {name!r}. Available target/engine choices: "
                        + ", ".join(sorted({f"{p.target}/{p.engine}:{p.repo_id}" for p in any_engine_matches}))
                        + ". Use --allow-non-8bit, or select an explicit --target/--provider that has an 8-bit artifact."
                    )
            return sorted(fallback, key=lambda p: (p.source_priority, p.repo_id))[0]

    if target_matches:
        available = ", ".join(sorted({f"{p.target}/{p.engine}:{p.repo_id}" for p in target_matches}))
        engine_msg = f" and engine {selected_engine!r}" if selected_engine else ""
        raise ValueError(
            f"No {'8-bit ' if require_8bit else ''}preset for {name!r} on target {selected_target!r}{engine_msg}. "
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
            f"No {'8-bit ' if require_8bit else ''}preset for {name!r} on target {selected_target!r}. "
            f"Available target/engine choices: {available}"
        )
    raise ValueError(f"Unknown vision model preset {name!r}. Known presets: {', '.join(known)}")


def default_download_root() -> Path:
    return Path(os.environ.get("ABSTRACTVISION_MODEL_DIR") or DEFAULT_MODEL_DIR).expanduser()


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

    resolved = snapshot_download(
        repo_id=str(repo_id),
        repo_type="model",
        revision=str(revision) if revision else None,
        token=token or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
        cache_dir=str(cache_dir.expanduser()) if cache_dir else None,
        local_files_only=bool(local_files_only),
        max_workers=max(1, int(max_workers)),
    )
    return Path(str(resolved))


def download_model_preset(
    preset: VisionModelDownloadPreset,
    *,
    model_dir: Optional[Path] = None,
    token: Optional[str] = None,
    max_workers: int = 4,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "Downloading model presets requires `huggingface_hub`. Install it or use the `hf download` CLI."
        ) from e

    root = Path(model_dir).expanduser() if model_dir is not None else default_download_root()
    local_dir = root / preset.local_dir_name
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=preset.repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        allow_patterns=list(preset.allow_patterns) or None,
        token=token or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        max_workers=max(1, int(max_workers)),
    )
    return local_dir


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
