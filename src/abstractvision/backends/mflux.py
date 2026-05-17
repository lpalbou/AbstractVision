from __future__ import annotations

import random
import tempfile
import importlib.util
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from ..errors import CapabilityNotSupportedError, OptionalDependencyMissingError
from ..model_downloads import (
    default_download_root,
    download_hf_repo_snapshot,
    find_model_preset,
    looks_like_hf_repo_id,
    model_presets,
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
    "flux2-klein-4b": "flux2-klein-4b",
    "flux-klein-4b": "flux2-klein-4b",
    "klein-4b": "flux2-klein-4b",
    "black-forest-labs/flux.2-klein-9b": "flux2-klein-9b",
    "deepsweet/flux.2-klein-9b-mlx-q8": "flux2-klein-9b",
    "flux2-klein-9b": "flux2-klein-9b",
    "flux-klein-9b": "flux2-klein-9b",
    "klein-9b": "flux2-klein-9b",
    "tongyi-mai/z-image-turbo": "z-image-turbo",
    "carsenk/z-image-turbo-mflux-8bit": "z-image-turbo",
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


def _first_available_mflux_preset(model_dir: Optional[str]) -> Any:
    root = Path(model_dir).expanduser() if model_dir else default_download_root()
    for preset in model_presets(target="mlx", engine="mflux", include_non_8bit=False):
        local_dir = root / preset.local_dir_name
        if _has_model_files(local_dir):
            return preset
    return None


class MFluxVisionBackend(VisionBackend):
    """Local Apple Silicon backend for MFLUX-compatible MLX image models."""

    def __init__(self, *, config: MFluxBackendConfig):
        self._cfg = config
        self._model: Any = None
        self._model_key: Optional[Tuple[Any, ...]] = None
        self._resolved_model_path: Optional[str] = None
        self._resolved_base_model: Optional[str] = None

    def unload(self) -> None:
        self._model = None
        self._model_key = None
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    def preload(self) -> None:
        self._ensure_model()

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
        root = Path(self._cfg.model_dir).expanduser() if self._cfg.model_dir else default_download_root()
        out = []
        for preset in model_presets(target="mlx", engine="mflux", include_non_8bit=False):
            local_dir = root / preset.local_dir_name
            if not _has_model_files(local_dir):
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
                        "local_dir": str(local_dir),
                        "repo_id": preset.repo_id,
                        "upstream_repo_id": preset.upstream_repo_id,
                        "quantization_bits": preset.quantization_bits,
                        **parameter_metadata,
                    },
                )
            )
        return out

    def _resolve_model(self) -> Tuple[str, str]:
        configured_model = str(self._cfg.model or "").strip()
        configured_base = _infer_base_model(self._cfg.base_model)
        root = Path(self._cfg.model_dir).expanduser() if self._cfg.model_dir else default_download_root()

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
                local_dir = root / preset.local_dir_name
                if _has_model_files(local_dir):
                    return str(local_dir), configured_base or preset.key
                if not self._cfg.allow_download:
                    raise OptionalDependencyMissingError(
                        f"MFLUX model preset {preset.key!r} is not downloaded at {local_dir}. "
                        f"Run: abstractvision download-model {preset.key} --provider mflux"
                    )
                return configured_model, configured_base or preset.key

            if _looks_like_path(configured_model):
                raise OptionalDependencyMissingError(f"MFLUX model path does not exist: {configured_model}")
            if looks_like_hf_repo_id(configured_model) and not self._cfg.allow_download:
                try:
                    cached = download_hf_repo_snapshot(configured_model, local_files_only=True)
                except Exception as e:
                    raise OptionalDependencyMissingError(
                        f"MFLUX model repo {configured_model!r} is not cached locally. "
                        "Pre-download it with `abstractvision download-model <org/name>` "
                        "or set ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD=1 to permit downloads."
                    ) from e
                base = configured_base or _infer_base_model(configured_model)
                if not base:
                    raise OptionalDependencyMissingError(
                        "Could not infer MFLUX base model. Set vision_mflux_base_model / "
                        "ABSTRACTVISION_MFLUX_BASE_MODEL."
                    )
                return str(cached), base
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
            return configured_model, base

        preset = _first_available_mflux_preset(self._cfg.model_dir)
        if preset is not None:
            return str(root / preset.local_dir_name), configured_base or preset.key

        raise OptionalDependencyMissingError(
            "MFLUX backend is not configured and no downloaded MFLUX preset was found. "
            "Set vision_mflux_model / ABSTRACTVISION_MFLUX_MODEL or run "
            "`abstractvision download-model flux2-klein-4b --provider mflux`."
        )

    def _ensure_model(self) -> Tuple[Any, _MFluxModelDef]:
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

    def _generate(
        self,
        request: ImageGenerationRequest,
        *,
        image_path: Optional[Path] = None,
        image_strength: Optional[float] = None,
    ) -> GeneratedAsset:
        model, model_def = self._ensure_model()
        extra = dict(request.extra or {})
        seed = int(request.seed) if request.seed is not None else random.randint(0, 1_000_000_000)
        steps = int(request.steps) if request.steps is not None else model_def.default_steps
        if model_def.family == "flux2" and steps < 2:
            raise ValueError("MFLUX FLUX.2 generation requires steps >= 2.")
        width = int(request.width) if request.width is not None else int(self._cfg.default_width)
        height = int(request.height) if request.height is not None else int(self._cfg.default_height)
        guidance = (
            float(request.guidance_scale)
            if request.guidance_scale is not None
            else model_def.default_guidance
        )

        if request.negative_prompt and not model_def.supports_negative_prompt:
            raise CapabilityNotSupportedError("MFLUX FLUX.2 models do not support negative_prompt.")
        if (
            guidance is not None
            and not model_def.supports_guidance_override
            and abs(float(guidance) - float(model_def.default_guidance or 0.0)) > 1e-9
        ):
            raise CapabilityNotSupportedError("MFLUX FLUX.2 distilled models require guidance_scale=1.0.")

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
        return self._generate(request)

    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
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
            return self._generate(gen_req, image_path=image_path, image_strength=image_strength)

    def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]:
        raise CapabilityNotSupportedError("MFLUX backend does not implement multi-view generation.")

    def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
        raise CapabilityNotSupportedError("MFLUX backend does not implement text_to_video.")

    def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
        raise CapabilityNotSupportedError("MFLUX backend does not implement image_to_video.")
