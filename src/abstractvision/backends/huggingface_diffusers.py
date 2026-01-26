from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..errors import CapabilityNotSupportedError, OptionalDependencyMissingError
from ..types import (
    GeneratedAsset,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageToVideoRequest,
    MultiAngleRequest,
    VideoGenerationRequest,
    VisionBackendCapabilities,
)
from .base_backend import VisionBackend


def _require_optional_dep(name: str, install_hint: str) -> None:
    raise OptionalDependencyMissingError(f"Optional dependency missing: {name}. Install via: {install_hint}")


def _lazy_import_diffusers():
    try:
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting, AutoPipelineForText2Image  # type: ignore
    except Exception:  # pragma: no cover
        _require_optional_dep("diffusers", "pip install 'diffusers'")
    return AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting


def _lazy_import_torch():
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        _require_optional_dep("torch", "pip install 'torch'")
    return torch


def _lazy_import_pil():
    try:
        from PIL import Image  # type: ignore
    except Exception:  # pragma: no cover
        _require_optional_dep("pillow", "pip install 'pillow'")
    return Image


def _torch_dtype_from_str(torch: Any, value: Optional[str]) -> Any:
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v or v == "auto":
        return None
    if v in {"float16", "fp16"}:
        return torch.float16
    if v in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if v in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {value!r}")


@dataclass(frozen=True)
class HuggingFaceDiffusersBackendConfig:
    """Config for a local Diffusers backend.

    Notes:
    - By default, this backend will not download models (`allow_download=False`).
      Pre-download models via `huggingface-cli download ...` or provide an on-disk path as `model_id`.
    """

    model_id: str
    device: str = "cpu"  # "cpu" | "cuda" | "mps" | ...
    torch_dtype: Optional[str] = None  # "float16" | "bfloat16" | "float32" | None
    allow_download: bool = False
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    use_safetensors: bool = True


class HuggingFaceDiffusersVisionBackend(VisionBackend):
    """Local generative vision backend using HuggingFace Diffusers (images only, phase 1)."""

    def __init__(self, *, config: HuggingFaceDiffusersBackendConfig):
        self._cfg = config
        self._pipelines: Dict[str, Any] = {}

    def get_capabilities(self) -> VisionBackendCapabilities:
        return VisionBackendCapabilities(
            supported_tasks=["text_to_image", "image_to_image"],
            supports_mask=None,  # depends on whether inpaint pipeline loads for the model
        )

    def _pipeline_common_kwargs(self) -> Dict[str, Any]:
        local_files_only = not bool(self._cfg.allow_download)
        kwargs: Dict[str, Any] = {
            "local_files_only": local_files_only,
            "use_safetensors": bool(self._cfg.use_safetensors),
        }
        if self._cfg.cache_dir:
            kwargs["cache_dir"] = str(self._cfg.cache_dir)
        if self._cfg.revision:
            kwargs["revision"] = str(self._cfg.revision)
        if self._cfg.variant:
            kwargs["variant"] = str(self._cfg.variant)
        return kwargs

    def _get_or_load_pipeline(self, kind: str) -> Any:
        existing = self._pipelines.get(kind)
        if existing is not None:
            return existing

        AutoPipelineForText2Image, AutoPipelineForImage2Image, AutoPipelineForInpainting = _lazy_import_diffusers()
        torch = _lazy_import_torch()

        torch_dtype = _torch_dtype_from_str(torch, self._cfg.torch_dtype)
        common = self._pipeline_common_kwargs()

        if kind == "t2i":
            pipe = AutoPipelineForText2Image.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
        elif kind == "i2i":
            pipe = AutoPipelineForImage2Image.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
        elif kind == "inpaint":
            pipe = AutoPipelineForInpainting.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
        else:
            raise ValueError(f"Unknown pipeline kind: {kind!r}")

        # Diffusers pipelines support `.to(<device>)` with a string.
        pipe = pipe.to(str(self._cfg.device))
        self._pipelines[kind] = pipe
        return pipe

    def _pil_from_bytes(self, data: bytes):
        Image = _lazy_import_pil()
        img = Image.open(io.BytesIO(bytes(data)))
        # Many pipelines expect RGB.
        return img.convert("RGB")

    def _png_bytes(self, img) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _seed_generator(self, seed: Optional[int]):
        if seed is None:
            return None
        torch = _lazy_import_torch()
        try:
            gen = torch.Generator(device=str(self._cfg.device))
        except Exception:
            gen = torch.Generator()
        gen.manual_seed(int(seed))
        return gen

    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
        pipe = self._get_or_load_pipeline("t2i")

        kwargs: Dict[str, Any] = {
            "prompt": request.prompt,
        }
        if request.negative_prompt is not None:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.width is not None:
            kwargs["width"] = int(request.width)
        if request.height is not None:
            kwargs["height"] = int(request.height)
        if request.steps is not None:
            kwargs["num_inference_steps"] = int(request.steps)
        if request.guidance_scale is not None:
            kwargs["guidance_scale"] = float(request.guidance_scale)
        gen = self._seed_generator(request.seed)
        if gen is not None:
            kwargs["generator"] = gen

        if isinstance(request.extra, dict) and request.extra:
            kwargs.update(dict(request.extra))

        out = pipe(**kwargs)
        images = getattr(out, "images", None)
        if not isinstance(images, list) or not images:
            raise ValueError("Diffusers pipeline returned no images")
        png = self._png_bytes(images[0])
        return GeneratedAsset(
            media_type="image",
            data=png,
            mime_type="image/png",
            metadata={"source": "diffusers", "model_id": self._cfg.model_id},
        )

    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
        if request.mask is not None:
            pipe = self._get_or_load_pipeline("inpaint")
        else:
            pipe = self._get_or_load_pipeline("i2i")

        img = self._pil_from_bytes(request.image)
        kwargs: Dict[str, Any] = {"prompt": request.prompt, "image": img}
        if request.mask is not None:
            kwargs["mask_image"] = self._pil_from_bytes(request.mask)
        if request.negative_prompt is not None:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.steps is not None:
            kwargs["num_inference_steps"] = int(request.steps)
        if request.guidance_scale is not None:
            kwargs["guidance_scale"] = float(request.guidance_scale)
        gen = self._seed_generator(request.seed)
        if gen is not None:
            kwargs["generator"] = gen

        if isinstance(request.extra, dict) and request.extra:
            kwargs.update(dict(request.extra))

        out = pipe(**kwargs)
        images = getattr(out, "images", None)
        if not isinstance(images, list) or not images:
            raise ValueError("Diffusers pipeline returned no images")
        png = self._png_bytes(images[0])
        return GeneratedAsset(
            media_type="image",
            data=png,
            mime_type="image/png",
            metadata={"source": "diffusers", "model_id": self._cfg.model_id},
        )

    def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]:
        raise CapabilityNotSupportedError("HuggingFaceDiffusersVisionBackend does not implement multi-view generation.")

    def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
        raise CapabilityNotSupportedError("HuggingFaceDiffusersVisionBackend does not implement text_to_video (phase 2).")

    def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
        raise CapabilityNotSupportedError("HuggingFaceDiffusersVisionBackend does not implement image_to_video (phase 2).")

