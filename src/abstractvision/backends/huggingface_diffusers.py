from __future__ import annotations

import io
import inspect
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
        import warnings

        # Some Diffusers modules decorate functions with `torch.autocast(device_type="cuda", ...)`,
        # which emits noisy warnings on non-CUDA machines (including Apple Silicon / MPS).
        warnings.filterwarnings("ignore", message=r".*CUDA is not available.*Disabling autocast\\..*", category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message=r".*device_type of 'cuda'.*CUDA is not available\\..*",
            category=UserWarning,
        )
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


def _default_torch_dtype_for_device(torch: Any, device: str) -> Any:
    d = str(device or "").strip().lower()
    if not d:
        return None
    if d.startswith("cuda"):
        return torch.float16
    # On Apple Silicon, float16 on MPS is the practical default for memory/speed.
    # Some models may produce NaNs/black images in fp16; in that case, override with `torch_dtype=float32`.
    if d == "mps" or d.startswith("mps:"):
        return torch.float16
    return None


def _require_device_available(torch: Any, device: str) -> None:
    d = str(device or "").strip().lower()
    if not d:
        return

    if d.startswith("cuda"):
        cuda = getattr(torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None) if cuda is not None else None
        ok = bool(is_available()) if callable(is_available) else False
        if not ok:
            raise ValueError(
                "Device 'cuda' was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or use device='cpu'."
            )

    if d == "mps" or d.startswith("mps:"):
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        is_available = getattr(mps, "is_available", None) if mps is not None else None
        ok = bool(is_available()) if callable(is_available) else False
        if not ok:
            raise ValueError(
                "Device 'mps' was requested, but torch.backends.mps.is_available() is False. "
                "On macOS this typically means you are not using an Apple Silicon + MPS-enabled PyTorch build. "
                "Use device='cpu', or use the stable-diffusion.cpp (sd-cli) backend for GGUF models."
            )


def _call_param_names(fn: Any) -> Optional[set[str]]:
    try:
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            if p.kind == p.VAR_KEYWORD:
                return None
        return {str(k) for k in sig.parameters.keys() if str(k) != "self"}
    except Exception:
        return None


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
    low_cpu_mem_usage: bool = True


class HuggingFaceDiffusersVisionBackend(VisionBackend):
    """Local generative vision backend using HuggingFace Diffusers (images only, phase 1)."""

    def __init__(self, *, config: HuggingFaceDiffusersBackendConfig):
        self._cfg = config
        self._pipelines: Dict[str, Any] = {}
        self._call_params: Dict[str, Optional[set[str]]] = {}

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
        _require_device_available(torch, self._cfg.device)

        torch_dtype = _torch_dtype_from_str(torch, self._cfg.torch_dtype)
        if torch_dtype is None:
            torch_dtype = _default_torch_dtype_for_device(torch, self._cfg.device)
        common = self._pipeline_common_kwargs()
        if bool(self._cfg.low_cpu_mem_usage):
            common["low_cpu_mem_usage"] = True

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
        self._call_params[kind] = _call_param_names(getattr(pipe, "__call__", None))
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
        call_params = self._call_params.get("t2i")

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
            if call_params is not None and "true_cfg_scale" in call_params:
                kwargs["true_cfg_scale"] = float(request.guidance_scale)
            else:
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
            call_params = self._call_params.get("inpaint")
        else:
            pipe = self._get_or_load_pipeline("i2i")
            call_params = self._call_params.get("i2i")

        img = self._pil_from_bytes(request.image)
        kwargs: Dict[str, Any] = {"prompt": request.prompt, "image": img}
        if request.mask is not None:
            kwargs["mask_image"] = self._pil_from_bytes(request.mask)
        if request.negative_prompt is not None:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.steps is not None:
            kwargs["num_inference_steps"] = int(request.steps)
        if request.guidance_scale is not None:
            if call_params is not None and "true_cfg_scale" in call_params:
                kwargs["true_cfg_scale"] = float(request.guidance_scale)
            else:
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
