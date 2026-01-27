from __future__ import annotations

import io
import inspect
import os
from contextlib import contextmanager
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
    import sys

    raise OptionalDependencyMissingError(
        f"Optional dependency missing: {name}. Install via: {install_hint} " f"(python={sys.executable})"
    )


def _lazy_import_diffusers():
    try:
        import warnings

        # Some Diffusers modules decorate functions with `torch.autocast(device_type="cuda", ...)`,
        # which emits noisy warnings on non-CUDA machines (including Apple Silicon / MPS).
        warnings.filterwarnings("ignore", message=r".*CUDA is not available.*Disabling autocast.*", category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message=r".*device_type of 'cuda'.*CUDA is not available.*",
            category=UserWarning,
        )
        import diffusers  # type: ignore
        from diffusers import DiffusionPipeline  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Optional dependency missing (or failed to import): diffusers. Install via: pip install 'diffusers'. "
            f"(python={__import__('sys').executable})"
        ) from e

    # AutoPipeline classes are optional here. Some environments may have diffusers installed but fail to import
    # AutoPipeline due to version mismatches with transformers/torch or other optional deps. We can still load
    # many text-to-image models via `DiffusionPipeline` and only require AutoPipeline for i2i/inpaint.
    AutoPipelineForText2Image = None
    AutoPipelineForImage2Image = None
    AutoPipelineForInpainting = None
    Flux2Pipeline = None
    try:
        from diffusers import AutoPipelineForText2Image as _AutoPipelineForText2Image  # type: ignore

        AutoPipelineForText2Image = _AutoPipelineForText2Image
    except Exception:
        pass
    try:
        from diffusers import AutoPipelineForImage2Image as _AutoPipelineForImage2Image  # type: ignore

        AutoPipelineForImage2Image = _AutoPipelineForImage2Image
    except Exception:
        pass
    try:
        from diffusers import AutoPipelineForInpainting as _AutoPipelineForInpainting  # type: ignore

        AutoPipelineForInpainting = _AutoPipelineForInpainting
    except Exception:
        pass
    try:
        from diffusers import Flux2Pipeline as _Flux2Pipeline  # type: ignore

        Flux2Pipeline = _Flux2Pipeline
    except Exception:
        pass

    return (
        DiffusionPipeline,
        AutoPipelineForText2Image,
        AutoPipelineForImage2Image,
        AutoPipelineForInpainting,
        Flux2Pipeline,
        getattr(diffusers, "__version__", "unknown"),
    )


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


@contextmanager
def _hf_offline_env(enabled: bool):
    """Force Hugging Face libraries into offline mode (no network calls)."""
    if not enabled:
        yield
        return

    # These are respected by huggingface_hub / transformers / diffusers.
    # We scope them to the load/call to avoid affecting the whole process when downloads are enabled later.
    vars_to_set = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        # Avoid any telemetry even in edge cases.
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    old = {k: os.environ.get(k) for k in vars_to_set.keys()}
    try:
        for k, v in vars_to_set.items():
            os.environ[k] = v
        yield
    finally:
        for k, prev in old.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


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
    if d == "mps" or d.startswith("mps:"):
        # Prefer bf16 on MPS when available: better numerical range than fp16 (helps large DiT models like Qwen).
        try:
            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None) if backends is not None else None
            is_available = getattr(mps, "is_available", None) if mps is not None else None
            ok = bool(is_available()) if callable(is_available) else False
            if ok and getattr(torch, "bfloat16", None) is not None:
                zeros = getattr(torch, "zeros", None)
                if callable(zeros):
                    t = zeros((1,), device="mps", dtype=torch.bfloat16)
                    del t
                    return torch.bfloat16
        except Exception:
            pass
        # Fallback: fp16 is widely supported and typically fast.
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


def _maybe_upcast_vae_for_mps(torch: Any, pipe: Any, device: str) -> None:
    d = str(device or "").strip().lower()
    if d != "mps" and not d.startswith("mps:"):
        return

    # On Apple Silicon, some pipelines can produce NaNs/black images when decoding with a float16 VAE.
    # A common fix is to keep the main model in fp16 but run VAE encode/decode in fp32.
    #
    # Diffusers pipelines do not consistently cast inputs to `vae.dtype` before calling `vae.encode/decode`.
    # If we upcast only the VAE weights to fp32 while the pipeline still produces fp16 latents/images,
    # PyTorch can raise dtype mismatch errors like:
    #   "Input type (c10::Half) and bias type (float) should be the same"
    #
    # To keep this backend robust across Diffusers versions, when we upcast the VAE we also wrap
    # `vae.encode` and `vae.decode` to cast their tensor inputs to the VAE's dtype.
    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    to_fn = getattr(vae, "to", None)
    if not callable(to_fn):
        return
    dtype = getattr(vae, "dtype", None)
    if dtype == getattr(torch, "float16", None):
        try:
            vae.to(dtype=torch.float32)
            _maybe_cast_vae_inputs_to_dtype(vae)
        except Exception:
            return


def _maybe_cast_pipe_modules_to_dtype(pipe: Any, *, dtype: Any) -> None:
    if dtype is None:
        return

    def _to(module: Any) -> None:
        if module is None:
            return
        to_fn = getattr(module, "to", None)
        if not callable(to_fn):
            return
        try:
            module.to(dtype=dtype)
        except Exception:
            return

    # Best-effort: different pipelines use different component names (unet vs transformer, etc).
    for attr in (
        "transformer",
        "unet",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "prior",
        "vae",
        "safety_checker",
    ):
        _to(getattr(pipe, attr, None))

    vae = getattr(pipe, "vae", None)
    if vae is not None:
        _to(getattr(vae, "encoder", None))
        _to(getattr(vae, "decoder", None))


def _maybe_cast_vae_inputs_to_dtype(vae: Any) -> None:
    if getattr(vae, "_abstractvision_casts_inputs_to_dtype", False):
        return

    try:
        import types

        def _wrap(name: str) -> None:
            orig = getattr(vae, name, None)
            if not callable(orig):
                return

            def wrapper(self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    dtype = getattr(self, "dtype", None)
                    x_dtype = getattr(x, "dtype", None)
                    to_fn = getattr(x, "to", None)
                    if dtype is not None and x_dtype is not None and x_dtype != dtype and callable(to_fn):
                        x = x.to(dtype=dtype)
                except Exception:
                    pass
                return orig(x, *args, **kwargs)

            setattr(vae, name, types.MethodType(wrapper, vae))

        _wrap("encode")
        _wrap("decode")
        setattr(vae, "_abstractvision_casts_inputs_to_dtype", True)
    except Exception:
        return


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
    auto_retry_fp32: bool = False
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

        (
            DiffusionPipeline,
            AutoPipelineForText2Image,
            AutoPipelineForImage2Image,
            AutoPipelineForInpainting,
            Flux2Pipeline,
            diffusers_version,
        ) = _lazy_import_diffusers()
        torch = _lazy_import_torch()
        _require_device_available(torch, self._cfg.device)

        torch_dtype = _torch_dtype_from_str(torch, self._cfg.torch_dtype)
        if torch_dtype is None:
            torch_dtype = _default_torch_dtype_for_device(torch, self._cfg.device)
        common = self._pipeline_common_kwargs()
        if bool(self._cfg.low_cpu_mem_usage):
            common["low_cpu_mem_usage"] = True

        pipe = None
        offline = not bool(self._cfg.allow_download)
        with _hf_offline_env(offline):
            if kind == "t2i":
                # Prefer AutoPipeline when available, but fall back to DiffusionPipeline for robustness.
                if AutoPipelineForText2Image is not None:
                    try:
                        pipe = AutoPipelineForText2Image.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
                    except ValueError as e:
                        msg = str(e)
                        if (
                            "AutoPipeline can't find a pipeline linked to" in msg
                            and "Flux2KleinPipeline" in msg
                            and Flux2Pipeline is not None
                        ):
                            pipe = Flux2Pipeline.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
                        else:
                            pipe = None
                if pipe is None:
                    try:
                        pipe = DiffusionPipeline.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
                    except Exception as e:
                        msg = str(e)
                        if "Flux2KleinPipeline" in msg and Flux2Pipeline is not None:
                            pipe = Flux2Pipeline.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
                        else:
                            raise
            elif kind == "i2i":
                if AutoPipelineForImage2Image is not None:
                    try:
                        pipe = AutoPipelineForImage2Image.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
                    except ValueError as e:
                        msg = str(e)
                        if (
                            "AutoPipeline can't find a pipeline linked to" in msg
                            and "Flux2KleinPipeline" in msg
                            and Flux2Pipeline is not None
                        ):
                            pipe = Flux2Pipeline.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
                        else:
                            pipe = None
                if pipe is None:
                    try:
                        pipe = DiffusionPipeline.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
                    except Exception as e:
                        msg = str(e)
                        if "Flux2KleinPipeline" in msg and Flux2Pipeline is not None:
                            pipe = Flux2Pipeline.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
                        else:
                            raise ValueError(
                                "Diffusers could not load an image-to-image pipeline for this model id. "
                                "Install/upgrade diffusers (and compatible transformers/torch), or use a model repo that "
                                "ships an image-to-image pipeline. "
                                f"(diffusers={diffusers_version})"
                            ) from e
            elif kind == "inpaint":
                if AutoPipelineForInpainting is None:
                    raise ValueError(
                        "Diffusers inpainting pipeline is not available in this environment. "
                        "Install/upgrade diffusers (and compatible transformers/torch). "
                        f"(diffusers={diffusers_version})"
                    )
                pipe = AutoPipelineForInpainting.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
            else:
                raise ValueError(f"Unknown pipeline kind: {kind!r}")

        # Diffusers pipelines support `.to(<device>)` with a string.
        pipe = pipe.to(str(self._cfg.device))
        _maybe_cast_pipe_modules_to_dtype(pipe, dtype=torch_dtype)
        _maybe_upcast_vae_for_mps(torch, pipe, self._cfg.device)
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
        d = str(self._cfg.device or "").strip().lower()
        gen_device = "cpu" if d == "mps" or d.startswith("mps:") else str(self._cfg.device)
        try:
            gen = torch.Generator(device=gen_device)
        except Exception:
            gen = torch.Generator()
        gen.manual_seed(int(seed))
        return gen

    def _is_probably_all_black_image(self, img: Any) -> bool:
        try:
            rgb = img.convert("RGB")
            extrema = rgb.getextrema()
            if isinstance(extrema, tuple) and len(extrema) == 2 and all(isinstance(x, int) for x in extrema):
                _, mx = extrema
                return mx <= 1
            if isinstance(extrema, tuple):
                return all(isinstance(x, tuple) and len(x) == 2 and int(x[1]) <= 1 for x in extrema)
        except Exception:
            return False
        return False

    def _pipe_call(self, pipe: Any, kwargs: Dict[str, Any]):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)
            offline = not bool(self._cfg.allow_download)
            with _hf_offline_env(offline):
                out = pipe(**kwargs)
        had_invalid_cast = any(
            issubclass(getattr(x, "category", Warning), RuntimeWarning)
            and "invalid value encountered in cast" in str(getattr(x, "message", ""))
            for x in w
        )
        return out, had_invalid_cast

    def _maybe_retry_fp32_on_invalid_output(self, *, kind: str, pipe: Any, kwargs: Dict[str, Any]) -> Optional[Any]:
        if not bool(getattr(self._cfg, "auto_retry_fp32", False)):
            return None
        torch = _lazy_import_torch()
        d = str(self._cfg.device or "").strip().lower()
        cfg_dtype = _torch_dtype_from_str(torch, self._cfg.torch_dtype)
        if cfg_dtype is None:
            cfg_dtype = _default_torch_dtype_for_device(torch, self._cfg.device)

        # Currently, we only auto-retry on Apple Silicon / MPS when running fp16,
        # because NaNs/black images are common for some models (e.g. Qwen Image).
        if not (d == "mps" or d.startswith("mps:")):
            return None
        if cfg_dtype != torch.float16:
            return None

        try:
            pipe_fp32 = pipe.to(device=str(self._cfg.device), dtype=torch.float32)
        except Exception:
            try:
                pipe_fp32 = pipe.to(dtype=torch.float32)
                pipe_fp32 = pipe_fp32.to(str(self._cfg.device))
            except Exception:
                return None

        _maybe_upcast_vae_for_mps(torch, pipe_fp32, self._cfg.device)
        self._pipelines[kind] = pipe_fp32
        self._call_params[kind] = _call_param_names(getattr(pipe_fp32, "__call__", None))

        out2, had_invalid_cast2 = self._pipe_call(pipe_fp32, kwargs)
        if had_invalid_cast2:
            raise ValueError(
                "Diffusers produced invalid pixel values (NaNs) while decoding the image "
                "(resulting in an all-black output). "
                "Tried an automatic fp32 retry on MPS and it still failed. "
                "Try setting torch_dtype=float32 explicitly, increasing steps, or use the stable-diffusion.cpp backend."
            )
        return out2

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
                # Some pipelines (e.g. Qwen Image) only enable CFG when a `negative_prompt`
                # is provided (even an empty one). Make `guidance_scale` behave consistently.
                if request.negative_prompt is None and (call_params is None or "negative_prompt" in call_params):
                    kwargs["negative_prompt"] = " "
            else:
                kwargs["guidance_scale"] = float(request.guidance_scale)
        gen = self._seed_generator(request.seed)
        if gen is not None:
            kwargs["generator"] = gen

        if isinstance(request.extra, dict) and request.extra:
            kwargs.update(dict(request.extra))

        out, had_invalid_cast = self._pipe_call(pipe, kwargs)
        retried_fp32 = False
        images = getattr(out, "images", None)
        if not isinstance(images, list) or not images:
            raise ValueError("Diffusers pipeline returned no images")
        if self._is_probably_all_black_image(images[0]):
            out2 = self._maybe_retry_fp32_on_invalid_output(kind="t2i", pipe=pipe, kwargs=kwargs)
            if out2 is not None:
                out = out2
                retried_fp32 = True
                images = getattr(out, "images", None)
                if not isinstance(images, list) or not images:
                    raise ValueError("Diffusers pipeline returned no images")
        if self._is_probably_all_black_image(images[0]):
            raise ValueError(
                "Diffusers produced an all-black image output. "
                + (
                    "An automatic fp32 retry was attempted and it still produced an all-black image. "
                    if retried_fp32
                    else "Try setting torch_dtype=bfloat16 (recommended on MPS) or torch_dtype=float32. "
                )
                + "Try increasing steps, adjusting guidance_scale, or use the stable-diffusion.cpp backend."
            )
        png = self._png_bytes(images[0])
        meta = {"source": "diffusers", "model_id": self._cfg.model_id}
        if retried_fp32:
            meta["retried_fp32"] = True
        if had_invalid_cast:
            meta["had_invalid_cast_warning"] = True
        try:
            current_pipe = self._pipelines.get("t2i", pipe)
            dtype = getattr(current_pipe, "dtype", None)
            device = getattr(current_pipe, "device", None)
            if dtype is not None:
                meta["dtype"] = str(dtype)
            if device is not None:
                meta["device"] = str(device)
        except Exception:
            pass
        return GeneratedAsset(
            media_type="image",
            data=png,
            mime_type="image/png",
            metadata=meta,
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
                if request.negative_prompt is None and (call_params is None or "negative_prompt" in call_params):
                    kwargs["negative_prompt"] = " "
            else:
                kwargs["guidance_scale"] = float(request.guidance_scale)
        gen = self._seed_generator(request.seed)
        if gen is not None:
            kwargs["generator"] = gen

        if isinstance(request.extra, dict) and request.extra:
            kwargs.update(dict(request.extra))

        out, had_invalid_cast = self._pipe_call(pipe, kwargs)
        retried_fp32 = False
        images = getattr(out, "images", None)
        if not isinstance(images, list) or not images:
            raise ValueError("Diffusers pipeline returned no images")
        if self._is_probably_all_black_image(images[0]):
            kind = "inpaint" if request.mask is not None else "i2i"
            out2 = self._maybe_retry_fp32_on_invalid_output(kind=kind, pipe=pipe, kwargs=kwargs)
            if out2 is not None:
                out = out2
                retried_fp32 = True
                images = getattr(out, "images", None)
                if not isinstance(images, list) or not images:
                    raise ValueError("Diffusers pipeline returned no images")
        if self._is_probably_all_black_image(images[0]):
            raise ValueError(
                "Diffusers produced an all-black image output. "
                + (
                    "An automatic fp32 retry was attempted and it still produced an all-black image. "
                    if retried_fp32
                    else "Try setting torch_dtype=bfloat16 (recommended on MPS) or torch_dtype=float32. "
                )
                + "Try increasing steps, adjusting guidance_scale, or use the stable-diffusion.cpp backend."
            )
        png = self._png_bytes(images[0])
        meta = {"source": "diffusers", "model_id": self._cfg.model_id}
        if retried_fp32:
            meta["retried_fp32"] = True
        if had_invalid_cast:
            meta["had_invalid_cast_warning"] = True
        try:
            kind = "inpaint" if request.mask is not None else "i2i"
            current_pipe = self._pipelines.get(kind, pipe)
            dtype = getattr(current_pipe, "dtype", None)
            device = getattr(current_pipe, "device", None)
            if dtype is not None:
                meta["dtype"] = str(dtype)
            if device is not None:
                meta["device"] = str(device)
        except Exception:
            pass
        return GeneratedAsset(
            media_type="image",
            data=png,
            mime_type="image/png",
            metadata=meta,
        )

    def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]:
        raise CapabilityNotSupportedError("HuggingFaceDiffusersVisionBackend does not implement multi-view generation.")

    def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
        raise CapabilityNotSupportedError("HuggingFaceDiffusersVisionBackend does not implement text_to_video (phase 2).")

    def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
        raise CapabilityNotSupportedError("HuggingFaceDiffusersVisionBackend does not implement image_to_video (phase 2).")
