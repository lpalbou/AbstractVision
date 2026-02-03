from __future__ import annotations

import io
import inspect
import os
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

    return (
        DiffusionPipeline,
        AutoPipelineForText2Image,
        AutoPipelineForImage2Image,
        AutoPipelineForInpainting,
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


_TRANSFORMERS_CLIP_POSITION_IDS_PATCHED = False


def _maybe_patch_transformers_clip_position_ids() -> None:
    """Fix Transformers v5 noisy LOAD REPORTs for common CLIP checkpoints.

    Transformers 5 logs a detailed load report when encountering unexpected keys like
    `*.embeddings.position_ids` in older CLIP checkpoints (e.g. SD1.5 text encoder / safety checker).

    The root cause is a small architecture/state-dict mismatch: those checkpoints include a persistent
    `position_ids` buffer, while newer CLIP embedding classes may not. We re-add that buffer so the
    checkpoint matches the instantiated model and no "UNEXPECTED" keys are reported.
    """

    global _TRANSFORMERS_CLIP_POSITION_IDS_PATCHED
    if _TRANSFORMERS_CLIP_POSITION_IDS_PATCHED:
        return

    try:
        import transformers  # type: ignore
        import torch as _torch  # type: ignore
    except Exception:
        return

    ver = str(getattr(transformers, "__version__", "0"))
    try:
        major = int(ver.split(".", 1)[0])
    except Exception:
        major = 0
    if major < 5:
        _TRANSFORMERS_CLIP_POSITION_IDS_PATCHED = True
        return

    try:
        from transformers.models.clip.modeling_clip import CLIPTextEmbeddings, CLIPVisionEmbeddings  # type: ignore
    except Exception:
        _TRANSFORMERS_CLIP_POSITION_IDS_PATCHED = True
        return

    def _patch(cls: Any) -> None:
        if bool(getattr(cls, "_abstractvision_position_ids_patched", False)):
            return
        orig_init = getattr(cls, "__init__", None)
        if not callable(orig_init):
            return

        def __init__(self, *args, **kwargs):  # type: ignore[no-redef]
            orig_init(self, *args, **kwargs)
            if hasattr(self, "position_ids"):
                # In Transformers 5, `position_ids` is sometimes registered as a non-persistent buffer
                # (`persistent=False`), so it isn't part of the state dict and is reported as UNEXPECTED
                # when loading older checkpoints that include it. Make it persistent.
                try:
                    buffers = getattr(self, "_buffers", None)
                    if isinstance(buffers, dict) and "position_ids" in buffers:
                        non_persistent = getattr(self, "_non_persistent_buffers_set", None)
                        if isinstance(non_persistent, set):
                            non_persistent.discard("position_ids")
                        return
                except Exception:
                    return
            pos_emb = getattr(self, "position_embedding", None)
            num = getattr(pos_emb, "num_embeddings", None) if pos_emb is not None else None
            if num is None:
                return
            try:
                position_ids = _torch.arange(int(num)).unsqueeze(0)
                self.register_buffer("position_ids", position_ids, persistent=True)
            except Exception:
                return

        setattr(cls, "__init__", __init__)
        setattr(cls, "_abstractvision_position_ids_patched", True)

    _patch(CLIPTextEmbeddings)
    _patch(CLIPVisionEmbeddings)
    _TRANSFORMERS_CLIP_POSITION_IDS_PATCHED = True


@contextmanager
def _hf_offline_env(enabled: bool):
    """Control Hugging Face offline mode within a scope.

    When `enabled=True`, we force offline mode (no network calls).
    When `enabled=False`, we force online mode (overrides e.g. HF_HUB_OFFLINE=1 in the user's shell).
    """

    # These are respected by huggingface_hub / transformers / diffusers.
    # We scope them to the load/call to avoid surprising other parts of the process.
    vars_to_set = {
        "HF_HUB_OFFLINE": "1" if enabled else "0",
        "TRANSFORMERS_OFFLINE": "1" if enabled else "0",
        "DIFFUSERS_OFFLINE": "1" if enabled else "0",
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
        # Default to fp16 on Apple Silicon for broad model compatibility.
        # (Some pipelines mix dtypes when using bf16, which can crash with matmul dtype mismatches.)
        #
        # You can still force bf16 explicitly via `torch_dtype=bfloat16`.
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


def _looks_like_dtype_mismatch_error(e: Exception) -> bool:
    msg = str(e or "")
    m = msg.lower()
    return (
        "must have the same dtype" in m
        or ("input type" in m and "bias type" in m and "should be the same" in m)
        or ("expected scalar type" in m and "but found" in m)
    )


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
        "model",
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

    # As a fallback, cast all registered components when available (covers pipelines that don't follow
    # the common attribute naming patterns above).
    comps = getattr(pipe, "components", None)
    if isinstance(comps, dict):
        for v in comps.values():
            _to(v)


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
    - Downloads are enabled by default so a fresh environment can work after a `pip install`.
    - To force offline mode (no network calls / cache-only), set `allow_download=False`.
    """

    model_id: str
    device: str = "cpu"  # "cpu" | "cuda" | "mps" | "auto" | ...
    torch_dtype: Optional[str] = None  # "float16" | "bfloat16" | "float32" | None
    allow_download: bool = True
    auto_retry_fp32: bool = True
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
        self._fused_lora_signature: Dict[str, Optional[str]] = {}
        self._rapid_transformer_key: Optional[str] = None
        self._rapid_transformer: Any = None
        self._resolved_device: Optional[str] = None

    def _effective_device(self, torch: Any) -> str:
        if self._resolved_device is not None:
            return self._resolved_device

        raw = str(getattr(self._cfg, "device", "") or "").strip()
        d = raw.lower()
        if not d or d in {"auto", "default"}:
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
                self._resolved_device = "cuda"
                return self._resolved_device

            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None) if backends is not None else None
            if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
                self._resolved_device = "mps"
                return self._resolved_device

            self._resolved_device = "cpu"
            return self._resolved_device

        # Normalize common spellings but preserve explicit device indexes (e.g. "cuda:0").
        if d == "gpu":
            self._resolved_device = "cuda"
        else:
            self._resolved_device = raw
        return self._resolved_device

    def preload(self) -> None:
        # Best-effort: preload the most common pipeline.
        self._get_or_load_pipeline("t2i")

    def unload(self) -> None:
        # Best-effort: release pipelines and GPU cache.
        pipes = list(self._pipelines.values())
        self._pipelines.clear()
        self._call_params.clear()
        self._fused_lora_signature.clear()
        self._rapid_transformer_key = None
        self._rapid_transformer = None

        # Drop references and aggressively collect.
        try:
            for p in pipes:
                try:
                    # Try to free adapter weights.
                    unfuse = getattr(p, "unfuse_lora", None)
                    if callable(unfuse):
                        unfuse()
                except Exception:
                    pass
                try:
                    unload = getattr(p, "unload_lora_weights", None)
                    if callable(unload):
                        unload()
                except Exception:
                    pass
        finally:
            pipes = []

        try:
            import gc

            gc.collect()
        except Exception:
            pass

        try:
            torch = _lazy_import_torch()
            cuda = getattr(torch, "cuda", None)
            if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
                empty = getattr(cuda, "empty_cache", None)
                if callable(empty):
                    empty()
                ipc_collect = getattr(cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()

            mps = getattr(torch, "mps", None)
            empty_mps = getattr(mps, "empty_cache", None) if mps is not None else None
            if callable(empty_mps):
                empty_mps()
        except Exception:
            pass

    def _lora_signature(self, loras: List[Dict[str, Any]]) -> Optional[str]:
        if not loras:
            return None
        parts: List[str] = []
        for spec in sorted(loras, key=lambda x: str(x.get("source") or "")):
            parts.append(
                "|".join(
                    [
                        str(spec.get("source") or ""),
                        str(spec.get("subfolder") or ""),
                        str(spec.get("weight_name") or ""),
                        str(spec.get("scale") or 1.0),
                    ]
                )
            )
        combined = "::".join(parts)
        return hashlib.md5(combined.encode("utf-8")).hexdigest()[:12]

    def _parse_loras(self, extra: Any) -> List[Dict[str, Any]]:
        if not isinstance(extra, dict) or not extra:
            return []

        raw: Any = None
        for k in ("loras", "loras_json", "lora", "lora_json"):
            if k in extra and extra.get(k) is not None:
                raw = extra.get(k)
                break
        if raw is None:
            return []

        import json

        items: Any = raw
        if isinstance(raw, str):
            s = raw.strip()
            if not s:
                return []
            # Prefer JSON, but allow a simple comma-separated list of sources.
            if s.startswith("[") or s.startswith("{"):
                try:
                    items = json.loads(s)
                except Exception:
                    items = raw
            if isinstance(items, str):
                parts = [p.strip() for p in items.split(",") if p.strip()]
                items = [{"source": p} for p in parts]

        if isinstance(items, dict):
            items = [items]
        if isinstance(items, str):
            return [{"source": items.strip()}] if items.strip() else []
        if not isinstance(items, list):
            return []

        out: List[Dict[str, Any]] = []
        for el in items:
            if isinstance(el, str):
                src = el.strip()
                if src:
                    out.append({"source": src, "scale": 1.0})
                continue
            if not isinstance(el, dict):
                continue
            src = str(el.get("source") or "").strip()
            if not src:
                continue
            spec: Dict[str, Any] = {"source": src}
            if el.get("subfolder") is not None:
                spec["subfolder"] = str(el.get("subfolder") or "").strip() or None
            if el.get("weight_name") is not None:
                spec["weight_name"] = str(el.get("weight_name") or "").strip() or None
            if el.get("adapter_name") is not None:
                spec["adapter_name"] = str(el.get("adapter_name") or "").strip() or None
            try:
                spec["scale"] = float(el.get("scale") if el.get("scale") is not None else 1.0)
            except Exception:
                spec["scale"] = 1.0
            out.append(spec)
        return out

    def _resolved_adapter_name(self, spec: Dict[str, Any]) -> str:
        name = str(spec.get("adapter_name") or "").strip()
        if name:
            return name
        key = "|".join([str(spec.get("source") or ""), str(spec.get("subfolder") or ""), str(spec.get("weight_name") or "")])
        return "lora_" + hashlib.md5(key.encode("utf-8")).hexdigest()[:12]

    def _apply_loras(self, *, kind: str, pipe: Any, extra: Any) -> Optional[str]:
        loras = self._parse_loras(extra)
        new_sig = self._lora_signature(loras)
        cur_sig = self._fused_lora_signature.get(kind)
        if new_sig == cur_sig:
            return cur_sig

        # Always clear previous adapters before applying a new set.
        if hasattr(pipe, "unfuse_lora"):
            try:
                pipe.unfuse_lora()
            except Exception:
                pass
        if hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

        if not loras:
            self._fused_lora_signature[kind] = None
            return None

        adapter_names: List[str] = []
        adapter_scales: List[float] = []

        with _hf_offline_env(not bool(self._cfg.allow_download)):
            for spec in loras:
                adapter_name = self._resolved_adapter_name(spec)
                adapter_names.append(adapter_name)
                adapter_scales.append(float(spec.get("scale") or 1.0))

                kwargs: Dict[str, Any] = {}
                if spec.get("weight_name"):
                    kwargs["weight_name"] = spec["weight_name"]
                if spec.get("subfolder"):
                    kwargs["subfolder"] = spec["subfolder"]
                kwargs["local_files_only"] = not bool(self._cfg.allow_download)
                if self._cfg.cache_dir:
                    kwargs["cache_dir"] = str(self._cfg.cache_dir)

                load_fn = getattr(pipe, "load_lora_weights", None)
                if not callable(load_fn):
                    raise ValueError("This diffusers pipeline does not support LoRA adapters (missing load_lora_weights).")
                load_fn(spec["source"], adapter_name=adapter_name, **kwargs)

            if hasattr(pipe, "set_adapters"):
                try:
                    pipe.set_adapters(adapter_names, adapter_weights=adapter_scales)
                except Exception:
                    pass

            if hasattr(pipe, "fuse_lora"):
                try:
                    pipe.fuse_lora()
                except Exception:
                    pass

            if hasattr(pipe, "unload_lora_weights"):
                try:
                    pipe.unload_lora_weights()
                except Exception:
                    pass

        self._fused_lora_signature[kind] = new_sig
        return new_sig

    def _maybe_apply_rapid_aio_transformer(self, *, pipe: Any, extra: Any, torch_dtype: Any) -> Optional[str]:
        """Optionally swap the pipeline's transformer with a Rapid-AIO distilled transformer.

        This is primarily useful for Qwen Image Edit pipelines (very fast 4-step inference), but we keep it
        generic: if a pipeline has a `.transformer` module and diffusers provides a compatible transformer
        class, we can hot-swap it.

        Downloads are enabled by default; set allow_download=False for cache-only/offline mode.
        """

        if not isinstance(extra, dict) or not extra:
            return None

        repo = None
        if extra.get("rapid_aio_repo"):
            repo = str(extra.get("rapid_aio_repo") or "").strip()
        elif extra.get("rapid_aio") is True:
            repo = "linoyts/Qwen-Image-Edit-Rapid-AIO"
        elif isinstance(extra.get("rapid_aio"), str) and str(extra.get("rapid_aio")).strip():
            repo = str(extra.get("rapid_aio")).strip()
        if not repo:
            return None

        subfolder = str(extra.get("rapid_aio_subfolder") or "transformer").strip() or "transformer"
        key = f"{repo}|{subfolder}|{torch_dtype}"
        if key == self._rapid_transformer_key and self._rapid_transformer is not None:
            tr = self._rapid_transformer
        else:
            try:
                from diffusers.models import QwenImageTransformer2DModel  # type: ignore
            except Exception:
                raise ValueError(
                    "Rapid-AIO transformer override requires diffusers.models.QwenImageTransformer2DModel, "
                    "which is not available in this diffusers build."
                )
            kwargs: Dict[str, Any] = {"subfolder": subfolder, "local_files_only": not bool(self._cfg.allow_download)}
            if self._cfg.cache_dir:
                kwargs["cache_dir"] = str(self._cfg.cache_dir)
            with _hf_offline_env(not bool(self._cfg.allow_download)):
                tr = QwenImageTransformer2DModel.from_pretrained(repo, torch_dtype=torch_dtype, **kwargs)
            torch = _lazy_import_torch()
            device = self._effective_device(torch)
            try:
                tr = tr.to(device=str(device), dtype=torch_dtype)
            except Exception:
                try:
                    tr = tr.to(dtype=torch_dtype)
                    tr = tr.to(str(device))
                except Exception:
                    pass

            self._rapid_transformer_key = key
            self._rapid_transformer = tr

        if hasattr(pipe, "register_modules"):
            try:
                pipe.register_modules(transformer=tr)
            except Exception:
                setattr(pipe, "transformer", tr)
        else:
            setattr(pipe, "transformer", tr)

        _maybe_cast_pipe_modules_to_dtype(pipe, dtype=torch_dtype)
        return repo

    def get_capabilities(self) -> VisionBackendCapabilities:
        return VisionBackendCapabilities(
            supported_tasks=["text_to_image", "image_to_image"],
            supports_mask=None,  # depends on whether inpaint pipeline loads for the model
        )

    def _pipeline_common_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "local_files_only": not bool(self._cfg.allow_download),
            "use_safetensors": bool(self._cfg.use_safetensors),
        }
        if self._cfg.cache_dir:
            kwargs["cache_dir"] = str(self._cfg.cache_dir)
        if self._cfg.revision:
            kwargs["revision"] = str(self._cfg.revision)
        if self._cfg.variant:
            kwargs["variant"] = str(self._cfg.variant)
        return kwargs

    def _hf_cache_root(self) -> Path:
        if self._cfg.cache_dir:
            return Path(self._cfg.cache_dir).expanduser()
        hub_cache = os.environ.get("HF_HUB_CACHE")
        if hub_cache:
            return Path(hub_cache).expanduser()
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home).expanduser() / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"

    def _resolve_snapshot_dir(self) -> Optional[Path]:
        model_id = str(self._cfg.model_id).strip()
        if not model_id:
            return None

        p = Path(model_id).expanduser()
        if p.exists():
            return p

        if "/" not in model_id:
            return None

        cache_root = self._hf_cache_root()
        repo_dir = cache_root / ("models--" + model_id.replace("/", "--"))
        snaps = repo_dir / "snapshots"
        if not snaps.is_dir():
            return None

        rev = str(self._cfg.revision or "main").strip() or "main"
        ref_file = repo_dir / "refs" / rev
        if ref_file.is_file():
            commit = ref_file.read_text(encoding="utf-8").strip()
            snap_dir = snaps / commit
            if snap_dir.is_dir():
                return snap_dir

        # Fallback: pick the most recently modified snapshot.
        candidates = [d for d in snaps.iterdir() if d.is_dir()]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.stat().st_mtime)

    def _preflight_check_model_index(self) -> None:
        snap = self._resolve_snapshot_dir()
        if snap is None:
            return
        idx_path = snap / "model_index.json"
        if not idx_path.is_file():
            return

        try:
            import json

            model_index = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            return

        class_name = str(model_index.get("_class_name") or "").strip()
        if not class_name:
            return

        (
            _DiffusionPipeline,
            _AutoPipelineForText2Image,
            _AutoPipelineForImage2Image,
            _AutoPipelineForInpainting,
            diffusers_version,
        ) = _lazy_import_diffusers()

        import diffusers as _diffusers  # type: ignore

        if not hasattr(_diffusers, class_name):
            required = str(model_index.get("_diffusers_version") or "unknown")
            install_hint = "pip install -U 'git+https://github.com/huggingface/diffusers@main'"
            install_hint_alt = "pip install -e '.[huggingface-dev]'"
            extra = ""
            if class_name == "Flux2KleinPipeline":
                extra = (
                    " Note: this model uses a different text encoder than the released Flux2Pipeline in diffusers 0.36 "
                    "(Klein uses Qwen3; Flux2Pipeline is built around Mistral3), so a newer diffusers is required."
                )
            raise ValueError(
                f"Diffusers pipeline class {class_name!r} is required by this model, but is not available in your "
                f"installed diffusers ({diffusers_version}). "
                f"The model's model_index.json was authored for diffusers {required}. "
                "This class is not available in the latest PyPI release at the time of writing. "
                f"Install a newer diffusers (offline runtime is still supported): {install_hint}. "
                f"If you're installing AbstractVision from a repo checkout, you can also use: {install_hint_alt}.{extra}"
            )

        # Optional: sanity-check that referenced Transformers classes exist to avoid late failures.
        try:
            import transformers  # type: ignore

            missing_tf: list[str] = []
            for v in model_index.values():
                if (
                    isinstance(v, list)
                    and len(v) == 2
                    and isinstance(v[0], str)
                    and isinstance(v[1], str)
                    and v[0].strip().lower() == "transformers"
                ):
                    tf_cls = v[1].strip()
                    if tf_cls and not hasattr(transformers, tf_cls):
                        missing_tf.append(tf_cls)
            if missing_tf:
                tf_ver = getattr(transformers, "__version__", "unknown")
                raise ValueError(
                    "This model references Transformers classes that are not available in your environment "
                    f"(transformers={tf_ver}): {', '.join(sorted(set(missing_tf)))}. "
                    "Upgrade transformers to a compatible version."
                )
        except ValueError:
            raise
        except Exception:
            pass

    def _get_or_load_pipeline(self, kind: str) -> Any:
        existing = self._pipelines.get(kind)
        if existing is not None:
            return existing

        (
            DiffusionPipeline,
            AutoPipelineForText2Image,
            AutoPipelineForImage2Image,
            AutoPipelineForInpainting,
            diffusers_version,
        ) = _lazy_import_diffusers()
        torch = _lazy_import_torch()
        device = self._effective_device(torch)
        _require_device_available(torch, device)

        self._preflight_check_model_index()
        _maybe_patch_transformers_clip_position_ids()

        torch_dtype = _torch_dtype_from_str(torch, self._cfg.torch_dtype)
        if torch_dtype is None:
            torch_dtype = _default_torch_dtype_for_device(torch, device)
        common = self._pipeline_common_kwargs()
        if bool(self._cfg.low_cpu_mem_usage):
            common["low_cpu_mem_usage"] = True

        # Auto-select checkpoint variants when appropriate (best-effort).
        # Prefer fp16 on GPU backends (CUDA/MPS) to cut memory/disk use, but never on CPU.
        auto_variant: Optional[str] = None
        if not str(getattr(self._cfg, "variant", "") or "").strip() and str(device).strip().lower() != "cpu":
            if torch_dtype == getattr(torch, "float16", object()):
                auto_variant = "fp16"

        def _looks_like_missing_variant_error(e: Exception, variant: str) -> bool:
            msg = str(e or "")
            m = msg.lower()
            v = str(variant or "").strip().lower()
            if not v:
                return False
            return (
                (f".{v}." in m or f" {v} " in m or f"'{v}'" in m)
                and (
                    "no such file" in m
                    or "does not exist" in m
                    or "not found" in m
                    or "is not present" in m
                    or "couldn't find" in m
                    or "cannot find" in m
                )
            )

        def _from_pretrained(cls: Any) -> Any:
            if auto_variant:
                common2 = dict(common)
                common2["variant"] = auto_variant
                try:
                    return cls.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common2)
                except Exception as e:
                    # If the repo doesn't provide the fp16 variant, fall back to regular weights.
                    if _looks_like_missing_variant_error(e, auto_variant):
                        return cls.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)
                    raise
            return cls.from_pretrained(self._cfg.model_id, torch_dtype=torch_dtype, **common)

        def _maybe_raise_offline_missing_model(e: Exception) -> None:
            model_id = str(self._cfg.model_id or "").strip()
            if not model_id or "/" not in model_id:
                return
            # If it's not in cache, provide a clearer message than the upstream
            # "does not appear to have a file named model_index.json" wording.
            if self._resolve_snapshot_dir() is not None:
                return
            msg = str(e)
            if "model_index.json" not in msg:
                return
            raise ValueError(
                f"Model {model_id!r} is not available locally and downloads are disabled. "
                "Either pre-download it (e.g. via `huggingface-cli download ...`) or enable downloads "
                "(set allow_download=True; for AbstractCore Server: set ABSTRACTCORE_VISION_ALLOW_DOWNLOAD=1). "
                "If the model is gated, accept its terms on Hugging Face and set `HF_TOKEN` before downloading."
            ) from e

        pipe = None
        with _hf_offline_env(not bool(self._cfg.allow_download)):
            if kind == "t2i":
                # Prefer AutoPipeline when available, but fall back to DiffusionPipeline for robustness.
                if AutoPipelineForText2Image is not None:
                    try:
                        pipe = _from_pretrained(AutoPipelineForText2Image)
                    except ValueError as e:
                        _maybe_raise_offline_missing_model(e)
                        pipe = None
                if pipe is None:
                    try:
                        pipe = _from_pretrained(DiffusionPipeline)
                    except Exception as e:
                        _maybe_raise_offline_missing_model(e)
                        raise
            elif kind == "i2i":
                if AutoPipelineForImage2Image is not None:
                    try:
                        pipe = _from_pretrained(AutoPipelineForImage2Image)
                    except ValueError as e:
                        _maybe_raise_offline_missing_model(e)
                        pipe = None
                if pipe is None:
                    try:
                        pipe = _from_pretrained(DiffusionPipeline)
                    except Exception as e:
                        _maybe_raise_offline_missing_model(e)
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
                pipe = _from_pretrained(AutoPipelineForInpainting)
            else:
                raise ValueError(f"Unknown pipeline kind: {kind!r}")

        # Diffusers pipelines support `.to(<device>)` with a string.
        pipe = pipe.to(str(device))
        _maybe_cast_pipe_modules_to_dtype(pipe, dtype=torch_dtype)
        _maybe_upcast_vae_for_mps(torch, pipe, device)
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
        d = str(self._effective_device(torch) or "").strip().lower()
        gen_device = "cpu" if d == "mps" or d.startswith("mps:") else str(self._effective_device(torch))
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

        call_kwargs = dict(kwargs)
        if callable(kwargs.get("__abstractvision_progress_callback")):
            progress_cb = kwargs.get("__abstractvision_progress_callback")
            total_steps = kwargs.get("__abstractvision_progress_total_steps")
            try:
                call_kwargs.pop("__abstractvision_progress_callback", None)
                call_kwargs.pop("__abstractvision_progress_total_steps", None)
            except Exception:
                pass
            try:
                call_kwargs = self._inject_progress_kwargs(
                    pipe=pipe,
                    kwargs=call_kwargs,
                    progress_callback=progress_cb,
                    total_steps=int(total_steps) if total_steps is not None else None,
                )
            except Exception:
                # Best-effort: never break inference for progress reporting.
                pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)
            with _hf_offline_env(not bool(self._cfg.allow_download)):
                out = pipe(**call_kwargs)
        had_invalid_cast = any(
            issubclass(getattr(x, "category", Warning), RuntimeWarning)
            and "invalid value encountered in cast" in str(getattr(x, "message", ""))
            for x in w
        )
        return out, had_invalid_cast

    def _pipe_progress_param_names(self, pipe: Any) -> set[str]:
        fn = getattr(pipe, "__call__", None)
        if not callable(fn):
            return set()
        try:
            sig = inspect.signature(fn)
        except Exception:
            return set()
        return {str(k) for k in sig.parameters.keys() if str(k) != "self"}

    def _inject_progress_kwargs(
        self,
        *,
        pipe: Any,
        kwargs: Dict[str, Any],
        progress_callback: Callable[[int, Optional[int]], None],
        total_steps: Optional[int],
    ) -> Dict[str, Any]:
        names = self._pipe_progress_param_names(pipe)
        if not names:
            return kwargs

        if "callback_on_step_end" in names:

            def _on_step_end(*args: Any, **kw: Any) -> Any:
                # Expected signature: (pipe, step, timestep, callback_kwargs)
                step = None
                cb_kwargs = None
                try:
                    if len(args) >= 2:
                        step = args[1]
                    if len(args) >= 4:
                        cb_kwargs = args[3]
                    if cb_kwargs is None:
                        cb_kwargs = kw.get("callback_kwargs")
                except Exception:
                    pass
                try:
                    if step is not None:
                        progress_callback(int(step) + 1, total_steps)
                except Exception:
                    pass
                return cb_kwargs if cb_kwargs is not None else {}

            kwargs["callback_on_step_end"] = _on_step_end
            # Avoid passing large tensors through callback_kwargs unless explicitly requested.
            if "callback_on_step_end_tensor_inputs" in names:
                kwargs.setdefault("callback_on_step_end_tensor_inputs", [])
            return kwargs

        if "callback" in names:

            def _callback(*args: Any, **_kw: Any) -> None:
                # Expected signature: (step, timestep, latents)
                try:
                    if args:
                        progress_callback(int(args[0]) + 1, total_steps)
                except Exception:
                    pass

            kwargs["callback"] = _callback
            if "callback_steps" in names:
                kwargs["callback_steps"] = 1
            return kwargs

        return kwargs

    def _maybe_retry_on_dtype_mismatch(
        self,
        *,
        kind: str,
        pipe: Any,
        kwargs: Dict[str, Any],
        error: Exception,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        total_steps: Optional[int] = None,
    ) -> Optional[Any]:
        if not bool(getattr(self._cfg, "auto_retry_fp32", False)):
            return None
        if not _looks_like_dtype_mismatch_error(error):
            return None

        torch = _lazy_import_torch()
        device = self._effective_device(torch)
        d = str(device or "").strip().lower()
        if not (d == "mps" or d.startswith("mps:")):
            return None

        current_dtype = getattr(pipe, "dtype", None)
        if current_dtype is None:
            current_dtype = _torch_dtype_from_str(torch, self._cfg.torch_dtype) or _default_torch_dtype_for_device(
                torch, device
            )

        candidates: list[Any] = []
        if current_dtype == getattr(torch, "bfloat16", object()):
            candidates.append(torch.float16)
        if current_dtype != getattr(torch, "float32", object()):
            candidates.append(torch.float32)

        for target in candidates:
            try:
                pipe2 = pipe.to(device=str(device), dtype=target)
            except Exception:
                try:
                    pipe2 = pipe.to(dtype=target)
                    pipe2 = pipe2.to(str(device))
                except Exception:
                    continue

            _maybe_upcast_vae_for_mps(torch, pipe2, device)
            self._pipelines[kind] = pipe2
            self._call_params[kind] = _call_param_names(getattr(pipe2, "__call__", None))

            try:
                call_kwargs = dict(kwargs)
                if progress_callback is not None:
                    call_kwargs["__abstractvision_progress_callback"] = progress_callback
                    call_kwargs["__abstractvision_progress_total_steps"] = total_steps
                out2, _had_invalid_cast2 = self._pipe_call(pipe2, call_kwargs)
                return out2
            except Exception:
                continue
        return None

    def _maybe_retry_fp32_on_invalid_output(
        self,
        *,
        kind: str,
        pipe: Any,
        kwargs: Dict[str, Any],
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
        total_steps: Optional[int] = None,
    ) -> Optional[Any]:
        if not bool(getattr(self._cfg, "auto_retry_fp32", False)):
            return None
        torch = _lazy_import_torch()
        device = self._effective_device(torch)
        d = str(device or "").strip().lower()
        cfg_dtype = _torch_dtype_from_str(torch, self._cfg.torch_dtype)
        if cfg_dtype is None:
            cfg_dtype = _default_torch_dtype_for_device(torch, device)

        # Currently, we only auto-retry on Apple Silicon / MPS when running fp16,
        # because NaNs/black images are common for some models (e.g. Qwen Image).
        if not (d == "mps" or d.startswith("mps:")):
            return None
        if cfg_dtype != torch.float16:
            return None

        try:
            pipe_fp32 = pipe.to(device=str(device), dtype=torch.float32)
        except Exception:
            try:
                pipe_fp32 = pipe.to(dtype=torch.float32)
                pipe_fp32 = pipe_fp32.to(str(device))
            except Exception:
                return None

        _maybe_upcast_vae_for_mps(torch, pipe_fp32, device)
        self._pipelines[kind] = pipe_fp32
        self._call_params[kind] = _call_param_names(getattr(pipe_fp32, "__call__", None))

        call_kwargs = dict(kwargs)
        if progress_callback is not None:
            call_kwargs["__abstractvision_progress_callback"] = progress_callback
            call_kwargs["__abstractvision_progress_total_steps"] = total_steps
        out2, had_invalid_cast2 = self._pipe_call(pipe_fp32, call_kwargs)
        if had_invalid_cast2:
            raise ValueError(
                "Diffusers produced invalid pixel values (NaNs) while decoding the image "
                "(resulting in an all-black output). "
                "Tried an automatic fp32 retry on MPS and it still failed. "
                "Try setting torch_dtype=float32 explicitly, increasing steps, or use the stable-diffusion.cpp backend."
            )
        return out2

    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
        return self.generate_image_with_progress(request, progress_callback=None)

    def generate_image_with_progress(
        self,
        request: ImageGenerationRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        pipe = self._get_or_load_pipeline("t2i")
        call_params = self._call_params.get("t2i")
        total_steps = int(request.steps) if request.steps is not None else None

        torch_dtype = getattr(pipe, "dtype", None)
        if torch_dtype is None:
            torch = _lazy_import_torch()
            device = self._effective_device(torch)
            torch_dtype = _torch_dtype_from_str(torch, self._cfg.torch_dtype) or _default_torch_dtype_for_device(torch, device)
        rapid_repo = self._maybe_apply_rapid_aio_transformer(pipe=pipe, extra=request.extra, torch_dtype=torch_dtype)
        lora_sig = self._apply_loras(kind="t2i", pipe=pipe, extra=request.extra)

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

        try:
            call_kwargs = dict(kwargs)
            if progress_callback is not None:
                call_kwargs["__abstractvision_progress_callback"] = progress_callback
                call_kwargs["__abstractvision_progress_total_steps"] = total_steps
            out, had_invalid_cast = self._pipe_call(pipe, call_kwargs)
        except Exception as e:
            out2 = self._maybe_retry_on_dtype_mismatch(
                kind="t2i",
                pipe=pipe,
                kwargs=kwargs,
                error=e,
                progress_callback=progress_callback,
                total_steps=total_steps,
            )
            if out2 is None:
                raise
            out, had_invalid_cast = out2, False
        retried_fp32 = False
        images = getattr(out, "images", None)
        if not isinstance(images, list) or not images:
            raise ValueError("Diffusers pipeline returned no images")
        if self._is_probably_all_black_image(images[0]):
            out2 = self._maybe_retry_fp32_on_invalid_output(
                kind="t2i",
                pipe=pipe,
                kwargs=kwargs,
                progress_callback=progress_callback,
                total_steps=total_steps,
            )
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
                    else "Try setting torch_dtype=float32. "
                )
                + "Try increasing steps, adjusting guidance_scale, or use the stable-diffusion.cpp backend."
            )
        png = self._png_bytes(images[0])
        meta = {"source": "diffusers", "model_id": self._cfg.model_id}
        if rapid_repo:
            meta["rapid_aio_repo"] = rapid_repo
        if lora_sig:
            meta["lora_signature"] = lora_sig
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
        return self.edit_image_with_progress(request, progress_callback=None)

    def edit_image_with_progress(
        self,
        request: ImageEditRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        if request.mask is not None:
            pipe = self._get_or_load_pipeline("inpaint")
            call_params = self._call_params.get("inpaint")
            kind = "inpaint"
        else:
            pipe = self._get_or_load_pipeline("i2i")
            call_params = self._call_params.get("i2i")
            kind = "i2i"

        total_steps = int(request.steps) if request.steps is not None else None

        torch_dtype = getattr(pipe, "dtype", None)
        if torch_dtype is None:
            torch = _lazy_import_torch()
            device = self._effective_device(torch)
            torch_dtype = _torch_dtype_from_str(torch, self._cfg.torch_dtype) or _default_torch_dtype_for_device(torch, device)
        rapid_repo = self._maybe_apply_rapid_aio_transformer(pipe=pipe, extra=request.extra, torch_dtype=torch_dtype)
        lora_sig = self._apply_loras(kind=kind, pipe=pipe, extra=request.extra)

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

        try:
            call_kwargs = dict(kwargs)
            if progress_callback is not None:
                call_kwargs["__abstractvision_progress_callback"] = progress_callback
                call_kwargs["__abstractvision_progress_total_steps"] = total_steps
            out, had_invalid_cast = self._pipe_call(pipe, call_kwargs)
        except Exception as e:
            out2 = self._maybe_retry_on_dtype_mismatch(
                kind=kind,
                pipe=pipe,
                kwargs=kwargs,
                error=e,
                progress_callback=progress_callback,
                total_steps=total_steps,
            )
            if out2 is None:
                raise
            out, had_invalid_cast = out2, False
        retried_fp32 = False
        images = getattr(out, "images", None)
        if not isinstance(images, list) or not images:
            raise ValueError("Diffusers pipeline returned no images")
        if self._is_probably_all_black_image(images[0]):
            kind = "inpaint" if request.mask is not None else "i2i"
            out2 = self._maybe_retry_fp32_on_invalid_output(
                kind=kind,
                pipe=pipe,
                kwargs=kwargs,
                progress_callback=progress_callback,
                total_steps=total_steps,
            )
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
        if rapid_repo:
            meta["rapid_aio_repo"] = rapid_repo
        if lora_sig:
            meta["lora_signature"] = lora_sig
        if retried_fp32:
            meta["retried_fp32"] = True
        if had_invalid_cast:
            meta["had_invalid_cast_warning"] = True
        try:
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
