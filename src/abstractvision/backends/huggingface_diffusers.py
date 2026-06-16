from __future__ import annotations

import io
import inspect
import os
import shutil
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..errors import CapabilityNotSupportedError, OptionalDependencyMissingError
from ..lora_adapters import (
    lora_adapter_signature,
    recommended_lora_request_overrides,
    resolve_request_lora_adapters,
    resolved_adapter_name,
    serialize_lora_adapters,
)
from ..model_capabilities import VisionModelCapabilitiesRegistry, VisionTaskSpec
from ..model_cache import (
    cached_hf_model_sources,
    hf_snapshot_has_incomplete_downloads,
    hf_snapshot_has_weight_files,
    hf_snapshot_is_usable,
    hf_snapshot_missing_indexed_weight_files,
    resolve_hf_repo_snapshot,
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


def _require_optional_dep(name: str, install_hint: str) -> None:
    import sys

    raise OptionalDependencyMissingError(
        f"Optional dependency missing: {name}. Install via: {install_hint} "
        f"(python={sys.executable})"
    )


_DIFFUSERS_RUNTIME_HINT = 'pip install "abstractvision[diffusers]"'
_LOCAL_RUNTIME_HINT = 'pip install "abstractvision[local]"'
_DIFFUSERS_CACHE_REQUIRED_FILES = ("model_index.json",)
_GLM_IMAGE_FALLBACK_CHAT_TEMPLATE = (
    "{%- for m in messages -%}\n"
    "{%- if m.content is string -%}\n"
    "{{ m.content }}\n"
    "{%- else -%}\n"
    "{%- for item in m.content -%}\n"
    "{%- if item.type == 'image' or item.get('image') is not none -%}\n"
    "<|dit_token_16384|><|image|><|dit_token_16385|>\n"
    "{%- elif item.type == 'text' -%}\n"
    "{{ item.text }}\n"
    "{%- endif -%}\n"
    "{%- endfor -%}\n"
    "{%- endif -%}\n"
    "{%- endfor -%}\n"
)
_ERNIE_PE_FALLBACK_CHAT_TEMPLATE = (
    "{{- bos_token }}[SYSTEM_PROMPT]你是一个专业的文生图 Prompt 增强助手。你将收到用户的简短图片描述及目标生成分辨率，请据此扩写为一段内容丰富、细节充分的视觉描述，以帮助文生图模型生成高质量的图片。仅输出增强后的描述，不要包含任何解释或前缀。[/SYSTEM_PROMPT]\n"
    "{%- for message in messages %}\n"
    "{%- if message['role'] == 'user' %}\n"
    "{{- '[INST]' + message['content'] + '[/INST]' }}\n"
    "{%- elif message['role'] == 'assistant' %}\n"
    "{%- generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}\n"
    "{%- endif %}\n"
    "{%- endfor %}\n"
)
_GENERIC_CHAT_TEMPLATE_WITH_ASSISTANT_PROMPT = (
    "{%- for message in messages -%}\n"
    "{%- if message.content is string -%}\n"
    "{%- set content = message.content -%}\n"
    "{%- else -%}\n"
    "{%- set content = '' -%}\n"
    "{%- endif -%}\n"
    "{%- if message.role in ['user', 'system'] -%}\n"
    "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n"
    "{%- elif message.role == 'assistant' -%}\n"
    "{{- '<|im_start|>assistant\\n' + content + '<|im_end|>\\n' }}\n"
    "{%- endif -%}\n"
    "{%- endfor -%}\n"
    "{%- if add_generation_prompt -%}\n"
    "{{- '<|im_start|>assistant\\n' }}\n"
    "{%- endif -%}\n"
)
_SUPPORTED_LOCAL_DIFFUSERS_TEXT_TO_VIDEO_MODELS = {
    "zai-org/cogvideox-2b",
    "thudm/cogvideox-2b",
}
_TEMPORARILY_DISABLED_LOCAL_DIFFUSERS_TASKS: Dict[str, set[str]] = {
    "zai-org/glm-image": {"text_to_image", "image_to_image"},
    "zai-org/cogvideox-2b": {"text_to_video"},
    "thudm/cogvideox-2b": {"text_to_video"},
}
_DEFAULT_REQUIRED_IMAGE_DIMENSION = 512


def _lazy_import_diffusers():
    try:
        import warnings

        # Some Diffusers modules decorate functions with `torch.autocast(device_type="cuda", ...)`,
        # which emits noisy warnings on non-CUDA machines (including Apple Silicon / MPS).
        warnings.filterwarnings(
            "ignore", message=r".*CUDA is not available.*Disabling autocast.*", category=UserWarning
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*device_type of 'cuda'.*CUDA is not available.*",
            category=UserWarning,
        )
        import diffusers  # type: ignore
        from diffusers import DiffusionPipeline  # type: ignore
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyMissingError(
            "Diffusers backend requested but the local Diffusers runtime is missing or failed to import. "
            f"Install via: {_DIFFUSERS_RUNTIME_HINT} (or {_LOCAL_RUNTIME_HINT}). "
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
        _require_optional_dep(
            "torch",
            f"{_DIFFUSERS_RUNTIME_HINT} (or {_LOCAL_RUNTIME_HINT}); install a CUDA-enabled PyTorch wheel first if you need NVIDIA GPU support",
        )
    return torch


def _lazy_import_pil():
    try:
        from PIL import Image  # type: ignore
    except Exception:  # pragma: no cover
        _require_optional_dep("pillow", f"{_DIFFUSERS_RUNTIME_HINT} (or {_LOCAL_RUNTIME_HINT})")
    return Image


def _lazy_import_qwen_image_transformer_2d_model():
    try:
        from diffusers.models import QwenImageTransformer2DModel  # type: ignore
    except Exception:
        raise ValueError(
            "Rapid-AIO transformer override requires diffusers.models.QwenImageTransformer2DModel, "
            "which is not available in this diffusers build."
        )
    return QwenImageTransformer2DModel


def _lazy_import_gguf_quantization_config():
    try:
        from diffusers import GGUFQuantizationConfig  # type: ignore
    except Exception:
        raise ValueError(
            "Diffusers GGUF loading requires diffusers.GGUFQuantizationConfig, "
            "which is not available in this diffusers build."
        )
    return GGUFQuantizationConfig


def _read_chat_template_file(snapshot_dir: Optional[Path], *relative_paths: str) -> Optional[str]:
    if snapshot_dir is None:
        return None
    for rel in relative_paths:
        candidate = snapshot_dir / rel
        try:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8").strip()
                if text:
                    return text
        except Exception:
            continue
    return None


def _maybe_set_chat_template(target: Any, template: Optional[str]) -> bool:
    if target is None or not str(template or "").strip():
        return False
    try:
        if getattr(target, "chat_template", None):
            return True
    except Exception:
        pass
    try:
        setattr(target, "chat_template", str(template))
        return True
    except Exception:
        return False


def _ensure_pipeline_chat_templates(
    pipe: Any,
    *,
    snapshot_dir: Optional[Path],
    model_id: Optional[str],
) -> None:
    model_hint = (
        " ".join(
            str(value or "")
            for value in (
                model_id,
                type(pipe).__name__,
                getattr(getattr(pipe, "tokenizer", None), "__class__", type(None)).__name__,
                getattr(getattr(pipe, "processor", None), "__class__", type(None)).__name__,
            )
        )
        .strip()
        .lower()
    )

    tokenizer = getattr(pipe, "tokenizer", None)
    processor = getattr(pipe, "processor", None)
    pe_tokenizer = getattr(pipe, "pe_tokenizer", None)

    tokenizer_targets = [
        (tokenizer, ("tokenizer/chat_template.jinja",), None),
        (
            pe_tokenizer,
            ("pe_tokenizer/chat_template.jinja", "pe/chat_template.jinja"),
            _ERNIE_PE_FALLBACK_CHAT_TEMPLATE if "ernie" in model_hint else None,
        ),
    ]
    if processor is not None:
        tokenizer_targets.append(
            (
                getattr(processor, "tokenizer", None),
                (
                    "processor/tokenizer/chat_template.jinja",
                    "tokenizer/chat_template.jinja",
                ),
                None,
            )
        )

    seen_tokenizers = set()
    for target, relative_paths, explicit_fallback in tokenizer_targets:
        if target is None or id(target) in seen_tokenizers:
            continue
        seen_tokenizers.add(id(target))
        if getattr(target, "chat_template", None):
            continue
        tokenizer_template = _read_chat_template_file(snapshot_dir, *relative_paths)
        if tokenizer_template is None:
            if explicit_fallback is not None:
                tokenizer_template = explicit_fallback
            elif any(token in model_hint for token in ("z-image", "flux2", "qwen")):
                tokenizer_template = _GENERIC_CHAT_TEMPLATE_WITH_ASSISTANT_PROMPT
        _maybe_set_chat_template(target, tokenizer_template)

    if processor is not None and not getattr(processor, "chat_template", None):
        processor_template = _read_chat_template_file(snapshot_dir, "processor/chat_template.jinja")
        if processor_template is None and "glm-image" in model_hint:
            processor_template = _GLM_IMAGE_FALLBACK_CHAT_TEMPLATE
        _maybe_set_chat_template(processor, processor_template)


_TRANSFORMERS_CLIP_POSITION_IDS_PATCHED = False
_TRANSFORMERS_GLM_IMAGE_MPS_GRID_SAMPLE_PATCHED = False


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


def _glm_image_position_embedding_with_cpu_grid_sample(
    self: Any,
    embeddings: Any,
    lengths: Any,
    image_shapes: Any,
    h_coords: Any,
    w_coords: Any,
    *,
    torch: Any,
    F: Any,
) -> Any:
    pos_embed_weight = self.position_embedding.weight
    hidden_size = pos_embed_weight.shape[1]
    device = pos_embed_weight.device

    if isinstance(lengths, list):
        lengths = torch.tensor(lengths, device=device, dtype=torch.long)

    orig_size_sq = pos_embed_weight.shape[0]
    orig_size = int(orig_size_sq**0.5)
    pos_embed_2d = (
        pos_embed_weight.view(orig_size, orig_size, hidden_size)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )

    target_h = torch.cat([image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]).to(
        device=device, dtype=torch.float32
    )
    target_w = torch.cat([image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]).to(
        device=device, dtype=torch.float32
    )

    norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
    norm_h = ((h_coords + 0.5) / target_h) * 2 - 1
    grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

    interpolated_embed_fp32 = F.grid_sample(
        pos_embed_2d.detach().to("cpu", dtype=torch.float32),
        grid.detach().to("cpu", dtype=torch.float32),
        mode=self.interpolated_method,
        align_corners=False,
        padding_mode="border",
    ).to(device=embeddings.device, dtype=torch.float32)

    adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(-1).permute(1, 0)
    adapted_pos_embed = adapted_pos_embed_fp32.to(pos_embed_weight.dtype).to(embeddings.device)
    return embeddings + adapted_pos_embed


def _maybe_patch_transformers_glm_image_mps_grid_sample() -> None:
    global _TRANSFORMERS_GLM_IMAGE_MPS_GRID_SAMPLE_PATCHED
    if _TRANSFORMERS_GLM_IMAGE_MPS_GRID_SAMPLE_PATCHED:
        return

    try:
        import torch as _torch  # type: ignore
        import torch.nn.functional as _F  # type: ignore
        from transformers.models.glm_image.modeling_glm_image import GlmImageVisionEmbeddings  # type: ignore
    except Exception:
        return

    if bool(getattr(GlmImageVisionEmbeddings, "_abstractvision_mps_grid_sample_patched", False)):
        _TRANSFORMERS_GLM_IMAGE_MPS_GRID_SAMPLE_PATCHED = True
        return

    orig_forward = getattr(GlmImageVisionEmbeddings, "forward", None)
    if not callable(orig_forward):
        return

    def _patched_forward(
        self: Any, embeddings: Any, lengths: Any, image_shapes: Any, h_coords: Any, w_coords: Any
    ) -> Any:
        try:
            return orig_forward(self, embeddings, lengths, image_shapes, h_coords, w_coords)
        except RuntimeError as e:
            msg = str(e or "")
            position_embedding = getattr(self, "position_embedding", None)
            weight = getattr(position_embedding, "weight", None)
            device = str(getattr(weight, "device", "") or "").lower()
            if "Unsupported Border padding mode" not in msg or not (
                device == "mps" or device.startswith("mps:")
            ):
                raise
            return _glm_image_position_embedding_with_cpu_grid_sample(
                self,
                embeddings,
                lengths,
                image_shapes,
                h_coords,
                w_coords,
                torch=_torch,
                F=_F,
            )

    setattr(GlmImageVisionEmbeddings, "forward", _patched_forward)
    setattr(GlmImageVisionEmbeddings, "_abstractvision_mps_grid_sample_patched", True)
    _TRANSFORMERS_GLM_IMAGE_MPS_GRID_SAMPLE_PATCHED = True


@contextmanager
def _hf_offline_env(enabled: bool):
    """Control Hugging Face offline mode within a scope.

    When `enabled=True`, we force offline mode (no network calls).
    When `enabled=False`, we permit online mode for explicitly enabled downloads.
    """

    # These are respected by huggingface_hub / transformers / diffusers.
    # We scope them to the load/call to avoid surprising other parts of the process.
    vars_to_set = {
        "HF_HUB_OFFLINE": "1" if enabled else "0",
        "TRANSFORMERS_OFFLINE": "1" if enabled else "0",
        "DIFFUSERS_OFFLINE": "1" if enabled else "0",
        # Avoid telemetry in both offline and explicitly-online scopes.
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    if enabled:
        # In cache-only mode, avoid implicit token lookup/use as well as network calls.
        vars_to_set["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
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


def _is_mps_device(device: Any) -> bool:
    d = str(device or "").strip().lower()
    return d == "mps" or d.startswith("mps:")


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


_INTERNAL_EXTRA_KEYS = {
    # LoRA plumbing: parsed and applied by AbstractVision, should not reach Diffusers pipelines.
    "loras",
    "loras_json",
    "lora",
    "lora_json",
    # Rapid-AIO transformer override: parsed and applied by AbstractVision.
    "rapid_aio_repo",
    "rapid_aio_subfolder",
    "rapid_aio",
}


def _forward_extra_kwargs(extra: Any, *, call_params: Optional[set[str]]) -> Dict[str, Any]:
    """Filter `request.extra` before forwarding to a Diffusers pipeline.

    The REPL forwards unknown `--flags` via `request.extra`. Many Diffusers pipelines have a strict
    `__call__` signature (no `**kwargs`), so passing unknown keys raises:
      TypeError: <Pipeline>.__call__() got an unexpected keyword argument ...

    Strategy:
    - Always drop AbstractVision-internal keys (LoRA / Rapid-AIO controls).
    - If we have a concrete parameter set for the pipeline call (no `**kwargs`), drop unknown keys.
    - If the pipeline supports `**kwargs` (call_params is None), keep keys (best-effort).
    """

    if not isinstance(extra, dict) or not extra:
        return {}

    out: Dict[str, Any] = {}
    for k, v in extra.items():
        if k is None or v is None:
            continue
        key = str(k).strip()
        if not key or key in _INTERNAL_EXTRA_KEYS or key.startswith("__abstractvision_"):
            continue
        out[key] = v

    if call_params is not None:
        out = {k: v for k, v in out.items() if k in call_params}
    return out


def _round_up_to_multiple(value: Optional[int], multiple_of: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        step = int(multiple_of)
    except Exception:
        return int(value)
    if step <= 1:
        return int(value)
    current = int(value)
    remainder = current % step
    if remainder == 0:
        return current
    return current + (step - remainder)


def _move_pipe_to_device(pipe: Any, *, device: str, dtype: Any = None) -> Any:
    last_error: Optional[Exception] = None
    if dtype is not None:
        try:
            return pipe.to(device=str(device), dtype=dtype)
        except Exception as exc:
            last_error = exc
        try:
            pipe = pipe.to(dtype=dtype)
            return pipe.to(str(device))
        except Exception as exc:
            last_error = exc
    try:
        return pipe.to(str(device))
    except Exception as exc:
        last_error = exc
    if last_error is not None:
        raise last_error
    return pipe


def _require_ffmpeg_binary() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    raise ValueError(
        "Local video export requires an `ffmpeg` executable on PATH. "
        "Install ffmpeg and retry local text_to_video generation."
    )


def _frames_to_mp4_bytes(frames: List[Any], *, fps: int) -> bytes:
    if not frames:
        raise ValueError("Cannot encode a video with no frames.")
    ffmpeg_bin = _require_ffmpeg_binary()
    with tempfile.TemporaryDirectory(prefix="abstractvision-video-") as td:
        root = Path(td)
        for idx, frame in enumerate(frames):
            frame_path = root / f"frame_{idx:05d}.png"
            frame.save(frame_path, format="PNG")
        out_path = root / "output.mp4"
        common = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(max(1, int(fps))),
            "-i",
            str(root / "frame_%05d.png"),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        for codec in ("libx264", "mpeg4"):
            cmd = [*common, "-c:v", codec, str(out_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and out_path.is_file():
                return out_path.read_bytes()
        raise ValueError(
            "ffmpeg could not encode the generated frames into MP4. "
            + (proc.stderr.strip() if proc.stderr else "No ffmpeg stderr output was captured.")
        )


def _maybe_enable_video_pipeline_memory_savers(pipe: Any) -> None:
    for attr in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        fn = getattr(pipe, attr, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    vae = getattr(pipe, "vae", None)
    for attr in ("enable_slicing", "enable_tiling"):
        fn = getattr(vae, attr, None) if vae is not None else None
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _maybe_upcast_vae_for_mps(
    torch: Any, pipe: Any, device: str, *, allow_fp32_vae: bool = True
) -> None:
    d = str(device or "").strip().lower()
    if d != "mps" and not d.startswith("mps:"):
        return
    if not allow_fp32_vae:
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


def _is_quantized_module(module: Any) -> bool:
    if module is None:
        return False
    return bool(
        getattr(module, "is_quantized", False)
        or getattr(module, "hf_quantizer", None) is not None
        or getattr(module, "quantization_method", None) is not None
    )


def _pipe_has_quantized_components(pipe: Any) -> bool:
    seen: set[int] = set()

    def check(module: Any) -> bool:
        if module is None:
            return False
        ident = id(module)
        if ident in seen:
            return False
        seen.add(ident)
        return _is_quantized_module(module)

    comps = getattr(pipe, "components", None)
    if isinstance(comps, dict):
        for v in comps.values():
            if check(v):
                return True

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
        if check(getattr(pipe, attr, None)):
            return True
    return False


def _maybe_cast_pipe_modules_to_dtype(pipe: Any, *, dtype: Any) -> None:
    if dtype is None:
        return

    def _to(module: Any) -> None:
        if module is None:
            return
        if _is_quantized_module(module):
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
                    if (
                        dtype is not None
                        and x_dtype is not None
                        and x_dtype != dtype
                        and callable(to_fn)
                    ):
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
class _DiffusersGGUFTransformerSpec:
    base_model_id: str
    gguf_repo_id: Optional[str]
    gguf_patterns: Tuple[str, ...]
    gguf_path: Optional[Path] = None


@dataclass(frozen=True)
class HuggingFaceDiffusersBackendConfig:
    """Config for a local Diffusers backend.

    Notes:
    - Downloads are disabled by default; the runtime is cache-only/offline unless explicitly configured.
    - Pre-download model artifacts separately, or set `allow_download=True` when you want Diffusers to fetch them.
    """

    model_id: str
    device: str = "cpu"  # "cpu" | "cuda" | "mps" | "auto" | ...
    torch_dtype: Optional[str] = None  # "float16" | "bfloat16" | "float32" | None
    allow_download: bool = False
    auto_retry_fp32: bool = True
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    use_safetensors: bool = True
    low_cpu_mem_usage: bool = True


class HuggingFaceDiffusersVisionBackend(VisionBackend):
    """Local generative vision backend using HuggingFace Diffusers."""

    def __init__(self, *, config: HuggingFaceDiffusersBackendConfig):
        self._cfg = config
        self._backend_lock = threading.RLock()
        self._pipelines: Dict[str, Any] = {}
        self._call_params: Dict[str, Optional[set[str]]] = {}
        self._warmed_pipeline_ids: Dict[str, int] = {}
        self._fused_lora_signature: Dict[str, Optional[str]] = {}
        self._rapid_transformer_key: Optional[str] = None
        self._rapid_transformer: Any = None
        self._resolved_device: Optional[str] = None
        self._capability_registry: Optional[VisionModelCapabilitiesRegistry] = None
        self._capability_registry_failed = False

    def _effective_device(self, torch: Any) -> str:
        if self._resolved_device is not None:
            return self._resolved_device

        raw = str(getattr(self._cfg, "device", "") or "").strip()
        d = raw.lower()
        if not d or d in {"auto", "default"}:
            cuda = getattr(torch, "cuda", None)
            if (
                cuda is not None
                and callable(getattr(cuda, "is_available", None))
                and cuda.is_available()
            ):
                self._resolved_device = "cuda"
                return self._resolved_device

            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None) if backends is not None else None
            if (
                mps is not None
                and callable(getattr(mps, "is_available", None))
                and mps.is_available()
            ):
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
        with self._backend_lock:
            kind = self._preload_pipeline_kind()
            pipe = self._get_or_load_pipeline(kind)
            if self._is_pipeline_warm(kind, pipe):
                return
            if kind == "t2v":
                self.generate_video(
                    self.normalize_video_generation_request(self._warmup_video_generation_request())
                )
            elif kind == "t2i":
                self.generate_image(self._warmup_generation_request())
            elif kind == "i2i":
                self.edit_image(self.normalize_image_edit_request(self._warmup_edit_request()))
            else:
                self._mark_pipeline_warm(kind, pipe)

    def unload(self) -> None:
        with self._backend_lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        # Best-effort: release pipelines and GPU cache.
        pipes = list(self._pipelines.values())
        self._pipelines.clear()
        self._call_params.clear()
        self._warmed_pipeline_ids.clear()
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
            if (
                cuda is not None
                and callable(getattr(cuda, "is_available", None))
                and cuda.is_available()
            ):
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

    def _registry(self) -> Optional[VisionModelCapabilitiesRegistry]:
        if self._capability_registry is not None:
            return self._capability_registry
        if self._capability_registry_failed:
            return None
        try:
            self._capability_registry = VisionModelCapabilitiesRegistry()
        except Exception:
            self._capability_registry_failed = True
            return None
        return self._capability_registry

    def _task_spec(self, task: str) -> Optional[VisionTaskSpec]:
        reg = self._registry()
        model_id = self._canonical_model_id()
        if reg is None or not model_id:
            return None
        try:
            return reg.get(model_id).tasks.get(str(task))
        except Exception:
            return None

    def _canonical_model_id(self, model_id: Optional[str] = None) -> str:
        raw = str(model_id if model_id is not None else self._cfg.model_id or "").strip()
        if not raw:
            return ""
        if model_id is None:
            gguf_spec = self._diffusers_gguf_transformer_spec()
            if gguf_spec is not None:
                raw = gguf_spec.base_model_id
        reg = self._registry()
        if reg is not None:
            try:
                return str(reg.get(raw).model_id or raw).strip()
            except Exception:
                return raw
        return raw

    def _diffusers_gguf_transformer_spec(self) -> Optional[_DiffusersGGUFTransformerSpec]:
        raw = str(self._cfg.model_id or "").strip()
        if not raw:
            return None

        raw_path = Path(raw).expanduser()
        if raw_path.is_file() and raw_path.suffix.lower() == ".gguf":
            base = self._base_model_id_for_gguf_name(raw_path.name)
            if base is None:
                return None
            return _DiffusersGGUFTransformerSpec(
                base_model_id=base,
                gguf_repo_id=None,
                gguf_patterns=(raw_path.name,),
                gguf_path=raw_path,
            )
        if raw_path.is_dir():
            gguf_path = self._find_first_file(raw_path, ("*.gguf",))
            if gguf_path is not None:
                base = self._base_model_id_for_gguf_name(gguf_path.name)
                if base is not None:
                    return _DiffusersGGUFTransformerSpec(
                        base_model_id=base,
                        gguf_repo_id=None,
                        gguf_patterns=(gguf_path.name,),
                        gguf_path=gguf_path,
                    )

        try:
            from ..model_downloads import find_model_preset

            preset = find_model_preset(raw, target="gguf", engine="diffusers", require_8bit=True)
            patterns = tuple(
                str(pattern)
                for pattern in preset.allow_patterns
                if ".gguf" in str(pattern).strip().lower()
            )
            return _DiffusersGGUFTransformerSpec(
                base_model_id=str(
                    preset.upstream_repo_id or self._canonical_model_id(preset.repo_id)
                ).strip(),
                gguf_repo_id=str(preset.repo_id).strip(),
                gguf_patterns=patterns or ("*.gguf",),
            )
        except Exception:
            pass

        lowered = raw.lower()
        if lowered in {
            "qwen-image-edit-2511-gguf",
            "qwen-image-edit-2511-gguf-q8_0",
            "unsloth/qwen-image-edit-2511-gguf",
        }:
            return _DiffusersGGUFTransformerSpec(
                base_model_id="Qwen/Qwen-Image-Edit-2511",
                gguf_repo_id="unsloth/Qwen-Image-Edit-2511-GGUF",
                gguf_patterns=("qwen-image-edit-2511-Q8_0.gguf",),
            )
        return None

    def _base_model_id_for_gguf_name(self, name: str) -> Optional[str]:
        normalized = str(name or "").strip().lower().replace("_", "-")
        if "qwen-image-edit-2511" in normalized:
            return "Qwen/Qwen-Image-Edit-2511"
        return None

    def _find_first_file(self, root: Path, patterns: Sequence[str]) -> Optional[Path]:
        for pattern in patterns:
            try:
                matches = sorted(path for path in root.glob(str(pattern)) if path.is_file())
            except Exception:
                matches = []
            if matches:
                return matches[0]
        return None

    def _resolve_gguf_transformer_path(self, spec: _DiffusersGGUFTransformerSpec) -> Path:
        if spec.gguf_path is not None:
            path = Path(spec.gguf_path).expanduser()
            if path.is_file():
                return path

        repo_id = str(spec.gguf_repo_id or "").strip()
        if repo_id:
            snapshot = resolve_hf_repo_snapshot(
                repo_id,
                revision=str(self._cfg.revision or "main").strip() or "main",
                extra_roots=self._hf_cache_roots(),
                require_weight_files=True,
            )
            if snapshot is not None:
                found = self._find_first_file(snapshot, spec.gguf_patterns)
                if found is not None:
                    return found

            if bool(self._cfg.allow_download):
                concrete_patterns = [
                    p
                    for p in spec.gguf_patterns
                    if p.lower().endswith(".gguf")
                    and "*" not in p
                    and "?" not in p
                    and "[" not in p
                ]
                if concrete_patterns:
                    try:
                        from huggingface_hub import hf_hub_download

                        return Path(
                            hf_hub_download(
                                repo_id=repo_id,
                                filename=concrete_patterns[0],
                                repo_type="model",
                                revision=str(self._cfg.revision or "main").strip() or None,
                                cache_dir=str(self._cfg.cache_dir) if self._cfg.cache_dir else None,
                                local_files_only=False,
                            )
                        )
                    except Exception:
                        pass

        raise ValueError(
            "Diffusers GGUF transformer artifact is not available locally. "
            f"Expected {spec.gguf_patterns!r}"
            + (f" in {repo_id!r}. " if repo_id else ". ")
            + "Download the GGUF artifact first with `abstractvision download qwen-image-edit-2511-gguf --provider sdcpp`."
        )

    def _load_gguf_transformer(
        self,
        *,
        spec: _DiffusersGGUFTransformerSpec,
        gguf_path: Path,
        torch: Any,
        torch_dtype: Any,
        device: str,
    ) -> Any:
        GGUFQuantizationConfig = _lazy_import_gguf_quantization_config()
        QwenImageTransformer2DModel = _lazy_import_qwen_image_transformer_2d_model()
        q_config = GGUFQuantizationConfig(
            compute_dtype=torch_dtype or getattr(torch, "bfloat16", None)
        )

        kwargs: Dict[str, Any] = {
            "config": spec.base_model_id,
            "subfolder": "transformer",
            "quantization_config": q_config,
            "torch_dtype": torch_dtype,
            "local_files_only": not bool(self._cfg.allow_download),
        }
        if self._cfg.cache_dir:
            kwargs["cache_dir"] = str(self._cfg.cache_dir)
        if self._cfg.revision:
            kwargs["revision"] = str(self._cfg.revision)
            kwargs["config_revision"] = str(self._cfg.revision)

        with _hf_offline_env(not bool(self._cfg.allow_download)):
            transformer = QwenImageTransformer2DModel.from_single_file(str(gguf_path), **kwargs)

        try:
            return transformer.to(str(device))
        except Exception:
            return transformer

    def _needs_list_wrapped_i2i_image(self, pipe: Any = None) -> bool:
        model_id = self._canonical_model_id().lower()
        if model_id == "zai-org/glm-image":
            return True
        pipe_name = type(pipe).__name__ if pipe is not None else ""
        return pipe_name == "GlmImagePipeline"

    def _preferred_torch_dtype_for_kind(
        self,
        kind: str,
        device: Any,
        torch: Any,
        torch_dtype: Any,
    ) -> Any:
        if kind not in {"i2i", "inpaint"}:
            return torch_dtype
        model_id = self._canonical_model_id().lower()
        device_name = str(device or "").strip().lower()
        if not (device_name == "mps" or device_name.startswith("mps:")):
            return torch_dtype
        gguf_spec = self._diffusers_gguf_transformer_spec()
        model_hint = (
            (gguf_spec.base_model_id if gguf_spec is not None else model_id)
            .replace("_", "-")
            .lower()
        )
        wants_bf16 = model_hint == "zai-org/glm-image" or "qwen-image-edit" in model_hint
        if not wants_bf16:
            return torch_dtype
        # Only override when dtype selection is automatic (no explicit torch_dtype was requested).
        explicit = str(getattr(self._cfg, "torch_dtype", "") or "").strip().lower()
        if explicit and explicit not in {"auto", "default"}:
            return torch_dtype
        bf16 = getattr(torch, "bfloat16", None)
        fp16 = getattr(torch, "float16", None)
        if bf16 is None:
            return torch_dtype
        if torch_dtype is None or torch_dtype == fp16:
            return bf16
        return torch_dtype

    def _is_temporarily_disabled_task(self, task: str, *, model_id: Optional[str] = None) -> bool:
        canonical = self._canonical_model_id(model_id).lower()
        disabled = _TEMPORARILY_DISABLED_LOCAL_DIFFUSERS_TASKS.get(canonical)
        if not disabled:
            return False
        return str(task) in disabled

    def _supports_local_text_to_video(self, model_id: Optional[str] = None) -> bool:
        resolved_model_id = str(
            model_id if model_id is not None else self._cfg.model_id or ""
        ).strip()
        if not resolved_model_id:
            return False
        if self._is_temporarily_disabled_task("text_to_video", model_id=resolved_model_id):
            return False
        reg = self._registry()
        if reg is not None:
            try:
                if "text_to_video" not in reg.get(resolved_model_id).tasks:
                    return False
            except Exception:
                return False
        return (
            self._canonical_model_id(resolved_model_id).lower()
            in _SUPPORTED_LOCAL_DIFFUSERS_TEXT_TO_VIDEO_MODELS
        )

    def _supports_local_image_to_video(self, model_id: Optional[str] = None) -> bool:
        _ = model_id
        return False

    def _supports_backend_task(self, task: str, *, model_id: Optional[str] = None) -> bool:
        if self._is_temporarily_disabled_task(task, model_id=model_id):
            return False
        if task == "text_to_video":
            return self._supports_local_text_to_video(model_id=model_id)
        if task == "image_to_video":
            return self._supports_local_image_to_video(model_id=model_id)
        if task == "multi_view_image":
            return False
        return True

    def _supported_task_names(self, model_id: Optional[str] = None) -> List[str]:
        model_spec = None
        reg = self._registry()
        resolved_model_id = (
            str(model_id).strip() if model_id is not None else self._canonical_model_id()
        )
        if reg is not None and resolved_model_id:
            try:
                model_spec = reg.get(resolved_model_id)
            except Exception:
                model_spec = None
        if model_spec is None:
            return ["image_to_image", "text_to_image"]
        return sorted(
            str(task_name)
            for task_name in model_spec.tasks.keys()
            if self._registry_task_supports_diffusers(model_spec.tasks.get(str(task_name)))
            and self._supports_backend_task(str(task_name), model_id=resolved_model_id)
        )

    def _registry_task_supports_diffusers(self, task_spec: Any) -> bool:
        requires = getattr(task_spec, "requires", None)
        if not isinstance(requires, dict):
            return True
        required_backend = str(requires.get("backend") or "").strip().lower().replace("_", "-")
        if not required_backend:
            return True
        return required_backend in {"diffusers", "hf-diffusers", "huggingface-diffusers"}

    def _preload_pipeline_kind(self) -> str:
        tasks = set(self._supported_task_names())
        if not tasks:
            raise CapabilityNotSupportedError(
                "This model is temporarily disabled in the local Diffusers backend pending runtime-quality investigation."
            )
        if "text_to_image" in tasks:
            return "t2i"
        if "text_to_video" in tasks:
            return "t2v"
        if "image_to_image" in tasks:
            return "i2i"
        return "t2i"

    def _warmup_generation_request(self) -> ImageGenerationRequest:
        return ImageGenerationRequest(
            prompt="abstractvision preload warmup",
            steps=1,
            seed=0,
        )

    def _warmup_video_generation_request(self) -> VideoGenerationRequest:
        return VideoGenerationRequest(
            prompt="abstractvision preload warmup",
            steps=1,
            num_frames=9,
            fps=8,
            seed=0,
        )

    def _warmup_edit_request(self) -> ImageEditRequest:
        Image = _lazy_import_pil()
        img = Image.new("RGB", (256, 256), color=(180, 180, 180))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return ImageEditRequest(
            prompt="abstractvision preload warmup",
            image=buf.getvalue(),
            # Some edit pipelines can return NaNs/all-black images at extremely low step counts.
            # Keep this small, but not minimal, so warmup is reliable.
            steps=2,
            seed=0,
            guidance_scale=1.0,
        )

    def _normalize_int_param(
        self,
        value: Optional[int],
        spec: Optional[Dict[str, Any]],
    ) -> Optional[int]:
        if not isinstance(spec, dict):
            return value
        out = int(value) if value is not None else None
        if out is None and spec.get("default") is not None:
            try:
                out = int(spec.get("default"))
            except Exception:
                out = None
        if spec.get("const") is not None:
            try:
                out = int(spec.get("const"))
            except Exception:
                pass
        if out is not None and spec.get("min") is not None:
            try:
                out = max(out, int(spec.get("min")))
            except Exception:
                pass
        if out is not None and spec.get("multiple_of") is not None:
            out = _round_up_to_multiple(out, spec.get("multiple_of"))
        return out

    def _normalize_float_param(
        self,
        value: Optional[float],
        spec: Optional[Dict[str, Any]],
    ) -> Optional[float]:
        if not isinstance(spec, dict):
            return value
        out = float(value) if value is not None else None
        if out is None and spec.get("default") is not None:
            try:
                out = float(spec.get("default"))
            except Exception:
                out = None
        if spec.get("const") is not None:
            try:
                out = float(spec.get("const"))
            except Exception:
                pass
        if out is not None and spec.get("min") is not None:
            try:
                out = max(out, float(spec.get("min")))
            except Exception:
                pass
        return out

    def _normalize_required_image_dimensions(
        self,
        *,
        width_value: Optional[int],
        height_value: Optional[int],
        params: Dict[str, Any],
        input_image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        width_spec = params.get("width")
        height_spec = params.get("height")

        width = self._normalize_int_param(width_value, width_spec)
        height = self._normalize_int_param(height_value, height_spec)

        width_required = isinstance(width_spec, dict) and bool(width_spec.get("required"))
        height_required = isinstance(height_spec, dict) and bool(height_spec.get("required"))
        width_auto = isinstance(width_spec, dict) and bool(
            width_spec.get("auto_derived_from_input")
        )
        height_auto = isinstance(height_spec, dict) and bool(
            height_spec.get("auto_derived_from_input")
        )

        if input_image_size is not None:
            image_width, image_height = input_image_size
            if width is None and (width_required or width_auto):
                width = self._normalize_int_param(int(image_width), width_spec)
            if height is None and (height_required or height_auto):
                height = self._normalize_int_param(int(image_height), height_spec)

        if width is None and width_required:
            width = self._normalize_int_param(_DEFAULT_REQUIRED_IMAGE_DIMENSION, width_spec)
        if height is None and height_required:
            height = self._normalize_int_param(_DEFAULT_REQUIRED_IMAGE_DIMENSION, height_spec)
        return width, height

    def normalize_image_generation_request(
        self,
        request: ImageGenerationRequest,
    ) -> ImageGenerationRequest:
        if request.control_image is not None:
            raise CapabilityNotSupportedError(
                "Diffusers backend does not support structured control images for text-to-image."
            )
        spec = self._task_spec("text_to_image")
        if spec is None:
            return request
        params = spec.params if isinstance(spec.params, dict) else {}
        negative_prompt = request.negative_prompt
        negative_spec = params.get("negative_prompt")
        if isinstance(negative_spec, dict) and negative_spec.get("supported") is False:
            negative_prompt = None
        width, height = self._normalize_required_image_dimensions(
            width_value=request.width,
            height_value=request.height,
            params=params,
        )
        adapter_defaults = recommended_lora_request_overrides(
            request.lora_adapters,
            extra=request.extra,
            task="text_to_image",
            model=self._cfg.model_id,
        )
        return replace(
            request,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=self._normalize_int_param(
                request.steps if request.steps is not None else adapter_defaults.get("steps"),
                params.get("steps"),
            ),
            guidance_scale=self._normalize_float_param(
                (
                    request.guidance_scale
                    if request.guidance_scale is not None
                    else adapter_defaults.get("guidance_scale")
                ),
                params.get("guidance_scale"),
            ),
        )

    def normalize_image_edit_request(
        self,
        request: ImageEditRequest,
    ) -> ImageEditRequest:
        spec = self._task_spec("image_to_image")
        if spec is None:
            return request
        params = spec.params if isinstance(spec.params, dict) else {}
        negative_prompt = request.negative_prompt
        negative_spec = params.get("negative_prompt")
        if isinstance(negative_spec, dict) and negative_spec.get("supported") is False:
            negative_prompt = None

        extra = dict(request.extra or {})
        width_value = extra.get("width")
        height_value = extra.get("height")
        input_image_size: Optional[Tuple[int, int]] = None
        if width_value is None or height_value is None:
            try:
                input_image_size = self._pil_from_bytes(request.image).size
            except Exception:
                input_image_size = None

        width, height = self._normalize_required_image_dimensions(
            width_value=int(width_value) if width_value is not None else None,
            height_value=int(height_value) if height_value is not None else None,
            params=params,
            input_image_size=input_image_size,
        )
        if width is not None:
            extra["width"] = width
        if height is not None:
            extra["height"] = height
        adapter_defaults = recommended_lora_request_overrides(
            request.lora_adapters,
            extra=request.extra,
            task="image_to_image",
            model=self._cfg.model_id,
        )
        return replace(
            request,
            negative_prompt=negative_prompt,
            steps=self._normalize_int_param(
                request.steps if request.steps is not None else adapter_defaults.get("steps"),
                params.get("steps"),
            ),
            guidance_scale=self._normalize_float_param(
                (
                    request.guidance_scale
                    if request.guidance_scale is not None
                    else adapter_defaults.get("guidance_scale")
                ),
                params.get("guidance_scale"),
            ),
            extra=extra,
        )

    def normalize_video_generation_request(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationRequest:
        spec = self._task_spec("text_to_video")
        if spec is None:
            return request
        params = spec.params if isinstance(spec.params, dict) else {}
        negative_prompt = request.negative_prompt
        negative_spec = params.get("negative_prompt")
        if isinstance(negative_spec, dict) and negative_spec.get("supported") is False:
            negative_prompt = None
        adapter_defaults = recommended_lora_request_overrides(
            request.lora_adapters,
            extra=request.extra,
            task="text_to_video",
            model=self._cfg.model_id,
        )
        return replace(
            request,
            negative_prompt=negative_prompt,
            width=self._normalize_int_param(request.width, params.get("width")),
            height=self._normalize_int_param(request.height, params.get("height")),
            fps=self._normalize_int_param(request.fps, params.get("fps")),
            num_frames=self._normalize_int_param(request.num_frames, params.get("num_frames")),
            steps=self._normalize_int_param(
                request.steps if request.steps is not None else adapter_defaults.get("steps"),
                params.get("steps"),
            ),
            guidance_scale=self._normalize_float_param(
                (
                    request.guidance_scale
                    if request.guidance_scale is not None
                    else adapter_defaults.get("guidance_scale")
                ),
                params.get("guidance_scale"),
            ),
        )

    def normalize_image_to_video_request(
        self,
        request: ImageToVideoRequest,
    ) -> ImageToVideoRequest:
        spec = self._task_spec("image_to_video")
        if spec is None:
            return request
        params = spec.params if isinstance(spec.params, dict) else {}
        negative_prompt = request.negative_prompt
        negative_spec = params.get("negative_prompt")
        if isinstance(negative_spec, dict) and negative_spec.get("supported") is False:
            negative_prompt = None
        adapter_defaults = recommended_lora_request_overrides(
            request.lora_adapters,
            extra=request.extra,
            task="image_to_video",
            model=self._cfg.model_id,
        )
        return replace(
            request,
            negative_prompt=negative_prompt,
            width=self._normalize_int_param(request.width, params.get("width")),
            height=self._normalize_int_param(request.height, params.get("height")),
            fps=self._normalize_int_param(request.fps, params.get("fps")),
            num_frames=self._normalize_int_param(request.num_frames, params.get("num_frames")),
            steps=self._normalize_int_param(
                request.steps if request.steps is not None else adapter_defaults.get("steps"),
                params.get("steps"),
            ),
            guidance_scale=self._normalize_float_param(
                (
                    request.guidance_scale
                    if request.guidance_scale is not None
                    else adapter_defaults.get("guidance_scale")
                ),
                params.get("guidance_scale"),
            ),
        )

    def _apply_loras(
        self,
        *,
        kind: str,
        pipe: Any,
        request_lora_adapters: Any = None,
        extra: Any,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        adapters = resolve_request_lora_adapters(request_lora_adapters, extra=extra)
        loras = serialize_lora_adapters(adapters)
        new_sig = lora_adapter_signature(adapters)
        cur_sig = self._fused_lora_signature.get(kind)
        if new_sig == cur_sig:
            return cur_sig, loras

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
            return None, []

        adapter_names: List[str] = []
        adapter_scales: List[float] = []

        with _hf_offline_env(not bool(self._cfg.allow_download)):
            for adapter_spec, spec in zip(adapters, loras):
                adapter_name = resolved_adapter_name(adapter_spec)
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
                    raise ValueError(
                        "This diffusers pipeline does not support LoRA adapters (missing load_lora_weights)."
                    )
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
        return new_sig, loras

    def _maybe_apply_rapid_aio_transformer(
        self, *, pipe: Any, extra: Any, torch_dtype: Any
    ) -> Optional[str]:
        """Optionally swap the pipeline's transformer with a Rapid-AIO distilled transformer.

        This is primarily useful for Qwen Image Edit pipelines (very fast 4-step inference), but we keep it
        generic: if a pipeline has a `.transformer` module and diffusers provides a compatible transformer
        class, we can hot-swap it.

        Downloads are disabled by default; set allow_download=True only when this override should be fetched online.
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
            QwenImageTransformer2DModel = _lazy_import_qwen_image_transformer_2d_model()
            kwargs: Dict[str, Any] = {
                "subfolder": subfolder,
                "local_files_only": not bool(self._cfg.allow_download),
            }
            if self._cfg.cache_dir:
                kwargs["cache_dir"] = str(self._cfg.cache_dir)
            with _hf_offline_env(not bool(self._cfg.allow_download)):
                tr = QwenImageTransformer2DModel.from_pretrained(
                    repo, torch_dtype=torch_dtype, **kwargs
                )
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
        supported_tasks = self._supported_task_names()
        task_spec = (
            self._task_spec("image_to_image") if "image_to_image" in set(supported_tasks) else None
        )
        supports_mask: Optional[bool] = None
        if task_spec is not None:
            params = task_spec.params if isinstance(task_spec.params, dict) else {}
            if "mask" in params:
                mask_meta = params.get("mask")
                if isinstance(mask_meta, dict):
                    supports_mask = False if mask_meta.get("supported") is False else True
                else:
                    supports_mask = True
            else:
                supports_mask = False
        return VisionBackendCapabilities(
            supported_tasks=supported_tasks,
            supports_mask=supports_mask,
            supports_control_image=False,
        )

    def _is_hf_model_cached(self, model_id: str) -> bool:
        model_id = str(model_id or "").strip()
        if not model_id or "/" not in model_id:
            return False
        return bool(
            cached_hf_model_sources(
                model_id,
                extra_roots=self._hf_cache_roots(),
                required_files=_DIFFUSERS_CACHE_REQUIRED_FILES,
                require_weight_files=True,
            )
        )

    def _discover_cached_hf_diffusers_models(self) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for cache_root in self._hf_cache_roots():
            try:
                candidates = list(cache_root.glob("models--*"))
            except Exception:
                continue
            for folder in candidates:
                name = folder.name
                if not name.startswith("models--"):
                    continue
                model_id = name[len("models--") :].replace("--", "/")
                if "/" not in model_id or model_id in seen:
                    continue
                snap = resolve_hf_repo_snapshot(
                    model_id,
                    extra_roots=[cache_root],
                    required_files=_DIFFUSERS_CACHE_REQUIRED_FILES,
                    require_weight_files=True,
                )
                if snap is None:
                    continue
                seen.add(model_id)
                out.append(model_id)
        return sorted(out)

    def _discover_local_diffusers_models(self) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for root in self._local_diffusers_roots():
            try:
                model_indexes = list(root.rglob("model_index.json"))
            except Exception:
                continue
            for model_index in model_indexes:
                folder = model_index.parent
                name = folder.name
                model_id = name.replace("__", "/") if "__" in name else str(folder)
                if not model_id or model_id in seen:
                    continue
                seen.add(model_id)
                out.append(model_id)
        return sorted(out)

    def list_provider_models(self, *, task: Optional[str] = None) -> List[ProviderModelInfo]:
        """Return locally cached Diffusers/Hugging Face image models.

        Catalog discovery is intentionally non-mutating: it reports cached models
        and the configured model when downloads are explicitly allowed, but it
        never downloads or switches the active pipeline.
        """

        task_s = str(task or "").strip()
        out: List[ProviderModelInfo] = []
        seen: set[str] = set()

        def add_model(
            model_id: str,
            *,
            tasks: List[str],
            cached: bool,
            raw_extra: Optional[Dict[str, Any]] = None,
        ) -> None:
            mid = str(model_id or "").strip()
            if not mid or mid in seen:
                return
            if task_s and task_s not in tasks:
                return
            seen.add(mid)
            raw: Dict[str, Any] = {
                "id": mid,
                "provider": "huggingface",
                "backend": "diffusers",
                "model": f"diffusers/{mid}",
                "routed_model": f"diffusers/{mid}",
                "local_cached": bool(cached),
            }
            if raw_extra:
                raw.update(raw_extra)
            out.append(
                ProviderModelInfo(
                    id=mid,
                    object="model",
                    owned_by=str(raw.get("provider") or "huggingface"),
                    capabilities=tuple(tasks),
                    raw=raw,
                )
            )

        def task_specs_for_model(model_id: str, tasks: Sequence[str]) -> Dict[str, Dict[str, Any]]:
            try:
                spec = VisionModelCapabilitiesRegistry().get(str(model_id))
            except Exception:
                return {}
            allowed = {str(task_name) for task_name in tasks}
            return {
                str(task_name): {
                    "inputs": list(task_spec.inputs),
                    "outputs": list(task_spec.outputs),
                    "params": dict(task_spec.params),
                    "requires": (
                        dict(task_spec.requires) if isinstance(task_spec.requires, dict) else None
                    ),
                }
                for task_name, task_spec in spec.tasks.items()
                if str(task_name) in allowed
            }

        try:
            from ..model_capabilities import VisionModelCapabilitiesRegistry

            reg = VisionModelCapabilitiesRegistry()
            for model_id in reg.list_models():
                spec = reg.get(model_id)
                tasks = self._supported_task_names(str(model_id))
                if task_s and task_s not in tasks:
                    continue
                if not tasks:
                    continue
                if not self._is_hf_model_cached(model_id):
                    continue
                add_model(
                    str(model_id),
                    tasks=tasks,
                    cached=True,
                    raw_extra={
                        "license": spec.license,
                        "notes": spec.notes,
                        "task_specs": {
                            str(task_name): {
                                "inputs": list(task_spec.inputs),
                                "outputs": list(task_spec.outputs),
                                "params": dict(task_spec.params),
                                "requires": (
                                    dict(task_spec.requires)
                                    if isinstance(task_spec.requires, dict)
                                    else None
                                ),
                            }
                            for task_name, task_spec in spec.tasks.items()
                            if str(task_name) in set(tasks)
                        },
                    },
                )
        except Exception:
            pass

        try:
            from ..model_downloads import model_presets

            for preset in model_presets(
                target="auto",
                engine="diffusers",
                include_non_8bit=True,
                include_all_targets=False,
            ):
                if str(preset.target or "").strip().lower() != "gguf":
                    continue
                base_model_id = str(preset.upstream_repo_id or preset.repo_id).strip()
                tasks = self._supported_task_names(base_model_id)
                if task_s and task_s not in tasks:
                    continue
                if not tasks:
                    continue
                gguf_cached_in = cached_hf_model_sources(
                    str(preset.repo_id),
                    extra_roots=self._hf_cache_roots(),
                    required_files=(),
                    require_weight_files=True,
                )
                base_cached = self._is_hf_model_cached(base_model_id)
                if (not gguf_cached_in or not base_cached) and not bool(self._cfg.allow_download):
                    continue
                add_model(
                    str(preset.key),
                    tasks=tasks,
                    cached=bool(gguf_cached_in) and base_cached,
                    raw_extra={
                        "base_model_id": base_model_id,
                        "canonical_model_id": base_model_id,
                        "download_repo_id": str(preset.repo_id),
                        "engine": str(preset.engine),
                        "target": str(preset.target),
                        "quantization_bits": preset.quantization_bits,
                        "variant": str(preset.display_name),
                        "local_cached": bool(gguf_cached_in) and base_cached,
                        "gguf_local_cached": bool(gguf_cached_in),
                        "base_local_cached": base_cached,
                        "cached_in": list(gguf_cached_in),
                        "task_specs": task_specs_for_model(base_model_id, tasks),
                    },
                )
        except Exception:
            pass

        for model_id in [
            *self._discover_cached_hf_diffusers_models(),
            *self._discover_local_diffusers_models(),
        ]:
            if model_id in seen:
                continue
            tasks = ["image_to_image", "text_to_image"]
            add_model(
                str(model_id),
                tasks=tasks,
                cached=True,
                raw_extra={
                    "discovered": True,
                    "notes": "Discovered from local Diffusers cache metadata (model_index.json present).",
                },
            )

        configured_model = str(self._cfg.model_id or "").strip()
        if configured_model and configured_model not in seen:
            cached = self._is_hf_model_cached(configured_model)
            if cached or bool(self._cfg.allow_download):
                add_model(
                    configured_model,
                    tasks=[str(t) for t in self.get_capabilities().supported_tasks],
                    cached=cached,
                    raw_extra={"configured": True},
                )

        return out

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

    def _framework_candidate_roots(self) -> List[Path]:
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

    def _hf_cache_roots(self) -> List[Path]:
        roots: List[Path] = [self._hf_cache_root()]
        for key in (
            "ABSTRACTVISION_HF_HUB_CACHE",
            "ABSTRACTCORE_VISION_HF_HUB_CACHE",
            "HF_HUB_CACHE_DIR",
        ):
            value = os.environ.get(key)
            if value:
                roots.append(Path(value).expanduser())
        for root in self._framework_candidate_roots():
            roots.append(root / "runtime" / "hf-hub")
            quarantine = root / "runtime" / "model-quarantine"
            try:
                if quarantine.is_dir():
                    roots.extend(path / "hf-hub" for path in quarantine.iterdir() if path.is_dir())
            except Exception:
                pass
        out: List[Path] = []
        seen: set[str] = set()
        for root in roots:
            try:
                resolved = root.expanduser()
            except Exception:
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            if resolved.is_dir():
                out.append(resolved)
        return out

    def _local_diffusers_roots(self) -> List[Path]:
        roots: List[Path] = []
        for key in (
            "ABSTRACTVISION_MODELS_DIR",
            "ABSTRACTVISION_MODEL_DIR",
            "ABSTRACTCORE_VISION_MODELS_DIR",
            "ABSTRACTCORE_VISION_MODEL_DIR",
        ):
            value = os.environ.get(key)
            if value:
                roots.append(Path(value).expanduser())
        for root in self._framework_candidate_roots():
            roots.append(root / "untracked" / "models" / "abstractvision")
            roots.append(root / "runtime" / "models" / "abstractvision")
            quarantine = root / "runtime" / "model-quarantine"
            try:
                if quarantine.is_dir():
                    roots.extend(path / "models" for path in quarantine.iterdir() if path.is_dir())
            except Exception:
                pass
        out: List[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root)
            if key in seen:
                continue
            seen.add(key)
            if root.is_dir():
                out.append(root)
        return out

    def _resolve_snapshot_dir(self, model_id: Optional[str] = None) -> Optional[Path]:
        model_id = str(model_id if model_id is not None else self._cfg.model_id).strip()
        if not model_id:
            return None

        p = Path(model_id).expanduser()
        if p.exists():
            return p

        if "/" not in model_id:
            return None
        return resolve_hf_repo_snapshot(
            model_id,
            revision=str(self._cfg.revision or "main").strip() or "main",
            extra_roots=self._hf_cache_roots(),
            required_files=_DIFFUSERS_CACHE_REQUIRED_FILES,
            require_weight_files=True,
        )

    def _resolve_any_snapshot_dir(self, model_id: Optional[str] = None) -> Optional[Path]:
        model_id = str(model_id if model_id is not None else self._cfg.model_id).strip()
        if not model_id or "/" not in model_id:
            return None
        return resolve_hf_repo_snapshot(
            model_id,
            revision=str(self._cfg.revision or "main").strip() or "main",
            extra_roots=self._hf_cache_roots(),
            reject_incomplete=False,
        )

    def _preflight_check_model_index(self, model_id: Optional[str] = None) -> None:
        snap = self._resolve_snapshot_dir(model_id=model_id)
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

        gguf_spec = self._diffusers_gguf_transformer_spec()
        if gguf_spec is not None and _is_mps_device(device):
            raise ValueError(
                "Diffusers GGUF is not a native Apple MPS 8-bit execution path. "
                "Diffusers currently dequantizes GGUF weights for non-CUDA linear layers, which can be slower "
                "and use similar peak memory to BF16/FP16 on MPS. Use the stable-diffusion.cpp provider instead: "
                "`abstractvision i2i --provider sdcpp --model qwen-image-edit-2511-gguf ...`."
            )
        preflight_model_id = gguf_spec.base_model_id if gguf_spec is not None else None
        self._preflight_check_model_index(model_id=preflight_model_id)
        _maybe_patch_transformers_clip_position_ids()

        torch_dtype = _torch_dtype_from_str(torch, self._cfg.torch_dtype)
        if torch_dtype is None:
            torch_dtype = _default_torch_dtype_for_device(torch, device)
        torch_dtype = self._preferred_torch_dtype_for_kind(kind, device, torch, torch_dtype)
        common = self._pipeline_common_kwargs()
        if bool(self._cfg.low_cpu_mem_usage):
            common["low_cpu_mem_usage"] = True

        gguf_transformer = None
        if gguf_spec is not None:
            if kind not in {"t2i", "i2i", "inpaint"}:
                raise ValueError(
                    "Diffusers GGUF transformer loading is only supported for image pipelines."
                )
            gguf_path = self._resolve_gguf_transformer_path(gguf_spec)
            gguf_transformer = self._load_gguf_transformer(
                spec=gguf_spec,
                gguf_path=gguf_path,
                torch=torch,
                torch_dtype=torch_dtype,
                device=str(device),
            )

        # Auto-select checkpoint variants when appropriate (best-effort).
        # Prefer fp16 on GPU backends (CUDA/MPS) to cut memory/disk use, but never on CPU.
        #
        # Important: many repos do NOT ship an fp16 variant, so we must fall back cleanly.
        auto_variant: Optional[str] = None
        if (
            gguf_spec is None
            and not str(getattr(self._cfg, "variant", "") or "").strip()
            and str(device).strip().lower() != "cpu"
        ):
            if torch_dtype == getattr(torch, "float16", object()):
                auto_variant = "fp16"

        load_model_id = str(
            gguf_spec.base_model_id if gguf_spec is not None else self._cfg.model_id
        )
        snap: Optional[Path] = None
        if not bool(self._cfg.allow_download):
            snap = self._resolve_snapshot_dir(model_id=load_model_id)
            if snap is not None:
                load_model_id = str(snap)

        def _from_pretrained(cls: Any) -> Any:
            load_common = dict(common)
            if gguf_transformer is not None:
                load_common["transformer"] = gguf_transformer
            if auto_variant:
                common2 = dict(load_common)
                common2["variant"] = auto_variant
                try:
                    return cls.from_pretrained(load_model_id, torch_dtype=torch_dtype, **common2)
                except Exception:
                    # If the repo doesn't provide the fp16 variant (common), fall back to regular weights.
                    return cls.from_pretrained(
                        load_model_id, torch_dtype=torch_dtype, **load_common
                    )
            return cls.from_pretrained(load_model_id, torch_dtype=torch_dtype, **load_common)

        def _maybe_raise_offline_missing_model(e: Exception) -> None:
            model_id = str(
                gguf_spec.base_model_id if gguf_spec is not None else self._cfg.model_id or ""
            ).strip()
            if not model_id or "/" not in model_id:
                return
            # If it's not in cache, provide a clearer message than the upstream
            # "does not appear to have a file named model_index.json" wording.
            if self._resolve_snapshot_dir(model_id=model_id) is not None:
                return
            msg = str(e)
            if "model_index.json" not in msg:
                return
            raise ValueError(
                f"Model {model_id!r} is not available locally and downloads are disabled. "
                "Pre-download it outside the REPL (for example with `huggingface-cli download ...`) or explicitly "
                "enable runtime downloads (set allow_download=True in Python or ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1). "
                "If the model is gated, accept its terms on Hugging Face and set `HF_TOKEN` before downloading."
            ) from e

        def _maybe_raise_incomplete_snapshot_error(e: Exception) -> None:
            model_id = str(
                gguf_spec.base_model_id if gguf_spec is not None else self._cfg.model_id or ""
            ).strip()
            snap = self._resolve_any_snapshot_dir(model_id=model_id)
            if not model_id or not snap:
                return
            if hf_snapshot_is_usable(
                snap,
                required_files=_DIFFUSERS_CACHE_REQUIRED_FILES,
                require_weight_files=True,
            ):
                return
            details: List[str] = []
            if hf_snapshot_has_incomplete_downloads(snap):
                details.append("interrupted download markers were found")
            if not hf_snapshot_has_weight_files(snap):
                details.append("model weight files are missing")
            missing_indexed = hf_snapshot_missing_indexed_weight_files(snap)
            if missing_indexed:
                preview = ", ".join(missing_indexed[:4])
                if len(missing_indexed) > 4:
                    preview += ", ..."
                details.append(f"indexed shard files are missing ({preview})")
            detail_s = "; ".join(details) if details else "required files are missing"
            raise ValueError(
                f"Local Diffusers snapshot for {model_id!r} is incomplete: {snap} ({detail_s}). "
                "Re-run `abstractvision download` for this model, or delete the broken cache snapshot and download it again."
            ) from e

        def _maybe_raise_missing_torchvision(e: Exception) -> None:
            msg = str(e or "")
            lowered = msg.lower()
            if "torchvision" not in lowered:
                return
            if "requires the torchvision library" not in lowered:
                return
            if "not found in your environment" not in lowered:
                return
            import sys

            raise OptionalDependencyMissingError(
                "Optional dependency missing: torchvision. "
                "Some Transformers processors used by Qwen Image Edit pipelines (e.g. Qwen2VLVideoProcessor) "
                "import torchvision at runtime. "
                f"Install via: {_DIFFUSERS_RUNTIME_HINT} (or {_LOCAL_RUNTIME_HINT}); or `pip install torchvision` "
                f"(python={sys.executable})"
            ) from e

        pipe = None
        with _hf_offline_env(not bool(self._cfg.allow_download)):
            if kind == "t2i":
                # Prefer AutoPipeline when available, but fall back to DiffusionPipeline for robustness.
                if AutoPipelineForText2Image is not None:
                    try:
                        pipe = _from_pretrained(AutoPipelineForText2Image)
                    except Exception as e:
                        _maybe_raise_offline_missing_model(e)
                        _maybe_raise_incomplete_snapshot_error(e)
                        _maybe_raise_missing_torchvision(e)
                        pipe = None
                if pipe is None:
                    try:
                        pipe = _from_pretrained(DiffusionPipeline)
                    except Exception as e:
                        _maybe_raise_offline_missing_model(e)
                        _maybe_raise_incomplete_snapshot_error(e)
                        _maybe_raise_missing_torchvision(e)
                        raise
            elif kind == "i2i":
                if AutoPipelineForImage2Image is not None:
                    try:
                        pipe = _from_pretrained(AutoPipelineForImage2Image)
                    except Exception as e:
                        _maybe_raise_offline_missing_model(e)
                        _maybe_raise_incomplete_snapshot_error(e)
                        _maybe_raise_missing_torchvision(e)
                        pipe = None
                if pipe is None:
                    try:
                        pipe = _from_pretrained(DiffusionPipeline)
                    except Exception as e:
                        _maybe_raise_offline_missing_model(e)
                        _maybe_raise_incomplete_snapshot_error(e)
                        _maybe_raise_missing_torchvision(e)
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
                try:
                    pipe = _from_pretrained(AutoPipelineForInpainting)
                except Exception as e:
                    _maybe_raise_offline_missing_model(e)
                    _maybe_raise_incomplete_snapshot_error(e)
                    _maybe_raise_missing_torchvision(e)
                    raise
            elif kind == "t2v":
                try:
                    pipe = _from_pretrained(DiffusionPipeline)
                except Exception as e:
                    _maybe_raise_offline_missing_model(e)
                    _maybe_raise_incomplete_snapshot_error(e)
                    _maybe_raise_missing_torchvision(e)
                    raise ValueError(
                        "Diffusers could not load a text-to-video pipeline for this model id. "
                        "Install/upgrade diffusers (and compatible transformers/torch), or use a model repo that "
                        "ships a text-to-video Diffusers pipeline. "
                        f"(diffusers={diffusers_version})"
                    ) from e
            else:
                raise ValueError(f"Unknown pipeline kind: {kind!r}")

        pipe = _move_pipe_to_device(
            pipe, device=str(device), dtype=None if gguf_transformer is not None else torch_dtype
        )
        template_snapshot = snap
        if template_snapshot is None:
            candidate = Path(str(load_model_id)).expanduser()
            if candidate.exists():
                template_snapshot = candidate
            else:
                template_snapshot = self._resolve_any_snapshot_dir(
                    model_id=gguf_spec.base_model_id if gguf_spec is not None else None
                )
        _ensure_pipeline_chat_templates(
            pipe, snapshot_dir=template_snapshot, model_id=str(self._cfg.model_id or "")
        )
        _maybe_cast_pipe_modules_to_dtype(pipe, dtype=torch_dtype)
        _maybe_upcast_vae_for_mps(torch, pipe, device, allow_fp32_vae=kind not in {"t2v", "i2v"})
        if kind in {"t2v", "i2v"}:
            _maybe_enable_video_pipeline_memory_savers(pipe)
        self._set_pipeline(kind, pipe)
        return pipe

    def _set_pipeline(self, kind: str, pipe: Any) -> Any:
        current = self._pipelines.get(kind)
        if current is not pipe:
            self._warmed_pipeline_ids.pop(kind, None)
        self._pipelines[kind] = pipe
        self._call_params[kind] = _call_param_names(getattr(pipe, "__call__", None))
        return pipe

    def _is_pipeline_warm(self, kind: str, pipe: Any) -> bool:
        return self._warmed_pipeline_ids.get(kind) == id(pipe)

    def _mark_pipeline_warm(self, kind: str, pipe: Any) -> None:
        self._warmed_pipeline_ids[kind] = id(pipe)

    def _pil_from_bytes(self, data: bytes):
        Image = _lazy_import_pil()
        img = Image.open(io.BytesIO(bytes(data)))
        # Many pipelines expect RGB.
        return img.convert("RGB")

    def _png_bytes(self, img) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _video_frames(self, out: Any) -> List[Any]:
        Image = _lazy_import_pil()
        raw_frames = getattr(out, "frames", None)
        if isinstance(raw_frames, list) and raw_frames:
            if isinstance(raw_frames[0], list):
                frames = raw_frames[0]
            else:
                frames = raw_frames
            if all(isinstance(frame, Image.Image) for frame in frames):
                return list(frames)
        raise ValueError("Diffusers pipeline returned no video frames")

    def _seed_generator(self, seed: Optional[int]):
        if seed is None:
            return None
        torch = _lazy_import_torch()
        d = str(self._effective_device(torch) or "").strip().lower()
        gen_device = (
            "cpu" if d == "mps" or d.startswith("mps:") else str(self._effective_device(torch))
        )
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
            if (
                isinstance(extrema, tuple)
                and len(extrema) == 2
                and all(isinstance(x, int) for x in extrema)
            ):
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
        if _pipe_has_quantized_components(pipe):
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
            current_dtype = _torch_dtype_from_str(
                torch, self._cfg.torch_dtype
            ) or _default_torch_dtype_for_device(torch, device)

        candidates: list[Any] = []
        if current_dtype == getattr(torch, "bfloat16", object()):
            candidates.append(torch.float16)
        allow_fp32_retry = kind not in {"t2v", "i2v"}
        if allow_fp32_retry and current_dtype != getattr(torch, "float32", object()):
            candidates.append(torch.float32)

        for target in candidates:
            try:
                pipe2 = _move_pipe_to_device(pipe, device=str(device), dtype=target)
            except Exception:
                continue

            _maybe_upcast_vae_for_mps(torch, pipe2, device)
            self._set_pipeline(kind, pipe2)

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
        if _pipe_has_quantized_components(pipe):
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
            pipe_fp32 = _move_pipe_to_device(pipe, device=str(device), dtype=torch.float32)
        except Exception:
            return None

        _maybe_upcast_vae_for_mps(torch, pipe_fp32, device)
        self._set_pipeline(kind, pipe_fp32)

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
        if not self._supports_backend_task("text_to_image"):
            raise CapabilityNotSupportedError(
                "This model is temporarily disabled for local Diffusers text_to_image generation."
            )
        with self._backend_lock:
            request = self.normalize_image_generation_request(request)
            pipe = self._get_or_load_pipeline("t2i")
            call_params = self._call_params.get("t2i")
            total_steps = int(request.steps) if request.steps is not None else None

            torch_dtype = getattr(pipe, "dtype", None)
            if torch_dtype is None:
                torch = _lazy_import_torch()
                device = self._effective_device(torch)
                torch_dtype = _torch_dtype_from_str(
                    torch, self._cfg.torch_dtype
                ) or _default_torch_dtype_for_device(torch, device)
            rapid_repo = self._maybe_apply_rapid_aio_transformer(
                pipe=pipe, extra=request.extra, torch_dtype=torch_dtype
            )
            lora_sig, lora_specs = self._apply_loras(
                kind="t2i",
                pipe=pipe,
                request_lora_adapters=request.lora_adapters,
                extra=request.extra,
            )

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
                    if request.negative_prompt is None and (
                        call_params is None or "negative_prompt" in call_params
                    ):
                        kwargs["negative_prompt"] = " "
                else:
                    kwargs["guidance_scale"] = float(request.guidance_scale)
            gen = self._seed_generator(request.seed)
            if gen is not None:
                kwargs["generator"] = gen

            kwargs.update(_forward_extra_kwargs(request.extra, call_params=call_params))

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
            current_pipe = self._pipelines.get("t2i", pipe)
            self._mark_pipeline_warm("t2i", current_pipe)
            png = self._png_bytes(images[0])
            meta = {"source": "diffusers", "model_id": self._cfg.model_id}
            if rapid_repo:
                meta["rapid_aio_repo"] = rapid_repo
            if lora_sig:
                meta["lora_signature"] = lora_sig
                meta["requested_lora_adapters"] = lora_specs
                meta["applied_lora_adapters"] = lora_specs
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
        if not self._supports_backend_task("image_to_image"):
            raise CapabilityNotSupportedError(
                "This model is temporarily disabled for local Diffusers image_to_image generation."
            )
        with self._backend_lock:
            request = self.normalize_image_edit_request(request)
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
                torch_dtype = _torch_dtype_from_str(
                    torch, self._cfg.torch_dtype
                ) or _default_torch_dtype_for_device(torch, device)
            rapid_repo = self._maybe_apply_rapid_aio_transformer(
                pipe=pipe, extra=request.extra, torch_dtype=torch_dtype
            )
            lora_sig, lora_specs = self._apply_loras(
                kind=kind,
                pipe=pipe,
                request_lora_adapters=request.lora_adapters,
                extra=request.extra,
            )

            img = self._pil_from_bytes(request.image)
            image_input: Any = (
                [img] if request.mask is None and self._needs_list_wrapped_i2i_image(pipe) else img
            )
            kwargs: Dict[str, Any] = {"prompt": request.prompt, "image": image_input}
            # Some modern edit pipelines (notably Qwen Image Edit) default to ~1MP output when
            # `width`/`height` are omitted, which can blow up memory unexpectedly. When the pipeline
            # supports explicit `width`/`height` and the caller didn't supply them, default to the
            # input image dimensions.
            try:
                extra = request.extra if isinstance(request.extra, dict) else {}
                wants_width = call_params is None or "width" in call_params
                wants_height = call_params is None or "height" in call_params
                has_width = isinstance(extra, dict) and extra.get("width") is not None
                has_height = isinstance(extra, dict) and extra.get("height") is not None
                in_w, in_h = img.size
                if wants_width and not has_width:
                    kwargs["width"] = int(in_w)
                if wants_height and not has_height:
                    kwargs["height"] = int(in_h)
            except Exception:
                pass
            if request.mask is not None:
                kwargs["mask_image"] = self._pil_from_bytes(request.mask)
            if request.negative_prompt is not None:
                kwargs["negative_prompt"] = request.negative_prompt
            if request.steps is not None:
                kwargs["num_inference_steps"] = int(request.steps)
            if request.guidance_scale is not None:
                if call_params is not None and "true_cfg_scale" in call_params:
                    kwargs["true_cfg_scale"] = float(request.guidance_scale)
                    if request.negative_prompt is None and (
                        call_params is None or "negative_prompt" in call_params
                    ):
                        kwargs["negative_prompt"] = " "
                else:
                    kwargs["guidance_scale"] = float(request.guidance_scale)
            gen = self._seed_generator(request.seed)
            if gen is not None:
                kwargs["generator"] = gen

            kwargs.update(_forward_extra_kwargs(request.extra, call_params=call_params))

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
            current_pipe = self._pipelines.get(kind, pipe)
            self._mark_pipeline_warm(kind, current_pipe)
            png = self._png_bytes(images[0])
            meta = {"source": "diffusers", "model_id": self._cfg.model_id}
            if rapid_repo:
                meta["rapid_aio_repo"] = rapid_repo
            if lora_sig:
                meta["lora_signature"] = lora_sig
                meta["requested_lora_adapters"] = lora_specs
                meta["applied_lora_adapters"] = lora_specs
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
        raise CapabilityNotSupportedError(
            "HuggingFaceDiffusersVisionBackend does not implement multi-view generation."
        )

    def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
        return self.generate_video_with_progress(request, progress_callback=None)

    def generate_video_with_progress(
        self,
        request: VideoGenerationRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        if not self._supports_backend_task("text_to_video"):
            raise CapabilityNotSupportedError(
                "Local Diffusers text_to_video is experimental and temporarily disabled for this model."
            )
        with self._backend_lock:
            pipe = self._get_or_load_pipeline("t2v")
            call_params = self._call_params.get("t2v")
            total_steps = int(request.steps) if request.steps is not None else None

            kwargs: Dict[str, Any] = {
                "prompt": request.prompt,
            }
            if request.negative_prompt is not None:
                kwargs["negative_prompt"] = request.negative_prompt
            if request.width is not None:
                kwargs["width"] = int(request.width)
            if request.height is not None:
                kwargs["height"] = int(request.height)
            if request.num_frames is not None:
                kwargs["num_frames"] = int(request.num_frames)
            if request.steps is not None:
                kwargs["num_inference_steps"] = int(request.steps)
            if request.guidance_scale is not None:
                kwargs["guidance_scale"] = float(request.guidance_scale)
            if call_params is None or "output_type" in call_params:
                kwargs["output_type"] = "pil"
            gen = self._seed_generator(request.seed)
            if gen is not None:
                kwargs["generator"] = gen

            kwargs.update(_forward_extra_kwargs(request.extra, call_params=call_params))

            try:
                call_kwargs = dict(kwargs)
                if progress_callback is not None:
                    call_kwargs["__abstractvision_progress_callback"] = progress_callback
                    call_kwargs["__abstractvision_progress_total_steps"] = total_steps
                out, _had_invalid_cast = self._pipe_call(pipe, call_kwargs)
            except Exception as e:
                out2 = self._maybe_retry_on_dtype_mismatch(
                    kind="t2v",
                    pipe=pipe,
                    kwargs=kwargs,
                    error=e,
                    progress_callback=progress_callback,
                    total_steps=total_steps,
                )
                if out2 is None:
                    raise
                out = out2

            frames = self._video_frames(out)
            fps = int(request.fps) if request.fps is not None else 8
            mp4 = _frames_to_mp4_bytes(frames, fps=fps)
            current_pipe = self._pipelines.get("t2v", pipe)
            self._mark_pipeline_warm("t2v", current_pipe)
            meta: Dict[str, Any] = {
                "source": "diffusers",
                "model_id": self._cfg.model_id,
                "fps": fps,
                "frame_count": len(frames),
            }
            try:
                width, height = frames[0].size
                meta["width"] = int(width)
                meta["height"] = int(height)
            except Exception:
                pass
            try:
                current_pipe = self._pipelines.get("t2v", pipe)
                dtype = getattr(current_pipe, "dtype", None)
                device = getattr(current_pipe, "device", None)
                if dtype is not None:
                    meta["dtype"] = str(dtype)
                if device is not None:
                    meta["device"] = str(device)
            except Exception:
                pass
            return GeneratedAsset(
                media_type="video",
                data=mp4,
                mime_type="video/mp4",
                metadata=meta,
            )

    def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
        raise CapabilityNotSupportedError(
            "HuggingFaceDiffusersVisionBackend does not implement local image_to_video yet."
        )
