from __future__ import annotations

import base64
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..artifacts import MediaStore, get_artifact_id, is_artifact_ref
from ..errors import AbstractVisionError, OptionalDependencyMissingError
from ..model_capabilities import VisionModelCapabilitiesRegistry
from ..types import LoRAAdapterSpec, ProviderAdapterInfo
from ..vision_manager import VisionManager


def _require_abstractcore_tool():
    try:
        from abstractcore import tool  # type: ignore
    except Exception as e:  # pragma: no cover (covered indirectly by import failures)
        raise OptionalDependencyMissingError(
            "AbstractCore is required for this integration. Install it via: pip install abstractcore"
        ) from e
    return tool


def _decode_base64_bytes(value: str) -> bytes:
    raw = str(value or "").strip()
    if not raw:
        return b""
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1].strip()
    # Best-effort: tolerate missing padding/newlines.
    raw = "".join(raw.split())
    pad = (-len(raw)) % 4
    if pad:
        raw = raw + ("=" * pad)
    return base64.b64decode(raw, validate=False)


def _require_store(vm: VisionManager) -> MediaStore:
    store = getattr(vm, "store", None)
    if store is None:
        raise AbstractVisionError("VisionManager.store is required for tool integration (artifact-ref outputs).")
    return store


def _provider_adapter_to_dict(info: ProviderAdapterInfo) -> Dict[str, Any]:
    return {
        "id": str(info.id),
        "repo_id": info.repo_id,
        "base_models": [str(item) for item in info.base_models],
        "compatible_models": [str(item) for item in info.compatible_models],
        "compatible_tasks": [str(item) for item in info.compatible_tasks],
        "suggested_target_roles": [str(item) for item in info.suggested_target_roles],
        "raw": dict(info.raw),
    }


def _resolve_input_bytes(
    *,
    store: MediaStore,
    artifact: Optional[Dict[str, Any]],
    b64: Optional[str],
    name: str,
    required: bool,
) -> Optional[bytes]:
    if artifact is not None:
        if not is_artifact_ref(artifact):
            raise ValueError(f"{name}: expected an artifact ref dict like {{'$artifact': '...'}}")
        return store.load_bytes(get_artifact_id(artifact))
    if b64 is not None:
        out = _decode_base64_bytes(b64)
        if required and not out:
            raise ValueError(f"{name}: base64 payload decoded to empty bytes")
        return out
    if required:
        raise ValueError(f"{name}: either {name}_artifact or {name}_b64 is required")
    return None


def _resolve_optional_input_bytes_list(
    *,
    store: MediaStore,
    items: Optional[Sequence[Dict[str, Any]]],
    name: str,
) -> List[bytes]:
    resolved: List[bytes] = []
    for index, item in enumerate(items or ()):
        if not isinstance(item, dict):
            raise ValueError(f"{name}[{index}]: expected an artifact ref or {{'b64': '...'}} dict")
        if is_artifact_ref(item):
            resolved.append(store.load_bytes(get_artifact_id(item)))
            continue
        b64_value = item.get("b64")
        if b64_value is None:
            b64_value = item.get("base64")
        if b64_value is None:
            b64_value = item.get("data")
        if b64_value is None:
            raise ValueError(
                f"{name}[{index}]: expected an artifact ref or a dict containing 'b64', 'base64', or 'data'"
            )
        payload = _decode_base64_bytes(str(b64_value))
        if not payload:
            raise ValueError(f"{name}[{index}]: base64 payload decoded to empty bytes")
        resolved.append(payload)
    return resolved


def _coerce_lora_adapters(value: Optional[Sequence[Dict[str, Any]]]) -> List[LoRAAdapterSpec]:
    adapters: List[LoRAAdapterSpec] = []
    for index, item in enumerate(value or ()):
        if isinstance(item, LoRAAdapterSpec):
            adapters.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(f"lora_adapters[{index}]: expected a dict")
        source = str(item.get("source") or "").strip()
        if not source:
            raise ValueError(f"lora_adapters[{index}]: 'source' is required")
        scale = item.get("scale")
        adapters.append(
            LoRAAdapterSpec(
                source=source,
                scale=float(scale) if scale is not None else None,
                weight_name=(
                    str(item.get("weight_name")).strip()
                    if item.get("weight_name") is not None
                    else None
                ),
                subfolder=(
                    str(item.get("subfolder")).strip()
                    if item.get("subfolder") is not None
                    else None
                ),
                adapter_name=(
                    str(item.get("adapter_name")).strip()
                    if item.get("adapter_name") is not None
                    else None
                ),
                target_role=(
                    str(item.get("target_role")).strip()
                    if item.get("target_role") is not None
                    else None
                ),
            )
        )
    return adapters


def make_vision_tools(
    *,
    vision_manager: VisionManager,
    model_id: str,
    registry: Optional[VisionModelCapabilitiesRegistry] = None,
) -> List[Callable[..., Any]]:
    """Create AbstractCore tools for generative vision (artifact-ref outputs).

    Tools are returned as normal Python callables decorated with `@abstractcore.tool`.
    """
    tool = _require_abstractcore_tool()
    reg = registry or VisionModelCapabilitiesRegistry()
    store = _require_store(vision_manager)
    model_id = str(model_id or "").strip()
    if not model_id:
        raise ValueError("model_id must be a non-empty string")

    def _task_param_value(task_name: str, param_name: str, fallback: Any = None) -> Any:
        try:
            task = reg.get(model_id).tasks.get(task_name)
        except Exception:
            task = None
        params = task.params if task is not None and isinstance(task.params, dict) else {}
        spec = params.get(param_name)
        if isinstance(spec, dict):
            if spec.get("const") is not None:
                return spec.get("const")
            if spec.get("default") is not None:
                return spec.get("default")
        return fallback

    def _require_artifact_ref_output(name: str, output: Any) -> Dict[str, Any]:
        if not (isinstance(output, dict) and is_artifact_ref(output)):
            raise AbstractVisionError(
                f"{name} expected artifact-ref output; ensure VisionManager.store is set."
            )
        return output

    def _require_artifact_ref_list(name: str, output: Any) -> List[Dict[str, Any]]:
        if not (
            isinstance(output, list)
            and all(isinstance(item, dict) and is_artifact_ref(item) for item in output)
        ):
            raise AbstractVisionError(
                f"{name} expected a list of artifact refs; ensure VisionManager.store is set."
            )
        return output

    @tool(
        name="vision_list_adapters",
        description="List locally discovered adapters compatible with the selected model and optional task.",
        tags=["vision", "catalog", "adapters"],
        when_to_use="Use before generation when you need to inspect compatible installed adapters for the active model.",
    )
    def vision_list_adapters(
        task: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return [
            _provider_adapter_to_dict(info)
            for info in vision_manager.list_provider_adapters(model=model_id, task=task)
        ]

    @tool(
        name="vision_text_to_image",
        description="Generate an image from a text prompt and return an artifact ref.",
        tags=["vision", "generate", "image"],
        when_to_use="Use when you need to create a new image from a prompt.",
    )
    def vision_text_to_image(
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        lora_adapters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        reg.require_support(model_id, "text_to_image")
        out = vision_manager.generate_image(
            prompt,
            negative_prompt=negative_prompt,
            width=width if width is not None else _task_param_value("text_to_image", "width", None),
            height=height if height is not None else _task_param_value("text_to_image", "height", None),
            steps=steps if steps is not None else _task_param_value("text_to_image", "steps", 10),
            guidance_scale=(
                guidance_scale
                if guidance_scale is not None
                else _task_param_value("text_to_image", "guidance_scale", None)
            ),
            seed=seed,
            lora_adapters=_coerce_lora_adapters(lora_adapters),
        )
        return _require_artifact_ref_output("vision_text_to_image", out)

    @tool(
        name="vision_text_to_image_batch",
        description="Generate multiple images from a text prompt and return artifact refs.",
        tags=["vision", "generate", "image", "batch"],
        when_to_use="Use when you need several text-to-image generations in one call, optionally with explicit seeds and a LoRA stack.",
    )
    def vision_text_to_image_batch(
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        count: int = 1,
        seeds: Optional[List[int]] = None,
        lora_adapters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        reg.require_support(model_id, "text_to_image")
        out = vision_manager.generate_image_batch(
            prompt,
            negative_prompt=negative_prompt,
            width=width if width is not None else _task_param_value("text_to_image", "width", None),
            height=height if height is not None else _task_param_value("text_to_image", "height", None),
            steps=steps if steps is not None else _task_param_value("text_to_image", "steps", 10),
            guidance_scale=(
                guidance_scale
                if guidance_scale is not None
                else _task_param_value("text_to_image", "guidance_scale", None)
            ),
            seed=seed,
            count=int(count),
            seeds=seeds,
            lora_adapters=_coerce_lora_adapters(lora_adapters),
        )
        return _require_artifact_ref_list("vision_text_to_image_batch", out)

    @tool(
        name="vision_image_to_image",
        description="Edit/transform an input image using a prompt and return an artifact ref.",
        tags=["vision", "edit", "image"],
        when_to_use="Use when you need to modify an existing image (optionally with a mask).",
    )
    def vision_image_to_image(
        prompt: str,
        image_artifact: Optional[Dict[str, Any]] = None,
        image_b64: Optional[str] = None,
        mask_artifact: Optional[Dict[str, Any]] = None,
        mask_b64: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        reference_images: Optional[List[Dict[str, Any]]] = None,
        lora_adapters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        reg.require_support(model_id, "image_to_image")
        image_bytes = _resolve_input_bytes(store=store, artifact=image_artifact, b64=image_b64, name="image", required=True)
        mask_bytes = _resolve_input_bytes(store=store, artifact=mask_artifact, b64=mask_b64, name="mask", required=False)
        reference_image_bytes = _resolve_optional_input_bytes_list(
            store=store,
            items=reference_images,
            name="reference_images",
        )
        extra: Dict[str, Any] = {}
        if reference_image_bytes:
            extra["reference_images"] = reference_image_bytes
        out = vision_manager.edit_image(
            prompt,
            image=image_bytes or b"",
            mask=mask_bytes,
            negative_prompt=negative_prompt,
            steps=steps if steps is not None else _task_param_value("image_to_image", "steps", 15),
            guidance_scale=(
                guidance_scale
                if guidance_scale is not None
                else _task_param_value("image_to_image", "guidance_scale", None)
            ),
            seed=seed,
            extra=extra,
            lora_adapters=_coerce_lora_adapters(lora_adapters),
        )
        return _require_artifact_ref_output("vision_image_to_image", out)

    @tool(
        name="vision_image_to_image_batch",
        description="Create multiple image edits from one input image and return artifact refs.",
        tags=["vision", "edit", "image", "batch"],
        when_to_use="Use when you need several image-edit generations in one call, optionally with explicit seeds, reference images, and a LoRA stack.",
    )
    def vision_image_to_image_batch(
        prompt: str,
        image_artifact: Optional[Dict[str, Any]] = None,
        image_b64: Optional[str] = None,
        mask_artifact: Optional[Dict[str, Any]] = None,
        mask_b64: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        count: int = 1,
        seeds: Optional[List[int]] = None,
        reference_images: Optional[List[Dict[str, Any]]] = None,
        lora_adapters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        reg.require_support(model_id, "image_to_image")
        image_bytes = _resolve_input_bytes(store=store, artifact=image_artifact, b64=image_b64, name="image", required=True)
        mask_bytes = _resolve_input_bytes(store=store, artifact=mask_artifact, b64=mask_b64, name="mask", required=False)
        reference_image_bytes = _resolve_optional_input_bytes_list(
            store=store,
            items=reference_images,
            name="reference_images",
        )
        extra: Dict[str, Any] = {}
        if reference_image_bytes:
            extra["reference_images"] = reference_image_bytes
        out = vision_manager.edit_image_batch(
            prompt,
            image=image_bytes or b"",
            mask=mask_bytes,
            negative_prompt=negative_prompt,
            steps=steps if steps is not None else _task_param_value("image_to_image", "steps", 15),
            guidance_scale=(
                guidance_scale
                if guidance_scale is not None
                else _task_param_value("image_to_image", "guidance_scale", None)
            ),
            seed=seed,
            count=int(count),
            seeds=seeds,
            extra=extra,
            lora_adapters=_coerce_lora_adapters(lora_adapters),
        )
        return _require_artifact_ref_list("vision_image_to_image_batch", out)

    @tool(
        name="vision_multi_view_image",
        description="Generate multiple views/angles of a concept and return artifact refs.",
        tags=["vision", "generate", "image", "multi_view"],
        when_to_use="Use when you need multiple consistent viewpoints (front/side/back).",
    )
    def vision_multi_view_image(
        prompt: str,
        reference_image_artifact: Optional[Dict[str, Any]] = None,
        reference_image_b64: Optional[str] = None,
        angles: Optional[List[str]] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        reg.require_support(model_id, "multi_view_image")
        ref_bytes = _resolve_input_bytes(
            store=store,
            artifact=reference_image_artifact,
            b64=reference_image_b64,
            name="reference_image",
            required=False,
        )
        kwargs: Dict[str, Any] = {}
        if ref_bytes is not None:
            kwargs["reference_image"] = ref_bytes
        if angles is not None:
            kwargs["angles"] = angles
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        resolved_steps = steps if steps is not None else _task_param_value("multi_view_image", "steps", 10)
        resolved_guidance = (
            guidance_scale
            if guidance_scale is not None
            else _task_param_value("multi_view_image", "guidance_scale", None)
        )
        if resolved_steps is not None:
            kwargs["steps"] = resolved_steps
        if resolved_guidance is not None:
            kwargs["guidance_scale"] = resolved_guidance
        if seed is not None:
            kwargs["seed"] = seed

        out = vision_manager.generate_angles(prompt, **kwargs)
        if not (isinstance(out, list) and all(isinstance(x, dict) and is_artifact_ref(x) for x in out)):
            raise AbstractVisionError("vision_multi_view_image expected a list of artifact refs; ensure VisionManager.store is set.")
        return out

    @tool(
        name="vision_image_upscale",
        description="Upscale or restore an input image and return an artifact ref.",
        tags=["vision", "upscale", "image"],
        when_to_use="Use when you need to increase image resolution without changing the image composition.",
    )
    def vision_image_upscale(
        image_artifact: Optional[Dict[str, Any]] = None,
        image_b64: Optional[str] = None,
        scale: Optional[float] = None,
        resolution: Optional[str] = None,
        softness: Optional[float] = None,
        seed: Optional[int] = None,
        quantize: Optional[int] = None,
        vae_tiling: Optional[bool] = None,
    ) -> Dict[str, Any]:
        reg.require_support(model_id, "image_upscale")
        image_bytes = _resolve_input_bytes(
            store=store,
            artifact=image_artifact,
            b64=image_b64,
            name="image",
            required=True,
        )
        out = vision_manager.upscale_image(
            image_bytes or b"",
            scale=scale if scale is not None else _task_param_value("image_upscale", "scale", None),
            resolution=(
                resolution
                if resolution is not None
                else _task_param_value("image_upscale", "resolution", None)
            ),
            softness=(
                softness
                if softness is not None
                else _task_param_value("image_upscale", "softness", None)
            ),
            seed=seed,
            quantize=(
                quantize
                if quantize is not None
                else _task_param_value("image_upscale", "quantize", None)
            ),
            vae_tiling=(
                vae_tiling
                if vae_tiling is not None
                else _task_param_value("image_upscale", "vae_tiling", None)
            ),
        )
        if not (isinstance(out, dict) and is_artifact_ref(out)):
            raise AbstractVisionError("vision_image_upscale expected artifact-ref output; ensure VisionManager.store is set.")
        return out

    @tool(
        name="vision_text_to_video",
        description="Generate a video from a text prompt and return an artifact ref.",
        tags=["vision", "generate", "video"],
        when_to_use="Use when you need to create a short video from a prompt.",
    )
    def vision_text_to_video(
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        num_frames: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        guidance_2: Optional[float] = None,
        flow_shift: Optional[float] = None,
        seed: Optional[int] = None,
        lora_adapters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        reg.require_support(model_id, "text_to_video")
        out = vision_manager.generate_video(
            prompt,
            negative_prompt=negative_prompt,
            width=width if width is not None else _task_param_value("text_to_video", "width", None),
            height=height if height is not None else _task_param_value("text_to_video", "height", None),
            fps=fps if fps is not None else _task_param_value("text_to_video", "fps", None),
            num_frames=(
                num_frames if num_frames is not None else _task_param_value("text_to_video", "num_frames", None)
            ),
            steps=steps if steps is not None else _task_param_value("text_to_video", "steps", 10),
            guidance_scale=(
                guidance_scale
                if guidance_scale is not None
                else _task_param_value("text_to_video", "guidance_scale", None)
            ),
            guidance_2=(
                guidance_2
                if guidance_2 is not None
                else _task_param_value("text_to_video", "guidance_2", None)
            ),
            flow_shift=(
                flow_shift
                if flow_shift is not None
                else _task_param_value("text_to_video", "flow_shift", None)
            ),
            seed=seed,
            lora_adapters=_coerce_lora_adapters(lora_adapters),
        )
        return _require_artifact_ref_output("vision_text_to_video", out)

    @tool(
        name="vision_text_to_video_batch",
        description="Generate multiple videos from a text prompt and return artifact refs.",
        tags=["vision", "generate", "video", "batch"],
        when_to_use="Use when you need several text-to-video generations in one call, optionally with explicit seeds and a LoRA stack.",
    )
    def vision_text_to_video_batch(
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        num_frames: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        guidance_2: Optional[float] = None,
        flow_shift: Optional[float] = None,
        seed: Optional[int] = None,
        count: int = 1,
        seeds: Optional[List[int]] = None,
        lora_adapters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        reg.require_support(model_id, "text_to_video")
        out = vision_manager.generate_video_batch(
            prompt,
            negative_prompt=negative_prompt,
            width=width if width is not None else _task_param_value("text_to_video", "width", None),
            height=height if height is not None else _task_param_value("text_to_video", "height", None),
            fps=fps if fps is not None else _task_param_value("text_to_video", "fps", None),
            num_frames=(
                num_frames if num_frames is not None else _task_param_value("text_to_video", "num_frames", None)
            ),
            steps=steps if steps is not None else _task_param_value("text_to_video", "steps", 10),
            guidance_scale=(
                guidance_scale
                if guidance_scale is not None
                else _task_param_value("text_to_video", "guidance_scale", None)
            ),
            guidance_2=(
                guidance_2
                if guidance_2 is not None
                else _task_param_value("text_to_video", "guidance_2", None)
            ),
            flow_shift=(
                flow_shift
                if flow_shift is not None
                else _task_param_value("text_to_video", "flow_shift", None)
            ),
            seed=seed,
            count=int(count),
            seeds=seeds,
            lora_adapters=_coerce_lora_adapters(lora_adapters),
        )
        return _require_artifact_ref_list("vision_text_to_video_batch", out)

    @tool(
        name="vision_image_to_video",
        description="Generate a video conditioned on an input image and return an artifact ref.",
        tags=["vision", "generate", "video"],
        when_to_use="Use when you need to animate an image into a video (optionally guided by a prompt).",
    )
    def vision_image_to_video(
        image_artifact: Optional[Dict[str, Any]] = None,
        image_b64: Optional[str] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        num_frames: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        guidance_2: Optional[float] = None,
        flow_shift: Optional[float] = None,
        seed: Optional[int] = None,
        lora_adapters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        reg.require_support(model_id, "image_to_video")
        image_bytes = _resolve_input_bytes(store=store, artifact=image_artifact, b64=image_b64, name="image", required=True)
        out = vision_manager.image_to_video(
            image=image_bytes or b"",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width if width is not None else _task_param_value("image_to_video", "width", None),
            height=height if height is not None else _task_param_value("image_to_video", "height", None),
            fps=fps if fps is not None else _task_param_value("image_to_video", "fps", None),
            num_frames=(
                num_frames if num_frames is not None else _task_param_value("image_to_video", "num_frames", None)
            ),
            steps=steps if steps is not None else _task_param_value("image_to_video", "steps", 10),
            guidance_scale=(
                guidance_scale
                if guidance_scale is not None
                else _task_param_value("image_to_video", "guidance_scale", None)
            ),
            guidance_2=(
                guidance_2
                if guidance_2 is not None
                else _task_param_value("image_to_video", "guidance_2", None)
            ),
            flow_shift=(
                flow_shift
                if flow_shift is not None
                else _task_param_value("image_to_video", "flow_shift", None)
            ),
            seed=seed,
            lora_adapters=_coerce_lora_adapters(lora_adapters),
        )
        return _require_artifact_ref_output("vision_image_to_video", out)

    @tool(
        name="vision_image_to_video_batch",
        description="Generate multiple videos from one input image and return artifact refs.",
        tags=["vision", "generate", "video", "batch"],
        when_to_use="Use when you need several image-to-video generations in one call, optionally with explicit seeds and a LoRA stack.",
    )
    def vision_image_to_video_batch(
        image_artifact: Optional[Dict[str, Any]] = None,
        image_b64: Optional[str] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        num_frames: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        guidance_2: Optional[float] = None,
        flow_shift: Optional[float] = None,
        seed: Optional[int] = None,
        count: int = 1,
        seeds: Optional[List[int]] = None,
        lora_adapters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        reg.require_support(model_id, "image_to_video")
        image_bytes = _resolve_input_bytes(store=store, artifact=image_artifact, b64=image_b64, name="image", required=True)
        out = vision_manager.image_to_video_batch(
            image=image_bytes or b"",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width if width is not None else _task_param_value("image_to_video", "width", None),
            height=height if height is not None else _task_param_value("image_to_video", "height", None),
            fps=fps if fps is not None else _task_param_value("image_to_video", "fps", None),
            num_frames=(
                num_frames if num_frames is not None else _task_param_value("image_to_video", "num_frames", None)
            ),
            steps=steps if steps is not None else _task_param_value("image_to_video", "steps", 10),
            guidance_scale=(
                guidance_scale
                if guidance_scale is not None
                else _task_param_value("image_to_video", "guidance_scale", None)
            ),
            guidance_2=(
                guidance_2
                if guidance_2 is not None
                else _task_param_value("image_to_video", "guidance_2", None)
            ),
            flow_shift=(
                flow_shift
                if flow_shift is not None
                else _task_param_value("image_to_video", "flow_shift", None)
            ),
            seed=seed,
            count=int(count),
            seeds=seeds,
            lora_adapters=_coerce_lora_adapters(lora_adapters),
        )
        return _require_artifact_ref_list("vision_image_to_video_batch", out)

    return [
        vision_list_adapters,
        vision_text_to_image,
        vision_text_to_image_batch,
        vision_image_to_image,
        vision_image_to_image_batch,
        vision_multi_view_image,
        vision_image_upscale,
        vision_text_to_video,
        vision_text_to_video_batch,
        vision_image_to_video,
        vision_image_to_video_batch,
    ]
