from __future__ import annotations

import base64
from typing import Any, Callable, Dict, List, Optional

from ..artifacts import MediaStore, get_artifact_id, is_artifact_ref
from ..errors import AbstractVisionError, OptionalDependencyMissingError
from ..model_capabilities import VisionModelCapabilitiesRegistry
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
        steps: Optional[int] = 10,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        reg.require_support(model_id, "text_to_image")
        out = vision_manager.generate_image(
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        if not (isinstance(out, dict) and is_artifact_ref(out)):
            raise AbstractVisionError("vision_text_to_image expected artifact-ref output; ensure VisionManager.store is set.")
        return out

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
        steps: Optional[int] = 10,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        reg.require_support(model_id, "image_to_image")
        image_bytes = _resolve_input_bytes(store=store, artifact=image_artifact, b64=image_b64, name="image", required=True)
        mask_bytes = _resolve_input_bytes(store=store, artifact=mask_artifact, b64=mask_b64, name="mask", required=False)
        out = vision_manager.edit_image(
            prompt,
            image=image_bytes or b"",
            mask=mask_bytes,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        if not (isinstance(out, dict) and is_artifact_ref(out)):
            raise AbstractVisionError("vision_image_to_image expected artifact-ref output; ensure VisionManager.store is set.")
        return out

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
        steps: Optional[int] = 10,
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
        if steps is not None:
            kwargs["steps"] = steps
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        if seed is not None:
            kwargs["seed"] = seed

        out = vision_manager.generate_angles(prompt, **kwargs)
        if not (isinstance(out, list) and all(isinstance(x, dict) and is_artifact_ref(x) for x in out)):
            raise AbstractVisionError("vision_multi_view_image expected a list of artifact refs; ensure VisionManager.store is set.")
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
        steps: Optional[int] = 10,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        reg.require_support(model_id, "text_to_video")
        out = vision_manager.generate_video(
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            fps=fps,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        if not (isinstance(out, dict) and is_artifact_ref(out)):
            raise AbstractVisionError("vision_text_to_video expected artifact-ref output; ensure VisionManager.store is set.")
        return out

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
        steps: Optional[int] = 10,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        reg.require_support(model_id, "image_to_video")
        image_bytes = _resolve_input_bytes(store=store, artifact=image_artifact, b64=image_b64, name="image", required=True)
        out = vision_manager.image_to_video(
            image=image_bytes or b"",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            fps=fps,
            num_frames=num_frames,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        if not (isinstance(out, dict) and is_artifact_ref(out)):
            raise AbstractVisionError("vision_image_to_video expected artifact-ref output; ensure VisionManager.store is set.")
        return out

    return [
        vision_text_to_image,
        vision_image_to_image,
        vision_multi_view_image,
        vision_text_to_video,
        vision_image_to_video,
    ]
