from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.request import Request, urlopen

from ..errors import CapabilityNotSupportedError
from ..types import (
    GeneratedAsset,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageToVideoRequest,
    VideoGenerationRequest,
    VisionBackendCapabilities,
)
from .base_backend import VisionBackend


def _join_url(base_url: str, path: str) -> str:
    b = str(base_url or "").rstrip("/")
    p = str(path or "").strip()
    if not p:
        return b
    if not p.startswith("/"):
        p = "/" + p
    return b + p


def _sniff_mime_type(content: bytes, fallback: str) -> str:
    b = bytes(content or b"")
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if b.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(b) >= 12 and b[4:8] == b"ftyp":
        return "video/mp4"
    return str(fallback or "application/octet-stream")


def _decode_b64(s: str) -> bytes:
    raw = str(s or "").strip()
    raw = "".join(raw.split())
    pad = (-len(raw)) % 4
    if pad:
        raw = raw + ("=" * pad)
    return base64.b64decode(raw, validate=False)


def _first_data_item(resp: Dict[str, Any]) -> Dict[str, Any]:
    data = resp.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return {}


def _multipart_form(
    *,
    fields: Dict[str, str],
    files: Dict[str, Tuple[str, bytes, str]],
) -> Tuple[bytes, str]:
    boundary = f"----abstractvision-{uuid.uuid4().hex}"
    parts: list[bytes] = []

    def _add(b: bytes) -> None:
        parts.append(b)

    for name, value in fields.items():
        _add(f"--{boundary}\r\n".encode("utf-8"))
        _add(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        _add(str(value).encode("utf-8"))
        _add(b"\r\n")

    for name, (filename, content, content_type) in files.items():
        _add(f"--{boundary}\r\n".encode("utf-8"))
        _add(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(
                "utf-8"
            )
        )
        _add(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        _add(bytes(content))
        _add(b"\r\n")

    _add(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    return body, boundary


@dataclass
class OpenAICompatibleBackendConfig:
    base_url: str
    api_key: Optional[str] = None
    model_id: Optional[str] = None
    timeout_s: float = 300.0

    # Endpoints (OpenAI-shaped HTTP).
    image_generations_path: str = "/images/generations"
    image_edits_path: str = "/images/edits"
    text_to_video_path: Optional[str] = None
    image_to_video_path: Optional[str] = None

    # Image-to-video request mode when enabled.
    image_to_video_mode: str = "multipart"  # "multipart" | "json_b64"


class OpenAICompatibleVisionBackend(VisionBackend):
    """Backend adapter for OpenAI-compatible endpoints (OpenAI-shaped HTTP).

    Notes:
    - Image endpoints are widely implemented (`/images/generations`, `/images/edits`).
    - Video endpoints are not standardized; they are optional and must be configured explicitly.
    """

    def __init__(self, *, config: OpenAICompatibleBackendConfig):
        self._cfg = config

    def get_capabilities(self) -> VisionBackendCapabilities:
        tasks = {"text_to_image", "image_to_image"}
        if self._cfg.text_to_video_path:
            tasks.add("text_to_video")
        if self._cfg.image_to_video_path:
            tasks.add("image_to_video")
        return VisionBackendCapabilities(
            supported_tasks=sorted(tasks),
            supports_mask=True,
        )

    def _headers(self, *, content_type: str) -> Dict[str, str]:
        headers = {"Content-Type": str(content_type)}
        if self._cfg.api_key:
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"
        return headers

    def _post_json(self, *, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = _join_url(self._cfg.base_url, path)
        body = json.dumps(payload).encode("utf-8")
        req = Request(url=url, data=body, method="POST", headers=self._headers(content_type="application/json"))
        with urlopen(req, timeout=float(self._cfg.timeout_s)) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid response: expected JSON object")
        return data

    def _post_multipart(self, *, path: str, fields: Dict[str, str], files: Dict[str, Tuple[str, bytes, str]]) -> Dict[str, Any]:
        url = _join_url(self._cfg.base_url, path)
        body, boundary = _multipart_form(fields=fields, files=files)
        ctype = f"multipart/form-data; boundary={boundary}"
        req = Request(url=url, data=body, method="POST", headers=self._headers(content_type=ctype))
        with urlopen(req, timeout=float(self._cfg.timeout_s)) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid response: expected JSON object")
        return data

    def _parse_media(self, resp: Dict[str, Any], *, fallback_mime: str) -> GeneratedAsset:
        item = _first_data_item(resp)
        if "b64_json" in item:
            content = _decode_b64(str(item.get("b64_json") or ""))
            mime = _sniff_mime_type(content, fallback_mime)
            media_type = "video" if mime.startswith("video/") else "image"
            return GeneratedAsset(media_type=media_type, data=content, mime_type=mime, metadata={"source": "b64_json"})
        if "url" in item and isinstance(item.get("url"), str):
            # Best-effort: download bytes.
            u = str(item.get("url"))
            req = Request(url=u, method="GET")
            with urlopen(req, timeout=float(self._cfg.timeout_s)) as resp2:
                content = resp2.read()
                ct = resp2.headers.get("Content-Type") or fallback_mime
            mime = _sniff_mime_type(content, str(ct))
            media_type = "video" if mime.startswith("video/") else "image"
            return GeneratedAsset(media_type=media_type, data=content, mime_type=mime, metadata={"source": "url", "url": u})
        raise ValueError("Invalid response: missing data[0].b64_json or data[0].url")

    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
        payload: Dict[str, Any] = {
            "prompt": request.prompt,
            "response_format": "b64_json",
            "n": 1,
        }
        if self._cfg.model_id:
            payload["model"] = self._cfg.model_id
        if request.negative_prompt is not None:
            payload["negative_prompt"] = request.negative_prompt
        if request.width is not None and request.height is not None:
            payload["size"] = f"{int(request.width)}x{int(request.height)}"
            payload["width"] = int(request.width)
            payload["height"] = int(request.height)
        if request.seed is not None:
            payload["seed"] = int(request.seed)
        if request.steps is not None:
            payload["steps"] = int(request.steps)
        if request.guidance_scale is not None:
            payload["guidance_scale"] = float(request.guidance_scale)
        if isinstance(request.extra, dict) and request.extra:
            payload.update(dict(request.extra))

        resp = self._post_json(path=self._cfg.image_generations_path, payload=payload)
        return self._parse_media(resp, fallback_mime="image/png")

    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
        # OpenAI-style image edits use multipart form data.
        fields: Dict[str, str] = {"prompt": request.prompt}
        if self._cfg.model_id:
            fields["model"] = self._cfg.model_id
        if request.negative_prompt is not None:
            fields["negative_prompt"] = request.negative_prompt

        files: Dict[str, Tuple[str, bytes, str]] = {
            "image": ("image.png", bytes(request.image), "image/png"),
        }
        if request.mask is not None:
            files["mask"] = ("mask.png", bytes(request.mask), "image/png")

        # Best-effort extra fields.
        if request.seed is not None:
            fields["seed"] = str(int(request.seed))
        if request.steps is not None:
            fields["steps"] = str(int(request.steps))
        if request.guidance_scale is not None:
            fields["guidance_scale"] = str(float(request.guidance_scale))
        if isinstance(request.extra, dict) and request.extra:
            for k, v in request.extra.items():
                if v is None:
                    continue
                fields[str(k)] = str(v)

        resp = self._post_multipart(path=self._cfg.image_edits_path, fields=fields, files=files)
        return self._parse_media(resp, fallback_mime="image/png")

    def generate_angles(self, request) -> list[GeneratedAsset]:
        raise CapabilityNotSupportedError("OpenAICompatibleVisionBackend does not implement multi-view generation.")

    def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
        if not self._cfg.text_to_video_path:
            raise CapabilityNotSupportedError("text_to_video is not configured for this backend.")
        payload: Dict[str, Any] = {"prompt": request.prompt, "response_format": "b64_json", "n": 1}
        if self._cfg.model_id:
            payload["model"] = self._cfg.model_id
        if request.negative_prompt is not None:
            payload["negative_prompt"] = request.negative_prompt
        if request.width is not None:
            payload["width"] = int(request.width)
        if request.height is not None:
            payload["height"] = int(request.height)
        if request.fps is not None:
            payload["fps"] = int(request.fps)
        if request.num_frames is not None:
            payload["num_frames"] = int(request.num_frames)
        if request.seed is not None:
            payload["seed"] = int(request.seed)
        if request.steps is not None:
            payload["steps"] = int(request.steps)
        if request.guidance_scale is not None:
            payload["guidance_scale"] = float(request.guidance_scale)
        if isinstance(request.extra, dict) and request.extra:
            payload.update(dict(request.extra))
        resp = self._post_json(path=str(self._cfg.text_to_video_path), payload=payload)
        return self._parse_media(resp, fallback_mime="video/mp4")

    def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
        if not self._cfg.image_to_video_path:
            raise CapabilityNotSupportedError("image_to_video is not configured for this backend.")

        if str(self._cfg.image_to_video_mode) == "json_b64":
            payload: Dict[str, Any] = {"image_b64": base64.b64encode(bytes(request.image)).decode("ascii")}
            if self._cfg.model_id:
                payload["model"] = self._cfg.model_id
            if request.prompt is not None:
                payload["prompt"] = request.prompt
            if request.negative_prompt is not None:
                payload["negative_prompt"] = request.negative_prompt
            if request.width is not None:
                payload["width"] = int(request.width)
            if request.height is not None:
                payload["height"] = int(request.height)
            if request.fps is not None:
                payload["fps"] = int(request.fps)
            if request.num_frames is not None:
                payload["num_frames"] = int(request.num_frames)
            if request.seed is not None:
                payload["seed"] = int(request.seed)
            if request.steps is not None:
                payload["steps"] = int(request.steps)
            if request.guidance_scale is not None:
                payload["guidance_scale"] = float(request.guidance_scale)
            if isinstance(request.extra, dict) and request.extra:
                payload.update(dict(request.extra))
            resp = self._post_json(path=str(self._cfg.image_to_video_path), payload=payload)
            return self._parse_media(resp, fallback_mime="video/mp4")

        fields: Dict[str, str] = {}
        if self._cfg.model_id:
            fields["model"] = self._cfg.model_id
        if request.prompt is not None:
            fields["prompt"] = request.prompt
        if request.negative_prompt is not None:
            fields["negative_prompt"] = request.negative_prompt
        if request.width is not None:
            fields["width"] = str(int(request.width))
        if request.height is not None:
            fields["height"] = str(int(request.height))
        if request.fps is not None:
            fields["fps"] = str(int(request.fps))
        if request.num_frames is not None:
            fields["num_frames"] = str(int(request.num_frames))
        if request.seed is not None:
            fields["seed"] = str(int(request.seed))
        if request.steps is not None:
            fields["steps"] = str(int(request.steps))
        if request.guidance_scale is not None:
            fields["guidance_scale"] = str(float(request.guidance_scale))
        if isinstance(request.extra, dict) and request.extra:
            for k, v in request.extra.items():
                if v is None:
                    continue
                fields[str(k)] = str(v)

        files = {"image": ("image.png", bytes(request.image), "image/png")}
        resp = self._post_multipart(path=str(self._cfg.image_to_video_path), fields=fields, files=files)
        return self._parse_media(resp, fallback_mime="video/mp4")
