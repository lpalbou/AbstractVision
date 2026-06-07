from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..errors import CapabilityNotSupportedError
from ..types import (
    GeneratedAsset,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageToVideoRequest,
    ProviderModelInfo,
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


def _format_http_error(e: HTTPError) -> str:
    try:
        body = e.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    body = body.strip()
    if len(body) > 1000:
        body = body[:1000] + "..."
    suffix = f": {body}" if body else ""
    return f"OpenAI-compatible provider request failed (status={getattr(e, 'code', 'unknown')}){suffix}"


def _model_family(model_id: Optional[str]) -> Optional[str]:
    model = str(model_id or "").strip().lower()
    if "/" in model:
        model = model.rsplit("/", 1)[-1].strip()
    if model.startswith("gpt-image-"):
        return "gpt-image"
    if model.startswith("dall-e-"):
        return "dall-e"
    return None


def _upstream_model_id(model_id: Optional[str]) -> str:
    model = str(model_id or "").strip()
    while "/" in model:
        provider, rest = model.split("/", 1)
        if provider.strip().lower() in {"openai", "openai-compatible", "openai_compatible"}:
            model = rest.strip()
            continue
        break
    return model


def _looks_like_openai_api(base_url: str) -> bool:
    return "api.openai.com" in str(base_url or "").lower()


def _normalise_catalog_token(value: str) -> str:
    out = str(value or "").strip().lower()
    for ch in ("-", "/", ".", " "):
        out = out.replace(ch, "_")
    return out


def _collect_catalog_tokens(value: Any) -> Set[str]:
    tokens: Set[str] = set()
    if value is None:
        return tokens
    if isinstance(value, str):
        token = _normalise_catalog_token(value)
        if token:
            tokens.add(token)
        return tokens
    if isinstance(value, (int, float, bool)):
        return tokens
    if isinstance(value, dict):
        for k, v in value.items():
            tokens.update(_collect_catalog_tokens(k))
            tokens.update(_collect_catalog_tokens(v))
        return tokens
    if isinstance(value, (list, tuple, set)):
        for v in value:
            tokens.update(_collect_catalog_tokens(v))
    return tokens


def _catalog_capability_tokens(raw: Dict[str, Any]) -> Set[str]:
    tokens: Set[str] = set()
    for key in (
        "capabilities",
        "modalities",
        "tasks",
        "supported_tasks",
        "features",
        "endpoints",
        "permissions",
    ):
        if key in raw:
            tokens.update(_collect_catalog_tokens(raw.get(key)))
    return tokens


def _task_catalog_aliases(task: Optional[str]) -> Set[str]:
    t = _normalise_catalog_token(str(task or ""))
    if not t:
        return set()
    aliases = {t}
    if t in {"text_to_image", "image_generation", "image_generations", "image"}:
        aliases.update(
            {
                "image",
                "images",
                "image_generation",
                "image_generations",
                "images_generations",
                "text_to_image",
                "t2i",
            }
        )
    if t in {"image_to_image", "image_edit", "image_edits", "inpaint", "image"}:
        aliases.update(
            {
                "image",
                "images",
                "image_edit",
                "image_edits",
                "images_edits",
                "image_to_image",
                "i2i",
                "inpaint",
            }
        )
    if t in {"text_to_video", "video_generation", "video_generations", "video"}:
        aliases.update(
            {
                "video",
                "videos",
                "video_generation",
                "video_generations",
                "videos_generations",
                "text_to_video",
                "t2v",
            }
        )
    if t in {"image_to_video", "video_edit", "video_edits", "video"}:
        aliases.update(
            {
                "video",
                "videos",
                "video_edit",
                "video_edits",
                "videos_edits",
                "image_to_video",
                "i2v",
            }
        )
    return aliases


def _catalog_entry_matches_task(
    info: ProviderModelInfo, *, task: Optional[str], openai_api: bool
) -> bool:
    aliases = _task_catalog_aliases(task)
    if not aliases:
        return True

    tokens = set(info.capabilities)
    if tokens:
        return bool(tokens.intersection(aliases))

    image_aliases = {
        "image",
        "images",
        "image_generation",
        "image_generations",
        "images_generations",
        "text_to_image",
        "t2i",
        "image_edit",
        "image_edits",
        "images_edits",
        "image_to_image",
        "i2i",
        "inpaint",
    }
    if aliases.intersection(image_aliases):
        if _model_family(info.id) in {"gpt-image", "dall-e"}:
            return True
        if openai_api:
            return False

    # OpenAI-compatible catalogs do not have a universal capability schema.
    # When a non-OpenAI provider omits capability metadata, expose the entry
    # instead of guessing it is incompatible.
    return not openai_api


def _size_value(width: Optional[int], height: Optional[int]) -> Optional[str]:
    if width is None or height is None:
        return None
    return f"{int(width)}x{int(height)}"


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
    models_path: str = "/models"


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

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        resp = self._get_json(path=str(self._cfg.models_path or "/models"))
        data = resp.get("data")
        if not isinstance(data, list):
            raise ValueError("Invalid response: expected JSON object with a data list")

        openai_api = _looks_like_openai_api(self._cfg.base_url)
        provider_name = "openai" if openai_api else "openai-compatible"
        models: List[ProviderModelInfo] = []
        for item in data:
            if isinstance(item, str):
                raw: Dict[str, Any] = {"id": item}
            elif isinstance(item, dict):
                raw = dict(item)
            else:
                continue
            model_id = raw.get("id") or raw.get("model") or raw.get("name")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            raw.setdefault("provider", provider_name)
            raw.setdefault("backend", "openai-compatible")
            raw.setdefault("routed_model", f"{provider_name}/{str(model_id).strip()}")
            created = raw.get("created")
            info = ProviderModelInfo(
                id=str(model_id).strip(),
                object=str(raw.get("object")) if raw.get("object") is not None else None,
                created=int(created) if isinstance(created, int) else None,
                owned_by=str(raw.get("owned_by")) if raw.get("owned_by") is not None else provider_name,
                capabilities=tuple(sorted(_catalog_capability_tokens(raw))),
                raw=raw,
            )
            if _catalog_entry_matches_task(info, task=task, openai_api=openai_api):
                models.append(info)
        return models

    def _headers(self, *, content_type: Optional[str] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = str(content_type)
        if self._cfg.api_key:
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"
        return headers

    def _get_json(self, *, path: str) -> Dict[str, Any]:
        url = _join_url(self._cfg.base_url, path)
        req = Request(url=url, method="GET", headers=self._headers())
        try:
            with urlopen(req, timeout=float(self._cfg.timeout_s)) as resp:
                raw = resp.read()
        except HTTPError as e:
            raise RuntimeError(_format_http_error(e)) from e
        except URLError as e:
            raise RuntimeError(f"OpenAI-compatible provider request failed: {e}") from e
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid response: expected JSON object")
        return data

    def _post_json(self, *, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = _join_url(self._cfg.base_url, path)
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            url=url,
            data=body,
            method="POST",
            headers=self._headers(content_type="application/json"),
        )
        try:
            with urlopen(req, timeout=float(self._cfg.timeout_s)) as resp:
                raw = resp.read()
        except HTTPError as e:
            raise RuntimeError(_format_http_error(e)) from e
        except URLError as e:
            raise RuntimeError(f"OpenAI-compatible provider request failed: {e}") from e
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid response: expected JSON object")
        return data

    def _post_multipart(
        self, *, path: str, fields: Dict[str, str], files: Dict[str, Tuple[str, bytes, str]]
    ) -> Dict[str, Any]:
        url = _join_url(self._cfg.base_url, path)
        body, boundary = _multipart_form(fields=fields, files=files)
        ctype = f"multipart/form-data; boundary={boundary}"
        req = Request(url=url, data=body, method="POST", headers=self._headers(content_type=ctype))
        try:
            with urlopen(req, timeout=float(self._cfg.timeout_s)) as resp:
                raw = resp.read()
        except HTTPError as e:
            raise RuntimeError(_format_http_error(e)) from e
        except URLError as e:
            raise RuntimeError(f"OpenAI-compatible provider request failed: {e}") from e
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
            return GeneratedAsset(
                media_type=media_type, data=content, mime_type=mime, metadata={"source": "b64_json"}
            )
        if "url" in item and isinstance(item.get("url"), str):
            # Best-effort: download bytes.
            u = str(item.get("url"))
            req = Request(url=u, method="GET")
            try:
                with urlopen(req, timeout=float(self._cfg.timeout_s)) as resp2:
                    content = resp2.read()
                    ct = resp2.headers.get("Content-Type") or fallback_mime
            except HTTPError as e:
                raise RuntimeError(_format_http_error(e)) from e
            except URLError as e:
                raise RuntimeError(f"OpenAI-compatible media download failed: {e}") from e
            mime = _sniff_mime_type(content, str(ct))
            media_type = "video" if mime.startswith("video/") else "image"
            return GeneratedAsset(
                media_type=media_type,
                data=content,
                mime_type=mime,
                metadata={"source": "url", "url": u},
            )
        raise ValueError("Invalid response: missing data[0].b64_json or data[0].url")

    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
        family = _model_family(self._cfg.model_id)
        openai_api = _looks_like_openai_api(self._cfg.base_url) or family is not None
        payload: Dict[str, Any] = {
            "prompt": request.prompt,
            "n": 1,
        }
        if self._cfg.model_id:
            payload["model"] = _upstream_model_id(self._cfg.model_id)

        # Real OpenAI image models have a narrower schema than many local
        # OpenAI-compatible servers. Keep local-compatible fields for unknown
        # models, but avoid sending unsupported fields to known OpenAI models.
        if family != "gpt-image":
            payload["response_format"] = "b64_json"
        if request.negative_prompt is not None and not openai_api:
            payload["negative_prompt"] = request.negative_prompt
        size = _size_value(request.width, request.height)
        if size:
            payload["size"] = size
        if size and not openai_api:
            payload["width"] = int(request.width)
            payload["height"] = int(request.height)
        if request.seed is not None and not openai_api:
            payload["seed"] = int(request.seed)
        if request.steps is not None and not openai_api:
            payload["steps"] = int(request.steps)
        if request.guidance_scale is not None and not openai_api:
            payload["guidance_scale"] = float(request.guidance_scale)
        if isinstance(request.extra, dict) and request.extra:
            payload.update(dict(request.extra))
        if payload.get("model") is not None:
            payload["model"] = _upstream_model_id(payload.get("model"))
        if family == "gpt-image":
            payload.pop("response_format", None)

        resp = self._post_json(path=self._cfg.image_generations_path, payload=payload)
        return self._parse_media(resp, fallback_mime="image/png")

    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
        family = _model_family(self._cfg.model_id)
        openai_api = _looks_like_openai_api(self._cfg.base_url) or family is not None
        # OpenAI-style image edits use multipart form data.
        fields: Dict[str, str] = {"prompt": request.prompt}
        if self._cfg.model_id:
            fields["model"] = _upstream_model_id(self._cfg.model_id)
        if request.negative_prompt is not None and not openai_api:
            fields["negative_prompt"] = request.negative_prompt

        image_field = "image[]" if family == "gpt-image" else "image"
        files: Dict[str, Tuple[str, bytes, str]] = {
            image_field: ("image.png", bytes(request.image), "image/png")
        }
        if request.mask is not None:
            files["mask"] = ("mask.png", bytes(request.mask), "image/png")

        # Best-effort extra fields.
        if request.seed is not None and not openai_api:
            fields["seed"] = str(int(request.seed))
        if request.steps is not None and not openai_api:
            fields["steps"] = str(int(request.steps))
        if request.guidance_scale is not None and not openai_api:
            fields["guidance_scale"] = str(float(request.guidance_scale))
        if isinstance(request.extra, dict) and request.extra:
            for k, v in request.extra.items():
                if v is None:
                    continue
                fields[str(k)] = str(v)
        if fields.get("model") is not None:
            fields["model"] = _upstream_model_id(fields.get("model"))

        resp = self._post_multipart(path=self._cfg.image_edits_path, fields=fields, files=files)
        return self._parse_media(resp, fallback_mime="image/png")

    def generate_angles(self, request) -> list[GeneratedAsset]:
        raise CapabilityNotSupportedError(
            "OpenAICompatibleVisionBackend does not implement multi-view generation."
        )

    def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
        if not self._cfg.text_to_video_path:
            raise CapabilityNotSupportedError("text_to_video is not configured for this backend.")
        payload: Dict[str, Any] = {"prompt": request.prompt, "response_format": "b64_json", "n": 1}
        if self._cfg.model_id:
            payload["model"] = _upstream_model_id(self._cfg.model_id)
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
        if request.guidance_2 is not None:
            payload["guidance_2"] = float(request.guidance_2)
        if isinstance(request.extra, dict) and request.extra:
            payload.update(dict(request.extra))
        if payload.get("model") is not None:
            payload["model"] = _upstream_model_id(payload.get("model"))
        resp = self._post_json(path=str(self._cfg.text_to_video_path), payload=payload)
        return self._parse_media(resp, fallback_mime="video/mp4")

    def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
        if not self._cfg.image_to_video_path:
            raise CapabilityNotSupportedError("image_to_video is not configured for this backend.")

        if str(self._cfg.image_to_video_mode) == "json_b64":
            payload: Dict[str, Any] = {
                "image_b64": base64.b64encode(bytes(request.image)).decode("ascii")
            }
            if self._cfg.model_id:
                payload["model"] = _upstream_model_id(self._cfg.model_id)
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
            if request.guidance_2 is not None:
                payload["guidance_2"] = float(request.guidance_2)
            if isinstance(request.extra, dict) and request.extra:
                payload.update(dict(request.extra))
            if payload.get("model") is not None:
                payload["model"] = _upstream_model_id(payload.get("model"))
            resp = self._post_json(path=str(self._cfg.image_to_video_path), payload=payload)
            return self._parse_media(resp, fallback_mime="video/mp4")

        fields: Dict[str, str] = {}
        if self._cfg.model_id:
            fields["model"] = _upstream_model_id(self._cfg.model_id)
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
        if request.guidance_2 is not None:
            fields["guidance_2"] = str(float(request.guidance_2))
        if isinstance(request.extra, dict) and request.extra:
            for k, v in request.extra.items():
                if v is None:
                    continue
                fields[str(k)] = str(v)
        if fields.get("model") is not None:
            fields["model"] = _upstream_model_id(fields.get("model"))

        files = {"image": ("image.png", bytes(request.image), "image/png")}
        resp = self._post_multipart(
            path=str(self._cfg.image_to_video_path), fields=fields, files=files
        )
        return self._parse_media(resp, fallback_mime="video/mp4")
