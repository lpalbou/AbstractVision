from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..errors import CapabilityNotSupportedError, GenerationCancelledError
from ..types import (
    MEMORY_MODE_DEFAULT,
    MEMORY_MODE_ENV_VAR,
    GeneratedAsset,
    GeneratedAudioTrack,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageToVideoRequest,
    ProviderModelInfo,
    VideoGenerationRequest,
    VisionBackendCapabilities,
)
from .base_backend import OPTIONAL_REQUEST_FIELDS, VisionBackend, _option_is_set

_INTERNAL_PROGRESS_EXTRA_KEYS = {
    "on_progress",
    "progress_event_callback",
    "progress_callback",
    "_step_progress_callback",
    "__abstractvision_progress_callback",
}


# AbstractCore's vision routes (relative to base_url). A remote whose GET /vision/models lists
# `memory_modes` for the configured model takes `memory_mode`, `quantize` and the soundtrack fields;
# without that route the remote is a plain OpenAI image endpoint and gets none of them.
_VISION_MODELS_PATH = "/vision/models"
_VISION_JOBS_PATH = "/vision/jobs"
_AUDIO_FIELDS = ("soundscape", "music", "generate_audio", "audio_shift")
_VIDEO_TASKS = {"text_to_video", "image_to_video"}
# How long a GET /vision/models answer is reused, the longest it may take, and the job poll period.
_DISCOVERY_TTL_S = 30.0
_DISCOVERY_TIMEOUT_S = 10.0
_JOB_POLL_INTERVAL_S = 0.5


def _join_url(base_url: str, path: str) -> str:
    b = str(base_url or "").rstrip("/")
    p = str(path or "").strip()
    if not p:
        return b
    if not p.startswith("/"):
        p = "/" + p
    return b + p


def _upstream_extra(extra: Any) -> Dict[str, Any]:
    if not isinstance(extra, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in extra.items():
        key_s = str(key)
        if key_s in _INTERNAL_PROGRESS_EXTRA_KEYS:
            continue
        if callable(value):
            continue
        out[key_s] = value
    return out


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


def _format_http_error(e: HTTPError, body: Optional[str] = None) -> str:
    if body is None:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
    body = body.strip()
    if len(body) > 1000:
        body = body[:1000] + "..."
    suffix = f": {body}" if body else ""
    return f"OpenAI-compatible provider request failed (status={getattr(e, 'code', 'unknown')}){suffix}"


def _http_error(e: HTTPError) -> Exception:
    """The exception for an HTTP error; a remote 422 `capability_not_supported` keeps its meaning."""
    try:
        body = e.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    try:
        err = json.loads(body).get("error") if body.strip() else None
    except (ValueError, AttributeError):
        err = None
    if isinstance(err, dict) and err.get("code") == "capability_not_supported":
        return CapabilityNotSupportedError(str(err.get("message") or "The remote refused the request."))
    return RuntimeError(_format_http_error(e, body))


def _form_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _discovery_entry(catalog: Mapping[str, Any], model_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The GET /vision/models entry of `model_id`, or of the remote's active model when None.

    Only an entry that lists `memory_modes` advertises the memory and sound fields.
    """
    wanted = {str(model_id or "").strip(), _upstream_model_id(model_id)} - {""}
    candidates = [catalog.get("active")] + (list(catalog.get("models") or []) if wanted else [])
    for entry in candidates:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("memory_modes"), list):
            continue
        ids = {str(entry.get(key) or "").strip() for key in ("id", "model_id", "load_id", "model")}
        if not wanted or wanted.intersection(ids):
            return dict(entry)
    return None


def _audio_track(audio: Any, download: Any) -> Optional[GeneratedAudioTrack]:
    """The GeneratedAudioTrack a video response's `audio` object describes.

    `audio.wav` ({"b64_json"} or {"url"}) references the soundtrack as WAV when the remote could
    not mux it into the MP4.
    """
    if not isinstance(audio, Mapping) or not audio.get("present"):
        return None
    wav = audio.get("wav") if isinstance(audio.get("wav"), Mapping) else None
    data: Optional[bytes] = None
    if wav is not None and isinstance(wav.get("b64_json"), str):
        data = _decode_b64(str(wav["b64_json"]))
    elif wav is not None and isinstance(wav.get("url"), str):
        data = download(str(wav["url"]))
    muxed = bool(audio["muxed"]) if isinstance(audio.get("muxed"), bool) else wav is None
    return GeneratedAudioTrack(
        sample_rate=int(audio.get("sample_rate") or 0),
        channels=int(audio.get("channels") or 0),
        duration_seconds=float(audio.get("duration_seconds") or 0.0),
        muxed=muxed,
        codec=str(audio["codec"]) if audio.get("codec") else None,
        data=None if muxed else data,
        mime_type=None if muxed or data is None else "audio/wav",
    )


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
    timeout_s: Optional[float] = 300.0

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
        self._catalog_lock = threading.Lock()
        self._catalog: Optional[Dict[str, Any]] = None
        self._catalog_at: Optional[float] = None
        # Why the last GET /vision/models failed (timeout, connection error, 5xx); None after an answer.
        self._discovery_failure: Optional[str] = None

    def get_capabilities(self) -> VisionBackendCapabilities:
        tasks = {"text_to_image", "image_to_image"}
        if self._cfg.text_to_video_path:
            tasks.add("text_to_video")
        if self._cfg.image_to_video_path:
            tasks.add("image_to_video")
        entry = self._remote_entry(refresh=False)
        return VisionBackendCapabilities(
            supported_tasks=sorted(tasks),
            supports_mask=True,
            supports_control_image=False,
            memory_modes=self.supported_memory_modes(),
            # cancel_event is checked before a run starts; on a remote that advertises the vision
            # routes, a set event also cancels the remote job, which its backend honours itself.
            cancellation="start",
            generates_audio=bool(entry.get("supports_audio")) if entry is not None else None,
        )

    def supported_memory_modes(self, task: Optional[str] = None) -> Tuple[str, ...]:
        """The default mode plus the modes the remote lists for the configured model.

        Read from the last GET /vision/models answer (no I/O here, nor in `get_capabilities()`);
        `list_provider_models()` and a request that sets one of the fields refresh it.
        """
        _ = task
        entry = self._remote_entry(refresh=False) or {}
        return tuple(dict.fromkeys([MEMORY_MODE_DEFAULT, *(str(m) for m in entry.get("memory_modes") or ())]))

    def supported_request_options(self, task: Optional[str] = None) -> FrozenSet[str]:
        entry = self._remote_entry(refresh=False)
        if entry is None:
            return frozenset()
        options: Set[str] = {"quantize"}
        if entry.get("supports_audio") and (task is None or str(task) in _VIDEO_TASKS):
            options.update(str(name) for name in entry.get("audio_fields") or () if name in _AUDIO_FIELDS)
        return frozenset(options)

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        catalog = self._remote_catalog(refresh=True)
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
            entry = _discovery_entry(catalog, str(model_id)) if catalog is not None else None
            for key in ("memory_modes", "supports_audio", "audio_fields"):
                if entry is not None and key in entry:
                    raw.setdefault(key, entry[key])
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

    def _control_timeout(self) -> Optional[float]:
        if self._cfg.timeout_s is None:
            return None
        timeout = float(self._cfg.timeout_s)
        return timeout if timeout > 0 else None

    def _generation_timeout(self) -> Optional[float]:
        return None

    def _remote_catalog(self, *, refresh: bool) -> Optional[Dict[str, Any]]:
        """The remote's GET /vision/models answer, reused for a short while; None when it has none.

        Only an answer is kept: the catalog, or a response without one (a 200 that lists no models,
        an HTTP 4xx: a plain OpenAI image endpoint). A request that failed (timeout, connection
        error, 5xx) keeps the previous answer, records why in `_discovery_failure`, and is made
        again by the next request that needs it.
        """
        if _looks_like_openai_api(self._cfg.base_url) or _model_family(self._cfg.model_id):
            return None
        with self._catalog_lock:
            fresh = self._catalog_at is not None and time.monotonic() - self._catalog_at < _DISCOVERY_TTL_S
            if fresh or not refresh:
                return self._catalog
        timeout = self._control_timeout()
        try:
            answer = self._get_json(
                path=_VISION_MODELS_PATH,
                timeout=_DISCOVERY_TIMEOUT_S if timeout is None else min(timeout, _DISCOVERY_TIMEOUT_S),
            )
        except ValueError:
            answer = {}  # a 200 that is not a JSON object
        except Exception as e:
            cause = e.__cause__
            if not (isinstance(cause, HTTPError) and int(cause.code) < 500):
                with self._catalog_lock:
                    self._discovery_failure = str(e) or type(e).__name__
                    return self._catalog
            answer = {}
        catalog = answer if isinstance(answer.get("models"), list) else None
        with self._catalog_lock:
            self._catalog = catalog
            self._catalog_at = time.monotonic()
            self._discovery_failure = None
        return catalog

    def _remote_entry(self, *, refresh: bool) -> Optional[Dict[str, Any]]:
        catalog = self._remote_catalog(refresh=refresh)
        return _discovery_entry(catalog, self._cfg.model_id) if catalog is not None else None

    def _unadvertised_reason(self) -> str:
        failure = self._discovery_failure
        if failure is not None:
            return f"GET {_VISION_MODELS_PATH} failed, so what the remote takes is unknown: {failure}"
        return f"the remote does not advertise them (no memory_modes for this model in GET {_VISION_MODELS_PATH})"

    @staticmethod
    def _extra_fields(request: Any) -> List[str]:
        extra = getattr(request, "extra", None)
        return [name for name in OPTIONAL_REQUEST_FIELDS if isinstance(extra, dict) and name in extra]

    def _prepare(self, request: Any, *, task: str) -> Tuple[Any, Dict[str, Any], bool]:
        """Check the request against what the remote takes.

        Returns the request, the fields to send and whether to run it as a remote job (a request
        with a `cancel_event`, on a remote that advertises the vision routes). `memory_mode` is sent
        when the request (or $ABSTRACTVISION_MEMORY_MODE) names one, the other fields when set; all
        only to a remote that advertises them. The same keys in `extra` travel with `extra` and are
        refused for any other remote.
        """
        extra = getattr(request, "extra", None)
        named = getattr(request, "memory_mode", None) or (extra or {}).get("memory_mode")
        needs_remote_entry = (
            getattr(request, "cancel_event", None) is not None
            or _option_is_set(named)
            or bool(str(os.environ.get(MEMORY_MODE_ENV_VAR) or "").strip())
            or bool(self._extra_fields(request))
            or any(_option_is_set(getattr(request, name, None)) for name in OPTIONAL_REQUEST_FIELDS)
        )
        if needs_remote_entry:
            self._remote_entry(refresh=True)
        advertised = self._remote_entry(refresh=False) is not None
        try:
            request, mode = self.resolve_memory_request(request, task=task)
        except CapabilityNotSupportedError as e:
            if advertised or self._discovery_failure is None:
                raise
            raise CapabilityNotSupportedError(f"{e} {self._unadvertised_reason()}.") from e
        if not advertised:
            in_extra = self._extra_fields(request)
            if in_extra:
                raise CapabilityNotSupportedError(
                    f"{type(self).__name__} does not support {', '.join(in_extra)}: {self._unadvertised_reason()}."
                )
            return request, {}, False
        fields: Dict[str, Any] = {}
        if _option_is_set(named) or mode != MEMORY_MODE_DEFAULT:
            fields["memory_mode"] = mode
        for name in OPTIONAL_REQUEST_FIELDS:
            value = getattr(request, name, None)
            if not _option_is_set(value):
                continue
            if name in (request.extra or {}) and request.extra[name] != value:
                raise ValueError(f"{name}={value!r} and extra[{name!r}]={request.extra[name]!r} disagree; pass one.")
            fields[name] = value
        return request, fields, getattr(request, "cancel_event", None) is not None

    def _send(
        self,
        path: str,
        job_route: str,
        request: Any,
        as_job: bool,
        *,
        payload: Optional[Dict[str, Any]] = None,
        fields: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, Tuple[str, bytes, str]]] = None,
    ) -> Dict[str, Any]:
        """POST the request (JSON `payload`, else multipart `fields`/`files`), as a remote job when `as_job`."""
        if as_job:
            return self._run_remote_job(
                job_route, cancel_event=request.cancel_event, payload=payload, fields=fields, files=files
            )
        if payload is not None:
            return self._post_json(path=path, payload=payload)
        return self._post_multipart(path=path, fields=fields or {}, files=files or {})

    def _run_remote_job(
        self,
        route: str,
        *,
        cancel_event: Any,
        payload: Optional[Dict[str, Any]] = None,
        fields: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, Tuple[str, bytes, str]]] = None,
    ) -> Dict[str, Any]:
        """Run the request as a remote job, polled until it ends; a set `cancel_event` cancels it.

        Whatever ends the polling before the job ends (the event, a network error, an interrupt)
        sends the job one best-effort cancel, so it does not keep the server busy. Returns the
        job's result, with the job's `memory_mode_used` and `audio` where the result has none
        (AbstractCore's image job routes record them on the job only).
        """
        path = f"{_VISION_JOBS_PATH}/{route}"
        try:
            if payload is not None:
                started = self._post_json(path=path, payload=payload)
            else:
                started = self._post_multipart(path=path, fields=fields or {}, files=files or {})
        except RuntimeError as e:
            if isinstance(e.__cause__, HTTPError) and e.__cause__.code == 409:
                raise RuntimeError(
                    f"Another generation is running on the server ({self._cfg.base_url}): a request with "
                    "cancel_event runs there as a job, which the server refuses until that generation ends."
                ) from e
            raise
        job_id = str(started.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("Invalid response: the remote job route returned no job_id")
        ended = False
        try:
            while True:
                if cancel_event.is_set():
                    raise GenerationCancelledError(
                        f"The request was cancelled (cancel_event is set); remote job {job_id} was asked to cancel."
                    )
                snap = self._get_json(path=f"{_VISION_JOBS_PATH}/{job_id}?consume=1")
                state = str(snap.get("state") or "")
                ended = state in {"succeeded", "cancelled", "failed"}
                if state == "succeeded":
                    result = snap.get("result")
                    if not isinstance(result, dict):
                        raise ValueError("Invalid response: the remote job has no result")
                    return {**{key: snap[key] for key in ("memory_mode_used", "audio") if key in snap}, **result}
                if state == "cancelled":
                    raise GenerationCancelledError(f"Remote job {job_id} was cancelled.")
                if state == "failed":
                    message = str(snap.get("error") or "unknown error")
                    if snap.get("error_code") == "capability_not_supported":
                        raise CapabilityNotSupportedError(message)
                    raise RuntimeError(f"Remote job {job_id} failed: {message}")
                time.sleep(_JOB_POLL_INTERVAL_S)
        except BaseException:
            if not ended:
                try:
                    self._post_json(
                        path=f"{_VISION_JOBS_PATH}/{job_id}/cancel", payload={}, timeout=self._control_timeout()
                    )
                except Exception:
                    pass
            raise

    def _get_json(self, *, path: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        url = _join_url(self._cfg.base_url, path)
        req = Request(url=url, method="GET", headers=self._headers())
        try:
            with urlopen(req, timeout=self._control_timeout() if timeout is None else timeout) as resp:
                raw = resp.read()
        except HTTPError as e:
            raise _http_error(e) from e
        except URLError as e:
            raise RuntimeError(f"OpenAI-compatible provider request failed: {e}") from e
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid response: expected JSON object")
        return data

    def _post_json(self, *, path: str, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        url = _join_url(self._cfg.base_url, path)
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            url=url,
            data=body,
            method="POST",
            headers=self._headers(content_type="application/json"),
        )
        try:
            with urlopen(req, timeout=self._generation_timeout() if timeout is None else timeout) as resp:
                raw = resp.read()
        except HTTPError as e:
            raise _http_error(e) from e
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
            with urlopen(req, timeout=self._generation_timeout()) as resp:
                raw = resp.read()
        except HTTPError as e:
            raise _http_error(e) from e
        except URLError as e:
            raise RuntimeError(f"OpenAI-compatible provider request failed: {e}") from e
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Invalid response: expected JSON object")
        return data

    def _download(self, url: str, *, fallback_mime: str) -> Tuple[bytes, str]:
        req = Request(url=url, method="GET")
        try:
            with urlopen(req, timeout=self._generation_timeout()) as resp:
                return resp.read(), str(resp.headers.get("Content-Type") or fallback_mime)
        except HTTPError as e:
            raise RuntimeError(_format_http_error(e)) from e
        except URLError as e:
            raise RuntimeError(f"OpenAI-compatible media download failed: {e}") from e

    def _parse_media(self, resp: Dict[str, Any], *, fallback_mime: str) -> GeneratedAsset:
        item = _first_data_item(resp)
        if "b64_json" in item:
            content = _decode_b64(str(item.get("b64_json") or ""))
            mime = _sniff_mime_type(content, fallback_mime)
            metadata: Dict[str, Any] = {"source": "b64_json"}
        elif "url" in item and isinstance(item.get("url"), str):
            # Best-effort: download bytes.
            u = str(item.get("url"))
            content, ct = self._download(u, fallback_mime=fallback_mime)
            mime = _sniff_mime_type(content, ct)
            metadata = {"source": "url", "url": u}
        else:
            raise ValueError("Invalid response: missing data[0].b64_json or data[0].url")
        media_type = "video" if mime.startswith("video/") else "image"
        mode_used = resp.get("memory_mode_used")
        if isinstance(mode_used, str) and mode_used:
            metadata["memory_mode_used"] = mode_used
        audio = None
        if media_type == "video":
            audio = _audio_track(
                item.get("audio") if isinstance(item.get("audio"), Mapping) else resp.get("audio"),
                lambda url: self._download(url, fallback_mime="audio/wav")[0],
            )
        return GeneratedAsset(media_type=media_type, data=content, mime_type=mime, metadata=metadata, audio=audio)

    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset:
        request, remote_fields, as_job = self._prepare(request, task="text_to_image")
        if request.control_image is not None:
            raise CapabilityNotSupportedError(
                "OpenAI-compatible backend does not support structured control images for text-to-image."
            )
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
        payload.update(_upstream_extra(request.extra))
        if payload.get("model") is not None:
            payload["model"] = _upstream_model_id(payload.get("model"))
        if family == "gpt-image":
            payload.pop("response_format", None)
        payload.update(remote_fields)

        resp = self._send(self._cfg.image_generations_path, "images/generations", request, as_job, payload=payload)
        return self._parse_media(resp, fallback_mime="image/png")

    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
        request, remote_fields, as_job = self._prepare(request, task="image_to_image")
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
        for k, v in _upstream_extra(request.extra).items():
            if v is None:
                continue
            fields[str(k)] = str(v)
        if fields.get("model") is not None:
            fields["model"] = _upstream_model_id(fields.get("model"))
        fields.update({name: _form_value(value) for name, value in remote_fields.items()})

        resp = self._send(self._cfg.image_edits_path, "images/edits", request, as_job, fields=fields, files=files)
        return self._parse_media(resp, fallback_mime="image/png")

    def generate_angles(self, request) -> list[GeneratedAsset]:
        raise CapabilityNotSupportedError(
            "OpenAICompatibleVisionBackend does not implement multi-view generation."
        )

    def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset:
        if not self._cfg.text_to_video_path:
            raise CapabilityNotSupportedError("text_to_video is not configured for this backend.")
        request, remote_fields, as_job = self._prepare(request, task="text_to_video")
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
        payload.update(_upstream_extra(request.extra))
        if payload.get("model") is not None:
            payload["model"] = _upstream_model_id(payload.get("model"))
        payload.update(remote_fields)
        resp = self._send(str(self._cfg.text_to_video_path), "videos/generations", request, as_job, payload=payload)
        return self._parse_media(resp, fallback_mime="video/mp4")

    def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset:
        if not self._cfg.image_to_video_path:
            raise CapabilityNotSupportedError("image_to_video is not configured for this backend.")
        request, remote_fields, as_job = self._prepare(request, task="image_to_video")

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
            payload.update(_upstream_extra(request.extra))
            if payload.get("model") is not None:
                payload["model"] = _upstream_model_id(payload.get("model"))
            payload.update(remote_fields)
            resp = self._send(str(self._cfg.image_to_video_path), "videos/edits", request, as_job, payload=payload)
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
        for k, v in _upstream_extra(request.extra).items():
            if v is None:
                continue
            fields[str(k)] = str(v)
        if fields.get("model") is not None:
            fields["model"] = _upstream_model_id(fields.get("model"))
        fields.update({name: _form_value(value) for name, value in remote_fields.items()})

        files = {"image": ("image.png", bytes(request.image), "image/png")}
        path = str(self._cfg.image_to_video_path)
        resp = self._send(path, "videos/edits", request, as_job, fields=fields, files=files)
        return self._parse_media(resp, fallback_mime="video/mp4")
