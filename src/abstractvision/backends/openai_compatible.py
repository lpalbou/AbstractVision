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
from urllib.parse import urlsplit
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
from .base_backend import (
    OPTIONAL_REQUEST_FIELDS,
    REQUEST_OPTION_FIELDS,
    SOUNDTRACK_REQUEST_FIELDS,
    VisionBackend,
    option_is_set,
)

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
_VIDEO_TASKS = {"text_to_video", "image_to_video"}
# How long a GET /vision/models answer is reused, the longest it may take, and the job poll period.
_DISCOVERY_TTL_S = 30.0
_DISCOVERY_TIMEOUT_S = 10.0
_JOB_POLL_INTERVAL_S = 0.5
# The statuses by which a remote says it has no such route (absent, not for GET, not implemented).
_NO_ROUTE_STATUSES = {404, 405, 501}


def _join_url(base_url: str, path: str) -> str:
    b = str(base_url or "").rstrip("/")
    p = str(path or "").strip()
    if not p:
        return b
    if not p.startswith("/"):
        p = "/" + p
    return b + p


def _upstream_extra(extra: Any) -> Dict[str, Any]:
    if not isinstance(extra, Mapping):
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


def _refusal(code: Any, message: Any, field: Any, supported: Any) -> Optional[CapabilityNotSupportedError]:
    """The CapabilityNotSupportedError a remote's structured error states; None unless its code is
    `capability_not_supported`. One reading for a synchronous error body and a failed job snapshot."""
    if code != "capability_not_supported":
        return None
    return CapabilityNotSupportedError(
        str(message or "The remote refused the request."),
        field=str(field) if field is not None else None,
        supported=supported if isinstance(supported, list) else None,
    )


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
    if isinstance(err, dict):
        refusal = _refusal(err.get("code"), err.get("message"), err.get("field"), err.get("supported"))
        if refusal is not None:
            return refusal
    return RuntimeError(_format_http_error(e, body))


def _form_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _discovery_entry(
    catalog: Mapping[str, Any], model: Optional[str], task: Optional[str]
) -> Optional[Dict[str, Any]]:
    """The GET /vision/models entry of `model`, the string a request sends (`_sent_model`).

    A named model selects the entry whose `routed_model` is that string (`provider/model`, the one
    name the server resolves to that entry, `<provider>/default` for a task's default); no model
    selects `defaults[task]`, the model the server runs for a request that names none. Only an
    entry that lists `memory_modes` advertises the memory and sound fields.
    """
    model = str(model or "").strip()
    defaults = catalog.get("defaults")
    default = defaults.get(task) if isinstance(defaults, Mapping) and task else None
    candidates = [*(catalog.get("models") or []), default] if model else [default]
    for entry in candidates:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("memory_modes"), list):
            continue
        if not model or str(entry.get("routed_model") or "").strip() == model:
            return dict(entry)
    return None


def _entry_memory_modes(entry: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
    """The default mode plus the modes a discovery entry lists."""
    listed = (entry or {}).get("memory_modes") or ()
    return tuple(dict.fromkeys([MEMORY_MODE_DEFAULT, *(str(m) for m in listed)]))


def _entry_request_options(entry: Optional[Mapping[str, Any]], task: Optional[str]) -> FrozenSet[str]:
    """The typed request options a discovery entry advertises for `task`."""
    if entry is None:
        return frozenset()
    options: Set[str] = {"quantize"}
    if entry.get("supports_audio") and (task is None or str(task) in _VIDEO_TASKS):
        audio_fields = entry.get("audio_fields") or ()
        options.update(str(name) for name in audio_fields if name in SOUNDTRACK_REQUEST_FIELDS)
    return frozenset(options)


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


def looks_like_openai_api(base_url: Optional[str]) -> bool:
    """Whether `base_url` is the OpenAI API: its host is api.openai.com or one of its subdomains
    (a regional host such as eu.api.openai.com). The rest of the URL is not read."""
    try:
        host = urlsplit(str(base_url or "").strip()).hostname or ""
    except ValueError:
        return False
    return host == "api.openai.com" or host.endswith(".api.openai.com")


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
        # Held for a GET /vision/models and the recording of its answer: a caller that needs the
        # answer while another caller is asking waits for it.
        self._catalog_lock = threading.Lock()
        self._catalog: Optional[Dict[str, Any]] = None
        self._catalog_at: Optional[float] = None
        # Why the last GET /vision/models failed (timeout, connection error, 5xx); None after an answer.
        self._discovery_failure: Optional[str] = None
        # Whether the remote serves the vision discovery route (AbstractCore's server): learned from
        # the first definite answer and kept, unlike the catalog's contents. None until then.
        self._serves_vision_routes: Optional[bool] = None
        # When the last GET /vision/models failed; None after an answer. A question asked only to
        # learn the remote's kind is not asked again within _DISCOVERY_TTL_S of a failure.
        self._discovery_failed_at: Optional[float] = None

    def get_capabilities(self) -> VisionBackendCapabilities:
        tasks = {"text_to_image", "image_to_image"}
        if self._cfg.text_to_video_path:
            tasks.add("text_to_video")
        if self._cfg.image_to_video_path:
            tasks.add("image_to_video")
        entry = self._remote_entry(self._sent_model({}), task=None)
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
        """The default mode plus the modes the remote lists for the configured model (with none
        configured, for the remote's default of `task`).

        Read from the last GET /vision/models answer (no I/O here, nor in `get_capabilities()`);
        `list_provider_models()` and a request that sets one of the fields refresh it. A request
        is checked against the entry of the model it sends (`_prepare`).
        """
        return _entry_memory_modes(self._remote_entry(self._sent_model({}), task=task))

    def supported_request_options(self, task: Optional[str] = None) -> FrozenSet[str]:
        return _entry_request_options(self._remote_entry(self._sent_model({}), task=task), task)

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        catalog = self._remote_catalog(refresh=True)
        if catalog is not None:
            # A remote that serves the vision routes (AbstractCore's server) lists its vision
            # models there; its GET /models lists its text models.
            return self._vision_catalog_models(catalog, task=task)
        models_path = str(self._cfg.models_path or "/models")
        try:
            resp = self._get_json(path=models_path)
        except RuntimeError as e:
            status = e.__cause__.code if isinstance(e.__cause__, HTTPError) else None
            if status in _NO_ROUTE_STATUSES:
                raise CapabilityNotSupportedError(
                    f"The remote at {self._cfg.base_url} has no model catalog: GET {models_path} answered HTTP {status}."
                ) from e
            raise
        data = resp.get("data")
        if not isinstance(data, list):
            raise ValueError("Invalid response: expected JSON object with a data list")

        openai_api = looks_like_openai_api(self._cfg.base_url)
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

    def _vision_catalog_models(
        self, catalog: Mapping[str, Any], *, task: Optional[str]
    ) -> List[ProviderModelInfo]:
        """The GET /vision/models entries, each named by its `routed_model`: the string a request
        sends as `model` to run it."""
        models: List[ProviderModelInfo] = []
        for item in catalog.get("models") or ():
            if not isinstance(item, Mapping):
                continue
            raw = dict(item)
            model_id = str(raw.get("routed_model") or raw.get("model") or raw.get("id") or "").strip()
            if not model_id:
                continue
            raw.setdefault("backend", "openai-compatible")
            owner = raw.get("owned_by") or raw.get("provider")
            info = ProviderModelInfo(
                id=model_id,
                object=str(raw.get("object")) if raw.get("object") is not None else None,
                created=raw.get("created") if isinstance(raw.get("created"), int) else None,
                owned_by=str(owner) if owner is not None else None,
                capabilities=tuple(sorted(_catalog_capability_tokens(raw))),
                raw=raw,
            )
            if _catalog_entry_matches_task(info, task=task, openai_api=False):
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
        again by the next request that needs it. A caller that refreshes while another is asking
        waits for that answer and reuses it.
        """
        if looks_like_openai_api(self._cfg.base_url):
            return None
        if not refresh:
            return self._catalog
        with self._catalog_lock:
            fresh = self._catalog_at is not None and time.monotonic() - self._catalog_at < _DISCOVERY_TTL_S
            if not fresh:
                self._ask_vision_models_locked()
            return self._catalog

    def _learn_remote_kind(self) -> None:
        """Ask GET /vision/models until the remote's kind is known, at most once per
        _DISCOVERY_TTL_S after a failure; a caller that arrives while the question is in flight
        waits for its answer."""
        if looks_like_openai_api(self._cfg.base_url):
            return
        with self._catalog_lock:
            failed_at = self._discovery_failed_at
            recently_failed = failed_at is not None and time.monotonic() - failed_at < _DISCOVERY_TTL_S
            if self._serves_vision_routes is None and not recently_failed:
                self._ask_vision_models_locked()

    def _ask_vision_models_locked(self) -> None:
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
                self._discovery_failure = str(e) or type(e).__name__
                self._discovery_failed_at = time.monotonic()
                return
            answer = {}
        catalog = answer if isinstance(answer.get("models"), list) else None
        if self._serves_vision_routes is None:
            self._serves_vision_routes = catalog is not None
        self._catalog = catalog
        self._catalog_at = time.monotonic()
        self._discovery_failure = None
        self._discovery_failed_at = None

    def _remote_entry(self, model: Optional[str], *, task: Optional[str]) -> Optional[Dict[str, Any]]:
        """The entry of `model` (a string `_sent_model` computed) in the last discovery answer."""
        catalog = self._catalog
        return _discovery_entry(catalog, model, task) if catalog is not None else None

    def _sent_model(self, extra: Any) -> Optional[str]:
        """The `model` a request sends: `extra["model"]`, else the configured model.

        A remote that serves the vision routes gets it as given: its `routed_model` values are the
        names it routes. Any other remote gets it without `openai/` and `openai-compatible/`
        prefixes.
        """
        model = self._given_model(extra)
        if model is None:
            return None
        return model if self._serves_vision_routes else _upstream_model_id(model)

    def _given_model(self, extra: Any) -> Optional[str]:
        """The model a request names: `extra["model"]`, else the configured model."""
        given = extra.get("model") if isinstance(extra, Mapping) else None
        model = given if given is not None else self._cfg.model_id
        return str(model).strip() if model else None

    def _unadvertised_reason(self) -> str:
        failure = self._discovery_failure
        if failure is not None:
            return f"GET {_VISION_MODELS_PATH} failed, so what the remote takes is unknown: {failure}"
        return f"the remote does not advertise them (no memory_modes for this model in GET {_VISION_MODELS_PATH})"

    def _with_extra(self, body: Dict[str, Any], extra: Any, *, form: bool) -> Dict[str, Any]:
        """`body` (a JSON payload or form fields) with the request's `extra` and its `model`.

        A remote known to serve the vision discovery route (AbstractCore's server, whose bodies
        refuse unknown top-level keys) gets `extra` nested: an `extra` object in JSON, an
        `extra_json` form field. Any other remote, and one whose kind stayed unknown (its
        discovery failed), gets the keys at the top level (on a form as strings, None dropped).
        `model` is `_sent_model`'s, `extra["model"]` included.
        """
        upstream = _upstream_extra(extra)
        upstream.pop("model", None)
        out = dict(body)
        if not self._serves_vision_routes:
            if form:
                upstream = {key: str(value) for key, value in upstream.items() if value is not None}
            out.update(upstream)
        elif upstream:
            out["extra_json" if form else "extra"] = json.dumps(upstream) if form else upstream
        model = self._sent_model(extra)
        if model is not None:
            out["model"] = model
        return out

    def _prepare(self, request: Any, *, task: str) -> Tuple[Any, Dict[str, Any], bool]:
        """Check the request against what the remote takes.

        Returns the request, the fields to send and whether to run it as a remote job (a request
        with a `cancel_event`, on a remote that advertises the vision routes). The options are
        resolved from their fields and `extra` against the discovery entry of the model the request
        sends (`_sent_model`); `memory_mode` is sent when the request (or
        $ABSTRACTVISION_MEMORY_MODE) names one, the other options when set, all only to a remote
        that advertises them.
        """
        extra = getattr(request, "extra", None)
        extra = extra if isinstance(extra, Mapping) else {}
        named = option_is_set(getattr(request, "memory_mode", None)) or option_is_set(extra.get("memory_mode"))
        needs_remote_entry = (
            getattr(request, "cancel_event", None) is not None
            or bool(str(os.environ.get(MEMORY_MODE_ENV_VAR) or "").strip())
            or any(option_is_set(getattr(request, name, None)) for name in REQUEST_OPTION_FIELDS)
            # `extra["memory_mode"]` is the option on every backend; the other `extra` keys are the
            # option only for a remote that advertises it, so they never ask on their own.
            or option_is_set(extra.get("memory_mode"))
        )
        given = self._given_model(extra)
        if needs_remote_entry:
            self._remote_catalog(refresh=True)
        elif set(_upstream_extra(extra)) - {"model"} or (given is not None and _upstream_model_id(given) != given):
            # The body shape of `extra`, and the string sent for a model named with an `openai/` or
            # `openai-compatible/` prefix, depend on the remote's kind.
            self._learn_remote_kind()
        entry = self._remote_entry(self._sent_model(extra), task=task)
        advertised = entry is not None
        try:
            request, mode = self._resolve_request_options(
                request,
                task=task,
                supported=_entry_memory_modes(entry),
                listed=_entry_request_options(entry, task),
            )
        except CapabilityNotSupportedError as e:
            if advertised or self._discovery_failure is None:
                raise
            raise CapabilityNotSupportedError(
                f"{e} {self._unadvertised_reason()}.", field=e.field, supported=e.supported
            ) from e
        if getattr(request, "cancel_event", None) is not None and self._serves_vision_routes is None and (
            self._discovery_failure is not None
        ):
            # Whether the remote can cancel a run is unknown: the request is not run uncancellable.
            raise CapabilityNotSupportedError(
                f"{type(self).__name__} cannot honour cancel_event: {self._unadvertised_reason()}.",
                field="cancel_event",
            )
        if not advertised:
            return request, {}, False
        fields: Dict[str, Any] = {}
        if named or mode != MEMORY_MODE_DEFAULT:
            fields["memory_mode"] = mode
        for name in OPTIONAL_REQUEST_FIELDS:
            value = getattr(request, name, None)
            if option_is_set(value):
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
                    refusal = _refusal(snap.get("error_code"), message, snap.get("error_field"), snap.get("error_supported"))
                    raise refusal or RuntimeError(f"Remote job {job_id} failed: {message}")
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
        # The mode the remote reports it ran with, under the key every backend uses; absent when
        # the remote does not say (a plain OpenAI endpoint).
        mode_used = resp.get("memory_mode_used")
        if isinstance(mode_used, str) and mode_used:
            metadata["memory_mode"] = mode_used
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
                "OpenAI-compatible backend does not support structured control images for text-to-image.",
                field="control_image",
                supported=(),
            )
        family = _model_family(self._cfg.model_id)
        openai_api = looks_like_openai_api(self._cfg.base_url) or family is not None
        payload: Dict[str, Any] = {
            "prompt": request.prompt,
            "n": 1,
        }
        if self._cfg.model_id:
            payload["model"] = self._sent_model(request.extra)

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
        payload = self._with_extra(payload, request.extra, form=False)
        if family == "gpt-image":
            payload.pop("response_format", None)
        payload.update(remote_fields)

        resp = self._send(self._cfg.image_generations_path, "images/generations", request, as_job, payload=payload)
        return self._parse_media(resp, fallback_mime="image/png")

    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset:
        request, remote_fields, as_job = self._prepare(request, task="image_to_image")
        family = _model_family(self._cfg.model_id)
        openai_api = looks_like_openai_api(self._cfg.base_url) or family is not None
        # OpenAI-style image edits use multipart form data.
        fields: Dict[str, str] = {"prompt": request.prompt}
        if self._cfg.model_id:
            fields["model"] = self._sent_model(request.extra)
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
        fields = self._with_extra(fields, request.extra, form=True)
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
            payload["model"] = self._sent_model(request.extra)
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
        payload = self._with_extra(payload, request.extra, form=False)
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
                payload["model"] = self._sent_model(request.extra)
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
            payload = self._with_extra(payload, request.extra, form=False)
            payload.update(remote_fields)
            resp = self._send(str(self._cfg.image_to_video_path), "videos/edits", request, as_job, payload=payload)
            return self._parse_media(resp, fallback_mime="video/mp4")

        fields: Dict[str, str] = {}
        if self._cfg.model_id:
            fields["model"] = self._sent_model(request.extra)
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
        fields = self._with_extra(fields, request.extra, form=True)
        fields.update({name: _form_value(value) for name, value in remote_fields.items()})

        files = {"image": ("image.png", bytes(request.image), "image/png")}
        path = str(self._cfg.image_to_video_path)
        resp = self._send(path, "videos/edits", request, as_job, fields=fields, files=files)
        return self._parse_media(resp, fallback_mime="video/mp4")
