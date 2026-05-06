from __future__ import annotations

import base64
import json
import os
import shlex
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .model_capabilities import VisionModelCapabilitiesRegistry
from .types import GeneratedAsset, ImageEditRequest, ImageGenerationRequest

DEFAULT_PLAYGROUND_HOST = "127.0.0.1"
DEFAULT_PLAYGROUND_PORT = 8091
DEFAULT_DIFFUSERS_MODEL_ID = "runwayml/stable-diffusion-v1-5"


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _env_bool(key: str, default: bool = False) -> bool:
    v = _env(key)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    s = str(value).strip()
    if not s:
        return None
    return int(s)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    return float(s)


def _redact(text: Any) -> str:
    raw = str(text or "")
    for key in (
        "OPENAI_API_KEY",
        "ABSTRACTVISION_API_KEY",
        "ABSTRACTVISION_UPSTREAM_API_KEY",
        "ABSTRACTCORE_SERVER_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        value = os.environ.get(key)
        if value:
            raw = raw.replace(str(value), "***")
    return raw


def _known_prefix(model_id: str) -> Tuple[Optional[str], str]:
    s = str(model_id or "").strip()
    if not s or "/" not in s:
        return None, s
    head, tail = s.split("/", 1)
    head = head.strip().lower()
    if head in {
        "diffusers",
        "huggingface",
        "hf",
        "mlx",
        "sdcpp",
        "stable-diffusion.cpp",
        "stable_diffusion_cpp",
        "stable-diffusion-cpp",
        "openai",
        "openai-compatible",
        "openai_compatible",
    }:
        return head, tail.strip()
    return None, s


def _is_default_alias(value: str) -> bool:
    return str(value or "").strip().lower() in {"", "default", "server/default"}


def normalize_model_id_for_backend(model_id: str) -> Tuple[str, Optional[str]]:
    """Return ``(backend_kind, backend_model_id)`` for playground model-load requests.

    AbstractVision's own registry uses raw Hugging Face repo ids, while some
    callers use an explicit backend prefix. The playground accepts both shapes
    so the local UI can stay independent without being brittle.
    """

    s = str(model_id or "").strip()
    if not s:
        raise ValueError("Missing required field: model_id")

    prefix, rest = _known_prefix(s)
    if prefix in {"diffusers", "huggingface", "hf", "mlx"}:
        return "diffusers", DEFAULT_DIFFUSERS_MODEL_ID if _is_default_alias(rest) else rest
    if prefix in {"sdcpp", "stable-diffusion.cpp", "stable_diffusion_cpp", "stable-diffusion-cpp"}:
        return "sdcpp", None if _is_default_alias(rest) else rest
    if prefix in {"openai", "openai-compatible", "openai_compatible"}:
        return "openai", None if _is_default_alias(rest) else rest

    # Raw registry ids like `runwayml/stable-diffusion-v1-5` are first-class in
    # AbstractVision and map to the local Diffusers backend by default.
    if _is_default_alias(s):
        return "diffusers", DEFAULT_DIFFUSERS_MODEL_ID
    return "diffusers", s


def _hf_cache_roots(extra_cache_dir: Optional[str] = None) -> List[Tuple[str, Path]]:
    roots: List[Tuple[str, Path]] = []

    def add(label: str, value: Optional[str]) -> None:
        if not value:
            return
        p = Path(str(value)).expanduser()
        if not any(existing.resolve() == p.resolve() for _, existing in roots):
            roots.append((label, p))

    add("configured cache", extra_cache_dir)
    add("HF_HUB_CACHE", _env("HF_HUB_CACHE"))
    hf_home = _env("HF_HOME")
    if hf_home:
        add("HF_HOME", str(Path(hf_home).expanduser() / "hub"))
    add("default HF cache", str(Path.home() / ".cache" / "huggingface" / "hub"))
    return roots


def _cached_hf_model_sources(model_id: str, *, cache_dir: Optional[str] = None) -> List[str]:
    if "/" not in str(model_id or ""):
        return []
    sources: List[str] = []
    repo_dir_name = "models--" + str(model_id).replace("/", "--")
    for label, root in _hf_cache_roots(cache_dir):
        snaps = root / repo_dir_name / "snapshots"
        try:
            if snaps.is_dir() and any(p.is_dir() for p in snaps.iterdir()):
                sources.append(label)
        except Exception:
            continue
    return sources


def _parse_json_bytes(body: bytes) -> Dict[str, Any]:
    if not body:
        return {}
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def _parse_extra_json(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    s = str(value or "").strip()
    if not s:
        return {}
    obj = json.loads(s)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError("extra_json must decode to an object")
    return obj


def _parse_multipart(
    content_type: str, body: bytes
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
    msg = BytesParser(policy=policy.default).parsebytes(header + body)
    if not msg.is_multipart():
        raise ValueError("Expected multipart/form-data request body")

    fields: Dict[str, str] = {}
    files: Dict[str, Dict[str, Any]] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_param("filename", header="content-disposition")
        if filename is not None:
            files[str(name)] = {
                "filename": str(filename),
                "content_type": part.get_content_type(),
                "content": payload,
            }
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[str(name)] = payload.decode(charset, errors="replace")
    return fields, files


def _asset_to_image_response(
    asset: GeneratedAsset, *, response_format: str = "b64_json"
) -> Dict[str, Any]:
    fmt = str(response_format or "b64_json").strip().lower()
    if fmt != "b64_json":
        raise ValueError(
            "The playground server currently supports response_format='b64_json' only."
        )
    return {
        "created": int(time.time()),
        "data": [
            {
                "b64_json": base64.b64encode(bytes(asset.data)).decode("ascii"),
                "mime_type": asset.mime_type,
                "metadata": dict(asset.metadata or {}),
            }
        ],
    }


def _request_kwargs(payload: Dict[str, Any], *, known: set) -> Dict[str, Any]:
    extra = _parse_extra_json(payload.get("extra"))
    for k, v in payload.items():
        if k not in known and v is not None:
            extra[k] = v
    return extra


@dataclass
class PlaygroundServerConfig:
    host: str = DEFAULT_PLAYGROUND_HOST
    port: int = DEFAULT_PLAYGROUND_PORT
    diffusers_device: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_DIFFUSERS_DEVICE", "auto") or "auto"
    )
    diffusers_torch_dtype: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE")
    )
    diffusers_allow_download: bool = field(
        default_factory=lambda: _env_bool(
            "ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD",
            _env_bool("ABSTRACTVISION_ALLOW_DOWNLOAD", False),
        )
    )
    diffusers_cache_dir: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_DIFFUSERS_CACHE_DIR")
    )
    diffusers_revision: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_DIFFUSERS_REVISION")
    )
    diffusers_variant: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_DIFFUSERS_VARIANT")
    )
    diffusers_auto_retry_fp32: bool = field(
        default_factory=lambda: _env_bool("ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32", True)
    )
    default_model_id: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_MODEL_ID", DEFAULT_DIFFUSERS_MODEL_ID)
        or DEFAULT_DIFFUSERS_MODEL_ID
    )

    backend_kind: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_BACKEND", "diffusers") or "diffusers"
    )
    openai_base_url: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_BASE_URL"))
    openai_api_key: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_API_KEY"))
    openai_timeout_s: float = field(
        default_factory=lambda: float(_env("ABSTRACTVISION_TIMEOUT_S", "300") or "300")
    )
    openai_image_generations_path: str = field(
        default_factory=lambda: _env(
            "ABSTRACTVISION_IMAGES_GENERATIONS_PATH", "/images/generations"
        )
        or "/images/generations"
    )
    openai_image_edits_path: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_IMAGES_EDITS_PATH", "/images/edits")
        or "/images/edits"
    )

    sdcpp_bin: str = field(
        default_factory=lambda: _env("ABSTRACTVISION_SDCPP_BIN", "sd-cli") or "sd-cli"
    )
    sdcpp_model: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_SDCPP_MODEL"))
    sdcpp_diffusion_model: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_SDCPP_DIFFUSION_MODEL")
    )
    sdcpp_vae: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_SDCPP_VAE"))
    sdcpp_llm: Optional[str] = field(default_factory=lambda: _env("ABSTRACTVISION_SDCPP_LLM"))
    sdcpp_llm_vision: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_SDCPP_LLM_VISION")
    )
    sdcpp_extra_args: Optional[str] = field(
        default_factory=lambda: _env("ABSTRACTVISION_SDCPP_EXTRA_ARGS")
    )


@dataclass
class _Job:
    job_id: str
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def snapshot(self, *, include_result: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "job_id": self.job_id,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.progress is not None:
            out["progress"] = dict(self.progress)
        if self.error:
            out["error"] = self.error
        if include_result and self.result is not None:
            out["result"] = self.result
        return out


class PlaygroundState:
    def __init__(self, config: PlaygroundServerConfig):
        self.config = config
        self.registry = VisionModelCapabilitiesRegistry()
        self._active_lock = threading.RLock()
        self._active_backend: Any = None
        self._active_backend_kind: Optional[str] = None
        self._active_model_id: Optional[str] = None
        self._active_loaded_at: Optional[float] = None
        self._jobs_lock = threading.RLock()
        self._jobs: Dict[str, _Job] = {}

    def active_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._active_lock:
            if not self._active_backend or not self._active_model_id:
                return None
            return {
                "model_id": self._active_model_id,
                "load_id": self._active_model_id,
                "backend": self._active_backend_kind,
                "loaded_at": self._active_loaded_at,
            }

    def list_models(self) -> Dict[str, Any]:
        models: List[Dict[str, Any]] = []
        allow_download = bool(self.config.diffusers_allow_download)
        configured = str(self.config.default_model_id or "").strip()

        for model_id in self.registry.list_models():
            spec = self.registry.get(model_id)
            cached_in = _cached_hf_model_sources(
                model_id, cache_dir=self.config.diffusers_cache_dir
            )
            if cached_in or allow_download:
                models.append(
                    {
                        "id": model_id,
                        "load_id": model_id,
                        "provider": spec.provider,
                        "backend": "diffusers",
                        "tasks": sorted(spec.tasks.keys()),
                        "cached": bool(cached_in),
                        "cached_in": cached_in if cached_in else ["download enabled"],
                    }
                )

        if configured and configured not in {m["id"] for m in models}:
            prefix, rest = _known_prefix(configured)
            if prefix in {"openai", "openai-compatible", "openai_compatible"}:
                label = configured
                load_id = configured
                backend = "openai"
                cached_in = ["configured remote"]
            elif str(self.config.backend_kind).strip().lower() in {"openai", "openai-compatible"}:
                rest = configured
                label = f"openai-compatible/{rest}" if rest else "openai-compatible/default"
                load_id = label
                backend = "openai"
                cached_in = ["configured remote"]
            else:
                label = configured
                load_id = configured
                backend = "diffusers"
                cached_in = _cached_hf_model_sources(
                    configured, cache_dir=self.config.diffusers_cache_dir
                )
                if not cached_in and allow_download:
                    cached_in = ["download enabled"]
            if cached_in:
                models.append(
                    {
                        "id": label,
                        "load_id": load_id,
                        "provider": "configured",
                        "backend": backend,
                        "tasks": ["text_to_image", "image_to_image"],
                        "cached": "cache" in ",".join(cached_in).lower(),
                        "cached_in": cached_in,
                    }
                )

        if str(self.config.backend_kind).strip().lower() in {
            "sdcpp",
            "stable-diffusion.cpp",
            "stable_diffusion_cpp",
        }:
            models.append(
                {
                    "id": "sdcpp/default",
                    "load_id": "sdcpp/default",
                    "provider": "configured",
                    "backend": "sdcpp",
                    "tasks": ["text_to_image", "image_to_image"],
                    "cached": True,
                    "cached_in": ["configured local files"],
                }
            )

        return {"models": models, "active": self.active_snapshot()}

    def unload_active(self) -> Dict[str, Any]:
        with self._active_lock:
            backend = self._active_backend
            self._active_backend = None
            self._active_backend_kind = None
            self._active_model_id = None
            self._active_loaded_at = None
        if backend is not None:
            unload = getattr(backend, "unload", None)
            if callable(unload):
                unload()
        return {"ok": True, "active": None}

    def load_model(self, requested_model_id: str) -> Dict[str, Any]:
        backend_kind, backend_model_id = normalize_model_id_for_backend(requested_model_id)

        backend = self._build_backend(backend_kind, backend_model_id)
        with self._active_lock:
            old = self._active_backend
            self._active_backend = None
            self._active_backend_kind = None
            self._active_model_id = None
            self._active_loaded_at = None
            if old is not None and callable(getattr(old, "unload", None)):
                old.unload()

            preload = getattr(backend, "preload", None)
            if callable(preload):
                preload()

            self._active_backend = backend
            self._active_backend_kind = backend_kind
            self._active_model_id = requested_model_id
            self._active_loaded_at = time.time()
            return {"ok": True, "active": self.active_snapshot()}

    def _build_backend(self, backend_kind: str, backend_model_id: Optional[str]) -> Any:
        if backend_kind == "diffusers":
            from .backends.huggingface_diffusers import (
                HuggingFaceDiffusersBackendConfig,
                HuggingFaceDiffusersVisionBackend,
            )

            model_id = (
                backend_model_id or self.config.default_model_id or DEFAULT_DIFFUSERS_MODEL_ID
            )
            cfg = HuggingFaceDiffusersBackendConfig(
                model_id=str(model_id),
                device=str(self.config.diffusers_device or "auto"),
                torch_dtype=self.config.diffusers_torch_dtype,
                allow_download=bool(self.config.diffusers_allow_download),
                auto_retry_fp32=bool(self.config.diffusers_auto_retry_fp32),
                cache_dir=self.config.diffusers_cache_dir,
                revision=self.config.diffusers_revision,
                variant=self.config.diffusers_variant,
            )
            return HuggingFaceDiffusersVisionBackend(config=cfg)

        if backend_kind == "sdcpp":
            from .backends.stable_diffusion_cpp import (
                StableDiffusionCppBackendConfig,
                StableDiffusionCppVisionBackend,
            )

            explicit_model = str(backend_model_id).strip() if backend_model_id else None
            cfg = StableDiffusionCppBackendConfig(
                sd_cli_path=str(self.config.sdcpp_bin),
                model=explicit_model or self.config.sdcpp_model,
                diffusion_model=None if explicit_model else self.config.sdcpp_diffusion_model,
                vae=None if explicit_model else self.config.sdcpp_vae,
                llm=None if explicit_model else self.config.sdcpp_llm,
                llm_vision=None if explicit_model else self.config.sdcpp_llm_vision,
                extra_args=(
                    shlex.split(str(self.config.sdcpp_extra_args))
                    if self.config.sdcpp_extra_args
                    else ()
                ),
            )
            return StableDiffusionCppVisionBackend(config=cfg)

        if backend_kind == "openai":
            from .backends.openai_compatible import (
                OpenAICompatibleBackendConfig,
                OpenAICompatibleVisionBackend,
            )

            if not self.config.openai_base_url:
                raise ValueError(
                    "OpenAI-compatible backend is not configured. Set ABSTRACTVISION_BASE_URL, "
                    "or select a local Diffusers/sdcpp model."
                )
            cfg = OpenAICompatibleBackendConfig(
                base_url=str(self.config.openai_base_url),
                api_key=str(self.config.openai_api_key) if self.config.openai_api_key else None,
                model_id=str(backend_model_id) if backend_model_id else None,
                timeout_s=float(self.config.openai_timeout_s),
                image_generations_path=str(self.config.openai_image_generations_path),
                image_edits_path=str(self.config.openai_image_edits_path),
            )
            return OpenAICompatibleVisionBackend(config=cfg)

        raise ValueError(f"Unknown backend kind: {backend_kind!r}")

    def _active_backend_or_raise(self) -> Tuple[Any, str]:
        backend = self._active_backend
        model_id = self._active_model_id
        if backend is None or not model_id:
            raise ValueError("No active model. Select and load a model first.")
        return backend, model_id

    def start_image_generation_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Missing required field: prompt")
        known = {
            "prompt",
            "model",
            "response_format",
            "negative_prompt",
            "width",
            "height",
            "steps",
            "guidance_scale",
            "seed",
            "extra",
        }
        request = ImageGenerationRequest(
            prompt=prompt,
            negative_prompt=(
                str(payload.get("negative_prompt")) if payload.get("negative_prompt") else None
            ),
            width=_to_int(payload.get("width")),
            height=_to_int(payload.get("height")),
            steps=_to_int(payload.get("steps")),
            guidance_scale=_to_float(payload.get("guidance_scale")),
            seed=_to_int(payload.get("seed")),
            extra=_request_kwargs(payload, known=known),
        )
        response_format = str(payload.get("response_format") or "b64_json")

        def run(progress_callback: Callable[[int, Optional[int]], None]) -> Dict[str, Any]:
            with self._active_lock:
                backend, _model_id = self._active_backend_or_raise()
                asset = backend.generate_image_with_progress(
                    request, progress_callback=progress_callback
                )
            return _asset_to_image_response(asset, response_format=response_format)

        return self._start_job(run, total_steps=request.steps)

    def start_image_edit_job(
        self,
        fields: Dict[str, str],
        files: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = str(fields.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Missing required field: prompt")
        image = files.get("image")
        if not image:
            raise ValueError("Missing required multipart file: image")
        mask = files.get("mask")
        extra = _parse_extra_json(fields.get("extra_json"))
        request = ImageEditRequest(
            prompt=prompt,
            image=bytes(image.get("content") or b""),
            mask=bytes(mask.get("content") or b"") if mask else None,
            negative_prompt=(
                str(fields.get("negative_prompt")) if fields.get("negative_prompt") else None
            ),
            steps=_to_int(fields.get("steps")),
            guidance_scale=_to_float(fields.get("guidance_scale")),
            seed=_to_int(fields.get("seed")),
            extra=extra,
        )

        def run(progress_callback: Callable[[int, Optional[int]], None]) -> Dict[str, Any]:
            with self._active_lock:
                backend, _model_id = self._active_backend_or_raise()
                asset = backend.edit_image_with_progress(
                    request, progress_callback=progress_callback
                )
            return _asset_to_image_response(asset, response_format="b64_json")

        return self._start_job(run, total_steps=request.steps)

    def _start_job(
        self,
        fn: Callable[[Callable[[int, Optional[int]], None]], Dict[str, Any]],
        *,
        total_steps: Optional[int],
    ) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        job = _Job(
            job_id=job_id,
            state="queued",
            progress={"step": 0, "total_steps": total_steps} if total_steps is not None else None,
        )
        with self._jobs_lock:
            self._jobs[job_id] = job

        def progress(step: int, total: Optional[int]) -> None:
            with self._jobs_lock:
                current = self._jobs.get(job_id)
                if current is None:
                    return
                current.progress = {"step": int(step), "total_steps": total}
                current.updated_at = time.time()

        def runner() -> None:
            with self._jobs_lock:
                current = self._jobs.get(job_id)
                if current is not None:
                    current.state = "running"
                    current.updated_at = time.time()
            try:
                result = fn(progress)
                with self._jobs_lock:
                    current = self._jobs.get(job_id)
                    if current is not None:
                        current.state = "succeeded"
                        current.result = result
                        current.updated_at = time.time()
            except Exception as e:
                with self._jobs_lock:
                    current = self._jobs.get(job_id)
                    if current is not None:
                        current.state = "failed"
                        current.error = _redact(str(e))
                        current.updated_at = time.time()

        threading.Thread(
            target=runner, name=f"abstractvision-job-{job_id[:8]}", daemon=True
        ).start()
        return job.snapshot(include_result=False)

    def get_job(self, job_id: str, *, consume: bool = False) -> Dict[str, Any]:
        with self._jobs_lock:
            job = self._jobs.get(str(job_id))
            if job is None:
                raise KeyError(str(job_id))
            snap = job.snapshot(include_result=True)
            if consume and job.state in {"succeeded", "failed"}:
                self._jobs.pop(str(job_id), None)
            return snap


def _playground_html_bytes() -> bytes:
    try:
        return (
            resources.files("abstractvision.playground")
            .joinpath("vision_playground.html")
            .read_bytes()
        )
    except Exception:
        pass

    # Repo checkout fallback for editable installs with unusual package-data handling.
    p = Path(__file__).resolve().parents[2] / "playground" / "vision_playground.html"
    if p.is_file():
        return p.read_bytes()
    raise FileNotFoundError(
        "playground/vision_playground.html was not found. Run this command from an AbstractVision checkout."
    )


def _make_handler(state: PlaygroundState) -> type[BaseHTTPRequestHandler]:
    html_cache = _playground_html_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "AbstractVisionPlayground/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")

        def _send_json(self, status: int, obj: Dict[str, Any]) -> None:
            body = json.dumps(obj, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, status: int, message: str) -> None:
            self._send_json(status, {"error": {"message": _redact(message), "type": "http_error"}})

        def _send_html(self) -> None:
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_cache)))
            self.end_headers()
            self.wfile.write(html_cache)

        def _read_body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or "0")
            return self.rfile.read(n) if n > 0 else b""

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                if path in {"/", "/vision_playground.html", "/playground/vision_playground.html"}:
                    self._send_html()
                    return
                if path == "/v1/models":
                    self._send_json(
                        200,
                        {
                            "object": "list",
                            "data": [{"id": "abstractvision/playground", "object": "model"}],
                        },
                    )
                    return
                if path == "/v1/vision/models":
                    self._send_json(200, state.list_models())
                    return
                prefix = "/v1/vision/jobs/"
                if path.startswith(prefix):
                    job_id = path[len(prefix) :].strip("/")
                    if not job_id:
                        self._send_error_json(404, "Missing job id")
                        return
                    consume = (parse_qs(parsed.query).get("consume") or ["0"])[0] in {
                        "1",
                        "true",
                        "yes",
                    }
                    try:
                        self._send_json(200, state.get_job(job_id, consume=consume))
                    except KeyError:
                        self._send_error_json(404, f"Unknown job id: {job_id}")
                    return
                self._send_error_json(404, f"Not found: {path}")
            except Exception as e:
                traceback.print_exc()
                self._send_error_json(500, str(e))

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                body = self._read_body()
                if path == "/v1/vision/model/load":
                    payload = _parse_json_bytes(body)
                    model_id = str(payload.get("model_id") or payload.get("model") or "").strip()
                    self._send_json(200, state.load_model(model_id))
                    return
                if path == "/v1/vision/model/unload":
                    self._send_json(200, state.unload_active())
                    return
                if path == "/v1/vision/jobs/images/generations":
                    payload = _parse_json_bytes(body)
                    self._send_json(200, state.start_image_generation_job(payload))
                    return
                if path == "/v1/vision/jobs/images/edits":
                    content_type = self.headers.get("Content-Type") or ""
                    fields, files = _parse_multipart(content_type, body)
                    self._send_json(200, state.start_image_edit_job(fields, files))
                    return
                self._send_error_json(404, f"Not found: {path}")
            except json.JSONDecodeError as e:
                self._send_error_json(400, f"Invalid JSON: {e}")
            except ValueError as e:
                self._send_error_json(400, str(e))
            except Exception as e:
                traceback.print_exc()
                self._send_error_json(500, str(e))

    return Handler


def run_playground_server(config: PlaygroundServerConfig) -> int:
    state = PlaygroundState(config)
    handler_cls = _make_handler(state)
    server = ThreadingHTTPServer((config.host, int(config.port)), handler_cls)
    url = f"http://{config.host}:{int(config.port)}/vision_playground.html"
    print("AbstractVision playground")
    print(f"- UI:  {url}")
    print(f"- API: http://{config.host}:{int(config.port)}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0
