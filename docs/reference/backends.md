# Backends (execution engines)

AbstractVision executes tasks via a `VisionBackend` adapter (`src/abstractvision/backends/base_backend.py`).
`VisionManager` is intentionally thin and delegates to the configured backend (`src/abstractvision/vision_manager.py`).

See also:
- Getting started (REPL examples): `docs/getting-started.md`
- Configuration (env vars / CLI flags): `docs/reference/configuration.md`

## Support matrix (built-in backends)

| Backend | Implementation | Tasks implemented | Notes |
|---|---|---|---|
| OpenAI-compatible HTTP | `src/abstractvision/backends/openai_compatible.py` | `text_to_image`, `image_to_image` (+ optional `text_to_video`, `image_to_video`) | Stdlib-only (`urllib`). Video is **opt-in** via configured paths. |
| Diffusers (local) | `src/abstractvision/backends/huggingface_diffusers.py` | `text_to_image`, `image_to_image` | Heavy deps (Torch/Diffusers). Supports cache-only/offline mode. |
| stable-diffusion.cpp (local GGUF) | `src/abstractvision/backends/stable_diffusion_cpp.py` | `text_to_image`, `image_to_image` | Uses `sd-cli` if present, else `stable-diffusion-cpp-python`. Qwen Image GGUF needs VAE + LLM components. |

Notes:
- `multi_view_image` (`VisionManager.generate_angles`) is part of the public API, but **no built-in backend implements it yet** (all raise `CapabilityNotSupportedError` today).

## OpenAI-compatible HTTP backend

**When to use**
- You already run a service that exposes OpenAI-shaped endpoints (local or remote).
- You want to keep inference out-of-process.

**Core config**
- `base_url` (required): points to a `/v1`-style root, e.g. `http://localhost:1234/v1`
- `api_key` (optional): sent as `Authorization: Bearer ...`
- `model_id` (optional): forwarded as `model` in requests

Code pointers:
- Config: `OpenAICompatibleBackendConfig` (`src/abstractvision/backends/openai_compatible.py`)
- Backend: `OpenAICompatibleVisionBackend` (`src/abstractvision/backends/openai_compatible.py`)

**Video endpoints (optional)**
`OpenAICompatibleVisionBackend` only enables:
- `text_to_video` if `text_to_video_path` is set
- `image_to_video` if `image_to_video_path` is set

## Diffusers backend (local)

**When to use**
- You want local inference for Diffusers pipelines (Stable Diffusion, Qwen Image, FLUX, GLM-Image, …).

Code pointers:
- Config: `HuggingFaceDiffusersBackendConfig` (`src/abstractvision/backends/huggingface_diffusers.py`)
- Backend: `HuggingFaceDiffusersVisionBackend` (`src/abstractvision/backends/huggingface_diffusers.py`)

**Offline / cache-only mode**
The backend supports cache-only mode by setting `allow_download=False` (see config/env in `docs/reference/configuration.md`).

## stable-diffusion.cpp backend (local GGUF)

**When to use**
- You want to run GGUF diffusion models locally (e.g. Qwen Image GGUF).

Runtime modes (auto-selected):
- **CLI mode** via `sd-cli` (stable-diffusion.cpp executable) when available in `PATH`
- **Python mode** via `stable-diffusion-cpp-python` when `sd-cli` is not available

Code pointers:
- Config: `StableDiffusionCppBackendConfig` (`src/abstractvision/backends/stable_diffusion_cpp.py`)
- Backend: `StableDiffusionCppVisionBackend` (`src/abstractvision/backends/stable_diffusion_cpp.py`)

