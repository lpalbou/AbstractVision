# Backends (execution engines)

AbstractVision executes tasks via a `VisionBackend` adapter ([`../../src/abstractvision/backends/base_backend.py`](../../src/abstractvision/backends/base_backend.py)).
`VisionManager` is intentionally thin and delegates to the configured backend ([`../../src/abstractvision/vision_manager.py`](../../src/abstractvision/vision_manager.py)).

See also:
- Getting started (REPL examples): [docs/getting-started.md](../getting-started.md)
- Configuration (env vars / CLI flags): [docs/reference/configuration.md](configuration.md)

## Support matrix (built-in backends)

| Backend | Implementation | Tasks implemented | Notes |
|---|---|---|---|
| OpenAI-compatible HTTP | [`openai_compatible.py`](../../src/abstractvision/backends/openai_compatible.py) | `text_to_image`, `image_to_image` (+ optional `text_to_video`, `image_to_video`) | Stdlib-only (`urllib`). Video is **opt-in** via configured paths. |
| Diffusers (local) | [`huggingface_diffusers.py`](../../src/abstractvision/backends/huggingface_diffusers.py) | `text_to_image`, `image_to_image` | Requires `abstractvision[diffusers]`. Supports cache-only/offline mode. |
| stable-diffusion.cpp (local GGUF/checkpoints) | [`stable_diffusion_cpp.py`](../../src/abstractvision/backends/stable_diffusion_cpp.py) | `text_to_image`, `image_to_image` | Uses external `sd-cli` if present, else `abstractvision[sdcpp]` python bindings. Start with single-file Stable Diffusion models; Qwen/FLUX GGUF may need VAE + LLM components. |

Notes:
- `multi_view_image` (`VisionManager.generate_angles`) is part of the public API, but **no built-in backend implements it yet** (all raise `CapabilityNotSupportedError` today).
- Backends may also expose best-effort `get_capabilities()`, `preload()`, `unload()`, `generate_image_with_progress(...)`, and `edit_image_with_progress(...)` hooks via the shared `VisionBackend` contract.

## OpenAI-compatible HTTP backend

**When to use**
- You already run a service that exposes OpenAI-shaped endpoints (local or remote).
- You want to keep inference out-of-process.

**Core config**
- `base_url` (required): points to a `/v1`-style root, e.g. `http://localhost:1234/v1`
- `api_key` (optional): sent as `Authorization: Bearer ...`
- `model_id` (optional): forwarded as `model` in requests
- `models_path` (default `/models`): provider catalog path for explicit model listing

Request shape:
- Unknown/local endpoints receive local extension fields when provided, including `steps`, `seed`, `guidance_scale`, `negative_prompt`, `width`, and `height`.
- Real OpenAI-looking endpoints and known OpenAI image models use the narrower OpenAI request shape; GPT image models do not receive unsupported local-only fields such as `steps`, `seed`, or `guidance_scale`.

Provider model catalogs:
- `OpenAICompatibleVisionBackend.list_provider_models(...)` queries `GET /models` by default.
- `VisionManager.list_provider_models(...)` delegates to the configured backend.
- The AbstractCore plugin exposes the same catalog through `llm.vision.list_provider_models(...)`.
- CLI examples: `abstractvision provider-models --openai --task text_to_image` and `abstractvision provider-models --base-url http://localhost:1234/v1 --task text_to_image`.
- Listing is explicit; AbstractVision does not use provider catalogs to silently select a model.

Code pointers:
- Config: `OpenAICompatibleBackendConfig` ([`../../src/abstractvision/backends/openai_compatible.py`](../../src/abstractvision/backends/openai_compatible.py))
- Backend: `OpenAICompatibleVisionBackend` ([`../../src/abstractvision/backends/openai_compatible.py`](../../src/abstractvision/backends/openai_compatible.py))

**Video endpoints (optional)**
`OpenAICompatibleVisionBackend` only enables:
- `text_to_video` if `text_to_video_path` is set
- `image_to_video` if `image_to_video_path` is set

## Diffusers backend (local)

**When to use**
- You want local inference for Diffusers pipelines.
- Start with `runwayml/stable-diffusion-v1-5` for the lowest-risk local test.
- Move to `black-forest-labs/FLUX.2-klein-4B` after that if you want a newer non-gated model and can install Diffusers `main`.

Install:
- `pip install "abstractvision[diffusers]"`
- For newer/unreleased pipeline classes: `pip install "abstractvision[diffusers-dev]"` plus Diffusers from source.

Code pointers:
- Config: `HuggingFaceDiffusersBackendConfig` ([`../../src/abstractvision/backends/huggingface_diffusers.py`](../../src/abstractvision/backends/huggingface_diffusers.py))
- Backend: `HuggingFaceDiffusersVisionBackend` ([`../../src/abstractvision/backends/huggingface_diffusers.py`](../../src/abstractvision/backends/huggingface_diffusers.py))

**Offline / cache-only mode**
The Python backend and REPL are cache-only by default (`allow_download=False`). Pre-download model weights separately,
or set `allow_download=True` / `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1` when runtime downloads are desired (see
config/env in [docs/reference/configuration.md](configuration.md)).

Config fields:
- `model_id`, `device`, `torch_dtype`
- `allow_download`, `auto_retry_fp32`
- `cache_dir`, `revision`, `variant`
- `use_safetensors`, `low_cpu_mem_usage`

## stable-diffusion.cpp backend (local GGUF/checkpoints)

**When to use**
- You want to run single-file Stable Diffusion checkpoints/GGUF or component-based GGUF diffusion models locally.

Runtime modes (auto-selected):
- **CLI mode** via `sd-cli` (stable-diffusion.cpp executable) when available in `PATH`
- **Python mode** via `stable-diffusion-cpp-python` when `sd-cli` is not available

Notes:
- If you care about **GPU acceleration** (macOS **Metal**, NVIDIA **CUDA**, etc.), prefer **CLI mode** via `sd-cli`.
- Python bindings run whatever backend the installed wheel was built with. On macOS, that often means **CPU-only**, so FLUX/Qwen-class models can be extremely slow.
- The optional python binding is constrained below `0.4.6` because that sdist
  currently misses vendored CMake files needed by native Linux builds.
- REPL selection supports both `/backend sdcpp <model.gguf|model.safetensors> [sd_cli_path]` and
  `/backend sdcpp <diffusion_model.gguf> <vae.safetensors> <llm.gguf> [sd_cli_path]`.
- Python code and AbstractCore plugin configuration can also pass component paths such as `clip_l`, `clip_g`, `t5xxl`, `llm_vision`, plus `extra_args`, `timeout_s`, and `cwd`.

Code pointers:
- Config: `StableDiffusionCppBackendConfig` ([`../../src/abstractvision/backends/stable_diffusion_cpp.py`](../../src/abstractvision/backends/stable_diffusion_cpp.py))
- Backend: `StableDiffusionCppVisionBackend` ([`../../src/abstractvision/backends/stable_diffusion_cpp.py`](../../src/abstractvision/backends/stable_diffusion_cpp.py))
