# Backends (execution engines)

AbstractVision executes tasks via a `VisionBackend` adapter ([`../../src/abstractvision/backends/base_backend.py`](../../src/abstractvision/backends/base_backend.py)).
`VisionManager` is intentionally thin and delegates to the configured backend ([`../../src/abstractvision/vision_manager.py`](../../src/abstractvision/vision_manager.py)).

See also:
- Getting started (interactive CLI examples): [docs/getting-started.md](../getting-started.md)
- Configuration (env vars / CLI flags): [docs/reference/configuration.md](configuration.md)

## Support matrix (built-in backends)

| Backend | Implementation | Tasks implemented | Notes |
|---|---|---|---|
| OpenAI-compatible HTTP | [`openai_compatible.py`](../../src/abstractvision/backends/openai_compatible.py) | `text_to_image`, `image_to_image` (+ optional `text_to_video`, `image_to_video`) | Stdlib-only (`urllib`). Video is **opt-in** via configured paths. |
| Diffusers (local) | [`huggingface_diffusers.py`](../../src/abstractvision/backends/huggingface_diffusers.py) | `text_to_image`, `image_to_image`, `text_to_video` (CogVideoX-2b family) | Requires `abstractvision[diffusers]`. Supports cache-only/offline mode. Local MP4 export uses `ffmpeg` from `PATH`. |
| MFLUX (local, Apple Silicon) | [`mflux.py`](../../src/abstractvision/backends/mflux.py) | `text_to_image`, `image_to_image` | Requires `abstractvision[mflux]` (or `abstractvision[all-apple]`). Uses downloaded 8-bit MLX/MFLUX preset snapshots from the Hugging Face cache; `ABSTRACTVISION_MODEL_DIR` is legacy migration input only. |
| stable-diffusion.cpp (local GGUF/checkpoints) | [`stable_diffusion_cpp.py`](../../src/abstractvision/backends/stable_diffusion_cpp.py) | `text_to_image`, `image_to_image` | Uses external `sd-cli` if present, else `abstractvision[sdcpp]` python bindings. Start with single-file Stable Diffusion models; curated Qwen/FLUX GGUF presets now auto-resolve required VAE + LLM companions from the cache. |

Notes:
- `multi_view_image` (`VisionManager.generate_angles`) is part of the public API, but **no built-in backend implements it yet** (all raise `CapabilityNotSupportedError` today).
- Backends may also expose best-effort `get_capabilities()`, `preload()`, `unload()`, `generate_image_with_progress(...)`, `edit_image_with_progress(...)`, and video progress hooks via the shared `VisionBackend` contract.
- Backends may also implement `normalize_image_generation_request(...)`, `normalize_image_edit_request(...)`, `normalize_video_generation_request(...)`, and `normalize_image_to_video_request(...)`. `VisionManager`, the CLI/REPL, the playground API, and the AbstractCore plugin all route through those hooks so model-specific defaults and constraints are applied consistently instead of being hard-coded in one surface.

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

Model downloads (curated):
- See what's downloadable for Diffusers:
  - `abstractvision model-catalog --provider diffusers` (add `--all-targets` to compare engines)
  - Tip: `--provider diffusers` implies `--target diffusers` (you usually set one or the other).
- Download a curated Diffusers snapshot into the Hugging Face cache (legacy `~/models` trees auto-migrate when encountered):
  - `abstractvision download-model stable-diffusion --provider diffusers`
  - `abstractvision download-model sd1.4 --provider diffusers`
  - `abstractvision download-model sd1.5-inpaint --provider diffusers`
  - `abstractvision download-model instruct-pix2pix --provider diffusers`
  - `abstractvision download-model sdxl-base --provider diffusers`
  - `abstractvision download-model sdxl-refiner --provider diffusers`
  - `abstractvision download-model sdxl-inpaint --provider diffusers`
  - `abstractvision download-model sdxl-turbo --provider diffusers`
  - `abstractvision download-model sd-turbo --provider diffusers`
  - `abstractvision download-model sd3-medium --provider diffusers`
  - `abstractvision download-model sd3.5-medium --provider diffusers`
  - `abstractvision download-model sd3.5-large --provider diffusers`
  - `abstractvision download-model sd3.5-large-turbo --provider diffusers`
  - `abstractvision download-model ernie-image --provider diffusers`
  - `abstractvision download-model qwen-image --provider diffusers`
  - `abstractvision download-model qwen-image-edit --provider diffusers`
  - `abstractvision download-model glm-image --provider diffusers`
  - `abstractvision download-model flux2-dev --provider diffusers`
  - `abstractvision download-model cogvideox-2b --provider diffusers`
  - `abstractvision download-model flux2-klein-4b --provider diffusers`
  - `abstractvision download-model z-image-turbo --provider diffusers`

One-shot generation (uses the cached snapshot when present):
- `abstractvision t2i --provider diffusers --model qwen-image "a studio photo of a ceramic teapot"`
- `abstractvision t2v --provider diffusers --model zai-org/CogVideoX-2b --diffusers-device mps --diffusers-torch-dtype float16 --num-frames 9 --steps 1 "a red fox walking through a snowy forest, cinematic"`

Code pointers:
- Config: `HuggingFaceDiffusersBackendConfig` ([`../../src/abstractvision/backends/huggingface_diffusers.py`](../../src/abstractvision/backends/huggingface_diffusers.py))
- Backend: `HuggingFaceDiffusersVisionBackend` ([`../../src/abstractvision/backends/huggingface_diffusers.py`](../../src/abstractvision/backends/huggingface_diffusers.py))

**Offline / cache-only mode**
The Python backend and interactive CLI are cache-only by default (`allow_download=False`). Pre-download model weights separately,
or set `allow_download=True` / `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1` when runtime downloads are desired (see
config/env in [docs/reference/configuration.md](configuration.md)).

Config fields:
- `model_id`, `device`, `torch_dtype`
- `allow_download`, `auto_retry_fp32`
- `cache_dir`, `revision`, `variant`
- `use_safetensors`, `low_cpu_mem_usage`

Runtime behavior notes:
- The Diffusers backend now reads packaged registry task metadata for known models when it normalizes requests.
- Local Diffusers `text_to_video` is intentionally narrow today: the first shipped path is the CogVideoX-2b family (`zai-org/CogVideoX-2b` / `THUDM/CogVideoX-2b`).
- Local video export requires an `ffmpeg` executable on `PATH` so generated frames can be packaged as MP4 artifact outputs.
- This is where model-specific defaults and constraints such as GLM `guidance_scale=1.5`, CogVideoX `720x480` / `8 fps` defaults, task-aware edit support, and unsupported-parameter dropping are enforced for all callers.

## MFLUX backend (local Apple Silicon)

**When to use**
- You are on Apple Silicon and want local 8-bit MLX generation through the optional MFLUX runtime.

Install:
- `pip install "abstractvision[mflux]"` (or `pip install "abstractvision[all-apple]"`)

Model presets:
- See what's downloadable for your machine/engine:
  - `abstractvision model-catalog --provider mflux` (add `--all` for full fallback list)
  - Tip: `--provider mflux` implies `--target mlx` (you usually set one or the other).
- Download a curated 8-bit preset into the Hugging Face cache (legacy `~/models` trees auto-migrate when encountered):
  - `abstractvision download-model flux2-klein-4b --provider mflux`
  - `abstractvision download-model flux2-klein-9b --provider mflux`
  - `abstractvision download-model qwen-image --provider mflux`
  - `abstractvision download-model z-image-turbo --provider mflux`
- Current shipped backend coverage is limited to those curated MFLUX families: `flux2-klein-4b`, `flux2-klein-9b`, `qwen-image`, and `z-image-turbo`.

Config/env:
- `ABSTRACTVISION_PROVIDER=mflux` (alias: `ABSTRACTVISION_BACKEND=mflux`)
- `ABSTRACTVISION_MFLUX_MODEL=flux2-klein-4b` (or routed ids like `mflux/flux2-klein-4b`)
- Optional: `ABSTRACTVISION_MFLUX_BASE_MODEL`, `ABSTRACTVISION_MFLUX_QUANTIZE`, `ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD`, `ABSTRACTVISION_MODEL_DIR` (legacy migration root only)

Non-curated MFLUX models:
- If you have an MFLUX-compatible Hugging Face repo id that is not in `model-presets`, you can still use it:
  - Pre-download it with `abstractvision download-model org/name` (HF cache) or `hf download org/name`
  - Set `ABSTRACTVISION_MFLUX_MODEL` to that repo id or local path (base model usually auto-infers; override with `ABSTRACTVISION_MFLUX_BASE_MODEL=qwen-image` if needed).

Runtime behavior notes:
- MFLUX request normalization is also backend-level, so distilled FLUX-family constraints such as `guidance_scale=1.0`, minimum step counts, and unsupported negative prompts are handled the same way through the CLI/REPL, playground API, and AbstractCore.

Code pointers:
- Config: `MFluxBackendConfig` ([`../../src/abstractvision/backends/mflux.py`](../../src/abstractvision/backends/mflux.py))
- Backend: `MFluxVisionBackend` ([`../../src/abstractvision/backends/mflux.py`](../../src/abstractvision/backends/mflux.py))

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
- Interactive CLI selection supports both `/backend sdcpp <model_key|model.gguf|model.safetensors> [sd_cli_path]` and
  `/backend sdcpp <diffusion_model.gguf> <vae.safetensors> <llm.gguf> [sd_cli_path]`.
- One-shot CLI, playground, and the AbstractCore plugin can also accept curated `sdcpp` model keys such as
  `flux2-klein-base-4b` or `qwen-image` after `abstractvision download-model ... --provider sdcpp`.
- Python code and AbstractCore plugin configuration can also pass component paths such as `clip_l`, `clip_g`, `t5xxl`, `llm_vision`, plus `extra_args`, `timeout_s`, and `cwd`.

Code pointers:
- Config: `StableDiffusionCppBackendConfig` ([`../../src/abstractvision/backends/stable_diffusion_cpp.py`](../../src/abstractvision/backends/stable_diffusion_cpp.py))
- Backend: `StableDiffusionCppVisionBackend` ([`../../src/abstractvision/backends/stable_diffusion_cpp.py`](../../src/abstractvision/backends/stable_diffusion_cpp.py))
