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
| Diffusers (local) | [`huggingface_diffusers.py`](../../src/abstractvision/backends/huggingface_diffusers.py) | `text_to_image`, `image_to_image` | Requires `abstractvision[diffusers]`. Supports cache-only/offline mode. Local `text_to_video` groundwork exists but is currently experimental and disabled from the normal local surfaces. |
| MLX-Gen (local, Apple Silicon) | [`mflux.py`](../../src/abstractvision/backends/mflux.py) | `text_to_image`, `image_to_image`, `text_to_video`, `image_to_video` | Requires `abstractvision[mlx-gen]` (or compatibility extra `abstractvision[mflux]`, or `abstractvision[all-apple]`). Uses downloaded AbstractFramework q4/q8 MLX-Gen image preset snapshots plus official FIBO/Wan runtime snapshots from the Hugging Face cache; `ABSTRACTVISION_MODEL_DIR` is legacy migration input only. |
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
  - `abstractvision catalog --provider diffusers` (add `--all-targets` to compare engines)
  - Tip: `--provider diffusers` implies `--target diffusers` (you usually set one or the other).
- Download a curated Diffusers snapshot into the Hugging Face cache (legacy `~/models` trees auto-migrate when encountered):
  - `abstractvision download stable-diffusion --provider diffusers`
  - `abstractvision download sd1.4 --provider diffusers`
  - `abstractvision download sd1.5-inpaint --provider diffusers`
  - `abstractvision download instruct-pix2pix --provider diffusers`
  - `abstractvision download sdxl-base --provider diffusers`
  - `abstractvision download sdxl-refiner --provider diffusers`
  - `abstractvision download sdxl-inpaint --provider diffusers`
  - `abstractvision download sdxl-turbo --provider diffusers`
  - `abstractvision download sd-turbo --provider diffusers`
  - `abstractvision download sd3-medium --provider diffusers`
  - `abstractvision download sd3.5-medium --provider diffusers`
  - `abstractvision download sd3.5-large --provider diffusers`
  - `abstractvision download sd3.5-large-turbo --provider diffusers`
  - `abstractvision download ernie-image --provider diffusers`
  - `abstractvision download qwen-image --provider diffusers`
  - `abstractvision download qwen-image-edit-2511 --provider diffusers`
  - `abstractvision download flux2-dev --provider diffusers`
  - `abstractvision download flux2-klein-4b --provider diffusers`
  - `abstractvision download z-image-turbo --provider diffusers`

One-shot generation (uses the cached snapshot when present):
- `abstractvision t2i --provider diffusers --model qwen-image "a studio photo of a ceramic teapot"`

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
- Local Diffusers `GLM-Image` is temporarily disabled for both `text_to_image` and `image_to_image` pending the follow-up in [`../backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md`](../backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md).
- Local Diffusers `text_to_video` is currently experimental and disabled from the normal local surfaces.
- Local video export still requires an `ffmpeg` executable on `PATH` whenever a local backend emits frames for MP4 packaging.
- This backend is where model-specific defaults and constraints such as packaged step counts, guidance defaults, dimension constraints, and unsupported-parameter dropping are enforced for all callers.

## MLX-Gen backend (local Apple Silicon)

**When to use**
- You are on Apple Silicon and want local quantized MLX generation through the optional MLX-Gen runtime.
- You want the AbstractFramework-published q4/q8 prepared folders from the [AbstractFramework/mlx-gen Hugging Face collection](https://huggingface.co/collections/AbstractFramework/mlx-gen/).
- You want the official MLX-Gen 0.18.7+ FIBO snapshots (`briaai/FIBO`, `briaai/Fibo-lite`, `briaai/Fibo-Edit`, `briaai/Fibo-Edit-RMBG`).
- You want the official Prism ML Bonsai ternary 2-bit checkpoint (`prism-ml/bonsai-image-ternary-4B-mlx-2bit`) for very small local `text_to_image`.
- You want local Wan 2.2 TI2V video generation (`text_to_video` and first-frame `image_to_video`) through MLX-Gen 0.18.7+.

Install:
- `pip install "abstractvision[mlx-gen]"` (or `pip install "abstractvision[all-apple]"`)
- `pip install "abstractvision[mflux]"` remains a compatibility alias for older install instructions.

Model presets:
- See what's downloadable for your machine/engine:
  - `abstractvision catalog --provider mlx-gen` (add `--all` for full fallback list)
  - Tip: `--provider mlx-gen` implies `--target mlx` (you usually set one or the other).
- Download an exact published model repo into the Hugging Face cache (legacy `~/models` trees auto-migrate when encountered):
  - `abstractvision download AbstractFramework/flux.2-klein-4b-4bit --provider mlx-gen`
  - `abstractvision download AbstractFramework/flux.2-klein-4b-8bit --provider mlx-gen`
  - `abstractvision download AbstractFramework/qwen-image-2512-4bit --provider mlx-gen`
  - `abstractvision download AbstractFramework/qwen-image-edit-2511-4bit --provider mlx-gen`
  - `abstractvision download AbstractFramework/z-image-4bit --provider mlx-gen`
  - `abstractvision download AbstractFramework/z-image-turbo-4bit --provider mlx-gen`
  - `abstractvision download AbstractFramework/ernie-image-turbo-4bit --provider mlx-gen`
  - `abstractvision download AbstractFramework/ernie-image-turbo-8bit --provider mlx-gen`
  - `abstractvision download briaai/FIBO --provider mlx-gen`
  - `abstractvision download briaai/Fibo-lite --provider mlx-gen`
  - `abstractvision download briaai/Fibo-Edit --provider mlx-gen`
  - `abstractvision download briaai/Fibo-Edit-RMBG --provider mlx-gen`
  - `abstractvision download prism-ml/bonsai-image-ternary-4B-mlx-2bit --provider mlx-gen`
  - `abstractvision download Wan-AI/Wan2.2-TI2V-5B-Diffusers --provider mlx-gen`
- q4 repos are the default recommendation for memory efficiency. Use the exact
  matching `AbstractFramework/...-8bit` model id when quality is paramount. Qwen
  and ERNIE q4 prepared folders can mix q4 and q8 components, but remain the
  default prepared choice.
- `abstractvision t2i`, `abstractvision i2i`, and Python callers select q4/q8
  by exact model id. Quantization is metadata of the published folder, not a
  generation-time parameter.
- Bonsai ternary is a pre-packed low-bit MLX artifact, not a q4/q8 prepared folder. Use the exact repo id; guidance is fixed at 1.0 and negative prompts are ignored. The binary 1-bit Bonsai checkpoint is not surfaced because stock MLX cannot run it yet.
- Current shipped backend coverage includes `text_to_image` for FLUX.2 klein/base, Qwen Image, Z-Image, Z-Image Turbo, ERNIE Image Turbo, FIBO, Fibo-lite, and Bonsai ternary. `image_to_image` edits are implemented for FLUX.2 klein/base, Qwen Image Edit, ERNIE Image Turbo, FIBO, Fibo-lite, Fibo-Edit, and Fibo-Edit-RMBG; FIBO Edit snapshots support masks where the runtime supports them.

One-shot shell commands:

```bash
abstractvision t2i --provider mlx-gen --model AbstractFramework/qwen-image-2512-4bit "a studio product photo of a white ceramic mug with the AbstractFramework logo" --steps 20 --guidance-scale 1.0 --open
abstractvision i2i --provider mlx-gen --model AbstractFramework/qwen-image-edit-2511-4bit --image ./input.png "replace the background with a clean white studio setup" --steps 20 --guidance-scale 2.5 --strength 0.75 --open
abstractvision t2i --provider mlx-gen --model briaai/FIBO "a studio product photo of a white ceramic mug with the AbstractFramework logo" --steps 50 --guidance-scale 4.0 --open
abstractvision t2i --provider mlx-gen --model prism-ml/bonsai-image-ternary-4B-mlx-2bit "a bonsai tree in a quiet ceramic studio" --steps 4 --guidance-scale 1.0 --open
abstractvision i2i --provider mlx-gen --model briaai/Fibo-Edit --image ./input.png "remove the background and keep the object edges clean" --steps 20 --guidance-scale 4.0 --open
abstractvision t2v --provider mlx-gen --model Wan-AI/Wan2.2-TI2V-5B-Diffusers "a red fox walking through a snowy forest, cinematic" --frames 121 --fps 24 --steps 50 --guidance-scale 5.0 --open
abstractvision i2v --provider mlx-gen --model Wan-AI/Wan2.2-TI2V-5B-Diffusers --image ./first-frame.png "slow camera push-in" --frames 121 --fps 24 --steps 50 --guidance-scale 5.0 --open
```

MLX-Gen Wan video progress is surfaced as normalized progress events. Shell
commands and the interactive CLI render frame/step progress by default; Python
and AbstractCore callers can pass `on_progress(event)` to receive the same
events.

Interactive CLI/REPL commands:

```text
/backend mlx-gen AbstractFramework/qwen-image-2512-4bit
/t2i "a studio product photo of a white ceramic mug with the AbstractFramework logo" --steps 20 --guidance-scale 1.0 --open
/backend mlx-gen AbstractFramework/qwen-image-edit-2511-4bit
/i2i --image ./input.png "replace the background with a clean white studio setup" --steps 20 --guidance-scale 2.5 --strength 0.75 --open
/backend mlx-gen briaai/FIBO
/t2i "a studio product photo of a white ceramic mug with the AbstractFramework logo" --steps 50 --guidance-scale 4.0 --open
/backend mlx-gen prism-ml/bonsai-image-ternary-4B-mlx-2bit
/t2i "a bonsai tree in a quiet ceramic studio" --steps 4 --guidance-scale 1.0 --open
/backend mlx-gen Wan-AI/Wan2.2-TI2V-5B-Diffusers
/t2v "a red fox walking through a snowy forest, cinematic" --frames 121 --fps 24 --steps 50 --guidance-scale 5.0 --open
/i2v --image ./first-frame.png "slow camera push-in" --frames 121 --fps 24 --steps 50 --guidance-scale 5.0 --open
```

Config/env:
- `ABSTRACTVISION_PROVIDER=mlx-gen` (alias: `ABSTRACTVISION_BACKEND=mlx-gen`)
- `ABSTRACTVISION_MFLUX_MODEL=AbstractFramework/flux.2-klein-4b-4bit` (or routed ids like `mlx-gen/AbstractFramework/flux.2-klein-4b-4bit`)
- Optional: `ABSTRACTVISION_MFLUX_BASE_MODEL`, `ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD`, `ABSTRACTVISION_MODEL_DIR` (legacy migration root only)
- Legacy `mflux` provider values and routed ids remain accepted as compatibility aliases.

Non-curated MLX-Gen models:
- If you have an MLX-Gen-compatible Hugging Face repo id that is not in `model-presets`, you can still use it:
  - Pre-download it with `abstractvision download org/name` (HF cache) or `hf download org/name`
  - Set `ABSTRACTVISION_MFLUX_MODEL` to that repo id or local path (base model usually auto-infers; override with `ABSTRACTVISION_MFLUX_BASE_MODEL=qwen-image` if needed).

Runtime behavior notes:
- MLX-Gen request normalization is backend-level, so model constraints such as fixed guidance for turbo/distilled families, minimum step counts, and unsupported negative prompts are handled the same way through the CLI/REPL, playground API, and AbstractCore.
- Local MLX-Gen `image_to_image` is supported for FLUX.2 klein/base, Qwen Image Edit, ERNIE Image Turbo, FIBO, and FIBO Edit models. Edit strength is passed as `strength` and normalized to MLX-Gen's `image_strength` parameter where the runtime supports it.
- Local MLX-Gen video is implemented for `Wan-AI/Wan2.2-TI2V-5B-Diffusers`. Defaults are 1280x704, 121 frames, 50 steps, 24 fps, and guidance 5.0; lower values are useful only for smoke tests.
- Wan requests can pass `max_sequence_length` through Python `extra={...}` or CLI/REPL `--max-sequence-length`.
- Generation does not silently download model files. Missing-cache errors tell you which `abstractvision download ... --provider mlx-gen` or `mlxgen` preparation step is needed.

Code pointers:
- Config: `MLXGenBackendConfig` / compatibility alias `MFluxBackendConfig` ([`../../src/abstractvision/backends/mflux.py`](../../src/abstractvision/backends/mflux.py))
- Backend: `MLXGenVisionBackend` / compatibility alias `MFluxVisionBackend` ([`../../src/abstractvision/backends/mflux.py`](../../src/abstractvision/backends/mflux.py))

## stable-diffusion.cpp backend (local GGUF/checkpoints)

**When to use**
- You want to run single-file Stable Diffusion checkpoints/GGUF or component-based GGUF diffusion models locally.

Runtime modes (auto-selected):
- **CLI mode** via `sd-cli` (stable-diffusion.cpp executable) when available in `PATH`
- **Python mode** via `stable-diffusion-cpp-python` when `sd-cli` is not available

Notes:
- If you care about **GPU acceleration** (macOS **Metal**, NVIDIA **CUDA**, etc.), prefer **CLI mode** via `sd-cli`.
- Python bindings run whatever backend the installed wheel was built with. On macOS, that often means **CPU-only**, so FLUX/Qwen-class models can be extremely slow.
- Operators who want to hide GGUF presets (and reject GGUF execution) on macOS can set `ABSTRACTVISION_DISABLE_GGUF_ON_MACOS=1`.
- The optional python binding is constrained below `0.4.6` because that sdist
  currently misses vendored CMake files needed by native Linux builds.
- Interactive CLI selection supports both `/backend sdcpp <model_key|model.gguf|model.safetensors> [sd_cli_path]` and
  `/backend sdcpp <diffusion_model.gguf> <vae.safetensors> <llm.gguf> [sd_cli_path]`.
- One-shot CLI, playground, and the AbstractCore plugin can also accept curated `sdcpp` model keys such as
  `flux2-klein-base-4b` or `qwen-image` after `abstractvision download ... --provider sdcpp`.
- Python code and AbstractCore plugin configuration can also pass component paths such as `clip_l`, `clip_g`, `t5xxl`, `llm_vision`, plus `extra_args`, `timeout_s`, and `cwd`.

Code pointers:
- Config: `StableDiffusionCppBackendConfig` ([`../../src/abstractvision/backends/stable_diffusion_cpp.py`](../../src/abstractvision/backends/stable_diffusion_cpp.py))
- Backend: `StableDiffusionCppVisionBackend` ([`../../src/abstractvision/backends/stable_diffusion_cpp.py`](../../src/abstractvision/backends/stable_diffusion_cpp.py))
