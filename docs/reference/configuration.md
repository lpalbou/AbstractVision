# Configuration (CLI)

AbstractVision configuration is intentionally simple:

- In Python, you configure backends by instantiating backend config objects (see [docs/reference/backends.md](backends.md)).
- The CLI/playground reads `ABSTRACTVISION_*` environment variables to set defaults ([`../../src/abstractvision/cli.py`](../../src/abstractvision/cli.py), [`../../src/abstractvision/playground_server.py`](../../src/abstractvision/playground_server.py)). The interactive command is `abstractvision cli` (legacy alias: `abstractvision repl`).
- The AbstractCore capability plugin also reads `owner.config` plus a small set of standard OpenAI environment aliases. Plugin-only aliases are called out below.

See also:
- Getting started (examples): [docs/getting-started.md](../getting-started.md)
- Backends: [docs/reference/backends.md](backends.md)

## CLI commands (overview)

Implemented in [`../../src/abstractvision/cli.py`](../../src/abstractvision/cli.py):

- `abstractvision models` — list known registry model ids (one per line); `--json` adds each model's downloads with the fields `show-model` prints
- `abstractvision tasks` — list known tasks
- `abstractvision show-model <id> [--json]` — print a model’s tasks + params and, for each download, its size, whether the Hugging Face repository is gated (and the licence to accept), whether it is in the local cache (and its size there), the `memory_mode` values this host can run and, for MLX-Gen packages whose capability rows publish measured runs, the peak memory of each mode and the smallest memory it was tested on; for MiniMax-H3 it also lists the schedules (`base_model`) and the adapter each loads
- `abstractvision catalog [--provider ...] [--task ...] [--json]` — browse the downloadable models: target, engine, bits (`*` marks the recommended package), access (gated or open), cache state and the memory modes this host can run
- `abstractvision download <model id or repo id> [--bits N] [--base-model SCHEDULE] [--dry-run]` — download a curated preset into the Hugging Face cache (see [Downloads](#downloads))
- `abstractvision adapters --provider mlx-gen --model <id> --task <task>` — list backend-discovered cached adapters for one exact route
- `abstractvision provider-models --openai --task text_to_image` — explicitly query the official OpenAI `/models` catalog
- `abstractvision provider-models --base-url http://localhost:1234/v1 --task text_to_image` — explicitly query an OpenAI-compatible provider catalog
- `abstractvision cli` — interactive testing (supports `openai`, `diffusers`, `mlx-gen`, `sdcpp`; legacy alias: `abstractvision repl`; `mflux` is accepted as a compatibility alias)
- `abstractvision playground [--host 127.0.0.1] [--port 8091]` — self-contained local web UI and `/v1/vision/*` API
- `abstractvision serve [--host 127.0.0.1] [--port 8091]` — alias for `abstractvision playground`
- `abstractvision t2i ...` / `abstractvision i2i ...` / `abstractvision t2v ...` / `abstractvision i2v ...` — one-shot commands using the configured provider/backend (`openai`/`openai-compatible` by default; also supports local `diffusers`, `mlx-gen`, and `sdcpp`; MLX-Gen provides image progress, shared LoRA adapters, multi-reference edits, SeedVR2 upscaling, and Wan `t2v`/`i2v`, including A14B task-specific checkpoints)
- `abstractvision upscale ...` — one-shot SeedVR2 image upscaling; defaults to `--provider mlx-gen --model AbstractFramework/seedvr2-3b-8bit --resolution 2x --softness 0.25` and supports `--scale`, `--resolution`, `--softness`, `--seed`, optional source-weight `--quantize`, and `--vae-tiling`

Note:
- `abstractvision t2i` / `abstractvision i2i` / `abstractvision t2v` / `abstractvision i2v` run on this machine's local engine by default with FLUX.2 [klein] 4B in 8 bits (MLX-Gen on Apple silicon, stable-diffusion.cpp on an NVIDIA card, Diffusers on a processor); `--base-url` selects the OpenAI-compatible HTTP backend and `--provider` picks another local engine.
- Local Diffusers requires `abstractvision[diffusers]`. stable-diffusion.cpp python binding fallback requires `abstractvision[sdcpp]`; external `sd-cli` can be used without the binding.
- One-shot image commands write to `ABSTRACTVISION_STORE_DIR` or `--store-dir` and print JSON artifact metadata followed by the local content path. They do not take an `--output` path flag.
- One-shot generation/edit/video commands support `--count` plus optional `--seeds` for reproducible batch orchestration above the exact single-run request surface.
- `width`/`height` are optional request overrides, not global defaults. Leave
  them unset to let the selected provider/model choose its default or `auto`
  size. Some models accept quick-test sizes such as `512x512`; others only
  accept provider-declared sizes such as `1024x1024`, `1024x1536`,
  `1536x1024`, or `auto`.

Examples:

```bash
abstractvision t2i --provider mlx-gen --model AbstractFramework/qwen-image-2512-4bit "a studio product photo of a white ceramic mug" --steps 20 --guidance-scale 1.0 --open
abstractvision i2i --provider mlx-gen --model AbstractFramework/qwen-image-edit-2511-4bit --image ./input.png "replace the background with a clean white studio setup" --steps 20 --guidance-scale 2.5 --strength 0.75 --open
abstractvision t2i --provider mlx-gen --model briaai/FIBO "a studio product photo of a white ceramic mug" --steps 50 --guidance-scale 4.0 --open
abstractvision t2i --provider mlx-gen --model prism-ml/bonsai-image-ternary-4B-mlx-2bit "a bonsai tree in a quiet ceramic studio" --steps 4 --guidance-scale 1.0 --open
abstractvision i2i --provider mlx-gen --model briaai/Fibo-Edit --image ./input.png "remove the background and keep the object edges clean" --steps 20 --guidance-scale 4.0 --open
abstractvision upscale --provider mlx-gen --model AbstractFramework/seedvr2-3b-8bit --image ./input.png --resolution 2x --softness 0.25 --open
abstractvision t2v --provider mlx-gen --model AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit "a red fox walking through a snowy forest, cinematic" --width 432 --height 240 --frames 41 --fps 10 --steps 20 --guidance-scale 4.0 --guidance-2 3.0 --open
abstractvision i2v --provider mlx-gen --model AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit --image ./first-frame.png "slow camera push-in" --width 432 --height 240 --frames 41 --fps 10 --steps 20 --guidance-scale 3.5 --guidance-2 3.5 --open
```

For MLX-Gen Wan, one-shot `t2v` and `i2v` show denoise-step progress with
frame context on stderr by default. `upscale` also shows denoise-step progress
by default. Add `--no-progress` when you need quiet shell output.

`t2i`, `i2i`, `upscale`, `t2v` and `i2v` take `--memory-mode default|low|lowest`
(default: `$ABSTRACTVISION_MEMORY_MODE`, else `default`) to lower the run's
memory peak at some cost in speed: MLX-Gen low-RAM mode, Diffusers CPU offload
on CUDA, stable-diffusion.cpp CPU offload of parameters. A mode the provider
cannot honour is refused. See [Memory modes](../api.md#memory-modes-memory_mode).

```bash
abstractvision t2v --provider mlx-gen --model AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit "a red fox walking through a snowy forest, cinematic" --memory-mode low --open
```

`t2i`, `i2i`, `t2v` and `i2v` take `--quantize BITS` (load-time quantization of a
source snapshot on MLX-Gen; refused elsewhere). `t2v` and `i2v` take the soundtrack
options of MiniMax-H3: `--soundscape`, `--music`, `--no-audio`, `--audio-shift`
(`--flow-shift` is its video shift):

```bash
abstractvision t2v --provider mlx-gen --model AbstractFramework/minimax-h3-8bit --mflux-base-model minimax-h3-turbo-544p "A red fox trots through fresh snow at dawn." --soundscape "Paws in dry snow, faint wind." --music "Sparse piano." --width 640 --height 352 --frames 124 --seed 42 --open
```

## Interactive CLI backend selection

Inside `abstractvision cli`:

- `/backend openai <base_url> [api_key] [model_id]`
- `/provider-models [--task text_to_image] [--json]` — query the configured OpenAI-compatible provider catalog
- `/backend mlx-gen <preset_or_local_path> [base_model]`
- `/backend diffusers <model_id_or_path> [device] [torch_dtype]`
- `/backend sdcpp <model_key|model.gguf|model.safetensors> [sd_cli_path]`
- `/backend sdcpp <diffusion_model.gguf> <vae.safetensors> <llm.gguf> [sd_cli_path]`

Run `/help` in the interactive CLI to see the full command list (generated by `_repl_help()` in [`../../src/abstractvision/cli.py`](../../src/abstractvision/cli.py)).

Interactive generation commands include `/t2i`, `/i2i`, `/t2v`, and `/i2v`. The local Diffusers `text_to_video` path remains experimental and disabled from the normal local surfaces, while MLX-Gen Wan supports local `t2v` and `i2v`. Interactive image commands follow the same size rule as one-shot commands: omit width and height unless you intentionally want a specific backend-supported size.

Examples:

```text
/backend mlx-gen AbstractFramework/qwen-image-2512-4bit
/t2i "a studio product photo of a white ceramic mug" --steps 20 --guidance-scale 1.0 --open
/backend mlx-gen AbstractFramework/qwen-image-edit-2511-4bit
/i2i --image ./input.png "replace the background with a clean white studio setup" --steps 20 --guidance-scale 2.5 --strength 0.75 --open
/backend mlx-gen briaai/FIBO
/t2i "a studio product photo of a white ceramic mug" --steps 50 --guidance-scale 4.0 --open
/backend mlx-gen prism-ml/bonsai-image-ternary-4B-mlx-2bit
/t2i "a bonsai tree in a quiet ceramic studio" --steps 4 --guidance-scale 1.0 --open
/backend mlx-gen AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit
/t2v "a red fox walking through a snowy forest, cinematic" --width 432 --height 240 --frames 41 --fps 10 --steps 20 --guidance-scale 4.0 --guidance-2 3.0 --open
/backend mlx-gen AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit
/i2v --image ./first-frame.png "slow camera push-in" --width 432 --height 240 --frames 41 --fps 10 --steps 20 --guidance-scale 3.5 --guidance-2 3.5 --open
```

Interactive `/t2v` and `/i2v` also show denoise-step video progress by default
and accept `--no-progress`.

Important runtime note:
- Model-specific request adjustments are not implemented only in the interactive UI. The shared backend normalization hooks are applied by `VisionManager` and by the playground API server, so the same parameter corrections reach `abstractvision cli`, the legacy `repl` alias, one-shot CLI commands, the playground, and the AbstractCore plugin.

## Environment variables

The interactive CLI state object (`_ReplState` in [`../../src/abstractvision/cli.py`](../../src/abstractvision/cli.py)) reads the common and backend env vars below. The playground server reads the same backend family plus a few playground-only Diffusers cache/version vars through `PlaygroundServerConfig` in [`../../src/abstractvision/playground_server.py`](../../src/abstractvision/playground_server.py).

### Common

- `ABSTRACTVISION_PROVIDER` — preferred backend selector: `openai`, `openai-compatible`, `diffusers`, `mlx-gen`, or `sdcpp` (alias: `ABSTRACTVISION_BACKEND`; `mflux` remains accepted as a compatibility alias)
  - AbstractCore plugin default: `openai` using `https://api.openai.com/v1` plus `OPENAI_API_KEY`
  - AbstractCore compatibility: selecting `abstractvision:openai-compatible` directly, or setting `OPENAI_BASE_URL` to a non-OpenAI endpoint, keeps compatible-endpoint semantics
  - if unset and `OPENAI_BASE_URL` is set, the interactive CLI/playground default to `openai`
  - if unset and no base URL is configured, no backend is selected until you use `/backend ...` or load a model explicitly
- `ABSTRACTVISION_STORE_DIR` — local artifact output directory (default: `~/.abstractvision/assets`)
- `ABSTRACTVISION_TIMEOUT_S` — optional timeout for OpenAI-compatible catalog/control calls. Image and video generation calls do not have a default timeout.
- `ABSTRACTVISION_MODEL` — preferred model selector for the current backend/provider (alias: `ABSTRACTVISION_MODEL_ID`)
- `ABSTRACTVISION_MODEL_ID` — model id for the current backend in the interactive CLI (legacy):
  - `openai`: sent as `model` in HTTP requests (optional; server-dependent)
  - `diffusers`: Diffusers model id or local path
- `ABSTRACTVISION_MODEL_DIR` — legacy preset root imported into the Hugging Face cache when older installs are migrated
- `ABSTRACTVISION_CAPABILITIES_MODEL_ID` — optional capability-gating model id (must exist in the registry)
- `ABSTRACTVISION_MEMORY_MODE` — `default`, `low` or `lowest`: the memory mode of every generation request that names none, in the CLI, the playground, Python and the AbstractCore plugin; a request value wins. A mode that comes from `$ABSTRACTVISION_MEMORY_MODE` is one value for every backend of the process, so a backend that cannot honour it for its model and device runs the lowest-memory mode it lists that is not below it (`lowest` becomes `low` where `low` is listed, else `default`) and logs that once; a mode passed on the request is refused instead. Warmups (`preload`) follow the same rule. See [Memory modes](../api.md#memory-modes-memory_mode).

Note:
- Curated `abstractvision download ...` presets are written to the Hugging Face cache by default, and older `~/models/<preset>` trees are imported into that same cache on first use.

### Downloads

`abstractvision download` takes an exact package repo id (`AbstractFramework/minimax-h3-4bit`), its
short name (`minimax-h3-4bit`), or a model id (`MiniMaxAI/MiniMax-H3`). A model id with several
packages selects the one the catalog marks as recommended (`*` in `catalog`), or the width
`--bits 4|8|16` names; `--allow-non-8bit` selects a model's source snapshot. A model id with several
packages and no recommendation asks for an exact repo id or `--bits`. On Apple silicon `--target auto`
is `mlx` (MLX-Gen packages first).

- Size and disk space: before anything is fetched, the files the preset selects are listed on the
  Hugging Face Hub (metadata only) and compared with the free space of the cache disk; a download that
  does not fit stops with a message naming its size. Without the listing (offline), the size is
  reported as unknown and the disk check is skipped.
- Resume: files already in the cache, complete or partial, are kept; running the same command again
  resumes an interrupted download. A download that ends with files missing says so and how to resume.
- Gated repositories (for example the `AbstractFramework/minimax-h3-*` packages): accept the licence on
  the repository page while logged in to Hugging Face, then authenticate (`hf auth login`, `HF_TOKEN`, or
  `--token`). A refused download ends with a message naming the page and the licence.
- `--base-model <schedule>`: for MLX-Gen models whose schedules load an adapter, also fetches that
  adapter into mlx-gen's LoRA cache. The MiniMax-H3 Turbo schedules `minimax-h3-turbo-544p` and
  `minimax-h3-turbo` load a lightx2v Turbo LoRA; `minimax-h3` loads none. Without `--base-model`, the
  command prints the schedules whose adapter is not cached.
- `--dry-run`: prints the selection, access, cache state, the size from the Hub listing, free disk
  space and adapter state, and downloads nothing.
- Generation never downloads gated MiniMax-H3 weights or their adapters, even with
  `ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD=1`.

```bash
abstractvision show-model MiniMaxAI/MiniMax-H3
abstractvision download MiniMaxAI/MiniMax-H3 --dry-run
abstractvision download MiniMaxAI/MiniMax-H3 --base-model minimax-h3-turbo-544p
abstractvision download MiniMaxAI/MiniMax-H3 --bits 4
```

### OpenAI / OpenAI-Compatible HTTP Backend

- `OPENAI_BASE_URL` — optional OpenAI-compatible `/v1` endpoint override; when unset, the official OpenAI profile uses `https://api.openai.com/v1`
- `OPENAI_API_KEY` — bearer token for official OpenAI and compatible `/v1` endpoints that require auth
- `ABSTRACTVISION_MODEL_ID` — optional remote model id/name (see also “Common”)
- `OPENAI_IMAGE_MODEL_ID` / `OPENAI_IMAGE_MODEL` — plugin-only OpenAI model aliases when `ABSTRACTVISION_MODEL_ID` / `vision_model_id` are unset
- `ABSTRACTVISION_MODELS_PATH` / `vision_models_path` — provider catalog path for explicit listing (default: `/models`, so a `/v1` base URL queries `/v1/models`)
- Many OpenAI-compatible providers expose `GET /models`. AbstractVision exposes that catalog via `abstractvision provider-models`, `VisionManager.list_provider_models(...)`, and the AbstractCore plugin method `llm.vision.list_provider_models(...)`; it does not call the catalog automatically or use it to select a model. The plugin uses its static default (`gpt-image-1`) unless a model id is configured.
- `ABSTRACTVISION_IMAGES_GENERATIONS_PATH` — default: `/images/generations`
- `ABSTRACTVISION_IMAGES_EDITS_PATH` — default: `/images/edits`
- `ABSTRACTVISION_TEXT_TO_VIDEO_PATH` — optional (enables `text_to_video`)
- `ABSTRACTVISION_IMAGE_TO_VIDEO_PATH` — optional (enables `image_to_video`)
- `ABSTRACTVISION_IMAGE_TO_VIDEO_MODE` — `multipart` (default) or `json_b64`

### Diffusers backend

- `ABSTRACTVISION_DIFFUSERS_DEVICE` — `auto` (default), `cpu`, `cuda`, `mps`, …
- `ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE` — optional (`float16`, `bfloat16`, `float32`)
- `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD` — `0` (default/cache-only/offline) or `1` to permit runtime downloads
- `ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32` — `1` (default) or `0` (MPS-only image fallback behavior; local video does not auto-promote above 16-bit on MPS)

Playground-only Diffusers vars:
- `ABSTRACTVISION_ALLOW_DOWNLOAD` — legacy fallback used only when `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD` is unset
- `ABSTRACTVISION_DIFFUSERS_CACHE_DIR` — optional Hugging Face cache directory override
- `ABSTRACTVISION_DIFFUSERS_REVISION` — optional model revision
- `ABSTRACTVISION_DIFFUSERS_VARIANT` — optional model variant

### MLX-Gen backend (Apple-first)

- `ABSTRACTVISION_MFLUX_MODEL` — exact published MLX-Gen repo id, local model path, or custom repo id (examples: `AbstractFramework/flux.2-klein-4b-4bit`, `AbstractFramework/ernie-image-turbo-8bit`, `briaai/FIBO`, `prism-ml/bonsai-image-ternary-4B-mlx-2bit`, `briaai/Fibo-Edit`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit`, `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit`, `/path/to/preset-dir`)
- `ABSTRACTVISION_MFLUX_BASE_MODEL` — optional base family for local paths or custom repos (`flux2-klein-4b`, `flux2-klein-9b`, `flux2-klein-base-4b`, `flux2-klein-base-9b`, `bonsai-image-ternary`, `z-image`, `z-image-turbo`, `qwen-image`, `qwen-image-edit-2511`, `ernie-image-turbo`, `fibo`, `fibo-lite`, `fibo-edit`, `fibo-edit-rmbg`, `wan2.2-ti2v-5b`, `wan2.2-t2v-a14b`, `wan2.2-i2v-a14b`), or the MiniMax-H3 schedule (`minimax-h3-turbo-544p`, `minimax-h3-turbo`, `minimax-h3`)
- `ABSTRACTVISION_MFLUX_ALLOW_DOWNLOAD` — `0` (default) or `1` to permit runtime downloads when a preset/repo is missing from the local cache
- `ABSTRACTVISION_MODEL_DIR` — legacy preset root only; curated downloads now land in the Hugging Face cache
- `ABSTRACTVISION_MLX_GEN_LORA_PATHS` / `ABSTRACTVISION_MFLUX_LORA_PATHS` — optional default LoRA adapters applied by the MLX-Gen backend when a request does not carry its own `lora_adapters`
- `ABSTRACTVISION_MLX_GEN_LORA_SCALES` / `ABSTRACTVISION_MFLUX_LORA_SCALES` — optional default LoRA scales aligned with the configured default paths
- `ABSTRACTVISION_MLX_GEN_LORA_TARGET_ROLES` / `ABSTRACTVISION_MFLUX_LORA_TARGET_ROLES` — optional default target roles aligned with the configured default paths; primarily useful for Wan routes
- Canonical provider/model routing is `mlx-gen` / `mlx-gen/<exact-huggingface-repo>`, for example `mlx-gen/AbstractFramework/flux.2-klein-4b-4bit`. Legacy `mflux` provider values, routed ids, and env var names are accepted for compatibility, but the model id itself should be the exact published repo id.

MLX-Gen LoRA notes:

- CLI callers should prefer repeated `--lora`, `--lora-scale`, and
  `--lora-target-role` flags.
- Python and AbstractCore callers should prefer the shared typed
  `lora_adapters` request field.
- Catalog discovery surfaces exact-route LoRA truth through `supports_lora`,
  `lora_status`, `lora_target_roles`, and `lora_validation_profile`.
- For LightX2V Qwen Lightning adapters, do not manually pair a BF16-trained
  Lightning LoRA with the raw unscaled FP8 Qwen base
  (`qwen_image_fp8_e4m3fn.safetensors`); upstream documents grid artifacts for
  that mix. See:
  <https://github.com/ModelTC/LightX2V-Qwen-Image-Lightning#-using-lightning-loras-with-fp8-models>
- For local LoRA-heavy Qwen and Wan runs through MLX-Gen, prefer curated
  `...-8bit` model ids when memory allows. q4 remains the lighter fallback.

Capability asset overrides:

- `ABSTRACTVISION_MODEL_CAPABILITIES_PATH` — optional filesystem path to a replacement `vision_model_capabilities.json`
- `ABSTRACTVISION_ADAPTER_CAPABILITIES_PATH` — optional filesystem path to a replacement `vision_adapter_capabilities.json`

These let operators update curated model/adapter metadata and default
parameters without changing Python code.

### stable-diffusion.cpp backend

- `ABSTRACTVISION_SDCPP_BIN` — `sd-cli` path/name (default: `sd-cli`)
- `ABSTRACTVISION_SDCPP_MODEL` — optional curated model key or single-file model path. For curated keys such as `flux2-klein-base-4b` or `qwen-image`, AbstractVision resolves cached VAE/LLM companions automatically.
- `ABSTRACTVISION_SDCPP_DIFFUSION_MODEL` — GGUF diffusion model path
- `ABSTRACTVISION_SDCPP_VAE` — VAE safetensors path (required for component-mode models like Qwen Image GGUF and FLUX.2 GGUF)
- `ABSTRACTVISION_SDCPP_LLM` — text encoder path (often GGUF; required for component-mode models like Qwen Image GGUF and FLUX.2 GGUF)
- `ABSTRACTVISION_SDCPP_LLM_VISION` — optional vision encoder GGUF path
- `ABSTRACTVISION_SDCPP_EXTRA_ARGS` — extra `sd-cli` flags (string, split like a shell)
- `ABSTRACTVISION_DISABLE_GGUF_ON_MACOS` — set to `1` to hide/disable GGUF/stable-diffusion.cpp catalog entries and execution on macOS (default: `0` / GGUF enabled)

Curated bundle flow:
- `abstractvision download flux2-klein-base-4b --provider sdcpp`
- `abstractvision t2i --provider sdcpp --model flux2-klein-base-4b "a studio photo of an espresso machine"`

This keeps the UX simple for external users: the package owns the bundle map and resolves side artifacts from the cache.

Tip:
- If you want **Metal** acceleration on macOS (Apple Silicon), install a Metal-capable `sd-cli` binary from
  stable-diffusion.cpp releases and point `ABSTRACTVISION_SDCPP_BIN` at it (or pass the path as the last arg to
  `/backend sdcpp ...` in the REPL). If you don’t, the backend may fall back to python bindings that run CPU-only.
