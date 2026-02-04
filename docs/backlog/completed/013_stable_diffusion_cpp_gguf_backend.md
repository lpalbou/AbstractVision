## Task 013: Add stable-diffusion.cpp (sd-cli) GGUF backend

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Main goals

- Add a dependency-light local backend that can run GGUF-based image generation via `sd-cli` (stable-diffusion.cpp).
- Make it easy to use **Qwen-Image-2512 GGUF** (and similar DiT/transformer diffusion models) with AbstractVision.
- Enable the same backend through AbstractCore’s OpenAI-compatible `/v1/images/*` endpoints.

## Secondary goals

- Keep the base install light (no torch/diffusers required).
- Provide clear “interactive testing” paths (REPL + small web playground calling AbstractCore).

---

## Context / problem

Users are downloading GGUF-quantized diffusion models (e.g. `unsloth/Qwen-Image-2512-GGUF:qwen-image-2512-Q4_K_M.gguf`) and want to run them locally.

Diffusers cannot load GGUF files, and llama.cpp does not recognize `general.architecture=qwen_image` in this GGUF. The practical local runtimes for these models today are:
- stable-diffusion.cpp (`sd-cli`)
- ComfyUI with ComfyUI-GGUF

AbstractVision needs a clean, minimal “local executable backend” so users can:
- use GGUF models without pulling in heavy Python ML stacks
- keep artifact-first outputs and stable API
- re-expose generation as OpenAI-compatible endpoints via AbstractCore

---

## Constraints

- Preserve the integrator-facing API (`VisionManager.generate_image/edit_image/...`).
- Artifact-first outputs (bytes → `GeneratedAsset` → stored as artifact refs).
- Dependency-light base install (no new heavy deps; use stdlib `subprocess`).
- No implicit model downloads by default.
- Cross-platform: Mac/Linux/Windows.

---

## Research, options, and references

- **Option A: Wrap stable-diffusion.cpp `sd-cli` via `subprocess`**
  - Pros: no heavy python deps, supports GGUF diffusion models, matches user intent (“local binary + local weights”), easiest to integrate with AbstractCore.
  - Cons: requires users to install `sd-cli` separately; error messages must be good.
  - References:
    - `unsloth/Qwen-Image-2512-GGUF` README (points to ComfyUI / stable-diffusion.cpp)
      - https://huggingface.co/unsloth/Qwen-Image-2512-GGUF
    - stable-diffusion.cpp CLI docs (flags, output behavior)
      - https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/examples/cli/README.md
    - stable-diffusion.cpp Qwen Image guide (required component files)
      - https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/docs/qwen_image.md

- **Option B: Add a ComfyUI backend**
  - Pros: users already have a UI; broad model/workflow support.
  - Cons: workflow JSON + node pack coordination is complex; not the simplest “clean + minimal” integration for phase 1.
  - References:
    - ComfyUI-GGUF: https://github.com/city96/ComfyUI-GGUF

- **Option C: Convert GGUF → Diffusers**
  - Pros: keeps everything in Python.
  - Cons: unrealistic / lossy; defeats the point of GGUF quantization and adds heavy dependencies.

---

## Decision

**Chosen approach**: Option A — implement a new AbstractVision backend that shells out to stable-diffusion.cpp `sd-cli`.

**Why**:
- Fastest path to “it works locally” for GGUF diffusion models.
- Keeps AbstractVision lightweight while supporting serious local inference.
- Naturally composes with AbstractCore’s OpenAI-compatible `/v1/images/*` endpoints.

---

## Dependencies

- **Backlog tasks**:
  - Completed: `docs/backlog/completed/010_huggingface_diffusers_backend_images.md`
  - Completed: `docs/backlog/completed/011_abstractcore_openai_compatible_vision_endpoints.md`
  - Completed: `docs/backlog/completed/012_packaging_extras_and_release_hygiene.md`

---

## Implementation plan

- Implement `StableDiffusionCppVisionBackend` + config in `abstractvision.backends`.
- Map `ImageGenerationRequest` and `ImageEditRequest` to `sd-cli` flags (`--prompt`, `--negative-prompt`, `--width`, `--height`, `--steps`, `--cfg-scale`, `--seed`, `--init-img`, `--mask`, `--strength`, etc).
- Support “component mode” (`--diffusion-model`, `--vae`, `--llm`) for Qwen Image.
- Allow passing additional `sd-cli` flags via `request.extra` (best-effort) and/or config defaults.
- Add REPL backend selection command (`/backend sdcpp ...`) and env vars.
- Add AbstractCore server backend mode `ABSTRACTCORE_VISION_BACKEND=sdcpp` (using AbstractVision backend internally).
- Add unit tests (mocking subprocess; no real inference).

---

## Success criteria

- AbstractVision can generate an image via `sd-cli` backend and store it as an artifact ref.
- AbstractCore `/v1/images/generations` works in `sdcpp` mode and returns `b64_json`.
- Failures are actionable (missing binary, missing model paths, missing component files).

---

## Test plan

- AbstractVision: `python -m unittest discover -s tests -p "test_*.py" -q`
- AbstractCore: `pytest -q abstractcore/tests/server/test_server_vision_image_endpoints.py`
- Manual smoke:
  - Install stable-diffusion.cpp `sd-cli` and run with Qwen-Image + VAE + Qwen2.5-VL text encoder (see completion report for exact commands).

---

## Report (fill only when completed)

### Summary

- Added a dependency-light `sd-cli` backend (`StableDiffusionCppVisionBackend`) that runs local GGUF diffusion models via `subprocess`.
- Added REPL support for `/backend sdcpp ...` and forwarded unknown REPL flags into backend `extra` for model/runtime-specific knobs.
- Extended AbstractCore Server `/v1/images/*` to support `ABSTRACTCORE_VISION_BACKEND=sdcpp` and added `extra_json` support for `/v1/images/edits`.
- Added a tiny web playground (`playground/vision_playground.html`) for interactive manual testing.

### Validation

- Tests:
  - `python -m unittest discover -s tests -p "test_*.py" -q`
  - `pytest -q abstractcore/tests/server/test_server_vision_image_endpoints.py`

### How to test (manual / interactive)

#### 1) Direct `sd-cli` smoke test (Qwen Image GGUF)

Qwen Image GGUF requires 3 local files:

- **Diffusion model (GGUF)**: e.g. `qwen-image-2512-Q4_K_M.gguf`
- **VAE**: `qwen_image_vae.safetensors` from `Comfy-Org/Qwen-Image_ComfyUI` (`split_files/vae/…`)
- **Text encoder (GGUF)**: a Qwen2.5-VL 7B Instruct GGUF (stable-diffusion.cpp uses it as `--llm`)

Run (adapt paths / options):

```bash
sd-cli \
  --diffusion-model /path/to/qwen-image-2512-Q4_K_M.gguf \
  --vae /path/to/qwen_image_vae.safetensors \
  --llm /path/to/Qwen2.5-VL-7B-Instruct-*.gguf \
  -p "a cinematic photo of a red fox in snow" \
  --cfg-scale 2.5 \
  --sampling-method euler \
  --offload-to-cpu \
  --diffusion-fa \
  --flow-shift 3 \
  -W 1024 -H 1024 \
  -o ./output.png
```

#### 2) AbstractVision REPL (local `sd-cli`)

```bash
abstractvision repl
```

Inside:

```text
/backend sdcpp /path/to/qwen-image-2512-Q4_K_M.gguf /path/to/qwen_image_vae.safetensors /path/to/Qwen2.5-VL-7B-Instruct-*.gguf sd-cli
/set width 1024
/set height 1024
/t2i "a cinematic photo of a red fox in snow" --sampling-method euler --offload-to-cpu --diffusion-fa --flow-shift 3 --open
```

#### 3) AbstractCore Server + Web Playground (recommended)

Start AbstractCore with `sdcpp` backend:

```bash
export ABSTRACTCORE_VISION_BACKEND=sdcpp
export ABSTRACTCORE_VISION_SDCPP_BIN=sd-cli
export ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL=/path/to/qwen-image-2512-Q4_K_M.gguf
export ABSTRACTCORE_VISION_SDCPP_VAE=/path/to/qwen_image_vae.safetensors
export ABSTRACTCORE_VISION_SDCPP_LLM=/path/to/Qwen2.5-VL-7B-Instruct-*.gguf
export ABSTRACTCORE_VISION_SDCPP_EXTRA_ARGS="--offload-to-cpu --diffusion-fa --sampling-method euler --flow-shift 3"
uvicorn abstractcore.server.app:app --port 8000
```

Serve the playground:

```bash
cd playground
python -m http.server 8080
```

Open `http://localhost:8080/vision_playground.html` and generate/edit images interactively.
