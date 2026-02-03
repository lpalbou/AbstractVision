# Getting Started

This guide helps you generate your first image using AbstractVision with different model families:

- **Diffusers (Python)**: Stable Diffusion / Qwen Image / FLUX 2 / GLM-Image
- **stable-diffusion.cpp**: GGUF diffusion models via pip-installable python bindings (no external `sd-cli` required) or the external `sd-cli` executable
- **AbstractCore Server**: OpenAI-compatible `/v1/images/*` endpoints + a tiny web playground UI

---

## 0) Install

From PyPI:

```bash
pip install abstractvision
```

**Important (as of January 27, 2026):** Some models require Diffusers **from GitHub `main`** because their pipeline
classes are not yet in the latest PyPI release. In this category today:

- `zai-org/GLM-Image` (`GlmImagePipeline`)
- `black-forest-labs/FLUX.2-klein-4B` (`Flux2KleinPipeline`)

If you're installing **AbstractVision from a repo checkout**, install the dev extra (this installs `diffusers@main` + compatible deps):

```bash
pip install -e ".[huggingface-dev]"
```

If you're installing **AbstractVision from PyPI** (as above), the PyPI release may not ship the `huggingface-dev` extra yet.
In that case, install Diffusers from source directly:

```bash
pip install -U "git+https://github.com/huggingface/diffusers@main"
```

Sanity check:

```bash
python -c "import diffusers; print(diffusers.__version__)"
python -c "import diffusers; print('GlmImagePipeline', hasattr(diffusers, 'GlmImagePipeline')); print('Flux2KleinPipeline', hasattr(diffusers, 'Flux2KleinPipeline'))"
```

Offline alternative (if you already have a local Diffusers checkout):

```bash
pip install -U -e /path/to/diffusers
```

Or, from a repo checkout (run in the `abstractvision/` repo root):

```bash
pip install -e .
```

No extras are required: AbstractVision is batteries-included (Diffusers + stable-diffusion.cpp python bindings), so a fresh environment should only need model weights.

---

## 1) First image (Diffusers)

By default, AbstractVision allows downloading models into your Hugging Face cache.
To force cache-only/offline mode, set:

```bash
export ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=0
```

```bash
export ABSTRACTVISION_BACKEND=diffusers
export ABSTRACTVISION_DIFFUSERS_DEVICE=mps
# mps = macOS Apple Silicon; use cuda/cpu on other machines
# Optional: override dtype (auto defaults to float16 on MPS for broad compatibility).
# - `float16` is usually the best speed/compatibility tradeoff on Apple Silicon
# - `bfloat16` can work for some models, but can trigger dtype-mismatch errors in some pipelines
# - `float32` is the most stable, but can require much more memory
# export ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE=bfloat16
# export ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE=float16
# export ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE=float32
```

Quick sanity check (macOS):

```bash
python -c "import torch; print('mps', torch.backends.mps.is_available(), 'cuda', torch.cuda.is_available())"
```

Start the REPL:

```bash
abstractvision repl
```

Then:

```text
/backend diffusers runwayml/stable-diffusion-v1-5 mps
/set guidance_scale 7
/set seed 42
/t2i "a cinematic photo of a red fox in snow" --open
```

Change settings by changing `/set …` values, or pass flags per request:

```text
/t2i "a watercolor painting of a lighthouse" --width 768 --height 768 --steps 30 --seed 123 --guidance-scale 6.5 --open
```

---

## 2) Qwen Image (Diffusers)

Qwen Image models in the registry:

- `Qwen/Qwen-Image` (older)
- `Qwen/Qwen-Image-2512` (newer)

Use the same Diffusers flow:

```text
/backend diffusers Qwen/Qwen-Image-2512 mps float16
/t2i "a poster with the word 'ABSTRACT' rendered perfectly in bold typography" --width 512 --height 512 --steps 10 --guidance-scale 2.5 --open
```

Notes:
- Qwen Image models are **large**.
- For best results, prefer the model card’s recommended sizes (e.g. 1328x1328 for 1:1). For quick tests, 512x512 is fine.
- On Apple Silicon (MPS), start with fp16 (default; best compatibility):
  - `ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE=float16` (or in the REPL: `/backend diffusers Qwen/Qwen-Image-2512 mps float16`)
- If you get NaNs/black images, try fp32 (this can require **very** large peak memory during load):
  - `ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE=float32` (or in the REPL: `/backend diffusers Qwen/Qwen-Image-2512 mps float32`)
- On Apple Silicon (MPS), AbstractVision upcasts the VAE to fp32 when using fp16 to avoid common “black image” issues.
- Automatic fp32 retry on all-black output is enabled by default on MPS (can increase peak memory):
  - disable with `ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32=0`
- In AbstractVision, `--guidance-scale` is mapped to Qwen’s `true_cfg_scale` when using Qwen pipelines (CFG). If you didn’t provide a `negative_prompt`, AbstractVision passes an empty one so CFG is actually enabled.

Tip: keep `guidance_scale` relatively low for some modern DiT models.

---

## 2.1) LoRA + Rapid-AIO (Diffusers)

AbstractVision can apply LoRA adapters (Diffusers adapter system) and optionally swap in a distilled “Rapid-AIO”
transformer for faster Qwen Image Edit inference.

These features can download from Hugging Face by default (same as model downloads). Use cache-only mode if needed:

```bash
export ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=0
```

LoRA example (REPL; note: `loras_json` is forwarded via `request.extra`):

```text
/backend diffusers Qwen/Qwen-Image-Edit-2511 mps float16
/t2i "a cinematic photo of a red fox in snow" --steps 8 --guidance-scale 1 --loras_json '[{"source":"lightx2v/Qwen-Image-Edit-2511-Lightning","scale":1.0}]' --open
```

Rapid-AIO example (distilled transformer override; Qwen Image Edit):

```text
/backend diffusers Qwen/Qwen-Image-Edit-2511 mps float16
/t2i "a cinematic photo of a red fox in snow" --steps 4 --guidance-scale 1 --rapid_aio_repo linoyts/Qwen-Image-Edit-Rapid-AIO --open
```

---

## 3) FLUX 2 (Diffusers)

FLUX 2 models in the registry:

- `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0, not gated)
- `black-forest-labs/FLUX.2-dev` (non-commercial license, gated on Hugging Face)

Sanity check:

```bash
python -c "import diffusers; print(diffusers.__version__)"
```

Notes:
- `FLUX.2-dev` uses Diffusers `Flux2Pipeline` and works on released Diffusers (0.36+).
- `FLUX.2-klein-4B` uses `Flux2KleinPipeline`, which is not available in the released Diffusers (0.36.0). It currently
  requires installing Diffusers from source (or use the AbstractVision dev extra):
  - `pip install -U "abstractvision[huggingface-dev]"`
  - `pip install -U "git+https://github.com/huggingface/diffusers@main"`

Recommended offline-friendly example (`FLUX.2-klein-4B`, not gated):

```text
/backend diffusers black-forest-labs/FLUX.2-klein-4B mps float16
/t2i "a minimalist product photo of a matte black espresso machine, studio lighting" --width 1024 --height 1024 --steps 10 --guidance-scale 1.0 --seed 0 --open
```

Example (`FLUX.2-dev`, gated; you must pre-download it into your HF cache first):

```text
/backend diffusers black-forest-labs/FLUX.2-dev mps
/t2i "a minimalist product photo of a matte black espresso machine, studio lighting" --width 1024 --height 1024 --steps 4 --guidance-scale 1.0 --seed 0 --open
```

If you use gated models (like `FLUX.2-dev`), you typically must accept the model’s terms on Hugging Face and set `HF_TOKEN` in your environment.

---

## 4) Stable Diffusion 3.5 (Diffusers, gated)

SD3.5 models (all gated on Hugging Face):

- `stabilityai/stable-diffusion-3.5-large-turbo`
- `stabilityai/stable-diffusion-3.5-large`
- `stabilityai/stable-diffusion-3.5-medium`

1) Accept the model terms on Hugging Face (in your browser).  
2) Export a token:

```bash
export HF_TOKEN=...   # your Hugging Face access token
```

Then in the REPL:

```text
/backend diffusers stabilityai/stable-diffusion-3.5-large-turbo mps
/t2i "a modern product photo of a watch, studio lighting" --width 1024 --height 1024 --steps 6 --guidance-scale 4 --seed 42 --open
```

Turbo models are usually best with low step counts (e.g. ~4–8).

---

## 5) Qwen-Image GGUF (stable-diffusion.cpp)

If you downloaded a GGUF diffusion model (like `unsloth/Qwen-Image-2512-GGUF:qwen-image-2512-Q4_K_M.gguf`), Diffusers cannot load it. Use the stable-diffusion.cpp backend instead (either via pip-installed python bindings or `sd-cli`).

### 5.1 Install stable-diffusion.cpp runtime

By default, `pip install abstractvision` includes the pip-installable stable-diffusion.cpp python bindings (`stable-diffusion-cpp-python`).

Alternative (external executable):

- Download `sd-cli` from: https://github.com/leejet/stable-diffusion.cpp/releases
- Ensure `sd-cli` is in your `PATH` (or pass a full path as the last arg to `/backend sdcpp …`).

### 5.2 Download the required Qwen Image VAE

```bash
curl -L -o ./qwen_image_vae.safetensors \\
  https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors
```

### 5.3 Run the REPL with `sdcpp` backend

```bash
abstractvision repl
```

Then:

```text
/backend sdcpp /path/to/qwen-image-2512-Q4_K_M.gguf ./qwen_image_vae.safetensors /path/to/Qwen2.5-VL-7B-Instruct-*.gguf
/set width 1024
/set height 1024
/t2i "a cinematic photo of a red fox in snow" --sampling-method euler --offload-to-cpu --diffusion-fa --flow-shift 3 --open
```

Any extra `--flag` you pass (like `--sampling-method euler`) is forwarded to the backend as `extra`.
- CLI mode: forwarded to `sd-cli`
- Python bindings mode: a small subset is supported (e.g. `--sampling-method`, `--steps`, `--cfg-scale`, `--seed`, `--offload-to-cpu`, `--diffusion-fa`, `--flow-shift`)

---

## 6) Web UI testing (recommended): AbstractCore Server + Playground

AbstractCore exposes:

- `POST /v1/images/generations`
- `POST /v1/images/edits`

### 6.0 Install AbstractCore Server (required)

The web UI is served by **AbstractCore**, not AbstractVision. Install AbstractCore with its server dependencies into the **same** environment:

From PyPI:

```bash
pip install "abstractcore[server]"
```

If you're installing **AbstractVision from a repo checkout**, do:

```bash
pip install "abstractcore[server]"
pip install -e .
```

(`abstractcore[server]` installs FastAPI + Uvicorn + python-multipart (required for `/v1/images/edits`) and AbstractVision (required for `/v1/images/*`).)

### 6.1 Start AbstractCore (auto backend; Stable Diffusion 1.5)

By default, AbstractCore uses `ABSTRACTCORE_VISION_BACKEND=auto` (you can leave it unset). The playground sends a `model` per request, so you usually don't need to set `ABSTRACTCORE_VISION_MODEL_ID`.

```bash
pip install "abstractcore[server]"
```

Then:

```bash
# Optional: choose where Diffusers runs. Default is `auto` (prefers `cuda`/`mps` when available, else `cpu`).
# export ABSTRACTCORE_VISION_DEVICE=auto  # or: cpu / cuda / mps
# Optional: disable downloads (default allows downloads).
# export ABSTRACTCORE_VISION_ALLOW_DOWNLOAD=0
python -m uvicorn abstractcore.server.app:app --port 8000
```

In the playground, set Model to `runwayml/stable-diffusion-v1-5`.

### 6.2 Start AbstractCore with `sdcpp` (GGUF / stable-diffusion.cpp)

This path supports two runtimes:

- pip-only (default): no external `sd-cli` is needed (uses `stable-diffusion-cpp-python`)
- external executable: install `sd-cli` and keep `ABSTRACTCORE_VISION_SDCPP_BIN` pointing to it

```bash
pip install "abstractcore[server]"
export ABSTRACTCORE_VISION_SDCPP_BIN=sd-cli   # optional; only used when the executable exists
export ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL=/path/to/qwen-image-2512-Q4_K_M.gguf
export ABSTRACTCORE_VISION_SDCPP_VAE=$PWD/qwen_image_vae.safetensors
export ABSTRACTCORE_VISION_SDCPP_LLM=/path/to/Qwen2.5-VL-7B-Instruct-*.gguf
export ABSTRACTCORE_VISION_SDCPP_EXTRA_ARGS="--offload-to-cpu --diffusion-fa --sampling-method euler --flow-shift 3"
python -m uvicorn abstractcore.server.app:app --port 8000
```

Tip: leave `ABSTRACTCORE_VISION_BACKEND` unset. In `auto` mode, the server routes Diffusers model ids like `runwayml/stable-diffusion-v1-5` to Diffusers, and local `.gguf` paths to stable-diffusion.cpp.

If you're installing **AbstractVision from a repo checkout**, do:

```bash
pip install "abstractcore[server]"
pip install -e .
```

### 6.3 Serve the playground page

```bash
cd abstractvision/playground
python -m http.server 8080
```

Open:

- `http://localhost:8080/vision_playground.html`

You can now interactively tweak prompt/steps/seed and (for edits) upload an input image + mask.
