# Getting Started

This guide helps you generate your first image using AbstractVision with different model families:

- **Diffusers (Python)**: Stable Diffusion / Qwen Image / FLUX 2 (requires `abstractvision[huggingface]`)
- **stable-diffusion.cpp (`sd-cli`)**: GGUF diffusion models (fastest path for GGUF)
- **AbstractCore Server**: OpenAI-compatible `/v1/images/*` endpoints + a tiny web playground UI

---

## 0) Install

From PyPI:

```bash
pip install "abstractvision[huggingface]"
```

Or, from a repo checkout (run in the `abstractvision/` repo root):

```bash
pip install -e ".[huggingface]"
```

This installs optional local image generation support (Diffusers). If you only want the lightweight core + OpenAI-compatible HTTP backend, you can do:

```bash
pip install abstractvision
```

Or, from a repo checkout:

```bash
pip install -e .
```

---

## 1) Fastest “first image” (Diffusers + auto-download)

AbstractVision’s Diffusers backend defaults to **cache-only** (no downloads) and forces Hugging Face **offline mode** (no network calls). For a quick start, opt-in to downloads:

```bash
export ABSTRACTVISION_BACKEND=diffusers
export ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1
export ABSTRACTVISION_DIFFUSERS_DEVICE=mps   
# mps = macOS Apple Silicon; use cuda/cpu on other machines
# Optional: override dtype (auto defaults to bf16 on MPS when supported).
# - `bfloat16` is a good default on Apple Silicon for numerical stability
# - `float16` can be faster, but some models can produce NaNs/black images
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
/set width 512
/set height 512
/set steps 10
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
/backend diffusers Qwen/Qwen-Image-2512 mps bfloat16
/t2i "a poster with the word 'ABSTRACT' rendered perfectly in bold typography" --width 512 --height 512 --steps 10 --guidance-scale 2.5 --open
```

Notes:
- Qwen Image models are **large**.
- For best results, prefer the model card’s recommended sizes (e.g. 1328x1328 for 1:1). For quick tests, 512x512 is fine.
- On Apple Silicon (MPS), start with bf16:
  - `ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE=bfloat16` (or in the REPL: `/backend diffusers Qwen/Qwen-Image-2512 mps bfloat16`)
- If you explicitly use fp16 and you get NaNs/black images, try fp32 (this can require **very** large peak memory during load):
  - `ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE=float32` (or in the REPL: `/backend diffusers Qwen/Qwen-Image-2512 mps float32`)
- On Apple Silicon (MPS), AbstractVision upcasts the VAE to fp32 when using fp16 to avoid common “black image” issues.
- Optional: enable a one-time automatic fp32 retry on all-black output (can increase peak memory a lot):
  - `ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32=1`
- In AbstractVision, `--guidance-scale` is mapped to Qwen’s `true_cfg_scale` when using Qwen pipelines (CFG). If you didn’t provide a `negative_prompt`, AbstractVision passes an empty one so CFG is actually enabled.

Tip: keep `guidance_scale` relatively low for some modern DiT models.

---

## 3) FLUX 2 (Diffusers)

FLUX 2 models in the registry:

- `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0, not gated)
- `black-forest-labs/FLUX.2-dev` (non-commercial license, gated on Hugging Face)

Some FLUX 2 repos reference a newer Diffusers pipeline class (`Flux2KleinPipeline`) than the latest released Diffusers. AbstractVision automatically falls back to a compatible loader (`Flux2Pipeline`) so `FLUX.2-klein-4B` works on released Diffusers (0.36+). Sanity check:

```bash
python -c "import diffusers; print(diffusers.__version__)"
```

Example (open klein 4B; model card defaults are very low steps):

```text
/backend diffusers black-forest-labs/FLUX.2-klein-4B mps
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

## 5) Qwen-Image GGUF (stable-diffusion.cpp `sd-cli`)

If you downloaded a GGUF diffusion model (like `unsloth/Qwen-Image-2512-GGUF:qwen-image-2512-Q4_K_M.gguf`), Diffusers cannot load it. Use `sd-cli` instead.

### 5.1 Install `sd-cli`

Download a stable-diffusion.cpp build from:

- https://github.com/leejet/stable-diffusion.cpp/releases

Ensure `sd-cli` is in your `PATH` (or use a full path in the `/backend sdcpp …` command below).

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
/backend sdcpp /path/to/qwen-image-2512-Q4_K_M.gguf ./qwen_image_vae.safetensors /path/to/Qwen2.5-VL-7B-Instruct-*.gguf sd-cli
/set width 1024
/set height 1024
/t2i "a cinematic photo of a red fox in snow" --sampling-method euler --offload-to-cpu --diffusion-fa --flow-shift 3 --open
```

Any extra `--flag` you pass (like `--sampling-method euler`) is forwarded to the backend as `extra` and translated to `sd-cli` flags.

---

## 6) Web UI testing (recommended): AbstractCore Server + Playground

AbstractCore exposes:

- `POST /v1/images/generations`
- `POST /v1/images/edits`

### 6.1 Start AbstractCore with `sdcpp` (GGUF)

```bash
export ABSTRACTCORE_VISION_BACKEND=sdcpp
export ABSTRACTCORE_VISION_SDCPP_BIN=sd-cli
export ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL=/path/to/qwen-image-2512-Q4_K_M.gguf
export ABSTRACTCORE_VISION_SDCPP_VAE=$PWD/qwen_image_vae.safetensors
export ABSTRACTCORE_VISION_SDCPP_LLM=/path/to/Qwen2.5-VL-7B-Instruct-*.gguf
export ABSTRACTCORE_VISION_SDCPP_EXTRA_ARGS="--offload-to-cpu --diffusion-fa --sampling-method euler --flow-shift 3"
python -m uvicorn abstractcore.server.app:app --port 8000
```

### 6.2 Serve the playground page

```bash
cd abstractvision/playground
python -m http.server 8080
```

Open:

- `http://localhost:8080/vision_playground.html`

You can now interactively tweak prompt/steps/seed and (for edits) upload an input image + mask.
