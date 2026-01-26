# Getting Started

This guide helps you generate your first image using AbstractVision with different model families:

- **Diffusers (Python)**: Stable Diffusion / Qwen Image / FLUX 2 (requires `abstractvision[huggingface]`)
- **stable-diffusion.cpp (`sd-cli`)**: GGUF diffusion models (fastest path for GGUF)
- **AbstractCore Server**: OpenAI-compatible `/v1/images/*` endpoints + a tiny web playground UI

---

## 0) Install (monorepo dev)

From `abstractvision/`:

```bash
pip install -e ".[huggingface]"
```

This installs optional local image generation support (Diffusers). If you only want the lightweight core + OpenAI-compatible HTTP backend, you can do:

```bash
pip install -e .
```

---

## 1) Fastest “first image” (Diffusers + auto-download)

AbstractVision’s Diffusers backend defaults to **cache-only** (no downloads). For a quick start, opt-in to downloads:

```bash
export ABSTRACTVISION_BACKEND=diffusers
export ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1
export ABSTRACTVISION_DIFFUSERS_DEVICE=mps   # macOS Apple Silicon; use cuda/cpu on other machines
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
/set steps 20
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
/backend diffusers Qwen/Qwen-Image-2512 mps
/t2i "a poster with the word 'ABSTRACT' rendered perfectly in bold typography" --width 1024 --height 1024 --steps 30 --guidance-scale 2.5 --open
```

Tip: keep `guidance_scale` relatively low for some modern DiT models.

---

## 3) FLUX 2 (Diffusers)

FLUX 2 models in the registry:

- `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0, not gated)
- `black-forest-labs/FLUX.2-dev` (non-commercial license, gated on Hugging Face)

Example (open klein 4B):

```text
/backend diffusers black-forest-labs/FLUX.2-klein-4B mps
/t2i "a minimalist product photo of a matte black espresso machine, studio lighting" --width 1024 --height 1024 --steps 30 --guidance-scale 3.5 --open
```

If you use gated models (like `FLUX.2-dev`), you typically must accept the model’s terms on Hugging Face and set `HF_TOKEN` in your environment.

---

## 4) Qwen-Image GGUF (stable-diffusion.cpp `sd-cli`)

If you downloaded a GGUF diffusion model (like `unsloth/Qwen-Image-2512-GGUF:qwen-image-2512-Q4_K_M.gguf`), Diffusers cannot load it. Use `sd-cli` instead.

### 4.1 Install `sd-cli`

Download a stable-diffusion.cpp build from:

- https://github.com/leejet/stable-diffusion.cpp/releases

Ensure `sd-cli` is in your `PATH` (or use a full path in the `/backend sdcpp …` command below).

### 4.2 Download the required Qwen Image VAE

```bash
curl -L -o ./qwen_image_vae.safetensors \\
  https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors
```

### 4.3 Run the REPL with `sdcpp` backend

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

## 5) Web UI testing (recommended): AbstractCore Server + Playground

AbstractCore exposes:

- `POST /v1/images/generations`
- `POST /v1/images/edits`

### 5.1 Start AbstractCore with `sdcpp` (GGUF)

```bash
export ABSTRACTCORE_VISION_BACKEND=sdcpp
export ABSTRACTCORE_VISION_SDCPP_BIN=sd-cli
export ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL=/path/to/qwen-image-2512-Q4_K_M.gguf
export ABSTRACTCORE_VISION_SDCPP_VAE=$PWD/qwen_image_vae.safetensors
export ABSTRACTCORE_VISION_SDCPP_LLM=/path/to/Qwen2.5-VL-7B-Instruct-*.gguf
export ABSTRACTCORE_VISION_SDCPP_EXTRA_ARGS="--offload-to-cpu --diffusion-fa --sampling-method euler --flow-shift 3"
uvicorn abstractcore.server.app:app --port 8000
```

### 5.2 Serve the playground page

```bash
cd abstractvision/playground
python -m http.server 8080
```

Open:

- `http://localhost:8080/vision_playground.html`

You can now interactively tweak prompt/steps/seed and (for edits) upload an input image + mask.

