# AbstractVision Playground (Web)

This is a tiny, dependency-free web UI for quickly testing AbstractCore’s OpenAI-compatible vision endpoints:

- `POST /v1/images/generations`
- `POST /v1/images/edits`

## 1) Start AbstractCore Server

From any environment where `abstractcore` is installed:

```bash
export ABSTRACTCORE_VISION_BACKEND=sdcpp   # or: diffusers / openai_compatible_proxy
python -m uvicorn abstractcore.server.app:app --port 8000
```

## 2) Serve this page

Browsers may block `file://` → `http://` requests; serve the page locally:

```bash
cd abstractvision/playground
python -m http.server 8080
```

Open:

- `http://localhost:8080/vision_playground.html`

Tip:
- The UI has an optional **Model** field for both endpoints; if you leave it empty, the server uses your env defaults.

## 3) Configure Qwen-Image-2512 GGUF (stable-diffusion.cpp)

You need stable-diffusion.cpp `sd-cli` plus the Qwen Image components (diffusion model + VAE + text encoder).

- `sd-cli` releases: https://github.com/leejet/stable-diffusion.cpp/releases
- Qwen Image VAE file: `split_files/vae/qwen_image_vae.safetensors` from `Comfy-Org/Qwen-Image_ComfyUI`

Example env (adjust paths):

```bash
export ABSTRACTCORE_VISION_BACKEND=sdcpp
export ABSTRACTCORE_VISION_SDCPP_BIN=sd-cli
export ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL=/Users/albou/.cache/huggingface/hub/models--unsloth--Qwen-Image-2512-GGUF/snapshots/1626d7531f84b4d2ea1cd6d2e69f41ec027dd354/qwen-image-2512-Q4_K_M.gguf
export ABSTRACTCORE_VISION_SDCPP_VAE=/path/to/qwen_image_vae.safetensors
export ABSTRACTCORE_VISION_SDCPP_LLM=/Users/albou/.cache/huggingface/hub/models--unsloth--Qwen2.5-VL-7B-Instruct-GGUF/snapshots/68bb8bc4b7df5289c143aaec0ab477a7d4051aab/Qwen2.5-VL-7B-Instruct-Q4_0.gguf
export ABSTRACTCORE_VISION_SDCPP_EXTRA_ARGS="--offload-to-cpu --diffusion-fa --sampling-method euler --flow-shift 3"
```

Then run:

```bash
uvicorn abstractcore.server.app:app --port 8000
```
