# AbstractVision Playground (Web)

This is a tiny, dependency-free web UI for quickly testing AbstractCore’s OpenAI-compatible vision endpoints:

- `POST /v1/images/generations`
- `POST /v1/images/edits`

## 1) Start AbstractCore Server

From any environment where `abstractcore[server]` is installed (FastAPI + Uvicorn + multipart support):

```bash
pip install "abstractcore[server]"
```

Make sure you're on an AbstractCore version that includes the vision endpoints (`/v1/images/*`).

If you're running from an **AbstractVision repo checkout**, do:

```bash
pip install "abstractcore[server]"
pip install -e .
python -m uvicorn abstractcore.server.app:app --port 8000
```

The image endpoints return `501` until you configure a backend (see examples below).

### Diffusers example (Stable Diffusion 1.5)

```bash
pip install "abstractcore[server]"
export ABSTRACTCORE_VISION_BACKEND=diffusers
export ABSTRACTCORE_VISION_MODEL_ID=runwayml/stable-diffusion-v1-5
export ABSTRACTCORE_VISION_DEVICE=mps   # or: cpu / cuda
# Optional: disable downloads (default allows downloads).
# export ABSTRACTCORE_VISION_ALLOW_DOWNLOAD=0
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

Recommended: install Qwen Image components (diffusion model + VAE + text encoder). The stable-diffusion.cpp python bindings are installed by default with `abstractcore[server]`.

- pip: `pip install "abstractcore[server]"`
- optional `sd-cli` releases (external executable): https://github.com/leejet/stable-diffusion.cpp/releases
- Qwen Image VAE file: `split_files/vae/qwen_image_vae.safetensors` from `Comfy-Org/Qwen-Image_ComfyUI`

Example env (adjust paths):

```bash
export ABSTRACTCORE_VISION_BACKEND=sdcpp
export ABSTRACTCORE_VISION_SDCPP_BIN=sd-cli   # optional; only used when the executable exists
export ABSTRACTCORE_VISION_SDCPP_DIFFUSION_MODEL=/Users/albou/.cache/huggingface/hub/models--unsloth--Qwen-Image-2512-GGUF/snapshots/1626d7531f84b4d2ea1cd6d2e69f41ec027dd354/qwen-image-2512-Q4_K_M.gguf
export ABSTRACTCORE_VISION_SDCPP_VAE=/path/to/qwen_image_vae.safetensors
export ABSTRACTCORE_VISION_SDCPP_LLM=/Users/albou/.cache/huggingface/hub/models--unsloth--Qwen2.5-VL-7B-Instruct-GGUF/snapshots/68bb8bc4b7df5289c143aaec0ab477a7d4051aab/Qwen2.5-VL-7B-Instruct-Q4_0.gguf
export ABSTRACTCORE_VISION_SDCPP_EXTRA_ARGS="--offload-to-cpu --diffusion-fa --sampling-method euler --flow-shift 3"
```

Then run:

```bash
uvicorn abstractcore.server.app:app --port 8000
```
