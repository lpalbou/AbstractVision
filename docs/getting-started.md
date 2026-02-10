# Getting Started

This guide helps you generate your first image using AbstractVision with the built-in backends:

- **OpenAI-compatible HTTP**: call a local/remote server that exposes OpenAI-shaped image endpoints
- **Diffusers (local Python)**: Stable Diffusion / Qwen Image / FLUX 2 / GLM-Image (and other Diffusers pipelines)
- **stable-diffusion.cpp (local GGUF)**: GGUF diffusion models via pip-installable python bindings (no external `sd-cli` required) or the external `sd-cli` executable
- **Playground (web, optional)**: static UI for AbstractCore Server vision job endpoints (`/v1/vision/*`)

See also:
- Docs index: [docs/README.md](README.md)
- FAQ: [docs/faq.md](faq.md)
- API reference: [docs/api.md](api.md)
- Architecture: [docs/architecture.md](architecture.md)
- Backends: [docs/reference/backends.md](reference/backends.md)
- Configuration (CLI/REPL env vars): [docs/reference/configuration.md](reference/configuration.md)
- Capability registry: [docs/reference/capabilities-registry.md](reference/capabilities-registry.md)
- Artifacts: [docs/reference/artifacts.md](reference/artifacts.md)
- AbstractCore integration: [docs/reference/abstractcore-integration.md](reference/abstractcore-integration.md)

---

## 0) Install

From PyPI:

```bash
pip install abstractvision
```

AbstractVision’s base install is **batteries included** (Torch + Diffusers + stable-diffusion.cpp bindings). Heavy modules are imported lazily, but the dependencies are still installed (see `pyproject.toml`).

If you see “missing pipeline class” errors for newer model families, install the `huggingface-dev` extra (to get compatible dependencies) and then install Diffusers from source (`main`).

If you're installing **AbstractVision from a repo checkout**, install the dev extra (compatible deps; does not include Diffusers `main`):

```bash
pip install -e ".[huggingface-dev]"
```

If you're installing **AbstractVision from PyPI**, you can install the extra directly:

```bash
pip install -U "abstractvision[huggingface-dev]"
```

Or install Diffusers from source directly:

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

Or, from a repo checkout (run in the repo root):

```bash
pip install -e .
```

No extras are required for most use cases: AbstractVision is batteries-included (Diffusers + stable-diffusion.cpp python bindings), so a fresh environment should only need model weights. Use `huggingface-dev` only when you need Diffusers `main`.

---

## Recommended default models (VRAM guide)

If you run **locally** (Diffusers backend) and want a reliable starting point, here are practical model picks from the packaged capability registry (`src/abstractvision/assets/vision_model_capabilities.json`).

Notes:
- VRAM needs vary with resolution, dtype, and pipeline implementation. Treat this as a starting point.
- Some models are **gated** on Hugging Face and require accepting terms + setting `HF_TOKEN`.
- If you want a non-gated modern image model, try `black-forest-labs/FLUX.2-klein-4B` (but it currently requires installing Diffusers from source; see the FLUX section below).

| GPU VRAM | Recommended model id | Why | Install / quickstart |
|---:|---|---|---|
| ≤ 16 GB | `runwayml/stable-diffusion-v1-5` | Small, stable, and widely compatible (Windows/Linux CUDA, macOS MPS) | `pip install abstractvision` then run the REPL using the snippet below |
| 32 GB | `stabilityai/stable-diffusion-3.5-large-turbo` | High-quality still images with low step counts (gated) | Accept model terms on HF, set `HF_TOKEN`, then use the SD3.5 section below |
| 64 GB | `Qwen/Qwen-Image-2512` | Strong prompt following and text rendering (large model) | Same as Diffusers setup; if pipeline import fails, use Diffusers `main` (see install section above) |
| 128 GB | `black-forest-labs/FLUX.2-dev` | Very high quality (very large; non-commercial license; gated) | Accept model terms on HF, set `HF_TOKEN`, then use the FLUX section below |

Recommended default (local, cross-platform) — Stable Diffusion 1.5:

```bash
pip install abstractvision
export ABSTRACTVISION_BACKEND=diffusers
export ABSTRACTVISION_MODEL_ID=runwayml/stable-diffusion-v1-5
export ABSTRACTVISION_DIFFUSERS_DEVICE=auto
abstractvision repl
```

Then type a prompt (plain text runs `/t2i`), or use `/t2i "..." --open`.

Jump to detailed recipes:
- Stable Diffusion 1.5: section **2) First image (Diffusers)**
- Qwen Image: section **3) Qwen Image (Diffusers)**
- FLUX 2: section **4) FLUX 2 (Diffusers)**
- SD3.5: section **5) Stable Diffusion 3.5 (Diffusers, gated)**

---

## 1) First image (OpenAI-compatible HTTP)

Use this path if you already have a server that exposes OpenAI-shaped image endpoints (e.g. a local model server).

One-shot (stores output via `LocalAssetStore` and prints an artifact ref + file path):

```bash
abstractvision t2i --base-url http://localhost:1234/v1 "a cinematic photo of a red fox in snow" --open
```

Interactive REPL:

```bash
abstractvision repl
```

```text
/backend openai http://localhost:1234/v1
/t2i "a watercolor painting of a lighthouse" --width 768 --height 768 --steps 20 --open
```

If your server also supports video endpoints, configure them via `ABSTRACTVISION_TEXT_TO_VIDEO_PATH` / `ABSTRACTVISION_IMAGE_TO_VIDEO_PATH` (see [docs/reference/configuration.md](reference/configuration.md)).

---

## 2) First image (Diffusers)

By default, AbstractVision allows downloading models into your Hugging Face cache.
To force cache-only/offline mode, set:

```bash
export ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=0
```

```bash
export ABSTRACTVISION_BACKEND=diffusers
export ABSTRACTVISION_DIFFUSERS_DEVICE=auto
# auto prefers cuda, then mps, then cpu. You can also set cuda/mps/cpu explicitly.
# Optional: override dtype (auto defaults to float16 on MPS for broad compatibility).
# - `float16` is usually the best speed/compatibility tradeoff on Apple Silicon
# - `bfloat16` can work for some models, but can trigger dtype-mismatch errors in some pipelines
# - `float32` is the most stable, but can require much more memory
# export ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE=bfloat16
# export ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE=float16
# export ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE=float32
```

Quick sanity check (device):

```bash
python -c "import torch; print('mps', torch.backends.mps.is_available(), 'cuda', torch.cuda.is_available())"
```

If you have an NVIDIA GPU but `cuda` is `False`, you likely installed a CPU-only PyTorch build. Follow the PyTorch install guide to install a CUDA-enabled wheel, then re-run the sanity check: <https://pytorch.org/get-started/locally/>.

Start the REPL:

```bash
abstractvision repl
```

Then:

```text
/backend diffusers runwayml/stable-diffusion-v1-5 auto
/set guidance_scale 7
/set seed 42
/t2i "a cinematic photo of a red fox in snow" --open
```

Change settings by changing `/set …` values, or pass flags per request:

```text
/t2i "a watercolor painting of a lighthouse" --width 768 --height 768 --steps 30 --seed 123 --guidance-scale 6.5 --open
```

---

## 3) Qwen Image (Diffusers)

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
- In AbstractVision, `--guidance-scale` is mapped to Qwen’s `true_cfg_scale` when using Qwen pipelines (CFG). If you set `--guidance-scale` but don’t provide a `negative_prompt`, AbstractVision passes a placeholder negative prompt (`" "`) so CFG is actually enabled.

Tip: keep `guidance_scale` relatively low for some modern DiT models.

---

## 3.1) LoRA + Rapid-AIO (Diffusers)

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

## 4) FLUX 2 (Diffusers)

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

## 5) Stable Diffusion 3.5 (Diffusers, gated)

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

## 6) Qwen-Image GGUF (stable-diffusion.cpp)

If you downloaded a GGUF diffusion model (like `unsloth/Qwen-Image-2512-GGUF:qwen-image-2512-Q4_K_M.gguf`), Diffusers cannot load it. Use the stable-diffusion.cpp backend instead (either via pip-installed python bindings or `sd-cli`).

### 6.1 Install stable-diffusion.cpp runtime

By default, `pip install abstractvision` includes the pip-installable stable-diffusion.cpp python bindings (`stable-diffusion-cpp-python`).

Alternative (external executable):

- Download `sd-cli` from: <https://github.com/leejet/stable-diffusion.cpp/releases>
- Ensure `sd-cli` is in your `PATH` (or pass a full path as the last arg to `/backend sdcpp …`).

### 6.2 Download the required Qwen Image VAE

```bash
curl -L -o ./qwen_image_vae.safetensors \\
  https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors
```

### 6.3 Run the REPL with `sdcpp` backend

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
- Python bindings mode: keys are mapped to python binding kwargs when supported; unsupported keys are ignored (see [`../src/abstractvision/backends/stable_diffusion_cpp.py`](../src/abstractvision/backends/stable_diffusion_cpp.py))

---

## 7) Web UI testing (optional): Playground

This repo includes a static, dependency-free web UI at `playground/vision_playground.html`.

It is designed to talk to an **AbstractCore Server** instance that implements the `/v1/vision/*` endpoints used by the page
(model list/load/unload and image generation/edit jobs). Evidence: see the fetch calls in `playground/vision_playground.html`.

For server requirements and the endpoint list, see `playground/README.md`.

### 7.1 Serve the playground page

```bash
cd playground
python -m http.server 8080
```

Open:

- `http://localhost:8080/vision_playground.html`

In the UI:
- Set the API Base URL (defaults to `http://localhost:8000`) and click **Ping**
- Select a cached model and load it
- Generate (T2I) or upload an input image (I2I) and run edits
