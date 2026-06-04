# Getting Started

This guide helps you generate your first image using AbstractVision with the built-in backends:

- **OpenAI-compatible HTTP**: call a local/remote server that exposes OpenAI-shaped image endpoints
- **Diffusers (local Python)**: Stable Diffusion / Qwen Image / FLUX 2 / other supported Diffusers pipelines
- **MLX-Gen (local Apple Silicon)**: q4/q8 AbstractFramework MLX-optimized image generation via the optional MLX-Gen 0.18.10+ runtime, official FIBO image models, and Wan 2.2 A14B `text_to_video` / first-frame `image_to_video`
- **stable-diffusion.cpp (local GGUF)**: GGUF diffusion models via `sd-cli` (recommended for GPU backends like **Metal**/**CUDA**) or via pip-installable python bindings (often **CPU-only** fallback)
- **Playground (web, optional)**: self-contained AbstractVision UI/API for local model loading and jobs (`/v1/vision/*`)

See also:
- Docs index: [docs/README.md](README.md)
- FAQ: [docs/faq.md](faq.md)
- Troubleshooting: [docs/troubleshooting.md](troubleshooting.md)
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

AbstractVision’s base install is lightweight. It includes the shared API, capability registry, artifact helpers, CLI, AbstractCore plugin entry point, and stdlib OpenAI-compatible HTTP backend. Local inference runtimes are explicit extras: install `abstractvision[diffusers]` for Torch/Diffusers, `abstractvision[sdcpp]` for the stable-diffusion.cpp python binding fallback, `abstractvision[mlx-gen]` for Apple Silicon MLX-Gen, or `abstractvision[all-apple]` for the full native macOS stack. `abstractvision[mflux]` remains available as a compatibility alias for older install instructions.

If you see “missing pipeline class” errors for newer model families, install the `diffusers-dev` extra (or compatibility alias `huggingface-dev`) to get compatible dependencies, then install Diffusers from source (`main`).

For that newer-pipeline workflow from a **repo checkout**, install the `diffusers-dev` extra (compatible deps; does not include Diffusers `main`):

```bash
pip install -e ".[diffusers-dev]"
```

If you're installing **AbstractVision from PyPI**, you can install the extra directly:

```bash
pip install -U "abstractvision[diffusers-dev]"
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

For contributor tooling from a repo checkout, use:

```bash
pip install -e ".[dev]"
```

For local Diffusers generation, install `abstractvision[diffusers]` before selecting the `diffusers` backend. Use `diffusers-dev` only when you need newer Diffusers-compatible dependency pins, and use `sdcpp` only when you want the optional stable-diffusion.cpp python binding fallback.

Optional extras:

| Extra | Use |
|---|---|
| `openai` | Empty official OpenAI provider intent marker; the HTTP backend is stdlib-only today. |
| `openai-compatible` | Empty local/remote OpenAI-shaped endpoint intent marker; the HTTP backend is stdlib-only today. |
| `diffusers` | Installs Torch/Diffusers and related packages for local Diffusers generation. |
| `sdcpp` | Installs `stable-diffusion-cpp-python` for the stable-diffusion.cpp pip binding fallback. |
| `mlx-gen` | Installs the optional MLX-Gen runtime for Apple Silicon MLX image/video generation. |
| `mflux` | Compatibility alias for the MLX-Gen runtime. |
| `apple` | Native macOS profile: Diffusers/Torch MPS, stable-diffusion.cpp bindings, and MLX-Gen. |
| `gpu` | GPU-friendly profile for Diffusers/Torch (does not include MLX-Gen). |
| `huggingface` | Compatibility alias for the historical Diffusers backend dependency set. |
| `local` | Convenience extra for both local backend dependency sets, including `sdcpp`. |
| `all` | All runtime backend dependencies, without contributor tooling. |
| `all-apple` | Aggregate native macOS profile: Diffusers/Torch MPS, stable-diffusion.cpp, and MLX-Gen. |
| `all-gpu` | Aggregate GPU profile (Diffusers + stable-diffusion.cpp bindings). |
| `abstractcore` | Empty compatibility marker; install AbstractCore in the host application environment. |

Contributor-only extras:

| Extra | Use |
|---|---|
| `diffusers-dev` / `huggingface-dev` | Looser dependency pins for newer/unreleased Diffusers pipelines. Install Diffusers `main` separately when a pipeline is not in the latest release. |
| `test` | Local test dependencies. |
| `docs` | Documentation build tooling. |
| `dev` | Full contributor workflow: tests, docs, packaging, formatting, release checks, and pre-commit. Do not use this as an application runtime profile. |

Optional (recommended): pre-download heavyweight model sets (so first-run doesn’t do surprise multi‑GB downloads):

```bash
python scripts/download_model_sets.py --list
python scripts/download_model_sets.py --plan --set sd15_diffusers
python scripts/download_model_sets.py --plan --set flux2_klein_4b_gguf
python scripts/download_model_sets.py --set sd15_diffusers
```

### 0.1 Hardware quickstart (macOS Metal vs NVIDIA CUDA vs CPU)

AbstractVision can run “locally” via three main routes:

- **Diffusers backend**: uses Torch device selection (`cuda` / `mps` / `cpu`).
- **MLX-Gen backend (`mlx-gen`)**: Apple Silicon MLX generation through the optional MLX-Gen runtime. q4 AbstractFramework model repos are the default recommendation; q8 variants are separate exact model ids for quality-focused runs.
- **stable-diffusion.cpp backend (`sdcpp`)**: runs GGUF diffusion models using:
  - `sd-cli` (**recommended** when you want GPU backends like **Metal** or **CUDA**)
  - or `stable-diffusion-cpp-python` (convenient, but often **CPU-only**, especially on macOS)

#### macOS (Apple Silicon, Metal)

- **Diffusers**: start with Stable Diffusion 1.5, then move up:
  - `/backend diffusers runwayml/stable-diffusion-v1-5 mps float16`
  - `/backend diffusers black-forest-labs/FLUX.2-klein-4B mps float16` (requires Diffusers `main` today)
- **GGUF (`sdcpp`)**: install `sd-cli` from stable-diffusion.cpp releases and use **CLI mode** for Metal speed:
  - Download: <https://github.com/leejet/stable-diffusion.cpp/releases>
  - Pick the Darwin arm64 zip (example asset name: `sd-…-bin-Darwin-macOS-…-arm64.zip`)
  - If macOS blocks execution, clear quarantine: `xattr -dr com.apple.quarantine /path/to/sd-cli`
  - In the REPL, pass the full path as the last arg to `/backend sdcpp …` (see section **6)**).

If you see `Using CPU backend` in logs, you’re on CPU (it will work, but can be extremely slow for large models).

#### NVIDIA (CUDA)

- Install a CUDA-enabled PyTorch wheel first (see <https://pytorch.org/get-started/locally/>).
- Use Diffusers with `cuda` + `float16`:
  - `/backend diffusers runwayml/stable-diffusion-v1-5 cuda float16`
- For GGUF (`sdcpp`) on NVIDIA, use an `sd-cli` build compiled with CUDA (stable-diffusion.cpp releases provide multiple assets depending on tag).

#### CPU-only

- Expect slow inference. Prefer smaller models and lower resolutions/steps.
- `sdcpp` via python bindings is the simplest “no external binary” option, but it will use whatever backend the wheel was compiled with (often CPU).

---

## Recommended default models (VRAM guide)

If you run **locally** (Diffusers backend) and want a reliable starting point, here are practical model picks from the packaged capability registry (`src/abstractvision/assets/vision_model_capabilities.json`).

Notes:
- VRAM needs vary with resolution, dtype, and pipeline implementation. Treat this as a starting point.
- Some models are **gated** on Hugging Face and require accepting terms + setting `HF_TOKEN`.
- If you want a non-gated modern image model, try `black-forest-labs/FLUX.2-klein-4B` (but it currently requires installing Diffusers from source; see the FLUX section below).

| GPU VRAM | Recommended model id | Why | Install / quickstart |
|---:|---|---|---|
| ≤ 16 GB | `runwayml/stable-diffusion-v1-5` | Small, stable, and widely compatible (Windows/Linux CUDA, macOS MPS) | `pip install "abstractvision[diffusers]"` then run the REPL using the snippet below |
| 24-32 GB | `black-forest-labs/FLUX.2-klein-4B` | Newer non-gated model, much smaller than FLUX.2-dev | Install Diffusers `main`, then use the FLUX.2 klein section below |
| 32 GB | `stabilityai/stable-diffusion-3.5-large-turbo` | High-quality still images with low step counts (gated) | Accept model terms on HF, set `HF_TOKEN`, then use the SD3.5 section below |
| 64 GB | `Qwen/Qwen-Image-2512` | Strong prompt following and text rendering (large model) | Same as Diffusers setup; if pipeline import fails, use Diffusers `main` (see install section above) |
| 128 GB | `black-forest-labs/FLUX.2-dev` | Very high quality (very large; non-commercial license; gated) | Accept model terms on HF, set `HF_TOKEN`, then use the FLUX section below |

macOS Metal (Apple Silicon) quick picks:

- If you want **local quantized FLUX.2** on Metal: start with `AbstractFramework/flux.2-klein-4b-4bit`, then use `AbstractFramework/flux.2-klein-4b-8bit` when quality is more important than memory.
- If you want a fast local FLUX.2 for iteration: `AbstractFramework/flux.2-klein-4b-4bit` through MLX-Gen is usually the most practical Apple Silicon starting point.
- If you want strong prompt following + text rendering: use exact MLX-Gen ids such as `AbstractFramework/qwen-image-2512-4bit` or `AbstractFramework/ernie-image-turbo-4bit`, select the matching `-8bit` repo when quality is more important than memory, or use `Qwen/Qwen-Image-2512` / `baidu/ERNIE-Image-Turbo` through Diffusers on `mps` when you want the full Diffusers path.

Recommended default (local, cross-platform) — Stable Diffusion 1.5:

```bash
pip install "abstractvision[diffusers]"
abstractvision download stable-diffusion --provider diffusers
export ABSTRACTVISION_BACKEND=diffusers
export ABSTRACTVISION_MODEL_ID=runwayml/stable-diffusion-v1-5
export ABSTRACTVISION_DIFFUSERS_DEVICE=auto
abstractvision cli
```

Then type a prompt (plain text runs `/t2i`), or use `/t2i "..." --open`.

Jump to detailed recipes:
- Stable Diffusion 1.5: section **1) First local image (Diffusers)**
- FLUX.2-klein-4B: section **2) Next small model (FLUX.2-klein-4B)**
- OpenAI-compatible HTTP: section **2.1) OpenAI-compatible HTTP**
- Apple Silicon MLX-Gen: section **2.2) Apple Silicon MLX-Gen (q4 first)**
- Qwen Image: section **3) Qwen Image (Diffusers)**
- FLUX 2 details: section **4) FLUX 2 (Diffusers)**
- SD3.5: section **5) Stable Diffusion 3.5 (Diffusers, gated)**

---

## 1) First local image (Diffusers)

The REPL is cache-only by default, so it will not download model weights. Download the model separately first:

```bash
abstractvision download stable-diffusion --provider diffusers
```

```bash
# Required for this local Diffusers recipe.
export ABSTRACTVISION_BACKEND=diffusers
export ABSTRACTVISION_MODEL_ID=runwayml/stable-diffusion-v1-5
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

Start the interactive CLI (`abstractvision repl` remains an alias):

```bash
abstractvision cli
```

With `ABSTRACTVISION_BACKEND=diffusers` and `ABSTRACTVISION_MODEL_ID` set above, the REPL uses `runwayml/stable-diffusion-v1-5`:

```text
/set guidance_scale 7
/set seed 42
/t2i "a cinematic photo of a red fox in snow" --width 512 --height 512 --steps 10 --open
```

Change settings by changing `/set …` values, or pass flags per request:

```text
/t2i "a watercolor painting of a lighthouse" --width 512 --height 512 --steps 20 --seed 123 --guidance-scale 6.5 --open
```

---

## 2) Next small model (FLUX.2-klein-4B)

After Stable Diffusion 1.5 works, `black-forest-labs/FLUX.2-klein-4B` is the next recommended local test. It is
non-gated and much smaller than FLUX.2-dev, but it currently needs Diffusers from source because released Diffusers
may not include `Flux2KleinPipeline`.

```bash
pip install -U "abstractvision[diffusers-dev]"
pip install -U "git+https://github.com/huggingface/diffusers@main"
```

Quick REPL test:

```text
/backend diffusers black-forest-labs/FLUX.2-klein-4B mps float16
/t2i "a product photo of a matte black espresso machine" --width 1024 --height 1024 --steps 4 --guidance-scale 1.0 --open
```

Use `cuda float16` on NVIDIA, or `auto` if you want AbstractVision/Torch to pick the device.

---

## 2.1) OpenAI-compatible HTTP

Use this path if you already have a server that exposes OpenAI-shaped image endpoints (e.g. a local model server).

For unknown or local OpenAI-compatible servers, AbstractVision forwards local extension fields such as `steps`, `seed`, `guidance_scale`, `width`, and `height`. For the real OpenAI API and known GPT image models, it suppresses unsupported local-only fields and sends the narrower OpenAI request shape.

List provider-advertised models explicitly:

```bash
abstractvision provider-models --openai --task text_to_image
abstractvision provider-models --base-url http://localhost:1234/v1 --task text_to_image
```

One-shot (stores output via `LocalAssetStore` and prints an artifact ref + file path):

```bash
abstractvision t2i --base-url http://localhost:1234/v1 "a watercolor painting of a lighthouse" --width 512 --height 512 --steps 10 --open
```

Interactive CLI/REPL:

```bash
abstractvision cli
```

```text
/backend openai http://localhost:1234/v1
/t2i "a watercolor painting of a lighthouse" --width 512 --height 512 --steps 10 --open
```

If your server also supports video endpoints, configure them via `ABSTRACTVISION_TEXT_TO_VIDEO_PATH` / `ABSTRACTVISION_IMAGE_TO_VIDEO_PATH` (see [docs/reference/configuration.md](reference/configuration.md)).

---

## 2.2) Apple Silicon MLX-Gen (q4 first)

Use this path on Apple Silicon when you want local MLX-optimized image/video models
without running a separate server. AbstractVision uses the `mlx-gen` Python API
in-process and expects prepared model folders to exist in the Hugging Face
cache. It does not silently download weights during generation.

```bash
pip install "abstractvision[models,mlx-gen]"
abstractvision catalog --provider mlx-gen
abstractvision download AbstractFramework/flux.2-klein-4b-4bit --provider mlx-gen
abstractvision download AbstractFramework/qwen-image-2512-4bit --provider mlx-gen
abstractvision download AbstractFramework/qwen-image-edit-2511-4bit --provider mlx-gen
abstractvision download AbstractFramework/ernie-image-turbo-4bit --provider mlx-gen
abstractvision download AbstractFramework/ernie-image-turbo-8bit --provider mlx-gen
abstractvision download briaai/FIBO --provider mlx-gen
abstractvision download briaai/Fibo-lite --provider mlx-gen
abstractvision download briaai/Fibo-Edit --provider mlx-gen
abstractvision download prism-ml/bonsai-image-ternary-4B-mlx-2bit --provider mlx-gen
abstractvision download Wan-AI/Wan2.2-TI2V-5B-Diffusers --provider mlx-gen
abstractvision download AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit --provider mlx-gen
abstractvision download AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit --provider mlx-gen
```

The default prepared choices are q4 repos from the
[AbstractFramework/mlx-gen Hugging Face collection](https://huggingface.co/collections/AbstractFramework/mlx-gen/).
Select q8 variants by passing the exact `AbstractFramework/...-8bit` model id
when quality is paramount and memory permits it. Quantization is part of the
published model folder, not a generation-time parameter.
Qwen and ERNIE q4 folders can mix q4 and q8 components, but remain the default
prepared choice.
Bonsai ternary is different: `prism-ml/bonsai-image-ternary-4B-mlx-2bit` is a
pre-packed MLX checkpoint consumed directly by MLX-Gen. Use the exact repo id,
keep guidance at 1.0, and do not use it for image-to-image.

One-shot shell commands store the output in the local asset store and print the
artifact ref followed by the local content path. Add `--open` to open the output
after generation, or `--store-dir <dir>` to select the store directory.

Text-to-image from the shell:

```bash
abstractvision t2i --provider mlx-gen --model AbstractFramework/qwen-image-2512-4bit "a studio product photo of a white ceramic mug with the AbstractFramework logo" --steps 20 --guidance-scale 1.0 --open
abstractvision t2i --provider mlx-gen --model prism-ml/bonsai-image-ternary-4B-mlx-2bit "a bonsai tree in a quiet ceramic studio" --steps 4 --guidance-scale 1.0 --open
```

Image-to-image/edit from the shell:

```bash
abstractvision i2i --provider mlx-gen --model AbstractFramework/qwen-image-edit-2511-4bit --image ./input.png "replace the background with a clean white studio setup" --steps 20 --guidance-scale 2.5 --strength 0.75 --open
abstractvision i2i --provider mlx-gen --model briaai/Fibo-Edit --image ./input.png "remove the background and keep the object edges clean" --steps 20 --guidance-scale 4.0 --open
```

Text-to-video and first-frame image-to-video from the shell:

```bash
abstractvision t2v --provider mlx-gen --model AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit "a red fox walking through a snowy forest, cinematic" --width 432 --height 240 --frames 41 --fps 10 --steps 20 --guidance-scale 4.0 --open
abstractvision i2v --provider mlx-gen --model AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit --image ./first-frame.png "slow camera push-in" --width 432 --height 240 --frames 41 --fps 10 --steps 20 --guidance-scale 4.0 --open
```

Wan 2.2 A14B uses 16px width/height multiples; `432x240` is valid for low-cost
local checks, while larger native sizes are more appropriate for quality review.

Video examples (5s MP4):

<div class="af_media_carousel" aria-label="AbstractVision text-to-video examples">
  <button class="af_media_carousel__btn af_media_carousel__btn--prev" type="button" aria-label="Previous video">‹</button>
  <div class="af_media_carousel__track" tabindex="0">
    <figure class="af_media_carousel__item">
      <video class="af_media_carousel__video" controls muted playsinline loop preload="metadata">
        <source src="../assets/videos/video-river.mp4" type="video/mp4" />
      </video>
      <figcaption class="af_media_carousel__caption"><strong>River</strong> — smooth, clean motion (5s)</figcaption>
    </figure>
    <figure class="af_media_carousel__item">
      <video class="af_media_carousel__video" controls muted playsinline loop preload="metadata">
        <source src="../assets/videos/video-space.mp4" type="video/mp4" />
      </video>
      <figcaption class="af_media_carousel__caption"><strong>Space</strong> — cinematic scene (5s)</figcaption>
    </figure>
    <figure class="af_media_carousel__item">
      <video class="af_media_carousel__video" controls muted playsinline loop preload="metadata">
        <source src="../assets/videos/t2v-example.mp4" type="video/mp4" />
      </video>
      <figcaption class="af_media_carousel__caption"><strong>T2V example</strong> — prior sample (5s)</figcaption>
    </figure>
  </div>
  <button class="af_media_carousel__btn af_media_carousel__btn--next" type="button" aria-label="Next video">›</button>
</div>

MLX-Gen image commands accept `--progress` for step progress. MLX-Gen video
commands print denoise-step progress with frame context while the video is
running. Use `--no-progress` for quiet video scripts. A complete local example
gallery with commands and bundled outputs is available in
[MLX-Gen local examples](mlx-gen-local-examples.md).

Interactive CLI/REPL (`abstractvision cli`; `abstractvision repl` remains an
alias) uses the same backend and request normalization:

```text
/backend mlx-gen AbstractFramework/flux.2-klein-4b-4bit
/t2i "a product photo of a matte black espresso machine" --steps 4 --guidance-scale 1.0 --open

/backend mlx-gen AbstractFramework/qwen-image-edit-2511-4bit
/i2i --image ./input.png "make it watercolor" --steps 20 --guidance-scale 2.5 --open

/backend mlx-gen briaai/FIBO
/t2i "a studio product photo of a white ceramic mug with the AbstractFramework logo" --steps 50 --guidance-scale 4.0 --open

/backend mlx-gen prism-ml/bonsai-image-ternary-4B-mlx-2bit
/t2i "a bonsai tree in a quiet ceramic studio" --steps 4 --guidance-scale 1.0 --open

/backend mlx-gen AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit
/t2v "a red fox walking through a snowy forest, cinematic" --width 432 --height 240 --frames 41 --fps 10 --steps 20 --guidance-scale 4.0 --open
/backend mlx-gen AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit
/i2v --image ./first-frame.png "slow camera push-in" --width 432 --height 240 --frames 41 --fps 10 --steps 20 --guidance-scale 4.0 --open
```

Interactive `/t2v` and `/i2v` use the same default progress display; add
`--no-progress` to suppress it.

Legacy `mflux` provider names and routed ids still work as aliases, but new
configuration should use `mlx-gen`.

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

These features follow the Diffusers download setting. The REPL is cache-only by default, so pre-download adapters or
Rapid-AIO weights separately before using repo ids here. If you intentionally want runtime downloads, set:

```bash
export ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1
```

Tip: Qwen Image Edit pipelines often default to a ~1MP output resolution when `width`/`height` are omitted. AbstractVision
defaults to the input image size for image-to-image edits to avoid unexpected memory spikes. To request a specific output
resolution, pass `--width`/`--height` flags in the REPL (forwarded via `request.extra`) or set them in the playground
Extra JSON field.

LoRA example (REPL; note: `loras_json` is forwarded via `request.extra`):

```text
/backend diffusers Qwen/Qwen-Image-Edit-2511 mps bfloat16
/i2i --image ./input.png "make it watercolor" --steps 8 --guidance-scale 1 --loras_json '[{"source":"lightx2v/Qwen-Image-Edit-2511-Lightning","scale":1.0}]' --open
```

Rapid-AIO example (distilled transformer override; Qwen Image Edit):

```text
/backend diffusers Qwen/Qwen-Image-Edit-2511 mps bfloat16
/i2i --image ./input.png "make it watercolor" --steps 4 --guidance-scale 1 --rapid_aio_repo linoyts/Qwen-Image-Edit-Rapid-AIO --open
```

---

## 4) FLUX 2 (Diffusers)

FLUX 2 models in the registry:

- `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0, not gated)
- `black-forest-labs/FLUX.2-klein-9B` (non-commercial license, gated on Hugging Face)
- `black-forest-labs/FLUX.2-dev` (non-commercial license, gated on Hugging Face)

Sanity check:

```bash
python -c "import diffusers; print(diffusers.__version__)"
```

Notes:
- `FLUX.2-dev` uses Diffusers `Flux2Pipeline` and works on released Diffusers (0.36+).
- `FLUX.2-klein-4B` and `FLUX.2-klein-9B` use `Flux2KleinPipeline`, which is not available in the released Diffusers (0.36.0). It currently
  requires installing Diffusers from source (with the `diffusers-dev` extra for compatible dependency pins):
  - `pip install -U "abstractvision[diffusers-dev]"`
  - `pip install -U "git+https://github.com/huggingface/diffusers@main"`

Recommended first FLUX example (`FLUX.2-klein-4B`, not gated):

```text
/backend diffusers black-forest-labs/FLUX.2-klein-4B mps float16
/t2i "a product photo of a matte black espresso machine" --width 1024 --height 1024 --steps 4 --guidance-scale 1.0 --seed 0 --open
```

Example (`FLUX.2-klein-9B`, gated; requires Diffusers `main` and HF access):

```text
/backend diffusers black-forest-labs/FLUX.2-klein-9B mps float16
/t2i "a minimalist product photo of a matte black espresso machine, studio lighting" --width 1024 --height 1024 --steps 4 --guidance-scale 1.0 --seed 0 --open
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

## 6) GGUF diffusion models (stable-diffusion.cpp)

If you downloaded a GGUF diffusion model (like Qwen Image GGUF or FLUX.2 GGUF), Diffusers cannot load it. Use the stable-diffusion.cpp backend instead (either via pip-installed python bindings or `sd-cli`).

### 6.1 Install stable-diffusion.cpp runtime

The base `pip install abstractvision` path does not install local inference runtimes. Use one of these explicit stable-diffusion.cpp runtime choices:

```bash
pip install "abstractvision[sdcpp]"
```

This pip binding path is convenient, but it may require a native build or run **CPU-only** depending on how the wheel was built.

Alternative (external executable):

- Download `sd-cli` from: <https://github.com/leejet/stable-diffusion.cpp/releases>
- Ensure `sd-cli` is in your `PATH` (or pass a full path as the last arg to `/backend sdcpp …`).

On macOS (Apple Silicon), **`sd-cli` is the recommended path** to get **Metal** acceleration. If you see `Using CPU backend`,
install `sd-cli` and re-run in CLI mode.

### 6.2 Single-file Stable Diffusion model

This is the lowest-friction `sdcpp` shape: one model file plus an optional `sd-cli` path. Use it for Stable Diffusion
1.x/2.x/SDXL checkpoints or GGUF conversions that stable-diffusion.cpp can load as `--model`.

```bash
abstractvision cli
```

```text
/backend sdcpp /path/to/sd-v1-5.gguf /path/to/sd-cli
/t2i "a watercolor painting of a lighthouse" --width 512 --height 512 --steps 10 --open
```

If `sd-cli` is already in your `PATH`, you can omit the final `/path/to/sd-cli` argument. If it is not available,
AbstractVision falls back to `stable-diffusion-cpp-python` when that package is installed, for example through `pip install "abstractvision[sdcpp]"`.

### 6.3 Curated `sdcpp` bundles

For Qwen Image and FLUX GGUF, external users should not have to manually locate a VAE or text encoder.
Download the curated bundle once, then use the model key everywhere:

```bash
abstractvision download qwen-image --provider sdcpp
abstractvision download flux2-klein-base-4b --provider sdcpp
```

AbstractVision resolves the required side artifacts from the Hugging Face cache automatically:

- CLI one-shot: `abstractvision t2i --provider sdcpp --model flux2-klein-base-4b "..."`
- REPL: `/backend sdcpp flux2-klein-base-4b /path/to/sd-cli`
- Playground / AbstractCore plugin: set the model key and let the package resolve the cached bundle

If a required companion file is missing, the package now fails early with a precise `download ... --provider sdcpp`
hint instead of starting generation and then failing deep in the runtime.

### 6.4 Run Qwen Image with `sdcpp`

```bash
abstractvision cli
```

Then:

```text
/backend sdcpp qwen-image /path/to/sd-cli
/set width 1024
/set height 1024
/t2i "a cinematic photo of a red fox in snow" --sampling-method euler --offload-to-cpu --diffusion-fa --flow-shift 3 --open
```

Any extra `--flag` you pass (like `--sampling-method euler`) is forwarded to the backend as `extra`.
- CLI mode: forwarded to `sd-cli`
- Python bindings mode: keys are mapped to python binding kwargs when supported; unsupported keys are ignored (see [`../src/abstractvision/backends/stable_diffusion_cpp.py`](../src/abstractvision/backends/stable_diffusion_cpp.py))
- Diffusers backend: only forwards kwargs that the pipeline `__call__` accepts; unknown keys are ignored (see [`../src/abstractvision/backends/huggingface_diffusers.py`](../src/abstractvision/backends/huggingface_diffusers.py))

### 6.5 FLUX.2-klein-4B / FLUX.2-klein-base-4B (GGUF) example

The curated `sdcpp` presets now download the main GGUF plus the required companion artifacts and let you refer to the
model by key afterward.

Example:

```text
/backend sdcpp flux2-klein-base-4b /path/to/sd-cli
/t2i "a product photo of a matte black espresso machine" --steps 4 --guidance-scale 1.0 --sampling-method euler --diffusion-fa --offload-to-cpu --open
```

Advanced/manual mode is still supported if you want to wire explicit component paths yourself:

```text
/backend sdcpp /path/to/flux-2-klein-base-4b-Q8_0.gguf /path/to/vae/diffusion_pytorch_model.safetensors /path/to/Qwen3-4B-Q4_K_M.gguf /path/to/sd-cli
```

FLUX.2-dev and Qwen Image GGUF are still heavier follow-ups, but the single-file Stable Diffusion path or the curated
klein bundle above should be the first local test on a fresh machine.

---

## 7) Web UI testing (optional): Playground

This repo includes a self-contained web UI and local API server. It is owned by
AbstractVision and does not require AbstractCore. Treat it as a local/dev
testing surface; use AbstractCore/Gateway for production routing,
authentication, and browser-origin policy.

### 7.1 Start the playground

```bash
abstractvision playground --port 8091
```

Open:

- `http://127.0.0.1:8091/vision_playground.html`

In the UI:
- The API Base URL defaults to the same process that serves the page
- Each task tab has its own model selector and unload button
- Switching models in a tab unloads the previously active backend first to free memory before the replacement model is loaded
- Generate (T2I), upload an input image (I2I), or inspect video-capable backend status. The Playground is still a dev surface; prefer the shell/REPL or AbstractCore for MLX-Gen Wan video smoke tests.

For the endpoint list, see `playground/README.md`.

### 7.2 Local text→video status

The packaged local Diffusers `text_to_video` groundwork remains experimental and
is currently disabled from the normal local surfaces. The practical local Apple
Silicon video path is MLX-Gen Wan 2.2, preferably the task-specific A14B
packages when memory allows.

Current policy:
- use `abstractvision t2v --provider mlx-gen --model AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` and `abstractvision i2v --provider mlx-gen --model AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit` for local Apple Silicon Wan A14B video;
- use the OpenAI-compatible backend when video is served remotely; and
- keep Diffusers local video behind the experimental follow-up in [`docs/backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md`](backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md).
