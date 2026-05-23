# AbstractVision

[![PyPI version](https://img.shields.io/pypi/v/abstractvision.svg)](https://pypi.org/project/abstractvision/)
[![CI](https://github.com/lpalbou/AbstractVision/actions/workflows/ci.yml/badge.svg)](https://github.com/lpalbou/AbstractVision/actions/workflows/ci.yml)
[![Tested Python](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2Flpalbou%2FAbstractVision%2Fmain%2F.github%2Fworkflows%2Fci.yml&query=%24.jobs.test.strategy.matrix%5B%22python-version%22%5D&label=tested%20python&color=blue)](https://github.com/lpalbou/AbstractVision/actions/workflows/ci.yml)
[![license](https://img.shields.io/github/license/lpalbou/AbstractVision)](https://github.com/lpalbou/AbstractVision/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/lpalbou/AbstractVision?style=social)](https://github.com/lpalbou/AbstractVision/stargazers)

Model-agnostic generative vision API (images, optional video) for Python and the Abstract* ecosystem.

## What you get

- A small orchestration API: [`VisionManager`](src/abstractvision/vision_manager.py)
- A packaged capability registry (“what models can do”): [`VisionModelCapabilitiesRegistry`](src/abstractvision/model_capabilities.py) backed by [`vision_model_capabilities.json`](src/abstractvision/assets/vision_model_capabilities.json)
- Shared model metadata that now also drives local catalog surfacing and backend request normalization across the CLI, playground, and AbstractCore paths
- Optional artifact-ref outputs (small JSON refs): [`LocalAssetStore`](src/abstractvision/artifacts.py) and [`RuntimeArtifactStoreAdapter`](src/abstractvision/artifacts.py)
- Built-in backends (execution engines): [`src/abstractvision/backends/`](src/abstractvision/backends/)
  - OpenAI-compatible HTTP: [`openai_compatible.py`](src/abstractvision/backends/openai_compatible.py)
  - Local Diffusers: [`huggingface_diffusers.py`](src/abstractvision/backends/huggingface_diffusers.py)
  - Local stable-diffusion.cpp / GGUF: [`stable_diffusion_cpp.py`](src/abstractvision/backends/stable_diffusion_cpp.py)
  - Local MFLUX / MLX bridge for curated Apple Silicon presets: [`mflux.py`](src/abstractvision/backends/mflux.py)
- CLI for manual testing (`abstractvision cli`, legacy alias: `abstractvision repl`): [`abstractvision`](src/abstractvision/cli.py)
- Self-contained local Playground UI/API: [`playground/vision_playground.html`](playground/vision_playground.html) (docs: [`playground/README.md`](playground/README.md))

## How it fits together (diagram)

```mermaid
flowchart LR
  Caller[Python / CLI / AbstractCore] --> VM[VisionManager]
  VM --> BE[VisionBackend]
  BE --> VM
  VM -->|optional| Store[MediaStore]
  Store --> Ref[Artifact ref dict]
  VM -->|no store| Asset["GeneratedAsset (bytes + mime)"]
```

## Status (current backend support)

- Development status: **Alpha** (0.x). The public API is stable-by-design, but breaking changes may still happen and will be called out in `CHANGELOG.md`.
- Built-in backends implement images: `text_to_image` and `image_to_image`.
- Local MFLUX supports `text_to_image`, and supports `image_to_image` for the curated FLUX.2 klein presets (`flux2-klein-4b`, `flux2-klein-9b`) (mask edits are not supported yet).
- Local Diffusers `text_to_video` remains experimental and is temporarily disabled from the normal local runtime surfaces pending [`docs/backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md`](docs/backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md).
- `image_to_video` is currently supported only via the OpenAI-compatible backend **when** endpoints are configured.
- `multi_view_image` is part of the public API (`VisionManager.generate_angles`) but no built-in backend implements it yet.

Details: [`docs/reference/backends.md`](docs/reference/backends.md).

## Installation

```bash
pip install abstractvision
```

The base install is lightweight. It includes the shared API, capability
registry, artifact helpers, CLI, AbstractCore plugin entry point, and the
stdlib OpenAI-compatible HTTP backend. Local inference runtimes are explicit
extras.

Optional extras:

| Extra | Use |
|---|---|
| `abstractvision[openai]` | Official OpenAI provider intent marker; no SDK dependency today. |
| `abstractvision[openai-compatible]` | Generic local/remote OpenAI-shaped endpoint intent marker; stdlib-only today. |
| `abstractvision[models]` | Curated Hugging Face download helpers for cache-backed local 8-bit vision model presets. |
| `abstractvision[diffusers]` | Install Torch/Diffusers and related packages for local Diffusers generation. |
| `abstractvision[huggingface]` | Compatibility alias for callers that still request the historical Diffusers extra. |
| `abstractvision[sdcpp]` | Install `stable-diffusion-cpp-python` for the pip binding fallback. |
| `abstractvision[mflux]` | Install the optional MFLUX/MLX Apple Silicon image runtime. |
| `abstractvision[local]` | Convenience for both local backend dependency sets, including `diffusers` and `sdcpp`. |
| `abstractvision[all]` | All runtime backend dependencies, without contributor tooling. |
| `abstractvision[apple]` / `abstractvision[all-apple]` | Native macOS Python profile: Diffusers/Torch MPS, stable-diffusion.cpp bindings, and MFLUX. |
| `abstractvision[gpu]` | GPU Diffusers/Torch profile. Install a CUDA/ROCm-enabled PyTorch wheel when needed. |
| `abstractvision[all-gpu]` | Full GPU-relevant local vision profile: Diffusers plus stable-diffusion.cpp bindings. |
| `abstractvision[abstractcore]` | Compatibility marker only; AbstractCore is still supplied by the host application. |

`stable-diffusion-cpp-python` is currently constrained below `0.4.6` because
that release's source distribution is missing vendored CMake files required by
native Linux builds.

Contributor-only extras:

| Extra | Use |
|---|---|
| `abstractvision[diffusers-dev]` / `abstractvision[huggingface-dev]` | Looser dependency pins for newer/unreleased Diffusers pipelines; install Diffusers `main` separately if needed. |
| `abstractvision[test]` | Local test dependencies. |
| `abstractvision[docs]` | Documentation build tooling. |
| `abstractvision[dev]` | Full contributor workflow: tests, docs, build, lint, formatting, and pre-commit. Do not use this as an application runtime profile. |

Note (CUDA): on Windows/Linux, `pip install "abstractvision[diffusers]"` may install a CPU-only PyTorch build. If you want to use an NVIDIA GPU, install a CUDA-enabled PyTorch build first (see <https://pytorch.org/get-started/locally/>) and verify `torch.cuda.is_available()` is `True`.

AbstractCore is not installed by AbstractVision. When an AbstractCore application
has AbstractVision installed in the same environment, AbstractCore can discover
the plugin entry point and use the integration modules lazily.

If you hit “missing pipeline class” errors for newer model families, see [`docs/getting-started.md`](docs/getting-started.md). In that case you may need Diffusers from source (`main`):

```bash
pip install -U "abstractvision[diffusers-dev]"
pip install -U "git+https://github.com/huggingface/diffusers@main"
```

For local development from a repo checkout:

```bash
pip install -e ".[dev]"
```

## Usage

Start here:
- Getting started: [`docs/getting-started.md`](docs/getting-started.md)
- FAQ: [`docs/faq.md`](docs/faq.md)
- Troubleshooting: [`docs/troubleshooting.md`](docs/troubleshooting.md)
- API reference: [`docs/api.md`](docs/api.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)
- Capability registry + catalog policy: [`docs/reference/capabilities-registry.md`](docs/reference/capabilities-registry.md), [`docs/adr/README.md`](docs/adr/README.md)
- Docs index: [`docs/README.md`](docs/README.md)

### First local model (8-bit first)

For local model downloads, prefer the curated 8-bit presets first. On macOS
they resolve to MLX artifacts that declare the `mflux` engine; on non-macOS
systems the default target is GGUF or an equivalent local-runtime artifact. The
downloader stores curated presets in the Hugging Face cache by default and
imports older `~/models/<preset>` trees on first use. It does not fall back to
full models unless you pass `--allow-non-8bit`.

```bash
pip install "abstractvision[models,mflux]"
abstractvision model-presets
abstractvision catalog --provider mflux
# Tip: `--provider mflux` implies `--target mlx` (you usually set one or the other).
abstractvision download flux2-klein-4b --provider mflux
abstractvision download flux2-klein-9b --provider mflux
abstractvision download qwen-image --provider mflux
abstractvision download z-image-turbo --provider mflux
abstractvision t2i --provider mflux --model flux2-klein-4b "a product photo of a matte black espresso machine" --steps 4 --guidance-scale 1.0
```

The shipped MFLUX backend currently supports the curated `flux2-klein-4b`,
`flux2-klein-9b`, `qwen-image`, and `z-image-turbo` preset families for local
`text_to_image`. For the FLUX.2 klein presets, it also supports `image_to_image`
edits (mask edits are not supported yet). Edit strength is passed as `strength`
and normalized to the underlying MFLUX `image_strength` parameter.

Stable Diffusion does not currently have a curated MLX 8-bit preset in
AbstractVision, so full Diffusers downloads remain explicit.

Install the Diffusers runtime extra, download a Diffusers snapshot, then select
the Diffusers backend explicitly:

```bash
pip install "abstractvision[models,diffusers]"
abstractvision catalog --provider diffusers
# Tip: `--provider diffusers` implies `--target diffusers` (you usually set one or the other).
abstractvision download stable-diffusion --provider diffusers
abstractvision download sd1.4 --provider diffusers
abstractvision download sd1.5-inpaint --provider diffusers
abstractvision download sdxl-base --provider diffusers
abstractvision download sdxl-inpaint --provider diffusers
abstractvision download sd3-medium --provider diffusers
abstractvision download sd3.5-large --provider diffusers
abstractvision download ernie-image --provider diffusers
abstractvision download qwen-image-edit-2511 --provider diffusers
abstractvision download flux2-dev --provider diffusers
export ABSTRACTVISION_BACKEND=diffusers
export ABSTRACTVISION_MODEL_ID=runwayml/stable-diffusion-v1-5
export ABSTRACTVISION_DIFFUSERS_DEVICE=auto
abstractvision cli
```

Notes:
- `abstractvision download qwen-image-edit-2511 --provider diffusers` downloads the curated official 16-bit Diffusers snapshot.
- `GLM-Image` remains in the packaged registry, but local Diffusers `GLM-Image` is temporarily disabled pending the follow-up tracked in [`docs/backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md`](docs/backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md).
- `CogVideoX-2b` downloads are still available for experimentation, but local `text_to_video` is currently marked experimental and disabled from the normal product surfaces.

For a fresh cache, you can also permit the interactive CLI to download missing files:

```bash
ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1 abstractvision cli
```

More recommendations by VRAM: [`docs/getting-started.md`](docs/getting-started.md).

### Capability-driven model selection

```python
from abstractvision import VisionModelCapabilitiesRegistry

reg = VisionModelCapabilitiesRegistry()
assert reg.supports("runwayml/stable-diffusion-v1-5", "text_to_image")
assert reg.supports("Qwen/Qwen-Image-Edit-2511", "image_to_image")

print(reg.list_tasks())
print(reg.models_for_task("text_to_image"))
print(reg.models_for_task("image_to_image"))
```

### Backend wiring + generation (artifact outputs)

The base install is import-light and does not install Torch/Diffusers. Heavy
local backend modules are imported lazily (see [`src/abstractvision/backends/__init__.py`](src/abstractvision/backends/__init__.py)).
Install `abstractvision[diffusers]` for local Diffusers, or
`abstractvision[sdcpp]` for the optional stable-diffusion.cpp python binding
fallback.

```python
from abstractvision import LocalAssetStore, VisionManager, VisionModelCapabilitiesRegistry, is_artifact_ref
from abstractvision.backends import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend

reg = VisionModelCapabilitiesRegistry()

backend = OpenAICompatibleVisionBackend(
    config=OpenAICompatibleBackendConfig(
        base_url="http://localhost:1234/v1",
        api_key="YOUR_KEY",      # optional for local servers
        model_id="REMOTE_MODEL", # optional (server-dependent)
    )
)

vm = VisionManager(
    backend=backend,
    store=LocalAssetStore(),         # enables artifact-ref outputs
    model_id="Qwen/Qwen-Image-Edit-2511",  # optional: capability gating
    registry=reg,                   # optional: reuse loaded registry
)

out = vm.generate_image("a cinematic photo of a red fox in snow")
assert is_artifact_ref(out)
print(out)  # {"$artifact": "...", "content_type": "...", ...}

png_bytes = vm.store.load_bytes(out["$artifact"])  # type: ignore[union-attr]
```

When installed next to AbstractCore, AbstractVision is also discovered as a
`llm.vision` capability plugin. The plugin defaults to the official OpenAI
image endpoint (`https://api.openai.com/v1`) and reads `OPENAI_API_KEY`.
Set `OPENAI_BASE_URL` when you need a local or remote compatible `/v1` server,
and use the same `OPENAI_API_KEY` bearer token if that endpoint requires auth.
Set `ABSTRACTVISION_BACKEND=openai-compatible` when you want to force
compatible-endpoint semantics. Set `ABSTRACTVISION_MODEL_ID`,
`OPENAI_IMAGE_MODEL_ID`, or `OPENAI_IMAGE_MODEL` when you need an explicit
image model (static default OpenAI model: `gpt-image-1`). AbstractVision does
not query provider `/models` catalogs to discover or select image models
automatically, but you can inspect them explicitly with
`abstractvision provider-models`, `VisionManager.list_provider_models(...)`,
or the AbstractCore plugin method `llm.vision.list_provider_models(...)`.
After inspection, set the model env var explicitly for newer provider models
when available to your account. Set `ABSTRACTVISION_BACKEND=mflux`,
`ABSTRACTVISION_BACKEND=diffusers`, or `ABSTRACTVISION_BACKEND=sdcpp` when you
want AbstractCore to launch local AbstractVision generation directly. For
MFLUX, set `ABSTRACTVISION_MFLUX_MODEL=flux2-klein-4b` or use routed model ids
such as `mflux/flux2-klein-4b`.

### Interactive testing (CLI)

```bash
abstractvision models
abstractvision provider-models --openai --task text_to_image
abstractvision provider-models --base-url http://localhost:1234/v1 --task text_to_image
abstractvision tasks
abstractvision show-model runwayml/stable-diffusion-v1-5

abstractvision cli
```

Inside the interactive CLI:

```text
/t2i "a watercolor painting of a lighthouse" --width 512 --height 512 --steps 10 --open
```

For a newer but still relatively small local model, try `black-forest-labs/FLUX.2-klein-4B` after installing Diffusers
from source (see [`docs/getting-started.md`](docs/getting-started.md)):

```text
/backend diffusers black-forest-labs/FLUX.2-klein-4B mps float16
/t2i "a product photo of a matte black espresso machine" --steps 4 --guidance-scale 1.0 --open
```

Local Diffusers `text_to_video` remains experimental and is temporarily
disabled from the normal bundled local surfaces. Use the OpenAI-compatible
backend for video today, or track the local follow-up in
[`docs/backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md`](docs/backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md).

For Apple Silicon 8-bit local generation through MFLUX:

```text
/backend mflux flux2-klein-4b
/t2i "a product photo of a matte black espresso machine" --steps 4 --guidance-scale 1.0 --open
```

OpenAI-compatible server example:

```text
/backend openai http://localhost:1234/v1
/t2i "a watercolor painting of a lighthouse" --width 512 --height 512 --steps 10 --open
```

The CLI/REPL can also be configured via `ABSTRACTVISION_*` env vars; see [`docs/reference/configuration.md`](docs/reference/configuration.md).

### Local web playground

The playground is owned by AbstractVision and runs without AbstractCore. It is
a local/dev testing surface; use AbstractCore/Gateway for production routing,
authentication, and browser-origin policy.

```bash
abstractvision playground --port 8091
```

Open `http://127.0.0.1:8091/vision_playground.html`. The page and the API are served by the same process.

Current behavior:
- The UI is split into task tabs (`Text→Image`, `Image→Image`, `Text→Video`, and a placeholder `Image→Video` tab for later work).
- Each active task tab has its own model selector and unload button. Switching models in a tab unloads the current active backend first to free memory before loading the replacement.
- The Image→Image tab is enabled only for models that both advertise `image_to_image` in the packaged capability registry and remain enabled by the selected backend.
- MFLUX FLUX.2 klein presets are surfaced for `Image→Image` edits (mask edits are not supported yet).
- The Text→Video tab is experimental; the bundled local server currently does not advertise a shipped local model there.
- Model-specific request normalization happens at the API/backend layer, not just in the page.
- Local video export packages generated frames into MP4 via an external `ffmpeg` binary on `PATH`.
- Response logs intentionally show only a shortened `b64_json` preview instead of the full base64 image payload.

One-shot commands default to the OpenAI-compatible HTTP backend, but they also support local providers:

```bash
abstractvision t2i --base-url http://localhost:1234/v1 "a studio photo of an espresso machine"
abstractvision i2i --base-url http://localhost:1234/v1 --image ./input.png "make it watercolor"
abstractvision t2i --provider diffusers --model qwen-image "a studio photo of an espresso machine"
```

#### Local GGUF via stable-diffusion.cpp

If you want to run GGUF diffusion models locally, use the stable-diffusion.cpp backend (`sdcpp`). Start with a
single-file Stable Diffusion model when possible; Qwen Image and FLUX GGUF component sets are heavier.

Recommended:
- `abstractvision` auto-installs `sd-cli` into `~/.abstractvision/bin` on first use (set `ABSTRACTVISION_SDCPP_AUTO_INSTALL=0` to disable).
- If you prefer python bindings: install `abstractvision[sdcpp]` (uses `stable-diffusion-cpp-python`).

Alternative (external executable): install `sd-cli` from <https://github.com/leejet/stable-diffusion.cpp/releases>.

In the REPL:

```text
/backend sdcpp /path/to/sd-v1-5.gguf /path/to/sd-cli
/t2i "a watercolor painting of a lighthouse" --width 512 --height 512 --steps 10 --open
```

Curated FLUX/Qwen GGUF bundle example:

```bash
abstractvision download flux2-klein-base-4b --provider sdcpp
abstractvision download qwen-image-edit-2511-gguf --provider sdcpp
```

```text
/backend sdcpp flux2-klein-base-4b /path/to/sd-cli
/t2i "a product photo of a matte black espresso machine" --steps 4 --guidance-scale 1.0 --sampling-method euler --diffusion-fa --offload-to-cpu --open
```

The package resolves the required VAE and text-encoder companions from the cache automatically for curated `sdcpp`
model keys. Manual component wiring remains available for advanced cases.

Extra flags are forwarded via `request.extra`. In CLI mode they are forwarded to `sd-cli`; in python bindings mode, keys are mapped to python binding kwargs when supported and unsupported keys are ignored.

### AbstractCore tool integration (artifact refs)

If you’re using AbstractCore tool calling, AbstractVision can expose vision tasks as tools:

```python
from abstractvision.integrations.abstractcore import make_vision_tools

tools = make_vision_tools(vision_manager=vm, model_id="Qwen/Qwen-Image-Edit-2511")
```

Install `abstractcore` in the host application environment when you use these helpers; it is not pulled in by AbstractVision.

## AbstractFramework ecosystem

AbstractVision is part of the **AbstractFramework** ecosystem and is designed to compose with:

- **AbstractFramework** (project hub): <https://github.com/lpalbou/AbstractFramework>
- **AbstractCore** (orchestration + tool calling): <https://github.com/lpalbou/abstractcore>
- **AbstractRuntime** (runtime services, including artifact storage): <https://github.com/lpalbou/abstractruntime>

In practice:
- AbstractVision standardizes *generative vision outputs* (image/video) behind `VisionManager`.
- AbstractCore can discover and use AbstractVision via the capability plugin (`src/abstractvision/integrations/abstractcore_plugin.py`) or you can expose vision tasks as tools (`src/abstractvision/integrations/abstractcore.py`).
- Artifact refs returned by AbstractVision are designed to travel across processes; `RuntimeArtifactStoreAdapter` bridges to an AbstractRuntime-style artifact store (`src/abstractvision/artifacts.py`).

## Project

- Release notes: [`CHANGELOG.md`](CHANGELOG.md)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security: [`SECURITY.md`](SECURITY.md)
- Acknowledgments: [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md)
- Agent docs: [`llms.txt`](llms.txt) and [`llms-full.txt`](llms-full.txt)

## Requirements

- Python >= 3.9

## License

MIT License - see LICENSE file for details.

## Author

Laurent-Philippe Albou

## Contact

contact@abstractcore.ai
