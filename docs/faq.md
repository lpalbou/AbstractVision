# FAQ

See also:
- Getting started: [docs/getting-started.md](getting-started.md)
- API reference: [docs/api.md](api.md)
- Architecture: [docs/architecture.md](architecture.md)
- Backends: [docs/reference/backends.md](reference/backends.md)
- Configuration: [docs/reference/configuration.md](reference/configuration.md)

## What is AbstractVision?

AbstractVision is a small, model-agnostic API for **generative vision outputs** (images, optional video) with:
- a small orchestrator ([`VisionManager`](../src/abstractvision/vision_manager.py))
- pluggable execution engines (“backends”) in [`../src/abstractvision/backends/`](../src/abstractvision/backends/)
- a packaged capability registry ([`vision_model_capabilities.json`](../src/abstractvision/assets/vision_model_capabilities.json))
- optional artifact-ref outputs via stores ([`../src/abstractvision/artifacts.py`](../src/abstractvision/artifacts.py))

## How does AbstractVision fit into AbstractFramework?

AbstractVision is part of the **AbstractFramework** ecosystem:

- **AbstractFramework** (project hub): <https://github.com/lpalbou/AbstractFramework>
- **AbstractCore** (orchestration + tool calling): <https://github.com/lpalbou/abstractcore>
- **AbstractRuntime** (runtime services, including artifact storage): <https://github.com/lpalbou/abstractruntime>

Where AbstractVision fits:
- It standardizes *generative vision outputs* behind `VisionManager` (library mode).
- AbstractCore can discover and use AbstractVision via the capability plugin (see [`../src/abstractvision/integrations/abstractcore_plugin.py`](../src/abstractvision/integrations/abstractcore_plugin.py) and the entry point in [`../pyproject.toml`](../pyproject.toml)).
- Artifact refs are designed to cross process boundaries; `RuntimeArtifactStoreAdapter` bridges to an AbstractRuntime-style artifact store (see [`../src/abstractvision/artifacts.py`](../src/abstractvision/artifacts.py)).

## What does AbstractVision support today?

- Built-in backends implement **images**: `text_to_image` and `image_to_image`.
- Video (`text_to_video`, `image_to_video`) works only via the OpenAI-compatible backend **when** video endpoints are configured.
- `multi_view_image` exists in the public API (`VisionManager.generate_angles`) but no built-in backend implements it yet (they raise `CapabilityNotSupportedError`).

Details: [docs/reference/backends.md](reference/backends.md).

## Which backend should I use?

- **OpenAI-compatible HTTP** ([`../src/abstractvision/backends/openai_compatible.py`](../src/abstractvision/backends/openai_compatible.py)): call a server that exposes OpenAI-shaped image endpoints (and optional video endpoints).
- **Diffusers (local)** ([`../src/abstractvision/backends/huggingface_diffusers.py`](../src/abstractvision/backends/huggingface_diffusers.py)): run Diffusers pipelines locally (heavy deps).
- **stable-diffusion.cpp (local GGUF)** ([`../src/abstractvision/backends/stable_diffusion_cpp.py`](../src/abstractvision/backends/stable_diffusion_cpp.py)): run GGUF diffusion models via `sd-cli` or `stable-diffusion-cpp-python`.

## What model should I start with (local)?

If you’re running locally via the Diffusers backend and want a reliable starting point, we recommend:

- **Default / ≤16GB VRAM (cross-platform)**: `runwayml/stable-diffusion-v1-5`

Quickstart:

```bash
abstractvision download-model stable-diffusion --provider diffusers
export ABSTRACTVISION_BACKEND=diffusers
export ABSTRACTVISION_MODEL_ID=runwayml/stable-diffusion-v1-5
export ABSTRACTVISION_DIFFUSERS_DEVICE=auto
abstractvision repl
```

More model recommendations (by VRAM tier) are in [docs/getting-started.md](getting-started.md).

After that works, `black-forest-labs/FLUX.2-klein-4B` is the recommended next local test for a newer non-gated model
(it currently requires Diffusers from source).

## Does `abstractvision t2i` run locally?

`abstractvision t2i` / `abstractvision i2i` are one-shot helpers for the **OpenAI-compatible HTTP backend** ([`../src/abstractvision/cli.py`](../src/abstractvision/cli.py)).

For local generation, use `abstractvision repl` with `/backend diffusers ...` or `/backend sdcpp ...`.

## Where do generated outputs go?

It depends on whether you configured a store:

- **CLI/REPL**: stores outputs in a local store by default (`LocalAssetStore`), under `~/.abstractvision/assets` unless `ABSTRACTVISION_STORE_DIR` is set ([`../src/abstractvision/artifacts.py`](../src/abstractvision/artifacts.py), [`../src/abstractvision/cli.py`](../src/abstractvision/cli.py)).
- **Python**:
  - if `VisionManager.store` is set, methods return an artifact ref dict (stored via `store.store_bytes(...)`)
  - otherwise they return a `GeneratedAsset` containing bytes ([`../src/abstractvision/types.py`](../src/abstractvision/types.py))

## What is an “artifact ref”?

An artifact ref is a small JSON dict that points to a stored blob. Minimal shape:

```json
{"$artifact":"<id>"}
```

Helpers: `is_artifact_ref()` / `make_media_ref()` in [`../src/abstractvision/artifacts.py`](../src/abstractvision/artifacts.py).

## How do I allow or block Diffusers downloads?

- REPL: cache-only is the default. Pre-download models separately, or set `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1` when you intentionally want runtime downloads ([`../src/abstractvision/cli.py`](../src/abstractvision/cli.py)).
- Python: `HuggingFaceDiffusersBackendConfig` defaults to `allow_download=False`; set `allow_download=True` only when you want runtime downloads ([`../src/abstractvision/backends/huggingface_diffusers.py`](../src/abstractvision/backends/huggingface_diffusers.py)).

## Why do I get “missing pipeline class” errors (e.g. `GlmImagePipeline`)?

Some newer pipelines may only exist on Diffusers GitHub `main`. Install:

- `pip install -U "abstractvision[diffusers-dev]"` (compatible dependency versions)
- `pip install -U "git+https://github.com/huggingface/diffusers@main"` (Diffusers `main`)

See: [docs/getting-started.md](getting-started.md).

## macOS (MPS): why do I get black images / dtype errors?

The Diffusers backend includes MPS-specific mitigations (e.g. VAE upcast and optional fp32 retry) in [`../src/abstractvision/backends/huggingface_diffusers.py`](../src/abstractvision/backends/huggingface_diffusers.py).

Common fixes:
- set `ABSTRACTVISION_DIFFUSERS_TORCH_DTYPE=float32` (more stable, higher memory)
- disable retry if memory is tight: `ABSTRACTVISION_DIFFUSERS_AUTO_RETRY_FP32=0`
- consider using the stable-diffusion.cpp backend for GGUF diffusion models ([docs/getting-started.md](getting-started.md))

## Windows/Linux (CUDA): why is `torch.cuda.is_available()` false?

On Windows/Linux, `pip install torch` (and packages that depend on `torch`) may install a CPU-only PyTorch build by default.

If you have an NVIDIA GPU and want CUDA acceleration:

1) Install a CUDA-enabled PyTorch wheel using the official selector: <https://pytorch.org/get-started/locally/>  
2) Verify:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available())"
```

## How do I pass advanced flags / parameters?

AbstractVision exposes an `extra` dict on requests ([`../src/abstractvision/types.py`](../src/abstractvision/types.py)), and the REPL forwards unknown `--flags` into `request.extra` ([`../src/abstractvision/cli.py`](../src/abstractvision/cli.py)).

Examples:
- Diffusers backend: accepts extra keys like `loras_json` and `rapid_aio_repo` (used by Qwen Image Edit flows; see [docs/getting-started.md](getting-started.md) and [`../src/abstractvision/backends/huggingface_diffusers.py`](../src/abstractvision/backends/huggingface_diffusers.py)).
- stable-diffusion.cpp backend:
  - CLI mode forwards flags to `sd-cli`
  - python-binding mode maps supported keys to binding kwargs and ignores unsupported keys ([`../src/abstractvision/backends/stable_diffusion_cpp.py`](../src/abstractvision/backends/stable_diffusion_cpp.py))

## What does the capability registry mean (and what does it not mean)?

The registry answers “what a model *claims* to support” (task keys/params) and can be used for **optional gating**:

- `VisionModelCapabilitiesRegistry.supports(...)` / `.require_support(...)` ([`../src/abstractvision/model_capabilities.py`](../src/abstractvision/model_capabilities.py))
- `VisionManager(model_id=...)` uses it to fail fast before calling a backend ([`../src/abstractvision/vision_manager.py`](../src/abstractvision/vision_manager.py))

It does **not** guarantee your configured backend can execute the task; backend support is a separate constraint ([docs/reference/backends.md](reference/backends.md)).

## I only need the HTTP backend. Do I have to install Torch/Diffusers?

No. The base install is lightweight and includes the stdlib OpenAI-compatible HTTP backend without Torch/Diffusers (see [`../pyproject.toml`](../pyproject.toml)). Heavy local backend modules are still imported lazily ([`../src/abstractvision/backends/__init__.py`](../src/abstractvision/backends/__init__.py)).

Install `abstractvision[diffusers]` only when you want local Diffusers generation. Use `abstractvision[sdcpp]` or an external `sd-cli` only when you need stable-diffusion.cpp.

## How do I integrate with AbstractCore?

Two options (details in [docs/reference/abstractcore-integration.md](reference/abstractcore-integration.md)):

- **Capability plugin**: [`../src/abstractvision/integrations/abstractcore_plugin.py`](../src/abstractvision/integrations/abstractcore_plugin.py) supports Diffusers, stable-diffusion.cpp, and OpenAI-compatible backends through env/config.
- **Tool helpers**: `make_vision_tools(...)` in [`../src/abstractvision/integrations/abstractcore.py`](../src/abstractvision/integrations/abstractcore.py) requires `VisionManager.store` for artifact-ref outputs.

AbstractCore is the host package; AbstractVision does not install it as a dependency.

## How do I run tests?

From the repo root:

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```
