# AbstractVision architecture

AbstractVision is a model-agnostic Python layer that standardizes **generative vision outputs** behind a small API:
text→image, image→image (and optionally video when a backend supports it).

This document describes the *current code in this repo* and links to the supporting reference docs.

See also:
- Docs index: [docs/README.md](README.md)
- Getting started: [docs/getting-started.md](getting-started.md)
- API reference: [docs/api.md](api.md)
- FAQ: [docs/faq.md](faq.md)
- ADR index: [docs/adr/README.md](adr/README.md)
- Backends: [docs/reference/backends.md](reference/backends.md)
- Capability registry: [docs/reference/capabilities-registry.md](reference/capabilities-registry.md)
- Artifacts: [docs/reference/artifacts.md](reference/artifacts.md)
- AbstractCore integration: [docs/reference/abstractcore-integration.md](reference/abstractcore-integration.md)

## AbstractFramework ecosystem (positioning)

AbstractVision is one component in the **AbstractFramework** ecosystem:

- **AbstractFramework** (project hub): <https://github.com/lpalbou/AbstractFramework>
- **AbstractCore** (orchestration + tool calling): <https://github.com/lpalbou/abstractcore>
- **AbstractRuntime** (runtime services, including artifact storage): <https://github.com/lpalbou/abstractruntime>

Where AbstractVision fits:
- AbstractVision focuses on *producing* images/videos (generators).
- AbstractCore focuses on orchestration, tool calling, and higher-level workflows (it can discover AbstractVision via the plugin entry point in `pyproject.toml` and `src/abstractvision/integrations/abstractcore_plugin.py`).
- AbstractRuntime provides runtime services and an artifact store interface; `RuntimeArtifactStoreAdapter` bridges AbstractVision to an AbstractRuntime-style artifact store (`src/abstractvision/artifacts.py`).

## Scope (and non-goals)

AbstractVision focuses on **producing** images/videos.

It is not the owner of “LLM image/video input attachments” (multimodal inputs to LLMs); those concerns live in higher-level layers (e.g., AbstractCore).

## Key components (with evidence pointers)

- **Orchestrator**: [`VisionManager`](../src/abstractvision/vision_manager.py)
  - Delegates execution to a backend.
  - Optionally gates requests using the capability registry when `model_id` is set.
  - Optionally stores outputs and returns artifact refs when `store` is set.
- **Backend contract**: [`VisionBackend`](../src/abstractvision/backends/base_backend.py)
  - Implementations live in [`../src/abstractvision/backends/`](../src/abstractvision/backends/).
- **Capability registry**: [`VisionModelCapabilitiesRegistry`](../src/abstractvision/model_capabilities.py)
  - Loads packaged data: [`vision_model_capabilities.json`](../src/abstractvision/assets/vision_model_capabilities.json).
  - Governs the model catalog, task metadata, and curated download surfaces as a product surface,
    not just as incidental documentation metadata. See [ADR 0005](adr/0005_curated_capability_registry_and_download_catalog.md).
- **Artifact outputs**: [`MediaStore`](../src/abstractvision/artifacts.py), [`LocalAssetStore`](../src/abstractvision/artifacts.py), [`RuntimeArtifactStoreAdapter`](../src/abstractvision/artifacts.py)
  - Artifact ref helper: `is_artifact_ref()` (see [`../src/abstractvision/artifacts.py`](../src/abstractvision/artifacts.py)).
- **CLI/REPL**: `abstractvision` entrypoint ([`../src/abstractvision/cli.py`](../src/abstractvision/cli.py))
  - Lets you inspect the registry and manually test generation backends.
- **AbstractCore integration**:
  - Capability plugin: [`../src/abstractvision/integrations/abstractcore_plugin.py`](../src/abstractvision/integrations/abstractcore_plugin.py) (registered in `pyproject.toml`)
  - Tool helpers: [`../src/abstractvision/integrations/abstractcore.py`](../src/abstractvision/integrations/abstractcore.py)

## High-level flow (library mode)

```mermaid
flowchart LR
  Caller[Caller<br/>(Python / CLI)] --> VM[VisionManager]
  VM -->|request dataclass| BE[VisionBackend]
  BE -->|GeneratedAsset| VM
  VM -->|store set| Store[MediaStore<br/>(LocalAssetStore / Runtime adapter)]
  Store --> Ref[Artifact ref dict]
  VM -->|store not set| Asset[GeneratedAsset<br/>(bytes + mime)]
```

Notes (anchored in code):
- `VisionManager` creates request dataclasses like `ImageGenerationRequest` / `ImageEditRequest` ([`../src/abstractvision/types.py`](../src/abstractvision/types.py)).
- When `store` is set, `VisionManager._maybe_store()` calls `store.store_bytes(...)` and returns an artifact ref dict ([`../src/abstractvision/vision_manager.py`](../src/abstractvision/vision_manager.py), [`../src/abstractvision/artifacts.py`](../src/abstractvision/artifacts.py)).

## Capability gating (model-level) vs runtime gating (backend-level)

AbstractVision separates two kinds of “can I do this?” checks:

1) **Model-level gating** (optional): “Does model X support task Y?”
   - Implemented by `VisionModelCapabilitiesRegistry.require_support(...)` ([`../src/abstractvision/model_capabilities.py`](../src/abstractvision/model_capabilities.py))
   - Used by `VisionManager._require_model_support(...)` when `VisionManager.model_id` is set ([`../src/abstractvision/vision_manager.py`](../src/abstractvision/vision_manager.py))

2) **Backend-level gating** (best-effort): “Does this configured backend support task Y / mask edits?”
   - Backends may implement `get_capabilities()` returning `VisionBackendCapabilities` ([`../src/abstractvision/types.py`](../src/abstractvision/types.py))
   - Enforced by `VisionManager._require_backend_support(...)` and mask checks in `VisionManager.edit_image(...)` ([`../src/abstractvision/vision_manager.py`](../src/abstractvision/vision_manager.py))

## Backend reality (what runs today)

The public API includes `text_to_video`, `image_to_video`, and `multi_view_image`, but backend support is currently limited:

- Built-in backends implement **images** (`text_to_image`, `image_to_image`):
  - OpenAI-compatible HTTP backend ([`../src/abstractvision/backends/openai_compatible.py`](../src/abstractvision/backends/openai_compatible.py))
  - Diffusers backend ([`../src/abstractvision/backends/huggingface_diffusers.py`](../src/abstractvision/backends/huggingface_diffusers.py))
  - stable-diffusion.cpp backend ([`../src/abstractvision/backends/stable_diffusion_cpp.py`](../src/abstractvision/backends/stable_diffusion_cpp.py))
  - MFLUX backend for curated Apple Silicon MLX presets ([`../src/abstractvision/backends/mflux.py`](../src/abstractvision/backends/mflux.py))
- Local MFLUX supports `text_to_image` and FLUX.2 klein `image_to_image` edits (no masks yet).
- Local Diffusers `text_to_video` remains experimental and is temporarily disabled from the normal local surfaces.
- `image_to_video` is still supported **only** by the OpenAI-compatible backend, and only when `text_to_video_path` / `image_to_video_path` are configured ([`../src/abstractvision/backends/openai_compatible.py`](../src/abstractvision/backends/openai_compatible.py)).
- No built-in backend implements `multi_view_image` yet (they raise `CapabilityNotSupportedError` in `generate_angles(...)`).

For a detailed support matrix and configuration options, see [docs/reference/backends.md](reference/backends.md).

## Catalog and compatibility policy

AbstractVision treats model discovery, curated downloads, and runtime compatibility as part of the
product surface:

- the registry is global and platform-neutral;
- host-specific recommendation happens later in catalog or preset selection;
- one parent model family may expose multiple engine-specific variants;
- official upstream repos are preferred, with curated community ports used only when they provide
  the best runtime-native artifact for a target engine.

That policy is defined in [ADR 0005](adr/0005_curated_capability_registry_and_download_catalog.md).

## AbstractCore plugin flow (framework integration)

AbstractVision can be discovered by AbstractCore via an entry point:
`[project.entry-points."abstractcore.capabilities_plugins"]` in [`../pyproject.toml`](../pyproject.toml).

```mermaid
flowchart LR
  AC[AbstractCore] -->|loads entry point| Plugin[AbstractVision plugin<br/>register(...)]
  Plugin --> Cap[VisionCapability<br/>(t2i/i2i/t2v/i2v)]
  Cap --> VM[VisionManager]
  VM --> BE{Configured backend}
  BE --> HTTP[OpenAI-compatible HTTP<br/>OpenAI or local /v1 server]
  BE --> HF[Local Diffusers]
  BE --> SDCPP[Local stable-diffusion.cpp]
```

Current plugin behavior (evidence in [`../src/abstractvision/integrations/abstractcore_plugin.py`](../src/abstractvision/integrations/abstractcore_plugin.py)):
- Default: OpenAI HTTP with backend id `abstractvision:openai`; the legacy backend id `abstractvision:openai-compatible` remains registered and preserves compatible-endpoint defaults when selected directly.
- Compatible endpoints should set `OPENAI_BASE_URL`; set `ABSTRACTVISION_BACKEND=openai-compatible` when you want to force compatible-endpoint semantics.
- Local Diffusers and stable-diffusion.cpp are supported when `vision_backend` / `ABSTRACTVISION_BACKEND` selects `diffusers` or `sdcpp`.
- Configuration is read from `owner.config` keys like `vision_base_url`, `vision_model_id`, `vision_backend`, and backend-specific keys, then falls back to `ABSTRACTVISION_*` and standard OpenAI env vars where relevant.

## Extending AbstractVision (practical steps)

- Add a new backend:
  1) Implement `VisionBackend` ([`../src/abstractvision/backends/base_backend.py`](../src/abstractvision/backends/base_backend.py))
  2) Add capability reporting via `get_capabilities()` when you can (optional)
  3) Add tests under [`../tests/`](../tests/)
- Update the registry:
  1) Edit [`../src/abstractvision/assets/vision_model_capabilities.json`](../src/abstractvision/assets/vision_model_capabilities.json)
  2) Validate by running the test suite (validator is wired into the registry loader)
  3) Use `abstractvision show-model <id>` to sanity-check task/param printing ([`../src/abstractvision/cli.py`](../src/abstractvision/cli.py))
