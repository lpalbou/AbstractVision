# AbstractVision architecture

AbstractVision is a model-agnostic Python layer that standardizes **generative vision outputs** behind a small API:
text→image, image→image (and optionally video when a backend supports it).

This document describes the *current code in this repo* and links to the supporting reference docs.

See also:
- Docs index: `docs/README.md`
- Getting started: `docs/getting-started.md`
- API reference: `docs/api.md`
- FAQ: `docs/faq.md`
- Backends: `docs/reference/backends.md`
- Capability registry: `docs/reference/capabilities-registry.md`
- Artifacts: `docs/reference/artifacts.md`
- AbstractCore integration: `docs/reference/abstractcore-integration.md`

## Scope (and non-goals)

AbstractVision focuses on **producing** images/videos.

It is not the owner of “LLM image/video input attachments” (multimodal inputs to LLMs); those concerns live in higher-level layers (e.g., AbstractCore).

## Key components (with evidence pointers)

- **Orchestrator**: `VisionManager` (`src/abstractvision/vision_manager.py`)
  - Delegates execution to a backend.
  - Optionally gates requests using the capability registry when `model_id` is set.
  - Optionally stores outputs and returns artifact refs when `store` is set.
- **Backend contract**: `VisionBackend` (`src/abstractvision/backends/base_backend.py`)
  - Implementations live in `src/abstractvision/backends/`.
- **Capability registry**: `VisionModelCapabilitiesRegistry` (`src/abstractvision/model_capabilities.py`)
  - Loads packaged data: `src/abstractvision/assets/vision_model_capabilities.json`.
- **Artifact outputs**: `MediaStore`, `LocalAssetStore`, `RuntimeArtifactStoreAdapter` (`src/abstractvision/artifacts.py`)
  - Artifact ref shape helper: `is_artifact_ref()` (`src/abstractvision/artifacts.py`).
- **CLI/REPL**: `abstractvision` entrypoint (`src/abstractvision/cli.py`)
  - Lets you inspect the registry and manually test generation backends.
- **AbstractCore integration**:
  - Capability plugin: `src/abstractvision/integrations/abstractcore_plugin.py` (registered in `pyproject.toml`)
  - Tool helpers: `src/abstractvision/integrations/abstractcore.py`

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
- `VisionManager` creates request dataclasses like `ImageGenerationRequest` / `ImageEditRequest` (`src/abstractvision/types.py`).
- When `store` is set, `VisionManager._maybe_store()` calls `store.store_bytes(...)` and returns an artifact ref dict (`src/abstractvision/vision_manager.py`, `src/abstractvision/artifacts.py`).

## Capability gating (model-level) vs runtime gating (backend-level)

AbstractVision separates two kinds of “can I do this?” checks:

1) **Model-level gating** (optional): “Does model X support task Y?”
   - Implemented by `VisionModelCapabilitiesRegistry.require_support(...)` (`src/abstractvision/model_capabilities.py`)
   - Used by `VisionManager._require_model_support(...)` when `VisionManager.model_id` is set (`src/abstractvision/vision_manager.py`)

2) **Backend-level gating** (best-effort): “Does this configured backend support task Y / mask edits?”
   - Backends may implement `get_capabilities()` returning `VisionBackendCapabilities` (`src/abstractvision/types.py`)
   - Enforced by `VisionManager._require_backend_support(...)` and mask checks in `VisionManager.edit_image(...)` (`src/abstractvision/vision_manager.py`)

## Backend reality (what runs today)

The public API includes `text_to_video`, `image_to_video`, and `multi_view_image`, but backend support is currently limited:

- Built-in backends implement **images** (`text_to_image`, `image_to_image`):
  - OpenAI-compatible HTTP backend (`src/abstractvision/backends/openai_compatible.py`)
  - Diffusers backend (`src/abstractvision/backends/huggingface_diffusers.py`)
  - stable-diffusion.cpp backend (`src/abstractvision/backends/stable_diffusion_cpp.py`)
- Video is supported **only** by the OpenAI-compatible backend, and only when `text_to_video_path` / `image_to_video_path` are configured (`src/abstractvision/backends/openai_compatible.py`).
- No built-in backend implements `multi_view_image` yet (they raise `CapabilityNotSupportedError` in `generate_angles(...)`).

For a detailed support matrix and configuration options, see `docs/reference/backends.md`.

## AbstractCore plugin flow (framework integration)

AbstractVision can be discovered by AbstractCore via an entry point:
`[project.entry-points."abstractcore.capabilities_plugins"]` in `pyproject.toml`.

```mermaid
flowchart LR
  AC[AbstractCore] -->|loads entry point| Plugin[AbstractVision plugin<br/>register(...)]
  Plugin --> Cap[VisionCapability<br/>(t2i/i2i/t2v/i2v)]
  Cap --> VM[VisionManager]
  VM --> BE[OpenAICompatibleVisionBackend]
  BE --> HTTP[OpenAI-shaped HTTP<br/>/images/generations, /images/edits]
```

Current plugin behavior (evidence in `src/abstractvision/integrations/abstractcore_plugin.py`):
- Only the OpenAI-compatible backend is supported via the plugin (v0).
- Configuration is read from `owner.config` keys like `vision_base_url` and falls back to `ABSTRACTVISION_*` env vars.

## Extending AbstractVision (practical steps)

- Add a new backend:
  1) Implement `VisionBackend` (`src/abstractvision/backends/base_backend.py`)
  2) Add capability reporting via `get_capabilities()` when you can (optional)
  3) Add tests under `tests/`
- Update the registry:
  1) Edit `src/abstractvision/assets/vision_model_capabilities.json`
  2) Validate by running the test suite (validator is wired into the registry loader)
  3) Use `abstractvision show-model <id>` to sanity-check task/param printing (`src/abstractvision/cli.py`)
