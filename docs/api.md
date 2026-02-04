# API reference

This document describes the **public, stable** Python API surface of `abstractvision` and points to the implementation.

See also:
- Getting started (end-to-end examples): `docs/getting-started.md`
- Architecture (how the pieces fit): `docs/architecture.md`
- Backends reference (support matrix): `docs/reference/backends.md`
- FAQ (common questions): `docs/faq.md`

## Public exports

The package exports the following symbols from `abstractvision` (see `src/abstractvision/__init__.py`):

- `VisionManager`
- `VisionModelCapabilitiesRegistry`
- `LocalAssetStore`
- `RuntimeArtifactStoreAdapter`
- `is_artifact_ref`

## Core concepts

### Tasks

`VisionManager` exposes one method per task (implementation: `src/abstractvision/vision_manager.py`):

- `generate_image(...)` → `text_to_image`
- `edit_image(...)` → `image_to_image`
- `generate_video(...)` → `text_to_video` (backend-dependent)
- `image_to_video(...)` → `image_to_video` (backend-dependent)
- `generate_angles(...)` → `multi_view_image` (API exists; no built-in backend implements it yet)

Task names are also used by the capability registry (`src/abstractvision/assets/vision_model_capabilities.json`).

### Backends

Backends are execution engines that implement the `VisionBackend` interface (`src/abstractvision/backends/base_backend.py`).

Built-in backends live in `src/abstractvision/backends/`:
- `OpenAICompatibleVisionBackend` (HTTP)
- `HuggingFaceDiffusersVisionBackend` (local Diffusers)
- `StableDiffusionCppVisionBackend` (local stable-diffusion.cpp / GGUF)

### Outputs: bytes vs artifact refs

`VisionManager` returns:

- `GeneratedAsset` (bytes) when no store is configured (`src/abstractvision/types.py`)
- an artifact ref `dict` when `VisionManager.store` is configured (via `MediaStore.store_bytes(...)`)

Artifact helpers and stores are defined in `src/abstractvision/artifacts.py`.

## VisionManager (orchestrator)

`VisionManager` is intentionally thin: it validates/gates best-effort and delegates to the configured backend.

Signature (see `src/abstractvision/vision_manager.py`):
- `backend`: a `VisionBackend` implementation (required to run anything)
- `store`: optional `MediaStore` to enable artifact-ref outputs
- `model_id`: optional capability-gating model id (must exist in the registry)
- `registry`: optional `VisionModelCapabilitiesRegistry` instance (reused when gating is enabled)

### Minimal example (OpenAI-compatible backend + artifact refs)

```python
from abstractvision import LocalAssetStore, VisionManager, is_artifact_ref
from abstractvision.backends import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend

backend = OpenAICompatibleVisionBackend(
    config=OpenAICompatibleBackendConfig(base_url="http://localhost:1234/v1")
)
store = LocalAssetStore()
vm = VisionManager(backend=backend, store=store)

ref = vm.generate_image("a studio photo of an espresso machine", width=768, height=768, steps=20)
assert is_artifact_ref(ref)
png_bytes = store.load_bytes(ref["$artifact"])
```

### Local example (Diffusers backend)

```python
from abstractvision import VisionManager
from abstractvision.backends import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

backend = HuggingFaceDiffusersVisionBackend(
    config=HuggingFaceDiffusersBackendConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        device="auto",
        allow_download=True,
    )
)
vm = VisionManager(backend=backend)
asset = vm.generate_image("a watercolor painting of a lighthouse", width=512, height=512, steps=10)
```

Note: for cache-only/offline mode, set `allow_download=False`.

## Passing advanced backend parameters (`extra`)

Request dataclasses include an `extra: dict` field (`src/abstractvision/types.py`). Use it to pass backend-specific parameters in a controlled way:

```python
asset_or_ref = vm.generate_image(
    "a product photo of a matte black espresso machine",
    steps=8,
    guidance_scale=1.0,
    extra={
        # Example keys used by some Diffusers flows:
        "loras_json": [{"source": "lightx2v/Qwen-Image-Edit-2511-Lightning", "scale": 1.0}],
        "rapid_aio_repo": "linoyts/Qwen-Image-Edit-Rapid-AIO",
    },
)
```

Backends may ignore unknown keys; consult the backend implementation and `docs/reference/backends.md`.

## Capability registry (what models can do)

The packaged registry is loaded by `VisionModelCapabilitiesRegistry` (`src/abstractvision/model_capabilities.py`).

```python
from abstractvision import VisionModelCapabilitiesRegistry

reg = VisionModelCapabilitiesRegistry()
print(reg.list_tasks())
print(reg.models_for_task("text_to_image"))

reg.require_support("Qwen/Qwen-Image-2512", "text_to_image")
```

Optional gating:
- If you construct `VisionManager(model_id=..., registry=...)`, the manager will fail fast on unsupported tasks before calling a backend (`src/abstractvision/vision_manager.py`).

Important: the registry is *not* a guarantee that your configured backend can execute a task at runtime.
Use `docs/reference/backends.md` for backend support.

## Artifacts and stores

Artifact helpers and store implementations live in `src/abstractvision/artifacts.py`:

- `LocalAssetStore` (standalone local files, default `~/.abstractvision/assets`)
- `RuntimeArtifactStoreAdapter` (duck-typed adapter for an external artifact store)
- `is_artifact_ref(...)` / `make_media_ref(...)`

See: `docs/reference/artifacts.md`.

## Errors you may want to handle

Common exceptions (defined in `src/abstractvision/errors.py`):

- `BackendNotConfiguredError` (calling `VisionManager` without a backend)
- `CapabilityNotSupportedError` (task isn’t supported by the model registry or backend)
- `UnknownModelError` (model id isn’t present in the registry)
- `OptionalDependencyMissingError` (backend dependency is missing, e.g. Diffusers/Torch)
