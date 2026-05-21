# API reference

This document describes the **public** Python API surface of `abstractvision` (0.x / Alpha) and points to the implementation.

See also:
- Getting started (end-to-end examples): [docs/getting-started.md](getting-started.md)
- Architecture (how the pieces fit): [docs/architecture.md](architecture.md)
- Backends reference (support matrix): [docs/reference/backends.md](reference/backends.md)
- FAQ (common questions): [docs/faq.md](faq.md)

## Public exports

The package exports the following symbols from `abstractvision` (see [`../src/abstractvision/__init__.py`](../src/abstractvision/__init__.py)):

- `VisionManager`
- `ProviderModelInfo`
- `VisionModelCapabilitiesRegistry`
- `LocalAssetStore`
- `RuntimeArtifactStoreAdapter`
- `is_artifact_ref`
- `__version__`

## Core concepts

### Tasks

`VisionManager` exposes one method per task (implementation: [`../src/abstractvision/vision_manager.py`](../src/abstractvision/vision_manager.py)):

- `generate_image(...)` → `text_to_image`
- `edit_image(...)` → `image_to_image`
- `generate_video(...)` → `text_to_video` (backend-dependent)
- `image_to_video(...)` → `image_to_video` (backend-dependent)
- `generate_angles(...)` → `multi_view_image` (API exists; no built-in backend implements it yet)

Task names are also used by the capability registry ([`../src/abstractvision/assets/vision_model_capabilities.json`](../src/abstractvision/assets/vision_model_capabilities.json)).

### Backends

Backends are execution engines that implement the `VisionBackend` interface ([`../src/abstractvision/backends/base_backend.py`](../src/abstractvision/backends/base_backend.py)).

Built-in backends live in [`../src/abstractvision/backends/`](../src/abstractvision/backends/):
- `OpenAICompatibleVisionBackend` (HTTP)
- `HuggingFaceDiffusersVisionBackend` (local Diffusers images + CogVideoX text-to-video)
- `StableDiffusionCppVisionBackend` (local stable-diffusion.cpp / GGUF)
- `MFluxVisionBackend` (local Apple Silicon MFLUX bridge for curated MLX presets)

Backend config classes are re-exported from `abstractvision.backends` via lazy imports (see [`../src/abstractvision/backends/__init__.py`](../src/abstractvision/backends/__init__.py)).

Provider catalog listing is exposed as a backend contract:

```python
from abstractvision.backends import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend

backend = OpenAICompatibleVisionBackend(
    config=OpenAICompatibleBackendConfig(base_url="http://localhost:1234/v1")
)
for model in backend.list_provider_models(task="text_to_image"):
    print(model.id)
```

For official OpenAI, use `base_url="https://api.openai.com/v1"` and an API key. Catalog listing is explicit and does not change the configured generation model.

When AbstractVision is loaded as an AbstractCore capability plugin, the plugin shim exposes the
same explicit catalog surface as `llm.vision.list_provider_models(task="text_to_image")`. It
returns JSON-safe dictionaries so Core/Gateway route code can avoid private backend reach-throughs.

### Outputs: bytes vs artifact refs

`VisionManager` returns:

- `GeneratedAsset` (bytes) when no store is configured ([`../src/abstractvision/types.py`](../src/abstractvision/types.py))
- an artifact ref `dict` when `VisionManager.store` is configured (via `MediaStore.store_bytes(...)`)

Artifact helpers and stores are defined in [`../src/abstractvision/artifacts.py`](../src/abstractvision/artifacts.py).

## VisionManager (orchestrator)

`VisionManager` is intentionally thin: it validates/gates best-effort and delegates to the configured backend.

Signature (see [`../src/abstractvision/vision_manager.py`](../src/abstractvision/vision_manager.py)):
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

Install `abstractvision[diffusers]` before using this backend.

```python
from abstractvision import VisionManager
from abstractvision.backends import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

backend = HuggingFaceDiffusersVisionBackend(
    config=HuggingFaceDiffusersBackendConfig(
        model_id="runwayml/stable-diffusion-v1-5",
        device="auto",
        allow_download=False,
    )
)
vm = VisionManager(backend=backend)
asset = vm.generate_image("a watercolor painting of a lighthouse", width=512, height=512, steps=10)
```

Note: `allow_download=False` is the default. Pre-download model weights separately, or set `allow_download=True` only when you want runtime downloads.

For local text→video, switch the Diffusers model id to `zai-org/CogVideoX-2b` (or `THUDM/CogVideoX-2b`) and call `generate_video(...)`. Generated MP4 outputs require an `ffmpeg` executable on `PATH`.

## Passing advanced backend parameters (`extra`)

Request dataclasses include an `extra: dict` field ([`../src/abstractvision/types.py`](../src/abstractvision/types.py)). Use it to pass backend-specific parameters in a controlled way:

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

Backends may ignore unknown keys; consult the backend implementation and [docs/reference/backends.md](reference/backends.md).

## Capability registry (what models can do)

The packaged registry is loaded by `VisionModelCapabilitiesRegistry` ([`../src/abstractvision/model_capabilities.py`](../src/abstractvision/model_capabilities.py)).

```python
from abstractvision import VisionModelCapabilitiesRegistry

reg = VisionModelCapabilitiesRegistry()
print(reg.list_tasks())
print(reg.models_for_task("text_to_image"))

reg.require_support("runwayml/stable-diffusion-v1-5", "text_to_image")
```

Optional gating:
- If you construct `VisionManager(model_id=..., registry=...)`, the manager will fail fast on unsupported tasks before calling a backend ([`../src/abstractvision/vision_manager.py`](../src/abstractvision/vision_manager.py)).

Important: the registry is *not* a guarantee that your configured backend can execute a task at runtime.
Use [docs/reference/backends.md](reference/backends.md) for backend support.

## Artifacts and stores

Artifact helpers and store implementations live in [`../src/abstractvision/artifacts.py`](../src/abstractvision/artifacts.py):

- `LocalAssetStore` (standalone local files, default `~/.abstractvision/assets`)
- `RuntimeArtifactStoreAdapter` (duck-typed adapter for an external artifact store)
- `is_artifact_ref(...)` / `make_media_ref(...)`

See: [docs/reference/artifacts.md](reference/artifacts.md).

## Errors you may want to handle

Common exceptions (defined in [`../src/abstractvision/errors.py`](../src/abstractvision/errors.py)):

- `BackendNotConfiguredError` (calling `VisionManager` without a backend)
- `CapabilityNotSupportedError` (task isn’t supported by the model registry or backend)
- `UnknownModelError` (model id isn’t present in the registry)
- `OptionalDependencyMissingError` (backend dependency is missing, e.g. Diffusers/Torch)
