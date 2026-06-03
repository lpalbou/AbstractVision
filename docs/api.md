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
- `HuggingFaceDiffusersVisionBackend` (local Diffusers images; local Diffusers `text_to_video` groundwork is currently quarantined)
- `StableDiffusionCppVisionBackend` (local stable-diffusion.cpp / GGUF)
- `MLXGenVisionBackend` / compatibility alias `MFluxVisionBackend` (local Apple Silicon MLX-Gen bridge for curated AbstractFramework q4/q8 MLX presets, official FIBO snapshots, and Wan video)

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

Image sizes are backend/model-specific. `width` and `height` arguments are
optional request overrides; omitting them lets the backend use its default or
`auto` behavior. Passing an unsupported size is expected to fail at the selected
provider/backend boundary rather than being silently rewritten by
AbstractVision.

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

`generate_video(...)` and `image_to_video(...)` are part of the public API. Local Diffusers video remains experimental and disabled from the normal local surfaces, while MLX-Gen 0.18.8+ supports Wan `text_to_video` and first-frame `image_to_video`, including A14B task-specific checkpoints. Generated MP4 outputs still require an `ffmpeg` executable on `PATH` whenever a backend returns frame sequences for local packaging.

### Local example (MLX-Gen backend)

Install `abstractvision[mlx-gen]` and pre-download the exact model repo first, for example `abstractvision download Wan-AI/Wan2.2-TI2V-5B-Diffusers --provider mlx-gen`.

```python
from pathlib import Path

from abstractvision import VisionManager
from abstractvision.backends import MLXGenBackendConfig, MLXGenVisionBackend

backend = MLXGenVisionBackend(
    config=MLXGenBackendConfig(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
)
vm = VisionManager(backend=backend)

def on_progress(event):
    print(f"{event.phase}: frame {event.frame}/{event.total_frames}")

asset = vm.generate_video(
    "a red fox walking through a snowy forest, cinematic",
    num_frames=121,
    fps=24,
    steps=50,
    guidance_scale=5.0,
    on_progress=on_progress,
    extra={"max_sequence_length": 256},
)

first_frame_asset = vm.image_to_video(
    image=Path("./first-frame.png").read_bytes(),
    prompt="slow camera push-in",
    num_frames=121,
    fps=24,
    steps=50,
    guidance_scale=5.0,
    on_progress=on_progress,
    extra={"max_sequence_length": 256},
)
```

For MLX-Gen Wan, `on_progress` receives an
`abstractvision.VideoProgressEvent` with `phase`, `frame`, `total_frames`,
`step`, `total_steps`, and normalized `progress` fields. The lower-level
`backend.generate_video_with_progress(...)` and
`backend.image_to_video_with_progress(...)` methods keep the existing
two-argument `(current, total)` callback for backend-agnostic progress bars.

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
