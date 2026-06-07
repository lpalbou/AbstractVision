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
- `upscale_image(...)` → `image_upscale`
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

`upscale_image(...)`, `generate_video(...)`, and `image_to_video(...)` are part
of the public API. MLX-Gen 0.18.13+ supports SeedVR2 `image_upscale`, Wan
`text_to_video`, and first-frame `image_to_video`, including A14B task-specific
checkpoints. Local Diffusers video remains experimental and disabled from the
normal local surfaces. Generated MP4 outputs still require an `ffmpeg`
executable on `PATH` whenever a backend returns frame sequences for local
packaging.

### Local example (MLX-Gen backend)

Install `abstractvision[mlx-gen]` and pre-download the exact model repo first,
for example `abstractvision download AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit --provider mlx-gen`.

```python
from pathlib import Path

from abstractvision import VisionManager
from abstractvision.backends import MLXGenBackendConfig, MLXGenVisionBackend

def on_progress(event):
    if event.total_frames:
        print(f"{event.phase}: frame {event.frame}/{event.total_frames}")
    else:
        print(f"{event.phase}: step {event.step}/{event.total_steps}")

image_backend = MLXGenVisionBackend(
    config=MLXGenBackendConfig(model="AbstractFramework/flux.2-klein-9b-8bit")
)
image_vm = VisionManager(backend=image_backend)

image_asset = image_vm.generate_image(
    "a studio product photo of a red toy race car",
    width=768,
    height=512,
    steps=12,
    guidance_scale=1.0,
    on_progress=on_progress,
)

edit_asset = image_vm.edit_image(
    "compose the subject using the second image as a style and layout reference",
    image=Path("./subject.png").read_bytes(),
    steps=12,
    guidance_scale=1.0,
    on_progress=on_progress,
    extra={"reference_images": [Path("./style-reference.png").read_bytes()]},
)

upscale_backend = MLXGenVisionBackend(config=MLXGenBackendConfig(model="AbstractFramework/seedvr2-3b-8bit"))
upscale_vm = VisionManager(backend=upscale_backend)

upscaled_asset = upscale_vm.upscale_image(
    image=Path("./subject.png").read_bytes(),
    scale="2x",
    seed=2405,
    on_progress=on_progress,
)

t2v_backend = MLXGenVisionBackend(
    config=MLXGenBackendConfig(model="AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit")
)
t2v_vm = VisionManager(backend=t2v_backend)

asset = t2v_vm.generate_video(
    "a red fox walking through a snowy forest, cinematic",
    width=432,
    height=240,
    num_frames=41,
    fps=10,
    steps=20,
    guidance_scale=4.0,
    guidance_2=3.0,
    on_progress=on_progress,
    extra={"max_sequence_length": 256},
)

i2v_backend = MLXGenVisionBackend(
    config=MLXGenBackendConfig(model="AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit")
)
i2v_vm = VisionManager(backend=i2v_backend)

first_frame_asset = i2v_vm.image_to_video(
    image=Path("./first-frame.png").read_bytes(),
    prompt="slow camera push-in",
    width=432,
    height=240,
    num_frames=41,
    fps=10,
    steps=20,
    guidance_scale=3.5,
    guidance_2=3.5,
    on_progress=on_progress,
    extra={"max_sequence_length": 256},
)
```

Wan 2.2 A14B has two guidance controls. Use `guidance_scale` for the
primary/high-noise stage and `guidance_2` for the second/low-noise stage. The
registry default is `guidance_2=3.0` for text-to-video A14B and `3.5` for
image-to-video A14B. Other video models should omit `guidance_2` unless their
registry task declares it.

For MLX-Gen, `on_progress` receives an `abstractvision.VideoProgressEvent`.
Image generation/editing/upscaling events carry `phase`, `step`, `total_steps`,
and denoise-step `progress`. Wan video events add `frame`, `total_frames`, and
`frame_progress`. The lower-level `backend.generate_image_with_progress(...)`,
`backend.edit_image_with_progress(...)`, `backend.upscale_image_with_progress(...)`,
`backend.generate_video_with_progress(...)`, and
`backend.image_to_video_with_progress(...)` methods keep the existing
two-argument `(current, total)` callback for backend-agnostic progress bars.
For MLX-Gen, that callback reports denoise step counts; use
`on_progress(event)` when a UI also needs video frame context.

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
