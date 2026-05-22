# Troubleshooting

See also:
- Getting started: [docs/getting-started.md](getting-started.md)
- Backends: [docs/reference/backends.md](reference/backends.md)
- Configuration: [docs/reference/configuration.md](reference/configuration.md)
- FAQ: [docs/faq.md](faq.md)

This page covers the current user-facing failure modes that are most likely when
using local backends, the playground, and the AbstractCore integration.

## Local backend says a dependency is missing

### Symptom

- `OptionalDependencyMissingError`
- import errors mentioning `diffusers`, `torch`, `stable_diffusion_cpp`, or `mflux`

### Likely cause

The base install is intentionally lightweight. Local runtimes are not installed
unless you choose the matching extra.

### Fix

- Diffusers: `pip install "abstractvision[diffusers]"`
- stable-diffusion.cpp bindings: `pip install "abstractvision[sdcpp]"`
- MFLUX on Apple Silicon: `pip install "abstractvision[mflux]"`

### Verify

- `abstractvision cli`
- `abstractvision catalog --provider diffusers`

## Local Diffusers cannot find the model

### Symptom

- local Diffusers generation fails before inference starts
- the error mentions a missing snapshot or cache-only/offline behavior

### Likely cause

The Diffusers backend is cache-only by default and the required model is not yet
present in the Hugging Face cache.

### Fix

- Pre-download the model with `abstractvision download ... --provider diffusers`
- or allow runtime downloads explicitly with `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1`

Examples:

```bash
abstractvision download stable-diffusion --provider diffusers
abstractvision download qwen-image-edit-2511 --provider diffusers
```

### Verify

- `abstractvision catalog --provider diffusers`
- `abstractvision show-model Qwen/Qwen-Image-Edit-2511`

## Local `text_to_video` is experimental and currently disabled

### Symptom

- the local playground has no active local `Text→Video` model choices
- local Diffusers `t2v` raises a capability/disabled error

### Likely cause

AbstractVision intentionally quarantines the current local `text_to_video`
groundwork because the operator validation bar is not met yet.

### Fix

- use the OpenAI-compatible backend if you need `text_to_video` today; or
- follow the backlog item that tracks the local re-validation work:
  [`docs/backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md`](backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md)

## `GLM-Image` is temporarily disabled in the local Diffusers backend

### Symptom

- `zai-org/GLM-Image` does not appear in local runtime-backed model selectors
- direct local Diffusers calls reject it as temporarily disabled

### Likely cause

Operator testing and runtime investigation showed that current local GLM output
quality/runtime behavior is not honest enough to ship as a working local
capability.

### Fix

Use another local Diffusers image model for now, for example:
- `runwayml/stable-diffusion-v1-5`
- `Qwen/Qwen-Image-Edit-2511`
- `black-forest-labs/FLUX.2-klein-4B`

The follow-up investigation is tracked in:
- [`docs/backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md`](backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md)

## MFLUX `image_to_image` is temporarily disabled

### Symptom

- MFLUX models do not appear in the playground `Image→Image` tab
- local MFLUX `i2i` calls raise a temporary capability error

### Likely cause

Operator tests showed that the current MFLUX `image_to_image` path does not yet
preserve scene structure reliably enough.

### Fix

- use MFLUX for `text_to_image` only for now; and
- use local Diffusers or `stable-diffusion.cpp` for `image_to_image`.

## `mps` was requested but is unavailable

### Symptom

- local Diffusers startup fails with an error mentioning `mps`

### Likely cause

PyTorch does not report Apple Metal / MPS as available in the current
environment.

### Checks

```bash
python - <<'PY'
import torch
print(torch.backends.mps.is_available())
PY
```

### Fix

- use a PyTorch build with MPS support on Apple Silicon;
- or switch the backend device to `cpu`.

### Verify

- rerun the check above;
- then retry with `--diffusers-device mps` or `--diffusers-device cpu`

## Playground panel is disabled

### Symptom

- Image → Image or Text → Video controls stay disabled in the playground

### Likely cause

The currently selected model does not advertise the required task in the
packaged capability registry, or the backend cannot really execute that task.

### Fix

Choose a model that advertises the task:

- image edits: a model with `image_to_image`
- local text-to-video: none are currently shipped as enabled local options in the bundled server

### Verify

- open `GET /v1/vision/models`
- inspect the selected model’s `tasks` and `task_specs`

## AbstractCore tool expected an artifact ref

### Symptom

- `vision_text_to_image`, `vision_image_to_image`, or `vision_text_to_video`
  reports that an artifact ref was expected

### Likely cause

`make_vision_tools(...)` expects `VisionManager.store` to be set so outputs can
be returned as artifact references instead of raw bytes.

### Fix

Create the manager with a store, for example `LocalAssetStore()` or a runtime
adapter.

### Verify

- rerun the tool call and confirm the result contains `"$artifact"`
