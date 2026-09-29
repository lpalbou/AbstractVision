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
- import errors mentioning `diffusers`, `torch`, `torchvision`, `stable_diffusion_cpp`, `mlxgen`, or `mflux`
- errors like `Qwen2VLVideoProcessor requires the Torchvision library...` when using `Qwen/Qwen-Image-Edit-*`

### Likely cause

The base install is intentionally lightweight. Local runtimes are not installed
unless you choose the matching extra.

### Fix

- Diffusers: `pip install "abstractvision[diffusers]"`
- If the error mentions missing `torchvision`: `pip install torchvision` (or upgrade/reinstall `abstractvision[diffusers]`)
- stable-diffusion.cpp bindings: `pip install "abstractvision[sdcpp]"`
- MLX-Gen: `pip install "abstractvision[mlx-gen]"` (or compatibility alias `abstractvision[mflux]`). MLX-Gen is validated on Apple Silicon. The extra also installs on Linux, where it is not validated yet (see [backlog 030](backlog/planned/030_gpu_parity_with_mlx_gen.md)); the `gpu` and `all-gpu` profiles do not include it, so install `abstractvision[mlx-gen]` explicitly there.

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

Another common case is an interrupted Hugging Face download: the snapshot
directory exists, but the repo still contains `.incomplete` blobs and the
package now rejects that cache entry as unusable until the download is resumed.

### Fix

- Pre-download the model with `abstractvision download ... --provider diffusers`
- or allow runtime downloads explicitly with `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1`
- if the repo already exists but is partial, rerun the same `abstractvision download ... --provider diffusers`
  command to resume it

Examples:

```bash
abstractvision download stable-diffusion --provider diffusers
abstractvision download qwen-image-edit-2511 --provider diffusers
```

### Verify

- `abstractvision catalog --provider diffusers`
- `abstractvision show-model Qwen/Qwen-Image-Edit-2511`

## Local Diffusers `text_to_video` is experimental and currently disabled

### Symptom

- the local playground has no active local `Text→Video` model choices
- local Diffusers `t2v` raises a capability/disabled error

### Likely cause

AbstractVision intentionally quarantines the current local Diffusers
`text_to_video` groundwork because the operator validation bar is not met yet.
This does not apply to the MLX-Gen Wan path.

### Fix

- use MLX-Gen Wan locally: `abstractvision t2v --provider mlx-gen --model AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit "prompt"`; or
- use the OpenAI-compatible backend when video is served remotely; or
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

## MLX-Gen `image_to_image` is missing or rejected

### Symptom

- `flux2-klein-4b`, `flux2-klein-9b`, `flux2-klein-base-*`, or `qwen-image-edit-*` (MLX-Gen) do not appear in the playground `Image→Image` tab; or
- local MLX-Gen `image_to_image` calls raise `CapabilityNotSupportedError`

### Likely cause

- You are on an older AbstractVision version where the Apple-local edit surface was narrower.
- The optional MLX-Gen extra is not installed (`abstractvision[mlx-gen]`).
- The q4/q8 prepared model is not present in the Hugging Face cache yet.
- You are attempting a mask or structured-control request on a route that does not advertise that capability.

### Fix

- Upgrade AbstractVision to a version that supports MLX-Gen q4/q8 presets.
- Install the backend extra: `pip install "abstractvision[mlx-gen]"`
- Download the prepared model first, for example `abstractvision download AbstractFramework/qwen-image-edit-2511-4bit --provider mlx-gen`.
- For MLX-Gen mask edits, use a validated masked-edit route such as `AbstractFramework/qwen-image-edit-2511-8bit`, `briaai/Fibo-Edit`, or `briaai/Fibo-Edit-RMBG`.
- For MLX-Gen structured control, use the validated base-Qwen route `AbstractFramework/qwen-image-8bit` and pass `--control-image` / `--control-strength` (or Python `control_image=` / `control_strength=`).
- If you need mask/control on another local route, use local Diffusers or `stable-diffusion.cpp` instead of expecting AbstractVision to silently fall back.

Notes:
- MLX-Gen edit strength is passed as `strength` and normalized to the runtime `image_strength` parameter where the model supports it.
- If you need stricter scene preservation, Diffusers often remains the more conservative baseline for `image_to_image`.

## MLX-Gen LoRA is rejected as incompatible

### Symptom

- a local MLX-Gen image or video run fails before generation with a LoRA
  compatibility error
- the message says the adapter targets a different base model than the selected
  route

### Likely cause

- the adapter model card targets a different base model family
- the selected route does not expose LoRA support
- a Wan route requires explicit target-role assignment and none was provided

### Fix

- inspect the route first with `abstractvision show-model <model-id>` or
  `abstractvision catalog --provider mlx-gen`
- confirm the route advertises `supports_lora: true`
- match the adapter to the route's base model family
- for Wan TI2V-5B, pass `--lora-target-role transformer`
- for Wan A14B, pass explicit
  `high_noise_transformer` / `low_noise_transformer` roles

### Verify

- rerun with the same model id and a compatible adapter
- confirm the generated metadata includes requested/applied LoRA details such as
  `requested_lora_adapters`, `lora_application_reports`,
  `lora_applied_file_count`, and `lora_applied_target_count`

## Wan video runs out of memory or is slow

### Symptom

- a Wan video generation fails with a Metal out-of-memory error, or the Mac
  starts swapping heavily
- a Wan video takes far longer than expected

### Likely cause

- the request asks for a larger canvas than the model's default (for example
  `1280x704` for TI2V-5B); memory and time grow with the frame size
- `vae_tiling` was switched off, so the VAE decode holds whole frames at once
- another large model is loaded in the same machine

### Fix

- leave `width`/`height` unset to use the model's default (832x480 for every
  Wan model), or request a smaller canvas; see
  [docs/faq.md](faq.md#how-much-memory-does-wan-video-need) for measured needs
- keep the tiled VAE decode on (the default); do not pass `--no-vae-tiling` /
  `extra={"vae_tiling": False}` on a memory-constrained Mac
- lower `--steps` for quick drafts: the step count changes the time, not the
  peak memory
- unload other models before a video run

## Diffusers on CUDA is slow or runs out of memory

### Symptom

- an image generation on an NVIDIA GPU takes many times longer than expected,
  and the result metadata reports `cpu_offload: "sequential"`
- the log shows `Diffusers: sequential CPU offload on cuda ... (much slower than
  model CPU offload)`
- a generation fails with `CUDA out of memory`

### Likely cause

- another process (for example a text model) holds part of the GPU, so the
  free memory cannot hold the pipeline's largest component and the backend
  falls back to sequential CPU offload
- `cpu_offload="model"` or `"none"` was set explicitly on a GPU that is too
  small for that placement

### Fix

- unload the other model from the GPU before generating; with enough free
  memory the backend uses model CPU offload or moves the pipeline whole
- leave `cpu_offload` at `"auto"` (the default) so the backend picks a
  placement that fits; see
  [docs/reference/backends.md](reference/backends.md#diffusers-backend-local)
  for how it decides
- use a smaller model or a quantized (GGUF) variant on small GPUs

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
- local MLX-Gen text-to-video: `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit` or `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- remote text-to-video: an OpenAI-compatible backend configured with a video endpoint

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
