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
- `abstractvision model-catalog --provider diffusers`

## Local Diffusers cannot find the model

### Symptom

- local Diffusers generation fails before inference starts
- the error mentions a missing snapshot or cache-only/offline behavior

### Likely cause

The Diffusers backend is cache-only by default and the required model is not yet
present in the Hugging Face cache.

### Fix

- Pre-download the model with `abstractvision download-model ... --provider diffusers`
- or allow runtime downloads explicitly with `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1`

Examples:

```bash
abstractvision download-model stable-diffusion --provider diffusers
abstractvision download-model cogvideox-2b --provider diffusers
```

### Verify

- `abstractvision model-catalog --provider diffusers`
- `abstractvision show-model zai-org/CogVideoX-2b`

## Local Diffusers text-to-video fails because `ffmpeg` is missing

### Symptom

- local `t2v` generation runs inference but fails while packaging the output
- the error says `ffmpeg` is required on `PATH`

### Likely cause

AbstractVision reuses the existing external `ffmpeg` binary path to package
generated frames into MP4. There is no bundled video encoder in the package.

### Fix

Install `ffmpeg` and make sure it is available on `PATH`.

On macOS with Homebrew:

```bash
brew install ffmpeg
```

### Verify

```bash
ffmpeg -version
```

Then retry:

```bash
abstractvision t2v --provider diffusers --model zai-org/CogVideoX-2b --diffusers-device mps --diffusers-torch-dtype float16 --num-frames 9 --steps 1 "a red fox walking through a snowy forest, cinematic"
```

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
- local text-to-video: `zai-org/CogVideoX-2b`

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
