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
- `ProviderAdapterInfo`
- `ProviderModelInfo`
- `VisionAdapterCapabilitiesRegistry`
- `VisionModelCapabilitiesRegistry`
- `LocalAssetStore`
- `RuntimeArtifactStoreAdapter`
- `LoRAAdapterSpec`
- `is_artifact_ref`
- `__version__`

## Core concepts

### Tasks

`VisionManager` exposes one method per task (implementation: [`../src/abstractvision/vision_manager.py`](../src/abstractvision/vision_manager.py)):

- `generate_image(...)` → `text_to_image`
- `generate_image_batch(...)` → repeated `text_to_image` orchestration with explicit seed planning
- `edit_image(...)` → `image_to_image`
- `edit_image_batch(...)` → repeated `image_to_image` orchestration with explicit seed planning
- `upscale_image(...)` → `image_upscale`
- `generate_video(...)` → `text_to_video` (backend-dependent)
- `generate_video_batch(...)` → repeated `text_to_video` orchestration with explicit seed planning
- `image_to_video(...)` → `image_to_video` (backend-dependent)
- `image_to_video_batch(...)` → repeated `image_to_video` orchestration with explicit seed planning
- `generate_angles(...)` → `multi_view_image` (API exists; no built-in backend implements it yet)

Task names are also used by the capability registry ([`../src/abstractvision/assets/vision_model_capabilities.json`](../src/abstractvision/assets/vision_model_capabilities.json)).

### Backends

Backends are execution engines that implement the `VisionBackend` interface ([`../src/abstractvision/backends/base_backend.py`](../src/abstractvision/backends/base_backend.py)).

Built-in backends live in [`../src/abstractvision/backends/`](../src/abstractvision/backends/):
- `OpenAICompatibleVisionBackend` (HTTP)
- `HuggingFaceDiffusersVisionBackend` (local Diffusers images; local Diffusers `text_to_video` groundwork is currently quarantined)
- `StableDiffusionCppVisionBackend` (local stable-diffusion.cpp / GGUF)
- `MLXGenVisionBackend` / compatibility alias `MFluxVisionBackend` (local Apple-first MLX-Gen bridge for curated AbstractFramework q4/q8 MLX presets, official FIBO snapshots, shared LoRA adapters, and Wan video)

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

Adapter inventory is exposed through the same explicit pattern:

```python
for adapter in backend.list_provider_adapters(
    model="AbstractFramework/qwen-image-edit-2509-8bit",
    task="image_to_image",
):
    print(adapter.id)
```

`ProviderAdapterInfo` rows are explicit discovery outputs. They do not mutate
the configured generation route.

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
from abstractvision import LoRAAdapterSpec, VisionManager
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
of the public API. MLX-Gen `0.40.1+` supports SeedVR2 `image_upscale`, Wan
`text_to_video`, and first-frame `image_to_video`, including A14B task-specific
checkpoints. Local Diffusers video remains experimental and disabled from the
normal local surfaces. Generated MP4 outputs still require an `ffmpeg`
executable on `PATH` whenever a backend returns frame sequences for local
packaging.

For MLX-Gen text-to-image, `ImageGenerationRequest` includes typed
`control_image` and `control_strength` fields for the validated structured-control
route `AbstractFramework/qwen-image-8bit`. The control image is a structure guide
such as edges, sketch, or pose, not a source-image edit input. These are not
generic cross-backend hints: unsupported backends reject them explicitly.

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
    lora_adapters=[
        LoRAAdapterSpec(
            source="prithivMLmods/Qwen-Image-2512-Pixel-Art-LoRA:Qwen-Image-2512-Master-Pixel-Art-LoRA.safetensors",
            scale=1.0,
        )
    ],
    on_progress=on_progress,
)

edit_asset = image_vm.edit_image(
    "compose the subject using the second image as a style and layout reference",
    image=Path("./subject.png").read_bytes(),
    steps=12,
    guidance_scale=1.0,
    lora_adapters=[
        LoRAAdapterSpec(
            source="fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors",
            scale=0.9,
        )
    ],
    on_progress=on_progress,
    extra={"reference_images": [Path("./style-reference.png").read_bytes()]},
)

upscale_backend = MLXGenVisionBackend(config=MLXGenBackendConfig(model="AbstractFramework/seedvr2-3b-8bit"))
upscale_vm = VisionManager(backend=upscale_backend)

upscaled_asset = upscale_vm.upscale_image(
    image=Path("./subject.png").read_bytes(),
    resolution="2x",
    softness=0.25,
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
    lora_adapters=[
        LoRAAdapterSpec(
            source="AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets:HSToric_color_Wan22_5b_LoRA.safetensors",
            scale=0.9,
            target_role="transformer",
        )
    ],
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
    lora_adapters=[
        LoRAAdapterSpec(
            source="AlekseyCalvin/HSToric_Color_Wan2.2_5B_LoRA_BySilverAgePoets:HSToric_color_Wan22_5b_LoRA.safetensors",
            scale=0.9,
            target_role="transformer",
        )
    ],
    on_progress=on_progress,
    extra={"max_sequence_length": 256},
)
```

Wan 2.2 A14B has two guidance controls. Use `guidance_scale` for the
primary/high-noise stage and `guidance_2` for the second/low-noise stage. The
registry default is `guidance_2=3.0` for text-to-video A14B and `3.5` for
image-to-video A14B. Other video models should omit `guidance_2` unless their
registry task declares it.

For TI2V-5B, use `flow_shift` directly when you need to override the route
default. The bundled visual proof uses `832x480` with `flow_shift=3.0`.

Wan video defaults come from one per-model table,
`WAN_VIDEO_DEFAULT_CANVASES` in
[`../src/abstractvision/backends/mflux.py`](../src/abstractvision/backends/mflux.py):
832x480 for every Wan model, 121 frames at 24 fps and 50 steps for TI2V-5B,
81 frames at 16 fps and 40 steps for the A14B models. Any field you pass
(`width`, `height`, `num_frames`, `fps`, `steps`) overrides the default;
`backend.list_provider_models(task="text_to_video")` reports the same values in
each model's `raw["parameter_defaults"]`. The Wan VAE decode runs in overlapping spatial tiles by
default, which bounds its memory; pass `extra={"vae_tiling": False}` to decode
whole frames at once. Generated asset metadata records `vae_tiling`. Measured
memory per size: [docs/faq.md](faq.md#how-much-memory-does-wan-video-need).

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

## Memory modes (`memory_mode`)

Every generation request (`generate_image`, `edit_image`, `upscale_image`,
`generate_video`, `image_to_video` and their `_batch` variants) takes
`memory_mode`: `"default"`, `"low"` or `"lowest"` (`MEMORY_MODES` in
[`../src/abstractvision/types.py`](../src/abstractvision/types.py)). The lower
modes trade speed for a lower memory peak. A request that names no mode uses
`$ABSTRACTVISION_MEMORY_MODE`, else `"default"`; a request value always wins.
The environment value is one setting for every backend in the process: a
backend that cannot honour it runs the lowest-memory mode it lists that is not
below it (`"lowest"` becomes `"low"` where `"low"` is listed, else `"default"`)
and logs that once per process, while a mode passed on the request is refused.
Backend warmups (`preload`) use the same rule; MLX-Gen warms up in `"default"`,
since `"low"` would release the model it preloads. On MLX-Gen, `preload()` loads
the model and runs one image generation as its warm-up on image and edit models;
video models (Wan, MiniMax-H3) are loaded without a warm-up run. When anything in
a preload fails, the model it loaded is dropped before the error is raised.

```python
asset_or_ref = vm.generate_video(
    "a red fox trots through fresh snow",
    memory_mode="low",
)
```

| Backend | `low` | `lowest` |
| --- | --- | --- |
| MLX-Gen, image and edit routes | MLX-Gen low-RAM mode, as `mlxgen generate --low-ram`: a 1 GB MLX cache from before the model loads to the end of the run, and mlx-gen's `MemorySaver`, which tiles the VAE decode, frees the text encoders before the denoising loop and the transformer after it. | `low` plus transformer block streaming (`--stream-blocks`), on a model whose MLX-Gen routes publish `supports_block_streaming`: accepted for MiniMax-H3, refused on every other current catalog model. |
| MLX-Gen, Wan video | As `mlxgen generate --low-ram` on one seed: the 1 GB cache, the MLX cache cleared after every step and transformer block, the denoisers released before the decode and, on the A14B models, the inactive expert released. | Refused on every current Wan route. |
| MLX-Gen, MiniMax-H3 | The 1 GB cache and the transformer freed before the decode (`release_transformer=True`); the model stays loaded. | `low` plus block streaming (`stream_blocks=True`). Refused with load-time `quantize`, and with an adapter that trains the AdaLN projections. |
| MLX-Gen, SeedVR2 upscale | As `mlxgen upscale --low-ram` on images: the 1 GB cache only. SeedVR2 keeps its own tiling and the loaded model. | Refused (SeedVR2 has no block streaming). |
| Diffusers on CUDA | Model CPU offload. | Sequential CPU offload. |
| Diffusers on CPU or MPS | Refused. | Refused. |
| stable-diffusion.cpp | CPU offload of the model parameters: `--offload-to-cpu` for `sd-cli`, `offload_params_to_cpu` for the Python binding. | Refused. |
| OpenAI-compatible | Sent to a remote that lists the mode for the model in `GET /vision/models` (AbstractCore's server); refused otherwise. | As `low`. |

A mode named on the request that the backend cannot honour for its configured
model and device raises `CapabilityNotSupportedError` before any model loads or
any request is sent,
whether the call comes through `VisionManager`, the AbstractCore plugin or a
direct backend call. `extra={"memory_mode": ...}` is read the same way as the
keyword; a keyword and an `extra` value that disagree raise `ValueError`.

Placement is fixed when a model loads, so a request may only raise it. On
Diffusers, the CPU offload that runs is the higher of the configured
`cpu_offload` (or what `"auto"` picks) and the one the request needs; a loaded
pipeline with less offload than a request needs is loaded again, and one with
more offload keeps serving later requests until it is unloaded. stable-diffusion.cpp
keeps an operator's `--offload-to-cpu` and loads the Python binding again
with `offload_params_to_cpu` when a `low` request needs it.

On MLX-Gen, `low` trades residency for memory: image, edit and Wan runs free
parts of the model, so the instance is dropped after the run and the next
request loads it again, a preloaded resident model included. Batches
(`generate_image_batch`, `generate_video_batch`, ...) keep the instance until
their last item, as mlx-gen does for several seeds (VisionManager marks batch
items only for backends whose `uses_batch_marker` is true; MLX-Gen is the one
built-in); FIBO, which mlx-gen does not
run in low-RAM mode with several seeds, runs each item as its own low-RAM run.
SeedVR2 keeps its instance. MLX-Gen holds one instance at a time: a request
whose model, adapters, `quantize` or block streaming differ from the loaded
instance's releases it, and clears the MLX cache, before the new one loads.

Discovery: `backend.get_capabilities().memory_modes` lists the modes a backend
can honour for its configured model and device, and `supports_block_streaming`
says whether the configured MLX-Gen model streams blocks. Each MLX-Gen provider
catalog entry carries `raw["parameter_constraints"]["memory_mode"]` =
`{"enum": [...], "block_streaming": <bool>}`, where `enum` holds the modes that
model can honour. A run with a mode other than `default` records `memory_mode`
in the result metadata (MLX-Gen also records `low_ram` and `stream_blocks`;
stable-diffusion.cpp records `offload_to_cpu`; Diffusers records `cpu_offload`).

## Video with sound (MiniMax-H3)

The MLX-Gen backend serves MiniMax-H3, which generates a video with a synchronized
stereo soundtrack: `text_to_video`, and `image_to_video` from a first frame. The
catalog entry is `MiniMaxAI/MiniMax-H3`; the published packages are
`AbstractFramework/minimax-h3-8bit` (64.3 GiB, recommended),
`AbstractFramework/minimax-h3-bf16` (115.1 GiB) and the mixed
`AbstractFramework/minimax-h3-4bit` (44.9 GiB). The packages hold the base
weights; `base_model` picks the schedule: `minimax-h3-turbo-544p` (8 steps,
960x544), `minimax-h3-turbo` (8 steps, 1344x768) or `minimax-h3` (50 steps).

```python
from abstractvision import VisionManager
from abstractvision.backends.mflux import MLXGenBackendConfig, MLXGenVisionBackend

backend = MLXGenVisionBackend(config=MLXGenBackendConfig(
    model="AbstractFramework/minimax-h3-8bit", base_model="minimax-h3-turbo-544p"))
clip = VisionManager(backend=backend).generate_video(
    "A red fox trots through fresh snow at dawn.",
    soundscape="Soft crunch of paws in dry snow, faint wind.", music="Sparse piano.",
    width=640, height=352, num_frames=124, seed=42, memory_mode="low")
```

Request fields (on `VideoGenerationRequest` and `ImageToVideoRequest`):

| Field | Meaning |
| --- | --- |
| `prompt` | The visual description, or a complete structured prompt with its section labels. Passed to MLX-Gen unchanged; MLX-Gen composes the structured prompt (and, for a first frame, its picture reference). |
| `soundscape`, `music` | The `overall_soundscape` and `non_diegetic_music` sections. |
| `generate_audio` | `False` writes a silent clip. Default: the soundtrack is generated. A boolean: any other type, the string `"false"` included, is refused. |
| `flow_shift`, `audio_shift` | The video and audio schedule shifts (MLX-Gen `video_shift` / `audio_shift`), positive numbers. |
| `width`, `height` | Given together; positive multiples of the route's `dimension_multiple` (32), with an aspect ratio inside MLX-Gen's window (1:4 to 4:1). Unset: the schedule's canvas, or for a first frame MiniMax-H3's own mapping of the keyframe's aspect ratio, which must then lie in the same window. |
| `num_frames` | Rounded up by MLX-Gen to its grid (`17n+5` at 24 fps, `124` to `345`); a count outside the grid is refused before any weights load. |
| `steps`, `seed`, `quantize`, `memory_mode`, `cancel_event` | As for other routes; `steps` is at least 1 and `seed` lies in `[0, 2**64 - 1]` (MLX's random key). `quantize` applies to the source snapshot only (8 or 4, the row's `validated_quantization_bits`). |
| `lora_adapters` | Adapters on the `transformer` role (the row's `lora_target_roles`). As in MLX-Gen, the list replaces the Turbo adapter a Turbo schedule attaches: include the Turbo file to keep the 8-step schedule. |

Values left unset stay unset, so MLX-Gen applies the schedule's own defaults.
Everything is checked before any weights load, against MLX-Gen's capability row
and its own MiniMax-H3 planning code, and every refusal names the field:

- What the route cannot apply, or a value outside what it accepts, raises
  `CapabilityNotSupportedError`: `negative_prompt`, `guidance_scale`,
  `guidance_2`, an `fps` other than 24, an `extra` key other than the progress
  callbacks and `memory_mode`, a canvas off the multiple or outside the aspect
  window, a frame count off the grid, `steps` below 1, a shift that is not
  positive.
- A value of the wrong type, or two fields that contradict each other, raises
  `ValueError`: a width that is not an integer, `soundscape` or `music` that is
  not a string, a non-boolean `generate_audio`, a prompt that already carries an
  `overall_soundscape:` (or `non_diegetic_music:`) section together with
  `soundscape=` (or `music=`), which MLX-Gen's own prompt composition refuses.
- A first frame is opened and decoded first: empty or undecodable bytes raise
  `ValueError` naming `image`. Wan routes check their values and keyframes the
  same way, with the same error types.
- `memory_mode="lowest"` with an adapter that trains the transformer's AdaLN
  projections (checked with MLX-Gen's own adapter test) is refused, naming the
  adapter: MLX-Gen must then hold the whole transformer in memory.

Every number the backend checks comes from MLX-Gen (the capability row, its
aspect window); `describe_routes()` returns the row. Wan and the other local
backends refuse `soundscape`, `music`, `generate_audio` and `audio_shift`; the
OpenAI-compatible backend sends them to a remote that advertises them for its
model and refuses them otherwise
([backends](reference/backends.md#openai-compatible-http-backend)).

The result is the MP4 with its AAC soundtrack. `GeneratedAsset.audio` is a
`GeneratedAudioTrack` (`sample_rate`, `channels`, `duration_seconds`, `muxed`,
`codec`); when MLX-Gen could not mux the track, `muxed` is false and `data`
holds it as WAV bytes. `GeneratedAsset.audio_summary()` reports the soundtrack
in the one shape every layer uses (None without one):

```python
{"present": True, "sample_rate": 32000, "channels": 2, "codec": "aac",
 "muxed": True, "duration_seconds": 5.1667}
```

`codec` is the WAV's (`"pcm_s16le"`) when the track is not muxed, and the
summary then adds `"wav"`: the WAV bytes on the asset, the stored WAV's artifact
ref once stored. With a store, `VisionManager` stores such a WAV as an artifact
of its own (tags `modality: audio`) and records the summary as `audio` in the
video's stored metadata and in the returned ref. `metadata` carries `prompt`
(the composed prompt), `seed`, `steps`, `width`, `height`, `num_frames`,
`requested_frames`, `duration_seconds`, the two shifts, and `mlx_gen`: MLX-Gen's
own metadata sidecar (`<clip>.metadata.json`), including its audio mux facts and
`runtime_memory` (the process's memory at the end of the save, peaks included).

Memory modes: `low` sets the 1 GB MLX cache and `release_transformer=True`
(the transformer is freed before the decode and read again by the next run; the
model stays loaded, batch items included). `lowest` adds block streaming
(`stream_blocks=True`): each transformer block is read from disk for the length
of its call. It works from a prepared package or from the 16-bit source, and is
refused with `quantize`. MLX-Gen's measured peaks for each package and mode are
in the route row's `measured_runs`; `abstractvision show-model MiniMaxAI/MiniMax-H3`
prints them per package, with the smallest memory each mode was tested on. When
MLX-Gen's preflight finds that a run cannot fit, the backend raises
`GpuOutOfMemoryError` naming the `memory_mode` the run used and the lower modes
to try.

The weights are never downloaded during generation, even with
`allow_download=True`: the AbstractFramework packages are gated, and the MiniMax
H3 Community License's grant excludes the European Union, the United Kingdom, the
Republic of Korea and the United States of America (the source repository
`MiniMaxAI/MiniMax-H3` is not gated; the same licence applies). Accept the
licence on the package page while logged in to Hugging Face and download
explicitly: `abstractvision download MiniMaxAI/MiniMax-H3` fetches the
recommended 8-bit package (`--bits 4` or `--bits 16` the others), and
`--base-model minimax-h3-turbo-544p` (or `minimax-h3-turbo`) also caches the
Turbo adapter that schedule loads. `minimax-h3` loads no adapter. See
[Downloads](reference/configuration.md#downloads).

## Host integration

What an application embedding `VisionManager` can rely on:

- **Engine metadata.** MLX-Gen results carry the engine's metadata under
  `metadata["mlx_gen"]` (image, edit, upscale and video routes).
- **Cancellation.** Every request takes `cancel_event` (anything with `is_set()`,
  such as `threading.Event`). A set event stops an MLX-Gen run at its next
  progress event, between denoising steps, and raises `GenerationCancelledError`.
  What stays loaded after a cancelled run depends on the family:
  - image, edit and Wan A14B keep the model in `"default"`;
  - SeedVR2 and MiniMax-H3 keep it in every mode (a MiniMax-H3 transformer freed
    before the decode is read again by the next run);
  - Wan TI2V-5B always drops it (it releases its denoiser before the decode);
  - image and edit runs in `"low"` drop it, a batch item included; Wan A14B runs
    in `"low"` drop it, except on a batch item before the last.

  Other backends check the event before a run starts; the OpenAI-compatible
  backend also cancels its remote job on a remote that advertises the vision
  routes. Such a request runs there as a job, which AbstractCore's server refuses
  while another generation runs (a request without `cancel_event` waits instead);
  the backend raises that as a `RuntimeError` saying so.
  `get_capabilities().cancellation` is `"step"` or `"start"`.
- **Load-time quantize.** `quantize` (bits) on generation requests.
  `get_capabilities().quantize_bits` lists the widths the configured model
  accepts: the route's `validated_quantization_bits`, else MLX-Gen's `--quantize`
  choices (for MiniMax-H3, its own 8 and 4). A prepared (already quantized)
  package, and the other local backends, refuse it; the OpenAI-compatible backend
  sends it to a remote that advertises it.
- **Metadata without local paths.** MLX-Gen results report the model as its repo
  id (a Hugging Face cache snapshot) or its directory name (another local path),
  and adapters the same way (`org/name:file.safetensors`, or the file name):
  every absolute path in the engine's metadata, whatever field carries it
  (`model`, `lora_paths`, `lora_application_reports`, ...), in `metadata["mlx_gen"]`
  and its top-level copies, and `requested_lora_adapters`; the paths of the temporary files AbstractVision wrote
  for a run (`image_path`, `video_path`, `audio_sidecar_path`) are left out.
- **Route rows.** `vm.describe_routes(model=None, task=None)` returns MLX-Gen's
  capability rows (schema 17) as JSON: canvas and frame rules, defaults, LoRA,
  audio, `supports_block_streaming`, `measured_runs`, quantization. Other
  backends return `[]`.
- **Progress.** `on_progress(event)` receives `VideoProgressEvent` with `phase`,
  `step`, `total_steps` (and `frame`, `total_frames` on video) as MLX-Gen emits
  them. An exception raised in a progress callback is logged once per run and
  does not stop the run; use `cancel_event` to stop one.
- **Embedding.** `import abstractvision` and constructing a `VisionManager` write
  no files, set no environment variables, configure no loggers and import neither
  AbstractCore nor an engine. MLX-Gen is imported at the first capability lookup
  or model load; its own import sets `TOKENIZERS_PARALLELISM=false` when unset.

## Shared LoRA adapters

AbstractVision exposes a typed LoRA contract across Python, CLI, and the
AbstractCore plugin through `LoRAAdapterSpec` plus the request field
`lora_adapters=[...]`.

Each adapter can carry:

- `source`
- `scale`
- `weight_name`
- `subfolder`
- `adapter_name`
- `target_role`

Wan TI2V-5B uses one role, `transformer`. Wan A14B routes require explicit
`high_noise_transformer` / `low_noise_transformer` assignment. Provider catalogs
surface exact-route LoRA truth through `supports_lora`, `lora_status`,
`lora_target_roles`, and `lora_validation_profile`.

## Passing advanced backend parameters (`extra`)

Request dataclasses include an `extra: dict` field ([`../src/abstractvision/types.py`](../src/abstractvision/types.py)). Use it to pass backend-specific parameters in a controlled way:

```python
asset_or_ref = vm.generate_image(
    "a product photo of a matte black espresso machine",
    steps=8,
    guidance_scale=1.0,
    extra={
        # Compatibility-only keys used by older Diffusers flows:
        "rapid_aio_repo": "linoyts/Qwen-Image-Edit-Rapid-AIO",
    },
)
```

Backends may ignore unknown keys; consult the backend implementation and
[docs/reference/backends.md](reference/backends.md). New callers should prefer
the typed shared `lora_adapters` field instead of `extra["loras*"]`. Legacy
Diffusers compatibility payloads such as `loras_json` still work, but they are
no longer the preferred public contract.

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

### Downloads and what is installed

The download presets live in
[`../src/abstractvision/model_downloads.py`](../src/abstractvision/model_downloads.py); the state of a
preset on this host in [`../src/abstractvision/model_inventory.py`](../src/abstractvision/model_inventory.py).
The CLI's `download`, `catalog`, `show-model` and `models --json` are built on them.

```python
from abstractvision.model_downloads import download_model_preset, find_model_preset, plan_model_preset_download
from abstractvision.model_inventory import preset_status

preset = find_model_preset("MiniMaxAI/MiniMax-H3")         # the recommended package: AbstractFramework/minimax-h3-8bit
preset = find_model_preset("MiniMaxAI/MiniMax-H3", bits=4)  # AbstractFramework/minimax-h3-4bit
print(preset.gated, preset.license_name, preset.recommended)  # from the catalog
plan = plan_model_preset_download(preset)  # reads the Hub listing; downloads nothing
print(plan.summary_lines(), plan.disk_space_error())  # size, files already cached, free disk space
print(preset_status(preset))  # installed / installed_bytes / partial / memory_modes / memory_measurements
snapshot_dir = download_model_preset(preset, report=print)
```

- `find_model_preset(name, *, target, engine, require_8bit, bits)`: an exact repo id selects that
  package; a model id with several packages selects the width `bits` names, else the package the
  catalog marks `recommended`, else raises `ValueError` listing the choices.
- `plan_model_preset_download(preset, *, token, all_files)`: a `HubDownloadPlan` with `total_bytes`
  (None when the Hub listing is not available), `present_bytes` (already in the cache, partial files
  included), `missing_bytes`, `free_bytes`, `summary_lines()` and `disk_space_error()` (the message,
  or None when the download fits or its size is unknown). `download --dry-run` prints it.
- `download_model_preset(preset, *, token, max_workers, all_files, report)`: sizes the download from
  the Hub listing before it starts and raises `InsufficientDiskSpaceError` (a `RuntimeError`) when the
  cache disk cannot hold what is still missing; resumes from files already in the cache; raises
  `HuggingFaceAccessError` naming the repository page and the licence when a gated repository refuses
  the account; raises `RuntimeError` when files are missing after the download. `report` receives the
  size and cache summary.
- `preset_status(preset)`: `installed` and `installed_bytes` (the complete cached snapshot and its size
  on disk), `partial`, `memory_modes` (what the backend serving the preset honours on this host; `None`
  when that backend cannot run here) and, for MLX-Gen prepared packages whose capability rows publish
  measured runs, `memory_measurements`: per mode the whole-process `peak_bytes` of the smallest measured
  canvas, the run's `width`, `height`, `frames` and `steps`, and `smallest_tested_machine_gb` where
  mlx-gen publishes it. The figures are read from the installed mlx-gen at run time.
- `abstractvision.backends.mflux.resolve_mlx_gen_schedule_adapter(base_model, download=False)`: the
  adapter an MLX-Gen schedule loads (the MiniMax-H3 Turbo LoRA) and its cached path, or `None` when the
  schedule loads none; `download=True` fetches it, as `abstractvision download --base-model` does.

## Artifacts and stores

Artifact helpers and store implementations live in [`../src/abstractvision/artifacts.py`](../src/abstractvision/artifacts.py):

- `LocalAssetStore` (standalone local files, default `~/.abstractvision/assets`)
- `RuntimeArtifactStoreAdapter` (duck-typed adapter for an external artifact store)
- `is_artifact_ref(...)` / `make_media_ref(...)`

See: [docs/reference/artifacts.md](reference/artifacts.md).

## Errors you may want to handle

Common exceptions (defined in [`../src/abstractvision/errors.py`](../src/abstractvision/errors.py)):

- `BackendNotConfiguredError` (calling `VisionManager` without a backend)
- `CapabilityNotSupportedError` (task isn’t supported by the model registry or backend, or the backend does not accept the requested `memory_mode` or another typed option)
- `GenerationCancelledError` (the request's `cancel_event` was set)
- `UnknownModelError` (model id isn’t present in the registry)
- `OptionalDependencyMissingError` (backend dependency is missing, e.g. Diffusers/Torch)
- `GpuOutOfMemoryError` (a CUDA out-of-memory during a Diffusers generation: the message names the model's size, its placement, what was free and what to change: a smaller canvas or fewer steps, sequential CPU offload, or the GGUF path; raised instead of torch's traceback. On MLX-Gen MiniMax-H3 routes, raised when MLX-Gen's preflight reports that the weights or a run cannot fit, naming the `memory_mode` it used and the lower ones; MLX-Gen 0.40.1 reports this for MiniMax-H3 only)
