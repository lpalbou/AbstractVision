## Task 030: GPU parity with MLX-Gen (every local capability Apple silicon has, on NVIDIA/Linux too)

**Date**: 2026-09-29  
**Status**: Planned  
**Priority**: P1  

---

## Main goals

- Everything a user can run locally on Apple silicon through MLX-Gen, a user of the `gpu` install
  setting (NVIDIA on Linux) can run locally too: same models or honest equivalents, same
  `VisionManager` calls, same AbstractCore routes and recommendations.
- The `gpu` setting installs only what actually runs on a GPU machine.

## Secondary goals

- One capability table (docs and `vision_model_capabilities.json`) that says, per model and task,
  which runtime serves it on `apple` and on `gpu`, with measured memory on both.

---

## Context / problem

The three install settings are light (remote only), `apple` and `gpu`. On `apple`, MLX-Gen gives
the curated local catalog: FLUX.2 Klein 4B/9B (text-to-image, image-to-image), Qwen-Image and
Qwen-Image-Edit (2509/2511, structured control, masked edits), Z-Image, ERNIE-Image-Turbo, Bria
FIBO / Fibo-lite / Fibo-Edit / Fibo-Edit-RMBG, SeedVR2 3B/7B upscaling, shared LoRA adapters, and
Wan 2.2 video (TI2V-5B text- and image-to-video, T2V-A14B, I2V-A14B) with per-model default
canvases and tiled VAE decode.

On `gpu` today (checked 2026-09-29):

- AbstractCore never routes MLX-Gen on a CUDA or ROCm host (`abstractcore/config/model_catalog.py`
  `_HOST_PREFERENCE`: `mlx-gen` only in the `metal` list).
- Yet `abstractvision[all-gpu]` installs `mlx-gen` on Linux (marker
  `platform_system == 'Darwin' or platform_system == 'Linux'`, `pyproject.toml:96-231`), and
  mlx-gen 0.38 requires `mlx[cuda13]` on Linux: about 2.1 GB of CUDA 13 wheels next to torch's
  CUDA 12 stack, never used, and a glibc 2.35 floor for the whole `gpu` setting (the gpu profile
  fails to resolve on older glibc). Found by the root 0.6.2 tag gate.
- Local Diffusers `text_to_video` is quarantined from the normal surfaces (task 0023) and local
  Diffusers `image_to_video` does not exist yet (task 0022). So a `gpu` user has no local video
  at all, while the registry lists diffusers repos for Wan 2.2.
- FIBO, SeedVR2 and the bonsai ternary model have no non-MLX path.

So the `gpu` setting pays for an engine it does not use and still lacks most of what `apple`
offers.

**Packaging part done (0.3.32, 2026-09-29, operator ruling):** `gpu` and `all-gpu` no longer
require `mlx-gen` (NVIDIA/Linux profiles: Diffusers/torch, plus stable-diffusion.cpp in
`all-gpu`, no MLX). The CUDA 13 wheels and the glibc 2.35 floor are gone from the `gpu` setting;
`tests/test_packaging_metadata.py::test_nvidia_gpu_profiles_never_pull_mlx` goes red if MLX comes
back. The explicit `mlx-gen` / `mflux` extras (and `apple`, `all-apple`, `all`) keep it, so
option A below can still be measured on NVIDIA by installing `abstractvision[mlx-gen]`. Plan step
2's marker half and the "installs no runtime AbstractCore never routes" criterion are met;
parity (steps 1 and 3-6) remains open.

---

## Constraints

- Keep the `VisionManager` contract and the AbstractCore plugin contract stable.
- Permissive licences only (ADR policy); heavy imports stay lazy.
- Never claim a capability on `gpu` without a real run on NVIDIA hardware (ADR 0008): the
  capability table only marks what was measured.
- Memory figures are engine measurements, labelled as such, never the model's requirement.
- No machine-specific heuristics; runtime choice stays explicit and operator-controlled (ADR 0006).

---

## Research, options, and references

- **Option A: MLX-Gen on CUDA.** mlx has a CUDA backend and mlx-gen declares `mlx[cuda13]` on
  Linux, so one runtime could serve both settings with the same weights (the AbstractFramework
  MLX quants). To measure: which mlx-gen pipelines actually run on CUDA 13, speed and memory
  against Diffusers on the same GPU, driver requirements (CUDA 13 driver), glibc floor. If it
  works, AbstractCore adds `mlx-gen` to the `cuda` preference list per validated model.
  - References: `https://pypi.org/project/mlx-gen/` (requires_dist), `https://pypi.org/project/mlx/`
    (`cuda13` extra).
- **Option B: Diffusers per family.** Use the upstream Diffusers pipelines where they exist
  (Wan 2.2 TI2V-5B / A14B T2V and I2V, FLUX.2 Klein, Qwen-Image / Qwen-Image-Edit, Z-Image),
  finishing 0022 and lifting the 0023 quarantine for Wan with per-model default canvases
  (832x480) and tiled VAE decode, as on Apple. Families with no Diffusers pipeline (FIBO,
  SeedVR2) need their upstream reference code or stay "not available on gpu yet", stated in
  the table.
- **Option C: stable-diffusion.cpp** (already in `gpu` through `sdcpp`) for the image families it
  supports with CUDA builds (FLUX, Qwen-Image, Z-Image GGUF): small memory, no torch.

Recommendation: measure A first on one NVIDIA machine (it decides whether the 2.1 GB is useful or
must go). If A is not production-ready: make the mlx-gen marker Darwin-only for `gpu`/`all-gpu`
(removes the CUDA 13 wheels and the glibc floor), and reach parity family by family with B, and C
for images where it is lighter.

---

## Plan (after the measurement)

1. Measure A on NVIDIA (image: FLUX.2 Klein 4B; video: Wan TI2V-5B at 832x480); record speed and
   peak memory next to Diffusers.
2. Decide the `gpu` runtime per family; fix the packaging marker accordingly.
3. Wan 2.2 local video on `gpu`: text-to-video and image-to-video with the same default canvases
   and a tiled decode; closes 0022/0023 for Wan.
4. Images: FLUX.2 Klein, Qwen-Image/Edit (including structured control and masked edits),
   Z-Image, ERNIE-Image-Turbo, with the shared LoRA contract (task 025).
5. FIBO and SeedVR2: port or state "not available on gpu yet".
6. AbstractCore: the `cuda` preference and the recommendations name the validated `gpu` models;
   the capability table shows both settings.

---

## Acceptance criteria

- [ ] A capability table (docs + registry) lists, per model and task, the runtime and measured
      memory on `apple` and on `gpu`; every "yes" on `gpu` has a recorded NVIDIA run.
- [ ] Wan 2.2 text-to-video and image-to-video run locally on `gpu` with the same defaults and
      overrides as on `apple`.
- [x] The `gpu` setting installs no runtime that AbstractCore never routes on a GPU host (0.3.32).
- [ ] AbstractCore recommendations for image and video on a CUDA host point to validated local
      models.

## Validation

Hermetic tests for routing and packaging markers (`uv pip compile` for `gpu` on manylinux 2.28 and
2.35); real runs on an NVIDIA machine for each validated model, with peak memory recorded.
