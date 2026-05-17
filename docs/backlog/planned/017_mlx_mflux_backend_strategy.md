## Task 017: MLX and MFLUX backend strategy

**Date**: 2026-05-15  
**Status**: Planned  
**Priority**: P1  

---

## Main goals

- Decide whether AbstractVision should use a native/general MLX image backend, an MFLUX-backed
  backend, or both for Apple Silicon image generation.
- Make low-bit Apple Silicon model downloads line up with runtime support, not just cache layout.
- Ensure Gateway and Flow only advertise local image models that AbstractVision can actually run.

## Secondary goals

- Keep base `abstractvision` import-light.
- Keep heavy local runtimes behind optional extras.
- Preserve the `VisionManager` and AbstractCore capability plugin contracts.

---

## Context / problem

AbstractVision has curated Apple Silicon download presets for 8-bit image-generation artifacts,
including FLUX.2 klein and Z-Image-Turbo. The currently practical Apple Silicon artifacts for
those models are MFLUX-compatible MLX layouts.

The runtime surface is now split:

- AbstractVision currently supports OpenAI-compatible HTTP, Hugging Face Diffusers, and
  stable-diffusion.cpp/GGUF backends.
- AbstractVision does not currently have a native MLX image backend.
- AbstractVision has a first MFLUX backend behind the optional `abstractvision[mflux]` extra for
  FLUX.2 klein and Z-Image-Turbo text-to-image generation.
- `mflux/...` is the explicit provider/model prefix for MFLUX-compatible models. `mlx/...` must not
  be treated as Diffusers; generic/native MLX remains a separate future engine decision.

The original failure mode was that an 8-bit MLX/MFLUX artifact could be downloaded successfully and
then fail inside an AbstractVision workflow because the workflow routed it to Diffusers. Diffusers
expects a Diffusers pipeline layout with `model_index.json`; MFLUX layouts use component folders
such as `transformer`, `text_encoder`, `tokenizer`, and `vae`.

## Constraints

- Do not make MFLUX, Torch, Diffusers, or stable-diffusion.cpp mandatory base dependencies.
- Do not treat a successful Hugging Face download as proof of runtime compatibility.
- Preserve offline generation after models are downloaded.
- Keep generated images as AbstractVision artifacts so Gateway ledgers and Flow run details stay
  consistent.
- Prefer permissive licensing for new runtime dependencies.
- Keep errors specific: missing `model_index.json` means "not a Diffusers pipeline", not simply
  "model missing".

---

## Research, options, and references

- **Current local runtime support**: AbstractVision backend exports OpenAI-compatible,
  Hugging Face Diffusers, stable-diffusion.cpp, and MFLUX backends. MFLUX is optional and does not
  import the `mflux` package until the backend loads a model.
  - References:
    - `src/abstractvision/backends/__init__.py`
    - `src/abstractvision/backends/mflux.py`
- **Current Flow/Gateway routing behavior**: raw Hugging Face-style repo ids and `mlx/...` routed
  requests can land in the Diffusers backend. That backend requires a Diffusers `model_index.json`.
  - References:
    - `src/abstractvision/playground_server.py`
    - `src/abstractvision/integrations/abstractcore_plugin.py`
    - `src/abstractvision/backends/huggingface_diffusers.py`
- **Current download preset behavior**: Apple Silicon 8-bit presets specify both artifact target
  (`target="mlx"`) and runtime engine (`engine="mflux"`). The old `runner` JSON key is kept as a
  compatibility alias for the first downloader iteration.
  - References:
    - `src/abstractvision/model_downloads.py`
- **Repository manifest check on 2026-05-15**: upstream full-model repos
  `black-forest-labs/FLUX.2-klein-4B`, `black-forest-labs/FLUX.2-klein-9B`, and
  `Tongyi-MAI/Z-Image-Turbo` expose `model_index.json` and are Diffusers-style pipeline repos.
  The 8-bit MLX/MFLUX repos `AITRADER/FLUX2-klein-4B-mlx-8bit`,
  `deepsweet/FLUX.2-klein-9B-MLX-Q8`, and `carsenk/z-image-turbo-mflux-8bit` do not expose
  `model_index.json`. BFL's FP8 side-artifact repos checked here also do not expose
  `model_index.json`, so they should not be treated as standalone Diffusers pipelines without an
  explicit loader strategy.
  - References:
    - `https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/main`
    - `https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/tree/main`
    - `https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main`

### Option A: Add an MFLUX backend first

Pros:

- Fastest path to make the current FLUX.2 klein and Z-Image-Turbo 8-bit downloads usable from
  AbstractVision, Gateway, and Flow.
- Keeps Apple Silicon memory use low.
- Can be isolated behind an optional extra and lazy imports.

Cons:

- Tied to MFLUX-supported model families.
- Needs careful subprocess or in-process cancellation and artifact handling.
- Adds another runtime dependency unless dependency selection is explicit.

### Option B: Add a native/general MLX image backend first

Pros:

- Better long-term fit if it can load real MLX image model layouts from configuration.
- Avoids binding AbstractVision to only MFLUX-supported architectures.
- Matches the preference for fewer specialized dependencies.

Cons:

- Larger implementation risk.
- May recreate a large part of Diffusers-style model loading in MLX.
- Needs proof that enough image model families can be loaded reliably.

### Option C: Keep only Diffusers and stable-diffusion.cpp for now

Pros:

- No new runtime dependency.
- Keeps backend surface small.

Cons:

- The Apple Silicon 8-bit MLX presets remain unusable from AbstractVision workflows.
- Users can download a model through AbstractVision but cannot generate with it through
  AbstractVision.

---

## Decision

**Chosen approach**: ship MFLUX as the first runnable Apple 8-bit bridge, and keep native/general
MLX as the preferred long-term strategy if feasibility is good.

**Why**:

- The immediate bug is runtime incompatibility, not download failure.
- The small MFLUX backend makes existing 8-bit Apple artifacts useful through AbstractVision today.
- A real MLX backend should not be promised until it can load model families without broad,
  fragile per-model rewrites.
- The downloader and model catalog must distinguish artifact format from runnable backend.

---

## Dependencies

- **ADRs**:
  - N/A
- **Backlog tasks**:
  - Completed: `docs/backlog/completed/007_local_hf_backend_strategy_diffusers.md`
  - Completed: `docs/backlog/completed/013_stable_diffusion_cpp_gguf_backend.md`
  - Completed: `docs/backlog/completed/015_vision_install_profiles_and_pending_defaults.md`

---

## Implementation plan

- Update model listing so downloadable presets and runnable backend models are distinct.
- Make `mlx/...` fail fast with an actionable unsupported-backend error unless a real MLX backend
  exists.
- Keep improving the MFLUX backend behind an optional extra, using the existing artifact output
  path.
- Evaluate a native/general MLX image backend with at least FLUX.2 klein, Z-Image-Turbo, and one
  Stable Diffusion family before deciding whether to make it a first-class backend.
- Add Gateway/Flow model selection metadata so users see only runnable provider/model pairs.
- Document exact command lines for models that are downloaded but not yet runnable through
  AbstractVision.

---

## Success criteria

- Apple Silicon 8-bit presets are never silently routed to Diffusers unless they are real
  Diffusers pipelines with `model_index.json`.
- The `mlx` provider/model prefix maps to a real backend or emits an actionable unsupported-backend
  error.
- At least one local Apple 8-bit image model can be generated through AbstractVision and Gateway
  without manually calling an external CLI. Initial smoke coverage includes FLUX.2 klein 4B,
  FLUX.2 klein 9B, and Z-Image-Turbo.
- Runtime extras remain optional and base `abstractvision` stays import-light.
- Docs explain when to choose Diffusers, MFLUX, native MLX, and stable-diffusion.cpp/GGUF.

---

## Test plan

- `PYTHONPATH=src python -m unittest tests.test_cli_smoke -q`
- `PYTHONPATH=src python -m unittest tests.test_abstractcore_plugin -q`
- Backend unit tests for unsupported `mlx/...` routing when no MLX backend is installed.
- Backend smoke test for selected Apple 8-bit models once an MFLUX or MLX backend exists.
- Manual Gateway/Flow run proving model selection routes to the intended backend.

---

## Report (fill only when completed)

### Summary

N/A

### Validation

- Tests: N/A
