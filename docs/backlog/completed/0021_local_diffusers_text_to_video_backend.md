# Completed: First local Diffusers text-to-video backend and surface integration

## Metadata
- Created: 2026-05-21
- Status: Completed
- Priority: P1
- Completed: 2026-05-21

## ADR status
- Governing ADRs:
  - [ADR 0003: Keep base packaging lightweight and put heavy runtimes behind explicit extras](../../adr/0003_lightweight_base_package_and_explicit_runtime_extras.md)
  - [ADR 0004: Keep the orchestrator thin and make model semantics backend-owned](../../adr/0004_thin_orchestrator_and_backend_owned_model_semantics.md)
  - [ADR 0005: Own a curated capability registry and cache-backed model catalog](../../adr/0005_curated_capability_registry_and_download_catalog.md)
  - [ADR 0006: Keep runtime selection explicit and operator-controlled](../../adr/0006_operator_control_configuration_precedence_and_explicit_network_use.md)
  - [ADR 0007: Disclose fallbacks and degraded modes explicitly](../../adr/0007_explicit_fallback_and_degraded_mode_disclosure.md)
  - [ADR 0008: Require validation and evidence-based change reporting](../../adr/0008_validation_and_evidence_based_change_reporting.md)
  - [ADR 0009: Keep docs, backlog, and ADRs code-first](../../adr/0009_code_first_docs_backlog_and_adr_discipline.md)
- ADR impact: None if the first implementation stays inside the existing backend-owned task model and explicit runtime-selection rules.

## Context
AbstractVision already owns the package boundary for both image and video generation. The public
API, registry, and AbstractCore integration already acknowledge `text_to_video` and
`image_to_video`, but local execution is still missing.

The immediate operator goal is narrow and practical:

- prepare one real local text-to-video test path on Apple Silicon;
- keep image-edit support working and visible across package surfaces;
- extend CLI, playground, and AbstractCore only where the runtime truth is real.

## Current code reality
Files and symbols re-checked on 2026-05-21:

- `README.md`
- `docs/architecture.md`
- `src/abstractvision/vision_manager.py`
- `src/abstractvision/backends/huggingface_diffusers.py`
- `src/abstractvision/backends/openai_compatible.py`
- `src/abstractvision/cli.py`
- `src/abstractvision/playground_server.py`
- `src/abstractvision/integrations/abstractcore.py`
- `src/abstractvision/integrations/abstractcore_plugin.py`
- `src/abstractvision/assets/vision_model_capabilities.json`

What already exists:

- `VisionManager.generate_video(...)` and `VisionManager.image_to_video(...)` are already part of
  the stable package contract.
- The OpenAI-compatible backend already implements optional remote `text_to_video` and
  `image_to_video` when endpoints are configured.
- AbstractCore already exposes `vision_text_to_video` and `vision_image_to_video`, and the plugin
  already routes `t2v` / `i2v`.
- Image editing is already implemented and exposed across the package:
  - manager `edit_image(...)`
  - CLI `i2i`
  - playground image-edit endpoint and UI panel
  - AbstractCore `vision_image_to_image`
- On this machine, there are already cached and loadable local image-edit models, including:
  - `black-forest-labs/FLUX.2-klein-4B` via `mflux` (8-bit)
  - `black-forest-labs/FLUX.2-klein-9B` via `mflux` (8-bit)
  - `black-forest-labs/FLUX.2-klein-base-4B` via `mflux` (8-bit)
  - `zai-org/GLM-Image` via Diffusers (16-bit)

What is missing or mismatched:

- `HuggingFaceDiffusersVisionBackend.generate_video(...)` and `.image_to_video(...)` still raise
  “phase 2” `CapabilityNotSupportedError`.
- The packaged capability registry includes several video families, but the local backend does not
  execute them yet.
- The registry does not currently carry a dedicated entry for `zai-org/CogVideoX-2b`, even though
  it is a pragmatic first local text-to-video test target.
- CLI has image-oriented one-shot commands (`t2i`, `i2i`) but no symmetric local `t2v` / `i2v`
  one-shot command yet.
- The playground currently filters its model list to image tasks only, so local video models will
  remain invisible there even after download.

## Problem
AbstractVision already claims a video-capable package boundary, but local video is still remote-only
in practice. That creates a gap between:

- public API shape;
- registry/catalog claims;
- downloaded model inventory;
- what CLI, playground, and local backends can actually execute.

## What we want to do
Ship one honest, narrow, local text-to-video path first, then expose it deliberately across the
surfaces that can support it.

The first target should be `zai-org/CogVideoX-2b` through Diffusers because it is Apache-2.0,
small enough to be practical for testing, and already published as a standard Diffusers pipeline.

## Why
- It gives the package a real local video baseline instead of a purely aspirational API.
- It keeps the first implementation constrained to a concrete model family and runtime.
- It supports Apple Silicon bring-up without inventing a new package split.
- It avoids conflating “downloadable video model” with “runnable local video backend”.

## Requirements
- Keep the default Hugging Face cache as the only snapshot location for Hugging Face downloads.
- Keep base package imports lightweight and all heavy runtimes behind explicit extras.
- Preserve the current `VisionManager` public contract and artifact-first outputs.
- Add explicit capability gating and clear runtime errors when local video is unavailable.
- Start with a concrete local `text_to_video` path before broadening to `image_to_video`.
- Do not regress current local image-edit support in CLI, playground, or AbstractCore.
- Make surface exposure honest:
  - expose only tasks that a selected backend/model can really run;
  - keep unsupported tasks visibly unsupported instead of silently routed elsewhere.

## Suggested implementation

### Phase 1: make one local video path real
- Add a first Diffusers local `text_to_video` implementation for the CogVideoX family.
- Add or normalize a curated registry/catalog entry for `zai-org/CogVideoX-2b`.
- Keep backend-owned model/runtime specifics inside the Diffusers backend rather than the manager.

### Phase 2: expose the runnable path
- Add a local CLI path for `text_to_video` once the backend works.
- Extend playground model discovery and UI so video-capable models can appear without breaking the
  current image-only flow.
- Re-check AbstractCore tool exposure so advertised video tools match runtime truth.

### Phase 3: decide whether to include local `image_to_video`
- Evaluate whether CogVideoX-style or another family gives a clean first local `image_to_video`
  implementation.
- Defer `image_to_video` if it adds too much runtime-specific complexity to the first milestone.

## Scope
Included:

- first local Diffusers `text_to_video` execution path;
- catalog/registry alignment for at least one concrete local video model;
- CLI/playground/AbstractCore exposure review tied to actual backend support;
- explicit validation that existing image-edit paths remain usable and correctly exposed.

## Non-goals
- Create a new `abstractvideo` package.
- Implement every local video family in one pass.
- Promise backend-agnostic local video parity across Diffusers, MFLUX, and `stable-diffusion.cpp`.
- Add timeline editing, audio handling, muxing, or long-form video workflow features.
- Move Hugging Face downloads into package-specific cache directories.

## Dependencies and related tasks
- Completed: [006_generative_vision_abstraction_api_and_backends.md](../completed/006_openai_compatible_backend_for_image_and_video.md)
- Completed: [007_local_hf_backend_strategy_diffusers.md](../completed/007_local_hf_backend_strategy_diffusers.md)
- Completed: [011_abstractcore_tool_integration_and_artifact_refs.md](../completed/011_abstractcore_tool_integration_and_artifact_refs.md)
- Completed: [016_abstractcore_plugin_catalog_discovery_surface.md](../completed/016_abstractcore_plugin_catalog_discovery_surface.md)
- Planned: [017_mlx_mflux_backend_strategy.md](../planned/017_mlx_mflux_backend_strategy.md)
- Planned: [020_adapter_aware_model_graph_and_catalog.md](../planned/020_adapter_aware_model_graph_and_catalog.md)

## Expected outcomes
- AbstractVision can run at least one real local `text_to_video` model through Diffusers.
- The first local video model is discoverable honestly in package catalog surfaces.
- CLI and playground stop being image-only where the backend/model truth supports video.
- AbstractCore video tools match what the package can really execute.
- Existing image-edit paths remain available with at least one cached/loadable local model on Apple
  Silicon.

## Validation
- Confirm `zai-org/CogVideoX-2b` is downloaded into the default Hugging Face cache and can be
  resolved offline after download.
- Run one small local `text_to_video` smoke test through the package backend.
- Verify CLI exposure and task gating for video-capable versus image-only models.
- Verify playground video-capable model listing and basic request flow once implemented.
- Re-run targeted tests covering:
  - manager capability gating
  - AbstractCore tool exposure
  - playground model/task listing
  - Diffusers backend task support

## Progress checklist
- [x] Normalize the first local video test target in registry/download surfaces.
- [x] Implement local Diffusers `text_to_video` for the first target family.
- [x] Add CLI support or explicit CLI deferral with honest messaging.
- [x] Extend playground discovery/UI beyond image-only filtering.
- [x] Re-check AbstractCore tool/runtime truth for video.
- [x] Validate that local image-edit exposure still works end to end.

## Guidance for the implementing agent
Re-check current code before implementation. Keep the first milestone narrow, package-owned, and
honest. Prefer one real local video path over a broad but misleading compatibility surface.

## Completion report

Date: 2026-05-21

### Summary

- Implemented the first local Diffusers `text_to_video` path through `zai-org/CogVideoX-2b` / `THUDM/CogVideoX-2b`.
- Kept the implementation inside the existing backend-owned task model: the manager stays thin, request normalization stays backend-local, and no new Python dependency was added.
- Exposed the runnable path consistently across the one-shot CLI, REPL, playground, and AbstractCore plugin surfaces.
- Preserved current image-edit support and verified that the local edit-capable models already cached on this machine remain usable.

### Files and symbols touched

- `src/abstractvision/backends/base_backend.py`
  - added default video normalization/progress hooks
- `src/abstractvision/vision_manager.py`
  - `generate_video(...)`
  - `image_to_video(...)`
- `src/abstractvision/backends/huggingface_diffusers.py`
  - `HuggingFaceDiffusersVisionBackend.preload()`
  - `HuggingFaceDiffusersVisionBackend.get_capabilities()`
  - `HuggingFaceDiffusersVisionBackend.generate_video_with_progress()`
  - `HuggingFaceDiffusersVisionBackend.generate_video()`
  - `_move_pipe_to_device(...)`
  - `_frames_to_mp4_bytes(...)`
  - `_supported_task_names(...)`
  - `normalize_video_generation_request(...)`
- `src/abstractvision/assets/vision_model_capabilities.json`
  - added `zai-org/CogVideoX-2b`
- `src/abstractvision/cli.py`
  - one-shot `t2v`
  - REPL `/t2v`
  - cache-backed Diffusers model-id normalization fix
- `src/abstractvision/playground_server.py`
  - `start_video_generation_job(...)`
  - backend/task-aware local model surfacing
- `src/abstractvision/playground/vision_playground.html`
  - Text → Video panel
- `src/abstractvision/integrations/abstractcore_plugin.py`
  - video residency aliases
  - request-scoped local `t2v` / `i2v` routing and loaded-model tracking
- `tests/test_huggingface_diffusers_backend.py`
- `tests/test_cli_smoke.py`
- `tests/test_playground_server.py`
- `tests/test_abstractcore_plugin.py`
- `tests/test_manager_capability_checks.py`
- `tests/test_vision_model_capabilities.py`
- `README.md`
- `docs/README.md`
- `docs/reference/backends.md`
- `docs/reference/configuration.md`
- `docs/reference/abstractcore-integration.md`
- `docs/api.md`
- `docs/getting-started.md`
- `docs/faq.md`
- `docs/architecture.md`
- `playground/README.md`
- `llms.txt`
- `CHANGELOG.md`

### Behavior changes

- Diffusers now advertises and executes local `text_to_video` only for the supported CogVideoX-2b family instead of exposing a broad but false video promise.
- Local video outputs are returned as MP4 artifacts/bytes using the same artifact-first contract as images.
- Apple Silicon CogVideoX loads now move to MPS with explicit `float16`; the new video path does not auto-promote above 16-bit on MPS.
- Playground model discovery is no longer image-only; it surfaces `text_to_video` models when the selected backend/model can really execute them.

### Validation

- Targeted regression suite:
  - `python -m pytest tests/test_huggingface_diffusers_backend.py tests/test_cli_smoke.py tests/test_playground_server.py tests/test_abstractcore_plugin.py tests/test_manager_capability_checks.py tests/test_vision_model_capabilities.py -q`
  - Result: `124 passed`
- Real local Diffusers CLI smoke on Apple Silicon:
  - `PYTHONPATH=src python -m abstractvision.cli t2v --provider diffusers --model zai-org/CogVideoX-2b --diffusers-device mps --diffusers-torch-dtype float16 --num-frames 9 --steps 1 --fps 8 "a red fox walking through a snowy forest, cinematic"`
  - Result: generated a local MP4 artifact through the package path.
- Real REPL smoke:
  - `/backend diffusers zai-org/CogVideoX-2b mps float16`
  - `/t2v ... --steps 1 --num-frames 9 --fps 8`
  - Result: generated a local MP4 artifact through the interactive path.
- Real playground/API smoke:
  - `GET /v1/vision/models` surfaced `zai-org/CogVideoX-2b` with `text_to_video`
  - `POST /v1/vision/jobs/videos/generations` succeeded with a real MP4 payload
- Real AbstractCore plugin smoke:
  - `_AbstractVisionCapability.t2v(... provider="diffusers", model="zai-org/CogVideoX-2b")`
  - Result: returned MP4 bytes and tracked/unloaded the request-warm loaded model correctly.

### Residual risks and follow-ups

- Local `image_to_video` is still intentionally deferred. The public API remains, but the first shipped local runtime milestone is `text_to_video` only.
- Local MP4 packaging depends on an external `ffmpeg` binary being present on `PATH`. This is documented and surfaces as a clear runtime error when missing.
- The first local Diffusers video surface is intentionally narrow. Expanding beyond the CogVideoX-2b family should come from fresh runtime evidence, not by widening backend claims speculatively.
