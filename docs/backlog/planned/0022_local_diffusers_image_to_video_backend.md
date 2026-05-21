# Planned: First local Diffusers image-to-video backend and surface integration

## Metadata
- Created: 2026-05-21
- Status: Planned
- Priority: P1
- Completed: N/A

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
AbstractVision now has one real local Diffusers `text_to_video` path through CogVideoX-2b. The
next missing local video milestone is `image_to_video`, because the public API, registry, and
AbstractCore surfaces already acknowledge it, but the local Diffusers backend still rejects it.

The immediate operator goal should stay narrow:

- add one honest local `image_to_video` path without new Python dependencies;
- keep the same package boundary (`abstractvision`, not `abstractvideo`);
- expose the runnable path consistently through CLI, playground, and AbstractCore only when the
  backend/model truth is real.

## Current code reality
Files and symbols re-checked on 2026-05-21:

- `src/abstractvision/backends/huggingface_diffusers.py`
- `src/abstractvision/vision_manager.py`
- `src/abstractvision/cli.py`
- `src/abstractvision/playground_server.py`
- `src/abstractvision/playground/vision_playground.html`
- `src/abstractvision/integrations/abstractcore.py`
- `src/abstractvision/integrations/abstractcore_plugin.py`
- `src/abstractvision/assets/vision_model_capabilities.json`
- `tests/test_huggingface_diffusers_backend.py`
- `tests/test_playground_server.py`
- `tests/test_abstractcore_plugin.py`

What already exists:

- `VisionManager.image_to_video(...)` is part of the stable package contract.
- The OpenAI-compatible backend already supports remote `image_to_video` when endpoints are
  configured.
- The AbstractCore plugin already exposes `i2v(...)` and normalizes residency task aliases for
  `image_to_video`.
- The packaged registry already carries several `image_to_video` candidates, including:
  - `zai-org/CogVideoX1.5-5B-I2V`
  - `Lightricks/LTX-2`
  - `tencent/HunyuanVideo-1.5`
  - `Wan-AI/Wan2.2-I2V-A14B`
  - `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- The installed Diffusers runtime on this machine already exposes official pipeline classes such as
  `CogVideoXImageToVideoPipeline`, `WanImageToVideoPipeline`, and
  `HunyuanVideoImageToVideoPipeline`.
- Local MP4 packaging is already solved in-package for video through the current external `ffmpeg`
  binary path.

What is still missing or mismatched:

- `HuggingFaceDiffusersVisionBackend._supports_local_image_to_video(...)` hard-codes `False`.
- `HuggingFaceDiffusersVisionBackend.image_to_video(...)` still raises
  `CapabilityNotSupportedError`.
- There is no one-shot CLI `i2v` command and no REPL `/i2v` command.
- The playground has no image-to-video endpoint or UI panel.
- No local `image_to_video` model is currently downloaded on this machine.

## Problem
The package now has a real local `text_to_video` story but still has no local `image_to_video`
path, so the public surface is still only half true for local video generation.

## What we want to do
Ship one honest, narrow, local Diffusers `image_to_video` path first, then expose it across the
interactive and plugin surfaces that can really support it.

## Why
- It completes the first practical local video milestone for the existing package boundary.
- It reuses the abstractions that were just proven by local `text_to_video` without introducing a
  separate package or dependency stack.
- It makes the plugin/playground/CLI video story consistent instead of having one local video task
  implemented and the other still remote-only.

## Requirements
- Do not add new Python dependencies.
- Keep Hugging Face downloads in the default Hugging Face cache.
- Keep backend-owned task semantics and request normalization.
- Keep Apple Silicon support explicit and defensible: stay at 16-bit for the local Diffusers video
  path unless a model family proves otherwise.
- Preserve the current artifact-first output contract and reuse the existing MP4 packaging path.
- Expose only models/tasks that the selected backend can really execute.

## Suggested implementation

### Phase 1: choose one first local I2V target
- Prefer `zai-org/CogVideoX1.5-5B-I2V` as the first candidate:
  - official Diffusers pipeline class already exists;
  - it stays in the CogVideoX family already used for the first local T2V path;
  - it requires no new Python dependency stack.
- Keep Wan/Hunyuan/LTX as later follow-ups unless runtime evidence shows a smaller or cleaner first
  path.

### Phase 2: implement backend truth
- Add backend gating for the first supported local `image_to_video` model family only.
- Add backend-owned normalization for `ImageToVideoRequest`, including width/height/frame defaults
  and unsupported-parameter filtering.
- Keep preload/warmup and unload semantics aligned with the current local video implementation.

### Phase 3: expose the runnable path
- Add one-shot CLI `i2v` and REPL `/i2v`.
- Extend playground discovery and UI with an image-to-video flow only for supported models.
- Keep AbstractCore plugin/runtime truth aligned with the backend support.

## Scope
Included:

- one real local Diffusers `image_to_video` path;
- backend/catalog/task-gating alignment for the first chosen model family;
- CLI/playground/AbstractCore exposure tied to real backend capability;
- validation that current image-edit and text-to-video behavior do not regress.

## Non-goals
- Broad local `image_to_video` parity across every video model family in one pass.
- Audio handling, timeline editing, muxing, or long-form video orchestration.
- A new package split such as `abstractvideo`.
- New runtime SDKs or helper libraries beyond what is already in the repo/runtime stack.

## Dependencies and related tasks
- Completed: [0021_local_diffusers_text_to_video_backend.md](../completed/0021_local_diffusers_text_to_video_backend.md)
- Planned: [017_mlx_mflux_backend_strategy.md](017_mlx_mflux_backend_strategy.md)
- Planned: [020_adapter_aware_model_graph_and_catalog.md](020_adapter_aware_model_graph_and_catalog.md)

## Expected outcomes
- AbstractVision can run at least one real local Diffusers `image_to_video` model.
- CLI, playground, and AbstractCore can expose that path honestly.
- The packaged registry and backend support no longer disagree for the first chosen local I2V
  model.
- Current local image-edit and local text-to-video behavior remain intact.

## Validation
- Download the chosen first I2V model into the default Hugging Face cache and resolve it offline.
- Run one small local `image_to_video` smoke through the package backend.
- Verify CLI, REPL, playground, and AbstractCore request routing for the supported I2V model.
- Re-run the relevant backend, manager, playground, plugin, and catalog tests.

## Progress checklist
- [ ] Confirm the first local I2V target and document why it is first.
- [ ] Implement local Diffusers `image_to_video` for that target family.
- [ ] Add CLI/REPL `i2v` support.
- [ ] Add playground I2V discovery/UI.
- [ ] Re-check AbstractCore plugin/runtime truth for `image_to_video`.
- [ ] Validate no regressions for local image-edit and local T2V.

## Guidance for the implementing agent
Re-check current runtime truth before implementation. Keep the first local I2V milestone narrow,
package-owned, and honest. If the chosen target cannot be made to work cleanly without new
dependencies or unacceptable runtime behavior, update this backlog item with the evidence and stop
instead of widening backend claims.
