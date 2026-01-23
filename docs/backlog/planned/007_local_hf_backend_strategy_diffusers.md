## Task 007: Local HuggingFace backend strategy (Diffusers/Transformers)

**Date**: 2026-01-23  
**Status**: Planned  
**Priority**: P1  

---

## Main goals

- Provide a local backend strategy that can execute supported tasks using HF ecosystems:
  - T2I / I2I via diffusers pipelines when available
  - (phase 2) T2V / I2V via diffusers pipelines when available
- Keep all heavy dependencies behind optional installation.

## Secondary goals

- Support “model family” configuration (Qwen/GLM/Z-Image/Wan/Hunyuan/Mochi/LTX) without changing the core API.

---

## Context / problem

Some users will want local inference. However, local video generation is high-variance in requirements. AbstractVision should still provide a consistent adapter interface and gracefully communicate missing dependencies or unsupported configurations.

---

## Constraints

- No hard dependency on torch/diffusers in base install.
- Capability support must be checked via `vision_model_capabilities.json` before calling.
- No implicit model downloads in library code; any prefetch/download must be explicit and documented.

---

## Research, options, and references

Use the seed model set in Task 003 as the initial coverage target.

---

## Decision

**Chosen approach**:
- Implement a `HuggingFaceLocalVisionBackend` with lazy imports and clear errors.
- Provide per-task pipeline selection internally (not in public API).

---

## Dependencies

- **Backlog tasks**:
  - Completed: `docs/backlog/completed/003_hf_model_landscape_and_capability_registry.md`
  - Completed: `docs/backlog/completed/005_core_api_tasks_and_abstractions.md`
  - Planned: `docs/backlog/planned/004_capability_schema_and_validation.md`
  - Planned: `docs/backlog/planned/008_asset_outputs_and_third_party_integration.md`

---

## Implementation plan

- Add optional extras: `abstractvision[huggingface]` with torch/diffusers/transformers (and explicit platform markers where needed).
- Phase 1 (images):
  - load pipeline based on task + model id
  - execute and store output via the configured store
  - return `GeneratedMediaRef` (artifact ref) instead of raw bytes by default
- Phase 2 (video):
  - treat as separate sub-scope with explicit constraints (ffmpeg/codecs, VRAM requirements, longer timeouts)
- Add smoke tests that only run when deps are installed.

---

## Success criteria

- Importing `abstractvision` remains light.
- With optional deps installed, a user can execute at least one model per task type locally.

---

## Test plan

- Pure unit tests for routing + capability checks (no heavy deps).
- Optional integration tests gated behind extras (skipped by default).

---

## Report (fill only when completed)

### Summary

TBD

### Validation

- Tests: TBD
