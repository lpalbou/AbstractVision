## Task 005: Core API tasks + abstraction contract (model-agnostic)

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Main goals

- Define the stable, model-agnostic API contract for generative vision:
  - Text → Image (T2I)
  - Image → Image (I2I)
  - Text → Video (T2V)
  - Image → Video (I2V)
  - Multi-view image generation (optional/advanced)
- Ensure the contract maps cleanly onto:
  - `vision_model_capabilities.json` task names and parameter schema
  - backend adapters (local or remote)

## Secondary goals

- Ensure the code layout follows `src/` packaging and “small focused files”.

---

## Context / problem

We need a stable API for users/third parties to call generative vision tasks without binding to:
- a specific model family (diffusers vs turbo vs remote API)
- a specific runtime (GPU vs CPU)
- a specific transport (local vs remote endpoint)

This is the “port” side of the ports/adapters architecture.

---

## Constraints

- The public contract must be driven by **tasks** and **parameters**, not by model internals.
- Keep method names and task names consistent with capabilities JSON.

---

## Research, options, and references

- Option A: a single `generate(task, **kwargs)` method
  - Minimal API, but invites backend-specific kwargs and weak typing.
- Option B (chosen): explicit methods + typed request objects + capability checks
  - Better long-term stability and clarity.

---

## Decision

**Chosen approach**:
- Provide a small number of task methods on `VisionManager`:
  - `generate_image(prompt, ...)`
  - `edit_image(prompt, image, ...)`
  - `generate_video(prompt, ...)`
  - `image_to_video(image, ...)`
  - `generate_angles(prompt, ...)` (optional)
- Backends implement `VisionBackend` with corresponding methods.

---

## Dependencies

- Depends on: `docs/backlog/completed/003_hf_model_landscape_and_capability_registry.md`
- Depends on: `docs/backlog/completed/004_capability_schema_and_validation.md`

---

## Implementation plan

- Ensure the `VisionBackend` interface covers I2V.
- Ensure task names used by registry are consistent with JSON.
- Keep request objects small and forward-compatible (extra params bag).

---

## Success criteria

- A user can select a model id, query capabilities, and decide which task calls are valid.
- Adding new models does not require API changes; only JSON updates (and backend support if new task types appear).

---

## Test plan

- Unit tests: registry + API imports work without heavy dependencies installed.

---

## Report (fill only when completed)

### Summary

- Implemented the minimal model-agnostic “ports” surface:
  - `VisionManager` orchestrator at `src/abstractvision/vision_manager.py`
  - request/response dataclasses at `src/abstractvision/types.py`
  - backend interface at `src/abstractvision/backends/base_backend.py`
- Kept the base install dependency-light (no heavy ML deps required to import and use the contract surface).

### Validation

- Tests: `python -m unittest discover -s tests -p "test_*.py" -q`
