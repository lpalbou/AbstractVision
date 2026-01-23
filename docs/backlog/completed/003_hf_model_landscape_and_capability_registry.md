## Task 003: HF model landscape + capability registry (single source of truth)

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Main goals

- Establish a repeatable process to **discover/track HuggingFace generative vision models** (image/video) and capture their capabilities in:
  - `src/abstractvision/assets/vision_model_capabilities.json`
- Define a small, stable **capability schema** that can describe what a model can do (tasks + parameters + constraints) without embedding backend-specific code.

## Secondary goals

- Make it easy to add new models later with a clear “how to update capabilities” checklist.
- Keep licensing information **informational** (metadata), not enforcement logic.

---

## Context / problem

AbstractVision’s job is to provide a clean abstraction and to describe model capabilities accurately. This requires a single source of truth that is:
- human-editable and reviewable (JSON)
- machine-consumable (registry + validation)
- stable over time (versioned schema)

We also need to support multiple families of models for the same task (T2I/I2I/T2V/I2V/multi-view), because best models change quickly and users choose their own.

---

## Constraints

- **Capabilities are descriptive**: we do not enforce whether a user should use a model.
- **Schema stability**: evolve schema via `schema_version` and additive changes.
- **No hardcoding**: capability logic must come from JSON, not scattered in code.

---

## Research, options, and references

**Model references (seed set provided by user):**
- T2I: `https://huggingface.co/Qwen/Qwen-Image-2512`
- I2I: `https://huggingface.co/Qwen/Qwen-Image-Edit-2511`
- Multi-view (LoRA): `https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`
- T2V: `https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B`
- T2V: `https://huggingface.co/tencent/HunyuanVideo-1.5`
- T2V: `https://huggingface.co/genmo/mochi-1-preview`
- T2I/I2I candidate: `https://huggingface.co/zai-org/GLM-Image`
- T2I turbo candidate: `https://huggingface.co/Tongyi-MAI/Z-Image-Turbo`
- I2V: `https://huggingface.co/Lightricks/LTX-2`

---

## Decision

**Chosen approach**:
- Maintain a versioned JSON file that enumerates:
  - model ids
  - task support (t2i, i2i, multi_view_image, t2v, i2v)
  - parameters and whether they are required/optional
  - requirements (e.g., “LoRA requires base model id”)
  - notes/constraints as structured metadata where possible

**Why**:
- Avoids coupling the public API to any one model or backend.
- Enables deterministic capability checks for routing and user UX.

---

## Dependencies

- **Backlog tasks**:
  - Planned: `docs/backlog/planned/004_capability_schema_and_validation.md`

---

## Implementation plan

- Define the capability schema rules in a short doc section (inside this task or an ADR later).
- Ensure `vision_model_capabilities.json` includes:
  - All user-provided models
  - Clear supported tasks per model
  - Parameter surface for each task
- Add/extend a code validator that checks:
  - schema shape
  - all tasks referenced exist in `tasks`
  - required fields present
  - referenced `base_model_id` exists in the same file

---

## Success criteria

- `vision_model_capabilities.json` is complete for the seed set.
- A registry can answer:
  - “does model X support task Y?”
  - “which models support task Y?”

---

## Test plan

- Add automated tests that assert:
  - every seed model id exists in the JSON
  - expected core tasks are present and mapped correctly

---

## Report (fill only when completed)

### Summary

- Implemented the capability “single source of truth” at `abstractvision/src/abstractvision/assets/vision_model_capabilities.json` (seed models + tasks).
- Added a small query registry (`VisionModelCapabilitiesRegistry`) at `abstractvision/src/abstractvision/model_capabilities.py`.
- Added basic unit tests to prevent drift at `abstractvision/tests/test_vision_model_capabilities.py`.

### Validation

- Tests: `python -m unittest discover -s abstractvision/tests -p "test_*.py" -q`
