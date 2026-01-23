## Task 010: README + examples for capability-driven model selection

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P1  

---

## Main goals

- Provide clear documentation and examples showing how users should:
  - inspect model capabilities
  - select a model for a task (T2I/I2I/T2V/I2V)
  - call the stable AbstractVision API
  - consume outputs as artifact refs (third-party friendly)
  - (optional) register and use the official AbstractCore tools

## Secondary goals

- Keep docs focused on “capabilities + abstraction” (not policy/endorsement).

---

## Context / problem

Without examples, third parties will guess how to use the system and may bypass the capability registry (causing runtime errors). We need to demonstrate the expected usage pattern: capability-driven selection.

---

## Constraints

- Docs must not require heavy dependencies.

---

## Research, options, and references

Use minimal examples that demonstrate capability checks and backend wiring patterns.

---

## Decision

**Chosen approach**:
- Update `README.md` with:
  - tasks supported
  - how to query `VisionModelCapabilitiesRegistry`
  - how to decide which model can do which task

---

## Dependencies

- **Backlog tasks**:
  - Completed: `docs/backlog/completed/003_hf_model_landscape_and_capability_registry.md`
  - Completed: `docs/backlog/completed/005_core_api_tasks_and_abstractions.md`
  - Completed: `docs/backlog/completed/008_asset_outputs_and_third_party_integration.md`
  - Completed: `docs/backlog/completed/011_abstractcore_tool_integration_and_artifact_refs.md`
  - Completed: `docs/backlog/completed/013_cli_repl_for_manual_testing.md`

---

## Implementation plan

- Add examples for:
  - list models for task (e.g. `models_for_task("text_to_video")`)
  - assert support
  - instantiate `VisionManager` with a backend (stubbed example)
  - store outputs and return artifact refs (Task 008 contract)
  - AbstractCore tool wiring (Task 011) as the “golden path” integration snippet

---

## Success criteria

- A new user can understand the model selection and capability flow without reading internal code.

---

## Test plan

- Manual doc sanity.

---

## Report (fill only when completed)

### Summary

- Updated `abstractvision/README.md` to document the intended “capability-driven selection” workflow end-to-end.
- Added copy-pastable examples for:
  - discovering models/tasks via `VisionModelCapabilitiesRegistry`
  - wiring an OpenAI-compatible backend + `VisionManager`
  - artifact-ref outputs via `LocalAssetStore`
  - interactive testing via `abstractvision repl`
  - AbstractCore tool wiring via `make_vision_tools(...)`

### Validation

- Tests: `python -m unittest discover -s abstractvision/tests -p "test_*.py" -q`
- Manual: README snippet sanity (imports + API names match the current codebase).
