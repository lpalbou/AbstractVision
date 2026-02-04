## Task 004: Capability schema + validation for `vision_model_capabilities.json`

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Main goals

- Formalize the **capability schema** used by `src/abstractvision/assets/vision_model_capabilities.json`.
- Implement a validator that fails fast on invalid schemas and provides actionable error messages.

## Secondary goals

- Keep the schema **minimal and extensible** (additive evolution).
- Avoid backend-specific leakage: keep the schema about capabilities, not implementations.

---

## Context / problem

The capabilities JSON is the routing and UX foundation for AbstractVision. If it becomes inconsistent (missing task, wrong params, dangling references), the whole abstraction becomes unreliable.

We want a validator that works in CI and during development, so updates are safe and reviewable.

---

## Constraints

- Must be dependency-light (standard library only).
- Must support schema versioning (`schema_version`) and allow additive fields.

---

## Research, options, and references

- Option A: No formal schema; rely on tests only.
  - Too fragile as the model list grows.
- Option B (chosen): “soft schema” + explicit validation checks in code.
  - Keeps flexibility while maintaining correctness.

---

## Decision

**Chosen approach**:
- Implement a `validate_capabilities_json()` function in `src/abstractvision/model_capabilities.py` (or a dedicated validator module).
- Add tests to enforce validation passes and that seed models are present.

---

## Dependencies

- Depends on: `docs/backlog/completed/003_hf_model_landscape_and_capability_registry.md`

---

## Implementation plan

- Validate top-level keys:
  - `schema_version` (string or number; normalize to string in errors)
  - `tasks` (object)
  - `models` (object keyed by model_id)
- Validate each model entry:
  - `provider` (string)
  - `license` (string; informational)
  - `tasks` (object)
- Validate each task entry:
  - `inputs` list, `outputs` list
  - `params` object where each param has at least `required: bool` (allow additive fields)
  - `requires` object optional; if it contains `base_model_id`, ensure it exists in `models`
- Validate internal consistency:
  - every model task key exists in top-level `tasks`
  - every model `tasks[task].params` only references params defined for that task (or explicitly allowed as “extra”)
- Make failures actionable:
  - raise exceptions that point to the exact model/task/field path

---

## Success criteria

- Invalid JSON yields a deterministic, actionable exception.
- CI tests prevent regressions and drift.

---

## Test plan

- Run unit tests:
  - schema validation passes for the committed JSON
  - schema validation fails for a small set of intentionally malformed fixtures

---

## Report (fill only when completed)

### Summary

- Added a dependency-light validator `validate_capabilities_json()` and wired it into `VisionModelCapabilitiesRegistry` load.
- Validator enforces:
  - required top-level keys (`schema_version`, `tasks`, `models`)
  - model entries have required fields (`provider`, `license`, `tasks`)
  - model task entries reference known top-level tasks
  - `inputs`/`outputs` are lists of non-empty strings
  - `params.*.required` is boolean
  - `requires.base_model_id` references an existing model id

Implementation:
- Validator: `src/abstractvision/model_capabilities.py`
- Unit tests: `tests/test_capabilities_schema_validation.py`

Error format:
- Raises `ValueError` with a structured “path” prefix, e.g.:
  - `Invalid capability asset at models['...'].tasks['...'].params['...'].required: expected boolean`

Notes:
- We intentionally do **not** validate “param names vs canonical task param lists” yet, because the top-level `tasks`
  entries currently only carry `description`. If we later add optional `tasks[task].params` definitions, we can tighten
  this validation additively without breaking existing JSON.

### Validation

- Tests:
  - `python -m unittest discover -s tests -p "test_*.py" -q`
