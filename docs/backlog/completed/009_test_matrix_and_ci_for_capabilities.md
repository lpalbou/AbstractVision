## Task 009: Test matrix + CI for capability coverage

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P1  

---

## Main goals

- Ensure the capability registry and abstractions do not drift by enforcing:
  - every “seed model” exists in `vision_model_capabilities.json`
  - every task has at least one model supporting it
  - required fields and references are valid (via Task 004 validator)

## Secondary goals

- Add optional integration tests (gated) for environments that have heavy deps or remote endpoints.

---

## Context / problem

Capabilities JSON is the foundation. A small regression (renamed model id, missing task mapping) silently breaks routing. We need a minimal but strict test suite for correctness.

---

## Constraints

- Base CI must not require heavy ML dependencies.

---

## Research, options, and references

Use unit tests for schema + registry checks, and keep integration tests optional (skipped by default).

---

## Decision

**Chosen approach**:
- Unit tests for:
  - registry loading
  - expected tasks per seed model
  - schema validation
- Optional integration tests:
  - `OPENAI_COMPAT_BASE_URL` + `OPENAI_COMPAT_API_KEY` env-gated
  - `abstractvision[huggingface]` env-gated

---

## Dependencies

- **Backlog tasks**:
  - Completed: `docs/backlog/completed/004_capability_schema_and_validation.md`
  - Completed: `docs/backlog/completed/003_hf_model_landscape_and_capability_registry.md`

---

## Implementation plan

- Expand unit tests to assert:
  - all task keys exist and are reachable
  - LoRA `base_model_id` references exist
  - every top-level task has at least one supporting model
  - validator error messages point to a model/task field path
- Add a minimal CI command recommendation (unittest/pytest).

---

## Success criteria

- Any accidental drift in the JSON is caught before merge.

---

## Test plan

- `python -m unittest discover -s abstractvision/tests -p "test_*.py" -q`

---

## Report (fill only when completed)

### Summary

- Added drift-protection tests that ensure:
  - the registry schema version is present
  - the expected task keys exist
  - every task has at least one supporting model
  - every model declares at least one task
- These checks complement the JSON schema validator by catching “registry coverage” regressions early.

### Validation

- Tests: `python -m unittest discover -s abstractvision/tests -p "test_*.py" -q`
- New tests live in `abstractvision/tests/test_capability_registry_coverage.py`.
