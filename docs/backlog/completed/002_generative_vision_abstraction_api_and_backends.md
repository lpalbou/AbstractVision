## Task 002: Generative vision abstraction (capability-driven contract + backend capabilities)

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Main goals

- Make the public `abstractvision` contract **capability-driven and safe-by-default**:
  - validate requested tasks/params early (consistent errors),
  - avoid backend-specific failures whenever possible.
- Standardize backend capability reporting so routers/UIs can introspect without model-specific code.
- Ensure the abstraction composes with tool calling and workflows by aligning outputs with **artifact refs** (Task 008).

## Secondary goals

- Keep the base install dependency-light (no torch/diffusers/httpx required just to import the contract).
- Keep the `VisionManager` surface stable and intuitive for end users (Task 005).

---

## Context / problem

We already have:
- a minimal, model-agnostic “ports” surface (`VisionManager`, request/response dataclasses) (Task 005),
- a model capability registry (`vision_model_capabilities.json`) (Task 003).

What’s missing for a clean end-user and third-party experience:
- `VisionManager` is currently a thin delegator and does **not** consult capabilities → invalid calls fail late and backend-specific.
- There is no standard `VisionBackendCapabilities` for runtime constraints (mask support, min/max resolution, video fps/frames, etc.).
- Outputs are currently bytes-heavy (`GeneratedAsset.data`) and don’t flow cleanly through durable state/tool outputs.

---

## Constraints

- No hard dependency on `abstractcore`/`abstractruntime` in the base install.
- No model-specific behavior in the public contract (quirks belong in backends and capability metadata).
- Keep the public method surface stable (Task 005).

---

## Research, options, and references

- **Option A**: Keep `VisionManager` as a pure delegator; require users to pre-check capabilities manually.
  - Pros: minimal code.
  - Cons: poor UX; every integrator duplicates validation/routing and error handling.
- **Option B (chosen)**: Capability-driven manager + backend capabilities (ports/adapters done properly).
  - Pros: consistent UX, early actionable errors, easier third-party integration and routing.

References:
- Capability registry: `abstractvision/src/abstractvision/assets/vision_model_capabilities.json`
- Registry loader: `abstractvision/src/abstractvision/model_capabilities.py`

---

## Decision

**Chosen approach**:
- Add a `VisionBackendCapabilities` structure (backend-level constraints/features) and use it for request validation.
- Make `VisionManager` capability-aware:
  - validate the requested task against the model registry (Task 004 ensures schema correctness),
  - validate request parameters against backend constraints when available,
  - raise consistent `CapabilityNotSupportedError` / `UnknownModelError` before calling the backend when possible.
- Treat artifact refs as the preferred output path (Task 008).

**Why**:
- Users get early, consistent, actionable errors.
- Third-party apps can build on a stable contract (capabilities + refs) without importing backend-specific stacks.

---

## Dependencies

- **Backlog tasks**:
  - Completed: `docs/backlog/completed/003_hf_model_landscape_and_capability_registry.md`
  - Completed: `docs/backlog/completed/005_core_api_tasks_and_abstractions.md`
  - Completed: `docs/backlog/completed/004_capability_schema_and_validation.md`
  - Completed: `docs/backlog/completed/008_asset_outputs_and_third_party_integration.md`

---

## Implementation plan

- Define `VisionBackendCapabilities` (minimal, additive):
  - supported tasks (if backend is more limited than the model registry)
  - `supports_mask`, `max_width/height`, `max_fps`, `max_frames`, etc. (keep optional)
- Update `VisionManager` to support validation:
  - accept a `model_id` (stored on manager or passed per call)
  - accept an optional `VisionModelCapabilitiesRegistry` instance
  - perform best-effort validation before dispatch
- Add unit tests with a fake backend:
  - ensure invalid tasks/params raise consistent errors
  - ensure no heavy deps required for tests

---

## Success criteria

- Unsupported tasks fail fast with `CapabilityNotSupportedError` (before backend invocation).
- Invalid parameter combinations fail fast with a clear, stable error message.
- Importing and using the contract surface does not require heavy ML dependencies.

---

## Test plan

- `python -m unittest discover -s abstractvision/tests -p "test_*.py" -q`

---

## Report (fill only when completed)

### Summary

- Added backend-level capability reporting:
  - `VisionBackendCapabilities` in `abstractvision/src/abstractvision/types.py`
  - `VisionBackend.get_capabilities()` default method in `abstractvision/src/abstractvision/backends/base_backend.py`
- Made `VisionManager` capability-aware (best-effort):
  - optional `model_id` + `registry` on the manager
  - early model-level gating via `VisionModelCapabilitiesRegistry.require_support(...)`
  - early backend-level gating via `supported_tasks` and `supports_mask` when backends expose capabilities
  - consistent `CapabilityNotSupportedError` before backend invocation when possible

Notes:
- We only enforce `supports_mask` and `supported_tasks` right now. Resolution/FPS/frame constraints are represented
  in `VisionBackendCapabilities` but not yet enforced (safe additive follow-up).

Usage (v0):
- Model-level gating (optional):
  - `VisionManager(model_id="Qwen/Qwen-Image-2512")` will auto-load the registry and raise `CapabilityNotSupportedError`
    if you call an unsupported task.
- Backend-level gating (optional):
  - backends can override `get_capabilities()` to declare constraints like `supports_mask=False`.

Unit tests:
- `abstractvision/tests/test_manager_capability_checks.py`

### Validation

- Tests:
  - `python -m unittest discover -s abstractvision/tests -p "test_*.py" -q`
