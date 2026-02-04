## Task 006: OpenAI-compatible backend adapter for image + video generation

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P1  

---

## Main goals

- Implement a backend adapter that can call an **OpenAI-compatible endpoint** for:
  - text→image
  - image→image (edits)
  - text→video (if exposed by the endpoint)
  - image→video (if exposed by the endpoint)

## Secondary goals

- Keep the backend optional (no hard dependency in base install).
- Normalize outputs to AbstractVision’s artifact-ref output contract (Task 008), with an optional bytes escape hatch.

---

## Context / problem

Local inference for video models is often heavy and hardware-dependent. Many deployments will expose an internal OpenAI-compatible service for image/video generation. AbstractVision should be able to call such endpoints without binding to a specific vendor SDK.

---

## Constraints

- Must route calls based on `vision_model_capabilities.json` tasks + params.
- Must not assume a universal “video generation” spec exists in OpenAI-compatible servers:
  - implement image endpoints first (widely compatible),
  - treat video as optional/configurable extensions.

---

## Research, options, and references

- Option A: Depend on an OpenAI SDK.
  - Ties to a vendor client library and drift.
- Option B (chosen): Use plain HTTP (e.g. `httpx`) behind an optional dependency.

---

## Decision

**Chosen approach**:
- Add `OpenAICompatibleVisionBackend` that:
  - takes `base_url`, `api_key`, `model_id`
  - calls the appropriate endpoint based on the requested task (images first; video optional)
  - validates task support via the capabilities registry before calling (Task 004 makes this reliable)
  - stores outputs and returns artifact refs (Task 008)

---

## Dependencies

- **Backlog tasks**:
  - Completed: `docs/backlog/completed/003_hf_model_landscape_and_capability_registry.md`
  - Completed: `docs/backlog/completed/005_core_api_tasks_and_abstractions.md`
  - Completed: `docs/backlog/completed/004_capability_schema_and_validation.md`
  - Completed: `docs/backlog/completed/008_asset_outputs_and_third_party_integration.md`

---

## Implementation plan

- Implement image generation/edit mapping first:
  - T2I (`/v1/images/generations`-style)
  - I2I (`/v1/images/edits`-style) (mask optional)
- Add an explicit configuration mechanism for “video endpoints” (optional):
  - only enable if user provides paths and the capability registry declares support
- Normalize responses:
  - accept base64-in-JSON or direct bytes, depending on server behavior
  - store bytes via the configured store and return a `GeneratedMediaRef` (Task 008)
- Add unit tests using mocked HTTP responses (no network).

---

## Success criteria

- Backend can be configured with a model id and refuses unsupported tasks before making HTTP calls.
- Backend produces normalized outputs.

---

## Test plan

- Unit tests with http mocking (no network).

---

## Report (fill only when completed)

### Summary

- Implemented an OpenAI-shaped HTTP backend using stdlib `urllib` (no third-party deps):
  - `src/abstractvision/backends/openai_compatible.py`
  - `OpenAICompatibleBackendConfig` (endpoints + auth + timeouts)
  - `OpenAICompatibleVisionBackend`:
    - T2I: `POST /images/generations` (JSON; prefers `b64_json`)
    - I2I: `POST /images/edits` (multipart; supports optional `mask`)
    - T2V/I2V: optional/custom endpoints (`text_to_video_path`, `image_to_video_path`)
- Added unit tests with mocked HTTP responses (no network):
  - `tests/test_openai_compatible_backend.py`

Notes:
- This backend returns `GeneratedAsset` bytes (by design of the backend interface). Artifact-ref storage and tagging
  happen in `VisionManager` when a `store` is configured (Task 008).

### Validation

- Tests:
  - `python -m unittest discover -s tests -p "test_*.py" -q`
