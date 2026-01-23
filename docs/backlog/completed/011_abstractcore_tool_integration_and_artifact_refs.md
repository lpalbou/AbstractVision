## Task 011: AbstractCore tool integration (vision tools returning artifact refs)

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Main goals

- Provide a first-class integration path so `abstractcore` users can call **generative vision** as normal tools:
  - T2I: `text_to_image`
  - I2I: `image_to_image`
  - T2V: `text_to_video`
  - I2V: `image_to_video`
  - (optional) multi-view image
- Ensure every tool returns a **small JSON artifact reference** (not raw bytes) so results can flow through:
  - AbstractCore tool calling
  - AbstractRuntime durable runs (vars remain JSON-serializable)
  - AbstractFlow workflows
  - AbstractCode UI / third-party UIs

## Secondary goals

- Keep the integration optional: `import abstractvision` must not require `abstractcore`.
- Provide a minimal “tool factory” that third parties can reuse (not tied to AbstractCode).

---

## Context / problem

Generative media outputs are large. For a good end-user and third-party experience, we want a “golden path” where:
- an agent/workflow can request an image/video,
- the output is stored durably,
- downstream components receive a small ref object they can display/open/forward.

The AbstractFramework already uses **artifact references** (`{"$artifact": "..."}`) for large payloads (AbstractRuntime).
AbstractVision should integrate with that model directly.

---

## Constraints

- No hard dependency on `abstractcore` in the base install.
- Tool outputs must be JSON-serializable and stable (avoid backend-native objects).
- No implicit model downloads in tools; any heavy dependencies remain behind explicit extras.

---

## Research, options, and references

- **Option A**: Expose only a Python API (`VisionManager`) and let each app wrap it into tools.
  - Pros: minimal surface.
  - Cons: every integrator reinvents tool schemas and output shapes; inconsistent third-party UX.
- **Option B (chosen)**: Ship an official `abstractvision.integrations.abstractcore` module that provides tool wrappers.
  - Pros: one canonical tool schema + one canonical artifact output shape; simplest adoption.

References:
- AbstractRuntime artifact refs: `abstractruntime/src/abstractruntime/storage/artifacts.py`
- AbstractRuntime tool execution: `abstractruntime/src/abstractruntime/integrations/abstractcore/tool_executor.py`

---

## Decision

**Chosen approach**:
- Add `abstractvision.integrations.abstractcore` that exposes a factory such as:
  - `make_vision_tools(*, vision_manager, store, model_id, registry=None) -> list[callable]`
- Each tool:
  - validates the requested task against the capabilities registry (best-effort)
  - calls the configured backend via `VisionManager`
  - stores the output via the store (local store or runtime artifact store adapter)
  - returns a `GeneratedMediaRef` containing at least `"$artifact"` and `content_type`

**Why**:
- This becomes the clean, intuitive, “one import” integration point for end users.
- Tool outputs are portable across apps and UIs because they use the framework artifact ref contract.

---

## Dependencies

- **Backlog tasks**:
  - Completed: `docs/backlog/completed/005_core_api_tasks_and_abstractions.md`
  - Completed: `docs/backlog/completed/008_asset_outputs_and_third_party_integration.md`
  - Completed: `docs/backlog/completed/004_capability_schema_and_validation.md` (prefer early validation errors)

---

## Implementation plan

- Implement the integration module with minimal surface area:
  - tool names, docstrings, and JSON schema designed for third parties
  - tools created as closures capturing `(vision_manager, store, model_id, registry)`
- Define a stable return shape (documented in Task 008):
  - always include `"$artifact"`
  - include `content_type`, `sha256`, `filename` when available
- Provide example wiring in docs (Task 010):
  - register tools in an AbstractCore agent/session and call them
- Add unit tests with:
  - a fake backend that returns deterministic bytes
  - a temp local store
  - assertions that tool outputs are small JSON and include `$artifact`

---

## Success criteria

- A third party can register the tools and receive artifact refs without writing glue code.
- Tool outputs are compatible with AbstractRuntime artifact references and safe to persist in run state.

---

## Test plan

- `python -m unittest discover -s abstractvision/tests -p "test_*.py" -q`

---

## Report (fill only when completed)

### Summary

- Added an optional AbstractCore integration module:
  - `abstractvision/src/abstractvision/integrations/abstractcore.py`
- Implemented `make_vision_tools(...)` which returns `@abstractcore.tool`-decorated callables:
  - `vision_text_to_image`
  - `vision_image_to_image`
  - `vision_multi_view_image`
  - `vision_text_to_video`
  - `vision_image_to_video`
- All tools:
  - validate support via `VisionModelCapabilitiesRegistry.require_support(model_id, task)`
  - require `VisionManager.store` so outputs are always returned as artifact refs (`{"$artifact": ...}`)
  - accept image inputs either as artifact refs (`image_artifact`) or base64 (`image_b64`) for compatibility.

Tool argument conventions (v0):
- Image inputs:
  - preferred: `image_artifact={"$artifact": "..."}` (no base64 in tool calls)
  - fallback: `image_b64="..."` (raw base64 or data URL)
- Mask inputs:
  - `mask_artifact` / `mask_b64` (optional)
- Multi-view reference image:
  - `reference_image_artifact` / `reference_image_b64` (optional)

Unit tests:
- `abstractvision/tests/test_abstractcore_tool_integration.py`

### Validation

- Tests:
  - `python -m unittest discover -s abstractvision/tests -p "test_*.py" -q`
