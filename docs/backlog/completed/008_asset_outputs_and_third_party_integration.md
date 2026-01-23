## Task 008: Artifact outputs (refs) for third-party integration

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Main goals

- Define a stable, JSON-serializable output contract for generated media (image/video) that scales beyond “return bytes”.
- Make outputs plug-and-play with the AbstractFramework ecosystem (AbstractRuntime artifacts, AbstractCore tool calling, AbstractCode UI).

## Secondary goals

- Provide a default local store for standalone users, plus an adapter for AbstractRuntime’s `ArtifactStore` when present.
- Keep a bytes escape hatch for advanced users/tests.

---

## Context / problem

Generated images/videos are large. Returning raw bytes in tool calls, workflow state, or API responses is often impractical and breaks interoperability.

The AbstractFramework already has an established pattern for large payloads: **artifact references** via `{"$artifact": "<id>"}` (AbstractRuntime).

AbstractVision should align with that contract so third-party systems can:
- store outputs in a durable artifact store,
- pass outputs through tool calls as small JSON,
- let UIs render/open the media without embedding blobs.

---

## Constraints

- No hard dependency on `abstractruntime` (but must be compatible with its artifact-ref shape).
- Base install must remain dependency-light (no heavy deps to *import* the contract).

---

## Research, options, and references

- **Option A**: Introduce a custom `AssetRef` contract unrelated to AbstractRuntime.
  - Pros: standalone.
  - Cons: diverges from the framework’s existing artifact reference contract; harder to integrate with tools/workflows/UIs.
- **Option B (chosen)**: Adopt the AbstractRuntime artifact reference shape and enrich it.
  - Keep `{"$artifact": "<id>"}` as the canonical reference (so it is compatible with AbstractRuntime’s `is_artifact_ref`).
  - Add optional metadata fields for better UX (filename, sha256, content_type, etc.).

References:
- AbstractRuntime artifact refs: `abstractruntime/src/abstractruntime/storage/artifacts.py`

---

## Decision

**Chosen approach**:
- Introduce a `GeneratedMediaRef` dict shape that is always an artifact ref, with optional UX metadata:
  - required: `"$artifact"`
  - recommended: `content_type`, `sha256`, `filename`, `metadata` (free-form JSON)
- Add a tiny storage interface with two implementations:
  - `LocalAssetStore` (writes files under `~/.abstractvision/assets/` or configured base dir; returns a ref)
  - `RuntimeArtifactStoreAdapter` (duck-typed wrapper around an AbstractRuntime `ArtifactStore`, without a hard dependency)

**Why**:
- Third-party integrations can treat every output as a small ref object.
- The same output object can be stored in runtime vars, transmitted over JSON, and rendered by clients.

---

## Dependencies

- **Backlog tasks**:
  - Completed: `docs/backlog/completed/005_core_api_tasks_and_abstractions.md`
  - Completed: `docs/backlog/completed/011_abstractcore_tool_integration_and_artifact_refs.md` (primary consumer)

---

## Implementation plan

- Define the output contract in one place:
  - `GeneratedMediaRef` (TypedDict or dataclass→dict helper), always containing `$artifact`.
- Implement storage helpers:
  - compute sha256
  - write bytes (local) or store to provided artifact store
- Update `VisionManager`/backends to support:
  - returning raw bytes (optional) OR returning/storing to `GeneratedMediaRef` (preferred)
- Add unit tests:
  - local store write/read
  - deterministic hashing
  - ref shape always contains `$artifact`

---

## Success criteria

- A generated image/video can be returned as a small JSON object carrying an artifact id.
- The output object is compatible with AbstractRuntime artifact refs and safe to embed in tool outputs.

---

## Test plan

- `python -m unittest discover -s abstractvision/tests -p "test_*.py" -q`

---

## Report (fill only when completed)

### Summary

- Implemented an artifact-ref-first output contract and storage helpers:
  - `abstractvision/src/abstractvision/artifacts.py`:
    - `make_media_ref()`, `is_artifact_ref()`, `compute_artifact_id()`
    - `LocalAssetStore` (standalone store under `~/.abstractvision/assets/` by default)
    - `RuntimeArtifactStoreAdapter` (duck-typed wrapper for AbstractRuntime `ArtifactStore`)
- Updated `VisionManager` to return artifact refs when a store is configured, while keeping raw-bytes `GeneratedAsset` as the escape hatch when no store is present:
  - `abstractvision/src/abstractvision/vision_manager.py`
- Exposed the integration surface in the public API:
  - `abstractvision/src/abstractvision/__init__.py`

Output contract (v0):
- Canonical reference: `{"$artifact": "<id>"}`
- Common UX fields (best-effort): `content_type`, `sha256`, `filename`, `size_bytes`, `metadata`

Local store layout (v0):
- Content: `~/.abstractvision/assets/<artifact_id>.<ext>`
- Metadata: `~/.abstractvision/assets/<artifact_id>.meta.json`

Tags (v0):
- When `VisionManager.store` is configured, generated outputs are stored with tags:
  - `kind=generated_media`, `modality=image|video`, `task=text_to_image|...`

Unit tests:
- `abstractvision/tests/test_artifact_outputs.py`

### Validation

- Tests:
  - `python -m unittest discover -s abstractvision/tests -p "test_*.py" -q`
