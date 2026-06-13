## Task 027: AbstractCore tool helper contract parity

**Date**: 2026-06-13  
**Status**: Completed  
**Priority**: P1

---

## Main goals

- Keep `make_vision_tools(...)` aligned with the shipped AbstractCore plugin contract.
- Surface adapter discovery, stacked LoRA adapters, batch generation, and multi-reference image edits through the legacy tool-helper path.

## Secondary goals

- Add explicit regression coverage so the helper path cannot silently lag behind the main plugin path again.

---

## Context / problem

`abstractvision.integrations.abstractcore_plugin` had already been updated to
surface provider adapter discovery plus first-class batch methods for image and
video generation. The older `abstractvision.integrations.abstractcore`
`make_vision_tools(...)` helper still exposed the earlier singular contract, so
hosts using tool helpers would not see the same feature set as hosts using the
capability plugin.

That was a release-shape bug, not an implementation detail. The package needed
one coherent AbstractCore integration story.

---

## Constraints

- Keep `VisionManager` as the execution boundary; do not duplicate batching or adapter compatibility logic in the helper layer.
- Preserve artifact-ref outputs so the helper path stays workflow-safe.
- Keep route-specific semantics backend-owned, consistent with the accepted ADRs.

---

## Research, options, and references

- **Option A**: leave `make_vision_tools(...)` as a narrow legacy surface
  - Pros: no code churn
  - Cons: two incompatible public AbstractCore integration paths; release drift
- **Option B**: retire `make_vision_tools(...)`
  - Pros: simpler long-term surface
  - Cons: breaking change for existing hosts; not requested here
- **Option C**: bring the helper path up to parity while keeping it thin
  - Pros: consistent public contract; preserves existing hosts; keeps truth in `VisionManager`
  - Cons: modest helper/test/doc expansion

References:
- `src/abstractvision/integrations/abstractcore.py`
- `src/abstractvision/integrations/abstractcore_plugin.py`
- `src/abstractvision/vision_manager.py`
- `docs/adr/0002_vision_overlays_and_adapter_composition_are_package_owned.md`
- `docs/adr/0004_thin_orchestrator_and_backend_owned_model_semantics.md`

---

## Decision

**Chosen approach**: Option C.

`make_vision_tools(...)` now exposes:
- `vision_list_adapters(...)`
- typed `lora_adapters=[...]` on `t2i`, `i2i`, `t2v`, and `i2v`
- batch helpers for `t2i`, `i2i`, `t2v`, and `i2v`
- multi-reference `reference_images=[...]` on image-edit helpers

The helper still delegates execution and seed planning to `VisionManager`.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0002_vision_overlays_and_adapter_composition_are_package_owned.md`
  - `docs/adr/0004_thin_orchestrator_and_backend_owned_model_semantics.md`
  - `docs/adr/0008_validation_and_evidence_based_change_reporting.md`
- **Backlog tasks**:
  - Completed: `docs/backlog/completed/011_abstractcore_tool_integration_and_artifact_refs.md`
  - Completed: `docs/backlog/completed/016_abstractcore_plugin_catalog_discovery_surface.md`
  - Completed: `docs/backlog/completed/025_shared_lora_request_contract_for_mlx_gen_and_diffusers.md`
  - Completed: `docs/backlog/completed/026_mlx_gen_batch_adapter_inventory_and_ti2v_proofs.md`

---

## Implementation plan

- Extend helper parsing for adapter payloads and reference-image payloads.
- Add adapter discovery and batch helper tools.
- Expand tests to cover adapter discovery, LoRA stacks, explicit seeds, and multi-reference image-edit forwarding.
- Update integration docs and changelog.

---

## Success criteria

- Tool-helper hosts can discover adapters for the selected model/task.
- Tool-helper hosts can request multiple outputs with explicit seeds in one call.
- Tool-helper hosts can pass stacked LoRA adapters through image and video requests.
- Image-edit tool helpers can forward multiple reference images.

---

## Test plan

- `pytest -q abstractvision/tests/test_abstractcore_tool_integration.py abstractvision/tests/test_abstractcore_plugin.py`

---

## Report (fill only when completed)

### Summary

The legacy AbstractCore tool-helper path now matches the released plugin path
for adapter discovery, LoRA stacks, batch generation, and multi-reference image
edits. The helper remains thin and delegates execution to `VisionManager`.

### Validation

- `pytest -q abstractvision/tests/test_abstractcore_tool_integration.py abstractvision/tests/test_abstractcore_plugin.py`
  - `57 passed`

