## Task 028: MLX-Gen Qwen mask/control sync and packaged adapter capability registry

**Date**: 2026-06-15  
**Status**: Completed  
**Priority**: P0

---

## Main goals

- Upgrade the optional MLX-Gen runtime floor to the latest published release.
- Surface the current MLX-Gen Qwen masked-edit and structured-control slices
  through stable AbstractVision request fields.
- Move package-owned adapter compatibility/default metadata out of Python
  branches and into a packaged JSON asset.

## Secondary goals

- Keep backend/runtime route truth explicit and fail closed on mismatches.
- Preserve the shared LoRA contract while making adapter defaults data-driven.
- Document the new authority boundary clearly in the core docs and backlog.

---

## Context / problem

Upstream `mlx-gen` moved ahead of the AbstractVision consumer boundary:

- `AbstractFramework/qwen-image-edit-2511-8bit` now exposes route-aware masked
  edit/inpaint.
- `AbstractFramework/qwen-image-8bit` now exposes route-aware structured
  control.
- AbstractVision still hard-coded MLX-Gen mask support to FIBO Edit families
  and had no first-class `control_image` request field.
- Adapter compatibility/defaults for Lightning families were also embedded in
  `lora_adapters.py`, which made routine catalog/default updates require Python
  code edits.

That combination left the runtime truth under-surfaced and the packaged policy
too entangled with execution code.

---

## Constraints

- Keep the orchestrator thin and backend/runtime semantics backend-owned.
- Do not silently reinterpret structured control as image edit or vice versa.
- Keep the shared request/dataclass surface additive rather than introducing
  new top-level task families.
- Keep adapter compatibility/default hints package-owned, but keep exact route
  execution truth backend-owned.

---

## Dependencies

- ADR 0004: [thin orchestrator and backend-owned model semantics](../../adr/0004_thin_orchestrator_and_backend_owned_model_semantics.md)
- ADR 0005: [curated capability registry and download catalog](../../adr/0005_curated_capability_registry_and_download_catalog.md)
- ADR 0007: [explicit fallback and degraded mode disclosure](../../adr/0007_explicit_fallback_and_degraded_mode_disclosure.md)
- Planned follow-up still open:
  [020_adapter_aware_model_graph_and_catalog.md](../planned/020_adapter_aware_model_graph_and_catalog.md)
- Related runtime-quality quarantine:
  [0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md](../planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md)

---

## Implementation plan

- Raise the MLX-Gen optional dependency floor to `0.18.19`.
- Add typed `control_image` / `control_strength` to
  `ImageGenerationRequest`, plus `supports_control_image` to
  `VisionBackendCapabilities`.
- Rework MLX-Gen capability surfacing to use route-aware metadata for mask and
  structured-control support instead of family-only gates.
- Add packaged `vision_adapter_capabilities.json` plus a loader/validator and
  refactor `lora_adapters.py` into a matcher/serializer layer over that asset.
- Add filesystem override paths for both the model and adapter capability
  assets.
- Repair the stale operator docs/backlog notes in the same pass.

---

## Success criteria

- Qwen 2511 masked edit no longer fails at the old FIBO-only gate.
- Base-Qwen structured control is available through typed public request
  fields, CLI flags, and route-aware backend gating.
- Known Lightning adapter defaults/compatibility notes are driven by packaged
  JSON instead of hard-coded Python branches.
- Unsupported backends and unsupported MLX routes fail closed instead of
  silently ignoring `control_image` or `mask`.

---

## Test plan

- `pytest -q tests/test_capabilities_schema_validation.py tests/test_manager_capability_checks.py tests/test_cli_smoke.py tests/test_huggingface_diffusers_backend.py tests/test_mflux_backend.py tests/test_packaging_metadata.py`
- `pytest -q tests/test_openai_compatible_backend.py tests/test_stable_diffusion_cpp_backend.py`
- `python -m compileall src/abstractvision`

---

## Report

### Summary

AbstractVision now targets `mlx-gen>=0.18.19,<0.19.0`, exposes typed
`control_image` / `control_strength` for validated MLX-Gen Qwen structured
control, and exposes route-aware masked edit for validated Qwen 2511 and FIBO
Edit rows.

The package now also ships a second curated capability asset,
`vision_adapter_capabilities.json`, alongside the existing model registry.
That asset owns adapter-family compatibility/default hints such as model
aliases, recommended `steps`, recommended `guidance_scale`, documentation
links, and local quantization notes. `lora_adapters.py` now loads and matches
those profiles instead of encoding the current Lightning families directly in
Python.

Both capability registries can now be redirected to external JSON files through
`asset_path=...` or environment overrides:

- `ABSTRACTVISION_MODEL_CAPABILITIES_PATH`
- `ABSTRACTVISION_ADAPTER_CAPABILITIES_PATH`

That keeps the code/package boundary stable while allowing curated metadata
updates without changing Python code.

### Validation

- Focused validation passed:
  - `pytest -q tests/test_capabilities_schema_validation.py tests/test_manager_capability_checks.py tests/test_cli_smoke.py tests/test_huggingface_diffusers_backend.py tests/test_mflux_backend.py tests/test_packaging_metadata.py`
  - `205 passed, 4 subtests passed`
  - `pytest -q tests/test_openai_compatible_backend.py tests/test_stable_diffusion_cpp_backend.py`
  - `25 passed, 1 skipped`
- Compile check passed:
  - `python -m compileall src/abstractvision`

### User-visible outcomes

- `abstractvision t2i --provider mlx-gen --model AbstractFramework/qwen-image-8bit --control-image ... --control-strength ...`
- `abstractvision i2i --provider mlx-gen --model AbstractFramework/qwen-image-edit-2511-8bit --mask ...`
- `abstractvision adapters --provider mlx-gen ... --json` now surfaces adapter
  defaults/documentation/quantization guidance from the packaged adapter
  registry rather than Python-only tables.
