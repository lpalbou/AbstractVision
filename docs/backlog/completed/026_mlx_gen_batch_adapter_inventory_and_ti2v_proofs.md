## Task 026: MLX-Gen batch generation, installed-adapter inventory, and corrected TI2V proofs

**Date**: 2026-06-13  
**Status**: Completed  
**Priority**: P0

---

## Main goals

- Upgrade AbstractVision to the current `mlx-gen` runtime floor.
- Expose simple public batch generation for images and videos.
- Expose backend-owned installed-adapter discovery for MLX-Gen routes.
- Correct the bundled TI2V-5B proof assets so the public validation uses a
  practical supported size.

## Secondary goals

- Keep the shared LoRA contract valid for zero, one, or many adapters.
- Make stacked-adapter usage explicit in Python, CLI, and docs.
- Keep adapter discovery honest by excluding full-model component files from the
  adapter inventory.

---

## Context / problem

AbstractVision already had the shared typed LoRA contract from Task 025, but it
still had three release-visible gaps:

- the package docs and proof bundle still centered on the older `mlx-gen`
  release and an undersized TI2V-5B visual example;
- users had no public batch generation surface even though repeated-seed MLX
  workflows are a common operator need;
- adapter discovery needed to distinguish real optional overlays from cached
  full-model component files.

That left the package functionally stronger than its public contract and made
the proof page weaker than the actual backend support.

---

## Constraints

- Keep the orchestrator thin and the exact route truth backend-owned.
- Do not invent a static adapter-capabilities asset when MLX-Gen can classify
  cached overlays dynamically enough for the current surface.
- Keep the single exact request dataclasses stable; batch generation belongs
  above them as orchestration.
- Keep public docs machine-independent even when the underlying proof runs used
  local caches and local source images.

---

## Dependencies

- ADR 0004: [thin orchestrator and backend-owned model semantics](../../adr/0004_thin_orchestrator_and_backend_owned_model_semantics.md)
- ADR 0005: [curated capability registry and download catalog](../../adr/0005_curated_capability_registry_and_download_catalog.md)
- Planned follow-up still open:
  [020_adapter_aware_model_graph_and_catalog.md](../planned/020_adapter_aware_model_graph_and_catalog.md)
- Completed prerequisite:
  [025_shared_lora_request_contract_for_mlx_gen_and_diffusers.md](025_shared_lora_request_contract_for_mlx_gen_and_diffusers.md)

---

## Implementation plan

- Raise the optional MLX-Gen runtime floor to the latest published release
  needed for video LoRA routing and current route behavior.
- Add and document public batch helpers:
  - `VisionManager.generate_image_batch(...)`
  - `VisionManager.edit_image_batch(...)`
  - `VisionManager.generate_video_batch(...)`
  - `VisionManager.image_to_video_batch(...)`
  - CLI `--count` / `--seeds`
- Add `VisionManager.list_provider_adapters(...)` plus MLX-Gen adapter
  discovery that filters cached overlays by route.
- Exclude full-model snapshots and component files from the adapter inventory.
- Rebuild the bundled proof page around real current runs:
  - batch T2I
  - stacked I2I
  - corrected TI2V-5B `832x480`
  - refreshed TI2V batch `832x480`
  - task-specific Wan A14B T2V `480x240`
  - task-specific Wan A14B I2V `480x240`

---

## Success criteria

- Public Python and CLI surfaces can request multiple outputs without changing
  the single exact request dataclasses.
- The shared LoRA contract works for base runs, single-adapter runs, and
  stacked-adapter runs.
- Installed adapter discovery returns optional overlays only and carries
  route-level compatibility truth.
- The bundled TI2V proof uses `832x480` and route-correct `flow_shift`.
- Public docs and generated `llms-full.txt` match the shipped code and proof
  assets.

---

## Test plan

- `pytest -q tests/test_mflux_backend.py tests/test_cli_smoke.py tests/test_manager_capability_checks.py tests/test_packaging_metadata.py`
- `python -m build`
- `python -m twine check dist/*`
- Real CLI proof runs for:
  - batch T2I
  - stacked I2I
  - TI2V-5B at `832x480`
  - batch T2V

---

## Report

### Summary

AbstractVision now targets `mlx-gen>=0.18.18,<0.19.0`, exposes first-class
batch generation above the single exact request surface, and exposes
backend-owned installed-adapter discovery for MLX-Gen routes through
`VisionManager.list_provider_adapters(...)` and `abstractvision adapters ...`.

The adapter inventory now filters out cached full-model component files so
operators see real optional overlays instead of `text_encoder` or `vae`
artifacts. The public docs were refreshed around the current runtime floor,
stacked-adapter usage, adapter discovery, and a corrected TI2V-5B proof at
`832x480`.

### Validation

- Focused validation passed:
  - `pytest -q tests/test_mflux_backend.py tests/test_cli_smoke.py tests/test_manager_capability_checks.py tests/test_packaging_metadata.py`
  - `137 passed, 4 subtests passed`
- The proof bundle under `docs/assets/mlx-gen-lora-examples/` now includes:
  - batch T2I outputs and contact sheet
  - stacked I2I source/output/metadata/progress
  - TI2V-5B `832x480` MP4, metadata, contact sheet, and progress log
  - refreshed TI2V batch outputs at `832x480`, plus contact sheet and progress log
  - task-specific Wan A14B T2V `480x240` MP4, metadata, contact sheet, and progress log
  - task-specific Wan A14B I2V `480x240` input, MP4, metadata, contact sheet, and progress log
  - provider adapter discovery JSON files and `show-model` output
- The bundle manifest is:
  [summary.json](../../assets/mlx-gen-lora-examples/summary.json)
