## Task 025: Shared LoRA request contract for MLX-Gen and Diffusers

**Date**: 2026-06-11  
**Status**: Completed  
**Priority**: P0

---

## Main goals

- Add one package-owned LoRA adapter request contract for `text_to_image`, `image_to_image`, `text_to_video`, and `image_to_video`.
- Upgrade AbstractVision to the current `mlx-gen` runtime floor and surface its route-level LoRA capability metadata.
- Keep exact LoRA compatibility truth backend-owned while making CLI, Python, and AbstractCore request surfaces consistent.

## Secondary goals

- Preserve compatibility with existing Diffusers `extra["loras*"]` callers.
- Preserve MLX-Gen backend-level default adapter config as a separate residency/runtime concern.
- Improve generated-media provenance so operators can see which adapters were requested and which ones actually applied.

---

## Context / problem

AbstractVision already has real LoRA execution paths, but the public contract is fragmented:

- Diffusers accepts LoRA payloads through `request.extra` keys such as `loras`, `loras_json`,
  `lora`, and `lora_json`.
- MLX-Gen exposes constructor-level `lora_paths` / `lora_scales` config in the backend, but not a
  request-level contract shared with the other surfaces.
- One-shot `abstractvision t2i/i2i/t2v/i2v` commands do not expose first-class LoRA flags.
- AbstractCore request whitelists do not preserve a typed adapter field, and `t2v` / `i2v` still
  drift from the typed request surface by dropping `guidance_2` into `extra`.
- Provider-model discovery does not surface the new MLX-Gen route-level LoRA fields such as
  `supports_lora`, `lora_status`, `lora_target_roles`, and `lora_validation_profile`.

That leaves users with a backend feature but not a stable package contract. The current state is
especially weak for Wan video LoRA, because A14B routes require explicit target-role assignment.

---

## Constraints

- Keep orchestration thin and backend semantics backend-owned per ADR 0004.
- Do not make the packaged registry the source of truth for MLX-Gen route-level LoRA validation.
- Keep backward compatibility for existing Diffusers `extra["loras*"]` users during migration.
- Do not promote adapter repositories as standalone runnable model families.
- Keep the base package lightweight and local runtimes optional.
- Do not create Apple-only public API semantics. The shared request contract must describe adapters,
  not a host platform.

---

## Research, options, and references

This section is self-contained on purpose so later implementation or review work can reuse it.

- **Option A: keep backend-specific LoRA passthrough only**
  - Lowest immediate code churn, but it keeps CLI/Python/Core inconsistent and keeps Wan target-role
    support effectively hidden.
  - Rejected because it does not create a trustworthy package contract.
  - References:
    - `src/abstractvision/backends/huggingface_diffusers.py`
    - `src/abstractvision/backends/mflux.py`

- **Option B: move exact LoRA truth into `vision_model_capabilities.json`**
  - Centralized, but wrong authority boundary. MLX-Gen now reports route-level LoRA truth dynamically,
    including validation status and target roles, and those fields are release-sensitive.
  - Rejected because it duplicates backend truth and will drift.
  - References:
    - `src/abstractvision/assets/vision_model_capabilities.json`
    - `docs/adr/0005_curated_capability_registry_and_download_catalog.md`

- **Option C: shared request contract in AbstractVision + dynamic route truth in backends**
  - AbstractVision owns one request shape across CLI/Python/Core.
  - Diffusers and MLX-Gen keep authority over compatibility, loader semantics, and validation truth.
  - Chosen because it matches ADR 0004 and the current MLX-Gen capability API.
  - References:
    - `docs/adr/0004_thin_orchestrator_and_backend_owned_model_semantics.md`
    - `docs/backlog/planned/020_adapter_aware_model_graph_and_catalog.md`
    - `https://github.com/lpalbou/mlx-gen`
    - `https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/lora.md`
    - `https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/api.md`

Additional upstream findings re-checked on 2026-06-11:

- `mlx-gen` latest release is `0.18.16`.
- `mlx-gen` now exposes route-level LoRA capability fields:
  `supports_lora`, `lora_status`, `lora_target_roles`, `lora_validation_profile`.
- Wan video LoRA is now supported through the shared runtime with:
  - one target role (`transformer`) on TI2V-5B routes;
  - two explicit roles (`high_noise_transformer`, `low_noise_transformer`) on A14B routes.
- MLX-Gen package metadata is no longer Darwin-only; upstream declares Linux/CUDA installation
  markers as well. AbstractVision still needs to decide what is release-validated versus merely
  installable.

---

## Decision

**Chosen approach**: add a typed request-level `lora_adapters` contract in AbstractVision, keep
backend-exact LoRA truth backend-owned, and surface MLX-Gen route metadata dynamically through
provider-model discovery.

**Why**:
- It gives users one stable package concept without inventing fake cross-backend compatibility.
- It keeps Wan target-role semantics honest instead of flattening them into image-style adapter
  behavior.
- It lets Diffusers and MLX-Gen keep their real loader/runtime constraints while still sharing CLI,
  Python, and AbstractCore request shapes.
- It preserves existing `extra["loras*"]` usage as a compatibility bridge instead of breaking older
  callers immediately.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0004_thin_orchestrator_and_backend_owned_model_semantics.md`
  - `docs/adr/0005_curated_capability_registry_and_download_catalog.md`
- **Backlog tasks**:
  - Planned: `docs/backlog/planned/020_adapter_aware_model_graph_and_catalog.md`
  - Completed: `docs/backlog/completed/016_abstractcore_plugin_catalog_discovery_surface.md`
  - Completed: `docs/backlog/completed/019_best_effort_preload_warmup_for_local_backends.md`

---

## Implementation plan

- Add a typed `LoRAAdapterSpec` request object and attach `lora_adapters` to the four shared
  generation/edit/video request types.
- Add package-owned normalization helpers that:
  - parse typed adapter lists;
  - bridge legacy `extra["loras*"]` payloads;
  - serialize requested adapters for metadata and docs.
- Upgrade the MLX-Gen optional dependency floor and update install markers to match the current
  upstream package metadata.
- Update the MLX-Gen backend to:
  - accept request-level adapters;
  - preserve backend-level default adapters as a separate config path;
  - pass Wan `lora_target_roles` when needed;
  - include adapter signature in its model residency/cache key;
  - surface route-level LoRA metadata through provider-model discovery;
  - preserve generated-image/video LoRA provenance from upstream metadata.
- Update the Diffusers backend to consume the shared adapter parser first, then fall back to legacy
  `extra["loras*"]`.
- Add explicit one-shot CLI LoRA flags for `t2i`, `i2i`, `t2v`, and `i2v`.
- Fix AbstractCore request whitelists so typed fields survive, including `guidance_2` for `t2v`
  and `i2v`.
- Add focused tests for typed request routing, metadata propagation, provider discovery metadata,
  and packaging markers.

---

## Success criteria

- Python callers can pass one `lora_adapters=[...]` contract to T2I, I2I, T2V, and I2V requests.
- One-shot CLI exposes first-class adapter flags on `t2i`, `i2i`, `t2v`, and `i2v`.
- MLX-Gen provider-model discovery surfaces route-level LoRA metadata for supported routes.
- Wan A14B requests can carry explicit target-role assignments through AbstractVision.
- Diffusers legacy `extra["loras*"]` callers still work.
- Generated metadata includes requested adapter provenance and backend-applied LoRA metadata where
  the backend can provide it.

---

## Test plan

- `PYTHONPATH=abstractvision/src python -m pytest abstractvision/tests/test_packaging_metadata.py -q`
- `PYTHONPATH=abstractvision/src python -m pytest abstractvision/tests/test_mflux_backend.py -q`
- `PYTHONPATH=abstractvision/src python -m pytest abstractvision/tests/test_huggingface_diffusers_backend.py -q`
- `PYTHONPATH=abstractvision/src python -m pytest abstractvision/tests/test_abstractcore_plugin.py -q`
- Manual proof:
  - MLX-Gen T2I LoRA run on a validated route.
  - MLX-Gen I2I LoRA run on a validated route.
  - MLX-Gen T2V LoRA run on a validated Wan route.
  - MLX-Gen I2V LoRA run on a validated Wan route.

---

## Report

### Summary

AbstractVision now owns one typed LoRA adapter request contract across `text_to_image`,
`image_to_image`, `text_to_video`, and `image_to_video`, while MLX-Gen and Diffusers keep
authority over route compatibility and runtime application rules.

The shipped implementation adds `LoRAAdapterSpec` and per-request `lora_adapters`, updates the
MLX-Gen backend to consume those adapters directly, preserves Diffusers `extra["loras*"]`
compatibility as a bridge, upgrades the optional MLX-Gen runtime floor to `0.18.16`, and surfaces
route-level LoRA truth through provider discovery (`supports_lora`, `lora_status`,
`lora_target_roles`, `lora_validation_profile`).

The AbstractCore plugin path now forwards typed LoRA adapters and backend-default LoRA config, and
the CLI exposes first-class repeated `--lora*` flags for `t2i`, `i2i`, `t2v`, and `i2v`.

During proof work, one real integration bug surfaced in MLX-Gen model config resolution for exact
Qwen edit variants. The fix was to prefer `ModelConfig.from_name(<exact-registry-id>)` before
falling back to generic config factories, so exact route identity, capability checks, and LoRA
compatibility stay aligned.

### Validation

- Focused tests passed:
  - `python -m pytest abstractvision/tests/test_mflux_backend.py -q`
  - `python -m pytest abstractvision/tests/test_cli_smoke.py abstractvision/tests/test_abstractcore_plugin.py abstractvision/tests/test_huggingface_diffusers_backend.py abstractvision/tests/test_packaging_metadata.py -q`
- Real MLX-Gen proof runs saved under `/private/tmp/abstractvision_lora_proofs_20260612_002853`:
  - T2I: `AbstractFramework/qwen-image-2512-8bit` with a pixel-art LoRA
  - I2I: `AbstractFramework/qwen-image-edit-2511-8bit` with a multi-angle LoRA
  - T2V: `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` with a Wan TI2V LoRA
  - I2V: `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit` with the same Wan TI2V LoRA
- Bundled documentation proof assets were copied into
  `docs/assets/mlx-gen-lora-examples/`, including:
  - `summary.json`
  - `provider_catalog_lora.json`
  - generated PNG/MP4 outputs
  - contact sheets
  - denoise-step progress logs
