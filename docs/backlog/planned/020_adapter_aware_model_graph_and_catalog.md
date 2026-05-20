# Planned: Model family, component, and overlay graph for catalog and curated local flows

## Metadata
- Created: 2026-05-20
- Status: Planned
- Priority: P1
- Completed: N/A

## ADR status
- Governing ADRs:
  - [ADR 0004: Keep the orchestrator thin and make model semantics backend-owned](../../adr/0004_thin_orchestrator_and_backend_owned_model_semantics.md)
  - [ADR 0005: Own a curated capability registry and cache-backed model catalog](../../adr/0005_curated_capability_registry_and_download_catalog.md)
  - [ADR 0006: Keep runtime selection explicit and operator-controlled](../../adr/0006_operator_control_configuration_precedence_and_explicit_network_use.md)
  - [ADR 0007: Disclose fallbacks and degraded modes explicitly](../../adr/0007_explicit_fallback_and_degraded_mode_disclosure.md)
  - [ADR 0008: Require validation and evidence-based change reporting](../../adr/0008_validation_and_evidence_based_change_reporting.md)
  - [ADR 0009: Keep docs, backlog, and ADRs code-first](../../adr/0009_code_first_docs_backlog_and_adr_discipline.md)
- ADR impact: None for the initial implementation if the work stays within the existing package-owned
  boundaries above. If implementation pressure requires a new cross-backend overlay contract or a
  materially different catalog authority boundary, revise ADR 0004 and/or ADR 0005 before closure.

## Context
AbstractVision now has a useful capability registry, a useful curated downloader, and a clearer
package policy baseline. What it still lacks is a clean taxonomy for the artifacts it surfaces.

The current planning problem is not “support adapters somehow.” It is more specific:

- external users should not need to understand internal side artifacts such as VAE or encoder files
  when a curated local flow exists;
- adapter-like repos must not be presented as standalone runnable models when the runtime cannot
  actually execute them that way;
- catalog, downloader, CLI, playground, and AbstractCore surfaces should agree on what is a full
  model, what is a required component, and what is an optional overlay.

The 2026-05-20 review surfaced the exact ambiguity to fix:

- `stable-diffusion.cpp` component stacks such as FLUX/Qwen require side artifacts like VAE and
  LLM/encoder files, but those are required components, not adapters;
- Diffusers LoRA and MFLUX LoRA inputs are optional overlays, not full model families;
- some Hugging Face repos look like “models” in an inventory pass but are really overlays or
  base-model-dependent attachments, for example:
  - `lightx2v/Qwen-Image-Lightning`
  - `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`
  - `black-forest-labs/FLUX.1-Redux-dev`

Those entries were removed from the curated registry the same day because the current product
surface does not yet model them honestly enough.

## Current code reality
Files and symbols re-checked before revising this item:

- `src/abstractvision/assets/vision_model_capabilities.json`
- `src/abstractvision/model_capabilities.py`
- `src/abstractvision/model_downloads.py`
- `src/abstractvision/vision_manager.py`
- `src/abstractvision/types.py`
- `src/abstractvision/backends/huggingface_diffusers.py`
- `src/abstractvision/backends/mflux.py`
- `src/abstractvision/cli.py`
- `src/abstractvision/integrations/abstractcore.py`
- `docs/getting-started.md`

What is already implemented:

- The registry schema supports `models -> downloads[] -> tasks`.
- `download-model`, `model-presets`, and `model-catalog` already expose a package-owned curated
  surface on top of the registry plus `_PRESETS`.
- `stable-diffusion.cpp` now has a package-owned curated bundle path via
  `resolve_sdcpp_model_selection(...)`, which resolves required cached companion artifacts for keys
  such as `flux2-klein-base-4b` and `qwen-image`.
- Diffusers already has real overlay behavior:
  - `_parse_loras()`
  - `_apply_loras()`
  - documented `loras_json` and `rapid_aio_repo` request-extra flows
- MFLUX already has backend-level `lora_paths` / `lora_scales` config support.
- `ImageGenerationRequest` and `ImageEditRequest` already include `extra`, so the package has a
  low-level carrier for backend-owned overlay metadata.

What is missing or brittle:

- The registry has no first-class distinction between:
  - parent model family
  - runtime variant
  - required component
  - optional overlay
- `VisionManager` gates only on task names. It does not interpret `task.requires`, and current
  shipped JSON does not yet exercise `base_model_id` as a real runtime contract.
- `_PRESETS` and curated bundle logic still encode important catalog truth in code rather than in a
  clearer package data model.
- AbstractCore tools expose fixed prompt/image/mask parameters only. They do not expose package-
  owned overlay composition or typed `extra` inputs.
- The current docs show working Diffusers overlay examples, but the catalog cannot yet tell a user
  whether a repo is a standalone model, a required component, or an overlay attachment.
- `multi_view_image` exists as a task key, but no shipped backend currently implements it, so LoRA
  repos that imply multi-view behavior must not be cataloged as if they were runnable tasks.

## Problem
The current registry and download surfaces are still too flat. They are good at listing model ids
and repo ids, but they are not yet a clean graph of:

- parent model families;
- runtime-specific variants;
- required components that must be assembled for one runtime path;
- optional overlays that only make sense with a compatible parent backend/model;
- runtime-ready versus download-only versus unsupported states.

That makes the curated catalog less honest than it should be and pushes too much repo knowledge
back onto the user.

## What we want to do
Introduce an explicit package-owned artifact taxonomy and use it to improve:

- `vision_model_capabilities.json`
- `model-catalog`
- `model-presets`
- `download-model`
- curated local runtime resolution
- overlay-aware generation surfaces where the backend already supports them

The first goal is not “generic adapter orchestration.” The first goal is to stop confusing model
families, required components, and optional overlays.

## Why
Users are not asking for a raw list of Hugging Face repos. They want reliable answers to:

- what model family this is;
- what I can run now;
- what I can only download;
- what extra components are silently required for this runtime;
- whether a repo is a full model, a required component, or an optional overlay.

Without a clearer graph, the registry keeps drifting toward an inventory dump instead of a product
surface for discovery, download, and execution.

## Decision boundaries

### 1. Required components are not overlays
VAE, text encoders, vision encoders, and similar side artifacts that are mandatory for a runtime
path belong in a required-component category. They should not be modeled as adapters.

### 2. Optional overlays are not standalone model families
LoRA, lightning/distilled transformer overrides, redux-style priors, and similar attachments should
not be promoted as first-class runnable models unless a shipped backend can really execute them as
such.

### 3. Package-owned local composition is allowed when the package can keep it honest
The existing `sdcpp` bundle resolver proves that AbstractVision can own required-component
resolution for curated flows. That pattern should be extended deliberately, not hidden.

### 4. Backend-owned overlay semantics must stay backend-owned
This task must not invent a fake generic cross-backend adapter runtime if Diffusers and MFLUX need
different semantics. A shared package surface is acceptable only if it preserves backend truth.

### 5. Tool exposure is conditional, not automatic
AbstractCore tool surfaces should expose overlay-aware inputs only after the package has a clean,
typed, package-owned contract for them. The current fixed-parameter tools are safer than exposing
aspirational generic adapter knobs.

## Requirements
- Keep one clear parent entry per real model family.
- Represent engine/format-specific runnable downloads as runtime variants, not as independent model
  families.
- Represent required side artifacts explicitly and separately from optional overlays.
- Preserve direct HF repo metadata where useful, but stop presenting overlays as if they were
  standalone runnable models.
- Surface runtime state clearly:
  - runnable now
  - curated local flow with package-owned required components
  - download-only
  - remote-only
  - overlay-only
  - requires unsupported backend work
- Keep curated local flows seamless where the package already owns the mapping.
- Expose overlay-aware generation only where a shipped backend already supports it, and reject it
  clearly elsewhere.
- Keep offline usage after download.
- Preserve backward compatibility where practical, or add an explicit compatibility layer if the
  registry schema evolves.

## Suggested implementation

### Phase 1: make the taxonomy explicit
- Add an additive artifact-role model that can distinguish:
  - parent model family
  - runtime variant
  - required component
  - optional overlay
- Prefer an additive schema evolution or compatibility translation layer over a risky big-bang
  rewrite.
- Document source/provenance per artifact role:
  - official
  - runtime-native community
  - fallback community

### Phase 2: align curated local flows with the taxonomy
- Move current special-case bundle knowledge toward package data where practical.
- Treat `sdcpp` FLUX/Qwen bundle resolution as the first concrete example of package-owned required
  component assembly.
- Keep errors actionable when required cached components are missing.

### Phase 3: expose overlays only where runtime support already exists
- Diffusers:
  - formalize the currently documented LoRA and Rapid-AIO request-extra paths
  - decide whether to keep them as explicit `extra` conventions or introduce a typed package-owned
    overlay field
- MFLUX:
  - decide whether current config-only LoRA support should stay config-bound or also become
    request-level when the runtime semantics are clear
- Do not promise cross-backend overlay parity if the runtimes are materially different.

### Phase 4: evaluate higher-level surfaces
- Update CLI and catalog output so overlays are not mistaken for runnable models.
- Only extend `integrations/abstractcore.py` after the package-owned overlay surface is explicit
  enough to expose safely.
- Add early rejection paths when a user selects an overlay without a compatible parent/runtime.

## Scope
Included:

- capability/download taxonomy work for model families versus variants versus required components
  versus overlays;
- downloader and catalog alignment with that taxonomy;
- curated local-flow improvement where the package can already own required-component resolution;
- overlay-aware generation surface design for backends that already support overlays;
- clear rejection behavior for unsupported overlay/runtime combinations.

## Non-goals
- Implement every overlay or adapter family immediately.
- Treat VAE/encoders as user-managed details in curated flows that the package can own.
- Build a generic “adapter engine” abstraction that hides runtime-specific semantics.
- Expose overlay parameters through AbstractCore tools before the package-facing contract is clean.
- Reclassify every historical repo in one risky pass without a compatibility strategy.
- Ship local video runtime work as part of this item.

## Dependencies and related tasks
- Completed: [003_hf_model_landscape_and_capability_registry.md](../completed/003_hf_model_landscape_and_capability_registry.md)
- Completed: [004_capability_schema_and_validation.md](../completed/004_capability_schema_and_validation.md)
- Completed: [011_abstractcore_tool_integration_and_artifact_refs.md](../completed/011_abstractcore_tool_integration_and_artifact_refs.md)
- Completed: [016_abstractcore_plugin_catalog_discovery_surface.md](../completed/016_abstractcore_plugin_catalog_discovery_surface.md)
- Completed: [019_best_effort_preload_warmup_for_local_backends.md](../completed/019_best_effort_preload_warmup_for_local_backends.md)
- Planned: [017_mlx_mflux_backend_strategy.md](017_mlx_mflux_backend_strategy.md)
- Docs reference: [docs/getting-started.md](../../getting-started.md)

## Expected outcomes
- The curated registry no longer confuses parent model families, required components, and overlays.
- Curated local component-based flows such as `sdcpp` are modeled as package-owned required
  component assembly rather than as hidden repo trivia.
- `model-catalog` and `download-model` can tell a user whether an artifact is runnable, download-
  only, a required component, or an optional overlay.
- Overlay-aware generation is either clearly supported end to end for a backend/model path or
  clearly rejected with an actionable error.
- AbstractCore integration does not advertise overlay composition beyond what the package can
  really honor.

## Validation
- `PYTHONPATH=src python - <<'PY'`
  `from abstractvision.model_capabilities import VisionModelCapabilitiesRegistry`
  `VisionModelCapabilitiesRegistry()`
  `print("ok")`
  `PY`
- `PYTHONPATH=src python -m abstractvision.cli model-catalog --json`
- `PYTHONPATH=src python -m abstractvision.cli model-presets --all-targets --all --json`
- `PYTHONPATH=src python -m abstractvision.cli show-model <model_id>`
- Add tests for:
  - additive artifact-role validation
  - catalog/download rendering for parent models versus components versus overlays
  - curated required-component resolution for at least one component-based local flow
  - overlay-aware request acceptance and rejection for Diffusers and MFLUX where applicable
  - AbstractCore tool exposure staying aligned with the supported overlay contract
- Manual smoke checks:
  - one Diffusers LoRA path
  - one curated `sdcpp` component bundle path
  - one MFLUX overlay path only if request-level overlay support is actually added

## Progress checklist
- [ ] Define the minimal additive taxonomy for parent families, runtime variants, required components, and overlays.
- [ ] Decide whether the schema evolves directly or through a compatibility translation layer.
- [ ] Move at least one existing component-based curated flow onto the clearer taxonomy.
- [ ] Update catalog/download output so overlays are not shown as standalone runnable models.
- [ ] Decide the package-owned overlay input contract for backends that already support overlays.
- [ ] Add tests and docs for runnable versus download-only versus component versus overlay states.

## Guidance for the implementing agent
Re-check the current code before implementing anything. Favor a small explicit taxonomy over a
clever graph rewrite. Keep required components and optional overlays separate. Do not preserve
adapter entries as fake standalone models just because the old JSON format was permissive. Treat
seamless curated local flows as a UX feature the package should own when it has enough knowledge to
do so honestly.
