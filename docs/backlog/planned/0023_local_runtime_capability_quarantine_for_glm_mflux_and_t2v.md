# Planned: Quarantine unreliable local runtime capabilities until they are re-validated

## Metadata
- Created: 2026-05-22
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
- ADR impact: None if we keep the quarantine backend-owned and continue to treat the packaged registry as model-family metadata rather than unconditional local-runtime truth.

## Context
AbstractVision currently has a clean packaged capability registry and backend-owned runtime gating, but recent operator testing showed that some local model/task combinations are not yet honest enough to expose as working local capabilities.

Three cases need to stay quarantined until they are re-validated:

- local Diffusers `zai-org/GLM-Image` / `GLM-Image` for both `text_to_image` and `image_to_image`;
- local MFLUX `image_to_image` for the current curated FLUX klein families;
- local Diffusers `text_to_video` for the current CogVideoX path.

## Current code reality
Files re-checked on 2026-05-22:

- `src/abstractvision/backends/huggingface_diffusers.py`
- `src/abstractvision/backends/mflux.py`
- `src/abstractvision/playground_server.py`
- `src/abstractvision/cli.py`
- `src/abstractvision/integrations/abstractcore_plugin.py`
- `src/abstractvision/assets/vision_model_capabilities.json`

The current runtime policy after the quarantine change is:

- local Diffusers `GLM-Image` is hidden from runtime-backed local surfaces and rejected by the backend for `text_to_image` and `image_to_image`;
- local MFLUX advertises `text_to_image` only and rejects `image_to_image`;
- local Diffusers CogVideoX `text_to_video` is marked experimental and disabled from normal local surfaces.

The packaged registry still records the broader model-family tasks. That is intentional: the registry says what the family can do in principle, while the backend says what AbstractVision can currently run honestly.

## Operator evidence to preserve

### 1) `GLM-Image` local Diffusers is not shippable yet
- Operator ran local `GLM-Image` from the playground and rejected the output quality even for `text_to_image`.
- Earlier runtime investigation also found `image_to_image` path-specific issues on Apple Silicon:
  - pipeline input shape differences (`image=[pil_image]` expectation);
  - PyTorch MPS failure in the GLM vision embedding path (`grid_sample(..., padding_mode="border")`);
  - instability on the image-conditioning path with `float16`.
- A narrow workaround path could make some Apple Silicon `i2i` runs execute, but that is not sufficient. The operator explicitly does not want a CPU-heavy fallback story except as a last resort, and the current local quality bar is not met.

### 2) MFLUX `image_to_image` fails structural fidelity
- Source image: snowy grounded spacecraft; prompt: watercolor.
- Diffusers FLUX klein produced a recognizable watercolor transform of the same scene.
- MFLUX FLUX klein produced a semantically wrong result with severe structure drift.
- Second source image: indoor room / living-room scene; prompt: watercolor.
- MFLUX preserved some coarse layout but still failed scene fidelity enough that the operator does not consider current local `i2i` acceptable.
- Conclusion: MFLUX is acceptable for `text_to_image` iteration today, but not for `image_to_image`.

### 3) Local `text_to_video` is wired but not honest enough yet
- CogVideoX local path can produce technically valid MP4 outputs.
- The operator observed outputs that “make no sense” semantically and does not want the package to present this as a working local capability today.
- Conclusion: keep the code and tests as internal groundwork if useful, but do not surface local `text_to_video` as working until there is a real quality acceptance pass.

## Problem
If the playground, CLI catalog, or plugin surfaces advertise these tasks as usable local capabilities, they overstate current product truth.

## Decision
Quarantine the unreliable local task paths now and only re-enable them after a targeted validation task produces evidence that the outputs are acceptable.

Current quarantine policy:

- local Diffusers `GLM-Image`: disable `text_to_image` and `image_to_image`;
- local MFLUX: disable `image_to_image`, keep `text_to_image`;
- local Diffusers CogVideoX: disable `text_to_video` from normal local surfaces and describe it as experimental / not working.

## Why
- This keeps the package honest.
- It preserves the clean abstraction boundary: runtime truth stays backend-owned.
- It avoids forcing low-confidence features through the playground, CLI, and AbstractCore plugin just because the model family metadata exists.

## Scope
Included:

- keep the runtime blacklist in backend capability surfacing;
- keep playground/CLI/plugin discovery aligned with that backend truth;
- preserve the investigation notes and operator examples;
- define what evidence is required before re-enabling these tasks.

Not included:

- solving GLM local quality/runtime issues in this task;
- solving MFLUX structural `i2i` fidelity in this task;
- solving local CogVideoX quality in this task;
- widening local video support.

## Re-enable criteria

### GLM local Diffusers
- reproducible `t2i` outputs that meet operator quality expectations;
- reproducible `i2i` outputs on Apple Silicon without unacceptable fallback behavior;
- no special-case runtime path that silently pushes most work to CPU.

### MFLUX `image_to_image`
- side-by-side evidence on at least two representative edit cases showing acceptable scene preservation compared with the current Diffusers baseline;
- clear task-specific constraints documented if MFLUX `i2i` needs narrower prompts/settings than `t2i`.

### Local `text_to_video`
- at least one local model/path that produces semantically acceptable short clips under documented settings;
- explicit hardware/settings guidance and representative validation clips;
- no claim of “working local video” before that quality bar is met.

## Dependencies and related tasks
- Planned: [017_mlx_mflux_backend_strategy.md](017_mlx_mflux_backend_strategy.md)
- Planned: [020_adapter_aware_model_graph_and_catalog.md](020_adapter_aware_model_graph_and_catalog.md)
- Planned: [0022_local_diffusers_image_to_video_backend.md](0022_local_diffusers_image_to_video_backend.md)

## Expected outcomes
- Runtime-backed local surfaces only expose local tasks we are willing to stand behind.
- The registry remains curated model metadata, while backend gating remains the source of local execution truth.
- Future re-enablement work has preserved evidence and explicit acceptance criteria instead of oral history.

## Validation
- Verify local provider-model listings exclude quarantined tasks.
- Verify playground task selectors do not surface quarantined local models/tasks.
- Verify backend direct calls raise clear capability errors for quarantined paths.
- Verify docs and release notes state the quarantine explicitly.

## Progress checklist
- [ ] Preserve all current operator/runtime findings in this backlog item.
- [ ] Keep runtime-backed local surfacing aligned with the quarantine policy.
- [ ] Revisit each quarantined path with explicit acceptance criteria before re-enabling it.

## Guidance for the implementing agent
Do not widen claims while this item is open. Treat the quarantine as intentional product truth, not as a temporary testing hack. If you later improve one of these paths, re-open the evidence, compare against the operator examples preserved here, and only then remove the corresponding backend blacklist.
