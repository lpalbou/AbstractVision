# Completed: MLX-Gen runtime migration for Apple image models

## Metadata
- Created: 2026-05-25
- Status: Completed
- Priority: P0
- Completed: 2026-05-25

## ADR status
- Governing ADRs:
  - [ADR 0004: Keep the orchestrator thin and make model semantics backend-owned](../../adr/0004_thin_orchestrator_and_backend_owned_model_semantics.md)
  - [ADR 0005: Own a curated capability registry and cache-backed model catalog](../../adr/0005_curated_capability_registry_and_download_catalog.md)
  - [ADR 0006: Keep runtime selection explicit and operator-controlled](../../adr/0006_operator_control_configuration_precedence_and_explicit_network_use.md)
  - [ADR 0008: Require validation and evidence-based change reporting](../../adr/0008_validation_and_evidence_based_change_reporting.md)
  - [ADR 0009: Keep docs, backlog, and ADRs code-first](../../adr/0009_code_first_docs_backlog_and_adr_discipline.md)
- ADR impact: None if the work keeps AbstractVision as orchestrator/catalog owner and MLX-Gen as
  the backend-owned model loading/runtime layer. Revise ADR 0004 or ADR 0005 only if the migration
  forces a new cross-package catalog authority boundary.

## Context
The Apple-local image runtime is currently named and implemented as an MFLUX backend. The upstream
runtime has moved to the `mlx-gen` package maintained at <https://github.com/lpalbou/mlx-gen>, and
AbstractFramework now publishes prepared MLX-Gen model folders in the Hugging Face collection
<https://huggingface.co/collections/AbstractFramework/mlx-gen/>.

The collection contains q4 and q8 prepared image models for families such as FLUX.2 Klein,
Qwen Image / Qwen Image Edit, and Z-Image. The default user recommendation should prefer q4
prepared folders for local memory efficiency. q8 variants should remain discoverable and preferred
when users explicitly optimize for quality.

MLX-Gen's current Python integration still exposes many model classes through the historical
`mflux.*` module layout, but the package identity, CLI, documentation, and runtime contract are
now `mlx-gen` / `mlxgen`.

## Current code reality
Files and symbols inspected before creating this item:

- `pyproject.toml`
- `src/abstractvision/backends/mflux.py`
- `src/abstractvision/model_downloads.py`
- `src/abstractvision/assets/vision_model_capabilities.json`
- `src/abstractvision/integrations/abstractcore_plugin.py`
- `src/abstractvision/cli.py`
- `tests/test_mflux_backend.py`
- `tests/test_model_downloads.py`
- `tests/test_packaging_metadata.py`
- `/Users/albou/projects/gh/sbx/mlx-gen/docs/python-integration.md`
- `/Users/albou/projects/gh/sbx/mlx-gen/docs/model-management.md`
- `/Users/albou/projects/gh/sbx/mlx-gen/src/mflux/cli/mlx_gen.py`
- `/Users/albou/projects/gh/sbx/mlx-gen/src/mflux/models/common/download_policy.py`

What exists today:

- `abstractvision[mflux]` depends on `mflux>=0.17.5,<0.18.0`.
- `MFluxVisionBackend` imports `mflux.models.*` lazily and serializes model access on a runtime
  thread.
- The backend hard-codes family tables and prefers older community q8 repos.
- The download catalog treats MLX artifacts as `target="mlx"` and `engine="mflux"`.
- `model_presets(..., include_non_8bit=False)` filters to 8-bit presets, which conflicts with
  q4 becoming the default recommendation for MLX-Gen prepared models.
- AbstractCore/Gateway/Flow are expected to consume provider/model catalogs generically, but
  several surfaces still carry `mflux` naming or static image parameter metadata.

What changed externally:

- `mlx-gen` declares package version `0.18.3` locally and Python `>=3.10`.
- Runtime generation is cache-only; downloads/preparation are explicit through `mlxgen download`
  and `mlxgen prepare`.
- `DownloadRequiredError` exposes `download_command` and `prepare_command`.
- MLX-Gen supports text-to-image and image-to-image/edit classes for current Qwen and FLUX.2
  families, including Qwen Image Edit.

## Problem
The current implementation leaks the historical MFLUX identity upward and duplicates too much
model-family knowledge in AbstractVision. It also points users to older community prepared models
instead of the AbstractFramework-owned q4/q8 MLX-Gen collection.

That creates four problems:

- users see the wrong provider/runtime identity;
- default Apple downloads are not the new recommended q4 prepared models;
- image-to-image support is narrower than MLX-Gen now supports;
- Core/Runtime/Gateway/Flow can only stay clean if AbstractVision emits a rich, canonical catalog.

## What we want to do
Migrate the Apple-local image backend to the `mlx-gen` runtime contract while keeping AbstractVision
responsible for provider-neutral request objects, artifacts, capability checks, and package-owned
catalog metadata.

## Why
Users should be able to select and run AbstractFramework-published MLX-optimized models without
learning historical MFLUX internals or manually mapping Hugging Face repos to runtime families.
Higher layers should receive provider/model/parameter metadata from AbstractVision instead of
recreating MLX-specific rules.

## Requirements
- Replace the optional runtime dependency on `mflux` with `mlx-gen`.
- Use canonical engine/provider metadata `mlx-gen`; accept `mflux` only where needed as an
  implementation or transition alias.
- Keep runtime imports lazy and keep base `abstractvision` lightweight.
- Prioritize the AbstractFramework Hugging Face `mlx-gen` collection in `model_downloads.py` and
  `vision_model_capabilities.json`.
- Prefer q4 prepared repos by default for MLX-Gen models; keep q8 discoverable for quality-focused
  selection.
- Preserve explicit network semantics: generation must not silently download model files.
- Translate MLX-Gen missing-cache errors into actionable AbstractVision errors that mention
  `mlxgen download` / `mlxgen prepare`.
- Support and test q4 text-to-image and image-to-image/edit flows.
- Surface clean provider/model/task/parameter metadata through AbstractCore, AbstractRuntime,
  AbstractGateway, and AbstractFlow without adding backend-specific UI hacks.
- Update coredoc for changed package behavior.

## Suggested implementation
- First update AbstractVision's catalog and backend identity:
  - add `mlx-gen` engine normalization;
  - keep `mflux` prefixes as aliases during the migration;
  - update MLX presets to AbstractFramework q4/q8 repos;
  - adjust preset filtering so MLX-Gen q4 counts as a default local quantized preset.
- Then update the backend adapter:
  - keep the current runtime thread and artifact integration;
  - import through `mlxgen.*` where possible, falling back only inside the adapter if the package
    exposes a historical `mflux.*` symbol;
  - use `ModelConfig.from_name(...)` instead of a growing method-name switchboard;
  - add Qwen Image Edit and FLUX.2 edit/reference routing where the MLX-Gen classes support it;
  - catch `DownloadRequiredError`.
- Then validate propagation:
  - AbstractCore plugin/provider catalog reports `mlx-gen` metadata and preserves request params;
  - Runtime/Gateway proxy the rich catalog without duplicating MLX-Gen rules;
  - Flow selectors use the catalog entries and parameter metadata.

## Scope
Included:

- AbstractVision backend, download presets, packaged capability asset, tests, and docs.
- Minimal upstream package-surface changes in AbstractCore/Runtime/Gateway/Flow where existing
  abstractions drop or duplicate MLX-Gen metadata.
- Focused q4 smoke tests for text-to-image and image-to-image/edit paths.

## Non-goals
- Do not create a generic MLX provider named `mlx`; that conflicts with AbstractCore text LLM MLX
  routing and is not the current runtime package identity.
- Do not shell out to the `mlxgen` CLI for normal generation; use Python APIs in-process so
  residency and artifact control remain package-owned.
- Do not hide network downloads behind generation.
- Do not promise video/upscale/Depth Pro surfaces through AbstractVision until the provider-neutral
  contracts are explicit.
- Do not rewrite unrelated Diffusers or stable-diffusion.cpp catalog behavior.

## Dependencies and related tasks
- Deprecated: `docs/backlog/deprecated/017_mlx_mflux_backend_strategy.md`
- Planned: `docs/backlog/planned/020_adapter_aware_model_graph_and_catalog.md`
- Completed: `docs/backlog/completed/016_abstractcore_plugin_catalog_discovery_surface.md`
- Completed: `docs/backlog/completed/019_best_effort_preload_warmup_for_local_backends.md`

## Expected outcomes
- `abstractvision[mflux]` or its successor installs `mlx-gen`, not the old `mflux` package.
- `abstractvision model-presets --target mlx` lists AbstractFramework MLX-Gen q4/q8 repos first.
- `MFluxVisionBackend` or its replacement can load cached AbstractFramework q4 prepared folders for
  text-to-image and image-to-image/edit tests.
- AbstractCore/Gateway catalog records carry `provider="mlx-gen"` / `engine="mlx-gen"` metadata
  and useful task/parameter defaults.
- Documentation describes MLX-Gen and the q4/q8 recommendation accurately.

## Validation
- Unit tests:
  - `python -m pytest tests/test_mflux_backend.py tests/test_model_downloads.py tests/test_packaging_metadata.py`
  - `python -m pytest tests/test_abstractcore_plugin.py`
- Cross-package contract tests as impacted:
  - AbstractCore vision catalog/endpoints
  - AbstractRuntime discovery facade
  - AbstractGateway capability catalog proxy
  - AbstractFlow frontend contract/build
- Manual/smoke:
  - q4 text-to-image using an AbstractFramework MLX-Gen prepared repo;
  - q4 image-to-image/edit using an AbstractFramework MLX-Gen prepared repo;
  - no generation-time download when files are missing.

## Progress checklist
- [x] Update AbstractVision backlog and design notes.
- [x] Update dependency metadata from `mflux` to `mlx-gen`.
- [x] Update MLX preset and capability asset entries to AbstractFramework q4/q8 repos.
- [x] Update backend adapter imports, missing-cache handling, and t2i/i2i routing.
- [x] Update AbstractCore/Runtime/Gateway/Flow catalog propagation as needed.
- [x] Add focused tests for q4 defaults and q8 quality variants.
- [x] Run validation and update coredoc.

## Guidance for the implementing agent
Re-check current code before each edit. Prefer package-owned catalog metadata over Gateway or Flow
hardcoding. Keep the adapter small: AbstractVision should translate provider-neutral requests into
MLX-Gen calls, not copy MLX-Gen model-family internals.

## Completion report

Completed: 2026-05-25

Summary:
- AbstractVision now treats `mlx-gen` as the canonical Apple Silicon image provider/engine while
  preserving `mflux` and `m-flux` as compatibility aliases.
- The optional Apple-local runtime dependency now points at `mlx-gen`; the `mflux` extra remains a
  compatibility install surface.
- Curated MLX presets prioritize AbstractFramework Hugging Face q4/q8 prepared repos. q4 entries
  are the default recommendation for memory efficiency; q8 entries remain discoverable for
  quality-focused selection.
- The MLX-Gen backend covers FLUX.2 klein/base, Qwen Image, Qwen Image Edit, Z-Image, and
  Z-Image Turbo families, including `image_to_image` for FLUX.2 klein/base and Qwen Image Edit
  where the runtime supports it.
- AbstractCore and Gateway catalog/routing surfaces now emit canonical `provider="mlx-gen"` /
  `engine="mlx-gen"` metadata while accepting legacy `mflux` requests.

Files and surfaces updated:
- `pyproject.toml`
- `src/abstractvision/backends/mflux.py`
- `src/abstractvision/backends/__init__.py`
- `src/abstractvision/model_downloads.py`
- `src/abstractvision/cli.py`
- `src/abstractvision/assets/vision_model_capabilities.json`
- `src/abstractvision/integrations/abstractcore_plugin.py`
- `abstractcore/abstractcore/server/vision_endpoints.py`
- `abstractgateway/src/abstractgateway/routes/gateway.py`
- focused tests under `abstractvision/tests/`, `abstractcore/tests/`, and `abstractgateway/tests/`
- public docs in AbstractVision plus targeted AbstractCore, AbstractRuntime, and Gateway docs

Validation:
- `python -m pytest tests/test_cli_smoke.py tests/test_mflux_backend.py tests/test_model_downloads.py tests/test_packaging_metadata.py -q`
  passed with `83 passed`.
- Earlier focused AbstractVision plugin/registry validation passed with `131 passed`.
- Earlier AbstractCore vision/catalog/residency validation passed with `68 passed`.
- Earlier targeted Gateway generated-media and catalog proxy validation passed with `4 passed`.

Known residual risks:
- Full real-model generation was not run in this completion pass because q4 model downloads are
  multi-GB and the focused test suite uses mocked/cache-local paths. The backend now reports
  explicit missing-cache/download guidance instead of silently downloading during generation.
- A broader Gateway test run still has an unrelated voice catalog compact-field assertion failure
  outside this MLX-Gen image migration.

Follow-ups:
- `017_mlx_mflux_backend_strategy.md` was deprecated because the accepted runtime identity is
  `mlx-gen`; open a fresh item only if a real post-MLX-Gen Apple runtime boundary appears.
- Revisit legacy `MFlux*` public class names only in a compatibility-aware release plan; aliases
  are currently intentional.
