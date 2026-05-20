# ADR 0001: AbstractVision Owns The Curated Cross-Platform Model Catalog And Compatibility Layer

## Status
Superseded by ADR 0005 and ADR 0006

## Note
This ADR captured the first attempt at catalog policy before the repo had a broader ADR baseline.
Keep it for history only. The active rules now live in:

- [ADR 0005: Own a curated capability registry and cache-backed model catalog](0005_curated_capability_registry_and_download_catalog.md)
- [ADR 0006: Keep runtime selection explicit and operator-controlled](0006_operator_control_configuration_precedence_and_explicit_network_use.md)

## Dates
- Proposed: 2026-05-20
- Accepted: 2026-05-20

## Areas affected
- `src/abstractvision/assets/vision_model_capabilities.json`
- `src/abstractvision/model_downloads.py`
- `src/abstractvision/cli.py`
- `src/abstractvision/backends/`
- `src/abstractvision/integrations/abstractcore.py`
- user-facing catalog, download, and runtime selection docs

## Authority owner
AbstractVision maintainers

## Adoption state
Accepted for all new model inventory and runtime integration work.

Current code only partially conforms:
- the capability registry and curated downloader already exist;
- host-aware target selection already exists;
- some registry entries are catalog-only because the local backend is not implemented yet;
- adapter-aware orchestration is not first-class yet.

This ADR is therefore binding immediately for new work and is also an adoption target for the
remaining drift.

## Context
AbstractFramework needs one stable place where users can discover compatible vision models, inspect
their task parameters, download the right artifact for their machine, and then run that model
through AbstractVision or AbstractCore without repo-specific guesswork.

Without an explicit policy, the registry drifts toward an inventory dump:
- host-specific bias leaks into the source of truth;
- standalone models, component artifacts, and adapters get conflated;
- catalog entries can imply local support that the runtime does not actually implement;
- users must manually assemble base models, adapters, or runtime-specific ports after using the
  catalog or downloader.

The product goal is different: when AbstractVision exposes a model family or downloadable runtime
variant, that surface should reduce integration effort rather than shift the burden to the user.

## Decision
AbstractVision owns the curated, cross-platform compatibility inventory for the vision models it
surfaces through AbstractFramework.

That ownership includes:
- maintaining the compatible model list;
- maintaining task metadata, parameter defaults, and model-specific constraints;
- maintaining curated download metadata for supported engines and artifact formats;
- maintaining the runtime normalization and compatibility work needed to make supported variants
  usable across Apple Silicon, NVIDIA GPU, AMD-capable GPU stacks, CPU/GGUF, and remote runtimes
  when practical.

The following rules now apply:

1. `vision_model_capabilities.json` is the authoritative source of truth for parent model families,
   task metadata, and downloadable variants.
2. The registry is global and platform-neutral. Host-specific recommendation belongs in selection
   logic such as `model-catalog` or `download-model`, not in the model inventory itself.
3. Each real model family should have one parent entry. Engine-specific artifacts, quantizations,
   and runtime-native ports belong under that parent as downloadable variants.
4. Official upstream repositories are preferred. Community repositories are allowed only when they
   provide a runtime-native or engine-specific artifact that has no official equivalent, and they
   must be labeled clearly with provenance and notes.
5. Adapter-only, component-only, or base-model-dependent repositories must not be promoted as
   standalone curated models unless AbstractVision has first-class orchestration for that shape.
6. If a model is cataloged but not runnable through a shipped backend, that limitation must be
   explicit in the registry and surfaced as a download-only or backend-not-supported state.
7. If a downloadable model variant is presented as a curated user path, the intended flow is:
   install the documented runtime extra, use `abstractvision model-catalog` or `download-model`,
   and then run the model through AbstractVision or AbstractCore without undocumented manual
   assembly steps. Known prerequisites such as provider tokens, gated-license acceptance, or
   runtime package installation are allowed, but hidden repo spelunking is not.
8. When compatibility gaps are primarily caused by our abstraction layer rather than by the
   upstream model, we should prefer rewriting our backend, downloader, or integration code to make
   the model usable across supported OS and hardware combinations instead of documenting permanent
   workarounds.
9. AbstractCore integration must consume the same curated compatibility truth and must not expose
   parameters, adapters, or ready-to-run claims that the actual runtime path cannot honor.

## Consequences
### Positive
- Users get a simpler model discovery and download path.
- The registry becomes a product surface rather than an unstructured repo list.
- Apple Silicon, GPU, GGUF, and remote runtime support can be compared under one parent model
  instead of being scattered across unrelated entries.
- Future backend or AbstractCore work has a clear bar for what “supported” means.

### Negative
- Maintaining the registry becomes an explicit engineering obligation, not opportunistic metadata.
- Some tempting Hugging Face repos will need to stay out of the curated catalog until runtime
  orchestration exists.
- New model additions require more research, provenance review, and smoke validation.
- Cross-platform support may require backend rewrites instead of doc-only patches.

### Neutral
- Download-only and remote-only entries can still exist, but they must be marked clearly.
- This ADR does not require every vision model family to support every engine.
- This ADR does not require local video runtime support today.

## Enforcement
- Changes to `vision_model_capabilities.json`, `model_downloads.py`, catalog selection, or
  AbstractCore model surfacing must follow this ADR.
- New model families must be added as parent entries with child download variants; do not create
  host-specific top-level duplicates when a parent family already exists.
- Reviewers should reject entries that:
  - prefer a host-specific port over the official upstream family in the source of truth;
  - present an adapter or component as a standalone curated model;
  - claim runtime readiness without a matching backend path or explicit limitation marker;
  - add community repos without a clear reason they are the best runtime-native choice.
- Contributor-facing docs must keep the curation rules visible in:
  - `docs/reference/capabilities-registry.md`
  - `CONTRIBUTING.md`
  - the ADR index in `docs/adr/README.md`
- When runtime behavior changes, AbstractCore integration and the catalog/download docs must be
  updated in the same pass.

## Validation
- Load the registry from the repo checkout:
  - `PYTHONPATH=src python - <<'PY'`
    `from abstractvision.model_capabilities import VisionModelCapabilitiesRegistry`
    `VisionModelCapabilitiesRegistry()`
    `print("ok")`
    `PY`
- Verify catalog output from the repo checkout:
  - `PYTHONPATH=src python -m abstractvision.cli model-catalog --json`
- Verify representative model inspection:
  - `PYTHONPATH=src python -m abstractvision.cli show-model <model_id>`
- For any changed curated download path, verify the helper resolves the intended preset or catalog
  entry for the relevant engine and target.
- When backend or AbstractCore behavior changes, run at least one end-to-end smoke path for each
  touched runtime family where feasible.

## Backlog links
- Related completed groundwork:
  - [`docs/backlog/completed/003_hf_model_landscape_and_capability_registry.md`](../backlog/completed/003_hf_model_landscape_and_capability_registry.md)
  - [`docs/backlog/completed/016_abstractcore_plugin_catalog_discovery_surface.md`](../backlog/completed/016_abstractcore_plugin_catalog_discovery_surface.md)
- Open adoption and adjacent work:
  - [`docs/backlog/planned/017_mlx_mflux_backend_strategy.md`](../backlog/planned/017_mlx_mflux_backend_strategy.md)
  - [`docs/backlog/planned/020_adapter_aware_model_graph_and_catalog.md`](../backlog/planned/020_adapter_aware_model_graph_and_catalog.md)

## Related
- [Capability registry reference](../reference/capabilities-registry.md)
- [Architecture](../architecture.md)
- [`src/abstractvision/model_capabilities.py`](../../src/abstractvision/model_capabilities.py)
- [`src/abstractvision/model_downloads.py`](../../src/abstractvision/model_downloads.py)
- [`src/abstractvision/cli.py`](../../src/abstractvision/cli.py)
- [`src/abstractvision/integrations/abstractcore.py`](../../src/abstractvision/integrations/abstractcore.py)
