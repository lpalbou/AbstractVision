# ADR 0005: Own a curated capability registry and cache-backed model catalog

Status: Accepted.

## Context

AbstractVision does more than expose a generic image API. It also ships a packaged registry,
download presets, model-catalog commands, and compatibility-oriented docs. Users therefore treat
the registry and catalog as product surfaces, not as incidental internal metadata.

The failure mode is predictable when this policy is left implicit:

- host-specific ports crowd out the real parent model family;
- catalog entries imply local support that the backend does not actually provide;
- official upstream weights, runtime-native community conversions, and raw side artifacts get
  mixed together without clear provenance;
- users must manually assemble hidden companion files after following a “curated” flow.

The current code already has the ingredients for a better contract:

- [`vision_model_capabilities.json`](../../src/abstractvision/assets/vision_model_capabilities.json)
  is packaged and validated;
- [`model_downloads.py`](../../src/abstractvision/model_downloads.py) drives curated download
  surfaces and host-aware selection;
- curated `sdcpp` bundle resolution now maps model keys to their required companion artifacts.

## Decision

AbstractVision owns a curated, package-level compatibility inventory and download surface.

The rules are:

1. The packaged capability registry is the authoritative source of truth for parent model families,
   task metadata, and curated download variants.
2. The registry stays global and platform-neutral. Host-specific recommendation belongs in catalog
   and preset selection logic, not in the model-family inventory itself.
3. Each real model family should have one parent entry. Engine-specific variants, quantizations,
   and runtime-native ports belong under that family.
4. Official upstream repositories are preferred. Community repositories are allowed only when they
   provide the best runtime-native artifact for a supported engine, and they must carry clear
   provenance notes.
5. Curated user flows must minimize undocumented manual assembly. If a curated local runtime path
   needs companion artifacts such as VAE or encoder files, AbstractVision should own the mapping
   and resolution for that flow.
6. Readiness claims must match shipped backend reality. Catalog-only or download-only entries are
   allowed, but they must be labeled honestly.
7. The registry and curated download surface must stay aligned. A curated model path should have a
   realistic runtime story, not just a repo id.

## Consequences

### Positive

- Users get a clearer discovery and download story.
- Parent families, runtime-native ports, and side artifacts are modeled more honestly.
- The package can keep improving local UX without turning the registry into an unstructured repo
  dump.

### Negative

- Curating the catalog becomes an explicit maintenance obligation.
- Adding new model families requires provenance review and runtime validation work.
- Some attractive community artifacts must stay out of the curated path until the runtime story is
  honest.

### Neutral

- This ADR does not require every model family to support every engine.
- Download-only entries can still exist when that is the truthful state.

## Enforcement

- Reviewers should reject registry or preset changes that:
  - duplicate a model family as host-specific top-level entries;
  - promote side artifacts or adapters as standalone curated models without first-class runtime
    orchestration;
  - claim runtime readiness without a matching backend path or explicit limitation;
  - use a community artifact without explaining why it is the right runtime-native choice.
- Changes to curated local flows must update both code and user-facing docs in the same pass.
- When a curated `sdcpp` or similar component-based path is added, the package should prefer
  package-owned bundle resolution over manual operator assembly.

## Validation

- Load and validate the packaged registry from a repo checkout.
- Verify representative `model-catalog`, `model-presets`, and `show-model` output.
- For changed curated download paths, verify the preset resolves the intended artifact and that the
  resulting runtime flow is either runnable or explicitly marked as not yet runnable.
- Add or keep focused tests for cache-backed bundle resolution when curated local flows depend on
  companion artifacts.

## Backlog links

- [docs/backlog/completed/003_hf_model_landscape_and_capability_registry.md](../backlog/completed/003_hf_model_landscape_and_capability_registry.md)
- [docs/backlog/completed/004_capability_schema_and_validation.md](../backlog/completed/004_capability_schema_and_validation.md)
- [docs/backlog/completed/016_abstractcore_plugin_catalog_discovery_surface.md](../backlog/completed/016_abstractcore_plugin_catalog_discovery_surface.md)
- [docs/backlog/completed/019_best_effort_preload_warmup_for_local_backends.md](../backlog/completed/019_best_effort_preload_warmup_for_local_backends.md)
- [docs/backlog/planned/020_adapter_aware_model_graph_and_catalog.md](../backlog/planned/020_adapter_aware_model_graph_and_catalog.md)

## Related

- [docs/reference/capabilities-registry.md](../reference/capabilities-registry.md)
- [docs/architecture.md](../architecture.md)
- [src/abstractvision/model_capabilities.py](../../src/abstractvision/model_capabilities.py)
- [src/abstractvision/model_downloads.py](../../src/abstractvision/model_downloads.py)
- [src/abstractvision/assets/vision_model_capabilities.json](../../src/abstractvision/assets/vision_model_capabilities.json)
