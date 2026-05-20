## ADR 0002: Vision overlays and adapter composition are package-owned

**Date**: 2026-05-20  
**Status**: Superseded by ADR 0004

---

## Note

This ADR captured one package boundary correctly, but it was too narrow and too implementation-led
to serve as the repo baseline. Keep it for history only. The active rule now lives in
[ADR 0004: Keep the orchestrator thin and make model semantics backend-owned](0004_thin_orchestrator_and_backend_owned_model_semantics.md).

## Context

`abstractvision` already has real backend-level overlay and adapter behavior, but it is not yet
documented as a durable package boundary.

Current code reality:

- The Diffusers backend parses LoRA-like overlay payloads from request data and applies them inside
  the backend implementation (`src/abstractvision/backends/huggingface_diffusers.py`).
- The MFLUX backend already has backend-owned LoRA inputs in its config/runtime path
  (`src/abstractvision/backends/mflux.py`).
- Local residency is already package-owned through the AbstractCore plugin integration
  (`src/abstractvision/integrations/abstractcore_plugin.py`).
- The model/catalog backlog already distinguishes parent models, variants, and adapter-style
  artifacts as a package concern (`docs/backlog/planned/020_adapter_aware_model_graph_and_catalog.md`).

At the same time, `abstractcore` treats vision as a capability plugin boundary, not as a native
text-provider family. That means Core should not become the source of truth for backend-specific
vision overlay semantics.

Without an ADR here, future work can drift in two bad directions:

- presenting adapter artifacts as if they were standalone runnable vision models;
- pushing backend-specific overlay rules up into `abstractcore`, where they will be less accurate
  and harder to maintain.

---

## Decision

### 1) Vision overlays/adapters are owned by `abstractvision`

Selection, validation, compatibility, and application of vision overlays remain package-owned.

This includes, for example:

- LoRA overlays for Diffusers-like backends;
- backend-specific overlay inputs such as MFLUX LoRA paths/scales;
- future adapter-style artifacts that require a specific parent model or backend family.

### 2) `abstractcore` may orchestrate, but must not define vision adapter semantics

When `abstractvision` is used through AbstractCore, Core may forward package-owned overlay payloads
and residency selectors, but it must not become the source of truth for:

- what counts as a valid overlay payload;
- what parent model a vision adapter requires;
- whether a backend can honor one overlay or many;
- how overlay application is cached, fused, or unloaded.

### 3) Catalogs must distinguish runnable models from adapter-style artifacts

A vision adapter artifact is not a first-class standalone model unless a backend can truly execute it
as one.

The package should keep separating:

- parent model family metadata;
- runtime/download variants;
- adapter or side artifacts that depend on a compatible parent.

### 4) Package-facing API should become explicit over time

The current `extra` / `loras*` request path is acceptable as an implementation bridge, but the
longer-term package contract should move toward an explicit overlay surface owned by `abstractvision`.

That can be a first-class `overlays` field, equivalent request metadata, or another package-local
contract, but the contract should live here rather than in Core.

---

## Consequences

### Positive

- Backend-specific overlay behavior stays close to the code that actually implements it.
- The package can model parent-model requirements honestly instead of flattening adapters into fake
  standalone models.
- AbstractCore integration stays simpler and more accurate.

### Negative / Risks

- The capability API exposed through AbstractCore remains somewhat asymmetric relative to text.
- Some current overlay behavior is still implicit (`extra`-driven) and needs cleanup before the
  package contract feels fully explicit.

---

## Related

- `src/abstractvision/backends/huggingface_diffusers.py`
- `src/abstractvision/backends/mflux.py`
- `src/abstractvision/integrations/abstractcore_plugin.py`
- `docs/backlog/planned/020_adapter_aware_model_graph_and_catalog.md`
- `docs/reference/abstractcore-integration.md`
