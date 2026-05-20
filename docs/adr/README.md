# ADRs

This directory records durable engineering policy for `abstractvision`.

## Accepted

- [ADR 0003: Keep the base package lightweight and make local runtimes explicit](0003_lightweight_base_package_and_explicit_runtime_extras.md)
  Base installs stay dependency-light; heavy local runtimes and model-download helpers remain optional extras.
- [ADR 0004: Keep the orchestrator thin and make model semantics backend-owned](0004_thin_orchestrator_and_backend_owned_model_semantics.md)
  `VisionManager` delegates; backend normalization and runtime-specific behavior stay in the backend/package layer.
- [ADR 0005: Own a curated capability registry and cache-backed model catalog](0005_curated_capability_registry_and_download_catalog.md)
  The packaged registry and curated download surfaces are product-facing compatibility contracts, not incidental metadata.
- [ADR 0006: Keep runtime selection explicit and operator-controlled](0006_operator_control_configuration_precedence_and_explicit_network_use.md)
  Config precedence, cache-only defaults, and model selection must stay explicit across CLI, playground, and AbstractCore.
- [ADR 0007: Disclose fallbacks and degraded modes explicitly](0007_explicit_fallback_and_degraded_mode_disclosure.md)
  Fallbacks that change provenance, quality, or runtime behavior must be surfaced rather than hidden.
- [ADR 0008: Require validation and evidence-based change reporting](0008_validation_and_evidence_based_change_reporting.md)
  Backend, catalog, and performance claims need reproducible checks and clear measured-versus-estimated reporting.
- [ADR 0009: Keep docs, backlog, and ADRs code-first](0009_code_first_docs_backlog_and_adr_discipline.md)
  Code is the operational source of truth; durable policy lives here, and execution history lives in `docs/backlog/`.

## Superseded

- [ADR 0001: AbstractVision owns the curated cross-platform model catalog and compatibility layer](0001_curated_cross_platform_model_catalog.md)
  Superseded by ADR 0005 and ADR 0006.
- [ADR 0002: Vision overlays and adapter composition are package-owned](0002_vision_overlays_and_adapter_composition_are_package_owned.md)
  Superseded by ADR 0004.

## Working Rules

- Use ADRs for durable engineering policy that should constrain future changes.
- Use `docs/backlog/` for planning state, implementation history, and completion reports.
- When code, docs, backlog, and ADRs drift, fix the drift explicitly instead of relying on chat history.
