# ADR 0004: Keep the orchestrator thin and make model semantics backend-owned

Status: Accepted.

## Context

AbstractVision exposes several public surfaces: Python API, CLI/REPL, playground, and AbstractCore
plugin integration. Those surfaces all touch the same runtime families, but they should not each
reinvent model-specific rules.

The current code is already moving toward a clean boundary:

- [`VisionManager`](../../src/abstractvision/vision_manager.py) stays thin and delegates execution;
- backends expose `normalize_image_generation_request(...)` and
  `normalize_image_edit_request(...)`;
- the CLI, playground, and AbstractCore plugin route requests through that shared backend-owned
  normalization path;
- residency and preload behavior are package/backend concerns, not AbstractCore-native semantics.

Without a durable rule, model-specific constraints, overlay behavior, component assembly, or
warmup/residency semantics will drift across surfaces and become inconsistent.

## Decision

AbstractVision keeps orchestration thin and treats model/runtime semantics as backend-owned package
behavior.

The rules are:

1. `VisionManager` owns request construction, optional capability gating, artifact storage, and
   backend delegation. It does not become the place where model-family rules are reimplemented.
2. Model-specific request normalization belongs in backend code, optionally using packaged registry
   metadata as input.
3. CLI, playground, and AbstractCore integration must go through the backend-owned normalization
   and execution path instead of encoding their own copies of model rules.
4. Overlay, adapter, component-bundle, preload, and local-residency semantics remain
   `abstractvision` package responsibilities. AbstractCore may orchestrate them, but it must not
   become their source of truth.
5. The capability registry describes model intent and packaged metadata. Runtime support is still
   decided by the configured backend.

## Consequences

### Positive

- Model-specific behavior stays consistent across Python, CLI, playground, and AbstractCore.
- Runtime-specific fixes land close to the backend code that actually implements them.
- Package-level semantics remain easier to test end-to-end.

### Negative

- Backends carry more responsibility for normalization and compatibility logic.
- Public surfaces must resist the temptation to add their own “small” model-specific shortcuts.

### Neutral

- This ADR does not force every backend to support every task.
- This ADR allows backend-owned evolution as new model families or adapter shapes appear.

## Enforcement

- Reviewers should reject model-specific parameter hacks added only in the CLI, playground, or
  AbstractCore plugin when the same rule should live in the backend path.
- New runtime-family semantics must identify the authoritative layer explicitly.
- If a cross-surface contract changes, the change must update every surface that delegates into it
  and must update the docs in the same pass.

## Validation

- Keep manager, CLI, playground, and AbstractCore tests passing for shared normalization behavior.
- When adding model-specific constraints, add at least one backend-level test and one surface-level
  integration check when feasible.
- Verify that observed behavior matches backend reality rather than only registry metadata.

## Backlog links

- [docs/backlog/completed/005_core_api_tasks_and_abstractions.md](../backlog/completed/005_core_api_tasks_and_abstractions.md)
- [docs/backlog/completed/016_abstractcore_plugin_catalog_discovery_surface.md](../backlog/completed/016_abstractcore_plugin_catalog_discovery_surface.md)
- [docs/backlog/completed/018_capability_residency_hooks.md](../backlog/completed/018_capability_residency_hooks.md)
- [docs/backlog/completed/019_best_effort_preload_warmup_for_local_backends.md](../backlog/completed/019_best_effort_preload_warmup_for_local_backends.md)
- [docs/backlog/planned/020_adapter_aware_model_graph_and_catalog.md](../backlog/planned/020_adapter_aware_model_graph_and_catalog.md)

## Related

- [docs/architecture.md](../architecture.md)
- [docs/reference/backends.md](../reference/backends.md)
- [src/abstractvision/vision_manager.py](../../src/abstractvision/vision_manager.py)
- [src/abstractvision/backends/base_backend.py](../../src/abstractvision/backends/base_backend.py)
- [src/abstractvision/integrations/abstractcore_plugin.py](../../src/abstractvision/integrations/abstractcore_plugin.py)
