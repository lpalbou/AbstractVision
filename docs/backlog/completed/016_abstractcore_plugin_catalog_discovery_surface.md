# Completed: Expose Provider Model Discovery Through AbstractCore Plugin

## Metadata
- Created: 2026-05-08
- Status: Completed
- Completed: 2026-05-08

## Context

AbstractVision now has an explicit provider catalog abstraction:

- `OpenAICompatibleVisionBackend.list_provider_models(...)`;
- `VisionManager.list_provider_models(...)`;
- CLI support through `abstractvision provider-models`.

That is the right package-owned direction. The remaining integration question is whether
AbstractCore can access the same discovery through the capability plugin boundary.

## Current Code Reality

- The AbstractCore plugin registers `abstractvision:openai` and the legacy
  `abstractvision:openai-compatible` backend id.
- `_AbstractVisionCapability` exposes generation methods (`t2i`, `i2i`, `t2v`, `i2v`).
- `_AbstractVisionCapability` does not currently expose `list_provider_models(...)`.
- Core Server would need either a private `_get_backend()` reach-through or a direct AbstractVision
  import to surface provider catalog routes.

## Problem

If Core Server must expose dynamic image model catalogs, the clean boundary is the capability plugin,
not package internals. Without a public plugin method, Core either cannot implement the route cleanly
or must depend on private AbstractVision implementation details.

## Proposed Direction

Add an optional public discovery method to the AbstractCore plugin shim:

```python
def list_provider_models(self, *, task: str | None = None) -> list[dict]:
    ...
```

The method should:

- call the configured backend's `list_provider_models(task=task)`;
- serialize `ProviderModelInfo` values into JSON-safe dictionaries;
- preserve raw provider metadata in a bounded `raw` field;
- fail clearly when the selected backend does not support provider catalogs;
- not change or auto-select the active model.

## Non-Goals

- Do not make provider catalog listing automatic model selection.
- Do not move Core Server route code into AbstractVision.
- Do not add Gateway auth/CORS/server policy to AbstractVision.
- Do not add Apple/GPU profile behavior as part of this discovery surface. Hardware-profile aliases
  are handled by the package install-profile work.

## Promotion Criteria

Promote before releasing AbstractVision as the package version that Core/Gateway depend on for
dynamic provider model discovery, unless maintainers explicitly accept a Core-side private adapter.

## Validation Ideas

- Unit test `_AbstractVisionCapability.list_provider_models(...)` with a fake backend.
- Test JSON-safe serialization of `ProviderModelInfo`.
- Test explicit failure for local backends that do not support provider catalogs.
- Keep existing generation plugin tests passing.

## Guidance For Implementing Agents

Keep this small. The backend and manager already own discovery; the plugin only needs to expose it
through the same boundary Core uses for generation.

## Completion Report

Date: 2026-05-08

### Decision

This backlog should be done. The provider catalog abstraction already exists in the backend,
manager, and CLI, but Core/Gateway should not need to call `_get_backend()` or import package
internals to expose a provider model route. A small plugin shim is the clean boundary.

### Summary

- Added `_AbstractVisionCapability.list_provider_models(task=...)`.
- Delegated catalog listing to the configured backend without changing the active generation model.
- Serialized `ProviderModelInfo` entries into JSON-safe dictionaries for Core route handlers.
- Preserved provider metadata in a bounded `raw` field with explicit truncation markers.
- Added clear failure for configured backends that inherit the no-op provider catalog method.
- Wired `vision_models_path` / `ABSTRACTVISION_MODELS_PATH` through plugin backend construction.
- Documented the plugin catalog surface and regenerated agent-facing docs.

### Files and Symbols Touched

- `src/abstractvision/integrations/abstractcore_plugin.py`
  - `_AbstractVisionCapability.list_provider_models`
  - `_provider_model_to_dict`
  - `_json_safe_provider_value`
  - `_backend_supports_provider_catalog`
- `tests/test_abstractcore_plugin.py`
- `README.md`
- `docs/api.md`
- `docs/reference/abstractcore-integration.md`
- `docs/reference/backends.md`
- `docs/reference/configuration.md`
- `llms.txt`
- `llms-full.txt`

### Validation

- `PYTHONPATH=src python -m unittest tests.test_abstractcore_plugin -q` passed, 15 tests.
- `python -m ruff check --ignore UP src/abstractvision/integrations/abstractcore_plugin.py tests/test_abstractcore_plugin.py` passed.
- `python scripts/generate_llms_full.py` regenerated `llms-full.txt`.
- `PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -q` passed, 88 tests.
- `python -m ruff check --ignore UP src/abstractvision/types.py src/abstractvision/backends/base_backend.py src/abstractvision/backends/openai_compatible.py src/abstractvision/vision_manager.py src/abstractvision/cli.py src/abstractvision/integrations/abstractcore_plugin.py tests/test_openai_compatible_backend.py tests/test_cli_smoke.py tests/test_manager_capability_checks.py tests/test_abstractcore_plugin.py` passed.
- `mkdocs build -q` passed. MkDocs Material emitted its upstream MkDocs 2.0 compatibility warning.
- `git diff --check` passed.

### Residual Risks

- Core/Gateway route implementation and readiness policy still belong downstream. AbstractVision now
  provides the plugin boundary needed for that work.
- Provider catalog metadata schemas vary by provider. The plugin preserves raw diagnostics, but
  callers should not treat provider catalog metadata as the packaged capability registry.
