# Completed: Capability-level residency hooks for warm image models

## Metadata
- Created: 2026-05-19
- Status: Completed
- Completed: 2026-05-19
- Priority: P1

## Current Code Reality

AbstractVision already has most of the low-level primitives needed for model residency:

- `VisionBackend.preload()` and `VisionBackend.unload()` exist on the shared backend contract;
- `MFluxVisionBackend`, `HuggingFaceDiffusersVisionBackend`, and `StableDiffusionCppVisionBackend`
  implement best-effort preload/unload behavior;
- the AbstractCore plugin caches routed backends in `_routed_backends`;
- repeated requests for the same provider/model within one long-lived
  `_AbstractVisionCapability` instance already reuse the same backend object, so same-model warm
  reuse exists implicitly today;
- the plugin already exposes `list_provider_models()` / `list_available_models()`, but that is
  provider catalog discovery, not resident-model listing;
- `PlaygroundState` already has explicit `load_model()`, `active_snapshot()`, and `unload_active()`
  for a single active model in the local playground surface.

What is still missing is a stable capability-level control surface that AbstractCore can use to
preload, inspect, and unload image models intentionally.

## Evaluation

This proposal is useful, especially for:

- long-lived Core worker processes that want deterministic warm local image models;
- local backends with expensive first-load cost (`mflux`, `diffusers`, `sdcpp`);
- operational control and debugging (`loaded_at`, `last_used_at`, explicit unload).

The current gap is narrower than “AbstractVision cannot keep models warm”:

- warm reuse for the same model already happens implicitly inside a single capability instance;
- the real missing piece is explicit control and honest observability across requests and model
  switches.

The original proposal is directionally correct, but it needs tightening:

- `_routed_backends` alone is not enough to answer “what is resident?” because a cached backend may
  already have been `unload()`ed;
- the plugin currently unloads the previous backend when generation switches to a different
  backend/model, so a preload-only API would be undone unless the switch behavior also changes;
- the example task and model ids do not match current AbstractVision terminology
  (`text_to_image` / `image_to_image`, and real image model ids such as `flux2-klein-4b`);
- exposing an internal `cache_key` as part of the public API would be brittle;
- allowing arbitrary per-request backend options such as `device`, `dtype`, or `quantize` in the
  first version would conflict with current backend-construction and cache-key semantics.

## Problem

Core wants generic `/acore/models/load`, `/acore/models/loaded`, and `/acore/models/unload`
behavior across modalities. For image generation, Core should not:

- reach into AbstractVision private plugin fields;
- guess whether a cached backend object is still loaded;
- depend on provider catalog listing as a proxy for warm model state.

Without an explicit plugin-owned residency surface:

- loaded state remains implicit and process-private;
- Core cannot reliably list which local image models are intentionally kept warm;
- a model preloaded for one request may be unloaded on the next model switch;
- long-lived worker processes cannot expose a clean, serializable residency contract.

## Recommended Proposal

Add explicit residency hooks to the AbstractVision AbstractCore capability object:

```python
def load_resident_model(request: Mapping[str, Any]) -> Mapping[str, Any]: ...
def list_resident_models(filters: Mapping[str, Any] | None = None) -> list[Mapping[str, Any]]: ...
def unload_resident_model(request: Mapping[str, Any]) -> Mapping[str, Any]: ...
```

These methods should be public plugin methods that Core calls through the same capability object it
already uses for generation and provider-catalog discovery.

### Request normalization

The first pass should accept only the fields that AbstractVision can key reliably today:

```json
{
  "task": "text_to_image",
  "provider": "mflux",
  "model": "flux2-klein-4b"
}
```

Notes:

- `task` may be accepted for Core protocol alignment and validation, but it should not be part of
  the residency identity in v1 because current local backends use the same loaded weights for both
  `text_to_image` and `image_to_image`;
- routed ids such as `mflux/flux2-klein-4b` or raw Diffusers ids such as
  `runwayml/stable-diffusion-v1-5` may still be accepted as convenience input, but the plugin
  should normalize them to one canonical `load_id`;
- do not add per-request backend-config overrides such as `device`, `dtype`, or `quantize` in v1.
  Those are currently process/config driven, and supporting them correctly would require changing
  backend construction and cache-key semantics. That can be a follow-up only if Core proves it
  needs multiple resident variants of the same model in one process.

### Public state shape

Return stable API fields, not plugin-internal tuple keys:

```json
{
  "task": "text_to_image",
  "provider": "mflux",
  "model": "flux2-klein-4b",
  "load_id": "mflux/flux2-klein-4b",
  "backend_kind": "mflux",
  "scope": "process",
  "state": "resident",
  "resident": true,
  "source": "explicit_preload",
  "loaded_at": 0,
  "last_used_at": 0,
  "error": null
}
```

Keep any backend cache key internal to the plugin.

### Internal design

The plugin should keep using `_routed_backends` as the backend object cache. It should not add a
second backend-object cache.

It should add a separate residency metadata sidecar, for example:

- `_resident_models: dict[backend_key, ResidentModelRecord]`
- `_active_request_backend_key: Optional[backend_key]`
- a small lock around residency state
- if concurrent generation/unload is expected, the same refcount/retire pattern already used by
  `PlaygroundState`

The key distinction is:

- `_routed_backends` answers “which backend objects have been constructed?”
- `_resident_models` answers “which models are intentionally kept warm right now?”

If both the playground and plugin continue to grow residency behavior, extract a small
package-owned helper after the first plugin implementation rather than duplicating the state machine
twice.

### Generation interaction

`load_resident_model()` should:

- resolve the same backend key used by generation;
- call backend-native `preload()`;
- record `loaded_at` and `last_used_at`;
- not generate an image.

`t2i()` / `i2i()` should:

- preserve today’s same-model reuse behavior;
- update `last_used_at` for resident entries;
- stop auto-unloading a backend on model switch if that backend was explicitly loaded as resident;
- continue auto-unloading non-resident backends on switch so current memory behavior remains
  conservative by default.

This is the main adjustment missing from the original proposal. Without it, a successful preload
would be undone as soon as the next request activates a different backend.

### Listing semantics

Expose two views:

- `list_loaded_models()` for all plugin-visible local loaded models in the current process;
- `list_resident_models()` for the explicit pinned subset.

That distinction is useful because AbstractVision already has opportunistic same-model warm reuse for
the active backend, even when a model was not explicitly preloaded.

`list_loaded_models()` / `list_resident_models()` should report plugin-owned, process-local state
only.

It should not inspect backend private fields such as `_model`, `_pipelines`, or `_py_model` to
guess whether a backend is loaded. A later optional backend hook such as `residency_info()` can be
added only if plugin-owned metadata proves insufficient.

### HTTP backends

OpenAI/OpenAI-compatible HTTP backends should be out of scope for this surface, even when the
endpoint is on `localhost`.

- `load_resident_model()` / `unload_resident_model()` should fail clearly for HTTP backends;
- `list_loaded_models()` / `list_resident_models()` should report only in-process local backends
  controlled by this plugin.

This keeps the contract honest. AbstractVision can control its own local backend objects; it cannot
prove or orchestrate model residency inside another server process over a generic OpenAI-shaped API.

## Worker Compatibility

The residency hooks should work inside a long-lived worker process, but the contract should be
explicitly scoped:

- residency is process-local, not cluster-global;
- responses must be deterministic and JSON-serializable;
- AbstractVision owns model-residency reporting, not worker lifecycle or orchestration policy.

## Non-Goals

- No cross-process or cluster-wide residency registry in v1.
- No attempt to pin models inside remote providers that expose only OpenAI-compatible inference.
- No automatic model selection from provider catalogs.
- No `VisionManager` public API expansion in the first pass.

## Success Criteria

- AbstractCore can warm a local AbstractVision model without private-field reach-through.
- AbstractCore can inspect both explicitly resident models and transient currently loaded local
  models.
- AbstractCore can list explicitly resident image models with stable provider/model/load metadata.
- Explicitly resident models survive generation requests that switch to other models until they are
  unloaded.
- Non-resident models keep the current unload-on-switch behavior.
- OpenAI-compatible HTTP backends are rejected for residency control rather than being reported as
  falsely resident.
- The implementation does not depend on backend private fields to report residency.

## Validation Ideas

- Unit tests for residency request normalization and canonical `load_id` generation.
- Plugin tests that repeated same-model generation still reuses the same backend object.
- Plugin tests that request-driven warm local backends appear in `list_loaded_models()`.
- Plugin tests that an explicitly resident backend is not unloaded when another model is generated.
- Plugin tests that `unload_resident_model()` clears metadata and calls `unload()`.
- Plugin tests that OpenAI-compatible HTTP backends are rejected for residency control.
- If concurrency is supported, tests that in-flight generation is not unloaded mid-request.

## Completion Report

Date: 2026-05-19

### Summary

- Added explicit process-local residency control to the AbstractCore plugin through
  `load_resident_model()`, `list_loaded_models()`, `list_resident_models()`, and
  `unload_resident_model()`.
- Preserved conservative unload-on-switch behavior for non-resident local backends while keeping
  explicitly resident backends loaded across model switches.
- Reported stable plugin-owned residency metadata (`load_id`, `backend_kind`, `resident`, `state`,
  timestamps, observed tasks) without depending on backend private fields.
- Rejected OpenAI/OpenAI-compatible HTTP backends for residency control so the contract stays
  honest about what AbstractVision can actually keep loaded in-process.
- Kept the public residency surface process-local and JSON-safe for Core/Gateway-style route
  adapters.

### Files And Symbols Touched

- `src/abstractvision/integrations/abstractcore_plugin.py`
  - `_AbstractVisionCapability.load_resident_model`
  - `_AbstractVisionCapability.list_loaded_models`
  - `_AbstractVisionCapability.list_resident_models`
  - `_AbstractVisionCapability.unload_resident_model`
  - `_AbstractVisionCapability.unload_model`
  - `_AbstractVisionCapability._record_loaded_model`
  - `_AbstractVisionCapability._activate_request_backend`
  - `_AbstractVisionCapability._find_loaded_backends`
  - `_AbstractVisionCapability._normalize_loaded_filters`
- `tests/test_abstractcore_plugin.py`
- `docs/reference/abstractcore-integration.md`
- `CHANGELOG.md`

### Behavior Changes

- Core/plugin callers can now preload, inspect, and unload local resident image backends
  intentionally instead of relying only on implicit same-model reuse.
- `list_loaded_models()` now distinguishes explicitly resident entries from transient request-loaded
  local backends.
- Explicitly resident local backends survive switches to other models until they are explicitly
  unloaded.
- OpenAI/OpenAI-compatible backends are refused for residency control, even when pointed at
  localhost.

### Validation

- Historical shipped validation is reflected in the 0.3.7 changelog entry and the plugin test
  coverage added with that release.
- `PYTHONPATH=src python -m unittest tests.test_abstractcore_plugin -q` passed on 2026-05-20
  during backlog verification, 33 tests.
- `git diff --check` passed on 2026-05-20 during backlog verification.

### Docs

- Public integration docs describe the residency control surface and its process-local scope.
- Changelog records the 0.3.7 residency feature and its semantics.

### Residual Risks

- The current residency surface is honest about loaded state, but it does not prove that a first
  matching inference is fully warmed. That follow-up belongs to
  `docs/backlog/proposed/2026-05-20_true_image_warmup_semantics.md`.
- Residency is process-local only; cluster-wide orchestration and worker lifecycle policy remain
  downstream concerns.

### Backlog / Code Drift Found

- This item remained in `docs/backlog/proposed/` even though the code, tests, docs, and changelog
  already showed it was complete.
- The item has now been moved to `docs/backlog/completed/` and its completion report recorded.

### Follow-ups

- Keep the narrower true-warmup follow-up proposed separately.
- If the repo later adds backlog overview or recurrent hygiene files, this completion should be
  reflected there as well.

### Priority Impact

- This no longer competes for implementation priority.
- Remaining priority is on whether true warmup semantics and stronger engine-specific warmup are
  worth adding beyond current residency.
