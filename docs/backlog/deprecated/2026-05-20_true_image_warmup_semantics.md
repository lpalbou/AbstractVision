# Deprecated: True Warmup Semantics For Resident Local Image Backends

## Metadata
- Created: 2026-05-20
- Status: Deprecated
- Deprecated: 2026-05-21
- Priority: P1

Historical note: the sections below preserve the original proposal shape, but this item is no
longer an active implementation target.

## Current Code Reality

`abstractvision` already has a real local residency surface:

- `load_resident_model(...)`
- backend `preload()`
- `list_loaded_models()` / `list_resident_models()`
- explicit unload

That work shipped in 0.3.7 and should remain the control plane for “keep this backend loaded in
this process”.

What it does **not** guarantee is a truly signature-exact warmed first inference.

- the shared `VisionBackend.preload()` contract is best-effort eager load/prepare;
- `loaded=true` means pinned/loaded in-process, not “next request is hot”;
- `mflux.preload()` now executes one representative warmup generation before returning;
- `huggingface_diffusers.preload()` now does the same for the common `text_to_image` pipeline;
- `stable-diffusion.cpp` python mode intentionally stays at eager model construction, and CLI mode
  still cannot retain process-local warm state across subprocesses.

2026-05-21 repo audit conclusion:

- no new load/list/unload contract is needed for Vision;
- current docs/tests already keep `loaded`/`resident` scoped to process-local loaded state;
- the stronger public `warmup` contract proposed below is still unimplemented;
- remote/OpenAI-compatible providers remain not-loaded from the local process perspective.

So there are two distinct semantics:

1. process-local residency / loaded state
2. true warmup for one backend instance and one representative request signature

The package should expose both when it can do so honestly.

## Problem

The current residency contract is useful, but it conflates “loaded” with “warmed” if callers or
docs over-interpret it.

The main goal of this follow-up is:

- achieve proper, engine-aware warmup when that is actually possible;
- report the real capability semantics cleanly so callers can tell what is only loaded vs what is
  truly warmed.

The proposal must stay narrow:

- no replacement of the residency API;
- no vague “hotness” marketing language;
- no claim that every local backend can provide the same warmup guarantees.

## Recommended Proposal

### 1. Keep residency semantics stable

Do not change the meaning of:

- `loaded`
- `state`
- `load_id`
- `backend_kind`
- current unload behavior

This item is a narrow follow-up to the completed residency work, not a redesign of it.

### 2. Extend `load_resident_model()` with optional warmup intent

Keep the current preload-only path as the default.

- omitting `warmup_request` means preload-only residency;
- when `warmup_request` is present, v1 still targets `text_to_image` only and should reject
  `image_to_image` warmup;
- `warmup_request: {}` means “run the backend's default representative text-to-image warmup
  profile”;
- backends may optionally accept a very small bounded request shape in v1, for example:

```json
{
  "provider": "mflux",
  "model": "flux2-klein-9b",
  "warmup_request": {
    "width": 1024,
    "height": 1024
  }
}
```

Do not expose prompt, seed, format, arbitrary `extra`, or backend construction overrides through
this public warmup input.

### 3. Make warmup backend-owned, not plugin-invented

The plugin should not invent a second hidden generation API with fake prompts or ad hoc request
rules.

Instead, add an optional backend-owned true-warmup hook for local engines, for example:

```python
def warmup_text_to_image(...) -> VisionWarmupResult: ...
```

The exact symbol can change, but the contract should be:

- the backend decides what work is required to make one normalized text-to-image signature truly
  warm;
- the backend may internally execute a representative inference and discard the output, or use a
  stronger equivalent mechanism if it can prove the same or better result;
- the backend returns structured warmup result data, including the canonical warmed signature and
  support status it actually exercised;
- the plugin only orchestrates lifecycle, records the result, and exposes it through the existing
  resident-model surface.

This keeps “true warmup” tied to the engine that actually knows whether retained benefit is real.

### 4. Keep warmup reporting additive and nested

Do not replace the current loaded-model record.

Add a nested `warmup` object only:

```json
{
  "warmup": {
    "support": "unsupported|signature_exact",
    "state": "cold|ready|failed",
    "signature": {
      "task": "text_to_image",
      "width": 1024,
      "height": 1024
    },
    "source": "explicit|request",
    "attempted_at": 0,
    "error": null
  }
}
```

Rules:

- `support=unsupported` means this backend/mode cannot honestly report true warmup readiness;
- `support=signature_exact` means the backend asserts retained benefit for exactly the recorded
  canonical signature on this resident backend instance;
- `state=ready` applies only to the recorded signature, not to all future request shapes;
- `state=failed` does not roll back residency;
- warmup metadata must be cleared when the resident backend instance is unloaded or recreated.

The keys inside `signature` are backend-defined but must be JSON-safe and stable within that backend
family. MFLUX should at minimum include `task`, `width`, and `height`, and may include other
normalized backend-relevant fields only if they materially affect retained warmness.

### 5. Initial engine scope: backend-owned reporting first

V1 true warmup should be restricted to engines that can honestly retain benefit **and** serialize
execution safely.

Initial targets should be whichever local backends can return a defensible signature result without
changing the residency contract:

- `mflux`: likely first target if it can report the exact width/height/model signature it exercised;
- `diffusers`: acceptable only if the backend proves retained benefit for a normalized request
  signature and uses the same lifetime protections as generation.

Remain preload-only / unsupported in v1:

- `stable-diffusion.cpp` CLI mode: unsupported, because each request shells out to a fresh process
- `stable-diffusion.cpp` python mode: defer until retained warm benefit and execution safety are
  proven

This is intentionally narrower than “all local backends”. The package should not claim parity where
it cannot yet defend the semantics.

### 6. Request-driven warming should be reflected when the backend can prove it

If a backend with `signature_exact` support completes a real ordinary local request successfully,
the loaded-model record should update the same nested `warmup` object with:

- `source: "request"`
- the exact canonical signature that was actually exercised

This keeps reporting honest after a real first inference, not only after explicit warmup.

### 7. Warmup must use the same lifetime protections as generation

Even though v1 can stay synchronous from the caller's perspective, internal coordination still
matters.

Implementation constraints:

- warmup must participate in the same backend lifetime/refcount protections as generation;
- warmup must **not** be implemented by reusing the public `t2i()` request path, because that would
  mutate active-routing state and request bookkeeping;
- warmup needs a private operation/load-generation token so stale completions cannot write `ready`
  or `failed` metadata onto a record that was unloaded or replaced mid-flight;
- repeated identical warmups should single-flight per resident backend instance when practical;
- warmup must not persist artifacts or create extra loaded-model records.

## Non-Goals

- No redesign of the completed residency API.
- No top-level `warm_level`.
- No `prepared` middle state.
- No schema replacement for loaded-model records.
- No `image_to_image` warmup in v1.
- No prompt/seed/format/raw-extra public warmup payload.
- No promise that every local backend will eventually support true warmup.
- No claim that `resident=true` means a generic hot first request.

## Success Criteria

- `load_resident_model()` without `warmup_request` preserves the current preload-only behavior.
- The package can distinguish “resident” from “true warmed” semantics clearly.
- MFLUX can warm one representative text-to-image signature and report it honestly.
- Warmup readiness is tied to the recorded signature and resident backend instance, not to a vague
  model-global “hot” flag.
- Warmup failure leaves the model resident and reports failure explicitly.
- Warmup metadata is cleared on unload/recreation and is not left behind by stale completions.
- Unsupported engines continue to expose residency only, without pretending to have true warmup.

## Validation Ideas

- Keep the existing residency plugin tests as compatibility guards.
- Add plugin tests that preload-only residency leaves `warmup.state=cold` or no successful warmup
  recorded.
- Add plugin tests that explicit warmup updates the resident record without storing artifacts.
- Add plugin tests that warmup failure returns a still-resident model plus explicit failure
  metadata.
- Add plugin tests that unload/reload clears previous warmup metadata.
- Add MFLUX tests proving preload-only does not hit the real generation path, while true warmup
  does.
- Add MFLUX tests proving the backend returns the canonical warmed signature it actually exercised.
- Add concurrency tests for unload/reload races so stale warmup completions cannot resurrect or
  overwrite old metadata.

## Deprecation report

Date: 2026-05-21

- The practical warmup goal behind this proposal was addressed by
  `docs/backlog/completed/019_best_effort_preload_warmup_for_local_backends.md`: `mflux` and
  `huggingface_diffusers` now run real preload warmup, while `stable-diffusion.cpp` intentionally
  stops at the strongest defensible eager preparation for its runtime model.
- The remaining work here is a new public semantics/reporting contract (`warmup_request`,
  backend-owned warmup results, and nested `warmup` metadata on loaded-model records). That
  contract is not present in `load_resident_model(...)`, `list_loaded_models()`, or the current
  integration tests.
- Current package docs already describe residency as process-local loaded state rather than a true
  warmness guarantee, so keeping this as active backlog would mostly invite premature API expansion
  rather than close a live product gap.
- If a future caller needs signature-exact warmup observability, create a new proposed item from
  the then-current code and benchmark reality instead of reviving this pre-019 design unchanged.
