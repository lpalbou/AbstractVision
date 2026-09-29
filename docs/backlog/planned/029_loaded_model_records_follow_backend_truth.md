## Task 029: Loaded-model records follow the backend's truth (Wan TI2V release after a run)

**Date**: 2026-09-29  
**Status**: Planned  
**Priority**: P1  

---

## Main goals

- `list_loaded_models` / `list_resident_models` in the AbstractCore plugin never report a model as
  loaded when the backend no longer holds it.
- A model the operator preloaded stays honest: either it stays warm, or the record says it was
  released and why.

## Secondary goals

- One rule for every backend that can drop its model on its own (today only the MLX-Gen Wan
  TI2V-5B path does), instead of a Wan special case in the plugin.

---

## Context / problem

Since 0.3.31 the MLX-Gen backend runs Wan 2.2 TI2V-5B with
`release_denoisers_before_decode=True`: the denoiser is freed before the VAE decode (the decode is
the run's memory peak). mlx-gen cannot reload a released TI2V denoiser, so after every TI2V run
the backend forgets its model (`MfluxBackend._drop_released_wan_model`,
`src/abstractvision/backends/mflux.py:1917`, called at `:4164`) and the next request rebuilds it
(about a minute, measured on Apple silicon).

The plugin does not learn about it. Its records live in
`AbstractVisionCapabilityPlugin._loaded_models` and are only removed on an explicit unload
(`src/abstractvision/integrations/abstractcore_plugin.py:1752-1830`). After one TI2V video:

- `list_loaded_models` still returns the TI2V record with `state` loaded and, for an explicit
  preload, `resident: True`;
- AbstractCore and the gateway show the model as loaded in the console and the model manager, and
  memory reports attribute memory to a model that is gone;
- the operator who preloaded TI2V to keep it warm pays the rebuild on the second request anyway,
  with nothing saying so.

Found by the abstractvision 0.3.31 tag gate (2026-09-29, `untracked/vision-gate/` in the
framework root).

---

## Constraints

- Keep the public plugin contract (`load_resident_model`, `list_loaded_models`,
  `list_resident_models`, `unload_resident_model` and the record fields) stable; add fields, do
  not rename.
- No polling threads; the truth is read when asked or pushed when the backend changes it.
- The memory saving of the release must stay the default (it is what keeps TI2V at 832x480 near
  16.3 GiB).

---

## Research, options, and references

- **Option A: the backend reports what it holds.** Backends expose a cheap
  `is_model_loaded()` (or `loaded_model_key`); `list_loaded_models` reconciles records against it
  and drops, or marks `state: "released"`, the ones the backend no longer holds. General, no
  per-model code in the plugin.
- **Option B: the backend emits a release event.** `_drop_released_wan_model` calls a callback
  the plugin registered; the plugin updates the record. Precise, but one more channel.
- **Option C: keep a preloaded TI2V warm.** When the model was explicitly preloaded
  (`resident: True`), run without `release_denoisers_before_decode` and accept the higher peak
  (23.4 GiB at 832x480 tiled vs 16.3), or reload into the kept instance if a future mlx-gen can.
  An operator choice, not a silent default.

Recommendation: A for correctness (every backend, every call), plus C as an explicit option on the
preload request (`keep_warm`), documented with its memory cost.

---

## Acceptance criteria

- [ ] After a TI2V run, `list_loaded_models` no longer lists the model as loaded (or lists it as
      `released` with a reason); a test goes RED without the fix.
- [ ] An explicit preload with the keep-warm option keeps the model loaded across two TI2V
      requests (the second request does no rebuild), with the documented peak.
- [ ] Docs (`docs/reference/abstractcore-integration.md`, CHANGELOG) state the release behaviour and the option.

## Validation

Hermetic unit tests with a stub backend that drops its model; one real TI2V run on cached weights
under the framework GPU lock, checking the record and the second request's latency.
