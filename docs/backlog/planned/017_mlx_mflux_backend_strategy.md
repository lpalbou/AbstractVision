## Task 017: Native MLX engine beyond MFLUX

**Date**: 2026-05-20
**Status**: Planned  
**Priority**: P1  

---

## Main goals

- Create a true first-class MLX image backend for AbstractVision that can run MLX-packaged vision
  models on Apple Silicon beyond the subset supported by MFLUX today.
- Treat MFLUX as the short-term compatibility bridge, not the long-term Apple runtime abstraction.
- Make MLX a real engine choice in AbstractVision, CLI, playground, and AbstractCore instead of a
  catalog/download target that is mostly routed through MFLUX-specific rules.

## Secondary goals

- Keep base `abstractvision` import-light and local runtimes behind optional extras.
- Preserve the `VisionManager`, artifact output, and AbstractCore capability-plugin contracts.
- Keep current MFLUX users working while the native MLX path is introduced incrementally.

---

## Context / problem

The product goal is broader than “make current MFLUX presets work.” AbstractVision is supposed to
help users discover, download, and run compatible vision models across machines and runtimes with
as little manual assembly as possible.

For Apple Silicon, the repo currently has many MLX-flavored download variants and an implemented
MFLUX backend, but it does **not** have a generic/native MLX backend. That means:

- MLX is not a true engine in the product today.
- Apple-local MLX support is limited to what the MFLUX runtime happens to support.
- Some curated MLX/MFLUX download paths are broader than what the MFLUX backend can actually run.

The correct long-term shape is:

- `mlx` becomes a first-class runtime/backend category in AbstractVision;
- `mflux` remains an adapter/bridge for the subset of models where it is the practical runtime;
- model discovery and download surfaces distinguish:
  - MLX download available;
  - runnable through MFLUX today;
  - runnable through native MLX engine;
  - download-only pending runtime support.

This task exists because Apple-local MLX support should be a real platform capability, not just a
set of MFLUX-specific exceptions.

## Current code reality

Files and symbols inspected before rewriting this item:

- `src/abstractvision/backends/mflux.py`
- `src/abstractvision/model_downloads.py`
- `src/abstractvision/playground_server.py`
- `src/abstractvision/integrations/abstractcore_plugin.py`
- `src/abstractvision/assets/vision_model_capabilities.json`
- `docs/reference/backends.md`
- `README.md`

What exists today:

- A shipped optional `MFluxVisionBackend` in `src/abstractvision/backends/mflux.py`.
- Apple-oriented curated download variants that use `target="mlx"` and `engine="mflux"`.
- AbstractCore routing that explicitly rejects generic `mlx/...` and tells callers to use
  `mflux/<preset>`.
- Playground and CLI catalog surfaces that already expose MLX-targeted downloads.

What is still wrong or incomplete:

- Generic MLX is **not** a real backend or provider today; it is still rejected in the plugin.
- The playground does not fail fast consistently on `mlx/...` model ids and can still default that
  shape to Diffusers routing.
- The MFLUX backend supports only a small hard-coded family table, while the curated MLX/MFLUX
  catalog is broader.
- There are multiple competing truths for Apple-local support:
  - registry/download metadata;
  - hard-coded preset table;
  - runtime family table in the MFLUX backend;
  - route-selection logic in CLI/playground/AbstractCore.

## Constraints

- Do not make MLX, MFLUX, Torch, Diffusers, or stable-diffusion.cpp mandatory base dependencies.
- Preserve offline generation after models are downloaded.
- Keep generated outputs as AbstractVision artifacts.
- Keep local runtime imports lazy and avoid model loads at import time.
- Prefer permissive licensing for new runtime dependencies.
- Preserve current MFLUX behavior as a compatibility bridge while the native MLX path is built.
- Follow [ADR 0005](../../adr/0005_curated_capability_registry_and_download_catalog.md) and
  [ADR 0006](../../adr/0006_operator_control_configuration_precedence_and_explicit_network_use.md):
  curated download variants should be runnable or clearly marked as download-only /
  backend-not-supported, and runtime selection must stay explicit.

---

## Research, options, and references

- **Current runtime implementation**:
  - MFLUX exists today as the only shipped Apple-local MLX-adjacent runtime path.
  - References:
    - `src/abstractvision/backends/mflux.py`
    - `src/abstractvision/backends/__init__.py`
- **Current routing reality**:
  - CLI, playground, and AbstractCore now fail fast on generic `mlx/...`; only explicit `mflux/...`
    routing is runnable today on Apple-local MLX-adjacent paths.
  - References:
    - `src/abstractvision/integrations/abstractcore_plugin.py`
    - `src/abstractvision/playground_server.py`
    - `src/abstractvision/cli.py`
- **Current catalog/download reality**:
  - Apple auto-targeting prefers `mlx`, and current curated Apple-local downloads are represented
    through `engine="mflux"` presets rather than a true native `mlx` engine.
  - References:
    - `src/abstractvision/model_downloads.py`
    - `src/abstractvision/assets/vision_model_capabilities.json`

### Option A: Keep MFLUX as the permanent Apple-local answer

Pros:

- Lowest short-term implementation cost.
- Already integrated into CLI, playground, and AbstractCore.
- Keeps current Apple 8-bit flows working where supported.

Cons:

- MLX remains a fake engine category rather than a real runtime.
- AbstractVision stays bound to MFLUX-supported model families.
- Hard-coded family/routing special cases will keep expanding.

### Option B: Build a native/general MLX backend and keep MFLUX as the bridge

Pros:

- Matches the real product goal: MLX as a first-class local engine on Apple Silicon.
- Decouples AbstractVision’s Apple strategy from one specialized runtime.
- Allows the catalog to represent “native MLX runnable” separately from “MFLUX runnable”.

Cons:

- Higher implementation and integration cost.
- Needs real feasibility proof across at least one non-MFLUX family.
- Requires a careful transition plan so current MFLUX users do not regress.

### Option C: Treat MLX as download-only for now and postpone runtime work

Pros:

- Avoids more short-term runtime complexity.
- Keeps backend surface smaller.

Cons:

- Fails the product goal for Apple-local MLX execution.
- Leaves users with downloadable artifacts that are not a first-class runtime path.
- Pushes more unsupported/manual assembly burden onto callers.

---

## Decision

**Chosen approach**: build a true native/general MLX backend for AbstractVision, while keeping
MFLUX as the short-term bridge and compatibility runtime during the transition.

**Why**:

- The long-term goal is to handle MLX vision models as a real engine category, not only through
  MFLUX-specific support tables.
- MFLUX is good enough as a bridge, but too narrow to be the terminal abstraction for Apple-local
  model execution.
- The repo already shows that MLX download variants matter to users; the missing piece is a real
  engine boundary and runtime support strategy.
- Truth-alignment work is still required, but only as a prerequisite to shipping a trustworthy MLX
  engine, not as the end goal of this task.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0005_curated_capability_registry_and_download_catalog.md`
  - `docs/adr/0006_operator_control_configuration_precedence_and_explicit_network_use.md`
- **Backlog tasks**:
  - Planned: `docs/backlog/planned/020_adapter_aware_model_graph_and_catalog.md`
  - Completed: `docs/backlog/completed/007_local_hf_backend_strategy_diffusers.md`
  - Completed: `docs/backlog/completed/013_stable_diffusion_cpp_gguf_backend.md`
  - Completed: `docs/backlog/completed/015_vision_install_profiles_and_pending_defaults.md`
  - Completed: `docs/backlog/completed/019_best_effort_preload_warmup_for_local_backends.md`

---

## Implementation plan

- Define the native MLX backend boundary:
  - provider/backend id;
  - configuration shape;
  - cache discovery expectations;
  - supported local model layout requirements.
- Keep MFLUX as a separate backend/provider during the transition; do not silently collapse `mlx`
  into `mflux`.
- Add a first native MLX backend implementation that can load:
  - at least one non-MFLUX family;
  - at least one currently curated Apple-local family from configuration rather than from a
    narrow hard-coded family switchboard.
- Tighten current runtime truth while the MLX backend is being built:
  - fail fast on `mlx/...` everywhere until the native backend exists;
  - stop advertising unsupported MFLUX families as runnable;
  - keep download-only Apple variants visible only with explicit non-runnable state.
- Add runtime-state metadata that distinguishes:
  - cached/downloadable;
  - runnable through MFLUX;
  - runnable through native MLX;
  - download-only.
- Update CLI, playground, and AbstractCore routing so `mlx` becomes a real backend/provider once
  the native engine exists.
- Revisit preset and registry generation so Apple-local download truth does not drift across
  `_PRESETS`, the registry, and backend family tables.

---

## Success criteria

- AbstractVision has a true native MLX backend/provider beyond MFLUX.
- Generic `mlx` model/provider selection is a real supported path, not just an error or alias.
- At least one non-MFLUX family is runnable through the native MLX backend.
- Apple-local catalog surfaces distinguish native-MLX runnable, MFLUX-runnable, and download-only
  entries clearly.
- Unsupported MFLUX-only families are not advertised as runnable.
- Runtime extras remain optional and base `abstractvision` stays import-light.
- Docs explain when to choose native MLX, MFLUX, Diffusers, and stable-diffusion.cpp/GGUF.

---

## Test plan

- `PYTHONPATH=src python -m unittest tests.test_model_downloads -q`
- `PYTHONPATH=src python -m unittest tests.test_cli_smoke -q`
- `PYTHONPATH=src python -m unittest tests.test_playground_server -q`
- `PYTHONPATH=src python -m unittest tests.test_abstractcore_plugin -q`
- Add dedicated backend tests for:
  - native MLX backend model loading;
  - unsupported `mlx/...` routing before engine availability;
  - supported `mlx/...` routing after engine availability;
  - separation between MFLUX-runnable and native-MLX-runnable catalog entries.
- Manual Apple Silicon smoke tests for:
  - one MFLUX-backed model;
  - one native-MLX-backed non-MFLUX family;
  - one download-only MLX entry with a clear non-runnable explanation.

---

## Report (fill only when completed)

### Summary

N/A

### Validation

- Tests: N/A
