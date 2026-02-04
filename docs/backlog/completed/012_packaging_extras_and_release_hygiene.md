## Task 012: Packaging + extras + release hygiene (third-party friendly installs)

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P1  

---

Update (2026-02-04):
- The current `pyproject.toml` is “batteries included” (base install pulls local-backend deps like Torch/Diffusers).
- Treat this backlog item as historical context for how we thought about extras; `pyproject.toml` is the current source of truth.

## Main goals

- Make `abstractvision` installation and dependency behavior predictable for third parties:
  - `pip install abstractvision` stays dependency-light and import-safe
  - heavy stacks are opt-in extras with clear names
- Align packaging metadata with the intended integration story (tools/backends/artifacts).

## Secondary goals

- Add minimal release hygiene:
  - a CHANGELOG entry workflow
  - version bump conventions

---

## Context / problem

Third parties adopt libraries based on install experience:
- base installs must not pull GPU stacks unexpectedly,
- optional features must be discoverable via extras,
- package metadata should reflect runtime behavior.

`abstractvision` currently uses a minimal `setup.py` and has no declared extras. As the project grows (HTTP backend, HF backend, tool integration), we need explicit packaging boundaries.

---

## Constraints

- No heavy dependencies in base install.
- Cross-platform where possible; use platform markers where not.

---

## Research, options, and references

- **Option A**: Keep `setup.py` and add conditional imports only.
  - Pros: minimal changes.
  - Cons: harder to express extras cleanly; drifts from the rest of the ecosystem.
- **Option B (chosen)**: Move to `pyproject.toml` (PEP 621) with explicit optional dependencies.
  - Pros: clearer third-party consumption; easier tooling; consistent with other packages in the workspace.

---

## Decision

**Chosen approach**:
- Introduce `pyproject.toml` for `abstractvision` and define clear extras, e.g.:
  - `abstractvision[openai-compatible]` → `httpx` (and any small helpers)
  - `abstractvision[huggingface]` → torch/diffusers/transformers (behind explicit opt-in)
  - `abstractvision[abstractcore]` → `abstractcore` (tool integration module)
- Ensure the capabilities JSON asset is included in wheels/sdists.

---

## Dependencies

- **Backlog tasks**:
  - Completed: `docs/backlog/completed/006_openai_compatible_backend_for_image_and_video.md`
  - Completed: `docs/backlog/completed/007_local_hf_backend_strategy_diffusers.md`
  - Completed: `docs/backlog/completed/011_abstractcore_tool_integration_and_artifact_refs.md`

---

## Implementation plan

- Add `pyproject.toml` with:
  - project metadata
  - package discovery (`src/` layout)
  - package data inclusion for `assets/*.json`
  - optional dependency groups (extras)
- Add a minimal `CHANGELOG.md` and version bump guideline.
- Add a packaging smoke check command to Task 009 CI notes:
  - install base + import + load registry

---

## Success criteria

- `pip install abstractvision` does not pull heavy ML deps and `import abstractvision` works.
- `pip install "abstractvision[openai-compatible]"` enables the HTTP backend.
- `pip install "abstractvision[huggingface]"` enables the local backend (where supported).

---

## Test plan

- `python -m unittest discover -s tests -p "test_*.py" -q`
- (optional) local packaging smoke: build wheel and install into a clean venv

---

## Report (fill only when completed)

### Summary

- Migrated `abstractvision` packaging to `pyproject.toml` (PEP 621) with:
  - dependency-light base install (`dependencies = []`)
  - optional extras for heavy/optional features:
    - `abstractvision[huggingface]` for local diffusers backend
    - `abstractvision[abstractcore]` for tool integration module
    - `abstractvision[openai-compatible]` kept (currently empty; forward-compatible)
  - console script entrypoint: `abstractvision = abstractvision.cli:main`
  - dynamic version from `abstractvision.__version__`
  - wheel/sdist inclusion for `assets/*.json`
- Added `CHANGELOG.md` with an initial 0.1.0 entry.

### Validation

- Tests: `python -m unittest discover -s tests -p "test_*.py" -q`
- Packaging sanity: `pyproject.toml` parses via `tomllib`.
- Editable install (src layout): `python -m pip install -e ".[huggingface]"`
