# Contributing to AbstractVision

Thanks for taking the time to contribute. This repository aims to stay small, stable-by-design, and easy to integrate.

AbstractVision is part of the **AbstractFramework** ecosystem:
- AbstractFramework: <https://github.com/lpalbou/AbstractFramework>
- AbstractCore: <https://github.com/lpalbou/abstractcore>
- AbstractRuntime: <https://github.com/lpalbou/abstractruntime>

## Ground rules

- Keep the public API stable (`VisionManager` in [`src/abstractvision/vision_manager.py`](src/abstractvision/vision_manager.py)).
- Prefer additive changes (new fields, new models, new backends) over breaking changes.
- Don’t commit model weights, large binaries, or cache artifacts.
- Make docs and examples match the code (the repo is intended to be “readme-first”).
- Keep imports lazy for heavy stacks (see [`src/abstractvision/backends/__init__.py`](src/abstractvision/backends/__init__.py)).

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Optional (if you work on AbstractCore integration locally):

```bash
python -m pip install abstractcore
```

The `abstractvision[abstractcore]` extra is only a compatibility marker. AbstractCore is intentionally supplied by the host application, not installed by AbstractVision.

## Run tests

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

## Common contribution types

### 1) Improve documentation

Core entrypoints:
- [`README.md`](README.md)
- [`docs/getting-started.md`](docs/getting-started.md)
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/api.md`](docs/api.md)
- [`docs/faq.md`](docs/faq.md)
- [`docs/adr/README.md`](docs/adr/README.md)

Doc hygiene checklist:
- Commands are copy/pastable.
- Links resolve (relative links are preferred).
- Claims about support status match the current code (see [`docs/reference/backends.md`](docs/reference/backends.md)).
- Major claims are anchored in evidence (link to the relevant `src/` implementation).
- Prefer diagrams in Mermaid when they improve clarity ([`docs/architecture.md`](docs/architecture.md) is the canonical place).

### 2) Add or update models in the capability registry

Source of truth:
- `src/abstractvision/assets/vision_model_capabilities.json`

Validator + loader:
- `src/abstractvision/model_capabilities.py`

Checklist:
- Add/update the model entry in the JSON.
- Keep the model family platform-neutral; add engine-specific artifacts as download variants rather
  than as host-specific duplicate models.
- Prefer official upstream repos. Use community repos only when they provide the best runtime-native
  artifact for a target engine and label them clearly.
- Do not treat adapters or component artifacts as standalone curated models unless the runtime path
  is first-class.
- Keep the change aligned with
  [ADR 0005](docs/adr/0005_curated_capability_registry_and_download_catalog.md),
  [ADR 0006](docs/adr/0006_operator_control_configuration_precedence_and_explicit_network_use.md),
  and [docs/reference/capabilities-registry.md](docs/reference/capabilities-registry.md).
- Run the unit tests (they validate schema + coverage).
- Sanity check CLI output:
  - `abstractvision show-model <model_id>`

### 3) Add a new backend

Backend interface:
- `src/abstractvision/backends/base_backend.py`

Where backends live:
- `src/abstractvision/backends/`

Checklist:
- Implement the `VisionBackend` methods (raise `CapabilityNotSupportedError` for unsupported tasks).
- Keep imports lazy (avoid importing Torch/Diffusers at module import time unless unavoidable).
- Add/extend tests under `tests/`.
- Document the backend in `docs/reference/backends.md` and, if user-facing, add a short section in `docs/getting-started.md`.

## Submitting a change

Please include:
- A short explanation of the change and why it’s needed.
- Test results (`python -m unittest ...`).
- Any doc updates required to keep the repository truthful.

## Questions / discussions

If you’re unsure about scope or design, open an issue with a minimal proposal and a concrete example (inputs/outputs).
