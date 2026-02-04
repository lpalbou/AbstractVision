# Contributing to AbstractVision

Thanks for taking the time to contribute. This repository aims to stay small, stable, and easy to integrate.

## Ground rules

- Keep the public API stable (`VisionManager` in `src/abstractvision/vision_manager.py`).
- Prefer additive changes (new fields, new models, new backends) over breaking changes.
- Don’t commit model weights, large binaries, or cache artifacts.
- Make docs and examples match the code (the repo is intended to be “readme-first”).

## Development setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## Run tests

```bash
python -m unittest discover -s tests -p "test_*.py" -q
```

## Common contribution types

### 1) Improve documentation

Core entrypoints:
- `README.md`
- `docs/getting-started.md`
- `docs/architecture.md`
- `docs/api.md`
- `docs/faq.md`

Doc hygiene checklist:
- Commands are copy/pastable.
- Links resolve (relative links are preferred).
- Claims about support status match the current code (see `docs/reference/backends.md`).

### 2) Add or update models in the capability registry

Source of truth:
- `src/abstractvision/assets/vision_model_capabilities.json`

Validator + loader:
- `src/abstractvision/model_capabilities.py`

Checklist:
- Add/update the model entry in the JSON.
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

