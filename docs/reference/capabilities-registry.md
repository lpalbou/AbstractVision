# Capability registry (`vision_model_capabilities.json`)

AbstractVision keeps a single packaged “source of truth” for what models can do:

- Asset: `src/abstractvision/assets/vision_model_capabilities.json`
- Loader + validator: `VisionModelCapabilitiesRegistry` / `validate_capabilities_json()` in `src/abstractvision/model_capabilities.py`

See also:
- CLI/REPL inspection commands: `docs/reference/configuration.md`
- Backends (execution reality): `docs/reference/backends.md`

## What the registry is used for

- **Discovery**: list known task keys and model ids.
- **Optional safety gating**:
  - `VisionManager(model_id=..., registry=...)` will fail fast if the model doesn’t support a task (`src/abstractvision/vision_manager.py`).
  - The CLI/REPL can enforce gating via `--capabilities-model-id` (CLI) or `/cap-model` (REPL).

Important:
- The registry describes **model capability intent**.
- Your configured backend still needs to implement the task at runtime (see backend support matrix in `docs/reference/backends.md`).

## Minimal Python usage

```python
from abstractvision import VisionModelCapabilitiesRegistry

reg = VisionModelCapabilitiesRegistry()
print(reg.schema_version())
print(reg.list_tasks())

assert reg.supports("Qwen/Qwen-Image-2512", "text_to_image")
print(reg.models_for_task("text_to_image"))
```

## JSON shape (high level)

The validator enforces a “soft schema”:

- Top-level keys:
  - `schema_version`
  - `tasks` (keyed by task name; includes human descriptions)
  - `models` (keyed by model id)
- Each model entry includes:
  - `provider` (string)
  - `license` (string; informational)
  - `tasks` (map of task name → task spec)
- Each task spec includes:
  - `inputs`, `outputs` (lists of strings)
  - `params` (object where each param has `required: bool`, plus additive fields)
  - optional `requires` for dependencies like `base_model_id`

