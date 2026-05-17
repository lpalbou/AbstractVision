# Capability registry (`vision_model_capabilities.json`)

AbstractVision keeps a single packaged “source of truth” for what models can do:

- Asset: [`../../src/abstractvision/assets/vision_model_capabilities.json`](../../src/abstractvision/assets/vision_model_capabilities.json)
- Loader + validator: `VisionModelCapabilitiesRegistry` / `validate_capabilities_json()` in [`../../src/abstractvision/model_capabilities.py`](../../src/abstractvision/model_capabilities.py)

See also:
- CLI/REPL inspection commands: [docs/reference/configuration.md](configuration.md)
- Backends (execution reality): [docs/reference/backends.md](backends.md)

## What the registry is used for

- **Discovery**: list known task keys and model ids.
- **Optional safety gating**:
  - `VisionManager(model_id=..., registry=...)` will fail fast if the model doesn’t support a task ([`../../src/abstractvision/vision_manager.py`](../../src/abstractvision/vision_manager.py)).
  - The CLI/REPL can enforce gating via `--capabilities-model-id` (CLI) or `/cap-model` (REPL).
- **Task-aware UI/catalog metadata**:
  - `image_to_image` in a model’s `tasks` map is the explicit “this model supports edits” signal.
  - The playground API surfaces that structured metadata as `task_specs` so local tooling can enable edit-only flows without re-encoding model rules elsewhere.
- **Request normalization hints**:
  - additive parameter metadata such as `default`, `const`, `min`, `multiple_of`, `supported`, and `auto_derived_from_input` can be consumed by backends to keep model-specific defaults and constraints in one packaged source of truth.

Important:
- The registry describes **model capability intent**.
- Your configured backend still needs to implement the task at runtime (see backend support matrix in [docs/reference/backends.md](backends.md)).

## Minimal Python usage

```python
from abstractvision import VisionModelCapabilitiesRegistry

reg = VisionModelCapabilitiesRegistry()
print(reg.schema_version())
print(reg.list_tasks())

assert reg.supports("runwayml/stable-diffusion-v1-5", "text_to_image")
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
  - optional `downloads` (list of downloadable artifacts; informational metadata used by `abstractvision show-model`)
  - `tasks` (map of task name → task spec)
- Each task spec includes:
  - `inputs`, `outputs` (lists of strings)
  - `params` (object where each param has `required: bool`, plus additive fields)
  - optional `requires` for dependencies like `base_model_id`

Examples of additive `params` metadata used in this repo:
- `default`: backend fills a missing value
- `const`: backend forces a model-specific fixed value
- `min`: backend clamps the value upward
- `multiple_of`: backend rounds dimensions to a required multiple
- `supported: false`: backend drops an unsupported optional parameter
- `auto_derived_from_input: true`: image-edit backends can infer a missing size from the input image

### Downloads (optional)

Some models include a `downloads` list to document what AbstractVision considers “downloadable” for common engines:

- `key`: the canonical short name used in `model-presets` / `download-model` (e.g. `qwen-image`, `flux2-klein-4b`)
- `engine`: `mflux`, `diffusers`, `stable-diffusion.cpp`, ...
- `target`: `mlx`, `diffusers`, `gguf`, ...
- `bits`: typically `8` (quantized) or `16` (full snapshot)
- `repo_id`: Hugging Face repo id to download
- optional `source`, `notes`

This metadata is descriptive; the curated preset list printed by `abstractvision model-catalog` remains the practical “what can I download” view.

In this repo, we keep `downloads` entries aligned with the curated preset table (`model_downloads._PRESETS`) so `abstractvision show-model` and the docs can reliably describe what the CLI can download.
