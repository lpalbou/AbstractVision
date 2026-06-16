# Capability registries (`vision_model_capabilities.json`, `vision_adapter_capabilities.json`)

AbstractVision keeps two packaged capability assets:

- Asset: [`../../src/abstractvision/assets/vision_model_capabilities.json`](../../src/abstractvision/assets/vision_model_capabilities.json)
- Loader + validator: `VisionModelCapabilitiesRegistry` / `validate_capabilities_json()` in [`../../src/abstractvision/model_capabilities.py`](../../src/abstractvision/model_capabilities.py)
- Asset: [`../../src/abstractvision/assets/vision_adapter_capabilities.json`](../../src/abstractvision/assets/vision_adapter_capabilities.json)
- Loader + validator: `VisionAdapterCapabilitiesRegistry` / `validate_adapter_capabilities_json()` in [`../../src/abstractvision/adapter_capabilities.py`](../../src/abstractvision/adapter_capabilities.py)

See also:
- CLI/REPL inspection commands: [docs/reference/configuration.md](configuration.md)
- Backends (execution reality): [docs/reference/backends.md](backends.md)
- ADR policy: [docs/adr/0005_curated_capability_registry_and_download_catalog.md](../adr/0005_curated_capability_registry_and_download_catalog.md)

## Policy boundary

The packaged assets are not just descriptive metadata. In this repo they feed:

- model discovery;
- adapter discovery hints and default parameter recommendations;
- task and parameter introspection;
- curated download surfacing;
- backend request normalization.

Under [ADR 0005](../adr/0005_curated_capability_registry_and_download_catalog.md), this inventory is:

- global rather than host-specific;
- organized around parent model families with engine-specific download variants;
- official-upstream-first, with clearly labeled community ports only when they are the best
  runtime-native choice for a target engine;
- explicit about download-only or backend-not-supported states.

Important boundary:

- `vision_model_capabilities.json` is the curated package view of model-family tasks, downloadable variants, and model-level defaults.
- `vision_adapter_capabilities.json` is the curated package view of adapter families, compatible base-model aliases, and adapter default/recommended parameters.
- Backend/runtime route truth still stays backend-owned. For example, MLX-Gen decides whether one exact local route currently supports `mask`, `control_image`, or LoRA target roles at runtime.

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
- **Adapter-aware request normalization hints**:
  - adapter profiles carry package-owned defaults such as recommended `steps`, `guidance_scale`, documentation links, and quantization notes.

Important:
- The registry describes **model capability intent**.
- Your configured backend still needs to implement the task at runtime (see backend support matrix in [docs/reference/backends.md](backends.md)).

## Override paths

Both registries can be redirected to local JSON files without changing Python code:

- `ABSTRACTVISION_MODEL_CAPABILITIES_PATH`
- `ABSTRACTVISION_ADAPTER_CAPABILITIES_PATH`

You can also pass `asset_path=...` directly when constructing either registry.
That keeps the code/package boundary stable while letting operators update
curated model/adapter metadata independently.

Practical workflow:

1. Copy the packaged JSON you want to override.
2. Edit the local copy.
3. Point the matching env var at that file.
4. Verify the effective metadata with:
   - `abstractvision show-model <model-id>`
   - `abstractvision adapters --provider mlx-gen --model <model-id> --task <task> --json`

## Curation rules

When editing the registry:

- keep one parent entry per real model family;
- keep host-specific or engine-specific artifacts in `downloads`, not as duplicated top-level
  model entries;
- prefer official upstream repos;
- use community repos only for runtime-native artifacts such as GGUF or MLX ports when no
  official equivalent exists;
- do not promote adapter-only or component-only repos as standalone curated models unless the
  runtime can orchestrate them as first-class flows;
- mark non-runnable local entries explicitly when the backend is not shipped yet.

## Minimal Python usage

```python
from abstractvision import VisionModelCapabilitiesRegistry
from abstractvision.adapter_capabilities import VisionAdapterCapabilitiesRegistry

reg = VisionModelCapabilitiesRegistry()
print(reg.schema_version())
print(reg.list_tasks())

assert reg.supports("runwayml/stable-diffusion-v1-5", "text_to_image")
print(reg.models_for_task("text_to_image"))

adapter_reg = VisionAdapterCapabilitiesRegistry()
profile = adapter_reg.match_profile(
    repo_id="lightx2v/Qwen-Image-2512-Lightning",
    relative_path="Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
)
print(profile.to_profile_dict()["recommended_parameters"])
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

- `key`: the canonical short name used in `model-presets` / `download` (e.g. `qwen-image`, `flux2-klein-4b`)
- `engine`: `mlx-gen`, `diffusers`, `stable-diffusion.cpp`, ...
- `target`: `mlx`, `diffusers`, `gguf`, ...
- `bits`: typically `8` (quantized) or `16` (full snapshot)
- `repo_id`: Hugging Face repo id to download
- optional `source`, `notes`

This metadata is descriptive, but in this repo it is maintained to align with the curated
download surfaces exposed by `abstractvision catalog`, `abstractvision model-presets`, and
`abstractvision download`.

In this repo, we keep `downloads` entries aligned with the curated preset table (`model_downloads._PRESETS`) so `abstractvision show-model` and the docs can reliably describe what the CLI can download.
That alignment is intentionally user-facing: if a variant is curated for download, it should have
clear provenance, a realistic runtime target, and a documented readiness story.
