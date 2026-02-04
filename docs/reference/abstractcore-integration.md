# AbstractCore integration

AbstractVision offers two integration surfaces for AbstractCore:

1) **Capability plugin** (so `abstractcore` can discover a vision backend)
2) **Tool helpers** (so you can expose vision tasks as tools with artifact-ref outputs)

Code pointers:
- Plugin: `src/abstractvision/integrations/abstractcore_plugin.py`
- Tools: `src/abstractvision/integrations/abstractcore.py`
- Entry point registration: `pyproject.toml` (`[project.entry-points.\"abstractcore.capabilities_plugins\"]`)

See also:
- Artifacts: `docs/reference/artifacts.md`
- Backends: `docs/reference/backends.md`

## 1) Capability plugin (AbstractCore → VisionCapability)

The plugin registers a backend id:

- `abstractvision:openai-compatible` (see `_AbstractVisionCapability.backend_id` in `src/abstractvision/integrations/abstractcore_plugin.py`)

Current behavior (v0):
- Only the **OpenAI-compatible HTTP backend** is supported via the plugin.
- The plugin reads AbstractCore owner config keys when present, and falls back to `ABSTRACTVISION_*` env vars.

Key config keys (owner.config):
- `vision_base_url` (required)
- `vision_api_key` (optional)
- `vision_model_id` (optional)
- `vision_timeout_s` (optional)
- Optional video endpoint keys:
  - `vision_text_to_video_path`
  - `vision_image_to_video_path`
  - `vision_image_to_video_mode`

## 2) Tool helpers (`make_vision_tools`)

`make_vision_tools(...)` builds AbstractCore `@tool` callables for:
- text→image
- image→image
- multi-view image
- text→video
- image→video

Important:
- Tool outputs are designed to be **artifact refs**, so `VisionManager.store` must be set (`src/abstractvision/integrations/abstractcore.py`).
- This module requires AbstractCore to be installed (install extra: `pip install "abstractvision[abstractcore]"`).

