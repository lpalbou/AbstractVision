# AbstractCore integration

AbstractVision offers two integration surfaces for AbstractCore:

1) **Capability plugin** (so `abstractcore` can discover a vision backend)
2) **Tool helpers** (so you can expose vision tasks as tools with artifact-ref outputs)

Code pointers:
- Plugin: [`../../src/abstractvision/integrations/abstractcore_plugin.py`](../../src/abstractvision/integrations/abstractcore_plugin.py)
- Tools: [`../../src/abstractvision/integrations/abstractcore.py`](../../src/abstractvision/integrations/abstractcore.py)
- Entry point registration: [`../../pyproject.toml`](../../pyproject.toml) (`[project.entry-points."abstractcore.capabilities_plugins"]`)

See also:
- Artifacts: [docs/reference/artifacts.md](artifacts.md)
- Backends: [docs/reference/backends.md](backends.md)

## 1) Capability plugin (AbstractCore → VisionCapability)

The plugin registers a backend id:

- `abstractvision:openai-compatible` (see `_AbstractVisionCapability.backend_id` in [`../../src/abstractvision/integrations/abstractcore_plugin.py`](../../src/abstractvision/integrations/abstractcore_plugin.py))

Current behavior:
- Default: local Diffusers with `runwayml/stable-diffusion-v1-5`, cache-only/offline unless `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1` is set.
- OpenAI-compatible HTTP: set `ABSTRACTVISION_BACKEND=openai` and `ABSTRACTVISION_BASE_URL`.
- stable-diffusion.cpp: set `ABSTRACTVISION_BACKEND=sdcpp` and configure a model path.
- The plugin reads AbstractCore owner config keys when present, then falls back to `ABSTRACTVISION_*` env vars.

Key config keys (owner.config):
- `vision_backend` (`diffusers`, `openai`, or `sdcpp`; default `diffusers`)
- `vision_model_id` (Diffusers/OpenAI-compatible model id; default `runwayml/stable-diffusion-v1-5` for Diffusers)
- `vision_device` / `vision_torch_dtype` / `vision_allow_download` (Diffusers)
- `vision_base_url` / `vision_api_key` (OpenAI-compatible)
- `vision_sdcpp_model` / `vision_sdcpp_diffusion_model` / `vision_sdcpp_bin` (stable-diffusion.cpp)
- `vision_timeout_s` (optional)
- Optional video endpoint keys:
  - `vision_text_to_video_path`
  - `vision_image_to_video_path`
  - `vision_image_to_video_mode`

Examples:

```bash
# Local Diffusers default. Pre-download weights first, or explicitly allow runtime downloads.
export ABSTRACTVISION_BACKEND=diffusers
export ABSTRACTVISION_MODEL_ID=runwayml/stable-diffusion-v1-5
export ABSTRACTVISION_DIFFUSERS_DEVICE=auto
```

```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o-mini")
png_bytes = llm.vision.t2i("a red square", width=512, height=512, steps=20)
```

```bash
# OpenAI-compatible HTTP backend, for example through AbstractCore Server.
export ABSTRACTVISION_BACKEND=openai
export ABSTRACTVISION_BASE_URL=http://localhost:8000/v1
export ABSTRACTVISION_MODEL_ID=server/default
```

## 2) Tool helpers (`make_vision_tools`)

`make_vision_tools(...)` builds AbstractCore `@tool` callables for:
- text→image
- image→image
- multi-view image
- text→video
- image→video

Important:
- Tool outputs are designed to be **artifact refs**, so `VisionManager.store` must be set ([`../../src/abstractvision/integrations/abstractcore.py`](../../src/abstractvision/integrations/abstractcore.py)).
- This module requires AbstractCore to be installed (install extra: `pip install "abstractvision[abstractcore]"`).

Tip (framework mode):
- If your runtime provides an artifact store (e.g. AbstractRuntime), use `RuntimeArtifactStoreAdapter` so tool outputs can be stored and referenced across processes (see [docs/reference/artifacts.md](artifacts.md)).
