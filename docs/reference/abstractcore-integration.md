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

The plugin registers these backend ids:

- `abstractvision:openai` (default backend id) and `abstractvision:openai-compatible` (legacy compatibility backend id) are registered by [`../../src/abstractvision/integrations/abstractcore_plugin.py`](../../src/abstractvision/integrations/abstractcore_plugin.py). The implementation also supports local backends.

Current behavior:
- Default `abstractvision:openai`: OpenAI HTTP (`https://api.openai.com/v1`). Set `OPENAI_API_KEY`.
- OpenAI model ids are configured, not discovered dynamically. Providers may expose an OpenAI-compatible `GET /models` catalog; AbstractVision exposes it through `abstractvision provider-models`, `VisionManager.list_provider_models(...)`, and the plugin method `llm.vision.list_provider_models(...)`, but the plugin does not call it automatically or use it to select a model. The static plugin default is `gpt-image-1`; set `OPENAI_IMAGE_MODEL_ID`, `OPENAI_IMAGE_MODEL`, `ABSTRACTVISION_MODEL_ID`, or `vision_model_id` for newer provider models.
- Compatible HTTP: set `OPENAI_BASE_URL` to a local/remote compatible `/v1` server. Set `ABSTRACTVISION_BACKEND=openai-compatible` when you want to force compatible-endpoint semantics.
- Legacy `abstractvision:openai-compatible`: keeps compatible-endpoint defaults when that backend id is selected directly.
- Local Diffusers: install `abstractvision[diffusers]`, then set `ABSTRACTVISION_BACKEND=diffusers` with `runwayml/stable-diffusion-v1-5` or another supported Diffusers model. It is cache-only/offline unless `ABSTRACTVISION_DIFFUSERS_ALLOW_DOWNLOAD=1` is set. Local `text_to_video` groundwork exists but is currently experimental and disabled from the normal local plugin surfaces.
- Local MFLUX (Apple Silicon): install `abstractvision[mflux]` (or `abstractvision[all-apple]`), then set `ABSTRACTVISION_BACKEND=mflux`. Download an 8-bit MLX/MFLUX preset with `abstractvision download flux2-klein-4b --provider mflux` (stored in the Hugging Face cache; `ABSTRACTVISION_MODEL_DIR` is only a legacy import root). Use routed model ids such as `mflux/flux2-klein-4b`. Current bundled MFLUX support is `text_to_image` only.
- stable-diffusion.cpp: set `ABSTRACTVISION_BACKEND=sdcpp` and configure either a model path or a curated model key such as `flux2-klein-base-4b`. Use an external `sd-cli`, or install `abstractvision[sdcpp]` for the python binding fallback.
- The plugin reads AbstractCore owner config keys when present, then falls back to `ABSTRACTVISION_*` env vars.
- Gateway/Core should pass process-level config or `owner.config` and report readiness; they should not mutate AbstractVision environment variables per request.

Key config keys (owner.config):
- `vision_backend_instance` / `vision_backend_factory` (advanced injection hooks; bypass env-driven backend creation)
- `vision_backend` (`openai`, `openai-compatible`, `diffusers`, `mflux`, or `sdcpp`; default `openai`)
- `vision_model_id` (Diffusers/OpenAI-compatible model id; default `gpt-image-1` only for the official OpenAI profile and `runwayml/stable-diffusion-v1-5` for Diffusers)
- `vision_device` / `vision_torch_dtype` / `vision_allow_download` / `vision_auto_retry_fp32` (Diffusers)
- `vision_base_url` / `vision_api_key` (OpenAI or compatible HTTP)
- `vision_mflux_model` / `vision_mflux_base_model` / `vision_mflux_quantize` / `vision_mflux_allow_download` (MFLUX)
- `vision_model_dir` (legacy preset import root used by MFLUX compatibility/migration helpers; new downloads land in the Hugging Face cache)
- `vision_sdcpp_model` / `vision_sdcpp_diffusion_model` / `vision_sdcpp_bin` (stable-diffusion.cpp; `vision_sdcpp_model` may be a curated model key such as `flux2-klein-base-4b`)
- `vision_sdcpp_vae` / `vision_sdcpp_llm` / `vision_sdcpp_llm_vision` / `vision_sdcpp_clip_l` / `vision_sdcpp_clip_g` / `vision_sdcpp_t5xxl` / `vision_sdcpp_extra_args` (stable-diffusion.cpp component mode)
- `vision_timeout_s` (optional)
- `vision_models_path` (optional provider catalog path; default `/models`)
- Optional video endpoint keys:
  - `vision_text_to_video_path`
  - `vision_image_to_video_path`
  - `vision_image_to_video_mode`

Env-only aliases:
- `ABSTRACTVISION_DIFFUSERS_MODEL_ID` is accepted for the Diffusers plugin backend before falling back to `ABSTRACTVISION_MODEL_ID`.
- `OPENAI_BASE_URL` is accepted by the OpenAI-shaped HTTP backend.
- Compatible OpenAI-shaped endpoints can use `OPENAI_API_KEY` when they require bearer auth.
- `OPENAI_IMAGE_MODEL_ID` and `OPENAI_IMAGE_MODEL` are accepted when `vision_model_id` / `ABSTRACTVISION_MODEL_ID` are unset.
- `ABSTRACTVISION_SDCPP_CLIP_L`, `ABSTRACTVISION_SDCPP_CLIP_G`, and `ABSTRACTVISION_SDCPP_T5XXL` are accepted for stable-diffusion.cpp component mode.

Examples:

```bash
# Local Diffusers. Pre-download weights first, or explicitly allow runtime downloads.
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
# OpenAI API.
export OPENAI_API_KEY=...
export OPENAI_IMAGE_MODEL=gpt-image-1
```

```bash
# Local OpenAI-compatible HTTP server, for example AbstractCore Server.
export ABSTRACTVISION_BACKEND=openai-compatible
export OPENAI_BASE_URL=http://localhost:8000/v1
export ABSTRACTVISION_MODEL_ID=server/default
```

### Provider Catalog Discovery

Core/Gateway hosts can inspect provider-advertised model catalogs through the same capability
object used for generation:

```python
models = llm.vision.list_provider_models(task="text_to_image")
for model in models:
    print(model["id"])
```

The return value is a JSON-safe list of dictionaries serialized from `ProviderModelInfo`. Raw
provider metadata is retained in a bounded `raw` field for diagnostics. This method is explicit
inspection only: it does not mutate the configured backend or select a generation model.

Backends that do not implement provider catalog listing raise a clear AbstractVision error instead
of returning a misleading empty catalog. Local Diffusers and stable-diffusion.cpp model discovery
remain separate local-backend concerns, while MFLUX and Diffusers provider listings reflect
cache-backed snapshots rather than a separate `~/models` download tree.

### Local Model Residency Control

Core/Gateway hosts can also control in-process local model residency through the same capability
object:

```python
llm.vision.load_resident_model(
    {"task": "text_to_image", "provider": "mflux", "model": "flux2-klein-4b"}
)

loaded = llm.vision.list_loaded_models()
resident = llm.vision.list_resident_models()

llm.vision.unload_resident_model(
    {"provider": "mflux", "model": "flux2-klein-4b"}
)
```

Notes:

- this surface is local-only and process-local;
- it controls only AbstractVision-managed in-process backends (`diffusers`, `mflux`, `sdcpp`);
- OpenAI/OpenAI-compatible HTTP backends are intentionally rejected here, even on `localhost`,
  because the plugin cannot honestly control another process's loaded-state.

`list_loaded_models()` reports both explicitly preloaded resident models and transient currently
loaded local models from recent generation requests. `list_resident_models()`
returns the explicit pinned subset.

Each loaded-model entry includes stable routing metadata such as `load_id`, `backend_kind`,
`resident`, and a `tasks` list of observed task aliases using that loaded backend.

For deterministic unload behavior, prefer:

- `load_id`, or
- both `provider`/`backend` and `model`.

Broad unload filters that match multiple loaded models are rejected as ambiguous.

## Shared request normalization

`llm.vision` calls go through `VisionManager`, which applies backend normalization hooks before
execution. That means model-specific constraints are shared with the CLI/REPL and playground API
instead of being reimplemented in the plugin layer.

Current examples:
- Distilled MFLUX/FLUX-family models can clamp guidance or minimum steps in the backend.
- Diffusers-backed image requests pick up registry-driven defaults such as recommended step counts,
  guidance defaults, and dimension constraints.
- Unsupported parameters, such as negative prompts on constrained model families, are dropped in
  the backend rather than surfacing as per-surface validation bugs.

## 2) Tool helpers (`make_vision_tools`)

`make_vision_tools(...)` builds AbstractCore `@tool` callables for:
- text→image
- image→image
- multi-view image
- text→video
- image→video

Important:
- Tool outputs are designed to be **artifact refs**, so `VisionManager.store` must be set ([`../../src/abstractvision/integrations/abstractcore.py`](../../src/abstractvision/integrations/abstractcore.py)).
- This module requires AbstractCore to be installed by the host application. AbstractVision does not install AbstractCore as a dependency.

Tip (framework mode):
- If your runtime provides an artifact store (e.g. AbstractRuntime), use `RuntimeArtifactStoreAdapter` so tool outputs can be stored and referenced across processes (see [docs/reference/artifacts.md](artifacts.md)).
