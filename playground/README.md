# AbstractVision Playground (Web)

This is a tiny web UI for testing AbstractVision locally. It is powered by the
self-contained `abstractvision playground` command; it does **not** require an
AbstractCore server.

The playground is a local/dev surface. Do not expose it as an authenticated
production serving boundary; use AbstractCore/Gateway for production routing,
authentication, and browser-origin policy.

## Required API endpoints

The page calls:

- `GET /v1/models` (ping)
- `GET /v1/vision/models` (list cached models + active model)
- `POST /v1/vision/model/load` (load a model into memory)
- `POST /v1/vision/jobs/images/generations` (start a text→image job)
- `POST /v1/vision/jobs/videos/generations` (start a text→video job)
- `POST /v1/vision/jobs/images/edits` (start an image→image job)
- `GET /v1/vision/jobs/{job_id}` (poll job status)
  - on success, the page calls `GET /v1/vision/jobs/{job_id}?consume=1` to fetch-and-consume the result

## 1) Start the local playground server

From an AbstractVision checkout:

```bash
PYTHONPATH=src python -m abstractvision playground --port 8091
```

Or, when installed:

```bash
abstractvision playground --port 8091
```

Quick sanity checks (should return JSON):

```bash
curl -s http://127.0.0.1:8091/v1/models | head
curl -s http://127.0.0.1:8091/v1/vision/models | head
```

## 2) Open the page

Open:

- `http://127.0.0.1:8091/vision_playground.html`

Usage notes:
- The page is split into task tabs (`Text→Image`, `Image→Image`, `Text→Video`, and a placeholder `Image→Video` tab for later work).
- Each active task tab has its own model selector and unload button.
- Selecting a different model unloads the current active backend first, then auto-loads the replacement to keep memory usage bounded.
- Raw Hugging Face model ids such as `runwayml/stable-diffusion-v1-5` load directly; no `diffusers/` provider prefix is required.
- For first tests, prefer a small cached model such as Stable Diffusion 1.5 before loading larger Qwen/FLUX models.
- The Image→Image panel is enabled only for models that both advertise `image_to_image` and remain enabled by backend runtime truth.
- MLX-Gen models are surfaced only for tasks the backend currently enables: curated q4/q8 text-to-image presets plus image-to-image for FLUX.2 klein/base and Qwen Image Edit (no masks yet).
- The bundled local `Text→Video` tab is experimental and currently expected to have no shipped local model options until [`docs/backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md`](../docs/backlog/planned/0023_local_runtime_capability_quarantine_for_glm_mflux_and_t2v.md) is resolved.
- Response logs intentionally show a shortened `b64_json` preview instead of the full base64 payload.
- “Extra JSON” is forwarded to the server:
  - T2I: merged into the JSON request body
  - I2I: sent as a string field `extra_json` in the multipart form body
- Model-specific request fixes happen in the backend/API layer, so the same normalization rules apply to the playground, CLI/REPL, and AbstractCore integration.
- Local video export requires an `ffmpeg` executable on `PATH` whenever a local backend emits frames for packaging.

## 3) stable-diffusion.cpp / GGUF notes

If your server is configured to run GGUF diffusion models via stable-diffusion.cpp, you’ll typically need:
- a diffusion model (`.gguf`)
- a VAE (`.safetensors`) for some families (e.g. Qwen Image GGUF)
- a text encoder/LLM (`.gguf`) for some families (e.g. Qwen Image GGUF)

Exact configuration is backend-specific; check AbstractVision’s backend docs.
