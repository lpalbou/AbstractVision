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
- You must **select a cached model** before running inference; selecting a different model auto-loads and switches to it.
- Raw Hugging Face model ids such as `runwayml/stable-diffusion-v1-5` load directly; no `diffusers/` provider prefix is required.
- For first tests, prefer a small cached model such as Stable Diffusion 1.5 before loading larger Qwen/FLUX models.
- The Image→Image panel is enabled only for models that advertise `image_to_image` in the packaged capability registry metadata returned by `/v1/vision/models`.
- Response logs intentionally show a shortened `b64_json` preview instead of the full base64 payload.
- “Extra JSON” is forwarded to the server:
  - T2I: merged into the JSON request body
  - I2I: sent as a string field `extra_json` in the multipart form body
- Model-specific request fixes happen in the backend/API layer, so the same normalization rules apply to the playground, CLI/REPL, and AbstractCore integration.

## 3) stable-diffusion.cpp / GGUF notes

If your server is configured to run GGUF diffusion models via stable-diffusion.cpp, you’ll typically need:
- a diffusion model (`.gguf`)
- a VAE (`.safetensors`) for some families (e.g. Qwen Image GGUF)
- a text encoder/LLM (`.gguf`) for some families (e.g. Qwen Image GGUF)

Exact configuration is backend-specific; check AbstractVision’s backend docs.
