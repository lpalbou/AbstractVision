# AbstractVision Playground (Web)

This is a tiny, dependency-free web UI for testing an **AbstractCore Server** instance that exposes the **vision job endpoints** used by the page.

Evidence: see the fetch calls in `playground/vision_playground.html`.

## Required server endpoints

The page expects:

- `GET /v1/models` (ping)
- `GET /v1/vision/models` (list cached models + active model)
- `POST /v1/vision/model/load` (load a model into memory)
- `POST /v1/vision/model/unload` (unload the active model)
- `POST /v1/vision/jobs/images/generations` (start a text→image job)
- `POST /v1/vision/jobs/images/edits` (start an image→image job)
- `GET /v1/vision/jobs/{job_id}` (poll job status)
  - on success, the page calls `GET /v1/vision/jobs/{job_id}?consume=1` to fetch-and-consume the result

## 1) Start a compatible server

Start your AbstractCore Server (or any server that implements the endpoints above) on `http://localhost:8000` (default in the UI).

This repo does not ship the server implementation; consult AbstractCore’s documentation for installation and startup.

Quick sanity checks (should return JSON):

```bash
curl -s http://localhost:8000/v1/models | head
curl -s http://localhost:8000/v1/vision/models | head
```

## 2) Serve this page

Browsers may block `file://` → `http://` requests; serve the page locally:

```bash
cd playground
python -m http.server 8080
```

Open:

- `http://localhost:8080/vision_playground.html`

Usage notes:
- You must **select a cached model** and load it before running inference.
- “Extra JSON” is forwarded to the server:
  - T2I: merged into the JSON request body
  - I2I: sent as a string field `extra_json` in the multipart form body

## 3) stable-diffusion.cpp / GGUF notes

If your server is configured to run GGUF diffusion models via stable-diffusion.cpp, you’ll typically need:
- a diffusion model (`.gguf`)
- a VAE (`.safetensors`) for some families (e.g. Qwen Image GGUF)
- a text encoder/LLM (`.gguf`) for some families (e.g. Qwen Image GGUF)

Exact configuration is server-specific; check your server’s documentation.
