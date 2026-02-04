# Changelog

## Unreleased

## 0.2.0

- Add stable-diffusion.cpp (`sd-cli`) backend for local GGUF diffusion models.
- REPL: forward unknown `--flags` as backend `extra` parameters.
- Add a tiny web playground (`playground/vision_playground.html`) for testing via AbstractCore Server vision endpoints (`/v1/vision/*`).

## 0.1.0

- Initial MVP: capability registry + schema validation.
- Artifact-first outputs via `LocalAssetStore` and runtime adapter.
- OpenAI-compatible HTTP backend for image generation/editing (optional video endpoints via config).
- Local Diffusers backend for images (opt-in deps).
- AbstractCore tool integration (`make_vision_tools`) with artifact refs.
- CLI/REPL for interactive manual testing.
