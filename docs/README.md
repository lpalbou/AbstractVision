# AbstractVision documentation

Start here if you’re new:

1) `README.md` (project overview + quickstart)
2) `docs/getting-started.md` (first image; Diffusers, stable-diffusion.cpp, OpenAI-compatible HTTP, playground)
3) `docs/architecture.md` (how the pieces fit together)

## Reference

- Backends: `docs/reference/backends.md`
- Configuration (CLI/REPL env vars + flags): `docs/reference/configuration.md`
- Capability registry (`vision_model_capabilities.json`): `docs/reference/capabilities-registry.md`
- Artifact refs + stores: `docs/reference/artifacts.md`
- AbstractCore integration (capability plugin + tools): `docs/reference/abstractcore-integration.md`

## Current implementation status (as shipped)

- Public API surface: `VisionManager` (`src/abstractvision/vision_manager.py`) exposes:
  - `generate_image` (`text_to_image`), `edit_image` (`image_to_image`)
  - `generate_video` (`text_to_video`), `image_to_video` (`image_to_video`) (backend-dependent)
  - `generate_angles` (`multi_view_image`) (API exists; no built-in backend implements it yet)
- Built-in backends implement:
  - **Images**: Diffusers, stable-diffusion.cpp, OpenAI-compatible HTTP (`src/abstractvision/backends/*`)
  - **Video**: OpenAI-compatible HTTP only, and only when endpoints are configured (`src/abstractvision/backends/openai_compatible.py`)

If you’re looking for “what can model X do?”, the single source of truth is the packaged registry:
`src/abstractvision/assets/vision_model_capabilities.json` (loaded by `VisionModelCapabilitiesRegistry` in `src/abstractvision/model_capabilities.py`).

## Internal engineering notes

`docs/backlog/` is an internal log (planned work + completion reports). It is not the normative user documentation surface.
