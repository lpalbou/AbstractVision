# AbstractVision documentation

This folder contains the user-facing documentation for `abstractvision`.

## Start here (new users)

1) [Project overview + quickstart](../README.md)  
2) [Getting started](getting-started.md) (shell and interactive `t2i`/`i2i`; then Diffusers, MLX-Gen, GGUF, OpenAI-compatible HTTP, Playground)
3) [Architecture](architecture.md) (how the pieces fit together)

## Quick reference

- [FAQ](faq.md)
- [Troubleshooting](troubleshooting.md)
- [API reference](api.md)
- [Architecture decisions](adr/README.md)
- [Backends](reference/backends.md)
- [Configuration (CLI/REPL env vars + flags)](reference/configuration.md)
- [Capability registry (`vision_model_capabilities.json`)](reference/capabilities-registry.md)
- [Artifacts (artifact refs + stores)](reference/artifacts.md)
- [AbstractCore integration (capability plugin + tools)](reference/abstractcore-integration.md)
- Agent-oriented docs: [`../llms.txt`](../llms.txt) and [`../llms-full.txt`](../llms-full.txt)

## AbstractFramework ecosystem

AbstractVision is part of the **AbstractFramework** ecosystem and is designed to compose with:

- **AbstractFramework** (project hub): <https://github.com/lpalbou/AbstractFramework>
- **AbstractCore** (orchestration + tool calling): <https://github.com/lpalbou/abstractcore>
- **AbstractRuntime** (runtime services, including artifact storage): <https://github.com/lpalbou/abstractruntime>

## Current implementation status (as shipped)

Public API surface: [`VisionManager`](../src/abstractvision/vision_manager.py) exposes:
- `generate_image` (`text_to_image`), `edit_image` (`image_to_image`)
- `generate_video` (`text_to_video`), `image_to_video` (`image_to_video`) (backend-dependent)
- `generate_angles` (`multi_view_image`) (API exists; no built-in backend implements it yet)

Built-in backends implement:
- **Images**: Diffusers, stable-diffusion.cpp, MLX-Gen, OpenAI-compatible HTTP ([`../src/abstractvision/backends/`](../src/abstractvision/backends/))
- **Current local policy**: MLX-Gen supports curated q4/q8 Apple Silicon image presets, official FIBO image snapshots, and Wan 2.2 TI2V video; local Diffusers `text_to_video` is experimental and temporarily disabled from normal local surfaces.
- **Video**:
  - MLX-Gen for Wan 2.2 TI2V local `text_to_video` and first-frame `image_to_video`
  - OpenAI-compatible HTTP for optional `text_to_video` / `image_to_video` when endpoints are configured ([`openai_compatible.py`](../src/abstractvision/backends/openai_compatible.py))

If you’re looking for “what can model X do?”, the single source of truth is the packaged registry:
[`../src/abstractvision/assets/vision_model_capabilities.json`](../src/abstractvision/assets/vision_model_capabilities.json) (loaded by `VisionModelCapabilitiesRegistry` in [`../src/abstractvision/model_capabilities.py`](../src/abstractvision/model_capabilities.py)).
The curation and cross-platform support policy for that registry is governed by
[ADR 0005](adr/0005_curated_capability_registry_and_download_catalog.md).

## Internal engineering notes

[`docs/adr/`](adr/) contains durable engineering policy.

[`docs/backlog/`](backlog/) is an internal log (planned work + completion reports). It is not the normative user documentation surface.

## Project

- Release notes: [`CHANGELOG.md`](../CHANGELOG.md)
- Contributing: [`CONTRIBUTING.md`](../CONTRIBUTING.md)
- Security: [`SECURITY.md`](../SECURITY.md)
- License: [`LICENSE`](../LICENSE)
- Acknowledgments: [`ACKNOWLEDGMENTS.md`](../ACKNOWLEDGMENTS.md)
