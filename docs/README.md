# AbstractVision documentation

This folder contains the user-facing documentation for `abstractvision`.

## Start here (new users)

1) [Project overview + quickstart](../README.md)  
2) [Getting started](getting-started.md) (first image; Diffusers, stable-diffusion.cpp, OpenAI-compatible HTTP, Playground)  
3) [Architecture](architecture.md) (how the pieces fit together)

## Quick reference

- [FAQ](faq.md)
- [API reference](api.md)
- [Backends](reference/backends.md)
- [Configuration (CLI/REPL env vars + flags)](reference/configuration.md)
- [Capability registry (`vision_model_capabilities.json`)](reference/capabilities-registry.md)
- [Artifacts (artifact refs + stores)](reference/artifacts.md)
- [AbstractCore integration (capability plugin + tools)](reference/abstractcore-integration.md)

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
- Built-in backends implement:
  - **Images**: Diffusers, stable-diffusion.cpp, OpenAI-compatible HTTP ([`../src/abstractvision/backends/`](../src/abstractvision/backends/))
  - **Video**: OpenAI-compatible HTTP only, and only when endpoints are configured ([`openai_compatible.py`](../src/abstractvision/backends/openai_compatible.py))

If you’re looking for “what can model X do?”, the single source of truth is the packaged registry:
[`../src/abstractvision/assets/vision_model_capabilities.json`](../src/abstractvision/assets/vision_model_capabilities.json) (loaded by `VisionModelCapabilitiesRegistry` in [`../src/abstractvision/model_capabilities.py`](../src/abstractvision/model_capabilities.py)).

## Internal engineering notes

[`docs/backlog/`](backlog/) is an internal log (planned work + completion reports). It is not the normative user documentation surface.

## Project

- Release notes: [`CHANGELOG.md`](../CHANGELOG.md)
- Contributing: [`CONTRIBUTING.md`](../CONTRIBUTING.md)
- Security: [`SECURITY.md`](../SECURITY.md)
- License: [`LICENSE`](../LICENSE)
- Acknowledgments: [`ACKNOWLEDMENTS.md`](../ACKNOWLEDMENTS.md)
