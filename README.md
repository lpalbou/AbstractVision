# AbstractVision

Model-agnostic generative vision API (images, optional video) for Python and the Abstract* ecosystem.

## What you get

- A small orchestration API: [`VisionManager`](src/abstractvision/vision_manager.py)
- A packaged capability registry (“what models can do”): [`VisionModelCapabilitiesRegistry`](src/abstractvision/model_capabilities.py) backed by [`vision_model_capabilities.json`](src/abstractvision/assets/vision_model_capabilities.json)
- Optional artifact-ref outputs (small JSON refs): [`LocalAssetStore`](src/abstractvision/artifacts.py) and [`RuntimeArtifactStoreAdapter`](src/abstractvision/artifacts.py)
- Built-in backends (execution engines): [`src/abstractvision/backends/`](src/abstractvision/backends/)
  - OpenAI-compatible HTTP: [`openai_compatible.py`](src/abstractvision/backends/openai_compatible.py)
  - Local Diffusers: [`huggingface_diffusers.py`](src/abstractvision/backends/huggingface_diffusers.py)
  - Local stable-diffusion.cpp / GGUF: [`stable_diffusion_cpp.py`](src/abstractvision/backends/stable_diffusion_cpp.py)
- CLI/REPL for manual testing: [`abstractvision`](src/abstractvision/cli.py)
- Optional static Playground UI (server-backed): [`playground/vision_playground.html`](playground/vision_playground.html) (docs: [`playground/README.md`](playground/README.md))

## How it fits together (diagram)

```mermaid
flowchart LR
  Caller[Python / CLI / AbstractCore] --> VM[VisionManager]
  VM --> BE[VisionBackend]
  BE --> VM
  VM -->|optional| Store[MediaStore]
  Store --> Ref[Artifact ref dict]
  VM -->|no store| Asset[GeneratedAsset (bytes + mime)]
```

## Status (current backend support)

- Development status: **Alpha** (0.x). The public API is stable-by-design, but breaking changes may still happen and will be called out in `CHANGELOG.md`.
- Built-in backends implement: `text_to_image` and `image_to_image`.
- Video (`text_to_video`, `image_to_video`) is supported only via the OpenAI-compatible backend **when** endpoints are configured.
- `multi_view_image` is part of the public API (`VisionManager.generate_angles`) but no built-in backend implements it yet.

Details: [`docs/reference/backends.md`](docs/reference/backends.md).

## Installation

```bash
pip install abstractvision
```

Install optional integrations:

```bash
pip install "abstractvision[abstractcore]"
```

If you hit “missing pipeline class” errors for newer model families, see [`docs/getting-started.md`](docs/getting-started.md). In practice you usually need Diffusers from source (`main`):

```bash
pip install -U "abstractvision[huggingface-dev]"
pip install -U "git+https://github.com/huggingface/diffusers@main"
```

For local dev (from a repo checkout):

```bash
pip install -e .
```

## Usage

Start here:
- Getting started: [`docs/getting-started.md`](docs/getting-started.md)
- FAQ: [`docs/faq.md`](docs/faq.md)
- API reference: [`docs/api.md`](docs/api.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)
- Docs index: [`docs/README.md`](docs/README.md)

### Capability-driven model selection

```python
from abstractvision import VisionModelCapabilitiesRegistry

reg = VisionModelCapabilitiesRegistry()
assert reg.supports("Qwen/Qwen-Image-2512", "text_to_image")

print(reg.list_tasks())
print(reg.models_for_task("text_to_image"))
```

### Backend wiring + generation (artifact outputs)

The default install is “batteries included” (Torch + Diffusers + stable-diffusion.cpp python bindings), but heavy
modules are imported lazily (see [`src/abstractvision/backends/__init__.py`](src/abstractvision/backends/__init__.py)).

```python
from abstractvision import LocalAssetStore, VisionManager, VisionModelCapabilitiesRegistry, is_artifact_ref
from abstractvision.backends import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend

reg = VisionModelCapabilitiesRegistry()

backend = OpenAICompatibleVisionBackend(
    config=OpenAICompatibleBackendConfig(
        base_url="http://localhost:1234/v1",
        api_key="YOUR_KEY",      # optional for local servers
        model_id="REMOTE_MODEL", # optional (server-dependent)
    )
)

vm = VisionManager(
    backend=backend,
    store=LocalAssetStore(),         # enables artifact-ref outputs
    model_id="zai-org/GLM-Image",    # optional: capability gating
    registry=reg,                   # optional: reuse loaded registry
)

out = vm.generate_image("a cinematic photo of a red fox in snow")
assert is_artifact_ref(out)
print(out)  # {"$artifact": "...", "content_type": "...", ...}

png_bytes = vm.store.load_bytes(out["$artifact"])  # type: ignore[union-attr]
```

### Interactive testing (CLI / REPL)

```bash
abstractvision models
abstractvision tasks
abstractvision show-model zai-org/GLM-Image

abstractvision repl
```

Inside the REPL:

```text
/backend openai http://localhost:1234/v1
/cap-model zai-org/GLM-Image
/set width 1024
/set height 1024
/t2i "a watercolor painting of a lighthouse" --open
```

The CLI/REPL can also be configured via `ABSTRACTVISION_*` env vars; see [`docs/reference/configuration.md`](docs/reference/configuration.md).

One-shot commands (OpenAI-compatible HTTP backend only):

```bash
abstractvision t2i --base-url http://localhost:1234/v1 "a studio photo of an espresso machine"
abstractvision i2i --base-url http://localhost:1234/v1 --image ./input.png "make it watercolor"
```

#### Local GGUF via stable-diffusion.cpp

If you want to run GGUF diffusion models locally (e.g. Qwen Image), use the stable-diffusion.cpp backend (`sdcpp`).

Recommended (pip-only; no external binary download): `pip install abstractvision` already includes the stable-diffusion.cpp python bindings (`stable-diffusion-cpp-python`).

Alternative (external executable):

- Install `sd-cli`: <https://github.com/leejet/stable-diffusion.cpp/releases>

In the REPL:

```text
/backend sdcpp /path/to/qwen-image-2512-Q4_K_M.gguf /path/to/qwen_image_vae.safetensors /path/to/Qwen2.5-VL-7B-Instruct-*.gguf
/t2i "a watercolor painting of a lighthouse" --sampling-method euler --offload-to-cpu --diffusion-fa --flow-shift 3 --open
```

Extra flags are forwarded via `request.extra`. In CLI mode they are forwarded to `sd-cli`; in python bindings mode, keys are mapped to python binding kwargs when supported and unsupported keys are ignored.

### AbstractCore tool integration (artifact refs)

If you’re using AbstractCore tool calling, AbstractVision can expose vision tasks as tools:

```python
from abstractvision.integrations.abstractcore import make_vision_tools

tools = make_vision_tools(vision_manager=vm, model_id="zai-org/GLM-Image")
```

## AbstractFramework ecosystem

AbstractVision is part of the **AbstractFramework** ecosystem and is designed to compose with:

- **AbstractFramework** (project hub): <https://github.com/lpalbou/AbstractFramework>
- **AbstractCore** (orchestration + tool calling): <https://github.com/lpalbou/abstractcore>
- **AbstractRuntime** (runtime services, including artifact storage): <https://github.com/lpalbou/abstractruntime>

In practice:
- AbstractVision standardizes *generative vision outputs* (image/video) behind `VisionManager`.
- AbstractCore can discover and use AbstractVision via the capability plugin (`src/abstractvision/integrations/abstractcore_plugin.py`) or you can expose vision tasks as tools (`src/abstractvision/integrations/abstractcore.py`).
- Artifact refs returned by AbstractVision are designed to travel across processes; `RuntimeArtifactStoreAdapter` bridges to an AbstractRuntime-style artifact store (`src/abstractvision/artifacts.py`).

## Project

- Release notes: [`CHANGELOG.md`](CHANGELOG.md)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security: [`SECURITY.md`](SECURITY.md)
- Acknowledgments: [`ACKNOWLEDMENTS.md`](ACKNOWLEDMENTS.md)

## Requirements

- Python >= 3.8

## License

MIT License - see LICENSE file for details.

## Author

Laurent-Philippe Albou

## Contact

contact@abstractcore.ai
