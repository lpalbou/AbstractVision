# AbstractVision

Model-agnostic generative vision API (images, optional video) for Python and the Abstract* ecosystem.

## What you get

- A stable task API: `VisionManager` (`src/abstractvision/vision_manager.py`)
- A packaged capability registry (“what models can do”): `VisionModelCapabilitiesRegistry` backed by `src/abstractvision/assets/vision_model_capabilities.json`
- Optional artifact-ref outputs (small JSON refs): `LocalAssetStore` / store adapters (`src/abstractvision/artifacts.py`)
- Built-in backends (`src/abstractvision/backends/`):
  - OpenAI-compatible HTTP (`openai_compatible.py`)
  - Local Diffusers (`huggingface_diffusers.py`)
  - Local stable-diffusion.cpp / GGUF (`stable_diffusion_cpp.py`)
- CLI/REPL for manual testing: `abstractvision ...` (`src/abstractvision/cli.py`)

## Status (current backend support)

- Built-in backends implement: `text_to_image` and `image_to_image`.
- Video (`text_to_video`, `image_to_video`) is supported only via the OpenAI-compatible backend **when** endpoints are configured.
- `multi_view_image` is part of the public API (`VisionManager.generate_angles`) but no built-in backend implements it yet.

Details: `docs/reference/backends.md`.

## Installation

```bash
pip install abstractvision
```

Install optional integrations:

```bash
pip install "abstractvision[abstractcore]"
```

Some newer model pipelines may require Diffusers from GitHub `main` (see `docs/getting-started.md`):

```bash
pip install -U "abstractvision[huggingface-dev]"
```

For local dev (from a repo checkout):

```bash
pip install -e .
```

## Usage

Start here:
- Getting started: `docs/getting-started.md`
- FAQ: `docs/faq.md`
- API reference: `docs/api.md`
- Architecture: `docs/architecture.md`
- Docs index: `docs/README.md`

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
modules are imported lazily (see `src/abstractvision/backends/__init__.py`).

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

The CLI/REPL can also be configured via `ABSTRACTVISION_*` env vars; see `docs/reference/configuration.md`.

One-shot commands (OpenAI-compatible HTTP backend only):

```bash
abstractvision t2i --base-url http://localhost:1234/v1 "a studio photo of an espresso machine"
abstractvision i2i --base-url http://localhost:1234/v1 --image ./input.png "make it watercolor"
```

#### Local GGUF via stable-diffusion.cpp

If you want to run GGUF diffusion models locally (e.g. Qwen Image), use the stable-diffusion.cpp backend (`sdcpp`).

Recommended (pip-only; no external binary download): `pip install abstractvision` already includes the stable-diffusion.cpp python bindings (`stable-diffusion-cpp-python`).

Alternative (external executable):

- Install `sd-cli`: https://github.com/leejet/stable-diffusion.cpp/releases

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

## Project

- Release notes: `CHANGELOG.md`
- Contributing: `CONTRIBUTING.md`
- Security: `SECURITY.md`
- Acknowledgments: `ACKNOWLEDMENTS.md`

## Requirements

- Python >= 3.8

## License

MIT License - see LICENSE file for details.

## Author

Laurent-Philippe Albou

## Contact

contact@abstractcore.ai
