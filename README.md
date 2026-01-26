# AbstractVision

Model-agnostic generative vision abstractions (image + video) for the Abstract* ecosystem.

## Overview

AbstractVision provides:
- A stable task API (`VisionManager`) for: text→image, image→image, multi-view image, text→video, image→video
- A single source of truth for “what models can do” (`VisionModelCapabilitiesRegistry`)
- Artifact-first outputs (small JSON refs) for tool calling + workflows + third-party integrations

The project’s responsibility is to:
- define a clean, stable task API (t2i, i2i, t2v, i2v, multi-view)
- describe what models can do via a single source of truth (`vision_model_capabilities.json`)

The user’s responsibility is to choose which model(s) to use (including any license/compliance considerations).

## Installation

```bash
pip install abstractvision
```

For local dev (from a repo checkout):

```bash
pip install -e .
```

## Usage

See `docs/getting-started.md` for step-by-step local generation with Diffusers, GGUF (`sd-cli`), and the web playground.

### Capability-driven model selection

```python
from abstractvision import VisionModelCapabilitiesRegistry

reg = VisionModelCapabilitiesRegistry()
assert reg.supports("Qwen/Qwen-Image-2512", "text_to_image")

print(reg.list_tasks())
print(reg.models_for_task("text_to_image"))
```

### Backend wiring + generation (artifact outputs)

The default install includes a dependency-light OpenAI-compatible HTTP backend.

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

The CLI/REPL can also be configured via env vars like `ABSTRACTVISION_BASE_URL`, `ABSTRACTVISION_API_KEY`, `ABSTRACTVISION_MODEL_ID`, and `ABSTRACTVISION_STORE_DIR`.

#### Local GGUF via stable-diffusion.cpp (`sd-cli`)

If you want to run GGUF diffusion models locally (e.g. Qwen Image), use the `sd-cli` backend.

- Install `sd-cli`: https://github.com/leejet/stable-diffusion.cpp/releases

In the REPL:

```text
/backend sdcpp /path/to/qwen-image-2512-Q4_K_M.gguf /path/to/qwen_image_vae.safetensors /path/to/Qwen2.5-VL-7B-Instruct-*.gguf sd-cli
/t2i "a watercolor painting of a lighthouse" --sampling-method euler --offload-to-cpu --diffusion-fa --flow-shift 3 --open
```

Extra flags are forwarded via `request.extra` (unknown `--foo-bar` flags become `extra={"foo_bar": ...}`).

### AbstractCore tool integration (artifact refs)

If you’re using AbstractCore tool calling, AbstractVision can expose vision tasks as tools:

```python
from abstractvision.integrations.abstractcore import make_vision_tools

tools = make_vision_tools(vision_manager=vm, model_id="zai-org/GLM-Image")
```

## Requirements

- Python >= 3.8

## License

MIT License - see LICENSE file for details.

## Author

Laurent-Philippe Albou

## Contact

contact@abstractcore.ai
