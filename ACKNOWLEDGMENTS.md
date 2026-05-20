# Acknowledgments

AbstractVision stands on the shoulders of excellent open-source projects and communities.

## Optional runtime dependencies (declared as extras)

- **Hugging Face Diffusers** (local pipeline runtime; used by the Diffusers backend): [`src/abstractvision/backends/huggingface_diffusers.py`](src/abstractvision/backends/huggingface_diffusers.py) (declared in the `diffusers`/`local`/`all` extras)
- **PyTorch** (tensor runtime for local inference; used via Diffusers): [`src/abstractvision/backends/huggingface_diffusers.py`](src/abstractvision/backends/huggingface_diffusers.py) (declared in the `diffusers`/`local`/`all` extras)
- **Hugging Face Transformers** (tokenizers/encoders used by some diffusion pipelines; imported by the Diffusers backend): [`src/abstractvision/backends/huggingface_diffusers.py`](src/abstractvision/backends/huggingface_diffusers.py) (declared in the `diffusers`/`local`/`all` extras)
- **Accelerate** (installed for ecosystem compatibility; used transitively by some pipelines): declared in optional extras in `pyproject.toml`
- **Safetensors** (model weight format support; used by Diffusers/Transformers): declared in optional extras in `pyproject.toml`
- **SentencePiece** (T5/tokenizer support for some model families): declared in optional extras in `pyproject.toml`
- **protobuf** (runtime dependency for some tokenizers/pipelines): declared in optional extras in `pyproject.toml`
- **einops** (tensor ops used by some modern architectures): declared in optional extras in `pyproject.toml`
- **PEFT** (LoRA adapter support used by Diffusers): declared in optional extras in `pyproject.toml`
- **Pillow** (image I/O utilities used by local backends): [`src/abstractvision/backends/huggingface_diffusers.py`](src/abstractvision/backends/huggingface_diffusers.py), [`src/abstractvision/backends/stable_diffusion_cpp.py`](src/abstractvision/backends/stable_diffusion_cpp.py) (declared in optional extras in `pyproject.toml`)
- **stable-diffusion-cpp-python** (python bindings used when `sd-cli` is not available): [`src/abstractvision/backends/stable_diffusion_cpp.py`](src/abstractvision/backends/stable_diffusion_cpp.py) (declared in the `sdcpp`/`local`/`all` extras)

## Runtime dependencies (transitive but central)

- **huggingface_hub** (model and adapter downloads; used by Diffusers/Transformers pipelines)

## Upstream projects

- **stable-diffusion.cpp** (upstream project that provides `sd-cli` and the core GGUF runtime wrapped by the bindings): [`src/abstractvision/backends/stable_diffusion_cpp.py`](src/abstractvision/backends/stable_diffusion_cpp.py)

## Referenced model/component publishers

- **Comfy-Org** (component artifacts referenced by docs and download helpers for `stable-diffusion.cpp` examples, notably Qwen Image and FLUX.2 VAE files): [`docs/getting-started.md`](docs/getting-started.md), [`scripts/download_model_sets.py`](scripts/download_model_sets.py)
- **Black Forest Labs** (official FLUX.2 upstream weights and side artifacts referenced by the capability registry and local-backend docs): [`src/abstractvision/assets/vision_model_capabilities.json`](src/abstractvision/assets/vision_model_capabilities.json), [`docs/getting-started.md`](docs/getting-started.md)
- **Qwen** (official Qwen Image / Qwen Image Edit upstream weights referenced by the capability registry and `stable-diffusion.cpp` component-mode docs): [`src/abstractvision/assets/vision_model_capabilities.json`](src/abstractvision/assets/vision_model_capabilities.json), [`docs/getting-started.md`](docs/getting-started.md)
- **Unsloth** (community GGUF conversions and companion encoder files referenced by the capability registry and some GGUF download presets): [`src/abstractvision/assets/vision_model_capabilities.json`](src/abstractvision/assets/vision_model_capabilities.json), [`scripts/download_model_sets.py`](scripts/download_model_sets.py)
- **leejet** (stable-diffusion.cpp maintainer-published runtime and GGUF conversions referenced by the `sdcpp` backend docs and GGUF download presets): [`src/abstractvision/backends/stable_diffusion_cpp.py`](src/abstractvision/backends/stable_diffusion_cpp.py), [`scripts/download_model_sets.py`](scripts/download_model_sets.py)

## Optional integrations

- **AbstractCore** (tool integration helpers + capability plugin): [`src/abstractvision/integrations/`](src/abstractvision/integrations/) (optional dependency in [`pyproject.toml`](pyproject.toml))

## Packaging

- **setuptools** and **wheel** (build system): [`pyproject.toml`](pyproject.toml)

## Community and contributors

Thanks to everyone who reports issues, suggests improvements, and contributes fixes or documentation updates.
