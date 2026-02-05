# Acknowledgments

AbstractVision stands on the shoulders of excellent open-source projects and communities.

## Direct dependencies (runtime)

- **Hugging Face Diffusers** (local pipeline runtime; used by the Diffusers backend): `src/abstractvision/backends/huggingface_diffusers.py` (declared in `pyproject.toml`)
- **PyTorch** (tensor runtime for local inference; used via Diffusers): `src/abstractvision/backends/huggingface_diffusers.py` (declared in `pyproject.toml`)
- **Hugging Face Transformers** (tokenizers/encoders used by some diffusion pipelines; imported by the Diffusers backend): `src/abstractvision/backends/huggingface_diffusers.py` (declared in `pyproject.toml`)
- **Accelerate** (installed for ecosystem compatibility; used transitively by some pipelines): declared in `pyproject.toml`
- **Safetensors** (model weight format support; used by Diffusers/Transformers): declared in `pyproject.toml`
- **SentencePiece** (T5/tokenizer support for some model families): declared in `pyproject.toml`
- **protobuf** (runtime dependency for some tokenizers/pipelines): declared in `pyproject.toml`
- **einops** (tensor ops used by some modern architectures): declared in `pyproject.toml`
- **PEFT** (LoRA adapter support used by Diffusers): declared in `pyproject.toml`
- **Pillow** (image I/O utilities used by local backends): `src/abstractvision/backends/huggingface_diffusers.py`, `src/abstractvision/backends/stable_diffusion_cpp.py` (declared in `pyproject.toml`)
- **stable-diffusion.cpp** (upstream GGUF runtime used by the stable-diffusion.cpp backend): `src/abstractvision/backends/stable_diffusion_cpp.py`
- **stable-diffusion-cpp-python** (pip-installable python bindings used when `sd-cli` is not available): `src/abstractvision/backends/stable_diffusion_cpp.py` (declared in `pyproject.toml`)

## Optional integrations

- **AbstractCore** (tool integration helpers + capability plugin): `src/abstractvision/integrations/` (optional dependency in `pyproject.toml`)

## Packaging

- **setuptools** and **wheel** (build system): `pyproject.toml`

## Community and contributors

Thanks to everyone who reports issues, suggests improvements, and contributes fixes or documentation updates.
