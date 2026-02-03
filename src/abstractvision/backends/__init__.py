"""Backend exports.

Important: this package must stay import-light.

Some backends are intentionally heavy (Torch/Diffusers). Import them lazily so
`import abstractvision` (and AbstractCore plugin discovery) does not pull GPU
stacks unless the caller explicitly selects a local backend.
"""

from .base_backend import VisionBackend

__all__ = [
    "VisionBackend",
    "OpenAICompatibleBackendConfig",
    "OpenAICompatibleVisionBackend",
    "HuggingFaceDiffusersBackendConfig",
    "HuggingFaceDiffusersVisionBackend",
    "StableDiffusionCppBackendConfig",
    "StableDiffusionCppVisionBackend",
]


def __getattr__(name: str):
    if name in {"OpenAICompatibleBackendConfig", "OpenAICompatibleVisionBackend"}:
        from .openai_compatible import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend

        return OpenAICompatibleBackendConfig if name == "OpenAICompatibleBackendConfig" else OpenAICompatibleVisionBackend

    if name in {"StableDiffusionCppBackendConfig", "StableDiffusionCppVisionBackend"}:
        from .stable_diffusion_cpp import StableDiffusionCppBackendConfig, StableDiffusionCppVisionBackend

        return StableDiffusionCppBackendConfig if name == "StableDiffusionCppBackendConfig" else StableDiffusionCppVisionBackend

    if name in {"HuggingFaceDiffusersBackendConfig", "HuggingFaceDiffusersVisionBackend"}:
        from .huggingface_diffusers import HuggingFaceDiffusersBackendConfig, HuggingFaceDiffusersVisionBackend

        return (
            HuggingFaceDiffusersBackendConfig
            if name == "HuggingFaceDiffusersBackendConfig"
            else HuggingFaceDiffusersVisionBackend
        )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
