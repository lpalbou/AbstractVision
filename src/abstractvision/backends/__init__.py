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
    "MLXGenBackendConfig",
    "MLXGenVisionBackend",
    "MFluxBackendConfig",
    "MFluxVisionBackend",
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

    if name in {"MLXGenBackendConfig", "MLXGenVisionBackend", "MFluxBackendConfig", "MFluxVisionBackend"}:
        from .mflux import MLXGenBackendConfig, MLXGenVisionBackend, MFluxBackendConfig, MFluxVisionBackend

        if name == "MLXGenBackendConfig":
            return MLXGenBackendConfig
        if name == "MLXGenVisionBackend":
            return MLXGenVisionBackend
        return MFluxBackendConfig if name == "MFluxBackendConfig" else MFluxVisionBackend

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
