"""abstractvision: Generative vision capabilities for abstractcore.ai.

The base install is lightweight and supports OpenAI-compatible HTTP backends,
shared contracts, capability metadata, and AbstractCore plugin discovery.
Local Diffusers, stable-diffusion.cpp, and MLX-Gen runtimes are explicit extras.
"""

from .artifacts import LocalAssetStore, RuntimeArtifactStoreAdapter, is_artifact_ref
from .adapter_capabilities import VisionAdapterCapabilitiesRegistry
from .model_capabilities import VisionModelCapabilitiesRegistry
from .types import (
    ImageUpscaleRequest,
    LoRAAdapterSpec,
    ProviderAdapterInfo,
    ProviderModelInfo,
    VideoProgressEvent,
)
from .vision_manager import VisionManager

__version__ = "0.3.27"
__author__ = "Laurent-Philippe Albou"
__email__ = "contact@abstractcore.ai"

__all__ = [
    "VisionManager",
    "VisionAdapterCapabilitiesRegistry",
    "ProviderAdapterInfo",
    "ProviderModelInfo",
    "VideoProgressEvent",
    "ImageUpscaleRequest",
    "LoRAAdapterSpec",
    "VisionModelCapabilitiesRegistry",
    "LocalAssetStore",
    "RuntimeArtifactStoreAdapter",
    "is_artifact_ref",
    "__version__",
]
