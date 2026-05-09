"""abstractvision: Generative vision capabilities for abstractcore.ai.

The base install is lightweight and supports OpenAI-compatible HTTP backends,
shared contracts, capability metadata, and AbstractCore plugin discovery.
Local Diffusers and stable-diffusion.cpp runtimes are explicit extras.
"""

from .artifacts import LocalAssetStore, RuntimeArtifactStoreAdapter, is_artifact_ref
from .model_capabilities import VisionModelCapabilitiesRegistry
from .types import ProviderModelInfo
from .vision_manager import VisionManager

__version__ = "0.3.4"
__author__ = "Laurent-Philippe Albou"
__email__ = "contact@abstractcore.ai"

__all__ = [
    "VisionManager",
    "ProviderModelInfo",
    "VisionModelCapabilitiesRegistry",
    "LocalAssetStore",
    "RuntimeArtifactStoreAdapter",
    "is_artifact_ref",
    "__version__",
]
