"""abstractvision: Generative vision capabilities for abstractcore.ai.

The default install is batteries-included (Diffusers + stable-diffusion.cpp python bindings),
so users generally only need to download model weights.
"""

from .artifacts import LocalAssetStore, RuntimeArtifactStoreAdapter, is_artifact_ref
from .model_capabilities import VisionModelCapabilitiesRegistry
from .vision_manager import VisionManager

__version__ = "0.2.1"
__author__ = "Laurent-Philippe Albou"
__email__ = "contact@abstractcore.ai"

__all__ = [
    "VisionManager",
    "VisionModelCapabilitiesRegistry",
    "LocalAssetStore",
    "RuntimeArtifactStoreAdapter",
    "is_artifact_ref",
    "__version__",
]
