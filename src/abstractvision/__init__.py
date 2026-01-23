"""abstractvision: Generative vision capabilities for abstractcore.ai.

This package is intentionally lightweight: the default install only ships the
stable API contract + capability registry. Heavy ML runtimes live behind
optional backends/extras.
"""

from .artifacts import LocalAssetStore, RuntimeArtifactStoreAdapter, is_artifact_ref
from .model_capabilities import VisionModelCapabilitiesRegistry
from .vision_manager import VisionManager

__version__ = "0.1.0"
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
