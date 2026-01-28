from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..types import (
    GeneratedAsset,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageToVideoRequest,
    MultiAngleRequest,
    VideoGenerationRequest,
    VisionBackendCapabilities,
)


class VisionBackend(ABC):
    """Backend interface for generative vision tasks."""

    def get_capabilities(self) -> Optional[VisionBackendCapabilities]:
        """Return backend-level capability constraints (optional)."""
        return None

    def preload(self) -> None:
        """Best-effort: load model weights into memory for faster first inference."""
        return None

    def unload(self) -> None:
        """Best-effort: release model weights from memory."""
        return None

    @abstractmethod
    def generate_image(self, request: ImageGenerationRequest) -> GeneratedAsset: ...

    @abstractmethod
    def edit_image(self, request: ImageEditRequest) -> GeneratedAsset: ...

    @abstractmethod
    def generate_angles(self, request: MultiAngleRequest) -> list[GeneratedAsset]: ...

    @abstractmethod
    def generate_video(self, request: VideoGenerationRequest) -> GeneratedAsset: ...

    @abstractmethod
    def image_to_video(self, request: ImageToVideoRequest) -> GeneratedAsset: ...
