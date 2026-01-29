from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

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

    def generate_image_with_progress(
        self,
        request: ImageGenerationRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        """Generate an image, optionally reporting progress (best-effort)."""
        _ = progress_callback
        return self.generate_image(request)

    def edit_image_with_progress(
        self,
        request: ImageEditRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        """Edit an image, optionally reporting progress (best-effort)."""
        _ = progress_callback
        return self.edit_image(request)

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
