from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

from ..types import (
    GeneratedAsset,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageToVideoRequest,
    ImageUpscaleRequest,
    MultiAngleRequest,
    ProviderAdapterInfo,
    ProviderModelInfo,
    VideoGenerationRequest,
    VisionBackendCapabilities,
)
from ..errors import CapabilityNotSupportedError


class VisionBackend(ABC):
    """Backend interface for generative vision tasks."""

    def normalize_image_generation_request(
        self,
        request: ImageGenerationRequest,
    ) -> ImageGenerationRequest:
        """Best-effort request normalization before execution."""
        return request

    def normalize_image_edit_request(
        self,
        request: ImageEditRequest,
    ) -> ImageEditRequest:
        """Best-effort request normalization before execution."""
        return request

    def normalize_video_generation_request(
        self,
        request: VideoGenerationRequest,
    ) -> VideoGenerationRequest:
        """Best-effort request normalization before execution."""
        return request

    def normalize_image_to_video_request(
        self,
        request: ImageToVideoRequest,
    ) -> ImageToVideoRequest:
        """Best-effort request normalization before execution."""
        return request

    def normalize_image_upscale_request(
        self,
        request: ImageUpscaleRequest,
    ) -> ImageUpscaleRequest:
        """Best-effort request normalization before execution."""
        return request

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

    def generate_video_with_progress(
        self,
        request: VideoGenerationRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        """Generate a video, optionally reporting progress (best-effort)."""
        _ = progress_callback
        return self.generate_video(request)

    def image_to_video_with_progress(
        self,
        request: ImageToVideoRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        """Generate a video from an image, optionally reporting progress (best-effort)."""
        _ = progress_callback
        return self.image_to_video(request)

    def upscale_image_with_progress(
        self,
        request: ImageUpscaleRequest,
        progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    ) -> GeneratedAsset:
        """Upscale an image, optionally reporting progress (best-effort)."""
        _ = progress_callback
        return self.upscale_image(request)

    def get_capabilities(self) -> Optional[VisionBackendCapabilities]:
        """Return backend-level capability constraints (optional)."""
        return None

    def list_provider_models(self, *, task: Optional[str] = None) -> Sequence[ProviderModelInfo]:
        """Return provider-advertised model entries, when the backend can query them.

        This is explicit provider catalog discovery. Backends must not use it to
        silently select or change the configured model.
        """
        _ = task
        return ()

    def list_provider_adapters(
        self,
        *,
        model: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Sequence[ProviderAdapterInfo]:
        """Return backend-discoverable adapter entries, when supported."""
        _ = model
        _ = task
        return ()

    def preload(self) -> None:
        """Best-effort eager load/prepare; does not guarantee a fully warmed first inference."""
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

    def upscale_image(self, request: ImageUpscaleRequest) -> GeneratedAsset:
        _ = request
        raise CapabilityNotSupportedError(
            f"{self.__class__.__name__} does not support image_upscale."
        )
