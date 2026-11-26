from typing import Any, Optional, Sequence


class AbstractVisionError(Exception):
    """Base exception for the abstractvision package."""


class BackendNotConfiguredError(AbstractVisionError):
    """Raised when a VisionManager method is called without a configured backend."""


class OptionalDependencyMissingError(AbstractVisionError):
    """Raised when an optional backend dependency is missing."""


class UnknownModelError(AbstractVisionError):
    """Raised when a model id is not present in the capability registry."""


class CapabilityNotSupportedError(AbstractVisionError):
    """Raised when a model/backend cannot satisfy a requested generative capability.

    When the refusal is about one request field, `field` names it and `supported` holds the
    values the backend accepts there when it knows them (empty: it accepts none); both are None
    otherwise.
    """

    def __init__(
        self, *args: Any, field: Optional[str] = None, supported: Optional[Sequence[Any]] = None
    ) -> None:
        super().__init__(*args)
        self.field = field
        self.supported = tuple(supported) if supported is not None else None


class GpuOutOfMemoryError(AbstractVisionError):
    """Raised when a local generation runs out of GPU memory; the message says what to change."""


class GenerationCancelledError(AbstractVisionError):
    """Raised when a request's `cancel_event` was set; the run stopped at a progress boundary."""
