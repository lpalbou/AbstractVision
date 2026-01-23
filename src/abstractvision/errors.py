class AbstractVisionError(Exception):
    """Base exception for the abstractvision package."""


class BackendNotConfiguredError(AbstractVisionError):
    """Raised when a VisionManager method is called without a configured backend."""


class OptionalDependencyMissingError(AbstractVisionError):
    """Raised when an optional backend dependency is missing."""


class UnknownModelError(AbstractVisionError):
    """Raised when a model id is not present in the capability registry."""


class CapabilityNotSupportedError(AbstractVisionError):
    """Raised when a model/backend cannot satisfy a requested generative capability."""

