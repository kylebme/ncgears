"""Exceptions raised by the public ncgear API."""


class NcgearError(Exception):
    """Base class for ncgear errors."""


class GeneratorNotFoundError(NcgearError):
    """Legacy error retained for callers of :func:`native_generator`."""


class GenerationError(NcgearError):
    """Raised when the requested design cannot be generated."""
