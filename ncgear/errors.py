"""Exceptions raised by the public ncgear API."""


class NcgearError(Exception):
    """Base class for ncgear errors."""


class GeneratorNotFoundError(NcgearError):
    """Raised when the native geometry generator cannot be located."""


class GenerationError(NcgearError):
    """Raised when the requested design cannot be generated."""
