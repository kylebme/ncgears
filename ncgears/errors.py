"""Exceptions raised by the public ncgears API."""


class ncgearsError(Exception):
    """Base class for ncgears errors."""


class GeneratorNotFoundError(ncgearsError):
    """Legacy error retained for callers of :func:`native_generator`."""


class GenerationError(ncgearsError):
    """Raised when the requested design cannot be generated."""
