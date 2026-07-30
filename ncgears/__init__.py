"""Python-only conjugate noncircular gear generation."""

from .api import generate, generate_from_centrode, generate_from_transmission
from .errors import GenerationError, GeneratorNotFoundError, ncgearsError
from .native import native_generator
from .result import GearPair

__all__ = [
    "GearPair",
    "GenerationError",
    "GeneratorNotFoundError",
    "generate",
    "generate_from_centrode",
    "generate_from_transmission",
    "native_generator",
    "ncgearsError",
]
__version__ = "0.2.1"
