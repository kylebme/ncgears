"""Compatibility shim for releases that used a native generator."""

from __future__ import annotations

from pathlib import Path

from .errors import GeneratorNotFoundError


def native_generator(override: str | Path | None = None) -> Path:
    """Raise a clear migration error.

    Version 0.2 and later run the Shapely geometry engine in the Python process.
    The function remains importable so older callers receive an actionable
    message instead of an ``ImportError``.
    """

    del override
    raise GeneratorNotFoundError(
        "ncgear no longer ships a native generator; call ncgear.generate() "
        "to use the in-process Python engine"
    )
