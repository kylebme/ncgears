"""Locate the native generator shipped in an ncgear wheel."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .errors import GeneratorNotFoundError


def native_generator(override: str | Path | None = None) -> Path:
    """Return the native generator executable.

    ``override`` is primarily useful to package maintainers. Users can also set
    ``NCGEAR_GENERATOR`` when testing a locally built generator.
    """

    executable = "ncgear_generate.exe" if os.name == "nt" else "ncgear_generate"
    candidates: list[Path] = []
    if override is not None:
        candidates.append(Path(override).expanduser())
    environment_override = os.environ.get("NCGEAR_GENERATOR")
    if environment_override:
        candidates.append(Path(environment_override).expanduser())

    package_directory = Path(__file__).resolve().parent
    candidates.extend(
        [
            package_directory / "bin" / executable,
            package_directory.parent / "build" / executable,
        ]
    )
    on_path = shutil.which("ncgear_generate")
    if on_path:
        candidates.append(Path(on_path))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise GeneratorNotFoundError(
        "Could not find the ncgear native generator. Install a platform wheel "
        "or build the project with CMake. Searched:\n"
        f"{searched}"
    )
