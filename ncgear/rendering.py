"""Optional Matplotlib rendering for generated gear pairs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .result import GearPair


def render_pair(pair: GearPair, output: str | Path) -> Path:
    """Render an assembled gear pair to a PNG or other Matplotlib format."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Rendering requires Matplotlib. Install it with "
            "`python -m pip install 'ncgear[plot]'`."
        ) from error

    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    drive = pair.drive_outline
    driven = pair.placed_driven_outline

    figure, axis = plt.subplots(figsize=(10, 6), dpi=180)
    axis.fill(drive[:, 0], drive[:, 1], color="#8eb6d8", alpha=0.82)
    axis.plot(drive[:, 0], drive[:, 1], color="#15324b", linewidth=0.7)
    axis.fill(driven[:, 0], driven[:, 1], color="#e4b07a", alpha=0.82)
    axis.plot(driven[:, 0], driven[:, 1], color="#6b3718", linewidth=0.7)
    axis.plot(
        [0.0, pair.center_distance],
        [0.0, 0.0],
        linestyle=":",
        marker="+",
        color="#5f6770",
        linewidth=0.8,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="#d7dce2", linewidth=0.5)
    axis.set_facecolor("#fafbfc")
    axis.set_title(pair.summary())
    axis.set_xlabel("millimetres")
    axis.set_ylabel("millimetres")
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination.resolve()
