"""Structured results and CAD-friendly exports."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import ezdxf
import numpy as np
from ezdxf import colors, units
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.figure import Figure

GearSelection = Literal["drive", "driven", "pair"]


def _load_outline(path: Path) -> NDArray[np.float64]:
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected a two-column gear outline in {path}")
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class GearPair:
    """A generated pair of gear outlines and its verification report."""

    directory: Path
    drive_outline: NDArray[np.float64]
    driven_outline: NDArray[np.float64]
    metadata: dict[str, Any]
    generator_log: str = ""

    @classmethod
    def load(cls, directory: str | Path, *, generator_log: str = "") -> GearPair:
        """Load a previously generated pair from a result directory."""

        result_directory = Path(directory).expanduser().resolve()
        metadata_path = result_directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(
            directory=result_directory,
            drive_outline=_load_outline(result_directory / "drive.csv"),
            driven_outline=_load_outline(result_directory / "driven.csv"),
            metadata=metadata,
            generator_log=generator_log,
        )

    def __fspath__(self) -> str:
        """Allow use anywhere an ``os.PathLike`` result directory is accepted."""

        return os.fspath(self.directory)

    def __truediv__(self, child: str | os.PathLike[str]) -> Path:
        """Preserve the convenient ``result / "drive.csv"`` idiom."""

        return self.directory / child

    @property
    def center_distance(self) -> float:
        return float(self.metadata["center_distance"])

    @property
    def drive_teeth(self) -> int:
        return int(self.metadata["drive_teeth"])

    @property
    def driven_teeth(self) -> int:
        return int(self.metadata["driven_teeth"])

    @property
    def ratio(self) -> float:
        """Average driven angular advance per unit of drive rotation."""

        return float(self.metadata["average_angular_ratio"])

    @property
    def maximum_transmission_error(self) -> float:
        return float(self.metadata["maximum_transmission_error"])

    @property
    def placed_driven_outline(self) -> NDArray[np.float64]:
        """Driven outline translated into its assembled position."""

        points = self.driven_outline.copy()
        points[:, 0] += self.center_distance
        return points

    def render(self, output: str | Path | None = None) -> Path:
        """Render an assembled PNG preview.

        Install the optional plotting dependency with ``pip install
        ncgears[plot]``.
        """

        from .rendering import render_pair

        destination = (
            Path(output) if output is not None else self.directory / "pair.png"
        )
        render_pair(self, destination)
        return destination.resolve()

    def plot(self, *, show: bool = True) -> Figure:
        """Create an interactive Matplotlib plot with a motion slider.

        The usual Matplotlib toolbar supports zooming and panning. Set
        ``show=False`` to customize or embed the returned Figure without
        immediately opening a window.

        Install the optional plotting dependency with ``pip install
        ncgears[plot]``.
        """

        from .rendering import plot_pair

        return plot_pair(self, show=show)

    def render_gif(
        self,
        output: str | Path | None = None,
        *,
        frames: int = 72,
        fps: int = 24,
        dpi: int = 100,
        show_axes: bool = True,
        show_title: bool = True,
    ) -> Path:
        """Render an animated GIF using the generated pair's motion law.

        Install the optional plotting dependency with ``pip install
        ncgears[plot]``.
        """

        from .rendering import render_pair_gif

        destination = (
            Path(output) if output is not None else self.directory / "pair.gif"
        )
        return render_pair_gif(
            self,
            destination,
            frames=frames,
            fps=fps,
            dpi=dpi,
            show_axes=show_axes,
            show_title=show_title,
        )

    def export_svg(
        self,
        output: str | Path,
        *,
        gear: GearSelection = "pair",
        stroke_width: float | None = None,
    ) -> Path:
        """Export one outline or the assembled pair as a dimensioned SVG."""

        outlines = self._selected_outlines(gear)
        all_points = np.vstack(tuple(outlines.values()))
        minimum = np.min(all_points, axis=0)
        maximum = np.max(all_points, axis=0)
        span = np.maximum(maximum - minimum, 1e-9)
        margin = 0.04 * float(max(span))
        width = float(span[0] + 2.0 * margin)
        height = float(span[1] + 2.0 * margin)
        line_width = stroke_width or 0.002 * float(max(span))
        colors = {"drive": "#245b82", "driven": "#a85e21"}

        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        elements: list[str] = []
        for label, points in outlines.items():
            coordinates = " ".join(f"{x:.12g},{-y:.12g}" for x, y in points)
            elements.append(
                f'  <polygon id="{label}" points="{coordinates}" '
                f'fill="none" stroke="{colors[label]}" '
                f'stroke-width="{line_width:.12g}" vector-effect="non-scaling-stroke"/>'
            )
        view_y = -float(maximum[1]) - margin
        document = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{minimum[0] - margin:.12g} {view_y:.12g} '
            f'{width:.12g} {height:.12g}" '
            'data-units="millimetres">\n' + "\n".join(elements) + "\n</svg>\n"
        )
        path.write_text(document, encoding="utf-8")
        return path.resolve()

    def export_dxf(
        self,
        output: str | Path,
        *,
        gear: GearSelection = "pair",
    ) -> Path:
        """Export one outline or the assembled pair as an ASCII DXF.

        The document uses millimetres and one closed lightweight polyline per
        gear. Pair exports place each outline on its own named layer.
        """

        outlines = self._selected_outlines(gear)
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        document = ezdxf.new("R2010")
        document.units = units.MM
        document.header["$MEASUREMENT"] = 1
        document.header["$PDMODE"] = 3  # display POINT entities as crosses
        modelspace = document.modelspace()
        layer_colors = {
            "drive": colors.BLUE,
            "driven": colors.RED,
        }

        for label, points in outlines.items():
            vertices = points
            if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
                vertices = vertices[:-1]
            layer = label.upper()
            document.layers.add(layer, color=layer_colors[label])
            modelspace.add_lwpolyline(
                vertices,
                format="xy",
                close=True,
                dxfattribs={"layer": layer},
            )
            center = (
                (self.center_distance, 0.0)
                if gear == "pair" and label == "driven"
                else (0.0, 0.0)
            )
            modelspace.add_point(center, dxfattribs={"layer": layer})

        document.saveas(path)
        return path.resolve()

    def _selected_outlines(self, gear: GearSelection) -> dict[str, NDArray[np.float64]]:
        if gear == "drive":
            return {"drive": self.drive_outline}
        if gear == "driven":
            return {"driven": self.driven_outline}
        if gear == "pair":
            return {
                "drive": self.drive_outline,
                "driven": self.placed_driven_outline,
            }
        raise ValueError("gear must be 'drive', 'driven', or 'pair'")

    def summary(self) -> str:
        """Return a compact human-readable verification summary."""

        return (
            f"{self.metadata['name']}: {self.drive_teeth}:{self.driven_teeth} teeth, "
            f"center distance {self.center_distance:.6g}, "
            f"ratio {self.ratio:.6g}, maximum transmission error "
            f"{math.degrees(self.maximum_transmission_error):.4g}°"
        )
