"""Optional Matplotlib rendering for generated gear pairs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._policy import (
    DEFAULT_GIF_DPI,
    DEFAULT_GIF_FPS,
    DEFAULT_GIF_FRAMES,
    FLOAT_COMPARISON_ULPS,
    MIN_GIF_FRAMES,
    RENDER_MARGIN_FRACTION,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .result import GearPair


def _load_motion(pair: GearPair) -> tuple[float, float, Callable[[float], float]]:
    """Load the generated motion law and return its active animation interval."""

    from scipy.interpolate import CubicHermiteSpline, CubicSpline

    metadata = pair.metadata
    closed = metadata.get("topology", "closed") == "closed"
    if not closed and ("active_start" not in metadata or "active_end" not in metadata):
        raise ValueError(
            "Open gear-pair metadata does not record its active interval; "
            "regenerate the pair before plotting its motion."
        )
    transmission_path = pair.directory / "transmission.csv"
    centrode_path = pair.directory / "centrode.csv"

    if transmission_path.exists():
        samples = np.loadtxt(transmission_path, delimiter=",", skiprows=1, ndmin=2)
        phi = samples[:, 0]
        psi = samples[:, 1]
        derivative = samples[:, 2]
        inferred_period = float(np.median(np.diff(phi))) * len(phi)
        period = float(metadata.get("period", inferred_period))
        domain_start = float(metadata.get("domain_start", phi[0]))
        active_start = float(metadata.get("active_start", domain_start))
        active_end = float(
            metadata.get(
                "active_end",
                active_start + period if closed else phi[-1],
            )
        )
        if closed:
            cycle_delta = float(
                metadata.get(
                    "cycle_delta",
                    float(metadata["average_angular_ratio"]) * period,
                )
            )
            endpoint = domain_start + period
            tolerance = (
                FLOAT_COMPARISON_ULPS * np.finfo(float).eps * max(1.0, abs(period))
            )
            if phi[-1] < endpoint - tolerance:
                phi = np.append(phi, endpoint)
                psi = np.append(psi, psi[0] + cycle_delta)
                derivative = np.append(derivative, derivative[0])
        spline = CubicHermiteSpline(phi, psi, derivative)
        psi_start = float(spline(active_start))
        return active_start, active_end, lambda angle: float(spline(angle)) - psi_start

    if centrode_path.exists():
        samples = np.loadtxt(centrode_path, delimiter=",", skiprows=1, ndmin=2)
        phi = samples[:, 0]
        radius = samples[:, 1]
        inferred_period = float(np.median(np.diff(phi))) * len(phi)
        period = float(metadata.get("period", inferred_period))
        domain_start = float(metadata.get("domain_start", phi[0]))
        active_start = float(metadata.get("active_start", domain_start))
        active_end = float(
            metadata.get(
                "active_end",
                active_start + period if closed else phi[-1],
            )
        )
        if closed:
            endpoint = domain_start + period
            tolerance = (
                FLOAT_COMPARISON_ULPS * np.finfo(float).eps * max(1.0, abs(period))
            )
            if phi[-1] < endpoint - tolerance:
                phi = np.append(phi, endpoint)
                radius = np.append(radius, radius[0])

        reference_distance = float(metadata["centrode_reference_center_distance"])
        ratio = radius / (reference_distance - radius)
        ratio_spline = CubicSpline(
            phi,
            ratio,
            bc_type="periodic" if closed else "not-a-knot",
        )
        integral = ratio_spline.antiderivative()
        psi_start = float(integral(active_start))
        return (
            active_start,
            active_end,
            lambda angle: float(integral(angle)) - psi_start,
        )

    raise ValueError(
        "Motion plotting requires transmission.csv or centrode.csv in the "
        f"gear-pair directory: {pair.directory}"
    )


def _plot_extents(pair: GearPair) -> tuple[float, float, float]:
    """Return drive radius, driven radius, and a shared plotting margin."""

    drive_radius = float(np.max(np.linalg.norm(pair.drive_outline, axis=1)))
    driven_radius = float(np.max(np.linalg.norm(pair.driven_outline, axis=1)))
    margin = RENDER_MARGIN_FRACTION * max(
        pair.center_distance + drive_radius + driven_radius,
        1.0,
    )
    return drive_radius, driven_radius, margin


def plot_pair(
    pair: GearPair,
    *,
    show: bool = True,
) -> Figure:
    """Create an interactive gear-pair plot with a drive-angle slider.

    The standard Matplotlib navigation toolbar remains available for zooming
    and panning. Set ``show=False`` to embed or further customize the returned
    figure before displaying it.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.transforms import Affine2D
        from matplotlib.widgets import Slider
    except ImportError as error:
        raise ImportError(
            "Interactive plotting requires Matplotlib. Install it with "
            "`python -m pip install 'ncgears[plot]'`."
        ) from error

    active_start, active_end, driven_motion = _load_motion(pair)
    drive_radius, driven_radius, margin = _plot_extents(pair)

    figure, axis = plt.subplots(figsize=(10, 6))
    figure.subplots_adjust(bottom=0.18)
    drive_patch = Polygon(
        pair.drive_outline,
        closed=True,
        facecolor="#8eb6d8",
        edgecolor="#15324b",
        linewidth=0.8,
        alpha=0.88,
    )
    driven_patch = Polygon(
        pair.driven_outline,
        closed=True,
        facecolor="#e4b07a",
        edgecolor="#6b3718",
        linewidth=0.8,
        alpha=0.88,
    )
    axis.add_patch(drive_patch)
    axis.add_patch(driven_patch)
    axis.plot(
        [0.0, pair.center_distance],
        [0.0, 0.0],
        linestyle=":",
        marker="+",
        color="#5f6770",
        linewidth=0.8,
    )
    axis.set_xlim(
        -drive_radius - margin,
        pair.center_distance + driven_radius + margin,
    )
    vertical_radius = max(drive_radius, driven_radius)
    axis.set_ylim(-vertical_radius - margin, vertical_radius + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="#d7dce2", linewidth=0.5)
    axis.set_facecolor("#fafbfc")
    axis.set_title(
        f"{pair.metadata['name']}: "
        f"{pair.drive_teeth}:{pair.driven_teeth} teeth"
    )
    axis.set_xlabel("millimetres")
    axis.set_ylabel("millimetres")

    slider_axis = figure.add_axes((0.16, 0.055, 0.68, 0.035))
    angle_slider = Slider(
        ax=slider_axis,
        label="Drive angle φ (rad)",
        valmin=active_start,
        valmax=active_end,
        valinit=active_start,
    )

    def update(phi: float) -> None:
        drive_angle = float(phi - active_start)
        driven_angle = -driven_motion(float(phi))
        drive_patch.set_transform(
            Affine2D().rotate(drive_angle) + axis.transData
        )
        driven_patch.set_transform(
            Affine2D()
            .rotate(driven_angle)
            .translate(pair.center_distance, 0.0)
            + axis.transData
        )
        figure.canvas.draw_idle()

    angle_slider.on_changed(update)
    update(active_start)

    # Matplotlib's callback registry holds weak references. Retaining the
    # Slider on the Figure keeps it responsive when callers only retain the
    # returned Figure.
    figure._ncgears_angle_slider = angle_slider
    if show:
        plt.show()
    return figure


def render_pair(pair: GearPair, output: str | Path) -> Path:
    """Render an assembled gear pair to a PNG or other Matplotlib format."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError as error:
        raise ImportError(
            "Rendering requires Matplotlib. Install it with "
            "`python -m pip install 'ncgears[plot]'`."
        ) from error

    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    drive = pair.drive_outline
    driven = pair.placed_driven_outline

    figure, axis = plt.subplots(figsize=(10, 6), dpi=180)
    drive_patch = Polygon(
        drive,
        closed=True,
        facecolor="#8eb6d8",
        edgecolor="#15324b",
        linewidth=0.7,
        alpha=0.82,
    )
    driven_patch = Polygon(
        driven,
        closed=True,
        facecolor="#e4b07a",
        edgecolor="#6b3718",
        linewidth=0.7,
        alpha=0.82,
    )
    # Limits are known from the input arrays. add_artist avoids Matplotlib
    # traversing every outline vertex merely to rediscover those bounds.
    axis.add_artist(drive_patch)
    axis.add_artist(driven_patch)
    all_points = np.vstack((drive, driven))
    minimum = np.min(all_points, axis=0)
    maximum = np.max(all_points, axis=0)
    span = maximum - minimum
    margin = 0.05 * np.maximum(span, np.finfo(float).eps)
    axis.set_xlim(minimum[0] - margin[0], maximum[0] + margin[0])
    axis.set_ylim(minimum[1] - margin[1], maximum[1] + margin[1])
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


def render_pair_gif(
    pair: GearPair,
    output: str | Path,
    *,
    frames: int = DEFAULT_GIF_FRAMES,
    fps: int = DEFAULT_GIF_FPS,
    dpi: int = DEFAULT_GIF_DPI,
    show_axes: bool = True,
    show_title: bool = True,
) -> Path:
    """Render the generated gear pair moving through its active interval."""

    if frames < MIN_GIF_FRAMES:
        raise ValueError(f"frames must be at least {MIN_GIF_FRAMES}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from matplotlib.transforms import Affine2D
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            "GIF rendering requires Matplotlib and Pillow. Install them with "
            "`python -m pip install 'ncgears[plot]'`."
        ) from error

    destination = Path(output).expanduser()
    if destination.suffix.lower() != ".gif":
        raise ValueError("GIF output path must end in .gif")
    destination.parent.mkdir(parents=True, exist_ok=True)

    active_start, active_end, driven_motion = _load_motion(pair)
    closed = pair.metadata.get("topology", "closed") == "closed"
    phases = np.linspace(
        active_start,
        active_end,
        frames,
        endpoint=not closed,
    )

    drive = pair.drive_outline
    driven = pair.driven_outline
    drive_radius, driven_radius, margin = _plot_extents(pair)

    figure, axis = plt.subplots(figsize=(9, 5.4), dpi=dpi)
    drive_patch = Polygon(
        drive,
        closed=True,
        facecolor="#8eb6d8",
        edgecolor="#15324b",
        linewidth=0.8,
        alpha=0.88,
    )
    driven_patch = Polygon(
        driven,
        closed=True,
        facecolor="#e4b07a",
        edgecolor="#6b3718",
        linewidth=0.8,
        alpha=0.88,
    )
    # The limits are set explicitly below, so avoid the O(vertices) automatic
    # data-limit scan performed by add_patch.
    axis.add_artist(drive_patch)
    axis.add_artist(driven_patch)
    axis.plot(
        [0.0, pair.center_distance],
        [0.0, 0.0],
        linestyle=":",
        marker="+",
        color="#5f6770",
        linewidth=0.8,
    )
    axis.set_xlim(-drive_radius - margin, pair.center_distance + driven_radius + margin)
    vertical_radius = max(drive_radius, driven_radius)
    axis.set_ylim(-vertical_radius - margin, vertical_radius + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_facecolor("#fafbfc")
    if show_axes:
        axis.grid(True, color="#d7dce2", linewidth=0.5)
        axis.set_xlabel("millimetres")
        axis.set_ylabel("millimetres")
    else:
        axis.set_axis_off()
    if show_title:
        axis.set_title(
            f"{pair.metadata['name']}: {pair.drive_teeth}:{pair.driven_teeth} teeth"
        )
    if show_axes or show_title:
        figure.tight_layout()
    else:
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    def update(phi: float) -> tuple[Polygon, Polygon]:
        drive_angle = float(phi - active_start)
        driven_angle = -driven_motion(float(phi))
        drive_patch.set_transform(Affine2D().rotate(drive_angle) + axis.transData)
        driven_patch.set_transform(
            Affine2D().rotate(driven_angle).translate(pair.center_distance, 0.0)
            + axis.transData
        )
        return drive_patch, driven_patch

    try:
        rendered_frames: list[Image.Image] = []
        for phase in phases:
            update(float(phase))
            figure.canvas.draw()
            size = figure.canvas.get_width_height()
            frame = Image.frombuffer(
                "RGBA",
                size,
                figure.canvas.buffer_rgba(),
                "raw",
                "RGBA",
                0,
                1,
            )
            # The canvas is reused for the next pose. Conversion both improves
            # GIF palette selection and gives this frame independent storage.
            rendered_frames.append(frame.convert("RGB"))
        rendered_frames[0].save(
            destination,
            save_all=True,
            append_images=rendered_frames[1:],
            duration=int(1000 / fps),
            loop=0,
        )
    finally:
        plt.close(figure)
    return destination.resolve()
