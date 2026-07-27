"""Optional Matplotlib rendering for generated gear pairs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .result import GearPair


def _load_motion(pair: GearPair) -> tuple[float, float, Callable[[float], float]]:
    """Load the generated motion law and return its active animation interval."""

    from scipy.interpolate import CubicHermiteSpline, CubicSpline

    metadata = pair.metadata
    closed = metadata.get("topology", "closed") == "closed"
    if not closed and (
        "active_start" not in metadata or "active_end" not in metadata
    ):
        raise ValueError(
            "Open gear-pair metadata does not record its active interval; "
            "regenerate the pair before rendering a GIF."
        )
    transmission_path = pair.directory / "transmission.csv"
    centrode_path = pair.directory / "centrode.csv"

    if transmission_path.exists():
        samples = np.loadtxt(
            transmission_path, delimiter=",", skiprows=1, ndmin=2
        )
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
            if phi[-1] < endpoint - 1e-10 * max(1.0, abs(period)):
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
            if phi[-1] < endpoint - 1e-10 * max(1.0, abs(period)):
                phi = np.append(phi, endpoint)
                radius = np.append(radius, radius[0])

        reference_distance = float(
            metadata["centrode_reference_center_distance"]
        )
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
        "GIF rendering requires transmission.csv or centrode.csv in the "
        f"gear-pair directory: {pair.directory}"
    )


def render_pair(pair: GearPair, output: str | Path) -> Path:
    """Render an assembled gear pair to a PNG or other Matplotlib format."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
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


def render_pair_gif(
    pair: GearPair,
    output: str | Path,
    *,
    frames: int = 72,
    fps: int = 24,
    dpi: int = 100,
    show_axes: bool = True,
    show_title: bool = True,
) -> Path:
    """Render the generated gear pair moving through its active interval."""

    if frames < 2:
        raise ValueError("frames must be at least 2")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Polygon
        from matplotlib.transforms import Affine2D
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
    drive_radius = float(np.max(np.linalg.norm(drive, axis=1)))
    driven_radius = float(np.max(np.linalg.norm(driven, axis=1)))
    margin = 0.08 * max(
        pair.center_distance + drive_radius + driven_radius,
        1.0,
    )

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
            f"{pair.metadata['name']}: "
            f"{pair.drive_teeth}:{pair.driven_teeth} teeth"
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
            Affine2D()
            .rotate(driven_angle)
            .translate(pair.center_distance, 0.0)
            + axis.transData
        )
        return drive_patch, driven_patch

    animation = FuncAnimation(
        figure,
        update,
        frames=phases,
        interval=1000.0 / fps,
        blit=True,
    )
    try:
        animation.save(destination, writer=PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(figure)
    return destination.resolve()
