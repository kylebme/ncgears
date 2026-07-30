#!/usr/bin/env python3
"""Render magnified root regions from a generated ncgears pair."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncgears import GearPair


def _representative_roots(
    outline: np.ndarray, teeth: int, module: float
) -> list[int]:
    points = outline[:-1]
    radii = np.linalg.norm(points, axis=1)
    minimum_distance = max(3, len(points) // max(1, 2 * teeth))
    candidates, _ = find_peaks(
        -radii,
        distance=minimum_distance,
        prominence=0.08 * module,
    )
    if len(candidates) < 4:
        candidates = np.argsort(radii)[: max(4, min(len(points), teeth))]
    angles = np.arctan2(points[candidates, 1], points[candidates, 0])
    selected: list[int] = []
    for target in (0.0, 0.5 * math.pi, math.pi, -0.5 * math.pi):
        distance = np.abs(np.angle(np.exp(1j * (angles - target))))
        for index in candidates[np.argsort(distance)]:
            candidate = int(index)
            if candidate not in selected:
                selected.append(candidate)
                break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render four representative root zooms for each gear."
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window-modules", type=float, default=1.8)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair = GearPair.load(args.directory)
    module = float(pair.metadata["module"])
    destination = args.output or pair.directory / "root_zooms.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), dpi=220)
    for row, (label, outline, teeth, color) in enumerate(
        (
            ("drive", pair.drive_outline, pair.drive_teeth, "#2d6f9f"),
            ("driven", pair.driven_outline, pair.driven_teeth, "#a85e23"),
        )
    ):
        points = outline[:-1]
        for column, root_index in enumerate(
            _representative_roots(outline, teeth, module)
        ):
            axis = axes[row, column]
            root = points[root_index]
            half_width = args.window_modules * module
            axis.plot(
                outline[:, 0],
                outline[:, 1],
                color=color,
                linewidth=1.1,
            )
            axis.scatter(
                [root[0]],
                [root[1]],
                color="#c51b3a",
                marker="+",
                s=30,
                zorder=3,
            )
            axis.set_xlim(root[0] - half_width, root[0] + half_width)
            axis.set_ylim(root[1] - half_width, root[1] + half_width)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, linewidth=0.35, color="#d5dbe2")
            angle = math.degrees(math.atan2(root[1], root[0]))
            axis.set_title(f"{label} root near {angle:.0f}°")
    figure.suptitle(
        f"{pair.metadata['name']}: analytical involute root inspection"
    )
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    print(destination.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
