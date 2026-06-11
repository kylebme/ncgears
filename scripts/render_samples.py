#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_outline(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def style_axis(axis: plt.Axes) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, color="#d7dce2", linewidth=0.5)
    axis.set_facecolor("#fafbfc")
    axis.tick_params(labelsize=8)


def save_single(points: np.ndarray, title: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 7), dpi=180)
    axis.fill(points[:, 0], points[:, 1], color="#c9d8e8", alpha=0.8)
    axis.plot(points[:, 0], points[:, 1], color="#15324b", linewidth=0.8)
    axis.plot(0.0, 0.0, marker="+", color="#b33a3a", markersize=8)
    axis.set_title(title)
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def render_sample(directory: Path) -> None:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    drive = load_outline(directory / "drive.csv")
    driven = load_outline(directory / "driven.csv")
    center_distance = float(metadata["center_distance"])
    driven_world = driven.copy()
    driven_world[:, 0] += center_distance

    save_single(drive, f"{metadata['name']} - drive gear", directory / "drive.png")
    save_single(driven, f"{metadata['name']} - driven gear", directory / "driven.png")

    figure, axis = plt.subplots(figsize=(11, 6), dpi=180)
    axis.fill(drive[:, 0], drive[:, 1], color="#8eb6d8", alpha=0.78)
    axis.plot(drive[:, 0], drive[:, 1], color="#15324b", linewidth=0.75)
    axis.fill(driven_world[:, 0], driven_world[:, 1], color="#e4b07a", alpha=0.78)
    axis.plot(
        driven_world[:, 0],
        driven_world[:, 1],
        color="#6b3718",
        linewidth=0.75,
    )
    axis.plot(
        [0.0, center_distance],
        [0.0, 0.0],
        linestyle=":",
        color="#5f6770",
        linewidth=0.8,
    )
    axis.plot(
        [0.0, center_distance],
        [0.0, 0.0],
        linestyle="none",
        marker="+",
        color="#b33a3a",
        markersize=8,
    )
    axis.set_title(
        f"{metadata['name']} - generated gear pair\n{metadata['description']}"
    )
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(directory / "pair.png")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("out"))
    parser.add_argument("--sample")
    args = parser.parse_args()

    sample_directories = sorted(
        path
        for path in args.input.iterdir()
        if (path / "metadata.json").exists()
        and (args.sample is None or path.name == args.sample)
    )
    if not sample_directories:
        raise SystemExit(f"No generated samples found under {args.input}")
    for directory in sample_directories:
        render_sample(directory)
        print(f"Rendered {directory / 'pair.png'}")


if __name__ == "__main__":
    main()
