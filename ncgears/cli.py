"""Small command-line companion to the Python API."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .api import generate_from_centrode, generate_from_transmission
from .errors import ncgearsError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ncgears",
        description="Generate a verified conjugate noncircular gear pair.",
    )
    parser.add_argument(
        "expression",
        help="SymPy expression in phi: a motion law by default, or r(phi) with --centrode",
    )
    parser.add_argument(
        "--centrode",
        action="store_true",
        help="interpret the expression as the drive pitch radius r(phi)",
    )
    parser.add_argument("--name", default="gear_pair")
    parser.add_argument("--teeth", type=int, default=16)
    parser.add_argument("--module", type=float, default=1.0)
    parser.add_argument("--pressure-angle", type=float, default=20.0)
    parser.add_argument("--open", action="store_true", dest="open_")
    parser.add_argument("--drive-start", type=float, default=0.0)
    parser.add_argument("--drive-end", type=float, default=2.0 * math.pi)
    parser.add_argument("--period", type=float, default=2.0 * math.pi)
    parser.add_argument(
        "--profile",
        choices=("involute", "cycloidal"),
        default="involute",
    )
    parser.add_argument(
        "--center-distance",
        type=float,
        help="centrode reference distance, in the expression's radius units",
    )
    parser.add_argument(
        "--cycle-delta",
        type=float,
        default=2.0 * math.pi,
        help="target mate advance for a centrode with solved center distance",
    )
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--samples-per-radian", type=int, default=110)
    parser.add_argument("--output", type=Path, default=Path("out"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="open an interactive Matplotlib plot with a motion slider",
    )
    parser.add_argument(
        "--gif",
        nargs="?",
        const=True,
        type=Path,
        metavar="PATH",
        help="render an animated GIF, optionally to PATH",
    )
    parser.add_argument("--gif-frames", type=int, default=72)
    parser.add_argument("--gif-fps", type=int, default=24)
    parser.add_argument("--svg", type=Path, help="also export an assembled SVG")
    parser.add_argument("--dxf", type=Path, help="also export an assembled DXF")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    common = {
        "name": args.name,
        "teeth": args.teeth,
        "module": args.module,
        "pressure_angle_deg": args.pressure_angle,
        "drive_start": args.drive_start,
        "drive_end": args.drive_end,
        "period": args.period,
        "open_": args.open_,
        "samples": args.samples,
        "profile": args.profile,
        "samples_per_radian": args.samples_per_radian,
        "output_directory": args.output,
        "render": args.render,
        "plot": args.plot,
    }
    try:
        if args.centrode:
            pair = generate_from_centrode(
                args.expression,
                reference_center_distance=args.center_distance,
                target_cycle_delta=args.cycle_delta,
                **common,
            )
        else:
            if args.center_distance is not None:
                raise ValueError("--center-distance requires --centrode")
            pair = generate_from_transmission(args.expression, **common)
        if args.svg:
            pair.export_svg(args.svg)
        if args.dxf:
            pair.export_dxf(args.dxf)
        if args.gif:
            pair.render_gif(
                None if args.gif is True else args.gif,
                frames=args.gif_frames,
                fps=args.gif_fps,
            )
    except (ncgearsError, ValueError, OSError) as error:
        parser.exit(2, f"ncgears: error: {error}\n")

    print(pair.summary())
    print(f"Files: {pair.directory}")
    return 0
