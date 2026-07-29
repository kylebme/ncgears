#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncgears import generate_from_centrode
from ncgears._policy import (
    DEFAULT_ADDENDUM_FACTOR,
    DEFAULT_CYCLOIDAL_ROLLING_FACTOR,
    DEFAULT_DEDENDUM_FACTOR,
    DEFAULT_FILLET_FACTOR,
    DEFAULT_INPUT_SAMPLES,
    DEFAULT_MODULE,
    DEFAULT_PADDING_PITCHES,
    DEFAULT_PRESSURE_ANGLE_DEG,
    DEFAULT_SAMPLES_PER_RADIAN,
    DEFAULT_TEETH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate noncircular gears from a polar drive-centrode expression."
    )
    parser.add_argument(
        "expression", help="Centrode radius r(phi) as a SymPy expression"
    )
    parser.add_argument("--name", default="centrode")
    parser.add_argument("--description")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--teeth", type=int, default=DEFAULT_TEETH)
    parser.add_argument("--module", type=float, default=DEFAULT_MODULE)
    parser.add_argument(
        "--pressure-angle-deg",
        type=float,
        default=DEFAULT_PRESSURE_ANGLE_DEG,
    )
    parser.add_argument(
        "--addendum-factor", type=float, default=DEFAULT_ADDENDUM_FACTOR
    )
    parser.add_argument(
        "--dedendum-factor", type=float, default=DEFAULT_DEDENDUM_FACTOR
    )
    parser.add_argument("--fillet-factor", type=float, default=DEFAULT_FILLET_FACTOR)
    parser.add_argument("--drive-start", type=float, default=0.0)
    parser.add_argument("--drive-end", type=float, default=2.0 * math.pi)
    parser.add_argument("--period", type=float, default=2.0 * math.pi)
    parser.add_argument("--samples", type=int, default=DEFAULT_INPUT_SAMPLES)
    parser.add_argument(
        "--padding-pitches", type=float, default=DEFAULT_PADDING_PITCHES
    )
    parser.add_argument(
        "--center-distance",
        type=float,
        help="Reference distance in the same arbitrary units as r(phi)",
    )
    parser.add_argument(
        "--cycle-delta",
        type=float,
        default=2.0 * math.pi,
        help="Net mate advance used when solving an omitted center distance",
    )
    parser.add_argument(
        "--profile", choices=("involute", "cycloidal"), default="involute"
    )
    parser.add_argument(
        "--cycloidal-rolling-factor",
        type=float,
        default=DEFAULT_CYCLOIDAL_ROLLING_FACTOR,
    )
    parser.add_argument(
        "--samples-per-radian",
        type=int,
        default=DEFAULT_SAMPLES_PER_RADIAN,
    )
    parser.add_argument("--out", type=Path, default=ROOT / "out")
    parser.add_argument("--no-render", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate_from_centrode(
        args.expression,
        name=args.name,
        description=args.description,
        teeth=args.teeth,
        module=args.module,
        pressure_angle_deg=args.pressure_angle_deg,
        addendum_factor=args.addendum_factor,
        dedendum_factor=args.dedendum_factor,
        fillet_factor=args.fillet_factor,
        drive_start=args.drive_start,
        drive_end=args.drive_end,
        period=args.period,
        open_=args.open,
        samples=args.samples,
        padding_pitches=args.padding_pitches,
        reference_center_distance=args.center_distance,
        target_cycle_delta=args.cycle_delta,
        profile=args.profile,
        cycloidal_rolling_factor=args.cycloidal_rolling_factor,
        samples_per_radian=args.samples_per_radian,
        output_directory=args.out,
        render=not args.no_render,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
