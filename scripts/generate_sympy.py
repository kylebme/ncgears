#!/usr/bin/env python3
"""Compatibility wrapper for the historical transmission script.

New code should use ``ncgears.generate`` or the installed ``ncgears`` command.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncgears import generate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate noncircular gears from a transmission expression."
    )
    parser.add_argument("expression")
    parser.add_argument("--name", default="expression")
    parser.add_argument("--open", action="store_true", dest="open_")
    parser.add_argument("--teeth", type=int, default=16)
    parser.add_argument("--module", type=float, default=1.0)
    parser.add_argument("--pressure-angle-deg", type=float, default=20.0)
    parser.add_argument("--addendum-factor", type=float, default=1.0)
    parser.add_argument("--dedendum-factor", type=float, default=1.2)
    parser.add_argument("--fillet-factor", type=float, default=0.3)
    parser.add_argument("--drive-start", type=float, default=0.0)
    parser.add_argument("--drive-end", type=float, default=2.0 * math.pi)
    parser.add_argument("--period", type=float, default=2.0 * math.pi)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--padding-pitches", type=float, default=7.0)
    parser.add_argument("--allow-nonconvex", action="store_true")
    parser.add_argument(
        "--profile", choices=("involute", "cycloidal"), default="involute"
    )
    parser.add_argument("--cycloidal-rolling-factor", type=float, default=0.35)
    parser.add_argument("--samples-per-radian", type=int, default=110)
    parser.add_argument("--out", type=Path, default=ROOT / "out")
    parser.add_argument("--no-render", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pair = generate(
        args.expression,
        name=args.name,
        teeth=args.teeth,
        module=args.module,
        pressure_angle_deg=args.pressure_angle_deg,
        addendum_factor=args.addendum_factor,
        dedendum_factor=args.dedendum_factor,
        fillet_factor=args.fillet_factor,
        drive_start=args.drive_start,
        drive_end=args.drive_end,
        period=args.period,
        open_=args.open_,
        samples=args.samples,
        padding_pitches=args.padding_pitches,
        profile=args.profile,
        cycloidal_rolling_factor=args.cycloidal_rolling_factor,
        samples_per_radian=args.samples_per_radian,
        output_directory=args.out,
        render=not args.no_render,
    )
    print(pair.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
