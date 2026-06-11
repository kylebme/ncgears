#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SAMPLES = [
    {
        "name": "closed_one_to_two_wave",
        "expression": "0.5*phi - 0.04*sin(2*phi)",
        "teeth": 30,
    },
    {
        "name": "closed_three_to_two_tri_lobe",
        "expression": "1.5*phi - 0.08*sin(3*phi)",
        "teeth": 24,
    },
    {
        "name": "closed_five_to_three_five_lobe",
        "expression": "(5/3)*phi - 0.03*sin(5*phi)",
        "teeth": 30,
    },
    {
        "name": "closed_two_to_one_double_lobe",
        "expression": "2*phi - 0.04*sin(2*phi)",
        "teeth": 30,
    },
    {
        "name": "open_accelerating_exponential",
        "expression": "1.1*phi + 0.22*(exp(0.3*phi)-1)",
        "teeth": 16,
        "open": True,
        "drive_end": 3.2,
    },
    {
        "name": "open_soft_s_curve",
        "expression": "1.25*phi + 0.12*tanh(1.1*(phi-1.5))",
        "teeth": 16,
        "open": True,
        "drive_end": 3.0,
    },
    {
        "name": "open_soft_step",
        "expression": "1.15*phi + 0.10*atan(2*(phi-1.5))",
        "teeth": 16,
        "open": True,
        "drive_end": 3.0,
    },
    {
        "name": "open_quadratic_ramp",
        "expression": "1.35*phi + 0.025*phi**2",
        "teeth": 16,
        "open": True,
        "drive_end": 3.2,
    },
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate the extended closed/open noncircular gear gallery."
    )
    parser.add_argument("--out", type=Path, default=root / "out")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    generator = Path(__file__).with_name("generate_sympy.py")
    for sample in SAMPLES:
        command = [
            sys.executable,
            str(generator),
            sample["expression"],
            "--name",
            sample["name"],
            "--teeth",
            str(sample["teeth"]),
            "--samples",
            "4096",
            "--samples-per-radian",
            "90",
            "--out",
            str(args.out),
            "--no-render",
        ]
        if sample.get("open"):
            command.extend(
                [
                    "--open",
                    "--drive-end",
                    str(sample["drive_end"]),
                    "--padding-pitches",
                    "8",
                ]
            )
        subprocess.run(command, check=True)

    if not args.no_render:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("render_samples.py")),
                "--input",
                str(args.out),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
