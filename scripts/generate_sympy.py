#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np
import sympy as sp


PHI = sp.Symbol("phi", real=True)
PARSE_LOCALS = {
    "phi": PHI,
    "pi": sp.pi,
    "E": sp.E,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
}


def parse_expression(expression: str) -> sp.Expr:
    parsed = sp.sympify(expression, locals=PARSE_LOCALS)
    extra_symbols = parsed.free_symbols - {PHI}
    if extra_symbols:
        names = ", ".join(sorted(str(symbol) for symbol in extra_symbols))
        raise ValueError(f"Expression may only contain phi; found: {names}")
    return parsed


def evaluate_expression(expr: sp.Expr, phis: np.ndarray) -> tuple[np.ndarray, ...]:
    derivatives = [expr] + [sp.diff(expr, PHI, order) for order in range(1, 4)]
    arrays: list[np.ndarray] = []
    for derivative in derivatives:
        try:
            function = sp.lambdify(PHI, derivative, modules=["numpy", "scipy"])
            values = np.asarray(function(phis), dtype=float)
        except Exception as error:
            raise ValueError(
                f"Could not numerically evaluate {sp.sstr(derivative)}: {error}"
            ) from error
        if values.shape == ():
            values = np.full_like(phis, float(values))
        if values.shape != phis.shape or not np.all(np.isfinite(values)):
            raise ValueError("Expression and its first three derivatives must stay finite.")
        arrays.append(values)
    if np.any(arrays[1] <= 1e-9):
        raise ValueError("Transmission derivative must stay strictly positive.")
    return tuple(arrays)


def write_transmission_csv(
    path: Path,
    phis: np.ndarray,
    arrays: tuple[np.ndarray, ...],
) -> None:
    values = np.column_stack((phis, *arrays))
    np.savetxt(
        path,
        values,
        delimiter=",",
        header="phi,psi,psi1,psi2,psi3",
        comments="",
        fmt="%.17g",
    )


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate noncircular gears from a SymPy transmission expression."
    )
    parser.add_argument("expression", help="SymPy expression in the variable phi")
    parser.add_argument("--name", default="expression")
    parser.add_argument("--open", action="store_true", help="Generate finite open gear segments")
    parser.add_argument("--teeth", type=int, default=16, help="Drive teeth or open-segment teeth")
    parser.add_argument("--module", type=float, default=1.0)
    parser.add_argument("--pressure-angle-deg", type=float, default=20.0)
    parser.add_argument("--addendum-factor", type=float, default=1.0)
    parser.add_argument("--dedendum-factor", type=float, default=1.2)
    parser.add_argument("--fillet-factor", type=float, default=0.3)
    parser.add_argument("--drive-start", type=float, default=0.0)
    parser.add_argument(
        "--drive-end",
        type=float,
        default=2.0 * math.pi,
        help="End of the active drive interval",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=2.0 * math.pi,
        help="Drive period for closed gears",
    )
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--padding-pitches", type=float, default=7.0)
    parser.add_argument(
        "--allow-nonconvex",
        action="store_true",
        help="Deprecated compatibility flag; nonconvex centrodes are supported automatically",
    )
    parser.add_argument(
        "--profile",
        choices=("involute", "cycloidal"),
        default="involute",
        help="Swept rack-cutter profile family",
    )
    parser.add_argument(
        "--cycloidal-rolling-factor",
        type=float,
        default=0.35,
        help="0..1 blend from straight to cycloidal rack flanks",
    )
    parser.add_argument("--samples-per-radian", type=int, default=110)
    parser.add_argument("--out", type=Path, default=root / "out")
    parser.add_argument(
        "--generator",
        type=Path,
        default=root / "build" / "ncgear_generate",
    )
    parser.add_argument("--no-render", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 1024:
        raise SystemExit("--samples must be at least 1024")
    if args.teeth < 6:
        raise SystemExit("--teeth must be at least 6")
    if not args.generator.exists():
        raise SystemExit(f"Generator executable not found: {args.generator}")

    expression = parse_expression(args.expression)
    active_start = args.drive_start
    active_end = args.drive_end
    if not active_start < active_end:
        raise SystemExit("--drive-start must be less than --drive-end")

    expression = sp.simplify(expression - expression.subs(PHI, active_start))
    if args.open:
        mean_pitch = (active_end - active_start) / float(args.teeth)
        padding = args.padding_pitches * mean_pitch
        domain_start = active_start - padding
        domain_end = active_end + padding
        phis = np.linspace(domain_start, domain_end, args.samples)
        cycle_delta = float(sp.N(expression.subs(PHI, active_end)))
    else:
        domain_start = active_start
        domain_end = active_start + args.period
        active_end = domain_end
        for order in range(1, 4):
            derivative = sp.diff(expression, PHI, order)
            start_value = float(sp.N(derivative.subs(PHI, domain_start)))
            end_value = float(sp.N(derivative.subs(PHI, domain_end)))
            if not math.isclose(start_value, end_value, rel_tol=1e-7, abs_tol=1e-7):
                raise SystemExit(
                    f"Closed mode requires derivative order {order} to be periodic."
                )
        cycle_delta = float(
            sp.N(
                expression.subs(PHI, domain_end)
                - expression.subs(PHI, domain_start)
            )
        )
        if cycle_delta <= 0.0:
            raise SystemExit("Closed mode requires a positive cycle advance.")
        exact_driven_teeth = args.teeth * args.period / cycle_delta
        driven_teeth = round(exact_driven_teeth)
        if driven_teeth <= 0 or not math.isclose(
            exact_driven_teeth, driven_teeth, rel_tol=1e-8, abs_tol=1e-8
        ):
            raise SystemExit(
                "Closed mode requires teeth * period / cycle advance to be an integer."
            )
        driven_cycle = args.period * driven_teeth / args.teeth
        for order in range(1, 4):
            derivative = sp.diff(expression, PHI, order)
            difference = sp.simplify(
                derivative.subs(PHI, PHI + driven_cycle) - derivative
            )
            if difference != 0:
                check_points = np.linspace(domain_start, domain_end, 257)
                function = sp.lambdify(PHI, difference, modules=["numpy", "scipy"])
                values = np.asarray(function(check_points), dtype=float)
                if values.shape == ():
                    values = np.full_like(check_points, float(values))
                if np.max(np.abs(values)) > 1e-7:
                    raise SystemExit(
                        "Closed mode requires derivatives to repeat over one "
                        "driven-gear revolution."
                    )
        sample_multiple = args.teeth // math.gcd(args.teeth, driven_teeth)
        closed_sample_count = (
            (args.samples + sample_multiple - 1) // sample_multiple
        ) * sample_multiple
        phis = np.linspace(
            domain_start, domain_end, closed_sample_count, endpoint=False
        )

    arrays = evaluate_expression(expression, phis)
    sample_directory = args.out / args.name
    sample_directory.mkdir(parents=True, exist_ok=True)
    transmission_path = sample_directory / "transmission.csv"
    write_transmission_csv(transmission_path, phis, arrays)

    command = [
        str(args.generator),
        "--transmission-csv",
        str(transmission_path),
        "--name",
        args.name,
        "--description",
        f"psi(phi) = {sp.sstr(expression)}",
        "--teeth",
        str(args.teeth),
        "--module",
        str(args.module),
        "--pressure-angle-deg",
        str(args.pressure_angle_deg),
        "--addendum-factor",
        str(args.addendum_factor),
        "--dedendum-factor",
        str(args.dedendum_factor),
        "--fillet-factor",
        str(args.fillet_factor),
        "--domain-start",
        repr(domain_start),
        "--domain-end",
        repr(domain_end),
        "--active-start",
        repr(active_start),
        "--active-end",
        repr(active_end),
        "--period",
        repr(args.period),
        "--cycle-delta",
        repr(cycle_delta),
        "--samples-per-radian",
        str(args.samples_per_radian),
        "--profile",
        args.profile,
        "--cycloidal-rolling-factor",
        str(args.cycloidal_rolling_factor),
        "--out",
        str(args.out),
    ]
    if args.open:
        command.append("--open")
    if args.allow_nonconvex:
        command.append("--allow-nonconvex")
    subprocess.run(command, check=True)

    if not args.no_render:
        renderer = Path(__file__).with_name("render_samples.py")
        subprocess.run(
            ["python3", str(renderer), "--input", str(args.out), "--sample", args.name],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
