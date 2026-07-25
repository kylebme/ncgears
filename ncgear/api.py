from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import sympy as sp


PHI = sp.Symbol("phi", real=True)
_PARSE_LOCALS = {
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


def _parse_expression(expression: str | sp.Expr) -> sp.Expr:
    parsed = sp.sympify(expression, locals=_PARSE_LOCALS)
    extra_symbols = parsed.free_symbols - {PHI}
    if extra_symbols:
        names = ", ".join(sorted(str(symbol) for symbol in extra_symbols))
        raise ValueError(f"Centrode expression may only contain phi; found: {names}")
    return parsed


def _evaluate_centrode(
    expression: sp.Expr, phis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    for order in range(3):
        derivative = sp.diff(expression, PHI, order)
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
            raise ValueError(
                "Centrode radius and its first two derivatives must stay finite."
            )
        arrays.append(values)
    if np.any(arrays[0] <= 0.0):
        raise ValueError("Centrode radius must stay strictly positive.")
    return arrays[0], arrays[1], arrays[2]


def generate_from_centrode(
    expression: str | sp.Expr,
    *,
    name: str = "centrode",
    description: str | None = None,
    teeth: int = 16,
    module: float = 1.0,
    pressure_angle_deg: float = 20.0,
    addendum_factor: float = 1.0,
    dedendum_factor: float = 1.2,
    fillet_factor: float = 0.3,
    drive_start: float = 0.0,
    drive_end: float = 2.0 * math.pi,
    period: float = 2.0 * math.pi,
    open_: bool = False,
    samples: int = 8192,
    padding_pitches: float = 7.0,
    reference_center_distance: float | None = None,
    target_cycle_delta: float = 2.0 * math.pi,
    profile: str = "involute",
    cycloidal_rolling_factor: float = 0.35,
    samples_per_radian: int = 110,
    output_directory: str | Path = "out",
    generator: str | Path = "build/ncgear_generate",
    render: bool = True,
) -> Path:
    """Generate a conjugate pair from the driving gear's polar centrode.

    ``expression`` is ``r(phi)``. Its length unit is arbitrary: the C++ core
    scales the curve to ``teeth * pi * module`` pitch length. A supplied
    ``reference_center_distance`` uses that same arbitrary unit and therefore
    controls the ratio. When it is omitted, the distance is solved so the net
    angular advance equals ``target_cycle_delta`` (2*pi by default).

    The returned path contains ``centrode.csv``, both gear outlines, metadata,
    and rendered PNGs unless ``render=False``.
    """

    if samples < 1024:
        raise ValueError("samples must be at least 1024")
    if teeth < 6:
        raise ValueError("teeth must be at least 6")
    if not drive_start < drive_end:
        raise ValueError("drive_start must be less than drive_end")
    if profile not in {"involute", "cycloidal"}:
        raise ValueError("profile must be 'involute' or 'cycloidal'")
    if (
        reference_center_distance is not None
        and reference_center_distance <= 0.0
    ):
        raise ValueError("reference_center_distance must be positive")

    parsed = _parse_expression(expression)
    if open_:
        mean_pitch = (drive_end - drive_start) / float(teeth)
        padding = padding_pitches * mean_pitch
        domain_start = drive_start - padding
        domain_end = drive_end + padding
        phis = np.linspace(domain_start, domain_end, samples)
    else:
        domain_start = drive_start
        domain_end = drive_start + period
        drive_end = domain_end
        for order in range(3):
            derivative = sp.diff(parsed, PHI, order)
            start_value = float(sp.N(derivative.subs(PHI, domain_start)))
            end_value = float(sp.N(derivative.subs(PHI, domain_end)))
            if not math.isclose(
                start_value, end_value, rel_tol=1e-7, abs_tol=1e-7
            ):
                raise ValueError(
                    f"Closed mode requires centrode derivative order {order} "
                    "to be periodic."
                )
        phis = np.linspace(domain_start, domain_end, samples, endpoint=False)

    arrays = _evaluate_centrode(parsed, phis)
    output_root = Path(output_directory)
    sample_directory = output_root / name
    sample_directory.mkdir(parents=True, exist_ok=True)
    centrode_path = sample_directory / "centrode.csv"
    np.savetxt(
        centrode_path,
        np.column_stack((phis, *arrays)),
        delimiter=",",
        header="phi,radius,radius1,radius2",
        comments="",
        fmt="%.17g",
    )

    generator_path = Path(generator)
    if not generator_path.exists():
        raise FileNotFoundError(f"Generator executable not found: {generator_path}")
    command = [
        str(generator_path),
        "--centrode-csv",
        str(centrode_path),
        "--name",
        name,
        "--description",
        description or f"drive centrode r(phi) = {sp.sstr(parsed)}",
        "--teeth",
        str(teeth),
        "--module",
        str(module),
        "--pressure-angle-deg",
        str(pressure_angle_deg),
        "--addendum-factor",
        str(addendum_factor),
        "--dedendum-factor",
        str(dedendum_factor),
        "--fillet-factor",
        str(fillet_factor),
        "--domain-start",
        repr(domain_start),
        "--domain-end",
        repr(domain_end),
        "--active-start",
        repr(drive_start),
        "--active-end",
        repr(drive_end),
        "--period",
        repr(period),
        "--cycle-delta",
        repr(target_cycle_delta),
        "--samples-per-radian",
        str(samples_per_radian),
        "--profile",
        profile,
        "--cycloidal-rolling-factor",
        str(cycloidal_rolling_factor),
        "--out",
        str(output_root),
    ]
    if reference_center_distance is not None:
        command.extend(
            ["--centrode-center-distance", repr(reference_center_distance)]
        )
    if open_:
        command.append("--open")
    subprocess.run(command, check=True)

    if render:
        renderer = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "render_samples.py"
        )
        subprocess.run(
            [
                "python3",
                str(renderer),
                "--input",
                str(output_root),
                "--sample",
                name,
            ],
            check=True,
        )
    return sample_directory
