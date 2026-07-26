"""High-level Python interface for noncircular gear generation."""

from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import sympy as sp

from .errors import GenerationError
from .native import native_generator
from .result import GearPair

PHI = sp.Symbol("phi", real=True)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
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
    "atan2": sp.atan2,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "erf": sp.erf,
    "Abs": sp.Abs,
}


def _parse_expression(expression: str | sp.Expr) -> sp.Expr:
    if not isinstance(expression, (str, sp.Expr)):
        raise TypeError("expression must be a string or a SymPy expression")
    try:
        parsed = sp.sympify(expression, locals=_PARSE_LOCALS)
    except (sp.SympifyError, TypeError, SyntaxError) as error:
        raise ValueError(f"Invalid expression: {error}") from error
    extra_symbols = parsed.free_symbols - {PHI}
    if extra_symbols:
        names = ", ".join(sorted(str(symbol) for symbol in extra_symbols))
        raise ValueError(f"Expression may only contain phi; found: {names}")
    return parsed


def _evaluate(
    expression: sp.Expr,
    phis: np.ndarray,
    derivative_orders: Sequence[int],
) -> tuple[np.ndarray, ...]:
    arrays: list[np.ndarray] = []
    for order in derivative_orders:
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
                "The expression and all required derivatives must stay finite "
                "over the design interval."
            )
        arrays.append(values)
    return tuple(arrays)


def _validate_common(
    *,
    name: str,
    teeth: int,
    module: float,
    pressure_angle_deg: float,
    addendum_factor: float,
    dedendum_factor: float,
    fillet_factor: float,
    drive_start: float,
    drive_end: float,
    period: float,
    samples: int,
    padding_pitches: float,
    profile: str,
    cycloidal_rolling_factor: float,
    samples_per_radian: int,
) -> None:
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(
            "name must start with a letter or digit and contain only letters, "
            "digits, dots, underscores, and hyphens"
        )
    if teeth < 6:
        raise ValueError("teeth must be at least 6")
    if module <= 0.0:
        raise ValueError("module must be positive")
    if not 0.0 < pressure_angle_deg < 45.0:
        raise ValueError("pressure_angle_deg must be between 0 and 45")
    if min(addendum_factor, dedendum_factor, fillet_factor) <= 0.0:
        raise ValueError("tooth height and fillet factors must be positive")
    if not drive_start < drive_end:
        raise ValueError("drive_start must be less than drive_end")
    if period <= 0.0:
        raise ValueError("period must be positive")
    if samples < 1024:
        raise ValueError("samples must be at least 1024")
    if padding_pitches < 2.5:
        raise ValueError("padding_pitches must be at least 2.5")
    if profile not in {"involute", "cycloidal"}:
        raise ValueError("profile must be 'involute' or 'cycloidal'")
    if not 0.0 <= cycloidal_rolling_factor <= 1.0:
        raise ValueError("cycloidal_rolling_factor must be between 0 and 1")
    if samples_per_radian < 20:
        raise ValueError("samples_per_radian must be at least 20")


def _write_samples(
    path: Path,
    header: str,
    phis: np.ndarray,
    arrays: Sequence[np.ndarray],
) -> None:
    np.savetxt(
        path,
        np.column_stack((phis, *arrays)),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.17g",
    )


def _run_generator(
    *,
    input_flag: str,
    input_path: Path,
    name: str,
    description: str,
    teeth: int,
    module: float,
    pressure_angle_deg: float,
    addendum_factor: float,
    dedendum_factor: float,
    fillet_factor: float,
    domain_start: float,
    domain_end: float,
    active_start: float,
    active_end: float,
    period: float,
    cycle_delta: float,
    open_: bool,
    profile: str,
    cycloidal_rolling_factor: float,
    samples_per_radian: int,
    output_root: Path,
    generator: str | Path | None,
    extra_arguments: Sequence[str] = (),
    render: bool,
) -> GearPair:
    executable = native_generator(generator)
    command = [
        str(executable),
        input_flag,
        str(input_path),
        "--name",
        name,
        "--description",
        description,
        "--teeth",
        str(teeth),
        "--module",
        repr(module),
        "--pressure-angle-deg",
        repr(pressure_angle_deg),
        "--addendum-factor",
        repr(addendum_factor),
        "--dedendum-factor",
        repr(dedendum_factor),
        "--fillet-factor",
        repr(fillet_factor),
        "--domain-start",
        repr(domain_start),
        "--domain-end",
        repr(domain_end),
        "--active-start",
        repr(active_start),
        "--active-end",
        repr(active_end),
        "--period",
        repr(period),
        "--cycle-delta",
        repr(cycle_delta),
        "--samples-per-radian",
        str(samples_per_radian),
        "--profile",
        profile,
        "--cycloidal-rolling-factor",
        repr(cycloidal_rolling_factor),
        "--out",
        str(output_root),
        *extra_arguments,
    ]
    if open_:
        command.append("--open")

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GenerationError(
            f"ncgear could not generate {name!r}: "
            f"{detail or f'generator exited with status {completed.returncode}'}"
        )

    pair = GearPair.load(
        output_root / name,
        generator_log=completed.stdout.strip(),
    )
    if render:
        pair.render()
    return pair


def generate_from_transmission(
    expression: str | sp.Expr,
    *,
    name: str = "gear_pair",
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
    profile: str = "involute",
    cycloidal_rolling_factor: float = 0.35,
    samples_per_radian: int = 110,
    output_directory: str | Path = "out",
    generator: str | Path | None = None,
    render: bool = False,
) -> GearPair:
    """Generate a conjugate pair from the motion law ``psi(phi)``.

    ``phi`` is the drive angle in radians and the expression's value is the
    driven angle. For example, ``"phi - 0.08*sin(2*phi)"`` produces a smooth
    variable-speed 1:1 pair. The returned :class:`GearPair` contains NumPy
    outlines, verification metadata, preview rendering, and SVG/DXF exports.
    """

    _validate_common(
        name=name,
        teeth=teeth,
        module=module,
        pressure_angle_deg=pressure_angle_deg,
        addendum_factor=addendum_factor,
        dedendum_factor=dedendum_factor,
        fillet_factor=fillet_factor,
        drive_start=drive_start,
        drive_end=drive_end,
        period=period,
        samples=samples,
        padding_pitches=padding_pitches,
        profile=profile,
        cycloidal_rolling_factor=cycloidal_rolling_factor,
        samples_per_radian=samples_per_radian,
    )
    parsed = _parse_expression(expression)
    parsed = sp.simplify(parsed - parsed.subs(PHI, drive_start))
    active_start = drive_start
    active_end = drive_end

    if open_:
        mean_pitch = (active_end - active_start) / float(teeth)
        padding = padding_pitches * mean_pitch
        domain_start = active_start - padding
        domain_end = active_end + padding
        phis = np.linspace(domain_start, domain_end, samples)
        cycle_delta = float(
            sp.N(parsed.subs(PHI, active_end) - parsed.subs(PHI, active_start))
        )
        if cycle_delta <= 0.0:
            raise ValueError("The transmission must advance over the active interval")
    else:
        domain_start = active_start
        domain_end = active_start + period
        active_end = domain_end
        for order in range(1, 4):
            derivative = sp.diff(parsed, PHI, order)
            start_value = float(sp.N(derivative.subs(PHI, domain_start)))
            end_value = float(sp.N(derivative.subs(PHI, domain_end)))
            if not math.isclose(start_value, end_value, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(
                    f"Closed mode requires transmission derivative order {order} "
                    "to be periodic"
                )
        cycle_delta = float(
            sp.N(parsed.subs(PHI, domain_end) - parsed.subs(PHI, domain_start))
        )
        if cycle_delta <= 0.0:
            raise ValueError("Closed mode requires a positive cycle advance")
        exact_driven_teeth = teeth * period / cycle_delta
        driven_teeth = round(exact_driven_teeth)
        if driven_teeth <= 0 or not math.isclose(
            exact_driven_teeth, driven_teeth, rel_tol=1e-8, abs_tol=1e-8
        ):
            raise ValueError(
                "Closed mode requires teeth * period / cycle advance to be an "
                "integer; adjust the drive tooth count or motion law"
            )
        driven_cycle = period * driven_teeth / teeth
        check_points = np.linspace(domain_start, domain_end, 257)
        for order in range(1, 4):
            derivative = sp.diff(parsed, PHI, order)
            difference = sp.simplify(
                derivative.subs(PHI, PHI + driven_cycle) - derivative
            )
            if difference != 0:
                values = _evaluate(difference, check_points, (0,))[0]
                if np.max(np.abs(values)) > 1e-7:
                    raise ValueError(
                        "Closed mode requires the motion derivatives to repeat "
                        "over one driven-gear revolution"
                    )
        sample_multiple = teeth // math.gcd(teeth, driven_teeth)
        sample_count = (
            (samples + sample_multiple - 1) // sample_multiple
        ) * sample_multiple
        phis = np.linspace(domain_start, domain_end, sample_count, endpoint=False)

    arrays = _evaluate(parsed, phis, (0, 1, 2, 3))
    if np.any(arrays[1] <= 1e-9):
        raise ValueError(
            "The transmission derivative must stay strictly positive over the "
            "complete sampled domain"
        )

    output_root = Path(output_directory).expanduser().resolve()
    sample_directory = output_root / name
    sample_directory.mkdir(parents=True, exist_ok=True)
    transmission_path = sample_directory / "transmission.csv"
    _write_samples(
        transmission_path,
        "phi,psi,psi1,psi2,psi3",
        phis,
        arrays,
    )
    return _run_generator(
        input_flag="--transmission-csv",
        input_path=transmission_path,
        name=name,
        description=description or f"psi(phi) = {sp.sstr(parsed)}",
        teeth=teeth,
        module=module,
        pressure_angle_deg=pressure_angle_deg,
        addendum_factor=addendum_factor,
        dedendum_factor=dedendum_factor,
        fillet_factor=fillet_factor,
        domain_start=domain_start,
        domain_end=domain_end,
        active_start=active_start,
        active_end=active_end,
        period=period,
        cycle_delta=cycle_delta,
        open_=open_,
        profile=profile,
        cycloidal_rolling_factor=cycloidal_rolling_factor,
        samples_per_radian=samples_per_radian,
        output_root=output_root,
        generator=generator,
        render=render,
    )


def generate_from_centrode(
    expression: str | sp.Expr,
    *,
    name: str = "centrode_pair",
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
    generator: str | Path | None = None,
    render: bool = False,
) -> GearPair:
    """Generate a conjugate pair from the drive pitch radius ``r(phi)``.

    The radius expression may use arbitrary units. It is scaled so the drive
    pitch length equals ``teeth * pi * module``. When no reference center
    distance is supplied, ncgear solves one for ``target_cycle_delta``.
    """

    _validate_common(
        name=name,
        teeth=teeth,
        module=module,
        pressure_angle_deg=pressure_angle_deg,
        addendum_factor=addendum_factor,
        dedendum_factor=dedendum_factor,
        fillet_factor=fillet_factor,
        drive_start=drive_start,
        drive_end=drive_end,
        period=period,
        samples=samples,
        padding_pitches=padding_pitches,
        profile=profile,
        cycloidal_rolling_factor=cycloidal_rolling_factor,
        samples_per_radian=samples_per_radian,
    )
    if reference_center_distance is not None and reference_center_distance <= 0.0:
        raise ValueError("reference_center_distance must be positive")
    if target_cycle_delta <= 0.0:
        raise ValueError("target_cycle_delta must be positive")

    parsed = _parse_expression(expression)
    active_start = drive_start
    active_end = drive_end
    if open_:
        mean_pitch = (active_end - active_start) / float(teeth)
        padding = padding_pitches * mean_pitch
        domain_start = active_start - padding
        domain_end = active_end + padding
        phis = np.linspace(domain_start, domain_end, samples)
    else:
        domain_start = active_start
        domain_end = active_start + period
        active_end = domain_end
        for order in range(3):
            derivative = sp.diff(parsed, PHI, order)
            start_value = float(sp.N(derivative.subs(PHI, domain_start)))
            end_value = float(sp.N(derivative.subs(PHI, domain_end)))
            if not math.isclose(start_value, end_value, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError(
                    f"Closed mode requires centrode derivative order {order} "
                    "to be periodic"
                )
        phis = np.linspace(domain_start, domain_end, samples, endpoint=False)

    arrays = _evaluate(parsed, phis, (0, 1, 2))
    if np.any(arrays[0] <= 0.0):
        raise ValueError(
            "The centrode radius must stay strictly positive over the complete "
            "sampled domain"
        )

    output_root = Path(output_directory).expanduser().resolve()
    sample_directory = output_root / name
    sample_directory.mkdir(parents=True, exist_ok=True)
    centrode_path = sample_directory / "centrode.csv"
    _write_samples(
        centrode_path,
        "phi,radius,radius1,radius2",
        phis,
        arrays,
    )
    extra_arguments: list[str] = []
    if reference_center_distance is not None:
        extra_arguments.extend(
            ["--centrode-center-distance", repr(reference_center_distance)]
        )
    return _run_generator(
        input_flag="--centrode-csv",
        input_path=centrode_path,
        name=name,
        description=description or f"drive centrode r(phi) = {sp.sstr(parsed)}",
        teeth=teeth,
        module=module,
        pressure_angle_deg=pressure_angle_deg,
        addendum_factor=addendum_factor,
        dedendum_factor=dedendum_factor,
        fillet_factor=fillet_factor,
        domain_start=domain_start,
        domain_end=domain_end,
        active_start=active_start,
        active_end=active_end,
        period=period,
        cycle_delta=target_cycle_delta,
        open_=open_,
        profile=profile,
        cycloidal_rolling_factor=cycloidal_rolling_factor,
        samples_per_radian=samples_per_radian,
        output_root=output_root,
        generator=generator,
        extra_arguments=extra_arguments,
        render=render,
    )


# The most common use starts with a motion law, so it receives the shortest
# name. The explicit function remains available for discoverability.
generate = generate_from_transmission
