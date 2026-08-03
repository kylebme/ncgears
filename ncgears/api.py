"""High-level Python interface for noncircular gear generation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import sympy as sp

from ._policy import (
    DEFAULT_ADDENDUM_FACTOR,
    DEFAULT_DEDENDUM_FACTOR,
    DEFAULT_FILLET_FACTOR,
    DEFAULT_INPUT_SAMPLES,
    DEFAULT_MODULE,
    DEFAULT_PADDING_PITCHES,
    DEFAULT_PRESSURE_ANGLE_DEG,
    DEFAULT_SAMPLES_PER_RADIAN,
    DEFAULT_TEETH,
    EXPRESSION_ENDPOINT_TOLERANCE,
    MAX_PRESSURE_ANGLE_DEG,
    MIN_INPUT_SAMPLES,
    MIN_MOTION_RATIO,
    MIN_PADDING_PITCHES,
    MIN_SAMPLES_PER_RADIAN,
    MIN_TEETH,
    TOOTH_COUNT_ABS_TOLERANCE,
)
from .engine import generate_geometry, load_engine_config
from .errors import GenerationError
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
    clearance: float,
    max_backlash_deg: float | None,
    drive_start: float,
    drive_end: float,
    period: float,
    samples: int,
    padding_pitches: float,
    samples_per_radian: int,
) -> None:
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(
            "name must start with a letter or digit and contain only letters, "
            "digits, dots, underscores, and hyphens"
        )
    if teeth < MIN_TEETH:
        raise ValueError(f"teeth must be at least {MIN_TEETH}")
    if module <= 0.0:
        raise ValueError("module must be positive")
    if not 0.0 < pressure_angle_deg < MAX_PRESSURE_ANGLE_DEG:
        raise ValueError(
            f"pressure_angle_deg must be between 0 and {MAX_PRESSURE_ANGLE_DEG:g}"
        )
    if addendum_factor <= 0.0 or dedendum_factor <= 0.0 or fillet_factor < 0.0:
        raise ValueError(
            "tooth height factors must be positive and fillet_factor "
            "must be nonnegative"
        )
    if dedendum_factor <= fillet_factor:
        raise ValueError("dedendum_factor must exceed fillet_factor")
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance must be finite and nonnegative")
    if max_backlash_deg is not None and (
        not math.isfinite(max_backlash_deg) or max_backlash_deg < 0.0
    ):
        raise ValueError("max_backlash_deg must be finite and nonnegative")
    if clearance != 0.0 and max_backlash_deg is not None:
        raise ValueError("clearance and max_backlash_deg are mutually exclusive")
    if not drive_start < drive_end:
        raise ValueError("drive_start must be less than drive_end")
    if period <= 0.0:
        raise ValueError("period must be positive")
    if samples < MIN_INPUT_SAMPLES:
        raise ValueError(f"samples must be at least {MIN_INPUT_SAMPLES}")
    if padding_pitches < MIN_PADDING_PITCHES:
        raise ValueError(f"padding_pitches must be at least {MIN_PADDING_PITCHES:g}")
    if samples_per_radian < MIN_SAMPLES_PER_RADIAN:
        raise ValueError(
            f"samples_per_radian must be at least {MIN_SAMPLES_PER_RADIAN}"
        )


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
    clearance: float,
    max_backlash_deg: float | None,
    domain_start: float,
    domain_end: float,
    active_start: float,
    active_end: float,
    period: float,
    cycle_delta: float,
    open_: bool,
    samples_per_radian: int,
    output_root: Path,
    generator: str | Path | None,
    extra_arguments: Sequence[str] = (),
    render: bool,
    plot: bool,
) -> GearPair:
    if generator is not None:
        raise ValueError(
            "generator overrides are not supported by the Python-only engine"
        )
    try:
        config = load_engine_config(
            input_flag=input_flag,
            input_path=input_path,
            name=name,
            description=description,
            teeth=teeth,
            module=module,
            pressure_angle_deg=pressure_angle_deg,
            addendum_factor=addendum_factor,
            dedendum_factor=dedendum_factor,
            fillet_factor=fillet_factor,
            clearance=clearance,
            max_backlash_deg=max_backlash_deg,
            domain_start=domain_start,
            domain_end=domain_end,
            active_start=active_start,
            active_end=active_end,
            period=period,
            cycle_delta=cycle_delta,
            open_=open_,
            extra_arguments=extra_arguments,
        )
        result = generate_geometry(config, samples_per_radian)
        result_directory = output_root / name
        result_directory.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            result_directory / "drive.csv",
            result.drive_outline,
            delimiter=",",
            header="x,y",
            comments="",
            fmt="%.17g",
        )
        np.savetxt(
            result_directory / "driven.csv",
            result.driven_outline,
            delimiter=",",
            header="x,y",
            comments="",
            fmt="%.17g",
        )
        (result_directory / "metadata.json").write_text(
            json.dumps(result.metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    except (ValueError, RuntimeError, OSError) as error:
        raise GenerationError(
            f"ncgears could not generate {name!r}: {error}"
        ) from error

    pair = GearPair.load(
        output_root / name,
        generator_log=result.log,
    )
    if render:
        pair.render()
    if plot:
        pair.plot()
    return pair


def generate_from_transmission(
    expression: str | sp.Expr,
    *,
    name: str = "gear_pair",
    description: str | None = None,
    teeth: int = DEFAULT_TEETH,
    module: float = DEFAULT_MODULE,
    pressure_angle_deg: float = DEFAULT_PRESSURE_ANGLE_DEG,
    addendum_factor: float = DEFAULT_ADDENDUM_FACTOR,
    dedendum_factor: float = DEFAULT_DEDENDUM_FACTOR,
    fillet_factor: float = DEFAULT_FILLET_FACTOR,
    clearance: float = 0.0,
    max_backlash_deg: float | None = None,
    drive_start: float = 0.0,
    drive_end: float = 2.0 * math.pi,
    period: float = 2.0 * math.pi,
    open_: bool = False,
    samples: int = DEFAULT_INPUT_SAMPLES,
    padding_pitches: float = DEFAULT_PADDING_PITCHES,
    samples_per_radian: int = DEFAULT_SAMPLES_PER_RADIAN,
    output_directory: str | Path = "out",
    generator: str | Path | None = None,
    render: bool = False,
    plot: bool = False,
) -> GearPair:
    """Generate a conjugate pair from the motion law ``psi(phi)``.

    ``phi`` is the drive angle in radians and the expression's value is the
    driven angle. For example, ``"phi - 0.08*sin(2*phi)"`` produces a smooth
    variable-speed 1:1 pair. The returned :class:`GearPair` contains NumPy
    outlines, verification metadata, preview rendering, and SVG/DXF exports.

    ``clearance`` is the inward face offset applied to each gear, normalized by
    ``module``. Alternatively, ``max_backlash_deg`` derives that offset from
    the requested maximum total angular free play of the driven gear. The two
    options are mutually exclusive.
    """

    _validate_common(
        name=name,
        teeth=teeth,
        module=module,
        pressure_angle_deg=pressure_angle_deg,
        addendum_factor=addendum_factor,
        dedendum_factor=dedendum_factor,
        fillet_factor=fillet_factor,
        clearance=clearance,
        max_backlash_deg=max_backlash_deg,
        drive_start=drive_start,
        drive_end=drive_end,
        period=period,
        samples=samples,
        padding_pitches=padding_pitches,
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
            if not math.isclose(
                start_value,
                end_value,
                rel_tol=EXPRESSION_ENDPOINT_TOLERANCE,
                abs_tol=EXPRESSION_ENDPOINT_TOLERANCE,
            ):
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
            exact_driven_teeth,
            driven_teeth,
            rel_tol=TOOTH_COUNT_ABS_TOLERANCE,
            abs_tol=TOOTH_COUNT_ABS_TOLERANCE,
        ):
            raise ValueError(
                "Closed mode requires teeth * period / cycle advance to be an "
                "integer; adjust the drive tooth count or motion law"
            )
        sample_multiple = teeth // math.gcd(teeth, driven_teeth)
        sample_count = (
            (samples + sample_multiple - 1) // sample_multiple
        ) * sample_multiple
        phis = np.linspace(domain_start, domain_end, sample_count, endpoint=False)

    arrays = _evaluate(parsed, phis, (0, 1, 2, 3))
    if np.any(arrays[1] <= MIN_MOTION_RATIO):
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
        clearance=clearance,
        max_backlash_deg=max_backlash_deg,
        domain_start=domain_start,
        domain_end=domain_end,
        active_start=active_start,
        active_end=active_end,
        period=period,
        cycle_delta=cycle_delta,
        open_=open_,
        samples_per_radian=samples_per_radian,
        output_root=output_root,
        generator=generator,
        render=render,
        plot=plot,
    )


def generate_from_centrode(
    expression: str | sp.Expr,
    *,
    name: str = "centrode_pair",
    description: str | None = None,
    teeth: int = DEFAULT_TEETH,
    module: float = DEFAULT_MODULE,
    pressure_angle_deg: float = DEFAULT_PRESSURE_ANGLE_DEG,
    addendum_factor: float = DEFAULT_ADDENDUM_FACTOR,
    dedendum_factor: float = DEFAULT_DEDENDUM_FACTOR,
    fillet_factor: float = DEFAULT_FILLET_FACTOR,
    clearance: float = 0.0,
    max_backlash_deg: float | None = None,
    drive_start: float = 0.0,
    drive_end: float = 2.0 * math.pi,
    period: float = 2.0 * math.pi,
    open_: bool = False,
    samples: int = DEFAULT_INPUT_SAMPLES,
    padding_pitches: float = DEFAULT_PADDING_PITCHES,
    reference_center_distance: float | None = None,
    target_cycle_delta: float = 2.0 * math.pi,
    samples_per_radian: int = DEFAULT_SAMPLES_PER_RADIAN,
    output_directory: str | Path = "out",
    generator: str | Path | None = None,
    render: bool = False,
    plot: bool = False,
) -> GearPair:
    """Generate a conjugate pair from the drive pitch radius ``r(phi)``.

    The radius expression may use arbitrary units. It is scaled so the drive
    pitch length equals ``teeth * pi * module``. When no reference center
    distance is supplied, ncgears solves one for ``target_cycle_delta``.
    ``clearance`` and ``max_backlash_deg`` have the same meanings as in
    :func:`generate_from_transmission`.
    """

    _validate_common(
        name=name,
        teeth=teeth,
        module=module,
        pressure_angle_deg=pressure_angle_deg,
        addendum_factor=addendum_factor,
        dedendum_factor=dedendum_factor,
        fillet_factor=fillet_factor,
        clearance=clearance,
        max_backlash_deg=max_backlash_deg,
        drive_start=drive_start,
        drive_end=drive_end,
        period=period,
        samples=samples,
        padding_pitches=padding_pitches,
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
            if not math.isclose(
                start_value,
                end_value,
                rel_tol=EXPRESSION_ENDPOINT_TOLERANCE,
                abs_tol=EXPRESSION_ENDPOINT_TOLERANCE,
            ):
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
        clearance=clearance,
        max_backlash_deg=max_backlash_deg,
        domain_start=domain_start,
        domain_end=domain_end,
        active_start=active_start,
        active_end=active_end,
        period=period,
        cycle_delta=target_cycle_delta,
        open_=open_,
        samples_per_radian=samples_per_radian,
        output_root=output_root,
        generator=generator,
        extra_arguments=extra_arguments,
        render=render,
        plot=plot,
    )


# The most common use starts with a motion law, so it receives the shortest
# name. The explicit function remains available for discoverability.
generate = generate_from_transmission
