"""Python geometry engine for conjugate noncircular gears.

Generalized involute flanks and rounded rack-tip fillets are evaluated from
their analytical envelopes. Shapely/GEOS supplies curve arrangement and
non-working-profile rolling interference removal. Legacy cycloidal profiles
retain the sampled cutter-envelope implementation.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson, simpson
from scipy.optimize import brentq, least_squares
from shapely import Geometry, affinity, make_valid, union_all
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import nearest_points

FloatArray = NDArray[np.float64]
_MAX_GEOMETRY_WORKERS = max(1, min(8, os.cpu_count() or 1))
_ANALYTIC_CHORD_TOLERANCE_FACTOR = 2.5e-5


@dataclass(frozen=True)
class EngineConfig:
    name: str
    description: str
    teeth: int
    module: float
    pressure_angle_deg: float
    addendum_factor: float
    dedendum_factor: float
    fillet_factor: float
    domain_start: float
    domain_end: float
    active_start: float
    active_end: float
    period: float
    cycle_delta: float
    open_: bool
    profile: str
    cycloidal_rolling_factor: float
    input_mode: str
    samples: FloatArray
    reference_center_distance: float = 0.0


@dataclass(frozen=True)
class EngineResult:
    drive_outline: FloatArray
    driven_outline: FloatArray
    metadata: dict[str, object]
    log: str


@dataclass(frozen=True)
class _AnalyticFlankGeometry:
    phase: float
    sign: int
    fillet_root_arc: float
    fillet_transition_arc: float
    flank_transition_arc: float
    flank_tip_arc: float
    addendum_tip_arc: float
    intersection_residual: float
    undercut: bool


@dataclass(frozen=True)
class _AnalyticGearResult:
    outline: FloatArray
    sample_count: int
    flank_sample_count: int
    maximum_envelope_residual: float
    maximum_tangency_residual: float
    maximum_chord_error: float
    maximum_intersection_residual: float
    maximum_join_gap: float
    maximum_fillet_root_residual: float
    undercut_count: int


class _QuinticSeries:
    """Uniform piecewise-quintic Hermite data with derivatives through order 3."""

    def __init__(
        self,
        values: FloatArray,
        first: FloatArray,
        second: FloatArray,
        *,
        domain_start: float,
        domain_end: float,
        period: float,
        periodic: bool,
        cycle_delta: float = 0.0,
    ) -> None:
        self.values = np.asarray(values, dtype=float)
        self.first = np.asarray(first, dtype=float)
        self.second = np.asarray(second, dtype=float)
        self.domain_start = domain_start
        self.domain_end = domain_end
        self.period = period
        self.periodic = periodic
        self.cycle_delta = cycle_delta
        count = len(self.values)
        self.step = (
            period / count if periodic else (domain_end - domain_start) / (count - 1)
        )
        segment_count = count if periodic else count - 1
        next_index = (np.arange(segment_count) + 1) % count
        y0 = self.values[:segment_count]
        y1 = self.values[next_index].copy()
        if periodic:
            y1[-1] += cycle_delta
        d0 = self.first[:segment_count]
        d1 = self.first[next_index]
        dd0 = self.second[:segment_count]
        dd1 = self.second[next_index]

        step = self.step
        a0 = y0
        a1 = step * d0
        a2 = 0.5 * step * step * dd0
        value_error = y1 - (a0 + a1 + a2)
        slope_error = step * d1 - (a1 + 2.0 * a2)
        curvature_error = step * step * dd1 - 2.0 * a2
        self.coefficients = (
            a0,
            a1,
            a2,
            10.0 * value_error - 4.0 * slope_error + 0.5 * curvature_error,
            -15.0 * value_error + 7.0 * slope_error - curvature_error,
            6.0 * value_error - 3.0 * slope_error + 0.5 * curvature_error,
        )

    def __call__(
        self, value: float | FloatArray, derivative: int = 0
    ) -> float | FloatArray:
        if derivative not in {0, 1, 2, 3}:
            raise ValueError("Derivative order must be between 0 and 3")

        scalar = np.ndim(value) == 0
        phi = np.asarray(value, dtype=float)
        count = len(self.values)
        if self.periodic:
            cycles = np.floor((phi - self.domain_start) / self.period)
            x = phi - cycles * self.period
            x = np.where(x < self.domain_start, x + self.period, x)
            x = np.where(x >= self.domain_start + self.period, x - self.period, x)
        else:
            cycles = np.zeros_like(phi)
            x = np.clip(phi, self.domain_start, self.domain_end)

        scaled = (x - self.domain_start) / self.step
        index = np.floor(scaled).astype(np.int64)
        t = scaled - index
        if not self.periodic:
            at_end = index >= count - 1
            index = np.where(at_end, count - 2, index)
            t = np.where(at_end, 1.0, t)
        index = np.clip(index, 0, count - 1)
        a0, a1, a2, a3, a4, a5 = (
            coefficient[index] for coefficient in self.coefficients
        )
        step = self.step

        if derivative == 0:
            result = a0 + t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
            result = result + cycles * self.cycle_delta
        elif derivative == 1:
            result = (
                a1 + t * (2.0 * a2 + t * (3.0 * a3 + t * (4.0 * a4 + t * 5.0 * a5)))
            ) / step
        elif derivative == 2:
            result = (2.0 * a2 + t * (6.0 * a3 + t * (12.0 * a4 + t * 20.0 * a5))) / (
                step * step
            )
        else:
            result = (6.0 * a3 + t * (24.0 * a4 + t * 60.0 * a5)) / (step**3)
        return float(result) if scalar else np.asarray(result, dtype=float)


class _IntegralTable:
    """Dense cumulative Simpson table with periodic continuation."""

    def __init__(
        self,
        function: Callable[[float | FloatArray], float | FloatArray],
        domain_start: float,
        domain_end: float,
        periodic: bool,
        interval_count: int = 32768,
    ) -> None:
        if not domain_start < domain_end:
            raise ValueError("Integral domain must have positive length")
        if interval_count % 2:
            interval_count += 1
        self.domain_start = domain_start
        self.domain_end = domain_end
        self.periodic = periodic
        self.x = np.linspace(domain_start, domain_end, interval_count + 1)
        values = np.asarray(function(self.x), dtype=float)
        self.prefix = cumulative_simpson(values, x=self.x, initial=0.0)
        self.domain_integral = float(self.prefix[-1])

    def _antiderivative(self, value: float | FloatArray) -> float | FloatArray:
        scalar = np.ndim(value) == 0
        values = np.asarray(value, dtype=float)
        length = self.domain_end - self.domain_start
        if self.periodic:
            cycles = np.floor((values - self.domain_start) / length)
            wrapped = values - cycles * length
            wrapped = np.where(wrapped < self.domain_start, wrapped + length, wrapped)
            wrapped = np.where(wrapped >= self.domain_end, wrapped - length, wrapped)
        else:
            tolerance = 1e-10
            if np.any(values < self.domain_start - tolerance) or np.any(
                values > self.domain_end + tolerance
            ):
                raise ValueError("Integral query is outside the motion domain")
            cycles = np.zeros_like(values)
            wrapped = np.clip(values, self.domain_start, self.domain_end)
        result = cycles * self.domain_integral + np.interp(wrapped, self.x, self.prefix)
        return float(result) if scalar else np.asarray(result, dtype=float)

    def integral(
        self, start: float | FloatArray, end: float | FloatArray
    ) -> float | FloatArray:
        return self._antiderivative(end) - self._antiderivative(start)


def _signed_area(points: FloatArray) -> float:
    vertices = points[:-1] if np.allclose(points[0], points[-1]) else points
    shifted = np.roll(vertices, -1, axis=0)
    return 0.5 * float(
        np.sum(vertices[:, 0] * shifted[:, 1] - vertices[:, 1] * shifted[:, 0])
    )


def _polygon_parts(geometry: Geometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    parts: list[Polygon] = []
    for child in getattr(geometry, "geoms", ()):
        parts.extend(_polygon_parts(child))
    return parts


def _clean_polygon(geometry: Geometry) -> Polygon:
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    parts = _polygon_parts(geometry)
    if not parts:
        raise RuntimeError("Geometry operation produced no polygonal body")
    return max(parts, key=lambda polygon: polygon.area)


def _outline(polygon: Polygon, tolerance: float) -> FloatArray:
    points = np.asarray(polygon.exterior.coords, dtype=float)
    if len(points) < 4:
        raise RuntimeError("Cutter sweep left fewer than three boundary points")

    # GEOS can retain nearly coincident and almost-collinear overlay vertices.
    unique = [points[0]]
    for point in points[1:-1]:
        if np.linalg.norm(point - unique[-1]) > tolerance * 0.05:
            unique.append(point)
    points = np.asarray(unique, dtype=float)

    changed = True
    while changed and len(points) > 3:
        changed = False
        keep = np.ones(len(points), dtype=bool)
        for index in range(len(points)):
            previous = points[(index - 1) % len(points)]
            current = points[index]
            following = points[(index + 1) % len(points)]
            a = current - previous
            b = following - current
            lengths = np.linalg.norm(a) + np.linalg.norm(b)
            twice_area = abs(float(a[0] * b[1] - a[1] * b[0]))
            if (
                float(np.dot(a, b)) >= 0.0
                and lengths > 0.0
                and twice_area / lengths < tolerance * 0.02
            ):
                keep[index] = False
                changed = True
        points = points[keep]
    if len(points) < 3:
        raise RuntimeError("Geometry cleanup removed the complete boundary")
    points = np.vstack((points, points[0]))
    if _signed_area(points) < 0.0:
        points = points[::-1].copy()
    return points


def _transform_outline(
    points: FloatArray, angle: float, translate_x: float = 0.0
) -> Polygon:
    vertices = points[:-1]
    cosine = math.cos(angle)
    sine = math.sin(angle)
    transformed = np.empty_like(vertices)
    transformed[:, 0] = cosine * vertices[:, 0] - sine * vertices[:, 1] + translate_x
    transformed[:, 1] = sine * vertices[:, 0] + cosine * vertices[:, 1]
    return Polygon(transformed)


def _open_sector(
    start_angle: float,
    end_angle: float,
    inner_radius: float,
    outer_radius: float,
) -> Polygon:
    span = end_angle - start_angle
    count = max(16, math.ceil(abs(span) * 160.0))
    angles = np.linspace(start_angle, end_angle, count + 1)
    outer = np.column_stack(
        (outer_radius * np.cos(angles), outer_radius * np.sin(angles))
    )
    inner = np.column_stack(
        (inner_radius * np.cos(angles), inner_radius * np.sin(angles))
    )
    return Polygon(np.vstack((outer, inner[::-1])))


def _read_sample_table(path: Path, columns: int) -> FloatArray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if values.ndim != 2 or values.shape[1] != columns:
        raise ValueError(f"Expected {columns} columns in {path}")
    if len(values) < 64 or not np.all(np.isfinite(values)):
        raise ValueError(f"Sample table {path} must contain at least 64 finite rows")
    return np.asarray(values, dtype=float)


def load_engine_config(
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
    extra_arguments: tuple[str, ...] | list[str],
) -> EngineConfig:
    if input_flag == "--transmission-csv":
        samples = _read_sample_table(input_path, 5)
        input_mode = "transmission"
        reference_center_distance = 0.0
    elif input_flag == "--centrode-csv":
        samples = _read_sample_table(input_path, 4)
        input_mode = "drive_centrode"
        reference_center_distance = 0.0
        arguments = list(extra_arguments)
        if "--centrode-center-distance" in arguments:
            index = arguments.index("--centrode-center-distance")
            reference_center_distance = float(arguments[index + 1])
    else:
        raise ValueError(f"Unsupported input mode: {input_flag}")
    return EngineConfig(
        name=name,
        description=description,
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
        input_mode=input_mode,
        samples=samples,
        reference_center_distance=reference_center_distance,
    )


def _materialize_centrode(config: EngineConfig) -> EngineConfig:
    if config.input_mode != "drive_centrode":
        return config
    radius_samples = config.samples[:, 1:]
    law = _QuinticSeries(
        radius_samples[:, 0],
        radius_samples[:, 1],
        radius_samples[:, 2],
        domain_start=config.domain_start,
        domain_end=config.domain_end,
        period=config.period,
        periodic=not config.open_,
    )
    check_phi = np.linspace(config.domain_start, config.domain_end, 8193)
    radii = np.asarray(law(check_phi), dtype=float)
    if np.any(radii <= 0.0) or not np.all(np.isfinite(radii)):
        raise ValueError("Centrode radius must stay positive and finite")
    maximum_radius = float(np.max(radii))
    integration_start = config.active_start if config.open_ else config.domain_start
    integration_end = (
        config.active_end if config.open_ else config.domain_start + config.period
    )

    integration_phi = np.linspace(integration_start, integration_end, 32769)
    integration_radii = np.asarray(law(integration_phi), dtype=float)

    def cycle_advance(center: float) -> float:
        return float(
            simpson(integration_radii / (center - integration_radii), x=integration_phi)
        )

    center = config.reference_center_distance
    if center == 0.0:
        lower = maximum_radius * (1.0 + 1e-10)
        upper = max(2.0 * maximum_radius, lower + 1.0)
        while cycle_advance(upper) > config.cycle_delta:
            upper *= 2.0
            if not math.isfinite(upper):
                raise ValueError("Could not solve a finite centrode center distance")
        for _ in range(80):
            midpoint = 0.5 * (lower + upper)
            if cycle_advance(midpoint) > config.cycle_delta:
                lower = midpoint
            else:
                upper = midpoint
        center = 0.5 * (lower + upper)
    if not center > maximum_radius or not math.isfinite(center):
        raise ValueError(
            "Centrode reference center distance must exceed its maximum radius"
        )

    sample_count = len(config.samples)
    sample_phi = (
        np.linspace(
            config.domain_start, config.domain_end, sample_count, endpoint=False
        )
        if not config.open_
        else np.linspace(config.domain_start, config.domain_end, sample_count)
    )
    radius = np.asarray(law(sample_phi, 0), dtype=float)
    radius1 = np.asarray(law(sample_phi, 1), dtype=float)
    radius2 = np.asarray(law(sample_phi, 2), dtype=float)
    gap = center - radius
    psi1 = radius / gap
    psi2 = center * radius1 / gap**2
    psi3 = center * (radius2 / gap**2 + 2.0 * radius1**2 / gap**3)
    ratio_integral = _IntegralTable(
        lambda phi: (
            np.asarray(law(phi), dtype=float)
            / (center - np.asarray(law(phi), dtype=float))
        ),
        config.domain_start,
        config.domain_end,
        not config.open_,
        max(4096, sample_count),
    )
    origin = config.active_start if config.open_ else config.domain_start
    psi = np.asarray(ratio_integral.integral(origin, sample_phi), dtype=float)
    cycle_delta = float(
        ratio_integral.integral(
            origin, config.active_end if config.open_ else origin + config.period
        )
    )
    transmission = np.column_stack((sample_phi, psi, psi1, psi2, psi3))
    return replace(
        config,
        samples=transmission,
        cycle_delta=cycle_delta,
        reference_center_distance=center,
    )


class _GearGenerator:
    def __init__(self, raw_config: EngineConfig) -> None:
        self.config = _materialize_centrode(raw_config)
        config = self.config
        self.closed = not config.open_
        self.alpha = math.radians(config.pressure_angle_deg)
        self.addendum = config.addendum_factor * config.module
        self.dedendum = config.dedendum_factor * config.module
        self.fillet_radius = config.fillet_factor * config.module
        self.motion = _QuinticSeries(
            config.samples[:, 1],
            config.samples[:, 2],
            config.samples[:, 3],
            domain_start=config.domain_start,
            domain_end=config.domain_end,
            period=config.period,
            periodic=self.closed,
            cycle_delta=config.cycle_delta,
        )
        self._validate()
        self.drive_teeth = config.teeth
        self.active_start = config.domain_start if self.closed else config.active_start
        self.active_end = (
            config.domain_start + config.period if self.closed else config.active_end
        )
        self.arc_integral = _IntegralTable(
            self._arc_density,
            config.domain_start,
            config.domain_end,
            self.closed,
        )
        self.total_integral = float(
            self.arc_integral.integral(self.active_start, self.active_end)
        )
        self.center_distance = (
            self.drive_teeth * math.pi * config.module / self.total_integral
        )
        if self.closed:
            exact_driven_teeth = self.drive_teeth * config.period / config.cycle_delta
            self.driven_teeth = round(exact_driven_teeth)
            if self.driven_teeth <= 0 or not math.isclose(
                exact_driven_teeth, self.driven_teeth, abs_tol=1e-7
            ):
                raise ValueError(
                    "Closed transmission requires an integral driven tooth count"
                )
            self.drive_cycle = config.period
            self.driven_cycle = config.period * self.driven_teeth / self.drive_teeth
            self._validate_closed_motion()
        else:
            self.driven_teeth = self.drive_teeth
            self.drive_cycle = self.active_end - self.active_start
            self.driven_cycle = self.drive_cycle
        self.average_ratio = float(
            (self._psi(self.active_end) - self._psi(self.active_start))
            / (self.active_end - self.active_start)
        )
        self.curvature_limit = math.sin(self.alpha) ** 2 / (
            self.dedendum - self.fillet_radius * (1.0 - math.sin(self.alpha))
        )
        self._measure_centrodes()
        self._rack_margin = 2e-4 * config.module
        self._rack_tooth_template = self._make_tooth_template(self._rack_margin)

    def _psi(self, phi: float | FloatArray, derivative: int = 0) -> float | FloatArray:
        return self.motion(phi, derivative)

    def _validate(self) -> None:
        config = self.config
        if config.teeth < 6 or config.module <= 0.0:
            raise ValueError("Invalid tooth count or module")
        if not 0.0 < self.alpha < 0.45 * math.pi:
            raise ValueError("Pressure angle must be between 0 and 81 degrees")
        if not (
            self.addendum > 0.0
            and self.dedendum > self.fillet_radius
            and self.fillet_radius >= 0.0
        ):
            raise ValueError("Invalid cutter dimensions")
        pitch = math.pi * config.module
        if 0.25 * pitch - self.dedendum * math.tan(self.alpha) <= 0.03 * config.module:
            raise ValueError(
                "Rack cutter tip is too narrow; reduce dedendum or pressure angle"
            )
        check_phi = np.linspace(config.domain_start, config.domain_end, 16385)
        ratios = np.asarray(self._psi(check_phi, 1), dtype=float)
        if (
            not np.all(np.isfinite(ratios))
            or np.any(ratios <= 1e-7)
            or np.any(ratios >= 1e7)
        ):
            raise ValueError("Motion law is not a bounded orientation-preserving map")

    def _validate_closed_motion(self) -> None:
        phi = self.active_start + self.config.period * np.arange(4096) / 4096.0
        shifted = phi + self.driven_cycle
        scale = max(1.0, abs(self.config.cycle_delta))
        if (
            np.max(np.abs((self._psi(shifted) - self._psi(phi)) - 2.0 * math.pi))
            > 3e-5 * scale
            or np.max(np.abs(self._psi(shifted, 1) - self._psi(phi, 1))) > 3e-5
            or np.max(np.abs(self._psi(shifted, 2) - self._psi(phi, 2))) > 1e-4
        ):
            raise ValueError(
                "Closed motion is not compatible with one driven-gear revolution"
            )

    def _w(self, phi: float | FloatArray) -> float | FloatArray:
        first = self._psi(phi, 1)
        return np.hypot(self._psi(phi, 2), first * (1.0 + first))

    def _arc_density(self, phi: float | FloatArray) -> float | FloatArray:
        first = self._psi(phi, 1)
        return self._w(phi) / (1.0 + first) ** 2

    def _drive_radius(self, phi: float | FloatArray) -> float | FloatArray:
        first = self._psi(phi, 1)
        return self.center_distance * first / (1.0 + first)

    def _driven_radius(self, phi: float | FloatArray) -> float | FloatArray:
        return self.center_distance / (1.0 + self._psi(phi, 1))

    def _drive_centrode(
        self, phi: float | FloatArray
    ) -> complex | NDArray[np.complex128]:
        values = np.asarray(phi, dtype=float)
        result = np.asarray(self._drive_radius(values), dtype=float) * np.exp(
            -1j * values
        )
        return complex(result) if result.ndim == 0 else result

    def _driven_centrode(
        self, phi: float | FloatArray
    ) -> complex | NDArray[np.complex128]:
        values = np.asarray(phi, dtype=float)
        result = -np.asarray(self._driven_radius(values), dtype=float) * np.exp(
            1j * np.asarray(self._psi(values), dtype=float)
        )
        return complex(result) if result.ndim == 0 else result

    def _drive_tangent(
        self, phi: float | FloatArray
    ) -> complex | NDArray[np.complex128]:
        values = np.asarray(phi, dtype=float)
        first = np.asarray(self._psi(values, 1), dtype=float)
        second = np.asarray(self._psi(values, 2), dtype=float)
        base = (second - 1j * first * (1.0 + first)) / np.asarray(
            self._w(values), dtype=float
        )
        result = base * np.exp(-1j * values)
        return complex(result) if result.ndim == 0 else result

    def _driven_tangent(
        self, phi: float | FloatArray
    ) -> complex | NDArray[np.complex128]:
        values = np.asarray(phi, dtype=float)
        first = np.asarray(self._psi(values, 1), dtype=float)
        second = np.asarray(self._psi(values, 2), dtype=float)
        base = (second - 1j * first * (1.0 + first)) / np.asarray(
            self._w(values), dtype=float
        )
        result = base * np.exp(1j * np.asarray(self._psi(values), dtype=float))
        return complex(result) if result.ndim == 0 else result

    def _rack_pose(self, phi: float, driven: bool) -> tuple[complex, complex]:
        if driven:
            pitch_point = complex(self._driven_centrode(phi))
            tangent = complex(self._driven_tangent(phi))
        else:
            pitch_point = complex(self._drive_centrode(phi))
            tangent = complex(self._drive_tangent(phi))
        return pitch_point, tangent

    def _drive_curvature(self, phi: FloatArray) -> FloatArray:
        first = np.asarray(self._psi(phi, 1))
        second = np.asarray(self._psi(phi, 2))
        w = np.asarray(self._w(phi))
        h = (
            (1.0 + first)
            * (first * (self._psi(phi, 3) - first - first**2) - 2.0 * second**2)
            / w**2
        )
        return (1.0 + first) ** 2 * h / (self.center_distance * w)

    def _driven_curvature(self, phi: FloatArray) -> FloatArray:
        first = np.asarray(self._psi(phi, 1))
        second = np.asarray(self._psi(phi, 2))
        w = np.asarray(self._w(phi))
        h = (
            (1.0 + first)
            * (first * (self._psi(phi, 3) + first**2 + first**3) - second**2)
            / w**2
        )
        return (1.0 + first) ** 2 * h / (self.center_distance * w)

    def _measure_centrodes(self) -> None:
        span = (
            max(self.drive_cycle, self.driven_cycle)
            if self.closed
            else self.config.domain_end - self.config.domain_start
        )
        start = self.active_start if self.closed else self.config.domain_start
        phi = np.linspace(start, start + span, 16385)
        drive_curvature = self._drive_curvature(phi)
        driven_curvature = self._driven_curvature(phi)
        drive_radius = np.asarray(self._drive_radius(phi))
        driven_radius = np.asarray(self._driven_radius(phi))
        self.maximum_drive_curvature = float(np.max(drive_curvature))
        self.minimum_driven_curvature = float(np.min(driven_curvature))
        self.maximum_pitch_radius = float(np.max([drive_radius, driven_radius]))
        self.minimum_pitch_radius = float(np.min([drive_radius, driven_radius]))
        self.centrodes_are_convex = (
            self.maximum_drive_curvature <= 1e-9
            and self.minimum_driven_curvature >= -1e-9
        )

    def _involute_tooth_template(self, margin: float) -> NDArray[np.complex128]:
        pitch = math.pi * self.config.module
        tangent = math.tan(self.alpha)
        root_half_width = 0.25 * pitch + self.addendum * tangent
        sharp_tip_half_width = 0.25 * pitch - self.dedendum * tangent
        maximum_radius = (
            0.9
            * sharp_tip_half_width
            * math.cos(self.alpha)
            / (1.0 - math.sin(self.alpha))
        )
        radius = min(max(self.fillet_radius, 0.0), maximum_radius)
        y_shift = -margin
        points = [
            complex(-0.5 * pitch, self.addendum + y_shift),
            complex(-root_half_width, self.addendum + y_shift),
        ]
        if radius <= 1e-12:
            points.extend(
                [
                    complex(-sharp_tip_half_width, -self.dedendum + y_shift),
                    complex(sharp_tip_half_width, -self.dedendum + y_shift),
                ]
            )
        else:
            transition = radius * (1.0 - math.sin(self.alpha)) / math.cos(self.alpha)
            left_center = complex(
                -sharp_tip_half_width + transition,
                -self.dedendum + radius + y_shift,
            )
            left_angles = np.linspace(math.pi + self.alpha, 1.5 * math.pi, 7)
            points.extend(left_center + radius * np.exp(1j * left_angles))
            right_center = complex(
                sharp_tip_half_width - transition,
                -self.dedendum + radius + y_shift,
            )
            points.append(complex(right_center.real, -self.dedendum + y_shift))
            right_angles = np.linspace(-0.5 * math.pi, -self.alpha, 7)
            points.extend(right_center + radius * np.exp(1j * right_angles))
        points.extend(
            [
                complex(root_half_width, self.addendum + y_shift),
                complex(0.5 * pitch, self.addendum + y_shift),
            ]
        )
        return np.asarray(points, dtype=np.complex128)

    def _cycloidal_tooth_template(self, margin: float) -> NDArray[np.complex128]:
        pitch = math.pi * self.config.module
        root_half_width = 0.25 * pitch + self.addendum * math.tan(self.alpha)
        tip_half_width = 0.25 * pitch - self.dedendum * math.tan(self.alpha)
        blend = min(max(self.config.cycloidal_rolling_factor, 0.0), 1.0)
        y_shift = -margin
        q = np.arange(15, dtype=float) / 14.0
        tau = math.pi * q
        x_fraction = (1.0 - blend) * q + blend * ((tau - np.sin(tau)) / math.pi)
        y_fraction = (1.0 - blend) * q + blend * (0.5 * (1.0 - np.cos(tau)))
        x = root_half_width + (tip_half_width - root_half_width) * x_fraction
        y = self.addendum + (-self.dedendum - self.addendum) * y_fraction + y_shift
        points = np.concatenate(
            (
                np.asarray(
                    [
                        complex(-0.5 * pitch, self.addendum + y_shift),
                        complex(-root_half_width, self.addendum + y_shift),
                    ]
                ),
                -x[1:] + 1j * y[1:],
                np.asarray([complex(tip_half_width, -self.dedendum + y_shift)]),
                x[:14][::-1] + 1j * y[:14][::-1],
                np.asarray([complex(0.5 * pitch, self.addendum + y_shift)]),
            )
        )
        return np.asarray(points, dtype=np.complex128)

    def _make_tooth_template(self, margin: float) -> NDArray[np.complex128]:
        if self.config.profile == "cycloidal":
            return self._cycloidal_tooth_template(margin)
        return self._involute_tooth_template(margin)

    def _make_rack(
        self,
        common_arc: float,
        phase_offset: float,
        half_width: float,
        top: float,
        margin: float,
    ) -> NDArray[np.complex128]:
        pitch = math.pi * self.config.module
        first_center = (
            phase_offset
            - common_arc
            + math.floor((-half_width - phase_offset + common_arc) / pitch) * pitch
        )
        center_start = first_center - 0.5 * pitch
        center_count = math.floor((half_width + pitch - center_start) / pitch) + 1
        centers = center_start + pitch * np.arange(center_count, dtype=float)
        template = (
            self._rack_tooth_template
            if margin == self._rack_margin
            else self._make_tooth_template(margin)
        )
        teeth = (centers[:, None] + template[None, :]).reshape(-1)
        boundary = np.empty(len(teeth) + 3, dtype=np.complex128)
        boundary[0] = complex(first_center - pitch, self.addendum - margin)
        boundary[1:-2] = teeth
        boundary[-2] = complex(teeth[-1].real, top)
        boundary[-1] = complex(boundary[0].real, top)
        return boundary

    @staticmethod
    def _rack_polygon(
        local: NDArray[np.complex128],
        pitch_point: complex,
        tangent: complex,
        outward_normal: complex,
    ) -> Polygon:
        keep = np.concatenate(([True], np.abs(np.diff(local)) > 1e-12))
        unique = local[keep]
        transformed = pitch_point + unique.real * tangent + unique.imag * outward_normal
        polygon = Polygon(np.column_stack((transformed.real, transformed.imag)))
        if not polygon.is_valid:
            polygon = _clean_polygon(make_valid(polygon))
        return polygon

    def _sweep_interval(self, driven: bool) -> tuple[float, float]:
        open_padding = 2.5 * (self.active_end - self.active_start) / self.drive_teeth
        start = (
            self.active_start
            if self.closed
            else max(self.config.domain_start, self.active_start - open_padding)
        )
        end = (
            self.active_start + (self.driven_cycle if driven else self.drive_cycle)
            if self.closed
            else min(self.config.domain_end, self.active_end + open_padding)
        )
        return start, end

    def _generate_swept_gear(
        self, driven: bool, samples_per_radian: int
    ) -> tuple[FloatArray, int]:
        cycle_start, cycle_end = self._sweep_interval(driven)
        teeth = self.driven_teeth if driven else self.drive_teeth
        phase_count = max(
            24 * teeth,
            math.ceil(abs(cycle_end - cycle_start) * samples_per_radian),
        )
        pitch = math.pi * self.config.module
        blank_radius = (
            self.maximum_pitch_radius + self.addendum + 2.5 * self.config.module
        )
        rack_half_width = 2.4 * blank_radius + 2.0 * pitch
        rack_top = 2.5 * blank_radius + pitch
        sweep_margin = self._rack_margin
        cutters: list[Polygon] = []
        for phi in np.linspace(cycle_start, cycle_end, phase_count, endpoint=False):
            phi = float(phi)
            common_arc = self.center_distance * float(
                self.arc_integral.integral(self.active_start, phi)
            )
            pitch_point, tangent = self._rack_pose(phi, driven)
            outward_normal = (-1j if driven else 1j) * tangent
            rack = self._make_rack(
                common_arc,
                0.0 if driven else 0.5 * pitch,
                rack_half_width,
                rack_top,
                sweep_margin,
            )
            cutters.append(
                self._rack_polygon(rack, pitch_point, tangent, outward_normal)
            )
        swept_cutters = union_all(cutters)
        blank = Point(0.0, 0.0).buffer(blank_radius, quad_segs=256)
        gear: Geometry = blank.difference(swept_cutters)
        if not self.closed:
            if driven:
                start_angle = float(self._psi(self.config.active_start)) + math.pi
                end_angle = float(self._psi(self.config.active_end)) + math.pi
            else:
                start_angle = -self.config.active_start
                end_angle = -self.config.active_end
            if abs(end_angle - start_angle) >= 2.0 * math.pi - 1e-8:
                raise ValueError(
                    "Open gear body span must be less than one body revolution"
                )
            inner_radius = max(
                0.08 * self.config.module, 0.22 * self.minimum_pitch_radius
            )
            gear = gear.intersection(
                _open_sector(start_angle, end_angle, inner_radius, blank_radius)
            )
        polygon = _clean_polygon(gear)
        return (
            _outline(polygon, 2e-7 * max(1.0, self.config.module)),
            phase_count,
        )

    def _generate_conjugate_mate(
        self, master: FloatArray, samples_per_radian: int
    ) -> tuple[FloatArray, int]:
        cycle_start, cycle_end = self._sweep_interval(True)
        phase_count = max(
            24 * self.driven_teeth,
            math.ceil(abs(cycle_end - cycle_start) * samples_per_radian),
        )
        blank_radius = (
            self.maximum_pitch_radius + self.addendum + 2.5 * self.config.module
        )
        cutter_tolerance = 2e-4 * max(1.0, self.config.module)
        master_polygon = _transform_outline(master, 0.0).simplify(
            cutter_tolerance, preserve_topology=True
        )
        # Rigid cutter poses are sampled rather than continuous.  Expanding the
        # conjugate cutter by the same tolerance used to simplify its outline
        # makes this approximation conservative and prevents phase-grid
        # coincidences from leaving tiny islands of interference.
        conservative_offset = max(cutter_tolerance, 1.75e-3 * self.config.module)
        master_polygon = master_polygon.buffer(
            conservative_offset, quad_segs=2, join_style="mitre"
        )
        simplified = _outline(_clean_polygon(master_polygon), 1e-10)
        points = simplified[:-1, 0] + 1j * simplified[:-1, 1]
        cutters: list[Polygon] = []
        psi_start = float(self._psi(self.active_start))
        for phi in np.linspace(cycle_start, cycle_end, phase_count, endpoint=False):
            drive_delta = float(phi) - self.active_start
            driven_delta = float(self._psi(phi)) - psi_start
            rotation = np.exp(1j * (drive_delta + driven_delta))
            translation = -self.center_distance * np.exp(1j * driven_delta)
            transformed = translation + rotation * points
            cutters.append(
                Polygon(np.column_stack((transformed.real, transformed.imag)))
            )
        swept_cutters = union_all(cutters)
        blank = Point(0.0, 0.0).buffer(blank_radius, quad_segs=256)
        gear: Geometry = blank.difference(swept_cutters)
        if not self.closed:
            inner_radius = max(
                0.08 * self.config.module, 0.22 * self.minimum_pitch_radius
            )
            gear = gear.intersection(
                _open_sector(
                    float(self._psi(self.config.active_start)) + math.pi,
                    float(self._psi(self.config.active_end)) + math.pi,
                    inner_radius,
                    blank_radius,
                )
            )
        return (
            _outline(_clean_polygon(gear), 2e-7 * max(1.0, self.config.module)),
            phase_count,
        )

    def _phi_from_common_arc(self, common_arc: FloatArray) -> FloatArray:
        """Invert pitch-curve arc using the dense monotone integration table."""

        values = np.asarray(common_arc, dtype=float)
        cycle_arc = self.center_distance * self.arc_integral.domain_integral
        if not self.closed:
            table_arc = self.center_distance * (
                self.arc_integral.prefix
                - float(self.arc_integral._antiderivative(self.active_start))
            )
            tolerance = 64.0 * np.finfo(float).eps * max(
                1.0, abs(float(table_arc[0])), abs(float(table_arc[-1]))
            )
            if np.any(values < table_arc[0] - tolerance) or np.any(
                values > table_arc[-1] + tolerance
            ):
                raise ValueError(
                    "Analytic gear curve extends outside the padded motion domain"
                )
            return np.interp(values, table_arc, self.arc_integral.x)

        cycles = np.floor(values / cycle_arc)
        remainder = values - cycles * cycle_arc
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(cycle_arc))
        at_cycle_end = np.abs(remainder - cycle_arc) <= tolerance
        cycles = np.where(at_cycle_end, cycles + 1.0, cycles)
        remainder = np.where(at_cycle_end, 0.0, remainder)
        base_phi = np.interp(
            remainder,
            self.center_distance * self.arc_integral.prefix,
            self.arc_integral.x,
        )
        return base_phi + cycles * self.config.period

    def _analytic_involute_points(
        self,
        *,
        driven: bool,
        tooth_phase: float,
        sign: int,
        common_arc: FloatArray,
    ) -> NDArray[np.complex128]:
        pitch = math.pi * self.config.module
        values = np.asarray(common_arc, dtype=float)
        phi = self._phi_from_common_arc(values)
        centrode = (
            np.asarray(self._driven_centrode(phi), dtype=np.complex128)
            if driven
            else np.asarray(self._drive_centrode(phi), dtype=np.complex128)
        )
        tangent = (
            np.asarray(self._driven_tangent(phi), dtype=np.complex128)
            if driven
            else np.asarray(self._drive_tangent(phi), dtype=np.complex128)
        )
        rack_coordinate = (
            sign * 0.25 * pitch - (values - tooth_phase * pitch)
        )
        direction = complex(math.cos(self.alpha), sign * math.sin(self.alpha))
        return (
            centrode
            + rack_coordinate * tangent * direction * math.cos(self.alpha)
        )

    def _analytic_offset_points(
        self,
        *,
        driven: bool,
        common_arc: FloatArray,
        height: float,
    ) -> NDArray[np.complex128]:
        values = np.asarray(common_arc, dtype=float)
        phi = self._phi_from_common_arc(values)
        centrode = (
            np.asarray(self._driven_centrode(phi), dtype=np.complex128)
            if driven
            else np.asarray(self._drive_centrode(phi), dtype=np.complex128)
        )
        tangent = (
            np.asarray(self._driven_tangent(phi), dtype=np.complex128)
            if driven
            else np.asarray(self._drive_tangent(phi), dtype=np.complex128)
        )
        outward_normal = (-1j if driven else 1j) * tangent
        return centrode + height * outward_normal

    def _analytic_fillet_points(
        self,
        *,
        driven: bool,
        tooth_phase: float,
        sign: int,
        common_arc: FloatArray,
    ) -> NDArray[np.complex128]:
        """Evaluate the exact rounded rack-tip envelope.

        The circle-envelope branch is chosen continuously from the exact
        dedendum contact.  For the convex orientation supported by the
        historical engine this reduces algebraically to its oriented-normal
        formula; unlike a curvature-sign switch, it stays continuous through
        nonconvex centrode inflections.
        """

        pitch = math.pi * self.config.module
        values = np.asarray(common_arc, dtype=float)
        phi = self._phi_from_common_arc(values)
        centrode = (
            np.asarray(self._driven_centrode(phi), dtype=np.complex128)
            if driven
            else np.asarray(self._drive_centrode(phi), dtype=np.complex128)
        )
        tangent = (
            np.asarray(self._driven_tangent(phi), dtype=np.complex128)
            if driven
            else np.asarray(self._drive_tangent(phi), dtype=np.complex128)
        )
        rack_coordinate = sign * 0.25 * pitch - (
            values - tooth_phase * pitch
        )
        root_height = self.dedendum - self.fillet_radius
        if driven:
            offset = (
                rack_coordinate
                - sign * self.fillet_radius / math.cos(self.alpha)
                - sign * root_height * math.tan(self.alpha)
                + 1j * root_height
            )
        else:
            offset = (
                rack_coordinate
                + sign * self.fillet_radius / math.cos(self.alpha)
                + sign * root_height * math.tan(self.alpha)
                - 1j * root_height
            )
        magnitude = np.abs(offset)
        if np.any(magnitude <= np.finfo(float).tiny):
            raise RuntimeError("Rounded rack-tip envelope has an undefined normal")
        midpoint = centrode + offset * tangent
        return midpoint + self.fillet_radius * offset / magnitude * tangent

    @staticmethod
    def _line_parameter(
        line: LineString, point: Point, parameters: FloatArray
    ) -> float:
        coordinates = np.asarray(line.coords, dtype=float)
        lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        return float(
            np.interp(
                line.project(point),
                cumulative,
                np.asarray(parameters, dtype=float),
            )
        )

    @staticmethod
    def _intersection_points(geometry: Geometry) -> list[Point]:
        if geometry.is_empty:
            return []
        if isinstance(geometry, Point):
            return [geometry]
        points: list[Point] = []
        for child in getattr(geometry, "geoms", ()):
            points.extend(_GearGenerator._intersection_points(child))
        return points

    def _analytic_flank_offset_intersection(
        self,
        *,
        driven: bool,
        tooth_phase: float,
        sign: int,
        height: float,
    ) -> tuple[float, float, float]:
        """Intersect an analytic flank with a parallel centrode offset.

        GEOS supplies robust candidate intersections of densely sampled
        LineStrings. SciPy then refines each two-parameter candidate against
        the analytic curves.
        """

        pitch = math.pi * self.config.module
        sine_cosine = math.sin(self.alpha) * math.cos(self.alpha)
        outward_orientation = -1 if driven else 1
        target_lambda = height / (outward_orientation * sign * sine_cosine)
        expected_flank_arc = (
            tooth_phase * pitch + sign * 0.25 * pitch - target_lambda
        )
        domain_low, domain_high = self._analytic_common_arc_bounds()
        flank_low = max(domain_low, expected_flank_arc - 1.25 * pitch)
        flank_high = min(domain_high, expected_flank_arc + 1.25 * pitch)
        offset_low = max(domain_low, tooth_phase * pitch - 1.75 * pitch)
        offset_high = min(domain_high, tooth_phase * pitch + 1.75 * pitch)
        flank_arc = np.linspace(
            flank_low,
            flank_high,
            257,
        )
        offset_arc = np.linspace(
            offset_low,
            offset_high,
            385,
        )
        flank_points = self._analytic_involute_points(
            driven=driven,
            tooth_phase=tooth_phase,
            sign=sign,
            common_arc=flank_arc,
        )
        offset_points = self._analytic_offset_points(
            driven=driven,
            common_arc=offset_arc,
            height=height,
        )
        flank_line = LineString(
            np.column_stack((flank_points.real, flank_points.imag))
        )
        offset_line = LineString(
            np.column_stack((offset_points.real, offset_points.imag))
        )
        candidate_points = self._intersection_points(
            flank_line.intersection(offset_line)
        )
        if not candidate_points:
            candidate_points = [nearest_points(flank_line, offset_line)[0]]

        lower = np.asarray([flank_arc[0], offset_arc[0]], dtype=float)
        upper = np.asarray([flank_arc[-1], offset_arc[-1]], dtype=float)
        candidates: list[tuple[float, float, float]] = []
        for point in candidate_points:
            initial = np.asarray(
                [
                    self._line_parameter(flank_line, point, flank_arc),
                    self._line_parameter(offset_line, point, offset_arc),
                ]
            )

            def residual(parameters: FloatArray) -> FloatArray:
                flank = self._analytic_involute_points(
                    driven=driven,
                    tooth_phase=tooth_phase,
                    sign=sign,
                    common_arc=np.asarray([parameters[0]]),
                )[0]
                offset = self._analytic_offset_points(
                    driven=driven,
                    common_arc=np.asarray([parameters[1]]),
                    height=height,
                )[0]
                difference = (flank - offset) / self.config.module
                return np.asarray([difference.real, difference.imag])

            solution = least_squares(
                residual,
                initial,
                bounds=(lower, upper),
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
                max_nfev=100,
            )
            geometric_residual = float(np.linalg.norm(residual(solution.x)))
            if solution.success and geometric_residual <= 2e-7:
                candidates.append(
                    (
                        float(solution.x[0]),
                        float(solution.x[1]),
                        geometric_residual * self.config.module,
                    )
                )
        if not candidates:
            member = "driven" if driven else "drive"
            boundary = "addendum" if height > 0.0 else "root"
            raise RuntimeError(
                f"Could not intersect {member} analytic flank with its "
                f"{boundary} boundary for tooth {tooth_phase}, sign {sign}"
            )
        return min(
            candidates,
            key=lambda candidate: (
                abs(candidate[0] - expected_flank_arc)
                + 0.25 * abs(candidate[1] - tooth_phase * pitch)
            ),
        )

    def _analytic_common_arc_bounds(self) -> tuple[float, float]:
        if self.closed:
            return -math.inf, math.inf
        return (
            self.center_distance
            * float(
                self.arc_integral.integral(
                    self.active_start, self.config.domain_start
                )
            ),
            self.center_distance
            * float(
                self.arc_integral.integral(
                    self.active_start, self.config.domain_end
                )
            ),
        )

    def _analytic_singular_arc(
        self,
        *,
        driven: bool,
        tooth_phase: float,
        sign: int,
        contact_arc: float,
        flank_tip_arc: float,
    ) -> float | None:
        """Locate the first cusp on the retained working-flank interval."""

        pitch = math.pi * self.config.module
        phase_arc = tooth_phase * pitch
        low = min(contact_arc, flank_tip_arc)
        high = max(contact_arc, flank_tip_arc)

        def equation(common_arc: float) -> float:
            phi = self._phi_from_common_arc(np.asarray([common_arc]))[0]
            curvature = float(
                self._driven_curvature(np.asarray([phi]))[0]
                if driven
                else self._drive_curvature(np.asarray([phi]))[0]
            )
            rack_coordinate = sign * 0.25 * pitch - (
                common_arc - phase_arc
            )
            return rack_coordinate * curvature - sign * math.tan(self.alpha)

        sample_count = 1024
        samples = np.linspace(low, high, sample_count + 1)
        phi = self._phi_from_common_arc(samples)
        curvature = (
            self._driven_curvature(phi)
            if driven
            else self._drive_curvature(phi)
        )
        rack_coordinate = sign * 0.25 * pitch - (samples - phase_arc)
        values = rack_coordinate * curvature - sign * math.tan(self.alpha)
        roots: list[float] = []
        for index in range(sample_count):
            lhs = float(values[index])
            rhs = float(values[index + 1])
            if not (math.isfinite(lhs) and math.isfinite(rhs)):
                continue
            if abs(lhs) <= 1e-12:
                roots.append(float(samples[index]))
            elif lhs * rhs < 0.0:
                roots.append(
                    float(
                        brentq(
                            equation,
                            float(samples[index]),
                            float(samples[index + 1]),
                            xtol=1e-13,
                            rtol=1e-13,
                        )
                    )
                )
        if math.isfinite(float(values[-1])) and abs(float(values[-1])) <= 1e-12:
            roots.append(float(samples[-1]))
        roots.sort()
        unique_roots: list[float] = []
        for root in roots:
            if not unique_roots or abs(root - unique_roots[-1]) > 1e-8:
                unique_roots.append(root)
        roots = unique_roots
        if not roots:
            return None
        return min(roots, key=lambda root: abs(root - contact_arc))

    def _analytic_curve_intersections(
        self,
        lhs: Callable[[FloatArray], NDArray[np.complex128]],
        lhs_bounds: tuple[float, float],
        rhs: Callable[[FloatArray], NDArray[np.complex128]],
        rhs_bounds: tuple[float, float],
        *,
        sample_count: int = 513,
    ) -> list[tuple[float, float, float]]:
        """Find curve intersections with GEOS and refine them with SciPy."""

        lhs_parameters = np.linspace(*lhs_bounds, sample_count)
        rhs_parameters = np.linspace(*rhs_bounds, sample_count)
        lhs_points = lhs(lhs_parameters)
        rhs_points = rhs(rhs_parameters)
        lhs_line = LineString(
            np.column_stack((lhs_points.real, lhs_points.imag))
        )
        rhs_line = LineString(
            np.column_stack((rhs_points.real, rhs_points.imag))
        )
        candidates: list[tuple[float, float]] = []
        for point in self._intersection_points(lhs_line.intersection(rhs_line)):
            candidates.append(
                (
                    self._line_parameter(lhs_line, point, lhs_parameters),
                    self._line_parameter(rhs_line, point, rhs_parameters),
                )
            )
        lhs_nearest, rhs_nearest = nearest_points(lhs_line, rhs_line)
        candidates.append(
            (
                self._line_parameter(lhs_line, lhs_nearest, lhs_parameters),
                self._line_parameter(rhs_line, rhs_nearest, rhs_parameters),
            )
        )

        lower = np.asarray(
            [min(lhs_bounds), min(rhs_bounds)], dtype=float
        )
        upper = np.asarray(
            [max(lhs_bounds), max(rhs_bounds)], dtype=float
        )
        refined: list[tuple[float, float, float]] = []
        for initial in candidates:
            def residual(parameters: FloatArray) -> FloatArray:
                difference = (
                    lhs(np.asarray([parameters[0]]))[0]
                    - rhs(np.asarray([parameters[1]]))[0]
                ) / self.config.module
                return np.asarray([difference.real, difference.imag])

            solution = least_squares(
                residual,
                np.asarray(initial),
                bounds=(lower, upper),
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
                max_nfev=150,
            )
            geometric_residual = (
                float(np.linalg.norm(residual(solution.x))) * self.config.module
            )
            if not solution.success or geometric_residual > 2e-7 * self.config.module:
                continue
            candidate = (
                float(solution.x[0]),
                float(solution.x[1]),
                geometric_residual,
            )
            if not any(
                abs(candidate[0] - existing[0]) < 1e-6
                and abs(candidate[1] - existing[1]) < 1e-6
                for existing in refined
            ):
                refined.append(candidate)
        return refined

    def _analytic_flank_geometry(
        self, *, driven: bool, tooth_phase: float, sign: int
    ) -> _AnalyticFlankGeometry:
        pitch = math.pi * self.config.module
        phase_arc = tooth_phase * pitch
        root_height = self.dedendum - self.fillet_radius
        fillet_dedendum_lambda = (
            root_height * math.tan(self.alpha)
            + self.fillet_radius / math.cos(self.alpha)
        )
        tangent_lambda = (
            root_height / math.sin(self.alpha) + self.fillet_radius
        ) / math.cos(self.alpha)
        target_orientation = 1 if driven else -1
        root_lambda = sign * target_orientation * fillet_dedendum_lambda
        contact_lambda = sign * target_orientation * tangent_lambda
        root_arc = phase_arc + sign * 0.25 * pitch - root_lambda
        contact_arc = phase_arc + sign * 0.25 * pitch - contact_lambda

        flank_tip_arc, addendum_tip_arc, tip_residual = (
            self._analytic_flank_offset_intersection(
                driven=driven,
                tooth_phase=tooth_phase,
                sign=sign,
                height=self.addendum,
            )
        )
        singular_arc = self._analytic_singular_arc(
            driven=driven,
            tooth_phase=tooth_phase,
            sign=sign,
            contact_arc=contact_arc,
            flank_tip_arc=flank_tip_arc,
        )
        if singular_arc is None:
            free = True
        else:
            singular_phi = self._phi_from_common_arc(
                np.asarray([singular_arc])
            )
            singular_curvature = float(
                self._driven_curvature(singular_phi)[0]
                if driven
                else self._drive_curvature(singular_phi)[0]
            )
            free = (
                singular_curvature <= self.curvature_limit + 1e-11
                if driven
                else -singular_curvature <= self.curvature_limit + 1e-11
            )
        transition_residual = 0.0
        flank_transition_arc = contact_arc
        fillet_transition_arc = contact_arc
        if free:
            flank_contact = self._analytic_involute_points(
                driven=driven,
                tooth_phase=tooth_phase,
                sign=sign,
                common_arc=np.asarray([contact_arc]),
            )[0]
            fillet_contact = self._analytic_fillet_points(
                driven=driven,
                tooth_phase=tooth_phase,
                sign=sign,
                common_arc=np.asarray([contact_arc]),
            )[0]
            transition_residual = abs(flank_contact - fillet_contact)
        else:
            if not driven and sign == -1:
                flank_extent = phase_arc
            elif not driven and sign == 1:
                flank_extent = phase_arc - pitch
            elif driven and sign == -1:
                flank_extent = phase_arc - 2.0 * pitch
            else:
                flank_extent = phase_arc + 2.0 * pitch
            flank_bounds = (
                min(contact_arc, flank_extent),
                max(contact_arc, flank_extent),
            )
            fillet_bounds = (
                min(root_arc, contact_arc),
                max(root_arc, contact_arc),
            )
            flank_curve = lambda values: self._analytic_involute_points(
                driven=driven,
                tooth_phase=tooth_phase,
                sign=sign,
                common_arc=values,
            )
            fillet_curve = lambda values: self._analytic_fillet_points(
                driven=driven,
                tooth_phase=tooth_phase,
                sign=sign,
                common_arc=values,
            )
            intersections = self._analytic_curve_intersections(
                flank_curve,
                flank_bounds,
                fillet_curve,
                fillet_bounds,
            )
            nontrivial = [
                intersection
                for intersection in intersections
                if abs(intersection[0] - contact_arc) > 1e-3 * pitch
                or abs(intersection[1] - contact_arc) > 1e-3 * pitch
            ]
            if not nontrivial:
                member = "driven" if driven else "drive"
                raise RuntimeError(
                    f"Could not resolve {member} undercut transition for tooth "
                    f"{tooth_phase:g}, sign {sign}"
                )
            selected = min(
                nontrivial,
                key=lambda intersection: abs(
                    intersection[0] - float(singular_arc)
                ),
            )
            flank_transition_arc = selected[0]
            fillet_transition_arc = selected[1]
            transition_residual = selected[2]

        return _AnalyticFlankGeometry(
            phase=tooth_phase,
            sign=sign,
            fillet_root_arc=root_arc,
            fillet_transition_arc=fillet_transition_arc,
            flank_transition_arc=flank_transition_arc,
            flank_tip_arc=flank_tip_arc,
            addendum_tip_arc=addendum_tip_arc,
            intersection_residual=max(transition_residual, tip_residual),
            undercut=not free,
        )

    def _sample_analytic_curve(
        self,
        function: Callable[[FloatArray], NDArray[np.complex128]],
        start: float,
        end: float,
        samples_per_radian: int,
        *,
        minimum_samples: int,
    ) -> tuple[NDArray[np.complex128], int, float]:
        endpoint_phi = self._phi_from_common_arc(
            np.asarray([start, end], dtype=float)
        )
        phi_span = abs(float(endpoint_phi[1] - endpoint_phi[0]))
        sample_count = max(
            minimum_samples,
            math.ceil(4.0 * phi_span * samples_per_radian),
        )
        target_error = (
            _ANALYTIC_CHORD_TOLERANCE_FACTOR * self.config.module
        )
        while True:
            parameters = np.linspace(start, end, sample_count + 1)
            points = function(parameters)
            midpoint_parameters = 0.5 * (
                parameters[:-1] + parameters[1:]
            )
            exact_midpoint = function(midpoint_parameters)
            segment_start = points[:-1]
            segment = points[1:] - segment_start
            denominator = np.maximum(
                np.abs(segment) ** 2, np.finfo(float).tiny
            )
            projection = np.clip(
                np.real(
                    (exact_midpoint - segment_start) * np.conj(segment)
                )
                / denominator,
                0.0,
                1.0,
            )
            chord_error = float(
                np.max(
                    np.abs(
                        exact_midpoint
                        - (segment_start + projection * segment)
                    )
                )
            )
            if chord_error <= target_error or sample_count >= 8192:
                return points, sample_count, chord_error
            sample_count *= 2

    def _sample_analytic_flank(
        self,
        *,
        driven: bool,
        geometry: _AnalyticFlankGeometry,
        samples_per_radian: int,
    ) -> tuple[NDArray[np.complex128], int, float, float, float]:
        """Sample and certify one exact generalized-involute working flank."""

        curve = lambda values: self._analytic_involute_points(
            driven=driven,
            tooth_phase=geometry.phase,
            sign=geometry.sign,
            common_arc=values,
        )
        points, sample_count, chord_error = self._sample_analytic_curve(
            curve,
            geometry.flank_transition_arc,
            geometry.flank_tip_arc,
            samples_per_radian,
            minimum_samples=128,
        )
        common_arc = np.linspace(
            geometry.flank_transition_arc,
            geometry.flank_tip_arc,
            sample_count + 1,
        )
        phi = self._phi_from_common_arc(common_arc)
        centrode = (
            np.asarray(self._driven_centrode(phi), dtype=np.complex128)
            if driven
            else np.asarray(self._drive_centrode(phi), dtype=np.complex128)
        )
        tangent = (
            np.asarray(self._driven_tangent(phi), dtype=np.complex128)
            if driven
            else np.asarray(self._drive_tangent(phi), dtype=np.complex128)
        )
        pitch = math.pi * self.config.module
        phase_arc = geometry.phase * pitch
        rack_coordinate = geometry.sign * 0.25 * pitch - (
            common_arc - phase_arc
        )
        direction = complex(
            math.cos(self.alpha), geometry.sign * math.sin(self.alpha)
        )
        local = (points - centrode) / tangent
        expected = rack_coordinate * direction * math.cos(self.alpha)
        envelope_residual = float(np.max(np.abs(local - expected)))

        midpoint_arc = 0.5 * (common_arc[:-1] + common_arc[1:])
        midpoint_phi = self._phi_from_common_arc(midpoint_arc)
        midpoint_tangent = (
            np.asarray(
                self._driven_tangent(midpoint_phi), dtype=np.complex128
            )
            if driven
            else np.asarray(
                self._drive_tangent(midpoint_phi), dtype=np.complex128
            )
        )
        midpoint_curvature = (
            self._driven_curvature(midpoint_phi)
            if driven
            else self._drive_curvature(midpoint_phi)
        )
        midpoint_lambda = geometry.sign * 0.25 * pitch - (
            midpoint_arc - phase_arc
        )
        derivative = midpoint_tangent * (
            1.0
            - math.cos(self.alpha) * direction
            + 1j
            * midpoint_curvature
            * midpoint_lambda
            * math.cos(self.alpha)
            * direction
        )
        generator_tangent = 1j * midpoint_tangent * direction
        regular = np.abs(derivative) > 1e-10
        if np.any(regular):
            normalized_generator = generator_tangent[regular] / np.abs(
                generator_tangent[regular]
            )
            normalized_derivative = derivative[regular] / np.abs(
                derivative[regular]
            )
            tangency_residual = float(
                np.max(
                    np.abs(
                        np.imag(
                            np.conj(normalized_generator)
                            * normalized_derivative
                        )
                    )
                )
            )
        else:
            tangency_residual = 0.0
        return (
            points,
            sample_count,
            envelope_residual,
            tangency_residual,
            chord_error,
        )

    def _analytic_root_blank(
        self, driven: bool, samples_per_radian: int
    ) -> tuple[Geometry, int, float]:
        """Build material inward of the exact dedendum offset."""

        teeth = self.driven_teeth if driven else self.drive_teeth
        if self.closed:
            start_arc = 0.0
            end_arc = teeth * math.pi * self.config.module
        else:
            start_arc, end_arc = self._analytic_common_arc_bounds()
        curve = lambda values: self._analytic_offset_points(
            driven=driven,
            common_arc=values,
            height=-self.dedendum,
        )
        root_points, sample_count, chord_error = self._sample_analytic_curve(
            curve,
            start_arc,
            end_arc,
            samples_per_radian,
            minimum_samples=max(512, 32 * teeth),
        )
        if self.closed:
            root = make_valid(
                Polygon(
                    np.column_stack((root_points.real, root_points.imag))
                )
            )
            hub_radius = max(
                0.08 * self.config.module,
                min(0.22 * self.minimum_pitch_radius, 0.5 * self.minimum_pitch_radius),
            )
            hub = Point(0.0, 0.0).buffer(hub_radius, quad_segs=128)
            return _clean_polygon(union_all([root, hub])), sample_count, chord_error

        common_arc = np.linspace(start_arc, end_arc, sample_count + 1)
        phi = self._phi_from_common_arc(common_arc)
        centrode = (
            np.asarray(self._driven_centrode(phi), dtype=np.complex128)
            if driven
            else np.asarray(self._drive_centrode(phi), dtype=np.complex128)
        )
        inner_radius = max(
            0.08 * self.config.module, 0.22 * self.minimum_pitch_radius
        )
        inner = centrode * inner_radius / np.abs(centrode)
        ring = np.concatenate((root_points, inner[::-1]))
        root = make_valid(
            Polygon(np.column_stack((ring.real, ring.imag)))
        )
        return root, sample_count, chord_error

    def _generate_analytic_involute_gear(
        self, driven: bool, samples_per_radian: int
    ) -> _AnalyticGearResult:
        """Arrange exact flank, rack-tip, addendum, and dedendum curves."""

        teeth = self.driven_teeth if driven else self.drive_teeth
        phases: list[float] = (
            [float(phase) for phase in range(teeth + 1)]
            if self.closed
            else [phase + 0.5 for phase in range(-1, teeth + 1)]
        )
        geometries: dict[tuple[float, int], _AnalyticFlankGeometry] = {}
        fillets: dict[tuple[float, int], NDArray[np.complex128]] = {}
        flanks: dict[tuple[float, int], NDArray[np.complex128]] = {}
        total_samples = 0
        flank_sample_count = 0
        maximum_envelope_residual = 0.0
        maximum_tangency_residual = 0.0
        maximum_chord_error = 0.0
        maximum_intersection_residual = 0.0
        maximum_join_gap = 0.0
        maximum_fillet_root_residual = 0.0
        undercut_count = 0

        for tooth_phase in phases:
            for sign in (-1, 1):
                geometry = self._analytic_flank_geometry(
                    driven=driven,
                    tooth_phase=tooth_phase,
                    sign=sign,
                )
                geometries[tooth_phase, sign] = geometry
                maximum_intersection_residual = max(
                    maximum_intersection_residual,
                    geometry.intersection_residual,
                )
                fillet_root = self._analytic_fillet_points(
                    driven=driven,
                    tooth_phase=tooth_phase,
                    sign=sign,
                    common_arc=np.asarray([geometry.fillet_root_arc]),
                )[0]
                dedendum_root = self._analytic_offset_points(
                    driven=driven,
                    common_arc=np.asarray([geometry.fillet_root_arc]),
                    height=-self.dedendum,
                )[0]
                maximum_fillet_root_residual = max(
                    maximum_fillet_root_residual,
                    abs(fillet_root - dedendum_root),
                )
                undercut_count += int(geometry.undercut)
                flank, count, envelope, tangency, chord = (
                    self._sample_analytic_flank(
                        driven=driven,
                        geometry=geometry,
                        samples_per_radian=samples_per_radian,
                    )
                )
                flanks[tooth_phase, sign] = flank
                total_samples += count
                flank_sample_count += count
                maximum_envelope_residual = max(
                    maximum_envelope_residual, envelope
                )
                maximum_tangency_residual = max(
                    maximum_tangency_residual, tangency
                )
                maximum_chord_error = max(maximum_chord_error, chord)
                fillet_curve = lambda values, phase=tooth_phase, side=sign: (
                    self._analytic_fillet_points(
                        driven=driven,
                        tooth_phase=phase,
                        sign=side,
                        common_arc=values,
                    )
                )
                fillet, count, chord = self._sample_analytic_curve(
                    fillet_curve,
                    geometry.fillet_root_arc,
                    geometry.fillet_transition_arc,
                    samples_per_radian,
                    minimum_samples=32,
                )
                fillets[tooth_phase, sign] = fillet
                total_samples += count
                maximum_chord_error = max(maximum_chord_error, chord)

        root_blank, count, chord = self._analytic_root_blank(
            driven, samples_per_radian
        )
        total_samples += count
        maximum_chord_error = max(maximum_chord_error, chord)
        tooth_bodies: list[Geometry] = [root_blank]

        def append_piece(
            ring: list[complex], points: NDArray[np.complex128]
        ) -> None:
            nonlocal maximum_join_gap
            piece = np.asarray(points, dtype=np.complex128).copy()
            if ring:
                gap = abs(ring[-1] - piece[0])
                maximum_join_gap = max(maximum_join_gap, gap)
                joint = 0.5 * (ring[-1] + piece[0])
                ring[-1] = joint
                piece[0] = joint
                ring.extend(piece[1:])
            else:
                ring.extend(piece)

        body_phases = range(teeth)
        for tooth in body_phases:
            tooth_phase = float(tooth) if self.closed else tooth + 0.5
            next_phase = tooth_phase + 1.0
            ring: list[complex] = []
            if driven:
                first = geometries[tooth_phase, 1]
                second = geometries[next_phase, -1]
                append_piece(ring, fillets[tooth_phase, 1])
                append_piece(ring, flanks[tooth_phase, 1])
                addendum_start = first.addendum_tip_arc
                addendum_end = second.addendum_tip_arc
                addendum, count, chord = self._sample_analytic_curve(
                    lambda values: self._analytic_offset_points(
                        driven=True,
                        common_arc=values,
                        height=self.addendum,
                    ),
                    addendum_start,
                    addendum_end,
                    samples_per_radian,
                    minimum_samples=32,
                )
                append_piece(ring, addendum)
                total_samples += count
                maximum_chord_error = max(maximum_chord_error, chord)
                append_piece(ring, flanks[next_phase, -1][::-1])
                append_piece(ring, fillets[next_phase, -1][::-1])
                dedendum_start = second.fillet_root_arc
                dedendum_end = first.fillet_root_arc
            else:
                minus = geometries[tooth_phase, -1]
                plus = geometries[tooth_phase, 1]
                append_piece(ring, fillets[tooth_phase, -1])
                append_piece(ring, flanks[tooth_phase, -1])
                addendum, count, chord = self._sample_analytic_curve(
                    lambda values: self._analytic_offset_points(
                        driven=False,
                        common_arc=values,
                        height=self.addendum,
                    ),
                    minus.addendum_tip_arc,
                    plus.addendum_tip_arc,
                    samples_per_radian,
                    minimum_samples=32,
                )
                append_piece(ring, addendum)
                total_samples += count
                maximum_chord_error = max(maximum_chord_error, chord)
                append_piece(ring, flanks[tooth_phase, 1][::-1])
                append_piece(ring, fillets[tooth_phase, 1][::-1])
                dedendum_start = plus.fillet_root_arc
                dedendum_end = minus.fillet_root_arc
            dedendum, count, chord = self._sample_analytic_curve(
                lambda values: self._analytic_offset_points(
                    driven=driven,
                    common_arc=values,
                    height=-self.dedendum,
                ),
                dedendum_start,
                dedendum_end,
                samples_per_radian,
                minimum_samples=32,
            )
            append_piece(ring, dedendum)
            total_samples += count
            maximum_chord_error = max(maximum_chord_error, chord)
            seam_gap = abs(ring[-1] - ring[0])
            maximum_join_gap = max(maximum_join_gap, seam_gap)
            seam = 0.5 * (ring[-1] + ring[0])
            ring[-1] = seam
            ring[0] = seam
            coordinates = np.asarray(ring, dtype=np.complex128)
            tooth_bodies.append(
                make_valid(
                    Polygon(
                        np.column_stack(
                            (coordinates.real, coordinates.imag)
                        )
                    )
                )
            )

        arranged: Geometry = union_all(tooth_bodies)
        if not self.closed:
            if driven:
                start_angle = float(self._psi(self.config.active_start)) + math.pi
                end_angle = float(self._psi(self.config.active_end)) + math.pi
            else:
                start_angle = -self.config.active_start
                end_angle = -self.config.active_end
            if abs(end_angle - start_angle) >= 2.0 * math.pi - 1e-8:
                raise ValueError(
                    "Open gear body span must be less than one body revolution"
                )
            inner_radius = max(
                0.08 * self.config.module, 0.22 * self.minimum_pitch_radius
            )
            outer_radius = (
                self.maximum_pitch_radius
                + self.addendum
                + 2.5 * self.config.module
            )
            arranged = arranged.intersection(
                _open_sector(
                    start_angle,
                    end_angle,
                    inner_radius,
                    outer_radius,
                )
            )
        polygon = _clean_polygon(arranged)
        outline = _outline(
            polygon, 2e-7 * max(1.0, self.config.module)
        )
        return _AnalyticGearResult(
            outline=outline,
            sample_count=total_samples,
            flank_sample_count=flank_sample_count,
            maximum_envelope_residual=maximum_envelope_residual,
            maximum_tangency_residual=maximum_tangency_residual,
            maximum_chord_error=maximum_chord_error,
            maximum_intersection_residual=maximum_intersection_residual,
            maximum_join_gap=maximum_join_gap,
            maximum_fillet_root_residual=maximum_fillet_root_residual,
            undercut_count=undercut_count,
        )

    @staticmethod
    def _place_geometry(
        geometry: Geometry, angle: float, translate_x: float = 0.0
    ) -> Geometry:
        placed = affinity.rotate(
            geometry, angle, origin=(0.0, 0.0), use_radians=True
        )
        if translate_x:
            placed = affinity.translate(placed, xoff=translate_x)
        return placed

    @staticmethod
    def _unplace_geometry(
        geometry: Geometry, angle: float, translate_x: float = 0.0
    ) -> Geometry:
        local = (
            affinity.translate(geometry, xoff=-translate_x)
            if translate_x
            else geometry
        )
        return affinity.rotate(
            local, -angle, origin=(0.0, 0.0), use_radians=True
        )

    def _analytic_pitch_material(self, driven: bool) -> Polygon:
        """Return material inward of the pitch curve for root-only trimming."""

        teeth = self.driven_teeth if driven else self.drive_teeth
        if self.closed:
            start_arc = 0.0
            end_arc = teeth * math.pi * self.config.module
        else:
            start_arc, end_arc = self._analytic_common_arc_bounds()
        sample_count = max(1024, 32 * teeth)
        common_arc = np.linspace(start_arc, end_arc, sample_count + 1)
        pitch_points = self._analytic_offset_points(
            driven=driven,
            common_arc=common_arc,
            height=0.0,
        )
        if self.closed:
            pitch = make_valid(
                Polygon(
                    np.column_stack(
                        (pitch_points.real, pitch_points.imag)
                    )
                )
            )
            hub_radius = max(
                0.08 * self.config.module, 0.20 * self.minimum_pitch_radius
            )
            hub = Point(0.0, 0.0).buffer(hub_radius, quad_segs=96)
            return _clean_polygon(union_all([pitch, hub]))

        phi = self._phi_from_common_arc(common_arc)
        centrode = (
            np.asarray(self._driven_centrode(phi), dtype=np.complex128)
            if driven
            else np.asarray(self._drive_centrode(phi), dtype=np.complex128)
        )
        inner_radius = max(
            0.08 * self.config.module, 0.20 * self.minimum_pitch_radius
        )
        inner = centrode * inner_radius / np.abs(centrode)
        ring = np.concatenate((pitch_points, inner[::-1]))
        return _clean_polygon(
            make_valid(
                Polygon(np.column_stack((ring.real, ring.imag)))
            )
        )

    def _analytic_open_closure_zone(self, driven: bool) -> Geometry | None:
        """Return the non-working end-relief band at each open end face."""

        if self.closed:
            return None
        if driven:
            start_angle = float(self._psi(self.config.active_start)) + math.pi
            end_angle = float(self._psi(self.config.active_end)) + math.pi
        else:
            start_angle = -self.config.active_start
            end_angle = -self.config.active_end
        inner_radius = max(
            0.08 * self.config.module, 0.20 * self.minimum_pitch_radius
        )
        outer_radius = (
            self.maximum_pitch_radius
            + self.addendum
            + 2.5 * self.config.module
        )
        boundaries = []
        for angle in (start_angle, end_angle):
            direction = np.asarray([math.cos(angle), math.sin(angle)])
            boundaries.append(
                LineString(
                    np.vstack(
                        (inner_radius * direction, outer_radius * direction)
                    )
                )
            )
        # The public API guarantees at least 2.5 pitch intervals of source
        # motion beyond each active endpoint. The same fringe identifies end
        # teeth that may be relieved by the mate without touching the
        # analytical working profile in the interior active span.
        return union_all(boundaries).buffer(
            2.5 * math.pi * self.config.module,
            cap_style="flat",
            join_style="mitre",
        )

    def _trim_rolling_nonworking_interference(
        self, drive: FloatArray, driven: FloatArray
    ) -> tuple[FloatArray, FloatArray, int, float, float, float]:
        """Trim non-working interference by rolling the finished pair.

        The analytical profiles remain authoritative outside their pitch
        curves and, for open profiles, outside the padded end-relief bands. At
        each rolling pose GEOS computes actual solid overlap; only portions in
        those non-working regions are eligible for removal. Four staggered
        phase grids prevent a favorable tooth-grid alignment from hiding an
        undercut.
        """

        drive_geometry: Geometry = Polygon(drive[:-1])
        driven_geometry: Geometry = Polygon(driven[:-1])
        drive_root_zone = self._analytic_pitch_material(False)
        driven_root_zone = self._analytic_pitch_material(True)
        drive_closure_zone = self._analytic_open_closure_zone(False)
        driven_closure_zone = self._analytic_open_closure_zone(True)
        drive_trim_zone = (
            union_all([drive_root_zone, drive_closure_zone])
            if drive_closure_zone is not None
            else drive_root_zone
        )
        driven_trim_zone = (
            union_all([driven_root_zone, driven_closure_zone])
            if driven_closure_zone is not None
            else driven_root_zone
        )
        phase_count = (
            max(96, 4 * max(self.drive_teeth, self.driven_teeth))
            if self.closed
            else 96
        )
        overlap_tolerance = (
            1e-6
            * self.config.module**2
            * max(self.drive_teeth, self.driven_teeth)
        )
        classification_tolerance = 1e-10 * self.config.module**2
        # Buffer by one certified chord-error allowance so finite-precision
        # Boolean boundaries do not leave a sliver of the measured overlap.
        trim_clearance = (
            _ANALYTIC_CHORD_TOLERANCE_FACTOR * self.config.module
        )
        psi_start = float(self._psi(self.active_start))
        initial_overlap = 0.0
        remaining_overlap = 0.0
        total_removed_area = 0.0
        total_phases = 0

        for offset in (0.0, 0.5, 0.25, 0.75):
            fractions = (
                (np.arange(phase_count) + offset) / phase_count
                if self.closed
                else np.clip(
                    (np.arange(phase_count) + offset)
                    / max(1, phase_count - 1),
                    0.0,
                    1.0,
                )
            )
            drive_cuts: list[Geometry] = []
            driven_cuts: list[Geometry] = []
            grid_maximum = 0.0

            def phase_root_cut(
                fraction: float,
                current_drive: Geometry = drive_geometry,
                current_driven: Geometry = driven_geometry,
            ) -> tuple[float, Geometry | None, Geometry | None, float]:
                phi = self.active_start + (
                    self.active_end - self.active_start
                ) * float(fraction)
                drive_angle = phi - self.active_start
                driven_angle = -(
                    float(self._psi(phi)) - psi_start
                )
                placed_drive = self._place_geometry(
                    current_drive, drive_angle
                )
                placed_driven = self._place_geometry(
                    current_driven,
                    driven_angle,
                    self.center_distance,
                )
                overlap = placed_drive.intersection(placed_driven)
                area = float(overlap.area)
                if area <= overlap_tolerance:
                    return area, None, None, 0.0
                placed_drive_root = self._place_geometry(
                    drive_root_zone, drive_angle
                )
                placed_driven_root = self._place_geometry(
                    driven_root_zone,
                    driven_angle,
                    self.center_distance,
                )
                drive_root_overlap = overlap.intersection(
                    placed_drive_root
                )
                driven_root_overlap = overlap.intersection(
                    placed_driven_root
                )
                drive_closure_overlap: Geometry | None = None
                driven_closure_overlap: Geometry | None = None
                coverage: list[Geometry] = [
                    drive_root_overlap,
                    driven_root_overlap,
                ]
                if drive_closure_zone is not None:
                    drive_closure_overlap = overlap.intersection(
                        self._place_geometry(
                            drive_closure_zone, drive_angle
                        )
                    )
                    coverage.append(drive_closure_overlap)
                if driven_closure_zone is not None:
                    driven_closure_overlap = overlap.intersection(
                        self._place_geometry(
                            driven_closure_zone,
                            driven_angle,
                            self.center_distance,
                        )
                    )
                    coverage.append(driven_closure_overlap)
                covered = union_all(coverage)
                uncovered_area = float(overlap.difference(covered).area)
                drive_overlap = union_all(
                    [
                        drive_root_overlap,
                        *(
                            [drive_closure_overlap]
                            if drive_closure_overlap is not None
                            else []
                        ),
                    ]
                )
                driven_overlap = union_all(
                    [
                        driven_root_overlap,
                        *(
                            [driven_closure_overlap]
                            if driven_closure_overlap is not None
                            else []
                        ),
                    ]
                )
                local_drive_cut = (
                    self._unplace_geometry(
                        drive_overlap, drive_angle
                    )
                    if not drive_overlap.is_empty
                    else None
                )
                local_driven_cut = (
                    self._unplace_geometry(
                        driven_overlap,
                        driven_angle,
                        self.center_distance,
                    )
                    if not driven_overlap.is_empty
                    else None
                )
                return (
                    area,
                    local_drive_cut,
                    local_driven_cut,
                    uncovered_area,
                )

            with ThreadPoolExecutor(
                max_workers=min(_MAX_GEOMETRY_WORKERS, len(fractions))
            ) as executor:
                phase_results = executor.map(
                    phase_root_cut, (float(value) for value in fractions)
                )
                for (
                    area,
                    drive_root_overlap,
                    driven_root_overlap,
                    uncovered_area,
                ) in phase_results:
                    grid_maximum = max(grid_maximum, area)
                    if uncovered_area > classification_tolerance:
                        raise RuntimeError(
                            "Rolling verification found interference on the "
                            "certified involute working flanks "
                            f"(area {uncovered_area:.9g})"
                        )
                    if drive_root_overlap is not None:
                        drive_cuts.append(drive_root_overlap)
                    if driven_root_overlap is not None:
                        driven_cuts.append(driven_root_overlap)
            total_phases += len(fractions)
            if offset == 0.0:
                initial_overlap = grid_maximum
            remaining_overlap = max(remaining_overlap, grid_maximum)
            if not drive_cuts and not driven_cuts:
                continue
            before_drive = float(drive_geometry.area)
            before_driven = float(driven_geometry.area)
            if drive_cuts:
                drive_cut = union_all(drive_cuts).buffer(
                    trim_clearance, quad_segs=2
                ).intersection(drive_trim_zone)
                drive_geometry = drive_geometry.difference(drive_cut)
            if driven_cuts:
                driven_cut = union_all(driven_cuts).buffer(
                    trim_clearance, quad_segs=2
                ).intersection(driven_trim_zone)
                driven_geometry = driven_geometry.difference(driven_cut)
            drive_geometry = _clean_polygon(drive_geometry)
            driven_geometry = _clean_polygon(driven_geometry)
            total_removed_area += (
                before_drive
                - float(drive_geometry.area)
                + before_driven
                - float(driven_geometry.area)
            )

        drive_outline = _outline(
            _clean_polygon(drive_geometry),
            2e-7 * max(1.0, self.config.module),
        )
        driven_outline = _outline(
            _clean_polygon(driven_geometry),
            2e-7 * max(1.0, self.config.module),
        )
        return (
            drive_outline,
            driven_outline,
            total_phases,
            initial_overlap,
            remaining_overlap,
            total_removed_area,
        )

    def _overlap(
        self,
        drive: FloatArray,
        driven: FloatArray,
        drive_angle: float,
        driven_angle: float,
    ) -> float:
        placed_drive = _transform_outline(drive, drive_angle)
        placed_driven = _transform_outline(driven, driven_angle, self.center_distance)
        return float(placed_drive.intersection(placed_driven).area)

    def _overlap_with_placed_drive(
        self,
        placed_drive: Polygon,
        driven: FloatArray,
        driven_angle: float,
    ) -> float:
        placed_driven = _transform_outline(driven, driven_angle, self.center_distance)
        return float(placed_drive.intersection(placed_driven).area)

    def _verify_pair(
        self, drive: FloatArray, driven: FloatArray
    ) -> tuple[float, float, int]:
        phase_count = (
            max(64, 4 * max(self.drive_teeth, self.driven_teeth)) if self.closed else 48
        )
        fractions = (
            np.arange(phase_count) / phase_count
            if self.closed
            else np.linspace(0.0, 1.0, phase_count)
        )
        psi_start = float(self._psi(self.active_start))

        def phase_overlap(fraction: float) -> float:
            phi = self.active_start + (self.active_end - self.active_start) * float(
                fraction
            )
            return self._overlap(
                drive,
                driven,
                phi - self.active_start,
                -(float(self._psi(phi)) - psi_start),
            )

        with ThreadPoolExecutor(
            max_workers=min(_MAX_GEOMETRY_WORKERS, phase_count)
        ) as executor:
            overlaps = list(executor.map(phase_overlap, fractions))
        maximum_overlap = max(overlaps)
        overlap_tolerance = (
            1e-6 * self.config.module**2 * max(self.drive_teeth, self.driven_teeth)
        )
        if maximum_overlap > overlap_tolerance:
            raise RuntimeError(
                "Continuous-phase verification found solid overlap area "
                f"{maximum_overlap:.9g} above tolerance {overlap_tolerance:.9g}"
            )

        contact_area_tolerance = 1e-11 * self.config.module**2
        recovery_count = 6 if self.closed else 4

        def recover_contact_delta(index: int) -> float:
            fraction = (index + 0.37) / recovery_count
            phi = self.active_start + (self.active_end - self.active_start) * fraction
            drive_angle = phi - self.active_start
            desired = -(float(self._psi(phi)) - psi_start)
            placed_drive = _transform_outline(drive, drive_angle)
            if (
                self._overlap_with_placed_drive(placed_drive, driven, desired)
                > contact_area_tolerance
            ):
                return 0.0
            best = math.inf
            for direction in (-1.0, 1.0):
                clear = 0.0
                collision = 1e-5
                found = False
                while collision <= 0.08:
                    if (
                        self._overlap_with_placed_drive(
                            placed_drive,
                            driven,
                            desired + direction * collision,
                        )
                        > contact_area_tolerance
                    ):
                        found = True
                        break
                    clear = collision
                    collision *= 2.0
                if not found:
                    continue
                for _ in range(16):
                    midpoint = 0.5 * (clear + collision)
                    if (
                        self._overlap_with_placed_drive(
                            placed_drive,
                            driven,
                            desired + direction * midpoint,
                        )
                        > contact_area_tolerance
                    ):
                        collision = midpoint
                    else:
                        clear = midpoint
                best = min(best, collision)
            if not math.isfinite(best):
                raise RuntimeError(
                    "Finished solids do not establish positive contact near "
                    "the requested motion"
                )
            return best

        with ThreadPoolExecutor(
            max_workers=min(_MAX_GEOMETRY_WORKERS, recovery_count)
        ) as executor:
            contact_deltas = list(
                executor.map(recover_contact_delta, range(recovery_count))
            )
        return maximum_overlap, max(contact_deltas), phase_count

    def _drive_centrode_outline_distance(self, drive: FloatArray) -> float | None:
        if not self.closed:
            return None
        sample_count = max(4097, 128 * self.drive_teeth + 1)
        phi = np.linspace(
            self.active_start,
            self.active_start + self.drive_cycle,
            sample_count,
        )
        radius = np.asarray(self._drive_radius(phi), dtype=float)
        centrode = radius * np.exp(-1j * phi)
        centrode_line = LineString(
            np.column_stack((centrode.real, centrode.imag))
        )
        return float(centrode_line.hausdorff_distance(LineString(drive)))

    def generate(self, samples_per_radian: int) -> EngineResult:
        if samples_per_radian < 20:
            raise ValueError("samples_per_radian must be at least 20")
        analytic_involute = self.config.profile == "involute"
        maximum_envelope_residual: float | None = None
        maximum_tangency_residual: float | None = None
        maximum_chord_error: float | None = None
        maximum_intersection_residual = 0.0
        maximum_join_gap = 0.0
        maximum_fillet_root_residual = 0.0
        analytic_undercut_count = 0
        rolling_trim_phase_count = 0
        rolling_initial_overlap = 0.0
        rolling_sampled_overlap = 0.0
        rolling_removed_area = 0.0
        analytic_curve_sample_count = 0
        analytic_flank_sample_count = 0
        if analytic_involute:
            drive_result = self._generate_analytic_involute_gear(
                False, samples_per_radian
            )
            driven_result = self._generate_analytic_involute_gear(
                True, samples_per_radian
            )
            drive = drive_result.outline
            driven = driven_result.outline
            drive_phases = 0
            driven_phases = 0
            analytic_curve_sample_count = (
                drive_result.sample_count + driven_result.sample_count
            )
            analytic_flank_sample_count = (
                drive_result.flank_sample_count
                + driven_result.flank_sample_count
            )
            maximum_envelope_residual = max(
                drive_result.maximum_envelope_residual,
                driven_result.maximum_envelope_residual,
            )
            maximum_tangency_residual = max(
                drive_result.maximum_tangency_residual,
                driven_result.maximum_tangency_residual,
            )
            maximum_chord_error = max(
                drive_result.maximum_chord_error,
                driven_result.maximum_chord_error,
            )
            maximum_intersection_residual = max(
                drive_result.maximum_intersection_residual,
                driven_result.maximum_intersection_residual,
            )
            maximum_join_gap = max(
                drive_result.maximum_join_gap,
                driven_result.maximum_join_gap,
            )
            maximum_fillet_root_residual = max(
                drive_result.maximum_fillet_root_residual,
                driven_result.maximum_fillet_root_residual,
            )
            analytic_undercut_count = (
                drive_result.undercut_count + driven_result.undercut_count
            )
            (
                drive,
                driven,
                rolling_trim_phase_count,
                rolling_initial_overlap,
                rolling_sampled_overlap,
                rolling_removed_area,
            ) = self._trim_rolling_nonworking_interference(drive, driven)
            if maximum_envelope_residual > 1e-8 * self.config.module:
                raise RuntimeError(
                    "Analytic involute envelope residual exceeds tolerance "
                    f"({maximum_envelope_residual:.9g})"
                )
            if maximum_tangency_residual > 1e-10:
                raise RuntimeError(
                    "Analytic involute tangency residual exceeds tolerance "
                    f"({maximum_tangency_residual:.9g})"
                )
            if maximum_chord_error > 3e-5 * self.config.module:
                raise RuntimeError(
                    "Analytic involute tessellation error exceeds tolerance "
                    f"({maximum_chord_error:.9g})"
                )
            if maximum_fillet_root_residual > 1e-8 * self.config.module:
                raise RuntimeError(
                    "Analytic rack-tip fillet does not meet the dedendum "
                    f"({maximum_fillet_root_residual:.9g})"
                )
        else:
            drive, drive_phases = self._generate_swept_gear(False, samples_per_radian)
            driven, driven_phases = self._generate_conjugate_mate(
                drive, samples_per_radian
            )
        drive_centrode_outline_distance = self._drive_centrode_outline_distance(drive)
        centrode_fidelity_tolerance = (
            max(self.addendum, self.dedendum)
            + self.fillet_radius
            + 0.05 * self.config.module
        )
        if (
            drive_centrode_outline_distance is not None
            and drive_centrode_outline_distance > centrode_fidelity_tolerance
        ):
            failure = (
                "Analytic involute arrangement did not preserve the requested "
                "drive centrode"
                if analytic_involute
                else "Rack sweep self-occluded the requested drive centrode"
            )
            raise RuntimeError(
                f"{failure} "
                f"(outline distance {drive_centrode_outline_distance:.9g}, "
                f"tooth-envelope tolerance {centrode_fidelity_tolerance:.9g})"
            )
        if self.closed:
            phi = np.linspace(self.active_start, self.active_end, 32769)
            pitch_area = float(
                simpson(0.5 * np.asarray(self._drive_radius(phi)) ** 2, x=phi)
            )
            outline_area = abs(_signed_area(drive))
            if outline_area < 0.75 * pitch_area:
                backend_name = (
                    "Analytic involute arrangement"
                    if analytic_involute
                    else "Rack sweep"
                )
                raise RuntimeError(
                    f"{backend_name} disconnected the intended drive-gear body "
                    f"(outline area {outline_area:.9g}, drive-centrode enclosed "
                    f"area {pitch_area:.9g})"
                )
        maximum_overlap, transmission_error, verification_phases = self._verify_pair(
            drive, driven
        )
        drive_radii = np.linalg.norm(drive[:-1], axis=1)
        driven_radii = np.linalg.norm(driven[:-1], axis=1)
        minimum_root_radius = float(min(np.min(drive_radii), np.min(driven_radii)))
        phi = np.linspace(self.active_start, self.active_end, 4097)
        maximum_ratio = float(np.max(self._psi(phi, 1)))
        contact_distance_bound = (
            self.addendum / max(1e-6, math.sin(self.alpha))
            + 0.5 * math.pi * self.config.module
        )
        total_phases = drive_phases + driven_phases
        profile_family = (
            "generalized_involute"
            if analytic_involute
            else f"{self.config.profile}_rack"
        )
        generation_backend = (
            "analytic_form" if analytic_involute else "sampled_cutter_sweep"
        )
        metadata: dict[str, object] = {
            "name": self.config.name,
            "description": self.config.description,
            "topology": "closed" if self.closed else "open",
            "input_mode": self.config.input_mode,
            "domain_start": self.config.domain_start,
            "domain_end": self.config.domain_end,
            "active_start": self.active_start,
            "active_end": self.active_end,
            "period": self.config.period,
            "cycle_delta": self.config.cycle_delta,
            "centrode_reference_center_distance": self.config.reference_center_distance,
            "profile_family": profile_family,
            "generation_backend": generation_backend,
            "geometry_backend": "shapely-geos",
            "geometry_precision": "double",
            "geometry_worker_limit": _MAX_GEOMETRY_WORKERS,
            "drive_teeth": self.drive_teeth,
            "driven_teeth": self.driven_teeth,
            "average_angular_ratio": self.average_ratio,
            "module": self.config.module,
            "pressure_angle_deg": self.config.pressure_angle_deg,
            "total_integral": self.total_integral,
            "center_distance": self.center_distance,
            "undercut_curvature_limit": self.curvature_limit,
            "maximum_join_gap": maximum_join_gap,
            "maximum_intersection_residual": maximum_intersection_residual,
            "maximum_fillet_root_residual": maximum_fillet_root_residual,
            "analytic_undercut_count": analytic_undercut_count,
            "rolling_nonworking_trim_scope": (
                "pitch_side_roots"
                if self.closed
                else "pitch_side_roots_and_open_end_relief"
            ),
            "rolling_nonworking_trim_phase_count": rolling_trim_phase_count,
            "rolling_nonworking_initial_overlap_area": rolling_initial_overlap,
            "rolling_nonworking_sampled_overlap_area": rolling_sampled_overlap,
            "rolling_nonworking_removed_area": rolling_removed_area,
            "placed_pair_overlap_area": maximum_overlap,
            "centrodes_are_convex": self.centrodes_are_convex,
            "maximum_drive_curvature": self.maximum_drive_curvature,
            "minimum_driven_curvature": self.minimum_driven_curvature,
            "drive_centrode_outline_distance": drive_centrode_outline_distance,
            "centrode_fidelity_tolerance": centrode_fidelity_tolerance,
            "cutter_sweep_phase_count": total_phases,
            "analytic_curve_sample_count": analytic_curve_sample_count,
            "analytic_flank_sample_count": analytic_flank_sample_count,
            "maximum_envelope_residual": maximum_envelope_residual,
            "maximum_envelope_tangency_residual": maximum_tangency_residual,
            "maximum_analytic_chord_error": maximum_chord_error,
            "nonworking_closure": (
                "analytic_rack_tip_and_dedendum_envelopes"
                if analytic_involute
                else "generated_cutter_root"
            ),
            "requested_fillet_radius": self.fillet_radius,
            "requested_fillet_applied_to_closure": True,
            "verification_phase_count": verification_phases,
            "sweep_angular_step": (
                0.0
                if analytic_involute
                else max(
                    self.drive_cycle / drive_phases,
                    self.driven_cycle / driven_phases,
                )
            ),
            "maximum_transmission_error": transmission_error,
            "maximum_sliding_velocity_factor": (1.0 + maximum_ratio)
            * contact_distance_bound,
            "minimum_root_radius": minimum_root_radius,
            "minimum_tip_thickness": 0.5 * math.pi * self.config.module
            - 2.0 * self.addendum * math.tan(self.alpha),
            "drive_area": _signed_area(drive),
            "driven_area": _signed_area(driven),
        }
        log = (
            f"Generated {self.config.name} with {generation_backend} "
            "and Shapely/GEOS: "
            f"{self.drive_teeth}:{self.driven_teeth} teeth, "
            + (
                f"{analytic_flank_sample_count} analytic flank samples"
                if analytic_involute
                else f"{total_phases} cutter poses"
            )
        )
        return EngineResult(drive, driven, metadata, log)


def generate_geometry(config: EngineConfig, samples_per_radian: int) -> EngineResult:
    """Generate gear outlines and verification metadata in-process."""

    return _GearGenerator(config).generate(samples_per_radian)
