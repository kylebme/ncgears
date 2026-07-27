"""Python geometry engine for conjugate noncircular gears.

The kinematics and rack-envelope construction are a direct Python
implementation of the original ncgear algorithm.  Shapely/GEOS supplies the
floating-point regularized polygon operations; the sampled cutter motion, not
the Boolean predicate representation, remains the dominant geometry
approximation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson, simpson
from shapely import Geometry, make_valid, union_all
from shapely.geometry import MultiPolygon, Point, Polygon

FloatArray = NDArray[np.float64]


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
        next_index = (index + 1) % count if self.periodic else index + 1

        y0 = self.values[index]
        y1 = self.values[next_index]
        if self.periodic:
            y1 = y1 + np.where(next_index == 0, self.cycle_delta, 0.0)
        d0 = self.first[index]
        d1 = self.first[next_index]
        dd0 = self.second[index]
        dd1 = self.second[next_index]

        step = self.step
        a0 = y0
        a1 = step * d0
        a2 = 0.5 * step * step * dd0
        value_error = y1 - (a0 + a1 + a2)
        slope_error = step * d1 - (a1 + 2.0 * a2)
        curvature_error = step * step * dd1 - 2.0 * a2
        a3 = 10.0 * value_error - 4.0 * slope_error + 0.5 * curvature_error
        a4 = -15.0 * value_error + 7.0 * slope_error - curvature_error
        a5 = 6.0 * value_error - 3.0 * slope_error + 0.5 * curvature_error

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

    def _drive_centrode(self, phi: float) -> complex:
        return float(self._drive_radius(phi)) * np.exp(-1j * phi)

    def _driven_centrode(self, phi: float) -> complex:
        return -float(self._driven_radius(phi)) * np.exp(1j * float(self._psi(phi)))

    def _drive_tangent(self, phi: float) -> complex:
        first = float(self._psi(phi, 1))
        return (
            complex(float(self._psi(phi, 2)), -first * (1.0 + first))
            / float(self._w(phi))
            * np.exp(-1j * phi)
        )

    def _driven_tangent(self, phi: float) -> complex:
        first = float(self._psi(phi, 1))
        return (
            complex(float(self._psi(phi, 2)), -first * (1.0 + first))
            / float(self._w(phi))
            * np.exp(1j * float(self._psi(phi)))
        )

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

    def _append_involute_tooth(
        self, boundary: list[complex], center: float, margin: float
    ) -> None:
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
        boundary.extend(
            [
                complex(center - 0.5 * pitch, self.addendum + y_shift),
                complex(center - root_half_width, self.addendum + y_shift),
            ]
        )
        if radius <= 1e-12:
            boundary.extend(
                [
                    complex(center - sharp_tip_half_width, -self.dedendum + y_shift),
                    complex(center + sharp_tip_half_width, -self.dedendum + y_shift),
                ]
            )
        else:
            transition = radius * (1.0 - math.sin(self.alpha)) / math.cos(self.alpha)
            left_center = complex(
                center - sharp_tip_half_width + transition,
                -self.dedendum + radius + y_shift,
            )
            for angle in np.linspace(math.pi + self.alpha, 1.5 * math.pi, 7):
                boundary.append(left_center + radius * np.exp(1j * angle))
            right_center = complex(
                center + sharp_tip_half_width - transition,
                -self.dedendum + radius + y_shift,
            )
            boundary.append(complex(right_center.real, -self.dedendum + y_shift))
            for angle in np.linspace(-0.5 * math.pi, -self.alpha, 7):
                boundary.append(right_center + radius * np.exp(1j * angle))
        boundary.extend(
            [
                complex(center + root_half_width, self.addendum + y_shift),
                complex(center + 0.5 * pitch, self.addendum + y_shift),
            ]
        )

    def _append_cycloidal_tooth(
        self, boundary: list[complex], center: float, margin: float
    ) -> None:
        pitch = math.pi * self.config.module
        root_half_width = 0.25 * pitch + self.addendum * math.tan(self.alpha)
        tip_half_width = 0.25 * pitch - self.dedendum * math.tan(self.alpha)
        blend = min(max(self.config.cycloidal_rolling_factor, 0.0), 1.0)
        y_shift = -margin
        boundary.extend(
            [
                complex(center - 0.5 * pitch, self.addendum + y_shift),
                complex(center - root_half_width, self.addendum + y_shift),
            ]
        )
        for index in range(1, 15):
            q = index / 14.0
            tau = math.pi * q
            x_fraction = (1.0 - blend) * q + blend * ((tau - math.sin(tau)) / math.pi)
            y_fraction = (1.0 - blend) * q + blend * (0.5 * (1.0 - math.cos(tau)))
            boundary.append(
                complex(
                    center
                    - (
                        root_half_width
                        + (tip_half_width - root_half_width) * x_fraction
                    ),
                    self.addendum
                    + (-self.dedendum - self.addendum) * y_fraction
                    + y_shift,
                )
            )
        boundary.append(complex(center + tip_half_width, -self.dedendum + y_shift))
        for index in range(13, -1, -1):
            q = index / 14.0
            tau = math.pi * q
            x_fraction = (1.0 - blend) * q + blend * ((tau - math.sin(tau)) / math.pi)
            y_fraction = (1.0 - blend) * q + blend * (0.5 * (1.0 - math.cos(tau)))
            boundary.append(
                complex(
                    center
                    + (
                        root_half_width
                        + (tip_half_width - root_half_width) * x_fraction
                    ),
                    self.addendum
                    + (-self.dedendum - self.addendum) * y_fraction
                    + y_shift,
                )
            )
        boundary.append(complex(center + 0.5 * pitch, self.addendum + y_shift))

    def _make_rack(
        self,
        common_arc: float,
        phase_offset: float,
        half_width: float,
        top: float,
        margin: float,
    ) -> list[complex]:
        pitch = math.pi * self.config.module
        first_center = (
            phase_offset
            - common_arc
            + math.floor((-half_width - phase_offset + common_arc) / pitch) * pitch
        )
        boundary = [complex(first_center - pitch, self.addendum - margin)]
        center = first_center - 0.5 * pitch
        while center <= half_width + pitch:
            if self.config.profile == "cycloidal":
                self._append_cycloidal_tooth(boundary, center, margin)
            else:
                self._append_involute_tooth(boundary, center, margin)
            center += pitch
        boundary.extend(
            [
                complex(boundary[-1].real, top),
                complex(boundary[0].real, top),
            ]
        )
        return boundary

    @staticmethod
    def _rack_polygon(
        local: list[complex],
        pitch_point: complex,
        tangent: complex,
        outward_normal: complex,
    ) -> Polygon:
        unique = [local[0]]
        for point in local[1:]:
            if abs(point - unique[-1]) > 1e-12:
                unique.append(point)
        transformed = [
            pitch_point + point.real * tangent + point.imag * outward_normal
            for point in unique
        ]
        polygon = Polygon([(point.real, point.imag) for point in transformed])
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
        sweep_margin = 2e-4 * self.config.module
        cutters: list[Polygon] = []
        for phi in np.linspace(cycle_start, cycle_end, phase_count, endpoint=False):
            common_arc = self.center_distance * float(
                self.arc_integral.integral(self.active_start, phi)
            )
            pitch_point = (
                self._driven_centrode(float(phi))
                if driven
                else self._drive_centrode(float(phi))
            )
            tangent = (
                self._driven_tangent(float(phi))
                if driven
                else self._drive_tangent(float(phi))
            )
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
        overlaps = []
        for fraction in fractions:
            phi = self.active_start + (self.active_end - self.active_start) * float(
                fraction
            )
            overlaps.append(
                self._overlap(
                    drive,
                    driven,
                    phi - self.active_start,
                    -(float(self._psi(phi)) - psi_start),
                )
            )
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
        contact_deltas: list[float] = []
        recovery_count = 6 if self.closed else 4
        for index in range(recovery_count):
            fraction = (index + 0.37) / recovery_count
            phi = self.active_start + (self.active_end - self.active_start) * fraction
            drive_angle = phi - self.active_start
            desired = -(float(self._psi(phi)) - psi_start)
            if (
                self._overlap(drive, driven, drive_angle, desired)
                > contact_area_tolerance
            ):
                contact_deltas.append(0.0)
                continue
            best = math.inf
            for direction in (-1.0, 1.0):
                clear = 0.0
                collision = 1e-5
                found = False
                while collision <= 0.08:
                    if (
                        self._overlap(
                            drive,
                            driven,
                            drive_angle,
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
                        self._overlap(
                            drive,
                            driven,
                            drive_angle,
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
            contact_deltas.append(best)
        return maximum_overlap, max(contact_deltas), phase_count

    def generate(self, samples_per_radian: int) -> EngineResult:
        if samples_per_radian < 20:
            raise ValueError("samples_per_radian must be at least 20")
        drive, drive_phases = self._generate_swept_gear(False, samples_per_radian)
        if self.config.profile == "cycloidal":
            driven, driven_phases = self._generate_conjugate_mate(
                drive, samples_per_radian
            )
        else:
            driven, driven_phases = self._generate_swept_gear(True, samples_per_radian)
        if self.closed:
            phi = np.linspace(self.active_start, self.active_end, 32769)
            pitch_area = float(
                simpson(0.5 * np.asarray(self._drive_radius(phi)) ** 2, x=phi)
            )
            outline_area = abs(_signed_area(drive))
            if outline_area < 0.75 * pitch_area:
                raise RuntimeError(
                    "Rack sweep disconnected the intended drive-gear body "
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
        metadata: dict[str, object] = {
            "name": self.config.name,
            "description": self.config.description,
            "topology": "closed" if self.closed else "open",
            "input_mode": self.config.input_mode,
            "centrode_reference_center_distance": self.config.reference_center_distance,
            "profile_family": f"{self.config.profile}_rack",
            "geometry_backend": "shapely-geos",
            "geometry_precision": "double",
            "drive_teeth": self.drive_teeth,
            "driven_teeth": self.driven_teeth,
            "average_angular_ratio": self.average_ratio,
            "module": self.config.module,
            "pressure_angle_deg": self.config.pressure_angle_deg,
            "total_integral": self.total_integral,
            "center_distance": self.center_distance,
            "undercut_curvature_limit": self.curvature_limit,
            "maximum_join_gap": 0.0,
            "maximum_intersection_residual": 0.0,
            "placed_pair_overlap_area": maximum_overlap,
            "centrodes_are_convex": self.centrodes_are_convex,
            "maximum_drive_curvature": self.maximum_drive_curvature,
            "minimum_driven_curvature": self.minimum_driven_curvature,
            "cutter_sweep_phase_count": total_phases,
            "verification_phase_count": verification_phases,
            "sweep_angular_step": max(
                self.drive_cycle / drive_phases,
                self.driven_cycle / driven_phases,
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
            f"Generated {self.config.name} with Shapely/GEOS: "
            f"{self.drive_teeth}:{self.driven_teeth} teeth, "
            f"{total_phases} cutter poses"
        )
        return EngineResult(drive, driven, metadata, log)


def generate_geometry(config: EngineConfig, samples_per_radian: int) -> EngineResult:
    """Generate gear outlines and verification metadata in-process."""

    return _GearGenerator(config).generate(samples_per_radian)
