"""Python geometry engine for conjugate noncircular gears.

Generalized involute flanks and rounded rack-tip fillets are evaluated from
their analytical envelopes. Shapely/GEOS supplies curve arrangement and
non-working-profile rolling interference removal.
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
from scipy.optimize import brentq, least_squares, minimize_scalar
from scipy.spatial import cKDTree
from shapely import Geometry, affinity, make_valid, union_all
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import nearest_points

from ._policy import (
    ANALYTIC_CHORD_ACCEPTANCE_SLACK,
    ANALYTIC_CHORD_TOLERANCE_FACTOR,
    ANALYTIC_CLOSURE_TOLERANCE_FACTOR,
    ANALYTIC_ENVELOPE_RESIDUAL_FACTOR,
    ANALYTIC_JOIN_TOLERANCE_FACTOR,
    ANALYTIC_REGULAR_DERIVATIVE_TOLERANCE,
    ANALYTIC_TANGENCY_RESIDUAL_TOLERANCE,
    ANGLE_CLOSURE_TOLERANCE,
    CENTRODE_ANALYSIS_INTERVALS_PER_INPUT_SEGMENT,
    CENTRODE_CONVEXITY_CURVATURE_FACTOR,
    CENTRODE_FIDELITY_ALLOWANCE_MODULES,
    CENTRODE_OUTLINE_SAMPLES_PER_TOOTH,
    CLEARANCE_BUFFER_QUADRANT_SEGMENTS,
    CONTACT_AREA_TOLERANCE_FACTOR,
    CONTACT_RECOVERY_PHASE_OFFSET,
    CONTACT_RECOVERY_PHASES_CLOSED,
    CONTACT_RECOVERY_PHASES_OPEN,
    CONTACT_SEARCH_ANGLE_TOLERANCE,
    CONTACT_SEARCH_INITIAL_ANGLE,
    CONTACT_SEARCH_MAX_ANGLE,
    CURVE_SAMPLES_PER_INPUT_RADIAN_FACTOR,
    CUSP_EQUATION_TOLERANCE,
    CUSP_INITIAL_SAMPLES,
    CUSP_MAX_SAMPLES,
    CUSP_PARAMETER_DEDUP_FACTOR,
    FLANK_OFFSET_SOLVER_MAX_EVALUATIONS,
    FLOAT_COMPARISON_ULPS,
    GEOMETRY_LENGTH_TOLERANCE_FACTOR,
    HYBRID_CUSP_CLEARANCE_FACTOR,
    INTEGRATION_INTERVAL_COUNT,
    INTERSECTION_CANDIDATE_OFFSET_WEIGHT,
    INTERSECTION_CONTACT_EXCLUSION_PITCHES,
    INTERSECTION_FLANK_HALF_WINDOW_PITCHES,
    INTERSECTION_MIN_SAMPLES,
    INTERSECTION_OFFSET_HALF_WINDOW_PITCHES,
    INTERSECTION_PARAMETER_DEDUP_FACTOR,
    INTERSECTION_RESIDUAL_FACTOR,
    INTERSECTION_SAMPLES_PER_PITCH,
    INTERSECTION_SOLVER_MAX_EVALUATIONS,
    INTERSECTION_SOLVER_TOLERANCE,
    MAX_ANALYTIC_CURVE_SAMPLES,
    MAX_GEOMETRY_WORKERS,
    MAX_MOTION_RATIO,
    MAX_PRESSURE_ANGLE_DEG,
    MAX_SUPPORT_RADIUS_PITCH_FACTOR,
    MIN_CENTRODE_ANALYSIS_INTERVALS,
    MIN_CLOSED_RING_COORDINATE_COUNT,
    MIN_CLOSURE_CURVE_SAMPLES,
    MIN_FLANK_CURVE_SAMPLES,
    MIN_MOTION_RATIO,
    MIN_POLYGON_VERTEX_COUNT,
    MIN_ROOT_CURVE_SAMPLES,
    MIN_SAMPLE_TABLE_ROWS,
    MIN_SAMPLES_PER_RADIAN,
    MIN_SUPPORT_RADIUS_MODULE_FACTOR,
    MIN_TEETH,
    MINIMUM_PITCH_AREA_FRACTION,
    OPEN_ANALYTIC_BACKING_RADIUS_PITCH_FACTOR,
    OUTLINE_COLLINEAR_TOLERANCE_FRACTION,
    OUTLINE_DUPLICATE_TOLERANCE_FRACTION,
    OVERLAP_AREA_TOLERANCE_FACTOR,
    OVERLAP_CONTACT_PAIR_ALLOWANCE,
    PERIODIC_FIRST_DERIVATIVE_TOLERANCE,
    PERIODIC_SECOND_DERIVATIVE_TOLERANCE,
    PERIODIC_VALUE_REL_TOLERANCE,
    PITCH_MASK_BUFFER_QUADRANT_SEGMENTS,
    PITCH_MASK_SUPPORT_RADIUS_PITCH_FACTOR,
    PROTECTED_CONTACT_RESIDUAL_FACTOR,
    PROTECTED_FLANK_GUARD_CHORD_FACTORS,
    RACK_MIN_TIP_THICKNESS_FACTOR,
    ROLLING_CONVERGENCE_AREA_FACTOR,
    ROLLING_MAX_PASSES,
    ROLLING_MIN_PHASES,
    ROLLING_PHASES_PER_TOOTH,
    ROLLING_STAGGER_OFFSETS,
    ROOT_CURVE_SAMPLES_PER_TOOTH,
    ROOT_SUPPORT_RADIUS_PITCH_FACTOR,
    SLIDING_SINE_FLOOR,
    SUPPORT_BUFFER_QUADRANT_SEGMENTS,
    TABULAR_ARRAY_DIMENSIONS,
    TOOTH_COUNT_ABS_TOLERANCE,
    TRIM_BUFFER_QUADRANT_SEGMENTS,
    UNDERCUT_FLANK_SEARCH_PITCHES,
    VERIFICATION_MIN_CLOSED_PHASES,
    VERIFICATION_MIN_OPEN_PHASES,
    VERIFICATION_PHASES_PER_TOOTH,
)

FloatArray = NDArray[np.float64]
_MAX_GEOMETRY_WORKERS = max(1, min(MAX_GEOMETRY_WORKERS, os.cpu_count() or 1))


def _floating_tolerance(*values: float) -> float:
    """Return a ULP-scale guard for dimensionless/angle comparisons."""

    scale = max(1.0, *(abs(value) for value in values))
    return FLOAT_COMPARISON_ULPS * np.finfo(float).eps * scale


def _length_tolerance(module: float, factor: float) -> float:
    """Convert a dimensionless policy factor to the current length unit."""

    return factor * module


def _analysis_interval_count(input_sample_count: int) -> int:
    """Resolve at least two analysis intervals per input Hermite segment."""

    count = max(
        MIN_CENTRODE_ANALYSIS_INTERVALS,
        CENTRODE_ANALYSIS_INTERVALS_PER_INPUT_SEGMENT * input_sample_count,
    )
    return count + count % 2


def _samples_for_pitch_span(low: float, high: float, pitch: float) -> int:
    """Derive an endpoint-inclusive intersection grid from its pitch span."""

    intervals = math.ceil(abs(high - low) / pitch * INTERSECTION_SAMPLES_PER_PITCH)
    return max(INTERSECTION_MIN_SAMPLES, intervals + 1)


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
    clearance_factor: float
    domain_start: float
    domain_end: float
    active_start: float
    active_end: float
    period: float
    cycle_delta: float
    open_: bool
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
    hybrid_undercut: bool


@dataclass(frozen=True)
class _ProtectedFlankSpan:
    driven: bool
    phase: float
    sign: int
    start_arc: float
    end_arc: float
    tip_arc: float


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
    maximum_hybrid_connector_length: float
    maximum_fillet_root_residual: float
    minimum_flank_regular_factor: float
    undercut_count: int
    hybrid_undercut_count: int
    protected_flanks: tuple[_ProtectedFlankSpan, ...]


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

    def shift_residual_bounds(
        self, shift: float, expected_value_delta: float
    ) -> tuple[float, float, float]:
        """Bound shift residuals over every polynomial segment.

        The union of the original and shifted knot grids partitions the domain
        into intervals on which each residual is a polynomial.  A degree-sized
        Chebyshev transform reconstructs that polynomial; the sum of absolute
        Chebyshev coefficients bounds it over the complete interval.  Unlike a
        fixed point grid, this cannot hide a compatible-grid alias between
        samples.
        """

        if not self.periodic:
            raise ValueError("Shift certification requires a periodic series")

        count = len(self.values)
        knots = self.domain_start + self.step * np.arange(count + 1)
        shifted_knots = self.domain_start + np.mod(
            knots[:-1] - shift - self.domain_start,
            self.period,
        )
        candidates = np.sort(
            np.concatenate(
                (
                    np.asarray([self.domain_start, self.domain_end]),
                    knots[1:-1],
                    shifted_knots,
                )
            )
        )
        merge_tolerance = _floating_tolerance(
            self.domain_start, self.domain_end, self.step
        )
        breakpoints = [float(candidates[0])]
        for candidate in candidates[1:]:
            value = float(candidate)
            if value - breakpoints[-1] > merge_tolerance:
                breakpoints.append(value)
            else:
                breakpoints[-1] = max(breakpoints[-1], value)
        if self.domain_end - breakpoints[-1] > merge_tolerance:
            breakpoints.append(self.domain_end)
        else:
            breakpoints[-1] = self.domain_end

        edges = np.asarray(breakpoints)
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        half_widths = 0.5 * (edges[1:] - edges[:-1])
        residual_bounds: list[float] = []
        polynomial_degree = len(self.coefficients) - 1
        for derivative in range(3):
            degree = polynomial_degree - derivative
            node_count = degree + 1
            nodes = np.cos(
                math.pi * (2.0 * np.arange(node_count) + 1.0) / (2.0 * node_count)
            )
            parameters = midpoints[:, None] + half_widths[:, None] * nodes[None, :]
            residual = np.asarray(self(parameters + shift, derivative)) - np.asarray(
                self(parameters, derivative)
            )
            if derivative == 0:
                residual = residual - expected_value_delta
            transform = np.linalg.inv(np.polynomial.chebyshev.chebvander(nodes, degree))
            coefficients = residual @ transform.T
            residual_bounds.append(float(np.max(np.sum(np.abs(coefficients), axis=1))))
        return residual_bounds[0], residual_bounds[1], residual_bounds[2]


class _IntegralTable:
    """Dense cumulative Simpson table with periodic continuation."""

    def __init__(
        self,
        function: Callable[[float | FloatArray], float | FloatArray],
        domain_start: float,
        domain_end: float,
        periodic: bool,
        interval_count: int = INTEGRATION_INTERVAL_COUNT,
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
            tolerance = _floating_tolerance(
                self.domain_start,
                self.domain_end,
                float(np.min(values)),
                float(np.max(values)),
            )
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
    if len(points) < MIN_CLOSED_RING_COORDINATE_COUNT:
        raise RuntimeError("Geometry operation left fewer than three boundary points")
    original_points = points.copy()

    # GEOS can retain nearly coincident and almost-collinear overlay vertices.
    points = points[:-1]
    duplicate_tolerance = tolerance * OUTLINE_DUPLICATE_TOLERANCE_FRACTION
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= duplicate_tolerance):
        # The usual path has no adjacent duplicates. Retain the sequential
        # filter for the uncommon case because a run of close points must be
        # compared with the last point that survived, not merely its neighbor.
        unique = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point - unique[-1]) > duplicate_tolerance:
                unique.append(point)
        points = np.asarray(unique, dtype=float)

    changed = True
    while changed and len(points) > MIN_POLYGON_VERTEX_COUNT:
        previous = np.roll(points, 1, axis=0)
        following = np.roll(points, -1, axis=0)
        incoming = points - previous
        outgoing = following - points
        lengths = np.linalg.norm(incoming, axis=1) + np.linalg.norm(outgoing, axis=1)
        twice_area = np.abs(
            incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
        )
        forward = np.einsum("ij,ij->i", incoming, outgoing) >= 0.0
        removable = (
            forward
            & (lengths > 0.0)
            & (twice_area < tolerance * OUTLINE_COLLINEAR_TOLERANCE_FRACTION * lengths)
        )
        changed = bool(np.any(removable))
        keep = ~removable
        points = points[keep]
    if len(points) < MIN_POLYGON_VERTEX_COUNT:
        raise RuntimeError("Geometry cleanup removed the complete boundary")
    points = np.vstack((points, points[0]))
    if not Polygon(points[:-1]).is_valid:
        # Removing an almost-collinear vertex can still close a very narrow
        # Boolean feature across its opposite edge.  The GEOS exterior supplied
        # on entry is valid, so retain it when simplification changes topology.
        points = original_points
    if _signed_area(points) < 0.0:
        points = points[::-1].copy()
    return points


def _transform_outline(
    points: FloatArray,
    angle: float,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
) -> Polygon:
    vertices = points[:-1]
    cosine = math.cos(angle)
    sine = math.sin(angle)
    transformed = np.empty_like(vertices)
    transformed[:, 0] = cosine * vertices[:, 0] - sine * vertices[:, 1] + translate_x
    transformed[:, 1] = sine * vertices[:, 0] + cosine * vertices[:, 1] + translate_y
    return Polygon(transformed)


def _read_sample_table(path: Path, columns: int) -> FloatArray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if values.ndim != TABULAR_ARRAY_DIMENSIONS or values.shape[1] != columns:
        raise ValueError(f"Expected {columns} columns in {path}")
    if len(values) < MIN_SAMPLE_TABLE_ROWS or not np.all(np.isfinite(values)):
        raise ValueError(
            f"Sample table {path} must contain at least "
            f"{MIN_SAMPLE_TABLE_ROWS} finite rows"
        )
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
    clearance_factor: float,
    domain_start: float,
    domain_end: float,
    active_start: float,
    active_end: float,
    period: float,
    cycle_delta: float,
    open_: bool,
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
        clearance_factor=clearance_factor,
        domain_start=domain_start,
        domain_end=domain_end,
        active_start=active_start,
        active_end=active_end,
        period=period,
        cycle_delta=cycle_delta,
        open_=open_,
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
    analysis_intervals = _analysis_interval_count(len(config.samples))
    check_phi = np.linspace(
        config.domain_start, config.domain_end, analysis_intervals + 1
    )
    radii = np.asarray(law(check_phi), dtype=float)
    if np.any(radii <= 0.0) or not np.all(np.isfinite(radii)):
        raise ValueError("Centrode radius must stay positive and finite")
    maximum_radius = float(np.max(radii))
    integration_start = config.active_start if config.open_ else config.domain_start
    integration_end = (
        config.active_end if config.open_ else config.domain_start + config.period
    )

    integration_phi = np.linspace(
        integration_start, integration_end, analysis_intervals + 1
    )
    integration_radii = np.asarray(law(integration_phi), dtype=float)

    def cycle_advance(center: float) -> float:
        return float(
            simpson(integration_radii / (center - integration_radii), x=integration_phi)
        )

    center = config.reference_center_distance
    if center == 0.0:
        lower_factor = 1.0 + _floating_tolerance(1.0)
        upper_factor = 2.0
        while cycle_advance(upper_factor * maximum_radius) > config.cycle_delta:
            upper_factor *= 2.0
            if not math.isfinite(upper_factor):
                raise ValueError("Could not solve a finite centrode center distance")
        center_factor = brentq(
            lambda factor: cycle_advance(factor * maximum_radius) - config.cycle_delta,
            lower_factor,
            upper_factor,
            xtol=INTERSECTION_SOLVER_TOLERANCE,
            rtol=max(
                INTERSECTION_SOLVER_TOLERANCE,
                4.0 * np.finfo(float).eps,
            ),
        )
        center = center_factor * maximum_radius
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
        _analysis_interval_count(sample_count),
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
        self.nominal_dedendum = config.dedendum_factor * config.module
        self.clearance = config.clearance_factor * config.module
        # The analytic rack closure already has a nominal addendum/dedendum
        # difference. Deepen it only when that difference is smaller than the
        # requested minimum clearance.
        self.dedendum = max(
            self.nominal_dedendum,
            self.addendum + self.clearance,
        )
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
                exact_driven_teeth,
                self.driven_teeth,
                rel_tol=TOOTH_COUNT_ABS_TOLERANCE,
                abs_tol=TOOTH_COUNT_ABS_TOLERANCE,
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
        if config.teeth < MIN_TEETH or config.module <= 0.0:
            raise ValueError("Invalid tooth count or module")
        maximum_pressure_angle = math.radians(MAX_PRESSURE_ANGLE_DEG)
        if not 0.0 < self.alpha < maximum_pressure_angle:
            raise ValueError(
                "Pressure angle must be between 0 and "
                f"{MAX_PRESSURE_ANGLE_DEG:g} degrees"
            )
        if not math.isfinite(self.clearance) or self.clearance < 0.0:
            raise ValueError("Clearance factor must be finite and nonnegative")
        if not (
            self.addendum > 0.0
            and self.dedendum > self.fillet_radius
            and self.fillet_radius >= 0.0
        ):
            raise ValueError("Invalid cutter dimensions")
        pitch = math.pi * config.module
        minimum_tip_half_width = RACK_MIN_TIP_THICKNESS_FACTOR * config.module
        if (
            0.25 * pitch - self.dedendum * math.tan(self.alpha)
            <= minimum_tip_half_width
        ):
            raise ValueError(
                "Rack cutter tip is too narrow; reduce dedendum or pressure angle"
            )
        analysis_intervals = _analysis_interval_count(len(config.samples))
        check_phi = np.linspace(
            config.domain_start, config.domain_end, analysis_intervals + 1
        )
        ratios = np.asarray(self._psi(check_phi, 1), dtype=float)
        if (
            not np.all(np.isfinite(ratios))
            or np.any(ratios <= MIN_MOTION_RATIO)
            or np.any(ratios >= MAX_MOTION_RATIO)
        ):
            raise ValueError("Motion law is not a bounded orientation-preserving map")

    def _validate_closed_motion(self) -> None:
        value_residual, first_residual, second_residual = (
            self.motion.shift_residual_bounds(
                self.driven_cycle,
                2.0 * math.pi,
            )
        )
        scale = max(1.0, abs(self.config.cycle_delta))
        if (
            value_residual > PERIODIC_VALUE_REL_TOLERANCE * scale
            or first_residual > PERIODIC_FIRST_DERIVATIVE_TOLERANCE
            or second_residual > PERIODIC_SECOND_DERIVATIVE_TOLERANCE
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
        base_intervals = _analysis_interval_count(len(self.config.samples))
        reference_span = (
            self.config.period
            if self.closed
            else self.config.domain_end - self.config.domain_start
        )
        interval_count = max(
            base_intervals,
            math.ceil(base_intervals * span / reference_span),
        )
        phi = np.linspace(start, start + span, interval_count + 1)
        drive_curvature = self._drive_curvature(phi)
        driven_curvature = self._driven_curvature(phi)
        drive_radius = np.asarray(self._drive_radius(phi))
        driven_radius = np.asarray(self._driven_radius(phi))
        self.maximum_drive_curvature = float(np.max(drive_curvature))
        self.minimum_driven_curvature = float(np.min(driven_curvature))
        self.maximum_pitch_radius = float(np.max([drive_radius, driven_radius]))
        self.minimum_pitch_radius = float(np.min([drive_radius, driven_radius]))
        curvature_tolerance = CENTRODE_CONVEXITY_CURVATURE_FACTOR / self.config.module
        self.centrodes_are_convex = (
            self.maximum_drive_curvature <= curvature_tolerance
            and self.minimum_driven_curvature >= -curvature_tolerance
        )

    def _support_radius(self, pitch_radius_factor: float) -> float:
        """Choose a connected backing radius without consuming the pitch body."""

        desired = max(
            MIN_SUPPORT_RADIUS_MODULE_FACTOR * self.config.module,
            pitch_radius_factor * self.minimum_pitch_radius,
        )
        return min(
            desired,
            MAX_SUPPORT_RADIUS_PITCH_FACTOR * self.minimum_pitch_radius,
        )

    def _overlap_area_tolerance(self) -> float:
        """Return the per-phase GEOS overlap allowance in squared units."""

        return (
            OVERLAP_AREA_TOLERANCE_FACTOR
            * self.config.module**2
            * OVERLAP_CONTACT_PAIR_ALLOWANCE
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
            tolerance = _floating_tolerance(
                float(table_arc[0]),
                float(table_arc[-1]),
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
        tolerance = _floating_tolerance(cycle_arc)
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
        rack_coordinate = sign * 0.25 * pitch - (values - tooth_phase * pitch)
        direction = complex(math.cos(self.alpha), sign * math.sin(self.alpha))
        return centrode + rack_coordinate * tangent * direction * math.cos(self.alpha)

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
        rack_coordinate = sign * 0.25 * pitch - (values - tooth_phase * pitch)
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
        expected_flank_arc = tooth_phase * pitch + sign * 0.25 * pitch - target_lambda
        domain_low, domain_high = self._analytic_common_arc_bounds()
        flank_low = max(
            domain_low,
            expected_flank_arc - INTERSECTION_FLANK_HALF_WINDOW_PITCHES * pitch,
        )
        flank_high = min(
            domain_high,
            expected_flank_arc + INTERSECTION_FLANK_HALF_WINDOW_PITCHES * pitch,
        )
        offset_low = max(
            domain_low,
            tooth_phase * pitch - INTERSECTION_OFFSET_HALF_WINDOW_PITCHES * pitch,
        )
        offset_high = min(
            domain_high,
            tooth_phase * pitch + INTERSECTION_OFFSET_HALF_WINDOW_PITCHES * pitch,
        )
        flank_arc = np.linspace(
            flank_low,
            flank_high,
            _samples_for_pitch_span(flank_low, flank_high, pitch),
        )
        offset_arc = np.linspace(
            offset_low,
            offset_high,
            _samples_for_pitch_span(offset_low, offset_high, pitch),
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
        flank_line = LineString(np.column_stack((flank_points.real, flank_points.imag)))
        offset_line = LineString(
            np.column_stack((offset_points.real, offset_points.imag))
        )
        candidate_points = self._intersection_points(
            flank_line.intersection(offset_line)
        )
        if not candidate_points:
            candidate_points = [nearest_points(flank_line, offset_line)[0]]

        parameter_scale = self.config.module
        lower = np.asarray([flank_arc[0], offset_arc[0]], dtype=float) / parameter_scale
        upper = (
            np.asarray([flank_arc[-1], offset_arc[-1]], dtype=float) / parameter_scale
        )
        candidates: list[tuple[float, float, float]] = []
        for point in candidate_points:
            initial = (
                np.asarray(
                    [
                        self._line_parameter(flank_line, point, flank_arc),
                        self._line_parameter(offset_line, point, offset_arc),
                    ]
                )
                / parameter_scale
            )

            def residual(scaled_parameters: FloatArray) -> FloatArray:
                parameters = scaled_parameters * parameter_scale
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
                xtol=INTERSECTION_SOLVER_TOLERANCE,
                ftol=INTERSECTION_SOLVER_TOLERANCE,
                gtol=INTERSECTION_SOLVER_TOLERANCE,
                max_nfev=FLANK_OFFSET_SOLVER_MAX_EVALUATIONS,
            )
            geometric_residual = float(np.linalg.norm(residual(solution.x)))
            if solution.success and geometric_residual <= INTERSECTION_RESIDUAL_FACTOR:
                parameters = solution.x * parameter_scale
                candidates.append(
                    (
                        float(parameters[0]),
                        float(parameters[1]),
                        geometric_residual * parameter_scale,
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
                + INTERSECTION_CANDIDATE_OFFSET_WEIGHT
                * abs(candidate[1] - tooth_phase * pitch)
            ),
        )

    def _analytic_common_arc_bounds(self) -> tuple[float, float]:
        if self.closed:
            return -math.inf, math.inf
        return (
            self.center_distance
            * float(
                self.arc_integral.integral(self.active_start, self.config.domain_start)
            ),
            self.center_distance
            * float(
                self.arc_integral.integral(self.active_start, self.config.domain_end)
            ),
        )

    def _analytic_singular_arcs(
        self,
        *,
        driven: bool,
        tooth_phase: float,
        sign: int,
        start_arc: float,
        end_arc: float,
    ) -> tuple[float, ...]:
        """Locate every cusp on a candidate working-flank interval."""

        pitch = math.pi * self.config.module
        phase_arc = tooth_phase * pitch
        low = min(start_arc, end_arc)
        high = max(start_arc, end_arc)

        def equation(common_arc: float) -> float:
            phi = self._phi_from_common_arc(np.asarray([common_arc]))[0]
            curvature = float(
                self._driven_curvature(np.asarray([phi]))[0]
                if driven
                else self._drive_curvature(np.asarray([phi]))[0]
            )
            rack_coordinate = sign * 0.25 * pitch - (common_arc - phase_arc)
            return rack_coordinate * curvature - sign * math.tan(self.alpha)

        parameter_tolerance = CUSP_PARAMETER_DEDUP_FACTOR * self.config.module
        solver_parameter_tolerance = max(
            INTERSECTION_SOLVER_TOLERANCE * self.config.module,
            _floating_tolerance(low, high),
        )
        previous_roots: list[float] | None = None
        roots: list[float] = []
        sample_count = CUSP_INITIAL_SAMPLES
        while True:
            samples = np.linspace(low, high, sample_count + 1)
            phi = self._phi_from_common_arc(samples)
            curvature = (
                self._driven_curvature(phi) if driven else self._drive_curvature(phi)
            )
            rack_coordinate = sign * 0.25 * pitch - (samples - phase_arc)
            values = rack_coordinate * curvature - sign * math.tan(self.alpha)
            candidates: list[float] = []
            finite = np.isfinite(values)
            lhs = values[:-1]
            rhs = values[1:]
            exact_root = finite[:-1] & (np.abs(lhs) <= CUSP_EQUATION_TOLERANCE)
            candidates.extend(float(value) for value in samples[:-1][exact_root])
            sign_change = finite[:-1] & finite[1:] & ~exact_root & (lhs * rhs < 0.0)
            for index in np.flatnonzero(sign_change):
                candidates.append(
                    float(
                        brentq(
                            equation,
                            float(samples[index]),
                            float(samples[index + 1]),
                            xtol=solver_parameter_tolerance,
                            rtol=INTERSECTION_SOLVER_TOLERANCE,
                        )
                    )
                )

            # A tangent/double root does not change sign. Minimize |f| around
            # every sampled local minimum so such a cusp is still discoverable.
            magnitudes = np.abs(values)
            local_minimum = (
                finite[1:-1]
                & (magnitudes[1:-1] <= magnitudes[:-2])
                & (magnitudes[1:-1] <= magnitudes[2:])
            )
            for index in np.flatnonzero(local_minimum) + 1:
                minimum = minimize_scalar(
                    lambda value: abs(equation(float(value))),
                    bounds=(
                        float(samples[index - 1]),
                        float(samples[index + 1]),
                    ),
                    method="bounded",
                    options={"xatol": solver_parameter_tolerance},
                )
                if minimum.success and float(minimum.fun) <= CUSP_EQUATION_TOLERANCE:
                    candidates.append(float(minimum.x))

            if finite[-1] and abs(float(values[-1])) <= CUSP_EQUATION_TOLERANCE:
                candidates.append(float(samples[-1]))
            candidates.sort()
            roots = []
            for root in candidates:
                if not roots or abs(root - roots[-1]) > parameter_tolerance:
                    roots.append(root)

            stable = (
                bool(roots)
                and previous_roots is not None
                and len(roots) == len(previous_roots)
                and all(
                    abs(current - previous) <= parameter_tolerance
                    for current, previous in zip(roots, previous_roots)
                )
            )
            if stable or sample_count >= CUSP_MAX_SAMPLES:
                break
            previous_roots = roots
            sample_count = min(2 * sample_count, CUSP_MAX_SAMPLES)

        return tuple(roots)

    def _analytic_singular_arc(
        self,
        *,
        driven: bool,
        tooth_phase: float,
        sign: int,
        contact_arc: float,
        flank_tip_arc: float,
    ) -> float | None:
        """Return the cusp nearest the nominal fillet contact, if any.

        Kept as a small compatibility wrapper for diagnostics.  Construction
        uses all roots returned by :meth:`_analytic_singular_arcs`.
        """

        roots = self._analytic_singular_arcs(
            driven=driven,
            tooth_phase=tooth_phase,
            sign=sign,
            start_arc=contact_arc,
            end_arc=flank_tip_arc,
        )
        if not roots:
            return None
        return min(roots, key=lambda root: abs(root - contact_arc))

    def _analytic_curve_intersections(
        self,
        lhs: Callable[[FloatArray], NDArray[np.complex128]],
        lhs_bounds: tuple[float, float],
        rhs: Callable[[FloatArray], NDArray[np.complex128]],
        rhs_bounds: tuple[float, float],
    ) -> list[tuple[float, float, float]]:
        """Find curve intersections with GEOS and refine them with SciPy."""

        pitch = math.pi * self.config.module
        lhs_parameters = np.linspace(
            *lhs_bounds,
            _samples_for_pitch_span(*lhs_bounds, pitch),
        )
        rhs_parameters = np.linspace(
            *rhs_bounds,
            _samples_for_pitch_span(*rhs_bounds, pitch),
        )
        lhs_points = lhs(lhs_parameters)
        rhs_points = rhs(rhs_parameters)
        lhs_line = LineString(np.column_stack((lhs_points.real, lhs_points.imag)))
        rhs_line = LineString(np.column_stack((rhs_points.real, rhs_points.imag)))
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

        parameter_scale = self.config.module
        lower = (
            np.asarray([min(lhs_bounds), min(rhs_bounds)], dtype=float)
            / parameter_scale
        )
        upper = (
            np.asarray([max(lhs_bounds), max(rhs_bounds)], dtype=float)
            / parameter_scale
        )
        refined: list[tuple[float, float, float]] = []
        for initial in candidates:

            def residual(scaled_parameters: FloatArray) -> FloatArray:
                parameters = scaled_parameters * parameter_scale
                difference = (
                    lhs(np.asarray([parameters[0]]))[0]
                    - rhs(np.asarray([parameters[1]]))[0]
                ) / self.config.module
                return np.asarray([difference.real, difference.imag])

            solution = least_squares(
                residual,
                np.asarray(initial) / parameter_scale,
                bounds=(lower, upper),
                xtol=INTERSECTION_SOLVER_TOLERANCE,
                ftol=INTERSECTION_SOLVER_TOLERANCE,
                gtol=INTERSECTION_SOLVER_TOLERANCE,
                max_nfev=INTERSECTION_SOLVER_MAX_EVALUATIONS,
            )
            geometric_residual = (
                float(np.linalg.norm(residual(solution.x))) * parameter_scale
            )
            if not solution.success or geometric_residual > _length_tolerance(
                self.config.module, INTERSECTION_RESIDUAL_FACTOR
            ):
                continue
            parameters = solution.x * parameter_scale
            candidate = (
                float(parameters[0]),
                float(parameters[1]),
                geometric_residual,
            )
            parameter_tolerance = _length_tolerance(
                self.config.module, INTERSECTION_PARAMETER_DEDUP_FACTOR
            )
            if not any(
                abs(candidate[0] - existing[0]) < parameter_tolerance
                and abs(candidate[1] - existing[1]) < parameter_tolerance
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
        fillet_dedendum_lambda = root_height * math.tan(
            self.alpha
        ) + self.fillet_radius / math.cos(self.alpha)
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
        singular_arcs = self._analytic_singular_arcs(
            driven=driven,
            tooth_phase=tooth_phase,
            sign=sign,
            start_arc=contact_arc,
            end_arc=flank_tip_arc,
        )
        free = not singular_arcs
        hybrid_undercut = False
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
                flank_extent = phase_arc - UNDERCUT_FLANK_SEARCH_PITCHES * pitch
            else:
                flank_extent = phase_arc + UNDERCUT_FLANK_SEARCH_PITCHES * pitch
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
                if abs(intersection[0] - contact_arc)
                > INTERSECTION_CONTACT_EXCLUSION_PITCHES * pitch
                or abs(intersection[1] - contact_arc)
                > INTERSECTION_CONTACT_EXCLUSION_PITCHES * pitch
            ]
            regular_intersections = [
                intersection
                for intersection in nontrivial
                if not self._analytic_singular_arcs(
                    driven=driven,
                    tooth_phase=tooth_phase,
                    sign=sign,
                    start_arc=intersection[0],
                    end_arc=flank_tip_arc,
                )
            ]
            if regular_intersections:
                selected = min(
                    regular_intersections,
                    key=lambda intersection: min(
                        abs(intersection[0] - singular) for singular in singular_arcs
                    ),
                )
                flank_transition_arc = selected[0]
                fillet_transition_arc = selected[1]
                transition_residual = selected[2]
            else:
                # The rounded rack-tip branch cannot provide a regular
                # transition.  Preserve only the tip-side regular component;
                # the gap to the nominal fillet becomes sacrificial stock for
                # the opposing-gear rolling cutter.
                singular_arc = min(
                    singular_arcs,
                    key=lambda singular: abs(singular - flank_tip_arc),
                )
                direction_to_tip = math.copysign(1.0, flank_tip_arc - singular_arc)
                clearance = HYBRID_CUSP_CLEARANCE_FACTOR * self.config.module
                flank_transition_arc = singular_arc + direction_to_tip * clearance
                if not (
                    min(singular_arc, flank_tip_arc)
                    < flank_transition_arc
                    < max(singular_arc, flank_tip_arc)
                ):
                    member = "driven" if driven else "drive"
                    raise RuntimeError(
                        f"{member.capitalize()} flank cusp leaves no regular "
                        f"working interval for tooth {tooth_phase:g}, sign {sign}"
                    )
                fillet_transition_arc = contact_arc
                hybrid_undercut = True

        retained_singular_arcs = self._analytic_singular_arcs(
            driven=driven,
            tooth_phase=tooth_phase,
            sign=sign,
            start_arc=flank_transition_arc,
            end_arc=flank_tip_arc,
        )
        if retained_singular_arcs:
            member = "driven" if driven else "drive"
            raise RuntimeError(
                f"{member.capitalize()} retained working flank contains a cusp "
                f"for tooth {tooth_phase:g}, sign {sign}"
            )

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
            hybrid_undercut=hybrid_undercut,
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
        endpoint_phi = self._phi_from_common_arc(np.asarray([start, end], dtype=float))
        phi_span = abs(float(endpoint_phi[1] - endpoint_phi[0]))
        sample_count = max(
            minimum_samples,
            math.ceil(
                CURVE_SAMPLES_PER_INPUT_RADIAN_FACTOR * phi_span * samples_per_radian
            ),
        )
        target_error = _length_tolerance(
            self.config.module, ANALYTIC_CHORD_TOLERANCE_FACTOR
        )
        while True:
            parameters = np.linspace(start, end, sample_count + 1)
            points = function(parameters)
            midpoint_parameters = 0.5 * (parameters[:-1] + parameters[1:])
            exact_midpoint = function(midpoint_parameters)
            segment_start = points[:-1]
            segment = points[1:] - segment_start
            denominator = np.maximum(np.abs(segment) ** 2, np.finfo(float).tiny)
            projection = np.clip(
                np.real((exact_midpoint - segment_start) * np.conj(segment))
                / denominator,
                0.0,
                1.0,
            )
            chord_error = float(
                np.max(np.abs(exact_midpoint - (segment_start + projection * segment)))
            )
            if (
                chord_error <= target_error
                or sample_count >= MAX_ANALYTIC_CURVE_SAMPLES
            ):
                return points, sample_count, chord_error
            sample_count = min(2 * sample_count, MAX_ANALYTIC_CURVE_SAMPLES)

    def _sample_analytic_flank(
        self,
        *,
        driven: bool,
        geometry: _AnalyticFlankGeometry,
        samples_per_radian: int,
    ) -> tuple[NDArray[np.complex128], int, float, float, float, float]:
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
            minimum_samples=MIN_FLANK_CURVE_SAMPLES,
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
        rack_coordinate = geometry.sign * 0.25 * pitch - (common_arc - phase_arc)
        direction = complex(math.cos(self.alpha), geometry.sign * math.sin(self.alpha))
        local = (points - centrode) / tangent
        expected = rack_coordinate * direction * math.cos(self.alpha)
        envelope_residual = float(np.max(np.abs(local - expected)))

        midpoint_arc = 0.5 * (common_arc[:-1] + common_arc[1:])
        midpoint_phi = self._phi_from_common_arc(midpoint_arc)
        midpoint_tangent = (
            np.asarray(self._driven_tangent(midpoint_phi), dtype=np.complex128)
            if driven
            else np.asarray(self._drive_tangent(midpoint_phi), dtype=np.complex128)
        )
        midpoint_curvature = (
            self._driven_curvature(midpoint_phi)
            if driven
            else self._drive_curvature(midpoint_phi)
        )
        midpoint_lambda = geometry.sign * 0.25 * pitch - (midpoint_arc - phase_arc)
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
        derivative_magnitude = np.abs(derivative)
        minimum_regular_factor = float(np.min(derivative_magnitude))
        if minimum_regular_factor <= ANALYTIC_REGULAR_DERIVATIVE_TOLERANCE:
            member = "driven" if driven else "drive"
            raise RuntimeError(
                f"{member.capitalize()} retained working flank is singular or "
                f"numerically unresolved for tooth {geometry.phase:g}, "
                f"sign {geometry.sign}"
            )
        normalized_generator = generator_tangent / np.abs(generator_tangent)
        normalized_derivative = derivative / derivative_magnitude
        tangency_residual = float(
            np.max(
                np.abs(np.imag(np.conj(normalized_generator) * normalized_derivative))
            )
        )
        return (
            points,
            sample_count,
            envelope_residual,
            tangency_residual,
            chord_error,
            minimum_regular_factor,
        )

    def _analytic_root_blank(
        self, driven: bool, samples_per_radian: int
    ) -> tuple[Geometry, int, float]:
        """Build a closed gear's material inward of the dedendum offset."""

        if not self.closed:
            raise RuntimeError("Closed root blank requested for an open gear")

        teeth = self.driven_teeth if driven else self.drive_teeth
        start_arc = 0.0
        end_arc = teeth * math.pi * self.config.module
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
            minimum_samples=max(
                MIN_ROOT_CURVE_SAMPLES,
                ROOT_CURVE_SAMPLES_PER_TOOTH * teeth,
            ),
        )
        root = make_valid(
            Polygon(np.column_stack((root_points.real, root_points.imag)))
        )
        hub_radius = self._support_radius(ROOT_SUPPORT_RADIUS_PITCH_FACTOR)
        hub = Point(0.0, 0.0).buffer(
            hub_radius, quad_segs=SUPPORT_BUFFER_QUADRANT_SEGMENTS
        )
        return (
            _clean_polygon(union_all([root, hub])),
            sample_count,
            chord_error,
        )

    def _generate_analytic_involute_gear(
        self, driven: bool, samples_per_radian: int
    ) -> _AnalyticGearResult:
        """Arrange exact flank, rack-tip, addendum, and dedendum curves."""

        teeth = self.driven_teeth if driven else self.drive_teeth
        phase_sides: list[tuple[float, int]] = (
            [(float(phase), sign) for phase in range(teeth + 1) for sign in (-1, 1)]
            if self.closed
            else [
                *[(phase + 0.5, sign) for phase in range(teeth) for sign in (-1, 1)],
                (teeth + 0.5, -1),
            ]
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
        maximum_hybrid_connector_length = 0.0
        maximum_fillet_root_residual = 0.0
        minimum_flank_regular_factor = math.inf
        undercut_count = 0
        hybrid_undercut_count = 0
        protected_flanks: list[_ProtectedFlankSpan] = []

        for tooth_phase, sign in phase_sides:
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
            # Closed profiles evaluate phase ``teeth`` as a periodic copy of
            # phase zero so the final body can join the first.  Keep that
            # geometry for arrangement, but do not count its two flanks twice.
            physical_flank = (
                tooth_phase < teeth if self.closed else tooth_phase < teeth + 0.5
            )
            if geometry.undercut and physical_flank:
                undercut_count += 1
            if geometry.hybrid_undercut and physical_flank:
                hybrid_undercut_count += 1
            flank, count, envelope, tangency, chord, regular_factor = (
                self._sample_analytic_flank(
                    driven=driven,
                    geometry=geometry,
                    samples_per_radian=samples_per_radian,
                )
            )
            flanks[tooth_phase, sign] = flank
            total_samples += count
            flank_sample_count += count
            maximum_envelope_residual = max(maximum_envelope_residual, envelope)
            maximum_tangency_residual = max(maximum_tangency_residual, tangency)
            maximum_chord_error = max(maximum_chord_error, chord)
            minimum_flank_regular_factor = min(
                minimum_flank_regular_factor, regular_factor
            )
            if physical_flank:
                active_low = 0.0
                active_high = teeth * math.pi * self.config.module
                span_low = max(
                    min(geometry.flank_transition_arc, geometry.flank_tip_arc),
                    active_low,
                )
                span_high = min(
                    max(geometry.flank_transition_arc, geometry.flank_tip_arc),
                    active_high,
                )
                if span_low < span_high:
                    protected_flanks.append(
                        _ProtectedFlankSpan(
                            driven=driven,
                            phase=tooth_phase,
                            sign=sign,
                            start_arc=span_low,
                            end_arc=span_high,
                            tip_arc=geometry.flank_tip_arc,
                        )
                    )
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
                minimum_samples=MIN_CLOSURE_CURVE_SAMPLES,
            )
            fillets[tooth_phase, sign] = fillet
            total_samples += count
            maximum_chord_error = max(maximum_chord_error, chord)

        if not self.closed:
            body_span = (
                float(self._psi(self.active_end)) - float(self._psi(self.active_start))
                if driven
                else self.active_end - self.active_start
            )
            if abs(body_span) >= 2.0 * math.pi - ANGLE_CLOSURE_TOLERANCE:
                raise ValueError(
                    "Open gear body span must be less than one body revolution"
                )
            active_start_arc = 0.0
            active_end_arc = teeth * math.pi * self.config.module
            boundary: list[complex] = []

            def append_clipped_curve(
                function: Callable[[FloatArray], NDArray[np.complex128]],
                start: float,
                end: float,
                *,
                minimum_samples: int,
            ) -> None:
                nonlocal total_samples, maximum_chord_error
                low = max(min(start, end), active_start_arc)
                high = min(max(start, end), active_end_arc)
                if low >= high:
                    return
                clipped_start, clipped_end = (
                    (low, high) if start <= end else (high, low)
                )
                piece, count, chord = self._sample_analytic_curve(
                    function,
                    clipped_start,
                    clipped_end,
                    samples_per_radian,
                    minimum_samples=minimum_samples,
                )
                total_samples += count
                maximum_chord_error = max(maximum_chord_error, chord)
                if boundary:
                    gap = abs(boundary[-1] - piece[0])
                    if gap <= _length_tolerance(
                        self.config.module, ANALYTIC_JOIN_TOLERANCE_FACTOR
                    ):
                        boundary.extend(piece[1:])
                        return
                boundary.extend(piece)

            involute = lambda phase, side: (
                lambda values: self._analytic_involute_points(
                    driven=driven,
                    tooth_phase=phase,
                    sign=side,
                    common_arc=values,
                )
            )
            fillet = lambda phase, side: (
                lambda values: self._analytic_fillet_points(
                    driven=driven,
                    tooth_phase=phase,
                    sign=side,
                    common_arc=values,
                )
            )
            offset = lambda height: (
                lambda values: self._analytic_offset_points(
                    driven=driven,
                    common_arc=values,
                    height=height,
                )
            )

            # An open gear is one ordered analytical boundary. Clip every
            # constituent curve in its rolling-arc parameter, rather than
            # intersecting a padded solid with radial sector faces. The
            # latter is only equivalent for circular constant-ratio gears.
            for tooth in range(teeth):
                tooth_phase = tooth + 0.5
                minus = geometries[tooth_phase, -1]
                plus = geometries[tooth_phase, 1]
                next_minus = geometries[tooth_phase + 1.0, -1]
                if driven:
                    append_clipped_curve(
                        involute(tooth_phase, -1),
                        minus.flank_tip_arc,
                        minus.flank_transition_arc,
                        minimum_samples=MIN_FLANK_CURVE_SAMPLES,
                    )
                    append_clipped_curve(
                        fillet(tooth_phase, -1),
                        minus.fillet_transition_arc,
                        minus.fillet_root_arc,
                        minimum_samples=MIN_CLOSURE_CURVE_SAMPLES,
                    )
                    append_clipped_curve(
                        offset(-self.dedendum),
                        minus.fillet_root_arc,
                        plus.fillet_root_arc,
                        minimum_samples=MIN_CLOSURE_CURVE_SAMPLES,
                    )
                    append_clipped_curve(
                        fillet(tooth_phase, 1),
                        plus.fillet_root_arc,
                        plus.fillet_transition_arc,
                        minimum_samples=MIN_CLOSURE_CURVE_SAMPLES,
                    )
                    append_clipped_curve(
                        involute(tooth_phase, 1),
                        plus.flank_transition_arc,
                        plus.flank_tip_arc,
                        minimum_samples=MIN_FLANK_CURVE_SAMPLES,
                    )
                    append_clipped_curve(
                        offset(self.addendum),
                        plus.addendum_tip_arc,
                        next_minus.addendum_tip_arc,
                        minimum_samples=MIN_CLOSURE_CURVE_SAMPLES,
                    )
                else:
                    append_clipped_curve(
                        fillet(tooth_phase, -1),
                        minus.fillet_root_arc,
                        minus.fillet_transition_arc,
                        minimum_samples=MIN_CLOSURE_CURVE_SAMPLES,
                    )
                    append_clipped_curve(
                        involute(tooth_phase, -1),
                        minus.flank_transition_arc,
                        minus.flank_tip_arc,
                        minimum_samples=MIN_FLANK_CURVE_SAMPLES,
                    )
                    append_clipped_curve(
                        offset(self.addendum),
                        minus.addendum_tip_arc,
                        plus.addendum_tip_arc,
                        minimum_samples=MIN_CLOSURE_CURVE_SAMPLES,
                    )
                    append_clipped_curve(
                        involute(tooth_phase, 1),
                        plus.flank_tip_arc,
                        plus.flank_transition_arc,
                        minimum_samples=MIN_FLANK_CURVE_SAMPLES,
                    )
                    append_clipped_curve(
                        fillet(tooth_phase, 1),
                        plus.fillet_transition_arc,
                        plus.fillet_root_arc,
                        minimum_samples=MIN_CLOSURE_CURVE_SAMPLES,
                    )
                    append_clipped_curve(
                        offset(-self.dedendum),
                        plus.fillet_root_arc,
                        next_minus.fillet_root_arc,
                        minimum_samples=MIN_CLOSURE_CURVE_SAMPLES,
                    )

            if len(boundary) < MIN_POLYGON_VERTEX_COUNT:
                raise RuntimeError(
                    "Generated open analytical boundary has too few points"
                )

            def inner_boundary(
                values: FloatArray,
            ) -> NDArray[np.complex128]:
                return OPEN_ANALYTIC_BACKING_RADIUS_PITCH_FACTOR * np.asarray(
                    self._driven_centrode(self._phi_from_common_arc(values))
                    if driven
                    else self._drive_centrode(self._phi_from_common_arc(values)),
                    dtype=np.complex128,
                )

            backing, count, chord = self._sample_analytic_curve(
                inner_boundary,
                active_end_arc,
                active_start_arc,
                samples_per_radian,
                minimum_samples=max(
                    MIN_ROOT_CURVE_SAMPLES,
                    ROOT_CURVE_SAMPLES_PER_TOOTH * teeth,
                ),
            )
            total_samples += count
            maximum_chord_error = max(maximum_chord_error, chord)
            if abs(boundary[-1] - backing[0]) > _length_tolerance(
                self.config.module, ANALYTIC_JOIN_TOLERANCE_FACTOR
            ):
                boundary.extend(backing)
            else:
                boundary.extend(backing[1:])
            if abs(boundary[-1] - boundary[0]) > _length_tolerance(
                self.config.module, ANALYTIC_CLOSURE_TOLERANCE_FACTOR
            ):
                boundary.append(boundary[0])
            else:
                boundary[-1] = boundary[0]

            coordinates = np.asarray(boundary, dtype=np.complex128)
            arranged = make_valid(
                Polygon(np.column_stack((coordinates.real, coordinates.imag)))
            )
            polygon = _clean_polygon(arranged)
            return _AnalyticGearResult(
                outline=_outline(
                    polygon,
                    _length_tolerance(
                        self.config.module,
                        GEOMETRY_LENGTH_TOLERANCE_FACTOR,
                    ),
                ),
                sample_count=total_samples,
                flank_sample_count=flank_sample_count,
                maximum_envelope_residual=maximum_envelope_residual,
                maximum_tangency_residual=maximum_tangency_residual,
                maximum_chord_error=maximum_chord_error,
                maximum_intersection_residual=(maximum_intersection_residual),
                maximum_join_gap=maximum_join_gap,
                maximum_hybrid_connector_length=(maximum_hybrid_connector_length),
                maximum_fillet_root_residual=(maximum_fillet_root_residual),
                minimum_flank_regular_factor=minimum_flank_regular_factor,
                undercut_count=undercut_count,
                hybrid_undercut_count=hybrid_undercut_count,
                protected_flanks=tuple(protected_flanks),
            )

        root_blank, count, chord = self._analytic_root_blank(driven, samples_per_radian)
        total_samples += count
        maximum_chord_error = max(maximum_chord_error, chord)
        tooth_bodies: list[Geometry] = [root_blank]

        def append_piece(
            ring: list[complex],
            points: NDArray[np.complex128],
            *,
            hybrid_connector: bool = False,
        ) -> None:
            nonlocal maximum_hybrid_connector_length, maximum_join_gap
            piece = np.asarray(points, dtype=np.complex128).copy()
            if ring:
                gap = abs(ring[-1] - piece[0])
                if hybrid_connector:
                    maximum_hybrid_connector_length = max(
                        maximum_hybrid_connector_length, gap
                    )
                    ring.extend(piece)
                    return
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
                append_piece(
                    ring,
                    flanks[tooth_phase, 1],
                    hybrid_connector=first.hybrid_undercut,
                )
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
                    minimum_samples=MIN_CLOSURE_CURVE_SAMPLES,
                )
                append_piece(ring, addendum)
                total_samples += count
                maximum_chord_error = max(maximum_chord_error, chord)
                append_piece(ring, flanks[next_phase, -1][::-1])
                append_piece(
                    ring,
                    fillets[next_phase, -1][::-1],
                    hybrid_connector=second.hybrid_undercut,
                )
                dedendum_start = second.fillet_root_arc
                dedendum_end = first.fillet_root_arc
            else:
                minus = geometries[tooth_phase, -1]
                plus = geometries[tooth_phase, 1]
                append_piece(ring, fillets[tooth_phase, -1])
                append_piece(
                    ring,
                    flanks[tooth_phase, -1],
                    hybrid_connector=minus.hybrid_undercut,
                )
                addendum, count, chord = self._sample_analytic_curve(
                    lambda values: self._analytic_offset_points(
                        driven=False,
                        common_arc=values,
                        height=self.addendum,
                    ),
                    minus.addendum_tip_arc,
                    plus.addendum_tip_arc,
                    samples_per_radian,
                    minimum_samples=MIN_CLOSURE_CURVE_SAMPLES,
                )
                append_piece(ring, addendum)
                total_samples += count
                maximum_chord_error = max(maximum_chord_error, chord)
                append_piece(ring, flanks[tooth_phase, 1][::-1])
                append_piece(
                    ring,
                    fillets[tooth_phase, 1][::-1],
                    hybrid_connector=plus.hybrid_undercut,
                )
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
                minimum_samples=MIN_CLOSURE_CURVE_SAMPLES,
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
                    Polygon(np.column_stack((coordinates.real, coordinates.imag)))
                )
            )

        arranged: Geometry = union_all(tooth_bodies)
        polygon = _clean_polygon(arranged)
        outline = _outline(
            polygon,
            _length_tolerance(self.config.module, GEOMETRY_LENGTH_TOLERANCE_FACTOR),
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
            maximum_hybrid_connector_length=maximum_hybrid_connector_length,
            maximum_fillet_root_residual=maximum_fillet_root_residual,
            minimum_flank_regular_factor=minimum_flank_regular_factor,
            undercut_count=undercut_count,
            hybrid_undercut_count=hybrid_undercut_count,
            protected_flanks=tuple(protected_flanks),
        )

    @staticmethod
    def _place_geometry(
        geometry: Geometry,
        angle: float,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
    ) -> Geometry:
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return affinity.affine_transform(
            geometry,
            [cosine, -sine, sine, cosine, translate_x, translate_y],
        )

    @staticmethod
    def _unplace_geometry(
        geometry: Geometry,
        angle: float,
        translate_x: float = 0.0,
        translate_y: float = 0.0,
    ) -> Geometry:
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return affinity.affine_transform(
            geometry,
            [
                cosine,
                sine,
                -sine,
                cosine,
                -cosine * translate_x - sine * translate_y,
                sine * translate_x - cosine * translate_y,
            ],
        )

    def _analytic_pitch_material(self, driven: bool) -> Polygon:
        """Return material inward of the pitch curve for root-only trimming."""

        teeth = self.driven_teeth if driven else self.drive_teeth
        start_arc = 0.0
        end_arc = teeth * math.pi * self.config.module
        sample_count = max(
            MIN_ROOT_CURVE_SAMPLES,
            ROOT_CURVE_SAMPLES_PER_TOOTH * teeth,
        )
        common_arc = np.linspace(start_arc, end_arc, sample_count + 1)
        pitch_points = self._analytic_offset_points(
            driven=driven,
            common_arc=common_arc,
            height=0.0,
        )
        if self.closed:
            pitch = make_valid(
                Polygon(np.column_stack((pitch_points.real, pitch_points.imag)))
            )
            hub_radius = self._support_radius(PITCH_MASK_SUPPORT_RADIUS_PITCH_FACTOR)
            hub = Point(0.0, 0.0).buffer(
                hub_radius,
                quad_segs=PITCH_MASK_BUFFER_QUADRANT_SEGMENTS,
            )
            return _clean_polygon(union_all([pitch, hub]))

        phi = self._phi_from_common_arc(common_arc)
        centrode = (
            np.asarray(self._driven_centrode(phi), dtype=np.complex128)
            if driven
            else np.asarray(self._drive_centrode(phi), dtype=np.complex128)
        )
        inner_radius = self._support_radius(PITCH_MASK_SUPPORT_RADIUS_PITCH_FACTOR)
        inner = centrode * inner_radius / np.abs(centrode)
        ring = np.concatenate((pitch_points, inner[::-1]))
        return _clean_polygon(
            make_valid(Polygon(np.column_stack((ring.real, ring.imag))))
        )

    def _protected_flank_points(
        self,
        span: _ProtectedFlankSpan,
        *,
        minimum_samples: int = MIN_CLOSURE_CURVE_SAMPLES,
    ) -> NDArray[np.complex128]:
        sample_count = max(
            minimum_samples,
            math.ceil(
                abs(span.end_arc - span.start_arc)
                / self.config.module
                * INTERSECTION_SAMPLES_PER_PITCH
                / math.pi
            ),
        )
        common_arc = np.linspace(
            span.start_arc,
            span.end_arc,
            sample_count + 1,
        )
        return self._analytic_involute_points(
            driven=span.driven,
            tooth_phase=span.phase,
            sign=span.sign,
            common_arc=common_arc,
        )

    def _protected_flank_guard(
        self,
        spans: tuple[_ProtectedFlankSpan, ...],
        target: Geometry,
    ) -> Geometry:
        """Return a target-interior material strip along retained flanks."""

        guard_width = _length_tolerance(
            self.config.module,
            PROTECTED_FLANK_GUARD_CHORD_FACTORS * ANALYTIC_CHORD_TOLERANCE_FACTOR,
        )
        guards = [
            LineString(
                np.column_stack(
                    (
                        (points := self._protected_flank_points(span)).real,
                        points.imag,
                    )
                )
            ).buffer(
                guard_width,
                quad_segs=TRIM_BUFFER_QUADRANT_SEGMENTS,
            )
            for span in spans
        ]
        if not guards:
            return Point(0.0, 0.0).buffer(0.0)
        return union_all(guards).intersection(target)

    def _rolling_core_guard(self) -> Geometry:
        """Protect the connected support core from mutual cutter erosion."""

        return Point(0.0, 0.0).buffer(
            self._support_radius(ROOT_SUPPORT_RADIUS_PITCH_FACTOR),
            quad_segs=SUPPORT_BUFFER_QUADRANT_SEGMENTS,
        )

    def _clip_protected_flanks_to_boundary(
        self,
        spans: tuple[_ProtectedFlankSpan, ...],
        outline: FloatArray,
        *,
        chord_error: float,
    ) -> tuple[tuple[_ProtectedFlankSpan, ...], int]:
        """Keep only the exposed component connected to each addendum tip."""

        boundary = LineString(outline)
        boundary_tolerance = chord_error + _length_tolerance(
            self.config.module,
            ANALYTIC_CHORD_TOLERANCE_FACTOR + GEOMETRY_LENGTH_TOLERANCE_FACTOR,
        )
        clipped: list[_ProtectedFlankSpan] = []
        clipped_count = 0
        sample_count = 2 * MIN_CLOSURE_CURVE_SAMPLES
        for span in spans:
            common_arc = np.linspace(
                span.start_arc,
                span.end_arc,
                sample_count + 1,
            )
            points = self._analytic_involute_points(
                driven=span.driven,
                tooth_phase=span.phase,
                sign=span.sign,
                common_arc=common_arc,
            )
            exposed = np.asarray(
                [
                    boundary.distance(Point(point.real, point.imag))
                    <= boundary_tolerance
                    for point in points
                ],
                dtype=bool,
            )
            tip_index = int(np.argmin(np.abs(common_arc - span.tip_arc)))
            if not exposed[tip_index]:
                member = "driven" if span.driven else "drive"
                raise RuntimeError(
                    f"{member.capitalize()} flank addendum end is not exposed "
                    f"for tooth {span.phase:g}, sign {span.sign}"
                )
            low_index = tip_index
            high_index = tip_index
            while low_index > 0 and exposed[low_index - 1]:
                low_index -= 1
            while high_index < sample_count and exposed[high_index + 1]:
                high_index += 1
            if low_index == high_index:
                member = "driven" if span.driven else "drive"
                raise RuntimeError(
                    f"{member.capitalize()} flank has no exposed working span "
                    f"for tooth {span.phase:g}, sign {span.sign}"
                )
            clipped_span = replace(
                span,
                start_arc=float(common_arc[low_index]),
                end_arc=float(common_arc[high_index]),
            )
            if low_index != 0 or high_index != sample_count:
                clipped_count += 1
            clipped.append(clipped_span)
        return tuple(clipped), clipped_count

    def _verify_protected_flanks(
        self,
        spans: tuple[_ProtectedFlankSpan, ...],
        outline: FloatArray,
        *,
        chord_error: float,
    ) -> tuple[float, float]:
        """Sample-check regularity and exposure of retained exact flanks."""

        boundary = LineString(outline)
        boundary_tolerance = chord_error + _length_tolerance(
            self.config.module,
            ANALYTIC_CHORD_TOLERANCE_FACTOR + GEOMETRY_LENGTH_TOLERANCE_FACTOR,
        )
        maximum_boundary_error = 0.0
        minimum_regular_factor = math.inf
        pitch = math.pi * self.config.module
        for span in spans:
            singular_arcs = self._analytic_singular_arcs(
                driven=span.driven,
                tooth_phase=span.phase,
                sign=span.sign,
                start_arc=span.start_arc,
                end_arc=span.end_arc,
            )
            if singular_arcs:
                member = "driven" if span.driven else "drive"
                raise RuntimeError(
                    f"{member.capitalize()} protected working flank contains a "
                    f"cusp for tooth {span.phase:g}, sign {span.sign}"
                )
            common_arc = np.linspace(
                span.start_arc,
                span.end_arc,
                MIN_CLOSURE_CURVE_SAMPLES + 1,
            )
            points = self._analytic_involute_points(
                driven=span.driven,
                tooth_phase=span.phase,
                sign=span.sign,
                common_arc=common_arc,
            )
            line = LineString(np.column_stack((points.real, points.imag)))
            if not line.is_simple:
                member = "driven" if span.driven else "drive"
                raise RuntimeError(
                    f"{member.capitalize()} protected working flank "
                    f"self-intersects for tooth {span.phase:g}, sign {span.sign}"
                )
            phi = self._phi_from_common_arc(common_arc)
            curvature = (
                self._driven_curvature(phi)
                if span.driven
                else self._drive_curvature(phi)
            )
            rack_coordinate = span.sign * 0.25 * pitch - (
                common_arc - span.phase * pitch
            )
            regular_factor = np.abs(
                rack_coordinate * curvature * math.cos(self.alpha)
                - span.sign * math.sin(self.alpha)
            )
            span_minimum = float(np.min(regular_factor))
            minimum_regular_factor = min(minimum_regular_factor, span_minimum)
            if span_minimum <= ANALYTIC_REGULAR_DERIVATIVE_TOLERANCE:
                member = "driven" if span.driven else "drive"
                raise RuntimeError(
                    f"{member.capitalize()} protected working flank is singular "
                    f"for tooth {span.phase:g}, sign {span.sign}"
                )
            span_boundary_error = max(
                boundary.distance(Point(point.real, point.imag)) for point in points
            )
            maximum_boundary_error = max(maximum_boundary_error, span_boundary_error)
            if span_boundary_error > boundary_tolerance:
                member = "driven" if span.driven else "drive"
                raise RuntimeError(
                    f"{member.capitalize()} protected working flank is not "
                    f"exposed on the finished boundary for tooth {span.phase:g}, "
                    f"sign {span.sign} (distance {span_boundary_error:.9g})"
                )
        return maximum_boundary_error, minimum_regular_factor

    def _verify_protected_contact_coverage(
        self,
        drive_spans: tuple[_ProtectedFlankSpan, ...],
        driven_spans: tuple[_ProtectedFlankSpan, ...],
    ) -> tuple[int, float, int, float]:
        """Measure exact paired-flank coverage at uncorrected sampled poses."""

        phase_count = (
            max(
                VERIFICATION_MIN_CLOSED_PHASES,
                VERIFICATION_PHASES_PER_TOOTH
                * max(self.drive_teeth, self.driven_teeth),
            )
            if self.closed
            else VERIFICATION_MIN_OPEN_PHASES
        )
        fraction_grids = [
            (
                (np.arange(phase_count) + offset) / phase_count
                if self.closed
                else np.clip(
                    (np.arange(phase_count) + offset) / max(1, phase_count - 1),
                    0.0,
                    1.0,
                )
            )
            for offset in ROLLING_STAGGER_OFFSETS
        ]
        fractions = np.unique(np.concatenate(fraction_grids))
        pitch = math.pi * self.config.module
        parameter_tolerance = _length_tolerance(
            self.config.module,
            INTERSECTION_PARAMETER_DEDUP_FACTOR,
        )
        geometric_tolerance = _length_tolerance(
            self.config.module,
            PROTECTED_CONTACT_RESIDUAL_FACTOR,
        )
        drive_cycle_arc = self.drive_teeth * pitch
        driven_cycle_arc = self.driven_teeth * pitch
        psi_start = float(self._psi(self.active_start))
        spans_by_member_and_sign = {
            (False, sign): tuple(span for span in drive_spans if span.sign == sign)
            for sign in (-1, 1)
        } | {
            (True, sign): tuple(span for span in driven_spans if span.sign == sign)
            for sign in (-1, 1)
        }

        def local_arc(value: float, cycle_arc: float) -> float:
            if not self.closed:
                return value
            result = value % cycle_arc
            return (
                0.0
                if math.isclose(
                    result,
                    cycle_arc,
                    abs_tol=parameter_tolerance,
                )
                else result
            )

        minimum_pair_count = math.inf
        maximum_contact_residual = 0.0
        covered_phase_count = 0
        for fraction in fractions:
            phi = self.active_start + (self.active_end - self.active_start) * float(
                fraction
            )
            common_arc = self.center_distance * float(
                self.arc_integral.integral(self.active_start, phi)
            )
            drive_arc = local_arc(common_arc, drive_cycle_arc)
            driven_arc = local_arc(common_arc, driven_cycle_arc)
            drive_angle = phi - self.active_start
            driven_angle = -(float(self._psi(phi)) - psi_start)
            relative_angle = driven_angle - drive_angle
            center_x = self.center_distance * math.cos(drive_angle)
            center_y = -self.center_distance * math.sin(drive_angle)
            pair_count = 0
            for sign in (-1, 1):
                matching_drive = [
                    span
                    for span in spans_by_member_and_sign[False, sign]
                    if span.start_arc - parameter_tolerance
                    <= drive_arc
                    <= span.end_arc + parameter_tolerance
                ]
                matching_driven = [
                    span
                    for span in spans_by_member_and_sign[True, sign]
                    if span.start_arc - parameter_tolerance
                    <= driven_arc
                    <= span.end_arc + parameter_tolerance
                ]
                for drive_span in matching_drive:
                    drive_lambda = sign * 0.25 * pitch - (
                        drive_arc - drive_span.phase * pitch
                    )
                    for driven_span in matching_driven:
                        driven_lambda = sign * 0.25 * pitch - (
                            driven_arc - driven_span.phase * pitch
                        )
                        if abs(drive_lambda - driven_lambda) > parameter_tolerance:
                            continue
                        drive_point = self._analytic_involute_points(
                            driven=False,
                            tooth_phase=drive_span.phase,
                            sign=sign,
                            common_arc=np.asarray([drive_arc]),
                        )[0]
                        driven_point = self._analytic_involute_points(
                            driven=True,
                            tooth_phase=driven_span.phase,
                            sign=sign,
                            common_arc=np.asarray([driven_arc]),
                        )[0]
                        placed_driven = driven_point * complex(
                            math.cos(relative_angle),
                            math.sin(relative_angle),
                        ) + complex(center_x, center_y)
                        residual = abs(drive_point - placed_driven)
                        maximum_contact_residual = max(
                            maximum_contact_residual, residual
                        )
                        if residual <= geometric_tolerance:
                            pair_count += 1
            minimum_pair_count = min(minimum_pair_count, pair_count)
            if pair_count > 0:
                covered_phase_count += 1
        if covered_phase_count == 0:
            raise RuntimeError(
                "Protected involute flanks provide no sampled exact conjugate "
                "contact candidates"
            )
        return (
            int(minimum_pair_count),
            maximum_contact_residual,
            len(fractions),
            covered_phase_count / len(fractions),
        )

    def _trim_rolling_nonworking_interference(
        self,
        drive: FloatArray,
        driven: FloatArray,
        drive_flanks: tuple[_ProtectedFlankSpan, ...],
        driven_flanks: tuple[_ProtectedFlankSpan, ...],
        *,
        apply_clearance: bool,
        drive_chord_error: float,
        driven_chord_error: float,
    ) -> tuple[
        FloatArray,
        FloatArray,
        tuple[_ProtectedFlankSpan, ...],
        tuple[_ProtectedFlankSpan, ...],
        int,
        float,
        float,
        float,
        int,
    ]:
        """Trim non-working interference by rolling the finished pair.

        On closed gears, retained analytic flanks and the connected support
        cores are protected. Any other overlapping material is eligible for
        removal, including a nonconvex cusp loop outside the pitch curve. Both
        gears and all staggered grids use the same per-pass snapshot, then the
        combined cuts repeat to a sampled fixed point. Open profiles retain the
        historical pitch-side-only, single-pass trim because their endpoint
        closures are not periodic cutter stock. For hybrid undercuts, requested
        clearance extends only the opposing addendum-tip material. The
        interleaved samples are combined and closed into one cutter envelope
        before removal; working flanks are never offset or used as clearance
        cutters.
        """

        drive_geometry: Geometry = Polygon(drive[:-1])
        driven_geometry: Geometry = Polygon(driven[:-1])
        drive_flank_guard = self._protected_flank_guard(drive_flanks, drive_geometry)
        driven_flank_guard = self._protected_flank_guard(driven_flanks, driven_geometry)
        drive_core_guard = self._rolling_core_guard().intersection(drive_geometry)
        driven_core_guard = self._rolling_core_guard().intersection(driven_geometry)
        drive_exclusion = union_all([drive_flank_guard, drive_core_guard])
        driven_exclusion = union_all([driven_flank_guard, driven_core_guard])
        drive_pitch_material = self._analytic_pitch_material(False)
        driven_pitch_material = self._analytic_pitch_material(True)
        drive_trim_zone = drive_geometry if self.closed else drive_pitch_material
        driven_trim_zone = driven_geometry if self.closed else driven_pitch_material
        phase_count = (
            max(
                ROLLING_MIN_PHASES,
                ROLLING_PHASES_PER_TOOTH * max(self.drive_teeth, self.driven_teeth),
            )
            if self.closed
            else ROLLING_MIN_PHASES
        )
        overlap_tolerance = self._overlap_area_tolerance()
        classification_tolerance = overlap_tolerance
        numerical_trim_clearance = _length_tolerance(
            self.config.module, ANALYTIC_CHORD_TOLERANCE_FACTOR
        )
        rolling_clearance = self.clearance if apply_clearance else 0.0
        convergence_tolerance = ROLLING_CONVERGENCE_AREA_FACTOR * self.config.module**2
        psi_start = float(self._psi(self.active_start))
        initial_overlap = 0.0
        remaining_overlap = 0.0
        total_removed_area = 0.0
        total_phases = 0
        pass_count = 0

        maximum_passes = ROLLING_MAX_PASSES if self.closed else 1
        for pass_index in range(maximum_passes):
            pass_count = pass_index + 1
            pass_removed_area = 0.0
            pass_maximum_overlap = 0.0
            pass_drive_snapshot = drive_geometry
            pass_driven_snapshot = driven_geometry
            empty_geometry = Point(0.0, 0.0).buffer(0.0)
            pass_clearance = rolling_clearance if pass_index == 0 else 0.0
            if pass_clearance > 0.0:
                drive_tip_material = pass_drive_snapshot.difference(
                    drive_pitch_material
                )
                driven_tip_material = pass_driven_snapshot.difference(
                    driven_pitch_material
                )
                drive_flank_source_exclusion = drive_flank_guard.buffer(
                    pass_clearance + numerical_trim_clearance,
                    quad_segs=CLEARANCE_BUFFER_QUADRANT_SEGMENTS,
                )
                driven_flank_source_exclusion = driven_flank_guard.buffer(
                    pass_clearance + numerical_trim_clearance,
                    quad_segs=CLEARANCE_BUFFER_QUADRANT_SEGMENTS,
                )
                pass_drive_clearance_shell = (
                    drive_tip_material.buffer(
                        pass_clearance,
                        quad_segs=CLEARANCE_BUFFER_QUADRANT_SEGMENTS,
                    )
                    .difference(pass_drive_snapshot)
                    .difference(drive_flank_source_exclusion)
                )
                pass_driven_clearance_shell = (
                    driven_tip_material.buffer(
                        pass_clearance,
                        quad_segs=CLEARANCE_BUFFER_QUADRANT_SEGMENTS,
                    )
                    .difference(pass_driven_snapshot)
                    .difference(driven_flank_source_exclusion)
                )
            else:
                pass_drive_clearance_shell = empty_geometry
                pass_driven_clearance_shell = empty_geometry
            pass_drive_clearance_cuts: list[Geometry] = []
            pass_driven_clearance_cuts: list[Geometry] = []
            for offset in ROLLING_STAGGER_OFFSETS:
                fractions = (
                    (np.arange(phase_count) + offset) / phase_count
                    if self.closed
                    else np.clip(
                        (np.arange(phase_count) + offset) / max(1, phase_count - 1),
                        0.0,
                        1.0,
                    )
                )
                drive_cuts: list[Geometry] = []
                driven_cuts: list[Geometry] = []
                grid_maximum = 0.0

                def phase_nonworking_cut(
                    fraction: float,
                    current_drive: Geometry = pass_drive_snapshot,
                    current_driven: Geometry = pass_driven_snapshot,
                    drive_clearance_shell: Geometry = pass_drive_clearance_shell,
                    driven_clearance_shell: Geometry = pass_driven_clearance_shell,
                    empty_cut: Geometry = empty_geometry,
                    clearance_distance: float = pass_clearance,
                ) -> tuple[
                    float,
                    Geometry | None,
                    Geometry | None,
                    Geometry | None,
                    Geometry | None,
                    float,
                ]:
                    phi = self.active_start + (
                        self.active_end - self.active_start
                    ) * float(fraction)
                    drive_angle = phi - self.active_start
                    driven_angle = -(float(self._psi(phi)) - psi_start)
                    relative_angle = driven_angle - drive_angle
                    center_x = self.center_distance * math.cos(drive_angle)
                    center_y = -self.center_distance * math.sin(drive_angle)
                    placed_driven = self._place_geometry(
                        current_driven,
                        relative_angle,
                        center_x,
                        center_y,
                    )
                    overlap = current_drive.intersection(placed_driven)
                    area = float(overlap.area)
                    if area <= overlap_tolerance and clearance_distance <= 0.0:
                        return area, None, None, None, None, 0.0
                    placed_driven_exclusion = self._place_geometry(
                        driven_exclusion,
                        relative_angle,
                        center_x,
                        center_y,
                    )
                    if self.closed:
                        drive_parts: list[Geometry] = []
                        driven_parts: list[Geometry] = []
                        assignment_tolerance = (
                            CONTACT_AREA_TOLERANCE_FACTOR * self.config.module**2
                        )
                        unremovable = overlap.intersection(
                            drive_exclusion
                        ).intersection(placed_driven_exclusion)
                        for component in _polygon_parts(overlap):
                            # A pointwise subtraction can hollow out the material
                            # immediately behind a protected flank. When the same
                            # overlap component contains no protected material on
                            # the other member, assign the complete cut to that
                            # other member instead. Requiring an exterior
                            # connection prevents a one-sided cut from creating
                            # an internal hole that the single-ring outline
                            # format cannot represent.
                            drive_protected_area = float(
                                component.intersection(drive_exclusion).area
                            )
                            driven_protected_area = float(
                                component.intersection(placed_driven_exclusion).area
                            )
                            if (
                                driven_protected_area <= assignment_tolerance
                                and drive_protected_area
                                > driven_protected_area + assignment_tolerance
                                and float(
                                    component.boundary.intersection(
                                        placed_driven.boundary
                                    ).length
                                )
                                > numerical_trim_clearance
                            ):
                                driven_parts.append(
                                    component.difference(placed_driven_exclusion)
                                )
                                continue
                            elif (
                                drive_protected_area <= assignment_tolerance
                                and driven_protected_area
                                > drive_protected_area + assignment_tolerance
                                and float(
                                    component.boundary.intersection(
                                        current_drive.boundary
                                    ).length
                                )
                                > numerical_trim_clearance
                            ):
                                drive_parts.append(
                                    component.difference(drive_exclusion)
                                )
                                continue
                            drive_parts.append(component.difference(drive_exclusion))
                            driven_parts.append(
                                component.difference(placed_driven_exclusion)
                            )
                        drive_overlap = union_all(drive_parts)
                        driven_overlap = union_all(driven_parts)
                    else:
                        placed_driven_trim_zone = self._place_geometry(
                            driven_trim_zone,
                            relative_angle,
                            center_x,
                            center_y,
                        )
                        drive_root_overlap = overlap.intersection(drive_trim_zone)
                        driven_root_overlap = overlap.intersection(
                            placed_driven_trim_zone
                        )
                        outside_root_zones = overlap.difference(
                            union_all([drive_root_overlap, driven_root_overlap])
                        )
                        unremovable = outside_root_zones
                        drive_overlap = drive_root_overlap
                        driven_overlap = driven_root_overlap
                    if clearance_distance > 0.0:
                        placed_driven_clearance_shell = self._place_geometry(
                            driven_clearance_shell,
                            relative_angle,
                            center_x,
                            center_y,
                        )
                        drive_clearance_overlap = current_drive.intersection(
                            placed_driven_clearance_shell
                        ).intersection(drive_trim_zone)
                        driven_clearance_overlap = placed_driven.intersection(
                            drive_clearance_shell
                        ).intersection(
                            self._place_geometry(
                                driven_trim_zone,
                                relative_angle,
                                center_x,
                                center_y,
                            )
                        )
                    else:
                        drive_clearance_overlap = empty_cut
                        driven_clearance_overlap = empty_cut
                    unremovable_area = float(unremovable.area)
                    local_drive_cut = (
                        make_valid(drive_overlap)
                        if not drive_overlap.is_empty
                        else None
                    )
                    local_driven_cut = (
                        make_valid(
                            self._unplace_geometry(
                                make_valid(driven_overlap),
                                relative_angle,
                                center_x,
                                center_y,
                            )
                        )
                        if not driven_overlap.is_empty
                        else None
                    )
                    local_drive_clearance_cut = (
                        make_valid(drive_clearance_overlap)
                        if not drive_clearance_overlap.is_empty
                        else None
                    )
                    local_driven_clearance_cut = (
                        make_valid(
                            self._unplace_geometry(
                                make_valid(driven_clearance_overlap),
                                relative_angle,
                                center_x,
                                center_y,
                            )
                        )
                        if not driven_clearance_overlap.is_empty
                        else None
                    )
                    return (
                        area,
                        local_drive_cut,
                        local_driven_cut,
                        local_drive_clearance_cut,
                        local_driven_clearance_cut,
                        unremovable_area,
                    )

                with ThreadPoolExecutor(
                    max_workers=min(_MAX_GEOMETRY_WORKERS, len(fractions))
                ) as executor:
                    phase_results = executor.map(
                        phase_nonworking_cut,
                        (float(value) for value in fractions),
                    )
                    for (
                        area,
                        drive_overlap,
                        driven_overlap,
                        drive_clearance_overlap,
                        driven_clearance_overlap,
                        unremovable_area,
                    ) in phase_results:
                        grid_maximum = max(grid_maximum, area)
                        if unremovable_area > classification_tolerance:
                            raise RuntimeError(
                                "Opposing-gear cutter interference cannot be "
                                "removed without entering protected working "
                                "flanks or support cores "
                                f"(area {unremovable_area:.9g})"
                            )
                        if drive_overlap is not None:
                            drive_cuts.append(drive_overlap)
                        if driven_overlap is not None:
                            driven_cuts.append(driven_overlap)
                        if drive_clearance_overlap is not None:
                            pass_drive_clearance_cuts.append(drive_clearance_overlap)
                        if driven_clearance_overlap is not None:
                            pass_driven_clearance_cuts.append(driven_clearance_overlap)
                total_phases += len(fractions)
                if pass_index == 0 and offset == 0.0:
                    initial_overlap = grid_maximum
                pass_maximum_overlap = max(pass_maximum_overlap, grid_maximum)
                if drive_cuts or driven_cuts:
                    before_drive = float(drive_geometry.area)
                    before_driven = float(driven_geometry.area)
                    if drive_cuts:
                        drive_cut = (
                            union_all(drive_cuts)
                            .buffer(
                                numerical_trim_clearance,
                                quad_segs=TRIM_BUFFER_QUADRANT_SEGMENTS,
                            )
                            .difference(drive_exclusion)
                            .intersection(drive_trim_zone)
                        )
                        drive_geometry = drive_geometry.difference(drive_cut)
                    if driven_cuts:
                        driven_cut = (
                            union_all(driven_cuts)
                            .buffer(
                                numerical_trim_clearance,
                                quad_segs=TRIM_BUFFER_QUADRANT_SEGMENTS,
                            )
                            .difference(driven_exclusion)
                            .intersection(driven_trim_zone)
                        )
                        driven_geometry = driven_geometry.difference(driven_cut)
                    drive_geometry = _clean_polygon(drive_geometry)
                    driven_geometry = _clean_polygon(driven_geometry)
                    removed_area = (
                        before_drive
                        - float(drive_geometry.area)
                        + before_driven
                        - float(driven_geometry.area)
                    )
                    pass_removed_area += removed_area
                    total_removed_area += removed_area

            if not (
                pass_removed_area > 0.0
                or pass_drive_clearance_cuts
                or pass_driven_clearance_cuts
            ):
                remaining_overlap = pass_maximum_overlap
                break

            def apply_clearance_envelope(
                geometry: Geometry,
                cuts: list[Geometry],
                flanks: tuple[_ProtectedFlankSpan, ...],
                core_guard: Geometry,
                trim_zone: Geometry,
                chord_error: float,
            ) -> tuple[Geometry, tuple[_ProtectedFlankSpan, ...]]:
                if not cuts:
                    return geometry, flanks
                raw_cut = (
                    union_all(cuts)
                    .buffer(
                        rolling_clearance,
                        quad_segs=CLEARANCE_BUFFER_QUADRANT_SEGMENTS,
                    )
                    .buffer(
                        -rolling_clearance,
                        quad_segs=CLEARANCE_BUFFER_QUADRANT_SEGMENTS,
                    )
                    .buffer(
                        numerical_trim_clearance,
                        quad_segs=TRIM_BUFFER_QUADRANT_SEGMENTS,
                    )
                    .intersection(trim_zone)
                )
                # First expose where the clearance envelope naturally meets
                # each analytic flank. Retain only the tip-connected working
                # portion; preserving the former root endpoint would leave a
                # thin attached spur between the new root and the old flank.
                provisional = _clean_polygon(
                    geometry.difference(raw_cut.difference(core_guard))
                )
                provisional_outline = _outline(
                    provisional,
                    _length_tolerance(
                        self.config.module,
                        GEOMETRY_LENGTH_TOLERANCE_FACTOR,
                    ),
                )
                clipped_flanks, _ = self._clip_protected_flanks_to_boundary(
                    flanks,
                    provisional_outline,
                    chord_error=chord_error,
                )
                flank_guard = self._protected_flank_guard(
                    clipped_flanks,
                    geometry,
                )
                exclusion = union_all([flank_guard, core_guard])
                cleared = geometry.difference(raw_cut.difference(exclusion))
                return _clean_polygon(cleared), clipped_flanks

            before_clearance_drive = float(drive_geometry.area)
            before_clearance_driven = float(driven_geometry.area)
            drive_geometry, drive_flanks = apply_clearance_envelope(
                drive_geometry,
                pass_drive_clearance_cuts,
                drive_flanks,
                drive_core_guard,
                drive_trim_zone,
                drive_chord_error,
            )
            driven_geometry, driven_flanks = apply_clearance_envelope(
                driven_geometry,
                pass_driven_clearance_cuts,
                driven_flanks,
                driven_core_guard,
                driven_trim_zone,
                driven_chord_error,
            )
            clearance_removed_area = (
                before_clearance_drive
                - float(drive_geometry.area)
                + before_clearance_driven
                - float(driven_geometry.area)
            )
            pass_removed_area += clearance_removed_area
            total_removed_area += clearance_removed_area
            drive_flank_guard = self._protected_flank_guard(
                drive_flanks,
                drive_geometry,
            )
            driven_flank_guard = self._protected_flank_guard(
                driven_flanks,
                driven_geometry,
            )
            drive_exclusion = union_all([drive_flank_guard, drive_core_guard])
            driven_exclusion = union_all([driven_flank_guard, driven_core_guard])
            remaining_overlap = pass_maximum_overlap
            if not self.closed or pass_removed_area <= convergence_tolerance:
                break
        else:
            raise RuntimeError(
                "Opposing-gear cutter trimming did not converge within "
                f"{maximum_passes} passes"
            )

        drive_outline = _outline(
            _clean_polygon(drive_geometry),
            _length_tolerance(self.config.module, GEOMETRY_LENGTH_TOLERANCE_FACTOR),
        )
        driven_outline = _outline(
            _clean_polygon(driven_geometry),
            _length_tolerance(self.config.module, GEOMETRY_LENGTH_TOLERANCE_FACTOR),
        )
        return (
            drive_outline,
            driven_outline,
            drive_flanks,
            driven_flanks,
            total_phases,
            initial_overlap,
            remaining_overlap,
            total_removed_area,
            pass_count,
        )

    @staticmethod
    def _overlap_with_placed_drive(
        placed_drive: Polygon,
        driven: FloatArray,
        driven_angle: float,
        center_x: float,
        center_y: float,
    ) -> float:
        placed_driven = _transform_outline(
            driven,
            driven_angle,
            center_x,
            center_y,
        )
        return float(placed_drive.intersection(placed_driven).area)

    def _verify_pair(
        self, drive: FloatArray, driven: FloatArray
    ) -> tuple[float, float, int]:
        phase_count = (
            max(
                VERIFICATION_MIN_CLOSED_PHASES,
                VERIFICATION_PHASES_PER_TOOTH
                * max(self.drive_teeth, self.driven_teeth),
            )
            if self.closed
            else VERIFICATION_MIN_OPEN_PHASES
        )
        fraction_grids = [
            (
                (np.arange(phase_count) + offset) / phase_count
                if self.closed
                else np.clip(
                    (np.arange(phase_count) + offset) / max(1, phase_count - 1),
                    0.0,
                    1.0,
                )
            )
            for offset in ROLLING_STAGGER_OFFSETS
        ]
        fractions = np.unique(np.concatenate(fraction_grids))
        psi_start = float(self._psi(self.active_start))
        placed_drive = Polygon(drive[:-1])

        def phase_overlap(fraction: float) -> float:
            phi = self.active_start + (self.active_end - self.active_start) * float(
                fraction
            )
            drive_angle = phi - self.active_start
            driven_angle = -(float(self._psi(phi)) - psi_start)
            return self._overlap_with_placed_drive(
                placed_drive,
                driven,
                driven_angle - drive_angle,
                self.center_distance * math.cos(drive_angle),
                -self.center_distance * math.sin(drive_angle),
            )

        with ThreadPoolExecutor(
            max_workers=min(_MAX_GEOMETRY_WORKERS, len(fractions))
        ) as executor:
            overlaps = list(executor.map(phase_overlap, fractions))
        maximum_overlap = max(overlaps)
        overlap_tolerance = self._overlap_area_tolerance()
        if maximum_overlap > overlap_tolerance:
            raise RuntimeError(
                "Staggered sampled-phase verification found solid overlap area "
                f"{maximum_overlap:.9g} above tolerance {overlap_tolerance:.9g}"
            )

        contact_area_tolerance = CONTACT_AREA_TOLERANCE_FACTOR * self.config.module**2
        recovery_count = (
            CONTACT_RECOVERY_PHASES_CLOSED
            if self.closed
            else CONTACT_RECOVERY_PHASES_OPEN
        )

        def recover_contact_delta(index: int) -> float:
            fraction = (index + CONTACT_RECOVERY_PHASE_OFFSET) / recovery_count
            phi = self.active_start + (self.active_end - self.active_start) * fraction
            drive_angle = phi - self.active_start
            desired = -(float(self._psi(phi)) - psi_start)
            relative_desired = desired - drive_angle
            center_x = self.center_distance * math.cos(drive_angle)
            center_y = -self.center_distance * math.sin(drive_angle)
            if (
                self._overlap_with_placed_drive(
                    placed_drive,
                    driven,
                    relative_desired,
                    center_x,
                    center_y,
                )
                > contact_area_tolerance
            ):
                return 0.0
            best = math.inf
            for direction in (-1.0, 1.0):
                clear = 0.0
                collision = CONTACT_SEARCH_INITIAL_ANGLE
                found = False
                while collision <= CONTACT_SEARCH_MAX_ANGLE:
                    if (
                        self._overlap_with_placed_drive(
                            placed_drive,
                            driven,
                            relative_desired + direction * collision,
                            center_x,
                            center_y,
                        )
                        > contact_area_tolerance
                    ):
                        found = True
                        break
                    clear = collision
                    collision *= 2.0
                if not found and clear < CONTACT_SEARCH_MAX_ANGLE:
                    collision = CONTACT_SEARCH_MAX_ANGLE
                    found = (
                        self._overlap_with_placed_drive(
                            placed_drive,
                            driven,
                            relative_desired + direction * collision,
                            center_x,
                            center_y,
                        )
                        > contact_area_tolerance
                    )
                if not found:
                    continue
                while collision - clear > CONTACT_SEARCH_ANGLE_TOLERANCE:
                    midpoint = 0.5 * (clear + collision)
                    if (
                        self._overlap_with_placed_drive(
                            placed_drive,
                            driven,
                            relative_desired + direction * midpoint,
                            center_x,
                            center_y,
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
        return maximum_overlap, max(contact_deltas), len(fractions)

    def _drive_centrode_outline_distance(self, drive: FloatArray) -> float | None:
        if not self.closed:
            return None
        sample_count = max(
            _analysis_interval_count(len(self.config.samples)) + 1,
            CENTRODE_OUTLINE_SAMPLES_PER_TOOTH * self.drive_teeth + 1,
        )
        phi = np.linspace(
            self.active_start,
            self.active_start + self.drive_cycle,
            sample_count,
        )
        radius = np.asarray(self._drive_radius(phi), dtype=float)
        centrode = radius * np.exp(-1j * phi)
        centrode_points = np.column_stack((centrode.real, centrode.imag))

        # GEOS' default Hausdorff metric is discrete: it measures every vertex
        # against the other complete LineString. Nearest-vertex distances form
        # a conservative upper bound for that metric and spatial trees compute
        # it in O(n log n), instead of the quadratic LineString comparison.
        # Fall back to the exact metric only near the acceptance threshold, so
        # this faster check can never admit a profile the old check rejected.
        outline_tree = cKDTree(drive)
        centrode_tree = cKDTree(centrode_points)
        centrode_to_outline = outline_tree.query(
            centrode_points,
            k=1,
            workers=_MAX_GEOMETRY_WORKERS,
        )[0]
        outline_to_centrode = centrode_tree.query(
            drive,
            k=1,
            workers=_MAX_GEOMETRY_WORKERS,
        )[0]
        distance_bound = float(
            max(np.max(centrode_to_outline), np.max(outline_to_centrode))
        )
        fidelity_tolerance = (
            max(self.addendum, self.dedendum)
            + self.fillet_radius
            + CENTRODE_FIDELITY_ALLOWANCE_MODULES * self.config.module
        )
        if distance_bound <= fidelity_tolerance:
            return distance_bound
        centrode_line = LineString(centrode_points)
        return float(centrode_line.hausdorff_distance(LineString(drive)))

    def generate(self, samples_per_radian: int) -> EngineResult:
        if samples_per_radian < MIN_SAMPLES_PER_RADIAN:
            raise ValueError(
                f"samples_per_radian must be at least {MIN_SAMPLES_PER_RADIAN}"
            )
        if _MAX_GEOMETRY_WORKERS > 1:
            with ThreadPoolExecutor(max_workers=2) as executor:
                drive_result, driven_result = executor.map(
                    lambda driven: self._generate_analytic_involute_gear(
                        driven,
                        samples_per_radian,
                    ),
                    (False, True),
                )
        else:
            drive_result = self._generate_analytic_involute_gear(
                False, samples_per_radian
            )
            driven_result = self._generate_analytic_involute_gear(
                True, samples_per_radian
            )
        drive = drive_result.outline
        driven = driven_result.outline
        drive_protected_flanks, drive_clipped_flank_count = (
            self._clip_protected_flanks_to_boundary(
                drive_result.protected_flanks,
                drive,
                chord_error=drive_result.maximum_chord_error,
            )
        )
        driven_protected_flanks, driven_clipped_flank_count = (
            self._clip_protected_flanks_to_boundary(
                driven_result.protected_flanks,
                driven,
                chord_error=driven_result.maximum_chord_error,
            )
        )
        drive_result = replace(
            drive_result,
            protected_flanks=drive_protected_flanks,
        )
        driven_result = replace(
            driven_result,
            protected_flanks=driven_protected_flanks,
        )
        boundary_clipped_flank_count = (
            drive_clipped_flank_count + driven_clipped_flank_count
        )
        analytic_curve_sample_count = (
            drive_result.sample_count + driven_result.sample_count
        )
        analytic_flank_sample_count = (
            drive_result.flank_sample_count + driven_result.flank_sample_count
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
        maximum_hybrid_connector_length = max(
            drive_result.maximum_hybrid_connector_length,
            driven_result.maximum_hybrid_connector_length,
        )
        maximum_fillet_root_residual = max(
            drive_result.maximum_fillet_root_residual,
            driven_result.maximum_fillet_root_residual,
        )
        analytic_undercut_count = (
            drive_result.undercut_count + driven_result.undercut_count
        )
        hybrid_undercut_count = (
            drive_result.hybrid_undercut_count + driven_result.hybrid_undercut_count
        )
        pretrim_drive_flank_error, pretrim_drive_regular_factor = (
            self._verify_protected_flanks(
                drive_result.protected_flanks,
                drive,
                chord_error=drive_result.maximum_chord_error,
            )
        )
        pretrim_driven_flank_error, pretrim_driven_regular_factor = (
            self._verify_protected_flanks(
                driven_result.protected_flanks,
                driven,
                chord_error=driven_result.maximum_chord_error,
            )
        )
        (
            drive,
            driven,
            trimmed_drive_flanks,
            trimmed_driven_flanks,
            rolling_trim_phase_count,
            rolling_initial_overlap,
            rolling_sampled_overlap,
            rolling_removed_area,
            rolling_trim_pass_count,
        ) = self._trim_rolling_nonworking_interference(
            drive,
            driven,
            drive_result.protected_flanks,
            driven_result.protected_flanks,
            apply_clearance=hybrid_undercut_count > 0,
            drive_chord_error=drive_result.maximum_chord_error,
            driven_chord_error=driven_result.maximum_chord_error,
        )
        drive_result = replace(
            drive_result,
            protected_flanks=trimmed_drive_flanks,
        )
        driven_result = replace(
            driven_result,
            protected_flanks=trimmed_driven_flanks,
        )
        posttrim_drive_flank_error, posttrim_drive_regular_factor = (
            self._verify_protected_flanks(
                drive_result.protected_flanks,
                drive,
                chord_error=drive_result.maximum_chord_error,
            )
        )
        posttrim_driven_flank_error, posttrim_driven_regular_factor = (
            self._verify_protected_flanks(
                driven_result.protected_flanks,
                driven,
                chord_error=driven_result.maximum_chord_error,
            )
        )
        (
            minimum_protected_contact_pairs,
            maximum_protected_contact_residual,
            protected_contact_verification_phases,
            protected_contact_coverage_fraction,
        ) = self._verify_protected_contact_coverage(
            drive_result.protected_flanks,
            driven_result.protected_flanks,
        )
        maximum_protected_flank_boundary_error = max(
            pretrim_drive_flank_error,
            pretrim_driven_flank_error,
            posttrim_drive_flank_error,
            posttrim_driven_flank_error,
        )
        minimum_flank_regular_factor = min(
            drive_result.minimum_flank_regular_factor,
            driven_result.minimum_flank_regular_factor,
            pretrim_drive_regular_factor,
            pretrim_driven_regular_factor,
            posttrim_drive_regular_factor,
            posttrim_driven_regular_factor,
        )
        envelope_tolerance = _length_tolerance(
            self.config.module, ANALYTIC_ENVELOPE_RESIDUAL_FACTOR
        )
        if maximum_envelope_residual > envelope_tolerance:
            raise RuntimeError(
                "Analytic involute envelope residual exceeds tolerance "
                f"({maximum_envelope_residual:.9g})"
            )
        if maximum_tangency_residual > ANALYTIC_TANGENCY_RESIDUAL_TOLERANCE:
            raise RuntimeError(
                "Analytic involute tangency residual exceeds tolerance "
                f"({maximum_tangency_residual:.9g})"
            )
        chord_acceptance_tolerance = _length_tolerance(
            self.config.module,
            ANALYTIC_CHORD_TOLERANCE_FACTOR * ANALYTIC_CHORD_ACCEPTANCE_SLACK,
        )
        if maximum_chord_error > chord_acceptance_tolerance:
            raise RuntimeError(
                "Analytic involute tessellation error exceeds tolerance "
                f"({maximum_chord_error:.9g})"
            )
        if maximum_fillet_root_residual > envelope_tolerance:
            raise RuntimeError(
                "Analytic rack-tip fillet does not meet the dedendum "
                f"({maximum_fillet_root_residual:.9g})"
            )
        drive_centrode_outline_distance = self._drive_centrode_outline_distance(drive)
        centrode_fidelity_tolerance = (
            max(self.addendum, self.dedendum)
            + self.fillet_radius
            + CENTRODE_FIDELITY_ALLOWANCE_MODULES * self.config.module
        )
        if (
            drive_centrode_outline_distance is not None
            and drive_centrode_outline_distance > centrode_fidelity_tolerance
        ):
            raise RuntimeError(
                "Analytic involute arrangement did not preserve the requested "
                "drive centrode "
                f"(outline distance {drive_centrode_outline_distance:.9g}, "
                f"tooth-envelope tolerance {centrode_fidelity_tolerance:.9g})"
            )
        if self.closed:
            area_intervals = _analysis_interval_count(len(self.config.samples))
            phi = np.linspace(
                self.active_start,
                self.active_end,
                area_intervals + 1,
            )
            pitch_area = float(
                simpson(0.5 * np.asarray(self._drive_radius(phi)) ** 2, x=phi)
            )
            outline_area = abs(_signed_area(drive))
            if outline_area < MINIMUM_PITCH_AREA_FRACTION * pitch_area:
                raise RuntimeError(
                    "Analytic involute arrangement disconnected the intended "
                    "drive-gear body "
                    f"(outline area {outline_area:.9g}, drive-centrode enclosed "
                    f"area {pitch_area:.9g})"
                )
        maximum_overlap, transmission_error, verification_phases = self._verify_pair(
            drive, driven
        )
        drive_radii = np.linalg.norm(drive[:-1], axis=1)
        driven_radii = np.linalg.norm(driven[:-1], axis=1)
        minimum_root_radius = float(min(np.min(drive_radii), np.min(driven_radii)))
        ratio_intervals = _analysis_interval_count(len(self.config.samples))
        phi = np.linspace(self.active_start, self.active_end, ratio_intervals + 1)
        maximum_ratio = float(np.max(self._psi(phi, 1)))
        contact_distance_bound = (
            self.addendum / max(SLIDING_SINE_FLOOR, math.sin(self.alpha))
            + 0.5 * math.pi * self.config.module
        )
        generation_backend = "hybrid_analytic_involute"
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
            "profile_family": "generalized_involute",
            "generation_backend": generation_backend,
            "geometry_backend": "shapely-geos",
            "geometry_precision": "double",
            "geometry_worker_limit": _MAX_GEOMETRY_WORKERS,
            "overlap_area_tolerance": self._overlap_area_tolerance(),
            "verification_method": "staggered_sampled_phase_grid",
            "verification_stagger_grid_count": len(ROLLING_STAGGER_OFFSETS),
            "drive_teeth": self.drive_teeth,
            "driven_teeth": self.driven_teeth,
            "average_angular_ratio": self.average_ratio,
            "module": self.config.module,
            "pressure_angle_deg": self.config.pressure_angle_deg,
            "requested_clearance_factor": self.config.clearance_factor,
            "requested_clearance": self.clearance,
            "nominal_dedendum": self.nominal_dedendum,
            "effective_dedendum": self.dedendum,
            "rolling_clearance_applied": (
                self.clearance > 0.0 and hybrid_undercut_count > 0
            ),
            "clearance_generation_method": (
                "analytic_dedendum_with_addendum_tip_cutter_envelope"
                if self.clearance > 0.0 and hybrid_undercut_count > 0
                else "analytic_dedendum"
            ),
            "total_integral": self.total_integral,
            "center_distance": self.center_distance,
            "undercut_curvature_limit": self.curvature_limit,
            "undercut_detection_method": "exact_per_flank_cusp_equation",
            "maximum_join_gap": maximum_join_gap,
            "maximum_hybrid_connector_length": (maximum_hybrid_connector_length),
            "maximum_intersection_residual": maximum_intersection_residual,
            "maximum_fillet_root_residual": maximum_fillet_root_residual,
            "analytic_undercut_count": analytic_undercut_count,
            "hybrid_undercut_count": hybrid_undercut_count,
            "protected_flank_count": (
                len(drive_result.protected_flanks) + len(driven_result.protected_flanks)
            ),
            "boundary_clipped_flank_count": boundary_clipped_flank_count,
            "minimum_flank_regular_factor": minimum_flank_regular_factor,
            "maximum_protected_flank_boundary_error": (
                maximum_protected_flank_boundary_error
            ),
            "minimum_protected_contact_pairs": (minimum_protected_contact_pairs),
            "maximum_protected_contact_residual": (maximum_protected_contact_residual),
            "protected_contact_verification_phase_count": (
                protected_contact_verification_phases
            ),
            "protected_contact_coverage_fraction": (
                protected_contact_coverage_fraction
            ),
            "rolling_nonworking_trim_scope": (
                "all_unprotected_tooth_material" if self.closed else "pitch_side_roots"
            ),
            "rolling_nonworking_trim_phase_count": rolling_trim_phase_count,
            "rolling_nonworking_trim_pass_count": rolling_trim_pass_count,
            "rolling_nonworking_initial_overlap_area": rolling_initial_overlap,
            "rolling_nonworking_sampled_overlap_area": rolling_sampled_overlap,
            "rolling_nonworking_removed_area": rolling_removed_area,
            "placed_pair_overlap_area": maximum_overlap,
            "centrodes_are_convex": self.centrodes_are_convex,
            "maximum_drive_curvature": self.maximum_drive_curvature,
            "minimum_driven_curvature": self.minimum_driven_curvature,
            "drive_centrode_outline_distance": drive_centrode_outline_distance,
            "centrode_fidelity_tolerance": centrode_fidelity_tolerance,
            "analytic_curve_sample_count": analytic_curve_sample_count,
            "analytic_flank_sample_count": analytic_flank_sample_count,
            "maximum_envelope_residual": maximum_envelope_residual,
            "maximum_envelope_tangency_residual": maximum_tangency_residual,
            "maximum_analytic_chord_error": maximum_chord_error,
            "nonworking_closure": (
                "analytic_rack_tip_and_dedendum_envelopes"
                if hybrid_undercut_count == 0
                else "analytic_fillets_with_opposing_gear_undercuts"
            ),
            "requested_fillet_radius": self.fillet_radius,
            "requested_fillet_applied_to_closure": (hybrid_undercut_count == 0),
            "fillet_closure_mode": (
                "analytic_only"
                if hybrid_undercut_count == 0
                else "analytic_with_hybrid_opposing_cutter"
            ),
            "verification_phase_count": verification_phases,
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
            f"{analytic_flank_sample_count} analytic flank samples"
        )
        return EngineResult(drive, driven, metadata, log)


def generate_geometry(config: EngineConfig, samples_per_radian: int) -> EngineResult:
    """Generate gear outlines and verification metadata in-process."""

    return _GearGenerator(config).generate(samples_per_radian)
