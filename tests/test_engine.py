from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

import ncgears
from ncgears._policy import (
    ANALYTIC_CHORD_ACCEPTANCE_SLACK,
    ANALYTIC_CHORD_TOLERANCE_FACTOR,
    ANALYTIC_ENVELOPE_RESIDUAL_FACTOR,
    ANALYTIC_TANGENCY_RESIDUAL_TOLERANCE,
    INTERSECTION_RESIDUAL_FACTOR,
    MAX_SUPPORT_RADIUS_PITCH_FACTOR,
    MIN_FLANK_CURVE_SAMPLES,
    ROLLING_MIN_PHASES,
    ROLLING_STAGGER_OFFSETS,
    ROOT_SUPPORT_RADIUS_PITCH_FACTOR,
    VERIFICATION_MIN_CLOSED_PHASES,
    VERIFICATION_MIN_OPEN_PHASES,
)
from ncgears.engine import EngineConfig, _GearGenerator


def _assert_verified_pair(pair: ncgears.GearPair) -> None:
    assert pair.metadata["geometry_backend"] == "shapely-geos"
    assert pair.metadata["geometry_precision"] == "double"
    assert pair.metadata["geometry_worker_limit"] >= 1
    assert pair.metadata["generation_backend"] == ("hybrid_analytic_involute")
    assert "cutter_sweep_phase_count" not in pair.metadata
    assert "sweep_angular_step" not in pair.metadata
    assert pair.metadata["analytic_flank_sample_count"] >= (
        MIN_FLANK_CURVE_SAMPLES * (pair.drive_teeth + pair.driven_teeth)
    )
    assert pair.metadata["nonworking_closure"] == (
        "analytic_rack_tip_and_dedendum_envelopes"
    )
    assert pair.metadata["requested_fillet_applied_to_closure"] is True
    assert pair.metadata["maximum_join_gap"] < (
        INTERSECTION_RESIDUAL_FACTOR * pair.metadata["module"]
    )
    assert pair.metadata["maximum_intersection_residual"] < (
        INTERSECTION_RESIDUAL_FACTOR * pair.metadata["module"]
    )
    assert pair.metadata["maximum_fillet_root_residual"] < (
        ANALYTIC_ENVELOPE_RESIDUAL_FACTOR * pair.metadata["module"]
    )
    assert pair.metadata["rolling_nonworking_trim_phase_count"] >= (
        ROLLING_MIN_PHASES * len(ROLLING_STAGGER_OFFSETS)
    )
    assert pair.metadata["maximum_envelope_residual"] < 1e-9
    assert (
        pair.metadata["maximum_envelope_tangency_residual"]
        < ANALYTIC_TANGENCY_RESIDUAL_TOLERANCE
    )
    assert pair.metadata["maximum_analytic_chord_error"] < (
        ANALYTIC_CHORD_TOLERANCE_FACTOR
        * ANALYTIC_CHORD_ACCEPTANCE_SLACK
        * pair.metadata["module"]
    )
    assert pair.metadata["verification_method"] == "staggered_sampled_phase_grid"
    assert pair.metadata["verification_stagger_grid_count"] == len(
        ROLLING_STAGGER_OFFSETS
    )
    base_phase_count = (
        VERIFICATION_MIN_CLOSED_PHASES
        if pair.metadata["topology"] == "closed"
        else VERIFICATION_MIN_OPEN_PHASES
    )
    minimum_verification_phases = base_phase_count * len(ROLLING_STAGGER_OFFSETS)
    if pair.metadata["topology"] == "open":
        # Shifted open grids share their clipped upper endpoint.
        minimum_verification_phases -= len(ROLLING_STAGGER_OFFSETS) - 1
    assert pair.metadata["verification_phase_count"] >= minimum_verification_phases
    assert pair.metadata["minimum_root_radius"] > 0.0
    assert pair.maximum_transmission_error < 0.01
    for outline in (pair.drive_outline, pair.driven_outline):
        assert outline.shape[1] == 2
        assert np.allclose(outline[0], outline[-1])
        polygon = Polygon(outline)
        assert polygon.is_valid
        assert polygon.area > 1.0


def test_hybrid_analytic_involute_is_the_only_geometry_engine() -> None:
    fields = EngineConfig.__dataclass_fields__

    assert "profile" not in fields
    assert "cycloidal_rolling_factor" not in fields
    for removed_method in (
        "_generate_swept_gear",
        "_generate_conjugate_mate",
        "_involute_tooth_template",
        "_cycloidal_tooth_template",
    ):
        assert not hasattr(_GearGenerator, removed_method)


def test_python_engine_generates_unequal_ratio_pair(tmp_path: Path) -> None:
    pair = ncgears.generate(
        "2*phi",
        name="ratio_two_to_one",
        teeth=12,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    assert pair.drive_teeth == 12
    assert pair.driven_teeth == 6
    assert pair.ratio == pytest.approx(2.0)
    assert pair.center_distance == pytest.approx(9.0, abs=1e-8)
    assert pair.metadata["generation_backend"] == ("hybrid_analytic_involute")
    assert pair.metadata["profile_family"] == "generalized_involute"


def test_closed_motion_validation_cannot_alias_between_grid_points() -> None:
    sample_count = 8192
    frequency = 2048
    perturbation = 2.2e-4
    phi = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
    sine = np.sin(frequency * phi)
    cosine = np.cos(frequency * phi)
    envelope = sine**4
    envelope_first = 4.0 * frequency * sine**3 * cosine
    envelope_second = 4.0 * frequency**2 * (3.0 * sine**2 * cosine**2 - sine**4)
    psi = 2.0 * phi + perturbation * envelope * np.sin(phi)
    psi_first = 2.0 + perturbation * (
        envelope_first * np.sin(phi) + envelope * np.cos(phi)
    )
    psi_second = perturbation * (
        envelope_second * np.sin(phi)
        + 2.0 * envelope_first * np.cos(phi)
        - envelope * np.sin(phi)
    )
    config = EngineConfig(
        name="alias_probe",
        description="former fixed-grid alias probe",
        teeth=12,
        module=1.0,
        pressure_angle_deg=20.0,
        addendum_factor=1.0,
        dedendum_factor=1.2,
        fillet_factor=0.3,
        domain_start=0.0,
        domain_end=2.0 * math.pi,
        active_start=0.0,
        active_end=2.0 * math.pi,
        period=2.0 * math.pi,
        cycle_delta=4.0 * math.pi,
        open_=False,
        input_mode="transmission",
        samples=np.column_stack((phi, psi, psi_first, psi_second)),
    )

    with pytest.raises(ValueError, match="one driven-gear revolution"):
        _GearGenerator(config)


def test_geometry_tolerances_scale_with_module(tmp_path: Path) -> None:
    modules = (1e-6, 1.0)
    pairs = [
        ncgears.generate(
            "phi",
            name=f"scale_{module:g}",
            teeth=12,
            module=module,
            samples=1024,
            samples_per_radian=20,
            output_directory=tmp_path,
        )
        for module in modules
    ]
    small, unit = pairs

    assert small.drive_outline.shape == unit.drive_outline.shape
    assert small.driven_outline.shape == unit.driven_outline.shape
    assert np.allclose(
        small.drive_outline / modules[0],
        unit.drive_outline,
        rtol=1e-9,
        atol=1e-9,
    )
    assert np.allclose(
        small.driven_outline / modules[0],
        unit.driven_outline,
        rtol=1e-9,
        atol=1e-9,
    )


def test_support_radius_cap_applies_to_small_pitch_bodies() -> None:
    generator = object.__new__(_GearGenerator)
    generator.config = SimpleNamespace(module=1.0)
    generator.minimum_pitch_radius = 0.1

    radius = generator._support_radius(ROOT_SUPPORT_RADIUS_PITCH_FACTOR)

    assert radius == pytest.approx(
        MAX_SUPPORT_RADIUS_PITCH_FACTOR * generator.minimum_pitch_radius
    )


def test_analytic_involute_retains_exact_circular_root_depth(
    tmp_path: Path,
) -> None:
    pair = ncgears.generate(
        "phi",
        name="circular_fillet_reference",
        teeth=20,
        fillet_factor=0.35,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    expected_root_radius = 0.5 * pair.center_distance - 1.2
    drive_radius = np.linalg.norm(pair.drive_outline[:-1], axis=1)
    driven_radius = np.linalg.norm(pair.driven_outline[:-1], axis=1)
    assert np.min(drive_radius) == pytest.approx(expected_root_radius, abs=3e-5)
    assert np.min(driven_radius) == pytest.approx(expected_root_radius, abs=3e-5)
    assert pair.metadata["analytic_undercut_count"] == 0
    assert pair.metadata["rolling_nonworking_removed_area"] == pytest.approx(0.0)


def test_closed_undercut_count_excludes_periodic_seam_flanks(
    tmp_path: Path,
) -> None:
    pair = ncgears.generate(
        "phi",
        name="circular_undercut_count",
        teeth=6,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    # At this pitch radius every physical flank is undercut: two flanks per
    # tooth on each of the two gears. The periodic seam copies are not physical
    # flanks and must not inflate the reported count.
    assert pair.metadata["analytic_undercut_count"] == 4 * pair.drive_teeth


def test_python_engine_solves_centrode_center_distance(tmp_path: Path) -> None:
    pair = ncgears.generate_from_centrode(
        "1 + 0.08*cos(2*phi)",
        name="centrode_two_lobe",
        teeth=20,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    assert pair.metadata["input_mode"] == "drive_centrode"
    assert pair.metadata["centrode_reference_center_distance"] > 1.08
    assert pair.ratio == pytest.approx(1.0, abs=1e-8)
    assert pair.metadata["generation_backend"] == ("hybrid_analytic_involute")


def test_python_engine_generates_open_segment(tmp_path: Path) -> None:
    pair = ncgears.generate(
        "1.8*phi + 0.03*sin(phi)",
        name="open_segment",
        teeth=12,
        open_=True,
        drive_end=2.4,
        padding_pitches=2.5,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    assert pair.metadata["topology"] == "open"
    assert pair.metadata["generation_backend"] == ("hybrid_analytic_involute")
    assert pair.metadata["profile_family"] == "generalized_involute"
    assert pair.ratio == pytest.approx(
        (1.8 * 2.4 + 0.03 * math.sin(2.4)) / 2.4,
        abs=1e-7,
    )


def test_open_high_ratio_needs_only_documented_minimum_padding(
    tmp_path: Path,
) -> None:
    pair = ncgears.generate(
        "2.2*phi",
        name="open_high_ratio_minimum_padding",
        teeth=12,
        open_=True,
        drive_end=2.6,
        padding_pitches=2.5,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    assert pair.ratio == pytest.approx(2.2)


def test_open_quadratic_uses_parameter_clipped_padding_invariant_boundary(
    tmp_path: Path,
) -> None:
    pairs = [
        ncgears.generate(
            "1.35*phi + 0.025*phi**2",
            name=f"open_quadratic_padding_{padding:g}",
            teeth=16,
            open_=True,
            drive_end=3.2,
            padding_pitches=padding,
            samples=2048,
            samples_per_radian=20,
            output_directory=tmp_path,
        )
        for padding in (2.5, 8.0)
    ]
    for pair in pairs:
        _assert_verified_pair(pair)
        assert pair.metadata["rolling_nonworking_trim_scope"] == ("pitch_side_roots")
        assert pair.metadata["rolling_nonworking_removed_area"] < 0.1

        # The open backing curve is the historical analytical closure:
        # one quarter of each centrode over the active interval. A radial
        # sector clip cannot satisfy this for an accelerating motion law.
        phi = np.linspace(0.0, 3.2, 33)
        psi = 1.35 * phi + 0.025 * phi**2
        ratio = 1.35 + 0.05 * phi
        center = pair.center_distance
        drive_inner = 0.25 * center * ratio / (1.0 + ratio) * np.exp(-1j * phi)
        driven_inner = -0.25 * center / (1.0 + ratio) * np.exp(1j * psi)
        for outline, inner in (
            (pair.drive_outline, drive_inner),
            (pair.driven_outline, driven_inner),
        ):
            boundary = LineString(outline)
            assert (
                max(boundary.distance(Point(value.real, value.imag)) for value in inner)
                < 2e-4
            )

    shallow_padding, deep_padding = pairs
    for member in ("drive_outline", "driven_outline"):
        shallow = Polygon(getattr(shallow_padding, member))
        deep = Polygon(getattr(deep_padding, member))
        assert shallow.hausdorff_distance(deep) < 2e-4
        assert abs(shallow.area - deep.area) < 2e-4


def test_mild_nonconvex_centrode_stays_within_tooth_envelope(
    tmp_path: Path,
) -> None:
    pair = ncgears.generate(
        "phi + 0.016*sin(5*phi)",
        name="nonconvex",
        teeth=60,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    assert pair.metadata["centrodes_are_convex"] is False
    assert pair.metadata["profile_family"] == "generalized_involute"
    assert pair.metadata["generation_backend"] == ("hybrid_analytic_involute")
    assert pair.metadata["maximum_drive_curvature"] > 0.0
    assert (
        pair.metadata["drive_centrode_outline_distance"]
        <= pair.metadata["centrode_fidelity_tolerance"]
    )


def test_deep_nonconvex_centrode_uses_analytic_involute_form(
    tmp_path: Path,
) -> None:
    pair = ncgears.generate_from_centrode(
        "1 + 0.08*cos(5*phi)",
        name="deep_nonconvex",
        teeth=100,
        target_cycle_delta=5.0 * math.pi,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    drive = Polygon(pair.drive_outline)
    relative_concavity = (drive.convex_hull.area - drive.area) / drive.area
    assert pair.drive_teeth == 100
    assert pair.driven_teeth == 40
    assert pair.metadata["centrodes_are_convex"] is False
    assert pair.metadata["generation_backend"] == ("hybrid_analytic_involute")
    assert pair.metadata["profile_family"] == "generalized_involute"
    assert relative_concavity > 0.05
    assert (
        pair.metadata["drive_centrode_outline_distance"]
        <= pair.metadata["centrode_fidelity_tolerance"]
    )
