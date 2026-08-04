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
    MIN_CLOSURE_CURVE_SAMPLES,
    MIN_FLANK_CURVE_SAMPLES,
    PROTECTED_CONTACT_RESIDUAL_FACTOR,
    ROOT_SUPPORT_RADIUS_PITCH_FACTOR,
    UNDERCUT_VERTEX_CHORD_TOLERANCE_FACTOR,
    VERIFICATION_MIN_CLOSED_PHASES,
    VERIFICATION_MIN_OPEN_PHASES,
    VERIFICATION_STAGGER_OFFSETS,
)
from ncgears.engine import (
    EngineConfig,
    _GearGenerator,
    _offset_outline_normal,
    _verify_single_connected_outline,
)


def _severe_boundary_reversal_count(outline: np.ndarray) -> int:
    points = outline[:-1]
    incoming = points - np.roll(points, 1, axis=0)
    outgoing = np.roll(points, -1, axis=0) - points
    return int(np.count_nonzero(np.einsum("ij,ij->i", incoming, outgoing) < 0.0))


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
        if pair.metadata["cutter_undercut_curve_count"] == 0
        else "analytic_fillets_with_addendum_vertex_undercuts"
    )
    assert pair.metadata["requested_fillet_applied_to_closure"] is (
        pair.metadata["cutter_undercut_curve_count"] == 0
    )
    assert pair.metadata["fillet_closure_mode"] == (
        "analytic_only"
        if pair.metadata["cutter_undercut_curve_count"] == 0
        else "analytic_with_addendum_vertex_cutter"
    )
    assert pair.metadata["undercut_detection_method"] == "exact_per_flank_cusp_equation"
    assert pair.metadata["maximum_join_gap"] < (
        INTERSECTION_RESIDUAL_FACTOR * pair.metadata["module"]
    )
    assert pair.metadata["maximum_intersection_residual"] < (
        INTERSECTION_RESIDUAL_FACTOR * pair.metadata["module"]
    )
    assert pair.metadata["maximum_fillet_root_residual"] < (
        ANALYTIC_ENVELOPE_RESIDUAL_FACTOR * pair.metadata["module"]
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
    assert pair.metadata["protected_flank_count"] > 0
    assert pair.metadata["minimum_flank_regular_factor"] > 0.0
    assert pair.metadata["maximum_protected_flank_boundary_error"] < (
        pair.metadata["maximum_analytic_chord_error"]
        + 2.0 * ANALYTIC_CHORD_TOLERANCE_FACTOR * pair.metadata["module"]
    )
    assert pair.metadata["minimum_protected_contact_pairs"] >= 0
    assert 0.0 < pair.metadata["protected_contact_coverage_fraction"] <= 1.0
    assert pair.metadata["maximum_protected_contact_residual"] < (
        PROTECTED_CONTACT_RESIDUAL_FACTOR * pair.metadata["module"]
    )
    assert pair.metadata["protected_contact_verification_phase_count"] >= (
        VERIFICATION_MIN_CLOSED_PHASES
        if pair.metadata["topology"] == "closed"
        else VERIFICATION_MIN_OPEN_PHASES
    )
    assert pair.metadata["cutter_undercut_method"] == (
        "opposing_addendum_edge_vertex_trajectories"
    )
    assert pair.metadata["cutter_undercut_trim_scope"] == (
        "intersected_analytic_root_regions"
    )
    assert pair.metadata["cutter_undercut_vertices_per_addendum"] == 2
    assert pair.metadata["cutter_undercut_clipped_flank_count"] >= 0
    curve_count = pair.metadata["cutter_undercut_curve_count"]
    assert curve_count >= pair.metadata["hybrid_undercut_count"]
    if curve_count == 0:
        assert pair.metadata["cutter_undercut_curve_sample_count"] == 0
        assert pair.metadata["cutter_undercut_removed_area"] == pytest.approx(0.0)
        assert pair.metadata["cutter_undercut_clipped_flank_count"] == 0
    else:
        assert pair.metadata["cutter_undercut_curve_sample_count"] >= (
            curve_count * MIN_CLOSURE_CURVE_SAMPLES
        )
        assert pair.metadata["cutter_undercut_removed_area"] > 0.0
        assert pair.metadata["cutter_undercut_maximum_chord_error"] <= (
            UNDERCUT_VERTEX_CHORD_TOLERANCE_FACTOR * pair.metadata["module"]
        )
    for removed_key in (
        "rolling_nonworking_trim_phase_count",
        "rolling_nonworking_trim_pass_count",
        "rolling_nonworking_cut_regularization",
        "rolling_nonworking_removed_area",
    ):
        assert removed_key not in pair.metadata
    assert pair.metadata["verification_method"] == "staggered_sampled_phase_grid"
    assert pair.metadata["verification_stagger_grid_count"] == len(
        VERIFICATION_STAGGER_OFFSETS
    )
    assert pair.metadata["outline_connectivity_verification"] == (
        "precision_noded_single_closed_loop"
    )
    assert pair.metadata["outline_connectivity_tolerance"] > 0.0
    assert pair.metadata["drive_outline_is_single_closed_loop"] is True
    assert pair.metadata["driven_outline_is_single_closed_loop"] is True
    base_phase_count = (
        VERIFICATION_MIN_CLOSED_PHASES
        if pair.metadata["topology"] == "closed"
        else VERIFICATION_MIN_OPEN_PHASES
    )
    minimum_verification_phases = base_phase_count * len(
        VERIFICATION_STAGGER_OFFSETS
    )
    if pair.metadata["topology"] == "open":
        # Shifted open grids share their clipped upper endpoint.
        minimum_verification_phases -= len(VERIFICATION_STAGGER_OFFSETS) - 1
    assert pair.metadata["verification_phase_count"] >= minimum_verification_phases
    assert pair.metadata["minimum_root_radius"] > 0.0
    assert pair.maximum_transmission_error < 0.01
    for outline in (pair.drive_outline, pair.driven_outline):
        assert outline.shape[1] == 2
        assert np.allclose(outline[0], outline[-1])
        polygon = Polygon(outline)
        assert polygon.is_valid
        assert polygon.area > 1.0


def test_outline_connectivity_verification_rejects_precision_hairpin() -> None:
    square = np.array(
        [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]]
    )
    _verify_single_connected_outline(square, label="drive", tolerance=1e-8)

    epsilon = 1e-10
    hairpin = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [2.0 + epsilon, 4.0],
            [2.0 + epsilon, epsilon],
            [2.0 - epsilon, epsilon],
            [2.0 - epsilon, 4.0],
            [0.0, 4.0],
            [0.0, 0.0],
        ]
    )
    assert LineString(hairpin).is_ring
    assert Polygon(hairpin).is_valid

    with pytest.raises(
        RuntimeError,
        match="Drive gear outline is not one connected closed loop",
    ):
        _verify_single_connected_outline(hairpin, label="drive", tolerance=1e-8)


def test_constant_normal_offset_erodes_one_connected_solid() -> None:
    square = np.array(
        [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]]
    )

    outline, minimum_offset = _offset_outline_normal(
        square,
        0.25,
        label="test",
        module=1.0,
    )

    assert Polygon(outline).bounds == pytest.approx((0.25, 0.25, 3.75, 3.75))
    assert Polygon(outline).area == pytest.approx(3.5**2)
    assert minimum_offset == pytest.approx(0.25)


def test_maximum_backlash_offsets_both_finished_gears(tmp_path: Path) -> None:
    requested_maximum = 1.5
    pair = ncgears.generate(
        "phi - 0.08*sin(2*phi)",
        name="maximum_backlash",
        teeth=16,
        max_backlash_deg=requested_maximum,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    assert pair.metadata["clearance_mode"] == "maximum_backlash"
    assert pair.clearance > 0.0
    assert pair.maximum_backlash_deg == pytest.approx(requested_maximum)
    assert 0.0 < pair.minimum_backlash_deg < pair.maximum_backlash_deg
    assert pair.metadata["preclearance_drive_area"] > pair.metadata["drive_area"]
    assert pair.metadata["preclearance_driven_area"] > pair.metadata["driven_area"]
    chord_tolerance = ANALYTIC_CHORD_TOLERANCE_FACTOR * pair.metadata["module"]
    for key in ("drive_minimum_face_offset", "driven_minimum_face_offset"):
        assert pair.metadata[key] + chord_tolerance >= pair.metadata[
            "clearance_distance"
        ]
    assert pair.metadata["preclearance_outline_connectivity_verified"] is True
    assert pair.metadata["postclearance_outline_connectivity_verified"] is True
    assert pair.metadata["contact_verification_stage"] == "preclearance"
    assert pair.metadata["placed_pair_overlap_area"] <= pair.metadata[
        "overlap_area_tolerance"
    ]


def test_mixed_harmonic_centrode_rejects_finished_geometry_interference(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ncgears.GenerationError,
        match="Addendum-vertex undercut curves left sampled solid overlap",
    ):
        ncgears.generate_from_centrode(
            "1 + 0.10*cos(phi) + 0.08*sin(2*phi) "
            "- 0.055*cos(3*phi) + 0.035*sin(5*phi)",
            name="mixed_harmonic_finished_geometry_interference",
            teeth=24,
            samples=1024,
            samples_per_radian=110,
            output_directory=tmp_path,
        )


def test_hybrid_analytic_involute_is_the_only_geometry_engine() -> None:
    fields = EngineConfig.__dataclass_fields__

    assert "profile" not in fields
    assert "cycloidal_rolling_factor" not in fields
    for removed_method in (
        "_generate_swept_gear",
        "_generate_conjugate_mate",
        "_involute_tooth_template",
        "_cycloidal_tooth_template",
        "_trim_rolling_nonworking_interference",
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
            clearance=0.04,
            samples=1024,
            samples_per_radian=20,
            output_directory=tmp_path,
        )
        for module in modules
    ]
    small, unit = pairs

    assert small.clearance == pytest.approx(0.04)
    assert unit.clearance == pytest.approx(0.04)
    assert small.maximum_backlash_deg == pytest.approx(
        unit.maximum_backlash_deg
    )
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
    assert pair.metadata["cutter_undercut_removed_area"] == pytest.approx(0.0)


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
        assert pair.metadata["cutter_undercut_trim_scope"] == (
            "intersected_analytic_root_regions"
        )
        assert pair.metadata["cutter_undercut_removed_area"] < 0.3

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
        assert shallow.hausdorff_distance(deep) < 1e-3
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
        # This high-curvature input needs enough source samples to avoid an
        # unstable near-zero-width branch on the driven outline.
        samples=4096,
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


def test_nonconvex_cusps_use_addendum_vertex_curve_undercuts(
    tmp_path: Path,
) -> None:
    pair = ncgears.generate(
        "phi + 0.0505*sin(4*phi)",
        name="nonconvex_hybrid_undercut",
        teeth=36,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    assert pair.metadata["centrodes_are_convex"] is False
    assert pair.metadata["hybrid_undercut_count"] == 8
    assert pair.metadata["maximum_hybrid_connector_length"] > 0.0
    assert (
        pair.metadata["analytic_undercut_count"]
        >= (pair.metadata["hybrid_undercut_count"])
    )
    assert pair.metadata["cutter_undercut_curve_count"] == (
        2 * pair.metadata["hybrid_undercut_count"]
    )
    assert pair.metadata["cutter_undercut_removed_area"] > 0.0
    assert pair.metadata["minimum_protected_contact_pairs"] >= 1
    assert pair.metadata["cutter_undercut_method"] == (
        "opposing_addendum_edge_vertex_trajectories"
    )
    # Smooth trajectory cuts should not recreate the direction-reversing
    # scallops left by unions of many discrete solid-cutter poses.
    assert _severe_boundary_reversal_count(pair.drive_outline) < 200
    assert _severe_boundary_reversal_count(pair.driven_outline) < 200


def test_cutter_undercut_does_not_leave_root_side_flank_slivers(
    tmp_path: Path,
) -> None:
    pair = ncgears.generate_from_centrode(
        "1 + 0.22*sin(phi) + 0.15*cos(2*phi) - 0.05*sin(3*phi)",
        name="cutter_undercut_flank_sliver",
        teeth=64,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    assert pair.metadata["analytic_undercut_count"] == 3
    assert pair.metadata["cutter_undercut_curve_count"] == 6
    assert pair.metadata["cutter_undercut_clipped_flank_count"] == 3

    # This window is the visible failure reported for the assembled driven
    # gear. The old symmetric flank guard doubled back by 2.84 radians here,
    # leaving a narrow material sliver between the flank and cutter curve.
    placed_driven = pair.placed_driven_outline
    edges = np.diff(placed_driven[:, 0] + 1j * placed_driven[:, 1])
    directions = edges / np.abs(edges)
    turns = np.angle(directions[1:] / directions[:-1])
    vertices = placed_driven[1:-1]
    reported_window = (
        (vertices[:, 0] > 49.56)
        & (vertices[:, 0] < 49.63)
        & (vertices[:, 1] > 42.62)
        & (vertices[:, 1] < 42.78)
    )
    assert np.any(reported_window)
    assert np.max(np.abs(turns[reported_window])) < 1.0
