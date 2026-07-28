from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

import ncgears


def _assert_verified_pair(pair: ncgears.GearPair) -> None:
    assert pair.metadata["geometry_backend"] == "shapely-geos"
    assert pair.metadata["geometry_precision"] == "double"
    assert pair.metadata["geometry_worker_limit"] >= 1
    if pair.metadata["generation_backend"] == "analytic_form":
        assert pair.metadata["cutter_sweep_phase_count"] == 0
        assert pair.metadata["analytic_flank_sample_count"] >= 128 * (
            pair.drive_teeth + pair.driven_teeth
        )
        assert pair.metadata["nonworking_closure"] == (
            "analytic_rack_tip_and_dedendum_envelopes"
        )
        assert pair.metadata["requested_fillet_applied_to_closure"] is True
        assert pair.metadata["maximum_join_gap"] < (
            2e-7 * pair.metadata["module"]
        )
        assert pair.metadata["maximum_intersection_residual"] < (
            2e-7 * pair.metadata["module"]
        )
        assert pair.metadata["maximum_fillet_root_residual"] < (
            1e-8 * pair.metadata["module"]
        )
        assert pair.metadata["rolling_nonworking_trim_phase_count"] >= 96
        assert pair.metadata["maximum_envelope_residual"] < 1e-9
        assert pair.metadata["maximum_envelope_tangency_residual"] < 1e-10
        assert pair.metadata["maximum_analytic_chord_error"] < (
            3e-5 * pair.metadata["module"]
        )
    else:
        assert pair.metadata["cutter_sweep_phase_count"] >= 24 * (
            pair.drive_teeth + pair.driven_teeth
        )
    assert pair.metadata["verification_phase_count"] >= 48
    assert pair.metadata["minimum_root_radius"] > 0.0
    assert pair.maximum_transmission_error < 0.01
    for outline in (pair.drive_outline, pair.driven_outline):
        assert outline.shape[1] == 2
        assert np.allclose(outline[0], outline[-1])
        polygon = Polygon(outline)
        assert polygon.is_valid
        assert polygon.area > 1.0


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
    assert pair.metadata["generation_backend"] == "analytic_form"
    assert pair.metadata["profile_family"] == "generalized_involute"


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
    assert np.min(drive_radius) == pytest.approx(
        expected_root_radius, abs=3e-5
    )
    assert np.min(driven_radius) == pytest.approx(
        expected_root_radius, abs=3e-5
    )
    assert pair.metadata["analytic_undercut_count"] == 0
    assert pair.metadata["rolling_nonworking_removed_area"] == pytest.approx(
        0.0
    )


def test_python_engine_solves_centrode_center_distance(tmp_path: Path) -> None:
    pair = ncgears.generate_from_centrode(
        "1 + 0.08*cos(2*phi)",
        name="centrode",
        teeth=12,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    assert pair.metadata["input_mode"] == "drive_centrode"
    assert pair.metadata["centrode_reference_center_distance"] > 1.08
    assert pair.ratio == pytest.approx(1.0, abs=1e-8)
    assert pair.metadata["generation_backend"] == "analytic_form"


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
    assert pair.metadata["generation_backend"] == "analytic_form"
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
        assert pair.metadata["rolling_nonworking_trim_scope"] == (
            "pitch_side_roots"
        )
        assert pair.metadata["rolling_nonworking_removed_area"] < 0.1

        # The open backing curve is the historical analytical closure:
        # one quarter of each centrode over the active interval. A radial
        # sector clip cannot satisfy this for an accelerating motion law.
        phi = np.linspace(0.0, 3.2, 33)
        psi = 1.35 * phi + 0.025 * phi**2
        ratio = 1.35 + 0.05 * phi
        center = pair.center_distance
        drive_inner = (
            0.25
            * center
            * ratio
            / (1.0 + ratio)
            * np.exp(-1j * phi)
        )
        driven_inner = (
            -0.25
            * center
            / (1.0 + ratio)
            * np.exp(1j * psi)
        )
        for outline, inner in (
            (pair.drive_outline, drive_inner),
            (pair.driven_outline, driven_inner),
        ):
            boundary = LineString(outline)
            assert max(
                boundary.distance(Point(value.real, value.imag))
                for value in inner
            ) < 2e-4

    shallow_padding, deep_padding = pairs
    for member in ("drive_outline", "driven_outline"):
        shallow = Polygon(getattr(shallow_padding, member))
        deep = Polygon(getattr(deep_padding, member))
        assert shallow.hausdorff_distance(deep) < 2e-4
        assert abs(shallow.area - deep.area) < 2e-4


def test_cycloidal_mate_uses_conservative_sampled_envelope(
    tmp_path: Path,
) -> None:
    pair = ncgears.generate(
        "phi - 0.08*sin(2*phi)",
        name="cycloidal",
        teeth=28,
        profile="cycloidal",
        dedendum_factor=1.15,
        fillet_factor=0.2,
        samples=1024,
        samples_per_radian=110,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    assert pair.metadata["profile_family"] == "cycloidal_rack"
    tolerance = 1e-6 * 28
    assert pair.metadata["placed_pair_overlap_area"] <= tolerance


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
    assert pair.metadata["generation_backend"] == "analytic_form"
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
        profile="involute",
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
    assert pair.metadata["generation_backend"] == "analytic_form"
    assert pair.metadata["profile_family"] == "generalized_involute"
    assert relative_concavity > 0.05
    assert (
        pair.metadata["drive_centrode_outline_distance"]
        <= pair.metadata["centrode_fidelity_tolerance"]
    )
