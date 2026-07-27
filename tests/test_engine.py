from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

import ncgears


def _assert_verified_pair(pair: ncgears.GearPair) -> None:
    assert pair.metadata["geometry_backend"] == "shapely-geos"
    assert pair.metadata["geometry_precision"] == "double"
    assert pair.metadata["geometry_worker_limit"] >= 1
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


def test_python_engine_generates_open_segment(tmp_path: Path) -> None:
    pair = ncgears.generate(
        "1.8*phi + 0.03*sin(phi)",
        name="open_segment",
        teeth=12,
        open_=True,
        drive_end=2.4,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    assert pair.metadata["topology"] == "open"
    assert pair.ratio == pytest.approx(
        (1.8 * 2.4 + 0.03 * math.sin(2.4)) / 2.4,
        abs=1e-7,
    )


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


def test_nonconvex_centrode_uses_global_sweep(tmp_path: Path) -> None:
    pair = ncgears.generate(
        "phi + 0.18*sin(2*phi)",
        name="nonconvex",
        teeth=32,
        samples=1024,
        samples_per_radian=20,
        output_directory=tmp_path,
    )

    _assert_verified_pair(pair)
    assert pair.metadata["centrodes_are_convex"] is False
    assert pair.metadata["maximum_drive_curvature"] > 0.0
