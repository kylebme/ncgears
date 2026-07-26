from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import ncgear
from ncgear.result import GearPair


def _fixture_pair(directory: Path) -> GearPair:
    directory.mkdir(parents=True)
    outline = np.array(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]]
    )
    np.savetxt(
        directory / "drive.csv",
        outline,
        delimiter=",",
        header="x,y",
        comments="",
    )
    np.savetxt(
        directory / "driven.csv",
        0.5 * outline,
        delimiter=",",
        header="x,y",
        comments="",
    )
    metadata = {
        "name": "fixture",
        "drive_teeth": 16,
        "driven_teeth": 8,
        "average_angular_ratio": 2.0,
        "center_distance": 12.5,
        "maximum_transmission_error": 1e-4,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return GearPair.load(directory)


def test_result_loads_and_exports_cad_formats(tmp_path: Path) -> None:
    pair = _fixture_pair(tmp_path / "fixture")

    assert pair.drive_teeth == 16
    assert pair.driven_teeth == 8
    assert pair.ratio == 2.0
    assert pair.placed_driven_outline[0, 0] == pytest.approx(12.0)
    assert Path(pair) == pair.directory
    assert pair / "drive.csv" == pair.directory / "drive.csv"

    svg = pair.export_svg(tmp_path / "pair.svg")
    dxf = pair.export_dxf(tmp_path / "pair.dxf")
    assert '<polygon id="drive"' in svg.read_text(encoding="utf-8")
    assert '<polygon id="driven"' in svg.read_text(encoding="utf-8")
    assert "LWPOLYLINE" in dxf.read_text(encoding="ascii")
    assert "$INSUNITS" in dxf.read_text(encoding="ascii")


def test_transmission_frontend_samples_motion_law(tmp_path: Path) -> None:
    sentinel = object()
    with patch("ncgear.api._run_generator", return_value=sentinel) as run:
        result = ncgear.generate(
            "phi - 0.08*sin(2*phi)",
            name="two_lobe",
            teeth=20,
            samples=1024,
            output_directory=tmp_path,
        )

    assert result is sentinel
    table = np.loadtxt(
        tmp_path / "two_lobe" / "transmission.csv",
        delimiter=",",
        skiprows=1,
    )
    assert table.shape == (1024, 5)
    assert np.min(table[:, 2]) > 0.0
    assert run.call_args.kwargs["cycle_delta"] == pytest.approx(2.0 * math.pi)
    assert run.call_args.kwargs["active_end"] == pytest.approx(2.0 * math.pi)


def test_centrode_frontend_samples_radius_and_derivatives(tmp_path: Path) -> None:
    with patch("ncgear.api._run_generator") as run:
        ncgear.generate_from_centrode(
            "1 + 0.08*cos(2*phi)",
            name="centrode",
            teeth=20,
            samples=1024,
            output_directory=tmp_path,
        )

    table = np.loadtxt(
        tmp_path / "centrode" / "centrode.csv",
        delimiter=",",
        skiprows=1,
    )
    assert table.shape == (1024, 4)
    assert table[0, 1] == pytest.approx(1.08)
    assert table[0, 2] == pytest.approx(0.0)
    assert table[0, 3] == pytest.approx(-0.32)
    assert run.call_args.kwargs["input_flag"] == "--centrode-csv"


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("other + phi", "only contain phi"),
        ("phi - 0.6*sin(2*phi)", "strictly positive"),
    ],
)
def test_invalid_transmission_is_rejected(
    expression: str, message: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=message):
        ncgear.generate(
            expression,
            samples=1024,
            output_directory=tmp_path,
        )


def test_result_name_cannot_escape_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="name must"):
        ncgear.generate(
            "phi",
            name="../outside",
            samples=1024,
            output_directory=tmp_path,
        )
