from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import ncgears
from ncgears.cli import build_parser
from ncgears.result import GearPair


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
        "topology": "closed",
        "input_mode": "transmission",
        "drive_teeth": 16,
        "driven_teeth": 8,
        "average_angular_ratio": 2.0,
        "center_distance": 12.5,
        "maximum_transmission_error": 1e-4,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return GearPair.load(directory)


def test_removed_profile_backends_are_not_public_options() -> None:
    for frontend in (
        ncgears.generate,
        ncgears.generate_from_transmission,
        ncgears.generate_from_centrode,
    ):
        parameters = inspect.signature(frontend).parameters
        assert "profile" not in parameters
        assert "cycloidal_rolling_factor" not in parameters

    assert "--profile" not in build_parser().format_help()


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


def test_result_renders_animated_gif(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pillow = pytest.importorskip("PIL.Image")
    pair = _fixture_pair(tmp_path / "fixture")
    phi = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
    np.savetxt(
        pair.directory / "transmission.csv",
        np.column_stack(
            (
                phi,
                2.0 * phi,
                np.full_like(phi, 2.0),
                np.zeros_like(phi),
                np.zeros_like(phi),
            )
        ),
        delimiter=",",
        header="phi,psi,psi1,psi2,psi3",
        comments="",
    )

    output = pair.render_gif(tmp_path / "pair.gif", frames=5, fps=8, dpi=40)

    assert output.read_bytes().startswith(b"GIF8")
    with pillow.open(output) as image:
        assert image.n_frames == 5


def test_transmission_frontend_samples_motion_law(tmp_path: Path) -> None:
    sentinel = object()
    with patch("ncgears.api._run_generator", return_value=sentinel) as run:
        result = ncgears.generate(
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
    with patch("ncgears.api._run_generator") as run:
        ncgears.generate_from_centrode(
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
        ncgears.generate(
            expression,
            samples=1024,
            output_directory=tmp_path,
        )


def test_result_name_cannot_escape_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="name must"):
        ncgears.generate(
            "phi",
            name="../outside",
            samples=1024,
            output_directory=tmp_path,
        )
