from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from unittest.mock import patch

import ezdxf
import numpy as np
import pytest
from ezdxf import units
from ezdxf.addons.drawing import Frontend, RenderContext, recorder

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
        assert "clearance_factor" in parameters

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


def test_dxf_export_is_valid_and_renders_expected_geometry(tmp_path: Path) -> None:
    pair = _fixture_pair(tmp_path / "fixture")
    output = pair.export_dxf(tmp_path / "pair.dxf")

    document = ezdxf.readfile(output)
    auditor = document.audit()

    assert not auditor.has_errors
    assert document.units == units.MM
    assert document.header["$MEASUREMENT"] == 1

    polylines = list(document.modelspace().query("LWPOLYLINE"))
    assert [polyline.dxf.layer for polyline in polylines] == ["DRIVE", "DRIVEN"]
    expected_outlines = (pair.drive_outline, pair.placed_driven_outline)
    for polyline, expected in zip(polylines, expected_outlines, strict=True):
        assert polyline.closed
        assert np.asarray(polyline.get_points("xy")) == pytest.approx(expected[:-1])

    points = list(document.modelspace().query("POINT"))
    assert [point.dxf.layer for point in points] == ["DRIVE", "DRIVEN"]
    assert [tuple(point.dxf.location) for point in points] == pytest.approx(
        [(0.0, 0.0, 0.0), (pair.center_distance, 0.0, 0.0)]
    )
    assert document.header["$PDMODE"] == 3

    backend = recorder.Recorder()
    Frontend(RenderContext(document), backend).draw_layout(document.modelspace())
    bounds = backend.player().bbox()
    expected_points = np.vstack(expected_outlines)
    expected_minimum = np.min(expected_points, axis=0)
    expected_maximum = np.max(expected_points, axis=0)
    assert bounds.extmin.x <= expected_minimum[0]
    assert bounds.extmin.y <= expected_minimum[1]
    assert bounds.extmax.x >= expected_maximum[0]
    assert bounds.extmax.y >= expected_maximum[1]


@pytest.mark.parametrize("gear", ["drive", "driven"])
def test_single_gear_dxf_centers_selected_gear_at_origin(
    gear: str, tmp_path: Path
) -> None:
    pair = _fixture_pair(tmp_path / "fixture")
    output = pair.export_dxf(tmp_path / f"{gear}.dxf", gear=gear)

    document = ezdxf.readfile(output)
    points = list(document.modelspace().query("POINT"))

    assert len(points) == 1
    assert points[0].dxf.layer == gear.upper()
    assert tuple(points[0].dxf.location) == pytest.approx((0.0, 0.0, 0.0))


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


def test_result_creates_interactive_motion_plot(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

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

    figure = pair.plot(show=False)
    slider = figure._ncgears_angle_slider
    drive_patch, driven_patch = figure.axes[0].patches
    initial_drive_transform = drive_patch.get_transform().get_matrix().copy()
    initial_driven_transform = driven_patch.get_transform().get_matrix().copy()

    slider.set_val(math.pi / 4.0)

    assert len(figure.axes) == 2
    assert slider.valmin == pytest.approx(0.0)
    assert slider.valmax == pytest.approx(2.0 * math.pi)
    assert not np.allclose(
        drive_patch.get_transform().get_matrix(),
        initial_drive_transform,
    )
    assert not np.allclose(
        driven_patch.get_transform().get_matrix(),
        initial_driven_transform,
    )
    plt.close(figure)


def test_cli_accepts_interactive_plot_option() -> None:
    args = build_parser().parse_args(["phi", "--plot"])

    assert args.plot is True


def test_cli_accepts_module_scaled_clearance() -> None:
    args = build_parser().parse_args(["phi", "--clearance-factor", "0.075"])

    assert args.clearance_factor == pytest.approx(0.075)


def test_transmission_frontend_samples_motion_law(tmp_path: Path) -> None:
    sentinel = object()
    with patch("ncgears.api._run_generator", return_value=sentinel) as run:
        result = ncgears.generate(
            "phi - 0.08*sin(2*phi)",
            name="two_lobe",
            teeth=20,
            samples=1024,
            output_directory=tmp_path,
            clearance_factor=0.075,
            plot=True,
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
    assert run.call_args.kwargs["clearance_factor"] == pytest.approx(0.075)
    assert run.call_args.kwargs["plot"] is True


def test_centrode_frontend_samples_radius_and_derivatives(tmp_path: Path) -> None:
    with patch("ncgears.api._run_generator") as run:
        ncgears.generate_from_centrode(
            "1 + 0.08*cos(2*phi)",
            name="centrode",
            teeth=20,
            samples=1024,
            output_directory=tmp_path,
            clearance_factor=0.125,
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
    assert run.call_args.kwargs["clearance_factor"] == pytest.approx(0.125)


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


def test_negative_clearance_is_rejected() -> None:
    with pytest.raises(ValueError, match="clearance_factor"):
        ncgears.generate("phi", clearance_factor=-0.01)
