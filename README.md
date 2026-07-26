# ncgear

`ncgear` generates conjugate noncircular gear pairs from a motion law or a
pitch-curve shape. Its public interface is Python; platform wheels bundle the
C++/CGAL geometry engine, so users do not need to build or call C++ code.

The generator creates 2D outlines, verifies the assembled pair for interference
and contact-motion error, and exports CSV, SVG, DXF, JSON, and optional PNG
files. It supports closed gears, finite open segments, nonconvex pitch curves,
unequal ratios, and involute-rack or cycloidal-rack tooth families.

> **Project status:** alpha. Generated geometry should be reviewed for the
> intended material, manufacturing process, load, speed, and tolerances.

## Install

Download a wheel for your operating system from the latest GitHub Actions run
or release, then install it with pip:

```bash
python -m pip install ncgear-0.1.0-*.whl
```

Once the project is published to PyPI, installation becomes:

```bash
python -m pip install ncgear
```

PNG previews are optional:

```bash
python -m pip install "ncgear[plot]"
```

Prebuilt wheels target 64-bit Linux, Windows, Intel macOS, and Apple Silicon
macOS. Building from source requires CMake 3.24+, a C++20 compiler, and CGAL.

## Generate a gear pair

Describe the desired relationship between the drive angle `phi` and the driven
angle. Here the driven gear speeds up and slows down twice per revolution while
returning to the same 1:1 average ratio:

```python
import ncgear

pair = ncgear.generate(
    "phi - 0.08*sin(2*phi)",
    teeth=24,
    module=1.5,
    name="two_lobe",
)

print(pair.summary())
pair.export_dxf("two_lobe.dxf")
pair.export_svg("two_lobe.svg")
```

`module` and all exported coordinates use millimetres. The returned
`GearPair` also provides:

```python
pair.drive_outline           # (N, 2) NumPy array
pair.driven_outline          # centered on its own shaft
pair.placed_driven_outline   # translated into assembled position
pair.center_distance
pair.drive_teeth
pair.driven_teeth
pair.ratio
pair.maximum_transmission_error
pair.metadata                # complete verification report
pair.directory               # CSV and JSON source files
pair.render()                # pair.png; requires ncgear[plot]
```

The output directory defaults to `out/<name>/`. Each successful generation
contains `drive.csv`, `driven.csv`, `metadata.json`, and the sampled input.

## Start from a pitch curve

If the drive gear's pitch radius is easier to describe than its motion law, use
a centrode expression:

```python
pair = ncgear.generate_from_centrode(
    "1 + 0.08*cos(2*phi)",
    teeth=20,
    module=1.0,
    name="centrode_two_lobe",
)
```

The radius may use arbitrary units; ncgear scales its arc length to the
requested tooth count and module. By default it solves the center distance for
one mate revolution. A specific ratio can be selected with
`target_cycle_delta`. For example, a five-lobed 5:2 angular ratio uses:

```python
import math

pair = ncgear.generate_from_centrode(
    "1 + 0.05*cos(5*phi)",
    teeth=100,
    target_cycle_delta=5 * math.pi,
    profile="cycloidal",
    name="five_to_two",
)
```

## Closed and open designs

Closed gears require a smooth, strictly increasing motion whose cycle advance
produces an integer mate tooth count. A simple 2:1 pair is:

```python
pair = ncgear.generate("2*phi", teeth=20)  # 20 drive teeth, 10 driven teeth
```

Finite, non-repeating motion can be generated as an open segment:

```python
pair = ncgear.generate(
    "1.8*phi + 0.03*sin(phi)",
    open_=True,
    drive_end=2.4,
    teeth=12,
    name="finite_segment",
)
```

Open results are only valid over the requested active interval and must not be
used as continuously rotating closed gears.

## Command line

Python is the primary API, but a small command is included for quick trials:

```bash
ncgear "phi - 0.08*sin(2*phi)" --teeth 24 --module 1.5 \
  --name two_lobe --dxf two_lobe.dxf --render

ncgear "1 + 0.08*cos(2*phi)" --centrode --teeth 20 \
  --name centrode_two_lobe
```

Run `ncgear --help` for all commonly used options.

## What is verified

The native engine uses swept rack-cutter solids and exact-construction polygon
Boolean operations. A successful result includes checks for:

- simple, hub-connected gear bodies
- sampled whole-cycle solid interference
- contact motion recovered from the finished outlines
- sweep resolution, root radius, tip thickness, and centrode curvature
- sliding-velocity and undercut diagnostics

These geometry checks are not load-rating or manufacturing certification.
See [physical and numerical limitations](docs/limitations.md) before fabricating
a design.

## Development

On Debian/Ubuntu:

```bash
sudo apt-get install cmake libcgal-dev
python -m pip install -e ".[dev]"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python -m pytest
python -m build
```

The GitHub Actions workflow runs the Python and C++ tests, then uses
`cibuildwheel` to produce repaired wheels for Linux, Windows, Intel macOS, and
Apple Silicon macOS.

## Method and prior work

The pitch-curve equations follow Uwe Bäsel,
["Determining the geometry of noncircular gears for given transmission
function"](https://arxiv.org/abs/1905.02642). Tooth geometry is constructed by
sweeping a parameterized rack cutter rather than assembling only locally convex
branches. The silhouette-fitting problem addressed by Xu et al.,
["Computational Design and Optimization of Non-Circular
Gears"](https://doi.org/10.1111/cgf.13939), is complementary: a fitted
transmission derivative or polar centrode can be passed into ncgear.

Contributions and reproducible test cases are welcome through the
[issue tracker](https://github.com/kylebme/geargen5-5/issues).

## License

ncgear is distributed under the GNU General Public License v3.0 or later. Its
2D Boolean geometry uses CGAL's GPL-licensed `Polygon_set_2` package. Projects
that cannot comply with the GPL need an appropriate commercial CGAL license
and should review the licensing of ncgear itself with their legal counsel.
