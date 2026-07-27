# ncgear

`ncgear` generates noncircular gear pairs from a transmission law or a
pitch-curve shape.

The generator creates 2D outlines, verifies the assembled pair for interference
and contact-motion error, and exports CSV, SVG, DXF, JSON, and PNG
files. It supports closed gears, finite open segments, nonconvex pitch curves,
unequal ratios, and involute-rack or cycloidal-rack tooth families.
The complete application and geometry pipeline are implemented in Python;
Shapely/GEOS provides robust floating-point polygon operations.

> **Project status:** alpha. Generated geometry should be reviewed for the
> intended material, manufacturing process, load, speed, and tolerances.

## Install

```bash
python -m pip install ncgear
```

PNG previews are optional:

```bash
python -m pip install "ncgear[plot]"
```

There is no compiler or system-level CGAL dependency. NumPy, SciPy, SymPy, and
Shapely publish wheels for the commonly used CPython platforms.

## Command line

Python is the primary API, but a small command is included for quick trials:

```bash
ncgear "phi - 0.08*sin(2*phi)" --teeth 24 --module 1.5 \
  --name two_lobe --dxf two_lobe.dxf --render

ncgear "1 + 0.08*cos(2*phi)" --centrode --teeth 20 \
  --name centrode_two_lobe
```

Run `ncgear --help` for all commonly used options.

## Basic python usage

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


## What is verified

The Python engine uses swept rack-cutter solids and Shapely/GEOS regularized
polygon operations. A successful result includes checks for:

- simple, hub-connected gear bodies
- sampled whole-cycle solid interference
- contact motion recovered from the finished outlines
- sweep resolution, root radius, tip thickness, and centrode curvature
- sliding-velocity and undercut diagnostics

`metadata.json` records `geometry_backend: "shapely-geos"`, double-precision
construction, cutter-pose count, and the maximum sweep step. Floating-point
Boolean error is normally far below the error from discretizing cutter motion;
increase `samples_per_radian` when a design operates close to its tolerances.

These geometry checks are not load-rating or manufacturing certification.
See [physical and numerical limitations](docs/limitations.md) before fabricating
a design.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check ncgear tests
python -m build
```

The GitHub Actions workflow tests Python 3.10–3.13, builds a platform-independent
ncgear wheel, and smoke-tests the installed wheel. Shapely supplies its GEOS
runtime through its own platform wheels.

### Migrating from 0.1

No generation or export call needs a native executable anymore. Remove
`generator=...` arguments and `NCGEAR_GENERATOR` configuration from existing
integrations. The legacy `native_generator()` symbol remains importable only to
raise an actionable migration error; it no longer locates or launches a binary.
Generated CSV and JSON layouts and the `GearPair` result API remain compatible.

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

## Project context

This project contains entirely AI generated code. This project has been my personal benchmark for 
determining how capable coding models are for over a year, but now they have saturated this benchmark, 
so I'm releasing the project as an alpha.

## License

ncgear is distributed under the GNU General Public License v3.0 or later. Its
2D Boolean geometry uses the BSD-licensed Shapely package and its LGPL-licensed
GEOS runtime.
