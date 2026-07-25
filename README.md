# General-Purpose Noncircular Gear Generator

This C++20/CGAL generator constructs conjugate noncircular gears for prescribed
motion laws. Its centrode equations descend from Uwe Bäsel's paper,
["Determining the geometry of noncircular gears for given transmission
function"](https://arxiv.org/abs/1905.02642), but tooth construction no longer
uses the paper's convex-only local branch assembly.

## Relation to existing work

Xu et al.,
["Computational Design and Optimization of Non-Circular
Gears"](https://doi.org/10.1111/cgf.13939), address a related but different
design problem. Their method starts from two target silhouettes, searches for a
rotation center in each, and modifies their polar boundary functions to obtain
a compatible closed gear pair. This is useful when resemblance to supplied
shapes is the primary objective. The present generator instead starts from a
transmission function or one driving centrode and treats the requested motion
as fixed. It does not currently optimize rotation centers or similarity to two
target silhouettes.

The pitch-curve kinematics are substantially the same. If `r_D(phi)` is the
driving pitch radius and `a` is the fixed center distance, both methods use

```text
psi1(phi) = r_D(phi) / (a - r_D(phi))
r_D(phi)  = a * psi1(phi) / (1 + psi1(phi))
r_F(phi)  = a / (1 + psi1(phi))
```

Xu et al. solve `a` by imposing the required integral of `psi1` over a drive
cycle, then integrate and invert the resulting motion to recover the mating
pitch curve. Direct centrode input in this project performs the same
center-distance solve and converts the result into the common motion-law
representation used by the rest of the generator. Consequently, the
silhouette and rotation-center optimization from Xu et al. could be used as an
upstream method here: its output transmission derivative or polar centrode is
compatible with this generator's input.

The tooth constructions are different. The Xu et al. paper describes
initializing involute teeth on the driver, increasing individual tooth heights
when needed for continued engagement, and obtaining the follower by
rotate-and-carve material removal. In the
[released implementation at revision
`5654e79`](https://github.com/xuhaocuhk/non-circular-gears/tree/5654e790907cb80d3481f11ccb80ef3482cc6c16),
the normal execution path instead displaces uniformly sampled boundary points
by a piecewise sinusoidal function with flat tip and root intervals
([`teeth_involute_sin`](https://github.com/xuhaocuhk/non-circular-gears/blob/5654e790907cb80d3481f11ccb80ef3482cc6c16/python_dual_gear/gear_tooth.py#L46-L62)).
That construction is not an involute in the gear-geometric sense: it has no
base curve, unwinding construction, or rack-generated envelope. The released
entry path also calls the routine with its torque and continued-engagement
height adjustments disabled
([`add_teeth`](https://github.com/xuhaocuhk/non-circular-gears/blob/5654e790907cb80d3481f11ccb80ef3482cc6c16/python_dual_gear/gear_tooth.py#L95-L104)).
The subsequently carved follower is nevertheless a sampled conjugate envelope
of that finished driver profile.

This project generates the involute-rack family by sweeping a straight-flanked
rack cutter with a rounded tip. Its cycloidal-rack family either uses the
corresponding rack sweep or carves the mate with the complete finished master.
These choices provide controlled module, pressure angle, addendum, dedendum,
and root-fillet parameters, but they do not establish that every accepted
design is suitable under load. Both projects approximate continuous relative
motion by finitely many poses. This project uses exact-construction Boolean
operations for the polygons at those poses and performs separate sampled
overlap and contact-motion checks; those checks reduce numerical ambiguity but
are not a proof over every continuous phase. Xu et al. report fabricated
examples, whereas this project does not yet include experimental load,
efficiency, wear, or lifetime validation.

Conjugacy should not be confused with low sliding. For an external pair, the
relative sliding speed at a contact point `Q` is proportional to
`(1 + psi1) * distance(Q, P)`, where `P` is the instantaneous pitch point.
Sliding is therefore zero only as contact passes through `P`, including for an
ordinary involute pair. The Xu et al. optimization minimizes silhouette change
and an idealized peak torque ratio; it does not optimize flank sliding or
friction. This project reports a conservative sliding-velocity factor but does
not minimize it or perform a tribological analysis.

The supported domains also differ. The released Xu et al. pipeline is aimed at
closed periodic silhouette-derived pairs and primarily uses integral cycle
multiplicity. This project accepts compatible rational closed ratios and finite
open motion segments. Conversely, Xu et al. provide automatic two-silhouette
fitting and rotation-center search, which are not implemented here. Both polar
representations require a single radius for each angle about the selected
center; silhouette features hidden behind an outer ray intersection cannot be
preserved exactly by that representation.

The current pipeline implements:

- drive and driven centrodes from a bounded, strictly increasing motion law
- one consistent quintic representation of sampled `psi`, `psi1`, and `psi2`
- global material removal by a complete swept rack-cutter solid
- automatic support for convex, inflected, and globally nonconvex centrodes
- rounded involute-rack and smooth cycloidal-rack cutter families
- direct conjugate-mate generation by sweeping the finished master gear
- natural cutter-generated undercut and root fillets
- exact-kernel regularized Boolean operations for cutter removal
- hub-connected component selection after the global sweep
- dense whole-cycle solid-overlap verification
- independent recovery of contact motion from the finished solids
- sliding-velocity, root-radius, curvature, and sweep-resolution diagnostics
- closed unequal-ratio pairs and finite open gear segments
- sampled motion laws generated from arbitrary SymPy expressions
- direct polar drive-centrode input, with automatic scale and mate recovery

Built-in samples:

- `paper`: the Section 8 example
- `two_lobe`: a smooth two-lobed transmission
- `asymmetric`: a nonsymmetric two-term Fourier transmission
- `three_lobe`: a smooth three-lobed transmission
- `nonconvex_inflected`: a regression with a drive-centrode curvature sign change
- `cycloidal_two_lobe`: a conjugate pair generated with the cycloidal rack

## Build and generate

```bash
python3 -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/ncgear_generate --sample all --out out
python3 scripts/render_samples.py --input out
python3 scripts/generate_gallery.py --out out
```

Each sample directory contains `drive.csv`, `driven.csv`, `metadata.json`,
`drive.png`, `driven.png`, and `pair.png`.

## Centrode input

Specify the driving gear's centrode directly as a positive polar radius
`r(phi)`. The radius is a shape in arbitrary units; the generator scales its
arc length to `teeth * pi * module`. The reference center distance uses the same
units and controls the angular ratio. If it is omitted, the generator solves it
for the requested cycle advance (one mate revolution by default):

```bash
python3 scripts/generate_centrode.py "1 + 0.08*cos(2*phi)" \
  --name centrode_two_lobe --teeth 20

python3 scripts/generate_centrode.py \
  "1 + 0.055*cos(phi) - 0.025*sin(2*phi)" \
  --name centrode_asymmetric --teeth 24
```

The same mode is available as a Python API:

```python
from ncgear import generate_from_centrode

directory = generate_from_centrode(
    "1 + 0.08*cos(2*phi)",
    name="centrode_two_lobe",
    teeth=20,
    output_directory="out",
)
```

Unequal tooth counts are selected through the mate's angular advance. For
example, a fivefold-symmetric 100:40 pair uses a `5:2` advance and the
conjugate-carved profile:

```python
import math

directory = generate_from_centrode(
    "1 + 0.08*cos(5*phi)",
    name="centrode_five_lobe_5_to_2",
    teeth=100,
    target_cycle_delta=5 * math.pi,
    profile="cycloidal",
    output_directory="out",
)
```

The centrode need not be convex. Very deep concavities can nevertheless make a
global rack sweep self-occlude and disconnect the intended gear body. Those
degenerate results are rejected even if a leftover component happens to pass a
pair-overlap check.

For direct C++/CLI integration, pass a uniformly sampled four-column
`phi,radius,radius1,radius2` file through `--centrode-csv`. Closed inputs omit
the duplicate endpoint; open inputs include it and require padding around the
active interval.

For a broader regression covering a focus-mounted ellipse, the paper reference,
a Pascal limacon, deep multi-lobed curves, mixed harmonics through order seven,
and silhouette-inspired rounded-square, heart, teardrop, kidney, triangle,
crescent, and organic centrodes:

```bash
python3 scripts/stress_centrodes.py --out out
```

The command continues after individual failures and writes the inputs,
diagnostics, rendered pairs, and a machine-readable
`out/centrode_stress_report.json`.

## SymPy transmission functions

The Python frontend differentiates a string expression three times with SymPy,
samples the expression and derivatives, and passes the resulting table to the
C++/CGAL generator:

```bash
# Closed 2:1 pair: 20 drive teeth and 10 driven teeth.
python3 scripts/generate_sympy.py "2*phi" \
  --name ratio_2_to_1 --teeth 20

# Finite variable-ratio segments with 12 teeth on each active segment.
python3 scripts/generate_sympy.py "1.8*phi + 0.03*sin(phi)" \
  --open --name open_variable_ratio --teeth 12 --drive-end 2.4

# Special functions supported by SymPy/SciPy are accepted as well.
python3 scripts/generate_sympy.py "phi + 0.05*erf(phi)" \
  --open --name open_erf --teeth 12 --drive-end 2.4

# Use the smooth cycloidal rack profile.
python3 scripts/generate_sympy.py "phi - 0.08*sin(2*phi)" \
  --name cycloidal_expression --teeth 28 --profile cycloidal \
  --cycloidal-rolling-factor 1
```

Closed functions must have periodic first through third derivatives over both
the drive period and one driven-gear revolution. Their cycle advance determines
the driven tooth count, and `z1 * period / cycle_delta` must be an integer.
Open functions require a bounded, strictly positive derivative on the padded
finite domain; their average ratio may be any positive value subject to cutter,
hub, solid-interference, and positive-contact checks.

`--allow-nonconvex` remains accepted for command-line compatibility but is no
longer needed. Nonconvexity is measured and reported, not treated as a special
generation mode.

The extended gallery contains four closed ratios (`1:2`, `3:2`, `5:3`, and
`2:1`) and four finite open transmissions (exponential acceleration, smooth
S-curve, localized step, and quadratic ramp). See
[`docs/limitations.md`](docs/limitations.md) for the remaining physical and
numerical boundaries.
