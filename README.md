# General-Purpose Noncircular Gear Generator

This C++20/CGAL generator constructs conjugate noncircular gears for prescribed
motion laws. Its centrode equations descend from Uwe Bäsel's paper,
["Determining the geometry of noncircular gears for given transmission
function"](https://arxiv.org/abs/1905.02642), but tooth construction no longer
uses the paper's convex-only local branch assembly.

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
