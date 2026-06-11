# Noncircular Gear Samples from arXiv:1905.02642

This is a fresh C++20/CGAL implementation of the construction in Uwe Basel's
paper, ["Determining the geometry of noncircular gears for given transmission
function"](https://arxiv.org/abs/1905.02642).

The generator implements:

- drive and driven centrodes from the transmission function
- rack-generated flank curves
- addendum and dedendum parallel curves
- rack-cutter fillets
- cusp and undercut checks
- the Section 7 branch assembly algorithm
- CGAL-seeded flank/fillet and flank/addendum intersection solving
- CGAL polygon simplicity validation
- exact-kernel CGAL polygon-set validation that the displayed pair does not overlap
- sampled transmission functions generated from arbitrary SymPy expressions
- closed unequal-ratio pairs and finite open gear segments

Built-in samples:

- `paper`: the Section 8 example
- `two_lobe`: a smooth two-lobed transmission
- `asymmetric`: a nonsymmetric two-term Fourier transmission
- `three_lobe`: a smooth three-lobed transmission

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
```

Closed functions must have periodic first through third derivatives over both
the drive period and one driven-gear revolution. Their cycle advance determines
the driven tooth count, and `z1 * period / cycle_delta` must be an integer.
Open functions require a strictly positive derivative on the padded finite
domain; their average ratio may be any positive value subject to the cutter and
polygon validity checks.

The extended gallery contains four closed ratios (`1:2`, `3:2`, `5:3`, and
`2:1`) and four finite open transmissions (exponential acceleration, smooth
S-curve, localized step, and quadratic ramp). See
[`docs/limitations.md`](docs/limitations.md) for the edge cases discovered
while building it.
