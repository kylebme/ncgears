# Physical and numerical boundaries

The generator treats convexity as a diagnostic rather than a construction
precondition. Inflections and nonconvex centrodes are handled by subtracting
the global swept cutter solid from a gear blank and selecting the largest
hub-connected result. Nonconvex centrodes are currently not properly supported and a best effort gear will be generated.

## What "general purpose" means

The program either returns a pair that passes its solid checks or rejects the
requested motion/design parameters with a specific error. It does not imply
that every mathematical scalar function is physically realizable by one pair
of fixed-center external gears.

A closed external pair requires:

- a bounded, strictly positive derivative `psi1`
- a sufficiently smooth motion law
- rational cycle advance and integral drive/driven tooth counts
- motion history compatible with both rigid gear revolutions
- enough root material for a hub-connected solid
- positive contact without solid interference

Exact dwells (`psi1 == 0`), infinite ratios, ratio reversals, discontinuous
velocity, irrational closed ratios, and incompatible driven-cycle histories
require an open segment or a different mechanism topology.

## Surface sliding

Conjugacy and surface sliding are different quantities. For an external pair,
the relative sliding speed at contact point `Q` is proportional to

```text
(1 + psi1) * distance(Q, P)
```

where `P` is the instantaneous pitch point. Ordinary involute and cycloidal
teeth therefore have zero sliding only as contact crosses the pitch point.
The generated pair has zero prescribed-motion error in its ideal envelope;
the metadata also reports a conservative sliding-velocity factor and the
angular correction at which the tessellated finished solids establish contact.

## Sampled motion representation

CSV input still contains `psi`, `psi1`, `psi2`, and `psi3` for compatibility.
Geometry now uses a single piecewise-quintic Hermite motion interpolant
constrained by `psi`, `psi1`, and `psi2`; all evaluated derivatives through
third order come from that interpolant. This prevents the mutually
inconsistent linear interpolation used by the original implementation.

Input samples are assumed to be uniformly spaced. The frontend writes uniform
samples and aligns a closed sample grid with the reduced tooth ratio.

## Swept-solid approximation

The continuum cutter motion is approximated by dense cutter poses with a small
conservative inward cutter margin. Regularized Boolean subtraction and
component selection use Shapely/GEOS double-precision floating-point geometry.
The cycloidal mate sweep also expands its master cutter by 0.00175 module to
keep its sampled envelope conservative between poses. Metadata records the
backend, precision, pose count, and maximum angular step so downstream tooling
can audit the resolution.

GEOS uses robust predicates but does not retain an exact rational construction
for every new vertex. In this algorithm, comparisons against the former CGAL
implementation show Boolean differences far below cutter-pose discretization
at normal gear scales. Very large coordinate offsets, modules close to machine
precision, or extremely thin remnants remain unsuitable inputs.

The finished polygons are checked at at least four phases per tooth over the
whole requested cycle. Several off-grid phases additionally recover the first
solid-contact angle on both sides of the requested output angle. This catches
phase errors that a construction-only check would miss, but it is not a formal
interval-arithmetic proof over every real-valued phase.

Increase `--samples-per-radian` when:

- tooth scale is very small relative to center distance
- the motion law has high-frequency content
- the reported contact correction is too large
- downstream manufacturing tolerance is close to the sweep step error

## Profile families

`involute` uses a straight-flanked rack with a rounded cutter tip and generates
both gears with complementary rack phase. `cycloidal` first generates the
master with a smooth cycloidal-eased rack flank controlled by
`--cycloidal-rolling-factor`, then generates the mate by sweeping the complete
finished master solid through the prescribed relative motion. The second
construction is slower but directly enforces conjugacy for a rack profile that
is not self-conjugate.

The cycloidal backend is a rack-generated cycloidal family, not a pin-wheel,
gerotor, or eccentric cycloidal reducer. Those require different relative
motion and cutter definitions.

## Open segments

Open gears are swept over the active interval plus 2.5 mean tooth pitches of
padding and then intersected with an annular body sector. The input CSV domain
must provide at least that much motion padding. Open output is intended for
finite operating ranges and must not be treated as a closed continuously
rotating pair.

## Engineering analysis not yet included

The geometry report does not replace:

- Hertzian contact-stress analysis
- tooth-root bending and fatigue analysis
- elastic transmission-error analysis
- lubrication and flash-temperature analysis
- three-dimensional lead/crowning design
- shaft, bearing, and housing deflection analysis
- manufacturing-process validation

Those require load, material, face width, speed, lubrication, and tolerance
inputs that are outside the present two-dimensional geometry contract.
