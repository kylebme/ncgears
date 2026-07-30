# Physical and numerical boundaries

`ncgears` has one tooth-geometry engine: hybrid analytical involute
construction. It evaluates generalized-involute working flanks, rounded
rack-tip fillet envelopes, and addendum/dedendum offsets directly. Shapely/GEOS
arranges those curves into solids, and a rolling-pair pass removes interference
only inside the pitch-side root regions.

No complete rack solid is swept through the gear. This is important for
nonconvex centrodes, where a remote part of a fictitious rack can cross a
concavity and erase valid material.

## What "general purpose" means

The program either returns a pair that passes its solid checks or rejects the
requested motion and design parameters with a specific error. It does not
imply that every scalar function is physically realizable by one pair of
fixed-center external gears.

A closed external pair requires:

- a bounded, strictly positive transmission derivative
- a sufficiently smooth motion law
- rational cycle advance and integral drive/driven tooth counts
- motion history compatible with both rigid gear revolutions
- enough root material for a connected solid
- positive contact without solid interference

Exact dwells, infinite ratios, ratio reversals, discontinuous velocity,
irrational closed ratios, and incompatible driven-cycle histories require an
open segment or a different mechanism topology.

## Analytical geometry and tessellation

The working flanks use the analytical straight-rack envelope equation. GEOS
finds candidate intersections between sampled analytical curves, and SciPy
refines them against the continuous parameterizations. Rounded rack-tip
fillets are analytical envelope branches, not post-processing arcs.

The delivered outline is still a tessellated polygon. Metadata reports:

- `maximum_envelope_residual`
- `maximum_envelope_tangency_residual`
- `maximum_intersection_residual`
- `maximum_fillet_root_residual`
- `maximum_analytic_chord_error`

All dimensional tolerances are factors of module defined in
`ncgears/_policy.py`; there is no hidden unit-sized tolerance floor. Very large
coordinate offsets, modules close to machine precision, or extremely thin
remnants remain unsuitable inputs.

## Rolling root generation and verification

Root undercuts and other non-working interference are resolved by running the
finished analytical pair through the requested motion. Material removal is
restricted to the pitch-side root masks, so the certified involute working
flanks are not modified.

The rolling trim and final solid checks use four staggered phase grids. Several
off-grid phases additionally recover the first solid-contact angle on both
sides of the requested pose. This catches grid-aligned errors, but it is a
dense sampled verification rather than a formal interval-arithmetic proof over
every real-valued phase.

Increase `samples_per_radian` when:

- the tooth scale is very small relative to center distance
- the motion law has high-frequency content
- the reported contact correction is too large
- manufacturing tolerance approaches the reported tessellation error

## Open segments

Open bodies are assembled from the same analytical flank, fillet, addendum, and
dedendum curves as closed gears. Each curve is clipped in rolling-arc parameter
space to the active interval. The body then closes along one quarter of the
centrode. Input padding supplies enough motion history to solve endpoint tooth
geometry, but does not define or clip the finished body.

An open output is intended for its finite operating range and must not be
treated as a continuously rotating closed pair.

## Sampled motion representation

CSV transmission input contains `psi`, `psi1`, `psi2`, and `psi3`. Geometry
uses a single piecewise-quintic Hermite motion interpolant constrained by
`psi`, `psi1`, and `psi2`; all evaluated derivatives through third order come
from that interpolant. Input samples are uniformly spaced, and closed sample
grids are aligned with the reduced tooth ratio.

## Surface sliding

Conjugacy and surface sliding are different quantities. For an external pair,
the relative sliding speed at contact point `Q` is proportional to

```text
(1 + psi1) * distance(Q, P)
```

where `P` is the instantaneous pitch point. An involute pair therefore has zero
sliding only as contact crosses the pitch point. Metadata reports a conservative
sliding-velocity factor and the angular correction at which the tessellated
finished solids establish contact.

## Engineering analysis not included

The geometry report does not replace:

- Hertzian contact-stress analysis
- tooth-root bending and fatigue analysis
- elastic transmission-error analysis
- lubrication and flash-temperature analysis
- three-dimensional lead/crowning design
- shaft, bearing, and housing deflection analysis
- manufacturing-process validation

Those require load, material, face width, speed, lubrication, and tolerance
inputs outside the present two-dimensional geometry contract.
