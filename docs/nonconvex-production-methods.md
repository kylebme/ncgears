# Direct profile construction for nonconvex centrodes

## Decision

The full-width rack sweep must not be extended to deep nonconvex centrodes. At
a concave pitch point, the tangent rack can intersect distant parts of the
workpiece. That explains the current backend's failure, but it is not a
manufacturing restriction when the finished outline will be molded, sintered,
or printed.

Generate the mathematical form directly with two profile-specific construction
engines:

1. **Generalized involute:** construct the local envelope of straight rack
   flanks analytically, assemble the valid branches against an explicit
   nonconvex blank, and certify the active flanks against the analytic envelope
   and base-curve involute identities.
2. **Strict cycloidal:** construct the tooth flanks as rolling-circle roulettes
   on the noncircular centrodes, assemble conjugate addendum/dedendum branches,
   and certify the rolling invariants and paired contact state.

Do not call an arbitrary finite rack, a locally fitted circular tooth, or a
conjugate sweep alone "involute" or "cycloidal."

## Current implementation

Closed nonconvex inputs with `profile="involute"` now automatically use
`generation_backend="analytic_form"`. The pure-Python implementation:

- evaluates the analytic straight-rack envelope branches in common pitch-arc
  coordinates, which remain monotone through centrode inflections;
- uses Shapely/GEOS for sampled curve intersections, validity repair, and
  tooth-body union;
- refines flank/addendum intersections with SciPy nonlinear least squares;
- stops each working flank at its first root-side cusp using SciPy's bracketed
  root solver;
- constructs root connectors and a radial inset as explicitly non-working
  closure geometry;
- verifies the finished pair for whole-cycle interference and off-grid contact
  recovery;
- records analytic equation, envelope-tangency, intersection, and chord-error
  residuals.

The legacy rack sweep remains in use for convex and open involute inputs.
Strict rolling-circle cycloidal construction and per-segment provenance remain
future work.

## Profile definitions

The API needs an explicit definition because "involute noncircular gear" is not
unique in the literature.

### Generalized involute

For this project, an involute working flank is the envelope of a straight rack
flank at fixed pressure angle while the rack pitch line rolls without slip on
the noncircular centrode. Equivalently, away from singular points, the flank is
an involute of the corresponding noncircular base curve.

For drive centrode `P(phi)`, tangent `T(phi)`, tooth midpoint `chi[k]`, pressure
angle `alpha`, and signed rack coordinate

```text
lambda(k, sign, phi)
    = sign * pi * module / 4
      - center_distance * arc_integral(chi[k], phi),
```

the candidate working flank is

```text
F(k, sign, phi)
    = P(phi)
      + lambda(k, sign, phi)
        * T(phi) * exp(sign * i * alpha) * cos(alpha).
```

This is the analytic envelope formula formerly implemented in commit
`9e64fe4`. It does not require constructing the complete rack solid. Bäsel
derives the associated base curves and proves that these rack-generated flanks
are their involutes.

The construction should use the direct flank formula through centrode
inflections. The base-curve formula contains `1 / curvature` and is unsuitable
as the primary numerical representation near zero curvature; it should be used
as a proof/checking representation on regular intervals.

### Strict cycloidal

A strict cycloidal flank is a roulette of a point fixed to a circle of radius
`rho` rolling without slip on a centrode. For each stored flank sample, the
construction must retain enough parameters to verify:

- the generating circle is tangent to the centrode;
- the traced point remains exactly `rho` from the circle center;
- the signed circle rotation equals traveled pitch-curve arc divided by `rho`;
- the paired addendum and dedendum use the corresponding common generating
  circle and tooth phase.

The present `_cycloidal_tooth_template` does not meet this definition. It blends
linear and cycloidal easing independently in its local x and y coordinates.
For blend values below one it is explicitly not a cycloid, and at blend one
the independent x/y scaling is generally an affine distortion of a cycloid.
The current mate sweep proves conjugacy to that eased master, not strict
cycloidal provenance.

The existing profile should therefore become `eased_rack` (with a compatibility
alias during migration), while a new `cycloidal` profile uses a rolling-circle
parameter such as `cycloidal_generating_radius`.

## Common envelope proof

For any compact generator curve `c(u)` placed by rigid transform

```text
G(phi, u) = R(phi) * c(u) + p(phi),
```

candidate generated points solve the envelope equation

```text
cross(partial G / partial u, partial G / partial phi) = 0.
```

At a regular solution, the generator normal has zero relative normal velocity.
This is the local meshing condition and is the core proof that the generated
flank is conjugate to its generator.

It is not sufficient on its own to prove that two independently generated
workpieces form the requested pair. The pair certificate must additionally
show that, at each intended contact:

- both flank points transform to the same world point;
- their normals are collinear and oppositely oriented;
- the common normal contains the prescribed instantaneous pitch point;
- relative normal velocity is zero;
- the contact parameters advance continuously without switching to a remote
  envelope branch.

## Form-replicated manufacturing

Molding and additive manufacturing reproduce a supplied form; they do not
require a rack, hob, or shaper to reach every point of the finished outline.
The rack or rolling circle in these constructions is therefore a mathematical
generator, not a required production tool.

Only the active working flanks need the involute or cycloidal certificate.
Tip lands, root closures, hub boundaries, and root fillets may be constructed
for clearance, strength, and minimum feature size provided adaptive
full-cycle contact tracing proves that they never enter the working contact
set. If such a closure does enter contact, it must either satisfy the selected
profile definition or the design must be rejected.

Manufacturing compensation belongs after certified two-dimensional geometry:

- axial mold draft may be added without changing the certified transverse
  section;
- uniform shrink scaling preserves the profile construction if the center
  distance, module, and all generator lengths are scaled together;
- nonuniform shrink, anisotropic print compensation, smoothing, and mesh
  decimation can change flank normals and must be applied to a copy followed
  by the same contact and profile checks;
- root fillet and minimum-wall adjustments must remain outside the certified
  active-flank parameter intervals.

## Candidate methods

| Method | Deep concavity | Profile proof | Role |
|---|---|---|---|
| Analytic local straight-flank envelope | Yes, subject to pair interference and root connectivity | Strong generalized-involute proof | Primary involute backend |
| Rolling-circle roulette | Yes, subject to rolling-circle reach and branch validity | Strong strict-cycloidal proof | Primary cycloidal backend |
| Noncircular pin-cycloid pair | Yes, subject to pin reach | Strong circular-pin/cycloidal-envelope proof | Optional distinct topology |
| Circular pinion-shaper envelope | Usually, subject to generator interference | Exact generator-envelope proof; family label depends on cutter and certificate | Reference or optional topology, not required for production |
| Full rack sweep | No for deep concavity | Strong only where the complete rack remains local | Legacy backend for convex/mild cases |
| Arbitrarily truncated rack | Sometimes visually | End caps and truncation can enter the envelope | Reject |
| Local osculating circular teeth | Approximate | Neither exact involute nor exact cycloidal | Reject |
| Master outline plus conjugate mate sweep | Produces a conjugate mate | Proves conjugacy only; does not prove both family labels | Pair validator/fallback, not a family proof |

## Recommended involute implementation

### 1. Restore the analytic geometry as a candidate-curve engine

Port the following pieces from commit `9e64fe4` into Python:

- equal-pitch tooth midpoint solution `chi[k]`;
- drive and driven centrodes and tangents;
- `lambda` and analytic drive/driven flank equations;
- addendum and dedendum parallel curves;
- cutter-tip fillet envelope and cusp equations.

Do not restore its convex-only directional branch choices.

### 2. Replace branch assumptions with an arrangement

For every tooth:

1. Generate both analytic flank branches over a conservative local contact
   interval.
2. Split them at derivative zeros, cusps, inflections, addendum/root
   intersections, self-intersections, and intersections with neighboring
   teeth.
3. Add the regularized nonconvex addendum blank and root/hub boundary.
4. Polygonize the complete curve arrangement.
5. Classify faces using tooth phase, local envelope orientation, hub
   connectivity, and intended pitch-curve containment.
6. Extract the boundary of the selected material faces.

This keeps remote portions of a physical rack body out of the mathematical
construction while retaining the exact straight-flank envelope.

The arrangement may join working branches with non-profiled tip lands and root
fillets. It must record the exact transition parameters so the verifier can
prove that all intended contact stays on certified working branches.

### 3. Preserve segment provenance

Each output boundary segment should retain:

```text
family: involute
source: rack_flank | cutter_fillet | addendum | dedendum | hub
gear: drive | driven
tooth_index
flank_sign
parameter_start
parameter_end
```

Profile proof must operate on these analytic segments before final tessellation.

## Recommended cycloidal implementation

### 1. Define rolling-circle parameters

Use a physical generating radius `rho`, not an easing factor. Validate it
against local centrode curvature, tooth pitch, cusp formation, addendum height,
and contact ratio.

### 2. Generate paired roulettes

Parameterize both centrodes by common pitch arc. For each tooth phase, roll the
generating circle along the appropriate centrode and construct the external and
internal roulette branches. Pair one gear's addendum-generating branch with the
mate's dedendum-generating branch using the same generating-circle state.

### 3. Assemble with the same arrangement kernel

Split roulette branches at cusps and intersections, polygonize them with the
blank/root curves, classify material faces, and retain generating-circle
parameters on every working segment.

### 4. Offer a pin-cycloid topology

If both members do not need conventional external teeth, a noncircular
pin-cycloid pair has a cleaner proof and a compact physical generator. Define
the pin member and its tool path first, then obtain the cycloidal member from
the conjugate envelope equation. Keep this as a distinct topology because its
contact, stress, backlash, and fabrication constraints differ from two
conventional toothed wheels.

## Optional generator-equivalence backends

A circular pinion generator can still be useful as an independent envelope
implementation or for reproducing an existing shaper-defined profile. It is
not needed to make a moldable or printable gear and is not the preferred fix
for this project.

If implemented, expose its mathematical generator parameters explicitly:

```text
generation_backend = "pinion_shaper"
shaper_teeth
shaper_profile
shaper_profile_shift
shaper_tip_clearance
```

Use the analytic envelope equation for the final flanks; a dense Boolean cutter
sweep may seed intervals but should not be the source of the certified working
curve. Cutter retraction and machine access are deliberately outside the
geometry contract.

Published CNC shaping work specifically identifies pinion shaping as the
alternative for noncircular external gears with concave pitch curves and
demonstrates a three-linkage process. That result supports the feasibility of
compact-generator envelopes, but it does not make a shaper backend necessary
for form-replicated manufacturing or prove that its flanks satisfy this
project's selected involute or cycloidal definition.

## Verification and proof certificate

A production result should not rely only on sampled polygon overlap. Store:

```text
profile_family
generation_backend
generator_parameters
maximum_envelope_residual
maximum_normal_velocity_residual
maximum_pitch_point_normal_residual
maximum_involute_residual        # involute only
maximum_roulette_radius_error    # cycloidal only
maximum_roulette_roll_error      # cycloidal only
maximum_centrode_outline_distance
minimum_nonworking_clearance
minimum_tooth_thickness
minimum_root_thickness
minimum_feature_radius
```

The involute certificate checks both the analytic rack-envelope equation and,
on regular base-curve intervals, the tangent-string/involute identity.

The cycloidal certificate checks the rolling-circle invariants and paired
roulette state.

Both certificates also require adaptive contact tracing over the complete
cycle, not only recovery of collision at six off-grid phases.

Exported manufacturing variants must name the compensation transform and
either reference an unchanged certified transverse section or carry a new
certificate produced after the transform.

## Acceptance cases

At minimum:

- circular gears against closed-form involute/cycloidal references;
- the Bäsel reference example and its published checkpoints;
- a mild five-lobe nonconvex case already accepted by the rack backend;
- `1 + 0.08*cos(5*phi)`, 100:40, which the full rack distorts by about
  5.89 modules;
- `crazy_heart`, whose full-rack result loses the intended cleft;
- a centrode with exact curvature-zero crossings;
- deliberately impossible cases rejected for pair interference, disconnected
  root material, inadequate tooth thickness, or an unavoidable roulette cusp;
- compensated manufacturing variants that preserve the certificate under
  uniform scaling and fail it under deliberately excessive anisotropic
  distortion.

Success requires centrode fidelity, profile-certificate residuals, valid simple
bodies, continuous intended contact on certified flanks, adequate root
connectivity and thickness, and no solid interference.

## Staged delivery

1. Rename the current `cycloidal` profile to `eased_rack` and make profile
   provenance explicit.
2. Port the old analytic involute equations and build the common arrangement
   kernel.
3. Ship `involute + analytic_form` for nonconvex inputs.
4. Implement strict rolling-circle cycloidal branches on the same arrangement
   kernel.
5. Replace sampled contact recovery with adaptive contact tracing and emit the
   proof certificate.
6. Add manufacturing-copy export with uniform shrink/draft metadata and
   mandatory revalidation for any operation that changes the transverse
   profile.

## Primary references

- Uwe Bäsel, *Determining the geometry of noncircular gears for given
  transmission function*, arXiv:1905.02642. The paper derives the analytic
  rack-envelope flanks and proves that they are involutes of noncircular base
  curves, while its construction algorithm assumes convex centrodes.
- F. Zheng, L. Hua, X. Han, B. Li, and D. Chen, *Linkage model and
  manufacturing process of shaping non-circular gears*, Mechanism and Machine
  Theory 96 (2016), 192-212, DOI 10.1016/j.mechmachtheory.2015.09.010. The
  paper identifies shaping as the alternative to hobbing for concave pitch
  curves and demonstrates the process on a CNC gear shaper.
- C.-F. Chen and C.-B. Tsay, *Computerized Tooth Profile Generation and
  Undercut Analysis of Noncircular Gears Manufactured With Shaper Cutters*,
  Journal of Mechanical Design 120(1) (1998), DOI 10.1115/1.2826682.
- G. Figliolini et al., *Synthesis of the base curves of non-circular gears via
  the return circle*, International Gear Conference (2014). This provides an
  independent base-curve/line-of-action construction for noncircular involute
  gears.
- L. Yu et al., *Design and Application of Non-Circular Gear with Cusp Pitch
  Curve*, Machines 10(11) (2022), 985, DOI 10.3390/machines10110985. This uses
  envelope synthesis with variable-involute and incomplete variable-cycloid
  branches and illustrates why cusp/concave regions need profile-specific
  treatment.
- C. Lin and Y. Wang, *Principle and Design of Noncircular Pin-Cycloid Gear
  Transmission*, Journal of Northeastern University Natural Science 35(8)
  (2014), 1190-1194, DOI 10.12068/j.issn.1005-3026.2014.08.028. This derives a
  noncircular pin member and its cycloidal conjugate from cutter-path and
  meshing equations.
