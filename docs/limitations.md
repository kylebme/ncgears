# Geometry limits found by the extended samples

The paper's convex-centrode assumptions are sufficient but are not the only
constraints needed by a rigid noncircular gear pair.

## Driven-cycle symmetry

For a closed pair, the transmission derivatives must repeat after the drive
angle corresponding to one complete driven-gear revolution. A 2:1 average
ratio therefore accepts a `sin(2*phi)` modulation but rejects `sin(3*phi)`.
Without this condition, successive revolutions demand different tooth shapes
from the same driven gear.

The Python frontend checks this symbolically/numerically. The C++ generator
checks it again against the sampled table.

## Sample-grid alignment

The driven-cycle phase shift must land on a sampled phase. The frontend rounds
the requested closed sample count upward to a multiple implied by the reduced
tooth ratio. This removed false seam errors in the 5:3 sample.

## Missing singularities

The paper's undercut procedure assumes every flank has a directional cusp
root. High tooth counts and centrode inflections can remove that root. In
experimental `--allow-nonconvex` mode, absence of a cusp root is treated as an
undercut-free flank, after which all CGAL trimming and polygon checks still
apply.

## Centrode inflections

Merely bypassing the convexity check is not sufficient for general nonconvex
centrodes. The paper's rounded-cutter fillet uses a curvature-dependent normal.
When curvature changes sign, that normal changes branch; tested examples then
lost required flank intersections or produced self-intersecting outlines.

The generator therefore keeps nonconvex operation opt-in and experimental.
Supporting arbitrary inflections correctly requires a global swept-cutter
envelope or equivalent branch-selection method, not just a relaxed sign test.
