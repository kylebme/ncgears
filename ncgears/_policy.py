"""Central numerical and product policy for :mod:`ncgears`.

The geometry engine necessarily uses finite sampling and floating-point
tolerances.  Keeping those choices here makes their units, intent, and coupling
visible.  Exact mathematical coefficients (for example, Hermite basis
coefficients and quarter-pitch tooth geometry) remain beside their equations.
"""

from __future__ import annotations

import math

# Public defaults and supported input range.
DEFAULT_TEETH = 16
DEFAULT_MODULE = 1.0
DEFAULT_PRESSURE_ANGLE_DEG = 20.0
DEFAULT_ADDENDUM_FACTOR = 1.0
DEFAULT_DEDENDUM_FACTOR = 1.2
DEFAULT_FILLET_FACTOR = 0.3
DEFAULT_INPUT_SAMPLES = 8192
DEFAULT_PADDING_PITCHES = 7.0
DEFAULT_SAMPLES_PER_RADIAN = 110
DEFAULT_GIF_FRAMES = 72
DEFAULT_GIF_FPS = 24
DEFAULT_GIF_DPI = 100

MIN_TEETH = 6
MAX_PRESSURE_ANGLE_DEG = 45.0
MIN_INPUT_SAMPLES = 1024
MIN_PADDING_PITCHES = 2.5
MIN_SAMPLES_PER_RADIAN = 20
MIN_GIF_FRAMES = 2

# Data-shape and polygon-topology invariants. These are named to distinguish
# format requirements from tunable sampling or tolerance policy.
TABULAR_ARRAY_DIMENSIONS = 2
OUTLINE_COORDINATE_COLUMNS = 2
MIN_POLYGON_VERTEX_COUNT = 3
MIN_CLOSED_RING_COORDINATE_COUNT = MIN_POLYGON_VERTEX_COUNT + 1

# Motion-law validation. Ratios outside this range are numerically singular for
# the pitch-radius equations even though they may still be mathematically
# positive. Tooth counts are dimensionless, so their tolerance is absolute.
MIN_MOTION_RATIO = 1e-7
MAX_MOTION_RATIO = 1e7
TOOTH_COUNT_ABS_TOLERANCE = 1e-8
RACK_MIN_TIP_THICKNESS_FACTOR = 0.03
# Direct SymPy evaluations at the two domain endpoints should agree much more
# closely than the fitted periodic interpolant is required to agree throughout.
EXPRESSION_ENDPOINT_TOLERANCE = 1e-7
PERIODIC_VALUE_REL_TOLERANCE = 3e-5
PERIODIC_FIRST_DERIVATIVE_TOLERANCE = 3e-5
PERIODIC_SECOND_DERIVATIVE_TOLERANCE = 1e-4

# Floating-point comparisons that only protect wrapping at a representable
# endpoint use a small number of ULPs, not a geometry-sized tolerance.
FLOAT_COMPARISON_ULPS = 64.0

# Sampling densities. Counts ending in +1 include both interval endpoints.
MAX_GEOMETRY_WORKERS = 8
MIN_SAMPLE_TABLE_ROWS = 64
INTEGRATION_INTERVAL_COUNT = 32768  # Must remain even for Simpson integration.
MIN_CENTRODE_ANALYSIS_INTERVALS = 4096
CENTRODE_ANALYSIS_INTERVALS_PER_INPUT_SEGMENT = 2
CENTRODE_OUTLINE_SAMPLES_PER_TOOTH = 128

# GEOS polygon construction and cleanup. Every dimensional value is multiplied
# by module before use; there is deliberately no unit-sized tolerance floor.
GEOMETRY_LENGTH_TOLERANCE_FACTOR = 2e-7
OUTLINE_DUPLICATE_TOLERANCE_FRACTION = 0.05
OUTLINE_COLLINEAR_TOLERANCE_FRACTION = 0.02
# Normalize finished rolling-cut boundaries at a finer precision than Boolean
# clearance or analytic tessellation, then verify the same precision exposes
# no surviving near-coincident slivers or hairpins.
OUTLINE_CONNECTIVITY_TOLERANCE_FACTOR = 1e-8
# A clearance operation creates circular joins where an inward parallel curve
# crosses a re-entrant polygon vertex. Keep those joins comfortably below the
# analytic outline chord-error budget without multiplying already-dense flank
# vertices excessively.
CLEARANCE_BUFFER_QUADRANT_SEGMENTS = 32
# Keep the backtracking-edge threshold below the analytic chord budget; smooth
# analytic vertices and genuine longer concave features remain untouched.
OUTLINE_BACKTRACK_TOLERANCE_MULTIPLIER = 100.0
ANALYTIC_CHORD_TOLERANCE_FACTOR = 2.5e-5
ANALYTIC_CHORD_ACCEPTANCE_SLACK = 1.2
ANALYTIC_JOIN_TOLERANCE_FACTOR = 1e-10
ANALYTIC_CLOSURE_TOLERANCE_FACTOR = 1e-12
ANALYTIC_ENVELOPE_RESIDUAL_FACTOR = 1e-8
ANALYTIC_TANGENCY_RESIDUAL_TOLERANCE = 1e-10
ANALYTIC_REGULAR_DERIVATIVE_TOLERANCE = 1e-10
CENTRODE_CONVEXITY_CURVATURE_FACTOR = 1e-9
ANGLE_CLOSURE_TOLERANCE = 1e-8

# Analytic curve intersection and cusp discovery. Search windows are measured
# in circular pitch; sample counts are derived from span instead of being tied
# to one particular window width. The 513-point floor preserves the resolution
# required by short, tightly looping undercut branches; span-derived density
# increases it for longer windows.
INTERSECTION_FLANK_HALF_WINDOW_PITCHES = 1.25
INTERSECTION_OFFSET_HALF_WINDOW_PITCHES = 1.75
INTERSECTION_SAMPLES_PER_PITCH = 128
INTERSECTION_MIN_SAMPLES = 513
INTERSECTION_RESIDUAL_FACTOR = GEOMETRY_LENGTH_TOLERANCE_FACTOR
INTERSECTION_PARAMETER_DEDUP_FACTOR = 1e-6
INTERSECTION_SOLVER_TOLERANCE = 1e-13
INTERSECTION_SOLVER_MAX_EVALUATIONS = 150
FLANK_OFFSET_SOLVER_MAX_EVALUATIONS = 100
INTERSECTION_CONTACT_EXCLUSION_PITCHES = 1e-3
INTERSECTION_CANDIDATE_OFFSET_WEIGHT = 0.25
UNDERCUT_FLANK_SEARCH_PITCHES = 2.0

CUSP_INITIAL_SAMPLES = 128
CUSP_MAX_SAMPLES = 4096
CUSP_EQUATION_TOLERANCE = 1e-12
CUSP_PARAMETER_DEDUP_FACTOR = 1e-8
# A hybrid undercut that cannot be joined to the analytic rack-tip fillet
# starts on the regular, tip-side component of the flank this far from the
# nearest cusp.  The value is an arc-length factor of module.
HYBRID_CUSP_CLEARANCE_FACTOR = 1e-4

# Adaptive tessellation. Working flanks receive more initial samples because
# their chord certificate is part of the result contract.
CURVE_SAMPLES_PER_INPUT_RADIAN_FACTOR = 4.0
MIN_FLANK_CURVE_SAMPLES = 128
MIN_CLOSURE_CURVE_SAMPLES = 32
MIN_ROOT_CURVE_SAMPLES = 512
ROOT_CURVE_SAMPLES_PER_TOOTH = 32
MAX_ANALYTIC_CURVE_SAMPLES = 8192

# Support/backing geometry. The cap prevents a minimum module-sized hub from
# consuming more than half of the minimum pitch radius on very small gears.
MIN_SUPPORT_RADIUS_MODULE_FACTOR = 0.08
ROOT_SUPPORT_RADIUS_PITCH_FACTOR = 0.22
PITCH_MASK_SUPPORT_RADIUS_PITCH_FACTOR = 0.20
MAX_SUPPORT_RADIUS_PITCH_FACTOR = 0.50
OPEN_ANALYTIC_BACKING_RADIUS_PITCH_FACTOR = 0.25
SUPPORT_BUFFER_QUADRANT_SEGMENTS = 128
PITCH_MASK_BUFFER_QUADRANT_SEGMENTS = 96

# Rolling root trimming and final solid verification. Staggered offsets are
# fractions of one grid cell. They reduce phase-grid alignment risk, but are
# still a sampled check and are reported as such in metadata.
ROLLING_MIN_PHASES = 96
ROLLING_PHASES_PER_TOOTH = 4
# Match the per-grid minimum used by the v0.2.1 Boolean generator when a
# sacrificial hybrid connector actually needs a cutter-generated undercut.
HYBRID_ROLLING_PHASES_PER_TOOTH = 24
ROLLING_STAGGER_OFFSETS = (0.0, 0.5, 0.25, 0.75)
VERIFICATION_MIN_CLOSED_PHASES = 64
VERIFICATION_MIN_OPEN_PHASES = 48
VERIFICATION_PHASES_PER_TOOTH = 4
OVERLAP_AREA_TOLERANCE_FACTOR = 1e-6
# A phase can involve a small contact neighborhood, not every tooth on a gear.
# Six unit-area cells cover the observed GEOS platform variation for the same
# tessellated contact while remaining independent of the total tooth count.
OVERLAP_CONTACT_PAIR_ALLOWANCE = 6
CONTACT_AREA_TOLERANCE_FACTOR = 1e-11
CONTACT_RECOVERY_PHASES_CLOSED = 6
CONTACT_RECOVERY_PHASES_OPEN = 4
# The golden-ratio conjugate avoids coincidence with dyadic/tooth phase grids.
CONTACT_RECOVERY_PHASE_OFFSET = 0.5 * (math.sqrt(5.0) - 1.0)
CONTACT_SEARCH_INITIAL_ANGLE = 1e-5
CONTACT_SEARCH_MAX_ANGLE = math.radians(5.0)
CONTACT_SEARCH_ANGLE_TOLERANCE = 1e-9
TRIM_BUFFER_QUADRANT_SEGMENTS = 2
# The v0.2.1 conjugate cutter used a 0.00175-module conservative expansion.
# Closing the hybrid pass's accumulated non-working cuts on that same scale
# fills pose-to-pose scallops without expanding the complete opposing gear at
# every working contact.
ROLLING_CUT_CLOSING_FACTOR = 1.75e-3
# Keep the regularization arc faceting below the analytic chord budget.
ROLLING_CUT_BUFFER_QUADRANT_SEGMENTS = 8
# One-sided material guards protect retained analytic flanks during mutual
# rolling cuts.  Make the guard wider than the Boolean trim clearance so the
# latter can never reach a retained flank through roundoff.
PROTECTED_FLANK_GUARD_CHORD_FACTORS = 4.0
PROTECTED_CONTACT_RESIDUAL_FACTOR = 1e-6
ROLLING_MAX_PASSES = 8
ROLLING_CONVERGENCE_AREA_FACTOR = 1e-9

# Broad topology/fidelity guards. These are policy thresholds, not numerical
# roundoff tolerances, and should be revisited with manufacturing validation.
CENTRODE_FIDELITY_ALLOWANCE_MODULES = 0.05
MINIMUM_PITCH_AREA_FRACTION = 0.75
SLIDING_SINE_FLOOR = 1e-6

# Rendering/export choices; these do not affect generated geometry.
SVG_MARGIN_FRACTION = 0.04
SVG_STROKE_WIDTH_FRACTION = 0.002
RENDER_MARGIN_FRACTION = 0.08
