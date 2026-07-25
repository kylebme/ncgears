#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ncgear import generate_from_centrode


@dataclass(frozen=True)
class StressCase:
    name: str
    expression: str
    teeth: int
    category: str
    reference_center_distance: float | None = None
    target_cycle_delta: float = 2.0 * math.pi
    profile: str = "involute"
    note: str = ""
    expected_to_pass: bool = True


CASES = [
    StressCase(
        name="stress_elliptical_e035",
        expression="(1 - 0.35**2)/(1 + 0.35*cos(phi))",
        teeth=32,
        category="known_pitch_curve",
        reference_center_distance=2.0,
        note="Focus-mounted ellipse with semi-major axis 1 and eccentricity 0.35.",
    ),
    StressCase(
        name="stress_basel_reference",
        expression=(
            "(1 - (2 - sqrt(2))*cos(phi))"
            "/(2 - (2 - sqrt(2))*cos(phi))"
        ),
        teeth=28,
        category="known_transmission",
        reference_center_distance=1.0,
        note="Drive centrode recovered from the paper's Section 8 motion law.",
    ),
    StressCase(
        name="stress_pascal_limacon",
        expression="1 + 0.18*cos(phi)",
        teeth=32,
        category="known_pitch_curve_family",
        note="A mild Pascal limacon, a standard noncircular pitch-curve family.",
    ),
    StressCase(
        name="stress_three_lobe_complex",
        expression=(
            "(1 - 0.18*cos(3*phi) + 0.035*cos(6*phi))"
            "/(2 - 0.18*cos(3*phi) + 0.035*cos(6*phi))"
        ),
        teeth=42,
        category="adversarial_harmonic",
        reference_center_distance=1.0,
        note="Three main lobes with a moderate sixth-order shoulder harmonic.",
    ),
    StressCase(
        name="stress_five_lobe_moderate",
        expression="(1 + 0.08*cos(5*phi))/(2 + 0.08*cos(5*phi))",
        teeth=60,
        category="adversarial_harmonic",
        reference_center_distance=1.0,
        note="Five lobes beyond the convex-centrode boundary.",
    ),
    StressCase(
        name="stress_five_lobe_5_to_2",
        expression="1 + 0.08*cos(5*phi)",
        teeth=100,
        category="unequal_ratio_multilobe",
        target_cycle_delta=5.0 * math.pi,
        profile="cycloidal",
        note=(
            "Nonconvex five-lobe centrode with a repeat-compatible 100:40 "
            "tooth closure."
        ),
    ),
    StressCase(
        name="stress_mixed_harmonics",
        expression=(
            "(1 + 0.08*cos(2*phi) - 0.06*sin(3*phi)"
            " + 0.045*cos(5*phi) - 0.025*sin(7*phi))"
            "/(2 + 0.08*cos(2*phi) - 0.06*sin(3*phi)"
            " + 0.045*cos(5*phi) - 0.025*sin(7*phi))"
        ),
        teeth=64,
        category="adversarial_harmonic",
        reference_center_distance=1.0,
        note="Asymmetric mixture through the seventh harmonic.",
    ),
    StressCase(
        name="stress_direct_rosette",
        expression="1 + 0.035*cos(3*phi) + 0.015*cos(7*phi)",
        teeth=56,
        category="direct_centrode",
        note="Direct radial input with incommensurate three- and seven-lobe content.",
    ),
    StressCase(
        name="crazy_rounded_square",
        expression="1 + 0.07*cos(4*phi) + 0.01*cos(8*phi)",
        teeth=64,
        category="silhouette_inspired",
        note="Rounded-square pitch shape inspired by silhouette-optimized gears.",
    ),
    StressCase(
        name="crazy_heart",
        expression=(
            "1 + 0.22*sin(phi) + 0.15*cos(2*phi) - 0.025*sin(3*phi)"
        ),
        teeth=64,
        category="silhouette_inspired",
        note="Heart-like asymmetric pitch shape with a pronounced upper cleft.",
    ),
    StressCase(
        name="crazy_teardrop",
        expression=(
            "1 + 0.22*cos(phi) + 0.07*cos(2*phi) - 0.035*cos(3*phi)"
        ),
        teeth=52,
        category="silhouette_inspired",
        note="Teardrop pitch shape with a narrow and a broad end.",
    ),
    StressCase(
        name="crazy_kidney_bean",
        expression=(
            "1 + 0.16*cos(phi) - 0.11*sin(2*phi) + 0.045*cos(3*phi)"
        ),
        teeth=56,
        category="silhouette_inspired",
        note="Bent kidney/bean pitch shape with broken mirror symmetry.",
    ),
    StressCase(
        name="crazy_rounded_triangle",
        expression="1 + 0.09*cos(3*phi) + 0.012*cos(6*phi)",
        teeth=72,
        category="silhouette_inspired",
        note="Rounded triangular pitch shape.",
    ),
    StressCase(
        name="crazy_crescent",
        expression=(
            "1 + 0.25*cos(phi) - 0.07*cos(2*phi) + 0.03*cos(3*phi)"
        ),
        teeth=64,
        category="silhouette_inspired",
        note="Strongly eccentric crescent/egg-like pitch shape.",
    ),
    StressCase(
        name="crazy_organic",
        expression=(
            "1 + 0.10*cos(phi) + 0.08*sin(2*phi) - 0.055*cos(3*phi)"
            " + 0.035*sin(5*phi)"
        ),
        teeth=64,
        category="silhouette_inspired",
        note="Free-form organic pitch shape with harmonics through order five.",
    ),
    StressCase(
        name="stress_limit_three_lobe_shoulder",
        expression=(
            "(1 - 0.30*cos(3*phi) + 0.08*cos(6*phi))"
            "/(2 - 0.30*cos(3*phi) + 0.08*cos(6*phi))"
        ),
        teeth=36,
        category="adversarial_harmonic",
        reference_center_distance=1.0,
        note="Expected quality-limit case: recovered motion error exceeds 0.01 rad.",
        expected_to_pass=False,
    ),
    StressCase(
        name="stress_limit_five_lobe_deep",
        expression="(1 + 0.45*cos(5*phi))/(2 + 0.45*cos(5*phi))",
        teeth=50,
        category="expected_contact_limit",
        reference_center_distance=1.0,
        note="Expected contact-limit case with a 2.64:1 ratio range.",
        expected_to_pass=False,
    ),
    StressCase(
        name="stress_limit_five_lobe_5_to_2",
        expression="1 + 0.18*cos(5*phi)",
        teeth=100,
        category="expected_disconnected_body_limit",
        target_cycle_delta=5.0 * math.pi,
        profile="cycloidal",
        note=(
            "Deep five-lobe 100:40 closure. The global rack sweep is "
            "expected to disconnect the intended star-shaped body."
        ),
        expected_to_pass=False,
    ),
    StressCase(
        name="stress_limit_mixed_harmonics",
        expression=(
            "(1 + 0.18*cos(2*phi) - 0.14*sin(3*phi)"
            " + 0.12*cos(5*phi) - 0.08*sin(7*phi))"
            "/(2 + 0.18*cos(2*phi) - 0.14*sin(3*phi)"
            " + 0.12*cos(5*phi) - 0.08*sin(7*phi))"
        ),
        teeth=56,
        category="expected_contact_limit",
        reference_center_distance=1.0,
        note="Expected contact-limit case with large mixed harmonics.",
        expected_to_pass=False,
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stress-test centrode input with known and adversarial curves."
    )
    parser.add_argument("--out", type=Path, default=ROOT / "out")
    parser.add_argument(
        "--generator",
        type=Path,
        default=ROOT / "build" / "ncgear_generate",
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--samples-per-radian", type=int, default=90)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--case",
        action="append",
        dest="selected_cases",
        help="Run only this case name; may be repeated.",
    )
    args = parser.parse_args()

    selected = CASES
    if args.selected_cases:
        names = set(args.selected_cases)
        selected = [case for case in CASES if case.name in names]
        missing = names - {case.name for case in selected}
        if missing:
            raise SystemExit(f"Unknown stress case(s): {', '.join(sorted(missing))}")

    results: list[dict[str, object]] = []
    for case in selected:
        started = time.monotonic()
        record: dict[str, object] = asdict(case)
        try:
            directory = generate_from_centrode(
                case.expression,
                name=case.name,
                description=case.note,
                teeth=case.teeth,
                samples=args.samples,
                reference_center_distance=case.reference_center_distance,
                target_cycle_delta=case.target_cycle_delta,
                profile=case.profile,
                samples_per_radian=args.samples_per_radian,
                output_directory=args.out,
                generator=args.generator,
                render=not args.no_render,
            )
            metadata = json.loads(
                (directory / "metadata.json").read_text(encoding="utf-8")
            )
            quality_passed = (
                metadata["placed_pair_overlap_area"] <= 1e-6 * case.teeth
                and metadata["maximum_transmission_error"] < 0.01
                and metadata["minimum_root_radius"] > 0.0
            )
            record.update(
                {
                    "status": "passed" if quality_passed else "quality_failed",
                    "elapsed_seconds": time.monotonic() - started,
                    "center_distance": metadata["center_distance"],
                    "average_angular_ratio": metadata[
                        "average_angular_ratio"
                    ],
                    "centrodes_are_convex": metadata[
                        "centrodes_are_convex"
                    ],
                    "placed_pair_overlap_area": metadata[
                        "placed_pair_overlap_area"
                    ],
                    "maximum_transmission_error": metadata[
                        "maximum_transmission_error"
                    ],
                    "minimum_root_radius": metadata["minimum_root_radius"],
                    "cutter_sweep_phase_count": metadata[
                        "cutter_sweep_phase_count"
                    ],
                }
            )
            label = "PASS" if quality_passed else "QUALITY_LIMIT"
            print(
                f"{label} {case.name}: "
                f"error={metadata['maximum_transmission_error']:.3g}, "
                f"overlap={metadata['placed_pair_overlap_area']:.3g}"
            )
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "elapsed_seconds": time.monotonic() - started,
                    "error": str(error),
                }
            )
            print(f"FAIL {case.name}: {error}", file=sys.stderr)
        observed_pass = record["status"] == "passed"
        record["expectation_met"] = observed_pass == case.expected_to_pass
        results.append(record)

    report = {
        "case_count": len(results),
        "passed": sum(record["status"] == "passed" for record in results),
        "quality_failed": sum(
            record["status"] == "quality_failed" for record in results
        ),
        "generation_failed": sum(
            record["status"] == "failed" for record in results
        ),
        "unexpected_outcomes": sum(
            not record["expectation_met"] for record in results
        ),
        "samples": args.samples,
        "samples_per_radian": args.samples_per_radian,
        "cases": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "centrode_stress_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    return 0 if report["unexpected_outcomes"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
