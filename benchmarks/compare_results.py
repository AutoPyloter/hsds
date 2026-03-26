"""
compare_results.py
==================
Side-by-side comparison of all benchmark methods for every problem.

For each problem a convergence-overlay plot is produced showing all
available methods on a single figure.  A formatted summary table is
printed to the console.

Output
------
    benchmarks/welded_beam_comparison.png
    benchmarks/pressure_vessel_comparison.png

Run
---
    python benchmarks/compare_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Allow importing benchmarks.utils from any working directory
BENCHMARKS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARKS_DIR))

from utils.plotter import plot_comparison  # noqa: E402

STATIC_PENALTY_LABEL = "Static Penalty"
FULL_PARAMETRIC_EXTREME_LABEL = "Full Parametric Extreme"

# ---------------------------------------------------------------------------
# Problem & method registry
# ---------------------------------------------------------------------------
# Each problem maps to an ordered list of (method_key, display_label, dir_path).
PROBLEMS: Dict[str, Dict[str, Any]] = {
    "welded_beam": {
        "title": "Welded Beam Design",
        "methods": [
            ("static_penalty", STATIC_PENALTY_LABEL, BENCHMARKS_DIR / "welded_beam" / "static_penalty"),
            ("dependent_space", "Dependent Space", BENCHMARKS_DIR / "welded_beam" / "dependent_space"),
        ],
    },
    "pressure_vessel": {
        "title": "Pressure Vessel Design",
        "methods": [
            ("static_penalty", STATIC_PENALTY_LABEL, BENCHMARKS_DIR / "pressure_vessel" / "static_penalty"),
            ("dependent_space", "Dependent Space", BENCHMARKS_DIR / "pressure_vessel" / "dependent_space"),
            ("semi_dependent", "Semi-Dependent", BENCHMARKS_DIR / "pressure_vessel" / "semi_dependent"),
            (
                "full_parametric_extreme",
                FULL_PARAMETRIC_EXTREME_LABEL,
                BENCHMARKS_DIR / "pressure_vessel" / "full_parametric_extreme",
            ),
        ],
    },
    "spring_design": {
        "title": "Spring Design",
        "methods": [
            ("static_penalty", STATIC_PENALTY_LABEL, BENCHMARKS_DIR / "spring_design" / "static_penalty"),
            ("semi_dependent", "Semi-Dependent", BENCHMARKS_DIR / "spring_design" / "semi_dependent"),
            (
                "full_parametric_extreme",
                FULL_PARAMETRIC_EXTREME_LABEL,
                BENCHMARKS_DIR / "spring_design" / "full_parametric_extreme",
            ),
        ],
    },
    "speed_reducer": {
        "title": "Speed Reducer Design",
        "methods": [
            ("static_penalty", STATIC_PENALTY_LABEL, BENCHMARKS_DIR / "speed_reducer" / "static_penalty"),
            ("semi_dependent", "Semi-Dependent", BENCHMARKS_DIR / "speed_reducer" / "semi_dependent"),
            (
                "full_parametric_extreme",
                FULL_PARAMETRIC_EXTREME_LABEL,
                BENCHMARKS_DIR / "speed_reducer" / "full_parametric_extreme",
            ),
        ],
    },
    "robot_gripper": {
        "title": "Robot Gripper Design",
        "methods": [
            ("static_penalty", STATIC_PENALTY_LABEL, BENCHMARKS_DIR / "robot_gripper" / "static_penalty"),
            (
                "full_parametric_extreme",
                FULL_PARAMETRIC_EXTREME_LABEL,
                BENCHMARKS_DIR / "robot_gripper" / "full_parametric_extreme",
            ),
        ],
    },
    "retaining_wall": {
        "title": "Retaining Wall Optimization",
        "methods": [
            ("static_penalty", STATIC_PENALTY_LABEL, BENCHMARKS_DIR / "retaining_wall" / "static_penalty"),
            (
                "dependent_space",
                "Dependent Space (Zero-Penalty)",
                BENCHMARKS_DIR / "retaining_wall" / "dependent_space",
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def _print_summary_table() -> None:
    """Print a formatted comparison table from summary.json files."""
    divider = "-" * 88
    header = f"  {'Problem':<22} {'Method':<28} {'Best Cost':>12} {'Penalty':>10} {'Time (s)':>10}"
    print("\n" + divider)
    print("  BENCHMARK COMPARISON SUMMARY")
    print(divider)
    print(header)
    print(divider)

    for name, cfg in PROBLEMS.items():
        for method_key, label, method_dir in cfg["methods"]:
            summary_path = method_dir / "summary.json"
            if not summary_path.exists():
                print(f"  {cfg['title']:<22} {label:<28} {'(not run)':>12}")
                continue
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            print(
                f"  {cfg['title']:<22} {label:<28} "
                f"{data['best_cost']:>12.4f} "
                f"{data['total_penalty']:>10.4f} "
                f"{data['execution_time']:>10.2f}"
            )
        print(divider)


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------
def _generate_comparison_plots() -> None:
    """Generate fitness comparison plots for each problem."""
    for name, cfg in PROBLEMS.items():
        csv_map: Dict[str, Path] = {}
        for method_key, label, method_dir in cfg["methods"]:
            csv_path = method_dir / "history_data.csv"
            if csv_path.exists():
                csv_map[label] = csv_path
            else:
                print(f"[SKIP] {label} — {csv_path} not found")

        if len(csv_map) < 2:
            print(f"[SKIP] {cfg['title']} — need at least 2 methods for comparison")
            continue

        out_path = BENCHMARKS_DIR / f"{name}_comparison.png"
        plot_comparison(
            csv_paths=csv_map,
            title=f"{cfg['title']} — Method Comparison",
            metric="fitness",
            save_path=out_path,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _print_summary_table()
    _generate_comparison_plots()
    print("\nDone. Comparison plots saved in benchmarks/ root.")


if __name__ == "__main__":
    main()
