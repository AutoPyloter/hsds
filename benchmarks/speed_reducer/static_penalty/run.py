"""
Speed Reducer Design — STATIC PENALTY
=====================================
The classic methodology where an unconstrained 7-dimensional bounding box
is sampled blindly. All 11 physical constraints (gear geometry, bending
stress, deflections, shafts) are enforced via a penalty function.

This operates as the academic baseline against which the Dependent Space
methods are benchmarked.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Path resolution for local testing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.utils.plotter import ConvergencePlotter
from harmonix import Continuous, DesignSpace, Minimization


def build_space() -> DesignSpace:
    """Constructs a rigid, unconstrained 7D bounding box."""
    space = DesignSpace()
    # x1: Face width
    space.add("b", Continuous(lo=2.6, hi=3.6))
    # x2: Module of teeth
    space.add("m", Continuous(lo=0.7, hi=0.8))
    # x3: Number of teeth on pinion
    space.add("z", Continuous(lo=17.0, hi=28.0))
    # x4: Length of first shaft between bearings
    space.add("l1", Continuous(lo=7.3, hi=8.3))
    # x5: Length of second shaft between bearings
    space.add("l2", Continuous(lo=7.8, hi=8.3))
    # x6: Diameter of first shaft
    space.add("d1", Continuous(lo=2.9, hi=3.9))
    # x7: Diameter of second shaft
    space.add("d2", Continuous(lo=5.0, hi=5.5))
    return space


def objective(config: Dict[str, Any]) -> Tuple[float, float]:
    b = config["b"]
    m = config["m"]
    z = config["z"]
    l1 = config["l1"]
    l2 = config["l2"]
    d1 = config["d1"]
    d2 = config["d2"]

    weight = (
        0.7854 * b * m**2 * (3.3333 * z**2 + 14.9334 * z - 43.0934)
        - 1.508 * b * (d1**2 + d2**2)
        + 7.4777 * (d1**3 + d2**3)
        + 0.7854 * (l1 * d1**2 + l2 * d2**2)
    )

    g = [
        27.0 / (b * m**2 * z) - 1.0,
        397.5 / (b * m**2 * z**2) - 1.0,
        1.93 * l1**3 / (m * z * d1**4) - 1.0,
        1.93 * l2**3 / (m * z * d2**4) - 1.0,
        math.sqrt((745.0 * l1 / (m * z)) ** 2 + 16.9e6) / (110.0 * d1**3) - 1.0,
        math.sqrt((745.0 * l2 / (m * z)) ** 2 + 157.5e6) / (85.0 * d2**3) - 1.0,
        m * z / 40.0 - 1.0,
        5.0 * m / b - 1.0,
        b / (12.0 * m) - 1.0,
        (1.5 * d1 + 1.9) / l1 - 1.0,
        (1.1 * d2 + 1.9) / l2 - 1.0,
    ]

    penalty_value = sum(max(0.0, c) ** 2 for c in g) * 1e8
    return weight, penalty_value


def main() -> None:
    current_dir = Path(__file__).parent

    space = build_space()
    optimizer = Minimization(space, objective)

    print("[Static Penalty] Commencing Speed Reducer Design optimization...")
    start_time = time.perf_counter()
    result = optimizer.optimize(
        memory_size=50,
        hmcr=0.95,
        par=0.35,
        max_iter=30000,
        use_cache=True,
        log_history=True,
        history_log_path=current_dir / "history_data.csv",
        history_every=100,
        verbose=False,
    )
    exec_time = time.perf_counter() - start_time

    print(f"Optimal Cost: {result.best_fitness:.8f}")
    print(f"Residual Penalty: {result.best_penalty:.8f}")
    print(f"Execution Time: {exec_time:.4f} seconds")

    summary = {
        "method": "static_penalty",
        "problem": "speed_reducer",
        "best_cost": float(result.best_fitness),
        "total_penalty": float(result.best_penalty),
        "execution_time": float(exec_time),
        "total_iterations": 30000,
    }
    with open(current_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plotter = ConvergencePlotter(current_dir / "history_data.csv")
    plotter.set_labels(title="Speed Reducer — Static Penalty")
    plotter.add_info_box(f"Cost: {result.best_fitness:.4f}\nPenalty: {result.best_penalty:.4f}")
    plotter.plot(save_path=current_dir / "convergence.png")


if __name__ == "__main__":
    main()
