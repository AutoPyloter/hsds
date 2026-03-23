"""
Speed Reducer Design — SEMI-DEPENDENT
=====================================
The first level of constraint embedding. The Face Width (`b`) is made
strictly dependent on all other variables, absorbing 6 out of 11 constraints
(`g1`, `g2`, `g3`, `g4`, `g8`, `g9`) into the boundary definition.

The rest (`g5`, `g6`, `g7`, `g10`, `g11`) remain in the penalty function.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.utils.plotter import ConvergencePlotter
from harmonix import Continuous, DesignSpace, Minimization


def _b_lower(ctx: Dict[str, float]) -> float:
    m = ctx["m"]
    z = ctx["z"]
    l1 = ctx["l1"]
    l2 = ctx["l2"]
    d1 = ctx["d1"]
    d2 = ctx["d2"]
    return max(
        2.6,
        27.0 / (m**2 * z),
        397.5 / (m**2 * z**2),
        1.93 * l1**3 / (m * z * d1**4),
        1.93 * l2**3 / (m * z * d2**4),
        5.0 * m,
    )


def _b_upper(ctx: Dict[str, float]) -> float:
    return float(min(3.6, 12.0 * ctx["m"]))


def build_space() -> DesignSpace:
    space = DesignSpace()
    space.add("m", Continuous(lo=0.7, hi=0.8))
    space.add("z", Continuous(lo=17.0, hi=28.0))
    space.add("l1", Continuous(lo=7.3, hi=8.3))
    space.add("l2", Continuous(lo=7.8, hi=8.3))
    space.add("d1", Continuous(lo=2.9, hi=3.9))
    space.add("d2", Continuous(lo=5.0, hi=5.5))
    space.add("b", Continuous(lo=_b_lower, hi=_b_upper))
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

    b_lo = _b_lower(config)
    b_hi = _b_upper(config)
    inversion_penalty = max(0.0, b_lo - b_hi) ** 2 * 1e8

    g = [
        math.sqrt((745.0 * l1 / (m * z)) ** 2 + 16.9e6) / (110.0 * d1**3) - 1.0,
        math.sqrt((745.0 * l2 / (m * z)) ** 2 + 157.5e6) / (85.0 * d2**3) - 1.0,
        m * z / 40.0 - 1.0,
        (1.5 * d1 + 1.9) / l1 - 1.0,
        (1.1 * d2 + 1.9) / l2 - 1.0,
    ]

    penalty_value = inversion_penalty + sum(max(0.0, c) ** 2 for c in g) * 1e8
    return weight, penalty_value


def main() -> None:
    current_dir = Path(__file__).parent

    space = build_space()
    optimizer = Minimization(space, objective)

    print("[Semi-Dependent] Commencing Speed Reducer Design optimization...")
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
        "method": "semi_dependent",
        "problem": "speed_reducer",
        "best_cost": float(result.best_fitness),
        "total_penalty": float(result.best_penalty),
        "execution_time": float(exec_time),
        "total_iterations": 30000,
    }
    with open(current_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plotter = ConvergencePlotter(current_dir / "history_data.csv")
    plotter.set_labels(title="Speed Reducer — Semi-Dependent")
    plotter.add_info_box(f"Cost: {result.best_fitness:.4f}\nPenalty: {result.best_penalty:.4f}")
    plotter.plot(save_path=current_dir / "convergence.png")


if __name__ == "__main__":
    main()
