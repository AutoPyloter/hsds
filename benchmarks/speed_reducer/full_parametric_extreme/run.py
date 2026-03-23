"""
Speed Reducer Design — FULL PARAMETRIC EXTREME (Zero-Penalty)
=============================================================
The ultimate translation of optimization constraints into search boundaries.
All 11 constraints are mapped into a strict evaluation hierarchy:
m -> l1, l2 -> z -> d1 -> d2 -> b

This ensures the algorithm natively spans only physically realizable gears,
achieving a true 0.0 penalty value dynamically.
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

# --- Geometric Boundary Helpers ---


def _z_upper(ctx: Dict[str, float]) -> float:
    return float(min(28.0, 40.0 / ctx["m"]))


def _d1_lo(ctx: Dict[str, float]) -> float:
    m, z, l1 = ctx["m"], ctx["z"], ctx["l1"]
    return float(max(2.9, math.pow(math.sqrt((745.0 * l1 / (m * z)) ** 2 + 16.9e6) / 110.0, 1.0 / 3.0)))


def _d1_hi(ctx: Dict[str, float]) -> float:
    return float(min(3.9, (ctx["l1"] - 1.9) / 1.5))


def _d2_lo(ctx: Dict[str, float]) -> float:
    m, z, l2 = ctx["m"], ctx["z"], ctx["l2"]
    return float(max(5.0, math.pow(math.sqrt((745.0 * l2 / (m * z)) ** 2 + 157.5e6) / 85.0, 1.0 / 3.0)))


def _d2_hi(ctx: Dict[str, float]) -> float:
    return float(min(5.5, (ctx["l2"] - 1.9) / 1.1))


def _b_lower(ctx: Dict[str, float]) -> float:
    m, z, l1, l2, d1, d2 = ctx["m"], ctx["z"], ctx["l1"], ctx["l2"], ctx["d1"], ctx["d2"]
    return float(
        max(
            2.6,
            27.0 / (m**2 * z),
            397.5 / (m**2 * z**2),
            1.93 * l1**3 / (m * z * d1**4),
            1.93 * l2**3 / (m * z * d2**4),
            5.0 * m,
        )
    )


def _b_upper(ctx: Dict[str, float]) -> float:
    return float(min(3.6, 12.0 * ctx["m"]))


def build_space() -> DesignSpace:
    space = DesignSpace()
    space.add("m", Continuous(lo=0.7, hi=0.8))
    space.add("l1", Continuous(lo=7.3, hi=8.3))
    space.add("l2", Continuous(lo=7.8, hi=8.3))
    space.add("z", Continuous(lo=17.0, hi=_z_upper))
    space.add("d1", Continuous(lo=_d1_lo, hi=_d1_hi))
    space.add("d2", Continuous(lo=_d2_lo, hi=_d2_hi))
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

    inversion_penalties = [
        max(0.0, 17.0 - _z_upper(config)),
        max(0.0, _d1_lo(config) - _d1_hi(config)),
        max(0.0, _d2_lo(config) - _d2_hi(config)),
        max(0.0, _b_lower(config) - _b_upper(config)),
    ]
    total_inversion_penalty = sum(p**2 for p in inversion_penalties) * 1e8

    return weight, total_inversion_penalty


def main() -> None:
    current_dir = Path(__file__).parent

    space = build_space()
    optimizer = Minimization(space, objective)

    print("[Full Parametric] Commencing Speed Reducer Design optimization...")
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
        "method": "full_parametric_extreme",
        "problem": "speed_reducer",
        "best_cost": float(result.best_fitness),
        "total_penalty": float(result.best_penalty),
        "execution_time": float(exec_time),
        "total_iterations": 30000,
    }
    with open(current_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plotter = ConvergencePlotter(current_dir / "history_data.csv")
    plotter.set_labels(title="Speed Reducer — Full Parametric Extreme")
    plotter.add_info_box(f"Cost: {result.best_fitness:.4f}\nPenalty: {result.best_penalty:.4f}")
    plotter.plot(save_path=current_dir / "convergence.png")


if __name__ == "__main__":
    main()
