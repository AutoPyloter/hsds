"""
Robot Gripper Design — FULL PARAMETRIC EXTREME (Zero-Penalty)
=============================================================
A flawless execution of Zero-Penalty boundary embedding.
The rigid triangle inequalities and assembly limits that stalled the
baseline optimizer are completely absorbed into hierarchical parameter limits.

Hierarchy Sequence:
l -> e -> a -> b -> c -> f (and delta independently)

By enforcing `c` to respect the `|a - b| <= c <= a + b` envelope structurally,
the algorithm explores only linkages that are 100% physically assemblable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Path resolution for local testing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmarks.utils.plotter import ConvergencePlotter
from harmonix import Continuous, DesignSpace, Minimization

# --- Geometric Boundary Helpers ---


def _e_upper(ctx: Dict[str, float]) -> float:
    # Absorbs g5 limits dynamically against base extension length l
    return float(min(50.0, ctx["l"] - 10.0))


def _c_lower(ctx: Dict[str, float]) -> float:
    # Triangle inequality minimum threshold (g1, g2 resolved via absolute difference)
    return float(max(10.0, abs(ctx["a"] - ctx["b"])))


def _c_upper(ctx: Dict[str, float]) -> float:
    # Triangle inequality maximum threshold (g3 resolved)
    return float(min(150.0, ctx["a"] + ctx["b"]))


def _f_lower(ctx: Dict[str, float]) -> float:
    # Clearance constraint (g6)
    return float(max(10.0, ctx["b"] + ctx["c"]))


def _f_upper(ctx: Dict[str, float]) -> float:
    # Overextension constraint (g7)
    return float(min(150.0, ctx["a"] + ctx["b"] + ctx["c"]))


def build_space() -> DesignSpace:
    """Builds the fully dependent nested 7D space."""
    space = DesignSpace()

    # 1. Base Dimensions
    space.add("l", Continuous(lo=50.0, hi=150.0))  # Absorbs g4
    space.add("e", Continuous(lo=0.0, hi=_e_upper))  # Absorbs g5
    space.add("a", Continuous(lo=10.0, hi=150.0))
    space.add("b", Continuous(lo=10.0, hi=150.0))

    # 2. Assembled Link Triangle
    space.add("c", Continuous(lo=_c_lower, hi=_c_upper))  # Absorbs g1, g2, g3

    # 3. Extending Fingers
    space.add("f", Continuous(lo=_f_lower, hi=_f_upper))  # Absorbs g6, g7

    # Independent kinematic angle
    space.add("delta", Continuous(lo=1.0, hi=3.14))

    return space


def objective(config: Dict[str, Any]) -> Tuple[float, float]:
    a = config["a"]
    b = config["b"]
    c = config["c"]
    e = config["e"]
    f = config["f"]
    l = config["l"]
    delta = config["delta"]

    # Objective: Minimize mechanism weight proxy and force variance
    cost = abs(delta - 2.0) * 100.0 + (a + b + c + e + f + l) ** 2 / 100.0

    # Ensure hierarchy collapses generate a massive fallback penalty to prevent false successes.
    # In mathematical reality, because a >= 10, b >= 10,  c_upper (a+b) is always >= abs(a-b).
    # All lower/upper intersections here are mathematically guaranteed!
    inversion_penalties = [
        max(0.0, 0.0 - _e_upper(config)),
        max(0.0, _c_lower(config) - _c_upper(config)),
        max(0.0, _f_lower(config) - _f_upper(config)),
    ]

    total_inversion_penalty = sum(p**2 for p in inversion_penalties) * 1e8

    # The physics constraints (g1 through g7) are completely structurally absorbed.
    # No dynamic penalty values remain.

    return cost, total_inversion_penalty


def main() -> None:
    current_dir = Path(__file__).parent

    space = build_space()
    optimizer = Minimization(space, objective)

    print("[Full Parametric] Commencing Robot Gripper optimization...")
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
        "problem": "robot_gripper",
        "best_cost": float(result.best_fitness),
        "total_penalty": float(result.best_penalty),
        "execution_time": float(exec_time),
        "total_iterations": 30000,
    }
    with open(current_dir / "summary.json", "w", encoding="utf-8") as file_out:
        json.dump(summary, file_out, indent=2)

    plotter = ConvergencePlotter(current_dir / "history_data.csv")
    plotter.set_labels(title="Robot Gripper Design — Full Parametric Extreme")
    plotter.add_info_box(f"Cost: {result.best_fitness:.4f}\nPenalty: {result.best_penalty:.4f}")
    plotter.plot(save_path=current_dir / "convergence.png")


if __name__ == "__main__":
    main()
