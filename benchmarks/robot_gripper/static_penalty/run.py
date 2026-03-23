"""
Robot Gripper Design — STATIC PENALTY
=====================================
Optimizes the geometry of a 7-parameter robotic linkage.
This baseline uses blind uniform sampling. The fundamental laws of
geometric assembly (triangle inequalities, maximum actuator extensions,
and finger offsets) are all relegated to a reactive penalty function.

This creates massive computational inefficiency as the optimizer wastes
cycles evaluating linkages that cannot physically be assembled.
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


def build_space() -> DesignSpace:
    """Constructs a rigid, unconstrained 7D bounding box."""
    space = DesignSpace()
    # x1: a (Link 1 length)
    space.add("a", Continuous(lo=10.0, hi=150.0))
    # x2: b (Link 2 length)
    space.add("b", Continuous(lo=10.0, hi=150.0))
    # x3: c (Link 3 length)
    space.add("c", Continuous(lo=10.0, hi=150.0))
    # x4: e (Pivot offset)
    space.add("e", Continuous(lo=0.0, hi=50.0))
    # x5: f (Finger length)
    space.add("f", Continuous(lo=10.0, hi=150.0))
    # x6: l (Actuator base length)
    space.add("l", Continuous(lo=10.0, hi=150.0))
    # x7: delta (Joint angle)
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

    # Constraints (g_i <= 0 construct)
    g = [
        # Triangle inequalities for link assembly
        c - (a + b),  # g1: a + b >= c
        a - (b + c),  # g2: b + c >= a
        b - (c + a),  # g3: c + a >= b
        # Actuator limits (Z_max = 50)
        50.0 - l,  # g4: l >= 50
        e - (l - 10.0),  # g5: e <= l - 10
        # Finger geometry limits
        (b + c) - f,  # g6: f >= b + c
        f - (a + b + c),  # g7: f <= a + b + c
    ]

    # Calculate static penalty for violations
    penalty_value = sum(max(0.0, c_val) ** 2 for c_val in g) * 1e8

    return cost, penalty_value


def main() -> None:
    current_dir = Path(__file__).parent

    space = build_space()
    optimizer = Minimization(space, objective)

    print("[Static Penalty] Commencing Robot Gripper optimization...")
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

    # Verification metric output
    print(f"Optimal Cost: {result.best_fitness:.8f}")
    print(f"Residual Penalty: {result.best_penalty:.8f}")
    print(f"Execution Time: {exec_time:.4f} seconds")

    # Serialise tracking metrics
    summary = {
        "method": "static_penalty",
        "problem": "robot_gripper",
        "best_cost": float(result.best_fitness),
        "total_penalty": float(result.best_penalty),
        "execution_time": float(exec_time),
        "total_iterations": 30000,
    }
    with open(current_dir / "summary.json", "w", encoding="utf-8") as file_out:
        json.dump(summary, file_out, indent=2)

    plotter = ConvergencePlotter(current_dir / "history_data.csv")
    plotter.set_labels(title="Robot Gripper Design — Static Penalty")
    plotter.add_info_box(f"Cost: {result.best_fitness:.4f}\nPenalty: {result.best_penalty:.4f}")
    plotter.plot(save_path=current_dir / "convergence.png")


if __name__ == "__main__":
    main()
