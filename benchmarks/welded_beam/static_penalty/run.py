"""
Welded Beam Design — STATIC PENALTY Baseline
=============================================
All seven constraints (g1–g7) are evaluated inside the penalty function.
The search space uses fixed, independent bounds with no physical embedding.

This serves as the baseline for comparison against the Dependent Space approach.

Variables
---------
    x1 (h): Weld thickness      [0.125, 5.0]  in
    x2 (l): Weld length          [0.1, 10.0]   in
    x3 (t): Beam width           [0.1, 10.0]   in
    x4 (b): Beam thickness       [0.125, 5.0]  in

Run
---
    python benchmarks/welded_beam/static_penalty/run.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Allow importing benchmarks.utils from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.plotter import ConvergencePlotter

from harmonix import Continuous, DesignSpace, Minimization

# ---------------------------------------------------------------------------
# Physical Constants
# ---------------------------------------------------------------------------
P: float = 6000.0  # Load (lb)
L: float = 14.0  # Beam length (in)
E: float = 30e6  # Young's modulus (psi)
G: float = 12e6  # Shear modulus (psi)
TAU_MAX: float = 13600.0  # Allowable shear stress (psi)
SIGMA_MAX: float = 30000.0  # Allowable bending stress (psi)
DELTA_MAX: float = 0.25  # Allowable deflection (in)


# ---------------------------------------------------------------------------
# Static Design Space  (no dependency, no physics embedding)
# ---------------------------------------------------------------------------
def build_static_space() -> DesignSpace:
    """
    Flat, independent bounds — the traditional formulation.
    Every constraint is left to the penalty function.
    """
    space = DesignSpace()
    space.add("x1", Continuous(0.125, 5.0))
    space.add("x2", Continuous(0.1, 10.0))
    space.add("x3", Continuous(0.1, 10.0))
    space.add("x4", Continuous(0.125, 5.0))
    return space


# ---------------------------------------------------------------------------
# Objective — ALL 7 constraints as penalty
# ---------------------------------------------------------------------------
def objective(harmony: Dict[str, Any]) -> Tuple[float, float]:
    """
    Evaluate fabrication cost and total constraint violation.

    Constraints
    -----------
    g1: Combined shear stress ≤ TAU_MAX
    g2: Bending stress ≤ SIGMA_MAX
    g3: x1 ≤ x4  (weld thickness ≤ beam thickness)
    g4: Weight / cost limit
    g5: x1 ≥ 0.125  (minimum weld size)
    g6: Deflection ≤ DELTA_MAX
    g7: Buckling load ≥ P
    """
    x1: float = harmony["x1"]
    x2: float = harmony["x2"]
    x3: float = harmony["x3"]
    x4: float = harmony["x4"]

    # --- Guard against degenerate designs ---
    if x1 <= 0 or x2 <= 0 or x3 <= 0 or x4 <= 0:
        return float("inf"), float("inf")

    # --- Derived quantities ---
    M = P * (L + x2 / 2.0)
    R = math.sqrt(x2**2 / 4.0 + ((x1 + x3) / 2.0) ** 2)
    J = 2 * (math.sqrt(2) * x1 * x2 * (x2**2 / 12.0 + ((x1 + x3) / 2.0) ** 2))
    if J == 0:
        return float("inf"), float("inf")

    tau_prime = P / (math.sqrt(2) * x1 * x2)
    tau_double_prime = M * R / J
    tau = math.sqrt(tau_prime**2 + 2 * tau_prime * tau_double_prime * (x2 / (2 * R)) + tau_double_prime**2)

    sigma = 6 * P * L / (x4 * x3**2)
    delta = 4 * P * L**3 / (E * x3**3 * x4)
    Pc = ((4.013 * E * math.sqrt((x3**2 * x4**6) / 36.0)) / L**2) * (1.0 - (x3 / (2 * L)) * math.sqrt(E / (4 * G)))

    # --- All 7 constraints ---
    g1 = tau - TAU_MAX
    g2 = sigma - SIGMA_MAX
    g3 = x1 - x4
    g4 = 0.10471 * x1**2 + 0.04811 * x3 * x4 * (14.0 + x2) - 5.0
    g5 = 0.125 - x1
    g6 = delta - DELTA_MAX
    g7 = P - Pc

    penalty: float = sum(max(0.0, g) for g in [g1, g2, g3, g4, g5, g6, g7])
    cost: float = 1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14.0 + x2)

    return cost, penalty


# ---------------------------------------------------------------------------
# Run & Record
# ---------------------------------------------------------------------------
HISTORY_EVERY: int = 100
MAX_ITER: int = 30_000
OUTPUT_DIR: Path = Path(__file__).resolve().parent


def main() -> None:
    space = build_static_space()

    t_start = time.perf_counter()
    result = Minimization(space, objective).optimize(
        memory_size=50,
        hmcr=0.90,
        par=0.40,
        max_iter=MAX_ITER,
        use_cache=True,
        log_history=True,
        history_log_path=OUTPUT_DIR / "history_data.csv",
        history_every=HISTORY_EVERY,
        verbose=False,
    )
    t_elapsed = time.perf_counter() - t_start

    # --- summary.json ---
    summary = {
        "method": "static_penalty",
        "problem": "welded_beam",
        "best_cost": round(result.best_fitness, 8),
        "total_penalty": round(result.best_penalty, 8),
        "execution_time": round(t_elapsed, 4),
        "total_iterations": result.iterations,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- convergence.png ---
    plotter = ConvergencePlotter(OUTPUT_DIR / "history_data.csv")
    plotter.set_labels(title="Welded Beam — Static Penalty")
    plotter.add_info_box(
        f"Cost: {result.best_fitness:.4f}\n" f"Penalty: {result.best_penalty:.4f}\n" f"Time: {t_elapsed:.2f}s"
    )
    plotter.plot(save_path=OUTPUT_DIR / "convergence.png")

    print(f"[Static Penalty] Optimal Cost: {result.best_fitness:.6f}")
    print(f"[Static Penalty] Penalty:      {result.best_penalty:.6f}")
    print(f"[Static Penalty] Time:         {t_elapsed:.2f}s")
    print(f"[Static Penalty] Files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
