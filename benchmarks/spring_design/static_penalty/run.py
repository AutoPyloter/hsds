"""
Tension/Compression Spring Design — STATIC PENALTY Baseline
============================================================
All four constraints (g1–g4) are evaluated inside the penalty function.
The search space uses fixed, independent bounds with no physics embedding.

Reference
---------
Belegundu, A. D. (1982). A study of mathematical programming methods
    for structural optimization. Dept. Civil Environ. Eng.,
    Univ. Iowa.

Arora, J. S. (2004). Introduction to Optimum Design (2nd ed.).
    Elsevier Academic Press.

Variables
---------
    d  (x1): Wire diameter         [0.05, 2.0]
    D  (x2): Mean coil diameter    [0.25, 1.3]
    N  (x3): Number of active coils [2.0, 15.0]

Objective
---------
    Minimise weight:  f = (N + 2) · D · d²

Constraints
-----------
    g1: 1 − D³·N / (71785·d⁴)                                    ≤ 0  (shear stress)
    g2: (4D² − d·D) / (12566·(D·d³ − d⁴)) + 1/(5108·d²) − 1     ≤ 0  (surge freq.)
    g3: 1 − 140.45·d / (D²·N)                                     ≤ 0  (deflection)
    g4: (d + D) / 1.5 − 1                                         ≤ 0  (outer diam.)

Known near-optimal
-------------------
    d ≈ 0.05169, D ≈ 0.35674, N ≈ 11.2885 → cost ≈ 0.012665

Run
---
    python benchmarks/spring_design/static_penalty/run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Allow importing benchmarks.utils from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.plotter import ConvergencePlotter

from harmonix import Continuous, DesignSpace, Minimization


# ---------------------------------------------------------------------------
# Static Design Space  (flat, independent bounds)
# ---------------------------------------------------------------------------
def build_static_space() -> DesignSpace:
    """Flat bounds from the original benchmark formulation."""
    space = DesignSpace()
    space.add("d", Continuous(0.05, 2.0))  # wire diameter
    space.add("D", Continuous(0.25, 1.3))  # mean coil diameter
    space.add("N", Continuous(2.0, 15.0))  # number of active coils
    return space


# ---------------------------------------------------------------------------
# Objective — ALL 4 constraints as penalty
# ---------------------------------------------------------------------------
def objective(harmony: Dict[str, Any]) -> Tuple[float, float]:
    """
    Evaluate spring weight and total constraint violation.

    All four constraints are handled in the penalty function.
    """
    d: float = harmony["d"]
    D: float = harmony["D"]
    N: float = harmony["N"]

    # --- Guard ---
    if d <= 0 or D <= 0 or N <= 0:
        return float("inf"), float("inf")

    # --- Weight (objective) ---
    cost: float = (N + 2.0) * D * d**2

    # --- All 4 constraints ---
    g1 = 1.0 - (D**3 * N) / (71785.0 * d**4)  # shear stress
    denom_g2 = 12566.0 * (D * d**3 - d**4)
    if abs(denom_g2) < 1e-30:
        return float("inf"), float("inf")
    g2 = (4.0 * D**2 - d * D) / denom_g2 + 1.0 / (5108.0 * d**2) - 1.0  # surge freq
    g3 = 1.0 - (140.45 * d) / (D**2 * N)  # deflection
    g4 = (d + D) / 1.5 - 1.0  # outer diameter

    penalty: float = sum(max(0.0, g) for g in [g1, g2, g3, g4])

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
        hmcr=0.95,
        par=0.35,
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
        "problem": "spring_design",
        "best_cost": round(result.best_fitness, 8),
        "total_penalty": round(result.best_penalty, 8),
        "execution_time": round(t_elapsed, 4),
        "total_iterations": result.iterations,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- convergence.png ---
    plotter = ConvergencePlotter(OUTPUT_DIR / "history_data.csv")
    plotter.set_labels(title="Spring Design — Static Penalty")
    plotter.add_info_box(
        f"Cost: {result.best_fitness:.6f}\n" f"Penalty: {result.best_penalty:.6f}\n" f"Time: {t_elapsed:.2f}s"
    )
    plotter.plot(save_path=OUTPUT_DIR / "convergence.png")

    print(f"[Static Penalty] Optimal Cost: {result.best_fitness:.8f}")
    print(f"[Static Penalty] Penalty:      {result.best_penalty:.8f}")
    print(f"[Static Penalty] Time:         {t_elapsed:.2f}s")
    print(f"[Static Penalty] Files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
