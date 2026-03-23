"""
Pressure Vessel Design — STATIC PENALTY Baseline
=================================================
All four constraints (g1–g4) are evaluated inside the penalty function.
The search space uses fixed, independent bounds with no physical embedding.

This serves as the baseline for comparison against the Dependent Space approach.

Reference
---------
Sandgren, E. (1990). Nonlinear integer and discrete programming in
    mechanical design optimization. Journal of Mechanical Design, 112(2).

Variables
---------
    Ts (x1): Shell thickness      [0.0625, 6.1875]  in
    Th (x2): Head thickness       [0.0625, 6.1875]  in
    R  (x3): Inner radius         [10, 200]          in
    L  (x4): Cylindrical length   [10, 240]          in

Run
---
    python benchmarks/pressure_vessel/static_penalty/run.py
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
MIN_VOLUME: float = 1_296_000.0  # Minimum required internal volume (in³)


# ---------------------------------------------------------------------------
# Static Design Space  (no dependency, no physics embedding)
# ---------------------------------------------------------------------------
def build_static_space() -> DesignSpace:
    """
    Flat, independent bounds — the traditional formulation.
    All four constraints are left to the penalty function.
    """
    space = DesignSpace()
    space.add("Ts", Continuous(0.0625, 6.1875))
    space.add("Th", Continuous(0.0625, 6.1875))
    space.add("R", Continuous(10.0, 200.0))
    space.add("L", Continuous(10.0, 240.0))
    return space


# ---------------------------------------------------------------------------
# Objective — ALL 4 constraints as penalty
# ---------------------------------------------------------------------------
def objective(harmony: Dict[str, Any]) -> Tuple[float, float]:
    """
    Evaluate fabrication cost and total constraint violation.

    Constraints
    -----------
    g1: -Ts + 0.0193·R ≤ 0          (shell hoop stress)
    g2: -Th + 0.00954·R ≤ 0         (head hoop stress)
    g3: -π·R²·L - (4/3)π·R³ + 1,296,000 ≤ 0   (minimum volume)
    g4: L - 240 ≤ 0                  (maximum length)

    Cost breakdown
    --------------
    Term 1:  0.6224 · Ts · R · L      — shell material + welding
    Term 2:  1.7781 · Th · R²         — two hemispherical heads
    Term 3:  3.1661 · Ts² · L         — longitudinal shell forming
    Term 4: 19.84   · Ts² · R         — circumferential shell forming
    """
    Ts: float = harmony["Ts"]
    Th: float = harmony["Th"]
    R: float = harmony["R"]
    L: float = harmony["L"]

    # --- Guard against degenerate designs ---
    if Ts <= 0 or Th <= 0 or R <= 0 or L <= 0:
        return float("inf"), float("inf")

    # --- Fabrication cost ---
    cost: float = 0.6224 * Ts * R * L + 1.7781 * Th * R**2 + 3.1661 * Ts**2 * L + 19.84 * Ts**2 * R

    # --- All 4 constraints ---
    g1 = -Ts + 0.0193 * R  # shell hoop
    g2 = -Th + 0.00954 * R  # head hoop
    g3 = -(math.pi * R**2 * L) - (4.0 / 3.0) * math.pi * R**3 + MIN_VOLUME  # volume
    g4 = L - 240.0  # length

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
        "problem": "pressure_vessel",
        "best_cost": round(result.best_fitness, 8),
        "total_penalty": round(result.best_penalty, 8),
        "execution_time": round(t_elapsed, 4),
        "total_iterations": result.iterations,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- convergence.png ---
    plotter = ConvergencePlotter(OUTPUT_DIR / "history_data.csv")
    plotter.set_labels(title="Pressure Vessel — Static Penalty")
    plotter.add_info_box(
        f"Cost: {result.best_fitness:.2f}\n" f"Penalty: {result.best_penalty:.4f}\n" f"Time: {t_elapsed:.2f}s"
    )
    plotter.plot(save_path=OUTPUT_DIR / "convergence.png")

    print(f"[Static Penalty] Optimal Cost: {result.best_fitness:.6f}")
    print(f"[Static Penalty] Penalty:      {result.best_penalty:.6f}")
    print(f"[Static Penalty] Time:         {t_elapsed:.2f}s")
    print(f"[Static Penalty] Files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
