"""
Pressure Vessel Design — DEPENDENT SPACE (Extreme)
===================================================
ASME hoop-stress equations are reverse-engineered and embedded directly
into the lower bounds of the search space, mathematically guaranteeing
that constraints g1 and g2 are never violated during generation.
The length limit (g4) is enforced by the upper bound of L.

Only the nonlinear volume constraint (g3), which couples R and L,
remains in the penalty function.

Reference
---------
Sandgren, E. (1990). Nonlinear integer and discrete programming in
    mechanical design optimization. Journal of Mechanical Design, 112(2).

Kannan, B. K., & Kramer, S. N. (1994). An augmented Lagrange multiplier
    based method for mixed integer discrete continuous optimization.
    Journal of Mechanical Design, 116(2).

Variables (declaration order matters for dependency resolution)
---------
    R  (x3): Inner radius         — INDEPENDENT  [10, 200]  in
    Ts (x1): Shell thickness      — DEPENDENT on R  (Ts ≥ 0.0193·R)
    Th (x2): Head thickness       — DEPENDENT on R  (Th ≥ 0.00954·R)
    L  (x4): Cylindrical length   — INDEPENDENT  [10, 240]  in

Known near-optimal solution
----------------------------
    Ts ≈ 0.8125,  Th ≈ 0.4375,  R ≈ 42.0985,  L ≈ 176.6366
    → cost ≈ 6059.714

Run
---
    python benchmarks/pressure_vessel/dependent_space/run.py
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
# Extreme Dependent Design Space
# ---------------------------------------------------------------------------
def build_extreme_dependent_space() -> DesignSpace:
    """
    Construct the design space with dynamic boundaries derived from
    ASME hoop-stress equations.

    Embedded constraints
    --------------------
    g1:  −Ts + 0.0193·R  ≤ 0   →   Ts ≥ 0.0193·R   (shell hoop stress)
    g2:  −Th + 0.00954·R ≤ 0   →   Th ≥ 0.00954·R   (head hoop stress)
    g4:  L − 240 ≤ 0            →   upper bound of L set to 240
    """
    space = DesignSpace()

    # -----------------------------------------------------------------
    # 1. INDEPENDENT: R (Inner Radius)
    #    Literature range: [10, 200] in.
    # -----------------------------------------------------------------
    space.add("R", Continuous(10.0, 200.0))

    # -----------------------------------------------------------------
    # 2. DEPENDENT: Ts (Shell Thickness)
    #    ASME hoop-stress rule for a thin-walled cylindrical shell
    #    under internal pressure:
    #        Ts ≥ P·R / (S·E − 0.6·P)
    #    Benchmark normalised form:
    #        g1: Ts ≥ 0.0193 · R
    # -----------------------------------------------------------------
    space.add(
        "Ts",
        Continuous(
            lambda ctx: 0.0193 * ctx["R"],  # physics-derived lower bound
            6.1875,
        ),
    )

    # -----------------------------------------------------------------
    # 3. DEPENDENT: Th (Head Thickness)
    #    ASME hoop-stress rule for a hemispherical head:
    #        Th ≥ P·R / (2·S·E − 0.2·P)
    #    Benchmark normalised form:
    #        g2: Th ≥ 0.00954 · R
    # -----------------------------------------------------------------
    space.add(
        "Th",
        Continuous(
            lambda ctx: 0.00954 * ctx["R"],  # physics-derived lower bound
            6.1875,
        ),
    )

    # -----------------------------------------------------------------
    # 4. INDEPENDENT: L (Cylindrical Length)
    #    g4: L ≤ 240  — enforced by upper bound.
    # -----------------------------------------------------------------
    space.add("L", Continuous(10.0, 240.0))

    return space


# ---------------------------------------------------------------------------
# Objective — only g3 (volume) remains as penalty
# ---------------------------------------------------------------------------
def objective(harmony: Dict[str, Any]) -> Tuple[float, float]:
    """
    The search space guarantees hoop-stress (g1, g2) and length (g4)
    feasibility.  Only the nonlinear volume constraint (g3) remains.

    Cost breakdown
    --------------
    Term 1:  0.6224 · Ts · R · L      — shell material + welding
    Term 2:  1.7781 · Th · R²         — two hemispherical heads
    Term 3:  3.1661 · Ts² · L         — longitudinal shell forming
    Term 4: 19.84   · Ts² · R         — circumferential shell forming

    Volume constraint (g3)
    ----------------------
    π·R²·L + (4/3)·π·R³ ≥ 1,296,000
    → −π·R²·L − (4/3)·π·R³ + 1,296,000 ≤ 0
    """
    R: float = harmony["R"]
    Ts: float = harmony["Ts"]
    Th: float = harmony["Th"]
    L: float = harmony["L"]

    # --- Guard against degenerate designs ---
    if R <= 0.0 or Ts <= 0.0 or Th <= 0.0 or L <= 0.0:
        return float("inf"), float("inf")

    # --- Fabrication cost ---
    cost: float = 0.6224 * Ts * R * L + 1.7781 * Th * R**2 + 3.1661 * Ts**2 * L + 19.84 * Ts**2 * R

    # --- Volume constraint (g3) ---
    volume: float = math.pi * R**2 * L + (4.0 / 3.0) * math.pi * R**3
    g3: float = -volume + MIN_VOLUME

    penalty: float = max(0.0, g3)

    return cost, penalty


# ---------------------------------------------------------------------------
# Run & Record
# ---------------------------------------------------------------------------
HISTORY_EVERY: int = 100
MAX_ITER: int = 30_000
OUTPUT_DIR: Path = Path(__file__).resolve().parent


def main() -> None:
    space = build_extreme_dependent_space()

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
        "method": "dependent_space",
        "problem": "pressure_vessel",
        "best_cost": round(result.best_fitness, 8),
        "total_penalty": round(result.best_penalty, 8),
        "execution_time": round(t_elapsed, 4),
        "total_iterations": result.iterations,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- convergence.png ---
    plotter = ConvergencePlotter(OUTPUT_DIR / "history_data.csv")
    plotter.set_labels(title="Pressure Vessel — Dependent Space")
    plotter.add_info_box(
        f"Cost: {result.best_fitness:.2f}\n" f"Penalty: {result.best_penalty:.4f}\n" f"Time: {t_elapsed:.2f}s"
    )
    plotter.plot(save_path=OUTPUT_DIR / "convergence.png")

    print(f"[Dependent Space] Optimal Cost: {result.best_fitness:.6f}")
    print(f"[Dependent Space] Penalty:      {result.best_penalty:.6f}")
    print(f"[Dependent Space] Time:         {t_elapsed:.2f}s")
    print(f"[Dependent Space] Files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
