"""
Pressure Vessel Design — SEMI-DEPENDENT SPACE
==============================================
Constraint-aware approach that respects the original benchmark variable
ranges while partially embedding physics into the design space.

Engineering Strategy
--------------------
This strategy occupies the middle ground between the naive static-penalty
baseline and the full parametric extreme:

1. **R stays at its original range [10, 200]** so that the benchmark
   formulation is not altered.
2. **Ts and Th** are made dependent on R via ASME hoop-stress equations
   (g1, g2 embedded).
3. **L is made dependent on R** via the inverted volume equation:
       L >= max(10, (V_min − (4/3)·π·R³) / (π·R²))
   This guarantees that any (R, L) pair satisfies g3 *whenever R is
   large enough to allow it*.
4. **For R < ~37.7 in**, no value of L ∈ [10, 240] can satisfy the
   volume constraint — the inverted formula yields L_min > 240.
   In this region the Continuous variable will have lo > hi, so a
   heavy infeasibility penalty is applied in the objective instead.

Why this matters
~~~~~~~~~~~~~~~~
By embedding L's lower bound, we eliminate ~90 % of volume-violating
candidates at the *generation* stage.  Only the small sliver of designs
where R < R_feasible still requires a penalty, and these are driven out
quickly by the heavy cost.

Reference
---------
Sandgren, E. (1990).  Nonlinear integer and discrete programming in
    mechanical design optimization.  *J. Mech. Design*, 112(2).

Variables (declaration order)
-----------------------------
    R  : Inner radius         — INDEPENDENT  [10, 200]
    Ts : Shell thickness      — DEPENDENT on R (g1: Ts ≥ 0.0193·R)
    Th : Head thickness       — DEPENDENT on R (g2: Th ≥ 0.00954·R)
    L  : Cylindrical length   — DEPENDENT on R (g3: volume ≥ V_min)
                                 upper bound = 240 (g4)

Run
---
    python benchmarks/pressure_vessel/semi_dependent/run.py
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
HEAVY_PENALTY: float = 1e9  # Penalty for geometrically infeasible R


# ---------------------------------------------------------------------------
# Helper: volume-derived minimum L
# ---------------------------------------------------------------------------
def _l_min_from_volume(R: float) -> float:
    """
    Invert the volume equation to find the minimum cylindrical length
    that satisfies g3 for a given inner radius R.

        V = π·R²·L + (4/3)·π·R³  ≥  V_min
        → L ≥ (V_min − (4/3)·π·R³) / (π·R²)

    When R is large enough the hemispherical heads alone provide
    sufficient volume, making L_min negative.  The result is clamped
    to a minimum of 10.0 (the benchmark lower bound for L).
    """
    head_volume: float = (4.0 / 3.0) * math.pi * R**3
    remaining: float = MIN_VOLUME - head_volume
    if remaining <= 0:
        return 10.0  # heads alone satisfy volume
    return remaining / (math.pi * R**2)


# ---------------------------------------------------------------------------
# Semi-Dependent Design Space
# ---------------------------------------------------------------------------
def build_semi_dependent_space() -> DesignSpace:
    """
    Construct the design space with partial physics embedding.

    Embedded constraints
    --------------------
    g1:  Ts ≥ 0.0193·R          (shell hoop stress)
    g2:  Th ≥ 0.00954·R         (head hoop stress)
    g3:  L  ≥ L_min(R)          (volume — when geometrically possible)
    g4:  L  ≤ 240               (upper bound)

    For R < ~37.7, L_min(R) > 240, so Continuous(lo, 240) gets lo > hi.
    The library clamps sampling to avoid crashes, but the objective
    function applies a heavy penalty to drive the optimizer away from
    this infeasible sliver.
    """
    space = DesignSpace()

    # 1. INDEPENDENT: R (Inner Radius) — original benchmark range
    space.add("R", Continuous(10.0, 200.0))

    # 2. DEPENDENT: Ts (Shell Thickness)
    space.add(
        "Ts",
        Continuous(
            lambda ctx: 0.0193 * ctx["R"],
            6.1875,
        ),
    )

    # 3. DEPENDENT: Th (Head Thickness)
    space.add(
        "Th",
        Continuous(
            lambda ctx: 0.00954 * ctx["R"],
            6.1875,
        ),
    )

    # 4. DEPENDENT: L (Cylindrical Length)
    #    lo = max(10, volume-derived minimum)
    #    hi = 240 (g4)
    space.add(
        "L",
        Continuous(
            lambda ctx: max(10.0, _l_min_from_volume(ctx["R"])),
            240.0,
        ),
    )

    return space


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def objective(harmony: Dict[str, Any]) -> Tuple[float, float]:
    """
    Fabrication cost with a heavy penalty for the infeasible-R region.

    When 10 ≤ R < ~37.7—the region where no valid L exists—the
    objective returns a massive penalty to drive the search away.
    For all other designs the penalty is 0.0 because the space
    guarantees g1–g4 satisfaction.
    """
    R: float = harmony["R"]
    Ts: float = harmony["Ts"]
    Th: float = harmony["Th"]
    L: float = harmony["L"]

    if R <= 0.0 or Ts <= 0.0 or Th <= 0.0 or L <= 0.0:
        return float("inf"), float("inf")

    # --- Check geometric feasibility of R ---------------------------------
    l_min_required: float = _l_min_from_volume(R)
    if l_min_required > 240.0:
        # This R is too small — no valid L can satisfy the volume.
        return HEAVY_PENALTY, HEAVY_PENALTY

    # --- Fabrication cost --------------------------------------------------
    cost: float = 0.6224 * Ts * R * L + 1.7781 * Th * R**2 + 3.1661 * Ts**2 * L + 19.84 * Ts**2 * R

    return cost, 0.0


# ---------------------------------------------------------------------------
# Run & Record
# ---------------------------------------------------------------------------
HISTORY_EVERY: int = 100
MAX_ITER: int = 30_000
OUTPUT_DIR: Path = Path(__file__).resolve().parent


def main() -> None:
    space = build_semi_dependent_space()

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
        "method": "semi_dependent",
        "problem": "pressure_vessel",
        "best_cost": round(result.best_fitness, 8),
        "total_penalty": round(result.best_penalty, 8),
        "execution_time": round(t_elapsed, 4),
        "total_iterations": result.iterations,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- convergence.png ---
    plotter = ConvergencePlotter(OUTPUT_DIR / "history_data.csv")
    plotter.set_labels(title="Pressure Vessel — Semi-Dependent Space")
    plotter.add_info_box(
        f"Cost: {result.best_fitness:.2f}\n" f"Penalty: {result.best_penalty:.4f}\n" f"Time: {t_elapsed:.2f}s"
    )
    plotter.plot(save_path=OUTPUT_DIR / "convergence.png")

    print(f"[Semi-Dependent] Optimal Cost: {result.best_fitness:.6f}")
    print(f"[Semi-Dependent] Penalty:      {result.best_penalty:.6f}")
    print(f"[Semi-Dependent] Time:         {t_elapsed:.2f}s")
    print(f"[Semi-Dependent] Files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
