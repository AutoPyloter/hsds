"""
Pressure Vessel Design — FULL PARAMETRIC EXTREME (Zero-Penalty)
================================================================
Every constraint is embedded analytically into the design space.
The objective function is pure fabrication cost — penalty is always 0.0.

Engineering Strategy Analysis
-----------------------------
This is the most advanced formulation in the benchmark suite.
Instead of hard-coding magic numbers (37.7, 67.6, etc.) we compute
every critical threshold dynamically from the physical constants.

**Key parametric helpers:**

``_compute_r_min(volume, l_max)``
    Solves the cubic equation  π·R²·L_max + (4/3)·π·R³ = V_min  for R
    using bisection.  This yields the smallest radius where a feasible
    cylindrical length still exists.

``_l_min_from_volume(R, volume)``
    Inverts the volume equation analytically:
        L ≥ (V_min − (4/3)·π·R³) / (π·R²)
    Clamped to 10.0 (benchmark lower bound).

**Embedded constraints:**

+------------+-------------------------------+----------------------------+
| Constraint | Physical meaning              | Embedding mechanism        |
+============+===============================+============================+
| g1         | Shell hoop stress             | Ts lower bound = 0.0193·R  |
| g2         | Head hoop stress              | Th lower bound = 0.00954·R |
| g3         | Min volume ≥ 1,296,000 in³    | L lower bound = L_min(R)   |
| g4         | Max cylinder length ≤ 240 in  | L upper bound = 240        |
| (implicit) | R must allow feasible L       | R lower bound = R_min      |
+------------+-------------------------------+----------------------------+

**Result:**  The penalty function returns 0.0 for *every* generated
design.  The optimizer searches exclusively within the feasible region,
maximising exploitation efficiency.

Reference
---------
Sandgren, E. (1990).  Nonlinear integer and discrete programming in
    mechanical design optimization.  *J. Mech. Design*, 112(2).

Kannan, B. K. & Kramer, S. N. (1994).  An augmented Lagrange multiplier
    based method for mixed integer discrete continuous optimization.
    *J. Mech. Design*, 116(2).

Known near-optimal solution
----------------------------
    Ts ≈ 0.8125,  Th ≈ 0.4375,  R ≈ 42.0985,  L ≈ 176.6366
    → cost ≈ 6059.714

Run
---
    python benchmarks/pressure_vessel/full_parametric_extreme/run.py
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
MIN_VOLUME: float = 1_296_000.0  # Required internal volume    (in³)
R_UPPER: float = 200.0  # Benchmark upper bound for R (in)
L_LOWER: float = 10.0  # Benchmark lower bound for L (in)
L_UPPER: float = 240.0  # Benchmark upper bound for L (in)
TS_UPPER: float = 6.1875  # Max shell thickness         (in)
TH_UPPER: float = 6.1875  # Max head thickness          (in)


# ---------------------------------------------------------------------------
# Parametric helpers — NO magic numbers
# ---------------------------------------------------------------------------
def _compute_r_min(
    volume: float,
    l_max: float,
    *,
    tol: float = 1e-10,
    max_steps: int = 200,
) -> float:
    """
    Find the smallest inner radius R for which the volume constraint
    g3 can be satisfied with L ∈ [L_lower, l_max].

    Solves:   π·R²·l_max + (4/3)·π·R³ = volume

    via bisection on f(R) = π·R²·(l_max + 4R/3) − volume.

    Parameters
    ----------
    volume : float
        Minimum required volume (in³).
    l_max : float
        Maximum allowable cylindrical length (in).
    tol : float
        Convergence tolerance on R.
    max_steps : int
        Maximum bisection iterations.

    Returns
    -------
    float
        R_min to full machine precision.
    """

    def f(R: float) -> float:
        return math.pi * R**2 * (l_max + 4.0 * R / 3.0) - volume

    # Bracket: f(0) = -volume < 0;  f(R_upper) >> 0 for any sane problem
    lo, hi = 0.0, R_UPPER
    for _ in range(max_steps):
        mid = (lo + hi) / 2.0
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol:
            break
    return hi  # conservative (slightly above true root)


def _l_min_from_volume(R: float, volume: float) -> float:
    """
    Invert the volume equation for the minimum cylindrical length.

        V = π·R²·L + (4/3)·π·R³  ≥  volume
        → L ≥ (volume − (4/3)·π·R³) / (π·R²)

    Returns at least ``L_LOWER`` (10.0 in).

    Parameters
    ----------
    R : float
        Inner radius (in).
    volume : float
        Required volume (in³).

    Returns
    -------
    float
        Lower bound for L.
    """
    head_volume: float = (4.0 / 3.0) * math.pi * R**3
    remaining: float = volume - head_volume
    if remaining <= 0.0:
        return L_LOWER  # heads alone satisfy volume
    return max(L_LOWER, remaining / (math.pi * R**2))


# ---------------------------------------------------------------------------
# Derived constant — computed at module load, never hard-coded
# ---------------------------------------------------------------------------
R_MIN: float = _compute_r_min(MIN_VOLUME, L_UPPER)


# ---------------------------------------------------------------------------
# Full Parametric Extreme Design Space
# ---------------------------------------------------------------------------
def build_full_extreme_space() -> DesignSpace:
    """
    Construct a design space where ALL constraints are embedded
    as dynamic variable bounds.

    Every variable is either independent with a physically
    tightened range, or dependent on previously declared variables
    through lambda closures.

    The optimizer will never generate an infeasible design.
    """
    space = DesignSpace()

    # 1. R (Inner Radius) — lower bound from parametric R_min
    space.add("R", Continuous(R_MIN, R_UPPER))

    # 2. Ts (Shell Thickness) — hoop-stress: Ts ≥ 0.0193·R
    space.add(
        "Ts",
        Continuous(
            lambda ctx: 0.0193 * ctx["R"],
            TS_UPPER,
        ),
    )

    # 3. Th (Head Thickness) — hoop-stress: Th ≥ 0.00954·R
    space.add(
        "Th",
        Continuous(
            lambda ctx: 0.00954 * ctx["R"],
            TH_UPPER,
        ),
    )

    # 4. L (Cylindrical Length) — volume: L ≥ L_min(R)
    space.add(
        "L",
        Continuous(
            lambda ctx: _l_min_from_volume(ctx["R"], MIN_VOLUME),
            L_UPPER,
        ),
    )

    return space


# ---------------------------------------------------------------------------
# Objective — PURE COST, zero penalty
# ---------------------------------------------------------------------------
def objective(harmony: Dict[str, Any]) -> Tuple[float, float]:
    """
    Fabrication cost only.  Penalty is guaranteed to be 0.0 by
    construction of the design space.

    Cost breakdown
    --------------
    0.6224 · Ts · R · L    — shell material & welding
    1.7781 · Th · R²       — two hemispherical heads
    3.1661 · Ts² · L       — longitudinal shell forming
    19.84  · Ts² · R       — circumferential shell forming
    """
    R: float = harmony["R"]
    Ts: float = harmony["Ts"]
    Th: float = harmony["Th"]
    L: float = harmony["L"]

    cost: float = 0.6224 * Ts * R * L + 1.7781 * Th * R**2 + 3.1661 * Ts**2 * L + 19.84 * Ts**2 * R

    return cost, 0.0


# ---------------------------------------------------------------------------
# Run & Record
# ---------------------------------------------------------------------------
HISTORY_EVERY: int = 100
MAX_ITER: int = 30_000
OUTPUT_DIR: Path = Path(__file__).resolve().parent


def main() -> None:
    # Print the dynamically computed R_min for transparency
    print(f"[Parametric] Computed R_min = {R_MIN:.6f} in")

    space = build_full_extreme_space()

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
        "method": "full_parametric_extreme",
        "problem": "pressure_vessel",
        "best_cost": round(result.best_fitness, 8),
        "total_penalty": round(result.best_penalty, 8),
        "execution_time": round(t_elapsed, 4),
        "total_iterations": result.iterations,
        "computed_r_min": round(R_MIN, 6),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- convergence.png ---
    plotter = ConvergencePlotter(OUTPUT_DIR / "history_data.csv")
    plotter.set_labels(title="Pressure Vessel — Full Parametric Extreme")
    plotter.add_info_box(
        f"Cost: {result.best_fitness:.2f}\n"
        f"Penalty: {result.best_penalty:.4f}\n"
        f"R_min: {R_MIN:.2f} in\n"
        f"Time: {t_elapsed:.2f}s"
    )
    plotter.plot(save_path=OUTPUT_DIR / "convergence.png")

    print(f"[Parametric] Optimal Cost: {result.best_fitness:.6f}")
    print(f"[Parametric] Penalty:      {result.best_penalty:.6f}")
    print(f"[Parametric] Time:         {t_elapsed:.2f}s")
    print(f"[Parametric] Files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
