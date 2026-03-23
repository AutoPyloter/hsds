"""
Tension/Compression Spring Design — FULL PARAMETRIC EXTREME
=============================================================
Maximum physics embedding with dynamically computed bounds.
No magic numbers — every threshold is derived from the physical constants.

Engineering Strategy Analysis
-----------------------------
The spring problem has four constraints.  Three can be analytically
inverted to yield single-variable bounds.  The fourth (g2, surge
frequency) is a rational function of d and D that cannot be cleanly
inverted for a single-variable bound.

**Parametric helpers:**

``_d_max(diam_limit, D_min)``
    Maximum wire diameter that still allows the coil to fit:
    d_max = diam_limit − D_min.

``_d_upper_from_g4(diam_limit, D_min)``
    Equivalent to _d_max — computed from the outer-diameter constraint.

``_n_min_shear(d, D)``
    Inverts g1:  N ≥ 71785·d⁴ / D³.

``_n_min_deflection(d, D)``
    Inverts g3:  N ≥ 140.45·d / D².

**Embedded constraints:**

+------------+-------------------------------+----------------------------+
| Constraint | Physical meaning              | Embedding mechanism        |
+============+===============================+============================+
| g4         | Outer diam. ≤ 1.5            | D upper bound = 1.5 − d   |
| g1         | Shear stress                  | N lower bound ≥ shear min  |
| g3         | Deflection                    | N lower bound ≥ defl. min  |
+------------+-------------------------------+----------------------------+

**Remaining in penalty:**

| g2 | Surge frequency — rational function, not invertible |

Note: g2 is *nearly always* satisfied when g1 and g3 are met (they
constrain the same geometric regime), so the effective penalty
is close to zero in practice.

Reference
---------
Belegundu, A. D. (1982).
Arora, J. S. (2004).

Known near-optimal
-------------------
    d ≈ 0.05169,  D ≈ 0.35674,  N ≈ 11.2885  →  cost ≈ 0.012665

Run
---
    python benchmarks/spring_design/full_parametric_extreme/run.py
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
# Physical Constants  (all thresholds derived from these, never hard-coded)
# ---------------------------------------------------------------------------
D_MIN: float = 0.25  # Benchmark lower bound for D
D_MAX: float = 1.3  # Benchmark upper bound for D
D_LOWER: float = 0.05  # Benchmark lower bound for d
N_MIN: float = 2.0  # Benchmark lower bound for N
N_MAX: float = 15.0  # Benchmark upper bound for N
DIAM_LIMIT: float = 1.5  # Outer diameter limit: d + D ≤ 1.5

# Constraint coefficients
SHEAR_COEFF: float = 71785.0  # g1: N ≥ SHEAR_COEFF · d⁴ / D³
DEFL_COEFF: float = 140.45  # g3: N ≥ DEFL_COEFF · d / D²
SURGE_DENOM_COEFF: float = 12566.0  # g2 denominator coefficient
SURGE_TERM_COEFF: float = 5108.0  # g2 second-term coefficient


# ---------------------------------------------------------------------------
# Parametric helpers — derived from constants, NO magic numbers
# ---------------------------------------------------------------------------
def _d_max_from_geometry(diam_limit: float, D_min: float) -> float:
    """
    Maximum wire diameter that allows at least the minimum coil diameter.

        g4:  d + D ≤ diam_limit
        ∴    d ≤ diam_limit − D_min
    """
    return diam_limit - D_min


def _n_min_shear(d: float, D: float) -> float:
    """
    Minimum active coils from the shear-stress constraint (g1).

        g1:  1 − D³·N / (SHEAR_COEFF · d⁴) ≤ 0
        ∴    N ≥ SHEAR_COEFF · d⁴ / D³
    """
    if D <= 0:
        return N_MAX
    return (SHEAR_COEFF * d**4) / (D**3)


def _n_min_deflection(d: float, D: float) -> float:
    """
    Minimum active coils from the deflection constraint (g3).

        g3:  1 − DEFL_COEFF · d / (D² · N) ≤ 0
        ∴    N ≥ DEFL_COEFF · d / D²
    """
    if D <= 0:
        return N_MAX
    return (DEFL_COEFF * d) / (D**2)


# ---------------------------------------------------------------------------
# Derived constants — computed at module load
# ---------------------------------------------------------------------------
D_UPPER_FOR_WIRE: float = _d_max_from_geometry(DIAM_LIMIT, D_MIN)


# ---------------------------------------------------------------------------
# Full Parametric Extreme Design Space
# ---------------------------------------------------------------------------
def build_full_extreme_space() -> DesignSpace:
    """
    Construct a design space where g1, g3, and g4 are embedded
    as dynamic variable bounds using parametric helper functions.

    Variable declaration order: d → D → N
    (Each variable may depend only on previously declared variables.)
    """
    space = DesignSpace()

    # 1. d (Wire Diameter) — independent, upper bound from geometry
    space.add("d", Continuous(D_LOWER, D_UPPER_FOR_WIRE))

    # 2. D (Mean Coil Diameter) — g4: D ≤ diam_limit − d
    space.add(
        "D",
        Continuous(
            D_MIN,
            lambda ctx: min(D_MAX, DIAM_LIMIT - ctx["d"]),
        ),
    )

    # 3. N (Active Coils) — g1 + g3: N ≥ max(shear_min, defl_min)
    space.add(
        "N",
        Continuous(
            lambda ctx: max(
                N_MIN,
                _n_min_shear(ctx["d"], ctx["D"]),
                _n_min_deflection(ctx["d"], ctx["D"]),
            ),
            N_MAX,
        ),
    )

    return space


# ---------------------------------------------------------------------------
# Objective — only g2 (surge frequency) remains
# ---------------------------------------------------------------------------
def objective(harmony: Dict[str, Any]) -> Tuple[float, float]:
    """
    Weight + surge-frequency penalty.

    g1, g3, g4 are guaranteed by the design space.
    Only g2 (surge frequency) can be violated.  In practice, designs
    that satisfy g1 and g3 almost always satisfy g2 as well, so the
    effective penalty is near zero.
    """
    d: float = harmony["d"]
    D: float = harmony["D"]
    N: float = harmony["N"]

    if d <= 0 or D <= 0 or N <= 0:
        return float("inf"), float("inf")

    # --- Weight ---
    cost: float = (N + 2.0) * D * d**2

    # --- g2 (surge frequency) ---
    denom = SURGE_DENOM_COEFF * (D * d**3 - d**4)
    if abs(denom) < 1e-30:
        return float("inf"), float("inf")

    g2 = (4.0 * D**2 - d * D) / denom + 1.0 / (SURGE_TERM_COEFF * d**2) - 1.0

    penalty: float = max(0.0, g2)

    return cost, penalty


# ---------------------------------------------------------------------------
# Run & Record
# ---------------------------------------------------------------------------
HISTORY_EVERY: int = 100
MAX_ITER: int = 30_000
OUTPUT_DIR: Path = Path(__file__).resolve().parent


def main() -> None:
    print(f"[Parametric] Computed d_max = {D_UPPER_FOR_WIRE:.4f}")

    space = build_full_extreme_space()

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
        "method": "full_parametric_extreme",
        "problem": "spring_design",
        "best_cost": round(result.best_fitness, 8),
        "total_penalty": round(result.best_penalty, 8),
        "execution_time": round(t_elapsed, 4),
        "total_iterations": result.iterations,
        "computed_d_max": round(D_UPPER_FOR_WIRE, 6),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- convergence.png ---
    plotter = ConvergencePlotter(OUTPUT_DIR / "history_data.csv")
    plotter.set_labels(title="Spring Design — Full Parametric Extreme")
    plotter.add_info_box(
        f"Cost: {result.best_fitness:.6f}\n"
        f"Penalty: {result.best_penalty:.6f}\n"
        f"d_max: {D_UPPER_FOR_WIRE:.4f}\n"
        f"Time: {t_elapsed:.2f}s"
    )
    plotter.plot(save_path=OUTPUT_DIR / "convergence.png")

    print(f"[Parametric] Optimal Cost: {result.best_fitness:.8f}")
    print(f"[Parametric] Penalty:      {result.best_penalty:.8f}")
    print(f"[Parametric] Time:         {t_elapsed:.2f}s")
    print(f"[Parametric] Files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
