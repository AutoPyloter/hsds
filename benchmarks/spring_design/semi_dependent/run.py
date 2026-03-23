"""
Tension/Compression Spring Design — SEMI-DEPENDENT SPACE
=========================================================
The outer-diameter constraint (g4), shear-stress constraint (g1), and
deflection constraint (g3) are embedded into the design space.
Only the surge-frequency constraint (g2) remains in the penalty function.

Engineering Strategy
--------------------
Declaration order:  d → D → N

**Embedded constraints:**

| Constraint | Embedding                                      |
|------------|------------------------------------------------|
| g4         | D ≤ 1.5 − d  → upper bound of D               |
| g1         | N ≥ 71785·d⁴ / D³  → lower bound of N         |
| g3         | N ≥ 140.45·d / D²  → lower bound of N         |

**Remaining in penalty:**

| Constraint | Reason                                         |
|------------|------------------------------------------------|
| g2         | Rational function of d and D — not analytically |
|            | invertible for a clean single-variable bound.   |

Reference
---------
Belegundu (1982); Arora (2004).

Variables (declaration order)
-----------------------------
    d  : Wire diameter        — INDEPENDENT  [0.05, 1.25]
    D  : Mean coil diameter   — DEPENDENT on d  (D ≤ 1.5 − d)
    N  : Active coils         — DEPENDENT on d, D  (g1, g3)

Run
---
    python benchmarks/spring_design/semi_dependent/run.py
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
# Physical Constants
# ---------------------------------------------------------------------------
D_MIN: float = 0.25  # Benchmark lower bound for D
D_MAX: float = 1.3  # Benchmark upper bound for D
N_MAX: float = 15.0  # Benchmark upper bound for N
DIAM_LIMIT: float = 1.5  # Outer diameter limit: d + D ≤ 1.5


# ---------------------------------------------------------------------------
# Semi-Dependent Design Space
# ---------------------------------------------------------------------------
def build_semi_dependent_space() -> DesignSpace:
    """
    Constructs a space with g1, g3, g4 embedded as dynamic bounds.

    d_max = DIAM_LIMIT − D_MIN = 1.25 (maximum wire diam. when coil is
    at its minimum size).

    D upper bound is clamped to min(D_MAX, 1.5 − d) to embed g4.

    N lower bound is max(2.0, shear_bound, deflection_bound) to embed
    g1 and g3.
    """
    space = DesignSpace()

    # 1. d (Wire Diameter) — independent
    d_max: float = DIAM_LIMIT - D_MIN  # 1.25
    space.add("d", Continuous(0.05, d_max))

    # 2. D (Mean Coil Diameter) — g4: D ≤ 1.5 − d
    space.add(
        "D",
        Continuous(
            D_MIN,
            lambda ctx: min(D_MAX, DIAM_LIMIT - ctx["d"]),
        ),
    )

    # 3. N (Active Coils) — g1 + g3
    def _n_lower(ctx: Dict[str, float]) -> float:
        d = ctx["d"]
        D = ctx["D"]
        # g1: N ≥ 71785·d⁴ / D³
        n_shear = (71785.0 * d**4) / (D**3) if D > 0 else N_MAX
        # g3: N ≥ 140.45·d / D²
        n_defl = (140.45 * d) / (D**2) if D > 0 else N_MAX
        return max(2.0, n_shear, n_defl)

    space.add(
        "N",
        Continuous(
            _n_lower,
            N_MAX,
        ),
    )

    return space


# ---------------------------------------------------------------------------
# Objective — only g2 (surge frequency) remains as penalty
# ---------------------------------------------------------------------------
def objective(harmony: Dict[str, Any]) -> Tuple[float, float]:
    """
    g1, g3, g4 are enforced by the space.  Only g2 can be violated.
    """
    d: float = harmony["d"]
    D: float = harmony["D"]
    N: float = harmony["N"]

    if d <= 0 or D <= 0 or N <= 0:
        return float("inf"), float("inf")

    # --- Weight ---
    cost: float = (N + 2.0) * D * d**2

    # --- g2 (surge frequency) ---
    denom = 12566.0 * (D * d**3 - d**4)
    if abs(denom) < 1e-30:
        return float("inf"), float("inf")

    g2 = (4.0 * D**2 - d * D) / denom + 1.0 / (5108.0 * d**2) - 1.0

    penalty: float = max(0.0, g2)

    return cost, penalty


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
        "method": "semi_dependent",
        "problem": "spring_design",
        "best_cost": round(result.best_fitness, 8),
        "total_penalty": round(result.best_penalty, 8),
        "execution_time": round(t_elapsed, 4),
        "total_iterations": result.iterations,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- convergence.png ---
    plotter = ConvergencePlotter(OUTPUT_DIR / "history_data.csv")
    plotter.set_labels(title="Spring Design — Semi-Dependent Space")
    plotter.add_info_box(
        f"Cost: {result.best_fitness:.6f}\n" f"Penalty: {result.best_penalty:.6f}\n" f"Time: {t_elapsed:.2f}s"
    )
    plotter.plot(save_path=OUTPUT_DIR / "convergence.png")

    print(f"[Semi-Dependent] Optimal Cost: {result.best_fitness:.8f}")
    print(f"[Semi-Dependent] Penalty:      {result.best_penalty:.8f}")
    print(f"[Semi-Dependent] Time:         {t_elapsed:.2f}s")
    print(f"[Semi-Dependent] Files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
