"""
Welded Beam Design — DEPENDENT SPACE (Extreme)
===============================================
Physical equations are reverse-engineered and embedded directly into the
lower bounds of the search space, mathematically guaranteeing that
constraints g2, g3, g5, and g6 are never violated during generation.

Only the secondary shear (g1), weight limit (g4), and buckling (g7)
constraints remain in the penalty function.

Variables (declaration order matters for dependency resolution)
---------
    x1 (h): Weld thickness       — INDEPENDENT, lower bound from g5
    x4 (b): Beam thickness       — DEPENDENT on x1 (g3: x4 ≥ x1)
    x2 (l): Weld length          — DEPENDENT on x1 (primary shear: g5-derived)
    x3 (t): Beam width           — DEPENDENT on x4 (bending g2 + deflection g6)

Run
---
    python benchmarks/welded_beam/dependent_space/run.py
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
# Extreme Dependent Design Space
# ---------------------------------------------------------------------------
def build_extreme_dependent_space() -> DesignSpace:
    """
    Constructs the design space with dynamic boundaries derived from
    physical laws.

    Embedded constraints
    --------------------
    g5:  x1 ≥ 0.125                          → lower bound of x1
    g3:  x4 ≥ x1                             → lower bound of x4
    g*:  x2 ≥ P / (√2 · x1 · TAU_MAX)       → primary shear lower bound
    g2:  x3 ≥ √(6PL / (x4 · SIGMA_MAX))     → bending stress lower bound
    g6:  x3 ≥ ∛(4PL³ / (E · x4 · DELTA_MAX))→ deflection lower bound
    """
    space = DesignSpace()

    # 1. INDEPENDENT: x1 (Weld Thickness)
    #    g5: 0.125 - x1 ≤ 0  →  x1 ≥ 0.125
    space.add("x1", Continuous(0.125, 2.0))

    # 2. DEPENDENT: x4 (Beam Thickness)
    #    g3: x1 - x4 ≤ 0  →  x4 ≥ x1
    space.add("x4", Continuous(lambda h: h["x1"], 2.0))

    # 3. DEPENDENT: x2 (Weld Length)
    #    tau_prime = P / (√2 · x1 · x2) ≤ TAU_MAX
    #    →  x2 ≥ P / (√2 · x1 · TAU_MAX)
    space.add(
        "x2",
        Continuous(
            lambda h: P / (math.sqrt(2) * h["x1"] * TAU_MAX),
            10.0,
        ),
    )

    # 4. DEPENDENT: x3 (Beam Width)
    #    a) Bending (g2): x3 ≥ √(6PL / (x4 · SIGMA_MAX))
    #    b) Deflection (g6): x3 ≥ ∛(4PL³ / (E · x4 · DELTA_MAX))
    space.add(
        "x3",
        Continuous(
            lambda h: max(
                0.1,
                math.sqrt((6 * P * L) / (h["x4"] * SIGMA_MAX)),
                ((4 * P * L**3) / (E * h["x4"] * DELTA_MAX)) ** (1.0 / 3.0),
            ),
            10.0,
        ),
    )

    return space


# ---------------------------------------------------------------------------
# Objective — only g1, g4, g7 remain as penalty
# ---------------------------------------------------------------------------
def objective(harmony: Dict[str, Any]) -> Tuple[float, float]:
    """
    Since the search space guarantees feasibility for bending (g2),
    deflection (g6), geometric limits (g3, g5), and primary shear,
    this function only checks combined shear (g1), weight limit (g4),
    and buckling (g7).
    """
    x1: float = harmony["x1"]
    x2: float = harmony["x2"]
    x3: float = harmony["x3"]
    x4: float = harmony["x4"]

    M = P * (L + x2 / 2.0)
    R = math.sqrt(x2**2 / 4.0 + ((x1 + x3) / 2.0) ** 2)
    J = 2 * (math.sqrt(2) * x1 * x2 * (x2**2 / 12.0 + ((x1 + x3) / 2.0) ** 2))

    if x1 == 0 or x2 == 0 or J == 0 or x3 == 0 or x4 == 0:
        return float("inf"), float("inf")

    tau_dash = P / (math.sqrt(2) * x1 * x2)
    tau_dash_dash = M * R / J
    tau = math.sqrt(tau_dash**2 + 2 * tau_dash * tau_dash_dash * (x2 / (2 * R)) + tau_dash_dash**2)
    Pc = ((4.013 * E * math.sqrt((x3**2 * x4**6) / 36.0)) / L**2) * (1.0 - (x3 / (2 * L)) * math.sqrt(E / (4 * G)))

    g1 = tau - TAU_MAX
    g4 = 0.10471 * x1**2 + 0.04811 * x3 * x4 * (14.0 + x2) - 5.0
    g7 = P - Pc

    penalty: float = max(0.0, g1) + max(0.0, g4) + max(0.0, g7)
    cost: float = 1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14.0 + x2)

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
        "problem": "welded_beam",
        "best_cost": round(result.best_fitness, 8),
        "total_penalty": round(result.best_penalty, 8),
        "execution_time": round(t_elapsed, 4),
        "total_iterations": result.iterations,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- convergence.png ---
    plotter = ConvergencePlotter(OUTPUT_DIR / "history_data.csv")
    plotter.set_labels(title="Welded Beam — Dependent Space")
    plotter.add_info_box(
        f"Cost: {result.best_fitness:.4f}\n" f"Penalty: {result.best_penalty:.4f}\n" f"Time: {t_elapsed:.2f}s"
    )
    plotter.plot(save_path=OUTPUT_DIR / "convergence.png")

    print(f"[Dependent Space] Optimal Cost: {result.best_fitness:.6f}")
    print(f"[Dependent Space] Penalty:      {result.best_penalty:.6f}")
    print(f"[Dependent Space] Time:         {t_elapsed:.2f}s")
    print(f"[Dependent Space] Files written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
