"""
examples/05_multi_objective.py
==============================
Multi-objective optimisation on the ZDT1 benchmark problem.

ZDT1 (Zitzler–Deb–Thiele #1)
------------------------------
Minimise:
    f1(x) = x1
    f2(x) = g(x) · [1 − √(x1 / g(x))]

    g(x)  = 1 + 9 · Σ(x2…xn) / (n − 1)

True Pareto front: f2 = 1 − √f1  for  f1 ∈ [0, 1]

This example also shows:
- how to inspect the Pareto archive
- how to measure approximation quality (mean distance to true front)
- how to use a per-iteration callback

Reference
---------
Zitzler, E., Deb, K., & Thiele, L. (2000).
    Comparison of multiobjective evolutionary algorithms: Empirical results.
    Evolutionary Computation, 8(2), 173–195.

Run
---
    python examples/05_multi_objective.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonix import DesignSpace, Continuous, MultiObjective

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
N = 5   # number of decision variables

space = DesignSpace()
space.add("x1", Continuous(0.0, 1.0))
for i in range(2, N + 1):
    space.add(f"x{i}", Continuous(0.0, 1.0))


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def zdt1(h):
    x1   = h["x1"]
    rest = [h[f"x{i}"] for i in range(2, N + 1)]
    g    = 1.0 + 9.0 * sum(rest) / (N - 1)
    f1   = x1
    f2   = g * (1.0 - math.sqrt(x1 / g))
    return (f1, f2), 0.0


# ---------------------------------------------------------------------------
# Optional: per-iteration callback to track progress
# ---------------------------------------------------------------------------
milestones = []

def on_iteration(iteration, partial):
    if iteration % 1000 == 0:
        milestones.append((iteration, len(partial.front)))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    optimizer = MultiObjective(space, zdt1)

    result = optimizer.optimize(
        memory_size  = 30,
        hmcr         = 0.85,
        par          = 0.35,
        max_iter     = 10_000,
        archive_size = 100,
        callback     = on_iteration,
        verbose      = False,
    )

    # --- Results ----------------------------------------------------------
    print(result)

    print("\nArchive growth:")
    for it, size in milestones:
        print(f"  iter {it:>6d}  →  {size:>3d} Pareto solutions")

    # --- Approximation quality -------------------------------------------
    errors = []
    for entry in result.front:
        f1, f2    = entry.objectives
        true_f2   = 1.0 - math.sqrt(max(0.0, f1))
        errors.append(abs(f2 - true_f2))

    mean_err = sum(errors) / len(errors) if errors else float("inf")
    print(f"\nMean distance to true Pareto front: {mean_err:.4f}")
    print("(values < 0.01 indicate a high-quality approximation)")

    # --- Sample from the front -------------------------------------------
    print("\nSample of Pareto-optimal solutions (f1, f2):")
    step = max(1, len(result.front) // 8)
    for entry in result.front[::step]:
        f1, f2 = entry.objectives
        print(f"  f1 = {f1:.4f}   f2 = {f2:.4f}")
