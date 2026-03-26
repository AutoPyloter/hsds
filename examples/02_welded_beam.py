"""
examples/02_welded_beam.py
==========================
Classic engineering benchmark: welded beam design.

Minimise the fabrication cost of a welded beam subject to shear stress,
normal stress, buckling, and deflection constraints.

Decision variables
------------------
h  : weld thickness   [0.125, 5.0]  in
l  : weld length      [0.100, 10.0] in
t  : beam thickness   [0.100, 10.0] in
b  : beam width       [0.125, 5.0]  in

Known near-optimal solution
---------------------------
h ≈ 0.206,  l ≈ 3.470,  t ≈ 9.037,  b ≈ 0.206  →  cost ≈ 1.724

Reference
---------
Ragsdell, K. M., & Phillips, D. T. (1976).
    Optimal design of a class of welded structures using geometric programming.
    Journal of Engineering for Industry, 98(3), 1021–1025.

Run
---
    python examples/02_welded_beam.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonix import Continuous, DesignSpace, Minimization

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
P = 6_000  # applied load  [lbf]
L = 14.0  # beam length   [in]
E = 30e6  # Young's modulus [psi]
G = 12e6  # shear modulus   [psi]
TAU_MAX = 13_600  # allowable shear stress   [psi]
SIGMA_MAX = 30_000  # allowable normal stress  [psi]
DELTA_MAX = 0.25  # allowable deflection     [in]


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
space = DesignSpace()
space.add("h", Continuous(0.125, 5.0))
space.add("l", Continuous(0.100, 10.0))
space.add("t", Continuous(0.100, 10.0))
space.add("b", Continuous(0.125, 5.0))


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def welded_beam(h):
    h_v, l, t, b = h["h"], h["l"], h["t"], h["b"]

    M = P * (L + l / 2)
    R = math.sqrt(l**2 / 4 + ((h_v + t) / 2) ** 2)
    J = 2 * h_v * l * math.sqrt(2) * (l**2 / 12 + ((h_v + t) / 2) ** 2)
    tau1 = P / (math.sqrt(2) * h_v * l)
    tau2 = M * R / J
    tau = math.sqrt(tau1**2 + tau2**2 + l * tau1 * tau2 / R)

    sigma = 6 * P * L / (b * t**2)
    delta = 4 * P * L**3 / (E * t**3 * b)
    critical_buckling_load = 4.013 * E * math.sqrt(t**2 * b**6 / 36) / L**2 * (1 - t / (2 * L) * math.sqrt(E / (4 * G)))
    cost = 1.10471 * h_v**2 * l + 0.04811 * t * b * (14 + l)

    violations = [
        tau - TAU_MAX,
        sigma - SIGMA_MAX,
        h_v - b,
        0.10471 * h_v**2 + 0.04811 * t * b * (14 + l) - 5.0,
        0.125 - h_v,
        delta - DELTA_MAX,
        P - critical_buckling_load,
    ]
    penalty = sum(max(0.0, g) for g in violations)
    return cost, penalty


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = Minimization(space, welded_beam).optimize(
        memory_size=30,
        hmcr=0.90,
        par=0.40,
        max_iter=15_000,
        verbose=False,
    )

    print(result)
    print("Known near-optimal cost: ~1.724")
    print(f"Gap to known optimum:    {abs(result.best_fitness - 1.724) / 1.724 * 100:.1f}%")
