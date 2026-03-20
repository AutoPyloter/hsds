"""
examples/03_rc_beam_design.py
=============================
Reinforced concrete beam design using built-in engineering spaces.

Problem
-------
Find the minimum-cost rectangular RC beam cross-section that satisfies
ACI 318-19 flexural strength requirements for a given factored moment Mu.

Design variables
----------------
b   : beam width           [200, 600] mm   — Discrete, 50 mm steps
d   : effective depth      [300, 900] mm   — Discrete, 50 mm steps
rebar : ACI 318 ductile bar arrangement   — ACIRebar (context-aware)

Objective
---------
Minimise total material cost:
    cost = unit_cost_concrete × b × (d + cover) + unit_cost_steel × A_s

Constraint
----------
Design moment capacity φMn ≥ Mu  (penalty = max(0, Mu - φMn))

Run
---
    python examples/03_rc_beam_design.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonix import DesignSpace, Discrete, Minimization, ACIRebar
from harmonix.spaces.engineering import _AREAS_50, _DIAMETERS, _COUNTS

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------
Mu        = 350.0    # factored moment demand        [kN·m]
fc        = 30.0     # concrete compressive strength [MPa]
fy        = 420.0    # steel yield strength          [MPa]
cover     = 60.0     # cover to bar centroid         [mm]

# Approximate unit costs (relative, not real market prices)
COST_CONCRETE = 1.0   # per mm³  (×10⁻⁶ → per cm³)
COST_STEEL    = 80.0  # per mm²  (per unit length)

# ---------------------------------------------------------------------------
# Design space
# ---------------------------------------------------------------------------
space = DesignSpace()
space.add("b", Discrete(200.0, 50.0, 600.0))          # width  [mm]
space.add("d", Discrete(300.0, 50.0, 900.0))          # depth  [mm]
space.add("rebar", ACIRebar(
    d_expr  = lambda ctx: ctx["d"] / 1000.0,          # mm → m
    cc_expr = cover,
    fc      = fc,
    fy      = fy,
))

# ---------------------------------------------------------------------------
# Helper: decode rebar code to steel area [mm²]
# ---------------------------------------------------------------------------
N_COUNTS = len(_COUNTS)

def steel_area(code: int) -> float:
    i = code // N_COUNTS
    j = code %  N_COUNTS
    return _AREAS_50[i] / 50.0 * _COUNTS[j]


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def objective(h):
    b     = h["b"]          # mm
    d     = h["d"]          # mm
    code  = h["rebar"]

    if code is None:
        return 1e12, 1e6

    A_s = steel_area(code)

    # ACI 318 β₁
    beta1 = 0.85 if fc <= 28 else max(0.65, 0.85 - 0.05 * (fc - 28) / 7)

    # Neutral axis depth [mm]
    a = A_s * fy / (0.85 * fc * b)

    # Nominal moment capacity [kN·m]
    phi_Mn = 0.9 * A_s * fy * (d - a / 2) * 1e-6   # N·mm → kN·m

    # Constraint: φMn ≥ Mu
    penalty = max(0.0, Mu - phi_Mn)

    # Cost: concrete volume (per unit span) + steel area
    total_height = d + cover
    cost = (
        COST_CONCRETE * b * total_height * 1e-6  +   # concrete
        COST_STEEL    * A_s * 1e-3                    # steel
    )
    return cost, penalty


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = Minimization(space, objective).optimize(
        memory_size = 30,
        hmcr        = 0.85,
        par         = 0.35,
        max_iter    = 5_000,
        verbose     = False,
    )

    print(result)

    # Decode and display the solution
    b    = result.best_harmony["b"]
    d    = result.best_harmony["d"]
    code = result.best_harmony["rebar"]

    rebar_var = space["rebar"]
    if code is not None:
        dia, n = rebar_var.decode(code)
        A_s    = steel_area(code)
        beta1  = 0.85 if fc <= 28 else max(0.65, 0.85 - 0.05 * (fc - 28) / 7)
        a      = A_s * fy / (0.85 * fc * b)
        phi_Mn = 0.9 * A_s * fy * (d - a / 2) * 1e-6

        print(f"\nSection:   {b:.0f} mm × {d:.0f} mm")
        print(f"Rebar:     {rebar_var.describe(code)}")
        print(f"A_s:       {A_s:.0f} mm²")
        print(f"φMn:       {phi_Mn:.1f} kN·m  (demand: {Mu} kN·m)")
        print(f"Penalty:   {result.best_penalty:.4f}")
