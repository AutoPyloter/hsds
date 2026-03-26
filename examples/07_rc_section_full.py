"""
examples/07_rc_section_full.py
===============================
Reinforced concrete beam — full material + geometry + rebar optimisation.

All three engineering variable types are used together:

    ConcreteGrade  — EN 206 concrete strength class (C20/25 … C50/60)
    Discrete       — beam width b [200, 600] mm, 50 mm steps
    Discrete       — effective depth d [300, 900] mm, 50 mm steps
    ACIRebar       — ACI 318 ductile bar arrangement (context-aware:
                     feasible codes depend on the sampled d and fc)

Problem
-------
Find the minimum-cost rectangular RC beam cross-section for a given
factored moment demand Mu, where cost = concrete volume + steel weight.

    minimise  cost = Cc · b · (d + cover) · L  +  Cs · A_s · L

    subject to  φ·Mn ≥ Mu          (ACI 318 flexural strength)
                b ≥ 200 mm
                d ≥ 300 mm

The concrete unit cost increases with grade (higher strength = more
expensive admixtures), so the optimiser must balance section size,
concrete grade, and steel arrangement simultaneously.

Run
---
    python examples/07_rc_section_full.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonix import (
    ACIRebar,
    ConcreteGrade,
    DesignSpace,
    Discrete,
    Minimization,
)
from harmonix.spaces.engineering import _AREAS_50, _COUNTS

# ---------------------------------------------------------------------------
# Design parameters
# ---------------------------------------------------------------------------
Mu = 400.0  # factored moment demand [kN·m]
fy = 420.0  # steel yield strength   [MPa]
cover = 60.0  # cover to bar centroid  [mm]
L = 1.0  # unit span for cost calc [m]

# Unit costs (relative, not real market values)
# Concrete cost scales with grade to reflect admixture cost
_CONCRETE_COST = {
    "C20/25": 0.80,
    "C25/30": 0.90,
    "C30/37": 1.00,
    "C35/45": 1.12,
    "C40/50": 1.25,
    "C45/55": 1.40,
    "C50/60": 1.60,
}
COST_STEEL = 85.0  # per mm² per metre of span

N_COUNTS = len(_COUNTS)

# ---------------------------------------------------------------------------
# Design space
# ---------------------------------------------------------------------------
space = DesignSpace()

# 1. Concrete grade — C20/25 to C50/60
concrete_var = ConcreteGrade(min_grade="C20/25", max_grade="C50/60")
space.add("grade_idx", concrete_var)

# 2. Beam width — 200 to 600 mm, 50 mm steps
space.add("b", Discrete(200.0, 50.0, 600.0))

# 3. Effective depth — 300 to 900 mm, 50 mm steps
space.add("d", Discrete(300.0, 50.0, 900.0))

# 4. Rebar — ACI 318, bounds depend on d and grade (fc)
# d is in mm from Discrete, but ACIRebar expects d in metres
rebar_var = ACIRebar(
    d_expr=lambda ctx: ctx["d"] / 1000.0,  # mm → m
    cc_expr=cover,
    fc=lambda ctx: concrete_var.decode(ctx["grade_idx"]).fck_MPa,
    fy=fy,
)
space.add("rebar", rebar_var)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def steel_area(code: int) -> float:
    """Total steel area [mm²] from rebar code."""
    i = code // N_COUNTS
    j = code % N_COUNTS
    return _AREAS_50[i] / 50.0 * _COUNTS[j]


def moment_capacity(b, d, steel_area_mm2, fc, fy) -> float:
    """ACI 318 nominal moment capacity φMn [kN·m]."""
    a = steel_area_mm2 * fy / (0.85 * fc * b)  # depth of stress block [mm]
    phi_mn = 0.9 * steel_area_mm2 * fy * (d - a / 2) * 1e-6  # N·mm → kN·m
    return phi_mn


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def objective(h):
    grade_idx = h["grade_idx"]
    b = h["b"]  # mm
    d = h["d"]  # mm
    code = h["rebar"]

    if code is None:
        return 1e12, 1e6

    grade = concrete_var.decode(grade_idx)
    fc = grade.fck_MPa
    steel_area_mm2 = steel_area(code)

    # Moment capacity
    phi_mn = moment_capacity(b, d, steel_area_mm2, fc, fy)

    # Constraint: φMn ≥ Mu
    penalty = max(0.0, Mu - phi_mn)

    # Cost
    unit_cost_concrete = _CONCRETE_COST.get(grade.name, 1.0)
    total_height = d + cover  # mm
    vol_concrete = b * total_height * 1e-6  # mm² → m²  (per m span)
    cost = unit_cost_concrete * vol_concrete * L + COST_STEEL * steel_area_mm2 * 1e-3 * L

    return cost, penalty


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Moment demand: Mu = {Mu} kN·m\n")

    result = Minimization(space, objective).optimize(
        memory_size=30,
        hmcr=0.85,
        par=0.40,
        bw_max=0.05,
        bw_min=0.001,
        max_iter=8_000,
        verbose=False,
    )

    print(result)

    # --- Decode and display ---
    grade_idx = result.best_harmony["grade_idx"]
    b = result.best_harmony["b"]
    d = result.best_harmony["d"]
    code = result.best_harmony["rebar"]

    grade = concrete_var.decode(grade_idx)
    fc = grade.fck_MPa
    steel_area_mm2 = steel_area(code)
    phi_mn = moment_capacity(b, d, steel_area_mm2, fc, fy)

    print("\nOptimal design:")
    print(f"  Concrete grade : {grade.name}  (fck = {fc} MPa, Ecm = {grade.Ecm_GPa} GPa)")
    print(f"  Section        : b = {b:.0f} mm × d = {d:.0f} mm")
    print(f"  {rebar_var.describe(code)}  (A_s = {steel_area_mm2:.0f} mm²)")
    print("\nStrength check:")
    print(f"  φMn = {phi_mn:.1f} kN·m  ≥  Mu = {Mu} kN·m  {'✓' if phi_mn >= Mu else '✗'}")
    print(f"\nPenalty : {result.best_penalty:.4f}  ({'feasible' if result.best_penalty <= 0 else 'INFEASIBLE'})")
