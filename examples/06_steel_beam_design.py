"""
examples/06_steel_beam_design.py
=================================
Steel beam design using the built-in section catalogue.

Problem
-------
Select the lightest standard I-section (IPE, HEA, or HEB) for a simply
supported steel beam that satisfies:

  1. Bending strength:   M_Ed <= φ · Wy · fy          (EC3 §6.2.5)
  2. Shear strength:     V_Ed <= φ · (Av · fy / √3)    (EC3 §6.2.6)
  3. Deflection limit:   δ_max <= L / 300              (SLS)
  4. Lateral-torsional buckling is ignored (fully restrained beam).

Objective
---------
Minimise self-weight (mass per unit length × span).

Design variables
----------------
section : SteelSection — any IPE, HEA, or HEB section

Loading
-------
  Span          L  = 6.0 m
  UDL (factored) w  = 40  kN/m   (dead + live, ULS)
  UDL (service)  w_s = 25 kN/m   (SLS)
  Steel grade   fy = 355 MPa
  E             = 210 000 MPa

Run
---
    python examples/06_steel_beam_design.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonix import DesignSpace, Minimization, SteelSection

# ---------------------------------------------------------------------------
# Design parameters
# ---------------------------------------------------------------------------
L    = 6.0          # span [m]
w    = 40.0         # factored UDL [kN/m]  — ULS
w_s  = 25.0         # service  UDL [kN/m]  — SLS
fy   = 355.0        # yield strength [MPa]
E    = 210_000.0    # elastic modulus [MPa]
phi  = 1.0          # resistance factor (EC3 uses γ_M0 = 1.0)

# Derived ULS demands
M_Ed = w * L**2 / 8          # [kN·m]
V_Ed = w * L / 2             # [kN]

# Derived SLS demand
M_ser = w_s * L**2 / 8       # [kN·m]

print(f"Design demands:")
print(f"  M_Ed = {M_Ed:.1f} kN·m")
print(f"  V_Ed = {V_Ed:.1f} kN")
print(f"  Span = {L:.1f} m\n")

# ---------------------------------------------------------------------------
# Design space
# ---------------------------------------------------------------------------
space = DesignSpace()
section_var = SteelSection(series=["IPE", "HEA", "HEB"])
space.add("section", section_var)

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def objective(harmony):
    idx = harmony["section"]
    sec = section_var.decode(idx)

    # Convert catalogue values to SI base units
    Wy_m3  = sec.Wy_cm3  * 1e-6   # cm³ → m³
    Iz_m4  = sec.Iy_cm4  * 1e-8   # cm⁴ → m⁴  (strong axis = Iy in catalogue)
    A_m2   = sec.A_cm2   * 1e-4   # cm² → m²
    tw_m   = sec.tw_mm   * 1e-3   # mm  → m
    h_m    = sec.h_mm    * 1e-3   # mm  → m

    fy_kPa = fy * 1e3              # MPa → kPa (= kN/m²)
    E_kPa  = E  * 1e3              # MPa → kPa

    # 1. Bending capacity [kN·m]
    M_Rd = phi * Wy_m3 * fy_kPa

    # 2. Shear capacity — shear area ≈ h × tw  (simplified)
    Av   = h_m * tw_m              # [m²]
    V_Rd = phi * Av * fy_kPa / math.sqrt(3)

    # 3. Deflection at midspan (elastic, simply supported UDL)
    #    δ = 5 w L⁴ / (384 E I)
    w_s_kPa = w_s                  # [kN/m] — already per unit length
    delta    = 5 * w_s_kPa * L**4 / (384 * E_kPa * Iz_m4)   # [m]
    delta_lim = L / 300

    # Penalties (sum of violations)
    penalty = 0.0
    penalty += max(0.0, M_Ed  - M_Rd)           # bending
    penalty += max(0.0, V_Ed  - V_Rd)           # shear
    penalty += max(0.0, delta - delta_lim) * 1e4 # deflection (scaled)

    # Objective: total beam weight [kg]
    weight = sec.mass_kg_m * L

    return weight, penalty


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    optimizer = Minimization(space, objective)

    result = optimizer.optimize(
        memory_size = 20,
        hmcr        = 0.85,
        par         = 0.35,
        bw_max      = 0.05,
        bw_min      = 0.001,
        max_iter    = 3_000,
        verbose     = False,
    )

    print(result)

    # Decode and verify
    idx = result.best_harmony["section"]
    sec = section_var.decode(idx)

    Wy_m3  = sec.Wy_cm3  * 1e-6
    Iz_m4  = sec.Iy_cm4  * 1e-8
    tw_m   = sec.tw_mm   * 1e-3
    h_m    = sec.h_mm    * 1e-3
    fy_kPa = fy * 1e3
    E_kPa  = E  * 1e3

    M_Rd  = Wy_m3 * fy_kPa
    Av    = h_m * tw_m
    V_Rd  = Av * fy_kPa / math.sqrt(3)
    delta = 5 * w_s * L**4 / (384 * E_kPa * Iz_m4)

    print(f"\nSelected section : {sec.name}")
    print(f"Series           : {sec.series}")
    print(f"Mass             : {sec.mass_kg_m:.1f} kg/m  →  "
          f"total {sec.mass_kg_m * L:.1f} kg")
    print(f"\nCapacity checks:")
    print(f"  Bending : M_Ed = {M_Ed:.1f} kN·m  ≤  M_Rd = {M_Rd:.1f} kN·m  "
          f"{'✓' if M_Ed <= M_Rd else '✗'}")
    print(f"  Shear   : V_Ed = {V_Ed:.1f} kN    ≤  V_Rd = {V_Rd:.1f} kN    "
          f"{'✓' if V_Ed <= V_Rd else '✗'}")
    print(f"  Deflect : δ = {delta*1000:.1f} mm  ≤  L/300 = {L/300*1000:.1f} mm  "
          f"{'✓' if delta <= L/300 else '✗'}")
    print(f"\nSection properties:")
    print(f"  h = {sec.h_mm:.0f} mm,  b = {sec.b_mm:.0f} mm")
    print(f"  A = {sec.A_cm2:.1f} cm²,  Iy = {sec.Iy_cm4:.0f} cm⁴,  "
          f"Wy = {sec.Wy_cm3:.0f} cm³")