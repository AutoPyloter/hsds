"""
Retaining Wall Optimization — DEPENDENT SPACE
======================================================
The ultimate geotechnical and concrete mapping.
Instead of evaluating stability constraints (overturning, sliding, eccentricity)
and ACI 318 rebar safety limits (rho_max, Mn, shear, spacing) as post-generation
penalties, this script surgically maps them onto the sampling boundaries.

Variables `x1` (Base Width) is restricted dynamically by a stability scanner.
Valid Rebar combinations are filtered natively, and `u` mapping coordinates
are projected exclusively onto structurally verified ACI configurations.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from harmonix import ACIRebar, Continuous, DesignSpace, Minimization
from harmonix.spaces.engineering import _AREAS_50, _COUNTS

H_STEM = 5.0  # m
GAMMA_SOIL = 18.0  # kN/m3
GAMMA_CONC = 24.0  # kN/m3
PHI_SOIL = math.radians(30.0)
KA = (1.0 - math.sin(PHI_SOIL)) / (1.0 + math.sin(PHI_SOIL))
FY_MPA = 420.0
COVER = 60.0  # mm
WIDTH_B = 1000.0  # mm

N_COUNTS = len(_COUNTS)


def steel_area(code: int) -> float:
    """Total steel area [mm2] from rebar code."""
    if code is None:
        return 0.0
    i = code // N_COUNTS
    j = code % N_COUNTS
    return _AREAS_50[i] / 50.0 * _COUNTS[j]


def get_beta1(fc: float) -> float:
    if fc <= 28.0:
        return 0.85
    return max(0.65, 0.85 - 0.05 * (fc - 28.0) / 7.0)


# --- Dependency Mappers ---


def _x3_min(ctx: Dict[str, float]) -> float:
    # Minimum bottom stem to resist Earth Pressure shear without links
    fc = round(ctx["fc"] / 5.0) * 5.0
    x4 = round(ctx["x4"] / 50.0) * 50.0

    Vu_stem = 1.6 * (0.5 * KA * GAMMA_SOIL * H_STEM**2) * 1000.0  # N
    # phi * Vc = 0.75 * 0.17 * sqrt(f'c) * bw * d
    shear_cap_per_mm = 0.75 * 0.17 * math.sqrt(fc) * WIDTH_B
    d_req = Vu_stem / shear_cap_per_mm

    return float(max(300.0, x4, math.ceil((d_req + COVER) / 50.0) * 50.0))


def check_geotech_stability(x1, x2, x3, x4, x5) -> bool:
    t_base_m = x5 / 1000.0
    h_tot = H_STEM + t_base_m
    l_heel = x1 - x2 - x3
    if l_heel < 0:
        return False

    Pa = 0.5 * KA * GAMMA_SOIL * (h_tot**2)
    Mo = Pa * (h_tot / 3.0)

    w_base = (x1 / 1000.0) * t_base_m * GAMMA_CONC
    w_base_x = (x1 / 1000.0) / 2.0

    w_stem_rect = (x4 / 1000.0) * H_STEM * GAMMA_CONC
    w_stem_rect_x = (x2 + x3 - x4 / 2.0) / 1000.0

    w_stem_tri = 0.5 * ((x3 - x4) / 1000.0) * H_STEM * GAMMA_CONC
    w_stem_tri_x = (x2 + (x3 - x4) * (2.0 / 3.0)) / 1000.0

    w_soil = (l_heel / 1000.0) * H_STEM * GAMMA_SOIL
    w_soil_x = (x1 - l_heel / 2.0) / 1000.0

    sum_W = w_base + w_stem_rect + w_stem_tri + w_soil
    Mr = (w_base * w_base_x) + (w_stem_rect * w_stem_rect_x) + (w_stem_tri * w_stem_tri_x) + (w_soil * w_soil_x)

    fs_ov = Mr / Mo if Mo > 0 else 100.0
    fs_sl = (sum_W * math.tan(PHI_SOIL)) / Pa
    e = abs((x1 / 2000.0) - ((Mr - Mo) / sum_W))

    return fs_ov >= 2.5 and fs_sl >= 2.5 and e <= (x1 / 6000.0)


def _x1_bounds(ctx: Dict[str, float]) -> Tuple[float, float]:
    x2 = round(ctx["x2"] / 50.0) * 50.0
    x3 = round(ctx["x3"] / 50.0) * 50.0
    x4 = round(ctx["x4"] / 50.0) * 50.0
    x5 = round(ctx["x5"] / 50.0) * 50.0

    valid_x1 = []
    for x1_val in range(2000, 6050, 50):
        if check_geotech_stability(x1_val, x2, x3, x4, x5):
            valid_x1.append(float(x1_val))

    if not valid_x1:
        return 6000.0, 2000.0  # Intentional inversion to trigger space drop
    return min(valid_x1), max(valid_x1)


def _x1_min(ctx):
    return _x1_bounds(ctx)[0]


def _x1_max(ctx):
    return _x1_bounds(ctx)[1]


def build_space() -> DesignSpace:
    space = DesignSpace()
    space.add("fc", Continuous(lo=20.0, hi=40.0))
    space.add("x4", Continuous(lo=300.0, hi=1000.0))
    space.add("x3", Continuous(lo=_x3_min, hi=1000.0))  # Embedded Shear
    space.add("x2", Continuous(lo=500.0, hi=2000.0))
    space.add("x5", Continuous(lo=300.0, hi=1000.0))
    space.add("x1", Continuous(lo=_x1_min, hi=_x1_max))  # Embedded Geotech

    # Native Harmonix ACIRebar spaces strictly managing ductility & fit
    space.add(
        "dc_stem",
        ACIRebar(
            d_expr=lambda ctx: (round(ctx["x3"] / 50.0) * 50.0 - COVER) / 1000.0,
            cc_expr=COVER,
            fc=lambda ctx: round(ctx["fc"] / 5.0) * 5.0,
            fy=FY_MPA,
        ),
    )

    space.add(
        "dc_base",
        ACIRebar(
            d_expr=lambda ctx: (round(ctx["x5"] / 50.0) * 50.0 - COVER) / 1000.0,
            cc_expr=COVER,
            fc=lambda ctx: round(ctx["fc"] / 5.0) * 5.0,
            fy=FY_MPA,
        ),
    )
    return space


def objective(config: Dict[str, Any]) -> Tuple[float, float]:
    x1 = round(config["x1"] / 50.0) * 50.0
    x2 = round(config["x2"] / 50.0) * 50.0
    x3 = round(config["x3"] / 50.0) * 50.0
    x4 = round(config["x4"] / 50.0) * 50.0
    x5 = round(config["x5"] / 50.0) * 50.0
    fc = round(config["fc"] / 5.0) * 5.0

    inversion_penalties = [max(0.0, _x3_min(config) - x3), max(0.0, _x1_min(config) - _x1_max(config))]
    space_penalty = sum(p**2 for p in inversion_penalties) * 1e8

    t_base_m = x5 / 1000.0
    l_heel = x1 - x2 - x3

    # Demands
    Pa_stem = 0.5 * KA * GAMMA_SOIL * (H_STEM**2)
    Mu_stem = 1.6 * Pa_stem * (H_STEM / 3.0)
    d_stem = x3 - COVER

    w_heel_total = 1.2 * (GAMMA_SOIL * H_STEM + GAMMA_CONC * t_base_m)
    Vu_heel_N = w_heel_total * (l_heel / 1000.0) * 1000.0
    Mu_heel = w_heel_total * ((l_heel / 1000.0) ** 2 / 2.0)
    d_base = x5 - COVER

    # Embed shear check for heel (dynamically penalized if it fails because x5 wasn't pruned)
    phi_Vc_heel = 0.75 * 0.17 * math.sqrt(fc) * WIDTH_B * d_base
    shear_pen = max(0.0, Vu_heel_N - phi_Vc_heel) ** 2 * 1e4

    code_stem = config["dc_stem"]
    code_base = config["dc_base"]

    if code_stem is None or code_base is None or l_heel < 0:
        return 1e12, 1e8 + space_penalty

    As_s = steel_area(code_stem)
    As_b = steel_area(code_base)

    # Check Flexure (Because ACIRebar handles rho & fit, but we must verify Moment limit)
    a_stem = (As_s * FY_MPA) / (0.85 * fc * WIDTH_B)
    phi_Mn_stem = 0.9 * As_s * FY_MPA * max(0.1, d_stem - a_stem / 2.0) * 1e-6
    mu_pen_stem = max(0.0, Mu_stem - phi_Mn_stem) ** 2 * 1e4

    a_base = (As_b * FY_MPA) / (0.85 * fc * WIDTH_B)
    phi_Mn_base = 0.9 * As_b * FY_MPA * max(0.1, d_base - a_base / 2.0) * 1e-6
    mu_pen_base = max(0.0, Mu_heel - phi_Mn_base) ** 2 * 1e4

    vol_conc = (x1 / 1000.0) * t_base_m + 0.5 * (x3 + x4) / 1000.0 * H_STEM
    cost_conc = (0.5 + 0.02 * fc) * vol_conc
    cost_steel = 50.0 * (As_s + As_b) / 1000.0
    total_cost = cost_conc + cost_steel

    return total_cost, space_penalty + shear_pen + mu_pen_stem + mu_pen_base


def main() -> None:
    current_dir = Path(__file__).parent

    space = build_space()
    optimizer = Minimization(space, objective)

    print("[Dependent Space] Commencing Retaining Wall optimization...")
    start_time = time.perf_counter()
    result = optimizer.optimize(
        memory_size=50,
        hmcr=0.95,
        par=0.35,
        max_iter=15000,
        use_cache=True,
        log_history=True,
        history_log_path=current_dir / "history_data.csv",
        history_every=50,
        verbose=False,
    )
    exec_time = time.perf_counter() - start_time

    print(f"Optimal Cost: {result.best_fitness:.4f}")
    print(f"Residual Penalty: {result.best_penalty:.8f}")
    print(f"Execution Time: {exec_time:.4f} seconds")

    summary = {
        "method": "dependent_space",
        "problem": "retaining_wall",
        "best_cost": float(result.best_fitness),
        "total_penalty": float(result.best_penalty),
        "execution_time": float(exec_time),
        "total_iterations": 15000,
    }
    with open(current_dir / "summary.json", "w", encoding="utf-8") as file_out:
        json.dump(summary, file_out, indent=2)


if __name__ == "__main__":
    main()
