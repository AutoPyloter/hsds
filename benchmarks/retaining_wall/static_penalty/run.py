"""
Retaining Wall Optimization — STATIC PENALTY
============================================
This executes a blind combinatorial search across 7 discrete engineering
variables (concrete geometry, material, and standard reinforcement catalogs).

All structural (ACI 318) and Geotechnical (FS Overturning/Sliding) laws
are implemented purely as reactive penalties in the objective landscape.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# Path resolution for local testing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from harmonix import Continuous, DesignSpace, Minimization

# Constants
H_STEM = 5.0  # m
GAMMA_SOIL = 18.0  # kN/m3
GAMMA_CONC = 24.0  # kN/m3
PHI_SOIL = math.radians(30.0)
KA = (1.0 - math.sin(PHI_SOIL)) / (1.0 + math.sin(PHI_SOIL))
FY_MPA = 420.0
COVER = 60.0  # mm
WIDTH_B = 1000.0  # mm (1m strip)
DIAMS = [12, 14, 16, 20, 25, 28, 32]


def get_beta1(fc: float) -> float:
    if fc <= 28.0:
        return 0.85
    return max(0.65, 0.85 - 0.05 * (fc - 28.0) / 7.0)


def build_space() -> DesignSpace:
    space = DesignSpace()
    # Continuous mapped to discrete steps conceptually, or let the continuous optimizer ride
    # and discretize internally. Here we use integer steps natively.
    space.add("x1", Continuous(lo=2000.0, hi=6000.0))  # Base Width
    space.add("x2", Continuous(lo=500.0, hi=2000.0))  # Toe Width
    space.add("x3", Continuous(lo=300.0, hi=1000.0))  # Bot Stem
    space.add("x4", Continuous(lo=300.0, hi=1000.0))  # Top Stem
    space.add("x5", Continuous(lo=300.0, hi=1000.0))  # Base Thick
    # Concrete 20 to 40 step 5
    space.add("fc", Continuous(lo=20.0, hi=40.0))
    # Rebar variables
    space.add("stem_dia_idx", Continuous(lo=0.0, hi=6.0))
    space.add("stem_count", Continuous(lo=2.0, hi=41.0))
    space.add("base_dia_idx", Continuous(lo=0.0, hi=6.0))
    space.add("base_count", Continuous(lo=2.0, hi=41.0))
    return space


def objective(config: Dict[str, Any]) -> Tuple[float, float]:
    # Discretize continuous sampler variables
    x1 = round(config["x1"] / 50.0) * 50.0
    x2 = round(config["x2"] / 50.0) * 50.0
    x3 = round(config["x3"] / 50.0) * 50.0
    x4 = round(config["x4"] / 50.0) * 50.0
    x5 = round(config["x5"] / 50.0) * 50.0
    fc = round(config["fc"] / 5.0) * 5.0

    idx_stem_dia = int(round(config["stem_dia_idx"]))
    n_s = int(round(config["stem_count"]))
    idx_base_dia = int(round(config["base_dia_idx"]))
    n_b = int(round(config["base_count"]))

    # --- Geometry Checks ---
    l_heel = x1 - x2 - x3
    if l_heel < 0:
        return 1e12, 1e8  # Physically impossible polygon

    # --- Loading & Geotech ---
    t_base_m = x5 / 1000.0
    h_tot = H_STEM + t_base_m

    # Active Earth Thrust
    Pa = 0.5 * KA * GAMMA_SOIL * (h_tot**2)
    Mo = Pa * (h_tot / 3.0)

    # Weights and Centers
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

    fs_overturning = Mr / Mo if Mo > 0 else 100.0
    fs_sliding = (sum_W * math.tan(PHI_SOIL)) / Pa

    # Eccentricity
    e = (x1 / 2000.0) - ((Mr - Mo) / sum_W)

    # --- Structural ACI Demands ---
    # Stem at base
    Pa_stem = 0.5 * KA * GAMMA_SOIL * (H_STEM**2)
    Vu_stem = 1.6 * Pa_stem
    Mu_stem = 1.6 * Pa_stem * (H_STEM / 3.0)
    d_stem = x3 - COVER

    # Heel at back of stem
    w_heel_total = 1.2 * (GAMMA_SOIL * H_STEM + GAMMA_CONC * t_base_m)
    Vu_heel = w_heel_total * (l_heel / 1000.0)
    Mu_heel = w_heel_total * ((l_heel / 1000.0) ** 2 / 2.0)
    d_base = x5 - COVER

    # Rebars
    db_s = DIAMS[idx_stem_dia]
    As_s = float(n_s) * math.pi * (db_s**2) / 4.0

    db_b = DIAMS[idx_base_dia]
    As_b = float(n_b) * math.pi * (db_b**2) / 4.0

    # --- Structural Penalties (g_i <= 0) ---
    beta1 = get_beta1(fc)
    rho_min = max(0.25 * math.sqrt(fc) / FY_MPA, 1.4 / FY_MPA)
    rho_max = 0.85 * beta1 * (fc / FY_MPA) * (3.0 / 8.0)

    def check_aci(M_u, V_u, d_eff, A_s, d_b, n):
        phi_Vc = 0.75 * 0.17 * math.sqrt(fc) * WIDTH_B * (d_eff / 1000.0)
        a = (A_s * FY_MPA) / (0.85 * fc * WIDTH_B)
        phi_Mn = 0.9 * A_s * FY_MPA * max(0.1, d_eff - a / 2.0) * 1e-6
        rho = A_s / (WIDTH_B * d_eff)
        # Spacing
        s_available = (WIDTH_B - 2.0 * COVER - n * d_b) / max(1, n - 1)
        s_min = max(25.0, d_b)

        return [M_u - phi_Mn, V_u - phi_Vc, rho_min - rho, rho - rho_max, s_min - s_available]

    g = [2.5 - fs_overturning, 2.5 - fs_sliding, abs(e) - (x1 / 6000.0)]
    g.extend(check_aci(Mu_stem, Vu_stem, d_stem, As_s, db_s, n_s))
    g.extend(check_aci(Mu_heel, Vu_heel, d_base, As_b, db_b, n_b))

    penalty_value = sum(max(0.0, limit) ** 2 for limit in g) * 1e6

    # --- Cost Evaluation ---
    vol_conc = (x1 / 1000.0) * t_base_m + 0.5 * (x3 + x4) / 1000.0 * H_STEM
    cost_conc = (0.5 + 0.02 * fc) * vol_conc
    cost_steel = 50.0 * (As_s + As_b) / 1000.0
    total_cost = cost_conc + cost_steel

    return total_cost, penalty_value


def main() -> None:
    current_dir = Path(__file__).parent

    space = build_space()
    optimizer = Minimization(space, objective)

    print("[Static Penalty] Commencing Retaining Wall optimization...")
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
        "method": "static_penalty",
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
