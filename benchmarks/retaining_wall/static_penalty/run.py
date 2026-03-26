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

from benchmarks.retaining_wall.common import (
    COVER,
    FY_MPA,
    WIDTH_B,
    compute_aci_demands,
    compute_cost,
    compute_geotech,
    get_beta1,
)
from harmonix import Continuous, DesignSpace, Minimization

# Constants
DIAMS = [12, 14, 16, 20, 25, 28, 32]


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

    # --- Loading & Geotech ---
    l_heel, fs_overturning, fs_sliding, e = compute_geotech(x1, x2, x3, x4, x5)
    if l_heel < 0:
        return 1e12, 1e8  # Physically impossible polygon

    # --- Structural ACI Demands ---
    (vu_stem, mu_stem, d_stem), (vu_heel, mu_heel, d_base) = compute_aci_demands(x3, x5, l_heel)

    # Rebars
    db_s = DIAMS[idx_stem_dia]
    as_s = float(n_s) * math.pi * (db_s**2) / 4.0

    db_b = DIAMS[idx_base_dia]
    as_b = float(n_b) * math.pi * (db_b**2) / 4.0

    # --- Structural Penalties (g_i <= 0) ---
    beta1 = get_beta1(fc)
    rho_min = max(0.25 * math.sqrt(fc) / FY_MPA, 1.4 / FY_MPA)
    rho_max = 0.85 * beta1 * (fc / FY_MPA) * (3.0 / 8.0)

    def check_aci(m_u, v_u, d_eff, a_s, d_b, n):
        phi_vc = 0.75 * 0.17 * math.sqrt(fc) * WIDTH_B * (d_eff / 1000.0)
        a = (a_s * FY_MPA) / (0.85 * fc * WIDTH_B)
        phi_mn = 0.9 * a_s * FY_MPA * max(0.1, d_eff - a / 2.0) * 1e-6
        rho = a_s / (WIDTH_B * d_eff)
        # Spacing
        s_available = (WIDTH_B - 2.0 * COVER - n * d_b) / max(1, n - 1)
        s_min = max(25.0, d_b)

        return [m_u - phi_mn, v_u - phi_vc, rho_min - rho, rho - rho_max, s_min - s_available]

    g = [2.5 - fs_overturning, 2.5 - fs_sliding, abs(e) - (x1 / 6000.0)]
    g.extend(check_aci(mu_stem, vu_stem, d_stem, as_s, db_s, n_s))
    g.extend(check_aci(mu_heel, vu_heel, d_base, as_b, db_b, n_b))

    penalty_value = sum(max(0.0, limit) ** 2 for limit in g) * 1e6

    # --- Cost Evaluation ---
    total_cost = compute_cost(x1, x3, x4, x5, fc, as_s, as_b)

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
