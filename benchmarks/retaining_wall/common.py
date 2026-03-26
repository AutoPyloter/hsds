"""
Shared Physical Laws and Engineering Constants for Retaining Wall Benchmarks.
Consolidates ACI 318 structural demands, Geotechnical stability, and Cost logic.
"""

from __future__ import annotations

import math
from typing import Tuple

# --- Constants ---
H_STEM = 5.0  # m
GAMMA_SOIL = 18.0  # kN/m3
GAMMA_CONC = 24.0  # kN/m3
PHI_SOIL = math.radians(30.0)
KA = (1.0 - math.sin(PHI_SOIL)) / (1.0 + math.sin(PHI_SOIL))
FY_MPA = 420.0
COVER = 60.0  # mm
WIDTH_B = 1000.0  # mm (1m strip)


def get_beta1(fc: float) -> float:
    """ACI 318-19 §22.2.2.4.3: Factor relating depth of equivalent rectangular compressive stress block."""
    if fc <= 28.0:
        return 0.85
    return max(0.65, 0.85 - 0.05 * (fc - 28.0) / 7.0)


def compute_geotech(x1: float, x2: float, x3: float, x4: float, x5: float) -> Tuple[float, float, float, float]:
    """
    Compute Geotechnical stability parameters.
    Returns: (l_heel, fs_overturning, fs_sliding, eccentricity)
    """
    t_base_m = x5 / 1000.0
    h_tot = H_STEM + t_base_m
    l_heel = x1 - x2 - x3

    if l_heel < 0:
        return l_heel, 0.0, 0.0, 1e9

    pa = 0.5 * KA * GAMMA_SOIL * (h_tot**2)
    mo = pa * (h_tot / 3.0)

    # Weights and Centers
    w_base = (x1 / 1000.0) * t_base_m * GAMMA_CONC
    w_base_x = (x1 / 1000.0) / 2.0

    w_stem_rect = (x4 / 1000.0) * H_STEM * GAMMA_CONC
    w_stem_rect_x = (x2 + x3 - x4 / 2.0) / 1000.0

    w_stem_tri = 0.5 * ((x3 - x4) / 1000.0) * H_STEM * GAMMA_CONC
    w_stem_tri_x = (x2 + (x3 - x4) * (2.0 / 3.0)) / 1000.0

    w_soil = (l_heel / 1000.0) * H_STEM * GAMMA_SOIL
    w_soil_x = (x1 - l_heel / 2.0) / 1000.0

    sum_w = w_base + w_stem_rect + w_stem_tri + w_soil
    mr = (w_base * w_base_x) + (w_stem_rect * w_stem_rect_x) + (w_stem_tri * w_stem_tri_x) + (w_soil * w_soil_x)

    fs_overturning = mr / mo if mo > 0 else 100.0
    fs_sliding = (sum_w * math.tan(PHI_SOIL)) / pa
    e = (x1 / 2000.0) - ((mr - mo) / sum_w)

    return l_heel, fs_overturning, fs_sliding, e


def compute_aci_demands(
    x3: float, x5: float, l_heel: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute ACI 318 demand forces.
    Returns: ((vu_stem, mu_stem, d_stem), (vu_heel, mu_heel, d_base))
    Units: Force in kN, Moment in kNm, Distance in mm.
    """
    t_base_m = x5 / 1000.0

    # Stem at base
    pa_stem = 0.5 * KA * GAMMA_SOIL * (H_STEM**2)
    vu_stem = 1.6 * pa_stem
    mu_stem = 1.6 * pa_stem * (H_STEM / 3.0)
    d_stem = x3 - COVER

    # Heel at back of stem
    w_heel_total = 1.2 * (GAMMA_SOIL * H_STEM + GAMMA_CONC * t_base_m)
    vu_heel = w_heel_total * (l_heel / 1000.0)
    mu_heel = w_heel_total * ((l_heel / 1000.0) ** 2 / 2.0)
    d_base = x5 - COVER

    return (vu_stem, mu_stem, d_stem), (vu_heel, mu_heel, d_base)


def compute_cost(x1: float, x3: float, x4: float, x5: float, fc: float, as_s: float, as_b: float) -> float:
    """
    Compute total material cost ($).
    as_s, as_b: steel areas in mm2.
    """
    t_base_m = x5 / 1000.0
    vol_conc = (x1 / 1000.0) * t_base_m + 0.5 * (x3 + x4) / 1000.0 * H_STEM
    cost_conc = (0.5 + 0.02 * fc) * vol_conc
    cost_steel = 50.0 * (as_s + as_b) / 1000.0
    return cost_conc + cost_steel
