"""
harmonix.spaces.engineering
====================================
Engineering search spaces — variable types whose feasibility rules
are governed by structural-engineering standards and material catalogues.

+----------------------+----------------------+----------------------------------+
| Class                | Registry name        | Domain                           |
+======================+======================+==================================+
| ``ACIRebar``         | ``aci_rebar``        | ACI 318 ductile bar arrangements |
+----------------------+----------------------+----------------------------------+
| ``ACIDoubleRebar``   | ``aci_rebar_double`` | ACI 318 double-row arrangements  |
+----------------------+----------------------+----------------------------------+
| ``SteelSection``     | ``steel_section``    | IPE / HEA / HEB / W catalogue    |
+----------------------+----------------------+----------------------------------+
| ``ConcreteGrade``    | ``concrete_grade``   | EN 206 strength classes          |
+----------------------+----------------------+----------------------------------+
| ``SoilSPT``          | ``soil_spt``         | SPT-N based soil profile classes |
+----------------------+----------------------+----------------------------------+
| ``SeismicZoneTBDY``  | ``seismic_tbdy``     | TBDY 2018 spectral parameters    |
+----------------------+----------------------+----------------------------------+

Each variable encodes a **bar arrangement** as a single integer *code*
so the Harmony Search memory can store and compare candidates as plain
numbers.  The encoding and decoding are handled transparently.

Encoding
--------
For a grid of ``n_diameters × n_counts`` valid combinations:

    code = diameter_index × n_count_steps + count_index   (0-based)

where ``count_index = count - counts[0]``.

The ``decode(code)`` method returns the human-readable ``(diameter_mm, count)``
pair.

ACI 318 feasibility rules applied
----------------------------------
* Minimum reinforcement ratio:
    ρ_min = max(3√fc' / fy,  200 / fy)     (ACI 318-19 §9.6.1.2)
* Maximum reinforcement ratio (tension-controlled section, φ = 0.90):
    ρ_max = 0.75 · β₁ · (fc'/fy) · [ε_cu / (ε_cu + ε_t,min)]
* Neutral axis depth limit:
    c < c_max = (ε_cu / (ε_cu + ε_t,min)) · d_eff
* Minimum clear spacing:
    s_clear ≥ max(d_b, 25 mm)   (ACI 318-19 §25.2.1)

All stresses in MPa, areas in mm², lengths in m (for d) or mm (for cover).

Usage
-----
.. code-block:: python

    from harmonix import DesignSpace
    from harmonix.spaces.engineering import ACIRebar

    space = DesignSpace()
    space.add("d",     Continuous(0.40, 0.80))   # effective depth [m]
    space.add("rebar", ACIRebar(
        d_expr  = lambda ctx: ctx["d"],
        cc_expr = 40.0,                          # cover [mm]
        fc = 30.0, fy = 420.0,
    ))

    result = optimizer.optimize(...)
    code = result.best_harmony["rebar"]
    dia, n = space["rebar"].decode(code)
    print(f"{n} bars of Ø{dia} mm")
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Dict, List, Optional, Tuple

from ..variables import Variable
from ..registry  import register_variable

Context = Dict[str, Any]


# ---------------------------------------------------------------------------
# Shared bar catalogue  (ASTM / ISO metric)
# ---------------------------------------------------------------------------

# Total area of 50 bars of each diameter  [mm²]
_AREAS_50: List[float] = [
    1_583.5, 3_562.8,  6_333.8,  9_896.6,
   14_251.1, 19_397.0, 25_335.4, 32_236.3,
   40_863.4, 50_369.3, 72_617.5, 129_059.6,
]
# Nominal diameters  [mm]
_DIAMETERS: List[float] = [
    6.30,  9.50, 12.70, 15.90,
   19.00, 22.20, 25.40, 28.70,
   32.30, 35.80, 43.00, 57.30,
]
_COUNTS: List[int] = list(range(4, 42))   # 4 … 41 bars


# ---------------------------------------------------------------------------
# ACI 318 limit computations  (shared between single- and double-row)
# ---------------------------------------------------------------------------

def _aci_limits(
    fc: float, fy: float
) -> Tuple[float, float, float, float, float]:
    """
    Return (β₁, φ, ε_cu, ρ_min, ρ_max) using MPa units throughout.

    Parameters
    ----------
    fc : float  — concrete compressive strength [MPa]
    fy : float  — steel yield strength [MPa]

    ACI 318-19 references
    ---------------------
    β₁          §22.2.2.4.3
    ρ_min        §9.6.1.2  →  max(0.25√fc/fy,  1.4/fy)
    ρ_max        tension-controlled limit (ε_t ≥ 0.004)
                 c/d_t ≤ ε_cu/(ε_cu + 0.004)  →  3/7
                 ρ_max = 0.85·β₁·(fc/fy)·(ε_cu/(ε_cu+0.004))
    """
    if fc <= 28:
        beta1 = 0.85
    elif fc < 56:
        beta1 = 0.85 - 0.05 * (fc - 28) / 7
    else:
        beta1 = 0.65

    phi   = 0.85
    eps_c = 0.003   # ACI crushing strain

    rho_min = max(
        0.25 * math.sqrt(fc) / fy,
        1.4 / fy,
    )
    rho_max = 0.85 * beta1 * (fc / fy) * (eps_c / (eps_c + 0.004))

    return beta1, phi, eps_c, rho_min, rho_max


def _bar_is_valid_single(
    dia: float, count: int, d_eff: float, cc: float,
    beta1: float, phi: float, eps_c: float,
    rho_min: float, rho_max: float,
    fc: float, fy: float, area50: float,
) -> bool:
    """True when the (diameter, count) combination satisfies all ACI limits."""
    area  = area50 / 50.0 * count                      # total steel area [mm²]
    d_mm  = d_eff * 1000.0                              # effective depth [mm]
    d_bar = cc + dia / 2.0                              # dist. from face to bar CL [mm]
    d_net = d_mm - d_bar                                # net effective depth [mm]
    b     = 1000.0                                      # per-metre strip width [mm]

    if d_net <= 0:
        return False

    rho     = area / (b * d_net)                        # reinforcement ratio
    c       = (area * fy) / (0.85 * beta1 * fc * b)    # neutral axis depth [mm]
    c_max   = d_net * eps_c / (eps_c + 0.004)          # tension-controlled limit [mm]
    spacing = (b - count * dia) / (count - 1)           # clear spacing [mm]
    min_sp  = max(dia, 25.0)

    return (
        rho_min <= rho <= rho_max
        and c < c_max
        and spacing >= min_sp
    )


# ---------------------------------------------------------------------------
# ACIRebar — single-row arrangement
# ---------------------------------------------------------------------------

@register_variable("aci_rebar")
class ACIRebar(Variable):
    """
    ACI 318 ductile single-row reinforcing bar arrangement.

    The search space consists of all (diameter, count) pairs that satisfy
    ACI 318-19 minimum/maximum reinforcement ratio, neutral-axis depth, and
    minimum clear-spacing requirements for a given section geometry.

    Each feasible combination is encoded as a non-negative integer *code*;
    the Harmony Search engine treats it as an opaque discrete value.

    Parameters
    ----------
    d_expr : float | callable
        Effective depth of the section [m].  Pass a ``lambda ctx: ...``
        for a depth that depends on previously-assigned variables.
    cc_expr : float | callable
        Concrete cover to bar centreline [mm].
    fc : float
        Concrete compressive strength [MPa].  Default 30 MPa.
    fy : float
        Steel yield strength [MPa].  Default 420 MPa.

    Examples
    --------
    >>> from harmonix.spaces.engineering import ACIRebar
    >>> var = ACIRebar(d_expr=0.55, cc_expr=40.0)
    >>> code = var.sample({})
    >>> dia, n = var.decode(code)
    >>> print(f"{n} bars of Ø{dia} mm")
    """

    def __init__(
        self,
        d_expr,
        cc_expr,
        fc = 30.0,
        fy = 420.0,
    ):
        self._d    = d_expr
        self._cc   = cc_expr
        self._fc   = fc    # may be float or callable(ctx) -> float
        self._fy   = fy    # may be float or callable(ctx) -> float
        self._n    = len(_COUNTS)       # number of count steps per diameter

    @property
    def fc(self): return self._fc if not callable(self._fc) else None
    @property
    def fy(self): return self._fy if not callable(self._fy) else None

    # --- geometry resolution -----------------------------------------------

    def _resolve(self, ctx: Context) -> Tuple[float, float, float, float]:
        d  = self._d(ctx)  if callable(self._d)  else float(self._d)
        cc = self._cc(ctx) if callable(self._cc) else float(self._cc)
        fc = self._fc(ctx) if callable(self._fc) else float(self._fc)
        fy = self._fy(ctx) if callable(self._fy) else float(self._fy)
        return d, cc, fc, fy

    # --- valid code enumeration --------------------------------------------

    def _valid_codes(self, ctx: Context) -> List[int]:
        d, cc, fc, fy = self._resolve(ctx)
        beta1, phi, eps_c, rho_min, rho_max = _aci_limits(fc, fy)
        codes: List[int] = []
        for i, (area50, dia) in enumerate(zip(_AREAS_50, _DIAMETERS)):
            for j, count in enumerate(_COUNTS):
                if _bar_is_valid_single(
                    dia, count, d, cc,
                    beta1, phi, eps_c, rho_min, rho_max,
                    fc, fy, area50,
                ):
                    codes.append(i * self._n + j)
        return codes

    # --- Variable interface ------------------------------------------------

    def sample(self, ctx: Context) -> Optional[int]:
        codes = self._valid_codes(ctx)
        return random.choice(codes) if codes else None

    def filter(self, candidates: List[int], ctx: Context) -> List[int]:
        valid = set(self._valid_codes(ctx))
        return [c for c in candidates if c in valid]

    def neighbor(self, value: int, ctx: Context) -> int:
        """
        Move to an adjacent cell in the (diameter × count) grid.

        Considers all 8 Moore-neighbours; returns a randomly selected
        feasible one, or the original value if no neighbours are feasible.
        """
        valid = set(self._valid_codes(ctx))
        if value not in valid:
            return value

        i = value // self._n
        j = value %  self._n
        neighbours: List[int] = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < len(_DIAMETERS) and 0 <= nj < len(_COUNTS):
                    code = ni * self._n + nj
                    if code in valid:
                        neighbours.append(code)
        return random.choice(neighbours) if neighbours else value

    # --- decode ------------------------------------------------------------

    def decode(self, code: int) -> Tuple[float, int]:
        """
        Return ``(diameter_mm, bar_count)`` for a given *code*.

        Raises
        ------
        ValueError
            When *code* is ``None`` (no feasible arrangement was found for
            the given section geometry).
        """
        if code is None:
            raise ValueError(
                "Cannot decode None: no feasible bar arrangement exists "
                "for the current section geometry and ACI 318 limits."
            )
        i = code // self._n
        j = code %  self._n
        return _DIAMETERS[i], _COUNTS[j]

    def describe(self, code: int) -> str:
        """Human-readable description of a *code*."""
        dia, n = self.decode(code)
        return f"{n} bars of Ø{dia:.2f} mm"


# ---------------------------------------------------------------------------
# ACIDoubleRebar — double-row arrangement
# ---------------------------------------------------------------------------

@register_variable("aci_rebar_double")
class ACIDoubleRebar(Variable):
    """
    ACI 318 ductile double-row reinforcing bar arrangement.

    Identical API to :class:`ACIRebar` but the feasibility check uses
    **two** effective depths: one for each layer of bars.

    The section is considered ductile when **both** layers independently
    satisfy ρ_min ≤ ρ ≤ ρ_max, the neutral-axis depth limit, and the
    minimum clear-spacing rule.

    Parameters
    ----------
    d1_expr : float | callable
        Effective depth of the first (outer) bar layer [m].
    d2_expr : float | callable
        Effective depth of the second (inner) bar layer [m].
    cc_expr : float | callable
        Concrete cover to bar centreline [mm].
    fc : float
        Concrete compressive strength [MPa].
    fy : float
        Steel yield strength [MPa].
    """

    def __init__(
        self,
        d1_expr,
        d2_expr,
        cc_expr,
        fc: float = 30.0,
        fy: float = 420.0,
    ):
        self._d1   = d1_expr
        self._d2   = d2_expr
        self._cc   = cc_expr
        self.fc    = float(fc)
        self.fy    = float(fy)
        self._n    = len(_COUNTS)

    def _resolve(self, ctx: Context) -> Tuple[float, float, float]:
        d1 = self._d1(ctx) if callable(self._d1) else float(self._d1)
        d2 = self._d2(ctx) if callable(self._d2) else float(self._d2)
        cc = self._cc(ctx) if callable(self._cc) else float(self._cc)
        return d1, d2, cc

    def _valid_codes(self, ctx: Context) -> List[int]:
        d1, d2, cc = self._resolve(ctx)
        beta1, phi, eps_c, rho_min, rho_max = _aci_limits(self.fc, self.fy)
        codes: List[int] = []
        for i, (area50, dia) in enumerate(zip(_AREAS_50, _DIAMETERS)):
            for j, count in enumerate(_COUNTS):
                valid_layer1 = _bar_is_valid_single(
                    dia, count, d1, cc,
                    beta1, phi, eps_c, rho_min, rho_max,
                    self.fc, self.fy, area50,
                )
                valid_layer2 = _bar_is_valid_single(
                    dia, count, d2, cc,
                    beta1, phi, eps_c, rho_min, rho_max,
                    self.fc, self.fy, area50,
                )
                if valid_layer1 and valid_layer2:
                    codes.append(i * self._n + j)
        return codes

    def sample(self, ctx: Context) -> Optional[int]:
        codes = self._valid_codes(ctx)
        return random.choice(codes) if codes else None

    def filter(self, candidates: List[int], ctx: Context) -> List[int]:
        valid = set(self._valid_codes(ctx))
        return [c for c in candidates if c in valid]

    def neighbor(self, value: int, ctx: Context) -> int:
        valid = set(self._valid_codes(ctx))
        if value not in valid:
            return value
        i = value // self._n
        j = value %  self._n
        neighbours: List[int] = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < len(_DIAMETERS) and 0 <= nj < len(_COUNTS):
                    code = ni * self._n + nj
                    if code in valid:
                        neighbours.append(code)
        return random.choice(neighbours) if neighbours else value

    def decode(self, code: int) -> Tuple[float, int]:
        """Return ``(diameter_mm, bar_count)`` for a given *code*."""
        i = code // self._n
        j = code %  self._n
        return _DIAMETERS[i], _COUNTS[j]

    def describe(self, code: int) -> str:
        dia, n = self.decode(code)
        return f"{n} bars of Ø{dia:.2f} mm (double row)"


# ===========================================================================
# SteelSection — IPE / HEA / HEB / W catalogue
# ===========================================================================





@dataclass
class SectionProperties:
    """
    Geometric and mechanical properties of a steel section.

    Attributes
    ----------
    name : str
        Section designation, e.g. ``"IPE 200"``.
    series : str
        Family, e.g. ``"IPE"``, ``"HEA"``, ``"HEB"``, ``"W"``.
    h_mm : float
        Total height [mm].
    b_mm : float
        Flange width [mm].
    tf_mm : float
        Flange thickness [mm].
    tw_mm : float
        Web thickness [mm].
    A_cm2 : float
        Cross-sectional area [cm²].
    Iy_cm4 : float
        Second moment of area about the strong axis [cm⁴].
    Wy_cm3 : float
        Elastic section modulus about the strong axis [cm³].
    Iz_cm4 : float
        Second moment of area about the weak axis [cm⁴].
    Wz_cm3 : float
        Elastic section modulus about the weak axis [cm³].
    mass_kg_m : float
        Mass per unit length [kg/m].
    """
    name:       str
    series:     str
    h_mm:       float
    b_mm:       float
    tf_mm:      float
    tw_mm:      float
    A_cm2:      float
    Iy_cm4:     float
    Wy_cm3:     float
    Iz_cm4:     float
    Wz_cm3:     float
    mass_kg_m:  float


# ---------------------------------------------------------------------------
# Built-in catalogue  (EN 10365:2017 + AISC 16th Ed. selection)
# ---------------------------------------------------------------------------

_BUILTIN_CATALOGUE: List[SectionProperties] = [
    # --- IPE series (European I-beams) ------------------------------------
    SectionProperties("IPE 80",   "IPE",  80,  46,  5.2, 3.8,  7.64,   80.1,  20.0,   8.49,  3.69,  6.0),
    SectionProperties("IPE 100",  "IPE", 100,  55,  5.7, 4.1,  10.3,  171,    34.2,   15.9,  5.79,  8.1),
    SectionProperties("IPE 120",  "IPE", 120,  64,  6.3, 4.4,  13.2,  318,    53.0,   27.7,  8.65,  10.4),
    SectionProperties("IPE 140",  "IPE", 140,  73,  6.9, 4.7,  16.4,  541,    77.3,   44.9,  12.3,  12.9),
    SectionProperties("IPE 160",  "IPE", 160,  82,  7.4, 5.0,  20.1,  869,   109,     68.3,  16.7,  15.8),
    SectionProperties("IPE 180",  "IPE", 180,  91,  8.0, 5.3,  23.9,  1320,  146,     101,   22.2,  18.8),
    SectionProperties("IPE 200",  "IPE", 200, 100,  8.5, 5.6,  28.5,  1940,  194,     142,   28.5,  22.4),
    SectionProperties("IPE 220",  "IPE", 220, 110,  9.2, 5.9,  33.4,  2770,  252,     205,   37.3,  26.2),
    SectionProperties("IPE 240",  "IPE", 240, 120,  9.8, 6.2,  39.1,  3890,  324,     284,   47.3,  30.7),
    SectionProperties("IPE 270",  "IPE", 270, 135, 10.2, 6.6,  45.9,  5790,  429,     420,   62.2,  36.1),
    SectionProperties("IPE 300",  "IPE", 300, 150, 10.7, 7.1,  53.8,  8360,  557,     604,   80.5,  42.2),
    SectionProperties("IPE 330",  "IPE", 330, 160, 11.5, 7.5,  62.6, 11770,  713,     788,   98.5,  49.1),
    SectionProperties("IPE 360",  "IPE", 360, 170, 12.7, 8.0,  72.7, 16270,  904,    1040,  123,   57.1),
    SectionProperties("IPE 400",  "IPE", 400, 180, 13.5, 8.6,  84.5, 23130, 1156,    1320,  146,   66.3),
    SectionProperties("IPE 450",  "IPE", 450, 190, 14.6, 9.4,  98.8, 33740, 1500,    1680,  177,   77.6),
    SectionProperties("IPE 500",  "IPE", 500, 200, 16.0,10.2, 116,   48200, 1928,    2140,  214,   90.7),
    SectionProperties("IPE 550",  "IPE", 550, 210, 17.2,11.1, 134,   67120, 2441,    2670,  255,  106),
    SectionProperties("IPE 600",  "IPE", 600, 220, 19.0,12.0, 156,   92080, 3069,    3390,  308,  122),
    # --- HEA series -------------------------------------------------------
    SectionProperties("HEA 100",  "HEA", 96,  100, 8.0, 5.0,  21.2,  349,   72.8,   134,   26.8,  16.7),
    SectionProperties("HEA 120",  "HEA",114,  120, 8.0, 5.0,  25.3,  606,  106,     231,   38.5,  19.9),
    SectionProperties("HEA 140",  "HEA",133,  140, 8.5, 5.5,  31.4, 1033,  155,     389,   55.6,  24.7),
    SectionProperties("HEA 160",  "HEA",152,  160, 9.0, 6.0,  38.8, 1673,  220,     616,   77.0,  30.4),
    SectionProperties("HEA 180",  "HEA",171,  180, 9.5, 6.0,  45.3, 2510,  294,     925,  103,   35.5),
    SectionProperties("HEA 200",  "HEA",190,  200,10.0, 6.5,  53.8, 3692,  389,    1336,  134,   42.3),
    SectionProperties("HEA 220",  "HEA",210,  220,11.0, 7.0,  64.3, 5410,  515,    1955,  178,   50.5),
    SectionProperties("HEA 240",  "HEA",230,  240,12.0, 7.5,  76.8, 7763,  675,    2769,  231,   60.3),
    SectionProperties("HEA 260",  "HEA",250,  260,12.5, 7.5,  86.8,10450,  836,    3668,  282,   68.2),
    SectionProperties("HEA 280",  "HEA",270,  280,13.0, 8.0,  97.3,13670, 1013,    4763,  340,   76.4),
    SectionProperties("HEA 300",  "HEA",290,  300,14.0, 8.5, 112,  18260, 1260,    6310,  421,   88.3),
    SectionProperties("HEA 320",  "HEA",310,  300,15.5, 9.0, 124,  22930, 1479,    6986,  466,   97.6),
    SectionProperties("HEA 340",  "HEA",330,  300,16.5, 9.5, 133,  27690, 1679,    7436,  496,  105),
    SectionProperties("HEA 360",  "HEA",350,  300,17.5,10.0, 143,  33090, 1891,    7887,  526,  112),
    SectionProperties("HEA 400",  "HEA",390,  300,19.0,11.0, 159,  45070, 2311,    8564,  571,  125),
    SectionProperties("HEA 450",  "HEA",440,  300,21.0,11.5, 178,  63720, 2896,    9465,  631,  140),
    SectionProperties("HEA 500",  "HEA",490,  300,23.0,12.0, 198,  86970, 3550,   10370,  691,  155),
    # --- HEB series -------------------------------------------------------
    SectionProperties("HEB 100",  "HEB", 100, 100,10.0, 6.0,  26.0,  450,   89.9,   167,   33.4,  20.4),
    SectionProperties("HEB 120",  "HEB", 120, 120,11.0, 6.5,  34.0,  864,  144,     318,   52.9,  26.7),
    SectionProperties("HEB 140",  "HEB", 140, 140,12.0, 7.0,  43.0, 1509,  216,     550,   78.5,  33.7),
    SectionProperties("HEB 160",  "HEB", 160, 160,13.0, 8.0,  54.3, 2492,  311,     889,  111,   42.6),
    SectionProperties("HEB 180",  "HEB", 180, 180,14.0, 8.5,  65.3, 3831,  426,    1363,  151,   51.2),
    SectionProperties("HEB 200",  "HEB", 200, 200,15.0, 9.0,  78.1, 5696,  570,    2003,  200,   61.3),
    SectionProperties("HEB 220",  "HEB", 220, 220,16.0, 9.5,  91.0, 8091,  736,    2843,  258,   71.5),
    SectionProperties("HEB 240",  "HEB", 240, 240,17.0,10.0, 106,  11260,  938,    3923,  327,   83.2),
    SectionProperties("HEB 260",  "HEB", 260, 260,17.5,10.0, 118,  14920, 1148,    5135,  395,   93.0),
    SectionProperties("HEB 280",  "HEB", 280, 280,18.0,10.5, 131,  19270, 1376,    6595,  471,  103),
    SectionProperties("HEB 300",  "HEB", 300, 300,19.0,11.0, 149,  25170, 1678,    8563,  571,  117),
    SectionProperties("HEB 320",  "HEB", 320, 300,20.5,11.5, 161,  30820, 1926,    9239,  616,  127),
    SectionProperties("HEB 340",  "HEB", 340, 300,21.5,12.0, 171,  36660, 2156,    9690,  646,  134),
    SectionProperties("HEB 360",  "HEB", 360, 300,22.5,12.5, 181,  43190, 2400,   10140,  676,  142),
    SectionProperties("HEB 400",  "HEB", 400, 300,24.0,13.5, 198,  57680, 2884,   10820,  721,  155),
    SectionProperties("HEB 450",  "HEB", 450, 300,26.0,14.0, 218,  79890, 3551,   11720,  781,  171),
    SectionProperties("HEB 500",  "HEB", 500, 300,28.0,14.5, 239, 107200, 4287,   12620,  842,  187),
    # --- W-sections (AISC, imperial-origin but metric properties) ----------
    SectionProperties("W 6x9",    "W",  152,  152,  5.8, 3.6,  17.1,  615,   80.8,   200,   26.3,  13.4),
    SectionProperties("W 8x18",   "W",  207,  134,  8.0, 5.8,  34.2, 2280,  220,    274,   40.8,  26.9),
    SectionProperties("W 10x22",  "W",  257,  146,  8.1, 5.8,  41.8, 4610,  359,    357,   48.9,  32.7),
    SectionProperties("W 12x26",  "W",  310,  165,  9.7, 5.8,  49.4, 8660,  559,    528,   64.0,  38.8),
    SectionProperties("W 14x30",  "W",  352,  171,  9.8, 6.9,  56.8,12400,  705,    582,   68.1,  44.6),
    SectionProperties("W 16x36",  "W",  403,  178, 10.9, 7.5,  68.4,18700,  928,    671,   75.4,  53.7),
    SectionProperties("W 18x46",  "W",  459,  191, 11.2, 7.6,  87.1,28600, 1246,    897,   94.0,  68.4),
    SectionProperties("W 21x57",  "W",  535,  214, 11.4, 8.1, 108,  50900, 1903,   1380,  129,   84.8),
    SectionProperties("W 24x68",  "W",  603,  228, 11.9, 9.1, 129,  76200, 2528,   1700,  149,  101),
    SectionProperties("W 27x84",  "W",  686,  254, 11.7, 9.7, 159, 119000, 3469,   2280,  180,  125),
    SectionProperties("W 30x90",  "W",  756,  267, 11.4, 9.4, 171, 150000, 3967,   2280,  170,  134),
    SectionProperties("W 33x118", "W",  845,  292, 14.0,11.2, 224, 237000, 5616,   3500,  240,  176),
    SectionProperties("W 36x135", "W",  912,  304, 15.2,11.9, 256, 328000, 7196,   4230,  279,  201),
]


def _load_catalogue_from_file(path) -> List[SectionProperties]:
    """
    Load a section catalogue from a JSON or CSV file.

    JSON format — list of objects with keys matching SectionProperties fields:

    .. code-block:: json

        [
          {"name": "IPE 200", "series": "IPE", "h_mm": 200, "b_mm": 100,
           "tf_mm": 8.5, "tw_mm": 5.6, "A_cm2": 28.5, "Iy_cm4": 1940,
           "Wy_cm3": 194, "Iz_cm4": 142, "Wz_cm3": 28.5, "mass_kg_m": 22.4}
        ]

    CSV format — header row must contain the same field names.
    """
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        return [SectionProperties(**row) for row in data]
    elif p.suffix.lower() == ".csv":
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            return [
                SectionProperties(
                    name=row["name"], series=row["series"],
                    h_mm=float(row["h_mm"]),   b_mm=float(row["b_mm"]),
                    tf_mm=float(row["tf_mm"]), tw_mm=float(row["tw_mm"]),
                    A_cm2=float(row["A_cm2"]), Iy_cm4=float(row["Iy_cm4"]),
                    Wy_cm3=float(row["Wy_cm3"]),Iz_cm4=float(row["Iz_cm4"]),
                    Wz_cm3=float(row["Wz_cm3"]),mass_kg_m=float(row["mass_kg_m"]),
                )
                for row in reader
            ]
    else:
        raise ValueError(f"Unsupported catalogue format: {p.suffix!r}. Use .json or .csv")


@register_variable("steel_section")
class SteelSection(Variable):
    """
    Variable whose domain is a catalogue of standard steel sections.

    The value stored in the harmony is an **index** into the section list
    (integer).  Use :meth:`decode` to retrieve the full
    :class:`SectionProperties` object.

    Parameters
    ----------
    series : str | list[str] | None
        Restrict the domain to one or more section families, e.g.
        ``"IPE"``, ``["HEA", "HEB"]``.  ``None`` means all series.
    catalogue : list[SectionProperties] | str | Path, optional
        Custom catalogue.  Pass a list of :class:`SectionProperties`,
        or a path to a ``.json`` / ``.csv`` file.  When omitted the
        built-in EN 10365 + AISC catalogue is used.

    Examples
    --------
    >>> var = SteelSection(series="IPE")
    >>> idx = var.sample({})
    >>> sec = var.decode(idx)
    >>> print(sec.name, sec.Iy_cm4)

    Using a custom catalogue file:

    >>> var = SteelSection(catalogue="my_sections.json")
    """

    def __init__(
        self,
        series=None,
        catalogue=None,
    ):
        # Resolve catalogue
        if catalogue is None:
            base = _BUILTIN_CATALOGUE
        elif isinstance(catalogue, (str, _Path)):
            base = _load_catalogue_from_file(catalogue)
        else:
            base = list(catalogue)

        # Filter by series
        if series is None:
            self._sections = base
        elif isinstance(series, str):
            self._sections = [s for s in base if s.series == series]
        else:
            keep = set(series)
            self._sections = [s for s in base if s.series in keep]

        if not self._sections:
            raise ValueError(
                f"No sections found for series={series!r}. "
                f"Available: {sorted({s.series for s in base})}"
            )

        self._indices = list(range(len(self._sections)))

    def sample(self, ctx) -> int:
        return random.choice(self._indices)

    def filter(self, candidates: List[int], ctx) -> List[int]:
        valid = set(self._indices)
        return [c for c in candidates if c in valid]

    def neighbor(self, value: int, ctx) -> int:
        """Move one step up or down in the catalogue (sorted by mass)."""
        if value not in self._indices:
            return self.sample(ctx)
        delta = random.choice([-1, 1])
        new_idx = max(0, min(len(self._indices) - 1, value + delta))
        return new_idx

    def decode(self, index: int) -> SectionProperties:
        """Return the :class:`SectionProperties` for *index*."""
        return self._sections[index]

    def describe(self, index: int) -> str:
        s = self.decode(index)
        return f"{s.name}  (A={s.A_cm2} cm², Iy={s.Iy_cm4} cm⁴, {s.mass_kg_m} kg/m)"

    @property
    def sections(self) -> List[SectionProperties]:
        """All sections in the current domain."""
        return list(self._sections)


# ===========================================================================
# ConcreteGrade — EN 206 strength classes
# ===========================================================================

@dataclass
class ConcreteGradeProperties:
    """
    Characteristic properties of a concrete strength class (EN 1992-1-1).

    Attributes
    ----------
    name : str
        Designation, e.g. ``"C30/37"``.
    fck_MPa : float
        Characteristic cylinder compressive strength [MPa].
    fck_cube_MPa : float
        Characteristic cube compressive strength [MPa].
    fcm_MPa : float
        Mean compressive strength (= fck + 8) [MPa].
    fctm_MPa : float
        Mean tensile strength [MPa].
    Ecm_GPa : float
        Secant modulus of elasticity [GPa].
    eps_cu : float
        Ultimate compressive strain [‰].
    """
    name:          str
    fck_MPa:       float
    fck_cube_MPa:  float
    fcm_MPa:       float
    fctm_MPa:      float
    Ecm_GPa:       float
    eps_cu:        float


_CONCRETE_GRADES: List[ConcreteGradeProperties] = [
    # name        fck   fck_cube  fcm    fctm   Ecm   eps_cu
    ConcreteGradeProperties("C12/15",  12,  15,  20,  1.57, 27.1, 3.5),
    ConcreteGradeProperties("C16/20",  16,  20,  24,  1.90, 29.0, 3.5),
    ConcreteGradeProperties("C20/25",  20,  25,  28,  2.21, 30.0, 3.5),
    ConcreteGradeProperties("C25/30",  25,  30,  33,  2.56, 31.5, 3.5),
    ConcreteGradeProperties("C30/37",  30,  37,  38,  2.90, 33.0, 3.5),
    ConcreteGradeProperties("C35/45",  35,  45,  43,  3.21, 34.0, 3.5),
    ConcreteGradeProperties("C40/50",  40,  50,  48,  3.51, 35.0, 3.5),
    ConcreteGradeProperties("C45/55",  45,  55,  53,  3.80, 36.0, 3.5),
    ConcreteGradeProperties("C50/60",  50,  60,  58,  4.07, 37.0, 3.4),
    ConcreteGradeProperties("C55/67",  55,  67,  63,  4.21, 38.0, 3.4),
    ConcreteGradeProperties("C60/75",  60,  75,  68,  4.35, 39.0, 3.3),
    ConcreteGradeProperties("C70/85",  70,  85,  78,  4.61, 41.0, 3.2),
    ConcreteGradeProperties("C80/95",  80,  95,  88,  4.84, 42.0, 3.0),
    ConcreteGradeProperties("C90/105", 90, 105,  98,  5.04, 44.0, 2.8),
]

_CONCRETE_INDEX: Dict[str, int] = {g.name: i for i, g in enumerate(_CONCRETE_GRADES)}


@register_variable("concrete_grade")
class ConcreteGrade(Variable):
    """
    Variable whose domain is a set of EN 206 concrete strength classes.

    The harmony stores an **index** into the grade list.  Use
    :meth:`decode` to retrieve the full :class:`ConcreteGradeProperties`.

    Parameters
    ----------
    min_grade : str, optional
        Minimum allowable grade name, e.g. ``"C25/30"``.
        Grades below this are excluded.
    max_grade : str, optional
        Maximum allowable grade name, e.g. ``"C60/75"``.

    Examples
    --------
    >>> var = ConcreteGrade(min_grade="C25/30", max_grade="C50/60")
    >>> idx = var.sample({})
    >>> print(var.decode(idx).name, var.decode(idx).fck_MPa, "MPa")
    """

    def __init__(
        self,
        min_grade: Optional[str] = None,
        max_grade: Optional[str] = None,
    ):
        lo = _CONCRETE_INDEX.get(min_grade, 0) if min_grade else 0
        hi = _CONCRETE_INDEX.get(max_grade, len(_CONCRETE_GRADES) - 1) if max_grade else len(_CONCRETE_GRADES) - 1
        if lo > hi:
            raise ValueError(f"min_grade '{min_grade}' is above max_grade '{max_grade}'.")
        self._grades  = _CONCRETE_GRADES[lo: hi + 1]
        self._indices = list(range(len(self._grades)))

    def sample(self, ctx) -> int:
        return random.choice(self._indices)

    def filter(self, candidates: List[int], ctx) -> List[int]:
        valid = set(self._indices)
        return [c for c in candidates if c in valid]

    def neighbor(self, value: int, ctx) -> int:
        delta   = random.choice([-1, 1])
        new_idx = max(0, min(len(self._indices) - 1, value + delta))
        return new_idx

    def decode(self, index: int) -> ConcreteGradeProperties:
        """Return :class:`ConcreteGradeProperties` for *index*."""
        return self._grades[index]

    def describe(self, index: int) -> str:
        g = self.decode(index)
        return f"{g.name}  (fck={g.fck_MPa} MPa, Ecm={g.Ecm_GPa} GPa)"


# ===========================================================================
# SoilSPT — SPT-N based soil classification
# ===========================================================================

@dataclass
class SoilProfile:
    """
    Geotechnical properties derived from SPT-N blow count.

    Attributes
    ----------
    name : str
        Profile label, e.g. ``"Medium dense sand"``.
    N_lo : int
        Lower bound SPT-N blow count (blows / 300 mm).
    N_hi : int
        Upper bound SPT-N blow count.
    site_class : str
        TBDY 2018 / ASCE 7 site class (ZA … ZE / A … E).
    phi_deg : float
        Estimated friction angle [degrees] (granular soils).
    cu_kPa : float
        Estimated undrained shear strength [kPa] (cohesive soils; 0 if N/A).
    Dr_pct : float
        Relative density estimate [%] (granular soils; 0 if N/A).
    vs30_mps : float
        Estimated shear-wave velocity Vs30 [m/s].
    """
    name:       str
    N_lo:       int
    N_hi:       int
    site_class: str
    phi_deg:    float
    cu_kPa:     float
    Dr_pct:     float
    vs30_mps:   float


_SOIL_PROFILES: List[SoilProfile] = [
    # Cohesionless (granular) soils
    SoilProfile("Very loose sand",        0,  4,  "ZE",  26,   0,   15,  130),
    SoilProfile("Loose sand",             4, 10,  "ZD",  28,   0,   30,  175),
    SoilProfile("Medium dense sand",     10, 30,  "ZC",  32,   0,   55,  260),
    SoilProfile("Dense sand",            30, 50,  "ZC",  36,   0,   75,  340),
    SoilProfile("Very dense sand/gravel",50, 100, "ZB",  40,   0,   90,  500),
    # Cohesive soils
    SoilProfile("Very soft clay",         0,  2,  "ZE",   0,  12,    0,   80),
    SoilProfile("Soft clay",              2,  4,  "ZE",   0,  25,    0,  120),
    SoilProfile("Medium stiff clay",      4,  8,  "ZD",   0,  50,    0,  170),
    SoilProfile("Stiff clay",             8, 15,  "ZD",   0, 100,    0,  220),
    SoilProfile("Very stiff clay",       15, 30,  "ZC",   0, 200,    0,  310),
    SoilProfile("Hard clay",             30, 50,  "ZC",   0, 400,    0,  420),
    # Rock / dense gravel
    SoilProfile("Soft rock / dense grav",50,100,  "ZB",  42,   0,   95,  600),
    SoilProfile("Soft rock",            100,200,  "ZB",   0,   0,    0,  800),
    SoilProfile("Rock",                 200,300,  "ZA",   0,   0,    0, 1200),
]


@register_variable("soil_spt")
class SoilSPT(Variable):
    """
    Variable whose domain is a set of SPT-N based soil profile classes.

    Parameters
    ----------
    site_classes : list[str] | None
        Restrict to specific TBDY / ASCE site classes,
        e.g. ``["ZC", "ZD"]``.  ``None`` = all classes.
    N_min, N_max : int, optional
        Additional SPT-N blow-count range filter.

    Examples
    --------
    >>> var = SoilSPT(site_classes=["ZC", "ZD"])
    >>> idx = var.sample({})
    >>> print(var.decode(idx).name, var.decode(idx).vs30_mps, "m/s")
    """

    def __init__(
        self,
        site_classes: Optional[List[str]] = None,
        N_min: int = 0,
        N_max: int = 9999,
    ):
        filtered = [
            p for p in _SOIL_PROFILES
            if p.N_lo >= N_min and p.N_hi <= N_max
            and (site_classes is None or p.site_class in site_classes)
        ]
        if not filtered:
            raise ValueError(
                f"No soil profiles match site_classes={site_classes}, "
                f"N_min={N_min}, N_max={N_max}."
            )
        self._profiles = filtered
        self._indices  = list(range(len(filtered)))

    def sample(self, ctx) -> int:
        return random.choice(self._indices)

    def filter(self, candidates: List[int], ctx) -> List[int]:
        valid = set(self._indices)
        return [c for c in candidates if c in valid]

    def neighbor(self, value: int, ctx) -> int:
        delta   = random.choice([-1, 1])
        new_idx = max(0, min(len(self._indices) - 1, value + delta))
        return new_idx

    def decode(self, index: int) -> SoilProfile:
        """Return :class:`SoilProfile` for *index*."""
        return self._profiles[index]

    def describe(self, index: int) -> str:
        p = self.decode(index)
        return (f"{p.name}  (N={p.N_lo}–{p.N_hi}, "
                f"site class {p.site_class}, Vs30≈{p.vs30_mps} m/s)")


# ===========================================================================
# SeismicZoneTBDY — TBDY 2018 spectral acceleration parameters
# ===========================================================================

@dataclass
class SeismicZone:
    """
    Spectral acceleration parameters for a TBDY 2018 seismic hazard level.

    Attributes
    ----------
    name : str
        Zone label, e.g. ``"DD-2 / ZC (moderate)"``.
    hazard_level : str
        Return-period code: DD-1 (2475 yr), DD-2 (475 yr),
        DD-3 (72 yr), DD-4 (43 yr).
    site_class : str
        TBDY site class ZA … ZE.
    Ss : float
        Short-period spectral acceleration [g] (0.2 s).
    S1 : float
        One-second spectral acceleration [g].
    SDS : float
        Design short-period spectral acceleration = Fs·Ss [g].
    SD1 : float
        Design 1-s spectral acceleration = F1·S1 [g].
    PGA : float
        Peak ground acceleration [g].
    """
    name:         str
    hazard_level: str
    site_class:   str
    Ss:           float
    S1:           float
    SDS:          float
    SD1:          float
    PGA:          float


# Representative TBDY 2018 / AFAD TDTH grid (DD-2, 475-yr return period)
# Ss and S1 values span the range of Turkey's seismic map; site amplification
# factors Fs and F1 applied for each site class per TBDY 2018 Table 2.3-2.4.
_SEISMIC_ZONES: List[SeismicZone] = [
    # ── ZA (rock, Vs30 > 1500 m/s) ─────────────────────────────────────────
    SeismicZone("DD-2 / ZA / Very low",  "DD-2","ZA",0.25,0.08,0.250,0.080,0.10),
    SeismicZone("DD-2 / ZA / Low",       "DD-2","ZA",0.50,0.14,0.500,0.140,0.20),
    SeismicZone("DD-2 / ZA / Moderate",  "DD-2","ZA",0.75,0.20,0.750,0.200,0.30),
    SeismicZone("DD-2 / ZA / High",      "DD-2","ZA",1.00,0.30,1.000,0.300,0.40),
    SeismicZone("DD-2 / ZA / Very high", "DD-2","ZA",1.50,0.45,1.500,0.450,0.60),
    # ── ZB (soft rock, 760–1500 m/s) ────────────────────────────────────────
    SeismicZone("DD-2 / ZB / Very low",  "DD-2","ZB",0.25,0.08,0.275,0.088,0.11),
    SeismicZone("DD-2 / ZB / Low",       "DD-2","ZB",0.50,0.14,0.550,0.154,0.22),
    SeismicZone("DD-2 / ZB / Moderate",  "DD-2","ZB",0.75,0.20,0.825,0.220,0.33),
    SeismicZone("DD-2 / ZB / High",      "DD-2","ZB",1.00,0.30,1.100,0.330,0.44),
    SeismicZone("DD-2 / ZB / Very high", "DD-2","ZB",1.50,0.45,1.650,0.495,0.66),
    # ── ZC (dense soil, 360–760 m/s) ────────────────────────────────────────
    SeismicZone("DD-2 / ZC / Very low",  "DD-2","ZC",0.25,0.08,0.313,0.112,0.12),
    SeismicZone("DD-2 / ZC / Low",       "DD-2","ZC",0.50,0.14,0.625,0.196,0.25),
    SeismicZone("DD-2 / ZC / Moderate",  "DD-2","ZC",0.75,0.20,0.900,0.280,0.36),
    SeismicZone("DD-2 / ZC / High",      "DD-2","ZC",1.00,0.30,1.200,0.420,0.48),
    SeismicZone("DD-2 / ZC / Very high", "DD-2","ZC",1.50,0.45,1.800,0.630,0.72),
    # ── ZD (stiff soil, 180–360 m/s) ────────────────────────────────────────
    SeismicZone("DD-2 / ZD / Very low",  "DD-2","ZD",0.25,0.08,0.350,0.136,0.14),
    SeismicZone("DD-2 / ZD / Low",       "DD-2","ZD",0.50,0.14,0.700,0.238,0.28),
    SeismicZone("DD-2 / ZD / Moderate",  "DD-2","ZD",0.75,0.20,1.050,0.340,0.42),
    SeismicZone("DD-2 / ZD / High",      "DD-2","ZD",1.00,0.30,1.400,0.510,0.56),
    SeismicZone("DD-2 / ZD / Very high", "DD-2","ZD",1.50,0.45,2.100,0.765,0.84),
    # ── ZE (soft clay, Vs30 < 180 m/s) ──────────────────────────────────────
    SeismicZone("DD-2 / ZE / Very low",  "DD-2","ZE",0.25,0.08,0.438,0.200,0.17),
    SeismicZone("DD-2 / ZE / Low",       "DD-2","ZE",0.50,0.14,0.875,0.350,0.35),
    SeismicZone("DD-2 / ZE / Moderate",  "DD-2","ZE",0.75,0.20,1.200,0.500,0.48),
    SeismicZone("DD-2 / ZE / High",      "DD-2","ZE",1.00,0.30,1.500,0.750,0.60),
    SeismicZone("DD-2 / ZE / Very high", "DD-2","ZE",1.50,0.45,2.000,1.125,0.80),
]


@register_variable("seismic_tbdy")
class SeismicZoneTBDY(Variable):
    """
    Variable whose domain is a set of TBDY 2018 seismic hazard zones.

    Each combination of hazard level, site class, and ground-motion
    intensity level is encoded as an index; use :meth:`decode` to
    retrieve the full :class:`SeismicZone` with Ss, S1, SDS, SD1, PGA.

    Parameters
    ----------
    hazard_levels : list[str] | None
        Filter by return period code: ``"DD-1"``, ``"DD-2"``,
        ``"DD-3"``, ``"DD-4"``.  ``None`` = all.
    site_classes : list[str] | None
        Filter by TBDY site class: ``"ZA"`` … ``"ZE"``.  ``None`` = all.

    Examples
    --------
    >>> var = SeismicZoneTBDY(hazard_levels=["DD-2"], site_classes=["ZC","ZD"])
    >>> idx = var.sample({})
    >>> z = var.decode(idx)
    >>> print(f"SDS={z.SDS} g, SD1={z.SD1} g")
    """

    def __init__(
        self,
        hazard_levels: Optional[List[str]] = None,
        site_classes:  Optional[List[str]] = None,
    ):
        filtered = [
            z for z in _SEISMIC_ZONES
            if (hazard_levels is None or z.hazard_level in hazard_levels)
            and (site_classes  is None or z.site_class  in site_classes)
        ]
        if not filtered:
            raise ValueError(
                f"No seismic zones match hazard_levels={hazard_levels}, "
                f"site_classes={site_classes}."
            )
        self._zones   = filtered
        self._indices = list(range(len(filtered)))

    def sample(self, ctx) -> int:
        return random.choice(self._indices)

    def filter(self, candidates: List[int], ctx) -> List[int]:
        valid = set(self._indices)
        return [c for c in candidates if c in valid]

    def neighbor(self, value: int, ctx) -> int:
        """Step to an adjacent zone (ordered by increasing SDS)."""
        ordered = sorted(
            self._indices, key=lambda i: self._zones[i].SDS
        )
        pos     = ordered.index(value) if value in ordered else 0
        delta   = random.choice([-1, 1])
        new_pos = max(0, min(len(ordered) - 1, pos + delta))
        return ordered[new_pos]

    def decode(self, index: int) -> SeismicZone:
        """Return :class:`SeismicZone` for *index*."""
        return self._zones[index]

    def describe(self, index: int) -> str:
        z = self.decode(index)
        return (f"{z.name}  "
                f"(SDS={z.SDS:.3f} g, SD1={z.SD1:.3f} g, PGA={z.PGA:.2f} g)")