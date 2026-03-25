"""
tests/test_spaces.py
====================
Tests for harmonix.spaces — math and engineering domain spaces.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from harmonix.spaces.engineering import (
    ACIDoubleRebar,
    ACIRebar,
    ConcreteGrade,
    SectionProperties,
    SeismicZoneTBDY,
    SoilSPT,
    SteelSection,
    _aci_limits,
    _load_catalogue_from_file,
)
from harmonix.spaces.math import (
    Fibonacci,
    NaturalNumber,
    NegativeInt,
    NegativeReal,
    PositiveReal,
    PowerOfTwo,
    PrimeVariable,
    WholeNumber,
    _fibonacci_in_range,
    _powers_of_two_in_range,
    _sieve,
)

# ---------------------------------------------------------------------------
# Math spaces — shared behaviour
# ---------------------------------------------------------------------------


class TestNaturalNumber:
    def test_sample_in_range(self):
        v = NaturalNumber(hi=20)
        for _ in range(50):
            s = v.sample({})
            assert 1 <= s <= 20

    def test_lo_must_be_positive(self):
        with pytest.raises(ValueError):
            NaturalNumber(hi=10, lo=0)

    def test_filter(self):
        v = NaturalNumber(hi=5)
        assert v.filter([0, 1, 3, 5, 6], {}) == [1, 3, 5]

    def test_neighbor_adjacent(self):
        v = NaturalNumber(hi=10)
        nb = v.neighbor(5, {})
        assert nb in {4, 6}

    def test_neighbor_at_bounds(self):
        v = NaturalNumber(hi=5)
        for _ in range(20):
            nb = v.neighbor(1, {})
            assert nb in {1, 2}
        for _ in range(20):
            nb = v.neighbor(5, {})
            assert nb in {4, 5}


class TestWholeNumber:
    def test_includes_zero(self):
        v = WholeNumber(hi=10)
        samples = {v.sample({}) for _ in range(200)}
        assert 0 in samples

    def test_filter(self):
        v = WholeNumber(hi=5)
        assert v.filter([-1, 0, 3, 6], {}) == [0, 3]

    def test_neighbor_clamps_at_bounds(self):
        v = WholeNumber(hi=5)
        for _ in range(20):
            assert v.neighbor(0, {}) in {0, 1}
        for _ in range(20):
            assert v.neighbor(5, {}) in {4, 5}


class TestNegativeInt:
    def test_all_negative(self):
        v = NegativeInt(lo=-20)
        for _ in range(50):
            assert v.sample({}) < 0

    def test_lo_must_be_negative(self):
        with pytest.raises(ValueError):
            NegativeInt(lo=0)

    def test_filter(self):
        v = NegativeInt(lo=-5)
        assert v.filter([-6, -5, -1, 0, 1], {}) == [-5, -1]

    def test_neighbor_clamps_at_bounds(self):
        v = NegativeInt(lo=-5)
        for _ in range(20):
            assert v.neighbor(-5, {}) in {-5, -4}
        for _ in range(20):
            assert v.neighbor(-1, {}) in {-2, -1}


class TestNegativeReal:
    def test_all_negative(self):
        v = NegativeReal(lo=-100.0)
        for _ in range(50):
            assert v.sample({}) < 0

    def test_lo_must_be_negative(self):
        with pytest.raises(ValueError):
            NegativeReal(lo=0.0)

    def test_filter_and_neighbor_clamp(self):
        v = NegativeReal(lo=-10.0)
        assert v.filter([-10.0, -5.0, 0.0, 1.0], {}) == [-10.0, -5.0]
        for _ in range(20):
            nb = v.neighbor(-1e-9, {})
            assert -10.0 <= nb <= -1e-9


class TestPositiveReal:
    def test_all_positive(self):
        v = PositiveReal(hi=100.0)
        for _ in range(50):
            assert v.sample({}) > 0

    def test_hi_must_be_positive(self):
        with pytest.raises(ValueError):
            PositiveReal(hi=-1.0)

    def test_filter_and_neighbor_clamp(self):
        v = PositiveReal(hi=10.0)
        assert v.filter([-1.0, 0.0, 1e-9, 3.0, 11.0], {}) == [1e-9, 3.0]
        for _ in range(20):
            nb = v.neighbor(10.0, {})
            assert 1e-9 <= nb <= 10.0


class TestPrimeVariable:
    def test_all_prime(self):
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        v = PrimeVariable(hi=50)
        for _ in range(50):
            s = v.sample({})
            assert is_prime(s), f"{s} is not prime"

    def test_in_range(self):
        v = PrimeVariable(lo=10, hi=30)
        for _ in range(50):
            s = v.sample({})
            assert 10 <= s <= 30

    def test_filter(self):
        v = PrimeVariable(hi=20)
        result = v.filter([1, 2, 3, 4, 5, 10, 11], {})
        assert set(result) == {2, 3, 5, 11}

    def test_neighbor_is_adjacent_prime(self):
        v = PrimeVariable(hi=50)
        primes = v._primes
        nb = v.neighbor(11, {})
        idx = primes.index(11)
        assert nb in (primes[idx - 1] if idx > 0 else 11, primes[idx + 1] if idx < len(primes) - 1 else 11)

    def test_no_primes_raises(self):
        with pytest.raises(ValueError):
            PrimeVariable(lo=14, hi=16)  # no primes between 14 and 16

    def test_invalid_neighbor_falls_back_to_sample(self, monkeypatch):
        v = PrimeVariable(hi=20)
        monkeypatch.setattr(random, "choice", lambda seq: seq[-1])
        assert v.neighbor(4, {}) == v._primes[-1]


class TestPowerOfTwo:
    def test_all_powers(self):
        v = PowerOfTwo(hi=128)
        valid = {1, 2, 4, 8, 16, 32, 64, 128}
        for _ in range(50):
            assert v.sample({}) in valid

    def test_neighbor_adjacent(self):
        v = PowerOfTwo(hi=64)
        nb = v.neighbor(8, {})
        assert nb in {4, 16}

    def test_filter_keeps_only_valid_powers(self):
        v = PowerOfTwo(hi=64)
        assert v.filter([1, 2, 3, 4, 5, 64, 65], {}) == [1, 2, 4, 64]

    def test_no_values_raises(self):
        with pytest.raises(ValueError):
            PowerOfTwo(lo=5, hi=6)  # no powers of two in [5,6]

    def test_invalid_neighbor_falls_back_to_sample(self, monkeypatch):
        v = PowerOfTwo(hi=64)
        monkeypatch.setattr(random, "choice", lambda seq: seq[0])
        assert v.neighbor(3, {}) == 1


class TestFibonacci:
    def test_all_fibonacci(self):
        fib_set = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}
        v = Fibonacci(hi=100)
        for _ in range(50):
            assert v.sample({}) in fib_set

    def test_filter(self):
        v = Fibonacci(hi=20)
        result = v.filter([1, 2, 4, 5, 7, 8], {})
        assert set(result) == {1, 2, 5, 8}

    def test_neighbor_adjacent_fibonacci(self):
        v = Fibonacci(hi=20)
        nb = v.neighbor(5, {})
        assert nb in {3, 8}

    def test_no_values_raises(self):
        with pytest.raises(ValueError):
            Fibonacci(lo=4, hi=4)

    def test_invalid_neighbor_falls_back_to_sample(self, monkeypatch):
        v = Fibonacci(hi=20)
        monkeypatch.setattr(random, "choice", lambda seq: seq[-1])
        assert v.neighbor(4, {}) == v._values[-1]


class TestMathHelpers:
    def test_sieve_small_limit(self):
        assert _sieve(1) == []

    def test_fibonacci_range_can_include_zero(self):
        assert _fibonacci_in_range(0, 3) == [0, 1, 1, 2, 3]

    def test_powers_of_two_range(self):
        assert _powers_of_two_in_range(3, 20) == [4, 8, 16]


# ---------------------------------------------------------------------------
# Engineering spaces
# ---------------------------------------------------------------------------


class TestACIRebar:
    def setup_method(self):
        self.var = ACIRebar(d_expr=0.55, cc_expr=40.0, fc=30.0, fy=420.0)

    def test_sample_not_none(self):
        code = self.var.sample({})
        assert code is not None

    def test_decode_returns_diameter_and_count(self):
        code = self.var.sample({})
        dia, n = self.var.decode(code)
        assert dia > 0
        assert 4 <= n <= 41

    def test_decode_none_raises(self):
        with pytest.raises(ValueError):
            self.var.decode(None)

    def test_filter_removes_invalid(self):
        codes = self.var._valid_codes({})
        filtered = self.var.filter(codes + [-1, 999999], {})
        assert -1 not in filtered
        assert 999999 not in filtered
        assert all(c in codes for c in filtered)

    def test_neighbor_stays_valid(self):
        code = self.var.sample({})
        valid = set(self.var._valid_codes({}))
        nb = self.var.neighbor(code, {})
        assert nb in valid

    def test_dependent_d(self):
        var = ACIRebar(
            d_expr=lambda ctx: ctx["d"],
            cc_expr=40.0,
            fc=30.0,
            fy=420.0,
        )
        code = var.sample({"d": 0.55})
        assert code is not None

    def test_describe(self):
        code = self.var.sample({})
        desc = self.var.describe(code)
        assert "bars" in desc
        assert "mm" in desc

    def test_fc_fy_properties_with_callables(self):
        var = ACIRebar(
            d_expr=0.55,
            cc_expr=40.0,
            fc=lambda ctx: 30.0,
            fy=lambda ctx: 420.0,
        )
        assert var.fc is None
        assert var.fy is None

    def test_sample_returns_none_when_geometry_is_infeasible(self):
        var = ACIRebar(d_expr=0.02, cc_expr=200.0, fc=30.0, fy=420.0)
        assert var.sample({}) is None

    def test_neighbor_returns_same_value_when_invalid(self):
        assert self.var.neighbor(-1, {}) == -1

    def test_neighbor_returns_same_value_when_no_adjacent_valid_codes(self, monkeypatch):
        code = self.var.sample({})
        monkeypatch.setattr(self.var, "_valid_codes", lambda ctx: [code])
        assert self.var.neighbor(code, {}) == code

    def test_aci_limits_high_fc_uses_minimum_beta1(self):
        beta1, *_ = _aci_limits(56.0, 420.0)
        assert beta1 == pytest.approx(0.65)


class TestACIDoubleRebar:
    def test_sample_not_none(self):
        var = ACIDoubleRebar(d1_expr=0.55, d2_expr=0.48, cc_expr=40.0)
        code = var.sample({})
        assert code is not None

    def test_decode(self):
        var = ACIDoubleRebar(d1_expr=0.55, d2_expr=0.48, cc_expr=40.0)
        code = var.sample({})
        if code is not None:
            dia, n = var.decode(code)
            assert dia > 0 and n >= 4

    def test_neighbor_returns_same_value_when_invalid(self):
        var = ACIDoubleRebar(d1_expr=0.55, d2_expr=0.48, cc_expr=40.0)
        assert var.neighbor(-1, {}) == -1

    def test_filter_removes_invalid_candidates(self):
        var = ACIDoubleRebar(d1_expr=0.55, d2_expr=0.48, cc_expr=40.0)
        filtered = var.filter([-1, 999999], {})
        assert filtered == []

    def test_neighbor_stays_valid(self):
        var = ACIDoubleRebar(d1_expr=0.55, d2_expr=0.48, cc_expr=40.0)
        code = var.sample({})
        valid = set(var._valid_codes({}))
        nb = var.neighbor(code, {})
        assert nb in valid

    def test_describe(self):
        var = ACIDoubleRebar(d1_expr=0.55, d2_expr=0.48, cc_expr=40.0)
        code = var.sample({})
        assert "double row" in var.describe(code)

    def test_neighbor_returns_same_value_when_no_adjacent_valid_codes(self, monkeypatch):
        var = ACIDoubleRebar(d1_expr=0.55, d2_expr=0.48, cc_expr=40.0)
        code = var.sample({})
        monkeypatch.setattr(var, "_valid_codes", lambda ctx: [code])
        assert var.neighbor(code, {}) == code


class TestSteelSection:
    def test_sample_valid_index(self):
        var = SteelSection(series="IPE")
        idx = var.sample({})
        assert idx in var._indices

    def test_decode_returns_properties(self):
        var = SteelSection(series="IPE")
        idx = var.sample({})
        sec = var.decode(idx)
        assert sec.series == "IPE"
        assert sec.Iy_cm4 > 0
        assert sec.mass_kg_m > 0

    def test_filter(self):
        var = SteelSection(series="IPE")
        result = var.filter(var._indices, {})
        assert set(result) == set(var._indices)

    def test_filter_removes_out_of_range(self):
        var = SteelSection(series="IPE")
        result = var.filter([9999, var._indices[0]], {})
        assert result == [var._indices[0]]

    def test_series_filter(self):
        var = SteelSection(series=["HEA", "HEB"])
        for idx in var._indices:
            sec = var.decode(idx)
            assert sec.series in ("HEA", "HEB")

    def test_invalid_series_raises(self):
        with pytest.raises(ValueError):
            SteelSection(series="NONEXISTENT")

    def test_neighbor_valid(self):
        var = SteelSection(series="IPE")
        idx = var.sample({})
        nb = var.neighbor(idx, {})
        assert nb in var._indices

    def test_describe(self):
        var = SteelSection(series="IPE")
        idx = var.sample({})
        desc = var.describe(idx)
        assert "IPE" in desc

    def test_neighbor_invalid_index_falls_back_to_sample(self, monkeypatch):
        var = SteelSection(series="IPE")
        monkeypatch.setattr(random, "choice", lambda seq: seq[0])
        assert var.neighbor(999, {}) == 0

    def test_sections_property_returns_copy(self):
        var = SteelSection(series="IPE")
        sections = var.sections
        sections.pop()
        assert len(sections) + 1 == len(var.sections)

    def test_custom_catalogue_list(self):
        catalogue = [
            SectionProperties("X 1", "X", 100, 50, 5, 5, 10, 20, 4, 5, 1, 7),
            SectionProperties("X 2", "X", 120, 60, 6, 6, 12, 30, 5, 6, 2, 8),
        ]
        var = SteelSection(series="X", catalogue=catalogue)
        assert len(var.sections) == 2

    def test_custom_catalogue_json_and_csv(self, tmp_path):
        rows = [
            {
                "name": "X 1",
                "series": "X",
                "h_mm": 100,
                "b_mm": 50,
                "tf_mm": 5,
                "tw_mm": 5,
                "A_cm2": 10,
                "Iy_cm4": 20,
                "Wy_cm3": 4,
                "Iz_cm4": 5,
                "Wz_cm3": 1,
                "mass_kg_m": 7,
            }
        ]
        json_path = tmp_path / "sections.json"
        csv_path = tmp_path / "sections.csv"
        bad_path = tmp_path / "sections.txt"
        json_path.write_text(json.dumps(rows))
        csv_path.write_text(
            "name,series,h_mm,b_mm,tf_mm,tw_mm,A_cm2,Iy_cm4,Wy_cm3,Iz_cm4,Wz_cm3,mass_kg_m\n"
            "X 1,X,100,50,5,5,10,20,4,5,1,7\n"
        )
        bad_path.write_text("invalid")

        assert _load_catalogue_from_file(json_path)[0].name == "X 1"
        assert _load_catalogue_from_file(csv_path)[0].series == "X"
        with pytest.raises(ValueError):
            _load_catalogue_from_file(bad_path)

    def test_constructor_loads_catalogue_from_file(self, tmp_path):
        rows = [
            {
                "name": "X 1",
                "series": "X",
                "h_mm": 100,
                "b_mm": 50,
                "tf_mm": 5,
                "tw_mm": 5,
                "A_cm2": 10,
                "Iy_cm4": 20,
                "Wy_cm3": 4,
                "Iz_cm4": 5,
                "Wz_cm3": 1,
                "mass_kg_m": 7,
            }
        ]
        json_path = tmp_path / "sections.json"
        json_path.write_text(json.dumps(rows))

        var = SteelSection(series="X", catalogue=str(json_path))

        assert len(var.sections) == 1
        assert var.decode(0).name == "X 1"


class TestConcreteGrade:
    def test_sample_in_range(self):
        var = ConcreteGrade(min_grade="C25/30", max_grade="C50/60")
        for _ in range(30):
            idx = var.sample({})
            g = var.decode(idx)
            assert 25 <= g.fck_MPa <= 50

    def test_filter(self):
        var = ConcreteGrade()
        result = var.filter(var._indices, {})
        assert set(result) == set(var._indices)

    def test_filter_removes_out_of_range(self):
        var = ConcreteGrade()
        result = var.filter([-1, 999, var._indices[0]], {})
        assert result == [var._indices[0]]

    def test_neighbor_valid(self):
        var = ConcreteGrade()
        idx = var.sample({})
        nb = var.neighbor(idx, {})
        assert nb in var._indices

    def test_grade_properties(self):
        var = ConcreteGrade()
        for idx in var._indices:
            g = var.decode(idx)
            assert g.fcm_MPa == g.fck_MPa + 8
            assert g.Ecm_GPa > 0

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            ConcreteGrade(min_grade="C50/60", max_grade="C25/30")

    def test_describe(self):
        var = ConcreteGrade()
        assert "fck=" in var.describe(var.sample({}))


class TestSoilSPT:
    def test_sample_valid(self):
        var = SoilSPT()
        idx = var.sample({})
        profile = var.decode(idx)
        assert profile.vs30_mps > 0

    def test_site_class_filter(self):
        var = SoilSPT(site_classes=["ZC", "ZD"])
        for idx in var._indices:
            p = var.decode(idx)
            assert p.site_class in ("ZC", "ZD")

    def test_filter_removes_out_of_range(self):
        var = SoilSPT()
        result = var.filter([-1, 999, var._indices[0]], {})
        assert result == [var._indices[0]]

    def test_no_match_raises(self):
        with pytest.raises(ValueError):
            SoilSPT(site_classes=["ZA"], N_min=200, N_max=201)

    def test_describe(self):
        var = SoilSPT()
        idx = var.sample({})
        desc = var.describe(idx)
        assert "N=" in desc

    def test_neighbor_valid(self):
        var = SoilSPT()
        idx = var.sample({})
        assert var.neighbor(idx, {}) in var._indices


class TestSeismicZoneTBDY:
    def test_sample_valid(self):
        var = SeismicZoneTBDY()
        idx = var.sample({})
        z = var.decode(idx)
        assert z.SDS > 0
        assert z.SD1 > 0

    def test_filter_by_hazard_level(self):
        var = SeismicZoneTBDY(hazard_levels=["DD-2"])
        for idx in var._indices:
            z = var.decode(idx)
            assert z.hazard_level == "DD-2"

    def test_filter_by_site_class(self):
        var = SeismicZoneTBDY(site_classes=["ZC"])
        for idx in var._indices:
            z = var.decode(idx)
            assert z.site_class == "ZC"

    def test_filter_removes_out_of_range(self):
        var = SeismicZoneTBDY()
        result = var.filter([-1, 999, var._indices[0]], {})
        assert result == [var._indices[0]]

    def test_no_match_raises(self):
        with pytest.raises(ValueError):
            SeismicZoneTBDY(hazard_levels=["DD-9"])

    def test_neighbor_ordered_by_sds(self):
        var = SeismicZoneTBDY(hazard_levels=["DD-2"], site_classes=["ZC"])
        idx = var.sample({})
        nb = var.neighbor(idx, {})
        assert nb in var._indices

    def test_describe(self):
        var = SeismicZoneTBDY()
        idx = var.sample({})
        desc = var.describe(idx)
        assert "SDS=" in desc

    def test_neighbor_invalid_value_falls_back_to_first_ordered(self, monkeypatch):
        var = SeismicZoneTBDY(hazard_levels=["DD-2"], site_classes=["ZC"])
        monkeypatch.setattr(random, "choice", lambda seq: seq[0])
        ordered = sorted(var._indices, key=lambda i: var._zones[i].SDS)
        assert var.neighbor(999, {}) == ordered[0]


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------


class TestParetoDominance:
    def test_dominates_basic(self):
        from harmonix.pareto import dominates

        assert dominates((1.0, 2.0), (2.0, 3.0))
        assert not dominates((2.0, 3.0), (1.0, 2.0))

    def test_dominates_equal_not_dominate(self):
        from harmonix.pareto import dominates

        assert not dominates((1.0, 2.0), (1.0, 2.0))

    def test_dominates_partial_better(self):
        from harmonix.pareto import dominates

        assert dominates((1.0, 2.0), (1.0, 3.0))
        assert not dominates((1.0, 3.0), (1.0, 2.0))

    def test_different_lengths_raises(self):
        from harmonix.pareto import dominates

        with pytest.raises(ValueError):
            dominates((1.0,), (1.0, 2.0))
