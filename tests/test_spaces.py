"""
tests/test_spaces.py
====================
Tests for harmonix.spaces — math and engineering domain spaces.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import harmonix  # triggers all space registrations
from harmonix.spaces.math import (
    NaturalNumber, WholeNumber, NegativeInt, NegativeReal,
    PositiveReal, PrimeVariable, PowerOfTwo, Fibonacci,
)
from harmonix.spaces.engineering import (
    ACIRebar, ACIDoubleRebar,
    SteelSection, ConcreteGrade,
    SoilSPT, SeismicZoneTBDY,
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


class TestNegativeReal:
    def test_all_negative(self):
        v = NegativeReal(lo=-100.0)
        for _ in range(50):
            assert v.sample({}) < 0

    def test_lo_must_be_negative(self):
        with pytest.raises(ValueError):
            NegativeReal(lo=0.0)


class TestPositiveReal:
    def test_all_positive(self):
        v = PositiveReal(hi=100.0)
        for _ in range(50):
            assert v.sample({}) > 0

    def test_hi_must_be_positive(self):
        with pytest.raises(ValueError):
            PositiveReal(hi=-1.0)


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
        assert nb in (primes[idx - 1] if idx > 0 else 11,
                      primes[idx + 1] if idx < len(primes) - 1 else 11)

    def test_no_primes_raises(self):
        with pytest.raises(ValueError):
            PrimeVariable(lo=14, hi=16)  # no primes between 14 and 16


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

    def test_no_values_raises(self):
        with pytest.raises(ValueError):
            PowerOfTwo(lo=5, hi=6)  # no powers of two in [5,6]


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
        assert all(c in codes for c in self.var.filter(codes, {}))

    def test_neighbor_stays_valid(self):
        code = self.var.sample({})
        valid = set(self.var._valid_codes({}))
        nb = self.var.neighbor(code, {})
        assert nb in valid

    def test_dependent_d(self):
        var = ACIRebar(
            d_expr  = lambda ctx: ctx["d"],
            cc_expr = 40.0,
            fc=30.0, fy=420.0,
        )
        code = var.sample({"d": 0.55})
        assert code is not None

    def test_describe(self):
        code = self.var.sample({})
        desc = self.var.describe(code)
        assert "bars" in desc
        assert "mm" in desc


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

    def test_no_match_raises(self):
        with pytest.raises(ValueError):
            SoilSPT(site_classes=["ZA"], N_min=200, N_max=201)

    def test_describe(self):
        var = SoilSPT()
        idx = var.sample({})
        desc = var.describe(idx)
        assert "N=" in desc


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

