"""
tests/test_hypothesis.py
========================
Property-based tests using Hypothesis.

These tests generate thousands of random inputs automatically to find
edge cases that hand-written tests might miss.

Covers:
- Continuous: sample always in [lo, hi], neighbor always in [lo, hi]
- Discrete: sample always on grid, neighbor always on grid
- Integer: sample always integer in [lo, hi]
- Categorical: sample always in choices
- DesignSpace: dependent bounds always respected
- HarmonyMemory: best/worst invariants
- EvaluationCache: cache never changes return value
- _compute_bw: always in [bw_min, bw_max]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from harmonix.logging import EvaluationCache
from harmonix.optimizer import HarmonyMemory, Minimization
from harmonix.space import DesignSpace
from harmonix.variables import Categorical, Continuous, Discrete, Integer

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def valid_bounds():
    """Strategy for (lo, hi) where lo <= hi."""
    return st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False).flatmap(
        lambda lo: st.tuples(
            st.just(lo),
            st.floats(min_value=lo, max_value=lo + 1e6, allow_nan=False, allow_infinity=False),
        )
    )


def valid_step_bounds():
    """Strategy for (lo, step, hi) where step > 0 and lo <= hi."""
    return st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False).flatmap(
        lambda lo: st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False).flatmap(
            lambda step: st.floats(min_value=lo, max_value=lo + step * 50, allow_nan=False, allow_infinity=False).map(
                lambda hi: (lo, step, hi)
            )
        )
    )


# ---------------------------------------------------------------------------
# Continuous
# ---------------------------------------------------------------------------


class TestContinuousProperties:
    @given(bounds=valid_bounds())
    @settings(max_examples=200)
    def test_sample_always_in_bounds(self, bounds):
        lo, hi = bounds
        v = Continuous(lo, hi)
        result = v.sample({})
        assert lo <= result <= hi, f"sample={result} not in [{lo}, {hi}]"

    @given(bounds=valid_bounds())
    @settings(max_examples=200)
    def test_neighbor_always_in_bounds(self, bounds):
        lo, hi = bounds
        v = Continuous(lo, hi)
        mid = (lo + hi) / 2
        for bw in [0.001, 0.05, 0.5]:
            result = v.neighbor(mid, {"__bw__": bw})
            assert lo <= result <= hi, f"neighbor={result} not in [{lo}, {hi}] with bw={bw}"

    @given(bounds=valid_bounds())
    @settings(max_examples=100)
    def test_filter_subset_of_candidates(self, bounds):
        lo, hi = bounds
        v = Continuous(lo, hi)
        candidates = [lo - 1, lo, (lo + hi) / 2, hi, hi + 1]
        result = v.filter(candidates, {})
        assert all(lo <= x <= hi for x in result)
        assert set(result).issubset(set(candidates))

    @given(st.floats(min_value=1e-6, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_lo_gt_hi_raises(self, delta):
        assume(delta > 1e-6)
        with pytest.raises(ValueError):
            Continuous(10.0 + delta, 10.0)


# ---------------------------------------------------------------------------
# Discrete
# ---------------------------------------------------------------------------


class TestDiscreteProperties:
    @given(params=valid_step_bounds())
    @settings(max_examples=200)
    def test_sample_on_grid(self, params):
        lo, step, hi = params
        assume(hi >= lo + step)
        v = Discrete(lo, step, hi)
        result = v.sample({})
        if result is None:
            return
        # _frange always includes hi as last element even if not on step grid.
        # So result must be either: lo + k*step for some k, OR exactly hi.
        on_step = abs((result - lo) % step) < step * 1e-5 or abs((result - lo) % step - step) < step * 1e-5
        is_endpoint = abs(result - lo) < step * 1e-5 or abs(result - hi) < step * 1e-5
        assert on_step or is_endpoint, f"sample={result} not on grid lo={lo}, step={step}, hi={hi}"

    @given(params=valid_step_bounds())
    @settings(max_examples=200)
    def test_sample_in_bounds(self, params):
        lo, step, hi = params
        assume(hi >= lo + step * 0.5)
        v = Discrete(lo, step, hi)
        result = v.sample({})
        if result is not None:
            # Allow small float tolerance
            assert lo - 1e-9 <= result <= hi + 1e-9, f"sample={result} not in [{lo}, {hi}]"

    @given(
        lo=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        step=st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_lo_gt_hi_raises(self, lo, step):
        with pytest.raises(ValueError):
            Discrete(lo + step * 2, step, lo)

    @given(
        lo=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        hi=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_negative_step_raises(self, lo, hi):
        with pytest.raises(ValueError):
            Discrete(lo, -1.0, hi)


# ---------------------------------------------------------------------------
# Integer
# ---------------------------------------------------------------------------


class TestIntegerProperties:
    @given(
        lo=st.integers(min_value=-1000, max_value=1000),
        hi=st.integers(min_value=-1000, max_value=1000),
    )
    @settings(max_examples=200)
    def test_sample_always_integer_in_bounds(self, lo, hi):
        assume(lo <= hi)
        v = Integer(lo, hi)
        result = v.sample({})
        assert isinstance(result, int)
        assert lo <= result <= hi

    @given(
        lo=st.integers(min_value=-100, max_value=100),
        hi=st.integers(min_value=-100, max_value=100),
    )
    @settings(max_examples=200)
    def test_neighbor_always_in_bounds(self, lo, hi):
        assume(lo <= hi)
        v = Integer(lo, hi)
        for val in [lo, hi, (lo + hi) // 2]:
            result = v.neighbor(val, {})
            assert isinstance(result, int)
            assert lo <= result <= hi, f"neighbor({val})={result} not in [{lo}, {hi}]"

    @given(
        lo=st.integers(min_value=-100, max_value=100),
        delta=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_lo_gt_hi_raises(self, lo, delta):
        with pytest.raises(ValueError):
            Integer(lo + delta, lo)


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------


class TestCategoricalProperties:
    @given(choices=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20, unique=True))
    @settings(max_examples=200)
    def test_sample_always_in_choices(self, choices):
        v = Categorical(choices)
        result = v.sample({})
        assert result in choices

    @given(choices=st.lists(st.text(min_size=1, max_size=10), min_size=2, max_size=20, unique=True))
    @settings(max_examples=200)
    def test_neighbor_always_in_choices(self, choices):
        v = Categorical(choices)
        for val in choices[:3]:
            result = v.neighbor(val, {})
            assert result in choices

    @given(choices=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=10, unique=True))
    @settings(max_examples=100)
    def test_filter_returns_valid_subset(self, choices):
        v = Categorical(choices)
        extra = [x + 1000 for x in choices]  # definitely not in choices
        candidates = choices[: len(choices) // 2] + extra
        result = v.filter(candidates, {})
        assert all(r in choices for r in result)


# ---------------------------------------------------------------------------
# DesignSpace dependent bounds
# ---------------------------------------------------------------------------


class TestDesignSpaceProperties:
    @given(
        a_lo=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        a_hi=st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_dependent_bounds_always_respected(self, a_lo, a_hi):
        assume(a_lo <= a_hi)
        space = DesignSpace()
        space.add("a", Continuous(a_lo, a_hi))
        space.add(
            "b",
            Continuous(
                lo=lambda ctx: ctx["a"],
                hi=lambda ctx: ctx["a"] + 1.0,
            ),
        )
        h = space.sample_harmony()
        assert h["b"] >= h["a"] - 1e-9
        assert h["b"] <= h["a"] + 1.0 + 1e-9


# ---------------------------------------------------------------------------
# HarmonyMemory invariants
# ---------------------------------------------------------------------------


class TestHarmonyMemoryProperties:
    @given(
        values=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=100)
    def test_best_fitness_le_worst(self, values):
        mem = HarmonyMemory(size=len(values), mode="min")
        for v in values:
            mem.add({"x": v}, v, 0.0)
        _, best_f, _ = mem.best()
        worst_idx = mem.worst_index()
        worst_f = mem._fitness[worst_idx]
        assert best_f <= worst_f

    @given(
        values=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=100)
    def test_best_is_minimum(self, values):
        mem = HarmonyMemory(size=len(values), mode="min")
        for v in values:
            mem.add({"x": v}, v, 0.0)
        _, best_f, _ = mem.best()
        assert best_f == min(values)


# ---------------------------------------------------------------------------
# EvaluationCache: never changes result
# ---------------------------------------------------------------------------


class TestEvaluationCacheProperties:
    @given(
        x=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        y=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_cache_returns_same_result(self, x, y):
        def obj(h):
            return h["x"] ** 2 + h["y"] ** 2, 0.0

        cache = EvaluationCache(obj, maxsize=512)
        h = {"x": x, "y": y}
        r1 = cache(h)
        r2 = cache(h)
        assert r1 == r2

    @given(
        x=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_cache_hit_count_correct(self, x):
        calls = [0]

        def obj(h):
            calls[0] += 1
            return h["x"] ** 2, 0.0

        cache = EvaluationCache(obj, maxsize=512)
        h = {"x": x}
        cache(h)
        cache(h)
        cache(h)
        assert calls[0] == 1
        assert cache.hits == 2


# ---------------------------------------------------------------------------
# _compute_bw: always in [bw_min, bw_max]
# ---------------------------------------------------------------------------


class TestComputeBwProperties:
    @given(
        bw_max=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
        bw_min=st.floats(min_value=0.0001, max_value=0.5, allow_nan=False, allow_infinity=False),
        t=st.integers(min_value=0, max_value=9999),
        T=st.integers(min_value=2, max_value=10000),
    )
    @settings(max_examples=300)
    def test_bw_always_in_range(self, bw_max, bw_min, t, T):
        assume(bw_min <= bw_max)
        assume(t < T)

        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        opt = Minimization(space, lambda h: (h["x"], 0.0))

        bw = opt._compute_bw(t, T, bw_max, bw_min)
        assert bw_min * 0.999 <= bw <= bw_max * 1.001, f"bw={bw} not in [{bw_min}, {bw_max}] at t={t}, T={T}"
