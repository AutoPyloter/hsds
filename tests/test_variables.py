"""
tests/test_variables.py
=======================
Unit tests for harmonix.variables — Continuous, Discrete, Integer, Categorical.

Covers:
- Continuous: sample, filter, neighbor, dependent bounds, lo>hi validation
- Discrete: sample, filter, neighbor, grid correctness, step/lo>hi validation
- Integer: sample, filter, neighbor, roundtrip via Discrete
- Categorical: sample, filter, neighbor, empty raises
- _frange helper: endpoint guarantee, floating-point safety
- _in_grid helper: tolerance behaviour
- Variable ABC: cannot be instantiated directly
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from harmonix.variables import (
    Categorical,
    Continuous,
    Discrete,
    Integer,
    Variable,
    _frange,
    _in_grid,
)

# ===========================================================================
# _frange helper
# ===========================================================================


class TestFrange:
    def test_simple_integer_steps(self):
        result = _frange(0.0, 1.0, 3.0)
        assert result == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_endpoint_always_included(self):
        """hi must always appear as the last element."""
        result = _frange(0.0, 3.0, 10.0)
        assert result[-1] == pytest.approx(10.0)

    def test_fractional_step(self):
        result = _frange(0.0, 0.5, 2.0)
        assert result == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])

    def test_single_element_lo_equals_hi(self):
        result = _frange(5.0, 1.0, 5.0)
        assert result == pytest.approx([5.0])

    def test_step_larger_than_range(self):
        """step > (hi - lo): should return [lo, hi]."""
        result = _frange(0.0, 100.0, 5.0)
        assert 0.0 in result
        assert 5.0 in result
        assert all(v <= 5.0 for v in result)

    def test_nonpositive_step_raises(self):
        with pytest.raises(ValueError, match="step"):
            _frange(0.0, 0.0, 10.0)

    def test_negative_step_raises(self):
        with pytest.raises(ValueError, match="step"):
            _frange(0.0, -1.0, 10.0)

    def test_floating_point_accumulation(self):
        """Values should be rounded — no floating-point drift."""
        result = _frange(0.0, 0.1, 1.0)
        assert len(result) == 11
        assert abs(result[-1] - 1.0) < 1e-9

    def test_negative_range(self):
        result = _frange(-3.0, 1.0, 0.0)
        assert result[0] == pytest.approx(-3.0)
        assert result[-1] == pytest.approx(0.0)


# ===========================================================================
# _in_grid helper
# ===========================================================================


class TestInGrid:
    def test_exact_match(self):
        assert _in_grid(2.0, [1.0, 2.0, 3.0])

    def test_within_tolerance(self):
        assert _in_grid(2.0 + 1e-10, [1.0, 2.0, 3.0])

    def test_outside_tolerance(self):
        assert not _in_grid(2.0 + 1e-8, [1.0, 2.0, 3.0])

    def test_empty_grid(self):
        assert not _in_grid(1.0, [])

    def test_custom_tolerance(self):
        assert _in_grid(2.05, [2.0], tol=0.1)
        assert not _in_grid(2.05, [2.0], tol=0.01)


# ===========================================================================
# Variable ABC
# ===========================================================================


class TestVariableABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Variable()

    def test_subclass_must_implement_all_methods(self):
        class Incomplete(Variable):
            def sample(self, ctx):
                return 1

            # filter ve neighbor eksik

        with pytest.raises(TypeError):
            Incomplete()

    def test_minimal_valid_subclass(self):
        class MinVar(Variable):
            def sample(self, ctx):
                return 42

            def filter(self, candidates, ctx):
                return [c for c in candidates if c == 42]

            def neighbor(self, value, ctx):
                return 42

        v = MinVar()
        assert v.sample({}) == 42
        assert v.filter([1, 42, 99], {}) == [42]
        assert v.neighbor(0, {}) == 42


# ===========================================================================
# Continuous
# ===========================================================================


class TestContinuousSample:
    def test_sample_within_bounds(self):
        v = Continuous(1.0, 5.0)
        for _ in range(100):
            s = v.sample({})
            assert 1.0 <= s <= 5.0

    def test_sample_returns_float(self):
        v = Continuous(0.0, 1.0)
        assert isinstance(v.sample({}), float)

    def test_sample_degenerate_lo_eq_hi(self):
        v = Continuous(3.0, 3.0)
        for _ in range(20):
            assert v.sample({}) == pytest.approx(3.0)

    def test_sample_dependent_lo(self):
        v = Continuous(lo=lambda ctx: ctx["a"], hi=10.0)
        for _ in range(50):
            s = v.sample({"a": 5.0})
            assert 5.0 <= s <= 10.0

    def test_sample_dependent_hi(self):
        v = Continuous(lo=0.0, hi=lambda ctx: ctx["b"])
        for _ in range(50):
            s = v.sample({"b": 3.0})
            assert 0.0 <= s <= 3.0

    def test_sample_both_dependent(self):
        v = Continuous(
            lo=lambda ctx: ctx["lo"],
            hi=lambda ctx: ctx["hi"],
        )
        for _ in range(50):
            s = v.sample({"lo": 2.0, "hi": 4.0})
            assert 2.0 <= s <= 4.0


class TestContinuousFilter:
    def test_filter_keeps_in_range(self):
        v = Continuous(2.0, 5.0)
        result = v.filter([1.0, 2.0, 3.5, 5.0, 6.0], {})
        assert result == pytest.approx([2.0, 3.5, 5.0])

    def test_filter_empty_input(self):
        v = Continuous(0.0, 1.0)
        assert v.filter([], {}) == []

    def test_filter_all_out_of_range(self):
        v = Continuous(10.0, 20.0)
        assert v.filter([1.0, 5.0, 25.0], {}) == []

    def test_filter_all_in_range(self):
        v = Continuous(0.0, 10.0)
        candidates = [0.0, 5.0, 10.0]
        assert v.filter(candidates, {}) == candidates

    def test_filter_with_dependent_bounds(self):
        v = Continuous(lo=lambda ctx: ctx["lo"], hi=lambda ctx: ctx["hi"])
        result = v.filter([0.0, 1.5, 3.0, 5.0], {"lo": 1.0, "hi": 4.0})
        assert result == pytest.approx([1.5, 3.0])

    def test_filter_boundary_values_included(self):
        v = Continuous(1.0, 4.0)
        result = v.filter([1.0, 4.0], {})
        assert 1.0 in result
        assert 4.0 in result


class TestContinuousNeighbor:
    def test_neighbor_within_bounds(self):
        v = Continuous(0.0, 10.0)
        for _ in range(100):
            nb = v.neighbor(5.0, {})
            assert 0.0 <= nb <= 10.0

    def test_neighbor_respects_bw(self):
        """Larger bw → larger average step."""
        v = Continuous(0.0, 100.0)
        random.seed(0)
        steps_large = [abs(v.neighbor(50.0, {"__bw__": 0.5}) - 50.0) for _ in range(300)]
        random.seed(0)
        steps_small = [abs(v.neighbor(50.0, {"__bw__": 0.01}) - 50.0) for _ in range(300)]
        assert sum(steps_large) > sum(steps_small) * 5

    def test_neighbor_default_bw_when_not_in_ctx(self):
        v = Continuous(0.0, 1.0)
        for _ in range(50):
            nb = v.neighbor(0.5, {})
            assert 0.0 <= nb <= 1.0

    def test_neighbor_zero_bw_stays_at_value(self):
        """bw=0 → sigma=0 → gauss(0,0)=0 → stays at value."""
        v = Continuous(0.0, 10.0)
        assert v.neighbor(5.0, {"__bw__": 0.0}) == pytest.approx(5.0)

    def test_neighbor_degenerate_lo_eq_hi(self):
        v = Continuous(3.0, 3.0)
        for _ in range(20):
            assert v.neighbor(3.0, {}) == pytest.approx(3.0)

    def test_neighbor_at_lower_bound_stays_in_range(self):
        v = Continuous(0.0, 1.0)
        for _ in range(50):
            assert 0.0 <= v.neighbor(0.0, {"__bw__": 0.5}) <= 1.0

    def test_neighbor_at_upper_bound_stays_in_range(self):
        v = Continuous(0.0, 1.0)
        for _ in range(50):
            assert 0.0 <= v.neighbor(1.0, {"__bw__": 0.5}) <= 1.0

    def test_neighbor_large_bw_clamped(self):
        """Çok büyük bw ile bile [lo, hi] dışına çıkmamalı."""
        v = Continuous(2.0, 3.0)
        for _ in range(200):
            nb = v.neighbor(2.5, {"__bw__": 100.0})
            assert 2.0 <= nb <= 3.0


class TestContinuousValidation:
    def test_lo_gt_hi_raises(self):
        with pytest.raises(ValueError, match="lo"):
            Continuous(5.0, 2.0)

    def test_lo_eq_hi_does_not_raise(self):
        v = Continuous(3.0, 3.0)
        assert isinstance(v, Continuous)

    def test_callable_lo_no_validation_at_init(self):
        """Callable bounds skip init validation — no error."""
        v = Continuous(lo=lambda ctx: 999.0, hi=1.0)
        assert isinstance(v, Continuous)

    def test_lo_gt_hi_error_contains_values(self):
        with pytest.raises(ValueError) as exc:
            Continuous(7.0, 2.0)
        assert "7.0" in str(exc.value)
        assert "2.0" in str(exc.value)


# ===========================================================================
# Discrete
# ===========================================================================


class TestDiscreteSample:
    def test_sample_on_grid(self):
        v = Discrete(0.0, 1.0, 5.0)
        grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        for _ in range(100):
            assert v.sample({}) in grid

    def test_sample_within_bounds(self):
        v = Discrete(0.0, 2.5, 10.0)
        for _ in range(50):
            s = v.sample({})
            assert 0.0 <= s <= 10.0

    def test_sample_single_element(self):
        v = Discrete(5.0, 1.0, 5.0)
        for _ in range(20):
            assert v.sample({}) == pytest.approx(5.0)

    def test_sample_returns_float(self):
        v = Discrete(0.0, 1.0, 5.0)
        assert isinstance(v.sample({}), float)

    def test_sample_with_dependent_bounds(self):
        v = Discrete(
            lo=lambda ctx: ctx["base"],
            step=1.0,
            hi=lambda ctx: ctx["base"] + 5.0,
        )
        for _ in range(30):
            s = v.sample({"base": 10.0})
            assert 10.0 <= s <= 15.0


class TestDiscreteFilter:
    def test_filter_keeps_grid_values(self):
        v = Discrete(0.0, 1.0, 4.0)
        result = v.filter([0.0, 0.5, 1.0, 2.0, 3.5, 4.0], {})
        assert set(result) == {0.0, 1.0, 2.0, 4.0}

    def test_filter_empty_input(self):
        v = Discrete(0.0, 1.0, 5.0)
        assert v.filter([], {}) == []

    def test_filter_all_off_grid(self):
        v = Discrete(0.0, 1.0, 5.0)
        assert v.filter([0.3, 1.7, 4.9], {}) == []

    def test_filter_tolerance_respected(self):
        """Floating-point yakın değerler grid üyesi sayılmalı."""
        v = Discrete(0.0, 0.1, 1.0)
        near_one = 1.0 + 5e-10  # tolerance içinde
        result = v.filter([near_one], {})
        assert len(result) == 1

    def test_filter_boundary_values(self):
        v = Discrete(2.0, 1.0, 5.0)
        result = v.filter([2.0, 5.0], {})
        assert 2.0 in result
        assert 5.0 in result


class TestDiscreteNeighbor:
    def test_neighbor_adjacent_step(self):
        v = Discrete(0.0, 1.0, 10.0)
        nb = v.neighbor(5.0, {})
        assert nb in {4.0, 6.0}

    def test_neighbor_at_lower_bound(self):
        v = Discrete(0.0, 1.0, 10.0)
        for _ in range(30):
            nb = v.neighbor(0.0, {})
            assert nb in {0.0, 1.0}

    def test_neighbor_at_upper_bound(self):
        v = Discrete(0.0, 1.0, 10.0)
        for _ in range(30):
            nb = v.neighbor(10.0, {})
            assert nb in {9.0, 10.0}

    def test_neighbor_single_element_returns_self(self):
        v = Discrete(5.0, 1.0, 5.0)
        for _ in range(20):
            assert v.neighbor(5.0, {}) == pytest.approx(5.0)

    def test_neighbor_off_grid_returns_unchanged(self):
        """Grid dışı değer gelirse aynısını döndür."""
        v = Discrete(0.0, 1.0, 5.0)
        assert v.neighbor(2.7, {}) == pytest.approx(2.7)

    def test_neighbor_stays_on_grid(self):
        v = Discrete(0.0, 0.5, 2.0)
        grid = {0.0, 0.5, 1.0, 1.5, 2.0}
        for val in grid:
            nb = v.neighbor(val, {})
            assert any(abs(nb - g) < 1e-9 for g in grid), f"neighbor({val})={nb} not on grid"


class TestDiscreteValidation:
    def test_lo_gt_hi_raises(self):
        with pytest.raises(ValueError, match="lo"):
            Discrete(10.0, 1.0, 5.0)

    def test_zero_step_raises(self):
        with pytest.raises(ValueError, match="step"):
            Discrete(0.0, 0.0, 10.0)

    def test_negative_step_raises(self):
        with pytest.raises(ValueError, match="step"):
            Discrete(0.0, -2.0, 10.0)

    def test_callable_bounds_skip_init_validation(self):
        """Callable bounds ile init'te hata olmamalı."""
        v = Discrete(
            lo=lambda ctx: ctx["x"],
            step=1.0,
            hi=lambda ctx: ctx["x"] + 5,
        )
        assert isinstance(v, Discrete)

    def test_step_error_message_mentions_step(self):
        with pytest.raises(ValueError) as exc:
            Discrete(0.0, -5.0, 10.0)
        assert "step" in str(exc.value).lower()


# ===========================================================================
# Integer
# ===========================================================================


class TestIntegerSample:
    def test_sample_is_integer(self):
        v = Integer(1, 10)
        for _ in range(50):
            s = v.sample({})
            assert isinstance(s, int)

    def test_sample_within_bounds(self):
        v = Integer(3, 8)
        for _ in range(100):
            s = v.sample({})
            assert 3 <= s <= 8

    def test_sample_lo_eq_hi(self):
        v = Integer(7, 7)
        for _ in range(20):
            assert v.sample({}) == 7

    def test_sample_covers_full_range(self):
        """Yeterli örnekte tüm değerler görünmeli."""
        v = Integer(1, 5)
        seen = {v.sample({}) for _ in range(500)}
        assert seen == {1, 2, 3, 4, 5}

    def test_sample_negative_range(self):
        v = Integer(-5, -1)
        for _ in range(50):
            s = v.sample({})
            assert -5 <= s <= -1


class TestIntegerFilter:
    def test_filter_keeps_integers_in_range(self):
        v = Integer(2, 6)
        result = v.filter([1, 2, 4, 6, 7], {})
        assert result == [2, 4, 6]

    def test_filter_removes_floats_off_grid(self):
        v = Integer(1, 5)
        result = v.filter([1, 2, 2.7, 5], {})
        # 2.7 float olarak grid'de değil
        assert all(isinstance(r, int) for r in result)

    def test_filter_empty(self):
        v = Integer(1, 5)
        assert v.filter([], {}) == []


class TestIntegerNeighbor:
    def test_neighbor_adjacent(self):
        v = Integer(1, 10)
        nb = v.neighbor(5, {})
        assert nb in {4, 6}

    def test_neighbor_is_integer(self):
        v = Integer(1, 10)
        for _ in range(50):
            nb = v.neighbor(5, {})
            assert isinstance(nb, int)

    def test_neighbor_at_lower_bound(self):
        v = Integer(1, 10)
        for _ in range(30):
            nb = v.neighbor(1, {})
            assert nb in {1, 2}

    def test_neighbor_at_upper_bound(self):
        v = Integer(1, 10)
        for _ in range(30):
            nb = v.neighbor(10, {})
            assert nb in {9, 10}

    def test_neighbor_single_value(self):
        v = Integer(5, 5)
        for _ in range(20):
            assert v.neighbor(5, {}) == 5


class TestIntegerValidation:
    def test_lo_gt_hi_raises(self):
        with pytest.raises(ValueError):
            Integer(10, 5)

    def test_lo_eq_hi_valid(self):
        v = Integer(3, 3)
        assert v.sample({}) == 3


# ===========================================================================
# Categorical
# ===========================================================================


class TestCategoricalSample:
    def test_sample_in_choices(self):
        v = Categorical(["a", "b", "c"])
        for _ in range(50):
            assert v.sample({}) in {"a", "b", "c"}

    def test_sample_single_choice(self):
        v = Categorical(["only"])
        for _ in range(20):
            assert v.sample({}) == "only"

    def test_sample_integer_choices(self):
        v = Categorical([10, 20, 30])
        for _ in range(50):
            assert v.sample({}) in {10, 20, 30}

    def test_sample_mixed_types(self):
        v = Categorical(["s235", 355, 3.14])
        for _ in range(30):
            assert v.sample({}) in {"s235", 355, 3.14}

    def test_sample_covers_all_choices(self):
        v = Categorical(["x", "y", "z"])
        seen = {v.sample({}) for _ in range(300)}
        assert seen == {"x", "y", "z"}


class TestCategoricalFilter:
    def test_filter_keeps_valid(self):
        v = Categorical(["a", "b", "c"])
        result = v.filter(["a", "d", "b", "e"], {})
        assert result == ["a", "b"]

    def test_filter_empty_input(self):
        v = Categorical(["a", "b"])
        assert v.filter([], {}) == []

    def test_filter_all_invalid(self):
        v = Categorical(["a", "b"])
        assert v.filter(["x", "y", "z"], {}) == []

    def test_filter_all_valid(self):
        v = Categorical(["a", "b", "c"])
        candidates = ["a", "b", "c"]
        assert v.filter(candidates, {}) == candidates

    def test_filter_preserves_order(self):
        v = Categorical(["z", "a", "m"])
        result = v.filter(["a", "z", "m"], {})
        assert result == ["a", "z", "m"]


class TestCategoricalNeighbor:
    def test_neighbor_different_from_value(self):
        v = Categorical(["a", "b", "c"])
        for _ in range(50):
            nb = v.neighbor("a", {})
            assert nb in {"b", "c"}

    def test_neighbor_in_choices(self):
        v = Categorical(["x", "y", "z"])
        for _ in range(50):
            assert v.neighbor("x", {}) in {"x", "y", "z"}

    def test_neighbor_single_choice_returns_self(self):
        v = Categorical(["only"])
        for _ in range(20):
            assert v.neighbor("only", {}) == "only"

    def test_neighbor_covers_other_choices(self):
        """Komşu, diğer seçeneklerin hepsine ulaşabilmeli."""
        v = Categorical(["a", "b", "c"])
        seen = {v.neighbor("a", {}) for _ in range(200)}
        assert seen == {"b", "c"}


class TestCategoricalValidation:
    def test_empty_choices_raises(self):
        with pytest.raises(ValueError):
            Categorical([])

    def test_single_element_valid(self):
        v = Categorical([42])
        assert isinstance(v, Categorical)


# ===========================================================================
# Bağımlı değişken zinciri — entegrasyon
# ===========================================================================


class TestDependentChain:
    def test_two_level_chain(self):
        """b her zaman a'dan büyük veya eşit olmalı."""
        a = Continuous(0.0, 5.0)
        b = Continuous(lo=lambda ctx: ctx["a"], hi=10.0)

        for _ in range(100):
            ctx = {}
            ctx["a"] = a.sample(ctx)
            ctx["b"] = b.sample(ctx)
            assert ctx["b"] >= ctx["a"] - 1e-9

    def test_three_level_chain(self):
        """a → b → c sıralaması korunmalı."""
        a = Continuous(0.0, 3.0)
        b = Continuous(lo=lambda ctx: ctx["a"], hi=6.0)
        c = Continuous(lo=lambda ctx: ctx["b"], hi=10.0)

        for _ in range(100):
            ctx = {}
            ctx["a"] = a.sample(ctx)
            ctx["b"] = b.sample(ctx)
            ctx["c"] = c.sample(ctx)
            assert ctx["a"] <= ctx["b"] <= ctx["c"]

    def test_filter_with_dependent_bounds(self):
        """Dependent filter doğru bağlamda çalışmalı."""
        b = Continuous(lo=lambda ctx: ctx["a"], hi=10.0)
        ctx = {"a": 5.0}
        result = b.filter([3.0, 5.0, 7.0, 11.0], ctx)
        assert result == pytest.approx([5.0, 7.0])

    def test_neighbor_with_dependent_bounds(self):
        """Dependent neighbor [a, 10] aralığında kalmalı."""
        b = Continuous(lo=lambda ctx: ctx["a"], hi=10.0)
        ctx = {"a": 6.0, "__bw__": 0.1}
        for _ in range(50):
            nb = b.neighbor(8.0, ctx)
            assert 6.0 <= nb <= 10.0

    def test_discrete_dependent_step(self):
        """Step de callable olabilmeli."""
        v = Discrete(
            lo=0.0,
            step=lambda ctx: ctx["step_size"],
            hi=10.0,
        )
        for _ in range(30):
            s = v.sample({"step_size": 2.0})
            assert s in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


# ===========================================================================
# __bw__ davranışı
# ===========================================================================


class TestBwBehaviour:
    def test_bw_key_not_leaked_to_sample(self):
        """__bw__ context içinde olsa da sample sonucu bounds içinde kalmalı."""
        v = Continuous(0.0, 1.0)
        for _ in range(50):
            s = v.sample({"__bw__": 99.0})
            assert 0.0 <= s <= 1.0

    def test_bw_ignored_by_discrete(self):
        """Discrete __bw__'yi görmezden gelmeli."""
        v = Discrete(0.0, 1.0, 10.0)
        for _ in range(50):
            nb = v.neighbor(5.0, {"__bw__": 0.0})
            assert nb in {4.0, 6.0}

    def test_bw_ignored_by_integer(self):
        """Integer __bw__'yi görmezden gelmeli."""
        v = Integer(1, 10)
        for _ in range(50):
            nb = v.neighbor(5, {"__bw__": 99.0})
            assert isinstance(nb, int)
            assert 1 <= nb <= 10

    def test_bw_ignored_by_categorical(self):
        """Categorical __bw__'yi görmezden gelmeli."""
        v = Categorical(["a", "b", "c"])
        for _ in range(30):
            nb = v.neighbor("a", {"__bw__": 99.0})
            assert nb in {"b", "c"}
