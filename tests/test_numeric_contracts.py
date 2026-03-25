"""
tests/test_numeric_contracts.py
==============================
Numerical edge-behavior and optimize-parameter contract tests.

These tests document current behavior around NaN/inf propagation and
parameter guards that already exist in the optimizer stack.
"""

import math

import pytest

from harmonix.optimizer import Minimization, MultiObjective
from harmonix.space import DesignSpace
from harmonix.variables import Continuous


def _space() -> DesignSpace:
    s = DesignSpace()
    s.add("x", Continuous(0.0, 1.0))
    return s


class TestNumericEdgeBehavior:
    def test_minimization_nan_fitness_propagates_to_result(self):
        result = Minimization(_space(), lambda h: (math.nan, 0.0)).optimize(memory_size=4, max_iter=5)
        assert math.isnan(result.best_fitness)

    def test_minimization_inf_fitness_propagates_to_result(self):
        result = Minimization(_space(), lambda h: (math.inf, 0.0)).optimize(memory_size=4, max_iter=5)
        assert math.isinf(result.best_fitness)
        assert result.best_fitness > 0

    def test_multiobjective_nan_component_propagates(self):
        result = MultiObjective(_space(), lambda h: ((math.nan, 1.0), 0.0)).optimize(memory_size=4, max_iter=5)
        assert any(math.isnan(entry.objectives[0]) for entry in result.front)

    def test_multiobjective_inf_component_propagates(self):
        result = MultiObjective(_space(), lambda h: ((math.inf, 1.0), 0.0)).optimize(memory_size=4, max_iter=5)
        assert any(math.isinf(entry.objectives[0]) for entry in result.front)


class TestOptimizeParameterValidation:
    def test_bw_max_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Minimization(_space(), lambda h: (h["x"], 0.0)).optimize(
                memory_size=5,
                max_iter=5,
                bw_max=0.0,
                bw_min=0.001,
            )

    def test_bw_min_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Minimization(_space(), lambda h: (h["x"], 0.0)).optimize(
                memory_size=5,
                max_iter=5,
                bw_max=0.1,
                bw_min=-0.1,
            )

    def test_bw_min_greater_than_bw_max_raises(self):
        with pytest.raises(ValueError, match="bw_min"):
            Minimization(_space(), lambda h: (h["x"], 0.0)).optimize(
                memory_size=5,
                max_iter=5,
                bw_max=0.01,
                bw_min=0.1,
            )

    def test_invalid_resume_value_raises_in_multiobjective(self):
        with pytest.raises(ValueError, match="resume"):
            MultiObjective(_space(), lambda h: ((h["x"], 1.0 - h["x"]), 0.0)).optimize(
                memory_size=5,
                max_iter=5,
                resume="bad-option",
            )

    def test_resume_missing_checkpoint_raises_in_multiobjective(self):
        with pytest.raises(FileNotFoundError):
            MultiObjective(_space(), lambda h: ((h["x"], 1.0 - h["x"]), 0.0)).optimize(
                memory_size=5,
                max_iter=5,
                checkpoint_path="definitely_missing_checkpoint_12345.json",
                resume="resume",
            )
