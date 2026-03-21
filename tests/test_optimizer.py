"""
tests/test_optimizer.py
=======================
Unit and integration tests for Minimization, Maximization, MultiObjective,
HarmonyMemory, and checkpoint/resume.
"""

import json
import math
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from harmonix.variables import Continuous, Discrete, Integer
from harmonix.space import DesignSpace
from harmonix.optimizer import (
    HarmonyMemory,
    Minimization,
    Maximization,
    MultiObjective,
    OptimizationResult,
)
from harmonix.pareto import ParetoResult


# ---------------------------------------------------------------------------
# HarmonyMemory
# ---------------------------------------------------------------------------

class TestHarmonyMemory:
    def _make_mem(self, mode="min"):
        mem = HarmonyMemory(size=5, mode=mode)
        for i in range(5):
            mem.add({"x": float(i)}, fitness=float(i), penalty=0.0)
        return mem

    def test_best_min(self):
        mem = self._make_mem("min")
        h, f, p = mem.best()
        assert f == 0.0

    def test_best_max(self):
        mem = self._make_mem("max")
        h, f, p = mem.best()
        assert f == 4.0

    def test_worst_min(self):
        mem = self._make_mem("min")
        idx = mem.worst_index()
        assert mem._fitness[idx] == 4.0

    def test_feasible_replaces_infeasible(self):
        mem = HarmonyMemory(size=3, mode="min")
        for i in range(3):
            mem.add({"x": float(i)}, fitness=float(i), penalty=1.0)
        replaced = mem.try_replace_worst({"x": 99.0}, fitness=99.0, penalty=0.0)
        assert replaced
        assert any(p <= 0 for p in mem._penalty)

    def test_infeasible_does_not_replace_feasible(self):
        mem = HarmonyMemory(size=3, mode="min")
        for i in range(3):
            mem.add({"x": float(i)}, fitness=float(i), penalty=0.0)
        replaced = mem.try_replace_worst({"x": 99.0}, fitness=99.0, penalty=5.0)
        assert not replaced

    def test_better_infeasible_replaces_worse_infeasible(self):
        mem = HarmonyMemory(size=3, mode="min")
        for i in range(3):
            mem.add({"x": float(i)}, fitness=float(i), penalty=float(i + 1))
        # penalty=0.5 is less than worst penalty (3.0)
        replaced = mem.try_replace_worst({"x": 0.0}, fitness=0.0, penalty=0.5)
        assert replaced

    def test_serialisation_roundtrip(self):
        mem = self._make_mem()
        data = mem.to_dict()
        mem2 = HarmonyMemory.from_dict(data)
        h1, f1, p1 = mem.best()
        h2, f2, p2 = mem2.best()
        assert f1 == f2


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------

class TestOptimizationResult:
    def test_repr_contains_fitness(self):
        r = OptimizationResult(
            best_harmony={"x": 1.0},
            best_fitness=3.14,
            best_penalty=0.0,
            iterations=100,
            elapsed_seconds=0.5,
        )
        assert "3.14" in repr(r)


# ---------------------------------------------------------------------------
# Minimization
# ---------------------------------------------------------------------------

class TestMinimization:
    def _sphere_space(self, n=2):
        space = DesignSpace()
        for i in range(n):
            space.add(f"x{i}", Continuous(-5.0, 5.0))
        return space

    def _sphere_obj(self, h):
        return sum(v ** 2 for v in h.values()), 0.0

    def test_returns_result(self):
        space = self._sphere_space()
        result = Minimization(space, self._sphere_obj).optimize(
            memory_size=10, max_iter=100
        )
        assert isinstance(result, OptimizationResult)

    def test_fitness_decreases_over_time(self):
        space = self._sphere_space()
        result = Minimization(space, self._sphere_obj).optimize(
            memory_size=10, max_iter=500
        )
        # Last recorded fitness should be <= first recorded fitness
        assert result.history[-1][0] <= result.history[0][0] + 1e-6

    def test_best_harmony_keys_match_space(self):
        space = self._sphere_space(3)
        result = Minimization(space, self._sphere_obj).optimize(
            memory_size=10, max_iter=50
        )
        assert set(result.best_harmony.keys()) == {"x0", "x1", "x2"}

    def test_feasible_solution_found(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 10.0))

        def obj(h):
            penalty = max(0.0, h["x"] - 5.0)
            return h["x"] ** 2, penalty

        result = Minimization(space, obj).optimize(memory_size=10, max_iter=300)
        assert result.best_penalty <= 0

    def test_callback_called(self):
        space = self._sphere_space()
        calls = []
        def cb(it, res):
            calls.append(it)
            if it >= 5:
                raise StopIteration

        Minimization(space, self._sphere_obj).optimize(
            memory_size=5, max_iter=1000, callback=cb
        )
        assert len(calls) == 5

    def test_callback_receives_correct_fitness_sign(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        fitness_values = []
        def cb(it, res):
            fitness_values.append(res.best_fitness)
            if it >= 3:
                raise StopIteration

        Minimization(space, lambda h: (h["x"] ** 2, 0.0)).optimize(
            memory_size=5, max_iter=1000, callback=cb
        )
        assert all(f >= 0 for f in fitness_values)

    def test_history_length(self):
        space = self._sphere_space()
        result = Minimization(space, self._sphere_obj).optimize(
            memory_size=5, max_iter=50
        )
        assert len(result.history) == 50

    def test_checkpoint_resume(self):
        space = self._sphere_space()
        import tempfile
        import os
        fd, fname = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(fname)          # remove so optimizer creates it fresh
        ckpt = Path(fname)

        try:
            # Run 50 iterations, save checkpoint
            opt1 = Minimization(space, self._sphere_obj)
            opt1.optimize(
                memory_size=10, max_iter=50,
                checkpoint_path=ckpt, checkpoint_every=50,
            )
            assert ckpt.exists()
            data = json.loads(ckpt.read_text())
            assert data["iteration"] == 50

            # Resume and run 50 more
            opt2 = Minimization(space, self._sphere_obj)
            result = opt2.optimize(
                memory_size=10, max_iter=100,
                checkpoint_path=ckpt, checkpoint_every=100,
            )
            assert result.iterations == 50  # only the resumed portion
        finally:
            ckpt.unlink(missing_ok=True)

    def test_known_minimum_sphere(self):
        """Sphere function minimum is 0 at the origin."""
        random.seed(0)
        space = DesignSpace()
        for i in range(3):
            space.add(f"x{i}", Continuous(-5.0, 5.0))

        result = Minimization(space, self._sphere_obj).optimize(
            memory_size=20, hmcr=0.85, par=0.35, max_iter=3000
        )
        assert result.best_fitness < 0.5, (
            f"Expected fitness < 0.5, got {result.best_fitness}"
        )

    def test_discrete_variable(self):
        space = DesignSpace()
        space.add("n", Discrete(0.0, 1.0, 10.0))

        def obj(h):
            return abs(h["n"] - 7.0), 0.0

        result = Minimization(space, obj).optimize(memory_size=10, max_iter=200)
        assert result.best_harmony["n"] == 7.0


# ---------------------------------------------------------------------------
# Maximization
# ---------------------------------------------------------------------------

class TestMaximization:
    def test_finds_maximum(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 10.0))
        result = Maximization(space, lambda h: (h["x"], 0.0)).optimize(
            memory_size=10, max_iter=500
        )
        assert result.best_fitness > 8.0

    def test_best_fitness_positive(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 10.0))
        result = Maximization(space, lambda h: (h["x"], 0.0)).optimize(
            memory_size=5, max_iter=100
        )
        assert result.best_fitness > 0

    def test_history_sign_correct(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 10.0))
        result = Maximization(space, lambda h: (h["x"], 0.0)).optimize(
            memory_size=5, max_iter=50
        )
        assert all(f >= 0 for f, _ in result.history)

    def test_callback_receives_positive_fitness(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 10.0))
        cb_values = []
        def cb(it, res):
            cb_values.append(res.best_fitness)
            if it >= 5:
                raise StopIteration

        Maximization(space, lambda h: (h["x"], 0.0)).optimize(
            memory_size=5, max_iter=1000, callback=cb
        )
        assert all(v >= 0 for v in cb_values), (
            f"Callback received negated values: {cb_values}"
        )


# ---------------------------------------------------------------------------
# MultiObjective
# ---------------------------------------------------------------------------

class TestMultiObjective:
    def _zdt1_space(self, n=3):
        space = DesignSpace()
        space.add("x1", Continuous(0.0, 1.0))
        for i in range(2, n + 1):
            space.add(f"x{i}", Continuous(0.0, 1.0))
        return space

    def _zdt1_obj(self, h, n=3):
        x1   = h["x1"]
        rest = [h[f"x{i}"] for i in range(2, n + 1)]
        g    = 1.0 + 9.0 * sum(rest) / (n - 1)
        f1   = x1
        f2   = g * (1.0 - math.sqrt(x1 / g))
        return (f1, f2), 0.0

    def test_returns_pareto_result(self):
        space = self._zdt1_space()
        result = MultiObjective(space, self._zdt1_obj).optimize(
            memory_size=10, max_iter=100, archive_size=20
        )
        assert isinstance(result, ParetoResult)

    def test_front_non_empty(self):
        space = self._zdt1_space()
        result = MultiObjective(space, self._zdt1_obj).optimize(
            memory_size=10, max_iter=200, archive_size=30
        )
        assert len(result.front) > 0

    def test_front_solutions_are_non_dominated(self):
        from harmonix.pareto import dominates
        space = self._zdt1_space()
        result = MultiObjective(space, self._zdt1_obj).optimize(
            memory_size=10, max_iter=300, archive_size=30
        )
        objectives = [e.objectives for e in result.front]
        for i, a in enumerate(objectives):
            for j, b in enumerate(objectives):
                if i != j:
                    assert not dominates(a, b) or not dominates(b, a), (
                        f"Solution {i} and {j} should not both dominate each other"
                    )

    def test_archive_history_length(self):
        space = self._zdt1_space()
        result = MultiObjective(space, self._zdt1_obj).optimize(
            memory_size=5, max_iter=50, archive_size=20
        )
        assert len(result.archive_history) == 50

    def test_callback_called(self):
        space = self._zdt1_space()
        calls = []
        def cb(it, res):
            calls.append(it)
            if it >= 3:
                raise StopIteration

        MultiObjective(space, self._zdt1_obj).optimize(
            memory_size=5, max_iter=1000, callback=cb
        )
        assert len(calls) == 3

    def test_checkpoint_resume(self):
        space = self._zdt1_space()
        import tempfile
        import os
        fd, fname = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(fname)
        ckpt = Path(fname)

        try:
            opt1 = MultiObjective(space, self._zdt1_obj)
            opt1.optimize(
                memory_size=10, max_iter=50,
                checkpoint_path=ckpt, checkpoint_every=50,
            )
            assert ckpt.exists()

            opt2 = MultiObjective(space, self._zdt1_obj)
            result = opt2.optimize(
                memory_size=10, max_iter=100,
                checkpoint_path=ckpt, checkpoint_every=100,
            )
            assert result.iterations == 50
        finally:
            ckpt.unlink(missing_ok=True)

    def test_approximates_pareto_front_zdt1(self):
        """ZDT1 true front: f2 = 1 - sqrt(f1). Mean error should be small."""
        random.seed(1)
        space = self._zdt1_space(n=3)
        result = MultiObjective(space, self._zdt1_obj).optimize(
            memory_size=20, max_iter=2000, archive_size=50
        )
        errors = [
            abs(e.objectives[1] - (1.0 - math.sqrt(max(0.0, e.objectives[0]))))
            for e in result.front
        ]
        mean_err = sum(errors) / len(errors) if errors else 999
        assert mean_err < 0.05, f"Mean ZDT1 front error too large: {mean_err:.4f}"
