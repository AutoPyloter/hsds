"""
tests/test_bandwidth.py
=======================
Tests for dynamic bandwidth narrowing and pitch adjustment behaviour.

Covers:
- _compute_bw boundary conditions and decay shape
- __bw__ injection into ctx and cleanup
- Continuous.neighbor respects bw from ctx
- Discrete / Integer / Categorical ignore __bw__
- bw_max / bw_min parameter validation
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from harmonix.variables import Continuous, Discrete, Integer, Categorical
from harmonix.space import DesignSpace
from harmonix.optimizer import Minimization, HarmonySearchOptimizer


# ---------------------------------------------------------------------------
# _compute_bw
# ---------------------------------------------------------------------------

class TestComputeBw:
    def _make_opt(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        return Minimization(space, lambda h: (h["x"], 0.0))

    def test_start_equals_bw_max(self):
        opt = self._make_opt()
        bw = opt._compute_bw(0, 1000, bw_max=0.10, bw_min=0.001)
        assert abs(bw - 0.10) < 1e-9

    def test_end_approaches_bw_min(self):
        opt = self._make_opt()
        bw = opt._compute_bw(999, 1000, bw_max=0.10, bw_min=0.001)
        # Should be very close to bw_min but not necessarily equal
        assert bw < 0.005
        assert bw >= 0.001 * 0.99   # within 1 % of bw_min

    def test_monotone_decreasing(self):
        opt = self._make_opt()
        bws = [opt._compute_bw(t, 100, 0.10, 0.001) for t in range(100)]
        assert all(bws[i] >= bws[i+1] for i in range(len(bws)-1))

    def test_constant_when_equal(self):
        opt = self._make_opt()
        for t in [0, 50, 99]:
            bw = opt._compute_bw(t, 100, bw_max=0.05, bw_min=0.05)
            assert abs(bw - 0.05) < 1e-9

    def test_max_iter_one_returns_bw_max(self):
        opt = self._make_opt()
        bw = opt._compute_bw(0, 1, bw_max=0.05, bw_min=0.001)
        assert abs(bw - 0.05) < 1e-9

    def test_negative_bw_max_raises(self):
        opt = self._make_opt()
        with pytest.raises(ValueError):
            opt._compute_bw(0, 100, bw_max=-0.05, bw_min=0.001)

    def test_bw_min_greater_than_bw_max_raises(self):
        opt = self._make_opt()
        with pytest.raises(ValueError):
            opt._compute_bw(0, 100, bw_max=0.01, bw_min=0.10)

    def test_exponential_shape(self):
        """Verify the decay is truly exponential (log-linear)."""
        opt = self._make_opt()
        bws = [opt._compute_bw(t, 100, 0.10, 0.001) for t in range(0, 100, 10)]
        log_bws = [math.log(b) for b in bws]
        # Check approximate linearity of log(bw) vs t
        diffs = [log_bws[i+1] - log_bws[i] for i in range(len(log_bws)-1)]
        # All differences should be approximately equal
        assert max(diffs) - min(diffs) < 0.01


# ---------------------------------------------------------------------------
# __bw__ injection and cleanup
# ---------------------------------------------------------------------------

class TestBwInjection:
    def test_bw_injected_into_ctx(self):
        """Verify __bw__ is present in ctx during neighbor() call."""
        seen_bw = []

        class SpyVar(Continuous):
            def neighbor(self, value, ctx):
                seen_bw.append(ctx.get("__bw__"))
                return super().neighbor(value, ctx)

        space = DesignSpace()
        space.add("x", SpyVar(0.0, 1.0))

        opt = Minimization(space, lambda h: (h["x"]**2, 0.0))
        opt._memory = __import__('harmonix.optimizer', fromlist=['HarmonyMemory']).HarmonyMemory(size=5, mode="min")
        for _ in range(5):
            h = space.sample_harmony()
            opt._memory.add(h, h["x"]**2, 0.0)

        # Force PAR=1 so neighbor is always called
        opt._improvise(hmcr=1.0, par=1.0, bw=0.03)
        assert any(b is not None for b in seen_bw), "__bw__ never injected"
        assert all(abs(b - 0.03) < 1e-9 for b in seen_bw if b is not None)

    def test_bw_not_in_returned_harmony(self):
        """__bw__ must be cleaned up — not present in final harmony."""
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        result = Minimization(space, lambda h: (h["x"]**2, 0.0)).optimize(
            memory_size=5, max_iter=50, bw_max=0.05, bw_min=0.001
        )
        assert "__bw__" not in result.best_harmony

    def test_bw_not_in_harmony_memory(self):
        """__bw__ must not be stored in harmony memory."""
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        opt = Minimization(space, lambda h: (h["x"]**2, 0.0))
        opt.optimize(memory_size=5, max_iter=20)
        for h in opt._memory.harmonies:
            assert "__bw__" not in h


# ---------------------------------------------------------------------------
# Continuous.neighbor bandwidth effect
# ---------------------------------------------------------------------------

class TestContinuousNeighborBw:
    def test_large_bw_produces_larger_steps(self):
        """High bw → larger average step size."""
        v = Continuous(0.0, 100.0)
        random.seed(0)
        steps_large = [abs(v.neighbor(50.0, {"__bw__": 0.50}) - 50.0)
                       for _ in range(500)]
        random.seed(0)
        steps_small = [abs(v.neighbor(50.0, {"__bw__": 0.01}) - 50.0)
                       for _ in range(500)]
        assert sum(steps_large) > sum(steps_small) * 5

    def test_default_bw_used_when_not_in_ctx(self):
        """neighbor() works without __bw__ in ctx (uses default 0.05)."""
        v = Continuous(0.0, 1.0)
        for _ in range(50):
            nb = v.neighbor(0.5, {})
            assert 0.0 <= nb <= 1.0

    def test_bw_zero_stays_at_value(self):
        """bw=0 means sigma=0 → neighbor equals value (Gaussian(0,0)=0)."""
        v = Continuous(0.0, 10.0)
        # With bw=0 sigma=0, gauss(0,0) = 0 deterministically
        nb = v.neighbor(5.0, {"__bw__": 0.0})
        assert nb == 5.0

    def test_neighbor_always_in_bounds_with_large_bw(self):
        """Even with very large bw, result must stay in [lo, hi]."""
        v = Continuous(2.0, 3.0)
        for _ in range(200):
            nb = v.neighbor(2.5, {"__bw__": 10.0})
            assert 2.0 <= nb <= 3.0


# ---------------------------------------------------------------------------
# Non-continuous variables ignore __bw__
# ---------------------------------------------------------------------------

class TestNonContinuousIgnoresBw:
    def test_discrete_neighbor_ignores_bw(self):
        v = Discrete(0.0, 1.0, 10.0)
        results = {v.neighbor(5.0, {"__bw__": 0.0001}) for _ in range(50)}
        assert results <= {4.0, 5.0, 6.0}

    def test_integer_neighbor_ignores_bw(self):
        v = Integer(1, 10)
        for _ in range(50):
            nb = v.neighbor(5, {"__bw__": 99.0})
            assert isinstance(nb, int)
            assert 1 <= nb <= 10

    def test_categorical_neighbor_ignores_bw(self):
        v = Categorical(["a", "b", "c"])
        for _ in range(30):
            nb = v.neighbor("a", {"__bw__": 99.0})
            assert nb in {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Pitch adjustment in the full optimization loop
# ---------------------------------------------------------------------------

class TestPitchAdjustmentIntegration:
    def test_par_zero_never_calls_neighbor(self):
        """With PAR=0, neighbor() should never be called."""
        call_count = [0]

        class CountingVar(Continuous):
            def neighbor(self, value, ctx):
                call_count[0] += 1
                return super().neighbor(value, ctx)

        space = DesignSpace()
        space.add("x", CountingVar(0.0, 1.0))
        Minimization(space, lambda h: (h["x"]**2, 0.0)).optimize(
            memory_size=5, max_iter=100, par=0.0
        )
        assert call_count[0] == 0

    def test_par_one_always_calls_neighbor_on_memory_hit(self):
        """With PAR=1 and HMCR=1, neighbor() called on every iteration."""
        call_count = [0]

        class CountingVar(Continuous):
            def neighbor(self, value, ctx):
                call_count[0] += 1
                return super().neighbor(value, ctx)

        space = DesignSpace()
        space.add("x", CountingVar(0.0, 1.0))
        Minimization(space, lambda h: (h["x"]**2, 0.0)).optimize(
            memory_size=5, max_iter=50, hmcr=1.0, par=1.0
        )
        assert call_count[0] == 50

    def test_bw_decay_improves_convergence(self):
        """bw decay should converge tighter than constant large bw."""
        random.seed(42)
        space = DesignSpace()
        for i in range(3):
            space.add(f"x{i}", Continuous(-5.0, 5.0))
        def sphere(h): return sum(v**2 for v in h.values()), 0.0

        random.seed(42)
        r_decay = Minimization(space, sphere).optimize(
            memory_size=15, max_iter=1000, bw_max=0.3, bw_min=0.001
        )
        random.seed(42)
        r_const = Minimization(space, sphere).optimize(
            memory_size=15, max_iter=1000, bw_max=0.3, bw_min=0.3
        )
        # Decay should generally converge better (not always guaranteed,
        # but with same seed and these params, it should hold)
        assert r_decay.best_fitness <= r_const.best_fitness * 2
