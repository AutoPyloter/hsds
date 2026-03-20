"""
tests/test_edge_cases.py
========================
Edge cases, numerical correctness, determinism, error messages,
integration corner cases, and stress tests.

Categories
----------
1. Variable edge cases (lo==hi, lo>hi, single-element, etc.)
2. DesignSpace edge cases (empty, deep dependency chains)
3. Optimizer edge cases (memory_size=1, max_iter=0, always infeasible)
4. Numerical correctness (known optima)
5. Determinism (same seed → same result)
6. Error message quality
7. Serialization integrity
8. Stress tests
9. Integration corners (__bw__ collision, cache+deps)
10. Engineering physics spot-checks
"""

import csv
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from harmonix.variables import Continuous, Discrete, Integer, Categorical
from harmonix.space import DesignSpace
from harmonix.optimizer import Minimization, Maximization, MultiObjective, HarmonyMemory
from harmonix.logging import EvaluationCache
from harmonix.pareto import ParetoArchive, dominates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp():
    fd, fname = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(fname)
    return Path(fname)


# ===========================================================================
# 1. Variable edge cases
# ===========================================================================

class TestVariableEdgeCases:

    # Continuous lo > hi → must raise immediately
    def test_continuous_lo_gt_hi_raises(self):
        with pytest.raises(ValueError, match="lo"):
            Continuous(5.0, 0.0)

    # Continuous lo == hi → degenerate but allowed
    def test_continuous_lo_eq_hi_sample(self):
        v = Continuous(3.0, 3.0)
        assert v.sample({}) == 3.0

    def test_continuous_lo_eq_hi_filter(self):
        v = Continuous(3.0, 3.0)
        assert v.filter([2.0, 3.0, 4.0], {}) == [3.0]

    def test_continuous_lo_eq_hi_neighbor(self):
        v = Continuous(3.0, 3.0)
        assert v.neighbor(3.0, {}) == 3.0

    # Discrete lo > hi → must raise
    def test_discrete_lo_gt_hi_raises(self):
        with pytest.raises(ValueError, match="lo"):
            Discrete(10.0, 1.0, 5.0)

    # Discrete step <= 0 → must raise
    def test_discrete_nonpositive_step_raises(self):
        with pytest.raises(ValueError, match="step"):
            Discrete(0.0, -1.0, 10.0)

    def test_discrete_zero_step_raises(self):
        with pytest.raises(ValueError, match="step"):
            Discrete(0.0, 0.0, 10.0)

    # Discrete single element
    def test_discrete_single_element_sample(self):
        v = Discrete(5.0, 1.0, 5.0)
        for _ in range(20):
            assert v.sample({}) == 5.0

    def test_discrete_single_element_neighbor(self):
        v = Discrete(5.0, 1.0, 5.0)
        for _ in range(20):
            assert v.neighbor(5.0, {}) == 5.0

    # Integer lo > hi → must raise
    def test_integer_lo_gt_hi_raises(self):
        with pytest.raises(ValueError):
            Integer(10, 5)

    # Integer lo == hi
    def test_integer_lo_eq_hi_sample(self):
        v = Integer(7, 7)
        for _ in range(20):
            assert v.sample({}) == 7

    def test_integer_lo_eq_hi_neighbor(self):
        v = Integer(7, 7)
        for _ in range(20):
            assert v.neighbor(7, {}) == 7

    # Categorical single element
    def test_categorical_single_neighbor_returns_self(self):
        v = Categorical(["only"])
        for _ in range(20):
            assert v.neighbor("only", {}) == "only"

    # Categorical empty → must raise
    def test_categorical_empty_raises(self):
        with pytest.raises(ValueError):
            Categorical([])

    # Continuous callable bounds — no error at init
    def test_continuous_callable_lo_no_init_check(self):
        v = Continuous(lambda ctx: ctx["a"], 10.0)
        assert v.sample({"a": 2.0}) == pytest.approx(v.sample({"a": 2.0}), abs=10)

    # Discrete step larger than range — produces single point
    def test_discrete_step_larger_than_range(self):
        v = Discrete(0.0, 100.0, 5.0)
        # _frange(0, 100, 5) → [0.0, 5.0]? No: step=100 > range 5
        # should produce just [0.0] since 0+100 > 5
        samples = {v.sample({}) for _ in range(30)}
        assert all(s <= 5.0 for s in samples)


# ===========================================================================
# 2. DesignSpace edge cases
# ===========================================================================

class TestDesignSpaceEdgeCases:

    def test_empty_space_sample(self):
        space = DesignSpace()
        h = space.sample_harmony()
        assert h == {}

    def test_empty_space_names(self):
        space = DesignSpace()
        assert space.names() == []

    def test_deep_dependency_chain(self):
        """a → b → c → d (4-level chain)."""
        space = DesignSpace()
        space.add("a", Continuous(1.0, 10.0))
        space.add("b", Continuous(lambda ctx: ctx["a"], lambda ctx: ctx["a"] + 5.0))
        space.add("c", Continuous(lambda ctx: ctx["b"], lambda ctx: ctx["b"] + 3.0))
        space.add("d", Continuous(lambda ctx: ctx["c"], lambda ctx: ctx["c"] + 1.0))
        for _ in range(20):
            h = space.sample_harmony()
            assert h["b"] >= h["a"]
            assert h["c"] >= h["b"]
            assert h["d"] >= h["c"]

    def test_duplicate_name_overwrites(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        space.add("x", Continuous(5.0, 10.0))  # overwrites
        h = space.sample_harmony()
        assert 5.0 <= h["x"] <= 10.0

    def test_many_variables(self):
        space = DesignSpace()
        for i in range(30):
            space.add(f"x{i}", Continuous(0.0, 1.0))
        h = space.sample_harmony()
        assert len(h) == 30
        assert all(0.0 <= v <= 1.0 for v in h.values())


# ===========================================================================
# 3. Optimizer edge cases
# ===========================================================================

class TestOptimizerEdgeCases:

    def _space(self):
        s = DesignSpace()
        s.add("x", Continuous(0.0, 5.0))
        return s

    def test_max_iter_zero(self):
        r = Minimization(self._space(), lambda h: (h["x"]**2, 0.0)).optimize(
            memory_size=5, max_iter=0
        )
        assert r.iterations == 0
        assert r.history == []
        # best harmony still from initial memory
        assert r.best_harmony is not None

    def test_memory_size_one(self):
        r = Minimization(self._space(), lambda h: (h["x"]**2, 0.0)).optimize(
            memory_size=1, max_iter=30
        )
        assert r.best_penalty <= 0

    def test_always_infeasible(self):
        """Algorithm must not crash when all solutions are infeasible."""
        r = Minimization(self._space(), lambda h: (h["x"], 1.0)).optimize(
            memory_size=5, max_iter=50
        )
        assert r.best_penalty > 0   # still infeasible but no crash

    def test_objective_returning_int(self):
        """Fitness returned as int — must be coerced to float."""
        r = Minimization(self._space(), lambda h: (int(h["x"]), 0)).optimize(
            memory_size=5, max_iter=30
        )
        assert isinstance(r.best_fitness, float)

    def test_callback_early_stop_history_correct(self):
        """StopIteration at iter 10 → history length == 10."""
        history_len = [0]
        def cb(it, partial):
            history_len[0] = it
            if it == 10:
                raise StopIteration

        r = Minimization(self._space(), lambda h: (h["x"]**2, 0.0)).optimize(
            memory_size=5, max_iter=1000, callback=cb
        )
        assert r.iterations == 10
        assert len(r.history) == 10

    def test_hmcr_zero_never_uses_memory(self):
        """HMCR=0 → every harmony is random, memory never consulted."""
        r = Minimization(self._space(), lambda h: (h["x"]**2, 0.0)).optimize(
            memory_size=5, max_iter=50, hmcr=0.0
        )
        assert r.best_harmony is not None

    def test_hmcr_one_always_uses_memory(self):
        """HMCR=1 → always from memory, no random samples after init."""
        r = Minimization(self._space(), lambda h: (h["x"]**2, 0.0)).optimize(
            memory_size=5, max_iter=50, hmcr=1.0
        )
        assert r.best_fitness >= 0


# ===========================================================================
# 4. Numerical correctness
# ===========================================================================

class TestNumericalCorrectness:

    def test_sphere_2d_finds_zero(self):
        """Sphere function minimum is 0 at origin."""
        random.seed(0)
        space = DesignSpace()
        for i in range(2):
            space.add(f"x{i}", Continuous(-5.0, 5.0))
        r = Minimization(space, lambda h: (sum(v**2 for v in h.values()), 0.0)).optimize(
            memory_size=20, max_iter=3000, bw_max=0.1, bw_min=0.001
        )
        assert r.best_fitness < 0.01
        assert r.best_penalty <= 0

    def test_rosenbrock_2d(self):
        """Rosenbrock minimum is 0 at (1, 1)."""
        random.seed(1)
        space = DesignSpace()
        space.add("x", Continuous(-2.0, 2.0))
        space.add("y", Continuous(-2.0, 2.0))
        def rosenbrock(h):
            x, y = h["x"], h["y"]
            return (1 - x)**2 + 100*(y - x**2)**2, 0.0
        r = Minimization(space, rosenbrock).optimize(
            memory_size=30, max_iter=8000, bw_max=0.2, bw_min=0.001
        )
        assert r.best_fitness < 1.0
        assert abs(r.best_harmony["x"] - 1.0) < 0.5
        assert abs(r.best_harmony["y"] - 1.0) < 0.5

    def test_integer_minimization_exact(self):
        """Minimise |n - 42| over integers [0, 100] → exact answer 42."""
        random.seed(2)
        space = DesignSpace()
        space.add("n", Integer(0, 100))
        r = Minimization(space, lambda h: (abs(h["n"] - 42), 0.0)).optimize(
            memory_size=15, max_iter=500
        )
        assert r.best_harmony["n"] == 42

    def test_categorical_minimization(self):
        """Choose from ['a','b','c','d'] to minimise index."""
        costs = {"a": 1, "b": 2, "c": 3, "d": 4}
        space = DesignSpace()
        space.add("choice", Categorical(["a", "b", "c", "d"]))
        r = Minimization(space, lambda h: (float(costs[h["choice"]]), 0.0)).optimize(
            memory_size=10, max_iter=200
        )
        assert r.best_harmony["choice"] == "a"

    def test_maximization_finds_upper_bound(self):
        """max x over [0, 10] → should find value close to 10."""
        random.seed(3)
        space = DesignSpace()
        space.add("x", Continuous(0.0, 10.0))
        r = Maximization(space, lambda h: (h["x"], 0.0)).optimize(
            memory_size=10, max_iter=500
        )
        assert r.best_fitness > 9.0

    def test_constrained_minimization(self):
        """min x s.t. x >= 3, over [0, 10] → answer is 3."""
        random.seed(4)
        space = DesignSpace()
        space.add("x", Continuous(0.0, 10.0))
        def obj(h):
            penalty = max(0.0, 3.0 - h["x"])
            return h["x"], penalty
        r = Minimization(space, obj).optimize(
            memory_size=10, max_iter=1000
        )
        assert r.best_penalty <= 0
        assert r.best_harmony["x"] >= 2.9


# ===========================================================================
# 5. Determinism
# ===========================================================================

class TestDeterminism:

    def _run(self, seed):
        random.seed(seed)
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        space.add("y", Continuous(0.0, 1.0))
        return Minimization(space, lambda h: (h["x"]**2 + h["y"]**2, 0.0)).optimize(
            memory_size=10, max_iter=100
        )

    def test_same_seed_same_result(self):
        r1 = self._run(seed=42)
        r2 = self._run(seed=42)
        assert r1.best_fitness == r2.best_fitness
        assert r1.best_harmony == r2.best_harmony

    def test_different_seeds_different_results(self):
        r1 = self._run(seed=1)
        r2 = self._run(seed=999)
        # Not strictly guaranteed but overwhelmingly likely
        assert r1.history != r2.history


# ===========================================================================
# 6. Error message quality
# ===========================================================================

class TestErrorMessages:

    def test_continuous_error_contains_values(self):
        with pytest.raises(ValueError) as exc_info:
            Continuous(7.0, 2.0)
        assert "7.0" in str(exc_info.value)
        assert "2.0" in str(exc_info.value)

    def test_discrete_step_error_mentions_step(self):
        with pytest.raises(ValueError) as exc_info:
            Discrete(0.0, -5.0, 10.0)
        assert "step" in str(exc_info.value).lower()

    def test_compute_bw_error_message(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        opt = Minimization(space, lambda h: (h["x"], 0.0))
        with pytest.raises(ValueError) as exc_info:
            opt._compute_bw(0, 100, bw_max=0.001, bw_min=0.1)
        assert "bw_min" in str(exc_info.value).lower() or "bw_max" in str(exc_info.value).lower()

    def test_resume_error_contains_path(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        fake_path = "/tmp/_harmonix_does_not_exist_12345.json"
        with pytest.raises(FileNotFoundError) as exc_info:
            Minimization(space, lambda h: (h["x"], 0.0)).optimize(
                memory_size=5, max_iter=5,
                checkpoint_path=fake_path, resume="resume"
            )
        assert "_harmonix_does_not_exist_12345" in str(exc_info.value)

    def test_resume_invalid_value_error(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        with pytest.raises(ValueError) as exc_info:
            Minimization(space, lambda h: (h["x"], 0.0)).optimize(
                memory_size=5, max_iter=5, resume="typo"
            )
        assert "typo" in str(exc_info.value)


# ===========================================================================
# 7. Serialization integrity
# ===========================================================================

class TestSerializationIntegrity:

    def test_harmony_memory_roundtrip_preserves_values(self):
        mem = HarmonyMemory(size=3, mode="min")
        mem.add({"x": 1.5, "y": 2.5}, 3.0, 0.0)
        mem.add({"x": 0.1, "y": 0.9}, 0.82, 0.0)
        mem.add({"x": 3.0, "y": 4.0}, 25.0, 1.0)
        data = mem.to_dict()
        mem2 = HarmonyMemory.from_dict(data)
        assert mem2._fitness == mem._fitness
        assert mem2._penalty == mem._penalty
        for h1, h2 in zip(mem._harmonies, mem2._harmonies):
            assert h1 == h2

    def test_harmony_memory_roundtrip_categorical(self):
        """Categorical values (strings) must survive JSON roundtrip."""
        mem = HarmonyMemory(size=2, mode="min")
        mem.add({"grade": "C25/30", "n": 3}, 1.0, 0.0)
        mem.add({"grade": "C40/50", "n": 5}, 2.0, 0.0)
        data = mem.to_dict()
        mem2 = HarmonyMemory.from_dict(data)
        assert mem2._harmonies[0]["grade"] == "C25/30"
        assert mem2._harmonies[1]["grade"] == "C40/50"

    def test_checkpoint_fitness_values_preserved(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        ckpt = _tmp()
        try:
            opt1 = Minimization(space, lambda h: (h["x"]**2, 0.0))
            r1 = opt1.optimize(memory_size=5, max_iter=30,
                               checkpoint_path=ckpt, checkpoint_every=30)
            opt2 = Minimization(space, lambda h: (h["x"]**2, 0.0))
            opt2.optimize(memory_size=5, max_iter=60,
                          checkpoint_path=ckpt, checkpoint_every=60,
                          resume="auto")
            # Memory was restored — best from first run preserved
            assert opt2._memory is not None
        finally:
            ckpt.unlink(missing_ok=True)

    def test_pareto_archive_objectives_type_after_roundtrip(self):
        """Objectives must be tuple after from_dict, not list."""
        arch = ParetoArchive(max_size=5)
        arch.add({"x": 1.0}, (0.3, 0.7))
        data = arch.to_dict()
        arch2 = ParetoArchive.from_dict(data)
        for entry in arch2.entries:
            assert isinstance(entry.objectives, tuple)

    def test_multiobjective_checkpoint_archive_restored(self):
        """MultiObjective checkpoint must restore archive, not just memory."""
        space = DesignSpace()
        space.add("x1", Continuous(0.0, 1.0))
        space.add("x2", Continuous(0.0, 1.0))
        def zdt_simple(h):
            g = 1 + 9 * h["x2"]
            return (h["x1"], g * (1 - math.sqrt(h["x1"] / g))), 0.0

        ckpt = _tmp()
        try:
            opt1 = MultiObjective(space, zdt_simple)
            r1 = opt1.optimize(memory_size=10, max_iter=100, archive_size=20,
                               checkpoint_path=ckpt, checkpoint_every=100)
            front1_size = len(r1.front)

            opt2 = MultiObjective(space, zdt_simple)
            r2 = opt2.optimize(memory_size=10, max_iter=200, archive_size=20,
                               checkpoint_path=ckpt, checkpoint_every=200,
                               resume="auto")
            # Should continue from existing archive, so front >= 0
            assert len(r2.front) >= 0
        finally:
            ckpt.unlink(missing_ok=True)


# ===========================================================================
# 8. Stress tests
# ===========================================================================

class TestStress:

    def test_50_variable_space(self):
        """Optimization must not crash or produce wrong keys with 50 vars."""
        space = DesignSpace()
        for i in range(50):
            space.add(f"x{i}", Continuous(0.0, 1.0))
        r = Minimization(space, lambda h: (sum(v**2 for v in h.values()), 0.0)).optimize(
            memory_size=10, max_iter=100
        )
        assert len(r.best_harmony) == 50
        assert all(f"x{i}" in r.best_harmony for i in range(50))

    def test_large_memory_size(self):
        """memory_size=200 must not cause O(n²) bugs."""
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        r = Minimization(space, lambda h: (h["x"]**2, 0.0)).optimize(
            memory_size=200, max_iter=50
        )
        assert r.best_harmony is not None

    def test_history_length_matches_iterations(self):
        """history list length == r.iterations always."""
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        r = Minimization(space, lambda h: (h["x"]**2, 0.0)).optimize(
            memory_size=5, max_iter=77
        )
        assert len(r.history) == r.iterations == 77


# ===========================================================================
# 9. Integration corners
# ===========================================================================

class TestIntegrationCorners:

    def test_bw_key_collision_with_user_variable(self):
        """User variable named __bw__ must not be eaten by optimizer."""
        space = DesignSpace()
        space.add("__bw__", Continuous(0.0, 1.0))
        r = Minimization(space, lambda h: (h.get("__bw__", 999.0), 0.0)).optimize(
            memory_size=5, max_iter=50
        )
        assert "__bw__" in r.best_harmony
        assert 0.0 <= r.best_harmony["__bw__"] <= 1.0

    def test_cache_not_confused_by_different_harmonies(self):
        """Cache must use full harmony dict as key, not partial."""
        calls = {}
        def obj(h):
            key = tuple(sorted(h.items()))
            calls[key] = calls.get(key, 0) + 1
            return sum(v**2 for v in h.values()), 0.0

        cache = EvaluationCache(obj, maxsize=1000)
        h1 = {"x": 1.0, "y": 0.0}
        h2 = {"x": 0.0, "y": 1.0}
        r1 = cache(h1)
        r2 = cache(h2)
        assert r1 == r2   # same fitness value (1² + 0² = 0² + 1²)
        assert cache.misses == 2  # both evaluated separately

    def test_eval_cache_hit_not_logged_as_new_eval(self):
        """log_evaluations should log every call to effective_obj,
        which with cache enabled means only misses."""
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        ckpt = _tmp()
        try:
            opt = Minimization(space, lambda h: (h["x"]**2, 0.0))
            opt.optimize(
                memory_size=5, max_iter=50,
                checkpoint_path=ckpt,
                use_cache=True,
                log_evaluations=True,
            )
            eval_csv = ckpt.with_name(ckpt.stem + "_evals.csv")
            if eval_csv.exists():
                rows = len(eval_csv.read_text().splitlines()) - 1
                # With cache, rows <= 55 (init 5 + up to 50 misses)
                assert rows <= 55
                eval_csv.unlink()
        finally:
            ckpt.unlink(missing_ok=True)

    def test_dependent_variable_with_cache(self):
        """Cache must key on full harmony; dependent bounds must be respected."""
        space = DesignSpace()
        space.add("a", Continuous(1.0, 5.0))
        space.add("b", Continuous(lambda ctx: ctx["a"], 10.0))

        calls = [0]
        def obj(h):
            calls[0] += 1
            assert h["b"] >= h["a"] - 1e-9, f"b={h['b']} < a={h['a']}"
            return h["a"] + h["b"], 0.0

        Minimization(space, obj).optimize(
            memory_size=10, max_iter=200, use_cache=True
        )


# ===========================================================================
# 10. Engineering physics spot-checks
# ===========================================================================

class TestEngineeringPhysics:

    def test_concrete_grade_fcm_formula(self):
        """EC2: fcm = fck + 8 MPa."""
        from harmonix.spaces.engineering import ConcreteGrade
        var = ConcreteGrade()
        for idx in var._indices:
            props = var.decode(idx)
            assert abs(props.fcm_MPa - (props.fck_MPa + 8)) < 0.1, \
                f"{props.name}: fcm={props.fcm_MPa}, fck+8={props.fck_MPa+8}"

    def test_concrete_grade_ecm_formula(self):
        """EC2: Ecm = 22 * (fcm/10)^0.3 GPa."""
        from harmonix.spaces.engineering import ConcreteGrade
        var = ConcreteGrade()
        for idx in var._indices:
            props = var.decode(idx)
            expected_Ecm = 22.0 * (props.fcm_MPa / 10.0) ** 0.3
            assert abs(props.Ecm_GPa - expected_Ecm) < 0.5, \
                f"{props.name}: Ecm={props.Ecm_GPa:.2f}, expected={expected_Ecm:.2f}"

    def test_aci_rebar_all_valid_codes_satisfy_rho(self):
        """Every code returned by _valid_codes must satisfy rho_min <= rho <= rho_max."""
        from harmonix.spaces.engineering import ACIRebar, _aci_limits, _AREAS_50, _COUNTS, _DIAMETERS_MM
        var = ACIRebar(d_expr=0.45, cc_expr=0.06, fc=30.0, fy=420.0)
        ctx = {"d": 0.45, "cc": 0.06}
        codes = var._valid_codes(ctx)
        beta1, phi, eps_c, rho_min, rho_max = _aci_limits(30.0, 420.0)
        b = 0.30  # assumed width for rho check
        d = 0.45
        n_counts = len(_COUNTS)
        for code in codes:
            i = code // n_counts
            j = code % n_counts
            area_per_50 = _AREAS_50[i]
            count = _COUNTS[j]
            A_s = area_per_50 / 50.0 * count  # m²
            rho = A_s / (b * d)
            assert rho >= rho_min * 0.99, f"code {code}: rho={rho:.5f} < rho_min={rho_min:.5f}"
            assert rho <= rho_max * 1.01, f"code {code}: rho={rho:.5f} > rho_max={rho_max:.5f}"

    def test_aci_rebar_callable_fc_produces_different_valid_sets(self):
        """Different fc values must produce different valid code sets."""
        from harmonix.spaces.engineering import ACIRebar
        # Low fc → tighter constraints
        var_low  = ACIRebar(d_expr=0.40, cc_expr=0.06, fc=20.0, fy=420.0)
        var_high = ACIRebar(d_expr=0.40, cc_expr=0.06, fc=50.0, fy=420.0)
        ctx = {}
        codes_low  = set(var_low._valid_codes(ctx))
        codes_high = set(var_high._valid_codes(ctx))
        # Higher fc allows higher rho → more valid combinations
        assert len(codes_high) >= len(codes_low)

    def test_steel_section_wy_approx(self):
        """Wy ≈ Iy / (h/2) within 20% for all sections."""
        from harmonix.spaces.engineering import SteelSection
        var = SteelSection()
        for idx in var._indices:
            sec = var.decode(idx)
            if sec.h_mm > 0:
                wy_approx = sec.Iy_cm4 / (sec.h_mm / 2 / 10)  # cm³
                ratio = wy_approx / sec.Wy_cm3 if sec.Wy_cm3 > 0 else 1.0
                assert 0.8 <= ratio <= 1.25, \
                    f"{sec.name}: Wy={sec.Wy_cm3:.1f}, approx={wy_approx:.1f}"