"""
tests/test_contracts.py
======================
Additional API-contract and invariant tests that complement the existing
unit/integration suite without duplicating its numerical checks.

Focus areas:
- cache key normalization and LRU semantics
- logger guard behavior for edge parameter values
- Pareto archive defensive-copy contracts
- Pareto selection helper invariants
"""

import csv
import math

from harmonix.logging import EvaluationCache, RunLogger
from harmonix.pareto import ParetoArchive, non_dominated_front


class TestEvaluationCacheContracts:
    def test_harmony_key_ignores_dict_insertion_order(self):
        calls = [0]

        def obj(h):
            calls[0] += 1
            return h["x"] + h["y"], 0.0

        cache = EvaluationCache(obj, maxsize=16)
        cache({"x": 1.0, "y": 2.0})
        cache({"y": 2.0, "x": 1.0})

        assert calls[0] == 1
        assert cache.misses == 1
        assert cache.hits == 1

    def test_lru_hit_refreshes_recency(self):
        calls = [0]

        def obj(h):
            calls[0] += 1
            return h["x"], 0.0

        cache = EvaluationCache(obj, maxsize=2)
        cache({"x": 1.0})
        cache({"x": 2.0})

        # Refresh x=1.0 so x=2.0 becomes least recently used.
        cache({"x": 1.0})
        cache({"x": 3.0})

        calls_before = calls[0]
        cache({"x": 2.0})
        assert calls[0] == calls_before + 1


class TestRunLoggerContracts:
    def test_history_every_zero_is_guarded_to_one(self, tmp_path):
        hist_csv = tmp_path / "history.csv"
        logger = RunLogger(variable_names=["x"], history_log_path=hist_csv, history_every=0)

        for it in range(1, 4):
            logger.log_iteration(it, {"x": float(it)}, float(it), 0.0)

        rows = hist_csv.read_text().splitlines()
        assert len(rows) == 4  # header + 3 rows

    def test_log_evaluation_preserves_variable_column_order(self, tmp_path):
        eval_csv = tmp_path / "evals.csv"
        logger = RunLogger(variable_names=["a", "b", "c"], eval_log_path=eval_csv)
        logger.log_evaluation(1, {"c": 30, "a": 10, "b": 20}, 1.23, 0.0)

        row = next(csv.DictReader(eval_csv.open()))
        assert [row["a"], row["b"], row["c"]] == ["10", "20", "30"]


class TestParetoArchiveContracts:
    def test_entries_property_returns_shallow_copy(self):
        arch = ParetoArchive(max_size=10)
        arch.add({"x": 1}, (0.0, 1.0))

        entries = arch.entries
        entries.clear()

        assert len(arch) == 1

    def test_front_returns_shallow_copy(self):
        arch = ParetoArchive(max_size=10)
        arch.add({"x": 1}, (0.0, 1.0))

        front = arch.front()
        front.pop()

        assert len(arch.front()) == 1

    def test_crowding_tournament_with_k_greater_than_archive_size(self):
        arch = ParetoArchive(max_size=10)
        arch.add({"x": 0}, (0.0, 1.0))
        arch.add({"x": 1}, (1.0, 0.0))

        winner = arch.crowding_tournament(k=99)
        assert winner is not None
        assert winner.objectives in {(0.0, 1.0), (1.0, 0.0)}


class TestParetoUtilityContracts:
    def test_non_dominated_front_preserves_duplicate_non_dominated_indices(self):
        vecs = [(0.0, 1.0), (0.0, 1.0), (1.0, 0.0)]
        idx = non_dominated_front(vecs)
        assert idx == [0, 1, 2]

    def test_archive_prune_keeps_non_dominated_solutions(self):
        arch = ParetoArchive(max_size=3)
        for i in range(6):
            arch.add({"x": i}, (float(i), float(10 - i)))

        objs = [e.objectives for e in arch.front()]
        for i, a in enumerate(objs):
            for j, b in enumerate(objs):
                if i != j:
                    assert not (a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1]))

    def test_archive_boundary_only_case_still_selects_entry(self):
        arch = ParetoArchive(max_size=2)
        arch.add({"x": 0}, (0.0, 1.0))
        arch.add({"x": 1}, (1.0, 0.0))

        winner = arch.crowding_tournament(k=2)
        assert winner is not None
        assert math.isfinite(winner.objectives[0])
