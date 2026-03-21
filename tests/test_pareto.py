"""
tests/test_pareto.py
====================
Tests for ParetoArchive, crowding_distances, non_dominated_front,
and ArchiveEntry serialization.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from harmonix.pareto import (
    ParetoArchive,
    ArchiveEntry,
    dominates,
    non_dominated_front,
    crowding_distances,
)


# ---------------------------------------------------------------------------
# dominates (already in test_spaces.py but extended here)
# ---------------------------------------------------------------------------

class TestDominates:
    def test_strictly_better_in_all(self):
        assert dominates((1.0, 1.0), (2.0, 2.0))

    def test_better_in_one_equal_in_other(self):
        assert dominates((1.0, 2.0), (1.0, 3.0))

    def test_equal_does_not_dominate(self):
        assert not dominates((1.0, 2.0), (1.0, 2.0))

    def test_worse_in_one(self):
        assert not dominates((2.0, 1.0), (1.0, 2.0))

    def test_three_objectives(self):
        assert dominates((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
        assert not dominates((1.0, 2.0, 1.0), (1.0, 1.0, 2.0))

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            dominates((1.0, 2.0), (1.0,))


# ---------------------------------------------------------------------------
# non_dominated_front
# ---------------------------------------------------------------------------

class TestNonDominatedFront:
    def test_single_solution(self):
        result = non_dominated_front([(1.0, 2.0)])
        assert result == [0]

    def test_all_dominated_except_one(self):
        vecs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        result = non_dominated_front(vecs)
        assert result == [0]

    def test_all_non_dominated(self):
        # Trade-off front: lower f1 → higher f2
        vecs = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
        result = non_dominated_front(vecs)
        assert set(result) == {0, 1, 2}

    def test_mixed(self):
        vecs = [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0), (5.0, 5.0)]
        result = non_dominated_front(vecs)
        assert set(result) == {0, 1, 2}
        assert 3 not in result


# ---------------------------------------------------------------------------
# crowding_distances
# ---------------------------------------------------------------------------

class TestCrowdingDistances:
    def test_boundaries_get_inf(self):
        vecs = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
        dists = crowding_distances(vecs)
        assert math.isinf(dists[0])
        assert math.isinf(dists[2])
        assert not math.isinf(dists[1])

    def test_two_solutions_both_inf(self):
        vecs = [(0.0, 0.0), (1.0, 1.0)]
        dists = crowding_distances(vecs)
        assert all(math.isinf(d) for d in dists)

    def test_uniform_spacing_equal_distance(self):
        vecs = [(0.0,), (0.25,), (0.5,), (0.75,), (1.0,)]
        dists = crowding_distances(vecs)
        # Interior points should all have equal distance
        interior = dists[1:-1]
        assert all(abs(d - interior[0]) < 1e-9 for d in interior)

    def test_empty_returns_empty(self):
        assert crowding_distances([]) == []


# ---------------------------------------------------------------------------
# ParetoArchive
# ---------------------------------------------------------------------------

class TestParetoArchive:
    def test_add_non_dominated(self):
        arch = ParetoArchive(max_size=10)
        arch.add({"x": 1}, (0.0, 1.0))
        arch.add({"x": 2}, (1.0, 0.0))
        assert len(arch) == 2

    def test_dominated_not_added(self):
        arch = ParetoArchive(max_size=10)
        arch.add({"x": 1}, (1.0, 1.0))
        arch.add({"x": 2}, (2.0, 2.0))  # dominated by (1,1)
        assert len(arch) == 1

    def test_new_dominates_existing(self):
        arch = ParetoArchive(max_size=10)
        arch.add({"x": 1}, (2.0, 2.0))
        arch.add({"x": 2}, (1.0, 1.0))  # dominates previous
        assert len(arch) == 1
        assert arch.entries[0].objectives == (1.0, 1.0)

    def test_max_size_enforced(self):
        arch = ParetoArchive(max_size=3)
        for i in range(10):
            arch.add({"x": i}, (float(i), float(10 - i)))
        assert len(arch) <= 3

    def test_front_all_non_dominated(self):
        arch = ParetoArchive(max_size=20)
        for i in range(5):
            arch.add({"x": i}, (float(i), float(4 - i)))
        front = arch.front()
        objs = [e.objectives for e in front]
        for i, a in enumerate(objs):
            for j, b in enumerate(objs):
                if i != j:
                    assert not dominates(a, b)

    def test_random_entry_returns_entry(self):
        arch = ParetoArchive(max_size=10)
        arch.add({"x": 1}, (0.5, 0.5))
        entry = arch.random_entry()
        assert entry is not None
        assert entry.objectives == (0.5, 0.5)

    def test_random_entry_none_when_empty(self):
        arch = ParetoArchive(max_size=10)
        assert arch.random_entry() is None

    def test_crowding_tournament_returns_least_crowded(self):
        arch = ParetoArchive(max_size=10)
        # Add 3 points: endpoints have inf distance, middle is finite
        arch.add({"x": 0}, (0.0, 1.0))
        arch.add({"x": 1}, (0.5, 0.5))
        arch.add({"x": 2}, (1.0, 0.0))
        # Tournament of all 3 should return an endpoint (inf distance)
        entry = arch.crowding_tournament(k=3)
        assert math.isinf(0.0) or entry.objectives in [(0.0, 1.0), (1.0, 0.0)]

    def test_serialization_roundtrip(self):
        arch = ParetoArchive(max_size=10)
        arch.add({"x": 1.0, "y": 2.0}, (0.5, 0.5))
        arch.add({"x": 2.0, "y": 1.0}, (0.8, 0.2))
        data = arch.to_dict()
        arch2 = ParetoArchive.from_dict(data)
        assert len(arch2) == len(arch)
        objs_orig = {e.objectives for e in arch.entries}
        objs_rest = {tuple(e.objectives) for e in arch2.entries}
        assert objs_orig == objs_rest

    def test_duplicate_not_double_added(self):
        arch = ParetoArchive(max_size=10)
        arch.add({"x": 1}, (0.5, 0.5))
        arch.add({"x": 2}, (0.5, 0.5))   # same objectives, different harmony
        # Both are non-dominated (equal, not dominating each other)
        # so both should be present
        assert len(arch) == 2


# ---------------------------------------------------------------------------
# ArchiveEntry
# ---------------------------------------------------------------------------

class TestArchiveEntry:
    def test_fields(self):
        entry = ArchiveEntry(harmony={"x": 1.0}, objectives=(0.5, 0.7))
        assert entry.harmony == {"x": 1.0}
        assert entry.objectives == (0.5, 0.7)
