"""
pareto.py
=========
Pareto dominance, archive management, and crowding-distance selection
for multi-objective Harmony Search.

Terminology
-----------
* **Objective vector**: a tuple of scalar fitness values, one per objective.
  All objectives are assumed to be *minimised*; negate maximised objectives
  in the user's ``objective`` function.
* **Dominance**: solution *a* dominates *b* (written a ≻ b) when
  ``a[i] <= b[i]`` for all objectives and ``a[i] < b[i]`` for at least one.
* **Pareto front**: the set of non-dominated solutions in the current archive.
* **Crowding distance**: a measure of density around each solution along
  the objective axes.  Higher crowding distance = more isolated = preferred
  when the archive overflows.

References
----------
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
    A fast and elitist multiobjective genetic algorithm: NSGA-II.
    *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

Harmony = Dict[str, Any]
ObjVector = Tuple[float, ...]  # one float per objective


# ---------------------------------------------------------------------------
# Dominance utilities
# ---------------------------------------------------------------------------


def dominates(a: ObjVector, b: ObjVector) -> bool:
    """
    Return ``True`` when *a* Pareto-dominates *b* (minimisation assumed).

    *a* dominates *b* iff:
    - ``a[i] <= b[i]`` for every objective *i*, **and**
    - ``a[i] <  b[i]`` for at least one objective *i*.
    """
    if len(a) != len(b):
        raise ValueError("Objective vectors must have the same length.")
    at_least_one_better = False
    for ai, bi in zip(a, b):
        if ai > bi:
            return False
        if ai < bi:
            at_least_one_better = True
    return at_least_one_better


def non_dominated_front(
    objective_vectors: List[ObjVector],
) -> List[int]:
    """
    Return indices of the non-dominated solutions in *objective_vectors*.

    This is an O(n²·m) scan; adequate for the archive sizes used in HS
    (typically ≤ 200 solutions).
    """
    n = len(objective_vectors)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objective_vectors[j], objective_vectors[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def crowding_distances(
    objective_vectors: List[ObjVector],
) -> List[float]:
    """
    Compute the crowding distance for each solution in *objective_vectors*.

    Boundary solutions (best/worst per objective) receive ``math.inf``.
    Interior solutions receive the sum of normalised distances to their
    neighbours across all objectives.

    Returns
    -------
    list of float
        Distance for each index in the same order as *objective_vectors*.
    """
    n = len(objective_vectors)
    if n == 0:
        return []
    n_obj = len(objective_vectors[0])
    distances = [0.0] * n

    for m in range(n_obj):
        # Sort by objective m
        order = sorted(range(n), key=lambda i, objective_index=m: objective_vectors[i][objective_index])
        f_min = objective_vectors[order[0]][m]
        f_max = objective_vectors[order[-1]][m]
        span = f_max - f_min

        # Boundary points get infinite distance
        distances[order[0]] = math.inf
        distances[order[-1]] = math.inf

        if span == 0:
            continue

        for k in range(1, n - 1):
            prev_val = objective_vectors[order[k - 1]][m]
            next_val = objective_vectors[order[k + 1]][m]
            distances[order[k]] += (next_val - prev_val) / span

    return distances


# ---------------------------------------------------------------------------
# Archive entry
# ---------------------------------------------------------------------------


@dataclass
class ArchiveEntry:
    harmony: Harmony
    objectives: ObjVector


# ---------------------------------------------------------------------------
# Pareto archive
# ---------------------------------------------------------------------------


class ParetoArchive:
    """
    Bounded archive that stores the current Pareto-non-dominated solutions.

    After every insertion the archive is pruned to *max_size* using
    crowding-distance selection: the solution with the smallest crowding
    distance (most crowded neighbourhood) is removed first.

    Parameters
    ----------
    max_size : int
        Maximum number of solutions retained.

    Examples
    --------
    >>> archive = ParetoArchive(max_size=50)
    >>> archive.add({"x": 1.0, "y": 2.0}, (3.0, 4.0))
    >>> archive.add({"x": 0.5, "y": 1.5}, (2.0, 5.0))
    >>> len(archive)
    2
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._entries: List[ArchiveEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[ArchiveEntry]:
        return list(self._entries)

    # --- insertion ---------------------------------------------------------

    def add(self, harmony: Harmony, objectives: ObjVector) -> None:
        """
        Attempt to add *(harmony, objectives)* to the archive.

        Steps:
        1. Reject if the candidate is dominated by any archived solution.
        2. Remove any archived solutions dominated by the candidate.
        3. Insert the candidate.
        4. If the archive exceeds *max_size*, prune by crowding distance.
        """
        # 1. Reject dominated candidates
        for entry in self._entries:
            if dominates(entry.objectives, objectives):
                return  # candidate is dominated — do not insert

        # 2. Remove solutions the candidate dominates
        self._entries = [e for e in self._entries if not dominates(objectives, e.objectives)]

        # 3. Insert
        self._entries.append(ArchiveEntry(harmony=harmony, objectives=objectives))

        # 4. Prune to max_size
        while len(self._entries) > self.max_size:
            self._prune_most_crowded()

    def _prune_most_crowded(self) -> None:
        vecs = [e.objectives for e in self._entries]
        dists = crowding_distances(vecs)
        # Remove the entry with the smallest finite crowding distance
        finite_indices = [i for i, d in enumerate(dists) if d != math.inf]
        if finite_indices:
            victim = min(finite_indices, key=lambda i: dists[i])
        else:
            victim = random.randrange(len(self._entries))
        self._entries.pop(victim)

    # --- selection for harmony generation ----------------------------------

    def random_entry(self) -> Optional[ArchiveEntry]:
        """Return a uniformly random archive entry, or None if empty."""
        return random.choice(self._entries) if self._entries else None

    def crowding_tournament(self, k: int = 2) -> Optional[ArchiveEntry]:
        """
        Select one entry via tournament on crowding distance.

        The winner is the *least crowded* (largest distance) candidate,
        promoting diversity in the harmony improvisation.
        """
        if not self._entries:
            return None
        k = min(k, len(self._entries))
        contestants = random.sample(self._entries, k)
        vecs = [e.objectives for e in self._entries]
        dists = crowding_distances(vecs)
        idx_map = {id(e): i for i, e in enumerate(self._entries)}
        return max(contestants, key=lambda e: dists[idx_map[id(e)]])

    # --- reporting ---------------------------------------------------------

    def front(self) -> List[ArchiveEntry]:
        """Return all current Pareto-non-dominated entries."""
        return list(self._entries)

    def to_dict(self) -> dict:
        return {
            "max_size": self.max_size,
            "entries": [{"harmony": e.harmony, "objectives": list(e.objectives)} for e in self._entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParetoArchive":
        archive = cls(max_size=data["max_size"])
        for raw in data["entries"]:
            archive._entries.append(
                ArchiveEntry(
                    harmony=raw["harmony"],
                    objectives=tuple(raw["objectives"]),
                )
            )
        return archive


# ---------------------------------------------------------------------------
# Result container for multi-objective runs
# ---------------------------------------------------------------------------


@dataclass
class ParetoResult:
    """
    Output of a :class:`~optimizer.MultiObjective` run.

    Attributes
    ----------
    front : list of ArchiveEntry
        Final Pareto-non-dominated solutions.
    archive_history : list of int
        Archive size recorded at each iteration.
    iterations : int
    elapsed_seconds : float
    """

    front: List[ArchiveEntry]
    archive_history: List[int]
    iterations: int
    elapsed_seconds: float

    def __repr__(self) -> str:
        n_obj = len(self.front[0].objectives) if self.front else 0
        lines = [
            "ParetoResult(",
            f"  Pareto front size = {len(self.front)}",
            f"  objectives        = {n_obj}",
            f"  iterations        = {self.iterations}",
            f"  elapsed           = {self.elapsed_seconds:.2f}s",
            "  front (first 5):",
        ]
        for entry in self.front[:5]:
            obj_str = ", ".join(f"{v:.4g}" for v in entry.objectives)
            lines.append(f"    objectives=({obj_str})")
        if len(self.front) > 5:
            lines.append(f"    … and {len(self.front) - 5} more")
        lines.append(")")
        return "\n".join(lines)
