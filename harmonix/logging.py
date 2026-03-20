"""
logging.py
==========
Logging, caching and persistence utilities for the Harmony Search optimiser.

Classes
-------
EvaluationCache
    LRU cache that wraps an objective function.  Identical harmonies are
    never evaluated twice.

RunLogger
    Handles all optional file-based logging:

    * **init log**      — the harmony memory as initialised (before any
                          improvisation).
    * **evaluation log** — every harmony that was passed to the objective,
                           with its fitness and penalty.
    * **history log**   — the best (fitness, penalty) recorded at each
                          iteration (or every N iterations).

All writers are CSV-based so results can be opened directly in Excel or
read with ``pandas.read_csv()``.

Usage
-----
These classes are instantiated automatically by the optimiser when the
corresponding flags are set in :meth:`optimize`.  You do not need to
create them manually.
"""

from __future__ import annotations

import csv
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

Harmony = Dict[str, Any]
Fitness = float
Penalty = float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(
    explicit: Optional[Path],
    checkpoint_path: Optional[Path],
    suffix: str,
) -> Optional[Path]:
    """
    Return the resolved log file path.

    Priority:
    1. *explicit* — the path the user passed directly.
    2. Derived from *checkpoint_path* by appending *suffix*.
    3. ``None`` — logging disabled.
    """
    if explicit is not None:
        return Path(explicit)
    if checkpoint_path is not None:
        base = Path(checkpoint_path)
        return base.with_name(base.stem + suffix + ".csv")
    return None


def _harmony_key(harmony: Harmony) -> tuple:
    """Convert a harmony dict to a hashable key for caching."""
    return tuple(sorted((k, v) for k, v in harmony.items()))


# ---------------------------------------------------------------------------
# EvaluationCache
# ---------------------------------------------------------------------------

class EvaluationCache:
    """
    LRU cache wrapper around an objective function.

    Identical harmonies (same variable values) are never re-evaluated.
    This is particularly valuable when:

    * The objective involves a finite-element simulation or other expensive
      computation.
    * The harmony memory tends to recycle the same high-quality solutions
      during late iterations.

    Parameters
    ----------
    objective : callable
        The original objective function.
    maxsize : int
        Maximum number of entries to keep in the cache.
        Oldest entries are evicted when the cache is full.

    Attributes
    ----------
    hits : int
        Number of cache hits (evaluations avoided).
    misses : int
        Number of cache misses (evaluations performed).

    Examples
    --------
    >>> cache = EvaluationCache(objective, maxsize=2048)
    >>> f, p = cache({"x": 1.0, "y": 2.0})
    >>> print(cache.hits, cache.misses)
    """

    def __init__(self, objective: Callable, maxsize: int = 4096):
        self._objective = objective
        self._maxsize   = maxsize
        self._cache: OrderedDict = OrderedDict()
        self.hits   = 0
        self.misses = 0

    def __call__(self, harmony: Harmony) -> Tuple[Fitness, Penalty]:
        key = _harmony_key(harmony)
        if key in self._cache:
            self._cache.move_to_end(key)   # mark as recently used
            self.hits += 1
            return self._cache[key]

        result = self._objective(harmony)
        self._cache[key] = result
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)  # evict oldest
        self.misses += 1
        return result

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self.hits = self.misses = 0

    def stats(self) -> str:
        total = self.hits + self.misses
        rate  = self.hits / total * 100 if total else 0
        return (
            f"EvaluationCache: {self.hits} hits / {total} total "
            f"({rate:.1f}% hit rate)  size={self.size}/{self._maxsize}"
        )


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------

class RunLogger:
    """
    File-based logger for optimisation runs.

    Manages up to three CSV log files:

    ``init_log_path``
        Written once after harmony memory initialisation.
        Columns: ``harmony_index``, all variable names, ``fitness``,
        ``penalty``, ``feasible``.

    ``eval_log_path``
        Appended on every objective evaluation (or skipped if disabled).
        Columns: ``wall_time``, ``iteration``, all variable names,
        ``fitness``, ``penalty``, ``feasible``.

    ``history_log_path``
        Appended every *history_every* iterations with the best solution
        found so far.
        Columns: ``iteration``, ``best_fitness``, ``best_penalty``,
        ``feasible``, all variable names of the best harmony.

    Parameters
    ----------
    variable_names : list of str
        Variable names in definition order (used as CSV column headers).
    init_log_path : Path or None
    eval_log_path : Path or None
    history_log_path : Path or None
    history_every : int
        Write to history log every this many iterations.
    """

    def __init__(
        self,
        variable_names:   List[str],
        init_log_path:    Optional[Path] = None,
        eval_log_path:    Optional[Path] = None,
        history_log_path: Optional[Path] = None,
        history_every:    int = 1,
    ):
        self._names        = variable_names
        self._init_path    = init_log_path
        self._eval_path    = eval_log_path
        self._hist_path    = history_log_path
        self._hist_every   = max(1, history_every)
        self._t0           = time.perf_counter()

        # Write CSV headers
        if self._init_path:
            self._write_csv(
                self._init_path,
                ["harmony_index"] + self._names + ["fitness", "penalty", "feasible"],
                mode="w",
            )
        if self._eval_path:
            self._write_csv(
                self._eval_path,
                ["wall_time_s", "iteration"] + self._names + ["fitness", "penalty", "feasible"],
                mode="w",
            )
        if self._hist_path:
            self._write_csv(
                self._hist_path,
                ["iteration", "best_fitness", "best_penalty", "feasible"]
                + ["best_" + n for n in self._names],
                mode="w",
            )

    # --- static helper ---

    @staticmethod
    def _write_csv(path: Path, row: list, mode: str = "a") -> None:
        with open(path, mode, newline="") as f:
            csv.writer(f).writerow(row)

    # --- public API --------------------------------------------------------

    def log_init(
        self,
        harmonies: List[Harmony],
        fitnesses: List[Fitness],
        penalties: List[Penalty],
    ) -> None:
        """Write the initial harmony memory to the init log."""
        if not self._init_path:
            return
        for idx, (h, f, p) in enumerate(zip(harmonies, fitnesses, penalties)):
            row = (
                [idx]
                + [h.get(n) for n in self._names]
                + [f, p, int(p <= 0)]
            )
            self._write_csv(self._init_path, row)

    def log_evaluation(
        self,
        iteration: int,
        harmony:   Harmony,
        fitness:   Fitness,
        penalty:   Penalty,
    ) -> None:
        """Append one evaluated harmony to the evaluation log."""
        if not self._eval_path:
            return
        elapsed = time.perf_counter() - self._t0
        row = (
            [f"{elapsed:.4f}", iteration]
            + [harmony.get(n) for n in self._names]
            + [fitness, penalty, int(penalty <= 0)]
        )
        self._write_csv(self._eval_path, row)

    def log_iteration(
        self,
        iteration:    int,
        best_harmony: Harmony,
        best_fitness: Fitness,
        best_penalty: Penalty,
    ) -> None:
        """Append best-so-far to the history log (respects *history_every*)."""
        if not self._hist_path:
            return
        if iteration % self._hist_every != 0:
            return
        row = (
            [iteration, best_fitness, best_penalty, int(best_penalty <= 0)]
            + [best_harmony.get(n) for n in self._names]
        )
        self._write_csv(self._hist_path, row)