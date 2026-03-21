"""
optimizer.py
============
Harmony Search optimisers: Minimization, Maximization, MultiObjective.

References
----------
Geem, Z. W., Kim, J. H., & Loganathan, G. V. (2001).
    A new heuristic optimization algorithm: Harmony search.
    *Simulation*, 76(2), 60–68.

Lee, K. S., & Geem, Z. W. (2005).
    A new meta-heuristic algorithm for continuous engineering optimization.
    *Computer Methods in Applied Mechanics and Engineering*, 194(36–38), 3902–3933.

Ricart, J., Hüttemann, G., Lima, J., & Barán, B. (2011).
    Multiobjective harmony search algorithm proposals.
    *Electronic Notes in Theoretical Computer Science*, 281, 51–67.
"""

from __future__ import annotations

import json
import math
import random
import time
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging import EvaluationCache, RunLogger, _resolve_path
from .pareto import ParetoArchive, ParetoResult
from .space import DesignSpace

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Harmony = Dict[str, Any]
Fitness = float
Penalty = float


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """
    Result of a single-objective optimisation run.

    Attributes
    ----------
    best_harmony : dict
    best_fitness : float
    best_penalty : float
    iterations : int
    elapsed_seconds : float
    history : list of (fitness, penalty)
        Best recorded at each iteration.
    """

    best_harmony: Harmony
    best_fitness: Fitness
    best_penalty: Penalty
    iterations: int
    elapsed_seconds: float
    history: List[Tuple[Fitness, Penalty]] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [
            "OptimizationResult(",
            f"  best_fitness  = {self.best_fitness:.6g}",
            f"  best_penalty  = {self.best_penalty:.6g}",
            f"  iterations    = {self.iterations}",
            f"  elapsed       = {self.elapsed_seconds:.2f}s",
            "  best_harmony  = {",
        ]
        for k, v in self.best_harmony.items():
            lines.append(f"    {k!r}: {v!r},")
        lines += ["  }", ")"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Harmony memory
# ---------------------------------------------------------------------------


class HarmonyMemory:
    """
    Fixed-size pool of harmonies with Deb-style constraint handling.

    Replacement policy
    ~~~~~~~~~~~~~~~~~~
    1. Feasible candidate replaces any infeasible worst.
    2. Among infeasible: smaller penalty wins.
    3. Among feasible: smaller (min) or larger (max) fitness wins.
    """

    def __init__(self, size: int, mode: str = "min"):
        self.size = size
        self.mode = mode
        self._harmonies: List[Harmony] = []
        self._fitness: List[Fitness] = []
        self._penalty: List[Penalty] = []

    @property
    def harmonies(self) -> List[Harmony]:
        return self._harmonies

    def __len__(self) -> int:
        return len(self._harmonies)

    def add(self, harmony: Harmony, fitness: Fitness, penalty: Penalty) -> None:
        self._harmonies.append(harmony)
        self._fitness.append(fitness)
        self._penalty.append(penalty)

    def _dominates(self, idx_a: int, idx_b: int) -> bool:
        """True when solution *a* is strictly better than *b*."""
        pa, pb = self._penalty[idx_a], self._penalty[idx_b]
        fa, fb = self._fitness[idx_a], self._fitness[idx_b]
        if pa <= 0 and pb > 0:
            return True
        if pa > 0 and pb <= 0:
            return False
        if pa > 0 and pb > 0:
            return pa < pb
        return fa < fb if self.mode == "min" else fa > fb

    def best_index(self) -> int:
        best = 0
        for i in range(1, len(self._harmonies)):
            if self._dominates(i, best):
                best = i
        return best

    def worst_index(self) -> int:
        worst = 0
        for i in range(1, len(self._harmonies)):
            if self._dominates(worst, i):
                worst = i
        return worst

    def best(self) -> Tuple[Harmony, Fitness, Penalty]:
        idx = self.best_index()
        return self._harmonies[idx], self._fitness[idx], self._penalty[idx]

    def try_replace_worst(self, harmony: Harmony, fitness: Fitness, penalty: Penalty) -> bool:
        """Replace worst if candidate dominates it.  Returns True on success."""
        w = self.worst_index()
        pw = self._penalty[w]
        fw = self._fitness[w]
        if penalty <= 0 and pw > 0:
            replace = True
        elif penalty > 0 and pw <= 0:
            replace = False
        elif penalty > 0 and pw > 0:
            replace = penalty < pw
        else:
            replace = fitness < fw if self.mode == "min" else fitness > fw
        if replace:
            self._harmonies[w] = harmony
            self._fitness[w] = fitness
            self._penalty[w] = penalty
        return replace

    # --- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "mode": self.mode,
            "harmonies": self._harmonies,
            "fitness": self._fitness,
            "penalty": self._penalty,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HarmonyMemory":
        mem = cls(size=data["size"], mode=data["mode"])
        mem._harmonies = data["harmonies"]
        mem._fitness = data["fitness"]
        mem._penalty = data["penalty"]
        return mem


# ---------------------------------------------------------------------------
# Base optimiser
# ---------------------------------------------------------------------------


class HarmonySearchOptimizer(ABC):
    """Abstract base for all Harmony Search optimisers."""

    def __init__(self, space: DesignSpace, objective: Callable):
        self.space = space
        self.objective = objective
        self._memory: Optional[HarmonyMemory] = None

    # --- core HS operators -------------------------------------------------

    def _improvise(self, hmcr: float, par: float, bw: float = 0.05) -> Harmony:
        """
        Single improvisation step (all variables, in definition order).

        For each variable:
        1. With probability HMCR — memory consideration:
           collect values from memory → filter to context-feasible ones
           → pick one → with probability PAR apply pitch adjustment.
        2. With probability (1 − HMCR) — random selection.

        Parameters
        ----------
        hmcr : float
            Harmony Memory Considering Rate.
        par : float
            Pitch Adjusting Rate.
        bw : float
            Current bandwidth for continuous pitch adjustment.
            Injected into ``ctx["__bw__"]`` so :class:`~variables.Continuous`
            can read it without changing the ``Variable`` interface.
        """
        ctx: Harmony = {"__bw__": bw}
        for name, var in self.space.items():
            if random.random() < hmcr:
                candidates = [h[name] for h in self._memory.harmonies]
                valid = var.filter(candidates, ctx=ctx)
                if valid:
                    base = random.choice(valid)
                    ctx[name] = var.neighbor(base, ctx=ctx) if random.random() < par else base
                    continue
            # Random selection (fallback when HMCR misses or no valid candidates)
            ctx[name] = var.sample(ctx=ctx)
        # Remove internal key before returning
        ctx.pop("__bw__", None)
        return ctx

    def _improvise_from_archive(self, hmcr: float, par: float, archive: ParetoArchive, bw: float = 0.05) -> Harmony:
        """
        Improvisation that draws base values from the Pareto archive.

        Values are filtered to those that are context-feasible at each step,
        exactly as in :meth:`_improvise`.  If no archive value survives the
        filter, falls back to a fresh sample.
        """
        ctx: Harmony = {"__bw__": bw}
        for name, var in self.space.items():
            if random.random() < hmcr and archive.entries:
                archive_values = [e.harmony[name] for e in archive.entries]
                valid = var.filter(archive_values, ctx=ctx)
                if valid:
                    base = random.choice(valid)
                    ctx[name] = var.neighbor(base, ctx=ctx) if random.random() < par else base
                    continue
            ctx[name] = var.sample(ctx=ctx)
        ctx.pop("__bw__", None)
        return ctx

    def _compute_bw(
        self,
        iteration: int,
        max_iter: int,
        bw_max: float,
        bw_min: float,
    ) -> float:
        """
        Compute the current pitch-adjustment bandwidth using exponential decay.

        .. math::

            bw(t) = bw_max * exp(-ln(bw_max/bw_min) * t / T)

        where *t* is the current iteration and *T* is ``max_iter``.

        When ``bw_max == bw_min`` (or ``max_iter == 0``) the bandwidth is
        constant and equals ``bw_max``.

        Parameters
        ----------
        iteration : int
            Current iteration index (0-based).
        max_iter : int
            Total number of improvisation steps.
        bw_max : float
            Initial bandwidth (fraction of domain width).
        bw_min : float
            Final bandwidth (fraction of domain width).
        """
        if bw_max <= 0 or bw_min <= 0:
            raise ValueError("bw_max and bw_min must be positive.")
        if bw_min > bw_max:
            raise ValueError("bw_min must be <= bw_max.")
        if max_iter <= 1 or bw_max == bw_min:
            return bw_max
        return bw_max * math.exp(-math.log(bw_max / bw_min) * iteration / max_iter)

    # --- run setup ---------------------------------------------------------

    def _setup_run(
        self,
        *,
        memory_size: int,
        mode: str,
        checkpoint_path: Optional[Path],
        resume: str,
        use_cache: bool,
        cache_maxsize: int,
        log_init: bool,
        init_log_path: Optional[Path],
        log_evaluations: bool,
        eval_log_path: Optional[Path],
        log_history: bool,
        history_log_path: Optional[Path],
        history_every: int,
    ):
        """
        Initialise or resume memory, wrap the objective with cache/logger,
        and return (start_iter, logger).

        Resume logic (controlled by *resume*):

        ``"auto"``
            Resume if *checkpoint_path* exists and is non-empty, otherwise
            start fresh.
        ``"new"``
            Always start fresh.  An existing checkpoint file is overwritten.
        ``"resume"``
            Always resume.  Raises ``FileNotFoundError`` if the checkpoint
            file does not exist.
        """
        ckpt = Path(checkpoint_path) if checkpoint_path else None

        # --- decide whether to resume ---
        should_resume = False
        if resume == "resume":
            if ckpt is None or not ckpt.exists():
                raise FileNotFoundError(f"resume='resume' but checkpoint not found: {ckpt}")
            should_resume = True
        elif resume == "auto":
            should_resume = ckpt is not None and ckpt.exists() and ckpt.stat().st_size > 0
        elif resume == "new":
            should_resume = False
        else:
            raise ValueError(f"resume must be 'auto', 'new', or 'resume'; got {resume!r}.")

        # --- initialise or restore memory ---
        if should_resume:
            start_iter = self.load_checkpoint(ckpt)
        else:
            self._memory = HarmonyMemory(size=memory_size, mode=mode)
            start_iter = 0

        # --- wrap objective with cache ---
        effective_obj = self.objective
        if use_cache:
            self._cache = EvaluationCache(self.objective, maxsize=cache_maxsize)
            effective_obj = self._cache
        else:
            self._cache = None

        # --- populate memory if starting fresh ---
        if not should_resume:
            for _ in range(memory_size):
                h = self.space.sample_harmony()
                f, p = effective_obj(h)
                self._memory.add(h, float(f), float(p))

        # --- resolve log paths ---
        init_path = _resolve_path(Path(init_log_path) if init_log_path else None, ckpt, "_init") if log_init else None
        eval_path = (
            _resolve_path(Path(eval_log_path) if eval_log_path else None, ckpt, "_evals") if log_evaluations else None
        )
        hist_path = (
            _resolve_path(Path(history_log_path) if history_log_path else None, ckpt, "_history")
            if log_history
            else None
        )

        logger = RunLogger(
            variable_names=self.space.names(),
            init_log_path=init_path,
            eval_log_path=eval_path,
            history_log_path=hist_path,
            history_every=history_every,
        )

        # Write init log if starting fresh and logging requested
        if not should_resume and log_init:
            logger.log_init(
                harmonies=self._memory.harmonies,
                fitnesses=self._memory._fitness,
                penalties=self._memory._penalty,
            )

        # Save initial checkpoint immediately
        if ckpt and not should_resume:
            self.save_checkpoint(ckpt, 0)

        return start_iter, logger, effective_obj

    # --- checkpoint --------------------------------------------------------

    def save_checkpoint(self, path: Path, iteration: int) -> None:
        """Serialise memory and iteration counter to a JSON file."""
        payload = {
            "iteration": iteration,
            "memory": self._memory.to_dict() if self._memory else None,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def load_checkpoint(self, path: Path) -> int:
        """Restore memory from JSON checkpoint.  Returns last completed iteration."""
        payload = json.loads(Path(path).read_text())
        self._memory = HarmonyMemory.from_dict(payload["memory"])
        return int(payload["iteration"])

    def optimize(self, **kwargs: Any) -> Any:
        """Override in subclasses."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Minimization
# ---------------------------------------------------------------------------


class Minimization(HarmonySearchOptimizer):
    """
    Single-objective Harmony Search minimiser.

    Parameters
    ----------
    space : DesignSpace
    objective : callable
        ``objective(harmony) -> (fitness: float, penalty: float)``

        *penalty* ≤ 0 means feasible.

    Examples
    --------
    >>> opt = Minimization(space, objective)
    >>> result = opt.optimize(memory_size=20, hmcr=0.85, par=0.35, max_iter=5000)
    >>> print(result.best_fitness)
    """

    def optimize(  # type: ignore[override]
        self,
        *,
        memory_size: int = 20,
        hmcr: float = 0.85,
        par: float = 0.35,
        max_iter: int = 5000,
        bw_max: float = 0.05,
        bw_min: float = 0.001,
        resume: str = "auto",
        checkpoint_path: Optional[Path] = None,
        checkpoint_every: int = 500,
        use_cache: bool = False,
        cache_maxsize: int = 4096,
        log_init: bool = False,
        init_log_path: Optional[Path] = None,
        log_evaluations: bool = False,
        eval_log_path: Optional[Path] = None,
        log_history: bool = False,
        history_log_path: Optional[Path] = None,
        history_every: int = 1,
        callback: Optional[Callable[[int, OptimizationResult], None]] = None,
        verbose: bool = False,
    ) -> OptimizationResult:
        """
        Run the minimisation loop.

        Parameters
        ----------
        memory_size : int
            Harmony Memory Size (HMS).
        hmcr : float
            Harmony Memory Considering Rate ∈ (0, 1).
        par : float
            Pitch Adjusting Rate ∈ (0, 1).
        max_iter : int
            Total improvisation steps.
        bw_max : float
            Initial pitch-adjustment bandwidth (fraction of domain width).
        bw_min : float
            Final bandwidth.  Decays exponentially from *bw_max* to *bw_min*.
        resume : str
            ``"auto"``   — resume if checkpoint exists, else start fresh.
            ``"new"``    — always start fresh (overwrites existing checkpoint).
            ``"resume"`` — always resume; raises if checkpoint not found.
        checkpoint_path : Path, optional
            JSON file for crash recovery.
        checkpoint_every : int
            Save checkpoint every this many iterations.
        use_cache : bool
            Cache objective evaluations.
        cache_maxsize : int
            Maximum entries in the evaluation cache (LRU).
        log_init : bool
            Write the initial harmony memory to a CSV file once at startup.
        init_log_path : Path, optional
        log_evaluations : bool
            Write every evaluated harmony to a CSV file.
        eval_log_path : Path, optional
        log_history : bool
            Write best-so-far at each iteration to a CSV file.
        history_log_path : Path, optional
        history_every : int
        callback : callable, optional
            Raise :exc:`StopIteration` for early exit.
        verbose : bool
        """
        start_iter, logger, effective_obj = self._setup_run(
            memory_size=memory_size,
            mode="min",
            checkpoint_path=checkpoint_path,
            resume=resume,
            use_cache=use_cache,
            cache_maxsize=cache_maxsize,
            log_init=log_init,
            init_log_path=init_log_path,
            log_evaluations=log_evaluations,
            eval_log_path=eval_log_path,
            log_history=log_history,
            history_log_path=history_log_path,
            history_every=history_every,
        )
        if verbose and start_iter > 0:
            print(f"[HS] Resumed from checkpoint at iteration {start_iter}.")

        history: List[Tuple[Fitness, Penalty]] = []
        t0 = time.perf_counter()
        ckpt = Path(checkpoint_path) if checkpoint_path else None

        try:
            for it in range(start_iter, max_iter):
                bw = self._compute_bw(it - start_iter, max_iter - start_iter, bw_max, bw_min)
                new_h = self._improvise(hmcr, par, bw)
                new_f, new_p = effective_obj(new_h)
                new_f, new_p = float(new_f), float(new_p)
                self._memory.try_replace_worst(new_h, new_f, new_p)

                best_h, best_f, best_p = self._memory.best()
                history.append((best_f, best_p))
                logger.log_evaluation(it + 1, new_h, new_f, new_p)
                logger.log_iteration(it + 1, best_h, best_f, best_p)

                if verbose:
                    print(f"[HS] iter {it + 1:>6d} | fitness = {best_f:.6g} | penalty = {best_p:.4g}")

                if callback is not None:
                    partial = OptimizationResult(
                        best_harmony=best_h,
                        best_fitness=best_f,
                        best_penalty=best_p,
                        iterations=it + 1,
                        elapsed_seconds=time.perf_counter() - t0,
                        history=history,
                    )
                    callback(it + 1, partial)

                if ckpt and (it + 1) % checkpoint_every == 0:
                    self.save_checkpoint(ckpt, it + 1)

        except StopIteration:
            pass

        elapsed = time.perf_counter() - t0
        best_h, best_f, best_p = self._memory.best()
        return OptimizationResult(
            best_harmony=best_h,
            best_fitness=best_f,
            best_penalty=best_p,
            iterations=len(history),
            elapsed_seconds=elapsed,
            history=history,
        )


class Maximization(HarmonySearchOptimizer):
    """
    Single-objective Harmony Search maximiser.

    Internally negates fitness for memory bookkeeping; the reported
    ``best_fitness`` is always the original (un-negated) value.

    Parameters
    ----------
    space : DesignSpace
    objective : callable
        ``objective(harmony) -> (fitness: float, penalty: float)``
    """

    def optimize(  # type: ignore[override]
        self,
        *,
        memory_size: int = 20,
        hmcr: float = 0.85,
        par: float = 0.35,
        max_iter: int = 5000,
        bw_max: float = 0.05,
        bw_min: float = 0.001,
        resume: str = "auto",
        checkpoint_path: Optional[Path] = None,
        checkpoint_every: int = 500,
        use_cache: bool = False,
        cache_maxsize: int = 4096,
        log_init: bool = False,
        init_log_path: Optional[Path] = None,
        log_evaluations: bool = False,
        eval_log_path: Optional[Path] = None,
        log_history: bool = False,
        history_log_path: Optional[Path] = None,
        history_every: int = 1,
        callback: Optional[Callable[[int, OptimizationResult], None]] = None,
        verbose: bool = False,
    ) -> OptimizationResult:
        """Run maximisation (see :meth:`Minimization.optimize` for parameter docs)."""

        wrapped_callback = None
        if callback is not None:

            def wrapped_callback(it, partial):
                restored = OptimizationResult(
                    best_harmony=partial.best_harmony,
                    best_fitness=-partial.best_fitness,
                    best_penalty=partial.best_penalty,
                    iterations=partial.iterations,
                    elapsed_seconds=partial.elapsed_seconds,
                    history=[(-f, p) for f, p in partial.history],
                )
                callback(it, restored)

        def _negated(harmony):
            f, p = self.objective(harmony)
            return -float(f), float(p)

        inner = Minimization(self.space, _negated)
        result = inner.optimize(
            memory_size=memory_size,
            hmcr=hmcr,
            par=par,
            max_iter=max_iter,
            bw_max=bw_max,
            bw_min=bw_min,
            resume=resume,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            use_cache=use_cache,
            cache_maxsize=cache_maxsize,
            log_init=log_init,
            init_log_path=init_log_path,
            log_evaluations=log_evaluations,
            eval_log_path=eval_log_path,
            log_history=log_history,
            history_log_path=history_log_path,
            history_every=history_every,
            callback=wrapped_callback,
            verbose=verbose,
        )
        self._memory = inner._memory

        return OptimizationResult(
            best_harmony=result.best_harmony,
            best_fitness=-result.best_fitness,
            best_penalty=result.best_penalty,
            iterations=result.iterations,
            elapsed_seconds=result.elapsed_seconds,
            history=[(-f, p) for f, p in result.history],
        )


# ---------------------------------------------------------------------------
# MultiObjective
# ---------------------------------------------------------------------------


class MultiObjective(HarmonySearchOptimizer):
    """
    Multi-objective Harmony Search with Pareto archive.

    Maintains a bounded archive of non-dominated solutions.  New harmonies
    are improvised using archive entries (crowding-distance tournament),
    driving the search toward the Pareto front while preserving diversity.

    Constraint handling
    ~~~~~~~~~~~~~~~~~~~
    Only feasible harmonies (penalty ≤ 0) enter the archive.
    Infeasible harmonies remain in working memory so the search can learn
    from near-feasible regions.

    Parameters
    ----------
    space : DesignSpace
    objective : callable
        ``objective(harmony) -> (objectives: tuple[float, ...], penalty: float)``

        All objectives are minimised.  Negate maximised objectives inside
        the callable.

    Examples
    --------
    >>> def bi_obj(h):
    ...     f1 = h["x"]**2 + h["y"]**2
    ...     f2 = (h["x"] - 2)**2 + (h["y"] - 2)**2
    ...     return (f1, f2), 0.0
    >>>
    >>> result = MultiObjective(space, bi_obj).optimize(max_iter=5000)
    >>> for e in result.front[:3]:
    ...     print(e.objectives)
    """

    def optimize(  # type: ignore[override]
        self,
        *,
        memory_size: int = 30,
        hmcr: float = 0.85,
        par: float = 0.35,
        max_iter: int = 5000,
        bw_max: float = 0.05,
        bw_min: float = 0.001,
        archive_size: int = 100,
        resume: str = "auto",
        checkpoint_path: Optional[Path] = None,
        checkpoint_every: int = 500,
        use_cache: bool = False,
        cache_maxsize: int = 4096,
        log_init: bool = False,
        init_log_path: Optional[Path] = None,
        log_evaluations: bool = False,
        eval_log_path: Optional[Path] = None,
        log_history: bool = False,
        history_log_path: Optional[Path] = None,
        history_every: int = 1,
        callback: Optional[Callable[[int, ParetoResult], None]] = None,
        verbose: bool = False,
    ) -> ParetoResult:
        """
        Run the multi-objective loop.

        Parameters
        ----------
        memory_size : int
            HMS for the working population.
        hmcr, par, bw_max, bw_min : float
            Standard HS parameters.
        max_iter : int
            Improvisation steps.
        archive_size : int
            Maximum Pareto archive capacity (pruned by crowding distance).
        resume : str
            ``"auto"`` / ``"new"`` / ``"resume"`` — same as Minimization.
        checkpoint_path, checkpoint_every : optional
            JSON crash-recovery.
        use_cache, cache_maxsize : optional
            Evaluation cache settings.
        log_init, log_evaluations, log_history : bool
            Enable CSV logging for init memory, all evaluations, and
            per-iteration best.
        history_every : int
            Write to history log every this many iterations.
        callback : callable, optional
        verbose : bool
        """
        # --- setup cache and logger ---
        effective_obj = self.objective
        if use_cache:
            self._cache = EvaluationCache(self.objective, maxsize=cache_maxsize)
            effective_obj = self._cache
        else:
            self._cache = None

        ckpt = Path(checkpoint_path) if checkpoint_path else None

        # --- resolve log paths ---
        init_path = _resolve_path(Path(init_log_path) if init_log_path else None, ckpt, "_init") if log_init else None
        eval_path = (
            _resolve_path(Path(eval_log_path) if eval_log_path else None, ckpt, "_evals") if log_evaluations else None
        )
        hist_path = (
            _resolve_path(Path(history_log_path) if history_log_path else None, ckpt, "_history")
            if log_history
            else None
        )

        logger = RunLogger(
            variable_names=self.space.names(),
            init_log_path=init_path,
            eval_log_path=eval_path,
            history_log_path=hist_path,
            history_every=history_every,
        )

        # --- decide resume ---
        archive = ParetoArchive(max_size=archive_size)
        start_iter = 0

        should_resume = False
        if resume == "resume":
            if ckpt is None or not ckpt.exists():
                raise FileNotFoundError(f"resume='resume' but checkpoint not found: {ckpt}")
            should_resume = True
        elif resume == "auto":
            should_resume = ckpt is not None and ckpt.exists() and ckpt.stat().st_size > 0
        elif resume == "new":
            should_resume = False
        else:
            raise ValueError(f"resume must be 'auto', 'new', or 'resume'; got {resume!r}.")

        if should_resume:
            payload = json.loads(ckpt.read_text())
            start_iter = int(payload["iteration"])
            self._memory = HarmonyMemory.from_dict(payload["memory"])
            archive = ParetoArchive.from_dict(payload["archive"])
            if verbose:
                print(f"[MO-HS] Resumed from checkpoint at iteration {start_iter}.")
        else:
            self._memory = HarmonyMemory(size=memory_size, mode="min")
            for _ in range(memory_size):
                h = self.space.sample_harmony()
                objs, p = effective_obj(h)
                objs = tuple(float(v) for v in objs)
                p = float(p)
                self._memory.add(h, objs[0], p)
                if p <= 0:
                    archive.add(h, objs)

            if log_init:
                logger.log_init(
                    harmonies=self._memory.harmonies,
                    fitnesses=self._memory._fitness,
                    penalties=self._memory._penalty,
                )
            if ckpt:
                payload0 = {
                    "iteration": 0,
                    "memory": self._memory.to_dict(),
                    "archive": archive.to_dict(),
                }
                ckpt.write_text(json.dumps(payload0, indent=2))

        archive_history: List[int] = []
        t0 = time.perf_counter()

        try:
            for it in range(start_iter, max_iter):
                bw = self._compute_bw(it - start_iter, max_iter - start_iter, bw_max, bw_min)
                if archive.entries:
                    new_h = self._improvise_from_archive(hmcr, par, archive, bw)
                else:
                    new_h = self._improvise(hmcr, par, bw)

                objs, p = effective_obj(new_h)
                objs = tuple(float(v) for v in objs)
                p = float(p)

                self._memory.try_replace_worst(new_h, objs[0], p)
                if p <= 0:
                    archive.add(new_h, objs)

                archive_history.append(len(archive))
                logger.log_evaluation(it + 1, new_h, objs[0], p)

                if verbose:
                    print(f"[MO-HS] iter {it + 1:>6d} | archive = {len(archive):>4d} solutions")

                if callback is not None:
                    partial = ParetoResult(
                        front=archive.front(),
                        archive_history=archive_history,
                        iterations=it + 1,
                        elapsed_seconds=time.perf_counter() - t0,
                    )
                    callback(it + 1, partial)

                if ckpt and (it + 1) % checkpoint_every == 0:
                    payload_it = {
                        "iteration": it + 1,
                        "memory": self._memory.to_dict(),
                        "archive": archive.to_dict(),
                    }
                    ckpt.write_text(json.dumps(payload_it, indent=2))

        except StopIteration:
            pass

        return ParetoResult(
            front=archive.front(),
            archive_history=archive_history,
            iterations=len(archive_history),
            elapsed_seconds=time.perf_counter() - t0,
        )
