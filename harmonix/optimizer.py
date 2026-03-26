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
import tempfile
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

# Hard cap to avoid accidental (or malicious) unbounded iteration counts.
MAX_ITER_CAP: int = 200_000


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

    def _improvise(
        self,
        hmcr: float,
        par: float,
        bw: float = 0.05,
        harmonies: Optional[List[Harmony]] = None,
    ) -> Harmony:
        """
        Single improvisation step (all variables, in definition order).

        For each variable:
        1. With probability HMCR — memory consideration:
           collect values from source (memory or provided harmonies) → filter to
           context-feasible ones → pick one → with probability PAR apply pitch adjustment.
        2. With probability (1 − HMCR) — random selection.

        Parameters
        ----------
        hmcr : float
        par : float
        bw : float
        harmonies : list of dict, optional
            Alternative source of harmonies (e.g. from Pareto archive).
            Defaults to ``self._memory.harmonies``.
        """
        ctx: Harmony = {"__bw__": bw}
        source = harmonies if harmonies is not None else (self._memory.harmonies if self._memory else [])

        for name, var in self.space.items():
            if random.random() < hmcr and source:
                candidates = [h[name] for h in source]
                valid = var.filter(candidates, ctx=ctx)
                if valid:
                    base = random.choice(valid)
                    ctx[name] = var.neighbor(base, ctx=ctx) if random.random() < par else base
                    continue
            ctx[name] = var.sample(ctx=ctx)

        ctx.pop("__bw__", None)
        return ctx

    def _improvise_from_archive(self, hmcr: float, par: float, archive: ParetoArchive, bw: float = 0.05) -> Harmony:
        """Improvisation that draws base values from the Pareto archive."""
        # Multi-objective entries have a .harmony attribute.
        archive_harmonies = [e.harmony for e in archive.entries]
        return self._improvise(hmcr, par, bw, harmonies=archive_harmonies)

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

    def _parse_optimize_kwargs(self, default_memory_size: int = 20, **kwargs: Any) -> Dict[str, Any]:
        """
        Extract and validate common optimizer parameters.

        Parameters
        ----------
        default_memory_size : int
        **kwargs : Any

        Returns
        -------
        params : dict
        """
        params: Dict[str, Any] = {
            "memory_size": int(kwargs.pop("memory_size", default_memory_size)),
            "hmcr": float(kwargs.pop("hmcr", 0.85)),
            "par": float(kwargs.pop("par", 0.35)),
            "max_iter": int(kwargs.pop("max_iter", 5000)),
            "bw_max": float(kwargs.pop("bw_max", 0.05)),
            "bw_min": float(kwargs.pop("bw_min", 0.001)),
            "resume": str(kwargs.pop("resume", "auto")),
            "checkpoint_every": int(kwargs.pop("checkpoint_every", 500)),
            "use_cache": bool(kwargs.pop("use_cache", False)),
            "cache_maxsize": int(kwargs.pop("cache_maxsize", 4096)),
            "log_init": bool(kwargs.pop("log_init", False)),
            "log_evaluations": bool(kwargs.pop("log_evaluations", False)),
            "log_history": bool(kwargs.pop("log_history", False)),
            "history_every": int(kwargs.pop("history_every", 1)),
            "archive_size": int(kwargs.pop("archive_size", 100)),
            "callback": kwargs.pop("callback", None),
            "verbose": bool(kwargs.pop("verbose", False)),
        }

        # Handle Paths
        for key in ["checkpoint_path", "init_log_path", "eval_log_path", "history_log_path"]:
            val = kwargs.pop(key, None)
            params[key] = Path(val) if val is not None else None

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        # Basic validation / capping
        if params["max_iter"] < 0:
            raise ValueError(f"max_iter must be non-negative; got {params['max_iter']}.")
        params["max_iter"] = min(params["max_iter"], MAX_ITER_CAP)
        params["checkpoint_every"] = max(1, params["checkpoint_every"])

        return params

    # --- run setup ---------------------------------------------------------

    def _decide_should_resume(self, *, ckpt: Optional[Path], resume: str) -> bool:
        if resume == "resume":
            if ckpt is None or not ckpt.exists():
                raise FileNotFoundError(f"resume='resume' but checkpoint not found: {ckpt}")
            return True
        if resume == "auto":
            return ckpt is not None and ckpt.exists() and ckpt.stat().st_size > 0
        if resume == "new":
            return False
        raise ValueError(f"resume must be 'auto', 'new', or 'resume'; got {resume!r}.")

    def _wrap_objective(self, *, use_cache: bool, cache_maxsize: int) -> Callable:
        effective_obj: Callable = self.objective
        if use_cache:
            self._cache = EvaluationCache(self.objective, maxsize=cache_maxsize)
            effective_obj = self._cache
        else:
            self._cache = None
        return effective_obj

    def _resolve_logger_paths(
        self,
        *,
        ckpt: Optional[Path],
        log_init: bool,
        init_log_path: Optional[Path],
        log_evaluations: bool,
        eval_log_path: Optional[Path],
        log_history: bool,
        history_log_path: Optional[Path],
    ) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        init_path = _resolve_path(init_log_path, ckpt, "_init") if log_init else None
        eval_path = _resolve_path(eval_log_path, ckpt, "_evals") if log_evaluations else None
        hist_path = _resolve_path(history_log_path, ckpt, "_history") if log_history else None
        return init_path, eval_path, hist_path

    def _init_or_resume_memory(
        self,
        *,
        should_resume: bool,
        memory_size: int,
        mode: str,
        ckpt: Optional[Path],
        effective_obj: Callable,
    ) -> int:
        if should_resume:
            return self.load_checkpoint(ckpt)  # type: ignore[arg-type]

        self._memory = HarmonyMemory(size=memory_size, mode=mode)
        for _ in range(memory_size):
            h = self.space.sample_harmony()
            f, p = effective_obj(h)
            self._memory.add(h, float(f), float(p))
        return 0

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
        should_resume = self._decide_should_resume(ckpt=ckpt, resume=resume)
        effective_obj = self._wrap_objective(use_cache=use_cache, cache_maxsize=cache_maxsize)
        start_iter = self._init_or_resume_memory(
            should_resume=should_resume,
            memory_size=memory_size,
            mode=mode,
            ckpt=ckpt,
            effective_obj=effective_obj,
        )

        # --- resolve log paths ---
        init_path, eval_path, hist_path = self._resolve_logger_paths(
            ckpt=ckpt,
            log_init=log_init,
            init_log_path=init_log_path,
            log_evaluations=log_evaluations,
            eval_log_path=eval_log_path,
            log_history=log_history,
            history_log_path=history_log_path,
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

    def _perform_iteration(
        self,
        *,
        it: int,
        start_iter: int,
        max_iter: int,
        bw_max: float,
        bw_min: float,
        hmcr: float,
        par: float,
        effective_obj: Callable,
        logger: RunLogger,
        history: List[Tuple[Fitness, Penalty]],
        verbose: bool,
        callback: Optional[Callable],
        ckpt: Optional[Path],
        checkpoint_every: int,
        t0: float,
    ) -> None:
        """Single step of the improvisation loop."""
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

    # --- checkpoint --------------------------------------------------------

    def save_checkpoint(self, path: Path, iteration: int) -> None:
        """Serialise memory and iteration counter to a JSON file."""
        payload = {
            "iteration": iteration,
            "memory": self._memory.to_dict() if self._memory else None,
        }
        # Path resolution and validation for security compliance (CWE-22)
        target_path = Path(path).resolve()
        base_path = Path.cwd().resolve()
        try:
            temp_base = Path(tempfile.gettempdir()).resolve()
        except Exception:
            temp_base = base_path

        def _is_relative_robust(p: Path, base: Path) -> bool:
            try:
                # Use string prefix check for cross-platform/case-insensitive robustness
                return str(p).lower().startswith(str(base).lower())
            except Exception:
                return False

        if not (_is_relative_robust(target_path, base_path) or _is_relative_robust(target_path, temp_base)):
            # Fallback to current directory for safety if traversal attempted elsewhere
            target_path = base_path / target_path.name

        target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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

    def optimize(self, **kwargs: Any) -> OptimizationResult:
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
            ``"auto"`` / ``"new"`` / ``"resume"``.
        checkpoint_path : Path, optional
        checkpoint_every : int
        use_cache : bool
        cache_maxsize : int
        log_init, log_evaluations, log_history : bool
        init_log_path, eval_log_path, history_log_path : Path, optional
        history_every : int
        callback : callable, optional
        verbose : bool
        """
        params = self._parse_optimize_kwargs(default_memory_size=20, **kwargs)

        start_iter, logger, effective_obj = self._setup_run(
            memory_size=params["memory_size"],
            mode="min",
            checkpoint_path=params["checkpoint_path"],
            resume=params["resume"],
            use_cache=params["use_cache"],
            cache_maxsize=params["cache_maxsize"],
            log_init=params["log_init"],
            init_log_path=params["init_log_path"],
            log_evaluations=params["log_evaluations"],
            eval_log_path=params["eval_log_path"],
            log_history=params["log_history"],
            history_log_path=params["history_log_path"],
            history_every=params["history_every"],
        )
        start_iter = max(0, int(start_iter))
        max_iter = params["max_iter"]
        if start_iter > max_iter:
            start_iter = max_iter
        if params["verbose"] and start_iter > 0:
            print(f"[HS] Resumed from checkpoint at iteration {start_iter}.")

        history: List[Tuple[Fitness, Penalty]] = []
        t0 = time.perf_counter()
        ckpt = params["checkpoint_path"]

        try:
            for it in range(start_iter, max_iter):
                self._perform_iteration(
                    it=it,
                    start_iter=start_iter,
                    max_iter=max_iter,
                    bw_max=params["bw_max"],
                    bw_min=params["bw_min"],
                    hmcr=params["hmcr"],
                    par=params["par"],
                    effective_obj=effective_obj,
                    logger=logger,
                    history=history,
                    verbose=params["verbose"],
                    callback=params["callback"],
                    ckpt=ckpt,
                    checkpoint_every=params["checkpoint_every"],
                    t0=t0,
                )
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

    def optimize(self, **kwargs: Any) -> OptimizationResult:
        """Run maximisation (see :meth:`Minimization.optimize` for parameter docs)."""
        params = self._parse_optimize_kwargs(default_memory_size=20, **kwargs)
        callback = params.pop("callback", None)

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
        result = inner.optimize(callback=wrapped_callback, **params)
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

    def _mo_init_from_checkpoint(
        self,
        *,
        ckpt: Path,
        verbose: bool,
    ) -> Tuple[int, HarmonyMemory, ParetoArchive]:
        payload = json.loads(ckpt.read_text())
        start_iter = int(payload["iteration"])
        memory = HarmonyMemory.from_dict(payload["memory"])
        archive = ParetoArchive.from_dict(payload["archive"])
        if verbose:
            print(f"[MO-HS] Resumed from checkpoint at iteration {start_iter}.")
        return start_iter, memory, archive

    def _mo_init_fresh(
        self,
        *,
        memory_size: int,
        archive_size: int,
        effective_obj: Callable,
        log_init: bool,
        logger: RunLogger,
        ckpt: Optional[Path],
        verbose: bool,
    ) -> Tuple[int, HarmonyMemory, ParetoArchive]:
        memory = HarmonyMemory(size=memory_size, mode="min")
        archive = ParetoArchive(max_size=archive_size)

        for _ in range(memory_size):
            h = self.space.sample_harmony()
            objs, p = effective_obj(h)
            objs = tuple(float(v) for v in objs)
            p = float(p)
            memory.add(h, objs[0], p)
            if p <= 0:
                archive.add(h, objs)

        # Log init memory to CSV.
        if log_init:
            logger.log_init(
                harmonies=memory.harmonies,
                fitnesses=memory._fitness,
                penalties=memory._penalty,
            )

        # Write checkpoint after initialisation.
        if ckpt:
            payload0 = {
                "iteration": 0,
                "memory": memory.to_dict(),
                "archive": archive.to_dict(),
            }
            ckpt.write_text(json.dumps(payload0, indent=2))

        _ = verbose  # reserved for future init logging
        return 0, memory, archive

    def _mo_init_state(
        self,
        *,
        ckpt: Optional[Path],
        should_resume: bool,
        memory_size: int,
        archive_size: int,
        effective_obj: Callable,
        log_init: bool,
        logger: RunLogger,
        verbose: bool,
    ) -> Tuple[int, HarmonyMemory, ParetoArchive]:
        if should_resume:
            assert ckpt is not None
            start_iter, memory, archive = self._mo_init_from_checkpoint(ckpt=ckpt, verbose=verbose)
        else:
            start_iter, memory, archive = self._mo_init_fresh(
                memory_size=memory_size,
                archive_size=archive_size,
                effective_obj=effective_obj,
                log_init=log_init,
                logger=logger,
                ckpt=ckpt,
                verbose=verbose,
            )
        self._memory = memory
        return start_iter, memory, archive

    def _mo_improvise_new_h(
        self,
        *,
        archive: ParetoArchive,
        hmcr: float,
        par: float,
        bw: float,
    ) -> Harmony:
        if archive.entries:
            return self._improvise_from_archive(hmcr, par, archive, bw)
        return self._improvise(hmcr, par, bw)

    def _mo_evaluate_new_h(
        self,
        *,
        effective_obj: Callable,
        new_h: Harmony,
    ) -> Tuple[Tuple[float, ...], float]:
        objs, p = effective_obj(new_h)
        objs = tuple(float(v) for v in objs)
        p = float(p)
        return objs, p

    def _mo_update_memory_and_archive(
        self,
        *,
        new_h: Harmony,
        objs: Tuple[float, ...],
        p: float,
        archive: ParetoArchive,
    ) -> None:
        self._memory.try_replace_worst(new_h, objs[0], p)
        if p <= 0:
            archive.add(new_h, objs)

    def _mo_maybe_log_and_callback(
        self,
        *,
        logger: RunLogger,
        it_plus_1: int,
        new_h: Harmony,
        obj0: float,
        p: float,
        archive: ParetoArchive,
        archive_history: List[int],
        callback: Optional[Callable[[int, ParetoResult], None]],
        verbose: bool,
        t0: float,
    ) -> None:
        logger.log_evaluation(it_plus_1, new_h, obj0, p)

        if verbose:
            print(f"[MO-HS] iter {it_plus_1:>6d} | archive = {len(archive):>4d} solutions")

        if callback is not None:
            partial = ParetoResult(
                front=archive.front(),
                archive_history=archive_history,
                iterations=it_plus_1,
                elapsed_seconds=time.perf_counter() - t0,
            )
            callback(it_plus_1, partial)

    def _mo_maybe_checkpoint(
        self,
        *,
        ckpt: Optional[Path],
        checkpoint_every: int,
        it_plus_1: int,
        archive: ParetoArchive,
    ) -> None:
        if ckpt and (it_plus_1 % checkpoint_every == 0):
            payload_it = {
                "iteration": it_plus_1,
                "memory": self._memory.to_dict(),
                "archive": archive.to_dict(),
            }
            ckpt.write_text(json.dumps(payload_it, indent=2))

    def _mo_iteration_update(
        self,
        *,
        it_info: Dict[str, Any],
        params: Dict[str, Any],
        archive: ParetoArchive,
        effective_obj: Callable,
        logger: RunLogger,
        archive_history: List[int],
        t0: float,
    ) -> None:
        it = it_info["it"]
        start_iter = it_info["start_iter"]
        max_iter = it_info["max_iter"]

        bw = self._compute_bw(it - start_iter, max_iter - start_iter, params["bw_max"], params["bw_min"])
        new_h = self._mo_improvise_new_h(archive=archive, hmcr=params["hmcr"], par=params["par"], bw=bw)
        objs, p = self._mo_evaluate_new_h(effective_obj=effective_obj, new_h=new_h)

        self._mo_update_memory_and_archive(new_h=new_h, objs=objs, p=p, archive=archive)
        archive_history.append(len(archive))

        it_plus_1 = it + 1
        self._mo_maybe_log_and_callback(
            logger=logger,
            it_plus_1=it_plus_1,
            new_h=new_h,
            obj0=objs[0],
            p=p,
            archive=archive,
            archive_history=archive_history,
            callback=params.get("callback"),
            verbose=params.get("verbose", False),
            t0=t0,
        )
        self._mo_maybe_checkpoint(
            ckpt=params.get("ckpt"),
            checkpoint_every=params.get("checkpoint_every", 500),
            it_plus_1=it_plus_1,
            archive=archive,
        )

    def optimize(self, **kwargs: Any) -> ParetoResult:
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
        use_cache, cache_maxsize : optional
        log_init, log_evaluations, log_history : bool
        history_every : int
        callback : callable, optional
        verbose : bool
        """
        params = self._parse_optimize_kwargs(default_memory_size=30, **kwargs)

        effective_obj = self._wrap_objective(use_cache=params["use_cache"], cache_maxsize=params["cache_maxsize"])
        ckpt = params["checkpoint_path"]

        init_path, eval_path, hist_path = self._resolve_logger_paths(
            ckpt=ckpt,
            log_init=params["log_init"],
            init_log_path=params["init_log_path"],
            log_evaluations=params["log_evaluations"],
            eval_log_path=params["eval_log_path"],
            log_history=params["log_history"],
            history_log_path=params["history_log_path"],
        )
        logger = RunLogger(
            variable_names=self.space.names(),
            init_log_path=init_path,
            eval_log_path=eval_path,
            history_log_path=hist_path,
            history_every=params["history_every"],
        )

        should_resume = self._decide_should_resume(ckpt=ckpt, resume=params["resume"])

        start_iter, _, archive = self._mo_init_state(
            ckpt=ckpt,
            should_resume=should_resume,
            memory_size=params["memory_size"],
            archive_size=params["archive_size"],
            effective_obj=effective_obj,
            log_init=params["log_init"],
            logger=logger,
            verbose=params["verbose"],
        )

        start_iter = max(0, int(start_iter))
        max_iter = params["max_iter"]
        if start_iter > max_iter:
            start_iter = max_iter

        archive_history: List[int] = []
        t0 = time.perf_counter()

        it_info = {"start_iter": start_iter, "max_iter": max_iter}
        # Simplify params mapping for inner method
        params["ckpt"] = ckpt

        try:
            for it in range(start_iter, max_iter):
                it_info["it"] = it
                self._mo_iteration_update(
                    it_info=it_info,
                    params=params,
                    archive=archive,
                    effective_obj=effective_obj,
                    logger=logger,
                    archive_history=archive_history,
                    t0=t0,
                )
        except StopIteration:
            pass

        return ParetoResult(
            front=archive.front(),
            archive_history=archive_history,
            iterations=len(archive_history),
            elapsed_seconds=time.perf_counter() - t0,
        )
