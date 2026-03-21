"""
variables.py
============
Design variable types for the Harmony Search framework.

Every variable must implement three methods:

    sample(ctx)              -> value
        Draw a random value from the full domain given the current
        dependency context.

    filter(candidates, ctx)  -> List[value]
        From a list of candidate values (taken from harmony memory),
        return only those that are valid under the current context.
        Used during the HMCR memory-consideration step.

    neighbor(value, ctx)     -> value
        Return a value adjacent to *value* within the current domain.
        Called only when PAR fires; must always return a feasible value.

The *ctx* argument is always a plain ``dict`` mapping previously-assigned
variable names to their current values.  Variables that do not depend on
others simply ignore it.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

Context = Dict[str, Any]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Variable(ABC):
    """Abstract base class for all design variables."""

    @abstractmethod
    def sample(self, ctx: Context) -> Any:
        """Draw a random feasible value given *ctx*."""

    @abstractmethod
    def filter(self, candidates: List[Any], ctx: Context) -> List[Any]:
        """Return the subset of *candidates* that is feasible given *ctx*."""

    @abstractmethod
    def neighbor(self, value: Any, ctx: Context) -> Any:
        """Return a neighbor of *value* that is feasible given *ctx*."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frange(lo: float, step: float, hi: float) -> List[float]:
    """
    Inclusive float range [lo, lo+step, …, hi].

    Values are rounded to 10 decimal places to avoid floating-point
    accumulation errors.  The endpoint is always appended explicitly so
    it is never missed when (hi - lo) is not an exact multiple of *step*.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}.")
    values: List[float] = []
    v = lo
    while v <= hi + 1e-9:
        values.append(round(v, 10))
        v += step
    # Guarantee the endpoint is present exactly once
    if not values or abs(values[-1] - hi) > 1e-9:
        values.append(round(hi, 10))
    return values


def _in_grid(value: float, grid: List[float], tol: float = 1e-9) -> bool:
    """Return True when *value* is within *tol* of any element in *grid*."""
    return any(abs(value - g) <= tol for g in grid)


# ---------------------------------------------------------------------------
# Continuous variable  [lo, hi]
# ---------------------------------------------------------------------------

class Continuous(Variable):
    """
    Uniformly distributed real-valued variable.

    Bounds may be fixed scalars or callables, enabling dynamic (dependent)
    bounds that resolve at sample-time from the current context.

    Parameters
    ----------
    lo, hi : float | callable
        Lower and upper bounds.  Pass a lambda for dependent bounds::

            Continuous(lo=lambda ctx: ctx["b"], hi=2.0)

    Examples
    --------
    >>> v = Continuous(0.1, 2.0)
    >>> 0.1 <= v.sample({}) <= 2.0
    True
    """

    def __init__(self, lo, hi):
        # Validate static bounds early (callable bounds checked at sample time)
        if not callable(lo) and not callable(hi):
            if float(lo) > float(hi):
                raise ValueError(
                    f"Continuous: lo ({lo}) must be <= hi ({hi})."
                )
        self._lo = lo
        self._hi = hi

    def _bounds(self, ctx: Context):
        lo = self._lo(ctx) if callable(self._lo) else self._lo
        hi = self._hi(ctx) if callable(self._hi) else self._hi
        return float(lo), float(hi)

    def sample(self, ctx: Context) -> float:
        lo, hi = self._bounds(ctx)
        return random.uniform(lo, hi)

    def filter(self, candidates: List[float], ctx: Context) -> List[float]:
        lo, hi = self._bounds(ctx)
        return [v for v in candidates if lo <= v <= hi]

    def neighbor(self, value: float, ctx: Context) -> float:
        """
        Perturb *value* by a Gaussian step, clamped to [lo, hi].

        Step size
        ---------
        σ = bw × (hi − lo)

        where *bw* (bandwidth) is read from ``ctx["__bw__"]`` when present,
        and defaults to 0.05 (5 % of domain width) otherwise.

        The optimizer injects ``ctx["__bw__"]`` automatically when
        ``bw_max`` / ``bw_min`` are supplied to :meth:`optimize`.  For
        variables that do not use bandwidth (Discrete, Categorical, …) the
        key is simply ignored.
        """
        lo, hi = self._bounds(ctx)
        width = hi - lo
        if width <= 0:
            return value
        bw = ctx.get("__bw__", 0.05)
        new_val = value + random.gauss(0.0, bw * width)
        return max(lo, min(hi, new_val))


# ---------------------------------------------------------------------------
# Discrete variable  {lo, lo+step, …, hi}
# ---------------------------------------------------------------------------

class Discrete(Variable):
    """
    Variable taking values on a regular grid ``{lo, lo+step, …, hi}``.

    All three grid parameters may be fixed scalars or callables.

    Parameters
    ----------
    lo, step, hi : float | callable

    Examples
    --------
    >>> v = Discrete(0.0, 0.5, 2.0)
    >>> v.sample({}) in [0.0, 0.5, 1.0, 1.5, 2.0]
    True
    """

    def __init__(self, lo, step, hi):
        if not callable(lo) and not callable(hi) and not callable(step):
            if float(step) <= 0:
                raise ValueError(f"Discrete: step ({step}) must be positive.")
            if float(lo) > float(hi):
                raise ValueError(f"Discrete: lo ({lo}) must be <= hi ({hi}).")
        self._lo   = lo
        self._step = step
        self._hi   = hi

    def _grid(self, ctx: Context) -> List[float]:
        lo   = self._lo(ctx)   if callable(self._lo)   else self._lo
        step = self._step(ctx) if callable(self._step) else self._step
        hi   = self._hi(ctx)   if callable(self._hi)   else self._hi
        return _frange(float(lo), float(step), float(hi))

    def sample(self, ctx: Context) -> float:
        grid = self._grid(ctx)
        return random.choice(grid) if grid else None

    def filter(self, candidates: List[float], ctx: Context) -> List[float]:
        # Use tolerance-based membership to avoid floating-point mismatches
        grid = self._grid(ctx)
        return [v for v in candidates if _in_grid(v, grid)]

    def neighbor(self, value: float, ctx: Context) -> float:
        """Move one step left or right on the grid."""
        grid = self._grid(ctx)
        # Find nearest grid index with tolerance
        nearest = min(range(len(grid)), key=lambda i: abs(grid[i] - value))
        if abs(grid[nearest] - value) > 1e-9:
            return value   # value not on grid; return unchanged
        candidates = []
        if nearest > 0:
            candidates.append(grid[nearest - 1])
        if nearest < len(grid) - 1:
            candidates.append(grid[nearest + 1])
        return random.choice(candidates) if candidates else value


# ---------------------------------------------------------------------------
# Categorical variable  {a, b, c, …}
# ---------------------------------------------------------------------------

class Categorical(Variable):
    """
    Variable taking values from an unordered finite set.

    The neighbor of a categorical value is another randomly chosen member
    (there is no meaningful adjacency for nominal data).

    Parameters
    ----------
    choices : sequence

    Examples
    --------
    >>> v = Categorical(["S235", "S275", "S355"])
    >>> v.sample({}) in ["S235", "S275", "S355"]
    True
    """

    def __init__(self, choices: Sequence):
        if not choices:
            raise ValueError("Categorical: choices must not be empty.")
        self._choices = list(choices)

    def sample(self, ctx: Context) -> Any:
        return random.choice(self._choices)

    def filter(self, candidates: List[Any], ctx: Context) -> List[Any]:
        valid = set(self._choices)
        return [v for v in candidates if v in valid]

    def neighbor(self, value: Any, ctx: Context) -> Any:
        others = [c for c in self._choices if c != value]
        return random.choice(others) if others else value


# ---------------------------------------------------------------------------
# Integer variable  {lo, lo+1, …, hi}
# ---------------------------------------------------------------------------

class Integer(Variable):
    """
    Integer-valued variable over the closed interval ``[lo, hi]``.

    Convenience wrapper around :class:`Discrete` with ``step=1``.

    Parameters
    ----------
    lo, hi : int | callable
        Inclusive bounds.
    """

    def __init__(self, lo, hi):
        self._inner = Discrete(lo, 1, hi)

    def sample(self, ctx: Context) -> int:
        v = self._inner.sample(ctx)
        return int(round(v)) if v is not None else None

    def filter(self, candidates: List[int], ctx: Context) -> List[int]:
        return [int(round(v)) for v in self._inner.filter(
            [float(c) for c in candidates], ctx
        )]

    def neighbor(self, value: int, ctx: Context) -> int:
        return int(round(self._inner.neighbor(float(value), ctx)))
