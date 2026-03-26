"""
harmonix.spaces.math
============================
Mathematical search spaces — variable types whose domains are defined
by classical number-theoretic or algebraic structures.

All classes follow the :class:`~harmonix.Variable` contract and
are pre-registered in the plugin registry under the names shown in the
table below.

+-------------------+-------------------+---------------------------------------+
| Class             | Registry name     | Domain                                |
+===================+===================+=======================================+
| ``NaturalNumber`` | ``natural``       | {1, 2, 3, …, hi}                      |
+-------------------+-------------------+---------------------------------------+
| ``WholeNumber``   | ``whole``         | {0, 1, 2, …, hi}                      |
+-------------------+-------------------+---------------------------------------+
| ``NegativeInt``   | ``negative_int``  | {lo, …, −1}                           |
+-------------------+-------------------+---------------------------------------+
| ``NegativeReal``  | ``negative_real`` | (lo, 0)  (lo < 0)                     |
+-------------------+-------------------+---------------------------------------+
| ``PositiveReal``  | ``positive_real`` | (0, hi]                               |
+-------------------+-------------------+---------------------------------------+
| ``PrimeVariable`` | ``prime``         | primes in [lo, hi]                    |
+-------------------+-------------------+---------------------------------------+
| ``PowerOfTwo``    | ``power_of_two``  | {2⁰, 2¹, …} ∩ [lo, hi]              |
+-------------------+-------------------+---------------------------------------+
| ``Fibonacci``     | ``fibonacci``     | Fibonacci numbers in [lo, hi]         |
+-------------------+-------------------+---------------------------------------+

Usage
-----
.. code-block:: python

    from harmonix import DesignSpace
    from harmonix.spaces.math import PrimeVariable, NaturalNumber

    space = DesignSpace()
    space.add("n_bars", NaturalNumber(hi=40))
    space.add("grid",   PowerOfTwo(lo=1, hi=256))
    space.add("p",      PrimeVariable(lo=2, hi=100))

Or via the registry:

.. code-block:: python

    from harmonix import create_variable
    var = create_variable("prime", lo=2, hi=100)
"""

from __future__ import annotations

import random
from typing import List

from ..registry import register_variable
from ..variables import Variable

Context = dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sieve(limit: int) -> List[int]:
    """Return all primes up to *limit* via the Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = bytearray(len(is_prime[i * i :: i]))
    return [i for i, v in enumerate(is_prime) if v]


def _fibonacci_in_range(lo: int, hi: int) -> List[int]:
    """Return Fibonacci numbers in the closed interval [lo, hi]."""
    fibs: List[int] = []
    a, b = 0, 1
    while a <= hi:
        if a >= lo:
            fibs.append(a)
        a, b = b, a + b
    return fibs


def _powers_of_two_in_range(lo: int, hi: int) -> List[int]:
    """Return powers of two in the closed interval [lo, hi]."""
    result: List[int] = []
    p = 1
    while p <= hi:
        if p >= lo:
            result.append(p)
        p <<= 1
    return result


# ---------------------------------------------------------------------------
# NaturalNumber  {1, 2, …, hi}
# ---------------------------------------------------------------------------


@register_variable("natural")
class NaturalNumber(Variable):
    """
    Variable whose domain is the natural numbers {1, 2, …, hi}.

    Parameters
    ----------
    hi : int
        Upper bound (inclusive).
    lo : int, optional
        Lower bound (default 1).  Must be ≥ 1.

    Examples
    --------
    >>> v = NaturalNumber(hi=20)
    >>> 1 <= v.sample({}) <= 20
    True
    """

    def __init__(self, hi: int, lo: int = 1):
        if lo < 1:
            raise ValueError("NaturalNumber: lo must be ≥ 1.")
        self._lo = int(lo)
        self._hi = int(hi)

    def sample(self, ctx: Context) -> int:
        return random.randint(self._lo, self._hi)  # NOSONAR

    def filter(self, candidates: List[int], ctx: Context) -> List[int]:
        return [v for v in candidates if isinstance(v, int) and self._lo <= v <= self._hi]

    def neighbor(self, value: int, ctx: Context) -> int:
        delta = random.choice([-1, 1])  # NOSONAR
        return max(self._lo, min(self._hi, value + delta))


# ---------------------------------------------------------------------------
# WholeNumber  {0, 1, 2, …, hi}
# ---------------------------------------------------------------------------


@register_variable("whole")
class WholeNumber(Variable):
    """
    Variable whose domain is the whole numbers {0, 1, …, hi}.

    Parameters
    ----------
    hi : int
        Upper bound (inclusive).
    """

    def __init__(self, hi: int):
        self._hi = int(hi)

    def sample(self, ctx: Context) -> int:
        return random.randint(0, self._hi)  # NOSONAR

    def filter(self, candidates: List[int], ctx: Context) -> List[int]:
        return [v for v in candidates if isinstance(v, int) and 0 <= v <= self._hi]

    def neighbor(self, value: int, ctx: Context) -> int:
        delta = random.choice([-1, 1])  # NOSONAR
        return max(0, min(self._hi, value + delta))


# ---------------------------------------------------------------------------
# NegativeInt  {lo, …, −1}
# ---------------------------------------------------------------------------


@register_variable("negative_int")
class NegativeInt(Variable):
    """
    Variable restricted to negative integers {lo, …, −1}.

    Parameters
    ----------
    lo : int
        Lower bound (inclusive, must be < 0).
    """

    def __init__(self, lo: int = -100):
        if lo >= 0:
            raise ValueError("NegativeInt: lo must be < 0.")
        self._lo = int(lo)

    def sample(self, ctx: Context) -> int:
        return random.randint(self._lo, -1)  # NOSONAR

    def filter(self, candidates: List[int], ctx: Context) -> List[int]:
        return [v for v in candidates if isinstance(v, int) and self._lo <= v <= -1]

    def neighbor(self, value: int, ctx: Context) -> int:
        delta = random.choice([-1, 1])  # NOSONAR
        return max(self._lo, min(-1, value + delta))


# ---------------------------------------------------------------------------
# NegativeReal  (lo, 0)
# ---------------------------------------------------------------------------


@register_variable("negative_real")
class NegativeReal(Variable):
    """
    Variable restricted to negative real numbers in (lo, 0).

    Parameters
    ----------
    lo : float
        Lower bound (must be < 0).  Default −1e6.
    """

    def __init__(self, lo: float = -1e6):
        if lo >= 0:
            raise ValueError("NegativeReal: lo must be < 0.")
        self._lo = float(lo)

    def sample(self, ctx: Context) -> float:
        return random.uniform(self._lo, -1e-9)  # NOSONAR

    def filter(self, candidates: List[float], ctx: Context) -> List[float]:
        return [v for v in candidates if self._lo <= v < 0]

    def neighbor(self, value: float, ctx: Context) -> float:
        width = abs(self._lo)
        new_v = value + random.gauss(0, 0.05 * width)  # NOSONAR
        return max(self._lo, min(-1e-9, new_v))


# ---------------------------------------------------------------------------
# PositiveReal  (0, hi]
# ---------------------------------------------------------------------------


@register_variable("positive_real")
class PositiveReal(Variable):
    """
    Variable restricted to positive real numbers in (0, hi].

    Parameters
    ----------
    hi : float
        Upper bound.  Default 1e6.
    """

    def __init__(self, hi: float = 1e6):
        if hi <= 0:
            raise ValueError("PositiveReal: hi must be > 0.")
        self._hi = float(hi)

    def sample(self, ctx: Context) -> float:
        return random.uniform(1e-9, self._hi)  # NOSONAR

    def filter(self, candidates: List[float], ctx: Context) -> List[float]:
        return [v for v in candidates if 0 < v <= self._hi]

    def neighbor(self, value: float, ctx: Context) -> float:
        new_v = value + random.gauss(0, 0.05 * self._hi)  # NOSONAR
        return max(1e-9, min(self._hi, new_v))


# ---------------------------------------------------------------------------
# PrimeVariable  — primes in [lo, hi]
# ---------------------------------------------------------------------------


@register_variable("prime")
class PrimeVariable(Variable):
    """
    Variable whose domain is the set of prime numbers in [lo, hi].

    The prime list is computed once at construction via the Sieve of
    Eratosthenes, so ``sample``, ``filter``, and ``neighbor`` are all O(1).

    Parameters
    ----------
    lo : int
        Lower bound (inclusive).  Default 2.
    hi : int
        Upper bound (inclusive).

    Examples
    --------
    >>> v = PrimeVariable(lo=2, hi=50)
    >>> v.sample({}) in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    True
    """

    def __init__(self, hi: int, lo: int = 2):
        all_primes = _sieve(hi)
        self._primes = [p for p in all_primes if p >= lo]
        self._prime_set = set(self._primes)
        if not self._primes:
            raise ValueError(f"No primes found in [{lo}, {hi}].")

    def sample(self, ctx: Context) -> int:
        return random.choice(self._primes)  # NOSONAR

    def filter(self, candidates: List[int], ctx: Context) -> List[int]:
        return [v for v in candidates if v in self._prime_set]

    def neighbor(self, value: int, ctx: Context) -> int:
        try:
            idx = self._primes.index(value)
            delta = random.choice([-1, 1])  # NOSONAR
            new_idx = max(0, min(len(self._primes) - 1, idx + delta))
            return self._primes[new_idx]
        except ValueError:
            return self.sample(ctx)


# ---------------------------------------------------------------------------
# PowerOfTwo  — {2⁰, 2¹, …} ∩ [lo, hi]
# ---------------------------------------------------------------------------


@register_variable("power_of_two")
class PowerOfTwo(Variable):
    """
    Variable restricted to powers of two in [lo, hi].

    Useful for architecture parameters such as layer widths, batch sizes,
    FFT sizes, and grid resolutions.

    Parameters
    ----------
    lo : int
        Minimum power-of-two value (default 1 = 2⁰).
    hi : int
        Maximum power-of-two value.

    Examples
    --------
    >>> v = PowerOfTwo(lo=1, hi=128)
    >>> v.sample({}) in [1, 2, 4, 8, 16, 32, 64, 128]
    True
    """

    def __init__(self, hi: int, lo: int = 1):
        self._values = _powers_of_two_in_range(int(lo), int(hi))
        if not self._values:
            raise ValueError(f"No powers of two found in [{lo}, {hi}].")

    def sample(self, ctx: Context) -> int:
        return random.choice(self._values)  # NOSONAR

    def filter(self, candidates: List[int], ctx: Context) -> List[int]:
        valid = set(self._values)
        return [v for v in candidates if v in valid]

    def neighbor(self, value: int, ctx: Context) -> int:
        try:
            idx = self._values.index(value)
            delta = random.choice([-1, 1])  # NOSONAR
            new_idx = max(0, min(len(self._values) - 1, idx + delta))
            return self._values[new_idx]
        except ValueError:
            return self.sample(ctx)


# ---------------------------------------------------------------------------
# Fibonacci  — Fibonacci numbers in [lo, hi]
# ---------------------------------------------------------------------------


@register_variable("fibonacci")
class Fibonacci(Variable):
    """
    Variable whose domain is the Fibonacci sequence restricted to [lo, hi].

    Parameters
    ----------
    lo : int
        Lower bound (inclusive, default 1).
    hi : int
        Upper bound (inclusive).

    Examples
    --------
    >>> v = Fibonacci(lo=1, hi=100)
    >>> v.sample({}) in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    True
    """

    def __init__(self, hi: int, lo: int = 1):
        self._values = _fibonacci_in_range(int(lo), int(hi))
        if not self._values:
            raise ValueError(f"No Fibonacci numbers found in [{lo}, {hi}].")

    def sample(self, ctx: Context) -> int:
        return random.choice(self._values)  # NOSONAR

    def filter(self, candidates: List[int], ctx: Context) -> List[int]:
        valid = set(self._values)
        return [v for v in candidates if v in valid]

    def neighbor(self, value: int, ctx: Context) -> int:
        try:
            idx = self._values.index(value)
            delta = random.choice([-1, 1])  # NOSONAR
            new_idx = max(0, min(len(self._values) - 1, idx + delta))
            return self._values[new_idx]
        except ValueError:
            return self.sample(ctx)
