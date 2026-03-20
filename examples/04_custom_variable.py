"""
examples/04_custom_variable.py
==============================
Demonstrates both ways to define a custom variable:

1. Subclass Variable  — full control, best for complex domains
2. make_variable      — factory function, best for quick prototyping

Problem: find the pair (p, q) of twin primes closest to 100
         that minimises |p - 100| + |q - 100|.

Twin primes are pairs (p, p+2) where both are prime, e.g. (11,13), (17,19).

Run
---
    python examples/04_custom_variable.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonix import (
    DesignSpace, Minimization,
    Variable, register_variable, make_variable,
)


# ---------------------------------------------------------------------------
# Helper: sieve of Eratosthenes
# ---------------------------------------------------------------------------

def sieve(limit: int):
    is_p = bytearray([1]) * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i :: i] = bytearray(len(is_p[i * i :: i]))
    return [i for i, v in enumerate(is_p) if v]


PRIMES     = sieve(500)
PRIME_SET  = set(PRIMES)
TWIN_PAIRS = [(p, p + 2) for p in PRIMES if p + 2 in PRIME_SET and p <= 500]
TWIN_FIRSTS  = [p for p, _ in TWIN_PAIRS]   # first of each twin prime pair


# ---------------------------------------------------------------------------
# Method 1: subclass Variable
# ---------------------------------------------------------------------------

@register_variable("twin_prime_first")
class TwinPrimeVariable(Variable):
    """
    Variable whose domain is the first element of each twin prime pair
    in the range [lo, hi].

    Example pairs: (3,5), (5,7), (11,13), (17,19), (29,31), …
    """

    def __init__(self, lo: int = 3, hi: int = 500):
        self._values = [p for p in TWIN_FIRSTS if lo <= p <= hi]
        self._value_set = set(self._values)
        if not self._values:
            raise ValueError(f"No twin prime pairs found in [{lo}, {hi}].")

    def sample(self, ctx) -> int:
        return random.choice(self._values)

    def filter(self, candidates, ctx):
        return [v for v in candidates if v in self._value_set]

    def neighbor(self, value, ctx) -> int:
        if value not in self._value_set:
            return self.sample(ctx)
        idx   = self._values.index(value)
        delta = random.choice([-1, 1])
        new_i = max(0, min(len(self._values) - 1, idx + delta))
        return self._values[new_i]

    def pair(self, value: int):
        """Return the twin prime pair (value, value+2)."""
        return value, value + 2


# ---------------------------------------------------------------------------
# Method 2: make_variable factory
# ---------------------------------------------------------------------------

# A variable that picks from Fibonacci numbers in [1, 200]
FIB = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
FIB_SET = set(FIB)

FibonacciVar = make_variable(
    sample   = lambda ctx: random.choice(FIB),
    filter   = lambda cands, ctx: [c for c in cands if c in FIB_SET],
    neighbor = lambda val, ctx: FIB[
        max(0, min(len(FIB) - 1,
            FIB.index(val) + random.choice([-1, 1])
            if val in FIB_SET else 0))
    ],
    name     = "fibonacci_200",
)


# ---------------------------------------------------------------------------
# Problem 1: find twin prime pair closest to 100
# ---------------------------------------------------------------------------

print("=" * 55)
print("Problem 1 — Twin prime pair closest to 100")
print("=" * 55)

space1 = DesignSpace()
twin_var = TwinPrimeVariable(lo=3, hi=300)
space1.add("p", twin_var)

def twin_objective(h):
    p = h["p"]
    q = p + 2
    fitness = abs(p - 100) + abs(q - 100)
    return fitness, 0.0

result1 = Minimization(space1, twin_objective).optimize(
    memory_size=15, max_iter=500, verbose=False
)

p_best = result1.best_harmony["p"]
q_best = p_best + 2
print(result1)
print(f"Best twin prime pair: ({p_best}, {q_best})")
print(f"Distance to 100:      {result1.best_fitness:.0f}")


# ---------------------------------------------------------------------------
# Problem 2: minimise |f - 100| where f is a Fibonacci number
# ---------------------------------------------------------------------------

print()
print("=" * 55)
print("Problem 2 — Fibonacci number closest to 100 (make_variable)")
print("=" * 55)

space2 = DesignSpace()
space2.add("f", FibonacciVar())

result2 = Minimization(space2, lambda h: (abs(h["f"] - 100), 0.0)).optimize(
    memory_size=10, max_iter=200, verbose=False
)

print(result2)
print(f"Closest Fibonacci to 100: {result2.best_harmony['f']}")
# Correct answer: 89 (|89-100|=11) or 144 (|144-100|=44) → 89
