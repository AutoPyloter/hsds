"""
examples/01_quickstart.py
=========================
Quickstart: minimise a simple 2D function with a linear constraint.

    minimise    f(x, y) = (x - 3)^2 + (y - 2)^2
    subject to  x + y <= 4

True optimum: x = 2, y = 2  →  f = 1  (on the constraint boundary)

Run
---
    python examples/01_quickstart.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonix import DesignSpace, Continuous, Minimization


# --- 1. Define the search space -------------------------------------------

space = DesignSpace()
space.add("x", Continuous(0.0, 5.0))
space.add("y", Continuous(0.0, 5.0))


# --- 2. Define the objective function -------------------------------------

def objective(harmony):
    x, y = harmony["x"], harmony["y"]
    fitness = (x - 3) ** 2 + (y - 2) ** 2
    penalty = max(0.0, x + y - 4.0)   # x + y <= 4
    return fitness, penalty


# --- 3. Run the optimiser --------------------------------------------------

optimizer = Minimization(space, objective)

result = optimizer.optimize(
    memory_size = 20,
    hmcr        = 0.85,
    par         = 0.35,
    max_iter    = 3000,
    verbose     = False,
)

# --- 4. Inspect results ---------------------------------------------------

print(result)
print(f"x + y = {result.best_harmony['x'] + result.best_harmony['y']:.4f}  "
      f"(constraint: <= 4.0)")
