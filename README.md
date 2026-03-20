# harmonix

**Harmony Search optimisation with dependent variable spaces and engineering domain catalogues.**

harmonix is a Python library for solving single- and multi-objective optimisation problems using the [Harmony Search](https://en.wikipedia.org/wiki/Harmony_search) metaheuristic. Its key design principle is *search-space first*: instead of just minimising a function, you describe the domain of each variable precisely — including dependencies between variables, discrete grids, catalogue lookups, and domain-specific feasibility rules — and let the algorithm handle the rest.

```python
from harmonix import DesignSpace, Continuous, Discrete, Minimization

space = DesignSpace()
space.add("h",  Continuous(0.30, 1.20))
space.add("bf", Continuous(lo=lambda ctx: ctx["h"] * 0.5,   # bf >= h/2
                           hi=lambda ctx: ctx["h"] * 2.0))  # bf <= 2h
space.add("n",  Discrete(4, 2, 20))

def objective(harmony):
    h, bf, n = harmony["h"], harmony["bf"], harmony["n"]
    cost    = 1.1 * h * bf + 0.04 * n
    penalty = max(0.0, h - 2 * bf)    # constraint: h <= 2*bf
    return cost, penalty

result = Minimization(space, objective).optimize(
    memory_size=20, hmcr=0.85, par=0.35, max_iter=5000
)
print(result)
```

---

## Installation

```bash
pip install harmonix
```

Requires Python 3.8+. No mandatory dependencies beyond the standard library.

---

## Core concepts

### Design variables

Every variable implements three methods that the algorithm calls internally:

| Method | Purpose |
|--------|---------|
| `sample(ctx)` | Draw a random feasible value |
| `filter(candidates, ctx)` | Keep only feasible values from harmony memory |
| `neighbor(value, ctx)` | Return an adjacent feasible value (pitch adjustment) |

The `ctx` argument is a dict of all variable values assigned earlier in the same harmony. This enables **dependent bounds** — the domain of a variable can depend on previously assigned variables.

### Built-in variable types

| Type | Domain |
|------|--------|
| `Continuous(lo, hi)` | ℝ ∩ \[lo, hi\] |
| `Discrete(lo, step, hi)` | {lo, lo+step, …, hi} |
| `Integer(lo, hi)` | {lo, lo+1, …, hi} |
| `Categorical(choices)` | finite label set |

All bounds accept callables for dependent domains:

```python
space.add("d",  Continuous(0.40, 1.20))
space.add("tw", Continuous(lo=lambda ctx: ctx["d"] / 50,
                           hi=lambda ctx: ctx["d"] / 10))
```

### Domain-specific variable spaces

harmonix ships a catalogue of ready-made variable types for common engineering and mathematical domains. Import them directly or use the registry:

```python
from harmonix import ACIRebar, SteelSection, ConcreteGrade, PrimeVariable

# ACI 318 ductile bar arrangement (single row)
space.add("rebar", ACIRebar(d_expr=lambda ctx: ctx["d"], cc_expr=40.0,
                            fc=30.0, fy=420.0))

# Standard steel I-section from built-in catalogue
space.add("section", SteelSection(series="IPE"))

# EN 206 concrete grade
space.add("concrete", ConcreteGrade(min_grade="C25/30", max_grade="C50/60"))

# Prime numbers
space.add("p", PrimeVariable(lo=2, hi=500))
```

Full catalogue:

| Category | Types |
|----------|-------|
| **Mathematical** | `NaturalNumber`, `WholeNumber`, `NegativeInt`, `NegativeReal`, `PositiveReal`, `PrimeVariable`, `PowerOfTwo`, `Fibonacci` |
| **Structural** | `ACIRebar`, `ACIDoubleRebar`, `SteelSection`, `ConcreteGrade` |
| **Geotechnical** | `SoilSPT` |
| **Seismic** | `SeismicZoneTBDY` |

All types are also accessible via the plugin registry:

```python
from harmonix import create_variable, list_variable_types
print(list_variable_types())
var = create_variable("aci_rebar", d_expr=0.55, cc_expr=40.0)
```

### Custom variables

**Subclass** `Variable` for full control:

```python
from harmonix import Variable, register_variable

@register_variable("my_type")
class MyVariable(Variable):
    def sample(self, ctx):    ...
    def filter(self, candidates, ctx): ...
    def neighbor(self, value, ctx):    ...
```

**Factory function** for quick prototyping:

```python
from harmonix import make_variable
import random

EvenVar = make_variable(
    sample   = lambda ctx: random.choice(range(2, 101, 2)),
    filter   = lambda cands, ctx: [c for c in cands if c % 2 == 0],
    neighbor = lambda val, ctx: val + random.choice([-2, 2]),
    name     = "even",
)
space.add("n", EvenVar())
```

---

## Optimisers

### Minimization

```python
result = Minimization(space, objective).optimize(
    memory_size      = 20,     # Harmony Memory Size (HMS)
    hmcr             = 0.85,   # Harmony Memory Considering Rate
    par              = 0.35,   # Pitch Adjusting Rate
    max_iter         = 5000,
    verbose          = True,
    callback         = my_callback,          # optional
    checkpoint_path  = "run.json",           # optional: crash recovery
    checkpoint_every = 500,
)
print(result.best_harmony)
print(result.best_fitness)
```

### Maximization

Same interface — negate internally, reports original sign.

```python
result = Maximization(space, objective).optimize(...)
```

### MultiObjective

```python
def objective(harmony):
    f1 = harmony["x"] ** 2
    f2 = (harmony["x"] - 2) ** 2
    return (f1, f2), 0.0    # tuple of objectives, penalty

result = MultiObjective(space, objective).optimize(
    max_iter     = 10_000,
    archive_size = 100,        # Pareto archive capacity
)

for entry in result.front:
    print(entry.objectives, entry.harmony)
```

### Callback and early stopping

```python
def my_callback(iteration, partial_result):
    print(iteration, partial_result.best_fitness)
    if partial_result.best_fitness < 1e-4:
        raise StopIteration    # stops the loop cleanly
```

### Crash recovery (checkpointing)

```python
# First run — saves state every 1000 iterations
result = optimizer.optimize(
    max_iter         = 50_000,
    checkpoint_path  = "run.json",
    checkpoint_every = 1000,
)

# If interrupted, re-run the same line — resumes automatically
result = optimizer.optimize(
    max_iter         = 50_000,
    checkpoint_path  = "run.json",
    checkpoint_every = 1000,
)
```

---

## Decoding engineering variables

Variables like `ACIRebar` and `SteelSection` store integer codes in the harmony. Use `decode()` to get the full properties:

```python
rebar_var = ACIRebar(d_expr=0.55, cc_expr=40.0)
code = result.best_harmony["rebar"]
diameter_mm, bar_count = rebar_var.decode(code)
print(rebar_var.describe(code))   # "8 bars of Ø19.00 mm"

section_var = SteelSection(series="IPE")
idx = result.best_harmony["section"]
sec = section_var.decode(idx)
print(sec.name, sec.Iy_cm4, "cm⁴")
```

### Steel section catalogue

The built-in catalogue covers IPE 80–600, HEA 100–500, HEB 100–500, and a selection of W-sections. You can override it with your own JSON or CSV file:

```python
# JSON: list of objects with keys matching SectionProperties fields
var = SteelSection(catalogue="my_sections.json")

# Filter to a subset of series
var = SteelSection(series=["HEA", "HEB"])
```

---

## Algorithm background

harmonix implements the Harmony Search algorithm with two key enhancements over the basic formulation:

**Intelligent pitch adjustment** — when PAR fires, `neighbor()` is called with the current dependency context, so the perturbed value is guaranteed to remain feasible. This replaces the common but incorrect approach of calling `sample()` again on PAR, which ignores the context and treats PAR as a second random draw.

**Dependent search spaces** — variables are sampled in definition order and each receives a context dict of previously assigned values. This means dependent bounds, catalogue filters, and feasibility checks can all reference earlier variables without any special handling in the optimizer loop.

### References

- Geem, Z. W., Kim, J. H., & Loganathan, G. V. (2001). A new heuristic optimization algorithm: Harmony search. *Simulation*, 76(2), 60–68.
- Lee, K. S., & Geem, Z. W. (2005). A new meta-heuristic algorithm for continuous engineering optimization. *Computer Methods in Applied Mechanics and Engineering*, 194(36–38), 3902–3933.
- Deb, K. (2000). An efficient constraint handling method for genetic algorithms. *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338.

---

## Project structure

```
harmonix/
├── harmonix/
│   ├── variables.py      # Continuous, Discrete, Integer, Categorical
│   ├── space.py          # DesignSpace
│   ├── optimizer.py      # Minimization, Maximization, MultiObjective
│   ├── pareto.py         # Pareto archive, crowding distance
│   ├── registry.py       # register_variable, make_variable
│   └── spaces/
│       ├── math.py       # Mathematical search spaces
│       └── engineering.py # Engineering domain spaces
├── examples/
├── tests/
├── pyproject.toml
└── LICENSE
```

---

## License

MIT © Abdulkadir Özcan
