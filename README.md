# harmonix

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19160019.svg)](https://doi.org/10.5281/zenodo.19160019)
[![PyPI](https://img.shields.io/pypi/v/harmonix-opt)](https://pypi.org/project/harmonix-opt/)
[![CI](https://github.com/AutoPyloter/harmonix/actions/workflows/ci.yml/badge.svg)](https://github.com/AutoPyloter/harmonix/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-325%20passed-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![codecov](https://codecov.io/github/AutoPyloter/harmonix/graph/badge.svg?token=H93FEVMFLS)](https://codecov.io/github/AutoPyloter/harmonix)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Downloads](https://static.pepy.tech/badge/harmonix-opt)](https://pepy.tech/project/harmonix-opt)
[![Downloads](https://static.pepy.tech/badge/harmonix-opt/month)](https://pepy.tech/project/harmonix-opt)
[![PyPI Downloads](https://img.shields.io/pypi/dm/harmonix-opt.svg)](https://pypi.org/project/harmonix-opt/)
[![GitHub Repo stars](https://img.shields.io/github/stars/AutoPyloter/harmonix?style=social)](https://github.com/AutoPyloter/harmonix/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![GitHub last commit](https://img.shields.io/github/last-commit/AutoPyloter/harmonix)](https://github.com/AutoPyloter/harmonix/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/AutoPyloter/harmonix)](https://github.com/AutoPyloter/harmonix/issues)
[![GitHub repo size](https://img.shields.io/github/repo-size/AutoPyloter/harmonix)]()
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Sourcery](https://img.shields.io/badge/Sourcery-enabled-brightgreen)](https://sourcery.ai)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/harmonix-opt)](https://pypi.org/project/harmonix-opt/)

**Harmony Search optimisation with dependent variable spaces and engineering domain catalogues.**

harmonix is a Python library for solving single- and multi-objective optimisation problems using the [Harmony Search](https://en.wikipedia.org/wiki/Harmony_search) metaheuristic. Its key design principle is *search-space first*: instead of just minimising a function, you describe the domain of each variable precisely — including dependencies between variables, discrete grids, catalogue lookups, and domain-specific feasibility rules — and let the algorithm handle the rest.

```python
from harmonix import DesignSpace, Continuous, Discrete, Minimization

space = DesignSpace()
space.add("h",  Continuous(0.30, 1.20))
space.add("bf", Continuous(lo=lambda ctx: ctx["h"] * 0.5,
                           hi=lambda ctx: ctx["h"] * 2.0))
space.add("n",  Discrete(4, 2, 20))

def objective(harmony):
    h, bf, n = harmony["h"], harmony["bf"], harmony["n"]
    cost    = 1.1 * h * bf + 0.04 * n
    penalty = max(0.0, h - 2 * bf)
    return cost, penalty

result = Minimization(space, objective).optimize(
    memory_size=20, hmcr=0.85, par=0.35, max_iter=5000
)
print(result)
```

---

## Installation

```bash
pip install harmonix-opt
```

Requires Python 3.8+. No mandatory dependencies beyond the standard library.

For development:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

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
| `Continuous(lo, hi)` | ℝ ∩ [lo, hi] |
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

harmonix ships a catalogue of ready-made variable types for common engineering and mathematical domains:

```python
from harmonix import ACIRebar, SteelSection, ConcreteGrade, PrimeVariable

# ACI 318 ductile bar arrangement — bounds depend on d and fc
space.add("rebar", ACIRebar(d_expr=lambda ctx: ctx["d"],
                            cc_expr=60.0,
                            fc=lambda ctx: ctx["grade"].fck_MPa,
                            fy=420.0))

# Standard steel I-section from built-in catalogue (IPE, HEA, HEB, W)
space.add("section", SteelSection(series=["IPE", "HEA"]))

# EN 206 concrete grade (C12/15 to C90/105)
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
    def sample(self, ctx):              ...
    def filter(self, candidates, ctx):  ...
    def neighbor(self, value, ctx):     ...
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
    memory_size      = 20,       # Harmony Memory Size (HMS)
    hmcr             = 0.85,     # Harmony Memory Considering Rate
    par              = 0.35,     # Pitch Adjusting Rate
    max_iter         = 5000,
    bw_max           = 0.05,     # Initial bandwidth (5% of domain width)
    bw_min           = 0.001,    # Final bandwidth (exponential decay)
    resume           = "auto",   # "auto" | "new" | "resume"
    checkpoint_path  = "run.json",
    checkpoint_every = 500,
    use_cache        = False,    # Cache identical harmony evaluations
    cache_maxsize    = 4096,
    log_init         = False,    # Write initial memory to CSV
    log_history      = False,    # Write best-per-iteration to CSV
    log_evaluations  = False,    # Write every evaluated harmony to CSV
    history_every    = 1,
    verbose          = True,
    callback         = my_callback,
)
print(result.best_harmony)
print(result.best_fitness)
```

### Maximization

Same interface — negates internally, reports original sign.

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
    archive_size = 100,
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

---

## Advanced features

### Dynamic bandwidth narrowing

The pitch adjustment step size decays exponentially from `bw_max` to `bw_min` over the run — wide exploration early, fine convergence late.

```python
result = Minimization(space, objective).optimize(
    bw_max=0.10,   # 10% of domain width at iteration 0
    bw_min=0.001,  # 0.1% at final iteration
    max_iter=5000,
)
```

Set `bw_max == bw_min` for constant bandwidth (original HS behaviour). Discrete and categorical variables are unaffected by bandwidth.

### Resume control

```python
# "auto"   — continue if checkpoint exists, start fresh otherwise (safe default)
# "new"    — always start fresh, overwrite any existing checkpoint
# "resume" — always continue; raises FileNotFoundError if checkpoint missing

result = optimizer.optimize(
    max_iter        = 50_000,
    checkpoint_path = "run.json",
    resume          = "auto",
)
```

The initial harmony memory is saved immediately at startup — even a run interrupted in the first seconds can be resumed cleanly.

### Evaluation cache

Identical harmonies are never re-evaluated when `use_cache=True`. Particularly valuable for expensive objectives (FEM, CFD, etc.).

```python
result = optimizer.optimize(use_cache=True, cache_maxsize=4096)
print(optimizer._cache.stats())
# EvaluationCache: 412 hits / 1005 total (41.0% hit rate)  size=593/4096
```

### CSV logging

```python
result = optimizer.optimize(
    checkpoint_path  = "run.json",
    log_init         = True,    # → run_init.csv     (initial memory)
    log_history      = True,    # → run_history.csv  (best per iteration)
    log_evaluations  = True,    # → run_evals.csv    (every evaluation)
    history_every    = 10,      # write history every 10 iterations
)
```

All CSV files are readable directly in Excel or with `pandas.read_csv()`.

---

## Decoding engineering variables

Variables like `ACIRebar` and `SteelSection` store integer codes in the harmony. Use `decode()` to get full properties:

```python
rebar_var = ACIRebar(d_expr=0.55, cc_expr=40.0)
code = result.best_harmony["rebar"]
diameter_mm, bar_count = rebar_var.decode(code)
print(rebar_var.describe(code))   # "8 bars of Ø19.00 mm"

section_var = SteelSection(series=["IPE"])
sec = section_var.decode(result.best_harmony["section"])
print(sec.name, sec.Iy_cm4, "cm4")

grade_var = ConcreteGrade()
grade = grade_var.decode(result.best_harmony["concrete"])
print(grade.name, grade.fck_MPa, "MPa", grade.Ecm_GPa, "GPa")
```

### Steel section catalogue

The built-in catalogue covers IPE 80–600, HEA 100–500, HEB 100–500, and W-sections. Override with your own file:

```python
var = SteelSection(catalogue="my_sections.json")  # custom catalogue
var = SteelSection(series=["HEA", "HEB"])          # filter series
```

---

## Algorithm background

harmonix implements Harmony Search with several enhancements:

**Dynamic bandwidth narrowing** — pitch adjustment step size decays exponentially. Early iterations explore broadly; late iterations converge precisely.

**Intelligent pitch adjustment** — `neighbor()` is called with the current dependency context so the perturbed value stays feasible. The common incorrect approach of calling `sample()` on PAR is avoided.

**Dependent search spaces** — variables are sampled in definition order; each receives a context dict of previously assigned values. Dependent bounds, catalogue filters, and feasibility checks can reference earlier variables without any special handling in the optimiser loop.

**Deb constraint handling** — feasible solutions always rank above infeasible ones; among infeasible solutions ranking is by total penalty.

### References

- Geem, Z. W., Kim, J. H., & Loganathan, G. V. (2001). A new heuristic optimization algorithm: Harmony search. *Simulation*, 76(2), 60–68.
- Lee, K. S., & Geem, Z. W. (2005). A new meta-heuristic algorithm for continuous engineering optimization. *Computer Methods in Applied Mechanics and Engineering*, 194(36–38), 3902–3933.
- Deb, K. (2000). An efficient constraint handling method for genetic algorithms. *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338.
- Ricart, J., Hüttemann, G., Lima, J., & Barán, B. (2011). Multiobjective harmony search algorithm proposals. *Electronic Notes in Theoretical Computer Science*, 281, 51–67.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

325 tests across 9 test files covering:

- All variable types — `sample`, `filter`, `neighbor`, edge cases, `lo > hi` validation
- DesignSpace — dependency chains, empty space, 50-variable stress test
- Optimisers — Minimization, Maximization, MultiObjective
- New features — bandwidth decay, resume modes, evaluation cache, CSV logging
- Pareto archive — dominance, crowding distance, serialization
- Engineering physics — EC2 formulas, ACI 318 feasibility, steel section properties
- Determinism — same seed produces identical results
- Numerical correctness — Sphere, Rosenbrock, constrained minimization

---

## Project structure

```
harmonix/
├── harmonix/
│   ├── variables.py       # Continuous, Discrete, Integer, Categorical
│   ├── space.py           # DesignSpace
│   ├── optimizer.py       # Minimization, Maximization, MultiObjective
│   ├── pareto.py          # Pareto archive, crowding distance
│   ├── registry.py        # register_variable, make_variable
│   ├── logging.py         # EvaluationCache, RunLogger
│   └── spaces/
│       ├── math.py        # Mathematical search spaces
│       └── engineering.py # Engineering domain spaces
├── examples/
│   ├── 01_quickstart.py
│   ├── 02_dependent_bounds.py
│   ├── 03_engineering_rc_beam.py
│   ├── 04_custom_variables.py
│   ├── 05_multi_objective.py
│   ├── 06_steel_beam_design.py
│   └── 07_rc_section_full.py
├── tests/                 # 325 tests across 9 files
├── requirements-dev.txt
├── ruff.toml
├── pyproject.toml
└── LICENSE
```

---

## Citation

If you use `harmonix-opt` in your research, please cite it as follows:

**APA:**
> Özcan, A. (2026). harmonix-opt: Harmony Search optimisation with dependent variable spaces (Version 1.0.1) [Computer software]. https://doi.org/10.5281/zenodo.19160019

**BibTeX:**
```bibtex
@software{ozcan_harmonix_2026,
  author       = {Özcan, Abdulkadir},
  title        = {harmonix-opt: Harmony Search optimisation with dependent variable spaces},
  month        = mar,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.1},
  doi          = {10.5281/zenodo.19160019},
  url          = {[https://doi.org/10.5281/zenodo.19160019](https://doi.org/10.5281/zenodo.19160019)}
}
```

---

## License

MIT © Abdulkadir Özcan
