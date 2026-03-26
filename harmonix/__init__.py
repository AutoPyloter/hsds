"""
PyHarmonySearch
===============
A clean, extensible Python implementation of the Harmony Search (HS)
metaheuristic optimisation algorithm.

Quickstart
----------
>>> from harmonix import DesignSpace, Continuous, Minimization
>>>
>>> space = DesignSpace()
>>> space.add("x", Continuous(0.0, 5.0))
>>> space.add("y", Continuous(0.0, 5.0))
>>>
>>> def objective(h):
...     fitness = (h["x"] - 2)**2 + (h["y"] - 3)**2
...     return fitness, 0.0
>>>
>>> result = Minimization(space, objective).optimize(max_iter=2000)
>>> print(result.best_harmony)

Custom variables
----------------
Three patterns are supported:

1. **Subclass** :class:`Variable` for full control.
2. **Factory**: use :func:`make_variable` to create a class from functions.
3. **Registry**: register a class with :func:`register_variable` and
   retrieve it later with :func:`create_variable`.

Built-in domain spaces
----------------------
Mathematical and engineering spaces live in :mod:`harmonix.spaces`
and are also re-exported here for convenience:

>>> from harmonix import PrimeVariable, ACIRebar
"""

# --- primitives ---
# --- logging / caching utilities ---
from .logging import EvaluationCache, RunLogger

# --- optimizers ---
from .optimizer import (
    HarmonyMemory,
    HarmonySearchOptimizer,
    Maximization,
    Minimization,
    MultiObjective,
    OptimizationResult,
)

# --- pareto ---
from .pareto import ArchiveEntry, ParetoArchive, ParetoResult, dominates

# --- registry ---
from .registry import (
    VariableAlreadyRegisteredError,
    VariableNotFoundError,
    create_variable,
    get_variable_class,
    list_variable_types,
    make_variable,
    register_variable,
    unregister_variable,
)

# --- space ---
from .space import DesignSpace

# --- built-in domain spaces (trigger registration via import) ---
from .spaces import (
    ACIDoubleRebar,
    # engineering — rebar
    ACIRebar,
    ConcreteGrade,
    ConcreteGradeProperties,
    Fibonacci,
    # math
    NaturalNumber,
    NegativeInt,
    NegativeReal,
    PositiveReal,
    PowerOfTwo,
    PrimeVariable,
    SectionProperties,
    SeismicZone,
    # engineering — seismic
    SeismicZoneTBDY,
    SoilProfile,
    # engineering — geotechnical
    SoilSPT,
    # engineering — sections & materials
    SteelSection,
    WholeNumber,
)
from .variables import Categorical, Continuous, Discrete, Integer, Variable

__all__ = [
    # primitives
    "Variable",
    "Continuous",
    "Discrete",
    "Integer",
    "Categorical",
    # space
    "DesignSpace",
    # optimizers
    "HarmonySearchOptimizer",
    "HarmonyMemory",
    "OptimizationResult",
    "Minimization",
    "Maximization",
    "MultiObjective",
    # pareto
    "ParetoArchive",
    "ParetoResult",
    "ArchiveEntry",
    "dominates",
    # registry
    "register_variable",
    "make_variable",
    "get_variable_class",
    "create_variable",
    "list_variable_types",
    "unregister_variable",
    "VariableNotFoundError",
    "VariableAlreadyRegisteredError",
    # logging / caching
    "EvaluationCache",
    "RunLogger",
    # math spaces
    "NaturalNumber",
    "WholeNumber",
    "NegativeInt",
    "NegativeReal",
    "PositiveReal",
    "PrimeVariable",
    "PowerOfTwo",
    "Fibonacci",
    # engineering — rebar
    "ACIRebar",
    "ACIDoubleRebar",
    # engineering — sections & materials
    "SteelSection",
    "SectionProperties",
    "ConcreteGrade",
    "ConcreteGradeProperties",
    # engineering — geotechnical
    "SoilSPT",
    "SoilProfile",
    # engineering — seismic
    "SeismicZoneTBDY",
    "SeismicZone",
]

__version__ = "1.0.2"
__author__ = "Abdulkadir Özcan"
__license__ = "MIT"
