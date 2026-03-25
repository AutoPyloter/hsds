"""
tests/test_space.py
===================
Unit tests for harmonix.space — DesignSpace.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from harmonix.space import DesignSpace
from harmonix.variables import Categorical, Continuous, Integer


class TestDesignSpaceConstruction:
    def test_add_and_contains(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        assert "x" in space
        assert "y" not in space

    def test_setitem_syntax(self):
        space = DesignSpace()
        space["x"] = Continuous(0.0, 1.0)
        assert "x" in space

    def test_getitem(self):
        space = DesignSpace()
        v = Continuous(0.0, 1.0)
        space.add("x", v)
        assert space["x"] is v

    def test_len(self):
        space = DesignSpace()
        space.add("a", Continuous(0.0, 1.0))
        space.add("b", Continuous(0.0, 1.0))
        assert len(space) == 2

    def test_names_preserves_order(self):
        space = DesignSpace()
        space.add("z", Continuous(0.0, 1.0))
        space.add("a", Continuous(0.0, 1.0))
        space.add("m", Continuous(0.0, 1.0))
        assert space.names() == ["z", "a", "m"]

    def test_init_from_dict(self):
        space = DesignSpace({"x": Continuous(0.0, 1.0), "n": Integer(1, 10)})
        assert len(space) == 2

    def test_add_wrong_type_raises(self):
        space = DesignSpace()
        with pytest.raises(TypeError):
            space.add("x", 42)

    def test_repr(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        assert "x" in repr(space)

    def test_iter_returns_variable_names(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        space.add("y", Continuous(0.0, 1.0))
        assert list(iter(space)) == ["x", "y"]

    def test_chaining(self):
        space = DesignSpace().add("x", Continuous(0.0, 1.0)).add("y", Continuous(0.0, 1.0))
        assert len(space) == 2


class TestDesignSpaceSampling:
    def test_sample_harmony_keys(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        space.add("n", Integer(1, 10))
        space.add("cat", Categorical(["a", "b"]))
        h = space.sample_harmony()
        assert set(h.keys()) == {"x", "n", "cat"}

    def test_sample_harmony_values_in_domain(self):
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        space.add("n", Integer(1, 5))
        for _ in range(50):
            h = space.sample_harmony()
            assert 0.0 <= h["x"] <= 1.0
            assert 1 <= h["n"] <= 5

    def test_dependent_bounds_respected(self):
        space = DesignSpace()
        space.add("lo", Continuous(0.0, 5.0))
        space.add("hi", Continuous(lo=lambda ctx: ctx["lo"], hi=10.0))
        for _ in range(100):
            h = space.sample_harmony()
            assert h["hi"] >= h["lo"], f"hi={h['hi']} < lo={h['lo']}"

    def test_three_level_dependency(self):
        space = DesignSpace()
        space.add("a", Continuous(0.0, 3.0))
        space.add("b", Continuous(lo=lambda ctx: ctx["a"], hi=6.0))
        space.add("c", Continuous(lo=lambda ctx: ctx["b"], hi=10.0))
        for _ in range(100):
            h = space.sample_harmony()
            assert h["a"] <= h["b"] <= h["c"]

    def test_iteration_order(self):
        space = DesignSpace()
        space.add("first", Continuous(0.0, 1.0))
        space.add("second", Continuous(0.0, 1.0))
        space.add("third", Continuous(0.0, 1.0))
        names = [name for name, _ in space.items()]
        assert names == ["first", "second", "third"]
