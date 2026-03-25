"""
tests/test_registry.py
======================
Tests for the plugin registry and make_variable factory.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from harmonix.registry import (
    VariableAlreadyRegisteredError,
    VariableNotFoundError,
    create_variable,
    get_variable_class,
    list_variable_types,
    make_variable,
    register_variable,
    unregister_variable,
)
from harmonix.variables import Variable

# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------


class TestBuiltinRegistrations:
    def test_builtins_are_registered(self):
        types = list_variable_types()
        for name in ("continuous", "discrete", "integer", "categorical"):
            assert name in types

    def test_engineering_spaces_registered(self):
        # Importing harmonix triggers space registration
        import harmonix  # noqa: F401

        types = list_variable_types()
        for name in ("aci_rebar", "steel_section", "concrete_grade", "soil_spt", "seismic_tbdy", "prime", "natural"):
            assert name in types


# ---------------------------------------------------------------------------
# register_variable
# ---------------------------------------------------------------------------


class TestRegisterVariable:
    def setup_method(self):
        # Clean up test registrations before each test
        for name in ("_test_var", "_test_var2"):
            try:
                unregister_variable(name)
            except VariableNotFoundError:
                pass

    def test_decorator_form(self):
        @register_variable("_test_var")
        class _TestVar(Variable):
            def sample(self, ctx):
                return 1

            def filter(self, c, ctx):
                return c

            def neighbor(self, v, ctx):
                return v

        assert get_variable_class("_test_var") is _TestVar
        unregister_variable("_test_var")

    def test_function_call_form(self):
        class _TestVar2(Variable):
            def sample(self, ctx):
                return 2

            def filter(self, c, ctx):
                return c

            def neighbor(self, v, ctx):
                return v

        register_variable("_test_var2", _TestVar2)
        assert get_variable_class("_test_var2") is _TestVar2
        unregister_variable("_test_var2")

    def test_case_insensitive(self):
        @register_variable("_test_var")
        class _TestVar(Variable):
            def sample(self, ctx):
                return 1

            def filter(self, c, ctx):
                return c

            def neighbor(self, v, ctx):
                return v

        assert get_variable_class("_TEST_VAR") is _TestVar
        assert get_variable_class("_Test_Var") is _TestVar
        unregister_variable("_test_var")

    def test_duplicate_raises(self):
        @register_variable("_test_var")
        class _TV1(Variable):
            def sample(self, ctx):
                return 1

            def filter(self, c, ctx):
                return c

            def neighbor(self, v, ctx):
                return v

        with pytest.raises(VariableAlreadyRegisteredError):

            @register_variable("_test_var")
            class _TV2(Variable):
                def sample(self, ctx):
                    return 2

                def filter(self, c, ctx):
                    return c

                def neighbor(self, v, ctx):
                    return v

        unregister_variable("_test_var")

    def test_overwrite_allowed(self):
        @register_variable("_test_var")
        class _TV1(Variable):
            def sample(self, ctx):
                return 1

            def filter(self, c, ctx):
                return c

            def neighbor(self, v, ctx):
                return v

        @register_variable("_test_var", overwrite=True)
        class _TV2(Variable):
            def sample(self, ctx):
                return 2

            def filter(self, c, ctx):
                return c

            def neighbor(self, v, ctx):
                return v

        assert get_variable_class("_test_var") is _TV2
        unregister_variable("_test_var")

    def test_non_variable_raises(self):
        with pytest.raises(TypeError):
            register_variable("_test_var", int)


# ---------------------------------------------------------------------------
# get_variable_class / create_variable
# ---------------------------------------------------------------------------


class TestLookup:
    def test_not_found_raises(self):
        with pytest.raises(VariableNotFoundError):
            get_variable_class("this_does_not_exist_xyz")

    def test_create_variable_instantiates(self):
        var = create_variable("continuous", 0.0, 1.0)
        assert isinstance(var, Variable)
        s = var.sample({})
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# make_variable
# ---------------------------------------------------------------------------


class TestMakeVariable:
    def test_returns_variable_subclass(self):
        MyVar = make_variable(
            sample=lambda ctx: 42,
            filter=lambda cands, ctx: [c for c in cands if c == 42],
            neighbor=lambda val, ctx: val,
        )
        assert issubclass(MyVar, Variable)

    def test_instance_methods_work(self):
        VALS = [2, 4, 6, 8]
        EvenVar = make_variable(
            sample=lambda ctx: random.choice(VALS),
            filter=lambda cands, ctx: [c for c in cands if c in VALS],
            neighbor=lambda val, ctx: val + 2,
        )
        v = EvenVar()
        assert v.sample({}) in VALS
        assert v.filter([2, 3, 4, 5], {}) == [2, 4]
        assert v.neighbor(2, {}) == 4

    def test_name_applied(self):
        MyVar = make_variable(
            sample=lambda ctx: 1,
            filter=lambda c, ctx: c,
            neighbor=lambda v, ctx: v,
            name="SpecialVar",
        )
        assert MyVar.__name__ == "SpecialVar"

    def test_register_flag(self):
        try:
            unregister_variable("_make_var_test")
        except VariableNotFoundError:
            pass

        MyVar = make_variable(
            sample=lambda ctx: 99,
            filter=lambda c, ctx: c,
            neighbor=lambda v, ctx: v,
            name="_make_var_test",
            register=True,
        )
        assert get_variable_class("_make_var_test") is MyVar
        unregister_variable("_make_var_test")
