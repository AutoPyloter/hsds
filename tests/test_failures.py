"""
tests/test_failures.py
=====================
High-value contract and failure-mode tests for optimizer, checkpoint,
and dependency behavior.

These tests focus on production-grade robustness:
- objective/callback exception propagation
- malformed multi-objective outputs
- corrupted/incomplete checkpoint handling
- dependency callable failure surfaces
- deterministic persisted state under fixed random seeds
"""

import json
import os
import random
import tempfile
from pathlib import Path

import pytest

from harmonix.optimizer import Minimization, MultiObjective
from harmonix.space import DesignSpace
from harmonix.variables import Continuous


def _tmp_json() -> Path:
    fd, fname = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(fname)
    return Path(fname)


def _simple_space() -> DesignSpace:
    space = DesignSpace()
    space.add("x", Continuous(0.0, 1.0))
    return space


class TestObjectiveAndCallbackFailures:
    def test_minimization_objective_exception_propagates(self):
        space = _simple_space()

        def obj(h):
            raise RuntimeError("objective crashed")

        with pytest.raises(RuntimeError, match="objective crashed"):
            Minimization(space, obj).optimize(memory_size=5, max_iter=10)

    def test_callback_non_stopiteration_exception_propagates(self):
        space = _simple_space()

        def cb(it, partial):
            if it == 3:
                raise RuntimeError("callback crashed")

        with pytest.raises(RuntimeError, match="callback crashed"):
            Minimization(space, lambda h: (h["x"], 0.0)).optimize(
                memory_size=5,
                max_iter=20,
                callback=cb,
            )

    def test_multiobjective_callback_non_stopiteration_exception_propagates(self):
        space = _simple_space()

        def cb(it, partial):
            if it == 2:
                raise ValueError("bad callback")

        with pytest.raises(ValueError, match="bad callback"):
            MultiObjective(space, lambda h: ((h["x"], 1.0 - h["x"]), 0.0)).optimize(
                memory_size=5,
                max_iter=20,
                callback=cb,
            )


class TestMalformedObjectiveOutputs:
    def test_multiobjective_scalar_objective_output_raises_typeerror(self):
        space = _simple_space()

        def bad_obj(h):
            return 1.23, 0.0

        with pytest.raises(TypeError):
            MultiObjective(space, bad_obj).optimize(memory_size=5, max_iter=10)

    def test_multiobjective_nonnumeric_objective_component_raises_valueerror(self):
        space = _simple_space()

        def bad_obj(h):
            return ("bad", 1.0), 0.0

        with pytest.raises(ValueError):
            MultiObjective(space, bad_obj).optimize(memory_size=5, max_iter=10)


class TestCheckpointFailures:
    def test_resume_with_invalid_json_checkpoint_raises(self):
        space = _simple_space()
        ckpt = _tmp_json()
        try:
            ckpt.write_text("{not valid json")
            with pytest.raises(json.JSONDecodeError):
                Minimization(space, lambda h: (h["x"], 0.0)).optimize(
                    memory_size=5,
                    max_iter=10,
                    checkpoint_path=ckpt,
                    resume="resume",
                )
        finally:
            ckpt.unlink(missing_ok=True)

    def test_resume_with_missing_memory_field_raises_keyerror(self):
        space = _simple_space()
        ckpt = _tmp_json()
        try:
            ckpt.write_text(json.dumps({"iteration": 7}, indent=2))
            with pytest.raises(KeyError):
                Minimization(space, lambda h: (h["x"], 0.0)).optimize(
                    memory_size=5,
                    max_iter=10,
                    checkpoint_path=ckpt,
                    resume="resume",
                )
        finally:
            ckpt.unlink(missing_ok=True)

    def test_multiobjective_resume_with_missing_archive_field_raises_keyerror(self):
        space = _simple_space()
        ckpt = _tmp_json()
        try:
            payload = {
                "iteration": 5,
                "memory": {
                    "size": 2,
                    "mode": "min",
                    "harmonies": [{"x": 0.1}, {"x": 0.2}],
                    "fitness": [0.1, 0.2],
                    "penalty": [0.0, 0.0],
                },
            }
            ckpt.write_text(json.dumps(payload, indent=2))
            with pytest.raises(KeyError):
                MultiObjective(space, lambda h: ((h["x"], 1.0 - h["x"]), 0.0)).optimize(
                    memory_size=5,
                    max_iter=10,
                    checkpoint_path=ckpt,
                    resume="resume",
                )
        finally:
            ckpt.unlink(missing_ok=True)


class TestDependencyFailures:
    def test_designspace_missing_dependency_key_surfaces_keyerror(self):
        space = DesignSpace()
        space.add("x", Continuous(lambda ctx: ctx["missing"], 1.0))

        with pytest.raises(KeyError, match="missing"):
            space.sample_harmony()


class TestDeterministicPersistedState:
    def test_same_seed_produces_identical_checkpoint_content(self):
        def run_once(seed, path):
            random.seed(seed)
            space = DesignSpace()
            space.add("x", Continuous(0.0, 1.0))
            space.add("y", Continuous(0.0, 1.0))
            Minimization(space, lambda h: (h["x"] ** 2 + h["y"] ** 2, 0.0)).optimize(
                memory_size=8,
                max_iter=15,
                checkpoint_path=path,
                checkpoint_every=15,
                resume="new",
            )
            return json.loads(path.read_text())

        ckpt1 = _tmp_json()
        ckpt2 = _tmp_json()
        try:
            data1 = run_once(123, ckpt1)
            data2 = run_once(123, ckpt2)
            assert data1 == data2
        finally:
            ckpt1.unlink(missing_ok=True)
            ckpt2.unlink(missing_ok=True)
