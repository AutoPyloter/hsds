"""
tests/test_multiobjective_logging.py
==================================
Focused coverage for multi-objective logging, checkpoint persistence,
and deterministic persisted artifacts under fixed seeds.
"""

import csv
import json
import os
import random
import tempfile
from pathlib import Path

from harmonix.optimizer import Minimization, MultiObjective
from harmonix.space import DesignSpace
from harmonix.variables import Continuous


def _tmp_json() -> Path:
    fd, fname = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(fname)
    return Path(fname)


def _biobj_space() -> DesignSpace:
    space = DesignSpace()
    space.add("x", Continuous(0.0, 1.0))
    space.add("y", Continuous(0.0, 1.0))
    return space


def _biobj(h):
    f1 = h["x"]
    f2 = 1.0 - h["y"]
    return (f1, f2), 0.0


class TestMultiObjectiveLogging:
    def test_log_init_creates_csv_with_expected_row_count(self):
        ckpt = _tmp_json()
        try:
            MultiObjective(_biobj_space(), _biobj).optimize(
                memory_size=6,
                max_iter=10,
                checkpoint_path=ckpt,
                log_init=True,
                resume="new",
            )
            init_csv = ckpt.with_name(ckpt.stem + "_init.csv")
            assert init_csv.exists()
            rows = init_csv.read_text().splitlines()
            assert len(rows) == 7  # header + memory_size
            init_csv.unlink()
        finally:
            ckpt.unlink(missing_ok=True)

    def test_log_evaluations_writes_one_row_per_iteration(self):
        ckpt = _tmp_json()
        try:
            MultiObjective(_biobj_space(), _biobj).optimize(
                memory_size=5,
                max_iter=12,
                checkpoint_path=ckpt,
                log_evaluations=True,
                resume="new",
            )
            eval_csv = ckpt.with_name(ckpt.stem + "_evals.csv")
            assert eval_csv.exists()
            rows = eval_csv.read_text().splitlines()
            assert len(rows) == 13  # header + max_iter
            eval_csv.unlink()
        finally:
            ckpt.unlink(missing_ok=True)

    def test_checkpoint_contains_archive_on_fresh_run(self):
        ckpt = _tmp_json()
        try:
            MultiObjective(_biobj_space(), _biobj).optimize(
                memory_size=5,
                max_iter=8,
                checkpoint_path=ckpt,
                checkpoint_every=8,
                resume="new",
            )
            payload = json.loads(ckpt.read_text())
            assert "archive" in payload
            assert "entries" in payload["archive"]
            assert isinstance(payload["archive"]["entries"], list)
        finally:
            ckpt.unlink(missing_ok=True)

    def test_resume_auto_continues_multiobjective_run(self):
        ckpt = _tmp_json()
        try:
            MultiObjective(_biobj_space(), _biobj).optimize(
                memory_size=5,
                max_iter=9,
                checkpoint_path=ckpt,
                checkpoint_every=9,
                resume="new",
            )
            resumed = MultiObjective(_biobj_space(), _biobj).optimize(
                memory_size=5,
                max_iter=15,
                checkpoint_path=ckpt,
                checkpoint_every=15,
                resume="auto",
            )
            assert resumed.iterations == 6
        finally:
            ckpt.unlink(missing_ok=True)


class TestDeterministicArtifacts:
    def test_same_seed_same_minimization_history_log_content(self, tmp_path):
        def run_once(seed, path):
            random.seed(seed)
            space = DesignSpace()
            space.add("x", Continuous(0.0, 1.0))
            space.add("y", Continuous(0.0, 1.0))
            Minimization(space, lambda h: (h["x"] ** 2 + h["y"] ** 2, 0.0)).optimize(
                memory_size=6,
                max_iter=10,
                log_history=True,
                history_log_path=path,
            )
            return path.read_text()

        hist1 = tmp_path / "h1.csv"
        hist2 = tmp_path / "h2.csv"
        assert run_once(321, hist1) == run_once(321, hist2)

    def test_multiobjective_eval_log_headers_are_stable(self):
        ckpt = _tmp_json()
        try:
            MultiObjective(_biobj_space(), _biobj).optimize(
                memory_size=4,
                max_iter=3,
                checkpoint_path=ckpt,
                log_evaluations=True,
                resume="new",
            )
            eval_csv = ckpt.with_name(ckpt.stem + "_evals.csv")
            row = next(csv.DictReader(eval_csv.open()))
            assert {"wall_time_s", "iteration", "x", "y", "fitness", "penalty", "feasible"}.issubset(set(row.keys()))
            eval_csv.unlink()
        finally:
            ckpt.unlink(missing_ok=True)
