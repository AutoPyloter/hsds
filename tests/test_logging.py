"""
tests/test_logging.py
=====================
Tests for EvaluationCache, RunLogger, and the resume / logging
parameters of Minimization and MultiObjective.

Covers:
- EvaluationCache: hits, misses, eviction, clear, stats
- RunLogger: CSV headers, row counts, content correctness
- resume="new" / "auto" / "resume" behaviour
- log_init / log_history / log_evaluations / history_every
- init checkpoint written immediately
"""

import csv
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from harmonix.variables import Continuous
from harmonix.space import DesignSpace
from harmonix.optimizer import Minimization, MultiObjective
from harmonix.logging import EvaluationCache, RunLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_space(n=2):
    space = DesignSpace()
    for i in range(n):
        space.add(f"x{i}", Continuous(-5.0, 5.0))
    return space

def _sphere(h):
    return sum(v**2 for v in h.values()), 0.0

def _tmp_json():
    fd, fname = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(fname)
    return Path(fname)


# ---------------------------------------------------------------------------
# EvaluationCache
# ---------------------------------------------------------------------------

class TestEvaluationCache:
    def test_no_double_evaluation(self):
        count = [0]
        def obj(h):
            count[0] += 1
            return h["x"] ** 2, 0.0

        cache = EvaluationCache(obj, maxsize=256)
        h = {"x": 3.14}
        cache(h)
        cache(h)
        cache(h)
        assert count[0] == 1
        assert cache.hits == 2
        assert cache.misses == 1

    def test_different_harmonies_evaluated(self):
        count = [0]
        def obj(h):
            count[0] += 1
            return h["x"] ** 2, 0.0

        cache = EvaluationCache(obj, maxsize=256)
        for x in [1.0, 2.0, 3.0]:
            cache({"x": x})
        assert count[0] == 3
        assert cache.hits == 0

    def test_lru_eviction(self):
        """Oldest entry evicted when cache is full."""
        call_count = [0]
        def obj(h):
            call_count[0] += 1
            return h["x"], 0.0

        cache = EvaluationCache(obj, maxsize=3)
        cache({"x": 1.0})
        cache({"x": 2.0})
        cache({"x": 3.0})
        assert cache.size == 3

        # Add a 4th — evicts x=1.0
        cache({"x": 4.0})
        assert cache.size == 3
        calls_before = call_count[0]

        # x=1.0 was evicted, should re-evaluate
        cache({"x": 1.0})
        assert call_count[0] == calls_before + 1

    def test_clear_resets(self):
        count = [0]
        cache = EvaluationCache(lambda h: (count.__setitem__(0, count[0]+1) or h["x"], 0.0))
        cache({"x": 1.0})
        cache({"x": 1.0})
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0
        cache({"x": 1.0})
        assert count[0] == 2   # re-evaluated after clear

    def test_stats_string(self):
        cache = EvaluationCache(lambda h: (h["x"], 0.0), maxsize=10)
        cache({"x": 1.0})
        cache({"x": 1.0})
        s = cache.stats()
        assert "hit" in s.lower()
        assert "50.0%" in s

    def test_cache_in_optimizer(self):
        space = _sphere_space()
        count = [0]
        def counted(h):
            count[0] += 1
            return _sphere(h)

        opt = Minimization(space, counted)
        opt.optimize(memory_size=5, max_iter=100, use_cache=True, cache_maxsize=256)
        assert opt._cache is not None
        assert opt._cache.hits > 0
        # Evaluations should be less than memory_size + max_iter
        assert count[0] < 5 + 100

    def test_no_cache_by_default(self):
        space = _sphere_space()
        opt = Minimization(space, _sphere)
        opt.optimize(memory_size=5, max_iter=20)
        assert opt._cache is None


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------

class TestRunLogger:
    def test_init_log_headers(self, tmp_path):
        init_csv = tmp_path / "init.csv"
        RunLogger(
            variable_names=["x", "y"],
            init_log_path=init_csv,
        )
        headers = init_csv.read_text().splitlines()[0].split(",")
        assert headers[0] == "harmony_index"
        assert "x" in headers
        assert "y" in headers
        assert "fitness" in headers
        assert "penalty" in headers
        assert "feasible" in headers

    def test_init_log_row_count(self, tmp_path):
        init_csv = tmp_path / "init.csv"
        logger = RunLogger(variable_names=["x"], init_log_path=init_csv)
        harmonies = [{"x": float(i)} for i in range(5)]
        fitnesses = [float(i)**2 for i in range(5)]
        penalties = [0.0] * 5
        logger.log_init(harmonies, fitnesses, penalties)
        rows = init_csv.read_text().splitlines()
        assert len(rows) == 6  # 1 header + 5 data

    def test_history_log_every(self, tmp_path):
        hist_csv = tmp_path / "hist.csv"
        logger = RunLogger(
            variable_names=["x"],
            history_log_path=hist_csv,
            history_every=5,
        )
        for it in range(1, 21):
            logger.log_iteration(it, {"x": float(it)}, float(it), 0.0)
        rows = hist_csv.read_text().splitlines()
        # iterations 5, 10, 15, 20 → 4 data rows + 1 header
        assert len(rows) == 5

    def test_history_log_every_1(self, tmp_path):
        hist_csv = tmp_path / "hist.csv"
        logger = RunLogger(variable_names=["x"], history_log_path=hist_csv,
                           history_every=1)
        for it in range(1, 11):
            logger.log_iteration(it, {"x": float(it)}, float(it), 0.0)
        rows = hist_csv.read_text().splitlines()
        assert len(rows) == 11  # 1 header + 10 data

    def test_eval_log_row_count(self, tmp_path):
        eval_csv = tmp_path / "evals.csv"
        logger = RunLogger(variable_names=["x"], eval_log_path=eval_csv)
        for i in range(7):
            logger.log_evaluation(i + 1, {"x": float(i)}, float(i), 0.0)
        rows = eval_csv.read_text().splitlines()
        assert len(rows) == 8  # 1 header + 7

    def test_feasible_column(self, tmp_path):
        init_csv = tmp_path / "init.csv"
        logger = RunLogger(variable_names=["x"], init_log_path=init_csv)
        logger.log_init([{"x": 1.0}, {"x": 2.0}], [1.0, 4.0], [0.0, 1.5])
        reader = list(csv.DictReader(init_csv.open()))
        assert reader[0]["feasible"] == "1"
        assert reader[1]["feasible"] == "0"

    def test_no_log_when_path_is_none(self):
        # Should not raise even with no paths
        logger = RunLogger(variable_names=["x"])
        logger.log_init([{"x": 1.0}], [1.0], [0.0])
        logger.log_evaluation(1, {"x": 1.0}, 1.0, 0.0)
        logger.log_iteration(1, {"x": 1.0}, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Resume parameter
# ---------------------------------------------------------------------------

class TestResume:
    def test_resume_auto_starts_fresh_when_no_file(self):
        space = _sphere_space()
        ckpt = _tmp_json()
        try:
            r = Minimization(space, _sphere).optimize(
                memory_size=5, max_iter=30,
                checkpoint_path=ckpt, checkpoint_every=30,
                resume="auto",
            )
            assert r.iterations == 30
        finally:
            ckpt.unlink(missing_ok=True)

    def test_resume_auto_continues_when_file_exists(self):
        space = _sphere_space()
        ckpt = _tmp_json()
        try:
            Minimization(space, _sphere).optimize(
                memory_size=5, max_iter=30,
                checkpoint_path=ckpt, checkpoint_every=30,
                resume="auto",
            )
            r2 = Minimization(space, _sphere).optimize(
                memory_size=5, max_iter=60,
                checkpoint_path=ckpt, checkpoint_every=60,
                resume="auto",
            )
            assert r2.iterations == 30
        finally:
            ckpt.unlink(missing_ok=True)

    def test_resume_new_overwrites_existing(self):
        space = _sphere_space()
        ckpt = _tmp_json()
        try:
            Minimization(space, _sphere).optimize(
                memory_size=5, max_iter=30,
                checkpoint_path=ckpt, checkpoint_every=30,
                resume="new",
            )
            # Second run with resume=new — should start fresh (60 iterations total)
            r2 = Minimization(space, _sphere).optimize(
                memory_size=5, max_iter=60,
                checkpoint_path=ckpt, checkpoint_every=60,
                resume="new",
            )
            assert r2.iterations == 60
        finally:
            ckpt.unlink(missing_ok=True)

    def test_resume_explicit_raises_when_no_file(self):
        space = _sphere_space()
        with pytest.raises(FileNotFoundError):
            Minimization(space, _sphere).optimize(
                memory_size=5, max_iter=10,
                checkpoint_path="/tmp/_harmonix_nonexistent_test.json",
                resume="resume",
            )

    def test_resume_invalid_value_raises(self):
        space = _sphere_space()
        with pytest.raises(ValueError):
            Minimization(space, _sphere).optimize(
                memory_size=5, max_iter=10,
                resume="invalid_option",
            )

    def test_init_checkpoint_written_immediately(self):
        """Checkpoint must exist after init (before any iterations)."""
        space = _sphere_space()
        ckpt = _tmp_json()
        try:
            Minimization(space, _sphere).optimize(
                memory_size=5, max_iter=20,
                checkpoint_path=ckpt, checkpoint_every=9999,
                resume="new",
            )
            assert ckpt.exists()
            data = json.loads(ckpt.read_text())
            # checkpoint_every=9999 means no mid-run saves, but init saves at 0
            assert "memory" in data
            assert data["memory"] is not None
        finally:
            ckpt.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Logging parameters in optimize()
# ---------------------------------------------------------------------------

class TestLoggingParameters:
    def test_log_init_creates_csv(self):
        space = _sphere_space()
        ckpt = _tmp_json()
        try:
            Minimization(space, _sphere).optimize(
                memory_size=5, max_iter=10,
                checkpoint_path=ckpt, log_init=True,
            )
            init_csv = ckpt.with_name(ckpt.stem + "_init.csv")
            assert init_csv.exists()
            rows = init_csv.read_text().splitlines()
            assert len(rows) == 6   # header + 5 harmonies
            init_csv.unlink()
        finally:
            ckpt.unlink(missing_ok=True)

    def test_log_history_row_count(self):
        space = _sphere_space()
        ckpt = _tmp_json()
        try:
            Minimization(space, _sphere).optimize(
                memory_size=5, max_iter=20,
                checkpoint_path=ckpt,
                log_history=True, history_every=5,
            )
            hist_csv = ckpt.with_name(ckpt.stem + "_history.csv")
            assert hist_csv.exists()
            rows = hist_csv.read_text().splitlines()
            assert len(rows) == 5   # header + iterations 5,10,15,20
            hist_csv.unlink()
        finally:
            ckpt.unlink(missing_ok=True)

    def test_log_evaluations_row_count(self):
        space = _sphere_space()
        ckpt = _tmp_json()
        try:
            Minimization(space, _sphere).optimize(
                memory_size=5, max_iter=10,
                checkpoint_path=ckpt, log_evaluations=True,
            )
            eval_csv = ckpt.with_name(ckpt.stem + "_evals.csv")
            assert eval_csv.exists()
            rows = eval_csv.read_text().splitlines()
            assert len(rows) == 11   # header + 10 evaluations
            eval_csv.unlink()
        finally:
            ckpt.unlink(missing_ok=True)

    def test_explicit_log_path(self, tmp_path):
        space = _sphere_space()
        hist_path = tmp_path / "my_history.csv"
        Minimization(space, _sphere).optimize(
            memory_size=5, max_iter=10,
            log_history=True, history_log_path=hist_path,
        )
        assert hist_path.exists()

    def test_no_log_files_by_default(self):
        """No CSV files created when logging flags are False."""
        space = _sphere_space()
        ckpt = _tmp_json()
        try:
            Minimization(space, _sphere).optimize(
                memory_size=5, max_iter=10,
                checkpoint_path=ckpt,
            )
            # Default: no log files
            assert not ckpt.with_name(ckpt.stem + "_init.csv").exists()
            assert not ckpt.with_name(ckpt.stem + "_history.csv").exists()
            assert not ckpt.with_name(ckpt.stem + "_evals.csv").exists()
        finally:
            ckpt.unlink(missing_ok=True)

    def test_history_values_monotone_feasible(self):
        """Best fitness in history should never increase (for feasible runs)."""
        space = _sphere_space()
        ckpt = _tmp_json()
        try:
            Minimization(space, _sphere).optimize(
                memory_size=10, max_iter=50,
                checkpoint_path=ckpt,
                log_history=True, history_every=1,
            )
            hist_csv = ckpt.with_name(ckpt.stem + "_history.csv")
            reader = list(csv.DictReader(hist_csv.open()))
            fitnesses = [float(r["best_fitness"]) for r in reader
                         if r["feasible"] == "1"]
            if len(fitnesses) > 1:
                assert all(fitnesses[i] >= fitnesses[i+1] - 1e-9
                           for i in range(len(fitnesses)-1))
            hist_csv.unlink()
        finally:
            ckpt.unlink(missing_ok=True)
