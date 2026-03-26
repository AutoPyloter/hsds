"""
tests/test_coverage_gaps.py
===========================
Targeted tests to cover specific branches and error handlers identified
as coverage gaps by Codecov.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure harmonix is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harmonix.optimizer import (
    HarmonyMemory,
    Minimization,
    MultiObjective,
)
from harmonix.space import DesignSpace
from harmonix.spaces.engineering import (
    SoilSPT,
    _DynamicGridVariable,
)
from harmonix.variables import Continuous

# ---------------------------------------------------------------------------
# harmonix/optimizer.py
# ---------------------------------------------------------------------------


class TestOptimizerGaps:
    def test_improvise_no_memory(self):
        """Covers optimizer.py:242 (source = []) when no memory exists."""
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        # Initialise base class directly to bypass auto-memory init if possible
        from harmonix.optimizer import HarmonySearchOptimizer

        class MockOptimizer(HarmonySearchOptimizer):
            def optimize(self, **kwargs):
                return self._improvise(hmcr=0.8, par=0.3)

        opt = MockOptimizer(space, lambda h: (h["x"], 0.0))
        # self._memory is None by default in __init__
        res = opt.optimize()
        assert "x" in res

    def test_parse_optimize_kwargs_invalid(self):
        """Covers optimizer.py:340-341 (Unexpected keyword argument(s))."""
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        opt = Minimization(space, lambda h: (h["x"], 0.0))
        with pytest.raises(TypeError, match="Unexpected keyword argument"):
            opt.optimize(invalid_param=123)

    def test_parse_optimize_kwargs_negative_iter(self):
        """Covers optimizer.py:345 (max_iter must be non-negative)."""
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        opt = Minimization(space, lambda h: (h["x"], 0.0))
        with pytest.raises(ValueError, match="max_iter must be non-negative"):
            opt.optimize(max_iter=-1)

    def test_checkpoint_path_traversal_safety(self, tmp_path):
        """Covers optimizer.py:552 (target_path = base_path / target_path.name)."""
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        opt = Minimization(space, lambda h: (h["x"], 0.0))
        opt._memory = HarmonyMemory(size=1)
        opt._memory.add({"x": 0.5}, 0.5, 0.0)

        # Attempt to save outside (traversal)
        # We use a path that is likely outside both CWD and /tmp on all platforms.
        traversal_path = Path("/secret.json")
        opt.save_checkpoint(traversal_path, 10)

        # It should have fallen back to Path.cwd() / "secret.json"
        fallback = Path.cwd() / "secret.json"
        try:
            assert fallback.exists()
        finally:
            fallback.unlink(missing_ok=True)

    def test_checkpoint_exception_handlers(self, tmp_path):
        """Covers optimizer.py:540 and 547-548 (except Exception blocks)."""
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        opt = Minimization(space, lambda h: (h["x"], 0.0))
        opt._memory = HarmonyMemory(size=1)
        opt._memory.add({"x": 0.5}, 0.5, 0.0)

        ckpt = tmp_path / "test_exc.json"

        # Mock tempfile.gettempdir to raise
        with patch("tempfile.gettempdir", side_effect=ValueError("boom")):
            opt.save_checkpoint(ckpt, 10)
            # It should have fallen back to CWD because tmp_dir is not temp_base or base_path
            fallback = Path.cwd() / ckpt.name
            try:
                assert fallback.exists()
            finally:
                fallback.unlink(missing_ok=True)

        # Note: We skip patching builtins.str as it's flagged by pre-commit hooks.
        # Coverage for the internal exception handler is provided by test_is_relative_robust_exception.

        # Simpler: just ensure the resolve failure is handled where it's supposed to be.
        with patch("pathlib.Path.resolve", side_effect=RuntimeError("resolve error")):
            with pytest.raises(RuntimeError, match="resolve error"):
                opt.save_checkpoint(ckpt, 10)

    def test_is_relative_robust_exception(self):
        """Directly trigger the catch block in _is_relative_robust."""
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        opt = Minimization(space, lambda h: (h["x"], 0.0))
        opt._memory = HarmonyMemory(size=1)
        opt._memory.add({"x": 0.5}, 0.5, 0.0)

        # Create a mock that raises in str() but has necessary attributes
        mock_p = MagicMock()
        mock_p.resolve.return_value = mock_p
        mock_p.name = "evil.json"
        # __str__ must be a real method returning a mock that raises on lower()
        # __str__ must raise to trigger the catch block
        mock_p.__str__ = MagicMock(side_effect=RuntimeError("catch me"))

        with patch("harmonix.optimizer.Path") as mock_path_cls:
            mock_path_cls.side_effect = lambda x: mock_p if x == "evil_trigger" else Path(x)
            mock_path_cls.cwd = Path.cwd
            # Now call save_checkpoint
            opt.save_checkpoint("evil_trigger", 10)

        fallback = Path.cwd() / "evil.json"
        try:
            assert fallback.exists()
        finally:
            fallback.unlink(missing_ok=True)

    def test_optimize_resume_at_max_iter(self, tmp_path):
        """Covers optimizer.py:641 and 1034 (start_iter = max_iter)."""
        space = DesignSpace()
        space.add("x", Continuous(0.0, 1.0))
        opt = Minimization(space, lambda h: (h["x"], 0.0))

        ckpt = tmp_path / "at_max.json"
        opt._memory = HarmonyMemory(size=5)
        for i in range(5):
            opt._memory.add({"x": 0.1 * i}, 0.1 * i, 0.0)
        opt.save_checkpoint(ckpt, 100)

        # Resume with max_iter=50 (where checkpoint is at 100)
        res = opt.optimize(max_iter=50, checkpoint_path=ckpt, resume="resume")
        assert res.iterations == 0

        # Same for MultiObjective
        def mo_obj(h):
            return (h["x"],), 0.0

        opt_mo = MultiObjective(space, mo_obj)
        # Create a valid MO checkpoint
        payload = {"iteration": 100, "memory": opt._memory.to_dict(), "archive": {"max_size": 10, "entries": []}}
        ckpt_mo = tmp_path / "at_max_mo.json"
        ckpt_mo.write_text(json.dumps(payload))

        res_mo = opt_mo.optimize(max_iter=50, checkpoint_path=ckpt_mo, resume="resume")
        assert len(res_mo.archive_history) == 0


# ---------------------------------------------------------------------------
# harmonix/spaces/engineering.py
# ---------------------------------------------------------------------------


class TestEngineeringGaps:
    def test_dynamic_grid_not_implemented(self):
        """Covers engineering.py:116 (raise NotImplementedError)."""
        var = _DynamicGridVariable()
        with pytest.raises(NotImplementedError):
            var._valid_codes({})

    def test_dynamic_grid_neighbor_edge_cases(self):
        """Covers engineering.py:139 (continue in neighbor loop)."""
        # ACIRebar is a subclass of _DynamicGridVariable
        from harmonix.spaces.engineering import ACIRebar

        var = ACIRebar(d_expr=0.5, cc_expr=40.0)

        # Force a value at the edge of the grid
        # _n is len(_COUNTS) which is 38 (4 to 41)
        # _DIAMETERS has 12 entries.
        # Max code = (11 * 38) + 37 = 455
        # Code 0 (dia 0, count 0) is an edge.

        # We need a context where code 0 is valid.
        # ctx = {}  # removed unused
        # Actually _valid_codes is called in neighbor.
        # If we pick a code at the very corner, some Moore offsets will be < 0 or > max.

        # Let's find a valid code at an edge.
        valid = var._valid_codes({})
        if valid:
            code = valid[0]  # Usually a small code
            # Testing neighbor(code) should trigger 'continue' for out-of-bounds offsets.
            res = var.neighbor(code, {})
            assert res in valid or res == code

    def test_soil_spt_invalid_kwargs(self):
        """Covers engineering.py:949 (Unexpected keyword argument(s))."""
        with pytest.raises(TypeError, match="Unexpected keyword argument"):
            SoilSPT(unknown_arg=True)

    def test_soil_spt_legacy_kwargs(self):
        """Ensures legacy kwargs are handled (and thus the block before error is covered)."""
        spt = SoilSPT(N_min=10, N_max=30)
        assert spt is not None
