"""Actual SciPy calls check the localized adapter against a known solution.

The small convex problem supplies analytical task and physical conditions.
It tests solver integration and the task-free objective allocation boundary;
robot, observed-field and full-scene behavior have independent tests/runs.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "tests"), str(PROJECT / "tests/planning")]


from manipulation.unified_solver import solve_trajectory as original_solve
from manipulation.preference_planner.solver import solve_trajectory


class QuadraticCase:
    """Minimize distance to (2,-2), optionally requiring x+y=0 and x<=1."""

    def __init__(self, constrained):
        self.constrained = constrained
        self.parameters = np.array([0., 1.])
        self.fixed_equality_rows = np.array([], dtype=int)
        self.equality_keys = ["sum"] if constrained else []
        self.kinematics = SimpleNamespace(frames=("frame",), joint_names=("x", "y"))
        self.geometry = SimpleNamespace(workspace_bytes=lambda: {})
        self.geometry_report = {}
        self.max_host_rss_bytes = 12 * 1024**3
        self.domain = SimpleNamespace(intervals=1, signed_constraints=self.physical)

    def physical(self, controls):
        """The fixed box [-3,3]^2 repeats to the cubic interface's row count."""
        matrix = np.tile(np.r_[np.eye(2), -np.eye(2)], (5, 1))
        return matrix @ controls.ravel() - 3., matrix

    def task(self, controls, *, solver_rows_out=None):
        empty = controls.ravel()[:0]
        if not self.constrained:
            return empty, empty
        if solver_rows_out is not None:
            solver_rows_out[0]["sum"] = torch.ones(1, dtype=torch.bool)
            solver_rows_out[1]["upper"] = torch.ones(1, dtype=torch.bool)
        return controls.sum().reshape(1), controls.ravel()[:1] - 1.

    def objective_components(self, controls):
        return {"quadratic": .5 * (controls - controls.new_tensor([[2., -2.]])).square().sum()}


@pytest.mark.parametrize("constrained", [False, True])
def test_localized_adapter_preserves_solution_and_empty_task_reverse_storage(constrained):
    """Compare two real solver calls and independently known convex optima."""
    results, reports = [], []
    for solve in (original_solve, solve_trajectory):
        report = {}
        results.append(solve(QuadraticCase(constrained), np.zeros((1, 2)),
            iterations=20, check_request=lambda: None, pending_geometry_bytes=0,
            report_out=report))
        reports.append(report)
    expected = [[1., -1.]] if constrained else [[2., -2.]]
    for result in results:
        assert result.report["success"]
        np.testing.assert_allclose(result.controls, expected, atol=1e-10, rtol=0.)
    np.testing.assert_allclose(results[0].controls, results[1].controls, atol=1e-12, rtol=0.)
    memory = reports[1]["solver_storage_bytes"]
    assert memory["batched_base_pinocchio_reverse"] > 0
    assert (memory["batched_returned_dense_jacobian"] > 0) == constrained
    assert reports[1]["latest_complete_trial"]["completed_components"]
