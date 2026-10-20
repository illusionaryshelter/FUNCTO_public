"""CPU checks for current-curve diagnostic probes, including infeasible trials."""
# BEGIN numerical probe tests do not require cuRobo or GPU execution.
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'planning'))

from manipulation.constrained_trajectory import CurveEvaluation
from manipulation.physical_curve import PhysicalCurve
from probe_process_gradient import gradient_diagnostics


class AnalyticObjective:
    """Real continuous joint constraints with a known differentiable objective."""
    def __init__(self):
        bounds = {kind: np.array([[-limit], [limit]])
                  for kind, limit in (('position', 1.), ('velocity', 10.),
                                      ('acceleration', 20.), ('jerk', 100.))}
        bounds['position'][0] = 0.
        self.physical = PhysicalCurve([0.], [0.], 6, 1., np.linspace(0., 8., 33), ('slide',), bounds,
                                      joint_tolerance=1e-6, velocity_tolerance=1e-6)
        self.calls = []

    def evaluate(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        self.calls.append(actions.copy())
        physical = self.physical.constraints(actions)
        difference = actions-np.linspace(.1, .6, actions.size).reshape(actions.shape)
        return CurveEvaluation(actions.copy(), float(.5*np.square(difference).sum()), difference.reshape(-1),
            physical.slack, physical.jacobian, physical.names, physical.fixed_slack, physical.fixed_names,
            physical, {'groups': {}, 'fixed_failures': {}, 'diagnostics': {}})


def test_probe_evaluates_both_infeasible_sides_without_domain_clipping():
    solver = AnalyticObjective()
    controls = np.zeros((6, 1))
    report = gradient_diagnostics(solver, controls)
    assert len(solver.calls) == 5
    assert report['baseline_accepted']
    assert report['curve_saved_node_count'] == 33
    direction = np.asarray(report['direction']).reshape(controls.shape)
    for index, row in enumerate(report['differences']):
        epsilon = row['epsilon']
        np.testing.assert_allclose(solver.calls[1+2*index], controls+epsilon*direction, rtol=0, atol=0)
        np.testing.assert_allclose(solver.calls[2+2*index], controls-epsilon*direction, rtol=0, atol=0)
        assert max(row['positive_violation'], row['negative_violation']) > 0
        assert not (row['positive_accepted'] and row['negative_accepted'])
        assert row['objective_derivative'] == pytest.approx(report['analytic_objective_derivative'], abs=1e-10)
        assert len(row['constraint_derivative']) == len(report['constraint_names'])
    np.testing.assert_array_equal(controls, np.zeros_like(controls))


def test_probe_retains_objective_and_signed_constraint_derivatives_as_distinct_values():
    solver = AnalyticObjective()
    controls = np.linspace(.1, .6, 6).reshape(6, 1)
    report = gradient_diagnostics(solver, controls)
    assert report['objective_gradient_max'] == 0.
    assert report['analytic_objective_derivative'] == 0.
    assert np.any(np.asarray(report['analytic_constraint_derivative']) != 0.)
    for row in report['differences']:
        assert row['objective_absolute_error'] < 1e-12
    assert report['constraint_names'] == list(solver.physical.constraints(controls).names)
# END a stationary objective does not erase physical-constraint derivatives.
