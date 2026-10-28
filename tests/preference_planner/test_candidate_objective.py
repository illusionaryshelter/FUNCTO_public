"""The exported-curve scorer preserves every numerical objective query.

An explicit algebraic robot and scene isolate time-grid identity from FK and
collision semantics. The maintained common objective and the production
backend scorer are both exercised, including an extra non-event query.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "tests"), str(PROJECT / "tests/planning")]

from manipulation.preference_planner.backend import PreferencePlanner
from manipulation.planning import _ProgressMap, _phase_aligned_parameters
from manipulation.preference_planner.objective import trajectory_objective_components
from test_objective import analytical_problem, identity_quaternions, AlgebraicScene

# BEGIN exact physical-query identity is part of common objective comparison.
@pytest.mark.parametrize("changed_clock", [False, True])
def test_exported_curve_score_matches_all_four_unscaled_terms(changed_clock):
    planner = object.__new__(PreferencePlanner)
    planner.problem = analytical_problem(process=True)
    planner.scene = AlgebraicScene()
    planner.budget = SimpleNamespace(knots=1, interpolation_steps=3, constraint_weight=7.)
    planner.formal_clock = dict(duration=9.3, native_dt=.03)
    planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    ordinary = np.linspace(0., 1., 16)
    parameters = np.unique(np.r_[ordinary, .5, .8, .23333333333333334])
    times = torch.tensor(parameters * 9.3, dtype=torch.float64)[None]
    progress = torch.tensor(parameters, dtype=torch.float64)[None]
    bounds = torch.tensor([[[-100.], [100.]]] * 4, dtype=torch.float64)
    planner.computation = SimpleNamespace(parameters=parameters, times=times, progress=progress,
        ordinary_indices=np.searchsorted(parameters, ordinary), bounds=bounds)
    planner.motion_configuration = dict(weight=[0.] * 4, squared_l2_regularization_weight=[2., 0., 0.],
        retime_weights=False, retime_regularization_weights=False, activation_distance=[0.] * 4)
    curve = SimpleNamespace(duration=9.3, progress_u=planner.progress_map.u,
                            progress_s=planner.progress_map.s)
    if changed_clock:
        curve = SimpleNamespace(duration=11.7, progress_u=np.array([0., .3, 1.]),
                                progress_s=np.array([0., .6, 1.]))
        mapping = _ProgressMap(curve.progress_u, curve.progress_s)
        expected_times = torch.tensor(_phase_aligned_parameters(planner.progress_map, mapping,
            parameters) * curve.duration, dtype=torch.float64)[None]
    else:
        expected_times = times
    queries = []

    def robot_state(trajectory, scene, actual_times, actual_progress):
        queries.append((actual_times.copy(), actual_progress.copy()))
        t = torch.tensor(actual_times, dtype=torch.float64)[None, :, None]
        p = torch.cat((.01 * t, .03 + .001 * t, t * 0.), -1)
        q, dq = p[..., :1], .01 + t * 0.
        states = (q, dq, q * 0., q * 0.)
        return states, (q, dq, ('joint',), {}, (p, identity_quaternions(p)), (p * 0., p * 0.))

    planner._candidate_robot_state = robot_state
    states, values = robot_state(curve, planner.scene, expected_times.numpy()[0], progress.numpy()[0])
    expected = trajectory_objective_components(problem=planner.problem, scene=planner.scene,
        states=states, robot_state=values, progress=progress, times=expected_times,
        ordinary_indices=planner.computation.ordinary_indices, bounds=bounds,
        motion_configuration=planner.motion_configuration, native_dt=.03, constraint_weight=7.)
    scored = planner._candidate_objective(curve, planner.scene)
    assert scored['objective_components'] == pytest.approx({k: float(v) for k, v in expected.items()}, abs=0., rel=0.)
    assert scored['fixed_objective'] == sum(float(v) for v in expected.values())
    np.testing.assert_array_equal(queries[0][0], queries[1][0])
    np.testing.assert_array_equal(queries[0][1], queries[1][1])
    assert len(queries[1][0]) == len(parameters)
# END a duplicated floating-point time cannot silently change preference weight.
