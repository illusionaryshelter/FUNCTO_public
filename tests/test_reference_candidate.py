# BEGIN tests migrated to native motion attempts and retained numerical contracts.
"""CPU contracts for preserving supplied curves at the shared delivery boundary.

Native geometry numerics remain integration coverage. These tests substitute
only its completed metric evidence, leaving request ordering, clock binding,
Hermite/domain checks and the early-return policy in production code.
"""
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from manipulation.planning import (
    JointTrajectory, PlanningBudget, _CuroboPlanner, _ProgressMap,
    _phase_aligned_parameters,
)
from manipulation.planning_problem import PlanningEvent
from test_motion_backend_contract import Native, Scene, problem, run




def supplied_curve():
    times = np.array([0., .13, .71, 1.])
    return JointTrajectory(("q",), times, (.1*times**2)[:, None], (.2*times)[:, None],
        np.array([0., .3, .6, 1.]), np.array([0., .5, .5, 1.]), (), (), {})


def test_phase_mapping_keeps_moving_plateau_and_first_arrival_of_added_hold():
    source = _ProgressMap(np.array([0., .3, .6, 1.]), np.array([0., .5, .5, 1.]))
    target = _ProgressMap(np.array([0., .2, .7, .9, 1.]), np.array([0., .5, .5, 1., 1.]))
    np.testing.assert_allclose(_phase_aligned_parameters(source, target, np.array([.3, .45, .6, 1.])),
                               [.2, .45, .7, .9], rtol=0, atol=2e-16)
    with pytest.raises(Exception, match="plateau motion"):
        _phase_aligned_parameters(source, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), np.array([.45]))




def test_terminal_evidence_after_last_tick_is_typed_unchecked_without_native_allocation():
    from manipulation.planning import MotionBackend
    request = problem()
    request.events = (PlanningEvent("terminal", "skill", 1., 197, "verification", "recorded terminal",
                                   "actual evidence", True),)
    duration = 9.850000000000293
    reference = JointTrajectory(("q",), np.array([0., duration]), np.zeros((2, 1)), np.zeros((2, 1)),
        np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    scene = Scene()
    scene.gripper = SimpleNamespace(control_timestep_s=.002)
    backend = MotionBackend({}, robot_validator=object())
    with patch("manipulation.planning._CuroboPlanner", side_effect=AssertionError("must reject before native allocation")) as factory:
        result = backend.plan(request, scene, joint_reference=reference)
    assert not result.success and result.execution_binding is None
    assert result.diagnostics["failure"] == "reference_clock_outside_curve"
    issue = result.diagnostics["reference_clock_issues"][0]
    assert issue["device_step"] == 4926 and issue["device_time_s"] == 9.852
    assert issue["curve_duration_s"] == duration and issue["status"] == "outside_curve_unchecked"
    assert result.request_ledger["consumed_planning"]["initializations"] == 0
    factory.assert_not_called()
    np.testing.assert_array_equal(reference.times, [0., duration])
    backend.close()
# END native attempt-planner test migration.
