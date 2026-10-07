"""CPU contract checks: wall milestones must describe actual dispatch, including failure."""

import json
from unittest.mock import patch

import pytest

from test_recorded_simulation import (
    _fixture, _GeometryChecker, _initial_feedback, _limits, _result, _execute_bound,
)
from manipulation.execution import ExecutionGuardError, FixedFingerTargets, TrajectoryExecutor


@pytest.mark.parametrize("failure", [None, "preflight", "step"])
def test_first_dispatch_is_once_and_failure_does_not_invent_physics(tmp_path, failure):
    """A failed native call keeps dispatch; rejected preflight never creates it."""
    backend, curve, targets = _fixture()
    checker = _GeometryChecker(backend.joint_names, preflight_passed=failure != "preflight")
    executor = TrajectoryExecutor(backend, geometry_checker=checker)
    original_step, dispatches = backend.step, []

    def step():
        # Inspect the milestone at the physical boundary itself, before the
        # existing mock integration; repeated calls must retain the same value.
        event = executor.last_execution_report["wallclock"]["events"]["first_trajectory_step_dispatched"]
        dispatches.append(dict(event))
        if failure == "step":
            raise RuntimeError("explicit step dispatch failure")
        return original_step()

    with patch.object(backend, "step", step), patch("manipulation.robosuite_execution.mujoco.mj_forward"):
        def run():
            return _execute_bound(executor, _result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets),
                initial_feedback=_initial_feedback(backend), execution_limits=_limits(backend, curve),
                output=tmp_path/"trial")
        if failure:
            with pytest.raises((ExecutionGuardError, RuntimeError)):
                run()
        else:
            run()
    report = executor.last_execution_report
    events = report["wallclock"]["events"]
    assert "execute_called" in events
    assert all(type(row["unix_time_ns"]) is int and row["elapsed_seconds"] >= 0 for row in events.values())
    if failure == "preflight":
        assert set(events) == {"execute_called"}
        assert not dispatches and report["executed_steps"] == 0
    else:
        assert events["execute_called"]["elapsed_seconds"] <= events["preflight_completed"]["elapsed_seconds"]
        assert events["preflight_completed"]["elapsed_seconds"] <= events["first_trajectory_step_dispatched"]["elapsed_seconds"]
        assert dispatches and all(row == dispatches[0] for row in dispatches)
        if failure == "step":
            assert report["executed_steps"] == 0
        else:
            assert report["executed_steps"] > 1
    persisted = json.loads((tmp_path/"trial"/"execution-report.json").read_text())
    assert persisted["wallclock"] == report["wallclock"]
