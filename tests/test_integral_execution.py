"""Finite executor contract tests; the clock double performs no native physics.

The real executor, curve evaluator, event binder and device rebinding
run against scripted robot feedback. The shared FUNCTO fixtures supply the same
formal evidence and event contracts used by their existing regression tests.
Controller algebra and native integral stepping are tested in their own module.
"""
from copy import deepcopy
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from manipulation.execution import ExecutionGuardError, FixedFingerTargets
from manipulation.execution_binding import curve_snapshot
from manipulation.planning import JointTrajectory
import test_recorded_simulation as fixtures
from test_recorded_simulation import (
    _bind_fingers, _execute_bound, _executor, _fixture, _initial_feedback, _limits, _result,
)


def _scripted_integral_backend(simulation):
    """Expose distinguishable control fields without modeling robot motion.

    The synthetic M=2I and explicit z make the recorded integral force
    independently identifiable. The clock leaves actual q/dq unchanged,
    matching the shared fixture's deliberate separation of command and state.
    """
    kp, kd = 1000., 200.
    slow = (kd - np.sqrt(kd * kd - 3 * kp)) / 3
    ki = slow * slow * (kd - 2 * slow)
    simulation._description["controller_policy"] = {
        "mode": "joint_trajectory_q_dq_ddq_integral",
        "arm_joint_names": list(simulation.arm_joint_names),
        "kp": [kp] * 7, "kd": [kd] * 7, "ki": [ki] * 7,
        "native_timestep_s": simulation.timestep,
        "reference_derivatives": ["q", "dq", "ddq"],
        "sampling_phase": "step_start",
        "acceleration_knot_side": "right_internal_last_left",
        "integral_update": "confirmed_native_step_torque_conditional",
        "torque_limits": [[-10., 10.]] * 7,
    }
    simulation.dispatches = []

    def apply_control(q, dq, ddq, fingers):
        index = len(simulation.dispatches)
        z = np.arange(1., 8.) * .01 + index * .001
        result = {
            "reference_arm_q": np.asarray(q).copy(),
            "reference_arm_dq": np.asarray(dq).copy(),
            "reference_arm_ddq": np.asarray(ddq).copy(),
            "arm_integral_state": z,
            "arm_integral_torque": 2 * z,
            "arm_position_error": np.asarray(q) - simulation.native_data.qpos[:7],
            # This is an explicit scripted initialization fact, not something
            # the executor infers from a curve name or an array difference.
            "arm_integral_rebased": index == 0,
        }
        parts = ("position_feedback", "velocity_feedback", "velocity_feedforward",
                 "acceleration_feedforward", "bias")
        for i, part in enumerate(parts):
            result["arm_" + part + "_torque"] = np.full(7, (i + 1) * .01)
        total = result["arm_integral_torque"] + sum(result["arm_" + part + "_torque"] for part in parts)
        applied = np.clip(total, -10., 10.)
        simulation.native_data.ctrl[:] = np.r_[applied, fingers]
        result.update(actuator_ctrl=simulation.native_data.ctrl.copy(),
            requested_arm_torque=total, arm_torque_saturated=applied != total,
            arm_control_saturated_coordinates=int(np.count_nonzero(applied != total)))
        # BEGIN explicitly scripted hand forces for the complete control record.
        result.update(finger_gravity_force=np.zeros(2), finger_position_servo_force=np.zeros(2),
            finger_requested_force=np.zeros(2), finger_bounded_force=np.zeros(2),
            finger_force_saturated=np.zeros(2, dtype=bool))
        # END this clock test does not infer task contact or physical forces.
        simulation.dispatches.append(dict(time=simulation.sim_time, fingers=np.asarray(fingers).copy(),
                                          **deepcopy(result)))
        return result

    def step():
        before = simulation.sim_time
        simulation._sim.step()
        simulation._step_attempts += 1
        simulation._last_step_accounting = {
            "attempt_index": simulation._step_attempts, "charged_steps": 1,
            "physical_steps": 1, "step_accounting_exact": True,
            "before_sim_time": before, "after_sim_time": simulation.sim_time,
            "clock_advance_confirmed": True,
        }
        return simulation.last_step_accounting

    simulation.apply_control = apply_control
    simulation.step = step
    return simulation


@pytest.mark.parametrize("duration", [.011, .012])
def test_same_time_arm_inputs_keep_endpoint_records_and_original_finger_events(tmp_path, duration):
    simulation, old, targets = _fixture()
    _scripted_integral_backend(simulation)
    old.events[2].update(kind="effect", requires_evidence=None, stop=False)
    q, dq = old.sample(0.)[:2]
    acceleration = np.full(7, .05)
    curve = JointTrajectory(old.joint_names, np.array([0., duration]),
        np.stack((q, q + dq * duration + .5 * acceleration * duration ** 2)),
        np.stack((dq, dq + acceleration * duration)), old.progress_u, old.progress_s,
        old.events, old.conditions, {})
    _bind_fingers(simulation, curve, targets)
    expected_events = deepcopy(curve.finger_trajectory.events)
    initial = _initial_feedback(simulation)
    output = tmp_path / "trial"
    report = _execute_bound(_executor(simulation), _result(curve),
        finger_targets=FixedFingerTargets.from_mapping(targets), execution_limits=_limits(simulation, curve),
        initial_feedback=initial, output=output)

    assert report["status"] == "curve_finished"
    assert report["executed_steps"] == 6
    assert report["last_curve_reference_time_s"] == duration
    assert report["last_arm_control_reference_time_s"] == .010
    assert report["barrier_reference_reached"]
    assert "last_command_sample_s" not in report
    assert "barrier_command_reached" not in report
    assert report["continuation"]["last_curve_time_s"] == duration

    with np.load(output / "physical-trajectory.npz", allow_pickle=False) as saved:
        # Actual arm inputs use the current control tick; state references
        # remain at the state time, including the final clamped endpoint.
        expected = curve.sample(saved["control_elapsed"])
        for key, values in zip(("reference_arm_q", "reference_arm_dq", "reference_arm_ddq"), expected[:3]):
            np.testing.assert_array_equal(saved[key], values)
        endpoint = curve.sample(saved["curve_time"])
        np.testing.assert_array_equal(saved["command_q"][:, :7], endpoint[0])
        np.testing.assert_array_equal(saved["arm_curve_dq"], endpoint[1])
        np.testing.assert_array_equal(saved["arm_curve_ddq"], endpoint[2])
        finger_times = np.minimum(saved["control_elapsed"] + simulation.timestep, duration)
        np.testing.assert_array_equal(saved["actuator_ctrl"][:, 7:], curve.finger_trajectory.sample(finger_times)[0])
        np.testing.assert_array_equal(saved["arm_position_error"], expected[0] - initial["q"][:7])
        np.testing.assert_array_equal(saved["arm_integral_torque"], 2 * saved["arm_integral_state"])
        np.testing.assert_array_equal(saved["arm_integral_rebased"], [True, False, False, False, False, False])
        components = [saved["arm_" + name + "_torque"] for name in (
            "position_feedback", "velocity_feedback", "velocity_feedforward",
            "acceleration_feedforward", "bias", "integral")]
        np.testing.assert_allclose(saved["requested_arm_torque"], sum(components), rtol=0., atol=1e-15)
        np.testing.assert_array_equal(saved["applied_arm_torque"], saved["actuator_ctrl"][:, :7])
    assert [(e["event_id"], e["applied_at_s"]) for e in report["events"]] == [
        (e["event_id"], e["applied_time_s"]) for e in expected_events]
    assert curve.finger_trajectory.events == expected_events


@pytest.mark.parametrize("missing_owner", [False, True])
def test_predicted_target_error_and_arrival_deadline_do_not_gate_close(tmp_path, missing_owner):
    relation = ({"target_owner_id": "unobserved_target", "T_owner_target": np.eye(4).tolist()}
                if missing_owner else None)
    simulation, curve, targets, result, limits = fixtures.ActualPoseGoalTests.fixture(actual_relation=relation)
    _scripted_integral_backend(simulation)
    # The nominal task goal is deliberately distant; the robot reference is
    # unchanged. A target-arrival deadline remains result-only evidence.
    result.execution_binding["problem"]["pose_events"][0]["target"][0][3] = .5
    curve.events[1]["timing"] = [fixtures.PhysicalEventTimingTests.window(
        "arrival", 0., .001, goal_id=["pose_events", 7])]
    _bind_fingers(simulation, curve, targets)
    result.execution_binding["curve"] = curve_snapshot(curve)
    result.execution_binding["validation"]["inputs"]["problem"] = deepcopy(result.execution_binding["problem"])
    with patch("recorded_skill_evaluation._actual_pose_goal",
               side_effect=AssertionError("Online execution must not call task-pose evaluation")):
        report = _execute_bound(_executor(simulation), result,
            finger_targets=FixedFingerTargets.from_mapping(targets), execution_limits=limits,
            initial_feedback=_initial_feedback(simulation), output=tmp_path / "trial")
    assert report["status"] == "curve_finished"
    assert report["pose_goal_evaluation"] == "simulation_posthoc_only"
    assert report["completed_goal_ids"] == []
    assert report["goal_evidence"] == []
    close = [event for event in report["events"] if event["kind"] == "close"]
    assert len(close) == 1 and close[0]["applied_at_s"] == .004
    assert not any(row["check_target"] == "arrival" for row in report["history"])


def test_robot_own_reference_tracking_check_remains_active(tmp_path):
    simulation, curve, targets, result, limits = fixtures.ActualPoseGoalTests.fixture()
    _scripted_integral_backend(simulation)
    simulation.scripted_tcp_estimate_offset[0] = .006
    executor = _executor(simulation)
    with pytest.raises(ExecutionGuardError):
        _execute_bound(executor, result, finger_targets=FixedFingerTargets.from_mapping(targets),
            execution_limits=limits, initial_feedback=_initial_feedback(simulation), output=tmp_path / "trial")
    report = executor.last_execution_report
    assert report["feedback_failure"]["violations"][0]["kind"] == "tcp_tracking"
    assert report["abort_hold"]["native_ctrl_written"]
    assert report["executed_steps"] == 0
