"""CPU algebra checks for analytic references and native control adaptation.

The controller fixture exposes the documented robosuite mass/bias equation
with a coupled two-joint matrix.  It does not create a MuJoCo model, integrate
physics, or provide evidence of actual tracking/contact performance.
"""

# BEGIN focused regression coverage for the mandatory q/dq/ddq contract.
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from manipulation.planning import JointTrajectory
from manipulation.robosuite_execution import RobosuiteExecutionBackend


class _PositionController:
    """Declared coupled algebra oracle, with no physical state updates."""

    def __init__(self, backend):
        self.backend = backend
        self.mass_matrix = np.array([[2., .5], [.5, 1.5]])
        self.kp, self.kd = np.array([10., 20.]), np.array([3., 4.])
        self.torque_compensation = np.array([.7, -.2])
        self.goals = []
        self.before_run = lambda: None

    def set_goal(self, goal):
        self.goals.append(goal.copy())

    def run_controller(self):
        self.before_run()
        # This matches the mature controller's update order.  The production
        # adapter must read the just-updated matrix and current named state.
        indexes = self.backend._qpos_ids[:2]
        self.joint_pos = self.backend.native_data.qpos[indexes].copy()
        self.joint_vel = self.backend.native_data.qvel[indexes].copy()
        return (self.mass_matrix @ (self.kp*(self.goals[-1]-self.joint_pos)
                                   - self.kd*self.joint_vel) + self.torque_compensation)


def _backend():
    """Noncontiguous native indices exercise explicit named actuator mapping."""
    backend = RobosuiteExecutionBackend.__new__(RobosuiteExecutionBackend)
    backend.arm_joint_names = ("shoulder", "elbow")
    backend.finger_joint_names = ("left_finger", "right_finger")
    backend.joint_names = backend.arm_joint_names + backend.finger_joint_names
    backend._arm_actuators = np.array([2, 0])
    backend._finger_actuators = np.array([3, 1])
    backend._actuator_ids = backend._qpos_ids = backend._qvel_ids = np.array([2, 0, 3, 1])
    backend._joint_ids = np.array([2, 0, 3, 1])
    backend._actuator_names = ("shoulder_motor", "elbow_motor", "left_servo", "right_servo")
    backend._dt = .002
    backend._trial_id = "algebra-only-control-fixture"
    backend._feedback_source = "declared synthetic feedback; not physical measurements"
    backend.native_data = SimpleNamespace(qpos=np.array([-.2, -.01, .1, .01]),
        qvel=np.array([-.4, 0., .3, 0.]), qacc=np.array([9., 8., 7., 6.]),
        ctrl=np.array([7., -.01, 8., .01]), time=3.)
    backend.native_model = SimpleNamespace(
        actuator_ctrlrange=np.array([[-100., 100.], [-.04, 0.], [-100., 100.], [0., .04]]),
        jnt_range=np.array([[-2., 2.], [-.04, 0.], [-2., 2.], [0., .04]]))
    backend._arm_controller = _PositionController(backend)
    grip = SimpleNamespace(goal=None)
    grip.set_goal = lambda goal: setattr(grip, "goal", goal.copy())
    grip.run_controller = lambda: grip.goal.copy()
    backend._grip_controller = grip
    return backend


def _curve():
    # The first piece is q=t.  At t=1 the second piece starts as
    # q=1+u+4u^2-3u^3: q/dq agree, but ddq jumps from 0 to 8.
    positions = np.array([0., 1., 3.])
    velocities = np.array([1., 1., 0.])
    return JointTrajectory(("axis_a", "axis_b"), np.array([0., 1., 2.]),
        np.column_stack((positions, 5.-2.*positions)),
        np.column_stack((velocities, -2.*velocities)),
        np.array([0., 1.]), np.array([0., 1.]), (), (), {})


# BEGIN named feedback for hold-only tests with a scripted, nonphysical clock.
def _hold_feedback(backend):
    """Read the fixture's current state without correcting any invalid value."""
    return {"joint_names": backend.joint_names, "arm_joint_names": backend.arm_joint_names,
        "finger_joint_names": backend.finger_joint_names, "trial_id": backend.trial_id,
        "source": backend.feedback_source, "sim_time": backend.sim_time,
        "timestamp_s": backend.sim_time, "feedback_id": backend.feedback_id,
        "q": backend.native_data.qpos[backend._qpos_ids].copy(),
        "dq": backend.native_data.qvel[backend._qvel_ids].copy()}
# END no fixture callback integrates a native model or synthesizes a real stop.


class AnalyticReferenceTests(unittest.TestCase):
    def test_same_polynomial_supplies_position_velocity_acceleration_and_progress(self):
        curve = _curve()
        q, dq, ddq, progress = curve.sample(np.array([0., .4, 1., 1.25, 2.]))
        qa = np.array([0., .4, 1., 1.453125, 3.])
        va = np.array([1., 1., 1., 2.4375, 0.])
        aa = np.array([0., 0., 8., 3.5, -10.])
        np.testing.assert_allclose(q, np.column_stack((qa, 5.-2.*qa)), rtol=0., atol=1e-14)
        np.testing.assert_allclose(dq, np.column_stack((va, -2.*va)), rtol=0., atol=1e-14)
        np.testing.assert_allclose(ddq, np.column_stack((aa, -2.*aa)), rtol=0., atol=1e-14)
        np.testing.assert_array_equal(progress, [0., .2, .5, .625, 1.])

    def test_internal_right_and_last_left_acceleration_do_not_smooth_c1_jump(self):
        curve = _curve()
        np.testing.assert_array_equal(curve.sample(np.nextafter(1., 0.))[2], [0., 0.])
        np.testing.assert_array_equal(curve.sample(1.)[2], [8., -16.])
        np.testing.assert_array_equal(curve.sample(curve.duration)[2], [-10., 20.])
        # The physical entrance's nonzero derivative remains exactly saved.
        np.testing.assert_array_equal(curve.sample(0.)[1], [1., -2.])

    def test_out_of_interval_or_nonfinite_query_is_not_clamped_by_sampler(self):
        curve = _curve()
        for invalid in (-.001, 2.001, np.nan, np.inf, [0., np.nan]):
            with self.subTest(query=invalid), self.assertRaises(ValueError):
                curve.sample(invalid)


class TrajectoryTorqueTests(unittest.TestCase):
    def test_coupled_torque_components_and_native_indices_are_distinct(self):
        backend = _backend()
        q, dq, ddq = np.array([.15, -.25]), np.array([-.1, .2]), np.array([.8, -.5])
        physical_before = {name: getattr(backend.native_data, name).copy() for name in ("qpos", "qvel", "qacc")}
        output = backend.apply_control(q, dq, ddq, np.array([.025, -.03]))
        # Hand-calculated coupled terms independently catch swapped desired
        # velocity/acceleration inputs, omitted bias, and diagonal-only mass.
        expected_components = {
            "arm_position_feedback_torque": [.5, -1.25],
            "arm_velocity_feedback_torque": [-1., 1.95],
            "arm_velocity_feedforward_torque": [-.2, 1.05],
            "arm_acceleration_feedforward_torque": [1.35, -.35],
            "arm_bias_torque": [.7, -.2]}
        for name, expected in expected_components.items():
            np.testing.assert_allclose(output[name], expected, rtol=0., atol=1e-14)
        np.testing.assert_allclose(output["requested_arm_torque"], [1.35, 1.2], rtol=0., atol=1e-14)
        np.testing.assert_allclose(output["actuator_ctrl"], [1.35, 1.2, .025, -.03], rtol=0., atol=1e-14)
        np.testing.assert_allclose(backend.native_data.ctrl, [1.2, -.03, 1.35, .025], rtol=0., atol=1e-14)
        self.assertEqual(output["arm_control_saturated_coordinates"], 0)
        for name, before in physical_before.items():
            np.testing.assert_array_equal(getattr(backend.native_data, name), before)
        self.assertEqual(backend.native_data.time, 3.)
        for name, supplied in (("reference_arm_q", q), ("reference_arm_dq", dq), ("reference_arm_ddq", ddq)):
            np.testing.assert_array_equal(output[name], supplied)
            self.assertFalse(np.shares_memory(output[name], supplied))

    def test_velocity_feedforward_cancels_damping_before_native_clipping(self):
        backend = _backend()
        backend.native_model.actuator_ctrlrange[backend._arm_actuators] = [-1., 1.]
        backend._arm_controller.kd[:] = 200.
        backend.native_data.qvel[backend._qvel_ids[:2]] = [2., -3.]
        q = backend.native_data.qpos[backend._qpos_ids[:2]].copy()
        dq = backend.native_data.qvel[backend._qvel_ids[:2]].copy()
        output = backend.apply_control(q, dq, np.zeros(2), np.array([.01, -.01]))
        self.assertGreater(np.max(np.abs(output["arm_velocity_feedback_torque"])), 100.)
        np.testing.assert_allclose(output["requested_arm_torque"], [.7, -.2], rtol=0., atol=1e-12)
        self.assertFalse(output["arm_torque_saturated"].any())
        np.testing.assert_array_equal(backend.native_data.qvel[backend._qvel_ids[:2]], dq)

    def test_acceleration_uses_mass_from_the_current_controller_update(self):
        backend = _backend()
        controller = backend._arm_controller
        controller.before_run = lambda: setattr(controller, "mass_matrix", np.array([[3., .25], [.25, 4.]]))
        q = backend.native_data.qpos[backend._qpos_ids[:2]].copy()
        dq = backend.native_data.qvel[backend._qvel_ids[:2]].copy()
        output = backend.apply_control(q, dq, np.array([1., 2.]), np.array([.01, -.01]))
        np.testing.assert_allclose(output["requested_arm_torque"], [4.2, 8.05], rtol=0., atol=1e-14)
        np.testing.assert_array_equal(output["arm_acceleration_feedforward_torque"], [3.5, 8.25])

    def test_only_total_torque_is_clipped_and_raw_components_remain_available(self):
        backend = _backend()
        backend.native_model.actuator_ctrlrange[backend._arm_actuators] = [[-.5, .5], [-.6, .6]]
        output = backend.apply_control(np.array([.15, -.25]), np.array([-.1, .2]),
                                       np.array([.8, -.5]), np.array([.025, -.03]))
        np.testing.assert_allclose(output["requested_arm_torque"], [1.35, 1.2], rtol=0., atol=1e-14)
        np.testing.assert_array_equal(output["actuator_ctrl"], [.5, .6, .025, -.03])
        np.testing.assert_array_equal(output["arm_torque_saturated"], [True, True])
        self.assertEqual(output["arm_control_saturated_coordinates"], 2)
        np.testing.assert_allclose(output["arm_acceleration_feedforward_torque"], [1.35, -.35], rtol=0., atol=1e-14)

    def test_missing_or_invalid_derivatives_reject_before_any_goal_or_native_write(self):
        backend = _backend()
        before = backend.native_data.ctrl.copy()
        with self.assertRaises(TypeError):
            backend.apply_control(np.zeros(2), np.array([.01, -.01]))
        arguments = [np.zeros(2), np.zeros(2), np.zeros(2), np.array([.01, -.01])]
        for slot in range(3):
            for invalid in (None, [0.], [0., np.nan], [np.inf, 0.]):
                supplied = arguments.copy()
                supplied[slot] = invalid
                with self.subTest(slot=slot, invalid=invalid), self.assertRaises(ValueError):
                    backend.apply_control(*supplied)
                np.testing.assert_array_equal(backend.native_data.ctrl, before)
                self.assertEqual(backend._arm_controller.goals, [])
                self.assertIsNone(backend._grip_controller.goal)

    def test_hold_commands_explicit_zero_derivatives_without_resetting_actual_velocity(self):
        backend = _backend()
        # BEGIN a persisting finger violation must not truncate bounded arm damping.
        backend.native_data.qpos[backend._qpos_ids[2]] = np.nextafter(.04, np.inf)
        # END this test still deliberately never supplies a converged velocity.
        positions = backend.native_data.qpos[backend._qpos_ids].copy()
        velocities = backend.native_data.qvel[backend._qvel_ids].copy()

        def feedback():
            return {"joint_names": backend.joint_names, "arm_joint_names": backend.arm_joint_names,
                "finger_joint_names": backend.finger_joint_names, "trial_id": backend.trial_id,
                "source": backend.feedback_source, "sim_time": backend.sim_time,
                "timestamp_s": backend.sim_time, "feedback_id": backend.feedback_id,
                "q": positions.copy(), "dq": velocities.copy()}

        def mocked_step():
            # A test-only clock leaves the synthetic state untouched.  This
            # verifies hold inputs and budget exhaustion, not physical stopping.
            backend.native_data.time += backend.timestep

        backend.proprioception = feedback
        backend.step = mocked_step
        original_time = backend.sim_time
        with patch.object(backend, "apply_control", wraps=backend.apply_control) as applied:
            result = backend.abort_hold(feedback(), np.array([.01, -.01]), expected_sim_time=original_time,
                stop_velocity=np.array([.01, .01, .001, .001]), max_steps=2)
        self.assertTrue(result["prepared"])
        self.assertEqual(result["physical_steps"], 2)
        self.assertFalse(result["physical_stop_confirmed"])
        # BEGIN exhausted stop and outstanding position failure are independent.
        self.assertFalse(result["joint_position_bounds_satisfied"])
        self.assertEqual(len(result["joint_position_violation_samples"]), 3)
        # END retaining the violation does not invent a third hold step.
        self.assertEqual(applied.call_count, 2)
        np.testing.assert_array_equal(result["arm_velocity_goal"], np.zeros(2))
        np.testing.assert_array_equal(result["arm_acceleration_goal"], np.zeros(2))
        self.assertEqual(len(result["controls"]), 2)
        for index, control in enumerate(result["controls"]):
            self.assertEqual(control["control_sim_time"], original_time+index*backend.timestep)
            np.testing.assert_array_equal(control["reference_arm_dq"], np.zeros(2))
            np.testing.assert_array_equal(control["reference_arm_ddq"], np.zeros(2))
            self.assertIn("arm_velocity_feedback_torque", control)
            self.assertIn("requested_arm_torque", control)
            self.assertIn("arm_torque_saturated", control)
        for call in applied.call_args_list:
            q, dq, ddq, fingers = call.args
            np.testing.assert_array_equal(q, positions[:2])
            np.testing.assert_array_equal(dq, np.zeros(2))
            np.testing.assert_array_equal(ddq, np.zeros(2))
            np.testing.assert_array_equal(fingers, [.01, -.01])
        np.testing.assert_array_equal(backend.native_data.qvel[backend._qvel_ids], velocities)

    # BEGIN out-of-range actual fingers never become a new arm hold goal.
    def test_finger_violation_allows_multiple_hold_ticks_without_clamping_actual_state(self):
        for finger_recovers in (False, True):
            with self.subTest(finger_recovers=finger_recovers):
                backend = _backend()
                finger_index = backend._qpos_ids[2]
                overshoot = np.nextafter(.04, np.inf)
                backend.native_data.qpos[finger_index] = overshoot
                arm_goal = backend.native_data.qpos[backend._qpos_ids[:2]].copy()
                actual_q_before = backend.native_data.qpos.copy()
                actual_qacc_before = backend.native_data.qacc.copy()
                original_time = backend.sim_time
                original_dq = backend.native_data.qvel.copy()
                finger_ctrl = backend.finger_control.copy()
                seen = []

                def scripted_step():
                    # Only this test double changes telemetry. The first
                    # response remains moving, so a one-tick workaround fails.
                    seen.append(backend.native_data.qpos.copy())
                    backend.native_data.time += backend.timestep
                    backend.native_data.qvel[:] = original_dq*.5 if len(seen) == 1 else 0.
                    if finger_recovers and len(seen) == 2:
                        backend.native_data.qpos[finger_index] = .04

                backend.proprioception = lambda: _hold_feedback(backend)
                backend.step = scripted_step
                with patch.object(backend, "apply_control", wraps=backend.apply_control) as applied:
                    result = backend.abort_hold(backend.proprioception(), finger_ctrl,
                        expected_sim_time=original_time,
                        stop_velocity=np.array([.01, .01, .001, .001]), max_steps=3)
                self.assertTrue(result["prepared"])
                self.assertTrue(result["native_ctrl_written"])
                self.assertTrue(result["physical_stop_confirmed"])
                self.assertEqual(result["physical_steps"], 2)
                self.assertEqual(applied.call_count, 2)
                self.assertEqual(result["joint_position_bounds_satisfied"], finger_recovers)
                self.assertEqual(len(result["controls"]), 2)
                for call in applied.call_args_list:
                    q, dq, ddq, fingers = call.args
                    np.testing.assert_array_equal(q, arm_goal)
                    np.testing.assert_array_equal(dq, np.zeros(2))
                    np.testing.assert_array_equal(ddq, np.zeros(2))
                    np.testing.assert_array_equal(fingers, finger_ctrl)
                for actual in seen:
                    np.testing.assert_array_equal(actual, actual_q_before)
                np.testing.assert_array_equal(backend.finger_control, finger_ctrl)
                np.testing.assert_array_equal(backend.native_data.qacc, actual_qacc_before)
                self.assertEqual(backend.native_data.qpos[finger_index], .04 if finger_recovers else overshoot)
                violations = result["joint_position_violation_samples"]
                self.assertEqual(len(violations), 2 if finger_recovers else 3)
                for index, row in enumerate(violations):
                    self.assertEqual(row["phase"], "entry" if index == 0 else "after_step")
                    self.assertAlmostEqual(row["sim_time"], original_time+index*backend.timestep)
                    self.assertEqual(row["violations"], [{"joint_name": "left_finger",
                        "measured": overshoot, "lower": 0., "upper": .04}])

    def test_invalid_arm_hold_goal_and_invalid_finger_command_never_write_control(self):
        # The finite finger *measurement* exception must not admit a bad arm
        # target or bad native finger *command*. Existing executor tests cover
        # unidentified/nonfinite feedback and clock rejection independently.
        for invalid in ("arm_below", "arm_above", "finger_command"):
            with self.subTest(invalid=invalid):
                backend = _backend()
                before = backend.native_data.ctrl.copy()
                fingers = backend.finger_control.copy()
                if invalid.startswith("arm_"):
                    bound, direction = (-2., -np.inf) if invalid == "arm_below" else (2., np.inf)
                    backend.native_data.qpos[backend._qpos_ids[0]] = np.nextafter(bound, direction)
                else:
                    fingers[0] = np.nextafter(.04, np.inf)
                backend.proprioception = lambda: _hold_feedback(backend)
                with patch.object(backend, "step") as step:
                    result = backend.abort_hold(backend.proprioception(), fingers,
                        expected_sim_time=backend.sim_time,
                        stop_velocity=np.array([.01, .01, .001, .001]), max_steps=3)
                step.assert_not_called()
                self.assertFalse(result["native_ctrl_written"])
                self.assertFalse(result["physical_stop_confirmed"])
                self.assertEqual(result["physical_steps"], 0)
                np.testing.assert_array_equal(backend.native_data.ctrl, before)
                self.assertEqual(backend._arm_controller.goals, [])
                if invalid.startswith("arm_"):
                    self.assertFalse(result["prepared"])
                    self.assertFalse(result["joint_position_bounds_satisfied"])

    def test_new_arm_range_violation_during_hold_stops_further_commands(self):
        backend = _backend()
        original_time = backend.sim_time
        overshoot = np.nextafter(2., np.inf)

        def scripted_step():
            backend.native_data.time += backend.timestep
            backend.native_data.qpos[backend._qpos_ids[0]] = overshoot
            # Zero velocity cannot turn an invalidated arm hold into success.
            backend.native_data.qvel[:] = 0.

        backend.proprioception = lambda: _hold_feedback(backend)
        backend.step = scripted_step
        with patch.object(backend, "apply_control", wraps=backend.apply_control) as applied:
            result = backend.abort_hold(backend.proprioception(), backend.finger_control,
                expected_sim_time=original_time,
                stop_velocity=np.array([.01, .01, .001, .001]), max_steps=3)
        self.assertTrue(result["native_ctrl_written"])
        self.assertEqual(result["physical_steps"], 1)
        self.assertEqual(applied.call_count, 1)
        self.assertFalse(result["physical_stop_confirmed"])
        self.assertFalse(result["joint_position_bounds_satisfied"])
        self.assertEqual(result["joint_position_violation_samples"], [{"phase": "after_step",
            "sim_time": original_time+backend.timestep, "violations": [{"joint_name": "shoulder",
                "measured": overshoot, "lower": -2., "upper": 2.}]}])

    # BEGIN each post-step position verdict belongs to that exact feedback sample.
    def test_invalid_hold_positions_do_not_inherit_the_entry_domain_verdict(self):
        for invalid_q in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid_q=invalid_q):
                backend = _backend()
                original_time = backend.sim_time

                def scripted_step():
                    backend.native_data.time += backend.timestep
                    backend.native_data.qpos[backend._qpos_ids[2]] = invalid_q
                    # Otherwise converged telemetry must not certify a stop
                    # when the current position itself cannot be validated.
                    backend.native_data.qvel[:] = 0.

                backend.proprioception = lambda: _hold_feedback(backend)
                backend.step = scripted_step
                with patch.object(backend, "apply_control", wraps=backend.apply_control) as applied:
                    result = backend.abort_hold(backend.proprioception(), backend.finger_control,
                        expected_sim_time=original_time,
                        stop_velocity=np.array([.01, .01, .001, .001]), max_steps=3)
                self.assertTrue(result["native_ctrl_written"])
                self.assertEqual(result["physical_steps"], 1)
                self.assertEqual(applied.call_count, 1)
                self.assertEqual(len(result["samples"]), 1)
                self.assertFalse(result["physical_stop_confirmed"])
                self.assertIsNone(result["joint_position_bounds_satisfied"])
                self.assertEqual(result["joint_position_violation_samples"], [])
                self.assertIn("Hold actual positions", result["reason"])

    def test_invalid_hold_velocity_cannot_erase_a_finite_position_fault(self):
        backend = _backend()
        original_time = backend.sim_time
        overshoot = np.nextafter(.04, np.inf)

        def scripted_step():
            backend.native_data.time += backend.timestep
            backend.native_data.qpos[backend._qpos_ids[2]] = overshoot
            backend.native_data.qvel[backend._qvel_ids[2]] = np.nan

        backend.proprioception = lambda: _hold_feedback(backend)
        backend.step = scripted_step
        with patch.object(backend, "apply_control", wraps=backend.apply_control) as applied:
            result = backend.abort_hold(backend.proprioception(), backend.finger_control,
                expected_sim_time=original_time,
                stop_velocity=np.array([.01, .01, .001, .001]), max_steps=3)
        self.assertTrue(result["native_ctrl_written"])
        self.assertEqual(result["physical_steps"], 1)
        self.assertEqual(applied.call_count, 1)
        self.assertFalse(result["physical_stop_confirmed"])
        self.assertFalse(result["joint_position_bounds_satisfied"])
        self.assertIn("Hold actual velocities", result["reason"])
        self.assertEqual(result["joint_position_violation_samples"], [{"phase": "after_step",
            "sim_time": original_time+backend.timestep, "violations": [{"joint_name": "left_finger",
                "measured": overshoot, "lower": 0., "upper": .04}]}])

    def test_nonfinite_hold_clock_cannot_confirm_otherwise_stopped_feedback(self):
        for invalid_time in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid_time=invalid_time):
                backend = _backend()
                original_time = backend.sim_time

                def scripted_step():
                    backend.native_data.time = invalid_time
                    backend.native_data.qvel[:] = 0.

                backend.proprioception = lambda: _hold_feedback(backend)
                backend.step = scripted_step
                with patch.object(backend, "apply_control", wraps=backend.apply_control) as applied:
                    result = backend.abort_hold(backend.proprioception(), backend.finger_control,
                        expected_sim_time=original_time,
                        stop_velocity=np.array([.01, .01, .001, .001]), max_steps=3)
                self.assertTrue(result["native_ctrl_written"])
                self.assertEqual(result["physical_steps"], 1)
                self.assertEqual(applied.call_count, 1)
                self.assertFalse(result["physical_stop_confirmed"])
                self.assertIn("clock", result["reason"].lower())
    # END invalid fresh telemetry neither inherits old truth nor sends another control.
    # END tests distinguish stopped velocities, position validity and applied controls.


if __name__ == "__main__":
    unittest.main()
# END these unit tests make no physical success claim.
