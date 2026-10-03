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


if __name__ == "__main__":
    unittest.main()
# END these unit tests make no physical success claim.
