"""CPU algebra checks for analytic references and native control adaptation.

The controller fixture exposes the documented robosuite mass/bias equation
with a coupled two-joint matrix. A separate tiny model checks zero-step FK.
No test integrates physics or proves actual tracking/contact performance.
"""

# BEGIN focused regression coverage for the mandatory q/dq/ddq contract.
from types import SimpleNamespace
import unittest
from unittest.mock import PropertyMock, patch

import numpy as np
# BEGIN the boundary regression uses only MuJoCo model FK, with zero steps.
import mujoco
# END no renderer or physics rollout is imported by this regression.

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
    # BEGIN the mock retains the production native-boundary accounting path.
    backend._step_attempts = 0
    backend._last_step_accounting = None
    backend._sim = SimpleNamespace(step=lambda: setattr(
        backend.native_data, "time", backend.native_data.time+backend.timestep))
    # END only the native integration itself is replaced by a scripted clock.
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
    # BEGIN no native model is constructed or forwarded by these mock tests.
    def setUp(self):
        forward = patch("manipulation.robosuite_execution.mujoco.mj_forward")
        self.forward = forward.start()
        self.addCleanup(forward.stop)
    # END production step accounting still runs around the mock native call.

    # BEGIN the installed joint controller works without live site observations.
    def test_installed_joint_controller_without_reference_preserves_torque(self):
        from robosuite.controllers.parts.generic.joint_pos import JointPositionController

        backend = _backend()
        # This simulator double has no site pose or Jacobian interface. The
        # model terms are explicit algebra, and forward performs no native work.
        backend.native_data.qM = np.zeros(10)
        backend.native_data.qfrc_bias = np.array([-.2, 0., .7, 0.])
        sim = SimpleNamespace(data=backend.native_data, forward=lambda: None,
            model=SimpleNamespace(nv=4, _model=object(),
                joint_id2name=lambda index: ("elbow", "right_finger", "shoulder", "left_finger")[index]))
        full_mass = np.eye(4)
        full_mass[np.ix_([2, 0], [2, 0])] = [[2., .5], [.5, 1.5]]

        def unpack_mass(model, destination, packed):
            destination[:] = full_mass

        with patch("mujoco.mj_fullM", side_effect=unpack_mass):
            backend._arm_controller = JointPositionController(sim,
                {"joints": [2, 0], "qpos": [2, 0], "qvel": [2, 0]},
                actuator_range=(np.full(2, -100.), np.full(2, 100.)), ref_name=None,
                input_type="absolute", kp=[10., 20.], kd=[3., 4.], lite_physics=False,
                interpolator=None, use_torque_compensation=True)
            output = backend.apply_control(np.array([.15, -.25]), np.array([-.1, .2]),
                np.array([.8, -.5]), np.array([.025, -.03]))
        self.assertEqual(backend._arm_controller.num_ref_sites, 0)
        self.assertEqual(backend._arm_controller.initial_ref_pos.shape, (0, 3))
        np.testing.assert_allclose(output["requested_arm_torque"], [1.35, 1.2], rtol=0., atol=1e-14)
        np.testing.assert_allclose(output["arm_bias_torque"], [.7, -.2], rtol=0., atol=1e-14)
        np.testing.assert_allclose(output["arm_acceleration_feedforward_torque"], [1.35, -.35], rtol=0., atol=1e-14)
        self.assertEqual(backend.native_data.time, 3.)
    # END the unchanged coupled law consumes model terms and encoder q/dq only.

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
        backend._sim.step = mocked_step
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
                backend._sim.step = scripted_step
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
        backend._sim.step = scripted_step
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
                backend._sim.step = scripted_step
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
        backend._sim.step = scripted_step
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
                backend._sim.step = scripted_step
                with patch.object(backend, "apply_control", wraps=backend.apply_control) as applied:
                    result = backend.abort_hold(backend.proprioception(), backend.finger_control,
                        expected_sim_time=original_time,
                        stop_velocity=np.array([.01, .01, .001, .001]), max_steps=3)
                self.assertTrue(result["native_ctrl_written"])
                self.assertEqual(result["physical_steps"], 1)
                self.assertEqual(applied.call_count, 1)
                self.assertFalse(result["physical_stop_confirmed"])
                self.assertIn("clock", result["reason"].lower())
                self.assertEqual(result["charged_steps"], 1)
                self.assertFalse(result["step_accounting_exact"])
    # END invalid fresh telemetry neither inherits old truth nor sends another control.
    # END tests distinguish stopped velocities, position validity and applied controls.

    # BEGIN completed integration survives every later feedback failure.
    def test_forward_failure_preserves_one_confirmed_and_charged_native_step(self):
        backend = _backend()
        primary = RuntimeError("derived feedback failed after integration")
        self.forward.side_effect = primary
        with self.assertRaises(RuntimeError) as raised:
            backend.step()
        self.assertIs(raised.exception, primary)
        accounting = backend.last_step_accounting
        self.assertEqual(accounting["physical_steps"], 1)
        self.assertEqual(accounting["charged_steps"], 1)
        self.assertTrue(accounting["step_accounting_exact"])
        self.assertTrue(accounting["native_step_returned"])
        self.assertTrue(accounting["clock_advance_confirmed"])
        self.assertEqual(accounting["attempt_index"], 1)
        accounting["physical_steps"] = 99
        self.assertEqual(backend.last_step_accounting["physical_steps"], 1)

    def test_native_exception_accounts_advanced_stationary_and_unknown_clocks(self):
        for after, confirmed, exact in ((3.002, 1, True), (3., 0, True),
                                         (3.003, 0, False), (np.nan, 0, False)):
            with self.subTest(after=after):
                backend = _backend()
                primary = RuntimeError("native integration failed")
                def native_step():
                    backend.native_data.time = after
                    raise primary
                backend._sim.step = native_step
                with self.assertRaises(RuntimeError) as raised:
                    backend.step()
                self.assertIs(raised.exception, primary)
                accounting = backend.last_step_accounting
                self.assertEqual(accounting["physical_steps"], confirmed)
                self.assertEqual(accounting["charged_steps"], 1)
                self.assertEqual(accounting["step_accounting_exact"], exact)
                self.assertFalse(accounting["native_step_returned"])
        self.forward.assert_not_called()

    def test_unreadable_post_exception_clock_does_not_replace_native_error(self):
        backend = _backend()
        primary = RuntimeError("original native error")
        class UnreadableClock:
            def __float__(self):
                raise ValueError("clock read also failed")
        def native_step():
            backend.native_data.time = UnreadableClock()
            raise primary
        backend._sim.step = native_step
        with self.assertRaises(RuntimeError) as raised:
            backend.step()
        self.assertIs(raised.exception, primary)
        accounting = backend.last_step_accounting
        self.assertEqual(accounting["charged_steps"], 1)
        self.assertEqual(accounting["physical_steps"], 0)
        self.assertFalse(accounting["step_accounting_exact"])
        self.assertIn("clock read also failed", accounting["clock_error"])

    def test_hold_counts_native_step_before_forward_or_proprioception_failure(self):
        for failure in ("forward", "proprioception"):
            with self.subTest(failure=failure):
                backend = _backend()
                initial = _hold_feedback(backend)
                primary = RuntimeError(f"{failure} failed after integration")
                self.forward.side_effect = primary if failure == "forward" else None
                backend.proprioception = lambda: (_ for _ in ()).throw(primary)
                result = backend.abort_hold(initial, backend.finger_control,
                    expected_sim_time=3., stop_velocity=np.array([.01, .01, .001, .001]), max_steps=3)
                self.assertEqual(result["physical_steps"], 1)
                self.assertEqual(result["charged_steps"], 1)
                self.assertTrue(result["step_accounting_exact"])
                self.assertFalse(result["physical_stop_confirmed"])
                self.assertIsNone(result["joint_position_bounds_satisfied"])
                self.assertEqual(result["samples"], [])
                self.assertIn(str(primary), result["reason"])
                self.assertEqual(backend._step_attempts, 1)

    def test_hold_native_failure_charges_attempt_without_free_retry(self):
        for after, confirmed, exact in ((3., 0, True), (3.002, 1, True), (3.003, 0, False)):
            with self.subTest(after=after):
                backend = _backend()
                initial = _hold_feedback(backend)
                def native_step():
                    backend.native_data.time = after
                    raise RuntimeError("native hold failure")
                backend._sim.step = native_step
                backend.proprioception = lambda: _hold_feedback(backend)
                result = backend.abort_hold(initial, backend.finger_control,
                    expected_sim_time=3., stop_velocity=np.array([.01, .01, .001, .001]), max_steps=3)
                self.assertEqual(result["physical_steps"], confirmed)
                self.assertEqual(result["charged_steps"], 1)
                self.assertEqual(result["step_accounting_exact"], exact)
                self.assertEqual(backend._step_attempts, 1)
                self.assertFalse(result["physical_stop_confirmed"])
                self.assertIn("native hold failure", result["reason"])

    def test_hold_accounting_read_failure_does_not_replace_native_failure(self):
        backend = _backend()
        initial = _hold_feedback(backend)
        backend._sim.step = lambda: (_ for _ in ()).throw(RuntimeError("primary native failure"))
        with patch.object(RobosuiteExecutionBackend, "last_step_accounting", new_callable=PropertyMock,
                          side_effect=ValueError("secondary accounting failure")):
            result = backend.abort_hold(initial, backend.finger_control,
                expected_sim_time=3., stop_velocity=np.array([.01, .01, .001, .001]), max_steps=3)
        self.assertIn("primary native failure", result["reason"])
        self.assertIn("secondary accounting failure", result["step_accounting_error"])
        self.assertEqual(result["charged_steps"], 1)
        self.assertFalse(result["step_accounting_exact"])

    def test_zero_hold_budget_observes_current_stop_without_control_or_step(self):
        for stopped in (False, True):
            with self.subTest(stopped=stopped):
                backend = _backend()
                # The supplied sample deliberately says the opposite. The
                # zero-budget verdict must read current named feedback.
                stale = _hold_feedback(backend)
                stale["dq"][:] = 0. if not stopped else .1
                backend.native_data.qvel[:] = 0. if stopped else .2
                backend.proprioception = lambda: _hold_feedback(backend)
                before = backend.native_data.ctrl.copy()
                with patch.object(backend, "apply_control") as applied, patch.object(backend, "step") as step:
                    result = backend.abort_hold(stale, backend.finger_control,
                        expected_sim_time=3., stop_velocity=np.array([.01, .01, .001, .001]), max_steps=0)
                applied.assert_not_called()
                step.assert_not_called()
                np.testing.assert_array_equal(backend.native_data.ctrl, before)
                self.assertFalse(result["prepared"])
                self.assertFalse(result["native_ctrl_written"])
                self.assertEqual(result["physical_steps"], 0)
                self.assertEqual(result["charged_steps"], 0)
                self.assertTrue(result["step_accounting_exact"])
                self.assertEqual(result["physical_stop_confirmed"], stopped)
                self.assertEqual(len(result["samples"]), 1)

    def test_zero_hold_budget_feedback_failure_cannot_claim_a_stop(self):
        backend = _backend()
        initial = _hold_feedback(backend)
        backend.proprioception = lambda: (_ for _ in ()).throw(RuntimeError("no current feedback"))
        with patch.object(backend, "apply_control") as applied, patch.object(backend, "step") as step:
            result = backend.abort_hold(initial, backend.finger_control,
                expected_sim_time=3., stop_velocity=np.array([.01, .01, .001, .001]), max_steps=0)
        applied.assert_not_called()
        step.assert_not_called()
        self.assertEqual((result["physical_steps"], result["charged_steps"]), (0, 0))
        self.assertFalse(result["physical_stop_confirmed"])
        self.assertIsNone(result["joint_position_bounds_satisfied"])
        self.assertIn("no current feedback", result["reason"])
    # END no mock sample or successful serializer determines consumed physics.


# BEGIN tiny zero-step FK verifies the runtime observation boundary.
class ProprioceptionBoundaryTests(unittest.TestCase):
    def test_only_encoders_commands_and_clock_feed_calibrated_pose_estimates(self):
        backend = _backend()
        backend.native_model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody><body name="base" pos="0.5 -0.2 1" euler="0 0 90">
              <body name="upper"><joint name="shoulder" axis="0 0 1"/>
                <geom type="sphere" size="0.01" mass="0.1"/>
                <body name="lower" pos="0.3 0 0"><joint name="elbow" axis="0 0 1"/>
                  <geom type="sphere" size="0.01" mass="0.1"/><site name="tcp" pos="0.2 0 0"/>
                  <body name="left"><joint name="left_finger" type="slide" axis="0 1 0"/>
                    <geom type="sphere" size="0.01" mass="0.1"/></body>
                  <body name="right"><joint name="right_finger" type="slide" axis="0 1 0"/>
                    <geom type="sphere" size="0.01" mass="0.1"/></body>
                </body>
              </body>
            </body></worldbody></mujoco>""")
        model = backend.native_model
        backend._base_id, backend._site_id = model.body("base").id, model.site("tcp").id
        backend._joint_ids = np.array([model.joint(name).id for name in backend.joint_names])
        backend._qpos_ids = model.jnt_qposadr[backend._joint_ids].copy()
        backend._qvel_ids = model.jnt_dofadr[backend._joint_ids].copy()

        class EncoderOnlyData(SimpleNamespace):
            def __getattribute__(self, name):
                if name in {"xpos", "xmat", "site_xpos", "site_xmat", "qacc", "actuator_force"}:
                    raise AssertionError(f"Unavailable runtime observation: {name}")
                return super().__getattribute__(name)

        backend.native_data = EncoderOnlyData(qpos=np.array([0., 0., .01, -.01]),
            qvel=np.array([.2, -.3, .001, -.001]), ctrl=np.array([7., -.01, 8., .01]), time=3.)
        # These guards fail the test if setup or sampling integrates, renders,
        # or refreshes full dynamics; only separate-data kinematics is allowed.
        with patch("mujoco.mj_step", side_effect=AssertionError("No physics steps")), \
                patch("mujoco.mj_forward", side_effect=AssertionError("FK only")):
            backend._prepare_reference_fk()
            first = backend.proprioception()
            np.testing.assert_allclose(first["base_pose"][:3, 3], [.5, -.2, 1.], rtol=0., atol=1e-14)
            np.testing.assert_allclose(first["tool_pose"][:3, 3], [.5, .3, 1.], rtol=0., atol=1e-14)
            backend.native_data.qpos[backend._qpos_ids[0]] = np.pi/2
            second = backend.proprioception()
            np.testing.assert_allclose(second["tool_pose"][:3, 3], [0., -.2, 1.], rtol=0., atol=1e-14)
            np.testing.assert_array_equal(second["base_pose"], first["base_pose"])
            np.testing.assert_array_equal(second["q"], backend.native_data.qpos)
            np.testing.assert_array_equal(second["dq"], backend.native_data.qvel)
            self.assertNotIn("qacc", second)
            self.assertNotIn("actuator_force", second)
            self.assertFalse(second["tool_pose_is_independent_observation"])
            self.assertEqual(second["tool_pose_source"], "forward_kinematics_from_measured_arm_q")
            self.assertEqual(second["base_pose_source"], "known_fixed_robot_model_calibration")
            self.assertFalse(np.shares_memory(second["base_pose"], backend._calibrated_base_pose))
            self.assertEqual(second["sim_time"], 3.)
            np.testing.assert_array_equal(second["actuator_ctrl"]["values"], [8., 7., .01, -.01])
            # A joint-driven declared base cannot become a frozen calibration.
            backend._base_id = model.body("upper").id
            with self.assertRaisesRegex(ValueError, "fixed base with no joint-driven ancestors"):
                backend._prepare_reference_fk()
# END no simulator pose, acceleration or force can masquerade as a sensor.


if __name__ == "__main__":
    unittest.main()
# END these unit tests make no physical success claim.
