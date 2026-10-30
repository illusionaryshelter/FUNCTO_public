"""Finite native-CPU checks of the integral backend.

The fixture contains two coupled hinge coordinates and one position-controlled
finger, with no scene/task geometry. It exercises the installed robosuite
controller and native MuJoCo step; declared actuator limits provide independent
force checks. No renderer or accelerator is used.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from manipulation.robosuite_execution import RobosuiteExecutionBackend
from manipulation.robot_gravity import RobotGravity



def robot_xml(*, ctrl_range=(-80., 80.), force_range=None, joint_force_range=None):
    """A fixed-base two-link robot with known inertia and zero world gravity.

    The two hinge axes are parallel, so the actual arm inertia is coupled.
    The actuator order differs from the named arm/finger coordinate order.
    Optional force bounds act at native actuator or joint input respectively.
    """
    ctrl = f'{ctrl_range[0]} {ctrl_range[1]}'
    force = ('' if force_range is None else
             f'forcelimited="true" forcerange="{force_range[0]} {force_range[1]}"')
    joint_force = ('' if joint_force_range is None else
                   f'actuatorfrclimited="true" actuatorfrcrange="{joint_force_range[0]} {joint_force_range[1]}"')
    return f'''<mujoco model="integral_robot_fixture">
      <compiler angle="radian"/><option timestep=".002" gravity="0 0 0"/>
      <worldbody><body name="base"><body name="upper">
        <joint name="shoulder" axis="0 1 0" range="-2 2" armature="1" {joint_force}/>
        <inertial mass="1.2" pos=".15 0 0" diaginertia=".02 .02 .02"/>
        <body name="lower" pos=".3 0 0">
          <joint name="elbow" axis="0 1 0" range="-2 2" armature="1" {joint_force}/>
          <inertial mass=".8" pos=".1 0 0" diaginertia=".01 .01 .01"/>
          <site name="tcp" pos=".2 0 0"/>
          <body name="finger_body" pos=".2 0 0">
            <joint name="finger" type="slide" axis="1 0 0" range="-.04 0" armature="1"/>
            <inertial mass=".1" pos="0 0 0" diaginertia=".001 .001 .001"/>
          </body>
        </body>
      </body></body></worldbody>
      <actuator>
        <position name="finger_servo" joint="finger" kp="1000" ctrlrange="-.04 0" forcerange="-20 20"/>
        <motor name="elbow_motor" joint="elbow" ctrlrange="{ctrl}" {force}/>
        <motor name="shoulder_motor" joint="shoulder" ctrlrange="{ctrl}" {force}/>
      </actuator>
    </mujoco>'''


def backend_from_sim(simulation):
    """Bind this fixture after native restoration, preserving its q/dq/ctrl."""
    return RobosuiteExecutionBackend(simulation, base_link="base", tcp_site="tcp",
        arm_joint_names=["shoulder", "elbow"], finger_joint_names=["finger"],
        arm_kp=1000., arm_kd=200.,
        velocity_limits={"shoulder": 5., "elbow": 5., "finger": .2},
        robot_gravity=RobotGravity(simulation.model._model, ["shoulder", "elbow", "finger"],
            source="Synthetic robot-only fixture with independent scratch state"),
        source="Explicit two-hinge integral controller fixture")


def native_backend(*, ctrl_range=(-80., 80.), force_range=None, joint_force_range=None):
    """Construct one small native plant without task objects or contacts."""
    from robosuite.utils.binding_utils import MjSim

    model = mujoco.MjModel.from_xml_string(robot_xml(ctrl_range=ctrl_range,
        force_range=force_range, joint_force_range=joint_force_range))
    simulation = MjSim(model)
    data = simulation.data._data
    data.qpos[:] = [.2, -.1, -.02]
    data.ctrl[model.actuator("finger_servo").id] = -.02
    simulation.forward()
    return backend_from_sim(simulation)


def current_reference(backend):
    """Return an explicit fixed goal at current measured arm coordinates."""
    return backend.native_data.qpos[backend._qpos_ids[:2]].copy()


def restore_integral(backend, values):
    """Change a test's declared controller state through the actual API."""
    state = backend.snapshot_control_state()
    state["integral_acceleration"] = np.asarray(values).tolist()
    backend.restore_control_state(state)


class _CountedCtrl(np.ndarray):
    """Observe actual ctrl writes while retaining the native shared buffer."""
    def __new__(cls, native):
        result = native.view(cls)
        result.writes = []
        return result

    def __array_finalize__(self, parent):
        self.writes = getattr(parent, "writes", [])

    def __setitem__(self, key, value):
        self.writes.append(np.asarray(key).copy())
        return super().__setitem__(key, value)


class _ObservedData:
    """Expose the same native data, instrumenting its control array only."""
    def __init__(self, native):
        self.native = native
        self.ctrl = _CountedCtrl(native.ctrl)

    def __getattr__(self, name):
        return getattr(self.native, name)


class IntegralBackendTests(unittest.TestCase):
    def test_total_components_and_one_arm_write_share_the_effective_domain(self):
        backend = native_backend(ctrl_range=(-80., 80.), force_range=(-20., 20.),
                                 joint_force_range=(-12., 15.))
        restore_integral(backend, [2., -3.])
        native = backend.native_data
        backend.native_data = _ObservedData(native)
        result = backend.apply_control(current_reference(backend)+[1., -1.],
            np.array([.1, -.2]), np.array([.3, -.4]), np.array([-.02]))
        expected_bounds = [[-12., 15.], [-12., 15.]]
        self.assertEqual(backend.describe()["controller_policy"]["torque_limits"], expected_bounds)
        terms = [result[key] for key in ("arm_position_feedback_torque",
            "arm_velocity_feedback_torque", "arm_velocity_feedforward_torque",
            "arm_acceleration_feedforward_torque", "arm_bias_torque", "arm_integral_torque")]
        np.testing.assert_allclose(sum(terms), result["requested_arm_torque"], rtol=0., atol=5e-13)
        np.testing.assert_array_equal(result["actuator_ctrl"][:2], [15., -12.])
        self.assertEqual(result["arm_control_saturated_coordinates"], 2)
        writes = backend.native_data.ctrl.writes
        self.assertEqual(sum(np.array_equal(row, backend._arm_actuators) for row in writes), 1)
        self.assertEqual(len(writes), 2)  # one arm vector and one finger vector
        mujoco.mj_forward(backend.native_model, native)
        np.testing.assert_array_equal(native.actuator_force[backend._arm_actuators], [15., -12.])
        np.testing.assert_array_equal(native.qfrc_actuator[backend._qvel_ids[:2]], [15., -12.])

    def test_same_time_writes_retain_z_and_only_last_held_error_is_integrated(self):
        backend = native_backend()
        q = current_reference(backend)
        for _ in range(5):
            output = backend.apply_control(q+[.002, -.001], np.zeros(2), np.zeros(2), [-.02])
            np.testing.assert_array_equal(output["arm_integral_state"], [0., 0.])
        output = backend.apply_control(q+[-.001, .002], np.zeros(2), np.zeros(2), [-.02])
        with self.assertRaises(ValueError):
            backend.snapshot_control_state()
        report = backend.step()
        state = backend.snapshot_control_state()
        expected = backend.timestep*backend._arm_ki*np.array([-.001, .002])
        np.testing.assert_allclose(state["integral_acceleration"], expected, rtol=0., atol=2e-16)
        self.assertTrue(report["integral_update_committed"])
        self.assertEqual(state["sim_time"], backend.sim_time)
        self.assertEqual(report["physical_steps"], 1)
        # A native advance with no new sampled control has no new held error.
        report = backend.step()
        self.assertFalse(report["integral_update_committed"])
        np.testing.assert_array_equal(backend.snapshot_control_state()["integral_acceleration"],
                                      state["integral_acceleration"])

    def test_conditional_integral_preserves_coupling_at_saturation(self):
        backend = native_backend(ctrl_range=(-2., 2.))
        q = current_reference(backend)
        controller = backend._arm_controller
        controller.set_goal(q)
        controller.run_controller()
        mass = controller.mass_matrix.copy()
        self.assertGreater(abs(mass[0, 1]), .001)
        # Both raw torque increments are positive. Coordinate zero is already
        # above its upper bound; coordinate one can still accept its increment.
        error = np.linalg.solve(mass, np.array([.001, .001]))
        desired_request = np.array([2.5, 0.])
        z = np.linalg.solve(mass, desired_request)-controller.kp*error
        restore_integral(backend, z)
        output = backend.apply_control(q+error, np.zeros(2), np.zeros(2), [-.02])
        np.testing.assert_allclose(output["requested_arm_torque"], desired_request, atol=1e-13, rtol=0.)
        report = backend.step()
        accepted = np.asarray(report["integral_accepted_torque_increment"])
        self.assertEqual(accepted[0], 0.)
        self.assertGreater(accepted[1], 0.)
        delta = np.asarray(backend.snapshot_control_state()["integral_acceleration"])-z
        np.testing.assert_allclose(mass@delta, accepted, rtol=0., atol=3e-16)
        self.assertNotEqual(delta[0], 0.)  # coupling differs from freezing z[0]

    def test_explicit_rebase_matches_once_and_subsequent_reference_keeps_z(self):
        backend = native_backend()
        q = current_reference(backend)
        backend.native_data.ctrl[backend._arm_actuators] = [3., -2.]
        backend.request_control_rebase()
        with self.assertRaises(ValueError):
            backend.snapshot_control_state()
        first = backend.apply_control(q+[.01, -.02], [.03, -.01], [.2, -.3], [-.02])
        np.testing.assert_allclose(first["actuator_ctrl"][:2], [3., -2.], rtol=0., atol=1e-13)
        self.assertTrue(first["arm_integral_rebased"])
        second = backend.apply_control(q+[.02, -.01], [.03, -.01], [.2, -.3], [-.02])
        np.testing.assert_array_equal(second["arm_integral_state"], first["arm_integral_state"])
        self.assertFalse(second["arm_integral_rebased"])
        self.assertGreater(np.max(abs(second["actuator_ctrl"][:2]-first["actuator_ctrl"][:2])), 1.)

    def test_stop_keeps_nonzero_integral_and_damping_even_with_pending_rebase(self):
        backend = native_backend()
        backend.native_data.qvel[backend._qvel_ids[:2]] = [.1, -.08]
        mujoco.mj_forward(backend.native_model, backend.native_data)
        z = np.array([5., -3.])
        restore_integral(backend, z)
        feedback = backend.proprioception()
        q = current_reference(backend)
        original = backend.apply_control(q+[.01, -.005], np.zeros(2), np.zeros(2), [-.02])
        self.assertGreater(np.max(abs(original["arm_position_feedback_torque"])), 1.)
        # A stopped entrance may occur before the caller's first normal write.
        backend.request_control_rebase()
        result = backend.abort_hold(feedback, [-.02], expected_sim_time=backend.sim_time,
            stop_velocity=np.full(3, 10.), max_steps=1)
        self.assertEqual(result["physical_steps"], 1, result)
        stop = result["controls"][0]
        np.testing.assert_array_equal(stop["arm_integral_state"], z)
        self.assertFalse(stop["arm_integral_rebased"])
        expected = (np.asarray(stop["arm_velocity_feedback_torque"])
                    + np.asarray(stop["arm_bias_torque"])+np.asarray(stop["arm_integral_torque"]))
        np.testing.assert_allclose(stop["requested_arm_torque"], expected, rtol=0., atol=1e-13)
        self.assertGreater(np.max(abs(expected-np.asarray(original["requested_arm_torque"]))), 1.)
        np.testing.assert_array_equal(stop["reference_arm_dq"], [0., 0.])
        np.testing.assert_array_equal(stop["reference_arm_ddq"], [0., 0.])

    def test_completed_native_step_commits_before_a_later_refresh_error(self):
        backend = native_backend()
        backend.apply_control(current_reference(backend)+[.001, -.002], [0., 0.], [0., 0.], [-.02])
        expected = backend.timestep*backend._arm_ki*np.array([.001, -.002])
        with patch("manipulation.robosuite_execution.mujoco.mj_forward", side_effect=RuntimeError("refresh failed")):
            with self.assertRaisesRegex(RuntimeError, "refresh failed"):
                backend.step()
        self.assertTrue(backend.last_step_accounting["integral_update_committed"])
        np.testing.assert_allclose(backend.snapshot_control_state()["integral_acceleration"], expected,
                                   rtol=0., atol=3e-16)

    def test_exception_after_native_advance_commits_but_stationary_error_does_not(self):
        for advance in (False, True):
            with self.subTest(advance=advance):
                backend = native_backend()
                error = np.array([.001, -.002])
                backend.apply_control(current_reference(backend)+error, [0., 0.], [0., 0.], [-.02])
                native_step = backend._sim.step

                def failed_step():
                    if advance:
                        native_step()
                    raise RuntimeError("native failure")

                with patch.object(backend._sim, "step", failed_step):
                    with self.assertRaisesRegex(RuntimeError, "native failure"):
                        backend.step()
                self.assertEqual(backend.last_step_accounting["integral_update_committed"], advance)
                expected = (backend.timestep*backend._arm_ki*error if advance else np.zeros(2))
                np.testing.assert_allclose(backend._integral_acceleration, expected, rtol=0., atol=3e-16)
                self.assertEqual(backend.last_step_accounting["physical_steps"], int(advance))

    def test_restored_state_retains_named_policy_time_and_next_output(self):
        left, right = native_backend(), native_backend()
        restore_integral(left, [2., -4.])
        state = left.snapshot_control_state()
        right.restore_control_state(state)
        args = (current_reference(left)+[.001, -.002], [.02, -.03], [.1, -.2], [-.02])
        a, b = left.apply_control(*args), right.apply_control(*args)
        np.testing.assert_array_equal(a["actuator_ctrl"], b["actuator_ctrl"])
        np.testing.assert_array_equal(a["arm_integral_state"], b["arm_integral_state"])
        for changed in ("arm_joint_names", "sim_time", "controller_policy"):
            with self.subTest(changed=changed):
                target = native_backend()
                invalid = deepcopy(state)
                if changed == "arm_joint_names":
                    invalid[changed] = invalid[changed][::-1]
                elif changed == "sim_time":
                    invalid[changed] += .002
                else:
                    invalid[changed]["torque_limits"] = [[-1., 1.], [-1., 1.]]
                with self.assertRaises(ValueError):
                    target.restore_control_state(invalid)

    def test_native_exception_stays_primary_if_integral_commit_also_fails(self):
        """Boundary accounting retains both errors after known physical work."""
        backend = native_backend()
        backend.apply_control(current_reference(backend)+[.001, -.002], [0., 0.], [0., 0.], [-.02])
        native_step = backend._sim.step

        def failed_after_step():
            native_step()
            raise RuntimeError("original native failure")

        with patch.object(backend._sim, "step", failed_after_step), patch(
                "manipulation.robosuite_execution.np.linalg.solve",
                side_effect=np.linalg.LinAlgError("declared commit failure")):
            with self.assertRaisesRegex(RuntimeError, "original native failure"):
                backend.step()
        report = backend.last_step_accounting
        self.assertEqual(report["physical_steps"], 1)
        self.assertFalse(report["integral_update_committed"])
        self.assertIn("declared commit failure", report["integral_update_error"])
        with self.assertRaises(ValueError):
            backend.snapshot_control_state()


if __name__ == "__main__":
    unittest.main()
