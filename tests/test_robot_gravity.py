"""Finite native-CPU checks for robot-model gravity inside position actuation.

The fixture is a two-coordinate robot with no task geometry. Its potential
energy gives an independent analytic gravity oracle. Only the hold-path test
advances one native step; the other checks evaluate forces without motion.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from manipulation.robot_gravity import RobotGravity
from manipulation.robosuite_execution import RobosuiteExecutionBackend


def robot_xml(base_angle=0.):
    """Declare a rotated base, a Y hinge and a local-X sliding finger.

    The hinge body has mass .4 kg and COM .1 m along X; the finger has
    mass .11 kg and its origin is .2 m along the same axis. Actuator order
    deliberately differs from joint order to exercise named bindings.
    """
    quaternion = f"{np.cos(base_angle / 2)} 0 {np.sin(base_angle / 2)} 0"
    return f'''<mujoco model="gravity_fixture">
      <compiler angle="radian"/>
      <option timestep="0.002" gravity="0 0 -9.81"/>
      <worldbody><body name="base" quat="{quaternion}">
        <body name="arm_body">
          <joint name="arm" type="hinge" axis="0 1 0" range="-3 3"/>
          <inertial mass="0.4" pos="0.1 0 0" diaginertia="0.01 0.01 0.01"/>
          <site name="tcp" pos="0.2 0 0"/>
          <body name="finger_body" pos="0.2 0 0">
            <joint name="finger" type="slide" axis="1 0 0" range="-0.04 0"
                   damping="100" armature="1" frictionloss="1"/>
            <inertial mass="0.11" pos="0 0 0" diaginertia="0.001 0.001 0.001"/>
          </body>
        </body>
      </body></worldbody>
      <actuator>
        <position name="finger_servo" joint="finger" kp="1000"
                  ctrlrange="-0.04 0" forcerange="-20 20"/>
        <motor name="arm_motor" joint="arm" ctrlrange="-100 100"/>
      </actuator>
    </mujoco>'''


def analytic_gravity(arm, finger, base_angle=0.):
    """Gradient of V=-9.81*sin(arm+base)*(.4*.1+.11*(.2+finger))."""
    angle = arm + base_angle
    return np.array([-9.81 * np.cos(angle) * (.4 * .1 + .11 * (.2 + finger)),
                     -.11 * 9.81 * np.sin(angle)])


def native_backend(xml, *, arm=.7, finger=-.02):
    """Use the installed native controllers on the explicit test robot."""
    from robosuite.utils.binding_utils import MjSim

    model = mujoco.MjModel.from_xml_string(xml)
    simulation = MjSim(model)
    data = simulation.data._data
    data.qpos[:] = [arm, finger]
    data.ctrl[model.actuator("finger_servo").id] = finger
    simulation.forward()
    return RobosuiteExecutionBackend(simulation, base_link="base", tcp_site="tcp",
        arm_joint_names=["arm"], finger_joint_names=["finger"], arm_kp=100., arm_kd=20.,
        velocity_limits={"arm": 2., "finger": .2}, robot_gravity=RobotGravity(mujoco.MjModel.from_xml_string(xml), ["arm", "finger"],
            source="Fixed synthetic robot model"), source="Declared two-coordinate robot fixture")


class RobotGravityTests(unittest.TestCase):
    def test_named_coordinates_and_rotated_base_match_potential_gradient(self):
        """Nonzero hinge angles exercise gravity sign and coordinate rotation."""
        for base_angle in (0., .43):
            with self.subTest(base_angle=base_angle), tempfile.TemporaryDirectory() as folder:
                model_path = Path(folder) / "robot.xml"
                model_path.write_text(robot_xml(base_angle))
                # The input/output sequence is intentionally finger then arm.
                gravity = RobotGravity(mujoco.MjModel.from_xml_path(str(model_path)), ("finger", "arm"), source=str(model_path))
                for arm, finger in ((.7, -.02), (-.7, -.03), (0., -.01)):
                    expected = analytic_gravity(arm, finger, base_angle)[[1, 0]]
                    np.testing.assert_allclose(gravity.compute(np.array([finger, arm])),
                                               expected, rtol=0., atol=2e-14)


class FingerGravityControlTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.xml = robot_xml()
        self.model_path = Path(self.folder.name) / "robot.xml"
        self.model_path.write_text(self.xml)
        self.backend = native_backend(self.xml)

    def test_initialization_preserves_state_and_refreshes_compensated_force(self):
        """Installation changes the force law, retaining measured and command state."""
        backend = self.backend
        initial = {name: np.array(getattr(backend.native_data, name), copy=True)
                   for name in ("qpos", "qvel", "ctrl")}
        moment = backend.sim_time
        finger_id = backend.native_model.actuator("finger_servo").id
        for name, expected in initial.items():
            np.testing.assert_array_equal(getattr(backend.native_data, name), expected)
        self.assertEqual(backend.sim_time, moment)
        expected = analytic_gravity(.7, -.02)[1]
        self.assertAlmostEqual(backend.native_data.actuator_force[finger_id], expected, places=13)

    def test_native_total_force_limit_and_original_position_commands(self):
        """P plus gravity is clipped once by the original native force limit."""
        backend = self.backend
        finger_id = backend.native_model.actuator("finger_servo").id
        for actual, command in ((-.02, -.02), (-.039, 0.), (-.001, -.04)):
            with self.subTest(actual=actual, command=command):
                backend.native_data.qpos[:] = [.7, actual]
                backend.native_data.qvel[:] = 0.
                mujoco.mj_forward(backend.native_model, backend.native_data)
                backend.apply_control(np.array([.7]), np.zeros(1), np.zeros(1), np.array([command]))
                mujoco.mj_forward(backend.native_model, backend.native_data)
                expected = np.clip(1000. * (command - actual) + analytic_gravity(.7, actual)[1], -20., 20.)
                self.assertAlmostEqual(backend.native_data.actuator_force[finger_id], expected, places=12)
                self.assertEqual(backend.native_data.ctrl[finger_id], command)
                np.testing.assert_array_equal(backend.native_model.actuator_forcerange[finger_id], [-20., 20.])

    def test_live_dynamic_arrays_do_not_supply_model_gravity(self):
        """An unrelated live bias value cannot become the feedforward input."""
        backend = self.backend
        backend.native_data.qpos[:] = [-.6, -.025]
        backend.native_data.qvel[:] = [.4, -.015]
        mujoco.mj_forward(backend.native_model, backend.native_data)
        # This explicit sentinel represents a forbidden source of compensation.
        # The installed adapter must recompute zero-velocity robot-model gravity
        # from named q; its base controller refreshes the real live bias later.
        backend.native_data.qfrc_bias[:] = 1e6
        backend.apply_control(np.array([-.6]), np.zeros(1), np.zeros(1), np.array([-.025]))
        finger_id = backend.native_model.actuator("finger_servo").id
        self.assertAlmostEqual(backend.native_model.actuator_biasprm[finger_id, 0],
                               analytic_gravity(-.6, -.025)[1], places=13)

    def test_normal_control_and_native_hold_recompute_the_same_gravity_path(self):
        """The actual abort implementation dispatches through the installed method."""
        backend = self.backend
        calls = []
        original = RobotGravity.compute

        def observed_compute(instance, q):
            calls.append(np.asarray(q).copy())
            return original(instance, q)

        with patch.object(RobotGravity, "compute", observed_compute):
            initialized = len(calls)
            backend.apply_control(np.array([.7]), np.zeros(1), np.zeros(1), np.array([-.02]))
            self.assertEqual(len(calls), initialized + 1)
            # A separate explicit feedback state verifies hold recomputation;
            # no normal trajectory or feedback is projected or hidden.
            backend.native_data.qpos[:] = [.3, -.02]
            backend.native_data.qvel[:] = 0.
            mujoco.mj_forward(backend.native_model, backend.native_data)
            feedback = backend.proprioception()
            before_hold = len(calls)
            result = backend.abort_hold(feedback, np.array([-.02]),
                expected_sim_time=backend.sim_time, stop_velocity=np.array([1., 1.]), max_steps=1)
            self.assertEqual(result["physical_steps"], 1, result)
            self.assertEqual(len(calls), before_hold + 1)
            np.testing.assert_array_equal(calls[-1], [.3, -.02])
            finger_id = backend.native_model.actuator("finger_servo").id
            self.assertAlmostEqual(backend.native_model.actuator_biasprm[finger_id, 0],
                                   analytic_gravity(.3, -.02)[1], places=13)
            self.assertEqual(backend.native_data.ctrl[finger_id], -.02)


if __name__ == "__main__":
    unittest.main()
