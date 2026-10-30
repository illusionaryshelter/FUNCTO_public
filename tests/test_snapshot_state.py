"""Native CPU checks for integral state across completed-boundary snapshots.

The shared fixture has two coupled arm hinges and one position-controlled
finger. Its geometry belongs only to this declared test robot. The tests use
the installed controller and native physics; their result concerns state
restoration, not task contact performance.
"""
from copy import copy
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent

# Establish the experiment package before the shared fixture imports its
# controller. All unchanged planning/model dependencies retain one identity.
import manipulation

from planning.simulation_snapshot import restore_native_snapshot, save_snapshot
from test_integral_backend import backend_from_sim, native_backend


def native_state(backend):
    """Read the complete declared integration vector for numerical comparison."""
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    result = np.empty(mujoco.mj_stateSize(backend.native_model, spec))
    mujoco.mj_getState(backend.native_model, backend.native_data, result, spec)
    return result


def arm_mass(backend):
    """Read the fixture's current native mass block in named arm order."""
    full = np.empty((backend.native_model.nv, backend.native_model.nv))
    mujoco.mj_fullM(backend.native_model, full, backend.native_data.qM)
    arm = backend._qvel_ids[:len(backend.arm_joint_names)]
    return full[np.ix_(arm, arm)]


def restored_backend(source, path):
    """Restore the native model before constructing its fresh controller."""
    from robosuite.utils.binding_utils import MjSim

    simulation = MjSim(copy(source.native_model))
    control = restore_native_snapshot(simulation.model._model, simulation.data._data, path)
    simulation.forward()
    result = backend_from_sim(simulation)
    result.restore_control_state(control)
    return result


class IntegralSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_complete_snapshot_resumes_the_same_future_controls_and_physics(self):
        """Nonzero integral and a physical load survive an actual file roundtrip."""
        original = native_backend(ctrl_range=(-10., 10.))
        state = original.snapshot_control_state()
        state["integral_acceleration"] = np.linalg.solve(arm_mass(original), [2., -3.]).tolist()
        original.restore_control_state(state)
        # This declared fixture load belongs only to the physical test. It is
        # preserved by native state and is never passed to the controller API.
        original.native_data.qfrc_applied[original._qvel_ids[:2]] = [.4, -.3]
        initial_q = original.proprioception()["q"][:2].copy()
        for index in range(4):
            original.apply_control(initial_q + [.001 * index, -.0006 * index],
                                   np.array([.03, -.02]), np.array([.1, -.07]),
                                   np.array([-.02]))
            original.step()
        path = save_snapshot(original, Path(self.folder.name) / "completed.npz")
        committed = original.snapshot_control_state()
        with np.load(path, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files),
                             {"integration_state", "state_spec", "control_state_json"})
            self.assertEqual(archive["control_state_json"].dtype.kind, "U")
            np.testing.assert_array_equal(archive["integration_state"], native_state(original))
        resumed = restored_backend(original, path)
        self.assertEqual(resumed.snapshot_control_state(), committed)
        np.testing.assert_array_equal(native_state(resumed), native_state(original))

        # The same absolute reference sequence follows both the uninterrupted
        # backend and the reconstructed backend. No output matching occurs.
        for index in range(12):
            arguments = (initial_q + [.004 + .0003 * index, -.003 - .0002 * index],
                         np.array([.025, -.015]), np.array([-.06, .04]), np.array([-.02]))
            expected, actual = original.apply_control(*arguments), resumed.apply_control(*arguments)
            for key in ("requested_arm_torque", "actuator_ctrl"):
                np.testing.assert_allclose(actual[key], expected[key], rtol=0., atol=1e-12)
            original.step()
            resumed.step()
            np.testing.assert_allclose(native_state(resumed), native_state(original), rtol=0., atol=1e-12)
            np.testing.assert_allclose(resumed.snapshot_control_state()["integral_acceleration"],
                                       original.snapshot_control_state()["integral_acceleration"],
                                       rtol=0., atol=1e-12)

    def test_equal_saturated_ctrl_preserves_distinct_integral_states(self):
        """Identical native snapshots can require different future commands."""
        originals = [native_backend(ctrl_range=(-10., 10.)) for _ in range(2)]
        paths = []
        for index, (backend, integral_torque) in enumerate(zip(originals, (20., 30.))):
            control = backend.snapshot_control_state()
            control["integral_acceleration"] = np.linalg.solve(
                arm_mass(backend), np.full(2, integral_torque)).tolist()
            backend.restore_control_state(control)
            feedback = backend.proprioception()
            command = backend.apply_control(feedback["q"][:2], feedback["dq"][:2],
                                            np.zeros(2), np.array([-.02]))
            self.assertTrue(np.all(command["arm_torque_saturated"]))
            backend.step()
            paths.append(save_snapshot(backend, Path(self.folder.name) / f"saturated-{index}.npz"))

        np.testing.assert_array_equal(native_state(originals[0]), native_state(originals[1]))
        first, second = (backend.snapshot_control_state() for backend in originals)
        self.assertNotEqual(first["integral_acceleration"], second["integral_acceleration"])
        resumed = [restored_backend(backend, path) for backend, path in zip(originals, paths)]

        # Cancel the common bias and half the two stored integral forces with
        # one shared reference acceleration. Both proportional/velocity errors
        # are zero at this instant. The remaining torques have opposite signs,
        # although both saved native ctrl arrays were the same saturated value.
        common = originals[0]
        feedback = common.proprioception()
        mass = arm_mass(common)
        arm_dofs = common._qvel_ids[:2]
        bias = common.native_data.qfrc_bias[arm_dofs]
        midpoint = .5 * (np.asarray(first["integral_acceleration"])
                          + np.asarray(second["integral_acceleration"]))
        acceleration = -midpoint - np.linalg.solve(mass, bias)
        arguments = (feedback["q"][:2], feedback["dq"][:2], acceleration, np.array([-.02]))
        outputs = []
        for expected_backend, actual_backend in zip(originals, resumed):
            expected = expected_backend.apply_control(*arguments)
            actual = actual_backend.apply_control(*arguments)
            np.testing.assert_allclose(actual["requested_arm_torque"], expected["requested_arm_torque"],
                                       rtol=0., atol=1e-12)
            np.testing.assert_allclose(actual["actuator_ctrl"], expected["actuator_ctrl"],
                                       rtol=0., atol=1e-12)
            outputs.append(actual["actuator_ctrl"][:2])
            expected_backend.step()
            actual_backend.step()
            np.testing.assert_allclose(native_state(actual_backend), native_state(expected_backend),
                                       rtol=0., atol=1e-12)
        self.assertGreater(float(np.max(np.abs(outputs[0] - outputs[1]))), 1.)


if __name__ == "__main__":
    unittest.main()
