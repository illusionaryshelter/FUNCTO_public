"""CPU checks of named targets, controller passthrough and execution clocks.

The mocked integration clock deliberately keeps measured state independent
from commanded positions. These tests verify the executor's data/control
contract; they do not simulate motion, contacts or physical grasp evidence.
"""

# BEGIN independent fixed-finger execution contract regressions.
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/"planning"))

from manipulation.planning import JointTrajectory
from recorded_simulation import FixedFingerTargets, RecordedSimulation


def _event(event_id, kind, time, evidence=None):
    return {"skill_id": "declared_skill", "event_id": event_id, "kind": kind,
            "time": time, "requires_evidence": evidence}


def _fixture():
    """A mocked clock/state source with the installed, real GRIP controller."""
    from robosuite.controllers.parts.gripper.simple_grip import SimpleGripController

    simulation = RecordedSimulation()
    simulation.arm_joint_names = tuple(f"arm_{index}" for index in range(7))
    simulation.finger_joint_names = ("independent_finger_left", "independent_finger_right")
    simulation.joint_names = simulation.arm_joint_names + simulation.finger_joint_names
    simulation._joint_ids = simulation._qpos_ids = simulation._qvel_ids = np.arange(9)
    simulation._actuator_ids = np.arange(9)
    simulation._arm_actuators, simulation._finger_actuators = np.arange(7), np.arange(7, 9)
    simulation._actuator_names = tuple(f"native_actuator_{index}" for index in range(9))
    simulation._base_id = simulation._site_id = 0
    simulation._dt = .002
    simulation._trial_id = str(uuid4())
    simulation._feedback_source = "explicit mocked robot feedback; no physical integration"
    simulation.restore_report = {"scope": "mocked integration clock only"}
    simulation.native_data = SimpleNamespace(
        qpos=np.r_[np.linspace(.1, .7, 7), .012, -.017],
        qvel=np.r_[np.linspace(.01, .07, 7), .03, -.04],
        ctrl=np.r_[np.zeros(7), .026, -.023], time=1.3,
        xmat=np.eye(3).reshape(1, 9), xpos=np.zeros((1, 3)),
        site_xmat=np.eye(3).reshape(1, 9), site_xpos=np.zeros((1, 3)))
    simulation.native_model = SimpleNamespace(
        actuator_ctrlrange=np.vstack((np.tile([-10., 10.], (7, 1)), [0., .04], [-.04, 0.])))

    def step():
        # Advancing a mock clock leaves actual state independent from goals.
        # This is intentionally not a robot dynamics model.
        simulation.native_data.time += simulation._dt

    simulation._sim = SimpleNamespace(data=simulation.native_data, forward=lambda: None, step=step)
    indexes = {"joints": [7, 8], "qpos": [7, 8], "qvel": [7, 8], "actuators": [7, 8]}
    simulation._grip_controller = SimpleGripController(simulation._sim, indexes,
        actuator_range=(np.array([0., -.04]), np.array([.04, 0.])),
        policy_freq=1/simulation._dt, use_action_scaling=False)
    simulation._grip_controller.set_goal(simulation.native_data.ctrl[7:].copy())
    simulation.arm_goals = []
    simulation._arm_controller = SimpleNamespace(
        set_goal=lambda goal: simulation.arm_goals.append(goal.copy()),
        run_controller=lambda: np.zeros(7))
    q, dq = simulation.native_data.qpos[:7], simulation.native_data.qvel[:7]
    events = (_event("open-original", "open", .001),
              _event("close-original", "close", .003),
              _event("holding-original", "verification", .0075, "Holding"),
              _event("hold-later", "hold", .009))
    curve = JointTrajectory(simulation.arm_joint_names, np.array([0., .012]),
        np.stack((q, q+dq*.012)), np.stack((dq, dq)),
        np.array([0., 1.]), np.array([0., 1.]), events, (), {})
    targets = {"joint_names": list(simulation.finger_joint_names),
        "source": "explicit independent target selection for controller contract test",
        "events": [
            {"skill_id": "declared_skill", "event_id": "open-original", "kind": "open", "positions": [.032, -.025]},
            {"skill_id": "declared_skill", "event_id": "close-original", "kind": "close", "positions": [.004, -.009]},
            {"skill_id": "declared_skill", "event_id": "hold-later", "kind": "hold", "positions": [.004, -.009]}]}
    return simulation, curve, targets


def _initial_feedback(simulation):
    """Explicit calibration metadata accompanies the measured robot state."""
    return {**simulation.proprioception(), "reference_frame": "declared_test_frame",
            "tool_link": "declared_test_tcp", "tool_transform": np.eye(4).tolist()}


class FixedFingerTargetTests(unittest.TestCase):
    def test_targets_bind_exact_event_identities_and_retimed_times(self):
        simulation, curve, value = _fixture()
        targets = FixedFingerTargets.from_mapping(value)
        value["events"][0]["positions"][0] = .039
        bound = targets.bind(curve, finger_joint_names=simulation.finger_joint_names,
                             ctrlrange=simulation.native_model.actuator_ctrlrange[7:])
        self.assertEqual([row["time"] for row in bound], [.001, .003, .009])
        self.assertEqual(bound[0]["finger_positions"], [.032, -.025])
        curve.events[1]["time"] = .005
        rebound = targets.bind(curve, finger_joint_names=simulation.finger_joint_names,
                               ctrlrange=simulation.native_model.actuator_ctrlrange[7:])
        self.assertEqual(rebound[1]["time"], .005)

    def test_missing_extra_relabelled_reordered_and_out_of_range_targets_fail(self):
        simulation, curve, value = _fixture()
        variants = []
        missing = deepcopy(value)
        missing["events"].pop()
        variants.append(missing)
        extra = deepcopy(value)
        extra["events"].append({**extra["events"][0], "event_id": "unrelated"})
        variants.append(extra)
        relabelled = deepcopy(value)
        relabelled["events"][1]["kind"] = "open"
        variants.append(relabelled)
        reordered = deepcopy(value)
        reordered["joint_names"].reverse()
        variants.append(reordered)
        out_of_range = deepcopy(value)
        out_of_range["events"][0]["positions"][1] = .001
        variants.append(out_of_range)
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                FixedFingerTargets.from_mapping(variant).bind(curve,
                    finger_joint_names=simulation.finger_joint_names,
                    ctrlrange=simulation.native_model.actuator_ctrlrange[7:])

    def test_target_input_cannot_override_clock_or_omit_numeric_coordinates(self):
        _, _, value = _fixture()
        variants = []
        timed = deepcopy(value)
        timed["events"][0]["time"] = 0.
        variants.append(timed)
        duplicate = deepcopy(value)
        duplicate["events"].append(deepcopy(duplicate["events"][0]))
        variants.append(duplicate)
        nonfinite = deepcopy(value)
        nonfinite["events"][1]["positions"][0] = float("nan")
        variants.append(nonfinite)
        incomplete = deepcopy(value)
        incomplete["events"][0]["positions"] = [.03]
        variants.append(incomplete)
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                FixedFingerTargets.from_mapping(variant)


class RecordedControlTests(unittest.TestCase):
    def test_installed_unscaled_grip_preserves_asymmetric_raw_native_commands(self):
        simulation, _, _ = _fixture()
        controller = simulation._grip_controller
        np.testing.assert_array_equal(controller.run_controller(), [.026, -.023])
        self.assertFalse(np.array_equal(controller.run_controller(), simulation.native_data.qpos[7:]))
        controller.set_goal(np.array([.004, -.009]))
        np.testing.assert_array_equal(controller.run_controller(), [.004, -.009])
        self.assertEqual(controller.control_dim, 2)

    def test_full_actual_entrance_checks_both_fingers_and_their_velocities(self):
        for field in ("q", "dq"):
            for column in (7, 8):
                simulation, curve, value = _fixture()
                feedback = _initial_feedback(simulation)
                feedback[field][column] += 1e-5
                with self.subTest(field=field, column=column), TemporaryDirectory() as temporary:
                    output = Path(temporary)/"trial"
                    with self.assertRaisesRegex(ValueError, "full robot state"):
                        simulation.execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
                            initial_feedback=feedback, evaluation_mode="accepted_plan", output=output,
                            start_position_tolerance=1., start_velocity_tolerance=1.)
                    self.assertFalse(output.exists())

    def test_native_step_clock_records_all_commands_and_actual_state_at_barrier(self):
        simulation, curve, value = _fixture()
        initial = _initial_feedback(simulation)
        original_events = deepcopy(curve.events)
        with TemporaryDirectory() as temporary, patch("recorded_simulation.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            report = simulation.execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
                initial_feedback=initial, evaluation_mode="accepted_plan", output=output, max_physics_steps=4)
            self.assertEqual(report["executed_steps"], 4)
            self.assertEqual(report["status"], "stopped_for_evidence")
            self.assertFalse(report["evidence_passed"])
            self.assertFalse(report["barrier_crossed"])
            self.assertEqual([(event["event_id"], event["applied_at_s"]) for event in report["events"]],
                             [("open-original", .002), ("close-original", .004)])
            self.assertEqual([row["event_id"] for row in report["unapplied_gripper_events"]], ["hold-later"])
            self.assertEqual(curve.events, original_events)
            np.testing.assert_allclose(simulation.arm_goals[-1], curve.sample(.0075)[0], atol=0., rtol=0.)
            with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["command_q"].shape, (5, 9))
                self.assertEqual(saved["actual_q"].shape, (5, 9))
                self.assertEqual(saved["actual_dq"].shape, (5, 9))
                self.assertEqual(saved["arm_curve_dq"].shape, (5, 7))
                self.assertEqual(saved["actuator_ctrl"].shape, (4, 9))
                self.assertNotIn("desired_dq", saved.files)
                np.testing.assert_array_equal(saved["actuator_ctrl"][:, 7:],
                    [[.026, -.023], [.032, -.025], [.004, -.009], [.004, -.009]])
                np.testing.assert_array_equal(saved["command_q"][0, 7:], [.026, -.023])
                np.testing.assert_array_equal(saved["actual_dq"][0, 7:], [.03, -.04])
                np.testing.assert_allclose(saved["control_elapsed"], [0., .002, .004, .006], atol=1e-15)
                self.assertEqual(saved["curve_time"][-1], .0075)
                self.assertAlmostEqual(saved["elapsed"][-1], .008)
            actual = json.loads(Path(report["barrier_feedback_path"]).read_text())
            self.assertEqual(actual["joint_names"], list(simulation.joint_names))
            np.testing.assert_array_equal(actual["q"], initial["q"])
            np.testing.assert_array_equal(actual["dq"], initial["dq"])
            self.assertEqual(actual["actuator_ctrl"]["values"][7:], [.004, -.009])
            self.assertEqual(actual["timestamp_s"], actual["sim_time"])
            self.assertEqual(actual["reference_frame"], initial["reference_frame"])
            self.assertEqual(actual["tool_link"], initial["tool_link"])
            self.assertEqual(actual["tool_transform"], initial["tool_transform"])
            self.assertTrue(actual["feedback_id"].startswith(simulation._trial_id+"@"))
            self.assertAlmostEqual(actual["timestamp_s"], 1.308)
            self.assertFalse(actual["evidence_passed"])
            self.assertNotIn("object_pose", actual)
            self.assertNotIn("contact", actual)

    def test_superseded_events_and_commands_at_barrier_are_not_reported_as_applied(self):
        simulation, curve, value = _fixture()
        curve.events[0]["time"], curve.events[1]["time"] = .0005, .0015
        curve.events[3]["time"] = curve.executable_until
        with TemporaryDirectory() as temporary, patch("recorded_simulation.mujoco.mj_forward"):
            report = simulation.execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
                initial_feedback=_initial_feedback(simulation), evaluation_mode="diagnostic_rejected_candidate",
                output=Path(temporary)/"trial", max_physics_steps=4)
        self.assertEqual([row["event_id"] for row in report["events"]], ["close-original"])
        self.assertEqual([row["event_id"] for row in report["superseded_gripper_events"]], ["open-original"])
        self.assertEqual([row["event_id"] for row in report["unapplied_gripper_events"]], ["open-original", "hold-later"])
        self.assertFalse(report["planner_accepted"])


if __name__ == "__main__":
    unittest.main()
# END independent fixed-finger execution contract regressions.
