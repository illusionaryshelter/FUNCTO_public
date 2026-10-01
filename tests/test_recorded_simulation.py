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
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent/"planning"))

from manipulation.planning import JointTrajectory
from recorded_simulation import ExecutionGuardError, ExecutionLimits, FixedFingerTargets, RecordedSimulation


def _event(event_id, kind, time, evidence=None):
    return {"skill_id": "declared_skill", "event_id": event_id, "kind": kind,
            "time": time, "requires_evidence": evidence, "stop": evidence is not None}


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
        actuator_ctrlrange=np.vstack((np.tile([-10., 10.], (7, 1)), [0., .04], [-.04, 0.])),
        jnt_limited=np.ones(9, dtype=bool),
        jnt_range=np.vstack((np.tile([-2., 2.], (7, 1)), [0., .04], [-.04, 0.])))
    # BEGIN explicitly synthetic physical limits and FK for control-contract tests.
    simulation._velocity_limits = dict(zip(simulation.joint_names, [1.]*7+[.2, .2]))
    simulation.reference_tool_pose = lambda arm_q: np.eye(4)
    # END explicitly synthetic physical limits and FK for control-contract tests.

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


# BEGIN explicit mock execution policy; these values are not robot calibration.
def _limit_values(simulation, curve):
    finger_state = {"position_bounds": [[.01, .02], [-.02, -.014]], "maximum_speed": [.05, .05]}
    return {"source": "Synthetic CPU contract-test limits; not measured hardware performance",
            "joint_names": list(simulation.joint_names), "arm_joint_names": list(simulation.arm_joint_names),
            "finger_joint_names": list(simulation.finger_joint_names),
            "arm_position_error": [.001]*7, "arm_velocity_error": [.001]*7,
            "tcp_position_error_m": .001, "tcp_rotation_error_rad": .001,
            "stop_arm_velocity": [.1]*7, "initial_fingers": deepcopy(finger_state),
            "events": [{"skill_id": event["skill_id"], "event_id": event["event_id"], **deepcopy(finger_state)}
                       for event in curve.events if event["kind"] in ("open", "close", "hold")]}


def _limits(simulation, curve):
    return ExecutionLimits.from_mapping(_limit_values(simulation, curve))
# END explicit mock execution policy; these values are not robot calibration.


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
                            execution_limits=_limits(simulation, curve),
                            start_position_tolerance=1., start_velocity_tolerance=1.)
                    self.assertFalse(output.exists())

    def test_native_step_clock_records_all_commands_and_actual_state_at_barrier(self):
        simulation, curve, value = _fixture()
        initial = _initial_feedback(simulation)
        original_events = deepcopy(curve.events)
        with TemporaryDirectory() as temporary, patch("recorded_simulation.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            report = simulation.execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
                initial_feedback=initial, execution_limits=_limits(simulation, curve),
                evaluation_mode="accepted_plan", output=output, max_physics_steps=4)
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
                execution_limits=_limits(simulation, curve),
                output=Path(temporary)/"trial", max_physics_steps=4)
        self.assertEqual([row["event_id"] for row in report["events"]], ["close-original"])
        self.assertEqual([row["event_id"] for row in report["superseded_gripper_events"]], ["open-original"])
        self.assertEqual([row["event_id"] for row in report["unapplied_gripper_events"]], ["open-original", "hold-later"])
        self.assertFalse(report["planner_accepted"])


# BEGIN physical feedback rejection and isolated native FK regressions.
class ExecutionGuardTests(unittest.TestCase):
    def test_missing_limits_extra_clock_and_unbound_finger_event_fail_before_physics(self):
        simulation, curve, value = _fixture()
        policy = _limit_values(simulation, curve)
        missing = deepcopy(policy)
        missing.pop("arm_position_error")
        with self.assertRaisesRegex(ValueError, "every physical limit"):
            ExecutionLimits.from_mapping(missing)
        timed = deepcopy(policy)
        timed["events"][0]["time"] = 0.
        with self.assertRaisesRegex(ValueError, "original event identity"):
            ExecutionLimits.from_mapping(timed)
        policy["events"].pop()
        with TemporaryDirectory() as temporary:
            output = Path(temporary)/"trial"
            with self.assertRaisesRegex(ValueError, "match every original"):
                simulation.execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
                    initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(policy),
                    evaluation_mode="accepted_plan", output=output)
            self.assertFalse(output.exists())
            self.assertEqual(simulation.native_data.time, 1.3)

    def test_close_gate_uses_original_anchor_and_never_sends_rejected_close(self):
        simulation, curve, value = _fixture()
        curve.events[0]["time"] = 0.
        limits = _limit_values(simulation, curve)
        limits["arm_position_error"] = [1e-5]*7
        initial = _initial_feedback(simulation)
        steps = []

        def step():
            simulation.native_data.time += simulation._dt
            elapsed = simulation.native_data.time-initial["sim_time"]
            q, dq, _ = curve.sample(elapsed)
            simulation.native_data.qpos[:7], simulation.native_data.qvel[:7] = q, dq
            steps.append(simulation.native_data.ctrl.copy())

        simulation._sim.step = step
        with TemporaryDirectory() as temporary, patch("recorded_simulation.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                simulation.execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
                    initial_feedback=initial, execution_limits=ExecutionLimits.from_mapping(limits),
                    evaluation_mode="accepted_plan", output=output)
            report = json.loads((output/"execution-report.json").read_text())
            failure = report["feedback_failure"]
            self.assertEqual(failure["point"], "command_event")
            self.assertEqual(failure["event_id"], "close-original")
            self.assertEqual(failure["curve_time_s"], .003)
            self.assertEqual(len(steps), 2)
            self.assertEqual(report["executed_steps"], 2)
            self.assertEqual([event["event_id"] for event in report["events"]], ["open-original"])
            self.assertIn("close-original", [event["event_id"] for event in report["unapplied_gripper_events"]])
            np.testing.assert_array_equal(simulation.native_data.ctrl[7:], [.032, -.025])
            np.testing.assert_array_equal(report["abort_hold"]["finger_position_goal"], [.032, -.025])
            self.assertTrue(report["abort_hold"]["prepared"])
            self.assertFalse(report["abort_hold"]["native_ctrl_written"])
            self.assertFalse(report["abort_hold"]["physical_stop_confirmed"])
            self.assertIsNone(report["barrier_feedback_path"])
            self.assertTrue(Path(report["robot_feedback_path"]).name == "failure-feedback.json")

    def test_post_step_failures_retain_actual_sample_and_cannot_advance_to_close(self):
        changes = (
            ("arm_position_tracking", lambda data: data.qpos.__setitem__(0, .11)),
            ("actual_finger_position", lambda data: data.qpos.__setitem__(7, .025)),
            ("actual_finger_velocity", lambda data: data.qvel.__setitem__(7, .08)),
            ("native_joint_position", lambda data: data.qpos.__setitem__(0, 2.1)),
            ("hardware_joint_velocity", lambda data: data.qvel.__setitem__(0, 1.1)),
            ("tcp_tracking", lambda data: data.site_xpos.__setitem__((0, 0), .01)),
            ("feedback_clock", lambda data: setattr(data, "time", data.time+.002)),
            ("nonfinite_feedback", lambda data: data.qvel.__setitem__(0, np.nan)),
        )
        for kind, change in changes:
            with self.subTest(kind=kind), TemporaryDirectory() as temporary, patch("recorded_simulation.mujoco.mj_forward"):
                simulation, curve, value = _fixture()
                initial = _initial_feedback(simulation)
                original_step = simulation._sim.step
                observed = []

                def step():
                    original_step()
                    change(simulation.native_data)

                simulation._sim.step = step
                output = Path(temporary)/"trial"
                with self.assertRaises(ExecutionGuardError):
                    simulation.execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
                        initial_feedback=initial, execution_limits=_limits(simulation, curve),
                        evaluation_mode="accepted_plan", output=output,
                        step_callback=lambda model, data, sample: observed.append(sample["q"].copy()))
                report = json.loads((output/"execution-report.json").read_text())
                self.assertEqual(report["status"], "feedback_rejected")
                self.assertEqual(report["executed_steps"], 1)
                self.assertEqual(report["feedback_failure"]["point"], "after_step")
                self.assertIn(kind, [item["kind"] for item in report["feedback_failure"]["violations"]])
                self.assertEqual(report["events"], [])
                self.assertEqual(len(observed), 1)
                if kind in ("feedback_clock", "native_joint_position", "nonfinite_feedback"):
                    self.assertFalse(report["abort_hold"]["prepared"])
                np.testing.assert_array_equal(simulation.native_data.ctrl[7:], [.026, -.023])
                with np.load(output/"physical-trajectory.npz") as saved:
                    self.assertEqual(saved["actual_q"].shape, (2, 9))
                    self.assertEqual(saved["actuator_ctrl"].shape, (1, 9))
                    np.testing.assert_array_equal(saved["actual_q"][-1], simulation.native_data.qpos)

    def test_original_stop_checks_actual_velocity_without_extending_physics(self):
        simulation, curve, value = _fixture()
        limits = _limit_values(simulation, curve)
        limits["stop_arm_velocity"] = [.001]*7
        with TemporaryDirectory() as temporary, patch("recorded_simulation.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                simulation.execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
                    initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(limits),
                    evaluation_mode="accepted_plan", output=output)
            report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["executed_steps"], 4)
            self.assertEqual(report["feedback_failure"]["curve_time_s"], .0075)
            self.assertIn("planned_stop_velocity", [row["kind"] for row in report["feedback_failure"]["violations"]])
            self.assertFalse(report["evidence_passed"])
            self.assertFalse(report["barrier_crossed"])


class ReferenceFKTests(unittest.TestCase):
    @staticmethod
    def _native_robot():
        model = mujoco.MjModel.from_xml_string('''
            <mujoco><compiler angle="radian"/><worldbody><body name="base">
              <body name="arm"><joint name="arm_q" type="hinge" axis="0 0 1" range="-2 2"/>
                <geom type="sphere" size=".02" mass="1"/><site name="tcp" pos=".1 0 0"/>
              </body>
              <body name="finger"><joint name="finger_q" type="slide" axis="0 1 0" range="0 .04"/>
                <geom type="sphere" size=".01" mass=".1"/><site name="finger_tcp" pos=".02 0 0"/>
              </body>
            </body></worldbody></mujoco>''')
        simulation = RecordedSimulation()
        simulation.native_model, simulation.native_data = model, mujoco.MjData(model)
        simulation.arm_joint_names, simulation.finger_joint_names = ("arm_q",), ("finger_q",)
        simulation.joint_names = simulation.arm_joint_names+simulation.finger_joint_names
        simulation._joint_ids = np.asarray([model.joint(name).id for name in simulation.joint_names])
        simulation._qpos_ids = model.jnt_qposadr[simulation._joint_ids].copy()
        simulation._base_id, simulation._site_id = model.body("base").id, model.site("tcp").id
        simulation.native_data.qpos[:] = [.2, .013]
        mujoco.mj_kinematics(model, simulation.native_data)
        return simulation

    def test_reference_fk_uses_isolated_data_and_never_steps_or_teleports_actual_robot(self):
        simulation = self._native_robot()
        actual_q = simulation.native_data.qpos.copy()
        actual_sites = simulation.native_data.site_xpos.copy()
        simulation._prepare_reference_fk()
        pose = simulation.reference_tool_pose(np.asarray([.5]))
        np.testing.assert_allclose(pose[:3, 3], [.1*np.cos(.5), .1*np.sin(.5), 0.], atol=1e-12)
        np.testing.assert_array_equal(simulation.native_data.qpos, actual_q)
        np.testing.assert_array_equal(simulation.native_data.site_xpos, actual_sites)
        self.assertEqual(simulation.native_data.time, 0.)
        self.assertIsNot(simulation._reference_data, simulation.native_data)

    def test_finger_dependent_tcp_requires_a_defined_reference_finger_path(self):
        simulation = self._native_robot()
        simulation._site_id = simulation.native_model.site("finger_tcp").id
        with self.assertRaisesRegex(ValueError, "only the explicitly controlled arm"):
            simulation._prepare_reference_fk()


class FingerConvergenceTests(unittest.TestCase):
    def test_contact_blocked_quiet_position_is_not_command_convergence(self):
        from probe_finger_control import convergence
        result = convergence(np.array([0., .1, .2, .3]), np.full((4, 2), .015), np.zeros((4, 2)),
                             np.zeros(2), np.full(2, .001), np.full(2, .001), .2)
        self.assertFalse(result["settled_at_end"])
        self.assertIsNone(result["first_settled_sim_time_s"])

    def test_lost_target_does_not_borrow_an_earlier_stable_window(self):
        from probe_finger_control import convergence
        positions = np.zeros((5, 2))
        positions[3, 0] = .01
        result = convergence(np.arange(5)*.1, positions, np.zeros_like(positions),
                             np.zeros(2), np.full(2, .001), np.full(2, .001), .2)
        self.assertEqual(result["first_settled_sim_time_s"], .2)
        self.assertFalse(result["settled_at_end"])
        self.assertEqual(result["final_consecutive_window_s"], 0.)
# END physical feedback rejection and isolated native FK regressions.


if __name__ == "__main__":
    unittest.main()
# END independent fixed-finger execution contract regressions.
