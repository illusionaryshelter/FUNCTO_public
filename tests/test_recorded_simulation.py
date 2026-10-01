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
# BEGIN exercise production execution through the thin recording adapter.
from manipulation.execution import ExecutionGuardError, ExecutionLimits, FixedFingerTargets, TrajectoryExecutor
from manipulation.robosuite_execution import RobosuiteExecutionBackend
from recorded_simulation import load_saved_trajectory
# END exercise production execution through the thin recording adapter.


def _event(event_id, kind, time, evidence=None):
    return {"skill_id": "declared_skill", "event_id": event_id, "kind": kind,
            "time": time, "progress": time, "requires_evidence": evidence, "stop": evidence is not None}


def _fixture():
    """A mocked clock/state source with the installed, real GRIP controller."""
    from robosuite.controllers.parts.gripper.simple_grip import SimpleGripController

    simulation = RobosuiteExecutionBackend.__new__(RobosuiteExecutionBackend)
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
    simulation._description = {"scope": "mocked integration clock only"}
    simulation.native_data = SimpleNamespace(
        qpos=np.r_[np.linspace(.1, .7, 7), .012, -.017],
        qvel=np.r_[np.linspace(.01, .07, 7), .03, -.04],
        ctrl=np.r_[np.zeros(7), .026, -.023], time=1.3,
        xmat=np.eye(3).reshape(1, 9), xpos=np.zeros((1, 3)),
        site_xmat=np.eye(3).reshape(1, 9), site_xpos=np.zeros((1, 3)))
    simulation.native_model = SimpleNamespace(
        site=lambda index: SimpleNamespace(name="declared_test_tcp"),
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


# BEGIN explicit fake geometry verdicts only for execution sequencing tests.
class _GeometryChecker:
    """Record exact inputs; supplied verdicts do not represent physical geometry."""

    name = "scripted_test_geometry_verdict"

    def __init__(self, joint_names, *, rejected_stage=None, rejected_time=None, preflight_passed=True):
        self.joint_names = tuple(joint_names)
        self.rejected_stage = rejected_stage
        self.rejected_time = rejected_time
        self.preflight_passed = preflight_passed
        self.states = []
        self.preflight_inputs = []

    def describe(self):
        return {"name": self.name, "joint_names": list(self.joint_names),
                "scope": "Scripted control-contract test verdicts; no independent geometry calculation",
                "pair_order": [["mock_link_a", "mock_link_b"]]}

    def preflight(self, trajectory, initial_feedback, execution_limits):
        self.preflight_inputs.append(deepcopy(initial_feedback))
        return {"passed": self.preflight_passed,
                "violations": [] if self.preflight_passed else [
                    {"kind": "scripted_preflight_rejection", "name": "test_reference", "measured": "rejected"}]}

    def check_state(self, feedback, *, stage, trajectory_time):
        self.states.append({"feedback": deepcopy(feedback), "stage": stage, "trajectory_time": trajectory_time})
        rejected = (stage == self.rejected_stage
                    and (self.rejected_time is None or trajectory_time == self.rejected_time))
        return {"passed": not rejected, "distances_m": [-.01 if rejected else .5],
                "violations": [] if not rejected else [{"kind": "scripted_geometry_rejection",
                    "name": "mock_link_a/mock_link_b", "measured": -.01}]}


def _executor(backend):
    return TrajectoryExecutor(backend, geometry_checker=_GeometryChecker(backend.joint_names))
# END explicit fake geometry verdicts only for execution sequencing tests.


def _initial_feedback(simulation):
    """Explicit calibration metadata accompanies the measured robot state."""
    return {**simulation.proprioception(), "tool_transform": simulation.tool_transform.tolist()}


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


# BEGIN require the saved hand contract even for rejected physical diagnostics.
class SavedExecutionContractTests(unittest.TestCase):
    def test_all_execution_modes_require_the_same_complete_saved_limits(self):
        from recorded_task_planning import save_curve

        simulation, curve, _ = _fixture()
        limits = _limits(simulation, curve)
        for mode, classification in (("accepted_plan", "planner_accepted"),
                                     ("diagnostic_rejected_candidate", "rejected_diagnostic_not_executable")):
            with self.subTest(mode=mode), TemporaryDirectory() as temporary:
                path = Path(temporary)/"curve.npz"
                save_curve(path, curve, classification=classification, execution_limits=limits.describe())
                loaded = load_saved_trajectory(path, evaluation_mode=mode, execution_limits=limits)
                self.assertEqual(loaded.events, curve.events)
                metadata_path = path.with_suffix(".json")
                original = json.loads(metadata_path.read_text())
                for missing in (True, False):
                    metadata = deepcopy(original)
                    if missing:
                        del metadata["execution_limits"]
                    else:
                        metadata["execution_limits"]["initial_fingers"]["position_bounds"][0][0] += .001
                    metadata_path.write_text(json.dumps(metadata))
                    with self.assertRaisesRegex(ValueError, "identical physical execution limits"):
                        load_saved_trajectory(path, evaluation_mode=mode, execution_limits=limits)
# END historical diagnostic files cannot enter another execution path.


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
                        _executor(simulation).execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
                            initial_feedback=feedback, evaluation_mode="accepted_plan", output=output,
                            execution_limits=_limits(simulation, curve),
                            start_position_tolerance=1., start_velocity_tolerance=1.)
                    self.assertFalse(output.exists())

    def test_native_step_clock_records_all_commands_and_actual_state_at_barrier(self):
        simulation, curve, value = _fixture()
        initial = _initial_feedback(simulation)
        original_events = deepcopy(curve.events)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            report = _executor(simulation).execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
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
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _executor(simulation).execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
                initial_feedback=_initial_feedback(simulation), evaluation_mode="diagnostic_rejected_candidate",
                execution_limits=_limits(simulation, curve),
                output=Path(temporary)/"trial", max_physics_steps=4)
        self.assertEqual([row["event_id"] for row in report["events"]], ["close-original"])
        self.assertEqual([row["event_id"] for row in report["superseded_gripper_events"]], ["open-original"])
        self.assertEqual([row["event_id"] for row in report["unapplied_gripper_events"]], ["open-original", "hold-later"])
        self.assertFalse(report["planner_accepted"])


# BEGIN physical feedback rejection and isolated native FK regressions.
class ExecutionGuardTests(unittest.TestCase):
    def test_non_grid_close_cannot_narrow_actual_interval_before_new_feedback(self):
        simulation, curve, value = _fixture()
        self.assertEqual(curve.events[1]["time"], .003)
        self.assertEqual(simulation._dt, .002)
        policy = _limit_values(simulation, curve)
        # This tighter interval contains the currently measured q, but there
        # is no new feedback at the future close event. It must not replace
        # the preceding allowed set during the delayed physical application.
        policy["events"][1]["position_bounds"] = [[.011, .019], [-.019, -.015]]
        with TemporaryDirectory() as temporary:
            output = Path(temporary)/"trial"
            with self.assertRaisesRegex(ValueError, "fresh measured feedback at the original evidence barrier"):
                _executor(simulation).execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
                    initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(policy),
                    evaluation_mode="accepted_plan", output=output)
            self.assertFalse(output.exists())
            self.assertEqual(simulation.native_data.time, 1.3)
            np.testing.assert_array_equal(simulation.native_data.ctrl[7:], [.026, -.023])

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
                _executor(simulation).execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
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
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                _executor(simulation).execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
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
            with self.subTest(kind=kind), TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
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
                    _executor(simulation).execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
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
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                _executor(simulation).execute(curve, finger_targets=FixedFingerTargets.from_mapping(value),
                    initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(limits),
                    evaluation_mode="accepted_plan", output=output)
            report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["executed_steps"], 4)
            self.assertEqual(report["feedback_failure"]["curve_time_s"], .0075)
            self.assertIn("planned_stop_velocity", [row["kind"] for row in report["feedback_failure"]["violations"]])
            self.assertFalse(report["evidence_passed"])
            self.assertFalse(report["barrier_crossed"])


# BEGIN mandatory production geometry gates and explicit diagnostic references.
class GeometryExecutionTests(unittest.TestCase):
    def test_geometry_checker_is_required_and_bound_to_all_actual_joint_names(self):
        simulation, _, _ = _fixture()
        with self.assertRaises(TypeError):
            TrajectoryExecutor(simulation)
        for checker in (None, lambda feedback: True, _GeometryChecker(simulation.arm_joint_names)):
            with self.subTest(checker=checker), self.assertRaisesRegex(ValueError, "named independent geometry checker"):
                TrajectoryExecutor(simulation, geometry_checker=checker)
        self.assertEqual(simulation.native_data.time, 1.3)
        np.testing.assert_array_equal(simulation.native_data.ctrl[7:], [.026, -.023])

    def test_rejected_preflight_persists_its_verdict_without_any_native_control_or_step(self):
        simulation, curve, targets = _fixture()
        checker = _GeometryChecker(simulation.joint_names, preflight_passed=False)
        with TemporaryDirectory() as temporary:
            output = Path(temporary)/"trial"
            with self.assertRaisesRegex(ExecutionGuardError, "geometry preflight"):
                TrajectoryExecutor(simulation, geometry_checker=checker).execute(curve,
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), evaluation_mode="diagnostic_control_reference", output=output)
            report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["feedback_failure"]["point"], "geometry_preflight")
            self.assertEqual(report["executed_steps"], 0)
            self.assertFalse(report["geometry_preflight"]["passed"])
            self.assertEqual(report["events"], [])
            self.assertEqual(checker.states, [])
            self.assertFalse(report["abort_hold"]["native_ctrl_written"])
            self.assertEqual(simulation.native_data.time, 1.3)
            np.testing.assert_array_equal(simulation.native_data.ctrl[7:], [.026, -.023])
            with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["actuator_ctrl"].shape, (0, 9))
                self.assertEqual(saved["actual_q"].shape, (1, 9))
                self.assertFalse(saved["geometry_checked"][0])
                self.assertTrue(np.isnan(saved["geometry_distances_m"][0]).all())

    def test_geometry_rejects_original_close_before_goal_change_and_receives_actual_fingers(self):
        simulation, curve, targets = _fixture()
        checker = _GeometryChecker(simulation.joint_names, rejected_stage="command_event", rejected_time=.003)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                TrajectoryExecutor(simulation, geometry_checker=checker).execute(curve,
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), evaluation_mode="accepted_plan", output=output)
            report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["executed_steps"], 2)
            self.assertEqual(report["feedback_failure"]["event_id"], "close-original")
            self.assertEqual(report["feedback_failure"]["curve_time_s"], .003)
            self.assertEqual([row["event_id"] for row in report["events"]], ["open-original"])
            self.assertEqual(report["geometry_check_counts"]["command_event"], 2)
            self.assertEqual(report["event_gate_checks"][-1]["geometry_check"]["distances_m"], [-.01])
            final_check = checker.states[-1]
            self.assertEqual(final_check["stage"], "command_event")
            np.testing.assert_array_equal(final_check["feedback"]["q"][7:], [.012, -.017])
            np.testing.assert_array_equal(final_check["feedback"]["actuator_ctrl"]["values"][7:], [.032, -.025])
            np.testing.assert_array_equal(simulation.native_data.ctrl[7:], [.032, -.025])
            self.assertFalse(report["abort_hold"]["physical_stop_confirmed"])

    def test_post_step_geometry_rejection_retains_the_completed_sample_and_pair_distances(self):
        simulation, curve, targets = _fixture()
        checker = _GeometryChecker(simulation.joint_names, rejected_stage="after_step")
        observed = []
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                TrajectoryExecutor(simulation, geometry_checker=checker).execute(curve,
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), evaluation_mode="diagnostic_rejected_candidate", output=output,
                    step_callback=lambda model, data, sample: observed.append(sample["geometry_check"]))
            report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["executed_steps"], 1)
            self.assertEqual(report["feedback_failure"]["point"], "after_step")
            self.assertEqual(len(observed), 1)
            self.assertFalse(observed[0]["passed"])
            with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["geometry_checked"], [True, True])
                np.testing.assert_array_equal(saved["geometry_passed"], [True, False])
                np.testing.assert_array_equal(saved["geometry_distances_m"], [[.5], [-.01]])
                self.assertEqual(saved["actual_q"].shape, (2, 9))
                self.assertEqual(saved["requested_arm_torque"].shape, (1, 7))
                self.assertEqual(saved["arm_torque_saturated"].shape, (1, 7))

    def test_checker_exception_is_a_rejection_with_unavailable_distance_evidence(self):
        simulation, curve, targets = _fixture()
        checker = _GeometryChecker(simulation.joint_names)
        checker.check_state = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("independent query unavailable"))
        with TemporaryDirectory() as temporary:
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                TrajectoryExecutor(simulation, geometry_checker=checker).execute(curve,
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), evaluation_mode="accepted_plan", output=output)
            report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["executed_steps"], 0)
            self.assertEqual(report["feedback_failure"]["violations"][0]["kind"], "geometry_checker_error")
            with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
                self.assertFalse(saved["geometry_checked"][0])
                self.assertFalse(saved["geometry_passed"][0])
                self.assertTrue(np.isnan(saved["geometry_distances_m"]).all())

    def test_control_reference_has_its_own_classification_and_supports_exact_entrance(self):
        simulation, curve, targets = _fixture()
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            report = _executor(simulation).execute(curve,
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=_limits(simulation, curve), evaluation_mode="diagnostic_control_reference", output=output,
                start_position_tolerance=0., start_velocity_tolerance=0.)
            self.assertFalse(report["planner_accepted"])
            self.assertEqual(report["reference_origin"], "explicit_control_diagnostic")
            self.assertEqual(report["executed_steps"], 4)
            self.assertTrue(report["geometry_preflight"]["passed"])
            self.assertEqual(report["geometry_check_counts"],
                             {"initial_state": 1, "before_step": 4, "command_event": 2, "after_step": 4})

    def test_initial_frame_or_tcp_relabelling_fails_before_creating_output(self):
        for field, wrong in (("reference_frame", "another_frame"), ("tool_link", "another_site"),
                             ("tool_transform", np.diag([1., 1., 1., 2.]))):
            simulation, curve, targets = _fixture()
            initial = _initial_feedback(simulation)
            initial[field] = wrong
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                output = Path(temporary)/"trial"
                with self.assertRaises(ValueError):
                    _executor(simulation).execute(curve, finger_targets=FixedFingerTargets.from_mapping(targets),
                        initial_feedback=initial, execution_limits=_limits(simulation, curve),
                        evaluation_mode="accepted_plan", output=output)
                self.assertFalse(output.exists())
                self.assertEqual(simulation.native_data.time, 1.3)
# END mandatory production geometry gates and explicit diagnostic references.


# BEGIN concrete native backend binding, independent of recorded datasets.
class RobosuiteBackendTests(unittest.TestCase):
    @staticmethod
    def _simulator(*, arm_attributes='gaintype="fixed" dyntype="none" gear="1"'):
        from robosuite.utils.binding_utils import MjSim

        model = mujoco.MjModel.from_xml_string(f'''
            <mujoco><compiler angle="radian"/><option timestep=".002"/>
              <worldbody><body name="base"><body name="arm">
                <joint name="arm_q" type="hinge" axis="0 0 1" range="-2 2"/>
                <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
                <geom type="sphere" size=".02"/><site name="tcp" pos=".1 0 0"/>
                <body name="left"><joint name="left_q" type="slide" axis="0 1 0" range="0 .04"/>
                  <geom type="sphere" size=".01" mass=".1"/></body>
                <body name="right"><joint name="right_q" type="slide" axis="0 1 0" range="-.04 0"/>
                  <geom type="sphere" size=".01" mass=".1"/></body>
              </body></body></worldbody>
              <actuator><general name="arm_motor" joint="arm_q" {arm_attributes}
                gainprm="1" biastype="none" ctrllimited="true" ctrlrange="-2 2"/>
                <position name="left_servo" joint="left_q" kp="100" ctrllimited="true" ctrlrange="0 .04"/>
                <position name="right_servo" joint="right_q" kp="100" ctrllimited="true" ctrlrange="-.04 0"/>
              </actuator></mujoco>''')
        sim = MjSim(model)
        sim.data._data.qpos[:] = [.2, .012, -.017]
        sim.data._data.ctrl[:] = [0., .026, -.023]
        mujoco.mj_forward(model, sim.data._data)
        return sim

    @staticmethod
    def _bind(sim):
        return RobosuiteExecutionBackend(sim, base_link="base", tcp_site="tcp", arm_joint_names=("arm_q",),
            finger_joint_names=("left_q", "right_q"), arm_kp=1000., arm_kd=200.,
            velocity_limits={"arm_q": 1., "left_q": .2, "right_q": .2}, source="Explicit tiny native CPU binding test")

    def test_existing_native_state_and_actuator_semantics_survive_binding_and_control(self):
        sim = self._simulator()
        q, dq, ctrl, initial_time = sim.data._data.qpos.copy(), sim.data._data.qvel.copy(), sim.data._data.ctrl.copy(), sim.data._data.time
        backend = self._bind(sim)
        np.testing.assert_array_equal(backend.native_data.qpos, q)
        np.testing.assert_array_equal(backend.native_data.qvel, dq)
        np.testing.assert_array_equal(backend.actuator_control, ctrl)
        self.assertEqual(backend.sim_time, initial_time)
        self.assertEqual(backend.timestep, .002)
        self.assertEqual(backend._arm_controller.input_type, "absolute")
        self.assertIsNone(backend._arm_controller.interpolator)
        self.assertIsNone(backend._grip_controller.interpolator)
        control = backend.apply_control(np.asarray([1.]), np.asarray([.004, -.009]))
        # BEGIN the backend reports only arm torque command clipping here.
        self.assertEqual(control["arm_control_saturated_coordinates"], 1)
        # END native gripper force limiting is a separate observation.
        self.assertTrue(control["arm_torque_saturated"][0])
        self.assertGreater(control["requested_arm_torque"][0], 2.)
        np.testing.assert_array_equal(control["actuator_ctrl"], [2., .004, -.009])
        np.testing.assert_array_equal(backend.proprioception()["q"], q)
        self.assertEqual(backend.sim_time, initial_time)
        feedback = backend.proprioception()
        self.assertEqual(feedback["reference_frame"], "world")
        self.assertEqual(feedback["tool_link"], "tcp")
        np.testing.assert_array_equal(feedback["tool_transform"], np.eye(4))

    def test_non_torque_arm_transmissions_are_rejected_before_control(self):
        variants = ('gaintype="affine" dyntype="none" gear="1"',
                    'gaintype="fixed" dyntype="filter" dynprm=".02" gear="1"',
                    'gaintype="fixed" dyntype="none" gear="1 1 0 0 0 0"')
        for attributes in variants:
            with self.subTest(attributes=attributes):
                sim = self._simulator(arm_attributes=attributes)
                initial_ctrl = sim.data._data.ctrl.copy()
                with self.assertRaisesRegex(ValueError, "unit-gain torque motors"):
                    self._bind(sim)
                self.assertEqual(sim.data._data.time, 0.)
                np.testing.assert_array_equal(sim.data._data.ctrl, initial_ctrl)
# END concrete native backend binding, independent of recorded datasets.


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
        simulation = RobosuiteExecutionBackend.__new__(RobosuiteExecutionBackend)
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
    # BEGIN force-limit observations use native enable flags and exact bounds.
    def test_force_limit_samples_are_distinct_from_tracking_and_disabled_ranges(self):
        from probe_mounted_execution import summarize_finger_force_limits

        forces = np.array([[-20., 20., 0.],
                           [np.nextafter(-20., 0.), np.nextafter(20., 0.), 0.],
                           [-20.0001, 20.0001, 0.],
                           [np.nan, 20., 0.]])
        result = summarize_finger_force_limits(np.array([0., .002, .004, .006]), forces,
            joint_names=("finger_a", "finger_b", "unlimited_finger"),
            actuator_names=("servo_a", "servo_b", "unlimited_servo"),
            forcelimited=[True, True, False], forcerange=[[-20., 20.], [-20., 20.], [0., 0.]])
        self.assertEqual(result["at_limit_sample_count"], [2, 3, 0])
        self.assertEqual(result["first_at_limit_elapsed_s"], [0., 0., None])
        self.assertEqual(result["last_at_limit_elapsed_s"], [.004, .006, None])
        self.assertEqual(result["nonfinite_force_sample_count"], [1, 0, 0])
        self.assertEqual(result["sample_count"], 4)
    # END one-ULP-inside samples and disabled zero ranges do not count.

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
