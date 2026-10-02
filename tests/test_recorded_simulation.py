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

from manipulation.planning import JointTrajectory, PlanningResult
# BEGIN exercise production execution through the thin recording adapter.
from manipulation.execution import ExecutionGuardError, ExecutionLimits, FixedFingerTargets, TrajectoryExecutor
from manipulation.gripper_trajectory import FingerTrajectory
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
        # BEGIN synthetic native dynamics are independent of mock q/dq/ctrl.
        qacc=np.linspace(-.4, .4, 9), actuator_force=np.arange(1., 10.),
        # END these sentinel arrays do not represent physical integration.
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
            # Small explicit moves fit this eight-millisecond mocked prefix;
            # unlike the old step commands they form a bounded C1 reference.
            {"skill_id": "declared_skill", "event_id": "open-original", "kind": "open", "positions": [.0121, -.0171]},
            {"skill_id": "declared_skill", "event_id": "close-original", "kind": "close", "positions": [.01205, -.01705]},
            {"skill_id": "declared_skill", "event_id": "hold-later", "kind": "hold", "positions": [.01205, -.01705]}]}
    _bind_fingers(simulation, curve, targets)
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

    # BEGIN record the exact scene handoff in this nonphysical test double.
    def bind_scene(self, binding, *, progress_at_time, nominal_fingers):
        self.scene_binding = deepcopy(binding)
        self.progress_at_time = progress_at_time
        self.nominal_fingers = nominal_fingers
    # END production geometry is independently tested elsewhere.

    def preflight(self, trajectory, initial_feedback, execution_limits, *, start_time=0., until=None,
                  current_time=None):
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


# BEGIN synthetic formal evidence for sequencing tests only.
def _result(curve):
    from manipulation.execution_binding import curve_snapshot
    identity = str(uuid4())
    result = PlanningResult(True, curve, {"scope": "synthetic CPU control-contract fixture"}, {
        "candidate_id": identity, "curve": curve_snapshot(curve),
        "problem": {"source": "synthetic CPU sequencing test, not native planner output",
                    "reference_frame": "world", "tool_link": "declared_test_tcp",
                    "base_pose": np.eye(4).tolist(), "tool_transform": np.eye(4).tolist()},
        "scene": {"gripper": None, "source": "scripted checker fixture"},
        "validation": {"candidate_id": identity, "completed": True,
            "native": {"accepted": True, "scope": "scripted sequencing evidence"},
            "export": {"u": [0., 1.], "violations": [0., 0.]}}, "execution_limits": None})
    result.execution_binding["validation"]["inputs"] = deepcopy({
        key: result.execution_binding[key] for key in ("problem", "scene")})
    return result
# END constructed evidence is never described as physical feasibility.


def _executor(backend):
    return TrajectoryExecutor(backend, geometry_checker=_GeometryChecker(backend.joint_names))
# END explicit fake geometry verdicts only for execution sequencing tests.


def _initial_feedback(simulation):
    """Explicit calibration metadata accompanies the measured robot state."""
    return {**simulation.proprioception(), "tool_transform": simulation.tool_transform.tolist()}


# BEGIN explicit CPU reference creation replaces position-step test fixtures.
def _bind_fingers(simulation, curve, targets):
    feedback = _initial_feedback(simulation)
    goals = FixedFingerTargets.from_mapping(targets)
    commands = goals.bind(curve, finger_joint_names=simulation.finger_joint_names,
                         ctrlrange=simulation.finger_ctrlrange)
    curve.finger_trajectory = FingerTrajectory.build(
        joint_names=simulation.finger_joint_names,
        initial_position=feedback["q"][7:], initial_velocity=feedback["dq"][7:], commands=commands,
        duration=curve.duration, executable_until=curve.executable_until, timestep=simulation.timestep,
        position_limits=simulation.joint_ranges[7:].T,
        velocity_limits=[[-.2, -.2], [.2, .2]], acceleration_limits=[[-1000., -1000.], [1000., 1000.]],
        source="Explicit tiny CPU clock fixture; its acceleration bound is not robot calibration",
        feedback_id=feedback["feedback_id"], timestamp_s=feedback["timestamp_s"])
    return curve.finger_trajectory
# END no fixture motion or feedback is generated by the commanded reference.


# BEGIN explicit mock execution policy; these values are not robot calibration.
def _limit_values(simulation, curve):
    finger_state = {"position_bounds": [[.01, .02], [-.02, -.014]], "maximum_speed": [.05, .05]}
    return {"source": "Synthetic CPU contract-test limits; not measured hardware performance",
            "joint_names": list(simulation.joint_names), "arm_joint_names": list(simulation.arm_joint_names),
            "finger_joint_names": list(simulation.finger_joint_names),
            "arm_position_error": [.001]*7, "arm_velocity_error": [.001]*7,
            "tcp_position_error_m": .001, "tcp_rotation_error_rad": .001,
            "stop_arm_velocity": [.1]*7, "stop_finger_velocity": [.05, .05], "initial_fingers": deepcopy(finger_state),
            "observation_policy": {"source": "Explicit synthetic synchronous observer policy",
                "period_s": simulation.timestep, "maximum_age_s": 0., "required_fields": []},
            "events": [{"skill_id": event["skill_id"], "event_id": event["event_id"], **deepcopy(finger_state)}
                       for event in curve.events if event["kind"] in ("open", "close", "hold")]}


def _limits(simulation, curve):
    return ExecutionLimits.from_mapping(_limit_values(simulation, curve))
# END explicit mock execution policy; these values are not robot calibration.


# BEGIN require the saved hand contract even for rejected physical diagnostics.
class SavedExecutionContractTests(unittest.TestCase):
    # BEGIN accepted labels cannot hide a stale candidate or another scene.
    def test_changed_candidate_scene_clock_or_missing_proof_never_reaches_control(self):
        changes = (
            lambda result: setattr(result, "success", False),
            lambda result: setattr(result, "execution_binding", None),
            lambda result: result.execution_binding["validation"].update(candidate_id="another candidate"),
            lambda result: result.execution_binding["scene"].update(source="another scene"),
            lambda result: result.execution_binding["problem"].update(source="another goal"),
            lambda result: result.trajectory.events[0].update(time=.002),
            lambda result: result.trajectory.progress_s.__setitem__(1, .9),
            lambda result: result.trajectory.position.__setitem__((1, 0), .15),
        )
        for change in changes:
            simulation, curve, targets = _fixture()
            result = _result(curve)
            change(result)
            with self.subTest(change=change), TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError):
                    _executor(simulation).execute(result,
                        finger_targets=FixedFingerTargets.from_mapping(targets),
                        initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                        output=Path(temporary)/"trial")
                self.assertEqual(simulation.sim_time, 1.3)
                self.assertEqual(simulation.arm_goals, [])
        simulation, curve, targets = _fixture()
        with TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "formal PlanningResult"):
            _executor(simulation).execute(curve, finger_targets=FixedFingerTargets.from_mapping(targets),
                initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                output=Path(temporary)/"trial")
    # END execution rejects stale binding before any actuator side effect.

    def test_planning_calibration_must_match_actual_even_when_both_local_records_agree(self):
        simulation, curve, targets = _fixture()
        for key, value in (("reference_frame", "other_world"), ("tool_link", "other_tcp"),
                           ("base_pose", (np.eye(4)+np.diag([0., 0., 0., .1])).tolist()),
                           ("tool_transform", (np.eye(4)+np.diag([0., 0., 0., .1])).tolist())):
            result = _result(curve)
            result.execution_binding["problem"][key] = value
            result.execution_binding["validation"]["inputs"]["problem"][key] = value
            with self.subTest(key=key), TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, "Planning and actual"):
                    _executor(simulation).execute(result, finger_targets=FixedFingerTargets.from_mapping(targets),
                        initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                        output=Path(temporary)/"trial")
                self.assertEqual(simulation.arm_goals, [])

    def test_formal_archive_requires_bound_limits_and_rejects_diagnostic_data(self):
        from recorded_task_planning import save_curve

        simulation, curve, _ = _fixture()
        limits = _limits(simulation, curve)
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"curve.npz"
            save_curve(path, curve, classification="planner_accepted", execution_limits=limits.describe(),
                       planning_result=_result(curve))
            loaded = load_saved_trajectory(path, execution_limits=limits)
            self.assertEqual(loaded.trajectory.events, curve.events)
            self.assertEqual(loaded.trajectory.finger_trajectory.describe(), curve.finger_trajectory.describe())
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
                    load_saved_trajectory(path, execution_limits=limits)
            save_curve(path, curve, classification="rejected_diagnostic_not_executable", execution_limits=limits.describe())
            with self.assertRaisesRegex(ValueError, "not executable"):
                load_saved_trajectory(path, execution_limits=limits)

    # BEGIN saved command curves cannot be replaced by old step targets.
    def test_saved_finger_polynomial_requires_matching_metadata_and_all_arrays(self):
        from recorded_task_planning import save_curve

        simulation, curve, _ = _fixture()
        limits = _limits(simulation, curve)
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"curve.npz"
            save_curve(path, curve, classification="planner_accepted", execution_limits=limits.describe(), planning_result=_result(curve))
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text())
            missing = deepcopy(metadata)
            missing["finger_trajectory"] = None
            metadata_path.write_text(json.dumps(missing))
            with self.assertRaisesRegex(ValueError, "complete saved planning finger reference"):
                load_saved_trajectory(path, execution_limits=limits)
            metadata_path.write_text(json.dumps(metadata))
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            arrays["finger_velocity"][0, 0] += .001
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "same original clock and polynomial"):
                load_saved_trajectory(path, execution_limits=limits)
    # END loading retains the original named reference without regeneration.
# END historical diagnostic files cannot enter another execution path.


class FixedFingerTargetTests(unittest.TestCase):
    def test_targets_bind_exact_event_identities_and_retimed_times(self):
        simulation, curve, value = _fixture()
        targets = FixedFingerTargets.from_mapping(value)
        value["events"][0]["positions"][0] = .039
        bound = targets.bind(curve, finger_joint_names=simulation.finger_joint_names,
                             ctrlrange=simulation.native_model.actuator_ctrlrange[7:])
        self.assertEqual([row["time"] for row in bound], [.001, .003, .009])
        self.assertEqual(bound[0]["finger_positions"], [.0121, -.0171])
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
    # BEGIN original events require the already archived nominal finger curve.
    def test_missing_or_changed_finger_reference_fails_before_control(self):
        for mutation in ("missing", "event_clock", "targets"):
            simulation, curve, value = _fixture()
            if mutation == "missing":
                curve.finger_trajectory = None
            elif mutation == "event_clock":
                curve.events[1]["time"] = .0035
            else:
                value["events"][1]["positions"][0] += .00001
            with self.subTest(mutation=mutation), TemporaryDirectory() as temporary:
                output = Path(temporary)/"trial"
                with self.assertRaises(ValueError):
                    _executor(simulation).execute(_result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
                        initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                        output=output)
                self.assertFalse(output.exists())
                self.assertEqual(simulation.sim_time, 1.3)
                np.testing.assert_array_equal(simulation.finger_control, [.026, -.023])

    def test_event_free_arm_diagnostic_explicitly_has_no_finger_reference(self):
        simulation, curve, value = _fixture()
        curve.events, value["events"], curve.finger_trajectory = (), [], None
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            report = _executor(simulation).execute(_result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
                initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                output=output)
            self.assertIsNone(report["finger_reference"])
            np.testing.assert_array_equal(simulation.finger_control, [.026, -.023])
            with np.load(output/"physical-trajectory.npz") as saved:
                self.assertFalse(saved["reference_finger_provided"])
                self.assertTrue(np.isnan(saved["reference_finger_q"]).all())
                self.assertTrue(np.isnan(saved["reference_finger_dq"]).all())

    def test_zero_closing_command_never_relabels_stalled_actual_fingers_or_certifies_holding(self):
        simulation, original, value = _fixture()
        # The mock deliberately keeps measured fingers at a nonzero width.
        # This represents a command/measurement disagreement, not simulated
        # contact evidence. Neither servo convergence nor Holding is inferred.
        simulation.native_data.qvel[:] = 0.
        q = simulation.native_data.qpos[:7].copy()
        curve = JointTrajectory(original.joint_names, np.array([0., .5]), np.stack((q, q)), np.zeros((2, 7)),
            original.progress_u, original.progress_s,
            (_event("open-original", "open", 0.), _event("close-original", "close", .01),
             _event("holding-original", "verification", .4, "Holding")), (), {})
        value["events"] = value["events"][:2]
        value["events"][0]["positions"] = [.012, -.017]
        value["events"][1]["positions"] = [0., 0.]
        _bind_fingers(simulation, curve, value)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            report = _executor(simulation).execute(_result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
                initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                output=output)
            self.assertEqual(report["status"], "stopped_for_evidence")
            self.assertFalse(report["finger_reference_actual_tracking_certified"])
            self.assertFalse(report["evidence_passed"])
            with np.load(output/"physical-trajectory.npz") as saved:
                np.testing.assert_array_equal(saved["actuator_ctrl"][-1, 7:], [0., 0.])
                np.testing.assert_array_equal(saved["reference_finger_q"][-1], [0., 0.])
                np.testing.assert_array_equal(saved["actual_q"][-1, 7:], [.012, -.017])
    # END command references stay separate from actual geometry and evidence.

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
                        _executor(simulation).execute(_result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
                            initial_feedback=feedback, output=output,
                            execution_limits=_limits(simulation, curve),
                            start_position_tolerance=1., start_velocity_tolerance=1.)
                    self.assertFalse(output.exists())

    def test_native_step_clock_records_all_commands_and_actual_state_at_barrier(self):
        simulation, curve, value = _fixture()
        initial = _initial_feedback(simulation)
        original_events = deepcopy(curve.events)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            report = _executor(simulation).execute(_result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
                initial_feedback=initial, execution_limits=_limits(simulation, curve),
                output=output, max_physics_steps=4)
            self.assertEqual(report["executed_steps"], 4)
            self.assertEqual(report["status"], "stopped_for_evidence")
            self.assertFalse(report["evidence_passed"])
            self.assertFalse(report["barrier_crossed"])
            self.assertEqual([(event["event_id"], event["applied_at_s"]) for event in report["events"]],
                             [("open-original", .002), ("close-original", .004)])
            self.assertEqual([row["event_id"] for row in report["unapplied_gripper_events"]], ["hold-later"])
            self.assertEqual(curve.events, original_events)
            np.testing.assert_allclose(simulation.arm_goals[-2], curve.sample(.0075)[0], atol=0., rtol=0.)
            with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["command_q"].shape, (5, 9))
                self.assertEqual(saved["actual_q"].shape, (5, 9))
                self.assertEqual(saved["actual_dq"].shape, (5, 9))
                self.assertEqual(saved["arm_curve_dq"].shape, (5, 7))
                self.assertEqual(saved["actuator_ctrl"].shape, (4, 9))
                self.assertNotIn("desired_dq", saved.files)
                expected_q, expected_dq = curve.finger_trajectory.sample(saved["curve_time"])
                np.testing.assert_array_equal(saved["actuator_ctrl"][:, 7:], expected_q[1:])
                np.testing.assert_array_equal(saved["reference_finger_q"], expected_q)
                np.testing.assert_array_equal(saved["reference_finger_dq"], expected_dq)
                self.assertTrue(saved["reference_finger_provided"])
                np.testing.assert_array_equal(saved["command_q"][0, 7:], [.026, -.023])
                np.testing.assert_array_equal(saved["actual_dq"][0, 7:], [.03, -.04])
                np.testing.assert_allclose(saved["control_elapsed"], [0., .002, .004, .006], atol=1e-15)
                self.assertEqual(saved["curve_time"][-1], .0075)
                self.assertAlmostEqual(saved["elapsed"][-1], .008)
            actual = json.loads(Path(report["barrier_feedback_path"]).read_text())
            self.assertEqual(actual["joint_names"], list(simulation.joint_names))
            np.testing.assert_array_equal(actual["q"], initial["q"])
            np.testing.assert_array_equal(actual["dq"], initial["dq"])
            np.testing.assert_array_equal(actual["actuator_ctrl"]["values"][7:], curve.finger_trajectory.sample(.0075)[0])
            self.assertEqual(actual["timestamp_s"], actual["sim_time"])
            self.assertEqual(actual["reference_frame"], initial["reference_frame"])
            self.assertEqual(actual["tool_link"], initial["tool_link"])
            self.assertEqual(actual["tool_transform"], initial["tool_transform"])
            self.assertTrue(actual["feedback_id"].startswith(simulation._trial_id+"@"))
            self.assertAlmostEqual(actual["timestamp_s"], 1.310)
            self.assertFalse(actual["evidence_passed"])
            self.assertNotIn("object_pose", actual)
            self.assertNotIn("contact", actual)

    # BEGIN raw dynamics and clipped command evidence remain independently timed.
    def test_native_dynamics_and_requested_applied_torque_are_recorded_without_differencing(self):
        simulation, curve, value = _fixture()
        initial = _initial_feedback(simulation)
        expected_acceleration = [simulation.native_data.qacc.copy()]
        expected_force = [simulation.native_data.actuator_force.copy()]
        advance = simulation._sim.step

        def step():
            advance()
            # q/dq deliberately stay constant while the mock exposes distinct
            # native telemetry. Differentiating dq or copying ctrl would fail.
            simulation.native_data.qacc += .25
            simulation.native_data.actuator_force += 2.
            expected_acceleration.append(simulation.native_data.qacc.copy())
            expected_force.append(simulation.native_data.actuator_force.copy())

        simulation._sim.step = step
        simulation._arm_controller.run_controller = lambda: np.full(7, 5.*len(simulation.arm_goals))
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            report = _executor(simulation).execute(_result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
                initial_feedback=initial, execution_limits=_limits(simulation, curve),
                output=output, max_physics_steps=4)
            with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["actual_qacc"], expected_acceleration[:5])
                np.testing.assert_array_equal(saved["actual_actuator_force"], expected_force[:5])
                self.assertEqual(len(saved["elapsed"]), 5)
                self.assertEqual(report["dependency_hold"]["physical_steps"], 1)
                np.testing.assert_array_equal(saved["actuator_names"], simulation.actuator_names)
                np.testing.assert_array_equal(saved["arm_actuator_names"], simulation.actuator_names[:7])
                np.testing.assert_array_equal(saved["requested_arm_torque"][:, 0], [5., 10., 15., 20.])
                np.testing.assert_array_equal(saved["applied_arm_torque"][:, 0], [5., 10., 10., 10.])
                np.testing.assert_array_equal(saved["applied_arm_torque"], saved["actuator_ctrl"][:, :7])
                np.testing.assert_array_equal(saved["arm_torque_saturated"][:, 0], [False, False, True, True])
                np.testing.assert_allclose(np.diff(saved["control_elapsed"]), simulation.timestep, atol=1e-15)
            self.assertEqual(report["arm_control_saturated_coordinates"], 14)
            final = json.loads(Path(report["barrier_feedback_path"]).read_text())
            np.testing.assert_array_equal(final["qacc"], expected_acceleration[-1])
            np.testing.assert_array_equal(final["actuator_force"], expected_force[-1])
    # END telemetry does not claim physical dynamics from the mocked clock.

    # BEGIN device quantization must preserve distinct original actions.
    def test_distinct_commands_cannot_disappear_in_one_activation_tick(self):
        simulation, curve, value = _fixture()
        curve.events[0]["time"], curve.events[1]["time"] = .0005, .0015
        with self.assertRaisesRegex(ValueError, "erases a required command sequence"):
            _bind_fingers(simulation, curve, value)
        self.assertEqual(simulation.sim_time, 1.3)
        self.assertEqual(simulation.arm_goals, [])
    # END identical command merging is not distinct open/close execution.


# BEGIN physical feedback rejection and isolated native FK regressions.
class DependencyExecutionTests(unittest.TestCase):
    """Effects are observed facts, separate from command playback and waits."""

    # BEGIN scene observations can run more slowly than actual robot guards.
    def test_slower_fresh_observations_keep_acquisition_time_and_control_frequency(self):
        simulation, curve, targets = _fixture()
        curve.events[2].update(kind="effect", requires_evidence=None, stop=False)
        _bind_fingers(simulation, curve, targets)
        limits = _limit_values(simulation, curve)
        limits["observation_policy"].update(period_s=.003, maximum_age_s=.005,
                                              required_fields=["binding_valid"])
        calls = []

        def observe(feedback, curve_time):
            calls.append(curve_time)
            return {"source": "Synthetic slower camera", "timestamp_s": feedback["timestamp_s"]-.001,
                    "binding_valid": True}

        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _executor(simulation).execute(_result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=ExecutionLimits.from_mapping(limits), output=Path(temporary)/"trial",
                observation_callback=observe)
        np.testing.assert_allclose(calls, [0., .004, .006, .010, .012])
        self.assertEqual(report["geometry_check_counts"]["after_step"], 6)
        self.assertAlmostEqual(report["observation_checks"][0]["sample"]["timestamp_s"], 1.299)
        self.assertAlmostEqual(report["observation_checks"][0]["age_s"], .001)

    def test_required_channel_cannot_be_satisfied_by_a_none_placeholder(self):
        for field in ("binding_valid", "scene_owner_poses", "scene_condition_evidence"):
            with self.subTest(field=field):
                simulation, curve, targets = _fixture()
                limits = _limit_values(simulation, curve)
                limits["observation_policy"]["required_fields"] = [field]

                def observe(feedback, curve_time):
                    return {"source": "Synthetic malformed channel", "timestamp_s": feedback["timestamp_s"],
                            field: None}

                with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
                    output = Path(temporary)/"trial"
                    with self.assertRaisesRegex(ValueError, "explicit valid value"):
                        _executor(simulation).execute(_result(curve),
                            finger_targets=FixedFingerTargets.from_mapping(targets),
                            initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(limits),
                            output=output, observation_callback=observe)
                    report = json.loads((output/"execution-report.json").read_text())
                self.assertEqual(report["executed_steps"], 0)
                self.assertTrue(report["abort_hold"]["native_ctrl_written"])

    def test_expired_required_observation_rejects_without_relabeling_its_timestamp(self):
        simulation, curve, targets = _fixture()
        limits = _limit_values(simulation, curve)
        limits["observation_policy"].update(period_s=.004, maximum_age_s=.005,
                                              required_fields=["binding_valid"])

        def observe(feedback, curve_time):
            return ({"source": "Synthetic dropped camera stream", "timestamp_s": feedback["timestamp_s"],
                     "binding_valid": True} if curve_time == 0 else None)

        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                _executor(simulation).execute(_result(curve),
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=ExecutionLimits.from_mapping(limits), output=output, observation_callback=observe)
            report = json.loads((output/"execution-report.json").read_text())
        self.assertEqual(report["feedback_failure"]["point"], "observation_freshness")
        self.assertEqual(report["executed_steps"], 3)
        self.assertEqual(report["feedback_failure"]["violations"][0]["last_observation"]["timestamp_s"], 1.3)
        self.assertTrue(report["abort_hold"]["native_ctrl_written"])

    def test_expired_optional_effect_becomes_unknown_without_a_new_barrier(self):
        simulation, curve, targets = _fixture()
        curve.events[2].update(kind="effect", requires_evidence=None, stop=False)
        curve.conditions = ({"condition_id": "Holding", "effect_requirement": "observed holding",
                             "skill_id": "declared_skill", "event_id": "holding-original"},)
        _bind_fingers(simulation, curve, targets)
        limits = _limit_values(simulation, curve)
        limits["observation_policy"].update(period_s=.004, maximum_age_s=.005)

        def observe(feedback, curve_time):
            return ({"source": "Synthetic initial effect", "timestamp_s": feedback["timestamp_s"],
                     "effects": {"Holding": {"value": "T", "source": "Synthetic predicate"}}}
                    if curve_time == 0 else None)

        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _executor(simulation).execute(_result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=ExecutionLimits.from_mapping(limits), output=Path(temporary)/"trial",
                observation_callback=observe)
        self.assertEqual(report["status"], "curve_finished")
        self.assertEqual(report["evaluated_events"][0]["evaluation"], "U")
        self.assertEqual(report["effect_result"], "U")
        self.assertIsNone(report["dependency_hold"])
    # END missing future evidence does not erase successful robot supervision.

    def test_true_current_prerequisite_extends_only_the_same_bound_curve(self):
        simulation, curve, targets = _fixture()
        curve.conditions = ({"condition_id": "entry_attachment", "kind": "observed_attachment"},)

        def observe(feedback, curve_time):
            return {"source": "explicit synthetic effect observation", "timestamp_s": feedback["timestamp_s"],
                    "effects": {"Holding": {"value": "T", "source": "synthetic independent predicate"},
                                "unrelated_observation": {"value": "U", "source": "not a required effect"}}}

        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _executor(simulation).execute(_result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=_limits(simulation, curve), output=Path(temporary)/"trial", observation_callback=observe)
        self.assertEqual(report["status"], "curve_finished")
        self.assertEqual(report["executed_steps"], 6)
        self.assertEqual(report["effect_result"], "T")
        self.assertIsNone(report["next_verification"])
        self.assertIsNone(report["dependency_hold"])
        self.assertEqual(report["extended_preflights"][0]["until"], curve.duration)
        self.assertIn("hold-later", [event["event_id"] for event in report["events"]])

    def test_false_prerequisite_physically_holds_without_sending_later_command(self):
        simulation, curve, targets = _fixture()

        def observe(feedback, curve_time):
            return {"source": "synthetic false effect", "timestamp_s": feedback["timestamp_s"],
                    "effects": {"Holding": {"value": "F", "source": "independent synthetic result"}}}

        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _executor(simulation).execute(_result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=_limits(simulation, curve), output=Path(temporary)/"trial", observation_callback=observe)
        self.assertEqual(report["status"], "effect_rejected")
        self.assertEqual(report["effect_result"], "F")
        self.assertEqual(report["executed_steps"], 4)
        self.assertTrue(report["dependency_hold"]["native_ctrl_written"])
        self.assertNotIn("hold-later", [event["event_id"] for event in report["events"]])
        self.assertEqual(report["continuation"]["consumed_physics_steps"], 5)

    def test_unknown_effect_without_dependency_does_not_create_a_barrier(self):
        simulation, curve, targets = _fixture()
        curve.events[2].update(kind="effect", requires_evidence=None, stop=False)
        curve.conditions = ({"condition_id": "actual_holding", "effect_requirement": "observed stable grasp",
            "skill_id": "declared_skill", "event_id": "holding-original"},)
        _bind_fingers(simulation, curve, targets)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _executor(simulation).execute(_result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=_limits(simulation, curve), output=Path(temporary)/"trial")
        self.assertEqual(report["status"], "curve_finished")
        self.assertEqual(report["effect_result"], "U")
        self.assertEqual(report["executed_steps"], 6)
        self.assertIsNone(report["dependency_hold"])
        self.assertEqual(report["evaluated_events"][0]["evaluation"], "U")
        self.assertEqual(report["continuation"]["evaluated_events"], report["evaluated_events"])
        self.assertNotIn("holding-original", [event["event_id"] for event in report["unconsumed_events"]])

    # BEGIN stop-only obligations use measured velocity even without evidence gates.
    def test_explicit_stop_without_dependency_checks_actual_velocity(self):
        simulation, curve, targets = _fixture()
        curve.events[2].update(kind="function", requires_evidence=None, stop=True, time=.008)
        _bind_fingers(simulation, curve, targets)
        limits = _limit_values(simulation, curve)
        limits["stop_arm_velocity"] = [.001]*7
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                _executor(simulation).execute(_result(curve),
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=ExecutionLimits.from_mapping(limits), output=output, max_hold_steps=2)
            report = json.loads((output/"execution-report.json").read_text())
        self.assertEqual(report["executed_steps"], 4)
        self.assertEqual(report["feedback_failure"]["point"], "event_gate")
        self.assertEqual(report["stopped_events"], [])
        self.assertTrue(report["abort_hold"]["native_ctrl_written"])
        self.assertNotIn("hold-later", [event["event_id"] for event in report["events"]])

    def test_passed_explicit_stop_is_consumed_without_adding_a_barrier(self):
        simulation, curve, targets = _fixture()
        curve.events[2].update(kind="function", requires_evidence=None, stop=True, time=.008)
        _bind_fingers(simulation, curve, targets)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _executor(simulation).execute(_result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=_limits(simulation, curve), output=Path(temporary)/"trial")
        self.assertEqual(report["status"], "curve_finished")
        self.assertEqual(report["stopped_events"][0]["event_id"], "holding-original")
        self.assertEqual(report["continuation"]["stopped_events"], report["stopped_events"])
        self.assertNotIn("holding-original", [event["event_id"] for event in report["unconsumed_events"]])
    # END a planned zero derivative alone never certifies an actual stop.

    # BEGIN continuation preserves work bounds and separates events from goals.
    def test_function_node_records_occurrence_without_claiming_goal_completion(self):
        simulation, curve, targets = _fixture()
        curve.events[2].update(kind="function", requires_evidence=None, stop=False)
        _bind_fingers(simulation, curve, targets)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _executor(simulation).execute(_result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=_limits(simulation, curve), output=Path(temporary)/"trial")
        self.assertEqual(report["reached_events"][0]["event_id"], "holding-original")
        self.assertEqual(report["continuation"]["reached_events"], report["reached_events"])
        self.assertEqual(report["completed_goal_ids"], [])
        self.assertEqual(report["goal_evidence"], [])

    def test_resumed_candidate_cannot_receive_a_fresh_default_step_budget(self):
        simulation, curve, targets = _fixture()
        result = _result(curve)
        result.execution_binding.update(physics_budget=5, consumed_physics_steps=3, remaining_physics_steps=2)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            with self.assertRaisesRegex(ValueError, "physics-step budget"):
                _executor(simulation).execute(result,
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), output=Path(temporary)/"trial")
        self.assertEqual(simulation.arm_goals, [])

    def test_hold_cost_is_cumulative_even_when_the_task_step_budget_is_spent(self):
        simulation, curve, targets = _fixture()
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _executor(simulation).execute(_result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=_limits(simulation, curve), output=Path(temporary)/"trial", max_physics_steps=4)
        self.assertTrue(report["dependency_hold"]["native_ctrl_written"])
        self.assertEqual(report["continuation"]["physics_budget"], 4)
        self.assertEqual(report["continuation"]["cumulative_physics_steps"], 5)
        self.assertEqual(report["continuation"]["remaining_physics_steps"], 0)

    def test_remaining_device_inputs_keep_last_applied_interval_and_original_targets(self):
        from manipulation.execution import continuation_execution_inputs
        simulation, curve, targets = _fixture()
        limits = _limits(simulation, curve)
        fixed = FixedFingerTargets.from_mapping(targets)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _executor(simulation).execute(_result(curve), finger_targets=fixed,
                initial_feedback=_initial_feedback(simulation), execution_limits=limits,
                output=Path(temporary)/"trial")
        remaining_targets, remaining_limits = continuation_execution_inputs(report["continuation"], fixed, limits)
        self.assertEqual([row["event_id"] for row in remaining_targets.events], ["hold-later"])
        self.assertEqual([row["event_id"] for row in remaining_limits.events], ["hold-later"])
        last = next(row for row in limits.events if row["event_id"] == "close-original")
        self.assertEqual(remaining_limits.initial_fingers,
                         {key: last[key] for key in ("position_bounds", "maximum_speed")})
        changed = deepcopy(targets)
        changed["events"][-1]["positions"][0] += .001
        with self.assertRaisesRegex(ValueError, "different actuator target"):
            continuation_execution_inputs(report["continuation"], FixedFingerTargets.from_mapping(changed), limits)
    # END emergency hold is bounded separately and never resets the task budget.

    def test_hold_budget_cannot_turn_nonzero_feedback_into_a_confirmed_stop(self):
        simulation, _, _ = _fixture()
        initial_q = simulation.native_data.qpos.copy()
        initial_dq = simulation.native_data.qvel.copy()
        with patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = simulation.abort_hold(simulation.proprioception(), simulation.finger_control,
                expected_sim_time=simulation.sim_time, stop_velocity=np.full(9, .001), max_steps=2)
        self.assertTrue(report["native_ctrl_written"])
        self.assertEqual(report["physical_steps"], 2)
        self.assertFalse(report["physical_stop_confirmed"])
        np.testing.assert_array_equal(simulation.native_data.qpos, initial_q)
        np.testing.assert_array_equal(simulation.native_data.qvel, initial_dq)


class ExecutionGuardTests(unittest.TestCase):
    # BEGIN reject malformed telemetry without adding physical spike thresholds.
    def test_native_dynamics_require_finite_values_and_exact_actuator_order(self):
        backend, curve, _ = _fixture()
        executor = _executor(backend)
        limits = _limits(backend, curve)
        binding = limits.bind(curve, joint_names=backend.joint_names, arm_joint_names=backend.arm_joint_names,
            finger_joint_names=backend.finger_joint_names, native_joint_ranges=backend.joint_ranges,
            named_velocity_limits=backend.velocity_limits)
        q, dq, _ = curve.sample(0.)
        for field in ("actuator_names", "qacc", "actuator_force"):
            with self.subTest(field=field):
                feedback = backend.proprioception()
                if field == "actuator_names":
                    feedback[field] = feedback[field][::-1]
                else:
                    feedback[field][2] = np.nan
                failures = executor._execution_violations(feedback, q, dq, np.eye(4),
                    limits=limits, binding=binding, finger_state=limits.initial_fingers,
                    expected_sim_time=backend.sim_time)
                expected = "feedback_identity" if field == "actuator_names" else "nonfinite_feedback"
                self.assertTrue(any(row["kind"] == expected and row["name"].startswith(field)
                                    for row in failures))
                self.assertEqual(backend.sim_time, 1.3)
    # END no actuator output is issued by telemetry validation.

    def test_non_grid_runtime_interval_uses_actual_feedback_instead_of_a_planning_sweep(self):
        simulation, curve, value = _fixture()
        self.assertEqual(curve.events[1]["time"], .003)
        self.assertEqual(simulation._dt, .002)
        policy = _limit_values(simulation, curve)
        # These are physical feedback guard limits. They may tighten at an
        # event when the actual sampled state satisfies them; they no longer
        # construct a planning geometry envelope or imply servo convergence.
        policy["events"][1]["position_bounds"] = [[.011, .019], [-.019, -.015]]
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            report = _executor(simulation).execute(_result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
                initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(policy),
                output=output)
            self.assertEqual(report["executed_steps"], 4)
            self.assertFalse(report["finger_reference_actual_tracking_certified"])
            with np.load(output/"physical-trajectory.npz") as saved:
                np.testing.assert_array_equal(saved["actual_finger_position_bounds"][2], policy["initial_fingers"]["position_bounds"])
                np.testing.assert_array_equal(saved["actual_finger_position_bounds"][3], policy["events"][1]["position_bounds"])
                np.testing.assert_array_equal(saved["actual_q"][:, 7:], np.tile([.012, -.017], (5, 1)))

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
                _executor(simulation).execute(_result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
                    initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(policy),
                    output=output)
            self.assertFalse(output.exists())
            self.assertEqual(simulation.native_data.time, 1.3)

    # BEGIN quantized commands gate the same actual-time arm query.
    def test_close_gate_uses_activation_clock_and_preserves_intent_time(self):
        simulation, curve, value = _fixture()
        curve.events[0]["time"] = 0.
        _bind_fingers(simulation, curve, value)
        limits = _limit_values(simulation, curve)
        limits["arm_position_error"] = [1e-5]*7
        initial = _initial_feedback(simulation)

        def step():
            simulation.native_data.time += simulation._dt
            elapsed = simulation.native_data.time-initial["sim_time"]
            q, dq, _ = curve.sample(min(elapsed, curve.executable_until))
            simulation.native_data.qpos[:7], simulation.native_data.qvel[:7] = q, dq

        simulation._sim.step = step
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _executor(simulation).execute(_result(curve),
                finger_targets=FixedFingerTargets.from_mapping(value), initial_feedback=initial,
                execution_limits=ExecutionLimits.from_mapping(limits), output=Path(temporary)/"trial")
        close = next(row for row in report["events"] if row["event_id"] == "close-original")
        gate = next(row for row in report["event_gate_checks"] if row["event_id"] == "close-original")
        self.assertEqual(close["time"], .003)
        self.assertEqual(close["applied_time_s"], .004)
        self.assertEqual(gate["curve_time_s"], .004)
        self.assertIsNone(report["feedback_failure"])
        self.assertEqual(report["effect_result"], "U")
    # END intent evidence cannot select a second robot/geometry clock.

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
                    _executor(simulation).execute(_result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
                        initial_feedback=initial, execution_limits=_limits(simulation, curve),
                        output=output,
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
                np.testing.assert_array_equal(simulation.native_data.ctrl[7:], curve.finger_trajectory.sample(.002)[0])
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
                _executor(simulation).execute(_result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
                    initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(limits),
                    output=output)
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

    def test_rejected_preflight_keeps_candidate_unapplied_and_records_physical_hold(self):
        simulation, curve, targets = _fixture()
        checker = _GeometryChecker(simulation.joint_names, preflight_passed=False)
        with TemporaryDirectory() as temporary:
            output = Path(temporary)/"trial"
            with self.assertRaisesRegex(ExecutionGuardError, "geometry preflight"):
                TrajectoryExecutor(simulation, geometry_checker=checker).execute(_result(curve),
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), output=output)
            report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["feedback_failure"]["point"], "geometry_preflight")
            self.assertEqual(report["executed_steps"], 0)
            self.assertFalse(report["geometry_preflight"]["passed"])
            self.assertEqual(report["events"], [])
            self.assertEqual(checker.states, [])
            self.assertTrue(report["abort_hold"]["native_ctrl_written"])
            self.assertAlmostEqual(simulation.native_data.time, 1.302)
            np.testing.assert_array_equal(simulation.native_data.ctrl[7:], [.026, -.023])
            with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["actuator_ctrl"].shape, (0, 9))
                self.assertEqual(saved["actual_q"].shape, (1, 9))
                self.assertFalse(saved["geometry_checked"][0])
                self.assertTrue(np.isnan(saved["geometry_distances_m"][0]).all())

    def test_geometry_rejects_original_close_before_goal_change_and_receives_actual_fingers(self):
        simulation, curve, targets = _fixture()
        checker = _GeometryChecker(simulation.joint_names, rejected_stage="command_event", rejected_time=.004)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                TrajectoryExecutor(simulation, geometry_checker=checker).execute(_result(curve),
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), output=output)
            report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["executed_steps"], 2)
            self.assertEqual(report["feedback_failure"]["event_id"], "close-original")
            self.assertEqual(report["feedback_failure"]["curve_time_s"], .004)
            self.assertEqual([row["event_id"] for row in report["events"]], ["open-original"])
            self.assertEqual(report["geometry_check_counts"]["command_event"], 2)
            self.assertEqual(report["event_gate_checks"][-1]["geometry_check"]["distances_m"], [-.01])
            final_check = checker.states[-1]
            self.assertEqual(final_check["stage"], "command_event")
            np.testing.assert_array_equal(final_check["feedback"]["q"][7:], [.012, -.017])
            np.testing.assert_array_equal(final_check["feedback"]["actuator_ctrl"]["values"][7:], curve.finger_trajectory.sample(.004)[0])
            np.testing.assert_array_equal(simulation.native_data.ctrl[7:], curve.finger_trajectory.sample(.004)[0])
            self.assertTrue(report["abort_hold"]["physical_stop_confirmed"])

    def test_post_step_geometry_rejection_retains_the_completed_sample_and_pair_distances(self):
        simulation, curve, targets = _fixture()
        checker = _GeometryChecker(simulation.joint_names, rejected_stage="after_step")
        observed = []
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                TrajectoryExecutor(simulation, geometry_checker=checker).execute(_result(curve),
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), output=output,
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
                TrajectoryExecutor(simulation, geometry_checker=checker).execute(_result(curve),
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), output=output)
            report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["executed_steps"], 0)
            self.assertEqual(report["feedback_failure"]["violations"][0]["kind"], "geometry_checker_error")
            with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
                self.assertFalse(saved["geometry_checked"][0])
                self.assertFalse(saved["geometry_passed"][0])
                self.assertTrue(np.isnan(saved["geometry_distances_m"]).all())

    def test_formal_result_identity_and_exact_entrance_are_preserved(self):
        simulation, curve, targets = _fixture()
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            report = _executor(simulation).execute(_result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=_limits(simulation, curve), output=output,
                start_position_tolerance=0., start_velocity_tolerance=0.)
            self.assertTrue(report["planner_accepted"])
            self.assertEqual(report["reference_origin"], "bound_planning_result")
            self.assertEqual(report["executed_steps"], 4)
            self.assertTrue(report["geometry_preflight"]["passed"])
            self.assertEqual(report["geometry_check_counts"],
                             {"initial_state": 1, "before_step": 4, "command_event": 2, "after_step": 4,
                              "event_gate": 0})

    def test_initial_frame_or_tcp_relabelling_fails_before_creating_output(self):
        for field, wrong in (("reference_frame", "another_frame"), ("tool_link", "another_site"),
                             ("tool_transform", np.diag([1., 1., 1., 2.]))):
            simulation, curve, targets = _fixture()
            initial = _initial_feedback(simulation)
            initial[field] = wrong
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                output = Path(temporary)/"trial"
                with self.assertRaises(ValueError):
                    _executor(simulation).execute(_result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                        initial_feedback=initial, execution_limits=_limits(simulation, curve),
                        output=output)
                self.assertFalse(output.exists())
                self.assertEqual(simulation.native_data.time, 1.3)
# END mandatory production geometry gates and explicit diagnostic references.


# BEGIN concrete native backend binding, independent of recorded datasets.
class RobosuiteBackendTests(unittest.TestCase):
    # BEGIN telemetry cannot confuse native actuator columns with joint columns.
    def test_native_dynamics_use_named_indices_and_independent_copies(self):
        backend, _, _ = _fixture()
        backend._qvel_ids = np.array([6, 5, 4, 3, 2, 1, 0, 8, 7])
        backend._actuator_ids = np.array([3, 4, 5, 6, 0, 1, 2, 8, 7])
        acceleration = backend.native_data.qacc[backend._qvel_ids].copy()
        force = backend.native_data.actuator_force[backend._actuator_ids].copy()
        feedback = backend.proprioception()
        self.assertEqual(feedback["actuator_names"], backend.actuator_names)
        np.testing.assert_array_equal(feedback["qacc"], acceleration)
        np.testing.assert_array_equal(feedback["actuator_force"], force)
        backend.native_data.qacc[:] = 100.
        backend.native_data.actuator_force[:] = -100.
        np.testing.assert_array_equal(feedback["qacc"], acceleration)
        np.testing.assert_array_equal(feedback["actuator_force"], force)
        self.assertEqual(backend.sim_time, 1.3)
    # END feedback sampling neither aliases native arrays nor advances physics.

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
