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
# BEGIN archive admission precedes JSON allocation and formal result loading.
from recorded_simulation import load_curve_archive, load_saved_trajectory
# END both inspection and execution share the same bounded first read.
# END exercise production execution through the thin recording adapter.


def _event(event_id, kind, time, evidence=None):
    return {"skill_id": "declared_skill", "event_id": event_id, "occurrence": 0, "kind": kind,
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
    # BEGIN this mock clock uses the same explicit native-attempt accounting.
    simulation._step_attempts = 0
    simulation._last_step_accounting = None
    # END no fixture count is inferred from successful telemetry capture.
    simulation._trial_id = str(uuid4())
    simulation._feedback_source = "explicit mocked robot feedback; no physical integration"
    simulation._description = {"scope": "mocked integration clock only"}
    # BEGIN the sequencing fixture declares the same complete controller contract.
    simulation._description["controller_policy"] = {
        "mode": "joint_trajectory_q_dq_ddq", "arm_joint_names": list(simulation.arm_joint_names),
        "kp": [1.]*7, "kd": [1.]*7, "native_timestep_s": simulation._dt,
        "reference_derivatives": ["q", "dq", "ddq"], "sampling_phase": "clamped_step_endpoint",
        "acceleration_knot_side": "right_internal_last_left"}
    # END this declared toy policy is not a physical robot model.
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
    # BEGIN static calibration and an explicit scripted estimate replace live site GT.
    simulation._calibrated_base_pose = np.eye(4)
    simulation.scripted_tcp_estimate_offset = np.zeros(3)
    native_proprioception = simulation.proprioception

    def scripted_proprioception():
        feedback = native_proprioception()
        # Only contract tests inject this declared estimate error; native body
        # and site arrays never provide the mock's online TCP pose.
        feedback["tool_pose"][:3, 3] += simulation.scripted_tcp_estimate_offset
        return feedback

    simulation.proprioception = scripted_proprioception
    # END the fixture's optional estimate error is not physical sensor evidence.
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
    # BEGIN model only the controller arithmetic, independently of mock integration.
    controller = SimpleNamespace(mass_matrix=np.eye(7), kp=np.ones(7), kd=np.ones(7),
        joint_pos=simulation.native_data.qpos[:7].copy(), joint_vel=simulation.native_data.qvel[:7].copy(),
        torque_compensation=np.zeros(7))

    def set_goal(goal):
        simulation.arm_goals.append(goal.copy())
        controller.joint_pos = simulation.native_data.qpos[:7].copy()
        controller.joint_vel = simulation.native_data.qvel[:7].copy()

    controller.set_goal = set_goal
    controller.run_controller = lambda: simulation.arm_goals[-1]-controller.joint_pos-controller.joint_vel
    simulation._arm_controller = controller
    # END real backend feedforward consumes explicit current matrices and named state.
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

    # BEGIN a frozen scene binding is a forbidden execution-side operation.
    def bind_scene(self, *args, **kwargs):
        raise AssertionError("Physical execution must never bind frozen world geometry")
    # END every sequencing path shares this fail-fast sentinel.

    # BEGIN actual preflight receives the current execution-local permission set.
    def preflight(self, trajectory, initial_feedback, execution_limits, *, start_time=0., until=None,
                  current_time=None):
        self.preflight_inputs.append(deepcopy(initial_feedback))
        return {"passed": self.preflight_passed,
                "violations": [] if self.preflight_passed else [
                    {"kind": "scripted_preflight_rejection", "name": "test_reference", "measured": "rejected"}]}
    # END nominal planner assumptions do not become an implicit actual latch.

    # BEGIN scripted geometry retains the explicit actual contact query mode.
    def check_state(self, feedback, *, stage, trajectory_time):
        self.states.append({"feedback": deepcopy(feedback), "stage": stage, "trajectory_time": trajectory_time})
        # END the double never derives permissions from a command or nominal clock.
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
        "timing_events": [deepcopy(event) for event in curve.events if event.get("timing")], "timing_checks": [],
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


# BEGIN sequencing tests explicitly prepare their new synthetic execution binding.
def _execute_bound(executor, result, *, result_only_goal_ids=(), **kwargs):
    """Bind fresh fixture evidence, then call the production executor unchanged.

    Existing device/continuation policies pass through untouched so rejection
    tests cannot silently migrate them. Tests of source-only authorization use
    TrajectoryExecutor.execute directly instead of this fixture setup helper.
    """
    from manipulation.execution_binding import rebind_execution_result

    if (isinstance(result, PlanningResult) and result.success is True
            and isinstance(result.execution_binding, dict)
            and "device_policy" not in result.execution_binding):
        rebound = rebind_execution_result(result,
            controller_policy=executor.backend.describe()["controller_policy"],
            result_only_goal_ids=list(result_only_goal_ids), finger_targets=kwargs["finger_targets"],
            execution_limits=kwargs["execution_limits"], source="Explicit CPU sequencing fixture device binding")
        result.execution_binding = rebound.execution_binding
    return executor.execute(result, **kwargs)
# END fixture setup does not weaken the production executor's mandatory rebind gate.


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
                    _execute_bound(_executor(simulation), result,
                        finger_targets=FixedFingerTargets.from_mapping(targets),
                        initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                        output=Path(temporary)/"trial")
                self.assertEqual(simulation.sim_time, 1.3)
                self.assertEqual(simulation.arm_goals, [])
        simulation, curve, targets = _fixture()
        with TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "formal PlanningResult"):
            _execute_bound(_executor(simulation), curve, finger_targets=FixedFingerTargets.from_mapping(targets),
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
                    _execute_bound(_executor(simulation), result, finger_targets=FixedFingerTargets.from_mapping(targets),
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
            # BEGIN classification is returned from the single validated decode.
            with patch("recorded_simulation.json.loads", wraps=json.loads) as decode:
                loaded = load_saved_trajectory(path)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(loaded.diagnostics["archive_classification"], "planner_accepted")
            self.assertEqual(loaded.execution_binding["execution_limits"], limits.describe())
            # END the existing altered-limit and rejected-label cases still fail.
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
                with self.assertRaisesRegex(ValueError, "physical contract|execution limits|every physical limit"):
                    load_saved_trajectory(path)
            save_curve(path, curve, classification="rejected_diagnostic_not_executable", execution_limits=limits.describe())
            with self.assertRaisesRegex(ValueError, "not executable"):
                load_saved_trajectory(path)

    # BEGIN oversized archives must fail before text, JSON or NPZ allocation.
    def test_oversized_archive_is_rejected_before_read_or_decode(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"oversized.npz"
            # A sparse metadata file exercises the real filesystem size check
            # without allocating a 512-MiB string or requiring an NPZ payload.
            with path.with_suffix(".json").open("wb") as stream:
                stream.truncate(512*1024**2+1)
            with patch("recorded_simulation.Path.read_text") as read, \
                    patch("recorded_simulation.json.loads") as decode, \
                    patch("recorded_simulation.np.load") as arrays:
                with self.assertRaisesRegex(ValueError, "512 MiB archive"):
                    load_curve_archive(path)
            read.assert_not_called()
            decode.assert_not_called()
            arrays.assert_not_called()
    # END ordinary metadata bounds and physical limits are not relaxed.

    # BEGIN reject an oversized write before it leaves a partial NPZ/JSON pair.
    def test_curve_writer_checks_shared_archive_bound_before_writing(self):
        from recorded_task_planning import save_curve

        simulation, curve, _ = _fixture()
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"oversized.npz"
            with patch("recorded_task_planning.MAX_CURVE_METADATA_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "archive size"):
                    save_curve(path, curve, classification="rejected_diagnostic_not_executable",
                               execution_limits=_limits(simulation, curve).describe())
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".json").exists())
    # END the test uses small inputs while exercising the real writer boundary.

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
                load_saved_trajectory(path)
            metadata_path.write_text(json.dumps(metadata))
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            arrays["finger_velocity"][0, 0] += .001
            np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, "same original clock and polynomial"):
                load_saved_trajectory(path)
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
                    _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
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
            report = _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
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
            report = _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
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
                        _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
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
            report = _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
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
            # BEGIN the four-step total leaves no separate hold control/step.
            np.testing.assert_allclose(simulation.arm_goals[-1], curve.sample(.0075)[0], atol=0., rtol=0.)
            self.assertFalse(report["dependency_hold"]["native_ctrl_written"])
            self.assertEqual(report["dependency_hold"]["physical_steps"], 0)
            # END the command at the barrier remains the last applied goal.
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
            self.assertAlmostEqual(actual["timestamp_s"], 1.308)
            self.assertFalse(actual["evidence_passed"])
            self.assertNotIn("object_pose", actual)
            self.assertNotIn("contact", actual)

    # BEGIN privileged dynamics do not enter online state or command archives.
    def test_online_execution_ignores_native_dynamics_and_keeps_command_torque_evidence(self):
        simulation, curve, value = _fixture()
        simulation.native_data.qacc[:] = np.nan
        simulation.native_data.actuator_force[:] = np.inf
        initial = _initial_feedback(simulation)
        simulation._arm_controller.run_controller = lambda: np.full(7, 5.*len(simulation.arm_goals))
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            report = _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
                initial_feedback=initial, execution_limits=_limits(simulation, curve),
                output=output, max_physics_steps=5)
            with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
                self.assertNotIn("actual_qacc", saved)
                self.assertNotIn("actual_actuator_force", saved)
                self.assertTrue(np.all(saved["tool_pose_source"] == "forward_kinematics_from_measured_arm_q"))
                self.assertFalse(saved["tool_pose_is_independent_observation"].any())
                self.assertEqual(len(saved["elapsed"]), 5)
                self.assertEqual(report["dependency_hold"]["physical_steps"], 1)
                np.testing.assert_array_equal(saved["actuator_names"], simulation.actuator_names)
                np.testing.assert_array_equal(saved["arm_actuator_names"], simulation.actuator_names[:7])
                # BEGIN the issued demand also contains the explicit 0.01-rad/s reference term.
                np.testing.assert_allclose(saved["requested_arm_torque"][:, 0], [5.01, 10.01, 15.01, 20.01], atol=1e-9, rtol=0.)
                np.testing.assert_allclose(saved["applied_arm_torque"][:, 0], [5.01, 10., 10., 10.], atol=1e-9, rtol=0.)
                np.testing.assert_array_equal(saved["applied_arm_torque"], saved["actuator_ctrl"][:, :7])
                np.testing.assert_array_equal(saved["arm_torque_saturated"][:, 0], [False, True, True, True])
                # END requested/applied controls remain distinct from measured force.
                np.testing.assert_allclose(np.diff(saved["control_elapsed"]), simulation.timestep, atol=1e-15)
            self.assertEqual(report["arm_control_saturated_coordinates"], 21)
            final = json.loads(Path(report["barrier_feedback_path"]).read_text())
            self.assertNotIn("qacc", final)
            self.assertNotIn("actuator_force", final)
            self.assertFalse(final["tool_pose_is_independent_observation"])
    # END independent evaluation owns privileged dynamic evidence separately.

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
            report = _execute_bound(_executor(simulation), _result(curve),
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
                        _execute_bound(_executor(simulation), _result(curve),
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
                _execute_bound(_executor(simulation), _result(curve),
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
            report = _execute_bound(_executor(simulation), _result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=ExecutionLimits.from_mapping(limits), output=Path(temporary)/"trial",
                observation_callback=observe)
        self.assertEqual(report["status"], "curve_finished")
        self.assertEqual(report["evaluated_events"][0]["evaluation"], "U")
        self.assertEqual(report["effect_result"], "U")
        self.assertIsNone(report["dependency_hold"])
    # END missing future evidence does not erase successful robot supervision.

    def test_true_current_prerequisite_requires_new_binding_before_suffix(self):
        simulation, curve, targets = _fixture()
        curve.conditions = ({"condition_id": "entry_attachment", "kind": "observed_attachment"},)

        def observe(feedback, curve_time):
            return {"source": "explicit synthetic effect observation", "timestamp_s": feedback["timestamp_s"],
                    "effects": {"Holding": {"value": "T", "source": "synthetic independent predicate"},
                                "unrelated_observation": {"value": "U", "source": "not a required effect"}}}

        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _execute_bound(_executor(simulation), _result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=_limits(simulation, curve), output=Path(temporary)/"trial", observation_callback=observe)
        self.assertEqual(report["status"], "needs_rebind")
        self.assertEqual(report["executed_steps"], 4)
        self.assertEqual(report["effect_result"], "T")
        self.assertIsNone(report["next_verification"])
        self.assertTrue(report["dependency_hold"]["prepared"])
        self.assertEqual(report["extended_preflights"], [])
        self.assertNotIn("hold-later", [event["event_id"] for event in report["events"]])

    def test_false_prerequisite_physically_holds_without_sending_later_command(self):
        simulation, curve, targets = _fixture()

        def observe(feedback, curve_time):
            return {"source": "synthetic false effect", "timestamp_s": feedback["timestamp_s"],
                    "effects": {"Holding": {"value": "F", "source": "independent synthetic result"}}}

        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _execute_bound(_executor(simulation), _result(curve),
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
            report = _execute_bound(_executor(simulation), _result(curve),
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
                _execute_bound(_executor(simulation), _result(curve),
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
            report = _execute_bound(_executor(simulation), _result(curve),
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
            report = _execute_bound(_executor(simulation), _result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=_limits(simulation, curve), output=Path(temporary)/"trial")
        self.assertEqual(report["reached_events"][0]["event_id"], "holding-original")
        self.assertEqual(report["continuation"]["reached_events"], report["reached_events"])
        self.assertEqual(report["completed_goal_ids"], [])
        self.assertEqual(report["goal_evidence"], [])

    def test_resumed_candidate_cannot_receive_a_fresh_default_step_budget(self):
        from manipulation.execution_binding import rebind_execution_result

        simulation, curve, targets = _fixture()
        # BEGIN model a legitimately bound candidate before any steps were consumed.
        result = rebind_execution_result(_result(curve),
            controller_policy=simulation.describe()["controller_policy"], result_only_goal_ids=[],
            finger_targets=FixedFingerTargets.from_mapping(targets), execution_limits=_limits(simulation, curve),
            source="Original controller policy before the synthetic continuation")
        # END the resumed budget remains the subject of this regression.
        result.execution_binding.update(physics_budget=5, consumed_physics_steps=3, remaining_physics_steps=2)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            with self.assertRaisesRegex(ValueError, "physics-step budget"):
                _execute_bound(_executor(simulation), result,
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), output=Path(temporary)/"trial")
        self.assertEqual(simulation.arm_goals, [])

    # BEGIN exhaustion never grants an extra native hold attempt.
    def test_normal_and_hold_share_one_total_step_budget(self):
        simulation, curve, targets = _fixture()
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _execute_bound(_executor(simulation), _result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=_limits(simulation, curve), output=Path(temporary)/"trial", max_physics_steps=4)
        self.assertFalse(report["dependency_hold"]["native_ctrl_written"])
        self.assertEqual(report["dependency_hold"]["charged_steps"], 0)
        self.assertEqual(report["dependency_hold"]["physical_steps"], 0)
        self.assertEqual(report["continuation"]["physics_budget"], 4)
        self.assertEqual(report["continuation"]["consumed_physics_steps"], 4)
        self.assertEqual(report["continuation"]["confirmed_physics_steps"], 4)
        self.assertTrue(report["continuation"]["step_accounting_exact"])
        self.assertEqual(report["continuation"]["cumulative_physics_steps"], 4)
        self.assertEqual(report["continuation"]["remaining_physics_steps"], 0)
        self.assertEqual(simulation._step_attempts, 4)
        self.assertEqual(len(simulation.arm_goals), 4)
    # END budget accounting does not change the original dependency outcome.

    # BEGIN stop work and failed captures retain the same finite allowance.
    def test_abort_hold_receives_only_the_one_remaining_attempt(self):
        simulation, curve, targets = _fixture()
        values = _limit_values(simulation, curve)
        values["stop_arm_velocity"] = [.0001]*7
        executor = _executor(simulation)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            with self.assertRaises(ExecutionGuardError):
                _execute_bound(executor, _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                    initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(values),
                    output=Path(temporary)/"trial", max_physics_steps=5, max_hold_steps=250)
        report = executor.last_execution_report
        self.assertEqual(report["executed_steps"], 4)
        self.assertEqual(report["abort_hold"]["charged_steps"], 1)
        self.assertFalse(report["abort_hold"]["physical_stop_confirmed"])
        self.assertEqual(report["physics_step_accounting"]["charged_steps"], 5)
        self.assertEqual(report["continuation"]["remaining_physics_steps"], 0)
        self.assertEqual(simulation._step_attempts, 5)

    def test_post_step_capture_errors_preserve_primary_and_native_accounting(self):
        for failure in ("forward", "record", "feedback", "clock"):
            with self.subTest(failure=failure):
                simulation, curve, targets = _fixture()
                initial = _initial_feedback(simulation)
                executor = _executor(simulation)
                primary = RuntimeError("primary "+failure)
                original_feedback = simulation.proprioception
                original_reference_pose = simulation.reference_tool_pose

                def feedback():
                    if simulation._step_attempts:
                        raise primary
                    return original_feedback()

                def forward(*args):
                    if failure == "forward" and simulation._step_attempts == 1:
                        raise primary

                def reference_pose(q):
                    # BEGIN isolate reference-record failure from measured-joint FK.
                    # This fixture keeps measured q fixed while the desired
                    # trajectory moves. Fresh feedback and the abort hold must
                    # retain measured-q FK; only recording desired-q FK fails.
                    measured_q = simulation.native_data.qpos[:len(simulation.arm_joint_names)]
                    if simulation._step_attempts and not np.array_equal(q, measured_q):
                        raise primary
                    return original_reference_pose(q)
                    # END the fault is selected by its input, never a call count.

                class UnreadableClock:
                    def __float__(self):
                        raise ValueError("native clock unavailable")

                def native_failure():
                    simulation.native_data.time = UnreadableClock()
                    raise primary

                if failure == "feedback":
                    simulation.proprioception = feedback
                elif failure == "record":
                    simulation.reference_tool_pose = reference_pose
                elif failure == "clock":
                    simulation._sim.step = native_failure
                with TemporaryDirectory() as temporary, patch(
                        "manipulation.robosuite_execution.mujoco.mj_forward", side_effect=forward):
                    output = Path(temporary)/"trial"
                    with self.assertRaises(RuntimeError) as caught:
                        _execute_bound(executor, _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                            initial_feedback=initial, execution_limits=_limits(simulation, curve),
                            output=output, max_physics_steps=4)
                    self.assertIs(caught.exception, primary)
                    retained = json.loads((output/"continuation.json").read_text())
                    report = json.loads((output/"execution-report.json").read_text())
                    self.assertEqual(report["normal_step_accounting"]["charged_steps"], 1)
                    self.assertEqual(report["executed_steps"], 0 if failure == "clock" else 1)
                    expected_charge = 2 if failure in ("forward", "record") else 1
                    self.assertEqual(retained["consumed_physics_steps"], expected_charge)
                    self.assertEqual(retained["confirmed_physics_steps"], 0 if failure == "clock" else expected_charge)
                    self.assertEqual(retained["remaining_physics_steps"], 4-expected_charge)
                    self.assertEqual(retained["step_accounting_exact"], failure != "clock")
                    # BEGIN a record-only fault still executes exactly one hold step.
                    if failure == "record":
                        self.assertEqual(simulation._step_attempts, 2)
                        self.assertEqual(report["abort_hold"]["charged_steps"], 1)
                        self.assertEqual(report["abort_hold"]["physical_steps"], 1)
                    # END one normal step plus one hold retains the original contract.
                    if failure in ("feedback", "clock"):
                        self.assertIsNone(retained["actual_feedback"])
                        self.assertFalse((output/"failure-feedback.json").exists())
                    if failure == "clock":
                        self.assertIsNone(retained["elapsed_physical_s"])
                    with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
                        self.assertEqual(len(saved["actual_q"]), 1)
    # END missing captured samples never refund real or uncertain native work.

    def test_remaining_device_inputs_keep_last_applied_interval_and_original_targets(self):
        from manipulation.execution import continuation_execution_inputs
        simulation, curve, targets = _fixture()
        limits = _limits(simulation, curve)
        fixed = FixedFingerTargets.from_mapping(targets)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _execute_bound(_executor(simulation), _result(curve), finger_targets=fixed,
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
        with self.assertRaisesRegex(ValueError, "original full device policy"):
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


# BEGIN independent evidence failures cannot select online execution outcomes.
class IndependentEvidenceSinkTests(unittest.TestCase):
    def test_first_initial_or_post_step_sink_failure_disables_only_collection(self):
        reports, recorded_controls = [], []
        for fail_at in (None, 0, 1):
            simulation, curve, targets = _fixture()
            curve.events[2].update(kind="effect", requires_evidence=None, stop=False)
            _bind_fingers(simulation, curve, targets)
            calls = []

            def collect(model, data, sample):
                calls.append(sample["elapsed"])
                if fail_at is not None and len(calls)-1 == fail_at:
                    raise RuntimeError("scripted independent sink failure")
                # A returned verdict and mutations of the detached sample
                # are deliberately inadmissible as online execution input.
                sample["q"][:] = np.nan
                sample["feedback"]["q"][:] = np.nan
                return {"value": "F"}

            with self.subTest(fail_at=fail_at), TemporaryDirectory() as temporary, patch(
                    "manipulation.robosuite_execution.mujoco.mj_forward"):
                output = Path(temporary)/"trial"
                report = _execute_bound(_executor(simulation), _result(curve),
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), output=output, step_callback=collect)
                with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
                    recorded_controls.append(saved["actuator_ctrl"].copy())
                    self.assertTrue(np.isfinite(saved["actual_q"]).all())
                reports.append(report)
                self.assertEqual(report["status"], "curve_finished")
                self.assertEqual(report["executed_steps"], 6)
                self.assertIsNone(report["feedback_failure"])
                sink = report["independent_evidence_sink"]
                self.assertTrue(sink["configured"])
                self.assertEqual(sink["disabled"], fail_at is not None)
                self.assertEqual(sink["completed"], fail_at is None)
                expected_calls = 7 if fail_at is None else fail_at+1
                self.assertEqual(sink["attempted_callbacks"], expected_calls)
                self.assertEqual(len(calls), expected_calls)
                self.assertEqual(sink["successful_callbacks"], 7 if fail_at is None else fail_at)
                if fail_at is None:
                    self.assertIsNone(sink["first_error"])
                    np.testing.assert_allclose(calls, np.arange(7)*simulation.timestep, atol=1e-15)
                else:
                    failure = sink["first_error"]
                    self.assertEqual(failure["type"], "RuntimeError")
                    self.assertEqual(failure["message"], "scripted independent sink failure")
                    self.assertEqual(failure["phase"], "initial_state" if fail_at == 0 else "after_step")
                    self.assertAlmostEqual(failure["sim_time"], 1.3+fail_at*simulation.timestep)
        for report, controls in zip(reports[1:], recorded_controls[1:]):
            self.assertEqual(report["physics_step_accounting"], reports[0]["physics_step_accounting"])
            self.assertEqual(report["events"], reports[0]["events"])
            np.testing.assert_array_equal(controls, recorded_controls[0])

    def test_sink_failure_at_rejected_step_preserves_primary_guard_failure(self):
        simulation, curve, targets = _fixture()
        checker = _GeometryChecker(simulation.joint_names, rejected_stage="after_step")
        calls = []

        def collect(model, data, sample):
            calls.append(sample["elapsed"])
            if sample["elapsed"] > 0:
                raise RuntimeError("independent sink unavailable")

        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError) as caught:
                _execute_bound(TrajectoryExecutor(simulation, geometry_checker=checker), _result(curve),
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), output=output, step_callback=collect)
            report = json.loads((output/"execution-report.json").read_text())
        self.assertEqual(report["error"], f"ExecutionGuardError: {caught.exception}")
        self.assertEqual(report["feedback_failure"]["point"], "after_step")
        self.assertEqual(report["status"], "feedback_rejected")
        self.assertEqual(report["normal_step_accounting"]["physical_steps"], 1)
        self.assertEqual(report["events"], [])
        self.assertEqual(len(calls), 2)
        sink = report["independent_evidence_sink"]
        self.assertFalse(sink["completed"])
        self.assertEqual(sink["successful_callbacks"], 1)
        self.assertEqual(sink["first_error"]["phase"], "after_step")

    def test_preflight_rejection_still_emits_exactly_one_initial_sample(self):
        simulation, curve, targets = _fixture()
        checker = _GeometryChecker(simulation.joint_names, preflight_passed=False)
        calls = []
        executor = TrajectoryExecutor(simulation, geometry_checker=checker)
        with TemporaryDirectory() as temporary:
            with self.assertRaises(ExecutionGuardError):
                _execute_bound(executor, _result(curve),
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), output=Path(temporary)/"trial",
                    step_callback=lambda model, data, sample: calls.append(sample["elapsed"]))
        self.assertEqual(calls, [0.])
        sink = executor.last_execution_report["independent_evidence_sink"]
        self.assertTrue(sink["completed"])
        self.assertEqual(sink["successful_callbacks"], 1)
# END evaluator collection coverage is separate from task and physics success.


class ExecutionGuardTests(unittest.TestCase):
    # BEGIN reject malformed telemetry without adding physical spike thresholds.
    def test_online_joint_feedback_requires_finite_values_and_exact_actuator_order(self):
        backend, curve, _ = _fixture()
        executor = _executor(backend)
        limits = _limits(backend, curve)
        binding = limits.bind(curve, joint_names=backend.joint_names, arm_joint_names=backend.arm_joint_names,
            finger_joint_names=backend.finger_joint_names, native_joint_ranges=backend.joint_ranges,
            named_velocity_limits=backend.velocity_limits)
        q, dq, _, _ = curve.sample(0.)
        for field in ("actuator_names", "q", "dq"):
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
            report = _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
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
                _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
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
            q, dq, _, _ = curve.sample(min(elapsed, curve.executable_until))
            simulation.native_data.qpos[:7], simulation.native_data.qvel[:7] = q, dq

        simulation._sim.step = step
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _execute_bound(_executor(simulation), _result(curve),
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
            ("tcp_tracking", lambda data: simulation.scripted_tcp_estimate_offset.__setitem__(0, .01)),
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
                    _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
                        initial_feedback=initial, execution_limits=_limits(simulation, curve),
                        output=output,
                        step_callback=lambda model, data, sample: observed.append(sample["q"].copy()))
                report = json.loads((output/"execution-report.json").read_text())
                self.assertEqual(report["status"], "feedback_rejected")
                self.assertEqual(report["executed_steps"], 1)
                self.assertEqual(report["feedback_failure"]["point"], "after_step")
                self.assertIn(kind, [item["kind"] for item in report["feedback_failure"]["violations"]])
                self.assertEqual(report["events"], [])
                self.assertEqual(len(observed), 2)
                if kind in ("feedback_clock", "native_joint_position", "nonfinite_feedback"):
                    self.assertFalse(report["abort_hold"]["prepared"])
                np.testing.assert_array_equal(simulation.native_data.ctrl[7:], curve.finger_trajectory.sample(.002)[0])
                with np.load(output/"physical-trajectory.npz") as saved:
                    self.assertEqual(saved["actual_q"].shape, (2, 9))
                    self.assertEqual(saved["actuator_ctrl"].shape, (1, 9))
                    np.testing.assert_array_equal(saved["actual_q"][-1], simulation.native_data.qpos)

    # BEGIN a real guard failure remains failed after a separately confirmed mock stop.
    def test_native_finger_overrun_keeps_primary_failure_and_counts_bounded_arm_stop(self):
        simulation, curve, targets = _fixture()
        initial = _initial_feedback(simulation)
        limits = _limit_values(simulation, curve)
        limits["stop_arm_velocity"] = [.01]*7
        overshoot = np.nextafter(.04, np.inf)
        commands, physical_samples, hold_entry = [], [], {}

        def scripted_step():
            # These three states are prescribed telemetry, not an integrator:
            # one rejected trajectory step followed by two genuine backend
            # hold calls, whose mocked feedback converges only on the second.
            commands.append(simulation.native_data.ctrl.copy())
            simulation.native_data.time += simulation.timestep
            if len(commands) == 1:
                q, dq, _, _ = curve.sample(simulation.timestep)
                simulation.native_data.qpos[:7], simulation.native_data.qvel[:7] = q, dq
                simulation.native_data.qpos[7] = overshoot
                hold_entry["q"] = simulation.native_data.qpos.copy()
                hold_entry["dq"] = simulation.native_data.qvel.copy()
            else:
                simulation.native_data.qvel[:7] = .02 if len(commands) == 2 else 0.
                simulation.native_data.qvel[7:] = 0.
            # The finger position remains out of bounds throughout stopping.
            # No test callback or backend is allowed to clip that measured q.
            self.assertEqual(simulation.native_data.qpos[7], overshoot)

        simulation._sim.step = scripted_step
        executor = _executor(simulation)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaisesRegex(ExecutionGuardError, "native_joint_position") as caught:
                _execute_bound(executor, _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                    initial_feedback=initial, execution_limits=ExecutionLimits.from_mapping(limits),
                    output=output, max_physics_steps=20, max_hold_steps=3,
                    step_callback=lambda model, data, row: physical_samples.append(deepcopy(row)))
            report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["status"], "feedback_rejected")
            self.assertEqual(report["error"], f"ExecutionGuardError: {caught.exception}")
            self.assertEqual(report["executed_steps"], 1)
            self.assertEqual(report["feedback_failure"]["point"], "after_step")
            self.assertEqual(report["feedback_failure"]["curve_time_s"], simulation.timestep)
            failures = report["feedback_failure"]["violations"]
            self.assertEqual([row["kind"] for row in failures],
                             ["native_joint_position", "actual_finger_position"])
            self.assertEqual(failures[0]["measured"], overshoot)
            np.testing.assert_array_equal(report["feedback_failure"]["actual_feedback"]["dq"], hold_entry["dq"])
            self.assertEqual(report["events"], [])
            self.assertIn("close-original", [row["event_id"] for row in report["unconsumed_events"]])
            self.assertFalse(report["evidence_passed"])
            self.assertEqual(report["completed_goal_ids"], [])
            hold = report["abort_hold"]
            self.assertTrue(hold["native_ctrl_written"])
            self.assertTrue(hold["physical_stop_confirmed"])
            self.assertFalse(hold["joint_position_bounds_satisfied"])
            self.assertEqual(hold["physical_steps"], 2)
            self.assertEqual(len(hold["joint_position_violation_samples"]), 3)
            for command in commands:
                np.testing.assert_array_equal(command[7:], commands[0][7:])
            for control in hold["controls"]:
                np.testing.assert_array_equal(control["reference_arm_q"], hold_entry["q"][:7])
                np.testing.assert_array_equal(control["reference_arm_dq"], np.zeros(7))
                np.testing.assert_array_equal(control["reference_arm_ddq"], np.zeros(7))
                np.testing.assert_array_equal(control["actuator_ctrl"][7:], commands[0][7:])
            continuation = report["continuation"]
            self.assertTrue(continuation["requires_rebind"])
            self.assertTrue(continuation["request_ledger"]["requires_continuation"])
            self.assertEqual(continuation["consumed_physics_steps"], 3)
            self.assertEqual(continuation["cumulative_physics_steps"], 3)
            self.assertEqual(continuation["remaining_physics_steps"], 17)
            self.assertAlmostEqual(continuation["elapsed_physical_s"], 3*simulation.timestep)
            np.testing.assert_array_equal(continuation["actual_feedback"]["q"], hold_entry["q"])
            np.testing.assert_array_equal(continuation["actual_feedback"]["dq"], np.zeros(9))
            self.assertEqual(len(physical_samples), 2)
            self.assertIsNotNone(executor.last_execution_report["feedback_failure"])
            with np.load(output/"physical-trajectory.npz") as saved:
                # The original failed sample remains the curve endpoint;
                # separately counted hold feedback must not replace its dq.
                self.assertEqual(saved["actual_q"].shape, (2, 9))
                self.assertEqual(saved["actuator_ctrl"].shape, (1, 9))
                np.testing.assert_array_equal(saved["actual_q"][-1], hold_entry["q"])
                np.testing.assert_array_equal(saved["actual_dq"][-1], hold_entry["dq"])
    # END converged hold velocity never upgrades the original task or position fault.

    def test_original_stop_checks_actual_velocity_without_extending_physics(self):
        simulation, curve, value = _fixture()
        limits = _limit_values(simulation, curve)
        limits["stop_arm_velocity"] = [.001]*7
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(value),
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
        # BEGIN callers retain the exact finalized report even when execute raises.
        executor = TrajectoryExecutor(simulation, geometry_checker=checker)
        self.assertIsNone(executor.last_execution_report)
        # END no report from an earlier invocation is borrowed.
        # BEGIN this mocked clock has no native MjData for the post-step refresh.
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            # END completing the fake hold lets the test inspect final step accounting.
            output = Path(temporary)/"trial"
            with self.assertRaisesRegex(ExecutionGuardError, "geometry preflight"):
                _execute_bound(executor, _result(curve),
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), output=output)
            report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["feedback_failure"]["point"], "geometry_preflight")
            self.assertEqual(report["executed_steps"], 0)
            self.assertFalse(report["geometry_preflight"]["passed"])
            self.assertEqual(report["events"], [])
            self.assertEqual(checker.states, [])
            self.assertTrue(report["abort_hold"]["native_ctrl_written"])
            # BEGIN retained diagnostics include the final physical hold, not a prefix copy.
            retained = executor.last_execution_report
            self.assertEqual(retained["continuation"]["consumed_physics_steps"], 1)
            self.assertEqual(retained["abort_hold"]["physical_steps"], 1)
            self.assertEqual(retained["robot_feedback_path"], report["robot_feedback_path"])
            self.assertEqual(retained["continuation_path"], report["continuation_path"])
            invalid = _result(curve)
            invalid.success = False
            with self.assertRaises(ValueError):
                _execute_bound(executor, invalid, finger_targets=FixedFingerTargets.from_mapping(targets),
                    initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                    output=Path(temporary)/"invalid")
            self.assertIsNone(executor.last_execution_report)
            # END a rejected new binding cannot leak the prior successful cleanup object.
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
                _execute_bound(TrajectoryExecutor(simulation, geometry_checker=checker), _result(curve),
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
                _execute_bound(TrajectoryExecutor(simulation, geometry_checker=checker), _result(curve),
                    finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                    execution_limits=_limits(simulation, curve), output=output,
                    step_callback=lambda model, data, sample: observed.append(sample["geometry_check"]))
            report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["executed_steps"], 1)
            self.assertEqual(report["feedback_failure"]["point"], "after_step")
            self.assertEqual(len(observed), 2)
            self.assertIsNone(observed[0])
            self.assertFalse(observed[1]["passed"])
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
                _execute_bound(TrajectoryExecutor(simulation, geometry_checker=checker), _result(curve),
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
            report = _execute_bound(_executor(simulation), _result(curve),
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
                    _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                        initial_feedback=initial, execution_limits=_limits(simulation, curve),
                        output=output)
                self.assertFalse(output.exists())
                self.assertEqual(simulation.native_data.time, 1.3)
# END mandatory production geometry gates and explicit diagnostic references.


# BEGIN concrete native backend binding, independent of recorded datasets.
class RobosuiteBackendTests(unittest.TestCase):
    # BEGIN online feedback reads named joint state and calibrated model FK only.
    def test_online_feedback_ignores_privileged_dynamics_and_live_pose_arrays(self):
        backend, _, _ = _fixture()
        backend._qvel_ids = np.array([6, 5, 4, 3, 2, 1, 0, 8, 7])
        backend._actuator_ids = np.array([3, 4, 5, 6, 0, 1, 2, 8, 7])
        velocity = backend.native_data.qvel[backend._qvel_ids].copy()
        controls = backend.native_data.ctrl[backend._actuator_ids].copy()
        for field in ("qacc", "actuator_force", "xmat", "xpos", "site_xmat", "site_xpos"):
            getattr(backend.native_data, field)[:] = np.nan
        feedback = backend.proprioception()
        self.assertEqual(feedback["actuator_names"], backend.actuator_names)
        self.assertNotIn("qacc", feedback)
        self.assertNotIn("actuator_force", feedback)
        self.assertFalse(feedback["tool_pose_is_independent_observation"])
        np.testing.assert_array_equal(feedback["base_pose"], np.eye(4))
        np.testing.assert_array_equal(feedback["tool_pose"], np.eye(4))
        backend.native_data.qvel[:] = 100.
        backend.native_data.ctrl[:] = -100.
        np.testing.assert_array_equal(feedback["dq"], velocity)
        np.testing.assert_array_equal(feedback["actuator_ctrl"]["values"], controls)
        self.assertEqual(backend.sim_time, 1.3)
    # END privileged NaNs cannot change online feedback or advance the mock clock.

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
        control = backend.apply_control(np.asarray([1.]), np.zeros(1), np.zeros(1), np.asarray([.004, -.009]))
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


# BEGIN actual event deadlines and retry identities use measured physical clocks.
# BEGIN actual direction obligations use actual measured coordinates/rates.
# BEGIN final source commands retain real activation and bounded common-clock coverage.
class TerminalSourceCommandTests(unittest.TestCase):
    @staticmethod
    def fixture():
        from manipulation.planning import (PlanningBudget, _initial_timing, _progress_for_cubic_boundaries,
            _fixed_spline_grid, _bind_event_clock)
        from manipulation.planning_problem import PlanningEvent
        from run_recorded_skill import finger_trajectory_factory

        simulation, _, _ = _fixture()
        simulation.native_data.qvel[:] = 0.
        commands = (PlanningEvent("source-open", "declared_skill", 0., 0, "open", "original first source frame"),
                    PlanningEvent("source-last-close", "declared_skill", 1., 197, "close", "original last source frame"))
        problem = SimpleNamespace(events=commands, pose_events=(), coordinate_events=(), supports=(),
            source_time_map={"source_times": [10., 10.1], "progress": [0., 1.]})
        budget, source = _initial_timing(problem, PlanningBudget(knots=6))
        mapping, _ = _progress_for_cubic_boundaries(source, budget.knots, np.zeros(7),
            stop_progress=(), anchors=np.array([0., 1.]))
        grid = _fixed_spline_grid(intervals=budget.knots+4, native_dt=budget.maximum_trajectory_dt,
            interpolation_steps=budget.interpolation_steps, interpolation_dt=budget.interpolation_dt)
        duration = grid["duration"]
        mapping = _bind_event_clock(problem, mapping, duration, simulation.timestep)
        events = tuple({**record["event"], **{key: deepcopy(value) for key, value in record.items() if key != "event"}}
                       for record in mapping.event_clock.values())
        q = simulation.native_data.qpos[:7].copy()
        curve = JointTrajectory(simulation.arm_joint_names, np.array([0., duration]), np.stack([q, q]),
            np.zeros((2, 7)), mapping.u, mapping.s, events, (), {})
        targets = {"joint_names": list(simulation.finger_joint_names),
            "source": "Explicit small CPU terminal command; no actual contact claim",
            "events": [{"skill_id": "declared_skill", "event_id": "source-open", "kind": "open", "positions": [.012, -.017]},
                       {"skill_id": "declared_skill", "event_id": "source-last-close", "kind": "close", "positions": [.01205, -.01705]}]}
        actual = _initial_feedback(simulation)
        input_feedback = {**actual, "start_q": actual["q"], "start_dq": actual["dq"], "sim_time_s": actual["timestamp_s"]}
        factory = finger_trajectory_factory(problem, input_feedback, FixedFingerTargets.from_mapping(targets),
            timestep=simulation.timestep, ctrlrange=simulation.finger_ctrlrange,
            position_limits=dict(zip(simulation.joint_names, simulation.joint_ranges)),
            velocity_limits=dict(zip(simulation.finger_joint_names, [.2, .2])),
            acceleration_limits=dict(zip(simulation.finger_joint_names, [1000., 1000.])))
        curve.finger_trajectory = factory(duration, mapping)
        return simulation, curve, targets, problem, mapping

    def test_last_source_command_uses_formal_terminal_plateau_and_is_actually_applied(self):
        simulation, curve, targets, _, mapping = self.fixture()
        terminal = curve.events[-1]
        self.assertEqual(terminal["source_frame"], 197)
        self.assertEqual(terminal["progress"], 1.)
        self.assertLess(terminal["activation_time_s"], curve.duration)
        self.assertEqual(float(mapping.anchor_parameters([1.])[0, 0]*curve.duration), terminal["activation_time_s"])
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                output=Path(temporary)/"trial")
        self.assertEqual(report["status"], "curve_finished")
        applied = report["events"][-1]
        self.assertEqual(applied["event_id"], "source-last-close")
        self.assertEqual(applied["source_frame"], 197)
        self.assertEqual(applied["applied_at_s"], terminal["activation_time_s"])
        np.testing.assert_allclose(curve.finger_trajectory.sample(curve.duration)[0], [.01205, -.01705], rtol=0, atol=1e-15)
        self.assertEqual(report["unapplied_gripper_events"], [])

    def test_terminal_command_with_no_formal_coverage_or_missed_deadline_remains_rejected(self):
        from manipulation.planning import _ProgressMap, _bind_event_clock, _validate_bound_event_times
        from manipulation.planning_problem import EventTimeConstraint, TimeBindingError
        from dataclasses import replace
        simulation, curve, _, problem, mapping = self.fixture()
        with self.assertRaisesRegex(ValueError, "activation/effect coverage"):
            _bind_event_clock(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                              curve.duration, simulation.timestep)
        terminal = problem.events[-1]
        bound = mapping.event_clock[(terminal.skill_id, terminal.event_id, terminal.occurrence)]
        problem.events = (problem.events[0], replace(terminal, timing=(EventTimeConstraint(
            check_target="activation", window_s=(None, bound["activation_time_s"]-simulation.timestep),
            time_basis="request_physical", origin_time_s=0., source="Original hard terminal-command deadline"),)))
        rebound = _bind_event_clock(problem, _ProgressMap(mapping.u, mapping.s), curve.duration, simulation.timestep)
        with self.assertRaises(TimeBindingError):
            _validate_bound_event_times(problem, rebound, curve.duration)
# END source-frame coverage does not authorize post-validation retiming or deadline changes.


class ActualDirectionEvidenceTests(unittest.TestCase):
    @staticmethod
    def fixture(kind="local_rate", *, start=0., end=1.):
        simulation, curve, targets = _fixture()
        curve.events[2].update(kind="function", requires_evidence=None, stop=False)
        _bind_fingers(simulation, curve, targets)
        result = _result(curve)
        requirement = {"start": start, "end": end, "coordinate_index": 0,
            "kind": kind, "sign": 1, "minimum": .005 if kind != "monotone" else 0.,
            "tolerance": 0., "source": "Explicit synthetic direction requirement",
            "origin_coordinate": float(simulation.native_data.qpos[0]) if kind == "net" else None,
            "origin_source": "Actual mock coordinate at original entrance" if kind == "net" else None}
        binding = result.execution_binding
        binding["problem"]["mechanisms"] = [{"directions": [requirement]}]
        binding["validation"]["inputs"]["problem"] = deepcopy(binding["problem"])
        binding.update(direction_identities=[[[3, 7]]], completed_direction_ids=[], direction_evidence=[],
                       retry_direction_ids=[])
        return simulation, curve, targets, result

    @staticmethod
    def execute(simulation, curve, targets, result, observer, output):
        return _execute_bound(_executor(simulation), result, finger_targets=FixedFingerTargets.from_mapping(targets),
            initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
            output=output, observation_callback=observer)

    def test_measured_rate_completes_original_direction_and_preserves_observation_interval(self):
        simulation, curve, targets, result = self.fixture()
        start = simulation.sim_time
        def observe(feedback, curve_time):
            # Mock q is intentionally static; this rate-specific proof reads
            # the separately supplied actual dq, never the cubic derivative.
            return {"source": "Explicit actual joint-rate channel", "timestamp_s": feedback["timestamp_s"],
                "directions": [{"identity": [3, 7], "value": "T" if feedback["dq"][0] >= .005 else "F",
                    "source": "Actual dq[0] from declared mechanism coordinate sensor",
                    "measured_rate": float(feedback["dq"][0]), "acquisition_interval_s": [start, feedback["timestamp_s"]],
                    "scope_complete": curve_time >= curve.duration}]}
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = self.execute(simulation, curve, targets, result, observe, Path(temporary)/"trial")
        self.assertEqual(report["status"], "curve_finished")
        self.assertEqual(report["direction_result"], "T")
        self.assertEqual(report["completed_direction_ids"], [[3, 7]])
        self.assertEqual(report["continuation"]["completed_direction_ids"], [[3, 7]])
        self.assertEqual(report["request_ledger"]["direction_evidence"], report["direction_evidence"])
        final = report["direction_evidence"][-1]
        self.assertTrue(final["completion_verified"])
        self.assertEqual(final["candidate_id"], result.execution_binding["candidate_id"])
        self.assertAlmostEqual(final["acquisition_interval_s"][1]-final["acquisition_interval_s"][0], curve.duration)

    def test_current_false_or_unknown_holds_and_future_unknown_does_not_block_entrance(self):
        for value, start, expected_steps, expected in (("F", 0., 0, "F"), ("U", 0., 0, "U"),
                                                      ("U", .5, 3, "U")):
            simulation, curve, targets, result = self.fixture(start=start)
            def observe(feedback, curve_time):
                return {"source": "Explicit actual direction observer", "timestamp_s": feedback["timestamp_s"],
                    "directions": [{"identity": [3, 7], "value": value, "source": "Current actual rate assessment",
                        "acquisition_interval_s": [feedback["timestamp_s"]]*2, "scope_complete": False}]}
            with self.subTest(value=value, start=start), TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
                output = Path(temporary)/"trial"
                with self.assertRaises(ExecutionGuardError):
                    self.execute(simulation, curve, targets, result, observe, output)
                report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["executed_steps"], expected_steps)
            self.assertEqual(report["direction_result"], expected)
            self.assertTrue(report["abort_hold"]["native_ctrl_written"])
            self.assertEqual(report["completed_direction_ids"], [])

    def test_net_completion_uses_actual_displacement_and_cannot_follow_reference_progress(self):
        simulation, curve, targets, result = self.fixture("net")
        initial = float(simulation.native_data.qpos[0])
        def observe(feedback, curve_time):
            change = float(feedback["q"][0])-initial
            return {"source": "Actual coordinate sensor", "timestamp_s": feedback["timestamp_s"],
                "directions": [{"identity": [3, 7], "value": "T" if change >= .005 else "F",
                    "source": "Measured displacement from original actual coordinate", "measured_change": change,
                    "acquisition_interval_s": [feedback["timestamp_s"]]*2, "scope_complete": curve_time >= curve.duration}]}
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                self.execute(simulation, curve, targets, result, observe, output)
            report = json.loads((output/"execution-report.json").read_text())
        self.assertEqual(report["executed_steps"], 6)
        self.assertEqual(report["direction_result"], "F")
        self.assertEqual(report["completed_direction_ids"], [])
        self.assertTrue(all(row["measured_change"] == 0. for row in report["direction_evidence"]))

    def test_scope_complete_without_full_actual_interval_remains_unverified(self):
        simulation, curve, targets, result = self.fixture("monotone")
        def observe(feedback, curve_time):
            return {"source": "Actual short-interval observer", "timestamp_s": feedback["timestamp_s"],
                "directions": [{"identity": [3, 7], "value": "T", "source": "Only this actual sample interval",
                    "acquisition_interval_s": [feedback["timestamp_s"]]*2, "scope_complete": True}]}
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                self.execute(simulation, curve, targets, result, observe, output)
            report = json.loads((output/"execution-report.json").read_text())
        self.assertEqual(report["direction_result"], "U")
        self.assertEqual(report["executed_steps"], 6)
        self.assertEqual(report["completed_direction_ids"], [])

        self.assertTrue(report["direction_evidence"][-1]["scope_complete"])
        self.assertFalse(report["direction_evidence"][-1]["completion_verified"])

    def test_retry_preserves_old_completion_fact_but_requires_new_acquisition(self):
        simulation, curve, targets, result = self.fixture()
        historical = {"identity": [3, 7], "candidate_id": "previous attempt", "value": "T",
            "source": "Old actual full-scope evidence", "scope_complete": True,
            "scope_started_at_s": 1., "acquisition_interval_s": [1., 1.2]}
        result.execution_binding.update(retry_direction_ids=[[3, 7]], direction_evidence=[historical])
        def observe(feedback, curve_time):
            return {"source": "Fresh assessment carrying an old witness", "timestamp_s": feedback["timestamp_s"],
                    "directions": [historical]}
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                self.execute(simulation, curve, targets, result, observe, output)
            report = json.loads((output/"execution-report.json").read_text())
        self.assertEqual(report["direction_evidence"][0], historical)
        self.assertEqual(report["direction_result"], "U")
        self.assertEqual(report["completed_direction_ids"], [])
        self.assertEqual(report["executed_steps"], 0)

    def test_nonperiodic_scope_boundaries_acquire_actual_evidence_without_moving_periodic_phase(self):
        simulation, curve, targets, result = self.fixture(start=.25, end=.75)
        initial_time = simulation.sim_time
        sampled = []
        def observe(feedback, curve_time):
            sampled.append(curve_time)
            return {"source": "Actual rate observer available at declared boundary polls",
                "timestamp_s": feedback["timestamp_s"], "directions": [{"identity": [3, 7], "value": "T",
                    "source": "Actual sensor interval retained by the observer",
                    "acquisition_interval_s": [initial_time, feedback["timestamp_s"]],
                    "scope_complete": curve_time >= .009}]}
        limits = _limit_values(simulation, curve)
        limits["observation_policy"].update(period_s=.006, maximum_age_s=.006)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _execute_bound(_executor(simulation), result, finger_targets=FixedFingerTargets.from_mapping(targets),
                initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(limits),
                output=Path(temporary)/"trial", observation_callback=observe)
        # Scope boundaries .003/.009 reach native guards .004/.010; the
        # ordinary .006/.012 acquisitions keep their original phase.
        np.testing.assert_allclose(sampled, [0., .004, .006, .010, .012])
        self.assertEqual(report["direction_result"], "T")
        boundaries = [row for row in report["observation_checks"] if row["poll_reason"] == "direction_boundary"]
        np.testing.assert_allclose([row["polled_at_s"]-initial_time for row in boundaries], [.004, .010])
# END clock progress and earlier attempts cannot certify actual direction.


# BEGIN actual task accuracy keeps the original allowed set, owner and clock.
class ActualPoseGoalTests(unittest.TestCase):
    @staticmethod
    def fixture(*, progress=.5, actual_relation=None):
        simulation, curve, targets = _fixture()
        curve.events[2].update(kind="function", requires_evidence=None, stop=False)
        _bind_fingers(simulation, curve, targets)
        result = _result(curve)
        result.execution_binding["problem"]["pose_events"] = [{"progress": progress,
            "tool_from_function": np.eye(4).tolist(), "target": np.eye(4).tolist(),
            "position_tolerance_m": .001, "rotation_tolerance_rad": .01,
            "source": "Original synthetic 1 mm functional task requirement", "actual_relation": actual_relation}]
        result.execution_binding["goal_identities"] = {"pose_events": [7]}
        result.execution_binding["validation"]["inputs"]["problem"] = deepcopy(result.execution_binding["problem"])
        values = _limit_values(simulation, curve)
        values["tcp_position_error_m"] = .005
        values["tcp_rotation_error_rad"] = .02
        return simulation, curve, targets, result, ExecutionLimits.from_mapping(values)

    @staticmethod
    def run_trial(fixture, *, observe=None, step_callback=None, rejects=False):
        simulation, curve, targets, result, limits = fixture
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            if rejects:
                with unittest.TestCase().assertRaises(ExecutionGuardError):
                    _execute_bound(_executor(simulation), result, finger_targets=FixedFingerTargets.from_mapping(targets),
                        initial_feedback=_initial_feedback(simulation), execution_limits=limits, output=output,
                        observation_callback=observe, step_callback=step_callback)
                return json.loads((output/"execution-report.json").read_text())
            return _execute_bound(_executor(simulation), result, finger_targets=FixedFingerTargets.from_mapping(targets),
                initial_feedback=_initial_feedback(simulation), execution_limits=limits, output=output,
                observation_callback=observe, step_callback=step_callback)

    def test_two_mm_actual_miss_fails_one_mm_goal_inside_five_mm_tracking_band(self):
        fixture = self.fixture()
        fixture[0].scripted_tcp_estimate_offset[0] = .002
        def observer(feedback, curve_time):
            return {"source": "An untrusted pose-completion label cannot replace measurement",
                "timestamp_s": feedback["timestamp_s"], "completed_goal_ids": [["pose_events", 7]]}
        report = self.run_trial(fixture, observe=observer, rejects=True)
        self.assertEqual(report["pose_goal_result"], "F")
        self.assertEqual(report["feedback_failure"]["violations"][0]["kind"], "actual_pose_goal_violated")
        self.assertEqual(report["completed_goal_ids"], [])
        proof = report["goal_evidence"][-1]
        self.assertAlmostEqual(proof["position_error_m"], .002)
        self.assertEqual(proof["requirement"]["position_tolerance_m"], .001)
        self.assertAlmostEqual(proof["evaluated_curve_time_s"], .006)
        self.assertTrue(report["abort_hold"]["native_ctrl_written"])

    def test_within_original_tolerance_completes_once_with_original_identity(self):
        fixture = self.fixture()
        simulation = fixture[0]
        simulation.scripted_tcp_estimate_offset[0] = .0005
        def after_step(model, data, row):
            # The once-goal has already been measured at .006 s. A later
            # departure inside the engineering band adds no invented hold goal.
            if row["elapsed"] >= .008:
                simulation.scripted_tcp_estimate_offset[0] = .003
        report = self.run_trial(fixture, step_callback=after_step)
        self.assertEqual(report["status"], "curve_finished")
        self.assertEqual(report["pose_goal_result"], "T")
        self.assertEqual(report["completed_goal_ids"], [["pose_events", 7]])
        self.assertEqual(len(report["pose_goal_checks"]), 1)
        proof = report["goal_evidence"][-1]
        self.assertTrue(proof["completion_verified"])
        self.assertAlmostEqual(proof["position_error_m"], .0005)
        self.assertEqual(proof["candidate_id"], fixture[3].execution_binding["candidate_id"])
        self.assertEqual(report["continuation"]["completed_goal_ids"], [["pose_events", 7]])

    def test_owner_relative_goal_without_current_owner_is_unknown_only_when_due(self):
        fixture = self.fixture(actual_relation={"target_owner_id": "current_part", "T_owner_target": np.eye(4).tolist()})
        report = self.run_trial(fixture, rejects=True)
        self.assertEqual(report["pose_goal_result"], "U")
        self.assertEqual(report["executed_steps"], 3)
        self.assertEqual(report["feedback_failure"]["violations"][0]["kind"], "actual_pose_goal_unverified")
        self.assertIsNone(report["goal_evidence"][-1]["actual_target_pose"])
        self.assertIn("current_part", report["goal_evidence"][-1]["reason"])
        self.assertTrue(report["continuation"]["requires_rebind"])

    def test_current_owner_pose_changes_relative_result_without_changing_frozen_world_target(self):
        for shift, expected in ((.0004, "T"), (.002, "F")):
            fixture = self.fixture(actual_relation={"target_owner_id": "current_part", "T_owner_target": np.eye(4).tolist()})
            current = np.eye(4)
            current[0, 3] = shift
            def observe(feedback, curve_time):
                return {"source": "Explicit current calibrated owner observation", "timestamp_s": feedback["timestamp_s"],
                    "scene_owner_poses": {"current_part": current.tolist()}}
            with self.subTest(shift=shift):
                report = self.run_trial(fixture, observe=observe, rejects=expected == "F")
                self.assertEqual(report["pose_goal_result"], expected)
                self.assertAlmostEqual(report["goal_evidence"][-1]["position_error_m"], shift)

    def test_functional_time_uses_first_nonzero_plateau_entrance(self):
        fixture = self.fixture()
        simulation, curve, _, result, _ = fixture
        curve.progress_u = np.asarray([0., .4, .7, 1.])
        curve.progress_s = np.asarray([0., .5, .5, 1.])
        from manipulation.execution_binding import curve_snapshot
        result.execution_binding["curve"] = curve_snapshot(curve)
        simulation.scripted_tcp_estimate_offset[0] = .0005
        report = self.run_trial(fixture)
        proof = report["goal_evidence"][-1]
        self.assertAlmostEqual(proof["scheduled_curve_time_s"], .0048)
        self.assertAlmostEqual(proof["evaluated_curve_time_s"], .006)
        self.assertEqual(proof["identity"], ["pose_events", 7])

    def test_one_ulp_future_functional_time_never_uses_the_earlier_native_tick(self):
        scheduled = .010000000000000002
        fixture = self.fixture(progress=scheduled/.012)
        report = self.run_trial(fixture)
        proof = report["goal_evidence"][-1]
        self.assertEqual(proof["scheduled_curve_time_s"], scheduled)
        self.assertEqual(proof["device_due_time_s"], .012)
        self.assertEqual(proof["physical_elapsed_s"], .012)
        self.assertEqual(proof["evaluated_curve_time_s"], .012)

    def test_terminal_actual_tick_preserves_the_clipped_original_curve_time(self):
        fixture = self.fixture(progress=1.)
        simulation, old, targets, result, limits = fixture
        duration = .011
        q, dq = old.sample(0.)[:2]
        curve = JointTrajectory(old.joint_names, np.asarray([0., duration]),
            np.stack((q, q+dq*duration)), np.stack((dq, dq)), old.progress_u, old.progress_s,
            old.events, old.conditions, {})
        _bind_fingers(simulation, curve, targets)
        result.trajectory = curve
        from manipulation.execution_binding import curve_snapshot
        result.execution_binding["curve"] = curve_snapshot(curve)
        report = self.run_trial((simulation, curve, targets, result, limits))
        proof = report["goal_evidence"][-1]
        self.assertEqual(proof["scheduled_curve_time_s"], duration)
        self.assertEqual(proof["device_due_time_s"], .012)
        self.assertEqual(proof["physical_elapsed_s"], .012)
        self.assertEqual(proof["evaluated_curve_time_s"], duration)

    def test_held_function_needs_current_relation_and_consumes_full_material_transform(self):
        from manipulation.execution import _actual_pose_goal
        from manipulation.planning_problem import PoseEventRequirement
        tool_from_function = np.eye(4)
        tool_from_function[0, 3] = .1
        requirement = PoseEventRequirement(.5, tool_from_function, tool_from_function, .001, .01,
            "Explicit synthetic held material", actual_relation={"function_owner_id": "held",
                "T_owner_function": np.eye(4), "attachment_condition_id": "actual_holding"})
        # BEGIN the direct helper fixture declares its model-estimate provenance.
        feedback = {"tool_pose": np.eye(4), "timestamp_s": 2., "source": "Actual named robot feedback", "feedback_id": "robot:2",
            "tool_pose_source": "forward_kinematics_from_measured_arm_q", "tool_pose_is_independent_observation": False}
        # END a verified attachment does not supply independent TCP sensing.
        self.assertEqual(_actual_pose_goal(requirement, feedback, None)["value"], "U")
        observed = {"timestamp_s": 2., "source": "Actual synchronized holding observation",
                    "scene_condition_evidence": {"actual_holding": True}}
        proof = _actual_pose_goal(requirement, feedback, observed)
        self.assertEqual(proof["value"], "T")
        np.testing.assert_array_equal(proof["actual_function_pose"], tool_from_function)
        # BEGIN the original estimate provenance survives functional composition.
        self.assertEqual(proof["measurement_sources"][0]["pose_source"], "forward_kinematics_from_measured_arm_q")
        self.assertFalse(proof["measurement_sources"][0]["independent_observation"])
        # END the material transform changes the pose, not its source class.

    def test_upstream_nonpose_completion_keeps_acquisition_clock_when_delivery_is_delayed(self):
        simulation, curve, targets = _fixture()
        curve.events[2].update(kind="function", requires_evidence=None, stop=False)
        _bind_fingers(simulation, curve, targets)
        result = _result(curve)
        result.execution_binding["problem"]["coordinate_events"] = [{
            "progress": 1., "source": "Explicit upstream native coordinate goal contract"}]
        result.execution_binding["goal_identities"] = {"coordinate_events": [3]}
        result.execution_binding["validation"]["inputs"]["problem"] = deepcopy(result.execution_binding["problem"])
        values = _limit_values(simulation, curve)
        values["observation_policy"]["maximum_age_s"] = .01
        def observer(feedback, curve_time):
            return {"source": "Explicit delayed upstream coordinate evaluation",
                "timestamp_s": feedback["timestamp_s"]-(.001 if curve_time >= .002 else 0.),
                "completed_goal_ids": [["coordinate_events", 3]] if curve_time >= .002 else []}
        report = self.run_trial((simulation, curve, targets, result, ExecutionLimits.from_mapping(values)), observe=observer)
        proof = report["goal_evidence"][-1]
        self.assertAlmostEqual(proof["physical_elapsed_s"], .001)
        self.assertAlmostEqual(proof["evaluated_curve_time_s"], .001)
        self.assertAlmostEqual(proof["consumer_elapsed_s"], .002)
        self.assertAlmostEqual(proof["timestamp_s"], 1.301)
# END these mock controller checks do not claim a formal physical grasp success.


# BEGIN actual arrival gates a named contact mode before a close control write.
class GraspContactActivationTests(unittest.TestCase):
    @staticmethod
    def fixture(*, reopen=False, zero_tick=False):
        simulation, curve, targets = _fixture()
        curve.events[1]["progress"] = .25
        curve.events[2].update(kind="function", requires_evidence=None, stop=False)
        if reopen:
            curve.events[3].update(kind="open", progress=.75)
            targets["events"][2].update(kind="open", positions=[.0121, -.0171])
        # BEGIN a zero-tick close has no earlier open event or inherited latch.
        close = curve.events[1]
        if zero_tick:
            close.update(time=0., progress=0.)
            curve.events = curve.events[1:]
            targets["events"] = targets["events"][1:]
        activation = {key: close[key] for key in ("skill_id", "event_id", "occurrence", "kind")}
        # END both fixtures retain the same explicit original close identity.
        condition = {"kind": "grasp_contact_activation", "condition_id": "declared-contact",
            "carrier_id": "perceived_part", "arrival_goal_id": ["pose_events", 1],
            "activation_event": activation, "interval": [close["progress"], .75 if reopen else 1.],
            "source": "Explicit synthetic frozen RGBD interaction binding",
            "contact_rule_ids": ["contact-rule"]}
        curve.conditions = (deepcopy(condition),)
        _bind_fingers(simulation, curve, targets)
        result = _result(curve)
        goal = {"progress": 1., "tool_from_function": np.eye(4).tolist(), "target": np.eye(4).tolist(),
            "position_tolerance_m": .001, "rotation_tolerance_rad": .01,
            "source": "Original synthetic interaction pose allowed set", "actual_relation": None}
        terminal = {**deepcopy(goal), "actual_relation": {"target_owner_id": "perceived_part",
            "T_owner_target": np.eye(4).tolist()}}
        result.execution_binding["problem"].update(conditions=[condition],
            pose_events=[terminal, {**goal, "progress": close["progress"],
                "source": {"kind": "frozen_perception_grasp_arrival", "carrier_id": "perceived_part",
                           "activation_event": deepcopy(activation)}}])
        result.execution_binding["goal_identities"] = {"pose_events": [0, 1]}
        # BEGIN fixture grants only the explicitly bound finite material rule.
        result.execution_binding["scene"]["contacts"] = [{
            "rule_id": "contact-rule", "actor": "robot", "target_owner": "perceived_part",
            "material_id": "named_left_pad", "sphere_indices": [0],
            "interval": deepcopy(condition["interval"]), "source": "Synthetic finite pad permission"}]
        result.execution_binding["scene"]["grasp_contact_activations"] = [deepcopy(condition)]
        # END scripted geometry verdicts do not claim a physical contact result.
        result.execution_binding["validation"]["inputs"] = deepcopy({
            key: result.execution_binding[key] for key in ("problem", "scene")})
        values = _limit_values(simulation, curve)
        values.update(tcp_position_error_m=.005, tcp_rotation_error_rad=.02)
        return simulation, curve, targets, result, ExecutionLimits.from_mapping(values)

    @staticmethod
    def run_trial(fixture, *, checker=None, callback=None, rejects=False):
        simulation, curve, targets, result, limits = fixture
        checker = checker or _GeometryChecker(simulation.joint_names)
        executor = TrajectoryExecutor(simulation, geometry_checker=checker)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            def run():
                return _execute_bound(executor, result, result_only_goal_ids=[["pose_events", 0]],
                    finger_targets=FixedFingerTargets.from_mapping(targets), execution_limits=limits,
                    initial_feedback=_initial_feedback(simulation), output=Path(temporary)/"trial",
                    step_callback=callback)
            if rejects:
                with unittest.TestCase().assertRaises(ExecutionGuardError):
                    run()
                return executor.last_execution_report, checker
            return run(), checker

    # BEGIN zero-tick entry proves arrival without world permission state.
    def test_zero_tick_arrival_is_checked_before_robot_preflight_and_close(self):
        fixture = self.fixture(zero_tick=True)
        simulation = fixture[0]
        simulation.scripted_tcp_estimate_offset[0] = .0005
        report, checker = self.run_trial(fixture)
        self.assertEqual(report["status"], "curve_finished")
        self.assertFalse(report["world_collision_checked"])
        self.assertNotIn("active_contact_rule_ids", report)
        np.testing.assert_array_equal(checker.preflight_inputs[0]["q"], checker.states[0]["feedback"]["q"])
        checks = report["grasp_contact_gate_checks"]
        self.assertEqual([row["point"] for row in checks], ["initial_contact_gate", "before_step"])
        self.assertEqual([row["measurement"]["value"] for row in checks], ["T", "T"])
        self.assertEqual([row["control_applied"] for row in checks], [False, True])
        self.assertEqual(report["events"][0]["kind"], "close")
        self.assertEqual(report["events"][0]["applied_at_s"], 0.)

    def test_zero_tick_missed_arrival_rejects_before_geometry_or_close_control(self):
        fixture = self.fixture(zero_tick=True)
        simulation = fixture[0]
        simulation.scripted_tcp_estimate_offset[0] = .002
        initial_control = simulation.finger_control.copy()
        checker = _GeometryChecker(simulation.joint_names, preflight_passed=False)
        report, _ = self.run_trial(fixture, checker=checker, rejects=True)
        self.assertEqual(report["feedback_failure"]["point"], "initial_contact_gate")
        self.assertEqual(report["feedback_failure"]["violations"][0]["kind"], "grasp_contact_arrival_rejected")
        self.assertEqual(report["executed_steps"], 0)
        self.assertEqual(report["events"], [])
        self.assertNotIn("active_contact_rule_ids", report)
        self.assertNotIn("grasp_contact_permission_changes", report)
        self.assertEqual(checker.preflight_inputs, [])
        self.assertEqual(checker.states, [])
        self.assertEqual([row["measurement"]["value"] for row in report["grasp_contact_gate_checks"]], ["F"])
        self.assertFalse(report["grasp_contact_gate_checks"][0]["control_applied"])
        self.assertTrue(report["abort_hold"]["native_ctrl_written"])
        np.testing.assert_array_equal(simulation.finger_control, initial_control)
    # END entry permissions remain provisional, and a miss preserves the old finger goal.

    def test_due_clock_and_old_completed_goal_cannot_close_without_current_arrival(self):
        fixture = self.fixture()
        simulation, curve, _, result, _ = fixture
        simulation.scripted_tcp_estimate_offset[0] = .002
        result.execution_binding["completed_goal_ids"] = [["pose_events", 1]]
        report, checker = self.run_trial(fixture, rejects=True)
        self.assertEqual(report["executed_steps"], 2)
        self.assertNotIn("active_contact_rule_ids", report)
        self.assertEqual([row["kind"] for row in report["events"]], ["open"])
        self.assertEqual(report["grasp_contact_gate_checks"][-1]["measurement"]["value"], "F")
        self.assertFalse(report["grasp_contact_gate_checks"][-1]["control_applied"])
        self.assertTrue(all("active_contact_rule_ids" not in row for row in checker.states))
        np.testing.assert_array_equal(simulation.finger_control, curve.finger_trajectory.sample(.004)[0])

    def test_same_tick_arrival_is_provisional_and_rechecked_before_close(self):
        fixture = self.fixture()
        simulation = fixture[0]
        def drift(model, data, row):
            if row["elapsed"] == .004:
                simulation.scripted_tcp_estimate_offset[0] = .002
        report, _ = self.run_trial(fixture, callback=drift, rejects=True)
        self.assertEqual([row["measurement"]["value"] for row in report["grasp_contact_gate_checks"]], ["T", "F"])
        self.assertIn(["pose_events", 1], report["completed_goal_ids"])
        self.assertNotIn("active_contact_rule_ids", report)
        self.assertFalse(any(row["kind"] == "close" for row in report["events"]))

    def test_valid_arrival_allows_its_contact_mode_without_adding_a_pose_hold_goal(self):
        fixture = self.fixture()
        simulation = fixture[0]
        def later_departure(model, data, row):
            if row["elapsed"] >= .006:
                simulation.scripted_tcp_estimate_offset[0] = .002
        checker = _GeometryChecker(simulation.joint_names)
        report, _ = self.run_trial(fixture, checker=checker, callback=later_departure)
        self.assertEqual(report["status"], "curve_finished")
        self.assertNotIn("active_contact_rule_ids", report)
        self.assertEqual(sum(row["control_applied"] for row in report["grasp_contact_gate_checks"]), 1)
        self.assertTrue(all(row["passed"] for row in report["grasp_contact_gate_checks"]))
        self.assertFalse(report["evidence_passed"])
        self.assertNotIn(["pose_events", 0], report["completed_goal_ids"])
        self.assertTrue(all("active_contact_rule_ids" not in row for row in checker.states))

    def test_new_mode_still_runs_geometry_guard_and_does_not_commit_a_rejected_close(self):
        fixture = self.fixture()
        checker = _GeometryChecker(fixture[0].joint_names, rejected_stage="command_event", rejected_time=.004)
        report, _ = self.run_trial(fixture, checker=checker, rejects=True)
        self.assertEqual(report["feedback_failure"]["point"], "command_event")
        self.assertTrue(report["grasp_contact_gate_checks"][-1]["passed"])
        self.assertFalse(report["grasp_contact_gate_checks"][-1]["control_applied"])
        self.assertNotIn("active_contact_rule_ids", report)
        self.assertNotIn("active_contact_rule_ids", checker.states[-1])
        self.assertFalse(any(row["kind"] == "close" for row in report["events"]))

    def test_open_keeps_original_event_order_without_permission_state(self):
        report, checker = self.run_trial(self.fixture(reopen=True))
        self.assertEqual(report["status"], "curve_finished")
        self.assertNotIn("active_contact_rule_ids", report)
        self.assertNotIn("grasp_contact_permission_changes", report)
        self.assertEqual([row["kind"] for row in report["events"]], ["open", "close", "open"])

    def test_contact_arrival_cannot_be_result_only_or_mismatched_to_another_owner(self):
        from manipulation.execution_binding import rebind_execution_result
        simulation, _, targets, result, limits = self.fixture()
        with self.assertRaisesRegex(ValueError, "contact-activation arrival cannot be result-only"):
            rebind_execution_result(result, controller_policy=simulation.describe()["controller_policy"],
                result_only_goal_ids=[["pose_events", 1]], finger_targets=FixedFingerTargets.from_mapping(targets),
                execution_limits=limits, source="Explicit synthetic forbidden result-only request")
        result.execution_binding["scene"]["contacts"][0]["target_owner"] = "another_part"
        result.execution_binding["validation"]["inputs"]["scene"] = deepcopy(result.execution_binding["scene"])
        with TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, "exact activation interval"):
            _execute_bound(_executor(simulation), result, result_only_goal_ids=[["pose_events", 0]],
                finger_targets=FixedFingerTargets.from_mapping(targets), initial_feedback=_initial_feedback(simulation),
                execution_limits=limits, output=Path(temporary)/"trial")
        self.assertEqual(simulation.arm_goals, [])
        self.assertEqual(simulation.sim_time, 1.3)
# END all controllers, geometry and clocks are doubles; no native step is issued.


class PhysicalEventTimingTests(unittest.TestCase):
    @staticmethod
    def window(target, lower, upper, **details):
        return {"check_target": target, "window_s": [lower, upper],
                "time_basis": "request_physical", "source": "Explicit synthetic physical deadline",
                "origin_time_s": 0., "origin_event": None, "goal_id": None, "condition_id": None,
                **details}

    def test_missed_activation_deadline_holds_before_actuator_application(self):
        simulation, curve, targets = _fixture()
        curve.events[1]["timing"] = [self.window("activation", None, .0035)]
        _bind_fingers(simulation, curve, targets)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                    initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve), output=output)
            report = json.loads((output/"execution-report.json").read_text())
        self.assertEqual([row["event_id"] for row in report["events"]], ["open-original"])
        self.assertEqual(report["feedback_failure"]["point"], "expired_event_window")
        self.assertEqual(report["timing_result"], "U")
        self.assertTrue(report["abort_hold"]["native_ctrl_written"])

    def test_actual_arrival_is_checked_on_original_request_time_not_reference_progress(self):
        for upper, accepted in ((20.005, True), (20.003, False)):
            simulation, curve, targets = _fixture()
            curve.events[2].update(kind="function", requires_evidence=None, stop=False,
                timing=[self.window("arrival", 20.003, upper, goal_id=["pose_events", 0])])
            _bind_fingers(simulation, curve, targets)
            result = _result(curve)
            result.execution_binding.update(request_time_origin_s=20., goal_identities={"pose_events": [0]})
            # BEGIN the executor independently measures the original pose set.
            result.execution_binding["problem"]["pose_events"] = [{"progress": 1.,
                "tool_from_function": np.eye(4).tolist(), "target": np.eye(4).tolist(),
                "position_tolerance_m": .001, "rotation_tolerance_rad": .01,
                "source": "Explicit fixed-world CPU arrival goal", "actual_relation": None}]
            # END an upstream completed-goal label alone has no pose authority.
            result.execution_binding["validation"]["inputs"]["problem"] = deepcopy(result.execution_binding["problem"])
            def observe(feedback, curve_time):
                return {"source": "Synthetic measured completion before soft source progress 1",
                    "timestamp_s": feedback["timestamp_s"],
                    "completed_goal_ids": [["pose_events", 0]] if curve_time >= .004 else []}
            with self.subTest(upper=upper), TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
                output = Path(temporary)/"trial"
                if accepted:
                    report = _execute_bound(_executor(simulation), result, finger_targets=FixedFingerTargets.from_mapping(targets),
                        initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                        output=output, observation_callback=observe)
                else:
                    with self.assertRaises(ExecutionGuardError):
                        _execute_bound(_executor(simulation), result, finger_targets=FixedFingerTargets.from_mapping(targets),
                            initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                            output=output, observation_callback=observe)
                    report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["timing_result"], "T" if accepted else "F")
            arrival = next(row for row in report["history"] if row["check_target"] == "arrival")
            self.assertAlmostEqual(arrival["time_s"], 20.004)
            self.assertEqual(arrival["value"], "T")
            self.assertEqual(report["completed_goal_ids"], [["pose_events", 0]] if accepted else [])

    def test_effect_window_uses_new_acquisition_and_preserves_old_false_true_history(self):
        for fresh in (False, True):
            simulation, curve, targets = _fixture()
            event = curve.events[2]
            event.update(kind="effect", requires_evidence=None, stop=False, occurrence=1,
                timing=[self.window("effect", .005, .009, condition_id="touch")])
            curve.conditions = ({"condition_id": "touch", "effect_requirement": "synthetic contact",
                                 "skill_id": event["skill_id"], "event_id": event["event_id"]},)
            _bind_fingers(simulation, curve, targets)
            previous = [{"skill_id": event["skill_id"], "event_id": event["event_id"], "occurrence": 0,
                "check_target": "effect", "time_s": value, "value": truth, "source": "previous physical occurrence"}
                for value, truth in ((-.02, "F"), (-.01, "T"))]
            result = _result(curve)
            result.execution_binding["request_ledger"] = {"history": deepcopy(previous)}
            values = _limit_values(simulation, curve)
            values["observation_policy"]["maximum_age_s"] = .1
            def observe(feedback, curve_time):
                return {"source": "synthetic independently timestamped effect",
                    "timestamp_s": feedback["timestamp_s"] if fresh else 1.29,
                    "effects": {"touch": {"value": "T" if curve_time >= .006 or not fresh else "F" if curve_time == .002 else "U",
                        "source": "current physical sample" if fresh else "earlier binding sample"}}}
            with self.subTest(fresh=fresh), TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
                output = Path(temporary)/"trial"
                if fresh:
                    report = _execute_bound(_executor(simulation), result, finger_targets=FixedFingerTargets.from_mapping(targets),
                        initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(values),
                        output=output, observation_callback=observe)
                else:
                    with self.assertRaises(ExecutionGuardError):
                        _execute_bound(_executor(simulation), result, finger_targets=FixedFingerTargets.from_mapping(targets),
                            initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(values),
                            output=output, observation_callback=observe)
                    report = json.loads((output/"execution-report.json").read_text())
            self.assertEqual(report["history"][:2], previous)
            self.assertEqual(report["timing_result"], "T" if fresh else "U")
            current_true = [row for row in report["history"] if row["occurrence"] == 1 and row["value"] == "T"
                            and row["check_target"] == "effect"]
            self.assertEqual(len(current_true), 1 if fresh else 0)
            if fresh:
                self.assertAlmostEqual(current_true[0]["time_s"], .006)
                self.assertEqual({row["value"] for row in report["effect_history"]}, {"F", "U", "T"})
                self.assertEqual(report["continuation"]["request_ledger"]["effect_history"], report["effect_history"])

    def test_relative_future_origin_rechecks_the_retained_actual_activation(self):
        simulation, curve, targets = _fixture()
        curve.events[2].update(kind="function", requires_evidence=None, stop=False)
        curve.events[0]["timing"] = [self.window("activation", -.008, -.008,
            origin_time_s=None, origin_event={"skill_id": "declared_skill", "event_id": "hold-later",
                                             "occurrence": 0, "check_target": "activation"})]
        _bind_fingers(simulation, curve, targets)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                output=Path(temporary)/"trial")
        self.assertEqual(report["status"], "curve_finished")
        self.assertEqual(report["timing_result"], "T")
        checks = [row for row in report["timing_checks"] if row["event_id"] == "open-original"]
        self.assertEqual({row["value"] for row in checks}, {"U", "T"})
        actual = next(row for row in report["history"] if row["event_id"] == "open-original")
        self.assertAlmostEqual(actual["time_s"], .002)

    def test_early_effect_state_is_not_a_witness_inside_a_later_window(self):
        simulation, curve, targets = _fixture()
        curve.events[2].update(kind="effect", requires_evidence=None, stop=False,
            timing=[self.window("effect", .005, .009, condition_id="touch")])
        curve.conditions = ({"condition_id": "touch", "effect_requirement": "windowed contact",
                             "skill_id": "declared_skill", "event_id": "holding-original"},)
        _bind_fingers(simulation, curve, targets)
        limits = _limit_values(simulation, curve)
        limits["observation_policy"].update(period_s=.1, maximum_age_s=.1)
        def observe(feedback, curve_time):
            return {"source": "single early physical sample", "timestamp_s": feedback["timestamp_s"],
                    "effects": {"touch": {"value": "T", "source": "predicate true before window opens"}}}
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            with self.assertRaises(ExecutionGuardError):
                _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                    initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(limits),
                    output=output, observation_callback=observe)
            report = json.loads((output/"execution-report.json").read_text())
        self.assertEqual(report["timing_result"], "U")
        self.assertEqual(report["evaluated_events"][0]["evaluation"], "U")
        self.assertEqual(report["effect_history"][0]["value"], "T")
        self.assertFalse(any(row["value"] == "T" and row["check_target"] == "effect" for row in report["history"]))

    def test_dependency_cannot_borrow_true_state_acquired_before_required_effect_window(self):
        simulation, curve, targets = _fixture()
        curve.events[2]["timing"] = [self.window("effect", .007, .009, condition_id="Holding")]
        _bind_fingers(simulation, curve, targets)
        limits = _limit_values(simulation, curve)
        limits["observation_policy"].update(period_s=.1, maximum_age_s=.1)
        def observe(feedback, curve_time):
            return {"source": "early independently measured state", "timestamp_s": feedback["timestamp_s"],
                    "effects": {"Holding": {"value": "T", "source": "true before required acquisition window"}}}
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                initial_feedback=_initial_feedback(simulation), execution_limits=ExecutionLimits.from_mapping(limits),
                output=Path(temporary)/"trial", observation_callback=observe)
        self.assertEqual(report["status"], "stopped_for_evidence")
        self.assertEqual(report["dependency_outcomes"][0]["evidence"]["value"], "U")
        self.assertEqual(report["timing_result"], "U")
        self.assertTrue(report["dependency_hold"]["native_ctrl_written"])
        self.assertNotIn("hold-later", [row["event_id"] for row in report["events"]])

    def test_occurred_event_keeps_original_witness_time_and_rejects_borrowed_retry_identity(self):
        for witness_occurrence, witness_time, outcome in ((1, 1.304, "valid"), (0, 1.304, "borrowed"), (1, 1.302, "outside")):
            simulation, curve, targets = _fixture()
            event = curve.events[2]
            event.update(kind="effect", requires_evidence=None, stop=False, occurrence=1,
                timing=[self.window("effect", .004, .006, condition_id="touch_happened")])
            curve.conditions = ({"condition_id": "touch_happened", "effect_requirement": "contact happened once",
                "evidence_kind": "occurred_event", "skill_id": event["skill_id"], "event_id": event["event_id"]},)
            _bind_fingers(simulation, curve, targets)
            def observe(feedback, curve_time):
                fact = {"value": "U", "source": "explicit event log assessment", "evidence_kind": "occurred_event"}
                if curve_time >= .004:
                    fact.update(value="T", witness_timestamp_s=witness_time,
                        event={"skill_id": event["skill_id"], "event_id": event["event_id"], "occurrence": witness_occurrence},
                        witness={"force_n": .02, "source": "synthetic physical sample at time 1.304"})
                return {"source": "new reading of an explicitly declared immutable event fact",
                    "timestamp_s": feedback["timestamp_s"], "effects": {"touch_happened": fact}}
            with self.subTest(witness_occurrence=witness_occurrence, outcome=outcome), TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
                output = Path(temporary)/"trial"
                if outcome == "valid":
                    report = _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                        initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                        output=output, observation_callback=observe)
                else:
                    expected = ValueError if outcome == "borrowed" else ExecutionGuardError
                    with self.assertRaises(expected):
                        _execute_bound(_executor(simulation), _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                            initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                            output=output, observation_callback=observe)
                    report = json.loads((output/"execution-report.json").read_text())
            if outcome == "valid":
                self.assertEqual(report["timing_result"], "T")
                witness = next(row for row in report["history"] if row["check_target"] == "effect" and row["value"] == "T")
                self.assertAlmostEqual(witness["time_s"], .004)
                latest = report["effect_history"][-1]
                self.assertAlmostEqual(latest["timestamp_s"], 1.304)
                self.assertGreater(latest["observed_at_s"], latest["timestamp_s"])
                self.assertEqual(report["evaluated_events"][0]["evaluation"], "T")
            else:
                self.assertTrue(report["abort_hold"]["native_ctrl_written"])
                if outcome == "borrowed":
                    self.assertFalse(any(row["value"] == "T" and row["check_target"] == "effect" for row in report["history"]))
                else:
                    self.assertEqual(report["timing_result"], "F")
                    actual = next(row for row in report["history"] if row["check_target"] == "effect" and row["value"] == "T")
                    self.assertAlmostEqual(actual["time_s"], .002)

    def test_retry_satisfies_current_occurrence_without_erasing_previous_unknown_window(self):
        simulation, curve, targets = _fixture()
        event = curve.events[2]
        event.update(kind="effect", requires_evidence=None, stop=False, occurrence=1,
            timing=[self.window("effect", 0., .015, condition_id="touch")])
        curve.conditions = ({"condition_id": "touch", "effect_requirement": "current contact",
                             "skill_id": event["skill_id"], "event_id": event["event_id"]},)
        _bind_fingers(simulation, curve, targets)
        result = _result(curve)
        previous_event = {**deepcopy(event), "occurrence": 0}
        previous_check = {"skill_id": event["skill_id"], "event_id": event["event_id"], "occurrence": 0,
            "check_target": "effect", "constraint_index": 0, "time_s": .003, "value": "U",
            "source": "old attempt ended without actual contact"}
        result.execution_binding.update(request_time_origin_s=.004,
            timing_events=[previous_event, deepcopy(event)], timing_checks=[previous_check])
        def observe(feedback, curve_time):
            return {"source": "new occurrence upstream sample", "timestamp_s": feedback["timestamp_s"],
                    "effects": {"touch": {"value": "T" if curve_time >= .006 else "U", "source": "new measured predicate"}}}
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _execute_bound(_executor(simulation), result, finger_targets=FixedFingerTargets.from_mapping(targets),
                initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                output=Path(temporary)/"trial", observation_callback=observe)
        self.assertEqual(report["timing_result"], "T")
        self.assertEqual(report["timing_checks"][0], previous_check)
        self.assertEqual(report["continuation"]["timing_events"], [previous_event, event])
        self.assertFalse(report["continuation"]["requires_rebind"])

    def test_continuation_resolves_old_actual_witness_without_reissuing_its_command(self):
        simulation, curve, targets = _fixture()
        # BEGIN a resumed occurrence inherits all four immutable device-policy fields.
        original_policy = {"finger_targets": deepcopy(targets), "execution_limits": _limits(simulation, curve).describe(),
            "controller_policy": simulation.describe()["controller_policy"], "result_only_goal_ids": []}
        # END no position-only authorization is manufactured by the continuation fixture.
        previous_event = deepcopy(curve.events[0])
        previous_event["timing"] = [self.window("activation", -.010, -.010, origin_time_s=None,
            origin_event={"skill_id": "declared_skill", "event_id": "close-original",
                          "occurrence": 0, "check_target": "activation"})]
        previous_fact = {"skill_id": "declared_skill", "event_id": "open-original", "occurrence": 0,
            "check_target": "activation", "time_s": .002, "value": "T", "source": "old actual actuator write"}
        previous_check = {**previous_fact, "constraint_index": 0, "value": "U", "evidence_value": "T",
            "origin_event": previous_event["timing"][0]["origin_event"], "reason": "Waiting for later actual close"}
        curve.events = tuple(curve.events[1:])
        curve.events[1].update(kind="function", requires_evidence=None, stop=False)
        targets["events"] = targets["events"][1:]
        _bind_fingers(simulation, curve, targets)
        result = _result(curve)
        result.execution_binding.update(request_time_origin_s=.008, timing_events=[previous_event],
            timing_checks=[previous_check], request_ledger={"history": [previous_fact]}, device_policy=original_policy)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _execute_bound(_executor(simulation), result, finger_targets=FixedFingerTargets.from_mapping(targets),
                initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                output=Path(temporary)/"trial")
        self.assertEqual(report["timing_result"], "T")
        self.assertEqual(report["history"][0], previous_fact)
        self.assertEqual([row["event_id"] for row in report["events"]], ["close-original", "hold-later"])

    def test_explicit_retry_reselects_original_device_policy_without_replaying_other_commands(self):
        from manipulation.execution import continuation_execution_inputs
        simulation, curve, targets = _fixture()
        result = _result(curve)
        result.execution_binding["request_ledger"] = {"original_events": deepcopy(list(curve.events)), "history": []}
        fixed, limits = FixedFingerTargets.from_mapping(targets), _limits(simulation, curve)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _execute_bound(_executor(simulation), result, finger_targets=fixed,
                initial_feedback=_initial_feedback(simulation), execution_limits=limits, output=Path(temporary)/"trial")
        retried, retried_limits = continuation_execution_inputs(report["continuation"], fixed, limits,
            rebind={"kind": "retry", "source": "explicit same-goal executor decision",
                    "retry_event_ids": [["declared_skill", "close-original"]], "retry_goal_ids": []})
        self.assertEqual([row["event_id"] for row in retried.events], ["close-original", "hold-later"])
        self.assertEqual(retried_limits.initial_fingers, report["continuation"]["active_finger_interval"])
        self.assertEqual(report["continuation"]["device_policy"]["finger_targets"], targets)

        # A caller bypassing the selection helper still cannot sneak a new
        # target through a freshly accepted candidate and its new reference.
        next_backend, next_curve, changed = _fixture()
        changed["events"][1]["positions"][0] += .00001
        next_curve.events[1]["occurrence"] = 1
        _bind_fingers(next_backend, next_curve, changed)
        next_result = _result(next_curve)
        next_result.execution_binding["device_policy"] = deepcopy(report["continuation"]["device_policy"])
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "original actuator policy"):
                _execute_bound(_executor(next_backend), next_result, finger_targets=FixedFingerTargets.from_mapping(changed),
                    initial_feedback=_initial_feedback(next_backend), execution_limits=_limits(next_backend, next_curve),
                    output=Path(temporary)/"trial")
        self.assertEqual(next_backend.arm_goals, [])
# END deadline failures cannot be hidden by a successful reference-clock lookup.


# BEGIN explicit trajectory-device rebinding and terminal result-only obligations.
class TrajectoryDeviceBindingTests(unittest.TestCase):
    @staticmethod
    def bind(simulation, result, targets, limits, result_ids=()):
        from manipulation.execution_binding import rebind_execution_result

        return rebind_execution_result(result,
            controller_policy=simulation.describe()["controller_policy"],
            result_only_goal_ids=list(result_ids), finger_targets=FixedFingerTargets.from_mapping(targets),
            execution_limits=limits, source={"kind": "explicit CPU contract test", "input": "unchanged formal fixture"})

    def test_source_and_old_position_policy_never_authorize_control_without_explicit_rebind(self):
        simulation, curve, targets = _fixture()
        fixed, limits = FixedFingerTargets.from_mapping(targets), _limits(simulation, curve)
        for old_policy in (None, {"finger_targets": targets, "execution_limits": limits.describe()}):
            result = _result(curve)
            if old_policy is not None:
                result.execution_binding["device_policy"] = old_policy
            with self.subTest(policy=old_policy is not None), TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, "explicit complete device rebind"):
                    _executor(simulation).execute(result, finger_targets=fixed, execution_limits=limits,
                        initial_feedback=_initial_feedback(simulation), output=Path(temporary)/"trial")
        self.assertEqual(simulation.arm_goals, [])
        self.assertEqual(simulation.sim_time, 1.3)

    def test_rebind_preserves_original_curve_proofs_and_every_consumed_planning_counter(self):
        simulation, curve, targets = _fixture()
        source = _result(curve)
        source.execution_binding["execution_limits"] = _limits(simulation, curve).describe()
        ledger = {"request_id": "same request", "requires_continuation": False,
            "planning_budget": {"maximum_trajectory_solves": 4, "planning_seconds": 240.},
            "consumed_planning": {"trajectory_solves": 3, "planning_seconds": 126.85213026600104},
            "history": [{"source": "preserved prior planning fact"}]}
        source.execution_binding["request_ledger"] = deepcopy(ledger)
        original = deepcopy(source.execution_binding)
        values = _limits(simulation, curve).describe()
        values.update(source="Same physical limits; optional observation channel removed", observation_policy=None)
        rebound = self.bind(simulation, source, targets, ExecutionLimits.from_mapping(values))
        self.assertEqual(source.execution_binding, original)
        self.assertIs(rebound.trajectory, source.trajectory)
        self.assertNotEqual(rebound.execution_binding["candidate_id"], original["candidate_id"])
        self.assertEqual(rebound.execution_binding["validation"]["source_candidate_id"], original["candidate_id"])
        self.assertEqual(rebound.execution_binding["request_ledger"], ledger)
        self.assertEqual(rebound.request_ledger, ledger)
        for key in ("curve", "problem", "scene"):
            self.assertEqual(rebound.execution_binding[key], original[key])
        for key in ("native", "export", "inputs"):
            self.assertEqual(rebound.execution_binding["validation"][key], original["validation"][key])
        self.assertEqual(simulation.arm_goals, [])

    def test_rebind_cannot_relax_physical_or_required_observation_limits_or_reset_an_execution(self):
        for change in ("physical", "required_observation", "executed"):
            simulation, curve, targets = _fixture()
            source = _result(curve)
            old = _limits(simulation, curve).describe()
            if change == "required_observation":
                old["observation_policy"]["required_fields"] = ["binding_valid"]
            source.execution_binding["execution_limits"] = deepcopy(old)
            changed = deepcopy(old)
            if change == "physical":
                changed["arm_position_error"][0] += .001
            elif change == "required_observation":
                changed["observation_policy"] = None
            else:
                source.execution_binding["request_ledger"] = {"requires_continuation": True}
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.bind(simulation, source, targets, ExecutionLimits.from_mapping(changed))
            self.assertEqual(simulation.arm_goals, [])

    def test_changed_controller_gain_or_phase_is_rejected_before_preflight_or_control(self):
        for key, value in (("kp", [2.]*7), ("sampling_phase", "step_start"),
                           ("reference_derivatives", ["q", "dq"]), ("kd", [float("nan")]*7)):
            simulation, curve, targets = _fixture()
            result = self.bind(simulation, _result(curve), targets, _limits(simulation, curve))
            result.execution_binding["device_policy"]["controller_policy"][key] = value
            executor = _executor(simulation)
            with self.subTest(key=key), TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                executor.execute(result, finger_targets=FixedFingerTargets.from_mapping(targets),
                    execution_limits=_limits(simulation, curve), initial_feedback=_initial_feedback(simulation),
                    output=Path(temporary)/"trial")
            self.assertEqual(executor.geometry_checker.preflight_inputs, [])
            self.assertEqual(simulation.arm_goals, [])

    def test_endpoint_q_dq_ddq_and_torque_terms_keep_the_same_clipped_reference_clock(self):
        simulation, old, targets = _fixture()
        old.events[2].update(kind="effect", requires_evidence=None, stop=False)
        duration, acceleration = .011, np.full(7, .05)
        q, dq = old.sample(0.)[:2]
        curve = JointTrajectory(old.joint_names, np.asarray([0., duration]),
            np.stack((q, q+dq*duration+.5*acceleration*duration**2)),
            np.stack((dq, dq+acceleration*duration)), old.progress_u, old.progress_s,
            old.events, old.conditions, {})
        _bind_fingers(simulation, curve, targets)
        initial = _initial_feedback(simulation)
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            output = Path(temporary)/"trial"
            report = _execute_bound(_executor(simulation), _result(curve),
                finger_targets=FixedFingerTargets.from_mapping(targets), execution_limits=_limits(simulation, curve),
                initial_feedback=initial, output=output)
            with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["curve_time"][-1], duration)
                self.assertEqual(saved["elapsed"][-1], .012)
                query = np.minimum(saved["control_elapsed"]+simulation.timestep, duration)
                expected = curve.sample(query)
                for key, values in zip(("reference_arm_q", "reference_arm_dq", "reference_arm_ddq"), expected[:3]):
                    np.testing.assert_array_equal(saved[key], values)
                np.testing.assert_array_equal(saved["arm_curve_ddq"], curve.sample(saved["curve_time"])[2])
                terms = [saved[name] for name in ("arm_position_feedback_torque", "arm_velocity_feedback_torque",
                    "arm_velocity_feedforward_torque", "arm_acceleration_feedforward_torque", "arm_bias_torque")]
                np.testing.assert_allclose(saved["requested_arm_torque"], sum(terms), rtol=0., atol=1e-15)
                self.assertNotIn("actual_qacc", saved)
                self.assertNotIn("actual_actuator_force", saved)
        self.assertEqual(report["status"], "curve_finished")
        np.testing.assert_array_equal(simulation.native_data.qvel, initial["dq"])

    def test_result_only_missing_or_false_terminal_pose_keeps_tail_without_claiming_completion(self):
        for kind in ("missing_owner", "measured_miss"):
            relation = ({"target_owner_id": "current_part", "T_owner_target": np.eye(4).tolist()}
                        if kind == "missing_owner" else None)
            simulation, curve, targets, source, limits = ActualPoseGoalTests.fixture(progress=1., actual_relation=relation)
            curve.progress_u, curve.progress_s = np.asarray([0., .5, 1.]), np.asarray([0., 1., 1.])
            from manipulation.execution_binding import curve_snapshot
            source.execution_binding["curve"] = curve_snapshot(curve)
            if kind == "measured_miss":
                simulation.scripted_tcp_estimate_offset[0] = .002
            result = self.bind(simulation, source, targets, limits, [["pose_events", 7]])
            with self.subTest(kind=kind), TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
                report = _executor(simulation).execute(result, finger_targets=FixedFingerTargets.from_mapping(targets),
                    execution_limits=limits, initial_feedback=_initial_feedback(simulation), output=Path(temporary)/"trial")
            self.assertEqual(report["status"], "curve_finished")
            self.assertEqual(report["executed_steps"], 6)
            self.assertEqual(report["pose_goal_result"], "U" if kind == "missing_owner" else "F")
            self.assertIsNone(report["execution_pose_goal_result"])
            self.assertEqual(report["completed_goal_ids"], [])
            self.assertEqual(len(report["pose_goal_checks"]), 1)
            self.assertEqual(report["pose_goal_checks"][0]["physical_elapsed_s"], .006)
            self.assertFalse(report["pose_goal_checks"][0]["required_for_execution"])
            self.assertFalse(report["continuation"]["requires_rebind"])
            self.assertEqual(report["continuation"]["device_policy"], result.execution_binding["device_policy"])

    def test_result_only_selection_cannot_remove_nonterminal_or_arrival_obligations(self):
        for kind in ("nonterminal", "arrival", "unknown", "duplicate"):
            simulation, curve, targets, result, limits = ActualPoseGoalTests.fixture(progress=.5 if kind == "nonterminal" else 1.)
            selected = [["pose_events", 7]]
            if kind == "arrival":
                curve.events[2]["timing"] = [PhysicalEventTimingTests.window("arrival", 0., .012, goal_id=["pose_events", 7])]
                from manipulation.execution_binding import curve_snapshot
                result.execution_binding["curve"] = curve_snapshot(curve)
            elif kind == "unknown":
                selected = [["pose_events", 8]]
            elif kind == "duplicate":
                selected *= 2
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.bind(simulation, result, targets, limits, selected)
            self.assertEqual(simulation.arm_goals, [])

    def test_result_only_goal_does_not_release_an_independent_unknown_effect_dependency(self):
        from manipulation.execution_binding import curve_snapshot

        simulation, curve, targets, source, limits = ActualPoseGoalTests.fixture(progress=1.,
            actual_relation={"target_owner_id": "current_part", "T_owner_target": np.eye(4).tolist()})
        curve.progress_u, curve.progress_s = np.asarray([0., .5, 1.]), np.asarray([0., 1., 1.])
        curve.events[2].update(kind="verification", requires_evidence="Holding", stop=True)
        _bind_fingers(simulation, curve, targets)
        source.execution_binding["curve"] = curve_snapshot(curve)
        result = self.bind(simulation, source, targets, limits, [["pose_events", 7]])
        with TemporaryDirectory() as temporary, patch("manipulation.robosuite_execution.mujoco.mj_forward"):
            report = _executor(simulation).execute(result, finger_targets=FixedFingerTargets.from_mapping(targets),
                execution_limits=limits, initial_feedback=_initial_feedback(simulation), output=Path(temporary)/"trial")
        self.assertEqual(report["status"], "stopped_for_evidence")
        self.assertEqual(report["executed_steps"], 4)
        self.assertEqual(report["pose_goal_result"], "U")
        self.assertTrue(report["dependency_hold"]["native_ctrl_written"])
        self.assertNotIn("hold-later", [row["event_id"] for row in report["events"]])
        self.assertEqual(report["completed_goal_ids"], [])
# END result grading cannot change online control prerequisites or original mathematical goals.


# BEGIN task-set enlargement preserves stricter proof and independent tolerances.
class TaskToleranceRebindingTests(unittest.TestCase):
    @staticmethod
    def source_result():
        simulation, curve, targets = _fixture()
        result = _result(curve)
        binding = result.execution_binding
        goals = [{"progress": progress, "tool_from_function": np.eye(4).tolist(),
            "target": np.eye(4).tolist(), "position_tolerance_m": .001,
            "rotation_tolerance_rad": float(np.pi/180),
            "source": {"skill_id": skill, "kind": kind}, "actual_relation": None}
            for progress, skill, kind in ((.4, "grasp_instance", "arrival"),
                (.7, "grasp_instance", "relation"), (1., "manipulation_instance", "functional_goal"))]
        binding["problem"].update(pose_events=goals, numerical={"position_m": 1e-5})
        binding["goal_identities"] = {"pose_events": [0, 1, 2]}
        binding["validation"]["inputs"] = deepcopy({key: binding[key] for key in ("problem", "scene")})
        binding["execution_limits"] = _limits(simulation, curve).describe()
        binding["request_ledger"] = {"request_id": "original_request", "requires_continuation": False,
            "consumed_planning": {"trajectory_solves": 2, "validation_queries": 17},
            "planning_budget": {"maximum_trajectory_solves": 4},
            "obligation_contract": {"pose_events": deepcopy(goals)}, "history": []}
        return simulation, result, targets

    @staticmethod
    def changes():
        return [{"goal_id": ["pose_events", index], "position_tolerance_m": .002,
                 "rotation_tolerance_rad": float(np.pi/180)} for index in (0, 1)]

    def test_selected_task_goals_change_while_old_proof_curve_and_limits_remain(self):
        from manipulation.execution_binding import check_result, relax_pose_goal_tolerances
        from manipulation.execution import _actual_pose_goal
        from manipulation.planning_problem import PoseEventRequirement

        _, source, _ = self.source_result()
        original = deepcopy(source.execution_binding)
        revised = relax_pose_goal_tolerances(source, changes=self.changes(), source="Explicit user task revision")
        binding = check_result(revised)
        self.assertEqual(source.execution_binding, original)
        self.assertIs(revised.trajectory, source.trajectory)
        for name in ("curve", "scene", "execution_limits"):
            self.assertEqual(binding[name], original[name])
        for name in ("inputs", "native", "export"):
            self.assertEqual(binding["validation"][name], original["validation"][name])
        self.assertEqual(binding["problem"]["pose_events"][2], original["problem"]["pose_events"][2])
        self.assertEqual(binding["problem"]["numerical"], original["problem"]["numerical"])
        self.assertNotEqual(binding["request_ledger"]["request_id"], original["request_ledger"]["request_id"])
        self.assertEqual(binding["request_ledger"]["consumed_planning"], original["request_ledger"]["consumed_planning"])
        self.assertEqual(binding["request_ledger"]["planning_budget"], original["request_ledger"]["planning_budget"])
        pose = np.eye(4); pose[0, 3] = .0015
        feedback = {"tool_pose": pose, "tool_pose_source": "measured-joint FK", "source": "synthetic FK test",
            "tool_pose_is_independent_observation": False, "timestamp_s": 1., "feedback_id": "same sample"}
        old_goal = PoseEventRequirement(**original["problem"]["pose_events"][0])
        new_goal = PoseEventRequirement(**binding["problem"]["pose_events"][0])
        self.assertEqual(_actual_pose_goal(old_goal, feedback, None)["value"], "F")
        self.assertEqual(_actual_pose_goal(new_goal, feedback, None)["value"], "T")

    def test_reject_shrinking_nonfinite_duplicate_unknown_and_unchanged_requests(self):
        from manipulation.execution_binding import relax_pose_goal_tolerances

        for kind in ("smaller_position", "smaller_angle", "nan", "pi", "boolean", "duplicate", "unknown", "extra", "same"):
            _, source, _ = self.source_result()
            changes = self.changes()
            if kind == "smaller_position": changes[0]["position_tolerance_m"] = .0005
            elif kind == "smaller_angle": changes[0]["rotation_tolerance_rad"] = .001
            elif kind == "nan": changes[0]["position_tolerance_m"] = float("nan")
            elif kind == "pi": changes[0]["rotation_tolerance_rad"] = float(np.pi)
            elif kind == "boolean": changes[0]["position_tolerance_m"] = True
            elif kind == "duplicate": changes[1]["goal_id"] = changes[0]["goal_id"]
            elif kind == "unknown": changes[0]["goal_id"] = ["pose_events", 99]
            elif kind == "extra": changes[0]["target"] = np.eye(4).tolist()
            else:
                for change in changes: change["position_tolerance_m"] = .001
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                relax_pose_goal_tolerances(source, changes=changes, source="Explicit test")

    def test_tampered_geometry_goal_or_ledger_cannot_claim_set_inclusion(self):
        from manipulation.execution_binding import check_result, relax_pose_goal_tolerances

        for kind in ("target", "numerical", "scene", "other_goal", "spent_budget"):
            _, source, _ = self.source_result()
            revised = relax_pose_goal_tolerances(source, changes=self.changes(), source="Explicit test")
            binding = revised.execution_binding
            if kind == "target": binding["problem"]["pose_events"][0]["target"][0][3] = .001
            elif kind == "numerical": binding["problem"]["numerical"]["position_m"] = .002
            elif kind == "scene": binding["scene"]["source"] = "different geometry"
            elif kind == "other_goal": binding["problem"]["pose_events"][2]["position_tolerance_m"] = .002
            else: binding["request_ledger"]["consumed_planning"]["trajectory_solves"] = 0
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                check_result(revised)

    def test_executed_or_already_derived_request_requires_its_own_new_validation(self):
        from manipulation.execution_binding import relax_pose_goal_tolerances

        for kind in ("physics", "continuation", "completed_goal", "derived"):
            _, source, _ = self.source_result()
            if kind == "physics": source.execution_binding["consumed_physics_steps"] = 1
            elif kind == "continuation": source.execution_binding["request_ledger"]["requires_continuation"] = True
            elif kind == "completed_goal": source.execution_binding["completed_goal_ids"] = [["pose_events", 0]]
            else: source = relax_pose_goal_tolerances(source, changes=self.changes(), source="First revision")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                relax_pose_goal_tolerances(source, changes=self.changes(), source="Explicit test")

    def test_archive_round_trip_and_device_rebinding_retain_original_task_evidence(self):
        from manipulation.execution_binding import check_result, relax_pose_goal_tolerances
        from recorded_task_planning import save_curve

        simulation, source, targets = self.source_result()
        revised = relax_pose_goal_tolerances(source, changes=self.changes(), source="Explicit test")
        limits = _limits(simulation, source.trajectory)
        revised = TrajectoryDeviceBindingTests.bind(simulation, revised, targets, limits)
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"trajectory.npz"
            save_curve(path, revised.trajectory, classification="planner_accepted",
                       execution_limits=limits.describe(), planning_result=revised)
            restored = load_saved_trajectory(path)
            check_result(restored)
            self.assertEqual(restored.execution_binding, revised.execution_binding)
            np.testing.assert_array_equal(restored.trajectory._curve.c, source.trajectory._curve.c)
            self.assertEqual(restored.execution_binding["validation"]["inputs"]["problem"]["pose_events"][0]["position_tolerance_m"], .001)
            self.assertEqual(restored.execution_binding["problem"]["pose_events"][0]["position_tolerance_m"], .002)
# END a task edit never masquerades as a new solve or controller calibration.


if __name__ == "__main__":
    unittest.main()
# END independent fixed-finger execution contract regressions.
