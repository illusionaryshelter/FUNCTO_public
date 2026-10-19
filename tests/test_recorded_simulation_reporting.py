"""CPU-only failure reporting; all physics, evaluators and renderers are doubles."""

# BEGIN a large generated report must not replace the original execution error.
from contextlib import ExitStack
# BEGIN live-session tests preserve caller-owned robot feedback by value.
from copy import deepcopy
# END mutations inside capture callbacks must not edit execution records.
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from test_recorded_simulation import (
    _fixture, _initial_feedback, _limit_values, _result, _executor, _limits, _execute_bound,
    # BEGIN reuse the existing CPU finger-reference adapter for each new request.
    _bind_fingers,
    # END no native physical state is synthesized by the runner tests.
)
from manipulation.execution import ExecutionGuardError, FixedFingerTargets, _feedback_json
import run_recorded_simulation as runner


# BEGIN shared explicit trial metadata for evidence-isolation CPU doubles.
def _runner_inputs(tmp_path, simulation, curve, targets, initial, *, skill_id="declared_skill"):
    def save(name, value):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        # BEGIN production feedback retains NumPy calibration arrays in memory.
        path.write_text(json.dumps(_feedback_json(value)))
        # END disk fixtures use the same explicit conversion as runner reports.
        return path

    fixture = {"joint_names": list(simulation.joint_names),
        "arm_joint_names": list(simulation.arm_joint_names), "finger_joint_names": list(simulation.finger_joint_names),
        "tool_link": simulation.tool_link, "base_link": "test_base", "tcp_site": simulation.tool_link,
        "episode_path": str(tmp_path/"recording"/"episode.h5"), "source_scene_xml": "unused_test_scene",
        "catalog_path": str(save("catalog.json", {})), "velocity_limits": simulation.velocity_limits}
    feedback = {**fixture, "segment_id": skill_id, "entry_frame": 0,
        "reference_frame": "world", "tool_transform": initial["tool_transform"],
        "start_q": initial["q"].tolist(), "start_dq": initial["dq"].tolist(),
        "base_pose": initial["base_pose"].tolist(), "eef_pose": initial["tool_pose"].tolist(),
        # BEGIN recorded metadata retains the same original measured sample.
        "sim_time_s": initial["sim_time"], "timestamp_s": initial["timestamp_s"],
        "feedback_id": initial["feedback_id"], "source": initial["source"]}
        # END moving a feedback file cannot change its reference identity.
    save("program.json", {"skill_kind": "grasp", "demo": {"roles": {}}})
    manifest = {"workspace_root": str(tmp_path), "source_recording": "recording", "segments": [{
        "id": skill_id, "program_path": "program.json", "source_library_segment": skill_id,
        "snapshot": {"entry_frame": 0, "reference_frame": "world", "views": [{"id": "test_view"}]}}]}
    save("recording/manifest.json", {"environment": {"control_freq_hz": 20, "controller_config": {"body_parts": {
        "test_arm": {"type": "JOINT_POSITION", "input_type": "absolute", "gripper": {"type": "GRIP"},
                     "interpolation": None, "kp": 1., "kd": 1.}}}}})
    evidence = {"actor": {"entity_name": "test_robot", "eef_site_name": simulation.tool_link,
        "contact_geom_groups": []}, "segments": [{"segment_id": skill_id, "interaction": "grasp",
        "carrier_body_name": "test_owner"}], "evaluation_owner_bindings": []}
    args = SimpleNamespace(output=tmp_path/"trial", trajectory=tmp_path/"formal.npz",
        manifest=save("manifest.json", manifest), robot_fixture=save("fixture.json", fixture),
        feedback=save("feedback.json", feedback), finger_targets=save("targets.json", targets),
        execution_limits=save("limits.json", _limit_values(simulation, curve)),
        evaluator_evidence=save("evidence.json", evidence),
        # BEGIN the test caller supplies the explicit independent scoring policy.
        result_only_goal_ids=[], diagnostic_goal_ids=[],
        # END no production compatibility default is needed for an old test double.
        segment=skill_id, controller_part="test_arm", views=["test_view"], no_video=False,
        max_physics_steps=30000, start_position_tolerance=2e-6, start_velocity_tolerance=2e-6)
    result = _result(curve)
    result.diagnostics["archive_classification"] = "planner_accepted"

    return args, result
# END no native scene, integration or renderer is constructed by this fixture.


@pytest.mark.parametrize("cleanup_failure", [False, True])
@pytest.mark.parametrize("archive_write_failure", [False, True])
# BEGIN summary counts distinguish confirmed motion, charged attempts and unknown clocks.
@pytest.mark.parametrize("normal_steps,hold_steps,normal_charge,hold_charge,normal_exact,hold_exact", [
    (0, 1, 0, 1, True, True),
    (0, 0, 1, 0, True, True),  # A charged attempt provably did not advance time.
    (0, 0, 1, 0, False, True),  # No confirmed step does not prove no motion.
    (2, 1, 2, 2, True, False),  # One uncertain hold follows known native motion.
])
def test_large_generated_report_is_reused_and_primary_error_survives_cleanup(
        tmp_path, cleanup_failure, archive_write_failure,
        normal_steps, hold_steps, normal_charge, hold_charge, normal_exact, hold_exact):
    simulation, curve, targets = _fixture()
    initial = _initial_feedback(simulation)
    primary = RuntimeError("original independent geometry rejection")
    confirmed, charged = normal_steps+hold_steps, normal_charge+hold_charge
    accounting_exact = normal_exact and hold_exact
    accounting = {"physical_steps": confirmed, "charged_steps": charged,
        "step_accounting_exact": accounting_exact, "execution_step_limit": 30000,
        "remaining_charged_steps": 30000-charged, "scope": "this execute invocation"}
    shared_report = {
        "status": "feedback_rejected", "executed_steps": normal_steps, "barrier_command_reached": False,
        "elapsed_sim_s": .002,
        "abort_hold": {"physical_steps": hold_steps, "charged_steps": hold_charge,
            "step_accounting_exact": hold_exact}, "dependency_hold": None,
        "normal_step_accounting": {"physical_steps": normal_steps, "charged_steps": normal_charge,
            "step_accounting_exact": normal_exact, "attempts": []},
        "physics_step_accounting": accounting,
        "independent_evidence_sink": {"configured": True, "completed": True, "disabled": False,
            "first_error": None, "attempted_callbacks": normal_steps+1, "successful_callbacks": normal_steps+1},
        "robot_feedback_path": "actual_failure_feedback", "barrier_feedback_path": None,
        "geometry_preflight": {"passed": np.bool_(True), "checked_interval_s": [0., .0075],
            "reference_collision_checked": True, "continuous_collision_certified": False,
            "observed_scene_checked": True, "unknown_space_checked": False,
            "nominal_scene": {"states": "must not be duplicated into the small summary"}},
        "feedback_failure": {"point": "after_step", "curve_time_s": .002,
            "physical_elapsed_s": .002, "actual_feedback": initial,
            "violations": [{"kind": "hardware_joint_velocity", "name": "arm_0",
                "measured": np.float64(1.1), "maximum_absolute": 1.}]},
        "continuation": {"actual_feedback": initial},
        "large_geometry_diagnostic": "x"*(16*1024**2+1)}

    args, result = _runner_inputs(tmp_path, simulation, curve, targets, initial)

    class Executor:
        last_execution_report = None

        def execute(self, *args, **kwargs):
            # BEGIN the runner must preserve the archived finger reference binding.
            assert kwargs["initial_feedback"]["feedback_id"] == curve.finger_trajectory.feedback_id
            assert kwargs["initial_feedback"]["timestamp_s"] == curve.finger_trajectory.timestamp_s
            # END source file paths never replace measured sample identities.
            self.last_execution_report = shared_report
            for index in range(normal_steps+1):
                kwargs["step_callback"](simulation.native_model, simulation.native_data, {"elapsed": index*.002})
            path = kwargs["output"]/"execution-report.json"
            path.parent.mkdir()
            if archive_write_failure:
                shared_report["persistence_errors"] = [{"artifact": path.name, "error": "OSError: synthetic sink failure"}]
            else:
                path.write_text(json.dumps(_feedback_json(shared_report)))
            raise primary

    class Evaluator:
        finalized = False

        def observe(self, *args):
            pass

        def finalize(self, output, *, execution_report, collection_report):
            self.finalized = True
            assert execution_report is shared_report
            if cleanup_failure:
                raise ValueError("independent evaluator cleanup failure")
            return {"physics_outcome": "not_evaluated", "physics_success": None,
                "evidence_coverage": {"complete": normal_exact}}

    class Videos:
        records = []
        closed = False
        captures = []

        def capture(self, *args, **kwargs):
            self.captures.append((args, kwargs))
            if cleanup_failure and kwargs.get("final"):
                raise ValueError("final frame capture failure")

        def close(self):
            self.closed = True
            if cleanup_failure:
                raise ValueError("video close failure")

    executor, evaluator, videos = Executor(), Evaluator(), Videos()
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner, "load_saved_trajectory", return_value=result))
        stack.enter_context(patch.object(runner, "restore_recorded_simulation", return_value=(simulation, {})))
        stack.enter_context(patch.object(runner, "build_robot_validator", return_value=object()))
        stack.enter_context(patch.object(runner, "TrajectoryExecutor", return_value=executor))
        stack.enter_context(patch.object(runner, "NativeSkillPhysicsEvaluator", return_value=evaluator))
        stack.enter_context(patch.object(runner, "_TrialVideos", return_value=videos))
        reads = stack.enter_context(patch.object(runner, "_read", wraps=runner._read))
        with pytest.raises(RuntimeError) as raised:
            runner.run(args)
    assert raised.value is primary
    report_path = args.output/"execution"/"execution-report.json"
    if archive_write_failure:
        assert not report_path.exists()
    else:
        assert report_path.stat().st_size > 16*1024**2
    assert all(Path(call.args[0]) != report_path for call in reads.call_args_list)
    assert evaluator.finalized and videos.closed
    report = json.loads((args.output/"simulation-report.json").read_text())
    assert report["error"] == "RuntimeError: original independent geometry rejection"
    assert report["trajectory_executed_steps"] == normal_steps
    assert report["hold_physics_steps"] == report["evaluator_excluded_hold_physics_steps"] == hold_steps
    assert report["confirmed_physics_steps"] == confirmed
    assert report["charged_physics_steps"] == charged
    assert report["step_accounting_exact"] is accounting_exact
    assert report["physics_step_accounting"] == accounting
    assert report["actual_physics_steps"] == (confirmed if accounting_exact else None)
    assert report["execution_tested"] is (True if normal_steps else False if normal_exact else None)
    assert report["actual_physics_tested"] is (True if confirmed else False if accounting_exact else None)
    assert report["robot_feedback_path"] == "actual_failure_feedback"
    assert report["source_candidate_id"] == result.execution_binding["candidate_id"]
    assert report["execution_candidate_id"] != report["source_candidate_id"]
    assert Path(report["rebound_trajectory_path"]).is_file()
    assert report["geometry_preflight_summary"] == {"passed": True, "checked_interval_s": [0., .0075],
        "reference_collision_checked": True, "continuous_collision_certified": False,
        "observed_scene_checked": True, "unknown_space_checked": False}
    assert report["feedback_failure_summary"]["first_violation"] == {
        "kind": "hardware_joint_velocity", "name": "arm_0", "measured": 1.1,
        "maximum_absolute": 1.}
    assert report["feedback_failure_summary"]["actual_feedback"] == _feedback_json(initial)
    assert report["final_robot_feedback"] == _feedback_json(initial)
    assert bool(report["execution_persistence_errors"]) is archive_write_failure
    assert (args.output/"exception.txt").read_text().rstrip().endswith(str(primary))
    assert videos.captures[-1] == ((.002,), {"final": True})
    if cleanup_failure:
        assert "evaluation_error" in report and "video_error" in report
        assert "video_final_capture_error" in report
    else:
        assert report["video_final_elapsed_s"] == .002
    assert simulation.sim_time == 1.3  # No native integration occurs in these doubles.
# END failed writes and cleanup cannot collapse uncertain accounting into an exact count.


# BEGIN independent evidence failures never alter a physical execution verdict.
@pytest.mark.parametrize("physical_failure", [False, True])
@pytest.mark.parametrize("failure_stage", ["evaluator_metadata", "evaluator_init", "evaluator_initial",
    "evaluator_step", "evaluator_finalize", "video_init", "video_initial", "video_step",
    "video_final", "video_close"])
def test_evidence_failures_do_not_stop_execution_or_replace_physical_error(
        tmp_path, physical_failure, failure_stage):
    simulation, curve, targets = _fixture()
    initial = _initial_feedback(simulation)
    args, result = _runner_inputs(tmp_path, simulation, curve, targets, initial)
    primary = ExecutionGuardError("physical guard owns this stop")
    evidence_error = ValueError("independent acquisition failed")
    held = int(physical_failure)
    execution = {"status": "feedback_rejected" if physical_failure else "curve_finished",
        "executed_steps": 2, "barrier_command_reached": not physical_failure,
        "elapsed_sim_s": .004+held*.002,
        "abort_hold": None if not held else {"physical_steps": held}, "dependency_hold": None,
        "normal_step_accounting": {"physical_steps": 2, "charged_steps": 2, "step_accounting_exact": True},
        "physics_step_accounting": {"physical_steps": 2+held, "charged_steps": 2+held,
            "step_accounting_exact": True},
        "independent_evidence_sink": {"configured": True, "completed": True, "disabled": False,
            "first_error": None, "attempted_callbacks": 3, "successful_callbacks": 3},
        "robot_feedback_path": "named_robot_feedback", "barrier_feedback_path": None,
        "continuation": {"actual_feedback": initial}}
    calls, observations, captures = [], [], []

    class Executor:
        last_execution_report = None

        def execute(self, *unused, **kwargs):
            calls.append("execute")
            self.last_execution_report = execution
            for index in range(3):
                calls.append(index)
                kwargs["step_callback"](simulation.native_model, simulation.native_data, {"elapsed": index*.002})
            if physical_failure:
                raise primary
            return execution

    class Evaluator:
        def observe(self, model, data, record):
            observations.append(record["elapsed"])
            if ((failure_stage == "evaluator_initial" and record["elapsed"] == 0.)
                    or (failure_stage == "evaluator_step" and record["elapsed"] == .002)):
                raise evidence_error

        def finalize(self, output, *, execution_report, collection_report):
            assert execution_report is execution
            if failure_stage == "evaluator_finalize":
                raise evidence_error
            complete = collection_report["completed"]
            return {"physics_outcome": ("incomplete_evidence" if not complete else
                                        "not_evaluated" if physical_failure else "passed"),
                "physics_success": True if complete and not physical_failure else None,
                "evidence_coverage": {"complete": complete}}

    class Videos:
        records = []
        closed = False

        def capture(self, elapsed, *, final=False):
            captures.append((elapsed, final))
            if ((failure_stage == "video_initial" and elapsed == 0.)
                    or (failure_stage == "video_step" and elapsed == .002)
                    or (failure_stage == "video_final" and final)):
                raise evidence_error

        def close(self):
            self.closed = True
            if failure_stage == "video_close":
                raise evidence_error

    videos = Videos()
    if failure_stage == "evaluator_metadata":
        args.evaluator_evidence.write_text("invalid independent evaluator JSON")
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner, "load_saved_trajectory", return_value=result))
        stack.enter_context(patch.object(runner, "restore_recorded_simulation", return_value=(simulation, {})))
        stack.enter_context(patch.object(runner, "build_robot_validator", return_value=object()))
        stack.enter_context(patch.object(runner, "TrajectoryExecutor", return_value=Executor()))
        stack.enter_context(patch.object(runner, "NativeSkillPhysicsEvaluator", return_value=Evaluator(),
            side_effect=evidence_error if failure_stage == "evaluator_init" else None))
        stack.enter_context(patch.object(runner, "_TrialVideos", return_value=videos,
            side_effect=evidence_error if failure_stage == "video_init" else None))
        if physical_failure:
            with pytest.raises(ExecutionGuardError) as raised:
                runner.run(args)
            assert raised.value is primary
        else:
            runner.run(args)
    report = json.loads((args.output/"simulation-report.json").read_text())
    assert calls == ["execute", 0, 1, 2]
    assert report["status"] == ("error" if physical_failure else "completed")
    assert report["trajectory_executed_steps"] == 2
    assert report["hold_physics_steps"] == held
    assert report["charged_physics_steps"] == 2+held
    assert report["evidence_complete"] is False
    if physical_failure:
        assert report["error"] == f"ExecutionGuardError: {primary}"
    failed_sink = "evaluator" if failure_stage.startswith("evaluator") else "video"
    state = report["evidence_collection"][failed_sink]
    assert state["disabled"] and not state["completed"] and state["first_error"] is not None
    if failed_sink == "evaluator":
        assert report["physics_success"] is None
        assert report["physics_outcome"] == "incomplete_evidence"
        assert len(captures) == 4 and videos.closed
        if failure_stage in ("evaluator_metadata", "evaluator_init"):
            assert observations == []
            assessment = json.loads((args.output/"evaluation-report.json").read_text())
            assert assessment["evidence_coverage"]["missing_normal_sample_index_ranges"] == [[0, 2]]
        elif failure_stage == "evaluator_initial":
            assert observations == [0.]
        elif failure_stage == "evaluator_step":
            assert observations == [0., .002]
    else:
        # Video is an optional witness; complete native predicates remain
        # independently gradable, just as in an explicit --no-video run.
        assert report["physics_success"] is (None if physical_failure else True)
        assert observations == [0., .002, .004]
        if failure_stage == "video_init":
            assert captures == []
        elif failure_stage == "video_initial":
            assert captures == [(0., False)] and videos.closed
        elif failure_stage == "video_step":
            assert captures == [(0., False), (.002, False)] and videos.closed
    assert simulation.sim_time == 1.3
# END all execution and acquisition are CPU doubles; no native step is issued.


# BEGIN a partly failed camera tick retains only frames actually written.
def test_partial_video_tick_does_not_invent_peer_frames_or_timestamps():
    video = runner._TrialVideos.__new__(runner._TrialVideos)
    video._counter, video._stride, video._last_elapsed = 0, 1, None
    video._simulation = SimpleNamespace(native_data=SimpleNamespace(time=3.4))
    video._cv2 = SimpleNamespace(cvtColor=lambda rgb, conversion: rgb, COLOR_RGB2BGR=1)
    written = []

    def failed_camera(data):
        raise ValueError("second camera capture failed")

    video._entries = [{"camera": SimpleNamespace(capture=capture),
        "writer": SimpleNamespace(write=lambda frame, index=index: written.append(index)),
        "times": [], "record": {"frame_count": 0}}
        for index, capture in enumerate((lambda data: {"rgb": np.zeros((1, 1, 3))}, failed_camera))]
    with pytest.raises(ValueError, match="second camera capture failed"):
        video.capture(.1)
    assert written == [0]
    assert video._entries[0]["record"]["frame_count"] == 1
    assert video._entries[0]["times"] == [[.1, 3.4]]
    assert video._entries[1]["record"]["frame_count"] == 0
    assert video._entries[1]["times"] == []
# END no renderer, codec or synthetic replacement frame is used.


# BEGIN generic upstream NumPy booleans survive the executor and every sink.
@pytest.mark.parametrize("failed_sink", [None, "continuation.json", "physical-trajectory.npz",
                                       "failure-feedback.json", "execution-report.json", "all"])
def test_numpy_observation_survives_speed_rejection_and_independent_writes(tmp_path, failed_sink):
    simulation, curve, targets = _fixture()
    initial = _initial_feedback(simulation)
    original_step = simulation._sim.step
    step_count = 0

    def step():
        nonlocal step_count
        original_step()
        step_count += 1
        # BEGIN inject one physical failure, followed by the fixture's stop.
        # This test covers evidence persistence, not an endlessly overspeeding
        # plant; a completed hold must retain its original one-step accounting.
        simulation.native_data.qvel[0] = 1.1 if step_count == 1 else 0.
        # END the first measured sample still triggers the real hardware guard.

    def observe(feedback, curve_time):
        # Explicit generic upstream evidence exercises serialization only.
        # It neither reads native object state nor synthesizes a sensor.
        return {"source": "Synthetic CPU upstream observation",
            "timestamp_s": feedback["timestamp_s"],
            "effects": {"test:observed_state": {"value": "U",
                "source": "Synthetic CPU observation quality flags",
                "metrics": {"measurement_valid": np.bool_(False), "frame_received": np.bool_(True)}}}}

    simulation._sim.step = step
    executor = _executor(simulation)
    primary = ExecutionGuardError("original measured hardware speed rejection")
    write_error = OSError("injected evidence sink failure")
    attempted = []
    write_text, save_npz = Path.write_text, np.savez_compressed

    def attempt(path, write):
        attempted.append(Path(path).name)
        if failed_sink in ("all", Path(path).name):
            raise write_error
        return write()

    with patch("manipulation.robosuite_execution.mujoco.mj_forward"), \
            patch("manipulation.execution.ExecutionGuardError", return_value=primary), \
            patch.object(Path, "write_text", lambda path, *a, **kw: attempt(path, lambda: write_text(path, *a, **kw))), \
            patch.object(np, "savez_compressed", lambda path, **kw: attempt(path, lambda: save_npz(path, **kw))):
        with pytest.raises(ExecutionGuardError) as raised:
            _execute_bound(executor, _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                initial_feedback=initial, execution_limits=_limits(simulation, curve),
                output=tmp_path/"execution", observation_callback=observe)
    assert raised.value is primary
    assert attempted == ["continuation.json", "physical-trajectory.npz", "failure-feedback.json", "execution-report.json"]
    report = executor.last_execution_report
    assert report["feedback_failure"]["point"] == "after_step"
    assert report["feedback_failure"]["violations"][0]["kind"] == "hardware_joint_velocity"
    assert report["executed_steps"] == report["abort_hold"]["physical_steps"] == 1
    assert report["continuation"]["consumed_physics_steps"] == 2
    raw_metrics = report["latest_observation"]["effects"]["test:observed_state"]["metrics"]
    assert isinstance(raw_metrics["measurement_valid"], np.bool_)
    assert isinstance(raw_metrics["frame_received"], np.bool_)
    failed = set(attempted if failed_sink == "all" else [] if failed_sink is None else [failed_sink])
    assert {row["artifact"] for row in report.get("persistence_errors", [])} == failed
    output = tmp_path/"execution"
    if "continuation.json" not in failed:
        continuation = json.loads((output/"continuation.json").read_text())
        assert continuation["effects"]["test:observed_state"]["value"] == "U"
        assert continuation["effects"]["test:observed_state"]["metrics"]["measurement_valid"] is False
        assert continuation["effects"]["test:observed_state"]["metrics"]["frame_received"] is True
    if "physical-trajectory.npz" not in failed:
        with np.load(output/"physical-trajectory.npz", allow_pickle=False) as saved:
            assert saved["actual_q"].shape == (2, 9)
            assert saved["actual_dq"][-1, 0] == 1.1
    if "failure-feedback.json" not in failed:
        feedback = json.loads((output/"failure-feedback.json").read_text())
        assert feedback["q"] == report["continuation"]["actual_feedback"]["q"].tolist()
        assert feedback["sim_time"] == report["continuation"]["actual_feedback"]["sim_time"]
    if "execution-report.json" not in failed:
        saved = json.loads((output/"execution-report.json").read_text())
        assert saved["error"] == f"ExecutionGuardError: {primary}"
        assert {row["artifact"] for row in saved.get("persistence_errors", [])} == failed


def test_first_sink_failure_is_raised_after_all_writers_when_execution_did_not_raise(tmp_path):
    simulation, curve, targets = _fixture()
    executor = _executor(simulation)
    first = OSError("continuation sink failed")
    attempted = []

    def fail(path, *args, **kwargs):
        attempted.append(Path(path).name)
        raise first if len(attempted) == 1 else OSError("later sink failed")

    with patch("manipulation.robosuite_execution.mujoco.mj_forward"), \
            patch.object(Path, "write_text", fail), patch.object(np, "savez_compressed", fail):
        with pytest.raises(OSError) as raised:
            _execute_bound(executor, _result(curve), finger_targets=FixedFingerTargets.from_mapping(targets),
                initial_feedback=_initial_feedback(simulation), execution_limits=_limits(simulation, curve),
                output=tmp_path/"execution")
    assert raised.value is first
    assert len(attempted) == 4
    assert len(executor.last_execution_report["persistence_errors"]) == 4


def test_feedback_scalar_conversion_does_not_stringify_unsupported_values():
    with pytest.raises(TypeError):
        json.dumps(_feedback_json({"unsupported": np.complex128(1+2j)}), allow_nan=False)
# END the online evidence verdict and primary guard remain unchanged.


def test_video_cleanup_attempts_every_writer_camera_and_timestamp_sink(tmp_path):
    """One failed writer must not leave the other native contexts unclosed."""
    video = runner._TrialVideos.__new__(runner._TrialVideos)
    primary = RuntimeError("first writer failed")
    actions = []

    def release(index):
        actions.append(("writer", index))
        if index == 0:
            raise primary

    video._entries = [{"writer": SimpleNamespace(release=lambda i=i: release(i)),
        "camera": SimpleNamespace(close=lambda i=i: actions.append(("camera", i))),
        "times": [[0., 1.3]], "record": {"capture_times_path": str(tmp_path/f"times-{i}.npy")}}
        for i in range(2)]
    with pytest.raises(RuntimeError) as raised:
        video.close()
    assert raised.value is primary
    assert actions == [("writer", 0), ("camera", 0), ("writer", 1), ("camera", 1)]
    for i in range(2):
        np.testing.assert_array_equal(np.load(tmp_path/f"times-{i}.npy"), [[0., 1.3]])
# END every assertion concerns reporting or owned cleanup, not physical feasibility.


# BEGIN borrowed sessions keep the real executor and only mock native integration.
class _LiveCaptureEvaluator:
    """Count actual executor callbacks without reading object or contact truth."""

    def __init__(self):
        self.samples = []

    def observe(self, model, data, record):
        self.samples.append((float(data.time), record["elapsed"]))

    def finalize(self, output, *, execution_report, collection_report):
        return {"physics_outcome": "not_evaluated", "physics_success": None,
            "evidence_coverage": {"complete": collection_report["completed"]}}


def _live_runner_inputs(tmp_path, simulation, curve, targets, *, skill_id):
    """Bind a distinct request to this backend's current, unmodified state."""
    initial = simulation.proprioception()
    for event in curve.events:
        event["skill_id"] = skill_id
    curve.events[2].update(kind="effect", requires_evidence=None, stop=False)
    curve.conditions = ({"condition_id": skill_id+":effect", "effect_requirement": "current actual evidence",
        "skill_id": skill_id, "event_id": curve.events[2]["event_id"]},)
    for event in targets["events"]:
        event["skill_id"] = skill_id
    _bind_fingers(simulation, curve, targets)
    args, result = _runner_inputs(tmp_path, simulation, curve, targets, initial, skill_id=skill_id)
    args.no_video = True
    # A live handoff does not decode or depend on the historical-entry schema.
    # Its provenance path may point at a complete production-feedback artifact.
    args.feedback.write_text(json.dumps(_feedback_json(initial)))
    return args, result, initial


def _patch_live_runner(stack, simulation, result, evaluator):
    stack.enter_context(patch.object(runner, "load_saved_trajectory", return_value=result))
    restore = stack.enter_context(patch.object(runner, "restore_recorded_simulation",
        side_effect=AssertionError("A borrowed session must never restore an episode")))
    stack.enter_context(patch.object(runner, "build_robot_validator", return_value=object()))
    executor = _executor(simulation)
    construct = stack.enter_context(patch.object(runner, "TrajectoryExecutor", return_value=executor))
    stack.enter_context(patch.object(runner, "NativeSkillPhysicsEvaluator", return_value=evaluator))
    videos = stack.enter_context(patch.object(runner, "_TrialVideos",
        side_effect=AssertionError("The no-video runner does not own the caller's camera")))
    stack.enter_context(patch("manipulation.robosuite_execution.mujoco.mj_forward"))
    return restore, construct, videos


def test_two_skills_borrow_one_backend_without_replacing_feedback_or_resetting_clock(tmp_path):
    simulation, curve, targets = _fixture()
    model, data, native_start = simulation.native_model, simulation.native_data, simulation.sim_time
    q, dq = data.qpos.copy(), data.qvel.copy()
    closes, captures, reports, feedback_ids = [], [], [], []
    simulation.close = lambda: closes.append("simulation")

    def capture(model_arg, data_arg, record):
        assert model_arg is model and data_arg is data
        captures.append((float(data_arg.time), record["elapsed"], record["feedback"]["feedback_id"]))
        # Independent sinks cannot mutate the executor's captured sample or
        # become an observation just by returning a semantic-looking object.
        record["feedback"]["q"][:] = 99.
        return {"effects": {"Holding": {"value": "T", "source": "not an observer"}}}

    for index in range(2):
        args, result, initial = _live_runner_inputs(tmp_path/str(index), simulation, curve, targets,
            skill_id=f"live_skill_{index}")
        evaluator = _LiveCaptureEvaluator()
        saved = deepcopy(initial)
        feedback_ids.append(initial["feedback_id"])
        with ExitStack() as stack:
            restore, construct, videos = _patch_live_runner(stack, simulation, result, evaluator)
            report = runner.run(args, simulation=simulation, initial_feedback=initial, step_callback=capture)
        reports.append(report)
        restore.assert_not_called()
        videos.assert_not_called()
        assert construct.call_args.args[0] is simulation
        assert len(evaluator.samples) == 7
        assert report["simulation_ownership"] == "borrowed" and report["restore"] is None
        assert report["trial_id"] == simulation.trial_id
        assert report["native_start_time_s"] == initial["sim_time"]
        assert report["native_end_time_s"] == pytest.approx(initial["sim_time"]+.012)
        assert report["evidence_collection"]["capture_callback"]["completed"]
        assert report["online_observation_provided"] is False
        execution = json.loads((args.output/"execution"/"execution-report.json").read_text())
        assert execution["status"] == "curve_finished" and execution["effect_result"] == "U"
        assert execution["continuation"]["actual_feedback"]["trial_id"] == simulation.trial_id
        assert execution["continuation"]["actual_feedback"]["tool_pose_source"] == initial["tool_pose_source"]
        final = json.loads(Path(execution["robot_feedback_path"]).read_text())
        assert final["initial_feedback_id"] == initial["feedback_id"]
        np.testing.assert_array_equal(initial["q"], saved["q"])
        np.testing.assert_array_equal(initial["dq"], saved["dq"])
        np.testing.assert_array_equal(final["q"], q)
    assert feedback_ids[0] != feedback_ids[1]
    assert reports[0]["native_end_time_s"] == reports[1]["native_start_time_s"]
    assert simulation.native_model is model and simulation.native_data is data and closes == []
    assert simulation.sim_time == pytest.approx(native_start+.024)
    assert len(captures) == 14 and captures[6][0] == captures[7][0]
    np.testing.assert_array_equal(data.qpos, q)
    np.testing.assert_array_equal(data.qvel, dq)


@pytest.mark.parametrize("field", ["q", "dq", "sim_time", "timestamp_s", "trial_id", "feedback_id"])
def test_borrowed_backend_rejects_stale_or_foreign_feedback_before_commands(tmp_path, field):
    simulation, curve, targets = _fixture()
    args, result, initial = _live_runner_inputs(tmp_path, simulation, curve, targets, skill_id="live_skill")
    if field in ("q", "dq"):
        initial[field][0] += .01
    elif field in ("sim_time", "timestamp_s"):
        initial[field] += .01
    else:
        initial[field] = "different_native_sample"
    before = deepcopy(simulation.proprioception())
    controls = simulation.native_data.ctrl.copy()
    with ExitStack() as stack:
        restore, construct, videos = _patch_live_runner(stack, simulation, result, _LiveCaptureEvaluator())
        with pytest.raises(ValueError, match="same current native|differs from the selected"):
            runner.run(args, simulation=simulation, initial_feedback=initial)
    restore.assert_not_called()
    construct.assert_not_called()
    videos.assert_not_called()
    assert simulation.sim_time == before["sim_time"]
    np.testing.assert_array_equal(simulation.native_data.qpos, before["q"])
    np.testing.assert_array_equal(simulation.native_data.qvel, before["dq"])
    np.testing.assert_array_equal(simulation.native_data.ctrl, controls)


@pytest.mark.parametrize("missing", ["simulation", "initial_feedback"])
def test_live_handoff_requires_backend_and_complete_feedback_together(tmp_path, missing):
    simulation, curve, targets = _fixture()
    args, result, initial = _live_runner_inputs(tmp_path, simulation, curve, targets, skill_id="live_skill")
    supplied = {"simulation": simulation, "initial_feedback": initial}
    supplied.pop(missing)
    with patch.object(runner, "restore_recorded_simulation") as restore:
        with pytest.raises(ValueError, match="must be supplied together"):
            runner.run(args, **supplied)
    restore.assert_not_called()
    assert simulation.sim_time == initial["sim_time"]


@pytest.mark.parametrize("failed_elapsed", [0., .002])
def test_borrowed_capture_failure_retains_peer_evidence_and_native_final_state(tmp_path, failed_elapsed):
    simulation, curve, targets = _fixture()
    args, result, initial = _live_runner_inputs(tmp_path, simulation, curve, targets, skill_id="live_skill")
    evaluator, captured = _LiveCaptureEvaluator(), []

    def capture(model, data, record):
        captured.append(record["elapsed"])
        if record["elapsed"] >= failed_elapsed:
            raise RuntimeError("external RGB-D capture failed")

    with ExitStack() as stack:
        _patch_live_runner(stack, simulation, result, evaluator)
        report = runner.run(args, simulation=simulation, initial_feedback=initial, step_callback=capture)
    state = report["evidence_collection"]["capture_callback"]
    assert report["status"] == "completed" and report["trajectory_executed_steps"] == 6
    assert state["disabled"] and not state["completed"]
    assert state["first_error"]["message"] == "external RGB-D capture failed"
    assert state["first_error"]["curve_time"] == failed_elapsed
    assert len(captured) == 1+int(failed_elapsed/simulation.timestep)
    assert len(evaluator.samples) == 7 and report["evidence_collection"]["evaluator"]["completed"]
    assert report["evidence_complete"] is False
    assert report["native_end_time_s"] == simulation.sim_time == pytest.approx(initial["sim_time"]+.012)


def test_borrowed_capture_failure_cannot_replace_a_later_native_error(tmp_path):
    simulation, curve, targets = _fixture()
    args, result, initial = _live_runner_inputs(tmp_path, simulation, curve, targets, skill_id="live_skill")
    primary = RuntimeError("native step failed after advancing its clock")
    step, closed = simulation._sim.step, []
    simulation.close = lambda: closed.append(True)

    def failed_step():
        step()
        raise primary

    def failed_capture(model, data, record):
        raise ValueError("independent capture failed first")

    simulation._sim.step = failed_step
    with ExitStack() as stack:
        _patch_live_runner(stack, simulation, result, _LiveCaptureEvaluator())
        with pytest.raises(RuntimeError) as raised:
            runner.run(args, simulation=simulation, initial_feedback=initial, step_callback=failed_capture)
    report = json.loads((args.output/"simulation-report.json").read_text())
    assert raised.value is primary and report["error"] == f"RuntimeError: {primary}"
    assert report["evidence_collection"]["capture_callback"]["first_error"]["message"] == "independent capture failed first"
    assert report["native_end_time_s"] == simulation.sim_time > initial["sim_time"]
    assert report["final_robot_feedback"]["sim_time"] == simulation.sim_time
    assert closed == []
# END live-session tests never instantiate a native model, renderer or GPU planner.


# BEGIN a validated recorded entrance can be retained for the next live skill.
def _recorded_ready_inputs(tmp_path, simulation, curve, targets):
    """Archive the original measured identity without rewriting the finger curve."""
    initial = simulation.proprioception()
    curve.events[2].update(kind="effect", requires_evidence=None, stop=False)
    curve.conditions = ({"condition_id": "declared_skill:effect",
        "effect_requirement": "current actual evidence", "skill_id": "declared_skill",
        "event_id": curve.events[2]["event_id"]},)
    _bind_fingers(simulation, curve, targets)
    args, _ = _runner_inputs(tmp_path, simulation, curve, targets, initial)
    args.no_video = True
    result = _result(curve)
    result.diagnostics["archive_classification"] = "planner_accepted"
    return args, result, initial


def test_recorded_ready_hook_retains_backend_and_original_identity_for_next_skill(tmp_path):
    simulation, curve, targets = _fixture()
    args, result, initial = _recorded_ready_inputs(tmp_path/"recorded", simulation, curve, targets)
    retained, lifecycle = [], []
    restore_report = {"source": {"entry_frame": 0}}
    before_controls = simulation.native_data.ctrl.copy()

    def ready(backend, restored):
        assert backend is simulation and restored == restore_report
        assert simulation.sim_time == initial["sim_time"] and simulation.arm_goals == []
        np.testing.assert_array_equal(simulation.native_data.ctrl, before_controls)
        retained.append(backend)
        lifecycle.append("ready")
        restored["source"]["entry_frame"] = 99  # The diagnostic copy cannot change the runner report.

    def capture(model, data, record):
        assert retained == [simulation] and lifecycle[0] == "ready"
        lifecycle.append("sample")

    with ExitStack() as stack:
        restore, _, _ = _patch_live_runner(stack, simulation, result, _LiveCaptureEvaluator())
        restore.side_effect = None
        restore.return_value = simulation, restore_report
        first = runner.run(args, on_simulation_ready=ready, step_callback=capture)
    restore.assert_called_once()
    assert lifecycle.count("ready") == 1 and first["restore"] == restore_report
    assert restore_report == {"source": {"entry_frame": 0}}
    first_feedback = json.loads(Path(first["robot_feedback_path"]).read_text())
    assert first_feedback["initial_feedback_id"] == initial["feedback_id"]
    assert first["simulation_ownership"] == "recorded_trial"
    assert first["native_end_time_s"] == pytest.approx(initial["sim_time"]+.012)

    second_args, second_result, current = _live_runner_inputs(tmp_path/"live", retained[0], curve, targets,
        skill_id="next_skill")
    with ExitStack() as stack:
        restore, _, _ = _patch_live_runner(stack, simulation, second_result, _LiveCaptureEvaluator())
        second = runner.run(second_args, simulation=retained[0], initial_feedback=current)
    restore.assert_not_called()
    assert second["simulation_ownership"] == "borrowed" and second["restore"] is None
    assert first["trial_id"] == second["trial_id"] == simulation.trial_id
    assert first["native_end_time_s"] == second["native_start_time_s"]
    assert second["native_end_time_s"] == pytest.approx(initial["sim_time"]+.024)
    second_feedback = json.loads(Path(second["robot_feedback_path"]).read_text())
    assert second_feedback["initial_feedback_id"] == current["feedback_id"]


@pytest.mark.parametrize("field", ["start_q", "start_dq", "sim_time_s"])
def test_recorded_ready_hook_is_not_called_when_the_entrance_differs(tmp_path, field):
    simulation, curve, targets = _fixture()
    args, result, initial = _recorded_ready_inputs(tmp_path, simulation, curve, targets)
    feedback = json.loads(args.feedback.read_text())
    if field == "sim_time_s":
        feedback[field] += .01
    else:
        feedback[field][0] += .01
    args.feedback.write_text(json.dumps(feedback))
    received = []
    with ExitStack() as stack:
        restore, construct, _ = _patch_live_runner(stack, simulation, result, _LiveCaptureEvaluator())
        restore.side_effect = None
        restore.return_value = simulation, {}
        with pytest.raises(ValueError, match="differs from the selected observed entrance"):
            runner.run(args, on_simulation_ready=lambda *args: received.append(args))
    restore.assert_called_once()
    construct.assert_not_called()
    assert received == [] and simulation.arm_goals == [] and simulation.sim_time == initial["sim_time"]


def test_recorded_ready_hook_failure_is_primary_and_prevents_commands(tmp_path):
    simulation, curve, targets = _fixture()
    args, result, initial = _recorded_ready_inputs(tmp_path, simulation, curve, targets)
    primary = RuntimeError("session could not retain the restored backend")
    before_controls = simulation.native_data.ctrl.copy()

    def fail_ready(backend, restored):
        raise primary

    with ExitStack() as stack:
        restore, construct, videos = _patch_live_runner(stack, simulation, result, _LiveCaptureEvaluator())
        restore.side_effect = None
        restore.return_value = simulation, {}
        with pytest.raises(RuntimeError) as raised:
            runner.run(args, on_simulation_ready=fail_ready)
    assert raised.value is primary
    construct.assert_not_called()
    videos.assert_not_called()
    assert simulation.arm_goals == [] and simulation.sim_time == initial["sim_time"]
    np.testing.assert_array_equal(simulation.native_data.ctrl, before_controls)
    report = json.loads((args.output/"simulation-report.json").read_text())
    assert report["error"] == f"RuntimeError: {primary}" and report["execution_tested"] is False
    assert report["native_start_time_s"] == report["native_end_time_s"]
    assert report["evidence_collection"]["evaluator"]["initialized"] is False


@pytest.mark.parametrize("borrowed", [False, True])
def test_simulation_ready_hook_rejects_invalid_ownership_or_callable_before_restore(tmp_path, borrowed):
    simulation, curve, targets = _fixture()
    args, result, initial = _recorded_ready_inputs(tmp_path, simulation, curve, targets)
    supplied = {"simulation": simulation, "initial_feedback": initial} if borrowed else {}
    hook = (lambda *args: None) if borrowed else object()
    message = "cannot be used with a borrowed simulation" if borrowed else "must be callable"
    with patch.object(runner, "restore_recorded_simulation") as restore:
        with pytest.raises(ValueError, match=message):
            runner.run(args, on_simulation_ready=hook, **supplied)
    restore.assert_not_called()
    assert simulation.arm_goals == [] and simulation.sim_time == initial["sim_time"]
# END callback delivery never authorizes geometry or advances the native clock.


# BEGIN assumption and observer plumbing cannot synthesize robot or effect evidence.
@pytest.mark.parametrize("supplied", [False, True])
def test_runner_forwards_attachment_assumptions_separately_from_observer(tmp_path, supplied):
    """Check both optional channels at executor admission without native stepping."""
    simulation, curve, targets = _fixture()
    args, result, initial = _live_runner_inputs(tmp_path, simulation, curve, targets, skill_id="live_skill")
    original_feedback = _feedback_json(initial)
    assumptions = {"bound_condition": {"authorization": "explicit test caller"}} if supplied else None
    original_assumptions = deepcopy(assumptions)
    observer = (lambda feedback, curve_time: None) if supplied else None
    admission = RuntimeError("test stops at executor admission")

    with ExitStack() as stack:
        _, construct, _ = _patch_live_runner(stack, simulation, result, _LiveCaptureEvaluator())
        execute = stack.enter_context(patch.object(construct.return_value, "execute", side_effect=admission))
        with pytest.raises(RuntimeError) as raised:
            runner.run(args, simulation=simulation, initial_feedback=initial,
                attachment_assumptions=assumptions, observation_callback=observer)
    assert raised.value is admission
    assert execute.call_args.kwargs["attachment_assumptions"] is assumptions
    assert execute.call_args.kwargs["observation_callback"] is observer
    assert _feedback_json(execute.call_args.kwargs["initial_feedback"]) == original_feedback
    assert _feedback_json(initial) == original_feedback and assumptions == original_assumptions
    assert simulation.sim_time == initial["sim_time"] and simulation.arm_goals == []
    report = json.loads((args.output/"simulation-report.json").read_text())
    assert report["attachment_assumptions_provided"] is supplied
    assert report["online_observation_provided"] is supplied


@pytest.mark.parametrize("supplied", [False, True])
def test_continuous_session_forwards_separate_inputs_on_restore_and_live_handoff(tmp_path, supplied):
    """Both session branches expose predicate inputs while preserving the same trial."""
    import continuous_simulation as continuous

    simulation, curve, targets = _fixture()
    args, _, initial = _recorded_ready_inputs(tmp_path/"inputs", simulation, curve, targets)
    session = continuous.ContinuousSimulationSession(manifest_path=args.manifest,
        fixture_path=args.robot_fixture, controller_part=args.controller_part, output=tmp_path/"session")
    assumptions = {"bound_condition": "explicit test caller"} if supplied else None
    observer = (lambda feedback, curve_time: None) if supplied else None
    calls = []

    def run_stage(spec, **kwargs):
        calls.append(kwargs)
        assert kwargs["attachment_assumptions"] is assumptions
        assert kwargs["observation_callback"] is observer
        if len(calls) == 1:
            assert "simulation" not in kwargs and "initial_feedback" not in kwargs
            kwargs["on_simulation_ready"](simulation, {})
        else:
            assert kwargs["simulation"] is simulation and "on_simulation_ready" not in kwargs
            assert _feedback_json(kwargs["initial_feedback"]) == _feedback_json(initial)
        return {"status": "completed", "physics_success": True,
                "native_start_time_s": initial["sim_time"]}

    with patch.object(continuous, "run", side_effect=run_stage), patch.object(
            session, "_integration_state", return_value=np.array([initial["sim_time"]])):
        for _ in range(2):
            report = session.execute(vars(args), attachment_assumptions=assumptions,
                observation_callback=observer)
            assert report["status"] == "completed"
    assert len(calls) == 2 and len(session.stages) == 2
    assert session.trial_id == simulation.trial_id and simulation.sim_time == initial["sim_time"]
    assert simulation.arm_goals == []
# END passing an input is never treated as an observation or physical outcome.
