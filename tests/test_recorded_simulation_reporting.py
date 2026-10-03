"""CPU-only failure reporting; all physics, evaluators and renderers are doubles."""

# BEGIN a large generated report must not replace the original execution error.
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from test_recorded_simulation import (
    _fixture, _initial_feedback, _limit_values, _result, _executor, _limits, _execute_bound,
)
from manipulation.execution import ExecutionGuardError, FixedFingerTargets, _feedback_json
import run_recorded_simulation as runner


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
        "robot_feedback_path": "actual_failure_feedback", "barrier_feedback_path": None,
        "geometry_preflight": {"passed": np.bool_(True), "checked_interval_s": [0., .0075],
            "reference_collision_checked": True, "continuous_collision_certified": False,
            "observed_scene_checked": True, "unknown_space_checked": False,
            "nominal_scene": {"states": "must not be duplicated into the small summary"}},
        "feedback_failure": {"point": "after_step", "curve_time_s": .002,
            "physical_elapsed_s": .002, "actual_feedback": initial,
            "violations": [{"kind": "arm_position_tracking", "name": "arm_0",
                "measured": np.float64(.11), "expected": .10002, "maximum_error": .001}]},
        "continuation": {"actual_feedback": initial},
        "large_geometry_diagnostic": "x"*(16*1024**2+1)}

    def save(name, value):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    fixture = {"joint_names": list(simulation.joint_names),
        "arm_joint_names": list(simulation.arm_joint_names), "finger_joint_names": list(simulation.finger_joint_names),
        "tool_link": simulation.tool_link, "base_link": "test_base", "tcp_site": simulation.tool_link,
        "episode_path": str(tmp_path/"recording"/"episode.h5"), "source_scene_xml": "unused_test_scene",
        "catalog_path": str(save("catalog.json", {})), "velocity_limits": simulation.velocity_limits}
    feedback = {**fixture, "segment_id": "declared_skill", "entry_frame": 0,
        "reference_frame": "world", "tool_transform": initial["tool_transform"],
        "start_q": initial["q"].tolist(), "start_dq": initial["dq"].tolist(),
        "base_pose": initial["base_pose"].tolist(), "eef_pose": initial["tool_pose"].tolist(),
        "sim_time_s": initial["sim_time"], "source": initial["source"]}
    save("program.json", {"skill_kind": "grasp", "demo": {"roles": {}}})
    manifest = {"workspace_root": str(tmp_path), "source_recording": "recording", "segments": [{
        "id": "declared_skill", "program_path": "program.json", "source_library_segment": "declared_skill",
        "snapshot": {"entry_frame": 0, "reference_frame": "world", "views": [{"id": "test_view"}]}}]}
    save("recording/manifest.json", {"environment": {"control_freq_hz": 20, "controller_config": {"body_parts": {
        "test_arm": {"type": "JOINT_POSITION", "input_type": "absolute", "gripper": {"type": "GRIP"},
                     "interpolation": None, "kp": 1., "kd": 1.}}}}})
    evidence = {"actor": {"entity_name": "test_robot", "eef_site_name": simulation.tool_link,
        "contact_geom_groups": []}, "segments": [{"segment_id": "declared_skill", "interaction": "grasp",
        "carrier_body_name": "test_owner"}], "evaluation_owner_bindings": []}
    args = SimpleNamespace(output=tmp_path/"trial", trajectory=tmp_path/"formal.npz",
        manifest=save("manifest.json", manifest), robot_fixture=save("fixture.json", fixture),
        feedback=save("feedback.json", feedback), finger_targets=save("targets.json", targets),
        execution_limits=save("limits.json", _limit_values(simulation, curve)),
        evaluator_evidence=save("evidence.json", evidence),
        result_only_goal_ids=[],
        segment="declared_skill", controller_part="test_arm", views=["test_view"], no_video=False,
        max_physics_steps=30000, start_position_tolerance=2e-6, start_velocity_tolerance=2e-6)
    result = _result(curve)
    result.diagnostics["archive_classification"] = "planner_accepted"

    class Executor:
        last_execution_report = None

        def execute(self, *args, **kwargs):
            self.last_execution_report = shared_report
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

        def finalize(self, output, *, execution_report):
            self.finalized = True
            assert execution_report is shared_report
            if cleanup_failure:
                raise ValueError("independent evaluator cleanup failure")
            return {"physics_outcome": "not_evaluated", "physics_success": None}

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
        "kind": "arm_position_tracking", "name": "arm_0", "measured": .11,
        "expected": .10002, "maximum_error": .001}
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


# BEGIN generic upstream NumPy booleans survive the executor and every sink.
@pytest.mark.parametrize("failed_sink", [None, "continuation.json", "physical-trajectory.npz",
                                       "failure-feedback.json", "execution-report.json", "all"])
def test_numpy_observation_survives_tracking_rejection_and_independent_writes(tmp_path, failed_sink):
    simulation, curve, targets = _fixture()
    initial = _initial_feedback(simulation)
    original_step = simulation._sim.step

    def step():
        original_step()
        simulation.native_data.qpos[0] = .11

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
    primary = ExecutionGuardError("original measured tracking rejection")
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
    assert report["feedback_failure"]["violations"][0]["kind"] == "arm_position_tracking"
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
            assert saved["actual_q"][-1, 0] == .11
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
