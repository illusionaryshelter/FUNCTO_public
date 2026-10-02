"""CPU-only failure reporting; all physics, evaluators and renderers are doubles."""

# BEGIN a large generated report must not replace the original execution error.
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from test_recorded_simulation import _fixture, _initial_feedback, _limit_values, _result
import run_recorded_simulation as runner


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_large_generated_report_is_reused_and_primary_error_survives_cleanup(tmp_path, cleanup_failure):
    simulation, curve, targets = _fixture()
    initial = _initial_feedback(simulation)
    primary = RuntimeError("original independent geometry rejection")
    shared_report = {
        "status": "feedback_rejected", "executed_steps": 0, "barrier_command_reached": False,
        "elapsed_sim_s": .002,
        "abort_hold": {"physical_steps": 1}, "dependency_hold": None,
        "robot_feedback_path": "actual_failure_feedback", "barrier_feedback_path": None,
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
        "carrier_body_name": "test_owner"}]}
    args = SimpleNamespace(output=tmp_path/"trial", trajectory=tmp_path/"formal.npz",
        manifest=save("manifest.json", manifest), robot_fixture=save("fixture.json", fixture),
        feedback=save("feedback.json", feedback), finger_targets=save("targets.json", targets),
        execution_limits=save("limits.json", _limit_values(simulation, curve)),
        evaluator_evidence=save("evidence.json", evidence), tactile_observer=save("tactile.json", {}),
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
            path.write_text(json.dumps(shared_report))
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

    class Sensor:
        observations = []
        describe_calls = 0

        def describe(self):
            self.describe_calls += 1
            if cleanup_failure and self.describe_calls > 1:
                raise ValueError("tactile persistence failure")
            return {"scope": "synthetic reporting test"}

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

    executor, evaluator, sensor, videos = Executor(), Evaluator(), Sensor(), Videos()
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner, "load_saved_trajectory", return_value=result))
        stack.enter_context(patch.object(runner, "restore_recorded_simulation", return_value=(simulation, {})))
        stack.enter_context(patch.object(runner, "build_robot_validator", return_value=object()))
        stack.enter_context(patch.object(runner, "TrajectoryExecutor", return_value=executor))
        stack.enter_context(patch.object(runner, "NativeSkillPhysicsEvaluator", return_value=evaluator))
        stack.enter_context(patch.object(runner, "_TrialVideos", return_value=videos))
        stack.enter_context(patch("simulated_tactile.SimulatedTactileObserver", return_value=sensor))
        reads = stack.enter_context(patch.object(runner, "_read", wraps=runner._read))
        with pytest.raises(RuntimeError) as raised:
            runner.run(args)
    assert raised.value is primary
    report_path = args.output/"execution"/"execution-report.json"
    assert report_path.stat().st_size > 16*1024**2
    assert all(Path(call.args[0]) != report_path for call in reads.call_args_list)
    assert evaluator.finalized and videos.closed
    report = json.loads((args.output/"simulation-report.json").read_text())
    assert report["error"] == "RuntimeError: original independent geometry rejection"
    assert report["trajectory_executed_steps"] == 0
    assert report["hold_physics_steps"] == report["actual_physics_steps"] == 1
    assert report["evaluator_excluded_hold_physics_steps"] == 1
    assert report["execution_tested"] is False
    assert report["actual_physics_tested"] is True
    assert report["robot_feedback_path"] == "actual_failure_feedback"
    assert (args.output/"exception.txt").read_text().rstrip().endswith(str(primary))
    assert videos.captures[-1] == ((.002,), {"final": True})
    if cleanup_failure:
        assert "evaluation_error" in report and "tactile_write_error" in report and "video_error" in report
        assert "video_final_capture_error" in report
    else:
        assert (args.output/"tactile-observations.json").is_file()
        assert report["video_final_elapsed_s"] == .002
    assert simulation.sim_time == 1.3  # No native integration occurs in these doubles.


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
