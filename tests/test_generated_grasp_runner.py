"""CPU evidence for generated action selection, errors and exact goal queries.

Subprocess mocks publish the same small report boundaries as the real runners.
The FK test uses one isolated analytic robot joint, never a scene or grasp
success fixture. No learned model, CUDA allocation or physical rollout runs.
"""

# BEGIN finite candidate orchestration retains complete current-skill contracts.
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
import run_generated_grasp as runner


def inputs():
    """Give the real manifest writer independent current and source identities."""
    scene = {"reference_frame": "world", "tool_id": "declared_hand", "tool_frame": "tcp",
             "carrier_id": "target", "carrier_pose": np.eye(4).tolist(),
             "region_transform": np.eye(4).tolist(), "eef_pose": np.eye(4).tolist(),
             "grasp_goal": {"T_carrier_eef": np.eye(4).tolist(),
                            "position_tolerance_m": .002,
                            "rotation_tolerance_rad": float(np.deg2rad(1.)),
                            "source": "independent prior and original allowed set"}}
    old = {"id": "perception/first", "scene_path": "observed/scene.json",
           "view_id": "camera", "selection_evidence_path": "observed/selection.json"}
    segment = {"id": "current", "program_path": "source/program.json",
               "snapshot": {"views": [{"id": "camera", "depth_path": "observed/depth.npy"}]},
               "grounding": {"candidates": [old], "selected_candidate": old["id"], "status": "selected"}}
    return {"scene": scene, "candidate": old, "segment": segment,
            "manifest": {"workspace_root": "/declared/source/workspace", "segments": [segment]},
            "command": [sys.executable, "-u", str(Path(runner.__file__).with_name("run_recorded_skill.py")),
                        "--manifest", "original-manifest.json", "--candidate", old["id"],
                        "--output", "original-output", "--planning-seconds", "120",
                        "--max-device-used-mib", "6144", "--max-host-rss-mib", "8192"]}


def candidate(index):
    target = np.eye(4)
    target[0, 3] = .03 * index
    return {"candidate_id": f"grasp_{index:04d}", "source": "demo_exact" if index == 0 else "diffusion",
            "quality": .4, "T_reference_tool": target}


def prepare_run(tmp_path, monkeypatch, *, pool=None, outcomes=None,
                generation_exit=0, maximum=8, wall=30.):
    """Mock only process/report and expensive FK boundaries, keeping run intact."""
    args = runner.parser().parse_args(["--planning-command", str(tmp_path / "command.json"),
        "--output", str(tmp_path / "result"), "--checkpoints", str(tmp_path / "weights"),
        "--tool-model", "robosuite_panda", "--max-candidate-attempts", str(maximum),
        "--search-wall-seconds", str(wall), "--max-device-used-mib", "10240"])
    bound = inputs()
    pool = [candidate(0), candidate(1)] if pool is None else pool
    outcomes = [{}] if outcomes is None else outcomes
    clock = SimpleNamespace(now=0.)
    calls = []
    monkeypatch.setattr(runner, "load_inputs", lambda _: deepcopy(bound))
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock.now))

    def child(command, log_path, timeout):
        assert timeout > 0
        if "--generate-only" in command:
            if generation_exit:
                return generation_exit
            runner.write_json(args.output / "generation.json", {"candidates": pool})
            runner.write_json(args.output / "gripper-geometry.json", {"T_tool_grasp": np.eye(4)})
            clock.now += 2.
            return 0
        result = outcomes[len(calls)]
        calls.append(list(command))
        output = Path(runner.option(command, "--output"))
        output.mkdir(parents=True)
        report = result.get("report", {"status": "completed", "planning_success": True})
        runner.write_json(output / "run-report.json", report)
        if not result.get("omit_planning_report"):
            (output / "planning").mkdir()
            runner.write_json(output / "planning" / "planning-report.json",
                              {"diagnostics": result.get("diagnostics", {})})
        clock.now += result.get("elapsed", 1.)
        return result.get("exit", 0)

    monkeypatch.setattr(runner, "run_child", child)
    return args, calls, clock, bound


def read_result(args):
    return json.loads((args.output / "result.json").read_text())


def rejected(failure="no_feasible_solution_found", error_type=None, **extra):
    return {"report": {"status": "completed", "planning_success": False, "failure": failure},
            "diagnostics": {"failure": failure, "error_type": error_type, **extra}}


# BEGIN the generator's common order is authoritative for all candidate sources.
@pytest.mark.parametrize("reject_first", [False, True])
def test_higher_ranked_alternative_precedes_prior_and_rejection_uses_same_order(
        tmp_path, monkeypatch, reject_first):
    alternative, prior = candidate(1), candidate(0)
    alternative.update(quality=.9, ranking_score=.8, rank=0)
    prior.update(quality=.4, ranking_score=.6, rank=1)
    args, calls, _, _ = prepare_run(tmp_path, monkeypatch, pool=[alternative, prior],
        outcomes=[rejected(), {}] if reject_first else [{}])
    assert runner.run(args) == 0
    result = read_result(args)
    expected = [alternative, prior] if reject_first else [alternative]
    assert [row["candidate_id"] for row in result["attempts"]] == [row["candidate_id"] for row in expected]
    assert result["selected"]["candidate_id"] == expected[-1]["candidate_id"]
    assert len(calls) == len(expected)
    assert result["selected"]["actual_holding_verified"] is False
# END neither preservation of the prior nor its feasibility restores priority.


def test_finite_search_failure_continues_and_manifest_keeps_one_current_action(tmp_path, monkeypatch):
    args, calls, _, original = prepare_run(tmp_path, monkeypatch,
        outcomes=[rejected(), {}])
    assert runner.run(args) == 0
    result = read_result(args)
    assert [row["status"] for row in result["attempts"]] == ["not_accepted", "planning_accepted"]
    assert result["selected"]["candidate_id"] == "grasp_0001"
    assert result["actual_execution_tested"] is False
    assert result["selected"]["actual_holding_verified"] is False
    for command in calls:
        # Generation's 10 GiB outer allowance cannot rewrite the established
        # planner request's smaller 6 GiB admission.
        assert runner.option(command, "--max-device-used-mib") == "6144"
    manifest = json.loads(Path(result["selected"]["manifest"]).read_text())
    segment = manifest["segments"][0]
    assert manifest["workspace_root"] == original["manifest"]["workspace_root"]
    assert segment["program_path"] == original["segment"]["program_path"]
    assert segment["snapshot"] == original["segment"]["snapshot"]
    assert segment["grounding"]["selected_candidate"] == "generated/grasp_0001"
    assert [row["id"] for row in segment["grounding"]["candidates"]] == ["generated/grasp_0001"]
    assert original["segment"]["grounding"]["selected_candidate"] == "perception/first"


@pytest.mark.parametrize(("failure", "error_type"), [
    ("planning_stage_failed", "RuntimeError"), ("planning_stage_failed", "ResourceLimitError"),
    ("planning_stage_failed", "MemoryError"), ("capacity_exceeded", "MotionPlanningError"),
    ("unsupported_model", "MotionPlanningError"), ("new_unknown_failure", None),
])
def test_runtime_resource_and_unknown_errors_stop_without_trying_another_grasp(
        tmp_path, monkeypatch, failure, error_type):
    args, calls, _, _ = prepare_run(tmp_path, monkeypatch,
        outcomes=[rejected(failure, error_type, failed_stage="main_solve")])
    assert runner.run(args) == 1
    result = read_result(args)
    assert result["status"] == "error" and result["selected"] is None
    assert len(calls) == 1
    assert result["attempts"][0]["failure"] == failure
    assert result["attempts"][0]["error_type"] == error_type


def test_generation_model_error_never_falls_back_to_the_old_grasp(tmp_path, monkeypatch):
    args, calls, _, _ = prepare_run(tmp_path, monkeypatch, generation_exit=3)
    assert runner.run(args) == 1
    result = read_result(args)
    assert result["status"] == "error" and result["selected"] is None
    assert calls == [] and result["attempts"] == []
    assert "Candidate generation failed" in result["error"]


def test_input_error_before_planning_report_preserves_the_original_reason(tmp_path, monkeypatch):
    args, calls, _, _ = prepare_run(tmp_path, monkeypatch, outcomes=[{
        "exit": 1, "omit_planning_report": True,
        "report": {"status": "error", "error": "RuntimeError: native actuator binding mismatch"}}])
    assert runner.run(args) == 1
    result = read_result(args)
    assert len(calls) == 1 and result["status"] == "error"
    assert "native actuator binding mismatch" in result["error"]
    assert "FileNotFoundError" not in result["error"]


def test_selected_attempt_keeps_nominal_goal_and_reference_parameters(tmp_path, monkeypatch):
    args, calls, _, _ = prepare_run(tmp_path, monkeypatch,
        pool=[candidate(0)])
    assert runner.run(args) == 0 and len(calls) == 1
    result = read_result(args)
    assert result["selected"]["candidate_id"] == "grasp_0000"
    scene = json.loads(Path(result["selected"]["scene"]).read_text())
    assert scene["grasp_goal"]["position_tolerance_m"] == .002
    assert scene["grasp_goal"]["rotation_tolerance_rad"] == np.deg2rad(1.)
    np.testing.assert_array_equal(scene["grasp_reference_prior"]["T_carrier_eef"], np.eye(4))


def test_maximum_candidate_attempts_and_finite_pool_have_different_outcomes(tmp_path, monkeypatch):
    args, calls, _, _ = prepare_run(tmp_path, monkeypatch, maximum=1, outcomes=[rejected()])
    assert runner.run(args) == 1 and len(calls) == 1
    assert read_result(args)["status"] == "search_budget_exhausted"


def test_exhausted_pool_is_not_reported_as_mathematical_infeasibility(tmp_path, monkeypatch):
    args, calls, _, _ = prepare_run(tmp_path, monkeypatch,
        pool=[candidate(0)], outcomes=[rejected()])
    assert runner.run(args) == 1 and len(calls) == 1
    assert read_result(args)["status"] == "no_accepted_candidate_in_generated_pool"


def test_search_wall_budget_is_shared_and_not_reset_for_another_candidate(tmp_path, monkeypatch):
    first = rejected()
    first["elapsed"] = 4.
    args, calls, _, _ = prepare_run(tmp_path, monkeypatch, wall=10., outcomes=[first, {}])
    assert runner.run(args) == 0 and len(calls) == 2
    assert [float(runner.option(row, "--planning-seconds")) for row in calls] == [8., 4.]


# BEGIN an accepted motion result cannot reset the request wall deadline.
def test_completed_planning_cannot_select_after_the_shared_deadline(tmp_path, monkeypatch):
    args, calls, _, _ = prepare_run(tmp_path, monkeypatch, wall=10., outcomes=[{"elapsed": 8.}])
    assert runner.run(args) == 1 and len(calls) == 1
    result = read_result(args)
    assert result["status"] == "search_budget_exhausted" and result["selected"] is None
# END selected candidates still do not certify actual physical Holding.
