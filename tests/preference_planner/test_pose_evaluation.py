"""Bound pose roles retain posthoc metrics and independent effect grading.

The pose matrices and T/F/U effect records below are analytical unit inputs.
The original evaluator performs their arithmetic and aggregation; the control
fixture exercises the original executor with synthetic robot feedback. These
checks establish responsibility boundaries, not physical skill success.
"""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "tests"), str(PROJECT / "tests/planning")]


from manipulation.execution_binding import plain
from manipulation.planning_problem import PoseEventRequirement
from manipulation.preference_planner.pose_roles import bind_pose_objectives, prior_goal_ids
from recorded_skill_evaluation import _actual_pose_goal, summarize_skill_evidence
from test_pose_roles import declarations, event, problem
# Import the fixture module without exporting its unittest class into this
# test namespace; pytest would otherwise collect that original class again.
import test_recorded_simulation as original_simulation


# BEGIN formal source roles select grading identities without changing measurements.
def bound_evidence(*, effect="T", required_offset_m=0.):
    """Use nonlocal original goal IDs and the actual pose diagnostic arithmetic."""
    original = problem([event(source={"origin": "declared prior fixture", "evidence_frames": [2, 5]}),
                        event(progress=1., source="explicit required fixture")])
    bound = bind_pose_objectives(original, declarations(original.pose_events, ["prior", "required"]))
    binding = {"candidate_id": "explicit-analytical-candidate", "problem": plain(bound),
               "goal_identities": {"pose_events": [7, 11]}}
    binding["problem"]["conditions"] = [{"condition_id": "declared_effect",
        "effect_requirement": "independent analytical effect predicate"}]
    goals = []
    for index, row, offset in zip((7, 11), binding["problem"]["pose_events"], (.03, required_offset_m)):
        # This is the same base-class reconstruction used by the executor and
        # result evaluator. Original source and diagnostic thresholds survive.
        requirement = PoseEventRequirement(**deepcopy(row))
        assert requirement.source == row["source"]
        pose = np.eye(4)
        pose[0, 3] = offset
        feedback = {"tool_pose": pose, "source": "analytical joint-feedback fixture",
            "tool_pose_source": "forward_kinematics_from_measured_arm_q",
            "tool_pose_is_independent_observation": False,
            "timestamp_s": 2. * requirement.progress, "feedback_id": "analytical-feedback"}
        measurement = _actual_pose_goal(requirement, feedback, None)
        goals.append({"identity": ["pose_events", index], "value": measurement["value"],
            "measurement": measurement, "position_tolerance_m": requirement.position_tolerance_m,
            "source": deepcopy(requirement.source)})
    report = {"candidate_id": binding["candidate_id"], "evaluation_kind": "grasp",
        "functional_goal_evidence": goals,
        "effect_evidence": [{"condition_id": "declared_effect", "value": effect,
            "source": "explicit analytical effect record; no physics replay"}],
        "process_evidence": [],
        "terminal_evidence": [{"predicate": "stable_grasp_at_final_stop", "value": "T",
            "source": "explicit analytical terminal record; no physics replay"}],
        "evidence_coverage": {"complete": True}, "completed_current_curve": True}
    trajectory = SimpleNamespace(duration=2., progress_u=np.array([0., 1.]),
                                 progress_s=np.array([0., 1.]), events=[])
    return binding, trajectory, report


@pytest.mark.parametrize("effect,success,outcome", [
    ("T", True, "passed"), ("U", None, "unverified"), ("F", False, "failed"),
])
def test_bound_prior_failure_preserves_metric_and_original_effect_truth(effect, success, outcome):
    """A prior's F is diagnostic; observed effect T/F/U keeps its own meaning."""
    binding, trajectory, report = bound_evidence(effect=effect)
    before = deepcopy((binding, report))
    selected = prior_goal_ids(binding)
    assert selected == [["pose_events", 7]]
    grade = summarize_skill_evidence(report, binding=binding, trajectory=trajectory,
                                     diagnostic_goal_ids=selected)
    assert grade["physics_success"] is success and grade["physics_outcome"] == outcome
    prior, required = grade["functional_goal_evidence"]
    assert prior["value"] == "F" and prior["evaluation_role"] == "diagnostic"
    assert prior["measurement"]["position_error_m"] == pytest.approx(.03)
    assert prior["position_tolerance_m"] == .01
    assert prior["source"]["evidence_frames"] == [2, 5]
    assert prior["source"]["pose_objective"]["role"] == "prior"
    assert required["value"] == "T" and required["evaluation_role"] == "required"
    assert grade["effect_evidence"][0]["value"] == effect
    assert grade["effect_evidence"][0]["evaluation_role"] == "required"
    assert (binding, report) == before


def test_required_pose_remains_required_when_independent_effect_is_true():
    """The explicit required allowed set is retained by original aggregation."""
    binding, trajectory, report = bound_evidence(required_offset_m=.03)
    grade = summarize_skill_evidence(report, binding=binding, trajectory=trajectory,
                                     diagnostic_goal_ids=prior_goal_ids(binding))
    assert grade["physics_success"] is False and grade["physics_outcome"] == "failed"
    assert grade["evaluation_policy"]["required_goal_ids"] == [["pose_events", 11]]
    required = grade["functional_goal_evidence"][1]
    assert required["value"] == "F" and required["evaluation_role"] == "required"
    assert grade["effect_evidence"][0]["value"] == "T"


def test_missing_diagnostic_identity_would_restore_required_grading():
    """The actual caller must pass bound roles into the existing evaluator API."""
    binding, trajectory, report = bound_evidence()
    grade = summarize_skill_evidence(report, binding=binding, trajectory=trajectory)
    assert grade["physics_success"] is False
    assert grade["functional_goal_evidence"][0]["evaluation_role"] == "required"
# END this contrast checks the caller contract, without changing evidence values.


# BEGIN original execution consumes close identity and clock, retaining source roles.
def test_prior_source_survives_executor_reconstruction_without_target_arrival_gate():
    """An estimated TCP disagreement still allows the declared close command.

    The 2 mm scripted error exceeds the fixture's 1 mm target tolerance while
    satisfying its independent robot-feedback contract. The native control
    fixture provides no current target truth and claims no physical grasp.
    """
    fixture = original_simulation.GraspContactActivationTests.fixture()
    simulation, _, _, result, _ = fixture
    binding = result.execution_binding
    original = problem([PoseEventRequirement(**deepcopy(row))
                        for row in binding["problem"]["pose_events"]])
    bound = bind_pose_objectives(original, declarations(original.pose_events, ["prior", "prior"]))
    binding["problem"]["pose_events"] = plain(bound.pose_events)
    binding["validation"]["inputs"]["problem"] = deepcopy(binding["problem"])
    simulation.scripted_tcp_estimate_offset[0] = .002
    report, _ = original_simulation.GraspContactActivationTests.run_trial(fixture)
    assert report["status"] == "curve_finished"
    assert "close" in [row["kind"] for row in report["events"]]
    assert report["grasp_command_checks"][-1]["control_applied"]
    assert all("measurement" not in row for row in report["grasp_command_checks"])
    assert report["completed_goal_ids"] == []
    assert prior_goal_ids(binding) == [["pose_events", 0], ["pose_events", 1]]
    assert binding["problem"]["pose_events"][1]["source"]["kind"] == "frozen_perception_grasp_arrival"
# END command completion and offline diagnostic roles remain separate records.
