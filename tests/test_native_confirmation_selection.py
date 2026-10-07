"""CPU selection regressions for conditional native world-collision checks.

Real cuRobo result objects retain seed/time indexing. Only complete native
geometry metrics are replaced, so these tests exercise production selection
and accounting without loading a GPU model or simulating Coal numerics.
"""
# BEGIN conditional native confirmation must not hide other admissible seeds.
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from manipulation.planning import _CuroboPlanner
import test_manipulation_planning as _planning_fixtures


def selection_fixture(*, scene=(.01, .02), task=(0., 0.), flags=(False, False)):
    """Keep two native seeds distinct while controlling their measured residuals."""
    result = _planning_fixtures.PlannerCandidateSelectionTests.native_candidates(flags=flags)
    constraints = result.metrics.costs_and_constraints.constraints
    constraints.names = ["process_scene", "skill_process"]
    constraints.values = [torch.tensor(values)[:, None, None].expand(2, 9, 1).clone()
                          for values in (scene, task)]
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.scene = object()
    planner.seed_strategies = ["reference", "alternate"]
    planner.last_optimizer_diagnostics = [{"seed": 0}, {"seed": 1}]
    planner.check_request = Mock()
    planner.export = Mock(side_effect=AssertionError("Selection must not deliver a curve"))
    # Seed controls are 1 and 2 in the existing native fixture. Sampling is
    # outside this selection test; preserving seed identity checks the gather.
    planner._native_curve = Mock(side_effect=lambda candidate: SimpleNamespace(
        seed_index=int(candidate.solution[0, 0, 0, 0].item())-1))
    planner._candidate_metrics = Mock(side_effect=AssertionError("Unexpected confirmation"))
    return planner, result


def completed(*, accepted, residual, points):
    """Evidence returned by the shared complete-candidate native metric path."""
    report = {"accepted": accepted, "native_success": accepted,
        "residuals": {"interpolated:process_scene": residual},
        "native_metrics": {"success": accepted},
        "seed_provenance": "supplied_joint_curve",
        "native_confirmation": {"status": "completed"}}
    return report, np.array([0., 1.]), points


def test_smaller_task_failure_cannot_hide_another_confirmable_sphere_seed():
    planner, result = selection_fixture(scene=(0., .02), task=(.00001, 0.))
    planner._candidate_metrics.side_effect = lambda curve, scene: completed(
        accepted=True, residual=0., points=7)
    candidate, curve, report = planner.select_candidate(result)
    assert report["accepted"] and report["selected_seed_index"] == curve.seed_index == 1
    assert candidate.solution[0, 0, 0, 0].item() == 2.
    assert report["confirmation_validation_points"] == 7
    assert not report["candidates"][0]["accepted"]
    assert report["seed_provenance"] == "alternate" and report["optimizer"] == {"seed": 1}
    assert report["sphere_proposal_metrics"]["rollout"]["constraints"]["process_scene"] > 0
    planner._candidate_metrics.assert_called_once_with(curve, planner.scene)
    planner.export.assert_not_called()


def test_confirmed_real_collisions_remain_rejected_and_charge_every_query():
    planner, result = selection_fixture(scene=(.02, .01))
    confirmed_order = []

    def confirm(curve, scene):
        confirmed_order.append(curve.seed_index)
        return completed(accepted=False, residual=.003, points=3 if curve.seed_index == 1 else 5)

    planner._candidate_metrics.side_effect = confirm
    _, curve, report = planner.select_candidate(result)
    assert not report["accepted"] and report["selected_seed_index"] == curve.seed_index == 1
    assert confirmed_order == [1, 0], "Existing severity order must be preserved"
    assert report["confirmation_validation_points"] == 8
    assert all(not row["accepted"] for row in report["candidates"])
    assert [row["confirmation_validation_points"] for row in report["candidates"]] == [5, 3]
    planner.export.assert_not_called()


def test_native_accepted_seed_does_not_repeat_complete_geometry_validation():
    planner, result = selection_fixture(scene=(.01, 0.), flags=(False, True))
    _, curve, report = planner.select_candidate(result)
    assert report["accepted"] and curve.seed_index == report["selected_seed_index"] == 1
    assert report["confirmation_validation_points"] == 0
    planner._candidate_metrics.assert_not_called()
    planner._native_curve.assert_called_once()
    planner.export.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_raw_scene_metrics_never_become_confirmation_candidates(value):
    planner, result = selection_fixture(scene=(value, 0.), task=(0., .01), flags=(True, False))
    _, _, report = planner.select_candidate(result)
    assert not report["accepted"] and report["confirmation_validation_points"] == 0
    assert not report["candidates"][0]["accepted"]
    planner._candidate_metrics.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), .0001])
def test_confirmation_success_flag_cannot_override_invalid_residual(value):
    planner, result = selection_fixture(scene=(.01, 0.), task=(0., .1))
    planner._candidate_metrics.side_effect = lambda curve, scene: completed(
        accepted=True, residual=value, points=4)
    _, _, report = planner.select_candidate(result)
    assert not report["accepted"], "Confirmation keeps the same finite, nonpositive acceptance rule"
    assert report["confirmation_validation_points"] == 4
    planner.export.assert_not_called()
# END conditional confirmation changes collision evidence, not acceptance rules.
