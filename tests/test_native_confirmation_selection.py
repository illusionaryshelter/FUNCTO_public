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
    # BEGIN same-query diagnostics separate collision, missing field and forbidden coverage.
    result.metrics.debug = torch.cat((constraints.values[0],
        torch.zeros_like(constraints.values[0]), torch.zeros_like(constraints.values[0]),
        torch.zeros_like(constraints.values[0])), dim=-1)
    # END the aggregate alone cannot classify any of these independent requirements.
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
        "collision_residuals": {"interpolated:scene_collision": residual},
        "scene_update_residuals": {"interpolated:scene_update": 0.},
        # BEGIN complete-candidate fixtures carry the independent coverage evidence.
        "coverage_residuals": {"interpolated:unknown_forbidden": 0.},
        "conditional_unknown_residuals": {"interpolated:conditional_unknown": 0.},
        # END proxy confirmation does not imply observed or permitted spatial coverage.
        "native_metrics": {"success": accepted},
        "seed_provenance": "supplied_joint_curve",
        "native_confirmation": {"status": "completed"}}
    return report, np.array([0., 1.]), points


def test_smaller_task_failure_cannot_hide_another_confirmable_sphere_seed():
    planner, result = selection_fixture(scene=(0., .02), task=(.00001, 0.))
    planner._candidate_metrics.side_effect = lambda curve, scene, *, native_query: completed(
        accepted=True, residual=0., points=7)
    candidate, curve, report = planner.select_candidate(result)
    assert report["accepted"] and report["selected_seed_index"] == curve.seed_index == 1
    assert candidate.solution[0, 0, 0, 0].item() == 2.
    assert report["confirmation_validation_points"] == 7
    assert report["collision_residuals"] == {"interpolated:scene_collision": 0.}
    assert not report["candidates"][0]["accepted"]
    assert report["seed_provenance"] == "alternate" and report["optimizer"] == {"seed": 1}
    assert report["sphere_proposal_metrics"]["rollout"]["constraints"]["process_scene"] > 0
    planner._candidate_metrics.assert_called_once_with(curve, planner.scene, native_query=True)
    planner.export.assert_not_called()


def test_confirmed_real_collisions_remain_rejected_and_charge_every_query():
    planner, result = selection_fixture(scene=(.02, .01))
    confirmed_order = []

    def confirm(curve, scene, *, native_query):
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
    planner._candidate_metrics.side_effect = lambda curve, scene, *, native_query: completed(
        accepted=True, residual=value, points=4)
    _, _, report = planner.select_candidate(result)
    assert not report["accepted"], "Confirmation keeps the same finite, nonpositive acceptance rule"
    assert report["confirmation_validation_points"] == 4
    planner.export.assert_not_called()
# END conditional confirmation changes collision evidence, not acceptance rules.


# BEGIN collision diagnostics cannot change the existing severity ordering.
def test_scene_support_severity_selects_same_seed_despite_opposite_collision_order():
    planner, result = selection_fixture(scene=(.02, .01), task=(.1, .1))
    # BEGIN retain all four diagnostic channels while testing collision ordering alone.
    result.metrics.debug = torch.stack((torch.zeros((9, 4)), torch.ones((9, 4))))
    # END channel count matches the production metric contract.
    result.metrics.debug[0, :, 1] = 100000.
    _, curve, report = planner.select_candidate(result)
    assert not report["accepted"] and curve.seed_index == report["selected_seed_index"] == 1
    assert report["collision_residuals"] == {"rollout:scene_collision": 1.}
    assert report["residuals"]["rollout:process_scene"] == pytest.approx(.01)
    planner._candidate_metrics.assert_not_called()
# END diagnostics are absent from acceptance and candidate severity sums.


# BEGIN a candidate support gap cannot override another existing feasible seed.
def test_unsupported_seed_does_not_block_another_accepted_seed():
    planner, result = selection_fixture(scene=(100000., 0.), flags=(False, True))
    result.metrics.debug.zero_()
    result.metrics.debug[0, :, 1] = 100000.
    _, curve, report = planner.select_candidate(result)
    assert report["accepted"] and curve.seed_index == 1
    assert report["candidates"][0]["scene_update_residuals"] == {"rollout:scene_update": 100000.}
    assert report["scene_update_residuals"] == {"rollout:scene_update": 0.}
    planner._candidate_metrics.assert_not_called()


def test_native_confirmation_report_replaces_same_candidate_support_evidence():
    planner, result = selection_fixture(scene=(.01, 0.), task=(0., .1))
    result.metrics.debug[0, :, 0] = 0.
    result.metrics.debug[0, :, 1] = .01
    def confirm(curve, scene, *, native_query):
        report, parameters, points = completed(accepted=False, residual=1., points=4)
        report["collision_residuals"] = {"interpolated:scene_collision": 0.}
        report["scene_update_residuals"] = {"interpolated:scene_update": 1.}
        return report, parameters, points
    planner._candidate_metrics.side_effect = confirm
    _, _, report = planner.select_candidate(result)
    assert not report["accepted"]
    assert report["scene_update_residuals"] == {"interpolated:scene_update": 1.}
    assert report["sphere_proposal_metrics"]["rollout"]["scene_update_residuals"]["scene_update"] == pytest.approx(.01)
# END query evidence stays attached to its producer when confirmation replaces a report.


# BEGIN clearing a collision proposal must preserve the independent forbidden-space failure.
def test_proxy_clear_keeps_confirmed_coverage_evidence_for_bounded_search():
    """Selection must expose the remaining route constraint without calling it collision.

    Raw and confirmed samples can measure different maxima. The complete
    candidate's coverage therefore replaces its earlier value, while the
    proposal report remains available to distinguish the two query scopes.
    The outer routing tests exercise OMPL; here the same selected candidate
    must retain exactly the typed evidence that makes that routing possible.
    """
    planner, result = selection_fixture(scene=(.07, 0.), task=(0., .1))
    result.metrics.debug[0, :, 0] = .02
    result.metrics.debug[0, :, 2] = .05

    def confirm(curve, scene, *, native_query):
        report, parameters, points = completed(accepted=False, residual=.04, points=4)
        report["collision_residuals"] = {"interpolated:scene_collision": 0.}
        report["coverage_residuals"] = {"interpolated:unknown_forbidden": .04}
        return report, parameters, points

    planner._candidate_metrics.side_effect = confirm
    _, curve, report = planner.select_candidate(result)
    assert not report["accepted"] and curve.seed_index == report["selected_seed_index"] == 0
    assert report["collision_residuals"] == {"interpolated:scene_collision": 0.}
    assert report["scene_update_residuals"] == {"interpolated:scene_update": 0.}
    assert report["coverage_residuals"] == {"interpolated:unknown_forbidden": .04}
    assert report["candidates"][0]["coverage_residuals"] == report["coverage_residuals"]
    proposal = report["sphere_proposal_metrics"]["rollout"]
    assert proposal["collision_residuals"]["scene_collision"] == pytest.approx(.02)
    assert proposal["coverage_residuals"]["unknown_forbidden"] == pytest.approx(.05)
    assert report["confirmation_validation_points"] == 4
    planner._candidate_metrics.assert_called_once_with(curve, planner.scene, native_query=True)
    planner.export.assert_not_called()
# END search evidence never weakens the original hard acceptance or grants an execution binding.


# BEGIN every accepted route retains unknown evidence from the selected queries.
def test_native_feasible_seed_keeps_conditional_unknown_without_confirmation():
    planner, result = selection_fixture(scene=(.01, 0.), flags=(False, True))
    result.metrics.debug[0, :, 3] = .08
    result.metrics.debug[1, :, 3] = .04
    _, curve, report = planner.select_candidate(result)
    assert report["accepted"] and curve.seed_index == 1
    assert report["conditional_unknown_residuals"]["rollout:conditional_unknown"] == pytest.approx(.04)
    assert report["candidates"][0]["conditional_unknown_residuals"]["rollout:conditional_unknown"] == pytest.approx(.08)
    planner._candidate_metrics.assert_not_called()


def test_native_confirmation_replaces_unknown_evidence_with_same_candidate_queries():
    planner, result = selection_fixture(scene=(.01, 0.), task=(0., .1))
    result.metrics.debug[0, :, 3] = .08

    def confirm(curve, scene, *, native_query):
        report, parameters, points = completed(accepted=True, residual=0., points=4)
        report["conditional_unknown_residuals"] = {"interpolated:conditional_unknown": .04}
        return report, parameters, points

    planner._candidate_metrics.side_effect = confirm
    _, curve, report = planner.select_candidate(result)
    assert report["accepted"] and curve.seed_index == 0
    assert report["conditional_unknown_residuals"] == {"interpolated:conditional_unknown": .04}
    assert report["candidates"][0]["conditional_unknown_residuals"] == report["conditional_unknown_residuals"]
    assert report["sphere_proposal_metrics"]["rollout"]["conditional_unknown_residuals"]["conditional_unknown"] == pytest.approx(.08)
    assert report["confirmation_validation_points"] == 4
# END confirmation may clear a proxy collision while preserving conditional uncertainty.
