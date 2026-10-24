# BEGIN describe the actual qualification scope of these CPU fixtures.
"""CPU contracts for immutable curves and complete candidate qualification.

Fixtures provide completed sampled and interval geometry evidence. Production
request accounting, clock binding, Hermite interpolation and continuous joint
domain checks determine the resulting candidate and its validation record.
"""
# END geometric accuracy has separate native-scene and interval test coverage.
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

# BEGIN qualification tests use the production cumulative-budget stop type.
from manipulation.planning import (
    JointTrajectory, PlanningBudget, PlanningWorkExhausted, _ConstrainedPlanner, _ProgressMap,
    _candidate_sample_times, _phase_aligned_parameters,
)
# END the same exception reaches the real request's incumbent delivery path.
from manipulation.planning_problem import PlanningEvent
from test_motion_backend_contract import Native, Scene, problem, run


# BEGIN load the production nested adapter without constructing CUDA resources.
# END execute the actual adapter and installed superclass, not a copied repair.


def supplied_curve():
    times = np.array([0., .13, .71, 1.])
    return JointTrajectory(("q",), times, (.1*times**2)[:, None], (.2*times)[:, None],
        np.array([0., .3, .6, 1.]), np.array([0., .5, .5, 1.]), (), (), {})


# BEGIN the candidate fixture separates completed sampled and interval evidence.
def candidate_fixture(*, accepted=True, components=None, native_outcomes=(), interval_residual=0.):
    """Exercise real qualification and accounting with explicit geometry results."""
    native = Native(native_outcomes)
    native.problem, native.scene, native.joint_names = problem(), Scene(), ("q",)
    native.start = SimpleNamespace(position=torch.zeros((1, 1)), velocity=torch.zeros((1, 1)))
    native.anchors = np.array([0., 1.])
    native.physical_domain = SimpleNamespace(_curve_map=SimpleNamespace(start_q=np.zeros(1), start_dq=np.zeros(1)))
    for name in ("bind_reference_candidate", "evaluate_candidate", "_candidate_domain_validation"):
        setattr(native, name, MethodType(getattr(_ConstrainedPlanner, name), native))

    def completed_metrics(curve, scene, *, record_work):
        native.calls.append("candidate")
        record_work(points=4)
        return {"accepted": accepted, "native_success": accepted,
            "residuals": {"interpolated:process_scene": 0. if accepted else 1.},
            "scene_components": components or {}}, np.array([0., 1.]), 4, (scene, [])
    native._candidate_metrics = Mock(side_effect=completed_metrics)
    # BEGIN fixed objective evidence has the same explicit fixture ownership.
    native._candidate_objective = Mock(return_value={
        "fixed_objective": 1., "objective_components": {"fixture_reference": 1.}})
    # END native cost numerics are covered by the reference-tracking tests.
    # Interval geometry is a separate required consumer. These two completed
    # endpoint queries are fixture evidence; production joint-domain checks,
    # residual combination and remaining query-budget arithmetic stay real.
    def completed_geometry(curve, scene, *, initial_states, remaining_points, record_work):
        record_work(points=2)
        return {"query_points": 2, "hard_scene_residual": interval_residual,
            "scene_categories": {"collision": interval_residual}, "scene_components": {},
            "scope": "CPU fixture of completed interval geometry evidence"}
    native._candidate_geometry = Mock(side_effect=completed_geometry)
    return native
# END geometry accuracy is covered by the native scene and interval test modules.


# BEGIN a complete supplied curve is delivered before any new initialization.
def test_supplied_curve_is_immutable_and_skips_all_later_initialization():
    reference = supplied_curve()
    originals = [value.copy() for value in (reference.times, reference.position, reference.velocity)]
    native = candidate_fixture()

    native.reference_seeds = Mock(side_effect=AssertionError("A qualified original needs no initializer"))
    result, backend = run(native, joint_reference=reference)
    try:
        assert result.success, result.diagnostics
        assert native.calls == ["candidate"]
        native.reference_seeds.assert_not_called()
        # The returned record owns its curve even after reusable buffers move.
        native.delivery_candidate.position[:] = .8
        native.last_validation = None
        for actual, original in zip((result.trajectory.times, result.trajectory.position, result.trajectory.velocity), originals):
            np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(result.trajectory.progress_u, reference.progress_u)
        np.testing.assert_array_equal(result.trajectory.progress_s, reference.progress_s)
        assert result.diagnostics["joint_reference"]["source_curve_preserved"]
        assert result.diagnostics["returned_candidate"]["fixed_objective"] == 1.
        assert "failure" not in result.diagnostics
        counts = result.request_ledger["consumed_planning"]
        assert counts["ik_batches"] == 0 and counts["trajectory_solves"] == 0
        assert counts["validation_passes"] == 1
        assert result.execution_binding["request_ledger"]["consumed_planning"] == counts
    finally:
        backend.close()
# END the original polynomial and progress map keep their completed qualification.


# BEGIN first qualification retains the supplied curve's complete physical clock.
def test_supplied_curve_keeps_its_own_clock_without_scoring_native_candidates():
    reference = supplied_curve()
    native = candidate_fixture(native_outcomes=(True, True))
    native.joint_reference_report = {"source_curve_preserved": False,
        "source_duration_s": reference.duration, "bound_duration_s": 2.}
    native.curve = JointTrajectory(("q",), np.array([0., 2.]), np.zeros((2, 1)),
        np.zeros((2, 1)), np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    native.select_candidate = Mock(side_effect=AssertionError("The supplied curve already qualified"))
    result, backend = run(native, joint_reference=reference)
    try:
        assert result.success, result.diagnostics
        assert native.calls == ["candidate"]
        native.select_candidate.assert_not_called()
        np.testing.assert_array_equal(result.trajectory.times, reference.times)
        np.testing.assert_array_equal(result.trajectory.position, reference.position)
        np.testing.assert_array_equal(result.trajectory.velocity, reference.velocity)
        np.testing.assert_array_equal(result.trajectory.progress_u, reference.progress_u)
        np.testing.assert_array_equal(result.trajectory.progress_s, reference.progress_s)
        assert result.diagnostics["joint_reference"]["source_curve_preserved"]
        assert result.diagnostics["joint_reference"]["bound_duration_s"] == reference.duration
        assert result.diagnostics["returned_candidate"]["fixed_objective"] == 1.
        assert result.diagnostics["attempts"] == []
        assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 0
    finally:
        backend.close()
# END no old callback adapter or unused quality-improvement route remains.


# BEGIN supplied curves obey the same explicit policy as native candidate routes.
def test_require_observation_returns_original_without_spatial_repair():
    gaps = {"unknown/forbidden/robot": {"maximum": .004, "time_s": .71, "progress": .6}}
    native = candidate_fixture(accepted=False, components=gaps)
    native.scene.unknown_policy = {"outside_regions": "require_observation", "conditional_regions": []}
    reference = supplied_curve()
    result, backend = run(native, scene=native.scene, joint_reference=reference)
    try:
        assert not result.success and result.execution_binding is None
        assert result.diagnostics["failure"] == "needs_observation"
        assert result.diagnostics["reference_candidate"]["scene_components"] == gaps
        assert native.calls == ["candidate"]
        np.testing.assert_array_equal(result.diagnostics["diagnostic_trajectory"].position, reference.position)
    finally:
        backend.close()


def test_forbidden_unknown_on_supplied_curve_enters_existing_finite_repair():
    gaps = {"unknown/forbidden/robot": {"maximum": .004, "time_s": .71, "progress": .6}}
    native = candidate_fixture(accepted=False, components=gaps, native_outcomes=(False, True))
    reference = supplied_curve()
    result, backend = run(native, scene=native.scene, joint_reference=reference)
    try:
        assert result.success, result.diagnostics
        assert native.calls == ["candidate", "seed", "reference", "solve", "export"]
        assert result.diagnostics["original_reference_candidate"]["scene_components"] == gaps
        assert not result.diagnostics["original_reference_candidate"]["accepted"]
        assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 1
    finally:
        backend.close()


@pytest.mark.parametrize("mode", ["forbid", "require_observation", "allow_attempt"])
def test_missing_model_support_still_rejects_supplied_curve_before_spatial_repair(mode):
    gaps = {"scene_update/robot/background": {"maximum": .004, "time_s": .71, "progress": .6}}
    native = candidate_fixture(accepted=False, components=gaps)
    native.scene.unknown_policy = {"outside_regions": mode, "conditional_regions": []}
    reference = supplied_curve()
    result, backend = run(native, scene=native.scene, joint_reference=reference)
    try:
        assert not result.success and result.execution_binding is None
        assert result.diagnostics["failure"] == "reference_outside_observation_scope"
        assert result.diagnostics["reference_scope_gaps"] == gaps
        assert native.calls == ["candidate"]
        np.testing.assert_array_equal(result.diagnostics["diagnostic_trajectory"].position, reference.position)
    finally:
        backend.close()
# END a policy may alter unknown handling but cannot clear missing numerical support.


def test_supported_collision_enters_existing_finite_repair_only_after_original_check():
    native = candidate_fixture(accepted=False, components={"collision/robot/object": {"maximum": .01}},
                               native_outcomes=(False, True))
    result, backend = run(native, joint_reference=supplied_curve())
    try:
        assert result.success, result.diagnostics
        assert native.calls == ["candidate", "seed", "reference", "solve", "export"]
        assert not result.diagnostics["original_reference_candidate"]["accepted"]
    finally:
        backend.close()


def test_phase_mapping_keeps_moving_plateau_and_first_arrival_of_added_hold():
    source = _ProgressMap(np.array([0., .3, .6, 1.]), np.array([0., .5, .5, 1.]))
    target = _ProgressMap(np.array([0., .2, .7, .9, 1.]), np.array([0., .5, .5, 1., 1.]))
    np.testing.assert_allclose(_phase_aligned_parameters(source, target, np.array([.3, .45, .6, 1.])),
                               [.2, .45, .7, .9], rtol=0, atol=2e-16)
    with pytest.raises(Exception, match="plateau motion"):
        _phase_aligned_parameters(source, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), np.array([.45]))


def test_nonuniform_query_grid_retains_every_original_piece_and_bounds_spacing():
    reference = supplied_curve()
    parameters = _candidate_sample_times(reference, PlanningBudget())
    assert set(reference.times) <= set(parameters)
    assert np.diff(parameters).max() <= float(np.float32(.025))
    assert parameters[0] == 0. and parameters[-1] == reference.duration
    q, dq, ddq, _ = reference.sample(parameters)
    np.testing.assert_allclose(q[:, 0], .1*parameters**2, atol=1e-15)
    np.testing.assert_allclose(dq[:, 0], .2*parameters, atol=1e-15)
    np.testing.assert_allclose(ddq[:, 0], .2, atol=1e-14)


# BEGIN exact saved-curve metrics and interval queries share one qualification.
@pytest.mark.parametrize("interval_residual", [0., 1e-12, .01])
def test_saved_curve_queries_sampled_and_interval_evidence(interval_residual):
    native = candidate_fixture(interval_residual=interval_residual)
    reference = supplied_curve()
    record_work = Mock()
    curve, attempt, points = native.evaluate_candidate(reference, record_work=record_work)
    assert attempt["accepted"] is (interval_residual == 0.)
    assert points == 6
    assert attempt["residuals"]["interval:process_scene"] == interval_residual
    native._candidate_metrics.assert_called_once_with(reference, native.scene, record_work=record_work)
    native._candidate_geometry.assert_called_once_with(reference, native.scene,
        initial_states=[], remaining_points=native.budget.max_validation_points-4, record_work=record_work)
    assert record_work.call_count == 3
    if interval_residual == 0.:
        assert native.last_validation["cubic_derivatives"]["acceleration"]["maximum_bound_excess"] == 0
        assert native.last_validation["continuous_geometry"] == attempt["continuous_geometry"]
    else:
        assert native.last_validation is None
    assert curve is native.delivery_candidate
# END a positive interval witness rejects the curve despite clear sampled metrics.


# BEGIN first-result delivery retains the original finite quality requirement.
@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_nonfinite_fixed_quality_cannot_qualify_a_clear_curve(score):
    native = candidate_fixture()
    native._candidate_objective.return_value = {"fixed_objective": score}
    _, report, _ = native.evaluate_candidate(supplied_curve(), record_work=Mock())
    assert not report["accepted"] and native.last_validation is None
    assert not np.isfinite(report["fixed_objective"])
# END recorded task quality is not converted into an optional acceptance check.




# BEGIN native direct-solve control flow shares the same buffer admission.
# END no prior candidate evaluation is required to initialize a direct solve.




def test_terminal_evidence_after_last_tick_is_typed_unchecked_without_native_allocation():
    from manipulation.planning import MotionBackend
    request = problem()
    request.events = (PlanningEvent("terminal", "skill", 1., 197, "verification", "recorded terminal",
                                   "actual evidence", True),)
    duration = 9.850000000000293
    reference = JointTrajectory(("q",), np.array([0., duration]), np.zeros((2, 1)), np.zeros((2, 1)),
        np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    scene = Scene()
    scene.gripper = SimpleNamespace(control_timestep_s=.002)
    backend = MotionBackend({}, robot_validator=object())
    with patch("manipulation.planning._ConstrainedPlanner", side_effect=AssertionError("must reject before native allocation")) as factory:
        result = backend.plan(request, scene, joint_reference=reference)
    assert not result.success and result.execution_binding is None
    assert result.diagnostics["failure"] == "reference_clock_outside_curve"
    issue = result.diagnostics["reference_clock_issues"][0]
    assert issue["device_step"] == 4926 and issue["device_time_s"] == 9.852
    assert issue["curve_duration_s"] == duration and issue["status"] == "outside_curve_unchecked"
    assert result.request_ledger["consumed_planning"]["initializations"] == 0
    factory.assert_not_called()
    np.testing.assert_array_equal(reference.times, [0., duration])
    backend.close()


# BEGIN native robot pair policy is owned by the registered scene model.
# test_native_scene_contract.test_scene_queries_only_required_pairs_and_serializes_complete_policy
# exercises required, excluded and mechanical-contact pairs through the current
# model query and serialization interface. This module covers candidate delivery.
# END the existing model-policy regression owns this separate component contract.
