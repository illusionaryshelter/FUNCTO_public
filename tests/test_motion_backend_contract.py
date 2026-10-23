"""CPU contracts for resident ownership and bounded motion-request decisions.

The native solver is replaced only at its constructor. These tests exercise the
real request ledger, selection orchestration and delivery boundary, not GPU
convergence or geometry accuracy.
"""
# BEGIN resident motion backend regression coverage.
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manipulation.planning import (
    JointTrajectory, MotionBackend, MotionPlanningError, PlanningBudget,
    _ConstrainedPlanner, _ProgressMap, _bind_motion_work_ledger, _verified_dependency_outcomes,
)
from manipulation.planning_problem import MotionPhase, PlanningProblem, PosePath


def problem():
    path = PosePath(np.array([0., 1.]), np.stack([np.eye(4), np.eye(4)]), "reference")
    return PlanningProblem(("q",), [0.], [0.], "tool", "world", np.eye(4), np.eye(4),
        (MotionPhase("phase", path),), (), source_timestamps=np.array([0., 1.]),
        source_progress=np.array([0., 1.]))


class Scene:
    gripper = None
    # BEGIN every production scene declares its unknown-space policy explicitly.
    unknown_policy = {"outside_regions": "forbid", "conditional_regions": []}
    # END orchestration fixtures do not rely on a missing-policy fallback.
    def clone_for_rollout(self):
        return deepcopy(self)
    def describe(self):
        return {"kind": "cpu_test_online_scene"}


class Native:
    """Deterministic solver outcomes; no geometry predicate is simulated."""
    def __init__(self, outcomes, *, collision=False, ompl=True):
        self.outcomes = iter(outcomes)
        self.collision, self.ompl_available = collision, ompl
        self.calls = []
        # BEGIN give the lazy planning-only adapter its named robot fixture.
        # These orchestration tests never query geometry; model access would
        # be a test error rather than an execution-validator factory fallback.
        self.check_request = lambda: None
        self.robot_validator = SimpleNamespace()
        # END geometry correctness remains covered by native confirmation tests.
        self.budget = PlanningBudget()
        core = lambda: SimpleNamespace(optimizers=[])
        self.ik = SimpleNamespace(config=SimpleNamespace(exit_early=False), core=core())
        self.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        self.scene_workspace_bytes = self.scene_cpu_workspace_bytes = self.scene_gpu_workspace_bytes = 0
        self.anchor_fk_workspace_bytes = 0
        self.anchor_capacity = 256
        self.validation_storage = {}
        self.route = "constrained"
        self.reference_ik_distinct_counts = self.reference_ik_queries = self.reference_ik_chains = []
        self.goal_ik_candidates = {}
        self.reference_initialization = {"status": "completed", "scene_constraints": "not_evaluated"}
        self.main_trajectory_iterations = self.budget.trajectory_iterations
        self.joint_reference_report = None
        self.initial_seed_strategies = self.seed_strategies = ["test", "test"]
        self.last_validation = {"u": np.array([0., 1.]), "violations": np.zeros(2)}
        limits = SimpleNamespace(joint_names=["q"], position=torch.tensor([[-1.], [1.]]),
            velocity=torch.tensor([[-1.], [1.]]), acceleration=torch.tensor([[-1.], [1.]]),
            jerk=torch.tensor([[-10.], [10.]]))
        self._verification_native = SimpleNamespace(get_cost=lambda _: SimpleNamespace(config=SimpleNamespace(joint_limits=limits)))
        self.curve = JointTrajectory(("q",), np.array([0., 1.]), np.zeros((2, 1)), np.zeros((2, 1)),
                                    np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    def initial_state_diagnostics(self):
        return {"passed": True}
    def reference_seeds(self, joint_reference=None):
        self.calls.append("seed")
        return torch.zeros((1, 2, 8, 1)), []
    # BEGIN request fixtures use the same terminal-evidence dispatch predicate.
    # END missing terminal evidence preserves the existing qualification route.
    def evaluate_reference(self, seed):
        self.calls.append("reference")
        return next(self.outcomes)
    def optimize(self, seed, *, iterations=None):
        self.calls.append("solve")
        return next(self.outcomes)
    def select_candidate(self, accepted, *, record_work):
        # BEGIN even a zero-query fixture starts one formal validation attempt.
        record_work(passes=1)
        # END query counts are charged only by actual completed fixture queries.
        report = {"accepted": accepted, "native_success": accepted, "seed_provenance": "test",
                  "fixed_objective": 1., "objective_components": {"motion_quality": 1.},
                  "residuals": {"process_scene" if self.collision else "skill_process": 0. if accepted else 1.},
                  # BEGIN recovery evidence is distinct from the hard scene aggregate.
                  "collision_residuals": {"rollout:scene_collision": float(self.collision and not accepted)},
                  "scene_update_residuals": {"rollout:scene_update": 0.},
                  "coverage_residuals": {"rollout:unknown_forbidden": 0.},
                  "conditional_unknown_residuals": {"rollout:conditional_unknown": 0.}}
                  # END the deterministic fixture supplies explicit query classification.
        return SimpleNamespace(solution=torch.zeros((1, 1, 8, 1))), self.curve, report
    def ompl_seed(self, seed, trajectory):
        self.calls.append("ompl")
        return (seed if self.ompl_available else None), {"state_queries": 1}
    # BEGIN the default fixture has no uncovered dense collision samples.
    # END dedicated recovery tests supply new physical-time evidence below.
    def export(self, result, native_report):
        # BEGIN delivery consumes the already completed candidate metrics.
        assert native_report["accepted"]
        self.last_validation = {"u": np.array([0., 1.]), "violations": np.zeros(2)}
        # END no second synthetic geometry verdict is introduced here.
        self.calls.append("export")
        return self.curve, 0
    def rebind(self, problem, scene, budget, **kwargs):
        self.calls.append("rebind")
        return self
    def close(self):
        self.calls.append("close")


def run(native, *, budget=None, scene=None, **kwargs):
    backend = MotionBackend({}, robot_validator=object(), budget=budget)
    with patch("manipulation.planning._ConstrainedPlanner", return_value=native), \
         patch("manipulation.execution_binding.bind_candidate", return_value={"candidate_id": "accepted"}):
        result = backend.plan(problem(), Scene() if scene is None else scene, **kwargs)
    return result, backend


def test_feasible_reference_is_retained_through_one_main_improvement():
    native = Native([True, True])
    result, backend = run(native)
    assert result.success and native.calls == ["seed", "reference", "export", "solve"]
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 1
    assert result.execution_binding["verification_joint_limits"]["joint_names"] == ["q"]
    assert result.diagnostics["joint_state_computation"] == {
        "kinematic_derivative_orders": [0, 1, 2, 3], "effort": "not_computed"}
    assert backend._planner is native
    backend.close()
    assert native.calls[-1] == "close"


# BEGIN exercise the real serialization handoff after native candidate selection.
def test_real_candidate_binding_preserves_observed_geometry_and_rejects_mutation(tmp_path):
    import json
    from manipulation.execution_binding import check_result
    from manipulation.observed_mapping import FrozenObservedField
    from test_native_scene_contract import make_scene

    # BEGIN the native candidate retains the current unified scene record.
    scene = make_scene(tmp_path)
    native = Native([True, True])
    backend = MotionBackend({}, robot_validator=object())
    try:
        with patch("manipulation.planning._ConstrainedPlanner", return_value=native):
            result = backend.plan(problem(), scene)
        assert result.success and native.calls.count("export") == 1
        binding = check_result(result)
        assert binding["scene"] == json.loads(json.dumps(scene.to_execution_binding()))
        restored = FrozenObservedField.from_record(binding["scene"]["background"])
        np.testing.assert_array_equal(restored.values, scene.background.values)
        np.testing.assert_array_equal(restored.observed, scene.background.observed)
        np.testing.assert_array_equal(restored.weights, scene.background.weights)
        result.execution_binding["scene"]["background"]["origin"][0] += .01
        with pytest.raises(ValueError, match="validated input binding"):
            check_result(result)
    finally:
        backend.close()
    # END field serialization retains raw samples and mutation detection.
# END accepted evidence names exactly the field record handed to execution.


# BEGIN only typed geometric failures may request the bounded second solve.
def test_failed_main_collision_gets_one_bounded_recovery():
    native = Native([False, False, True], collision=True)
    result, backend = run(native)
    assert result.success
    assert native.calls.count("solve") == 2 and native.calls.count("export") == 1
    assert native.calls.count("ompl") == 1
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 2
    backend.close()


def test_task_only_failure_does_not_add_a_full_solve():
    native = Native([False, False])
    result, backend = run(native)
    assert not result.success and result.execution_binding is None
    assert native.calls.count("solve") == 1 and "export" not in native.calls
    assert result.diagnostics["recovery"] is None
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 1
    backend.close()
# END functional continuation runs inside each original native solve.


def test_recovery_failure_returns_no_execution_binding_or_export():
    native = Native([False, False, False], collision=True)
    result, backend = run(native)
    assert not result.success and result.execution_binding is None
    assert result.diagnostics["failure"] == "no_feasible_solution_found"
    assert result.diagnostics["diagnostic_trajectory"] is native.curve
    assert native.calls.count("solve") == 2 and "export" not in native.calls
    backend.close()


# BEGIN a model-support gap cannot spend the collision recovery branch.
def test_positive_scene_support_only_preserves_hard_failure_and_skips_collision_recovery():
    native = Native([False, False, False])
    select = native.select_candidate
    def missing_support(accepted, *, record_work):
        candidate, curve, report = select(accepted, record_work=record_work)
        report["residuals"] = {"rollout:process_scene": 400000., "interpolated:process_scene": 500000.}
        report["collision_residuals"] = {"rollout:scene_collision": 0., "interpolated:scene_collision": 0.}
        report["scene_update_residuals"] = {"rollout:scene_update": 400000., "interpolated:scene_update": 500000.}
        return candidate, curve, report
    native.select_candidate = missing_support
    result, backend = run(native)
    try:
        assert not result.success and result.execution_binding is None
        assert "ompl" not in native.calls and "export" not in native.calls
        assert native.calls.count("solve") == 1
        assert result.diagnostics["failure"] == "no_feasible_solution_found"
        assert result.diagnostics["scene_update_residuals"]["interpolated:scene_update"] == 500000.
        assert "failure_scope" in result.diagnostics
    finally:
        backend.close()
# END one rejected exploration candidate does not prove a request-wide scene gap.


# BEGIN forbidden coverage searches a new path without inventing collision evidence.
@pytest.mark.parametrize("search_succeeds", [False, True])
def test_forbidden_coverage_uses_one_bounded_search_without_collision_witnesses(search_succeeds):
    native = Native([False, False, True], ompl=search_succeeds)
    select = native.select_candidate
    def covered(accepted, *, record_work):
        candidate, curve, report = select(accepted, record_work=record_work)
        report["residuals"] = {"interpolated:process_scene": 0. if accepted else 4000.}
        report["coverage_residuals"] = {"interpolated:unknown_forbidden": 0. if accepted else 4000.}
        # BEGIN final policy classification consumes named same-candidate evidence.
        report["scene_components"] = {} if accepted else {"unknown/forbidden/robot": {"maximum": 4000.}}
        # END the aggregate coverage channel alone is insufficient for a policy verdict.
        return candidate, curve, report
    native.select_candidate = covered
    result, backend = run(native)
    try:
        assert result.success is search_succeeds
        assert native.calls.count("ompl") == 1
        assert native.calls.count("solve") == (2 if search_succeeds else 1)
        assert result.diagnostics["recovery"] == ("ompl" if search_succeeds else "ompl_failed")
        ledger = result.request_ledger
        assert ledger["consumed_planning"]["search_queries"] == 1
        assert ledger["consumed_planning"]["trajectory_solves"] == (2 if search_succeeds else 1)
        assert ledger["planning_budget"]["maximum_trajectory_solves"] == 2
        assert ledger["planning_budget"]["planning_seconds"] == 120.
        if not search_succeeds:
            assert result.execution_binding is None and "export" not in native.calls
            assert result.diagnostics["coverage_residuals"]["interpolated:unknown_forbidden"] > 0
            assert result.diagnostics["failure"] == "unknown_space_forbidden"
    finally:
        backend.close()


def test_forbidden_coverage_cannot_replenish_a_zero_search_budget():
    native = Native([False, False])
    select = native.select_candidate
    def forbidden(accepted, *, record_work):
        candidate, curve, report = select(accepted, record_work=record_work)
        report["residuals"] = {"interpolated:process_scene": 1.}
        report["coverage_residuals"] = {"interpolated:unknown_forbidden": 1.}
        return candidate, curve, report
    native.select_candidate = forbidden
    result, backend = run(native, budget=PlanningBudget(search_seconds=0.))
    try:
        assert not result.success and result.execution_binding is None
        assert "ompl" not in native.calls and "export" not in native.calls
        assert native.calls.count("solve") == 1
        assert result.request_ledger["consumed_planning"]["search_queries"] == 0
        assert result.request_ledger["planning_budget"]["search_seconds"] == 0.
    finally:
        backend.close()


@pytest.mark.parametrize("component", ["contact/required/rule", "unsupported/contact/robot/owner"])
def test_other_scene_terms_cannot_masquerade_as_forbidden_coverage(component):
    native = Native([False, False])
    select = native.select_candidate
    def other_scene_failure(accepted, *, record_work):
        candidate, curve, report = select(accepted, record_work=record_work)
        report["residuals"] = {"interpolated:process_scene": 1.}
        report["scene_components"] = {component: {"maximum": 1.}}
        return candidate, curve, report
    native.select_candidate = other_scene_failure
    result, backend = run(native)
    try:
        assert not result.success and result.execution_binding is None
        assert "ompl" not in native.calls and "export" not in native.calls
        assert native.calls.count("solve") == 1
    finally:
        backend.close()
# END search eligibility leaves the original hard result and request ledger intact.


# BEGIN observation policy returns evidence without buying more unproductive motion work.
@pytest.mark.parametrize("stage", ["initial", "reference", "main"])
def test_require_observation_stops_at_first_confirmed_coverage_only_candidate(stage):
    native = Native([False, False])
    world = Scene()
    world.unknown_policy = {"outside_regions": "require_observation", "conditional_regions": []}
    unknown = {"unknown/forbidden/robot": {"maximum": .04},
               "conditional/unknown/payload": {"maximum": .02}}
    if stage == "initial":
        native.initial_state_diagnostics = lambda: {
            "passed": False, "residuals": {"scene": .04, "position": 0., "velocity": 0.},
            "scene_components": unknown}
    else:
        select = native.select_candidate
        selections = []

        def confirmed(accepted, *, record_work):
            candidate, curve, report = select(accepted, record_work=record_work)
            selections.append(1)
            if stage == "reference" or len(selections) == 2:
                report.update(residuals={"interpolated:process_scene": .04},
                    coverage_residuals={"interpolated:unknown_forbidden": .04},
                    scene_components=unknown, observed_geometry={"status": "completed"})
            return candidate, curve, report

        native.select_candidate = confirmed
    result, backend = run(native, scene=world)
    try:
        assert not result.success and result.trajectory is None and result.execution_binding is None
        assert result.diagnostics["failure"] == "needs_observation"
        assert native.calls == ([] if stage == "initial" else
            ["seed", "reference"] if stage == "reference" else ["seed", "reference", "solve"])
        assert result.request_ledger["consumed_planning"]["search_queries"] == 0
        assert result.request_ledger["consumed_planning"]["trajectory_solves"] == int(stage == "main")
    finally:
        backend.close()


def test_require_observation_does_not_relabel_a_mixed_collision_failure():
    native = Native([False, False], ompl=False)
    world = Scene()
    world.unknown_policy = {"outside_regions": "require_observation", "conditional_regions": []}
    select = native.select_candidate

    def mixed(accepted, *, record_work):
        candidate, curve, report = select(accepted, record_work=record_work)
        report.update(residuals={"interpolated:process_scene": .05},
            coverage_residuals={"interpolated:unknown_forbidden": .04},
            collision_residuals={"interpolated:scene_collision": .01},
            scene_components={"unknown/forbidden/robot": {"maximum": .04},
                              "collision/robot/background": {"maximum": .01}})
        return candidate, curve, report

    native.select_candidate = mixed
    result, backend = run(native, scene=world)
    try:
        assert not result.success and result.execution_binding is None
        assert result.diagnostics["failure"] == "no_feasible_solution_found"
        assert native.calls == ["seed", "reference", "solve", "ompl"]
    finally:
        backend.close()
# END external observation acquisition is not simulated or promised by this status.


# BEGIN a policy verdict needs complete named evidence, never residual subtraction.
@pytest.mark.parametrize("mode,expected", [
    ("forbid", "unknown_space_forbidden"),
    ("require_observation", "needs_observation"),
    ("allow_attempt", None),
])
def test_unknown_policy_failure_only_classifies_finite_coverage_evidence(mode, expected):
    from manipulation.planning import _unknown_policy_failure

    report = {"residuals": {"interpolated:process_scene": .04, "interpolated:skill_process": 0.},
        "scene_components": {"unknown/forbidden/robot": {"maximum": .04},
                             "conditional/unknown/payload": {"maximum": .02}}}
    assert _unknown_policy_failure(report, {"outside_regions": mode, "conditional_regions": []}) == expected


@pytest.mark.parametrize("fault", ["missing_components", "aggregate_only", "collision", "support", "contact",
    "unsupported", "task", "joint", "nan_component", "inf_component", "nan_metric", "inf_metric",
    "negative_inf_component", "zero_unknown"])
def test_unknown_policy_failure_cannot_erase_other_failures_or_missing_evidence(fault):
    from manipulation.planning import _unknown_policy_failure

    report = {"residuals": {"interpolated:process_scene": .04},
        "scene_components": {"unknown/forbidden/robot": {"maximum": .04}}}
    other_components = {"collision": "collision/robot/background", "support": "scene_update/robot/background",
        "contact": "contact/required/rule", "unsupported": "unsupported/contact/robot/owner"}
    if fault == "missing_components":
        report.pop("scene_components")
    elif fault == "aggregate_only":
        report["scene_components"] = {}
    elif fault in other_components:
        # A tiny independent violation remains decisive beside a larger coverage gap.
        report["scene_components"][other_components[fault]] = {"maximum": 1e-30}
    elif fault in {"task", "joint"}:
        report["residuals"]["interpolated:" + ("skill_process" if fault == "task" else "cspace")] = 1e-30
    elif fault in {"nan_metric", "inf_metric"}:
        report["residuals"]["interpolated:process_scene"] = float("nan" if fault == "nan_metric" else "inf")
    else:
        report["scene_components"]["unknown/forbidden/robot"]["maximum"] = {
            "nan_component": float("nan"), "inf_component": float("inf"),
            "negative_inf_component": float("-inf"), "zero_unknown": 0.}[fault]
    assert _unknown_policy_failure(report,
        {"outside_regions": "require_observation", "conditional_regions": []}) is None
# END a selected candidate's policy reason is not a proof about every possible route.


# BEGIN collision witnesses consume the original unique second solve.
# END second-solve rejection remains a finite planning failure.


# BEGIN real sampled-collision classification selects the existing recovery branch.
# END evidence classification neither adds a solve nor bypasses final acceptance.


def test_failed_ompl_cannot_spend_an_al_or_second_solve_slot():
    native = Native([False, False], collision=True, ompl=False)
    result, backend = run(native)
    assert not result.success and native.calls.count("solve") == 1
    assert "al" not in native.calls and "export" not in native.calls
    backend.close()


def test_two_requests_reuse_native_instances_and_reset_binding():
    native = Native([True, True, True, True])
    backend = MotionBackend({}, robot_validator=object())
    with patch("manipulation.planning._ConstrainedPlanner", return_value=native) as factory, \
         patch("manipulation.execution_binding.bind_candidate", return_value={"candidate_id": "accepted"}):
        first, second = backend.plan(problem(), Scene()), backend.plan(problem(), Scene())
    assert first.success and second.success and factory.call_count == 1
    assert native.calls.count("rebind") == 1
    assert first.request_ledger["request_id"] != second.request_ledger["request_id"]
    backend.close()


@pytest.mark.parametrize("kwargs,reason", [({"cancelled": lambda: True}, "cancelled"),
                                           ({"deadline": 0.}, "deadline_exceeded")])
def test_cancelled_or_expired_request_never_allocates_native(kwargs, reason):
    native = Native([])
    result, backend = run(native, **kwargs)
    assert not result.success and result.diagnostics["failure"] == reason
    assert backend._planner is None and native.calls == []
    backend.close()


def test_busy_backend_cannot_bind_close_or_release_another_request():
    backend = MotionBackend({}, robot_validator=object())
    backend._claim()
    try:
        result = backend.plan(problem(), Scene())
        assert result.diagnostics["failure"] == "backend_busy"
        assert backend._busy.locked()
        with pytest.raises(MotionPlanningError, match="still owns"):
            backend.close()
    finally:
        backend._release()
    backend.close()


def test_new_motion_allowance_requires_boundary_t_evidence_and_forward_progress():
    event = {"skill_id": "s", "event_id": "e", "occurrence": 0, "kind": "effect",
             "requires_evidence": "held", "progress": .5}
    old_block = {"requires_rebind": True, "boundary_event": event, "end_progress": .5}
    continuation = {"original_binding": {"curve": {"events": [event]}, "motion_block": old_block},
        "verified_events": [event], "elapsed_physical_s": 1.,
        "dependency_outcomes": [{"event": event, "evidence": {"value": "T", "source": "actual"},
                                 "feedback_id": "actual_1", "physical_elapsed_s": 1.}]}
    ledger = {"motion_request": {"index": 0, "planning_baseline": {"trajectory_solves": 0}}}
    block = {"start_progress": .5, "end_progress": 1.}
    assert _verified_dependency_outcomes(continuation) == {"held"}
    baseline = _bind_motion_work_ledger(ledger, block, continuation, {"trajectory_solves": 2})
    assert baseline["trajectory_solves"] == 2 and ledger["motion_request"]["index"] == 1
    for failure in ("F", "U"):
        continuation["dependency_outcomes"][0]["evidence"]["value"] = failure
        previous = deepcopy(ledger)
        assert not _verified_dependency_outcomes(continuation)
        _bind_motion_work_ledger(ledger, block, continuation, {"trajectory_solves": 4})
        assert ledger == previous
# END no native CUDA call or simulated physical completion is needed for these contracts.


# BEGIN support evidence cannot cancel the existing fixed-target finite search.
@pytest.mark.parametrize("outcomes,collision,expected_solves", [
    ([False, True], False, 1),
    ([False, False, True], True, 2),
])
def test_candidate_support_gaps_do_not_early_stop_main_or_recovery(outcomes, collision, expected_solves):
    native = Native(outcomes, collision=collision)
    select = native.select_candidate
    source = problem()
    source_positions = source.phases[0].reference.poses.copy()
    def with_support(accepted, *, record_work):
        candidate, curve, report = select(accepted, record_work=record_work)
        gap = 0. if accepted else 100000.
        report["scene_update_residuals"] = {"rollout:scene_update": gap}
        report["residuals"]["rollout:process_scene"] = gap
        return candidate, curve, report
    native.select_candidate = with_support
    backend = MotionBackend({}, robot_validator=object())
    with patch("manipulation.planning._ConstrainedPlanner", return_value=native), \
         patch("manipulation.execution_binding.bind_candidate", return_value={"candidate_id": "accepted"}):
        result = backend.plan(source, Scene())
    try:
        assert result.success, result.diagnostics
        assert native.calls.count("solve") == expected_solves
        assert result.diagnostics["reference_candidate"]["scene_update_residuals"] == {"rollout:scene_update": 100000.}
        assert result.diagnostics["scene_update_residuals"] == {"rollout:scene_update": 0.}
        if expected_solves == 2:
            assert "ompl" in native.calls
        np.testing.assert_array_equal(source.phases[0].reference.poses, source_positions)
    finally:
        backend.close()
# END only completed hard metrics determine whether a fixed transfer is achieved.


# BEGIN qualified incumbents own delivery independently of later solver buffers.
@pytest.mark.parametrize("improved_score,expected_end,exports", [(0.5, .2, 2), (2., .1, 1)])
def test_complete_candidates_compare_one_fixed_objective(improved_score, expected_end, exports):
    native = Native([(True, 1., .1), (True, improved_score, .2)])
    select = native.select_candidate

    def scored(outcome, *, record_work):
        accepted, score, end = outcome
        native.curve = JointTrajectory(("q",), np.array([0., 1.]), np.array([[0.], [end]]),
            np.zeros((2, 1)), np.array([0., 1.]), np.array([0., 1.]), (), (), {})
        result, curve, report = select(accepted, record_work=record_work)
        report.update(fixed_objective=score, objective_components={"continuous_reference": score})
        return result, curve, report

    native.select_candidate = scored
    result, backend = run(native)
    try:
        assert result.success
        assert result.trajectory.position[-1, 0] == expected_end
        assert result.diagnostics["returned_candidate"]["fixed_objective"] == min(1., improved_score)
        assert native.calls.count("solve") == 1 and native.calls.count("export") == exports
    finally:
        backend.close()


def test_improvement_budget_keeps_completed_record_after_mutable_buffers_change():
    from manipulation.planning import PlanningWorkExhausted
    native = Native([True])
    original = native.curve.position.copy()

    def exhausted(seed, *, iterations=None):
        native.calls.append("solve")
        native.curve.position[:] = .8
        native.last_validation["violations"][:] = 10.
        raise PlanningWorkExhausted("Test improvement exhausted its existing allowance")

    native.optimize = exhausted
    result, backend = run(native)
    try:
        assert result.success and result.execution_binding is not None
        np.testing.assert_array_equal(result.trajectory.position, original)
        assert result.diagnostics["returned_candidate"]["fixed_objective"] == 1.
        assert result.diagnostics["improvement_end"]["reason"] == "cumulative_planning_budget_exhausted"
        assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 1
        assert "failure" not in result.diagnostics
    finally:
        backend.close()


@pytest.mark.parametrize("error", [MotionPlanningError("cancelled", "cancelled during improvement"),
    MotionPlanningError("deadline_exceeded", "deadline during improvement"), RuntimeError("native error")])
def test_nonbudget_improvement_errors_retain_failure_semantics(error):
    native = Native([True])

    def fail(seed, *, iterations=None):
        raise error

    native.optimize = fail
    result, backend = run(native)
    try:
        assert not result.success and result.execution_binding is None and result.trajectory is None
        assert result.diagnostics["error_type"] == type(error).__name__
        assert "improvement_end" not in result.diagnostics
    finally:
        backend.close()
# END budget exhaustion cannot promote an unfinished or cancelled computation.
