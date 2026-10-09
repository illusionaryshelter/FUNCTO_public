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
    _ProgressMap, _bind_motion_work_ledger, _verified_dependency_outcomes,
)
from manipulation.planning_problem import MotionPhase, PlanningProblem, PosePath


def problem():
    path = PosePath(np.array([0., 1.]), np.stack([np.eye(4), np.eye(4)]), "reference")
    return PlanningProblem(("q",), [0.], [0.], "tool", "world", np.eye(4), np.eye(4),
        (MotionPhase("phase", path),), (), source_timestamps=np.array([0., 1.]),
        source_progress=np.array([0., 1.]))


class Scene:
    gripper = None
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
        self.trajopt = SimpleNamespace(core=core())
        self.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        self.scene_workspace_bytes = self.scene_cpu_workspace_bytes = self.scene_gpu_workspace_bytes = 0
        self.anchor_fk_workspace_bytes = 0
        self.anchor_capacity = 256
        self.validation_storage = {}
        self.route = "constrained"
        self.functional_constraints = SimpleNamespace(keys=("event",))
        self.reference_ik_distinct_counts = self.reference_ik_queries = self.reference_ik_chains = []
        self.reference_seed_fits = self.goal_ik_fit_hints = []
        self.joint_reference_report = None
        self.initial_seed_strategies = self.seed_strategies = ["test", "test"]
        self.last_validation = {"u": np.array([0., 1.]), "violations": np.zeros(2)}
        limits = SimpleNamespace(joint_names=["q"], position=torch.tensor([[-1.], [1.]]),
            velocity=torch.tensor([[-1.], [1.]]), acceleration=torch.tensor([[-1.], [1.]]))
        self._verification_native = SimpleNamespace(get_cost=lambda _: SimpleNamespace(config=SimpleNamespace(joint_limits=limits)))
        self.curve = JointTrajectory(("q",), np.array([0., 1.]), np.zeros((2, 1)), np.zeros((2, 1)),
                                    np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    def initial_state_diagnostics(self):
        return {"passed": True}
    def reference_seeds(self, joint_reference=None):
        self.calls.append("seed")
        return torch.zeros((1, 2, 8, 1)), []
    def evaluate_reference(self, seed):
        self.calls.append("reference")
        return next(self.outcomes)
    def optimize(self, seed):
        self.calls.append("solve")
        return next(self.outcomes)
    def select_candidate(self, accepted):
        report = {"accepted": accepted, "native_success": accepted, "seed_provenance": "test",
                  "residuals": {"process_scene" if self.collision else "skill_process": 0. if accepted else 1.},
                  # BEGIN recovery evidence is distinct from the hard scene aggregate.
                  "collision_residuals": {"rollout:scene_collision": float(self.collision and not accepted)},
                  "scene_update_residuals": {"rollout:scene_update": 0.}}
                  # END the deterministic fixture supplies explicit query classification.
        return SimpleNamespace(solution=torch.zeros((1, 1, 8, 1))), self.curve, report
    def ompl_seed(self, seed, trajectory):
        self.calls.append("ompl")
        return (seed if self.ompl_available else None), {"state_queries": 1}
    # BEGIN the default fixture has no uncovered dense collision samples.
    def promote_collision_witnesses(self, result):
        return {"changed": False, "parameters": []}
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


def run(native, **kwargs):
    backend = MotionBackend({}, robot_validator=object())
    with patch("manipulation.planning._CuroboPlanner", return_value=native), \
         patch("manipulation.execution_binding.bind_candidate", return_value={"candidate_id": "accepted"}):
        result = backend.plan(problem(), Scene(), **kwargs)
    return result, backend


def test_feasible_reference_delivers_once_without_optimization():
    native = Native([True])
    result, backend = run(native)
    assert result.success and native.calls == ["seed", "reference", "export"]
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 0
    assert result.execution_binding["verification_joint_limits"]["joint_names"] == ["q"]
    assert backend._planner is native
    backend.close()
    assert native.calls[-1] == "close"


# BEGIN exercise the real serialization handoff after native candidate selection.
def test_real_candidate_binding_roundtrips_online_geometry_and_rejects_mutation():
    from manipulation.execution_binding import check_result
    from manipulation.online_geometry import FrozenESDF, OnlinePlanningScene
    from manipulation.planning_scene import ObservedSpace

    shape = (7, 7, 7)
    observed = np.ones(shape, dtype=bool)
    # BEGIN serialization retains an explicit empty obstacle source.
    from observed_geometry_fixtures import point_source
    field = FrozenESDF.from_source(point_source(), origin=[-.3]*3, shape=shape, voxel_size=.1,
        owner_ids=("background",),
        source="fixed observed test field")
    # END observed-free evidence is independently serialized by coverage.
    coverage = ObservedSpace(observed, origin=[-.3]*3, voxel_size=.1,
        observation_ids=("frame",), reference_frame="world")
    scene = OnlinePlanningScene(background=field, owners=(), coverage=coverage,
        reference_frame="world", robot_sphere_materials=("body",),
        robot_geometry_source="native test spheres", has_gripper=False,
        unknown_policy={"outside_regions": "forbid", "conditional_regions": []})
    scene.bind_online_robot(robot_sphere_geometry=[{"link_name": "tool", "sphere_indices": [0],
        "centers_local": [[0., 0., 0.]], "radii": [.01]}], robot_self_collision_pairs=[])
    native = Native([True])
    backend = MotionBackend({}, robot_validator=object())
    try:
        with patch("manipulation.planning._CuroboPlanner", return_value=native):
            result = backend.plan(problem(), scene)
        assert result.success and native.calls.count("export") == 1
        binding = check_result(result)
        restored = OnlinePlanningScene.from_binding(binding["scene"])
        spheres = np.array([[0., 0., 0., .01]])
        assert restored.query_state(spheres, np.eye(4), .5) == scene.query_state(spheres, np.eye(4), .5)
        result.execution_binding["scene"]["robot_sphere_geometry"][0]["radii"][0] = .02
        with pytest.raises(ValueError, match="validated input binding"):
            check_result(result)
    finally:
        backend.close()
# END accepted evidence names exactly the field record handed to execution.


# BEGIN only the existing collision branch may request a second full solve.
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
@pytest.mark.parametrize("functional", [False, True])
def test_positive_scene_support_only_preserves_hard_failure_and_skips_collision_recovery(functional):
    native = Native([False, False, False])
    native.functional_constraints.keys = ("event",) if functional else ()
    select = native.select_candidate
    def missing_support(accepted):
        candidate, curve, report = select(accepted)
        report["residuals"] = {"rollout:process_scene": 400000., "interpolated:process_scene": 500000.}
        report["collision_residuals"] = {"rollout:scene_collision": 0., "interpolated:scene_collision": 0.}
        report["scene_update_residuals"] = {"rollout:scene_update": 400000., "interpolated:scene_update": 500000.}
        return candidate, curve, report
    native.select_candidate = missing_support
    native.promote_collision_witnesses = lambda _: pytest.fail("Support gap is not collision evidence")
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


# BEGIN collision witnesses consume the original unique second solve.
@pytest.mark.parametrize("accepted", [False, True])
def test_new_dense_witnesses_continue_current_controls_without_search_or_third_solve(accepted):
    native = Native([False, False, accepted], collision=True)
    original_select = native.select_candidate
    optimized_controls = torch.full((1, 1, 8, 1), .37)
    recovery_seeds = []

    def select(outcome):
        selected, curve, report = original_select(outcome)
        selected.solution = optimized_controls.clone()
        return selected, curve, report

    def promote(result):
        native.calls.append("promote")
        torch.testing.assert_close(result.solution, optimized_controls)
        return {"changed": True, "parameters": [.37], "added_count": 1}

    original_optimize = native.optimize
    def optimize(seed):
        recovery_seeds.append(seed.clone())
        return original_optimize(seed)

    native.select_candidate, native.promote_collision_witnesses = select, promote
    native.optimize = optimize
    result, backend = run(native)
    assert result.success is accepted
    assert result.diagnostics["recovery"] == "collision_witnesses"
    assert native.calls.count("solve") == 2 and native.calls.count("promote") == 1
    assert "ompl" not in native.calls
    torch.testing.assert_close(recovery_seeds[1], optimized_controls.expand_as(recovery_seeds[1]))
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 2
    assert result.request_ledger["planning_budget"]["planning_seconds"] == 120.
    if not accepted:
        assert result.execution_binding is None and "export" not in native.calls
    backend.close()
# END second-solve rejection remains a finite planning failure.


# BEGIN real sampled-collision classification selects the existing recovery branch.
@pytest.mark.parametrize("already_detected", [True, False])
def test_sampled_collision_evidence_selects_search_or_missing_time_recovery(already_detected):
    from manipulation.planning import _dense_collision_witnesses

    native = Native([False, False, True], collision=True)
    original_select = native.select_candidate

    def metric(values):
        collision = torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1)
        return SimpleNamespace(debug=torch.cat((collision, torch.zeros_like(collision)), dim=-1),
            costs_and_constraints=SimpleNamespace(constraints=SimpleNamespace(
                names=["process_scene"], values=[collision])))

    def select(accepted):
        candidate, curve, report = original_select(accepted)
        # Constant positive evidence has no sampling gap. The alternative
        # has one collision strictly between ordinary zero-valued samples.
        candidate.batch_size = candidate.num_seeds = 1
        candidate.metrics = metric([1., 1., 1.] if already_detected else [0., 0., 0.])
        candidate.interpolated_metrics = metric([1.]*9 if already_detected else
                                                [0., 1., 0., 0., 0., 0., 0., 0., 0.])
        candidate.interpolated_last_tstep = torch.tensor([[9]])
        candidate.interpolated_trajectory = SimpleNamespace(position=torch.zeros(1, 9, 1))
        if not accepted:
            report["collision_residuals"] = {
                "rollout:scene_collision": float(already_detected), "interpolated:scene_collision": 1.}
        return candidate, curve, report

    def promote(candidate):
        report = _dense_collision_witnesses(candidate, regular_horizon=3,
            anchor_parameters=(0., 1.), capacity=4)
        return {**report, "changed": bool(report["parameters"])}

    native.select_candidate, native.promote_collision_witnesses = select, promote
    result, backend = run(native)
    try:
        assert result.success
        assert result.diagnostics["recovery"] == ("ompl" if already_detected else "collision_witnesses")
        assert native.calls.count("ompl") == int(already_detected)
        assert native.calls.count("solve") == 2
        assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 2
    finally:
        backend.close()
# END evidence classification neither adds a solve nor bypasses final acceptance.


def test_failed_ompl_cannot_spend_an_al_or_second_solve_slot():
    native = Native([False, False], collision=True, ompl=False)
    result, backend = run(native)
    assert not result.success and native.calls.count("solve") == 1
    assert "al" not in native.calls and "export" not in native.calls
    backend.close()


def test_two_requests_reuse_native_instances_and_reset_binding():
    native = Native([True, True])
    backend = MotionBackend({}, robot_validator=object())
    with patch("manipulation.planning._CuroboPlanner", return_value=native) as factory, \
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
    def with_support(accepted):
        candidate, curve, report = select(accepted)
        gap = 0. if accepted else 100000.
        report["scene_update_residuals"] = {"rollout:scene_update": gap}
        report["residuals"]["rollout:process_scene"] = gap
        return candidate, curve, report
    native.select_candidate = with_support
    backend = MotionBackend({}, robot_validator=object())
    with patch("manipulation.planning._CuroboPlanner", return_value=native), \
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
