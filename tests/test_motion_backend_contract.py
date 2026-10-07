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
        # BEGIN satisfy the shared native-confirmation lifecycle in CPU fixtures.
        # These orchestration tests never query geometry. A bound adapter with
        # an empty report lets metric-buffer/cleanup tests reach their sentries.
        self.check_request = lambda: None
        self.robot_validator = SimpleNamespace(online_collision_confirmation=
            lambda scene, **kwargs: SimpleNamespace(last_report={}))
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
                  "residuals": {"process_scene" if self.collision else "skill_process": 0. if accepted else 1.}}
        return SimpleNamespace(solution=torch.zeros((1, 1, 8, 1))), self.curve, report
    def ompl_seed(self, seed, trajectory):
        self.calls.append("ompl")
        return (seed if self.ompl_available else None), {"state_queries": 1}
    # BEGIN the default fixture has no uncovered dense collision samples.
    def promote_collision_witnesses(self, result):
        return {"changed": False, "parameters": []}
    # END dedicated recovery tests supply new physical-time evidence below.
    def update_functional_constraints(self, result):
        self.calls.append("al")
        return {"changed": True}
    def reset_functional_constraints(self):
        self.calls.append("reset_al")
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
    field = FrozenESDF(np.ones(shape, dtype=np.float32), origin=[-.3]*3, voxel_size=.1,
        site_valid=observed, observed_sign=observed, owner_ids=("background",),
        source="fixed observed test field")
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


@pytest.mark.parametrize("collision,recovery", [(True, "ompl"), (False, "al")])
def test_failed_main_gets_exactly_one_exclusive_recovery(collision, recovery):
    native = Native([False, False, True], collision=collision)
    result, backend = run(native)
    assert result.success
    assert native.calls.count("solve") == 2 and native.calls.count("export") == 1
    assert native.calls.count(recovery) == 1
    assert ("al" if collision else "ompl") not in native.calls
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 2
    backend.close()


def test_recovery_failure_returns_no_execution_binding_or_export():
    native = Native([False, False, False])
    result, backend = run(native)
    assert not result.success and result.execution_binding is None
    assert result.diagnostics["failure"] == "no_feasible_solution_found"
    assert result.diagnostics["diagnostic_trajectory"] is native.curve
    assert native.calls.count("solve") == 2 and "export" not in native.calls
    backend.close()


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
    assert native.calls.count("al") == 1 and "reset_al" not in native.calls and "ompl" not in native.calls
    torch.testing.assert_close(recovery_seeds[1], optimized_controls.expand_as(recovery_seeds[1]))
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 2
    assert result.request_ledger["planning_budget"]["planning_seconds"] == 120.
    if not accepted:
        assert result.execution_binding is None and "export" not in native.calls
    backend.close()
# END second-solve rejection remains a finite planning failure.


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
