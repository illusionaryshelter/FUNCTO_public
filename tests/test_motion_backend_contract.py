# BEGIN tests migrated to native motion attempts and retained numerical contracts.
"""CPU contracts for resident ownership and bounded motion-request decisions.

The native solver is replaced only at its constructor. These tests exercise the
real request ledger, selection orchestration and delivery boundary, not GPU
convergence or geometry accuracy.
"""
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
        # These orchestration tests never query geometry; model access would
        # be a test error rather than an execution-validator factory fallback.
        self.check_request = lambda: None
        self.robot_validator = SimpleNamespace()
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
        self.reference_ik_queries = []
        self.reference_seed_fits = []
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
    def optimize(self, seed):
        self.calls.append("solve")
        return next(self.outcomes)
    def select_candidate(self, accepted):
        report = {"accepted": accepted, "native_success": accepted, "seed_provenance": "test",
                  "residuals": {"process_scene" if self.collision else "cspace": 0. if accepted else 1.},
                  "collision_residuals": {"rollout:scene_collision": float(self.collision and not accepted)}}
        return SimpleNamespace(solution=torch.zeros((1, 1, 8, 1))), self.curve, report
    def ompl_seed(self, trajectory):
        self.calls.append("ompl")
        return (torch.zeros((1, 2, 8, 1)) if self.ompl_available else None), {"state_queries": 1}
    def export(self, result, native_report):
        assert native_report["accepted"]
        self.last_validation = {"u": np.array([0., 1.]), "violations": np.zeros(2)}
        self.calls.append("export")
        return self.curve, 0
    def rebind(self, problem, scene, budget, **kwargs):
        self.calls.append("rebind")
        return self
    def close(self):
        self.calls.append("close")


def run(native, *, budget=None, scene=None, **kwargs):
    backend = MotionBackend({}, robot_validator=object(), budget=budget)
    with patch("manipulation.planning._CuroboPlanner", return_value=native), \
         patch("manipulation.execution_binding.bind_candidate", return_value={"candidate_id": "accepted"}):
        result = backend.plan(problem(), Scene() if scene is None else scene, **kwargs)
    return result, backend


def test_nominal_reference_always_receives_one_native_optimization():
    native = Native([True])
    result, backend = run(native)
    assert result.success and native.calls == ["seed", "solve", "export"]
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 1
    assert result.execution_binding["verification_joint_limits"]["joint_names"] == ["q"]
    assert backend._planner is native
    backend.close()
    assert native.calls[-1] == "close"


def test_timeout_retains_actual_query_counts_without_delivering_a_partial_candidate():
    from manipulation.planning import MotionPlanningError
    native = Native([False])
    counts = {"mode": "trajectory_cost", "calls": 3, "states": 48}
    native._manager_type = SimpleNamespace(query_diagnostics=lambda: [dict(counts)])
    def stop(seed):
        native.calls.append("solve")
        raise MotionPlanningError("deadline_exceeded", "actual solve consumed request deadline")
    native.optimize = stop
    result, backend = run(native)
    try:
        assert not result.success and result.trajectory is None and result.execution_binding is None
        assert result.diagnostics["failure"] == "deadline_exceeded"
        assert result.diagnostics["rollout_query_counts"] == [counts]
        assert "export" not in native.calls
    finally:
        backend.close()


def test_real_candidate_binding_roundtrips_online_geometry_and_rejects_mutation():
    from manipulation.execution_binding import check_result
    from manipulation.online_geometry import FrozenESDF, OnlinePlanningScene
    from manipulation.planning_scene import ObservedSpace

    shape = (7, 7, 7)
    observed = np.ones(shape, dtype=bool)
    field = FrozenESDF(np.full(shape, 1e4, np.float32), origin=[-.3]*3, voxel_size=.1,
        owner_ids=("background",),
        source="fixed observed test field")
    coverage = ObservedSpace(observed, origin=[-.3]*3, voxel_size=.1,
        observation_ids=("frame",), reference_frame="world")
    scene = OnlinePlanningScene(background=field, owners=(), coverage=coverage,
        reference_frame="world", robot_sphere_materials=("body",),
        robot_geometry_source="native test spheres", has_gripper=False)
    # BEGIN explicit native fixture for the saved collision contract.
    from native_scene_fixtures import bind_fixed_robot
    bind_fixed_robot(scene)
    # END no robot geometry is inferred from query-time cover spheres.
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


def test_failed_main_collision_gets_one_bounded_recovery():
    native = Native([False, True], collision=True)
    result, backend = run(native)
    assert result.success
    assert native.calls.count("solve") == 2 and native.calls.count("export") == 1
    assert native.calls.count("ompl") == 1
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 2
    backend.close()


def test_noncollision_motion_failure_does_not_add_a_full_solve():
    native = Native([False, False])
    result, backend = run(native)
    assert not result.success and result.execution_binding is None
    assert native.calls.count("solve") == 1 and "export" not in native.calls
    assert result.diagnostics["recovery"] is None
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 1
    backend.close()


def test_recovery_failure_returns_no_execution_binding_or_export():
    native = Native([False, False], collision=True)
    result, backend = run(native)
    assert not result.success and result.execution_binding is None
    assert result.diagnostics["failure"] == "no_model_feasible_candidate_found"
    assert result.diagnostics["diagnostic_trajectory"] is native.curve
    assert native.calls.count("solve") == 2 and "export" not in native.calls
    backend.close()


def test_failed_connection_cannot_spend_a_second_solve():
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
# END native attempt-planner test migration.
