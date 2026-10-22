"""CPU regressions for the initializer's constraint and request boundaries."""

# BEGIN scope changes retain original task rules and actual request consumption.
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints

from manipulation.planning import PlanningBudget
from manipulation.planning_problem import JointCoupling
from test_event_allowed_set_ik import event_problem, state_at
from test_motion_backend_contract import Native, run as run_native
from test_scene_query_performance import manager_fixture


def task_forward(fixture, position):
    """Keep the actual manager tensor identities in a native result container."""
    state = state_at(fixture, position)
    state.joint_state.dt = torch.ones(position.shape[0], dtype=position.dtype)
    constraints = fixture.manager.compute_costs(state)
    return SimpleNamespace(costs_and_constraints=CostsAndConstraints(
        costs=CostCollection(), constraints=constraints))


def test_restored_scene_scope_rechecks_same_curve_and_invalidates_task_only_evidence(manager_fixture):
    """A task-feasible curve can have a positive, independently queried scene term."""
    with manager_fixture("trajectory_optimizer", required=(.5,), capacity=4, task=event_problem()) as fixture:
        position = torch.tensor([[[.4, 0.], [.4, 0.], [.4, 0.]]])
        fixture.scene.evaluate = Mock(side_effect=AssertionError("Scene is outside the initializer scope"))
        fixture.configuration.set_scene_constraints(False)
        task_result = task_forward(fixture, position)
        task_feasible, task_signed, _ = fixture.manager.optimizer_evidence(task_result)
        assert task_feasible.tolist() == [True]
        assert "process_scene" not in task_result.costs_and_constraints.constraints.names
        fixture.scene.evaluate.assert_not_called()

        # A reported controller condition belongs to the complete scene query,
        # even when all functional position and orientation events are legal.
        def scene_condition(spheres, tool_position, quaternion, progress, joints, names,
                            *, components_out, **kwargs):
            violation = torch.full_like(progress, .125)
            components_out["controller/declared_condition"] = violation
            return violation

        fixture.scene.evaluate = Mock(side_effect=scene_condition)
        fixture.configuration.set_scene_constraints(True)
        with pytest.raises(RuntimeError, match="no same-query"):
            fixture.manager.optimizer_evidence(task_result)
        full_result = task_forward(fixture, position)
        full_feasible, full_signed, _ = fixture.manager.optimizer_evidence(full_result)
        assert full_feasible.tolist() == [False]
        assert "process_scene" in full_result.costs_and_constraints.constraints.names
        assert fixture.scene.evaluate.call_count == 1
        for key in task_signed:
            torch.testing.assert_close(full_signed[key], task_signed[key])
        with pytest.raises(RuntimeError, match="different native query"):
            fixture.manager.optimizer_evidence(task_result)


def test_task_only_scope_retains_process_coupling_beside_satisfied_event(manager_fixture):
    """A legal event does not erase a declared actuator relation on the path."""
    task = event_problem()
    coupling = JointCoupling(task.joint_names, np.array([[0., 1.]]), np.array([0.]),
                             "Declared analytic actuator relation")
    task = replace(task, couplings=(coupling,))
    with manager_fixture("trajectory_optimizer", required=(.5,), capacity=4, task=task) as fixture:
        fixture.configuration.set_scene_constraints(False)
        fixture.scene.evaluate = Mock(side_effect=AssertionError("Scene is outside the initializer scope"))
        # Both rows reach exactly the same legal event at t=1 s. Row zero
        # violates the independent second-joint relation only at t=0 and 2 s.
        position = torch.tensor([[[.4, .1], [.4, 0.], [.4, .1]],
                                 [[.4, 0.], [.4, 0.], [.4, 0.]]])
        result = task_forward(fixture, position)
        feasible, signed, _ = fixture.manager.optimizer_evidence(result)
        assert all(bool((value <= 0).all()) for value in signed.values())
        assert feasible.tolist() == [False, True]
        collection = result.costs_and_constraints.constraints
        process = collection.values[collection.names.index("skill_process")].flatten(1).amax(-1)
        assert process[0] > 0 and process[1] == 0
        fixture.scene.evaluate.assert_not_called()


def test_interrupted_initialization_consumes_first_solve_and_retains_completed_records():
    """The real public ledger accounts work before an initializer can return."""
    native = Native([])
    completed = {"stage": "coarse", "seeds": [{"seed_index": 0, "controls": [[0.]]}]}

    def interrupted_reference(joint_reference=None):
        native.calls.append("seed")
        native.reference_initialization = {"status": "interrupted", "coarse": completed,
            "scene_constraints": "not_evaluated"}
        native.initialization_candidate = native.curve
        raise TimeoutError("Declared initializer deadline")

    native.reference_seeds = interrupted_reference
    result, backend = run_native(native, budget=PlanningBudget(maximum_trajectory_solves=1))
    try:
        assert not result.success and result.execution_binding is None
        assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 1
        assert result.diagnostics["failed_stage"] == "reference_initialization"
        assert result.diagnostics["reference_initialization"]["coarse"] == completed
        assert result.diagnostics["diagnostic_trajectory"] is native.curve
        assert native.calls == ["seed"]
    finally:
        backend.close()
# END diagnostics describe completed computations without authorizing execution.
