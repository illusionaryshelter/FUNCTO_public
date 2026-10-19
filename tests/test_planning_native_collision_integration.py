"""Planner integration keeps native world residuals and other hard terms separate."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from manipulation.planning import _ProgressMap, _scene_boundary_evaluate
import test_manipulation_planning as _fixtures

WORLD = "collision/robot/background"
NONWORLD = "contact/required/material"


def test_clear_world_cannot_clear_a_contact_failure_on_another_boundary_side():
    def evaluate(spheres, p, r, s, *args, components_out, **kwargs):
        world = torch.where(s < .5, 10., 0.)
        nonworld = torch.where(s > .5, 7., 0.)
        components_out.update({WORLD: world, NONWORLD: nonworld})
        return world+nonworld
    scene = SimpleNamespace(required_progress_anchors=(.5,), contacts=(), evaluate=evaluate,
                            native_collision_components=frozenset({WORLD}))
    s = torch.tensor([[.5]])
    evidence = {}
    raw = _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
        _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
        torch.zeros((1, 1, 1, 4)), torch.zeros((1, 1, 3)), torch.tensor([[[1., 0., 0., 0.]]]),
        s, torch.zeros((1, 1, 1)), ('q',), trajectory_time=s, normalized_time=s,
        scene_evidence_out=evidence, optimizer_candidate_evidence=True)
    assert raw.item() == 10.
    assert evidence['optimizer_hard_invalid'].item() == 1.


@pytest.fixture
def native_fixture():
    """Keep cuRobo's cost collection while replacing only CUDA FK and kernels."""
    case = _fixtures.NativeCostManagerOwnershipTests()
    case.setUp()
    try:
        yield case.anchor_self_collision_fixture
    finally:
        case.doCleanups()


def attach_world_fixture(fixture, *, world_value=1., nonworld_value=.3):
    calls = []
    fixture.scene.native_collision_components = frozenset({WORLD})
    def evaluate(spheres, p, *args, components_out, **kwargs):
        calls.append(1)
        world = torch.ones_like(p[..., 0]) * world_value
        nonworld = torch.ones_like(world) * nonworld_value
        components_out.update({WORLD: world, NONWORLD: nonworld})
        return world + nonworld
    fixture.scene.evaluate = evaluate
    return calls


@pytest.mark.parametrize("required", [(), (0., 1.)])
def test_native_manager_sums_the_same_world_and_nonworld_query_once(native_fixture, required):
    with native_fixture("trajectory_optimizer", required=required, capacity=4) as fixture:
        calls = attach_world_fixture(fixture)
        costs = fixture.manager.compute_costs(fixture.state)
        scene = costs.values[costs.names.index("process_scene")]
        assert calls == [1]
        # Regular and actual exact anchors each contribute once. There is no
        # spatial-length multiplier or extra sampling of the world query.
        expected = 777. * (2+len(required))*1.3
        torch.testing.assert_close(scene.sum((1, 2)), torch.full((2,), expected))



@pytest.mark.parametrize("hard", [0., 1e-30])
def test_raw_world_violation_stays_infeasible_and_tiny_nonworld_stays_hard(native_fixture, hard):
    from curobo._src.rollout.metrics import CostsAndConstraints, RolloutResult
    with native_fixture("trajectory_optimizer", required=(0., 1.), capacity=4) as fixture:
        # The huge confirmable term cannot erase the tiny required contact
        # violation through subtracting rounded aggregate/world values.
        attach_world_fixture(fixture, world_value=1e20, nonworld_value=hard)
        costs = fixture.manager.compute_costs(fixture.state)
        result = RolloutResult(costs_and_constraints=CostsAndConstraints(constraints=costs))
        eligible, feasible, _, objective = fixture.manager.optimizer_evidence(result)
        assert eligible.tolist() == [hard == 0., hard == 0.]
        assert not feasible.any()
        assert torch.isfinite(objective).all() and (objective > 0).all()
# END the optimizer score never substitutes for original hard scene evidence.
