"""Verify the geometric world objective at the actual planner boundaries."""

# BEGIN same-query path ownership, mode ordering, gradients and hard acceptance.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.collision_path import SphereCollisionPath
from manipulation.planning import _ProgressMap, _scene_boundary_evaluate
import test_manipulation_planning as _fixtures


WORLD = "collision/robot/background"
NONWORLD = "contact/required/material"


def boundary(evaluate, progress, x, *, anchors=(), valid=None):
    """Use one analytic joint as sphere x without bypassing boundary expansion."""
    position = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), -1)
    spheres = torch.cat((position, torch.ones_like(x[..., None])), -1).unsqueeze(-2)
    quaternion = x.new_zeros((*x.shape, 4))
    quaternion[..., 0] = 1.
    scene = SimpleNamespace(required_progress_anchors=anchors, contacts=(), evaluate=evaluate,
                            native_world_collision_components=frozenset({WORLD}))
    objective, evidence = {}, {}
    raw = _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
        _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), spheres, position, quaternion,
        progress, x[..., None], ("joint_x",), trajectory_time=progress, normalized_time=progress,
        optimizer_scene_out=objective, scene_evidence_out=evidence,
        optimizer_candidate_evidence=True,
        sample_valid=torch.ones_like(progress, dtype=torch.bool) if valid is None else valid)
    return raw, objective, evidence


def test_nonworld_boundary_max_is_independent_of_world_winning_mode():
    def evaluate(spheres, p, r, s, *args, components_out, collision_paths_out, **kwargs):
        # The old material side has the largest complete sum. The required
        # contact penalty wins on the opposite side and must still cost seven.
        world = torch.where(s < .5, 10., 1.)
        nonworld = torch.where(s > .5, 7., 0.)
        components_out.update({WORLD: world, NONWORLD: nonworld})
        collision_paths_out[WORLD] = SphereCollisionPath(world[..., None],
            spheres[..., :3], s.new_ones(1), ((s >= .5)[..., None],))
        return world + nonworld

    raw, objective, evidence = boundary(evaluate, torch.tensor([[.5]]),
                                        torch.tensor([[0.]]), anchors=(.5,))
    torch.testing.assert_close(raw, torch.tensor([[10.]]))
    torch.testing.assert_close(objective["nonworld_values"], torch.tensor([[7.]]))
    torch.testing.assert_close(objective["world_objective"], torch.tensor([10.]))
    assert evidence["optimizer_hard_invalid"].item() == 1.


def test_exact_first_queries_reorder_left_exact_right_before_path_edges():
    queried = []

    def evaluate(spheres, p, r, s, *args, components_out, collision_paths_out, **kwargs):
        queried.append(s.detach().clone())
        mode = (s >= .5)[..., None]
        # A mode transition changes the target coordinate frame by 100 m.
        # That frame jump must create neither an arc nor an extra peak.
        centres = spheres[..., :3] + mode[..., None] * s.new_tensor([100., 0., 0.])
        world = torch.ones_like(s)
        components_out[WORLD] = world
        collision_paths_out[WORLD] = SphereCollisionPath(world[..., None], centres,
                                                         s.new_ones(1), (mode,))
        return world

    s = torch.tensor([[0., .5, 1.]])
    _, objective, _ = boundary(evaluate, s, torch.tensor([[0., 1., 2.]]), anchors=(.5,))
    assert len(queried) == 1
    torch.testing.assert_close(queried[0][:, :3], s)
    assert queried[0][0, 3] < .5 < queried[0][0, 4]
    # Peak 1 plus one metre before and one metre after the frame change.
    torch.testing.assert_close(objective["world_objective"], torch.tensor([3.]))


def test_invalid_anchor_padding_does_not_sever_real_path_or_propagate_nan():
    def evaluate(spheres, p, r, s, *args, components_out, collision_paths_out, **kwargs):
        world = p[..., 0] * 0. + 1.
        components_out[WORLD] = world
        collision_paths_out[WORLD] = SphereCollisionPath(world[..., None], spheres[..., :3],
                                                         s.new_ones(1))
        return world

    _, objective, _ = boundary(evaluate, torch.tensor([[0., .25, 1.]]),
        torch.tensor([[0., float("nan"), 2.]]), valid=torch.tensor([[True, False, True]]))
    torch.testing.assert_close(objective["world_objective"], torch.tensor([3.]))


def test_boundary_joint_gradient_matches_complete_objective_finite_difference():
    def evaluate(spheres, p, r, s, *args, components_out, collision_paths_out, **kwargs):
        # Nonlinear residuals exercise both the residual derivative and the
        # target-relative arc-length derivative through the same joint tensor.
        world = (p[..., 0] + .2).square()
        components_out[WORLD] = world
        collision_paths_out[WORLD] = SphereCollisionPath(world[..., None], spheres[..., :3],
            s.new_tensor([.4]), ((s >= .5)[..., None],))
        return world

    s = torch.tensor([[0., .5, 1.]], dtype=torch.float64)
    q = torch.tensor([[0., .8, 2.]], dtype=torch.float64, requires_grad=True)
    def objective(value):
        return boundary(evaluate, s, value, anchors=(.5,))[1]["world_objective"].sum()
    gradient = torch.autograd.grad(objective(q), q)[0]
    finite_difference = torch.empty_like(q)
    for index in range(q.numel()):
        delta = torch.zeros_like(q)
        delta.view(-1)[index] = 1e-6
        finite_difference.view(-1)[index] = (objective(q.detach()+delta)-objective(q.detach()-delta))/2e-6
    assert torch.isfinite(gradient).all()
    assert (gradient.abs() > 1e-5).all()
    torch.testing.assert_close(gradient, finite_difference, rtol=1e-7, atol=1e-7)


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
    fixture.scene.native_world_collision_components = frozenset({WORLD})
    def evaluate(spheres, p, *args, components_out, collision_paths_out=None, **kwargs):
        calls.append(1)
        world = torch.ones_like(p[..., 0]) * world_value
        nonworld = torch.ones_like(world) * nonworld_value
        components_out.update({WORLD: world, NONWORLD: nonworld})
        if collision_paths_out is not None:
            collision_paths_out[WORLD] = SphereCollisionPath(world[..., None],
                spheres[..., :1, :3], p.new_tensor([.2]))
        return world + nonworld
    fixture.scene.evaluate = evaluate
    return calls


@pytest.mark.parametrize("required", [(), (0., 1.)])
def test_native_manager_owns_world_path_once_and_keeps_nonworld_time_sum(native_fixture, required):
    with native_fixture("trajectory_optimizer", required=required, capacity=4) as fixture:
        calls = attach_world_fixture(fixture)
        costs = fixture.manager.compute_costs(fixture.state)
        scene = costs.values[costs.names.index("process_scene")]
        assert calls == [1]
        # Both batches travel 2 m with a fixed model radius of .2 m. Exact
        # endpoint anchors and inactive capacity slots cannot multiply J=11.
        expected = 777. * (11. + (2+len(required))*.3)
        torch.testing.assert_close(scene.sum((1, 2)), torch.full((2,), expected))
        gradient = torch.autograd.grad(scene.sum(), fixture.q)[0]
        expected_gradient = torch.zeros_like(fixture.q)
        expected_gradient[:, 0, 0] = -777./.2
        expected_gradient[:, 1, 0] = 777./.2
        torch.testing.assert_close(gradient, expected_gradient)


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
