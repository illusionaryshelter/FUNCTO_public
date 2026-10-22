"""Exercise scene modes, hard evidence and ordinary scores at planner boundaries."""

# BEGIN current scene-query and ordinary-objective integration contract.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.planning import _ProgressMap, _scene_boundary_evaluate
import test_manipulation_planning as _fixtures


COLLISION = "collision/robot/background"
CONTACT = "contact/required/material"


def boundary(evaluate, progress, x, *, anchors=()):
    """Supply analytic FK while retaining production mode expansion and reduction."""
    position = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), -1)
    quaternion = x.new_zeros((*x.shape, 4))
    quaternion[..., 0] = 1.
    scene = SimpleNamespace(required_progress_anchors=anchors, contacts=(), evaluate=evaluate)
    evidence, diagnostics = {}, {}
    raw = _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
        _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
        x.new_zeros((*x.shape, 0, 4)), position, quaternion,
        progress, x[..., None], ("joint_x",), trajectory_time=2*progress,
        normalized_time=progress, scene_evidence_out=evidence,
        component_diagnostics=diagnostics, optimizer_candidate_evidence=True)
    return raw, evidence, diagnostics


def test_boundary_keeps_collision_and_contact_witnesses_from_different_modes():
    def evaluate(spheres, p, r, s, *args, components_out, **kwargs):
        # The left mode has the greatest aggregate. Contact is violated only
        # on the right, so its diagnostic must retain that separate witness.
        collision = torch.where(s < .5, 10., 1.)
        contact = torch.where(s > .5, 7., 0.)
        components_out.update({COLLISION: collision, CONTACT: contact})
        return collision + contact

    raw, evidence, diagnostics = boundary(evaluate, torch.tensor([[.5]]),
                                         torch.tensor([[0.]]), anchors=(.5,))
    torch.testing.assert_close(raw, torch.tensor([[10.]]))
    assert evidence["collision"].item() == 10.
    assert evidence["optimizer_hard_invalid"].item() == 1.
    assert diagnostics[CONTACT]["maximum"] == 7.
    assert diagnostics[CONTACT]["progress"] > .5
    assert diagnostics[CONTACT]["time_s"] == 1.


def test_exact_samples_precede_side_queries_with_same_physical_state_and_time():
    queried = []

    def evaluate(spheres, p, r, s, q, names, *, components_out, trajectory_time, **kwargs):
        queried.append((s.detach().clone(), p.detach().clone(), q.detach().clone(),
                        trajectory_time.detach().clone()))
        collision = torch.ones_like(s)
        components_out[COLLISION] = collision
        return collision

    s = torch.tensor([[0., .5, 1.]])
    x = torch.tensor([[0., 1., 2.]])
    boundary(evaluate, s, x, anchors=(.5,))
    assert len(queried) == 1
    progress, position, joint, time = queried[0]
    torch.testing.assert_close(progress[:, :3], s)
    assert progress[0, 3] < .5 < progress[0, 4]
    torch.testing.assert_close(position[0, 3:], position[0, 1:2].expand(2, 3))
    torch.testing.assert_close(joint[0, 3:], joint[0, 1:2].expand(2, 1))
    torch.testing.assert_close(time, torch.tensor([[0., 1., 2., 1., 1.]]))


def test_boundary_joint_gradient_matches_original_scene_finite_difference():
    def evaluate(spheres, p, r, s, *args, components_out, **kwargs):
        # The side changes the residual function, while the repeated FK
        # sample remains connected to its original joint tensor.
        collision = (p[..., 0] + .2).square() * (1. + .5*(s > .5))
        contact = .05*(p[..., 0] - .1).square()
        components_out.update({COLLISION: collision, CONTACT: contact})
        return collision + contact

    s = torch.tensor([[0., .5, 1.]], dtype=torch.float64)
    q = torch.tensor([[0., .8, 2.]], dtype=torch.float64, requires_grad=True)
    def objective(value):
        return boundary(evaluate, s, value, anchors=(.5,))[0].sum()
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
    """Keep the native cost manager while replacing CUDA FK and pair kernels."""
    case = _fixtures.NativeCostManagerOwnershipTests()
    case.setUp()
    try:
        yield case.anchor_self_collision_fixture
    finally:
        case.doCleanups()


@pytest.mark.parametrize("required", [(), (0., 1.)])
def test_native_manager_folds_actual_scene_anchors_once_with_their_gradients(native_fixture, required):
    with native_fixture("trajectory_optimizer", required=required, capacity=4) as fixture:
        calls = []
        def evaluate(spheres, p, *args, components_out, **kwargs):
            calls.append(p.shape[1])
            collision = (p[..., 0]+.2).square()
            contact = p[..., 1]+.3
            components_out.update({COLLISION: collision, CONTACT: contact})
            return collision + contact
        fixture.scene.evaluate = evaluate
        costs = fixture.manager.compute_costs(fixture.state)
        scene = costs.values[costs.names.index("process_scene")]
        assert calls == [2+len(required)]
        # Endpoint anchors repeat their physical endpoint exactly once; the
        # unused capacity contributes neither residuals nor derivatives.
        multiplicity = 1+len(required)//2
        expected = 777.*multiplicity*((fixture.q[..., 0]+.2).square()+.3).sum(-1)
        torch.testing.assert_close(scene.sum((1, 2)), expected)
        gradient = torch.autograd.grad(scene.sum(), fixture.q)[0]
        expected_gradient = 777.*multiplicity*torch.stack(
            (2*(fixture.q[..., 0]+.2), torch.ones_like(fixture.q[..., 1])), -1)
        torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("collision,contact,expected_feasible", [
    (0., 0., True), (1e20, 0., False), (0., 1e-30, False), (1e20, 1e-30, False)])
def test_original_collision_and_contact_components_each_determine_feasibility(
        native_fixture, collision, contact, expected_feasible):
    from curobo._src.rollout.metrics import CostsAndConstraints, RolloutResult
    with native_fixture("trajectory_optimizer", required=(0., 1.), capacity=4) as fixture:
        def evaluate(spheres, p, *args, components_out, **kwargs):
            components_out[COLLISION] = torch.full_like(p[..., 0], collision)
            components_out[CONTACT] = torch.full_like(p[..., 0], contact)
            return sum(components_out.values())
        fixture.scene.evaluate = evaluate
        constraints = fixture.manager.compute_costs(fixture.state)
        result = RolloutResult(costs_and_constraints=CostsAndConstraints(constraints=constraints))
        feasible, signed, objective = fixture.manager.optimizer_evidence(result)
        assert feasible.tolist() == [expected_feasible, expected_feasible]
        assert signed == {}
        torch.testing.assert_close(objective, torch.zeros(2))


def test_fixed_score_uses_ordinary_costs_while_scene_preferences_and_hybrid_keep_their_roles(native_fixture):
    from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutResult
    with native_fixture("trajectory_optimizer", required=(0., 1.), capacity=4) as fixture:
        ordinary = CostCollection()
        ordinary.add(torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]),
                     "reference_pose")
        ordinary.add(torch.tensor([[[.25], [.75]], [[1.], [2.]]]), "joint_regularization")
        for scale in (1., 10000.):
            fixture.problem.optimizer_preferences = lambda p, *args, **kwargs: torch.full_like(p[..., 0], scale)
            fixture.scene.optimizer_preferences = lambda progress, *args: torch.full_like(progress, 2*scale)
            def evaluate(spheres, p, *args, components_out, **kwargs):
                # This models a positive differentiable local objective in
                # a state whose original entity query has zero hard residual.
                components_out[COLLISION] = torch.zeros_like(p[..., 0])
                return torch.full_like(p[..., 0], 3*scale)
            fixture.scene.evaluate = evaluate
            constraints = fixture.manager.compute_costs(fixture.state)
            hybrid = CostCollection()
            hybrid.add(torch.full((2, 2, 1), -.5), "joint_limit")
            result = RolloutResult(costs_and_constraints=CostsAndConstraints(
                costs=ordinary, constraints=constraints, hybrid_costs_constraints=hybrid))
            feasible, _, score = fixture.manager.optimizer_evidence(result)
            assert feasible.tolist() == [True, True]
            torch.testing.assert_close(score, torch.tensor([11., 29.]))
            hybrid.values[0][0, 0] = 1e-30
            feasible, _, score = fixture.manager.optimizer_evidence(result)
            assert feasible.tolist() == [False, True]
            torch.testing.assert_close(score, torch.tensor([11., 29.]))
# END candidate scores and original hard scene evidence have distinct consumers.
