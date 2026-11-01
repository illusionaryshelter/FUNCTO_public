"""CPU checks of the reference term at its real shared-objective boundary.

Real BoundToolReference and PlanningProblem objects supply reference and
event semantics. Explicit tensor states isolate the reference from geometry;
the scene declares an empty relationship set and its collision cost is unused.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                 str(Path(__file__).resolve().parents[2] / 'tests'),
                 str(Path(__file__).resolve().parents[2] / 'tests/planning')]

from manipulation.planning_problem import (
    MotionPhase, PlanningProblem, PoseEventRequirement, PosePath,
    quaternion_multiply,
)
from manipulation.preference_planner.objective import (
    trajectory_objective_components as original_objective_components,
)
from manipulation.preference_planner.pose_roles import bind_pose_objectives
from manipulation.timed_reference import BoundToolReference
from manipulation.unified_solver import time_mean_weights
from manipulation.direct_planner.reference_objective import ReferenceObjective, trajectory_objective_components


def pose(position=(0., 0., 0.), rotation=(0., 0., 0.)):
    """Make a declared pose with metre translation and rotation-vector input."""
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    return result


def moving_reference():
    """A composed source, a hold and a measured nonzero entrance twist."""
    path = PosePath([0., .4, 1.], np.stack((pose(), pose((.1, .2, .1), (.1, .2, 0.)),
        pose((.3, -.1, .2), (0., .2, .6)))), source='declared source trajectory',
        attachment=pose((.08, -.03, .02), (.1, -.2, .3)))
    path.parent = PosePath([0., 1.], np.stack((pose((.1, .1, .2)),
        pose((.2, .0, .1), (0., .3, 0.)))), source='declared parent motion')
    initial = pose((-.2, .1, .3), (.2, -.1, -.3))
    twist = np.array([.1, -.05, .02, .1, -.1, .2])
    phase = MotionPhase('analytical interval', path, {'reference_boundaries': {
        'source': 'analytical boundary declaration', 'initial_pose': path.sample_numpy(0.),
        'goals': [dict(progress=.8, pose=path.sample_numpy(.8), source='original join')]}})
    problem = SimpleNamespace(phases=(phase,))
    reference = BoundToolReference(problem, [0., .3, .6, .8, 1.], [0., .4, .4, .8, 1.],
        2., initial, twist)
    return problem, reference


def constant_problem():
    """Retain a material lever arm and one original event prior."""
    path = PosePath([0., 1.], np.stack((np.eye(4), np.eye(4))), source='constant TCP')
    material = pose((.05, -.02, .03), (.1, .2, -.1))
    event = PoseEventRequirement(.5, material, material, .002, .01,
        source='declared material event')
    problem = PlanningProblem(('joint',), [0.], [0.], 'tool', 'reference',
        np.eye(4), np.eye(4), (MotionPhase('analytical interval', path),), pose_events=(event,))
    return bind_pose_objectives(problem, [dict(goal_id=['pose_events', 0], role='prior',
        scales=dict(position=.02, rotation=.1), scale_source='explicit analytical scales')])


# BEGIN declared empty relationships distinguish the removed collision objective.
class EmptyRelationshipScene:
    """Declare an empty relation set and reject any collision-soft-cost call."""

    contacts = ()
    _initial_support_models = {}
    gripper = None

    def optimization_objective(self, position, quaternion, progress, q, names, *,
                               trajectory_time, robot_link_poses, report_out=None):
        raise AssertionError('Nonpenetration soft objective was requested')
# END the real relation values and gradients have a separate native fixture.


def objective_arguments(problem, position, quaternion, times, progress, ordinary):
    """Build the existing one-candidate state layout without a second FK."""
    q = position.new_zeros((*position.shape[:2], 1))
    states = (q, q, q, q)
    return dict(problem=problem, scene=EmptyRelationshipScene(), states=states,
        robot_state=(q, q, ('joint',), {}, (position, quaternion),
                     (position * 0., position * 0.)),
        times=times[None], progress=progress[None], ordinary_indices=ordinary,
        bounds=position.new_tensor([[[-100.], [100.]]] * 4),
        motion_configuration=dict(weight=[0.] * 4, squared_l2_regularization_weight=[0.] * 3,
            retime_weights=False, retime_regularization_weights=False, activation_distance=[0.] * 4),
        native_dt=.1, constraint_weight=100000.)


def test_real_reference_samples_and_clock_are_frozen_once(monkeypatch):
    _, reference = moving_reference()
    times = torch.tensor([0., .17, .6, .95, 1.6, 2.], dtype=torch.float64)
    expected = reference.sample(times)
    calls = []
    sample = reference.sample

    def recorded_sample(actual_times):
        calls.append(actual_times.detach().clone())
        return sample(actual_times)

    monkeypatch.setattr(reference, 'sample', recorded_sample)
    declaration = ReferenceObjective(.02, .1, .25, 0., 1.)
    bound = declaration.bind(reference, times)
    torch.testing.assert_close(bound.position, expected[0], atol=0., rtol=0.)
    torch.testing.assert_close(bound.quaternion, expected[1], atol=0., rtol=0.)
    np.testing.assert_array_equal(bound.weights.numpy(), time_mean_weights(times.numpy()))
    assert not bound.position.requires_grad and not bound.quaternion.requires_grad
    for _ in range(3):
        assert float(bound.cost(*expected)) < 1.e-24
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], times, atol=0., rtol=0.)
    times[1] = .18
    assert float(bound.physical_times[1]) == .17


# BEGIN shared scoring preserves original terms and explicit relation ownership.
def test_shared_objective_weights_once_and_keeps_material_event():
    problem = constant_problem()
    times = torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)
    ordinary = np.array([0, 2, 4])
    reference = BoundToolReference(problem, [0., 1.], [0., 1.], 1., np.eye(4), np.zeros(6))
    declaration = ReferenceObjective(.02, .1, .3, 0., 1.)
    bound = declaration.bind(reference, times[ordinary])
    position = torch.tensor([.01, .03, 0.], dtype=torch.float64).expand(1, len(times), 3).clone().requires_grad_()
    angle = .2
    quaternion = torch.tensor([np.cos(angle / 2), 0., 0., np.sin(angle / 2)],
                              dtype=torch.float64).expand(1, len(times), 4)
    arguments = objective_arguments(problem, position, quaternion, times, times, ordinary)
    original_material = problem.pose_events[0].tool_from_function.copy()
    base = original_objective_components(**arguments, include_geometry=False)
    produced, relation_report = [], {}

    def producer(name, function):
        produced.append(name)
        return function()

    components = trajectory_objective_components(reference_objective=bound, **arguments,
        material_samples={}, producer=producer, scene_relation_report=relation_report)
    for name, value in base.items():
        torch.testing.assert_close(components[name], value, atol=0., rtol=0.)
    expected_reference = 100000. * .3 * ((.01**2 + .03**2) / .02**2
                                        + 4*np.sin(angle/2)**2 / .1**2)
    assert float(components['reference_prior'].detach()) == pytest.approx(expected_reference)
    # The event's point moves with its original material lever arm. An event
    # expressed at the bare TCP would produce a different position error.
    arm = original_material[:3, 3]
    material_error = np.array([.01, .03, 0.]) + Rotation.from_rotvec([0., 0., angle]).apply(arm) - arm
    expected_event = 100000. * ((material_error @ material_error) / .02**2 + angle**2 / .1**2)
    assert float(components['pose_prior'].detach()) == pytest.approx(expected_event)
    np.testing.assert_array_equal(problem.pose_events[0].tool_from_function, original_material)
    assert problem.pose_events[0].source['pose_objective']['role'] == 'prior'
    gradient, = torch.autograd.grad(components['reference_prior'], position)
    assert float(gradient[..., 0].sum()) == pytest.approx(2 * 100000. * .3 * .01 / .02**2)
    with torch.no_grad():
        saved_components = trajectory_objective_components(reference_objective=bound, **arguments,
            material_samples={})
    assert {key: float(value) for key, value in saved_components.items()} == {
        key: float(value.detach()) for key, value in components.items()}
    assert 'observed_geometry' not in components and float(components['scene_relations'].detach()) == 0.
    assert produced == ['objective_base', 'scene_relations', 'reference_prior', 'tool_angular_motion']
    assert relation_report['required_contact_count'] == relation_report['initial_support_count'] == 0
    report = bound.report(100000.)
    assert report['relative_weight'] == .3 and report['effective_absolute_weight'] == 30000.
# END candidate and solver values retain the same one-time weights.


def test_extra_event_query_changes_event_once_without_reference_integration():
    problem = constant_problem()
    times = torch.tensor([0., .25, .5, 1.], dtype=torch.float64)
    progress = torch.tensor([0., .5, .5, 1.], dtype=torch.float64)
    ordinary = np.array([0, 2, 3])
    reference = BoundToolReference(problem, times, progress, 1., np.eye(4), np.zeros(6))
    bound = ReferenceObjective(.02, .1, .25, 0., 1.).bind(reference, times[ordinary])
    position = torch.tensor([.01, 0., 0.], dtype=torch.float64).expand(1, len(times), 3).clone()
    position[0, 1, 0] = .2
    quaternion = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(1, len(times), 4)
    components = trajectory_objective_components(reference_objective=bound,
        material_samples={},
        **objective_arguments(problem, position, quaternion, times, progress, ordinary))
    assert float(components['reference_prior']) == pytest.approx(100000. * .25 * .01**2 / .02**2)
    assert float(components['pose_prior']) == pytest.approx(100000. * .2**2 / .02**2)


def test_candidate_clock_rebinds_source_and_preserves_entry_and_composition():
    problem, source = moving_reference()
    declaration = ReferenceObjective(.02, .1, .4, 0., 1.)
    times = torch.tensor([0., .5, 1., 2.5, 4.], dtype=torch.float64)
    u, s = np.array([0., .2, .5, .85, 1.]), np.array([0., .4, .4, .8, 1.])
    saved_source = source.sample_numpy(np.linspace(0., 2., 9))
    rebound = declaration.bind_clock(source, problem=problem, progress_u=u, progress_s=s,
        duration=4., physical_times=times)
    expected = BoundToolReference(problem, u, s, 4., source.initial_pose, source.initial_twist)
    position, quaternion = expected.sample(times)
    torch.testing.assert_close(rebound.position, position, atol=0., rtol=0.)
    torch.testing.assert_close(rebound.quaternion, quaternion, atol=0., rtol=0.)
    np.testing.assert_allclose(rebound.position[0], source.initial_pose[:3, 3], atol=1.e-15, rtol=0.)
    assert expected.nominal_phases[0] is source.nominal_phases[0]
    assert expected.nominal_phases[0].attachment is not None and expected.nominal_phases[0].parent is not None
    np.testing.assert_array_equal(source.sample_numpy(np.linspace(0., 2., 9)), saved_source)
    assert np.max(abs(source.sample_numpy(1.) - expected.sample_numpy(2.))) > .01
    assert rebound.report(3.)['physical_times_s'] == times.tolist()


@pytest.mark.parametrize('angle', [0., .3, np.pi - 1.e-7, np.pi + 1.e-7])
def test_rotation_identity_antipodes_and_each_pi_side(angle):
    problem = constant_problem()
    reference = BoundToolReference(problem, [0., 1.], [0., 1.], 1., np.eye(4), np.zeros(6))
    bound = ReferenceObjective(1., 1., 1., 0., 1.).bind(reference, [0., 1.])
    theta = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
    quaternion = torch.stack((torch.cos(theta / 2), theta * 0., theta * 0., torch.sin(theta / 2))).expand(2, 4)
    value = bound.cost(bound.position, quaternion)
    gradient, = torch.autograd.grad(value, theta)
    # BEGIN independent SO(3) matrix distance is smooth on both sides of pi.
    expected = np.linalg.norm(np.eye(3)-Rotation.from_rotvec([0., 0., angle]).as_matrix())**2/2
    assert float(value.detach()) == pytest.approx(expected, abs=1.e-14)
    assert float(gradient) == pytest.approx(2*np.sin(angle), abs=1.e-13)
    # END sign-invariant reference scoring leaves hard event angles unchanged.
    assert torch.isfinite(gradient)
    torch.testing.assert_close(value.detach(), bound.cost(bound.position, -quaternion.detach()), atol=0., rtol=0.)


def test_reference_pose_gradient_matches_finite_difference():
    _, reference = moving_reference()
    bound = ReferenceObjective(.03, .4, .7, 0., 1.).bind(reference, [0., .4, .9, 1.5, 2.])
    error = torch.tensor([.01, -.02, .03, .9, .13, -.19, .24], dtype=torch.float64, requires_grad=True)

    def cost(value):
        return bound.cost(bound.position + value[:3],
            quaternion_multiply(bound.quaternion, value[3:]))

    assert torch.autograd.gradcheck(cost, (error,), eps=1.e-6, atol=2.e-8, rtol=2.e-6)


# BEGIN a constant angular rate has an analytic physical-time integral.
def test_angular_motion_integral_and_gradient_use_the_physical_clock():
    problem = constant_problem()
    reference = BoundToolReference(problem, [0., 1.], [0., 1.], 1., np.eye(4), np.zeros(6))
    times = torch.tensor([0., .2, .55, 1.], dtype=torch.float64)
    bound = ReferenceObjective(.02, .1, 1., 3., 1.).bind(reference, times)
    speed = torch.tensor(.8, dtype=torch.float64, requires_grad=True)
    phase = speed*times/2
    zeros = torch.zeros_like(phase)
    quaternion = torch.stack((torch.cos(phase),zeros,zeros,torch.sin(phase)),dim=-1)[None]
    cost = bound.angular_motion_cost(quaternion)
    torch.testing.assert_close(cost, 3*speed.square(), rtol=1e-12, atol=1e-12)
    derivative, = torch.autograd.grad(cost, speed)
    torch.testing.assert_close(derivative, 6*speed, rtol=1e-12, atol=1e-12)
# END the same term is used for numerical and exported-curve objectives.
