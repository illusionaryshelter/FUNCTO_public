"""Necessary counterexamples for native hand pairs and shared state meaning."""
# BEGIN convex replacement is exact in scope, while other collisions remain.
from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from manipulation.online_geometry import FrozenESDF, OnlinePlanningScene
from manipulation.planning_scene import GripperState, ObservedSpace, PrismaticGripper, _self_pair_sum


def hand(*, nominal=False):
    initial = np.array([-.05, .05])
    def factory(target):
        def build(duration, mapping):
            # BEGIN the synthetic linear geometry declares its complete scope.
            from manipulation.gripper_trajectory import NominalFingerGeometry
            return NominalFingerGeometry(('left', 'right'), [0., duration], [initial, target],
                np.tile((np.asarray(target)-initial)/duration, (2, 1)), duration, duration,
                'explicit synthetic linear finger fixture')
            # END command/geometry comparisons still use different targets.
        return build
    vertices = np.asarray(list(product((-.01, .01), (-.1, .1), (-.01, .01))))
    radii = float(np.linalg.norm(vertices[0]))
    matrices = np.zeros((2, 3, 2))
    matrices[0, 0, 0] = matrices[1, 0, 1] = 1.
    geometry = [{'geometry_id': f'native_{index}', 'sphere_indices': [index],
                 'vertices': vertices.tolist(), 'joint_translation': matrices[index].tolist(),
                 'source': 'complete native box with its one enclosing sphere'} for index in range(2)]
    result = PrismaticGripper(controller='timed_reference', joint_names=('left', 'right'),
        current_positions=initial, joint_limits=[[-.2, .2], [-.2, .2]], joint_bounds=None,
        coupling_matrix=np.empty((0, 2)), coupling_offset=[], sphere_indices=(0, 1),
        local_spheres=[[0., 0., 0., radii]]*2, sphere_joint_translation=matrices,
        self_collision_pairs=((0, 1), (0, 2)), commands=(), source='native prismatic test assembly',
        control_timestep_s=.01, finger_trajectory_factory=factory([0., 0.]),
        nominal_finger_trajectory_factory=(lambda duration, mapping, reference:
            factory([-.06, .06])(duration, mapping)) if nominal else None,
        self_collision_geometry=geometry)
    result.bind_clock(1., None)
    return result


def state(hand, values, uncertain=False, half=None):
    values = torch.tensor([values], dtype=torch.float64)
    return GripperState(values, torch.zeros_like(values) if half is None else torch.tensor([half], dtype=torch.float64),
                        values.new_zeros(values.shape[:-1]), torch.full(values.shape[:-1], uncertain))


def scene(current_hand, *, distance=1.):
    values = np.full((11, 11, 11), distance, np.float32)
    valid = np.ones_like(values, bool)
    # BEGIN synthetic source values define material before proposal construction.
    from observed_geometry_fixtures import field_from_samples
    background = field_from_samples(values, origin=[-1.]*3, voxel_size=.2, observed=valid,
        owner_ids=('background',), source='bounded observed test source')
    # END collision proposals cannot replace source evidence.
    coverage = ObservedSpace(valid, origin=[-1.]*3, voxel_size=.2, observation_ids=('A',), reference_frame='world')
    result = OnlinePlanningScene(background=background, owners=(), coverage=coverage,
        reference_frame='world', robot_sphere_materials=(None,)*3, robot_geometry_source='test native geometry',
        gripper=current_hand, unknown_policy={'outside_regions': 'forbid', 'conditional_regions': []})
    radius = current_hand.local_spheres[0, 3]
    result.bind_online_robot(robot_sphere_geometry=[{'link_name': f'link_{i}', 'sphere_indices': [i],
        'centers_local': [[0., 0., 0.]], 'radii': [radius if i < 2 else .01]} for i in range(3)],
        robot_self_collision_pairs=[[0, 1], [0, 2]])
    return result


def test_shared_state_distinguishes_command_nominal_and_measured():
    command = hand()
    progress = torch.tensor([[0., 0.]], dtype=torch.float64)
    time = torch.tensor([[0., .5]], dtype=torch.float64)
    value = command.state(progress, None, None, trajectory_time=time)
    assert value.uncertain.tolist() == [[False, True]]
    assert value.coordinates[0, 0].tolist() == [-.05, .05]
    assert value.half[0, 1].tolist() == [.2, .2]
    nominal = hand(nominal=True)
    prediction = nominal.state(progress, None, None, trajectory_time=time)
    assert not prediction.uncertain.any()
    assert prediction.coordinates[0, 1].tolist() == [-.055, .055]
    measured = nominal.state(progress, torch.tensor([[[-.07, .07], [-.08, .08]]], dtype=torch.float64),
        ('left', 'right'), trajectory_time=time, measured=True)
    assert measured.coordinates[0, 1].tolist() == [-.08, .08]
    assert not measured.uncertain.any()
    with pytest.raises(ValueError, match='complete named'):
        nominal.state(progress, None, None, trajectory_time=time, measured=True)


def test_native_convex_pair_replaces_false_sphere_overlap_and_keeps_real_collision_gradient():
    current = hand(nominal=True)
    q = state(current, [[-.05, .05]])
    model = current.self_collision_model
    for rot, translation in [(np.eye(3), np.zeros(3)), (Rotation.from_rotvec([.4, -.2, .3]).as_matrix(), np.array([.2, -.3, .5]))]:
        p = torch.tensor(translation, dtype=torch.float64).reshape(1, 1, 3)
        r = torch.tensor(Rotation.from_matrix(rot).as_quat()[[3, 0, 1, 2]], dtype=torch.float64).reshape(1, 1, 4)
        balls = current.spheres(torch.zeros((1, 1, 3, 4), dtype=torch.float64), p, r, q.coordinates, q.half)
        assert _self_pair_sum(balls, np.array([[0, 1]])).item() > .1
        assert model.components(q)[0].item() == 0.
    collision = state(current, [[-.002, .003]])
    collision.coordinates.requires_grad_()
    hard, conditional = model.components(collision)
    assert hard.item() == pytest.approx(.015)
    assert conditional.item() == 0.
    hard.sum().backward()
    np.testing.assert_allclose(collision.coordinates.grad, [[[1., -1.]]], atol=2e-15, rtol=0)
    unresolved = state(current, [[0., 0.]], uncertain=True, half=[[.2, .2]])
    assert model.components(unresolved)[0].item() == 0.
    assert model.components(unresolved)[1].item() > 0.


def test_external_self_pairs_and_world_collisions_survive_exact_hand_replacement():
    current = hand(nominal=True)
    world = scene(current)
    assert world.native_self_collision_pairs == [[0, 2]]
    balls = np.array([[-.05, 0, 0, .102], [.05, 0, 0, .102], [-.05, 0, 0, .01]])
    result = world.query_state(balls, np.eye(4), .5, joint_position=[-.05, .05],
                               joint_names=('left', 'right'), trajectory_time=.5)
    assert result['components']['collision/robot/self'] > 0
    balls[2, :3] = [.5, .5, .5]
    assert scene(current, distance=0.).query_state(balls, np.eye(4), .5,
        joint_position=[-.05, .05], joint_names=('left', 'right'), trajectory_time=.5)['components']['collision/robot/background'] > 0


def test_binding_roundtrip_reuses_same_actual_geometry_and_rejects_incomplete_pair_identity():
    original = scene(hand(nominal=True))
    binding = original.to_execution_binding()
    restored = OnlinePlanningScene.from_binding(binding)
    balls = np.array([[-.05, 0, 0, .102], [.05, 0, 0, .102], [.5, .5, .5, .01]])
    for q in ([-.05, .05], [-.002, .003]):
        args = dict(joint_position=q, joint_names=('left', 'right'), trajectory_time=.5)
        assert original.query_state(balls, np.eye(4), .5, **args) == restored.query_state(balls, np.eye(4), .5, **args)
    corrupt = deepcopy(binding)
    corrupt['gripper']['prepared_self_collision']['replaced_pairs'] = []
    with pytest.raises(ValueError, match='Serialized prismatic'):
        OnlinePlanningScene.from_binding(corrupt)
    corrupt = deepcopy(binding)
    corrupt['robot_self_collision_pairs'] = [[0, 2]]
    with pytest.raises(ValueError, match='runtime self-pair policy'):
        OnlinePlanningScene.from_binding(corrupt)
# END no test grants permission to a complete finger or scene owner.
