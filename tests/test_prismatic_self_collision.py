"""Native hand geometry shares the ordinary complete robot collision model."""
# BEGIN complete URDF solids replace the deleted hand-only separating-axis path.
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from manipulation.online_geometry import FrozenESDF, OnlinePlanningScene
from manipulation.planning_scene import ObservedSpace, PrismaticGripper


URDF = '''<robot name="two_prismatic_materials">
  <link name="base"/><link name="left_link"/><link name="right_link"/>
  <joint name="left" type="prismatic"><parent link="base"/><child link="left_link"/>
    <axis xyz="1 0 0"/><limit lower="-.2" upper=".2" effort="10" velocity="1"/></joint>
  <joint name="right" type="prismatic"><parent link="base"/><child link="right_link"/>
    <axis xyz="1 0 0"/><limit lower="-.2" upper=".2" effort="10" velocity="1"/></joint>
</robot>'''


def hand(*, nominal=False):
    """Keep the existing command/nominal helper used by finger-clock tests."""
    initial = np.array([-.05, .05])
    def factory(target):
        def build(duration, mapping):
            from manipulation.gripper_trajectory import NominalFingerGeometry
            return NominalFingerGeometry(('left', 'right'), [0., duration], [initial, target],
                np.tile((np.asarray(target)-initial)/duration, (2, 1)), duration, duration,
                'explicit synthetic linear finger fixture')
        return build
    radii = float(np.linalg.norm([.01, .1, .01]))
    matrices = np.zeros((2, 3, 2))
    matrices[0, 0, 0] = matrices[1, 0, 1] = 1.
    result = PrismaticGripper(controller='timed_reference', joint_names=('left', 'right'),
        current_positions=initial, joint_limits=[[-.2, .2], [-.2, .2]], joint_bounds=None,
        coupling_matrix=np.empty((0, 2)), coupling_offset=[], sphere_indices=(0, 1),
        local_spheres=[[0., 0., 0., radii]]*2, sphere_joint_translation=matrices,
        commands=(), source='native prismatic test assembly',
        control_timestep_s=.01, finger_trajectory_factory=factory([0., 0.]),
        nominal_finger_trajectory_factory=(lambda duration, mapping, reference:
            factory([-.06, .06])(duration, mapping)) if nominal else None)
    result.bind_clock(1., None)
    return result


def scene(current_hand, *, distance=1., external_position=(.5, .5, .5), base=None):
    """Bind all finger and external self geometry through the ordinary API."""
    values = np.full((11, 11, 11), distance, np.float32)
    background = FrozenESDF(values, origin=[-1.]*3, voxel_size=.2,
        owner_ids=('background',), source='bounded observed test source')
    coverage = ObservedSpace(np.ones_like(values, bool), origin=[-1.]*3, voxel_size=.2,
        observation_ids=('A',), reference_frame='world')
    radius = current_hand.local_spheres[0, 3]
    frames = ('left_link', 'right_link', 'base')
    centres, radii = [[0.]*3, [0.]*3, list(external_position)], [radius, radius, .01]
    bodies = [{'body_id': f'body_{i}', 'frame': frame, 'geometry_names': [f'native_{i}'],
        'sphere_indices': [i], 'local_spheres': [[*centres[i], radii[i]]],
        'source': 'explicit complete registered component'} for i, frame in enumerate(frames)]
    result = OnlinePlanningScene(background=background, owners=(), coverage=coverage,
        reference_frame='world', robot_sphere_materials=(None,)*3, robot_geometry_source='test native geometry',
        gripper=current_hand, robot_collision_bodies=bodies)
    vertices = np.asarray(list(product((-.01, .01), (-.1, .1), (-.01, .01))))
    geometry = [dict(name=f'native_{i}', frame=frames[i], kind='convex',
                     vertices_frame=vertices.tolist(), source='complete native finger convex') for i in range(2)]
    pose = np.eye(4)
    pose[:3, 3] = external_position
    geometry.append(dict(name='native_2', frame='base', kind='box', dimensions=[.02]*3,
                         T_frame_geometry=pose.tolist(), source='external native robot component'))
    result.bind_online_robot(robot_sphere_geometry=[
        {'link_name': frame, 'sphere_indices': [i], 'centers_local': [centres[i]], 'radii': [radii[i]]}
        for i, frame in enumerate(frames)],
        native_robot_model={'urdf_xml': URDF, 'geometries': geometry},
        native_joint_names=('left', 'right'), native_self_pairs=[
            {'first': 'native_0', 'second': 'native_1', 'margin': 0.},
            {'first': 'native_0', 'second': 'native_2', 'margin': 0.}],
        robot_model={'T_reference_base': (np.eye(4) if base is None else base).tolist()})
    return result


def measured_components(current, q, *, base=None):
    transform = np.eye(4) if base is None else base
    progress = q.new_full(q.shape[:2], .5)
    position = q.new_tensor(transform[:3, 3]).expand(*q.shape[:2], 3)
    quaternion = q.new_tensor(Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]).expand(*q.shape[:2], 4)
    balls = q.new_zeros((*q.shape[:2], 3, 4))
    balls[..., 2, :3] = q.new_tensor([.5, .5, .5])
    balls[..., 2, 3] = .01
    return current.components(balls, position, quaternion, progress, q, ('left', 'right'),
        trajectory_time=progress, active_contact_rule_ids=())


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


@pytest.mark.parametrize('rotated', [False, True])
def test_native_fingers_remove_proxy_overlap_and_keep_real_collision_gradient(rotated):
    base = np.eye(4)
    if rotated:
        base[:3, :3] = Rotation.from_rotvec([.4, -.2, .3]).as_matrix()
        base[:3, 3] = [.2, -.3, .5]
    current = scene(hand(nominal=True), base=base)
    clear = torch.tensor([[[-.05, .05]]], dtype=torch.float64, requires_grad=True)
    assert 2*current.gripper.local_spheres[0, 3]-.1 > .1
    terms = measured_components(current, clear, base=base)
    assert terms['collision/robot/self'].item() == 0.
    collision = torch.tensor([[[-.002, .003]]], dtype=torch.float64, requires_grad=True)
    hard = measured_components(current, collision, base=base)['collision/robot/self']
    assert hard.item() == pytest.approx(.015, abs=2e-8)
    hard.sum().backward()
    torch.testing.assert_close(collision.grad, collision.new_tensor([[[1., -1.]]]), rtol=0, atol=2e-8)


def test_external_self_pairs_and_observed_fields_remain_checked():
    q = torch.tensor([[[-.05, .05]]], dtype=torch.float64)
    external = scene(hand(nominal=True), external_position=(-.04, 0., 0.))
    assert measured_components(external, q)['collision/robot/self'].item() == pytest.approx(.01, abs=2e-8)
    observed = scene(hand(nominal=True), distance=0.)
    terms = measured_components(observed, q)
    assert terms['collision/robot/self'].item() == 0.
    assert terms['collision/robot/background'].item() > .2


def test_binding_roundtrip_retains_the_complete_native_model_and_measured_values():
    original = scene(hand(nominal=True))
    binding = original.to_execution_binding()
    restored = OnlinePlanningScene.from_binding(binding)
    assert restored.native_robot_model == original.native_robot_model
    assert restored.native_self_pairs == original.native_self_pairs
    assert 'prepared_self_collision' not in binding['gripper']
    for coordinates in ([-.05, .05], [-.002, .003]):
        q = torch.tensor([[coordinates]], dtype=torch.float64)
        actual = measured_components(restored, q)
        for name, value in measured_components(original, q).items():
            torch.testing.assert_close(value, actual[name], rtol=0, atol=0)
    corrupt = deepcopy(binding)
    corrupt['native_self_pairs'][0]['first'] = 'absent_component'
    with pytest.raises(ValueError, match='registered solids'):
        OnlinePlanningScene.from_binding(corrupt)
# END nominal commands and measured execution share one registered robot model.
