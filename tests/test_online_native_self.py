"""Scene self-collision semantics come from registered solids and current q."""
# BEGIN native pair behavior and lifecycle replace removed sphere-index caches.
import numpy as np
import torch

from manipulation.online_geometry import FrozenESDF, OnlinePlanningScene
from manipulation.planning_scene import ObservedSpace, PrismaticGripper


URDF = '''<robot name="arm_palm_and_finger">
  <link name="base"/><link name="tool"/><link name="palm_link"/><link name="finger_link"/>
  <joint name="arm" type="prismatic"><parent link="base"/><child link="tool"/>
    <axis xyz="1 0 0"/><limit lower="-.5" upper=".5" effort="10" velocity="1"/></joint>
  <joint name="palm_fixed" type="fixed"><parent link="tool"/><child link="palm_link"/>
    <origin xyz=".01 0 0"/></joint>
  <joint name="finger" type="prismatic"><parent link="tool"/><child link="finger_link"/>
    <axis xyz="1 0 0"/><limit lower="-.2" upper=".2" effort="10" velocity="1"/></joint>
</robot>'''


def make_scene(*, independent_hand=False, background_distance=1.):
    radius = float(np.linalg.norm([.02]*3))
    hand = None if not independent_hand else PrismaticGripper(
        controller='binary', joint_names=('finger',), current_positions=[-.01],
        joint_limits=[[-.2, .2]], joint_bounds=None, coupling_matrix=np.empty((0, 1)),
        coupling_offset=[], sphere_indices=(0, 1), local_spheres=[[0., 0., 0., radius], [.01, 0., 0., radius]],
        sphere_joint_translation=[[[1.], [0.], [0.]], [[0.], [0.], [0.]]],
        commands=(), source='explicit independently controlled finger and invariant palm',
        control_timestep_s=.01)
    values = np.full((11, 11, 11), background_distance, np.float32)
    background = FrozenESDF(values, origin=[-.5]*3, voxel_size=.1,
        owner_ids=('background',), source='explicit ordinary field')
    coverage = ObservedSpace(np.ones_like(values, bool), origin=[-.5]*3, voxel_size=.1,
        observation_ids=('current',), reference_frame='world')
    frames, names = ('finger_link', 'palm_link', 'base'), ('finger_solid', 'palm_solid', 'arm_solid')
    centres = [[0., 0., 0.], [0., 0., 0.], [.035, 0., 0.]]
    bodies = [{'body_id': name, 'frame': frame, 'geometry_names': [name], 'sphere_indices': [i],
               'local_spheres': [[*centres[i], radius]], 'source': 'registered component identity'}
              for i, (name, frame) in enumerate(zip(names, frames))]
    current = OnlinePlanningScene(background=background, owners=(), coverage=coverage,
        reference_frame='world', robot_sphere_materials=(None,)*3, robot_geometry_source='native CPU fixture',
        gripper=hand, has_gripper=hand is not None, robot_collision_bodies=bodies)
    geometry = []
    for name, frame, centre in zip(names, frames, centres):
        pose = np.eye(4)
        pose[:3, 3] = centre
        geometry.append(dict(name=name, frame=frame, kind='box', dimensions=[.04]*3,
                             T_frame_geometry=pose.tolist(), source='registered analytic box'))
    sphere_geometry = [{'link_name': frame, 'sphere_indices': [i], 'centers_local': [centres[i]], 'radii': [radius]}
                       for i, frame in enumerate(frames)]
    current.bind_online_robot(robot_sphere_geometry=sphere_geometry,
        native_robot_model={'urdf_xml': URDF, 'geometries': geometry}, native_joint_names=('arm', 'finger'),
        native_self_pairs=[{'first': 'finger_solid', 'second': 'palm_solid', 'margin': 0.},
                           {'first': 'palm_solid', 'second': 'arm_solid', 'margin': 0.}],
        robot_model={'T_reference_base': np.eye(4).tolist()})
    return current


def evaluate(current, q, *, measured=False):
    progress = q.new_tensor([[0., .5]]).expand(q.shape[:2])
    position = torch.stack((q[..., 0], q[..., 0]*0, q[..., 0]*0), -1)
    quaternion = q.new_tensor([1., 0., 0., 0.]).expand(*q.shape[:2], 4)
    balls = q.new_tensor([[0., 0., 0., np.sqrt(3)*.02], [.01, 0., 0., np.sqrt(3)*.02],
                         [.035, 0., 0., np.sqrt(3)*.02]]).expand(*q.shape[:2], 3, 4)
    names = ('arm', 'finger') if q.shape[-1] == 2 else ('arm',)
    return current.components(balls, position, quaternion, progress, q, names,
        trajectory_time=progress, active_contact_rule_ids=() if measured else None)


def rebind(current, pairs):
    current.bind_online_robot(robot_sphere_geometry=current.robot_sphere_geometry,
        native_robot_model=current.native_robot_model, native_joint_names=current.native_joint_names,
        native_self_pairs=pairs, robot_model=current.robot_model)


def test_native_self_values_and_joint_gradients_do_not_come_from_proxy_locations():
    current = make_scene()
    q = torch.tensor([[[0., -.01], [.1, -.1]]], dtype=torch.float64, requires_grad=True)
    hard = evaluate(current, q)['collision/robot/self']
    # First: finger-palm .02 plus palm-arm .015. Second: both pairs clear.
    torch.testing.assert_close(hard, q.new_tensor([[.035, 0.]]), rtol=0, atol=2e-8)
    hard.sum().backward()
    torch.testing.assert_close(q.grad, q.new_tensor([[[1., 1.], [0., 0.]]]), rtol=0, atol=2e-8)


def test_unknown_future_finger_removes_only_its_native_dependents():
    current = make_scene(independent_hand=True, background_distance=0.)
    q = torch.zeros((1, 2, 1), dtype=torch.float64, requires_grad=True)
    terms = evaluate(current, q)
    # Native tree says the finger depends on its unknown joint; palm and arm
    # do not. The palm-arm collision persists after the nominal model ends.
    torch.testing.assert_close(terms['collision/robot/self'], q.new_tensor([[.035, .015]]), rtol=0, atol=2e-8)
    terms['collision/robot/self'].sum().backward()
    torch.testing.assert_close(q.grad, torch.ones_like(q), rtol=0, atol=2e-8)
    radius = np.sqrt(3)*.02
    torch.testing.assert_close(terms['collision/robot/background'], q.new_tensor([[3*radius, 2*radius]]),
                               rtol=0, atol=2e-8)
    actual = torch.tensor([[[0., -.01], [0., -.01]]], dtype=torch.float64)
    measured = evaluate(current, actual, measured=True)
    torch.testing.assert_close(measured['collision/robot/self'], q.new_tensor([[.035, .035]]), rtol=0, atol=2e-8)


def test_clone_retains_old_native_policy_after_rebinding_original():
    original = make_scene()
    clone = original.clone_for_rollout()
    previous = clone.native_collision_model
    q = torch.tensor([[[0., -.01], [0., -.01]]], dtype=torch.float64)
    before = evaluate(clone, q)['collision/robot/self']
    rebind(original, [{'first': 'palm_solid', 'second': 'arm_solid', 'margin': 0.}])
    assert clone.native_collision_model is previous
    assert original.native_collision_model is not previous
    torch.testing.assert_close(evaluate(clone, q)['collision/robot/self'], before, rtol=0, atol=0)
    torch.testing.assert_close(evaluate(original, q)['collision/robot/self'], q.new_tensor([[.015, .015]]),
                               rtol=0, atol=2e-8)


def test_archive_preserves_native_pairs_and_unknown_measured_state_semantics():
    current = make_scene(independent_hand=True)
    binding = current.to_execution_binding()
    restored = OnlinePlanningScene.from_binding(binding)
    assert restored.native_joint_names == current.native_joint_names
    assert restored.native_self_pairs == current.native_self_pairs
    assert restored.native_robot_model == current.native_robot_model
    assert 'robot_self_collision_pairs' not in binding
    for q, measured in [(torch.zeros((1, 2, 1), dtype=torch.float64), False),
                        (torch.tensor([[[0., -.01], [0., -.01]]], dtype=torch.float64), True)]:
        expected, actual = evaluate(current, q, measured=measured), evaluate(restored, q, measured=measured)
        assert expected.keys() == actual.keys()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
# END neither caches nor sphere ordering can replace the actual solid policy.
