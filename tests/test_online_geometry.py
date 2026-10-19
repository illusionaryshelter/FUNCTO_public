"""CPU contract tests for bounded ordinary collision attempts."""
# BEGIN pair scope, native-grid adaptation, missing evidence and model limits.
from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import pytest
import torch

from manipulation.online_geometry import FrozenESDF, FrozenPrimitives, OnlinePlanningScene
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import ObservedSpace, ContactRule, CollisionOwner


def field(distance=None, owner='background'):
    values = np.full((21, 21, 21), 1., np.float32) if distance is None else distance
    return FrozenESDF(values, origin=[-.1]*3, voxel_size=.01,
                      owner_ids=(owner,), source='explicit ordinary distance model')


def scene(*, background=None):
    coverage = ObservedSpace(np.zeros((21, 21, 21), bool), origin=[-.1]*3,
        voxel_size=.01, observation_ids=('current_rgbd',), reference_frame='world')
    result = OnlinePlanningScene(background=field() if background is None else background,
        owners=(), coverage=coverage, reference_frame='world', robot_sphere_materials=('body',),
        robot_geometry_source='calibrated robot', has_gripper=False)
    result.bind_online_robot(robot_sphere_geometry=[{'link_name': 'link', 'sphere_indices': [0],
        'centers_local': [[0., 0., 0.]], 'radii': [.01]}], robot_self_collision_pairs=[])
    return result


def test_unknown_and_grid_exterior_allow_attempt_without_claiming_observed_free():
    current = scene()
    for position in (0., .2):
        result = current.query_state(np.array([[position, 0., 0., .01]]), np.eye(4), 0.)
        assert result['passed']
        assert result['observation']['status'] == 'unknown'
        assert result['observation']['unknown_centres'] == 1
        assert not any(name.startswith(('unknown/', 'scene_update/')) for name in result['components'])
    assert not current.coverage.known_free.any()


def test_known_model_collision_still_rejects_in_unknown_space_and_has_gradient():
    axis = np.linspace(-.1, .1, 21)
    values = np.broadcast_to(axis[:, None, None], (21, 21, 21)).astype(np.float32)
    current = scene(background=field(values))
    report = current.query_state(np.array([[-.01, 0., 0., .01]]), np.eye(4), 0.)
    assert not report['passed']
    assert report['observation']['status'] == 'unknown'
    body = torch.tensor([[[[-.01, 0., 0., .01]]]], dtype=torch.float64, requires_grad=True)
    cost = current.evaluate(body, body.new_zeros((1, 1, 3)), body.new_tensor([[[1., 0., 0., 0.]]]), body.new_zeros((1, 1)))
    cost.sum().backward()
    assert body.grad[0, 0, 0, 0] < 0


def test_native_voxel_xyz_sign_centres_and_interpolation_follow_mapper_contract():
    # Unequal dimensions and an affine XYZ field detect accidental transposes,
    # sign flips and half-cell shifts without requiring CUDA or full geometry.
    shape, h, center = (3, 4, 5), .1, np.array([.2, -.3, .4])
    origin = center-(np.array(shape)-1)*h/2
    xyz = np.stack(np.meshgrid(*(origin[i]+np.arange(shape[i])*h for i in range(3)), indexing='ij'), -1)
    distances = xyz[..., 0]+2*xyz[..., 1]+3*xyz[..., 2]-.5
    voxel = SimpleNamespace(pose=[*center, 1., 0., 0., 0.], voxel_size=h,
                            feature_tensor=torch.tensor(distances, dtype=torch.float32))
    query = FrozenESDF.from_native_voxel(voxel, owner_ids=('wall',), source='Mapper CPU export fixture')
    np.testing.assert_allclose(query.origin, origin)
    assert query.distance.shape == shape
    points = torch.tensor(np.array([xyz[0, 0, 0], xyz[-1, -1, -1], center]), dtype=torch.float64)
    expected = points[:, 0]+2*points[:, 1]+3*points[:, 2]-.5
    torch.testing.assert_close(query.signed_distance(points), expected, rtol=1e-6, atol=1e-7)
    empty = FrozenESDF.from_native_voxel(SimpleNamespace(pose=voxel.pose, voxel_size=h,
        feature_tensor=torch.full(shape, 1e4)), owner_ids=('background',), source='empty native field')
    assert empty(torch.tensor([[*center, .01]])).item() == 0


def test_exact_pair_spheres_and_interval_are_the_only_interaction_permission():
    original = scene()
    path = PosePath(np.array([0., 1.]), np.stack([np.eye(4)]*2), 'current owner')
    owners = tuple(CollisionOwner(name,
        FrozenPrimitives(primitives=[{'kind': 'sphere', 'dimensions': [.02]}], owner_ids=(name,), source=name),
        path, name, geometry_complete=True) for name in ('target', 'other'))
    rule = ContactRule('interaction', 'robot', 'target', (.25, .75), (0,), 'bound contact role')
    current = OnlinePlanningScene(background=original.background, owners=owners, coverage=original.coverage,
        reference_frame='world', robot_sphere_materials=('finger', 'arm'), robot_geometry_source='robot',
        contacts=(rule,), has_gripper=False)
    body = np.array([[0., 0., 0., .01], [0., 0., 0., .01]])
    inside = current.query_state(body, np.eye(4), .5)
    outside = current.query_state(body, np.eye(4), .1)
    assert inside['components']['collision/robot/target'] == pytest.approx(.03)
    assert outside['components']['collision/robot/target'] == pytest.approx(.06)
    assert inside['components']['collision/robot/other'] == pytest.approx(.06)
    assert not inside['passed']
    assert not any(name.startswith('contact/') for name in inside['components'])


def test_current_archive_has_same_model_and_never_constructs_old_source_geometry():
    current = scene()
    known = np.zeros((21, 21, 21), bool)
    known[10, 10, 10] = True
    current.coverage = ObservedSpace(known, origin=[-.1]*3, voxel_size=.01,
        observation_ids=('current_rgbd',), reference_frame='world', known_free=np.zeros_like(known))
    record = current.to_execution_binding()
    assert record['schema'] == 'online_scene_v4'
    assert record['background']['kind'] == 'frozen_esdf_v1'
    assert not {'source_geometry', 'contact_model'} & record['background'].keys()
    restored = OnlinePlanningScene.from_binding(record)
    assert restored.coverage._known[10, 10, 10]
    assert not restored.coverage.known_free[10, 10, 10]
    body = np.array([[0., 0., 0., .01]])
    assert current.query_state(body, np.eye(4), .5) == restored.query_state(body, np.eye(4), .5)
    old = deepcopy(record)
    old['schema'] = 'online_scene_v3'
    with pytest.raises(ValueError, match='schema'):
        OnlinePlanningScene.from_binding(old)
    assert not hasattr(current.background, 'source_geometry')
    assert not hasattr(current, 'native_world_collision_components')


def test_primitive_pose_and_dimensions_keep_exact_native_meaning():
    query = FrozenPrimitives(primitives=[{'kind': 'cuboid', 'dimensions': [.02, .04, .06],
        'pose': [.05, 0., 0., 1., 0., 0., 0.]}], owner_ids=('mount',), source='calibrated primitive')
    points = torch.tensor([[.05, 0., 0.], [.07, 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(query.signed_distance(points), torch.tensor([-.01, .01], dtype=torch.float64))
    torch.testing.assert_close(FrozenESDF.from_record(query.record()).signed_distance(points), query.signed_distance(points))


def test_nonfinite_state_and_malformed_interaction_cannot_be_reported_clear():
    with pytest.raises(ValueError, match='finite'):
        scene().query_state(np.array([[np.nan, 0., 0., .01]]), np.eye(4), 0.)
    with pytest.raises(ValueError, match='unique sphere'):
        ContactRule('bad', 'robot', 'target', (0., 1.), (0, 0), 'test')


def test_pair_workspace_accounts_for_real_expansion_without_source_reservations():
    current = scene()
    current.robot_sphere_materials = ('body', 'body')
    current.bind_online_robot(robot_sphere_geometry=[{'link_name': 'link', 'sphere_indices': [0, 1],
        'centers_local': [[0., 0., 0.], [1., 0., 0.]], 'radii': [.01, .01]}],
        robot_self_collision_pairs=[[0, 1]]*2048)
    for states, expanded_pairs in ((1, 2048), (4096, 1048576)):
        allocation = current.estimated_workspace_allocation(1, states)
        assert allocation['gpu_live'] >= expanded_pairs*24*4
        assert allocation['cpu_live'] >= expanded_pairs*24*8
# END tests exercise the ordinary model, not expected task-effect success.
