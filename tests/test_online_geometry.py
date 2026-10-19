"""CPU contract tests for bounded ordinary collision attempts."""
# BEGIN pair scope, native-grid adaptation, missing evidence and model limits.
from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import pytest
import torch

from manipulation.online_geometry import FrozenESDF, FrozenPrimitives, OnlinePlanningScene
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, ObservedSpace, ContactRule, CollisionOwner
from native_scene_fixtures import bind_fixed_robot


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
    return bind_fixed_robot(result)


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
    bind_fixed_robot(current)
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
    assert record['schema'] == 'online_scene_v5'
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


def test_complete_primitive_requires_bound_native_robot_instead_of_proxy_fallback():
    original = scene()
    current = OnlinePlanningScene(background=original.background, owners=(), coverage=original.coverage,
        reference_frame='world', robot_sphere_materials=('body',), robot_geometry_source='unbound',
        has_gripper=False)
    with pytest.raises(ValueError, match='bound complete native robot'):
        current.query_state(np.array([[0., 0., 0., .01]]), np.eye(4), .5)


# BEGIN scene-level native distance checks catch duplicate proxy vetoes and gradients.
def native_box_scene(*, target_x=.04, contact=None, attached=False, sphere_count=1):
    original = scene()
    transform = np.eye(4)
    transform[0, 3] = target_x
    target = FrozenPrimitives(primitives=[{'kind': 'cuboid', 'dimensions': [.02, .08, .04]}],
        owner_ids=('target',), source='explicit complete obstacle')
    owner = CollisionOwner('target', target,
        PosePath(np.array([0., 1.]), np.stack([transform]*2), 'frozen target'),
        'complete target', geometry_complete=True,
        local_spheres=[[0., 0., 0., .05]] if attached else None,
        attachments=(Attachment((0., 1.), [target_x, 0., 0.], [1., 0., 0., 0.], 'held', None),)
            if attached else ())
    groups = [{'body_id': 'thin', 'frame': 'slider', 'geometry_names': ['thin'],
               'sphere_indices': list(range(sphere_count)),
               'local_spheres': [[0., 0., 0., .09]]*sphere_count,
               'source': 'complete thin source box'}]
    current = OnlinePlanningScene(background=original.background, owners=(owner,), coverage=original.coverage,
        reference_frame='world', robot_sphere_materials=('thin',)*sphere_count,
        robot_geometry_source='registered thin box', has_gripper=False,
        robot_collision_bodies=groups, contacts=() if contact is None else (contact,))
    urdf = ('<robot name="linear"><link name="base"/><link name="slider"/>'
            '<joint name="slide" type="prismatic"><parent link="base"/><child link="slider"/>'
            '<axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/>'
            '</joint></robot>')
    native = {'urdf_xml': urdf, 'geometries': [{'name': 'thin', 'frame': 'slider', 'kind': 'box',
        'dimensions': [.02, .16, .025], 'T_frame_geometry': np.eye(4).tolist(),
        'source': 'complete original thin box'}]}
    current.bind_online_robot(robot_sphere_geometry=[{'link_name': 'slider',
        'sphere_indices': list(range(sphere_count)), 'centers_local': [[0., 0., 0.]]*sphere_count,
        'radii': [.09]*sphere_count}], native_robot_model=native, native_joint_names=('slide',),
        native_self_pairs=(), robot_model={'T_reference_base': np.eye(4).tolist()})
    return current


def box_terms(current, q, progress=.5):
    position = torch.cat((q, torch.zeros((*q.shape[:-1], 2), dtype=q.dtype)), -1)
    spheres = torch.cat((position[..., None, :], q.new_full((*q.shape[:2], 1, 1), .09)), -1)
    return current.components(spheres, position, q.new_tensor([1., 0., 0., 0.]).expand(*q.shape[:2], 4),
        q.new_full(q.shape[:2], progress), joint_position=q, joint_names=('slide',))


def test_scene_uses_entity_zero_set_and_gradient_at_entry_optimization_and_restoration():
    current = native_box_scene()
    owner, = current.owners
    # The conservative proxy overlaps even though the original solids have
    # a 20 mm x gap. This positive proxy term must not survive as another cost.
    assert owner.query(torch.tensor([[-.04, 0., 0., .09]], dtype=torch.float64)).item() > 0
    q = torch.zeros((1, 1, 1), dtype=torch.float64, requires_grad=True)
    terms = box_terms(current, q)
    assert terms['collision/robot/target'].item() == 0
    sum(value.sum() for value in terms.values()).backward()
    torch.testing.assert_close(q.grad, torch.zeros_like(q))
    restored = OnlinePlanningScene.from_binding(current.to_execution_binding())
    for model in (current, restored):
        clear = model.query_state(np.array([[0., 0., 0., .09]]), np.eye(4), .5,
            joint_position=[0.], joint_names=('slide',))
        assert clear['passed']
        penetrating = torch.tensor([[[.025]]], dtype=torch.float64, requires_grad=True)
        collision = box_terms(model, penetrating)['collision/robot/target']
        assert collision.item() == pytest.approx(.005, abs=2e-8)
        collision.sum().backward()
        assert penetrating.grad.item() == pytest.approx(1., abs=1e-8)


def test_scene_attachment_gradient_cancels_both_robot_and_owner_motion():
    current = native_box_scene(target_x=.015, attached=True)
    q = torch.tensor([[[.02], [-.03]]], dtype=torch.float64, requires_grad=True)
    collision = box_terms(current, q)['collision/robot/target']
    torch.testing.assert_close(collision, q.new_full((1, 2), .005), rtol=0, atol=2e-8)
    collision.sum().backward()
    torch.testing.assert_close(q.grad, torch.zeros_like(q), rtol=0, atol=2e-8)


def test_partial_native_component_permission_is_rejected_once_with_rule_identity():
    partial = ContactRule('partial-contact', 'robot', 'target', (.25, .75), (0,), 'explicit material role')
    with pytest.raises(ValueError, match='partial-contact.*part of component robot/thin'):
        native_box_scene(contact=partial, sphere_count=2)
    complete = ContactRule('whole-contact', 'robot', 'target', (.25, .75), (0, 1), 'complete component role')
    current = native_box_scene(contact=complete, sphere_count=2)
    spheres = np.array([[.025, 0., 0., .09]]*2)
    for progress, expected in ((.5, 0.), (.1, .005)):
        report = current.query_state(spheres, np.eye(4), progress,
            joint_position=[.025], joint_names=('slide',))
        assert report['components']['collision/robot/target'] == pytest.approx(expected, abs=2e-8)


def test_unpermitted_native_self_and_world_need_no_sphere_material_component_map():
    original = scene()
    target = FrozenPrimitives(primitives=[{'kind': 'sphere', 'dimensions': [.01]}],
        owner_ids=('target',), source='complete target sphere')
    path = PosePath(np.array([0., 1.]), np.stack([np.eye(4)]*2), 'fixed target')
    owner = CollisionOwner('target', target, path, 'complete target', geometry_complete=True)
    current = OnlinePlanningScene(background=original.background, owners=(owner,), coverage=original.coverage,
        reference_frame='world', robot_sphere_materials=('first', 'second'),
        robot_geometry_source='complete native solids without materials', has_gripper=False)
    centers = [[0., 0., 0.], [.015, 0., 0.]]
    geometry = []
    for index, name in enumerate(('first', 'second')):
        transform = np.eye(4)
        transform[:3, 3] = centers[index]
        geometry.append({'name': name, 'frame': 'base', 'kind': 'sphere', 'dimensions': [.01],
            'T_frame_geometry': transform.tolist(), 'source': 'explicit native sphere'})
    arguments = dict(robot_sphere_geometry=[{'link_name': 'base', 'sphere_indices': [0, 1],
        'centers_local': centers, 'radii': [.01, .01]}],
        native_robot_model={'urdf_xml': '<robot name="fixed"><link name="base"/></robot>', 'geometries': geometry},
        native_joint_names=(), native_self_pairs=[{'first': 'first', 'second': 'second', 'margin': 0.}],
        robot_model={'T_reference_base': np.eye(4).tolist()})
    current.bind_online_robot(**arguments)
    assert current.robot_collision_bodies == ()
    restored = OnlinePlanningScene.from_binding(current.to_execution_binding())
    for value in (current, restored):
        report = value.query_state(np.array([[*center, .01] for center in centers]), np.eye(4), .5)
        assert report['components']['collision/robot/self'] == pytest.approx(.005, abs=2e-8)
        assert report['components']['collision/robot/target'] == pytest.approx(.025, abs=2e-8)
        separated = np.eye(4)
        separated[0, 3] = .1
        report = value.query_state(np.array([[*center, .01] for center in centers]), np.eye(4), .5,
            owner_poses={'target': separated})
        assert report['components']['collision/robot/target'] == 0
        assert report['components']['collision/robot/self'] == pytest.approx(.005, abs=2e-8)
    current.contacts = (ContactRule('unmapped-contact', 'robot', 'target', (0., 1.), (0,), 'declared material'),)
    with pytest.raises(ValueError, match='unmapped-contact.*without a complete registered robot component'):
        current.bind_online_robot(**arguments)


def test_alias_without_world_pairs_needs_no_unused_exclusion_component_map():
    original = native_box_scene()
    robot = deepcopy(original.native_robot_model)
    robot['geometries'].append({'name': 'mount', 'frame': 'base', 'kind': 'box', 'dimensions': [.02]*3,
        'T_frame_geometry': np.eye(4).tolist(), 'source': 'registered fixed mount'})
    primitive = {'kind': 'cuboid', 'dimensions': [.02]*3, 'native_binding': {'geometry_name': 'mount'}}
    query = FrozenPrimitives(primitives=[primitive], owner_ids=('installation',), source='same native mount')
    path = PosePath(np.array([0., 1.]), np.stack([np.eye(4)]*2), 'same fixed native pose')
    owner = CollisionOwner('installation', query, path, 'same native mount', geometry_complete=True,
        native_geometry_names=('mount',), robot_excluded_spheres=(0,),
        robot_exclusion_source='registered self policy', robot_exclusion_geometry_source='native source')
    current = OnlinePlanningScene(background=original.background, owners=(owner,), coverage=original.coverage,
        reference_frame='world', robot_sphere_materials=('thin',), robot_geometry_source='native source', has_gripper=False)
    arguments = dict(robot_sphere_geometry=original.robot_sphere_geometry, native_robot_model=robot,
        native_joint_names=('slide',), native_self_pairs=[{'first': 'thin', 'second': 'mount', 'margin': 0.}],
        robot_model=original.robot_model)
    current.bind_online_robot(**arguments)
    report = current.query_state(np.array([[.015, 0., 0., .09]]), np.eye(4), .5,
        joint_position=[.015], joint_names=('slide',))
    assert report['components']['collision/robot/self'] == pytest.approx(.005, abs=2e-8)
    assert 'collision/robot/installation' not in report['components']
    assert current._native_excluded_pairs == ()
    separated = current.query_state(np.array([[.04, 0., 0., .09]]), np.eye(4), .5,
        joint_position=[.04], joint_names=('slide',))
    assert separated['components']['collision/robot/self'] == 0
    # An extra nonalias target component leaves real world pairs. Its explicit
    # exclusion must still resolve a full robot component, even if separated.
    owner.query = FrozenPrimitives(primitives=[primitive, {'kind': 'sphere', 'dimensions': [.01],
        'pose': [.5, 0., 0., 1., 0., 0., 0.]}], owner_ids=('installation',), source='mount and extra obstacle')
    with pytest.raises(ValueError, match='robot exclusion for installation.*without a complete registered robot component'):
        current.bind_online_robot(**arguments)
# END generic geometry tests never name a task, robot brand or saved failure pair.
# END tests exercise the ordinary model, not expected task-effect success.
