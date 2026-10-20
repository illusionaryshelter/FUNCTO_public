"""Current support is a finite entrance relation, not future placement pose."""

# BEGIN necessary CPU counterexamples for the independent support capability.
from copy import deepcopy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manipulation.initial_support import InitialSupportRelation, PreparedInitialSupport
from manipulation.online_geometry import FrozenESDF, FrozenPrimitives, OnlinePlanningScene
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, CollisionOwner, ObservedSpace
from observed_geometry_fixtures import point_source, surface_source, source_from_samples


def relation(**changes):
    return InitialSupportRelation(**{
        'relation_id': 'entry_support', 'actor': 'payload', 'target_owner': 'support',
        'interval': (0., 1.), 'plane_origin': [0., 0., 0.], 'plane_normal': [0., 0., 1.],
        'source': {'kind': 'observed_plane_with_declared_initial_support', 'observation_ids': ['camera'],
                   'assumption': 'Currently supported according to declared observation context',
                   'scope': 'Independent finite observed plane layer, not free-space evidence'}, **changes})


def owners(geometry=None, *, entrance_z=.001):
    pose = np.eye(4)
    pose[2, 3] = entrance_z
    path = lambda matrix: PosePath(np.array([0., 1.]), np.stack((matrix, matrix)), 'current measured/model pose')
    query = FrozenPrimitives(primitives=[{'kind': 'cuboid', 'dimensions': [.004]*3,
        'pose': [0., 0., 0., 1., 0., 0., 0.],
        'native_binding': {'geometry_name': 'payload_box', 'sphere_indices': [0],
                           'subdivisions': [1, 1, 1], 'source': 'complete native box'}}],
        owner_ids=('payload',), source='complete original rigid box')
    actor = CollisionOwner('payload', query, path(pose), 'current attached object', True,
        local_spheres=[[0., 0., 0., .004]], sphere_materials=(None,),
        attachments=(Attachment((0., 1.), [0., 0., 0.], [1., 0., 0., 0.], 'holding', None),))
    geometry = point_source(((0., 0., 0.),)) if geometry is None else geometry
    target = CollisionOwner('support', FrozenESDF.from_source(geometry, origin=[-.025]*3,
        shape=(51,)*3, voxel_size=.001, owner_ids=('support',), source='current plane observation'),
        path(np.eye(4)), 'independent support plane mask', False)
    return actor, target


def poses(position=(0., 0., .001), quaternion=(1., 0., 0., 0.), *, requires_grad=False):
    return (torch.tensor([[position]], dtype=torch.float64, requires_grad=requires_grad),
            torch.tensor([[quaternion]], dtype=torch.float64))


def test_source_selection_accepts_touching_volume_and_keeps_noninitial_material():
    # A perfect zero-height TSDF plane still contains negative solid pieces.
    # Those are valid current support; they need not fit a zero-thickness band.
    values = np.broadcast_to(np.array([-.001, 0., .001], np.float32), (3, 3, 3))
    geometry = source_from_samples(values, origin=[-.001]*3, voxel_size=.001)
    actor, target = owners(geometry, entrance_z=.002)
    model = PreparedInitialSupport(relation(), actor, target, 1e-6)
    assert model.kind == 'empty'
    assert len(model.selected_component_ids) == geometry.component_count
    assert model.baseline_height == 0.
    # The independently observed plane has other material outside the entrance
    # contact. Moving toward that material cannot expand the frozen inventory.
    mixed = surface_source([[[0., 0., 0.]], [[.006, 0., -.001]]])
    actor, target = owners(mixed)
    model = PreparedInitialSupport(relation(), actor, target, 1e-6)
    np.testing.assert_array_equal(model.selected_component_ids, [0])
    assert model.kind == 'field'
    body = torch.tensor([[[[.006, 0., -.001, .0005]]]], dtype=torch.float64)
    assert model.collision_values(body, target.query(body), torch.ones((1, 1), dtype=torch.bool)).item() > 0


def test_observed_support_component_need_not_fit_future_rectangle_or_ideal_plane():
    # An observed source atom can extend beyond a future functional footprint
    # and across the fitted plane. Its entrance intersection, not either ideal
    # boundary, defines this current support's finite material inventory.
    geometry = surface_source([[[0., 0., 0.], [.004, 0., .0002]]])
    actor, target = owners(geometry)
    model = PreparedInitialSupport(relation(), actor, target, 1e-6)
    assert model.kind == 'empty'
    np.testing.assert_array_equal(model.selected_component_ids, [0])


def test_relative_whole_body_height_rejects_deeper_but_has_no_departure_or_pose_lock():
    actor, target = owners()
    model = PreparedInitialSupport(relation(), actor, target, 1e-6)
    assert model.initial_height == pytest.approx(-.001)
    assert model.baseline_height == pytest.approx(-.001)
    origin = poses((0., 0., 0.))
    for position in ((0., 0., .001), (.001, 0., .001), (0., 0., .02), (0., 0., .001)):
        active, residual = model.components(poses(position), origin, torch.tensor([[.5]], dtype=torch.float64))
        assert active.item() and residual.item() == 0
    # Rotate around the support normal: no exact demonstrated orientation is required.
    _, residual = model.components(poses(quaternion=(0., 0., 0., 1.)), origin,
                                   torch.tensor([[.5]], dtype=torch.float64))
    assert residual.item() == 0
    body = poses((0., 0., -.001), requires_grad=True)
    _, residual = model.components(body, origin, torch.tensor([[.5]], dtype=torch.float64))
    assert residual.item() == pytest.approx(.002-1e-6)
    assert torch.autograd.grad(residual.sum(), body[0])[0][0, 0, 2].item() == -1.
    inactive, residual = model.components(body, origin, torch.tensor([[1.]], dtype=torch.float64))
    assert not inactive.item() and residual.item() == 0


def test_initial_support_needs_independent_plane_evidence_and_existing_attachment():
    with pytest.raises(ValueError, match='independent observed-plane'):
        relation(source={'kind': 'arbitrary_owner'})
    actor, target = owners()
    actor.attachments = ()
    with pytest.raises(ValueError, match='existing entrance attachment'):
        PreparedInitialSupport(relation(), actor, target, 1e-6)


def test_frozen_support_restores_exact_inventory_without_rebuilding_field_and_rejects_fake_empty():
    actor, target = owners(point_source(((0., 0., 0.), (.006, 0., 0.))))
    model = PreparedInitialSupport(relation(), actor, target, 1e-6)
    record = json.loads(json.dumps(model.record()))
    with patch.object(target.query.source_geometry, 'distance_field', side_effect=AssertionError('no remapping')):
        restored = PreparedInitialSupport(relation(), actor, target, 1e-6, record=record)
    np.testing.assert_array_equal(restored.query.distance, model.query.distance)
    np.testing.assert_array_equal(restored.selected_component_ids, model.selected_component_ids)
    altered = deepcopy(record)
    altered['kind'] = 'empty'
    altered.pop('distance')
    with pytest.raises(ValueError, match='remaining source material'):
        PreparedInitialSupport(relation(), actor, target, 1e-6, record=altered)


def scene():
    actor, target = owners()
    background = FrozenESDF.from_source(point_source(), origin=[-.1]*3, shape=(5,)*3,
        voxel_size=.05, owner_ids=('background',), source='empty background')
    result = OnlinePlanningScene(background=background, owners=(actor, target),
        coverage=ObservedSpace(np.ones((5,)*3, bool), origin=[-.1]*3, voxel_size=.05,
                               observation_ids=('camera',), reference_frame='world'),
        reference_frame='world', robot_sphere_materials=(None,), robot_geometry_source='test robot',
        has_gripper=False, unknown_policy={'conditional_regions': [], 'outside_regions': 'forbid'},
        initial_support_relations=(relation(),))
    result.bind_online_robot(robot_sphere_geometry=[{'link_name': 'tool', 'sphere_indices': [0],
        'centers_local': [[0., 0., 0.]], 'radii': [.001]}], robot_self_collision_pairs=[])
    return result


def test_scene_support_is_actor_scoped_and_roundtrip_keeps_the_same_components():
    original = scene()
    body = torch.tensor([[[[0., 0., 0., .001]]]], dtype=torch.float64)
    position, quaternion = poses()
    progress = torch.tensor([[.5]], dtype=torch.float64)
    components = original.components(body, position, quaternion, progress)
    assert components['collision/payload/support'].item() == 0
    assert components['collision/robot/support'].item() > 0
    assert components['initial_support/depth/entry_support'].item() == 0
    restored = OnlinePlanningScene.from_binding(json.loads(json.dumps(original.to_execution_binding())))
    actual = restored.components(body, position, quaternion, progress)
    assert components.keys() == actual.keys()
    assert all(torch.equal(value, actual[key]) for key, value in components.items())
    assert original.estimated_workspace_allocation(1, 2)['cpu_shared'] > 0


def test_source_exclusion_never_accepts_foreign_or_fractional_component_identity():
    geometry = point_source(((0., 0., 0.),))
    for invalid in ([1], [-1], [.5], [0, 0]):
        with pytest.raises(ValueError, match='distinct current source identities'):
            geometry.distance_field(origin=[-.01]*3, shape=(21,)*3, voxel_size=.001,
                                    excluded_component_ids=invalid)
# END no simulation, perception model, package installation or GPU is needed.
