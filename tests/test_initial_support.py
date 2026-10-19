"""Current support permits one entrance subset without a depth proof."""
# BEGIN generic owner/support identities exercise the one-query model contract.
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manipulation.initial_support import InitialSupportRelation, PreparedInitialSupport
from manipulation.online_geometry import FrozenESDF, FrozenPrimitives, OnlinePlanningScene
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, CollisionOwner
from test_online_geometry import scene as empty_scene
from native_scene_fixtures import bind_fixed_robot


def setup():
    initial = np.eye(4)
    initial[2, 3] = .03
    payload_query = FrozenPrimitives(primitives=[{'kind': 'box', 'dimensions': [.02, .02, .1]}],
        owner_ids=('payload',), source='explicit bounded payload')
    actor = CollisionOwner('payload', payload_query,
        PosePath(np.array([0., 1.]), np.stack([initial]*2), 'current payload'), 'current payload',
        geometry_complete=True, local_spheres=[[0., 0., -.02, .02], [0., 0., .06, .02]],
        sphere_materials=('lower', 'upper'), attachments=(Attachment((0., 1.), [0., 0., .03],
            [1., 0., 0., 0.], 'held', None),))
    # BEGIN current support is the observed field consumed in production.
    values = np.broadcast_to(np.linspace(-.2, .2, 41, dtype=np.float32)[None, None, :], (41, 41, 41))
    query = FrozenESDF(values, origin=[-.2]*3, voxel_size=.01,
        owner_ids=('support',), source='observed affine z support field')
    # END the entrance sphere subset does not authorize a complete solid pair.
    target = CollisionOwner('support', query, PosePath(np.array([0., 1.]), np.stack([np.eye(4)]*2),
        'current support'), 'current support', geometry_complete=True)
    relation = InitialSupportRelation('support-entry', 'payload', 'support', (0., 1.), [0., 0., 0.],
        [0., 0., 1.], {'kind': 'observed_plane_with_declared_initial_support', 'observation_ids': ['rgbd'],
        'assumption': 'current payload starts on this support', 'scope': 'this payload/support pair'})
    return actor, target, relation


def test_only_entrance_contact_spheres_are_permitted_and_interval_expires():
    actor, target, relation = setup()
    support = PreparedInitialSupport(relation, actor, target)
    assert support.sphere_indices == (0,)
    ordinary = torch.tensor([[[.1, .2], [.1, .2]]], dtype=torch.float64, requires_grad=True)
    values = support.collision_values(ordinary, torch.tensor([[.5, 1.]], dtype=torch.float64))
    torch.testing.assert_close(values, torch.tensor([[[0., .2], [.1, .2]]], dtype=torch.float64))
    values.sum().backward()
    torch.testing.assert_close(ordinary.grad, torch.tensor([[[0., 1.], [1., 1.]]], dtype=torch.float64))
    restored = PreparedInitialSupport(relation, actor, target, record=support.record())
    assert restored.sphere_indices == support.sphere_indices


def test_no_source_component_or_depth_certificate_is_constructed():
    actor, target, relation = setup()
    with patch.object(FrozenESDF, '__call__', wraps=target.query.__call__) as query:
        support = PreparedInitialSupport(relation, actor, target)
        assert query.call_count == 1
    assert not hasattr(support, 'actor_vertices')
    assert not hasattr(support, 'selected_component_ids')
    assert set(support.record()) == {'model', 'relation', 'sphere_indices'}


def test_support_never_clears_upper_body_other_owner_or_background():
    actor, target, relation = setup()
    original = empty_scene()
    obstacle = deepcopy(target)
    obstacle.owner_id = 'other'
    obstacle.query.owner_ids = frozenset(('other',))
    current = OnlinePlanningScene(background=original.background, owners=(actor, target, obstacle),
        coverage=original.coverage, reference_frame='world', robot_sphere_materials=('arm',),
        robot_geometry_source='robot', has_gripper=False, initial_support_relations=(relation,))
    bind_fixed_robot(current, centers=[[.3, 0., .3]])
    spheres = np.array([[.3, 0., .3, .01]])
    entrance = current.query_state(spheres, np.eye(4), .5)
    assert entrance['components']['collision/payload/support'] == 0
    assert entrance['components']['collision/payload/other'] > 0
    pose = np.eye(4)
    pose[2, 3] = -.09
    lowered = current.query_state(spheres, pose, .5)
    assert lowered['components']['collision/payload/support'] > 0
    assert not any(name.startswith('initial_support/depth') for name in lowered['components'])


def test_initial_support_does_not_silently_exempt_a_complete_native_target():
    actor, target, relation = setup()
    original = empty_scene()
    target.query = FrozenPrimitives(primitives=[{'kind': 'cuboid', 'dimensions': [.4, .4, .1],
        'pose': [0., 0., -.05, 1., 0., 0., 0.]}], owner_ids=('support',), source='complete solid')
    with pytest.raises(ValueError, match='support-entry.*observed ESDF target'):
        OnlinePlanningScene(background=original.background, owners=(actor, target), coverage=original.coverage,
            reference_frame='world', robot_sphere_materials=('arm',), robot_geometry_source='robot',
            has_gripper=False, initial_support_relations=(relation,))
# END ordinary current support does not become a global collision exemption.
