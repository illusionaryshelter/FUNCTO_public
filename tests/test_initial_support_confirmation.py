"""Native confirmation consumes the same finite initial-support inventory."""

# BEGIN independent confirmation regressions use existing native box fixtures.
from dataclasses import replace

import numpy as np
import pytest
import torch

from manipulation.initial_support import InitialSupportRelation
from manipulation.online_geometry import OnlinePlanningScene
from manipulation.online_native_confirmation import NativeCollisionConfirmation
from manipulation.planning_scene import Attachment
from test_online_native_confirmation import _adapter, _candidate_evidence, _field, _payload, _setup, _state


def support_scene(tmp_path, *, additional_points=(), other_target=False):
    """Bind a small complete held box to an observed plane at its current pose.

    The first zero-source point lies inside the entrance box and below the
    declared support plane. Additional points explicitly represent geometry
    that may share this owner but must not inherit that initial permission.
    """
    checker, original = _setup(tmp_path, query=_field(points=((0., 0., -.0002), *additional_points)))
    payload = _payload(original)
    payload = replace(payload, attachments=(Attachment((0., 1.), np.zeros(3),
        np.array([1., 0., 0., 0.]), 'current-holding', 'release'),))
    owners = (*original.owners[:-1], payload)
    if other_target:
        query = _field(points=((0., 0., -.0002),))
        query.owner_ids = frozenset(('other',))
        owners += (replace(owners[0], owner_id='other', query=query),)
    relation = InitialSupportRelation(relation_id='current-support', actor='payload',
        target_owner='part', interval=(0., 1.), patch_origin=np.zeros(3),
        patch_axes=np.eye(3), patch_half_size=[.005, .005], source={
            'kind': 'observed_plane_with_declared_initial_support', 'observation_ids': ['test'],
            'assumption': 'Synthetic current box rests on this observed plane',
            'scope': 'Only finite entrance-intersecting material on the support side'})
    scene = OnlinePlanningScene(background=original.background, owners=owners, coverage=original.coverage,
        reference_frame=original.reference_frame, robot_sphere_materials=original.robot_sphere_materials,
        robot_geometry_source=original.robot_geometry_source, robot_collision_bodies=original.robot_collision_bodies,
        has_gripper=False, unknown_policy=original.unknown_policy, initial_support_relations=(relation,))
    return checker, scene


@pytest.mark.parametrize('active,actor,target,points,x,cleared', [
    (True, 'payload', 'part', (), 0., True),
    (False, 'payload', 'part', (), 0., False),
    (True, 'robot', 'part', (), 0., False),
    (True, 'payload', 'other', (), 0., False),
    (True, 'payload', 'part', ((0., 0., .0002),), 0., False),
    (True, 'payload', 'part', ((.003, 0., -.0002),), .003, False),
])
def test_native_support_is_scoped_to_same_actor_target_and_frozen_source_inventory(
        tmp_path, active, actor, target, points, x, cleared):
    checker, scene = support_scene(tmp_path, additional_points=points, other_target=target == 'other')
    model = scene._initial_support_models['current-support']
    np.testing.assert_array_equal(model.selected_component_ids, [0])
    poses = {owner.owner_id: np.eye(4) for owner in scene.owners}
    poses['payload'][0, 3] = x
    report = NativeCollisionConfirmation(checker, scene).confirm(
        robot_state=_state(checker) if actor == 'robot' else None,
        candidates={target: [0]}, owner_poses=poses, actor=actor, effective_contact_rule_ids=(),
        effective_initial_support_ids=('current-support',) if active else ())
    assert (target in report['cleared_targets']) is cleared
    if not cleared:
        assert target in report['collision_targets'] or target in report['unresolved_targets']
    expected = ['current-support'] if active and actor == 'payload' and target == 'part' else []
    assert report['pairs'][0]['effective_initial_support_ids'] == expected


def test_native_support_rejects_unknown_relation_and_preserves_height_residual(tmp_path):
    checker, scene = support_scene(tmp_path)
    with pytest.raises(ValueError, match='registered scene relations'):
        NativeCollisionConfirmation(checker, scene).confirm(robot_state=None,
            candidates={'part': [0]}, owner_poses={'part': np.eye(4), 'payload': np.eye(4)},
            actor='payload', effective_contact_rule_ids=(), effective_initial_support_ids=('unknown',))
    evidence = _candidate_evidence([[[.01]]])
    evidence['targets'] = {'collision/payload/part': evidence['targets']['collision/robot/part']}
    evidence['owner_poses']['payload'] = evidence['owner_poses']['part']
    evidence['effective_initial_supports'] = {'current-support': torch.ones((1, 1), dtype=torch.bool)}
    depth = torch.tensor([[.002]], dtype=torch.float64)
    costs = {'collision/payload/part': torch.tensor([[.01]], dtype=torch.float64),
        'initial_support/depth/current-support': depth}
    result = _adapter(checker, scene)(costs, evidence)
    assert result['collision/payload/part'].item() == 0.
    assert result['initial_support/depth/current-support'] is depth
    assert depth.item() == .002
# END native clears never change relative-height or unrelated scene constraints.
