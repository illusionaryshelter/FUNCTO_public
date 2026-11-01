"""Declared scene relationship preferences on the existing physical queries.

The scene owns observed material, contact declarations, attachment/release
poses and prepared initial-support models. This module composes their existing
differentiable APIs into the original state-count mean. GeometryConstraints
binds the shared MaterialSamples inventory before a candidate is scored.
"""

# BEGIN declared relationship costs share the original scene APIs and material.
import numpy as np
import torch

from manipulation.sampled_geometry_objective import contact_objective


def scene_relationship_objective(scene, position, quaternion, progress,
        joint_position, joint_names, *, trajectory_time, material_samples,
        report_out=None):
    """Return the original required-contact and initial-support scalar sum.

    Poses use the scene reference frame, metres and unit wxyz quaternions.
    ``progress`` and ``trajectory_time`` retain the caller's complete ordinary
    and exact-event query layout. The original permission API supplies active
    and hand-certainty masks; the original owner API supplies first-release
    anchoring. Each retained term is normalized by the number of query states,
    exactly as in OnlinePlanningScene.optimization_objective. The common
    trajectory objective applies constraint_weight once to the returned value.
    """
    required = tuple(rule for rule in scene.contacts if rule.required)
    models = tuple(scene._initial_support_models.values())
    zero = position.sum() * 0 + quaternion.sum() * 0 + joint_position.sum() * 0
    report = dict(scope='declared_required_contact_and_initial_support',
        normalization='sum_over_all_query_states_divided_by_state_count',
        state_count=progress.numel(), required_contact_count=len(required),
        initial_support_count=len(models), active_contact_state_count=0,
        conditional_contact_state_count=0, contact_without_zero_samples=[],
        relationship_components={})
    values = []
    if required or models:
        relationship_components = {}
        poses = scene._owner_poses(position, quaternion, progress, None,
                                   relationship_components)
        identity = torch.zeros_like(quaternion)
        identity[..., 0] = 1
        poses['background'] = torch.zeros_like(position), identity
        hand = (None if scene.gripper is None or not required else
                scene.gripper.state(progress, joint_position, joint_names,
                    trajectory_time=trajectory_time, measured=False))
        coordinates = None if hand is None else hand.coordinates
        uncertain = None if hand is None else hand.uncertain
        for rule in required:
            actor = (position, quaternion) if rule.actor == 'robot' else poses[rule.actor]
            policy, normal, conditional = scene._permission(rule, actor,
                poses[rule.target_owner], progress, coordinates, uncertain_hand=uncertain)
            samples = material_samples['field', rule.target_owner]
            if not len(samples.zero_points):
                report['contact_without_zero_samples'].append(rule.rule_id)
            report['conditional_contact_state_count'] += int(torch.count_nonzero(conditional))
            for index in np.ndindex(tuple(progress.shape)):
                if bool(policy['active'][index]) and not bool(conditional[index]):
                    if scene.check_request is not None:
                        scene.check_request()
                    values.append(normal[index].square() + contact_objective(samples,
                        policy['normals'][index], policy['bounds'][index]))
                    report['active_contact_state_count'] += 1
        for model in models:
            components = model.components(poses[model.relation.actor],
                poses[model.relation.target_owner], progress,
                check_request=scene.check_request)
            values.append((components[f'initial_support/{model.relation.relation_id}']
                           / model.actor_field.voxel_size).square().sum())
        report['relationship_components'] = {key: value.detach().cpu().tolist()
                                              for key, value in relationship_components.items()}
    result = zero + (torch.stack(values).sum() / progress.numel() if values else zero)
    if report_out is not None:
        report_out.update(report)
    return result
# END complete collision and relation qualification remains in the scene exit.
