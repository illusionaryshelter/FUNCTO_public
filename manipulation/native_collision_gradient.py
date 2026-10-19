"""First derivatives of Coal body/target penetration in the existing state.

Coal selects the active component and its witnesses. Pinocchio differentiates
robot material points; Torch differentiates owner poses and attachments. The
forward scalar is the native penetration, never an ESDF or sphere distance.
"""

import numpy as np
import torch

from .planning_problem import quaternion_rotate


def confirmed_world_costs(record, robot_state, evidence, owner_matrices, validator,
                          batch, index, original):
    """Return one scalar per completely resolved actor/target at one state.

    A native numerical boundary or missing result remains an explicit pending
    confirmation; the caller retains its positive hard proposal. A resolved
    collision sums each body's maximum component penetration exactly once.
    For each selected feature p2-p1=d*n, so dc=n·(dp1-dp2) for c=-d>0.
    Detaching witness coordinates is the envelope derivative, not a detached
    collision objective. Feature switches may be nonsmooth.
    """
    actor = record['actor']
    rows = record['pairs']
    unresolved = set(record['unresolved_targets'])
    active = [row for row in rows if row['target_owner'] not in unresolved
              and row['status'] == 'collision']
    reference = next(iter(original.values()))
    joints = evidence['joint_position']
    use_gradient = torch.is_grad_enabled() and (
        (joints is not None and joints.requires_grad)
        or any(p.requires_grad or q.requires_grad for p, q in evidence['owner_poses'].values()))
    robot_jacobians, named_tensors = {}, None
    if use_gradient and actor == 'robot' and active:
        # The confirmer already resolved the complete named q, including the
        # hand's actual/nominal state at this time. Reuse that exact mapping.
        named = {name: joints[batch, index, column]
                 for column, name in enumerate(evidence['joint_names'])}
        if evidence['finger_positions'] is not None:
            named.update({name: evidence['finger_positions'][batch, index, column]
                          for column, name in enumerate(validator.finger_joint_names)})
        named_tensors = torch.stack([named[name] for name in robot_state['joint_names']])
        points = []
        for row in active:
            target = owner_matrices.get(row['target_owner'], np.eye(4))
            point = target[:3, :3] @ np.asarray(row['witness_native'])+target[:3, 3]
            points.append((row['actor_geometry'], point))
        jacobians = validator.checker.geometry_point_jacobians(
            robot_state, points, validator.T_reference_base)
        robot_jacobians = {id(row): jacobian for row, jacobian in zip(active, jacobians)}

    result = {name: reference.new_zeros(()) for name in
              (*record['cleared_targets'], *record['collision_targets']) if name not in unresolved}
    for row in active:
        target_name = row['target_owner']
        value = reference.new_tensor(-row['minimum_queried_distance_m'])
        if use_gradient:
            target = owner_matrices.get(target_name, np.eye(4))
            normal = reference.new_tensor(target[:3, :3] @ np.asarray(row['normal_actor_to_target']))
            native_point = target[:3, :3] @ np.asarray(row['witness_native'])+target[:3, 3]
            if actor == 'robot':
                jacobian = reference.new_tensor(robot_jacobians[id(row)])
                actor_point = jacobian @ (named_tensors-named_tensors.detach())
            else:
                owner = owner_matrices[actor]
                local = owner[:3, :3].T @ (native_point-owner[:3, 3])
                p, q = evidence['owner_poses'][actor]
                actor_point = p[batch, index]+quaternion_rotate(
                    q[batch, index], reference.new_tensor(local))
            if target_name == 'background':
                target_point = reference.new_tensor(row['witness_target'])
            else:
                p, q = evidence['owner_poses'][target_name]
                target_point = p[batch, index]+quaternion_rotate(
                    q[batch, index], reference.new_tensor(row['witness_target']))
            linear = torch.dot(normal, actor_point-target_point)
            # Preserve Coal's forward value bit-for-bit while supplying its
            # local first derivative. No pose or sphere residual is substituted.
            value = value+(linear-linear.detach())
        result[target_name] = result[target_name]+value
    return result
