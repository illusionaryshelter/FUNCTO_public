"""Original task rows with local derivatives for finite support halfspaces.

PlanningProblem.constraint_components remains the value and row-layout owner.
Only its declared body/footprint blocks have an explicit analytic derivative;
all other named conditions retain their original Torch graph. The caller owns
the same-point graph lifetime, fixed solver-row selection and geometry rows.

The fixed official robot and installed end-effector keep their model across
tasks. This module uses the existing registered Pinocchio model, calibration,
named robot states and same-call frame Jacobians. Task material points and
support halfspaces are consumed from the existing requirement declaration.
"""

from dataclasses import dataclass

import numpy as np
import torch


from manipulation.planning_problem import (
    _compose, _first_event_query, _quaternion, quaternion_conjugate,
    quaternion_multiply, quaternion_rotate,
)


# BEGIN equivalent smooth event balls retain the original explicit tolerances.
def normalized_pose_event(requirement, position, quaternion):
    """Return dimensionless inequalities for the original pose-event set.

    Position uses the squared metric ball. Full rotation uses the squared
    quaternion-vector norm, equal to sin(theta/2)^2. With an explicitly
    declared axis, the scalar and axis projection form the twist quaternion;
    its orthogonal vector determines swing. These identities preserve the
    original position, swing and finite principal-twist tolerance sets.
    The original signed zero-yaw equality remains outside this transformation.
    """
    p, q = _compose(position, quaternion, requirement.tool_from_function)
    target = q.new_tensor(_quaternion(requirement.target))
    relative = quaternion_multiply(quaternion_conjugate(target), q)
    norm2 = relative.square().sum(-1)
    vector2 = relative[..., 1:].square().sum(-1)
    values = {'position': (p-p.new_tensor(requirement.target[:3, 3])).square().sum(-1)
                         /requirement.position_tolerance_m**2-1.}
    if requirement.yaw_axis is None:
        values['rotation'] = vector2/(norm2*np.sin(requirement.rotation_tolerance_rad/2)**2)-1.
    else:
        projection = (relative[..., 1:]*q.new_tensor(requirement.yaw_axis)).sum(-1)
        twist2 = relative[..., 0].square()+projection.square()
        # Compute the perpendicular vector directly, avoiding cancellation
        # when the requested swing is close to zero.
        perpendicular = relative[..., 1:]-projection[..., None]*q.new_tensor(requirement.yaw_axis)
        values['rotation'] = perpendicular.square().sum(-1)/(norm2*np.sin(requirement.rotation_tolerance_rad/2)**2)-1.
        if 0. < requirement.yaw_tolerance_rad < np.pi:
            values['yaw'] = projection.square()/(twist2*np.sin(requirement.yaw_tolerance_rad/2)**2)-1.
    return values
# END acceptance continues to report the source event's metres and radians.


@dataclass(frozen=True)
class SupportLinearization:
    """One signed support block factored through its event joint coordinates.

    ``joint_jacobian[row,joint]`` uses the computation's controlled named-joint
    order and original physical row units. ``coefficient_row[control]`` is the
    original cubic position row multiplied by TerminalHold.matrix. Keeping
    these two arrays avoids a dense control Jacobian during value-only calls.
    """

    joint_jacobian: np.ndarray
    coefficient_row: np.ndarray
    source_rows: np.ndarray

    # BEGIN producer-owned local slots replace full control Jacobian expansion.
    def row_values(self, row, columns):
        """Fill one declared sparse row in [reduced control,named joint] order."""
        joints = self.joint_jacobian.shape[1]
        return (self.joint_jacobian[int(row), columns % joints]
                * self.coefficient_row[columns // joints])
    # END no body-by-all-controls matrix is created by this producer.


def linearized_robot_state(computation, q, dq):
    """Return original robot_tool_states products plus same-call frame J.

    This explicit computation boundary uses the same gripper state, nominal
    finger-trajectory and fixed-tool APIs as unified_solver.robot_tool_states.
    It changes the native state call to states_with_jacobians so that task
    values, remaining AD rows and local support derivatives share one FK.
    """
    problem, scene = computation.problem, computation.scene
    names = tuple(computation.joint_names)
    progress, times = computation.progress, computation.times
    hand = scene.gripper
    if hand is not None:
        missing = [name for name in hand.joint_names if name not in names]
        if missing:
            hand_state = hand.state(progress, q, names, trajectory_time=times)
            coordinates = hand_state.coordinates
            rate = torch.zeros_like(coordinates)
            model = hand.nominal_finger_trajectory
            if model is not None:
                modeled = model.valid_mask_torch(times)
                _, model_rate = model.sample_torch(torch.where(modeled, times, 0.))
                rate = torch.where(modeled[..., None], model_rate, rate)
            columns = [hand.joint_names.index(name) for name in missing]
            q = torch.cat((q, coordinates[..., columns]), -1)
            dq = torch.cat((dq, rate[..., columns]), -1)
            names += tuple(missing)
    poses, velocities, jacobians = computation.kinematics.states_with_jacobians(q, dq, names)
    position, quaternion = poses[problem.tool_link]
    linear, angular = velocities[problem.tool_link]
    tool_position = q.new_tensor(problem.tool_transform[:3, 3])
    tool_quaternion = q.new_tensor(_quaternion(problem.tool_transform))
    offset = quaternion_rotate(quaternion, tool_position.expand_as(position))
    tool = position + offset, quaternion_multiply(quaternion, tool_quaternion)
    twist = linear + torch.linalg.cross(angular, offset, dim=-1), angular
    return (q, dq, names, poses, tool, twist), jacobians


def _support_factors(computation, requirement, robot_state, jacobians, components, material_indices):
    """Pull one original event's rigid material points through frame J and W E."""
    _, _, model_names, poses, tool, _ = robot_state
    mask = _first_event_query(computation.progress, requirement.progress, computation.times)
    # TrajectoryComputation represents one candidate. The same earliest event
    # query used by constraint_components selects its physical state and clock.
    event_column = int(mask[0].to(torch.int64).argmax())
    position = poses[computation.problem.tool_link][0]
    # Compose exactly the producer's normalized quaternions and fixed offsets.
    # The body-point computation is small and retains no derivative graph;
    # native frame poses/J are the same values already used by all task rows.
    with torch.no_grad():
        carrier_p, carrier_q = _compose(tool[0][0, event_column],
            tool[1][0, event_column], requirement.tool_from_carrier)
        material = carrier_p + quaternion_rotate(carrier_q,
            carrier_p.new_tensor(requirement.body_points[material_indices]))
        offset = (material - position[0, event_column]).numpy()
    controlled_columns = [model_names.index(name) for name in computation.joint_names]
    frame = jacobians[computation.problem.tool_link][0, event_column].detach().numpy()
    frame = frame[:, controlled_columns]
    # Every point column is the frame-origin velocity plus omega cross r.
    point_jacobian = frame[:3][None] + np.cross(
        frame[3:].T[None], offset[:, None, :]).transpose(0, 2, 1)
    event_time = float(computation.times[0, event_column])
    curve_row = computation.full_domain._curve_map.query(
        np.array([event_time / computation.duration]), 0)[0][0]
    coefficient_row = curve_row @ computation.terminal.matrix
    result = {}
    for name in components:
        if name == 'body':
            joint_rows = -np.einsum('pcj,c->pj', point_jacobian, requirement.surface.normal)
            source_rows = material_indices.copy()
        else:
            directions = requirement.surface.basis @ requirement.surface._hull.equations[:, :2].T
            joint_rows = np.einsum('pcj,ce->pej', point_jacobian, directions).reshape(
                -1, len(computation.joint_names))
            source_rows = (material_indices[:, None]*directions.shape[1]
                           + np.arange(directions.shape[1])[None]).ravel()
        result[name] = SupportLinearization(joint_rows, coefficient_row, source_rows)
    return result


def task_linearization(computation, controls, *, support_material_indices,
                       solver_rows_out=None, robot_state=None, jacobians=None):
    """Return task rows, normalized event sets and factored analytic support blocks.

    Returns ``((equalities,inequalities),(equality_blocks,inequality_blocks))``.
    Each block dictionary maps the original condition name to ``(slice,factor)``;
    slices address the solver-local rows in their equality/inequality vector.
    A factor's row_values() fills only requested structural columns. The
    explicit support_material_indices were bound once from immutable points.

    The unchanged condition producer creates values and declaration masks.
    Analytic body/footprint values are detached before concatenation, so the
    remaining AD graph contains no reverse traversal of those support blocks.
    Geometry is a separate fixed block in the planner's solver binding.
    """
    # BEGIN a solve point shares the same native states with its full objective.
    if robot_state is None:
        q, dq = computation.states(controls)[:2]
        robot_state, jacobians = linearized_robot_state(computation, q, dq)
    # END standalone value/derivative inspection may produce its own state batch.
    complete_q, complete_dq, names, _, tool, twist = robot_state
    groups = computation.problem.constraint_components(*tool, computation.progress,
        complete_q, names, complete_dq, trajectory_time=computation.times,
        tool_linear_velocity=twist[0], tool_angular_velocity=twist[1],
        solver_rows_out=solver_rows_out)
    # BEGIN transform only the event rows actually declared by the task producer.
    for index, requirement in enumerate(computation.problem.pose_events):
        keys = {name: f'function_pose:{index}:{name}' for name in requirement.component_tolerances}
        if not any(key in groups[1] for key in keys.values()):
            continue
        mask = _first_event_query(computation.progress, requirement.progress, computation.times)
        sample = int(mask[0].to(torch.int64).argmax())
        normalized = normalized_pose_event(requirement, tool[0][:, sample], tool[1][:, sample])
        for name, value in normalized.items():
            if keys[name] in groups[1]:
                groups[1][keys[name]] = value
    # END original masks, equality rows and component identities are unchanged.
    factors = {}
    for index, requirement in enumerate(computation.problem.supports):
        components = tuple(name for name in requirement.hard_components
            if name in {'body', 'footprint'} and f'support:{index}:{name}' in groups[1])
        if components:
            # BEGIN exact halfspace reduction changes only solver-local row indices.
            # Complete source body/contact points and all soft sums remain in the
            # original requirement. Each retained polygon edge remains explicit.
            material_indices = support_material_indices[index]
            local = _support_factors(computation, requirement, robot_state, jacobians,
                                     components, material_indices)
            for name in components:
                key = f'support:{index}:{name}'
                factors[key] = local[name]
                selected = (groups[1][key][..., material_indices] if name == 'body'
                            else groups[1][key][..., material_indices, :])
                groups[1][key] = selected.detach()
                if solver_rows_out is not None:
                    mask = solver_rows_out[1][key]
                    solver_rows_out[1][key] = (mask[..., material_indices] if name == 'body'
                                              else mask[..., material_indices, :])
            # END original row identity is carried by each factor's source_rows.
    vectors, blocks = [], ({}, {})
    for kind, group in enumerate(groups):
        offset, values = 0, []
        for name, value in group.items():
            flat = value.reshape(-1)
            if kind == 1 and name in factors:
                blocks[1][name] = slice(offset, offset + flat.numel()), factors[name]
            values.append(flat)
            offset += flat.numel()
        vectors.append(torch.cat(values) if values else controls.reshape(-1)[:0])
    return tuple(vectors), blocks
