"""Event-informed reference initialization from the registered robot model.

The input model is the same fixed-base scalar-joint Pinocchio model used by
the planner. Its placements, motion subspaces, calibration and tool transform
define the CasADi expressions. No robot-specific inverse-kinematics formulas
or task names enter the computation. This module produces numerical samples;
the existing affine curve retains ownership of measured q0 and dq0.
"""

# BEGIN one model-derived FK expression and bounded whole-reference initializer.
from time import perf_counter

import casadi as ca
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _quaternion_product(first, second):
    """Compose wxyz unit quaternions without choosing a new matrix sign."""
    return ca.vertcat(first[0]*second[0]-ca.dot(first[1:], second[1:]),
        first[0]*second[1:]+second[0]*first[1:]+ca.cross(first[1:], second[1:]))


def _fixed_quaternion(transform):
    return ca.DM(Rotation.from_matrix(np.asarray(transform)[:3, :3]).as_quat()[[3, 0, 1, 2]])


class SymbolicKinematics:
    """Compile one named tool's FK and first derivatives from Pinocchio data.

    ``fixed_positions`` names the complete model's scalar joint coordinates.
    ``joint_names`` identifies the optimized subset. Joint motions use the
    model's six-dimensional motion subspace at its neutral configuration;
    revolute, prismatic and scalar screw joints share this exponential map.
    The model is read in place. Official geometry and its fixed relationships
    remain reusable because the installed robot configuration is unchanged.
    """

    def __init__(self, model, *, joint_names, fixed_positions, frame,
                 reference_from_base, frame_from_tool):
        self.joint_names = tuple(joint_names)
        q = ca.SX.sym('joint_position', len(self.joint_names))
        coordinates = {**fixed_positions, **dict(zip(self.joint_names, ca.vertsplit(q)))}
        neutral = pin.neutral(model)
        target = model.frames[model.getFrameId(frame)]
        ancestors, parent = set(), int(target.parentJoint)
        while parent:
            ancestors.add(parent)
            parent = int(model.parents[parent])
        poses = {0: ca.DM(reference_from_base)}
        lifts = {0: _fixed_quaternion(reference_from_base)}
        for index in range(1, model.njoints):
            if index not in ancestors:
                continue
            joint = model.joints[index]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError('The registered FK interface requires scalar joints')
            data = joint.createData()
            joint.calc(data, neutral)
            twist = np.asarray(data.S).reshape(6)
            linear, angular = twist[:3], twist[3:]
            coordinate = coordinates[str(model.names[index])] - neutral[joint.idx_q]
            speed = float(np.linalg.norm(angular))
            motion = ca.SX.eye(4)
            if speed == 0.:
                motion[:3, 3] = ca.DM(linear) * coordinate
                motion_lift = ca.DM([1., 0., 0., 0.])
            else:
                axis = angular / speed
                cross = np.array([[0., -axis[2], axis[1]],
                                  [axis[2], 0., -axis[0]],
                                  [-axis[1], axis[0], 0.]])
                angle = speed * coordinate
                rotation = ca.DM.eye(3) + ca.sin(angle)*cross + (1-ca.cos(angle))*(cross@cross)
                motion[:3, :3] = rotation
                motion[:3, 3] = ((ca.DM.eye(3)-rotation) @ np.cross(angular, linear)
                    + ca.DM(angular*float(angular@linear))*coordinate) / speed**2
                motion_lift = ca.vertcat(ca.cos(angle/2), ca.DM(axis)*ca.sin(angle/2))
            poses[index] = (poses[int(model.parents[index])] @ model.jointPlacements[index].homogeneous
                            @ data.M.homogeneous @ motion)
            lifts[index] = _quaternion_product(_quaternion_product(
                _quaternion_product(lifts[int(model.parents[index])],
                    _fixed_quaternion(model.jointPlacements[index].homogeneous)),
                _fixed_quaternion(data.M.homogeneous)), motion_lift)
        tool = poses[int(target.parentJoint)] @ target.placement.homogeneous @ ca.DM(frame_from_tool)
        flat = ca.vertcat(tool[:3, 3], ca.vec(tool[:3, :3]))
        self.pose = ca.Function('registered_tool_pose', [q], [tool])
        self.flat = ca.Function('registered_tool_coordinates', [q], [flat])
        self.jacobian = ca.Function('registered_tool_jacobian', [q], [ca.jacobian(flat, q)])
        # The continuous quaternion product distinguishes joint paths that
        # reach the same rotation matrix after different full revolutions.
        lift = _quaternion_product(_quaternion_product(lifts[int(target.parentJoint)],
            _fixed_quaternion(target.placement.homogeneous)), _fixed_quaternion(frame_from_tool))
        coordinates_lift = ca.vertcat(tool[:3, 3], lift)
        self.orientation_lift = ca.Function('registered_orientation_lift', [q], [lift])
        self.lift_coordinates = ca.Function('registered_lift_coordinates', [q], [coordinates_lift])
        self.lift_jacobian = ca.Function('registered_lift_jacobian', [q], [ca.jacobian(coordinates_lift, q)])

    def sample(self, positions):
        """Return [sample,4,4] poses in the calibrated reference frame."""
        positions = np.asarray(positions, float).reshape(-1, len(self.joint_names))
        packed = np.asarray(self.pose.map(len(positions))(positions.T))
        return np.stack(np.hsplit(packed, len(positions)))


def bounded_pose_fit(kinematics, pose, seed, bounds, *, position_scale,
                     rotation_scale, maximum_evaluations):
    """Return an ordinary bounded numerical proposal and its actual error.

    The matrix residual supplies a smooth local least-squares initializer.
    Reported orientation error uses the principal rotation angle. Main
    trajectory objectives retain their own declared rotation metric.
    """
    target = np.r_[pose[:3, 3], pose[:3, :3].ravel(order='F')]
    scale = np.r_[np.full(3, 1/position_scale),
                  np.full(9, 1/(np.sqrt(2)*rotation_scale))]
    seed = np.clip(seed, bounds[0], bounds[1])
    fit = least_squares(lambda value: (np.asarray(kinematics.flat(value)).ravel()-target)*scale,
        seed, jac=lambda value: np.asarray(kinematics.jacobian(value))*scale[:, None],
        bounds=(bounds[0], bounds[1]), max_nfev=maximum_evaluations,
        ftol=1e-8, xtol=1e-8, gtol=1e-8)
    actual = np.asarray(kinematics.pose(fit.x))
    return fit.x, dict(position_m=float(np.linalg.norm(actual[:3, 3]-pose[:3, 3])),
        rotation_rad=float(Rotation.from_matrix(pose[:3, :3].T@actual[:3, :3]).magnitude()),
        cost=float(fit.cost), evaluations=int(fit.nfev), status=int(fit.status))


# BEGIN smooth measured-entry connections replace backward branch splicing.
def event_reference_samples(kinematics, *, times, reference, start_q, start_dq,
        position_bounds, velocity_bounds, event_times, position_scale,
        rotation_scale, random_seed, proposal_count, maximum_evaluations,
        check_request=lambda: None):
    """Propose bounded terminal states and connect from the measured q/dq.

    Every terminal proposal is obtained from the same final reference pose.
    A quintic connection retains entrance velocity and reaches zero terminal
    velocity and acceleration at the first declared positive event time.
    Its normalized joint travel is retained as a descriptive quantity;
    the main fixed numerical problem ranks every fitted connection.
    All candidate Cartesian paths and fit errors remain in the report.
    """
    started = perf_counter()
    times, targets = np.asarray(times, float), np.asarray(reference, float)
    start_q, start_dq = np.asarray(start_q), np.asarray(start_dq)
    bounds = np.asarray(position_bounds)
    speeds = np.maximum(np.abs(velocity_bounds[0]), np.abs(velocity_bounds[1]))
    seeds = [start_q, bounds.mean(0)]
    generator = np.random.default_rng(random_seed)
    seeds.extend(generator.uniform(bounds[0], bounds[1]) for _ in range(max(0, proposal_count-2)))
    candidates, records = [], []
    duration = float(min([value for value in event_times if value > 0.] or [times[-1]]))
    u = np.clip(times/duration, 0., 1.)
    blend, entrance_rate = u**3*(10+u*(-15+6*u)), u-6*u**3+8*u**4-3*u**5
    delta_t = np.diff(times)
    weights = (np.r_[delta_t, 0.]+np.r_[0., delta_t])/(2*(times[-1]-times[0]))
    for seed in seeds[:proposal_count]:
        check_request()
        terminal, fit = bounded_pose_fit(kinematics, targets[-1], seed, bounds,
            position_scale=position_scale, rotation_scale=rotation_scale,
            maximum_evaluations=maximum_evaluations)
        values = (start_q+blend[:, None]*(terminal-start_q)
                  +duration*entrance_rate[:, None]*start_dq)
        actual = kinematics.sample(values)
        position_error = np.linalg.norm(actual[:, :3, 3]-targets[:, :3, 3], axis=1)
        rotation_error = Rotation.from_matrix(np.einsum('nji,njk->nik',
            targets[:, :3, :3], actual[:, :3, :3])).magnitude()
        increments = Rotation.from_matrix(np.einsum('nji,njk->nik',
            actual[:-1, :3, :3], actual[1:, :3, :3])).magnitude()
        travel = float(np.sum(((terminal-start_q)/speeds)**2))
        records.append(dict(**fit, joint_position=terminal.tolist(),
            normalized_joint_travel=travel, connection_duration_s=duration,
            whole_reference=dict(path_length_m=float(np.linalg.norm(np.diff(actual[:, :3, 3], axis=0), axis=1).sum()),
                rotation_travel_rad=float(increments.sum()),
                position_rms_m=float(np.sqrt(weights @ position_error**2)),
                rotation_rms_rad=float(np.sqrt(weights @ rotation_error**2)))))
        candidates.append(values)
    return np.stack(candidates), dict(strategy='bounded_terminal_proposals_smooth_measured_connection',
        proposals=records, elapsed_seconds=perf_counter()-started,
        scope='All numerical proposals from fixed measured q/dq; the original complete problem selects an initial curve')
# END original reference and robot feedback remain inputs, not fitted replacements.
