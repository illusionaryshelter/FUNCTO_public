"""Fit numerical coefficient proposals to the complete existing reference.

This initializer uses the registered robot, the measured-entry cubic and its
declared terminal map. A bounded sparse least-squares calculation moves each
terminal connection toward the complete reference. The same main numerical
problem subsequently scores task, geometry and continuous physical values.
"""

# BEGIN complete-reference proposal fitting replaces an endpoint-only seed.
from time import perf_counter

import casadi as ca
import numpy as np
from scipy.optimize import least_squares
import torch

from manipulation.planning_problem import _first_event_query, _quaternion
from manipulation.preference_planner.pose_roles import pose_objective
from .reference_initialization import SymbolicKinematics, _quaternion_product


def _conjugate(q):
    return ca.vertcat(q[0], -q[1:])


def _rotate(q, v):
    cross = ca.cross(q[1:], v)
    return v+2*(q[0]*cross+ca.cross(q[1:], cross))/ca.sumsqr(q)


def _rotation_vector(q):
    """Use the original principal rotation-vector identity limit."""
    q = q/ca.sqrt(ca.sumsqr(q))
    q = ca.if_else(q[0] < 0, -q, q)
    length = ca.sqrt(ca.sumsqr(q[1:])+1e-24)
    return q[1:]*(2*ca.atan2(length, q[0])/length)


def _state_residuals(states, bounds, configuration, native_dt, weights, multiplier):
    """Factor the existing joint STATE cost, including its physical clock."""
    bound_weights = np.asarray(configuration['weight'][:4], float)
    regularization = np.asarray(configuration['squared_l2_regularization_weight'][:3], float)
    if configuration['retime_weights']:
        bound_weights = bound_weights*np.array([1., native_dt, native_dt**2, native_dt**3])
    if configuration['retime_regularization_weights']:
        regularization = regularization*np.array([native_dt, native_dt**2, native_dt**3])
    margin = np.asarray(configuration['activation_distance'][:4])[:, None]*(bounds[:, 1]-bounds[:, 0])
    lower, upper = bounds[:, 0]+margin, bounds[:, 1]-margin
    residuals = []
    for order, state in enumerate(states):
        distance = (ca.fmax(ca.repmat(ca.DM(lower[order]).T, len(weights), 1)-state, 0)
                    +ca.fmax(state-ca.repmat(ca.DM(upper[order]).T, len(weights), 1), 0))
        scale = np.sqrt(.5*multiplier*bound_weights[order]*weights)
        residuals.append(ca.vec(distance*ca.repmat(ca.DM(scale), 1, state.size2())))
        if order:
            scale = np.sqrt(.5*multiplier*regularization[order-1]*weights)
            residuals.append(ca.vec(state*ca.repmat(ca.DM(scale), 1, state.size2())))
    return ca.vertcat(*residuals)


def coefficient_position_box(curve, terminal, bounds):
    """Bound this initializer's complete cubic coefficients in the joint box.

    The local terminal map retains independent prefix coefficients and one
    shared suffix coefficient. Every complete B-spline coefficient therefore
    depends on at most one free row. Intersecting those one-variable intervals
    gives a finite box. The convex-hull property bounds the resulting position
    curve. Main planning keeps its original, larger exact-extremum domain.
    """
    weights = curve.matrix@terminal.matrix
    offset = curve.coefficients(terminal.offset)
    rows, columns = np.nonzero(weights)
    factors = weights[rows, columns, None]
    first, second = (bounds[0]-offset[rows])/factors, (bounds[1]-offset[rows])/factors
    lower = np.full((terminal.matrix.shape[1], len(curve.start_q)), -np.inf)
    upper = np.full_like(lower, np.inf)
    np.maximum.at(lower, columns, np.minimum(first, second))
    np.minimum.at(upper, columns, np.maximum(first, second))
    return lower.ravel(), upper.ravel()


class ReferenceProjection:
    """One sparse residual function shared by all current numerical proposals.

    Reference, angular motion and joint motion retain their actual objective
    definitions. Required pose events supply normalized geometric residuals
    solely for initialization. All other task and scene declarations continue
    to be scored by the original complete problem before main IPOPT.
    """

    def __init__(self, computation, context, *, check_request):
        started = perf_counter()
        self.check_request = check_request
        reference, terminal = computation.reference_objective, computation.terminal
        declaration = reference.declaration
        ordinary = list(map(int, computation.ordinary_indices))
        times, weights = reference.physical_times.numpy(), reference.weights.numpy()
        rows, joints = terminal.matrix.shape[1], len(computation.joint_names)
        self.shape = rows, joints
        self.bounds = coefficient_position_box(computation.full_domain._curve_map,
            terminal, computation.bounds.numpy()[0])
        x = ca.SX.sym('reference_projection_coefficients', rows*joints)
        controls = ca.reshape(x, joints, rows).T
        states = [ca.sparsify(ca.DM(matrix.numpy()@terminal.matrix))@controls
                  +ca.DM(matrix.numpy()@terminal.offset+offset.numpy())
                  for matrix, offset in computation.rows]
        q, _, names, _, _, _ = context['robot_state']
        nominal = q.detach().numpy()[0]
        columns = {name: index for index, name in enumerate(computation.joint_names)}
        complete = ca.horzcat(*[states[0][:, columns[name]] if name in columns
            else ca.DM(nominal[:, index]) for index, name in enumerate(names)])
        model = SymbolicKinematics(computation.kinematics.model, joint_names=names,
            fixed_positions={}, frame=computation.problem.tool_link,
            reference_from_base=computation.problem.base_pose,
            frame_from_tool=computation.problem.tool_transform)
        poses = model.lift_coordinates.map(len(nominal))(complete.T)
        reference_residuals, angular_residuals = [], []
        for local, sample in enumerate(ordinary):
            relative = _quaternion_product(_conjugate(ca.DM(reference.quaternion[local].numpy())),
                                           poses[3:, sample])
            scale = np.sqrt(computation.constraint_weight*declaration.relative_weight*weights[local])
            reference_residuals.extend((scale*(poses[:3, sample]-reference.position[local].numpy())
                /declaration.position_scale_m,
                scale*2*relative[1:]/ca.sqrt(ca.sumsqr(relative))/declaration.rotation_scale_rad))
            if local:
                increment = _quaternion_product(_conjugate(poses[3:, ordinary[local-1]]), poses[3:, sample])
                scale = np.sqrt(declaration.angular_speed_weight_s2
                    /((times[local]-times[local-1])*(times[-1]-times[0])))
                angular_residuals.append(scale*_rotation_vector(increment))
        event_residuals = []
        for event in computation.problem.pose_events:
            if (pose_objective(event)['role'] != 'required' or not
                    computation.problem.start_progress <= event.progress <= computation.problem.end_progress):
                continue
            mask = _first_event_query(computation.progress, event.progress, computation.times)
            sample = int(mask[0].to(dtype=torch.int64).argmax())
            function_p = poses[:3, sample]+_rotate(poses[3:, sample], ca.DM(event.tool_from_function[:3, 3]))
            function_q = _quaternion_product(poses[3:, sample], ca.DM(_quaternion(event.tool_from_function)))
            relative = _quaternion_product(_conjugate(ca.DM(_quaternion(event.target))), function_q)
            event_residuals.append((function_p-event.target[:3, 3])/event.position_tolerance_m)
            norm = ca.sqrt(ca.sumsqr(relative))
            if event.yaw_axis is None:
                event_residuals.append(relative[1:]/(norm*np.sin(event.rotation_tolerance_rad/2)))
            else:
                projection = ca.dot(relative[1:], ca.DM(event.yaw_axis))
                perpendicular = relative[1:]-projection*ca.DM(event.yaw_axis)
                event_residuals.append(perpendicular/(norm*np.sin(event.rotation_tolerance_rad/2)))
                if 0. < event.yaw_tolerance_rad < np.pi:
                    event_residuals.append(projection/(ca.sqrt(relative[0]**2+projection**2)
                        *np.sin(event.yaw_tolerance_rad/2)))
                elif event.yaw_tolerance_rad == 0.:
                    twist = ca.vertcat(relative[0], projection*ca.DM(event.yaw_axis))
                    event_residuals.append(ca.dot(_rotation_vector(twist), ca.DM(event.yaw_axis)))
        components = dict(reference_prior=ca.vertcat(*reference_residuals),
            tool_angular_motion=ca.vertcat(*angular_residuals),
            motion_quality=_state_residuals([state[ordinary, :] for state in states],
                computation.bounds.numpy(), computation.motion_configuration, computation.native_dt,
                weights, declaration.joint_motion_weight),
            event_initialization=ca.vertcat(*event_residuals))
        residual = ca.vertcat(*components.values())
        jacobian = ca.jacobian(residual, x)
        self.residual = ca.Function('complete_reference_projection', [x], [residual])
        self.jacobian = ca.Function('complete_reference_projection_jacobian', [x], [jacobian])
        self.component_names = tuple(components)
        self.component_costs = ca.Function('complete_reference_projection_costs', [x],
            [ca.vertcat(*[ca.sumsqr(value) for value in components.values()])])
        self.report = dict(build_seconds=perf_counter()-started, variables=x.numel(),
            residuals=residual.numel(), jacobian_nonzeros=jacobian.nnz(),
            coordinate_box='Complete original cubic position coefficients in original joint ranges',
            scope='Numerical initial coefficients; original complete problem supplies subsequent task and scene values')

    def fit(self, initial, *, maximum_evaluations):
        """Return one bounded coefficient proposal with complete fit provenance."""
        started = perf_counter()
        seed = np.clip(np.asarray(initial).ravel(), *self.bounds)

        def residual(x):
            self.check_request()
            return np.asarray(self.residual(x)).ravel()

        def jacobian(x):
            self.check_request()
            return self.jacobian(x).sparse()

        result = least_squares(residual, seed, jac=jacobian, bounds=self.bounds,
            max_nfev=maximum_evaluations, x_scale='jac',
            ftol=1e-6, xtol=1e-6, gtol=1e-6)
        costs = np.asarray(self.component_costs(result.x)).ravel()
        return result.x.reshape(self.shape), dict(status=int(result.status),
            message=result.message, evaluations=int(result.nfev), jacobian_evaluations=int(result.njev),
            cost=float(2*result.cost), optimality=float(result.optimality),
            components=dict(zip(self.component_names, map(float, costs))),
            input_box_projection_linf=float(np.max(np.abs(seed-np.asarray(initial).ravel()))),
            elapsed_seconds=perf_counter()-started)
# END original task, scene and continuous physical conditions own all later decisions.
