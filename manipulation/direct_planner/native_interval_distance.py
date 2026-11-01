"""Registered robot distances at bracketed stationary trajectory times.

The original sample clock brackets scalar distance minima. Each bracket uses
SciPy Brent on the distance time derivative, then queries the same Coal solid
pair at the root. Its envelope derivative enters the existing interval row.
"""

# BEGIN temporal queries share original solids, curve and nominal joint state.
import numpy as np
import pinocchio as pin
from scipy.optimize import brentq
from scipy.interpolate import PPoly

from manipulation.geometry_motion_bound import SphereBody, build_geometry_motion_bound
from manipulation.implicit_distance import box_corners
from manipulation.interval_geometry import _polynomial_ranges
from .sparse_rows import coefficient_columns
from .native_gradient import native_joint_gradient


class NativeIntervalDistance:
    """Bind native pair intervals, spline support and point-speed coefficients.

    Covers derived from registered official robot entities only bound speed.
    Coal queries retain those entities and their fixed local transforms. A
    common ancestor moves both bodies rigidly, so its contribution to their
    distance and relative speed is identically zero.
    """

    def __init__(self, bound):
        self.bound = bound
        self.base, self.scene = bound.owner.computation, bound.owner.scene
        self.curve = self.base.domain._curve_map
        self.model = self.scene.robot_model.model
        self.data = self.model.createData()
        self.names = tuple(str(name) for name in self.model.names[1:])
        self.controlled = np.array([self.names.index(name) for name in self.base.joint_names])
        self.q_columns = np.array([self.model.joints[i].idx_q for i in range(1, self.model.njoints)])
        self.v_columns = np.array([self.model.joints[i].idx_v for i in range(1, self.model.njoints)])
        neutral = pin.neutral(self.model)
        self.translating = []
        for column, name in enumerate(self.base.joint_names):
            joint = self.model.joints[self.model.getJointId(name)]
            data = joint.createData(); joint.calc(data, neutral)
            if np.any(np.asarray(data.S).reshape(6)[:3] != 0.):
                self.translating.append(column)
        self.calibration = np.asarray(self.scene.robot_validator.T_reference_base)
        self.bodies, covers, ancestors = {}, [], {}
        for name, (frame, geometry, _, local) in self.scene.robot_model._geometries.items():
            identity = 'robot/'+name
            frame_id = self.model.getFrameId(frame)
            self.bodies[identity] = frame, frame_id, local
            geometry.computeLocalAABB()
            corners = box_corners(np.asarray(geometry.aabb_local.min_), np.asarray(geometry.aabb_local.max_))
            local_corners = corners@local[:3, :3].T+local[:3, 3]
            radius = np.nextafter(np.linalg.norm(local_corners, axis=1).max(), np.inf)
            covers.append(SphereBody(identity, frame, [[0., 0., 0., radius]]))
            joint, chain = int(self.model.frames[frame_id].parentJoint), set()
            while joint:
                chain.add(joint-1)
                joint = int(self.model.parents[joint])
            ancestors[identity] = chain
        speed = build_geometry_motion_bound(model=self.model,
            T_reference_base=self.calibration, bodies=covers)
        speed_by_name = dict(zip(speed.body_names, speed.coefficients))
        self.common, self.relative_coefficients, self.intervals = {}, {}, {}
        # BEGIN structural masks are shared by scalar and batched derivative calls.
        self.common_ancestor_masks = {}
        # END current poses never change common-ancestor membership.
        self.interval_indices, self.interval_times = {}, {}
        self.speed_clocks, self.speed_membership = {}, {}
        self.support_columns = {}
        # BEGIN interval support depends on the curve and clock, not the entity pair.
        knots = np.arange(self.curve.intervals+1)*self.curve.knot_dt
        support_by_interval = {}
        joint_columns = np.arange(bound.joints)
        # END each original row below references its interval's same coefficient set.
        for pair, rows in bound.native_rows.items():
            self.common[pair] = np.array(sorted(ancestors[pair[0]] & ancestors[pair[1]]), dtype=int)
            self.common_ancestor_masks[pair] = np.zeros(len(self.names), dtype=bool)
            self.common_ancestor_masks[pair][self.common[pair]] = True
            coefficients = speed_by_name[pair[0]]+speed_by_name[pair[1]]
            coefficients[self.common[pair]] = 0.
            self.relative_coefficients[pair] = coefficients
            # Exact physical states precede event-side copies. Unique times
            # therefore retain the original exact state at each boundary.
            times = np.array([bound.records[row]['time_s'] for row in rows])
            _, unique = np.unique(times, return_index=True)
            ordered = rows[unique]
            self.intervals[pair] = np.stack((ordered[:-1], ordered[1:]), axis=1)
            self.interval_indices[pair] = np.searchsorted(rows, self.intervals[pair])
            self.interval_times[pair] = np.array([
                [bound.records[left]['time_s'], bound.records[right]['time_s']]
                for left, right in self.intervals[pair]], dtype=float).reshape(-1, 2)
            # BEGIN local original polynomial segments bound each query interval.
            # Closed overlap includes adjacent segments at exact boundaries;
            # it can only enlarge the rate bound used to omit distant roots.
            clock = self.interval_times[pair].tobytes()
            self.speed_clocks[pair] = clock
            # BEGIN identical interval clocks share their exact segment membership.
            if clock not in self.speed_membership:
                self.speed_membership[clock] = (
                    (knots[1:][None] >= self.interval_times[pair][:, 0, None])
                    & (knots[:-1][None] <= self.interval_times[pair][:, 1, None]))
            # END pair-specific relative speed coefficients remain separate.
            # END the original Coal query still determines every distance value.
            for left, right in self.intervals[pair]:
                a, b = bound.records[left]['time_s'], bound.records[right]['time_s']
                # BEGIN calculate this original interval's support once across pairs.
                key = (a, b)
                if key not in support_by_interval:
                    partition = np.r_[a, knots[(knots > a) & (knots < b)], b]
                    query = np.r_[partition, (partition[:-1]+partition[1:])/2]
                    basis = self.curve.query(query/self.base.duration, 0)[0]@bound.terminal.matrix
                    support_by_interval[key] = coefficient_columns(basis, joint_columns, bound.joints)
                self.support_columns[int(left)] = support_by_interval[key]
                # END the row identity and full possible sparse derivative columns remain.
        self.fixed_speeds = np.zeros(len(self.names))
        hand = self.scene.gripper
        nominal = None if hand is None else hand.nominal_finger_trajectory
        if nominal is not None and nominal.valid_until_s > 0.:
            lower, upper = _polynomial_ranges(nominal._curve, 0.,
                min(self.base.duration, nominal.valid_until_s), 1)
            for name, value in zip(nominal.joint_names, np.maximum(abs(lower), abs(upper))):
                self.fixed_speeds[self.names.index(name)] = value

    def _joint_gradient(self, pair, numeric, origins, jacobians):
        """Pull original nearest-point covectors into complete named joints."""
        # BEGIN the compiled ufunc owns only the original small-array arithmetic.
        return native_joint_gradient(numeric, origins[0], origins[1], jacobians[0], jacobians[1],
                                     self.common_ancestor_masks[pair])
        # END original Pinocchio and Coal continue producing all model values.

    def _at(self, pair, moment, polynomial, complete_entry):
        """Query one original pair at an explicit physical curve time."""
        self.scene.check_request()
        # BEGIN one original polynomial replaces repeated basis assembly.
        # Coefficients below are the original affine curve's coefficients in
        # physical seconds. SciPy evaluates the same position and derivative.
        q, dq = polynomial(moment), polynomial(moment, nu=1)
        complete = complete_entry.copy()
        # END no fitted or sampled replacement trajectory is introduced.
        rate = np.zeros(len(self.names))
        complete[self.controlled], rate[self.controlled] = q, dq
        hand = self.scene.gripper
        if hand is not None:
            missing = [name for name in hand.joint_names if name not in self.base.joint_names]
            nominal = hand.nominal_finger_trajectory
            if missing and nominal is not None:
                # Both interval endpoints already have an active native row.
                # The declared validity prefix covers their interior. The
                # same polynomial's NumPy API avoids rebuilding a Torch hand
                # state for every scalar root evaluation.
                position, velocity = nominal.sample(np.array([moment]))
                for name in missing:
                    column = hand.joint_names.index(name)
                    complete[self.names.index(name)] = position[0, column]
                    rate[self.names.index(name)] = velocity[0, column]
        model_q = np.empty(self.model.nq)
        model_q[self.q_columns] = complete
        pin.computeJointJacobians(self.model, self.data, model_q)
        pin.updateFramePlacements(self.model, self.data)
        matrices = np.empty((1, 2, 4, 4))
        origins, jacobians = [], []
        for side, name in enumerate(pair):
            _, frame_id, local = self.bodies[name]
            pose = self.calibration@self.data.oMf[frame_id].homogeneous
            matrices[0, side] = pose@local
            original = pin.getFrameJacobian(self.model, self.data, frame_id, pin.LOCAL_WORLD_ALIGNED)
            jacobian = np.vstack((self.calibration[:3, :3]@original[:3, self.v_columns],
                                  self.calibration[:3, :3]@original[3:, self.v_columns]))
            origins.append(pose[:3, 3])
            jacobians.append(jacobian)
        numeric = self.bound.owner._native_tables[pair].query(matrices,
            np.zeros((1, 1), dtype=bool), self.bound.native_request)[0, 0]
        gradient = self._joint_gradient(pair, numeric, origins, jacobians)
        return numeric, gradient, float(gradient@rate)

    def evaluate(self, controls, context, native_numeric, active):
        """Return right-endpoint or stationary interval winners and evidence.

        A Lipschitz distance bound skips intervals wholly in the constant
        negative tail. Every remaining negative-to-positive derivative bracket
        supplies an actual original-entity query; a possible approach alone
        has no collision value. The finite bracket scope is retained in data.
        """
        extrema, _ = self.curve._segment_extrema(controls, 1)
        # BEGIN reuse each original segment's exact continuous rate extrema.
        segment_speeds = np.max(np.abs(extrema), axis=2)
        interval_speeds = {}
        for clock, membership in self.speed_membership.items():
            current = np.broadcast_to(self.fixed_speeds, (len(membership), len(self.names))).copy()
            current[:, self.controlled] = np.max(
                np.where(membership[:, :, None], segment_speeds[None], 0.), axis=1)
            interval_speeds[clock] = current
        # END unrelated high rates elsewhere on the curve do not enlarge this interval.
        # A translating optimized joint can enlarge an ancestor's lever arm.
        # The reusable motion coefficients use registered position ranges;
        # an out-of-range numerical trial therefore receives no far pruning.
        coefficients = self.curve.coefficients(controls)
        translating = np.array(self.translating, dtype=int)
        native_columns = self.q_columns[self.controlled[translating]]
        bounded_levers = (np.all(coefficients[:, translating] >= self.model.lowerPositionLimit[native_columns])
                          and np.all(coefficients[:, translating] <= self.model.upperPositionLimit[native_columns]))
        order = [context['complete_joint_names'].index(name) for name in self.names]
        rates = context['complete_dq'][0].numpy()[:, order]
        # BEGIN immutable per-point polynomial and named entrance are shared by roots.
        polynomial_coefficients = np.einsum('ksn,nj->ksj', self.curve.polynomial_weights,
                                            controls-self.curve.start_q)
        polynomial_coefficients += self.curve.polynomial_velocity[:, :, None]*self.curve.start_dq
        polynomial_coefficients /= self.curve.knot_dt**np.arange(3, -1, -1)[:, None, None]
        polynomial_coefficients[-1] += self.curve.start_q
        polynomial = PPoly(polynomial_coefficients,
            np.arange(self.curve.intervals+1)*self.curve.knot_dt, extrapolate=False)
        complete_entry = context['complete_q'][0, 0].numpy()[order]
        # END endpoint values and all root calls still belong to these controls.
        winners, brackets, calls = {}, 0, 0
        for pair, intervals in self.intervals.items():
            rows = self.bound.native_rows[pair]
            columns = self.bound.record_columns[rows]
            origins, jacobians = [], []
            for name in pair:
                frame = self.bodies[name][0]
                origins.append(context['links'][frame][0][0].numpy()[columns])
                jacobians.append(context['frame_jacobians'][frame][0].numpy()[columns][:, :, order])
            gradient = self._joint_gradient(pair, native_numeric[rows], origins, jacobians)
            derivative = np.einsum('ij,ij->i', gradient, rates[columns])
            speed = interval_speeds[self.speed_clocks[pair]]@self.relative_coefficients[pair]
            # BEGIN select endpoint winners and brackets with fixed interval arrays.
            left, right = intervals.T
            left_index, right_index = self.interval_indices[pair].T
            a, b = self.interval_times[pair].T
            available = active[left] & active[right]
            left_distance, right_distance = native_numeric[left, 0], native_numeric[right, 0]
            best_distances = np.minimum(left_distance, right_distance)
            right_wins = available & (right_distance < left_distance) & (right_distance < self.bound.distance_unit)
            for ordinal in np.flatnonzero(right_wins):
                left, right = intervals[ordinal]
                a, b = self.interval_times[pair][ordinal]
                winners[int(left)] = dict(row=int(left), time_s=float(b), interval_s=[float(a), float(b)],
                    kind='right_endpoint', numeric=native_numeric[right],
                    gradient=gradient[self.interval_indices[pair][ordinal, 1], self.controlled])
            bracket_mask = available & (derivative[left_index] < 0.) & (derivative[right_index] > 0.)
            if bounded_levers:
                widths = self.interval_times[pair][:, 1]-self.interval_times[pair][:, 0]
                bracket_mask &= best_distances-speed*widths/2 < self.bound.distance_unit
            # END array selection preserves all original active root brackets.
            for ordinal in np.flatnonzero(bracket_mask):
                left, right = intervals[ordinal]
                left_index, right_index = self.interval_indices[pair][ordinal]
                a, b = self.interval_times[pair][ordinal]
                best_distance = best_distances[ordinal]
                brackets += 1
                # Endpoints already belong to the same original query. Brent
                # may return one when a stationary point rounds to a boundary.
                cache = {float(moment): (native_numeric[row], gradient[index], derivative[index])
                    for row, index, moment in ((left, left_index, a), (right, right_index, b))}

                def slope(moment):
                    nonlocal calls
                    if moment in cache:
                        return cache[moment][2]
                    value = self._at(pair, moment, polynomial, complete_entry)
                    calls += 1
                    cache[moment] = value
                    return value[2]

                moment = brentq(slope, a, b, xtol=1e-10, rtol=8*np.finfo(float).eps)
                numeric, current_gradient, _ = cache[moment]
                if numeric[0] < best_distance and numeric[0] < self.bound.distance_unit:
                    winners[int(left)] = dict(row=int(left), time_s=float(moment), kind='stationary',
                        interval_s=[float(a), float(b)], numeric=numeric,
                        gradient=current_gradient[self.controlled])
        return list(winners.values()), dict(stationary_brackets=brackets, original_pair_queries=calls,
            interval_winners=len(winners), stationary_winners=sum(row['kind'] == 'stationary' for row in winners.values()),
            scope='Minimum over both original endpoints and bracketed distance stationary times')
# END each winner belongs to the same fixed interval row and original models.
