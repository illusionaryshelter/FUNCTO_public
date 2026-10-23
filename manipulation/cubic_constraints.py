"""Affine cubic queries, continuous physical inequalities and sample fitting.

The fixed measured entrance and physical clock leave only the free position
coefficients as optimization variables. SciPy supplies polynomial roots and
NumPy supplies the unconstrained least-squares seed calculation.
"""

# BEGIN the constraint representation is independent of numerical search.
from __future__ import annotations

import numpy as np
from scipy.interpolate import PPoly
import torch

from .cubic_trajectory import CubicBasis


def _numpy(value):
    """Copy an external Torch value to the CPU numerical interface if needed."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


class CubicAffineMap(CubicBasis):
    """Bind measured q0/dq0 to the shared fixed-clock cubic basis."""

    # BEGIN measured entrance vectors and clock define one affine curve.
    def __init__(self, start_q, start_dq, knots, knot_dt):
        super().__init__(knots, knot_dt)
        self.start_q, self.start_dq = _numpy(start_q).reshape(-1), _numpy(start_dq).reshape(-1)
        if (self.start_q.shape != self.start_dq.shape
                or not np.isfinite(self.start_q).all() or not np.isfinite(self.start_dq).all()):
            raise ValueError("Cubic entrance needs matching finite measured q0/dq0")

    def coefficients(self, actions):
        """Complete coefficients, computed relative to q0 before translation."""
        actions = _numpy(actions)
        return (self.start_q + np.einsum("cn,...nj->...cj", self.matrix, actions-self.start_q)
                + self.velocity_coefficients[:, None]*self.knot_dt*self.start_dq)

    def query(self, progress, derivative=0):
        weights, velocity = self.weights(_numpy(progress), derivative)
        return self._affine_query(weights, velocity, derivative)

    def query_knot_times(self, times, derivative=0):
        """Build rows at original segment coordinates, including jerk sides."""
        weights, velocity = self.weights_at_knot_times(_numpy(times), derivative)
        return self._affine_query(weights, velocity, derivative)

    def _affine_query(self, weights, velocity, derivative):
        offset = velocity[:, None]*self.start_dq-weights.sum(-1, keepdims=True)*self.start_q
        if derivative == 0:
            offset = offset+self.start_q
        return weights, offset

    def evaluate(self, actions, progress, derivative=0):
        actions = _numpy(actions)
        if actions.shape[-2:] != (self.knots, len(self.start_q)):
            raise ValueError("Cubic sampling requires [free controls,joints] trailing axes")
        weights, velocity = self.weights(_numpy(progress), derivative)
        values = (np.einsum("tn,...nj->...tj", weights, actions-self.start_q)
                  + velocity[:, None]*self.start_dq)
        return values+self.start_q if derivative == 0 else values
    # END all derivatives act on relative controls; public actions stay positions.

    # BEGIN raw continuous inequalities and seed samples share the affine curve.
    def _segment_extrema(self, actions, derivative):
        """Return per-segment minima/maxima and earliest active knot times.

        SciPy finds every interior root of the next derivative. Endpoints
        complete the extrema of each closed cubic segment. At a tie the
        earliest witness supplies one active subgradient; the function is
        piecewise differentiable and is not assigned a fictitious smooth
        derivative across active-witness changes.
        """
        count, dof = self.intervals, len(self.start_q)
        coefficients = np.einsum("ksn,nj->ksj", self.polynomial_weights, actions-self.start_q)
        coefficients += self.polynomial_velocity[:, :, None]*self.start_dq
        polynomial = PPoly(coefficients, np.arange(count+1, dtype=float), extrapolate=False)
        roots_by_joint = polynomial.derivative(derivative+1).roots(extrapolate=False)
        endpoints = np.arange(count+1, dtype=float)
        weights, velocity = self.weights_at_knot_times(endpoints, derivative)
        endpoints_value = weights@(actions-self.start_q)+velocity[:, None]*self.start_dq
        if derivative == 0:
            endpoints_value += self.start_q
        candidates = np.stack((endpoints_value[:-1], endpoints_value[1:]), axis=-1)
        selected = np.stack((candidates.argmin(-1), candidates.argmax(-1)), axis=-1)
        values = np.take_along_axis(candidates, selected, axis=-1)
        times = np.arange(count, dtype=float)[:, None, None]+selected
        for joint in range(dof):
            roots = roots_by_joint[joint]
            roots = roots[np.isfinite(roots) & (roots > 0.) & (roots < count)]
            # A root at a breakpoint is already represented on both segments.
            roots = roots[roots != np.floor(roots)]
            weights, velocity = self.weights_at_knot_times(roots, derivative)
            root_values = weights@(actions[:, joint]-self.start_q[joint])+velocity*self.start_dq[joint]
            if derivative == 0:
                root_values += self.start_q[joint]
            for moment, value in zip(roots, root_values):
                segment = int(moment)
                for side, improves in enumerate((value < values[segment, joint, 0],
                                                  value > values[segment, joint, 1])):
                    earlier_tie = value == values[segment, joint, side] and moment < times[segment, joint, side]
                    if improves or earlier_tie:
                        values[segment, joint, side] = value
                        times[segment, joint, side] = moment
        return values, times

    def signed_constraints(self, actions, bounds, *, with_witnesses=False):
        """Return exact continuous ``g <= 0`` and its selected Jacobian.

        ``actions`` has shape [free controls,joints], ``bounds`` is [4,2,joints],
        and Jacobian columns use C-order action flattening. Rows are ordered
        by derivative, segment, joint, then lower/upper side. Acceleration
        additionally has its left/right endpoint axis before the bound side.
        Position and velocity retain separate per-segment extrema. Their
        unique active extrema use the envelope derivative: stationarity
        removes the moving-root term. At tied extrema the earliest active
        branch gives a valid subgradient. Acceleration endpoint and constant
        segment jerk rows are affine, including both sides of jerk jumps.
        """
        actions, bounds = _numpy(actions), _numpy(bounds)
        dof = len(self.start_q)
        if actions.shape != (self.knots, dof) or bounds.shape != (4, 2, dof):
            raise ValueError("Continuous inequalities require [controls,joints] and [4,2,joints]")
        # BEGIN allocate the fixed physical Jacobian once for this query.
        # Each segment contributes 2+2+4+2 rows per joint. Derivative blocks
        # write directly into this matrix instead of retaining a second
        # complete matrix during concatenation.
        row_count = 10*self.intervals*dof
        residuals = np.empty(row_count)
        jacobian = np.zeros((row_count, self.knots, dof))
        witnesses, offset = [], 0
        # END the solver admission estimate can count one owned physical matrix.
        segments = np.arange(self.intervals)[:, None, None]
        for derivative in range(4):
            if derivative < 2:
                values, times = self._segment_extrema(actions, derivative)
            elif derivative == 2:
                times = np.broadcast_to(segments[..., None]+np.array([0., 1.])[None, None, :, None],
                                        (self.intervals, dof, 2, 2))
            else:
                times = np.broadcast_to(segments+.5, (self.intervals, dof, 2))
            # Each row names one joint, so it touches only that joint's free
            # control columns. Absolute controls themselves remain unbounded.
            indices = np.indices(times.shape)
            joints = indices[1].ravel()
            sides = indices[-1].ravel()
            signs = 2.*sides-1.
            flat_times = times.ravel()
            weights, velocity = self.weights_at_knot_times(flat_times, derivative)
            if derivative >= 2:
                values = (weights*(actions-self.start_q)[:, joints].T).sum(-1)
                values += velocity*self.start_dq[joints]
            else:
                values = values.ravel()
            row_indices = offset+np.arange(len(flat_times))
            residuals[row_indices] = signs*(values-bounds[derivative, sides, joints])
            jacobian[row_indices, :, joints] = signs[:, None]*weights
            offset += len(flat_times)
            if with_witnesses:
                witnesses.append({"derivative_order": np.full(len(flat_times), derivative, dtype=int),
                    "segment_index": indices[0].ravel(), "joint_column": joints,
                    "bound_side": sides, "time_s": flat_times*self.knot_dt,
                    "value": values, "bound": bounds[derivative, sides, joints]})
        result = residuals, jacobian.reshape(row_count, -1)
        if with_witnesses:
            return (*result, {name: np.concatenate([part[name] for part in witnesses])
                              for name in witnesses[0]})
        return result

    def fit_samples(self, progress, targets, prior_controls=None):
        """Least-squares sample seed with q0/dq0 and clock already eliminated.

        This computes only an initial coefficient estimate. The complete
        physical, task and geometry constraints belong to the subsequent
        constrained solve. In an underdetermined fit, NumPy's minimum-norm
        correction retains the supplied prior in the observation nullspace.
        The report includes the same solve's singular values and spectral
        condition number; a zero smallest singular value is recorded as None.
        """
        targets = _numpy(targets)
        weights, velocity = self.weights(_numpy(progress))
        if targets.shape != (len(weights), len(self.start_q)) or not np.isfinite(targets).all():
            raise ValueError("Cubic seed samples require finite [query,joint] targets")
        prior = (np.broadcast_to(self.start_q, (self.knots, len(self.start_q))).copy()
                 if prior_controls is None else _numpy(prior_controls))
        if prior.shape != (self.knots, len(self.start_q)) or not np.isfinite(prior).all():
            raise ValueError("Cubic seed prior must match the finite free-control layout")
        target_delta = targets-self.start_q-velocity[:, None]*self.start_dq
        # BEGIN report the observation conditioning already computed by lstsq.
        # A full-rank, exact interpolation can still amplify its samples.
        # Singular values describe that linear map; they do not change the
        # unconstrained seed or introduce a feasibility decision.
        correction, _, rank, singular = np.linalg.lstsq(
            weights, target_delta-weights@(prior-self.start_q), rcond=None)
        condition = float(singular[0]/singular[-1]) if singular[-1] > 0. else None
        # END a zero singular value has no finite spectral condition number.
        controls = prior+correction
        residual = self.evaluate(controls, progress)-targets
        return controls, {"success": True, "reason": "unconstrained_sample_fit", "rank": int(rank),
            "sample_matrix_singular_values": singular.tolist(),
            "sample_matrix_condition_number": condition,
            "reference_rms_joint_units": float(np.sqrt(np.mean(residual**2))),
            "reference_max_error_joint_units": float(np.max(np.abs(residual)))}
    # END physical feasibility is reported by signed rows, not imposed on the seed.

# END physical rows and seeds refer to exactly the same cubic coefficients.
