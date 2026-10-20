"""One delivered Hermite64 curve and its fixed physical constraint rows.

The free variables retain the existing cubic coefficient meaning. This module
owns no optimizer, robot geometry, task objective, or execution decision. Its
constraint rows describe the actual float64 q/dq samples that are exported and
reconstructed as a Hermite polynomial on the request's fixed time grid.
"""

# BEGIN shared physical curve and canonical numerical acceptance contract.
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PPoly

from .cubic_trajectory import CubicBasis


PHYSICAL_FTOL = 1e-6


def aggregate_violation(slack):
    """Sum negative canonical slacks in the declared one-SI-unit scales.

    Invalid numerical output cannot satisfy a physical inequality. Returning
    infinity keeps that failure explicit when an external solver reports NaN.
    This is not a replacement geometry value or a relaxed physical boundary.
    """
    values = np.asarray(slack, dtype=np.float64)
    if not np.isfinite(values).all():
        return float("inf")
    return float(np.maximum(-values, 0.).sum())


@dataclass(frozen=True)
class PhysicalConstraints:
    """Variable rows for a solver plus immutable failures for the same report."""
    slack: np.ndarray
    jacobian: np.ndarray
    names: tuple[str, ...]
    fixed_slack: np.ndarray
    fixed_names: tuple[str, ...]
    raw_residuals: dict
    reports: dict

    @property
    def violation(self):
        """The single candidate criterion includes unrepairable fixed rows."""
        return aggregate_violation(self.slack)+aggregate_violation(self.fixed_slack)


def _polynomial_minimum(coefficients):
    """Find every segment's minimum on normalized local time [0, 1].

    Coefficients have shape (4, segments, components), highest power first.
    SciPy solves derivative roots independently inside each polynomial piece;
    discontinuity roots are disabled because both segment endpoints are
    already evaluated. Ties keep an endpoint or the earliest internal root.
    """
    segments, components = coefficients.shape[1:]
    at_zero, at_one = coefficients[-1], coefficients.sum(axis=0)
    use_one = at_one < at_zero
    minimum = np.where(use_one, at_one, at_zero).copy()
    witness = use_one.astype(np.float64)
    polynomial = PPoly(coefficients, np.arange(segments+1, dtype=np.float64), extrapolate=False)
    roots = polynomial.derivative().roots(discontinuity=False, extrapolate=False)
    for component in range(components):
        values = np.asarray(roots[component], dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0.) & (values < segments)]
        index = np.floor(values).astype(np.int64)
        local = values-index
        interior = (local > 0.) & (local < 1.)
        index, local = index[interior], local[interior]
        if not len(local):
            continue
        a, b, c, d = coefficients[:, index, component]
        candidate = ((a*local+b)*local+c)*local+d
        root_minimum = np.full(segments, np.inf)
        np.minimum.at(root_minimum, index, candidate)
        root_witness = np.full(segments, np.inf)
        selected = candidate == root_minimum[index]
        np.minimum.at(root_witness, index[selected], local[selected])
        better = root_minimum < minimum[:, component]
        minimum[better, component] = root_minimum[better]
        witness[better, component] = root_witness[better]
    return minimum, witness


def _factor_coefficients(coefficients, order):
    """Divide only predeclared exact factors tau**order, retaining 4 rows."""
    result = np.zeros_like(coefficients)
    for power in range(4):
        selected = order == power
        for coefficient in range(4-power):
            result[coefficient+power, selected] = coefficients[coefficient, selected]
    return result


def _canonical_weights(coefficients, segments, order, witness):
    """Action derivative of a factored polynomial at its minimizing time."""
    powers = 3-np.arange(4)[:, None]-order[None]
    multipliers = witness[None]**np.maximum(powers, 0)
    multipliers *= powers >= 0
    return np.einsum("kr,krc->rc", multipliers, coefficients[:, segments])


class PhysicalCurve:
    """Bind one affine Hermite64 family and one canonical physical row layout.

    ``controls`` is the number of free coefficients, not a seed array. Actions
    are either (controls, joints) or flattened in that order. ``bounds`` maps
    position/velocity/acceleration/jerk to the original (2, joints) limits.
    Couplings use the existing JointCoupling names/matrix/target contract;
    stop_times are physical seconds on this already-frozen clock.

    All constraint values retain their SI units (rad or m and derivatives,
    with the coupling's original coordinate units). The canonical scale is
    one such unit. Dividing by dimensionless local tau therefore does not
    change units, but does change finite-tolerance semantics: the fixed grid
    and factor layout must be reused for optimization and candidate checks.
    """

    def __init__(self, q0, dq0, controls, knot_dt, times, joint_names, bounds, *,
                 couplings=(), stop_times=(), joint_tolerance, velocity_tolerance):
        self.joint_names = tuple(joint_names)
        self.q0, self.dq0 = np.asarray(q0, dtype=np.float64), np.asarray(dq0, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64).copy()
        self.controls, self.dof = controls, len(self.joint_names)
        self.basis = CubicBasis(controls, knot_dt)
        self.duration = self.basis.intervals*self.basis.knot_dt
        if (not self.dof or len(set(self.joint_names)) != self.dof
                or self.q0.shape != (self.dof,) or self.dq0.shape != (self.dof,)
                or not np.isfinite(self.q0).all() or not np.isfinite(self.dq0).all()
                or self.times.ndim != 1 or len(self.times) < 2
                or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0.)
                or self.times[0] != 0. or self.times[-1] != self.duration):
            raise ValueError("Physical curve requires finite named entrance data and the complete fixed clock")
        if any(not np.isfinite(value) or value <= 0.
               for value in (joint_tolerance, velocity_tolerance)):
            raise ValueError("Coupling and stop bands must be the existing positive numerical tolerances")
        self.bounds = {name: np.asarray(bounds[name], dtype=np.float64).copy()
                       for name in ("position", "velocity", "acceleration", "jerk")}
        if any(value.shape != (2, self.dof) or not np.isfinite(value).all()
               or np.any(value[0] > value[1]) for value in self.bounds.values()):
            raise ValueError("Physical bounds must contain finite ordered limits for every named joint")
        self.stop_times = np.unique(np.asarray(stop_times, dtype=np.float64))
        if (self.stop_times.ndim != 1 or not np.isfinite(self.stop_times).all()
                or np.any(self.stop_times < 0.) or np.any(self.stop_times > self.duration)):
            raise ValueError("Explicit stops must use finite times inside the fixed curve")
        self.joint_tolerance, self.velocity_tolerance = float(joint_tolerance), float(velocity_tolerance)
        self._interval_dt = np.diff(self.times)
        # Exact entrance rows come from CubicBasis, and remain exact Hermite
        # input nodes. Their structural dependence never reads current actions.
        knot_times = self.times/self.basis.knot_dt
        knot_times[[0, -1]] = [0., self.basis.intervals]
        rows = [self.basis.weights_at_knot_times(knot_times, derivative) for derivative in (0, 1)]
        self._node_weights = tuple(row[0] for row in rows)
        self._node_bias = tuple((float(derivative == 0)-row[0].sum(axis=1))[:, None]*self.q0
                                + row[1][:, None]*self.dq0 for derivative, row in enumerate(rows))
        self._weight_curve = CubicHermiteSpline(self.times, *self._node_weights)
        self._bias_curve = CubicHermiteSpline(self.times, *self._node_bias)
        self._groups = []
        self._fixed_endpoint_failures = []
        self._fixed_violation_lower_bound = 0.
        self._constant_legal_count = 0
        for derivative, kind in enumerate(self.bounds):
            linear, offset, names, categories = [], [], [], []
            for side, sign in (("lower", 1.), ("upper", -1.)):
                for joint, joint_name in enumerate(self.joint_names):
                    linear.append(sign*np.eye(self.dof)[joint])
                    offset.append(-self.bounds[kind][0, joint] if sign > 0. else self.bounds[kind][1, joint])
                    names.append(f"{kind}/{joint_name}/{side}")
                    categories.append(f"continuous_{kind}")
            if derivative == 0:
                for index, coupling in enumerate(couplings):
                    columns = [self.joint_names.index(name) for name in coupling.joint_names]
                    matrix = np.zeros((len(coupling.target), self.dof))
                    for source, column in enumerate(columns):
                        matrix[:, column] += coupling.matrix[:, source]
                    for side, sign in (("lower", 1.), ("upper", -1.)):
                        for row, target in enumerate(coupling.target):
                            linear.append(sign*matrix[row])
                            offset.append(self.joint_tolerance-sign*target)
                            names.append(f"coupling:{index}/row:{row}/{side}")
                            categories.append(f"coupling:{index}")
            self._bind_group(derivative, np.asarray(linear), np.asarray(offset), tuple(names), tuple(categories))
        self._stop_weights, self._stop_bias = self.sample_map(self.stop_times, derivative=1)
        fixed_slacks = [group['fixed_slack'] for group in self._groups]
        fixed_names = [name for group in self._groups for name in group['fixed_names']]
        self._stop_variable = np.any(self._stop_weights != 0., axis=1)
        for index in np.flatnonzero(~self._stop_variable):
            for side, sign in (("lower", 1.), ("upper", -1.)):
                slack = self.velocity_tolerance+sign*self._stop_bias[index]
                failed = slack < 0.
                fixed_slacks.append(slack[failed])
                fixed_names.extend(f"stop:{index}/{name}/{side}" for joint, name in enumerate(self.joint_names)
                                   if failed[joint])
                self._fixed_violation_lower_bound += aggregate_violation(slack[failed])
        self.fixed_slack = np.concatenate(fixed_slacks)
        self.fixed_names = tuple(fixed_names)
        self.fixed_violation_lower_bound = float(self._fixed_violation_lower_bound)

    def _normalized_coefficients(self, polynomial, derivative):
        """Keep physical derivatives, then normalize only the local time."""
        source = polynomial.derivative(derivative).c
        coefficients = np.zeros((4, len(self.times)-1, source.shape[-1]))
        powers = np.arange(source.shape[0]-1, -1, -1)
        coefficients[derivative:] = source*self._interval_dt[None, :, None]**powers[:, None, None]
        return coefficients

    def _bind_group(self, derivative, linear, offset, names, categories):
        weights = self._normalized_coefficients(self._weight_curve, derivative)
        bias = np.einsum("ksj,cj->ksc", self._normalized_coefficients(self._bias_curve, derivative), linear)
        bias[-1] += offset
        has_coordinate = np.any(linear != 0., axis=1)
        dependent = np.any(weights != 0., axis=(0, 2))[:, None] & has_coordinate[None]
        order = np.zeros(dependent.shape, dtype=np.int64)
        # These tests establish an affine identity over every action. They
        # never examine the current polynomial value, an evaluated Jacobian,
        # or an epsilon. Nonzero near-boundary input is not a removable zero.
        for power in range(3):
            independent_coefficient = np.all(weights[3-power] == 0., axis=1)[:, None] | ~has_coordinate[None]
            removable = (order == power) & independent_coefficient & (bias[3-power] == 0.)
            order[removable] += 1
        canonical_bias = _factor_coefficients(bias, order)
        piece_minimum, _ = _polynomial_minimum(canonical_bias)
        fixed_minimum = np.min(np.where(~dependent, piece_minimum, np.inf), axis=0)
        fixed_bad = fixed_minimum < 0.
        self._constant_legal_count += int(np.count_nonzero(~dependent & (piece_minimum >= 0.)))
        row_names = np.asarray([[f"{name}/segment:{segment}" for name in names]
                                for segment in range(len(self.times)-1)], dtype=object)
        # A variable polynomial can still have a fixed negative endpoint.
        # Expose that impossibility without adding a duplicate solver row or
        # counting the same segment's violation twice.
        segment_rows = np.indices(dependent.shape)[0]
        flat_segments, flat_order = segment_rows.ravel(), order.ravel()
        immutable_minimum = np.full(dependent.shape, np.inf)
        for endpoint in (0., 1.):
            endpoint_weights = _canonical_weights(weights, flat_segments, flat_order,
                np.full(flat_order.shape, endpoint)).reshape(*dependent.shape, self.controls)
            fixed_endpoint = np.all(endpoint_weights == 0., axis=-1) | ~has_coordinate[None]
            endpoint_value = canonical_bias[-1] if endpoint == 0. else canonical_bias.sum(axis=0)
            immutable_minimum = np.minimum(immutable_minimum,
                np.where(fixed_endpoint, endpoint_value, np.inf))
            for segment, component in np.argwhere(dependent & fixed_endpoint & (endpoint_value < 0.)):
                self._fixed_endpoint_failures.append({"name": str(row_names[segment, component]),
                    "tau": endpoint, "slack": float(endpoint_value[segment, component])})
        self._fixed_violation_lower_bound += aggregate_violation(fixed_minimum[fixed_bad])
        immutable_minimum = np.min(np.where(dependent, immutable_minimum, np.inf), axis=0)
        self._fixed_violation_lower_bound += aggregate_violation(immutable_minimum[np.isfinite(immutable_minimum)])
        self._groups.append(dict(derivative=derivative, linear=linear, offset=offset, names=names,
            categories=categories, weights=weights, order=order, dependent=dependent, fixed_bad=fixed_bad,
            variable_components=np.any(dependent, axis=0), fixed_slack=fixed_minimum[fixed_bad].copy(),
            fixed_names=tuple(np.asarray(names)[fixed_bad]), row_names=row_names))

    def _actions(self, actions):
        values = np.asarray(actions, dtype=np.float64)
        if values.shape == (self.controls*self.dof,):
            values = values.reshape(self.controls, self.dof)
        if values.shape != (self.controls, self.dof) or not np.isfinite(values).all():
            raise ValueError("Physical actions must be finite free coefficients in named joint order")
        return values

    def export_nodes(self, actions):
        """Return the original float64 Hermite inputs without an evaluation roundtrip."""
        values = self._actions(actions)
        return tuple(weights@values+bias for weights, bias in zip(self._node_weights, self._node_bias))

    def curve(self, actions):
        """Rebuild exactly the float64 q/dq node arrays used for delivery."""
        return CubicHermiteSpline(self.times, *self.export_nodes(actions))

    def sample_map(self, query_times, derivative=0):
        """Return the fixed affine Hermite map W@actions+b at physical times.

        W has shape (queries, controls), b (queries, joints). The mathematically
        affine map is also the action derivative of the exported polynomial;
        its input-node reconstruction can differ by float64 roundoff. Both
        constraint and candidate checks always evaluate ``curve(actions)``.
        """
        times = np.asarray(query_times, dtype=np.float64)
        if (times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0.)
                or np.any(times > self.duration) or derivative not in (0, 1, 2, 3)):
            raise ValueError("Physical samples require in-domain times and derivative 0..3")
        return self._weight_curve(times, nu=derivative), self._bias_curve(times, nu=derivative)

    def samples(self, actions, query_times, derivative=0):
        """Evaluate the actual delivered polynomial, including breakpoint sides."""
        self.sample_map(query_times, derivative)
        return self.curve(actions)(np.asarray(query_times, dtype=np.float64), nu=derivative)

    def constraints(self, actions):
        """Return fixed canonical inequalities and raw physical diagnostics."""
        values = self._actions(actions)
        curve = self.curve(values)
        slacks, jacobians, names = [], [], []
        active_constraints = []
        residuals, reports = {}, {}
        for group in self._groups:
            coefficients = np.einsum("ksj,cj->ksc", self._normalized_coefficients(curve, group['derivative']), group['linear'])
            coefficients[-1] += group['offset']
            raw, raw_witness = _polynomial_minimum(coefficients)
            canonical, witness = _polynomial_minimum(_factor_coefficients(coefficients, group['order']))
            component = np.flatnonzero(group['variable_components'])
            segment = np.argmin(np.where(group['dependent'], canonical, np.inf), axis=0)[component]
            local_weights = _canonical_weights(group['weights'], segment,
                group['order'][segment, component], witness[segment, component])
            jacobian = (local_weights[:, :, None]*group['linear'][component, None, :]).reshape(len(segment), -1)
            slacks.append(canonical[segment, component])
            jacobians.append(jacobian)
            names.extend(group['names'][column] for column in component)
            active_constraints.extend({"name": group['names'][column], "segment": int(interval),
                "tau": float(witness[interval, column]), "factor_order": int(group['order'][interval, column]),
                "time_s": float(self.times[interval]+witness[interval, column]*self._interval_dt[interval])}
                for interval, column in zip(segment, component))
            for category in dict.fromkeys(group['categories']):
                selected = np.array([value == category for value in group['categories']])
                columns = np.flatnonzero(selected)
                minimum = raw[:, selected]
                interval, local_column = np.unravel_index(np.argmin(minimum), minimum.shape)
                column = columns[local_column]
                violation = max(0., -float(minimum[interval, local_column]))
                residuals[category] = violation
                reports[category] = {"maximum_bound_excess": violation, "maximum_violation": None}
                if violation:
                    tau = float(raw_witness[interval, column])
                    reports[category]['maximum_violation'] = {
                        "constraint": group['names'][column], "segment": int(interval),
                        "time_s": float(self.times[interval]+tau*self._interval_dt[interval]),
                        "tau": tau, "signed_slack": -violation}
        if len(self.stop_times):
            velocity = curve(self.stop_times, nu=1)
            for index, (time_s, weights, current) in enumerate(zip(
                    self.stop_times, self._stop_weights, velocity)):
                category = f"stop:{index}"
                violation = float(np.maximum(np.abs(current)-self.velocity_tolerance, 0.).max())
                residuals[category] = violation
                reports[category] = {"time_s": float(time_s), "maximum_bound_excess": violation}
                for side, sign in (("lower", 1.), ("upper", -1.)):
                    slack = self.velocity_tolerance+sign*current
                    row_names = [f"stop:{index}/{name}/{side}" for name in self.joint_names]
                    if self._stop_variable[index]:
                        slacks.append(slack)
                        jacobians.append((sign*weights[None, :, None]*np.eye(self.dof)[:, None]).reshape(self.dof, -1))
                        names.extend(row_names)
        reports['canonical'] = {
            "scope": "fixed Hermite64 segments; local-time factors; one-SI-unit scales; minimum over variable pieces per joint/derivative/side or coupling row",
            "ftol": PHYSICAL_FTOL, "constant_legal_rows_omitted": self._constant_legal_count,
            "fixed_endpoint_failures": list(self._fixed_endpoint_failures),
            "fixed_violation_lower_bound": self.fixed_violation_lower_bound,
            "active_constraints": active_constraints}
        return PhysicalConstraints(np.concatenate(slacks), np.concatenate(jacobians), tuple(names),
            self.fixed_slack.copy(), self.fixed_names, residuals, reports)

# END optimization, candidate qualification and export share these same rows.
