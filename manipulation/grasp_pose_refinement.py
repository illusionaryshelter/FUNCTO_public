"""Refine a frozen tool orientation against one observed background field.

The TCP origin stays fixed. Two orthonormal tool-model axes define a pure
swing chart about the original orientation; there is no independent spin
coordinate. Official robot shapes and their fixed/tool-joint transforms are
caller data. The field keeps its original RGB-D values and observed support.
SLSQP uses finite material samples for guidance, and the original registered
entity query supplies the final geometric classification.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import pinocchio as pin
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .native_field_distance import query_native_field


# BEGIN fixed-origin orientation refinement over observed TSDF support.
def _check_parts(tool_pose, parts, field):
    """Query each explicit component/state against the unchanged world field."""
    records = []
    for part in parts:
        pose = tool_pose @ np.asarray(part['T_tool_geometry'])
        started = perf_counter()
        answer = query_native_field(part['geometry'], pose, field, np.eye(4))
        records.append(dict(id=part['id'], geometry_id=part['geometry_id'],
            state_id=part['state_id'], T_world_geometry=pose,
            query_seconds=perf_counter()-started, **answer))
    return records


def _relation(records):
    """Aggregate the existing entity states without reclassifying uncertainty."""
    if any(row['status'] == 'intersection' for row in records):
        return 'intersection'
    if all(row['status'] == 'clear' for row in records):
        return 'clear_represented_set'
    return 'unresolved'


def refine_pose(T_world_tool, parts, field, basis_tool, max_tilt_rad, field_margin_m):
    """Return one fixed-origin, two-axis orientation proposal and its evidence.

    Parts supplies unique id, geometry_id, state_id, registered Coal
    geometry, T_tool_geometry and guide_points_tool for each tested component.
    Guide points are in the same TCP frame as basis_tool's two orthonormal
    columns. Their cross product names the excluded spin axis. Field is the
    existing world-frame FrozenObservedField; field_margin_m is a target
    TSDF scalar value, not a Euclidean clearance or a query precision.

    Only a confirmed original intersection invokes SLSQP. A clear baseline is
    already the zero-angle solution; a baseline with only unresolved queries
    retains its original uncertainty. An optimizer failure still publishes
    its returned proposal and the original entity queries at that proposal.
    """
    started = perf_counter()
    original = np.asarray(T_world_tool).copy()
    rotation0, origin = original[:3, :3], original[:3, 3]
    basis = np.asarray(basis_tool)
    axis = np.cross(basis[:, 0], basis[:, 1])
    sizes = np.asarray([len(part['guide_points_tool']) for part in parts], dtype=int)
    ends = np.cumsum(sizes)
    starts = np.r_[0, ends[:-1]]
    points = np.concatenate([np.asarray(part['guide_points_tool']) for part in parts])
    before_started = perf_counter()
    before_checks = _check_parts(original, parts, field)
    before_seconds = perf_counter()-before_started
    before_status = _relation(before_checks)
    history = dict(evaluations=[], iterations=[])
    cached_q, cached = None, None
    field_seconds = 0.

    def guide_report(sample):
        """Serialize observed values with explicit indices, never unknown fills."""
        known = sample['known']
        supported = np.flatnonzero(known)
        rows = []
        for part, first, last in zip(parts, starts, ends):
            count = int(known[first:last].sum())
            rows.append(dict(id=part['id'], geometry_id=part['geometry_id'],
                state_id=part['state_id'], point_count=int(last-first),
                known_count=count, unknown_count=int(last-first-count)))
        worst = None
        if len(supported):
            index = int(supported[np.argmin(sample['values'][supported])])
            part_index = int(np.searchsorted(ends, index, side='right'))
            part = parts[part_index]
            worst = dict(id=part['id'], geometry_id=part['geometry_id'],
                state_id=part['state_id'], global_sample_index=index,
                component_sample_index=int(index-starts[part_index]),
                point_tool=points[index], point_world=sample['world'][index],
                field_value_m=float(sample['values'][index]),
                field_gradient_world=sample['gradients'][index],
                dphi_dq_m_per_rad=sample['dphi_dq'][index],
                normalized_constraint=float(sample['constraints'][index]),
                normalized_constraint_jacobian=sample['jacobian'][index])
        return dict(point_count=len(points), known_count=int(known.sum()),
            unknown_count=int((~known).sum()), parts=rows,
            supported_indices=supported, unsupported_indices=np.flatnonzero(~known),
            supported_values_m=sample['values'][known],
            supported_gradients_world=sample['gradients'][known],
            supported_arithmetic_errors_m=sample['errors'][known],
            worst_sample=worst,
            inactive_constraint_encoding='unsupported row: constant 1, zero Jacobian; no field observation',
            field_margin_m=float(field_margin_m), field_voxel_size_m=float(field.voxel_size))

    def evaluate(q):
        """Share this exact rotation's field values and analytic derivatives."""
        nonlocal cached_q, cached, field_seconds
        if cached_q is not None and np.array_equal(q, cached_q):
            return cached
        stamp = perf_counter()
        q = np.asarray(q)
        vector = basis @ q
        increment = Rotation.from_rotvec(vector).as_matrix()
        rotation = rotation0 @ increment
        relative_world = points @ rotation.T
        world = relative_world + origin
        values, known, gradients, errors = field.sample(world)
        # Pinocchio Jexp3 uses right/body differentiation of Exp(vector).
        # World angular velocity therefore includes the current increment.
        angular_axes = rotation @ pin.Jexp3(vector) @ basis
        dphi_dq = np.cross(relative_world, gradients) @ angular_axes
        # Unsupported material has no observed constraint. A constant positive
        # inequality removes that row from SLSQP's active set; raw field values
        # and the support mask remain untouched and separately reported.
        constraints = np.ones(len(points)+1)
        jacobian = np.zeros((len(points)+1, 2))
        constraints[:-1][known] = (values[known]-field_margin_m)/field.voxel_size
        jacobian[:-1][known] = dphi_dq[known]/field.voxel_size
        constraints[-1] = 1.-float(q @ q)/max_tilt_rad**2
        jacobian[-1] = -2*q/max_tilt_rad**2
        cached_q = q.copy()
        cached = dict(q=q.copy(), world=world, values=values, known=known,
            gradients=gradients, errors=errors, dphi_dq=dphi_dq,
            constraints=constraints, jacobian=jacobian, rotation=rotation)
        seconds = perf_counter()-stamp
        field_seconds += seconds
        report = guide_report(cached)
        history['evaluations'].append(dict(index=len(history['evaluations']),
            q=q.copy(), angle_rad=float(np.linalg.norm(q)), objective=.5*float(q@q),
            known_count=report['known_count'], unknown_count=report['unknown_count'],
            worst_sample=report['worst_sample'], tilt_constraint=float(constraints[-1]),
            elapsed_seconds=seconds))
        return cached

    def iteration(q):
        """Retain numerical iterations independently of entity classification."""
        sample = evaluate(q)
        history['iterations'].append(dict(index=len(history['iterations']),
            q=np.asarray(q).copy(), angle_rad=float(np.linalg.norm(q)),
            objective=.5*float(np.asarray(q)@q),
            evaluation_index=len(history['evaluations'])-1,
            minimum_normalized_constraint=float(sample['constraints'].min())))

    zero = np.zeros(2)
    initial = evaluate(zero)
    guide_before = guide_report(initial)
    optimization_seconds = 0.
    after_seconds = 0.
    if before_status == 'intersection':
        optimization_started = perf_counter()
        solved = minimize(lambda q: .5*float(q@q), zero, jac=lambda q: q,
            method='SLSQP',
            constraints=dict(type='ineq', fun=lambda q: evaluate(q)['constraints'],
                             jac=lambda q: evaluate(q)['jacobian']),
            callback=iteration)
        optimization_seconds = perf_counter()-optimization_started
        q = np.asarray(solved.x).copy()
        final = evaluate(q)
        proposed = original.copy()
        proposed[:3, :3] = final['rotation']
        optimizer = dict(called=True, success=bool(solved.success), status=int(solved.status),
            message=str(solved.message), iterations=int(solved.nit),
            function_evaluations=int(solved.nfev), jacobian_evaluations=int(solved.njev),
            objective=float(solved.fun), method='SLSQP with SciPy defaults and analytic derivatives',
            field_constraints_applied=True,
            maximum_sampled_constraint_violation=float(max(0., -final['constraints'][:-1].min())),
            tilt_bound_excess_rad=float(max(0., np.linalg.norm(q)-max_tilt_rad)))
        after_started = perf_counter()
        after_checks = _check_parts(proposed, parts, field)
        after_seconds = perf_counter()-after_started
        proposal_status = 'optimizer_proposal'
    else:
        q = zero
        final = initial
        proposed = original.copy()
        after_checks = before_checks
        proposal_status = ('unchanged_clear_baseline' if before_status == 'clear_represented_set'
                           else 'unchanged_unresolved_baseline')
        optimizer = dict(called=False, success=None, status=None, message=proposal_status,
            iterations=0, function_evaluations=0, jacobian_evaluations=0,
            objective=0., method='not_invoked', field_constraints_applied=False,
            maximum_sampled_constraint_violation=None, tilt_bound_excess_rad=0.)
    after_status = _relation(after_checks)
    angle = float(np.linalg.norm(q))
    return dict(status=after_status, proposal_status=proposal_status,
        T_world_tool_original=original, T_world_tool_proposed=proposed, q=q,
        angle_rad=angle, angle_deg=float(np.rad2deg(angle)),
        basis_tool=basis.copy(), axis_tool=axis, rotation_vector_tool=basis@q,
        max_tilt_rad=float(max_tilt_rad), field_margin_m=float(field_margin_m),
        before_status=before_status, after_status=after_status,
        before_checks=before_checks, after_checks=after_checks,
        guide_before=guide_before, guide_after=guide_report(final),
        optimizer=optimizer, history=history,
        timings=dict(before_checks_seconds=before_seconds,
            optimization_seconds=optimization_seconds, after_checks_seconds=after_seconds,
            guide_evaluation_seconds=field_seconds,
            guide_evaluations=len(history['evaluations']), total_seconds=perf_counter()-started,
            scope='refine_pose only; geometry and background field are already constructed'),
        quality_status='not_evaluated_by_geometric_refinement',
        physical_status='missing_not_executed',
        scope='fixed TCP origin, two-axis local swing, observed background and official robot entities')

# END fixed-origin orientation refinement over observed TSDF support.
