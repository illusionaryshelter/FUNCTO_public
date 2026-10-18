"""Bounded one-to-one observed correspondence proposals for dual-view transfer.

These fits enlarge the finite transform pool. They do not certify feature
identity, scale, rotation, or the final local region. One assignment is frozen
before fitting its region-only and full subsets, and no nearest-point update
is performed after either fit. All candidates need the normal final evaluator.
"""

from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ..sim3 import apply_sim3, fit_sim3, rank
from .proposals import COMPATIBILITY_RATIO, valid_transform


# BEGIN fixed-support partial correspondence refinement without camera snapping.
def partial_refits(source_pack, surface, domain, proposal):
    """Return up to two fits from one optional, capacity-one assignment.

    A real edge costs ``1-u + distance/tolerance`` and each source row has an
    independent dummy opportunity of cost one. Only strictly cheaper real
    edges are feasible. Unknown appearance, incompatible known normals and
    distant points cannot be forced into a fit to reach a minimum pair count.
    Full-domain utilities and the spatial tree are reused for every proposal.
    """
    started = perf_counter()
    queries = np.asarray(source_pack['query_xyz'], dtype=float)
    regional = np.asarray(source_pack['query_region'], dtype=bool)
    transform = proposal['transform']
    tolerance = COMPATIBILITY_RATIO * float(source_pack['size']) * transform[0]
    diagnostics = {'parent_proposal_id': proposal['proposal_id'],
        'tolerance_m': float(tolerance), 'candidate_union_count': 0,
        'distance_pair_count': 0, 'feasible_edge_count': 0,
        'assigned_query_count': 0, 'assigned_region_count': 0,
        'assignment_seconds': 0., 'fit_seconds': 0., 'fits': []}
    candidates = []

    def finish():
        diagnostics['refinement_seconds'] = perf_counter() - started
        return candidates, diagnostics

    if (len(queries) < 6 or domain['tree'] is None
            or not np.isfinite(tolerance) or tolerance <= 0):
        diagnostics['unavailable_reason'] = 'insufficient_input_for_partial_assignment'
        return finish()

    mapped = apply_sim3(queries, *transform)
    balls = domain['tree'].query_ball_point(mapped,
        np.nextafter(tolerance, np.inf), workers=1)
    nonempty = [np.asarray(ids, dtype=np.int64) for ids in balls if len(ids)]
    if not nonempty:
        diagnostics['unavailable_reason'] = 'no_entity_measurement_in_fixed_search_radius'
        return finish()
    columns = np.unique(np.concatenate(nonempty))
    target_ids = domain['surface_ids'][columns]
    target = domain['xyz'][columns]
    diagnostics.update(candidate_union_count=len(columns),
                       distance_pair_count=int(len(queries) * len(columns)))

    assignment_started = perf_counter()
    # cdist avoids a temporary [Q,N,3] tensor; only the union of exact spatial
    # neighborhoods is evaluated. Tree inclusion cannot relax the explicit gate.
    distance = cdist(mapped, target)
    utility = np.asarray(domain['utility'][:, columns], dtype=float)
    known = domain['inverse'][:, columns] >= 0
    cost = 1. - utility + distance / tolerance
    feasible = known & (distance <= tolerance) & (cost < 1.)
    # BEGIN geometry-specific evidence: curved correspondences do not require
    # normals. The planar arithmetic/order below remains exactly unchanged.
    geometry = source_pack['region_geometry']
    if geometry == 'planar':
        normals = np.asarray(source_pack['query_normals'], dtype=float) @ transform[1].T
        normal_valid = np.asarray(source_pack['query_normal_valid'], dtype=bool)
        agreement = normals @ np.asarray(surface['normals'][target_ids], dtype=float).T
        feasible &= ~normal_valid[:, None] | (agreement >= surface['normal_min_agreement'])
    elif geometry != 'surface':
        raise ValueError('Unknown source region_geometry')
    # END no invented normal or failure fallback for the curved path.
    cost[~feasible] = np.inf
    diagnostics['feasible_edge_count'] = int(feasible.sum())

    # There are Q dummy columns, rather than Q-6. Thus even every row can remain
    # unmatched; six usable pairs must emerge from evidence, never capacity.
    extended = np.column_stack((cost, np.ones((len(queries), len(queries)))))
    rows, assigned_columns = linear_sum_assignment(extended)
    real = assigned_columns < len(columns)
    rows, assigned_columns = rows[real], assigned_columns[real]
    matched_ids = target_ids[assigned_columns]
    matched_target = target[assigned_columns]
    diagnostics.update(assignment_seconds=perf_counter() - assignment_started,
        assigned_query_count=len(rows), assigned_region_count=int(regional[rows].sum()),
        assigned_query_rows=rows.tolist(), assigned_surface_ids=matched_ids.tolist())

    fit_started = perf_counter()
    for scope, use in (('region', regional[rows]), ('all', np.ones(len(rows), bool))):
        fixed_rows, fixed_target, fixed_ids = rows[use], matched_target[use], matched_ids[use]
        record = {'scope': scope, 'fixed_support_count': len(fixed_rows),
                  'fixed_region_count': int(regional[fixed_rows].sum()), 'retained': False}
        diagnostics['fits'].append(record)
        if len(fixed_rows) < 6 or not regional[fixed_rows].any():
            record['reason'] = 'fewer_than_six_pairs_or_no_region_pair'
            continue
        if min(rank(queries[fixed_rows])[0], rank(fixed_target)[0]) < 2:
            record['reason'] = 'fixed_support_rank_below_two'
            continue
        try:
            fitted = fit_sim3(queries[fixed_rows], fixed_target)
        except (ValueError, np.linalg.LinAlgError):
            record['reason'] = 'fixed_support_fit_failed'
            continue
        if not valid_transform(fitted):
            record['reason'] = 'fitted_transform_outside_declared_domain'
            continue
        residuals = np.linalg.norm(apply_sim3(queries[fixed_rows], *fitted)-fixed_target, axis=1)
        supported = residuals <= tolerance
        final_rows = fixed_rows[supported]
        record.update(final_support_count=len(final_rows),
            final_region_count=int(regional[final_rows].sum()),
            fixed_support_rms_m=float(np.sqrt(np.mean(residuals**2))))
        if (len(final_rows) < 6 or not regional[final_rows].any()
                or min(rank(queries[final_rows])[0], rank(fixed_target[supported])[0]) < 2):
            record['reason'] = 'insufficient_final_fixed_support'
            continue
        provenance = {'origin': f'one_partial_{scope}_refit',
            'parent_proposal_id': proposal['proposal_id'],
            'query_rows': fixed_rows.tolist(), 'surface_ids': fixed_ids.tolist(),
            'final_query_rows': final_rows.tolist(),
            'fixed_residuals_m': residuals.tolist(), 'tolerance_m': float(tolerance)}
        candidates.append({'transform': fitted, 'origins': [provenance]})
        record['retained'] = True
    diagnostics['fit_seconds'] = perf_counter() - fit_started
    return finish()
# END fixed-support partial correspondence refinement.
