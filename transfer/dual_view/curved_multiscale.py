"""Three-scale measured relations for the explicitly selected surface model.

No files, model features, target annotations or poses are read by these
functions. The correspondence assignment is frozen before its score interval
is computed. An interval bounds this operational score, not pose probability.
"""

# BEGIN surface-only multiscale relation objective and uncertainty bounds.
import numpy as np
from scipy.optimize import linear_sum_assignment


def prepare_multiscale(source):
    """Freeze regional relation edges and three source-only support domains.

    The center determines support domains only; it is not a target anchor,
    sampled contact, or the demonstration's functional coordinate frame. Every
    relation connects two frozen query rows and touches the demonstration.
    Full observed source geometry determines the radii, while the existing
    query budget determines which independent measured rows are scored.
    """
    xyz = np.asarray(source['query_xyz'], float)
    region = np.asarray(source['query_region'], bool)
    center = np.asarray(source['center'], float)
    entity = np.asarray(source['entity_xyz'], float)
    entity_region = np.asarray(source['entity_region'], bool)
    local = np.concatenate((entity[entity_region], xyz[region]))
    local_radius = float(np.linalg.norm(local-center, axis=1).max())
    whole_radius = max(local_radius,
        float(np.linalg.norm(entity-center, axis=1).max()))
    middle_radius = np.sqrt(local_radius*whole_radius)
    distance = np.linalg.norm(xyz-center, axis=1)
    domains = np.stack((region,
        region | (distance <= np.nextafter(middle_radius, np.inf)),
        np.ones(len(xyz), bool)))
    first, second = np.meshgrid(np.flatnonzero(region), np.arange(len(xyz)), indexing='ij')
    edges = np.stack((first.ravel(), second.ravel()), axis=1)
    edges = np.unique(np.sort(edges[edges[:, 0] != edges[:, 1]], axis=1), axis=0)
    masks = domains[:, edges[:, 0]] & domains[:, edges[:, 1]]
    return dict(xyz=xyz.copy(), source_position_error_m=np.asarray(
        source['query_position_error_m'], float).copy(), query_region=region.copy(),
        edges=edges, masks=masks, domains=domains,
        center=center.copy(), radii_m=[local_radius, middle_radius, whole_radius],
        names=['local_region', 'regional_neighborhood', 'entity_relative_position'],
        unique_support_domain_count=len({row.tobytes() for row in domains}))


def geometry_correspondences(surface, projections, query_region):
    """Assign supported real observations with capacity one per target element.

    There are at most two measured choices per source row. The assignment
    first maximizes regional matches, then total matches, then minimizes actual
    position residual. For Q rows, integer priorities (Q+1)^2 and Q+1 strictly
    dominate the sum of at most Q residuals normalized into [0,1]. They encode
    a lexicographic rule rather than tuned tradeoff weights. Each row also has
    a dummy, so unavailable edges cannot be selected. A supported row lost to
    capacity remains an unresolved association; it is not a known mismatch.
    """
    support = np.column_stack([p['supported'] for p in projections])
    rows, views = np.nonzero(support)
    observations = np.asarray([projections[view]['observation_indices'][row]
        for row, view in zip(rows, views)], np.int64)
    elements = surface['observation_to_element'][observations]
    unique, columns = np.unique(elements, return_inverse=True)
    residual = np.asarray([projections[view]['residual_m'][row]
        for row, view in zip(rows, views)], float)
    q = len(support)
    cost = np.full((q, len(unique)+q), np.inf)
    cost[:, len(unique):] = 0.
    choices = {}
    for row, column, observation, view, error in zip(rows, columns, observations, views, residual):
        key = (int(row), int(column))
        # Iteration follows declared view order; equality retains the first.
        if key not in choices or error < choices[key][2]:
            choices[key] = (int(observation), int(view), float(error))
    maximum = float(residual.max(initial=0.))
    regional = np.asarray(query_region, bool)
    for (row, column), (_, _, error) in choices.items():
        normalized = error/maximum if maximum > 0 else 0.
        cost[row, column] = -((q+1)**2*int(regional[row])+(q+1)-normalized)
    assigned_rows, assigned_columns = linear_sum_assignment(cost)
    assigned = np.full(q, -1, np.int64)
    real = assigned_columns < len(unique)
    assigned[assigned_rows[real]] = assigned_columns[real]
    matched = assigned >= 0
    selected_view = np.full(len(support), -1, np.int64)
    actual = np.full(len(support), -1, np.int64)
    selected_elements = np.full(len(support), -1, np.int64)
    for row in np.flatnonzero(matched):
        actual[row], selected_view[row], _ = choices[(int(row), int(assigned[row]))]
        selected_elements[row] = unique[assigned[row]]
    target = np.full((len(support), 3), np.nan)
    error = np.full(len(support), np.nan)
    target[matched] = surface['observation_xyz'][actual[matched]]
    error[matched] = surface['observation_position_error_m'][actual[matched]]
    return dict(matched=matched, selected_view=selected_view,
        observation_indices=actual, element_ids=selected_elements,
        target_xyz=target, target_position_error_m=error,
        unknown=~matched, capacity_lost=support.any(1) & ~matched,
        supported_row_count=int(support.any(1).sum()), matched_count=int(matched.sum()),
        capacity_lost_count=int(np.count_nonzero(support.any(1) & ~matched)),
        unknown_count=int((~matched).sum()))


def score_multiscale(context, transform, correspondence, *, registration_error_m):
    """Bound one fixed correspondence rule's three-scale operational objective.

    For an edge i,j, d is the norm of the observed target relation minus the
    candidate's transformed source relation. Its measurement error eta is the
    sum of all four endpoint position bounds, scaling source bounds by s.
    Reverse triangle inequality bounds the unperturbed d by [max(0,d-eta),
    d+eta]. The decreasing similarity g(d)=1/(1+(d/(2*tau))^2) converts this to a
    valid score interval. Tau is the existing registration/model allowance;
    it normalizes the objective and is never included in measurement eta.

    Both missing and capacity-unresolved edges have [0,1]. Every scale uses
    its original edge count, and the geometric mean is monotone in its three
    inputs. Bounds are conditional on the frozen candidate and assignment:
    they do not certify all correspondences, segmentation, or physical pose.
    """
    tau = float(registration_error_m)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('registration_error_m must be finite and positive')
    scale, rotation, _ = transform
    edges = context['edges']
    left, right = edges.T
    matched = np.asarray(correspondence['matched'], bool)
    lost = np.asarray(correspondence['capacity_lost'], bool)
    capacity_unresolved = lost[left] | lost[right]
    measured = matched[left] & matched[right]
    unknown = ~measured
    point = np.zeros(len(edges))
    lower = np.zeros(len(edges))
    upper = unknown.astype(float)
    selected = np.flatnonzero(measured)
    if len(selected):
        a, b = left[selected], right[selected]
        target = correspondence['target_xyz']
        expected = float(scale)*(context['xyz'][b]-context['xyz'][a]) @ np.asarray(rotation).T
        residual = np.linalg.norm((target[b]-target[a])-expected, axis=1)
        source_error = context['source_position_error_m']
        target_error = correspondence['target_position_error_m']
        eta = float(scale)*(source_error[a]+source_error[b])+target_error[a]+target_error[b]
        similarity = lambda value: 1./(1.+(value/(2*tau))**2)
        point[selected] = similarity(residual)
        lower[selected] = similarity(residual+eta)
        upper[selected] = similarity(np.maximum(0., residual-eta))
    scales = []
    for name, radius, domain, mask in zip(context['names'], context['radii_m'],
            context['domains'], context['masks']):
        count = int(mask.sum())
        if count == 0:
            raise ValueError('Each structural scale requires at least one source relation')
        scales.append(dict(name=name, radius_m=float(radius),
            source_row_count=int(domain.sum()), relation_count=count,
            measured_relation_count=int((mask & measured).sum()),
            unknown_relation_count=int((mask & unknown).sum()),
            capacity_unresolved_relation_count=int((mask & capacity_unresolved).sum()),
            observed_score=float(point[mask].sum()/count),
            lower_bound=float(lower[mask].sum()/count),
            upper_bound=float(upper[mask].sum()/count)))
    geometric_mean = lambda key: float(np.prod([item[key] for item in scales])**(1./3.))
    return dict(score=geometric_mean('observed_score'),
        lower_bound=geometric_mean('lower_bound'), upper_bound=geometric_mean('upper_bound'),
        scales=scales, unique_support_domain_count=context['unique_support_domain_count'],
        registration_normalizer_m=2*tau,
        context_scope='whole_observed_source_entity_relations_at_frozen_query_rows',
        interval_scope='measurement_bounds_and_missing_relations_under_fixed_geometric_assignment',
        is_probability=False, independent_scale_evidence=False)

# END surface-only multiscale relation objective and uncertainty bounds.
