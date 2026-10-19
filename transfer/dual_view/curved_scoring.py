"""Source-weighted geometry and appearance support for curved candidates.

The source scopes and appearance reference distributions are fixed before
candidate evaluation. Scores describe measured support, never calibrated pose
probability. Missing observations contribute no support and remain explicit.
"""

# BEGIN absolute-position and same-measurement appearance scoring.
import numpy as np
from scipy.optimize import linear_sum_assignment


# A predeclared geometry-first design preference, not estimated uncertainty.
# These coefficients are independent of objects, candidates and trial results.
GEOMETRY_WEIGHT = 2. / 3.
APPEARANCE_WEIGHT = 1. / 3.


def prepare_scoring(source):
    """Give each distinct source support scope equal total query weight.

    Keep the established local-region, regional-neighborhood and whole-entity
    scopes. A query's weight is the mean of 1/|D| over scopes containing it;
    therefore all query weights sum to one. Repeated scopes count once. This
    is a source-demonstration relevance prior, not independent measurements
    or a variance estimate. No candidate can change these weights.
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
    proposed_domains = (region,
        region | (distance <= np.nextafter(middle_radius, np.inf)),
        np.ones(len(xyz), bool))
    names = ('local_region', 'regional_neighborhood', 'entity_relative_position')
    radii = (local_radius, middle_radius, whole_radius)
    domains, aliases, scope_radii, seen = [], [], [], {}
    for name, radius, domain in zip(names, radii, proposed_domains):
        key = domain.tobytes()
        if key in seen:
            aliases[seen[key]].append(name)
        else:
            seen[key] = len(domains)
            domains.append(domain.copy())
            aliases.append([name])
            scope_radii.append(float(radius))
    domains = np.stack(domains)
    weights = (domains / domains.sum(axis=1)[:, None]).mean(axis=0)
    return dict(xyz=xyz.copy(), domains=domains,
        scope_aliases=aliases, radii_m=scope_radii, query_weights=weights,
        unique_support_domain_count=len(domains),
        weighting_definition='equal_total_mass_per_distinct_source_scope')


def geometry_correspondences(surface, projections, query_region):
    """Assign actual observations with capacity one per target element.

    Preserve the existing lexicographic rule: maximize regional matches,
    then total matches, then minimize position residual. For Q rows, integer
    priorities (Q+1)^2 and Q+1 dominate all normalized residuals together.
    They are assignment priorities, not the new scoring coefficients.
    Unassigned rows remain unknown, including those lost to target capacity.
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
        # Declared view order owns exact position-residual ties.
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
    target[matched] = surface['observation_xyz'][actual[matched]]
    return dict(matched=matched, selected_view=selected_view,
        observation_indices=actual, element_ids=selected_elements,
        target_xyz=target,
        unknown=~matched, capacity_lost=support.any(1) & ~matched,
        supported_row_count=int(support.any(1).sum()), matched_count=int(matched.sum()),
        capacity_lost_count=int(np.count_nonzero(support.any(1) & ~matched)),
        unknown_count=int((~matched).sum()))


def assigned_appearance(sampled, correspondence, supports):
    """Calibrate only the observation/view selected by geometric assignment.

    Per-view reference distributions use the same native pixel sampler as
    sampled['view_scores']. The other view cannot lend its response to this
    correspondence, even when its raw dot product is higher. Missing native
    responses or empty reference distributions retain NaN, not a raw zero.
    """
    q = len(correspondence['matched'])
    raw, utility = np.full(q, np.nan), np.full(q, np.nan)
    for view, support in enumerate(supports):
        rows = np.flatnonzero(correspondence['matched']
            & (correspondence['selected_view'] == view))
        raw[rows] = sampled['view_scores'][rows, view]
        utility[rows] = support.utility(raw[rows], rows)
    return dict(raw_responses=raw, utilities=utility)


def score_candidate(context, transform, correspondence, appearance_utilities, *,
                    registration_error_m):
    """Combine absolute-position support and empirical appearance support.

    For a selected measurement, e_i=y_i-(s R x_i+t) and
    g_i=1/(1+(||e_i||/tau)^2). This responds to a common translation error;
    the former pair residual ||e_j-e_i|| could not. Absolute residuals already
    constrain relative shape, so pair scores are not added a second time.
    Tau is the existing single-point compatibility scale, not sensor sigma.

    With source-only weights w summing to one, G=sum(w*g), A=sum(w*a), and
    S=(2G+A)/3. The 2:1 coefficients are an explicit geometry-first preference,
    not calibrated confidence or a claim about each modality's actual ranking
    influence. Neither the source weights nor the coefficients are normalized
    by a candidate's known count. With fixed correspondences and references,
    hiding a contribution cannot increase S. Full reassignment or a rebuilt
    reference distribution is outside that monotonicity statement.
    """
    tau = float(registration_error_m)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('registration_error_m must be finite and positive')
    matched = np.asarray(correspondence['matched'], bool)
    appearance = np.asarray(appearance_utilities, float)
    if appearance.shape != matched.shape or np.isinf(appearance).any():
        raise ValueError('Appearance utilities must align with queries and use NaN for missing values')
    finite = np.isfinite(appearance)
    if np.any(finite & ((appearance < 0.) | (appearance > 1.))):
        raise ValueError('Known appearance utilities must lie in [0,1]')
    scale, rotation, translation = transform
    mapped = float(scale)*context['xyz'] @ np.asarray(rotation).T + translation
    residual = np.full(len(matched), np.nan)
    residual[matched] = np.linalg.norm(
        correspondence['target_xyz'][matched]-mapped[matched], axis=1)
    geometry = np.zeros(len(matched))
    geometry[matched] = 1. / (1. + (residual[matched]/tau)**2)
    known = matched & finite
    appearance_contribution = np.where(known, appearance, 0.)
    weights = context['query_weights']
    g = float(weights @ geometry)
    a = float(weights @ appearance_contribution)
    scopes = []
    for aliases, radius, domain in zip(context['scope_aliases'],
            context['radii_m'], context['domains']):
        scopes.append(dict(names=list(aliases), radius_m=radius,
            source_query_count=int(domain.sum()),
            matched_query_count=int((domain & matched).sum()),
            appearance_known_query_count=int((domain & known).sum()),
            geometry_score=float(geometry[domain].mean()),
            appearance_score=float(appearance_contribution[domain].mean())))
    return dict(score=GEOMETRY_WEIGHT*g+APPEARANCE_WEIGHT*a,
        geometry_score=g, appearance_score=a,
        geometry_coefficient=GEOMETRY_WEIGHT, appearance_coefficient=APPEARANCE_WEIGHT,
        weighted_geometry_score=GEOMETRY_WEIGHT*g,
        weighted_appearance_score=APPEARANCE_WEIGHT*a,
        query_weights=weights.tolist(), geometry_support=geometry.tolist(),
        absolute_position_residuals_m=[float(x) if np.isfinite(x) else None for x in residual],
        appearance_support=[float(x) if valid else None for x, valid in zip(appearance, known)],
        weighted_geometry_contributions=(GEOMETRY_WEIGHT*weights*geometry).tolist(),
        weighted_appearance_contributions=(APPEARANCE_WEIGHT*weights*appearance_contribution).tolist(),
        geometry_known_weight=float(weights[matched].sum()),
        geometry_unknown_weight=float(weights[~matched].sum()),
        appearance_known_weight=float(weights[known].sum()),
        appearance_unknown_weight=float(weights[~known].sum()),
        geometry_normalizer_m=tau, source_scopes=scopes,
        unique_support_domain_count=context['unique_support_domain_count'],
        source_weighting=context['weighting_definition'],
        coefficient_basis='predeclared_geometry_first_design_preference_not_calibrated_reliability',
        is_probability=False)
# END one scoring path; no pair-score or missing-appearance fallback.
