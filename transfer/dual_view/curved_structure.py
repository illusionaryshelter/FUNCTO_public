"""Observed entity evidence for the explicitly selected surface model.

Every source entity measurement is checked under the same proposed Sim(3).
Missing/occluded measurements stay unknown. A sparse maximum-cardinality
matching prevents many source measurements from manufacturing independent
support at one target element. Neither its count nor finite-pool agreement is
a proof that unobserved structure or a functional identity is correct.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from ..sim3 import apply_sim3
from .curved_geometry import project_measurements


def entity_evidence(source, surface, transform, registration_error_m):
    """Check the entire observed source entity, with O(N) association storage.

    There are at most two supported edges per source row, one per camera.
    Fusion already defines the target element capacity. Sparse matching only
    selects among these measured edges; it never searches for another point,
    fills a missing correspondence, or uses an appearance threshold.
    """
    mapped = apply_sim3(source['entity_xyz'], *transform)
    projections = [project_measurements(surface, mapped, view,
        registration_error_m=registration_error_m,
        query_position_error_m=transform[0]*source['entity_position_error_m'])
        for view in range(2)]
    support = np.column_stack([p['supported'] for p in projections])
    conflict = np.column_stack([p['conflict'] for p in projections]).any(1)
    known = support.any(1)
    context_rows = np.flatnonzero(~source['entity_region'])
    edge_rows, edge_views = np.nonzero(support[context_rows])
    observations = np.array([projections[view]['observation_indices'][context_rows[row]]
        for row, view in zip(edge_rows, edge_views)], np.int64)
    elements = surface['observation_to_element'][observations]
    domain, columns = np.unique(elements, return_inverse=True)
    graph = coo_matrix((np.ones(len(columns)), (edge_rows, columns)),
        shape=(len(context_rows), len(domain))).tocsr()
    assignment = maximum_bipartite_matching(graph, perm_type='column')
    selected_rows = np.flatnonzero(assignment >= 0)
    selected_observations, selected_views = [], []
    # Each row has at most two choices. The first declared camera owns an
    # exact element tie, consistently with the ordinary appearance assignment.
    row_edges = {}
    for row, column, observation, view in zip(edge_rows, columns, observations, edge_views):
        row_edges.setdefault((int(row), int(column)), (int(observation), int(view)))
    for row in selected_rows:
        observation, view = row_edges[(int(row), int(assignment[row]))]
        selected_observations.append(observation)
        selected_views.append(view)
    matched_rows = context_rows[selected_rows]
    matched_observations = np.asarray(selected_observations, np.int64)
    context = ~source['entity_region']
    summary = dict(status='evaluated', source_entity_count=len(mapped),
        source_context_count=int(context.sum()), support_count=int(known.sum()),
        conflict_count=int(conflict.sum()), unknown_count=int((~(known | conflict)).sum()),
        context_support_count=int(np.count_nonzero(known & context)),
        context_conflict_count=int(np.count_nonzero(conflict & context)),
        context_unknown_count=int(np.count_nonzero(~(known | conflict) & context)),
        distinct_context_correspondence_count=len(matched_rows),
        matching='sparse_maximum_cardinality_actual_observation_edges',
        context_source_indices=source['entity_indices'][matched_rows].tolist(),
        context_observation_indices=matched_observations.tolist(),
        context_view_indices=selected_views,
        contradicted_source_indices=source['entity_indices'][conflict].tolist(),
        minimum_evidence_is_identity_certificate=False)
    return summary, source['entity_xyz'][matched_rows], surface['observation_xyz'][matched_observations]


def compare_transforms(source, records):
    """Compare all admitted finite estimates on the same source entity rows.

    The triangle inequality motivates adding two declared registration bounds
    and their scaled source measurement bounds. These bounds describe this
    estimator's admissible spatial disagreement, not calibrated pose errors.
    Pairwise checks are necessary: agreement with one reference alone is not
    transitive. A small score gap, a shared graph seed, and an unseen source
    point cannot erase a disagreeing finite estimate.
    """
    disagreements, largest_ratio = [], 0.
    for left in range(len(records)):
        mapped_left = apply_sim3(source['entity_xyz'], *records[left]['transform'])
        for right in range(left+1, len(records)):
            a, b = records[left], records[right]
            bounds = (a['summary']['registration_error_budget_m']
                + b['summary']['registration_error_budget_m']
                + (a['transform'][0]+b['transform'][0])*source['entity_position_error_m'])
            mapped_right = apply_sim3(source['entity_xyz'], *b['transform'])
            distance = np.linalg.norm(mapped_left-mapped_right, axis=1)
            ratio = np.divide(distance, bounds, out=np.full_like(distance, np.inf), where=bounds > 0)
            ratio[(bounds == 0) & (distance == 0)] = 0.
            largest_ratio = max(largest_ratio, float(ratio.max(initial=0)))
            if np.any(distance > np.nextafter(bounds, np.inf)):
                row = int(np.argmax(ratio))
                disagreements.append(dict(
                    hypothesis_ids=[a['summary']['hypothesis_id'], b['summary']['hypothesis_id']],
                    maximum_disagreement_m=float(distance.max(initial=0)),
                    worst_source_index=int(source['entity_indices'][row]),
                    worst_distance_m=float(distance[row]), declared_bound_m=float(bounds[row])))
    return dict(agree=not disagreements, compared_candidate_count=len(records),
        compared_pair_count=len(records)*(len(records)-1)//2,
        maximum_disagreement_to_declared_bound=largest_ratio,
        disagreements=disagreements, comparison_domain='all_actual_source_entity_measurements',
        agreement_is_calibrated_confidence=False, global_uniqueness_certified=False)
