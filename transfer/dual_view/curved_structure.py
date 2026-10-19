"""Observed entity evidence for the explicitly selected surface model.

Every source entity measurement is checked under the same proposed Sim(3).
Missing/occluded measurements stay unknown. These full-entity counts describe
source measurement rows; they are not independent correspondence votes.
Query correspondence capacity belongs to the separate multiscale module.
"""

import numpy as np

from ..sim3 import apply_sim3
from .curved_geometry import project_measurements


# BEGIN full-entity evidence supplies refutation and visibility, not another fit.
def entity_evidence(source, surface, transform, registration_error_m):
    """Summarize every observed source row under the same proposed transform.

    Support in either camera counts a source row once, even when both cameras
    see it. A conflict in either camera remains a conflict despite support in
    the other. Unknown means neither support nor conflict. Supported rows may
    share target measurements, so their number does not supply independent
    evidence or a fitting gate. No appearance threshold or matching is used.
    """
    mapped = apply_sim3(source['entity_xyz'], *transform)
    projections = [project_measurements(surface, mapped, view,
        registration_error_m=registration_error_m,
        query_position_error_m=transform[0]*source['entity_position_error_m'])
        for view in range(2)]
    support = np.column_stack([p['supported'] for p in projections])
    conflict = np.column_stack([p['conflict'] for p in projections]).any(1)
    known = support.any(1)
    context = ~source['entity_region']
    return dict(status='evaluated', source_entity_count=len(mapped),
        source_context_count=int(context.sum()), support_count=int(known.sum()),
        conflict_count=int(conflict.sum()), unknown_count=int((~(known | conflict)).sum()),
        context_support_count=int(np.count_nonzero(known & context)),
        context_conflict_count=int(np.count_nonzero(conflict & context)),
        context_unknown_count=int(np.count_nonzero(~(known | conflict) & context)),
        support_count_unit='source_measurement_rows_not_independent_correspondences',
        contradicted_source_indices=source['entity_indices'][conflict].tolist(),
        support_and_conflict_count=int(np.count_nonzero(known & conflict)))
# END original source identities survive without a redundant context assignment.


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
