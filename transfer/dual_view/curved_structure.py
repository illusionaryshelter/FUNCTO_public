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
# The former pair-budget comparison was removed: score selection has no
# cross-candidate error allowance. Raw pose differences are diagnostics only.
