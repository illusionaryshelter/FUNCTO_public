"""Construct typed scene-query values from explicit CPU fixture declarations."""

import numpy as np

from manipulation.pair_evidence import PairBatch, EvidenceState


def records(batch):
    """Inspect complete output values at the test's assertion boundary."""
    return [batch.materialize(source, pair) for source, pair in batch.order]


def declared_state(time_s, side, pairs, progress=0., residual=0.):
    """One fixture state explicitly declares every observed/relationship pair."""
    batch = PairBatch()
    batch.bind_clock(np.array([[time_s]]), np.array([[progress]]), np.array([[progress]]))
    for row in pairs:
        payload = {name: value for name, value in row.items() if name not in
            {'component', 'first', 'second', 'geometry_entity_key', 'permission_kind',
             'state_index', 'query_index', 'side', 'time_s'}}
        batch.append_observed((0, 0), row['component'], row['first'], row['second'], payload,
            geometry_entity_key=row['geometry_entity_key'],
            permission_kind=row.get('permission_kind', 'none'))
    batch.seal()
    batch.set_query_layout(np.array([[0, 0]]), np.array([side]))
    return EvidenceState(float(time_s), int(side), float(progress), float(residual), batch, (0,))


def state_record(state):
    """Read a typed fixture state using its full original pair sequence."""
    pairs = []
    for source, pair in state.tokens():
        row = state.batch.materialize(source, pair)
        pairs.append({name: value for name, value in row.items()
                      if name not in {'state_index', 'query_index', 'side', 'time_s'}})
    return dict(time_s=state.time_s, side=state.side, progress=state.progress,
                residuals={'scene': state.residual}, pairs=pairs)
