"""Complete observed payload, absence and relation-side semantics."""
import numpy as np
from manipulation.pair_evidence import PairBatch, UNRESOLVED, FINITE_REGION


def test_variable_observed_payload_presence_nan_and_relation_side_are_lossless():
    batch = PairBatch()
    times = np.array([[1., 1.]])
    progress = np.array([[.5, np.nextafter(.5, 0.)]])
    permission = np.array([[.5, .5]])
    batch.bind_clock(times, progress, permission)
    payloads = [dict(status="unresolved", lower_m=float("nan"), upper_m=0.,
                    full_entity_lower_m=None, witness_first=[1., 2., 3.], witness_second=[4., 5., 6.],
                    witness_first_bounds=[[.9, 1.9, 2.9], [1.1, 2.1, 3.1]],
                    witness_coordinates="interval_midpoints_for_display_and_derivative",
                    queries=7, model_extra={"support_cells": [2, 9, 12]}),
                dict(status="internal_assembly", lower_m=None)]
    entity = {"first": "robot/solid", "second": "owner/observed"}
    for s, payload in enumerate(payloads):
        batch.append_observed((0, s), "collision/robot/observed", "solid", "observed", payload,
            geometry_entity_key=entity,
            permission_kind="finite_region" if s == 0 else "declared_rigid_assembly")
    batch.seal()
    batch.set_query_layout([[0, 0], [0, 0]], [0, -1])
    assert batch.observed_payloads[0][0] is payloads[0]
    assert batch.status[0, 0] == UNRESOLVED
    assert batch.permission_kind[0, 0] == FINITE_REGION
    assert batch.lower_valid[0, 0] and np.isnan(batch.lower[0, 0])
    assert batch.upper_valid[0, 0] and batch.upper[0, 0] == 0.
    assert not batch.full_lower_valid[0, 0] and not batch.lower_valid[1, 0]
    row = batch.materialize(0, 0)
    assert row["model_extra"] == payloads[0]["model_extra"]
    assert row["witness_first_bounds"] == payloads[0]["witness_first_bounds"]
    assert "full_entity_lower_m" in row and row["full_entity_lower_m"] is None
    assert "upper_m" not in batch.materialize(1, 0)
    assert batch.materialize(1, 0)["query_index"] == [0, 0]
    assert batch.materialize(1, 0)["side"] == -1
    assert "side" not in batch.materialize(1, 0, query_layout=False)



# BEGIN original payload identity and physical state grouping are observable semantics.
def test_shared_nan_keeps_original_tuple_ranking_in_both_consumer_orders():
    from manipulation.pair_evidence import _geometry_query_states, selected_witness_records
    batch = PairBatch()
    times = np.array([[1., 0.]])
    batch.bind_clock(times, times, times)
    shared_nan = float('nan')
    for index in range(2):
        batch.append_observed((0, index), 'collision/fixture', 'body', 'field',
            dict(status='unresolved', lower_m=shared_nan, upper_m=.2,
                 witness_first=[1., 2., 3.], witness_second=[4., 5., 6.], ordinal=index),
            geometry_entity_key={'first':'robot/body', 'second':'owner/field'})
    batch.seal()
    batch.set_query_layout(batch.state_index, [0, 0])
    states = _geometry_query_states(batch, [1., 0.], [1., 0.], [0., 0.])
    for kwargs in (dict(batch=batch), dict(states=states)):
        selected, total = selected_witness_records(**kwargs)
        assert total == 2 and len(selected) == 1
        assert selected[0]['ordinal'] == 1 and selected[0]['time_s'] == 0.
        assert selected[0]['first'] == 'robot/body'


def test_pair_free_queries_keep_exact_physical_states_and_residuals():
    from manipulation.pair_evidence import _geometry_query_states
    batch = PairBatch()
    times = np.array([[0., .5, .5, .5]])
    batch.bind_clock(times, times, times)
    batch.seal()
    batch.set_query_layout([[0, 0], [0, 1], [0, 1], [0, 1]], [0, 0, -1, 1])
    states = _geometry_query_states(batch, [0., .5], [0., 1.], [0., .2])
    assert [(s.time_s, s.side, s.progress, s.residual) for s in states] == [(0., 0, 0., 0.), (.5, 0, 1., .2)]
    assert all(not s.source_rows and not s.pair_columns for s in states)
# END unknown observations and empty pair sets remain distinct values.
