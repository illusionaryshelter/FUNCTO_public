"""Small lifecycle checks for full source residency and pre-allocation refusal."""

# BEGIN distinguish immutable source ownership from query and raw-reader work.
import gc
from unittest.mock import Mock
import weakref

import numpy as np
import pytest

from manipulation.contact_geometry import ObservedContactGeometry
from manipulation import tsdf_geometry
from manipulation.resource_admission import ResourceLimitError
from manipulation.tsdf_geometry import ObservedTsdfGeometry, iter_source_components
from test_tsdf_query_optimization import component_record


def test_pending_residency_counts_shared_maps_once_and_releases_retired_maps():
    # Collect retired scenes from preceding tests before measuring this
    # lifecycle; the later collection must only remove this test's first map.
    gc.collect()
    before = ObservedTsdfGeometry.pending_storage_total_nbytes()
    first = ObservedTsdfGeometry(component_record([[[x, 0., 0.]] for x in (0., 1., 2.)]))
    shared = first
    reference = weakref.ref(first)
    assert ObservedTsdfGeometry.pending_storage_total_nbytes()-before == 3*4096
    first.collision_component(1)
    assert shared.pending_storage_nbytes == 2*4096
    second = ObservedTsdfGeometry(component_record([[[0., 1., 0.]], [[1., 1., 0.]]]))
    assert ObservedTsdfGeometry.pending_storage_total_nbytes()-before == 4*4096
    del first, shared
    gc.collect()
    assert reference() is None
    assert ObservedTsdfGeometry.pending_storage_total_nbytes()-before == second.pending_storage_nbytes


def test_capacity_refusal_precedes_source_payload_decoding(monkeypatch):
    record = component_record([[[0., 0., 0.]]])
    error = ResourceLimitError("host_process_rss", 11*1024**3, 8*1024**3)
    admission = Mock(side_effect=error)
    decode = Mock(side_effect=AssertionError("A refused map must not decode its source arrays"))
    monkeypatch.setattr(tsdf_geometry, "require_host_memory", admission)
    monkeypatch.setattr(tsdf_geometry, "decode_array", decode)
    with pytest.raises(ResourceLimitError) as caught:
        ObservedTsdfGeometry(record)
    assert caught.value is error
    assert admission.call_args.args[0] >= 4096
    decode.assert_not_called()


def test_contact_vertex_reader_uses_identical_sources_without_a_native_map(monkeypatch):
    record = component_record([[[0., 0., 0.]], [[1., 0., 0.], [1., .2, 0.], [1., 0., .2]]],
        [[[2., 0., 0.], [2.2, 0., 0.], [2., .2, 0.], [2., 0., .2]]], cells=(0, 1))
    record.update(name="source", pose=[0., 0., 0., 1., 0., 0., 0.], source="complete synthetic signed field")
    native = ObservedTsdfGeometry(record)
    expected = list(native.iter_components())
    raw = list(iter_source_components(record))
    assert [row[:2] for row in raw] == [row[:2] for row in expected]
    for actual, wanted in zip(raw, expected):
        np.testing.assert_array_equal(actual[2], wanted[2])
    before = ObservedTsdfGeometry.pending_storage_total_nbytes()
    construction = Mock(side_effect=AssertionError("Raw contact vertices do not need a second collision map"))
    monkeypatch.setattr(ObservedTsdfGeometry, "__init__", construction)
    contact = ObservedContactGeometry({"solids": [record], "query_gaps": []})
    assert len(contact.vertices) == len(expected)
    for actual, (_, _, wanted) in zip(contact.vertices, expected):
        np.testing.assert_array_equal(actual, wanted)
    assert ObservedTsdfGeometry.pending_storage_total_nbytes() == before
    construction.assert_not_called()
# END no large native inventory, CUDA allocation or full planner is needed.
