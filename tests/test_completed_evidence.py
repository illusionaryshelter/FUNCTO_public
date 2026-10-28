"""Completed evidence retains values, ownership and one archive per record."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "planning"))

from manipulation.completed_evidence import completed_evidence
from manipulation.evidence_archive import read_evidence_json
from manipulation.execution_binding import plain
from recorded_task_planning import _write_diagnostic_json


def test_complete_record_detaches_once_and_is_shared_by_later_snapshots():
    source = {"intervals": [{"bounds": np.array([.1, .3]), "status": "unresolved"}]}
    evidence = completed_evidence(source)
    source["intervals"][0]["bounds"][:] = -1.
    assert evidence == {"intervals": [{"bounds": [.1, .3], "status": "unresolved"}]}
    assert deepcopy(evidence) is plain(evidence) is completed_evidence(evidence) is evidence
    copies = deepcopy({"export": evidence, "candidate": evidence})
    assert copies["export"] is copies["candidate"] is evidence
    with pytest.raises(TypeError, match="read-only"):
        evidence["intervals"][0]["status"] = "proved"
    with pytest.raises(TypeError, match="read-only"):
        evidence["intervals"].append({})
    with pytest.raises(TypeError, match="read-only"):
        evidence["intervals"][0]["bounds"][0] = 1.


def test_nonfinite_diagnostics_keep_original_formal_conversion():
    record = completed_evidence({"positive": float("inf"), "negative": float("-inf")})
    assert np.isinf(record["positive"])
    formal = plain(record)
    assert formal == {"positive": {"unbounded": "+"}, "negative": {"unbounded": "-"}}
    assert plain(record) is formal
    with pytest.raises(ValueError, match="NaN"):
        plain(completed_evidence({"failure": float("nan")}))
    assert plain(completed_evidence({1: {2: 3.}})) == {"1": {"2": 3.}}


def test_multiple_reports_archive_one_complete_record_without_merging_distinct_records(tmp_path):
    record = completed_evidence({"pairs": [{"entity": "robot/finger", "distances": [.01, -.02]}],
                                 "status": "rejected"})
    same_values = completed_evidence(dict(record))
    original = {"returned": record, "candidates": [record, same_values]}
    _write_diagnostic_json(tmp_path / "planning-report.json", original)
    evidence_file = tmp_path / "evidence" / f"{record.record_id}.json"
    written = evidence_file.stat().st_mtime_ns
    _write_diagnostic_json(tmp_path / "trajectory.json", {"validation": record})
    assert evidence_file.stat().st_mtime_ns == written
    assert len(list((tmp_path / "evidence").glob("*.json"))) == 2
    manifest = json.loads((tmp_path / "planning-report.json").read_text())
    assert manifest["returned"] == {"$ref": f"evidence/{record.record_id}.json"}
    restored = read_evidence_json(tmp_path / "planning-report.json")
    assert restored == original
    assert restored["returned"] is restored["candidates"][0]
    assert restored["candidates"][1] is not restored["returned"]
    with pytest.raises(TypeError, match="read-only"):
        restored["returned"]["pairs"][0]["distances"].clear()
    total = sum(path.stat().st_size for path in (tmp_path / "planning-report.json",
        evidence_file, tmp_path / "evidence" / f"{same_values.record_id}.json"))
    assert read_evidence_json(tmp_path / "planning-report.json", max_bytes=total) == original
    with pytest.raises(ValueError, match="byte limit"):
        read_evidence_json(tmp_path / "planning-report.json", max_bytes=total-1)


def test_immutable_observation_arrays_encode_once_and_keep_binding_headers_independent(monkeypatch, tmp_path):
    import manipulation.observed_mapping as mapping
    from manipulation.planning_scene import ObservedSpace
    from manipulation.collision_records import decode_array
    values = np.arange(8, dtype=np.float32).reshape(2, 2, 2)-4
    observed = np.ones_like(values, dtype=bool)
    weights = np.ones_like(values)
    field = mapping.FrozenObservedField(values, observed, weights=weights,
        origin=[0., 0., 0.], voxel_size=.01, name="observation", source="test RGB-D")
    original, encoded = mapping.encode_array, []

    def encode(value):
        encoded.append(value)
        return original(value)

    monkeypatch.setattr(mapping, "encode_array", encode)
    first, second = field.record(), field.clone_for_rollout().record()
    assert len(encoded) == 3
    for name, array in (("values", values), ("observed", observed), ("weights", weights)):
        assert first[name] is second[name]
        np.testing.assert_array_equal(decode_array(first[name]), array)
    first["origin"][0] = 10.
    assert second["origin"] == [0., 0., 0.]
    coverage = ObservedSpace(observed, origin=[0., 0., 0.], voxel_size=.01,
                             observation_ids=["view"], reference_frame="world")
    a, b = coverage.record(), coverage.clone_for_rollout().record()
    assert a["known_free"] is b["known_free"]
    np.testing.assert_array_equal(decode_array(a["known_free"]), observed)
    _write_diagnostic_json(tmp_path / "scene.json", {"field": second, "coverage": a})
    restored = read_evidence_json(tmp_path / "scene.json")
    rebuilt = mapping.FrozenObservedField.from_record(restored["field"])
    np.testing.assert_array_equal(rebuilt.values, field.values)
    np.testing.assert_array_equal(rebuilt.observed, field.observed)
    np.testing.assert_array_equal(rebuilt.weights, field.weights)
