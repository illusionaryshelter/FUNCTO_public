"""CPU regressions for cold-history native distance batches and full reports."""

# BEGIN exact floating-point and lifecycle checks for native many-pair queries.
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
import struct
import sys
from unittest.mock import patch

import coal
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import manipulation.robot_geometry_check as module
from manipulation.robot_geometry_check import NativePrimitive, NativeSolid, PairQuery, RobotGeometryChecker


URDF = """<robot name="translation_probe"><link name="base"/><link name="x"/>
<link name="y"/><link name="z"/>
<joint name="qx" type="prismatic"><parent link="base"/><child link="x"/>
<axis xyz="1 0 0"/><limit lower="-4" upper="4" effort="1" velocity="1"/></joint>
<joint name="qy" type="prismatic"><parent link="x"/><child link="y"/>
<axis xyz="0 1 0"/><limit lower="-4" upper="4" effort="1" velocity="1"/></joint>
<joint name="qz" type="prismatic"><parent link="y"/><child link="z"/>
<axis xyz="0 0 1"/><limit lower="-4" upper="4" effort="1" velocity="1"/></joint>
</robot>"""


def assert_bits_equal(expected, actual):
    """Compare complete report structure and every binary64 word, including -0."""
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_bits_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(expected, actual):
            assert_bits_equal(left, right)
    elif isinstance(expected, float):
        assert struct.pack("=d", actual) == struct.pack("=d", expected)
    else:
        assert actual == expected


def scalar_distances(checker, pairs, poses):
    """Independent reference uses Coal's original per-call fresh solver API."""
    transforms = {name: coal.Transform3s(pose[:3, :3], pose[:3, 3]) for name, pose in poses.items()}
    for pair in pairs:
        result = coal.DistanceResult()
        coal.distance(checker._geometries[pair.first][1], transforms[pair.first],
                      checker._geometries[pair.second][1], transforms[pair.second], checker._request, result)
        yield result


@contextmanager
def scalar_reference():
    with patch.object(RobotGeometryChecker, "_native_pair_distances", scalar_distances):
        yield


def make_checker(tmp_path, extra=()):
    path = tmp_path / "robot.urdf"
    path.write_text(URDF)
    # Many coplanar support vertices expose history-dependent GJK witnesses.
    angles = np.linspace(0., 2*np.pi, 48, endpoint=False)
    vertices = np.asarray([[np.cos(angle), np.sin(angle), height]
                           for height in (-.5, .5) for angle in angles])
    return RobotGeometryChecker(path, base_frame="base", geometries=[
        NativeSolid("prism", "base", vertices, "symmetric native convex prism"),
        NativePrimitive("box", "z", "box", (1., 1., 1.), np.eye(4), "native moving box"),
        *extra])


def query(checker, q, pairs=None):
    return checker.query(joint_names=("qx", "qy", "qz"), joint_positions=q,
                         T_reference_base=np.eye(4), frames=("z",),
                         pairs=(PairQuery("prism", "box", "required", "explicit pair"),)
                         if pairs is None else pairs)


def test_pristine_history_keeps_all_report_words_under_repeated_and_reversed_states(tmp_path):
    checker = make_checker(tmp_path)
    rng = np.random.default_rng(202610041)
    states = []
    for index in range(18):
        translation = rng.normal(size=3) * rng.uniform(.01, 2.)
        if index % 3 == 0:
            translation = np.asarray([0., 0., 1. + .1*(index % 5)])
        states.append(-translation)
    states += [np.asarray([0., 0., value]) for value in
               (1., np.nextafter(1., 0.), np.nextafter(1., 2.), -0.)]
    for state in states + states[::-1]:
        with scalar_reference():
            expected = query(checker, state)
        assert_bits_equal(expected, query(checker, state))


def test_pair_orientation_policy_empty_queries_and_rebinding_preserve_reports(tmp_path):
    local = np.eye(4)
    local[0, 3] = 3.
    checker = make_checker(tmp_path, (NativePrimitive("fixed", "base", "box", (.2, .2, .2), local,
                                                     "native fixed box"),))
    tables = [
        (PairQuery("prism", "box", "required", "first"),
         PairQuery("fixed", "box", "excluded", "excluded still retains distance")),
        (PairQuery("box", "fixed", "mechanical_contact", "explicit allowance"),
         PairQuery("box", "prism", "diagnostic", "reversed witnesses")),
        (), (PairQuery("prism", "box", "required", "first"),)]
    for pairs in tables:
        with scalar_reference():
            expected = query(checker, [.1, -.2, 1.2], pairs)
        assert_bits_equal(expected, query(checker, [.1, -.2, 1.2], pairs))


def test_more_than_one_bounded_block_preserves_original_pair_order(tmp_path, monkeypatch):
    extra = []
    for index in range(17):
        local = np.eye(4)
        local[0, 3] = index*.13 + 3.
        extra.append(NativePrimitive(str(index), "base", "box", (.1, .1, .1), local, "native separated box"))
    checker = make_checker(tmp_path, extra)
    pairs = tuple(PairQuery(first, second, "diagnostic", "complete explicit pair table")
                  for first, second in combinations(checker.geometry_names, 2))
    with scalar_reference():
        expected = query(checker, [0., .1, 1.5], pairs)
    sizes = []
    original = module.pin.computeDistances

    def recorded(model, data):
        sizes.append(len(model.collisionPairs))
        return original(model, data)

    monkeypatch.setattr(module.pin, "computeDistances", recorded)
    assert_bits_equal(expected, query(checker, [0., .1, 1.5], pairs))
    assert sizes == [128, len(pairs)-128]
    schema = checker._native_distance_batches
    assert [len(block[1].collisionPairs) for block in schema[1]] == sizes
    # All blocks survive a second state; only their mutable solvers are reset.
    old_models = [block[1] for block in schema[1]]
    query(checker, [0., .2, 1.4], pairs)
    assert all(before is after[1] for before, after in zip(old_models, checker._native_distance_batches[1]))
    assert checker.native_query_storage_nbytes > sum(block[1].collisionPairMapping.nbytes for block in schema[1])


def test_native_exception_and_nonfinite_evidence_never_return_a_success(tmp_path, monkeypatch):
    checker = make_checker(tmp_path)
    original = module.pin.computeDistances

    def failed(*unused):
        raise RuntimeError("native query failure with pair context")

    monkeypatch.setattr(module.pin, "computeDistances", failed)
    with pytest.raises(RuntimeError, match="native query failure"):
        query(checker, [0., 0., 2.])
    # Input errors remain before the native routine, regardless of its failure.
    with pytest.raises(ValueError, match="finite"):
        query(checker, [0., 0., np.nan])

    def nonfinite(model, data):
        value = original(model, data)
        data.distanceResults[0].min_distance = np.nan
        return value

    monkeypatch.setattr(module.pin, "computeDistances", nonfinite)
    with pytest.raises(RuntimeError, match="nonfinite geometry evidence for prism, box"):
        query(checker, [0., 0., 2.])
    monkeypatch.setattr(module.pin, "computeDistances", original)
    with scalar_reference():
        expected = query(checker, [0., 0., 2.])
    assert_bits_equal(expected, query(checker, [0., 0., 2.]))
# END no tolerance, sampling reduction, GPU or simulation participates.
