"""CPU regressions for cold-history native distance batches and full reports."""

# BEGIN exact floating-point and lifecycle checks for native many-pair queries.
from contextlib import contextmanager
from copy import deepcopy
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


# BEGIN one scalar library reference checks both public and compact consumers.
class ScalarPairTable:
    """Independent cold Coal calls implement only the distance output contract.

    Projection bounds have separate high-precision tests. These comparisons
    disable their mask so that they isolate distance, witness and report
    preservation across the C++ boundary.
    """

    def __init__(self, shapes, indices):
        self.shapes, self.indices = tuple(shapes), np.array(indices, copy=True)

    def query(self, poses, mask, request):
        assert not np.asarray(mask).any()
        output = np.zeros((len(poses), len(self.indices), 12))
        for state, matrices in enumerate(poses):
            transforms = [coal.Transform3s(pose[:3, :3], pose[:3, 3]) for pose in matrices]
            for index, (first, second) in enumerate(self.indices):
                result = coal.DistanceResult()
                coal.distance(self.shapes[first], transforms[first], self.shapes[second],
                              transforms[second], request, result)
                a, b, normal = (np.asarray(value, dtype=float) for value in
                    (result.getNearestPoint1(), result.getNearestPoint2(), result.normal))
                output[state, index, :10] = np.r_[result.min_distance, a, b, normal]
                output[state, index, 10] = np.linalg.norm(b-a-result.min_distance*normal)
        return output


@contextmanager
def scalar_reference(checker):
    """Replace only this call's table, restoring the real cached native owner."""
    native = checker._native_pair_table
    checker._native_pair_table = None
    try:
        with patch.object(module, "NativePairTable", ScalarPairTable):
            yield
    finally:
        checker._native_pair_table = native
# END independent distances do not reuse production solver histories.


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


def test_cold_queries_keep_all_report_words_under_repeated_and_reversed_states(tmp_path):
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
        with scalar_reference(checker):
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
        with scalar_reference(checker):
            expected = query(checker, [.1, -.2, 1.2], pairs)
        assert_bits_equal(expected, query(checker, [.1, -.2, 1.2], pairs))


# BEGIN compact batches preserve explicit state/pair order and table ownership.
def test_multiple_chunks_preserve_pair_order_and_share_the_public_query_table(tmp_path):
    extra = []
    for index in range(17):
        local = np.eye(4)
        local[0, 3] = index*.13 + 3.
        extra.append(NativePrimitive(str(index), "base", "box", (.1, .1, .1), local, "native separated box"))
    checker = make_checker(tmp_path, extra)
    pairs = tuple(PairQuery(first, second, "diagnostic", "complete explicit pair table")
                  for first, second in combinations(checker.geometry_names, 2))
    with scalar_reference(checker):
        expected = query(checker, [0., .1, 1.5], pairs)
    assert_bits_equal(expected, query(checker, [0., .1, 1.5], pairs))
    table = checker._native_pair_table[2]
    # 171 pairs cross the declared 4096-query chunk at 24 states. Each
    # output row belongs to the original state and supplied endpoint order.
    states = np.asarray([[0., .1+.001*index, 1.5] for index in range(49)])
    batches = list(checker.query_pair_batches(joint_names=("qx", "qy", "qz"),
        joint_positions=states, T_reference_base=np.eye(4), pairs=pairs,
        projection_mask=np.zeros((len(states), len(pairs)), dtype=bool)))
    assert [len(batch) for batch in batches] == [23, 23, 3]
    assert checker._native_pair_table[2] is table
    flattened = [state for batch in batches for state in batch]
    for index in (0, 22, 23, 45, 46, 48):
        with scalar_reference(checker):
            reference = query(checker, states[index], pairs)
        for actual, row in zip(flattened[index], reference["pairs"]):
            np.testing.assert_array_equal(actual[:10], [row["signed_distance_m"],
                *row["witness_first"], *row["witness_second"], *row["normal_first_to_second"]])
    # Policies belong to the report. Reusing exactly these oriented endpoints
    # with different explicit policy metadata does not rebuild native shapes.
    changed = tuple(PairQuery(pair.first, pair.second, "excluded", "changed declaration") for pair in pairs)
    result = query(checker, states[0], changed)
    assert checker._native_pair_table[2] is table
    assert all(row["policy"] == "excluded" for row in result["pairs"])
    saved = deepcopy(batches)
    query(checker, states[-1], tuple(reversed(pairs)))
    assert checker._native_pair_table[2] is not table
    assert_bits_equal(saved, batches)
# END numeric results outlive table replacement and subsequent queries.


# BEGIN native failures and cancellation cannot deliver a completed report.
def test_native_exception_and_nonfinite_input_never_return_a_success(tmp_path, monkeypatch):
    checker = make_checker(tmp_path)
    original = module.NativePairTable

    class FailedTable:
        def __init__(self, *args):
            pass

        def query(self, *args):
            raise RuntimeError("native query failure at state 0 pair 0")

    monkeypatch.setattr(module, "NativePairTable", FailedTable)
    with pytest.raises(RuntimeError, match="native query failure"):
        query(checker, [0., 0., 2.])
    # Input errors remain before the native routine, regardless of its failure.
    with pytest.raises(ValueError, match="finite"):
        query(checker, [0., 0., np.nan])

    monkeypatch.setattr(module, "NativePairTable", original)
    checker._native_pair_table = None
    with scalar_reference(checker):
        expected = query(checker, [0., 0., 2.])
    assert_bits_equal(expected, query(checker, [0., 0., 2.]))
    table = checker._native_pair_table[2]
    poses = np.broadcast_to(np.eye(4), (1, 2, 4, 4)).copy()
    poses[0, 0, 0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        table.query(poses, np.zeros((1, 1), bool), checker._request)


def test_cancellation_after_native_chunk_precedes_its_first_delivered_state(tmp_path, monkeypatch):
    checker = make_checker(tmp_path)
    original = module.NativePairTable
    calls = []

    class RecordedTable:
        def __init__(self, *args):
            self.native = original(*args)

        def query(self, poses, mask, request):
            result = self.native.query(poses, mask, request)
            calls.append(len(poses))
            return result

    monkeypatch.setattr(module, "NativePairTable", RecordedTable)
    pairs = (PairQuery("prism", "box", "required", "explicit pair"),)
    def check():
        if calls:
            raise TimeoutError("declared request deadline")
    batches = checker.query_pair_batches(joint_names=("qx", "qy", "qz"),
        joint_positions=np.tile([0., 0., 2.], (3, 1)), T_reference_base=np.eye(4),
        pairs=pairs, projection_mask=np.zeros((3, 1), dtype=bool), check_request=check)
    with pytest.raises(TimeoutError, match="request deadline"):
        next(batches)
    assert calls == [3]
# END no candidate can consume numeric evidence after its cancellation check.
# END no tolerance, sampling reduction, GPU or simulation participates.
