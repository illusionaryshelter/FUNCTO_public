"""Distance records cover every retained domain and reuse exact pose bounds."""

# BEGIN analytic shapes and raw supported cells exercise the query contract.
import coal
import numpy as np
import pytest

from manipulation.implicit_distance import ObservedField, PairPermission, query_distance
import manipulation.native_field_distance as native


def two_separated_cells():
    values = -np.ones((11, 2, 2))
    observed = np.zeros_like(values, bool)
    observed[:2] = True
    observed[9:11] = True
    return ObservedField(values, observed, origin=[-.06, -.005, -.005], voxel_size=.02)


def central_cell():
    return ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
                         origin=[-.01] * 3, voxel_size=.02)


@pytest.mark.parametrize("native_first", [False, True])
def test_clear_lower_bound_is_the_minimum_over_all_separated_branches(native_first):
    # The root box contains the central entity. Its two actual cells are
    # separated by .03 m and .11 m; the larger bound must not replace the
    # smaller one when the final branch is processed.
    field = two_separated_cells()
    if native_first:
        result = native.query_native_field(coal.Box(.02, .02, .02), np.eye(4), field, np.eye(4))
    else:
        result = query_distance(central_cell(), np.eye(4), field, np.eye(4))
    assert result["status"] == "clear" and result["queries"] > 0
    assert .02999999 < result["lower_m"] <= .03
    assert result["distance_scope"] == "unpermitted_point_pairs_at_query_pose"


@pytest.mark.parametrize("native_first", [False, True])
def test_full_permission_and_empty_observation_do_not_claim_positive_clearance(native_first):
    field = central_cell()
    whole = (np.vstack((np.eye(3), -np.eye(3))), np.ones(6))
    permission = PairPermission(((whole, whole),))
    if native_first:
        result = native.query_native_field(coal.Box(.02, .02, .02), np.eye(4), field,
                                           np.eye(4), permission=permission)
    else:
        result = query_distance(field, np.eye(4), field, np.eye(4), permission=permission)
    assert result["status"] == "clear" and result["lower_m"] == 0.
    assert not result.get("empty_set", False)


def test_empty_field_returns_before_native_support_queries(monkeypatch):
    field = ObservedField(-np.ones((2, 2, 2)), np.zeros((2, 2, 2), bool),
                          origin=np.zeros(3), voxel_size=1.)

    def unexpected(*args, **kwargs):
        raise AssertionError("An empty represented set needs no native support query")

    monkeypatch.setattr(native, "native_projection_bounds", unexpected)
    result = native.query_native_field(coal.Box(.02, .02, .02), np.eye(4), field, np.eye(4))
    assert result["status"] == "clear" and result["empty_set"]
    assert result["lower_m"] is None and result["upper_m"] is None


@pytest.mark.parametrize("translation", [0., .02, .05])
def test_prepared_bounds_preserve_full_same_pose_evidence(monkeypatch, translation):
    shape, field = coal.Box(.02, .02, .02), central_cell()
    pose = np.eye(4)
    pose[0, 3] = translation
    baseline = native.query_native_field(shape, pose, field, np.eye(4))
    prepared = (native._native_bounds(shape, native._transform(np.eye(4))),
                native._native_bounds(shape, native._transform(pose)))

    def unexpected(*args, **kwargs):
        raise AssertionError("The query must reuse its supplied exact-pose bounds")

    monkeypatch.setattr(native, "_native_bounds", unexpected)
    reused = native.query_native_field(shape, pose, field, np.eye(4), native_bounds=prepared)
    assert reused == baseline
# END reuse affects work organization, while original entities own evidence.


# BEGIN finite permissions retain independent whole-entity separation bounds.
@pytest.mark.parametrize("native_first", [False, True])
def test_permission_cannot_remove_complete_entity_separation_evidence(native_first):
    field = central_cell()
    pose = np.eye(4)
    pose[0, 3] = .1
    whole = (np.vstack((np.eye(3), -np.eye(3))), np.ones(6))
    permission = PairPermission(((whole, whole),))
    if native_first:
        result = native.query_native_field(coal.Box(.02, .02, .02), pose,
                                           field, np.eye(4), permission=permission)
    else:
        result = query_distance(central_cell(), pose, field, np.eye(4), permission=permission)
    assert result["status"] == "clear" and result["lower_m"] == 0.
    assert .07999999 < result["full_entity_lower_m"] <= .08


@pytest.mark.parametrize("native_first", [False, True])
def test_permission_does_not_turn_overlapping_enclosures_into_a_collision(native_first):
    field = central_cell()
    whole = (np.vstack((np.eye(3), -np.eye(3))), np.ones(6))
    permission = PairPermission(((whole, whole),))
    if native_first:
        result = native.query_native_field(coal.Box(.02, .02, .02), np.eye(4),
                                           field, np.eye(4), permission=permission)
    else:
        result = query_distance(field, np.eye(4), field, np.eye(4), permission=permission)
    assert result["status"] == "clear"
    assert result["full_entity_lower_m"] == 0.
    assert not result["intersection"]
# END clipping changes permission evidence, never the complete geometric set.
