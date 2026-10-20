"""CPU regressions for background identity uncertainty at image boundaries."""

# BEGIN preserve the approved one-pixel boundary without relabelling geometry.
import numpy as np

from manipulation.planning_observation import _partition_masks, _self_filtered_views
from manipulation.robot_self_filter import AMBIGUOUS_SHADOW, ENVIRONMENT, SELF


def observation():
    """Use explicit measured rays so exclusion cannot manufacture free space."""
    depth = np.ones((9, 9))
    return {
        "image": np.zeros((9, 9, 3), np.uint8), "depth": depth, "K": np.eye(3),
        "T_output_camera": np.eye(4), "reference_frame": "world", "owner_masks": {},
        "self_filter": {
            "labels": np.full((9, 9), ENVIRONMENT, np.uint8),
            "free_ray_depth_m": depth.copy(), "source": "synthetic measured rays",
        },
    }


def partition(record, owner_ids):
    normalized = _self_filtered_views({"camera": record}, "world", (.1, 3.))
    return _partition_masks(normalized, owner_ids)["camera"]


def test_eight_neighbours_are_withheld_but_entity_and_free_rays_are_unchanged():
    record = observation()
    owner = np.zeros((9, 9), bool)
    owner[4, 4] = True
    record["owner_masks"] = {"part": owner}
    original_depth = record["depth"].copy()
    original_rays = record["self_filter"]["free_ray_depth_m"].copy()
    original_labels = record["self_filter"]["labels"].copy()
    result = partition(record, ("part",))
    expected_background = np.ones((9, 9), bool)
    expected_background[3:6, 3:6] = False
    np.testing.assert_array_equal(result[None], expected_background)
    np.testing.assert_array_equal(result["part"], owner)
    np.testing.assert_array_equal(record["owner_masks"]["part"], owner)
    np.testing.assert_array_equal(record["depth"], original_depth)
    np.testing.assert_array_equal(record["self_filter"]["free_ray_depth_m"], original_rays)
    np.testing.assert_array_equal(record["self_filter"]["labels"], original_labels)
    assert not result[None].flags.writeable


def test_original_background_partition_protects_retained_entity_without_owner_mask():
    record = observation()
    record["background_mask"] = np.ones((9, 9), bool)
    record["background_mask"][4, 4] = False
    original = record["background_mask"].copy()
    result = partition(record, ())
    expected = np.ones((9, 9), bool)
    expected[3:6, 3:6] = False
    np.testing.assert_array_equal(result[None], expected)
    np.testing.assert_array_equal(record["background_mask"], original)
    assert set(result) == {None}


def test_adjacent_entities_retain_their_distinct_pixel_identities():
    record = observation()
    first, second = np.zeros((9, 9), bool), np.zeros((9, 9), bool)
    first[4, 4], second[4, 5] = True, True
    record["owner_masks"] = {"first": first, "second": second}
    result = partition(record, ("first", "second"))
    expected = np.ones((9, 9), bool)
    expected[3:6, 3:7] = False
    np.testing.assert_array_equal(result[None], expected)
    np.testing.assert_array_equal(result["first"], first)
    np.testing.assert_array_equal(result["second"], second)


def test_robot_shadow_and_invalid_pixels_do_not_seed_boundary_or_wrap_image():
    record = observation()
    owner = np.zeros((9, 9), bool)
    owner[0, 0] = owner[4, 4] = owner[4, 7] = owner[7, 4] = True
    record["owner_masks"] = {"part": owner}
    labels = record["self_filter"]["labels"]
    labels[4, 4], labels[4, 7], labels[7, 4] = SELF, AMBIGUOUS_SHADOW, AMBIGUOUS_SHADOW
    record["depth"][7, 4] = 0.
    record["self_filter"]["free_ray_depth_m"][labels != ENVIRONMENT] = 0.
    result = partition(record, ("part",))
    expected = labels == ENVIRONMENT
    expected[:2, :2] = False
    np.testing.assert_array_equal(result[None], expected)
    assert result[None][-1, -1]
    assert result[None][5, 5]
    assert result[None][5, 8]
    assert result[None][8, 5]


def test_retained_background_does_not_require_a_background_partition():
    record = observation()
    normalized = _self_filtered_views({"camera": record}, "world", (.1, 3.))
    assert _partition_masks(normalized, (), build_background=False) == {"camera": {}}
# END no observation model or threshold changes are part of these regressions.
