"""Current raw-field coverage and explicit unknown-policy scene behavior."""

# BEGIN represented occupied cells replace obsolete robot-sphere coverage fixtures.
import numpy as np
import pytest
import torch

from manipulation.implicit_distance import ObservedField
from manipulation.observed_coverage import query_coverage
from manipulation.planning_scene import ObservedSpace
from geometry_objective_fixtures import _moving_scene


def coverage(known):
    return ObservedSpace(known, origin=[-.02] * 3, voxel_size=.01,
        observation_ids=("measured-view",), reference_frame="world")


def occupied():
    return ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
                         origin=[-.008] * 3, voxel_size=.016)


def region(lower_x=.005):
    return {"lower_m": [lower_x, -.03, -.03], "upper_m": [.03, .03, .03],
            "source": "declared finite region"}


def test_measured_and_declared_regions_cover_original_occupied_material():
    known = np.zeros((5, 5, 5), bool); known[:, :, :3] = True
    space, field = coverage(known), occupied()
    assert query_coverage(field, np.eye(4), space, [])["status"] == "unknown_witness"
    assert query_coverage(field, np.eye(4), space, [region()])["status"] == "covered"
    np.testing.assert_array_equal(space.known_free, known)


def test_adjacent_finite_regions_jointly_cover_the_original_occupied_set():
    known = np.zeros((5, 5, 5), bool); known[:, :, :3] = True
    lower, upper = region(), region()
    lower["upper_m"][1] = 0.; upper["lower_m"][1] = 0.
    for item in (lower, upper):
        assert query_coverage(occupied(), np.eye(4), coverage(known), [item])["status"] == "unknown_witness"
    assert query_coverage(occupied(), np.eye(4), coverage(known), [lower, upper])["status"] == "covered"


def test_actual_occupied_witness_outside_permission_remains_unknown():
    known = np.zeros((5, 5, 5), bool); known[:, :, :3] = True
    known[1:4, 1:4, 1] = False
    result = query_coverage(occupied(), np.eye(4), coverage(known), [region()])
    assert result["status"] == "unknown_witness" and result["witness"][0] < .005


def test_finite_permission_can_cover_material_beyond_the_observation_map():
    transform = np.eye(4); transform[0, 3] = .2
    permission = {"lower_m": [.18, -.03, -.03], "upper_m": [.22, .03, .03]}
    space = coverage(np.zeros((5, 5, 5), bool))
    assert query_coverage(occupied(), transform, space, [permission])["status"] == "covered"
    transform[0, 3] = .18
    assert query_coverage(occupied(), transform, space, [permission])["status"] == "unknown_witness"


def scene_components(scene, x=-.005):
    position = torch.tensor([[[x, 1., 0.]]], dtype=torch.float64)
    quaternion = torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64)
    progress = torch.tensor([[.5]], dtype=torch.float64)
    joints = torch.tensor([[[.04, -.02]]], dtype=torch.float64)
    return scene.components(None, position, quaternion, progress, joints,
        ("left_slide", "right_slide"), trajectory_time=progress)


@pytest.mark.parametrize("action", ["forbid", "require_observation", "allow_attempt"])
def test_unknown_action_only_changes_its_component_role_and_preserves_collision(tmp_path, action):
    scene = _moving_scene(tmp_path)
    scene.unknown_policy = {"conditional_regions": [], "outside_regions": action}
    terms = scene_components(scene)
    prefix = "conditional/" if action == "allow_attempt" else ""
    assert terms[f"{prefix}unknown/forbidden/material"].item() == 1.
    assert terms["collision/material/background"].item() == 1.
    assert not any("unknown/forbidden/robot" in name for name in terms)
    assert scene.last_query_report["robot_view_coverage_status"] == "not_queried"
    assert scene.last_query_report["owner_coverage"]["material"][0]["status"] == "unknown_witness"


@pytest.mark.parametrize("action", ["forbid", "require_observation", "allow_attempt"])
def test_scene_clone_and_execution_record_preserve_original_unknown_evidence(tmp_path, action):
    scene = _moving_scene(tmp_path)
    scene.unknown_policy = {"conditional_regions": [], "outside_regions": action}
    clone = scene.clone_for_rollout()
    original, copied = scene_components(scene), scene_components(clone)
    for key in original:
        torch.testing.assert_close(original[key], copied[key])
    record = scene.to_execution_binding()
    assert record["unknown_policy"] == scene.unknown_policy
    assert clone.coverage is scene.coverage
    np.testing.assert_array_equal(scene.coverage.known_free, clone.coverage.known_free)
# END measured coverage, declared permission and geometric intersection remain separate.
