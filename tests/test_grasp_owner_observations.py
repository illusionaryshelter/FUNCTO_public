"""CPU contracts for distinct reference, owner and static-background evidence.

Coincident points with different element identities stay independent. Image
tests use explicit measured partitions and a declared classifier result, so
they inspect ownership routing without loading a GPU mapper or learned model.
"""
# BEGIN observed-material identity and partition contracts for grasp inputs.
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_array_equal
from PIL import Image
import pytest

from manipulation import cross_instance_grasp as grasp
from manipulation import grounding_geometry_evidence as geometry
from manipulation.observations import backproject
from manipulation.planning_observation import _partition_masks
from manipulation.robot_self_filter import ENVIRONMENT


def element_inputs():
    # Element 501 is coincident with owner element 7. It still has no owner
    # input row: positional matching would incorrectly steer that native row.
    return dict(
        target_owner_points_world_m=np.array([[.1, .2, .8], [.3, .4, .9],
                                              [.5, .6, 1.], [.7, .8, 1.1]]),
        target_owner_element_ids=np.array([41, 7, 90, 22]),
        target_points_world_m=np.array([[.5, .6, 1.], [.1, .2, .8], [.3, .4, .9]]),
        target_element_ids=np.array([90, 41, 501]),
        matching_target_indices=np.array([1, 0, 1, 2]),
        source_row_ids=np.array([2, 9, 14, 27]))


def test_region_steering_uses_exact_ids_and_keeps_owner_input_order():
    arrays = element_inputs()
    original = {name: value.copy() for name, value in arrays.items()}
    camera = np.eye(4)
    camera[:3, 3] = [.2, -.1, .3]
    points, mask, report = grasp.record_owner_model_input(arrays, camera, "camera-left")
    assert_array_equal(points, ((original["target_owner_points_world_m"]-camera[:3, 3])
                               @ camera[:3, :3]).astype(np.float32))
    assert_array_equal(mask, [True, False, True, False])
    assert_array_equal(arrays["grasp_model_input_element_ids"], [41, 7, 90, 22])
    assert_array_equal(arrays["grasp_model_region_element_ids"], [41, 90, 41, 501])
    assert_array_equal(arrays["grasp_model_region_owner_indices"], [0, 2, 0, -1])
    assert_array_equal(arrays["grasp_model_region_owner_supported"], [True, True, True, False])
    assert_array_equal(arrays["grasp_model_region_unsupported_element_ids"], [501])
    assert_array_equal(arrays["grasp_model_region_unsupported_source_row_ids"], [27])
    assert report["region_point_count"] == 2
    assert report["supported_region_source_row_count"] == 3
    assert report["unsupported_region_source_row_count"] == 1
    for name, value in original.items():
        assert_array_equal(arrays[name], value)


def test_same_reference_and_owner_domain_preserves_original_model_bytes():
    arrays = element_inputs()
    arrays["target_element_ids"] = arrays["target_owner_element_ids"].copy()
    arrays["matching_target_indices"] = np.array([3, 0, 3, 2])
    camera = np.eye(4)
    camera[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    camera[:3, 3] = [.3, .2, -.1]
    original_points = ((arrays["target_owner_points_world_m"]-camera[:3, 3])
                       @ camera[:3, :3]).astype(np.float32)
    original_mask = np.zeros(len(original_points), bool)
    original_mask[arrays["matching_target_indices"]] = True
    points, mask, _ = grasp.record_owner_model_input(arrays, camera, "camera-left")
    assert points.dtype == original_points.dtype and points.tobytes() == original_points.tobytes()
    assert mask.dtype == original_mask.dtype and mask.tobytes() == original_mask.tobytes()
    assert not len(arrays["grasp_model_region_unsupported_element_ids"])


@pytest.mark.parametrize("ids", [np.array([41, 7, 90]), np.array([41, 7, 41, 22])])
def test_owner_element_identity_is_one_to_one_with_actual_input_rows(ids):
    arrays = element_inputs()
    arrays["target_owner_element_ids"] = ids
    with pytest.raises(ValueError, match="one-to-one|one point per"):
        grasp.record_owner_model_input(arrays, np.eye(4), "camera-left")


def _bank(mask, depth, K):
    y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    return dict(object_mask=mask, xyz=backproject(depth, K, mask=mask),
                pixels=np.column_stack((x, y)))


def _view(*, background_known, same_domain=False):
    shape = (9, 11)
    owner = np.zeros(shape, bool)
    owner[2:5, 2:5] = True
    reference = owner.copy() if same_domain else np.zeros(shape, bool)
    if not same_domain:
        reference[3, 3] = reference[5, 5] = True
    background = np.zeros(shape, bool)
    if same_domain:
        background = ~owner
    elif background_known:
        background[:, 7:] = True
    depth = np.ones(shape)
    K = np.array([[100., 0., 5.], [0., 100., 4.], [0., 0., 1.]])
    owner_bank = _bank(owner, depth, K)
    return dict(id="camera-left", image=Image.fromarray(np.zeros((*shape, 3), np.uint8)),
        depth_m=depth, K=K, T_world_camera_cv=np.eye(4),
        observation=owner_bank if same_domain else _bank(reference, depth, K),
        owner_observation=owner_bank, owner_mask=owner,
        owner_evidence=dict(owner_id="registered-owner", scope="visible_owner_evidence",
                            source="declared current calibrated mask",
                            reference_material_owner_id="registered-owner",
                            reference_material_source="declared local reference material"),
        background_mask=background)


def _classification(monkeypatch):
    monkeypatch.setattr(geometry, "robot_self_filter_shapes", lambda fixture, frames: ())

    def measured_environment(*, observed_depth, **parameters):
        evidence = dict(labels=np.full(observed_depth.shape, ENVIRONMENT, np.uint8),
            free_ray_depth_m=observed_depth.copy(), source="declared CPU classifier fixture")
        return dict(self_filter=evidence, provenance=dict(source=evidence["source"]))

    monkeypatch.setattr(geometry, "classify_depth", measured_environment)


def _prepare(view, local_material_pixels):
    local_material_pixels = np.asarray(local_material_pixels, dtype=np.int64).reshape(-1, 2)
    return geometry.prepare_grasp_observations([view], fixture={}, capture_frames={},
        reference_frame="world", owner_id="registered-owner",
        local_material_observation_pixels=local_material_pixels,
        local_material_observation_view_indices=np.zeros(len(local_material_pixels), dtype=np.int64),
        mapping_settings=dict(depth_range_m=[.05, 5.], self_filter_padding_scale=1.,
            self_filter_padding_offset_m=.01, self_filter_source="declared CPU fixture"))


@pytest.mark.parametrize("background_known", [False, True])
def test_part_and_owner_masks_stay_separate_and_background_is_explicit(monkeypatch, background_known):
    _classification(monkeypatch)
    view = _view(background_known=background_known)
    reference, owner, background = (view["observation"]["object_mask"].copy(),
        view["owner_mask"].copy(), view["background_mask"].copy())
    normalized, partitions, reports, arrays = _prepare(view, view["observation"]["pixels"])
    assert_array_equal(normalized[view["id"]]["owner_masks"]["registered-owner"], owner)
    assert_array_equal(normalized[view["id"]]["background_candidate_mask"], background)
    assert_array_equal(partitions[view["id"]]["registered-owner"], owner)
    admitted = partitions[view["id"]][None]
    assert not np.any(admitted & ~background)
    assert bool(admitted.any()) is background_known
    assert reports[view["id"]]["unassigned_observed_pixels"] > 0
    assert_array_equal(arrays["view_0_reference_mask"], reference)
    assert_array_equal(arrays["view_0_semantic_owner_mask"], owner)
    assert reference[5, 5] and not owner[5, 5]
    assert_array_equal(view["observation"]["object_mask"], reference)
    assert_array_equal(view["owner_mask"], owner)
    assert_array_equal(view["background_mask"], background)


def test_declared_reference_material_outside_owner_stays_out_of_model_background(monkeypatch):
    _classification(monkeypatch)
    view = _view(background_known=True)
    view["background_mask"] = ~view["owner_mask"]
    original = {name: value.copy() for name, value in
        (("reference", view["observation"]["object_mask"]),
         ("owner", view["owner_mask"]), ("background", view["background_mask"]))}
    _, partitions, reports, arrays = _prepare(view, view["observation"]["pixels"])
    assert original["reference"][5, 5] and not original["owner"][5, 5]
    assert original["background"][5, 5]
    assert not arrays["view_0_background_model_mask"][5, 5]
    assert not partitions[view["id"]][None][5, 5]
    assert arrays["view_0_admitted_local_material_mask"][5, 5]
    assert not arrays["view_0_admitted_owner_mask"][5, 5]
    assert reports[view["id"]]["background_admission"]["excluded_local_material_pixels"] == 1
    assert_array_equal(view["observation"]["object_mask"], original["reference"])
    assert_array_equal(view["owner_mask"], original["owner"])
    assert_array_equal(view["background_mask"], original["background"])


def test_unassigned_reference_has_no_inferred_owner_material_pixels(monkeypatch):
    _classification(monkeypatch)
    view = _view(background_known=False)
    view["owner_evidence"]["reference_material_owner_id"] = "another-declared-owner"
    _, partitions, reports, arrays = _prepare(view, np.empty((0, 2), np.int64))
    assert not arrays["view_0_local_material_mask"].any()
    assert not arrays["view_0_admitted_local_material_mask"].any()
    assert not partitions[view["id"]][None].any()
    assert reports[view["id"]]["background_admission"]["local_material_measurement_count"] == 0
    assert reports[view["id"]]["unassigned_observed_pixels"] == view["depth_m"].size-int(view["owner_mask"].sum())


def test_object_scope_explicit_background_matches_original_partition_bytes(monkeypatch):
    _classification(monkeypatch)
    view = _view(background_known=True, same_domain=True)
    normalized, partitions, _, _ = _prepare(view, view["observation"]["pixels"])
    original = {view["id"]: {key: value for key, value in normalized[view["id"]].items()
                            if key not in {"background_mask", "background_candidate_mask", "local_material_mask"}}}
    original_partitions = _partition_masks(original, ("registered-owner",), build_background=True)
    for identity in ("registered-owner", None):
        assert partitions[view["id"]][identity].tobytes() == original_partitions[view["id"]][identity].tobytes()
    assert view["observation"] is view["owner_observation"]


def test_native_empty_pool_keeps_unsupported_element_and_actual_input_evidence(monkeypatch):
    arrays = element_inputs()
    arrays.update(source_frame_world=np.eye(4), view_ids=np.array(["camera-left"]))
    region = dict(status="completed", arrays=arrays, input_context={}, context={})
    view = dict(id="camera-left", T_world_camera_cv=np.eye(4),
        owner_evidence=dict(owner_id="registered-owner", scope="visible_owner_evidence",
                            source="declared current mask"))
    monkeypatch.setattr(grasp, "source_capture_grasp", lambda *args, **kwargs:
                        (np.eye(4), np.eye(4), {}))
    from manipulation import tool_geometry
    monkeypatch.setattr(tool_geometry, "load_robosuite_panda_tool", lambda **kwargs:
        SimpleNamespace(grasp_generation_geometry=lambda: dict(T_tool_grasp=np.eye(4))))
    monkeypatch.setattr(grasp, "AnyGraspBackend", lambda **kwargs: object())

    def empty_pool(*args, **kwargs):
        raise grasp.NoGraspCandidates(dict(generated_count=0))

    monkeypatch.setattr(grasp, "generate_and_rank", empty_pool)
    with pytest.raises(grasp.NoGraspCandidates) as caught:
        grasp.run_cross_instance_grasp(dict(representation=dict(kind="grasp")), region, [view],
            feedback=dict(tool_id="fixed-tool", tool_frame="tcp", eef_pose=np.eye(4)),
            robot_fixture={}, owner_id="registered-owner", mapping_settings={},
            max_tilt_rad=.2, field_margin_m=.0001, approach_thresh_rad=.2)
    saved = caught.value.diagnostics["model_input_arrays"]
    assert_array_equal(saved["grasp_model_input_points_world_m"], arrays["target_owner_points_world_m"])
    assert_array_equal(saved["grasp_model_region_unsupported_element_ids"], [501])
    assert_array_equal(saved["grasp_model_region_unsupported_source_row_ids"], [27])
# END caller masks and element identities retain their declared data roles.
