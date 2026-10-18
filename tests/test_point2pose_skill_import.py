"""CPU checks of the formal recorded-estimate to portable-bundle boundary."""

# BEGIN observation evidence cannot be replaced by retained tracker matrices.
from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from config.import_point2pose_skill import import_point2pose_skill
from manipulation.skill import load_grounding_skill


def pose(angle=0., xyz=(0., 0., 0.)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
    value[:3, 3] = xyz
    return value


@pytest.fixture
def recorded(tmp_path):
    directory = tmp_path / "tracking"
    directory.mkdir()
    frames, times = np.array([11, 19, 30]), np.array([2., 2.1, 2.35])
    fixed = pose(.3, (.2, -.4, .6))[None]
    motion = np.stack([pose(angle, (0., .1 * index, 0.)) for index, angle in enumerate([.1, .2, .5])])[:, None]
    camera = np.repeat(np.eye(4)[None], 3, axis=0)
    body = motion @ fixed
    K = np.array([[20., 0., 2.5], [0., 20., 1.5], [0., 0., 1.]])
    arrays = dict(object_ids=np.array(["tracked_part"]), frame_indices=frames, tracker_frame_ids=np.arange(3),
        frame_names=np.array(["a", "b", "c"]), timestamp_s=times, timestamp_available=np.ones(3, bool),
        K=np.repeat(K[None], 3, axis=0), T_world_camera_cv=camera, world_camera_available=np.ones(3, bool),
        T_initial_camera_B=fixed, T_camera_initial_map=motion, T_camera_B=body, T_world_B=camera[:, None] @ body)
    np.savez(directory / "estimated_poses_by_frame.npz", **arrays)
    report = dict(status="completed", processed_frames=3, object_ids=["tracked_part"],
        frames=[dict(frame_index=int(frames[index]), tracker_frame_id=index, frame_name=name,
                     timestamp_s=float(times[index]), camera_pose_available=True,
                     objects=[dict(object_index=0, status=status, raw_test_marker=index)])
                for index, (name, status) in enumerate(zip(["a", "b", "c"], ["initialized", "warning", "frontend_condition_met"]))])
    (directory / "tracking_evidence.json").write_text(json.dumps(report))
    y, x = np.indices((4, 6))
    pixels = np.column_stack((x.ravel(), y.ravel()))
    xyz = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    region = np.ones((4, 6), bool)
    # BEGIN recorded fixture has a declared planar observed region.
    source = dict(region_geometry='planar', image=np.zeros((4, 6, 3), np.uint8), depth_m=np.ones((4, 6), np.float32),
        crop_xyxy=np.array([0, 0, 6, 4]), frame_index=19, timestamp_s=2.1, owner_id="tracked_part",
        region_image_mask=region, provenance={"region_source": "Given image mask; no tracking role"},
        observation=dict(xyz=xyz, pixels=pixels, object_mask=region, entity_member=np.ones(len(pixels), bool),
            region_mask=np.ones(len(pixels), bool), evidence_only=np.zeros(len(pixels), bool), K=K, T_world_camera_cv=np.eye(4)))
    # END explicit geometry declaration.
    metadata = dict(skill_id="recorded_interval", owner_role="carrier", roles={"object": "supplied semantic role"},
        semantic=dict(object_description="container", interaction_part_description="outer surface",
            interaction_reference=dict(text="container", scope="object"), description="Given observed interval.",
            reason="Use the recorded visual region."), provenance={"gt_scope": "Old pose scope must not survive"})
    eef_poses = np.stack([pose(-.1, (.5, .2, .8 + .1 * index)) for index in range(3)])
    eef = dict(frame_indices=frames, timestamps=times, T_world_eef=eef_poses,
               source="Synchronized robot tool observations", world_alignment_source="Recorded tool-to-world calibration")
    kwargs = dict(metadata=metadata, source=source, kind="grasp", owner_bindings={"carrier": "tracked_part"},
                  binding_source="Declared rigid instance mask order", eef=eef)
    return directory, arrays, report, kwargs


def test_same_tracker_chart_saved_and_loaded_without_fabricated_interaction(tmp_path, recorded):
    directory, arrays, report, kwargs = recorded
    F = pose(-.2, (.02, .03, 1.))
    kwargs["function_frame"] = dict(T_world_function=F, frame_index=19, timestamp_s=2.1,
                                    source="Provided functional frame in the observed image world")
    result = import_point2pose_skill(directory, tmp_path / "bundle", **kwargs)
    assert result["status"] == "saved_observation_segment"
    bundle = load_grounding_skill(tmp_path / "bundle")
    B = arrays["T_world_B"][1, 0]
    L = np.linalg.inv(B) @ F
    np.testing.assert_allclose(bundle["source"]["T_world_body_at_image"], B)
    np.testing.assert_allclose(bundle["source"]["L_body_region"], L)
    np.testing.assert_allclose(bundle["representation"]["L_G"], L)
    expected = np.linalg.inv(L) @ np.linalg.inv(arrays["T_world_B"][:, 0]) @ kwargs["eef"]["T_world_eef"]
    np.testing.assert_allclose(bundle["representation"]["H_g"], expected, atol=1e-12)
    np.testing.assert_array_equal(bundle["representation"]["frame_indices"], [11, 19, 30])
    np.testing.assert_array_equal(bundle["representation"]["timestamps"], [2., 2.1, 2.35])
    assert bundle["representation"]["events"] == []
    assert "interaction_pose" in bundle["representation"]["summaries"]["H_g"]["missing"]
    provenance = bundle["metadata"]["provenance"]
    assert "gt_scope" not in provenance
    assert provenance["motion"]["kind"] == "point2pose_estimate"
    assert provenance["tracking_evidence"] == report
    assert provenance["eef"]["source"] == kwargs["eef"]["source"]


def test_measured_source_region_frame_and_missing_eef_are_preserved(tmp_path, recorded):
    directory, arrays, _, kwargs = recorded
    kwargs["eef"] = None
    result = import_point2pose_skill(directory, tmp_path / "bundle", **kwargs)
    source = result["source"]
    world = source["T_world_body_at_image"] @ source["L_body_region"]
    np.testing.assert_allclose(world[:3, 3], kwargs["source"]["observation"]["xyz"].mean(axis=0), atol=1e-12)
    assert "H_g" in result["representation"]["missing"]
    assert "C_D" not in result["representation"]
    np.testing.assert_allclose(result["representation"]["T_world_carrier"], arrays["T_world_B"][:, 0])


def test_manipulation_binds_reference_region_and_carrier_tracks_separately(tmp_path, recorded):
    directory, arrays, report, kwargs = recorded
    second_fixed = pose(-.6, (.4, .5, -.2))
    arrays["object_ids"] = np.array(["tracked_part", "reference_part"])
    arrays["T_initial_camera_B"] = np.concatenate([arrays["T_initial_camera_B"], second_fixed[None]])
    second_motion = np.stack([pose(.05 * index, (.2, -.1, 0.)) for index in range(3)])
    arrays["T_camera_initial_map"] = np.concatenate([arrays["T_camera_initial_map"], second_motion[:, None]], axis=1)
    arrays["T_camera_B"] = arrays["T_camera_initial_map"] @ arrays["T_initial_camera_B"]
    arrays["T_world_B"] = arrays["T_world_camera_cv"][:, None] @ arrays["T_camera_B"]
    np.savez(directory / "estimated_poses_by_frame.npz", **arrays)
    report["object_ids"] = ["tracked_part", "reference_part"]
    for row in report["frames"]:
        row["objects"].append(dict(row["objects"][0], object_index=1))
    (directory / "tracking_evidence.json").write_text(json.dumps(report))
    kwargs.update(kind="manipulation", eef=None,
                  owner_bindings={"carrier": "tracked_part", "reference": "reference_part"})
    kwargs["metadata"]["owner_role"] = "reference"
    kwargs["source"]["owner_id"] = "reference_part"
    result = import_point2pose_skill(directory, tmp_path / "bundle", **kwargs)
    representation = result["representation"]
    L = result["source"]["L_body_region"]
    expected = np.linalg.inv(L) @ np.linalg.inv(arrays["T_world_B"][:, 1]) @ arrays["T_world_B"][:, 0]
    np.testing.assert_allclose(representation["H_m"], expected, atol=1e-12)
    np.testing.assert_allclose(result["source"]["T_world_body_at_image"], arrays["T_world_B"][1, 1])
    assert "C_D" in representation["missing"]


@pytest.mark.parametrize("status", ["stale", "missing", "failed", "initialized", "unrecognized"])
def test_every_original_frame_requires_current_bound_owner_evidence(tmp_path, recorded, status):
    directory, _, report, kwargs = recorded
    # The source image row remains usable. A different row must still block
    # conversion instead of being dropped or read as an observed old matrix.
    report["frames"][2]["objects"][0]["status"] = status
    (directory / "tracking_evidence.json").write_text(json.dumps(report))
    output = tmp_path / "bundle"
    result = import_point2pose_skill(directory, output, **kwargs)
    assert result["status"] == "missing_source_observations"
    assert result["missing"][0]["frame_index"] == 30
    assert not output.exists()


def test_failed_recording_cannot_be_imported_as_complete_interval(tmp_path, recorded):
    directory, _, report, kwargs = recorded
    report.update(status="failed", failed_frame=30, error="Recorded native failure")
    (directory / "tracking_evidence.json").write_text(json.dumps(report))
    result = import_point2pose_skill(directory, tmp_path / "bundle", **kwargs)
    assert result["status"] == "missing_source_observations"
    assert result["missing"][0]["failed_frame"] == 30
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("availability", ["timestamp_available", "world_camera_available"])
def test_missing_clock_or_world_is_not_filled_in(tmp_path, recorded, availability):
    directory, arrays, _, kwargs = recorded
    arrays[availability][2] = False
    np.savez(directory / "estimated_poses_by_frame.npz", **arrays)
    result = import_point2pose_skill(directory, tmp_path / "bundle", **kwargs)
    assert result["status"] == "missing_source_observations"
    assert result["missing"][0]["reason"] == availability + "_missing"
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("field", ["T_world_body_at_image", "L_body_region", "frame_construction"])
def test_prior_body_chart_fields_are_rejected(tmp_path, recorded, field):
    directory, _, _, kwargs = recorded
    kwargs["source"][field] = np.eye(4)
    with pytest.raises(ValueError, match="only visual"):
        import_point2pose_skill(directory, tmp_path / "bundle", **kwargs)
    assert not (tmp_path / "bundle").exists()


def test_report_eef_and_given_function_frame_must_share_original_observations(tmp_path, recorded):
    directory, _, report, kwargs = recorded
    bad = deepcopy(kwargs)
    bad["eef"]["timestamps"][1] += .01
    with pytest.raises(ValueError, match="EEF observations"):
        import_point2pose_skill(directory, tmp_path / "eef", **bad)
    bad = deepcopy(kwargs)
    bad["function_frame"] = dict(T_world_function=np.eye(4), frame_index=11, timestamp_s=2., source="Other image")
    with pytest.raises(ValueError, match="Provided world function frame"):
        import_point2pose_skill(directory, tmp_path / "function", **bad)
    report["frames"][2]["frame_index"] = 31
    (directory / "tracking_evidence.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="align with every"):
        import_point2pose_skill(directory, tmp_path / "mismatch", **kwargs)


def test_events_require_real_provenance_and_matching_time(tmp_path, recorded):
    directory, _, _, kwargs = recorded
    kwargs.update(events=[dict(kind="observed_boundary", frame_index=19, timestamp=2.1)])
    with pytest.raises(ValueError, match="event_source"):
        import_point2pose_skill(directory, tmp_path / "missing", **kwargs)
    kwargs["event_source"] = "Explicit recorded annotation at original frame 19"
    result = import_point2pose_skill(directory, tmp_path / "present", **kwargs)
    assert result["representation"]["events"] == kwargs["events"]
    assert result["representation"]["interaction_time"] is None
# END passing these checks establishes representation consistency, not accuracy.
