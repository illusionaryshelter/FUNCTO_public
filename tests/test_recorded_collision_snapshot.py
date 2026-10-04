"""CPU regressions for complete, non-executable collision diagnostic evidence."""

# BEGIN candidate snapshots retain observed inputs without borrowing acceptance.
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
from recorded_task_planning import save_collision_snapshot
from manipulation.planning import JointTrajectory, _layer_query_record
from manipulation.collision_records import decode_array, encode_array


def candidate():
    """Use a rejected curve with an intermediate close and asymmetric fingers."""
    curve = JointTrajectory(("arm",), np.array([0., 1.]), np.array([[0.], [.1]]),
        np.zeros((2, 1)), np.array([0., .5, 1.]), np.array([0., .5, 1.]),
        ({"event_id": "close", "skill_id": "sample", "occurrence": 0,
          "kind": "close", "time": .5, "activation_time_s": .5, "requires_evidence": None},),
        (), {"scene": .01})
    finger_record = {"joint_names": ["left", "right"], "times": [0., .5, 1.],
                     "position": [[.02, -.03], [.02, -.03], [0., 0.]],
                     "velocity": [[0., 0.]]*3}
    curve.finger_trajectory = SimpleNamespace(describe=lambda: deepcopy(finger_record))
    scene = {"reference_frame": "world", "background": {"geometry": {
        "solids": [{"name": "table_cells", "occupied_centres": [[0., 0., 0.]], "voxel_size": .015}]}},
        "owners": [{"owner_id": "observed_target", "geometry": {
            "solids": [{"name": "target_cells", "occupied_centres": [[.1, .2, .3]]}]}}],
        "gripper": {"nominal_model": deepcopy(finger_record), "command_reference": deepcopy(finger_record)},
        "robot_owner_permissions": [{"target_owner": "observed_target", "interval": [.5, 1.]}],
        "coverage": {"observation_ids": ["current_rgbd"], "known_packbits": [7]},
        "observation_report": {"source": "current candidate observations"}}
    validation = {"u": np.array([0., .25, .5, .75, 1.]),
                  "violations": np.array([0., 0., .01, 0., 0.]),
                  "coordinate_direction_resolution": {"maximum_physical_time_gap_s": .25}}
    planner = SimpleNamespace(problem={"base_pose": np.eye(4), "tool_link": "tcp",
        "start_q": np.array([0.]), "start_dq": np.array([0.])},
        _verification_scene=SimpleNamespace(to_execution_binding=lambda: scene),
        formal_clock={"duration": 1., "interpolation_dt": .25}, last_validation=validation)
    return planner, curve, scene


def test_rejected_snapshot_owns_complete_scene_clock_curve_and_query_scope(tmp_path):
    planner, curve, scene = candidate()
    captured = {"status": "completed", "calls": [{"time_s": encode_array(np.array([[.5, .5, .5]], np.float32)),
        "progress": encode_array(np.array([[.5, .4999999701976776, .5000000596046448]], np.float32)),
        "robot_owner_permission_progress": encode_array(np.array([[.5, .5, .5000000596046448]], np.float32))}], "errors": []}
    target = tmp_path/"candidate-00-scene.json"
    save_collision_snapshot(target, planner, curve, curve_path="candidate-00.npz",
                            validation_status="completed", query_capture=captured)
    # A later seed can reuse every mutable source buffer without rewriting
    # this candidate's target, nominal fingers or completed query evidence.
    scene["owners"][0]["owner_id"] = "different_candidate_target"
    planner.last_validation["u"][1] = .125
    captured["calls"][0]["progress"] = encode_array(np.zeros((1, 3), np.float32))
    document = json.loads(target.read_text())
    assert document["classification"] == "diagnostic_collision_snapshot_not_executable"
    assert "execution_binding" not in document
    assert document["curve_path"] == "candidate-00.npz"
    assert document["scene"]["owners"][0]["owner_id"] == "observed_target"
    assert document["scene"]["background"]["geometry"]["solids"][0]["voxel_size"] == .015
    assert document["scene"]["coverage"]["observation_ids"] == ["current_rgbd"]
    assert document["scene"]["gripper"]["nominal_model"]["position"][0] == [.02, -.03]
    assert document["curve"]["fingers"] == document["scene"]["gripper"]["command_reference"]
    assert document["curve"]["events"][0]["activation_time_s"] == .5
    assert document["validation"]["u"] == [0., .25, .5, .75, 1.]
    assert document["validation"]["violations"][2] == .01
    assert decode_array(document["scene_query_capture"]["calls"][0]["progress"])[0][1] < .5
    np.testing.assert_array_equal(document["curve"]["coefficients"], curve._curve.c)
    assert curve.residuals == {"scene": .01}


def test_unfinished_export_never_copies_a_previous_candidates_validation(tmp_path):
    planner, curve, _ = candidate()
    planner.last_validation = {"u": [0., 1.], "violations": [0., 0.], "candidate": "previous_seed"}
    target = tmp_path/"candidate-scene.json"
    save_collision_snapshot(target, planner, curve, curve_path="candidate.npz",
        validation_status="not_completed", query_capture={"status": "not_recorded", "calls": [], "errors": []})
    document = json.loads(target.read_text())
    assert document["validation"] is None
    assert document["validation_status"] == "not_completed"
    assert document["scene_query_capture"]["status"] == "not_recorded"
    assert "previous_seed" not in target.read_text()


def test_scene_capture_keeps_native_float32_sides_and_arrival_permission():
    middle = torch.tensor(.5, dtype=torch.float32)
    progress = torch.stack((middle, torch.nextafter(middle, torch.tensor(-torch.inf)),
                            torch.nextafter(middle, torch.tensor(torch.inf))))[None]
    times = torch.full((1, 3), .5, dtype=torch.float32)
    permission = progress.clone()
    permission[:, 1] = middle
    positions = torch.zeros((1, 3, 1), dtype=torch.float32)
    # BEGIN the diagnostic copies the actual production query record.
    saved = _layer_query_record(times, progress, permission, positions, ("arm",), {}, ("exact", "left", "right"))
    # END no observer reconstructs a second set of mode probes.
    progress.fill_(0.)
    permission.fill_(0.)
    positions.fill_(99.)
    assert saved["progress_dtype"] == "torch.float32"
    assert decode_array(saved["progress"])[0][1] == np.nextafter(np.float32(.5), np.float32(-np.inf))
    assert decode_array(saved["progress"])[0][2] == np.nextafter(np.float32(.5), np.float32(np.inf))
    assert decode_array(saved["robot_owner_permission_progress"])[0][1] == .5
    np.testing.assert_array_equal(decode_array(saved["time_s"]), [[.5, .5, .5]])
    np.testing.assert_array_equal(decode_array(saved["joint_position"]), np.zeros((1, 3, 1)))


def test_snapshot_failure_does_not_leave_a_complete_looking_file(tmp_path):
    planner, curve, scene = candidate()
    scene["observation_report"]["invalid"] = float("nan")
    target = tmp_path/"invalid-scene.json"
    with pytest.raises(ValueError, match="NaN"):
        save_collision_snapshot(target, planner, curve, curve_path="candidate.npz",
            validation_status="completed", query_capture={"status": "not_recorded", "calls": [], "errors": []})
    assert not target.exists()
    assert curve.residuals == {"scene": .01}
# END archival failures cannot manufacture validation or execution permission.
