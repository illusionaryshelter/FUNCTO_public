"""Exercise persisted perception failures and cross-view identity boundaries.

Only the learned/remote model calls are replaced. The portable package reader,
semantic protocol, missing-input region adapter and local HTML writer run as
they do in the real command. These checks do not claim model accuracy.
"""
import json
from unittest.mock import Mock

import numpy as np

from config.ground_skill import ground_skill
from manipulation.skill import save_grounding_skill
from tests.test_manipulation_skill import source_inputs, detection


def inputs(directory):
    metadata, source, representation = source_inputs()
    metadata.update(tool_id="observed_tool", tool_frame="control_frame")
    source.update(feature=np.ones((1, 3, 4, 4), np.float32),
                  feature_model={"name": "explicit CPU fixture"})
    package = directory / "skill"
    save_grounding_skill(package, metadata=metadata, source=source, representation=representation)
    scene = directory / "scene"
    scene.mkdir()
    (scene / "robot_feedback.json").write_text(json.dumps({"tool_id": "observed_tool", "tool_frame": "control_frame"}))
    (scene / "layout.json").write_text(json.dumps({"sim_time_s": 4.}))
    for name in ("view_a", "view_b"):
        view = scene / name
        view.mkdir()
        source["image"].save(view / "rgb.png")
        np.save(view / "depth_m.npy", source["depth_m"])
        (view / "calibration.json").write_text(json.dumps({
            "K": source["observation"]["K"].tolist(),
            "T_world_camera_cv": np.eye(4).tolist(), "timestamp_s": 4.}))
    return package, scene


def test_transport_failure_preserves_both_actual_detector_results(tmp_path, monkeypatch):
    package, scene = inputs(tmp_path)
    detector = Mock(return_value=[detection("container")])
    monkeypatch.setattr("local_models.sam31.detect", detector)
    monkeypatch.setattr("local_models.gemini.Gemini.chat_json", Mock(side_effect=RuntimeError("explicit transport failure")))
    output = tmp_path / "failed"
    report = ground_skill(package, scene, output, instruction="Use the observed container")
    assert report["status"] == "failed"
    assert report["failed_stage"] == "select_current_entity"
    assert report["region_status"] == "missing"
    assert detector.call_count == 2
    for name in ("view_a", "view_b"):
        record = json.loads((output / (name + "-detections") / "candidates.json").read_text())
        assert record["candidate_count"] == 1 and record["masks_modified"] is False
    html = (output / "index.html").read_text()
    assert 'width="12" height="10"' in html
    assert "explicit transport failure" in html and 'href="reference.npz"' not in html


def test_unconfirmed_two_view_identity_never_enters_region_registration(tmp_path, monkeypatch):
    package, scene = inputs(tmp_path)
    monkeypatch.setattr("local_models.sam31.detect", Mock(return_value=[detection("container")]))
    response = {"views": {name: {"candidate_ids": [17], "reason": "One visible candidate"}
                           for name in ("view_a", "view_b")},
                "same_entity": False, "same_entity_reason": "These are distinct instances"}
    monkeypatch.setattr("local_models.gemini.Gemini.chat_json", Mock(return_value=response))
    feature = Mock(side_effect=AssertionError("Unassociated instances cannot request features"))
    registration = Mock(side_effect=AssertionError("Unassociated instances cannot enter registration"))
    monkeypatch.setattr("local_models.marco.extract_image", feature)
    monkeypatch.setattr("transfer.dual_view.pipeline.prepare", Mock(return_value={"scope": "CPU test"}))
    monkeypatch.setattr("transfer.dual_view.pipeline.run", registration)
    output = tmp_path / "missing"
    report = ground_skill(package, scene, output, instruction="Use one particular container")
    assert report["status"] == "completed" and report["region_status"] == "missing"
    assert report["reference_status"] == "missing"
    assert report["selection"]["same_entity"] is False
    feature.assert_not_called()
    registration.assert_not_called()
