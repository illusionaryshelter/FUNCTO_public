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
    # BEGIN current entry evidence accompanies the tool identity in real input.
    (scene / "robot_feedback.json").write_text(json.dumps({"tool_id": "observed_tool",
        "tool_frame": "control_frame", "eef_pose": np.eye(4).tolist()}))
    # END fixture uses the same feedback contract as the frozen scene.
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


# BEGIN reject an explicitly contradictory attachment before expensive work.
def test_current_attachment_tool_identity_cannot_conflict_with_feedback(tmp_path, monkeypatch):
    package, scene = inputs(tmp_path)
    detector = Mock(side_effect=AssertionError("Invalid current coordinates must fail before detection"))
    monkeypatch.setattr("local_models.sam31.detect", detector)
    report = ground_skill(package, scene, tmp_path / "conflicting", instruction="Use the observed container",
        relations={"A": np.eye(4).tolist(), "A_source": "measured current attachment",
                   "A_context": {"tool_frame": "different_control_frame"}})
    assert report["status"] == "failed" and report["failed_stage"] == "validate_current_relations"
    assert "A_context.tool_frame" in report["error"]
    detector.assert_not_called()
# END the test requires no additional observations or model calls.


# BEGIN the formal export preserves nominal geometry and explicit tool boundaries.
def test_successful_entry_exports_nominal_path_and_declared_tool_boundaries(tmp_path, monkeypatch):
    from scipy.spatial.transform import Rotation
    package, scene = inputs(tmp_path)
    actual, function = np.eye(4), np.eye(4)
    actual[:3, 3], function[:3, 3] = [-.2, .1, .6], [.4, -.3, .8]
    actual[:3, :3] = Rotation.from_euler("y", .4).as_matrix()
    function[:3, :3] = Rotation.from_euler("z", -.3).as_matrix()
    feedback = json.loads((scene / "robot_feedback.json").read_text())
    feedback["eef_pose"] = actual.tolist()
    (scene / "robot_feedback.json").write_text(json.dumps(feedback))
    monkeypatch.setattr("local_models.sam31.detect", Mock(return_value=[detection("container")]))
    selection = {"views": {name: {"candidate_ids": [17], "reason": "Same observed fixture"}
                             for name in ("view_a", "view_b")},
                 "same_entity": True, "same_entity_reason": "Shared fixture identity"}
    monkeypatch.setattr("local_models.gemini.Gemini.chat_json", Mock(return_value=selection))
    monkeypatch.setattr("local_models.marco.extract_image", Mock(return_value=np.ones((1, 3, 4, 4))))
    monkeypatch.setattr("transfer.dual_view.pipeline.prepare", Mock(return_value={"scope": "CPU fixture"}))
    monkeypatch.setattr("config.ground_skill.run_region_transfer", Mock(return_value={
        "status": "accepted_observed_region", "reasons": [],
        "reference_frame": {"T_world_region": function}, "outputs": None}))
    output = tmp_path / "success"
    report = ground_skill(package, scene, output, instruction="Use the observed container")
    assert report["status"] == "completed" and report["reference_status"] == "reference"
    with np.load(output / "reference.npz", allow_pickle=False) as saved:
        nominal_start = saved["eef_poses"][0].copy()
        assert np.linalg.norm(nominal_start - actual) > .1
        np.testing.assert_allclose(saved["eef_poses"][-1], function, atol=1e-12)
        np.testing.assert_allclose(saved["eef_poses"], saved["bound_eef_poses"], atol=1e-12)
    declaration = report["reference_boundary_checks"]["eef"]
    assert declaration["status"] == "declared_for_physical_time_binding"
    np.testing.assert_allclose(declaration["initial_pose"], actual, atol=1e-12)
    np.testing.assert_allclose(declaration["nominal_interaction"], function, atol=1e-12)
    assert report["planning_reference"]["status"] == "nominal_path_and_boundaries"
    boundary = report["reference_boundaries"]["eef"]
    np.testing.assert_allclose(boundary["initial_pose"], actual, atol=1e-12)
    np.testing.assert_allclose(boundary["goals"][0]["pose"], function, atol=1e-12)
    assert boundary["goals"][0]["progress"] == 1.
    np.testing.assert_allclose(np.load(output / "planner-reference-samples.npy")[0], nominal_start, atol=1e-12)
    html = (output / "index.html").read_text()
    panels = json.loads(html.split("const panels=", 1)[1].split(";const images=", 1)[0])
    for panel in panels[1:]:
        markers = {row["kind"]: row for row in panel["markers"]}
        assert set(markers) == {"declared_entrance", "nominal_start", "nominal_event"}
        calibration = json.loads((scene / panel["id"] / "calibration.json").read_text())
        K = np.asarray(calibration["K"])
        for name, pose in (("declared_entrance", actual), ("nominal_start", nominal_start),
                           ("nominal_event", function)):
            pixel = K @ pose[:3, 3]
            np.testing.assert_allclose(markers[name]["pixel"], pixel[:2]/pixel[2], atol=1e-12)
        assert markers["nominal_event"]["progress"] == 1.
    assert "物体名义路径" in html and "工具名义路径" in html
# END the real generator, native path adapter and HTML consume one boundary schema.
