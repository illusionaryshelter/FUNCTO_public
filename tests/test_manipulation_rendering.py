"""Coordinate oracles for the object renderer; no model, GPU, or GUI use."""

# BEGIN rendering checks independent of legacy function/grasp landmarks.
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from manipulation.visualization import render_result, transform_points


def example_stage():
    initial, goal = np.eye(4), np.eye(4)
    initial[:3, 3] = [1, -2, 0.5]
    goal[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    goal[:3, 3] = [2, 3, 0.7]
    # Body coordinates deliberately have nonzero mean and no privileged point
    # at the origin. Recentring these would silently change the pose convention.
    points = np.array([[0.2, 0.3, 0.1], [0.3, 0.3, 0.1], [0.2, 0.5, 0.1], [0.2, 0.3, 0.4]])
    return {
        "poses": np.stack((initial, goal)), "body_points": points, "goal_pose": goal,
        "frame_indices": [10, 20], "carrier_id": "drawer", "reference_id": "cabinet",
        "reference_frame": "registered_camera", "model": {"kind": "free"},
        "task": {"relation": "transfer", "interaction": "push"},
        "goal_checks": {"provided_error_m": np.float64(0.)},
    }


class RenderingCoordinatesTest(unittest.TestCase):
    def test_column_transform_preserves_nonzero_local_origin_and_rigidity(self):
        stage = example_stage()
        body, goal = stage["body_points"], stage["goal_pose"]
        expected = (goal @ np.column_stack((body, np.ones(len(body)))).T).T[:, :3]
        moved = transform_points(body, goal)
        np.testing.assert_allclose(moved, expected, atol=1e-12)
        np.testing.assert_allclose(moved[0], [1.7, 3.2, 0.8], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(moved[:, None] - moved[None], axis=-1),
                                   np.linalg.norm(body[:, None] - body[None], axis=-1), atol=1e-12)
        # An alternate pivot at the cloud centroid must not match this result.
        self.assertGreater(np.linalg.norm(moved.mean(0) - goal[:3, 3]), 0.1)

    def test_non_tool_stage_renders_and_writes_numeric_coordinate_evidence(self):
        stage = example_stage()
        original = copy.deepcopy(stage)
        with tempfile.TemporaryDirectory() as folder:
            paths = render_result(stage, folder)
            self.assertEqual(set(paths), {"trajectory_png", "coordinate_checks_json"})
            with Image.open(paths["trajectory_png"]) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 500)
            report = json.loads(Path(paths["coordinate_checks_json"]).read_text())["stages"][0]
        self.assertEqual(report["snapshot_frame_ids"], [10, 20])
        self.assertEqual(report["goal_geometry"]["status"], "not_supplied")
        self.assertLess(report["max_probe_pair_distance_error_m"], 1e-12)
        self.assertLess(report["endpoint_position_error_m"], 1e-12)
        self.assertLess(report["endpoint_rotation_error_rad"], 1e-12)
        np.testing.assert_allclose(report["probe_reference_points_m"][1],
                                   transform_points(stage["body_points"], stage["goal_pose"]))
        np.testing.assert_array_equal(stage["body_points"], original["body_points"])
        np.testing.assert_array_equal(stage["poses"], original["poses"])
        self.assertNotIn("function_point", stage)
        self.assertNotIn("grasp_point", stage)

    def test_stages_keep_distinct_body_frames_and_snapshot_budget(self):
        first, second = example_stage(), example_stage()
        first["poses"] = np.repeat(first["poses"][:1], 20, axis=0)
        first["goal_pose"] = first["poses"][-1]
        first["frame_indices"] = np.arange(20)
        first["body_points"] = np.tile(first["body_points"], (1000, 1))
        second["carrier_id"], second["reference_id"] = "cup", "drawer"
        second["reference_frame"] = "another_declared_frame"
        second["body_points"] = second["body_points"] + [10, 0, 0]
        with tempfile.TemporaryDirectory() as folder:
            paths = render_result({"stages": [first, second]}, folder)
            report = json.loads(Path(paths["coordinate_checks_json"]).read_text())["stages"]
        self.assertEqual([item["carrier_id"] for item in report], ["drawer", "cup"])
        self.assertEqual(report[1]["reference_frame"], "another_declared_frame")
        self.assertLessEqual(len(report[0]["snapshot_indices"]) + 1, 6)
        self.assertEqual(report[0]["displayed_body_points"], 2500)
        self.assertGreater(report[1]["probe_body_points_m"][0][0], 10)

    def test_intermediate_goal_event_is_not_mistaken_for_the_endpoint(self):
        stage = example_stage()
        poses = np.repeat(stage["poses"][:1], 20, axis=0)
        poses[7] = stage["goal_pose"]
        poses[-1] = stage["goal_pose"]
        poses[-1, 0, 3] += 1.0
        stage.update(poses=poses, frame_indices=np.arange(100, 120), goal_frame_index=7)
        with tempfile.TemporaryDirectory() as folder:
            paths = render_result(stage, folder)
            report = json.loads(Path(paths["coordinate_checks_json"]).read_text())["stages"][0]
        self.assertEqual(report["goal_frame_id"], 107)
        self.assertIn(7, report["snapshot_indices"])
        self.assertEqual(len(report["snapshot_indices"]), 5)
        self.assertAlmostEqual(report["goal_position_error_m"], 0.0)
        self.assertAlmostEqual(report["goal_rotation_error_rad"], 0.0)
        self.assertAlmostEqual(report["endpoint_position_error_m"], 1.0)

    def test_background_requires_and_applies_carrier_membership(self):
        stage = example_stage()
        carrier_cloud = transform_points(stage["body_points"], stage["poses"][0])
        stage["scene_points"] = np.vstack((carrier_cloud, [[-1, 0, 0], [-1, 1, 0]]))
        with tempfile.TemporaryDirectory() as folder:
            paths = render_result(stage, folder)
            omitted = json.loads(Path(paths["coordinate_checks_json"]).read_text())["stages"][0]["background"]
            self.assertEqual(omitted["status"], "omitted")
            stage["scene_carrier_mask"] = np.array([True] * len(carrier_cloud) + [False, False])
            paths = render_result(stage, folder)
            shown = json.loads(Path(paths["coordinate_checks_json"]).read_text())["stages"][0]["background"]
        self.assertEqual(shown["removed_carrier_points"], 4)
        self.assertEqual(shown["displayed_points"], 2)

    def test_supplied_support_is_transformed_from_its_actual_plane_coordinates(self):
        stage = example_stage()
        stage["task"]["relation"] = "on"
        stage["goal_geometry"] = {
            "origin": [1, 2, 3], "basis": [[0, 0], [1, 0], [0, 1]], "normal": [1, 0, 0],
            "boundary": [[0, 0], [1, 0], [1, 2], [0, 2]], "source": "observed_patch",
        }
        with tempfile.TemporaryDirectory() as folder:
            paths = render_result(stage, folder)
            geometry = json.loads(Path(paths["coordinate_checks_json"]).read_text())["stages"][0]["goal_geometry"]
        vertices = np.asarray(geometry["vertices_reference_m"])
        np.testing.assert_allclose(vertices[:, 0], 1)
        np.testing.assert_allclose(vertices.min(0), [1, 2, 3])
        np.testing.assert_allclose(vertices.max(0), [1, 3, 5])
        self.assertEqual(geometry["source"], "observed_patch")

    def test_known_halfspaces_render_exact_interior_without_guessing_outer_box(self):
        stage = example_stage()
        stage["task"]["relation"] = "inside"
        # The known free interior is [0,1] x [0,2] x [0,3], unrelated to the
        # trajectory's display limits or an inferred object bounding box.
        normals = np.vstack((np.eye(3), -np.eye(3)))
        stage["goal_geometry"] = {
            "halfspaces": np.column_stack((normals, [-1, -2, -3, 0, 0, 0])),
            "source": "observed_free_interior",
        }
        with tempfile.TemporaryDirectory() as folder:
            paths = render_result(stage, folder)
            geometry = json.loads(Path(paths["coordinate_checks_json"]).read_text())["stages"][0]["goal_geometry"]
        self.assertEqual(geometry["status"], "displayed")
        vertices = np.asarray(geometry["vertices_reference_m"])
        np.testing.assert_allclose(vertices.min(0), [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(vertices.max(0), [1, 2, 3], atol=1e-12)

    def test_invalid_pose_and_mismatched_frame_ids_are_rejected(self):
        for invalid in ("reflection", "nan", "frame_ids", "negative_frame_ids", "unsigned_reversed"):
            stage = example_stage()
            if invalid == "reflection":
                stage["poses"][0, 0, 0] = -1
            elif invalid == "nan":
                stage["body_points"][0, 0] = np.nan
            elif invalid == "negative_frame_ids":
                stage["frame_indices"] = [-1, 20]
            elif invalid == "unsigned_reversed":
                stage["frame_indices"] = np.array([2**64 - 1, 0], dtype=np.uint64)
            else:
                stage["frame_indices"] = [20, 10]
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    render_result(stage, folder)
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_large_unsigned_frame_ids_do_not_overflow_signed_comparisons(self):
        stage = example_stage()
        stage["frame_indices"] = np.array([0, 2**64 - 1], dtype=np.uint64)
        with tempfile.TemporaryDirectory() as folder:
            paths = render_result(stage, folder)
            report = json.loads(Path(paths["coordinate_checks_json"]).read_text())["stages"][0]
        self.assertEqual(report["snapshot_frame_ids"], [0, 2**64 - 1])
        self.assertEqual(report["goal_frame_id"], 2**64 - 1)


if __name__ == "__main__":
    unittest.main()
# END rendering checks independent of legacy function/grasp landmarks.
