"""Small raw RGB-D integration tests without GPU models or network requests.

SAM roles and Gemini semantics are supplied at their public boundaries. Image
matching is replaced with independently known pixel correspondences; this does
not measure SD-DINO quality. FoundationPose is supplied at its model-pose
boundary; relative-pose conversion, calibrated backprojection, cross-scene rigid
registration, motion transfer and support goals remain real. These checks do
not establish the pose model's accuracy on a real demonstration.
"""

# BEGIN raw RGB-D manipulation integration tests.
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from manipulation.observations import track_rgbd
from manipulation.perception import ground_rgbd, run_rgbd
from manipulation.pipeline import run_stage


def translation(xyz):
    matrix = np.eye(4)
    matrix[:3, 3] = xyz
    return matrix


class KnownRGBDScene:
    """A textured rigid patch moving in front of a fixed reference plane.

    Carrier depth is 1.2 m and reference depth 1.8 m. The destination reference
    shifts by (4, 2) pixels, corresponding to a known metric translation. The
    destination carrier shifts by (9, 3) pixels instead: its initial placement
    differs from the demonstration, so reference binding and carrier binding
    must not be confused when constructing the goal.
    """

    width, height = 160, 128
    fx, fy = 100., 110.
    carrier_depth, reference_depth = 1.2, 1.8
    reference_shift = np.array([4., 2.])
    carrier_shift = np.array([9., 3.])

    def __init__(self, root, *, camera_motion=False, small_carrier=False):
        import cv2

        self.demo, self.scene = Path(root) / "demo", Path(root) / "scene"
        self.camera_motion = camera_motion
        self.K = np.array([[self.fx, 0., self.width / 2],
                           [0., self.fy, self.height / 2], [0., 0., 1.]])
        rng = np.random.default_rng(472)
        background = cv2.GaussianBlur(rng.integers(0, 256, (self.height, self.width), np.uint8), (3, 3), 0)
        body_height, body_width = (24, 28) if small_carrier else (56, 64)
        body_texture = cv2.GaussianBlur(rng.integers(0, 256, (body_height, body_width), np.uint8), (3, 3), 0)
        self.demo_carrier = np.zeros((self.height, self.width), bool)
        self.demo_carrier[36:36+body_height, 18:18+body_width] = True
        self.demo_reference = np.zeros_like(self.demo_carrier)
        self.demo_reference[8:118, 8:148] = True  # Deliberately includes the part.
        self.scene_carrier = np.zeros_like(self.demo_carrier)
        self.scene_carrier[39:39+body_height, 27:27+body_width] = True
        self.scene_reference = np.zeros_like(self.demo_carrier)
        self.scene_reference[10:120, 12:152] = True
        for directory, frames in ((self.demo, 3), (self.scene, 1)):
            (directory / "rgb").mkdir(parents=True)
            (directory / "depth").mkdir()
            (directory / "intrinsic.json").write_text(json.dumps({
                "width": self.width, "height": self.height,
                "intrinsic_matrix": self.K.ravel(order="F").tolist(),
            }))
            # BEGIN explicit, local pose-model input for the synthetic fixture.
            # The small closed cube only exercises model-file plumbing: no
            # estimator is run against it and it is not a reconstruction of the
            # textured patch. Model identity is never inferred from its label.
            mesh = directory / "fixture_model.obj"
            mesh.write_text(
                "v -0.01 -0.01 -0.01\nv 0.01 -0.01 -0.01\n"
                "v 0.01 0.01 -0.01\nv -0.01 0.01 -0.01\n"
                "v -0.01 -0.01 0.01\nv 0.01 -0.01 0.01\n"
                "v 0.01 0.01 0.01\nv -0.01 0.01 0.01\n"
                "f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\n"
                "f 1 2 6\nf 1 6 5\nf 2 3 7\nf 2 7 6\n"
                "f 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n")
            (directory / "config.json").write_text(json.dumps({
                "depth_scale": 1000,
                "pose_models": {
                    "carrier": {"mesh": mesh.name, "scale": 1.0},
                    "reference": {"mesh": mesh.name, "scale": [1., 1., 1.]},
                },
            }))
            # END explicit local pose-model fixture.
            for index in range(frames):
                if directory == self.demo:
                    gray, x, y = background.copy(), 18 + 2 * index, 36 + index
                    if camera_motion:
                        # A common +0.036 m camera-coordinate translation gives
                        # 2 px on the 1.8 m reference and 3 px on the 1.2 m part.
                        # The reference tracker must remove that common motion.
                        gray = cv2.warpAffine(gray, np.array([[1., 0., 2 * index], [0., 1., 0.]]),
                                              (self.width, self.height))
                        x += 3 * index
                else:
                    gray = cv2.warpAffine(background, np.array([[1., 0., 4.], [0., 1., 2.]]),
                                          (self.width, self.height))
                    x, y = 27, 39
                gray[y:y+body_height, x:x+body_width] = body_texture
                rgb = np.repeat(gray[..., None], 3, axis=2)
                depth = np.full((self.height, self.width), 1800, np.uint16)
                depth[y:y+body_height, x:x+body_width] = 1200
                Image.fromarray(rgb).save(directory / "rgb" / f"{index:05d}.png")
                Image.fromarray(depth).save(directory / "depth" / f"{index:05d}.png")
                if index == 0:
                    if directory == self.demo:
                        self.demo_image = rgb
                    else:
                        self.scene_image = rgb
        self.match_calls = []

    # BEGIN analytic camera-from-model observations at the FoundationPose API.
    def known_model_poses(self, paths, depth_paths, *, K, depth_scale, masks, models):
        """Supply two independently prescribed rigid model poses per RGB frame.

        Both model frames have arbitrary fixed axes and origins. A correct
        adapter must remove common camera motion without treating a model's
        origin as a pixel-centroid landmark or cancelling carrier motion.
        """
        if len(paths) != len(depth_paths) or len(masks) != 2 or len(models) != 2:
            raise AssertionError("Both rigid models must share the same aligned RGB-D timeline")
        np.testing.assert_allclose(K, self.K, atol=0)
        np.testing.assert_allclose(depth_scale, .001, atol=0)
        np.testing.assert_array_equal(masks[0], self.demo_carrier)
        np.testing.assert_array_equal(masks[1], self.demo_reference & ~self.demo_carrier)
        body = translation([-.25, -.12, self.carrier_depth])
        reference = translation([.1, .08, self.reference_depth])
        body[:3, :3] = Rotation.from_euler("xyz", [.2, -.1, .3]).as_matrix()
        reference[:3, :3] = Rotation.from_euler("xyz", [-.15, .25, -.2]).as_matrix()
        poses = []
        for index in range(len(paths)):
            camera = translation([.036 * index if self.camera_motion else 0., 0., 0.])
            moving = translation([2 * index * self.carrier_depth / self.fx,
                                  index * self.carrier_depth / self.fy, 0.])
            poses.append([camera @ moving @ body, camera @ reference])
        return {"camera_from_model": np.asarray(poses),
                "validated": np.ones((len(paths), 2), dtype=bool),
                "diagnostic_source": np.array("analytic API-boundary fixture; no model inference")}
    # END analytic FoundationPose observations.

    def task(self, relation="transfer"):
        return {"relation": relation, "interaction": "none", "motion": "auto",
                "carrier": {"demo_label": "patch", "scene_label": "patch"},
                "reference": {"demo_label": "backing", "scene_label": "backing"},
                "goal_description": "Move the demonstrated patch relative to the fixed backing."}

    def detector(self, image, labels):
        pixels = np.asarray(image)
        if np.array_equal(pixels, self.demo_image):
            masks = [self.demo_carrier, self.demo_reference]
        elif np.array_equal(pixels, self.scene_image):
            masks = [self.scene_carrier, self.scene_reference]
        else:
            raise AssertionError("Detector did not receive an original calibrated image")
        detections = []
        for name, mask in zip(("patch", "backing"), masks):
            yy, xx = np.nonzero(mask)
            detections.append({"label": name, "score": 0.99, "mask": mask.copy(),
                               "bbox": [float(xx.min()), float(yy.min()),
                                        float(xx.max()+1), float(yy.max()+1)]})
        if set(labels) != {"patch", "backing"}:
            raise AssertionError("Unexpected semantic concepts")
        return detections

    @staticmethod
    def _crop_origin(full, cropped):
        """Locate the actual crop, independently of the production padding rule."""
        import cv2

        array = np.asarray(cropped)
        scores = cv2.matchTemplate(full, array, cv2.TM_SQDIFF)
        _, _, location, _ = cv2.minMaxLoc(scores)
        x, y = location
        if not np.array_equal(full[y:y+array.shape[0], x:x+array.shape[1]], array):
            raise AssertionError("Matcher received an image other than an exact observation crop")
        return np.asarray([x, y], float)

    def known_matches(self, first, second, mask1, mask2, pixels, *, extractor=None):
        """Return known physical correspondence, not a mocked metric transform.

        A crop offset error in production therefore reaches real 3D registration
        and can change/fail the asserted transform instead of cancelling in a
        stubbed registration function.
        """
        origin1 = self._crop_origin(self.demo_image, first)
        origin2 = self._crop_origin(self.scene_image, second)
        is_carrier = int(mask1.sum()) < 5000
        shift = self.carrier_shift if is_carrier else self.reference_shift
        targets = np.asarray(pixels) + origin1 + shift - origin2
        nearest = np.rint(targets).astype(int)
        visible = mask2[nearest[:, 1], nearest[:, 0]]
        # The carrier's changed placement can occlude an old reference feature.
        # The boundary stub supplies no confident match for that observation;
        # production must discard it before constructing the 3D rigid fit.
        if not visible.all():
            yy, xx = np.nonzero(mask2)
            targets[~visible] = [xx[0], yy[0]]
        self.match_calls.append({"role": "carrier" if is_carrier else "reference",
                                 "offsets": [origin1.copy(), origin2.copy()]})
        return targets, {"scores": visible.astype(float).tolist(), "numerical_ties": (~visible).tolist(),
                         "source": "test-only known image correspondence"}

    def reference_transform(self):
        return translation([self.reference_shift[0] * self.reference_depth / self.fx,
                            self.reference_shift[1] * self.reference_depth / self.fy, 0.])

    def carrier_transform(self):
        return translation([self.carrier_shift[0] * self.carrier_depth / self.fx,
                            self.carrier_shift[1] * self.carrier_depth / self.fy, 0.])


class RawRGBDPipelineTest(unittest.TestCase):
    def test_grounding_tracks_raw_images_and_preserves_body_reference_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = KnownRGBDScene(directory, camera_motion=True)
            client = Mock()
            client.chat_json.return_value = fixture.task()

            def near_tied(*args, **kwargs):
                targets, diagnostics = fixture.known_matches(*args, **kwargs)
                # Close scores at neighbouring dense pixels are not a failed
                # physical correspondence when the 3D consensus is consistent.
                diagnostics["numerical_ties"] = [True] * len(targets)
                return targets, diagnostics

            with patch("manipulation.perception.match_points", side_effect=near_tied), \
                    patch("local_models.foundationpose.track", side_effect=fixture.known_model_poses), \
                    patch("manipulation.perception.track_rgbd", wraps=track_rgbd) as real_tracker:
                observed = ground_rgbd(fixture.demo, fixture.scene, "Move this patch", client=client,
                                       detector=fixture.detector, extractor=Mock())
            self.assertEqual(real_tracker.call_count, 1)
            used_body = real_tracker.call_args.args[1]
            used_reference = real_tracker.call_args.kwargs["reference_mask"]
            self.assertFalse(np.any(used_body & used_reference))
            self.assertTrue(np.any(fixture.demo_carrier & fixture.demo_reference))
            self.assertEqual({call["role"] for call in fixture.match_calls}, {"carrier", "reference"})
            for call in fixture.match_calls:
                self.assertTrue(all(np.any(offset > 0) for offset in call["offsets"]))
            reference, carrier = fixture.reference_transform(), fixture.carrier_transform()
            np.testing.assert_allclose(observed["scene"]["reference_transform"], reference, atol=1e-12)
            np.testing.assert_allclose(observed["scene"]["initial_pose"], carrier @ observed["demo_poses"][0], atol=1e-12)
            np.testing.assert_allclose(observed["scene"]["preferred_goal_pose"], reference @ observed["demo_poses"][-1], atol=1e-12)
            demo_relative = observed["demo_poses"] @ np.linalg.inv(observed["demo_poses"][0])
            expected = [translation([2 * frame * 1.2 / fixture.fx, frame * 1.2 / fixture.fy, 0.])
                        for frame in range(3)]
            # The independent model frames exercise actual relative-pose
            # conversion; camera motion must not become carrier motion.
            np.testing.assert_allclose(demo_relative[:, :3, 3], np.asarray(expected)[:, :3, 3], atol=0.005)
            np.testing.assert_allclose(demo_relative[:, :3, :3], np.asarray(expected)[:, :3, :3], atol=0.003)
            # Reconstructing scene points from the returned fixed body frame
            # must recover the actual visible destination carrier surface.
            scene = observed["scene"]
            cloud = scene["body_points"] @ scene["initial_pose"][:3, :3].T + scene["initial_pose"][:3, 3]
            np.testing.assert_allclose(cloud[:, 2], 1.2, atol=1e-12)
            result = run_stage(observed)
            self.assertEqual(result["model"]["kind"], "free")
            np.testing.assert_allclose(result["poses"][-1], reference @ observed["demo_poses"][-1], atol=1e-10)
            self.assertIsNone(result["interaction_geometry"])
            self.assertNotIn("function", observed)
            self.assertNotIn("grasp", observed)
            self.assertEqual(observed["reference_frame"], "scene_camera")

    def test_run_rgbd_estimates_support_and_saves_complete_numeric_output(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = KnownRGBDScene(directory, small_carrier=True)
            client = Mock()
            client.chat_json.side_effect = [fixture.task("on"), {"candidate_id": 1}]
            destination = Path(directory) / "run"
            with patch("manipulation.perception.match_points", side_effect=fixture.known_matches), \
                    patch("local_models.foundationpose.track", side_effect=fixture.known_model_poses):
                result = run_rgbd(fixture.demo, fixture.scene, "Place the patch on the backing", destination,
                                  client=client, detector=fixture.detector, extractor=Mock())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(client.chat_json.call_count, 2)
            self.assertEqual(result["goal_checks"]["geometry_source"],
                             "semantically_selected_observed_RGBD_support_patch")
            self.assertLess(result["goal_checks"]["support_gap_m"], 1e-7)
            self.assertLess(result["goal_checks"]["boundary_violation_m"], 1e-7)
            end = result["poses"][-1]
            world = result["body_points"] @ end[:3, :3].T + end[:3, 3]
            self.assertAlmostEqual(float(world[:, 2].max()), 1.8, places=7)
            self.assertTrue(np.all(world[:, 2] <= 1.8 + 1e-7))
            report = json.loads((destination / "run.json").read_text())
            saved = json.loads((destination / "trajectory.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["output_type"], "object_trajectory")
            np.testing.assert_allclose(saved["poses"], result["poses"])
            self.assertTrue((destination / "observations.json").is_file())
            for domain in ("demo", "scene"):
                body = np.asarray(Image.open(destination / domain / "carrier_mask.png")) > 0
                reference = np.asarray(Image.open(destination / domain / "reference_mask.png")) > 0
                self.assertFalse(np.any(body & reference))
            with self.assertRaises(FileExistsError):
                run_rgbd(fixture.demo, fixture.scene, "repeat", destination, client=client)

    # BEGIN pose-estimation failure is never replaced with an old tracker.
    def test_pose_tracking_failure_is_recorded_and_not_reported_as_a_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = KnownRGBDScene(directory)
            client = Mock()
            client.chat_json.return_value = fixture.task()
            destination = Path(directory) / "failed"
            with patch("manipulation.perception.match_points") as matcher, \
                    patch("local_models.foundationpose.track", side_effect=ValueError(
                        "FoundationPose rejected the carrier at frame 1")) as estimator:
                with self.assertRaisesRegex(ValueError, "FoundationPose rejected"):
                    run_rgbd(fixture.demo, fixture.scene, "Move the patch", destination,
                             client=client, detector=fixture.detector, extractor=Mock())
                estimator.assert_called_once()
                matcher.assert_not_called()
            report = json.loads((destination / "run.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("FoundationPose rejected", report["error"])
            self.assertFalse((destination / "trajectory.json").exists())

    def test_semantic_roles_cannot_replace_missing_pose_models(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = KnownRGBDScene(directory)
            (fixture.demo / "config.json").write_text(json.dumps({"depth_scale": 1000}))
            client = Mock(chat_json=Mock(return_value=fixture.task()))
            destination = Path(directory) / "missing-models"
            with patch("manipulation.perception.match_points") as matcher, \
                    patch("local_models.foundationpose.track") as estimator:
                with self.assertRaisesRegex(ValueError, "pose_models"):
                    run_rgbd(fixture.demo, fixture.scene, "Move the patch", destination,
                             client=client, detector=fixture.detector, extractor=Mock())
                estimator.assert_not_called()
                matcher.assert_not_called()
            self.assertEqual(json.loads((destination / "run.json").read_text())["status"], "failed")
            self.assertFalse((destination / "trajectory.json").exists())
    # END explicit pose-estimation/model-input failure.

    # BEGIN container placement needs no additional free-interior geometry.
    def test_inside_transfers_demonstrated_goal_without_inventing_cavity(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = KnownRGBDScene(directory)
            client = Mock()
            client.chat_json.return_value = fixture.task("inside")
            destination = Path(directory) / "inside"
            with patch("manipulation.perception.match_points", side_effect=fixture.known_matches), \
                    patch("local_models.foundationpose.track", side_effect=fixture.known_model_poses):
                result = run_rgbd(fixture.demo, fixture.scene, "Put the object inside its container",
                                  destination, client=client, detector=fixture.detector, extractor=Mock())
            observed = json.loads((destination / "observations.json").read_text())
            initial = fixture.carrier_transform() @ np.asarray(observed["demo_poses"])[0]
            goal = fixture.reference_transform() @ np.asarray(observed["demo_poses"])[-1]
            np.testing.assert_allclose(result["poses"][0], initial, atol=1e-10)
            np.testing.assert_allclose(result["poses"][-1], goal, atol=1e-10)
            self.assertEqual(result["task"]["relation"], "inside")
            self.assertIsNone(result["goal_geometry"])
            self.assertEqual(result["goal_checks"]["containment_status"], "not_verified")
            self.assertNotIn("interior_violation_m", result["goal_checks"])
            self.assertNotIn("minimum_clearance_m", result["goal_checks"])
            self.assertFalse((fixture.scene / "geometry.json").exists())
            # Rendering must not fabricate a box when the cavity is unobserved.
            from manipulation.visualization import _goal_geometry
            geometry, _ = _goal_geometry(result)
            self.assertIsNone(geometry)
    # END ordinary container placement regression.

    def test_ambiguous_cross_scene_matching_cannot_generate_a_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = KnownRGBDScene(directory)
            client = Mock()
            client.chat_json.return_value = fixture.task()

            def ambiguous(*args, **kwargs):
                targets, diagnostics = fixture.known_matches(*args, **kwargs)
                targets[:] = targets[0]
                diagnostics["scores"] = [1.0] * len(targets)
                return targets, diagnostics

            destination = Path(directory) / "ambiguous"
            with patch("manipulation.perception.match_points", side_effect=ambiguous), \
                    patch("local_models.foundationpose.track", side_effect=fixture.known_model_poses):
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    run_rgbd(fixture.demo, fixture.scene, "Move the patch", destination,
                             client=client, detector=fixture.detector, extractor=Mock())
            self.assertEqual(json.loads((destination / "run.json").read_text())["status"], "failed")
            self.assertFalse((destination / "trajectory.json").exists())

    def test_tool_use_semantics_routes_to_preserved_bridge_without_object_tracking(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = KnownRGBDScene(directory)
            # BEGIN the preserved annotated subset does not invoke FoundationPose.
            for root in (fixture.demo, fixture.scene):
                (root / "config.json").write_text(json.dumps({"depth_scale": 1000}))
            # END preserved subset model independence.
            client = Mock()
            client.chat_json.return_value = fixture.task("tool_use")
            destination = Path(directory) / "tool"
            # Only routing is under test: executing the released bridge requires
            # its own annotations/models and is covered by separate regressions.
            expected = {"status": "passed", "output_type": "object_trajectory"}
            with patch("manipulation.tool_use.run_tool_use", return_value=expected) as bridge, \
                    patch("manipulation.perception.track_rgbd") as tracker, \
                    patch("manipulation.perception.select_roles") as selection:
                result = run_rgbd(fixture.demo, fixture.scene, "Use this tool", destination, client=client)
            self.assertEqual(result, expected)
            bridge.assert_called_once_with(fixture.demo, fixture.scene, fixture.task("tool_use"),
                                           "Use this tool", destination, client=client)
            tracker.assert_not_called()
            selection.assert_not_called()
            self.assertEqual(json.loads((destination / "run.json").read_text())["status"], "passed")

    def test_legacy_solver_exit_is_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = KnownRGBDScene(directory)
            client = Mock(chat_json=Mock(return_value=fixture.task("tool_use")))
            destination = Path(directory) / "solver-exit"
            with patch("manipulation.tool_use.run_tool_use", side_effect=SystemExit(1)):
                with self.assertRaises(SystemExit):
                    run_rgbd(fixture.demo, fixture.scene, "Repeat the tool action", destination, client=client)
            report = json.loads((destination / "run.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("SystemExit", report["error"])
            self.assertFalse((destination / "trajectory.json").exists())


if __name__ == "__main__":
    unittest.main()
# END raw RGB-D manipulation integration tests.
