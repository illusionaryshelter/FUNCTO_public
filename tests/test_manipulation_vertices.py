"""CPU checks for measured vertex hypotheses and explicit VLM associations.

These tests exercise actual OpenCV observation on synthetic RGB-D and injected
corner edge cases. The JSON client is mocked: no camera, dataset, credentials,
GPU, object truth, model call, or recognition-success claim is involved.
"""

# BEGIN visible-vertex observation and association contracts.
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from manipulation import vertices
from manipulation.vertices import observe_vertex_candidates, associate_vertex_candidates


def rgbd_fixture():
    rgb = np.full((64, 64, 3), 20, np.uint8)
    rgb[20:44, 20:44] = 230
    owner = np.zeros((64, 64), bool)
    owner[20:44, 20:44] = True
    depth = np.full((64, 64), 1.2)
    K = np.array([[200., 0., 32.], [0., 200., 32.], [0., 0., 1.]])
    return Image.fromarray(rgb), depth, K, owner


def observation(ids):
    """Fabricate already-observed pixels only to isolate response validation."""
    rows = []
    for index, number in enumerate(ids):
        u, v = 12+7*index, 14+4*index
        rows.append({
            "id": number, "status": "candidate", "initial_pixel": [u, v],
            "corner_pixel": [u+.2, v+.1], "depth_pixel": [u, v],
            "depth_m": 1.2, "point_camera": [(u-32)*1.2/200, (v-32)*1.2/200, 1.2],
            "corner_response": .1,
            "measurement_scope": "actual depth pixel point, not an exact 3D vertex",
        })
    return {"image_size": [64, 64], "candidates": rows, "candidate_count": len(rows)}


def response(pairs=(), *, status="matched"):
    return {"status": status,
            "pairs": [{"source_id": a, "target_id": b, "reason": "Visible edge structure supports this pair."}
                      for a, b in pairs],
            "source_rejected_ids": [], "target_rejected_ids": [],
            "reason": "Visible support or an explicit observation limitation."}


def associate(source, current, answer):
    image = Image.new("RGB", (64, 64), (50, 80, 120))
    mask = np.ones((64, 64), bool)
    client = Mock(chat_json=Mock(return_value=answer))
    result = associate_vertex_candidates(image, image, mask, mask, source, current, client=client)
    return result, client


class VertexObservationTest(unittest.TestCase):
    def test_native_rgb_detection_preserves_actual_depth_pixel_geometry(self):
        image, depth, K, owner = rgbd_fixture()
        originals = np.asarray(image).copy(), depth.copy(), K.copy(), owner.copy()
        report = observe_vertex_candidates(image, depth, K, owner)
        self.assertEqual(report["image_size"], [64, 64])
        self.assertEqual(report["status"], "observed")
        self.assertGreater(report["candidate_count"], 0)
        self.assertLessEqual(report["candidate_count"], 32)
        self.assertFalse(report["plane_or_box_fit_used"])
        for row in report["candidates"]:
            if row["status"] != "candidate":
                continue
            u, v = row["depth_pixel"]
            expected = np.linalg.inv(K) @ [u, v, 1.] * depth[v, u]
            np.testing.assert_allclose(row["point_camera"], expected, atol=1e-12)
            self.assertTrue(owner[v, u])
            self.assertIn("not an exact 3D vertex", row["measurement_scope"])
        for actual, original in zip((np.asarray(image), depth, K, owner), originals):
            np.testing.assert_array_equal(actual, original)

    def patched_observation(self, initial, refined, *, owner=None):
        image, depth, K, default_owner = rgbd_fixture()
        owner = np.ones_like(default_owner) if owner is None else owner
        scores = np.full(depth.shape, .1, np.float32)
        for index, (u, v) in enumerate(initial):
            scores[int(v), int(u)] = .1+.1*index
        with patch.object(vertices.cv2, "goodFeaturesToTrack", return_value=np.asarray(initial, np.float32)[:, None]), \
             patch.object(vertices.cv2, "cornerSubPix", return_value=np.asarray(refined, np.float32)[:, None]), \
             patch.object(vertices.cv2, "cornerMinEigenVal", return_value=scores):
            return observe_vertex_candidates(image, depth, K, owner)

    def test_subpixel_convergence_keeps_one_measured_vote(self):
        report = self.patched_observation([[10, 10], [14, 10]], [[12., 10.], [13.3, 10.]])
        self.assertEqual(report["candidate_count"], 1)
        rejected, kept = report["candidates"]
        self.assertEqual(rejected["reason"], "subpixel convergence duplicate")
        self.assertNotEqual(rejected["depth_pixel"], kept["depth_pixel"])
        self.assertEqual(rejected["rejected_duplicate_of"], kept["id"])

    def test_distinct_corner_locations_cannot_reuse_one_raw_depth_sample(self):
        owner = np.zeros((64, 64), bool)
        owner[12, 12] = True
        report = self.patched_observation([[11, 12], [14, 12]], [[10.9, 12.], [13.1, 12.]], owner=owner)
        self.assertEqual(report["candidate_count"], 1)
        first, second = report["candidates"]
        self.assertGreater(np.linalg.norm(np.asarray(first["corner_pixel"])-second["corner_pixel"]), 2)
        self.assertEqual(first["reason"], "duplicate raw-depth sample")
        self.assertEqual(first["depth_pixel"], [12, 12])
        self.assertEqual(second["depth_pixel"], [12, 12])

    def test_nearby_depth_proxy_uses_its_own_ray_not_the_rgb_corner_ray(self):
        report = self.patched_observation([[12, 12]], [[12.4, 12.3]])
        row = report["candidates"][0]
        self.assertEqual(row["depth_pixel"], [12, 12])
        self.assertGreater(row["depth_sample_distance_px"], 0)
        np.testing.assert_allclose(row["point_camera"], [-.12, -.12, 1.2], atol=1e-12)

    def test_large_refinement_is_rejected_without_replacement_geometry(self):
        report = self.patched_observation([[10, 10]], [[14., 10.]])
        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(report["status"], "insufficient")
        self.assertNotIn("point_camera", report["candidates"][0])

    def test_unavailable_depth_and_texture_do_not_invent_corners(self):
        image, depth, K, owner = rgbd_fixture()
        depth[owner] = np.nan
        with patch.object(vertices.cv2, "goodFeaturesToTrack") as detector:
            report = observe_vertex_candidates(image, depth, K, owner)
        self.assertEqual(report["candidate_count"], 0)
        detector.assert_not_called()
        report = observe_vertex_candidates(Image.new("RGB", image.size), np.ones_like(depth), K, owner)
        self.assertEqual(report["candidate_count"], 0)

    def test_invalid_shape_mask_intrinsics_are_rejected(self):
        image, depth, K, owner = rgbd_fixture()
        arguments = [(image, depth[:-1], K, owner), (image, depth, K, owner.astype(np.uint8)),
                     (image, depth, np.zeros((3, 3)), owner),
                     (image, depth, K, np.zeros_like(owner))]
        for args in arguments:
            with self.subTest(args=[np.shape(item) for item in args]), self.assertRaises(ValueError):
                observe_vertex_candidates(*args)


class VertexAssociationTest(unittest.TestCase):
    def test_variable_pair_count_is_not_a_three_point_recipe(self):
        for count in (1, 2, 5):
            source_ids = list(range(10, 10+count))
            target_ids = list(range(30, 30+count))
            with self.subTest(count=count):
                result, client = associate(observation(source_ids), observation(target_ids), response(zip(source_ids, target_ids)))
                self.assertEqual(result["status"], "matched")
                self.assertEqual(len(result["pairs"]), count)
                self.assertEqual(client.chat_json.call_count, 1)

    def test_ids_define_pairs_without_implicit_sorted_or_image_order(self):
        source, current = observation([37, 4, 90]), observation([31, 5, 92])
        proposed = [(90, 5), (37, 92), (4, 31)]
        answer = response(proposed)
        result, _ = associate(source, current, answer)
        self.assertEqual([(row["source_id"], row["target_id"]) for row in result["pairs"]], proposed)
        self.assertEqual(result["pairs"][0]["source"]["id"], 90)
        self.assertEqual(result["pairs"][0]["target"]["id"], 5)
        source["candidates"].reverse()
        current["candidates"].reverse()
        reordered, _ = associate(source, current, answer)
        self.assertEqual(reordered["pairs"], result["pairs"])

    def test_prompt_only_sends_ids_and_rgb_not_recorded_metric_or_task_metadata(self):
        source, current = observation([7]), observation([19])
        source["private_source_pose"] = "DO_NOT_SEND"
        current["skill_name"] = "DO_NOT_SEND"
        result, client = associate(source, current, response([(7, 19)]))
        prompt = client.chat_json.call_args.args[0]
        self.assertTrue(prompt.startswith(vertices._ASSOCIATION_INSTRUCTION))
        self.assertNotIn("DO_NOT_SEND", prompt)
        payload = prompt.split("\nCandidate IDs:\n", 1)[1]
        self.assertNotIn("point_camera", payload)
        self.assertNotIn("depth", payload)
        self.assertEqual(len(client.chat_json.call_args.kwargs["images"]), 4)
        self.assertEqual(result["trace"]["native_image_sizes"], [[64, 64], [64, 64]])

    def test_unresolved_status_is_preserved_without_actionable_pairs(self):
        for status in ("ambiguous", "insufficient"):
            with self.subTest(status=status):
                result, _ = associate(observation([1]), observation([9]), response(status=status))
                self.assertEqual(result["status"], status)
                self.assertEqual(result["pairs"], [])

    def test_no_observed_candidates_skips_model_call(self):
        result, client = associate(observation([]), observation([9]), response([(1, 9)]))
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["trace"]["vlm_calls"], 0)
        self.assertIsNone(result["selection"])
        client.chat_json.assert_not_called()

    def test_malformed_association_cannot_introduce_or_reuse_measurements(self):
        good = response([(1, 9)])
        cases = []
        for field, value in (("status", "unknown"), ("status", []), ("reason", ""),
                             ("pairs", []), ("source_rejected_ids", [1]),
                             ("target_rejected_ids", [9]), ("source_rejected_ids", [2, 2]),
                             ("source_rejected_ids", [True]), ("target_rejected_ids", [88])):
            item = deepcopy(good)
            item[field] = value
            cases.append(item)
        for pair in ({"source_id": True, "target_id": 9, "reason": "x"},
                     {"source_id": 1, "target_id": 88, "reason": "x"},
                     {"source_id": 1, "target_id": 9, "reason": ""}):
            item = deepcopy(good)
            item["pairs"] = [pair]
            cases.append(item)
        cases.extend([response([(1, 9), (1, 8)]), response([(1, 9), (2, 9)]),
                      response([(1, 9)], status="ambiguous"), {**good, "pose": np.eye(4).tolist()}])
        for answer in cases:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                associate(observation([1, 2]), observation([9, 8]), answer)

    def test_persisted_duplicate_depth_measurements_are_rejected_before_vlm(self):
        source = observation([1, 2])
        source["candidates"][1]["depth_pixel"] = source["candidates"][0]["depth_pixel"].copy()
        image = Image.new("RGB", (64, 64))
        mask = np.ones((64, 64), bool)
        client = Mock()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            associate_vertex_candidates(image, image, mask, mask, source, observation([9, 8]), client=client)
        client.chat_json.assert_not_called()

    def test_selection_and_input_records_are_not_mutated(self):
        source, current = observation([1]), observation([9])
        answer = response([(1, 9)])
        original = deepcopy((source, current, answer))
        result, _ = associate(source, current, answer)
        result["pairs"][0]["source"]["point_camera"][0] = 999.
        result["selection"]["reason"] = "modified by caller"
        self.assertEqual((source, current, answer), original)


if __name__ == "__main__":
    unittest.main()
# END visible-vertex observation and association contracts.
