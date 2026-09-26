"""CPU contracts for semantic grounding and geometric descriptor correspondence.

All perception/HTTP boundaries are injected: these tests must not load weights,
initialize CUDA, or contact the gateway.
"""

# BEGIN independent manipulation-grounding regression tests.
import copy
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from manipulation import grounding as grounding_module
from manipulation.grounding import infer_task, match_points, select_roles


def task():
    return {
        "relation": "on", "interaction": "grasp", "motion": "free",
        "carrier": {"demo_label": "cup", "scene_label": "vessel"},
        "reference": {"demo_label": "table", "scene_label": "shelf"},
        "goal_description": "Move the left vessel onto the visible support.",
    }


def detection(label, xyxy, shape=(6, 10)):
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = xyxy
    mask[y1:y2, x1:x2] = True
    return {"label": label, "score": 0.8, "bbox": list(xyxy), "mask": mask}


class TaskGroundingTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (10, 6))

    def test_final_demo_context_and_no_mandatory_tool_points(self):
        client = Mock()
        response = task()
        response.update(relation="articulation", interaction="push", motion="auto")
        client.chat_json.return_value = response
        result = infer_task("Complete the demonstrated push", self.image, self.image,
                            demo_final_image=self.image, client=client)
        self.assertEqual(result, response)
        self.assertEqual(len(client.chat_json.call_args.kwargs["images"]), 3)
        self.assertIn("demonstration final scene", client.chat_json.call_args.args[0])
        self.assertNotIn("grasp_point", result)

    def test_invalid_semantics_and_vlm_geometry_are_rejected(self):
        invalid = []
        for field, value in [("motion", "drawer"), ("relation", None), ("interaction", []),
                             ("axis", [0, 0, 1]), ("goal_description", "")]:
            item = task()
            item[field] = value
            invalid.append(item)
        item = task()
        item["carrier"]["demo_label"] = " "
        invalid.append(item)
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                infer_task("Move", self.image, self.image,
                           client=Mock(chat_json=Mock(return_value=response)))

    def test_single_candidates_skip_vlm_and_keep_overlapping_part_masks(self):
        response = task()
        response["carrier"] = dict(demo_label="door", scene_label="door")
        response["reference"] = dict(demo_label="cabinet", scene_label="cabinet")
        detections = [detection("door", (3, 1, 7, 5)), detection("cabinet", (1, 0, 9, 6))]
        detector, client = Mock(return_value=detections), Mock()
        roles = select_roles(self.image, response, detector=detector, client=client)
        self.assertEqual(roles["carrier"]["label"], "door")
        self.assertEqual(roles["reference"]["label"], "cabinet")
        client.chat_json.assert_not_called()
        self.assertEqual(detector.call_args.args[1], ["door", "cabinet"])
        self.assertNotIn("candidate_id", detections[0])

    def test_ambiguous_instances_follow_validated_ids_instead_of_score(self):
        detections = [detection("vessel", (0, 0, 2, 2)),
                      detection("vessel", (5, 0, 8, 3)), detection("shelf", (0, 4, 10, 6))]
        detections[0]["score"], detections[1]["score"] = 0.99, 0.5
        client = Mock(chat_json=Mock(return_value={"carrier": 2, "reference": 3}))
        roles = select_roles(self.image, task(), domain="scene",
                             detector=Mock(return_value=detections), client=client)
        self.assertEqual(roles["carrier"]["candidate_id"], 2)
        self.assertEqual(len(client.chat_json.call_args.kwargs["images"]), 2)
        self.assertIn('"allowed_ids"', client.chat_json.call_args.args[0])

    def test_selection_rejects_wrong_role_bool_unknown_and_unresolved_ids(self):
        detections = [detection("cup", (0, 0, 2, 2)), detection("cup", (5, 0, 8, 3)),
                      detection("table", (0, 4, 10, 6))]
        for response in ({"carrier": 3, "reference": 1}, {"carrier": True, "reference": 3},
                         {"carrier": 10, "reference": 3}, {"carrier": None, "reference": 3},
                         {"carrier": 1, "reference": 3, "axis": [1, 0, 0]}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                select_roles(self.image, task(), detector=Mock(return_value=detections),
                             client=Mock(chat_json=Mock(return_value=response)))

    def test_missing_identical_and_malformed_instances_stop(self):
        valid = [detection("cup", (0, 0, 2, 2)), detection("table", (0, 4, 10, 6))]
        same = [detection("cup", (0, 0, 2, 2)), detection("table", (0, 0, 2, 2))]
        variants = [valid[:1], same]
        for field, value in [("bbox", [0, 0, 11, 5]), ("score", np.nan),
                             ("mask", np.zeros((6, 10))), ("mask", np.ones((6, 10)) * 0.5)]:
            malformed = copy.deepcopy(valid)
            malformed[0][field] = value
            variants.append(malformed)
        for detections in variants:
            with self.subTest(detections=detections), self.assertRaises(ValueError):
                select_roles(self.image, task(), detector=Mock(return_value=detections))

    def test_same_label_roles_need_distinct_instances(self):
        response = task()
        response["reference"] = response["carrier"].copy()
        detection1 = detection("cup", (0, 0, 2, 2))
        with self.assertRaisesRegex(ValueError, "same instance"):
            select_roles(self.image, response, detector=Mock(return_value=[detection1]))


class PointMatchingTest(unittest.TestCase):
    # BEGIN exhaustive-pixel regressions, including an independent dense oracle.
    @staticmethod
    def dense_sample(features, pixels, image_size):
        """Independent tiny reference using SciPy's linear image interpolation.

        Production code must not allocate this dense target descriptor matrix;
        this helper is limited to the small synthetic fixture below.
        """
        width, height = image_size
        size = max(width, height)
        pad = np.array([(size-width)//2, (size-height)//2])
        locations = (pixels + pad + 0.5) / size
        locations = locations * np.array(features.shape[-2:][::-1]) - 0.5
        return np.column_stack([
            map_coordinates(channel.astype(float), locations[:, ::-1].T,
                            order=1, mode="nearest", prefilter=False)
            for channel in features[0]
        ])

    def test_all_pixel_search_matches_independent_dense_cosine_reference(self):
        rng = np.random.default_rng(5823)
        image1, image2 = Image.new("RGB", (17, 9)), Image.new("RGB", (37, 23))
        first = rng.normal(size=(1, 7, 5, 4)).astype(np.float32)
        second = rng.normal(size=(1, 7, 4, 6)).astype(np.float32)
        pixels = rng.uniform([0, 0], [16, 8], size=(35, 2))
        mask1 = np.ones((9, 17), bool)
        mask2 = rng.uniform(size=(23, 37)) > 0.05
        mask2[:, :4] = False
        mask2[:, -4:] = False
        self.assertGreater(mask2.sum(), 512)
        target_y, target_x = np.nonzero(mask2)
        target_pixels = np.column_stack((target_x, target_y))
        source_vectors = self.dense_sample(first, pixels, image1.size)
        target_vectors = self.dense_sample(second, target_pixels, image2.size)
        source_vectors /= np.linalg.norm(source_vectors, axis=1, keepdims=True)
        target_vectors /= np.linalg.norm(target_vectors, axis=1, keepdims=True)
        dense_scores = np.clip(source_vectors @ target_vectors.T, -1, 1)
        expected_indices = np.argmax(dense_scores, axis=1)
        expected_scores = np.max(dense_scores, axis=1)
        expected_seconds = np.partition(dense_scores, -2, axis=1)[:, -2]
        with patch("manipulation.grounding._sample", wraps=grounding_module._sample) as sampler:
            output, diagnostic = match_points(
                image1, image2, mask1, mask2, pixels,
                extractor=Mock(return_value=(first, second)),
            )
        np.testing.assert_array_equal(output, target_pixels[expected_indices])
        np.testing.assert_allclose(diagnostic["scores"], expected_scores, atol=2e-14, rtol=0)
        np.testing.assert_allclose(diagnostic["second_scores"], expected_seconds, atol=2e-14, rtol=0)
        np.testing.assert_allclose(diagnostic["margins"], expected_scores-expected_seconds,
                                   atol=3e-14, rtol=0)
        self.assertEqual(diagnostic["candidate_count"], int(mask2.sum()))
        self.assertEqual(diagnostic["masked_pixel_count"], int(mask2.sum()))
        self.assertEqual(diagnostic["excluded_zero_descriptors"], 0)
        # Both dimensions exceed a batch. Verify the actual sampler calls obey
        # the promised memory bound, rather than trusting a diagnostic constant.
        source_counts, target_counts = [], []
        for call in sampler.call_args_list:
            if call.args[0].shape[-2:] == (5, 4):
                source_counts.append(len(call.args[1]))
            else:
                target_counts.append(len(call.args[1]))
        self.assertEqual(source_counts, [32, 3])
        self.assertGreater(len(target_counts), 2)
        self.assertLessEqual(max(target_counts), 512)

    def test_cosine_peak_between_feature_cell_centers_is_not_discarded(self):
        image = Image.new("RGB", (9, 9))
        source = np.ones((1, 2, 2, 2), np.float32)
        target = np.zeros_like(source)
        target[0, 0, :, 0] = 1
        target[0, 1, :, 1] = 1
        mask2 = np.zeros((9, 9), bool)
        mask2[4, :] = True
        output, diagnostic = match_points(
            image, image, np.ones((9, 9), bool), mask2, [[2, 2]],
            extractor=Mock(return_value=(source, target)),
        )
        # Native cell centers map to x=1.75 and 6.25. Their nearest pixels 2/6
        # both miss the unique cosine maximum at the central original pixel 4.
        np.testing.assert_array_equal(output, [[4, 4]])
        self.assertEqual(diagnostic["candidate_count"], 9)
        self.assertAlmostEqual(diagnostic["scores"][0], 1, places=13)
        self.assertFalse(diagnostic["numerical_ties"][0])

    def test_zero_candidates_and_top_two_ties_across_target_batches(self):
        image = Image.new("RGB", (1025, 1))
        source = np.array([[[[1]], [[0]]]], dtype=np.float32)
        target = np.zeros((1, 2, 1, 1025), np.float32)
        target[0, 0, 0, :] = -1
        target[0, :, 0, 1:11] = 0
        target[0, 0, 0, [511, 1024]] = 1
        mask = np.ones((1, 1025), bool)
        output, diagnostic = match_points(
            image, image, mask, mask, [[0, 0], [1024, 0]],
            extractor=Mock(return_value=(source, target)),
        )
        np.testing.assert_array_equal(output, [[511, 0], [511, 0]])
        self.assertEqual(diagnostic["candidate_count"], 1015)
        self.assertEqual(diagnostic["excluded_zero_descriptors"], 10)
        self.assertEqual(diagnostic["scores"], [1., 1.])
        self.assertEqual(diagnostic["second_scores"], [1., 1.])
        self.assertEqual(diagnostic["numerical_ties"], [True, True])
        self.assertEqual(diagnostic["duplicate_target_groups"], [[0, 1]])
    # END exhaustive-pixel regressions.

    def test_negative_similarity_cannot_choose_outside_mask(self):
        image = Image.new("RGB", (2, 2))
        source = np.tile(np.array([1.0, 0.0], dtype=np.float32)[None, :, None, None], (1, 1, 2, 2))
        target = np.zeros_like(source)
        target[0, :, :, :] = np.array([1.0, 0.0])[:, None, None]
        target[0, :, 1, 0], target[0, :, 1, 1] = [-1.0, 0.0], [-0.5, np.sqrt(0.75)]
        target_mask = np.array([[False, False], [True, True]])
        result, diagnostic = match_points(image, image, np.ones((2, 2), bool), target_mask,
                                          [[0, 0]], extractor=Mock(return_value=(source, target)))
        np.testing.assert_array_equal(result, [[1, 1]])
        self.assertAlmostEqual(diagnostic["scores"][0], -0.5, places=6)
        self.assertEqual(diagnostic["nonpositive_scores"], [True])

    def test_nonsquare_images_have_exact_padding_and_single_pixel_candidates(self):
        image1, image2 = Image.new("RGB", (4, 2), "red"), Image.new("RGB", (2, 4), "blue")
        source = np.zeros((1, 2, 4, 4), dtype=np.float32)
        target = np.zeros_like(source)
        # Original (2,0) becomes square (2,1); target (1,3) becomes (2,3).
        source[0, :, 1, 2] = [0.6, 0.8]
        target[0, :, 3, 2] = [0.6, 0.8]
        mask2 = np.zeros((4, 2), bool)
        mask2[3, 1] = True
        extractor = Mock(return_value=(source, target))
        result, diagnostic = match_points(image1, image2, np.ones((2, 4), bool), mask2,
                                          [[2, 0]], extractor=extractor)
        np.testing.assert_array_equal(result, [[1, 3]])
        self.assertEqual(diagnostic["padding_xy"], [[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual([im.size for im in extractor.call_args.args], [(4, 4), (4, 4)])
        self.assertEqual(extractor.call_args.args[0].getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(diagnostic["second_scores"], [None])

    def test_bilinear_source_vector_at_subpixel_location(self):
        image = Image.new("RGB", (2, 2))
        source = np.zeros((1, 2, 2, 2), np.float32)
        source[0, :, :, 0], source[0, :, :, 1] = [[1], [0]], [[0], [1]]
        target = np.zeros_like(source)
        target[0, :, 0, 0], target[0, :, 1, 1] = [1, 0], [1, 1]
        mask = np.eye(2, dtype=bool)
        result, diagnostic = match_points(image, image, np.ones((2, 2), bool), mask,
                                          [[0.5, 0]], extractor=Mock(return_value=(source, target)))
        np.testing.assert_array_equal(result, [[1, 1]])
        self.assertAlmostEqual(diagnostic["scores"][0], 1, places=6)

    def test_subpixel_mask_lookup_matches_depth_backprojection_rounding(self):
        image = Image.new("RGB", (2, 2))
        mask = np.zeros((2, 2), bool)
        mask[0, 0] = True
        features = np.ones((1, 2, 2, 2), np.float32)
        # observations.backproject samples depth[0, 0] at the .5 tie; semantic
        # mask validity must use the same rounding rule before backprojection.
        result, _ = match_points(image, image, mask, mask, [[0.5, 0]],
                                 extractor=Mock(return_value=(features, features)))
        np.testing.assert_array_equal(result, [[0, 0]])

    def test_ambiguity_and_many_to_one_are_reported_without_fake_unique_points(self):
        image = Image.new("RGB", (2, 2))
        features = np.ones((1, 2, 2, 2), np.float32)
        result, diagnostic = match_points(image, image, np.ones((2, 2), bool),
                                          np.ones((2, 2), bool), [[0, 0], [1, 1]],
                                          extractor=Mock(return_value=(features, features)))
        np.testing.assert_array_equal(result, [[0, 0], [0, 0]])
        self.assertEqual(diagnostic["numerical_ties"], [True, True])
        self.assertEqual(diagnostic["duplicate_target_groups"], [[0, 1]])

    def test_thin_mask_between_feature_centers_retains_a_real_candidate(self):
        image = Image.new("RGB", (8, 8))
        mask = np.zeros((8, 8), bool)
        mask[0, 7] = True
        features = np.ones((1, 2, 2, 2), np.float32)
        result, diagnostic = match_points(image, image, mask, mask, [[7, 0]],
                                          extractor=Mock(return_value=(features, features)))
        np.testing.assert_array_equal(result, [[7, 0]])
        self.assertEqual(diagnostic["candidate_count"], 1)

    def test_invalid_masks_pixels_and_zero_descriptors_fail_explicitly(self):
        image, mask = Image.new("RGB", (2, 2)), np.ones((2, 2), bool)
        features = np.ones((1, 2, 2, 2), np.float32)
        extractor = Mock(return_value=(features, features))
        for pixels in ([[np.nan, 0]], [[2, 0]], [[-0.1, 0]], [0, 1]):
            with self.subTest(pixels=pixels), self.assertRaises(ValueError):
                match_points(image, image, mask, mask, pixels, extractor=extractor)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            match_points(image, image, mask, ~mask, [[0, 0]], extractor=extractor)
        with self.assertRaisesRegex(ValueError, "mask1"):
            match_points(image, image, np.eye(2, dtype=bool), mask, [[1, 0]], extractor=extractor)
        extractor.assert_not_called()
        for pair in ((features, np.zeros_like(features)), (np.zeros_like(features), features)):
            with self.subTest(pair=pair), self.assertRaisesRegex(ValueError, "descriptor"):
                match_points(image, image, mask, mask, [[0, 0]], extractor=Mock(return_value=pair))

    def test_no_required_anchors_skips_extraction(self):
        image, mask, extractor = Image.new("RGB", (2, 2)), np.ones((2, 2), bool), Mock()
        result, diagnostic = match_points(image, image, mask, mask, np.empty((0, 2)), extractor=extractor)
        self.assertEqual(result.shape, (0, 2))
        self.assertFalse(diagnostic["extractor_called"])
        extractor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
# END independent manipulation-grounding regression tests.
