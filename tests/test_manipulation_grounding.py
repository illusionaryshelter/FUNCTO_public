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

    # BEGIN preserve simultaneous semantic and motion requirements for routing.
    def test_planar_tool_semantics_are_retained_instead_of_silently_weakened(self):
        response = task()
        response.update(relation="tool_use", interaction="tool", motion="planar")
        result = infer_task("Sweep along the tabletop while keeping contact", self.image,
                            self.image, client=Mock(chat_json=Mock(return_value=response)))
        self.assertEqual(result["relation"], "tool_use")
        self.assertEqual(result["motion"], "planar")
        # The execution entry must reject an unsupported combination explicitly;
        # grounding must not erase a requirement to make that route succeed.
        self.assertEqual(set(result), set(task()))
    # END simultaneous requirements.

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
    # BEGIN native MARCO contracts; synthetic maps require neither CUDA nor weights.
    def test_native_sampling_logits_and_unmasked_decoder_with_bounded_queries(self):
        import torch
        from torch.nn import functional as F
        from local_models import marco

        rng = np.random.default_rng(5823)
        image1, image2 = Image.new("RGB", (420, 210)), Image.new("RGB", (840, 840))
        first = rng.normal(size=(1, 7, 5, 4)).astype(np.float32)
        second = rng.normal(size=(1, 7, 4, 6)).astype(np.float32)
        pixels = rng.uniform([0, 0], [419, 209], size=(35, 2))
        # Independent SciPy interpolation oracle for the native source rule:
        # rounded image scaling followed by feature-size / inference-size.
        locations = pixels * [2., 2.] * [4/840, 5/840]
        vectors = np.column_stack([map_coordinates(channel, locations[:, ::-1].T,
            order=1, mode="nearest", prefilter=False) for channel in first[0]])
        logits = vectors @ second[0].reshape(7, -1)
        native = marco._geometry().kernel_softargmax_get_matches_logits(
            torch.from_numpy(logits.reshape(1, 35, 4, 6)), 10., 7)
        expected = marco._geometry().scaling_coordinates(native, (4, 6), (840, 840))[0].numpy()
        with patch("torch.nn.functional.grid_sample", wraps=F.grid_sample) as sampler:
            result, diagnostic = match_points(image1, image2, np.ones((210, 420), bool),
                np.ones((840, 840), bool), pixels, extractor=Mock(return_value=(first, second)))
        np.testing.assert_allclose(result, expected, atol=2e-4, rtol=0)
        np.testing.assert_allclose(diagnostic["scores"], logits.max(axis=1), atol=2e-6, rtol=0)
        self.assertEqual([call.args[1].shape[1] for call in sampler.call_args_list], [32, 3])
        self.assertEqual(diagnostic["matcher"], "MARCO")
        self.assertEqual(diagnostic["candidate_count"], 24)

    def test_default_provider_is_marco_and_preserves_nonsquare_rgb(self):
        image1, image2 = Image.new("RGB", (420, 210), "red"), Image.new("RGB", (210, 420), "blue")
        features = np.ones((1, 2, 4, 4), np.float32)
        mask = np.zeros((420, 210), bool)
        mask[280, 140] = True
        with patch("local_models.marco.extract_pair", return_value=(features, features)) as extractor:
            result, diagnostic = match_points(image1, image2, np.ones((210, 420), bool), mask, [[100, 30]])
        # BEGIN native output scaling is FP32, not exact rational arithmetic.
        np.testing.assert_allclose(result, [[839/6, 839/3]],
                                   atol=np.spacing(np.float32(840)), rtol=0)
        # END one resized-coordinate ULP bounds this arithmetic discrepancy.
        self.assertEqual([image.size for image in extractor.call_args.args], [(420, 210), (210, 420)])
        self.assertEqual(extractor.call_args.args[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(diagnostic["candidate_count"], 1)
        self.assertEqual(diagnostic["second_scores"], [None])

    def test_negative_logits_cannot_choose_a_forbidden_maximum(self):
        image = Image.new("RGB", (840, 840))
        first = np.ones((1, 1, 2, 2), np.float32)
        second = np.full_like(first, 1000.)
        second[0, 0, 0, 0] = -1.
        mask = np.zeros((840, 840), bool); mask[0, 0] = True
        result, diagnostic = match_points(image, image, np.ones_like(mask), mask, [[0, 0]],
                                          extractor=Mock(return_value=(first, second)))
        np.testing.assert_array_equal(result, [[0, 0]])
        self.assertEqual(diagnostic["scores"], [-1.])
        self.assertEqual(diagnostic["nonpositive_scores"], [True])

    def test_unsupported_thin_mask_fails_without_snapping_or_fallback(self):
        image = Image.new("RGB", (840, 840))
        features = np.ones((1, 2, 4, 4), np.float32)
        mask = np.zeros((840, 840), bool); mask[420, 420] = True
        with self.assertRaisesRegex(ValueError, "No MARCO feature cell"):
            match_points(image, image, np.ones_like(mask), mask, [[0, 0]],
                         extractor=Mock(return_value=(features, features)))

    def test_masked_softargmax_in_a_hole_fails_instead_of_inventing_support(self):
        image = Image.new("RGB", (840, 840))
        features = np.zeros((1, 2, 2, 2), np.float32)
        mask = np.eye(840, dtype=bool)
        mask[1:-1] = False  # Two distant admissible cells; their mean is not observed.
        with self.assertRaisesRegex(ValueError, "not snapped"):
            match_points(image, image, np.ones_like(mask), mask, [[0, 0]],
                         extractor=Mock(return_value=(features, features)))

    def test_duplicate_predictions_and_tied_logits_remain_visible(self):
        image = Image.new("RGB", (840, 840))
        features = np.ones((1, 2, 4, 4), np.float32)
        result, diagnostic = match_points(image, image, np.ones((840, 840), bool),
            np.ones((840, 840), bool), [[0, 0], [839, 839]], extractor=Mock(return_value=(features, features)))
        np.testing.assert_array_equal(result[0], result[1])
        self.assertEqual(diagnostic["numerical_ties"], [True, True])
        self.assertEqual(diagnostic["duplicate_target_groups"], [[0, 1]])

    def test_source_mask_uses_depth_backprojection_tie_rounding(self):
        image = Image.new("RGB", (2, 2))
        mask = np.zeros((2, 2), bool); mask[0, 0] = True
        features = np.ones((1, 2, 2, 2), np.float32)
        result, _ = match_points(image, image, mask, mask, [[0.5, 0]],
                                 extractor=Mock(return_value=(features, features)))
        np.testing.assert_array_equal(result, [[0, 0]])

    def test_invalid_masks_queries_and_feature_contracts_fail_explicitly(self):
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
        for pair in ((features.astype(np.float64), features), (features, features*float("nan")),
                     (features[:, :, :1], features), (features, features[:, :1]), (features,)):
            with self.subTest(shapes=[x.shape for x in pair]), self.assertRaises(ValueError):
                match_points(image, image, mask, mask, [[0, 0]], extractor=Mock(return_value=pair))

    def test_no_required_anchors_skips_extraction(self):
        image, mask, extractor = Image.new("RGB", (2, 2)), np.ones((2, 2), bool), Mock()
        result, diagnostic = match_points(image, image, mask, mask, np.empty((0, 2)), extractor=extractor)
        self.assertEqual(result.shape, (0, 2))
        self.assertFalse(diagnostic["extractor_called"])
        extractor.assert_not_called()
    # END native MARCO correspondence contracts.


if __name__ == "__main__":
    unittest.main()
# END independent manipulation-grounding regression tests.
