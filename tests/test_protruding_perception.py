"""CPU contracts for observed backing-plane selection, with no model calls.

Small synthetic masks and an injected extractor/JSON client verify evidence
boundaries and selection semantics. The selected IDs are mock decisions: these
tests do not establish actual VLM recognition quality or plane-fit accuracy.
"""

# BEGIN independent local backing-plane semantic and evidence contracts.
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from manipulation.protruding_perception import observed_backing_planes


EXTRACTOR = "manipulation.protruding_perception._plane_candidates"


def observation():
    """Only RGB-D, calibration and the local part are known; no owner exists."""
    image = Image.new("RGB", (80, 60), (35, 70, 105))
    depth = np.ones((60, 80), dtype=float)
    K = np.array([[80., 0., 40.], [0., 80., 30.], [0., 0., 1.]])
    region = np.zeros((60, 80), dtype=bool)
    region[24:30, 32:44] = True
    return image, depth, K, region


def plane(mask, number):
    """Extractor output fixture; its tangent and normal have no semantic sign."""
    y, x = np.nonzero(mask)
    return {
        "candidate_id": number, "mask": mask.copy(),
        "bbox": [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)],
        "origin": np.array([.02, -.01, 1.]),
        "normal": np.array([0., 0., -1.]),
        "first_axis": np.array([-1., 0., 0.]),
        "sample_inlier_count": int(mask.sum()),
        "observed_pixel_count": int(mask.sum()), "plane_rms_m": .0002,
        "fit_evidence": {"observed_only": True, "residuals_m": [.0001, .0003]},
    }


def alternatives():
    large = np.zeros((60, 80), bool)
    large[17:23, 25:49] = True
    small = np.zeros_like(large)
    small[31:35, 38:47] = True
    return [plane(large, 2), plane(small, 9)]


def client_for(number, reason="Visible evidence identifies this backing surface."):
    return Mock(chat_json=Mock(return_value={"candidate_id": number, "reason": reason}))


class ProtrudingPerceptionTest(unittest.TestCase):
    def test_local_observation_can_select_nonlargest_plane_without_an_owner_mask(self):
        args = observation()
        candidates = alternatives()
        self.assertLess(candidates[1]["mask"].sum(), candidates[0]["mask"].sum())
        client = client_for(9)
        with patch(EXTRACTOR, return_value=(candidates, 300)) as extract:
            selected, diagnostics = observed_backing_planes(*args, client=client)
        extract.assert_called_once()
        client.chat_json.assert_called_once()
        self.assertEqual(selected["candidate_id"], 9)
        self.assertEqual(diagnostics["selected_candidate_id"], 9)
        self.assertEqual(diagnostics["candidate_count"], 2)
        self.assertEqual(diagnostics["sampled_point_count"], 300)
        self.assertEqual([row["candidate_id"] for row in diagnostics["candidates"]], [2, 9])
        np.testing.assert_array_equal(selected["mask"], candidates[1]["mask"])
        # This selector must not orient normals or manufacture grasp geometry.
        np.testing.assert_array_equal(selected["normal"], candidates[1]["normal"])
        np.testing.assert_array_equal(selected["first_axis"], candidates[1]["first_axis"])
        self.assertIn("unoriented", diagnostics["normal_sign"])
        for key in ("owner_mask", "pose", "grasp_frame", "holding"):
            self.assertNotIn(key, selected)

    def test_single_plane_still_requires_one_semantic_call(self):
        candidates = alternatives()[:1]
        client = client_for(2, reason="  This is the observed attached surface.  ")
        with patch(EXTRACTOR, return_value=(candidates, 200)):
            selected, diagnostics = observed_backing_planes(*observation(), client=client)
        client.chat_json.assert_called_once()
        self.assertEqual(selected["candidate_id"], 2)
        self.assertEqual(diagnostics["selection_reason"], "This is the observed attached surface.")
        self.assertTrue(diagnostics["vlm_used"])

    def test_extraction_uses_only_local_context_valid_depth_outside_exact_region(self):
        image, depth, K, region = observation()
        # A real mask hole remains observed context, rather than being filled
        # by the region's rectangular envelope. Invalid context stays excluded.
        region[26, 36] = False
        for x, value in zip((25, 26, 27, 28), (0., -1., np.nan, np.inf)):
            depth[17, x] = value
        depth[59, 79] = 100.  # Valid but well outside the declared local window.
        captured = {}

        def extract(seen_depth, seen_K, valid):
            captured["valid"] = valid.copy()
            np.testing.assert_array_equal(seen_depth, depth)
            np.testing.assert_array_equal(seen_K, K)
            return [plane(valid, 2)], int(valid.sum())

        with patch(EXTRACTOR, side_effect=extract):
            _, diagnostics = observed_backing_planes(image, depth, K, region, client=client_for(2))
        # The fixture part is 12 pixels long, yielding a declared 8-pixel
        # margin. These explicit bounds independently test off-by-one errors.
        expected = np.zeros((60, 80), bool)
        expected[16:38, 24:52] = True
        expected &= np.isfinite(depth) & (depth > 0) & ~region
        np.testing.assert_array_equal(captured["valid"], expected)
        self.assertTrue(captured["valid"][26, 36])
        self.assertFalse(np.any(captured["valid"] & region))
        self.assertFalse(captured["valid"][59, 79])
        self.assertEqual(diagnostics["context_box_xyxy"], [24, 16, 52, 38])
        self.assertEqual(diagnostics["context_valid_depth_pixels"], int(expected.sum()))

    def test_context_is_clipped_at_image_edges_without_wrapping(self):
        image, depth, K, region = observation()
        region[:] = False
        region[1:7, 2:14] = True
        captured = {}

        def extract(_depth, _K, valid):
            captured["valid"] = valid.copy()
            return [plane(valid, 2)], int(valid.sum())

        with patch(EXTRACTOR, side_effect=extract):
            _, diagnostics = observed_backing_planes(image, depth, K, region, client=client_for(2))
        expected = np.zeros_like(region)
        expected[:15, :22] = True
        expected &= ~region
        np.testing.assert_array_equal(captured["valid"], expected)
        self.assertEqual(diagnostics["context_box_xyxy"], [0, 0, 22, 15])

    def test_region_depth_can_be_missing_without_claiming_its_geometry_recovered(self):
        image, depth, K, region = observation()
        depth[region] = np.nan
        client = client_for(2)
        with patch(EXTRACTOR, return_value=(alternatives()[:1], 200)):
            selected, diagnostics = observed_backing_planes(image, depth, K, region, client=client)
        client.chat_json.assert_called_once()
        self.assertEqual(diagnostics["interaction_valid_depth_pixels"], 0)
        self.assertGreater(diagnostics["interaction_pixels"], 0)
        self.assertNotIn("interaction_geometry", selected)

    def test_no_plane_candidates_fails_without_vlm_or_window_retry(self):
        client = Mock()
        with patch(EXTRACTOR, return_value=([], 150)) as extract:
            with self.assertRaisesRegex(ValueError, "no sufficiently supported"):
                observed_backing_planes(*observation(), client=client)
        extract.assert_called_once()
        client.chat_json.assert_not_called()

    def test_insufficient_local_depth_fails_before_extraction_and_vlm(self):
        image, _, K, region = observation()
        depth = np.zeros(region.shape)
        depth[16, 24:47] = 1.  # Exactly 23 context samples, all outside the part.
        client = Mock()
        with patch(EXTRACTOR) as extract:
            with self.assertRaisesRegex(ValueError, "fewer than 24"):
                observed_backing_planes(image, depth, K, region, client=client)
        extract.assert_not_called()
        client.chat_json.assert_not_called()

    def test_null_is_explicit_abstention_without_largest_plane_fallback(self):
        client = client_for(None, "The visible attachment is ambiguous.")
        with patch(EXTRACTOR, return_value=(alternatives(), 300)) as extract:
            with self.assertRaisesRegex(ValueError, "No unambiguous.*attachment is ambiguous"):
                observed_backing_planes(*observation(), client=client)
        extract.assert_called_once()
        client.chat_json.assert_called_once()

    def test_unknown_or_coerced_candidate_ids_are_rejected(self):
        for number in (0, -2, 1, 99, True, False, 2.0, "2", [], [2], np.int64(2)):
            client = client_for(number)
            with self.subTest(candidate_id=number), patch(EXTRACTOR, return_value=(alternatives(), 300)):
                with self.assertRaisesRegex(ValueError, "existing observed candidate ID"):
                    observed_backing_planes(*observation(), client=client)
            client.chat_json.assert_called_once()

    def test_malformed_selection_and_vlm_geometry_are_rejected(self):
        invalid = [None, [], {}, {"candidate_id": 2}, {"reason": "yes"},
                   {"candidate_id": 2, "reason": ""},
                   {"candidate_id": 2, "reason": "   "},
                   {"candidate_id": 2, "reason": None},
                   {"candidate_id": 2, "reason": 5},
                   {"candidate_id": 2, "reason": "yes", "normal": [0, 0, -1]},
                   {"candidate_id": 2, "reason": "yes", "pose": np.eye(4).tolist()}]
        for reply in invalid:
            client = Mock(chat_json=Mock(return_value=reply))
            with self.subTest(reply=reply), patch(EXTRACTOR, return_value=(alternatives(), 300)):
                with self.assertRaisesRegex(ValueError, "exactly candidate_id"):
                    observed_backing_planes(*observation(), client=client)
            client.chat_json.assert_called_once()

    def test_invalid_inputs_fail_before_geometry_or_semantic_calls(self):
        image, depth, K, region = observation()
        cases = [
            (np.zeros((60, 80, 3), np.uint8), depth, K, region),
            (image, depth, K, np.zeros_like(region)),
            (image, depth, K, np.ones((3, 4), bool)),
            (image, depth, K, np.full(region.shape, .5)),
            (image, depth, K, np.full(region.shape, np.nan)),
            (image, depth, K, region.astype(complex)),
            (image, np.ones((3, 4)), K, region),
            (image, depth.astype(str), K, region),
            (image, depth.astype(bool), K, region),
            (image, depth.astype(complex), K, region),
            (image, depth, np.zeros((3, 3)), region),
            (image, depth, np.eye(4), region),
            (image, depth, np.full((3, 3), np.nan), region),
        ]
        for index, args in enumerate(cases):
            client = Mock()
            with self.subTest(case=index), patch(EXTRACTOR) as extract:
                with self.assertRaises(ValueError):
                    observed_backing_planes(*args, client=client)
            extract.assert_not_called()
            client.chat_json.assert_not_called()

    def test_selected_plane_is_a_deep_copy_and_inputs_and_diagnostics_remain_observed(self):
        image, depth, K, region = observation()
        candidates = alternatives()
        before = deepcopy(candidates)
        old_image, old_depth, old_K, old_region = np.asarray(image).copy(), depth.copy(), K.copy(), region.copy()
        with patch(EXTRACTOR, return_value=(candidates, 300)):
            selected, diagnostics = observed_backing_planes(image, depth, K, region, client=client_for(9))
        # Downstream sign work and annotations must not rewrite saved evidence.
        selected["normal"] *= -1
        selected["origin"] += 1
        selected["mask"][:] = False
        selected["bbox"][0] = -999
        selected["fit_evidence"]["residuals_m"].append(100)
        for saved, expected in zip(diagnostics["candidates"], before):
            for key in ("mask", "origin", "normal", "first_axis"):
                np.testing.assert_array_equal(saved[key], expected[key])
            self.assertEqual(saved["bbox"], expected["bbox"])
            self.assertEqual(saved["fit_evidence"], expected["fit_evidence"])
        np.testing.assert_array_equal(np.asarray(image), old_image)
        np.testing.assert_array_equal(depth, old_depth)
        np.testing.assert_array_equal(K, old_K)
        np.testing.assert_array_equal(region, old_region)

    def test_prompt_only_sends_candidate_identity_and_display_extent_as_json(self):
        client = client_for(9)
        with patch(EXTRACTOR, return_value=(alternatives(), 300)):
            observed_backing_planes(*observation(), client=client)
        payload = json.loads(client.chat_json.call_args.args[0].split("\nObserved candidates:\n", 1)[1])
        self.assertEqual(payload["allowed_ids"], [2, 9])
        self.assertEqual(set(payload), {"context_box_xyxy", "allowed_ids", "candidates"})
        for row in payload["candidates"]:
            self.assertEqual(set(row), {"candidate_id", "bbox"})

    def test_transport_error_propagates_once_without_geometry_fallback(self):
        client = Mock(chat_json=Mock(side_effect=RuntimeError("injected gateway failure")))
        with patch(EXTRACTOR, return_value=(alternatives(), 300)) as extract:
            with self.assertRaisesRegex(RuntimeError, "injected gateway failure"):
                observed_backing_planes(*observation(), client=client)
        extract.assert_called_once()
        client.chat_json.assert_called_once()
# END independent local backing-plane semantic and evidence contracts.
