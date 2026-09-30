"""CPU checks for VLM query proposals; responses are mocks, not visual results.

The tests inspect the real prompt, annotated images, program-defined search
windows and failure boundaries. They do not contact Gemini, load weights, or
claim that a real VLM can identify these synthetic shapes.
"""

# BEGIN coarse source-query proposal contracts.
import json
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from manipulation.grounding import propose_query_regions


class QueryProposalTest(unittest.TestCase):
    def setUp(self):
        yy, xx = np.indices((100, 100))
        rgb = np.stack((xx+30, yy+40, np.full_like(xx, 130)), axis=-1).astype(np.uint8)
        self.source_image = Image.fromarray(rgb)
        self.current_image = Image.fromarray(rgb[:, ::-1].copy())
        self.source_owner = np.zeros((100, 100), bool)
        self.source_owner[5:95, 5:95] = True
        self.source_region = np.zeros((100, 100), bool)
        self.source_region[12:30, 12:30] = True
        self.current_owner = np.zeros((100, 100), bool)
        # Only grid cells 1 and 25 contain owner pixels. Empty cells must never
        # become selectable merely because they have a valid row/column index.
        self.current_owner[5:19, 5:19] = True
        self.current_owner[80:95, 80:95] = True
        self.pixels = np.array([[20., 20.], [65., 65.], [80., 30.]])
        self.description = "Contact the demonstrated working edge; preserve its opening direction."
        self.query_descriptions = ["contact anchor", "direction toward opening", "plane orientation witness"]

    def call(self, response=None, **changes):
        client = Mock()
        client.chat_json.return_value = response if response is not None else {"cells": [1, 1, 25]}
        arguments = {
            "source_image": self.source_image, "current_image": self.current_image,
            "source_owner_mask": self.source_owner, "current_owner_mask": self.current_owner,
            "source_region_mask": self.source_region, "query_pixels": self.pixels,
            "description": self.description, "query_descriptions": self.query_descriptions,
            "client": client,
        }
        arguments.update(changes)
        return propose_query_regions(**arguments), client

    def test_one_real_prompt_consumes_semantics_and_preserves_owner_context(self):
        result, client = self.call()
        client.chat_json.assert_called_once()
        prompt = client.chat_json.call_args.args[0]
        details = json.loads(prompt.split("\n", 1)[1])
        self.assertEqual(details["description"], self.description)
        self.assertEqual([item["description"] for item in details["queries"]], self.query_descriptions)
        self.assertEqual([item["id"] for item in details["queries"]], [1, 2, 3])
        self.assertEqual([item["id"] for item in details["allowed_cells"]], [1, 25])
        self.assertIn("outside the red region", prompt)
        self.assertIn("occluded", prompt)
        self.assertIn("return null", prompt)
        self.assertIn("do not estimate a precise mask", prompt)
        images = client.chat_json.call_args.kwargs["images"]
        self.assertEqual([image.size for image in images], [(480, 480), (480, 480)])
        self.assertEqual([image.mode for image in images], ["RGB", "RGB"])
        # The provided interaction area is highlighted, while an unmarked owner
        # interior pixel remains unchanged. The source is not cropped to its part.
        reference_panel = np.asarray(self.source_image.resize((480, 480), Image.Resampling.BICUBIC))
        self.assertFalse(np.array_equal(np.asarray(images[0])[74, 74], reference_panel[74, 74]))
        np.testing.assert_array_equal(np.asarray(images[0])[242, 242], reference_panel[242, 242])
        self.assertFalse(np.array_equal(np.asarray(images[0])[309:319, 309:319], reference_panel[309:319, 309:319]))
        self.assertEqual(result["diagnostics"]["client_calls"], 1)

    def test_same_cell_queries_are_allowed_and_windows_are_clipped_half_open(self):
        result, _ = self.call()
        self.assertEqual(result["boxes"].dtype.kind, "i")
        np.testing.assert_array_equal(result["boxes"], [[0, 0, 25, 25], [0, 0, 25, 25], [75, 75, 100, 100]])
        self.assertEqual(result["diagnostics"]["cell_boxes"], [[0, 0, 20, 20], [0, 0, 20, 20], [80, 80, 100, 100]])
        self.assertIn("right and bottom exclusive", result["diagnostics"]["box_convention"])
        self.assertIn("not a current interaction mask", result["diagnostics"]["scope"])
        # A coarse window may include background. Only the caller's intersection
        # with owner and valid depth defines the actual dense search candidates.
        x0, y0, x1, y1 = result["boxes"][0]
        self.assertTrue(self.current_owner[y0:y1, x0:x1].any())
        self.assertFalse(self.current_owner[y0:y1, x0:x1].all())

    def test_source_queries_outside_part_are_valid_and_inputs_remain_unchanged(self):
        self.assertFalse(self.source_region[65, 65])
        image_before = [np.asarray(self.source_image).copy(), np.asarray(self.current_image).copy()]
        array_before = [value.copy() for value in (self.source_owner, self.current_owner, self.source_region, self.pixels)]
        result, _ = self.call()
        self.assertEqual(result["diagnostics"]["query_count"], 3)
        for actual, before in zip((self.source_image, self.current_image), image_before):
            np.testing.assert_array_equal(np.asarray(actual), before)
        for actual, before in zip((self.source_owner, self.current_owner, self.source_region, self.pixels), array_before):
            np.testing.assert_array_equal(actual, before)

    def test_null_correspondence_stops_without_guessing(self):
        with self.assertRaisesRegex(ValueError, "Unresolved.*query 2"):
            self.call({"cells": [1, None, 25]})

    def test_nonowner_unknown_bool_and_string_cell_ids_are_rejected(self):
        for number in (2, 0, 26, True, "1", 1.0, []):
            with self.subTest(number=number), self.assertRaisesRegex(ValueError, "Invalid current-owner cell"):
                self.call({"cells": [number, 1, 25]})

    def test_malformed_response_or_extra_metric_fields_are_rejected(self):
        responses = [[], {}, {"cells": [1]}, {"cells": "1,1,25"},
                     {"cells": [1, 1, 25], "axis": [1, 0, 0]}, {"boxes": [[0, 0, 1, 1]]}]
        for response in responses:
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, "exactly one cells list"):
                self.call(response)

    def test_invalid_source_evidence_is_rejected_before_client_call(self):
        outside_region = np.zeros_like(self.source_region)
        outside_region[:2, :2] = True
        variants = [
            {"source_region_mask": outside_region},
            {"source_region_mask": np.zeros((100, 100), bool)},
            {"current_owner_mask": np.zeros((100, 100), bool)},
            {"source_owner_mask": np.ones((100, 100)) * .5},
            {"current_owner_mask": np.ones((100, 99), bool)},
            {"query_pixels": [[0, 0], [65, 65], [80, 30]]},
            {"query_pixels": [[20, np.nan], [65, 65], [80, 30]]},
            {"query_pixels": [[-1, 20], [65, 65], [80, 30]]},
            {"query_pixels": [[100, 20], [65, 65], [80, 30]]},
            {"query_pixels": np.empty((0, 2))},
            {"query_pixels": np.ones((3, 2), bool)},
            {"description": " "},
            {"query_descriptions": ["one"]},
            {"query_descriptions": ["contact", " ", "plane"]},
        ]
        for changes in variants:
            client = Mock()
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.call(client=client, **changes)
            client.chat_json.assert_not_called()

    def test_noisy_source_region_is_preserved_and_spill_is_diagnosed(self):
        source_region = self.source_region.copy()
        source_region[4, 20] = True
        before = source_region.copy()
        result, client = self.call(source_region_mask=source_region)
        self.assertEqual(result["diagnostics"]["source_interaction_pixels_outside_owner"], 1)
        prompt = client.chat_json.call_args.args[0]
        self.assertIn("selected owner takes priority", prompt)
        self.assertEqual(json.loads(prompt.split("\n", 1)[1])["source_interaction_pixels_outside_owner"], 1)
        np.testing.assert_array_equal(source_region, before)

    def test_small_crops_have_readable_panels_but_native_search_boxes(self):
        source_image, current_image = Image.new("RGB", (50, 60)), Image.new("RGB", (60, 40))
        source_owner, current_owner = np.ones((60, 50), bool), np.ones((40, 60), bool)
        source_region = np.zeros_like(source_owner)
        source_region[5:20, 5:20] = True
        result, client = self.call(source_image=source_image, current_image=current_image,
                                   source_owner_mask=source_owner, current_owner_mask=current_owner,
                                   source_region_mask=source_region, query_pixels=[[10, 10], [30, 20], [20, 40]])
        self.assertEqual([image.size for image in client.chat_json.call_args.kwargs["images"]],
                         [(400, 480), (480, 320)])
        np.testing.assert_array_equal(result["boxes"], [[0, 0, 15, 10], [0, 0, 15, 10], [45, 30, 60, 40]])
        self.assertEqual(result["diagnostics"]["display"]["source"],
                         {"native_size": [50, 60], "panel_size": [400, 480], "scale_xy": [8., 8.]})
        self.assertEqual(result["diagnostics"]["display"]["current"],
                         {"native_size": [60, 40], "panel_size": [480, 320], "scale_xy": [8., 8.]})
        self.assertEqual(source_image.size, (50, 60))
        self.assertEqual(current_image.size, (60, 40))

    def test_small_images_skip_empty_grid_cells_and_keep_positive_windows(self):
        image = Image.new("RGB", (3, 2), (70, 90, 110))
        source_owner = np.ones((2, 3), bool)
        current_owner = np.zeros((2, 3), bool)
        current_owner[1, 2] = True
        result, _ = self.call(
            {"cells": [25]}, source_image=image, current_image=image,
            source_owner_mask=source_owner, current_owner_mask=current_owner,
            source_region_mask=source_owner, query_pixels=[[0, 0]], query_descriptions=["working edge"],
        )
        self.assertEqual(result["diagnostics"]["allowed_cells"], [{"id": 25, "box_xyxy": [2, 1, 3, 2]}])
        np.testing.assert_array_equal(result["boxes"], [[1, 0, 3, 2]])

    # BEGIN observed current parts guide semantics without excluding owner cells.
    def test_optional_part_enters_request_without_changing_query_domain(self):
        baseline, baseline_client = self.call()
        mask = np.zeros_like(self.current_owner)
        mask[8:11, 8:11] = True
        mask[4, 8] = True  # Deliberate segmentation spill; do not trim it away.
        evidence = {"mask": mask, "owner_id": "selected_object",
                    "description": "the observed working edge",
                    "source": "detector candidate 2, semantic interaction-region selection"}
        before = mask.copy()
        result, client = self.call(current_interaction_evidence=evidence)
        prompt = client.chat_json.call_args.args[0]
        context = json.loads(prompt.split("\n", 1)[1])
        actual = context["current_interaction_evidence"]
        self.assertIn("BLUE", prompt)
        self.assertIn("queries need not lie inside", prompt)
        self.assertEqual(actual["description"], evidence["description"])
        self.assertEqual(actual["source"], evidence["source"])
        self.assertEqual(actual["mask_pixel_count"], 10)
        self.assertEqual(actual["pixels_outside_owner"], 1)
        self.assertEqual(actual["cell_coverage"],
                         [{"cell_id": 1, "part_owner_pixels": 9},
                          {"cell_id": 25, "part_owner_pixels": 0}])
        # Cell 25 has no part pixels and is still a valid direction witness.
        self.assertEqual(context["allowed_cells"], baseline["diagnostics"]["allowed_cells"])
        np.testing.assert_array_equal(result["boxes"], baseline["boxes"])
        self.assertEqual(result["diagnostics"]["current_interaction_evidence"], actual)
        before_images = baseline_client.chat_json.call_args.kwargs["images"]
        after_images = client.chat_json.call_args.kwargs["images"]
        np.testing.assert_array_equal(after_images[0], before_images[0])
        self.assertFalse(np.array_equal(after_images[1], before_images[1]))
        np.testing.assert_array_equal(mask, before)

    def test_invalid_part_observation_fails_before_semantic_request(self):
        good = {"mask": self.current_owner.copy(), "owner_id": "object",
                "description": "working region", "source": "observed selected candidate"}
        variants = [None, {**good, "owner_id": " "}, {**good, "description": ""},
                    {**good, "source": []}, {**good, "mask": np.ones((100, 99), bool)},
                    {**good, "mask": good["mask"].astype(float)},
                    {**good, "mask": ~self.current_owner}, {**good, "pose": np.eye(4)}]
        # None is the declared absence case and retains the existing interface.
        baseline, old_client = self.call()
        absent, absent_client = self.call(current_interaction_evidence=variants.pop(0))
        self.assertEqual(old_client.chat_json.call_args.args[0], absent_client.chat_json.call_args.args[0])
        np.testing.assert_array_equal(baseline["boxes"], absent["boxes"])
        for evidence in variants:
            model = Mock()
            with self.subTest(fields=list(evidence)), self.assertRaises(ValueError):
                self.call(current_interaction_evidence=evidence, client=model)
            model.chat_json.assert_not_called()
    # END optional observed-part request contracts.


if __name__ == "__main__":
    unittest.main()
# END coarse source-query proposal contracts.
