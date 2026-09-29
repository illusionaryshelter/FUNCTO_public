"""CPU contracts for reusable, view-local owner and interaction selection.

Synthetic masks and an injected JSON client exercise routing and validation;
they do not measure Gemini recognition quality. No model, GPU, credentials,
network request, recorded asset, depth recovery or pose estimator is used.
"""

# BEGIN independent owner/interaction candidate selection contracts.
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from manipulation import selection
from manipulation.selection import select_grounding_candidates


def candidate(box, *, number=None, label="object", score=.8):
    """An actual candidate owns its own mask; no object-wide union is implied."""
    mask = np.zeros((10, 12), dtype=bool)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = True
    result = {"label": label, "score": score, "bbox": list(box), "mask": mask}
    if number is not None:
        result["candidate_id"] = number
    return result


def fixture():
    source_mask = np.zeros((10, 12), dtype=bool)
    source_mask[2:5, 3:7] = True
    source = {
        "image": Image.new("RGB", (12, 10), (40, 70, 110)),
        "mask": source_mask,
        "description": "The demonstration contacted the upper surface of the left object.",
        "query_descriptions": ["demonstrated contact area"],
        "owner_mask": np.ones((10, 12), dtype=bool),
    }
    views = {"left": {
        "image": Image.new("RGB", (12, 10), (130, 80, 30)),
        "candidates": [candidate((1, 1, 5, 5), number=7, score=.99),
                       candidate((7, 4, 11, 9), number=31, score=.25)],
    }}
    return source, views


def response(ids, *, view="left", reason="The visible evidence supports this selection."):
    return {"views": {view: {"candidate_ids": ids, "reason": reason}}}


def context_from(client):
    prompt = client.chat_json.call_args.args[0]
    prefix, payload = prompt.split("\nInput data:\n", 1)
    return prefix, json.loads(payload)


def nested_keys(value):
    if isinstance(value, dict):
        return set(value).union(*(nested_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(nested_keys(item) for item in value))
    return set()


class GroundingCandidateSelectionTest(unittest.TestCase):
    def test_fixed_prompt_separates_current_instruction_and_demonstration_evidence(self):
        source, views = fixture()
        # Rich source records must not leak unrelated identifiers or geometry
        # into a semantic-selection prompt. Only declared evidence is needed.
        source["source_id"] = "DO_NOT_SEND_SOURCE_INSTANCE_ID"
        source["T_world_object"] = np.eye(4)
        requests = ("Place the held cup on the right object's support surface.",
                    "Grasp the handle of the right object before pulling it.")
        prefixes = []
        for instruction in requests:
            client = Mock(chat_json=Mock(return_value=response([31])))
            result = select_grounding_candidates(source, instruction, views, selection_scope="interaction_region", client=client)
            prefix, data = context_from(client)
            prefixes.append(prefix)
            self.assertEqual(result["views"]["left"]["status"], "selected")
            self.assertEqual(result["selection_scope"], "interaction_region")
            self.assertEqual(data["current_instruction"], instruction)
            self.assertEqual(data["selection_scope"], "interaction_region")
            self.assertEqual(data["source_evidence"], {
                "description": source["description"],
                "query_descriptions": source["query_descriptions"],
            })
            self.assertIn("views", data)
            self.assertIn("image_order", data)
            self.assertNotIn("score", nested_keys(data))
            self.assertNotIn("mask", nested_keys(data))
            self.assertNotIn("T_world_object", nested_keys(data))
            self.assertNotIn("DO_NOT_SEND_SOURCE_INSTANCE_ID", client.chat_json.call_args.args[0])
        self.assertEqual(prefixes, [selection._SELECTION_INSTRUCTION] * 2)

    def test_scopes_are_explicit_data_with_the_same_fixed_prompt_and_distinct_masks(self):
        source, views = fixture()
        self.assertGreater(source["owner_mask"].sum(), source["mask"].sum())
        instruction = "Ground the next requested action from the observed candidates."
        prefixes = []
        for scope in ("owner", "interaction_region"):
            client = Mock(chat_json=Mock(return_value=response([7])))
            result = select_grounding_candidates(source, instruction, views,
                                                 selection_scope=scope, client=client)
            prefix, data = context_from(client)
            prefixes.append(prefix)
            self.assertEqual(result["selection_scope"], scope)
            self.assertEqual(data["selection_scope"], scope)
            self.assertEqual(data["current_instruction"], instruction)
            self.assertEqual(data["source_evidence"]["description"], source["description"])
            np.testing.assert_array_equal(result["views"]["left"]["regions"][0]["mask"],
                                          views["left"]["candidates"][0]["mask"])
        self.assertEqual(prefixes, [selection._SELECTION_INSTRUCTION] * 2)
        # The common rules must name both meanings; scope selection belongs to
        # explicit data, not a task-name branch or a silently changed template.
        self.assertIn("owner", prefixes[0])
        self.assertIn("interaction_region", prefixes[0])

    def test_owner_scope_requires_owner_evidence_and_unknown_scope_is_rejected(self):
        source, views = fixture()
        minimal = {key: source[key] for key in ("image", "mask", "description")}
        client = Mock(chat_json=Mock(return_value=response([7])))
        result = select_grounding_candidates(minimal, "Use its demonstrated working region", views,
                                             selection_scope="interaction_region", client=client)
        self.assertTrue(result["vlm_used"])
        self.assertEqual(result["views"]["left"]["candidate_ids"], [7])
        client.chat_json.assert_called_once()
        # An interaction mask cannot stand in for the containing owner mask.
        for owner in (None, np.zeros((10, 12), bool), np.ones((2, 2), bool)):
            current_source = dict(minimal)
            if owner is not None:
                current_source["owner_mask"] = owner
            client = Mock()
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                select_grounding_candidates(current_source, "Select the demonstrated owner", views,
                                             selection_scope="owner", client=client)
            client.chat_json.assert_not_called()
        for scope in (None, "", "grasp", "support", "OWNER"):
            client = Mock()
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                select_grounding_candidates(source, "Choose a region", views,
                                             selection_scope=scope, client=client)
            client.chat_json.assert_not_called()

    def test_single_candidate_still_requires_semantic_verification(self):
        source, views = fixture()
        views["left"]["candidates"] = views["left"]["candidates"][:1]
        for ids, status in (([7], "selected"), ([], "no_match"), (None, "unresolved")):
            with self.subTest(ids=ids):
                client = Mock(chat_json=Mock(return_value=response(ids)))
                result = select_grounding_candidates(source, "Use the requested part", views, selection_scope="interaction_region", client=client)
                client.chat_json.assert_called_once()
                item = result["views"]["left"]
                self.assertEqual(item["status"], status)
                self.assertEqual(item["candidate_ids"], ids)
                self.assertEqual(len(item["regions"]), 1 if ids else 0)

    def test_alternatives_keep_separate_existing_masks_and_response_order(self):
        source, views = fixture()
        client = Mock(chat_json=Mock(return_value=response(
            [31, 7], reason="Both visible regions satisfy the instruction; either is acceptable.")))
        result = select_grounding_candidates(source, "Either suitable surface is acceptable", views, selection_scope="interaction_region", client=client)
        item = result["views"]["left"]
        self.assertEqual(item["status"], "alternatives")
        self.assertEqual(item["candidate_ids"], [31, 7])
        self.assertEqual([region["candidate_id"] for region in item["regions"]], [31, 7])
        by_id = {entry["candidate_id"]: entry for entry in views["left"]["candidates"]}
        for region in item["regions"]:
            np.testing.assert_array_equal(region["mask"], by_id[region["candidate_id"]]["mask"])
        self.assertFalse(np.array_equal(item["regions"][0]["mask"], item["regions"][1]["mask"]))

    def test_ids_are_local_to_each_view_without_cross_view_identity_assumptions(self):
        source, views = fixture()
        views["right"] = {"image": Image.new("RGB", (12, 10), "green"),
                          "candidates": [candidate((6, 1, 11, 5), number=7)]}
        selected = {"views": {name: {"candidate_ids": [7], "reason": "Valid in this view."}
                              for name in views}}
        result = select_grounding_candidates(source, "Use the relevant visible part", views, selection_scope="interaction_region",
                                            client=Mock(chat_json=Mock(return_value=selected)))
        for name in views:
            np.testing.assert_array_equal(result["views"][name]["regions"][0]["mask"],
                                          views[name]["candidates"][0]["mask"])
        self.assertFalse(np.array_equal(result["views"]["left"]["regions"][0]["mask"],
                                        result["views"]["right"]["regions"][0]["mask"]))
        selected["views"]["right"]["candidate_ids"] = [31]  # Exists only in the left view.
        with self.assertRaises(ValueError):
            select_grounding_candidates(source, "Use the relevant part", views, selection_scope="interaction_region",
                                       client=Mock(chat_json=Mock(return_value=selected)))

    def test_selection_uses_returned_id_instead_of_detector_score_without_mutation(self):
        source, views = fixture()
        before_source, before_views = deepcopy(source), deepcopy(views)
        selected = select_grounding_candidates(source, "Select the right region", views, selection_scope="interaction_region",
                                             client=Mock(chat_json=Mock(return_value=response([31]))))
        region = selected["views"]["left"]["regions"][0]
        self.assertEqual(region["candidate_id"], 31)
        self.assertEqual(region["score"], .25)
        np.testing.assert_array_equal(region["mask"], before_views["left"]["candidates"][1]["mask"])
        self.assertEqual(set(source), set(before_source))
        self.assertEqual(source["image"].tobytes(), before_source["image"].tobytes())
        self.assertEqual(source["query_descriptions"], before_source["query_descriptions"])
        for name in ("mask", "owner_mask"):
            np.testing.assert_array_equal(source[name], before_source[name])
        self.assertEqual(views["left"]["image"].tobytes(), before_views["left"]["image"].tobytes())
        for original, current in zip(before_views["left"]["candidates"], views["left"]["candidates"]):
            self.assertEqual(set(original), set(current))
            self.assertEqual(original["bbox"], current["bbox"])
            np.testing.assert_array_equal(original["mask"], current["mask"])

    def test_candidate_display_includes_mask_pixels_outside_the_detector_box(self):
        source, views = fixture()
        observed = candidate((1, 1, 3, 3), number=7)
        # A detector's box may overlap its valid mask without containing every
        # component. Keep the remote component beyond the display crop's normal
        # context margin so a box-only crop really would hide it.
        observed["mask"] = np.zeros((48, 64), dtype=bool)
        observed["mask"][1:3, 1:3] = True
        observed["mask"][40:44, 55:60] = True
        views["left"]["image"] = Image.new("RGB", (64, 48), (130, 80, 30))
        views["left"]["candidates"] = [observed]
        client = Mock(chat_json=Mock(return_value=response([7])))
        real_panels = selection._panels
        candidate_crops = []

        def inspect_panels(image, candidates):
            # Scope the observation to actual candidate presentation: the
            # source-detail crop has a separate purpose and different bounds.
            with patch.object(selection, "_context_crop", wraps=selection._context_crop) as crop:
                rendered = real_panels(image, candidates)
            candidate_crops.extend(call.args[1] for call in crop.call_args_list)
            return rendered

        with patch.object(selection, "_panels", side_effect=inspect_panels):
            result = select_grounding_candidates(source, "Inspect the complete candidate", views,
                                                 selection_scope="interaction_region", client=client)
        client.chat_json.assert_called_once()
        self.assertEqual(len(candidate_crops), 1)
        left, top, right, bottom = candidate_crops[0]
        rows, columns = np.nonzero(observed["mask"])
        # Check the observable coverage property, not equality to the private
        # bounding-box implementation or any particular crop padding policy.
        self.assertTrue(np.all((columns >= left) & (columns < right)))
        self.assertTrue(np.all((rows >= top) & (rows < bottom)))
        np.testing.assert_array_equal(result["views"]["left"]["regions"][0]["mask"], observed["mask"])

    def test_explicit_noncontiguous_ids_are_preserved_and_missing_ids_use_input_index(self):
        source, views = fixture()
        views["left"]["candidates"] = [candidate((1, 1, 3, 3)),
                                        candidate((4, 1, 6, 3), number=8),
                                        candidate((7, 5, 10, 9))]
        result = select_grounding_candidates(source, "Any of these may satisfy the task", views, selection_scope="interaction_region",
                                            client=Mock(chat_json=Mock(return_value=response([3, 1, 8]))))
        self.assertEqual([item["candidate_id"] for item in result["views"]["left"]["regions"]], [3, 1, 8])
        self.assertNotIn("candidate_id", views["left"]["candidates"][0])
        self.assertNotIn("candidate_id", views["left"]["candidates"][2])

    def test_empty_view_remains_visual_context_in_a_joint_request(self):
        source, views = fixture()
        empty_image = Image.new("RGB", (12, 10), (3, 217, 91))
        views["empty"] = {"image": empty_image, "candidates": []}
        answer = response([7])
        answer["views"]["empty"] = {"candidate_ids": [], "reason": "No observed candidates in this view."}
        client = Mock(chat_json=Mock(return_value=answer))
        result = select_grounding_candidates(source, "Select the actual interaction part", views, selection_scope="interaction_region", client=client)
        client.chat_json.assert_called_once()
        empty = result["views"]["empty"]
        self.assertEqual(empty["status"], "no_candidates")
        self.assertEqual(empty["candidate_ids"], [])
        self.assertEqual(empty["regions"], [])
        images = client.chat_json.call_args.kwargs["images"]
        self.assertTrue(any(image.size == empty_image.size and image.tobytes() == empty_image.tobytes()
                            for image in images))
        _, data = context_from(client)
        self.assertIn("empty", json.dumps(data["image_order"]))

    def test_all_empty_views_return_no_candidates_without_a_vlm_call(self):
        source, views = fixture()
        views["left"]["candidates"] = []
        views["right"] = {"image": Image.new("RGB", (12, 10), "blue"), "candidates": []}
        client = Mock()
        result = select_grounding_candidates(source, "Find a usable interaction region", views, selection_scope="interaction_region", client=client)
        self.assertFalse(result["vlm_used"])
        client.chat_json.assert_not_called()
        self.assertEqual(set(result["views"]), set(views))
        for item in result["views"].values():
            self.assertEqual(item["status"], "no_candidates")
            self.assertEqual(item["candidate_ids"], [])
            self.assertEqual(item["regions"], [])
            self.assertIsInstance(item["reason"], str)
            self.assertTrue(item["reason"].strip())

    def test_response_requires_exact_top_level_and_view_keys(self):
        source, views = fixture()
        answer = response([7])
        invalid = [None, [], {}, {**answer, "pose": np.eye(4).tolist()}, {"views": []},
                   {"views": {}}, {"views": {"unobserved_view": answer["views"]["left"]}},
                   {"views": {**answer["views"], "extra": answer["views"]["left"]}}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_grounding_candidates(source, "Choose a region", views, selection_scope="interaction_region",
                                           client=Mock(chat_json=Mock(return_value=value)))

    def test_view_response_rejects_missing_reason_geometry_and_malformed_fields(self):
        source, views = fixture()
        valid = response([7])["views"]["left"]
        invalid = [None, {}, {"candidate_ids": [7]}, {"reason": "No IDs."},
                   {**valid, "mask": [[True]]}, {**valid, "pose": np.eye(4).tolist()},
                   {**valid, "score": .99}, {**valid, "reason": " "},
                   {**valid, "reason": 4}, {**valid, "candidate_ids": "7"},
                   {**valid, "candidate_ids": 7}, {**valid, "candidate_ids": (7,)}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_grounding_candidates(source, "Choose a region", views, selection_scope="interaction_region",
                                           client=Mock(chat_json=Mock(return_value={"views": {"left": value}})))

    def test_selected_ids_reject_unknown_duplicates_and_numeric_coercion(self):
        source, views = fixture()
        for ids in ([True], [7.0], ["7"], [-1], [0], [999], [7, 7], [31, 7, 31]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                select_grounding_candidates(source, "Choose a region", views, selection_scope="interaction_region",
                                           client=Mock(chat_json=Mock(return_value=response(ids))))

    def test_an_empty_view_cannot_invent_ids_or_claim_visual_uncertainty(self):
        source, views = fixture()
        views["empty"] = {"image": Image.new("RGB", (12, 10), "black"), "candidates": []}
        for ids in (None, [7], [999]):
            answer = response([7])
            answer["views"]["empty"] = {"candidate_ids": ids, "reason": "Model output for an empty view."}
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                select_grounding_candidates(source, "Choose a region", views, selection_scope="interaction_region",
                                           client=Mock(chat_json=Mock(return_value=answer)))

    def test_input_ids_must_be_positive_unique_integers_before_inference(self):
        for value in (0, -1, True, 1.0, "7", 31):
            source, views = fixture()
            views["left"]["candidates"][0]["candidate_id"] = value
            client = Mock()
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_grounding_candidates(source, "Choose a region", views, selection_scope="interaction_region", client=client)
            client.chat_json.assert_not_called()
        source, views = fixture()
        del views["left"]["candidates"][0]["candidate_id"]  # Its required implicit ID is 1.
        views["left"]["candidates"][1]["candidate_id"] = 1
        client = Mock()
        with self.assertRaises(ValueError):
            select_grounding_candidates(source, "Choose a region", views, selection_scope="interaction_region", client=client)
        client.chat_json.assert_not_called()

    def test_invalid_candidate_geometry_fails_before_semantic_inference(self):
        changes = [("mask", np.zeros((10, 12), bool)), ("mask", np.ones((2, 2), bool)),
                   ("mask", np.full((10, 12), .5)), ("score", np.nan),
                   ("bbox", [-1, 0, 4, 4])]
        for key, value in changes:
            source, views = fixture()
            views["left"]["candidates"][0][key] = value
            client = Mock()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                select_grounding_candidates(source, "Choose a region", views, selection_scope="interaction_region", client=client)
            client.chat_json.assert_not_called()

    def test_invalid_source_instruction_or_view_collection_fails_before_inference(self):
        cases = ("empty instruction", "empty description", "empty source mask", "wrong source image",
                 "blank query description", "nonlist query descriptions", "empty views")
        for case in cases:
            source, views = fixture()
            instruction = "Choose the requested region"
            if case == "empty instruction":
                instruction = " "
            elif case == "empty description":
                source["description"] = " "
            elif case == "empty source mask":
                source["mask"][:] = False
            elif case == "wrong source image":
                source["image"] = np.zeros((10, 12, 3), dtype=np.uint8)
            elif case == "blank query description":
                source["query_descriptions"] = [""]
            elif case == "nonlist query descriptions":
                source["query_descriptions"] = "contact surface"
            else:
                views = {}
            client = Mock()
            with self.subTest(case=case), self.assertRaises(ValueError):
                select_grounding_candidates(source, instruction, views, selection_scope="interaction_region", client=client)
            client.chat_json.assert_not_called()

    def test_transport_failure_propagates_without_retry_or_automatic_selection(self):
        source, views = fixture()
        views["left"]["candidates"] = views["left"]["candidates"][:1]
        client = Mock(chat_json=Mock(side_effect=RuntimeError("injected gateway failure")))
        with self.assertRaisesRegex(RuntimeError, "injected gateway failure"):
            select_grounding_candidates(source, "Verify the only candidate", views, selection_scope="interaction_region", client=client)
        client.chat_json.assert_called_once()


if __name__ == "__main__":
    unittest.main()
# END independent owner/interaction candidate selection contracts.
