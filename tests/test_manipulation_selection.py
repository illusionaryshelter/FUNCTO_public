"""CPU contracts for view-local owner, evidence and interaction selection.

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
    # BEGIN selected-owner context must enter the real semantic request.
    def test_selected_current_owner_is_visible_and_described_without_changing_candidates(self):
        source, views = fixture()
        owner = np.zeros((10, 12), dtype=bool)
        owner[2:8, 6:11] = True
        views["left"]["selected_owner"] = {
            "owner_id": "observed_object", "mask": owner,
            "description": "The currently selected rigid object.",
            "source": "Independent visible owner selection on this image.",
        }
        original = deepcopy(views)
        client = Mock(chat_json=Mock(return_value=response(
            [31], reason="This part belongs to the green selected object.")))
        result = select_grounding_candidates(source, "Use this object's demonstrated region", views,
                                             selection_scope="interaction_region", client=client)
        prefix, context = context_from(client)
        self.assertIn("mask overlap", prefix)
        self.assertEqual(context["views"]["left"]["allowed_ids"], [7, 31])
        self.assertEqual(context["views"]["left"]["selected_owner"], {
            key: views["left"]["selected_owner"][key]
            for key in ("owner_id", "description", "source")})
        images = client.chat_json.call_args.kwargs["images"]
        current_owner_image = next(entry["image"] - 1 for entry in context["image_order"]
                                   if entry["content"] == "view left: selected current owner highlighted GREEN")
        panel = np.asarray(images[current_owner_image], dtype=int)
        raw = np.asarray(views["left"]["image"], dtype=int)
        self.assertTrue(np.all(panel[..., 1][owner] > raw[..., 1][owner]))
        np.testing.assert_array_equal(panel[~owner], raw[~owner])
        np.testing.assert_array_equal(result["views"]["left"]["regions"][0]["mask"],
                                      original["left"]["candidates"][1]["mask"])
        np.testing.assert_array_equal(views["left"]["selected_owner"]["mask"], owner)
        for before, after in zip(original["left"]["candidates"], views["left"]["candidates"]):
            np.testing.assert_array_equal(before["mask"], after["mask"])

    def test_owner_overlap_does_not_resolve_semantic_membership(self):
        source, views = fixture()
        # Both candidates overlap this mask. Neither overlap nor a single
        # detector label may turn the model's abstention into a selected part.
        views["left"]["selected_owner"] = {
            "owner_id": "current_owner", "mask": np.ones((10, 12), dtype=bool),
            "description": "Visible evidence of the already selected owner.",
            "source": "Current owner perception.",
        }
        for ids, status in ((None, "unresolved"), ([], "no_match")):
            client = Mock(chat_json=Mock(return_value=response(ids)))
            selected = select_grounding_candidates(source, "Use a part of the selected owner", views,
                                                   selection_scope="interaction_region", client=client)
            client.chat_json.assert_called_once()
            self.assertEqual(selected["views"]["left"]["status"], status)
            self.assertEqual(selected["views"]["left"]["regions"], [])

    def test_malformed_selected_owner_or_wrong_scope_fails_before_model_call(self):
        source, views = fixture()
        valid = {"owner_id": "current_owner", "mask": np.ones((10, 12), bool),
                 "description": "Selected visible owner.", "source": "Current image perception."}
        invalid = [None, {key: value for key, value in valid.items() if key != "source"},
                   {**valid, "body_pose": np.eye(4)}, {**valid, "description": " "},
                   {**valid, "owner_id": 123}, {**valid, "mask": np.ones((3, 3), bool)},
                   {**valid, "mask": np.zeros((10, 12), bool)}]
        for record in invalid:
            views["left"]["selected_owner"] = record
            client = Mock()
            with self.subTest(record=record), self.assertRaises(ValueError):
                select_grounding_candidates(source, "Choose this owner's part", views,
                                             selection_scope="interaction_region", client=client)
            client.chat_json.assert_not_called()
        views["left"]["selected_owner"] = valid
        for scope in ("owner", "owner_evidence"):
            client = Mock()
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                select_grounding_candidates(source, "Choose visible evidence", views,
                                             selection_scope=scope, client=client)
            client.chat_json.assert_not_called()
    # END selected-owner visual association is independent of mask overlap.

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


# BEGIN same-owner visible evidence without a manufactured whole-object mask.
class GroundingOwnerEvidenceSelectionTest(unittest.TestCase):
    @staticmethod
    def answer(ids, same_owner, reason="The visible panels belong to the same requested reference body."):
        result = response(ids, reason=reason)
        result["views"]["left"]["same_owner"] = same_owner
        return result

    def test_partial_source_evidence_is_sufficient_and_has_its_own_annotation(self):
        source, views = fixture()
        del source["owner_mask"]
        source["description"] = "A visible surface of the demonstrated reference body."
        before = source["image"].tobytes()
        client = Mock(chat_json=Mock(return_value=self.answer([7], True)))
        result = select_grounding_candidates(source, "Locate the reference body's visible geometry", views,
                                             selection_scope="owner_evidence", client=client)
        item = result["views"]["left"]
        self.assertEqual(result["selection_scope"], "owner_evidence")
        self.assertEqual(item["status"], "selected")
        self.assertIs(item["same_owner"], True)
        self.assertNotIn("owner_mask", source)
        self.assertNotIn("owner_mask", nested_keys(result))
        self.assertEqual(source["image"].tobytes(), before)
        # The partial source mask is visible evidence, not a red contact label.
        image = client.chat_json.call_args.kwargs["images"][0]
        original = np.asarray(source["image"], dtype=int)
        annotated = np.asarray(image, dtype=int)
        self.assertTrue(np.all(annotated[..., 1][source["mask"]] > original[..., 1][source["mask"]]))
        np.testing.assert_array_equal(annotated[~source["mask"]], original[~source["mask"]])
        _, data = context_from(client)
        self.assertIn("green visible owner evidence", data["image_order"][0]["content"])

    def test_multiple_same_owner_regions_are_evidence_not_target_alternatives(self):
        source, views = fixture()
        # Overlap does not create another instance or require mask union. The
        # caller receives exactly the independent semantic detector outputs.
        views["left"]["candidates"][1] = candidate((3, 3, 8, 8), number=31)
        before = deepcopy(views)
        result = select_grounding_candidates(source, "Use visible surfaces of the selected reference", views,
                                             selection_scope="owner_evidence",
                                             client=Mock(chat_json=Mock(return_value=self.answer([31, 7], True))))
        item = result["views"]["left"]
        self.assertEqual(item["status"], "selected")
        self.assertIs(item["same_owner"], True)
        self.assertEqual(item["candidate_ids"], [31, 7])
        self.assertEqual([region["candidate_id"] for region in item["regions"]], [31, 7])
        originals = {region["candidate_id"]: region for region in before["left"]["candidates"]}
        for region in item["regions"]:
            np.testing.assert_array_equal(region["mask"], originals[region["candidate_id"]]["mask"])
        for current, original in zip(views["left"]["candidates"], before["left"]["candidates"]):
            np.testing.assert_array_equal(current["mask"], original["mask"])

    def test_scope_uses_one_automatic_prompt_and_allowlists_source_evidence(self):
        source, views = fixture()
        source["body_id"] = "DO_NOT_SEND_SIMULATOR_BODY_ID"
        source["T_world_body"] = np.eye(4)
        prefixes = []
        for instruction in ("Locate the reference for a rotational motion.",
                            "Find evidence of the support for placing the held object."):
            client = Mock(chat_json=Mock(return_value=self.answer([7], True)))
            select_grounding_candidates(source, instruction, views,
                                        selection_scope="owner_evidence", client=client)
            prefix, data = context_from(client)
            prefixes.append(prefix)
            self.assertEqual(data["current_instruction"], instruction)
            self.assertEqual(data["selection_scope"], "owner_evidence")
            self.assertEqual(data["source_evidence"]["description"], source["description"])
            self.assertNotIn("body_id", nested_keys(data))
            self.assertNotIn("T_world_body", nested_keys(data))
            self.assertNotIn("DO_NOT_SEND_SIMULATOR_BODY_ID", client.chat_json.call_args.args[0])
        self.assertEqual(prefixes, [selection._SELECTION_INSTRUCTION] * 2)
        self.assertIn("same_owner=true", prefixes[0])
        self.assertIn("not evidence of one rigid owner", prefixes[0])
        self.assertIn("NOT interchangeable target alternatives", prefixes[0])

    def test_selected_evidence_requires_explicit_boolean_same_owner(self):
        source, views = fixture()
        for ids in ([7], [7, 31]):
            for assertion in (None, False, 1, "true"):
                with self.subTest(ids=ids, assertion=assertion), self.assertRaises(ValueError):
                    select_grounding_candidates(source, "Use the requested reference", views,
                                                 selection_scope="owner_evidence",
                                                 client=Mock(chat_json=Mock(return_value=self.answer(ids, assertion))))
        # A valid legacy decision must not silently become an evidence claim.
        with self.assertRaises(ValueError):
            select_grounding_candidates(source, "Use the requested reference", views,
                                         selection_scope="owner_evidence",
                                         client=Mock(chat_json=Mock(return_value=response([7]))))

    def test_unresolved_instances_cannot_be_promoted_to_a_same_owner_set(self):
        source, views = fixture()
        for ids, status in ((None, "unresolved"), ([], "no_match")):
            client = Mock(chat_json=Mock(return_value=self.answer(ids, None,
                reason="The supplied views do not establish the requested target's identity.")))
            result = select_grounding_candidates(source, "Locate the requested reference", views,
                                                 selection_scope="owner_evidence", client=client)
            item = result["views"]["left"]
            self.assertEqual(item["status"], status)
            self.assertIsNone(item["same_owner"])
            self.assertEqual(item["regions"], [])
            for assertion in (True, False, "unknown", 0):
                with self.subTest(ids=ids, assertion=assertion), self.assertRaises(ValueError):
                    select_grounding_candidates(source, "Locate the requested reference", views,
                                                 selection_scope="owner_evidence",
                                                 client=Mock(chat_json=Mock(return_value=self.answer(ids, assertion))))

    def test_no_candidates_need_no_inference_or_membership_assertion(self):
        source, views = fixture()
        views["left"]["candidates"] = []
        client = Mock()
        result = select_grounding_candidates(source, "Locate visible reference evidence", views,
                                             selection_scope="owner_evidence", client=client)
        client.chat_json.assert_not_called()
        self.assertFalse(result["vlm_used"])
        item = result["views"]["left"]
        self.assertEqual(item["status"], "no_candidates")
        self.assertEqual(item["candidate_ids"], [])
        self.assertEqual(item["regions"], [])
        self.assertIsNone(item["same_owner"])

    def test_evidence_ids_stay_view_local_and_do_not_allow_geometry_fields(self):
        source, views = fixture()
        views["right"] = {"image": Image.new("RGB", (12, 10), "green"),
                          "candidates": [candidate((6, 1, 11, 5), number=7)]}
        decision = self.answer([7], True)
        decision["views"]["right"] = {"candidate_ids": [7], "same_owner": True,
                                        "reason": "The local mask belongs to the selected reference."}
        result = select_grounding_candidates(source, "Locate the selected reference", views,
                                             selection_scope="owner_evidence",
                                             client=Mock(chat_json=Mock(return_value=decision)))
        for name in views:
            np.testing.assert_array_equal(result["views"][name]["regions"][0]["mask"],
                                          views[name]["candidates"][0]["mask"])
        # Explicit semantic membership cannot make another view's ID available
        # or attach model-invented geometry to a detection.
        for change in ({"candidate_ids": [31]}, {"pose": np.eye(4).tolist()}):
            bad = deepcopy(decision)
            bad["views"]["right"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                select_grounding_candidates(source, "Locate the selected reference", views,
                                             selection_scope="owner_evidence",
                                             client=Mock(chat_json=Mock(return_value=bad)))

    def test_existing_scopes_keep_alternatives_and_reject_the_new_response_field(self):
        source, views = fixture()
        for scope in ("owner", "interaction_region"):
            with self.subTest(scope=scope):
                result = select_grounding_candidates(source, "Either target is acceptable", views,
                                                     selection_scope=scope,
                                                     client=Mock(chat_json=Mock(return_value=response([7, 31]))))
                self.assertEqual(result["views"]["left"]["status"], "alternatives")
                self.assertNotIn("same_owner", result["views"]["left"])
                with self.assertRaises(ValueError):
                    select_grounding_candidates(source, "Choose an allowed target", views,
                                                 selection_scope=scope,
                                                 client=Mock(chat_json=Mock(return_value=self.answer([7], True))))
# END explicit same-owner evidence and unchanged existing-scope contracts.


if __name__ == "__main__":
    unittest.main()
# END independent owner/interaction candidate selection contracts.
