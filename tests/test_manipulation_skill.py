"""Offline-artifact routing contracts, independent of models and simulator GT."""

# BEGIN actual file loading and object/component separation regressions.
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from manipulation.skill import load_grounding_skill, observe_interaction_reference


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def bundle(root, *, scope="object", geometry_kind="query_frame"):
    """Small portable artifact with distinct owner and recorded contact masks."""
    manifest = {
        "schema": "functo.grounding_skill.1", "owner_role": "carrier",
        "geometry_kind": geometry_kind,
        "files": {"semantic": "semantic.json", "template": "template.json",
                  "program": "program.json", "observations": "observations.npz", "image": "rgb.png"},
        "provenance": {"source_episode": "/unavailable/offline/episode.h5"},
    }
    semantic = {
        "object_description": "container", "interaction_part_description": "outer grasping region",
        "interaction_reference": {"text": "container" if scope == "object" else "container handle",
                                  "scope": scope},
        "description": "Establish the demonstrated grasp of the container.",
        "reason": "The chosen observation makes the demonstrated region distinguishable.",
    }
    template = {"owner_id": "source_owner", "region_id": "recorded_contact"}
    if geometry_kind == "query_frame":
        template.update(query_pixels=[[2, 2], [3, 3], [4, 2]],
                        query_descriptions=["contact anchor", "axis witness", "plane witness"],
                        frame_recipe={"kind": "frame", "origin_weights": [1, 0, 0],
                                      "axis_weights": [-1, 1, 0], "plane_weights": [-1, 0, 1]})
    if geometry_kind == "protruding_line":
        manifest["files"]["geometry"] = "geometry.json"
        write_json(root / "geometry.json", {"anchor_pixel": [3, 3], "source_frame": np.eye(4).tolist()})
    contact = np.zeros((10, 12), bool)
    contact[3:5, 3:5] = True
    owner = np.zeros_like(contact)
    owner[1:9, 1:11] = True
    arrays = {"mask": contact, "owner_mask": owner, "depth": np.full((10, 12), 1., np.float32),
              "K": np.array([[10., 0., 6.], [0., 10., 5.], [0., 0., 1.]]),
              "T_camera_owner": np.eye(4)}
    if geometry_kind == "protruding_line":
        arrays["front_mask"] = contact.copy()
    np.savez(root / "observations.npz", **arrays)
    Image.new("RGB", (12, 10), (25, 50, 100)).save(root / "rgb.png")
    program = {"skill_kind": "grasp", "demo": {
        "carrier_id": "source_owner", "T_carrier_eef": [np.eye(4).tolist()],
        "source": {"trajectory_file": "/unavailable/offline/trajectories.h5"}}}
    for filename, value in (("skill.json", manifest), ("semantic.json", semantic),
                            ("template.json", template), ("program.json", program)):
        write_json(root / filename, value)
    return semantic, arrays, program


def detection(label):
    mask = np.zeros((10, 12), bool)
    mask[2:8, 2:10] = True
    return {"label": label, "candidate_id": 17, "score": .6, "bbox": [2, 2, 10, 8], "mask": mask}


class GroundingSkillTest(unittest.TestCase):
    def test_offline_semantics_drive_same_runtime_entry_for_object_and_component(self):
        for scope, kind in (("object", "query_frame"), ("object", "vertices"),
                            ("part", "protruding_line")):
            with self.subTest(scope=scope, kind=kind), tempfile.TemporaryDirectory() as directory:
                semantic, arrays, program = bundle(Path(directory), scope=scope, geometry_kind=kind)
                skill = load_grounding_skill(directory)
                # Loading is complete despite intentionally nonexistent provenance paths.
                self.assertEqual(skill["program"], program)
                self.assertEqual(skill["semantic"], semantic)
                self.assertEqual(skill["source_template"]["description"], semantic["description"])
                for key, array in arrays.items():
                    np.testing.assert_array_equal(skill["source_template"][key], array)
                if kind != "query_frame":
                    self.assertNotIn("query_descriptions", skill["source_template"])
                row = detection(semantic["interaction_reference"]["text"])
                detector = Mock(return_value=[row])
                client = Mock(chat_json=Mock(return_value={"views": {"camera": {
                    "candidate_ids": [17], "reason": "This visible instance matches the current request."}}}))
                current = Image.new("RGB", (12, 10), (80, 120, 40))
                result = observe_interaction_reference(
                    skill, "Use the right instance", {"camera": {"image": current}},
                    detector=detector, client=client)
                self.assertEqual(detector.call_args.args[1], [semantic["interaction_reference"]["text"]])
                expected_scope = "owner" if scope == "object" else "interaction_reference"
                self.assertEqual(result["selection_scope"], expected_scope)
                context = json.loads(client.chat_json.call_args.args[0].split("\nInput data:\n", 1)[1])
                self.assertEqual(context["current_instruction"], "Use the right instance")
                self.assertEqual(context["source_evidence"]["interaction_reference"], semantic["interaction_reference"])
                self.assertEqual(context["source_evidence"]["description"], semantic["description"])
                self.assertNotIn("source_owner", client.chat_json.call_args.args[0])
                selected = result["selection"]["views"]["camera"]["regions"][0]
                np.testing.assert_array_equal(selected["mask"], row["mask"])
                # A selected component remains a component; neither the caller's
                # source masks nor the returned candidate gain an owner mask.
                self.assertNotIn("owner_mask", selected)
                self.assertNotIn("owner_mask", result)

    def test_part_reference_needs_no_current_whole_owner_and_does_not_force_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle(Path(directory), scope="part", geometry_kind="protruding_line")
            skill = load_grounding_skill(directory)
            views = {"occluded": {"image": Image.new("RGB", (12, 10))}}
            client = Mock()
            result = observe_interaction_reference(skill, "Use this instance", views,
                                                   detector=Mock(return_value=[]), client=client)
            self.assertEqual(result["selection"]["views"]["occluded"]["status"], "no_candidates")
            client.chat_json.assert_not_called()
            client.chat_json.return_value = {"views": {"occluded": {
                "candidate_ids": None, "reason": "The owning instance is obscured."}}}
            result = observe_interaction_reference(
                skill, "Use this instance", views,
                detector=Mock(return_value=[detection("container handle")]), client=client)
            self.assertEqual(result["selection"]["views"]["occluded"]["status"], "unresolved")
            self.assertEqual(result["selection"]["views"]["occluded"]["regions"], [])

    def test_wrong_query_result_is_rejected_before_semantic_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle(Path(directory))
            skill = load_grounding_skill(directory)
            client = Mock()
            with self.assertRaisesRegex(ValueError, "unrequested label"):
                observe_interaction_reference(
                    skill, "Use this instance", {"camera": {"image": Image.new("RGB", (12, 10))}},
                    detector=Mock(return_value=[detection("caller override")]), client=client)
            client.chat_json.assert_not_called()

    def test_component_selection_accepts_separate_current_owner_context(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle(Path(directory), scope="part", geometry_kind="protruding_line")
            skill = load_grounding_skill(directory)
            owner_mask = np.zeros((10, 12), bool)
            owner_mask[1:9, 1:11] = True
            selected_owner = {"owner_id": "current_instance", "mask": owner_mask,
                              "description": "Observed selected container", "source": "Current image detection"}
            client = Mock(chat_json=Mock(return_value={"views": {"camera": {
                "candidate_ids": [17], "reason": "This component belongs to the selected green owner."}}}))
            row = detection("container handle")
            result = observe_interaction_reference(
                skill, "Use this container", {"camera": {
                    "image": Image.new("RGB", (12, 10)), "selected_owner": selected_owner}},
                detector=Mock(return_value=[row]), client=client)
            context = json.loads(client.chat_json.call_args.args[0].split("\nInput data:\n", 1)[1])
            self.assertEqual(context["views"]["camera"]["selected_owner"]["owner_id"], "current_instance")
            self.assertIn("mask overlap", client.chat_json.call_args.args[0])
            np.testing.assert_array_equal(result["selection"]["views"]["camera"]["regions"][0]["mask"], row["mask"])
            np.testing.assert_array_equal(selected_owner["mask"], owner_mask)

    def test_geometry_metadata_cannot_silently_override_offline_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle(root)
            template = json.loads((root / "template.json").read_text())
            template["description"] = "An obsolete handwritten semantic request."
            write_json(root / "template.json", template)
            with self.assertRaisesRegex(ValueError, "duplicates semantics"):
                load_grounding_skill(root)

    def test_payload_cannot_turn_object_reference_into_another_query_or_asset_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for text in ("another object", "container_1", "container\nnew instruction"):
                semantic, _, _ = bundle(root)
                semantic["interaction_reference"]["text"] = text
                write_json(root / "semantic.json", semantic)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_grounding_skill(root)
            for text in ("container_handle", "container\nhandle"):
                semantic, _, _ = bundle(root, scope="part", geometry_kind="protruding_line")
                semantic["interaction_reference"]["text"] = text
                write_json(root / "semantic.json", semantic)
                with self.subTest(text=text), self.assertRaisesRegex(ValueError, "natural noun phrase"):
                    load_grounding_skill(root)

    def test_protruding_attachment_cannot_hide_missing_or_unrelated_source_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for failure in ("missing_front", "nonbinary_front", "unrelated_front", "bad_anchor", "bad_frame"):
                _, arrays, _ = bundle(root, scope="part", geometry_kind="protruding_line")
                geometry = json.loads((root / "geometry.json").read_text())
                if failure == "missing_front":
                    del arrays["front_mask"]
                elif failure == "nonbinary_front":
                    arrays["front_mask"] = arrays["front_mask"].astype(np.uint8)
                elif failure == "unrelated_front":
                    arrays["front_mask"][0, 0] = True
                elif failure == "bad_anchor":
                    geometry["anchor_pixel"] = [100, 100]
                else:
                    geometry["source_frame"] = np.zeros((4, 4)).tolist()
                np.savez(root / "observations.npz", **arrays)
                write_json(root / "geometry.json", geometry)
                with self.subTest(failure=failure), self.assertRaises(ValueError):
                    load_grounding_skill(root)

    def test_vertices_cannot_reintroduce_discarded_three_point_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle(root, geometry_kind="vertices")
            template = json.loads((root / "template.json").read_text())
            template["query_descriptions"] = ["obsolete fixed point"]
            write_json(root / "template.json", template)
            with self.assertRaisesRegex(ValueError, "obsolete query-frame"):
                load_grounding_skill(root)

    def test_bundle_cannot_read_external_payloads_and_cannot_guess_missing_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle(root)
            manifest = json.loads((root / "skill.json").read_text())
            manifest["files"]["program"] = "../outside.json"
            write_json(root / "skill.json", manifest)
            with self.assertRaisesRegex(ValueError, "missing or escapes"):
                load_grounding_skill(root)
            bundle(root)
            semantic = json.loads((root / "semantic.json").read_text())
            del semantic["interaction_reference"]
            write_json(root / "semantic.json", semantic)
            with self.assertRaisesRegex(ValueError, "complete offline"):
                load_grounding_skill(root)

    def test_in_memory_template_cannot_bypass_the_offline_semantic_record(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle(Path(directory))
            skill = load_grounding_skill(directory)
            changed = deepcopy(skill)
            changed["source_template"]["description"] = "Use another interaction"
            detector = Mock()
            with self.assertRaisesRegex(ValueError, "differs from its offline artifact"):
                observe_interaction_reference(
                    changed, "Use this instance", {"camera": {"image": Image.new("RGB", (12, 10))}},
                    detector=detector, client=Mock())
            detector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
# END portable dataflow and distinct object/component observation contracts.
