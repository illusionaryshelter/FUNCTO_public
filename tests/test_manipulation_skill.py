"""Version-2 portable bundles and semantic instance observation."""
# BEGIN production/consumption replaces the obsolete query-recipe bundle.
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
from PIL import Image
from manipulation.representation import build_representation
from manipulation.skill import save_grounding_skill, load_grounding_skill, observe_interaction_reference


def source_inputs(scope="object", contact=True):
    owner = np.zeros((10, 12), bool)
    owner[1:9, 1:11] = True
    region = np.zeros_like(owner)
    if contact:
        region[3:5, 3:5] = True
    y, x = np.nonzero(owner)
    pixels = np.column_stack((x, y))
    K = np.array([[10., 0., 6.], [0., 10., 5.], [0., 0., 1.]])
    xyz = np.column_stack((pixels, np.ones(len(x)))) @ np.linalg.inv(K).T
    bank = dict(xyz=xyz, pixels=pixels, object_mask=owner, entity_member=np.ones(len(x), bool),
                region_mask=region[y, x], evidence_only=np.zeros(len(x), bool), K=K, T_world_camera_cv=np.eye(4))
    source = dict(image=Image.new("RGB", (12, 10), (25, 50, 100)), observation=bank,
                  depth_m=np.ones((10, 12), np.float32), crop_xyxy=np.array([0, 0, 12, 10]),
                  timestamp_s=4., frame_index=17, owner_id="source_body", T_world_body_at_image=np.eye(4))
    if contact:
        source.update(region_image_mask=region, L_body_region=np.eye(4))
    semantic = dict(object_description="container", interaction_part_description="outer grasp region" if contact else "",
        interaction_reference=dict(text="container" if scope == "object" else "container handle", scope=scope),
        description="Use this demonstrated object relation.", reason="Recorded source visual evidence.")
    metadata = dict(skill_id="recorded_segment", owner_role="carrier", semantic=semantic,
        roles={"master": {"entity": "container", "body": "source_body"}, "slave": "gripper"},
        provenance={"recording": "/unavailable/original/recording.h5"}, task={})
    body = np.repeat(np.eye(4)[None], 3, axis=0)
    eef = body.copy()
    eef[:, 2, 3] = [.2, .1, .0]
    rep = build_representation(kind="grasp", frame_indices=[17, 20, 25], timestamps=[4., 4.5, 5.],
        T_world_carrier=body, T_world_eef=eef, L_G=np.eye(4), approach_time=4., interaction_time=5.)
    return metadata, source, rep


def bundle(root, **kwargs):
    meta, source, rep = source_inputs(**kwargs)
    save_grounding_skill(root, metadata=meta, source=source, representation=rep)
    return meta, source, rep


def detection(label):
    mask = np.zeros((10, 12), bool)
    mask[2:8, 2:10] = True
    return dict(label=label, candidate_id=17, score=.6, bbox=[2, 2, 10, 8], mask=mask)


class GroundingSkillTest(unittest.TestCase):
    def test_self_contained_roundtrip_retains_point_identity_motion_and_missing_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            original, copied = Path(directory)/"original", Path(directory)/"copy"
            meta, source, rep = bundle(original)
            shutil.copytree(original, copied)
            shutil.rmtree(original)
            skill = load_grounding_skill(copied)
            self.assertEqual(skill["metadata"], meta)
            for name, value in source["observation"].items():
                np.testing.assert_array_equal(skill["source"]["observation"][name], value)
            np.testing.assert_array_equal(skill["representation"]["H_g"], rep["H_g"])
            np.testing.assert_array_equal(skill["representation"]["timestamps"], [4., 4.5, 5.])
            self.assertIn("feature", skill["source"]["missing"])
            self.assertNotIn("source_template", skill)
            self.assertNotIn("program", skill)

    def test_saved_feature_and_source_frame_are_bundle_owned(self):
        with tempfile.TemporaryDirectory() as directory:
            meta, source, rep = source_inputs()
            source.update(feature=np.arange(48, dtype=np.float32).reshape(1, 3, 4, 4),
                          feature_model={"name": "fixture spatial extractor", "resolution": 4})
            root = Path(directory)/"skill"
            save_grounding_skill(root, metadata=meta, source=source, representation=rep)
            loaded = load_grounding_skill(root)
            np.testing.assert_array_equal(loaded["source"]["feature"], source["feature"])
            np.testing.assert_array_equal(loaded["source"]["L_body_region"], source["L_body_region"])

    def test_offline_semantics_drive_object_and_part_selection(self):
        for scope in ("object", "part"):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)/"skill"
                meta, _, _ = bundle(root, scope=scope)
                phrase = meta["semantic"]["interaction_reference"]["text"]
                row = detection(phrase)
                detector = Mock(return_value=[row])
                client = Mock(chat_json=Mock(return_value={"views": {"camera": {
                    "candidate_ids": [17], "reason": "This instance satisfies the current request."}}}))
                client.chat_json.return_value.update(same_entity=True, same_entity_reason="One selected view only")
                result = observe_interaction_reference(load_grounding_skill(root), "Use the right instance",
                    {"camera": {"image": Image.new("RGB", (12, 10))}}, detector=detector, client=client)
                self.assertEqual(detector.call_args.args[1], [phrase])
                self.assertEqual(result["selection_scope"], "entity" if scope == "object" else "interaction_reference")
                context = json.loads(client.chat_json.call_args.args[0].split("\nInput data:\n", 1)[1])
                self.assertEqual(context["current_instruction"], "Use the right instance")
                self.assertNotIn("source_body", client.chat_json.call_args.args[0])
                np.testing.assert_array_equal(result["selection"]["views"]["camera"]["regions"][0]["mask"], row["mask"])

    def test_contact_free_does_not_create_contact_region(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"skill"
            bundle(root, contact=False)
            skill = load_grounding_skill(root)
            client = Mock(chat_json=Mock(return_value={"views": {"camera": {
                "candidate_ids": [17], "reason": "One visible reference entity."}}}))
            client.chat_json.return_value.update(same_entity=True, same_entity_reason="One selected view only")
            result = observe_interaction_reference(skill, "Use this reference",
                {"camera": {"image": Image.new("RGB", (12, 10))}},
                detector=Mock(return_value=[detection("container")]), client=client)
            self.assertEqual(result["selection_scope"], "entity")
            prompt = client.chat_json.call_args.args[0]
            self.assertIn("several independently moving rigid bodies", prompt)
            self.assertIn("no contact region supplied", prompt)
            self.assertNotIn("region_image_mask", skill["source"])
            self.assertFalse(skill["source"]["observation"]["region_mask"].any())

    def test_part_can_be_missing_without_object_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"skill"
            bundle(root, scope="part")
            client = Mock()
            result = observe_interaction_reference(load_grounding_skill(root), "Use this instance",
                {"camera": {"image": Image.new("RGB", (12, 10))}}, detector=Mock(return_value=[]), client=client)
            self.assertEqual(result["selection"]["views"]["camera"]["status"], "no_candidates")
            self.assertIsNone(result["selection"]["same_entity"])
            client.chat_json.assert_not_called()

    # BEGIN component identity needs the same joint evidence as entity identity.
    def test_joint_selection_preserves_uncertain_entity_or_component_identity(self):
        for scope in ("object", "part"):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)/"skill"
                bundle(root, scope=scope)
                skill = load_grounding_skill(root)
                phrase = skill["metadata"]["semantic"]["interaction_reference"]["text"]
                views = {name: {"image": Image.new("RGB", (12, 10))} for name in ("a", "b")}
                for same in (True, False, None):
                    client = Mock(chat_json=Mock(return_value={
                        "views": {name: {"candidate_ids": [17], "reason": "Selected reference in this view"} for name in views},
                        "same_entity": same, "same_entity_reason": "Recorded cross-view visual evidence or uncertainty"}))
                    result = observe_interaction_reference(skill, "Use the intended container", views,
                        detector=Mock(return_value=[detection(phrase)]), client=client)
                    self.assertIs(result["selection"]["same_entity"], same)
                    self.assertEqual(result["selection"]["association_scope"], "cross_view")
                    self.assertEqual(client.chat_json.call_count, 1)
                    self.assertIn("same particular component instance", client.chat_json.call_args.args[0])

    def test_component_selection_cannot_omit_its_instance_association(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"skill"
            bundle(root, scope="part")
            client = Mock(chat_json=Mock(return_value={"views": {"camera": {
                "candidate_ids": [17], "reason": "A matching component name is visible."}}}))
            with self.assertRaises(ValueError):
                observe_interaction_reference(load_grounding_skill(root), "Use the component",
                    {"camera": {"image": Image.new("RGB", (12, 10))}},
                    detector=Mock(return_value=[detection("container handle")]), client=client)
    # END no selected component receives a fabricated cross-view instance ID.

    def test_old_schema_and_external_payload_are_rejected(self):
        for bad in ("schema", "path"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)/"skill"
                bundle(root)
                manifest = json.loads((root/"skill.json").read_text())
                if bad == "schema":
                    manifest["schema"] = "functo.grounding_skill.1"
                else:
                    manifest["files"]["motion"] = "../outside.npz"
                (root/"skill.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    load_grounding_skill(root)

    def test_source_time_pixel_identity_frame_and_body_must_agree_with_motion(self):
        for bad in ("time", "region", "frame", "body", "role"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                meta, source, rep = source_inputs()
                if bad == "time":
                    source["timestamp_s"] = 4.2
                elif bad == "region":
                    source["observation"]["region_mask"] = np.roll(source["observation"]["region_mask"], 1)
                elif bad == "frame":
                    source["L_body_region"][0, 3] = .02
                elif bad == "body":
                    source["T_world_body_at_image"][0, 3] = .02
                else:
                    meta["owner_role"] = "reference"
                with self.assertRaises(ValueError):
                    save_grounding_skill(Path(directory)/"skill", metadata=meta, source=source, representation=rep)
# END new portable bundle tests.
