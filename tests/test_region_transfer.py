"""CPU checks for selected-entity inputs and source-frame propagation.

Only the existing GPU pipeline boundary is mocked. Point backprojection,
source query identities, candidate choice, SE(3) composition and the Open3D
local rigid fit execute normally; no model or CUDA allocation is requested.
"""

# BEGIN portable dual-view adapter regression checks.
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.region_transfer import (frame_from_deformation,
    prepare_inputs, propagate_region_frame, run_region_transfer, select_reference_transform)
from transfer.segmentation import prepare_selected_observation


def pose(rotation=(0., 0., 0.), translation=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
    result[:3, 3] = translation
    return result


class SelectedObservationTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((6, 7, 3), np.uint8)
        self.depth = np.full((6, 7), 2.)
        self.K = np.array([[20., 0., 3.], [0., 20., 2.5], [0., 0., 1.]])
        self.selected = dict(label="object", score=.1, candidate_id=7,
                             bbox=[0., 0., 7., 6.], mask=np.zeros((6, 7), bool))
        self.selected["mask"][1:4, 1:4] = True

    def test_only_supplied_low_score_instance_is_prepared_and_never_reselected(self):
        result = prepare_selected_observation(self.rgb, self.depth, self.K, np.eye(4), self.selected)
        np.testing.assert_array_equal(result["observation"]["object_mask"], self.selected["mask"])
        self.assertEqual(result["selection"]["policy"], "provided_selected_instance")
        self.assertEqual(result["selection"]["candidate_id"], 7)
        self.depth[self.selected["mask"]] = np.nan
        missing = prepare_selected_observation(self.rgb, self.depth, self.K, np.eye(4), self.selected)
        self.assertFalse(missing["prepared"])
        self.assertEqual(missing["reasons"], ["selected_instance_without_valid_depth"])
        # The other valid image pixels cannot become an alternative instance.
        self.assertIsNone(missing["observation"])

    def test_given_source_region_survives_a_missed_contact_patch(self):
        region = np.zeros((6, 7), bool)
        region[4, 5] = True
        result = prepare_selected_observation(self.rgb, self.depth, self.K, np.eye(4),
                                              self.selected, source_region_mask=region)
        observation = result["observation"]
        np.testing.assert_array_equal(observation["pixels"][observation["evidence_only"]], [[5, 4]])
        np.testing.assert_array_equal(observation["object_mask"], self.selected["mask"])
        self.assertEqual(result["diagnostics"]["source_evidence_coverage_fraction"], 0.)


class RegionTransferTests(unittest.TestCase):
    def setUp(self):
        rgb = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
        depth = np.full((8, 9), 1.5)
        K = np.array([[50., 0., 4.], [0., 50., 3.5], [0., 0., 1.]])
        entity = np.ones((8, 9), bool)
        region = np.zeros_like(entity)
        region[2:6, 2:7] = True
        detection = dict(label="object", score=.4, bbox=[0., 0., 9., 8.], mask=entity)
        prepared = prepare_selected_observation(rgb, depth, K, np.eye(4), detection,
                                                source_region_mask=region)
        observation = prepared["observation"]
        # The point bank order deliberately differs from raster order and
        # from the selected query array; neither may be mistaken for the other.
        order = np.r_[np.arange(1, len(observation["xyz"]), 2), np.arange(0, len(observation["xyz"]), 2)]
        for key in ("xyz", "pixels", "entity_member", "region_mask", "evidence_only"):
            observation[key] = observation[key][order]
        feature = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
        self.source = dict(observation=observation, depth_m=depth, crop_xyxy=prepared["crop_xyxy"],
            feature=feature, timestamp_s=12., frame_index=120, owner_id="source_part",
            T_world_body_at_image=pose((.1, .2, .3), (.4, -.5, .6)),
            L_body_region=pose((-.2, .1, .4), (.03, .02, -.01)))
        target = prepare_selected_observation(rgb, depth, K, np.eye(4), detection)
        self.views = [dict(id=name, entity_id="chosen_cabinet", timestamp_s=20.+index,
            depth_m=depth, K=K, T_world_camera_cv=np.eye(4), observation=target["observation"],
            crop_xyxy=target["crop_xyxy"], feature=feature.copy(), missing_reasons=[])
            for index, name in enumerate(("view_a", "view_b"))]
        self.state_source = "Frozen geometry; no physics step between these two observations"
        self.owner = dict(owner_id="moving_drawer", T_world_body=pose((.2, -.1, .1), (.2, .3, .4)),
                          timestamp_s=20., source="Same frozen state, separately recorded region body")

    def inference_result(self, accepted=True):
        first = dict(hypothesis_id=9, score=8., admitted_to_H=True,
                     transform=[1.7, Rotation.from_euler("z", .4).as_matrix(), [.2, -.1, .3]],
                     paired_query_rows=[0, 2], paired_surface_indices=[1, 3])
        second = dict(first, hypothesis_id=2,
                      transform=[1.2, Rotation.from_euler("z", -.3).as_matrix(), [-.1, .2, .1]])
        return dict(accepted=accepted, status="accepted_observed_region" if accepted else "rejected_ambiguous_region",
            reasons=[] if accepted else ["competing_regions"], finite_candidate_unique=False,
            sim3_scale=None, sim3_rotation=None, sim3_translation=None,
            modes=[dict(type="primary", hypothesis_indices=[9, 2], sample_indices=np.array([2, 4]))]
                if accepted else [],
            diagnostics=dict(hypotheses=[first, second], quality_warnings=["tied_estimates_common_region_only"]))

    def test_resident_input_uses_one_source_map_and_keeps_missing_view_depth(self):
        self.views[1].update(observation=None, feature=None, missing_reasons=["entity_occluded"])
        loaded, features = prepare_inputs(self.source, self.views, same_state_source=self.state_source)
        self.assertIs(loaded["source"], self.source["observation"])
        self.assertIs(features["view_a"][0], self.source["feature"])
        self.assertNotIn("view_b", features)
        self.assertIs(loaded["views"][1]["depth_m"], self.views[1]["depth_m"])
        self.assertEqual(loaded["views"][1]["missing_reasons"], ["entity_occluded"])
        self.assertEqual(loaded["source_timestamp_s"], 12.)
        self.views[1]["entity_id"] = "other_cabinet"
        with self.assertRaisesRegex(ValueError, "same already selected entity"):
            prepare_inputs(self.source, self.views, same_state_source=self.state_source)

    def test_pairing_uses_raw_point_bank_and_tie_keeps_original_region_decision(self):
        result = self.inference_result()
        outputs = dict(result=result, surface=dict(xyz=np.zeros((5, 3))), source={"fixture": "prepared_queries"})
        with patch("transfer.dual_view.pipeline.run", return_value=outputs) as numerical:
            returned = run_region_transfer(self.source, self.views, same_state_source=self.state_source,
                                           current_region_owner=self.owner)
        numerical.assert_called_once()
        self.assertIs(returned["outputs"], outputs)
        self.assertFalse(result["finite_candidate_unique"])
        self.assertIsNone(result["sim3_rotation"])
        selected = returned["selected_transform"]
        self.assertEqual(selected["hypothesis_id"], 2)
        np.testing.assert_array_equal(selected["source_point_indices"], returned["loaded"]["query_indices"][[0, 2]])
        np.testing.assert_array_equal(selected["paired_surface_indices"], [1, 3])
        self.assertEqual(returned["reference_frame"]["current_owner_id"], "moving_drawer")
        self.assertEqual(returned["reference_frame"]["target_entity_id"], "chosen_cabinet")
        self.assertEqual(returned["reference_frame"]["current_owner_same_state_source"], self.state_source)
        np.testing.assert_allclose(self.owner["T_world_body"] @ returned["reference_frame"]["L_body_region"],
                                   returned["reference_frame"]["T_world_region"], atol=1e-14)

    def test_frame_maps_image_time_origin_and_only_rotates_unit_axes(self):
        selected = select_reference_transform(self.inference_result())
        frame = propagate_region_frame(self.source, selected)
        source_at_image = self.source["T_world_body_at_image"] @ self.source["L_body_region"]
        expected = pose()
        expected[:3, :3] = selected["rotation"] @ source_at_image[:3, :3]
        expected[:3, 3] = selected["scale"] * selected["rotation"] @ source_at_image[:3, 3] + selected["translation"]
        np.testing.assert_allclose(frame["T_world_region"], expected, atol=1e-14)
        np.testing.assert_allclose(np.linalg.det(frame["T_world_region"][:3, :3]), 1., atol=1e-14)
        self.assertIsNone(frame["L_body_region"])
        self.assertEqual(frame["source_timestamp_s"], 12.)
        # Mapping a different later body pose would give a different position:
        # the adapter must use the pose stored at the actual region image time.
        later = source_at_image.copy()
        later[0, 3] += .2
        self.assertGreater(np.linalg.norm(frame["T_world_region"][:3, 3] -
            (selected["scale"] * selected["rotation"] @ later[:3, 3] + selected["translation"])), .1)

    def test_rejected_and_missing_evidence_do_not_create_a_reference_frame(self):
        rejected = dict(result=self.inference_result(False), surface={}, source={})
        with patch("transfer.dual_view.pipeline.run", return_value=rejected):
            returned = run_region_transfer(self.source, self.views, same_state_source=self.state_source)
        self.assertIsNone(returned["reference_frame"])
        self.assertEqual(returned["reasons"], ["competing_regions"])
        del self.source["feature"]
        with patch("transfer.dual_view.pipeline.run") as numerical:
            returned = run_region_transfer(self.source, self.views, same_state_source=self.state_source)
        numerical.assert_not_called()
        self.assertEqual(returned["status"], "missing")
        self.assertIn("missing_source_feature", returned["reasons"])

    def test_local_binding_needs_owner_and_region_same_state_evidence(self):
        selected = select_reference_transform(self.inference_result())
        self.owner["timestamp_s"] = 19.
        with self.assertRaisesRegex(ValueError, "same-state evidence"):
            propagate_region_frame(self.source, selected, current_region_owner=self.owner)
        # A recorded static interval permits different capture times without
        # relabeling an older pose as a new measured pose.
        result = propagate_region_frame(self.source, selected, current_region_owner=self.owner,
            same_state_source="The owner remained stationary from the pose sample through both RGB-D images")
        self.assertEqual(result["current_owner_pose_source"]["timestamp_s"], 19.)
        self.assertIn("stationary", result["current_owner_same_state_source"])

    def test_malformed_depth_pixel_binding_cannot_enter_numerical_pipeline(self):
        self.source["observation"]["xyz"][0, 0] += .01
        with self.assertRaises(AssertionError):
            prepare_inputs(self.source, self.views, same_state_source=self.state_source)


class DeformationFrameTests(unittest.TestCase):
    def test_local_fit_uses_coarse_points_and_source_indices_once(self):
        original = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .2, 0.], [.1, .2, 0.], [1., 1., 1.]])
        coarse = pose((.2, -.3, .4), (.7, -.5, 1.))
        points = original @ coarse[:3, :3].T + coarse[:3, 3]
        correction = pose((-.05, .08, .03), (.01, -.02, .03))
        deformed = points @ correction[:3, :3].T + correction[:3, 3]
        deformed[-1] = [30., 40., 50.]  # Nonregional context must not define the local frame.
        result = frame_from_deformation(points, deformed, np.array([3, 0, 2, 1]), coarse)
        np.testing.assert_allclose(result["T_world_region"], correction @ coarse, atol=1e-12)
        self.assertLess(result["rigid_fit_rms_m"], 1e-12)
        with self.assertRaisesRegex(ValueError, "noncollinear"):
            frame_from_deformation(np.c_[np.arange(4.), np.zeros((4, 2))],
                                   np.c_[np.arange(4.), np.zeros((4, 2))], np.arange(4), np.eye(4))


if __name__ == "__main__":
    unittest.main()
# END no test interprets a generated reference as physical task success.
