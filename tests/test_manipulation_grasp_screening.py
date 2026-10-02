"""Integration contracts: screening never deletes the generated skill evidence."""
from copy import deepcopy
import unittest

import numpy as np

from manipulation.perception import ground_grasp_candidates
from manipulation.tool_geometry import load_robosuite_panda_tool


class GraspScreeningIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_robosuite_panda_tool(tool_id="test_tool", tool_frame="test_grip_site")

    def fixture(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        program = {"skill_kind": "grasp", "demo": {
            "carrier_id": "offline_owner", "tool_id": self.tool.tool_id,
            "tool_frame": self.tool.tool_frame, "T_carrier_eef": poses,
            "frame_indices": [10, 20, 30], "timestamps": [0., .1, .2],
            "protected_start_frame": 20,
            "gripper_events": [{"event_id": "opening", "frame_index": 10, "command": "open"},
                               {"event_id": "closing", "frame_index": 20, "command": "close"}]}}
        scene = {"carrier_id": "observed_owner", "reference_frame": "world",
                 "tool_id": self.tool.tool_id, "tool_frame": self.tool.tool_frame,
                 "eef_pose": np.eye(4), "carrier_pose": np.eye(4), "region_transform": np.eye(4),
                 "grasp_goal": {"T_carrier_eef": np.eye(4), "position_tolerance_m": .002,
                                "rotation_tolerance_rad": .03, "source": "declared screening fixture task relation"}}
        view = {"owner_id": "observed_owner", "reference_frame": "world",
                "depth": np.ones((15, 15)), "owner_mask": np.zeros((15, 15), bool),
                "K": np.array([[1000., 0., 7.], [0., 1000., 7.], [0., 0., 1.]]),
                "T_output_camera": np.eye(4)}
        return program, scene, view

    def call(self, program, scenes, view):
        return ground_grasp_candidates(program, scenes, {"camera": view}, self.tool,
            current_joint_positions_m=self.tool.open_joint_positions_m,
            contact_tolerance_m=.001, penetration_tolerance_m=.00025)

    # BEGIN allowed incomplete observation selects an attempt, not a held-object fact.
    def test_missing_observation_preserves_full_candidate_and_unknown_holding(self):
        program, scene, view = self.fixture()
        result = self.call(program, {"candidate": scene}, view)
        self.assertEqual(result["status"], "selected")
        self.assertIsNotNone(result["selected_trajectory"])
        self.assertIn("candidate", result["grasp_geometry_validation"]["unresolved_candidate_ids"])
        reference = result["trajectories"]["candidate"]
        np.testing.assert_array_equal(reference["eef_poses"], program["demo"]["T_carrier_eef"])
        self.assertEqual(reference["gripper_events"], program["demo"]["gripper_events"])
        self.assertEqual(reference["holding_status"], "requires_verification")
        self.assertFalse(result["robot_execution_verified"])
    # END eligibility does not certify physical execution or effects.

    def test_nominal_palm_penetration_retains_goal_set_and_complete_reference(self):
        program, scene, view = self.fixture()
        geom = next(g for g in self.tool.collision_geometries if g["role"] == "palm")
        state = self.tool.at(self.tool.open_joint_positions_m)
        pose = next(g["T_tool_geom"] for g in state["geoms"] if g["geom_id"] == geom["geom_id"])
        center = pose[:3, :3] @ geom["vertices"].mean(0) + pose[:3, 3]
        view["T_output_camera"][:3, 3] = center - [0., 0., 1.]
        view["owner_mask"][:] = True
        result = self.call(program, {"candidate": scene}, view)
        report = result["grasp_geometry_validation"]["candidates"]["candidate"]
        self.assertEqual(report["status"], "unresolved")
        self.assertEqual(report["terminal_visible_geometry_status"], "rejected")
        self.assertFalse(report["allowed_set_infeasibility_proven"])
        self.assertIn("candidate", result["grasp_geometry_validation"]["eligible_candidate_ids"])
        self.assertIn("candidate", result["trajectories"])
        np.testing.assert_array_equal(result["trajectories"]["candidate"]["eef_poses"][1:],
                                      program["demo"]["T_carrier_eef"][1:])
        self.assertNotIn("grasp_geometry_validation", scene)

    def test_candidates_cannot_mix_feedback_times_or_owners(self):
        program, scene, view = self.fixture()
        for key in ("eef_pose", "carrier_id"):
            other = deepcopy(scene)
            if key == "eef_pose":
                other[key][0, 3] += .01
            else:
                other[key] = "different_owner"
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.call(program, {"a": scene, "b": other}, view)


if __name__ == "__main__":
    unittest.main()
