"""Measured hand binding must never substitute servo targets for feedback."""

# BEGIN post-grasp state and geometry binding regressions.
from copy import deepcopy
import unittest

import numpy as np

from manipulation.gripper_state import build_post_grasp_robot_config


class MeasuredHandBindingTests(unittest.TestCase):
    def inputs(self):
        names = ["arm_b", "finger_b", "arm_a", "finger_a"]
        robot = {"robot_cfg": {"kinematics": {
            "cspace": {"joint_names": ["arm_a", "arm_b", "finger_a", "finger_b"],
                       "default_joint_position": [0.3, 0.4, 0.04, -0.04]},
            "lock_joints": {"finger_a": 0.04, "finger_b": -0.04},
            "collision_spheres": {"hand": [{"center": [0., 0., 0.], "radius": 0.01}]},
            "self_collision_ignore": {"hand": ["arm"]},
        }}}
        feedback = {
            "joint_names": names, "q": [0.2, -0.017, 0.1, 0.021],
            "dq": [0.02, -0.0002, 0.01, 0.0003],
            "arm_joint_names": ["arm_a", "arm_b"],
            "finger_joint_names": ["finger_a", "finger_b"],
            "feedback_id": "live-closure-stop", "timestamp_s": 2.5,
            "source": "Named measured robot encoders", "reference_frame": "world",
            "base_pose": np.eye(4).tolist(), "tool_pose": np.eye(4).tolist(),
            "tool_link": "tool", "tool_transform": np.eye(4).tolist(),
            "commanded_finger_positions": [0., 0.],
        }
        return robot, feedback

    def test_measured_asymmetric_fingers_remain_external_to_optimizer(self):
        robot, feedback = self.inputs()
        before = deepcopy((robot, feedback))
        result = build_post_grasp_robot_config(robot, feedback)
        kinematics = result["robot_config"]["robot_cfg"]["kinematics"]
        self.assertEqual(kinematics["lock_joints"], {"finger_a": 0.021, "finger_b": -0.017})
        self.assertEqual(kinematics["cspace"]["default_joint_position"], [0.3, 0.4, 0.021, -0.017])
        self.assertEqual(result["planning_feedback"]["joint_names"], ["arm_a", "arm_b"])
        self.assertEqual(result["planning_feedback"]["start_q"], [0.1, 0.2])
        self.assertEqual(result["planning_feedback"]["start_dq"], [0.01, 0.02])
        self.assertEqual(result["hand_binding"]["finger_velocities"], [0.0003, -0.0002])
        self.assertNotIn("carrier_id", result["hand_binding"])
        self.assertNotIn("T_carrier_eef", result["hand_binding"])
        self.assertEqual(result["full_feedback"]["q"], feedback["q"])
        self.assertEqual(kinematics["collision_spheres"], robot["robot_cfg"]["kinematics"]["collision_spheres"])
        self.assertEqual(kinematics["self_collision_ignore"], robot["robot_cfg"]["kinematics"]["self_collision_ignore"])
        self.assertEqual((robot, feedback), before)

    # BEGIN object observations cannot alter a robot-only hand binding.
    def test_unknown_or_false_holding_does_not_change_encoder_geometry(self):
        robot, feedback = self.inputs()
        expected = build_post_grasp_robot_config(robot, feedback)
        for value in (False, None):
            feedback["scene_condition_evidence"] = {"Holding": value}
            actual = build_post_grasp_robot_config(robot, feedback)
            self.assertEqual(actual["robot_config"], expected["robot_config"])
            self.assertEqual(actual["hand_binding"], expected["hand_binding"])
    # END no scene/owner/time proof is requested for measured finger q.

    def test_missing_state_or_changed_active_layout_is_rejected(self):
        changes = (
            lambda r, f: f.update(q=[0., 0., 0.]),
            lambda r, f: f.update(dq=[0., float("nan"), 0., 0.]),
            lambda r, f: f.update(arm_joint_names=["arm_b", "arm_a"]),
            lambda r, f: f.update(finger_joint_names=["finger_a", "finger_a"]),
            lambda r, f: r["robot_cfg"]["kinematics"]["lock_joints"].update(arm_a=0.),
        )
        for change in changes:
            with self.subTest(change=change):
                robot, feedback = self.inputs()
                change(robot, feedback)
                with self.assertRaises(ValueError):
                    build_post_grasp_robot_config(robot, feedback)


if __name__ == "__main__":
    unittest.main()
# END post-grasp state and geometry binding regressions.
