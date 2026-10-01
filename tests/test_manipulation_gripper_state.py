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
        scene = {
            "state": "current_observed", "timestamp_s": 2.5, "reference_frame": "world",
            "carrier_id": "observed-part",
            "observation_ids": ["live-view"], "initial_pose": np.eye(4).tolist(),
            "eef_pose": np.eye(4).tolist(),
            "holding": {"verified": True, "source": "Explicit synthetic unit-test contact evidence",
                        "robot_feedback_id": "live-closure-stop", "carrier_id": "observed-part",
                        "observation_ids": ["live-view"], "T_carrier_eef": np.eye(4).tolist()},
        }
        return robot, feedback, scene

    def test_measured_asymmetric_fingers_remain_external_to_optimizer(self):
        robot, feedback, scene = self.inputs()
        before = deepcopy((robot, feedback, scene))
        result = build_post_grasp_robot_config(robot, feedback, scene)
        kinematics = result["robot_config"]["robot_cfg"]["kinematics"]
        self.assertEqual(kinematics["lock_joints"], {"finger_a": 0.021, "finger_b": -0.017})
        self.assertEqual(kinematics["cspace"]["default_joint_position"], [0.3, 0.4, 0.021, -0.017])
        self.assertEqual(result["planning_feedback"]["joint_names"], ["arm_a", "arm_b"])
        self.assertEqual(result["planning_feedback"]["start_q"], [0.1, 0.2])
        self.assertEqual(result["planning_feedback"]["start_dq"], [0.01, 0.02])
        self.assertEqual(result["hand_binding"]["finger_velocities"], [0.0003, -0.0002])
        self.assertEqual(result["hand_binding"]["carrier_id"], "observed-part")
        self.assertEqual(result["full_feedback"]["q"], feedback["q"])
        self.assertEqual(kinematics["collision_spheres"], robot["robot_cfg"]["kinematics"]["collision_spheres"])
        self.assertEqual(kinematics["self_collision_ignore"], robot["robot_cfg"]["kinematics"]["self_collision_ignore"])
        self.assertEqual((robot, feedback, scene), before)

    def test_unconfirmed_or_stale_holding_cannot_create_hand_model(self):
        changes = (
            lambda f, s: s["holding"].update(verified=False),
            lambda f, s: s["holding"].update(robot_feedback_id="old-snapshot"),
            lambda f, s: s["holding"].update(source=""),
            lambda f, s: s["holding"].update(carrier_id="another-part"),
            lambda f, s: s["holding"].update(observation_ids=["old-view"]),
            lambda f, s: s.update(timestamp_s=2.4),
            lambda f, s: s.update(state="predicted"),
            lambda f, s: s.update(observation_ids=[]),
            lambda f, s: s.update(reference_frame="another-frame"),
            lambda f, s: s["eef_pose"][0].__setitem__(3, 0.01),
            lambda f, s: s["holding"]["T_carrier_eef"][1].__setitem__(3, 0.01),
        )
        for change in changes:
            with self.subTest(change=change):
                robot, feedback, scene = self.inputs()
                change(feedback, scene)
                with self.assertRaises(ValueError):
                    build_post_grasp_robot_config(robot, feedback, scene)

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
                robot, feedback, scene = self.inputs()
                change(robot, feedback)
                with self.assertRaises(ValueError):
                    build_post_grasp_robot_config(robot, feedback, scene)


if __name__ == "__main__":
    unittest.main()
# END post-grasp state and geometry binding regressions.
