"""Independent fixture-feature checks against the actual standalone robot tool."""

# BEGIN calibrated fixture/profile consistency regressions.
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
from generated_grasp_tool_binding import validate_tool_binding
from manipulation.contact_binding import calibrate_box_material_features
from manipulation.tool_geometry import load_robosuite_panda_tool


class GeneratedGraspToolBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_robosuite_panda_tool(tool_id="declared_robot", tool_frame="measured_tcp")
        # Authored local +Y/-Y faces are chosen independently of the tool
        # adapter's automatic inward-face selection. The fixture calibration
        # then derives their zero-coordinate corners and affine motion.
        faces = [
            {"material_id": "alpha", "native_geometry": "finger1_pad_collision",
             "face_axis": 1, "face_sign": 1, "source": "native inward box face"},
            {"material_id": "beta", "native_geometry": "finger2_pad_collision",
             "face_axis": 1, "face_sign": -1, "source": "native inward box face"},
        ]
        calibrated = calibrate_box_material_features(
            cls.tool._model, tcp_site="grip_site", joint_positions={"finger_joint1": .013, "finger_joint2": -.021},
            finger_joint_names=("finger_joint1", "finger_joint2"), material_faces=faces)
        cls.rules = [{"rule_id": f"permission_{index}", **calibrated[name]}
                     for index, name in enumerate(("alpha", "beta"))]
        cls.limits = cls.tool.joint_ranges_m.copy()

    def test_same_native_geometry_passes_with_independent_coordinate_probes(self):
        result = validate_tool_binding(self.tool, self.rules, self.limits)
        self.assertEqual(result["status"], "matched_native_tool_geometry")
        self.assertEqual(result["material_to_profile_pad_index"], {"alpha": 0, "beta": 1})
        self.assertEqual(len(result["checked_configurations"]), 5)
        closed = next(item for item in result["checked_configurations"] if item["configuration"] == "closed")
        self.assertAlmostEqual(closed["native_signed_gap_m"], -.001, places=12)
        self.assertFalse(result["target_contact_verified"])
        self.assertFalse(result["holding_verified"])

    def test_material_and_corner_order_can_change_without_changing_geometry(self):
        rules = deepcopy(list(reversed(self.rules)))
        for rule in rules:
            order = [2, 0, 3, 1]
            for key in ("feature_points", "feature_joint_translation"):
                rule[key] = np.asarray(rule[key])[order].tolist()
        rules.append({**deepcopy(rules[0]), "rule_id": "second_patch_same_material"})
        result = validate_tool_binding(self.tool, rules, self.limits)
        self.assertEqual(result["material_to_profile_pad_index"], {"alpha": 0, "beta": 1})

    def test_coordinate_domain_mismatch_is_rejected_without_permutation_or_sign_fix(self):
        for limits in (self.limits[::-1], self.limits + [.0, .01], np.zeros((2, 2))):
            with self.subTest(limits=limits.tolist()), self.assertRaisesRegex(ValueError, "coordinate domain"):
                validate_tool_binding(self.tool, self.rules, limits)

    def test_shifted_or_wrong_normal_faces_are_rejected(self):
        shifted = deepcopy(self.rules)
        shifted[0]["feature_points"] = (np.asarray(shifted[0]["feature_points"]) + [0., 0., .01]).tolist()
        flipped = deepcopy(self.rules)
        flipped[0]["feature_normal"] = (-np.asarray(flipped[0]["feature_normal"])).tolist()
        for rules in (shifted, flipped):
            with self.subTest(), self.assertRaisesRegex(ValueError, "inner pad geometry"):
                validate_tool_binding(self.tool, rules, self.limits)

    def test_same_union_bounds_do_not_hide_different_material_faces(self):
        rules = deepcopy(self.rules)
        # Swap one corner between materials. The union is unchanged, but
        # neither permission may replace its actual native material rectangle.
        rules[0]["feature_points"][0], rules[1]["feature_points"][0] = (
            rules[1]["feature_points"][0], rules[0]["feature_points"][0])
        with self.assertRaisesRegex(ValueError, "inner pad geometry"):
            validate_tool_binding(self.tool, rules, self.limits)

    def test_synchronous_closure_cannot_conceal_wrong_affine_coupling(self):
        rules = deepcopy(self.rules)
        affine = np.asarray(rules[0]["feature_joint_translation"])
        affine[:, 2, :] += .2  # Wrong dz = .2 * (q1 + q2) cancels on open/mid/closed.
        rules[0]["feature_joint_translation"] = affine.tolist()
        with self.assertRaisesRegex(ValueError, "independent_vertex"):
            validate_tool_binding(self.tool, rules, self.limits)

    def test_alternative_permission_cannot_redefine_the_same_material(self):
        rules = deepcopy(self.rules)
        alternative = deepcopy(rules[0])
        alternative["rule_id"] = "second_patch"
        alternative["feature_joint_translation"][0][0][0] += .01
        rules.append(alternative)
        with self.assertRaisesRegex(ValueError, "Alternative patches disagree"):
            validate_tool_binding(self.tool, rules, self.limits)


if __name__ == "__main__":
    unittest.main()
# END calibration comparison adds no target or execution evidence.
