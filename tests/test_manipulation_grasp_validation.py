"""Synthetic calibrated surfaces test the grasp-geometry evidence contract."""

# BEGIN self-contained geometry tests; no simulator scene, assets or GT fixture.
from copy import deepcopy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.grasp_validation import screen_grasp_candidates


def pose(x=0., y=0., z=0.):
    value = np.eye(4)
    value[:3, 3] = [x, y, z]
    return value


def box(name, role, half):
    half = np.asarray(half, float)
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*half
    planes = np.concatenate((np.column_stack((np.eye(3), -half)),
                             np.column_stack((-np.eye(3), -half))))
    return {"geom_id": name, "role": role, "vertices": vertices, "halfspaces": planes}


class Tool:
    tool_id, tool_frame = "shared_test_tool", "known_tcp"
    open_joint_positions_m = np.array([.02, -.02])
    closed_joint_positions_m = np.zeros(2)
    joint_ranges_m = np.array([[0., .02], [-.02, 0.]])

    def __init__(self):
        self.collision_geometries = [box("palm", "palm", [.03, .02, .005]),
                                     box("finger0", "finger", [.002, .01, .01]),
                                     box("finger1", "finger", [.002, .01, .01]),
                                     box("pad0", "pad", [.0005, .01, .01]),
                                     box("pad1", "pad", [.0005, .01, .01])]

    def at(self, q):
        q = np.asarray(q, float)
        if q.shape != (2,) or not np.isfinite(q).all() or np.any(q < self.joint_ranges_m[:, 0]) or np.any(q > self.joint_ranges_m[:, 1]):
            raise ValueError("Explicit finite joint coordinates in the mechanical range required")
        poses = {"palm": pose(z=-.03), "finger0": pose(x=-q[0]-.003),
                 "finger1": pose(x=-q[1]+.003), "pad0": pose(x=-q[0]-.0005),
                 "pad1": pose(x=-q[1]+.0005)}
        pads = []
        for i, x in enumerate((-q[0], -q[1])):
            mat = pose(x=x)
            mat[:3, :3] = np.column_stack(([0., 1. if i == 0 else -1., 0.],
                                           [0., 0., 1.], [1. if i == 0 else -1., 0., 0.]))
            pads.append({"pad_index": i, "geom_id": "pad"+str(i), "T_tool_pad": mat,
                         "half_sizes_xy_m": np.array([.01, .01])})
        return {"tool_id": self.tool_id, "tool_frame": self.tool_frame, "joint_positions_m": q.copy(),
                "geoms": [{"geom_id": k, "T_tool_geom": v} for k, v in poses.items()],
                "pads": pads, "signed_gap_m": float(q[0]-q[1])}


def view(side, half_width=.005, shift_y=0., shift_z=0.):
    camera = pose(x=side*.2, y=shift_y, z=shift_z)
    camera[:3, :3] = np.column_stack(([0., -side, 0.], [0., 0., 1.], [-side, 0., 0.]))
    return {"reference_frame": "world", "owner_id": "selected_target",
            "depth": np.full((21, 21), .2-half_width),
            "K": np.array([[500., 0., 10.], [0., 500., 10.], [0., 0., 1.]]),
            "owner_mask": np.ones((21, 21), bool), "T_output_camera": camera}


def trajectory():
    return {"reference_frame": "world", "tool_id": Tool.tool_id, "tool_frame": Tool.tool_frame,
            "eef_poses": np.array([pose(z=-.05), pose(), pose()]), "frame_indices": [0, 1, 2],
            "gripper_events": [{"frame_index": 0, "command": "open"}, {"frame_index": 1, "command": "close"}],
            "connection_required": False}


class GraspValidationTest(unittest.TestCase):
    def run_check(self, candidates=None, views=None, tool=None, **options):
        defaults = dict(reference_frame="world", current_eef_pose=pose(z=-.05),
                        current_joint_positions_m=np.array([.02, -.02]),
                        contact_tolerance_m=.001, penetration_tolerance_m=.00025)
        defaults.update(options)
        return screen_grasp_candidates({"one": trajectory()} if candidates is None else candidates,
                                       {"left": view(-1), "right": view(1)} if views is None else views,
                                       Tool() if tool is None else tool, **defaults)

    def test_bilateral_observed_contact_keeps_target_and_trajectory_unchanged(self):
        candidates = {"one": trajectory()}
        original = deepcopy(candidates)
        result = self.run_check(candidates)
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_candidate_id"], "one")
        row = result["candidates"]["one"]
        self.assertEqual(row["terminal_closure"]["status"], "supported")
        self.assertAlmostEqual(row["terminal_closure"]["signed_gap_m"], .01)
        self.assertGreater(row["terminal_closure"]["conditional_closure_samples"], 2)
        self.assertFalse(row["moving_closure_suffix_verified"])
        self.assertFalse(row["actual_holding_verified"])
        np.testing.assert_array_equal(candidates["one"]["eef_poses"], original["one"]["eef_poses"])

    # BEGIN thin measured surfaces keep positions without inventing normals.
    def test_two_pixel_surface_uses_one_sided_neighbors_across_background_edge(self):
        views = {"left": view(-1), "right": view(1)}
        for row in views.values():
            row["owner_mask"][:] = False
            row["owner_mask"][:, 9:11] = True
            row["depth"][~row["owner_mask"]] += .1
        result = self.run_check(views=views)
        self.assertEqual(result["status"], "selected")
        for row in result["observations"].values():
            self.assertEqual(row["measured_owner_samples"], 42)
            self.assertEqual(row["observable_normal_samples"], 42)
            self.assertEqual(row["locally_supported_position_samples"], 0)
        closure = result["candidates"]["one"]["terminal_closure"]
        self.assertTrue(all(s["positive_contact_samples"] > 0 for s in closure["sides"]))

    def test_single_pixel_line_keeps_position_but_cannot_supply_surface_normal(self):
        views = {"left": view(-1), "right": view(1)}
        for row in views.values():
            row["owner_mask"][:] = False
            row["owner_mask"][:, 10] = True
            row["depth"][~row["owner_mask"]] += .1
        result = self.run_check(views=views)
        self.assertEqual(result["status"], "unresolved")
        for row in result["observations"].values():
            self.assertEqual(row["measured_owner_samples"], 21)
            self.assertEqual(row["observable_normal_samples"], 0)
        self.assertEqual(result["rejected_candidate_ids"], [])

    def test_isolated_boundary_position_inside_finger_is_unknown_with_witness(self):
        weak = view(-1, .023)
        weak["owner_mask"][:] = False
        weak["owner_mask"][10, 10] = True
        result = self.run_check(views={"left": view(-1), "right": view(1), "weak": weak})
        row = result["candidates"]["one"]
        self.assertEqual(row["status"], "unresolved")
        self.assertEqual(row["terminal_closure"]["status"], "supported")
        self.assertEqual(row["terminal_visible_geometry_status"], "unresolved")
        self.assertEqual(result["observations"]["weak"]["measured_owner_samples"], 1)
        solids = [s for record in row["interference"] for s in record["solids"]]
        self.assertTrue(solids)
        self.assertTrue(all(s["evidence_status"] == "unresolved" for s in solids))
        self.assertTrue(all(not s["witness"]["locally_supported_position"] for s in solids))
        self.assertTrue(all(not s["witness"]["observable_normal"] for s in solids))

    def test_first_weak_approach_hit_does_not_hide_later_supported_collision(self):
        weak = view(-1, .023, shift_z=-.02)
        weak["owner_mask"][:] = False
        weak["owner_mask"][10, 10] = True
        result = self.run_check(views={"left": view(-1), "right": view(1),
                                      "weak": weak, "solid": view(-1, .023)})
        row = result["candidates"]["one"]
        self.assertEqual(row["status"], "rejected")
        records = [r for r in row["interference"] if r["phase"] == "planned_open_approach"]
        self.assertGreaterEqual(len(records), 2)
        self.assertTrue(all(s["evidence_status"] == "unresolved" for s in records[0]["solids"]))
        self.assertTrue(any(s["evidence_status"] == "rejected" for s in records[-1]["solids"]))
    # END local evidence tests, independent of experiment candidates or success.

    def test_missing_opposite_side_and_missing_part_are_unknown(self):
        for views in ({"one": view(-1)}, {"left": view(-1), "right": view(1)}):
            if "right" in views:
                views["right"]["interaction_mask"] = np.zeros((21, 21), bool)
            result = self.run_check(views=views)
            self.assertEqual(result["status"], "unresolved")
            self.assertEqual(result["rejected_candidate_ids"], [])

    def test_non_intended_finger_interference_rejects(self):
        views = {"left": view(-1), "right": view(1), "interfering": view(-1, .023)}
        result = self.run_check(views=views)
        self.assertEqual(result["candidates"]["one"]["status"], "rejected")
        self.assertTrue(result["candidates"]["one"]["interference"])

    def test_intermediate_closure_collision_is_checked(self):
        class ExtraFingerTool(Tool):
            def __init__(self):
                super().__init__()
                self.collision_geometries.append(box("extra", "finger", [.001, .002, .002]))

            def at(self, q):
                state = super().at(q)
                state["geoms"].append({"geom_id": "extra", "T_tool_geom": pose(x=-q[0]-.003, y=.03)})
                return state

        views = {"left": view(-1), "right": view(1), "protrusion": view(-1, .013, shift_y=.03)}
        row = self.run_check(views=views, tool=ExtraFingerTool())["candidates"]["one"]
        self.assertEqual(row["status"], "rejected")
        self.assertFalse(row["approach_requires_replanning"])
        conflicts = [r for r in row["interference"] if r["phase"] == "terminal_conditional_closure"]
        self.assertTrue(conflicts)
        self.assertGreater(conflicts[0]["closure_fraction"], 0.)
        self.assertLess(conflicts[0]["closure_fraction"], row["terminal_closure"]["closure_fraction"])
        self.assertEqual(conflicts[0]["solids"][0]["geom_id"], "extra")

    def test_approach_conflict_does_not_claim_terminal_grasp_impossible(self):
        candidate = trajectory()
        candidate["eef_poses"] = np.array([pose(z=-.05), pose(z=.03), pose(), pose()])
        candidate["frame_indices"] = [0, 1, 2, 3]
        candidate["gripper_events"][1]["frame_index"] = 2
        result = self.run_check({"one": candidate})["candidates"]["one"]
        self.assertTrue(result["approach_requires_replanning"])
        self.assertEqual(result["terminal_visible_geometry_status"], "supported")
        self.assertEqual(result["status"], "rejected")

    def test_single_sheet_does_not_become_two_sided_capture(self):
        result = self.run_check(views={"left": view(-1, 0.), "right": view(1, 0.)})
        self.assertEqual(result["status"], "unresolved")
        closure = result["candidates"]["one"]["terminal_closure"]
        self.assertFalse(closure["independent_opposing_witnesses"])

    # BEGIN contacts need observed local normals, not a finite contact area.
    def test_curved_surface_can_have_only_point_or_line_in_contact_band(self):
        y, x = np.indices((21, 21))
        for kind, expected_count in (("point", 1), ("line", 3)):
            with self.subTest(contact=kind):
                views = {"left": view(-1), "right": view(1)}
                curvature = (x-10)**2 + ((y-10)**2 if kind == "point" else 0)
                for row in views.values():
                    row["owner_mask"][:] = False
                    row["owner_mask"][9:12, 9:12] = True
                    row["K"][0, 0] = row["K"][1, 1] = 50.
                    row["depth"] += .0015*curvature
                result = self.run_check(views=views)
                self.assertEqual(result["status"], "selected")
                closure = result["candidates"]["one"]["terminal_closure"]
                self.assertEqual([s["positive_contact_samples"] for s in closure["sides"]],
                                 [expected_count, expected_count])
                self.assertGreater(closure["opposing_witness_separation_m"], 0.)
                self.assertTrue(all(w["observable_normal"] for w in closure["opposing_witness_pair"]))

    def test_off_center_zero_thickness_sheet_does_not_gain_thickness_from_tolerance(self):
        transform = pose(.3, -.1, .2)
        transform[:3, :3] = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
        for gauge in (np.eye(4), transform):
            with self.subTest(gauge_changed=not np.array_equal(gauge, np.eye(4))):
                candidate = trajectory()
                candidate["eef_poses"] = gauge @ candidate["eef_poses"]
                # Both cameras measure the same plane x=+0.3 mm. Its visible
                # normals face opposite cameras, and different sample IDs do
                # not imply distinct material sides or positive thickness.
                views = {"left": view(-1, -.0003), "right": view(1, .0003)}
                for row in views.values():
                    row["T_output_camera"] = gauge @ row["T_output_camera"]
                result = self.run_check({"one": candidate}, views,
                                        current_eef_pose=gauge @ pose(z=-.05))
                closure = result["candidates"]["one"]["terminal_closure"]
                self.assertEqual(result["status"], "unresolved")
                self.assertGreater(closure["signed_gap_m"], 0.)
                self.assertLessEqual(abs(closure["opposing_witness_separation_m"]), 1e-12)
                self.assertFalse(closure["independent_opposing_witnesses"])
    # END point/line support and geometric independence regressions.

    def test_zero_thickness_stays_unknown_under_changed_world_gauge(self):
        transform = pose(.3, -.1, .2)
        transform[:3, :3] = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
        candidate = trajectory()
        candidate["eef_poses"] = transform @ candidate["eef_poses"]
        views = {"left": view(-1, 0.), "right": view(1, 0.)}
        for row in views.values():
            row["T_output_camera"] = transform @ row["T_output_camera"]
        result = self.run_check({"one": candidate}, views, current_eef_pose=transform @ pose(z=-.05))
        self.assertEqual(result["status"], "unresolved")

    def test_closed_physical_contact_limit_is_gauge_invariant(self):
        transform = pose(.3, -.1, .2)
        transform[:3, :3] = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
        for extra, expected in ((.0012499, "selected"), (.00125, "selected"), (.0012501, "unresolved")):
            for gauge in (np.eye(4), transform):
                with self.subTest(extra=extra, gauge_changed=not np.array_equal(gauge, np.eye(4))):
                    candidate = trajectory()
                    candidate["eef_poses"] = gauge @ candidate["eef_poses"]
                    views = {"left": view(-1, .005), "right": view(1, .005+extra)}
                    for row in views.values():
                        row["T_output_camera"] = gauge @ row["T_output_camera"]
                    result = self.run_check({"one": candidate}, views, current_eef_pose=gauge @ pose(z=-.05))
                    self.assertEqual(result["status"], expected)

    def test_world_gauge_and_candidate_order_do_not_change_selection(self):
        transform = pose(.3, -.1, .2)
        transform[:3, :3] = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
        candidates = {"z": trajectory(), "a": trajectory()}
        for row in candidates.values():
            row["eef_poses"] = transform @ row["eef_poses"]
        views = {"left": view(-1), "right": view(1)}
        for row in views.values():
            row["T_output_camera"] = transform @ row["T_output_camera"]
        result = self.run_check(candidates, views, current_eef_pose=transform @ pose(z=-.05))
        self.assertEqual(result["status"], "ambiguous")
        self.assertEqual(result["eligible_candidate_ids"], ["a", "z"])
        self.assertIsNone(result["selected_candidate_id"])

    def test_native_feedback_and_metric_tolerances_are_required(self):
        for value in ([np.nan, 0.], [.021, 0.], [0.], None):
            with self.subTest(q=value), self.assertRaises((ValueError, TypeError)):
                self.run_check(current_joint_positions_m=value)
        for key in ("contact_tolerance_m", "penetration_tolerance_m"):
            for value in (0., -1., np.nan, True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.run_check(**{key: value})

    def test_missing_open_intention_does_not_invent_an_opening_action(self):
        candidate = trajectory()
        candidate["gripper_events"] = [{"frame_index": 1, "command": "close"}]
        result = self.run_check({"one": candidate}, current_joint_positions_m=[.005, -.005])
        self.assertEqual(result["status"], "unresolved")
        self.assertIn("No opening intention", result["candidates"]["one"]["reason"])

    def test_unknown_action_does_not_veto_supported_same_function_actions(self):
        other = trajectory()
        other["eef_poses"][-1] = pose(y=.05)
        for count, expected in ((1, "selected"), (2, "ambiguous")):
            with self.subTest(supported_actions=count):
                candidates = {"valid_"+str(i): trajectory() for i in range(count)}
                candidates["missing"] = other
                result = self.run_check(candidates)
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["eligible_candidate_ids"], ["valid_"+str(i) for i in range(count)])
                self.assertEqual(result["unresolved_candidate_ids"], ["missing"])
                self.assertEqual(result["selected_candidate_id"], "valid_0" if count == 1 else None)
                self.assertEqual(set(result["candidates"]), set(candidates))


if __name__ == "__main__":
    unittest.main()
# END calibrated synthetic evidence tests, independent of dated experiment files.
