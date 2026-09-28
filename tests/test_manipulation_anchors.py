"""CPU contracts for fixed anchors, independent frame IDs and body ownership.

Known local geometry and independently constructed rigid motions give exact
expected positions. These tests concern coordinate algebra, not visual pose
accuracy or whether an observed contact is physically fixed to an object.
"""

# BEGIN owned fixed-anchor tests.
from copy import deepcopy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from manipulation.anchors import bind_anchors, propagate_anchors


def pose(angles, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def point_at(matrix, point):
    return (matrix @ np.r_[point, 1.])[:3]


def record(name, owner, kind, frame_index, value):
    return {"name": name, "owner": owner, "kind": kind,
            "frame_index": frame_index, "value": value}


class OwnedAnchorTest(unittest.TestCase):
    def setUp(self):
        self.frames = np.array([5, 9, 14])
        self.poses = {
            "body": np.array([
                pose((.4, -.2, .3), (.6, -.3, 1.2)),
                pose((-.2, .5, .8), (.7, -.1, 1.3)),
                pose((.1, -.6, -.4), (.9, .2, 1.1)),
            ]),
            "reference": np.array([
                pose((-.4, .3, .6), (-.5, .1, 1.8)),
                pose((.5, -.3, .2), (-.4, .3, 1.9)),
                pose((.2, .1, -.6), (-.2, .2, 1.5)),
            ]),
        }

    def test_nonzero_observation_frame_binds_exact_material_point(self):
        local = np.array([.15, -.07, .09])
        observed = point_at(self.poses["body"][2], local)
        anchors = [record("contact", "body", "point", 14, observed)]
        bindings = bind_anchors(anchors, self.poses, self.frames)
        np.testing.assert_allclose(bindings[0]["local_value"], local, atol=1e-14)
        tracks = propagate_anchors(bindings, self.poses, self.frames)
        expected = [point_at(matrix, local) for matrix in self.poses["body"]]
        np.testing.assert_allclose(tracks["anchors"][0]["values"], expected, atol=1e-14)
        np.testing.assert_array_equal(tracks["frame_indices"], self.frames)
        self.assertEqual(tracks["anchors"][0]["frame_index"], 14)
        # Binding has its own local array and never overwrites the measurement.
        np.testing.assert_array_equal(anchors[0]["value"], observed)
        tracks["anchors"][0]["local_value"][:] = 0
        np.testing.assert_allclose(bindings[0]["local_value"], local, atol=1e-14)

    def test_direction_ignores_translation_and_preserves_supplied_length(self):
        direction = np.array([.2, -.5, 2.])
        observed = self.poses["body"][1, :3, :3] @ direction
        bindings = bind_anchors([record("normal", "body", "direction", 9, observed)],
                                self.poses, self.frames)
        result = propagate_anchors(bindings, self.poses, self.frames)["anchors"][0]["values"]
        expected = self.poses["body"][:, :3, :3] @ direction
        np.testing.assert_allclose(result, expected, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), np.linalg.norm(direction), atol=1e-14)
        moved = {owner: sequence.copy() for owner, sequence in self.poses.items()}
        moved["body"][:, :3, 3] += [7., -2., 3.]
        translated = propagate_anchors(bindings, moved, self.frames)["anchors"][0]["values"]
        np.testing.assert_array_equal(translated, result)

    def test_each_owner_has_independent_motion_and_same_owner_distances_stay_3d_rigid(self):
        local_points = {"body": np.array([.11, .02, -.09]),
                        "reference": np.array([-.13, .04, .07])}
        anchors = [record("center", owner, "point", 5, point_at(self.poses[owner][0], local))
                   for owner, local in local_points.items()]
        offset = np.array([.03, -.06, .17])
        anchors.append(record("edge", "body", "point", 9,
                              point_at(self.poses["body"][1], local_points["body"] + offset)))
        tracks = propagate_anchors(bind_anchors(anchors, self.poses, self.frames),
                                   self.poses, self.frames)["anchors"]
        for track in tracks[:2]:
            expected = [point_at(matrix, local_points[track["owner"]])
                        for matrix in self.poses[track["owner"]]]
            np.testing.assert_allclose(track["values"], expected, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(tracks[2]["values"] - tracks[0]["values"], axis=1),
                                   np.linalg.norm(offset), atol=1e-14)

    def test_reference_coordinates_and_body_gauge_do_not_change_physical_binding(self):
        local = np.array([.06, -.08, .13])
        observed = point_at(self.poses["body"][1], local)
        anchors = [record("site", "body", "point", 9, observed)]
        bindings = bind_anchors(anchors, self.poses, self.frames)
        expected = propagate_anchors(bindings, self.poses, self.frames)["anchors"][0]["values"]
        reference_change = pose((.7, -.4, .1), (-.5, 1.4, .2))
        changed_poses = {owner: reference_change @ sequence for owner, sequence in self.poses.items()}
        changed_anchor = [record("site", "body", "point", 9, point_at(reference_change, observed))]
        changed_bindings = bind_anchors(changed_anchor, changed_poses, self.frames)
        np.testing.assert_allclose(changed_bindings[0]["local_value"], local, atol=1e-14)
        changed = propagate_anchors(changed_bindings, changed_poses, self.frames)["anchors"][0]["values"]
        np.testing.assert_allclose(changed, [point_at(reference_change, value) for value in expected], atol=1e-14)
        # A different body origin/orientation is only a gauge change when the
        # supplied poses and the bound local coordinates change consistently.
        gauge = pose((-.3, .6, -.2), (.2, -.1, .07))
        gauged_poses = {"body": self.poses["body"] @ np.linalg.inv(gauge)}
        gauged = bind_anchors(anchors, gauged_poses, self.frames)
        np.testing.assert_allclose(gauged[0]["local_value"], point_at(gauge, local), atol=1e-14)
        result = propagate_anchors(gauged, gauged_poses, self.frames)["anchors"][0]["values"]
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_first_available_frame_and_later_subsequence_do_not_rebind(self):
        # The first available frame is 5, not zero; propagation can subsequently
        # start at 9 without retaining an observation at the old binding time.
        measured = np.array([.3, .5, 1.1])
        bindings = bind_anchors([record("site", "body", "point", 5, measured)], self.poses, self.frames)
        full = propagate_anchors(bindings, self.poses, self.frames)
        later = propagate_anchors(bindings, {"body": self.poses["body"][1:]}, self.frames[1:])
        np.testing.assert_allclose(full["anchors"][0]["values"][0], measured, atol=1e-14)
        np.testing.assert_array_equal(later["anchors"][0]["values"], full["anchors"][0]["values"][1:])
        self.assertEqual(later["anchors"][0]["frame_index"], 5)

    def test_zero_anchors_needs_no_artificial_owner_or_geometry(self):
        bindings = bind_anchors([], {}, self.frames)
        self.assertEqual(bindings, [])
        result = propagate_anchors(bindings, {}, self.frames)
        self.assertEqual(result["anchors"], [])
        np.testing.assert_array_equal(result["frame_indices"], self.frames)

    def test_unknown_owner_bad_anchor_metadata_and_invalid_geometry_are_rejected(self):
        valid = record("site", "body", "point", 9, [0., 0., 1.])
        changes = [
            {"name": ""}, {"owner": "absent"}, {"kind": "normal"},
            {"frame_index": True}, {"frame_index": 9.}, {"frame_index": -1},
            {"frame_index": 1}, {"value": [np.nan, 0., 1.]},
            {"value": [np.inf, 0., 1.]}, {"value": [[0., 0., 1.]]},
            {"kind": "direction", "value": [0., 0., 0.]},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                bind_anchors([{**valid, **change}], self.poses, self.frames)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            bind_anchors([valid, deepcopy(valid)], self.poses, self.frames)
        missing = {key: value for key, value in valid.items() if key != "owner"}
        with self.assertRaises(ValueError):
            bind_anchors([missing], self.poses, self.frames)
        with self.assertRaises(ValueError):
            bind_anchors([dict(valid, local_value=[0., 0., 0.])], self.poses, self.frames)

    def test_frame_ids_and_pose_validity_are_not_silently_repaired(self):
        invalid_frames = ([], [5, 5, 14], [14, 9, 5], [-1, 5, 9], [5., 9., 14.],
                          [False, True, True], [True, 9, 14], [[5, 9, 14]], [5, 9])
        for frames in invalid_frames:
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                bind_anchors([], self.poses, frames)
        bad_poses = [np.empty((0, 4, 4)), np.eye(4), np.repeat(np.eye(4)[None], 2, axis=0)]
        for element, value in (((3, 0), .1), ((0, 0), -1.), ((1, 1), 1.1), ((0, 3), np.nan)):
            matrix = np.repeat(np.eye(4)[None], 3, axis=0)
            matrix[(1, *element)] = value
            bad_poses.append(matrix)
        for prediction in bad_poses:
            with self.subTest(shape=prediction.shape), self.assertRaises(ValueError):
                bind_anchors([], {"body": prediction}, self.frames)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            bind_anchors([], {"": self.poses["body"]}, self.frames)

    def test_loaded_bindings_are_checked_before_propagation(self):
        binding = {"name": "site", "owner": "body", "kind": "point",
                   "frame_index": 0, "local_value": [.2, .1, -.1]}
        # Frame 0 need not occur: this is already a local binding, not another
        # request to reinterpret an observed world point at frame zero.
        self.assertEqual(len(propagate_anchors([binding], self.poses, self.frames)["anchors"]), 1)
        for change in ({"owner": "missing"}, {"local_value": [np.nan, 0., 1.]},
                       {"kind": "direction", "local_value": [0., 0., 0.]}, {"frame_index": -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                propagate_anchors([{**binding, **change}], self.poses, self.frames)


if __name__ == "__main__":
    unittest.main()
# END owned fixed-anchor tests.
