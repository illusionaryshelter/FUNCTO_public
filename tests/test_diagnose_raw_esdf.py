"""Check the native diagnostic on known signed fields and grid transforms."""
import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
import numpy as np
from planning.diagnose_raw_esdf import diagnose, read_bounded_json


def record(field, pose=None):
    field = np.asarray(field, dtype=np.float16)
    return {"name": "synthetic", "kind": "voxel", "shape_xyz": list(field.shape),
            "voxel_size": .01, "pose": pose or [0, 0, 0, 1, 0, 0, 0],
            "field_dtype": "<f2", "field_bytes_hex": field.tobytes().hex(),
            "occupancy_rule": "ESDF_sample_nonpositive",
            "occupied_flat_indices": np.flatnonzero(field.reshape(-1) <= 0).tolist()}


class TestArchivedNativeESDF(unittest.TestCase):
    def test_signed_centres_interpolation_and_activation(self):
        field = np.broadcast_to(np.array([-.01, 0., .01, .02])[:, None, None], (4, 3, 3))
        queries = [{"center_in_parent_frame_m": [x, 0, 0], "radius_m": .003}
                   for x in [-.015, -.005, .005, 0]]
        queries.append({"center_in_parent_frame_m": [.005, 0, 0], "radius_m": .003,
                        "activation_distance_m": .01})
        result = diagnose(record(field), queries)["queries"]
        expected = np.array([-.01, 0, .01, .005], dtype=np.float16).astype(float)
        np.testing.assert_allclose([r["raw_interpolated_esdf_m"] for r in result[:4]], expected, atol=2e-6)
        self.assertLess(result[0]["sphere_surface_clearance_m"], 0)
        self.assertGreater(result[2]["sphere_surface_clearance_m"], 0)
        self.assertEqual(result[2]["native_activated_collision_cost"], 0)
        self.assertGreater(result[4]["native_activated_collision_cost"], 0)
        self.assertTrue(all(not r["free_space_certified"] for r in result))

    def test_zero_band_gradient_and_unknown_sentinel(self):
        zero = diagnose(record(np.zeros((3, 3, 3))),
            [{"center_in_parent_frame_m": [0, 0, 0], "radius_m": .01}])["queries"][0]
        self.assertTrue(zero["eight_raw_corners_all_zero"])
        self.assertEqual(zero["raw_sampler_gradient_norm"], 0)
        self.assertAlmostEqual(zero["native_activated_collision_cost"], .01, places=7)
        unknown = diagnose(record(np.full((3, 3, 3), 10000.)),
            [{"center_in_parent_frame_m": [0, 0, 0], "radius_m": .01}])["queries"][0]
        self.assertEqual(unknown["raw_interpolated_esdf_m"], 10000.)
        self.assertEqual(unknown["native_activated_collision_cost"], 0)
        self.assertFalse(unknown["free_space_certified"])

    def test_pose_applied_once_and_boundary_disclosed(self):
        field = np.broadcast_to(np.array([-.01, 0., .01])[:, None, None], (3, 3, 3))
        # Rotate local +X onto world +Y and translate by [1,2,3].
        q = np.sqrt(.5)
        result = diagnose(record(field, [1, 2, 3, q, 0, 0, q]), [
            {"center_in_parent_frame_m": [1, 2.01, 3], "radius_m": 0},
            {"center_in_parent_frame_m": [10, 10, 10], "radius_m": 0}])["queries"]
        self.assertAlmostEqual(result[0]["raw_interpolated_esdf_m"], float(np.float16(.01)), places=6)
        self.assertFalse(result[1]["all_interpolation_corners_in_bounds"])
        self.assertEqual(result[1]["raw_interpolated_esdf_m"], 100.)
        self.assertFalse(result[1]["free_space_certified"])

    def test_rejects_malformed_shape_field_pose_and_parameters(self):
        base = record(np.zeros((3, 3, 3)))
        query = [{"center_in_parent_frame_m": [0, 0, 0], "radius_m": .01}]
        for key, value in [("shape_xyz", [2**62, 4, 4]), ("shape_xyz", [3., 3, 3]),
                           ("field_bytes_hex", "00"), ("voxel_size", 0.),
                           ("voxel_size", 1e-44),
                           ("voxel_size", float("nan")), ("pose", [0, 0, 0, 0, 0, 0, 0]),
                           ("occupied_flat_indices", []), ("occupancy_rule", "positive")]:
            with self.subTest(key=key, value=value):
                bad = deepcopy(base)
                bad[key] = value
                with self.assertRaises(ValueError):
                    diagnose(bad, query)
        for key, value in [("radius_m", -.1), ("activation_distance_m", -.1),
                           ("weight", float("nan")), ("radius_m", [.01]),
                           ("activation_distance_m", [.01]), ("weight", [1.]),
                           ("center_in_parent_frame_m", [0, float("inf"), 0]),
                           ("center_in_parent_frame_m", [1e30, 0, 0])]:
            with self.subTest(query_key=key):
                bad = deepcopy(query)
                bad[0][key] = value
                with self.assertRaises(ValueError):
                    diagnose(base, bad)

    def test_limits_input_bytes_before_json_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"too-large.json"
            path.write_bytes(b"not-json"*20)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                read_bounded_json(path, 16)


if __name__ == "__main__":
    unittest.main()
