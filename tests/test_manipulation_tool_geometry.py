"""CPU-only checks of actual tool assets, coordinates and convex collision data."""

# BEGIN native-tool geometry regressions, independent of scene/object state.
import unittest

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from manipulation.tool_geometry import _collision_geometry, load_robosuite_panda_tool


class ToolGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_robosuite_panda_tool(
            tool_id="known_parallel_jaw_tool", tool_frame="measured_control_site")

    def test_real_pad_faces_are_not_grip_site_or_finger_joint_positions(self):
        # These positions follow the authored Panda body/site/geom transforms:
        # .0524 + .056 - .015 - .097 = -.0036 m relative to grip_site.
        for q in ((0., 0.), (.04, -.04), (.011, -.029)):
            state = self.tool.at(q)
            left, right = [pad["T_tool_pad"] for pad in state["pads"]]
            np.testing.assert_allclose(left[:3, 3], [-q[0] + .0005, 0., -.0036],
                                       atol=1e-12, rtol=0)
            np.testing.assert_allclose(right[:3, 3], [-q[1] - .0005, 0., -.0036],
                                       atol=1e-12, rtol=0)
            np.testing.assert_allclose(left[:3, 2], [1., 0., 0.], atol=1e-12)
            np.testing.assert_allclose(right[:3, 2], [-1., 0., 0.], atol=1e-12)
            self.assertAlmostEqual(state["signed_gap_m"], q[0] - q[1] - .001, places=12)
            for pad in state["pads"]:
                np.testing.assert_allclose(pad["half_sizes_xy_m"], [.008, .008])
                rotation = pad["T_tool_pad"][:3, :3]
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation), 1., places=12)
        np.testing.assert_allclose(self.tool.signed_gap_range_m, [-.001, .079], atol=1e-12)

    def test_asymmetric_finger_motion_preserves_palm_and_only_moves_its_finger(self):
        before = self.tool.at([.01, -.02])
        after = self.tool.at([.025, -.02])
        by_name = {item["name"]: item["geom_id"] for item in self.tool.collision_geometries}
        first = {item["geom_id"]: item["T_tool_geom"] for item in before["geoms"]}
        second = {item["geom_id"]: item["T_tool_geom"] for item in after["geoms"]}
        for name, index in by_name.items():
            expected = np.array([-.015, 0., 0.]) if name.startswith("finger1") else np.zeros(3)
            np.testing.assert_allclose(second[index][:3, 3] - first[index][:3, 3],
                                       expected, atol=1e-12)
            np.testing.assert_allclose(second[index][:3, :3], first[index][:3, :3], atol=1e-12)
        np.testing.assert_allclose(self.tool.joint_axes_tool, [[-1., 0., 0.], [-1., 0., 0.]],
                                   atol=1e-12)

    def test_queries_require_explicit_valid_native_coordinates(self):
        for q in ([.02], [.02, .02], [-.001, -.01], [.041, -.01],
                  [np.nan, 0.], [True, False], ["0", "0"], None):
            with self.subTest(q=q), self.assertRaises(ValueError):
                self.tool.at(q)
        with self.assertRaises(TypeError):
            self.tool.at()

    def test_convex_planes_use_real_vertices_and_metric_normals(self):
        for geom in self.tool.collision_geometries:
            vertices, planes = geom["vertices"], geom["halfspaces"]
            np.testing.assert_allclose(np.linalg.norm(planes[:, :3], axis=1), 1., atol=1e-12)
            self.assertLessEqual(np.max(vertices @ planes[:, :3].T + planes[:, 3]), 1e-10)
            if geom["role"] != "pad":
                self.assertGreater(len(planes), 6)  # A hand/finger is not an AABB.
            center = vertices.mean(axis=0)
            self.assertLess(np.max(center @ planes[:, :3].T + planes[:, 3]), 0.)

    def test_compiled_mesh_recentering_rotation_and_scale_are_applied_once(self):
        # A deliberately off-centre, nonuniformly scaled tetrahedron exercises
        # compilation's mesh-frame shift/rotation independently of Panda. The
        # expected world points come directly from authored vertices and poses.
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        scale = np.array([.03, .02, .01])
        body_pos, geom_pos = np.array([.2, -.1, .4]), np.array([.03, .04, -.02])
        body_rotation = Rotation.from_euler("z", .7)
        geom_rotation = Rotation.from_euler("xy", [.3, -.5])
        def quat_wxyz(rotation):
            q = rotation.as_quat()
            return " ".join(map(str, q[[3, 0, 1, 2]]))
        xml = f'''<mujoco><asset><mesh name="shape"
          vertex="{' '.join(map(str, vertices.ravel()))}" scale="{' '.join(map(str, scale))}"/>
          </asset><worldbody><body pos="{' '.join(map(str, body_pos))}"
          quat="{quat_wxyz(body_rotation)}"><geom name="shape_geom" type="mesh" mesh="shape"
          pos="{' '.join(map(str, geom_pos))}" quat="{quat_wxyz(geom_rotation)}"/>
          </body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        geom = _collision_geometry(model, model.geom("shape_geom").id, "palm")
        observed = geom["vertices"] @ data.geom_xmat[0].reshape(3, 3).T + data.geom_xpos[0]
        expected = body_rotation.apply(geom_rotation.apply(vertices * scale) + geom_pos) + body_pos
        self.assertLess(np.max(cKDTree(expected).query(observed)[0]), 1e-8)
        self.assertLess(np.max(cKDTree(observed).query(expected)[0]), 1e-8)
        self.assertGreater(np.linalg.norm(model.mesh_pos[0]), .001)


if __name__ == "__main__":
    unittest.main()
# END native-tool geometry regressions, independent of scene/object state.
