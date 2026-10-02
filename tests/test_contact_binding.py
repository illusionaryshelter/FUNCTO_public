"""CPU robot-only material binding; no renderer, physics or object state."""

# BEGIN named material/cover/FK regressions.
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

from manipulation.contact_binding import calibrate_box_material_features, build_material_contact_rules

sys.path.insert(0, str(Path(__file__).resolve().parent/"planning"))
from robot_fixture import _geom_cover, bind_fixture_contact_materials


_ROBOT = """<mujoco><worldbody><body name="base"><site name="tcp"/>
  <body name="left" pos="-.01 0 0"><joint name="left_q" type="slide" axis="-1 0 0" range="0 .04"/>
    <geom name="left_pad" type="box" size=".004 .008 .008" mass=".1"/></body>
  <body name="right" pos=".01 0 0"><joint name="right_q" type="slide" axis="-1 0 0" range="-.04 0"/>
    <geom name="right_pad" type="box" size=".004 .008 .008" mass=".1"/></body>
</body></worldbody></mujoco>"""
_FACES = [dict(material_id="left_inner", native_geometry="left_pad", face_axis=0, face_sign=1,
               source="explicit test face"),
          dict(material_id="right_inner", native_geometry="right_pad", face_axis=0, face_sign=-1,
               source="explicit opposite test face")]


class ContactBindingTests(unittest.TestCase):
    def test_native_face_has_full_polygon_and_affine_named_motion(self):
        model = mujoco.MjModel.from_xml_string(_ROBOT)
        result = calibrate_box_material_features(model, tcp_site="tcp",
            joint_positions={"left_q": .023, "right_q": -.017},
            finger_joint_names=("left_q", "right_q"), material_faces=_FACES)
        left, right = result["left_inner"], result["right_inner"]
        self.assertEqual(np.asarray(left["feature_points"]).shape, (4, 3))
        np.testing.assert_allclose(left["feature_normal"], [1., 0., 0.])
        np.testing.assert_allclose(right["feature_normal"], [-1., 0., 0.])
        mapping = np.asarray(left["feature_joint_translation"])
        np.testing.assert_allclose(mapping[:, 0, 0], -1.)
        np.testing.assert_allclose(mapping[:, :, 1], 0.)
        self.assertAlmostEqual(left["native_thickness_m"], .008)
        self.assertLess(left["affine_fk_max_error_m"], 1e-12)

    def test_runtime_permutation_and_exact_cover_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            xml = directory/"robot.xml"
            xml.write_text(_ROBOT)
            model = mujoco.MjModel.from_xml_path(str(xml))
            records, spheres = [], {}
            for face in _FACES:
                geom = model.geom(face["native_geometry"]).id
                balls, record, _ = _geom_cover(model, geom, {"subdivisions": [1, 2, 2]}, directory/f"{geom}.npz")
                records.append(record)
                spheres[record["link"]] = balls
            fixture = {"sphere_count": 8, "robot_mjcf_path": str(xml), "tcp_site": "tcp",
                       "collision_geometries": records}
            robot = {"robot_cfg": {"kinematics": {"collision_link_names": ["left", "right"], "collision_spheres": spheres}}}
            feedback = {"joint_names": ["right_q", "left_q"], "start_q": [-.02, .01],
                        "finger_joint_names": ["left_q", "right_q"]}
            ordering = {"left": [4, 5, 6, 7], "right": [0, 1, 2, 3]}
            binding = bind_fixture_contact_materials(fixture, robot, ordering, feedback, _FACES)
            self.assertEqual(binding["robot_sphere_materials"], ["right_inner"]*4+["left_inner"]*4)
            self.assertEqual(binding["materials"]["left_inner"]["sphere_indices"], [4, 5, 6, 7])
            self.assertEqual(binding["native_geometry_materials"], {"left_pad": "left_inner", "right_pad": "right_inner"})
            declaration = dict(rule_id="local", material_id="left_inner", target_owner="observed",
                interval=(0., 1.), patch_origin=[0., 0., 0.], patch_axes=np.eye(3), patch_half_size=[.01, .01],
                gap_bounds=(-.001, .001), normal_tolerance_rad=.01, modeling_allowance_m=.001,
                geometry_error_m=.001, minimum_thickness_m=.008, source="explicit observed test patch")
            rule, = build_material_contact_rules(binding, [declaration])
            self.assertEqual(asdict(rule)["sphere_indices"], (4, 5, 6, 7))
            spheres["left"][0]["radius"] += .001
            with self.assertRaisesRegex(ValueError, "sphere cover"):
                bind_fixture_contact_materials(fixture, robot, ordering, feedback, _FACES)

    def test_missing_feedback_and_undeclared_material_axis_fail(self):
        model = mujoco.MjModel.from_xml_string(_ROBOT)
        with self.assertRaisesRegex(ValueError, "every isolated robot"):
            calibrate_box_material_features(model, tcp_site="tcp", joint_positions={"left_q": .01},
                finger_joint_names=("left_q", "right_q"), material_faces=_FACES)
        with self.assertRaisesRegex(ValueError, "native box face"):
            calibrate_box_material_features(model, tcp_site="tcp", joint_positions={"left_q": .01, "right_q": -.01},
                finger_joint_names=("left_q", "right_q"), material_faces=[dict(_FACES[0], face_axis=3)])


if __name__ == "__main__":
    unittest.main()
# END binding does not synthesize contact evidence or alter collision pairs.
