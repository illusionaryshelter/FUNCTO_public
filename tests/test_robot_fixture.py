"""CPU proofs and counterexamples for native robot self-query certificates."""
# BEGIN native-solid certificates, source reloads and effective-query regressions.
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import h5py
from mpmath.ctx_iv import MPIntervalContext
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

sys.path.insert(0, str(Path(__file__).parent / "planning"))
from robot_fixture import (
    _collision_query_contract, _effective_ignored_pairs, _geom_cover, _hinge_separation_certificate,
    _interval_bounds, _interval_rotation, _fixed_installation_records,
    export_fixture, fixed_installation_proxy, validate_fixture_collision_contract,
)


def _scene(*, extra_parent="", extra_child="", joint_type="hinge", extra_scene=""):
    # Broad thin native boxes are separated axially while their single-sphere
    # covers overlap. Names and dimensions are independent of any robot/task.
    return f'''<mujoco><compiler angle="radian"/><option ccd_tolerance="0.000001"/>
      <asset/><worldbody><body name="anchor">
        <geom name="anchor_geom" type="box" size="1 1 .05"/>{extra_parent}
        <body name="rotor" pos="0 0 .4">
          <joint name="spin" type="{joint_type}" axis="0 0 1" limited="true" range="-1 1"/>
          <geom name="rotor_geom" type="box" size="1 1 .05"/>
          <site name="tool" pos="0 0 .1"/>{extra_child}
        </body></body></worldbody>{extra_scene}</mujoco>'''


def _model(**kwargs):
    return mujoco.MjModel.from_xml_string(_scene(**kwargs))


def _config(links=("anchor", "rotor")):
    return {"collision_link_names": list(links),
            "collision_spheres": {name: [{"center": [0., 0., 0.], "radius": 2.}]
                                  for name in links},
            "collision_sphere_buffer": 0., "self_collision_buffer": {name: 0. for name in links}}


class NativeSeparationTest(unittest.TestCase):
    def certificate(self, model):
        return _hinge_separation_certificate(model, model.body("anchor").id, model.body("rotor").id)

    def test_separated_native_solids_complete_only_their_overlapping_sphere_query(self):
        model, config = _model(), _config()
        before = deepcopy(config)
        contract = _collision_query_contract(model, config)
        self.assertEqual(contract["native_policy_exclusions"], {})
        self.assertEqual(contract["native_enabled_sphere_pair_count"], 1)
        self.assertEqual(contract["effective_query_pairs"], [])
        self.assertEqual(contract["effective_query_sphere_pair_count"], 0)
        certificate, = contract["certificate_completed_pairs"]
        self.assertEqual(certificate["links"], ["anchor", "rotor"])
        self.assertGreater(certificate["separation_lower_bound_m"], .299999999999)
        self.assertGreater(certificate["certified_slack_lower_bound_m"], .299998999999)
        self.assertTrue(certificate["native_pair_remains_enabled"])
        self.assertEqual(config, before)  # Environmental sphere geometry is retained.
        self.assertEqual(_effective_ignored_pairs(contract), {"anchor": ["rotor"]})

    def test_arbitrary_axis_and_body_geom_frames_enclose_native_extrema(self):
        model = _model()
        parent, child = model.body("anchor").id, model.body("rotor").id
        axis = np.array([1., -2., 3.]); axis /= np.linalg.norm(axis)
        rotation = Rotation.from_euler("xyz", [.4, -.2, .3])
        model.jnt_axis[0] = axis
        model.body_pos[child] = [.31, -.22, .17]
        model.body_quat[child] = rotation.as_quat()[[3, 0, 1, 2]]
        model.geom_size[:] = [.015, .025, .01]
        model.geom_pos[model.geom("anchor_geom").id] = model.body_pos[child]+rotation.apply(-.2*axis)
        model.geom_pos[model.geom("rotor_geom").id] = .2*axis
        model.geom_quat[:] = Rotation.from_euler("yz", [.17, .11]).as_quat()[[3, 0, 1, 2]]
        # These test-local pose edits invalidate compilation's same-frame
        # shortcut; require native FK to evaluate the supplied geom poses.
        model.geom_sameframe[:] = 0  # mjSAMEFRAME_NONE
        certificate = self.certificate(model)
        self.assertIsNotNone(certificate)
        data = mujoco.MjData(model)
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        # These independent native-FK checks test the frame algebra, not the
        # all-angle proof: the latter follows from invariant axial projection.
        for angle in (-3.1, -.4, 0., 2.3, 3.1):
            data.qpos[0] = angle
            mujoco.mj_kinematics(model, data)
            world_axis = data.xmat[parent].reshape(3, 3) @ rotation.apply(axis)
            origin = data.xpos[child]
            for row in certificate["all_enabled_geometries"]:
                geom = model.geom(row["geom"]).id
                vertices = corners*model.geom_size[geom]
                world = vertices @ data.geom_xmat[geom].reshape(3, 3).T+data.geom_xpos[geom]
                projection = (world-origin) @ world_axis
                lower, upper = row["axis_projection_interval_m"]
                self.assertGreaterEqual(projection.min(), lower-2e-15)
                self.assertLessEqual(projection.max(), upper+2e-15)

    def test_reverse_axis_order_is_supported_without_link_name_rules(self):
        model = _model()
        model.body_pos[model.body("rotor").id, 2] = -.4
        certificate = self.certificate(model)
        self.assertEqual(certificate["separation_direction"], "parent_above_child")
        self.assertGreater(certificate["separation_lower_bound_m"], .29)

    def test_every_enabled_geometry_participates_and_disabled_visuals_do_not(self):
        extra = '<geom name="second" type="box" size=".1 .1 .1" pos="0 0 -.4"{mask}/>'
        self.assertIsNone(self.certificate(_model(extra_child=extra.format(mask=""))))
        certificate = self.certificate(_model(extra_child=extra.format(mask=' contype="0" conaffinity="0"')))
        self.assertIsNotNone(certificate)
        self.assertEqual({row["geom"] for row in certificate["all_enabled_geometries"]},
                         {"anchor_geom", "rotor_geom"})

    def test_fixed_child_keeps_its_own_geometry_and_query(self):
        model = _model(extra_child='<body name="mount" pos="0 0 .3"><geom name="mount_geom" type="box" size=".1 .1 .1"/></body>')
        config = _config(("anchor", "rotor", "mount"))
        contract = _collision_query_contract(model, config)
        self.assertEqual(contract["native_policy_exclusions"], {"rotor": ["mount"]})
        self.assertEqual([row["links"] for row in contract["certificate_completed_pairs"]], [["anchor", "rotor"]])
        self.assertEqual(contract["effective_query_pairs"], [["anchor", "mount"]])
        self.assertEqual(contract["native_enabled_sphere_pair_count"], 2)
        self.assertEqual(contract["effective_query_sphere_pair_count"], 1)

    def test_installed_native_pair_builder_consumes_the_effective_query_set(self):
        import torch
        from curobo.types import DeviceCfg
        from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg
        model = _model(extra_child='<body name="mount" pos="0 0 .3"><geom name="mount_geom" type="box" size=".1 .1 .1"/></body>')
        config = _config(("anchor", "rotor", "mount"))
        contract = _collision_query_contract(model, config)
        # This is the same installed builder used by KinematicsLoader before
        # kernels run; it checks actual library semantics on bounded CPU data.
        native = SelfCollisionKinematicsCfg.create_from_link_pairs(
            collision_link_names=config["collision_link_names"],
            link_name_to_sphere_index={name: index for index, name in enumerate(config["collision_link_names"])},
            self_collision_link_pair_ignores=_effective_ignored_pairs(contract),
            self_collision_link_padding=deepcopy(config["self_collision_buffer"]),
            all_link_spheres=torch.tensor([[0., 0., 0., 2.]]*3),
            link_index_to_sphere_index=torch.tensor([0, 1, 2], dtype=torch.int16),
            device_cfg=DeviceCfg(torch.device("cpu")))
        self.assertEqual(native.collision_pairs.tolist(), [[0, 2]])
        self.assertEqual(native.collision_pairs.device.type, "cpu")
        self.assertFalse(torch.cuda.is_initialized())

    def test_nonhinge_nonorigin_and_multijoint_bodies_keep_queries(self):
        self.assertIsNone(self.certificate(_model(joint_type="slide")))
        model = _model(); model.jnt_pos[0, 0] = .01
        self.assertIsNone(self.certificate(model))
        model = _model(extra_child='<joint name="other" type="hinge" axis="1 0 0" limited="true" range="-1 1"/>')
        self.assertIsNone(self.certificate(model))

    def test_unsupported_geometry_cannot_complete_a_query(self):
        xml = _scene().replace('name="rotor_geom" type="box" size="1 1 .05"',
                               'name="rotor_geom" type="sphere" size=".05"')
        self.assertIsNone(self.certificate(mujoco.MjModel.from_xml_string(xml)))

    def test_mesh_uses_every_compiled_vertex_including_interior_points(self):
        vertices = "-.02 -.02 -.02  .02 -.02 -.02  0 .02 -.02  0 0 .02  0 0 0"
        xml = _scene().replace('<asset/>', f'<asset><mesh name="shape" vertex="{vertices}"/></asset>')
        xml = xml.replace('name="rotor_geom" type="box" size="1 1 .05"',
                          'name="rotor_geom" type="mesh" mesh="shape"')
        model = mujoco.MjModel.from_xml_string(xml)
        certificate = self.certificate(model)
        record = next(row for row in certificate["all_enabled_geometries"] if row["geom"] == "rotor_geom")
        self.assertEqual(record["representation"], "native_mesh_convex_collision_hull")
        self.assertEqual(record["vertex_count"], int(model.mesh_vertnum[0]))
        self.assertGreater(record["vertex_count"], 4)

    def test_search_margin_is_not_replaced_by_the_smaller_margin_minus_gap(self):
        model = _model(); model.geom_margin[:] = [.04, .07]; model.geom_gap[:] = [.02, .06]
        certificate = self.certificate(model)
        self.assertAlmostEqual(certificate["required_distance_upper_bound_m"], .070001)
        self.assertEqual(certificate["native_geom_pair_parameters"][0]["effective_gap_m"], .06)
        model.geom_margin[0] = .4; model.geom_gap[0] = .4
        self.assertIsNone(self.certificate(model))

    def test_override_gap_and_native_tolerance_are_complete_distance_conditions(self):
        model = _model()
        model.opt.o_margin = .4
        self.assertIsNotNone(self.certificate(model))  # Override is still disabled.
        model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
        self.assertIsNone(self.certificate(model))
        model = _model(); model.geom_gap[:] = -.4
        self.assertIsNone(self.certificate(model))
        model = _model(); model.opt.ccd_tolerance = .3
        self.assertIsNone(self.certificate(model))

    def test_touching_numerically_uncertain_and_within_tolerance_gaps_are_not_certified(self):
        for separation in (.1, .1000005):
            model = _model(); model.body_pos[model.body("rotor").id, 2] = separation
            self.assertIsNone(self.certificate(model))
        model = _model(); model.opt.ccd_tolerance = 0
        model.body_pos[model.body("rotor").id, 2] = np.nextafter(.1, np.inf)
        self.assertIsNone(self.certificate(model))

    def test_nonfinite_geometry_margins_and_axis_fail_closed(self):
        for field in ("geom_pos", "geom_margin", "jnt_axis"):
            model = _model()
            getattr(model, field).flat[0] = np.nan
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.certificate(model)

    def test_explicit_extra_buffers_retain_the_original_queries(self):
        for global_buffer, local_buffer in ((.001, 0.), (0., .001)):
            config = _config()
            config["collision_sphere_buffer"] = global_buffer
            config["self_collision_buffer"]["anchor"] = local_buffer
            contract = _collision_query_contract(_model(), config)
            self.assertEqual(contract["certificate_completed_pairs"], [])
            self.assertEqual(contract["effective_query_pairs"], [["anchor", "rotor"]])

    def test_native_mask_policy_is_not_relabelled_as_a_geometric_certificate(self):
        model = _model(); model.geom_contype[:] = 0; model.geom_conaffinity[:] = 0
        contract = _collision_query_contract(model, _config())
        self.assertEqual(contract["native_policy_exclusions"], {"anchor": ["rotor"]})
        self.assertEqual(contract["certificate_completed_pairs"], [])
        self.assertEqual(contract["native_enabled_sphere_pair_count"], 0)

    def test_explicit_pair_rules_require_a_matching_exporter(self):
        model = _model(extra_scene='<contact><pair geom1="anchor_geom" geom2="rotor_geom"/></contact>')
        with self.assertRaisesRegex(ValueError, "without extensions"):
            _collision_query_contract(model, _config())

    def test_large_native_geom_norm_residual_cannot_be_normalized_away(self):
        model = _model(); model.opt.ccd_tolerance = 0
        parent_geom, child_geom = model.geom("anchor_geom").id, model.geom("rotor_geom").id
        scale = float(2**40)
        model.geom_size[parent_geom] = [scale, scale, scale]
        model.geom_size[child_geom] = [1., 1., 1.]
        model.body_pos[model.body("rotor").id, 2] = scale+1.+2**-11
        self.assertIsNotNone(self.certificate(model))
        # The stored quaternion is almost unit, but its native homogeneous
        # matrix expands this huge geom enough to close the real axial gap.
        model.geom_quat[parent_geom, 0] = 1.+2**-51
        self.assertIsNone(self.certificate(model))
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, model.geom_quat[parent_geom])
        self.assertGreater(matrix[8]*scale,
                           model.body_pos[model.body("rotor").id, 2]-1.)

    def test_sphere_cover_uses_the_same_raw_native_geom_matrix(self):
        model = _model()
        geom = model.geom("anchor_geom").id
        model.geom_size[geom] = [2.**40, 2.**40, 2.**40]
        model.geom_quat[geom, 0] = 1.+2.**-51
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"cover.npz"
            _, record, geometry = _geom_cover(model, geom, {"subdivisions": [1, 1, 1]}, path)
            native_matrix = np.empty(9)
            mujoco.mju_quat2Mat(native_matrix, model.geom_quat[geom])
            expected = geometry["vertices"] @ native_matrix.reshape(3, 3).T+model.geom_pos[geom]
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["cell_vertices_body"], expected)
                np.testing.assert_array_equal(saved["T_body_geom"][:3, :3], native_matrix.reshape(3, 3))
            self.assertGreaterEqual(record["minimum_cell_enclosure_margin_m"], 1e-6)

    def test_interval_endpoints_enclose_native_polynomial_and_do_not_mutate_global_precision(self):
        from mpmath import iv
        original = iv.prec
        context = MPIntervalContext(); context.prec = 113
        quaternion = np.array([1.+2**-51, 0., 0., 0.])
        matrix = _interval_rotation(context, quaternion, normalize=False)
        native = np.empty(9); mujoco.mju_quat2Mat(native, quaternion)
        for value, actual in zip(sum(matrix, []), native):
            lower, upper = _interval_bounds(value)
            self.assertLessEqual(lower, actual)
            self.assertGreaterEqual(upper, actual)
        self.assertEqual(iv.prec, original)


class FixtureReloadTest(unittest.TestCase):
    def test_source_export_reload_and_tamper_checks_allow_new_joint_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Include a fixed descendant outside the certified pair: its FK
            # and its buffer must still bind to the ORIGINAL source model.
            mount = '<body name="mount" pos=".2 .3 .4"><geom name="mount_geom" type="box" size=".01 .01 .01"/></body>'
            source = root / "scene.xml"; source.write_text(_scene(extra_child=mount))
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"joints": [{"name": "spin", "qpos_count": 1,
                                "qvel_count": 1, "qpos_start": 0, "qvel_start": 0}]}))
            episode = root / "episode.h5"
            with h5py.File(episode, "w") as stream:
                stream.create_dataset("states/qpos", data=np.array([[-.8], [0.], [.8]]))
            arguments = dict(scene_path=source, catalog_path=catalog, episode_path=episode,
                             root_body="anchor", collision_root="anchor", tcp_site="tool",
                             arm_joint_names=["spin"], finger_joint_names=[],
                             velocity_limits={"spin": 1.}, effort_limits={"spin": 2.},
                             cover_specifications={name: {"subdivisions": [1, 1, 1]}
                                                   for name in ("anchor_geom", "rotor_geom", "mount_geom")},
                             initial_frame=0, output=root/"fixture", binding_audit_path=catalog,
                             max_acceleration=2., max_jerk=10.)
            export_fixture(**arguments)
            fixture = json.loads((root/"fixture/manifest.json").read_text())
            config = yaml.safe_load((root/"fixture/robot.yml").read_text())
            report = validate_fixture_collision_contract(fixture, config)
            self.assertEqual(report["complete_native_geometry_count"], 3)
            self.assertEqual(report["certificate_completed_pairs"], [["anchor", "rotor"]])
            self.assertEqual(report["effective_query_sphere_pair_count"], 1)
            changed_pose = deepcopy(config)
            changed_pose["robot_cfg"]["kinematics"]["lock_joints"] = {"spin": .3}
            changed_pose["robot_cfg"]["kinematics"]["cspace"]["default_joint_position"] = [.3]
            self.assertEqual(validate_fixture_collision_contract(fixture, changed_pose), report)
            evidence_path = Path(fixture["collision_geometries"][0]["certificate_path"])
            with np.load(evidence_path, allow_pickle=False) as saved:
                original_evidence = {key: saved[key].copy() for key in saved.files}
            wrong_evidence = {key: value.copy() for key, value in original_evidence.items()}
            wrong_evidence["cell_vertices_body"] *= .1
            wrong_evidence["sphere_radii"] *= .2
            wrong_config = deepcopy(config)
            wrong_config["robot_cfg"]["kinematics"]["collision_spheres"]["anchor"][0]["radius"] = float(wrong_evidence["sphere_radii"][0])
            np.savez_compressed(evidence_path, **wrong_evidence)
            with self.assertRaisesRegex(ValueError, "complete native-solid partition"):
                validate_fixture_collision_contract(fixture, wrong_config)
            # A missing corner also cannot be hidden behind a still-enclosing
            # ball and an otherwise unchanged array of original native vertices.
            wrong_evidence = {key: value.copy() for key, value in original_evidence.items()}
            wrong_evidence["cell_vertices_body"] = wrong_evidence["cell_vertices_body"][:-1]
            wrong_evidence["cell_vertex_offsets"][-1] -= 1
            np.savez_compressed(evidence_path, **wrong_evidence)
            with self.assertRaisesRegex(ValueError, "complete native-solid partition"):
                validate_fixture_collision_contract(fixture, config)
            np.savez_compressed(evidence_path, **original_evidence)
            for alteration in ("sphere", "ignore", "buffer", "uncertified_negative_buffer", "certificate",
                               "extra_links", "extra_spheres", "joint_offset", "cspace_override", "urdf_path"):
                wrong_config, wrong_fixture = deepcopy(config), deepcopy(fixture)
                kin = wrong_config["robot_cfg"]["kinematics"]
                if alteration == "sphere":
                    kin["collision_spheres"]["anchor"][0]["radius"] *= .9
                elif alteration == "ignore":
                    kin["self_collision_ignore"] = {}
                elif alteration == "buffer":
                    kin["collision_sphere_buffer"] = .001
                elif alteration == "uncertified_negative_buffer":
                    kin["self_collision_buffer"]["mount"] = -.001
                elif alteration == "certificate":
                    wrong_fixture["certificate_completed_pairs"][0]["joint_axis_child"] = [1., 0., 0.]
                elif alteration == "extra_links":
                    kin["extra_links"] = {"mount": {"fixed_transform": np.eye(4).tolist()}}
                elif alteration == "extra_spheres":
                    kin["extra_collision_spheres"] = {"mount": 0}
                elif alteration == "joint_offset":
                    kin["joint_offset"] = [1., .1]
                elif alteration == "cspace_override":
                    kin["cspace"]["position_limit_clip"] = .1
                else:
                    alternate = root/"alternate.urdf"
                    alternate.write_text(Path(fixture["urdf_path"]).read_text())
                    kin["urdf_path"] = str(alternate)
                with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                    validate_fixture_collision_contract(wrong_fixture, wrong_config)
            urdf_path = Path(fixture["urdf_path"])
            original_urdf = urdf_path.read_text()
            robot_path = Path(fixture["robot_mjcf_path"])
            original_robot = robot_path.read_text()
            for alteration in ("hinge_axis", "hinge_origin", "tcp_origin", "fixed_origin", "joint_type",
                               "mimic", "joint_parent", "robot_and_urdf_fixed_origin"):
                xml = ET.fromstring(original_urdf)
                spin = xml.find("joint[@name='spin']")
                if alteration == "hinge_axis":
                    spin.find("axis").set("xyz", "1 0 0")
                elif alteration == "hinge_origin":
                    spin.find("origin").set("xyz", "0 0 .2")
                elif alteration == "tcp_origin":
                    xml.find("joint[@name='tool_fixed']/origin").set("xyz", "0 0 .2")
                elif alteration in ("fixed_origin", "robot_and_urdf_fixed_origin"):
                    xml.find("joint[@name='mount_fixed']/origin").set("xyz", ".2 .3 .45")
                    if alteration == "robot_and_urdf_fixed_origin":
                        native_xml = ET.fromstring(original_robot)
                        native_xml.find(".//body[@name='mount']").set("pos", ".2 .3 .45")
                        robot_path.write_text(ET.tostring(native_xml, encoding="unicode"))
                elif alteration == "joint_type":
                    spin.set("type", "prismatic")
                elif alteration == "mimic":
                    ET.SubElement(spin, "mimic", joint="spin", multiplier="1", offset=".1")
                else:
                    xml.find("joint[@name='mount_fixed']/parent").set("link", "anchor")
                urdf_path.write_text(ET.tostring(xml, encoding="unicode"))
                with self.subTest(alteration=alteration), self.assertRaisesRegex(ValueError, "original native FK"):
                    validate_fixture_collision_contract(fixture, config)
                urdf_path.write_text(original_urdf)
                robot_path.write_text(original_robot)
            original = source.read_text()
            for model_path, original_xml in ((source, original), (robot_path, original_robot)):
                xml = ET.fromstring(original_xml)
                xml.find(".//joint[@name='spin']").set("ref", ".1")
                model_path.write_text(ET.tostring(xml, encoding="unicode"))
                with self.subTest(native_ref_source=model_path.name), self.assertRaisesRegex(ValueError, "zero ref"):
                    validate_fixture_collision_contract(fixture, config)
                model_path.write_text(original_xml)
            source.write_text(original.replace('size="1 1 .05"', 'size="1 1 .06"', 1))
            with self.assertRaisesRegex(ValueError, "geometry or pose changed"):
                validate_fixture_collision_contract(fixture, config)
            source.write_text(_scene(extra_parent='<geom name="new_native_geom" type="box" size=".01 .01 .01"/>'))
            with self.assertRaisesRegex(ValueError, "every original enabled"):
                validate_fixture_collision_contract(fixture, config)
            with self.assertRaises(FileExistsError):
                export_fixture(**arguments)


# BEGIN full-assembly ownership and native primitive regression cases.
class FixedInstallationTest(unittest.TestCase):
    def test_fixed_primitives_survive_export_without_becoming_robot_spheres(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = '''<mujoco><compiler angle="radian"/><asset/><worldbody>
              <body name="base" pos=".2 -.1 .8">
                <body name="arm_root"><geom name="arm_geom" type="box" size=".1 .1 .05"/>
                  <body name="rotor" pos="0 0 .3"><joint name="spin" type="hinge" axis="0 0 1" limited="true" range="-1 1"/>
                    <geom name="rotor_geom" type="box" size=".1 .1 .05"/><site name="tool" pos="0 0 .1"/>
                  </body>
                </body>
                <body name="installation" pos="0 0 -.2">
                  <geom name="column" type="cylinder" size=".18 .31"/>
                  <geom name="foot" type="box" pos=".1 0 -.35" size=".2 .25 .05"/>
                  <geom name="visual_only" type="sphere" size=".2" contype="0" conaffinity="0"/>
                </body>
              </body></worldbody></mujoco>'''
            source = root/"scene.xml"; source.write_text(xml)
            catalog = root/"catalog.json"
            catalog.write_text(json.dumps({"joints": [{"name": "spin", "qpos_count": 1,
                "qvel_count": 1, "qpos_start": 0, "qvel_start": 0}]}))
            episode = root/"episode.h5"
            with h5py.File(episode, "w") as stream:
                stream.create_dataset("states/qpos", data=np.array([[-.2], [.2]]))
            fixture = export_fixture(
                scene_path=source, catalog_path=catalog, episode_path=episode,
                root_body="base", collision_root="arm_root", tcp_site="tool",
                arm_joint_names=["spin"], finger_joint_names=[], velocity_limits={"spin": 1.},
                effort_limits={"spin": 2.}, cover_specifications={
                    name: {"subdivisions": [1, 1, 1]} for name in ("arm_geom", "rotor_geom")},
                initial_frame=0, output=root/"fixture", binding_audit_path=catalog,
                max_acceleration=2., max_jerk=10.)
            config = yaml.safe_load(Path(fixture["robot_config_path"]).read_text())
            self.assertEqual(fixture["sphere_count"], 2)
            self.assertEqual(fixture["robot_visual_root"], "base")
            self.assertEqual(fixture["robot_collision_root"], "arm_root")
            rows = {row["geom"]: row for row in fixture["fixed_collision_geometries"]}
            self.assertEqual(set(rows), {"column", "foot"})
            self.assertEqual(rows["column"]["shape_kind"], "cylinder")
            self.assertEqual(rows["column"]["dimensions"], [.18, .62])
            self.assertEqual(rows["foot"]["dimensions"], [.4, .5, .1])
            # Base and arm_root are welded; a genuinely moving rotor remains
            # an enabled pair against the same fixed installation geometry.
            self.assertEqual(rows["column"]["robot_excluded_links"], ["arm_root"])
            self.assertAlmostEqual(rows["column"]["T_base_geom"][2][3], -.2)
            robot = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])
            self.assertEqual(robot.geom_type[robot.geom("column").id], mujoco.mjtGeom.mjGEOM_CYLINDER)
            urdf = ET.parse(fixture["urdf_path"])
            self.assertIsNotNone(urdf.find(".//collision[@name='column']/geometry/cylinder"))
            report = validate_fixture_collision_contract(fixture, config)
            self.assertEqual(report["complete_fixed_installation_geometry_count"], 2)
            for key, value in (("dimensions", [.17, .62]), ("robot_excluded_links", ["arm_root", "rotor"])):
                wrong = deepcopy(fixture)
                wrong["fixed_collision_geometries"][0][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Fixed installation native"):
                    validate_fixture_collision_contract(wrong, config)
            wrong = deepcopy(fixture); wrong["fixed_collision_geometries"].pop()
            with self.assertRaisesRegex(ValueError, "Every native fixed installation"):
                validate_fixture_collision_contract(wrong, config)
            wrong = deepcopy(fixture); wrong["robot_visual_root"] = "arm_root"
            with self.assertRaisesRegex(ValueError, "complete calibrated robot assembly"):
                validate_fixture_collision_contract(wrong, config)

    def test_unmodelled_moving_body_cannot_be_relabelled_as_fixed_installation(self):
        with self.assertRaisesRegex(ValueError, "rigidly fixed"):
            _fixed_installation_records(_model(), root_body="anchor",
                sphere_geom_names={"anchor_geom"}, sphere_links=["anchor"])

    def test_cylinder_proxy_encloses_native_side_and_end_caps(self):
        from scipy.spatial import ConvexHull
        prepared = fixed_installation_proxy({"geom": "column", "shape_kind": "cylinder",
                                             "dimensions": [.18, .62]})
        vertices = np.asarray(prepared["arguments"]["vertices"])
        planes = ConvexHull(vertices).equations
        # Half-step angles hit the critical side midpoints where an inscribed
        # visual mesh would exclude the true analytic cylinder surface.
        angles = (np.arange(128)+.5)*2*np.pi/128
        points = np.array([[.18*np.cos(angle), .18*np.sin(angle), z]
                           for angle in angles for z in (-.31, 0., .31)])
        self.assertLessEqual(float((points@planes[:, :3].T+planes[:, 3]).max()), 0.)
        self.assertEqual(prepared["approximation"]["sections"], 64)
        self.assertLess(prepared["approximation"]["maximum_radial_expansion_m"], .0003)
        self.assertGreater(prepared["approximation"]["minimum_axial_cover_allowance_m"], 0.)
# END fixed geometry retains native identities and does not consume a moving sphere budget.


if __name__ == "__main__":
    unittest.main()
# END native-solid certificates, source reloads and effective-query regressions.
