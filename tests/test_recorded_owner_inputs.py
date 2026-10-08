"""CPU input contracts for independently observed carrier/reference owners."""

# BEGIN current snapshots, complete source faces and held intervals stay explicit.
from copy import deepcopy
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent/"planning"))
from manipulation.contact_binding import build_material_contact_rules
from recorded_owner_geometry import bind_box_owner_materials, prepare_box_owner
from run_recorded_skill import load_contact_bindings, recorded_owner_observation_inputs, require_feedback_identity


def held_problem():
    initial, attachment = np.eye(4), np.eye(4)
    initial[:3, 3] = [1., 2., 3.]
    attachment[0, 3] = .07
    metadata = dict(carrier_id="payload", owner_initial_pose=initial, mode="held",
        attachment=attachment, attachment_interval=[0., .8],
        attachment_condition_id="actual-holding", release_condition_id="actual-release")
    condition = dict(kind="observed_attachment", condition_id="actual-holding",
        carrier_id="payload", T_carrier_eef=attachment.tolist(), progress=0.)
    return SimpleNamespace(phases=[SimpleNamespace(metadata=metadata)], conditions=[condition])


def complete_owner(*, T_owner_source=None, box_quaternion=None):
    boxes = [dict(name="base", source_geom_id=1, position=[0., 0., 0.],
                  quaternion_wxyz=([1., 0., 0., 0.] if box_quaternion is None else box_quaternion),
                  halfsize=[.02, .03, .004]),
             dict(name="side", source_geom_id=2, position=[.02, 0., .02],
                  quaternion_wxyz=[1., 0., 0., 0.], halfsize=[.004, .03, .02])]
    # This source-to-owner transform is deliberately different from both
    # current world poses: native face binding must stay in the owner frame.
    mapping = np.eye(4)
    mapping[:3, 3] = [.1, -.2, .3]
    if T_owner_source is not None:
        mapping = T_owner_source
    prepared = prepare_box_owner(held_problem(), "payload", boxes=boxes,
        T_owner_source=mapping,
        geometry_source=dict(geometry_complete=True, enabled_geom_ids=[1, 2],
                             source_body_name="rigid_body", catalog_path="offline_source.json"),
        mapping_source="Explicit static source-body gauge mapping", cell_edge_m=.02, max_spheres=128)
    prepared["material_features"] = [dict(material_id="support_face", native_geometry="base",
        feature_kind="polygon", face_axis=2, face_sign=-1,
        source="Explicit complete native base box lower face")]
    return prepared


def contact_declaration():
    return dict(actor="payload", rule_id="finite-support", material_id="support_face",
        target_owner="reference", interval=(.6, .8), patch_origin=[0., 0., 0.],
        patch_axes=np.eye(3), patch_half_size=[.04, .06], gap_bounds=(-.0001, .0001),
        normal_tolerance_rad=.001, modeling_allowance_m=.0001, geometry_error_m=.0001,
        minimum_thickness_m=.008, source="Current finite reference-local observed support patch")


# BEGIN the actual recorded parser keeps initial support outside material binding.
def initial_support_declaration():
    return dict(relation_id="current-support", actor="payload", target_owner="initial_support",
        interval=[0., .8], plane_origin=[0., 0., 0.], plane_normal=[0., 0., 1.],
        source={"kind": "observed_plane_with_declared_initial_support",
            "observation_ids": ["current"], "assumption": "Current payload rests on this observed plane",
            "scope": "Entire source belongs to the independently observed plane; initial full-actor intersections only"})


def test_recorded_parser_preserves_independent_initial_support_and_material_declarations(tmp_path):
    declaration = initial_support_declaration()
    # An unrelated future material declaration passes through verbatim; the
    # initial support relation has no material label, gap or normal tolerance.
    inputs = dict(material_faces=[], contacts=[{"rule_id": "future-support"}],
        initial_support_relations=[declaration], source="Current explicit contact observations")
    path = tmp_path/"contacts.json"
    path.write_text(json.dumps(inputs))
    actual, (relation,) = load_contact_bindings(path)
    assert actual == inputs
    assert relation.relation_id == declaration["relation_id"]
    assert relation.actor == "payload" and relation.target_owner == "initial_support"
    assert tuple(relation.interval) == (0., .8)
    assert relation.source == declaration["source"]
    np.testing.assert_array_equal(relation.plane_normal, [0., 0., 1.])


@pytest.mark.parametrize("value", [None, {}, "support", ["support"]])
def test_recorded_parser_rejects_unstructured_initial_support(value, tmp_path):
    path = tmp_path/"contacts.json"
    path.write_text(json.dumps(dict(material_faces=[], contacts=[], source="Current observation",
        initial_support_relations=value)))
    with pytest.raises(ValueError, match="list of typed relation records"):
        load_contact_bindings(path)


def test_recorded_parser_cannot_supply_an_initial_penetration_allowance(tmp_path):
    declaration = dict(initial_support_declaration(), modeling_allowance_m=.007)
    path = tmp_path/"contacts.json"
    path.write_text(json.dumps(dict(material_faces=[], contacts=[], source="Current observation",
        initial_support_relations=[declaration])))
    with pytest.raises(TypeError):
        load_contact_bindings(path)
# END baseline height and source inventory must be computed by the scene.


def test_explicit_feedback_identity_survives_copied_file_and_missing_id_is_rejected():
    feedback = {"feedback_id": "capture-42:robot", "source": "capture.json"}
    assert require_feedback_identity(feedback) == "capture-42:robot"
    assert feedback["feedback_id"] != feedback["source"]
    for identity in (None, "", " "):
        with pytest.raises(ValueError, match="input adapter"):
            require_feedback_identity({"feedback_id": identity})


def test_every_reference_and_carrier_view_keeps_its_explicit_pose_and_mask():
    reference = np.eye(4)
    reference[1, 3] = -4.
    current = dict(carrier_id="payload", reference_id="reference", reference_pose=reference)
    rows = [[owner, view, f"{owner}-{view}.npy"]
            for owner in ("payload", "reference") for view in ("left", "right")]
    frames, masks = recorded_owner_observation_inputs(held_problem(), current,
        [{"id": "left"}, {"id": "right"}], rows, source="current-scene.json")
    np.testing.assert_array_equal(frames["payload"]["T_output_owner"][:3, 3], [1., 2., 3.])
    np.testing.assert_array_equal(frames["reference"]["T_output_owner"], reference)
    assert set(masks) == {(owner, view) for owner in frames for view in ("left", "right")}
    for bad in (rows[:-1], rows+rows[:1], [["left", "old-two-field.npy"]]):
        with pytest.raises(ValueError):
            recorded_owner_observation_inputs(held_problem(), current,
                [{"id": "left"}, {"id": "right"}], bad, source="current-scene.json")
    with pytest.raises(ValueError, match="current reference_pose"):
        recorded_owner_observation_inputs(held_problem(), {"carrier_id": "payload", "reference_id": "reference"},
            [{"id": "left"}], rows, source="current-scene.json")


# BEGIN static contact targets are observations even when absent from goals.
def test_additional_observed_owner_keeps_its_frame_and_requires_every_view_mask():
    support = np.eye(4)
    support[:3, 3] = [-.4, .2, .7]
    current = dict(carrier_id="payload", observation_owners=[{
        "owner_id": "initial_support", "T_output_owner": support,
        "source": "Current synchronized calibrated support observation"}])
    original = deepcopy(current)
    rows = [[owner, view, f"{owner}-{view}.npy"]
            for owner in ("payload", "initial_support") for view in ("left", "right")]
    views = [{"id": "left"}, {"id": "right"}]
    frames, masks = recorded_owner_observation_inputs(held_problem(), current,
        views, rows, source="current-scene.json")
    assert set(frames) == {"payload", "initial_support"}
    np.testing.assert_array_equal(frames["initial_support"]["T_output_owner"], support)
    np.testing.assert_array_equal(current["observation_owners"][0]["T_output_owner"],
                                  original["observation_owners"][0]["T_output_owner"])
    assert frames["initial_support"]["source"] == current["observation_owners"][0]["source"]
    assert set(masks) == {(owner, view) for owner in frames for view in ("left", "right")}
    with pytest.raises(ValueError, match="exactly one current owner mask"):
        recorded_owner_observation_inputs(held_problem(), current, views, rows[:-1], source="current-scene.json")


@pytest.mark.parametrize("identity", ["payload", "reference", "robot", "background", "observed_background", "", None])
def test_additional_owner_cannot_replace_an_existing_or_reserved_identity(identity):
    current = dict(carrier_id="payload", reference_id="reference", reference_pose=np.eye(4),
        observation_owners=[{"owner_id": identity, "T_output_owner": np.eye(4), "source": "actual capture"}])
    with pytest.raises(ValueError, match="distinct physical identities"):
        recorded_owner_observation_inputs(held_problem(), current, [{"id": "left"}], [], source="scene.json")


@pytest.mark.parametrize("changes,reason", [({"source": ""}, "distinct physical identities"),
    ({"T_output_owner": np.zeros((4, 4))}, "invalid homogeneous last row"),
    ({"geometry_complete": True}, "requires exactly")])
def test_additional_owner_requires_a_valid_current_frame_and_cannot_claim_hidden_geometry(changes, reason):
    record = {"owner_id": "extra", "T_output_owner": np.eye(4), "source": "actual capture", **changes}
    with pytest.raises(ValueError, match=reason):
        recorded_owner_observation_inputs(held_problem(),
            {"carrier_id": "payload", "observation_owners": [record]},
            [{"id": "left"}], [["payload", "left", "payload.npy"], ["extra", "left", "extra.npy"]],
            source="scene.json")
# END additional partitions use the existing strict per-owner mask contract.


def test_owner_material_is_the_real_planar_face_with_its_full_solid_cover():
    prepared = complete_owner()
    binding = bind_box_owner_materials(prepared)
    material = binding["materials"]["support_face"]
    points = np.asarray(material["feature_points"])
    assert points.shape == (4, 3)
    np.testing.assert_allclose(points.min(axis=0), [.08, -.23, .296])
    np.testing.assert_allclose(points.max(axis=0), [.12, -.17, .296])
    np.testing.assert_array_equal(material["feature_normal"], [0., 0., -1.])
    assert material["feature_joint_translation"] is None
    assert material["feature_kind"] == "polygon"
    assert material["native_thickness_m"] == .008
    begin, end = prepared["provenance"]["coverage_cells"][0]["sphere_range"]
    assert material["sphere_indices"] == list(range(begin, end))
    assert binding["sphere_materials"][begin:end] == ["support_face"]*(end-begin)
    assert all(label is None for label in binding["sphere_materials"][end:])
    assert binding["native_geometry_materials"] == {"payload/base": "support_face"}
    combined = dict(materials={}, owner_materials={"payload": binding["materials"]})
    rule, = build_material_contact_rules(combined, [contact_declaration()])
    assert rule.actor == "payload" and rule.target_owner == "reference"
    assert rule.interval == (.6, .8)
    np.testing.assert_allclose(rule.feature_points, points)
    # Bounds remain caller evidence. The binding never expands a patch to the
    # reference owner's extent or turns another payload solid into material.
    np.testing.assert_array_equal(rule.patch_half_size, [.04, .06])
    with pytest.raises(ValueError, match="bound actor"):
        build_material_contact_rules(combined, [dict(contact_declaration(), actor="different")])
    with pytest.raises(ValueError, match="native binding"):
        build_material_contact_rules(combined, [dict(contact_declaration(), feature_points=points)])


# BEGIN complete input geometry enters the production builder with all contact rules.
def test_complete_payload_enters_initial_scene_with_native_contacts_and_current_evidence():
    from unittest.mock import patch
    from curobo.perception import MapperCfg
    import torch
    from manipulation.online_geometry import FrozenESDF
    from manipulation.planning_observation import build_scene_from_rgbd
    from recorded_owner_geometry import create_box_owner
    from test_manipulation_planning_observation import view
    from observed_geometry_fixtures import point_source

    prepared = complete_owner()
    binding = bind_box_owner_materials(prepared)
    contacts = build_material_contact_rules(
        dict(materials={}, owner_materials={"payload": binding["materials"]}), [contact_declaration()])
    prepared["sphere_materials"] = binding["sphere_materials"]
    owner = create_box_owner(prepared, contact_voxel_size_m=.0015)
    record = view()
    identities = ("payload", "reference", "initial_support")
    masks = {identity: np.zeros(record["depth"].shape, bool) for identity in identities}
    for index, identity in enumerate(identities):
        masks[identity][45+index, 45+index] = True
    record["owner_masks"] = masks
    frames = {identity: {"T_output_owner": np.eye(4), "source": "Current calibrated observation"}
              for identity in identities}
    frames["payload"]["T_output_owner"] = prepared["initial_pose"]
    cfg = MapperCfg(extent_meters_xyz=(.2, .2, .2), voxel_size=.01,
        extent_esdf_meters_xyz=(.2, .2, .2), esdf_voxel_size=.01,
        grid_center=torch.tensor([0., 0., 1.]), image_height=96, image_width=96, device="cuda:0")
    mapped = {}
    def make(normalized, partition_masks, identity, config, inverse, *, query_center=None,
             max_host_rss_bytes=None):
        # Replace only device mapping. Production input checks, final scene
        # construction and native material/sphere registration remain active.
        assert not partition_masks["current"][None][45, 45]
        shape = np.ceil(np.asarray(config.extent_esdf_meters_xyz)/config.esdf_voxel_size).astype(int)
        center = config.grid_center.numpy() if query_center is None else np.asarray(query_center)
        query = FrozenESDF.from_source(point_source(),
            origin=center-(shape-1)*config.esdf_voxel_size/2, shape=tuple(shape),
            voxel_size=config.esdf_voxel_size,
            owner_ids=("observed_background" if identity is None else identity,),
            source="Explicit synthetic CPU field")
        mapped[identity] = query
        return query, {"shape_xyz": shape.tolist()}

    policy = {"conditional_regions": [], "outside_regions": "forbid"}
    with patch("manipulation.planning_observation._map_online_partition", side_effect=make), \
         patch("manipulation.planning_observation._require_mapping_memory"), \
         patch("torch.cuda.mem_get_info", side_effect=AssertionError("CPU contract touched CUDA")):
        result = build_scene_from_rgbd({"current": record}, frames, mapper_cfg=cfg,
            depth_error_m=.003, owner_voxel_sizes_m={"payload": .0015, "reference": .005, "initial_support": .01},
            reference_frame="world", robot_sphere_materials=(None,), robot_geometry_source="Native test robot",
            complete_owners=(owner,), contacts=contacts, native_geometry_materials=binding["native_geometry_materials"],
            has_gripper=False, unknown_policy=policy, max_mapping_bytes=1024**3)
    assert set(mapped) == {None, "reference", "initial_support"}
    assert result.owners[0] is owner and owner.query.contact_voxel_size_m == .0015
    assert result.owners[1].query is mapped["reference"]
    assert result.owners[2].query is mapped["initial_support"]
    assert result.background is mapped[None] and result.coverage.observation_ids == ("current",)
    assert result.contacts == contacts and result.native_geometry_materials == binding["native_geometry_materials"]
    assert result.robot_owner_permissions == () and result.unknown_policy == policy
    np.testing.assert_array_equal(owner.pose_path.poses[0], prepared["initial_pose"])
    assert owner.attachments[0].interval == tuple(prepared["attachments"][0]["interval"])
    assert owner.attachments[0].condition_id == "actual-holding"
    assert owner.attachments[0].release_condition_id == "actual-release"
# END no discarded observed payload field or temporary contact subset exists.


@pytest.mark.parametrize("change", ["partial", "nonface", "extra_points", "missing_cover", "overlap"])
def test_material_binding_rejects_partial_geometry_and_non_native_face_edits(change):
    prepared = deepcopy(complete_owner())
    if change == "partial":
        prepared["geometry_complete"] = False
    elif change == "nonface":
        prepared["material_features"][0]["face_axis"] = 3
    elif change == "extra_points":
        prepared["material_features"][0]["feature_points"] = [[0., 0., 0.]]*24
    elif change == "missing_cover":
        prepared["provenance"]["coverage_cells"].pop()
    else:
        prepared["provenance"]["coverage_cells"][1]["sphere_range"][0] = 0
    with pytest.raises(ValueError):
        bind_box_owner_materials(prepared)


# BEGIN nonidentity source/box rotations expose local-frame mixing errors.
def native_edge_owner():
    owner_source = np.eye(4)
    owner_source[:3, :3] = Rotation.from_euler("xyz", [.27, -.41, .63]).as_matrix()
    owner_source[:3, 3] = [.12, -.23, .34]
    source_box = Rotation.from_euler("xyz", [-.19, .32, -.47])
    prepared = complete_owner(T_owner_source=owner_source,
        box_quaternion=source_box.as_quat()[[3, 0, 1, 2]].tolist())
    rotation = owner_source[:3, :3] @ source_box.as_matrix()
    deviation = 2.5e-5
    hint = rotation @ np.array([np.sin(deviation), .6*np.cos(deviation), -.8*np.cos(deviation)])
    prepared["material_features"] = [dict(material_id="support_edge", native_geometry="base",
        feature_kind="edge", edge_axis=0, edge_signs=[1, -1], normal_hint=hint.tolist(),
        source="Whole native X edge at box-local positive Y and negative Z; hint is owner-local")]
    return prepared, owner_source, rotation, hint


def test_true_native_edge_retains_two_endpoints_and_projects_hint_in_owner_frame():
    prepared, owner_source, rotation, hint = native_edge_owner()
    original_hint = deepcopy(prepared["material_features"][0]["normal_hint"])
    binding = bind_box_owner_materials(prepared)
    feature = binding["materials"]["support_edge"]
    points = np.asarray(feature["feature_points"])
    corners_box = np.array([[-.02, .03, -.004], [.02, .03, -.004]])
    expected = corners_box @ rotation.T+owner_source[:3, 3]
    np.testing.assert_allclose(points, expected, rtol=0, atol=1e-15)
    assert points.shape == (2, 3) and feature["feature_kind"] == "edge"
    audit = feature["feature_geometry_audit"]
    np.testing.assert_allclose(audit["adjacent_normals_owner"], np.array([[0., 1., 0.], [0., 0., -1.]]) @ rotation.T,
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(feature["feature_normal"], rotation @ [0., .6, -.8], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(audit["normal_hint_owner"], hint)
    assert prepared["material_features"][0]["normal_hint"] == original_hint
    assert audit["normal_cone_projection_angle_rad"] == pytest.approx(2.5e-5, abs=1e-15)
    assert audit["normal_cone_projection_residual"] == pytest.approx(np.sin(2.5e-5), abs=1e-15)
    np.testing.assert_allclose(audit["normal_cone_nonnegative_coefficients"],
                               np.array([.6, .8])*np.cos(2.5e-5), rtol=0, atol=1e-15)
    assert audit["normal_hint_endpoint_height_difference_m"] == pytest.approx(.04*np.sin(2.5e-5), abs=1e-15)
    assert audit["native_vertex_roundtrip_max_error_m"] < 1e-15
    assert audit["support_plane_max_residual_m"] < 1e-15
    assert feature["native_thickness_m"] == .008
    assert "not an edge thickness" in audit["native_thickness_source"]
    begin, end = prepared["provenance"]["coverage_cells"][0]["sphere_range"]
    assert feature["sphere_indices"] == list(range(begin, end))
    assert binding["native_geometry_materials"] == {"payload/base": "support_edge"}


def test_edge_permission_copies_native_dimension_without_expanding_contact_tolerances():
    prepared, _, _, _ = native_edge_owner()
    native = bind_box_owner_materials(prepared)
    combined = dict(materials={}, owner_materials={"payload": native["materials"]})
    declaration = dict(contact_declaration(), material_id="support_edge", normal_tolerance_rad=1e-4)
    rule, = build_material_contact_rules(combined, [declaration])
    assert rule.feature_kind == "edge" and rule.feature_points.shape == (2, 3)
    assert rule.normal_tolerance_rad == 1e-4
    assert rule.gap_bounds == (-1e-4, 1e-4)
    assert rule.numerical_position_tolerance_m == 1e-6
    # BEGIN registered online geometry retains the same nonidentity owner map.
    model = rule.online_material
    assert model["kind"] == "native_box_edge_v1" and model["native_geometry"] == "base"
    assert model["sphere_indices"] == list(rule.sphere_indices)
    assert rule._native_edge_geometry["lateral_depth_ratio"] == pytest.approx(4/3)
    from itertools import product
    pose = np.asarray(model["T_actor_box"])
    vertices = np.asarray(list(product((-1., 1.), repeat=3)))*model["half_size_m"]
    np.testing.assert_allclose(rule._native_edge_geometry["vertices_actor"],
        vertices @ pose[:3, :3].T+pose[:3, 3], rtol=0, atol=1e-15)
    assert np.linalg.norm(pose[:3, 3]) > .1
    assert not np.allclose(pose[:3, :3], np.eye(3))
    # END source/owner translations and rotations are not silently identities.
    for changed in ({"feature_kind": "polygon"}, {"feature_points": [[0., 0., 0.]]*4},
                    {"feature_normal": [0., 0., -1.]}):
        with pytest.raises(ValueError, match="native binding"):
            build_material_contact_rules(combined, [{**declaration, **changed}])


# BEGIN actor confirmation can validate all boxes, including nonmaterial ones.
def test_complete_owner_serializes_exact_primitive_sphere_registration():
    from recorded_owner_geometry import create_box_owner
    prepared = complete_owner()
    native = bind_box_owner_materials(prepared)
    prepared["sphere_materials"] = native["sphere_materials"]
    owner = create_box_owner(prepared, contact_voxel_size_m=.001)
    primitives = owner.query.primitives
    assert len(primitives) == len(prepared["cuboids"])
    groups = {row["box_name"]: row for row in prepared["provenance"]["coverage_cells"]}
    for box, primitive in zip(prepared["cuboids"], primitives):
        group = groups[box["name"]]
        assert primitive["pose"] == box["pose"] and primitive["dimensions"] == box["dims"]
        assert primitive["native_binding"] == {"geometry_name": box["name"],
            "sphere_indices": list(range(*group["sphere_range"])),
            "subdivisions": group["subdivisions"], "source": prepared["source"]}
    assert owner.query.record()["primitives"] == list(primitives)
# END geometry provenance does not change the analytic primitive union.


@pytest.mark.parametrize("changes", [
    {"feature_kind": "point"}, {"edge_axis": 3}, {"edge_signs": [1]},
    {"edge_signs": [1, 0]}, {"normal_hint": [0., 0., -2.]},
    {"normal_hint": [float("nan"), 0., -1.]}, {"feature_points": [[0., 0., 0.], [0., 0., 1.]]},
])
def test_native_edge_rejects_unsupported_dimension_invalid_identity_or_fabricated_points(changes):
    prepared, _, _, _ = native_edge_owner()
    prepared["material_features"][0].update(changes)
    with pytest.raises(ValueError):
        bind_box_owner_materials(prepared)


def test_edge_hint_outside_both_incident_outward_halfspaces_has_no_supporting_projection():
    prepared, _, rotation, _ = native_edge_owner()
    prepared["material_features"][0]["normal_hint"] = (rotation @ [0., -.6, .8]).tolist()
    with pytest.raises(ValueError, match="nonzero support"):
        bind_box_owner_materials(prepared)
# END the exact edge/cone stays in the same declared frame as functional geometry.
# END no renderer, physics state, GPU or task-specific native model is required.
