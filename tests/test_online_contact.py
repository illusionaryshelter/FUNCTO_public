"""Necessary CPU counterexamples for prepared finite planar contact fields."""

# BEGIN fixed-region queries retain obstacles and use finite native material.
from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from manipulation.online_contact import PreparedContactFields, native_material_conditions
from manipulation.online_geometry import FrozenESDF, FrozenPrimitives, OnlinePlanningScene
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import ContactRule, PlanningScene, CollisionOwner, ObservedSpace, PrismaticGripper


def rule(name="left", x=-.006, *, required=False, sphere_indices=(0,), material="pad"):
    points = np.array([[-.02, -.015, 0.], [.02, -.015, 0.], [.02, .015, 0.], [-.02, .015, 0.]])
    return ContactRule(rule_id=name, actor="robot", target_owner="part", interval=(.2, .8),
        sphere_indices=sphere_indices, material_id=material, feature_points=points,
        feature_normal=np.array([0., 0., -1.]), patch_origin=np.array([x, 0., 0.]),
        patch_axes=np.eye(3), patch_half_size=np.array([.002, .002]), gap_bounds=(-.0015, .0015),
        normal_tolerance_rad=.2, modeling_allowance_m=.0015, geometry_error_m=.0015,
        minimum_thickness_m=.008, source="explicit finite test patch", required=required,
        online_material={"kind": "native_planar_extrusion_v1", "thickness_m": .008,
            "native_geometry": "pad_box", "sphere_indices": list(sphere_indices),
            "source": "complete native box extrusion", "sphere_binding_source": "verified test native cover"})


# BEGIN a rotated native box meets the plane on a true edge, not a thin face.
def edge_rule(*, required=False, allowance=.001, patch_half=(.015, .01)):
    half = np.array([.01, .004, .004])
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("x", 45., degrees=True).as_matrix()
    pose[:3, 3] = -(np.array([0., -half[1], -half[2]]) @ pose[:3, :3].T)
    endpoints = np.array([[-half[0], -half[1], -half[2]],
                          [half[0], -half[1], -half[2]]]) @ pose[:3, :3].T+pose[:3, 3]
    return ContactRule(rule_id="edge", actor="payload", target_owner="part", interval=(.2, .8),
        sphere_indices=(0,), material_id="native_bottom_edge", feature_kind="edge",
        feature_points=endpoints, feature_normal=np.array([0., 0., -1.]),
        patch_origin=np.zeros(3), patch_axes=np.eye(3), patch_half_size=np.array(patch_half),
        gap_bounds=(-.001, .001), normal_tolerance_rad=.2,
        modeling_allowance_m=allowance, geometry_error_m=allowance,
        minimum_thickness_m=.008, source="complete rotated native box edge", required=required,
        online_material={"kind": "native_box_edge_v1", "native_geometry": "edge_box",
            "thickness_m": .008, "source": "original native box dimensions and material edge",
            "sphere_indices": [0], "sphere_binding_source": "complete native box cell cover",
            "T_actor_box": pose.tolist(), "half_size_m": half.tolist(),
            "edge_axis": 0, "edge_signs": [-1, -1]})
# END neither the source solid nor its material has an invented bottom plate.


# BEGIN test surfaces are explicit source components, never numerical field zeros.
def field(points=((-0.006, 0., 0.), (.006, 0., 0.)), *, size=.001):
    from manipulation.collision_records import encode_array
    from manipulation.online_observed_geometry import FrozenObservedGeometry
    axis = np.arange(-30, 31) * size
    vertices = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    record = {"kind": "tsdf", "name": "part", "source": "observed isolated zero samples",
        "pose": [0., 0., 0., 1., 0., 0., 0.], "origin": [axis[0]]*3,
        "shape_xyz": [601]*3, "voxel_size": size/10,
        "surface_vertices": encode_array(vertices),
        "surface_offsets": encode_array(np.arange(len(vertices)+1, dtype=np.int64)),
        "negative_vertices": encode_array(np.empty((0, 3), dtype=np.float64)),
        "negative_offsets": encode_array(np.zeros(1, dtype=np.int64)),
        "negative_cell_flat_indices": encode_array(np.empty(0, dtype=np.int64))}
    geometry = FrozenObservedGeometry(record)
    return FrozenESDF.from_source(geometry, origin=[axis[0]]*3, shape=(61,)*3,
        voxel_size=size, owner_ids=("part",), source="observed finite test surface components")
# END derived proposal fields and original source positions are tested separately.


def prepare(query, rules, radius=.01):
    query.prepare_contacts(tuple(rules), {item.rule_id: radius for item in rules})
    return query


def policy(contact_rule, *, position=(0., 0., 0.), rotation=None, active=True, horizon=1):
    p = torch.as_tensor(position, dtype=torch.float64).reshape(1, 1, 3).expand(1, horizon, 3)
    quaternion = ([1., 0., 0., 0.] if rotation is None else
                  Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]])
    q = torch.as_tensor(quaternion, dtype=torch.float64).reshape(1, 1, 4).expand(1, horizon, 4)
    origin, identity = torch.zeros_like(p), torch.zeros_like(q)
    identity[..., 0] = 1.
    progress = torch.full((1, horizon), .5 if active else .1, dtype=torch.float64)
    value, _, _ = PlanningScene._permission(None, contact_rule,
        torch.zeros((1, horizon, 1, 4), dtype=torch.float64), (p, q), (origin, identity), progress, None, None)
    normal = torch.as_tensor(np.asarray(contact_rule.feature_normal) @ (
        np.eye(3) if rotation is None else rotation).T, dtype=torch.float64).reshape(1, 1, 3).expand(1, horizon, 3)
    return value, normal


def spheres(x=0., z=0., radius=.01, requires_grad=False):
    return torch.tensor([[[[x, 0., z, radius]]]], dtype=torch.float64, requires_grad=requires_grad)


# BEGIN finite permission cannot remove another component sharing its proposal cell.
def test_source_components_sharing_a_raster_cell_keep_unpermitted_material():
    item = replace(rule(), patch_half_size=np.array([.0001, .0001]))
    query = prepare(field(points=((-0.006, 0., 0.), (-0.0058, 0., 0.))), [item])
    result, missing = query.contact_query(spheres(x=-.006, radius=.001), [(item, policy(item)[0])])
    assert result.item() > 0 and missing.item() == 0
    region = query._contact_model.regions[item.rule_id]
    assert region["removed_elements"] == 1 and region["element_kind"] == "source_component"
    assert query._contact_model.channels[0]["kind"] == "field"


def test_mapped_required_witnesses_ignore_fake_numerical_zeroes():
    item = rule(required=True)
    query = field(points=((.006, 0., 0.),))
    with pytest.raises(ValueError, match="no observed surface witness"):
        PreparedContactFields.prepare(distance=np.zeros_like(query.distance), origin=query.origin,
            voxel_size=query.voxel_size, rules=(item,), maximum_radii={item.rule_id: .01},
            source="proposal zeros are not material", source_geometry=query.source_geometry)


def test_required_witnesses_retain_original_source_coordinates():
    item = rule(required=True)
    point = (-.0058, .0002, .0001)
    query = prepare(field(points=(point,)), [item])
    witnesses = query._contact_model.regions[item.rule_id]["surface_witnesses"]
    np.testing.assert_array_equal(witnesses, np.array([point]))
    assert not np.allclose((witnesses-query.origin)/query.voxel_size,
                           np.rint((witnesses-query.origin)/query.voxel_size))
# END source material is filtered before creating numerical fields or contact witnesses.


def test_same_sphere_keeps_second_obstacle_outside_permitted_patch_and_gradient():
    item = rule()
    query = prepare(field(), [item])
    body = spheres(requires_grad=True)
    values, missing = query.contact_query(body, [(item, policy(item)[0])])
    assert missing.item() == 0 and values.item() > 0
    gradient = torch.autograd.grad(values.sum(), body)[0]
    assert torch.isfinite(gradient).all() and gradient[0, 0, 0, 0] > 0
    assert not hasattr(query, "_contact_regions")


# BEGIN a larger single field does not increase the aggregate contact bank.
def test_record_capacity_does_not_expand_contact_bank_or_allocate_rejected_fields():
    # Four overlapping permissions produce fifteen nonempty combinations.
    # This 32.25 MiB broadcast source fits the new single-record contract,
    # but its complete combination bank exceeds the unchanged 256 MiB cap.
    values = np.broadcast_to(np.float32(1.), (129, 256, 256))
    rules = tuple(rule(f"patch{index}", x=index * .001) for index in range(4))
    with patch("manipulation.online_contact.require_host_memory") as admission, \
            patch("manipulation.online_contact.distance_transform_edt") as edt:
        with pytest.raises(ValueError, match="field combinations exceed the fixed storage budget"):
            PreparedContactFields.prepare(distance=values, source_geometry=None,
                origin=np.zeros(3), voxel_size=.001, rules=rules,
                maximum_radii={item.rule_id: .01 for item in rules}, source="capacity boundary")
    admission.assert_not_called()
    edt.assert_not_called()
# END aggregate rejection precedes dense allocation or EDT computation.


def test_two_rules_use_true_removed_union_and_inactive_rule_cannot_borrow_it():
    left, right = rule(), rule("right", .006)
    query = prepare(field(), [left, right])
    both, missing = query.contact_query(spheres(), [(left, policy(left)[0]), (right, policy(right)[0])])
    assert missing.item() == 0 and both.item() == 0
    one, _ = query.contact_query(spheres(), [(left, policy(left)[0]), (right, policy(right, active=False)[0])])
    assert one.item() > 0
    none, _ = query.contact_query(spheres(), [(left, policy(left, active=False)[0]),
                                            (right, policy(right, active=False)[0])])
    assert torch.equal(none, query(spheres()))


def test_nonmaterial_sphere_still_queries_complete_owner():
    item = rule()
    query = prepare(field(points=((-0.006, 0., 0.),)), [item])
    body = torch.cat((spheres(), spheres()), dim=-2)
    result, _ = query.contact_query(body, [(item, policy(item)[0])])
    assert result[0, 0, 0] == 0 and result[0, 0, 1] > 0


def test_optional_active_rule_far_outside_owner_is_not_a_map_support_failure():
    item = rule()
    query = prepare(field(), [item])
    result, missing = query.contact_query(spheres(x=2.), [(item, policy(item)[0])])
    assert result.item() == 0 and missing.item() == 0


def test_nonempty_field_boundary_requires_support_when_sphere_can_hit_owner():
    item = rule()
    query = prepare(field(), [item], radius=.04)
    result, missing = query.contact_query(spheres(x=.02, radius=.04), [(item, policy(item)[0])])
    assert missing.item() == 1
    assert torch.equal(result, query(spheres(x=.02, radius=.04)))


def test_empty_forbidden_set_has_no_implicit_edt_boundary_obstacle():
    item = rule()
    query = prepare(field(points=((-0.006, 0., 0.),)), [item], radius=.04)
    result, missing = query.contact_query(spheres(radius=.04), [(item, policy(item)[0])])
    assert result.item() == 0 and missing.item() == 0
    assert query._contact_model.channels[0]["kind"] == "empty"


def test_large_pad_small_patch_and_absent_optional_contact_do_not_force_contact():
    item = rule()
    query = prepare(field(points=((-0.006, 0., 0.),)), [item])
    current, normal = policy(item)
    terms, possible = native_material_conditions(item, current, normal, query._contact_model)
    assert all(value.item() == 0 for value in terms.values())
    assert possible.item()
    far, normal = policy(item, position=(0., 0., .02))
    terms, possible = native_material_conditions(item, far, normal, query._contact_model)
    assert all(value.item() == 0 for value in terms.values())
    assert not possible.item()


# BEGIN optional normal scope follows the complete solid, not a time window alone.
@pytest.mark.parametrize("kind", ["planar", "edge"])
def test_optional_possible_contact_separates_far_wrong_normal_from_side_material(kind):
    item = (replace(rule(x=0.), patch_half_size=np.array([.03, .03]))
            if kind == "planar" else edge_rule())
    rotation = Rotation.from_euler("x", 90., degrees=True).as_matrix()
    # The point is inside the turned complete solid, away from its contact
    # feature. Testing only the front feature would miss this side collision.
    query = prepare(field(points=((0., -.004, 0.),)), [item])
    near, normal = policy(item, rotation=rotation)
    assert not near["normal_valid"].item()
    _, possible = native_material_conditions(item, near, normal, query._contact_model)
    assert possible.dtype == torch.bool and possible.shape == near["active"].shape
    assert possible.item()
    far, normal = policy(item, position=(0., 0., .1), rotation=rotation)
    assert not far["normal_valid"].item()
    terms, possible = native_material_conditions(item, far, normal, query._contact_model)
    assert not possible.item()
    assert all(value.item() == 0 for value in terms.values())
    empty = prepare(field(points=()), [item])
    terms, possible = native_material_conditions(item, near, normal, empty._contact_model)
    assert not possible.item()
    assert all(value.item() == 0 for value in terms.values())
    inactive, normal = policy(item, rotation=rotation, active=False)
    _, possible = native_material_conditions(item, inactive, normal, query._contact_model)
    assert not possible.item()
# END an optional permission never makes alignment a goal while separated.


def test_actual_target_height_and_negative_gap_do_not_double_allowance():
    item = rule()
    item.modeling_allowance_m = .001
    item.gap_bounds = (-.001, .001)
    # The actual source point is at z=.0005. A face at -.001 is 1.5 mm
    # behind it, although its declared plane-to-plane gap is legal. Its
    # rasterized proposal cell must not add another half voxel of material.
    query = prepare(field(points=((-0.006, 0., .0005),), size=.0005), [item])
    current, normal = policy(item, position=(0., 0., -.001))
    terms, possible = native_material_conditions(item, current, normal, query._contact_model)
    assert terms["gap"].item() == 0
    assert terms["material_band"].item() == pytest.approx(.0005-item.numerical_position_tolerance_m, abs=1e-12)


def test_tilted_body_support_includes_back_projection_and_invalid_normal_is_finite():
    item = rule()
    query = prepare(field(points=((-0.006, 0., 0.),)), [item])
    for angle in (10., 90., 180.):
        rotation = Rotation.from_euler("y", angle, degrees=True).as_matrix()
        current, normal = policy(item, position=(0., 0., -.001), rotation=rotation)
        terms, possible = native_material_conditions(item, current, normal, query._contact_model)
        assert all(torch.isfinite(value).all() for value in terms.values())
    current, normal = policy(item, position=(0., 0., -.001),
                             rotation=Rotation.from_euler("y", -10., degrees=True).as_matrix())
    assert native_material_conditions(item, current, normal, query._contact_model)[0]["material_band"].item() > 0


# BEGIN side-wedge and endpoint counterexamples protect the finite edge contract.
def test_edge_positive_allowance_cannot_admit_a_side_wedge_of_the_complete_box():
    item = edge_rule(allowance=.001)
    # This observed point is inside the original box but 0.2 mm sideways from
    # its edge. A 1 mm support-plane-only allowance would wrongly admit it.
    query = prepare(field(points=((0., .0002, .0003),)), [item])
    current, normal = policy(item)
    current["feature_points"].requires_grad_(True)
    terms, possible = native_material_conditions(item, current, normal, query._contact_model)
    assert terms["gap"].item() == 0
    assert terms["material_band"].item() > .00029
    gradient = torch.autograd.grad(terms["material_band"].sum(), current["feature_points"])[0]
    assert torch.isfinite(gradient).all()
    # A point on the actual edge remains admissible under the identical model.
    clear = prepare(field(points=((0., 0., 0.),)), [item])
    assert all(value.item() == 0 for value in native_material_conditions(
        item, current, normal, clear._contact_model)[0].values())


def test_edge_gap_uses_only_the_real_segment_inside_the_patch():
    item = replace(edge_rule(patch_half=(.001, .002)), gap_bounds=(-.0003, .0003))
    query = prepare(field(points=((0., 0., 0.),)), [item])
    current, normal = policy(item, rotation=Rotation.from_euler("y", 10., degrees=True).as_matrix())
    assert current["feature_points"][..., 2].abs().max().item() > .001
    assert native_material_conditions(item, current, normal, query._contact_model)[0]["gap"].item() == 0


def test_required_edge_contact_cannot_borrow_points_beyond_its_endpoint_or_side():
    item = edge_rule(required=True)
    for point in ((.010005, 0., 0.), (0., .000005, 0.)):
        query = prepare(field(points=(point,)), [item])
        assert query.required_contact(item, policy(item)[0]).item() > 0
    query = prepare(field(points=((.01, 0., 0.),)), [item])
    assert query.required_contact(item, policy(item)[0]).item() == 0


def test_edge_registration_rejects_support_cone_boundary_and_modified_native_endpoints():
    item = edge_rule()
    adjacent = item._native_edge_geometry["adjacent_normals_actor"]
    for normal in (adjacent[0], adjacent[0]+1e-15*adjacent[1], -item.feature_normal):
        with pytest.raises(ValueError, match="supporting normal cone"):
            replace(item, feature_normal=normal)
    points = item.feature_points.copy()
    points[1, 0] -= .001
    with pytest.raises(ValueError, match="complete registered native edge"):
        replace(item, feature_points=points)
    model = deepcopy(item.online_material)
    model["T_actor_box"][0][0] += 1e-7
    with pytest.raises(ValueError, match="rigid orthonormal"):
        replace(item, online_material=model)


def test_edge_record_roundtrip_preserves_native_box_and_forbidden_side_band():
    item = edge_rule()
    query = prepare(field(points=((0., .0002, .0003),)), [item])
    with patch("manipulation.online_contact.distance_transform_edt", side_effect=AssertionError("not runtime")):
        restored = FrozenESDF.from_record(query.record())
        restored.prepare_contacts((item,), {item.rule_id: .01})
    assert restored._contact_model.rule_records[0]["online_material"] == item.online_material
    current, normal = policy(item)
    original, original_possible = native_material_conditions(item, current, normal, query._contact_model)
    actual, actual_possible = native_material_conditions(item, current, normal, restored._contact_model)
    assert all(torch.equal(original[name], actual[name]) for name in original)
    assert torch.equal(original_possible, actual_possible)
# END accepted planar contacts still use their original formula and tolerances.


def test_required_contact_uses_same_point_observed_evidence_and_absence_is_capability_gap():
    item = rule(required=True)
    query = prepare(field(points=((-0.006, 0., 0.),)), [item])
    assert query.required_contact(item, policy(item)[0]).item() == 0
    assert query.required_contact(item, policy(item, position=(.1, 0., 0.))[0]).item() > 0
    empty = field(points=())
    with pytest.raises(ValueError, match="no observed surface witness"):
        prepare(empty, [item])


def test_current_binding_reuses_exact_arrays_without_preparation_or_legacy_fallback():
    item = rule()
    query = prepare(field(), [item])
    record = query.record()
    with patch("manipulation.online_contact.distance_transform_edt", side_effect=AssertionError("not runtime")):
        restored = FrozenESDF.from_record(record)
        restored.prepare_contacts((item,), {item.rule_id: .01})
        original = query.contact_query(spheres(), [(item, policy(item)[0])])
        actual = restored.contact_query(spheres(), [(item, policy(item)[0])])
    assert all(torch.equal(a, b) for a, b in zip(original, actual))
    assert record["kind"] == "frozen_observed_field_v3"
    record["kind"] = "frozen_esdf_v2"
    with pytest.raises(ValueError, match="current frozen-field schema"):
        FrozenESDF.from_record(record)


def test_unregistered_material_or_too_many_joint_rule_channels_is_rejected():
    with pytest.raises(ValueError, match="registered complete native material"):
        prepare(field(), [replace(rule(), online_material=None)])
    items = [rule(f"rule{i}") for i in range(5)]
    with pytest.raises(ValueError, match="simultaneous-rule"):
        prepare(field(), items)


def test_primitive_contact_preparation_retains_remote_primitive_of_same_owner():
    item = rule(x=0.)
    query = FrozenPrimitives(primitives=[
        {"kind": "sphere", "dimensions": [.0003]},
        {"kind": "sphere", "dimensions": [.0003], "pose": [.02, 0., 0., 1., 0., 0., 0.]}],
        owner_ids=("part",), source="complete two-primitive registered owner", contact_voxel_size_m=.001)
    prepare(query, [item], radius=.004)
    result, missing = query.contact_query(spheres(x=.02, radius=.004), [(item, policy(item)[0])])
    assert result.item() > 0 and missing.item() == 0


def test_full_scene_modes_material_guards_and_execution_use_same_frozen_model():
    item = rule()
    owner_field = field(points=((-0.006, 0., 0.),))
    background = field(points=())
    background.owner_ids = frozenset(("background",))
    known = np.ones(owner_field.distance.shape, bool)
    coverage = ObservedSpace(known, origin=owner_field.origin, voxel_size=owner_field.voxel_size,
        observation_ids=("current",), reference_frame="world")
    owner = CollisionOwner("part", owner_field, PosePath(np.array([0., 1.]),
        np.stack((np.eye(4), np.eye(4))), "fixed observed owner"), "observed owner", True)
    scene = OnlinePlanningScene(background=background, owners=(owner,), coverage=coverage,
        reference_frame="world", robot_sphere_materials=("pad",), robot_geometry_source="native model",
        contacts=(item,), unknown_policy={"conditional_regions": [], "outside_regions": "forbid"},
        native_geometry_materials={"pad_box": "pad"})
    scene.bind_online_robot(robot_sphere_geometry=[{"link_name": "pad", "sphere_indices": [0],
        "centers_local": [[0., 0., 0.]], "radii": [.01]}], robot_self_collision_pairs=[])
    body = spheres().numpy()[0, 0]
    assert scene.query_state(body, np.eye(4), .5, active_contact_rule_ids={item.rule_id})["passed"]
    assert not scene.query_state(body, np.eye(4), .5, active_contact_rule_ids=set())["passed"]
    assert not scene.query_state(body, np.eye(4), .1, active_contact_rule_ids={item.rule_id})["passed"]
    moved = np.eye(4)
    moved[2, 3] = -.002
    guarded = scene.query_state(body, moved, .5, active_contact_rule_ids={item.rule_id})
    assert not guarded["passed"] and guarded["components"]["contact/material_band/left"] > 0
    with patch("manipulation.online_contact.distance_transform_edt", side_effect=AssertionError("no execution rebuild")):
        restored = OnlinePlanningScene.from_binding(scene.to_execution_binding())
        assert restored.query_state(body, moved, .5, active_contact_rule_ids={item.rule_id}) == guarded


def test_required_contact_with_command_only_fingers_retains_unresolved_obligation():
    item = rule(required=True)
    translation = np.zeros((*item.feature_points.shape, 1))
    translation[:, 0, 0] = 1.
    item = replace(item, feature_joint_translation=translation)
    hand = PrismaticGripper(controller='binary', joint_names=('finger',), current_positions=[0.],
        joint_limits=[[-.01, .01]], joint_bounds=None, coupling_matrix=np.empty((0, 1)), coupling_offset=[],
        sphere_indices=(0,), local_spheres=[[0., 0., 0., .01]], sphere_joint_translation=[[[1.], [0.], [0.]]],
        self_collision_pairs=(), commands=(), source='unmodelled independently commanded finger', control_timestep_s=.01)
    owner_field = field(points=((-0.006, 0., 0.),))
    background = field(points=())
    background.owner_ids = frozenset(('background',))
    coverage = ObservedSpace(np.ones(owner_field.distance.shape, bool), origin=owner_field.origin,
        voxel_size=owner_field.voxel_size, observation_ids=('A',), reference_frame='world')
    owner = CollisionOwner('part', owner_field, PosePath(np.array([0., 1.]),
        np.stack((np.eye(4), np.eye(4))), 'fixed observed owner'), 'observed owner', True)
    scene = OnlinePlanningScene(background=background, owners=(owner,), coverage=coverage,
        reference_frame='world', robot_sphere_materials=('pad',), robot_geometry_source='native model',
        contacts=(item,), gripper=hand, unknown_policy={'conditional_regions': [], 'outside_regions': 'forbid'},
        native_geometry_materials={'pad_box': 'pad'})
    scene.bind_online_robot(robot_sphere_geometry=[{'link_name': 'pad', 'sphere_indices': [0],
        'centers_local': [[0., 0., 0.]], 'radii': [.01]}], robot_self_collision_pairs=[])
    terms = scene.components(spheres(), torch.zeros((1, 1, 3), dtype=torch.float64),
        torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64), torch.tensor([[.5]], dtype=torch.float64),
        trajectory_time=torch.tensor([[1.]], dtype=torch.float64))
    assert terms['conditional/finger/left'].item() == 1.
    assert terms['unsupported/required_contact_geometry/left'].item() == 1.
    actual = scene.query_state(spheres().numpy()[0, 0], np.eye(4), .5, joint_position=[0.],
        joint_names=('finger',), trajectory_time=1., active_contact_rule_ids={'left'})
    assert actual['components']['unsupported/required_contact_geometry/left'] == 0.
    assert actual['components']['conditional/finger/left'] == 0.


@pytest.mark.parametrize("corruption", ["nonfinite_region", "missing_membership", "zero_radius", "missing_required_witness"])
def test_loading_rejects_incomplete_or_nonfinite_contact_models(corruption):
    item = rule(required=corruption == "missing_required_witness")
    row = prepare(field(), [item]).record()
    model = row["contact_model"]
    if corruption == "nonfinite_region":
        model["regions"][item.rule_id]["upper_patch"][0] = float("nan")
    elif corruption == "missing_membership":
        model["groups"][0]["sphere_indices"] = []
    elif corruption == "zero_radius":
        model["maximum_radii"][item.rule_id] = 0.
    else:
        from manipulation.collision_records import encode_array
        model["regions"][item.rule_id]["surface_witnesses"] = encode_array(np.empty((0, 3)))
    with pytest.raises(ValueError):
        FrozenESDF.from_record(row)


def test_workspace_counts_unique_source_arrays_and_both_float_caches_on_cpu_and_gpu():
    query = prepare(field(), [rule()])
    coverage = SimpleNamespace(_known=np.ones((2, 2, 2), bool), _known_free=np.ones((2, 2, 2), bool),
                              # BEGIN the coverage source now stores squared distances.
                              _distance_squared=torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32))
    # END the storage size and resource contract are unchanged.
    # BEGIN resource fixture includes the bound scene's empty integer partitions.
    scene = SimpleNamespace(background=query, owners=(), coverage=coverage, robot_sphere_materials=("pad",), gripper=None,
                            # BEGIN empty policy shares the original coverage allocation.
                            _permitted_coverage=coverage, unknown_policy={'conditional_regions': []},
                            # END a second policy field is tested with real online scenes.
                            robot_sphere_geometry=({'link_name': 'pad'},),
                            _self_collision_pair_arrays=(np.empty((0, 2), np.int64), np.empty((0, 2), np.int64)))
    # END fixed pair storage is exercised separately with a real robot binding.
    base = OnlinePlanningScene.estimated_workspace_allocation(scene, 1, 1)
    # BEGIN source components and proposal fields are counted without native-site masks.
    # The frozen source owns its index/cache reservation once; sibling scenes
    # borrow it. The comparison below checks that adding a sibling cannot
    # silently reserve a second source or another pair of field caches.
    assert base["cpu_shared"] >= query.distance.nbytes + query._contact_model.storage_nbytes
    assert base["gpu_shared"] >= query.distance.nbytes + query._contact_model.storage_nbytes
    # END exact aggregate byte formulas belong to the shared scene accountant.
    # BEGIN shared static fields add no moving-owner coverage queries.
    scene.owners = (SimpleNamespace(query=query.clone_for_rollout(), moving=False),)
    # END their field caches still alias the original storage.
    shared = OnlinePlanningScene.estimated_workspace_allocation(scene, 1, 1)
    assert shared == base
    # The actual CPU caches may coexist; no per-membership bank duplication.
    for dtype in (torch.float32, torch.float64):
        value, _ = policy(rule())
        value = {name: tensor.to(dtype=dtype) if isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
                 else tensor for name, tensor in value.items()}
        query.contact_query(spheres().to(dtype), [(rule(), value)])
    banks = [key for key in query._contact_model._device if key[0] == "fields"]
    assert len(banks) == 2
    assert base["cpu_live"] >= query._contact_model.storage_nbytes
# END observations, interval latches, finite materials and outside obstacles remain distinct.
