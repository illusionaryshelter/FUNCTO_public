"""CPU counterexamples for shared layers, complete envelopes and sensor margin."""

# BEGIN each test binds explicit native geometry and the same recorded body choice.
from copy import deepcopy

import coal
import numpy as np
import pytest
import torch

from manipulation.collision_records import decode_array, encode_array
from manipulation.robot_geometry_check import (
    NativePrimitive, RobotGeometryChecker, _overlap_within_region, _solid_axis_bounds,
)
from manipulation.voxel_layers import FineRegion
from test_independent_scene_validation import binding, checker, rigid, rule


def region_record(margin=0.):
    shape = (21, 21, 21)
    return {"shape_xyz": list(shape), "pose": [0., 0., 0., 1., 0., 0., 0.],
            "voxel_size_m": .02, "numerical_support_m": .04,
            "measurement_allowance_m": margin, "measurement_evidence": "bounded sensor error test",
            "source": "complete same-view support fixture",
            "source_bits_hex": np.full(shape, 3, np.uint8).tobytes().hex(),
            "raw_free_packbits_hex": np.packbits(np.zeros(shape, np.uint8)).tobytes().hex()}


def layered_binding(*, fine_top=-.01, coarse_top=.005, margin=0., contacts=()):
    model = binding(contacts)
    model["robot_sphere_materials"] = ["contact_pad"]
    model["robot_collision_bodies"] = [{"body_id": "complete_pad", "sphere_indices": [0],
        "geometry_names": ["pad"], "frame": "base", "local_spheres": [[0., 0., .049, .09]]}]
    original = model["owners"][0]["geometry"]
    layers = {name: deepcopy(original) for name in ("coarse", "fine")}
    for name, top in (("coarse", coarse_top), ("fine", fine_top)):
        layers[name]["solids"][0]["pose"][2] = top-.05
    model["owners"][0]["geometry"] = {"layers": layers, "fine_region": region_record(margin)}
    return model


def change_array(record, key, index, value):
    """Mutate test evidence while keeping its exact archive representation."""
    values = decode_array(record[key]).copy()
    values[index] = value
    record[key] = encode_array(values)


def saved_selection(model, *, fine=True):
    selected = FineRegion(model["owners"][0]["geometry"]["fine_region"]).select(
        torch.tensor([[[[0., 0., .049, .09]]]], dtype=torch.float64), model["robot_collision_bodies"]).record()
    change_array(selected, "body_fine", (0, 0, 0), fine)
    return {"robot/target": selected}


def evaluate(checker, model, *, selection=None, index=(0, 0), select=False, **kwargs):
    checker.bind_scene(model)
    state = checker.query(joint_names=checker.joint_names, joint_positions=[0., 0.],
                          T_reference_base=np.eye(4), pairs=())
    return checker.query_scene(state, tool_pose=np.eye(4), progress=.5, finger_positions=[0., 0.],
        state_kind="nominal", layer_selection=selection, layer_selection_index=index,
        select_layers=select, **kwargs)


@pytest.mark.parametrize("fine,passed", [(True, True), (False, False)])
def test_only_saved_layer_is_queried_even_when_other_layer_disagrees(checker, fine, passed):
    model = layered_binding()
    result = evaluate(checker, model, selection=saved_selection(model, fine=fine))
    assert result["passed"] is passed
    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["layer"] == ("fine" if fine else "coarse")
    assert result["layer_selection_checks"][0]["envelope_contained"]


def test_missing_record_does_not_default_to_coarse_or_reselect(checker, monkeypatch):
    model = layered_binding()
    monkeypatch.setattr(FineRegion, "select", lambda *args: pytest.fail("independent replay must not reselect"))
    result = evaluate(checker, model)
    assert not result["passed"] and not result["pairs"]
    assert result["query_gaps"][0]["reason"] == "shared_layer_selection_missing"


def test_new_state_selection_is_explicit_once_and_archived(checker, monkeypatch):
    model = layered_binding()
    calls, original = [], FineRegion.select
    def tracked(self, *args):
        calls.append(1)
        return original(self, *args)
    monkeypatch.setattr(FineRegion, "select", tracked)
    result = evaluate(checker, model, select=True)
    assert result["passed"] and len(calls) == 1
    assert decode_array(result["layer_selection"]["robot/target"]["body_fine"]).tolist() == [[[True]]]
    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluate(checker, model, selection=result["layer_selection"], select=True)


def test_no_fk_rounding_allowance_is_silently_added_to_saved_envelope(checker):
    model = layered_binding(margin=.001)
    selected = saved_selection(model)
    # Pad max x plus sensor allowance is exactly .051. A saved bound even
    # 1e-10 m smaller must be a query gap, without inflating any tolerance.
    change_array(selected["robot/target"], "influence_aabb_grid_m", (0, 0, 0, 1, 0), .051-1e-10)
    result = evaluate(checker, model, selection=selected)
    assert not result["passed"] and not result["pairs"]
    gap = result["query_gaps"][0]
    assert gap["reason"] == "native_body_outside_shared_query_envelope"
    assert gap["maximum_envelope_excess_m"] == pytest.approx(1e-10, abs=1e-15)


@pytest.mark.parametrize("damage", ["body_id", "shape", "boolean", "indices", "legacy", "float32", "base64"])
def test_malformed_shared_record_is_explicit_gap(checker, damage):
    model = layered_binding()
    saved = saved_selection(model)
    row = saved["robot/target"]
    if damage == "body_id":
        row["body_ids"] = ["another_body"]
    elif damage == "shape":
        row["influence_aabb_grid_m"] = encode_array(np.empty(0, dtype=np.float64))
    elif damage == "boolean":
        row["body_fine"] = encode_array(np.array([[[1]]], dtype=np.int64))
    elif damage == "indices":
        change_array(row, "index_lower", (0, 0, 0), [1000, 1000, 1000])
    elif damage == "legacy":
        row["body_fine"] = decode_array(row["body_fine"]).tolist()
    elif damage == "float32":
        row["influence_aabb_grid_m"] = encode_array(decode_array(row["influence_aabb_grid_m"]).astype(np.float32))
    else:
        row["body_fine"]["data_base64"] = "invalid"
    result = evaluate(checker, model, selection=saved)
    assert not result["passed"] and not result["pairs"]
    assert result["query_gaps"][0]["reason"] == "shared_layer_selection_invalid"


def test_batch_horizon_index_consumes_that_exact_archived_decision(checker):
    model = layered_binding()
    saved = saved_selection(model)
    row = saved["robot/target"]
    for key in ("body_fine", "influence_aabb_grid_m", "index_lower", "index_upper"):
        row[key] = encode_array(np.repeat(decode_array(row[key]), 2, axis=1))
    change_array(row, "body_fine", (0, 1, 0), False)
    result = evaluate(checker, model, selection=saved, index=(0, 1))
    assert not result["passed"] and result["pairs"][0]["layer"] == "coarse"
    archived = result["layer_selection"]["robot/target"]
    assert decode_array(archived["body_fine"]).tolist() == [[[False]]]
    for key in ("body_fine", "influence_aabb_grid_m", "index_lower", "index_upper"):
        assert set(archived[key]) == {"shape", "dtype", "data_base64"}
        np.testing.assert_array_equal(decode_array(archived[key]), decode_array(row[key])[:, 1:2])


@pytest.mark.parametrize("fine", [False, True])
def test_sensor_margin_rejects_geometrically_separated_entity_on_both_layers(checker, fine):
    model = layered_binding(fine_top=-.0015, coarse_top=-.0015, margin=.001)
    result = evaluate(checker, model, selection=saved_selection(model, fine=fine))
    assert not result["passed"]
    row = result["pairs"][0]
    assert row["relation"] == "separated"
    assert row["signed_distance_m"] == pytest.approx(.0005, abs=2e-6)
    assert row["measurement_margin_violated"] and row["contacts"] == []
    assert result["violations"][0]["kind"] == "independent_scene_measurement_margin"


def test_local_material_permission_checks_expanded_intersection_with_unchanged_allowance(checker):
    contact = rule(feature_points=[[-.06, -.06, -.001], [.06, -.06, -.001],
                                  [.06, .06, -.001], [-.06, .06, -.001]])
    model = layered_binding(fine_top=-.0015, coarse_top=-.0015, margin=.001, contacts=[contact])
    result = evaluate(checker, model, selection=saved_selection(model))
    assert result["passed"]
    row = result["pairs"][0]
    assert row["permission"] == contact["rule_id"]
    assert row["contact_region_checks"][0]["margin_geometry"].startswith("support_plane_outer_polytope")
    assert checker._scene_binding["contacts"][0]["modeling_allowance_m"] == .003
    # Uninflated pad fits exactly in this polygon. Its measurement band does
    # not; an empty raw intersection cannot vacuously authorize the band.
    model["contacts"][0] = rule()
    rejected = evaluate(checker, model, selection=saved_selection(model))
    assert not rejected["passed"]
    assert rejected["pairs"][0]["contact_region_checks"][0]["contained"] is False


def test_complete_group_cannot_validate_only_one_of_its_native_components(checker):
    solids = [NativePrimitive("pad", "base", "box", (.1, .1, .1), rigid(.049), "first native component"),
              NativePrimitive("second", "base", "box", (.1, .1, .1), rigid(.249), "second native component")]
    both = RobotGeometryChecker(checker.urdf_path, base_frame="base", geometries=solids)
    model = layered_binding()
    model["robot_collision_bodies"][0]["geometry_names"].append("second")
    result = evaluate(both, model, selection=saved_selection(model))
    assert not result["passed"] and not result["pairs"]
    assert result["query_gaps"][0]["reason"] == "native_body_outside_shared_query_envelope"


def test_unbound_native_actor_never_inherits_another_body_layer(checker):
    solids = [NativePrimitive(name, "base", "box", (.1, .1, .1), rigid(.049), "explicit native solid")
              for name in ("pad", "unbound")]
    model_checker = RobotGeometryChecker(checker.urdf_path, base_frame="base", geometries=solids)
    model = layered_binding()
    result = evaluate(model_checker, model, selection=saved_selection(model))
    assert not result["passed"]
    assert any(gap["reason"] == "native_actor_body_identity_unbound" for gap in result["query_gaps"])
    assert all(row["actor_geometry"] == "pad" for row in result["pairs"])


def test_rotated_cylinder_envelope_uses_analytic_support():
    shape, pose = coal.Cylinder(.1, .4), np.eye(4)
    angle = .7
    pose[:3, :3] = [[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]
    low, high = _solid_axis_bounds(None, shape, pose)
    expected = [.1*np.cos(angle)+.2*np.sin(angle), .1, .1*np.sin(angle)+.2*np.cos(angle)]
    np.testing.assert_allclose(high, expected)
    np.testing.assert_allclose(low, -np.asarray(expected))


def test_layered_map_cannot_be_used_as_complete_moving_actor(checker):
    model = layered_binding()
    model["owners"][0]["moving"] = True
    with pytest.raises(ValueError, match="complete moving actor"):
        checker.bind_scene(model)


def test_unselected_layer_query_gap_cannot_contaminate_selected_inventory(checker):
    model = layered_binding()
    model["owners"][0]["geometry"]["layers"]["coarse"]["query_gaps"] = ["coarse_gap"]
    accepted = evaluate(checker, model, selection=saved_selection(model))
    assert accepted["passed"] and not accepted["query_gaps"]
    rejected = evaluate(checker, model, selection=saved_selection(model, fine=False))
    assert any(gap["reason"] == "coarse_gap" for gap in rejected["query_gaps"])
# END the archived decision never depends on which layer would have lower cost.

# BEGIN moving payloads and finite occupied leaves retain the same selection law.
def test_complete_moving_owner_uses_its_own_cover_and_selected_target_layer(checker):
    model = layered_binding()
    payload = deepcopy(binding()["owners"][0])
    payload.update(owner_id="payload", moving=True, local_spheres=[[0., 0., 0., .018]],
                   sphere_cover_error_m=.002)
    pose = rigid(.049)
    pose[0, 3] = .12
    payload["pose_path"]["poses"] = [pose.tolist()]*2
    payload["geometry"]["solids"][0].update(name="payload_solid", dimensions=[.02]*3,
                                             pose=[0., 0., 0., 1., 0., 0., 0.])
    model["owners"].append(payload)
    result = evaluate(checker, model, select=True)
    assert result["passed"]
    record = result["layer_selection"]["payload/target"]
    assert record["body_ids"] == ["payload"] and decode_array(record["body_fine"]).tolist() == [[[True]]]
    low, high = decode_array(record["influence_aabb_grid_m"])[0, 0, 0]
    np.testing.assert_allclose(high-low, [.04]*3)
    rows = [row for row in result["pairs"] if row["actor"] == "payload" and row["target_owner"] == "target"]
    assert len(rows) == 1 and rows[0]["layer"] == "fine"


def test_required_contact_uses_selected_field_occupancy(checker):
    model = layered_binding(contacts=[rule(required=True)])
    result = evaluate(checker, model, selection=saved_selection(model))
    assert not result["passed"]
    assert result["violations"][0]["kind"] == "independent_required_contact"
    checks = result["required_contact_checks"][0]["target_components"]
    assert all(check["layer"] == "fine" for check in checks)
    assert not result["required_contact_checks"][0]["target_occupancy_in_region"]


def test_required_contact_cannot_borrow_material_feature_from_another_body(checker):
    model = layered_binding(fine_top=-.0015, coarse_top=-.0015, margin=.001, contacts=[rule()])
    model["contacts"][0]["required"] = True
    model["contacts"][0]["sphere_indices"] = [9]
    result = evaluate(checker, model, selection=saved_selection(model))
    assert not result["passed"]
    assert any(gap["reason"] == "required_contact_layer_selection_unbound" for gap in result["query_gaps"])


def test_margin_finite_cell_query_does_not_drop_originally_disjoint_leaves():
    from test_robot_geometry_check import corners
    actor = corners(.05)
    occupied = {"voxel_boxes": np.array([[0., 0., -.151, .2]]), "pose": np.eye(4)}
    wide = (np.array([[1., 0., 0.]]), np.array([.06]))
    raw = _overlap_within_region(actor, occupied, wide)
    assert raw["intersection_empty"]
    expanded = _overlap_within_region(actor, occupied, wide, first_margin_m=.002)
    assert expanded["completed"] and expanded["contained"] and not expanded.get("intersection_empty", False)
    narrow = (np.array([[1., 0., 0.]]), np.array([.0505]))
    rejected = _overlap_within_region(actor, occupied, narrow, first_margin_m=.002)
    assert rejected["completed"] and not rejected["contained"]


def test_margin_plane_expansion_is_conservative_at_sharp_corners():
    from test_robot_geometry_check import corners
    actor = corners(.05)
    # The target touches the square-corner outer approximation while lying
    # outside the true radius .01 rounded corner. Permission remains unproven.
    target = corners(.0001)+[.059, .059, 0.]
    region = (np.array([[1., 1., 0.]]), np.array([.11]))
    raw = _overlap_within_region(actor, target, region)
    assert raw["intersection_empty"]
    expanded = _overlap_within_region(actor, target, region, first_margin_m=.01)
    assert expanded["completed"] and not expanded["contained"]
# END finite-region permission is conservative while raw Coal distances retain native-model evidence.

# BEGIN one required material rule cannot mix or omit complete-body layer evidence.
def two_body_case(checker, *, second_z=.049):
    solids = [NativePrimitive("pad", "base", "box", (.1, .1, .1), rigid(.049), "first material body"),
              NativePrimitive("second", "base", "box", (.1, .1, .1), rigid(second_z), "second material body")]
    native = RobotGeometryChecker(checker.urdf_path, base_frame="base", geometries=solids)
    model = layered_binding(contacts=[rule(required=True, sphere_indices=(0, 1))])
    model["native_geometry_materials"]["second"] = "contact_pad"
    model["robot_sphere_materials"].append("contact_pad")
    model["robot_collision_bodies"].append({"body_id": "second_body", "sphere_indices": [1],
        "geometry_names": ["second"], "frame": "base", "local_spheres": [[0., 0., .049, .09]]})
    record = FineRegion(model["owners"][0]["geometry"]["fine_region"]).select(
        torch.tensor([[[[0., 0., .049, .09], [0., 0., .049, .09]]]], dtype=torch.float64),
        model["robot_collision_bodies"]).record()
    return native, model, {"robot/target": record}


def test_required_material_rule_rejects_mixed_coarse_and_fine_bodies(checker):
    native, model, selected = two_body_case(checker)
    change_array(selected["robot/target"], "body_fine", (0, 0, 1), False)
    result = evaluate(native, model, selection=selected)
    assert not result["passed"] and not result["required_contact_checks"]
    gap = next(gap for gap in result["query_gaps"] if gap["reason"] == "required_contact_mixed_body_layers")
    assert gap["selected_layers"] == ["coarse", "fine"]


def test_required_material_rule_cannot_ignore_body_with_invalid_envelope(checker):
    native, model, selected = two_body_case(checker, second_z=.249)
    result = evaluate(native, model, selection=selected)
    assert not result["passed"] and not result["required_contact_checks"]
    gap = next(gap for gap in result["query_gaps"] if gap["reason"] == "required_contact_layer_selection_unbound")
    assert gap["missing_actor_geometries"] == ["second"]
# END required occupancy is checked only after every participating body's layer is bound.

# BEGIN replay authenticates saved support instead of trusting a copied fine flag.
def test_forged_fine_flag_cannot_hide_a_source_support_hole(checker, monkeypatch):
    model = layered_binding()
    region = model["owners"][0]["geometry"]["fine_region"]
    bits = np.frombuffer(bytes.fromhex(region["source_bits_hex"]), np.uint8).copy().reshape(region["shape_xyz"])
    bits[10, 10, 10] = 0
    region["source_bits_hex"] = bits.tobytes().hex()
    selected = saved_selection(model, fine=True)
    monkeypatch.setattr(FineRegion, "select", lambda *args: pytest.fail("tampered replay must not reselect"))
    result = evaluate(checker, model, selection=selected)
    assert not result["passed"] and not result["pairs"]
    assert result["query_gaps"][0]["reason"] == "shared_fine_layer_source_support_invalid"
    assert not result["layer_selection_checks"][0]["saved_envelope_fine_supported"]


def test_enlarged_but_self_consistent_saved_fine_envelope_must_remain_inside_support(checker):
    model = layered_binding()
    record = FineRegion(model["owners"][0]["geometry"]["fine_region"]).select(
        torch.tensor([[[[0., 0., .049, .18]]]], dtype=torch.float64), model["robot_collision_bodies"]).record()
    assert decode_array(record["body_fine"]).tolist() == [[[False]]]
    change_array(record, "body_fine", (0, 0, 0), True)
    result = evaluate(checker, model, selection={"robot/target": record})
    assert not result["passed"] and not result["pairs"]
    gap = result["query_gaps"][0]
    assert gap["reason"] == "shared_fine_layer_source_support_invalid"
    assert gap["envelope_contained"] and gap["saved_indices_match_envelope"]


@pytest.mark.parametrize("fine", [False, True])
def test_modified_stencil_indices_cannot_shrink_the_saved_query_envelope(checker, fine):
    model = layered_binding()
    selected = saved_selection(model, fine=fine)
    old = decode_array(selected["robot/target"]["index_lower"])[0, 0, 0, 0]
    change_array(selected["robot/target"], "index_lower", (0, 0, 0, 0), old+1)
    result = evaluate(checker, model, selection=selected)
    assert not result["passed"] and not result["pairs"]
    assert result["query_gaps"][0]["reason"] == "shared_layer_indices_do_not_match_envelope"


def test_coarse_choice_can_retain_a_valid_record_of_an_unsupported_envelope(checker):
    model = layered_binding(coarse_top=-.01)
    region = model["owners"][0]["geometry"]["fine_region"]
    region["source_bits_hex"] = bytes(np.prod(region["shape_xyz"])).hex()
    selected = saved_selection(model, fine=False)
    result = evaluate(checker, model, selection=selected)
    assert result["passed"] and result["pairs"][0]["layer"] == "coarse"
    assert not result["layer_selection_checks"][0]["saved_envelope_fine_supported"]
    assert result["layer_selection_checks"][0]["saved_indices_match_envelope"]
# END invalid fine evidence causes an explicit gap and no alternative-layer collision query.
