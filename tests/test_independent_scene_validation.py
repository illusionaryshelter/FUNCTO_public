"""CPU counterexamples for independent scene and finite material permissions."""

# BEGIN independent scene tests use native filled solids, never planner sphere costs.
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.tsdf_geometry import tsdf_geometry_from_samples
from manipulation.planning_scene import ContactRule
from manipulation.robot_geometry_check import NativePrimitive, RobotGeometryChecker
from test_robot_geometry_check import URDF


def rigid(z=0.):
    pose = np.eye(4)
    pose[2, 3] = z
    return pose


def rule(**changes):
    args = dict(rule_id="pad_to_target", actor="robot", target_owner="target", interval=(0., 1.),
        sphere_indices=(0,), material_id="contact_pad", feature_points=[[-.05, -.05, -.001], [.05, -.05, -.001],
            [.05, .05, -.001], [-.05, .05, -.001]], feature_normal=[0., 0., -1.],
        patch_origin=[0., 0., 0.], patch_axes=np.eye(3), patch_half_size=[.2, .2],
        gap_bounds=(-.002, .002), normal_tolerance_rad=.01, modeling_allowance_m=.003,
        geometry_error_m=.003, minimum_thickness_m=.1, source="calibrated finite pad and patch")
    args.update(changes)
    return asdict(ContactRule(**args))


def binding(contacts=()):
    return {"reference_frame": "world", "native_geometry_materials": {"pad": "contact_pad"},
        "contacts": list(contacts), "background": {"geometry": {"solids": [], "query_gaps": []}},
        "owners": [{"owner_id": "target", "source": "complete target box", "geometry_complete": True,
            "moving": False, "attachments": [], "native_geometry_names": [], "pose_interval": [0., 1.],
            "pose_path": {"progress": [0., 1.], "poses": [np.eye(4).tolist()]*2,
                "source": "current target observation", "attachment": np.eye(4).tolist(),
                "coordinates": None, "motion_model": None},
            "geometry": {"solids": [{"name": "slab", "kind": "cuboid", "dimensions": [1., 1., .1],
                "pose": [0., 0., -.05, 1., 0., 0., 0.], "source": "native target dimensions"}], "query_gaps": []}}]}


@pytest.fixture
def checker(tmp_path):
    path = tmp_path / "model.urdf"
    path.write_text(URDF)
    return RobotGeometryChecker(path, base_frame="base", geometries=[
        NativePrimitive("pad", "base", "box", (.1, .1, .1), rigid(.049), "complete pad native solid")])


def query(checker, model, *, progress=.5, kind="actual", condition=None, owner_poses=None):
    checker.bind_scene(model)
    robot = checker.query(joint_names=checker.joint_names, joint_positions=[0., 0.],
                          T_reference_base=np.eye(4), pairs=())
    return checker.query_scene(robot, tool_pose=np.eye(4), progress=progress, finger_positions=[0., 0.],
                               state_kind=kind, condition_evidence=condition, owner_poses=owner_poses)


def test_local_permission_checks_whole_overlap_and_keeps_raw_contacts(checker):
    result = query(checker, binding([rule()]))
    assert result["passed"]
    row = result["pairs"][0]
    assert row["signed_distance_m"] < 0 and row["contacts"]
    assert row["permission"] == "pad_to_target"
    assert row["contact_region_checks"][0]["method"] == "whole_convex_intersection_support"
    # A small material feature at the same nearest point cannot authorize the
    # rest of the same pad/target overlap outside that feature.
    narrow = rule(feature_points=[[-.01, -.01, -.001], [.01, -.01, -.001],
                                  [.01, .01, -.001], [-.01, .01, -.001]])
    rejected = query(checker, binding([narrow]))
    assert not rejected["passed"] and rejected["violations"]
    assert rejected["pairs"][0]["contact_region_checks"][0]["contained"] is False


# BEGIN an observed local bar may touch only part of a larger native pad.
def test_partial_target_patch_permits_only_the_actual_solid_intersection(checker):
    model = binding([rule(patch_half_size=[.015, .015])])
    model["owners"][0]["geometry"]["solids"][0]["dimensions"] = [.02, .02, .1]
    accepted = query(checker, model)
    assert accepted["passed"] and accepted["pairs"][0]["permission"] == "pad_to_target"
    # The same owner also has material outside this small permitted region.
    # A second complete target solid cannot borrow the first bar's contact.
    model["owners"][0]["geometry"]["solids"].append({"name": "outside_bar", "kind": "cuboid",
        "dimensions": [.01, .01, .1], "pose": [.03, 0., -.05, 1., 0., 0., 0.],
        "source": "independent second observed component"})
    rejected = query(checker, model)
    assert not rejected["passed"] and rejected["violations"]
    assert any(row.get("target_geometry") == "outside_bar" and row["permission"] is None for row in rejected["pairs"])
# END a larger feature is legal only where the whole physical overlap is local.


@pytest.mark.parametrize("change", ["material", "owner", "time", "patch", "depth"])
def test_wrong_material_owner_patch_time_or_depth_is_not_a_pair_exemption(checker, change):
    model = binding([rule()])
    if change == "material":
        model["native_geometry_materials"]["pad"] = "finger_back"
    elif change == "owner":
        model["contacts"][0]["target_owner"] = "other_target"
    elif change == "time":
        model["contacts"][0]["interval"] = [0., .4]
    elif change == "patch":
        model["contacts"][0]["patch_origin"] = [.3, 0., 0.]
    else:
        model["contacts"][0]["modeling_allowance_m"] = .0001
    result = query(checker, model)
    assert not result["passed"] and result["violations"]
    assert result["pairs"][0]["permission"] is None


def test_permission_and_required_contact_are_independent(checker):
    model = binding([rule(required=True)])
    model["owners"][0]["pose_path"]["poses"] = [rigid(-1.).tolist()]*2
    result = query(checker, model)
    assert result["pairs"][0]["relation"] == "separated"
    assert result["violations"][0]["kind"] == "independent_required_contact"


def test_required_contact_cannot_use_an_empty_part_of_the_declared_patch(checker):
    for solids in ([], [{"name": "remote_target", "kind": "cuboid", "dimensions": [.01]*3,
                         "pose": [.3, 0., -.005, 1., 0., 0., 0.], "source": "complete remote target"}]):
        model = binding([rule(required=True)])
        model["owners"][0]["geometry"]["solids"] = solids
        result = query(checker, model)
        assert not result["passed"]
        assert result["violations"][0]["reason"] == "required_target_occupancy_absent_from_finite_region"
        model["contacts"][0]["required"] = False
        assert query(checker, model)["passed"]


def test_unknown_conditional_attachment_is_not_an_automatic_holding_gate(checker):
    model = binding()
    owner = model["owners"][0]
    owner["moving"] = True
    owner["attachments"] = [{"interval": [0., 1.], "position_tool_owner": [0., 0., 0.],
        "quaternion_tool_owner": [1., 0., 0., 0.], "condition_id": "candidate_grasp", "release_condition_id": None}]
    result = query(checker, model)
    assert result["passed"] and not result["query_gaps"]
    assert not result["observed_scene_checked"]
    assert any(row["kind"] == "conditional_scene_collision" for row in result["unresolved_assumptions"])
    rejected = query(checker, model, condition={"candidate_grasp": False})
    assert any(row["kind"] == "current_scene_binding_invalid" for row in rejected["violations"])
    # A current observed target pose makes this an actual known pad conflict.
    actual = query(checker, model, owner_poses={"target": np.eye(4)})
    assert not actual["passed"] and actual["violations"][0]["kind"] == "independent_scene_collision"


# BEGIN observed negative volume and a zero set have distinct obligations.
def test_tsdf_negative_volume_detects_contained_source_without_a_zero_surface(checker):
    geometry = tsdf_geometry_from_samples(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        origin=[-.1, -.1, -.05], voxel_size=.2, name="observed_material", source="observed negative samples")
    model = binding()
    model["owners"][0]["geometry"] = geometry
    result = query(checker, model)
    assert not result["passed"] and result["violations"]
    assert geometry["field_summary"]["surface_components"] == 0
    assert any(row.get("target_tsdf_component", [None])[0] == "negative_cell" for row in result["pairs"])


@pytest.mark.parametrize("sample, observed, occupied", [(-.01, True, True), (0., True, True),
    (-0., True, True), (.01, True, False), (-.01, False, False)])
def test_isolated_tsdf_evidence_is_a_point_without_an_invented_voxel(checker, sample, observed, occupied):
    values = np.full((1, 1, 1), sample)
    geometry = tsdf_geometry_from_samples(values, np.full(values.shape, observed),
        origin=[.005, .005, .005], voxel_size=.01, name="isolated_sample", source="explicit observed sample")
    model = binding()
    model["owners"][0]["geometry"] = geometry
    result = query(checker, model)
    assert result["passed"] is not occupied
    assert geometry["field_summary"]["observed_samples"] == int(observed)
    # A point just outside the box stays outside; its old full sample cube
    # would have overlapped the box's lower z=-0.001 face.
    if occupied:
        separated = tsdf_geometry_from_samples(values, np.full(values.shape, observed),
            origin=[.005, .005, -.002], voxel_size=.01, name="isolated_sample", source="explicit observed sample")
        model["owners"][0]["geometry"] = separated
        assert query(checker, model)["passed"]
# END unknown values neither fill space nor supply a required target surface.


# BEGIN raw-distance reports retain their numeric schema without claiming a solid sign.
def test_separated_tsdf_surface_reports_distance_and_partial_observation(checker):
    values = np.broadcast_to([.05, -.05], (2, 2, 2))
    model = binding()
    model["owners"][0]["geometry"] = tsdf_geometry_from_samples(values, np.ones(values.shape, bool),
        origin=[-.05, -.05, .15], voxel_size=.1, name="observed_plane", source="observed plane samples")
    result = query(checker, model)
    assert result["passed"] and not result["complete_scene_geometry"]
    row = result["pairs"][0]
    assert row["signed_distance_m"] > .1
    assert row["diagnostic_only"] and row["distance_scope"] == "observed_zero_surface_only"
# END containment is checked independently by the signed field components.

# END counterexamples preserve actual physical violations and allowed uncertainty.

# BEGIN owner predictions preserve multi-phase time and actual observation scope.
def test_partial_observation_scope_survives_an_actual_owner_pose(checker):
    model = binding([rule()])
    model["owners"][0]["geometry_complete"] = False
    result = query(checker, model, owner_poses={"target": np.eye(4)})
    assert result["passed"] and not result["complete_scene_geometry"]
    assert {"owner": "target", "kind": "partial_observed_geometry"} in result["unresolved_assumptions"]


def test_global_phase_progress_is_not_silently_normalized(checker):
    model = binding([rule(interval=(1., 2.))])
    assert query(checker, model, progress=1.5)["passed"]


def test_same_candidate_release_pose_is_used_after_release(checker):
    model = binding([rule()])
    owner = model["owners"][0]
    owner["moving"] = True
    owner["attachments"] = [{"interval": [0., 1.], "position_tool_owner": [0., 0., 0.],
        "quaternion_tool_owner": [1., 0., 0., 0.], "condition_id": "held", "release_condition_id": "released"}]
    checker.bind_scene(model)
    checker._scene_release_predictions[("target", 0)] = {"pose": rigid(-1.), "time_s": .5}
    robot = checker.query(joint_names=checker.joint_names, joint_positions=[0., 0.], T_reference_base=np.eye(4), pairs=())
    result = checker.query_scene(robot, tool_pose=np.eye(4), progress=1.5, finger_positions=[0., 0.], state_kind="nominal")
    assert result["passed"] and not result["query_gaps"]
    assert result["pairs"][0]["relation"] == "separated"
    assert any(row["kind"] == "predicted_release_pose" for row in result["unresolved_assumptions"])


def test_current_owner_observation_cannot_override_a_refuted_release(checker):
    model = binding()
    model["owners"][0]["attachments"] = [{"interval": [0., 1.], "position_tool_owner": [0., 0., 0.],
        "quaternion_tool_owner": [1., 0., 0., 0.], "condition_id": "held", "release_condition_id": "released"}]
    result = query(checker, model, progress=1.5, condition={"released": False}, owner_poses={"target": rigid(-1.)})
    assert not result["passed"]
    assert any(row.get("reason") == "release_hypothesis_refuted" for row in result["violations"])
# END no reference pose or unconfirmed effect creates an actual Holding/release fact.

# BEGIN native installation aliases reuse exact solids without exempting payloads.
def test_explicit_native_alias_avoids_duplicate_self_collision_but_checks_payload(checker):
    model = binding()
    fixture = model["owners"][0]
    fixture["native_geometry_names"] = ["pad"]
    fixture["geometry"]["solids"] = [{"name": "installation_proxy", "kind": "cuboid", "dimensions": [.11]*3,
        "pose": [0., 0., .049, 1., 0., 0., 0.], "source": "outward planning proxy for exact native pad"}]
    # BEGIN a declared static owner is not a second moving robot actor.
    # Coincident static background must not introduce a new trajectory edge;
    # the same obstacle still applies to a moving payload below.
    model["background"]["geometry"]["solids"] = [{"name": "static_background", "kind": "cuboid",
        "dimensions": [.1]*3, "pose": [0., 0., .049, 1., 0., 0., 0.], "source": "static world geometry"}]
    # END no ESDF clipping or contact allowance resolves static ownership.
    first = query(checker, model)
    assert first["passed"] and not first["pairs"]
    payload = deepcopy(fixture)
    payload.update(owner_id="payload", moving=True, native_geometry_names=[])
    payload["geometry"]["solids"][0]["name"] = "carried_body"
    model["owners"].append(payload)
    second = query(checker, model, kind="nominal")
    assert not second["passed"]
    assert any(row["actor"] == "payload" and row["target_owner"] == "target"
               and row["signed_distance_m"] < 0 for row in second["pairs"])
    # Only an unchanged native placement can identify the same solid.
    fixture["geometry"]["solids"][0]["pose"][0] += .01
    moved = query(checker, model, kind="nominal")
    assert any(row["reason"] == "fixed_native_alias_pose_changed" for row in moved["query_gaps"])
# END alias identity never means all bodies may collide with this installation.


# BEGIN static ownership changes only explicit aliases, never robot obligations.
def test_nonalias_fixed_robot_base_still_collides_with_background(checker):
    model = binding()
    model["owners"] = []
    model["background"]["geometry"]["solids"] = [{"name": "background", "kind": "cuboid",
        "dimensions": [.1]*3, "pose": [0., 0., .049, 1., 0., 0., 0.], "source": "observed world obstacle"}]
    result = query(checker, model)
    assert not result["passed"]
    violation = result["violations"][0]
    assert violation["actor"] == "robot" and violation["target_owner"] == "background"
    assert violation["evidence"]["actor_geometry"] == "pad"
    assert violation["evidence"]["signed_distance_m"] < 0


def test_static_alias_retains_native_robot_installation_pair_gate(checker):
    from manipulation.robot_geometry_check import PairQuery
    from manipulation.robot_validation import RobotStateValidator

    arm_local = rigid(.049)
    arm_local[1, 3] = -.2  # The URDF's left slide origin is +.2 on y.
    robot = RobotGeometryChecker(checker.urdf_path, base_frame="base", geometries=[
        NativePrimitive("installation", "base", "box", (.1, .1, .1), rigid(.049), "native installation"),
        NativePrimitive("arm", "left", "box", (.1, .1, .1), arm_local, "native moving robot")])
    model = binding()
    model["native_geometry_materials"] = {}
    target = model["owners"][0]
    target["native_geometry_names"] = ["installation"]
    target["geometry"]["solids"] = [{"name": "installation_proxy", "kind": "cuboid", "dimensions": [.11]*3,
        "pose": [0., 0., .049, 1., 0., 0., 0.], "source": "outward planning installation proxy"}]
    gate = RobotStateValidator(robot, joint_names=robot.joint_names,
        arm_joint_names=robot.joint_names[:1], finger_joint_names=robot.joint_names[1:],
        pairs=[PairQuery("arm", "installation", "required", "native enabled robot/installation pair")],
        named_velocity_limits={name: 1. for name in robot.joint_names}, reference_frame="world",
        T_reference_base=np.eye(4), tool_frame="base", T_frame_tcp=np.eye(4),
        fk_translation_tolerance_m=1e-8, fk_rotation_tolerance_rad=1e-8, source="explicit native pair test")
    gate.bind_scene(model, progress_at_time=lambda t: t)
    feedback = {"joint_names": robot.joint_names, "arm_joint_names": robot.joint_names[:1],
        "finger_joint_names": robot.joint_names[1:], "q": [0., 0.], "dq": [0., 0.],
        "reference_frame": "world", "tool_link": "base", "tool_transform": np.eye(4),
        "base_pose": np.eye(4), "tool_pose": np.eye(4)}
    result = gate.check_state(feedback, stage="initial_state", trajectory_time=0.)
    assert not result["passed"]
    assert any(row["kind"] == "independent_robot_collision" for row in result["violations"])
    assert len(result["distances_m"]) == 1 and result["distances_m"][0] < 0
    rows = result["scene"]["pairs"]
    assert len(rows) == 1 and rows[0]["actor_geometry"] == "arm"
    assert rows[0]["target_geometry"] == "installation"
    assert rows[0]["native_pair_evidence"]["policy"] == "required"
    assert rows[0]["native_pair_evidence"]["signed_distance_m"] == result["distances_m"][0]
# END complete self-policy and every moving actor-to-installation edge survive.

# BEGIN exact endpoint permissions cannot hide a forbidden adjacent mode.
def test_independent_preflight_checks_both_contact_modes_at_the_same_state(checker):
    from manipulation.robot_validation import RobotStateValidator
    from test_robot_validation import trajectory

    gate = RobotStateValidator(checker, joint_names=checker.joint_names,
        arm_joint_names=checker.joint_names[:1], finger_joint_names=checker.joint_names[1:], pairs=(),
        named_velocity_limits={name: 1. for name in checker.joint_names}, reference_frame="world",
        T_reference_base=np.eye(4), tool_frame="base", T_frame_tcp=np.eye(4),
        fk_translation_tolerance_m=1e-8, fk_rotation_tolerance_rad=1e-8, source="independent boundary fixture")
    curve = trajectory([.03, .03], [0., 0.])
    # BEGIN the joint sample now includes analytic acceleration before progress.
    gate.bind_scene(binding([rule(interval=(.5, 1.))]), progress_at_time=lambda t: float(curve.sample(t)[3]))
    # END independent contact modes still consume the original progress clock.
    result = gate._nominal_scene_preflight(curve, np.asarray([.5]))
    exact = next(row for row in result["states"] if row["mode"] == "bound_progress")
    preceding = next(row for row in result["states"] if row["mode"] == "before_switch")
    assert exact["scene"]["passed"]
    assert not preceding["scene"]["passed"] and not result["passed"]
    assert exact["time_s"] == preceding["time_s"] == .5
    assert exact["scene"]["pairs"][0]["signed_distance_m"] == preceding["scene"]["pairs"][0]["signed_distance_m"]
    # A dwell at the boundary has old mode only at entry, new mode only at exit.
    curve.progress_u = np.asarray([0., .4, .6, 1.])
    curve.progress_s = np.asarray([0., .5, .5, 1.])
    assert [mode for _, mode in gate._scene_progress_modes(curve, .4)] == ["bound_progress", "before_switch"]
    assert [mode for _, mode in gate._scene_progress_modes(curve, .5)] == ["bound_progress"]
    assert [mode for _, mode in gate._scene_progress_modes(curve, .6)] == ["bound_progress", "after_switch"]
    curve.progress_u = curve.progress_u.astype(np.float32)
    curve.progress_s = curve.progress_s.astype(np.float32)
    boundary_times = gate._scene_query_times(curve, np.asarray([0., 1.]), 0., 1.)
    entry = float(curve.progress_u[1])
    assert entry in boundary_times
    assert [mode for _, mode in gate._scene_progress_modes(curve, entry)] == ["bound_progress", "before_switch"]
# END samples retain q/t/finger geometry while discrete laws change sides.

# BEGIN pair ordering and complete occupied-leaf localization are invariant.
def test_moving_owner_pair_permission_is_independent_of_owner_order(checker):
    far = np.eye(4)
    far[0, 3] = 3.
    robot = RobotGeometryChecker(checker.urdf_path, base_frame="base", geometries=[
        NativePrimitive("far", "base", "box", (.1, .1, .1), far, "unrelated native robot body")])
    model = binding()
    a = model["owners"][0]
    a.update(owner_id="A", moving=True)
    b = deepcopy(a)
    b["owner_id"] = "B"
    b["geometry"]["solids"] = [{"name": "pad", "kind": "cuboid", "dimensions": [.1]*3,
        "pose": [0., 0., .049, 1., 0., 0., 0.], "source": "native moving owner pad"}]
    model["owners"] = [a, b]
    model["native_geometry_materials"] = {"B/pad": "contact_pad"}
    model["contacts"] = [rule(actor="B", target_owner="A")]
    first = query(robot, model, kind="nominal")
    assert first["passed"]
    assert any(row.get("permission") == "pad_to_target" for row in first["pairs"])
    model["owners"].reverse()
    assert query(robot, model, kind="nominal")["passed"]
    model["contacts"][0]["required"] = True
    model["native_geometry_materials"].clear()
    incomplete = query(robot, model, kind="nominal")
    assert not incomplete["passed"]
    assert any(row["reason"] == "required_material_native_geometry_binding_missing" for row in incomplete["query_gaps"])


def test_all_intersecting_tsdf_components_must_fit_the_local_material_feature(checker):
    values = np.ones((7, 1, 1))
    observed = np.zeros(values.shape, bool)
    values[0, 0, 0] = values[5, 0, 0] = -.01
    observed[0, 0, 0] = observed[5, 0, 0] = True
    model = binding([rule()])
    model["owners"][0]["geometry"] = tsdf_geometry_from_samples(values, observed,
        origin=[-.005, -.005, -.0005], voxel_size=.01,
        name="observed_points", source="two disjoint observed negative samples")
    accepted = query(checker, model)
    assert accepted["passed"]
    assert accepted["pairs"][0]["component_summary"]["permitted"] == 2
    model["contacts"] = [rule(feature_points=[[-.015, -.015, -.001], [.015, -.015, -.001],
        [.015, .015, -.001], [-.015, .015, -.001]])]
    rejected = query(checker, model)
    assert not rejected["passed"] and rejected["violations"]
    assert any(check.get("contained") is False for row in rejected["pairs"] for check in row["contact_region_checks"])
# END no first witness or iteration order can suppress a second illegal contact.


# BEGIN allowed conditional collisions cannot hide a later mandatory query gap.
def test_conditional_component_collisions_continue_until_a_hard_query_failure(checker):
    cylinder = RobotGeometryChecker(checker.urdf_path, base_frame="base", geometries=[
        NativePrimitive("pad", "base", "cylinder", (.05, .1), rigid(.049), "native curved pad")])
    model = binding([rule()])
    owner = model["owners"][0]
    owner["moving"] = True
    owner["attachments"] = [{"interval": [0., 1.], "position_tool_owner": [0., 0., 0.],
        "quaternion_tool_owner": [1., 0., 0., 0.], "condition_id": "candidate_attachment",
        "release_condition_id": None}]
    owner["geometry"] = tsdf_geometry_from_samples(-np.ones((3, 1, 1)), np.ones((3, 1, 1), bool),
        origin=[0., 0., .049], voxel_size=.02475, name="observed_points", source="three observed negative points")
    result = query(cylinder, model)
    # The first two deep overlaps remain conditional. The last shallow one
    # attempts finite permission, which needs a complete curved-region adapter.
    # An early stop after either allowed hypothesis would incorrectly pass.
    assert not result["passed"]
    assert any(item["reason"] == "local_contact_region_query_incomplete" for item in result["query_gaps"])
    conditional = [item for item in result["unresolved_assumptions"] if item["kind"] == "conditional_scene_collision"]
    assert len(conditional) == 1 and conditional[0]["component_issue_count"] == 2
    summary = result["pairs"][0]["component_summary"]
    assert summary["tested"] == 3 and summary["stopped_after_pair_failure"]
# END one report slot summarizes all evaluated pieces without changing acceptance.


# BEGIN separated summaries must retain first ties and later decisive failures.
@pytest.mark.parametrize("interior_point", [False, True])
def test_separated_tsdf_summary_preserves_ties_and_later_collision(checker, interior_point):
    cylinder = RobotGeometryChecker(checker.urdf_path, base_frame="base", geometries=[
        NativePrimitive("pad", "base", "cylinder", (.05, .1), rigid(.049), "native curved pad")])
    values = -np.ones((3, 3, 1))
    observed = np.zeros(values.shape, bool)
    observed[0, 0, 0] = observed[0, 2, 0] = True
    observed[2, 0, 0] = observed[2, 2, 0] = True
    if interior_point:
        observed[0, 1, 0] = True
    model = binding()
    model["owners"][0]["geometry"] = tsdf_geometry_from_samples(values, observed,
        origin=[-.045, -.045, .049], voxel_size=.045,
        name="observed_points", source="equal-distance corners and optional interior point")
    result = query(cylinder, model)
    row, = result["pairs"]
    assert result["passed"] is (not interior_point)
    assert row["component_summary"]["tested"] == (2 if interior_point else 4)
    assert row["component_summary"]["permitted"] == 0
    assert row["component_summary"]["stopped_after_pair_failure"] is interior_point
    # The four corners are outside the round solid but inside its AABB.
    # Equal later minima keep the first corner. An inserted occupied point
    # inside the cylinder must still replace it and stop this rejected pair.
    assert row["target_tsdf_component"][1] == (1 if interior_point else 0)
    assert (row["signed_distance_m"] < 0.) is interior_point
    assert len(result["violations"]) == int(interior_point)
# END unused detail omission cannot turn a later intersecting piece into a pass.


# BEGIN rebinding resets all exact surface results before new shape identities.
def test_rebinding_scene_discards_surface_distance_results(checker):
    values = np.ones((2, 2, 2))
    values[:, :, 0] = -1.
    model = binding()
    model["owners"][0]["geometry"] = tsdf_geometry_from_samples(values,
        np.ones(values.shape, bool), origin=[1., 1., 1.], voxel_size=.01,
        name="distant_surface", source="observed surface for binding isolation")
    assert query(checker, model)["passed"]
    assert checker._scene_surface_distance_cache
    checker.bind_scene(model)
    assert checker._scene_surface_distance_cache == {}
# END numerically similar replacement scenes still receive fresh native queries.
