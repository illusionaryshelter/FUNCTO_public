"""Small CPU regressions for the independent native-solid robot diagnostic."""

# BEGIN independent geometry diagnostics without CUDA, physics or scene state.
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation.robot_geometry_check import (
    NativePrimitive, NativeSolid, PairQuery, RobotGeometryChecker,
)


URDF = """<robot name="two_independent_slides">
  <link name="base"/><link name="left"/><link name="right"/><link name="tool"/>
  <joint name="left_slide" type="prismatic"><parent link="base"/><child link="left"/>
    <origin xyz="0 0.2 0"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="0.1" effort="10" velocity="1"/>
  </joint>
  <joint name="right_slide" type="prismatic"><parent link="base"/><child link="right"/>
    <origin xyz="0 -0.2 0"/><axis xyz="1 0 0"/>
    <limit lower="-0.1" upper="0" effort="10" velocity="1"/>
  </joint>
  <joint name="tool_fixed" type="fixed"><parent link="base"/><child link="tool"/>
    <origin xyz="0 0 0.3" rpy="0 0 1.5707963267948966"/>
  </joint>
</robot>"""


def corners(half=0.02):
    return half * np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])


@pytest.fixture
def urdf(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text(URDF)
    return path


def checker(path, extra=()):
    return RobotGeometryChecker(path, base_frame="base", geometries=[
        NativeSolid("left_shape", "left", corners(), "native left box corners"),
        NativeSolid("right_shape", "right", corners(), "native right box corners"), *extra])


def query(model, **changes):
    args = dict(joint_names=["left_slide", "right_slide"], joint_positions=[0.04, -0.02],
                T_reference_base=np.eye(4), frames=["tool"],
                pairs=[PairQuery("left_shape", "right_shape", "required", "explicit test pair")])
    args.update(changes)
    return model.query(**args)


def test_named_order_and_asymmetric_finger_coordinates(urdf):
    model = checker(urdf)
    normal = query(model)
    shuffled = query(model, joint_names=["right_slide", "left_slide"], joint_positions=[-0.02, 0.04])
    assert normal["frame_poses"] == shuffled["frame_poses"]
    assert normal["pairs"] == shuffled["pairs"]
    np.testing.assert_allclose(np.asarray(normal["frame_poses"]["left"])[:3, 3], [0.04, 0.2, 0])
    np.testing.assert_allclose(np.asarray(normal["frame_poses"]["right"])[:3, 3], [-0.02, -0.2, 0])
    changed = query(model, joint_positions=[0.08, -0.02])
    assert normal["frame_poses"]["right"] == changed["frame_poses"]["right"]
    assert normal["frame_poses"]["left"] != changed["frame_poses"]["left"]


@pytest.mark.parametrize("names,positions", [
    (["left_slide"], [0.04]),
    (["left_slide", "left_slide"], [0.04, 0.04]),
    (["left_slide", "unknown"], [0.04, 0.0]),
    (["left_slide", "right_slide"], [0.04, np.nan]),
])
def test_incomplete_or_nonfinite_coordinates_rejected(urdf, names, positions):
    with pytest.raises(ValueError):
        query(checker(urdf), joint_names=names, joint_positions=positions)


def test_world_base_fixed_frame_and_local_geometry_applied_once(urdf):
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    local = np.eye(4)
    local[:3, :3], local[:3, 3] = rotation, [0.1, 0.02, 0.03]
    vertices = corners() * [1., 2., 3.]
    baked = vertices @ rotation.T + local[:3, 3]
    model = checker(urdf, [
        NativeSolid("baked", "left", baked, "compiled vertices and geom transform"),
        NativePrimitive("analytic", "left", "box", (0.04, 0.08, 0.12), local, "native box"),
    ])
    world = np.eye(4)
    world[:3, :3], world[:3, 3] = rotation, [2., -1., 0.5]
    result = query(model, T_reference_base=world, pairs=[
        PairQuery("baked", "right_shape", "diagnostic", "equivalent geometry"),
        PairQuery("analytic", "right_shape", "diagnostic", "equivalent geometry")])
    assert result["pairs"][0]["signed_distance_m"] == pytest.approx(
        result["pairs"][1]["signed_distance_m"], abs=2e-6)
    expected_tool = np.eye(4)
    expected_tool[:3, :3], expected_tool[:3, 3] = rotation, [0, 0, 0.3]
    np.testing.assert_allclose(result["frame_poses"]["tool"], world @ expected_tool, atol=1e-12)
    np.testing.assert_allclose(result["geometry_poses"]["analytic"],
                               np.asarray(result["frame_poses"]["left"]) @ local, atol=1e-12)


def test_nested_solids_report_penetration_even_without_surface_crossing(urdf):
    model = checker(urdf, [NativeSolid("outer", "base", corners(1.), "native closed convex volume")])
    row = query(model, pairs=[PairQuery("outer", "right_shape", "diagnostic", "containment regression")])["pairs"][0]
    assert row["signed_distance_m"] < -0.5
    assert row["relation"] == "penetrating"
    assert row["witness_identity_error_m"] < 1e-10


def test_cylinder_remains_analytic_and_local_z_axis_rotates(urdf):
    local = np.eye(4)
    local[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    far = np.eye(4)
    far[:3, 3] = [0.5, 0, 0]
    model = checker(urdf, [
        NativePrimitive("cylinder", "base", "cylinder", (0.1, 0.4), local, "native cylinder"),
        NativePrimitive("box", "base", "box", (0.1, 0.1, 0.1), far, "native box")])
    row = query(model, pairs=[PairQuery("cylinder", "box", "required", "assembly pair")])["pairs"][0]
    assert row["signed_distance_m"] == pytest.approx(0.25, abs=2e-6)


def test_near_tolerance_is_reported_without_waiving_collision(urdf):
    displaced = np.eye(4)
    displaced[0, 3] = 2. + 0.25e-6
    model = checker(urdf, [
        NativePrimitive("a", "base", "box", (2., 2., 2.), np.eye(4), "native box"),
        NativePrimitive("b", "base", "box", (2., 2., 2.), displaced, "native box")])
    result = query(model, pairs=[PairQuery("a", "b", "required", "explicit numerical boundary")])
    assert result["pairs"][0]["relation"] == "numerical_boundary"
    # The known geometric gap is smaller than Coal's default solver tolerance.
    # Its returned value can be zero; asserting sub-tolerance accuracy or a
    # positive sign would contradict the diagnostic's numerical-boundary rule.
    band = max(result["numerics"]["gjk_tolerance"], result["numerics"]["epa_tolerance"])
    assert abs(result["pairs"][0]["signed_distance_m"]) <= band
    assert result["numerics"]["tolerance_is_error_bound"] is False


def test_observed_limit_violation_is_preserved_not_clipped(urdf):
    result = query(checker(urdf), joint_positions=[0.12, -0.02])
    assert result["joint_positions"][0] == 0.12
    assert set(result["joint_limit_violations"]) == {"left_slide"}
    assert np.asarray(result["frame_poses"]["left"])[0, 3] == 0.12


def test_explicit_pair_policy_never_suppresses_raw_geometry(urdf):
    model = checker(urdf)
    required = query(model)["pairs"][0]
    excluded = query(model, pairs=[PairQuery("left_shape", "right_shape", "excluded", "native exclusion record")])["pairs"][0]
    assert excluded["signed_distance_m"] == required["signed_distance_m"]
    assert excluded["policy_source"] == "native exclusion record"
    assert query(model, pairs=[])["pairs"] == []
    with pytest.raises(ValueError, match="Duplicate"):
        query(model, pairs=[PairQuery("left_shape", "right_shape", "required", "a"),
                            PairQuery("right_shape", "left_shape", "excluded", "b")])
    with pytest.raises(ValueError, match="Unknown URDF frame"):
        query(model, frames=["unknown"])
    with pytest.raises(ValueError, match="Pair endpoints"):
        query(model, pairs=[PairQuery("left_shape", "unknown", "required", "a")])


def test_wrong_base_reflection_and_degenerate_solid_rejected(urdf):
    with pytest.raises(ValueError, match="unique fixed URDF root"):
        RobotGeometryChecker(urdf, base_frame="left", geometries=[])
    reflected = np.eye(4)
    reflected[0, 0] = -1
    with pytest.raises(ValueError, match="proper rigid"):
        query(checker(urdf), T_reference_base=reflected)
    with pytest.raises(ValueError, match="full-dimensional"):
        checker(urdf, [NativeSolid("plane", "base", np.zeros((4, 3)), "invalid planar surface")])
# END independent geometry diagnostics without CUDA, physics or scene state.


# BEGIN source clearance summaries preserve per-component policy and query scope.
def _clearance_checker(urdf, *, rotated=False, installation=False):
    from test_independent_scene_validation import rigid
    pose = rigid(.049)
    if rotated:
        angle = np.pi/4
        pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.],
                        [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]
    shapes = [NativePrimitive("pad", "base", "box", (.1, .1, .1), pose, "native source pad")]
    if installation:
        shapes.append(NativePrimitive("installation", "base", "box", (.1, .1, .1),
                                      rigid(-.06), "native source installation"))
    return RobotGeometryChecker(urdf, base_frame="base", geometries=shapes)


def test_clearance_keeps_unpermitted_piece_when_the_summary_representative_is_permitted(urdf):
    from manipulation.tsdf_geometry import tsdf_geometry_from_samples
    from test_independent_scene_validation import binding, query as scene_query, rule
    model = binding([rule()])
    values, observed = -np.ones((2, 2, 1)), np.zeros((2, 2, 1), bool)
    observed[0, 0, 0] = observed[1, 1, 0] = True
    model["owners"][0]["geometry"] = tsdf_geometry_from_samples(values, observed,
        origin=[0., 0., -.0005], voxel_size=.06, name="two_points",
        source="permitted underside point and separated corner point")
    result = scene_query(_clearance_checker(urdf, rotated=True), model)
    assert result["passed"]
    row, = result["pairs"]
    assert row["permission"] == "pad_to_target" and row["signed_distance_m"] < 0
    assert row["component_summary"]["tested"] == 2
    clearance = result["unpermitted_clearance"]
    assert clearance["measured_count"] == 1
    assert clearance["minimum_recorded_distance_m"] == pytest.approx(np.sqrt(2)*.06-.05, abs=2e-6)
    assert clearance["complete_query"] and not clearance["distance_complete"]


@pytest.mark.parametrize("policy,count", [("required", 1), ("excluded", 0), ("mechanical_contact", 0)])
def test_clearance_includes_only_required_native_installation_pairs(urdf, policy, count):
    from test_independent_scene_validation import binding
    model, native = binding(), _clearance_checker(urdf, installation=True)
    owner = model["owners"][0]
    owner["native_geometry_names"] = ["installation"]
    owner["geometry"]["solids"][0].update(dimensions=[.1]*3, pose=[0., 0., -.06, 1., 0., 0., 0.])
    native.bind_scene(model)
    state = query(native, joint_positions=[0., 0.], pairs=[
        PairQuery("pad", "installation", policy, "explicit native source pair policy")])
    result = native.query_scene(state, tool_pose=np.eye(4), progress=.5, finger_positions=[0., 0.],
                                state_kind="actual")
    clearance = result["unpermitted_clearance"]
    assert clearance["complete_query"] and clearance["distance_complete"]
    assert clearance["measured_count"] == count
    if count:
        assert clearance["minimum_recorded_distance_m"] == pytest.approx(.009, abs=2e-6)
        assert clearance["minimum_pair"]["distance_scope"] == "native_required_pair"
    else:
        assert clearance["minimum_recorded_distance_m"] is None


def test_early_rejection_and_query_gaps_cannot_claim_complete_clearance(urdf):
    from manipulation.tsdf_geometry import tsdf_geometry_from_samples
    from test_independent_scene_validation import binding, query as scene_query
    model = binding()
    model["owners"][0]["geometry"] = tsdf_geometry_from_samples(-np.ones((3, 1, 1)),
        np.ones((3, 1, 1), bool), origin=[0., 0., -.0005], voxel_size=.01,
        name="inside_points", source="negative source samples")
    result = scene_query(_clearance_checker(urdf), model)
    assert not result["passed"]
    summary = result["unpermitted_clearance"]
    assert not summary["complete_query"] and not summary["distance_complete"]
    assert "component_enumeration_stopped_after_rejection" in summary["limitations"]
    model = binding()
    model["background"]["geometry"]["query_gaps"] = ["missing required source geometry"]
    result = scene_query(_clearance_checker(urdf), model)
    assert not result["unpermitted_clearance"]["complete_query"]
    assert "required_query_gaps" in result["unpermitted_clearance"]["limitations"]


def test_surface_measurement_never_claims_global_distance_or_infers_patch_permission(urdf):
    import json
    from manipulation.tsdf_geometry import tsdf_geometry_from_samples
    from test_independent_scene_validation import binding, query as scene_query, rule
    values = np.broadcast_to([.05, -.05], (2, 2, 2))
    model = binding()
    model["owners"][0]["geometry"] = tsdf_geometry_from_samples(values, np.ones(values.shape, bool),
        origin=[-.05, -.05, .15], voxel_size=.1, name="plane", source="observed plane samples")
    result = scene_query(_clearance_checker(urdf), model)
    summary = result["unpermitted_clearance"]
    assert result["passed"] and summary["complete_query"] and not summary["distance_complete"]
    assert summary["minimum_recorded_distance_m"] > .1
    assert summary["minimum_pair"]["distance_scope"] == "observed_zero_surface_only"
    json.dumps(summary, allow_nan=False)
    model["contacts"] = [rule()]
    result = scene_query(_clearance_checker(urdf), model)
    summary = result["unpermitted_clearance"]
    assert result["passed"] and summary["complete_query"]
    assert summary["minimum_recorded_distance_m"] is None
    assert summary["omitted_ambiguous_surface_count"] == 1


def test_pad_patch_cannot_suppress_an_explicitly_different_arm_material_distance(urdf):
    from manipulation.tsdf_geometry import tsdf_geometry_from_samples
    from test_independent_scene_validation import binding, query as scene_query, rigid, rule
    native = RobotGeometryChecker(urdf, base_frame="base", geometries=[
        NativePrimitive("pad", "base", "box", (.1, .1, .1), rigid(.049), "native contact pad"),
        NativePrimitive("arm", "base", "box", (.05, .05, .05), rigid(.049), "native arm body")])
    model = binding([rule()])
    model["native_geometry_materials"]["arm"] = "arm_surface"
    values = np.broadcast_to([.05, -.05], (2, 2, 2))
    model["owners"][0]["geometry"] = tsdf_geometry_from_samples(values, np.ones(values.shape, bool),
        origin=[-.05, -.05, .15], voxel_size=.1, name="plane", source="observed plane samples")
    result = scene_query(native, model)
    summary = result["unpermitted_clearance"]
    assert result["passed"] and summary["complete_query"]
    assert summary["measured_count"] == 1 and summary["omitted_ambiguous_surface_count"] == 1
    assert summary["minimum_pair"]["actor_geometry"] == "arm"
    assert summary["minimum_recorded_distance_m"] == pytest.approx(.126, abs=2e-6)
    # A removed binding is unknown, not evidence that no pad rule can apply.
    del model["native_geometry_materials"]["arm"]
    uncertain = scene_query(native, model)["unpermitted_clearance"]
    assert uncertain["minimum_recorded_distance_m"] is None
    assert uncertain["omitted_ambiguous_surface_count"] == 2


def test_analytic_finite_patch_does_not_measure_clearance_to_the_unpermitted_remainder(urdf):
    from test_independent_scene_validation import binding, query as scene_query, rule
    result = scene_query(_clearance_checker(urdf), binding([rule()]))
    assert result["passed"] and result["complete_scene_geometry"]
    summary = result["unpermitted_clearance"]
    assert summary["complete_query"] and not summary["distance_complete"]
    assert summary["minimum_recorded_distance_m"] is None
    assert summary["omitted_finite_patch_subset_count"] == 1
    assert "finite_patch_unpermitted_subset_distance_unmeasured" in summary["limitations"]
# END none of these statistics bypasses the original source collision verdict.
