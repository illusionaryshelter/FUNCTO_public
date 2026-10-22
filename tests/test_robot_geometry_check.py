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


# BEGIN schema reuse preserves fresh numerical work and original validation.
def test_repeated_schema_reuses_names_and_frames_but_queries_each_new_state(urdf, monkeypatch):
    from unittest.mock import Mock
    import manipulation.robot_geometry_check as module

    model = checker(urdf)
    frame_id, name = model._frame_id, module._name
    frames = Mock(wraps=frame_id)
    names = Mock(wraps=name)
    monkeypatch.setattr(model, "_frame_id", frames)
    monkeypatch.setattr(module, "_name", names)
    guard = Mock()
    first = query(model, check_request=guard)
    schema = model._query_schema
    assert frames.call_count > 0 and names.call_count > 0
    assert guard.call_count == 1 + len(first["pairs"])
    frames.reset_mock(); names.reset_mock(); guard.reset_mock()

    # query() constructs new equal PairQuery records and a new frame list.
    # Equal declared values reuse metadata while the actual q changes.
    second = query(model, joint_positions=[.08, -.03], check_request=guard)
    assert model._query_schema is schema
    frames.assert_not_called()
    names.assert_not_called()
    assert guard.call_count == 1 + len(second["pairs"])
    assert second["joint_positions"] == [.08, -.03]
    assert second["geometry_poses"] != first["geometry_poses"]
    assert second["pairs"] != first["pairs"]

    # Original numeric validation runs even for an identical request schema.
    with pytest.raises(ValueError, match="finite"):
        query(model, joint_positions=[np.nan, -.03])
    reflection = np.eye(4)
    reflection[0, 0] = -1
    with pytest.raises(ValueError, match="proper rigid"):
        query(model, T_reference_base=reflection)
    assert model._query_schema is schema


def test_schema_changes_keep_pair_policy_source_frame_and_named_order_validation(urdf):
    from unittest.mock import Mock

    model = checker(urdf)
    reference = query(model)
    original_schema = model._query_schema
    invalid = [
        ({"pairs": [PairQuery("left_shape", "unknown", "required", "source")]}, "Pair endpoints"),
        ({"pairs": [PairQuery("left_shape", "right_shape", "unsupported", "source")]}, "Pair policy"),
        ({"pairs": [PairQuery("left_shape", "right_shape", "required", "")]}, "pair policy source"),
        ({"pairs": [PairQuery("left_shape", "right_shape", "required", "a"),
                    PairQuery("right_shape", "left_shape", "excluded", "b")]}, "Duplicate"),
        ({"frames": ["unknown"]}, "Unknown URDF frame"),
        ({"joint_names": ["left_slide", "left_slide"]}, "every model joint exactly once")]
    for changed, message in invalid:
        guard = Mock()
        with pytest.raises(ValueError, match=message):
            query(model, **changed, check_request=guard)
        assert model._query_schema is original_schema
        guard.assert_called_once_with()

    # The same endpoints with changed policy/source form a distinct schema.
    changed = query(model, pairs=[PairQuery("left_shape", "right_shape", "excluded", "new explicit source")])
    assert model._query_schema is not original_schema
    assert changed["pairs"][0]["policy"] == "excluded"
    assert changed["pairs"][0]["policy_source"] == "new explicit source"
    assert changed["pairs"][0]["signed_distance_m"] == reference["pairs"][0]["signed_distance_m"]
    shuffled = query(model, joint_names=["right_slide", "left_slide"], joint_positions=[-.02, .04])
    assert shuffled["geometry_poses"] == reference["geometry_poses"]
    assert shuffled["pairs"] == reference["pairs"]
# END the bounded cache owns one complete schema and no state-dependent values.
