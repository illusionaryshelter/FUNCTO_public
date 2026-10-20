"""Scene integration of native model policy and observed task fields."""

# BEGIN use real native queries with a small analytic FK fixture.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.online_geometry import OnlinePlanningScene
from manipulation.observed_mapping import FrozenObservedField
from manipulation.planning_scene import ObservedSpace
from manipulation.planning_scene import CollisionOwner
from manipulation.planning_problem import PosePath
from manipulation.robot_geometry_check import NativePrimitive, PairQuery, RobotGeometryChecker
from test_robot_geometry_check import URDF


def make_scene(tmp_path, *, gap=.001, fixed=(), occupied_background=False, policy="required",
               second_pose=None):
    path = tmp_path / "robot.urdf"
    path.write_text(URDF)
    second = np.eye(4)
    second[0, 3] = .02 + gap
    if second_pose is not None:
        second = second_pose
    checker = RobotGeometryChecker(path, base_frame="base", geometries=(
        NativePrimitive("first", "base", "box", (.02, .02, .02), np.eye(4), "source"),
        NativePrimitive("second", "base", "box", (.02, .02, .02), second, "source")))
    background = FrozenObservedField(-np.ones((2, 2, 2)),
        np.full((2, 2, 2), occupied_background), weights=np.ones((2, 2, 2)),
        origin=[-.005, -.005, -.005], voxel_size=.01, name="background", source="RGB-D")
    space = ObservedSpace(np.zeros((2, 2, 2), bool), origin=[0, 0, 0], voxel_size=.01,
        observation_ids=("view",), reference_frame="world")
    scene = OnlinePlanningScene(background=background, owners=(), coverage=space,
        reference_frame="world", robot_model=checker, fixed_geometry_names=fixed, tool_geometry_names=(),
        has_gripper=False, unknown_policy={"conditional_regions": [], "outside_regions": "forbid"})
    pair = PairQuery("first", "second", policy, "source native pair")
    scene.robot_pairs = (pair,)
    scene.robot_validator = SimpleNamespace(checker=checker, T_reference_base=np.eye(4))

    class BaseKinematics:
        def poses(self, q, names):
            zero = q[..., :1] * 0
            return {"base": (zero.expand(*q.shape[:-1], 3),
                             torch.cat((zero + 1, zero.expand(*q.shape[:-1], 3)), -1))}

    scene.kinematics = BaseKinematics()
    return scene


def evaluate(scene, role="acceptance"):
    q = torch.tensor([[[.04, -.02]]], dtype=torch.float64, requires_grad=True)
    return scene.components(None, torch.zeros((1, 1, 3), dtype=torch.float64),
        torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64),
        torch.zeros((1, 1), dtype=torch.float64), q, ("left_slide", "right_slide"),
        trajectory_time=torch.zeros((1, 1)), query_role=role)


def test_native_structure_gap_is_independent_of_task_acceptance_margin(tmp_path):
    scene = make_scene(tmp_path)
    for role in ("optimization", "acceptance"):
        costs = evaluate(scene, role)
        assert costs["collision/robot/self"].item() == 0
        assert costs["unresolved/robot/self"].item() == 0
        assert scene.last_query_report["status_counts"]["separated"] == 1
        assert not any("coverage" in key or "unknown" in key for key in costs)


# BEGIN optimization and acceptance share task-field nonpenetration semantics.
@pytest.mark.parametrize("gap,status", [(1e-3, "clear"), (1e-7, "clear"),
                                      (0., "unresolved"), (-1e-3, "intersection")])
def test_task_field_gap_and_boundary_have_the_same_meaning_in_both_roles(tmp_path, gap, status):
    from manipulation.implicit_distance import ObservedField
    scene = make_scene(tmp_path)
    scene.background = ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
                                     origin=[-.005, .01 + gap, -.005], voxel_size=.01)
    for role in ("optimization", "acceptance"):
        costs = evaluate(scene, role)
        assert (sum(costs.values()).item() > 0) == (status == "intersection")
        assert scene.last_query_report["status_counts"].get(status, 0) > 0
        assert scene.last_query_report["collision_rule"] == "nonpenetration"
        if status == "unresolved":
            assert any(row["status"] == "unresolved"
                       for row in scene.last_query_report["nonclear_pairs"])
# END numerical uncertainty is visible without creating a clearance gate.


@pytest.mark.parametrize("gap,collision,unresolved", [(-.001, True, False), (0., False, True)])
def test_self_penetration_and_numerical_boundary_remain_distinct(tmp_path, gap, collision, unresolved):
    costs = evaluate(make_scene(tmp_path, gap=gap))
    assert (costs["collision/robot/self"].item() > 0) == collision
    assert (costs["unresolved/robot/self"].item() > 0) == unresolved


def test_mechanical_contact_policy_does_not_become_a_required_pair(tmp_path):
    costs = evaluate(make_scene(tmp_path, gap=-.001, policy="mechanical_contact"))
    assert costs["collision/robot/self"].item() == 0
    assert costs["unresolved/robot/self"].item() == 0


def test_fixed_installation_role_is_explicit_even_with_constant_link_fk(tmp_path):
    robot = make_scene(tmp_path, occupied_background=True)
    assert evaluate(robot)["collision/robot/background"].item() > 0
    fixed = make_scene(tmp_path, occupied_background=True, fixed=("first", "second"))
    assert "collision/robot/background" not in evaluate(fixed)


def test_scene_record_retains_model_policy_and_calibration(tmp_path):
    scene = make_scene(tmp_path, fixed=("second",))
    saved = scene.to_execution_binding()
    assert saved["fixed_geometry_names"] == ["second"]
    assert saved["robot_model"]["base_frame"] == "base"
    assert saved["robot_model"]["T_reference_base"] == np.eye(4).tolist()
    assert saved["robot_model"]["pairs"][0]["policy"] == "required"
    assert saved["robot_model"]["geometries"][1]["T_frame_geometry"][0][3] == .021
    assert "robot_bodies" not in saved


def test_primitive_fixed_rotation_and_translation_are_applied_once(tmp_path, monkeypatch):
    from scipy.spatial.transform import Rotation
    from manipulation.online_geometry import _matrices
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [.3, -.7, .2]).as_matrix()
    transform[:3, 3] = [.2, .3, .4]
    scene = make_scene(tmp_path, second_pose=transform)
    consumed = []

    def inspect(first, first_pose, first_matrix, second, second_pose, second_matrix, **kwargs):
        consumed.append(first_matrix)
        np.testing.assert_allclose(_matrices(first_pose), first_matrix, atol=1e-14)
        return first_pose[0].sum() * 0, {"status": "clear"}

    monkeypatch.setattr("manipulation.online_geometry.pair_residual", inspect)
    evaluate(scene)
    reference = scene.robot_model.query(joint_names=("left_slide", "right_slide"),
        joint_positions=(.04, -.02), T_reference_base=np.eye(4), pairs=())
    np.testing.assert_allclose(consumed[1], reference["geometry_poses"]["second"], atol=1e-14)


def test_fixed_installation_still_constrains_a_moving_observed_owner(tmp_path):
    scene = make_scene(tmp_path, fixed=("first", "second"))
    field = FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[-.005] * 3, voxel_size=.01,
        name="moving", source="observed task entity")
    scene.owners = (CollisionOwner("moving", field,
        PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared path"),
        "observed task entity", moving=True),)
    costs = evaluate(scene)
    assert costs["collision/robot/moving"].item() > 0
    rows = scene.last_query_report["nonclear_pairs"]
    assert any(row["first"] == "first" and row["second"] == "moving" for row in rows)
# END no rendered robot pixels participate in the native model's validity.
