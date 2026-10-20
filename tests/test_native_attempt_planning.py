"""CPU regressions for nominal objectives and basic-motion acceptance.

The actual cuRobo collections and rollout dispatch execute on CPU. Analytic
FK replaces only the GPU geometry kernel; these tests do not claim robot
reachability, task success, or physical collision accuracy.
"""

# BEGIN native objective/constraint separation and bounded cubic regression.
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manipulation.cubic_seed import CubicSeed, cubic_position_report
from manipulation.planning import (
    MotionPlanningError, PlanningBudget, _CuroboPlanner, _ProgressMap,
    _minimum_cost_candidate,
)


        # END no extra query or new physical threshold is needed to classify.


def test_candidate_selection_uses_total_nominal_cost_and_rejects_nonfinite_values():
    assert _minimum_cost_candidate([80., 2.], [True, True]) == 1
    assert _minimum_cost_candidate([.1, 2.], [False, True]) == 1
    assert _minimum_cost_candidate([np.nan, 2.], [True, True]) == 1
    assert _minimum_cost_candidate([1., np.inf], [True, True]) == 0
    with pytest.raises(MotionPlanningError, match="no finite objective"):
        _minimum_cost_candidate([np.nan, np.inf], [True, True])






def test_ompl_connection_preserves_release_history_for_later_single_state_queries():
    from unittest.mock import Mock
    from curobo.types import DeviceCfg, JointState
    from manipulation.online_geometry import FrozenESDF, FrozenPrimitives, OnlinePlanningScene
    from manipulation.planning import JointTrajectory
    from manipulation.planning_problem import PosePath
    from manipulation.planning_scene import Attachment, CollisionOwner, ObservedSpace

    background = FrozenESDF(np.full((3, 3, 3), 10., np.float32), origin=[-2.]*3,
        voxel_size=2., owner_ids=("background",), source="empty ordinary model")
    owner = CollisionOwner("payload", FrozenPrimitives(primitives=[{"kind": "sphere", "dimensions": [.02]}],
        owner_ids=("payload",), source="payload model"), PosePath(np.array([0., 1.]),
        np.stack((np.eye(4), np.eye(4))), "current owner"), "current owner", True,
        local_spheres=[[0., 0., 0., .02]], sphere_materials=("body",), attachments=(
            Attachment((0., .5), [0., 0., 0.], [1., 0., 0., 0.], "hold", "release"),))
    coverage = ObservedSpace(np.zeros((3, 3, 3), bool), origin=[-2.]*3, voxel_size=2.,
        observation_ids=("analytic",), reference_frame="world")
    scene = OnlinePlanningScene(background=background, owners=(owner,), coverage=coverage,
        reference_frame="world", robot_sphere_materials=("body",),
        robot_geometry_source="analytic robot", has_gripper=False)
    body = np.array([[1., 1., 1., .02]])
    # BEGIN register the same Cartesian kinematics used by the planner below.
    # The robot body is fixed on base; x/y/z move only the tool and payload.
    # A zero-DOF native fixture would contradict the declared planning joints.
    urdf = '<robot name="cartesian"><link name="base"/>'
    for name, parent, child, axis in (("x", "base", "lx", "1 0 0"),
            ("y", "lx", "ly", "0 1 0"), ("z", "ly", "tool", "0 0 1")):
        urdf += (f'<link name="{child}"/><joint name="{name}" type="prismatic">'
            f'<parent link="{parent}"/><child link="{child}"/><axis xyz="{axis}"/>'
            '<limit lower="-2" upper="2" effort="10" velocity="10"/></joint>')
    urdf += '</robot>'
    body_pose = np.eye(4); body_pose[:3, 3] = body[0, :3]
    scene.bind_online_robot(
        robot_sphere_geometry=[{"link_name": "base", "sphere_indices": [0],
            "centers_local": body[:, :3].tolist(), "radii": body[:, 3].tolist()}],
        native_robot_model={"urdf_xml": urdf, "geometries": [{"name": "body", "frame": "base",
            "kind": "sphere", "dimensions": [.02], "T_frame_geometry": body_pose.tolist(),
            "source": "explicit fixed robot body"}]}, native_joint_names=("x", "y", "z"),
        native_self_pairs=[], robot_model={"T_reference_base": np.eye(4).tolist(),
            "tool_frame": "tool", "T_frame_tcp": np.eye(4).tolist()})
    # END release-history checks still use the original independent owner.
    assert not scene.query_state(body, np.eye(4), .75,
        joint_position=np.zeros(3), joint_names=("x", "y", "z"))["passed"]

    def kinematics(joints):
        q = joints.position
        rotation = q.new_zeros((*q.shape[:2], 1, 4)); rotation[..., 0] = 1.
        return SimpleNamespace(tool_poses=SimpleNamespace(tool_frames=["tool"],
            position=q[..., None, :], quaternion=rotation),
            robot_spheres=q.new_tensor(body).expand(*q.shape[:2], 1, 4).clone())

    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.torch, planner.device_cfg, planner.JointState = torch, DeviceCfg("cpu"), JointState
    planner.joint_names, planner.scene = ("x", "y", "z"), scene
    planner.problem = SimpleNamespace(tool_link="tool", transform_tool_pose=lambda p, r: (p, r),
                                      transform_robot_spheres=lambda value: value)
    planner.check_request = Mock()
    planner.budget = PlanningBudget(search_seconds=1.)
    planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    planner.formal_clock = {"duration": 1.}
    planner._verification_fk = SimpleNamespace(compute_kinematics=kinematics)
    planner._RobotState = lambda **kwargs: SimpleNamespace(**kwargs,
        robot_spheres=kwargs["cuda_robot_model_state"].robot_spheres)
    limits = SimpleNamespace(joint_names=list(planner.joint_names),
                              position=torch.tensor([[-2.]*3, [2.]*3]))
    planner.joint_limits = limits
    planner.seed_curve = CubicSeed(torch.zeros((1, 3)), torch.zeros((1, 3)), .1, 8)
    trajectory = JointTrajectory(planner.joint_names, np.array([0., 1.]),
        np.array([[0., 0., 0.], [.4, 0., 0.]]), np.array([[.4, 0., 0.], [.4, 0., 0.]]),
        np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    seeds, report = planner.ompl_seed(trajectory)
    assert seeds is not None, report
    assert len(report["stages"]) == 2 and report["state_queries"] > 0
    assert scene._fixed_release_poses == {}, "Full optimization must retain variable release poses"


def test_cubic_seed_fits_reference_without_certifying_it_and_preserves_entrance():
    from scipy.interpolate import CubicHermiteSpline
    from test_cubic_trajectory import independent_curve
    q0, dq0 = torch.tensor([[.2, -.3]]), torch.tensor([[.1, -.2]])
    seed = CubicSeed(q0, dq0, .4, 8)
    u = np.linspace(0., 1., 20)
    target = np.column_stack((2*u, -u))
    controls, report = seed.fit(u, target)
    assert report["method"] == "least_squares" and torch.isfinite(controls).all()
    fitted = independent_curve(q0[0].double().numpy(), dq0[0].double().numpy(),
                               .4, controls.double().numpy())
    np.testing.assert_allclose(fitted(0.), q0[0].numpy(), rtol=0., atol=1e-12)
    np.testing.assert_allclose(fitted(0., 1)/.4, dq0[0].numpy(), rtol=0., atol=1e-12)
    # Actual delivered motion, unlike a numerical guess, is checked against
    # explicitly supplied physical limits at its analytic interior extrema.
    curve = CubicHermiteSpline([0., 1.], [[0.], [0.]], [[5.], [-5.]])
    report = cubic_position_report(curve, np.array([-1.]), np.array([1.]))
    assert report["maximum_bound_excess"] == pytest.approx(.25)
# END tests exercise semantic decisions, native dispatch and independent algebra.
