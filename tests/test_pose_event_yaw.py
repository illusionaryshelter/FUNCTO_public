"""CPU counterexamples for declared material-frame twist intervals.

These tests compare finite rotations and complete calibrated transform chains.
They exercise the production constraints without loading robot geometry.
"""
from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from manipulation.planning import _CuroboPlanner
from manipulation.planning_problem import MotionPhase, PlanningProblem, PoseEventRequirement, PosePath


def pose(rotation=None, position=(0., 0., 0.)):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation.as_matrix()
    result[:3, 3] = position
    return result


def event(**changes):
    values = dict(progress=1., tool_from_function=np.eye(4), target=np.eye(4),
                  position_tolerance_m=.002, rotation_tolerance_rad=np.deg2rad(1),
                  source="explicit analytic declaration", yaw_axis=[0., 0., 1.],
                  yaw_tolerance_rad=np.pi)
    return PoseEventRequirement(**(values | changes))


def problem(requirement):
    path = PosePath(np.array([0., 1.]), np.repeat(np.eye(4)[None], 2, 0), "analytic reference")
    return PlanningProblem(joint_names=("q",), start_q=np.zeros(1), start_dq=np.zeros(1),
        tool_link="tool", reference_frame="world", base_pose=np.eye(4), tool_transform=np.eye(4),
        phases=(MotionPhase("reference", path),), pose_events=(requirement,))


def tensors(matrix):
    return (torch.tensor(matrix[:3, 3], dtype=torch.float64).reshape(1, 1, 3),
            torch.tensor(Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]],
                         dtype=torch.float64).reshape(1, 1, 4))


@pytest.mark.parametrize("yaw", [-179., -30., 0., 90., 180.])
def test_finite_swing_twist_is_not_rotation_vector_projection(yaw):
    relative = Rotation.from_euler("x", 30, degrees=True)*Rotation.from_euler("z", yaw, degrees=True)
    values = event().errors_numpy(pose(relative))
    assert np.rad2deg(values["rotation"]) == pytest.approx(30.)
    assert np.rad2deg(values["yaw"]) == pytest.approx(abs(yaw))
    complete = event(yaw_axis=None, yaw_tolerance_rad=None).errors_numpy(pose(relative))
    assert complete["rotation"] == pytest.approx(relative.magnitude())


def test_finite_yaw_limit_is_independent_of_remaining_tilt_budget():
    requirement = event(yaw_tolerance_rad=np.deg2rad(30))
    actual = pose(Rotation.from_euler("x", .2, degrees=True)*Rotation.from_euler("z", 30.5, degrees=True))
    values = requirement.errors_numpy(actual)
    assert values["rotation"] < requirement.rotation_tolerance_rad
    assert values["yaw"] > requirement.yaw_tolerance_rad
    p, q = tensors(actual)
    assert float(requirement.evaluate(p, q)) > 0


def test_arbitrary_axis_calibration_and_world_basis_preserve_component_errors():
    basis = Rotation.from_euler("xyz", [.4, -.7, .3])
    axis = basis.apply([0., 0., 1.])
    target = pose(Rotation.from_euler("xyz", [.1, .2, .8]), (.4, .2, -.1))
    attachment = pose(Rotation.from_euler("xyz", [-.6, .3, .1]), (.08, -.02, .03))
    relative = basis*Rotation.from_euler("x", .6, degrees=True)*Rotation.from_euler("z", 120, degrees=True)*basis.inv()
    material = target@pose(relative)
    tool = material@np.linalg.inv(attachment)
    requirement = event(target=target, tool_from_function=attachment, yaw_axis=axis)
    values = requirement.errors_numpy(tool)
    assert values["position"] < 1e-14
    assert np.rad2deg(values["rotation"]) == pytest.approx(.6)
    assert np.rad2deg(values["yaw"]) == pytest.approx(120.)
    world = pose(Rotation.from_euler("xyz", [.3, .7, -.9]), (.2, -.5, .1))
    transformed = replace(requirement, target=world@target).errors_numpy(world@tool)
    for key in values:
        assert transformed[key] == pytest.approx(values[key], abs=1e-14)


@pytest.mark.parametrize("axis,angle", [([0, 0, 1], 0.), ([0, 0, 1], np.pi), ([1, 0, 0], np.pi)])
def test_free_yaw_has_no_constraint_or_centering_cost_and_finite_gradients(axis, angle):
    requirement = event()
    task = problem(requirement)
    p, q = tensors(pose(Rotation.from_rotvec(np.array(axis)*angle)))
    q.requires_grad_()
    progress = torch.ones((1, 1), dtype=torch.float64)
    signed, _ = task.functional_inequalities(p, q, progress)
    assert set(signed) == {"function_pose:0:position", "function_pose:0:rotation"}
    assert set(signed) == set(task.functional_constraint_keys)
    cost = task.optimizer_preferences(p, q, progress)
    cost.sum().backward()
    assert torch.isfinite(q.grad).all()
    if axis == [0, 0, 1]:
        assert float(cost.detach()) < 1e-20
        assert all(float(value.detach()) <= 0 for value in signed.values())
    else:
        assert float(signed["function_pose:0:rotation"].detach()) > 0


def test_zero_width_yaw_stays_exact_and_serializes_through_binding():
    from manipulation.execution_binding import plain
    requirement = event(yaw_tolerance_rad=0.)
    restored = PoseEventRequirement(**plain(requirement))
    p, q = tensors(pose(Rotation.from_euler("z", .01)))
    signed, _ = problem(restored).functional_inequalities(p, q, torch.ones((1, 1)))
    assert torch.isfinite(signed["function_pose:0:yaw"]).all()
    assert float(signed["function_pose:0:yaw"]) > 0
    assert restored.component_tolerances["yaw"] == 0


def test_native_nominal_buffer_preserves_complete_material_tcp_conversion():
    requirement = event(target=pose(Rotation.from_euler("xyz", [.2, -.3, .4]), (.4, .1, .2)),
                        tool_from_function=pose(Rotation.from_euler("x", .2), (.08, .01, -.02)))
    task = problem(requirement)
    task.base_pose = pose(Rotation.from_euler("y", .4), (-.1, .2, 0.))
    task.tool_transform = pose(Rotation.from_euler("x", -.3), (.01, .02, .03))
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.problem, planner.GoalToolPose = task, SimpleNamespace
    planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float64))
    native = planner._functional_goal(requirement)
    assert native.position.shape == (1, 1, 1, 1, 3)
    q = native.quaternion[0, 0, 0, 0].numpy()
    link = pose(Rotation.from_quat(q[[1, 2, 3, 0]]), native.position[0, 0, 0, 0].numpy())
    material = task.base_pose@link@task.tool_transform@requirement.tool_from_function
    np.testing.assert_allclose(material, requirement.target, atol=1e-14)


def test_yaw_interval_disables_native_single_pose_convergence():
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.problem, planner.scene = problem(event()), SimpleNamespace(contacts=())
    planner.trajopt = SimpleNamespace(disable_tool_pose_tracking=Mock(),
        enable_tool_pose_tracking=Mock(), disable_joint_position_tracking=Mock())
    planner._functional_goal = Mock(return_value=SimpleNamespace(tool_frames=["tool"]))
    planner._native_request(torch.zeros((1, 1, 4, 1)))
    assert planner.route == "constrained"
    planner.trajopt.disable_tool_pose_tracking.assert_called_once()
    planner.trajopt.enable_tool_pose_tracking.assert_not_called()


def test_free_yaw_allowed_set_accepts_arbitrary_continuous_angles():
    requirement = event()
    for angle in (-np.pi, -.713529, .238214, 2.98732):
        values = requirement.errors_numpy(pose(Rotation.from_euler("z", angle)))
        assert all(values[name] <= bound for name, bound in requirement.component_tolerances.items())


def test_grounding_preserves_entry_yaw_freedom_and_source_full_pose():
    from test_manipulation_program import fixture
    from manipulation.program import ground_program
    program, scene = fixture()
    declaration = program["demo"]["functional_events"][0]
    declaration.update(frame_index=100, yaw_axis=[0, 0, 1], yaw_tolerance_rad=np.pi)
    initial = pose(Rotation.from_euler("z", 2.1))
    scene["initial_pose"] = initial
    scene["eef_pose"] = initial@scene["holding"]["T_carrier_eef"]
    source = program["demo"]["T_reference_carrier"].copy()
    grounded = ground_program(program, scene)
    np.testing.assert_allclose(grounded["object_poses"][0], initial)
    np.testing.assert_array_equal(program["demo"]["T_reference_carrier"], source)
    from manipulation.planning_problem import build_planning_problem
    compiled = build_planning_problem([dict(skill_id="declared", program=program, scene=scene)],
        joint_names=("q",), start_q=np.zeros(1), start_dq=np.zeros(1), tool_link="tool",
        reference_frame=scene["reference_frame"], base_pose=np.eye(4), tool_transform=np.eye(4))
    assert compiled.pose_events[0].yaw_tolerance_rad == np.pi
    np.testing.assert_array_equal(compiled.pose_events[0].yaw_axis, [0, 0, 1])
    declaration["yaw_tolerance_rad"] = .5
    with pytest.raises(ValueError, match="initial functional allowed set"):
        ground_program(program, scene)


@pytest.mark.parametrize("bound,expected", [(np.pi, "T"), (np.pi/6, "F"), (0., "F")])
def test_posthoc_yaw_grading_and_online_control_have_distinct_consumers(bound, expected):
    from test_recorded_simulation import ActualPoseGoalTests
    from recorded_skill_evaluation import _actual_pose_goal
    from manipulation.execution_binding import plain
    actual = pose(Rotation.from_euler("z", 1.2))
    requirement = event(yaw_tolerance_rad=bound,
        actual_relation={"function_owner_id": "material", "T_owner_function": np.eye(4)})
    observed = dict(scene_owner_poses={"material": actual}, timestamp_s=2., source="synthetic posthoc input")
    grade = _actual_pose_goal(requirement, {}, observed)
    assert grade["value"] == expected
    assert grade["yaw_error_rad"] == pytest.approx(1.2)
    assert grade["rotation_error_rad"] < 1e-14
    # The serialized new declaration is also consumed by the real executor.
    # Missing owner observations and a large frozen target offset still leave
    # close and subsequent commands governed by the original robot clock.
    fixture = ActualPoseGoalTests.fixture(actual_relation=plain(requirement.actual_relation))
    binding = fixture[3].execution_binding
    binding["problem"]["pose_events"][0].update(
        yaw_axis=[0, 0, 1], yaw_tolerance_rad=bound, target=pose(position=(.5, 0, 0)).tolist())
    binding["validation"]["inputs"]["problem"] = deepcopy(binding["problem"])
    report = ActualPoseGoalTests.run_trial(fixture)
    assert report["status"] == "curve_finished"
    assert "close" in [row["kind"] for row in report["events"]]
    assert report["goal_evidence"] == []
