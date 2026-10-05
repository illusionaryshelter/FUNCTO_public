"""CPU integration checks for the planner's map replacement boundary."""

# BEGIN witnesses retain the native curve/clock through a solver rebuild.
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from manipulation.physical_joint_domain import CubicJointDomain
from manipulation.planning import (
    PlanningBudget, ResourceLimitError, _CuroboPlanner,
    _planning_workspace_bytes, _require_planning_memory,
)
from manipulation.voxel_layers import LayeredVoxelQuery


# BEGIN CPU fixtures expose each preflight boundary without touching CUDA.
@pytest.fixture(autouse=True)
def admission(monkeypatch):
    result = SimpleNamespace(storage=Mock(return_value={"encoded_record_bytes_upper": 7}),
                             workspace=Mock(return_value=(101, 103, False)), gpu=Mock())
    monkeypatch.setattr("manipulation.planning._validation_storage_admission", result.storage)
    monkeypatch.setattr("manipulation.planning._planning_workspace_bytes", result.workspace)
    monkeypatch.setattr("manipulation.planning._require_planning_memory", result.gpu)
    return result
# END the separately imported capacity functions remain available for pure tests.


def planner_fixture(replace=True):
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.torch = torch
    planner.device_cfg = SimpleNamespace(device="cpu", to_device=lambda value: torch.as_tensor(value, dtype=torch.float32))
    planner.budget = PlanningBudget()
    planner.robot_metadata = {"active_joint_names": ["joint"], "links": 1, "spheres": 1}
    planner.joint_names = ("joint",)
    planner.JointState = SimpleNamespace(from_position=lambda q, **kwargs: SimpleNamespace(position=q))
    planner.problem = SimpleNamespace(tool_link="tcp", transform_tool_pose=lambda p, r: (p, r),
                                     transform_robot_spheres=lambda value: value)
    planner.physical_domain = CubicJointDomain(torch.tensor([-2.]), torch.tensor([2.]),
        torch.tensor([[.2]]), torch.tensor([[.05]]), .1, 4)
    planner.progress_map = SimpleNamespace(u=np.array([0., 1.]), s=np.array([0., 2.]),
        anchor_parameters=lambda anchors: np.array([[.137, .274], [.811, 1.622]]))
    planner.anchors = np.array([.274, 1.622])
    planner.refinement_parameters = np.array([.477])
    planner.formal_clock = {"duration": .8}
    planner.trajopt = SimpleNamespace(horizon=9, destroy=Mock())
    background = LayeredVoxelQuery.__new__(LayeredVoxelQuery)
    background.refinement_source = object()
    seen = []

    def fk(state):
        q = state.position
        centers = torch.cat((q, torch.zeros_like(q), torch.zeros_like(q)), -1)
        radii = torch.full_like(q, .01)
        orientation = torch.zeros((*q.shape[:2], 1, 4))
        orientation[..., 0] = 1.
        return SimpleNamespace(robot_spheres=torch.cat((centers, radii), -1).unsqueeze(-2),
            tool_poses=SimpleNamespace(tool_frames=["tcp"], position=centers.unsqueeze(-2),
                                       quaternion=orientation))

    def witnesses(spheres, position, quaternion, progress, joints, names, *, trajectory_time, include_satisfied):
        assert include_satisfied is True
        seen.append((joints.clone(), trajectory_time.clone(), progress.clone()))
        return [{"time_s": float(trajectory_time[0, 0]), "body_id": "body", "positive_spheres": 1,
                 "fine_supported": False, "guidance_weight": 0., "coarse_cost": .2,
                 "fine_cost": .1, "guidance_cost": .2}]

    verification = object()
    replacement = SimpleNamespace(clone_for_rollout=Mock(return_value=verification),
                                  background_refinement_witnesses=witnesses)
    planner.scene = SimpleNamespace(background=background, background_refinement_witnesses=witnesses,
        refine_background=Mock(return_value=(replacement if replace else None, {"reason": "test"})))
    planner._verification_fk = SimpleNamespace(compute_kinematics=fk)
    planner._verification_scene = object()
    planner._check_validation_work = Mock()
    planner.refine_validation = Mock()
    raw = torch.full((1, 1, 4, 1), .3)
    seed = planner.physical_domain.initialize(raw)
    return planner, seed, seen, replacement, verification


def test_witness_queries_use_actual_cubic_exact_times_and_replace_before_rebuild(admission):
    planner, seed, seen, replacement, verification = planner_fixture()
    old_scene, old_verification, old_solver = planner.scene, planner._verification_scene, planner.trajopt
    stages = []

    def gpu_check(*args):
        assert planner.scene is old_scene
        assert planner._verification_scene is old_verification
        assert planner.trajopt is old_solver
        replacement.clone_for_rollout.assert_not_called()
        admission.storage.assert_called_once_with(planner.problem, replacement, planner.robot_metadata, planner.budget)
        admission.workspace.assert_called_once()
        stages.append("admitted")

    admission.gpu.side_effect = gpu_check

    def rebuild(parameters):
        assert stages == ["admitted"]
        assert planner.scene is replacement
        assert planner._verification_scene is verification
        np.testing.assert_array_equal(parameters, [.477])
        stages.append("rebuilt")

    planner.refine_validation.side_effect = rebuild
    result = planner.refine_background_for_seed(seed)
    joints, times, progress = seen[0]
    u = times.numpy()[0]/.8
    for anchor in (.137, .811, .477):
        assert np.any(np.isclose(u, anchor, atol=1e-7, rtol=0))
    expected = planner.physical_domain.evaluate(seed[0].numpy(), u)
    np.testing.assert_allclose(joints.numpy(), expected, atol=1e-7, rtol=0)
    np.testing.assert_allclose(progress.numpy()[0], 2*u, atol=1e-7, rtol=0)
    assert result["updated"] is True
    planner.refine_validation.assert_called_once()
    assert planner._check_validation_work.call_count >= 3
    assert stages == ["admitted", "rebuilt"]
    assert result["planner_admission"]["scene_workspace_bytes"] == 101


def test_refused_refinement_retains_scene_and_solver():
    planner, seed, _, _, _ = planner_fixture(replace=False)
    original = planner.scene
    result = planner.refine_background_for_seed(seed)
    assert result["updated"] is False
    assert planner.scene is original
    planner.refine_validation.assert_not_called()


def test_builder_api_error_is_not_silently_treated_as_no_refinement():
    planner, seed, _, _, _ = planner_fixture()
    planner.scene.refine_background.side_effect = RuntimeError("native mapper failure")
    with pytest.raises(RuntimeError, match="native mapper failure"):
        planner.refine_background_for_seed(seed)
    planner.refine_validation.assert_not_called()


# BEGIN a refused trial cannot invalidate the already running planner.
@pytest.mark.parametrize("boundary", ["workspace", "gpu"])
def test_candidate_capacity_refusal_preserves_all_running_objects(admission, boundary):
    planner, seed, _, replacement, _ = planner_fixture()
    old_scene, old_verification, old_solver = planner.scene, planner._verification_scene, planner.trajopt
    getattr(admission, boundary).side_effect = ResourceLimitError("test_scene_budget", 933, 644)
    was_cuda_initialized = torch.cuda.is_initialized()
    result = planner.refine_background_for_seed(seed)
    assert result["updated"] is False
    assert result["status"] == "resource_refused"
    assert result["resource_refusal"]["resource"] == "test_scene_budget"
    assert result["resource_refusal"]["required_bytes"] == 933
    assert result["resource_refusal"]["limit_bytes"] == 644
    assert planner.scene is old_scene
    assert planner._verification_scene is old_verification
    assert planner.trajopt is old_solver
    old_solver.destroy.assert_not_called()
    replacement.clone_for_rollout.assert_not_called()
    planner.refine_validation.assert_not_called()
    assert torch.cuda.is_initialized() == was_cuda_initialized


@pytest.mark.parametrize("boundary", ["storage", "workspace", "gpu"])
def test_plain_preflight_value_error_keeps_its_identity(admission, boundary):
    planner, seed, _, _, _ = planner_fixture()
    old_scene, old_verification, old_solver = planner.scene, planner._verification_scene, planner.trajopt
    error = ValueError("invalid API layout, not a byte refusal")
    getattr(admission, boundary).side_effect = error
    with pytest.raises(ValueError) as captured:
        planner.refine_background_for_seed(seed)
    assert captured.value is error
    assert planner.scene is old_scene
    assert planner._verification_scene is old_verification
    assert planner.trajopt is old_solver
    old_solver.destroy.assert_not_called()
    planner.refine_validation.assert_not_called()


def test_capacity_functions_report_the_exact_failed_byte_comparison():
    scene = SimpleNamespace(required_progress_anchors=(), estimated_workspace_allocation=lambda *args, **kwargs:
        {"retained": 0, "live": 11, "shared_grids": 0, "shared_cpu": 0, "graph_live": 0})
    with pytest.raises(ResourceLimitError) as captured:
        _planning_workspace_bytes(scene, {}, PlanningBudget(max_scene_workspace_bytes=10), 2)
    assert (captured.value.resource, captured.value.required_bytes, captured.value.limit_bytes) == (
        "scene_workspace", 11, 10)
    reserve = 2*1024**3
    cuda = SimpleNamespace(mem_get_info=Mock(return_value=(reserve-1, 12*1024**3)),
        memory_reserved=Mock(return_value=0), memory_allocated=Mock(return_value=0))
    with pytest.raises(ResourceLimitError) as captured:
        _require_planning_memory(SimpleNamespace(cuda=cuda), "cpu", 100, 200)
    assert (captured.value.resource, captured.value.required_bytes, captured.value.limit_bytes) == (
        "gpu_native_reserve", reserve, reserve-1)
    cuda.mem_get_info.return_value = reserve+50, 12*1024**3
    with pytest.raises(ResourceLimitError) as captured:
        _require_planning_memory(SimpleNamespace(cuda=cuda), "cpu", 100, 200)
    assert (captured.value.resource, captured.value.required_bytes, captured.value.limit_bytes) == (
        "gpu_planner_workspace", reserve+300, reserve+50)
# END shape/runtime failures propagate; byte-cap refusals preserve the old scene.
# END map proposals never acquire a trajectory acceptance verdict.
