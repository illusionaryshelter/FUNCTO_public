"""Exact candidate queries connect interval evidence to original scene verdicts."""
# BEGIN CPU integration keeps real native/field queries and an analytic robot FK.
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import coal
import numpy as np
import pytest
import torch

from manipulation.implicit_distance import ObservedField
from manipulation.native_field_distance import query_native_field
from manipulation.planning import JointTrajectory, _CuroboPlanner, _ProgressMap, _scene_boundary_evaluate
from test_interval_geometry import ConstantMotion


def candidate():
    return JointTrajectory(("q",), np.array([0., .025]), np.array([[0.], [.025]]),
        np.ones((2, 1)), np.array([0., 1.]), np.array([0., 1.]), (), (), {})


def planner(monkeypatch, *, permission="none", residual=0., release=False):
    curve = candidate()
    instance = _CuroboPlanner.__new__(_CuroboPlanner)
    instance.torch = torch
    instance.joint_names = ("q",)
    instance.device_cfg = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float64))
    instance.JointState = SimpleNamespace
    instance._RobotState = lambda **kwargs: kwargs["cuda_robot_model_state"]
    instance.check_request = Mock()
    instance.anchors = np.array([.5]) if release else np.array([])
    instance.budget = SimpleNamespace(max_validation_points=100, validation_spatial_step_m=.01)
    instance.problem = SimpleNamespace(tool_link="tool", tool_transform=np.eye(4), start_progress=0.,
        end_progress=1., transform_tool_pose=lambda p, q: (p, q), transform_robot_spheres=lambda x: x)
    instance._candidate_domain_validation = lambda value: {}
    shape = coal.Box(.002, .002, .002)
    field = ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        origin=[.0115, -.001, -.001], voxel_size=.002)
    queries = []

    def fk(js):
        p = torch.cat((js.position, torch.zeros((*js.position.shape[:2], 2))), -1)
        q = p.new_zeros((*p.shape[:2], 1, 4)); q[..., 0] = 1
        return SimpleNamespace(tool_poses=SimpleNamespace(tool_frames=["tool"], position=p.unsqueeze(-2),
            quaternion=q), robot_spheres=p.new_zeros((*p.shape[:2], 0, 4)))

    def evaluate(spheres, p, quaternion, progress, joints, names, *, trajectory_time,
                 components_out, geometry_evidence_out, contact_permission_progress=None, **kwargs):
        queries.append(trajectory_time.numpy().copy())
        permissions = progress if contact_permission_progress is None else contact_permission_progress
        values = torch.zeros_like(progress)
        for index in np.ndindex(progress.shape):
            matrix = np.eye(4); matrix[0, 3] = float(p[index][0])
            evidence = query_native_field(shape, matrix, field, np.eye(4))
            if permission != "none":
                evidence = {"status": "clear", "lower_m": 100.}
            values[index] = float(evidence["status"] == "intersection")
            geometry_evidence_out.append({"state_index": list(index), "time_s": float(trajectory_time[index]),
                "progress": float(progress[index]), "permission_progress": float(permissions[index]),
                "first": "body", "second": "background", "component": "collision/body/background",
                "geometry_entity_key": {"first": "robot/body", "second": "background"},
                "permission_kind": permission, **evidence})
        components_out["collision/body/background"] = values
        components_out["gripper/controller"] = torch.full_like(progress, residual)
        return sum(components_out.values())

    instance._verification_fk = SimpleNamespace(compute_kinematics=fk)
    relations = (SimpleNamespace(interval=(0., .5), release_condition_id="declared_release"),) if release else ()
    scene = SimpleNamespace(owners=(SimpleNamespace(attachments=relations),), contacts=(),
        required_progress_anchors=(.5,) if release else (), evaluate=evaluate)
    instance.scene = scene
    monkeypatch.setattr("manipulation.interval_geometry.SceneMotionBounds.from_scene",
        lambda *args, **kwargs: ConstantMotion(boundary=.0125 if release else None))
    report = {"accepted": True, "native_success": True, "residuals": {"interpolated:process_scene": 0.},
        "collision_residuals": {}, "scene_update_residuals": {}, "coverage_residuals": {},
        "conditional_unknown_residuals": {}}
    return instance, curve, report, queries


def test_interval_collision_rejects_sampled_candidate_and_keeps_exact_failure(monkeypatch):
    instance, curve, report, queries = planner(monkeypatch)
    returned, result, points = instance.evaluate_candidate(curve, native_report=report)
    assert returned is curve and not result["accepted"]
    assert result["residuals"]["interval:process_scene"] == 1.
    assert result["collision_residuals"]["interval:scene_collision"] == 1.
    assert result["continuous_geometry"]["confirmed_intersections"][0]["time_s"] == .0125
    assert points == 3 and len(queries) == 2
    assert instance.last_validation is None


def test_unresolved_finite_permission_preserves_warning_and_original_hard_result(monkeypatch):
    instance, curve, report, _ = planner(monkeypatch, permission="finite_region")
    _, result, _ = instance.evaluate_candidate(curve, native_report=report)
    assert result["accepted"]
    geometry = result["continuous_geometry"]
    assert geometry["unresolved_intervals"] and not geometry["proved_intervals"]
    assert not geometry["confirmed_intersections"]
    assert instance.export(None, result) == (curve, 0)


def test_interval_sampling_keeps_controller_residual_separate_from_intersection(monkeypatch):
    instance, curve, report, _ = planner(monkeypatch, permission="finite_region", residual=.2)
    _, result, _ = instance.evaluate_candidate(curve, native_report=report)
    assert not result["accepted"] and result["residuals"]["interval:process_scene"] == .2
    assert result["collision_residuals"]["interval:scene_collision"] == 0.
    assert result["continuous_geometry"]["scene_components"]["gripper/controller"]["maximum"] == .2


def test_release_query_keeps_original_anchor_and_physical_sides_in_one_call(monkeypatch):
    instance, curve, report, queries = planner(monkeypatch, release=True)
    _, result, _ = instance.evaluate_candidate(curve, native_report=report)
    assert all(np.any(times == .0125) for times in queries)
    rows = result["continuous_geometry"]["confirmed_intersections"]
    assert {row["side"] for row in rows if row["time_s"] == .0125} == {-1, 0, 1}


def test_rejected_task_or_joint_domain_does_not_spend_interval_geometry_work(monkeypatch):
    instance, curve, report, queries = planner(monkeypatch)
    instance._candidate_domain_validation = lambda value: value.residuals.update(continuous_position=.1) or {}
    _, result, points = instance.evaluate_candidate(curve, native_report=report)
    assert not result["accepted"] and points == 0 and not queries
    assert "continuous_geometry" not in result
# END only actual scene violations reject; interval proof gaps retain their own meaning.
