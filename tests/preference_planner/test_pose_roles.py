"""Exercise explicit roles through the original FUNCTO public computations.

Analytical material poses isolate the event contract.  These are geometry,
serialization and request-identity tests, not simulated grasp evidence.
"""

from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "tests"), str(PROJECT / "tests/planning")]


from manipulation.cubic_constraints import CubicAffineMap
from manipulation.execution_binding import plain
from manipulation.planning import _obligation_contract
from manipulation.planning_problem import (
    MotionPhase, PlanningEvent, PlanningProblem, PoseEventRequirement, PosePath,
)
from manipulation.planning_scene import validate_grasp_contact_arrival
from manipulation.unified_solver import TrajectoryComputation
from manipulation.preference_planner.pose_roles import (
    ObjectivePoseEvent, bind_pose_objectives, pose_objective,
    pose_objective_components, pose_objective_scales, prior_goal_ids,
)

torch.set_num_threads(1)


def event(progress=.5, *, yaw=None, source=None, actual_relation=None):
    """Declare analytical pose accuracy independently from preference scales."""
    return PoseEventRequirement(progress, np.eye(4), np.eye(4), .01, .1,
        {"origin": "analytical material relation"} if source is None else source,
        actual_relation=actual_relation,
        yaw_axis=None if yaw is None else np.array([0., 0., 1.]),
        yaw_tolerance_rad=yaw)


def problem(events):
    """A complete one-phase request retains measured nonzero entry velocity."""
    path = PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))),
                    "analytical source representation")
    return PlanningProblem(("arm",), [0.], [.03], "tool", "world", np.eye(4), np.eye(4),
        (MotionPhase("reference", path),), pose_events=tuple(events),
        source_timestamps=np.array([0., 2.]), source_progress=np.array([0., 1.]))


def declarations(events, roles):
    """Test input supplies every scale; the binder never estimates one."""
    return [{"goal_id": ["pose_events", index], "role": role,
             "scales": {name: value if value > 0 else .02
                        for name, value in PoseEventRequirement.component_tolerances.fget(item).items()},
             "scale_source": "Original declared quality normalization; explicit .02 rad for zero-width yaw"}
            for index, (item, role) in enumerate(zip(events, roles))]


def query():
    position = torch.tensor([[[.03, 0., 0.]] * 3], dtype=torch.float64, requires_grad=True)
    quaternion = torch.tensor([[[1., 0., 0., 0.]] * 3], dtype=torch.float64)
    progress = torch.tensor([[0., .5, 1.]], dtype=torch.float64)
    return position, quaternion, progress, progress * 2.


def test_prior_changes_original_hard_consumers_and_keeps_all_measurements():
    original = problem([event()])
    bound = bind_pose_objectives(original, declarations(original.pose_events, ["prior"]))
    p, q, progress, times = query()
    assert type(bound) is type(original)
    assert [item.name for item in fields(bound.pose_events[0])] == [
        item.name for item in fields(PoseEventRequirement)]
    eq, iq = bound.constraint_components(p, q, progress, trajectory_time=times)
    assert not any(key.startswith("function_pose:") for group in (eq, iq) for key in group)
    assert set(iq) == {"progress:lower", "progress:upper"}
    residual = bound.residuals(p, q, progress, trajectory_time=times)["function_pose:0"]
    assert residual.shape == progress.shape and residual.dtype == p.dtype
    assert residual.device == p.device and residual.requires_grad
    torch.testing.assert_close(residual, torch.zeros_like(progress))
    torch.testing.assert_close(torch.autograd.grad(residual.sum(), p)[0], torch.zeros_like(p))
    assert bound.goal_residual(p, q, progress, goal_ids=(("pose_events", 0),)).shape == (1, 3, 0)
    before = original.measurements(p, q, progress, trajectory_time=times)
    after = bound.measurements(p, q, progress, trajectory_time=times)
    assert before.keys() == after.keys()
    for name in before:
        torch.testing.assert_close(before[name], after[name], rtol=0., atol=0.)
    assert float(after["function:0:position_error_m"].detach().max()) == .03
    assert "pose_objective" not in original.pose_events[0].source


def test_same_time_mixed_roles_preserve_required_rows_and_point_projection():
    original = problem([event(), event(yaw=0.)])
    bound = bind_pose_objectives(original, declarations(original.pose_events, ["prior", "required"]))
    p, q, progress, times = query()
    expected = original.constraint_components(p, q, progress, trajectory_time=times)
    actual = bound.constraint_components(p, q, progress, trajectory_time=times)
    for expected_group, actual_group in zip(expected, actual):
        kept = {key: value for key, value in expected_group.items() if not key.startswith("function_pose:0:")}
        assert actual_group.keys() == kept.keys()
        for key in kept:
            torch.testing.assert_close(actual_group[key], kept[key], rtol=0., atol=0.)
    expected_point = original.goal_residual(p, q, progress, goal_ids=(("pose_events", 1),), normalized=True)
    actual_point = bound.goal_residual(p, q, progress,
        goal_ids=(("pose_events", 0), ("pose_events", 1)), normalized=True)
    torch.testing.assert_close(actual_point, expected_point, rtol=0., atol=0.)


def test_original_solver_fixed_yaw_layout_tracks_only_required_events():
    original = problem([event(0., yaw=0.), event(0., yaw=0.), event(1., yaw=0.)])
    bound = bind_pose_objectives(original, declarations(original.pose_events, ["prior", "required", "required"]))
    curve = CubicAffineMap([0.], [.03], 5, .1)
    domain = SimpleNamespace(_curve_map=curve, intervals=curve.intervals,
                             _constraint_bounds=np.ones((4, 1)))
    computation = TrajectoryComputation(problem=bound, scene=None, kinematics=None,
        domain=domain, joint_names=("arm",), parameters=np.array([0., .5, 1.]),
        progress=np.array([0., .5, 1.]), ordinary_indices=np.arange(3),
        motion_configuration={}, native_dt=.1, constraint_weight=1., max_host_rss_bytes=None)
    assert computation.equality_keys == ["function_pose:1:yaw", "function_pose:2:yaw"]
    np.testing.assert_array_equal(computation.fixed_equality_rows, [0])
    np.testing.assert_array_equal(computation.variable_equality_rows, [1])


def test_free_yaw_reuses_original_geometry_without_a_preference_component():
    original = problem([event(yaw=np.pi)])
    supplied = declarations(original.pose_events, ["prior"])
    supplied[0]["scales"] = {"position": .04, "rotation": .3}
    bound = bind_pose_objectives(original, supplied)
    selected = bound.pose_events[0]
    p = torch.tensor([[.03, 0., 0.]], dtype=torch.float64, requires_grad=True)
    quaternions = [torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64),
                   torch.tensor([[np.sqrt(.5), 0., 0., np.sqrt(.5)]], dtype=torch.float64)]
    costs = []
    for q in quaternions:
        values = pose_objective_components(selected, p, q)
        assert set(values) == {"position", "rotation"}
        costs.append(sum((values[name] / scale).square()
                         for name, scale in pose_objective_scales(selected).items()))
    torch.testing.assert_close(costs[0], costs[1], rtol=0., atol=1e-28)
    assert float(torch.autograd.grad(costs[1].sum(), p)[0][0, 0]) > 0.
    assert selected.position_tolerance_m == .01
    assert selected.rotation_tolerance_rad == .1
    assert pose_objective_scales(selected) == supplied[0]["scales"]
    assert float(selected.error_components(p, quaternions[1])["full_rotation"].detach()) > 1.


def test_original_plain_reconstruction_retains_roles_and_diagnostic_thresholds():
    original = problem([event(), event(source="original scalar provenance")])
    bound = bind_pose_objectives(original, declarations(original.pose_events, ["prior", "required"]))
    rows = plain(bound.pose_events)
    reconstructed = [PoseEventRequirement(**deepcopy(row)) for row in rows]
    assert pose_objective(reconstructed[0])["role"] == "prior"
    assert reconstructed[0].component_tolerances == original.pose_events[0].component_tolerances
    assert reconstructed[1].source["original_source"] == "original scalar provenance"
    # Original evaluator reconstruction intentionally retains diagnostic T/F
    # thresholds. The role is consumed by result aggregation, not by deleting
    # those measured errors or replacing a diagnostic F with T.
    p, q, _, _ = query()
    assert float(reconstructed[0].evaluate(p, q).detach().max()) > 0.
    binding = {"problem": {"pose_events": rows},
               "goal_identities": {"pose_events": [7, 11]}}
    assert prior_goal_ids(binding) == [["pose_events", 7]]


def test_role_and_scale_changes_are_part_of_original_continuation_contract():
    original = problem([event(actual_relation={"target_owner_id": "declared_owner",
                                               "T_owner_target": np.eye(4)})])
    supplied = declarations(original.pose_events, ["prior"])
    bound = bind_pose_objectives(original, supplied)
    base = _obligation_contract(bound)
    changed_role = deepcopy(supplied)
    changed_role[0]["role"] = "required"
    assert _obligation_contract(bind_pose_objectives(original, changed_role)) != base
    changed_scale = deepcopy(supplied)
    changed_scale[0]["scales"]["position"] *= 2.
    assert _obligation_contract(bind_pose_objectives(original, changed_scale)) != base
    same = bind_pose_objectives(bound, supplied)
    assert _obligation_contract(same) == base
    block, record = bound.bound_motion_block()
    assert _obligation_contract(block) == base
    assert record["goal_identities"]["pose_events"] == [0]
    np.testing.assert_array_equal(block.pose_events[0].actual_relation["T_owner_target"], np.eye(4))
    assert isinstance(block.pose_events[0], ObjectivePoseEvent)
    np.testing.assert_array_equal(block.start_dq, original.start_dq)
    np.testing.assert_array_equal(block.source_timestamps, original.source_timestamps)


def test_arrival_identity_and_original_source_survive_role_binding():
    activation = {"skill_id": "declared_skill", "event_id": "declared_command", "occurrence": 0, "kind": "close"}
    source = {"kind": "frozen_perception_grasp_arrival", "carrier_id": "declared_owner",
              "activation_event": activation, "original_evidence": {"frames": [2, 5, 9]}}
    original = problem([event(), event(source=source)])
    original.events = (PlanningEvent("declared_command", "declared_skill", .5, 5, "close", source),)
    condition = {"kind": "grasp_contact_activation", "condition_id": "declared_contact",
                 "carrier_id": "declared_owner", "arrival_goal_id": ["pose_events", 1],
                 "activation_event": activation, "interval": [.5, 1.], "source": deepcopy(source)}
    original.conditions = (condition,)
    bound = bind_pose_objectives(original, declarations(original.pose_events, ["required", "prior"]))
    validate_grasp_contact_arrival(bound.conditions[0], bound.pose_events[1])
    assert bound.conditions == original.conditions
    assert bound.pose_events[1].actual_relation is None
    assert bound.pose_events[1].progress == bound.events[0].progress
    for name, value in source.items():
        assert bound.pose_events[1].source[name] == value
    reconstructed = PoseEventRequirement(**plain(bound.pose_events[1]))
    validate_grasp_contact_arrival(bound.conditions[0], reconstructed)


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "free_yaw_scale", "zero_scale"])
def test_incomplete_or_inconsistent_explicit_declarations_are_rejected(change):
    original = problem([event(yaw=np.pi)])
    supplied = declarations(original.pose_events, ["prior"])
    if change == "missing":
        supplied = []
    elif change == "duplicate":
        supplied *= 2
    elif change == "unknown":
        supplied[0]["role"] = "implicit"
    elif change == "free_yaw_scale":
        supplied[0]["scales"]["yaw"] = .2
    else:
        supplied[0]["scales"]["position"] = 0.
    with pytest.raises(ValueError):
        bind_pose_objectives(original, supplied)
