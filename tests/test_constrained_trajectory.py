"""CPU integration of actual Hermite values, FK gradients and SLSQP rows.

The translation FK and plane distance below are independent analytic fixtures;
registered-solid geometry and native GPU derivatives have separate checks.
"""
# BEGIN explicit constrained-planner regression coverage.
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manipulation.constrained_trajectory import ConstrainedTrajectory
from manipulation.physical_curve import PhysicalCurve, PHYSICAL_FTOL
from manipulation.planning import PlanningBudget, _ProgressMap, _CuroboPlanner


class JointState:
    @staticmethod
    def from_position(value, joint_names):
        return SimpleNamespace(position=value)


class TranslationFK:
    """A single prismatic axis: p=(q,0,0), R=I, Jv=(1,0,0)."""
    def compute_kinematics(self, joints):
        q = joints.position
        p = torch.cat((q, q*0., q*0.), -1)
        rotation = q.new_zeros((*q.shape[:2], 1, 4)); rotation[..., 0] = 1.
        jacobian = q.new_zeros((*q.shape[:2], 1, 6, 1)); jacobian[..., 0, 0] = 1.
        return SimpleNamespace(tool_poses=SimpleNamespace(position=p[:, :, None], quaternion=rotation,
            tool_frames=["tool"]), tool_jacobians=jacobian,
            robot_spheres=torch.cat((p, q*0.+.01), -1)[:, :, None])


class PlaneScene:
    """Signed distance to an analytic plane; q is the solid's surface location."""
    def __init__(self, fixed_slack=1.):
        self.fixed_slack = fixed_slack
    def optimizer_preferences(self, progress, *args):
        return torch.zeros_like(progress)
    def signed_constraints(self, spheres, p, r, progress, q, names, *, trajectory_time, joint_variable_mask):
        movable = joint_variable_mask.any(-1)
        value = .55-q[..., 0]
        valid = torch.stack((~movable, movable), -1)
        values = torch.stack((value, value), -1)
        return SimpleNamespace(slack={"collision/robot/plane": values,
            "fixed/model": torch.full_like(values, self.fixed_slack)},
            valid={"collision/robot/plane": valid,
                   "fixed/model": torch.stack((torch.ones_like(movable), torch.zeros_like(movable)), -1)},
            variable={"collision/robot/plane": torch.tensor([False, True]),
                      "fixed/model": torch.tensor([False, True])}, fixed_failures={}, diagnostics={})


def fixture(*, fixed_slack=1., events=(), mapping=None):
    budget = PlanningBudget(knots=6, interpolation_steps=4, maximum_trajectory_dt=.125,
        interpolation_dt=.03125, constraint_weight=1., trajectory_iterations=100,
        interpolation_buffer_size=512)
    bounds = {name: np.array([[-limit], [limit]]) for name, limit in
              (("position", .58), ("velocity", 10.), ("acceleration", 100.), ("jerk", 1000.))}
    physical = PhysicalCurve([.2], [0.], 8, .5, np.arange(161)*.03125, ("q",), bounds,
        joint_tolerance=1e-4, velocity_tolerance=1e-4)
    problem = SimpleNamespace(tool_link="tool", mechanisms=(), stop_progress=(), events=events,
        conditions=(), transform_tool_pose=lambda p, r: (p, r), transform_robot_spheres=lambda v: v,
        residuals=lambda p, *args, **kwargs: {"nominal": p[..., 0]-.8},
        optimizer_preferences=lambda p, *args, **kwargs: p[..., 0]*0.)
    mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.])) if mapping is None else mapping
    solver = ConstrainedTrajectory(physical, problem=problem, scene=PlaneScene(fixed_slack),
        progress_map=mapping, anchors=np.array([0., 1.]), budget=budget, fk=TranslationFK(),
        joint_state_type=JointState, device="cpu",
        smooth_config={"squared_l2_regularization_weight": [0.]*5, "retime_regularization_weights": True})
    return solver, problem, mapping


def planner_fixture(solver, problem, mapping):
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.physical_curve, planner.problem, planner.progress_map = solver.physical, problem, mapping
    planner.joint_names, planner.finger_trajectory = ("q",), None
    planner.seed_strategies = ["first", "second"]
    planner.last_optimizer_diagnostics = [{"success": False}, {"success": False}]
    return planner


def test_joint_and_scene_constraints_beat_soft_target_from_an_infeasible_seed():
    solver, problem, mapping = fixture()
    initial = np.full((8, 1), .6)
    first = solver.evaluate(initial)
    assert first.physical.violation > 0.
    assert first.violation > first.physical.violation
    final, report = solver.solve(initial)
    assert final.accepted, report
    assert final.violation <= PHYSICAL_FTOL
    assert final.objective > 0.  # The nominal target remains beyond the plane.
    planner = planner_fixture(solver, problem, mapping)
    planner.last_optimizer_diagnostics = [report, report]
    trajectory, selected = planner.select_candidate([first, final])
    assert selected["selected_seed_index"] == 1 and selected["accepted"]
    assert planner.last_validation["violations"].max() == 0.
    np.testing.assert_array_equal(trajectory._curve.c, solver.physical.curve(final.actions).c)
    with patch.object(solver.physical, "constraints", side_effect=AssertionError("export rechecked")):
        assert planner.export(trajectory, selected) == (trajectory, 0)


def test_actual_value_gradient_and_scene_rows_match_independent_differences():
    solver, _, _ = fixture()
    actions = np.linspace(.23, .4, 8)[:, None]
    result = solver.evaluate(actions)
    direction = np.arange(1., 9.).reshape(8, 1)/8.
    # FK's float32 input is explicit; choose a finite difference above its ULP.
    delta = 2e-4
    plus, minus = solver.evaluate(actions+delta*direction), solver.evaluate(actions-delta*direction)
    expected = (plus.objective-minus.objective)/(2*delta)
    assert float(result.objective_gradient@direction.ravel()) == pytest.approx(expected, rel=4e-4)
    index = result.names.index("scene/collision/robot/plane/variable")
    assert float(result.jacobian[index]@direction.ravel()) == pytest.approx(
        (plus.slack[index]-minus.slack[index])/(2*delta), abs=1e-10)
    # The cache is exactly one completed evaluation, not a feasible-history rule.
    assert solver.evaluate(actions-delta*direction) is minus


def test_fixed_negative_rows_share_slsqp_total_tolerance_and_input_failure_is_visible():
    solver, _, _ = fixture(fixed_slack=-.4e-6)
    initial = np.full((8, 1), .3)
    observed = {}
    def minimize(objective, x, **kwargs):
        rows = kwargs["constraints"]["fun"](x)
        jacobian = kwargs["constraints"]["jac"](x)
        observed.update(rows=rows, jacobian=jacobian)
        return SimpleNamespace(x=x, success=True, status=0, nit=0, message="fixture")
    with patch("manipulation.constrained_trajectory.minimize", side_effect=minimize):
        result, _ = solver.solve(initial)
    assert observed["rows"][-1] == pytest.approx(-.4e-6)
    np.testing.assert_array_equal(observed["jacobian"][-1], 0.)
    assert result.violation == pytest.approx(.4e-6) and result.accepted
    failed_solver, _, _ = fixture(fixed_slack=-2e-6)
    with patch("manipulation.constrained_trajectory.minimize", side_effect=AssertionError("fixed failure sent to solver")):
        failed, report = failed_solver.solve(initial)
    assert not failed.accepted and report["iterations"] == 0


def test_candidate_ranking_uses_same_objective_and_numerical_contract_even_on_solver_failure():
    solver, problem, mapping = fixture()
    valid = solver.evaluate(np.full((8, 1), .3))
    cheaper = replace(valid, objective=valid.objective-1., slack=np.r_[valid.slack, -2e-6],
        names=valid.names+("extra physical condition",), jacobian=np.vstack((valid.jacobian, np.zeros(8))))
    planner = planner_fixture(solver, problem, mapping)
    _, report = planner.select_candidate([cheaper, valid])
    assert report["selected_seed_index"] == 1
    assert report["accepted"] and not report["optimizer"]["success"]
    borderline = replace(valid, slack=np.r_[valid.slack, -.5e-6], names=valid.names+("numeric boundary",),
        jacobian=np.vstack((valid.jacobian, np.zeros(8))))
    _, report = planner.select_candidate([borderline, valid])
    assert report["accepted"] and report["numerical_status"] == "warning"
    assert -.5e-6 in report["canonical_constraints"]["slack"]


def test_export_preserves_original_event_identity_plateau_and_evidence_barrier():
    from dataclasses import asdict
    from manipulation.planning_problem import PlanningEvent, event_occurrence_key
    event = PlanningEvent("verify", "original", 1., 79, "verification", "source event", "Holding", True)
    clock = {event_occurrence_key(event): {"event": asdict(event), "time": 1., "hold_until": 2.25}}
    mapping = _ProgressMap(np.array([0., .2, .45, 1.]), np.array([0., 1., 1., 2.]), clock)
    solver, problem, mapping = fixture(events=(event,), mapping=mapping)
    evaluated = solver.evaluate(np.full((8, 1), .3))
    planner = planner_fixture(solver, problem, mapping)
    trajectory, report = planner.select_candidate([evaluated, evaluated])
    assert report["accepted"] and len(trajectory.events) == 1
    assert trajectory.events[0]["source_frame"] == 79
    assert trajectory.executable_until == 1.
    assert trajectory.events[0]["hold_until"] == 2.25
    copy = trajectory.next_verification; copy["time"] = 5.
    assert trajectory.executable_until == 1.
# END actual curve and common qualification rule regression coverage.
