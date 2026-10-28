"""CPU lifecycle integration through the original selector and qualifier.

These tests use the actual cubic export, complete physical-domain validation,
terminal equations, preference-planner generator and original candidate selector.
Only task/scene evidence producers and the numerical solver are controlled
test doubles.  Their explicit reports test lifecycle semantics; they do not
establish that any particular observed scene is collision free.
"""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "tests"), str(PROJECT / "tests/planning")]


from manipulation.preference_planner import backend
from manipulation.preference_planner.backend import PreferencePlanner, _hold_polynomial_report
from manipulation.physical_joint_domain import CubicJointDomain
from manipulation.planning import JointTrajectory, _ProgressMap
from manipulation.planning_problem import NumericalTolerance
from manipulation.unified_solver import ConstrainedSolution
from manipulation.preference_planner.terminal import TerminalHold


class EvidencePlanner(PreferencePlanner):
    """Supply explicit task/scene reports while retaining actual qualification.

    No native IK or CUDA workspace is initialized.  The same original saved
    polynomial, bound joint limits and request ledger remain active.  A test
    can choose a task violation or a continuous geometry verdict independently
    of numerical solver status.
    """

    def __init__(self, *, hold_start=.5, baseline=0):
        self.torch = torch
        self.joint_names = ("joint_a", "joint_b")
        q0 = torch.tensor([.2, -.3], dtype=torch.float64)
        bounds = torch.tensor([[[-1e6, -1e6], [1e6, 1e6]]] * 4, dtype=torch.float64)
        self.physical_domain = CubicJointDomain(bounds, q0, torch.zeros_like(q0), .05, 21)
        self.action_horizon = 21
        self.formal_clock = {"duration": 1., "knot_dt": .05}
        clock_u, clock_z = (([0., 1.], [1., 1.]) if hold_start == 0.
                            else ([0., hold_start, 1.], [0., 1., 1.]))
        self.progress_map = _ProgressMap(np.asarray(clock_u), np.asarray(clock_z))
        self.terminal = TerminalHold.from_clock(self.physical_domain._curve_map,
            clock_u, clock_z, source="explicit CPU numerical test clock")
        self.problem = SimpleNamespace(numerical=NumericalTolerance(), events=(), conditions=())
        limits = SimpleNamespace(joint_names=self.joint_names, position=bounds[0],
            velocity=bounds[1], acceleration=bounds[2], jerk=bounds[3])
        self._verification_native = SimpleNamespace(get_cost=lambda name:
            SimpleNamespace(config=SimpleNamespace(joint_limits=limits)))
        self.finger_trajectory = None
        self.scene = SimpleNamespace()
        self.budget = SimpleNamespace(max_validation_points=100, trajectory_iterations=3,
                                      maximum_trajectory_solves=2)
        self.robot_metadata, self.anchor_capacity = {}, 0
        self.computation = SimpleNamespace(trial_history=[])
        self.geometry_constraints = SimpleNamespace(row_count=0, workspace_bytes=lambda: {})
        self.pose_objectives = []
        self.seed_strategies = ["original_reference_initializer"]
        self.request_work = {"trajectory_solves": baseline + 1}
        self.request_work_baseline = {"trajectory_solves": baseline}
        self.solve_history, self.candidate_history, self.checks = [], [], []
        self._current_control_curve = self._current_full_controls = None
        self.last_candidate_report = self.last_validation = None
        self._candidate_added_rows = 0
        self.task_violation = 0.
        self.rows_per_qualification = 0
        self.qualification_calls = self.geometry_calls = 0
        self.geometry_report = {
            "confirmed_intersections": [], "proved_intervals": [], "unresolved_intervals": [],
            "query_points": 0, "hard_scene_residual": 0.,
            "scene_categories": {}, "scene_components": {},
        }

    def check_request(self, category=None):
        """Observe the same request-relative counter checked before an exchange."""
        used = self.request_work["trajectory_solves"] - self.request_work_baseline["trajectory_solves"]
        self.checks.append((category, used))
        if category == "trajectory_solves":
            assert used < self.budget.maximum_trajectory_solves

    def _candidate_metrics(self, trajectory, candidate_scene, *, record_work):
        """Produce a controlled complete finite-query result for this curve."""
        self.qualification_calls += 1
        self.geometry_constraints.row_count += self.rows_per_qualification
        self._candidate_added_rows += self.rows_per_qualification
        record_work(points=3)
        report = {"accepted": self.task_violation == 0.,
            "residuals": {"task": self.task_violation, "process_scene": 0.},
            "collision_residuals": {"collision": 0.}, "coverage_residuals": {},
            "scene_update_residuals": {}, "conditional_unknown_residuals": {},
            "scene_components": {}}
        return report, np.array([0., .5, 1.]), 3, (candidate_scene, [])

    def _candidate_geometry(self, trajectory, candidate_scene, **kwargs):
        """Return explicit known/unknown/intersection evidence, without geometry."""
        self.geometry_calls += 1
        return deepcopy(self.geometry_report)

    def _candidate_objective(self, trajectory, candidate_scene):
        """Keep candidate ordering finite without constructing robot tool poses."""
        return {"fixed_objective": 0., "objective_components": {"test_value": 0.}}


def zero_controls(planner):
    """A constant arm curve exactly satisfies the measured static entrance."""
    return np.broadcast_to(planner.physical_domain._curve_map.start_q, (21, 2)).copy()


def record_qualification(planner, trajectory):
    """Record the original qualifier's pass and finite-query accounting."""
    work = {"passes": 0, "points": 0}

    def record(**increments):
        for name, value in increments.items():
            work[name] += value

    result = planner.evaluate_candidate(trajectory, record_work=record)
    return result, work


def select(planner, results):
    """Drive the production selector, including its early accepted return."""
    delivered = []
    chosen = planner.select_candidate(results, record_work=lambda **kwargs: None,
        on_candidate=lambda result, report: delivered.append(deepcopy(report)))
    return chosen, delivered


def lightweight_workspace(monkeypatch):
    """Scene storage is absent from this explicitly synthetic lifecycle test."""
    monkeypatch.setattr(backend, "_planning_workspace_bytes",
                        lambda *args, **kwargs: SimpleNamespace(pending_cpu_bytes=0))


def test_supplied_curve_cannot_bypass_complete_terminal_polynomial_check():
    """Equal endpoint positions and zero final speed do not establish a hold."""
    planner = EvidencePlanner()
    q0 = planner.physical_domain._curve_map.start_q
    curve = JointTrajectory(planner.joint_names, np.array([0., .5, .75, 1.]),
        np.broadcast_to(q0, (4, 2)).copy(), np.array([[0., 0.], [0., 0.], [1., 0.], [0., 0.]]),
        planner.progress_map.u.copy(), planner.progress_map.s.copy(), (), (), {})
    np.testing.assert_array_equal(curve._curve(.5), curve._curve(1.))
    np.testing.assert_array_equal(curve._curve([.5, 1.], 1), 0.)
    measured = _hold_polynomial_report(curve, .5)
    assert measured["maximum_position_change_rad"] > .01
    assert measured["maximum_velocity_rad_s"] >= 1.
    (_, report, _), work = record_qualification(planner, curve)
    assert not report["accepted"]
    assert report["residuals"]["reference_terminal_position"] > 0.
    assert report["residuals"]["reference_terminal_velocity"] > 0.
    assert planner.last_validation is None
    assert planner.geometry_calls == 0  # Original task/domain failure ordering.
    assert work == {"passes": 1, "points": 3}


def test_original_selector_applies_terminal_check_to_numerical_candidates():
    """An optimizer success flag cannot qualify a moving terminal suffix."""
    planner = EvidencePlanner()
    controls = zero_controls(planner)
    controls[-1, 0] += .01
    (_, curve, report), _ = select(planner,
        [ConstrainedSolution(controls, {"success": True, "solver_called": True})])
    assert planner.qualification_calls == 1
    assert curve is planner.delivery_candidate
    assert not report["accepted"]
    assert report["residuals"]["reference_terminal_velocity"] > 0.
    assert planner.last_validation is None


# BEGIN the original qualifier consumes violations, retaining interval scope separately.
@pytest.mark.parametrize("outcome", ["unresolved", "intersection", "conditional", "clear"])
def test_original_geometry_residuals_preserve_distinct_evidence(outcome):
    """Unresolved intervals add no rejection; actual scene violations do."""
    planner = EvidencePlanner()
    if outcome == "unresolved":
        planner.geometry_report["unresolved_intervals"] = [
            {"start_s": 0., "end_s": .5, "reason": "sampling_capacity"}]
    elif outcome == "intersection":
        planner.geometry_report.update(confirmed_intersections=[{"time_s": .4, "status": "intersection"}],
            hard_scene_residual=.003, scene_categories={"collision": .003})
    elif outcome == "conditional":
        planner.geometry_report["unresolved_intervals"] = [
            {"start_s": .1, "end_s": 1., "reason": "unmodelled_entity_motion"}]
        planner.geometry_report["scene_categories"] = {"conditional_unknown": 1.}
    curve = planner._controls_curve(zero_controls(planner))
    (_, report, _), work = record_qualification(planner, curve)
    assert report["accepted"] == (outcome != "intersection")
    assert (planner.last_validation is not None) == report["accepted"]
    assert "interval:known_geometry_unresolved" not in report["residuals"]
    assert report["collision_residuals"]["interval:scene_collision"] == (.003 if outcome == "intersection" else 0.)
    assert report["conditional_unknown_residuals"]["interval:conditional_unknown"] == float(outcome == "conditional")
    assert report["continuous_geometry"] == planner.geometry_report
    if report["accepted"]:
        assert planner.last_validation["reference_termination"]["requested_hold_start_s"] == .5
        assert planner.last_validation["continuous_geometry"] == planner.geometry_report
    assert work == {"passes": 1, "points": 3}
# END acceptance and original diagnostic records share the same parent result.


# BEGIN real nonempty scene integration exercises the inherited interval callback.
@pytest.mark.parametrize("released_owner", [False, True])
def test_interval_witness_collection_preserves_real_scene_and_parent_accounting(tmp_path, monkeypatch, released_owner):
    """Original Coal/TSDF queries, FK and motion bounds run through both paths.

    The experiment may consume an already completed entity record to form a
    necessary row. It must retain the parent's collision evidence and point
    accounting, including exact release state and adjacent relation modes.
    """
    from manipulation.preference_planner.geometry import GeometryConstraints
    from manipulation.execution_binding import plain
    from manipulation.observed_mapping import FrozenObservedField
    from manipulation.planning import _ConstrainedPlanner
    from manipulation.planning_problem import PosePath
    from manipulation.planning_scene import Attachment, CollisionOwner
    from manipulation.unified_solver import robot_tool_states
    from test_geometry import computation

    base, controls = computation(tmp_path)
    base.problem.start_progress, base.problem.end_progress = 0., 1.
    if released_owner:
        field = FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
            weights=np.ones((2, 2, 2)), origin=[-.001] * 3, voxel_size=.002,
            name="material", source="analytic observed-field unit fixture")
        base.scene.owners = (CollisionOwner("material", field,
            PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared path"),
            "observed material", moving=True,
            attachments=(Attachment((0., .5), [0., -.2, 0.], [1., 0., 0., 0.],
                                    "declared", "release"),)),)
        base.scene.required_progress_anchors = (.5,)

    times = np.array([0., .5, 1.]) * base.duration
    positions = base.domain._curve_map.evaluate(controls, times / base.duration)
    velocities = np.zeros_like(positions)
    curve = JointTrajectory(base.joint_names, times, positions, velocities,
        np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    progress = torch.tensor([[0., .5, 1.]], dtype=torch.float64)
    _, _, _, _, tool, _ = robot_tool_states(problem=base.problem, scene=base.scene,
        kinematics=base.kinematics, q=torch.tensor(positions)[None],
        dq=torch.tensor(velocities)[None], joint_names=base.joint_names,
        progress=progress, times=torch.tensor(times)[None])
    candidate_scene = base.scene.with_release_history(*tool, progress)
    original_evaluate = candidate_scene.evaluate
    original_report = deepcopy(candidate_scene.last_query_report)

    planner = PreferencePlanner.__new__(PreferencePlanner)
    planner.torch, planner.problem, planner.joint_names = torch, base.problem, base.joint_names
    planner.pin_kinematics, planner.anchors = base.kinematics, np.array([0., .5, 1.])
    planner.budget = SimpleNamespace(validation_spatial_step_m=.001)
    planner.check_request = lambda: None
    planner.geometry_constraints = GeometryConstraints(base)
    planner._current_control_curve, planner._current_full_controls = curve, controls
    planner._candidate_evidence_scene, planner._candidate_added_rows = candidate_scene, 0

    baseline_work, captured_work, clones = [], [], []
    baseline = _ConstrainedPlanner._candidate_geometry(planner, curve, candidate_scene,
        initial_states=(), remaining_points=20, record_work=lambda **value: baseline_work.append(value))
    # The baseline also performs real queries. Reset only its display report
    # so subsequent assertions concern mutation by the experiment alone.
    candidate_scene.last_query_report = original_report
    clone_for_rollout = candidate_scene.clone_for_rollout

    def retain_clone():
        clone = clone_for_rollout()
        clones.append(clone)
        return clone

    monkeypatch.setattr(candidate_scene, "clone_for_rollout", retain_clone)
    actual = planner._candidate_geometry(curve, candidate_scene, initial_states=(),
        remaining_points=20, record_work=lambda **value: captured_work.append(value))
    assert actual["confirmed_intersections"]
    assert plain(actual) == plain(baseline)
    assert captured_work == baseline_work
    assert sum(value["points"] for value in captured_work) == actual["query_points"]
    assert planner._candidate_added_rows == planner.geometry_constraints.row_count > 0
    assert candidate_scene.evaluate == original_evaluate
    assert candidate_scene.last_query_report == original_report
    clone, = clones
    assert clone is not candidate_scene and type(clone) is type(candidate_scene)
    assert clone._fixed_release_poses.keys() == candidate_scene._fixed_release_poses.keys()
    for key, pose in candidate_scene._fixed_release_poses.items():
        for actual_pose, expected_pose in zip(clone._fixed_release_poses[key], pose):
            torch.testing.assert_close(actual_pose, expected_pose, rtol=0, atol=0)
    if released_owner:
        assert candidate_scene._fixed_release_poses
        sides = [row.progress for row in planner.geometry_constraints.witnesses
                 if row.time_s == .5 * base.duration]
        assert min(sides) < .5 < max(sides)
# END the collector adds rows without another query, scene mutation or verdict.


def test_zero_free_variables_use_original_qualifier_without_slsqp(monkeypatch):
    """The unique constant curve still receives complete physical acceptance."""
    lightweight_workspace(monkeypatch)
    planner = EvidencePlanner(hold_start=0.)
    assert planner.terminal.matrix.shape == (21, 0)

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("A zero-dimensional problem must not enter SciPy")

    monkeypatch.setattr(backend, "solve_trajectory", unexpected_solver)
    seed = torch.tensor(zero_controls(planner))[None, None]
    (result, curve, report), delivered = select(planner, planner.optimize(seed))
    assert report["accepted"] and delivered[0]["accepted"]
    assert result.controls.shape == (21, 2)
    assert planner.qualification_calls == 1
    assert len(planner.solve_history) == 1
    assert not planner.solve_history[0]["solver_called"]
    assert planner.request_work["trajectory_solves"] == 1
    np.testing.assert_array_equal(curve._curve(np.linspace(0., 1., 71), 1), 0.)


@pytest.mark.parametrize("snapshot_keys, source", [
    (("latest_trial", "latest_complete_trial", "last_callback_iterate"), "diagnostic_optimizer_trial"),
    (("last_callback_iterate",), "last_solver_iterate"),
    (("latest_trial",), "diagnostic_optimizer_trial"),
])
def test_interrupted_solver_snapshots_expand_before_original_curve_export(monkeypatch, snapshot_keys, source):
    """Stored diagnostic coefficients must use the original full cubic layout."""
    lightweight_workspace(monkeypatch)
    planner = EvidencePlanner()
    full = planner.terminal.expand(np.full((planner.terminal.matrix.shape[1], 2), .01))
    reduced = planner.terminal.reduce(full)

    class Interrupted(RuntimeError):
        pass

    def interrupted(computation, initial, *, report_out, **kwargs):
        for key in snapshot_keys:
            report_out[key] = {"controls": initial.tolist(), "completed_components": ["task"]}
        raise Interrupted("original numerical deadline")

    monkeypatch.setattr(backend, "solve_trajectory", interrupted)
    with pytest.raises(Interrupted, match="original numerical deadline"):
        next(planner.optimize(torch.tensor(full)[None, None]))
    runtime, = planner.solve_history
    for key in snapshot_keys:
        np.testing.assert_allclose(runtime[key]["controls"], full, atol=2e-16)
        np.testing.assert_allclose(runtime[key]["reduced_controls"], reduced, atol=2e-16)
    diagnostic = planner.diagnostic_optimizer_candidate
    assert diagnostic is not None and planner.last_validation is None
    sample = np.linspace(0., 1., 51)
    np.testing.assert_allclose(diagnostic._curve(sample),
        planner.physical_domain.evaluate(full, sample), atol=2e-16)
    assert planner.diagnostic_optimizer_provenance["validation_status"] == "not_completed"
    assert planner.diagnostic_optimizer_provenance["source"] == source
    assert runtime["objective_query_range"] == [0, 0]


def test_diagnostic_export_failure_preserves_original_solver_interruption(monkeypatch):
    """A failed passive export must not replace the solver's original cause."""
    lightweight_workspace(monkeypatch)
    planner = EvidencePlanner()

    class Interrupted(RuntimeError):
        pass

    def interrupted(computation, initial, *, report_out, **kwargs):
        report_out["latest_complete_trial"] = {"controls": initial.tolist()}
        raise Interrupted("original solver cause")

    def broken_export(controls):
        assert controls.shape == (21, 2)
        raise ValueError("diagnostic Hermite export failed")

    monkeypatch.setattr(backend, "solve_trajectory", interrupted)
    monkeypatch.setattr(planner, "_controls_curve", broken_export)
    with pytest.raises(Interrupted, match="original solver cause"):
        next(planner.optimize(torch.tensor(zero_controls(planner))[None, None]))
    runtime, = planner.solve_history
    assert runtime["diagnostic_curve_export_error"] == {
        "error_type": "ValueError", "reason": "diagnostic Hermite export failed"}
    assert planner.diagnostic_optimizer_candidate is None
    assert planner.last_validation is None


def test_added_geometry_workspace_reaches_existing_solver_memory_admission(monkeypatch):
    """The original scene and added witness computation coexist in one cap."""
    planner = EvidencePlanner()
    planner.geometry_constraints.workspace_bytes = lambda: {"forward": 23, "reverse": 31}
    monkeypatch.setattr(backend, "_planning_workspace_bytes",
                        lambda *args, **kwargs: SimpleNamespace(pending_cpu_bytes=11))

    def completed(computation, initial, *, pending_geometry_bytes, iterations, report_out, **kwargs):
        assert pending_geometry_bytes == 65
        assert iterations == 2
        report_out.update(solver_called=True, state="completed")
        return ConstrainedSolution(initial.copy(), report_out)

    monkeypatch.setattr(backend, "solve_trajectory", completed)
    result = next(planner.optimize(torch.tensor(zero_controls(planner))[None, None], iterations=2))
    assert result.controls.shape == (21, 2)
    assert planner.solve_history[0]["geometry_constraint_workspace_bytes"] == {"forward": 23, "reverse": 31}


@pytest.mark.parametrize("new_rows, violation, expected_calls", [(1, .1, 2), (0, .1, 1), (1, 0., 1)])
def test_exchange_uses_shared_cumulative_solve_ledger(monkeypatch, new_rows, violation, expected_calls):
    """Only a rejected curve with new rows spends a second request solve slot."""
    lightweight_workspace(monkeypatch)
    planner = EvidencePlanner(baseline=7)
    planner.rows_per_qualification, planner.task_violation = new_rows, violation
    calls = []

    def completed(computation, initial, *, report_out, **kwargs):
        calls.append(initial.copy())
        computation.trial_history.append({"controls": initial.tolist()})
        report_out.update(solver_called=True, success=True, state="completed")
        return ConstrainedSolution(initial.copy(), report_out)

    monkeypatch.setattr(backend, "solve_trajectory", completed)
    (_, _, report), delivered = select(planner,
        planner.optimize(torch.tensor(zero_controls(planner))[None, None]))
    assert len(calls) == len(planner.solve_history) == len(delivered) == expected_calls
    assert planner.request_work["trajectory_solves"] == 7 + expected_calls
    assert [entry for entry in planner.checks if entry[0] == "trajectory_solves"] == (
        [("trajectory_solves", 1)] if expected_calls == 2 else [])
    assert report["accepted"] == (violation == 0.)
    assert [item["objective_query_range"] for item in planner.solve_history] == (
        [[0, 1], [1, 2]] if expected_calls == 2 else [[0, 1]])
    if expected_calls == 2:
        assert delivered[1]["seed_provenance"] == "same_curve_registered_geometry_constraints"


@pytest.mark.parametrize("new_rows", [0, 1])
def test_multiple_original_seeds_preserve_batch_accounting_and_result_identity(monkeypatch, new_rows):
    """Original K seeds share a batch; an exchange consumes its next slot.

    FUNCTO's trajectory_solves counter counts original seed batches.  Serial
    evaluation of a second seed in the admitted batch therefore remains
    permitted after an exchange, with its own original provenance.  This test
    checks that ordering without changing the established budget meaning.
    """
    lightweight_workspace(monkeypatch)
    planner = EvidencePlanner(baseline=7)
    planner.seed_strategies = ["source_a", "source_b"]
    planner.task_violation, planner.rows_per_qualification = .1, new_rows
    calls = []

    def completed(computation, initial, *, report_out, **kwargs):
        calls.append(initial.copy())
        report_out.update(solver_called=True, success=True, state="completed")
        return ConstrainedSolution(initial.copy(), report_out)

    monkeypatch.setattr(backend, "solve_trajectory", completed)
    seeds = torch.tensor(np.stack([zero_controls(planner)] * 2))[None]
    (_, _, report), delivered = select(planner, planner.optimize(seeds))
    assert not report["accepted"]
    assert len(calls) == 2 + new_rows
    assert planner.request_work["trajectory_solves"] == 8 + new_rows
    assert [item["seed_provenance"] for item in delivered] == (
        ["source_a", "same_curve_registered_geometry_constraints", "source_b"]
        if new_rows else ["source_a", "source_b"])
    assert [item["original_seed_index"] for item in planner.solve_history] == (
        [0, 0, 1] if new_rows else [0, 1])
    assert [item["exchange_index"] for item in planner.solve_history] == (
        [0, 1, 0] if new_rows else [0, 0])

