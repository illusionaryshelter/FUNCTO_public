"""CPU end-to-end constrained curves use one model, clock and task allowed set."""

# BEGIN physical controls, real Pinocchio FK and declared task rows meet in SLSQP.
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pinocchio as pin
import pytest
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation
import torch

from manipulation.physical_joint_domain import CubicJointDomain
from manipulation.pin_kinematics import PinKinematics
from manipulation.resource_admission import ResourceLimitError
from manipulation.planning_problem import (
    JointCoupling, MotionPhase, PlanningEvent, PlanningProblem, PoseEventRequirement, PosePath,
)
from manipulation.unified_solver import (
    TrajectoryComputation, motion_cost, robot_tool_states, solve_trajectory, task_jacobian, time_mean_weights,
)


URDF = """<robot name="constrained_curve_fixture">
<link name="base"/><link name="x_link"/><link name="y_link"/><link name="tool"/>
<joint name="x" type="prismatic"><parent link="base"/><child link="x_link"/>
<axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/></joint>
<joint name="y" type="prismatic"><parent link="x_link"/><child link="y_link"/>
<axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/></joint>
<joint name="turn" type="revolute"><parent link="y_link"/><child link="tool"/>
<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="10" velocity="1"/></joint>
</robot>"""


def motion_configuration():
    return dict(weight=[2., 3., 4., 5., 0.], activation_distance=[.05]*5,
        squared_l2_regularization_weight=[.1, .01, .001, 0., 0.],
        retime_weights=True, retime_regularization_weights=True)


def computation(*, extra_parameters=(), entry_yaw_event=False, terminal_yaw_tolerance=0.,
                production_weights=False):
    """A reachable calibrated pose, exact axis angle, actuator coupling and stop."""
    names = ("x", "y", "turn")
    model = pin.buildModelFromXML(URDF)
    base = np.eye(4)
    base[:3, :3] = Rotation.from_rotvec([.2, -.1, .3]).as_matrix()
    base[:3, 3] = [.4, -.2, .1]
    tool = np.eye(4)
    tool[:3, :3] = Rotation.from_rotvec([.1, .2, -.1]).as_matrix()
    tool[:3, 3] = [.04, -.03, .02]
    if entry_yaw_event:
        base, tool = np.eye(4), np.eye(4)
    kinematics = PinKinematics(model, frames=("tool",), T_reference_base=base)
    entry, rate = np.array([.02, .01, .03]), np.array([.01, .005, .02])
    if entry_yaw_event:
        entry[2] = 0.
    goal = np.array([.2, .1, .15])
    goal_link = np.eye(4)
    goal_link[:3, :3] = Rotation.from_rotvec([0., 0., goal[2]]).as_matrix()
    goal_link[:3, 3] = [goal[0], goal[1], 0.]
    target = base @ goal_link @ tool
    pose_events = (PoseEventRequirement(1., np.eye(4), target, .005, .02,
        "declared terminal material relation", yaw_axis=tool[:3, :3].T @ [0., 0., 1.],
        yaw_tolerance_rad=terminal_yaw_tolerance),)
    if entry_yaw_event:
        entry_target = np.eye(4)
        entry_target[:2, 3] = entry[:2]
        pose_events = (PoseEventRequirement(0., np.eye(4), entry_target, .005, .02,
            "declared entrance material relation", yaw_axis=[0., 0., 1.],
            yaw_tolerance_rad=0.), *pose_events)
    reference = PosePath([0., 1.], np.repeat(np.eye(4)[None], 2, axis=0), "ordinary seed reference")
    problem = PlanningProblem(names, entry, rate, "tool", "world", base, tool,
        (MotionPhase("reference", reference),),
        pose_events=pose_events,
        couplings=(JointCoupling(("x", "y"), [[-.5, 1.]], [0.], "actuator coupling"),),
        events=(PlanningEvent("stop", "instance", 1., 1, "hold", "explicit endpoint stop", stop=True),))
    bounds = np.repeat(np.array([[[-1.], [1.]], [[-1.], [1.]],
                                 [[-4.], [4.]], [[-16.], [16.]]]), 3, axis=2)
    domain = CubicJointDomain(torch.tensor(bounds), torch.tensor(entry), torch.tensor(rate), .5, 6)
    ordinary = np.linspace(0., 1., 9)
    parameters = np.unique(np.r_[ordinary, problem.anchors, extra_parameters])
    indices = np.searchsorted(parameters, ordinary)
    # This fixture declares no scene entities. The separate real geometry
    # objective tests cover TSDF/native pairs; this zero has no collision claim.
    scene = SimpleNamespace(gripper=None, optimization_objective=lambda position, quaternion, progress,
        joint_position, joint_names, **kwargs: joint_position.sum()*0.)
    configuration = motion_configuration()
    constraint_weight = 10.
    if production_weights:
        from curobo._src.util_file import load_yaml, join_path, get_task_configs_path
        configuration = load_yaml(join_path(get_task_configs_path(),
            "trajopt/lbfgs_bspline_trajopt.yml"))["rollout"]["cost_cfg"]["cspace_cfg"]
        constraint_weight = 1e5
    bound = TrajectoryComputation(problem=problem, scene=scene, kinematics=kinematics, domain=domain,
        joint_names=names, parameters=parameters, progress=parameters, ordinary_indices=indices,
        motion_configuration=configuration, native_dt=.1, constraint_weight=constraint_weight,
        max_host_rss_bytes=8*1024**3)
    initial = np.broadcast_to(entry, (domain.knots, 3)).copy()
    return bound, initial


def test_vectorized_task_jacobian_through_pin_matches_scalar_vjp_and_finite_difference():
    bound, initial = computation()
    rng = np.random.default_rng(12)
    controls = torch.tensor(initial + rng.normal(0., .03, initial.shape), requires_grad=True)
    function = lambda value: torch.cat(bound.task(value))
    vectorized = torch.autograd.functional.jacobian(function, controls, vectorize=True)
    ordinary = torch.autograd.functional.jacobian(function, controls, vectorize=False)
    torch.testing.assert_close(vectorized, ordinary, rtol=1e-12, atol=1e-12)
    epsilon = 1e-6
    differences = torch.stack([(function(controls+epsilon*direction)-function(controls-epsilon*direction))/(2*epsilon)
        for direction in torch.eye(controls.numel(), dtype=controls.dtype).reshape(-1, *controls.shape)], -1)
    torch.testing.assert_close(vectorized.reshape(len(differences), -1), differences,
                               rtol=2e-6, atol=2e-8)
    # The production path differentiates a scalar reverse product by each
    # control column; it returns the same complete first-derivative matrix.
    actual = task_jacobian(function(controls), controls,
                           check_request=lambda: None, admit_work=lambda: None)
    np.testing.assert_allclose(actual, ordinary.detach().numpy().reshape(len(differences), -1),
                               rtol=1e-12, atol=1e-12)
    assert bound.task(controls)[0].numel() == 1


# BEGIN ordinary-reference and fixed-entry requests retain exact zero task rows.
@pytest.mark.parametrize("rows", [0, 3, 9])
def test_scalar_jacobian_preserves_an_empty_or_constant_vector(rows):
    controls = torch.arange(6., dtype=torch.float64).reshape(2, 3).requires_grad_()
    values = torch.full((rows,), 2., dtype=controls.dtype)
    admitted = []
    jacobian = task_jacobian(values, controls, check_request=lambda: None,
                            admit_work=lambda: admitted.append(True))
    assert jacobian.shape == (rows, controls.numel()) and not jacobian.any() and admitted == [True]
    assert bool((values == 2.).all())


@pytest.mark.parametrize("fixed_entry", [False, True])
def test_control_column_jacobian_handles_only_constant_task_rows(fixed_entry):
    bound, initial = computation(entry_yaw_event=fixed_entry)
    problem = replace(bound.problem, pose_events=bound.problem.pose_events[:1] if fixed_entry else (),
                      couplings=(), events=())
    bound = TrajectoryComputation(problem=problem, scene=bound.scene, kinematics=bound.kinematics,
        domain=bound.domain, joint_names=bound.joint_names, parameters=bound.parameters,
        progress=bound.progress.numpy()[0], ordinary_indices=bound.ordinary_indices,
        motion_configuration=bound.motion_configuration, native_dt=bound.native_dt,
        constraint_weight=bound.constraint_weight, max_host_rss_bytes=bound.max_host_rss_bytes)
    controls = torch.tensor(initial, requires_grad=True)
    eq, iq = bound.task(controls)
    values = torch.cat((eq[bound.variable_equality_rows], iq))
    jacobian = task_jacobian(values, controls, check_request=lambda: None, admit_work=lambda: None)
    assert jacobian.shape == (values.numel(), controls.numel()) and not jacobian.any()
    result = solve_trajectory(bound, initial, report_out={}, iterations=1,
        check_request=lambda: None, pending_geometry_bytes=0)
    assert result.report["solver_called"] and result.report["solver_task_equalities"] == 0
    if fixed_entry:
        assert result.report["fixed_entry_equalities"] == {"function_pose:0:yaw": 0.}
    else:
        assert result.report["solver_task_inequalities"] == 0
        assert result.report["task_jacobian_method"] == "none"
# END constant declarations need no exception handling or alternate algorithm.


# BEGIN the numerical interface submits only proven variable declaration rows.
def test_few_event_rows_use_pin_scalar_vjp_and_cached_original_values(monkeypatch):
    import manipulation.unified_solver as solver
    bound, initial = computation(terminal_yaw_tolerance=np.pi)
    bound.problem = replace(bound.problem, couplings=(), events=())
    controls = torch.tensor(initial+.03, dtype=torch.float64)
    first_event_row = 2*len(bound.parameters)
    original = bound.task(controls)[1][first_event_row:].numpy()
    derivative = torch.autograd.functional.jacobian(
        lambda value: bound.task(value)[1][first_event_row:], controls).numpy().reshape(2, -1)
    report = {}

    def inspect(function, x0, **kwargs):
        assert len(kwargs["constraints"]) == 2
        task = kwargs["constraints"][1]
        np.testing.assert_allclose(task["fun"](x0), -original, rtol=0., atol=0.)
        np.testing.assert_allclose(task["jac"](x0), -derivative, rtol=2e-13, atol=2e-13)
        recorded = report["evaluation_seconds"].copy()
        for _ in range(3):
            task["fun"](x0)
            task["jac"](x0)
            function(x0)
        assert report["evaluation_seconds"] == recorded
        return SimpleNamespace(x=x0.copy(), fun=function(x0)[0], success=True,
                               status=0, message="selected task interface checked", nit=0)

    monkeypatch.setattr(solver, "minimize", inspect)
    result = solve_trajectory(bound, controls.numpy(), report_out=report, iterations=1,
        check_request=lambda: None, pending_geometry_bytes=0)
    assert report["complete_task_inequalities"] == first_event_row+2
    assert report["solver_task_inequalities"] == 2
    assert report["task_jacobian_method"] == "scalar_task_rows"
    assert report["task_evaluations"] == 2 and report["objective_evaluations"] == 1
    assert report["evaluation_seconds"]["task"] > 0.
    assert [row["key"] for row in report["fixed_task_constraints"]] == ["progress:lower", "progress:upper"]
    for row in report["fixed_task_constraints"]:
        assert row["row_indices"] == list(range(len(bound.parameters)))
        assert all(row["finite"]) and row["feasible"]
    np.testing.assert_array_equal(result.controls, controls.numpy())


@pytest.mark.parametrize("progress", [-.1, float("nan")])
def test_invalid_fixed_progress_is_reported_before_the_solver(progress, monkeypatch):
    import manipulation.unified_solver as solver
    bound, initial = computation()
    bound.progress[0, 0] = progress
    def forbidden_solver(*args, **kwargs):
        raise AssertionError("A fixed progress violation cannot be changed by controls")
    monkeypatch.setattr(solver, "minimize", forbidden_solver)
    result = solve_trajectory(bound, initial, report_out={}, iterations=1,
        check_request=lambda: None, pending_geometry_bytes=0)
    assert not result.report["success"] and not result.report["solver_called"]
    record = next(row for row in result.report["fixed_task_constraints"] if row["key"] == "progress:lower")
    assert record["row_indices"][0] == 0 and not record["feasible"]
    assert record["finite"][0] == bool(np.isfinite(progress))
    if np.isfinite(progress):
        assert record["values"][0] == pytest.approx(.1)
    assert result.report["task_evaluations"] == 1 and result.report["objective_evaluations"] == 0
    np.testing.assert_array_equal(result.controls, initial)
# END missing/violated fixed inputs keep their original component evidence.


def test_full_objective_gradient_and_event_grid_keep_physical_motion_weights():
    bound, initial = computation()
    expanded, _ = computation(extra_parameters=(.137, .37, .83))
    values = torch.tensor(initial + np.random.default_rng(7).normal(0., .02, initial.shape), requires_grad=True)
    actual = bound.objective_components(values, include_geometry=False)
    other = expanded.objective_components(values, include_geometry=False)
    torch.testing.assert_close(actual["motion_quality"], other["motion_quality"], rtol=0, atol=0)
    # Check each reported term independently; optional event centering is
    # absent from this feasibility objective and remains in candidate scoring.
    for name in actual:
        assert torch.autograd.gradcheck(
            lambda value: bound.objective_components(value, include_geometry=False)[name],
            (values,), eps=1e-6, atol=3e-6, rtol=3e-6)


# BEGIN first-result optimization preserves task rows while dropping quality centering.
def test_search_uses_feasibility_preferences_with_unchanged_task_values_and_jacobian():
    bound, initial = computation()
    controls = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    rows_before = torch.cat(bound.task(controls))
    derivative_before = task_jacobian(rows_before, controls,
        check_request=lambda: None, admit_work=lambda: None)
    costs = bound.objective_components(controls)
    assert set(costs) == {'motion_quality', 'task_preferences', 'observed_geometry'}
    assert costs['task_preferences'] == 0.
    feasibility_gradient, = torch.autograd.grad(sum(costs.values()), controls)
    q, dq, _, _ = bound.states(controls)
    _, _, _, _, tool, twist = bound.robot_states(q, dq)
    quality = bound.problem.optimizer_preferences(*tool, bound.progress,
        trajectory_time=bound.times, tool_linear_velocity=twist[0],
        tool_angular_velocity=twist[1], phase='quality').sum()*bound.constraint_weight
    quality_gradient, = torch.autograd.grad(quality, controls)
    assert quality > 0. and quality_gradient.abs().max() > 1.
    assert not torch.equal(feasibility_gradient, feasibility_gradient+quality_gradient)
    rows_after = torch.cat(bound.task(controls))
    derivative_after = task_jacobian(rows_after, controls,
        check_request=lambda: None, admit_work=lambda: None)
    torch.testing.assert_close(rows_before, rows_after, rtol=0., atol=0.)
    np.testing.assert_array_equal(derivative_before, derivative_after)
# END the same hard set remains independent of the optional recorded quality score.


# BEGIN interrupted real SLSQP retains actual same-control computation evidence.
def test_budget_interrupt_preserves_evaluated_trial_counts_and_geometry_identity():
    from manipulation.planning import PlanningWorkExhausted
    bound, initial = computation()
    report, actual = {}, dict(objective=0, task=0, physical=0)
    original_objective, original_task = bound.objective_components, bound.task
    original_physical = bound.domain.signed_constraints

    def objective(controls):
        actual["objective"] += 1
        values = original_objective(controls)
        # This provenance marker labels the real objective evaluation. The
        # fixture has no scene entities and makes no collision assertion.
        bound.geometry_report.update(evaluated_controls=controls.detach().tolist())
        return values

    def task(controls, *, solver_rows_out=None):
        actual["task"] += 1
        return original_task(controls, solver_rows_out=solver_rows_out)

    def physical(controls):
        actual["physical"] += 1
        return original_physical(controls)

    bound.objective_components, bound.task, bound.domain.signed_constraints = objective, task, physical
    error = PlanningWorkExhausted("Existing request budget ended at an actual SLSQP iterate")

    def check():
        snapshot = report.get("latest_complete_trial")
        if (report.get("callback_iterations", 0) and snapshot is not None
                and not np.array_equal(snapshot["controls"], initial)):
            raise error

    with pytest.raises(PlanningWorkExhausted) as raised:
        solve_trajectory(bound, initial, iterations=160, check_request=check,
                         pending_geometry_bytes=0, report_out=report)
    assert raised.value is error
    assert report["state"] == "interrupted" and report["solver_called"]
    assert report["completed_evaluations"] == actual
    assert {kind: report[kind + "_evaluations"] for kind in actual} == actual
    assert report["objective_scale"] >= 1. and report["elapsed_seconds"] > 0
    trial = report["latest_complete_trial"]
    assert not np.array_equal(trial["controls"], initial)
    assert set(trial["completed_components"]) == {"objective", "task", "physical"}
    np.testing.assert_array_equal(trial["controls"], trial["geometry_objective"]["evaluated_controls"])
    np.testing.assert_array_equal(trial["controls"], report["last_callback_iterate"]["controls"])
    bound.geometry_report["evaluated_controls"] = initial.tolist()
    assert trial["geometry_objective"]["evaluated_controls"] != initial.tolist()


def test_interrupted_objective_keeps_partial_trial_without_borrowing_previous_geometry(monkeypatch):
    import manipulation.unified_solver as solver
    from manipulation.planning import PlanningWorkExhausted
    bound, initial = computation()
    report = {}
    original = bound.objective_components
    error = PlanningWorkExhausted("Budget ended inside an objective evaluation")
    calls = 0
    clock = [0.]
    monkeypatch.setattr(solver, "perf_counter", lambda: clock[0])

    def objective(controls):
        nonlocal calls
        calls += 1
        clock[0] += calls
        if calls == 2:
            raise error
        return original(controls)

    bound.objective_components = objective
    with pytest.raises(PlanningWorkExhausted) as raised:
        solve_trajectory(bound, initial, iterations=160, check_request=lambda: None,
                         pending_geometry_bytes=0, report_out=report)
    assert raised.value is error
    assert report["objective_evaluations"] == 2 and report["completed_evaluations"]["objective"] == 1
    assert report["evaluation_seconds"]["objective"] == 3.
    assert not np.array_equal(report["latest_trial"]["controls"], initial)
    assert "objective" not in report["latest_trial"]["completed_components"]
    assert "geometry_objective" not in report["latest_trial"]
# END unfinished evaluation and completed evidence retain distinct identities.


@pytest.mark.parametrize("production_weights", [False, True])
def test_slsqp_from_task_infeasible_seed_preserves_entry_and_exports_original_polynomial(production_weights):
    bound, initial = computation(production_weights=production_weights)
    eq_initial, iq_initial = bound.task(torch.tensor(initial))
    assert eq_initial.abs().max() > .1 and iq_initial.max() > .1
    requests = []
    result = solve_trajectory(bound, initial, report_out={}, iterations=160, check_request=lambda: requests.append(None), pending_geometry_bytes=0)
    assert result.report["success"], result.report
    assert result.report["maximum_control_update"] > .05
    assert result.report["objective_evaluations"] > 1 and requests
    equality, inequality = bound.task(torch.tensor(result.controls))
    # These bounds measure numerical solver accuracy for a zero-width yaw.
    # Strict original acceptance is exercised separately with declared finite
    # yaw bounds; library success itself does not certify an exact equality.
    assert equality.abs().max() < 1e-8
    assert inequality.max() < 1e-8
    assert bound.domain.signed_constraints(result.controls)[0].max() < 1e-8
    times, position, velocity = bound.domain._curve_map.hermite_nodes(
        result.controls, bound.problem.start_q, bound.problem.start_dq)
    curve = CubicHermiteSpline(times, position, velocity)
    np.testing.assert_array_equal(curve(0.), bound.problem.start_q)
    np.testing.assert_array_equal(curve(0., 1), bound.problem.start_dq)
    query = bound.times.numpy()[0]
    # Saved Hermite values are the original polynomial, including physical
    # derivatives and the explicitly declared stop at its event instant.
    for derivative, expected in enumerate(bound.states(torch.tensor(result.controls))):
        np.testing.assert_allclose(curve(query, derivative), expected.numpy()[0], atol=2e-14, rtol=1e-12)
    assert np.max(np.abs(curve(times[-1], 1))) <= bound.problem.numerical.velocity + 1e-8
    q = torch.tensor(curve(query))[None]
    dq = torch.tensor(curve(query, 1))[None]
    q, dq, names, _, tool, twist = robot_tool_states(problem=bound.problem, scene=bound.scene,
        kinematics=bound.kinematics, q=q, dq=dq, joint_names=bound.joint_names,
        progress=bound.progress, times=bound.times)
    residuals = bound.problem.residuals(*tool, bound.progress, q, names, dq,
        trajectory_time=bound.times, tool_linear_velocity=twist[0], tool_angular_velocity=twist[1])
    assert max(float(value.max()) for value in residuals.values()) < 1e-8


def test_saved_curve_robot_tool_computation_uses_its_own_queries_without_rebinding():
    bound, initial = computation()
    controls = initial + np.random.default_rng(4).normal(0., .02, initial.shape)
    times_before = bound.times.clone()
    parameters = np.array([0., .23, .61, 1.])
    q = torch.tensor(bound.domain.evaluate(controls, parameters))[None]
    dq = torch.tensor(bound.domain.evaluate(controls, parameters, 1))[None]
    result = robot_tool_states(problem=bound.problem, scene=bound.scene, kinematics=bound.kinematics,
        q=q, dq=dq, joint_names=bound.joint_names, progress=torch.tensor(parameters)[None],
        times=torch.tensor(parameters*bound.duration)[None])
    assert result[4][0].shape == (1, 4, 3)
    torch.testing.assert_close(bound.times, times_before, rtol=0, atol=0)
    # Independent model multiplication checks that base calibration is applied
    # once and the TCP offset uses the tool link's actual rotation.
    for index, state in enumerate(q.numpy()[0]):
        link = np.eye(4)
        link[:3, :3] = Rotation.from_rotvec([0., 0., state[2]]).as_matrix()
        link[:3, 3] = [state[0], state[1], 0.]
        expected = bound.problem.base_pose @ link @ bound.problem.tool_transform
        np.testing.assert_allclose(result[4][0][0, index], expected[:3, 3], atol=1e-14)


def test_host_capacity_is_checked_before_any_dense_task_jacobian(monkeypatch):
    bound, initial = computation()
    bound.max_host_rss_bytes = 1
    def forbidden_derivative(*args, **kwargs):
        raise AssertionError("Task derivatives must follow host-memory admission")
    monkeypatch.setattr(torch.autograd, "grad", forbidden_derivative)
    with pytest.raises(ResourceLimitError, match="host_process_rss"):
        solve_trajectory(bound, initial, report_out={}, iterations=1, check_request=lambda: None, pending_geometry_bytes=0)


# BEGIN independent capacity checks cannot admit a jointly excessive peak.
def test_scene_and_solver_storage_are_admitted_as_one_live_host_peak(monkeypatch):
    import psutil
    import manipulation.resource_admission as admission

    bound, initial = computation()
    baseline = solve_trajectory(bound, initial, report_out={}, iterations=1,
        check_request=lambda: None, pending_geometry_bytes=0)
    solver_bytes = sum(baseline.report["solver_storage_bytes"].values())
    pending_scene = solver_bytes
    resident = 1024**3
    bound.max_host_rss_bytes = resident + solver_bytes + pending_scene//2
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=resident), children=lambda **_: [])
    monkeypatch.setattr(admission, "_HOST_PROCESS_ROOT_IDENTITY", None)
    monkeypatch.setattr(psutil, "Process", lambda: process)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=16*1024**3))
    # Both former independent requests fit. Their simultaneous arrays do not.
    for increment in (solver_bytes, pending_scene):
        admission.require_host_memory(increment, pending_existing_bytes=0,
                                      max_rss_bytes=bound.max_host_rss_bytes)
    def forbidden_derivative(*args, **kwargs):
        raise AssertionError("A jointly excessive scene/solver peak must precede derivatives")
    monkeypatch.setattr(torch.autograd, "grad", forbidden_derivative)
    with pytest.raises(ResourceLimitError, match="host_process_rss") as caught:
        solve_trajectory(bound, initial, report_out={}, iterations=1, check_request=lambda: None,
                         pending_geometry_bytes=pending_scene)
    assert caught.value.required_bytes == resident + solver_bytes + pending_scene
# END no large buffer is allocated to verify the shared eight-GiB boundary.


# BEGIN established workspaces belong to RSS on subsequent task callbacks.
def test_repeated_task_admission_counts_resident_solver_and_scene_storage_once(monkeypatch):
    import psutil
    import manipulation.resource_admission as admission

    bound, initial = computation()
    baseline = solve_trajectory(bound, initial, report_out={}, iterations=1,
        check_request=lambda: None, pending_geometry_bytes=0)
    solver_bytes = sum(baseline.report["solver_storage_bytes"].values())
    pending_scene = solver_bytes
    resident = 1024**3
    requests = []
    actual_require = admission.require_host_memory

    def admitted_increment(required_bytes, **kwargs):
        requests.append(required_bytes)
        actual_require(required_bytes, **kwargs)

    # All initial scene/solver arrays fit simultaneously. On each later
    # callback they are resident, leaving room for this newly returned task
    # matrix and its temporary column, not another entire solver allocation.
    initial_peak = solver_bytes + pending_scene
    bound.max_host_rss_bytes = resident + initial_peak + solver_bytes//2
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(
        rss=resident if len(requests) == 1 else resident+initial_peak), children=lambda **_: [])
    monkeypatch.setattr(admission, "_HOST_PROCESS_ROOT_IDENTITY", None)
    monkeypatch.setattr(psutil, "Process", lambda: process)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=16*1024**3))
    monkeypatch.setattr("manipulation.unified_solver.require_host_memory", admitted_increment)
    result = solve_trajectory(bound, initial, report_out={}, iterations=2,
        check_request=lambda: None, pending_geometry_bytes=pending_scene)
    assert result.report["solver_called"] and len(requests) >= 3
    assert requests[0] == initial_peak and all(value < solver_bytes//2 for value in requests[1:])
# END ordinary task queries preserve the same physical memory limit.


# BEGIN fixed entry equations and objective units preserve the original problem.
@pytest.mark.parametrize("terminal_yaw_tolerance", [0., .02])
def test_fixed_entry_equalities_are_verified_then_eliminated_by_curve_structure(terminal_yaw_tolerance):
    bound, initial = computation(entry_yaw_event=True, terminal_yaw_tolerance=terminal_yaw_tolerance,
                                 production_weights=True)
    np.testing.assert_array_equal(bound.fixed_equality_rows, [0])
    original = bound.task(torch.tensor(initial))[0]
    perturbed = bound.task(torch.tensor(initial+.2))[0]
    assert original[0] == perturbed[0] == 0.
    result = solve_trajectory(bound, initial, report_out={}, iterations=160, check_request=lambda: None, pending_geometry_bytes=0)
    assert result.report["success"], result.report
    assert result.report["fixed_entry_equalities"] == {"function_pose:0:yaw": 0.}
    assert result.report["solver_task_equalities"] == int(terminal_yaw_tolerance == 0.)
    equality, inequality = bound.task(torch.tensor(result.controls))
    assert equality[0] == 0. and equality.abs().max() < 1e-8
    assert inequality.max() < 1e-8
    assert bound.domain.signed_constraints(result.controls)[0].max() < 1e-8
    if terminal_yaw_tolerance > 0.:
        assert not equality.any()
        # Feasibility search may stop infinitesimally outside an active bound.
        # Its SLSQP success is numerical evidence, not strict saved-curve acceptance.
        assert inequality.max() > 0.
        q, dq, _, _ = bound.states(torch.tensor(result.controls))
        q, dq, names, _, tool, twist = bound.robot_states(q, dq)
        strict = bound.problem.residuals(*tool, bound.progress, q, names, dq,
            trajectory_time=bound.times, tool_linear_velocity=twist[0], tool_angular_velocity=twist[1])
        assert any(bool((value > 0.).any()) for value in strict.values())


def test_production_weight_solver_success_keeps_strict_task_rejection_separate():
    bound, initial = computation(terminal_yaw_tolerance=.02, production_weights=True)
    result = solve_trajectory(bound, initial, report_out={}, iterations=160, check_request=lambda: None, pending_geometry_bytes=0)
    assert result.report["success"], result.report
    controls = torch.tensor(result.controls)
    equality, inequality = bound.task(controls)
    # BEGIN numerical library termination cannot authorize an outside boundary point.
    assert not equality.numel() and 0. < inequality.max() < 1e-8
    q, dq, _, _ = bound.states(controls)
    q, dq, names, _, tool, twist = bound.robot_states(q, dq)
    strict = bound.problem.residuals(*tool, bound.progress, q, names, dq,
        trajectory_time=bound.times, tool_linear_velocity=twist[0], tool_angular_velocity=twist[1])
    assert any(bool((value > 0.).any()) for value in strict.values())
    # END original raw residuals retain their strict nonpositive acceptance rule.
    assert bound.domain.signed_constraints(result.controls)[0].max() < 1e-8


def test_violated_fixed_entry_equality_is_retained_and_prevents_the_library_call(monkeypatch):
    import manipulation.unified_solver as solver
    bound, initial = computation(entry_yaw_event=True)
    bound.problem.pose_events[0].target[:3, :3] = Rotation.from_rotvec([0., 0., .01]).as_matrix()
    def forbidden_solve(*args, **kwargs):
        raise AssertionError("A fixed violated entry equation cannot be optimized")
    monkeypatch.setattr(solver, "minimize", forbidden_solve)
    result = solve_trajectory(bound, initial, report_out={}, iterations=160, check_request=lambda: None, pending_geometry_bytes=0)
    assert not result.report["success"] and not result.report["solver_called"]
    assert result.report["iterations"] == 0
    assert result.report["fixed_entry_equalities"]["function_pose:0:yaw"] == pytest.approx(-.01)
    np.testing.assert_array_equal(result.controls, initial)
    assert bound.task(torch.tensor(result.controls))[0][0].abs() > .009


def test_a_zero_trial_jacobian_does_not_classify_a_future_equation_as_fixed_entry():
    bound, initial = computation(entry_yaw_event=True)
    # This robot rotates around z. Its material x-axis twist equation has a
    # zero derivative at this trial, but the event still occurs after t=0.
    bound.problem.pose_events[1].yaw_axis = np.array([1., 0., 0.])
    jacobian = torch.autograd.functional.jacobian(lambda value: bound.task(value)[0],
                                                  torch.tensor(initial))
    assert not jacobian[1].any()
    np.testing.assert_array_equal(bound.fixed_equality_rows, [0])
    np.testing.assert_array_equal(bound.variable_equality_rows, [1])
    result = solve_trajectory(bound, initial, report_out={}, iterations=1,
        check_request=lambda: None, pending_geometry_bytes=0)
    assert result.report["solver_called"] and result.report["solver_task_equalities"] == 1


def test_one_positive_objective_divisor_preserves_values_derivatives_and_order(monkeypatch):
    import manipulation.unified_solver as solver
    bound, initial = computation(production_weights=True)
    recorded = []
    def inspect_objective(function, x0, **kwargs):
        for point in (x0, x0+.001):
            controls = torch.tensor(point.reshape(initial.shape), requires_grad=True)
            raw = sum(bound.objective_components(controls).values())
            gradient = torch.autograd.grad(raw, controls)[0].detach().numpy().ravel()
            value, derivative = function(point)
            recorded.append((float(raw.detach()), gradient, value, derivative))
        # This test isolates the numerical interface; the preceding tests run
        # the actual SLSQP implementation and validate its resulting curves.
        return SimpleNamespace(x=x0.copy(), fun=recorded[0][2], success=True,
                               status=0, message="objective interface checked", nit=0)
    monkeypatch.setattr(solver, "minimize", inspect_objective)
    result = solve_trajectory(bound, initial, report_out={}, iterations=1, check_request=lambda: None, pending_geometry_bytes=0)
    divisor = result.report["objective_scale"]
    assert divisor > 1.
    assert divisor == max(1., abs(recorded[0][0]), float(np.abs(recorded[0][1]).max()))
    for raw, gradient, value, derivative in recorded:
        assert value*divisor == pytest.approx(raw, rel=2e-16)
        np.testing.assert_allclose(derivative*divisor, gradient, rtol=2e-16, atol=1e-12)
    assert np.sign(recorded[1][0]-recorded[0][0]) == np.sign(recorded[1][2]-recorded[0][2])
    assert result.report["objective"] == pytest.approx(recorded[0][0], rel=2e-16)
# END no constraint, original tolerance or physical candidate metric is rescaled.


@pytest.mark.parametrize("retime", [False, True])
def test_motion_cost_matches_installed_state_kernel_value_and_weighted_gradients(retime):
    from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
    from curobo._src.cost.cost_cspace_state import StateCSpaceCost
    from curobo._src.robot.types.joint_limits import JointLimits
    from curobo._src.state.state_joint import JointState
    from curobo.types import DeviceCfg

    configuration = dict(weight=[2., 3., 5., 7., 0.], activation_distance=[.1, .2, .15, .1, 0.],
        squared_l2_regularization_weight=[.4, .6, .8, 0., 0.],
        retime_weights=retime, retime_regularization_weights=retime)
    bounds = torch.tensor([[[-1., -2.], [1., 2.]], [[-2., -3.], [2., 3.]],
                           [[-4., -5.], [4., 5.]], [[-6., -7.], [6., 7.]]], dtype=torch.float32)
    device = DeviceCfg("cpu")
    limits = JointLimits(["a", "b"], *bounds, bounds[3], device)
    config = CSpaceCostCfg(**configuration, cost_type="STATE", dof=2,
                          device_cfg=device, joint_limits=limits, use_grad_input=True)
    native = StateCSpaceCost(config)
    native.setup_batch_tensors(1, 4)
    values = torch.tensor([[[[.9, -1.7], [1.1, -.1], [-.9, 1.8], [.4, -.8]]],
                          [[[1.7, -2.6], [2.2, -.5], [-1.8, 2.8], [.4, -.8]]],
                          [[[3.2, -4.4], [4.1, -.1], [-3.6, 4.6], [.4, -.8]]],
                          [[[5.8, -6.5], [6.1, -.1], [-5.6, 6.6], [.4, -.8]]]],
                         dtype=torch.float32, requires_grad=True)
    native_dt = .2
    state = JointState(values[0], values[1], values[2], ["a", "b"], values[3],
                       dt=torch.tensor([native_dt]))
    weights = torch.tensor(time_mean_weights([0., .2, .7, 1.]), dtype=torch.float32)[None]
    native_value = (native.forward(state) * weights[..., None]).sum()
    native_gradient = torch.autograd.grad(native_value, values)[0]
    actual_values = values.detach().double().requires_grad_()
    actual = motion_cost(tuple(actual_values), bounds.double(), configuration, native_dt, weights.double())
    gradient = torch.autograd.grad(actual, actual_values)[0]
    torch.testing.assert_close(actual, native_value.detach().double(), rtol=3e-6, atol=1e-7)
    torch.testing.assert_close(gradient, native_gradient.double(), rtol=3e-6, atol=3e-7)
# END task acceptance is checked on the saved curve after the library terminates.
