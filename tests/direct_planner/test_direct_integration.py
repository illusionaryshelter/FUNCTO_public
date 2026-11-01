"""CPU checks of the shared objective and fixed-problem IPOPT lifecycle.

The original small robot fixture supplies calibrated FK and cubic curves.
The real IPOPT quadratic checks numerical-call accounting and delivery after
completed candidate feedback. Scene algebra has independent original-model
fixtures; this lifecycle fixture does not supply collision evidence.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                 str(Path(__file__).resolve().parents[2] / 'tests'),
                 str(Path(__file__).resolve().parents[2] / 'tests/planning')]

from manipulation.direct_planner import direct_preference as direct
from manipulation.direct_planner.direct_computation import DirectComputation
from manipulation.direct_planner.differentiable_kinematics import DifferentiablePinKinematics
from manipulation.planning import JointTrajectory, PlanningWorkExhausted, _ProgressMap, _phase_aligned_parameters
from manipulation.preference_planner.pose_roles import bind_pose_objectives
from manipulation.timed_reference import BoundToolReference
from manipulation.unified_solver import TrajectoryComputation
from manipulation.direct_planner.reference_objective import ReferenceObjective
from .test_shared_solver import QuadraticComputation
from .test_support_linearization import curve_fixture
from manipulation.direct_planner.terminal_binding import bind_terminal_hold


# BEGIN explicit empty geometry block for the isolated objective fixture.
class EmptyGeometry:
    """Explicit empty row block for the isolated robot/task objective fixture."""

    row_count = 0
    row_keys = columns_by_row = ()

    def bind_solver(self, terminal):
        from manipulation.direct_planner.sparse_rows import SparseRows
        self.pattern = SparseRows((), terminal.matrix.shape[1]*terminal.offset.shape[1])
        return self

    @staticmethod
    def evaluate(full_controls, shared_context=None):
        return SimpleNamespace(values=np.empty(0), jacobian_values=lambda: np.empty(0))
# END complete observed-geometry integration uses its separate real-scene fixture.


def robot_case():
    """Bind one real robot curve and an explicit prior/reference declaration."""
    original, controls = curve_fixture.computation(extra_parameters=(.43,))
    # BEGIN this isolated objective fixture declares an empty relation inventory.
    original.scene.contacts = ()
    original.scene._initial_support_models = {}
    # END material/contact integration uses a separate real observed-scene fixture.
    problem = bind_pose_objectives(original.problem, [dict(goal_id=['pose_events', 0],
        role='prior', scales=dict(position=.02, rotation=.1, yaw=.1),
        scale_source='analytical test declaration')])
    kinematics = DifferentiablePinKinematics(original.kinematics.model,
        frames=original.kinematics.frames, T_reference_base=problem.base_pose)
    base = TrajectoryComputation(problem=problem, scene=original.scene, kinematics=kinematics,
        domain=original.domain, joint_names=original.joint_names, parameters=original.parameters,
        progress=original.progress.numpy()[0], ordinary_indices=original.ordinary_indices,
        motion_configuration=original.motion_configuration, native_dt=original.native_dt,
        constraint_weight=original.constraint_weight, max_host_rss_bytes=original.max_host_rss_bytes)
    terminal = bind_terminal_hold(base.domain._curve_map, [0., .8, 1.], [0., 1., 1.],
        source='declared analytical terminal interval')
    robot = base.robot_states(*base.states(torch.tensor(controls))[:2])
    initial = np.eye(4)
    initial[:3, 3] = robot[4][0][0, 0].numpy()
    initial[:3, :3] = Rotation.from_quat(robot[4][1][0, 0].numpy()[[1, 2, 3, 0]]).as_matrix()
    twist = torch.cat(robot[5], -1)[0, 0].numpy()
    reference = BoundToolReference(problem, [0., 1.], [0., 1.], base.duration, initial, twist)
    declaration = ReferenceObjective(.03, .2, .4, 0., 1.)
    mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    geometry = EmptyGeometry()
    computation = DirectComputation(base=base, terminal=terminal, geometry=geometry,
        reference_objective=declaration.bind(reference, base.times[0, base.ordinary_indices]))
    planner = direct.DirectPreferencePlanner.__new__(direct.DirectPreferencePlanner)
    planner.__dict__.update(computation=computation, terminal=terminal, geometry_constraints=geometry,
        problem=problem, scene=base.scene, pin_kinematics=kinematics, joint_names=base.joint_names,
        torch=torch, bound_reference=reference, reference_objective_declaration=declaration,
        progress_map=mapping,
        formal_clock=dict(duration=base.duration, native_dt=base.native_dt),
        budget=SimpleNamespace(constraint_weight=base.constraint_weight),
        motion_configuration=base.motion_configuration)
    reduced = terminal.reduce(controls + np.random.default_rng(8).normal(0., .02, controls.shape))
    return planner, reduced


def exported_curve(planner, reduced, *, time_scale=1., progress_u=None, progress_s=None):
    """Export the same original polynomial, optionally under another clock."""
    curve = planner.computation.full_domain._curve_map
    times, position, velocity = curve.hermite_nodes(planner.terminal.expand(reduced),
        curve.start_q, curve.start_dq)
    return JointTrajectory(planner.joint_names, times*time_scale, position, velocity/time_scale,
        planner.progress_map.u if progress_u is None else np.asarray(progress_u),
        planner.progress_map.s if progress_s is None else np.asarray(progress_s), (), {}, {})


def test_real_robot_solver_and_exported_curve_share_every_objective_component():
    planner, reduced = robot_case()
    controls = torch.tensor(reduced, dtype=torch.float64, requires_grad=True)
    expected = planner.computation.objective_components(controls)
    actual = planner._candidate_objective(exported_curve(planner, reduced), planner.scene)
    assert set(actual['objective_components']) == set(expected)
    for name, value in expected.items():
        assert actual['objective_components'][name] == pytest.approx(float(value.detach()), rel=2e-12, abs=1e-9)
    gradient, = torch.autograd.grad(sum(expected.values()), controls)
    assert torch.isfinite(gradient).all()
    linearized, _ = planner.computation.task_linearization(controls)
    complete = planner.computation.task(controls)
    for got, wanted in zip(linearized, complete):
        torch.testing.assert_close(got, wanted, rtol=0., atol=0.)


def test_exported_different_clock_uses_its_own_physical_reference_samples():
    planner, reduced = robot_case()
    trajectory = exported_curve(planner, reduced, time_scale=1.7,
        progress_u=[0., .3, .65, 1.], progress_s=[0., .45, .45, 1.])
    actual = planner._candidate_objective(trajectory, planner.scene)
    mapping = _ProgressMap(trajectory.progress_u, trajectory.progress_s)
    times = _phase_aligned_parameters(planner.progress_map, mapping,
        planner.computation.parameters)*trajectory.duration
    indices = planner.computation.ordinary_indices
    expected_reference = planner.reference_objective_declaration.bind_clock(planner.bound_reference,
        problem=planner.problem, progress_u=mapping.u, progress_s=mapping.s,
        duration=trajectory.duration, physical_times=times[indices])
    _, robot = planner._candidate_robot_state(trajectory, planner.scene, times,
        planner.computation.progress.numpy()[0])
    expected = planner.budget.constraint_weight * expected_reference.cost(
        robot[4][0][:, indices], robot[4][1][:, indices])
    assert actual['objective_components']['reference_prior'] == pytest.approx(float(expected))
    assert actual['reference_objective']['physical_times_s'] == times[indices].tolist()
    assert actual['reference_objective']['effective_absolute_weight'] == 4.


class IdentityTerminal:
    report = {}

    @staticmethod
    def reduce(value):
        return np.asarray(value).copy()

    expand = reduce


def numeric_planner(monkeypatch, *, maximum=2):
    """Use the real adapter with an explicitly separate qualification fixture."""
    planner = direct.DirectPreferencePlanner.__new__(direct.DirectPreferencePlanner)
    computation = QuadraticComputation()
    computation.trial_history = []
    geometry = SimpleNamespace(row_count=0, workspace_bytes=lambda: {})
    planner.__dict__.update(torch=torch, terminal=IdentityTerminal(), computation=computation,
        geometry_constraints=geometry, budget=SimpleNamespace(trajectory_iterations=30,
            maximum_trajectory_solves=maximum), scene=None, robot_metadata=None, anchor_capacity=0,
        progress_map=SimpleNamespace(u=[0., 1.]), pose_objectives=[],
        seed_strategies=['declared_first', 'declared_second'], solve_history=[],
        request_work={'trajectory_solves': 0}, request_work_baseline={},
        last_candidate_report={'accepted': True}, remaining_request_seconds=lambda: 10.)
    planner._controls_curve = lambda full: np.asarray(full).copy()

    def check(stage=None):
        if stage == 'trajectory_solves' and planner.request_work['trajectory_solves'] >= maximum:
            raise PlanningWorkExhausted('analytical solve cap')

    planner.check_request = check
    planner.numerical_problem = computation.bind_solver(np.zeros((1, 2)), check_request=check)
    monkeypatch.setattr(direct, '_planning_workspace_bytes', lambda *args, **kwargs:
        SimpleNamespace(pending_cpu_bytes=0))
    return planner




def test_each_declared_source_uses_one_call_and_stale_acceptance_cannot_stop_it(monkeypatch):
    planner = numeric_planner(monkeypatch)
    results = list(planner.optimize(torch.zeros((1, 2, 1, 2), dtype=torch.float64)))
    assert len(results) == planner.request_work['trajectory_solves'] == 2
    assert [row['seed_index'] for row in planner.solve_history] == [0, 1]


def test_complete_rejection_at_cap_does_not_submit_another_problem(monkeypatch):
    planner = numeric_planner(monkeypatch, maximum=1)
    iterator = planner.optimize(torch.zeros((1, 2, 1, 2), dtype=torch.float64))
    next(iterator)
    planner.last_candidate_report = {'accepted': False}
    with pytest.raises(PlanningWorkExhausted):
        next(iterator)
    assert planner.request_work['trajectory_solves'] == 1
    assert len(planner.solve_history) == 2
    assert not planner.solve_history[-1]['solver_called']
    assert np.isfinite(planner.diagnostic_optimizer_candidate).all()
    assert planner.diagnostic_optimizer_provenance['validation_status'] == 'not_completed'


def test_adapter_preflight_failure_does_not_count_as_an_ipopt_submission(monkeypatch):
    planner = numeric_planner(monkeypatch)

    def reject(*args, report_out, **kwargs):
        report_out.update(state='interrupted', solver_called=False)
        raise ValueError('explicit fixed-problem declaration error')

    monkeypatch.setattr(direct, 'solve_trajectory', reject)
    with pytest.raises(ValueError, match='fixed-problem declaration'):
        list(planner.optimize(torch.zeros((1, 1, 1, 2), dtype=torch.float64)))
    assert planner.request_work['trajectory_solves'] == 0
    assert planner.solve_history[0]['solver_called'] is False


# BEGIN terminal affine singleton initialization needs no numerical proposal.
def test_terminal_singleton_constructs_exact_curve_without_reference_ik():
    """The original terminal affine offset determines every initial coefficient."""
    planner = direct.DirectPreferencePlanner.__new__(direct.DirectPreferencePlanner)
    controls = np.array([[.2, -.1], [.2, -.1]])
    planner.terminal = SimpleNamespace(matrix=np.empty((2, 0)), offset=controls, report={},
        reduce=lambda value: np.empty((0, 2)))
    planner.check_request = lambda: None
    planner.computation = SimpleNamespace(bind_solver=lambda value, check_request: SimpleNamespace(shape=value.shape))
    planner.torch = torch
    planner._controls_curve = lambda value: np.asarray(value).copy()
    seeds, counts = planner.reference_seeds()
    np.testing.assert_array_equal(seeds.numpy()[0, 0], controls)
    np.testing.assert_array_equal(planner.initialization_candidate, controls)
    assert counts == planner.reference_ik_queries == []
    assert planner.initial_seed_strategies == ['terminal_unique_curve']
    assert planner.reference_initialization['status'] == 'terminal_unique_curve'
# END exact controls retain their own complete-candidate qualification path.


def test_nonfinite_result_does_not_reuse_initializers_accepted_verdict(monkeypatch):
    """The original selector skips this result without setting a candidate verdict."""
    planner = numeric_planner(monkeypatch)
    actual_solve, calls = direct.solve_trajectory, []

    def solve(*args, **kwargs):
        calls.append(None)
        if len(calls) == 1:
            report = kwargs['report_out']
            kwargs['on_solver_start']()
            report.update(state='completed', solver_called=True, success=False)
            return direct.ConstrainedSolution(np.full((1, 2), np.nan), report)
        return actual_solve(*args, **kwargs)

    monkeypatch.setattr(direct, 'solve_trajectory', solve)
    results = list(planner.optimize(torch.zeros((1, 2, 1, 2), dtype=torch.float64)))
    assert len(calls) == len(results) == planner.request_work['trajectory_solves'] == 2
    assert not np.isfinite(results[0].controls).all()
    np.testing.assert_allclose(results[1].controls, [[1., .5]], atol=2e-5)


# BEGIN the actual candidate method must consume its retained numerical point.
def test_real_candidate_dispatch_uses_retained_scene_and_original_joint_domain(tmp_path, monkeypatch):
    from manipulation.direct_planner.batch_distance import BatchDistance
    from manipulation.direct_planner import geometry_constraints as geometry_module
    from .test_sparse_computation import observed_direct_case

    monkeypatch.setattr(geometry_module, 'BatchDistance',
        lambda geometries, *, query_sources: BatchDistance(geometries,
            query_sources=query_sources, device='cpu'))
    computation, reduced = observed_direct_case(tmp_path)
    computation.scene.check_request = lambda: None
    point = computation.bind_solver(reduced, check_request=lambda: None).evaluate(reduced)
    computation.final_point = point
    planner = direct.DirectPreferencePlanner.__new__(direct.DirectPreferencePlanner)
    domain, names = computation.full_domain, computation.joint_names
    limits = SimpleNamespace(joint_names=list(names), **{name: computation.bounds[index]
        for index, name in enumerate(('position', 'velocity', 'acceleration', 'jerk'))})
    planner.__dict__.update(computation=computation, problem=computation.problem,
        scene=computation.scene, terminal=computation.terminal, joint_names=names,
        physical_domain=domain, progress_map=direct._ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
        pose_objectives=[], candidate_history=[], candidate_evaluations=[],
        _current_full_controls=point.full_controls.copy(),
        joint_limits=limits)
    trajectory = exported_curve(planner, reduced)

    def old_scene_call(*args, **kwargs):
        raise AssertionError('Candidate entered the obsolete full scene query')

    monkeypatch.setattr(computation.scene, 'evaluate', old_scene_call)
    work = {'passes': 0, 'points': 0}

    def record_work(**values):
        for name, value in values.items():
            work[name] += value

    actual, report, count = planner.evaluate_candidate(trajectory, record_work=record_work,
        objective=dict(fixed_objective=point.objective_value, objective_components=point.objective_components))
    assert actual is trajectory
    assert work == {'passes': 1, 'points': count}
    assert count == len(computation.parameters)
    assert not report['accepted']
    assert report['residuals']['process_scene'] == pytest.approx(max(0., point.geometry.spatial_values.max()))
    assert 'continuous_position' in report['residuals']
    assert planner.candidate_evaluations[-1]['completed']
    assert planner.last_candidate_report is report
# END no parent material-exchange method can satisfy this integration check.
