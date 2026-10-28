"""CPU request-lifecycle tests with explicit candidate evidence and costs.

The real ledger, request checks, original selector and first qualified result
publication execute. A deterministic planner supplies candidate qualification
and costs; a local binding fixture copies their evidence. No test constructs a
native solver or establishes physical/observed-scene feasibility.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "tests"), str(PROJECT / "tests/planning")]


from manipulation.preference_planner import lifecycle
from manipulation.preference_planner.lifecycle import plan_trajectory
from manipulation.execution_binding import curve_snapshot
from manipulation.planning import MotionPlanningError, PlanningBudget, PlanningWorkExhausted
from manipulation.planning_problem import PoseEventRequirement
from manipulation.unified_solver import ConstrainedSolution
from manipulation.preference_planner.pose_roles import bind_pose_objectives
from recorded_task_planning import _diagnostic_value, _json_value
from test_motion_backend_contract import Native, Scene, problem


class CandidatePlanner(Native):
    """Reuse the existing CPU request fixture and supply a finite result stream."""

    # BEGIN explicit rejected candidate evidence exercises the existing repair route.
    def __init__(self, *, reference=(4., True), proposals=(), interruption=None,
                 collision_candidates=()):
        super().__init__(())
        self.rows = {index: row for index, row in enumerate((reference, *proposals))}
        self.proposal_ids = list(range(1, len(self.rows)))
        self.interruption = interruption
        self.collision_candidates = set(collision_candidates)
        self.seed_strategies = self.initial_seed_strategies = [
            f'analytical_source_{index}' for index in range(max(1, len(proposals)))]
        self.scored, self.validated = [], []
    # END the fixture never constructs scene geometry or native solver state.

    def diagnostic_snapshot(self):
        """Expose fixture evidence through the production result interface."""
        return {"joint_problem": {"candidates": list(self.validated)}, "objective_trials": []}

    def evaluate_reference(self, seed):
        self.calls.append('reference')
        return [self.result(0)]

    def result(self, number):
        # The internal solver number deliberately differs from the declared
        # common objective; result publication preserves the checked report.
        return ConstrainedSolution(np.array([[float(number)]]),
                                    {'success': True, 'scaled_objective': -1000.*number})

    def optimize(self, seed, *, iterations=None):
        self.calls.append('solve')
        for number in self.proposal_ids:
            yield self.result(number)
        if self.interruption is not None:
            self.diagnostic_optimizer_candidate = self._controls_curve(np.array([[99.]]))
            self.diagnostic_optimizer_provenance = {'source': 'analytical_interrupted_trial',
                                                   'validation_status': 'not_completed'}
            raise self.interruption

    def _controls_curve(self, controls):
        curve = deepcopy(self.curve)
        curve.test_number = int(controls[0, 0])
        return curve

    def _candidate_objective(self, curve, scene):
        self.scored.append(curve.test_number)
        value = self.rows[curve.test_number][0]
        return {'fixed_objective': value, 'objective_components': {'pose_prior': value}}

    # BEGIN supplied and generated curves share explicit qualification evidence.
    def bind_reference_candidate(self, curve):
        self.calls.append('bind_reference')
        return deepcopy(curve), self.scene, self.progress_map

    def evaluate_candidate(self, curve, *, record_work, objective=None, candidate_scene=None):
        self.validated.append(curve.test_number)
        record_work(passes=1, points=3)
        accepted = self.rows[curve.test_number][1]
        if objective is None:
            objective = self._candidate_objective(curve, candidate_scene)
        report = {**objective, 'accepted': accepted, 'native_success': True,
            'residuals': {'required': 0. if accepted else 1.},
            'collision_residuals': {'scene_collision': float(curve.test_number in self.collision_candidates)},
            'coverage_residuals': {},
            'scene_update_residuals': {}, 'conditional_unknown_residuals': {}}
        self.last_validation = ({'candidate_number': curve.test_number,
                                 'owned_values': np.array([curve.test_number], float)} if accepted else None)
        self.delivery_candidate = curve
        return curve, report, 3
    # END each report belongs to the curve passed to the original selector.

    def select_candidate(self, results, *, record_work, on_candidate):
        # Both role branches use the original first-accepted selector. A
        # completed local solve may stop before later declared seed sources.
        from manipulation.planning import _ConstrainedPlanner
        return _ConstrainedPlanner.select_candidate(self, results,
            record_work=record_work, on_candidate=on_candidate)

    def export(self, result, report):
        self.calls.append('export')
        return self.delivery_candidate, 0


class SuppliedBackend:
    """Explicit CPU owner using the existing bind_request backend protocol."""

    def __init__(self, planner):
        self.planner, self.claimed, self.released = planner, False, False

    def _claim(self):
        self.claimed = True

    def _release(self):
        self.released = True

    def bind_request(self, problem, scene, *, budget, **kwargs):
        self.planner.problem, self.planner.scene, self.planner.budget = problem, scene, budget
        return self.planner


# BEGIN requests retain explicit role data across initial, retry and continuation calls.
def request_problem(role='prior'):
    request = problem()
    request.pose_events = (PoseEventRequirement(.5, np.eye(4), np.eye(4), .002, .03,
                                               'analytical event'),)
    return bind_pose_objectives(request, [dict(goal_id=['pose_events', 0], role=role,
        scales={'position': .002, 'rotation': .03}, scale_source='analytical numerical scale')])


def run(monkeypatch, planner, *, role='prior', binding_interrupt_at=None, request=None,
        budget=None, **kwargs):
    request = request_problem(role) if request is None else request

    def binding(curve, *, validation, native_report, **kwargs):
        if curve.test_number == binding_interrupt_at:
            raise PlanningWorkExhausted('budget ended during candidate binding')
        return {'candidate_id': str(curve.test_number), 'curve': curve_snapshot(curve),
            'validation': {'export': deepcopy(validation)}, 'native_report': deepcopy(native_report)}

    # Only this lifecycle's imported binding symbol is substituted.
    # The original planning entry and backend classes retain their behavior.
    monkeypatch.setattr(lifecycle, 'bind_candidate', binding)
    backend = SuppliedBackend(planner)
    result = plan_trajectory(request, Scene(), {}, robot_validator=object(),
        budget=budget, backend=backend, **kwargs)
    assert backend.claimed == backend.released
    return result
# END the real request ledger and continuation validation remain active.


# BEGIN first complete qualification is independent of role and later proposal cost.
@pytest.mark.parametrize('role', ['required', 'prior'])
@pytest.mark.parametrize('later_cost', [1., 4., 9.])
def test_complete_reference_returns_before_any_repair(monkeypatch, role, later_cost):
    planner = CandidatePlanner(proposals=((later_cost, True),),
        interruption=PlanningWorkExhausted('unrequested later work'))
    result = run(monkeypatch, planner, role=role)
    assert result.success, result.diagnostics
    assert planner.calls == ['seed', 'reference', 'export']
    assert result.trajectory.test_number == 0
    assert planner.scored == planner.validated == [0]
    assert result.diagnostics['fixed_objective'] == 4.
    work = result.request_ledger['consumed_planning']
    assert work['trajectory_solves'] == 0 and work['validation_passes'] == 1
    assert work['validation_queries'] == 4  # Initial state and one explicit 3-query candidate.
    assert result.execution_binding['request_ledger']['consumed_planning'] == work
    assert result.request_ledger['binding_records'][-1]['accepted']
    assert result.diagnostics['returned_candidate']['objective_components'] == {'pose_prior': 4.}


@pytest.mark.parametrize('role', ['required', 'prior'])
def test_complete_supplied_curve_returns_before_reference_initialization(monkeypatch, role):
    planner = CandidatePlanner(proposals=((1., True),))
    supplied = deepcopy(planner.curve)
    supplied.test_number = 0
    result = run(monkeypatch, planner, role=role, joint_reference=supplied)
    assert result.success, result.diagnostics
    assert planner.calls == ['bind_reference']
    assert planner.scored == planner.validated == [0]
    assert result.diagnostics['route'] == 'candidate_metrics'
    work = result.request_ledger['consumed_planning']
    assert work['ik_batches'] == work['trajectory_solves'] == 0
    assert result.diagnostics['reference_clock']['duration'] == supplied.duration
    assert result.execution_binding['curve'] == curve_snapshot(supplied)


def test_first_delivery_owns_curve_report_and_binding_after_backend_reuse(monkeypatch):
    planner = CandidatePlanner()
    result = run(monkeypatch, planner)
    assert result.success, result.diagnostics
    planner.last_validation['owned_values'][:] = -100.
    planner.delivery_candidate.position[:] = -100.
    result.diagnostics['reference_candidate']['objective_components']['pose_prior'] = -100.
    np.testing.assert_array_equal(result.execution_binding['validation']['export']['owned_values'], [0.])
    np.testing.assert_array_equal(result.trajectory.position, 0.)
    assert result.diagnostics['returned_candidate']['objective_components']['pose_prior'] == 4.
    assert result.execution_binding['native_report']['objective_components']['pose_prior'] == 4.
# END accepted reference paths require no prospective objective comparison.


# BEGIN rejected references retain necessary optimization and geometric recovery.
def test_smaller_cost_cannot_select_a_rejected_candidate(monkeypatch):
    planner = CandidatePlanner(reference=(4., False),
        proposals=((.01, False), (3., True), (1., True)))
    result = run(monkeypatch, planner)
    assert result.success, result.diagnostics
    assert result.trajectory.test_number == 2
    assert planner.validated == [0, 1, 2]
    assert [row['accepted'] for row in result.diagnostics['attempts']] == [False, True]
    assert result.diagnostics['fixed_objective'] == 3.
    work = result.request_ledger['consumed_planning']
    assert work['trajectory_solves'] == 1 and work['validation_passes'] == 3
    assert work['validation_queries'] == 10


def test_initial_geometry_failure_uses_original_search_before_repair(monkeypatch):
    planner = CandidatePlanner(reference=(4., False), proposals=((3., True), (1., True)),
        collision_candidates=(0,))
    result = run(monkeypatch, planner)
    assert result.success, result.diagnostics
    assert planner.calls == ['seed', 'reference', 'ompl', 'solve', 'export']
    assert planner.validated == [0, 1]
    assert result.diagnostics['recovery'] == 'ompl'
    work = result.request_ledger['consumed_planning']
    assert work['search_queries'] == work['trajectory_solves'] == 1


def test_geometry_first_discovered_after_main_solve_uses_remaining_recovery(monkeypatch):
    planner = CandidatePlanner(reference=(4., False), proposals=((3., False), (2., True)),
        collision_candidates=(1,))
    batches = iter(([planner.result(1)], [planner.result(2)]))

    def optimize(seed, *, iterations=None):
        planner.calls.append('solve')
        return next(batches)

    monkeypatch.setattr(planner, 'optimize', optimize)
    result = run(monkeypatch, planner)
    assert result.success, result.diagnostics
    assert planner.calls == ['seed', 'reference', 'solve', 'ompl', 'solve', 'export']
    assert result.trajectory.test_number == 2
    assert planner.validated == [0, 1, 2]
    work = result.request_ledger['consumed_planning']
    assert work['search_queries'] == 1 and work['trajectory_solves'] == 2
# END only completed qualified repairs finish their original result stream.


# BEGIN incomplete qualification and original interruptions never publish execution authority.
@pytest.mark.parametrize('error', [MotionPlanningError('cancelled', 'explicit cancellation'),
    MotionPlanningError('deadline_exceeded', 'external deadline'), RuntimeError('declared model API error')])
def test_other_interruptions_keep_their_original_failure_type(monkeypatch, error):
    result = run(monkeypatch, CandidatePlanner(reference=(4., False), interruption=error))
    assert not result.success and result.trajectory is None and result.execution_binding is None
    assert result.diagnostics['error_type'] == type(error).__name__
    assert result.diagnostics['reason'] == str(error)
    assert 'optimization_interruption' not in result.diagnostics
    assert result.request_ledger['consumed_planning']['trajectory_solves'] == 1


def test_exhaustion_preserves_the_interrupted_trial_and_failed_request(monkeypatch):
    planner = CandidatePlanner(reference=(4., False), proposals=((.01, False),),
        interruption=PlanningWorkExhausted('budget before any accepted curve'))
    result = run(monkeypatch, planner)
    assert not result.success and result.execution_binding is None
    assert result.diagnostics['error_type'] == 'PlanningWorkExhausted'
    assert result.diagnostics['failure'] == 'cumulative_planning_budget_exhausted'
    assert result.diagnostics['diagnostic_trajectory'] is planner.diagnostic_optimizer_candidate
    assert result.diagnostics['diagnostic_candidate_source'] == 'analytical_interrupted_trial'
    assert result.diagnostics['diagnostic_validation_status'] == 'not_completed'
    # The existing writer saves the top-level diagnostic curve separately.
    report = {name: value for name, value in result.diagnostics.items() if name != 'diagnostic_trajectory'}
    encoded = json.dumps(_diagnostic_value(report), default=_json_value, allow_nan=False)
    assert json.loads(encoded)['error_type'] == 'PlanningWorkExhausted'
    assert not result.request_ledger['binding_records'][-1]['accepted']


@pytest.mark.parametrize('binding_number', [0, 1])
def test_binding_interruption_keeps_accepted_metrics_from_authorizing_execution(monkeypatch, binding_number):
    planner = CandidatePlanner(reference=(4., binding_number == 0), proposals=((1., True),))
    result = run(monkeypatch, planner, binding_interrupt_at=binding_number)
    assert not result.success and result.trajectory is None and result.execution_binding is None
    assert result.diagnostics['error_type'] == 'PlanningWorkExhausted'
    assert 'qualified_candidate_ready' not in result.diagnostics['wallclock']['events']


def test_cancellation_at_binding_check_prevents_publication(monkeypatch):
    planner = CandidatePlanner()
    result = run(monkeypatch, planner, cancelled=lambda: 'export' in planner.calls)
    assert not result.success and result.execution_binding is None
    assert result.diagnostics['failure'] == 'cancelled'
    assert 'qualified_candidate_ready' not in result.diagnostics['wallclock']['events']
# END the final request check remains inside the qualification boundary.


# BEGIN retries and actual execution continuation keep their original ledger semantics.
def test_failed_request_retry_keeps_consumed_work_and_returns_its_first_complete_reference(monkeypatch):
    failed = run(monkeypatch, CandidatePlanner(reference=(4., False), proposals=((3., False),)))
    assert not failed.success
    saved = deepcopy(failed.request_ledger)
    planner = CandidatePlanner(proposals=((1., True),))
    result = run(monkeypatch, planner, request_ledger=failed.request_ledger)
    assert result.success, result.diagnostics
    assert failed.request_ledger == saved
    assert result.request_ledger['request_id'] == saved['request_id']
    assert result.request_ledger['binding_records'][-1]['mode'] == 'planning_retry'
    assert result.request_ledger['consumed_planning']['trajectory_solves'] == 1
    assert result.request_ledger['consumed_planning']['validation_passes'] == 3
    assert planner.calls == ['seed', 'reference', 'export']


def test_verified_execution_continuation_returns_first_qualified_remaining_curve(monkeypatch):
    from test_planning_binding import PlanningBindingTest

    request = request_problem()
    continuation = PlanningBindingTest().continuation(request)
    original = continuation['original_binding']
    # This synthetic first execution used a qualified reference, hence no solve.
    original['consumed_planning']['trajectory_solves'] = 0
    original['consumed_planning']['validation_passes'] = 1
    original['request_ledger']['consumed_planning'] = deepcopy(original['consumed_planning'])
    saved = deepcopy(continuation)
    result = run(monkeypatch, CandidatePlanner(), request=request, continuation=continuation)
    assert result.success, result.diagnostics
    assert continuation == saved
    binding = result.execution_binding
    assert binding['request_time_origin_s'] == .25
    assert binding['consumed_physics_steps'] == 350
    assert binding['remaining_physics_steps'] == 650
    assert binding['request_ledger']['binding_records'][-1]['mode'] == 'continuation'
    assert binding['request_ledger']['consumed_planning']['trajectory_solves'] == 0
    assert binding['goal_identities']['pose_events'] == [0]
# END the accepted suffix retains verified physical time and remaining goal identities.
