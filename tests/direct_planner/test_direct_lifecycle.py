"""CPU checks for the experiment's initialization and delivery boundaries.

The existing FUNCTO fixture supplies deterministic candidate evidence. The
actual direct lifecycle, cumulative work ledger, selector and continuation
validation execute. This fixture counts each submitted numerical invocation
at its optimizer boundary, matching the direct backend's ownership contract;
it does not represent native convergence or observed-scene feasibility.
"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

EXPERIMENT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                 str(Path(__file__).resolve().parents[2] / 'tests'),
                 str(Path(__file__).resolve().parents[2] / 'tests/planning')]

from manipulation.direct_planner import direct_lifecycle
from manipulation.execution_binding import curve_snapshot
from manipulation.planning import MotionPlanningError, PlanningBudget, PlanningWorkExhausted


specification = importlib.util.spec_from_file_location(
    "original_direct_lifecycle_fixture", PROJECT / "tests/preference_planner/test_lifecycle.py")
fixture = importlib.util.module_from_spec(specification)
specification.loader.exec_module(fixture)


class CountingPlanner(fixture.CandidatePlanner):
    """Submit one deterministic solve per requested seed and record its source."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # BEGIN expose the actual direct planner's joint-limit input contract.
        self.joint_limits = self._verification_native.get_cost('cspace').config.joint_limits
        # END the inherited test fixture supplies the same original bounds.
        self.initial_seed_strategies = self.seed_strategies = ["measured_joint_coefficients"]
        # BEGIN the fixture declares the same affine-domain dimension contract.
        self.terminal = SimpleNamespace(matrix=np.eye(1))
        # END ordinary fixture requests each retain one free control.
        self.submitted_seed_values = []

    # BEGIN numerical initializers expose their curve without a candidate query.
    def reference_seeds(self, joint_reference=None):
        self.received_joint_reference = joint_reference
        seeds, success = super().reference_seeds(joint_reference=joint_reference)
        self.initialization_candidate = self._controls_curve(seeds[0, 0].numpy())
        return seeds, success
    # END this is the same curve-production contract as the original initializer.

    def optimize(self, seed, *, iterations=None):
        """Charge exactly where this fixture starts its numerical invocation."""
        # BEGIN the declared zero-dimensional domain produces its sole curve.
        if self.terminal.matrix.shape[1] == 0:
            self.calls.append('unique')
            yield self.result(0)
            return
        # END this mirrors the direct optimize entry, without a numerical call.
        for index, controls in enumerate(seed[0]):
            self.check_request("trajectory_solves")
            self.request_work["trajectory_solves"] += 1
            self.calls.append("solve")
            self.submitted_seed_values.append(float(controls[0, 0]))
            if self.interruption is not None:
                self.diagnostic_optimizer_candidate = self._controls_curve(np.array([[99.]]))
                self.diagnostic_optimizer_provenance = {
                    "source": "analytical_interrupted_trial", "validation_status": "not_completed"}
                raise self.interruption
            yield self.result(self.proposal_ids[index])


def run(monkeypatch, planner, *, role="prior", budget=None, request=None,
        binding_interrupt_at=None, **kwargs):
    """Use the local public signature with the original owned binding fixture."""
    request = fixture.request_problem(role) if request is None else request
    budget = PlanningBudget(search_seconds=0., trajectory_seeds=1) if budget is None else budget

    def bind(curve, *, validation, native_report, **unused):
        if curve.test_number == binding_interrupt_at:
            raise PlanningWorkExhausted("budget ended during candidate binding")
        return {"candidate_id": str(curve.test_number), "curve": curve_snapshot(curve),
                "validation": {"export": deepcopy(validation)},
                "native_report": deepcopy(native_report)}

    monkeypatch.setattr(direct_lifecycle, "bind_candidate", bind)
    backend = fixture.SuppliedBackend(planner)
    result = direct_lifecycle.plan_trajectory(request, fixture.Scene(),
        backend=backend, budget=budget, **kwargs)
    assert backend.claimed == backend.released
    return result


@pytest.mark.parametrize("role", ["prior", "required"])
@pytest.mark.parametrize("initializer_accepted", [False, True])
def test_numerical_initializer_always_reaches_solver(monkeypatch, role, initializer_accepted):
    """Both event roles retain their evidence while a direct request optimizes."""
    planner = CountingPlanner(reference=(4., initializer_accepted), proposals=((1., True),))
    result = run(monkeypatch, planner, role=role)
    assert result.success, result.diagnostics
    assert planner.calls == ["seed", "solve", "export"]
    assert planner.validated == [1]
    assert result.trajectory.test_number == 1
    assert result.diagnostics["reference_candidate"]["validation_status"] == 'not_checked'
    assert 'accepted' not in result.diagnostics["reference_candidate"]
    assert planner.scored == [1]
    assert result.diagnostics["fixed_objective"] == 1.
    work = result.request_ledger["consumed_planning"]
    assert work["trajectory_solves"] == work["validation_passes"] == 1
    assert work["validation_queries"] == 4
    assert result.execution_binding["request_ledger"]["consumed_planning"] == work


# BEGIN a zero-dimensional domain still needs a complete candidate verdict.
@pytest.mark.parametrize('accepted', [False, True])
def test_singleton_curve_has_complete_verdict_without_numerical_call(monkeypatch, accepted):
    planner = CountingPlanner(reference=(2., accepted))
    planner.terminal = SimpleNamespace(matrix=np.empty((1, 0)))
    result = run(monkeypatch, planner)
    assert result.success is accepted, result.diagnostics
    assert planner.calls == ['seed', 'unique', *(['export'] if accepted else [])]
    assert planner.validated == [0]
    work = result.request_ledger['consumed_planning']
    assert work['trajectory_solves'] == work['ik_batches'] == 0
    assert work['validation_passes'] == 1
    assert result.diagnostics['attempts'][0]['accepted'] is accepted
# END the unique curve's actual acceptance or refusal remains explicit.


# BEGIN the current direct lifecycle consumes one numerically ranked seed.
@pytest.mark.parametrize("count", [1, 2])
def test_search_allowance_does_not_add_an_unused_solver_branch(monkeypatch, count):
    planner = CountingPlanner(reference=(4., True), proposals=((1., True),))
    result = run(monkeypatch, planner,
        budget=PlanningBudget(search_seconds=1., trajectory_seeds=count))
    assert result.success, result.diagnostics
    assert planner.calls == ['seed', 'solve', 'export']
    assert planner.submitted_seed_values == [0.]
    assert result.diagnostics['recovery'] is None and not result.diagnostics['used_ompl']
    work = result.request_ledger['consumed_planning']
    assert work['search_queries'] == 0 and work['trajectory_solves'] == 1
# END solver budgets do not select a different algorithm.


@pytest.mark.parametrize("role", ["prior", "required"])
def test_explicit_joint_reference_remains_initializer_for_the_same_solver(monkeypatch, role):
    """A supplied reference does not bypass numerical optimization or delivery."""
    planner = CountingPlanner(proposals=((1., True),))
    supplied = deepcopy(planner.curve)
    supplied.test_number = 0
    result = run(monkeypatch, planner, role=role, joint_reference=supplied,
        budget=PlanningBudget(search_seconds=1.))
    assert result.success, result.diagnostics
    assert planner.received_joint_reference is supplied
    assert planner.calls == ['seed', 'solve', 'export'] and planner.validated == [1]
    work = result.request_ledger['consumed_planning']
    assert work['trajectory_solves'] == work['ik_batches'] == 1 and work['search_queries'] == 0
    assert result.execution_binding['curve'] == curve_snapshot(result.trajectory)


@pytest.mark.parametrize("error", [PlanningWorkExhausted("controlled numerical interruption"),
    MotionPlanningError("cancelled", "explicit cancellation"), RuntimeError("declared model API error")])
def test_interrupted_solver_keeps_unchecked_initializer_unbound(monkeypatch, error):
    """An unqueried initializer and interrupted trial own separate provenance."""
    planner = CountingPlanner(reference=(4., True), interruption=error)
    result = run(monkeypatch, planner)
    assert not result.success and result.trajectory is None and result.execution_binding is None
    assert result.diagnostics["reference_candidate"]["validation_status"] == 'not_checked'
    assert planner.validated == [] and planner.scored == []
    assert result.diagnostics["error_type"] == type(error).__name__
    assert result.diagnostics["reason"] == str(error)
    assert result.diagnostics["diagnostic_candidate_source"] == "analytical_interrupted_trial"
    assert result.diagnostics["diagnostic_validation_status"] == "not_completed"
    assert result.diagnostics["diagnostic_trajectory"].test_number == 99
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 1


# BEGIN interruptions before the first numerical point preserve unchecked provenance.
def test_preflight_interruption_keeps_initializer_without_scene_evidence(monkeypatch):
    """A solve admission failure neither queries nor relabels the initializer."""
    class BeforeSubmission(CountingPlanner):
        def optimize(self, seed, *, iterations=None):
            raise PlanningWorkExhausted('request ended before numerical submission')

    planner = BeforeSubmission(reference=(4., True))
    result = run(monkeypatch, planner)
    assert not result.success and result.execution_binding is None
    assert planner.calls == ['seed'] and planner.validated == planner.scored == []
    assert result.diagnostics['diagnostic_candidate_source'] == 'reference_initializer'
    assert result.diagnostics['diagnostic_validation_status'] == 'not_checked'
    assert result.diagnostics['diagnostic_trajectory'].test_number == 0
    work = result.request_ledger['consumed_planning']
    assert work['trajectory_solves'] == work['validation_passes'] == 0
    assert work['validation_queries'] == 1


# END unchecked initializer interruptions retain their original provenance.


def test_binding_interruption_keeps_candidate_metrics_without_execution_authority(monkeypatch):
    """The final request boundary still precedes publication of the owned curve."""
    # BEGIN reproduce the real generator's suspended solver-return snapshot.
    class ReturnedCurvePlanner(CountingPlanner):
        def optimize(self, seed, *, iterations=None):
            for candidate in super().optimize(seed, iterations=iterations):
                self.diagnostic_optimizer_candidate = self._controls_curve(np.array([[1.]]))
                self.diagnostic_optimizer_provenance = {
                    'source': 'solver_return_candidate', 'validation_status': 'not_completed'}
                yield candidate

    planner = ReturnedCurvePlanner(proposals=((1., True),))
    # END the selector invokes binding before that generator resumes.
    result = run(monkeypatch, planner, binding_interrupt_at=1)
    assert not result.success and result.execution_binding is None
    assert result.diagnostics["attempts"][0]["accepted"]
    assert "qualified_candidate_ready" not in result.diagnostics["wallclock"]["events"]
    assert result.diagnostics["error_type"] == "PlanningWorkExhausted"
    # BEGIN completed qualification keeps its exact curve and verdict.
    assert result.diagnostics['diagnostic_candidate_source'] == 'validated_optimizer_candidate'
    assert result.diagnostics['diagnostic_validation_status'] == 'completed'
    assert result.diagnostics['diagnostic_candidate_report']['accepted']
    assert result.diagnostics['diagnostic_trajectory'].test_number == 1
    # END execution authority still requires the interrupted binding to finish.


def test_failed_retry_retains_cumulative_work_and_requires_a_new_solve(monkeypatch):
    """A retry consumes the remaining solve slot while preserving input ownership."""
    failed = run(monkeypatch, CountingPlanner(reference=(4., False), proposals=((3., False),)))
    assert not failed.success
    saved = deepcopy(failed.request_ledger)
    planner = CountingPlanner(proposals=((1., True),))
    result = run(monkeypatch, planner, request_ledger=failed.request_ledger)
    assert result.success, result.diagnostics
    assert failed.request_ledger == saved
    assert result.request_ledger["request_id"] == saved["request_id"]
    assert result.request_ledger["binding_records"][-1]["mode"] == "planning_retry"
    assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 2
    assert result.request_ledger["consumed_planning"]["validation_passes"] == 2
    assert planner.calls == ["seed", "solve", "export"]


def test_verified_continuation_keeps_actual_clock_and_remaining_work(monkeypatch):
    """The original continuation API owns feedback, task identities and physics time."""
    from test_planning_binding import PlanningBindingTest

    request = fixture.request_problem()
    continuation = PlanningBindingTest().continuation(request)
    original = continuation["original_binding"]
    original["consumed_planning"]["trajectory_solves"] = 0
    original["consumed_planning"]["validation_passes"] = 1
    original["request_ledger"]["consumed_planning"] = deepcopy(original["consumed_planning"])
    budget = PlanningBudget(**original["planning_budget"])
    saved = deepcopy(continuation)
    planner = CountingPlanner(proposals=((1., True),))
    result = run(monkeypatch, planner, request=request, budget=budget, continuation=continuation)
    assert result.success, result.diagnostics
    assert continuation == saved
    binding = result.execution_binding
    assert binding["request_time_origin_s"] == .25
    assert binding["consumed_physics_steps"] == 350 and binding["remaining_physics_steps"] == 650
    assert binding["request_ledger"]["binding_records"][-1]["mode"] == "continuation"
    assert binding["request_ledger"]["consumed_planning"]["trajectory_solves"] == 1
    assert binding["goal_identities"]["pose_events"] == [0]
    assert planner.budget.search_seconds == pytest.approx(1.3)
