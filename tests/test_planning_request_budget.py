"""CPU failure accounting at the public planning entry; no native solver runs."""

# BEGIN independent failure/retry regressions for the original request budget.
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from manipulation.planning import PlanningBudget, plan_trajectory
from test_motion_backend_contract import problem, Scene, Native, run as run_native


class PlanningRequestBudgetTest(unittest.TestCase):
    """Native initialization failures must consume work without authorizing motion."""

    def setUp(self):
        # Initialization is deliberately replaced before any scene or CUDA
        # operation. The entry needs only the explicitly empty event ledger.
        self.problem = problem()
        self.scene = Scene()
        self.budget = PlanningBudget(maximum_trajectory_solves=2)
        # BEGIN native construction is mocked; this sentinel tests argument identity.
        self.robot_validator = object()
        # END these accounting tests never claim physical geometry validation.

    def test_failed_native_initializations_share_a_finite_request_budget(self):
        with patch("manipulation.planning._ConstrainedPlanner", side_effect=TypeError("declared native API failure")) as native:
            first = plan_trajectory(self.problem, self.scene, {}, robot_validator=self.robot_validator, budget=self.budget)
            saved = deepcopy(first.request_ledger)
            second = plan_trajectory(self.problem, self.scene, {}, robot_validator=self.robot_validator, request_ledger=first.request_ledger)
            third = plan_trajectory(self.problem, self.scene, {}, robot_validator=self.robot_validator, request_ledger=second.request_ledger)
        self.assertEqual(native.call_count, 2)
        # BEGIN every admitted retry carries the same caller-owned source model.
        self.assertTrue(all(call.kwargs["robot_validator"] is self.robot_validator for call in native.call_args_list))
        # END exhausted work does not replace the validator or reach construction.
        self.assertEqual(first.request_ledger, saved)
        self.assertEqual(first.diagnostics["failed_stage"], "initialization")
        self.assertEqual(first.diagnostics["error_type"], "TypeError")
        self.assertEqual(first.diagnostics["reason"], "declared native API failure")
        for index, result in enumerate((first, second, third)):
            self.assertFalse(result.success)
            self.assertIsNone(result.trajectory)
            self.assertIsNone(result.execution_binding)
            self.assertEqual(result.request_ledger["request_id"], saved["request_id"])
            work = result.request_ledger["consumed_planning"]
            self.assertEqual(work["initializations"], min(index+1, 2))
            self.assertEqual(work["trajectory_solves"], 0)
            self.assertGreater(work["planning_seconds"], 0.)
        self.assertEqual(third.diagnostics["failure"], "cumulative_planning_budget_exhausted")
        self.assertGreaterEqual(third.request_ledger["consumed_planning"]["planning_seconds"],
                                second.request_ledger["consumed_planning"]["planning_seconds"])
        # BEGIN a carried computation ledger is not actual-world continuation.
        self.assertEqual(first.request_ledger["binding_records"][-1]["mode"], "initial")
        for result in (second, third):
            self.assertEqual(result.request_ledger["binding_records"][-1]["mode"], "planning_retry")
            self.assertFalse(result.request_ledger["requires_continuation"])
        self.assertEqual(second.request_ledger["binding_records"][:-1], saved["binding_records"])
        self.assertEqual(third.request_ledger["binding_records"][:-1],
                         second.request_ledger["binding_records"])
        # END exhausted retries append their own mode without rewriting history.

    def test_rejected_budget_change_retains_prior_work_and_source_evidence(self):
        with patch("manipulation.planning._ConstrainedPlanner", side_effect=RuntimeError("initialization failed")) as native:
            first = plan_trajectory(self.problem, self.scene, {}, robot_validator=self.robot_validator, budget=self.budget)
            first.request_ledger["effects"]["already_observed"] = {
                "value": "T", "source": "earlier actual feedback", "timestamp_s": 3.}
            saved = deepcopy(first.request_ledger)
            rejected = plan_trajectory(self.problem, self.scene, {}, robot_validator=self.robot_validator, request_ledger=first.request_ledger,
                budget=replace(self.budget, ik_seeds=7))
        self.assertEqual(native.call_count, 1)
        self.assertFalse(rejected.success)
        self.assertEqual(first.request_ledger, saved)
        self.assertIsNotNone(rejected.request_ledger)
        self.assertEqual(rejected.request_ledger["request_id"], saved["request_id"])
        self.assertEqual(rejected.request_ledger["planning_budget"], saved["planning_budget"])
        self.assertEqual(rejected.request_ledger["effects"], saved["effects"])
        self.assertGreaterEqual(rejected.request_ledger["consumed_planning"]["binding_attempts"],
                                saved["consumed_planning"]["binding_attempts"]+1)
        self.assertGreaterEqual(rejected.request_ledger["consumed_planning"]["planning_seconds"],
                                saved["consumed_planning"]["planning_seconds"])

    def test_rejected_execution_record_still_accounts_against_supplied_request(self):
        with patch("manipulation.planning._ConstrainedPlanner", side_effect=RuntimeError("initialization failed")) as native:
            first = plan_trajectory(self.problem, self.scene, {}, robot_validator=self.robot_validator, budget=self.budget)
            # BEGIN a requested actual continuation keeps its distinct mode.
            continuation_attempt = plan_trajectory(self.problem, self.scene, {}, robot_validator=self.robot_validator, request_ledger=first.request_ledger,
                continuation={"failure_reason": "missing actual continuation evidence"})
            # END rejection still reports which binding path was requested.
            rejected = plan_trajectory(self.problem, self.scene, {}, robot_validator=self.robot_validator, request_ledger=first.request_ledger,
                continuation={"failure_reason": "missing actual continuation evidence"},
                rebind={"kind": "remaining", "source": "explicit Executor continuation"})
        self.assertEqual(native.call_count, 1)
        self.assertFalse(rejected.success)
        self.assertIsNotNone(rejected.request_ledger)
        self.assertEqual(rejected.request_ledger["request_id"], first.request_ledger["request_id"])
        self.assertGreater(rejected.request_ledger["consumed_planning"]["binding_attempts"],
                           first.request_ledger["consumed_planning"]["binding_attempts"])
        # BEGIN an explicit rebind keeps its own mode despite rejection.
        self.assertFalse(continuation_attempt.success)
        self.assertEqual(continuation_attempt.request_ledger["binding_records"][-1]["mode"], "continuation")
        self.assertEqual(rejected.request_ledger["binding_records"][-1]["mode"], "remaining")
        # END this reporting fix does not reclassify explicit executor decisions.

    # BEGIN physical allocation changes preserve the original request's work.
    def test_explicit_resource_change_is_recorded_and_survives_failed_initialization(self):
        original = self.budget
        changed = replace(original, max_host_rss_bytes=12*1024**3,
            max_device_used_bytes=8*1024**3, max_scene_gpu_workspace_bytes=3*1024**3)
        with patch("manipulation.planning._ConstrainedPlanner", side_effect=RuntimeError("native allocation refused")) as native:
            first = plan_trajectory(self.problem, self.scene, {}, robot_validator=self.robot_validator, budget=original)
            saved = deepcopy(first.request_ledger)
            second = plan_trajectory(self.problem, self.scene, {}, robot_validator=self.robot_validator,
                request_ledger=first.request_ledger, budget=changed)
        self.assertEqual(native.call_count, 2)
        self.assertEqual(first.request_ledger, saved)
        for index, result in enumerate((second,), start=2):
            self.assertEqual(result.diagnostics["failed_stage"], "initialization")
            ledger = result.request_ledger
            self.assertEqual(ledger["request_id"], saved["request_id"])
            self.assertEqual(ledger["consumed_planning"]["initializations"], index)
            self.assertEqual(ledger["consumed_planning"]["trajectory_solves"], 0)
            self.assertEqual(ledger["planning_budget"]["max_host_rss_bytes"], 12*1024**3)
            self.assertEqual(ledger["planning_budget"]["maximum_trajectory_solves"], 2)
            self.assertEqual(ledger["binding_records"][:len(saved["binding_records"])], saved["binding_records"])
            self.assertEqual(native.call_args_list[index-1].args[3], changed)
        changes = [row for row in second.request_ledger["binding_records"] if row["mode"] == "resource_limits_changed"]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["previous"]["max_host_rss_bytes"], 8*1024**3)
        self.assertEqual(changes[0]["current"]["max_host_rss_bytes"], 12*1024**3)
        self.assertEqual(changes[0]["consumed_planning"]["initializations"], 1)

    def test_resource_change_does_not_authorize_a_work_budget_change(self):
        with patch("manipulation.planning._ConstrainedPlanner", side_effect=RuntimeError("initialization failed")) as native:
            first = plan_trajectory(self.problem, self.scene, {}, robot_validator=self.robot_validator, budget=self.budget)
            rejected = plan_trajectory(self.problem, self.scene, {}, robot_validator=self.robot_validator,
                request_ledger=first.request_ledger,
                budget=replace(self.budget, max_host_rss_bytes=12*1024**3, max_validation_points=8192))
        self.assertEqual(native.call_count, 1)
        self.assertEqual(rejected.diagnostics["error_type"], "ContinuationBindingError")
        self.assertEqual(rejected.request_ledger["planning_budget"], first.request_ledger["planning_budget"])
        self.assertEqual(rejected.request_ledger["consumed_planning"]["binding_attempts"], 2)
        self.assertFalse(any(row["mode"] == "resource_limits_changed"
            for row in rejected.request_ledger["binding_records"]))
    # END an explicit physical quota never replenishes samples, solves or time.

    # BEGIN started validations and completed queries consume the original ledger.
    def test_last_admitted_validation_delivers_without_recharging_geometry(self):
        for solves, seeds in ((1, 1), (1, 2), (2, 1)):
            with self.subTest(solves=solves, seeds=seeds):
                native = Native([False, True])
                select = native.select_candidate
                query_counts = iter((3, 11))

                def interval_checked(accepted, *, record_work, on_candidate):
                    candidate, curve, report = select(accepted, record_work=record_work, on_candidate=lambda *_: None)
                    points = next(query_counts)
                    record_work(points=points)
                    report['interval_query_points'] = points
                    native.delivery_candidate = curve
                    on_candidate(candidate, report)
                    return candidate, curve, report

                native.select_candidate = interval_checked
                result, backend = run_native(native,
                    budget=replace(self.budget, maximum_trajectory_solves=solves, trajectory_seeds=seeds))
                try:
                    self.assertTrue(result.success, result.diagnostics.get('reason'))
                    self.assertIsNotNone(result.execution_binding)
                    self.assertEqual(native.calls.count('solve'), 1)
                    self.assertEqual(native.calls.count('export'), 1)
                    work = result.request_ledger['consumed_planning']
                    self.assertEqual(work['validation_passes'], 2)
                    self.assertEqual(work['validation_queries'], 1 + 3 + 11)
                finally:
                    backend.close()

    def test_late_recovery_validation_fits_the_derived_capacity(self):
        native = Native([False, False, True], collision=True)
        select = native.select_candidate
        completed = []

        def interval_checked(accepted, *, record_work, on_candidate):
            candidate, curve, report = select(accepted, record_work=record_work, on_candidate=lambda *_: None)
            record_work(points=11)
            completed.append(accepted)
            report['interval_query_points'] = 11
            if len(completed) == 1:
                # Geometry failure first appears after the main solve, so
                # the remaining batch receives the one search opportunity.
                report['collision_residuals'] = {'rollout:scene_collision': 0.}
            native.delivery_candidate = curve
            on_candidate(candidate, report)
            return candidate, curve, report

        native.select_candidate = interval_checked
        result, backend = run_native(native, budget=replace(self.budget, trajectory_seeds=1))
        try:
            self.assertTrue(result.success, result.diagnostics)
            self.assertEqual(result.request_ledger['consumed_planning']['validation_passes'], 3)
            self.assertEqual(result.request_ledger['consumed_planning']['validation_queries'], 1 + 33)
            self.assertEqual(completed, [False, False, True])
            self.assertEqual(native.calls, ['seed', 'reference', 'solve', 'ompl', 'solve', 'export'])
            self.assertIsNotNone(result.execution_binding)
        finally:
            backend.close()

    def test_export_reuses_qualification_without_charging_another_validation(self):
        native = Native([True, True])
        select = native.select_candidate
        completed = []

        def qualified(accepted, *, record_work, on_candidate):
            candidate, curve, report = select(accepted, record_work=record_work, on_candidate=lambda *_: None)
            record_work(points=7)
            completed.append(accepted)
            report['interval_query_points'] = 7
            native.delivery_candidate = curve
            on_candidate(candidate, report)
            return candidate, curve, report

        native.select_candidate = qualified
        native.export = Mock(wraps=native.export)
        result, backend = run_native(native, budget=replace(self.budget, trajectory_seeds=1))
        try:
            self.assertTrue(result.success)
            self.assertEqual(completed, [True])
            self.assertEqual(native.export.call_count, 1)
            self.assertEqual(result.request_ledger['consumed_planning']['validation_passes'], 1)
            self.assertEqual(result.request_ledger['consumed_planning']['validation_queries'], 1 + 7)
        finally:
            backend.close()

    def test_first_qualification_avoids_a_later_partial_validation(self):
        for reference_passed in (False, True):
            with self.subTest(reference_passed=reference_passed):
                native = Native([reference_passed, False])
                select = native.select_candidate
                clock, attempts = [0.], []

                def timed_query(accepted, *, record_work, on_candidate):
                    candidate, curve, report = select(accepted, record_work=record_work, on_candidate=lambda *_: None)
                    attempts.append(accepted)
                    if len(attempts) == 1:
                        record_work(points=7)
                        report['interval_query_points'] = 7
                        native.delivery_candidate = curve
                        on_candidate(candidate, report)
                        return candidate, curve, report
                    # The first batch of the second validation completed as
                    # its original wall-time allowance expired. Committing
                    # those five points invokes the real budget check before
                    # another batch or qualification result can be produced.
                    clock[0] = self.budget.planning_seconds + 1.
                    record_work(points=5)
                    self.fail('An exhausted validation started another query')

                native.select_candidate = timed_query
                with patch('manipulation.planning.time.monotonic', side_effect=lambda: clock[0]):
                    result, backend = run_native(native, budget=replace(self.budget, trajectory_seeds=1))
                try:
                    self.assertEqual(result.success, reference_passed)
                    work = result.request_ledger['consumed_planning']
                    if reference_passed:
                        self.assertEqual(attempts, [True])
                        self.assertEqual(work['validation_passes'], 1)
                        self.assertEqual(work['validation_queries'], 1 + 7)
                        self.assertEqual(work['trajectory_solves'], 0)
                        self.assertIsNotNone(result.execution_binding)
                        self.assertEqual(native.calls.count('export'), 1)
                    else:
                        self.assertEqual(attempts, [False, False])
                        self.assertEqual(work['validation_passes'], 2)
                        self.assertEqual(work['validation_queries'], 1 + 7 + 5)
                        self.assertIsNone(result.execution_binding)
                        self.assertEqual(result.diagnostics['failure'], 'cumulative_planning_budget_exhausted')
                        self.assertNotIn('export', native.calls)
                finally:
                    backend.close()
    # END export consumes completed evidence; partial work survives an interrupted query.



if __name__ == "__main__":
    unittest.main()
# END a planning failure cannot reset the request or turn historical facts into motion permission.

# BEGIN derived validation bounds include both curves admitted per binding.
def test_validation_caps_follow_binding_and_optimization_batch_arithmetic():
    import pytest
    from manipulation.planning import (_check_work_budget, _PLANNING_COUNTS,
        _PLANNING_SECONDS, PlanningWorkExhausted)
    for batches in (1, 2):
        for seeds in (1, 2):
            budget = PlanningBudget(maximum_trajectory_solves=batches,
                trajectory_seeds=seeds, ik_seeds=1, max_validation_points=32, max_task_anchors=32)
            passes = batches*(seeds+2)
            queries = 32*passes+batches
            work = dict.fromkeys((*_PLANNING_COUNTS, *_PLANNING_SECONDS), 0)
            work.update(validation_passes=passes-1, validation_queries=queries)
            with patch('manipulation.planning.time.monotonic', return_value=0.):
                _check_work_budget(budget, work, started=0., initial_seconds=0., stage='validation_passes')
                work['validation_passes'] += 1
                _check_work_budget(budget, work, started=0., initial_seconds=0.)
                with pytest.raises(PlanningWorkExhausted, match='validation_passes'):
                    _check_work_budget(budget, work, started=0., initial_seconds=0., stage='validation_passes')
                work['validation_queries'] += 1
                with pytest.raises(PlanningWorkExhausted, match='validation_queries'):
                    _check_work_budget(budget, work, started=0., initial_seconds=0.)


def test_supplied_local_and_main_curves_fit_a_single_seed_single_batch_budget():
    native = Native([False, True])
    reference = deepcopy(native.curve)
    source_scene = Scene()
    native.bind_reference_candidate = lambda curve: (curve, source_scene, native.progress_map)
    base_select = native.select_candidate

    def supplied(curve, *, candidate_scene, record_work):
        _, _, report = base_select(False, record_work=record_work, on_candidate=lambda *_: None)
        record_work(points=7)
        return curve, report, 7

    def native_curve(accepted, *, record_work, on_candidate):
        result, curve, report = base_select(accepted, record_work=record_work, on_candidate=lambda *_: None)
        record_work(points=7)
        native.delivery_candidate = curve
        on_candidate(result, report)
        return result, curve, report

    native.evaluate_candidate, native.select_candidate = supplied, native_curve
    result, backend = run_native(native, joint_reference=reference,
        budget=PlanningBudget(maximum_trajectory_solves=1, trajectory_seeds=1))
    try:
        assert result.success, result.diagnostics
        work = result.request_ledger['consumed_planning']
        assert work['validation_passes'] == 3 and work['validation_queries'] == 22
        assert work['trajectory_solves'] == 1
    finally:
        backend.close()
# END public time, seed, point and optimization-batch allowances remain unchanged.
