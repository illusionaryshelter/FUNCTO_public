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
        with patch("manipulation.planning._CuroboPlanner", side_effect=TypeError("declared native API failure")) as native:
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
        with patch("manipulation.planning._CuroboPlanner", side_effect=RuntimeError("initialization failed")) as native:
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
        with patch("manipulation.planning._CuroboPlanner", side_effect=RuntimeError("initialization failed")) as native:
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
        with patch("manipulation.planning._CuroboPlanner", side_effect=RuntimeError("native allocation refused")) as native:
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
        with patch("manipulation.planning._CuroboPlanner", side_effect=RuntimeError("initialization failed")) as native:
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

    # BEGIN reused delivery metrics cannot spend a second native-query pass.
    def test_last_native_solve_delivers_without_recharging_completed_geometry(self):
        for solves in (1, 2):
            with self.subTest(solves=solves):
                native = Native([False] * (solves-1) + [True], collision=True)
                result, backend = run_native(native,
                    budget=replace(self.budget, maximum_trajectory_solves=solves, trajectory_seeds=1))
                try:
                    self.assertTrue(result.success, result.diagnostics.get('reason'))
                    self.assertIsNotNone(result.execution_binding)
                    self.assertEqual(native.calls.count('solve'), solves)
                    self.assertEqual(native.calls.count('export'), 1)
                    work = result.request_ledger['consumed_planning']
                    self.assertEqual(work['validation_passes'], 0)
                    self.assertEqual(work['validation_queries'], 1)
                finally:
                    backend.close()

    def test_delivery_that_reports_new_queries_still_consumes_a_pass(self):
        native = Native([True])
        export = native.export

        def queried_delivery(*args):
            curve, _ = export(*args)
            return curve, 7

        native.export = queried_delivery
        result, backend = run_native(native, budget=replace(self.budget, trajectory_seeds=1))
        try:
            self.assertTrue(result.success)
            self.assertEqual(result.request_ledger['consumed_planning']['validation_passes'], 1)
            self.assertEqual(result.request_ledger['consumed_planning']['validation_queries'], 8)
        finally:
            backend.close()
    # END native checks stay bounded; polynomial consistency and wall time remain required.


if __name__ == "__main__":
    unittest.main()
# END a planning failure cannot reset the request or turn historical facts into motion permission.
