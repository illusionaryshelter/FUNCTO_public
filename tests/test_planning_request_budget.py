"""CPU failure accounting at the public planning entry; no native solver runs."""

# BEGIN independent failure/retry regressions for the original request budget.
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from manipulation.planning import PlanningBudget, plan_trajectory


class PlanningRequestBudgetTest(unittest.TestCase):
    """Native initialization failures must consume work without authorizing motion."""

    def setUp(self):
        # Initialization is deliberately replaced before any scene or CUDA
        # operation. The entry needs only the explicitly empty event ledger.
        self.problem = SimpleNamespace(events=())
        # The admission owns a scene copy even when native initialization fails.
        self.scene = SimpleNamespace(clone_for_rollout=lambda: SimpleNamespace())
        self.budget = PlanningBudget(maximum_trajectory_solves=2)

    def test_failed_native_initializations_share_a_finite_request_budget(self):
        with patch("manipulation.planning._CuroboPlanner", side_effect=TypeError("declared native API failure")) as native:
            first = plan_trajectory(self.problem, self.scene, {}, budget=self.budget)
            saved = deepcopy(first.request_ledger)
            second = plan_trajectory(self.problem, self.scene, {}, request_ledger=first.request_ledger)
            third = plan_trajectory(self.problem, self.scene, {}, request_ledger=second.request_ledger)
        self.assertEqual(native.call_count, 2)
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
            first = plan_trajectory(self.problem, self.scene, {}, budget=self.budget)
            first.request_ledger["effects"]["already_observed"] = {
                "value": "T", "source": "earlier actual feedback", "timestamp_s": 3.}
            saved = deepcopy(first.request_ledger)
            rejected = plan_trajectory(self.problem, self.scene, {}, request_ledger=first.request_ledger,
                budget=replace(self.budget, maximum_trajectory_solves=3))
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
            first = plan_trajectory(self.problem, self.scene, {}, budget=self.budget)
            # BEGIN a requested actual continuation keeps its distinct mode.
            continuation_attempt = plan_trajectory(self.problem, self.scene, {}, request_ledger=first.request_ledger,
                continuation={"failure_reason": "missing actual continuation evidence"})
            # END rejection still reports which binding path was requested.
            rejected = plan_trajectory(self.problem, self.scene, {}, request_ledger=first.request_ledger,
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


if __name__ == "__main__":
    unittest.main()
# END a planning failure cannot reset the request or turn historical facts into motion permission.
