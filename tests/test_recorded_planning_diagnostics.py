"""CPU-only evidence regressions for native rejection and interrupted export."""

# BEGIN diagnostic failures never change numerical result or exception identity.
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent/"planning"))
from manipulation import planning
from recorded_task_planning import _ExportValidationRecorder, plan_and_record
from test_manipulation_planning import PlannerCandidateSelectionTests


def batch(*, end, witness, clearance):
    return {"planned_points": 8, "checked_points": end, "curve_duration_s": 2.,
        "checked_time_interval_s": [0., end/8],
        "source_world_collision": {"checked_states": 3*end, "failed_states": 1,
            "worst_state": {"residual": 1., "name": witness}},
        "source_self_collision_batch": {"checked_states": 2,
            "unmodelled_geometry_names": [], "worst_state": {"residual": 1.,
                "minimum_required_clearance_m": clearance, "name": witness}},
        "robot_coverage": {"checked_states": end, "failed_witnesses": []},
        "analytic_acceleration": {"bounds_satisfied": True}}


def test_partial_progress_owns_first_worst_witness_and_preserves_error_type(tmp_path):
    errors = []
    path = tmp_path/"export-00-validation.json"
    recorder = _ExportValidationRecorder(path, native_result="native-result-00.npz",
                                        seed_index=1, observer_errors=errors)
    first = batch(end=2, witness="first", clearance=-.01)
    recorder.observe(first)
    first["source_world_collision"]["worst_state"]["name"] = "mutated next batch"
    first["source_self_collision_batch"]["worst_state"]["name"] = "mutated next batch"
    assert recorder.document["source_world_collision"]["worst_state"]["name"] == "first"
    recorder.observe(batch(end=4, witness="equal later", clearance=-.01))
    recorder.finish(error=planning.PlanningWorkExhausted("same original deadline"))
    saved = json.loads(path.read_text())
    assert saved["status"] == "partial" and not saved["validation_complete"]
    assert saved["completed_batches"] == 2 and saved["checked_points"] == 4
    assert saved["planned_points"] == 8 and saved["checked_time_interval_s"] == [0., .5]
    assert saved["source_self_collision"]["checked_states"] == 4
    assert saved["source_self_collision"]["worst_state"]["name"] == "first"
    assert saved["source_world_collision"]["worst_state"]["name"] == "equal later"
    assert saved["exception"] == {"type": "PlanningWorkExhausted", "message": "same original deadline"}
    assert saved["budget_exhausted"] and saved["seed_index"] == 1
    assert saved["process_constraints"]["status"] == "not_archived"
    assert not errors


def test_failed_atomic_progress_write_retains_last_complete_document(tmp_path):
    errors = []
    path = tmp_path/"export-00-validation.json"
    recorder = _ExportValidationRecorder(path, native_result="native-result-00.npz",
                                        seed_index=0, observer_errors=errors)
    recorder.observe(batch(end=2, witness="first", clearance=-.01))
    prior = path.read_bytes()
    with patch.object(Path, "replace", side_effect=OSError("diagnostic disk failure")):
        recorder.observe(batch(end=4, witness="later", clearance=-.02))
        recorder.finish(error=planning.PlanningWorkExhausted("original deadline"))
    assert path.read_bytes() == prior
    assert len(errors) == 1 and "diagnostic disk failure" in errors[0]
    assert not path.with_name(path.name+".tmp").exists()


@pytest.mark.parametrize("diagnostic_write_fails", [False, True])
def test_native_metrics_are_saved_before_failed_export_without_repeating_metrics(tmp_path, diagnostic_write_fails):
    problem = SimpleNamespace(events=(), phase_count=1, anchors=np.array([0., 1.]))
    # BEGIN only native-accepted seeds reach independent export validation.
    native = PlannerCandidateSelectionTests.native_candidates(
        flags=(True, True), native_violations=(0., 0.))
    # END this fixture tests interrupted export, not the native rejection path.
    planner, _ = PlannerCandidateSelectionTests.selection_fixture((.1, .2))
    del planner.export  # Exercise the recorder's class wrapper, not a fixture mock.
    planner.trajopt = SimpleNamespace(config=SimpleNamespace(
        position_tolerance=.001, orientation_tolerance=.02))
    output = tmp_path/"run"
    exhausted = planning.PlanningWorkExhausted("original cumulative planning deadline")
    calls = []
    original_metrics = planning._CuroboPlanner.native_diagnostics

    def metrics(self, result, seed_index=None):
        calls.append(seed_index)
        return original_metrics(self, result, seed_index=seed_index)

    def export(self, result):
        if not diagnostic_write_fails:
            saved = json.loads((output/"native-result-00-metrics.json").read_text())
            assert saved["seeds"][0]["seed_index"] == 0
            assert saved["seeds"][0]["metrics"]["success"] is True
            assert saved["seeds"][0]["positive_constraint_names"]["rollout"] == []
        self._export_progress_observer(batch(end=2, witness="actual first batch", clearance=-.01))
        raise exhausted

    def run(*args, **kwargs):
        # BEGIN exercise the same keyword-only call as seed/block observation.
        planner.select_candidate(native, candidate_indices=np.array([0]), first_feasible=True,
                                 origin={"kind": "initial_seed", "iterations": [0]})
        # END a recording failure must still preserve the export exception.
        raise AssertionError("The original budget exception must propagate")

    original_write = Path.write_text
    def write(path, *args, **kwargs):
        if diagnostic_write_fails and (path.name.endswith(".tmp") or path.name == "exception.txt"):
            raise OSError("diagnostic disk failure")
        return original_write(path, *args, **kwargs)

    with patch.object(planning._CuroboPlanner, "native_diagnostics", new=metrics), \
            patch.object(planning._CuroboPlanner, "export", new=export), \
            patch.object(planning, "plan_trajectory", side_effect=run), \
            patch.object(Path, "write_text", new=write), \
            patch("torch.cuda.max_memory_allocated", return_value=0), \
            patch("torch.cuda.max_memory_reserved", return_value=0), \
            pytest.raises(planning.PlanningWorkExhausted) as caught:
        plan_and_record(problem, None, {}, output, robot_validator=object(), budget=planning.PlanningBudget(),
                        provenance={"scope": "CPU diagnostic regression"}, execution_limits={})
    assert caught.value is exhausted and calls == [0]
    assert not (output/"trajectory.npz").exists()
    if not diagnostic_write_fails:
        saved = json.loads((output/"native-result-00-metrics.json").read_text())
        assert saved["acceptance_semantics"]["position_tolerance"] == .001
        partial = json.loads((output/"export-00-validation.json").read_text())
        assert partial["checked_points"] == 2 and not partial["validation_complete"]
        assert partial["exception"]["type"] == "PlanningWorkExhausted"
        final = json.loads((output/"planning-report.json").read_text())
        assert final["error"].startswith("PlanningWorkExhausted:")
# END neither partial evidence nor an observer failure grants execution authority.


# BEGIN the real candidate observer must cross the scoped recorder unchanged.
@pytest.mark.parametrize("origin_kind", ["initial_seed", "optimization_block"])
def test_seed_and_block_observation_passes_recorder_and_reuses_checked_candidate(tmp_path, origin_kind):
    """Use real native CPU results and selection; only FK/geometry are fixture oracles.

    Seed zero is excluded by the functional screen, and seed one qualifies.
    The terminal selector sees the same controls and must reuse that verdict
    without a second export. No planning binding is manufactured by this
    recording integration test.
    """
    problem = SimpleNamespace(events=(), phase_count=1, anchors=np.array([0., 1.]))
    native = PlannerCandidateSelectionTests.native_candidates(
        flags=(False, True), native_violations=(.1, 0.))
    planner, exported = PlannerCandidateSelectionTests.selection_fixture((.1, 0.))
    fixture_export = planner.export
    del planner.export  # Include the recorder's export wrapper in this call chain.
    planner.joint_names = ("x", "y", "z")
    planner.trajopt = SimpleNamespace(horizon=9, config=SimpleNamespace(
        position_tolerance=.001, orientation_tolerance=.02))
    planner._functional_candidate_mask = Mock(return_value=np.array([False, True]))
    planner._observed_native_result = Mock(return_value=native)
    planner._optimizer_reports = Mock(return_value=deepcopy(planner.last_optimizer_diagnostics))
    planner._check_validation_work = Mock()
    planner._candidate_qualified = Mock()
    planner._observed_verdicts, planner.candidate_observations = [], []
    previous_optimizer_diagnostics = planner.last_optimizer_diagnostics
    origin = {"kind": origin_kind, "iterations": [0 if origin_kind == "initial_seed" else 4]}
    outcome = SimpleNamespace(success=False, diagnostics={"scope": "CPU recording integration"},
                              trajectory=None, request_ledger=None)
    selected_calls = []
    original_selector = planning._CuroboPlanner.select_candidate

    def select(self, result, **options):
        selection = original_selector(self, result, **options)
        selected_calls.append((options, selection))
        return selection

    def export(self, result):
        return fixture_export(result)

    def run(*args, **kwargs):
        planner._observe_actions(native.solution[0], origin, native.solution.clone(), [[], []])
        terminal = planner.select_candidate(native)
        assert terminal is selected_calls[-1][1]
        assert terminal[2]["previously_considered"]
        return outcome

    output = tmp_path/"run"
    with patch.object(planning._CuroboPlanner, "select_candidate", new=select), \
            patch.object(planning._CuroboPlanner, "export", new=export), \
            patch.object(planning, "plan_trajectory", side_effect=run), \
            patch("torch.cuda.max_memory_allocated", return_value=0), \
            patch("torch.cuda.max_memory_reserved", return_value=0):
        recorded = plan_and_record(problem, None, {}, output, robot_validator=object(),
            budget=planning.PlanningBudget(), provenance={"scope": "CPU observation integration"},
            execution_limits={})
    assert recorded is outcome
    assert planning._CuroboPlanner.select_candidate is original_selector
    assert exported == [1]
    assert len(selected_calls) == 2
    options, selected = selected_calls[0]
    np.testing.assert_array_equal(options["candidate_indices"], [1])
    assert options["first_feasible"] is True and options["origin"] is origin
    assert selected_calls[1][0] == {}
    planner._candidate_qualified.assert_called_once()
    curve, validation, attempt = planner._candidate_qualified.call_args.args
    assert curve is selected[1] and attempt is selected[2]
    assert attempt["candidate_origin"] == origin and validation["measurements"]["candidate_index"] == 1
    assert planner.last_optimizer_diagnostics is previous_optimizer_diagnostics
    assert not planner._observing_candidate
    saved = json.loads((output/"planning-report.json").read_text())
    initial_metrics = json.loads((output/"native-result-00-metrics.json").read_text())
    assert [entry["seed_index"] for entry in initial_metrics["seeds"]] == [1]
    assert len(saved["export_validations"]) == 1
    assert [entry["validation_status"] for entry in saved["candidates"]] == [
        "completed", "completed_prior_observation"]
    assert not (output/"trajectory.npz").exists()
# END diagnostic recording cannot broaden selection or independently authorize motion.
