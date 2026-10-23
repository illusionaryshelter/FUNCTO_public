"""CPU checks for passive result recording and exact curve archives."""
# BEGIN recording never initiates candidate selection or another validation.
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
from recorded_task_planning import save_curve, plan_and_record


# BEGIN candidate reports reuse production calculus without reclassification.
@pytest.mark.parametrize("classification", ["validated_candidate_pending_result", "rejected_diagnostic_not_executable"])
def test_curve_record_keeps_classification_and_separates_full_curve_from_barrier(tmp_path, classification):
    from manipulation.planning import JointTrajectory

    # Coordinate a is t**3 then linear, with a left-limit acceleration of six
    # at t=1. The original barrier at .5 permits only the prefix whose peak is
    # three. A record must expose both scopes without declaring a new plan.
    events = ({"event_id": "verify", "skill_id": "observed_skill", "kind": "verification",
               "time": .5, "requires_evidence": "actual synchronized evidence"},)
    curve = JointTrajectory(("b", "a"), np.array([0., 1., 2.]),
        np.array([[0., 0.], [0., 1.], [0., 4.]]),
        np.array([[0., 0.], [0., 3.], [0., 3.]]),
        np.array([0., 1.]), np.array([0., 1.]), events, (), {"unchanged": 0.})
    limits = {"joint_names": ["a", "b"], "acceleration": [[-5., -.1], [5., .1]]}
    target = tmp_path / "curve.npz"
    save_curve(target, curve, classification=classification, execution_limits={"source": "explicit test contract"},
               joint_limits=limits)
    metadata = json.loads(target.with_suffix(".json").read_text())
    assert metadata["classification"] == classification
    assert metadata["residuals"] == {"unchanged": 0.}
    np.testing.assert_array_equal(metadata["verification_joint_limits"]["acceleration"], limits["acceleration"])
    full, prefix = (metadata["cubic_derivatives"][key] for key in ("full_curve", "executable_prefix"))
    assert full["checked_interval_s"] == [0., 2.]
    assert full["acceleration"]["maximum_bound_excess"] == 1.
    assert full["continuity"]["acceleration_maximum_absolute_jump"] == [0., 6.]
    assert prefix["checked_interval_s"] == [0., .5]
    assert prefix["acceleration"]["maximum"] == [0., 3.]
    assert prefix["acceleration"]["bounds_satisfied"]
    with np.load(target, allow_pickle=False) as arrays:
        assert arrays["classification"].item() == classification
        np.testing.assert_array_equal(arrays["times"], curve.times)
        np.testing.assert_array_equal(arrays["position"], curve.position)
        np.testing.assert_array_equal(arrays["velocity"], curve.velocity)


# BEGIN archive provenance distinguishes an unchecked trial and a checked rejected seed.
@pytest.mark.parametrize("source,status", [("diagnostic_optimizer_trial", "not_completed"),
                                           ("validated_reference_seed", "completed")])
def test_passive_failure_recorder_calls_planner_once_and_keeps_first_cause(tmp_path, monkeypatch, source, status):
    from manipulation import planning
    @dataclass
    class Budget:
        maximum_trajectory_solves: int = 2
    curve = planning.JointTrajectory(("q",), np.array([0., 1.]), np.array([[0.], [1.]]),
        np.zeros((2, 1)), np.array([0., 1.]), np.array([0., 1.]), (), (), {"collision": .1})
    curve.satisfied_dependencies = ("verified prefix",)
    result = planning.PlanningResult(False, None,
        {"failure": "no_solution_within_budget", "diagnostic_trajectory": curve,
         "diagnostic_candidate_source": source, "diagnostic_validation_status": status},
        request_ledger={"spent_solves": 2})
    calls = []
    backend = object()
    def plan(*args, **kwargs):
        calls.append(kwargs)
        return result
    monkeypatch.setattr(planning, "plan_trajectory", plan)
    output = tmp_path / "run"
    returned = plan_and_record(SimpleNamespace(phase_count=1), object(), {}, output,
        robot_validator=object(), budget=Budget(), backend=backend, provenance={}, execution_limits={})
    assert returned is result
    assert len(calls) == 1 and calls[0]["backend"] is backend
    assert not (output / "trajectory.npz").exists()
    report = json.loads((output / "planning-report.json").read_text())
    assert report["diagnostics"]["failure"] == "no_solution_within_budget"
    assert report["candidates"][0]["curve_saved"]
    metadata = json.loads((output / "diagnostic-trajectory.json").read_text())
    assert metadata["execution_binding"] is None
    assert metadata["satisfied_dependencies"] == ["verified prefix"]
    assert metadata["validation_status"] == status and metadata["source"] == source
    entry, = report["candidates"]
    assert entry["source"] == source and entry["validation_status"] == status
    assert entry["classification"] == metadata["classification"] == "rejected_diagnostic_not_executable"
# END provenance changes neither original failure nor execution classification.
# END the archived diagnostic never receives formal execution authorization.

# BEGIN the real formal binding consumes and preserves the online scene record.
def test_real_binding_and_observed_field_roundtrip_keep_candidate_and_consumed_dependency(tmp_path):
    from manipulation.execution_binding import bind_candidate, check_result
    from manipulation.planning import JointTrajectory, PlanningResult
    # BEGIN use the current registered-robot/observed-field serialization contract.
    from manipulation.observed_mapping import FrozenObservedField
    from test_native_scene_contract import make_scene
    from test_motion_backend_contract import problem
    original = make_scene(tmp_path)
    curve = JointTrajectory(("q",), np.array([0., 1.]), np.zeros((2, 1)), np.zeros((2, 1)),
        np.array([0., 1.]), np.array([0., 1.]),
        ({"event_id": "observed", "skill_id": "s", "time": 0., "progress": 0., "requires_evidence": "held"},), (), {},
        satisfied_dependencies=("held",))
    binding = bind_candidate(curve, problem=problem(), scene=original,
        validation={"u": np.array([0., 1.]), "violations": np.zeros(2)}, native_report={"accepted": True})
    result = PlanningResult(True, curve, {}, binding)
    checked = check_result(result)
    assert checked["curve"]["satisfied_dependencies"] == ["held"]
    assert checked["curve"]["next_verification"] is None
    restored = FrozenObservedField.from_record(checked["scene"]["background"])
    assert checked["scene"] == json.loads(json.dumps(original.to_execution_binding()))
    np.testing.assert_array_equal(restored.values, original.background.values)
    np.testing.assert_array_equal(restored.observed, original.background.observed)
    np.testing.assert_array_equal(restored.weights, original.background.weights)
    # END archive verification consumes the same fields as the production binding.
    curve.satisfied_dependencies = ()
    with pytest.raises(ValueError, match="(?i)curve|trajectory|candidate"):
        check_result(result)
# END no fake binding or source geometry query bypasses this handoff check.
