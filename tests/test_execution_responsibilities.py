"""Robot supervision and current task facts do not query the frozen world."""
# BEGIN focused execution responsibilities, using bounded CPU control fixtures.
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from manipulation.execution import ExecutionLimits, FixedFingerTargets, TrajectoryExecutor
from manipulation.online_geometry import OnlinePlanningScene
from manipulation.observed_mapping import FrozenObservedField
from test_recorded_simulation import (_GeometryChecker, _bind_fingers, _execute_bound, _fixture,
    _initial_feedback, _limit_values, _result)


def attachment_scene():
    return {"owners": [{"owner_id": "payload", "attachments": [{
        "condition_id": "Holding", "interval": [0., .75],
        "release_condition_id": "Release"}]}]}


@pytest.mark.parametrize("mode", ["normal", "false_dependency", "stale_holding", "no_observer"])
def test_actual_execution_never_queries_frozen_fields_or_requires_holding_proof(tmp_path, monkeypatch, mode):
    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen world geometry was consumed during physical execution")
    # BEGIN guard the currently implemented field and planning query APIs.
    monkeypatch.setattr(FrozenObservedField, "sample", forbidden)
    monkeypatch.setattr(OnlinePlanningScene, "components", forbidden)
    monkeypatch.setattr(OnlinePlanningScene, "evaluate", forbidden)
    # END physical control consumes robot feedback and independent task facts.
    simulation, curve, targets = _fixture()
    if mode != "false_dependency":
        curve.events[2].update(kind="effect", requires_evidence=None, stop=False)
        _bind_fingers(simulation, curve, targets)
    result = _result(curve)
    result.execution_binding["scene"].update(attachment_scene())
    result.execution_binding["validation"]["inputs"] = deepcopy({
        key: result.execution_binding[key] for key in ("problem", "scene")})
    initial = _initial_feedback(simulation)
    initial["scene_condition_evidence"] = {"Holding": True}
    values = _limit_values(simulation, curve)
    if mode == "no_observer":
        values["observation_policy"] = None
    def observe(feedback, curve_time):
        if mode == "stale_holding" and curve_time > 0:
            return None
        row = {"source": "Synchronous synthetic task observer", "timestamp_s": feedback["timestamp_s"],
               "scene_condition_evidence": {"Holding": True}}
        if mode == "false_dependency":
            row.update(scene_condition_evidence={"Holding": False},
                scene_owner_poses={"payload": np.eye(4).tolist()},
                effects={"Holding": {"value": "F", "source": "Explicit required effect failure"}})
        return row
    gate = _GeometryChecker(simulation.joint_names)
    executor = TrajectoryExecutor(simulation, geometry_checker=gate)
    with patch("manipulation.robosuite_execution.mujoco.mj_forward"):
        def run():
            return _execute_bound(executor, result, finger_targets=FixedFingerTargets.from_mapping(targets),
                initial_feedback=initial, execution_limits=ExecutionLimits.from_mapping(values),
                observation_callback=None if mode == "no_observer" else observe, output=tmp_path/"trial")
        # BEGIN nominal attachment use has no missing/stale Holding gate.
        report = run()
        assert report["status"] == ("effect_rejected" if mode == "false_dependency" else "curve_finished")
        if mode == "false_dependency":
            assert report["effect_result"] == "F"
            assert report["dependency_hold"]["native_ctrl_written"]
            assert not any(row["event_id"] == "hold-later" for row in report["events"])
        # END the independently declared task-effect dependency still applies.
    assert not report["world_collision_checked"]
    assert "active_contact_rule_ids" not in report
    assert gate.preflight_inputs and gate.states
# END current owner coordinates do not make a failed required Holding effect true.
