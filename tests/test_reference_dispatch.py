"""CPU request dispatch with exact initialization curves and terminal evidence.

The request ledger, curve serialization and dispatch predicate are production
code. Native solves and full-scene verdicts are declared fixture outcomes;
these tests establish scheduling, not geometry accuracy or GPU convergence.
"""

# BEGIN terminal functional evidence belongs to the exact returned polynomial.
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.execution_binding import curve_snapshot
from manipulation.planning import _CuroboPlanner
from test_motion_backend_contract import Native, Scene, problem, run


def terminal_fixture(native, *, change=None):
    """Bind two actual exported cubic curves to explicit functional evidence."""
    native.problem, native.joint_names = problem(), ("q",)
    native.start = SimpleNamespace(position=torch.zeros(1, 1), velocity=torch.zeros(1, 1))
    native.formal_clock = {"knot_dt": .1, "duration": 1.}
    native.trajopt.action_horizon = 8
    native.finger_trajectory = None
    seeds = torch.zeros(1, 2, 8, 1)
    curves = [_CuroboPlanner._controls_curve(native, action) for action in seeds[0]]
    rows = [{"seed_index": index, "controls": action.tolist(),
        "trajectory": curve_snapshot(curve), "functional_inequalities": {"event": .2}}
        for index, (action, curve) in enumerate(zip(seeds[0], curves))]
    if change is not None:
        change(rows)
    native.reference_initialization = {"status": "completed", "refined": {"seeds": rows}}

    def reference_seeds(joint_reference=None):
        native.calls.append("seed")
        return seeds, []

    native.reference_seeds = reference_seeds
    return seeds


def test_all_same_curve_functional_failures_skip_reference_scan_and_run_main_solve():
    native = Native([True])
    seeds = terminal_fixture(native)
    evidence = deepcopy(native.reference_initialization)
    assert native.known_initialization_failures(seeds) == [
        {"seed_index": index, "functional_inequalities": {"event": .2}} for index in range(2)]
    result, backend = run(native)
    try:
        assert result.success and native.calls == ["seed", "solve", "export"]
        assert result.diagnostics["reference_candidate"]["qualification"] == "skipped_known_functional_failure"
        assert "accepted" not in result.diagnostics["reference_candidate"]
        assert result.request_ledger["consumed_planning"]["trajectory_solves"] == 1
        assert native.reference_initialization == evidence
    finally:
        backend.close()


@pytest.mark.parametrize("changed", ["mixed", "missing_task", "nonfinite_task", "other_controls",
    "other_entry", "other_velocity", "other_clock", "other_progress", "missing_curve", "missing_seed"])
def test_incomplete_or_different_terminal_evidence_retains_reference_scan(changed):
    native = Native([False, True])

    def change(rows):
        row = rows[1]
        if changed == "mixed":
            row["functional_inequalities"]["event"] = 0.
        elif changed == "missing_task":
            row.pop("functional_inequalities")
        elif changed == "nonfinite_task":
            row["functional_inequalities"]["event"] = float("inf")
        elif changed == "other_controls":
            row["controls"][3][0] = .1
        elif changed == "other_entry":
            row["trajectory"]["position"][0][0] = .1
        elif changed == "other_velocity":
            row["trajectory"]["velocity"][0][0] = .1
        elif changed == "other_clock":
            row["trajectory"]["times"][3] += .001
        elif changed == "other_progress":
            row["trajectory"]["progress_u"][1] = .9
        elif changed == "missing_curve":
            row.pop("trajectory")
        else:
            rows.pop()

    seeds = terminal_fixture(native, change=change)
    assert native.known_initialization_failures(seeds) is None
    result, backend = run(native)
    try:
        assert result.success and native.calls == ["seed", "reference", "solve", "export"]
        assert result.diagnostics["reference_candidate"]["accepted"] is False
    finally:
        backend.close()


def test_terminal_geometry_unknown_does_not_replace_the_existing_observation_policy():
    native = Native([False])
    terminal_fixture(native, change=lambda rows: [row.update(functional_inequalities={}) for row in rows])
    world = Scene()
    world.unknown_policy = {"outside_regions": "require_observation", "conditional_regions": []}
    original = native.select_candidate

    def select(accepted, *, record_work):
        candidate, curve, report = original(accepted, record_work=record_work)
        report.update(residuals={"interpolated:process_scene": .04},
            coverage_residuals={"interpolated:unknown_forbidden": .04},
            scene_components={"unknown/forbidden/robot": {"maximum": .04}})
        return candidate, curve, report

    native.select_candidate = select
    result, backend = run(native, scene=world)
    try:
        assert not result.success and result.diagnostics["failure"] == "needs_observation"
        assert native.calls == ["seed", "reference"]
        assert result.execution_binding is None
    finally:
        backend.close()


def test_supplied_joint_reference_always_retains_original_and_proposal_qualification():
    from test_reference_candidate import candidate_fixture, supplied_curve

    native = candidate_fixture(native_outcomes=(True, True))
    seeds = terminal_fixture(native)
    assert native.known_initialization_failures(seeds)
    reference = supplied_curve()
    original = reference.position.copy()
    result, backend = run(native, joint_reference=reference)
    try:
        assert result.success
        assert native.calls == ["candidate", "seed", "reference", "solve"]
        assert result.diagnostics["original_reference_candidate"]["accepted"]
        assert "qualification" not in result.diagnostics["reference_candidate"]
        np.testing.assert_array_equal(reference.position, original)
        np.testing.assert_array_equal(result.trajectory.position, original)
    finally:
        backend.close()
# END a scheduling shortcut never supplies execution qualification.
