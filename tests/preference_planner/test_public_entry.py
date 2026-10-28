"""Check independent public dispatch and owned diagnostics without GPU work.

The real request lifecycle and resident owner execute with the existing CPU
candidate fixture. Numerical optimization and geometry have separate tests.
"""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'tests')]

from manipulation import planning as original_planning
from manipulation.execution_binding import curve_snapshot
from manipulation.preference_planner import PreferenceMotionBackend, plan_trajectory
from manipulation.preference_planner import backend as preference_backend, lifecycle
from test_lifecycle import CandidatePlanner, request_problem
from test_motion_backend_contract import Scene


@pytest.mark.parametrize('resident', [False, True])
def test_public_entry_selects_preference_backend_and_preserves_first_delivery(monkeypatch, resident):
    """Both public routes use the new lifecycle and their correct owner lifetime."""
    native = CandidatePlanner(proposals=((.01, True),))
    constructed = []

    def native_factory(problem, scene, robot, budget, device, **kwargs):
        native.problem, native.scene, native.budget = problem, scene, budget
        constructed.append(kwargs)
        return native

    def forbidden_original_entry(*args, **kwargs):
        raise AssertionError('Preference entry dispatched to the original lifecycle')

    def binding(curve, *, validation, native_report, **kwargs):
        return {'candidate_id': str(curve.test_number), 'curve': curve_snapshot(curve),
                'validation': {'export': deepcopy(validation)}, 'native_report': deepcopy(native_report)}

    monkeypatch.setattr(preference_backend, 'PreferencePlanner', native_factory)
    monkeypatch.setattr(original_planning, 'plan_trajectory', forbidden_original_entry)
    monkeypatch.setattr(lifecycle, 'bind_candidate', binding)
    owner = PreferenceMotionBackend({}, robot_validator=object(), device='cpu') if resident else None
    if resident:
        result = owner.plan(request_problem(), Scene())
    else:
        result = plan_trajectory(request_problem(), Scene(), {}, robot_validator=object(), device='cpu')
    assert result.success, result.diagnostics
    assert len(constructed) == 1
    assert result.request_ledger['consumed_planning']['trajectory_solves'] == 0
    assert native.validated == [0]
    assert result.diagnostics['preference_planner']['joint_problem']['candidates'] == [0]
    assert native.calls == ['seed', 'reference', 'export'] + ([] if resident else ['close'])
    native.validated.append(99)
    assert result.diagnostics['preference_planner']['joint_problem']['candidates'] == [0]
    if owner is not None:
        assert owner._planner is native
        owner.close()
        assert native.calls[-1] == 'close'


def test_diagnostic_snapshot_preserves_failed_numeric_trials_and_owns_arrays():
    """A nonfinite diagnostic remains evidence and cannot replace the verdict."""
    native = object.__new__(preference_backend.PreferencePlanner)
    native.terminal = SimpleNamespace(report={'matrix': np.eye(2)})
    native.pose_objectives = []
    native.geometry_constraints = SimpleNamespace(describe=lambda: {'rows': []})
    native.solve_history = [{'state': 'interrupted', 'value': np.nan}]
    native.candidate_history = [{'accepted': False, 'controls': [[1.]]}]
    native.computation = SimpleNamespace(trial_history=[{'cost': np.nan}])
    record = native.diagnostic_snapshot()
    native.terminal.report['matrix'][0, 0] = 9.
    native.candidate_history[0]['controls'][0][0] = 8.
    assert record['joint_problem']['reference_termination']['matrix'][0, 0] == 1.
    assert record['joint_problem']['candidates'][0]['controls'] == [[1.]]
    assert np.isnan(record['joint_problem']['solves'][0]['value'])
    assert np.isnan(record['objective_trials'][0]['cost'])
