"""Check public report and binding isolation across native request lifetimes.

Only numerical candidate evaluation is substituted. Production selection,
qualification, binding, publication and repeated request handling execute.
"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from manipulation import planning
from manipulation.execution_binding import check_result, plain
from test_motion_backend_contract import Native, Scene, problem

# BEGIN completed CPU evidence has the production allocation/replacement lifecycle.
class OwnedScene(Scene):
    def to_execution_binding(self):
        return {'source': 'synthetic CPU ownership fixture', 'entries': [[1., 2.]]}


class OwnedReports(Native):
    """Represent the actual report allocation/replacement ownership contract."""
    select_candidate = planning._ConstrainedPlanner.select_candidate
    export = planning._ConstrainedPlanner.export

    def __init__(self, outcomes, *, collision=False, observation=False):
        super().__init__(outcomes, collision=collision)
        self.serial = 0
        self.scene = OwnedScene()
        self.observation = observation
        if observation:
            self.scene.unknown_policy = {'outside_regions': 'require_observation', 'conditional_regions': []}

    def evaluate_reference(self, seed):
        self.calls.append('reference')
        self.pending_acceptance = next(self.outcomes)
        return [SimpleNamespace(controls=np.zeros((2, 1)), report={'success': True, 'message': 'synthetic'})]

    def optimize(self, seed, *, iterations=None):
        self.calls.append('solve')
        self.pending_acceptance = next(self.outcomes)
        return [SimpleNamespace(controls=np.zeros((2, 1)), report={'success': True, 'message': 'synthetic'})]

    def _controls_curve(self, controls):
        return deepcopy(self.curve)

    def _candidate_objective(self, curve, scene):
        return {'fixed_objective': 1., 'objective_components': {'motion_quality': 1.}}

    def evaluate_candidate(self, trajectory, *, record_work, objective):
        record_work(passes=1, points=2)
        self.serial += 1
        geometry = {'proved_intervals': [{'start_s': 0., 'end_s': 1.,
            'pairs': [{'endpoint_lower_m': [.2, .3], 'relative_speed_m_s': .01,
                       'geometry_entity_key': {'first': 'robot/a', 'second': 'background'}}]}],
            'confirmed_intersections': [], 'unresolved_intervals': [], 'serial': self.serial}
        accepted = self.pending_acceptance
        report = {'accepted': accepted, 'residuals': {'task': 0. if accepted else 1.},
            'collision_residuals': {'scene': float(self.collision and not accepted)}, 'coverage_residuals': {},
            'scene_update_residuals': {}, 'conditional_unknown_residuals': {},
            'continuous_geometry': geometry, **objective}
        if self.observation and not accepted:
            report['residuals'] = {'scene': 1.}
            report['scene_components'] = {'unknown/forbidden/background': {'maximum': 1.}}
            report['coverage_residuals'] = {'scene': 1.}
        self.last_validation = ({'u': np.array([0., 1.]), 'violations': np.zeros(2),
                                 'continuous_geometry': geometry} if accepted else None)
        self.delivery_candidate = trajectory
        return trajectory, report, 2

    def rebind(self, source, scene, budget, **kwargs):
        self.calls.append('rebind')
        self.last_validation = None
        self.scene = scene
        return self


@pytest.mark.parametrize('outcomes,second_request,collision,observation', [
    ([True, True], True, False, False),
    ([False, True], False, False, False),
    ([False, True], False, True, False),
    ([False, False], False, False, False),
    ([False], False, False, True),
])
def test_completed_reports_preserve_public_and_cross_request_isolation(
        outcomes, second_request, collision, observation):
    native = OwnedReports(outcomes, collision=collision, observation=observation)
    backend = planning.MotionBackend({}, robot_validator=object())
    with patch.object(planning, '_ConstrainedPlanner', return_value=native):
        first = backend.plan(problem(), native.scene)
        reference_before = deepcopy(first.diagnostics['reference_candidate'])
        formal_before = plain(check_result(first)) if first.success else None
        residuals_before = deepcopy(first.diagnostics['residuals'])
        first.diagnostics['reference_candidate']['residuals']['ownership_sentinel'] = 42.
        assert first.diagnostics['residuals'] == residuals_before, 'reference/root public diagnostic alias'
        del first.diagnostics['reference_candidate']['residuals']['ownership_sentinel']
        if second_request:
            second = backend.plan(problem(), OwnedScene())
            assert second.success
            assert first.diagnostics['reference_candidate'] == reference_before
            assert plain(check_result(first)) == formal_before
        if first.success:
            root_before = deepcopy(first.diagnostics['continuous_geometry'])
            returned_before = deepcopy(first.diagnostics['returned_candidate'])
            first.diagnostics['reference_candidate']['continuous_geometry']['serial'] = -1
            assert first.diagnostics['continuous_geometry'] == root_before
            assert first.diagnostics['returned_candidate'] == returned_before
            assert plain(check_result(first)) == formal_before
            first.diagnostics['reference_candidate'] = reference_before
    backend.close()

# END mutating a public analysis view cannot alter another view or formal binding.
