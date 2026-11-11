"""Rejected direct candidates preserve collision and unknown-space meanings."""

import pytest

from manipulation.direct_planner.candidate_result import candidate_failure


@pytest.mark.parametrize('action, expected', [
    ('require_observation', 'needs_observation'),
    ('forbid', 'unknown_space_forbidden'),
    ('allow_attempt', 'no_feasible_solution_found'),
])
def test_only_forbidden_coverage_has_policy_outcome(action, expected):
    report = dict(residuals={'gripper/controller': 0., 'process_scene': 0.,
                            'unknown/forbidden/entity': 1.},
                  scene_components={'gripper/controller': 0., 'unknown/forbidden/entity': 1.})
    assert candidate_failure(report, {'outside_regions': action}) == expected


@pytest.mark.parametrize('extra', [
    {'process_scene': 13.8}, {'continuous_velocity': .1},
    {'process_scene': float('nan')}, {'process_scene': float('inf')},
])
def test_other_failed_or_nonfinite_condition_keeps_candidate_rejection(extra):
    report = dict(residuals={'unknown/forbidden/entity': 1., **extra},
                  scene_components={'gripper/controller': 0.})
    assert candidate_failure(report, {'outside_regions': 'require_observation'}) == 'no_feasible_solution_found'


def test_real_scalar_scene_shape_does_not_replace_collision_with_type_error():
    report = dict(residuals={'process_scene': 13.811176576791262, 'gripper/controller': 0.},
                  scene_components={'gripper/controller': 0.})
    assert candidate_failure(report, {'outside_regions': 'require_observation'}) == 'no_feasible_solution_found'
