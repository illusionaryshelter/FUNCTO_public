"""CPU regressions for spatially bounded unknown-space permission.

The fixtures separate observed cells, authorized unknown volume and occupied
geometry. No table, task identity or sensor-specific assumption creates free
space; planning and execution must apply the same finite permission.
"""
# BEGIN regressions for combined observed and conditionally permitted coverage.
import numpy as np
import torch

from manipulation.online_geometry import FrozenESDF, OnlinePlanningScene
from manipulation.planning_scene import ObservedSpace


_ORIGIN = np.array([-.3, -.3, -.3])
_SHAPE = (61, 61, 61)
_STEP = .01


def _region(lower_x=.005):
    return {'lower_m': [lower_x, -.25, -.25], 'upper_m': [.25, .25, .25],
            'source': 'explicit bounded test permission'}


def _half_observed():
    # ObservedSpace stores Z,Y,X: x=0 is observed and x=0.01 is unknown.
    known = np.zeros(_SHAPE, dtype=bool)
    known[:, :, :31] = True
    return known


def _scene(known, regions, *, occupied=False):
    coverage = ObservedSpace(known, origin=_ORIGIN, voxel_size=_STEP,
                             observation_ids=('measured-test-view',), reference_frame='world')
    valid = np.full(_SHAPE, occupied, dtype=bool)
    background = FrozenESDF(np.full(_SHAPE, -.01 if occupied else 1., np.float32),
        origin=_ORIGIN, voxel_size=_STEP, site_valid=valid, observed_sign=valid,
        owner_ids=('background',), source='fixed measured test geometry')
    result = OnlinePlanningScene(background=background, owners=(), coverage=coverage,
        reference_frame='world', robot_sphere_materials=('body',),
        robot_geometry_source='test sphere geometry', has_gripper=False,
        unknown_policy={'conditional_regions': regions, 'outside_regions': 'forbid'})
    result.bind_online_robot(robot_sphere_geometry=[{
        'link_name': 'link', 'sphere_indices': [0], 'centers_local': [[0., 0., 0.]],
        'radii': [.04]}], robot_self_collision_pairs=[])
    return result


def _sphere(x=-.015, radius=.04):
    return torch.tensor([[x, 0., 0., radius]], dtype=torch.float64)


def test_mixed_observed_and_permitted_unknown_volume_passes_without_free_evidence():
    known = _half_observed()
    scene = _scene(known, [_region()])
    spheres = _sphere()
    # The sphere extends left of the permission, but that part is observed.
    original_gap = scene.coverage.violation(spheres).amax()
    assert original_gap > 0
    hard, conditional = scene._unknown(spheres)
    assert hard == 0 and conditional == original_gap
    report = scene.query_state(spheres.numpy(), np.eye(4), .5)
    assert report['passed'] and report['conditional_unknown']
    assert np.array_equal(scene.coverage.known_free, known)
    assert np.array_equal(scene.coverage._known, known)


def test_unknown_voxel_outside_permission_still_rejects_mixed_sphere():
    known = _half_observed()
    known[30, 30, 26] = False  # x=-0.04: inside the sphere, outside permission.
    scene = _scene(known, [_region()])
    hard, _ = scene._unknown(_sphere())
    assert hard > 0
    assert not scene.query_state(_sphere().numpy(), np.eye(4), .5)['passed']


def test_observation_and_adjacent_permission_boxes_jointly_cover_one_sphere():
    lower = _region()
    upper = _region()
    # y=0.005 is the shared face of the y=0 and y=0.01 cells. Each
    # permission alone leaves an unknown half uncovered; their union with
    # observed x<=0 covers the ball without requiring any one box to do so.
    lower['upper_m'][1] = .005
    upper['lower_m'][1] = .005
    known, spheres = _half_observed(), _sphere()
    for region in (lower, upper):
        hard, _ = _scene(known, [region])._unknown(spheres)
        assert hard > 0
    combined = _scene(known, [lower, upper])
    hard, conditional = combined._unknown(spheres)
    assert hard == 0
    assert conditional == combined.coverage.violation(spheres).amax()


def test_permission_must_cover_entire_unknown_cell_not_only_its_centre():
    known = np.ones(_SHAPE, dtype=bool)
    known[30, 30, 30] = False
    # Cell centre x=0 is permitted, but its lower face x=-0.005 is not.
    scene = _scene(known, [_region(lower_x=-.002)])
    hard, _ = scene._unknown(_sphere(radius=.02))
    assert hard > 0


def test_complete_sphere_inside_permission_can_be_outside_observation_map():
    region = {'lower_m': [.4, -.1, -.1], 'upper_m': [.8, .1, .1],
              'source': 'explicit finite permission outside observed map'}
    scene = _scene(np.zeros(_SHAPE, dtype=bool), [region])
    hard, conditional = scene._unknown(_sphere(x=.6))
    assert hard == 0 and conditional > 0
    # Moving the sphere across the finite permission boundary remains forbidden.
    hard, _ = scene._unknown(_sphere(x=.39))
    assert hard > 0


def test_empty_permission_keeps_original_unknown_residual_and_storage():
    scene = _scene(_half_observed(), [])
    hard, conditional = scene._unknown(_sphere())
    assert hard == scene.coverage.violation(_sphere()).amax()
    assert conditional == 0
    assert scene._permitted_coverage is scene.coverage


def test_known_collision_survives_mixed_unknown_permission():
    scene = _scene(_half_observed(), [_region()], occupied=True)
    report = scene.query_state(_sphere().numpy(), np.eye(4), .5)
    assert report['components']['unknown/forbidden/robot'] == 0
    assert report['components']['collision/robot/background'] > 0
    assert not report['passed']


def test_mixed_coverage_execution_binding_reconstructs_identical_decision():
    scene = _scene(_half_observed(), [_region()])
    restored = OnlinePlanningScene.from_binding(scene.to_execution_binding())
    sphere = _sphere().numpy()
    original = scene.query_state(sphere, np.eye(4), .5)
    assert original['passed']
    assert restored.query_state(sphere, np.eye(4), .5) == original
    assert np.array_equal(restored.coverage.known_free, scene.coverage.known_free)


def test_permission_field_is_shared_by_rollouts_and_counted_in_memory_budget():
    known = _half_observed()
    scene = _scene(known, [_region()])
    cloned = scene.clone_for_rollout()
    assert cloned._permitted_coverage is scene._permitted_coverage
    assert cloned.coverage is scene.coverage
    baseline = _scene(known, []).estimated_workspace_allocation(1, 1)
    current = scene.estimated_workspace_allocation(1, 1)
    # Each additional distance grid needs at least float32 plus float64 caches
    # on both CPU and GPU; candidate clones must not multiply that reservation.
    cells = known.size
    assert current['cpu_shared'] >= baseline['cpu_shared'] + 12 * cells
    assert current['gpu_shared'] >= baseline['gpu_shared'] + 12 * cells
    assert cloned.estimated_workspace_allocation(1, 1) == current


def test_remaining_forbidden_unknown_has_finite_nonzero_position_gradient():
    known = _half_observed()
    known[30, 30, 23] = False  # x=-0.07 remains a forbidden unknown cell.
    scene = _scene(known, [_region()])
    spheres = _sphere(x=-.02).requires_grad_()
    hard, conditional = scene._unknown(spheres)
    assert hard > 0 and conditional > hard
    hard.backward()
    assert torch.isfinite(spheres.grad).all()
    assert spheres.grad[0, :3].abs().max() > 0
# END finite permissions change admissibility without manufacturing observations.
