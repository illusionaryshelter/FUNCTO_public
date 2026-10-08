"""CPU regressions for spatially bounded unknown-space permission.

The fixtures separate observed cells, authorized unknown volume and occupied
geometry. No table, task identity or sensor-specific assumption creates free
space; planning and execution must apply the same finite permission.
"""
# BEGIN regressions for combined observed and conditionally permitted coverage.
import numpy as np
import pytest
import torch

from manipulation.online_geometry import FrozenESDF, OnlinePlanningScene
from manipulation.planning_scene import ObservedSpace
from manipulation.resource_admission import ResourceLimitError
from observed_geometry_fixtures import field_from_samples


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
    # BEGIN source material and free-ray observations remain independent fixtures.
    background = field_from_samples(np.full(_SHAPE, -.01 if occupied else 1., np.float32),
        origin=_ORIGIN, voxel_size=_STEP, observed=valid,
        owner_ids=('background',), source='fixed measured test geometry')
    # END no proposal sign establishes observed occupancy.
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
    # BEGIN the derived numerical domain can exceed the observation domain.
    # It owns no duplicate observation evidence, but all its distance samples
    # must be charged even when the permission extends outside the old map.
    field = scene._permitted_coverage
    assert not isinstance(field, ObservedSpace)
    assert not any(hasattr(field, name) for name in (
        '_known', '_known_free', 'observation_ids', 'reference_frame', 'occupancy_evidence'))
    cells = field._distance_squared.numel()
    # END source provenance remains exclusively on the original coverage.
    assert current['cpu_shared'] >= baseline['cpu_shared'] + 12 * cells
    assert current['gpu_shared'] >= baseline['gpu_shared'] + 12 * cells
    assert cloned.estimated_workspace_allocation(1, 1) == current


def test_remaining_forbidden_unknown_has_finite_nonzero_position_gradient():
    known = _half_observed()
    known[30, 30, 23] = False  # x=-0.07 remains a forbidden unknown cell.
    scene = _scene(known, [_region()])
    # BEGIN cross the actual unknown-cell face, not just the former loose bound.
    spheres = _sphere(x=-.02, radius=.05).requires_grad_()
    # END tightening the certificate must retain a gradient for true overlap.
    hard, conditional = scene._unknown(spheres)
    assert hard > 0 and conditional > hard
    hard.backward()
    assert torch.isfinite(spheres.grad).all()
    assert spheres.grad[0, :3].abs().max() > 0


# BEGIN mixed coverage spans the old map; analytic boxes retain thin permissions.
@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('direction', (-1, 1))
def test_observation_and_permission_jointly_cover_sphere_across_old_map_face(axis, direction):
    known = np.ones(_SHAPE, dtype=bool)
    lower, upper = np.full(3, -.1), np.full(3, .1)
    lower[axis], upper[axis] = ((.285, .5) if direction > 0 else (-.5, -.285))
    region = {'lower_m': lower.tolist(), 'upper_m': upper.tolist(),
              'source': 'explicit permission crossing one observation boundary'}
    scene = _scene(known, [region])
    spheres = torch.tensor([[0., 0., 0., .04]], dtype=torch.float64)
    spheres[0, axis] = direction * .29
    # The sphere crosses the old centre domain at +/-0.3, and its inward
    # part lies outside this box. Neither whole-map nor whole-box containment
    # can pass it; only their actual spatial union supplies the certificate.
    xyz, radius = spheres[0, :3].numpy(), float(spheres[0, 3])
    assert not ((xyz-radius >= lower) & (xyz+radius <= upper)).all()
    original = scene.coverage.violation(spheres).amax()
    assert original > 0
    hard, conditional = scene._unknown(spheres)
    assert hard == 0 and conditional == original
    assert np.array_equal(scene.coverage._known, known)
    # One actual cross-boundary query also checks execution reconstruction;
    # the remaining face orientations exercise the same geometric rule.
    if axis == 0 and direction == 1:
        restored = OnlinePlanningScene.from_binding(scene.to_execution_binding())
        assert restored.query_state(spheres.numpy(), np.eye(4), .5) == scene.query_state(
            spheres.numpy(), np.eye(4), .5)


def test_subvoxel_permission_has_continuous_sphere_threshold_and_finite_gradient():
    # All x coordinates and the radius below are exactly representable. The
    # permission is thinner than one voxel and is outside observed space, so
    # a full-cell mask cannot explain acceptance or a correcting derivative.
    left, width, radius = .375, 1/256, 1/1024
    region = {'lower_m': [left, -.1, -.1], 'upper_m': [left+width, .1, .1],
              'source': 'explicit subvoxel-width permission'}
    scene = _scene(np.zeros(_SHAPE, dtype=bool), [region])
    step = 1e-7
    values = []
    for offset in (-step, 0., step):
        spheres = _sphere(x=left+radius+offset, radius=radius).requires_grad_()
        hard, conditional = scene._unknown(spheres)
        values.append(float(hard.detach()))
        assert conditional > 0
        hard.backward()
        assert torch.isfinite(spheres.grad).all()
        if offset < 0:
            # Moving inward reduces a real sphere protrusion. The former
            # whole-ball Boolean branch instead left this near-feasible query
            # with the constant all-unknown residual and no position gradient.
            assert hard > 0 and spheres.grad[0, 0] < -.99
    assert values[2] == 0
    # Only floating-point guard error is allowed at exact tangency, not a
    # physical collision tolerance or a voxel-sized jump in the residual.
    assert values[1] <= 1e-12
    assert abs(values[0]-step) <= 1e-12
    for outside in (left-step, left+width+step):
        hard, _ = scene._unknown(_sphere(x=outside, radius=radius))
        assert hard > 0


def test_large_permission_domain_is_refused_before_dense_allocation(monkeypatch):
    from manipulation import online_geometry

    scene = _scene(_half_observed(), [])
    scene.unknown_policy['conditional_regions'] = [{
        'lower_m': [0., 0., 0.], 'upper_m': [1e12, 1e12, 1e12],
        'source': 'deliberately unaffordable finite permission'}]
    admissions = []

    def reject(new_bytes, *, pending_existing_bytes, max_rss_bytes):
        # The domain product must remain a Python integer; overflowing int64
        # must not turn this enormous request into a small admitted estimate.
        assert type(new_bytes) is int and new_bytes > 2**63
        admissions.append(new_bytes)
        raise ResourceLimitError('test_preparation', new_bytes, max_rss_bytes)

    def forbid_allocation(*args, **kwargs):
        raise AssertionError('dense policy allocation preceded memory admission')

    monkeypatch.setattr(online_geometry, 'require_host_memory', reject)
    # Coverage and its native arrays were constructed before patching NumPy.
    # Only the attempted policy rebuild is under observation; no huge array
    # is allowed even if a regression moves allocation before its guard.
    monkeypatch.setattr(online_geometry.np, 'zeros', forbid_allocation)
    monkeypatch.setattr(online_geometry.np, 'arange', forbid_allocation)
    with pytest.raises(ResourceLimitError, match='test_preparation'):
        scene._prepare_unknown_coverage()
    assert len(admissions) == 1
# END policy changes supply a shared continuous certificate, never free evidence.
# END finite permissions change admissibility without manufacturing observations.
