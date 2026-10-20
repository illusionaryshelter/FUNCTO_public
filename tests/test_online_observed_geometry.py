"""CPU counterexamples for shared source geometry, fields and permissions."""

# BEGIN exact observed geometry is independent of approximate distance seeds.
from types import SimpleNamespace
from unittest.mock import Mock, patch

import coal
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from manipulation.collision_records import encode_array, MAX_ARRAY_BYTES
from manipulation.online_observed_geometry import FrozenObservedGeometry
from manipulation.resource_admission import ResourceLimitError
from manipulation.tsdf_geometry import build_tsdf_geometry, tsdf_geometry_from_samples


def source(values, *, origin=(0., 0., 0.), h=.01, observed=None):
    values = np.asarray(values, dtype=np.float32)
    record = tsdf_geometry_from_samples(values,
        np.ones(values.shape, dtype=bool) if observed is None else observed,
        origin=origin, voxel_size=h, name='observed', source='synthetic measured TSDF')['solids'][0]
    return FrozenObservedGeometry(record)


def points_source(points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return FrozenObservedGeometry({
        'kind': 'tsdf', 'name': 'isolated', 'source': 'isolated observed zero samples',
        'pose': [0., 0., 0., 1., 0., 0., 0.],
        'origin': [-.1, -.1, -.1], 'shape_xyz': [21, 21, 21], 'voxel_size': .01,
        'surface_vertices': encode_array(points),
        'surface_offsets': encode_array(np.arange(len(points)+1, dtype=np.int64)),
        'negative_vertices': encode_array(np.empty((0, 3), dtype=np.float64)),
        'negative_offsets': encode_array(np.array([0], dtype=np.int64)),
        'negative_cell_flat_indices': encode_array(np.empty(0, dtype=np.int64)),
    })


def rule(identity='left', *, center=(-.004, 0., 0.), half=(.001, .001), thickness=.001):
    return SimpleNamespace(rule_id=identity, patch_origin=np.asarray(center), patch_axes=np.eye(3),
        patch_half_size=np.asarray(half), modeling_allowance_m=thickness,
        numerical_position_tolerance_m=0.)


def native_distance(actor, center, component):
    _, _, shape, pose, _ = component
    request, result = coal.DistanceRequest(), coal.DistanceResult()
    request.enable_signed_distance = True
    coal.distance(actor, coal.Transform3s(np.eye(3), np.asarray(center, dtype=float)),
        shape, coal.Transform3s(pose[:3, :3], pose[:3, 3]), request, result)
    return result.min_distance


def sample_field(field, origin, h, points):
    # Mirror FrozenESDF's XYZ-to-Torch indexing with double-precision
    # interpolation of the actual stored float32 proposal samples.
    volume = torch.tensor(field.transpose(2, 1, 0).copy(), dtype=torch.float64)[None, None]
    indices = (torch.tensor(np.asarray(points), dtype=torch.float64)-torch.tensor(origin))/h
    grid = 2*indices/torch.tensor(np.asarray(field.shape)-1)-1
    return F.grid_sample(volume, grid.reshape(1, 1, 1, -1, 3),
        align_corners=True, mode='bilinear').numpy().reshape(-1)


def test_positive_tsdf_seed_height_is_not_confirmation_material():
    z = np.array([-.01, 0., .01], dtype=np.float32)
    geometry = source(np.broadcast_to(z, (3, 3, 3)), origin=(-.01, -.01, -.01))
    np.testing.assert_allclose(geometry.bounds, [[-.01, -.01, -.01], [.01, .01, 0.]])
    # A proposal may be conservative near the surface, but native source
    # geometry contains no 5 mm cube at a fake ESDF seed z=+7.5 mm.
    values = geometry.distance_field([-.04]*3, (9, 9, 9), .01)
    assert sample_field(values, [-.04]*3, .01, [[0., 0., .008]])[0] < .002
    sphere = coal.Sphere(.002)
    distances = [native_distance(sphere, [0., 0., .008], geometry.collision_component(index))
                 for index in geometry.component_indices(None)]
    assert min(distances) > .0059
    assert any(native_distance(sphere, [0., 0., -.003], geometry.collision_component(index)) < 0
               for index in geometry.component_indices(None))


def test_all_negative_containment_and_shared_box_do_not_depend_on_zero_surface():
    geometry = source(-np.ones((3, 2, 2)), origin=(.1, -.2, .3))
    assert geometry.component_count == 2
    before = geometry.pending_cache_bytes
    first, second = geometry.collision_component(0), geometry.collision_component(1)
    assert first[0] == second[0] == 'negative_cell'
    assert first[2] is second[2]
    assert first[4] is second[4]
    assert not geometry._cache
    assert geometry.pending_cache_bytes == 0 < before
    assert native_distance(coal.Sphere(.001), [.105, -.195, .305], first) < 0
    np.testing.assert_allclose(first[4] @ first[3][:3, :3].T+first[3][:3, 3],
                               geometry._vertices(0)[2], atol=1e-16)


def test_point_segment_and_zero_tetrahedron_keep_source_extent_and_local_frames():
    isolated = points_source([[.017, -.021, .006]])
    component = isolated.collision_component(0)
    assert isinstance(component[2], coal.Sphere) and component[2].radius == 0
    np.testing.assert_allclose(component[4], [[0., 0., 0.]])
    np.testing.assert_allclose(component[4] @ component[3][:3, :3].T+component[3][:3, 3],
                               [[.017, -.021, .006]])
    values = np.ones((2, 2, 2), dtype=np.float32)
    values[0, 0, 0] = values[1, 0, 0] = 0
    line = source(values, origin=(.13, -.07, .2))
    segments = []
    for index in line.component_indices(None):
        item = line.collision_component(index)
        np.testing.assert_allclose(item[4] @ item[3][:3, :3].T+item[3][:3, 3],
                                   line._vertices(index)[2], atol=1e-16)
        if isinstance(item[2], coal.Capsule):
            segments.append(item)
            assert item[2].radius == 0
    assert segments
    zero = source(np.zeros((2, 2, 2)))
    assert zero.component_count == 6
    assert any(native_distance(coal.Sphere(.0001), [.002, .003, .004], zero.collision_component(index)) <= 0
               for index in zero.component_indices(None))


def test_closed_aabb_index_and_cache_are_source_local_and_shared():
    geometry = points_source([[0., 0., 0.], [.02, 0., 0.]])
    assert geometry.component_indices(([0., 0., 0.], [0., 0., 0.])).tolist() == [0]
    assert geometry.component_indices(([.005]*3, [.01]*3)).tolist() == []
    first = geometry.collision_component(1)
    storage, pending = geometry.storage_nbytes, geometry.pending_cache_bytes
    assert geometry.collision_component(1) is first
    assert geometry.storage_nbytes == storage and geometry.pending_cache_bytes == pending
    assert not first[3].flags.writeable and not first[4].flags.writeable
    copied_record = geometry.record()
    copied_record['origin'][0] = 123.
    restored = FrozenObservedGeometry(geometry.record())
    assert restored.record() == geometry.record()
    np.testing.assert_array_equal(restored.bounds, geometry.bounds)


def test_permissions_filter_whole_components_before_shared_cell_rasterization():
    geometry = points_source([[-.004, 0., 0.], [.004, 0., 0.]])
    left, right = rule(), rule('right', center=(.004, 0., 0.))
    geometry.prepare_rules([left, right])
    assert geometry.component_indices(None, ['left']).tolist() == [1]
    assert geometry.component_indices(None, ['right']).tolist() == [0]
    region = geometry.rule_region('left')
    assert region['removed_components'] == 1
    np.testing.assert_array_equal(region['lower_patch'], [0., 0., 0.])
    np.testing.assert_array_equal(geometry.surface_points('left'), [[-.004, 0., 0.]])
    values = geometry.distance_field([-.04]*3, (5, 5, 5), .02, ['left'])
    assert values[2, 2, 2] < 0  # The unpermitted point shares this raster cell.
    assert geometry.distance_field([-.04]*3, (5, 5, 5), .02, ['left', 'right']) is None
    geometry.prepare_rules([right, left])  # Geometry-equivalent rule order is harmless.
    with pytest.raises(ValueError, match='cannot change'):
        geometry.prepare_rules([rule(center=(-.003, 0., 0.)), right])
    with pytest.raises(ValueError, match='prepared'):
        geometry.component_indices(None, ['missing'])


def test_partial_component_permission_never_removes_its_unpermitted_end():
    values = np.ones((2, 2, 2), dtype=np.float32)
    values[0, 0, 0] = values[1, 0, 0] = 0
    geometry = source(values, origin=(-.005, 0., 0.))
    permission = rule(center=(-.005, 0., 0.), half=(.001, .001))
    geometry.prepare_rules([permission])
    retained = geometry.component_indices(None, ['left'])
    assert retained.size
    assert any(np.ptp(geometry._vertices(index)[2][:, 0]) > .009 for index in retained)
    assert np.all(geometry.surface_points('left')[:, 0] == -.005)


@pytest.mark.parametrize('h,shape', [(.01, (21, 21, 21)), (.025, (9, 9, 9))])
def test_interpolated_proposal_is_a_lower_bound_for_isolated_source_point(h, shape):
    point = np.array([.023, .001, -.017])
    geometry = points_source([point])
    origin = np.array([-.1]*3)
    field = geometry.distance_field(origin, shape, h)
    assert field.dtype == np.float32 and not field.flags.writeable
    probes = np.random.default_rng(42).uniform(-.095, .095, (400, 3))
    sampled = sample_field(field, origin, h, probes)
    actual = np.linalg.norm(probes-point, axis=1)
    assert np.all(sampled <= actual)
    assert sample_field(field, origin, h, [point])[0] < 0


def test_subvoxel_point_on_shared_cell_boundary_is_never_lost():
    geometry = points_source([[.005, 0., 0.]])
    field = geometry.distance_field([-.02]*3, (5, 5, 5), .01)
    assert field[2, 2, 2] < 0 and field[3, 2, 2] < 0
    assert sample_field(field, [-.02]*3, .01, [[.005, 0., 0.]])[0] < 0


def test_empty_and_all_occupied_fields_do_not_use_edt_virtual_exterior():
    empty = points_source([])
    assert empty.bounds is None and empty.component_count == 0
    assert empty.distance_field([0.]*3, (2, 2, 2), .01) is None
    geometry = source(-np.ones((2, 2, 2)), h=.01)
    with patch('manipulation.online_observed_geometry.distance_transform_edt',
               side_effect=AssertionError('all-occupied grid must have an explicit branch')):
        field = geometry.distance_field([0.]*3, (2, 2, 2), .01)
    assert np.isfinite(field).all() and (field < 0).all()
    np.testing.assert_array_equal(field, np.full(field.shape, field.flat[0]))


def test_source_outside_grid_is_rejected_instead_of_silently_cropped():
    geometry = points_source([[.5, 0., 0.]])
    with pytest.raises(ValueError, match='extends beyond'):
        geometry.distance_field([-.02]*3, (5, 5, 5), .01)
    with pytest.raises(ValueError, match='bounded'):
        geometry.distance_field([0.]*3, (MAX_ARRAY_BYTES//4+1, 2, 2), .01)


def test_budget_refusal_precedes_source_decode_and_query_field_allocation():
    record = points_source([[0., 0., 0.]]).record()
    refusal = ResourceLimitError('test', 2, 1)
    with patch('manipulation.online_observed_geometry.require_host_memory', side_effect=refusal), \
         patch('manipulation.online_observed_geometry._decode_tsdf_source') as decode:
        with pytest.raises(ResourceLimitError):
            FrozenObservedGeometry(record)
        decode.assert_not_called()
    geometry = FrozenObservedGeometry(record, max_host_rss_bytes=3*1024**3)
    with patch('manipulation.online_observed_geometry.require_host_memory', side_effect=refusal) as admit, \
         patch('manipulation.online_observed_geometry.np.zeros') as allocate:
        with pytest.raises(ResourceLimitError):
            geometry.distance_field([-.02]*3, (5, 5, 5), .01)
        allocate.assert_not_called()
        assert admit.call_args.kwargs['max_rss_bytes'] == 3*1024**3


def test_empty_permission_binding_cannot_later_be_mutated_by_a_clone():
    geometry = points_source([[0., 0., 0.]])
    geometry.prepare_rules(())
    geometry.prepare_rules(())
    with pytest.raises(ValueError, match='cannot change'):
        geometry.prepare_rules([rule(center=(0., 0., 0.))])


def test_source_export_and_component_traversal_admit_before_their_allocations():
    export = Mock()
    mapper = SimpleNamespace(config=SimpleNamespace(enable_static=False, grid_shape=(2, 2, 2), max_blocks=1),
                             tsdf=SimpleNamespace(block_size=2, export_blocks=export))
    refusal = ResourceLimitError('test', 2, 1)
    with patch('manipulation.tsdf_geometry.require_host_memory', side_effect=refusal) as admit:
        with pytest.raises(ResourceLimitError):
            build_tsdf_geometry(mapper, 'source', 'CPU fixture', max_host_rss_bytes=3*1024**3)
        export.assert_not_called()
        assert admit.call_args.kwargs['max_rss_bytes'] == 3*1024**3
    values = np.ones((2, 2, 2), dtype=np.float32)
    with patch('manipulation.tsdf_geometry.require_host_memory', side_effect=refusal), \
         patch('manipulation.tsdf_geometry.np.zeros') as allocate:
        with pytest.raises(ResourceLimitError):
            tsdf_geometry_from_samples(values, np.ones_like(values, dtype=bool), origin=[0.]*3,
                voxel_size=.01, name='source', source='CPU fixture')
        allocate.assert_not_called()
# END no test imports a Mapper instance, CUDA context or simulator model.
