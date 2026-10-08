"""Explicit observed-source fixtures; proposal values never invent material."""

# BEGIN shared source construction for the migrated online geometry tests.
import numpy as np

from manipulation.collision_records import encode_array
from manipulation.online_geometry import FrozenESDF
from manipulation.online_observed_geometry import FrozenObservedGeometry
from manipulation.tsdf_geometry import tsdf_geometry_from_samples


def source_from_samples(values, *, origin, voxel_size, observed=None, name='test'):
    values = np.asarray(values, dtype=np.float32)
    observed = np.ones(values.shape, bool) if observed is None else np.asarray(observed, dtype=bool)
    return FrozenObservedGeometry(tsdf_geometry_from_samples(values, observed,
        origin=origin, voxel_size=voxel_size, name=name, source='explicit synthetic TSDF')['solids'][0])


def point_source(points=(), *, name='test'):
    """Explicit isolated zero samples, with no artificial voxel extent."""
    return surface_source([np.asarray(point).reshape(1, 3) for point in points], name=name)


def surface_source(components=(), *, name='test'):
    """Explicit zero-section convex components for narrow-phase fixtures."""
    components = [np.asarray(points, dtype=np.float64).reshape(-1, 3) for points in components]
    vertices = np.concatenate(components) if components else np.empty((0, 3), dtype=np.float64)
    offsets = np.r_[0, np.cumsum([len(points) for points in components])].astype(np.int64)
    return FrozenObservedGeometry({'kind': 'tsdf', 'name': name,
        'source': 'explicit isolated observed zero samples',
        'origin': [0., 0., 0.], 'shape_xyz': [1, 1, 1], 'voxel_size': .001,
        'pose': [0., 0., 0., 1., 0., 0., 0.],
        'surface_vertices': encode_array(vertices),
        'surface_offsets': encode_array(offsets),
        'negative_vertices': encode_array(np.empty((0, 3), dtype=np.float64)),
        'negative_offsets': encode_array(np.zeros(1, dtype=np.int64)),
        'negative_cell_flat_indices': encode_array(np.empty(0, dtype=np.int64))})


def field_from_samples(values, *, origin, voxel_size, observed=None, owner_ids=(), source='test source'):
    geometry = source_from_samples(values, origin=origin, voxel_size=voxel_size, observed=observed)
    return FrozenESDF.from_source(geometry, origin=origin, shape=np.shape(values),
        voxel_size=voxel_size, owner_ids=owner_ids, source=source)
# END helper inputs distinguish observation samples from derived proposal fields.
