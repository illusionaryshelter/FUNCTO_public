"""One frozen observed TSDF source for proposal fields and native confirmation.

The source consists of affine zero sections, negative convex pieces and full
negative sample-centre cells. A numerical field conservatively encloses that
source; its zero samples are never reconstructed as physical obstacle cubes.
Only local source components requested by native confirmation become Coal
shapes. Prepared native slots retain these immutable local transforms, without
a global surface BVH, world-pose cache or second scene query.
"""

# BEGIN shared observed components replace ESDF-seed obstacle material.
from copy import deepcopy
import math

import coal
import numpy as np
from scipy.ndimage import distance_transform_edt

from .collision_records import MAX_ARRAY_BYTES, _array_layout
from .resource_admission import DEFAULT_MAX_HOST_RSS_BYTES, require_host_memory
from .tsdf_geometry import _CORNERS, _decode_tsdf_source, coal_component


_ARRAY_NAMES = ('surface_vertices', 'surface_offsets', 'negative_vertices',
                'negative_offsets', 'negative_cell_flat_indices')
_CHUNK = 16_384
_SHAPE_BYTES = 4096


def _readonly(array):
    array.flags.writeable = False
    return array


class FrozenObservedGeometry:
    """Immutable source arrays and index shared by every clone of one scene.

    ``storage_nbytes`` includes resident source/index storage and conservative
    charges for constructed native shapes. ``pending_cache_bytes`` covers all
    remaining lazy shape slots. Fully negative cells share one Box and never
    reserve or retain a separate shape or world transform per grid cell.
    """

    def __init__(self, record, *, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
        self.max_host_rss_bytes = max_host_rss_bytes
        sizes = {name: _array_layout(record[name]['shape'], record[name]['dtype'])[1]
                 for name in _ARRAY_NAMES}
        counts = tuple(record[name]['shape'][0]-int(name.endswith('offsets')) for name in
                       ('surface_offsets', 'negative_offsets', 'negative_cell_flat_indices'))
        if min(counts) < 0:
            raise ValueError('Observed source component counts must be nonnegative')
        count = sum(counts)
        self.maximum_cache_bytes = (counts[0]+counts[1]+int(counts[2] > 0))*_SHAPE_BYTES
        # Decode scratch, retained source arrays and complete tree construction
        # are admitted before decoding base64 or allocating native index nodes.
        require_host_memory(int(3*sum(sizes.values())+count*384+8*1024**2),
            pending_existing_bytes=self.maximum_cache_bytes,
            max_rss_bytes=self.max_host_rss_bytes)
        if not np.array_equal(np.asarray(record['pose']), [0., 0., 0., 1., 0., 0., 0.]):
            raise ValueError('Frozen observed source vertices must already use their integration frame')
        self._record = deepcopy(record)
        (self.origin, self.shape_xyz, self.voxel_size, self.surface_vertices,
         self.surface_offsets, self.negative_vertices, self.negative_offsets,
         self.negative_cell_flat_indices) = _decode_tsdf_source(self._record)
        self._counts = counts
        self.component_count = count
        lower, upper = [], []
        for vertices, offsets in ((self.surface_vertices, self.surface_offsets),
                                  (self.negative_vertices, self.negative_offsets)):
            if len(offsets) > 1:
                lower.append(np.minimum.reduceat(vertices, offsets[:-1], axis=0))
                upper.append(np.maximum.reduceat(vertices, offsets[:-1], axis=0))
        if counts[2]:
            cells = np.column_stack(np.unravel_index(
                self.negative_cell_flat_indices, tuple(self.shape_xyz-1)))
            starts = self.origin+cells*self.voxel_size
            lower.append(starts)
            upper.append(starts+self.voxel_size)
        self._lower = np.ascontiguousarray(np.concatenate(lower) if lower else np.empty((0, 3)))
        self._upper = np.ascontiguousarray(np.concatenate(upper) if upper else np.empty((0, 3)))
        self.bounds = ((_readonly(self._lower.min(0)), _readonly(self._upper.max(0)))
                       if count else None)
        # Reuse the existing tree's closed AABB overlap operation. The separate
        # prepared table below stores local shapes, never distance bounds.
        from ._coal_scene_batch import ComponentAABBTree
        self._index = ComponentAABBTree(self._lower, self._upper, np.ones(count, dtype=bool))
        arrays = (self.origin, self.shape_xyz, self.surface_vertices, self.surface_offsets,
                  self.negative_vertices, self.negative_offsets, self.negative_cell_flat_indices,
                  self._lower, self._upper)
        for array in arrays:
            _readonly(array)
        encoded_bytes = sum(len(record[name]['data_base64']) for name in _ARRAY_NAMES)
        self._base_storage_bytes = int(sum(array.nbytes for array in arrays)
            + encoded_bytes+self._index.storage_nbytes+16_384)
        self._cache = {}
        self._negative_cell_shape = None
        self._box_local = _readonly((_CORNERS-.5)*self.voxel_size)
        self._base_storage_bytes += self._box_local.nbytes
        self._cache_bytes = 0
        self._rule_records = None
        self._rule_bits = {}
        self._removed_by_rule = _readonly(np.zeros(count, dtype=np.uint16))
        self._regions = {}
        # BEGIN one frozen source owns the existing native batch's local slots.
        # Include the table and prepared-slot capacity before lazy allocation;
        # neither trajectory length nor the number of rollout clones multiplies it.
        self._prepared_capacity_bytes = 8192+1024*count
        self._base_storage_bytes += self._prepared_capacity_bytes
        self._prepared_components = None
        # END no candidate, rule or measured distance enters this shared state.

    def prepared_components(self):
        """Borrow immutable target-local slots for the installed batch API."""
        if self._prepared_components is None:
            from ._coal_scene_batch import ComponentTable, PreparedComponents
            require_host_memory(self._prepared_capacity_bytes,
                pending_existing_bytes=self.pending_cache_bytes,
                max_rss_bytes=self.max_host_rss_bytes)
            table = ComponentTable(self.component_count)
            prepared = PreparedComponents(table)
            if table.storage_nbytes+prepared.storage_nbytes > self._prepared_capacity_bytes:
                raise RuntimeError('Native component slots exceed their admitted source capacity')
            self._prepared_components = prepared
        return self._prepared_components

    @property
    def storage_nbytes(self):
        return self._base_storage_bytes+self._removed_by_rule.nbytes+self._cache_bytes

    @property
    def pending_cache_bytes(self):
        return self.maximum_cache_bytes-self._cache_bytes

    def record(self):
        """Export the same source values; callers cannot mutate this binding."""
        return deepcopy(self._record)

    def _vertices(self, component_id):
        if not isinstance(component_id, (int, np.integer)) or not 0 <= component_id < self.component_count:
            raise ValueError('Observed component ID must belong to this frozen source')
        component_id = int(component_id)
        if component_id < self._counts[0]:
            return ('surface', component_id,
                    self.surface_vertices[self.surface_offsets[component_id]:self.surface_offsets[component_id+1]])
        local = component_id-self._counts[0]
        if local < self._counts[1]:
            return ('negative', local,
                    self.negative_vertices[self.negative_offsets[local]:self.negative_offsets[local+1]])
        local -= self._counts[1]
        return 'negative_cell', local, self._lower[component_id]+_CORNERS*self.voxel_size

    def prepare_rules(self, rules):
        """Bind complete-component permission once, shared by every consumer.

        Empty permissions are a complete binding too. A scene must finish its
        contact registration before either confirmation or contact preparation
        first freezes this source; a clone cannot mutate that frozen contract.
        """
        rules = tuple(rules)
        if len(rules) > 16 or len({rule.rule_id for rule in rules}) != len(rules):
            raise ValueError('Observed source permissions require at most sixteen unique rules')
        records = {}
        for rule in rules:
            records[rule.rule_id] = (tuple(np.asarray(rule.patch_origin).tolist()),
                tuple(np.asarray(rule.patch_axes).reshape(-1).tolist()),
                tuple(np.asarray(rule.patch_half_size).tolist()), float(rule.modeling_allowance_m),
                float(rule.numerical_position_tolerance_m))
        if self._rule_records is not None:
            if records != self._rule_records:
                raise ValueError('Frozen observed source permissions cannot change after preparation')
            return
        require_host_memory(int(self.component_count*2+_CHUNK*512),
            pending_existing_bytes=self.pending_cache_bytes, max_rss_bytes=self.max_host_rss_bytes)
        removed = np.zeros(self.component_count, dtype=np.uint16)
        regions, bits = {}, {}
        for ordinal, rule in enumerate(rules):
            axes = np.asarray(rule.patch_axes, dtype=float)
            origin = np.asarray(rule.patch_origin, dtype=float)
            widths = np.r_[rule.patch_half_size, rule.modeling_allowance_m]
            if (axes.shape != (3, 3) or origin.shape != (3,) or widths.shape != (3,)
                    or not np.isfinite(axes).all() or not np.isfinite(origin).all()
                    or not np.isfinite(widths).all() or np.any(widths < 0)
                    or not np.isfinite(rule.numerical_position_tolerance_m)
                    or rule.numerical_position_tolerance_m < 0):
                raise ValueError('Finite source permission needs valid local patch geometry')
            bit = 1 << ordinal
            bits[rule.rule_id] = bit
            region_lower, region_upper, selected_count = np.full(3, np.inf), np.full(3, -np.inf), 0
            # The original CSR components contain at most six vertices. Work
            # in bounded slices; no source-by-rule-by-grid tensor is retained.
            base = 0
            for vertices, offsets in ((self.surface_vertices, self.surface_offsets),
                                      (self.negative_vertices, self.negative_offsets)):
                for start in range(0, len(offsets)-1, _CHUNK):
                    stop = min(len(offsets)-1, start+_CHUNK)
                    first, last = offsets[start], offsets[stop]
                    local = (vertices[first:last]-origin) @ axes
                    starts = offsets[start:stop]-first
                    lo = np.minimum.reduceat(local, starts, axis=0)
                    hi = np.maximum.reduceat(local, starts, axis=0)
                    inside = (np.maximum(np.abs(lo), np.abs(hi))
                              <= widths+rule.numerical_position_tolerance_m).all(-1)
                    removed[base+start:base+stop] |= inside.astype(np.uint16)*bit
                    if inside.any():
                        region_lower = np.minimum(region_lower, lo[inside].min(0))
                        region_upper = np.maximum(region_upper, hi[inside].max(0))
                        selected_count += int(inside.sum())
                base += len(offsets)-1
            support = np.abs(axes).sum(0)*self.voxel_size/2
            for start in range(base, self.component_count, _CHUNK):
                stop = min(self.component_count, start+_CHUNK)
                center = ((self._lower[start:stop]+self._upper[start:stop])/2-origin) @ axes
                inside = (np.abs(center)+support <= widths+rule.numerical_position_tolerance_m).all(-1)
                removed[start:stop] |= inside.astype(np.uint16)*bit
                if inside.any():
                    region_lower = np.minimum(region_lower, (center[inside]-support).min(0))
                    region_upper = np.maximum(region_upper, (center[inside]+support).max(0))
                    selected_count += int(inside.sum())
            regions[rule.rule_id] = {
                'lower_patch': _readonly(region_lower if selected_count else np.zeros(3)),
                'upper_patch': _readonly(region_upper if selected_count else np.zeros(3)),
                'removed_components': selected_count,
            }
        self._rule_records, self._rule_bits = records, bits
        self._removed_by_rule, self._regions = _readonly(removed), regions

    def _active_bits(self, active_rule_ids):
        identities = frozenset(active_rule_ids)
        if not identities <= self._rule_bits.keys():
            raise ValueError('Active permission must name a prepared observed-source rule')
        return sum(self._rule_bits[name] for name in identities)

    def component_indices(self, bounds, active_rule_ids=()):
        """Return source IDs intersecting a local closed box after permission."""
        bits = self._active_bits(active_rule_ids)
        if bounds is None:
            ids = np.arange(self.component_count, dtype=np.intp)
        else:
            lower, upper = (np.asarray(value, dtype=np.float64) for value in bounds)
            if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
                    or np.any(lower > upper)):
                raise ValueError('Observed source query requires a finite ordered local AABB')
            ids = self._index.overlap_ids(lower, upper)
        return ids[(self._removed_by_rule[ids] & bits) == 0] if bits else ids

    def collision_component(self, component_id):
        """Create only the requested local Coal shape; share it across queries."""
        if not isinstance(component_id, (int, np.integer)) or not 0 <= component_id < self.component_count:
            raise ValueError('Observed component ID must belong to this frozen source')
        identity = int(component_id)
        if identity >= self._counts[0]+self._counts[1]:
            if self._negative_cell_shape is None:
                require_host_memory(_SHAPE_BYTES, pending_existing_bytes=self.pending_cache_bytes-_SHAPE_BYTES,
                    max_rss_bytes=self.max_host_rss_bytes)
                self._negative_cell_shape = coal.Box(*([self.voxel_size]*3))
                self._cache_bytes += _SHAPE_BYTES
            pose = np.eye(4)
            pose[:3, 3] = self._lower[identity]+self.voxel_size/2
            return ('negative_cell', identity-self._counts[0]-self._counts[1],
                    self._negative_cell_shape, _readonly(pose), self._box_local)
        if identity not in self._cache:
            kind, local, vertices = self._vertices(identity)
            require_host_memory(_SHAPE_BYTES, pending_existing_bytes=self.pending_cache_bytes-_SHAPE_BYTES,
                max_rss_bytes=self.max_host_rss_bytes)
            shape, pose = coal_component(vertices)
            # Point and segment primitives use translated/rotated native
            # frames. Return vertices in that shape frame, not source axes;
            # composing the returned pose exactly once recovers the source.
            local_vertices = _readonly((vertices-pose[:3, 3]) @ pose[:3, :3])
            self._cache[identity] = kind, local, shape, _readonly(pose), local_vertices
            self._cache_bytes += _SHAPE_BYTES
        return self._cache[identity]

    def rule_region(self, rule_id):
        """Bounds of fully permitted components in this rule's patch frame."""
        if rule_id not in self._regions:
            raise ValueError('A source rule region requires its frozen permission binding')
        return dict(self._regions[rule_id])

    def surface_points(self, rule_id):
        """Return real zero-section vertices of fully permitted components."""
        if rule_id not in self._rule_bits:
            raise ValueError('Source surface witnesses require a prepared rule')
        selected = (self._removed_by_rule[:self._counts[0]] & self._rule_bits[rule_id]) != 0
        # Selection and unique sorting borrow only observed zero vertices.
        # Negative-piece vertices and conservative raster cells are not contact
        # witnesses. The temporary result is owned by the contact preparer.
        count = len(self.surface_vertices)
        require_host_memory(int(count*96+len(selected)*8),
            pending_existing_bytes=self.pending_cache_bytes, max_rss_bytes=self.max_host_rss_bytes)
        mask = np.repeat(selected, np.diff(self.surface_offsets))
        return _readonly(np.unique(self.surface_vertices[mask], axis=0))

    def distance_field(self, origin, shape, voxel_size, active_rule_ids=(), *, excluded_component_ids=()):
        """Rasterize the retained source and return a conservative proposal field.

        Component AABBs over-cover the source. Grid cells intersecting those
        boxes form a second over-cover; neither becomes confirmation geometry.
        Outside occupied cells, centre EDT minus one cell half-diagonal and
        one interpolation half-diagonal bounds source distance from below.
        Occupied centres remain negative so interpolation cannot invent free
        gaps. Values are not physical penetration depths inside the source.
        """
        origin = np.asarray(origin, dtype=float)
        shape = tuple(shape)
        h = float(voxel_size)
        if (origin.shape != (3,) or not np.isfinite(origin).all() or len(shape) != 3
                or any(not isinstance(value, (int, np.integer)) or value < 2 for value in shape)
                or not np.isfinite(h) or h <= 0 or math.prod(shape)*4 > MAX_ARRAY_BYTES):
            raise ValueError('Observed proposal field needs a bounded finite XYZ sample grid')
        shape = tuple(int(value) for value in shape)
        ids = self.component_indices(None, active_rule_ids)
        # BEGIN a separately bound initial support uses exact source identities.
        # The caller established entrance native intersection and finite plane
        # scope. This operation never turns an arbitrary grid cell into material.
        excluded = np.asarray(excluded_component_ids)
        if excluded.size:
            if (excluded.ndim != 1 or excluded.dtype.kind not in 'iu'
                    or np.any(excluded < 0) or np.any(excluded >= self.component_count)
                    or len(np.unique(excluded)) != len(excluded)):
                raise ValueError('Excluded observed components must be distinct current source identities')
            ids = ids[np.isin(ids, excluded, assume_unique=True, invert=True)]
        # END ordinary fields and existing material-rule subsets are unchanged.
        if not len(ids):
            return None
        # Silent clipping would omit source material and destroy the lower
        # bound. Numerical query padding is prepared by the caller instead.
        grid_lower = origin-h/2
        grid_upper = origin+(np.asarray(shape)-1)*h+h/2
        if (np.any(self._lower[ids] < grid_lower) or np.any(self._upper[ids] > grid_upper)):
            raise ValueError('Observed source geometry extends beyond the complete proposal grid')
        cells = math.prod(shape)
        require_host_memory(int(cells*96+len(ids)*64),
            pending_existing_bytes=self.pending_cache_bytes, max_rss_bytes=self.max_host_rss_bytes)
        occupied = np.zeros(shape, dtype=bool)
        # Outward rounding at each arithmetic step retains a source point on
        # a shared cell face on both sides, including sub-voxel components.
        lower = np.nextafter(self._lower[ids]-origin, -np.inf)
        lower = np.nextafter(lower/h, -np.inf)
        lower = np.ceil(np.nextafter(lower-.5, -np.inf)).astype(np.int64)
        upper = np.nextafter(self._upper[ids]-origin, np.inf)
        upper = np.nextafter(upper/h, np.inf)
        upper = np.floor(np.nextafter(upper+.5, np.inf)).astype(np.int64)
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, np.asarray(shape)-1)
        for lo, hi in zip(lower, upper):
            occupied[tuple(slice(int(a), int(b)+1) for a, b in zip(lo, hi))] = True
        margin = math.sqrt(3.)*h
        if occupied.all():
            # scipy's all-foreground EDT has a virtual outside-array zero;
            # that is not observed material/free space. A negative constant
            # is a sufficient proposal for a fully occupied finite grid.
            values = np.full(shape, -margin, dtype=np.float64)
        else:
            values = distance_transform_edt(~occupied, sampling=h)
            inside = distance_transform_edt(occupied, sampling=h)
            values[occupied] = -inside[occupied]
            values -= margin
        return _readonly(np.nextafter(values.astype(np.float32), np.float32(-np.inf)))
# END every field, permission and native query derives from one frozen source.
