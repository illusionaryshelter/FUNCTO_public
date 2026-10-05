"""One observed TSDF reconstruction for surface distances and negative material.

The finite-element cells below join observed sample centres, not voxel faces.
Each complete tetrahedron has an affine TSDF: its zero section and its clipped
negative convex piece therefore describe the same geometry. Unknown samples
never acquire a value or an occupied neighbourhood.
"""

from itertools import combinations
import math
from weakref import WeakSet

import coal
import numpy as np
from scipy.spatial import ConvexHull

from .collision_records import MAX_ARRAY_BYTES, _array_layout, decode_array, encode_array
from .resource_admission import DEFAULT_MAX_HOST_RSS_BYTES, require_host_memory


# BEGIN reconstruct observed affine TSDF geometry without ESDF threshold solids.
# The six tetrahedra are the permutations of the three coordinate increments
# along the 000--111 body diagonal. Applying this same subdivision to every
# cell gives identical diagonals on both sides of every shared square face.
_CORNERS = np.asarray([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.int64)
_TETRAHEDRA = np.asarray(((0, 4, 6, 7), (0, 4, 5, 7), (0, 2, 6, 7),
                         (0, 2, 3, 7), (0, 1, 5, 7), (0, 1, 3, 7)), dtype=np.int64)
_EDGES = tuple(combinations(range(4), 2))


def _require_sample_array_budget(shape):
    """Admit the dense float32 source against the existing record-array limit."""
    if math.prod(shape) * np.dtype(np.float32).itemsize > MAX_ARRAY_BYTES:
        raise ValueError("Observed TSDF source samples exceed the existing 32 MiB collision-array budget")


class _Components:
    """Accumulate a bounded CSR vertex array; no point becomes a full voxel."""

    def __init__(self):
        self.vertices = []
        self.offsets = [0]

    def append(self, vertices):
        points = np.unique(np.asarray(vertices, dtype=np.float64).reshape(-1, 3), axis=0)
        end = self.offsets[-1] + len(points)
        if end * 3 * np.dtype(np.float64).itemsize > MAX_ARRAY_BYTES:
            raise ValueError("Observed TSDF components exceed the collision-record array budget")
        self.vertices.append(points)
        self.offsets.append(end)

    def arrays(self):
        vertices = np.concatenate(self.vertices) if self.vertices else np.empty((0, 3), dtype=np.float64)
        return vertices, np.asarray(self.offsets, dtype=np.int64)


def tsdf_geometry_from_samples(sdf, observed, *, origin, voxel_size, name, source):
    """Build a collision inventory from CPU XYZ sample arrays.

    ``origin`` is the centre of sample ``[0, 0, 0]``. A tetrahedron exists only
    when all four of its samples are observed. Its affine negative half-space
    is represented exactly by the negative/zero original vertices and the
    zero crossings on its six edges; a whole environment convex hull is never
    taken. Fully negative eight-corner cells use a compact integer inventory.

    Observed negative or zero samples that belong to no retained tetrahedron
    remain point components. A four-zero tetrahedron is a zero-valued region,
    so its entire convex tetrahedron, rather than merely its boundary, remains
    a surface component for containment queries.
    """
    values = np.asarray(sdf)
    valid = np.asarray(observed)
    _require_sample_array_budget(values.shape)
    start = np.asarray(origin, dtype=np.float64)
    size = float(voxel_size)
    if (values.ndim != 3 or any(n < 1 for n in values.shape) or values.dtype.kind not in "iuf"
            or valid.shape != values.shape or valid.dtype != np.dtype(bool)
            or start.shape != (3,) or not np.isfinite(start).all()
            or not np.isfinite(size) or size <= 0 or not np.isfinite(values[valid]).all()
            or not isinstance(name, str) or not name or not isinstance(source, str) or not source):
        raise ValueError("TSDF geometry requires finite observed XYZ samples, a metric centre lattice, name and source")
    surface, negative = _Components(), _Components()
    covered = np.zeros(values.shape, dtype=bool)
    cell_shape = tuple(n - 1 for n in values.shape)
    negative_cells = np.empty(0, dtype=np.int64)

    if all(cell_shape):
        fully_negative = np.ones(cell_shape, dtype=bool)
        candidate = np.zeros(cell_shape, dtype=bool)
        corner_slices = [tuple(slice(int(k), int(k) + n) for k, n in zip(corner, cell_shape))
                         for corner in _CORNERS]
        for slices in corner_slices:
            fully_negative &= valid[slices] & (values[slices] < 0)
            candidate |= valid[slices] & (values[slices] <= 0)
        negative_cells = np.flatnonzero(fully_negative).astype(np.int64, copy=False)
        for slices in corner_slices:
            covered[slices] |= fully_negative
        candidate &= ~fully_negative

        for flat in np.flatnonzero(candidate):
            lower = np.asarray(np.unravel_index(flat, cell_shape), dtype=np.int64)
            indices = lower + _CORNERS
            # Only eight source values need double-precision interpolation;
            # retain the full dense snapshot in its original float32 storage.
            cell_values = values[tuple(indices.T)].astype(np.float64, copy=False)
            cell_valid = valid[tuple(indices.T)]
            points = start + indices * size
            for tetra in _TETRAHEDRA:
                if not cell_valid[tetra].all():
                    continue
                scalar = cell_values[tetra]
                if scalar.min() > 0:
                    continue
                vertices = points[tetra]
                crossing = []
                for a, b in _EDGES:
                    if (scalar[a] < 0 < scalar[b]) or (scalar[b] < 0 < scalar[a]):
                        crossing.append(vertices[a] + scalar[a] / (scalar[a] - scalar[b])
                                        * (vertices[b] - vertices[a]))
                crossing = np.asarray(crossing, dtype=np.float64).reshape(-1, 3)
                if scalar.min() < 0:
                    negative.append(np.concatenate((vertices[scalar <= 0], crossing)))
                if scalar.min() <= 0 <= scalar.max():
                    surface.append(np.concatenate((vertices[scalar == 0], crossing)))
                represented = indices[tetra[scalar <= 0]]
                covered[tuple(represented.T)] = True

    # Isolated evidence has zero spatial extent. Expanding it would reintroduce
    # the very sample-to-filled-cube geometry error this representation removes.
    for index in np.argwhere(valid & ~covered & (values <= 0)):
        point = start + index * size
        (negative if values[tuple(index)] < 0 else surface).append(point[None])

    surface_vertices, surface_offsets = surface.arrays()
    negative_vertices, negative_offsets = negative.arrays()
    record = {
        "kind": "tsdf", "name": name, "source": source,
        "pose": [0., 0., 0., 1., 0., 0., 0.],
        "voxel_size": size, "shape_xyz": list(values.shape), "origin": start.tolist(),
        "surface_vertices": encode_array(surface_vertices),
        "surface_offsets": encode_array(surface_offsets),
        "negative_vertices": encode_array(negative_vertices),
        "negative_offsets": encode_array(negative_offsets),
        "negative_cell_flat_indices": encode_array(negative_cells),
    }
    summary = {
        "representation": "observed_affine_tsdf",
        "observed_samples": int(np.count_nonzero(valid)),
        "negative_samples": int(np.count_nonzero(valid & (values < 0))),
        "zero_samples": int(np.count_nonzero(valid & (values == 0))),
        "surface_components": len(surface_offsets) - 1,
        "negative_components": len(negative_offsets) - 1,
        "negative_cells": len(negative_cells),
        "tsdf_voxel_size_m": size,
        "surface_model": "fixed_six_tetrahedra_affine_zero_sections",
        "negative_model": "same_affine_negative_pieces_and_observed_isolated_points",
    }
    return {"solids": [record], "query_gaps": [], "field_summary": summary}


def build_tsdf_geometry(mapper, name, source):
    """Export the live Mapper's observed dynamic TSDF into an owned CPU record.

    The public block export contains ``(sum(weight*sdf), sum(weight))`` and
    signed, centred block keys. Decode those samples using the exact native
    coordinate and minimum-weight conventions. Mesh extraction and ESDF
    seeding are not involved, and the Mapper is never modified.
    """
    config = mapper.config
    if config.enable_static:
        raise ValueError("An observed TSDF snapshot cannot include the Mapper static-history channel")
    shape = tuple(int(n) for n in config.grid_shape[::-1])
    # Admit before public export clones GPU blocks or dense CPU buffers exist.
    _require_sample_array_budget(shape)
    size = float(config.voxel_size)
    centre = config.grid_center.detach().cpu().numpy().astype(np.float64)
    origin = centre - (np.asarray(shape) - 1) * size / 2
    blocks = mapper.tsdf.export_blocks()
    keys = blocks["active_block_coords"].detach().cpu().numpy()
    data = blocks["block_data"].detach().cpu().numpy()
    del blocks
    block_size = int(mapper.tsdf.block_size)
    block_offset = (np.ceil(np.asarray(shape) / block_size).astype(np.int64)) // 2
    sdf = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    for key, payload in zip(keys, data):
        base = (np.asarray(key, dtype=np.int64) + block_offset) * block_size
        lower = np.maximum(base, 0)
        upper = np.minimum(base + block_size, shape)
        if np.any(lower >= upper):
            continue
        region = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        local = tuple(slice(int(a), int(b)) for a, b in zip(lower - base, upper - base))
        # Native local_idx = z*B*B + y*B + x. The collision record is XYZ.
        weighted = np.asarray(payload, dtype=np.float32).reshape(block_size, block_size, block_size, 2)
        weighted = weighted.transpose(2, 1, 0, 3)[local]
        weight = weighted[..., 1]
        valid = weight >= float(config.minimum_tsdf_weight)
        observed[region] = valid
        np.divide(weighted[..., 0], weight, out=sdf[region], where=valid)
    return tsdf_geometry_from_samples(sdf, observed, origin=origin, voxel_size=size, name=name, source=source)


def _component_rank(vertices):
    """Use only float64 round-off when classifying affine component dimension."""
    shifted = vertices - vertices[0]
    _, singular, basis = np.linalg.svd(shifted, full_matrices=False)
    tolerance = 8 * np.finfo(np.float64).eps * max(1., float(np.max(np.abs(vertices))))
    return int(np.count_nonzero(singular > tolerance)), basis


def _component_triangles(vertices, rank=None, basis=None):
    """Triangulate a convex section with Qhull; preserve point/line components."""
    if rank is None:
        rank, basis = _component_rank(vertices)
    if rank < 2:
        return np.empty((0, 3, 3), dtype=np.float64)
    if rank == 3:
        return vertices[ConvexHull(vertices).simplices]
    projected = (vertices - vertices[0]) @ basis[:2].T
    order = ConvexHull(projected).vertices
    faces = np.asarray([(order[0], order[i], order[i + 1]) for i in range(1, len(order) - 1)])
    return vertices[faces]


def coal_component(vertices):
    """Return an exact Coal component and its pose in these vertex coordinates.

    Points and segments use zero-radius native primitives; planar components
    use triangle BVHs; full-dimensional pieces use their own convex hull.
    This never convexifies a union of independent TSDF components.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("A TSDF component needs finite Nx3 vertices")
    rank, basis = _component_rank(points)
    pose = np.eye(4)
    if rank == 0:
        pose[:3, 3] = points[0]
        return coal.Sphere(0.), pose
    if rank == 1:
        coordinate = (points - points[0]) @ basis[0]
        first, last = points[np.argmin(coordinate)], points[np.argmax(coordinate)]
        direction = last - first
        length = float(np.linalg.norm(direction))
        direction /= length
        axis = np.eye(3)[np.argmin(np.abs(direction))]
        x_axis = np.cross(axis, direction)
        x_axis /= np.linalg.norm(x_axis)
        pose[:3, :3] = np.column_stack((x_axis, np.cross(direction, x_axis), direction))
        pose[:3, 3] = (first + last) / 2
        return coal.Capsule(0., length), pose
    if rank == 2:
        triangles = _component_triangles(points, rank, basis)
        shape = coal.BVHModelOBBRSS()
        shape.beginModel(len(triangles), 3 * len(triangles))
        for triangle in triangles:
            shape.addTriangle(*triangle)
        shape.endModel()
        return shape, pose
    native = coal.StdVec_Vec3s()
    for point in points:
        native.append(point)
    return coal.ConvexBase.convexHull(native, True, "Qt"), pose


# BEGIN raw contact readers share source decoding without collision allocation.
def _decode_component_arrays(record, kind, maximum_vertices):
    vertices = decode_array(record[kind+"_vertices"])
    offsets = decode_array(record[kind+"_offsets"])
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype.kind != "f"
            or not np.isfinite(vertices).all() or offsets.ndim != 1 or offsets.dtype.kind != "i"
            or not len(offsets) or offsets[0] != 0 or offsets[-1] != len(vertices)
            or np.any(np.diff(offsets) < 1) or np.any(np.diff(offsets) > maximum_vertices)):
        raise ValueError("TSDF component vertices and CSR offsets must describe complete bounded convex pieces")
    return vertices, offsets


def _decode_tsdf_source(record):
    """Validate the one common source definition before any native adaptation."""
    if record["kind"] != "tsdf":
        raise ValueError("Observed components need an affine TSDF record")
    origin = np.asarray(record["origin"], dtype=np.float64)
    shape = np.asarray(record["shape_xyz"], dtype=np.int64)
    size = float(record["voxel_size"])
    if (origin.shape != (3,) or not np.isfinite(origin).all()
            or shape.shape != (3,) or np.any(shape < 1)
            or not np.isfinite(size) or size <= 0):
        raise ValueError("TSDF component records need a finite positive sample-centre lattice")
    surface_vertices, surface_offsets = _decode_component_arrays(record, "surface", 4)
    negative_vertices, negative_offsets = _decode_component_arrays(record, "negative", 6)
    cells = decode_array(record["negative_cell_flat_indices"])
    if (cells.ndim != 1 or cells.dtype.kind != "i" or np.any(cells < 0)
            or np.any(cells >= np.prod(shape-1)) or (len(cells) > 1 and np.any(np.diff(cells) <= 0))):
        raise ValueError("Negative TSDF cells need unique ordered in-range centre-cell indices")
    return origin, shape, size, surface_vertices, surface_offsets, negative_vertices, negative_offsets, cells


def iter_source_components(record):
    """Yield original ordered vertices to raw consumers without a shape/tree.

    Contact guidance only consumes vertices and constructs its own required
    contact representation. Reserving a complete unused Coal shape inventory
    for that reader would misstate its resource needs and rebuild geometry.
    """
    origin, shape, size, sv, so, nv, no, cells = _decode_tsdf_source(record)
    for kind, vertices, offsets in (("surface", sv, so), ("negative", nv, no)):
        for index in range(len(offsets)-1):
            yield kind, index, vertices[offsets[index]:offsets[index+1]]
    for index, flat in enumerate(cells):
        indices = np.asarray(np.unravel_index(flat, tuple(shape-1)))
        lower = origin+indices*size
        yield "negative_cell", index, lower+_CORNERS*size
# END raw and native consumers use identical validation and vertex arithmetic.


class ObservedTsdfGeometry:
    """Immutable source components, one spatial tree and lifetime-owned shapes.

    Bounds and all returned vertices use the Mapper integration frame. The
    caller applies an owner's current pose once. The tree preserves the closed
    AABB candidate predicate. Shapes are constructed only on first use and stay
    with this source map; query poses, distances and permissions never do.
    """

    # BEGIN count every still-live map's unmaterialized shapes before allocation.
    # Weak ownership neither prolongs retired maps nor duplicates the reservation
    # when several scene snapshots share the same unchanged coarse/owner field.
    _live_geometries = WeakSet()

    @classmethod
    def pending_storage_total_nbytes(cls):
        return sum(value.pending_storage_nbytes for value in cls._live_geometries)

    @property
    def pending_storage_nbytes(self):
        """Admitted shapes not yet represented in the process's resident bytes."""
        return (self._collision_cache_capacity-len(self._collision_cache))*4096

    @staticmethod
    def initial_allocation_bound(record):
        """Bound decoding, tree/BVH construction and full future shape residency.

        The codec's existing layout checks run without decoding any payload.
        Four triangles per zero component cover both tetrahedral and planar
        sections. Extra source/triangle copies cover the temporary construction
        lists; the map keeps only the final arrays and native indices afterward.
        """
        sizes = {}
        for name in ("surface_vertices", "surface_offsets", "negative_vertices",
                     "negative_offsets", "negative_cell_flat_indices"):
            layout = record[name]
            if not isinstance(layout, dict) or set(layout) != {"shape", "dtype", "data_base64"}:
                raise ValueError("TSDF arrays require the complete bounded collision codec")
            _, sizes[name] = _array_layout(layout["shape"], layout["dtype"])
        counts = []
        for name in ("surface_offsets", "negative_offsets", "negative_cell_flat_indices"):
            shape = record[name]["shape"]
            if len(shape) != 1 or (name.endswith("offsets") and shape[0] < 1):
                raise ValueError("TSDF offsets and cell identities require one bounded axis")
            counts.append(shape[0]-int(name.endswith("offsets")))
        count = sum(counts)
        # 256 B per component is the native tree's complete construction reserve,
        # not just sizeof(node); 64 B covers Python-owned boxes/flags/index scratch.
        return (3*sum(sizes.values()) + count*(4096+256+64)
                + counts[0]*(4*512+512) + 8*1024**2)
    # END full map capacity is reserved before a sparse first query can hide it.

    def __init__(self, record, *, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
        if record["kind"] != "tsdf":
            raise ValueError("ObservedTsdfGeometry needs an affine TSDF record")
        # BEGIN refuse before decoding arrays or constructing native tree nodes.
        require_host_memory(self.initial_allocation_bound(record),
            pending_existing_bytes=self.pending_storage_total_nbytes(),
            max_rss_bytes=max_host_rss_bytes)
        try:
            from ._coal_scene_batch import ComponentAABBTree
        except ImportError as error:
            raise RuntimeError("Build the project Coal adapter with PYTHONNOUSERSITE=1 "
                               "python tests/planning/build_coal_scene_batch.py") from error
        # END there is one production search implementation, not a legacy fallback.
        (self.origin, self.shape_xyz, self.voxel_size, self.surface_vertices, self.surface_offsets,
         self.negative_vertices, self.negative_offsets, self.negative_cell_flat_indices) = _decode_tsdf_source(record)
        cells = self.negative_cell_flat_indices
        cell_shape = self.shape_xyz - 1
        self._counts = (len(self.surface_offsets) - 1, len(self.negative_offsets) - 1, len(cells))
        lower, upper = [], []
        for vertices, offsets in ((self.surface_vertices, self.surface_offsets),
                                  (self.negative_vertices, self.negative_offsets)):
            if len(offsets) > 1:
                lower.append(np.minimum.reduceat(vertices, offsets[:-1], axis=0))
                upper.append(np.maximum.reduceat(vertices, offsets[:-1], axis=0))
        if len(cells):
            indices = np.column_stack(np.unravel_index(cells, tuple(cell_shape)))
            cell_lower = self.origin + indices * self.voxel_size
            lower.append(cell_lower)
            upper.append(cell_lower + self.voxel_size)
        self._lower = np.concatenate(lower) if lower else np.empty((0, 3))
        self._upper = np.concatenate(upper) if upper else np.empty((0, 3))
        self.bounds = (self._lower.min(axis=0), self._upper.max(axis=0)) if lower else None
        # BEGIN point/segment native frames require their own enclosure proof.
        # Until such a proof exists these leaves are always visited, while still
        # participating in the exact original local AABB candidate selection.
        self._prunable = np.ones(sum(self._counts), dtype=bool)
        # END ordinary area/volume pieces retain identity local transforms.

        # One BVH supplies the positive distance to all two-dimensional zero
        # sections. Lower-dimensional zero evidence has separate exact native
        # primitives. Zero tetrahedra also remain components for containment.
        triangles, lower_dimensional = [], []
        for index in range(self._counts[0]):
            points = self.surface_vertices[self.surface_offsets[index]:self.surface_offsets[index + 1]]
            rank, basis = _component_rank(points)
            if rank < 2:
                lower_dimensional.append(index)
                self._prunable[index] = False
            else:
                triangles.append(_component_triangles(points, rank, basis))
        self.surface_lower_dimensional_indices = np.asarray(lower_dimensional, dtype=np.int64)
        self.surface_shape = None
        triangle_count = sum(len(item) for item in triangles)
        if triangle_count:
            self.surface_shape = coal.BVHModelOBBRSS()
            self.surface_shape.beginModel(triangle_count, 3 * triangle_count)
            for batch in triangles:
                for triangle in batch:
                    self.surface_shape.addTriangle(*triangle)
            self.surface_shape.endModel()
        # BEGIN classify negative pieces with the same native-shape SVD rule.
        # Group equal vertex counts and process bounded blocks: NumPy performs
        # the same per-piece SVD without a Python call for every untouched leaf.
        sizes = np.diff(self.negative_offsets)
        for size in range(1, 7):
            indices = np.flatnonzero(sizes == size)
            for start in range(0, len(indices), 4096):
                selected = indices[start:start+4096]
                if size < 3:
                    self._prunable[self._counts[0]+selected] = False
                    continue
                points = self.negative_vertices[
                    self.negative_offsets[selected, None]+np.arange(size)]
                _, singular, _ = np.linalg.svd(points-points[:, :1], full_matrices=False)
                tolerance = 8*np.finfo(np.float64).eps*np.maximum(1., np.abs(points).max(axis=(1, 2)))
                self._prunable[self._counts[0]+selected] = (singular > tolerance[:, None]).sum(axis=1) >= 2
        self._spatial_index = ComponentAABBTree(self._lower, self._upper, self._prunable)
        # END one native tree serves both raw candidate selection and bound search.
        arrays = (self.origin, self.shape_xyz, self.surface_vertices, self.surface_offsets,
                  self.negative_vertices, self.negative_offsets, self.negative_cell_flat_indices,
                  self._lower, self._upper, self._prunable, self.surface_lower_dimensional_indices)
        # Include native search/BVH storage in the existing scene CPU budget.
        self.storage_nbytes = sum(array.nbytes for array in arrays) + triangle_count * 512
        self.storage_nbytes += self._spatial_index.storage_nbytes
        # BEGIN every source component is built at most once during this map.
        # Keep the established 4 KiB per-shape allowance, but reserve all unique
        # IDs rather than an evicting subset. Query batches borrow these objects.
        self._collision_cache_capacity = sum(self._counts)
        self._collision_cache = {}
        self.storage_nbytes += self.collision_cache_storage_nbytes
        self._live_geometries.add(self)
        # END map identity, not state count, owns the complete residency bound.

    # BEGIN expose the persistent portion already included in storage_nbytes.
    @property
    def collision_cache_storage_nbytes(self):
        """Full source-component residency reserve, independent of occupancy."""
        return self._collision_cache_capacity * 4096
    # END a query borrows cached shapes without cloning this reserve.

    # BEGIN select exact component IDs before deciding whether vertices are needed.
    def iter_components(self, bounds=None, *, surface=True, negative=True):
        """Yield ``(kind, index, vertices)`` for every requested AABB candidate.

        Kinds are ``surface``, ``negative`` and ``negative_cell``. ``bounds``
        is an optional pair of lower/upper XYZ vectors in the same frame as
        the vertices. A zero-valued tetrahedron is intentionally a complete
        convex surface component, allowing a caller to catch containment.
        """
        # BEGIN share candidate identity without preparing unused vertices.
        for kind, index in self._iter_component_ids(bounds, surface=surface, negative=negative):
            yield kind, index, self._component_vertices(kind, index)
        # END public component coordinates and enumeration stay unchanged.

    def _iter_component_ids(self, bounds=None, *, surface=True, negative=True):
        """Select the original ordered IDs before any component is materialized."""
        for index in self.component_indices(bounds, surface=surface, negative=negative):
            yield self.component_identity(int(index))

    # BEGIN native candidate IDs do not allocate Python tuples for discarded leaves.
    def component_indices(self, bounds=None, *, surface=True, negative=True):
        count = sum(self._counts)
        if bounds is None:
            candidates = np.arange(count, dtype=np.intp)
        else:
            lower, upper = np.asarray(bounds, dtype=np.float64)
            if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all() or np.any(lower > upper):
                raise ValueError("TSDF candidate bounds require finite ordered XYZ corners")
            candidates = self._spatial_index.overlap_ids(lower, upper)
        if not surface:
            candidates = candidates[candidates >= self._counts[0]]
        if not negative:
            candidates = candidates[candidates < self._counts[0]]
        return candidates

    def component_identity(self, flat_index):
        if type(flat_index) is not int or not 0 <= flat_index < sum(self._counts):
            raise ValueError("A TSDF component identity must belong to this source map")
        surfaces, negatives, _ = self._counts
        if flat_index < surfaces:
            return "surface", flat_index
        if flat_index < surfaces+negatives:
            return "negative", flat_index-surfaces
        return "negative_cell", flat_index-surfaces-negatives
    # END flat indices preserve the former surface/negative/cell enumeration.

    def _component_vertices(self, kind, index):
        """Materialize one selected component with the original arithmetic."""
        # BEGIN only a raw-component consumer or a cache miss needs vertices.
        if kind == "surface":
            return self.surface_vertices[self.surface_offsets[index]:self.surface_offsets[index + 1]]
        if kind == "negative":
            return self.negative_vertices[self.negative_offsets[index]:self.negative_offsets[index + 1]]
        # The private ID iterator supplies only the three validated kinds.
        offset = self._counts[0] + self._counts[1] + index
        return self._lower[offset] + _CORNERS * self.voxel_size
        # END cache hits avoid both array slicing and eight-corner allocation.
    # END both public raw components and native cache misses use the same vertices.

    # BEGIN reuse only snapshot-local Coal geometry, never query decisions.
    def iter_collision_components(self, bounds=None):
        """Yield candidate IDs, native shapes, local poses and native-frame points.

        Coal's distance/collision calls read these shapes. Point/segment
        primitives use translated/rotated coordinates, so their original
        vertices must undergo the same local conversion used before caching.
        Callers still compose the owner's current world pose for every query.
        Every cache entry lives until the source map is released. A later state
        or a new rollout never rebuilds or evicts an already used component.
        """
        for flat_index in self.component_indices(bounds):
            yield self.collision_component(int(flat_index))

    def collision_component(self, flat_index):
        """Construct a selected native leaf once, preserving original arithmetic."""
        kind, index = self.component_identity(flat_index)
        key = (kind, index)
        component = self._collision_cache.get(key)
        if component is None:
            vertices = self._component_vertices(kind, index)
            shape, local = coal_component(vertices)
            points = (vertices-local[:3, 3]) @ local[:3, :3]
            local.setflags(write=False)
            points.setflags(write=False)
            component = (shape, local, points)
            self._collision_cache[key] = component
        return kind, index, *component
    # END no world pose, distance/contact result or permission enters the cache.

    # BEGIN spatial-query scheduling now owns the only original-order batches.
    # tsdf_spatial_query.iter_component_distances borrows collision_component
    # entries and defers preparation errors until their original source order.
    # END the former unfiltered batch iterator has no remaining callers.
# END one observed reconstruction supplies both zero distances and negative material.
