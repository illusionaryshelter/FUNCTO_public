"""Nonpenetration queries between observed trilinear occupied sets.

The TSDF values define a set; they are not Euclidean distances.  A query
searches pairs of spatial boxes containing that set.  Box distances are lower
bounds, while pairs of points in the sets give upper bounds. Supported cells
are subdivided to distinguish volume overlap from unresolved boundary domains.
The original samples and support define every geometric decision.
"""

# BEGIN direct observed-set queries replace surface/solid conversion.
from dataclasses import dataclass
import heapq
from itertools import count
import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


CORNERS = np.asarray([(x, y, z) for x in (0., 1.) for y in (0., 1.)
                      for z in (0., 1.)], dtype=np.float64)
_EPS = np.finfo(np.float64).eps
# Each evaluated coordinate uses at most 32 scalar operations.  This is a
# floating-point forward-error allowance, not a physical/model tolerance.
_GAMMA = 32 * _EPS / (1 - 32 * _EPS)
_EDGES = tuple((i, j) for i in range(8) for j in range(i + 1, 8)
               if np.count_nonzero(CORNERS[i] != CORNERS[j]) == 1)


def transform_points(points, pose):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def transform_point_error(points, pose):
    """Bound forward-transform arithmetic before cancellation occurs."""
    return _GAMMA * (np.abs(points) @ np.abs(pose[:3, :3]).T + np.abs(pose[:3, 3]))


def inverse_points(points, pose):
    return (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]


def inverse_point_error(points, pose):
    """Bound arithmetic and finite orthogonality error of the rigid inverse."""
    rotation = pose[:3, :3]
    delta = np.asarray(points) - pose[:3, 3]
    arithmetic = _GAMMA * ((np.abs(points) + np.abs(pose[:3, 3])) @ np.abs(rotation))
    defect = np.linalg.norm(rotation.T @ rotation - np.eye(3), ord=np.inf)
    defect += _GAMMA * float(np.linalg.norm(np.abs(rotation).T @ np.abs(rotation), ord=np.inf))
    if defect >= 1:
        raise ValueError("A field pose must have an invertible rigid rotation")
    inverse_error = (np.max(np.abs(delta), axis=-1) * np.linalg.norm(rotation, ord=1)
                     * defect / (1 - defect))
    return arithmetic + inverse_error[..., None]


def box_corners(lower, upper):
    return lower + CORNERS * (upper - lower)


def world_bounds(lower, upper, pose):
    """Enclose the transformed box including arithmetic round-off."""
    middle, half = (lower + upper) / 2, (upper - lower) / 2
    center = pose[:3, :3] @ middle + pose[:3, 3]
    radius = np.abs(pose[:3, :3]) @ half
    error = _GAMMA * (np.abs(pose[:3, :3]) @ (np.abs(middle) + half)
                      + np.abs(pose[:3, 3]) + np.abs(center) + radius)
    return (np.nextafter(center - radius - error, -np.inf),
            np.nextafter(center + radius + error, np.inf))


def box_distance(a, b):
    gap = np.maximum(0., np.maximum(a[0] - b[1], b[0] - a[1]))
    value = float(np.linalg.norm(gap))
    error = _GAMMA * (value + float(np.max(np.abs(np.r_[a[0], a[1], b[0], b[1]]))))
    return max(0., float(np.nextafter(value - error, -np.inf)))


def trilinear(values, coordinates):
    """Return value, analytic local-coordinate gradient, and rounding bound."""
    u = np.asarray(coordinates, dtype=np.float64)
    weights = np.where(CORNERS == 1., u[..., None, :], 1. - u[..., None, :])
    product = weights.prod(-1)
    value = np.sum(product * values, axis=-1)
    derivatives = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        derivatives.append(np.sum(weights[..., other].prod(-1)
                                  * (2 * CORNERS[:, axis] - 1) * values, axis=-1))
    gradient = np.stack(derivatives, axis=-1)
    error = _GAMMA * np.sum(np.abs(product * values), axis=-1)
    return value, gradient, error


class PairPermission:
    """A union of finite, world-frame permitted point-pair regions.

    Each entry contains ``((normals_a, bounds_a), (normals_b, bounds_b))``.
    A point pair is permitted when both endpoints satisfy the corresponding
    halfspaces of at least one entry. Interval, direction and relationship
    activation are evaluated by the caller before constructing this query.
    """

    def __init__(self, regions, *, excluded_second_cells=()):
        self.regions = tuple(regions)
        # An entrance-support declaration may remove whole original cells
        # from this pair's second domain. Cell identity implements that exact
        # subset without testing its own rounded box against itself.
        self.excluded_second_cells = frozenset(excluded_second_cells)

    def excludes_cells(self, first, second):
        return second >= 0 and second in self.excluded_second_cells

    @staticmethod
    def _classify(box, region):
        normals, bounds = region
        normals, bounds = np.asarray(normals, float), np.asarray(bounds, float)
        low, high = box
        minimum_terms = np.where(normals >= 0, normals * low, normals * high)
        maximum_terms = np.where(normals >= 0, normals * high, normals * low)
        minimum, maximum = minimum_terms.sum(-1), maximum_terms.sum(-1)
        low_error = _GAMMA * np.abs(minimum_terms).sum(-1)
        high_error = _GAMMA * np.abs(maximum_terms).sum(-1)
        if np.all(maximum + high_error <= bounds):
            return "inside"
        if np.any(minimum - low_error > bounds):
            return "outside"
        return "mixed"

    def boxes(self, first, second):
        mixed = False
        for a, b in self.regions:
            ca, cb = self._classify(first, a), self._classify(second, b)
            if ca == cb == "inside":
                return "permitted"
            if ca != "outside" and cb != "outside":
                mixed = True
        return "mixed" if mixed else "forbidden"


@dataclass(frozen=True)
class _Node:
    lower: np.ndarray
    upper: np.ndarray
    children: tuple = ()
    cell: int = -1


class ObservedField:
    """An immutable local TSDF and its actual interpolation support.

    Occupancy exists only in cells whose eight samples were observed. Unknown
    samples never receive a positive value. The spatial tree indexes cells
    that may contain nonpositive values, not a replacement solid.
    """

    def __init__(self, values, observed, *, origin, voxel_size):
        self.values = np.array(values, dtype=np.float32, copy=True)
        self.observed = np.array(observed, dtype=bool, copy=True)
        self.origin = np.asarray(origin, dtype=np.float64).copy()
        self.voxel_size = float(voxel_size)
        if (self.values.ndim != 3 or min(self.values.shape) < 2
                or self.observed.shape != self.values.shape or self.origin.shape != (3,)
                or not np.isfinite(self.origin).all() or not self.voxel_size > 0
                or not np.isfinite(self.values[self.observed]).all()):
            raise ValueError("An observed field needs finite supported XYZ samples and metric coordinates")
        self.shape = np.asarray(self.values.shape, dtype=np.int64)
        self.cell_shape = self.shape - 1
        valid = np.ones(tuple(self.cell_shape), dtype=bool)
        minimum = np.full(tuple(self.cell_shape), np.inf, dtype=np.float32)
        for corner in CORNERS.astype(int):
            slices = tuple(slice(int(k), int(k + n)) for k, n in zip(corner, self.cell_shape))
            valid &= self.observed[slices]
            np.minimum(minimum, self.values[slices], out=minimum)
        self.valid_cells = valid
        self.cells = np.argwhere(valid & (minimum <= 0))
        self.cell_origins = self.origin + self.cells * self.voxel_size
        self._cell_values = (self.values[tuple((self.cells[:, None, :] +
                             CORNERS.astype(int)[None]).transpose(2, 0, 1))].astype(float)
                             if len(self.cells) else np.empty((0, 8)))
        centers = self.cell_origins + self.voxel_size / 2
        self.tree = cKDTree(centers) if len(centers) else None
        # BEGIN local surface directions index supported zero-bearing cells.
        # Occupied interior cells remain in the collision tree. They cannot
        # displace a more distant observed zero from the direction query's
        # finite candidate list. A constant-zero cell has no surface normal.
        minimum = self._cell_values.min(axis=1) if len(self.cells) else np.empty(0)
        maximum = self._cell_values.max(axis=1) if len(self.cells) else np.empty(0)
        # A nonconstant nonnegative trilinear cell is strictly positive at
        # every interior point: all eight interpolation weights are positive.
        # Its occupied zero subset has no volume interior. Keep that subset
        # for surface tasks and report its possible contact as unresolved.
        self._boundary_only_cells = (minimum == 0) & (maximum > 0)
        self._zero_cell_ids = np.flatnonzero((minimum <= 0) & (maximum >= 0)
                                           & (minimum < maximum))
        self._zero_tree = cKDTree(centers[self._zero_cell_ids]) if len(self._zero_cell_ids) else None
        # END this additional index never changes occupied cells or their tree.
        self.nodes = []
        # BEGIN batch the same initial slab bounds without changing tree order.
        # A small batch bounds temporary interpolation storage independently
        # of map size. Subdivision later uses the same scalar tighten method.
        cell_lower, cell_upper = self._initial_cell_bounds()
        # END the indexed set and its floating-point bounds remain identical.

        def build(node):
            indices = self.tree.indices[node.start_idx:node.end_idx]
            lower = self.cell_origins[indices].min(0)
            upper = self.cell_origins[indices].max(0) + self.voxel_size
            identity = len(self.nodes)
            self.nodes.append(None)
            if node.split_dim < 0:
                children = []
                for index in indices:
                    children.append(len(self.nodes))
                    self.nodes.append(_Node(cell_lower[index], cell_upper[index],
                                            cell=int(index)))
            else:
                children = [build(node.lesser), build(node.greater)]
            self.nodes[identity] = _Node(lower, upper, tuple(children))
            return identity

        self.root = build(self.tree.tree) if self.tree is not None else None
        # BEGIN both immutable indices account for their own retained arrays.
        for array in (self.values, self.observed, self.origin, self.valid_cells,
                      self.cells, self.cell_origins, self._cell_values, self._zero_cell_ids,
                      self._boundary_only_cells):
            array.flags.writeable = False
        self.storage_nbytes = sum(a.nbytes for a in (
            self.values, self.observed, self.valid_cells, self.cells,
            self.cell_origins, self._cell_values, self._zero_cell_ids,
            self._boundary_only_cells)) + 256 * len(self.nodes)
        if self._zero_tree is not None:
            self._zero_tree.data.flags.writeable = False
            self.storage_nbytes += self._zero_tree.data.nbytes + self._zero_tree.indices.nbytes
        # END the occupied representation retains its existing storage account.

    # BEGIN vectorized initial bounds retain scalar tighten arithmetic.
    def _initial_cell_bounds(self):
        """Evaluate independent cells in batches, retaining x/y/z ordering.

        The formulas match ``cell_value`` and ``tighten`` including coordinate
        error, endpoint slope bounds and outward rounding. Only independent
        cells share a NumPy operation; each axis consumes the previous axis's
        bounds. Parent tree boxes and query-time refinement are unchanged.
        """
        lower = self.cell_origins.copy()
        upper = lower + self.voxel_size
        for start in range(0, len(self.cells), 1024):
            stop = min(start + 1024, len(self.cells))
            lo, hi = lower[start:stop], upper[start:stop]
            samples = self._cell_values[start:stop]
            scale_error = _GAMMA * np.max(np.abs(samples), axis=-1)
            indices = np.arange(start, stop)[:, None]
            for axis in range(3):
                points = lo[:, None, :] + CORNERS * (hi - lo)[:, None, :]
                actual, _, rounding = self.cell_value(indices, points)
                values = actual - rounding - scale_error[:, None]
                a = values[:, CORNERS[:, axis] == 0]
                b = values[:, CORNERS[:, axis] == 1]
                span = hi[:, axis] - lo[:, axis]
                shrink_low = np.all(a > 0, axis=-1) & np.any(b <= 0, axis=-1)
                shrink_high = (~shrink_low & np.all(b > 0, axis=-1)
                               & np.any(a <= 0, axis=-1))
                for selected, low_side in ((shrink_low, True), (shrink_high, False)):
                    ids = np.flatnonzero(selected)
                    if not len(ids):
                        continue
                    eligible = b[ids] <= 0 if low_side else a[ids] <= 0
                    roots = np.full(a[ids].shape, np.inf if low_side else -np.inf)
                    np.divide(a[ids], a[ids] - b[ids], out=roots, where=eligible)
                    fraction = roots.min(-1) if low_side else roots.max(-1)
                    value = lo[ids, axis] + span[ids] * fraction
                    error = _GAMMA * (np.abs(value) + span[ids])
                    if low_side:
                        lo[ids, axis] = np.maximum(lo[ids, axis],
                            np.nextafter(value - error, -np.inf))
                    else:
                        hi[ids, axis] = np.minimum(hi[ids, axis],
                            np.nextafter(value + error, np.inf))
        return lower, upper
    # END vectorization changes construction cost, not represented geometry.

    @property
    def bounds(self):
        if self.root is None:
            return None
        root = self.nodes[self.root]
        return root.lower, root.upper

    def cell_value(self, index, points, point_error=0.):
        u = (np.asarray(points) - self.cell_origins[index]) / self.voxel_size
        value, gradient, error = trilinear(self._cell_values[index], u)
        coordinate_error = (np.asarray(point_error) + _GAMMA * (
            np.abs(points) + np.abs(self.cell_origins[index])
            + self.voxel_size * np.abs(u))) / self.voxel_size
        # The derivative in one axis is bilinear in the other two. Its
        # extrema are therefore among that axis's four endpoint differences.
        values = self._cell_values[index]
        # BEGIN the same slope bound accepts scalar or batched cell indices.
        slope = np.stack([np.max(np.stack([
            np.abs(values[..., j] - values[..., i]) for i, j in _EDGES
            if CORNERS[i, axis] != CORNERS[j, axis]], axis=-1), axis=-1)
            for axis in range(3)], axis=-1)
        # END coordinate/error formulas are shared by both construction paths.
        error = error + np.sum(coordinate_error * slope, axis=-1)
        return value, gradient / self.voxel_size, error

    def sample(self, points):
        """Evaluate only fully supported cells, retaining an explicit valid mask."""
        points = np.asarray(points, dtype=float)
        layout = points.shape[:-1]
        flat = points.reshape(-1, 3)
        grid = (flat - self.origin) / self.voxel_size
        indices = np.floor(grid).astype(np.int64)
        # A sample at the final lattice plane belongs to the adjacent last cell.
        indices = np.minimum(indices, self.cell_shape - 1)
        inside = ((grid >= 0) & (grid <= self.shape - 1)).all(-1)
        clipped = np.clip(indices, 0, self.cell_shape - 1)
        valid = inside & self.valid_cells[tuple(clipped.T)]
        # A supported cell is closed. At an internal lattice plane either
        # adjacent cell may supply the same interpolated value; prefer the
        # floor cell, then inspect the other incident cells when necessary.
        for corner in CORNERS.astype(int)[1:]:
            candidate = indices - corner
            eligible = (inside & ~valid
                        & ((corner == 0) | (grid == indices)).all(-1)
                        & ((candidate >= 0) & (candidate < self.cell_shape)).all(-1))
            selected = np.flatnonzero(eligible)
            if len(selected):
                selected = selected[self.valid_cells[tuple(candidate[selected].T)]]
                clipped[selected] = candidate[selected]
                valid[selected] = True
        value, gradient, error = (np.full(len(flat), np.nan), np.zeros_like(flat),
                                  np.full(len(flat), np.nan))
        selected = np.flatnonzero(valid)
        if len(selected):
            samples = self.values[tuple((clipped[selected, None, :] +
                                  CORNERS.astype(int)[None]).transpose(2, 0, 1))].astype(float)
            v, g, e = trilinear(samples, grid[selected] - clipped[selected])
            coordinate_error = _GAMMA * (
                np.abs(flat[selected]) + np.abs(self.origin)
                + self.voxel_size * np.abs(grid[selected])) / self.voxel_size
            slope = np.stack([np.max(np.stack(
                [np.abs(samples[:, j] - samples[:, i]) for i, j in _EDGES
                 if CORNERS[i, axis] != CORNERS[j, axis]], -1), -1) for axis in range(3)], -1)
            e += np.sum(coordinate_error * slope, -1)
            value[selected], gradient[selected], error[selected] = v, g / self.voxel_size, e
        return (value.reshape(layout), valid.reshape(layout),
                gradient.reshape(*layout, 3), error.reshape(layout))

    def node_values(self, node):
        return self.cell_value(node.cell, box_corners(node.lower, node.upper))[0]

    def tighten(self, node):
        """Clip only provably positive face slabs from a mixed cell's box.

        At a fixed axis coordinate the minimum of a trilinear function is at
        one of the four other-coordinate corners. Its first/last possible
        zero therefore bounds the occupied projection on that axis. These
        analytic bounds avoid subdividing planar free slabs into tiny boxes.
        """
        lower, upper = node.lower.copy(), node.upper.copy()
        scale = float(np.max(np.abs(self._cell_values[node.cell])))
        error = _GAMMA * scale
        for axis in range(3):
            actual, _, rounding = self.cell_value(node.cell, box_corners(lower, upper))
            values = actual - rounding - error
            first = np.flatnonzero(CORNERS[:, axis] == 0)
            last = np.flatnonzero(CORNERS[:, axis] == 1)
            a, b = values[first], values[last]
            span = upper[axis] - lower[axis]
            if np.all(a > 0) and np.any(b <= 0):
                roots = a[b <= 0] / (a[b <= 0] - b[b <= 0])
                value = lower[axis] + span * float(roots.min())
                lower[axis] = max(lower[axis], np.nextafter(
                    value - _GAMMA * (abs(value) + span), -np.inf))
            elif np.all(b > 0) and np.any(a <= 0):
                roots = a[a <= 0] / (a[a <= 0] - b[a <= 0])
                value = lower[axis] + span * float(roots.max())
                upper[axis] = min(upper[axis], np.nextafter(
                    value + _GAMMA * (abs(value) + span), np.inf))
        return _Node(lower, upper, cell=node.cell)

    def occupied_points(self, node):
        """Return points belonging to the local cell set, including zero roots."""
        if node.cell < 0:
            return np.empty((0, 3))
        points = box_corners(node.lower, node.upper)
        values, _, errors = self.cell_value(node.cell, points)
        kept = list(points[values <= -errors])
        # Original samples with value zero are members of the closed set.
        # Their membership is input data, not a sign inferred from rounded
        # interpolation. This also preserves zero faces and all-zero cells.
        original = self.cell_origins[node.cell] + CORNERS * self.voxel_size
        original_inside = ((original >= node.lower) & (original <= node.upper)).all(-1)
        kept.extend(original[original_inside & (self._cell_values[node.cell] <= 0)])
        # A zero computed from a sign-changing edge is retained as a local
        # optimization seed. It is not used to claim strict interior overlap.
        for a, b in _EDGES:
            if values[a] * values[b] < 0:
                u = values[a] / (values[a] - values[b])
                point = points[a] + u * (points[b] - points[a])
                # Along this edge the interpolant is linear. Move toward
                # its negative endpoint by the computed arithmetic error,
                # then verify membership with the same evaluation bound.
                negative_index = a if values[a] < 0 else b
                negative = points[negative_index]
                v, _, e = self.cell_value(node.cell, point)
                fraction = min(1., max(64 * _EPS, 4 * (abs(float(v)) + float(e))
                                      / abs(float(values[negative_index]))))
                point = point + fraction * (negative - point)
                v, _, e = self.cell_value(node.cell, point)
                if v <= -e:
                    kept.append(point)
        return np.asarray(kept, dtype=float).reshape(-1, 3)

    def split(self, node):
        if node.children:
            return tuple(self.nodes[i] for i in node.children)
        axis = int(np.argmax(node.upper - node.lower))
        mid = (node.lower[axis] + node.upper[axis]) / 2
        if mid == node.lower[axis] or mid == node.upper[axis]:
            return ()
        high, low = node.upper.copy(), node.lower.copy()
        high[axis], low[axis] = mid, mid
        result = []
        for child in (_Node(node.lower, high, cell=node.cell),
                      _Node(low, node.upper, cell=node.cell)):
            values, _, errors = self.cell_value(
                child.cell, box_corners(child.lower, child.upper))
            if float(np.min(values - errors)) <= 0:
                result.append(self.tighten(child))
        return tuple(result)

    def local_zero(self, point):
        """Find a local zero-section point solely for an optimization direction."""
        # BEGIN query the surface candidate index, retaining original cell IDs.
        if self._zero_tree is None:
            return None
        _, indices = self._zero_tree.query(np.asarray(point), k=min(8, len(self._zero_cell_ids)))
        best = None
        for candidate in np.atleast_1d(indices):
            identity = int(self._zero_cell_ids[candidate])
            values = self._cell_values[identity]
            # END all candidates contain a supported nonconstant zero section.
            lower = self.cell_origins[identity]
            seeds = []
            for a, b in _EDGES:
                if values[a] * values[b] <= 0 and values[a] != values[b]:
                    seeds.append(CORNERS[a] + values[a] / (values[a] - values[b])
                                 * (CORNERS[b] - CORNERS[a]))
            if not seeds:
                continue
            target = (np.asarray(point) - lower) / self.voxel_size
            initial = min(seeds, key=lambda p: float(np.sum((p - target) ** 2)))
            result = minimize(lambda u: float(np.sum((u - target) ** 2)), initial,
                jac=lambda u: 2 * (u - target), bounds=[(0., 1.)] * 3, method="SLSQP",
                constraints={"type": "eq", "fun": lambda u: trilinear(values, u)[0],
                             "jac": lambda u: trilinear(values, u)[1]},
                options={"maxiter": 24, "ftol": 1e-12})
            if not result.success:
                continue
            zero = lower + result.x * self.voxel_size
            _, gradient, _ = self.cell_value(identity, zero)
            norm = float(np.linalg.norm(gradient))
            if not norm > 0:
                continue
            distance = float(np.linalg.norm(zero - point))
            if best is None or distance < best[0]:
                best = distance, zero, gradient / norm
        return best


def query_distance(first, first_pose, second, second_pose, *,
                   precision_m=1e-4, permission=None, check_request=None):
    """Check penetration between the two supported occupied sets.

    ``permission`` classifies world-space box pairs as ``permitted``,
    ``forbidden`` or ``mixed``. Only wholly permitted domains are skipped.
    A witness uses its arithmetic uncertainty box, so a mixed permission
    boundary cannot establish a forbidden contact. Distance intervals remain
    diagnostic; positive clearance is not a constraint.
    """
    # BEGIN one nonpenetration rule applies to optimization and acceptance.
    precision = float(precision_m)
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("Penetration queries need positive finite numerical precision")
    if first.root is None or second.root is None:
        return {"status": "clear", "lower_m": None, "upper_m": None, "empty_set": True,
                "intersection": False, "queries": 0, "witness_first": None,
                "witness_second": None, "scope": "represented_observed_occupancy",
                "distance_scope": "unpermitted_point_pairs_at_query_pose"}
    poses = (np.asarray(first_pose, float), np.asarray(second_pose, float))
    fields = (first, second)
    pending, serial = [], count()
    upper, witness = math.inf, None
    unresolved_lower = math.inf
    # BEGIN every separated branch contributes to the complete-domain bound.
    separated_lower = math.inf
    # END this is a distance record, not a positive-clearance requirement.
    visits = 0

    def push(a, b):
        nonlocal separated_lower
        if permission is not None and permission.excludes_cells(a.cell, b.cell):
            return
        wa, wb = world_bounds(a.lower, a.upper, poses[0]), world_bounds(b.lower, b.upper, poses[1])
        policy = "forbidden" if permission is None else permission.boxes(wa, wb)
        if policy == "permitted":
            return
        lower = box_distance(wa, wb)
        if lower > 0:
            separated_lower = min(separated_lower, lower)
        else:
            # BEGIN expose unpermitted domains before refining permission seams.
            # A mixed box contains both permitted and unpermitted points. FIFO
            # expansion exposes its interior children; choosing the smallest
            # mixed box first can starve those children along a boundary.
            diameter = float(np.linalg.norm(a.upper-a.lower)+np.linalg.norm(b.upper-b.lower))
            heapq.heappush(pending, (lower, int(policy == "mixed"),
                0. if policy == "mixed" else diameter, next(serial), a, b, wa, wb))
            # END only traversal order changes; every geometric bound is retained.

    push(first.nodes[first.root], second.nodes[second.root])
    while pending:
        lower, _, _, _, a, b, wa, wb = heapq.heappop(pending)
        visits += 1
        if check_request is not None and visits % 64 == 1:
            check_request()
        if lower > 0:
            continue
        if a.cell >= 0 and b.cell >= 0:
            # BEGIN zero-only boundary material cannot establish volume overlap.
            # Spatial bounds already failed to prove separation. Retain that
            # boundary uncertainty, then inspect the other cell pairs; a
            # lower-dimensional zero must not starve a volume intersection.
            if first._boundary_only_cells[a.cell] or second._boundary_only_cells[b.cell]:
                unresolved_lower = min(unresolved_lower, lower)
                continue
            # END constant-zero cells retain their full represented volume.
            domain_upper = math.inf
            local_a, local_b = first.occupied_points(a), second.occupied_points(b)
            pa, pb = transform_points(local_a, poses[0]), transform_points(local_b, poses[1])
            errors_a = transform_point_error(local_a, poses[0])
            errors_b = transform_point_error(local_b, poses[1])
            if len(pa) and len(pb):
                distances = np.linalg.norm(pa[:, None] - pb[None], axis=-1)
                for flat in np.argsort(distances, axis=None):
                    i, j = np.unravel_index(flat, distances.shape)
                    if permission is not None and permission.boxes(
                            (pa[i] - errors_a[i], pa[i] + errors_a[i]),
                            (pb[j] - errors_b[j], pb[j] + errors_b[j])) != "forbidden":
                        continue
                    value = float(distances[i, j])
                    value += float(np.linalg.norm(errors_a[i]) + np.linalg.norm(errors_b[j]))
                    value = float(np.nextafter(value + _GAMMA * (
                        value + np.abs(pa[i]).sum() + np.abs(pb[j]).sum()), np.inf))
                    domain_upper = min(domain_upper, value)
                    if value < upper:
                        upper, witness = value, (pa[i].copy(), pb[j].copy())
                    break
            # Interior points establish overlap; bounding boxes alone do not.
            overlap_low, overlap_high = np.maximum(wa[0], wb[0]), np.minimum(wa[1], wb[1])
            candidates = [*pa, *pb]
            if np.all(overlap_low <= overlap_high):
                candidates.append((overlap_low + overlap_high) / 2)
            for point in candidates:
                xa, xb = inverse_points(point, poses[0]), inverse_points(point, poses[1])
                xa_error, xb_error = (inverse_point_error(point, poses[0]),
                                      inverse_point_error(point, poses[1]))
                support_a, support_b = first.cell_origins[a.cell], second.cell_origins[b.cell]
                if (np.any(xa - xa_error <= support_a)
                        or np.any(xa + xa_error >= support_a + first.voxel_size)
                        or np.any(xb - xb_error <= support_b)
                        or np.any(xb + xb_error >= support_b + second.voxel_size)):
                    continue
                va, _, ea = first.cell_value(a.cell, xa, xa_error)
                vb, _, eb = second.cell_value(b.cell, xb, xb_error)
                # Strict support interiors distinguish a shared cell face
                # from volume overlap. Nonpositive corner data imply a full
                # occupied cell, including an identically zero interpolant.
                inside_a = va < -ea or np.all(first._cell_values[a.cell] <= 0)
                inside_b = vb < -eb or np.all(second._cell_values[b.cell] <= 0)
                if inside_a and inside_b:
                    if permission is not None and permission.boxes(
                            (point, point), (point, point)) != "forbidden":
                        continue
                    return {"status": "intersection", "lower_m": 0., "upper_m": 0.,
                            "intersection": True, "queries": visits,
                            "witness_first": point.tolist(), "witness_second": point.tolist(),
                            "scope": "represented_observed_occupancy",
                            "distance_scope": "unpermitted_point_pairs_at_query_pose"}
            if ((domain_upper <= precision
                 and np.min(overlap_high - overlap_low) <= precision)
                    or max(np.linalg.norm(a.upper - a.lower),
                           np.linalg.norm(b.upper - b.lower)) <= precision):
                # A thin possible overlap is unresolved at this numerical
                # precision, with actual points also that close. An incidental
                # thin overlap of loose boxes still needs refinement. Continue other
                # domains: a contact here must not hide penetration elsewhere.
                unresolved_lower = min(unresolved_lower, lower)
                continue
        axis = int(np.linalg.norm(b.upper - b.lower) > np.linalg.norm(a.upper - a.lower))
        selected = (a, b)[axis]
        children = fields[axis].split(selected)
        if not children:
            unresolved_lower = min(unresolved_lower, lower)
            continue
        for child in children:
            push(child, b) if axis == 0 else push(a, child)
    unresolved = np.isfinite(unresolved_lower)
    # BEGIN a clear result retains the minimum over all separated domains.
    # Fully permitted domains contribute no finite distance. If permissions
    # remove every domain, zero is a valid bound and does not mean contact.
    lower = unresolved_lower if unresolved else separated_lower
    lower = float(lower) if np.isfinite(lower) else 0.
    # END finite permission regions remain specific to this query's pose.
    return {"status": "unresolved" if unresolved else "clear",
            "lower_m": lower,
            "upper_m": upper if np.isfinite(upper) else None,
            "intersection": None if unresolved else False,
            "queries": visits,
            "witness_first": None if witness is None else witness[0].tolist(),
            "witness_second": None if witness is None else witness[1].tolist(),
            "scope": "represented_observed_occupancy",
            "distance_scope": "unpermitted_point_pairs_at_query_pose"}
    # END unresolved domains never become positive collision evidence.
# END every definitive result refers to the same supported implicit sets.
