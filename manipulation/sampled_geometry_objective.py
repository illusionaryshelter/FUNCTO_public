"""A fixed finite optimization objective over original represented geometry.

Material coordinates are fixed when the objective is constructed. Each trial
evaluates the original trilinear field or the registered native shape at those
coordinates after the declared rigid transforms. The derivative is the
derivative of that same scalar wherever its active support, permission and
nearest-feature branches are unchanged. Original set queries own acceptance.
"""

from itertools import product

import coal
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ._coal_scene_batch import native_feasible_point_bounds
# BEGIN reuse the existing coordinate round-off bound for material selection.
from .implicit_distance import (CORNERS, ObservedField, _EDGES, _GAMMA,
                                transform_point_error, world_bounds)
# END no separate geometry representation or numerical tolerance is introduced.
from .native_field_distance import _local_interval, _native_bounds, _transform
from .planning_problem import quaternion_conjugate, quaternion_rotate


def _world(points, pose):
    """Transform fixed entity-frame material points into the reference frame."""
    local = torch.as_tensor(points, dtype=pose[0].dtype, device=pose[0].device)
    return pose[0] + quaternion_rotate(pose[1].expand(len(local), 4), local)


def _local(world, pose):
    return quaternion_rotate(quaternion_conjugate(pose[1]).expand(len(world), 4), world - pose[0])


def _numeric_points(points, source_pose, target_pose):
    """Evaluate coordinate changes before retaining only active derivatives.

    Dense observed material arrays stay in NumPy. Autograd later stores only
    negative active samples, whose selected coordinate expressions determine
    the exact same objective derivative away from branch changes.
    """
    source_position, source_quaternion = [item.detach().cpu().numpy() for item in source_pose]
    target_position, target_quaternion = [item.detach().cpu().numpy() for item in target_pose]
    source_rotation = Rotation.from_quat(source_quaternion[[1, 2, 3, 0]]).as_matrix()
    target_rotation = Rotation.from_quat(target_quaternion[[1, 2, 3, 0]]).as_matrix()
    world = points @ source_rotation.T + source_position
    return world, (world - target_position) @ target_rotation


def _allowed(world, permission):
    """Apply the existing finite point-pair domain to an overlap point.

    A sample contributes only where the same reference-frame point belongs
    to both represented entities. Thus both endpoints of a permitted pair
    coincide; the two original finite regions are still tested separately.
    Branch derivatives are zero in the interior of each permission region.
    """
    allowed = np.zeros(len(world), dtype=bool)
    if permission is not None:
        for first, second in permission.regions:
            allowed |= (np.all(world @ np.asarray(first[0]).T <= first[1], axis=-1)
                        & np.all(world @ np.asarray(second[0]).T <= second[1], axis=-1))
    return allowed


class MaterialSamples:
    """Fixed original occupied material and zero-edge points of one entity.

    Observed sample cell identities are retained, including duplicate points
    shared by adjacent cells, so an initial-support declaration removes only
    its selected original cells. Native points are constructed by the existing
    feasible-point API from the local projection box's 3-by-3-by-3 grid. The
    box only proposes coordinates; it never supplies collision material.
    """

    def __init__(self, geometry):
        self.geometry = geometry
        # BEGIN index immutable original cell ranges without changing material identity.
        self.cell_ranges = self.cell_radius = None
        if isinstance(geometry, ObservedField):
            self.cell_ranges = np.empty((len(geometry.cells), 2), dtype=np.int64)
            self.cell_radius = np.zeros(3)
            parts, cells, offset = [], [], 0
            for node in geometry.nodes:
                if node.cell < 0:
                    continue
                points = geometry.occupied_points(node)
                self.cell_ranges[node.cell] = offset, offset + len(points)
                offset += len(points)
                parts.append(points)
                cells.append(np.full(len(points), node.cell, dtype=np.int64))
            self.points = np.concatenate(parts) if parts else np.empty((0, 3))
            self.cells = np.concatenate(cells) if cells else np.empty(0, dtype=np.int64)
            del parts, cells
            # The tree and material arrays store rounded coordinates. A
            # nominal voxel_size/2 can exclude an actual closed endpoint.
            # Bound the stored subtraction in fixed-size NumPy batches, as
            # in the field's original leaf construction, without one array
            # operation per cell or another material-sized retained buffer.
            for start in range(0, len(self.points), 1024):
                points = self.points[start:start + 1024]
                centers = geometry.tree.data[self.cells[start:start + 1024]]
                radius = np.nextafter(np.abs(points - centers), np.inf).max(0)
                self.cell_radius = np.maximum(self.cell_radius, radius)
            self.cell_ranges.flags.writeable = self.cell_radius.flags.writeable = False
            # END leaf order, duplicate coordinates and cell identities are retained.
            self.length_scale = geometry.voxel_size
            # Original zero vertices and analytic edge roots are objective
            # samples, not a reconstructed surface or an occupancy proxy.
            zero_points = []
            for origin, values in zip(geometry.cell_origins, geometry._cell_values):
                zero_points.extend(origin + CORNERS[values == 0] * geometry.voxel_size)
                for first, second in _EDGES:
                    if values[first] * values[second] < 0:
                        fraction = values[first] / (values[first] - values[second])
                        zero_points.append(origin + geometry.voxel_size * (
                            CORNERS[first] + fraction * (CORNERS[second] - CORNERS[first])))
            self.zero_points = np.unique(np.asarray(zero_points).reshape(-1, 3), axis=0)
            self.native_bounds = None
        else:
            identity = _transform(np.eye(4))
            self.native_bounds = _native_bounds(geometry, identity)
            proposals = np.asarray(list(product(*[np.linspace(low, high, 3)
                for low, high in self.native_bounds.T])))
            points = [np.mean(np.asarray(native_feasible_point_bounds(
                geometry, identity, point)["local_bounds"]), axis=0) for point in proposals]
            self.points = np.unique(np.asarray(points), axis=0)
            self.cells = np.full(len(self.points), -1, dtype=np.int64)
            self.zero_points = np.empty((0, 3))
            self.length_scale = float(np.linalg.norm(self.native_bounds[1] - self.native_bounds[0]))
        self.points.flags.writeable = self.cells.flags.writeable = self.zero_points.flags.writeable = False
        # BEGIN immutable ranges bound the existing objective, not occupancy tests.
        # point_bounds encloses exactly the retained source coordinates. For
        # a field, a negative trilinear value requires a cell with at least
        # one nonpositive vertex. The original root encloses every such full
        # supported cell, including cells whose zero part has no volume.
        # Account for the same field lookup's subtraction/division rounding;
        # this numerical enclosure is unrelated to physical clearance.
        self.point_bounds = (np.stack((self.points.min(0), self.points.max(0)))
                             if len(self.points) else None)
        if isinstance(geometry, ObservedField):
            if geometry.root is None:
                self.negative_bounds = None
            else:
                root = geometry.nodes[geometry.root]
                bounds = np.stack((root.lower, root.upper))
                error = _GAMMA * (np.abs(bounds) + np.abs(geometry.origin)
                                 + np.abs(bounds - geometry.origin) + geometry.voxel_size)
                self.negative_bounds = np.stack((np.nextafter(bounds[0]-error[0], -np.inf),
                                                np.nextafter(bounds[1]+error[1], np.inf)))
        else:
            # These are the existing outward native support projections,
            # enclosing every original-shape point with negative distance.
            self.negative_bounds = self.native_bounds
        for bounds in (self.point_bounds, self.negative_bounds):
            if bounds is not None:
                bounds.flags.writeable = False
        # END the fixed sample count, coordinates and normalization are unchanged.

    # BEGIN exact closed-box queries reuse the original field's centre tree.
    def indices_in_box(self, bounds):
        """Select original point indices whose cells can meet a local closed box.

        cKDTree returns original cell IDs. A containing infinity-norm ball
        proposes cells, an axis comparison retains the closed query box, and
        the stored start offsets restore the original material summation order.
        The query radius includes only coordinate arithmetic, not clearance.
        """
        if self.geometry.tree is None:
            return np.empty(0, dtype=np.int64)
        low = np.nextafter(np.asarray(bounds[0]) - self.cell_radius, -np.inf)
        high = np.nextafter(np.asarray(bounds[1]) + self.cell_radius, np.inf)
        center = low + (high - low) / 2
        radius = max(np.max(center - low), np.max(high - center))
        error = _GAMMA * (np.max(np.abs(center)) + np.max(np.abs(low)) + np.max(np.abs(high)))
        radius = np.nextafter(radius + error, np.inf)
        cells = np.asarray(self.geometry.tree.query_ball_point(
            center, radius, p=np.inf, eps=0., workers=1, return_sorted=False), dtype=np.int64)
        centers = self.geometry.tree.data[cells]
        cells = cells[np.all((centers >= low) & (centers <= high), axis=1)]
        del centers
        cells = cells[np.argsort(self.cell_ranges[cells, 0], kind="stable")]
        ranges = self.cell_ranges[cells]
        lengths = ranges[:, 1] - ranges[:, 0]
        offsets = np.cumsum(lengths) - lengths
        indices = np.arange(lengths.sum(), dtype=np.int64)
        indices += np.repeat(ranges[:, 0] - offsets, lengths)
        return indices
    # END selection preserves every original record, including duplicate points.


# BEGIN enclose the preimage of the actually evaluated floating coordinates.
def _material_indices(source, source_pose, target, target_pose):
    """Retain every source sample that can have negative target distance.

    With column-vector transforms (A,a) and (B,b), _numeric_points evaluates
    w_hat=fl(A*x+a), y_hat=fl(B.T*(w_hat-b)). Its negative target box is first
    expanded by both forward errors. Mapping that box through B rather than
    B**(-T) requires the finite orthogonality defect below; the final existing
    source inverse interval also encloses A's inverse defect. All quantities
    are numerical length bounds, independent of any task or physical margin.
    """
    if not isinstance(source.geometry, ObservedField):
        return np.arange(len(source.points), dtype=np.int64)
    if source.point_bounds is None or target.negative_bounds is None:
        return np.empty(0, dtype=np.int64)
    matrices = []
    for position, quaternion in (source_pose, target_pose):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(
            quaternion.detach().cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
        matrix[:3, 3] = position.detach().cpu().numpy()
        matrices.append(matrix)
    source_matrix, target_matrix = matrices
    rotation, translation = target_matrix[:3, :3], target_matrix[:3, 3]
    absolute_rotation = np.abs(rotation)
    source_size = np.max(np.abs(source.point_bounds), axis=0)
    forward_error = transform_point_error(source_size, source_matrix)
    world = world_bounds(*source.point_bounds, source_matrix)
    world_size = np.max(np.abs(world), axis=0)
    error = (_GAMMA * ((world_size + np.abs(translation)) @ absolute_rotation)
             + forward_error @ absolute_rotation)
    # The final inflation encloses arithmetic used to evaluate the positive
    # error expressions themselves, in addition to the original transforms.
    error = np.nextafter(error * (1 + _GAMMA), np.inf)
    target_bounds = (np.nextafter(target.negative_bounds[0] - error, -np.inf),
                     np.nextafter(target.negative_bounds[1] + error, np.inf))
    defect = (np.linalg.norm(rotation.T @ rotation - np.eye(3), ord=np.inf)
              + _GAMMA * np.linalg.norm(absolute_rotation.T @ absolute_rotation, ord=np.inf))
    defect = np.nextafter(defect * (1 + _GAMMA), np.inf)
    # Rotation.from_quat supplies a rigid rotation, hence defect < 1.
    # B**(-T)-B = B*((B.T*B)**(-1)-I); multiply its infinity-norm
    # bound by the target-box coordinate magnitude to obtain a length.
    inverse_error = (np.linalg.norm(rotation, ord=np.inf) * defect / (1 - defect)
                     * np.max(np.abs(target_bounds)))
    inverse_error = np.nextafter(inverse_error * (1 + _GAMMA), np.inf)
    world = world_bounds(*target_bounds, target_matrix)
    world = (np.nextafter(world[0] - inverse_error, -np.inf),
             np.nextafter(world[1] + inverse_error, np.inf))
    return source.indices_in_box(_local_interval(world, source_matrix))
# END one immutable index and one selection rule precede the original queries.


# BEGIN whole-pair zero tests use the existing coordinate interval arithmetic.
def _pair_separated(first, first_pose, second, second_pose):
    """Prove that both finite negative-part sums and their pose gradients vanish.

    For each direction, transform the source sample box into the target
    coordinates, enclosing forward arithmetic and the rigid inverse's
    arithmetic/orthogonality error. Strict box separation then excludes
    every source sample from the target's possible negative-value region.
    Both directions must pass. Finite permissions only remove contributions,
    so the same implication holds for any region or original-cell exclusion.
    Overlapping or touching boxes supply no conclusion and run the full sum.
    """
    matrices = []
    for position, quaternion in (first_pose, second_pose):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(
            quaternion.detach().cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
        matrix[:3, 3] = position.detach().cpu().numpy()
        matrices.append(matrix)
    for source, source_matrix, target, target_matrix in (
            (first, matrices[0], second, matrices[1]),
            (second, matrices[1], first, matrices[0])):
        if source.point_bounds is None or target.negative_bounds is None:
            continue
        relative = _local_interval(world_bounds(*source.point_bounds, source_matrix), target_matrix)
        if not (np.any(relative[1] < target.negative_bounds[0])
                or np.any(target.negative_bounds[1] < relative[0])):
            return False
    return True
# END boxes prove zero objective contributions without classifying collisions.


def _field_direction(source, source_pose, target, target_pose, permission, *, reverse, report):
    """Mean squared supported TSDF violation with its exact spatial gradient."""
    zero = source_pose[0].sum() * 0 + target_pose[0].sum() * 0
    if len(source.points) == 0:
        return zero
    # BEGIN query only selected original records; normalize by the complete set.
    indices = _material_indices(source, source_pose, target, target_pose)
    if not len(indices):
        return zero
    world, relative = _numeric_points(source.points[indices], source_pose, target_pose)
    excluded = () if permission is None or reverse else permission.excluded_second_cells
    values, supported, gradients, _ = target.geometry.sample(
        relative, excluded_cells=excluded)
    active = supported & ~_allowed(world, permission)
    if reverse and permission is not None and permission.excluded_second_cells:
        active &= ~np.isin(source.cells[indices], tuple(permission.excluded_second_cells))
    report["supported_sample_count"] += int(np.count_nonzero(supported))
    report["unsupported_sample_count"] += int(np.count_nonzero(~supported))
    report["active_sample_count"] += int(np.count_nonzero(active))
    selected = np.flatnonzero(active & (values < 0))
    if not len(selected):
        return zero
    # This reconnects the analytic derivative of the actually evaluated
    # trilinear scalar. It is not a frozen tangent evaluated as a trial value.
    coordinates = _local(_world(source.points[indices[selected]], source_pose), target_pose)
    gradient = coordinates.new_tensor(gradients[selected])
    signed = coordinates.new_tensor(values[selected]) + (
        (coordinates - coordinates.detach()) * gradient).sum(-1)
    report["negative_sample_count"] += int(np.count_nonzero(values[selected] < 0))
    report["zero_gradient_negative_count"] += int(np.count_nonzero(
        (values[selected] < 0) & np.all(gradients[selected] == 0, axis=-1)))
    return torch.relu(-signed / target.length_scale).square().sum() / len(source.points)
    # END permissions and diagnostic counters apply to the actual original queries.


def _native_direction(source, source_pose, target, target_pose, permission, *, report):
    """Query original observed material against the unchanged native entity.

    The zero-radius sphere denotes a mathematical query point. It does not
    approximate either entity. Box pruning removes only samples with a
    nonnegative native distance, whose squared negative-part cost is zero.
    Coal owns the signed value and nearest-feature normal in native coordinates.
    """
    zero = source_pose[0].sum() * 0 + target_pose[0].sum() * 0
    if len(source.points) == 0:
        return zero
    # BEGIN the same index selects original observed material for native queries.
    indices = _material_indices(source, source_pose, target, target_pose)
    if not len(indices):
        return zero
    world, numeric = _numeric_points(source.points[indices], source_pose, target_pose)
    active = np.all((numeric >= target.native_bounds[0]) & (numeric <= target.native_bounds[1]), axis=-1)
    active &= ~_allowed(world, permission)
    if permission is not None and permission.excluded_second_cells:
        active &= ~np.isin(source.cells[indices], tuple(permission.excluded_second_cells))
    selected = np.flatnonzero(active)
    report["native_point_query_count"] += len(selected)
    if not len(selected):
        return zero
    request = coal.DistanceRequest()
    request.enable_signed_distance = True
    point, identity = coal.Sphere(0.), _transform(np.eye(4))
    values, gradients = [], []
    for index in selected:
        result = coal.DistanceResult()
        distance = coal.distance(target.geometry, identity, point,
            coal.Transform3s(np.eye(3), numeric[index]), request, result)
        values.append(distance)
        gradients.append(np.asarray(result.normal, dtype=float))
    coordinates = _local(_world(source.points[indices[selected]], source_pose), target_pose)
    signed = coordinates.new_tensor(values) + ((coordinates - coordinates.detach())
        * coordinates.new_tensor(np.asarray(gradients))).sum(-1)
    report["negative_native_sample_count"] += int(np.count_nonzero(np.asarray(values) < 0))
    # The observed field's existing metric resolution normalizes both
    # directions of this mixed pair; no task-specific scale is introduced.
    return torch.relu(-signed / source.length_scale).square().sum() / len(source.points)
    # END the original native distance, finite permission and full denominator remain.


def _pair_value(first, first_pose, second, second_pose, *, permission, report):
    """Evaluate both finite directions before reducing their pose derivative."""
    report = {} if report is None else report
    for key in ("supported_sample_count", "unsupported_sample_count", "active_sample_count",
                "negative_sample_count", "zero_gradient_negative_count", "native_point_query_count",
                "negative_native_sample_count"):
        report.setdefault(key, 0)
    forward = _field_direction(first, first_pose, second, second_pose, permission,
                               reverse=False, report=report)
    backward = (_field_direction(second, second_pose, first, first_pose, permission,
                                reverse=True, report=report)
                if isinstance(first.geometry, ObservedField) else
                _native_direction(second, second_pose, first, first_pose, permission, report=report))
    return (forward + backward) / 2


def pair_objective(first, first_pose, second, second_pose, *, permission=None, report=None):
    """Return the actual finite value and its first derivative to both poses.

    One temporary point graph is reduced to fourteen pose derivatives before
    processing the next pair. The caller's trajectory graph therefore stores
    positions/quaternions per pair rather than material points per trajectory
    state. SLSQP consumes this first derivative; a second derivative through
    the detached spatial/nearest-feature query is outside this API's scope.
    """
    report = {} if report is None else report
    coordinates = torch.cat((*first_pose, *second_pose))
    # BEGIN skip only two proven-zero finite sums, preserving all pose links.
    report.setdefault("zero_contribution_separated_pair_count", 0)
    if _pair_separated(first, first_pose, second, second_pose):
        report["zero_contribution_separated_pair_count"] += 1
        for key in ("supported_sample_count", "unsupported_sample_count", "active_sample_count",
                    "negative_sample_count", "zero_gradient_negative_count", "native_point_query_count",
                    "negative_native_sample_count"):
            report.setdefault(key, 0)
        return coordinates.sum() * 0
    # END support/unknown counts describe only material points actually queried.
    if not torch.is_grad_enabled() or not coordinates.requires_grad:
        return _pair_value(first, first_pose, second, second_pose,
                           permission=permission, report=report)
    with torch.enable_grad():
        local = coordinates.detach().requires_grad_(True)
        value = _pair_value(first, (local[:3], local[3:7]), second,
            (local[7:10], local[10:14]), permission=permission, report=report) + local.sum() * 0
        derivative = torch.autograd.grad(value, local)[0]
    return value.detach() + ((coordinates - coordinates.detach()) * derivative).sum()


def contact_objective(samples, normals, bounds):
    """Minimum region violation, retaining only the selected branch's graph."""
    if not len(samples.zero_points):
        return normals.sum() * 0 + bounds.sum() * 0
    numeric = np.maximum(samples.zero_points @ normals.detach().cpu().numpy().T
                         - bounds.detach().cpu().numpy(), 0.)
    selected = int(np.argmin(np.square(numeric).sum(-1)))
    point = normals.new_tensor(samples.zero_points[selected])
    violation = torch.relu(point @ normals.T - bounds)
    return violation.square().sum() / samples.length_scale ** 2


def native_pair_objective(first, first_pose, first_matrix, second, second_pose, second_matrix):
    """Squared signed native penetration with the same nearest-point derivative.

    At a unique smooth closest feature, the signed-distance derivative is
    n.T @ (J_second - J_first). Coal supplies the signed scalar and witnesses
    at this trial; rotations and translations of those material witnesses
    remain connected to the supplied differentiable robot link poses.
    """
    from .implicit_distance import inverse_points
    request, result = coal.DistanceRequest(), coal.DistanceResult()
    request.enable_signed_distance = True
    distance = coal.distance(first.geometry, _transform(first_matrix), second.geometry,
                             _transform(second_matrix), request, result)
    # BEGIN the squared negative part has zero value and derivative for d >= 0.
    # The scene adds one global zero connection to all planning controls.
    # Nonnegative native pairs therefore need neither witness reads nor a
    # separate zero-valued rotation graph; the original distance still runs.
    if distance >= 0:
        return first_pose[0].new_zeros(())
    # END negative distances retain their original witness derivative below.
    first_point = inverse_points(np.asarray(result.getNearestPoint1()), first_matrix)
    second_point = inverse_points(np.asarray(result.getNearestPoint2()), second_matrix)
    relative = _world(second_point[None], second_pose)[0] - _world(first_point[None], first_pose)[0]
    derivative = (relative * relative.new_tensor(np.asarray(result.normal))).sum()
    signed = derivative.new_tensor(distance) + derivative - derivative.detach()
    length = (first.length_scale + second.length_scale) / 2
    return torch.relu(-signed / length).square()
