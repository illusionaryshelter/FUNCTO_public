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
from .implicit_distance import CORNERS, ObservedField, _EDGES, _GAMMA, world_bounds
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
        if isinstance(geometry, ObservedField):
            leaves = [node for node in geometry.nodes if node.cell >= 0]
            self.points = (np.concatenate([geometry.occupied_points(node) for node in leaves])
                           if leaves else np.empty((0, 3)))
            self.cells = (np.concatenate([np.full(len(geometry.occupied_points(node)), node.cell,
                                                 dtype=np.int64) for node in leaves])
                          if leaves else np.empty(0, dtype=np.int64))
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
    world, relative = _numeric_points(source.points, source_pose, target_pose)
    excluded = () if permission is None or reverse else permission.excluded_second_cells
    values, supported, gradients, _ = target.geometry.sample(
        relative, excluded_cells=excluded)
    active = supported & ~_allowed(world, permission)
    if reverse and permission is not None and permission.excluded_second_cells:
        active &= ~np.isin(source.cells, tuple(permission.excluded_second_cells))
    report["supported_sample_count"] += int(np.count_nonzero(supported))
    report["unsupported_sample_count"] += int(np.count_nonzero(~supported))
    report["active_sample_count"] += int(np.count_nonzero(active))
    selected = np.flatnonzero(active & (values < 0))
    if not len(selected):
        return zero
    # This reconnects the analytic derivative of the actually evaluated
    # trilinear scalar. It is not a frozen tangent evaluated as a trial value.
    coordinates = _local(_world(source.points[selected], source_pose), target_pose)
    gradient = coordinates.new_tensor(gradients[selected])
    signed = coordinates.new_tensor(values[selected]) + (
        (coordinates - coordinates.detach()) * gradient).sum(-1)
    report["negative_sample_count"] += int(np.count_nonzero(values[selected] < 0))
    report["zero_gradient_negative_count"] += int(np.count_nonzero(
        (values[selected] < 0) & np.all(gradients[selected] == 0, axis=-1)))
    return torch.relu(-signed / target.length_scale).square().sum() / len(source.points)


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
    world, numeric = _numeric_points(source.points, source_pose, target_pose)
    active = np.all((numeric >= target.native_bounds[0]) & (numeric <= target.native_bounds[1]), axis=-1)
    active &= ~_allowed(world, permission)
    if permission is not None and permission.excluded_second_cells:
        active &= ~np.isin(source.cells, tuple(permission.excluded_second_cells))
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
    coordinates = _local(_world(source.points[selected], source_pose), target_pose)
    signed = coordinates.new_tensor(values) + ((coordinates - coordinates.detach())
        * coordinates.new_tensor(np.asarray(gradients))).sum(-1)
    report["negative_native_sample_count"] += int(np.count_nonzero(np.asarray(values) < 0))
    # The observed field's existing metric resolution normalizes both
    # directions of this mixed pair; no task-specific scale is introduced.
    return torch.relu(-signed / source.length_scale).square().sum() / len(source.points)


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
    first_point = inverse_points(np.asarray(result.getNearestPoint1()), first_matrix)
    second_point = inverse_points(np.asarray(result.getNearestPoint2()), second_matrix)
    relative = _world(second_point[None], second_pose)[0] - _world(first_point[None], first_pose)[0]
    derivative = (relative * relative.new_tensor(np.asarray(result.normal))).sum()
    signed = derivative.new_tensor(distance) + derivative - derivative.detach()
    length = (first.length_scale + second.length_scale) / 2
    return torch.relu(-signed / length).square()
