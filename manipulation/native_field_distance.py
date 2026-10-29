"""Distance from a known convex robot entity to a supported observed set.

The robot keeps its registered Coal geometry. The task object keeps its
original trilinear cells. Spatial boxes bound the search only: their overlap
never establishes intersection of these two represented sets.
"""

# BEGIN mixed model/observation query; neither entity is converted to the other.
import heapq
from itertools import count
from functools import lru_cache

import coal
import numpy as np

from ._coal_scene_batch import native_projection_bounds, native_feasible_point_bounds
# BEGIN projection arithmetic shares the field-query numerical implementation.
from ._geometry_kernels import norm3, projection_lower
# END Coal continues to supply directions and complete native support bounds.
from .implicit_distance import (
    _GAMMA, box_corners, box_distance, inverse_points, inverse_point_error,
    transform_points, transform_point_error, world_bounds as _world_bounds,
)


# BEGIN complete numerical inputs identify the same outward-rounded transform.
@lru_cache(maxsize=65536)
def _prepared_world_bounds(lower_key, upper_key, pose_key):
    arrays = [np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)
              for dtype, shape, raw in (lower_key, upper_key, pose_key)]
    result = _world_bounds(*arrays)
    for value in result:
        value.flags.writeable = False
    return result


def world_bounds(lower, upper, pose):
    """Reuse a pure box transform by dtype, shape and every input byte."""
    return _prepared_world_bounds(*((value.dtype.str, value.shape, value.tobytes())
                                   for value in (lower, upper, pose)))
# END field distances, permissions and classifications remain actual queries.


def _transform(matrix):
    return coal.Transform3s(matrix[:3, :3], matrix[:3, 3])


def _native_bounds(geometry, transform):
    intervals = [native_projection_bounds(geometry, transform, axis)
                 for axis in np.eye(3)]
    return np.asarray(intervals, dtype=float).T



def _projection_gap(geometry, transform, box):
    """Certify a lower bound using a direction proposed by Coal.

    GJK's tolerance is not a distance error bound. Only the native support
    interval and the box's outward projection determine this lower bound.
    A poor direction merely gives zero and leaves the domain to refinement.
    """
    center = (box[0] + box[1]) / 2
    request, result = coal.DistanceRequest(), coal.DistanceResult()
    coal.distance(geometry, transform, coal.Box(*(box[1] - box[0])),
                  coal.Transform3s(np.eye(3), center), request, result)
    direction = np.asarray(result.normal, dtype=float)
    if not np.isfinite(direction).all():
        raise RuntimeError("Coal supplied a nonfinite native/box search direction")
    # BEGIN compile the original sqrt(dot) and outward projection calculation.
    length = norm3(direction[0], direction[1], direction[2])
    if length == 0:
        return 0.
    native_low, native_high = native_projection_bounds(geometry, transform, direction)
    return projection_lower(box[0], box[1], direction, native_low, native_high, length)
    # END the original full native/box projection supplies the same lower bound.


def _local_interval(world, pose):
    corners = box_corners(*world)
    local, error = inverse_points(corners, pose), inverse_point_error(corners, pose)
    return (np.nextafter((local - error).min(0), -np.inf),
            np.nextafter((local + error).max(0), np.inf))


def query_native_field(geometry, native_pose, field, field_pose, *,
                       precision_m=1e-4, permission=None, check_request=None,
                       native_bounds=None):
    """Check penetration over unpermitted native and observed point pairs.

    A finite contact permission constrains both endpoints. When its boundary
    cuts the native shape, a local search box B denotes K intersect B, where K
    is the unchanged native shape. B is never accepted as robot material.
    Feasible native witnesses retain interval enclosures of the constructed
    point. Their midpoint is used only for display and a local derivative.

    ``native_bounds`` optionally supplies the same entity's (local, world)
    projection bounds. The scene owns local bounds with its immutable model
    and world bounds only for this query pose. A standalone query computes
    these bounds directly. Returned distance bounds concern unpermitted point
    pairs at this pose; changing a contact permission changes that domain.
    ``full_entity_lower_m`` independently bounds complete-entity separation.
    With a permission it uses the unchanged entities' whole-set enclosures;
    without one the complete query can supply a tighter bound.
    """
    # BEGIN positive clearance is not part of represented-entity collision.
    precision = float(precision_m)
    if not np.isfinite(precision) or precision <= 0:
        raise ValueError("Penetration queries need positive finite numerical precision")
    native_pose, field_pose = np.asarray(native_pose, float), np.asarray(field_pose, float)
    transform = _transform(native_pose)
    pending, serial = [], count()
    upper, witness, visits = np.inf, None, 0
    unresolved_lower = np.inf
    # BEGIN retain bounds from every separated search domain, not one leaf.
    separated_lower = np.inf
    # END permitted or empty domains do not contribute a finite distance.
    full_entity_lower = None

    def report(status, lower, *, intersection=None, empty=False):
        output = dict(status=status, lower_m=lower,
            # BEGIN complete-entity evidence has its own unambiguous domain.
            full_entity_lower_m=lower if permission is None else full_entity_lower,
            # END possible enclosure overlap is never collision evidence.
            upper_m=None if not np.isfinite(upper) else float(upper),
            intersection=intersection, queries=visits,
            witness_first=None, witness_second=None,
            scope="known_native_and_represented_observed_set",
            distance_scope="unpermitted_point_pairs_at_query_pose")
        if witness is not None:
            a, b = witness
            output.update(witness_first=((a[0] + a[1]) / 2).tolist(),
                witness_second=((b[0] + b[1]) / 2).tolist(),
                witness_first_bounds=np.asarray(a).tolist(),
                witness_second_bounds=np.asarray(b).tolist(),
                witness_coordinates="interval_midpoints_for_display_and_derivative")
        if empty:
            output["empty_set"] = True
        return output

    # BEGIN empty observations need no shape support queries.
    if field.root is None:
        return report("clear", None, intersection=False, empty=True)
    if native_bounds is None:
        native_local = _native_bounds(geometry, _transform(np.eye(4)))
        native_world = _native_bounds(geometry, transform)
    else:
        native_local, native_world = native_bounds
    # END reuse concerns only geometry bounds, never query classification.
    # BEGIN retain the same whole-field range for the root query and proof.
    root = field.nodes[field.root]
    field_world = world_bounds(root.lower, root.upper, field_pose)
    full_entity_lower = box_distance(native_world, field_world)
    # END this independent bound survives finite permission clipping.

    def push(native_box, node, partitioned=False):
        nonlocal separated_lower
        if permission is not None and permission.excludes_cells(-1, node.cell):
            return
        wa = native_world if not partitioned else world_bounds(*native_box, native_pose)
        wb = field_world if node is root else world_bounds(node.lower, node.upper, field_pose)
        policy = "forbidden" if permission is None else permission.boxes(wa, wb)
        if policy == "permitted":
            return
        lower = full_entity_lower if node is root and not partitioned else box_distance(wa, wb)
        if lower > 0:
            separated_lower = min(separated_lower, lower)
            return
        lower = max(lower, _projection_gap(geometry, transform, wb))
        if lower > 0:
            separated_lower = min(separated_lower, lower)
            return
        # K intersect B is empty if the full native set is separated from B.
        # The converse is deliberately not asserted.
        if partitioned and _projection_gap(geometry, transform, wa) > 0:
            return
        diameter = float(np.linalg.norm(node.upper - node.lower))
        heapq.heappush(pending, (lower, int(policy == "mixed"),
            0. if policy == "mixed" else diameter, next(serial),
            native_box, node, wa, wb, policy, partitioned))

    push(native_local, root)
    while pending:
        lower, _, _, _, native_box, node, wa, wb, policy, partitioned = heapq.heappop(pending)
        visits += 1
        if check_request is not None and visits % 64 == 1:
            check_request()
        if node.cell >= 0:
            # BEGIN a supported nonconstant nonnegative cell has only boundary occupancy.
            # It remains available to surface queries. This volume query
            # records possible boundary contact and continues other domains.
            if field._boundary_only_cells[node.cell]:
                unresolved_lower = min(unresolved_lower, lower)
                continue
            # END an identically zero cell still represents occupied volume.
            domain_upper = np.inf
            local_points = field.occupied_points(node)
            points = transform_points(local_points, field_pose)
            errors = transform_point_error(local_points, field_pose)
            # The box midpoint also proposes a feasible native point inside a
            # negative cell. It supplies no independent membership evidence.
            proposals = [*points, (np.maximum(wa[0], wb[0]) + np.minimum(wa[1], wb[1])) / 2]
            for index, wanted in enumerate(proposals):
                feasible = native_feasible_point_bounds(geometry, transform, np.asarray(wanted, float))
                local, world = (np.asarray(feasible[key], float)
                                for key in ("local_bounds", "world_bounds"))
                if partitioned and (np.any(local[0] < native_box[0])
                                    or np.any(local[1] > native_box[1])):
                    continue
                field_local = _local_interval(world, field_pose)
                support = field.cell_origins[node.cell]
                if (np.all(field_local[0] > support)
                        and np.all(field_local[1] < support + field.voxel_size)):
                    center = (field_local[0] + field_local[1]) / 2
                    radius = np.nextafter(
                        np.maximum(center - field_local[0], field_local[1] - center), np.inf)
                    value, _, error = field.cell_value(node.cell, center, radius)
                    forbidden = permission is None or permission.boxes(world, world) == "forbidden"
                    # An occupied open neighborhood of an actual native
                    # point proves volume overlap for a solid native entity.
                    # Nonpositive corners also cover a constant zero cell.
                    inside = value < -error or np.all(field._cell_values[node.cell] <= 0)
                    if inside and forbidden:
                        upper, witness = 0., (world, world)
                        return report("intersection", 0., intersection=True)
                if index == len(points):
                    continue
                target = (np.nextafter(points[index] - errors[index], -np.inf),
                          np.nextafter(points[index] + errors[index], np.inf))
                if permission is not None and permission.boxes(world, target) != "forbidden":
                    continue
                delta = np.maximum(np.abs(world[0] - target[1]), np.abs(world[1] - target[0]))
                distance = float(np.linalg.norm(delta))
                error = _GAMMA * (distance + np.abs(world).sum() + np.abs(target).sum())
                candidate_upper = float(np.nextafter(distance + error, np.inf))
                domain_upper = min(domain_upper, candidate_upper)
                if candidate_upper < upper:
                    upper, witness = candidate_upper, (world, target)
            diameter = float(np.linalg.norm(node.upper - node.lower))
            native_diameter = float(np.linalg.norm(native_box[1] - native_box[0]))
            overlap = np.minimum(wa[1], wb[1]) - np.maximum(wa[0], wb[0])
            if ((domain_upper <= precision and np.min(overlap) <= precision)
                    or (diameter <= precision and (policy != "mixed" or native_diameter <= precision))):
                # Only this domain is unresolved. Other supported cells can
                # still provide penetration evidence for the same pair.
                unresolved_lower = min(unresolved_lower, lower)
                continue
        # Only finite permission seams need native-domain refinement.
        # Otherwise the full native shape is queried against smaller cells.
        split_native = (policy == "mixed" and not node.children
                        and np.linalg.norm(native_box[1] - native_box[0])
                        > np.linalg.norm(node.upper - node.lower))
        if split_native:
            axis = int(np.argmax(native_box[1] - native_box[0]))
            middle = (native_box[0][axis] + native_box[1][axis]) / 2
            if middle in (native_box[0][axis], native_box[1][axis]):
                unresolved_lower = min(unresolved_lower, lower)
                continue
            hi, lo = native_box[1].copy(), native_box[0].copy()
            hi[axis], lo[axis] = middle, middle
            push((native_box[0], hi), node, True)
            push((lo, native_box[1]), node, True)
        else:
            children = field.split(node)
            if not children:
                unresolved_lower = min(unresolved_lower, lower)
            for child in children:
                push(native_box, child, partitioned)
    if np.isfinite(unresolved_lower):
        return report("unresolved", unresolved_lower)
    # BEGIN the minimum retained bound covers the complete remaining domain.
    # With no finite separated branch, permissions or empty intersections
    # removed every domain; zero remains a valid lower bound and is not a
    # claim that the physical entities touch.
    lower = float(separated_lower) if np.isfinite(separated_lower) else 0.
    return report("clear", lower, intersection=False)
    # END no positive geometric clearance is added to the collision rule.
    # END numerical uncertainty is reported separately from penetration.
# END every definitive result concerns actual native/observed point pairs.
