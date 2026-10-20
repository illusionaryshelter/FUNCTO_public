"""Model-relative robot identity and shadow classification for calibrated RGB-D.

Each supplied component is a filled native collision solid. Coal performs point
and finite-segment distance queries, including true Euclidean sphere padding.
The result does not assert physical material identity: collision geometry may
approximate the robot, and calibration/synchronization remain caller inputs.
"""

# BEGIN three-state collision-model self filtering, independent of experiment assets.
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import coal
import numpy as np

from .observations import _intrinsics
from .robot_geometry_check import _convex_solid, _name, _rigid_pose


SELF = np.uint8(1)
ENVIRONMENT = np.uint8(2)
AMBIGUOUS_SHADOW = np.uint8(3)
_MAX_PIXELS = 2_000_000
_CHUNK = 16_384


@dataclass(frozen=True)
class RobotCollisionShape:
    """One native component in geometry-local coordinates, with a rigid pose.

    ``box`` dimensions are full x/y/z lengths; ``cylinder`` dimensions are
    radius/full z-axis length. ``convex`` vertices describe one filled native
    convex component. Scaling is about this explicitly supplied geometry's
    local origin, never its link or world origin; the adapter must preserve
    the native geometry centre when constructing this record.
    """

    name: str
    kind: str
    T_world_geometry: np.ndarray
    source: str
    dimensions: tuple = ()
    vertices: np.ndarray | None = None


def validate_self_filter(value, depth_m):
    """Validate the only production observation contract, without mutation.

    Unknown/SELF pixels contribute neither measured surfaces nor free rays.
    ENVIRONMENT rays retain exactly their original finite positive depth; an
    endpoint cannot be extended, guessed, or silently restored by a consumer.
    """
    depth = np.asarray(depth_m, dtype=float)
    if (depth.ndim != 2 or not depth.size or depth.size > _MAX_PIXELS
            or not isinstance(value, Mapping)
            or set(value) != {"labels", "free_ray_depth_m", "source"}):
        raise ValueError("Self filter needs bounded H,W depth and exactly labels/free_ray_depth_m/source")
    labels, free = np.asarray(value["labels"]), np.asarray(value["free_ray_depth_m"], dtype=float)
    if (labels.dtype != np.uint8 or labels.shape != depth.shape
            or not np.isin(labels, (SELF, ENVIRONMENT, AMBIGUOUS_SHADOW)).all()
            or free.shape != depth.shape or not np.isfinite(free).all() or np.any(free < 0)):
        raise ValueError("Self-filter labels must be aligned uint8 three-state values and free depths finite nonnegative")
    source = _name(value["source"], "self-filter source")
    environment = labels == ENVIRONMENT
    if (np.any(environment & (~np.isfinite(depth) | (depth <= 0)))
            or not np.array_equal(free[environment], depth[environment])
            or np.any(free[~environment] != 0)):
        raise ValueError("Only valid ENVIRONMENT pixels may retain their exact original free-ray depth")
    return {"labels": labels, "free_ray_depth_m": free, "source": source}


def _shape(record, scale):
    """Build a maintained-library solid; no triangle-surface containment test."""
    if record.kind == "convex":
        if record.dimensions:
            raise ValueError("A convex component has vertices, not primitive dimensions")
        vertices = np.asarray(record.vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not 4 <= len(vertices) <= 100_000:
            raise ValueError("A self-filter convex component exceeds the native vertex budget")
        return _convex_solid(vertices * scale)
    if record.vertices is not None:
        raise ValueError("An analytic primitive cannot also supply mesh vertices")
    dimensions = np.asarray(record.dimensions, dtype=float)
    count = {"box": 3, "cylinder": 2}.get(record.kind)
    if (count is None or dimensions.shape != (count,) or not np.isfinite(dimensions).all()
            or np.any(dimensions <= 0)):
        raise ValueError("Self-filter primitives require native positive box or cylinder dimensions")
    dimensions = dimensions * scale
    if not np.isfinite(dimensions).all():
        raise ValueError("Scaled primitive dimensions must remain finite")
    return coal.Box(*dimensions) if record.kind == "box" else coal.Cylinder(*dimensions)


def _world_bounds(shape, pose, numerical_band):
    """Enclose the library's local AABB, including its swept-sphere padding."""
    shape.computeLocalAABB()
    lower, upper = np.asarray(shape.aabb_local.min_), np.asarray(shape.aabb_local.max_)
    centre = pose[:3, :3] @ ((lower + upper) / 2) + pose[:3, 3]
    extent = np.abs(pose[:3, :3]) @ ((upper - lower) / 2) + numerical_band
    if not np.isfinite(np.r_[centre, extent]).all():
        raise ValueError("Robot self-filter geometry bounds must remain finite")
    return np.nextafter(centre - extent, -np.inf), np.nextafter(centre + extent, np.inf)


def _segment_rotations(directions):
    """Place each Coal capsule's z axis along its finite observation segment."""
    helper = np.zeros_like(directions)
    helper[:, 0] = 1.
    close = np.abs(directions[:, 0]) > .9
    helper[close] = [0., 1., 0.]
    first = np.cross(helper, directions)
    first /= np.linalg.norm(first, axis=1)[:, None]
    return np.stack((first, np.cross(directions, first), directions), axis=2)


def classify_depth(*, observed_depth, K, T_world_camera, geometries,
                   padding_scale, padding_offset_m, source):
    """Return three-state labels and conservative per-pixel free-ray depths.

    SELF wins when an endpoint is strictly inside any nominal component, even
    if another component shadows it: this is explicitly model-relative point
    ownership. Otherwise a padded finite-segment hit is AMBIGUOUS_SHADOW. Only
    ENVIRONMENT retains evidence. The numerical band uses unchanged Coal
    GJK/EPA defaults as a conservative diagnostic convention, not a proved error
    bound or a sensor accuracy claim. Query exceptions abort; nonfinite results
    remain ambiguous. No GPU, renderer, scene object state or policy is used.
    """
    depth = np.asarray(observed_depth, dtype=float)
    if depth.ndim != 2 or not depth.size or depth.size > _MAX_PIXELS:
        raise ValueError("Self filtering requires a nonempty bounded H,W depth image")
    matrix = _intrinsics(K)
    camera = _rigid_pose(T_world_camera, "T_world_camera")
    source = _name(source, "self-filter source")
    if (isinstance(padding_scale, (bool, np.bool_)) or not np.isscalar(padding_scale)
            or not np.isfinite(padding_scale) or padding_scale < 1
            or isinstance(padding_offset_m, (bool, np.bool_)) or not np.isscalar(padding_offset_m)
            or not np.isfinite(padding_offset_m) or padding_offset_m < 0):
        raise ValueError("Self filtering requires explicit finite padding_scale >= 1 and padding_offset_m >= 0")
    if not isinstance(geometries, Sequence) or not 0 < len(geometries) <= 256:
        raise ValueError("Self filtering requires one to 256 explicit native collision components")
    request = coal.DistanceRequest()
    request.enable_signed_distance = True
    band = max(float(request.gjk_tolerance), float(request.epa_tolerance))
    bodies, names = [], set()
    for record in geometries:
        if not isinstance(record, RobotCollisionShape) or record.name in names:
            raise ValueError("Self-filter components must have unique explicit shape records")
        names.add(_name(record.name, "collision component name"))
        _name(record.source, "collision component source")
        pose = _rigid_pose(record.T_world_geometry, "T_world_geometry")
        nominal = _shape(record, 1.)
        # With unit scale the library clone preserves the same native convex
        # hull exactly and avoids an unnecessary second hull construction.
        padded = nominal.clone() if padding_scale == 1 else _shape(record, float(padding_scale))
        padded.setSweptSphereRadius(float(padding_offset_m))
        bodies.append((record, nominal, padded, coal.Transform3s(pose[:3, :3], pose[:3, 3]),
                       _world_bounds(nominal, pose, band), _world_bounds(padded, pose, band)))
    labels = np.full(depth.size, AMBIGUOUS_SHADOW, dtype=np.uint8)
    valid = np.flatnonzero(np.isfinite(depth.ravel()) & (depth.ravel() > 0))
    counts = {"point_queries": 0, "segment_queries": 0, "nonfinite_queries": 0}
    point, segment, query_pose, result = coal.Sphere(0.), coal.Capsule(0., 1.), coal.Transform3s(), coal.DistanceResult()
    origin = camera[:3, 3]
    inverse_K = np.linalg.inv(matrix)
    # A finite chunk bounds all N-by-3 and N-by-3-by-3 intermediates. Neither
    # geometry count nor image size multiplies retained query storage.
    for start in range(0, len(valid), _CHUNK):
        pixels = valid[start:start + _CHUNK]
        rows, cols = np.divmod(pixels, depth.shape[1])
        rays = np.column_stack((cols, rows, np.ones(len(rows)))) @ inverse_K.T
        with np.errstate(over="ignore", invalid="ignore"):
            endpoints = (rays * depth.ravel()[pixels, None]) @ camera[:3, :3].T + origin
            delta = endpoints - origin
            lengths = np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])
        finite = np.isfinite(endpoints).all(axis=1) & np.isfinite(lengths) & (lengths > 0)
        state = np.where(finite, ENVIRONMENT, AMBIGUOUS_SHADOW).astype(np.uint8)
        for _, nominal, _, pose, (lower, upper), _ in bodies:
            candidates = np.flatnonzero(finite & (state != SELF)
                                       & np.all((endpoints >= lower) & (endpoints <= upper), axis=1))
            query_pose.setIdentity()
            for index in candidates:
                query_pose.setTranslation(endpoints[index])
                result.clear()
                distance = float(coal.distance(nominal, pose, point, query_pose, request, result))
                counts["point_queries"] += 1
                if not np.isfinite(distance):
                    state[index] = AMBIGUOUS_SHADOW
                    counts["nonfinite_queries"] += 1
                elif distance < -band:
                    state[index] = SELF
                elif distance <= band:
                    state[index] = AMBIGUOUS_SHADOW
        # A segment's bounding box only rejects impossible candidates. Coal,
        # not this broad phase, decides intersection with each filled solid.
        segment_lower, segment_upper = np.minimum(endpoints, origin), np.maximum(endpoints, origin)
        for _, _, padded, pose, _, (lower, upper) in bodies:
            candidates = np.flatnonzero(finite & (state == ENVIRONMENT)
                                       & np.all((segment_upper >= lower) & (segment_lower <= upper), axis=1))
            rotations = _segment_rotations(delta[candidates] / lengths[candidates, None])
            for row, index in enumerate(candidates):
                segment.halfLength = float(lengths[index] / 2)
                query_pose.setRotation(rotations[row])
                query_pose.setTranslation(origin + delta[index] / 2)
                result.clear()
                distance = float(coal.distance(padded, pose, segment, query_pose, request, result))
                counts["segment_queries"] += 1
                if not np.isfinite(distance):
                    state[index] = AMBIGUOUS_SHADOW
                    counts["nonfinite_queries"] += 1
                elif distance <= band:
                    state[index] = AMBIGUOUS_SHADOW
        labels[pixels] = state
    labels = labels.reshape(depth.shape)
    filtered = validate_self_filter({"labels": labels,
        "free_ray_depth_m": np.where(labels == ENVIRONMENT, depth, 0.), "source": source}, depth)
    return {"self_filter": filtered, "provenance": {
        "source": source, "padding_scale": float(padding_scale), "padding_offset_m": float(padding_offset_m),
        "scale_centre": "each component's explicit geometry-local origin; pose translation unchanged",
        "padding_definition": "uniform local scaling followed by Coal Euclidean swept-sphere dilation",
        "numerical_boundary_m": band, "numerical_boundary_is_proved_error_bound": False,
        "counts": {**counts, "self_pixels": int(np.count_nonzero(labels == SELF)),
                   "environment_pixels": int(np.count_nonzero(labels == ENVIRONMENT)),
                   "ambiguous_shadow_pixels": int(np.count_nonzero(labels == AMBIGUOUS_SHADOW))},
        "geometries": [{"name": row[0].name, "kind": row[0].kind, "source": row[0].source,
                        "T_world_geometry": np.asarray(row[0].T_world_geometry).tolist()} for row in bodies],
        "input_depth_modified": False, "renderer_used": False,
        "free_ray_policy": "only ENVIRONMENT retains original depth; SELF and AMBIGUOUS_SHADOW are zero",
        "self_scope": "strict interior of declared nominal collision model; not certified physical robot identity",
        "shadow_scope": "closed finite camera-to-observation segment; not an infinite ray",
    }}
# END only explicit environment observations contribute surface or free-space evidence.
