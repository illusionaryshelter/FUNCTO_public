"""CPU bounds on point motion from the already loaded native robot tree.

The result covers every point of each supplied local sphere union. A caller
must bind the complete entity's existing outer cover, not selected witnesses.
No FK, trajectory resampling, collision decision or physical error margin is
computed here. Bounds concern the continuous model represented by the supplied
numbers; they do not bound CUDA rounding, tracking error or omitted geometry.
"""

# BEGIN complete sphere-cover motion bounds from existing kinematic metadata.
from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np


def _finite(value):
    if not math.isfinite(value):
        raise ValueError("Geometry motion bounds must remain finite")
    return value


def _add_up(first, second):
    """Round a nonnegative sum outwards, preserving exact structural zeros."""
    if first == 0:
        return second
    if second == 0:
        return first
    return _finite(math.nextafter(_finite(first + second), math.inf))


def _mul_up(first, second):
    if first == 0 or second == 0:
        return 0.
    return _finite(math.nextafter(_finite(first * second), math.inf))


def _norm_up(vector):
    squared = 0.
    for value in vector:
        squared = _add_up(squared, _mul_up(abs(float(value)), abs(float(value))))
    return 0. if squared == 0 else _finite(math.nextafter(math.sqrt(squared), math.inf))


def _product_interval(first, second):
    """Enclose an exact product of two stored binary64 inputs."""
    if first == 0 or second == 0:
        return 0., 0.
    value = _finite(first * second)
    return math.nextafter(value, -math.inf), math.nextafter(value, math.inf)


def _linear_norm_up(matrix):
    """Bound ||A||_2 by sqrt(||A.T A||_infinity), with outward arithmetic.

    Native fixed rotations are stored in float32 and need not be exactly
    orthogonal. Treating their norm as exactly one could underestimate the
    bound. Signed intervals retain cancellation in the Gram matrix, so this
    remains close to one for rotations instead of multiplying sqrt(3) per link.
    """
    gram_abs = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            lower = upper = 0.
            for row in range(3):
                lo, hi = _product_interval(float(matrix[row, i]), float(matrix[row, j]))
                if lo != 0 or hi != 0:
                    lower = _finite(math.nextafter(_finite(lower + lo), -math.inf))
                    upper = _finite(math.nextafter(_finite(upper + hi), math.inf))
            gram_abs[i, j] = max(abs(lower), abs(upper))
    maximum = 0.
    for row in gram_abs:
        total = 0.
        for value in row:
            total = _add_up(total, float(value))
        maximum = max(maximum, total)
    return 0. if maximum == 0 else _finite(math.nextafter(math.sqrt(maximum), math.inf))


def _affine_abs_up(scale, offset, lower, upper):
    """Maximum |scale*q+offset| over the declared full joint interval."""
    maximum = 0.
    for position in (lower, upper):
        lo, hi = _product_interval(scale, position)
        lo = _finite(math.nextafter(_finite(lo + offset), -math.inf))
        hi = _finite(math.nextafter(_finite(hi + offset), math.inf))
        maximum = max(maximum, abs(lo), abs(hi))
    return maximum


def _array(value, label, *, integer=False):
    # The caller owns the single explicit device-to-CPU metadata copy. This
    # helper must not accidentally transfer tensors or initialize a GPU model.
    device = getattr(value, "device", None)
    if device is not None and str(device) != "cpu":
        raise ValueError(f"{label} must already be CPU metadata")
    result = np.asarray(value)
    if (result.dtype.kind not in ("iu" if integer else "iuf")
            or not np.isfinite(result).all()):
        raise ValueError(f"{label} must contain finite {'integer' if integer else 'numeric'} values")
    return result.astype(np.int64 if integer else np.float64, copy=True)


def _names(values, label):
    result = tuple(values)
    if (len(result) != len(set(result))
            or any(not isinstance(value, str) or not value.strip() for value in result)):
        raise ValueError(f"{label} must contain unique nonempty names")
    return result


@dataclass(frozen=True)
class SphereBody:
    """One complete cover in a named native link frame.

    ``spheres`` has shape [S,4], with local centres and nonnegative radii.
    ``affine_translation`` explicitly maps each independently moving scalar
    joint name to [S,3] local centre translations per unit q. Pass {} for a
    rigid cover. For the existing timed hand, use its TCP-local spheres and
    ``sphere_joint_translation`` here, rather than its locked native fingers.
    Already transformed/offset spheres must not be transformed a second time.
    """

    name: str
    link: str
    spheres: object
    affine_translation: Mapping


@dataclass(frozen=True)
class GeometryMotionBound:
    """L[g,j] bounds point speed by sum_j L[g,j] * abs(dq[j])."""

    body_names: tuple
    joint_names: tuple
    coefficients: np.ndarray
    joint_position_limits: np.ndarray
    local_radius_bounds_m: np.ndarray

    def speed_bounds(self, joint_speed_upper):
        """Return one m/s bound per body for complete named |dq| bounds.

        Speeds must cover every native and affine coordinate. A missing timed
        finger or unbound joint is an error, not an inferred zero velocity.
        The declared position ranges used to build L must hold on the interval.
        """
        if (not isinstance(joint_speed_upper, Mapping)
                or set(joint_speed_upper) != set(self.joint_names)):
            raise ValueError("Speed bounds must name every native and affine joint exactly")
        speeds = _array([joint_speed_upper[name] for name in self.joint_names], "joint speed bounds")
        if speeds.shape != (len(self.joint_names),) or np.any(speeds < 0):
            raise ValueError("Joint speed bounds must be finite nonnegative scalars")
        result = np.zeros(len(self.body_names), dtype=float)
        for i, row in enumerate(self.coefficients):
            for coefficient, speed in zip(row, speeds):
                result[i] = _add_up(float(result[i]), _mul_up(float(coefficient), float(speed)))
        return result


def build_geometry_motion_bound(*, fixed_transforms, link_map, joint_map,
                                joint_map_type, joint_offset_map, joint_names,
                                joint_position_limits, link_name_to_idx_map,
                                bodies, extra_joint_limits):
    """Build small CPU coefficients from CuRobo KinematicsParams metadata.

    Copy the identically named native arrays once after model loading.
    ``joint_position_limits`` is ``params.joint_limits.position`` [2,J]; the
    native ``joint_offset_map`` is accepted in its flat [2*N] layout. Parent
    indices come directly from ``link_map``; no alternate FK/tree is built.
    Native root index 0 is fixed and its stored base transform is included.

    ``extra_joint_limits`` is mandatory: {} explicitly declares no extra
    affine coordinates, otherwise name -> (lower, upper) supplies every such
    coordinate. Affine translation may also use an existing native coordinate.
    Unmodelled hand motion cannot be represented by silently omitting this
    declaration. Freely moving scene owners are not part of this fixed-base
    tree; their independently bounded speeds must enter relative_speed_bound.

    The native CUDA forward kernel uses joint types -1 and 0..5 (fixed,
    xyz-prismatic, xyz-revolute), with negative axes/mimic in scale*q+offset.
    Legacy negative-axis enum codes 6..11 are rejected rather than assumed to
    have the same kernel semantics. Locked joints must already be folded into
    fixed transforms by the existing loader.

    Proof: starting at each body's full local radius, walk toward the base.
    A rotating edge contributes |scale|*current_radius to its coordinate; a
    translating edge contributes |scale| and increases the radius by its full
    interval extension. The fixed linear transform multiplies both bounds by
    its operator-norm bound and its translation increases radius. Triangle
    inequalities and addition for repeated mimic coordinates give L. This
    also covers interior sphere points, not only centres or mesh vertices.
    """
    transforms = _array(fixed_transforms, "fixed transforms")
    if transforms.ndim != 3 or transforms.shape[1:] != (3, 4) or not len(transforms):
        raise ValueError("Native fixed transforms must have shape [links,3,4]")
    count = len(transforms)
    parents = _array(link_map, "link map", integer=True)
    joints = _array(joint_map, "joint map", integer=True)
    types = _array(joint_map_type, "joint types", integer=True)
    offsets = _array(joint_offset_map, "joint offsets")
    if offsets.shape == (2*count,):
        offsets = offsets.reshape(count, 2)
    if (parents.shape != (count,) or joints.shape != (count,) or types.shape != (count,)
            or offsets.shape != (count, 2) or np.any(parents < 0) or np.any(parents >= count)
            or parents[0] != 0 or types[0] != -1 or joints[0] != -1
            or np.any((types < -1) | (types > 5))):
        raise ValueError("Native tree indices, fixed root or joint type codes are unsupported")
    names = _names(joint_names, "Native joints")
    limits = _array(joint_position_limits, "joint position limits")
    if (limits.shape != (2, len(names)) or np.any(limits[0] > limits[1])
            or np.any(joints[types == -1] != -1)
            or np.any(joints[types != -1] < 0) or np.any(joints[types != -1] >= len(names))):
        raise ValueError("Every moving edge needs a named coordinate and finite ordered position bounds")
    if not isinstance(link_name_to_idx_map, Mapping):
        raise ValueError("Native link names must map to their exact indices")
    links = dict(link_name_to_idx_map)
    _names(links, "Native links")
    if (any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in links.values()) or set(links.values()) != set(range(count))
            or len(links) != count):
        raise ValueError("Native link names must uniquely cover every tree index")
    # Check the whole admitted tree once, including branches without a body.
    for start in range(1, count):
        seen, link = set(), start
        while link:
            if link in seen:
                raise ValueError("Native parent links contain a cycle")
            seen.add(link)
            link = int(parents[link])
    if not isinstance(extra_joint_limits, Mapping):
        raise ValueError("Explicit extra affine joint limits are required, including {} when absent")
    extra_names = _names(extra_joint_limits, "Extra affine joints")
    if set(names) & set(extra_names):
        raise ValueError("An affine motion reuses an existing joint's bounds rather than redefining them")
    if extra_names:
        extra = _array([extra_joint_limits[name] for name in extra_names], "extra joint limits")
        if extra.shape != (len(extra_names), 2) or np.any(extra[:, 0] > extra[:, 1]):
            raise ValueError("Extra affine joints need finite ordered (lower,upper) bounds")
        limits = np.concatenate((limits, extra.T), axis=1)
    names = names + extra_names
    columns = {name: i for i, name in enumerate(names)}
    bodies = tuple(bodies)
    if not bodies or any(not isinstance(body, SphereBody) for body in bodies):
        raise ValueError("At least one explicitly bound complete SphereBody is required")
    body_names = _names([body.name for body in bodies], "Collision bodies")
    linear = [_linear_norm_up(transform[:, :3]) for transform in transforms]
    translation = [_norm_up(transform[:, 3]) for transform in transforms]
    coefficients = np.zeros((len(bodies), len(names)), dtype=float)
    radii = np.zeros(len(bodies), dtype=float)
    used_extra = set()
    for body_index, body in enumerate(bodies):
        if body.link not in links or not isinstance(body.affine_translation, Mapping):
            raise ValueError("Each body needs a native link and explicit local affine-motion mapping")
        spheres = _array(body.spheres, "complete local sphere cover")
        if (spheres.ndim != 2 or spheres.shape[1:] != (4,) or not len(spheres)
                or np.any(spheres[:, 3] < 0)):
            raise ValueError("A complete active cover requires [S,4] spheres with nonnegative radii")
        # Negative-radius native attachment slots are inactive, not entities.
        # The binding must exclude those slots, never discard active spheres.
        local_radii = np.array([_add_up(_norm_up(row[:3]), float(row[3])) for row in spheres])
        row = coefficients[body_index]
        for name, value in body.affine_translation.items():
            if name not in columns:
                raise ValueError(f"Affine body motion lacks joint position bounds: {name}")
            mapping = _array(value, "affine sphere translations")
            if mapping.shape != (len(spheres), 3):
                raise ValueError("Affine translations must provide one local 3-vector per cover sphere")
            column = columns[name]
            norm = [_norm_up(value) for value in mapping]
            extent = max(abs(float(limits[0, column])), abs(float(limits[1, column])))
            for sphere_index, gain in enumerate(norm):
                local_radii[sphere_index] = _add_up(float(local_radii[sphere_index]), _mul_up(gain, extent))
            row[column] = max(norm)
            used_extra.add(name)
        radius = float(local_radii.max())
        radii[body_index] = radius
        link = int(links[body.link])
        while True:
            kind = int(types[link])
            if kind != -1:
                column = int(joints[link])
                scale, offset = map(float, offsets[link])
                if kind >= 3:
                    row[column] = _add_up(float(row[column]), _mul_up(abs(scale), radius))
                else:
                    row[column] = _add_up(float(row[column]), abs(scale))
                    extension = _affine_abs_up(scale, offset, *map(float, limits[:, column]))
                    radius = _add_up(radius, extension)
            for column in range(len(names)):
                row[column] = _mul_up(float(row[column]), linear[link])
            radius = _add_up(_mul_up(radius, linear[link]), translation[link])
            if link == 0:
                break
            link = int(parents[link])
    if not set(extra_names) <= used_extra:
        raise ValueError("Every extra affine joint must be explicitly used by a body's motion mapping")
    for value in (coefficients, limits, radii):
        value.setflags(write=False)
    return GeometryMotionBound(body_names, names, coefficients, limits, radii)


def relative_speed_bound(actor_speed_m_s, *, target_speed_m_s):
    """Bound separation change by both point speeds; static means explicit 0.

    A moving owner with no motion bound must remain unresolved. Requiring the
    target argument prevents silently treating absent owner motion as static.
    For two robot bodies pass their respective speed_bounds entries. For an
    attached payload bind its complete cover to the tool frame before building
    L; for a separately moving owner supply its independently derived bound.
    """
    values = _array([actor_speed_m_s, target_speed_m_s], "actor and target speed bounds")
    if values.shape != (2,) or np.any(values < 0):
        raise ValueError("Actor and target point-speed bounds must be nonnegative scalars")
    return _add_up(float(values[0]), float(values[1]))
# END no FK, curve sampling, collision acceptance or physical margins are changed.
