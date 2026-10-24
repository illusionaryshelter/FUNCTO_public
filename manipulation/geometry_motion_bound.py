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
import pinocchio as pin


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

    Pinocchio placements and calibration store binary64 rotations whose
    entries need not be exactly orthogonal. Assuming norm one can underestimate the
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


def _array(value, label):
    # BEGIN covers and scalar speed inputs use the existing CPU model domain.
    # Pinocchio owns the robot metadata; this helper validates supplied cover
    # arrays and speed bounds without moving tensors between devices.
    device = getattr(value, "device", None)
    if device is not None and str(device) != "cpu":
        raise ValueError(f"{label} must already be CPU metadata")
    result = np.asarray(value)
    if result.dtype.kind not in "iuf" or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain finite numeric values")
    return result.astype(np.float64, copy=True)
    # END all admitted numerical arrays are explicit binary64 CPU values.


def _names(values, label):
    result = tuple(values)
    if (len(result) != len(set(result))
            or any(not isinstance(value, str) or not value.strip() for value in result)):
        raise ValueError(f"{label} must contain unique nonempty names")
    return result


@dataclass(frozen=True)
class SphereBody:
    """One complete rigid sphere cover in an existing Pinocchio frame.

    ``spheres`` has shape [S,4], with local centres and nonnegative radii.
    All robot and finger motion belongs to the same registered joint tree;
    attached observed fields retain their declared rigid tool relation.
    """

    name: str
    link: str
    spheres: object


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

        Speeds must cover every registered robot coordinate. A missing timed
        finger or unbound joint is an error, not an inferred zero velocity.
        The declared position ranges used to build L must hold on the interval.
        """
        if (not isinstance(joint_speed_upper, Mapping)
                or set(joint_speed_upper) != set(self.joint_names)):
            raise ValueError("Speed bounds must name every registered robot joint exactly")
        speeds = _array([joint_speed_upper[name] for name in self.joint_names], "joint speed bounds")
        if speeds.shape != (len(self.joint_names),) or np.any(speeds < 0):
            raise ValueError("Joint speed bounds must be finite nonnegative scalars")
        result = np.zeros(len(self.body_names), dtype=float)
        for i, row in enumerate(self.coefficients):
            for coefficient, speed in zip(row, speeds):
                result[i] = _add_up(float(result[i]), _mul_up(float(coefficient), float(speed)))
        return result


# BEGIN bind complete-entity motion to the current registered Pinocchio model.
def build_geometry_motion_bound(*, model, T_reference_base, bodies):
    """Bound point motion in the calibrated reference frame using Pinocchio.

    ``model`` is the exact fixed-base model admitted by RobotGeometryChecker:
    independent bounded scalar revolute/prismatic joints and fixed frames.
    ``T_reference_base`` is that checker's calibration from robot base to the
    geometry reference frame. ``bodies`` bind full local covers to existing
    model frames; the native collision entities themselves remain unchanged.

    Pinocchio ``frame.placement`` maps a frame into its parent joint, while
    ``jointPlacements[j]`` maps joint j's zero pose into its parent joint.
    ``joint.calc`` supplies each admitted joint's constant motion subspace
    S=[v;w], with linear rows first. This reads analytic joint metadata at
    neutral configuration; it does not estimate a velocity maximum by FK
    samples. For a local full-cover radius r, the joint contributes
    ||v|| + ||w|| r to its own coordinate. Prismatic motion also increases r
    by ||v|| max(|q_min|, |q_max|) before the ancestor's lever is bounded.

    Starting with the frame placement, walk these exact parent relations.
    Each fixed rotation's outward operator-norm bound scales all accumulated
    coefficients and the radius; its translation enlarges only the radius.
    The final calibration rotation scales the whole coefficient row. Its
    fixed translation has no velocity contribution. These triangle-inequality
    bounds cover every sphere point throughout the declared joint domain.
    """
    names = tuple(str(name) for name in model.names[1:])
    joints = tuple(model.joints[index] for index in range(1, model.njoints))
    columns = [int(joint.idx_q) for joint in joints]
    limits = np.array([model.lowerPositionLimit[columns], model.upperPositionLimit[columns]])
    neutral = pin.neutral(model)
    gains = []
    for joint in joints:
        data = joint.createData()
        joint.calc(data, neutral)
        subspace = np.asarray(data.S).reshape(6)
        gains.append((_norm_up(subspace[:3]), _norm_up(subspace[3:])))
    linear = [_linear_norm_up(placement.rotation) for placement in model.jointPlacements]
    translation = [_norm_up(placement.translation) for placement in model.jointPlacements]
    base_linear = _linear_norm_up(np.asarray(T_reference_base, float)[:3, :3])
    bodies = tuple(bodies)
    body_names = _names([body.name for body in bodies], "Collision bodies")
    coefficients = np.zeros((len(bodies), len(names)), dtype=float)
    radii = np.zeros(len(bodies), dtype=float)
    for body_index, body in enumerate(bodies):
        frame_id = int(model.getFrameId(body.link))
        if frame_id >= model.nframes:
            raise ValueError(f"Complete cover needs an existing robot frame: {body.link}")
        spheres = _array(body.spheres, "complete local sphere cover")
        if (spheres.ndim != 2 or spheres.shape[1:] != (4,) or not len(spheres)
                or np.any(spheres[:, 3] < 0)):
            raise ValueError("A complete active cover requires [S,4] spheres with nonnegative radii")
        radius = max(_add_up(_norm_up(sphere[:3]), float(sphere[3])) for sphere in spheres)
        radii[body_index] = radius
        frame = model.frames[frame_id]
        radius = _add_up(_mul_up(radius, _linear_norm_up(frame.placement.rotation)),
                         _norm_up(frame.placement.translation))
        row = coefficients[body_index]
        joint_id = int(frame.parentJoint)
        while joint_id:
            column = joint_id-1
            linear_gain, angular_gain = gains[column]
            row[column] = _add_up(linear_gain, _mul_up(angular_gain, radius))
            extent = max(abs(float(limits[0, column])), abs(float(limits[1, column])))
            radius = _add_up(radius, _mul_up(linear_gain, extent))
            for index in range(len(names)):
                row[index] = _mul_up(float(row[index]), linear[joint_id])
            radius = _add_up(_mul_up(radius, linear[joint_id]), translation[joint_id])
            joint_id = int(model.parents[joint_id])
        for column in range(len(names)):
            row[column] = _mul_up(float(row[column]), base_linear)
    for value in (coefficients, limits, radii):
        value.setflags(write=False)
    return GeometryMotionBound(body_names, names, coefficients, limits, radii)
# END native topology and calibration are consumed without a second model.


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
