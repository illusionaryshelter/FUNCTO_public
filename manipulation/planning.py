"""Generate one robot joint trajectory for a complete grounded skill window.

cuRobo V2 owns forward kinematics, inverse kinematics, joint constraints and
the final cubic trajectory.  Skill and scene violations enter its optimizer
through the same cost-manager extension used for result evaluation.  No
perception model, CUDA context or robot connection is initialized on import.

The public result contains measured numerical residuals and original event
identities.  It does not manufacture grasp evidence or execute robot commands.
"""

# BEGIN complete-window robot planning and generation-time process constraints.
from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
import gc
import time
from uuid import uuid4
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline, CubicHermiteSpline
from .planning_problem import event_occurrence_key
from .geometry_motion_bound import SphereBody, build_geometry_motion_bound
from .collision_records import encode_array
# BEGIN physical memory ownership is independent of optimizer workspace size.
from .resource_admission import (
    ResourceLimitError, require_host_memory,
    DEFAULT_MAX_HOST_RSS_BYTES, MAX_HOST_RSS_BYTES,
    DEFAULT_MAX_DEVICE_USED_BYTES, MAX_DEVICE_USED_BYTES, MIN_DEVICE_FREE_BYTES,
)
# END source geometry and native CUDA buffers use the same typed refusals.

# BEGIN final validation separates total finite work from live geometry storage.
_VALIDATION_CHUNK_SIZE = 256
_NATIVE_STAGE_POINT_LIMIT = 8192
_VALIDATION_RECORD_BYTE_LIMIT = 512 * 1024**2
_PROCESS_GEOMETRY_BYTE_LIMIT = 256 * 1024**2
# END increasing final work cannot enlarge native anchor/search batches.

# BEGIN reusable physical-domain contract for native numerical iterates.
from .physical_joint_domain import (CubicJointDomain, JointPositionDomain,
                                    JointDomainInitializationError, cubic_position_report)
# END importing this CPU module does not allocate any CUDA workspace.


# BEGIN one exact output grid for the unchanged native cubic clock.
def _fixed_spline_grid(*, intervals, native_dt, interpolation_steps, interpolation_dt):
    """Admit an output grid only when it preserves the native physical clock.

    The installed dense kernel reconstructs its knot interval as float32
    ``output_dt * subdivisions``. Check that product AND the saved binary64
    time axes before using it. Rounding only proposes an integer subdivision
    count; it never authorizes retiming an incompatible input.
    """
    if (type(intervals) is not int or intervals < 1
            or type(interpolation_steps) is not int or interpolation_steps < 1):
        raise ValueError("Native spline intervals and subdivisions must be positive integers")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        raw_dt = np.float32(native_dt)
        output_dt = np.float32(interpolation_dt)
        knot_dt = np.float32(raw_dt * np.float32(interpolation_steps))
    if (not np.isfinite([raw_dt, output_dt, knot_dt]).all()
            or min(raw_dt, output_dt, knot_dt) <= 0):
        raise ValueError("Spline timing requires finite positive native float32 intervals")
    subdivisions = int(round(float(knot_dt) / float(output_dt)))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        reconstructed = np.float32(output_dt * np.float32(subdivisions))
    if (subdivisions < 1 or reconstructed != knot_dt
            or float(output_dt) * subdivisions != float(knot_dt)
            or float(raw_dt) * interpolation_steps != float(knot_dt)):
        raise ValueError("Exact output grid must preserve the native spline knot interval; retiming is unsupported")
    count = intervals * subdivisions + 1
    duration = intervals * float(knot_dt)
    if (count - 1) * float(output_dt) != duration:
        raise ValueError("Exact output grid changed the native spline duration")
    return {"native_dt": float(raw_dt), "knot_dt": float(knot_dt),
            "interpolation_dt": float(output_dt), "subdivisions": subdivisions,
            "valid_count": count, "duration": duration}
# END incompatible grids never change dt, controls, or boundary derivatives.


@dataclass(frozen=True)
class PlanningBudget:
    """Explicit local resource and numerical-work limits, for a single request.

    These defaults bound the current 16 GB RAM / 12 GB GPU integration. More
    seeds or a larger batch must not be silently allocated by native defaults.
    The interpolation buffer covers the largest allowed native knot interval.
    Native solvers use eager execution for the dynamic process/scene tensors.
    Required source vertices use their actual count; ``max_refinement_points``
    bounds only additional queries. Both share ``max_validation_points`` and
    the checked scene/native allocation budgets.
    ``maximum_trajectory_dt`` fixes both native time bounds for this request:
    process feasibility and evidence stops take precedence over compression.
    """

    ik_seeds: int = 8
    trajectory_seeds: int = 2
    knots: int = 8
    interpolation_steps: int = 4
    ik_iterations: int = 40
    trajectory_iterations: int = 100
    interpolation_dt: float = 0.025
    maximum_trajectory_dt: float = 0.2
    interpolation_buffer_size: int = 512
    constraint_weight: float = 100000.0
    # BEGIN reserve optional refinement work separately from required evidence.
    max_refinement_points: int = 64
    # END reserve optional refinement work separately from required evidence.
    max_robot_spheres: int = 256
    max_robot_dof: int = 16
    max_robot_links: int = 64
    # BEGIN GPU scene workspaces exclude all resident CPU source geometry.
    max_scene_gpu_workspace_bytes: int = 1024**3
    max_host_rss_bytes: int = DEFAULT_MAX_HOST_RSS_BYTES
    max_device_used_bytes: int = DEFAULT_MAX_DEVICE_USED_BYTES
    # END larger physical limits require an explicit request, never a fallback.
    max_validation_points: int = 4096
    validation_joint_step: float = 0.025
    # BEGIN metric sampling also has meaning for analytic solids without voxels.
    validation_spatial_step_m: float = 0.001
    # END this relative-displacement bound is not a physical collision margin.
    search_seconds: float = 3.0
    maximum_trajectory_solves: int = 3
    # BEGIN wall work is cumulative across binding attempts, independently of physics.
    planning_seconds: float = 120.0
    # END a stage may finish over this limit, but its candidate cannot be accepted.
    random_seed: int = 123

    def __post_init__(self):
        if (not 1 <= self.trajectory_seeds <= 2 or not self.trajectory_seeds <= self.ik_seeds <= 8
                or not 6 <= self.knots <= 24 or not 1 <= self.interpolation_steps <= 8
                or not 1 <= self.ik_iterations <= 200 or not 1 <= self.trajectory_iterations <= 300
                # BEGIN admit bounded complete-volume robot cover refinement.
                # This is storage capacity, not a collision clearance or an
                # acceptance tolerance. Scene/native memory preflight remains
                # mandatory for every actual sphere count and query horizon.
                or not 1 <= self.max_robot_spheres <= 640
                # END geometric coverage and cumulative work limits are unchanged.
                or not 1 <= self.max_robot_dof <= 16
                or not 1 <= self.max_robot_links <= 64
                or not 32 <= self.max_validation_points <= 65536
                or not 1 <= self.maximum_trajectory_solves <= 4):
            raise ValueError("Planning budget exceeds the bounded local single-batch configuration")
        # BEGIN bound optional samples without capping a skill's source knots.
        if (type(self.max_refinement_points) is not int
                or not 0 <= self.max_refinement_points <= self.max_validation_points):
            raise ValueError("Refinement points must fit the finite validation-point budget")
        # END bound optional samples without capping a skill's source knots.
        # BEGIN physical host/device caps are separate from GPU scene capacity.
        if (type(self.max_scene_gpu_workspace_bytes) is not int
                or not 0 <= self.max_scene_gpu_workspace_bytes <= 6*1024**3):
            raise ValueError("GPU scene workspaces must remain within the local six-GiB allocation budget")
        if (type(self.max_host_rss_bytes) is not int
                or not 0 < self.max_host_rss_bytes <= MAX_HOST_RSS_BYTES):
            raise ValueError("Host RSS must remain within the explicit ten-GiB maximum")
        if (type(self.max_device_used_bytes) is not int
                or not 0 < self.max_device_used_bytes <= MAX_DEVICE_USED_BYTES):
            raise ValueError("Device usage must remain within the explicit eight-GiB maximum")
        # END defaults retain the existing half-host/half-device resource limits.
        for name in ("interpolation_dt", "maximum_trajectory_dt", "constraint_weight", "validation_joint_step",
                     "validation_spatial_step_m", "planning_seconds"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.search_seconds) or self.search_seconds < 0:
            raise ValueError("search_seconds must be finite and nonnegative")
        # BEGIN admit the exact grid before any native buffer allocation.
        # The selected cubic has degree+1 virtual intervals. Output sampling
        # must preserve its original physical clock, including nonzero dq0.
        grid = _fixed_spline_grid(intervals=self.knots + 4,
            native_dt=self.maximum_trajectory_dt, interpolation_steps=self.interpolation_steps,
            interpolation_dt=self.interpolation_dt)
        maximum_count = grid["valid_count"]
        # END retain all original buffer, validation and workspace bounds.
        native_count = (self.knots + 4) * self.interpolation_steps + 1
        if (self.interpolation_buffer_size < maximum_count
                or self.interpolation_buffer_size <= native_count
                or self.interpolation_buffer_size > 2048):
            raise ValueError("Interpolation buffer must cover the bounded cubic trajectory and differ from its rollout horizon")


@dataclass
class JointTrajectory:
    """An exported cubic joint curve and its exact skill-progress mapping.

    All arrays are CPU NumPy arrays in named active-joint order.  The cubic
    reconstruction uses every native knot subdivision and its derivative;
    it therefore represents the exported cuRobo cubic, rather than an EEF
    interpolation or a new smoothing pass.
    """

    joint_names: tuple[str, ...]
    times: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    progress_u: np.ndarray
    progress_s: np.ndarray
    events: tuple[dict, ...]
    conditions: tuple[dict, ...]
    residuals: dict[str, float]
    # BEGIN archive the exact independently controlled finger reference.
    finger_trajectory: object | None = None
    # END a missing hand curve is valid only for plans without hand events.
    _curve: CubicHermiteSpline = field(init=False, repr=False)

    def __post_init__(self):
        self._curve = CubicHermiteSpline(self.times, self.position, self.velocity, axis=0)

    @property
    def duration(self):
        return float(self.times[-1])

    @property
    def next_verification(self):
        """The first original evidence barrier, without claiming it was met."""
        barriers = [event for event in self.events if event["requires_evidence"] is not None]
        return deepcopy(min(barriers, key=lambda event: event["time"])) if barriers else None

    @property
    def executable_until(self):
        """Limit of this conditional plan before new actual feedback is needed."""
        barrier = self.next_verification
        return self.duration if barrier is None else float(barrier["time"])

    def sample(self, times):
        # BEGIN one exact polynomial query owns all three control references.
        """Return q, dq, ddq and progress at the same finite curve time.

        SciPy evaluates an internal breakpoint on its right-hand piece; the
        final endpoint uses the last piece's left limit.  Thus acceleration
        is the analytic, one-sided second derivative of this saved C1 curve,
        not an estimated derivative, a new smoothing pass, or actual qacc.
        Independent acceleration validation still checks both knot sides.
        """
        query = np.asarray(times, dtype=float)
        if not np.isfinite(query).all() or np.any(query < 0) or np.any(query > self.duration):
            raise ValueError("Trajectory queries must lie within its finite time interval")
        return (self._curve(query), self._curve(query, 1), self._curve(query, 2),
                np.interp(query / self.duration, self.progress_u, self.progress_s))
        # END no q-only or derivative-free control sampling fallback.


# BEGIN continuous acceleration bounds on the exact exported Hermite curve.
def cubic_derivative_report(trajectory, acceleration_bounds, *, joint_names, until=None):
    """Check each scalar cubic's acceleration range, including both knot sides.

    Acceleration is affine within a cubic piece, so its extrema occur at that
    piece's endpoints. An internal knot has two acceleration limits: evaluating
    only ``curve(knot, 2)`` would lose the previous piece's left limit. A supplied
    ``until`` clips the last piece at the original execution barrier; future
    pieces do not enter that prefix report. Limits are bound by joint identity
    and compared in binary64 without adding a numerical or physical allowance.

    The input remains the original CubicHermiteSpline. Its C1 construction and
    raw floating-point knot jumps are reported; acceleration jumps do not add a
    C2 requirement. This checks a reference polynomial, not actual robot motion,
    continuous collision, jerk at jumps, or actuator forces.
    """
    names, curve_names = tuple(joint_names), tuple(trajectory.joint_names)
    bounds = np.asarray(acceleration_bounds, dtype=float)
    if (not names or len(set(names)) != len(names) or len(set(curve_names)) != len(curve_names)
            or set(names) != set(curve_names) or bounds.shape != (2, len(names))
            or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1])):
        raise ValueError("Continuous acceleration checks require complete finite named lower/upper bounds")
    bounds = bounds[:, [names.index(name) for name in curve_names]]
    times = np.asarray(trajectory.times, dtype=float)
    position, velocity = np.asarray(trajectory.position, dtype=float), np.asarray(trajectory.velocity, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or times[0] != 0.
            or np.any(np.diff(times) <= 0.) or position.shape != (len(times), len(names))
            or velocity.shape != position.shape or not np.isfinite(position).all()
            or not np.isfinite(velocity).all()):
        raise ValueError("Continuous acceleration checks require a complete finite ordered Hermite curve")
    curve = CubicHermiteSpline(times, position, velocity, axis=0)
    if (not np.array_equal(curve.x, trajectory._curve.x)
            or not np.array_equal(curve.c, trajectory._curve.c)):
        raise ValueError("Exported arrays differ from the executable Hermite polynomial")
    stop = float(times[-1] if until is None else until)
    if not np.isfinite(stop) or not 0. <= stop <= times[-1]:
        raise ValueError("Derivative report interval must lie inside the original curve")
    indices = np.flatnonzero(times[:-1] < stop)
    if not len(indices):
        # A zero-length prefix reports the initial right derivative only.
        indices = np.array([0])
    ends = np.minimum(times[indices+1], stop)
    widths = (ends-times[indices])[:, None]
    coefficients = curve.c[:, indices]
    acceleration = np.stack((2*coefficients[1],
                             6*coefficients[0]*widths+2*coefficients[1]), axis=1)
    values = acceleration.reshape(-1, len(names))
    endpoint_times = np.stack((times[indices], ends), axis=1).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite analytic acceleration on the exported polynomial")
    columns = np.arange(len(names))
    low_indices, high_indices = values.argmin(axis=0), values.argmax(axis=0)
    minimum, maximum = values[low_indices, columns], values[high_indices, columns]
    lower_excess, upper_excess = np.maximum(bounds[0]-minimum, 0.), np.maximum(maximum-bounds[1], 0.)
    excess = np.stack((lower_excess, upper_excess))
    maximum_excess = float(excess.max())
    witness = None
    if maximum_excess > 0.:
        bound_index, joint_index = np.unravel_index(excess.argmax(), excess.shape)
        endpoint = (low_indices if bound_index == 0 else high_indices)[joint_index]
        witness = {"joint_name": curve_names[joint_index], "time_s": float(endpoint_times[endpoint]),
                   "segment_index": int(indices[endpoint//2]),
                   "side": "segment_start" if endpoint % 2 == 0 else "segment_end_left_limit",
                   "acceleration": float(values[endpoint, joint_index]),
                   "bound": float(bounds[bound_index, joint_index]), "excess": maximum_excess}

    # Only internal breakpoints of the requested interval have two executed
    # sides. In particular, a barrier exactly at a knot excludes future ddq.
    internal = np.flatnonzero(times[1:-1] < stop)
    c, h = curve.c, np.diff(times)[:, None]
    q_end = ((c[0]*h+c[1])*h+c[2])*h+c[3]
    dq_end = (3*c[0]*h+2*c[1])*h+c[2]
    ddq_end = 6*c[0]*h+2*c[1]
    jumps = {
        "position": c[3, 1:]-q_end[:-1],
        "velocity": c[2, 1:]-dq_end[:-1],
        "acceleration": 2*c[1, 1:]-ddq_end[:-1],
    }
    continuity = {"c1_by_hermite_construction": True, "internal_breakpoints": len(internal),
                  "acceleration_jump_is_rejection": False}
    for label, jump in jumps.items():
        selected = jump[internal]
        if not np.isfinite(selected).all():
            raise ValueError("Nonfinite derivative jump on the exported polynomial")
        continuity[f"{label}_maximum_absolute_jump"] = (
            np.abs(selected).max(axis=0) if len(selected) else np.zeros(len(names))).tolist()
    return {
        "scope": "binary64 polynomial acceleration ranges and knot continuity; no actual-motion or collision certification",
        "checked_interval_s": [0., stop], "full_curve_duration_s": float(times[-1]),
        "joint_names": list(curve_names), "checked_piece_count": len(indices),
        "acceleration_bounds": bounds.tolist(),
        "acceleration": {"minimum": minimum.tolist(), "maximum": maximum.tolist(),
                         "minimum_time_s": endpoint_times[low_indices].tolist(),
                         "maximum_time_s": endpoint_times[high_indices].tolist(),
                         "maximum_absolute": np.maximum(np.abs(minimum), np.abs(maximum)).tolist(),
                         "lower_excess": lower_excess.tolist(), "upper_excess": upper_excess.tolist(),
                         "maximum_bound_excess": maximum_excess, "bounds_satisfied": maximum_excess == 0.,
                         "maximum_violation": witness},
        "continuity": continuity,
    }
# END continuous acceleration bounds; no C2 requirement or curve modification.


# BEGIN complete-body spatial sampling of the original cubic, with finite work.
def _cubic_speed_upper(curve, left, right):
    """Enclose |dq| on pieces split at every original polynomial breakpoint.

    Endpoints and any quadratic vertex exhaust derivative extrema. Directed
    binary64 interval operations include evaluation and vertex-division
    rounding; no sampled Jacobian or endpoint displacement replaces this bound.
    """
    index = np.clip(np.searchsorted(curve.x, left, side="right")-1, 0, len(curve.x)-2)
    down, up = lambda x: np.nextafter(x, -np.inf), lambda x: np.nextafter(x, np.inf)
    def add(x, y):
        return down(x[0]+y[0]), up(x[1]+y[1])
    def multiply(x, y):
        terms = np.stack([x[i]*y[j] for i in (0, 1) for j in (0, 1)])
        return down(terms.min(axis=0)), up(terms.max(axis=0))
    def absolute(x):
        return np.maximum(abs(x[0]), abs(x[1]))
    a, b, c = curve.c[:3, index]
    qa, qb = multiply((a, a), (3., 3.)), multiply((b, b), (2., 2.))
    lo, hi = (left-curve.x[index])[:, None], (right-curve.x[index])[:, None]
    result = np.zeros_like(c)
    for x in (lo, hi):
        point = (down(x), up(x))
        result = np.maximum(result, absolute(add(multiply(add(multiply(qa, point), qb), point), (c, c))))
    # At an interior vertex the derivative equals c-b*b/(3*a). Include it
    # whenever the outward root enclosure intersects this interval.
    nonzero = a != 0
    denominator = tuple(np.where(nonzero, value, 1.) for value in qa)
    reciprocal = (down(1./denominator[1]), up(1./denominator[0]))
    root = multiply((-b, -b), reciprocal)
    vertex = add((c, c), multiply(multiply((b, b), (b, b)),
                                (-reciprocal[1], -reciprocal[0])))
    present = nonzero & (root[1] >= lo) & (root[0] <= hi)
    result = np.maximum(result, np.where(present, absolute(vertex), 0.))
    if not np.isfinite(result).all():
        raise ValueError("Cubic speed bound is nonfinite")
    return result


def _export_sample_times(trajectory, motion, *, spatial_step, anchors, budget, owner_speed_per_progress=0.,
                         finger_geometry=None):
    """Bound all body and pair displacement, retaining the old joint-step rule.

    Twice the fastest body's speed bounds relative motion of every robot
    pair. This deliberately includes native-excluded pairs rather than risk
    omitting a required pair.
    External owner rates and every attached owner's complete cover also enter
    the maximum. The spatial step is a sampling resolution, never a physical error margin
    or a continuous collision-freedom certificate.
    """
    curves = [(tuple(trajectory.joint_names), trajectory._curve)]
    # Executable finger commands and nominal future geometry are distinct.
    # The caller supplies precisely the model selected by scene.hand.state;
    # trajectory.finger_trajectory retains its original command meaning.
    fingers = finger_geometry
    if fingers is not None:
        curves.append((tuple(fingers.joint_names), fingers._curve))
    if sorted(name for names, _ in curves for name in names) != sorted(motion.joint_names):
        raise ValueError("Every body-motion coordinate requires exactly one original cubic")
    # The full-range lever arms in L are valid only inside their declared
    # position ranges. Establish that premise analytically before creating
    # any spatial evidence; eventual sampled native rejection is insufficient.
    for names, curve in curves:
        columns = [motion.joint_names.index(name) for name in names]
        domain = cubic_position_report(curve, *motion.joint_position_limits[:, columns])
        if domain["maximum_bound_excess"] > 0:
            raise ValueError("Spatial motion bound requires the complete original arm/finger cubic "
                             "inside its declared joint position limits")
    if not np.isfinite(spatial_step) or spatial_step <= 0:
        raise ValueError("Spatial validation needs a finite positive observed voxel resolution")
    breaks = np.unique(np.concatenate([*(curve.x for _, curve in curves), np.asarray(anchors, float),
        (trajectory.times[:-1]+trajectory.times[1:])/2,
        trajectory.progress_u*trajectory.duration if owner_speed_per_progress else np.empty(0)]))
    if breaks[0] != 0 or breaks[-1] != trajectory.duration:
        raise ValueError("Validation anchors must lie within the unchanged physical clock")
    speed_arrays = [(names, _cubic_speed_upper(curve, breaks[:-1], breaks[1:])) for names, curve in curves]
    widths = np.diff(breaks)
    # A global bound on the actual piecewise-linear clock is sufficient for
    # external motion and remains valid at every slope switch and plateau.
    progress_rate = float(np.max(np.nextafter(np.diff(trajectory.progress_s), np.inf)
        / np.nextafter(np.diff(trajectory.progress_u)*trajectory.duration, -np.inf)))
    relative, joint_required = [], []
    for i, width in enumerate(widths):
        speed = {name: float(values[i, j]) for names, values in speed_arrays for j, name in enumerate(names)}
        body_speed = motion.speed_bounds(speed)
        moving_speed = np.nextafter(owner_speed_per_progress*progress_rate, np.inf)
        pair_speed = np.nextafter(2*max(float(body_speed.max()), moving_speed), np.inf)
        relative.append(np.nextafter(width*pair_speed, np.inf))
        arm_speed = [speed[name] for name in trajectory.joint_names]
        joint_required.append(width*np.linalg.norm(arm_speed)/budget.validation_joint_step)
    required = np.maximum(np.nextafter(np.asarray(relative)/spatial_step, np.inf), joint_required)
    if not np.isfinite(required).all():
        raise ValueError("Final curve spatial validation requires nonfinite work")
    counts = [max(1, int(np.ceil(value))) for value in required]
    total = 1+sum(counts)
    if total > budget.max_validation_points:
        raise ValueError(f"Final curve validation exceeds its declared finite sample budget: requires {total} points, allowed {budget.max_validation_points}; "
                         f"body-pair spatial step={spatial_step} m, live chunk={_VALIDATION_CHUNK_SIZE}")
    queries = np.concatenate(([0.], *[np.linspace(a, b, count+1)[1:]
        for a, b, count in zip(breaks[:-1], breaks[1:], counts)]))
    return queries, {"query_count": total, "point_budget": budget.max_validation_points,
        "chunk_size": _VALIDATION_CHUNK_SIZE, "spatial_step_m": float(spatial_step),
        "maximum_interval_relative_displacement_bound_m": float(np.max(np.asarray(relative)/counts)),
        "body_names": list(motion.body_names), "joint_names": list(motion.joint_names),
        "coefficients_m_per_joint_unit": motion.coefficients.tolist(),
        "relative_motion": "twice maximum complete-body point speed; every required self pair included",
        "scene_owner_motion": "bound external PosePath and mechanism rate plus every attachment cover",
        "owner_speed_m_per_progress": float(owner_speed_per_progress),
        "scope": "continuous-model displacement bounds between finite queries; not continuous collision proof"}


def _owner_path_speed(owner):
    """Bound any owner point's m/progress for the existing PosePath models.

    A global maximum across path pieces is conservative across all attachment
    modes. Release has zero subsequent speed; attached motion is separately
    represented as a native-tree SphereBody. Unknown custom pose samplers are
    rejected because neither a motion flag nor a few poses bounds their path.
    """
    from .planning_problem import PosePath
    from .geometry_motion_bound import _norm_up, _add_up, _mul_up
    path = owner.pose_path
    if not isinstance(path, PosePath):
        raise ValueError(f"No point-motion bound for owner {owner.owner_id!r}'s custom pose sampler")
    if (path.motion_model is None and np.array_equal(path.poses,
            np.broadcast_to(path.poses[0], path.poses.shape))):
        return 0.
    if (path.motion_model is not None and np.array_equal(path.coordinates,
            np.broadcast_to(path.coordinates[0], path.coordinates.shape))):
        # Compare unwrapped mechanism coordinates: equal endpoint poses could
        # still hide a complete revolution, whereas equal coordinates cannot.
        return 0.
    if owner.local_spheres is None:
        raise ValueError(f"Moving owner {owner.owner_id!r} lacks its complete declared sphere cover")
    rho = max(_add_up(_norm_up(row[:3]), _add_up(float(row[3]), owner.sphere_cover_error_m))
              for row in owner.local_spheres)
    lever = _add_up(rho, _norm_up(path.attachment[:3, 3]))
    rates = []
    for i, width in enumerate(np.diff(path.progress)):
        model = path.motion_model
        if model is None:
            linear = _norm_up(path.poses[i+1, :3, 3]-path.poses[i, :3, 3])
            a, b = path._quaternions[i:i+2]
            # 4*tan(theta/4) bounds both slerp and the native normalized-linear
            # small-angle branch's maximum SO(3) angular speed per fraction.
            # Quaternion chord/sum norms give tan(theta/4) without acos loss.
            b = b if np.dot(a, b) >= 0 else -b
            angular = np.nextafter(4*_norm_up(a-b)/np.linalg.norm(a+b), np.inf)
            rate = _add_up(linear, _mul_up(angular, lever))
        else:
            delta = path.coordinates[i+1]-path.coordinates[i]
            if model.kind == "revolute":
                rate = _mul_up(abs(float(delta[0])), _add_up(lever,
                    _norm_up(model.reference_pose[:3, 3]-model.axis_point)))
            elif model.kind == "prismatic":
                rate = _mul_up(abs(float(delta[0])), _norm_up(model.axis))
            elif model.kind in {"planar", "translation_plane"}:
                linear = _add_up(_mul_up(abs(float(delta[0])), _norm_up(model.basis[:, 0])),
                                 _mul_up(abs(float(delta[1])), _norm_up(model.basis[:, 1])))
                rate = linear if model.kind == "translation_plane" else _add_up(linear,
                    _mul_up(abs(float(delta[2])), lever))
            else:
                raise ValueError(f"Unsupported owner motion kind {model.kind!r}")
        rates.append(np.nextafter(rate/width/(owner.pose_interval[1]-owner.pose_interval[0]), np.inf))
    return max(rates, default=0.)


def _geometry_finger_model(hand):
    """Choose exactly the independent polynomial used by PrismaticGripper.state."""
    if hand is not None and (hand.controller == "timed_reference" or hand.nominal_finger_trajectory is not None):
        model = hand.nominal_finger_trajectory or hand.finger_trajectory
        if model is None:
            raise ValueError("Timed finger geometry requires its bound original polynomial")
        return model
    return None


def _export_motion_inputs(params, scene, problem, *, spatial_step_m):
    """Bind full native covers and timed affine fingers without a second FK."""
    cpu = lambda value: value.detach().cpu().numpy()
    spheres = cpu(params.link_spheres).reshape(-1, 4)
    indices = cpu(params.link_sphere_idx_map).reshape(-1)
    links = dict(params.link_name_to_idx_map)
    names = tuple(params.joint_names)
    hand = scene.gripper
    replaced = set(() if hand is None else hand.sphere_indices)
    bodies, extra = [], {}
    for link, index in links.items():
        selected = [i for i in np.flatnonzero(indices == index) if i not in replaced and spheres[i, 3] >= 0]
        if selected:
            bodies.append(SphereBody("native/"+link, link, spheres[selected], {}))
    if hand is not None:
        local = np.asarray(hand.local_spheres, float).copy()
        local_mapping = np.asarray(hand.sphere_joint_translation, float)
        model = _geometry_finger_model(hand)
        envelope = hand.controller == "binary" and model is None
        if envelope:
            # Reproduce the complete mechanical-box spheres used by hand.state
            # and hand.spheres. The envelope is rigid in TCP coordinates; its
            # bound says nothing about unobserved physical finger velocities.
            middle = np.asarray(hand.joint_limits).mean(axis=1)
            half = np.diff(hand.joint_limits, axis=1).reshape(-1)/2
            local[:, :3] += np.einsum("scj,j->sc", local_mapping, middle)
            local[:, 3] += np.linalg.norm(np.einsum("scj,j->sc", abs(local_mapping), half), axis=1)
        transform = np.asarray(problem.tool_transform, float)
        local[:, :3] = local[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        mapping = np.einsum("ab,sbj->saj", transform[:3, :3], local_mapping)
        bodies.append(SphereBody("unknown_hand_mechanical_envelope" if envelope else "actual_query_hand",
            problem.tool_link, local, {} if envelope else
            {name: mapping[:, :, j] for j, name in enumerate(hand.joint_names)}))
        extra = {} if envelope else {name: hand.joint_limits[j] for j, name in enumerate(hand.joint_names) if name not in names}
    owner_speed = 0.
    from scipy.spatial.transform import Rotation
    for owner in scene.owners:
        owner_speed = max(owner_speed, _owner_path_speed(owner))
        for index, relation in enumerate(owner.attachments):
            local = np.asarray(owner.local_spheres, float).copy()
            rotation = Rotation.from_quat(np.roll(relation.quaternion_tool_owner, -1)).as_matrix()
            local[:, :3] = local[:, :3] @ rotation.T + relation.position_tool_owner
            transform = np.asarray(problem.tool_transform, float)
            local[:, :3] = local[:, :3] @ transform[:3, :3].T + transform[:3, 3]
            local[:, 3] += owner.sphere_cover_error_m
            bodies.append(SphereBody(f"owner/{owner.owner_id}/attachment/{index}", problem.tool_link, local, {}))
    resolutions = []
    def collect(query):
        if hasattr(query, "coarse") and hasattr(query, "fine"):
            collect(query.coarse)
            collect(query.fine)
            return
        model = getattr(getattr(query, "scene_collision", None), "scene_model", None)
        for voxel in getattr(model, "voxel", ()) or ():
            resolutions.append(float(voxel.voxel_size))
        for solid in getattr(query, "independent_geometry", {}).get("solids", ()):
            # BEGIN source TSDF spacing also bounds formal geometric sampling.
            if solid["kind"] == "tsdf":
                resolutions.append(float(solid["voxel_size"]))
            # END the removed occupied-ESDF voxel record is not an alternate model.
    if scene.background is not None:
        collect(scene.background)
    for owner in scene.owners:
        collect(owner.query)
    if (not np.isfinite(spatial_step_m) or spatial_step_m <= 0
            or not np.isfinite(resolutions).all() or (resolutions and min(resolutions) <= 0)):
        raise ValueError("Spatial validation requires finite positive metric and voxel resolutions")
    motion = build_geometry_motion_bound(**{key: cpu(getattr(params, key)) for key in
        ("fixed_transforms", "link_map", "joint_map", "joint_map_type", "joint_offset_map")},
        joint_names=names, joint_position_limits=cpu(params.joint_limits.position),
        link_name_to_idx_map=links, bodies=bodies, extra_joint_limits=extra)
    step = min(spatial_step_m, min(resolutions)/2) if resolutions else spatial_step_m
    return motion, step, owner_speed
# END no source resampling, timing change, or physical margin is introduced.


@dataclass
class PlanningResult:
    success: bool
    trajectory: JointTrajectory | None
    diagnostics: dict
    # BEGIN bind accepted candidates to their executable input and clock.
    execution_binding: dict | None = None
    # BEGIN even a failed initialization returns the reusable same-request ledger.
    request_ledger: dict | None = None
    # END formal execution authorization remains independent of retained history.
    # END failed results never carry an execution authorization.


# BEGIN size required evidence and native anchor buffers from the actual request.
def _anchor_capacity(progress_map, anchors, budget, refinement_parameters=()):
    """Keep every required clock anchor; bound only additional refinement.

    A stopped source vertex can have two clock parameters. Count those actual
    pairs before reserving optional samples, so no source point or pause is
    removed to fit the refinement allowance.
    """
    required = len(progress_map.anchor_parameters(anchors))
    capacity = min(budget.max_validation_points, _NATIVE_STAGE_POINT_LIMIT)
    if required > capacity:
        raise ValueError("Required source anchors exceed the finite validation-point budget")
    additional = np.unique(np.asarray(refinement_parameters, float))
    refinement_limit = min(budget.max_refinement_points, capacity-required)
    if (not np.isfinite(additional).all() or np.any(additional < 0) or np.any(additional > 1)
            or len(additional) > refinement_limit):
        raise ValueError("Refinement samples exceed the finite refinement-point budget")
    return required+refinement_limit


def _validation_storage_admission(problem, scene, robot_metadata, budget):
    """Bound cumulative CPU records and full-history process geometry pre-GPU.

    A 256-state native batch does not bound retained records. Each actor/target
    decision owns one bool, six float64 bounds and six int64 stencil indices
    per body/state. Encode each bounded chunk without Python scalar lists.
    The fixed 512-MiB encoded-payload cap leaves room for candidate/capture and
    JSON copies within the local RAM budget. Full-history support/interior
    tensors have a separate 256-MiB peak cap inside the native work reserve.
    """
    from .voxel_layers import LayeredVoxelQuery
    total = budget.max_validation_points
    modes = 3 if scene.required_progress_anchors else 1
    dof = len(robot_metadata["active_joint_names"])
    # UTF-8 identifiers may expand to JSON escapes. Six times their byte
    # length bounds that expansion; fixed dictionary/shape overhead is per
    # array/chunk, never per numeric element.
    text_size = lambda value: 6*len(str(value).encode("utf-8"))
    chunk_count = (total+_VALIDATION_CHUNK_SIZE-1)//_VALIDATION_CHUNK_SIZE
    query_metadata = 4096+sum(text_size(name) for name in robot_metadata["active_joint_names"])
    targets = {"background": scene.background, **{owner.owner_id: owner.query for owner in scene.owners}}
    layered = [(name, query) for name, query in targets.items() if isinstance(query, LayeredVoxelQuery)]
    actors = {"robot": [row["body_id"] for row in scene.robot_collision_bodies]} if layered else {}
    actors.update({owner.owner_id: [owner.owner_id] for owner in scene.owners if owner.moving})
    if layered and not actors["robot"]:
        raise ValueError("Layer recording requires every robot body's explicit identity")
    pairs = [(actor, target, body_ids) for actor, body_ids in actors.items()
             for target, _ in layered if actor != target]
    encoded_bytes = 0
    for start in range(0, total, _VALIDATION_CHUNK_SIZE):
        typed = modes*min(_VALIDATION_CHUNK_SIZE, total-start)
        # Actual native arrays are float32; float64 reserves CPU oracle use
        # too, retaining exact dtype without assuming a smaller byte width.
        encoded_bytes += sum(4*((typed*8*width+2)//3) for width in (1, 1, 1, dof)) + query_metadata
        for actor, target, body_ids in pairs:
            cells = typed*len(body_ids)
            encoded_bytes += 128*cells+4*((cells+2)//3)
            encoded_bytes += 4096+text_size(actor)+text_size(target)+sum(map(text_size, body_ids))
    if encoded_bytes > _VALIDATION_RECORD_BYTE_LIMIT:
        raise ValueError(f"Complete typed collision records require at most {encoded_bytes} encoded bytes; "
                         f"allowed {_VALIDATION_RECORD_BYTE_LIMIT}; reduce declared total work or scene complexity")
    # Complete history retains scalar residuals and measurements for every
    # requirement, not merely the largest geometric temporary. Include the
    # small q/TCP/twist arrays and sorting/coordinate work used by mechanisms.
    residual_count = (1+len(problem.pose_relations)+len(problem.supports)+len(problem.interiors)
        +len(problem.pose_events)+len(problem.coordinate_events)+len(problem.couplings)
        +sum(event.stop for event in problem.events)
        +sum(1+int(item.coordinate_limits is not None)+len(item.directions) for item in problem.mechanisms))
    measurement_count = (4*len(problem.phases)+2*len(problem.mechanisms)
                         +2*len(problem.pose_events)+len(problem.coordinate_events))
    process_retained = 8*total*(4*dof+68+residual_count+measurement_count)
    process_peak = 8*total*128
    for item in problem.supports:
        body, contact, planes = len(item.body_points), len(item.contact_points), len(item.surface._hull.equations)
        # Two dense point/plane intermediates plus rotated points, quaternion
        # cross products, contacts and scalar outputs; eight bytes also covers
        # the independent float64 CPU tests. No autograd graph is retained.
        process_peak = max(process_peak, 8*total*(2*body*planes+24*body+16*contact+64))
    for item in problem.interiors:
        body, planes = len(item.body_points), len(item.halfspaces)
        process_peak = max(process_peak, 8*total*(2*body*planes+24*body+64))
    for item in problem.couplings:
        rows, columns = item.matrix.shape
        process_peak = max(process_peak, 8*(total*(4*rows+columns)+rows*columns))
    process_peak += process_retained
    if process_peak > _PROCESS_GEOMETRY_BYTE_LIMIT:
        raise ValueError(f"Full-history process tensors require at most {process_peak} bytes; "
                         f"allowed {_PROCESS_GEOMETRY_BYTE_LIMIT}; original geometry cannot be silently thinned")
    return {"encoded_record_bytes_upper": int(encoded_bytes),
        "encoded_record_byte_limit": _VALIDATION_RECORD_BYTE_LIMIT,
        "process_geometry_peak_bytes_upper": int(process_peak),
        "process_geometry_peak_byte_limit": _PROCESS_GEOMETRY_BYTE_LIMIT,
        "physical_states": total, "typed_modes": modes, "chunks": chunk_count,
        "layered_actor_target_pairs": len(pairs),
        "layered_body_records_per_typed_state": sum(len(row[2]) for row in pairs)}


# BEGIN one value carries independent host/device ownership through rebuilds.
@dataclass(frozen=True)
class _PlanningWorkspace:
    """Scene capacity plus one live query, with optional native graph pools."""

    cpu_shared_bytes: int
    cpu_live_bytes: int
    gpu_eager_bytes: int
    gpu_graph_bytes: int
    anchor_fk_bytes: int
    trajectory_graph: bool

    @property
    def cpu_bytes(self):
        return self.cpu_shared_bytes+self.cpu_live_bytes

    @property
    def gpu_bytes(self):
        return self.gpu_eager_bytes+(self.gpu_graph_bytes if self.trajectory_graph else 0)

    @property
    def total_bytes(self):
        """Diagnostic scene total; it is never a CUDA allocation request."""
        return self.cpu_bytes+self.gpu_bytes
# END graph admission changes only the two separately retained GPU pools.


def _planning_workspace_bytes(scene, robot_metadata, budget, anchor_capacity):
    """Return CPU/GPU scene capacity, anchor FK and graph eligibility.

    The main native rollout retains its original control horizon. Each trajectory
    manager evaluates the required anchors with a separate native FK model,
    then folds their penalties into that rollout's existing final cost column.
    """
    horizon = (budget.knots+4)*budget.interpolation_steps+1
    # Native LBFGS has two rollout instances and four line-search particles.
    # Include optimization/metrics, dense export and independent verification.
    # BEGIN reference-only IK owns no independent scene query workspaces.
    # Reference IK retains native pose/self-collision and physical q-domain
    # costs. Scene/coverage queries belong to complete trajectory candidates.
    # The dense/native metrics and final CPU-solid validation never construct
    # a coverage FK or its backward graph. Their other scene queries remain.
    allocation_shapes = (
        (4, budget.trajectory_seeds, horizon+anchor_capacity, True),
        (4, budget.trajectory_seeds*4, horizon+anchor_capacity, True),
        (2, budget.trajectory_seeds, budget.interpolation_buffer_size+anchor_capacity, False),
        (1, 1, min(budget.max_validation_points, _VALIDATION_CHUNK_SIZE), False),
        (2, 1, anchor_capacity, False),
    )
    # END source reference IK no longer reserves unused scene/coverage buffers.
    # BEGIN use actual sequential native forward/backward memory lifetimes.
    # Scene switches still query every exact/left/right sample. Only native
    # query buffers persist across rollouts. Installed gradient_opt_core
    # executes evaluate_action then backward(retain_graph=False) before another
    # initial/exploration rollout; metrics consume its detached best action.
    # The solver factories keep CPU metrics outside capture. Eligible owned
    # optimizer cores additionally retain two private graph pools below.
    # The final two anchor-shaped reserves also cover the simultaneous OMPL
    # chart-scene stack: at most anchor_capacity one-state scene clones, whose
    # aggregate sphere buffers fit one such reserve. Scalar slots are counted
    # below per state as well, so this split cannot hide many small caches.
    scene_factor = 3 if scene.required_progress_anchors else 1
    allocations = [(copies, scene.estimated_workspace_allocation(batch, scene_factor*length,
                        robot_coverage_horizon=length if guidance else 0))
                   for copies, batch, length, guidance in allocation_shapes]
    # BEGIN CPU residency cannot consume the GPU scene budget or graph pools.
    gpu_eager_bytes = (sum(copies*value["gpu_retained"] for copies, value in allocations)
                       + max(value["gpu_live"] for _, value in allocations)
                       + max(value["gpu_shared"] for _, value in allocations))
    cpu_shared_bytes = max(value["cpu_shared"] for _, value in allocations)
    cpu_live_bytes = max(value["cpu_live"] for _, value in allocations)
    # END geometry/query sets and the unchanged hard memory budget are preserved.
    # END all live optimization scratch remains under the unchanged hard budget.
    if gpu_eager_bytes > budget.max_scene_gpu_workspace_bytes:
        # BEGIN report the actual bound so a refused trial is reproducible.
        raise ResourceLimitError("scene_gpu_workspace", gpu_eager_bytes, budget.max_scene_gpu_workspace_bytes,
            description="Independent GPU scene query workspaces exceed the declared local allocation budget")
        # END reporting a shortfall does not alter the allocation contract.
    # BEGIN choose capture scope before allocation, preserving eager admission.
    # Each trajectory entry/inner executor retains a private pool in addition
    # to the ordinary live scratch used by warmup and metrics. A finite contact
    # scene needs CPU cost geometry. A scene whose two graph pools exceed the
    # declared capacity retains the unchanged native eager optimizer as well;
    # neither case changes required queries, arithmetic or iteration counts.
    # BEGIN CPU source validation has no allocations inside either graph pool.
    graph_live = max(value["gpu_graph_live"] for (_, value), (_, _, _, guidance)
                     in zip(allocations, allocation_shapes) if guidance)
    # END the ordinary live bound above still reserves all validation scratch.
    graph_scene_bytes = gpu_eager_bytes+2*graph_live
    trajectory_graph = not scene.contacts and graph_scene_bytes <= budget.max_scene_gpu_workspace_bytes
    # END no independent captured pool is treated as reusable eager scratch.
    dof, links, spheres = (len(robot_metadata["active_joint_names"]),
                           robot_metadata["links"], robot_metadata["spheres"])
    # Installed KinematicsFusedFunction.create_buffers uses float32 and this
    # planner requests exactly one tool frame. Per [batch, anchor] it allocates
    # pose/COM values and gradients (22 floats), sphere values/gradients (8*S),
    # a 6*D Jacobian and two D joint gradients, and 12*L cumulative transforms.
    # The same conservative rollout counts above give 22*K anchor batches.
    # Main FK, general pose/sphere transform graphs, verification, model and
    # optimizer/trajectory storage retain the separate two-GiB allowance in
    # the device-memory preflight below. The scene budget is specifically for
    # dedicated geometry queries, not a claim about all planner allocations.
    # BEGIN reserve the independent self-collision buffers at exact anchors.
    # SelfCollisionCost stores one float output, four float gradients and one
    # uint8 sparse flag per sphere, plus two int16 indices/one float per block.
    # Its smallest native block covers 2048 pairs; all unordered sphere pairs
    # bound the blocks without constructing a dense pair matrix. Production
    # costs do not request the optional per-pair diagnostic distance tensor.
    self_blocks = (spheres*(spheres-1)//2+2047)//2048
    self_bytes_per_state = 4+17*spheres+8*self_blocks
    bytes_per_state = 4*(22+8*spheres+8*dof+12*links)+self_bytes_per_state
    # END anchor collision buffers share the existing finite admission limit.
    anchor_fk_bytes = 22*budget.trajectory_seeds*anchor_capacity*bytes_per_state
    return _PlanningWorkspace(int(cpu_shared_bytes), int(cpu_live_bytes), int(gpu_eager_bytes),
                              int(2*graph_live), int(anchor_fk_bytes), bool(trajectory_graph))
    # END current device usage is checked separately before any native allocation.


def _require_planning_host_memory(workspace, *, max_host_rss_bytes):
    """Reserve live CPU work and every still-lazy, identity-deduplicated shape."""
    from .tsdf_geometry import ObservedTsdfGeometry

    require_host_memory(workspace.cpu_live_bytes,
        pending_existing_bytes=ObservedTsdfGeometry.pending_storage_total_nbytes(),
        max_rss_bytes=max_host_rss_bytes)


def _require_planning_memory(torch, device, workspace, *, max_device_used_bytes):
    """Admit Torch buffers without promising cached memory to native allocators.

    CollisionBuffer and native FK buffers use torch.zeros. Unused capacity in
    this process's Torch allocator can therefore serve those tensors after a
    schedule replacement. Existing Mapper fields are already allocated. The
    separate two-GiB allowance covers main FK, general pose/sphere transform
    graphs, solver/model and trajectory storage; it must still fit driver-
    reported free memory independently of the geometry-query tensor cache.

    CPU source geometry is already host-admitted and never enters this check.
    If the extra graph pools do not fit but the eager solver does, retain the
    existing eager path with the same geometry, samples and optimizer work.
    """
    if type(max_device_used_bytes) is not int or not 0 < max_device_used_bytes <= MAX_DEVICE_USED_BYTES:
        raise ValueError("Device admission requires an explicit limit no larger than eight GiB")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reusable_torch_bytes = max(0, torch.cuda.memory_reserved(device)-torch.cuda.memory_allocated(device))
    other_native_bytes = 2*1024**3
    # BEGIN expose only explicit capacity comparisons as resource refusals.
    description = "Available GPU memory does not cover the bounded planner, anchor FK and scene workspaces"
    if free_bytes < other_native_bytes:
        raise ResourceLimitError("gpu_native_reserve", other_native_bytes, free_bytes,
                                 description=description)
    def refusal(candidate):
        required_bytes = other_native_bytes+candidate.gpu_bytes+candidate.anchor_fk_bytes
        available_bytes = free_bytes+reusable_torch_bytes
        if available_bytes < required_bytes:
            return ResourceLimitError("gpu_planner_workspace", required_bytes, available_bytes,
                                      description=description)
        # Cached Torch pages are already included in device-used. Only their
        # unused part can satisfy future tensors without increasing that value;
        # the native reserve cannot borrow those allocator-owned pages.
        new_bytes = other_native_bytes+max(0, candidate.gpu_bytes+candidate.anchor_fk_bytes-reusable_torch_bytes)
        projected_used = total_bytes-free_bytes+new_bytes
        if projected_used > max_device_used_bytes:
            return ResourceLimitError("gpu_device_used", projected_used, max_device_used_bytes,
                                      description="Planner allocations exceed the request device-used limit")
        if new_bytes+MIN_DEVICE_FREE_BYTES > free_bytes:
            return ResourceLimitError("gpu_free_reserve", new_bytes+MIN_DEVICE_FREE_BYTES, free_bytes,
                                      description="Device memory must retain two GiB after planner allocations")
        return None

    error = refusal(workspace)
    if error is not None and workspace.trajectory_graph:
        eager = replace(workspace, trajectory_graph=False)
        eager_error = refusal(eager)
        if eager_error is None:
            return eager
        error = eager_error
    if error is not None:
        raise error
    return workspace
    # END CUDA API errors retain their original exception identity.
# END size required evidence and native anchor buffers from the actual request.


def _bounded_robot_configuration(robot, problem, scene, budget):
    """Check the actual robot layout before native tensor/solver allocation.

    The installed native full configuration loader is not CPU-only: locked
    joints and even URDF inertial metadata invoke CUDA internally.  Reuse its
    existing ``yourdfpy`` metadata parser without meshes or FK instead.  Only
    selected link/joint names and sphere counts are inspected here; native
    cuRobo still constructs all transforms and performs every FK/IK operation.
    """
    from pathlib import Path
    import yourdfpy
    from curobo.content import get_assets_path, get_robot_configs_path
    from curobo._src.util.config_io import load_yaml, join_path

    def bounded_file(path):
        # A small robot description must not cause an unbounded CPU parse
        # before its declared joint/link/sphere sizes can be inspected.
        if Path(path).stat().st_size > 4*1024**2:
            raise ValueError("Robot description files must fit the four-MiB metadata budget")
        return str(path)

    data = robot if isinstance(robot, dict) else load_yaml(
        bounded_file(join_path(get_robot_configs_path(), robot)))
    kinematics = data["robot_cfg"]["kinematics"]
    if (kinematics.get("num_envs", 1) != 1 or kinematics.get("load_meshes", False)
            or kinematics.get("load_tool_frames_with_mesh", False)):
        raise ValueError("Planning requires one robot environment and its requested tool frame without mesh allocation")
    if not kinematics.get("load_collision_spheres", True):
        raise ValueError("Complete robot collision spheres are required for planning")
    collision_links = kinematics.get("collision_link_names", [])
    extra_links = kinematics.get("extra_links") or {}
    if (len(collision_links) > budget.max_robot_links or len(extra_links) > budget.max_robot_links
            or len(set(collision_links)) != len(collision_links)):
        raise ValueError("Declared robot links exceed the bounded model or contain duplicate collision links")
    if any(not isinstance(value, dict) for value in extra_links.values()):
        raise ValueError("Robot configuration must contain YAML-style extra-link dictionaries")
    cspace = kinematics.get("cspace") or {}
    if not isinstance(cspace, dict):
        raise ValueError("Robot configuration must contain a YAML-style cspace dictionary")
    for value in cspace.values():
        if isinstance(value, (list, tuple, np.ndarray)) and (
                len(value) > budget.max_robot_dof or np.asarray(value).ndim != 1):
            raise ValueError("Robot cspace arrays exceed the bounded independent-joint layout")

    spheres = kinematics.get("collision_spheres")
    if isinstance(spheres, str):
        spheres = load_yaml(bounded_file(join_path(get_robot_configs_path(), spheres)))["collision_spheres"]
    if not isinstance(spheres, dict):
        raise ValueError("Planning requires an explicit native robot sphere dictionary")
    extra_spheres = kinematics.get("extra_collision_spheres") or {}
    if (sum(len(value) for value in spheres.values()) > budget.max_robot_spheres
            or any(type(count) is not int or count < 0 or count > budget.max_robot_spheres
                   for count in extra_spheres.values())):
        raise ValueError("Robot collision geometry exceeds the declared sphere budget before native allocation")
    # Native extra_collision_spheres replaces that link's list, including
    # inactive attachment slots.  Count its real allocation, not positive
    # radii only, before native self-collision constructs an S-by-S table.
    counts = {name: len(value) for name, value in spheres.items()}
    counts.update(extra_spheres)
    if sum(counts.values()) > budget.max_robot_spheres:
        raise ValueError("Robot attachment sphere slots exceed the declared sphere budget")
    if any(name not in counts for name in collision_links):
        raise ValueError("Every native collision link requires an explicit sphere model")
    sphere_count = sum(counts[name] for name in collision_links)
    if not 0 < sphere_count <= budget.max_robot_spheres:
        raise ValueError("Robot collision model must fit the declared nonempty sphere budget")
    if len(scene.robot_sphere_materials) != sphere_count:
        raise ValueError("Scene robot-sphere material identities must match the complete native sphere layout")

    asset_root = kinematics.get("external_asset_path") or get_assets_path()
    urdf_path = bounded_file(join_path(asset_root, kinematics["urdf_path"]))
    parsed = yourdfpy.URDF.load(urdf_path, load_meshes=False, load_collision_meshes=False,
                                build_scene_graph=False, build_collision_scene_graph=False)
    parents = {joint.child: joint.parent for joint in parsed.joint_map.values()}
    driven = {}
    for joint in parsed.joint_map.values():
        if joint.type not in ("fixed", "prismatic", "revolute", "continuous"):
            raise ValueError("Robot joints must use the native one-coordinate kinematic types")
        if joint.type != "fixed":
            driven[joint.child] = joint.name if joint.mimic is None else joint.mimic.joint
    for name, link in extra_links.items():
        parents[name] = link["parent_link_name"]
        if link.get("child_link_name") is not None:
            parents[link["child_link_name"]] = name
        if link["joint_type"] != "FIXED":
            driven[name] = link["joint_name"]

    base = kinematics["base_link"]
    if base not in parsed.link_map and base not in extra_links:
        raise ValueError("The declared robot base is absent from its URDF and extra links")
    selected = {base}
    # Match the native model's selected tree: requested tool, collision links,
    # and all explicit extra links. This checks metadata connectivity only.
    for target in [problem.tool_link, *collision_links, *extra_links]:
        link, chain = target, set()
        while link != base:
            if link in chain or link not in parents:
                raise ValueError("A selected robot link has no acyclic chain to its declared base")
            chain.add(link)
            selected.add(link)
            if len(selected) > budget.max_robot_links:
                raise ValueError("Native robot link workspaces exceed the declared link budget")
            link = parents[link]
    unlocked = {driven[link] for link in selected if link != base and link in driven}
    # Current occupied-volume evidence may request extra link poses, but it
    # may not enlarge the already bounded planning model to obtain them.
    required_links = scene.required_robot_links
    if (len(required_links) > budget.max_robot_links or len(set(required_links)) != len(required_links)
            or not set(required_links) <= selected):
        raise ValueError("Occupied robot evidence links must belong to the bounded planning model")
    if len(unlocked) > budget.max_robot_dof:
        # Native locking evaluates the complete unlocked tree once, so bound
        # that intermediate model as well as the final active joint set.
        raise ValueError("Native unlocked-joint workspaces exceed the declared DOF budget")
    locked = kinematics.get("lock_joints") or {}
    active = unlocked-set(locked)
    if active != set(problem.joint_names):
        raise ValueError("Robot active joints must equal the provided feedback joints; configure locked or driven gripper joints explicitly")

    configuration = deepcopy(data)
    configuration["robot_cfg"]["kinematics"]["tool_frames"] = [problem.tool_link]
    # Resolve the already checked geometry once, preserving native extra-slot
    # semantics and avoiding a second unbounded file read inside the solver.
    configuration["robot_cfg"]["kinematics"]["collision_spheres"] = deepcopy(spheres)
    metadata = {"active_joint_names": sorted(active), "unlocked_dof": len(unlocked),
                "links": len(selected), "selected_link_names": sorted(selected), "spheres": sphere_count}
    return configuration, metadata


@dataclass(frozen=True)
class _ProgressMap:
    """A fixed, monotone phase schedule; repeated s values encode pauses."""

    u: np.ndarray
    s: np.ndarray
    # BEGIN original intent and device activation are distinct bound facts.
    event_clock: dict = field(default_factory=dict, compare=False)
    # END only initial binding creates this record.

    def __post_init__(self):
        if (self.u.ndim != 1 or self.s.shape != self.u.shape or len(self.u) < 2
                or not np.isfinite(self.u).all() or not np.isfinite(self.s).all()
                or self.u[0] != 0 or self.u[-1] != 1 or self.s[0] < 0
                or np.any(np.diff(self.u) <= 0) or np.any(np.diff(self.s) < 0)
                or np.any(np.diff(self.u.astype(np.float32)) <= 0)):
            raise ValueError("A progress map needs increasing u and nondecreasing s")

    def anchor_parameters(self, anchors):
        """Preserve both ends of an s plateau, including a paused event."""
        pairs = []
        for value in anchors:
            equal = np.flatnonzero(self.s == value)
            if len(equal):
                pairs.append((float(self.u[equal[0]]), float(value)))
                if equal[-1] != equal[0]:
                    pairs.append((float(self.u[equal[-1]]), float(value)))
            else:
                right = int(np.searchsorted(self.s, value, side="right"))
                if right == 0 or right == len(self.s):
                    raise ValueError("An event anchor is outside the planned progress interval")
                fraction = (value - self.s[right-1]) / (self.s[right] - self.s[right-1])
                pairs.append((float(self.u[right-1] + fraction * (self.u[right] - self.u[right-1])), float(value)))
        pairs = np.asarray(sorted(set(pairs)), dtype=float).reshape(-1, 2)
        # Required source knots and event boundaries must remain distinct in
        # the native float32 process tensors.  Reject an unrepresentable clock
        # instead of silently merging or moving a supplied hard anchor.
        if any(len(np.unique(pairs[:, column])) != len(np.unique(pairs[:, column].astype(np.float32)))
               for column in (0, 1)):
            raise ValueError("Required progress anchors exceed native float32 clock precision")
        return pairs


# BEGIN source rhythm proposes the initial clock before any solver is created.
def _initial_timing(problem, budget):
    """Quantize a source-time proposal within this device's finite capacity.

    Source duration/rhythm are soft initialization information. The native
    interval cap bounds one request; an overlong demonstration may therefore
    propose a compressed clock. Hard event deadlines remain separate inputs.
    All later seed and refinement operations retain the returned clock.
    """
    source = problem.source_time_map
    seconds, progress = np.asarray(source["source_times"], float), np.asarray(source["progress"], float)
    if (seconds.ndim != 1 or seconds.shape != progress.shape or len(seconds) < 2
            or not np.isfinite(seconds).all() or np.any(np.diff(seconds) <= 0)):
        raise ValueError("Initial scheduling needs explicit increasing source seconds and their progress")
    source_duration = float(seconds[-1]-seconds[0])
    output_dt = float(np.float32(budget.interpolation_dt))
    maximum = _fixed_spline_grid(intervals=budget.knots+4,
        native_dt=budget.maximum_trajectory_dt, interpolation_steps=budget.interpolation_steps,
        interpolation_dt=budget.interpolation_dt)
    subdivisions = min(maximum["subdivisions"], max(1,
        int(np.ceil(source_duration/((budget.knots+4)*output_dt)))))
    # A power-of-two divisor preserves the existing binary32 grid contract;
    # enumerate exact realizable intervals when the subdivision count does not.
    selected = None
    for count in range(subdivisions, maximum["subdivisions"]+1):
        native_dt = float(np.float32(output_dt*count/budget.interpolation_steps))
        try:
            _fixed_spline_grid(intervals=budget.knots+4, native_dt=native_dt,
                interpolation_steps=budget.interpolation_steps, interpolation_dt=output_dt)
        except ValueError:
            continue
        selected = replace(budget, maximum_trajectory_dt=native_dt)
        break
    if selected is None:
        raise ValueError("Source-time proposal has no exact representable device clock within the budget")
    return selected, _ProgressMap((seconds-seconds[0])/source_duration, progress.copy())
# END no search path or candidate geometry can subsequently retime this binding.

# BEGIN quantize command-associated queries before freezing the binding.
def _bind_event_clock(problem, progress_map, duration, timestep=None):
    from .gripper_trajectory import _activation
    source = {}
    command_keys = []
    tick_progress = set()
    tick_entries = []
    for event in problem.events:
        identity = event_occurrence_key(event)
        parameters = progress_map.anchor_parameters([event.progress])[:, 0]
        intent = float(parameters[0]*duration)
        command = event.kind in {"open", "close", "hold"}
        needs_tick = command or event.requires_evidence is not None or event.stop
        if needs_tick and timestep is None:
            raise ValueError("Command timing requires the bound device control timestep")
        step, applied = _activation(intent, timestep) if needs_tick else (None, None)
        if needs_tick:
            if applied > duration or (command and applied == duration):
                raise ValueError("A bound command has no activation/effect coverage inside the valid trajectory")
            tick_entries.append((event, step, float(parameters[0]), applied/duration))
        source[identity] = {"event": asdict(event), "intent_time_s": intent, "activation_time_s": applied if command else None,
                            "consumption_time_s": applied if needs_tick and not command else None}
        if command:
            source[identity].update(bound_activation_step=step, request_activation_step=step,
                request_activation_time_s=applied, request_intent_time_s=intent,
                request_time_origin_step=0, device_timestep_s=float(timestep))
    # Multiple ordinary intentions on the same actuator tick share the
    # controller's existing last-intention policy. Preserve every identity;
    # supersession here is a schedule prediction, not an application fact.
    by_step = {}
    for entry in tick_entries:
        if entry[0].kind in {"open", "close", "hold"}:
            by_step.setdefault(entry[1], []).append(entry)
    superseded = set()
    true_goals = {goal.progress for name in ("pose_events", "coordinate_events", "supports")
                  for goal in getattr(problem, name)}
    for entries in by_step.values():
        ordered = sorted(entries, key=lambda item: (item[2], item[0].source_frame))
        for event, _, _, _ in ordered[:-1]:
            if (event.stop or event.requires_evidence is not None or event.progress in true_goals
                    or any(item.check_target == "activation" for item in event.timing)):
                raise ValueError("Device quantization supersedes a command with a required physical occurrence")
            key = event_occurrence_key(event)
            superseded.add(key)
            source[key]["scheduled_superseded_by"] = event_occurrence_key(ordered[-1][0])
    for event, _, before_u, after_u in tick_entries:
        if event_occurrence_key(event) not in superseded:
            command_keys.append((before_u, after_u))
            tick_progress.add(event.progress)
    # Independent functional events retain their own bound physical instant.
    # Only queries explicitly sharing a command's progress move to its tick;
    # ordinary reference samples between these anchors remain interpolated.
    independent = [event.progress for event in problem.events if event.progress not in tick_progress]
    independent += [goal.progress for name in ("pose_events", "coordinate_events", "supports")
                    for goal in getattr(problem, name) if goal.progress not in tick_progress]
    fixed = [(float(value), float(value)) for value in
             progress_map.anchor_parameters(independent)[:, 0]] if independent else []
    moves = dict([(0., 0.), (1., 1.)]+fixed+command_keys)
    before = np.asarray(sorted(moves))
    after = np.asarray([moves[value] for value in before])
    if after[0] != 0 or after[-1] != 1 or np.any(np.diff(after) <= 0):
        raise ValueError("Device quantization merges or reverses required command times")
    # A required event can lie inside a linear reference interval. Include
    # its old time explicitly before moving it, otherwise endpoint-only
    # interpolation could silently leave the event at the unquantized time.
    keys = np.union1d(progress_map.u, before)
    mapping = _ProgressMap(np.interp(keys, before, after), np.interp(keys, progress_map.u, progress_map.s))
    for event in problem.events:
        record = source[event_occurrence_key(event)]
        parameters = mapping.anchor_parameters([event.progress])[:, 0]
        record.update(time=(record["intent_time_s"] if record["activation_time_s"] is not None
                            else (record["consumption_time_s"] if record["consumption_time_s"] is not None
                                  else float(parameters[0]*duration))),
                      hold_until=float(parameters[-1]*duration))
    return _ProgressMap(mapping.u, mapping.s, source)
# END OMPL/refinement/export only read this record and never repeat quantization.


# BEGIN hard time requirements inspect the final tick-bound schedule.
def _validate_bound_event_times(problem, progress_map, duration, origin_s=0., history=(),
                                goal_identities=None, timing_events=()):
    """Check explicit windows against actual history or this tick-bound proposal.

    Consumed nodes may still own an unresolved time relation. Their recorded
    witness survives while a future origin is supplied by this candidate.
    Only the latest selected occurrence of an original event owns the current
    completion obligation; older attempts remain unchanged in the ledger.
    """
    from .planning_problem import PlanningEvent, TimeBindingError, resolve_event_window
    goal_identities = ({name: list(range(len(getattr(problem, name))))
                       for name in ("pose_events", "coordinate_events", "supports")}
                      if goal_identities is None else goal_identities)
    current = {event_occurrence_key(event): event for event in problem.events}
    all_events = {event_occurrence_key(row): PlanningEvent(**row) for row in timing_events}
    all_events.update(current)
    latest = {}
    for key in all_events:
        latest[key[:2]] = max(latest.get(key[:2], -1), key[2])
    predicted, planned, checks = [], {}, []

    def candidate_fact(key, target, value, **identity):
        predicted.append({"skill_id": key[0], "event_id": key[1], "occurrence": key[2],
                          "check_target": target, "time_s": origin_s+value, **identity})

    # A declared future origin can be another current command, goal arrival
    # or effect observation node even when the origin has no own time window.
    for key, event in current.items():
        record = progress_map.event_clock[key]
        if record.get("activation_time_s") is not None:
            candidate_fact(key, "activation", record["activation_time_s"])
        for collection in ("pose_events", "coordinate_events", "supports"):
            for index, goal in enumerate(getattr(problem, collection)):
                if abs(goal.progress-event.progress) <= 1e-7:
                    candidate_fact(key, "arrival", float(progress_map.anchor_parameters([goal.progress])[0, 0]*duration),
                                   goal_id=[collection, goal_identities[collection][index]])
        for condition in getattr(problem, "conditions", ()):
            if (condition.get("skill_id"), condition.get("event_id")) == key[:2] and condition.get("effect_requirement"):
                candidate_fact(key, "effect", record["time"], condition_id=condition["condition_id"])
        for constraint in event.timing:
            if constraint.check_target == "activation":
                value = record.get("activation_time_s")
                if value is None:
                    raise TimeBindingError("An activation window requires a real device command activation")
            elif constraint.check_target == "arrival":
                collection, original_index = constraint.goal_id
                if original_index not in goal_identities[collection]:
                    raise TimeBindingError("An arrival requirement refers to a goal absent from this occurrence")
                goal = getattr(problem, collection)[goal_identities[collection].index(original_index)]
                value = float(progress_map.anchor_parameters([goal.progress])[0, 0]*duration)
            else:
                value = record["time"]
            identity = {"goal_id": None if constraint.goal_id is None else list(constraint.goal_id),
                        "condition_id": constraint.condition_id}
            candidate_fact(key, constraint.check_target, value, **identity)
            planned[(key, constraint.check_target, constraint.goal_id, constraint.condition_id)] = origin_s+value

    for key, event in all_events.items():
        if key[2] != latest[key[:2]]:
            continue
        for constraint in event.timing:
            facts = [row for row in history if event_occurrence_key(row) == key and row.get("value") == "T"
                and row.get("check_target") == constraint.check_target
                and (constraint.goal_id is None or tuple(row.get("goal_id", ())) == constraint.goal_id)
                and (constraint.condition_id is None or row.get("condition_id") == constraint.condition_id)]
            value = min(row["time_s"] for row in facts) if facts else planned.get(
                (key, constraint.check_target, constraint.goal_id, constraint.condition_id))
            # A pending observed effect can be assessed again from fresh actual
            # data. An absent consumed activation/arrival witness cannot be
            # reconstructed from a demonstration or its former scheduled time.
            if value is None:
                if constraint.check_target != "effect":
                    raise TimeBindingError("A consumed timed event lacks its actual occurrence witness")
                value = origin_s
            earliest, latest_time = resolve_event_window(constraint, history, predicted_origins=predicted)
            if constraint.check_target == "effect":
                # An old true state observed before this effect window is
                # history, not a completion witness for the timed obligation.
                applicable = [row for row in facts if (earliest is None or row["time_s"] >= earliest-1e-9)
                              and (latest_time is None or row["time_s"] <= latest_time+1e-9)]
                facts = applicable
                value = (min(row["time_s"] for row in facts) if facts else max(origin_s,
                    planned.get((key, constraint.check_target, constraint.goal_id, constraint.condition_id), origin_s),
                    origin_s if earliest is None else earliest))
            if latest_time is not None and value > latest_time+1e-9:
                raise TimeBindingError(f"Event {key} {constraint.check_target} exceeds its original physical deadline")
            if constraint.check_target != "effect" and earliest is not None and value < earliest-1e-9:
                raise TimeBindingError(f"Event {key} {constraint.check_target} precedes its explicit physical window")
            checks.append({"event": key, "check_target": constraint.check_target,
                "scheduled_request_time_s": value, "window_s": [earliest, latest_time],
                "status": "requires_actual_effect_evidence" if constraint.check_target == "effect" and not facts
                          else "actual_time_inside_window" if facts else "scheduled_time_inside_window"})
    return checks

# END computation time is accounted separately and cannot shift these origins.



# BEGIN same-request continuation consumes facts and the original clock.
class ContinuationBindingError(ValueError):
    """Remaining work lacks facts, budget or a compatible original clock."""


# BEGIN one ledger accounts for successful, failed and rebound work alike.
_PLANNING_COUNTS = ("binding_attempts", "initializations", "ik_batches", "ik_queries", "search_queries",
                    "trajectory_solves", "validation_passes", "validation_queries")
_PLANNING_SECONDS = ("search_seconds", "planning_seconds")


def _obligation_contract(problem):
    """Keep original semantic identities and thresholds, allowing fresh geometry.

    Current poses, attachment transforms and observed surfaces are rebindable
    inputs. A new observation cannot erase a goal/process, reorder its original
    index, or relax a declared functional threshold under the same request ID.
    """
    from .execution_binding import plain
    columns = {
        "pose_events": ("progress", "source", "position_tolerance_m", "rotation_tolerance_rad"),
        "coordinate_events": ("progress", "source", "target_coordinate", "tolerance", "kind", "coordinate_mode", "coordinate_origin"),
        "supports": ("progress", "source", "margin", "gap_tolerance", "normal_tolerance"),
        "pose_relations": ("start", "end", "source", "pose_bounds"),
        "mechanisms": ("end", "source", "directions"),
        "interiors": ("start", "end", "source", "margin"),
        "couplings": ("joint_names", "matrix", "target", "source"),
    }
    contract = {}
    for collection, names in columns.items():
        contract[collection] = [{"type": type(item).__name__, **{name: plain(getattr(item, name)) for name in names},
            # BEGIN rebinding cannot turn an owner-relative goal into a world goal.
            **({"actual_relation": (None if item.actual_relation is None else
                {key: plain(value) for key, value in item.actual_relation.items()
                 if not key.startswith("T_")})} if collection == "pose_events" else {}),
            # END current local transforms remain rebindable geometric inputs.
            **({"kind": item.model.kind} if collection == "mechanisms" else {})}
            for item in getattr(problem, collection, ())]
    contract["conditions"] = [{name: plain(row[name]) for name in
        ("condition_id", "kind", "skill_id", "event_id", "effect_requirement", "requires_evidence") if name in row}
        for row in getattr(problem, "conditions", ())]
    contract["source_time_map"] = {name: plain(getattr(problem, name, None))
                                   for name in ("source_timestamps", "source_progress")}
    contract["source_phases"] = [{"source": plain(phase.reference.source),
                                  "skill_id": phase.metadata.get("skill_id")}
                                 for phase in getattr(problem, "phases", ())]
    return contract


def _restore_original_direction_origins(problem, contract):
    """Retain actual net-change baselines without freezing fresh mechanism poses.

    A compiler can derive an entrance baseline from its freshly observed
    initial coordinate. That is correct only for a new request. For the same
    request, recover the original evidenced baseline when every direction
    instruction is unchanged. An explicitly different baseline is rejected
    by the semantic contract instead of silently accepted as a new goal.
    """
    from .execution_binding import plain
    saved_mechanisms = contract["mechanisms"]
    if not saved_mechanisms or len(saved_mechanisms) != len(getattr(problem, "mechanisms", ())):
        return
    for mechanism, saved in zip(problem.mechanisms, saved_mechanisms):
        if len(mechanism.directions) != len(saved["directions"]):
            continue
        directions = []
        for direction, original in zip(mechanism.directions, saved["directions"]):
            current = plain(direction)
            semantics = lambda row: {name: value for name, value in row.items()
                                     if name not in {"origin_coordinate", "origin_source"}}
            automatic = (direction.start == mechanism.start
                and direction.origin_coordinate == float(mechanism.initial_coordinates[direction.coordinate_index])
                and plain(direction.origin_source) == plain(mechanism.source))
            if (direction.kind == "net" and original["origin_coordinate"] is not None
                    and semantics(current) == semantics(original)
                    and (direction.origin_coordinate is None or automatic)):
                direction = replace(direction, origin_coordinate=original["origin_coordinate"],
                                    origin_source=deepcopy(original["origin_source"]))
            directions.append(direction)
        mechanism.directions = tuple(directions)


class PlanningWorkExhausted(ValueError):
    """The original request has no remaining admitted computation budget."""


def _request_work_ledger(problem, budget, previous=None, continuation=None):
    """Copy an append-only request record, never mutate a caller's failed result."""
    from .execution_binding import plain
    ledger = ({"request_id": uuid4().hex, "planning_budget": asdict(budget),
        "requires_continuation": False,
        "consumed_planning": {**dict.fromkeys(_PLANNING_COUNTS, 0),
                              **dict.fromkeys(_PLANNING_SECONDS, 0.)},
        "original_events": plain(problem.events), "obligation_contract": _obligation_contract(problem),
        "history": [], "event_records": [], "timing_events": [], "timing_checks": [],
        "completed_goal_ids": [], "goal_evidence": [], "completed_direction_ids": [], "direction_evidence": [],
        "effects": {}, "effect_history": [], "binding_records": []}
        if previous is None else deepcopy(previous))
    if (not isinstance(ledger.get("request_id"), str) or not ledger["request_id"]
            or type(ledger.get("requires_continuation")) is not bool
            or ledger.get("planning_budget") != asdict(budget)):
        raise ContinuationBindingError("A same-request ledger cannot reset its identity or planning budget")
    if not isinstance(ledger.get("obligation_contract"), dict):
        raise ContinuationBindingError("The request ledger lacks its original goal/process identity contract")
    consumed = ledger.get("consumed_planning", {})
    for name in (*_PLANNING_COUNTS, *_PLANNING_SECONDS):
        value = consumed.get(name)
        if (not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
                or (name in _PLANNING_COUNTS and type(value) is not int)):
            raise ContinuationBindingError("The request ledger needs every cumulative planning counter")
    for name in ("original_events", "history", "event_records", "timing_events", "timing_checks", "completed_goal_ids", "goal_evidence",
                 "completed_direction_ids", "direction_evidence", "effect_history", "binding_records"):
        if not isinstance(ledger.get(name), list):
            raise ContinuationBindingError(f"The request ledger lacks its append-only {name}")
    if continuation is not None:
        original = continuation["original_binding"]
        old = original["request_ledger"]
        if old["requires_continuation"] and not ledger["requires_continuation"]:
            raise ContinuationBindingError("A request cannot erase its actual execution continuation requirement")
        if old["request_id"] != ledger["request_id"]:
            raise ContinuationBindingError("Execution and planning ledgers belong to different requests")
        if (ledger["original_events"] != old["original_events"]
                or ledger["obligation_contract"] != old["obligation_contract"]):
            raise ContinuationBindingError("A request ledger cannot rewrite original source or goal/process identities")
        if not {tuple(value) for value in old["completed_goal_ids"]} <= {tuple(value) for value in ledger["completed_goal_ids"]}:
            raise ContinuationBindingError("A request ledger cannot erase historical completed goals")
        if not {tuple(value) for value in old["completed_direction_ids"]} <= {tuple(value) for value in ledger["completed_direction_ids"]}:
            raise ContinuationBindingError("A request ledger cannot erase historical completed directions")
        if any(consumed[name] < old["consumed_planning"][name] for name in consumed):
            raise ContinuationBindingError("Cumulative planning work must retain all prior attempts")
        for name in ("history", "event_records", "timing_events", "timing_checks", "goal_evidence", "direction_evidence", "effect_history", "binding_records"):
            if ledger[name][:len(old[name])] != old[name]:
                raise ContinuationBindingError(f"The request ledger rewrote prior {name}")
        # Append actual execution facts once; a historical T is retained here,
        # not promoted to currently applicable scene evidence in a new binding.
        for name in ("history", "timing_checks", "goal_evidence", "completed_goal_ids", "completed_direction_ids", "direction_evidence", "effect_history"):
            for row in continuation.get(name, ()):
                if row not in ledger[name]:
                    ledger[name].append(deepcopy(row))
        for name in ("applied_events", "superseded_events", "evaluated_events", "stopped_events", "reached_events", "verified_events"):
            for row in continuation.get(name, ()):
                record = {"record_kind": name, **deepcopy(row)}
                if record not in ledger["event_records"]:
                    ledger["event_records"].append(record)
        ledger["effects"].update(deepcopy(continuation.get("effects", {})))
        ledger["requires_continuation"] = True
    return ledger


def _check_work_budget(budget, consumed, *, started, initial_seconds, stage=None):
    """Check at stage boundaries; report real elapsed overrun, never pretend preemption."""
    consumed["planning_seconds"] = initial_seconds+time.monotonic()-started
    cap = budget.maximum_trajectory_solves
    if consumed["planning_seconds"] >= budget.planning_seconds:
        raise PlanningWorkExhausted("The original cumulative planning wall-time budget is exhausted")
    stage_caps = {"binding_attempts": cap, "initializations": cap, "ik_batches": cap,
        "ik_queries": budget.max_validation_points*budget.ik_seeds*cap,
        "trajectory_solves": cap,
        "validation_passes": cap*budget.trajectory_seeds,
        "validation_queries": budget.max_validation_points*cap*budget.trajectory_seeds+cap,
        "search_queries": budget.max_validation_points*cap}
    for name, limit in stage_caps.items():
        if consumed[name] > limit or (stage == name and consumed[name] >= limit):
            raise PlanningWorkExhausted(f"The original cumulative {name} budget is exhausted")
# END derived capacities include every declared native candidate, without new services.


# BEGIN completion evidence is checked before removing any original goal.
def _verified_completed_goals(continuation, original, all_goals, *, elapsed,
                              curve_elapsed, timestep, progress_map):
    """Consume concrete actual-goal records, never a bare completion label.

    The frozen execution binding owns each goal's geometry and physical clock.
    A freshly rebound target can move, so current observations cannot rewrite
    a previous candidate's proof. Previously accepted proofs remain in the
    request ledger; new pose proofs additionally retain their actual poses and
    metric errors, which are recomputed against the original allowed set here.
    """
    from .motion import _pose_array
    from .planning_problem import EventTimeConstraint, PoseEventRequirement, resolve_event_window, TimeBindingError
    from .gripper_trajectory import _activation
    from .execution import _actual_pose_goal

    def identities(rows):
        if (not isinstance(rows, (tuple, list)) or any(
                not isinstance(row, (tuple, list)) or len(row) != 2
                or not isinstance(row[0], str) or type(row[1]) is not int
                for row in rows)):
            raise ContinuationBindingError("Completed goals need original collection/index identities")
        values = {tuple(row) for row in rows}
        if not values <= all_goals:
            raise ContinuationBindingError("Completed goal identity does not belong to the original request")
        return values

    previous = identities(original.get("completed_goal_ids", []))
    claimed = identities(continuation.get("completed_goal_ids", []))
    ledger = original["request_ledger"]
    if not previous <= identities(ledger["completed_goal_ids"]):
        raise ContinuationBindingError("Prior completed goals lack their preserved request ledger")
    for identity in previous:
        if not any(tuple(row.get("identity", ())) == identity
                   and row.get("value") == "T" and row.get("completion_verified") is True
                   and isinstance(row.get("candidate_id"), str) and row["candidate_id"]
                   for row in ledger["goal_evidence"]):
            raise ContinuationBindingError("Prior completed goals lack their preserved actual T evidence")
    new = claimed-previous
    if not new:
        return previous | claimed
    feedback = continuation["actual_feedback"]
    timestamp = feedback.get("timestamp_s")
    if (isinstance(timestamp, bool) or not isinstance(timestamp, (float, int))
            or not np.isfinite(timestamp)):
        raise ContinuationBindingError("Goal completion needs synchronized actual continuation time")
    origin = float(timestamp)-elapsed
    active_events = {}
    for event in (*ledger["timing_events"], *original["curve"].get("events", ())):
        event_id = event_occurrence_key(event)
        if event_id[:2] not in active_events or event_id[2] >= event_occurrence_key(active_events[event_id[:2]])[2]:
            active_events[event_id[:2]] = event
    for identity in new:
        proofs = [row for row in continuation.get("goal_evidence", ())
                  if isinstance(row, dict) and tuple(row.get("identity", ())) == identity]
        verified = False
        for row in proofs:
            if (row.get("candidate_id") != original["candidate_id"] or row.get("value") != "T"
                    or row.get("completion_verified") is not True
                    or any(not isinstance(row.get(key), str) or not row[key].strip()
                           for key in ("source", "feedback_id"))):
                continue
            times = [row.get(key) for key in ("timestamp_s", "physical_elapsed_s", "evaluated_curve_time_s")]
            if (any(isinstance(value, bool) or not isinstance(value, (float, int))
                    or not np.isfinite(value) for value in times)):
                continue
            acquired, physical, curve_time = times
            if (not 0 <= physical <= elapsed+1e-8 or not 0 <= curve_time <= curve_elapsed+1e-8
                    or abs(acquired-origin-physical) > 1e-8):
                continue
            goals = original["problem"][identity[0]]
            indices = original.get("goal_identities", {}).get(identity[0], list(range(len(goals))))
            if identity[1] not in indices:
                continue
            requirement = goals[indices.index(identity[1])]
            if row.get("requirement") != requirement or not row.get("measurement_sources"):
                continue
            if identity[0] == "pose_events":
                scheduled = row.get("scheduled_curve_time_s")
                expected = float(progress_map.anchor_parameters([requirement["progress"]])[0, 0]
                                 * original["curve"]["duration"])
                if (not isinstance(scheduled, (float, int)) or not np.isfinite(scheduled)
                        or abs(scheduled-expected) > 1e-8
                        or row.get("device_due_time_s") != _activation(scheduled, timestep)[1]):
                    continue
                # An explicit arrival window permits an observed member at
                # any original physical time in that window. Without one,
                # only the first native acquisition of the bound point counts.
                arrivals = [item for event in active_events.values() for item in event.get("timing", ())
                            if item["check_target"] == "arrival" and tuple(item["goal_id"]) == identity]
                if arrivals:
                    try:
                        windows = [resolve_event_window(EventTimeConstraint(**item),
                            continuation.get("history", ())) for item in arrivals]
                    except (TypeError, ValueError, TimeBindingError):
                        continue
                    lower = max((value for value, _ in windows if value is not None), default=None)
                    upper = min((value for _, value in windows if value is not None), default=None)
                    request_time = float(original.get("request_time_origin_s", 0.))+physical
                    if (row.get("evaluation_window_s") != [lower, upper]
                            or (lower is not None and request_time < lower-1e-9)
                            or (upper is not None and request_time > upper+1e-9)):
                        continue
                else:
                    native_due = _activation(scheduled, timestep)[1]
                    # BEGIN a passed dependency extends the same frozen curve.
                    # executable_until is its initial prerequisite prefix, not
                    # a permanent end: actual T may authorize later intervals.
                    # U/F ends this execution after a hold, so the acquisition
                    # still uses the original native tick and final curve cap.
                    first_tick = min(native_due, original["curve"]["duration"])
                    if (row.get("evaluation_window_s") is not None
                            or abs(curve_time-first_tick) > 1e-8 or abs(physical-native_due) > 1e-8):
                        continue
                    # END no initial-prefix clamp discards a later actual goal proof.
                try:
                    # Reconstruct the original raw measurements, so two
                    # mutually consistent but shifted result poses cannot
                    # override a fixed target or fabricate an owner pose.
                    sources = row["measurement_sources"]
                    if not isinstance(sources, list) or not all(isinstance(item, dict) for item in sources):
                        continue
                    measured_feedback = {"timestamp_s": acquired, "feedback_id": row["feedback_id"]}
                    observation = {"scene_owner_poses": {}, "scene_condition_evidence": {}}
                    for item in sources:
                        if not isinstance(item.get("source"), str) or not item["source"].strip():
                            raise ValueError("Raw actual measurements require provenance")
                        when = float(item["timestamp_s"])
                        if not np.isfinite(when):
                            raise ValueError("Raw actual measurement timestamps must be finite")
                        if item["kind"] == "actual_robot_tcp":
                            if abs(when-acquired) > 1e-8 or item["feedback_id"] != row["feedback_id"]:
                                raise ValueError("Actual TCP measurement uses another feedback sample")
                            pose = _pose_array(item["pose"], "Measured TCP pose").tolist()
                            if "tool_pose" in measured_feedback and measured_feedback["tool_pose"] != pose:
                                raise ValueError("Actual TCP measurements disagree")
                            # BEGIN continuation preserves the original TCP estimate provenance.
                            measured_feedback.update(tool_pose=pose, source=item["source"],
                                tool_pose_source=item["pose_source"],
                                tool_pose_is_independent_observation=item["independent_observation"])
                            # END replay cannot relabel model FK as an independent observation.
                        elif item["kind"] in ("current_owner_pose", "current_verified_attachment"):
                            policy = original["execution_limits"]["observation_policy"]
                            # BEGIN current-state freshness matches the executor's observation policy.
                            # A still-current owner/holding sample may predate
                            # this binding; only event/effect occurrence proof
                            # has a separate binding-origin eligibility rule.
                            if (policy is None or when > acquired+1e-8
                                    or acquired-when > policy["maximum_age_s"]+1e-8):
                                raise ValueError("Current owner/attachment evidence is not applicable")
                            # END future or expired state observations remain inadmissible.
                            if ("timestamp_s" in observation and
                                    (observation["timestamp_s"] != when or observation["source"] != item["source"])):
                                raise ValueError("Current owner measurements lack one synchronized observation")
                            observation.update(timestamp_s=when, source=item["source"])
                            if item["kind"] == "current_owner_pose":
                                pose = _pose_array(item["pose"], "Measured owner pose").tolist()
                                previous_pose = observation["scene_owner_poses"].get(item["owner_id"])
                                if previous_pose is not None and previous_pose != pose:
                                    raise ValueError("Current owner measurements disagree")
                                observation["scene_owner_poses"][item["owner_id"]] = pose
                            else:
                                if item["value"] is not True:
                                    raise ValueError("Attachment observation is not verified")
                                observation["scene_condition_evidence"][item["condition_id"]] = True
                        else:
                            raise ValueError("Pose evidence contains an unsupported measurement source")
                    actual_check = _actual_pose_goal(PoseEventRequirement(**requirement), measured_feedback,
                        observation if "timestamp_s" in observation else None)
                    if actual_check["value"] != "T":
                        continue
                    if any(not np.allclose(_pose_array(row.get(key), key), actual_check[key], atol=1e-10, rtol=0)
                           for key in ("actual_function_pose", "actual_target_pose")):
                        continue
                    position, rotation = actual_check["position_error_m"], actual_check["rotation_error_rad"]
                    recorded = np.asarray([row.get("position_error_m"), row.get("rotation_error_rad")], float)
                except (KeyError, TypeError, ValueError):
                    continue
                if (not np.isfinite(recorded).all()
                        or not np.allclose(recorded, [position, rotation], atol=1e-10, rtol=0)
                        or position > requirement["position_tolerance_m"]
                        or rotation > requirement["rotation_tolerance_rad"]):
                    continue
            verified = True
            break
        if not verified:
            raise ContinuationBindingError("Goal completion needs actual allowed-set T evidence from this executed candidate")
    return previous | claimed
# END effect success and source progress cannot erase a distinct task goal.


def _prepare_continuation(problem, scene, budget, continuation, *, rebind=None):
    """Validate a fresh actual/scene binding without replaying consumed events.

    A continuation is the executor's existing result record. It is not a
    request to restore the old actual state or guess past task success. The
    original absolute progress, event identities and cumulative solve budget
    survive; angular history and past goal effects require supplied facts.
    """
    from .execution_binding import plain
    from .planning_problem import event_occurrence_key
    if rebind is not None and (not isinstance(rebind, dict)
            or rebind.get("kind") not in {"remaining", "retry"} or not rebind.get("source")):
        raise ContinuationBindingError("An explicit rebind needs remaining/retry kind and its executor source")
    if not isinstance(continuation, dict) or not isinstance(continuation.get("original_binding"), dict):
        raise ContinuationBindingError("Continuation requires its original formal execution binding")
    original = continuation["original_binding"]
    if continuation.get("candidate_id") != original.get("candidate_id"):
        raise ContinuationBindingError("Continuation candidate identity differs from its original binding")
    # BEGIN physical window origins retain their original actual history.
    previous_history = original["request_ledger"]["history"]
    history = continuation.get("history", [])
    if (not isinstance(history, (list, tuple))
            or list(history[:len(previous_history)]) != previous_history):
        raise ContinuationBindingError("Continuation cannot erase or rewrite original physical timing history")
    # END a later repeated fact cannot move an already observed window origin.
    _restore_original_direction_origins(problem, original["request_ledger"]["obligation_contract"])
    if _obligation_contract(problem) != original["request_ledger"]["obligation_contract"]:
        raise ContinuationBindingError("A fresh binding cannot remove, reorder or change original goal/process obligations")
    if "planning_budget" not in original or "consumed_planning" not in original:
        raise ContinuationBindingError("Continuation lacks the original cumulative planning budget")
    request_budget = PlanningBudget(**original["planning_budget"])
    if budget is not None and asdict(budget) != asdict(request_budget):
        raise ContinuationBindingError("A continuation cannot reset or enlarge its original planning budget")
    consumed = deepcopy(continuation.get("consumed_planning", original["consumed_planning"]))
    for key in (*_PLANNING_COUNTS, *_PLANNING_SECONDS):
        value = consumed.get(key)
        if (not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
                or value < original["consumed_planning"].get(key, 0)):
            raise ContinuationBindingError("Cumulative planning work must retain all prior attempts")
    if type(consumed["trajectory_solves"]) is not int:
        raise ContinuationBindingError("Consumed trajectory solves must be an integer count")
    # Actual simulation/device steps, including holds, belong to the original
    # skill as well. A fresh binding cannot manufacture a fresh step budget.
    physics_cap = continuation.get("physics_budget")
    physics_consumed = continuation.get("cumulative_physics_steps")
    physics_remaining = continuation.get("remaining_physics_steps")
    if (type(physics_cap) is not int or physics_cap <= 0
            or type(physics_consumed) is not int or physics_consumed < 0
            or type(physics_remaining) is not int or physics_remaining < 0
            or physics_consumed < original.get("consumed_physics_steps", 0)
            or physics_cap != original.get("physics_budget")
            or physics_remaining > max(0, physics_cap-physics_consumed)):
        raise ContinuationBindingError("Continuation must retain its original cumulative physics-step budget")
    if physics_remaining == 0:
        raise ContinuationBindingError("The current skill's cumulative physics-step budget is exhausted")
    local_steps = continuation.get("consumed_physics_steps")
    # BEGIN physical time uses confirmed steps; attempts consume budget only.
    confirmed_steps = continuation.get("confirmed_physics_steps")
    accounting_exact = continuation.get("step_accounting_exact")
    # END no legacy charge-as-time interpretation is accepted implicitly.
    timestep = continuation.get("device_timestep_s")
    if (type(local_steps) is not int or local_steps < 0
            # BEGIN uncertain clock records cannot bind a new physical suffix.
            or type(confirmed_steps) is not int or not 0 <= confirmed_steps <= local_steps
            or accounting_exact is not True
            # END a known zero-step failed attempt may still carry one charge.
            or not isinstance(timestep, (float, int)) or not np.isfinite(timestep) or timestep <= 0
            or physics_consumed != original.get("consumed_physics_steps", 0)+local_steps):
        raise ContinuationBindingError("Continuation requires cumulative charges, exact confirmed steps and the unchanged native device timestep")
    remaining_solves = request_budget.maximum_trajectory_solves-consumed["trajectory_solves"]
    if remaining_solves <= 0:
        raise ContinuationBindingError("The current skill's cumulative trajectory-solve budget is exhausted")
    remaining = replace(request_budget, maximum_trajectory_solves=remaining_solves,
                        search_seconds=max(0., request_budget.search_seconds-consumed["search_seconds"]))
    feedback = continuation.get("actual_feedback", {})
    # BEGIN failed final capture cannot be replaced by stale or absent feedback.
    if not isinstance(feedback, dict):
        raise ContinuationBindingError("Continuation needs captured synchronized actual feedback")
    # END the failed attempt's conservative charge remains in the request.
    names = list(feedback.get("joint_names", ()))
    if not names or len(set(names)) != len(names) or not set(problem.joint_names) <= set(names):
        raise ContinuationBindingError("Continuation needs synchronized named actual q/dq feedback")
    try:
        columns = [names.index(name) for name in problem.joint_names]
        actual_q, actual_dq = np.asarray(feedback["q"], float)[columns], np.asarray(feedback["dq"], float)[columns]
    except (KeyError, IndexError, ValueError) as error:
        raise ContinuationBindingError("Continuation has incomplete actual joint feedback") from error
    if not np.array_equal(problem.start_q, actual_q) or not np.array_equal(problem.start_dq, actual_dq):
        raise ContinuationBindingError("Fresh planning actual differs from the executor's synchronized continuation feedback")
    hand = scene.gripper
    if hand is not None:
        if hand.control_timestep_s != timestep:
            raise ContinuationBindingError("Continuation cannot change the original native device control timestep")
        if not set(hand.joint_names) <= set(names) or not np.array_equal(hand.current_positions,
                np.asarray(feedback["q"], float)[[names.index(name) for name in hand.joint_names]]):
            raise ContinuationBindingError("Continuation hand model is not bound to the current actual fingers")
    curve = original.get("curve", {})
    duration, actual_elapsed = curve.get("duration"), continuation.get("elapsed_physical_s")
    # BEGIN advance the original clock only by verified native integration.
    elapsed = confirmed_steps*float(timestep)
    # END budget reservations never synthesize physical elapsed time.
    if (not isinstance(actual_elapsed, (int, float)) or not np.isfinite(actual_elapsed)
            or abs(actual_elapsed-elapsed) > 1e-8):
        raise ContinuationBindingError("Actual elapsed physics time differs from its native device step ledger")
    if (not isinstance(duration, (int, float)) or not isinstance(elapsed, (int, float))
            or not np.isfinite([duration, elapsed]).all() or elapsed < 0 or duration <= 0
            or (rebind is None and elapsed >= duration)):
        raise ContinuationBindingError("The remaining original physical clock is exhausted or unavailable")
    old_u, old_s = np.asarray(curve.get("progress_u"), float), np.asarray(curve.get("progress_s"), float)
    try:
        original_map = _ProgressMap(old_u, old_s)
    except ValueError as error:
        raise ContinuationBindingError("Continuation has no complete frozen progress map") from error
    curve_elapsed = elapsed if rebind is None else continuation.get("last_curve_time_s")
    if not isinstance(curve_elapsed, (int, float)) or not np.isfinite(curve_elapsed) or not 0 <= curve_elapsed <= duration:
        raise ContinuationBindingError("Rebinding requires the last actually executed curve time, separately from hold time")
    start_progress = float(np.interp(curve_elapsed/duration, original_map.u, original_map.s))
    if start_progress >= problem.phase_count and rebind is None:
        raise ContinuationBindingError("The original skill progress is finished; unresolved effects need assessment or an explicit same-goal retry")
    request_events = deepcopy(original["request_ledger"]["original_events"])
    source_events = {(row["skill_id"], row["event_id"]): row for row in request_events}
    current_events = {(event.skill_id, event.event_id): event for event in problem.events}
    if set(current_events) != set(source_events):
        raise ContinuationBindingError("Fresh continuation must retain the original skill and event identities before filtering")
    for identity, event in current_events.items():
        current, source = plain(event), deepcopy(source_events[identity])
        current.pop("occurrence")
        source.pop("occurrence")
        if current != source:
            raise ContinuationBindingError("Continuation changed an original event/source/goal association")
    remaining_rows = continuation.get("remaining_events")
    if not isinstance(remaining_rows, (list, tuple)):
        raise ContinuationBindingError("Continuation lacks its consumed/unconsumed event ledger")
    pending_ids = {event_occurrence_key(row) for row in remaining_rows}
    old_occurrences = {event_occurrence_key(row): row for row in curve.get("events", ())}
    old_rows = {(row["skill_id"], row["event_id"]): row for row in curve.get("events", ())}
    if not pending_ids <= set(old_occurrences):
        raise ContinuationBindingError("Remaining event ledger does not belong to the original candidate")
    applied = set(tuple(value) for value in original.get("consumed_events", ()))
    for row in (*continuation.get("applied_events", ()), *continuation.get("superseded_events", ())):
        applied.add(event_occurrence_key(row))
    if applied & pending_ids:
        raise ContinuationBindingError("An applied command cannot be replayed as an unconsumed event")
    evaluated = {event_occurrence_key(row) for row in continuation.get("evaluated_events", ())}
    stopped = {event_occurrence_key(row) for row in continuation.get("stopped_events", ())}
    reached = {event_occurrence_key(row) for row in continuation.get("reached_events", ())}
    witnessed = applied | evaluated | stopped | reached | {
        event_occurrence_key(row) for row in continuation.get("verified_events", ())}
    # Ordinary effect evaluation consumes its observation node even when
    # the value is F/U. It never claims a successful effect or a held object.
    for identity in evaluated:
        event = current_events.get(identity[:2])
        if (event is None or event.kind != "effect" or event.requires_evidence is not None
                or (event.stop and identity not in stopped)):
            raise ContinuationBindingError("An evaluation ledger entry is not an ordinary effect node")
        pending_ids.discard(identity)
    for identity in reached:
        event = current_events.get(identity[:2])
        if event is None or event.kind != "function" or event.requires_evidence is not None:
            raise ContinuationBindingError("A reached-node ledger entry is not an independent functional event")
        pending_ids.discard(identity)
    consumed_events = (set(old_occurrences)-pending_ids) | applied
    if not consumed_events <= witnessed:
        raise ContinuationBindingError("Consumed events need actual application or verification ledger entries")
    for identity in pending_ids:
        row = old_occurrences[identity]
        due = row.get("activation_time_s") if row.get("activation_time_s") is not None else row["time"]
        if due < elapsed and rebind is None:
            raise ContinuationBindingError(f"Unconsumed event {identity} is overdue on the frozen schedule; an explicit rebind is required")
    # BEGIN consumed goals require the executor's actual allowance/time proof.
    all_goals = {(name, index) for name in ("pose_events", "coordinate_events", "supports")
                 for index in range(len(getattr(problem, name)))}
    completed_goals = _verified_completed_goals(continuation, original, all_goals,
        elapsed=elapsed, curve_elapsed=curve_elapsed, timestep=float(timestep), progress_map=original_map)
    # END an ID, progress traversal, or observed effect cannot consume a pose goal.
    # Actual direction completion is not implied by traversing source progress.
    all_directions = {(index, column) for index, mechanism in enumerate(problem.mechanisms)
                      for column in range(len(mechanism.directions))}
    completed_directions = {tuple(value) for value in original.get("completed_direction_ids", ())}
    claimed_directions = {tuple(value) for value in continuation.get("completed_direction_ids", ())}
    if not (completed_directions | claimed_directions) <= all_directions:
        raise ContinuationBindingError("A direction completion must retain its original mechanism/direction identity")
    for identity in claimed_directions-completed_directions:
        direction = problem.mechanisms[identity[0]].directions[identity[1]]
        rows = [row for row in continuation.get("direction_evidence", ())
                if tuple(row.get("identity", ())) == identity and row.get("candidate_id") == original["candidate_id"]
                and row.get("value") == "T" and row.get("scope_complete") is True
                and row.get("completion_verified") is True
                and isinstance(row.get("source"), str) and row["source"].strip()]
        # BEGIN completion consumes the executor's covered actual scope.
        # An observer's raw scope_complete label is retained as evidence but
        # cannot alone turn a short or stale interval into a completed guard.
        # Net displacement needs its original coordinate origin and endpoint;
        # monotone/rate obligations additionally cover the whole observed scope.
        declared_net_origin = (direction.kind == "net" and direction.origin_coordinate is not None
                               and bool(direction.origin_source))
        verified = False
        for row in rows:
            interval = np.asarray(row.get("acquisition_interval_s"), float)
            scope_start, scope_end = row.get("scope_started_at_s"), row.get("scope_ended_at_s")
            if (interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] > interval[1]
                    or not isinstance(scope_end, (int, float)) or not np.isfinite(scope_end)
                    or interval[1] < scope_end-1e-8):
                continue
            scope_covered = (isinstance(scope_start, (int, float)) and np.isfinite(scope_start)
                             and scope_start <= scope_end and interval[0] <= scope_start+1e-8)
            if declared_net_origin or scope_covered:
                verified = True
                break
        if not verified:
            raise ContinuationBindingError("Direction completion needs actual scoped T evidence from this executed candidate")
        # END continuation cannot upgrade an observer's unverified completion label.
    completed_directions.update(claimed_directions)
    # BEGIN explicit rebinding preserves obligations but rebuilds the soft schedule.
    retry_ids, retry_goals, retry_directions = set(), set(), set()
    if rebind is not None and rebind["kind"] == "retry":
        retry_ids = {tuple(value) for value in rebind.get("retry_event_ids", ())}
        retry_goals = {tuple(value) for value in rebind.get("retry_goal_ids", ())}
        retry_directions = {tuple(value) for value in rebind.get("retry_direction_ids", ())}
        all_goals = {(name, index) for name in ("pose_events", "coordinate_events", "supports")
                     for index in range(len(getattr(problem, name)))}
        if (not retry_ids <= set(source_events) or not retry_goals <= all_goals or not retry_directions <= all_directions
                or not (retry_ids or retry_goals or retry_directions)):
            raise ContinuationBindingError("A same-goal retry must select original event/goal identities")
    selected_events = {}
    for identity in pending_ids:
        selected_events[identity[:2]] = replace(current_events[identity[:2]], occurrence=identity[2])
    for identity in retry_ids:
        if any(key[:2] == identity for key in pending_ids):
            raise ContinuationBindingError("Retry selects consumed occurrences; pending work uses remaining rebind")
        occurrences = [key[2] for key in (*old_occurrences, *applied) if key[:2] == identity]
        occurrences.extend(event_occurrence_key(row)[2] for row in continuation.get("history", ())
                           if event_occurrence_key(row)[:2] == identity)
        selected_events[identity] = replace(current_events[identity], occurrence=max(occurrences, default=0)+1)
    remaining_goal_ids = {(name, index) for name in ("pose_events", "coordinate_events", "supports")
        for index in range(len(getattr(problem, name))) if (name, index) not in completed_goals or (name, index) in retry_goals}
    if rebind is not None:
        required_progress = [event.progress for event in selected_events.values()]
        required_progress += [getattr(problem, name)[index].progress for name, index in remaining_goal_ids]
        required_progress += [problem.mechanisms[index].directions[column].start for index, column in retry_directions]
        if required_progress:
            start_progress = min(start_progress, min(required_progress))
        # A terminal unresolved effect/goal needs a nonempty new source window;
        # taking the preceding source interval proposes timing only, not q or a
        # re-execution of already consumed commands.
        if start_progress >= problem.phase_count:
            start_progress = float(problem.source_progress[-2])
    # END original frame/progress associations remain unchanged across occurrences.
    for collection in ("pose_events", "coordinate_events", "supports"):
        for index, requirement in enumerate(getattr(problem, collection)):
            if rebind is None and requirement.progress < start_progress and (collection, index) not in completed_goals:
                raise ContinuationBindingError(f"Past goal {collection}:{index} needs actual completion evidence")
    for index, requirement in enumerate(problem.mechanisms):
        for column, direction in enumerate(requirement.directions):
            if (direction.end < start_progress and (index, column) not in completed_directions
                    and (index, column) not in retry_directions):
                raise ContinuationBindingError(f"Past direction {index}:{column} needs actual scoped completion evidence or an explicit retry")
        if any(direction.kind == "net" and direction.origin_coordinate is None
               and direction.start < start_progress <= direction.end for direction in requirement.directions):
            raise ContinuationBindingError(f"mechanism:{index} net direction needs its actually observed original scope-start coordinate")
        if (requirement.end >= start_progress and requirement.angular_column is not None
                and requirement.start != start_progress
                and not requirement.model.diagnostics.get("initial_coordinates_source")
                and requirement.model.diagnostics.get("initial_coordinate_convention")
                    != "unique actual angular branch within current physical limits"):
            raise ContinuationBindingError(f"mechanism:{index} needs current observed lifted coordinates and a fresh actual-start scope")
    # Goal ledger identities always use the original request's collection
    # indices. Filtering a remaining suffix must never renumber a proof.
    goal_identities = {collection: [index for index, item in enumerate(getattr(problem, collection))
                                  if item.progress >= start_progress and (collection, index) in remaining_goal_ids]
                       for collection in ("pose_events", "coordinate_events", "supports")}
    direction_identities = [[(index, column) for column in range(len(item.directions))]
                            for index, item in enumerate(problem.mechanisms) if item.end >= start_progress]
    rebound = deepcopy(problem)
    rebound.start_progress = start_progress
    rebound.events = tuple(deepcopy(event) for event in selected_events.values())
    # BEGIN a measured angular branch anchors this binding's actual suffix.
    # The compiler validates this explicit lifted coordinate against fresh
    # observed pose. Move only the mechanism's coordinate-history entrance;
    # original direction scopes, net origins and entry-delta targets remain.
    # Mutating the private object also preserves coordinate-goal references
    # to this same mechanism instead of creating two inconsistent lift scopes.
    for mechanism in rebound.mechanisms:
        if (mechanism.angular_column is not None and mechanism.start < start_progress <= mechanism.end
                and (mechanism.model.diagnostics.get("initial_coordinates_source")
                     or mechanism.model.diagnostics.get("initial_coordinate_convention")
                        == "unique actual angular branch within current physical limits")):
            mechanism.start = start_progress
    # END reference angles cannot supply this actual branch evidence.
    for collection in ("pose_events", "coordinate_events", "supports"):
        setattr(rebound, collection, tuple(item for index, item in enumerate(getattr(rebound, collection))
                                           if index in goal_identities[collection]))
    for collection in ("pose_relations", "mechanisms", "interiors"):
        setattr(rebound, collection, tuple(item for item in getattr(rebound, collection) if item.end >= start_progress))
    source_time = problem.progress_to_source_time(start_progress)
    selected = problem.source_progress > start_progress
    rebound.source_timestamps = np.r_[source_time, problem.source_timestamps[selected]]
    rebound.source_progress = np.r_[start_progress, problem.source_progress[selected]]
    rebound.__post_init__()
    # A hand factory consumes only this candidate's clock event records. Keep
    # the scene's declared command identities in that same selected set; its
    # original source policy is validated before changing occurrence numbers.
    if hand is not None:
        policies = {(row["skill_id"], row["event_id"]): row for row in hand.commands}
        commands = []
        for event in rebound.events:
            if event.kind not in {"open", "close", "hold"}:
                continue
            policy = policies.get((event.skill_id, event.event_id))
            if policy is None or policy["command"] != event.kind or policy["progress"] != event.progress:
                raise ContinuationBindingError("Rebinding cannot invent or change an original hand command policy")
            commands.append({**deepcopy(policy), "occurrence": event.occurrence})
        hand.commands = tuple(commands)
    # Only the private binding problem is filtered after validation succeeds.
    # Hand factories consume event metadata from the same frozen clock.
    for item in fields(problem):
        if item.init:
            setattr(problem, item.name, getattr(rebound, item.name))
    return remaining, {"original_duration": float(duration), "elapsed": float(elapsed),
        # BEGIN event-clock origins count only confirmed physical advancement.
        "actual_elapsed": float(actual_elapsed), "elapsed_device_steps": confirmed_steps,
        # END accumulated planning/execution budgets retain their separate charges.
        "device_timestep_s": float(timestep),
        "u": original_map.u.copy(), "s": original_map.s.copy(),
        "events": {identity: deepcopy(old_occurrences[identity]) for identity in pending_ids},
        "request_budget": asdict(request_budget), "consumed_planning": consumed,
        "physics_budget": physics_cap, "consumed_physics_steps": physics_consumed,
        "remaining_physics_steps": physics_remaining,
        "request_events": request_events, "consumed_events": sorted(consumed_events),
        "goal_identities": goal_identities,
        "retry_goal_ids": sorted(retry_goals),
        "direction_identities": direction_identities, "retry_direction_ids": sorted(retry_directions),
        "completed_direction_ids": sorted(completed_directions),
        "rebind": deepcopy(rebind),
        "scene_anchors": np.asarray(scene.required_progress_anchors, float).copy() if rebind is not None else None,
        "control_timestep_s": timestep if rebind is not None else None,
        "history": deepcopy(continuation.get("history", ())),
        "timing_events": deepcopy(continuation.get("request_ledger", original["request_ledger"])["timing_events"]),
        "device_policy": deepcopy(continuation.get("device_policy")),
        "active_finger_interval": deepcopy(continuation.get("active_finger_interval")),
        "completed_goal_ids": sorted(completed_goals), "effects": deepcopy(continuation.get("effects", {})),
        "previous_request_time_origin_s": float(original.get("request_time_origin_s", 0.)),
        "request_time_origin_s": float(original.get("request_time_origin_s", 0.))+float(elapsed),
        "request_time_origin_step": int(original.get("request_time_origin_step", 0))+local_steps}


def _continuation_timing(problem, budget, continuation):
    """Use the original suffix, or rebuild every clock consumer for an explicit rebind."""
    # BEGIN explicit rebind is a new synchronized binding with original time origins.
    if continuation.get("rebind") is not None:
        selected, mapping = _initial_timing(problem, budget)
        anchors = np.unique(np.r_[problem.anchors, continuation["scene_anchors"]])
        anchors = anchors[anchors >= problem.start_progress]
        mapping, _ = _progress_for_cubic_boundaries(mapping, selected.knots, problem.start_dq,
            stop_progress=problem.stop_progress, anchors=anchors)
        grid = _fixed_spline_grid(intervals=selected.knots+4, native_dt=selected.maximum_trajectory_dt,
            interpolation_steps=selected.interpolation_steps, interpolation_dt=selected.interpolation_dt)
        mapping = _bind_event_clock(problem, mapping, grid["duration"], continuation["control_timestep_s"])
        for record in mapping.event_clock.values():
            record["request_intent_time_s"] = continuation["request_time_origin_s"]+record["intent_time_s"]
            if record.get("activation_time_s") is not None:
                record["request_time_origin_step"] = continuation["request_time_origin_step"]
                record["request_activation_step"] = continuation["request_time_origin_step"]+record["bound_activation_step"]
                record["request_activation_time_s"] = continuation["request_time_origin_s"]+record["activation_time_s"]
        _validate_bound_event_times(problem, mapping, grid["duration"],
            continuation["request_time_origin_s"], continuation["history"], continuation["goal_identities"],
            continuation["timing_events"])
        return selected, mapping
    # END ordinary same-binding continuation retains its already frozen schedule.
    remaining = continuation["original_duration"]-continuation["elapsed"]
    proposal = SimpleNamespace(source_time_map={
        "source_times": np.array([0., remaining]),
        "progress": np.array([problem.start_progress, float(problem.phase_count)])})
    selected, _ = _initial_timing(proposal, budget)
    grid = _fixed_spline_grid(intervals=selected.knots+4,
        native_dt=selected.maximum_trajectory_dt, interpolation_steps=selected.interpolation_steps,
        interpolation_dt=selected.interpolation_dt)
    if grid["duration"] < remaining:
        raise ContinuationBindingError("The remaining frozen schedule exceeds the original native capacity")
    original_times = continuation["u"]*continuation["original_duration"]
    keep = original_times > continuation["elapsed"]
    seconds = np.r_[0., original_times[keep]-continuation["elapsed"]]
    progress = np.r_[problem.start_progress, continuation["s"][keep]]
    if seconds[-1] < grid["duration"]:
        seconds, progress = np.r_[seconds, grid["duration"]], np.r_[progress, progress[-1]]
    clock = {}
    for identity, event in continuation["events"].items():
        record = {name: deepcopy(event[name]) for name in ("time", "hold_until", "intent_time_s", "activation_time_s", "consumption_time_s") if name in event}
        for name in ("time", "hold_until", "intent_time_s", "activation_time_s", "consumption_time_s"):
            if record.get(name) is not None:
                record[name] = float(record[name])-continuation["elapsed"]
        record["request_intent_time_s"] = (event.get("request_intent_time_s")
            if event.get("request_intent_time_s") is not None else
            float(event.get("intent_time_s", event["time"]))+continuation["previous_request_time_origin_s"])
        if event.get("activation_time_s") is not None:
            timestep = continuation["device_timestep_s"]
            if event.get("device_timestep_s") != timestep:
                raise ContinuationBindingError("A pending command changed its bound device timestep")
            step = event["request_activation_step"]-continuation["request_time_origin_step"]
            if step < 0:
                raise ContinuationBindingError("A pending command missed its original device activation tick")
            record.update(activation_time_s=step*timestep, bound_activation_step=step,
                request_activation_step=event["request_activation_step"],
                request_activation_time_s=event["request_activation_time_s"],
                request_time_origin_step=continuation["request_time_origin_step"], device_timestep_s=timestep,
                time=max(0., record["time"]))
            if record["request_intent_time_s"] == record["request_activation_time_s"]:
                record["time"] = record["intent_time_s"] = record["activation_time_s"]
        elif record.get("time", 0.) < 0:
            raise ContinuationBindingError("A pending event missed its original physical time")
        clock[identity] = record
        record["event"] = asdict(next(event for event in problem.events if event_occurrence_key(event) == identity))
    return selected, _ProgressMap(seconds/grid["duration"], progress, clock)
# END a new actual binding retains past results and original pending clocks.


# BEGIN native-cubic scalar clock with bounded, source-preserving linearization.
_PROGRESS_PARAMETER_ERROR = 1e-6
_MAX_PROGRESS_KEYS = 16384


def _progress_for_cubic_boundaries(progress_map, knots, start_velocity, *, stop_progress, anchors):
    """Return the saved progress map and its shared original-path parameter.

    Native cubic motion has fixed boundary derivatives. A linear reference
    clock demanding nonzero progress speed right up to a stopped boundary
    cannot generally fit that basis. Build a monotone scalar clock in the
    same uniform cubic basis instead: ``z(new_u)`` parameterizes the ORIGINAL
    path, and the returned skill progress is ``old_s(z(new_u))``. OMPL joint
    initializers must use that same z, including inside old progress pauses.

    Four equal coefficients create a complete constant cubic interval with
    zero first/second derivatives on either side. Only explicit verification
    stops add these intervals; ordinary path vertices remain ordinary points.
    Every original path vertex and mandatory progress anchor is retained.

    The saved representation stays piecewise linear. For a uniform cubic,
    ``|d2z/d(native_t)2| <= max|diff(coefficients, 2)|``. Splitting each native
    interval into L pieces bounds linearization error by D2/(8*L**2). The
    normalized-parameter bound 1e-6 is NOT a physical skill tolerance. Native
    and exported process checks both consume this same saved finite map.
    """
    if type(knots) is not int or not 1 <= knots <= 24:
        raise ValueError("The scalar progress clock requires the bounded native control horizon")
    stop_progress = np.unique(np.asarray(stop_progress, float))
    stop_pairs = progress_map.anchor_parameters(stop_progress)
    # First arrival owns the verification event. Any original s plateau's
    # other endpoint and all intervening q-path points are retained below;
    # adding the clock must not delete motion in a conditional search seed.
    stops = np.asarray([stop_pairs[stop_pairs[:, 1] == value, 0][0] for value in stop_progress])
    stops = np.unique(stops[(stops > 0) & (stops < 1)])
    boundaries = np.r_[0., stops, 1.]
    count = len(stops)
    # Each segment needs a positive increment, and each interior stop needs
    # three additional copies of its reached value: four equal coefficients.
    if knots < 4*count+1:
        raise ValueError("The native control horizon cannot represent every required verification stop")
    remaining = knots-4*count-1
    shares = remaining*np.diff(boundaries)
    increments = 1+np.floor(shares).astype(int)
    order = np.argsort(-(shares-np.floor(shares)), kind="stable")
    increments[order[:remaining-int(np.floor(shares).sum())]] += 1
    moving_start = not np.all(np.asarray(start_velocity) == 0)
    # V2's actual cubic boundary coefficients for p=0, a=0 are
    # [-v,0,v,2v]. A positive scalar v starts progress immediately while the
    # actual robot still receives its own original measured joint velocity.
    initial_rate = boundaries[1]/(increments[0]+2) if moving_start else 0.
    prefix = np.array([-initial_rate, 0., initial_rate, 2*initial_rate])
    free = []
    for index, (left, right, steps) in enumerate(zip(boundaries[:-1], boundaries[1:], increments)):
        segment = (np.arange(3, steps+3, dtype=float)*initial_rate
                   if index == 0 and moving_start else np.linspace(left, right, steps+1)[1:])
        segment[-1] = right
        free.extend(segment)
        if index < count:
            free.extend([right]*3)
    coefficients = np.r_[prefix, free, np.ones(3)]
    intervals = knots+4
    spline = BSpline(np.arange(-3., knots+8.), coefficients, 3)
    second_derivative_bound = float(np.abs(np.diff(coefficients, n=2)).max())
    # Reserve one subdivision beyond the ceiling so replacing a regular
    # sample by a nearby required float32 key has a strict error margin.
    subdivisions = max(1, int(np.ceil(np.sqrt(second_derivative_bound/(8*_PROGRESS_PARAMETER_ERROR))))+1)
    grid_count = intervals*subdivisions+1
    if grid_count > _MAX_PROGRESS_KEYS:
        raise ValueError("The scalar progress clock exceeds its 16384-key linearization budget")
    native_t = np.arange(grid_count, dtype=float)/subdivisions
    parameter = spline(native_t)
    # Constant spans have an exact mathematical value. Writing that value
    # avoids tiny numerical oscillations from summing equal B-spline terms.
    for interval in range(intervals):
        local = coefficients[interval:interval+4]
        if np.all(local == local[0]):
            parameter[interval*subdivisions:(interval+1)*subdivisions+1] = local[0]
    parameter[0], parameter[-1] = 0., 1.
    warp = _ProgressMap(native_t/intervals, parameter)

    source_pairs = progress_map.anchor_parameters(np.unique(np.r_[np.asarray(anchors, float), stop_progress]))
    retained_parameters = np.unique(np.r_[progress_map.u, source_pairs[:, 0]])
    inverse_pairs = warp.anchor_parameters(retained_parameters)
    # Required source keys own their native float32 value. Only an auxiliary
    # regular-grid key may be replaced when both become the same native key;
    # anchor_parameters already rejects aliases between two required keys.
    auxiliary = ~np.isin(warp.u.astype(np.float32), inverse_pairs[:, 0].astype(np.float32))
    new_u = np.unique(np.r_[warp.u[auxiliary], inverse_pairs[:, 0]])
    if len(new_u) > _MAX_PROGRESS_KEYS:
        raise ValueError("Source-preserving progress clock exceeds its 16384-key total budget")
    parameter = np.interp(new_u, warp.u, warp.s)
    parameter[np.searchsorted(new_u, inverse_pairs[:, 0])] = inverse_pairs[:, 1]
    # The difference of the old and new piecewise-linear maps is linear on
    # their common refinement, so its maximum occurs at one of their keys.
    # Add this exact finite-map perturbation to the cubic interpolation bound.
    perturbation = max(float(np.abs(np.interp(warp.u, new_u, parameter)-warp.s).max()),
                       float(np.abs(np.interp(new_u, warp.u, warp.s)-parameter).max()))
    if second_derivative_bound/(8*subdivisions**2)+perturbation > _PROGRESS_PARAMETER_ERROR:
        raise ValueError("Required progress keys exceed the scalar clock's linearization error bound")
    warp = _ProgressMap(new_u, parameter)
    progress = np.interp(warp.s, progress_map.u, progress_map.s)
    # Inverse/forward floating-point interpolation can differ by one final
    # digit. The original supplied progress values retain their exact identity.
    supplied = dict(zip(progress_map.u, progress_map.s))
    supplied.update(dict(source_pairs))
    progress = np.asarray([supplied.get(value, current) for value, current in zip(warp.s, progress)])
    return _ProgressMap(warp.u.copy(), progress), warp
# END native-cubic scalar clock with bounded, source-preserving linearization.


def _torch_interpolate(keys, values, query):
    """Small tensor interpolation for the fixed progress schedule only."""
    import torch
    right = torch.searchsorted(keys, query.contiguous(), right=True).clamp(1, len(keys)-1)
    alpha = ((query-keys[right-1]) / (keys[right]-keys[right-1])).clamp(0, 1)
    return values[right-1] + alpha * (values[right]-values[right-1])


def _hermite_at_parameters(position, velocity, dt, counts, parameters):
    """Evaluate q and dq at exact u values without interpolating FK poses.

    The native degree-three samples include every polynomial breakpoint.
    Adjacent q/dq Hermite evaluation is therefore exact up to floating-point
    arithmetic.  Counts are per seed and exclude padded final-state entries.
    """
    import torch
    batch, _, dof = position.shape
    coordinates = parameters * (counts[:, None]-1)
    left = torch.minimum(coordinates.floor().long(), counts[:, None]-2).clamp_min(0)
    fraction = (coordinates-left).unsqueeze(-1)
    gather = left.unsqueeze(-1).expand(batch, -1, dof)
    q0, q1 = position.gather(1, gather), position.gather(1, gather+1)
    v0, v1 = velocity.gather(1, gather), velocity.gather(1, gather+1)
    delta = dt.reshape(batch, 1, 1)
    a = fraction
    q = ((2*a**3-3*a**2+1)*q0 + (a**3-2*a**2+a)*delta*v0
         + (-2*a**3+3*a**2)*q1 + (a**3-a**2)*delta*v1)
    dq = (((6*a**2-6*a)*q0 + (-6*a**2+6*a)*q1)/delta
          + (3*a**2-4*a+1)*v0 + (3*a**2-2*a)*v1)
    return q, dq


# BEGIN project more tightly than every existing mechanism acceptance band.
def _projection_tolerance(problem):
    """Set native projection accuracy without changing equalities or acceptance."""
    bounds = [problem.numerical.position_m, problem.numerical.rotation_rad, problem.numerical.joint]
    for relation in problem.mechanisms:
        model = relation.model
        bounds.extend((model.position_tolerance, model.rotation_tolerance))
    return min(bounds)*.1
# END project more tightly than every existing mechanism acceptance band.


# BEGIN combine completed OMPL term flags before the single host decision.
def _ompl_terms_accepted(terms):
    """Keep every existing FP comparison and synchronize one boolean only.

    OMPL constructs all term tensors before acceptance. They share the state
    device; collecting their finite/nonpositive flags changes no floating
    operation, sum, tolerance, or report value. Boolean AND is exact, unlike
    reassociating a floating-point reduction. The empty collection keeps
    Python ``all``'s identity, although production also supplies scene/task
    terms. Only the final scalar crosses the device/host boundary.
    """
    import torch
    flags = [torch.isfinite(term).all() & (term <= 0).all() for term in terms.values()]
    return bool(torch.stack(flags).all().item()) if flags else True
# END no floating-point term is recomputed, combined, or rounded here.


def _ompl_stage(start, goal, lower, upper, valid, equality, *, tolerance,
                step, seconds, sample_limit, query_parameters=()):
    """Run native directed RRTConnect in (q,u=t/T), retaining checked edge samples.

    ``equality(x, derivative)`` returns true equality residuals or their
    Jacobian.  An empty residual selects ordinary OMPL space; inequality
    bands are checked by ``valid`` and are never collapsed to center lines.
    The same exporter accepts edges during tree growth and reconstructs the
    solution, including the native geodesic's final delta-length tail.
    """
    from ompl import base as ob, geometric as og

    dimension = len(start)
    query_parameters = np.unique(np.asarray(query_parameters, float))
    if (not np.isfinite(query_parameters).all() or np.any(query_parameters < lower[-1])
            or np.any(query_parameters > upper[-1])):
        raise ValueError("OMPL query times must lie in the frozen binding interval")
    deadline = time.monotonic()+seconds
    residual = np.asarray(equality(start, False), dtype=float)
    count = len(residual)
    report = {"space": "projected" if count else "ordinary", "equalities": count,
              "motion_checks": 0, "validity_checks": 0, "rejections": {}}
    if count >= dimension:
        return None, {**report, "failure": "equality_codimension_not_below_state_dimension"}

    class ProcessConstraint(ob.Constraint):
        def __init__(self):
            super().__init__(dimension, count, tolerance)

        def function(self, values, output):
            output[:] = equality(np.asarray(values), False)

        def jacobian(self, values, output):
            output[:] = equality(np.asarray(values), True)

    constraint = ProcessConstraint() if count else None
    if count:
        # Newton projection requires locally independent equality rows.  A
        # degenerate system is a bounded seed-search failure, never permission
        # to silently remove a process equality.
        for endpoint in (start, goal):
            jacobian = np.asarray(equality(endpoint, True))
            if not np.isfinite(jacobian).all() or np.linalg.matrix_rank(jacobian, tol=1e-7) < count:
                return None, {**report, "failure": "singular_equality_jacobian"}

    ambient = ob.RealVectorStateSpace(dimension)
    bounds = ob.RealVectorBounds(dimension)
    for index, (lo, hi) in enumerate(zip(lower, upper)):
        bounds.setLow(index, float(lo))
        bounds.setHigh(index, float(hi))
    ambient.setBounds(bounds)
    space = ob.ProjectedStateSpace(ambient, constraint) if count else ambient
    if count:
        space.setDelta(step*2)
    information = ob.ConstrainedSpaceInformation(space) if count else ob.SpaceInformation(space)
    information.setStateValidityCheckingResolution(min(.05, step/np.linalg.norm(upper-lower)))

    def vector(state):
        return np.asarray([state[index] for index in range(dimension)], dtype=float)

    def valid_array(value):
        report["validity_checks"] += 1
        reason = None
        if time.monotonic() > deadline:
            reason = "time_budget"
        elif report["validity_checks"] > 2*sample_limit:
            reason = "sample_budget"
        elif not np.isfinite(value).all():
            reason = "nonfinite_state"
        elif np.any(value < lower) or np.any(value > upper):
            reason = "state_bounds"
        elif constraint is not None and not constraint.isSatisfied(value):
            reason = "equality"
            if "first_equality_rejection" not in report:
                remaining = np.asarray(equality(value, False), dtype=float)
                report["first_equality_rejection"] = {
                    "progress": float(value[-1]), "tolerance": float(tolerance),
                    "l2": float(np.linalg.norm(remaining)), "max_abs": float(np.max(np.abs(remaining)))}
        elif not valid(value):
            reason = "process_or_scene"
        if reason is not None:
            report["last_rejection"] = reason
            report["rejections"][reason] = report["rejections"].get(reason, 0)+1
            return False
        return True

    information.setStateValidityChecker(lambda state: valid_array(vector(state)))
    information.setup()
    native_validator = information.getMotionValidator()
    accepted_edges = {}

    def edge_key(first, last):
        return first.tobytes(), last.tobytes()

    def checked_edge(a, b):
        first, last = vector(a), vector(b)
        key = edge_key(first, last)
        if key in accepted_edges:
            return accepted_edges[key]
        if (last[-1] < first[-1] or (last[-1] == first[-1] and not np.array_equal(last, first))
                or not valid_array(first) or not valid_array(last)):
            return None
        if not native_validator.checkMotion(a, b):
            return None
        # Installed OMPL does not bind discreteGeodesic's pair return type.
        # Its native path container exposes exactly that geodesic instead.
        path = og.PathGeometric(information)
        path.append(a)
        path.append(b)
        path.interpolate()
        vertices = np.asarray([vector(path.getState(i)) for i in range(path.getStateCount())])
        if (len(vertices) > sample_limit or not np.array_equal(vertices[0], first)
                or not np.array_equal(vertices[-1], last)):
            return None
        checked = [first]
        for left, right in zip(vertices[:-1], vertices[1:]):
            subdivisions = max(1, int(np.ceil(np.linalg.norm(right-left)/step)))
            if len(checked)+subdivisions > sample_limit:
                return None
            # BEGIN every crossed event/plateau boundary owns an exact time query.
            if right[-1] < left[-1] or (right[-1] == left[-1] and not np.array_equal(right, left)):
                return None
            fractions = np.arange(1, subdivisions+1, dtype=float)/subdivisions
            required = query_parameters[(query_parameters > left[-1]) & (query_parameters < right[-1])]
            if len(required):
                fractions = np.unique(np.r_[fractions, (required-left[-1])/(right[-1]-left[-1])])
            if len(checked)+len(fractions) > sample_limit:
                return None
            for fraction in fractions:
                point = right.copy() if fraction == 1. else left+(right-left)*fraction
                # Projection is allowed to change q, never the crossed query's time.
                if fraction != 1. and constraint is not None:
                    point = _project_boundary(point, equality, tolerance, deadline=deadline)
                    if point is None:
                        return None
                if (point[-1] < checked[-1][-1]
                        or (point[-1] == checked[-1][-1] and not np.array_equal(point, checked[-1]))
                        or not valid_array(point)):
                    return None
                checked.append(point)
            # END monotone u also permits changing q during a frozen-s plateau.
        samples = np.asarray(checked)
        if not np.array_equal(samples[-1], last):
            return None
        # Bound retained edge geometry independently of RRT's native tree.
        if len(accepted_edges) >= 256:
            accepted_edges.pop(next(iter(accepted_edges)))
        accepted_edges[key] = samples
        return samples

    class ForwardValidator(ob.MotionValidator):
        def __init__(self):
            super().__init__(information)

        def checkMotion(self, a, b, *unused):
            report["motion_checks"] += 1
            return checked_edge(a, b) is not None

    information.setMotionValidator(ForwardValidator())
    try:
        a, b = space.allocState(), space.allocState()
        if constraint is not None:
            a.copy(start.tolist())
            b.copy(goal.tolist())
        else:
            for index in range(dimension):
                a[index], b[index] = float(start[index]), float(goal[index])
        report["initialization_seconds"] = seconds-max(0., deadline-time.monotonic())
        for label, value in (("start", start), ("goal", goal)):
            if not valid_array(value):
                failure = ("search_budget_exhausted_during_boundary_check"
                           if report["last_rejection"] in ("time_budget", "sample_budget")
                           else "boundary_candidate_not_feasible")
                return None, {**report, "failure": failure, "boundary": label}
        # Try the native connector before sampling unrelated configurations.
        # Projected space uses its checked manifold geodesic here; ordinary
        # space uses its checked linear edge. Both retain the exact caller
        # endpoints and every validity/progress/tail check used by RRT edges.
        report["motion_checks"] += 1
        direct = checked_edge(a, b)
        report["direct_connection_accepted"] = direct is not None
        if direct is not None:
            return direct, {**report, "samples": len(direct), "strategy": "native_direct_connection"}
        if time.monotonic() >= deadline:
            return None, {**report, "failure": "whole_search_time_budget_exhausted"}
        definition = ob.ProblemDefinition(information)
        definition.setStartAndGoalStates(a, b)
        search = og.RRTConnect(information)
        search.setIntermediateStates(False)
        search.setProblemDefinition(definition)
        search.setup()
        # Setup and native equality/Jacobian checks consume this same stage
        # budget; passing its original duration again would grant extra time.
        search.solve(max(0., deadline-time.monotonic()))
        if not definition.hasExactSolution():
            return None, {**report, "failure": "no_exact_path_within_search_budget"}
        path = definition.getSolutionPath()
        segments = [checked_edge(path.getState(i), path.getState(i+1))
                    for i in range(path.getStateCount()-1)]
        if not segments or any(segment is None for segment in segments):
            return None, {**report, "failure": "solution_edge_export_rejected"}
        samples = np.concatenate((segments[0], *(segment[1:] for segment in segments[1:])))
        if (len(samples) > sample_limit or np.any(np.diff(samples[:, -1]) < 0)
                or not np.array_equal(samples[0], start) or not np.array_equal(samples[-1], goal)):
            return None, {**report, "failure": "solution_samples_exceed_budget_or_change_endpoints"}
        return samples, {**report, "samples": len(samples), "strategy": "rrt_connect"}
    except TimeoutError:
        return None, {**report, "failure": "whole_search_time_budget_exhausted"}
    finally:
        # Native SI owns the Python callback, whose closure owns SI.  Break
        # that cycle explicitly so repeated requests release native objects.
        information.setMotionValidator(native_validator)


# BEGIN native boundary projection keeps allowed goals as sets of solutions.
def _independent_projection(equality, witness):
    """Select independent local equations without dropping their validity test.

    OMPL's fixed-codimension adapter cannot store inactive zero rows or
    duplicate equations. Pivoting selects coordinates for its Newton step;
    the complete original relations still decide state and edge acceptance.
    """
    from scipy.linalg import qr
    jacobian = np.asarray(equality(witness, True), float)
    if not len(jacobian):
        return equality, 0
    if not np.isfinite(jacobian).all():
        return None, 0
    _, triangular, order = qr(jacobian.T, pivoting=True, mode="economic")
    rank = int(np.linalg.matrix_rank(triangular, tol=1e-7))
    if not rank:
        # A singular real relation is a failed local chart, never an ordinary
        # unconstrained search substituted for the missing manifold.
        return None, 0
    rows = np.sort(order[:rank])
    return lambda value, derivative: equality(value, derivative)[rows], rank


def _project_allowed_boundary(value, residual, process_count, tolerance, lower, upper, *, deadline):
    """Use native OMPL Newton projection into true boundary allowed sets.

    The callback concatenates signed process equations and nonnegative goal
    violations. It adds no nominal target or equation for a free direction.
    Independent rows are refreshed between bounded native Newton iterations,
    so a satisfied band contributes no fictitious equality. Time is fixed.
    Physical bounds and the original goal thresholds are checked unchanged.
    """
    from ompl import base as ob
    from scipy.linalg import qr
    point = np.asarray(value, float).copy()
    dimension = len(point)
    if not np.isfinite(point).all() or np.any(point < lower) or np.any(point > upper):
        return None
    for _ in range(24):
        if time.monotonic() >= deadline:
            return None
        error = np.asarray(residual(point, False), float)
        if not np.isfinite(error).all():
            return None
        if (np.all(np.abs(error[:process_count]) <= tolerance)
                and np.all(error[process_count:] <= 0)):
            return point if np.all(point >= lower) and np.all(point <= upper) else None
        jacobian = np.asarray(residual(point, True), float)
        if not np.isfinite(jacobian).all():
            return None
        # The last coordinate is the frozen query time, not a Newton degree
        # of freedom. Choosing rows on the joint columns also permits a full
        # D-DOF pose goal in the D+1 dimensional time-state representation.
        _, triangular, order = qr(jacobian[:, :-1].T, pivoting=True, mode="economic")
        rank = int(np.linalg.matrix_rank(triangular, tol=1e-7))
        if not rank:
            return None
        rows = np.sort(order[:rank])

        class AllowedBoundary(ob.Constraint):
            def __init__(self):
                # The callback's goal entries already subtract the original
                # allowed error. A positive Newton convergence allowance here
                # would stop just outside that set (notably at a support
                # plane). One native step must act on every positive goal
                # violation; our outer test still uses the declared process
                # projection tolerance and exact original goal membership.
                super().__init__(dimension, rank, np.finfo(float).tiny)
                self.setMaxIterations(1)

            def function(self, values, output):
                output[:] = residual(np.asarray(values), False)[rows]

            def jacobian(self, values, output):
                result = residual(np.asarray(values), True)[rows].copy()
                result[:, -1] = 0.
                output[:] = result

        constraint = AllowedBoundary()
        # A one-iteration return of False means not yet converged, not an
        # invalid numeric update. The next iteration verifies every original
        # residual and selects its current independent rows again.
        constraint.project(point)
        point[-1] = value[-1]
        if not np.isfinite(point).all() or np.any(point < lower) or np.any(point > upper):
            return None
    return None
# END boundary witnesses remain revisable proposals within the same goal.


def _project_boundary(value, equality, tolerance, *, deadline=None):
    """Refine an IK boundary using native projection while preserving its s.

    Alternating native manifold projection and the fixed progress plane is
    bounded.  Failure remains a rejected seed; measured q0 never enters this
    function.  No custom inverse-kinematics or projection solver is used.
    """
    from ompl import base as ob
    dimension, count = len(value), len(equality(value, False))
    if not count:
        return value.copy()
    if count >= dimension:
        return None

    class BoundaryConstraint(ob.Constraint):
        def __init__(self):
            super().__init__(dimension, count, tolerance)

        def function(self, values, output):
            output[:] = equality(np.asarray(values), False)

        def jacobian(self, values, output):
            output[:] = equality(np.asarray(values), True)

    constraint = BoundaryConstraint()
    point = value.copy()
    for _ in range(12):
        if deadline is not None and time.monotonic() >= deadline:
            return None
        if constraint.isSatisfied(point):
            return point
        if not constraint.project(point) or not np.isfinite(point).all():
            return None
        point[-1] = value[-1]
    return point if constraint.isSatisfied(point) else None


def _orientation_vjp_type():
    """Connect native quaternion values to native world geometric Jacobians.

    V2's native ToolPoseCost encodes angular gradients as ``q * omega`` and
    its FK backward consumes that matching body encoding.  An ordinary Torch
    scalar loss instead supplies a Euclidean quaternion gradient; feeding it
    to that native pair rotates its angular gradient into the wrong frame.
    The native world-frame geometric Jacobian supplies the correct derivative
    for custom process/scene losses without changing FK or native pose costs.

    The independent FK central-difference probe checks this chain, including
    orientations near 180 degrees where the original mismatch changes signs.
    """
    import torch

    class WorldQuaternionVJP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, joint_position, quaternion, angular_jacobian):
            # Shapes are [B,H,D], [B,H,L,4] (wxyz), [B,H,L,3,D].
            # Native buffers stay valid until this rollout's backward pass.
            ctx.save_for_backward(quaternion.detach(), angular_jacobian.detach())
            return quaternion.clone()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, gradient):
            quaternion, jacobian = ctx.saved_tensors
            w, vector = quaternion[..., :1], quaternion[..., 1:]
            gw, gv = gradient[..., :1], gradient[..., 1:]
            # qdot = 1/2 * [0, omega_world] * q.  Its transpose maps a
            # quaternion covector to the world angular covector below.
            angular = .5*(-vector*gw + w*gv + torch.linalg.cross(vector, gv))
            joint_gradient = torch.einsum("...lkd,...lk->...d", jacobian, angular)
            # The quaternion argument deliberately gets no derivative:
            # otherwise native FK's distinct quaternion convention would
            # contribute a second, incorrect angular gradient.
            return joint_gradient, None, None

    return WorldQuaternionVJP


def _native_gradient_layout_type():
    """Meet native FK's contiguous-gradient contract after Torch view ops.

    Concatenating main and exact-anchor samples yields strided gradient slices
    for batch sizes above one.  Cloning forward values does not change that
    backward layout, so normalize it at the native output boundary itself.
    This identity changes neither FK values nor their mathematical gradient.
    """
    import torch

    class NativeGradientLayout(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, gradient):
            return gradient.contiguous()

    return NativeGradientLayout


# BEGIN preserve a gradient toward the explicit verification-stop command.
def _verification_stop_preference(progress, joint_velocity, stop_progress, *, trajectory_time=None):
    """Squared speed at each declared stop's first physical occurrence.

    A verification stop commands zero velocity. Its numerical acceptance
    band still belongs exclusively to PlanningProblem.residuals(). Keeping
    this optimizer objective active inside that band avoids terminating on
    its floating-point edge; ordinary motion receives no added objective.
    """
    import torch
    from .planning_problem import _first_event_query
    stopped = torch.zeros_like(progress, dtype=torch.bool)
    for value in np.unique(stop_progress):
        # Use exactly the same progress support as the hard stop residual.
        stopped = stopped | _first_event_query(progress, value, trajectory_time)
    return torch.where(stopped, joint_velocity.square().sum(-1), torch.zeros_like(progress))
# END preserve a gradient toward the explicit verification-stop command.


# BEGIN one formal output clock and native named-body FK for timed hand geometry.
def _formal_native_clock(trajopt):
    """Bind hand, native rollout and dense export to one unchanged cubic clock."""
    # BEGIN use the same admitted grid as the native interpolation adapter.
    from curobo._src.types.control_space import ControlSpace
    config = trajopt.config
    if config.minimum_trajectory_dt != config.maximum_trajectory_dt:
        raise ValueError("A separately timed hand requires one fixed request-wide native dt")
    control = trajopt.transition_model.control_space
    if control != ControlSpace.BSPLINE_3:
        raise ValueError("The formal clock must use the selected native cubic spline representation")
    grid = _fixed_spline_grid(
        intervals=ControlSpace.spline_total_knots(control, trajopt.action_horizon),
        native_dt=config.maximum_trajectory_dt, interpolation_steps=trajopt.interpolation_steps,
        interpolation_dt=config.interpolation_dt)
    return {**grid, "source": "unchanged native cubic clock with exact float32 output grid"}
    # END dense sampling never invents a new physical duration.


class _RobotCoverageFK:
    """An independent same-model FK buffer for native solid body frames."""

    def __init__(self, problem, scene, robot_dict, device_cfg):
        self.problem, self.scene = problem, scene
        self.fk = None
        if scene.robot_coverage is None:
            return
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import JointState
        from curobo.content import get_assets_path
        from curobo._src.util.config_io import join_path
        from pathlib import Path
        self.JointState = JointState
        configuration = deepcopy(robot_dict)
        kinematics = configuration["robot_cfg"]["kinematics"]
        urdf = join_path(kinematics.get("external_asset_path") or get_assets_path(), kinematics["urdf_path"])
        if Path(urdf).resolve() != Path(scene._coverage_geometry["urdf_path"]).resolve():
            raise ValueError("Native coverage FK must use the exact independently checked URDF")
        locks = kinematics.get("lock_joints", {})
        if (scene.gripper is not None
                and scene.gripper.controller in {"timed_reference", "binary"}
                and scene.gripper.nominal_finger_trajectory is not None):
            for name in scene.gripper.joint_names:
                if name not in locks:
                    raise ValueError("Coverage may unlock only the explicitly independent finger joints")
                del locks[name]
        self.fk = Kinematics(KinematicsCfg.from_robot_yaml_file(
            configuration, tool_frames=list(scene.robot_coverage.required_link_names), device_cfg=device_cfg),
            compute_jacobian=True, compute_spheres=False, compute_com=False)
        if set(self.fk.joint_names) != set(scene.robot_coverage.baseline_joint_names):
            raise ValueError("Coverage FK and the native solid oracle must use exactly the same full joint set")
        self._quaternion = _orientation_vjp_type()
        self._layout = _native_gradient_layout_type()

    def poses(self, joint_position, joint_names, trajectory_time):
        if self.fk is None:
            return None
        import torch
        from scipy.spatial.transform import Rotation
        from .planning_problem import quaternion_multiply
        from .optimizer_device_checks import constant_tensor
        names, full = self.scene.full_robot_positions(joint_position, joint_names, trajectory_time)
        columns = [names.index(name) for name in self.fk.joint_names]
        # BEGIN retain integer indices and pose constants in optimizer captures.
        q = full[..., constant_tensor(columns, full, dtype=torch.long)].contiguous()
        state = self.fk.compute_kinematics(self.JointState.from_position(q, joint_names=self.fk.joint_names))
        p = self._layout.apply(state.tool_poses.position)
        r = self._quaternion.apply(q, state.tool_poses.quaternion.detach(), state.tool_jacobians[..., 3:, :].detach())
        base = constant_tensor(self.problem.base_pose, q)
        base_quaternion = constant_tensor(
            Rotation.from_matrix(self.problem.base_pose[:3, :3]).as_quat()[[3, 0, 1, 2]], q)
        # END FK, matrix products and quaternion arithmetic remain unchanged.
        p = p@base[:3, :3].T+base[:3, 3]
        r = quaternion_multiply(base_quaternion, r)
        return {name: (p[..., index, :], r[..., index, :]) for index, name in enumerate(self.fk.tool_frames)}


def _timed_native_state(state, problem, scene, joint_names, trajectory_time, *, sphere_padding=0.):
    """Replace only timed finger spheres before the unchanged native costs.

    cuRobo still owns the arm FK and all native collision pairs. A shallow
    state copy keeps its buffers immutable while the ordinary Torch graph
    carries finger geometry derivatives through the same actual TCP pose.
    """
    hand = scene.gripper
    if hand is None or hand.controller not in {"timed_reference", "binary"}:
        return state
    import torch
    from .optimizer_device_checks import constant_tensor
    q = state.joint_state.position
    tool = state.tool_poses
    index = tool.tool_frames.index(problem.tool_link)
    layout = _native_gradient_layout_type()
    r = _orientation_vjp_type().apply(q, tool.quaternion.detach(),
                                     state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach())
    p, r = problem.transform_tool_pose(layout.apply(tool.position)[:, :, index], r[:, :, index])
    spheres = problem.transform_robot_spheres(layout.apply(state.robot_spheres))
    progress = torch.zeros_like(trajectory_time)  # Timed state depends exclusively on elapsed time.
    fingers, half, _ = hand.state(progress, q, joint_names, trajectory_time=trajectory_time)
    spheres = hand.spheres(spheres, p, r, fingers, half)
    # BEGIN native self collision shares the scene's scoped uncertainty policy.
    # The backend explicitly ignores pairs containing a negative padded radius.
    # Only moving finger proxies after t=0 lack a physical model; fixed palm,
    # arm and every measured entrance sphere retain all native collision pairs.
    if hand.nominal_finger_trajectory is None:
        moving = np.zeros(spheres.shape[-2], dtype=bool)
        for sphere_index, translation in zip(hand.sphere_indices, hand.sphere_joint_translation):
            moving[sphere_index] = bool(np.any(translation != 0))
        uncertain = ((trajectory_time > 0)[..., None]
                     & constant_tensor(moving, spheres, dtype=torch.bool)[None, None])
        padding = constant_tensor(sphere_padding, spheres)
        radius = torch.where(uncertain, -1.-padding.abs(), spheres[..., 3])
        spheres = torch.cat((spheres[..., :3], radius[..., None]), -1)
    # END command completion is never treated as actual finger penetration.
    # BEGIN model pose conversion is cached without changing its input or dtype.
    base = constant_tensor(problem.base_pose, q)
    # END the original inverse rotation still evaluates in the same order.
    spheres = torch.cat(((spheres[..., :3]-base[:3, 3])@base[:3, :3], spheres[..., 3:4]), -1)
    result = copy(state)
    result.cuda_robot_model_state = copy(state.cuda_robot_model_state)
    result.cuda_robot_model_state.robot_spheres = spheres.contiguous()
    return result
# END extra FK changes neither the optimizer variables nor its single IK target.


# BEGIN use each native manager's own effective sphere padding.
def _native_collision_padding(manager):
    cost = manager.get_cost("self_collision")
    return 0. if cost is None or not cost.enabled else cost.config.self_collision_kin_config.sphere_padding
# END a disabled moving proxy remains negative after native padding is added.


# BEGIN scene switches share q/t while querying distinct adjacent modes.
def _scene_boundary_evaluate(scene, problem, progress_map, spheres, position, quaternion,
                             progress, joint_position, joint_names, *, trajectory_time,
                             normalized_time, layer_selection_out=None, source_geometry=False,
                             valid=None, diagnostics=None, source_components_out=None):
    """Check material/attachment mode limits without changing physical time.

    Float64 neighbors vanish when native inputs are float32. Construct the
    side in the actual progress dtype, retaining original samples first so
    release FK keeps its original exact witness. On a progress plateau only
    its entrance consumes the old side and only its exit consumes the new
    side; an action during that plateau never regains an expired old mode.
    """
    import torch
    from .optimizer_device_checks import constant_tensor
    # BEGIN the same mode expansion owns both proxy guidance and source acceptance.
    # This is a call-site mode, not a configurable collision fallback. Formal
    # metrics/search/export explicitly choose source geometry; only optimizer
    # objectives retain the differentiable sphere/ESDF evaluation.
    evaluate = scene.evaluate_source if source_geometry else scene.evaluate
    source_arguments = ({"valid": valid, "diagnostics": diagnostics,
                         "components_out": source_components_out} if source_geometry else {})
    # END actual arrival permissions are preserved identically in both modes.
    anchors = [float(value) for value in scene.required_progress_anchors
               if problem.start_progress <= value <= problem.phase_count]
    if not anchors:
        selected = {} if layer_selection_out is not None else None
        result = evaluate(spheres, position, quaternion, progress, joint_position,
                          joint_names, trajectory_time=trajectory_time, **source_arguments,
                          **({"layer_selection_out": selected} if selected is not None else {}))
        # BEGIN retain pre-reduction state identity only for native diagnostic calls.
        if source_geometry and diagnostics is not None and diagnostics.get("capture_components"):
            from .planning_diagnostics import scene_component_failures
            try:
                diagnostics["component_failures"] = scene_component_failures(
                    source_components_out, trajectory_time, progress, progress, joint_position,
                    position, quaternion, modes=("exact",))
            except Exception as error:
                diagnostics["component_capture_error"] = f"{type(error).__name__}: {error}"
        # END diagnostic conversion cannot alter the evaluated residual.
        if layer_selection_out is not None:
            layer_selection_out.update(_layer_query_record(trajectory_time, progress, progress,
                joint_position, joint_names, selected, ("exact",)))
        return result
    left = torch.zeros_like(progress, dtype=torch.bool)
    right = torch.zeros_like(progress, dtype=torch.bool)
    for anchor in anchors:
        parameters = progress_map.anchor_parameters([anchor])[:, 0]
        # BEGIN preserve exact native-dtype anchor constants during replay.
        matching = progress == constant_tensor(anchor, progress)
        if anchor > problem.start_progress:
            left |= matching & (normalized_time <= constant_tensor(parameters[0], normalized_time))
        if anchor < problem.phase_count:
            right |= matching & (normalized_time >= constant_tensor(parameters[-1], normalized_time))
        # END only constant storage changes, including each original FP32 cast.
    lower = torch.where(left, torch.nextafter(progress, torch.full_like(progress, -float("inf"))), progress)
    upper = torch.where(right, torch.nextafter(progress, torch.full_like(progress, float("inf"))), progress)
    # BEGIN an arrival authorizes its explicit pair before same-state mode probes.
    # Keep the previous material/attachment law, but never revoke a new pair
    # permission by applying that law counterfactually to an already-arrived
    # state. Earlier actual samples and all interval-end probes stay unchanged.
    permission_lower = lower
    for rule in scene.robot_owner_permissions:
        arrival = constant_tensor(rule.interval[0], progress)
        permission_lower = torch.where((progress == arrival) & (lower < arrival), progress, permission_lower)
    # END equality is in the native query dtype; no tolerance advances arrival.
    repeated = lambda value: torch.cat((value, value, value), dim=1)
    selected = {} if layer_selection_out is not None else None
    expanded_progress = torch.cat((progress, lower, upper), dim=1)
    permission_progress = torch.cat((progress, permission_lower, upper), dim=1)
    # BEGIN padding masks and diagnostics follow exact/left/right state identity.
    expanded_components = {} if source_components_out is not None else None
    if source_geometry:
        source_arguments.update(valid=None if valid is None else repeated(valid),
                                components_out=expanded_components)
    terms = evaluate(repeated(spheres), repeated(position), repeated(quaternion),
        expanded_progress, repeated(joint_position), joint_names,
        # BEGIN pair authorization and other discontinuous laws retain separate scope.
        trajectory_time=repeated(trajectory_time),
        robot_owner_permission_progress=permission_progress,
        **source_arguments,
        **({"layer_selection_out": selected} if selected is not None else {}))
        # END q, physical time and the command clock are unchanged.
    # BEGIN preserve each original mode before taking its maximum.
    if source_geometry and diagnostics is not None and diagnostics.get("capture_components"):
        from .planning_diagnostics import scene_component_failures
        try:
            diagnostics["component_failures"] = scene_component_failures(
                expanded_components, repeated(trajectory_time), expanded_progress, permission_progress,
                repeated(joint_position), repeated(position), repeated(quaternion),
                modes=("exact", "left", "right"))
        except Exception as error:
            diagnostics["component_capture_error"] = f"{type(error).__name__}: {error}"
    # END no extra geometry query or acceptance calculation is performed.
    if layer_selection_out is not None:
        layer_selection_out.update(_layer_query_record(repeated(trajectory_time), expanded_progress,
            permission_progress, repeated(joint_position), joint_names, selected, ("exact", "left", "right")))
    if source_components_out is not None:
        source_components_out.update({name: torch.stack(value.split(progress.shape[1], dim=1)).amax(0)
                                      for name, value in expanded_components.items()})
    # END source world checks never see native dense-buffer padding as a state.
    return torch.stack(terms.split(progress.shape[1], dim=1)).amax(0)


def _layer_query_record(times, progress, permission, joints, names, selected, modes):
    """Retain the actual typed query and selected layer for independent binding."""
    cpu = lambda value: encode_array(value.detach().cpu())
    return {"time_s": cpu(times), "progress": cpu(progress),
        "robot_owner_permission_progress": cpu(permission), "joint_position": cpu(joints),
        "joint_names": list(names), "time_dtype": str(times.dtype), "progress_dtype": str(progress.dtype),
        "query_order": list(modes), "layer_selection": selected}
# END query-side selection never changes the command or nominal hand clock.


def _process_manager_type(problem, scene, progress_map, anchors, budget,
                          refinement_parameters=(), *, clock=None, seed_parameter=None, robot_dict=None,
                          phase="feasibility", functional_constraints=None):
    """Build V2's documented manager extension after GPU use is requested.

    Configuration construction must return the derived type and class_type;
    registering a custom name alone is not dispatched by V2's base manager.
    Only constraint_cfg enables this extension. Built-in joint constraints
    remain intact; optimizer spheres guide motion and metrics use source solids.
    """
    import torch
    from curobo.kinematics import Kinematics
    from curobo.types import JointState
    from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
    from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
    # BEGIN exact-anchor collision checks use the same native implementation.
    from curobo._src.cost.cost_self_collision import SelfCollisionCost
    from curobo._src.state.state_robot import RobotState
    # END no separate collision formula or policy is introduced.
    orientation_vjp = _orientation_vjp_type()
    native_layout = _native_gradient_layout_type()
    # BEGIN one fixed functional objective is shared by this solver's rollouts.
    from .functional_augmented_lagrangian import FunctionalAugmentedLagrangian
    if functional_constraints is None:
        functional_constraints = FunctionalAugmentedLagrangian(problem.functional_constraint_keys)
    # END a standalone manager uses the same zero-multiplier objective.

    class ProcessCostManager(RobotCostManager):
        def initialize_from_config(self, config, transition_model, scene_collision_checker=None, **kwargs):
            # BEGIN give each native manager its own mutable cost configuration.
            # Native initialization divides self-collision weight by its
            # interpolation count and binds cspace limits/types in place. The
            # rollout template is shared by several managers and rebuilds;
            # preserve it so each instance receives the original parameters.
            config = deepcopy(config)
            # END give each native manager its own mutable cost configuration.
            super().initialize_from_config(config, transition_model, scene_collision_checker, **kwargs)
            self._process_mode = config.process_mode
            self._dense_counts = None
            if not self._process_mode:
                return
            transition_model.robot_model.compute_jacobian = True
            self._scene = scene.clone_for_rollout()
            self._joint_names = tuple(transition_model.joint_names)
            # BEGIN ordinary reference IK does not certify a task trajectory.
            # The retained private hand clock supplies native self-collision
            # geometry, but nominal pose samples need neither scene coverage
            # nor functional/attachment feasibility. Those checks remain on
            # every full trajectory and on independent search/verification.
            if self._process_mode.startswith("seed_"):
                return
            # END do not allocate coverage or anchor FK for reference IK.
            self._coverage_fk = (_RobotCoverageFK(problem, self._scene, robot_dict, self.device_cfg)
                                 if scene.robot_coverage is not None
                                 and (scene.gripper is None or scene.gripper.controller == "position_tracking"
                                      or scene.gripper.nominal_finger_trajectory is not None)
                                 and self._process_mode == "trajectory_optimizer" else None)
            self._tool_index = transition_model.robot_model.tool_frames.index(problem.tool_link)
            self._map_u = self.device_cfg.to_device(progress_map.u)
            self._map_s = self.device_cfg.to_device(progress_map.s)
            anchor_pairs = progress_map.anchor_parameters(anchors)
            if len(refinement_parameters):
                additional = np.asarray(refinement_parameters, float)
                anchor_pairs = np.unique(np.concatenate((anchor_pairs, np.column_stack(
                    (additional, np.interp(additional, progress_map.u, progress_map.s))))), axis=0)
            self._anchor_u = self.device_cfg.to_device(anchor_pairs[:, 0]).unsqueeze(0)
            self._anchor_s = self.device_cfg.to_device(anchor_pairs[:, 1]).unsqueeze(0)
            # BEGIN bind small AL constants before any native graph capture.
            if self._process_mode == "trajectory_optimizer":
                functional_constraints.bind(self._anchor_u)
            # END outer updates always precede a complete solver rebuild.
            # A separate native FK instance owns the anchor buffers.  Reusing
            # the main rollout FK would overwrite values needed by backward.
            self._anchor_fk = Kinematics(transition_model.robot_model.config, compute_jacobian=True)
            # BEGIN every added task/refinement time also checks self collision.
            # The main cost owns mutable CUDA forward/backward buffers. A
            # separate instance preserves both graphs, while copying its
            # already-normalized config avoids dividing the weight twice.
            native_self = self.get_cost("self_collision")
            self._anchor_self_collision = (None if native_self is None else
                SelfCollisionCost(deepcopy(native_self.config)))
            # END ordinary reference IK creates no additional check buffers.

        def set_dense_counts(self, counts):
            self._dense_counts = counts.reshape(-1).long()

        def compute_costs(self, state, cost_collection=None, goal=None, **kwargs):
            # BEGIN proxy geometry supplies gradients only; source solids own acceptance.
            if self._process_mode.endswith("_optimizer") and self._scene.gripper is not None and self._scene.gripper.controller in {"timed_reference", "binary"}:
                batch, horizon = state.joint_state.position.shape[:2]
                if self._process_mode.startswith("seed_"):
                    query_time = (seed_parameter*clock["duration"]).expand(batch, horizon)
                else:
                    counts = (torch.full((batch,), horizon, dtype=torch.long, device=state.joint_state.position.device)
                              if self._dense_counts is None else self._dense_counts)
                    query_time = (torch.arange(horizon, device=counts.device, dtype=state.joint_state.position.dtype)[None]
                                  /(counts[:, None]-1)).clamp(0, 1)*clock["duration"]
                state = _timed_native_state(state, problem, self._scene, self._joint_names, query_time,
                                            sphere_padding=_native_collision_padding(self))
            collection = super().compute_costs(state, cost_collection, goal, **kwargs)
            if self._process_mode == "seed_metrics":
                q = state.joint_state.position
                query_time = (seed_parameter*clock["duration"]).expand(*q.shape[:2])
                collection.add(self._scene.check_robot_self_collision(
                    q, self._joint_names, query_time).unsqueeze(-1), "source_self_collision")
            # END reference IK receives the same physical time as its optimizer.
            # BEGIN native IK remains an initializer even if its reference collides.
            # Native pose and physical joint-domain evaluation remain intact.
            # Only complete candidate curves receive the
            # scene/process penalties below; a reference IK success is never
            # a planning, goal-completion or execution authorization.
            if not self._process_mode or self._process_mode.startswith("seed_"):
                return collection
            # END full trajectory constraints still run in both optimizer/metrics.
            q, dq = state.joint_state.position, state.joint_state.velocity
            batch, horizon, _ = q.shape
            tool = state.tool_poses
            quaternion = orientation_vjp.apply(q, tool.quaternion.detach(),
                                               state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach())
            position = native_layout.apply(tool.position)[:, :, self._tool_index]
            quaternion = quaternion[:, :, self._tool_index]
            # BEGIN local native-coordinate rates use differentiable physical J(q) dq.
            need_rates = any(direction.kind == "local_rate" for mechanism in problem.mechanisms
                             for direction in mechanism.directions)
            twist = (torch.einsum("...kd,...d->...k", native_layout.apply(
                state.cuda_robot_model_state.tool_jacobians)[:, :, self._tool_index], dq) if need_rates else None)
            # END Jacobian backward retains dJ/dq as well as the velocity derivative.
            spheres = native_layout.apply(state.robot_spheres)
            counts = (torch.full((batch,), horizon, dtype=torch.long, device=q.device)
                      if self._dense_counts is None else self._dense_counts)
            u = (torch.arange(horizon, device=q.device, dtype=q.dtype)[None, :]
                 / (counts[:, None]-1)).clamp(0, 1)
            progress = _torch_interpolate(self._map_u, self._map_s, u)
            anchor_u = self._anchor_u.expand(batch, -1)
            anchor_s = self._anchor_s.expand(batch, -1)
            anchor_q, anchor_dq = _hermite_at_parameters(q, dq, state.joint_state.dt, counts, anchor_u)
            # BEGIN retain the actual anchor state for native self collision.
            anchor_joints = JointState.from_position(anchor_q.contiguous(), joint_names=list(self._joint_names))
            anchor_state = self._anchor_fk.compute_kinematics(anchor_joints)
            native_self = self.get_cost("self_collision")
            if native_self is not None and native_self.enabled:
                anchor_native = _timed_native_state(
                    RobotState(joint_state=anchor_joints, cuda_robot_model_state=anchor_state),
                    problem, self._scene, self._joint_names, anchor_u*clock["duration"],
                    sphere_padding=_native_collision_padding(self))
                self._anchor_self_collision.setup_batch_tensors(batch, anchor_u.shape[1])
                anchor_collision = self._anchor_self_collision.forward(
                    native_layout.apply(anchor_native.robot_spheres).contiguous())
                # These values already contain the native optimizer weight
                # and policy; metrics use source solids instead. Do not square or multiply them
                # by the separate task/scene constraint weight below.
                folded_collision = torch.cat((q.new_zeros((batch, horizon-1, 1)),
                    anchor_collision.sum(1, keepdim=True)), dim=1)
                collection.add(folded_collision, "anchor_self_collision")
            # END anchor sphere penalties retain native optimizer gradients.
            anchor_quaternion = orientation_vjp.apply(anchor_q, anchor_state.tool_poses.quaternion.detach(),
                                                      anchor_state.tool_jacobians[..., 3:, :].detach())
            anchor_position = native_layout.apply(anchor_state.tool_poses.position)[:, :, self._tool_index]
            if need_rates:
                anchor_twist = torch.einsum("...kd,...d->...k", native_layout.apply(
                    anchor_state.tool_jacobians)[:, :, self._tool_index], anchor_dq)
                twist = torch.cat((twist, anchor_twist), 1)
            position = torch.cat((position, anchor_position), 1)
            quaternion = torch.cat((quaternion, anchor_quaternion[:, :, self._tool_index]), 1)
            spheres = torch.cat((spheres, native_layout.apply(anchor_state.robot_spheres)), 1)
            q, dq = torch.cat((q, anchor_q), 1), torch.cat((dq, anchor_dq), 1)
            progress = torch.cat((progress, anchor_s), 1)
            # End-point families consume a whole ordered trajectory and the
            # actual functional-event FK.  No nearest-grid event substitution.
            all_u = torch.cat((u, anchor_u), 1)
            order = torch.argsort(all_u, dim=1, stable=True)
            inverse = torch.argsort(order, dim=1)

            def ordered(value):
                index = order.reshape(*order.shape, *((1,)*(value.ndim-2))).expand_as(value)
                return value.gather(1, index)

            link_quaternion = ordered(quaternion)
            tool_linear, tool_angular = ((None, None) if twist is None else problem.transform_tool_velocity(
                ordered(twist)[..., :3], ordered(twist)[..., 3:], link_quaternion))
            position, quaternion = problem.transform_tool_pose(ordered(position), link_quaternion)
            spheres = problem.transform_robot_spheres(ordered(spheres))
            q, dq, progress = ordered(q), ordered(dq), ordered(progress)
            trajectory_time = ordered(all_u)*clock["duration"]
            # BEGIN functional point inequalities have exactly one optimizer owner.
            process_terms = problem.residuals(position, quaternion, progress, q, self._joint_names,
                joint_velocity=dq, trajectory_time=trajectory_time, tool_linear_velocity=tool_linear,
                tool_angular_velocity=tool_angular)
            process_value = sum(value for name, value in process_terms.items()
                if self._process_mode != "trajectory_optimizer"
                or not name.startswith(("function_pose:", "function_coordinate:")))
            # END metrics retain the complete original hard residual dictionary.
            # BEGIN only real native states enter source world/self acceptance.
            valid = torch.arange(horizon, device=q.device)[None, :] < counts[:, None]
            source_valid = (ordered(torch.cat((valid, torch.ones_like(anchor_u, dtype=torch.bool)), 1))
                            if self._process_mode == "trajectory_metrics" else None)
            # BEGIN native acceptance keeps component identity before anchor folding.
            scene_diagnostics = ({"capture_components": True}
                                 if self._process_mode == "trajectory_metrics" else None)
            scene_components = {} if scene_diagnostics is not None else None
            # END optimizer costs do not allocate diagnostic state snapshots.
            scene_value = _scene_boundary_evaluate(self._scene, problem, progress_map,
                spheres, position, quaternion, progress, q, self._joint_names,
                trajectory_time=trajectory_time, normalized_time=ordered(all_u),
                source_geometry=self._process_mode == "trajectory_metrics", valid=source_valid,
                diagnostics=scene_diagnostics, source_components_out=scene_components)
            # BEGIN retain process/self failure times before folding exact anchors.
            source_self = None
            if self._process_mode == "trajectory_metrics":
                source_self = self._scene.check_robot_self_collision(
                    q, self._joint_names, trajectory_time, valid=source_valid)
            # END the same self query supplies both evidence and native constraints.
            # BEGIN copy only this metrics call; later calls replace all prior evidence.
            if scene_diagnostics is not None:
                try:
                    if "component_capture_error" in scene_diagnostics:
                        raise ValueError(scene_diagnostics["component_capture_error"])
                    original = state.joint_state
                    captured = [value.detach().cpu().numpy().copy()
                                for value in (original.position, original.velocity, order)]
                    from .planning_diagnostics import capture_component_failures
                    scene_diagnostics["component_failures"].extend(capture_component_failures(
                        {"skill_process": process_value*source_valid,
                         "source_self_collision": source_self},
                        trajectory_time, progress, q, position, quaternion,
                        normalized_time=ordered(all_u)))
                    normalized_times = ordered(all_u).detach().cpu().numpy()
                    for row in scene_diagnostics["component_failures"]:
                        column = int(captured[2][row["batch_index"], row["ordered_query_index"]])
                        row.update(sample_kind="regular" if column < horizon else "anchor",
                                   sample_index=column if column < horizon else column-horizon,
                                   normalized_time=float(normalized_times[
                                       row["batch_index"], row["ordered_query_index"]]))
                    scene_diagnostics.pop("capture_components")
                    self.last_source_diagnostics = {
                        "input_position": captured[0], "input_velocity": captured[1],
                        "component_failures": scene_diagnostics.pop("component_failures"),
                        "world_geometry": scene_diagnostics}
                except Exception as error:
                    self.last_source_diagnostics = {"error": f"{type(error).__name__}: {error}"}
            # END evidence failures remain separate from every numerical residual.
            # END optimizer world gradients continue through the original ESDF.
            # BEGIN all hard self checks use the same source geometry and exact query times.
            if self._process_mode == "trajectory_metrics":
                source_self = source_self.gather(1, inverse)
                folded_self = torch.cat((q.new_zeros((batch, horizon-1)),
                    source_self[:, horizon:].sum(1, keepdim=True)), 1)
                collection.add((source_self[:, :horizon]+folded_self).unsqueeze(-1),
                               "source_self_collision")
            # END padded dense-buffer states are excluded before CPU geometry queries.
            # BEGIN retain the original summed scene objective before squaring.
            # Fixed installation proxies and same-time hand geometry are
            # identical on both sides of a material mode boundary. Adding
            # their guidance after the boundary maximum therefore preserves
            # max(environment + self) and its original cross derivatives.
            if self._process_mode == "trajectory_optimizer":
                scene_value = scene_value + self._scene.robot_self_collision_guidance(
                    spheres, position, quaternion, progress, q, self._joint_names,
                    trajectory_time=trajectory_time)
            # END metrics retain environment constraints and source self checks separately.
            values = [("skill_process", process_value), ("process_scene", scene_value)]
            # A squared optimizer penalty has a gradient proportional to
            # the remaining violation. Metrics retain the original raw
            # nonnegative values and therefore their original zero sets.
            if self._process_mode == "trajectory_optimizer":
                values = [(name, value.square()) for name, value in values]
                # BEGIN signed inequalities enter PHR directly, never squared twice.
                signed, _ = problem.functional_inequalities(
                    position, quaternion, progress, trajectory_time=trajectory_time)
                augmented = sum(functional_constraints.terms(signed).values(), q.new_zeros(batch))
                folded_augmented = torch.cat((q.new_zeros((batch, horizon-1, 1)),
                                               augmented[:, None, None]), dim=1)
                collection.add(folded_augmented * budget.constraint_weight,
                               "functional_augmented_lagrangian")
                # END original tolerance and first-arrival identity belong to the problem.
                if self._coverage_fk is not None:
                    values.append(("robot_new_volume_guidance", self._scene.robot_coverage_guidance(
                        self._coverage_fk.poses(q, self._joint_names, trajectory_time), progress).square()))
                # BEGIN retain a gradient toward source-declared pose equalities.
                # The problem owns equality-axis selection and its complete
                # endpoint-family reference; free pose axes gain no target.
                values.append(("skill_pose_equality_preference", problem.optimizer_preferences(
                    position, quaternion, progress, trajectory_time=trajectory_time,
                    tool_linear_velocity=tool_linear, tool_angular_velocity=tool_angular,
                    phase=phase)))
                # END retain a gradient toward source-declared pose equalities.
                # A position-tracking gripper has an exact physical command,
                # even inside its numerical acceptance band. Preserve a
                # gradient toward that command to avoid stopping on the
                # band's floating-point edge. Allowed pose/contact freedoms
                # receive no invented centering preference.
                values.append(("gripper_command_preference", self._scene.optimizer_preferences(
                    progress, q, self._joint_names)))
                # BEGIN optimize the actual zero-speed verification command.
                values.append(("verification_stop_preference", _verification_stop_preference(
                    progress, dq, problem.stop_progress, trajectory_time=trajectory_time)))
                # END optimize the actual zero-speed verification command.
            for name, value in values:
                value = value.gather(1, inverse)
                regular = value[:, :horizon] * valid
                # Keep CostCollection's native [B,H,*] shape.  Summing exact
                # anchor penalties into its final column retains their full
                # gradient and makes any anchor violation fail feasibility.
                folded = torch.cat((torch.zeros_like(regular[:, :-1]), value[:, horizon:].sum(1, keepdim=True)), 1)
                collection.add((regular+folded).unsqueeze(-1) * budget.constraint_weight, name)
            return collection

    @dataclass
    class ProcessCostManagerCfg(RobotCostManagerCfg):
        process_mode: str = ""

        def __post_init__(self):
            self.class_type = ProcessCostManager

        @staticmethod
        def create(data_dict, scene_collision_checker=None, device_cfg=None):
            base = RobotCostManagerCfg.create(data_dict, scene_collision_checker, device_cfg)
            values = {item.name: getattr(base, item.name) for item in fields(base) if item.name != "class_type"}
            return ProcessCostManagerCfg(**values, process_mode=data_dict.get("process_mode", ""))

    return ProcessCostManagerCfg


# BEGIN exact native candidate extraction before independent verification.
def _trajectory_candidate(result, index):
    """Select one original seed through V2's own complete result gather.

    V2 has no arbitrary-index result accessor. Its top-k gather selects from
    ``seed_rank`` for actions, flags and valid lengths. JointState's native
    indexer separately handles the full joint states and their active-joint
    controls; these need not have the same DOF when some joints are locked.
    Never flatten seeds into one apparent temporal trajectory.
    """
    if result.batch_size != 1 or not 0 <= index < result.num_seeds:
        raise ValueError("Candidate extraction requires one batch and an existing seed index")
    if result.num_seeds == 1:
        return result
    ordered = copy(result)
    ordered.seed_rank = result.seed_rank.new_tensor(
        [[index, *(other for other in range(result.num_seeds) if other != index)]])
    # Native postprocessing augments state positions with locked joints but
    # intentionally leaves their action knots in active-joint coordinates.
    # Its result gather assumes both have position's DOF. The JointState
    # indexer instead slices each tensor on the seed axis without that false
    # reshape, preserving both coordinate layouts and every timing field.
    ordered.js_solution = ordered.interpolated_trajectory = None
    candidate = ordered.get_topk_seeds(1)
    for name in ("js_solution", "interpolated_trajectory"):
        state = getattr(result, name)
        single = None if state is None else state[:, index:index+1].clone()
        if state is not None and state.dt is not None:
            # JointState's general indexer treats a leading singleton dt as
            # shared timing. Here dt is explicitly [batch,seed], so select
            # its seed column too, even though batch is exactly one.
            single.dt = state.dt[:, index:index+1].clone()
        setattr(candidate, name, single)
    return candidate
# END exact native candidate extraction before independent verification.


# BEGIN one native solve owns bounded source reuse across repeated observations.
def _bind_candidate_source_world_reuse(optimizer):
    """Bind candidate state reports and completed batches to native calls.

    Every ordinary call resets the short state-report segment. Its matching
    dense call ends that segment, while each scene's last completed source
    batch survives until solve cleanup. All native metrics still execute;
    only exact source-world inputs may reuse a completed geometry verdict.
    """
    from functools import wraps
    ordinary = optimizer.metrics_rollout
    dense = optimizer.additional_metrics_rollouts["interpolated_rollout"]
    ordinary_scene = ordinary.metrics_constraint_manager._scene
    dense_scene = dense.metrics_constraint_manager._scene
    ordinary_metrics = ordinary.compute_metrics_from_action
    dense_metrics = optimizer._interpolate_and_compute_metrics

    @wraps(ordinary_metrics)
    def evaluate_ordinary(*args, **kwargs):
        session = getattr(ordinary_scene, "_source_world_reuse", None)
        if session is None or session.binding_for(ordinary_scene) is None:
            session = ordinary_scene.begin_source_world_reuse(dense_scene)
        session.begin_candidate()
        try:
            return ordinary_metrics(*args, **kwargs)
        except BaseException:
            ordinary_scene.end_source_world_reuse()
            raise

    @wraps(dense_metrics)
    def evaluate_dense(*args, **kwargs):
        try:
            return dense_metrics(*args, **kwargs)
        except BaseException:
            ordinary_scene.end_source_world_reuse()
            raise
        finally:
            session = getattr(ordinary_scene, "_source_world_reuse", None)
            if session is not None:
                session.finish_candidate()

    ordinary.compute_metrics_from_action = evaluate_ordinary
    optimizer._interpolate_and_compute_metrics = evaluate_dense
    return ordinary_scene.end_source_world_reuse
# END every solve/error closes both scene slots; dense success only ends its segment.


class _CuroboPlanner:
    """One request's native solvers and independent verification workspaces."""

    # BEGIN every planner binds one explicit source geometry and pair policy.
    def __init__(self, problem, scene, robot, budget, device, *, robot_validator, continuation=None):
        self.ik = self.trajopt = None
        # BEGIN keep full float32 geometry through the complete native request.
        # TF32 truncates even an identity matrix product, so its coordinate
        # quantization can exceed the physical collision tolerance. Precision
        # must cover backward as well as forward; a cost-only context is not
        # sufficient. Save caller settings before cuRobo's import side effects.
        import torch
        self._precision_torch = torch
        self._caller_precision = (torch.backends.cuda.matmul.allow_tf32,
                                  torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        # END close() restores both settings on success and initialization error.
        try:
            self._initialize(problem, scene, robot, budget, device,
                             robot_validator=robot_validator, continuation=continuation)
        except BaseException:
            # A later native configuration failure must release an IK solver
            # already created in this process before propagating the error.
            self.close()
            raise

    def _initialize(self, problem, scene, robot, budget, device, *, robot_validator, continuation=None):
        # Validate grounded evidence and bound independent scene workspaces
        # before importing native GPU solvers or creating any of their buffers.
        scene.bind_robot_validator(robot_validator, problem)
        self.robot_validator = robot_validator
        scene.validate_problem(problem)
        # END binding precedes native allocation and applies to direct planner construction.
        budget, initial_progress = (_initial_timing(problem, budget) if continuation is None
                                    else _continuation_timing(problem, budget, continuation))
        anchors = np.unique(np.concatenate((problem.anchors, np.asarray(scene.required_progress_anchors, float))))
        # BEGIN preserve all required source points within the total query budget.
        if (len(anchors) > budget.max_validation_points or not np.isfinite(anchors).all()
                or np.any(anchors < 0) or np.any(anchors > problem.phase_count)):
            raise ValueError("All required source knots and scene anchors must fit the finite planning horizon and budget")
        if problem.reference_frame != scene.reference_frame:
            raise ValueError("Problem and planning scene must use the same explicitly calibrated reference frame")
        self.progress_map = initial_progress
        if continuation is None:
            self.progress_map, _ = _progress_for_cubic_boundaries(initial_progress,
                budget.knots, problem.start_dq, stop_progress=problem.stop_progress, anchors=anchors)
        initial_clock = _fixed_spline_grid(intervals=budget.knots+4,
            native_dt=budget.maximum_trajectory_dt, interpolation_steps=budget.interpolation_steps,
            interpolation_dt=budget.interpolation_dt)
        # BEGIN reject an intrinsically undersized export budget before native allocation.
        # Export checks at least two subdivisions of every native output
        # interval, even for a constant curve. This is a necessary lower
        # bound only: actual joint motion and exact semantic anchors can add
        # queries, whose unchanged limits are still checked by export().
        minimum_validation_points = 2*(initial_clock["valid_count"]-1)+1
        if minimum_validation_points > budget.max_validation_points:
            raise ValueError(
                "Final curve validation requires at least "
                f"{minimum_validation_points} points, allowed {budget.max_validation_points}; "
                f"bound clock duration_s={initial_clock['duration']!r}, "
                f"interpolation_dt_s={initial_clock['interpolation_dt']!r}, "
                f"native_samples={initial_clock['valid_count']} "
                "(two subdivisions per output interval; anchors or motion may require more)")
        # END admission never changes the clock, resolution, anchors or declared capacity.
        if continuation is None:
            self.progress_map = _bind_event_clock(problem, self.progress_map, initial_clock["duration"],
                None if scene.gripper is None else scene.gripper.control_timestep_s)
        # BEGIN freeze synchronized hand inputs before any rollout owns a clone.
        # IK, trajectory managers and independent verification each deep-copy
        # the scene. Binding only the parent after their construction leaves
        # those copies without a command/nominal polynomial. The admitted grid
        # already defines the exact native duration; verify it against the
        # constructed solver below without regenerating these commands.
        hand = scene.gripper
        if hand is not None:
            hand.bind_clock(initial_clock["duration"], self.progress_map)
        self.finger_trajectory = None if hand is None else hand.finger_trajectory
        # END every private scene receives the same already frozen hand clock.
        # BEGIN build source world shapes once from the finalized request inputs.
        # Continuation filtering and hand-clock binding precede this snapshot.
        # Native managers/search children share its immutable geometry while
        # supplying their own owner/release poses for every source query.
        scene.bind_source_scene(max_host_rss_bytes=budget.max_host_rss_bytes)
        # END execution keeps its own independently bound checker and arrival latches.
        self.anchor_capacity = _anchor_capacity(self.progress_map, anchors, budget)
        robot_dict, self.robot_metadata = _bounded_robot_configuration(robot, problem, scene, budget)
        # BEGIN native and source FK must consume the same complete robot description.
        # Compare the parsed XML with the validator's consumed snapshot, not
        # with its mutable source file. Runtime native tree overrides have no
        # counterpart in the bound source model and cannot change this tree.
        from curobo.content import get_assets_path
        from pathlib import Path
        import xml.etree.ElementTree as ET
        kinematics = robot_dict["robot_cfg"]["kinematics"]
        native_urdf = Path(kinematics.get("external_asset_path") or get_assets_path()) / kinematics["urdf_path"]
        checker = robot_validator.checker
        if (kinematics["base_link"] != robot_validator.checker.base_frame
                or kinematics.get("extra_links") or kinematics.get("add_object_link", False)
                or ET.tostring(ET.parse(native_urdf).getroot(), encoding="unicode")
                != checker.model_record()["urdf_xml"]):
            raise ValueError("Native planning requires the consumed source URDF/base without tree overrides")
        locks = kinematics.get("lock_joints") or {}
        independent_hand = hand is not None and hand.controller in {"timed_reference", "binary"}
        allowed_locks = dict(zip(hand.joint_names, hand.current_positions)) if independent_hand else {}
        if (not set(locks) <= set(allowed_locks) or any(
                value != allowed_locks[name] and value != float(np.float32(allowed_locks[name]))
                for name, value in locks.items())):
            raise ValueError("Native joint locks may only retain the measured independent finger coordinates")
        if independent_hand:
            joint = int(checker.model.frames[checker._frame_id(robot_validator.tool_frame)].parentJoint)
            while joint:
                if str(checker.model.names[joint]) in hand.joint_names:
                    raise ValueError("An independently controlled finger cannot be an ancestor of the native TCP")
                joint = int(checker.model.parents[joint])
        # END source identity is established before allocating GPU solvers.
        self.validation_storage = _validation_storage_admission(problem, scene, self.robot_metadata, budget)
        workspace = _planning_workspace_bytes(scene, self.robot_metadata, budget, self.anchor_capacity)
        _require_planning_host_memory(workspace, max_host_rss_bytes=budget.max_host_rss_bytes)
        # END preserve all required source points within the total query budget.
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
        from curobo.kinematics import Kinematics
        from curobo.types import JointState, GoalToolPose
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.util.config_io import load_yaml, join_path
        from curobo.content import get_task_configs_path

        # BEGIN undo the native seed-IK module's global TF32 import assignment.
        # Apply after native imports and before any native solver/FK buffers or
        # geometry calculations. cuDNN is not used here; preserve its caller
        # setting rather than leaking the dependency's unrelated assignment.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = self._caller_precision[1]
        # END the single-request lifecycle owns forward and backward precision.
        self.problem, self.scene, self.budget = problem, scene, budget
        self.torch, self.JointState, self.GoalToolPose = torch, JointState, GoalToolPose
        self.device_cfg = DeviceCfg(device)
        # BEGIN include request-sized anchor FK in native allocation admission.
        workspace = _require_planning_memory(torch, self.device_cfg.device, workspace,
                                             max_device_used_bytes=budget.max_device_used_bytes)
        self.scene_workspace_bytes = workspace.total_bytes
        self.scene_cpu_workspace_bytes, self.scene_gpu_workspace_bytes = workspace.cpu_bytes, workspace.gpu_bytes
        self.anchor_fk_workspace_bytes = workspace.anchor_fk_bytes
        self._trajectory_optimizer_graph_admitted = workspace.trajectory_graph
        # END include request-sized anchor FK in native allocation admission.
        self.ik = self.trajopt = None
        self.anchors = anchors
        self.robot_dict = robot_dict
        self.seed_parameter = self.device_cfg.to_device([[0.]])
        # BEGIN native callbacks inherit the admitted clock during construction.
        self.formal_clock = dict(initial_clock)
        # END construction cannot observe a missing or independently chosen clock.
        # BEGIN continuation multipliers survive only this attempt's solver rebuilds.
        from .functional_augmented_lagrangian import FunctionalAugmentedLagrangian
        self.functional_constraints = FunctionalAugmentedLagrangian(problem.functional_constraint_keys)
        # END all first-round seeds start from the same zero multiplier vector.
        self._manager_type = _process_manager_type(
            problem, scene, self.progress_map, self.anchors, budget,
            clock=self.formal_clock, seed_parameter=self.seed_parameter, robot_dict=self.robot_dict,
            functional_constraints=self.functional_constraints)
        self.refinement_parameters = np.empty(0)
        # BEGIN one attempt owns its phase and every discovered failure time.
        from .planning_diagnostics import AttemptWitnesses
        self.phase = "feasibility"
        self.witnesses = AttemptWitnesses()
        self.last_working_set = {}
        self.goal_ik_candidates = {}
        self.last_optimizer_diagnostics = []
        # END neither solver rebuilds nor candidate selection reset this state.

        def configuration(name, mode):
            data = deepcopy(load_yaml(join_path(get_task_configs_path(), name)))
            constraint = data["rollout"].setdefault("constraint_cfg", {})
            # All environment geometry is owned by PlanningScene in reference
            # coordinates. Spheres remain an optimizer gradient only.
            constraint.pop("scene_collision_cfg", None)
            # BEGIN source solids replace only the self component of native feasibility.
            if mode.endswith("_metrics"):
                constraint.pop("self_collision_cfg", None)
            # END native joint-domain checks keep their original policy.
            constraint["process_mode"] = mode
            return data

        ik_optimizer = configuration("ik/lbfgs_ik.yml", "seed_optimizer")
        # BEGIN retain the native quality configuration for the second phase.
        trajectory_optimizer = configuration("trajopt/lbfgs_bspline_trajopt.yml", "trajectory_optimizer")
        # BEGIN find a feasible curve before spending motion freedom on quality.
        self._quality_optimizer_config = deepcopy(trajectory_optimizer)
        trajectory_optimizer["rollout"]["cost_cfg"]["cspace_cfg"]["squared_l2_regularization_weight"] = [0.]*5
        # Native limit penalties and independent constraints stay enabled.
        # END Phase B restores the original motion regularization configuration.
        # END independent hard constraints decide whether either curve is usable.
        for name, config, iterations in (("IK", ik_optimizer, budget.ik_iterations),
                                         ("trajectory", trajectory_optimizer, budget.trajectory_iterations)):
            inner_iterations = config["optimizer"]["inner_iters"]
            if iterations % inner_iterations:
                raise ValueError(f"{name} iterations must be a multiple of native inner_iters={inner_iterations}")
        ik_cfg = InverseKinematicsCfg.create(
            robot=deepcopy(robot_dict), optimizer_configs=[ik_optimizer],
            metrics_rollout=configuration("metrics_base.yml", "seed_metrics"),
            device_cfg=self.device_cfg, num_seeds=budget.ik_seeds, seed_solver_num_seeds=budget.ik_seeds,
            max_batch_size=1, max_goalset=1, use_cuda_graph=False,
            position_tolerance=problem.numerical.position_m,
            orientation_tolerance=problem.numerical.rotation_rad,
            cost_manager_config_instance_type=self._manager_type,
            override_optimizer_num_iters={"lbfgs": budget.ik_iterations}, random_seed=budget.random_seed)
        # BEGIN configure the checked physical domain through the native API.
        # The factory does not expose this dataclass field. Set it before
        # constructing the solver so the optional LM pre-solver cannot use its
        # separate narrowed/clamped domain. Halton seeds and native L-BFGS still
        # supply the declared branches through the domain-restricted core.
        ik_cfg.use_lm_seed = False
        # END no independent optimizer may precede the checked native core.
        # BEGIN evaluate the declared alternative IK branches before returning.
        # Native early exit only requires one successful seed per batch.
        # Even one requested trajectory can benefit from choosing among the
        # several IK postures already admitted by its separate seed budget.
        if budget.ik_seeds > 1:
            ik_cfg.exit_early = False
        # END evaluate the declared alternative IK branches before returning.
        self.ik = InverseKinematics(ik_cfg)
        self.joint_names = tuple(self.ik.joint_names)
        if (set(self.joint_names) != set(self.robot_metadata["active_joint_names"])
                or self.ik.kinematics.total_spheres != self.robot_metadata["spheres"]):
            raise RuntimeError("Native robot layout differs from its checked metadata")
        columns = [problem.joint_names.index(name) for name in self.joint_names]
        self.start = JointState.from_numpy(list(self.joint_names), problem.start_q[columns][None],
                                          velocity=problem.start_dq[columns][None], device_cfg=self.device_cfg)
        self._verification_fk = Kinematics(self.ik.kinematics.config, compute_jacobian=True)
        self._validate_occupied_robot_geometry()
        self._orientation_vjp = _orientation_vjp_type()
        self._native_layout = _native_gradient_layout_type()
        self._verification_scene = scene.clone_for_rollout()
        transition = deepcopy(load_yaml(join_path(get_task_configs_path(), "trajopt/transition_bspline_trajopt.yml")))
        transition["transition_model_cfg"].update(n_knots=budget.knots, interpolation_steps=budget.interpolation_steps)
        self._trajectory_configuration = {
            "robot": deepcopy(robot_dict),
            "optimizer_configs": [trajectory_optimizer],
            "metrics_rollout": configuration("metrics_base.yml", "trajectory_metrics"),
            "transition_model": transition, "device_cfg": self.device_cfg,
            "num_seeds": budget.trajectory_seeds, "max_batch_size": 1, "max_goalset": 1,
            "use_cuda_graph": False, "random_seed": budget.random_seed,
            # BEGIN inert native pose storage has no trajectory acceptance role.
            "position_tolerance": problem.numerical.position_m,
            "orientation_tolerance": problem.numerical.rotation_rad,
            # END typed functional requirements own their original tolerances.
            "cost_manager_config_instance_type": self._manager_type,
            "override_optimizer_num_iters": {"lbfgs": budget.trajectory_iterations},
            # Native retiming runs even with zero finetuning attempts. A
            # different dt changes derivative constraints and, for nonzero
            # measured dq0, the cubic's fixed boundary controls. Keep one
            # request-wide physical clock through optimization and metrics.
            "minimum_trajectory_dt": budget.maximum_trajectory_dt,
            "maximum_trajectory_dt": budget.maximum_trajectory_dt,
            "interpolation_dt": budget.interpolation_dt,
            "interpolation_buffer_size": budget.interpolation_buffer_size,
        }

        class ProcessTrajectoryOptimizer(TrajectoryOptimizer):
            # BEGIN install sessions on each construction, including grid rebuilds.
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.end_source_world_reuse = _bind_candidate_source_world_reuse(self)
                self._source_world_solve_active = False

            def solve_pose(self, *args, **kwargs):
                # BEGIN a new native call cannot inherit a previous solve's batches.
                self.end_source_world_reuse()
                self._source_world_solve_active = True
                # END observations below share only this call's frozen binding.
                try:
                    return super().solve_pose(*args, **kwargs)
                finally:
                    self._source_world_solve_active = False
                    self.end_source_world_reuse()
            # END an error between ordinary metrics and interpolation cannot leak.

            # BEGIN explicit solver replacement also releases bounded CPU evidence.
            def destroy(self):
                self.end_source_world_reuse()
                return super().destroy()
            # END a rebuilt objective starts with two empty source batch slots.

            def get_interpolated_trajectory(self, js_optimized):
                # BEGIN preserve raw controls and boundary derivatives on one clock.
                from curobo._src.curobolib.cuda_ops.trajectory import get_bspline_interpolation
                from curobo._src.util.trajectory import TrajInterpolationType

                grid = _formal_native_clock(self)
                if self.config.interpolation_type != TrajInterpolationType.BSPLINE_KNOTS_CUDA:
                    raise ValueError("Exact native cubic export requires the configured B-spline interpolation")
                batch = js_optimized.position.shape[0]
                if (js_optimized.position.ndim != 3 or not 1 <= batch <= self.config.num_seeds
                        or js_optimized.position.shape[1:] != (self.horizon, self.action_dim)
                        or js_optimized.knot is None
                        or js_optimized.knot.shape != (batch, self.action_horizon, self.action_dim)):
                    raise ValueError("Native interpolation state does not match the bounded active-joint layout")
                # Only a few timing scalars synchronize here. No trajectory
                # array is copied to CPU, and no action or derivative is changed.
                for name, expected in (("dt", grid["native_dt"]), ("knot_dt", grid["knot_dt"])):
                    value = getattr(js_optimized, name)
                    if (value is None or value.numel() != batch
                            or value.dtype != torch.float32 or not bool((value == expected).all())):
                        raise ValueError("Native result changed the fixed request-wide spline clock")
                capacity = self.config.interpolation_buffer_size
                if (grid["valid_count"] > capacity or capacity <= self.horizon or capacity > 2048):
                    raise ValueError("Native interpolation exceeds its declared fixed buffer capacity")
                shape = (batch, capacity, self.action_dim)
                if self._interpolated_traj_buffer is None or self._interpolated_traj_buffer.shape[0] != batch:
                    self._interpolated_traj_buffer = JointState.zeros(shape, self.device_cfg,
                                                                     joint_names=self.joint_names)
                elif self._interpolated_traj_buffer.shape != shape:
                    raise ValueError("Native interpolation buffer changed its declared shape")
                if self._interpolation_dt_buffer is None or self._interpolation_dt_buffer.shape != (batch,):
                    self._interpolation_dt_buffer = torch.full((batch,), grid["interpolation_dt"],
                        dtype=torch.float32, device=self.device_cfg.device)
                elif not bool((self._interpolation_dt_buffer == grid["interpolation_dt"]).all()):
                    raise ValueError("Native interpolation buffer changed its exact output dt")
                counts = torch.full((batch,), grid["valid_count"], dtype=torch.int32,
                                    device=self.device_cfg.device)
                goal = self.core._goal_buffer
                js_optimized.control_space = self.transition_model.control_space
                state = get_bspline_interpolation(js_optimized, self._interpolated_traj_buffer,
                    self._interpolation_dt_buffer, current_state=goal.current_js,
                    goal_state=goal.seed_goal_js, start_idx=goal.idxs_current_js,
                    goal_idx=goal.idxs_seed_goal_js,
                    use_implicit_goal_state=goal.seed_enable_implicit_goal_js,
                    interpolated_horizon=counts, bspline_degree=3)
                rollout = self.additional_metrics_rollouts["interpolated_rollout"]
                # Both managers retain valid-prefix masking before the native
                # caller computes dense metrics and its original success flag.
                rollout.constraint_manager.set_dense_counts(counts)
                rollout.metrics_constraint_manager.set_dense_counts(counts)
                return state, counts, False
                # END no native resize, retiming, interpolation switch or metric bypass.

        self._optimizer_class = ProcessTrajectoryOptimizer
        self._optimizer_config_class = TrajectoryOptimizerCfg
        self.trajopt = ProcessTrajectoryOptimizer(TrajectoryOptimizerCfg.create(**deepcopy(self._trajectory_configuration)))
        self._bind_formal_clock()
        self._initialize_verification_constraints()
        self._bind_physical_domains()

    # BEGIN bind the independently controlled fingers to native formal output.
    def _bind_formal_clock(self):
        # BEGIN confirm native construction preserved the already bound inputs.
        native = _formal_native_clock(self.trajopt)
        if any(native.get(name) != value for name, value in self.formal_clock.items()):
            raise ValueError("Native solver changed the admitted synchronized clock")
        self.formal_clock.update(native)
        # END do not regenerate hand commands after private rollouts were cloned.

    def _seed_u(self, progress):
        """Choose the earliest plateau point for initializer-only state queries."""
        values, indices = np.unique(self.progress_map.s, return_index=True)
        return float(np.interp(progress, values, self.progress_map.u[indices]))
    # END formal export always evaluates its actual time, not this seed inverse.

    # BEGIN bind occupancy evidence to the actual bounded current robot model.
    def _validate_occupied_robot_geometry(self):
        """Verify q0 spheres and requested link poses before trajectory setup.

        Collision spheres come from the actual planner's FK configuration.
        Additional pose frames use the same already checked robot dictionary
        and cannot add links, joints, meshes or optimizer tool targets. CPU
        float64 frame transforms match the occupancy adapter's evidence path.
        """
        from scipy.spatial.transform import Rotation
        from curobo.kinematics import Kinematics, KinematicsCfg

        required = self.scene.required_robot_links
        with self.torch.no_grad():
            state = self._verification_fk.compute_kinematics(self.JointState.from_position(
                self.start.position[:, None].contiguous(), joint_names=list(self.joint_names)))
            spheres = state.robot_spheres[0, 0].detach().cpu().numpy().astype(float)
            poses = {}
            if required:
                # The preflight already proved every requested link belongs
                # to this selected tree. Only this single-state FK exposes
                # more frames; IK retains its one-tool seed allocation.
                pose_fk = Kinematics(KinematicsCfg.from_robot_yaml_file(
                    deepcopy(self.robot_dict), tool_frames=list(required), device_cfg=self.device_cfg),
                    compute_spheres=False, compute_com=False)
                if (set(pose_fk.joint_names) != set(self.joint_names)
                        or not set(pose_fk.config.kinematics_config.link_name_to_idx_map)
                        <= set(self.robot_metadata["selected_link_names"])):
                    raise ValueError("Occupied-link FK changed the bounded planning robot tree")
                columns = [self.joint_names.index(name) for name in pose_fk.joint_names]
                posed = pose_fk.compute_kinematics(self.JointState.from_position(
                    self.start.position[:, None, columns].contiguous(), joint_names=pose_fk.joint_names))
                positions = posed.tool_poses.position[0, 0].detach().cpu().numpy().astype(float)
                quaternions = posed.tool_poses.quaternion[0, 0].detach().cpu().numpy().astype(float)
                for name in required:
                    index = pose_fk.tool_frames.index(name)
                    pose = np.eye(4)
                    pose[:3, 3] = positions[index]
                    pose[:3, :3] = Rotation.from_quat(quaternions[index, [1, 2, 3, 0]]).as_matrix()
                    poses[name] = self.problem.base_pose@pose
        spheres[:, :3] = spheres[:, :3]@self.problem.base_pose[:3, :3].T+self.problem.base_pose[:3, 3]
        self.scene.validate_robot_geometry(spheres, poses)
    # END bind occupancy evidence to the actual bounded current robot model.

    def _initialize_verification_constraints(self):
        from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
        from curobo._src.state.state_robot import RobotState
        self._RobotState = RobotState
        # The base manager supplies native joint/derivative limits at extra
        # control-curve samples. Source self collision is checked separately
        # using the same scene interface as metrics and the planning entrance.
        self._verification_native = RobotCostManager(self.device_cfg)
        self._verification_native.initialize_from_config(
            deepcopy(self.trajopt.config.metrics_rollout_config.constraint_cfg),
            self.trajopt.transition_model)

    # BEGIN constrain the actual cubic throughout native initialization/updates.
    def _bind_physical_domains(self):
        # BEGIN FUNCTO owns the optimizer/goal extensions on its solver instances.
        from .curobo_solver_components import bind_solver_components

        limits = self._verification_native.get_cost("cspace").config.joint_limits
        columns = [limits.joint_names.index(name) for name in self.joint_names]
        bounds = limits.position[:, columns]
        self.position_domain = JointPositionDomain(bounds[0], bounds[1])
        self.physical_domain = CubicJointDomain(bounds[0], bounds[1],
            self.start.position, self.start.velocity, self.formal_clock["knot_dt"],
            self.trajopt.action_horizon)
        # BEGIN solver metrics remain eager; only admitted CUDA cores capture.
        cuda = self.device_cfg.device.type == "cuda"
        bind_solver_components(self.ik, self.position_domain, use_optimizer_graph=cuda)
        bind_solver_components(self.trajopt, self.physical_domain,
            use_optimizer_graph=cuda and self._trajectory_optimizer_graph_admitted)
        # END native fixed-iteration/kernel requirements remain explicit errors.
        # END the installed cuRobo library needs no physical-domain/cache patch.
    # END limits are the model's full physical q interval, never a seed box.

    # BEGIN reject an immutable infeasible entrance before running native IK.
    def initial_state_diagnostics(self):
        """Check only necessary conditions known at the measured entrance.

        The complete native STATE cost requires acceleration, jerk and torque
        inputs. Those measurements are absent, so inspect its effective q/dq
        limits directly and query the bound source solids for self collision.
        Scene terms retain their normal conservative meaning, including
        unknown space and a binary gripper's reachable envelope; a positive
        term need not mean that the actual robot is physically in contact.
        """
        torch = self.torch
        cspace = self._verification_native.get_cost("cspace")
        if cspace is None or not cspace.enabled:
            raise RuntimeError("Entrance checks require the enabled native cspace constraints")
        limits = cspace.config.joint_limits
        if (len(limits.joint_names) != len(self.joint_names)
                or set(limits.joint_names) != set(self.joint_names)):
            raise RuntimeError("Native entrance limits must match the active feedback joints")
        columns = [limits.joint_names.index(name) for name in self.joint_names]
        q, dq = self.start.position[:, None], self.start.velocity[:, None]
        joint_bounds = {}
        with torch.no_grad():
            components = {}
            for name, values in (("position", q), ("velocity", dq)):
                bounds = getattr(limits, name)[:, columns]
                violations = torch.relu(bounds[0]-values) + torch.relu(values-bounds[1])
                components[f"native:{name}"] = violations
                joint_bounds[name] = {
                    "measured": values[0, 0].detach().cpu().tolist(),
                    "lower": bounds[0].detach().cpu().tolist(),
                    "upper": bounds[1].detach().cpu().tolist(),
                    "violation": violations[0, 0].detach().cpu().tolist(),
                }
            # Construct only the measured fields. Native from_position/from_numpy
            # helpers otherwise synthesize zero acceleration and jerk tensors.
            measured = self.JointState(position=q.contiguous(), velocity=dq.contiguous(),
                                       joint_names=list(self.joint_names))
            state = self._verification_fk.compute_kinematics(measured)
            # BEGIN the fixed entrance uses measured coordinates and the complete source pair policy.
            source_self_evidence = {}
            components["source_self_collision"] = self._verification_scene.check_robot_self_collision(
                q, self.joint_names, q.new_zeros(q.shape[:2]), diagnostics=source_self_evidence)
            # END a sphere overlap cannot reject a source-separated entrance.
            tool = state.tool_poses
            tool_index = tool.tool_frames.index(self.problem.tool_link)
            position, quaternion = self.problem.transform_tool_pose(
                tool.position[:, :, tool_index], tool.quaternion[:, :, tool_index])
            # BEGIN archive arithmetic differences at identical native entrance coordinates.
            # The source validator's model tolerance describes double-precision
            # model comparisons; it is not a new float32 or task acceptance band.
            from scipy.spatial.transform import Rotation
            validator = self.robot_validator
            full_q, _ = validator._planning_positions(q[0, 0].detach().cpu().numpy(), self.joint_names,
                time_s=0., gripper=self._verification_scene.gripper)
            source_state = validator.checker.query(joint_names=validator.joint_names,
                joint_positions=full_q, T_reference_base=validator.T_reference_base,
                pairs=(), frames=[validator.tool_frame])
            source_tcp = np.asarray(source_state["frame_poses"][validator.tool_frame]) @ validator.T_frame_tcp
            native_tcp = np.eye(4)
            native_tcp[:3, 3] = position[0, 0].detach().cpu().numpy()
            native_tcp[:3, :3] = Rotation.from_quat(
                quaternion[0, 0].detach().cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
            model_fk = {**validator._pose_difference(native_tcp, source_tcp),
                "joint_names": list(validator.joint_names), "joint_position": full_q.tolist(),
                "native_dtype": str(q.dtype), "source_dtype": "float64",
                "scope": "same native entrance coordinates; numerical diagnostic, not a collision or task tolerance"}
            # END both geometric interpretations retain their original numeric precision.
            progress = q.new_full(q.shape[:2], self.problem.start_progress)
            unresolved_scene = {}
            # BEGIN the entrance uses the same source world law as every candidate.
            source_world_evidence, scene_components = {}, {}
            _scene_boundary_evaluate(self._verification_scene, self.problem, self.progress_map,
                self.problem.transform_robot_spheres(state.robot_spheres), position,
                quaternion, progress, q, self.joint_names,
                trajectory_time=torch.zeros_like(progress), normalized_time=torch.zeros_like(progress),
                source_geometry=True, diagnostics=source_world_evidence,
                source_components_out=scene_components)
            for name, value in scene_components.items():
                if name.startswith("unknown/") and not name.startswith("unknown/release_anchor/"):
                    unresolved_scene[name] = float(value.max().item())
                else:
                    components[f"scene:{name}"] = value
            # END individual terms retain the worst same-state material mode.
            residuals = {name: float(value.max().item()) for name, value in components.items()}
            coverage = self._verification_scene.check_robot_coverage(
                q, self.joint_names, torch.zeros_like(progress), entrance=True)
            if coverage:
                residuals["scene:unknown/new_robot_volume"] = float(any(not item["attempt_allowed"] for item in coverage))
        failed = [name for name, value in residuals.items() if not np.isfinite(value) or value > 0]
        # BEGIN preserve nonfinite failures in strict JSON diagnostics.
        # JSON null plus the explicit component list retains the rejection
        # without emitting NaN/Infinity or losing the report during saving.
        nonfinite = [name for name, value in residuals.items() if not np.isfinite(value)]
        for name in nonfinite:
            residuals[name] = None
        # END preserve nonfinite failures in strict JSON diagnostics.
        return {
            "passed": not failed, "residuals": residuals, "failed_components": failed,
            "nonfinite_components": nonfinite,
            "joint_names": list(self.joint_names), "joint_bounds": joint_bounds,
            "robot_coverage": coverage, "unresolved_scene": unresolved_scene,
            "source_self_collision": source_self_evidence,
            "source_world_collision": source_world_evidence,
            "source_native_fk": model_fk,
            "scope": {
                "kind": "necessary_conditions_at_fixed_measured_entrance",
                "progress": self.problem.start_progress, "active_joint_position_velocity": "native_effective_limits",
                "source_self_collision_checked": True,
                "scene": "source robot geometry against the same selected observed/native owner geometry",
                "not_checked": ["acceleration", "jerk", "torque", "unprovided_locked_joint_velocity",
                                "future_trajectory_feasibility", "physical_task_success"],
            },
        }
    # END reject an immutable infeasible entrance before running native IK.

    def close(self):
        # BEGIN release native callback cycles at the request lifecycle boundary.
        # Native destroy() resets CUDA graphs. LBFGS and GradientOptCore also
        # retain each other through bound callbacks, so drop every solver
        # reference before collecting their now-unreachable tensor workspaces.
        # BEGIN restore caller precision even if native destruction raises.
        try:
            for name in ("ik", "trajopt"):
                solver = getattr(self, name)
                if solver is not None:
                    solver.destroy()
                setattr(self, name, None)
            del solver
            gc.collect()
        finally:
            prior = getattr(self, "_caller_precision", None)
            if prior is not None:
                self._precision_torch.backends.cuda.matmul.allow_tf32 = prior[0]
                self._precision_torch.backends.cudnn.allow_tf32 = prior[1]
                self._caller_precision = None
        # END repeated close() cannot overwrite a later caller setting.
        # END release native callback cycles at the request lifecycle boundary.

    def _goal(self, progress):
        s = self.device_cfg.to_device([[float(progress)]])
        position, quaternion = self.problem.reference_to_link_pose(*self.problem.sample_reference(s))
        return self.GoalToolPose(tool_frames=[self.problem.tool_link],
                                 position=position[:, :, None, None].contiguous(),
                                 quaternion=quaternion[:, :, None, None].contiguous())

    # BEGIN whole-chain selection and native-basis soft reference fitting.
    def _seed_velocity_bounds(self):
        """Read existing physical speed bounds in the action's named joint order."""
        limits = self._verification_native.get_cost("cspace").config.joint_limits
        columns = [limits.joint_names.index(name) for name in self.joint_names]
        return limits.velocity.detach().cpu().numpy()[:, columns]

    def reference_seeds(self):
        """Build bounded initializers from explicit reference-query evidence.

        Native outputs retain their diagnostics, but only successful pose IK
        passing static task/source checks enters the next warm start or the
        anchor graph. Failed queries are filled from successful anchors and
        measured q0, never from a failed native posture. Samples are fitted in
        the actual native cubic representation before becoming optimizer actions.
        """
        from .planning_ik import select_ik_chains
        torch, count = self.torch, self.budget.trajectory_seeds
        velocity_bounds = self._seed_velocity_bounds()
        previous = self.start.position[:, None].clone()
        warm_start_query = None
        control_u = (np.arange(self.trajopt.action_horizon)+3.)/(self.trajopt.action_horizon+4.)
        pools, evidence, success_counts = [], [], []
        self.reference_ik_queries, self.reference_ik_distinct_counts = [], []
        for query_index, u in enumerate(control_u):
            progress = float(np.interp(u, self.progress_map.u, self.progress_map.s))
            self.seed_parameter.fill_(float(u))
            result = self.ik.solve_pose(self._goal(progress), current_state=self.start,
                                       seed_config=previous, return_seeds=self.budget.ik_seeds)
            values = self.ik.get_active_js(result.js_solution).position[0].detach().cpu().numpy()
            flags = result.success[0].detach().cpu().numpy()
            success_counts.append(int((flags & np.isfinite(values).all(-1)).sum()))
            ranked, reports = self._rank_ik_candidates(values, flags,
                previous[0].detach().cpu().numpy(), float(u))
            # BEGIN failed reference poses remain evidence, never future warm starts.
            qualified = [int(index) for index in ranked if all(reports[index][name] for name in
                ("native_pose_ik_success", "task_satisfied", "source_valid"))]
            self.reference_ik_distinct_counts.append(len(qualified))
            for index, report in enumerate(reports):
                report["reference_status"] = ("SOLVED" if np.isfinite(values[index]).all()
                    and all(report[name] for name in
                            ("native_pose_ik_success", "task_satisfied", "source_valid"))
                    else "FAILED_REFERENCE")
            # Native result errors describe convergence to the nominal pose;
            # static task and geometry evidence above remain independent.
            for attribute, name in (("position_error", "nominal_position_error_m"),
                                    ("rotation_error", "nominal_rotation_error_rad")):
                measurement = getattr(result, attribute, None)
                if measurement is not None:
                    errors = measurement[0].detach().cpu().numpy().reshape(-1)
                    if errors.shape != (len(reports),):
                        raise ValueError("Native IK convergence errors must align with returned candidates")
                    for report, error in zip(reports, errors):
                        report[name] = float(error) if np.isfinite(error) else None
            pools.append(values)
            evidence.append(reports)
            self.reference_ik_queries.append({"u": float(u),
                "time_s": float(u*self.formal_clock["duration"]),
                "reference_status": "SOLVED" if qualified else "FAILED_REFERENCE",
                "seed_status": "SOLVED" if qualified else "FALLBACK_SEED",
                "qualified_candidate_indices": qualified,
                "warm_start_from_query": warm_start_query,
                "candidates": [{"joint_position": [float(value) if np.isfinite(value) else None
                                                    for value in row], "report": report}
                               for row, report in zip(values, reports)]})
            if qualified:
                # Static checks apply at this query only. Carrying this pool
                # supplies numerical IK initializers, not acceptance evidence
                # for later task modes, permissions, or physical times.
                previous = self.device_cfg.to_device(values[qualified][None]).contiguous()
                warm_start_query = query_index
            # END an empty/failed query leaves the most recent successful pool intact.
        chains = select_ik_chains(pools, evidence, self.start.position.detach().cpu().numpy(),
            control_u*self.formal_clock["duration"], velocity_bounds, count=count,
            joint_tolerance=self.problem.numerical.joint)
        self.reference_ik_chains = chains
        # BEGIN failure gaps do not erase distinct successful anchor branches.
        # Every returned chain contains at least one qualified native anchor;
        # interpolation fills its missing samples without using failed IK.
        retained = chains[:count]
        targets = [np.asarray(chain["joint_positions"]) for chain in retained]
        self.initial_seed_strategies = ["reference_ik" if chain["reference_success"]
                                       else "reference_ik_with_fallback" for chain in retained]
        if len(retained) > 1:
            self.initial_seed_strategies[1] += "_alternative"
        while len(targets) < count:
            targets.append(np.broadcast_to(self.start.position[0].detach().cpu().numpy(),
                                            (self.trajopt.action_horizon, len(self.joint_names))).copy())
            self.initial_seed_strategies.append("measured_state_constant")
        # END all-failed references use only the independent measured-state strategy.
        controls, self.reference_seed_fits = [], []
        for branch, target in enumerate(targets):
            if self.initial_seed_strategies[branch] == "measured_state_constant":
                value = self.trajopt._trajectory_seed_generator.generate_constant_seeds(self.start.position, 1)
                control = self.physical_domain.initialize(value).reshape(self.trajopt.action_horizon, -1)
                fit = {"success": True, "source": "measured_state_constant_free_controls"}
            else:
                control, fit = self.physical_domain.fit_reference(control_u,
                    self.device_cfg.to_device(target), velocity_bounds=velocity_bounds)
                if control is None:
                    # Optional reference fitting cannot prove a task impossible.
                    # Keep its failure evidence and use the existing measured
                    # initializer, whose dynamics still require Phase A checks.
                    value = self.trajopt._trajectory_seed_generator.generate_constant_seeds(self.start.position, 1)
                    control = self.physical_domain.initialize(value).reshape(self.trajopt.action_horizon, -1)
                    self.initial_seed_strategies[branch] = "measured_state_after_reference_fit_failure"
                    targets[branch] = self.physical_domain.evaluate(control, control_u)
                    fit = {**fit, "initializer_after_failure": "measured_state_constant",
                           "initializer_scope": "Position legal; dynamic/task constraints require optimization"}
            controls.append(control)
            self.reference_seed_fits.append(fit)
        seeds = torch.stack(controls)[None]
        pool = self._build_goal_ik_candidates(seeds)
        self.goal_ik_fit_hints = []
        for branch in range(count):
            samples, postures = list(control_u), list(targets[branch])
            for parameter, entry in sorted(pool.items()):
                candidates = [row for row in entry["candidates"]
                              if row["report"]["task_satisfied"] and row["report"]["source_valid"]]
                if not candidates or parameter == 0.:
                    continue
                rank = min(branch, len(candidates)-1)
                samples.append(float(parameter))
                postures.append(candidates[rank]["joint_position"])
                self.goal_ik_fit_hints.append({"event_u": float(parameter), "branch": branch,
                    "rank": rank, "scope": "Soft observation at the exact event time; no interpolation equality"})
            if len(samples) == len(control_u):
                continue
            control, fit = self.physical_domain.fit_reference(np.asarray(samples),
                self.device_cfg.to_device(np.asarray(postures)), prior_controls=seeds[0, branch],
                velocity_bounds=velocity_bounds)
            self.reference_seed_fits[branch]["functional_hint_fit"] = fit
            for hint in self.goal_ik_fit_hints:
                if hint["branch"] == branch:
                    hint["applied"] = control is not None
            if control is None:
                # A numerical failure to fit optional hints must not erase the
                # preceding initializer or remove the actual hard event goals.
                self.initial_seed_strategies[branch] += "+functional_hint_fit_failed"
                continue
            seeds[0, branch] = control
            self.initial_seed_strategies[branch] += "+ranked_functional_ik"
        self.physical_domain.validate(seeds)
        self.seed_strategies = list(self.initial_seed_strategies)
        return seeds.contiguous(), success_counts
    # END q0/dq0, native clock, hard task checks and admitted seed count are fixed.

    # BEGIN source-supported spatial refinement precedes each frozen solve.
    def refine_background_for_seed(self, seed, *, rebuild=True):
        """Rebuild local background guidance from this initializer's witnesses.

        A witness is a complete body's positive coarse proxy query at a real
        physical time. Sample the same native regular grid and exact anchors
        used by trajectory costs. These extra FK/proxy queries propose map
        resolution only; they neither certify the curve nor consume a source
        geometry acceptance pass. The request's wall deadline still applies.
        Every later solve starts here, including a changed OMPL initializer.
        """
        from .voxel_layers import LayeredVoxelQuery

        background = self.scene.background
        if (not isinstance(background, LayeredVoxelQuery)
                or getattr(background, "refinement_source", None) is None):
            return {"updated": False, "reason": "no_observed_background_refinement_source"}
        self.physical_domain.validate(seed)
        anchors = self.progress_map.anchor_parameters(self.anchors)[:, 0]
        parameters = np.unique(np.r_[np.linspace(0., 1., self.trajopt.horizon),
                                     anchors, self.refinement_parameters])
        # CPU cubic evaluation uses the actual free controls and fixed q0/dq0;
        # interpolating those free controls themselves would be another curve.
        controls = seed[0].detach().cpu().numpy()
        q = self.physical_domain.evaluate(controls, parameters)
        progress = np.interp(parameters, self.progress_map.u, self.progress_map.s)
        times = parameters*self.formal_clock["duration"]
        check = getattr(self, "_check_validation_work", None)

        def collect(scene):
            rows = []
            with self.torch.no_grad():
                for begin in range(0, len(parameters), _VALIDATION_CHUNK_SIZE):
                    if check is not None:
                        check()
                    end = min(begin+_VALIDATION_CHUNK_SIZE, len(parameters))
                    joints = self.device_cfg.to_device(q[:, begin:end]).contiguous()
                    state = self._verification_fk.compute_kinematics(self.JointState.from_position(
                        joints, joint_names=list(self.joint_names)))
                    tool = state.tool_poses
                    index = tool.tool_frames.index(self.problem.tool_link)
                    position, quaternion = self.problem.transform_tool_pose(
                        tool.position[:, :, index], tool.quaternion[:, :, index])
                    spheres = self.problem.transform_robot_spheres(state.robot_spheres)
                    s = self.device_cfg.to_device(progress[None, begin:end]).expand(len(controls), -1)
                    t = self.device_cfg.to_device(times[None, begin:end]).expand(len(controls), -1)
                    rows.extend(scene.background_refinement_witnesses(
                        spheres, position, quaternion, s, joints, self.joint_names,
                        trajectory_time=t, include_satisfied=True))
            return rows

        def summarize(rows):
            # Costs sum sphere penalties at the sampled body/time queries;
            # they are neither signed distances nor source collision verdicts.
            bodies = {}
            for row in rows:
                body = bodies.setdefault(row["body_id"], {"queries": 0, "positive_spheres": 0,
                    "source_supported_queries": 0, "full_fine_weight_queries": 0,
                    "coarse_cost_sum": 0., "guidance_cost_sum": 0.,
                    "fine_cost_sum_on_supported_queries": 0.})
                body["queries"] += 1
                body["positive_spheres"] += row["positive_spheres"]
                body["source_supported_queries"] += int(row["fine_supported"])
                body["full_fine_weight_queries"] += int(row["guidance_weight"] == 1.)
                body["coarse_cost_sum"] += row["coarse_cost"]
                body["guidance_cost_sum"] += row["guidance_cost"]
                if row["fine_supported"]:
                    body["fine_cost_sum_on_supported_queries"] += row["fine_cost"]
            return {"body_queries": len(rows), "bodies": bodies,
                    "largest_coarse_query": None if not rows else max(rows, key=lambda row: row["coarse_cost"]),
                    "scope": "Same cubic and times; proxy penalties only, not geometry acceptance"}

        before = collect(self.scene)
        witnesses = [row for row in before if row["guidance_weight"] < 1.]
        if check is not None:
            check()
        # BEGIN host construction refusals preserve the still-running scene.
        # Source-array/tree allocation is admitted inside the builder, before
        # a replacement exists. Only typed capacity refusals are recoverable;
        # mapper/API/model errors retain their original exception identity.
        construction_refusal = None
        try:
            replacement, report = self.scene.refine_background(witnesses)
        except ResourceLimitError as error:
            construction_refusal = {"resource": error.resource, "required_bytes": error.required_bytes,
                                    "limit_bytes": error.limit_bytes, "reason": str(error)}
        if construction_refusal is not None:
            gc.collect()
            if check is not None:
                check()
            return {"status": "resource_refused", "updated": False,
                    "resource_refusal": construction_refusal, "sampled_times": len(parameters),
                    "sampled_seeds": len(controls), "before": summarize(before)}
        # END failed construction never reaches scene/solver replacement.
        report = {**report, "sampled_times": len(parameters), "sampled_seeds": len(controls),
                  "updated": replacement is not None, "before": summarize(before)}
        if replacement is None:
            return report
        # BEGIN admit the complete replacement while the working scene survives.
        # Mapping's own capacity check does not include source-solid snapshots,
        # dense query scratch, candidate records or planner FK. None of those
        # limits may be discovered only after replacing the old scene/solver.
        # Live GPU admission is deliberately conservative: the old solver is
        # still allocated and its storage is not promised as already reusable.
        refusal = None
        try:
            storage = _validation_storage_admission(self.problem, replacement, self.robot_metadata, self.budget)
            capacity = _anchor_capacity(self.progress_map, self.anchors, self.budget, self.refinement_parameters)
            workspace = _planning_workspace_bytes(replacement, self.robot_metadata, self.budget, capacity)
            _require_planning_host_memory(workspace, max_host_rss_bytes=self.budget.max_host_rss_bytes)
            workspace = _require_planning_memory(self.torch, self.device_cfg.device, workspace,
                                                 max_device_used_bytes=self.budget.max_device_used_bytes)
        except ResourceLimitError as error:
            refusal = {"resource": error.resource, "required_bytes": error.required_bytes,
                       "limit_bytes": error.limit_bytes, "reason": str(error)}
        if refusal is not None:
            # Leave the exception scope before collection: its traceback would
            # otherwise keep the refused scene alive through the preflight frame.
            del replacement
            gc.collect()
            report.update(status="resource_refused", updated=False, resource_refusal=refusal)
            if check is not None:
                check()
            return report
        report["planner_admission"] = {"scene_workspace_bytes": workspace.total_bytes,
            "scene_cpu_workspace_bytes": workspace.cpu_bytes, "scene_gpu_workspace_bytes": workspace.gpu_bytes,
            "anchor_fk_workspace_bytes": workspace.anchor_fk_bytes, "trajectory_graph": workspace.trajectory_graph,
            "validation_storage": deepcopy(storage)}
        self.validation_storage = storage
        # END ordinary geometry/shape/runtime errors are never resource refusals.
        # BEGIN source geometry and guidance have separate replacement scopes.
        # The factory rebinds the source checker only for a new observed field;
        # changed guidance boxes retain the existing immutable source shapes.
        # Either replacement changes rollout inputs, so rebuild all trajectory
        # rollouts/captures while retaining the frozen hand/trajectory clock.
        # END a reused field does not imply unchanged optimization guidance.
        self.scene = replacement
        self._verification_scene = replacement.clone_for_rollout()
        # BEGIN outer AL/working-set transitions combine all changes into one rebuild.
        if rebuild:
            self.refine_validation(self.refinement_parameters)
        # END callers deferring this rebuild must do it before the next solve.
        report["after"] = summarize(collect(replacement))
        # IK seed metrics use only native pose/domain and robot self geometry;
        # they do not query background fields. Static IK screening and OMPL use
        # this new verification scene, so the unchanged IK core needs no rebuild.
        if check is not None:
            check()
        return report
    # END no map or blend-region mutation occurs inside an optimizer block.

    def refine_validation(self, refinement_parameters, *, phase=None):
        """Rebuild only validation workspaces in the already frozen binding.

        Release old solver workspaces before constructing new rollouts. IK and
        immutable robot/scene configuration are retained; no second planner
        and no duplicate native roadmap is allocated.
        """
        # BEGIN apply the same point and memory admission to every new clock.
        progress_map = self.progress_map
        # BEGIN replace gradient query slots; the independent witness ledger survives.
        parameters = np.unique(np.asarray(refinement_parameters, float))
        selected_phase = self.phase if phase is None else phase
        if selected_phase not in {"feasibility", "quality"}:
            raise ValueError("Unknown trajectory optimization phase")
        # END capacity is admitted before destroying the working solver.
        capacity = _anchor_capacity(progress_map, self.anchors, self.budget, parameters)
        workspace = _planning_workspace_bytes(self.scene, self.robot_metadata, self.budget, capacity)
        _require_planning_host_memory(workspace, max_host_rss_bytes=self.budget.max_host_rss_bytes)
        self.trajopt.destroy()
        self.trajopt = None
        # Native destroy() releases graphs, while optimizer callbacks can keep
        # rollout tensors in a reference cycle. Collect only at this explicit
        # replacement boundary before querying live allocator statistics.
        gc.collect()
        workspace = _require_planning_memory(self.torch, self.device_cfg.device, workspace,
                                             max_device_used_bytes=self.budget.max_device_used_bytes)
        self.refinement_parameters = parameters
        self.phase = selected_phase
        self.anchor_capacity = capacity
        self.scene_workspace_bytes = workspace.total_bytes
        self.scene_cpu_workspace_bytes, self.scene_gpu_workspace_bytes = workspace.cpu_bytes, workspace.gpu_bytes
        self.anchor_fk_workspace_bytes = workspace.anchor_fk_bytes
        self._trajectory_optimizer_graph_admitted = workspace.trajectory_graph
        manager_type = _process_manager_type(self.problem, self.scene, progress_map, self.anchors,
                                             self.budget, parameters,
                                             clock=self.formal_clock, seed_parameter=self.seed_parameter,
                                             robot_dict=self.robot_dict, phase=self.phase,
                                             functional_constraints=self.functional_constraints)
        # BEGIN rebuilding avoids stale captured Python branches and cost weights.
        optimizer = deepcopy(self._quality_optimizer_config)
        if self.phase == "feasibility":
            optimizer["rollout"]["cost_cfg"]["cspace_cfg"]["squared_l2_regularization_weight"] = [0.]*5
        self._trajectory_configuration["optimizer_configs"] = [optimizer]
        # END one solver at a time preserves the admitted memory bound.
        self._trajectory_configuration["cost_manager_config_instance_type"] = manager_type
        self.trajopt = self._optimizer_class(
            self._optimizer_config_class.create(**deepcopy(self._trajectory_configuration)))
        # BEGIN refinement may add queries but cannot regenerate hand commands.
        if _formal_native_clock(self.trajopt) != self.formal_clock:
            raise ValueError("Validation refinement changed the frozen native clock")
        self._initialize_verification_constraints()
        self._bind_physical_domains()
        # END schedule changes require a separately bound planning request.

    # BEGIN scoped spatial search on one already bound physical clock.
    def ompl_seed(self, initial_seed, candidate_trajectory):
        """Return a spatial initializer and evidence; never produce a new clock.

        Only real relation switches, goal times and scene switches divide
        local OMPL charts. Their boundary configurations are sampled and
        projected into the original allowed sets, then revised on failure.
        Reference controls initialize these proposals without fixing them.
        A selected prefix supplies its actual release FK to its child scene.
        """
        torch = self.torch
        deadline = time.monotonic()+self.budget.search_seconds
        report = {"branches": [], "success": False, "state": "q,u=t/T",
                  "budget_seconds": self.budget.search_seconds, "clock_changed": False}
        lower = self.trajopt.transition_model.action_bound_lows.detach().cpu().numpy().reshape(-1)
        upper = self.trajopt.transition_model.action_bound_highs.detach().cpu().numpy().reshape(-1)
        tolerance = _projection_tolerance(self.problem)
        map_u, map_s = self.device_cfg.to_device(self.progress_map.u), self.device_cfg.to_device(self.progress_map.s)
        anchor_pairs = self.progress_map.anchor_parameters(self.anchors)
        query_u = anchor_pairs[:, 0]
        anchor_progress = dict(anchor_pairs)
        if self.finger_trajectory is not None:
            query_u = np.unique(np.r_[query_u,
                [event["applied_time_s"]/self.formal_clock["duration"] for event in self.finger_trajectory.events]])
        # The scene evaluator checks adjacent modes at this same q/u. Moving
        # float64 times by one ULP would vanish in the native float32 inputs.
        query_u = np.unique(query_u)
        start = np.r_[self.start.position[0].detach().cpu().numpy(), 0.]
        # Ordinary source phases and reference sample density never create
        # search boundaries. Plateaus retain both physical-time endpoints.
        boundary_times = [0., 1.]
        for name in ("pose_relations", "mechanisms", "interiors"):
            for requirement in getattr(self.problem, name):
                if requirement.start >= self.problem.start_progress:
                    boundary_times.append(self.progress_map.anchor_parameters([requirement.start])[0, 0])
                if requirement.end >= self.problem.start_progress:
                    boundary_times.append(self.progress_map.anchor_parameters([requirement.end])[-1, 0])
        goals_by_time = {}
        for name in ("pose_events", "coordinate_events", "supports"):
            for index, requirement in enumerate(getattr(self.problem, name)):
                parameter = self.progress_map.anchor_parameters([requirement.progress])[0, 0]
                boundary_times.append(parameter)
                goals_by_time.setdefault(float(parameter), []).append((name, index))
        scene_anchors = [value for value in self.scene.required_progress_anchors
                         if value >= self.problem.start_progress]
        if scene_anchors:
            boundary_times.extend(self.progress_map.anchor_parameters(scene_anchors)[:, 0])
        boundaries = np.unique(boundary_times)
        report["chart_times"] = boundaries.tolist()
        work = {"state_queries": 0}
        rng = np.random.default_rng(self.budget.random_seed)
        source_controls = initial_seed[0].detach().cpu().numpy()
        invalid_states = []

        def goals_at_time(parameter):
            return tuple(goals_by_time.get(float(parameter), ()))

        def state_parts(values):
            count = int(np.prod(values.shape[:-1]))
            if (time.monotonic() >= deadline
                    or work["state_queries"]+count > min(self.budget.max_validation_points, _NATIVE_STAGE_POINT_LIMIT)):
                raise TimeoutError("The shared OMPL time/state-query budget was exhausted")
            work["state_queries"] += count
            joints = values[..., :-1].contiguous()
            zero = torch.zeros_like(joints)
            js = self.JointState(position=joints, velocity=zero, acceleration=zero, jerk=zero,
                dt=self.device_cfg.to_device([self.budget.interpolation_dt]), joint_names=list(self.joint_names))
            state = self._verification_fk.compute_kinematics(js)
            tools = state.tool_poses
            index = tools.tool_frames.index(self.problem.tool_link)
            quaternion = self._orientation_vjp.apply(joints, tools.quaternion.detach(),
                state.tool_jacobians[..., 3:, :].detach())
            position = self._native_layout.apply(tools.position)
            p, r = self.problem.transform_tool_pose(position[:, :, index], quaternion[:, :, index])
            return js, state, p, r, _torch_interpolate(map_u, map_s, values[..., -1])

        def equations(scene, active, goals=()):
            cache = {}
            def residual_function(values):
                js, _, p, r, progress = state_parts(values.reshape(1, 1, -1))
                process = self.problem.project_residual(p, r, progress,
                    joint_position=js.position, joint_names=self.joint_names, active_requirements=active)
                drive = scene.project_residual(progress, js.position, self.joint_names)
                goal = self.problem.goal_residual(p, r, progress, goal_ids=goals)
                return torch.cat((process, drive, goal), -1).reshape(-1)

            def evaluate(values, derivative):
                # Native setup, Newton projection and geodesic export often
                # repeat an identical query. Cache only immutable CPU numeric
                # answers for this exact chart/scene; no tensor graph or
                # result is shared with a different candidate branch.
                key = (np.asarray(values, dtype=float).tobytes(), derivative)
                if key in cache:
                    return cache[key]
                values_t = self.device_cfg.to_device(np.asarray(values))
                if derivative:
                    with torch.enable_grad():
                        if residual_function(values_t).numel() == 0:
                            return np.empty((0, len(values)))
                        answer = torch.autograd.functional.jacobian(residual_function, values_t,
                            create_graph=False, vectorize=False)
                else:
                    with torch.no_grad():
                        answer = residual_function(values_t)
                result = answer.detach().cpu().numpy().astype(float)
                if len(cache) < min(self.budget.max_validation_points, _NATIVE_STAGE_POINT_LIMIT):
                    cache[key] = result
                return result
            return evaluate

        def valid(values, scene):
            with torch.no_grad():
                values_t = self.device_cfg.to_device(values[None, None])
                js, state, p, r, progress = state_parts(values_t)
                clock_time = values_t[..., -1]*self.formal_clock["duration"]
                terms = {"process": self.problem.evaluate(p, r, progress, js.position,
                    self.joint_names, joint_velocity=js.velocity,
                    point_goal_ids=goals_at_time(float(values[-1])), direction_mode="point"),
                    "scene": _scene_boundary_evaluate(scene, self.problem, self.progress_map,
                        self.problem.transform_robot_spheres(state.robot_spheres), p, r, progress,
                        js.position, self.joint_names, trajectory_time=clock_time,
                        normalized_time=values_t[..., -1], source_geometry=True)}
                # BEGIN OMPL shares the metrics source-solid self-collision contract.
                native = self._verification_native.compute_costs(
                    self._RobotState(joint_state=js, cuda_robot_model_state=state))
                terms.update({f"native:{name}": value for name, value in zip(native.names, native.values)})
                terms["source_self_collision"] = scene.check_robot_self_collision(
                    js.position, self.joint_names, clock_time)
                # END all other native, task and scene constraints remain hard.
                coverage = scene.check_robot_coverage(js.position, self.joint_names, clock_time,
                    entrance=bool(values[-1] == 0 and np.array_equal(values, start)))
                if coverage:
                    terms["robot_query_incomplete"] = progress.new_tensor([[float(any(
                        not row["attempt_allowed"] for row in coverage))]])
                # BEGIN all existing tensors precede one exact boolean host read.
                accepted = _ompl_terms_accepted(terms)
                # END rejected-state diagnostics retain their original order/values.
                if not accepted and len(invalid_states) < 3:
                    invalid_states.append({"u": float(values[-1]), "progress": float(progress.item()),
                        "terms": {name: float(term.max().item()) for name, term in terms.items()}})
                return accepted

        # BEGIN a two-attempt boundary budget includes a non-reference posture.
        def proposals(begin, parameter):
            # BEGIN a retained alternative posture can escape a failed trajectory basin.
            entry = self.goal_ik_candidates.get(float(parameter))
            if entry and entry["candidates"]:
                # The primary proposal already shaped the trajectory seed.
                # Preserve one random attempt in the existing two-attempt budget.
                row = entry["candidates"][min(1, len(entry["candidates"])-1)]
                yield "ranked_functional_ik", np.r_[row["joint_position"], parameter]
            else:
                yield "candidate_curve", np.r_[candidate_trajectory._curve(parameter*candidate_trajectory.duration), parameter]
            # END actual prefix release geometry and allowed-set projection recheck all proposals.
            # Reference hints must not consume every small attempt budget.
            # This proposal still uses the same allowed-set projection and
            # complete state/edge validity checks as every other member.
            yield "joint_box_random", np.r_[rng.uniform(lower, upper), parameter]
            # BEGIN proposal states come from the represented cubic, not coefficients.
            for controls in source_controls:
                yield "reference_curve", np.r_[self.physical_domain.evaluate(
                    controls, np.asarray([parameter]))[0], parameter]
            # END the original scene/projection checks still qualify each proposal.
            yield "previous_boundary", np.r_[begin[:-1], parameter]
            # Sampling the actual joint box permits different support tangents,
            # yaw and redundant IK branches. There is no reference corridor.
            while True:
                yield "joint_box_random", np.r_[rng.uniform(lower, upper), parameter]
        # END proposal ordering changes neither attempts nor time/query limits.

        # An explicit stack permits bounded backtracking without recursive
        # depth or copying all prior path samples into every branch frame.
        scene = self.scene.clone_for_rollout()
        frames = [{"start": start, "scene": scene, "proposals": proposals(start, boundaries[1]),
                   "attempts": 0, "path": None}]
        try:
            while frames and time.monotonic() < deadline:
                stage = len(frames)-1
                frame = frames[-1]
                lo, hi = boundaries[stage:stage+2]
                # Every visit receives finite alternative members; exhausting
                # them returns to the preceding allowed set, not a new clock.
                if frame["attempts"] >= self.budget.ik_seeds:
                    frames.pop()
                    continue
                frame["attempts"] += 1
                begin, scene = frame["start"], frame["scene"]
                # BEGIN retain the initializer identity through projection.
                proposal_source, goal = next(frame["proposals"])
                # END the source labels a proposal, not a validity certificate.
                goal_s = float(anchor_progress.get(hi, np.interp(hi, self.progress_map.u, self.progress_map.s)))
                middle_s = float(np.interp((lo+hi)/2, self.progress_map.u, self.progress_map.s))
                active = self.problem.projection_requirements(middle_s)
                boundary_active = self.problem.projection_requirements(goal_s)
                goal_ids = goals_at_time(float(hi))
                boundary_equations = equations(scene, boundary_active)
                process_count = len(boundary_equations(goal, False))
                goal = _project_allowed_boundary(goal, equations(scene, boundary_active, goal_ids),
                    process_count, tolerance, np.r_[lower, lo], np.r_[upper, hi], deadline=deadline)
                # BEGIN rejected projections retain their proposal source too.
                evidence = {"chart": stage, "time_interval": [float(lo), float(hi)],
                            "goals": list(goal_ids), "attempt": frame["attempts"],
                            "proposal_source": proposal_source}
                # END provenance does not alter projection or validity results.
                if goal is None:
                    report["branches"].append({**evidence, "failure": "allowed_boundary_projection_failed"})
                    continue
                evidence["boundary_state"] = goal.tolist()
                constraint, rank = _independent_projection(equations(scene, active), goal)
                if constraint is None:
                    report["branches"].append({**evidence, "failure": "singular_active_relation_chart"})
                    continue
                invalid_states.clear()
                # Reserve work for alternate boundary members and later
                # charts. All chart construction and queries share deadline.
                remaining_stages = len(boundaries)-1-stage
                seconds = max(0., deadline-time.monotonic())/(remaining_stages+1)
                validity_cache = {}
                def stage_valid(value):
                    key = np.asarray(value, dtype=float).tobytes()
                    if key not in validity_cache:
                        validity_cache[key] = valid(value, scene)
                    return validity_cache[key]
                path, stage_report = _ompl_stage(begin, goal, np.r_[lower, lo], np.r_[upper, hi],
                    stage_valid, constraint, tolerance=tolerance,
                    step=self.budget.validation_joint_step, seconds=seconds,
                    sample_limit=min(self.budget.max_validation_points, _NATIVE_STAGE_POINT_LIMIT)-work["state_queries"],
                    query_parameters=query_u[(query_u >= lo) & (query_u <= hi)])
                evidence.update(stage_report, independent_equalities=rank,
                                first_invalid_states=deepcopy(invalid_states))
                report["branches"].append(evidence)
                if path is None:
                    continue
                path = path[np.r_[True, np.any(path[1:] != path[:-1], axis=1)]]
                if np.any(np.diff(path[:, -1]) <= 0):
                    evidence["failure"] = "nonmonotone_or_instantaneous_joint_motion"
                    continue
                frame["path"] = path
                if sum(len(item["path"])-1 for item in frames)+1 > min(self.budget.max_validation_points, _NATIVE_STAGE_POINT_LIMIT):
                    evidence["failure"] = "whole_path_sample_budget"
                    continue
                if stage+2 < len(boundaries):
                    # This exact endpoint is the selected prefix's release
                    # FK. It is never taken from a demonstration or failed
                    # whole-curve witness, and sibling scenes remain private.
                    with torch.no_grad():
                        _, _, p, r, progress = state_parts(self.device_cfg.to_device(goal[None, None]))
                        child_scene = scene.with_release_history(p, r, progress)
                    frames.append({"start": goal, "scene": child_scene,
                        "proposals": proposals(goal, boundaries[stage+2]), "attempts": 0, "path": None})
                    continue
                whole_path = np.concatenate((frames[0]["path"], *(item["path"][1:] for item in frames[1:])))
                # Local angular goal screening cannot prove winding. Recheck
                # the ordered full path with its actual start and all goal
                # queries before using it as a complete spatial initializer.
                with torch.no_grad():
                    positions, rotations, progress_values = [], [], []
                    for chunk in np.array_split(whole_path, max(1, int(np.ceil(len(whole_path)/32)))):
                        _, _, p, r, progress = state_parts(self.device_cfg.to_device(chunk[None]))
                        positions.append(p.clone()); rotations.append(r.clone()); progress_values.append(progress.clone())
                    full_progress = torch.cat(progress_values, 1)
                    q = self.device_cfg.to_device(whole_path[None, :, :-1])
                    process = self.problem.evaluate(torch.cat(positions, 1), torch.cat(rotations, 1),
                        full_progress, q, self.joint_names, joint_velocity=torch.zeros_like(q),
                        trajectory_time=self.device_cfg.to_device(whole_path[None, :, -1])*self.formal_clock["duration"],
                        direction_mode="path")
                    if (not bool(torch.isfinite(process).all()) or not bool((process <= 0).all())
                            or self.problem.verification_issues(full_progress)):
                        evidence["failure"] = "complete_path_process_or_history_rejected"
                        continue
                # BEGIN OMPL point samples also require the native representation map.
                controls, fit = self.physical_domain.fit_reference(whole_path[:, -1],
                    self.device_cfg.to_device(whole_path[:, :-1]), prior_controls=initial_seed[0, 0],
                    velocity_bounds=self._seed_velocity_bounds())
                if controls is None:
                    evidence.update(failure="native_cubic_fit_failed", fit=fit)
                    continue
                report.update(success=True, samples=len(whole_path), state_queries=work["state_queries"],
                              cubic_fit=fit)
                return controls[None, None].contiguous(), report
                # END fitted curves still need full trajectory optimization/verification.
        except TimeoutError:
            report["budget_exhausted"] = True
        report["state_queries"] = work["state_queries"]
        report["failure"] = "no_complete_time_state_seed_within_budget"
        return None, report
    # END OMPL has no schedule replacement return value or side effect.

    # BEGIN bounded true-goal proposals retain static evidence before continuity.
    def _functional_goal(self, target):
        """Convert one explicitly declared functional pose into native link coordinates.

        The centre is an IK proposal only. All other goals and the original
        allowed tolerances are checked separately; no reference sample becomes
        a hard goal and no full pose is invented for a coordinate/support goal.
        """
        from scipy.spatial.transform import Rotation
        pose = target.target @ np.linalg.inv(target.tool_from_function)
        position = self.device_cfg.to_device(pose[:3, 3][None, None])
        quaternion = self.device_cfg.to_device(
            Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]][None, None])
        position, quaternion = self.problem.reference_to_link_pose(position, quaternion)
        return self.GoalToolPose(tool_frames=[self.problem.tool_link],
            position=position[:, :, None, None].contiguous(),
            quaternion=quaternion[:, :, None, None].contiguous())

    def _rank_ik_candidates(self, candidates, success, previous, u, goal_ids=()):
        """Screen finite CPU joint candidates at one physical time, then rank.

        Returned indices preserve up to the existing IK seed count, independently
        of the TrajOpt seed count. Reports describe static point constraints,
        known source solids and the existing attempt-allowed coverage policy.
        Zero velocities merely satisfy the point API: stop/rate/trajectory
        history requirements are not certified. The caller must charge every
        finite state to its original cumulative validation budget before entry.
        """
        from .planning_ik import rank_ik_candidates
        values = np.asarray(candidates)
        flags = np.asarray(success, bool)
        prior = np.asarray(previous)
        if (values.ndim != 2 or values.shape[1] != len(self.joint_names)
                or len(values) > self.budget.ik_seeds or flags.shape != (len(values),)
                or not np.isfinite(u) or not 0 <= u <= 1):
            raise ValueError("IK static screening must fit the declared joint/seed/clock bounds")
        goal_ids = tuple(goal_ids)
        phase = (float(getattr(self.problem, goal_ids[0][0])[goal_ids[0][1]].progress)
                 if goal_ids else float(np.interp(u, self.progress_map.u, self.progress_map.s)))
        if any(getattr(self.problem, name)[index].progress != phase for name, index in goal_ids):
            raise ValueError("One IK query cannot combine goals at different physical arrivals")
        lower = self.trajopt.transition_model.action_bound_lows.detach().cpu().numpy().reshape(-1)
        upper = self.trajopt.transition_model.action_bound_highs.detach().cpu().numpy().reshape(-1)
        reports = []
        for index, row in enumerate(values):
            report = {"native_pose_ik_success": bool(flags[index]), "candidate_source": "native_ik",
                "task_satisfied": False,
                "source_valid": False, "environment_clearance_m": None,
                "self_clearance_m": None, "joint_limit_margin_fraction": None,
                "progress": phase, "u": float(u), "time_s": float(u*self.formal_clock["duration"]),
                "goal_ids": [list(value) for value in goal_ids],
                "scope": "Static point requirements and modeled source geometry; initializer only",
                "not_checked": ["joint_velocity", "acceleration", "jerk", "torque",
                                "stop_dynamics", "winding_and_direction_history", "complete_trajectory"]}
            reports.append(report)
            if not np.isfinite(row).all():
                report["failure"] = "nonfinite_ik_candidate"
                continue
            # Each candidate receives a private scene. In particular, an absent
            # earlier release FK stays unknown/release_anchor; no demonstration
            # or another candidate is allowed to provide that release state.
            scene = self._verification_scene.clone_for_rollout()
            with self.torch.no_grad():
                q = self.device_cfg.to_device(row[None, None]).contiguous()
                zero = self.torch.zeros_like(q)
                state = self._verification_fk.compute_kinematics(self.JointState(
                    position=q, velocity=zero, acceleration=zero, jerk=zero,
                    dt=self.device_cfg.to_device([self.budget.interpolation_dt]),
                    joint_names=list(self.joint_names)))
                tools = state.tool_poses
                tool_index = tools.tool_frames.index(self.problem.tool_link)
                position, quaternion = self.problem.transform_tool_pose(
                    tools.position[:, :, tool_index], tools.quaternion[:, :, tool_index])
                progress = q.new_full(q.shape[:2], phase)
                parameter = q.new_full(q.shape[:2], float(u))
                times = q.new_full(q.shape[:2], report["time_s"])
                task = self.problem.residuals(position, quaternion, progress, q, self.joint_names,
                    joint_velocity=zero, trajectory_time=times, point_goal_ids=goal_ids,
                    direction_mode="point")
                # Static IK has no actual velocity. The explicit stop still
                # belongs to complete TrajOpt and dense validation, unchanged.
                task = {name: value for name, value in task.items() if not name.startswith("stop:")}
                world, components, own = {}, {}, {}
                _scene_boundary_evaluate(scene, self.problem, self.progress_map,
                    self.problem.transform_robot_spheres(state.robot_spheres), position, quaternion,
                    progress, q, self.joint_names, trajectory_time=times,
                    normalized_time=parameter, source_geometry=True, diagnostics=world,
                    source_components_out=components)
                self_residual = scene.check_robot_self_collision(q, self.joint_names, times, diagnostics=own)
                coverage = scene.check_robot_coverage(q, self.joint_names, times)
            task_values = {name: float(value.max().item()) for name, value in task.items()}
            source_values = {name: float(value.max().item()) for name, value in components.items()
                             if not name.startswith("unknown/") or name.startswith("unknown/release_anchor/")}
            source_values["source_self_collision"] = float(self_residual.max().item())
            source_values["joint_position_bounds"] = float(np.maximum(lower-row, 0.).sum()
                                                           +np.maximum(row-upper, 0.).sum())
            task_ok = all(np.isfinite(value) and value <= 0 for value in task_values.values())
            coverage_ok = all(item["attempt_allowed"] for item in coverage)
            source_ok = coverage_ok and all(np.isfinite(value) and value <= 0 for value in source_values.values())
            clearance = world["unpermitted_clearance"]
            # complete_query concerns this declared source query; distance_complete
            # additionally describes global geometry coverage. Observed TSDF
            # distances remain explicitly scoped measurements, never a proof.
            incomplete_subset = bool(set(clearance["limitations"]) & {
                "finite_patch_unpermitted_subset_distance_unmeasured",
                "surface_permission_not_resolved_by_finite_patch"})
            environment = (clearance["minimum_recorded_distance_m"]
                           if source_ok and clearance["complete_query"] and not incomplete_subset else None)
            self_clearance = own["worst_state"]["minimum_required_clearance_m"]
            widths = upper-lower
            positive_width = widths > 0
            margin = (float((np.minimum(row-lower, upper-row)[positive_width]
                              /widths[positive_width]).min()) if positive_width.any() else None)
            report.update(task_satisfied=task_ok, source_valid=source_ok,
                environment_clearance_m=environment, self_clearance_m=self_clearance,
                joint_limit_margin_fraction=margin,
                task_residuals={name: value if np.isfinite(value) else None for name, value in task_values.items()},
                source_residuals={name: value if np.isfinite(value) else None for name, value in source_values.items()},
                source_clearance=clearance, robot_coverage=coverage,
                unmodelled_geometry_names=sorted(set(world["unmodelled_geometry_names"])
                                                |set(own["unmodelled_geometry_names"])),
                full_robot_coverage_certified=bool(coverage) and all(item["passed"] for item in coverage))
        # BEGIN nominal reference success and actual goal satisfaction stay distinct.
        selected = rank_ik_candidates(values, reports, prior, count=self.budget.ik_seeds,
            joint_tolerance=self.problem.numerical.joint,
            query_kind="functional" if goal_ids else "reference")
        # END a functional allowed set never acquires an extra centre-pose equality.
        return selected, reports

    def _build_goal_ik_candidates(self, reference_seeds):
        """Retain bounded CPU proposals at actual full-pose functional events.

        One existing-size IK batch is queried per distinct event time. When
        several full-pose events share that time, the first supplies a nominal
        IK centre and all same-time goals remain part of static screening. This
        is a finite proposal set, not a complete search of their intersection.
        Coordinate/support goals never acquire an invented full TCP target.
        Native calls remain on the same tracked solver and frozen clock.
        """
        self.goal_ik_candidates = {}
        controls = reference_seeds[0].detach().cpu().numpy()
        targets = sorted(enumerate(self.problem.pose_events), key=lambda item: (item[1].progress, item[0]))
        for _, target in targets:
            if target.progress < self.problem.start_progress:
                continue
            u = float(self.progress_map.anchor_parameters([target.progress])[0, 0])
            if u in self.goal_ik_candidates:
                continue
            # BEGIN native IK warm starts use actual curve states at the event time.
            previous = self.physical_domain.evaluate(controls, np.asarray([u]))[:, 0]
            # END interpolation of free coefficients would omit the fixed boundaries.
            self.seed_parameter.fill_(u)
            result = self.ik.solve_pose(self._functional_goal(target), current_state=self.start,
                seed_config=self.device_cfg.to_device(previous[None]).contiguous(),
                return_seeds=self.budget.ik_seeds)
            candidates = self.ik.get_active_js(result.js_solution).position[0].detach().cpu().numpy()
            success = result.success[0].detach().cpu().numpy()
            # One physical event query cannot borrow a merely nearby event.
            # The general helper's numerical neighbor band is for projection;
            # this pool retains exact declared occurrence identity.
            goal_ids = tuple((name, index) for name, index in
                             self.problem.goal_requirements_at(target.progress)
                             if getattr(self.problem, name)[index].progress == target.progress)
            selected, reports = self._rank_ik_candidates(candidates, success, previous, u, goal_ids)
            self.goal_ik_candidates[u] = {"progress": float(target.progress), "u": u,
                "time_s": u*self.formal_clock["duration"], "goal_ids": [list(value) for value in goal_ids],
                "candidates": [{"joint_position": candidates[index].tolist(), "report": reports[index]}
                               for index in selected],
                "scope": "Finite full-pose IK proposals; every OMPL prefix and final trajectory must recheck them"}
        return self.goal_ik_candidates
    # END static candidates cannot manufacture release history or trajectory feasibility.

    # BEGIN independently qualify bounded seed and exploration observations.
    def _functional_candidate_mask(self, actions):
        """Screen exact pose-event anchors; this is never a feasibility verdict.

        Use the native sampled cubic and the same Hermite anchor evaluation as
        ProcessCostManager. Source solids, all other process requirements and
        continuous derivative bounds remain mandatory in metrics/export.
        """
        goals = [goal for goal in self.problem.pose_events
                 if goal.progress >= self.problem.start_progress]
        count = actions.shape[0]
        if not goals:
            return np.ones(count, dtype=bool)
        self._charge_candidate_queries(count*(self.trajopt.horizon+len(goals)))
        with self.torch.no_grad():
            state = self.trajopt.metrics_rollout.compute_state_from_action(actions)
            joint = state.joint_state
            # BEGIN keep each point goal paired with its own first arrival.
            # A combined anchor query sorts and deduplicates goals and expands
            # pauses into two endpoints. Its columns cannot be indexed in the
            # original goal order. Like _first_event_query, a pose event owns
            # only its earliest arrival, not a maintained pose over a pause.
            # One query per goal also matches the FK work charged above.
            parameters = np.asarray([self.progress_map.anchor_parameters([goal.progress])[0, 0]
                                     for goal in goals])
            # END repeated progress values still retain every separate goal.
            u = self.device_cfg.to_device(parameters)[None].expand(count, -1)
            counts = self.torch.full((count,), joint.position.shape[1], device=actions.device,
                                     dtype=self.torch.long)
            q, _ = _hermite_at_parameters(joint.position, joint.velocity, joint.dt, counts, u)
            kinematics = self._verification_fk.compute_kinematics(self.JointState.from_position(
                q.contiguous(), joint_names=list(self.joint_names)))
            tools = kinematics.tool_poses
            index = tools.tool_frames.index(self.problem.tool_link)
            position = self._native_layout.apply(tools.position)[:, :, index]
            quaternion = tools.quaternion[:, :, index]
            position, quaternion = self.problem.transform_tool_pose(position, quaternion)
            eligible = self.torch.ones(count, dtype=self.torch.bool, device=actions.device)
            for column, goal in enumerate(goals):
                p, r = goal.errors(position[:, column], quaternion[:, column])
                eligible &= (self.torch.isfinite(p) & self.torch.isfinite(r)
                             & (p <= goal.position_tolerance_m) & (r <= goal.rotation_tolerance_rad))
        return eligible.detach().cpu().numpy()

    def _observed_native_result(self, actions):
        """Reuse native postprocessing on its unchanged complete seed layout."""
        count = actions.shape[0]
        if self.trajopt.solve_state.batch_size != 1 or self.trajopt.solve_state.num_seeds != count:
            raise ValueError("Observed actions changed the active native seed layout")
        # Extra source metrics are real validation work, even when no candidate
        # survives to export. The native oracle also retains its own work limit.
        self._charge_candidate_queries(count*(self.trajopt.horizon+self.formal_clock["valid_count"]
                                              +2*(len(self.anchors)+len(self.refinement_parameters))))
        with self.torch.no_grad():
            # BEGIN observations have the same bounded candidate lifecycle as solve.
            completed = False
            try:
                metrics = self.trajopt.metrics_rollout.compute_metrics_from_action(actions)
                joint = metrics.state.joint_state
                joint.joint_names = self.trajopt.joint_names
                joint.knot = metrics.actions
                joint.knot_dt = joint.dt*self.trajopt.interpolation_steps
                interpolated, trajectory, last = self.trajopt._interpolate_and_compute_metrics(joint)
                result = self.trajopt._get_result(actions, metrics, interpolated, trajectory, last, count, count)
                result = self.trajopt._get_best_result(result, count)
                completed = True
                return result
            finally:
                # Standalone diagnostic callers have no enclosing solve finalizer.
                if not completed or not getattr(self.trajopt, "_source_world_solve_active", False):
                    self.trajopt.end_source_world_reuse()
            # END normal dense completion keeps only the two bounded batch slots.

    def _observe_actions(self, actions, origin, initial, seen):
        """Validate new block states without replacing the optimizer's state."""
        controls = actions.detach().cpu().numpy()
        fresh = np.asarray([not any(np.array_equal(value, old) for old in seen[index])
                            for index, value in enumerate(controls)])
        if not fresh.any():
            return
        for index in np.flatnonzero(fresh):
            seen[index].append(controls[index].copy())
        eligible = self._functional_candidate_mask(actions) & fresh
        summary = {**deepcopy(origin), "eligible_seed_indices": np.flatnonzero(eligible).tolist(),
                   "accepted": False, "selection_policy": "first_complete_feasible_candidate",
                   "scope": "pose-event screen followed by complete native/export validation"}
        self.candidate_observations.append(summary)
        if not eligible.any():
            return
        self._check_validation_work("validation_passes")
        result = self._observed_native_result(actions)
        previous = self.last_optimizer_diagnostics
        try:
            self._observing_candidate = True
            self.last_optimizer_diagnostics = self._optimizer_reports(result.solution, initial)
            _, trajectory, attempt = self.select_candidate(result,
                candidate_indices=np.flatnonzero(eligible), first_feasible=True, origin=origin)
            summary.update(accepted=attempt["accepted"], residuals=deepcopy(attempt["residuals"]),
                           selected_seed_index=attempt["selected_seed_index"],
                           validation_points=attempt["validation_points"])
            if attempt["accepted"]:
                # Capture this validation before any later native metrics or
                # export can replace the planner's mutable diagnostic pointer.
                self._candidate_qualified(trajectory, self.last_validation, attempt)
        finally:
            self._observing_candidate = False
            self.last_optimizer_diagnostics = previous

    def _optimizer_reports(self, solution, initial):
        core_reports = [optimizer._core.solve_diagnostics for optimizer in self.trajopt.core.optimizers]
        reports = []
        for index in range(solution.shape[1]):
            displacement = (solution[0, index]-initial[0, index]).abs()
            reports.append({"seed_index": index,
                "unchanged_from_seed": bool((displacement == 0).all().item()),
                "maximum_control_update": float(displacement.max().item()),
                "cores": [rows[index] for rows in core_reports]})
        return reports
    # END functional screening cannot itself authorize execution.

    # BEGIN the selected curve alone advances the functional AL continuation.
    def update_functional_constraints(self, result):
        """Measure signed event inequalities on the original regular cubic.

        Use the same float32 Hermite anchors, complete ordered history and
        functional-frame FK as the optimizer. This inexpensive constraint
        measurement is neither source-geometry validation nor acceptance.
        Updating the objective requires rebuilding the solver before reuse.
        """
        if result.batch_size != 1 or result.num_seeds != 1:
            raise ValueError("Functional multipliers require one selected continuation curve")
        if not self.functional_constraints.keys:
            return {"changed": False, "reason": "no_functional_point_constraints"}
        torch = self.torch
        native = self.trajopt.get_active_js(result.js_solution)
        pairs = self.progress_map.anchor_parameters(self.anchors)
        # BEGIN preserve the optimizer's complete coordinate-unwrapping history.
        # Extra working-set samples can change adjacent angular increments;
        # omitting them would update duals from a different coordinate event.
        if len(self.refinement_parameters):
            additional = self.refinement_parameters
            pairs = np.unique(np.concatenate((pairs, np.column_stack((additional,
                np.interp(additional, self.progress_map.u, self.progress_map.s))))), axis=0)
        # END goal anchors and active query samples match ProcessCostManager.
        horizon = native.position.shape[-2]
        self._charge_candidate_queries(horizon+len(pairs))
        with torch.no_grad():
            q = native.position.reshape(1, horizon, -1)
            dq = native.velocity.reshape_as(q)
            counts = torch.full((1,), horizon, device=q.device, dtype=torch.long)
            u = torch.arange(horizon, device=q.device, dtype=q.dtype)[None]/(horizon-1)
            s = _torch_interpolate(self.device_cfg.to_device(self.progress_map.u),
                                   self.device_cfg.to_device(self.progress_map.s), u)
            anchor_u = self.device_cfg.to_device(pairs[:, 0])[None]
            anchor_s = self.device_cfg.to_device(pairs[:, 1])[None]
            anchor_q, _ = _hermite_at_parameters(q, dq, native.dt, counts, anchor_u)
            all_u = torch.cat((u, anchor_u), 1)
            order = torch.argsort(all_u, dim=1, stable=True)
            q = torch.cat((q, anchor_q), 1).gather(1, order[..., None].expand(-1, -1, q.shape[-1]))
            progress = torch.cat((s, anchor_s), 1).gather(1, order)
            times = all_u.gather(1, order)*self.formal_clock["duration"]
            state = self._verification_fk.compute_kinematics(self.JointState.from_position(
                q.contiguous(), joint_names=list(self.joint_names)))
            index = state.tool_poses.tool_frames.index(self.problem.tool_link)
            position, quaternion = self.problem.transform_tool_pose(
                state.tool_poses.position[:, :, index], state.tool_poses.quaternion[:, :, index])
            signed, _ = self.problem.functional_inequalities(
                position, quaternion, progress, trajectory_time=times)
            measurements = self.problem.measurements(position, quaternion, progress, trajectory_time=times)
            measurements = {name: float(value.amax().item()) for name, value in measurements.items()
                            if name.startswith(("function:", "coordinate:"))}
        report = self.functional_constraints.update(signed)
        report["measurements"] = measurements
        report["measurement_scope"] = "selected regular cubic with exact first-arrival anchors"
        return report

    def reset_functional_constraints(self):
        """An independent spatial initializer starts a new multiplier sequence."""
        self.functional_constraints.reset()
    # END solver rebuilds reset curvature and scalar-best records for the new objective.

    def optimize(self, seed):
        self.physical_domain.validate(seed)
        # BEGIN functional-frame goals have one owner and their original tolerances.
        # Native link tracking is a second soft centre objective, including
        # inside the allowed set. ProcessCostManager owns all trajectory goals;
        # the native API only needs an inert H=1 pose for its storage layout.
        self.trajopt.disable_tool_pose_tracking()
        goal = self._goal(self.problem.phase_count)
        # END IK retains its own native pose tracking for proposal generation.
        # BEGIN result provenance uses the full unsorted native seed layout.
        # Returning every seed makes native get_topk_seeds return the original
        # layout; per-seed history and input/output displacement remain aligned.
        initial = seed.detach().clone()
        # BEGIN solve-local observers have no references in captured closures.
        cores = [optimizer._core for optimizer in self.trajopt.core.optimizers]
        observing = callable(getattr(self, "_candidate_qualified", None))
        seen = [[] for _ in range(seed.shape[1])]
        # Keep only small verdicts. Complete validation evidence is serialized
        # once into the bound archive, never retained for every explored curve.
        self._observed_verdicts = []
        if observing:
            for core in cores:
                core.candidate_observer = lambda actions, origin: self._observe_actions(actions, origin, initial, seen)
        try:
            result = self.trajopt.solve_pose(
                goal_tool_poses=goal, current_state=self.start,
                seed_traj=seed, num_seeds=seed.shape[1], return_seeds=seed.shape[1],
                dt=self.torch.full((1, seed.shape[1]), self.budget.maximum_trajectory_dt,
                                   dtype=seed.dtype, device=seed.device),
                use_implicit_goal=False, goal_state=None,
                finetune_attempts=0, finetune_dt_scale=1.)
        finally:
            for core in cores:
                core.candidate_observer = None
        # END partial solves cannot retain stale callbacks into another request.
        if result.solution.shape != initial.shape:
            raise ValueError("Native trajectory solving changed the admitted seed layout")
        self.last_optimizer_diagnostics = self._optimizer_reports(result.solution, initial)
        return result
        # END unchanged output is separate from movement of native exploration states.
    # END intermediate and terminal goals share the same typed requirement owner.

    def native_diagnostics(self, result, seed_index=None):
        """Expose the native reason for rejection, including dense metrics.

        Native pose tracking is disabled for trajectories. Typed skill and
        source constraints supply the complete discrete verdict. Every native
        accepted curve still receives the independent :meth:`export` checks.
        """
        success = result.success if seed_index is None else result.success[:, seed_index]
        report = {"success": bool(success.all().item())}

        def maximum(value):
            # Native rollout batch is [batch*seeds,H,...]. This planner's
            # batch is explicitly one, so this is an exact seed slice.
            if seed_index is not None:
                if result.batch_size != 1 or value.shape[0] != result.num_seeds:
                    raise ValueError("Native metrics do not match the single-batch seed layout")
                value = value[seed_index]
            return float(value.max().item())

        for label, metrics in (("rollout", result.metrics),
                               ("interpolated", result.interpolated_metrics)):
            if metrics is None:
                continue
            constraints = metrics.costs_and_constraints.constraints
            report[label] = {
                "constraints": {name: maximum(value)
                                for name, value in zip(constraints.names, constraints.values)},
                "convergence": {name: maximum(value)
                                for name, value in zip(metrics.convergence.names, metrics.convergence.values)},
            }
            # BEGIN preserve native violation columns and match pre-fold scene evidence.
            # The last cost column can contain extra anchors. Its index alone
            # is not a physical failure time; source_trace keeps that identity.
            if seed_index is not None:
                try:
                    samples = []
                    for name, value in zip(constraints.names, constraints.values):
                        array = value[seed_index].detach().cpu().numpy()
                        for index in np.argwhere((array > 0) | ~np.isfinite(array)):
                            row = {"constraint": name, "index": index.tolist(),
                                   "value": float(array[tuple(index)])}
                            # BEGIN only non-folded native columns identify a time directly.
                            count = (self.trajopt.horizon if label == "rollout"
                                     else self.formal_clock["valid_count"])
                            if name == "cspace" and int(index[0]) < count:
                                row["normalized_time"] = float(np.float32(index[0])/np.float32(count-1))
                            # END scene/process/self times come from their original trace.
                            samples.append(row)
                    report[label]["violating_constraint_samples"] = samples
                    from .planning_diagnostics import bind_trace_to_native_seed
                    rollout = (self.trajopt.metrics_rollout if label == "rollout" else
                               self.trajopt.additional_metrics_rollouts["interpolated_rollout"])
                    state = result.js_solution if label == "rollout" else result.interpolated_trajectory
                    report[label]["source_trace"] = bind_trace_to_native_seed(
                        getattr(rollout.metrics_constraint_manager, "last_source_diagnostics", None),
                        state, seed_index, self.joint_names)
                    # BEGIN distinguish a replayed report from new source work.
                    source_scene = getattr(rollout.metrics_constraint_manager, "_scene", None)
                    execution = getattr(source_scene, "_last_source_world_execution", None)
                    if execution is not None:
                        report[label]["source_query_execution"] = deepcopy(execution)
                    # END geometry evidence itself keeps its original exact fields.
                except Exception as error:
                    report[label]["source_trace"] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
            # END a trace failure never changes native acceptance or selection.
        return report

    # BEGIN feasibility precedes quality and rejected native curves never execute.
    def candidate_quality(self, trajectory, validation):
        """Compare all accepted curves with the same CPU functional."""
        from .planning_quality import trajectory_quality
        limits = self._verification_native.get_cost("cspace").config.joint_limits
        return trajectory_quality(trajectory, validation["measurements"], self.problem,
            limit_joint_names=limits.joint_names,
            velocity_bounds=limits.velocity.detach().cpu().numpy(),
            acceleration_bounds=limits.acceleration.detach().cpu().numpy(),
            jerk_bounds=limits.jerk.detach().cpu().numpy())

    def select_candidate(self, result, *, candidate_indices=None, first_feasible=False, origin=None):
        """Keep failure evidence from every visited seed and accept only checked curves.

        The first feasible Phase A curve returns immediately so another seed
        cannot spend the budget needed to bind it. Phase B ranks independently
        accepted curves by the same quality functional used for that saved A.
        Native seed_cost is diagnostic only: its objective changes by phase.
        """
        if result.batch_size != 1 or not 1 <= result.num_seeds <= self.budget.trajectory_seeds:
            raise ValueError("Native results exceed the declared single-batch candidate budget")
        reports, selected, selected_key = [], None, None
        indices = range(result.num_seeds) if candidate_indices is None else candidate_indices
        for index in indices:
            index = int(index)
            native_metrics = self.native_diagnostics(result, seed_index=index)
            self.witnesses.add_native(native_metrics, duration=self.formal_clock["duration"])
            native_success = native_metrics["success"]
            candidate = _trajectory_candidate(result, index)
            report = {"seed_index": index, "native_success": native_success,
                "native_metrics": native_metrics, "native_seed_cost": float(result.seed_cost[0, index].item()),
                "seed_provenance": self.seed_strategies[index], "phase": self.phase,
                "optimizer": deepcopy(self.last_optimizer_diagnostics[index])}
            if origin is not None:
                report["candidate_origin"] = deepcopy(origin)
            validation = None
            # BEGIN final scalar-best output may already have been fully checked.
            # Its exact controls, scene and clock are unchanged within a solve.
            # Reuse a small verdict; execution still uses the existing bound
            # archive, never reconstructed or incomplete validation evidence.
            prior_verdict = None
            if origin is None:
                controls = candidate.solution[0, 0].detach().cpu().numpy()
                required_history = set(self.witnesses.parameters.tolist())
                prior_verdict = next((row for row in getattr(self, "_observed_verdicts", ())
                    if np.array_equal(controls, row["controls"])
                    and required_history.issubset(row["validated_history_parameters"])), None)
            # END no native cost or approximate control comparison identifies a curve.
            try:
                if native_success and prior_verdict is not None:
                    trajectory, points = self._native_curve(candidate), 0
                    trajectory.residuals = deepcopy(prior_verdict["residuals"])
                elif native_success:
                    trajectory, points = self.export(candidate)
                    validation = self.last_validation
                    self.witnesses.add_validation(validation)
                else:
                    trajectory, points = self._native_curve(candidate), 0
            except PlanningWorkExhausted:
                raise
            except ValueError as error:
                reports.append({**report, "accepted": False, "export_error": str(error)})
                continue
            accepted = native_success and all(np.isfinite(value) and value <= 0
                                               for value in trajectory.residuals.values())
            violations = [*trajectory.residuals.values(), *(
                value for label in ("rollout", "interpolated") if label in native_metrics
                for value in native_metrics[label]["constraints"].values())]
            severity = (sum(max(0., value) for value in violations)
                        if all(np.isfinite(violations)) else float("inf"))
            report.update(accepted=accepted, validation_points=points,
                independent_validation="completed" if native_success else "skipped_native_rejection",
                residuals=trajectory.residuals, continuation_violation=severity)
            if native_success and prior_verdict is not None:
                report.update(independent_validation="completed_prior_observation",
                    previously_considered=True, original_validation_points=prior_verdict["validation_points"],
                    candidate_origin=deepcopy(prior_verdict["candidate_origin"]),
                    measurements=deepcopy(prior_verdict["measurements"]),
                    numerical_tolerances=deepcopy(prior_verdict["numerical_tolerances"]))
            if validation is not None:
                report.update(measurements=validation["measurements"],
                              numerical_tolerances=validation["numerical_tolerances"])
            # BEGIN a completed hard check can refute an identical saved candidate.
            # Native rejection already forbids this curve; native success needs
            # a completed export verdict. Exceptions never reach this callback.
            # Observe every seed before selecting the next continuation.
            rejected = getattr(self, "_candidate_rejected", None)
            if not accepted and callable(rejected):
                rejected(trajectory, report)
            # END unfinished checks and unrelated curves cannot revoke an archive.
            if accepted:
                report["quality"] = (deepcopy(prior_verdict["quality"]) if prior_verdict is not None
                                     else self.candidate_quality(trajectory, validation))
            # BEGIN only a complete accepted verdict can avoid another export.
            # Failed exports must supply fresh per-time violations to the next
            # working set; retaining just their scalar residuals loses that data.
            if origin is not None and validation is not None and accepted:
                verdict = {name: deepcopy(report[name]) for name in (
                    "residuals", "validation_points", "candidate_origin", "measurements", "numerical_tolerances")}
                verdict["quality"] = deepcopy(report["quality"])
                verdict["controls"] = candidate.solution[0, 0].detach().cpu().numpy().copy()
                # BEGIN cache reuse also requires all currently known historical queries.
                # New witnesses from another observed curve do not certify
                # these times on this curve; a larger history forces export.
                verdict["validated_history_parameters"] = self.witnesses.parameters.tolist()
                # END accepted controls alone no longer identify the entire validation input.
                self._observed_verdicts.append(verdict)
            # END a rejected observed curve is rechecked before continuing its optimization.
            reports.append(report)
            key = (0, report["quality"]["score"], index) if accepted else (1, severity, index)
            if selected_key is None or key < selected_key:
                selected_key = key
                selected = candidate, trajectory, validation, report
            if accepted and (first_feasible or self.phase == "feasibility"):
                break
        if selected is None:
            errors = "; ".join(f"seed {item['seed_index']}: {item['export_error']}" for item in reports)
            raise ValueError(f"No native candidate has an exportable bounded curve: {errors}")
        candidate, trajectory, validation, report = selected
        self.last_validation = validation
        # BEGIN only the selected continuation supplies current working-set priorities.
        # Other seeds and block observations already contributed to history;
        # their failures must not masquerade as this candidate's current state.
        if origin is None:
            self.witnesses.set_current(report["native_metrics"], validation=validation,
                                       duration=self.formal_clock["duration"])
        # END observer selection cannot overwrite the final continuation's evidence.
        attempt = {**report, "selected_seed_index": report["seed_index"], "candidates": reports,
                   "failure_witness_parameters": self.witnesses.parameters.tolist()}
        return candidate, trajectory, attempt

    def refinement_samples(self):
        """Schedule bounded gradient queries; full history remains in export checks."""
        fixed = self.progress_map.anchor_parameters(self.anchors)[:, 0]
        capacity = min(self.budget.max_refinement_points,
                       min(self.budget.max_validation_points, _NATIVE_STAGE_POINT_LIMIT)-len(fixed))
        # BEGIN exact fixed/native representations do not consume extra query slots.
        # Existing extra points are deliberately excluded from covered_fixed:
        # they remain subject to this selection and may need their slot again.
        regular = np.arange(self.trajopt.horizon, dtype=np.float32)/np.float32(self.trajopt.horizon-1)
        covered_fixed = np.unique(np.r_[fixed, fixed.astype(np.float32).astype(float)])
        parameters = self.witnesses.select_parameters(fixed_parameters=covered_fixed,
            regular_parameters=regular, max_points=capacity)
        self.last_working_set = self.witnesses.last_selection
        # END finite capacity changes optimization scheduling, never the history ledger.
        return None if np.array_equal(parameters, self.refinement_parameters) else parameters
    # END the attempt witness set is independent of the selected continuation.

    # BEGIN cheap native reconstruction supplies initializers, never acceptance.
    def _native_curve(self, result):
        """Copy the bound native cubic without independent geometry queries."""
        if result.batch_size != 1 or result.num_seeds != 1:
            raise ValueError("Export requires exactly one native batch and one selected seed")
        native = self.trajopt.get_active_js(result.get_interpolated_plan())
        position = native.position.detach().cpu().numpy().reshape(-1, len(self.joint_names))
        velocity = native.velocity.detach().cpu().numpy().reshape(position.shape)
        dt = float(native.dt.detach().cpu().reshape(-1)[0])
        if (len(position) < 2 or not np.isfinite(position).all() or not np.isfinite(velocity).all()
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError("Native interpolation produced an invalid finite joint curve")
        times = np.arange(len(position), dtype=float) * dt
        if (len(times) != self.formal_clock["valid_count"] or dt != self.formal_clock["interpolation_dt"]
                or times[-1] != self.formal_clock["duration"]):
            raise ValueError("Native candidate changed the physical clock bound to the finger reference")
        event_records = []
        for event in self.problem.events:
            # BEGIN export the same flattened command record as the bound hand.
            clock = self.progress_map.event_clock[event_occurrence_key(event)]
            if clock["event"] != asdict(event):
                raise ValueError("Exported event differs from its frozen occurrence binding")
            event_records.append({**deepcopy(clock["event"]),
                                  **{key: deepcopy(value) for key, value in clock.items() if key != "event"}})
            # END the internal event payload is provenance, not a second command field.
        trajectory = JointTrajectory(self.joint_names, times, position, velocity,
                                     self.progress_map.u.copy(), self.progress_map.s.copy(),
                                     tuple(event_records), deepcopy(self.problem.conditions), {})
        trajectory.finger_trajectory = self.finger_trajectory
        return trajectory
    # END native rejection can still initialize OMPL on the identical clock.

    def export(self, result):
        """Build the same cubic on CPU, then check exact anchors and midpoints."""
        trajectory = self._native_curve(result)
        torch = self.torch
        times, position, velocity = trajectory.times, trajectory.position, trajectory.velocity
        dt = self.formal_clock["interpolation_dt"]
        anchor_pairs = self.progress_map.anchor_parameters(self.anchors)
        # BEGIN every historical failure remains a mandatory independent query.
        # Moving a point out of the gradient working set does not establish
        # positive clearance. The original spatial/query budget admits this
        # complete union or rejects export; no witness is silently truncated.
        validation_anchors = np.unique(np.r_[anchor_pairs[:, 0], self.witnesses.parameters])
        # END source goals and witness times share the unchanged physical clock.
        # BEGIN complete the existing native acceleration bound between samples.
        # These are the actual verification manager's named limits, after
        # native model construction. Input YAML scalars and sampled right-side
        # accelerations cannot substitute for the complete exported curve.
        cspace = self._verification_native.get_cost("cspace")
        if cspace is None or not cspace.enabled:
            raise ValueError("Export requires enabled native joint-state limits")
        limits = cspace.config.joint_limits
        derivatives = cubic_derivative_report(
            trajectory, limits.acceleration.detach().cpu().numpy(), joint_names=limits.joint_names)
        # END no new bound, tolerance, C2 requirement or retiming is introduced.
        # BEGIN bounded spatial work uses original cubics and complete body covers.
        motion, spatial_step, owner_speed = _export_motion_inputs(
            self._verification_fk.config.kinematics_config, self._verification_scene, self.problem,
            spatial_step_m=self.budget.validation_spatial_step_m)
        queries, spatial_report = _export_sample_times(trajectory, motion, spatial_step=spatial_step,
            owner_speed_per_progress=owner_speed, anchors=validation_anchors*times[-1], budget=self.budget,
            finger_geometry=_geometry_finger_model(self._verification_scene.gripper))
        spatial_report["finger_geometry_scope"] = ("complete rigid mechanical envelope; unknown physical finger speed"
            if "unknown_hand_mechanical_envelope" in motion.body_names else
            "same nominal-or-command polynomial or active coordinates selected by scene.gripper.state")
        q, dq, _ddq, progress = trajectory.sample(queries)
        q_t = self.device_cfg.to_device(q[None])
        dq_t = self.device_cfg.to_device(dq[None])
        s_t = self.device_cfg.to_device(progress[None])
        time_t = self.device_cfg.to_device(queries[None])
        scene = self._verification_scene
        positions, rotations, linear, angular = [], [], [], []
        geometry_terms, layer_queries = {}, []
        # BEGIN retain bounded source-solid witnesses for every exported validation chunk.
        source_self_reports = []
        # END complete per-pair arrays never accumulate across the trajectory.
        # BEGIN world diagnostics retain one worst source state for the complete curve.
        source_world_report = {}
        # END every state is checked even when its proxy cost vanishes.
        coverage_count, coverage_passed, coverage_failed = 0, True, []
        coverage_unknown = np.zeros(len(queries))
        needs_twist = any(direction.kind == "local_rate" for mechanism in self.problem.mechanisms
                          for direction in mechanism.directions)
        def check_work():
            # The outer request ledger owns the original wall-work deadline.
            # Check between bounded native batches, never reset it per chunk.
            callback = getattr(self, "_check_validation_work", None)
            if callback is not None:
                callback()
        # BEGIN completed batches may publish existing CPU-only evidence.
        # This optional recorder hook never observes an unfinished batch or
        # borrows last_validation from a previous candidate. In particular it
        # does not reduce/copy native tensor costs or repeat a geometry query.
        # A recorder owns any copies it retains. Ordinary diagnostic failures
        # cannot change verification, work checks, or candidate acceptance.
        progress_observer = getattr(self, "_export_progress_observer", None)
        def publish_progress(completed_points, source_self=None):
            if progress_observer is None:
                return
            try:
                progress_observer({
                    "planned_points": len(queries), "checked_points": completed_points,
                    "curve_duration_s": trajectory.duration,
                    "checked_time_interval_s": (None if not completed_points else
                        [float(queries[0]), float(queries[completed_points-1])]),
                    "source_world_collision": source_world_report,
                    "source_self_collision_batch": source_self,
                    "robot_coverage": {"checked_states": coverage_count,
                        "passed": coverage_passed,
                        "attempt_disallowed_states": int(np.count_nonzero(coverage_unknown[:completed_points])),
                        "failed_witnesses": coverage_failed},
                    "analytic_acceleration": derivatives,
                })
            except Exception:
                # The experiment recorder reports its own I/O errors. Other
                # optional observers are equally unable to alter the result.
                pass
        publish_progress(0)
        # END no partial record supplies a planner validation verdict.
        def fk_at(moments, values, rates):
            check_work()
            joint_state = self.JointState(position=values, velocity=rates,
                acceleration=self.device_cfg.to_device(trajectory._curve(moments, 2)[None]),
                jerk=self.device_cfg.to_device(trajectory._curve(moments, 3)[None]),
                dt=self.device_cfg.to_device([dt]), joint_names=list(self.joint_names))
            state = self._verification_fk.compute_kinematics(joint_state)
            tool = state.tool_poses
            index = tool.tool_frames.index(self.problem.tool_link)
            p, r = self.problem.transform_tool_pose(tool.position[:, :, index], tool.quaternion[:, :, index])
            return joint_state, state, index, p, r
        with torch.no_grad():
            # Freeze every release from the complete ordered anchor history
            # before splitting geometry. A later batch must not substitute
            # its first local sample for an earlier actual release pose.
            if any(relation.release_condition_id is not None for owner in scene.owners
                   for relation in owner.attachments):
                anchor_times = anchor_pairs[:, 0]*times[-1]
                aq, adq, _, aprogress = trajectory.sample(anchor_times)
                anchor_p, anchor_r = [], []
                for begin in range(0, len(anchor_times), _VALIDATION_CHUNK_SIZE):
                    end = begin+_VALIDATION_CHUNK_SIZE
                    _, _, _, p, r = fk_at(anchor_times[begin:end],
                        self.device_cfg.to_device(aq[None, begin:end]),
                        self.device_cfg.to_device(adq[None, begin:end]))
                    anchor_p.append(p.clone())
                    anchor_r.append(r.clone())
                scene = scene.with_release_history(torch.cat(anchor_p, dim=1), torch.cat(anchor_r, dim=1),
                                                   self.device_cfg.to_device(aprogress[None]))
                del anchor_p, anchor_r
            for begin in range(0, len(queries), _VALIDATION_CHUNK_SIZE):
                end = min(begin+_VALIDATION_CHUNK_SIZE, len(queries))
                selected = slice(begin, end)
                joint_state, state, tool_index, p, r = fk_at(queries[selected], q_t[:, selected], dq_t[:, selected])
                # FK reuses its output buffers. Own only small TCP/twist arrays;
                # complete robot sphere arrays never survive this iteration.
                positions.append(p.clone())
                rotations.append(r.clone())
                if needs_twist:
                    twist = torch.einsum("...kd,...d->...k", state.tool_jacobians[:, :, tool_index], dq_t[:, selected])
                    v, w = self.problem.transform_tool_velocity(twist[..., :3], twist[..., 3:],
                                                                 state.tool_poses.quaternion[:, :, tool_index])
                    linear.append(v.clone())
                    angular.append(w.clone())
                layer_record = {}
                components = {"scene": _scene_boundary_evaluate(scene,
                    self.problem, self.progress_map, self.problem.transform_robot_spheres(state.robot_spheres),
                    p, r, s_t[:, selected], q_t[:, selected], self.joint_names,
                    trajectory_time=time_t[:, selected],
                    normalized_time=self.device_cfg.to_device((queries[selected]/times[-1])[None]),
                    layer_selection_out=layer_record, source_geometry=True,
                    diagnostics=source_world_report)}
                layer_record["sample_start"] = begin
                layer_record["sample_count"] = end-begin
                layer_queries.append(layer_record)
                native_constraints = self._verification_native.compute_costs(
                    self._RobotState(joint_state=joint_state, cuda_robot_model_state=state))
                for name, value in zip(native_constraints.names, native_constraints.values):
                    components[f"native:{name}"] = value
                # BEGIN export repeats the same source-solid decision on each actual query state.
                source_report = {}
                components["source_self_collision"] = scene.check_robot_self_collision(
                    q_t[:, selected], self.joint_names, time_t[:, selected], diagnostics=source_report)
                source_self_reports.append(source_report)
                # END joint/derivative constraints remain owned by the native manager.
                for name, value in components.items():
                    reduced = value.reshape(value.shape[0], value.shape[1], -1).amax(-1)
                    geometry_terms.setdefault(name, []).append(reduced.clone())
                # The shared complete-solid oracle consumes bounded CPU batches
                # with the true entrance flag exactly once, preserving its own
                # cumulative finite-work accounting and unknown-space policy.
                coverage = []
                start = begin
                if begin == 0:
                    coverage = scene.check_robot_coverage(torch.as_tensor(q[None, :1], dtype=torch.float64),
                        self.joint_names, torch.as_tensor(queries[None, :1], dtype=torch.float64), entrance=True)
                    start = 1
                if start < end:
                    coverage += scene.check_robot_coverage(torch.as_tensor(q[None, start:end], dtype=torch.float64),
                        self.joint_names, torch.as_tensor(queries[None, start:end], dtype=torch.float64))
                if coverage:
                    if len(coverage) != end-begin:
                        raise ValueError("Complete-solid coverage returned an incomplete validation chunk")
                    coverage_count += len(coverage)
                    coverage_unknown[selected] = [0. if row["attempt_allowed"] else 1. for row in coverage]
                    coverage_passed &= all(row["passed"] for row in coverage)
                    coverage_failed.extend({"time_s": float(queries[begin+i]), "result": row}
                        for i, row in enumerate(coverage) if not row["passed"] and len(coverage_failed) < 16)
                    coverage_failed = coverage_failed[:16]
                # BEGIN checkpoint only after every original batch operation.
                publish_progress(end, source_report)
                # END the existing cumulative-budget boundary remains here.
                check_work()
            # Process constraints may use ordered winding/first-event history.
            # Evaluate them once on the full small TCP/q arrays, never on an
            # isolated geometry chunk. Native spheres are no longer required.
            p, r = torch.cat(positions, dim=1), torch.cat(rotations, dim=1)
            tool_linear = torch.cat(linear, dim=1) if needs_twist else None
            tool_angular = torch.cat(angular, dim=1) if needs_twist else None
            components = self.problem.residuals(p, r, s_t, q_t, self.joint_names, joint_velocity=dq_t,
                trajectory_time=time_t, tool_linear_velocity=tool_linear, tool_angular_velocity=tool_angular)
            components.update({name: torch.cat(values, dim=1) for name, values in geometry_terms.items()})
            residuals = {name: float(value.max().item()) for name, value in components.items()}
            measurements = {name: float(value.max().item()) for name, value in
                            self.problem.measurements(p, r, s_t, trajectory_time=time_t).items()}
            violation = torch.zeros_like(s_t)
            for name, value in components.items():
                reduced = value.reshape(value.shape[0], value.shape[1], -1).amax(-1)
                violation = torch.maximum(violation, reduced)
                if residuals[name] > 0:
                    index = int(reduced.reshape(-1).argmax().item())
                    measurements[f"{name}:maximum_violation_u"] = float(queries[index]/times[-1])
                    measurements[f"{name}:maximum_violation_progress"] = float(progress[index])
                    measurements[f"{name}:maximum_violation_time"] = float(queries[index])
            self.last_validation = {
                "measurements": measurements, "numerical_tolerances": asdict(self.problem.numerical),
                "u": queries/times[-1], "violations": violation.detach().cpu().numpy().reshape(-1),
                "spatial_resolution": spatial_report, "scene_layer_queries": layer_queries,
                "source_self_collision": {"checked_states": sum(
                    report["checked_states"] for report in source_self_reports), "chunks": source_self_reports},
                "source_world_collision": source_world_report,
                "coordinate_direction_resolution": {"query_count": len(queries),
                    "maximum_physical_time_gap_s": float(np.max(np.diff(queries))),
                    "local_rate": "instantaneous native geometric Jacobian times actual cubic joint velocity",
                    "scope": "finite query validation; no continuous-time sign certificate"},
            }
            if coverage_count:
                residuals["unknown/new_robot_volume"] = float(coverage_unknown.max())
                self.last_validation["violations"] = np.maximum(self.last_validation["violations"], coverage_unknown)
                self.last_validation["robot_coverage"] = {"queries": coverage_count,
                    "attempt_allowed": bool(not coverage_unknown.any()), "passed": coverage_passed,
                    "failed": coverage_failed,
                    "scope": "native solid volume at every exported validation sample; not continuous collision proof"}
        # END geometry is bounded in memory; chronological process history is complete.
        # BEGIN final scope and physical-domain checks refer to this exact curve.
        scope_issues = self.problem.verification_issues(progress)
        self.last_validation["scope_issues"] = scope_issues
        residuals["required_relation_query_incomplete"] = float(bool(scope_issues))
        columns = [limits.joint_names.index(name) for name in self.joint_names]
        position_limits = limits.position[:, columns].detach().cpu().numpy()
        position_report = cubic_position_report(trajectory._curve, *position_limits)
        residuals["continuous_position"] = position_report["maximum_bound_excess"]
        self.last_validation["continuous_position"] = position_report
        # END sampled native success does not replace polynomial or history checks.
        residuals["initial_position"] = max(0., float(np.max(np.abs(position[0]-self.start.position[0].detach().cpu().numpy())))
                                              - self.problem.numerical.joint)
        residuals["initial_velocity"] = max(0., float(np.max(np.abs(velocity[0]-self.start.velocity[0].detach().cpu().numpy())))
                                              - self.problem.numerical.velocity)
        # BEGIN require the existing acceleration limits on every cubic piece.
        # This residual is raw joint-coordinate acceleration excess, not a
        # weighted native cost. The original candidate selector already rejects
        # every positive residual. Jump magnitudes remain diagnostic only.
        residuals["continuous_acceleration"] = derivatives["acceleration"]["maximum_bound_excess"]
        self.last_validation["cubic_derivatives"] = derivatives
        witness = derivatives["acceleration"]["maximum_violation"]
        if witness is not None:
            parameter = witness["time_s"]/times[-1]
            self.last_validation["measurements"].update({
                "continuous_acceleration:maximum_violation_time": witness["time_s"],
                "continuous_acceleration:maximum_violation_u": parameter,
                "continuous_acceleration:maximum_violation_progress": float(np.interp(
                    parameter, self.progress_map.u, self.progress_map.s)),
            })
            # The existing bounded refinement consumes these coordinates and
            # residuals, not measurements. Retain the exact analytic witness
            # even if that time already has a sampled right-side acceleration;
            # cubic_derivatives above identifies the failing piece and side.
            self.last_validation["u"] = np.r_[self.last_validation["u"], parameter]
            self.last_validation["violations"] = np.r_[
                self.last_validation["violations"], witness["excess"]]
        # END continuous acceleration acceptance; C1/C2 reports do not add gates.
        trajectory.residuals = residuals
        return trajectory, len(queries)


# BEGIN request-wide execution, failure and rebind accounting.
def plan_trajectory(problem, scene, robot, *, robot_validator, budget=None, device="cuda:0", continuation=None,
                    request_ledger=None, rebind=None):
    """Generate one fully checked candidate within the original request budget.

    ``request_ledger`` continues failed planning attempts. ``continuation``
    carries actual execution facts. An explicit ``rebind`` rebuilds the soft
    schedule of remaining work or creates selected retry occurrences; ordinary
    continuation preserves the original frozen clock. Every request admitted
    after source-model binding, including initialization failure, carries the cumulative ledger. No GPU stage is
    claimed to be preemptible: elapsed overrun rejects its candidate.
    """
    from .execution_binding import bind_candidate, plain
    from .planning_problem import TimeBindingError
    started = time.monotonic()
    prior = planner = ledger = None
    # BEGIN only a fully validated and bound CPU candidate enters this archive.
    best_attempt = None
    # END the shared request ledger remains outside candidate snapshots.
    initial_seconds = 0.
    original_export = original_ik_solve = original_ik_rank = None
    diagnostics = {"attempts": []}
    outcome = PlanningResult(False, None, diagnostics)
    stage = "request_binding"
    try:
        # BEGIN every request owns its mutable scene and binds mandatory source solids before its ledger.
        scene = scene.clone_for_rollout()
        scene.bind_robot_validator(robot_validator, problem)
        # END clones share the immutable model policy, not candidate hand clocks or query evidence.
        previous = request_ledger
        if previous is None and isinstance(continuation, dict) and isinstance(continuation.get("original_binding"), dict):
            previous = continuation.get("request_ledger", continuation["original_binding"].get("request_ledger"))
        if previous is not None:
            original_budget = PlanningBudget(**previous["planning_budget"])
        else:
            original_budget = PlanningBudget() if budget is None else budget
        # BEGIN execution facts enter a trusted work ledger only after validation.
        # An executor may already have appended its new observations to the
        # supplied ledger. Keep their computation history, but restore the
        # last frozen binding's facts until this continuation passes. A failed
        # request must never return unverified completion IDs as trusted facts.
        execution_fields = ("history", "event_records", "timing_events", "timing_checks",
            "completed_goal_ids", "goal_evidence", "completed_direction_ids", "direction_evidence",
            "effects", "effect_history")
        ledger_input = previous
        if isinstance(continuation, dict) and isinstance(continuation.get("original_binding"), dict):
            original_ledger = continuation["original_binding"]["request_ledger"]
            ledger_input = deepcopy(previous)
            for name in execution_fields:
                ledger_input[name] = deepcopy(original_ledger[name])
            diagnostics["continuation_facts_pending_validation"] = {
                name: deepcopy(continuation[name]) for name in (*execution_fields, "actual_feedback")
                if name in continuation}
        ledger = _request_work_ledger(problem, original_budget, ledger_input)
        if continuation is not None:
            ledger["requires_continuation"] = True
        # END pending facts remain diagnostics, not completed obligations.
        consumed = ledger["consumed_planning"]
        initial_seconds = consumed["planning_seconds"]
        request_budget = asdict(original_budget)
        diagnostics.update(planning_budget=request_budget, consumed_planning=consumed)
        def check(next_stage=None):
            _check_work_budget(original_budget, consumed, started=started,
                               initial_seconds=initial_seconds, stage=next_stage)
        check("binding_attempts")
        consumed["binding_attempts"] += 1
        if previous is not None and budget is not None and asdict(budget) != asdict(original_budget):
            raise ContinuationBindingError("A same request cannot reset or enlarge its original planning budget")
        # BEGIN each admitted binding owns mutable problem and device state.
        # Initial clock construction can bind a hand before native allocation
        # fails. A private problem is therefore required for both first and
        # subsequent attempts, just as for remaining-obligation filtering.
        problem = deepcopy(problem)
        if previous is not None:
            _restore_original_direction_origins(problem, ledger["obligation_contract"])
        # END each admitted binding owns mutable problem and device state.
        if _obligation_contract(problem) != ledger["obligation_contract"]:
            raise ContinuationBindingError("A same request cannot remove, reorder or change original goal/process obligations")
        if continuation is not None:
            if not isinstance(continuation, dict) or not isinstance(continuation.get("original_binding"), dict):
                raise ContinuationBindingError("Continuation requires its original formal execution binding")
            if previous is None:
                raise ContinuationBindingError("Continuation lacks the original cumulative request ledger")
            # Binding owns its filtered obligations and hand occurrences. The
            # caller retains the full source problem after any failed attempt;
            # the scene clone shares only immutable geometry and the existing
            # cumulative robot oracle, never a reset query allowance.
            current = deepcopy(continuation)
            current["consumed_planning"] = deepcopy(consumed)
            budget, prior = _prepare_continuation(problem, scene, original_budget, current, rebind=rebind)
            # BEGIN commit actual facts only after their identities, clock and proofs pass.
            ledger = _request_work_ledger(problem, original_budget, ledger, continuation)
            consumed = ledger["consumed_planning"]
            diagnostics["consumed_planning"] = consumed
            diagnostics.pop("continuation_facts_pending_validation", None)
            # END a rejected continuation still retains its real planning work counters.
            prior["history"] = deepcopy(ledger["history"])
            prior["timing_events"] = deepcopy(ledger["timing_events"])
        else:
            if ledger["requires_continuation"] or any(ledger[name] for name in execution_fields if name != "timing_events"):
                raise ContinuationBindingError("An executed request needs its actual continuation facts before another binding")
            if plain(problem.events) != ledger["original_events"]:
                raise ContinuationBindingError("A failed-request retry changed the original event/source requirements")
            if rebind is not None and (not rebind.get("source") or rebind.get("kind") != "remaining"):
                raise ContinuationBindingError("Creating retry occurrences requires an actual execution continuation")
            remaining_solves = original_budget.maximum_trajectory_solves-consumed["trajectory_solves"]
            if remaining_solves <= 0:
                raise PlanningWorkExhausted("The original cumulative trajectory-solve budget is exhausted")
            budget = replace(original_budget, maximum_trajectory_solves=remaining_solves,
                search_seconds=max(0., original_budget.search_seconds-consumed["search_seconds"]))
        check("initializations")
        stage = "initialization"
        consumed["initializations"] += 1
        planner = _CuroboPlanner(problem, scene, robot, budget, device,
                                 robot_validator=robot_validator, continuation=prior)
        check()
        diagnostics.update(backend="curobo_v2", batch_size=1, ik_seeds=budget.ik_seeds,
            trajectory_seeds=budget.trajectory_seeds, used_ompl=False,
            timing_policy="explicit_rebind_then_frozen" if rebind is not None else "source_proposal_then_frozen_binding",
            native_trajectory_dt=planner.budget.maximum_trajectory_dt,
            progress_parameter_error_bound=_PROGRESS_PARAMETER_ERROR, progress_clock_max_keys=_MAX_PROGRESS_KEYS,
            scene=scene.describe(), scene_workspace_bytes=planner.scene_workspace_bytes,
            scene_cpu_workspace_bytes=planner.scene_cpu_workspace_bytes,
            scene_gpu_workspace_bytes=planner.scene_gpu_workspace_bytes,
            anchor_fk_workspace_bytes=planner.anchor_fk_workspace_bytes, exact_anchor_capacity=planner.anchor_capacity,
            validation_storage=deepcopy(getattr(planner, "validation_storage", {})))
        stage = "initial_state_verification"
        consumed["validation_queries"] += 1
        diagnostics["initial_state"] = planner.initial_state_diagnostics()
        check()
        if not diagnostics["initial_state"]["passed"]:
            diagnostics["failure"] = "initial_state_infeasible"
            return outcome
        grid = _fixed_spline_grid(intervals=planner.budget.knots+4,
            native_dt=planner.budget.maximum_trajectory_dt, interpolation_steps=planner.budget.interpolation_steps,
            interpolation_dt=planner.budget.interpolation_dt)
        diagnostics["event_time_checks"] = _validate_bound_event_times(problem, planner.progress_map, grid["duration"],
            0. if prior is None else prior["request_time_origin_s"], ledger["history"],
            None if prior is None else prior["goal_identities"], ledger["timing_events"])

        # Count actual export passes for all declared candidates. An export
        # exception consumes its admitted maximum query allowance because its
        # partial native work cannot be reconstructed from a missing result.
        planner._check_validation_work = check
        original_export = planner.export
        def tracked_export(*args, **kwargs):
            check("validation_passes")
            consumed["validation_passes"] += 1
            observation = bool(getattr(planner, "_observing_candidate", False))
            if observation:
                work = diagnostics.setdefault("candidate_observation_work", {})
                work["export_passes"] = work.get("export_passes", 0)+1
            queries = budget.max_validation_points
            try:
                exported = original_export(*args, **kwargs)
                queries = int(exported[1])
            finally:
                consumed["validation_queries"] += queries
                if observation:
                    work["export_queries"] = work.get("export_queries", 0)+queries
            # BEGIN preserve an original environment/API exception during charging.
            check()
            return exported
            # END only a normally returned export is checked for subsequent exhaustion.
        planner.export = tracked_export
        # BEGIN static IK screening shares the request's finite source-state budget.
        if callable(getattr(planner, "_rank_ik_candidates", None)):
            original_ik_rank = planner._rank_ik_candidates
            def tracked_ik_rank(candidates, *args, **kwargs):
                check("validation_queries")
                consumed["validation_queries"] += int(np.isfinite(candidates).all(axis=-1).sum())
                check()
                result = original_ik_rank(candidates, *args, **kwargs)
                check()
                return result
            planner._rank_ik_candidates = tracked_ik_rank
        # END errors preserve both their original identity and admitted work charge.
        # Reference IK and any later native IK calls share the same counter.
        # Seed proposal loops with no native IK are charged by search work.
        if callable(getattr(planner.ik, "solve_pose", None)):
            original_ik_solve = planner.ik.solve_pose
            def tracked_ik(*args, **kwargs):
                check("ik_queries")
                goal = args[0] if args else kwargs["goal_tool_poses"]
                # return_seeds is only an output count; native IK still
                # evaluates its configured seed population for each goal.
                queries = int(goal.batch_size)*max(int(planner.ik.config.num_seeds),
                                                   int(kwargs.get("return_seeds", 1)))
                if consumed["ik_queries"]+queries > original_budget.max_validation_points*original_budget.ik_seeds*original_budget.maximum_trajectory_solves:
                    raise PlanningWorkExhausted("The original cumulative ik_queries budget is exhausted")
                consumed["ik_queries"] += queries
                # BEGIN an IK runtime error must not become an ignorable quality timeout.
                result = original_ik_solve(*args, **kwargs)
                check()
                return result
                # END counts were charged before the native call, including errors.
            planner.ik.solve_pose = tracked_ik
        check("ik_batches")
        consumed["ik_batches"] += 1
        stage = "reference_ik"
        seeds, ik_success = planner.reference_seeds()
        check()
        diagnostics.update(reference_ik_success_counts=ik_success,
            ik_exit_early=bool(planner.ik.config.exit_early),
            reference_ik_distinct_counts=planner.reference_ik_distinct_counts,
            initial_seed_strategies=planner.initial_seed_strategies,
            reference_ik_queries=deepcopy(planner.reference_ik_queries),
            reference_ik_chains=deepcopy(planner.reference_ik_chains),
            reference_seed_fits=deepcopy(planner.reference_seed_fits))
        diagnostics["functional_ik_candidates"] = deepcopy(getattr(planner, "goal_ik_candidates", {}))
        diagnostics["functional_ik_fit_hints"] = deepcopy(planner.goal_ik_fit_hints)
        # BEGIN seed, block and final candidates share one complete binding boundary.
        def qualify_candidate(trajectory, validation, attempt):
            nonlocal best_attempt
            if not attempt["accepted"] or validation is None:
                raise ValueError("Only a completely validated candidate may enter execution binding")
            if best_attempt is not None and attempt["quality"]["score"] >= best_attempt["quality"]["score"]:
                attempt["quality_decision"] = "retained_better_or_equal_feasible_candidate"
                return False
            diagnostics.update(executable_until=trajectory.executable_until,
                               next_verification=trajectory.next_verification)
            binding = bind_candidate(trajectory, problem=problem, scene=scene,
                validation=deepcopy(validation), native_report=attempt)
            for event in problem.events:
                if event.timing and not any(event_occurrence_key(row) == event_occurrence_key(event)
                                            for row in ledger["timing_events"]):
                    ledger["timing_events"].append(plain(event))
            retry_goal_ids = set() if prior is None else set(prior["retry_goal_ids"])
            retry_direction_ids = set() if prior is None else set(prior["retry_direction_ids"])
            binding.update(planning_budget=deepcopy(request_budget),
                request_events=deepcopy(ledger["original_events"]),
                consumed_events=[] if prior is None else prior["consumed_events"],
                goal_identities=({name: list(range(len(getattr(problem, name))))
                    for name in ("pose_events", "coordinate_events", "supports")}
                    if prior is None else prior["goal_identities"]),
                completed_goal_ids=[deepcopy(value) for value in
                                    (ledger["completed_goal_ids"] if prior is None else prior["completed_goal_ids"])
                                    if tuple(value) not in retry_goal_ids],
                goal_evidence=deepcopy(ledger["goal_evidence"]),
                direction_identities=([[(index, column) for column in range(len(mechanism.directions))]
                    for index, mechanism in enumerate(getattr(problem, "mechanisms", ()))]
                    if prior is None else deepcopy(prior["direction_identities"])),
                completed_direction_ids=[deepcopy(value) for value in
                    (ledger["completed_direction_ids"] if prior is None else prior["completed_direction_ids"])
                    if tuple(value) not in retry_direction_ids],
                retry_direction_ids=sorted(retry_direction_ids),
                direction_evidence=deepcopy(ledger["direction_evidence"]),
                history=deepcopy(ledger["history"]), effect_history=deepcopy(ledger["effect_history"]),
                timing_events=deepcopy(ledger["timing_events"]), timing_checks=deepcopy(ledger["timing_checks"]),
                request_time_origin_s=0. if prior is None else prior["request_time_origin_s"],
                request_time_origin_step=0 if prior is None else prior["request_time_origin_step"])
            if prior is not None:
                binding.update({name: deepcopy(prior[name]) for name in (
                    "physics_budget", "consumed_physics_steps", "remaining_physics_steps",
                    "device_policy", "active_finger_interval")})
            # All work required to qualify and bind this candidate must
            # finish within budget. Cleanup after that fact is recorded,
            # but cannot erase an already available qualified candidate.
            check()
            diagnostics["qualified_candidate_elapsed_seconds"] = time.monotonic()-started
            outcome.success, outcome.trajectory, outcome.execution_binding = True, trajectory, binding
            # BEGIN preserve the qualified curve before rebuilding the sole solver.
            best_attempt = deepcopy(attempt)
            diagnostics["best_feasible_quality"] = deepcopy(attempt["quality"])
            return True

        def charge_candidate_queries(count):
            check("validation_queries")
            consumed["validation_queries"] += int(count)
            work = diagnostics.setdefault("candidate_observation_work", {})
            work["screen_and_native_queries"] = work.get("screen_and_native_queries", 0)+int(count)
            check()

        planner._candidate_qualified = qualify_candidate
        # BEGIN newly discovered history can explicitly refute a prior identical curve.
        def reject_candidate(trajectory, report):
            nonlocal best_attempt
            saved = outcome.trajectory
            if (best_attempt is not None and saved is not None
                    and trajectory.joint_names == saved.joint_names
                    and all(np.array_equal(getattr(trajectory, name), getattr(saved, name))
                            for name in ("times", "position", "velocity"))):
                diagnostics["invalidated_feasible_candidate"] = {
                    "reason": "same_curve_failed_additional_validation", "rejection": deepcopy(report)}
                diagnostics.pop("best_feasible_quality", None)
                best_attempt = None
                outcome.success, outcome.trajectory, outcome.execution_binding = False, None, None
        planner._candidate_rejected = reject_candidate
        # END another candidate's failure alone leaves the original complete binding intact.
        planner._charge_candidate_queries = charge_candidate_queries
        planner.candidate_observations = diagnostics.setdefault("candidate_observations", [])
        planner._observed_verdicts = []
        # END observations use the original request ledger and explicit validation.
        reference_seeds, searched = seeds, False
        background_already_refined = False
        for iteration in range(budget.maximum_trajectory_solves):
            # BEGIN map refinement is fixed before the next native solve/observer.
            refine_background = getattr(planner, "refine_background_for_seed", None)
            if callable(refine_background) and not background_already_refined:
                stage = "background_refinement"
                report = (refine_background(seeds) if best_attempt is None else
                          {"updated": False, "reason": "preserve_qualified_candidate_scene"})
                diagnostics.setdefault("background_refinements", []).append(report)
                scene = planner.scene
                diagnostics["scene"] = scene.describe()
                check()
            background_already_refined = False
            # END each later binding names the scene actually used for validation.
            check("trajectory_solves")
            consumed["trajectory_solves"] += 1
            stage = "trajectory_optimization_and_verification"
            native_result, trajectory, attempt = planner.select_candidate(planner.optimize(seeds))
            # BEGIN rejected continuations advance their own functional objective.
            # A completely accepted candidate must be bound before optional
            # queries can exhaust work. Its existing duals also remain valid
            # during optional quality work; acceptance needs no further AL step.
            attempt["functional_augmented_lagrangian"] = (
                planner.update_functional_constraints(native_result)
                if not attempt["accepted"] and best_attempt is None else
                {"changed": False, "reason": "preserve_completely_qualified_candidate"})
            # END map changes cannot skip or duplicate this multiplier update.
            check()
            native_success = attempt["native_success"]
            diagnostics["attempts"].append(attempt)
            diagnostics.update(attempt)
            if attempt["accepted"]:
                if attempt.get("previously_considered"):
                    if best_attempt is None:
                        raise RuntimeError("A qualified observation lost its execution archive")
                    attempt["quality_decision"] = "retained_previously_bound_observation"
                    break
                if not qualify_candidate(trajectory, planner.last_validation, attempt):
                    break
                if planner.phase == "quality" or iteration+1 >= budget.maximum_trajectory_solves:
                    break
                stage = "quality_phase_initialization"
                check("trajectory_solves")
                seeds = native_result.solution.detach().clone()
                planner.seed_strategies = [attempt["seed_provenance"]]
                # A complete accepted curve has no current violations. Old
                # evidence stays in independent validation, not gradient slots.
                additional = planner.refinement_samples()
                attempt["working_set"] = deepcopy(planner.last_working_set)
                planner.refine_validation(planner.refinement_parameters if additional is None else additional,
                                          phase="quality")
                attempt["next_action"] = "improve_bound_feasible_curve"
                check()
                continue
                # END optional quality work consumes the original shared budget.
            if best_attempt is not None:
                attempt["quality_decision"] = "retained_feasible_candidate_after_rejection"
                break
            # BEGIN all changes to the next subproblem share one rebuild boundary.
            stagnant = attempt["optimizer"]["unchanged_from_seed"]
            if stagnant:
                attempt["stop_reason"] = "optimizer_returned_unchanged_infeasible_curve"
            if iteration+1 >= budget.maximum_trajectory_solves:
                break
            stage = "validation_refinement"
            additional = planner.refinement_samples()
            attempt["working_set"] = deepcopy(planner.last_working_set)
            parameters = planner.refinement_parameters if additional is None else additional
            check()
            map_changed = False
            if callable(refine_background):
                stage = "rejected_candidate_background_refinement"
                refinement_seed = native_result.solution.detach().clone()
                report = refine_background(refinement_seed, rebuild=False)
                diagnostics.setdefault("background_refinements", []).append({**report, "phase": "rejected_candidate"})
                scene = planner.scene
                diagnostics["scene"] = scene.describe()
                check()
                map_changed = report["updated"]
            functional_changed = attempt["functional_augmented_lagrangian"]["changed"]
            if map_changed or functional_changed or (native_success and additional is not None):
                seeds = native_result.solution.detach().clone()
                planner.seed_strategies = [attempt["seed_provenance"]]
                planner.refine_validation(parameters, phase="feasibility")
                background_already_refined = True
                attempt.update(next_action="optimize_updated_constraints", refinement_parameters=parameters.tolist(),
                    subproblem_changes={"functional_parameters": functional_changed,
                                        "background": map_changed, "working_set": additional is not None})
                check()
                continue
            # END no old curvature, cost tracker or capture survives changed objective inputs.
            if not searched and budget.search_seconds > 0:
                searched, stage = True, "bounded_seed_search"
                search_started = time.monotonic()
                try:
                    path_seed, search_report = planner.ompl_seed(reference_seeds, trajectory)
                    consumed["search_queries"] += int(search_report.get("state_queries", 0))
                except Exception:
                    consumed["search_queries"] += budget.max_validation_points
                    raise
                finally:
                    consumed["search_seconds"] += time.monotonic()-search_started
                check()
                diagnostics["ompl"] = search_report
                if path_seed is not None:
                    seeds, planner.seed_strategies = path_seed, ["ompl"]
                    # BEGIN an independent initializer owns a fresh multiplier sequence.
                    planner.reset_functional_constraints()
                    # Its new spatial path can require another observed map.
                    # Defer that replacement too, so reset, map and query-set
                    # changes still construct only one next native solver.
                    if callable(refine_background):
                        stage = "searched_candidate_background_refinement"
                        report = refine_background(seeds, rebuild=False)
                        diagnostics.setdefault("background_refinements", []).append(
                            {**report, "phase": "ompl_initializer"})
                        scene = planner.scene
                        diagnostics["scene"] = scene.describe()
                        check()
                    planner.refine_validation(parameters, phase="feasibility")
                    background_already_refined = True
                    # END independent validation still checks every historical witness.
                    diagnostics["used_ompl"] = True
                    attempt["next_action"] = "optimize_complete_ompl_seed"
                    continue
            if additional is not None:
                seeds = native_result.solution.detach().clone()
                planner.seed_strategies = [attempt["seed_provenance"]]
                planner.refine_validation(additional, phase="feasibility")
                background_already_refined = True
                attempt.update(next_action="refine_uncovered_curve_samples", refinement_parameters=additional.tolist())
                check()
                continue
            break
            # END every failure remains recorded even when no new solve is useful.
        if best_attempt is not None:
            diagnostics.update(best_attempt)
            diagnostics["returned_candidate"] = deepcopy(best_attempt)
            return outcome
        diagnostics["failure"] = ("optimization_stagnated" if diagnostics["attempts"][-1]["optimizer"]["unchanged_from_seed"]
            else "exported_curve_constraint_violation" if diagnostics["native_success"]
            else "complete_process_optimization_not_feasible")
        return outcome
    except Exception as error:
        # BEGIN optional work exhaustion cannot erase a previously bound feasible curve.
        if isinstance(error, PlanningWorkExhausted) and best_attempt is not None:
            diagnostics.update(best_attempt)
            diagnostics.update(quality_stop_reason=str(error), quality_stopped_stage=stage,
                               returned_candidate=deepcopy(best_attempt))
            return outcome
        # END unexpected runtime/API/cleanup failures remain explicit failures below.
        diagnostics.update(failure=("cumulative_planning_budget_exhausted" if isinstance(error, PlanningWorkExhausted)
            else "continuation_binding_gap" if isinstance(error, ContinuationBindingError)
            else "event_time_requirement_unsatisfied" if isinstance(error, TimeBindingError)
            else "physical_domain_initializer_not_constructed" if isinstance(error, JointDomainInitializationError)
            else "planning_stage_failed"), failed_stage=stage, reason=str(error), error_type=type(error).__name__)
        outcome.success, outcome.trajectory, outcome.execution_binding = False, None, None
        return outcome
    finally:
        if planner is not None:
            if original_export is not None:
                planner.export = original_export
            if original_ik_solve is not None:
                planner.ik.solve_pose = original_ik_solve
            if original_ik_rank is not None:
                planner._rank_ik_candidates = original_ik_rank
            for name in ("scene_workspace_bytes", "scene_cpu_workspace_bytes", "scene_gpu_workspace_bytes",
                         "anchor_fk_workspace_bytes", "anchor_capacity"):
                diagnostics["exact_anchor_capacity" if name == "anchor_capacity" else name] = getattr(planner, name)
            try:
                planner.close()
            except Exception as error:
                outcome.success, outcome.trajectory, outcome.execution_binding = False, None, None
                diagnostics.update(failure="planner_cleanup_failed", reason=str(error))
        diagnostics["elapsed_seconds"] = time.monotonic()-started
        if ledger is not None:
            ledger["consumed_planning"]["planning_seconds"] = initial_seconds+diagnostics["elapsed_seconds"]
            if ledger["consumed_planning"]["planning_seconds"] >= ledger["planning_budget"]["planning_seconds"]:
                diagnostics["wall_time_overrun_s"] = ledger["consumed_planning"]["planning_seconds"]-ledger["planning_budget"]["planning_seconds"]
                diagnostics["planning_budget_exhausted"] = True
                diagnostics["budget_exhausted_after_qualification"] = outcome.success
            # BEGIN distinguish planning-only retries from actual-world continuation.
            # A previous work ledger carries cumulative computation, but does
            # not by itself supply executed state, events or a physical clock.
            # Append the accurate mode without relabelling historical records.
            ledger["binding_records"].append({
                "candidate_id": None if outcome.execution_binding is None else outcome.execution_binding.get("candidate_id"),
                "mode": "initial" if previous is None else
                        "planning_retry" if continuation is None and rebind is None else
                        "continuation" if rebind is None else
                        rebind.get("kind") if isinstance(rebind, dict) else "invalid_rebind",
                "source": deepcopy(rebind.get("source")) if isinstance(rebind, dict) else None,
                "request_time_origin_s": (None if continuation is not None and prior is None else
                                           0. if prior is None else prior["request_time_origin_s"]),
                "accepted": outcome.success, "failed_stage": diagnostics.get("failed_stage"),
                "failure": diagnostics.get("failure"), "reason": diagnostics.get("reason"),
                "consumed_planning": deepcopy(ledger["consumed_planning"])})
            # END the mode describes this attempt, not all earlier request work.
            outcome.request_ledger = deepcopy(ledger)
            if outcome.execution_binding is not None:
                outcome.execution_binding["consumed_planning"] = deepcopy(ledger["consumed_planning"])
                outcome.execution_binding["request_ledger"] = deepcopy(ledger)
# END every admitted attempt retains its work even when no curve was produced.


# END complete-window robot planning and generation-time process constraints.
