"""Robot-independent requirements for one complete current skill.

All poses use column vectors, metres, and wxyz quaternions. Progress indexes
reference intervals, not time: phase i owns [i, i+1], including both boundary states.
The numerical allowance below is reported separately from a skill's permitted
motion. Nothing in this module declares an observed grasp or executes a robot.
Torch is imported only when differentiable evaluation is requested.
"""

# BEGIN shared skill-process constraints for generation and verification.
from __future__ import annotations

from dataclasses import dataclass, field, replace
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from .motion import MotionModel, _inverse, _pose_array, _poses_array


def _tensor(value, like):
    # BEGIN only optimizer scopes retain host constants for graph replay.
    from .optimizer_device_checks import constant_tensor
    return constant_tensor(value, like)
    # END tensor inputs and all numeric conversions retain their original path.


def quaternion_conjugate(q):
    """Conjugate unit wxyz quaternions without detaching their gradients."""
    return q * _tensor([1., -1., -1., -1.], q)


def quaternion_multiply(a, b):
    """Hamilton product with ordinary Torch broadcasting."""
    import torch
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    av, bv = torch.broadcast_tensors(av, bv)
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv)), -1)


def quaternion_rotate(q, points):
    """Rotate vectors; a further point-set axis may be present before XYZ."""
    import torch
    if points.ndim == q.ndim + 1:
        q = q.unsqueeze(-2)
    v, points = torch.broadcast_tensors(q[..., 1:], points)
    cross = torch.linalg.cross(v, points)
    return points + 2 * (q[..., :1] * cross + torch.linalg.cross(v, cross))


def quaternion_rotvec(q):
    """Principal rotation vector with a finite derivative at the identity."""
    import torch
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-15)
    q = torch.where(q[..., :1] < 0, -q, q)
    length = torch.sqrt((q[..., 1:] ** 2).sum(-1, keepdim=True) + 1e-24)
    return q[..., 1:] * (2 * torch.atan2(length, q[..., :1]) / length)


def _quaternion(pose):
    return Rotation.from_matrix(np.asarray(pose)[..., :3, :3]).as_quat()[..., [3, 0, 1, 2]]


def _slerp(a, b, fraction):
    import torch
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(max=1. - 1e-7)
    angle = torch.acos(dot)
    fraction = fraction[..., None]
    spherical = (torch.sin((1-fraction)*angle)*a + torch.sin(fraction*angle)*b) / torch.sin(angle)
    linear = (1-fraction)*a + fraction*b
    result = torch.where(dot > .9995, linear, spherical)
    return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(1e-15)


def _interpolate(knots, values, progress):
    """Piecewise linear scalar/vector interpolation, preserving progress grad."""
    import torch
    keys, data = _tensor(knots, progress), _tensor(values, progress)
    if len(keys) == 1:
        return data[0].expand(*progress.shape, *data.shape[1:])
    index = torch.searchsorted(keys, progress.contiguous(), right=True).clamp(1, len(keys)-1)
    fraction = ((progress-keys[index-1])/(keys[index]-keys[index-1])).clamp(0, 1)
    extra = (None,) * (data.ndim-1)
    return data[index-1] + fraction[(...,)+extra] * (data[index]-data[index-1])


@dataclass
class PosePath:
    """A current-scene reference, optionally parameterized by mechanism FK.

    A mechanism path interpolates its signed/unwrapped coordinates and then
    evaluates FK. It never interpolates a hinge as Cartesian chord segments.
    ``attachment`` maps the path body to the requested tool, if applicable.
    """
    progress: np.ndarray
    poses: np.ndarray
    source: object
    motion_model: MotionModel | None = None
    coordinates: np.ndarray | None = None
    attachment: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self):
        self.progress = np.asarray(self.progress, float).copy()
        self.poses = _poses_array(self.poses, "reference poses")
        self.attachment = _pose_array(self.attachment, "attachment")
        if (self.progress.shape != (len(self.poses),) or not np.isfinite(self.progress).all()
                or np.any(np.diff(self.progress) <= 0) or self.progress[0] != 0
                or (len(self.progress) > 1 and self.progress[-1] != 1) or not self.source):
            raise ValueError("A pose path needs evidenced, ordered progress covering [0,1]")
        if self.motion_model is not None:
            if self.motion_model.kind == "free":
                raise ValueError("Use pose interpolation for free motion, not principal rotvec coordinates")
            self.coordinates = np.asarray(self.coordinates, float).copy()
            if (self.coordinates.shape != (len(self.poses), self.motion_model.dimension)
                    or not np.isfinite(self.coordinates).all()
                    or not np.allclose(self.motion_model.poses(self.coordinates), self.poses, atol=1e-7, rtol=0)):
                raise ValueError("Mechanism coordinates must reproduce the supplied body poses")
        elif self.coordinates is not None:
            raise ValueError("Coordinates require a current mechanism model")
        self._quaternions = _quaternion(self.poses)

    def sample(self, progress):
        """Return position (...,3), quaternion (...,4), on the query device."""
        import torch
        if self.motion_model is None:
            position = _interpolate(self.progress, self.poses[:, :3, 3], progress)
            keys = _tensor(self.progress, progress)
            values = _tensor(self._quaternions, progress)
            if len(keys) == 1:
                quaternion = values[0].expand(*progress.shape, 4)
            else:
                index = torch.searchsorted(keys, progress.contiguous(), right=True).clamp(1, len(keys)-1)
                fraction = ((progress-keys[index-1])/(keys[index]-keys[index-1])).clamp(0, 1)
                quaternion = _slerp(values[index-1], values[index], fraction)
        else:
            model = self.motion_model
            coordinate = _interpolate(self.progress, self.coordinates, progress)
            base_p, base_q = _tensor(model.reference_pose[:3, 3], progress), _tensor(_quaternion(model.reference_pose), progress)
            axis = _tensor(model.axis, progress)
            if model.kind in {"revolute", "planar"}:
                angle = coordinate[..., 0] if model.kind == "revolute" else coordinate[..., 2]
                half = angle[..., None]/2
                turn = torch.cat((torch.cos(half), torch.sin(half)*axis), -1)
                quaternion = quaternion_multiply(turn, base_q)
            else:
                quaternion = base_q.expand(*progress.shape, 4)
            if model.kind == "revolute":
                center = _tensor(model.axis_point, progress)
                position = center + quaternion_rotate(turn, (base_p-center).expand(*progress.shape, 3))
            elif model.kind == "prismatic":
                position = base_p + coordinate*axis
            else:
                position = base_p + coordinate[..., :2] @ _tensor(model.basis[:, :2].T, progress)
        offset = _tensor(self.attachment[:3, 3], progress).expand(*progress.shape, 3)
        return (position + quaternion_rotate(quaternion, offset),
                quaternion_multiply(quaternion, _tensor(_quaternion(self.attachment), progress)))

    def sample_numpy(self, progress):
        import torch
        values = torch.as_tensor(np.asarray(progress), dtype=torch.float64)
        position, quaternion = self.sample(values)
        output = np.broadcast_to(np.eye(4), (*values.shape, 4, 4)).copy()
        output[..., :3, 3] = position.detach().numpy()
        output[..., :3, :3] = Rotation.from_quat(
            quaternion.detach().numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(*values.shape, 3, 3)
        return output


@dataclass(frozen=True)
class NumericalTolerance:
    """Solver acceptance errors; these are never inferred skill freedom.

    Pose intervals use target-local XYZ and rotation-vector components, so
    ``position_m`` and ``rotation_rad`` apply to each component of those
    intervals. Reports expose both component maxima and vector norms; a norm
    must not be compared as if it were this componentwise allowance.
    """
    position_m: float = 1e-4
    rotation_rad: float = 1e-3
    joint: float = 1e-5
    velocity: float = 1e-4

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("Numerical tolerances must be finite and positive")


# BEGIN ordinary phases carry reference data without imposing robot constraints.
@dataclass
class MotionPhase:
    """One reference/interaction interval; neither its nodes nor end are goals."""
    name: str
    reference: PosePath
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A reference phase needs a nonempty name")


@dataclass
class PoseProcessRequirement:
    """An explicitly required pose relation on an independent progress scope.

    Bounds use target-local XYZ/rotation-vector components. ``source`` must
    identify the task/model evidence requiring this relation, rather than the
    mere existence of a demonstration path. Ordinary reference phases never
    create this requirement. Its path is a relation target, not a joint path.
    """
    start: float
    end: float
    reference: PosePath
    pose_bounds: np.ndarray
    source: object

    def __post_init__(self):
        self.pose_bounds = np.asarray(self.pose_bounds, float).copy()
        if (not self.source or not 0 <= self.start < self.end < np.inf
                or self.pose_bounds.shape != (6, 2) or np.isnan(self.pose_bounds).any()
                or np.any(self.pose_bounds[:, 0] > self.pose_bounds[:, 1])
                or np.isposinf(self.pose_bounds[:, 0]).any()
                or np.isneginf(self.pose_bounds[:, 1]).any()):
            raise ValueError("A required pose relation needs evidenced scope and valid six-axis bounds")

    def error(self, position, quaternion, progress):
        local = ((progress-self.start)/(self.end-self.start)).clamp(0, 1)
        return _pose_error(position, quaternion, *self.reference.sample(local))
# END ordinary phases carry reference data without imposing robot constraints.


# BEGIN explicit physical-time relations and occurrence-aware evidence identities.
class TimeBindingError(ValueError):
    """An explicit event-time relation cannot be resolved from its own evidence."""


def event_occurrence_key(event):
    """Identify one occurrence; original event names alone never identify a retry."""
    get = event.get if isinstance(event, dict) else lambda key: getattr(event, key)
    identity = (get("skill_id"), get("event_id"), get("occurrence"))
    if (any(not isinstance(value, str) or not value for value in identity[:2])
            or type(identity[2]) is not int or identity[2] < 0):
        raise TimeBindingError("Event evidence requires skill_id, event_id and nonnegative occurrence")
    return identity


@dataclass(frozen=True)
class EventTimeConstraint:
    """An evidenced window in the original request's physical seconds.

    Offsets are relative to exactly one explicit origin. A goal arrival and
    an observed effect name their separate evidence targets. Planning wall
    time and demonstration timestamps are never substituted for physics time.
    """
    check_target: str
    window_s: tuple[float | None, float | None]
    time_basis: str
    source: object
    origin_time_s: float | None = None
    origin_event: dict | None = None
    goal_id: tuple[str, int] | None = None
    condition_id: str | None = None

    def __post_init__(self):
        window = tuple(self.window_s)
        if (self.check_target not in {"activation", "arrival", "effect"}
                or self.time_basis != "request_physical" or not self.source
                or len(window) != 2 or all(value is None for value in window)
                or any(value is not None and not np.isfinite(value) for value in window)
                or (None not in window and window[0] > window[1])):
            raise TimeBindingError("A time requirement needs an explicit physical-time window and source")
        if (self.origin_time_s is None) == (self.origin_event is None):
            raise TimeBindingError("A time requirement needs exactly one explicit origin")
        if self.origin_time_s is not None and not np.isfinite(self.origin_time_s):
            raise TimeBindingError("Physical origin time must be finite")
        if self.origin_event is not None:
            event_occurrence_key(self.origin_event)
            if self.origin_event.get("check_target") not in {"activation", "arrival", "effect"}:
                raise TimeBindingError("Origin event must identify its witnessed check target")
        if self.check_target == "arrival":
            if (self.goal_id is None or len(self.goal_id) != 2
                    or self.goal_id[0] not in {"pose_events", "coordinate_events", "supports"}
                    or type(self.goal_id[1]) is not int or self.goal_id[1] < 0):
                raise TimeBindingError("An arrival window needs the original goal collection and index")
            object.__setattr__(self, "goal_id", tuple(self.goal_id))
        if self.check_target == "effect" and (not isinstance(self.condition_id, str) or not self.condition_id):
            raise TimeBindingError("An effect window needs its independently evaluated condition ID")
        object.__setattr__(self, "window_s", window)
        object.__setattr__(self, "origin_event", deepcopy(self.origin_event))


def resolve_event_window(constraint, history, *, predicted_origins=()):
    """Resolve an absolute window; only actual T witnesses can be past origins.

    Predictions are accepted only when the caller explicitly supplies them for
    a future origin in this candidate. They are never appended to real history.
    Repeated observations of the same occurrence use its first actual T
    witness. Retries cannot inherit it because their occurrence is different.
    """
    if isinstance(constraint, dict):
        constraint = EventTimeConstraint(**constraint)
    origin = constraint.origin_time_s
    if origin is None:
        key = event_occurrence_key(constraint.origin_event)
        target = constraint.origin_event["check_target"]
        def selected(row):
            return (event_occurrence_key(row) == key and row.get("check_target") == target
                    and ("goal_id" not in constraint.origin_event or tuple(row.get("goal_id") or ())
                         == tuple(constraint.origin_event["goal_id"]))
                    and ("condition_id" not in constraint.origin_event
                         or row.get("condition_id") == constraint.origin_event["condition_id"]))
        witnesses = [row for row in history if selected(row) and row.get("value") == "T"]
        identities = {(str(row.get("goal_id")), row.get("condition_id")) for row in witnesses}
        if len(identities) > 1:
            raise TimeBindingError("An ambiguous origin must name its goal_id or condition_id")
        values = [row["time_s"] for row in witnesses]
        if not values:
            values = [row["time_s"] for row in predicted_origins
                      if selected(row)]
        if not values or not np.isfinite(values).all():
            raise TimeBindingError("Time origin lacks one unambiguous actual witness or explicit candidate prediction")
        origin = float(min(values))
    return tuple(None if value is None else float(origin)+value for value in constraint.window_s)
# END explicit physical-time relations and occurrence-aware evidence identities.


@dataclass(frozen=True)
class PlanningEvent:
    """Original source identity plus a unique skill-instance location.

    A verification barrier grants no future evidence. If it needs a stop,
    zero joint velocity is part of the generated trajectory at this anchor.
    """
    event_id: str
    skill_id: str
    progress: float
    source_frame: int
    kind: str
    source: object
    requires_evidence: str | None = None
    stop: bool = False
    # BEGIN a retry is a new occurrence of the original event, with original deadlines.
    occurrence: int = 0
    timing: tuple[EventTimeConstraint, ...] = ()

    def __post_init__(self):
        event_occurrence_key(self)
        object.__setattr__(self, "timing", tuple(
            value if isinstance(value, EventTimeConstraint) else EventTimeConstraint(**value)
            for value in self.timing))
    # END source identity and evidence history survive binding changes.


@dataclass
class JointCoupling:
    """Actual actuator coupling A q = b; not an inferred finger symmetry."""
    joint_names: tuple[str, ...]
    matrix: np.ndarray
    target: np.ndarray
    source: object

    def __post_init__(self):
        self.matrix, self.target = np.asarray(self.matrix, float), np.asarray(self.target, float)
        if (self.matrix.ndim != 2 or self.matrix.shape != (len(self.target), len(self.joint_names))
                or not self.source or not np.isfinite(self.matrix).all() or not np.isfinite(self.target).all()):
            raise ValueError("Joint coupling requires finite named coefficients and actuator evidence")


def _pose_error(position, quaternion, target_position, target_quaternion):
    import torch
    inverse = quaternion_conjugate(target_quaternion)
    return torch.cat((quaternion_rotate(inverse, position-target_position),
                      quaternion_rotvec(quaternion_multiply(inverse, quaternion))), -1)


# BEGIN point events use one actual arrival, even when progress has a plateau.
def _first_event_query(progress, anchor, trajectory_time=None):
    """Select exactly one earliest event query, never an entire progress plateau.

    Complete rollout callers append exact anchors after the regular samples,
    so they provide actual query times. Ordered CPU/export queries may omit
    times. Duplicate representations of the same physical instant count once.
    """
    import torch
    matches = (progress-anchor).abs() <= 1e-7
    if trajectory_time is not None:
        if trajectory_time.shape != progress.shape:
            raise ValueError("Event query times must align with the complete progress horizon")
        earliest = torch.where(matches, trajectory_time, float("inf")).amin(-1, keepdim=True)
        matches = matches & (trajectory_time == earliest)
    first = matches.to(torch.int64).argmax(-1, keepdim=True)
    indices = torch.arange(progress.shape[-1], device=progress.device)
    return matches & (indices == first)
# END separate process requirements alone impose a maintained relation.


def anchor_pose(position, quaternion, progress, anchor):
    """Read an exact anchor from the full horizon, allowing duplicate queries.

    The backend appends actual FK at each event before calling constraints.
    Missing anchors deliberately produce nonfinite values, never a nearest
    sample surrogate. Outer validation must reject nonfinite solver results.
    """
    import torch
    # BEGIN first-arrival event semantics on an ordered progress plateau.
    # s can pause while redundant joints move. Averaging all matching poses
    # could invent an event pose that never occurred; use its first actual
    # arrival, consistently with release geometry and the execution barrier.
    matches = (progress-anchor).abs() <= 1e-7
    index = matches.to(torch.int64).argmax(-1)
    p = torch.gather(position, -2, index[..., None, None].expand(*index.shape, 1, 3))
    q = torch.gather(quaternion, -2, index[..., None, None].expand(*index.shape, 1, 4))
    found = matches.any(-1)[..., None, None]
    p = torch.where(found, p, torch.full_like(p, float("nan")))
    q = torch.where(found, q, torch.full_like(q, float("nan")))
    return p, q
    # END first-arrival event semantics on an ordered progress plateau.


def _compose(p, q, transform):
    t = _tensor(transform[:3, 3], p).expand_as(p)
    return p+quaternion_rotate(q, t), quaternion_multiply(q, _tensor(_quaternion(transform), q))


# BEGIN preserve the current mechanism's original full-body compatibility metric.
def _mechanism_errors(model, tool_from_carrier, position, quaternion):
    """Differentiable counterpart of MotionModel.coordinates_of's pose check.

    Project the actual path body onto the same physical model, then measure
    position norm and SO(3) angle. The tool offset is removed before projecting:
    a small tool error does not establish compatibility of an attached body.
    This pointwise membership check does not recover winding. The independent
    mechanism requirement lifts the ordered actual candidate coordinates.
    """
    import torch
    p, q = _compose(position, quaternion, tool_from_carrier)
    base_p = _tensor(model.reference_pose[:3, 3], p)
    base_q = _tensor(_quaternion(model.reference_pose), q)
    axis = _tensor(model.axis, p)
    projected_q = base_q.expand_as(q)
    if model.kind in {"revolute", "planar"}:
        relative = quaternion_multiply(q, quaternion_conjugate(base_q))
        w, along = relative[..., 0], (relative[..., 1:]*axis).sum(-1)
        sine, cosine = 2*w*along, w.square()-along.square()
        # These are exactly MotionModel._angles's skew/trace projection.
        # A half-turn about a perpendicular axis has no unique projected turn.
        # Choose zero only at machine precision, before atan2, to avoid an
        # undefined derivative; its actual SO(3) residual remains pi and fails
        # the original model tolerance, rather than becoming a zero quaternion.
        singular = w.square()+along.square() <= torch.finfo(q.dtype).eps**2
        angle = torch.atan2(torch.where(singular, torch.zeros_like(sine), sine),
                            torch.where(singular, torch.ones_like(cosine), cosine))
        half = angle[..., None]/2
        turn = torch.cat((torch.cos(half), torch.sin(half)*axis), -1)
        projected_q = quaternion_multiply(turn, base_q)
    if model.kind == "revolute":
        center = _tensor(model.axis_point, p)
        projected_p = center+quaternion_rotate(turn, (base_p-center).expand_as(p))
    elif model.kind == "prismatic":
        projected_p = base_p+((p-base_p)*axis).sum(-1, keepdim=True)*axis
    else:
        basis = _tensor(model.basis[:, :2], p)
        projected_p = base_p+((p-base_p) @ basis) @ basis.T
    error = _pose_error(p, q, projected_p, projected_q)
    return (torch.linalg.vector_norm(error[..., :3], dim=-1),
            torch.linalg.vector_norm(error[..., 3:], dim=-1))
# END preserve the current mechanism's original full-body compatibility metric.


# BEGIN independently scoped current mechanism and lifted coordinate checks.
@dataclass(frozen=True)
class CoordinateDirectionRequirement:
    """An evidenced signed change, monotonicity, or physical coordinate rate.

    ``minimum`` is a displacement for ``net`` and a rate for ``local_rate``;
    ``monotone`` uses zero and permits only the stated displacement tolerance.
    No direction is inferred from the demonstration's coordinate sequence.
    """
    start: float
    end: float
    coordinate_index: int
    kind: str
    sign: int
    minimum: float
    tolerance: float
    source: object
    origin_coordinate: float | None = None
    origin_source: object = None

    def __post_init__(self):
        if (not self.source or not 0 <= self.start < self.end < np.inf
                or type(self.coordinate_index) is not int or self.coordinate_index < 0
                or self.kind not in {"net", "monotone", "local_rate"}
                or type(self.sign) is not int or self.sign not in {-1, 1}
                or not 0 <= self.minimum < np.inf or not 0 <= self.tolerance < np.inf
                or (self.kind == "monotone" and self.minimum != 0)):
            raise ValueError("Coordinate direction needs an evidenced scope, signed kind and finite threshold")
        if ((self.origin_coordinate is None) != (self.origin_source is None)
                or (self.origin_coordinate is not None and
                    (self.kind != "net" or not np.isfinite(self.origin_coordinate) or not self.origin_source))):
            raise ValueError("A net direction origin needs its actual coordinate and evidence together")


@dataclass
class MechanismRequirement:
    """Current physical manifold/limits, independent of the seed reference.

    Angular coordinates are lifted from the ordered candidate trajectory, with
    the actual initial coordinate fixing its branch. A single state can check
    manifold membership and principal-angle feasibility only; it cannot certify
    winding. Final verification must include the complete scoped trajectory.
    """
    start: float
    end: float
    model: MotionModel
    tool_from_carrier: np.ndarray
    coordinate_limits: np.ndarray | None
    initial_coordinates: np.ndarray
    source: object
    directions: tuple[CoordinateDirectionRequirement, ...] = ()

    def __post_init__(self):
        self.directions = tuple(self.directions)
        self.tool_from_carrier = _pose_array(self.tool_from_carrier, "T_tool_carrier")
        self.initial_coordinates = np.asarray(self.initial_coordinates, float).copy()
        if (not self.source or not 0 <= self.start <= self.end < np.inf
                or self.model.kind == "free" or self.initial_coordinates.shape != (self.model.dimension,)
                or not np.isfinite(self.initial_coordinates).all()):
            raise ValueError("A mechanism requires current model evidence, scope and initial coordinates")
        if self.coordinate_limits is not None:
            self.coordinate_limits = np.asarray(self.coordinate_limits, float).copy()
            if (self.coordinate_limits.shape != (self.model.dimension, 2)
                    or not np.isfinite(self.coordinate_limits).all()
                    or np.any(self.coordinate_limits[:, 0] > self.coordinate_limits[:, 1])):
                raise ValueError("Current mechanism coordinate limits must be finite ordered bounds")
        if any(not isinstance(value, CoordinateDirectionRequirement)
               or value.coordinate_index >= self.model.dimension
               or value.end > self.end for value in self.directions):
            raise ValueError("Coordinate directions must lie within their current mechanism and coordinate dimension")
        self.directions = tuple(replace(value, origin_coordinate=float(self.initial_coordinates[value.coordinate_index]),
            origin_source=self.source) if value.kind == "net" and value.start == self.start
            and value.origin_coordinate is None else value for value in self.directions)

    @property
    def angular_column(self):
        return 0 if self.model.kind == "revolute" else 2 if self.model.kind == "planar" else None

    def coordinates(self, position, quaternion, progress, *, trajectory_time=None):
        """Recover actual candidate coordinates; no reference values enter the lift."""
        import torch
        p, q = _compose(position, quaternion, self.tool_from_carrier)
        model = self.model
        base_p = _tensor(model.reference_pose[:3, 3], p)
        axis = _tensor(model.axis, p)
        if model.kind == "prismatic":
            return ((p-base_p)*axis).sum(-1, keepdim=True)
        planar = (p-base_p) @ _tensor(model.basis[:, :2], p)
        if model.kind == "translation_plane":
            return planar
        relative = quaternion_multiply(q, quaternion_conjugate(_tensor(_quaternion(model.reference_pose), q)))
        along, w = (relative[..., 1:]*axis).sum(-1), relative[..., 0]
        singular = w.square()+along.square() <= torch.finfo(q.dtype).eps**2
        angle = torch.atan2(torch.where(singular, 0., 2*w*along),
                            torch.where(singular, 1., w.square()-along.square()))
        # Rollout and exact-event queries can be concatenated out of order.
        # Physical time orders motion even when progress is on a plateau.
        # Progress is sufficient only for a local seed query with no clock.
        # BEGIN finite-time validation stays on-device inside optimizer blocks.
        if trajectory_time is not None:
            from .optimizer_device_checks import require_tensor
            message = "Coordinate history needs finite physical times aligned with its queries"
            if trajectory_time.shape != progress.shape:
                raise ValueError(message)
            valid = require_tensor(torch.isfinite(trajectory_time).all(), message)
            if valid is not None:
                trajectory_time = torch.where(valid, trajectory_time, torch.zeros_like(trajectory_time))
        # END valid times retain the original stable sort and cumsum order.
        order = torch.argsort(progress if trajectory_time is None else trajectory_time, dim=-1, stable=True)
        sorted_angle = torch.gather(angle, -1, order)
        sorted_progress = torch.gather(progress, -1, order)
        within = (sorted_progress >= self.start-1e-7) & (sorted_progress <= self.end+1e-7)
        first = within.to(torch.int64).argmax(-1, keepdim=True)
        initial_angle = torch.gather(sorted_angle, -1, first)
        initial = float(self.initial_coordinates[self.angular_column])
        branch = initial_angle + 2*np.pi*torch.round((initial-initial_angle)/(2*np.pi))
        delta = sorted_angle[..., 1:]-sorted_angle[..., :-1]
        increments = torch.atan2(torch.sin(delta), torch.cos(delta))
        cumulative = torch.cat((torch.zeros_like(sorted_angle[..., :1]), torch.cumsum(increments, -1)), -1)
        lifted = branch+cumulative-torch.gather(cumulative, -1, first)
        complete_start = ((sorted_progress-self.start).abs() <= 1e-7).any(-1, keepdim=True)
        # Without actual-start history this is only a local screening query.
        lifted = torch.where(complete_start, lifted, angle.gather(-1, order))
        angle = torch.gather(lifted, -1, torch.argsort(order, dim=-1))
        return angle[..., None] if model.kind == "revolute" else torch.cat((planar, angle[..., None]), -1)

    def errors(self, position, quaternion):
        return _mechanism_errors(self.model, self.tool_from_carrier, position, quaternion)

    def project_residual(self, position, quaternion, progress):
        """Independent manifold equations; the allowed native coordinate stays free."""
        import torch
        p, q = _compose(position, quaternion, self.tool_from_carrier)
        model = self.model
        base_p, base_q = _tensor(model.reference_pose[:3, 3], p), _tensor(_quaternion(model.reference_pose), q)
        basis = _tensor(model.basis, p)
        relative = quaternion_multiply(q, quaternion_conjugate(base_q))
        rotation = quaternion_rotvec(relative)
        if model.kind in {"prismatic", "translation_plane"}:
            translation = (p-base_p) @ (basis[:, :2] if model.kind == "prismatic" else basis[:, 2:])
            return torch.cat((translation, rotation), -1)
        angle = self.coordinates(position, quaternion, progress)[..., self.angular_column]
        turn = torch.cat((torch.cos(angle[..., None]/2),
                          torch.sin(angle[..., None]/2)*basis[:, 2]), -1)
        if model.kind == "revolute":
            center = _tensor(model.axis_point, p)
            translation = p-center-quaternion_rotate(turn, (base_p-center).expand_as(p))
        else:
            translation = (p-base_p) @ basis[:, 2:]
        projected = quaternion_multiply(turn, base_q)
        off_axis = quaternion_rotvec(quaternion_multiply(q, quaternion_conjugate(projected))) @ basis[:, :2]
        return torch.cat((translation, off_axis), -1)

    def history_complete(self, progress):
        return ((progress-self.start).abs() <= 1e-7).any(-1)

    # BEGIN true coordinate rates and path-dependent direction obligations.
    def coordinate_rates(self, quaternion, tool_linear_velocity, tool_angular_velocity):
        """Map actual reference-frame tool twist to native carrier coordinates."""
        import torch
        offset = quaternion_rotate(quaternion, _tensor(self.tool_from_carrier[:3, 3], quaternion).expand_as(tool_linear_velocity))
        linear = tool_linear_velocity+torch.linalg.cross(tool_angular_velocity, offset)
        axis = _tensor(self.model.axis, linear)
        if self.model.kind == "prismatic":
            return (linear*axis).sum(-1, keepdim=True)
        planar = linear @ _tensor(self.model.basis[:, :2], linear)
        if self.model.kind == "translation_plane":
            return planar
        carrier_q = quaternion_multiply(quaternion, _tensor(_quaternion(self.tool_from_carrier), quaternion))
        relative = quaternion_multiply(carrier_q,
            quaternion_conjugate(_tensor(_quaternion(self.model.reference_pose), quaternion)))
        w, vector = relative[..., :1], relative[..., 1:]
        along = (vector*axis).sum(-1, keepdim=True)
        # Differentiate the same projected angle used by coordinates(), not
        # just omega dot axis. They coincide on the exact manifold; this form
        # also handles its explicitly permitted small off-axis pose errors.
        numerator = w*(w*(tool_angular_velocity*axis).sum(-1, keepdim=True)
            +(torch.linalg.cross(tool_angular_velocity, vector)*axis).sum(-1, keepdim=True))
        numerator = numerator+along*(tool_angular_velocity*vector).sum(-1, keepdim=True)
        angular = numerator/(w.square()+along.square()).clamp_min(torch.finfo(w.dtype).eps**2)
        return angular if self.model.kind == "revolute" else torch.cat((planar, angular), -1)

    def direction_residuals(self, position, quaternion, progress, *, trajectory_time,
                            tool_linear_velocity=None, tool_angular_velocity=None, mode="trajectory",
                            use_tolerance=True):
        """Check finite ordered queries; only final curves use instantaneous rates.

        A geometric OMPL path uses physical-time secants solely to screen a
        seed. Its downstream cubic must still pass true J(q) dq rate checks.
        Single-state seed membership deliberately proves no path direction.
        """
        import torch
        if mode not in {"trajectory", "path", "point"}:
            raise ValueError("Unknown coordinate direction query mode")
        if not self.directions or mode == "point":
            return {}
        if trajectory_time is None or trajectory_time.shape != progress.shape:
            raise ValueError("Coordinate direction checks need the actual physical-time queries")
        order = torch.argsort(trajectory_time, dim=-1, stable=True)
        restore = torch.argsort(order, dim=-1)
        times = torch.gather(trajectory_time, -1, order)
        scoped_progress = torch.gather(progress, -1, order)
        coordinates = self.coordinates(position, quaternion, progress, trajectory_time=trajectory_time)
        coordinates = torch.gather(coordinates, -2, order[..., None].expand_as(coordinates))
        rates = None
        if any(value.kind == "local_rate" for value in self.directions) and mode == "trajectory":
            if tool_linear_velocity is None or tool_angular_velocity is None:
                raise ValueError("Local coordinate rate requires the actual tool twist from J(q) dq")
            rates = self.coordinate_rates(quaternion, tool_linear_velocity, tool_angular_velocity)
            rates = torch.gather(rates, -2, order[..., None].expand_as(rates))
        result = {}
        for index, direction in enumerate(self.directions):
            tolerance = direction.tolerance if use_tolerance else 0.
            active = (scoped_progress >= max(direction.start, self.start)-1e-7) & (scoped_progress <= direction.end+1e-7)
            values = coordinates[..., direction.coordinate_index]
            violation = torch.zeros_like(values)
            if direction.kind == "net":
                at_start = (scoped_progress-direction.start).abs() <= 1e-7
                at_end = (scoped_progress-direction.end).abs() <= 1e-7
                first = at_start.to(torch.int64).argmax(-1, keepdim=True)
                last = progress.shape[-1]-1-torch.flip(at_end, [-1]).to(torch.int64).argmax(-1, keepdim=True)
                origin = values.gather(-1, first) if direction.origin_coordinate is None else direction.origin_coordinate
                change = direction.sign*(values.gather(-1, last)-origin)
                error = torch.relu(direction.minimum-change-tolerance)
                complete = at_end.any(-1, keepdim=True) & (at_start.any(-1, keepdim=True)
                    if direction.origin_coordinate is None else torch.ones_like(at_start[..., :1]))
                error = torch.where(complete, error, float("inf"))
                # A continuation keeps the original requirement identity but
                # does not replay a scope wholly before its actual suffix.
                error = torch.where(scoped_progress.amin(-1, keepdim=True) > direction.end+1e-7, 0., error)
                violation = violation.scatter(-1, last, error)
            elif direction.kind == "local_rate" and rates is not None:
                violation = torch.where(active, torch.relu(direction.minimum-
                    direction.sign*rates[..., direction.coordinate_index]-tolerance), 0.)
            else:
                elapsed = times[..., 1:]-times[..., :-1]
                pair = active[..., 1:] & active[..., :-1] & (elapsed > 0)
                signed = direction.sign*(values[..., 1:]-values[..., :-1])
                if direction.kind == "local_rate":
                    signed = signed/elapsed.clamp_min(torch.finfo(elapsed.dtype).tiny)
                error = torch.relu(direction.minimum-signed-tolerance)
                violation = torch.cat((violation[..., :1], torch.where(pair, error, 0.)), -1)
            result[f"direction:{index}:{direction.kind}"] = torch.gather(violation, -1, restore)
        return result
    # END true coordinate rates and path-dependent direction obligations.
# END independently scoped current mechanism and lifted coordinate checks.


@dataclass
class SupportRequirement:
    """A finite observed support and a specific material face at one event."""
    progress: float
    tool_from_carrier: np.ndarray
    body_points: np.ndarray
    contact_points: np.ndarray
    outward_normal: np.ndarray
    surface: object
    gap_tolerance: float
    normal_tolerance: float
    margin: float
    source: object

    def __post_init__(self):
        from .goals import SupportSurface, _points
        from .regions import _support_arguments, _unit
        self.tool_from_carrier = _pose_array(self.tool_from_carrier, "T_tool_carrier")
        self.body_points = _points(self.body_points, "body points")
        self.contact_points = _points(self.contact_points, "material face points")
        self.outward_normal = _unit(self.outward_normal, "material outward normal")
        _support_arguments(self.surface, self.margin, self.gap_tolerance, self.normal_tolerance)
        if not isinstance(self.surface, SupportSurface) or not self.source:
            raise ValueError("Support constraints need observed finite geometry and a source")

    def evaluate(self, position, quaternion, numerical):
        """Return a nonnegative per-pose violation, preserving tangent/yaw freedom."""
        import torch
        p, q = _compose(position, quaternion, self.tool_from_carrier)
        body = p[..., None, :]+quaternion_rotate(q, _tensor(self.body_points, p).expand(*p.shape[:-1], *self.body_points.shape))
        contact = p[..., None, :]+quaternion_rotate(q, _tensor(self.contact_points, p).expand(*p.shape[:-1], *self.contact_points.shape))
        origin, normal = _tensor(self.surface.origin, p), _tensor(self.surface.normal, p)
        gaps = (contact-origin) @ normal
        height = (body-origin) @ normal
        xy = (body-origin) @ _tensor(self.surface.basis, p)
        equations = _tensor(self.surface._hull.equations, p)
        footprint = (xy @ equations[:, :2].T+equations[:, 2]).amax(dim=(-1, -2))
        outward = quaternion_rotate(q, _tensor(self.outward_normal, q).expand(*q.shape[:-1], 3))
        # Chord distance is monotone in angle on [0, pi] and avoids acos's
        # infinite derivative when a correctly aligned face has dot == 1.
        chord = torch.linalg.vector_norm(outward+normal, dim=-1)
        angular_bound = 2*np.sin(self.normal_tolerance/2)
        return (torch.relu(gaps.abs().amax(-1)-self.gap_tolerance)
                + torch.relu(-height.amin(-1)-numerical.position_m)
                + torch.relu(footprint+self.margin-numerical.position_m)
                + torch.relu(chord-angular_bound))


@dataclass
class PoseEventRequirement:
    """The original full material-frame event, with its own metric thresholds.

    These acceptance thresholds are not inferred freedom for the neighbouring
    motion. In particular, a larger generic solver tolerance never replaces a
    tighter threshold already specified by the source functional event.

    ``actual_relation=None`` explicitly describes a fixed tool calibration
    against a fixed world target. Owner-relative goals instead name the
    currently observed target owner and its material target transform. A held
    function additionally names its owner/material frame and the already bound
    attachment condition. The execution consumer must obtain fresh owner poses
    or a currently verified attachment; planning-time poses are not observations.
    """
    progress: float
    tool_from_function: np.ndarray
    target: np.ndarray
    position_tolerance_m: float
    rotation_tolerance_rad: float
    source: object
    # BEGIN explicit execution geometry; provenance remains opaque evidence.
    actual_relation: dict | None = None
    # END no runtime owner relation is inferred from an event name or source.

    def __post_init__(self):
        self.tool_from_function = _pose_array(self.tool_from_function, "T_tool_function")
        self.target = _pose_array(self.target, "functional target")
        if (not self.source or not np.isfinite(self.progress)
                or not 0 < self.position_tolerance_m < np.inf
                or not 0 < self.rotation_tolerance_rad < np.pi):
            raise ValueError("A material-frame event needs finite metric thresholds and its original source")
        # BEGIN validate each explicitly requested current material-frame input.
        if self.actual_relation is not None:
            relation = self.actual_relation
            allowed = {"function_owner_id", "T_owner_function", "attachment_condition_id",
                       "target_owner_id", "T_owner_target"}
            if not isinstance(relation, dict) or not relation or set(relation)-allowed:
                raise ValueError("actual_relation must declare supported current owner-frame inputs")
            relation = deepcopy(relation)
            for owner_key, pose_key in (("function_owner_id", "T_owner_function"),
                                        ("target_owner_id", "T_owner_target")):
                if (owner_key in relation) != (pose_key in relation):
                    raise ValueError("An actual owner relation needs its owner ID and material transform")
                if owner_key in relation:
                    if not isinstance(relation[owner_key], str) or not relation[owner_key].strip():
                        raise ValueError("An actual owner relation needs a nonempty owner ID")
                    relation[pose_key] = _pose_array(relation[pose_key], pose_key)
            condition = relation.get("attachment_condition_id")
            if "attachment_condition_id" in relation and (
                    "function_owner_id" not in relation or not isinstance(condition, str) or not condition.strip()):
                raise ValueError("An actual attachment relation needs its function owner and condition ID")
            self.actual_relation = relation
        # END absent current evidence remains an execution-level unknown.

    def errors(self, position, quaternion):
        import torch
        p, q = _compose(position, quaternion, self.tool_from_function)
        error = _pose_error(p, q, _tensor(self.target[:3, 3], p), _tensor(_quaternion(self.target), q))
        return torch.linalg.vector_norm(error[..., :3], dim=-1), torch.linalg.vector_norm(error[..., 3:], dim=-1)

    def evaluate(self, position, quaternion):
        import torch
        p, r = self.errors(position, quaternion)
        return torch.relu(p-self.position_tolerance_m)+torch.relu(r-self.rotation_tolerance_rad)


# BEGIN native goals compare lifted candidate coordinates, including winding.
@dataclass
class CoordinateEventRequirement:
    """A signed metric/unwrapped native event in one current mechanism."""
    progress: float
    mechanism: MechanismRequirement
    target_coordinate: float
    tolerance: float
    source: object
    coordinate_mode: str = "absolute"
    coordinate_origin: float | None = None

    def __post_init__(self):
        if (self.mechanism.model.kind not in {"prismatic", "revolute"} or not self.source
                or not self.mechanism.start <= self.progress <= self.mechanism.end
                or not np.isfinite(self.target_coordinate) or not 0 < self.tolerance < np.inf):
                raise ValueError("A native event needs a current mechanism, evidenced target and threshold")
        if (self.coordinate_mode not in {"absolute", "entry_delta"}
                or (self.coordinate_mode == "absolute" and self.coordinate_origin is not None)
                or (self.coordinate_mode == "entry_delta" and (self.coordinate_origin is None
                    or not np.isfinite(self.coordinate_origin)))):
            raise ValueError("An entry-delta goal needs its explicitly preserved original entry coordinate")

    @property
    def absolute_target(self):
        return self.target_coordinate if self.coordinate_mode == "absolute" else self.coordinate_origin+self.target_coordinate

    @property
    def kind(self):
        return self.mechanism.model.kind

    def error(self, position, quaternion, progress, *, trajectory_time=None):
        import torch
        coordinate = self.mechanism.coordinates(position, quaternion, progress, trajectory_time=trajectory_time)[..., 0]
        difference = coordinate-self.absolute_target
        if self.kind == "revolute":
            local = torch.atan2(torch.sin(difference), torch.cos(difference))
            difference = torch.where(self.mechanism.history_complete(progress)[..., None], difference, local)
        return difference.abs()

    def evaluate(self, position, quaternion, progress, *, trajectory_time=None):
        import torch
        return torch.relu(self.error(position, quaternion, progress, trajectory_time=trajectory_time)-self.tolerance)
# END native goals compare lifted candidate coordinates, including winding.


@dataclass
class InteriorRequirement:
    """All material body samples remain inside evidenced convex free geometry."""
    start: float
    end: float
    tool_from_carrier: np.ndarray
    body_points: np.ndarray
    halfspaces: np.ndarray
    margin: float
    source: object

    def __post_init__(self):
        from .goals import InteriorRegion, _points
        self.tool_from_carrier = _pose_array(self.tool_from_carrier, "T_tool_carrier")
        self.body_points = _points(self.body_points, "body points")
        self.halfspaces = InteriorRegion(self.halfspaces, str(self.source)).halfspaces
        if not self.source or not 0 <= self.start <= self.end or not np.isfinite(self.margin) or self.margin < 0:
            raise ValueError("Interior requirements need a bounded interval, margin, and evidence")

    def evaluate(self, position, quaternion, numerical):
        import torch
        p, q = _compose(position, quaternion, self.tool_from_carrier)
        points = p[..., None, :]+quaternion_rotate(q, _tensor(self.body_points, p).expand(*p.shape[:-1], *self.body_points.shape))
        h = _tensor(self.halfspaces, p)
        error = (points @ h[:, :3].T+h[:, 3]+self.margin).amax(dim=(-1, -2))
        return torch.relu(error-numerical.position_m)


# BEGIN one current skill with separate goals, process relations and references.
@dataclass
class PlanningProblem:
    """One current skill; source timing and real requirements have separate owners.

    Pose paths initialize the search and support deviation reports. Only the
    explicit goal/process collections below constrain generated robot motion.
    ``source_timestamps`` and ``source_progress`` retain the bound source map;
    the planner may choose execution timing once and freeze that binding.
    """
    joint_names: tuple[str, ...]
    start_q: np.ndarray
    start_dq: np.ndarray
    tool_link: str
    reference_frame: str
    base_pose: np.ndarray
    tool_transform: np.ndarray
    phases: tuple[MotionPhase, ...]
    events: tuple[PlanningEvent, ...] = ()
    couplings: tuple[JointCoupling, ...] = ()
    numerical: NumericalTolerance = field(default_factory=NumericalTolerance)
    conditions: tuple[dict, ...] = ()
    supports: tuple[SupportRequirement, ...] = ()
    interiors: tuple[InteriorRequirement, ...] = ()
    pose_events: tuple[PoseEventRequirement, ...] = ()
    coordinate_events: tuple[CoordinateEventRequirement, ...] = ()
    pose_relations: tuple[PoseProcessRequirement, ...] = ()
    mechanisms: tuple[MechanismRequirement, ...] = ()
    source_timestamps: np.ndarray | None = None
    source_progress: np.ndarray | None = None
    start_progress: float = 0.

    def __post_init__(self):
        self.joint_names = tuple(self.joint_names)
        for name in ("phases", "events", "couplings", "conditions", "supports", "interiors",
                     "pose_events", "coordinate_events", "pose_relations", "mechanisms"):
            setattr(self, name, tuple(getattr(self, name)))
        self.start_q, self.start_dq = np.asarray(self.start_q, float).copy(), np.asarray(self.start_dq, float).copy()
        if (not self.joint_names or len(set(self.joint_names)) != len(self.joint_names)
                or any(not isinstance(n, str) or not n for n in self.joint_names)
                or self.start_q.shape != (len(self.joint_names),) or self.start_dq.shape != self.start_q.shape
                or not np.isfinite(self.start_q).all() or not np.isfinite(self.start_dq).all()
                or not self.phases or not self.tool_link or not self.reference_frame):
            raise ValueError("A planning problem needs current named joint state, calibrated frames and references")
        self.base_pose = _pose_array(self.base_pose, "T_reference_robot_base")
        self.tool_transform = _pose_array(self.tool_transform, "T_link_tool")
        if not np.isfinite(self.start_progress) or not 0 <= self.start_progress < self.phase_count:
            raise ValueError("Actual entry progress must lie in the remaining original skill interval")
        skill_ids = [phase.metadata["skill_id"] for phase in self.phases if "skill_id" in phase.metadata]
        skill_ids.extend(event.skill_id for event in self.events)
        if any(not isinstance(value, str) or not value for value in skill_ids) or len(set(skill_ids)) > 1:
            raise ValueError("A planning problem contains exactly one current skill identity")
        identities = set()
        for event in self.events:
            identity = (event.skill_id, event.event_id)
            if (identity in identities or not event.event_id or not event.skill_id or not event.source
                    or not np.isfinite(event.progress) or not 0 <= event.progress <= self.phase_count
                    or type(event.source_frame) is not int or event.source_frame < 0):
                raise ValueError("Events need unique IDs, source evidence and exact original frame/progress")
            identities.add(identity)
        for requirement in (*self.pose_relations, *self.mechanisms, *self.interiors):
            if requirement.end > self.phase_count:
                raise ValueError("A process requirement extends beyond the current skill")
        for requirement in (*self.pose_events, *self.coordinate_events, *self.supports):
            if not 0 <= requirement.progress <= self.phase_count:
                raise ValueError("A functional event lies outside the current skill")
        # A resumed angular event must use the same newly observed coordinate
        # origin as its manifold/limit checks. Keeping a stale event mechanism
        # would silently reduce its turn goal to a modulo-angle query.
        if any(not any(event.mechanism is mechanism for mechanism in self.mechanisms)
               for event in self.coordinate_events):
            raise ValueError("Native coordinate goals must reference the independently bound mechanism instance")
        if (self.source_timestamps is None) != (self.source_progress is None):
            raise ValueError("Source timing requires both timestamps and corresponding progress")
        if self.source_timestamps is not None:
            self.source_timestamps = np.asarray(self.source_timestamps, float).copy()
            self.source_progress = np.asarray(self.source_progress, float).copy()
            if (self.source_timestamps.ndim != 1 or len(self.source_timestamps) < 2
                    or self.source_progress.shape != self.source_timestamps.shape
                    or not np.isfinite(self.source_timestamps).all() or not np.isfinite(self.source_progress).all()
                    or np.any(np.diff(self.source_timestamps) <= 0) or np.any(np.diff(self.source_progress) <= 0)
                    or self.source_progress[0] != self.start_progress or self.source_progress[-1] != self.phase_count):
                raise ValueError("The original timestamp/progress map must increase across the complete current skill")

    @property
    def phase_count(self):
        return len(self.phases)

    @property
    def source_time_map(self):
        """Return source seconds and progress; no phase-count clock is invented."""
        if self.source_timestamps is None:
            raise ValueError("The planning problem has no bound source timestamp map")
        return {"source_times": self.source_timestamps.copy(), "progress": self.source_progress.copy()}

    def source_time_to_progress(self, source_time):
        mapping = self.source_time_map
        return np.interp(source_time, mapping["source_times"], mapping["progress"])

    def progress_to_source_time(self, progress):
        mapping = self.source_time_map
        return np.interp(progress, mapping["progress"], mapping["source_times"])

    @property
    def terminal_pose_target(self):
        """A true complete terminal pose goal for native H=1 goal consumers."""
        return next((goal for goal in self.pose_events if goal.progress == self.phase_count), None)

    @property
    def anchors(self):
        """Exact evaluation sites, never fixed IK states or mandatory robot poses."""
        anchors = np.unique([self.start_progress, float(self.phase_count), *(e.progress for e in self.events),
                          *(r.progress for r in self.supports),
                          *(s for r in self.interiors for s in (r.start, r.end)),
                          *(r.progress for r in self.pose_events),
                          *(r.progress for r in self.coordinate_events),
                          *(s for r in self.mechanisms for s in (r.start, r.end)),
                          *(s for r in self.mechanisms for d in r.directions for s in (d.start, d.end)),
                          *(r.start+float(s)*(r.end-r.start)
                            for r in self.pose_relations for s in r.reference.progress)])
        return anchors[anchors >= self.start_progress]

    @property
    def stop_progress(self):
        return np.array([event.progress for event in self.events if event.stop and event.progress >= self.start_progress], float)

    def sample_reference(self, progress):
        import torch
        position, quaternion = self.phases[0].reference.sample(progress.clamp(0, 1))
        for index, phase in enumerate(self.phases[1:], 1):
            p, q = phase.reference.sample((progress-index).clamp(0, 1))
            selected = (progress >= index)[..., None]
            position, quaternion = torch.where(selected, p, position), torch.where(selected, q, quaternion)
        return position, quaternion

    def transform_tool_pose(self, position, quaternion):
        bp, bq = _tensor(self.base_pose[:3, 3], position), _tensor(_quaternion(self.base_pose), quaternion)
        tool_q = quaternion_multiply(bq, quaternion)
        tool_p = bp + quaternion_rotate(bq, position)
        return (tool_p + quaternion_rotate(tool_q, _tensor(self.tool_transform[:3, 3], position).expand_as(position)),
                quaternion_multiply(tool_q, _tensor(_quaternion(self.tool_transform), quaternion)))

    def reference_to_link_pose(self, position, quaternion):
        inverse_tool, inverse_base = _inverse(self.tool_transform), _inverse(self.base_pose)
        link_position = position + quaternion_rotate(quaternion, _tensor(inverse_tool[:3, 3], position).expand_as(position))
        link_quaternion = quaternion_multiply(quaternion, _tensor(_quaternion(inverse_tool), quaternion))
        bq = _tensor(_quaternion(inverse_base), quaternion)
        return (_tensor(inverse_base[:3, 3], position) + quaternion_rotate(bq, link_position),
                quaternion_multiply(bq, link_quaternion))

    def transform_robot_spheres(self, spheres):
        import torch
        centers = quaternion_rotate(_tensor(_quaternion(self.base_pose), spheres), spheres[..., :3])
        return torch.cat((centers + _tensor(self.base_pose[:3, 3], spheres), spheres[..., 3:]), -1)

    # BEGIN preserve physical velocity through the calibrated base/TCP transform.
    def transform_tool_velocity(self, linear, angular, link_quaternion):
        """Transform native base-frame link twist to reference-frame tool twist."""
        import torch
        base = _tensor(_quaternion(self.base_pose), linear)
        world_linear, world_angular = quaternion_rotate(base, linear), quaternion_rotate(base, angular)
        tool_orientation = quaternion_multiply(base, link_quaternion)
        offset = quaternion_rotate(tool_orientation, _tensor(self.tool_transform[:3, 3], linear).expand_as(linear))
        return world_linear+torch.linalg.cross(world_angular, offset), world_angular
    # END preserve physical velocity through the calibrated base/TCP transform.

    def residuals(self, position, quaternion, progress, joint_position=None,
                  joint_names=None, joint_velocity=None, *, trajectory_time=None, point_goal_ids=None,
                  tool_linear_velocity=None, tool_angular_velocity=None, direction_mode="trajectory"):
        """Nonnegative violations of real requirements only, without cancellation."""
        import torch
        if position.shape != (*progress.shape, 3) or quaternion.shape != (*progress.shape, 4):
            raise ValueError("Pose arrays must align with the complete progress horizon")
        result = {"progress_bounds": torch.relu(self.start_progress-progress)+torch.relu(progress-self.phase_count)}
        selected_goals = None if point_goal_ids is None else set(point_goal_ids)
        def event_mask(collection, index, requirement):
            return (_first_event_query(progress, requirement.progress, trajectory_time)
                    if selected_goals is None or (collection, index) in selected_goals else torch.zeros_like(progress, dtype=torch.bool))
        tolerance = _tensor([self.numerical.position_m]*3+[self.numerical.rotation_rad]*3, position)
        for index, requirement in enumerate(self.pose_relations):
            error = requirement.error(position, quaternion, progress)
            bounds = _tensor(requirement.pose_bounds, position)
            violation = (torch.relu(bounds[:, 0]-error-tolerance)+torch.relu(error-bounds[:, 1]-tolerance)).sum(-1)
            mask = (progress >= requirement.start-1e-7) & (progress <= requirement.end+1e-7)
            result[f"pose_relation:{index}"] = torch.where(mask, violation, 0.)
        for index, requirement in enumerate(self.mechanisms):
            model = requirement.model
            p, r = requirement.errors(position, quaternion)
            mask = (progress >= requirement.start-1e-7) & (progress <= requirement.end+1e-7)
            violation = torch.relu(p-model.position_tolerance-1e-10)+torch.relu(r-model.rotation_tolerance-1e-10)
            result[f"mechanism:{index}"] = torch.where(mask, violation, 0.)
            if requirement.coordinate_limits is not None:
                coordinates = requirement.coordinates(position, quaternion, progress, trajectory_time=trajectory_time)
                bounds = _tensor(requirement.coordinate_limits, position)
                excess = torch.relu(bounds[:, 0]-coordinates)+torch.relu(coordinates-bounds[:, 1])
                angular = requirement.angular_column
                if angular is not None:
                    # A local state has no winding history. Check whether any
                    # equivalent angular lift is permitted, leaving actual turn
                    # counting to complete-trajectory evaluation below.
                    angle = coordinates[..., angular]
                    low, high = bounds[angular]
                    branch = torch.ceil((low-angle)/(2*np.pi))
                    allowed = angle+2*np.pi*branch <= high+1e-9
                    local_excess = torch.where(allowed, 0., torch.minimum(
                        torch.remainder(low-angle, 2*np.pi), torch.remainder(angle-high, 2*np.pi)))
                    excess = excess.clone()
                    excess[..., angular] = torch.where(requirement.history_complete(progress)[..., None],
                                                       excess[..., angular], local_excess)
                result[f"mechanism_limits:{index}"] = torch.where(mask, excess.sum(-1), 0.)
            for name, value in requirement.direction_residuals(position, quaternion, progress,
                    trajectory_time=trajectory_time, tool_linear_velocity=tool_linear_velocity,
                    tool_angular_velocity=tool_angular_velocity, mode=direction_mode).items():
                result[f"mechanism:{index}:{name}"] = value
        for index, requirement in enumerate(self.supports):
            # BEGIN evaluate finite support only at its existing unique arrival.
            # event_mask already selects at most one physical query per batch,
            # including unsorted query times and duplicate plateau samples.
            # Gather that pose before expanding the complete body/plane arrays;
            # the output and its gradients still occupy the original horizon.
            mask = event_mask("supports", index, requirement)
            selected = mask.to(torch.int64).argmax(-1, keepdim=True)
            event_position = position.gather(-2, selected[..., None].expand(*selected.shape, 3))
            event_quaternion = quaternion.gather(-2, selected[..., None].expand(*selected.shape, 4))
            value = requirement.evaluate(event_position, event_quaternion, self.numerical)
            value = torch.where(mask.any(-1, keepdim=True), value, 0.)
            # A caller may use a different progress dtype; retain the original
            # support evaluator's pose dtype rather than the clock's dtype.
            result[f"support:{index}"] = torch.zeros_like(progress, dtype=value.dtype).scatter(-1, selected, value)
            # END absent or unselected events retain zero residual; finite
            # pose inputs also retain their zero gradient outside the event.
        for index, requirement in enumerate(self.interiors):
            mask = (progress >= requirement.start-1e-7) & (progress <= requirement.end+1e-7)
            result[f"interior:{index}"] = torch.where(mask, requirement.evaluate(position, quaternion, self.numerical), 0.)
        for index, requirement in enumerate(self.pose_events):
            result[f"function_pose:{index}"] = torch.where(event_mask("pose_events", index, requirement),
                requirement.evaluate(position, quaternion), 0.)
        for index, requirement in enumerate(self.coordinate_events):
            result[f"function_coordinate:{index}"] = torch.where(event_mask("coordinate_events", index, requirement),
                requirement.evaluate(position, quaternion, progress, trajectory_time=trajectory_time), 0.)
        if self.couplings:
            if joint_position is None or joint_names is None:
                raise ValueError("Actuator coupling requires named robot joint positions")
            for index, coupling in enumerate(self.couplings):
                columns = [tuple(joint_names).index(name) for name in coupling.joint_names]
                # BEGIN cache only the original integer advanced-index array.
                from .optimizer_device_checks import constant_tensor
                column_index = constant_tensor(columns, joint_position, dtype=torch.int64)
                error = joint_position[..., column_index] @ _tensor(coupling.matrix.T, joint_position)-_tensor(coupling.target, joint_position)
                # END the matrix multiplication and residual order are unchanged.
                result[f"coupling:{index}"] = torch.relu(error.abs()-self.numerical.joint).sum(-1)
        for event in self.events:
            if event.stop:
                if joint_velocity is None:
                    raise ValueError("An explicitly required stop needs actual joint velocity evaluation")
                result[f"stop:{event.skill_id}:{event.event_id}"] = torch.where(_first_event_query(progress, event.progress, trajectory_time),
                    torch.relu(joint_velocity.abs()-self.numerical.velocity).sum(-1), 0.)
        return result

    def evaluate(self, position, quaternion, progress, joint_position=None,
                 joint_names=None, joint_velocity=None, *, trajectory_time=None, point_goal_ids=None,
                 tool_linear_velocity=None, tool_angular_velocity=None, direction_mode="trajectory"):
        return sum(self.residuals(position, quaternion, progress, joint_position,
                                  joint_names, joint_velocity, trajectory_time=trajectory_time,
                                  point_goal_ids=point_goal_ids, tool_linear_velocity=tool_linear_velocity,
                                  tool_angular_velocity=tool_angular_velocity, direction_mode=direction_mode).values())

    # BEGIN exact signed point inequalities for the functional AL objective.
    @property
    def functional_constraint_keys(self):
        """Stable event/component identities for the current request suffix."""
        return tuple(
            [f"function_pose:{index}:{component}"
             for index, requirement in enumerate(self.pose_events)
             if requirement.progress >= self.start_progress
             for component in ("position", "rotation")]
            + [f"function_coordinate:{index}"
               for index, requirement in enumerate(self.coordinate_events)
               if requirement.progress >= self.start_progress])

    def functional_inequalities(self, position, quaternion, progress, *, trajectory_time=None):
        """Return signed event residuals and their single physical query masks.

        For input progress ``[batch, horizon]``, residual values have shape
        ``[batch]`` and Boolean masks have shape ``[batch, horizon]``. A pose
        event supplies separate position-norm and rotation-norm inequalities;
        a coordinate event uses its existing signed/unwrapped history before
        taking the absolute error. Every value is ``error/original_tolerance-1``.

        The first actual event instant is used even when progress pauses or an
        exact anchor duplicates a regular sample. A missing required event is
        NaN, not a falsely feasible zero. Events before a continued request's
        start are absent from both dictionaries and retain their original IDs.
        """
        import torch
        if position.shape != (*progress.shape, 3) or quaternion.shape != (*progress.shape, 4):
            raise ValueError("Pose arrays must align with the complete progress horizon")
        signed, masks = {}, {}

        def add(key, error, tolerance, mask):
            value = torch.where(mask, error/tolerance-1., 0.).sum(-1)
            signed[key] = torch.where(mask.any(-1), value, float("nan"))
            masks[key] = mask

        for index, requirement in enumerate(self.pose_events):
            if requirement.progress < self.start_progress:
                continue
            mask = _first_event_query(progress, requirement.progress, trajectory_time)
            position_error, rotation_error = requirement.errors(position, quaternion)
            add(f"function_pose:{index}:position", position_error, requirement.position_tolerance_m, mask)
            add(f"function_pose:{index}:rotation", rotation_error, requirement.rotation_tolerance_rad, mask)
        for index, requirement in enumerate(self.coordinate_events):
            if requirement.progress < self.start_progress:
                continue
            mask = _first_event_query(progress, requirement.progress, trajectory_time)
            error = requirement.error(position, quaternion, progress, trajectory_time=trajectory_time)
            add(f"function_coordinate:{index}", error, requirement.tolerance, mask)
        return signed, masks
    # END these normalized optimizer inequalities do not change hard residuals.

    # BEGIN keep optional goal centering separate from functional feasibility.
    def optimizer_preferences(self, position, quaternion, progress, *, trajectory_time=None,
                              tool_linear_velocity=None, tool_angular_velocity=None, phase="quality"):
        """Guide process relations and optional quality-stage goal centering.

        Functional feasibility is supplied only by the signed inequality AL
        terms. ``quality`` additionally prefers the original goal center inside
        the same enforced tolerance. Process equations and directional commands
        retain their existing semantics in both phases; this method never
        changes the independently evaluated acceptance residuals.
        """
        import torch
        if phase not in {"feasibility", "quality"}:
            raise ValueError("Optimization phase must be feasibility or quality")
        result = torch.zeros_like(progress)
        for requirement in self.pose_relations:
            exact = np.isfinite(requirement.pose_bounds).all(axis=1) & (requirement.pose_bounds[:, 0] == requirement.pose_bounds[:, 1])
            if exact.any():
                # BEGIN expand a static host mask before graph capture.
                from .optimizer_device_checks import constant_tensor
                exact_columns = constant_tensor(np.flatnonzero(exact), position, dtype=torch.long)
                error = requirement.error(position, quaternion, progress)[..., exact_columns]
                # END selected coordinates keep the original ascending mask order.
                preference = (error-_tensor(requirement.pose_bounds[exact, 0], position)).square().sum(-1)
                mask = (progress >= requirement.start-1e-7) & (progress <= requirement.end+1e-7)
                result = result+torch.where(mask, preference, 0.)
        for requirement in self.pose_events if phase == "quality" else ():
            p, r = requirement.errors(position, quaternion)
            # BEGIN scale real goal objectives by their declared functional accuracy.
            # Raw metres/radians squared make a precise goal arbitrarily weak
            # beside native smoothness costs. This fixed rule changes numerical
            # conditioning only; residuals and final allowed sets stay in their
            # original units, and free support/region axes gain no target.
            preference = (p/requirement.position_tolerance_m).square()+(r/requirement.rotation_tolerance_rad).square()
            result = result+torch.where(_first_event_query(progress, requirement.progress, trajectory_time), preference, 0.)
            # END goal precision is input semantics, never a per-skill weight.
        for requirement in self.coordinate_events if phase == "quality" else ():
            error = requirement.error(position, quaternion, progress, trajectory_time=trajectory_time)
            # BEGIN use the same relative-accuracy rule for native coordinate goals.
            result = result+torch.where(_first_event_query(progress, requirement.progress, trajectory_time),
                                        (error/requirement.tolerance).square(), 0.)
            # END absolute and entry-relative goals retain their own metric tolerance.
        # BEGIN direction objectives use one semantic scale and the stated threshold.
        # Prefer reaching the declared minimum rather than resting at the
        # outer edge of its tolerated error. This is an optimizer preference;
        # acceptance still uses each original allowance in raw physical units.
        for requirement in self.mechanisms:
            values = requirement.direction_residuals(position, quaternion, progress,
                trajectory_time=trajectory_time, tool_linear_velocity=tool_linear_velocity,
                tool_angular_velocity=tool_angular_velocity, use_tolerance=False)
            for direction, residual in zip(requirement.directions, values.values()):
                if direction.tolerance > 0:
                    scale = direction.tolerance
                elif direction.minimum > 0:
                    scale = direction.minimum
                else:
                    scale = (requirement.model.rotation_tolerance if direction.coordinate_index == requirement.angular_column
                             else requirement.model.position_tolerance)
                    if direction.kind == "local_rate":
                        duration = (trajectory_time.amax(-1, keepdim=True)-trajectory_time.amin(-1, keepdim=True))
                        scale = scale/duration.clamp_min(torch.finfo(duration.dtype).eps)
                result = result+(residual/scale).square()
        # END no per-skill weights or extra hard direction constraints are added.
        return result
    # END functional goal centers cannot compete during the feasibility phase.

    def measurements(self, position, quaternion, progress, *, trajectory_time=None):
        """Report source deviations separately from independent physical/goal errors."""
        import torch
        result = {}
        for index, phase in enumerate(self.phases):
            p, q = phase.reference.sample((progress-index).clamp(0, 1))
            error = _pose_error(position, quaternion, p, q)
            mask = (progress >= index-1e-7) & (progress <= index+1+1e-7)
            prefix = f"phase:{index}:original_reference"
            result[f"{prefix}_position_error_m"] = torch.where(mask, torch.linalg.vector_norm(error[..., :3], dim=-1), 0.)
            result[f"{prefix}_rotation_error_rad"] = torch.where(mask, torch.linalg.vector_norm(error[..., 3:], dim=-1), 0.)
            result[f"{prefix}_position_component_error_m"] = torch.where(mask, error[..., :3].abs().amax(-1), 0.)
            result[f"{prefix}_rotation_component_error_rad"] = torch.where(mask, error[..., 3:].abs().amax(-1), 0.)
        for index, requirement in enumerate(self.mechanisms):
            p, r = requirement.errors(position, quaternion)
            mask = (progress >= requirement.start-1e-7) & (progress <= requirement.end+1e-7)
            result[f"mechanism:{index}:position_error_m"] = torch.where(mask, p, 0.)
            result[f"mechanism:{index}:rotation_error_rad"] = torch.where(mask, r, 0.)
        for index, requirement in enumerate(self.pose_events):
            p, r = requirement.errors(position, quaternion)
            mask = _first_event_query(progress, requirement.progress, trajectory_time)
            result[f"function:{index}:position_error_m"] = torch.where(mask, p, 0.)
            result[f"function:{index}:rotation_error_rad"] = torch.where(mask, r, 0.)
        for index, requirement in enumerate(self.coordinate_events):
            unit = "m" if requirement.kind == "prismatic" else "rad"
            result[f"coordinate:{index}:error_{unit}"] = torch.where(_first_event_query(progress, requirement.progress, trajectory_time),
                requirement.error(position, quaternion, progress, trajectory_time=trajectory_time), 0.)
        return result

    def verification_issues(self, progress):
        """Require actual goal/scope queries and the beginning of angular history.

        Sampling resolution/continuous interval coverage remains the curve
        verifier's responsibility. A point query alone proves no winding.
        """
        import torch
        query = torch.as_tensor(progress)
        issues = []
        for anchor in self.anchors:
            if not ((query-anchor).abs() <= 1e-7).any(-1).all():
                issues.append(f"required query at progress {anchor:g} is absent")
        for index, requirement in enumerate(self.mechanisms):
            if requirement.angular_column is not None and not requirement.history_complete(query).all():
                issues.append(f"mechanism:{index}: angular history does not include the actual start")
        return issues

    # BEGIN expose remaining work without resetting source identity or the clock.
    def remaining_obligations(self, progress, applied_ids=()):
        """Read a suffix of this binding; this does not construct a resumed plan.

        ``applied_ids`` contains (skill_id, event_id, occurrence) triples. An unapplied command
        before ``progress`` remains visible as overdue; elapsed time is not
        evidence that it was applied. Scope intersections retain their original
        requirement objects so target paths are never silently reparameterized.
        Ordinary continuation retains the frozen clock. Explicit rebinding may
        rebuild the schedule while retaining declared absolute time relations.
        """
        if not np.isfinite(progress) or not 0 <= progress <= self.phase_count:
            raise ValueError("Continuation progress must lie in the original current skill")
        applied = set()
        for identity in applied_ids:
            if (not isinstance(identity, (tuple, list)) or len(identity) != 3
                    or any(not isinstance(value, str) or not value for value in identity[:2])
                    or type(identity[2]) is not int or identity[2] < 0):
                raise ValueError("Applied event identities must be (skill_id, event_id, occurrence) triples")
            applied.add(tuple(identity))
        pending = [event for event in self.events if event_occurrence_key(event) not in applied]
        scoped = {}
        for name in ("pose_relations", "mechanisms", "interiors"):
            scoped[name] = [{"requirement": deepcopy(requirement),
                             "remaining_scope": (max(float(progress), requirement.start), requirement.end)}
                            for requirement in getattr(self, name) if requirement.end >= progress]
        actual_inputs = ["synchronized actual joint positions and velocities",
                         "original frozen execution binding, elapsed physical time and event application ledger"]
        for index, requirement in enumerate(self.mechanisms):
            if requirement.end >= progress and requirement.angular_column is not None:
                actual_inputs.append(f"mechanism:{index}: observed current lifted coordinate including executed turns")
        owners = sorted({phase.metadata["carrier_id"] for phase in self.phases
                         if phase.metadata.get("carrier_id") is not None})
        if owners:
            actual_inputs.append("current owner poses and attachment/release facts for: "+", ".join(owners))
        return {
            "progress": float(progress), "original_phase_count": self.phase_count,
            "phases": tuple((index, deepcopy(phase)) for index, phase in enumerate(self.phases)
                            if index+1 >= progress),
            "events": tuple(deepcopy(pending)),
            "overdue_event_ids": tuple(event_occurrence_key(event) for event in pending
                                       if event.progress < progress and event.kind in {"open", "close", "hold"}),
            "goals": {name: tuple(deepcopy(requirement) for requirement in getattr(self, name)
                                   if requirement.progress >= progress)
                      for name in ("pose_events", "coordinate_events", "supports")},
            "past_goal_assessment_required": tuple((name, index) for name in
                ("pose_events", "coordinate_events", "supports") for index, requirement in
                enumerate(getattr(self, name)) if requirement.progress < progress),
            "process": scoped, "conditions": deepcopy(self.conditions),
            "source_time_map": self.source_time_map, "required_actual_inputs": tuple(actual_inputs),
            "clock_policy": "ordinary continuation uses elapsed+t; explicit rebinding preserves declared physical-time relations",
        }
    # END expose remaining work without resetting source identity or the clock.

    # BEGIN projection scopes and true goal sets are shared with the seed planner.
    def projection_requirements(self, progress, side="at"):
        """Select actual equations in a scope; a boundary can be queried one-sided."""
        if side not in {"at", "left", "right"} or not np.isfinite(progress):
            raise ValueError("Projection scope needs finite progress and at/left/right side")
        selected = []
        for collection in ("pose_relations", "mechanisms"):
            for index, requirement in enumerate(getattr(self, collection)):
                active = (requirement.start <= progress <= requirement.end if side == "at" else
                          requirement.start < progress <= requirement.end if side == "left" else
                          requirement.start <= progress < requirement.end)
                if collection == "pose_relations":
                    bounds = requirement.pose_bounds
                    active = active and bool((np.isfinite(bounds[:, 0]) & (bounds[:, 0] == bounds[:, 1])).any())
                if active:
                    selected.append((collection, index))
        return tuple(selected)+tuple(("couplings", index) for index in range(len(self.couplings)))

    def goal_requirements_at(self, progress):
        """Identify original true goals at this progress, never reference endpoints."""
        return tuple((name, index) for name in ("pose_events", "coordinate_events", "supports")
                     for index, goal in enumerate(getattr(self, name)) if abs(goal.progress-progress) <= 1e-7)

    def goal_residual(self, position, quaternion, progress, *, goal_ids):
        """Differentiable violations of allowed goal sets for initialization only.

        Winding acceptance still requires the complete ordered candidate path;
        a local coordinate query is only a modulo-angle seed screening query.
        """
        import torch
        pieces = []
        for collection, index in goal_ids:
            if collection not in {"pose_events", "coordinate_events", "supports"}:
                raise ValueError("Unknown true goal collection")
            goal = getattr(self, collection)[index]
            if collection == "pose_events":
                p, r = goal.errors(position, quaternion)
                pieces.extend((torch.relu(p-goal.position_tolerance_m)[..., None],
                               torch.relu(r-goal.rotation_tolerance_rad)[..., None]))
            elif collection == "coordinate_events":
                pieces.append(goal.evaluate(position, quaternion, progress)[..., None])
            else:
                pieces.append(goal.evaluate(position, quaternion, self.numerical)[..., None])
        return torch.cat(pieces, -1) if pieces else position[..., :0]

    def project_residual(self, position, quaternion, progress, phase_index=None,
                         joint_position=None, joint_names=None, active_requirements=None):
        """Real equality/manifold equations, never source-stage or band centers."""
        import torch
        pieces = []
        selected = None if active_requirements is None else set(active_requirements)
        if selected is not None and any(name not in {"pose_relations", "mechanisms", "couplings"}
                or not 0 <= index < len(getattr(self, name)) for name, index in selected):
            raise ValueError("Unknown active projection requirement")
        for index, requirement in enumerate(self.pose_relations):
            if selected is not None and ("pose_relations", index) not in selected:
                continue
            exact = np.isfinite(requirement.pose_bounds[:, 0]) & (requirement.pose_bounds[:, 0] == requirement.pose_bounds[:, 1])
            if exact.any():
                # BEGIN static equality masks need no device nonzero operation.
                from .optimizer_device_checks import constant_tensor
                exact_columns = constant_tensor(np.flatnonzero(exact), position, dtype=torch.long)
                error = requirement.error(position, quaternion, progress)[..., exact_columns]-_tensor(requirement.pose_bounds[exact, 0], position)
                # END all residual rows and subsequent reductions retain their order.
                mask = ((progress >= requirement.start-1e-7) & (progress <= requirement.end+1e-7))[..., None]
                pieces.append(torch.where(mask, error, 0.) if selected is None else error)
        for index, requirement in enumerate(self.mechanisms):
            if selected is not None and ("mechanisms", index) not in selected:
                continue
            error = requirement.project_residual(position, quaternion, progress)
            mask = ((progress >= requirement.start-1e-7) & (progress <= requirement.end+1e-7))[..., None]
            pieces.append(torch.where(mask, error, 0.) if selected is None else error)
        for index, coupling in enumerate(self.couplings):
            if selected is not None and ("couplings", index) not in selected:
                continue
            if joint_position is None or joint_names is None:
                raise ValueError("Projection needs named joints for actuator coupling")
            columns = [tuple(joint_names).index(name) for name in coupling.joint_names]
            # BEGIN projection keeps the exact column order on either device.
            from .optimizer_device_checks import constant_tensor
            index = constant_tensor(columns, joint_position, dtype=torch.int64)
            pieces.append(joint_position[..., index] @ _tensor(coupling.matrix.T, joint_position)-_tensor(coupling.target, joint_position))
            # END no coupling rows are reordered or combined.
        return torch.cat(pieces, -1) if pieces else position[..., :0]
    # END projection selects equations without inactive zero rows or invented poses.
# END one current skill with separate goals, process relations and references.


# BEGIN source timestamps, rather than row counts, parameterize each reference.
def _path_from_rows(poses, source, *, timestamps, model=None, coordinates=None, attachment=None):
    times = np.asarray(timestamps, float)
    if times.shape != (len(poses),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("A source reference needs its original ordered timestamps")
    progress = ((times-times[0])/(times[-1]-times[0]) if len(times) > 1 else np.array([0.]))
    return PosePath(progress, poses, source, model, coordinates,
                    np.eye(4) if attachment is None else attachment)
# END source timestamps, rather than row counts, parameterize each reference.


def _model_from_diagnostics(value):
    # BEGIN require the current grounding model's complete serialization.
    if value["kind"] == "free":
        return None
    return MotionModel(value["kind"], np.asarray(value["coordinates"], float),
                       np.asarray(value["reference_pose"], float), np.asarray(value["basis"], float),
                       None if value["axis_point"] is None else np.asarray(value["axis_point"], float),
                       value["position_tolerance"], value["rotation_tolerance"], value["diagnostics"])
    # END require the current grounding model's complete serialization.


# BEGIN compile exactly one current skill without source-phase obligations.
def build_planning_problem(skills, *, joint_names, start_q, start_dq, tool_link,
                           reference_frame, base_pose, tool_transform,
                           numerical=NumericalTolerance(), couplings=()):
    """Bind one current skill's actual entry, independent goals and source map.

    Every ordinary source row remains available for initialization/deviation
    reporting. Only explicit functional events, physical models, scoped
    geometry and requested execution events become planning requirements.
    Effects remain to be evaluated; they do not automatically require a stop
    or grant evidence to a future skill.
    """
    from .grasp import validate_timeline
    from .program import ground_program, _current_model, _current_initial_coordinates
    from .pipeline import _surface, _interior

    if not isinstance(skills, (list, tuple)) or len(skills) != 1:
        raise ValueError("Provide exactly one current skill")
    spec = skills[0]
    skill_id = spec.get("skill_id")
    if not isinstance(skill_id, str) or not skill_id:
        raise ValueError("The current skill needs a nonempty instance ID")
    program, scene = deepcopy(spec["program"]), deepcopy(spec["scene"])
    demo = program["demo"]
    frames, times = validate_timeline(demo["frame_indices"], demo["timestamps"])
    if len(frames) < 2:
        raise ValueError("A robot planning request requires a source interval with positive duration")
    identity = (demo.get("tool_id"), demo.get("tool_frame"))
    if (any(not isinstance(value, str) or not value for value in identity)
            or identity != (scene.get("tool_id"), scene.get("tool_frame"))):
        raise ValueError("The current skill requires the same demonstrated/current tool_id and tool_frame")
    if scene.get("reference_frame") != reference_frame:
        raise ValueError("Grounding and collision queries must share one explicit reference frame")
    carrier = scene.get("carrier_id")
    if not isinstance(carrier, str) or not carrier:
        raise ValueError("The current skill needs its physical carrier identity")
    initial_key = "carrier_pose" if program["skill_kind"] == "grasp" else "initial_pose"
    initial = _pose_array(scene[initial_key], "current skill carrier pose")
    scene[initial_key] = initial
    source = {"skill_id": skill_id, "frame_indices": frames.tolist(),
              "timestamps": times.tolist(), "kind": "current_binding_and_original_source"}
    grounded = ground_program(program, scene)
    phases, events, conditions, supports, interiors, pose_events, coordinate_events, mechanisms = [], [], [], [], [], [], [], []
    frame_progress = {}

    def append_interval(begin, end, path, metadata):
        """A mode boundary partitions reference coordinates, never robot q."""
        if begin == end:
            frame_progress[int(frames[begin])] = float(len(phases))
            return
        index = len(phases)
        phases.append(MotionPhase(f"{skill_id}:{begin}-{end}", path, metadata={
            "skill_id": skill_id, "carrier_id": carrier, "owner_initial_pose": initial.tolist(),
            "source_frames": frames[begin:end+1].tolist(), **metadata}))
        for row in range(begin, end+1):
            frame_progress[int(frames[row])] = index+float(path.progress[row-begin])

    if program["skill_kind"] == "grasp":
        path = _path_from_rows(grounded["eef_poses"], source, timestamps=times)
        candidate = grounded["candidate_T_carrier_eef"]
        append_interval(0, len(frames)-1, path, {"mode": "grasp",
            "protected_start_frame": grounded["protected_start_frame"],
            "candidate_attachment": candidate.tolist()})
        # BEGIN grasp precision belongs to the explicit current task relation.
        # The shared PoseEvent preserves its entire metric/angle allowed set;
        # its nominal transform is an initializer, not an extra exact equality.
        # Source poses and the contact-evidence frame supply no extra goals.
        grasp_goal = grounded["grasp_goal"]
        pose_events.append(PoseEventRequirement(1., np.eye(4), grounded["goal_pose"],
            grasp_goal["position_tolerance_m"], grasp_goal["rotation_tolerance_rad"],
            {"kind": "bound_grasp_relation", "carrier_id": carrier,
             "binding_source": deepcopy(grasp_goal["source"]), **source},
            actual_relation={"target_owner_id": carrier,
                             "T_owner_target": grasp_goal["T_carrier_eef"]}))
        # END numerical convergence settings cannot widen or supply grasp task accuracy.
        # BEGIN bind effect assessment to the exact original event identity.
        conditions.append({"kind": "candidate_attachment", "skill_id": skill_id,
            "event_id": "__stable_holding__",
            "condition_id": f"{skill_id}:stable_holding", "carrier_id": carrier,
            "T_carrier_eef": candidate.tolist(), "progress": 1.,
            "effect_requirement": "stable_grasp_from_current_synchronized_evidence"})
        events.append(PlanningEvent("__stable_holding__", skill_id, 1., int(frames[-1]), "effect", source))
        # END no effect association is inferred from its name or clock value.
        # BEGIN each original close requires arrival before its contact permission.
        # Keep the terminal owner-relative task result at index zero. These
        # additional point goals instead compare measured robot FK with the
        # frozen perception-bound command target; they neither claim a fresh
        # owner observation nor impose a maintained pose during closure.
        commands = grounded["gripper_events"]
        for index, command in enumerate(commands):
            if command["command"] != "close":
                continue
            begin = frame_progress[int(command["frame_index"])]
            end = next((frame_progress[int(later["frame_index"])]
                        for later in commands[index+1:] if later["command"] == "open"), 1.)
            activation = {"skill_id": skill_id, "event_id": command["event_id"],
                          "occurrence": command.get("occurrence", 0), "kind": "close"}
            arrival_source = {**source, "kind": "frozen_perception_grasp_arrival",
                "carrier_id": carrier, "binding_source": deepcopy(grasp_goal["source"]),
                "activation_event": deepcopy(activation),
                "semantics": "Robot FK arrival at the frozen perception-bound interaction target before close; not a current owner observation or Holding evidence"}
            arrival_id = ["pose_events", len(pose_events)]
            pose_events.append(PoseEventRequirement(begin, np.eye(4), grounded["goal_pose"],
                grasp_goal["position_tolerance_m"], grasp_goal["rotation_tolerance_rad"],
                arrival_source, actual_relation=None))
            conditions.append({"kind": "grasp_contact_activation",
                "condition_id": f"{skill_id}:grasp_contact:{command['event_id']}:{activation['occurrence']}",
                "carrier_id": carrier, "arrival_goal_id": arrival_id,
                "activation_event": activation, "interval": [begin, end],
                "source": deepcopy(arrival_source)})
        # END source close identity/progress and the complete skill clock stay intact.
    elif program["skill_kind"] == "manipulation":
        objects, eef = grounded["object_poses"], grounded["eef_poses"]
        attachment = _pose_array(scene["holding"]["T_carrier_eef"], "actual attachment")
        attachment_id = f"{skill_id}:actual_holding"
        last_held = len(objects)-1
        release = demo.get("release_frame")
        held_end = 1. if last_held else 0.
        conditions.append({"kind": "observed_attachment", "skill_id": skill_id,
            "condition_id": attachment_id, "carrier_id": carrier,
            "T_carrier_eef": attachment.tolist(), "source": deepcopy(scene["holding"]), "progress": 0.})
        model, limits = (None, None) if scene.get("motion_model") is None else _current_model(scene["motion_model"])
        mechanism = None
        if model is not None:
            mechanism = MechanismRequirement(0., held_end, model, _inverse(attachment), limits,
                _current_initial_coordinates(scene["motion_model"], model, initial,
                    require_lift=bool(spec.get("coordinate_directions")) or any(
                        event.get("constraint_kind") == "joint_coordinate" for event in demo["functional_events"])),
                scene["motion_model"]["source"])
            mechanisms.append(mechanism)
            # Reference coordinates are optional initialization data. The
            # authoritative mechanism above was bound directly from the scene.
            reference_model = _model_from_diagnostics(grounded["motion_checks"]["model"])
            coordinates = reference_model.coordinates
        else:
            reference_model, coordinates = None, None
        release_id = f"{skill_id}:actual_release" if release is not None else None
        append_interval(0, last_held, _path_from_rows(objects, source, timestamps=times[:last_held+1],
            model=reference_model, coordinates=coordinates, attachment=attachment),
            {"mode": "held", "attachment": attachment.tolist(), "conditional": False,
             "attachment_condition_id": attachment_id, "attachment_interval": [0., held_end],
             "release_condition_id": release_id})
        if last_held < len(frames)-1:
            append_interval(last_held, len(frames)-1,
                _path_from_rows(eef[last_held:], source, timestamps=times[last_held:]),
                {"mode": "released_retreat", "release_progress": held_end})
        for event in demo["functional_events"]:
            progress = frame_progress[int(event["frame_index"])]
            binding = scene["functional_bindings"][event["event_id"]]
            check = next(row for row in grounded["function_checks"] if row["event_id"] == event["event_id"])
            if program["task"]["relation"] == "on":
                face = binding["on_geometry"]
                supports.append(SupportRequirement(progress, _inverse(attachment), scene["body_points"],
                    face["carrier_region_points"], face["carrier_outward_normal"], _surface(scene["support"]),
                    face["gap_tolerance"], face["normal_tolerance"], scene.get("margin", 0.), binding["source"]))
            elif event.get("constraint_kind") == "joint_coordinate":
                coordinate_events.append(CoordinateEventRequirement(progress, mechanism,
                    float(np.asarray(check["raw_target_coordinates"])[0]), event["coordinate_tolerance"],
                    {"event": event["source"], "binding": check["binding_source"], "entry": check["entry_source"]},
                    coordinate_mode=check["coordinate_mode"], coordinate_origin=None if check["entry_coordinates"] is None
                    else float(np.asarray(check["entry_coordinates"])[0])))
            else:
                material = _pose_array(binding["carrier_frame"], "current material function frame")
                pose_events.append(PoseEventRequirement(progress, _inverse(attachment) @ material,
                    np.asarray(check["goal_pose"]) @ material,
                    event["position_tolerance_m"], event["rotation_tolerance_rad"], event["source"],
                    # BEGIN current carrier/reference observations own actual satisfaction.
                    actual_relation={"function_owner_id": carrier, "T_owner_function": material,
                        "attachment_condition_id": attachment_id,
                        "target_owner_id": scene["reference_id"],
                        "T_owner_target": _inverse(_pose_array(scene["reference_pose"], "Reference pose"))
                                          @ np.asarray(check["goal_pose"]) @ material}))
                    # END the initial reference-owner pose only binds planning coordinates.
            if program["task"]["relation"] == "inside" and "interior" in scene:
                interior = _interior(scene["interior"])
                interiors.append(InteriorRequirement(progress, progress, _inverse(attachment), scene["body_points"],
                    interior.halfspaces, scene.get("margin", 0.), interior.source))
            # Original timing and explicit execution requirements annotate
            # this same occurrence; a deadline never creates a second action.
            events.append(PlanningEvent(event["event_id"], skill_id, progress, int(event["frame_index"]),
                "function", event["source"], event.get("requires_evidence"), event.get("stop", False),
                event.get("occurrence", 0), tuple(event.get("timing", ()))))
        if "entry_region" in scene:
            interior = _interior(scene["entry_region"])
            begin = frame_progress[int(frames[scene["entry_start_index"]])]
            interiors.append(InteriorRequirement(begin, held_end, _inverse(attachment), scene["body_points"],
                interior.halfspaces, scene.get("margin", 0.), interior.source))
        if release is not None:
            # BEGIN release outcome and its event share one explicit identity.
            conditions.append({"kind": "predicted_release", "carrier_id": carrier,
                "skill_id": skill_id, "event_id": "__release_verified__",
                "condition_id": release_id, "progress": held_end, "T_carrier_eef": attachment.tolist(),
                "nominal_pose": objects[-1].tolist(), "effect_requirement": "actual_release_from_current_feedback"})
            events.append(PlanningEvent("__release_verified__", skill_id, held_end, int(release), "effect", source))
            # END release timing alone is not evidence of a release outcome.
    else:
        raise ValueError("Only compiled grasp/manipulation programs define a current skill")
    for event in grounded["gripper_events"]:
        events.append(PlanningEvent(event["event_id"], skill_id, frame_progress[int(event["frame_index"])],
            int(event["frame_index"]), event["command"], {"original_event": deepcopy(event), **source},
            event.get("requires_evidence"), event.get("stop", False), event.get("occurrence", 0), tuple(event.get("timing", ()))))
    # Only explicitly supplied execution prerequisites or stops enter this list.
    # Evidence checks and zero velocity are independent fields, not implications.
    for event in spec.get("execution_events", ()):
        frame = event["frame_index"]
        if frame not in frame_progress:
            raise ValueError("An execution event must name an original source frame")
        events.append(PlanningEvent(event["event_id"], skill_id, frame_progress[frame], int(frame),
            event["kind"], event["source"], event.get("requires_evidence"), event.get("stop", False),
            event.get("occurrence", 0), tuple(event.get("timing", ()))))
    # BEGIN only explicit native direction declarations become process obligations.
    direction_specs = spec.get("coordinate_directions", ())
    if direction_specs:
        if len(mechanisms) != 1:
            raise ValueError("Coordinate directions require one independently evidenced current mechanism")
        directions = []
        for row in direction_specs:
            if row["start_frame"] not in frame_progress or row["end_frame"] not in frame_progress:
                raise ValueError("A coordinate direction must name original source scope frames")
            directions.append(CoordinateDirectionRequirement(frame_progress[row["start_frame"]],
                frame_progress[row["end_frame"]], row["coordinate_index"], row["kind"], row["sign"],
                row["minimum"], row["tolerance"], row["source"], row.get("origin_coordinate"), row.get("origin_source")))
        mechanisms[0].directions = tuple(directions)
        mechanisms[0].__post_init__()
    # END reference increments alone never imply monotonicity or minimum rate.
    return PlanningProblem(tuple(joint_names), start_q, start_dq, tool_link, reference_frame,
        base_pose, tool_transform, tuple(phases), tuple(events), tuple(couplings), numerical,
        tuple(conditions), tuple(supports), tuple(interiors), tuple(pose_events), tuple(coordinate_events),
        mechanisms=tuple(mechanisms), source_timestamps=times,
        source_progress=np.asarray([frame_progress[int(frame)] for frame in frames]))
# END compile exactly one current skill without source-phase obligations.
# END shared skill-process constraints for generation and verification.
