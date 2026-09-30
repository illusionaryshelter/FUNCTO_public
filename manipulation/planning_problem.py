"""Robot-independent requirements for a bounded, jointly planned skill window.

All poses use column vectors, metres, and wxyz quaternions. Progress indexes
skills/phases, not time: phase i owns [i, i+1], including both boundary states.
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
    import torch
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


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


@dataclass
class MotionPhase:
    """One fixed interaction mode, with explicit allowed pose-error intervals.

    Bounds are target-local XYZ and rotation-vector components, shape (6,2).
    Equal bounds are equalities; finite-width bounds are inequalities; an
    unbounded component is free. ``free`` constrains only the endpoint unless
    ``require_end=False`` supplies an independent functional endpoint instead.
    """
    name: str
    reference: PosePath
    kind: str = "pose"
    pose_bounds: np.ndarray = field(default_factory=lambda: np.zeros((6, 2)))
    require_end: bool = True
    metadata: dict = field(default_factory=dict)
    endpoint_family: dict | None = None
    frame_anchor: dict | None = None

    def __post_init__(self):
        self.pose_bounds = np.asarray(self.pose_bounds, float).copy()
        if (not self.name or self.kind not in {"free", "pose"}
                or self.pose_bounds.shape != (6, 2) or np.isnan(self.pose_bounds).any()
                or np.any(self.pose_bounds[:, 0] > self.pose_bounds[:, 1])):
            raise ValueError("A phase needs a name, valid kind and ordered six-axis bounds")
        if self.endpoint_family is not None and (self.kind != "pose" or self.reference.motion_model is not None):
            raise ValueError("Free endpoint transfer applies to a necessary free-motion family, not a mechanism")
        self._body_reference = replace(self.reference, attachment=np.eye(4))
        if self.frame_anchor is not None:
            self.frame_anchor = deepcopy(self.frame_anchor)
            self.frame_anchor["reference_pose"] = _pose_array(self.frame_anchor["reference_pose"], "anchor reference pose")
            if not np.isfinite(self.frame_anchor["progress"]) or self.frame_anchor["progress"] < 0:
                raise ValueError("A predicted frame needs a finite source anchor")


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
    """
    progress: float
    tool_from_function: np.ndarray
    target: np.ndarray
    position_tolerance_m: float
    rotation_tolerance_rad: float
    source: object

    def __post_init__(self):
        self.tool_from_function = _pose_array(self.tool_from_function, "T_tool_function")
        self.target = _pose_array(self.target, "functional target")
        if (not self.source or not np.isfinite(self.progress)
                or not 0 < self.position_tolerance_m < np.inf
                or not 0 < self.rotation_tolerance_rad < np.pi):
            raise ValueError("A material-frame event needs finite metric thresholds and its original source")

    def errors(self, position, quaternion):
        import torch
        p, q = _compose(position, quaternion, self.tool_from_function)
        error = _pose_error(p, q, _tensor(self.target[:3, 3], p), _tensor(_quaternion(self.target), q))
        return torch.linalg.vector_norm(error[..., :3], dim=-1), torch.linalg.vector_norm(error[..., 3:], dim=-1)

    def evaluate(self, position, quaternion):
        import torch
        p, r = self.errors(position, quaternion)
        return torch.relu(p-self.position_tolerance_m)+torch.relu(r-self.rotation_tolerance_rad)


@dataclass
class CoordinateEventRequirement:
    """A native mechanism event, with distance or unwrapped-angle provenance.

    FK supplies a local error around the expected lifted reference. The full
    coordinate path is separately preserved by PosePath, so an equal final
    rotation never substitutes for executing the demonstrated winding.
    """
    progress: float
    tool_from_carrier: np.ndarray
    target: np.ndarray
    axis: np.ndarray
    kind: str
    target_coordinate: float
    tolerance: float
    source: object

    def __post_init__(self):
        from .regions import _unit
        self.tool_from_carrier = _pose_array(self.tool_from_carrier, "T_tool_carrier")
        self.target = _pose_array(self.target, "mechanism event target")
        self.axis = _unit(self.axis, "mechanism axis")
        if (self.kind not in {"prismatic", "revolute"} or not self.source
                or not np.isfinite(self.target_coordinate) or not 0 < self.tolerance < np.inf):
            raise ValueError("A native coordinate event needs its original units, target and threshold")

    def error(self, position, quaternion):
        p, q = _compose(position, quaternion, self.tool_from_carrier)
        axis = _tensor(self.axis, p)
        if self.kind == "prismatic":
            return ((p-_tensor(self.target[:3, 3], p))*axis).sum(-1).abs()
        # Left-relative rotation vectors are expressed in the world frame, as
        # is the declared joint axis. The local branch is around the expected
        # coordinate, rather than an angle recovered modulo 2pi from scratch.
        turn = quaternion_multiply(q, quaternion_conjugate(_tensor(_quaternion(self.target), q)))
        return (quaternion_rotvec(turn)*axis).sum(-1).abs()

    def evaluate(self, position, quaternion):
        import torch
        return torch.relu(self.error(position, quaternion)-self.tolerance)


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


@dataclass
class PlanningProblem:
    """One shared joint trajectory for the current skill and at most one next.

    ``base_pose`` and ``tool_transform`` are mandatory calibration inputs; the
    named robot link is never silently equated with a demonstrated TCP frame.
    A skill may contain several phases. Low-level phase inputs need not name
    skills, but all explicitly declared phase/event skill IDs share this limit.
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

    def __post_init__(self):
        self.joint_names = tuple(self.joint_names)
        self.phases, self.events, self.couplings = tuple(self.phases), tuple(self.events), tuple(self.couplings)
        self.start_q, self.start_dq = np.asarray(self.start_q, float).copy(), np.asarray(self.start_dq, float).copy()
        if (not self.joint_names or len(set(self.joint_names)) != len(self.joint_names)
                or any(not isinstance(n, str) or not n for n in self.joint_names)
                or self.start_q.shape != (len(self.joint_names),) or self.start_dq.shape != self.start_q.shape
                or not np.isfinite(self.start_q).all() or not np.isfinite(self.start_dq).all()
                or not self.phases or not self.tool_link or not self.reference_frame):
            raise ValueError("A planning problem needs current named joint state, calibrated frames, and phases")
        self.base_pose = _pose_array(self.base_pose, "T_reference_robot_base")
        self.tool_transform = _pose_array(self.tool_transform, "T_link_tool")
        # BEGIN bound declared skill identities without limiting motion phases.
        # Direct phase-level callers may omit skill identities. When supplied,
        # phase and event IDs describe one shared window, not separate budgets.
        skill_ids = [phase.metadata["skill_id"] for phase in self.phases
                     if "skill_id" in phase.metadata]
        skill_ids.extend(event.skill_id for event in self.events)
        if any(not isinstance(skill_id, str) or not skill_id for skill_id in skill_ids):
            raise ValueError("Declared planning skill IDs must be nonempty strings")
        if len(set(skill_ids)) > 2:
            raise ValueError("A planning window supports the current skill and at most one next skill")
        # END bound declared skill identities without limiting motion phases.
        for index, phase in enumerate(self.phases):
            if phase.frame_anchor is not None and phase.frame_anchor["progress"] > index:
                raise ValueError("A predicted owner frame must come from this phase's entrance or an earlier boundary")
        identities = set()
        for event in self.events:
            identity = (event.skill_id, event.event_id)
            if (identity in identities or not event.event_id or not event.skill_id or not event.source
                    or not np.isfinite(event.progress) or not 0 <= event.progress <= self.phase_count
                    or type(event.source_frame) is not int or event.source_frame < 0):
                raise ValueError("Events need unique skill/event IDs and exact original frame/progress")
            identities.add(identity)
            if event.requires_evidence and not event.stop:
                raise ValueError("An evidence barrier needs a planned stop; moving verification is not supplied")

    @property
    def phase_count(self):
        return len(self.phases)

    @property
    def anchors(self):
        # BEGIN preserve every required reference breakpoint in generation.
        # A short source excursion can lie wholly between uniform rollout
        # samples.  Candidate-joint speed cannot reveal a motion it omitted,
        # so the source breakpoints themselves are mandatory FK query sites.
        # Free approach samples remain initializers, not process constraints.
        return np.unique([*range(self.phase_count+1), *(e.progress for e in self.events),
                          *(index + float(s) for index, phase in enumerate(self.phases)
                            if phase.kind == "pose" for s in phase.reference.progress),
                          *(r.progress for r in self.supports),
                          *(s for r in self.interiors for s in (r.start, r.end)),
                          *(r.progress for r in self.pose_events),
                          *(r.progress for r in self.coordinate_events),
                          *(p.frame_anchor["progress"] for p in self.phases if p.frame_anchor is not None)])
        # END preserve every required reference breakpoint in generation.

    @property
    def stop_progress(self):
        return np.array([e.progress for e in self.events if e.stop], float)

    def sample_reference(self, progress):
        import torch
        position, quaternion = self.phases[0].reference.sample(progress.clamp(0, 1))
        if hasattr(self, "_fixed_anchors"):
            position, quaternion = self._phase_reference(0, position, quaternion, progress)
        for index, phase in enumerate(self.phases[1:], 1):
            p, q = phase.reference.sample((progress-index).clamp(0, 1))
            if hasattr(self, "_fixed_anchors"):
                p, q = self._phase_reference(index, position, quaternion, progress)
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

    def _anchor(self, position, quaternion, progress, anchor):
        if hasattr(self, "_fixed_anchors"):
            p, q = self._fixed_anchors[float(anchor)]
            return (_tensor(p, position).expand(*progress.shape[:-1], 1, 3),
                    _tensor(q, quaternion).expand(*progress.shape[:-1], 1, 4))
        return anchor_pose(position, quaternion, progress, anchor)

    def with_fixed_endpoints(self, position, quaternion, progress):
        """Freeze one complete search proposal; the original problem is unchanged.

        All exact anchors are required. This is only for OMPL's per-state
        projection and validity checks. Whole-window optimization must consume
        the original problem again, so successful early boundaries stay free.
        """
        import torch
        p = torch.as_tensor(position, dtype=torch.float64).detach().cpu()
        q = torch.as_tensor(quaternion, dtype=torch.float64).detach().cpu()
        input_s = torch.as_tensor(progress).detach().cpu()
        if not input_s.dtype.is_floating_point:
            input_s = input_s.to(torch.float64)
        s = input_s.to(torch.float64)
        if s.ndim != 2 or s.shape[0] != 1 or p.shape != (*s.shape, 3) or q.shape != (*s.shape, 4):
            raise ValueError("A fixed whole-window proposal needs [1,A,3/4] poses and [1,A] progress")
        frozen = deepcopy(self)
        frozen._fixed_anchors = {}
        for value in self.anchors:
            # Compare in the input precision first. A native float32 event at
            # e.g. s=17+1/3 is the same declared anchor after rounding, not a
            # nearby trajectory sample selected by nearest-neighbour search.
            represented = float(input_s.new_tensor(value))
            if not (input_s == input_s.new_tensor(value)).any():
                raise ValueError("Search proposals must include every exact problem anchor")
            ap, aq = anchor_pose(p, q, s, represented)
            if not torch.isfinite(ap).all() or not torch.isfinite(aq).all():
                raise ValueError("Search proposals must include every exact problem anchor")
            frozen._fixed_anchors[float(value)] = (ap.numpy()[0, 0].copy(), aq.numpy()[0, 0].copy())
        return frozen

    def _phase_reference(self, index, position, quaternion, progress):
        """Differentiate the existing endpoint-transfer family, not a free path.

        Material-frame B(s) is transferred as B(s) inv(B(0)) X(0), then
        receives the same linear translation and left-Slerp endpoint correction
        as motion.transfer_motion. Only explicitly functional endpoints move.
        """
        import torch
        phase = self.phases[index]
        local = (progress-index).clamp(0, 1)

        def warped(p, q):
            if phase.frame_anchor is None:
                return p, q
            ap, aq = self._anchor(position, quaternion, progress, phase.frame_anchor["progress"])
            reference = phase.frame_anchor["reference_pose"]
            rq, rp = _tensor(_quaternion(reference), q), _tensor(reference[:3, 3], p)
            delta_q = quaternion_multiply(aq, quaternion_conjugate(rq))
            return ap+quaternion_rotate(delta_q, p-rp), quaternion_multiply(delta_q, q)

        if phase.endpoint_family is None:
            p, q = phase.reference.sample(local)
            return warped(p, q)
        body_path = phase._body_reference
        p, q = warped(*body_path.sample(local))
        zero, one = torch.zeros_like(progress[..., :1]), torch.ones_like(progress[..., :1])
        p0, q0 = warped(*body_path.sample(zero))
        p1, q1 = warped(*body_path.sample(one))
        if phase.endpoint_family.get("move_start", False):
            start_p, start_q = self._anchor(position, quaternion, progress, index)
            start_p, start_q = _compose(start_p, start_q, _inverse(phase.reference.attachment))
        else:
            start_p, start_q = p0, q0
        cq = quaternion_multiply(quaternion_conjugate(q0), start_q)
        cp = quaternion_rotate(quaternion_conjugate(q0), start_p-p0)
        mapped_p = p+quaternion_rotate(q, cp.expand_as(p))
        mapped_q = quaternion_multiply(q, cq)
        mapped_end_p = p1+quaternion_rotate(q1, cp)
        mapped_end_q = quaternion_multiply(q1, cq)
        mode = phase.endpoint_family["end"]
        if mode != "unanchored":
            if mode == "variable":
                end_p, end_q = self._anchor(position, quaternion, progress, index+1)
                end_p, end_q = _compose(end_p, end_q, _inverse(phase.reference.attachment))
            else:
                end_p, end_q = p1, q1
            mapped_p = mapped_p+local[..., None]*(end_p-mapped_end_p)
            correction = quaternion_multiply(end_q, quaternion_conjugate(mapped_end_q)).expand_as(q)
            identity = torch.zeros_like(correction)
            identity[..., 0] = 1
            mapped_q = quaternion_multiply(_slerp(identity, correction, local), mapped_q)
        return _compose(mapped_p, mapped_q, phase.reference.attachment)

    def residuals(self, position, quaternion, progress, joint_position=None,
                  joint_names=None, joint_velocity=None):
        """Nonnegative per-term violations; no term can cancel another one."""
        import torch
        if position.shape != (*progress.shape, 3) or quaternion.shape != (*progress.shape, 4):
            raise ValueError("Pose arrays must align with the full progress horizon")
        result = {"progress_bounds": torch.relu(-progress) + torch.relu(progress-self.phase_count)}
        tolerance = _tensor([self.numerical.position_m]*3 + [self.numerical.rotation_rad]*3, position)
        for index, phase in enumerate(self.phases):
            local = (progress-index).clamp(0, 1)
            p, q = self._phase_reference(index, position, quaternion, progress)
            error = _pose_error(position, quaternion, p, q)
            bounds = _tensor(phase.pose_bounds, position)
            violation = (torch.relu(bounds[:, 0]-error-tolerance) + torch.relu(error-bounds[:, 1]-tolerance)).sum(-1)
            mask = ((progress >= index-1e-7) & (progress <= index+1+1e-7)
                    if phase.kind == "pose" else (progress-index-1).abs() <= 1e-7)
            if phase.kind == "free" and not phase.require_end:
                mask = torch.zeros_like(mask)
            result[f"phase:{index}:{phase.name}"] = torch.where(mask, violation, torch.zeros_like(violation))
        for index, requirement in enumerate(self.supports):
            violation = requirement.evaluate(position, quaternion, self.numerical)
            result[f"support:{index}"] = torch.where((progress-requirement.progress).abs() <= 1e-7,
                                                       violation, torch.zeros_like(violation))
        for index, requirement in enumerate(self.interiors):
            violation = requirement.evaluate(position, quaternion, self.numerical)
            mask = (progress >= requirement.start-1e-7) & (progress <= requirement.end+1e-7)
            result[f"interior:{index}"] = torch.where(mask, violation, torch.zeros_like(violation))
        for index, requirement in enumerate(self.pose_events):
            violation = requirement.evaluate(position, quaternion)
            result[f"function_pose:{index}"] = torch.where((progress-requirement.progress).abs() <= 1e-7,
                                                           violation, torch.zeros_like(violation))
        for index, requirement in enumerate(self.coordinate_events):
            violation = requirement.evaluate(position, quaternion)
            result[f"function_coordinate:{index}"] = torch.where((progress-requirement.progress).abs() <= 1e-7,
                                                                 violation, torch.zeros_like(violation))
        if self.couplings:
            if joint_position is None or joint_names is None:
                raise ValueError("Actuator coupling requires named robot joint positions")
            for index, coupling in enumerate(self.couplings):
                columns = [tuple(joint_names).index(name) for name in coupling.joint_names]
                error = joint_position[..., columns] @ _tensor(coupling.matrix.T, joint_position) - _tensor(coupling.target, joint_position)
                result[f"coupling:{index}"] = torch.relu(error.abs()-self.numerical.joint).sum(-1)
        for event in self.events:
            if event.stop:
                if joint_velocity is None:
                    raise ValueError("Verification stops require actual joint velocity evaluation")
                mask = (progress-event.progress).abs() <= 1e-7
                result[f"stop:{event.skill_id}:{event.event_id}"] = torch.where(
                    mask, torch.relu(joint_velocity.abs()-self.numerical.velocity).sum(-1), torch.zeros_like(progress))
        return result

    def evaluate(self, position, quaternion, progress, joint_position=None,
                 joint_names=None, joint_velocity=None):
        return sum(self.residuals(position, quaternion, progress, joint_position,
                                  joint_names, joint_velocity).values())

    # BEGIN retain an optimizer gradient for source pose equalities.
    def optimizer_preferences(self, position, quaternion, progress):
        """Squared source equalities, without inventing centers of allowed bands.

        Equal finite bounds specify an actual pose component. Numerical
        acceptance tolerances do not make that component a skill freedom:
        targeting the equality also avoids stopping at the sampling grid's
        tolerance edge. Finite-width and unbounded axes receive no preference.
        References retain their actual functional endpoint and release-anchor
        dependence, so a whole allowed endpoint family remains available.
        Only the trajectory optimizer consumes this value; raw residuals and
        final acceptance keep their original, separate tolerance contract.
        """
        import torch
        if position.shape != (*progress.shape, 3) or quaternion.shape != (*progress.shape, 4):
            raise ValueError("Pose arrays must align with the full progress horizon")
        result = torch.zeros_like(progress)
        for index, phase in enumerate(self.phases):
            equalities = (np.isfinite(phase.pose_bounds).all(axis=1)
                          & (phase.pose_bounds[:, 0] == phase.pose_bounds[:, 1]))
            if not equalities.any() or (phase.kind == "free" and not phase.require_end):
                continue
            p, q = self._phase_reference(index, position, quaternion, progress)
            # Slice finite components before subtraction: free infinite bounds
            # must never enter arithmetic or introduce NaNs into autograd.
            columns = np.flatnonzero(equalities).tolist()
            error = _pose_error(position, quaternion, p, q)[..., columns]
            target = _tensor(phase.pose_bounds[equalities, 0], position)
            preference = (error-target).square().sum(-1)
            mask = ((progress >= index-1e-7) & (progress <= index+1+1e-7)
                    if phase.kind == "pose" else (progress-index-1).abs() <= 1e-7)
            result = result + torch.where(mask, preference, torch.zeros_like(preference))
        return result
    # END retain an optimizer gradient for source pose equalities.

    def measurements(self, position, quaternion, progress):
        """Report constraint-reference errors and original-reference deviations.

        Existing ``*_error_*`` fields measure the reference used by the hard
        constraints, including explicitly allowed endpoint/frame transfers.
        ``original_reference_*`` instead compares to the grounded input path
        at the same source progress, before any candidate-dependent transfer.
        These deviations cover each entire phase, including free approaches;
        they are measurements, not an invented task acceptance threshold.
        """
        import torch
        values = {}
        for index, phase in enumerate(self.phases):
            p, q = self._phase_reference(index, position, quaternion, progress)
            error = _pose_error(position, quaternion, p, q)
            mask = ((progress >= index-1e-7) & (progress <= index+1+1e-7)
                    if phase.kind == "pose" else (progress-index-1).abs() <= 1e-7)
            if phase.kind == "free" and not phase.require_end:
                mask = torch.zeros_like(mask)
            values[f"phase:{index}:position_error_m"] = torch.where(mask, torch.linalg.vector_norm(error[..., :3], dim=-1), 0.)
            values[f"phase:{index}:rotation_error_rad"] = torch.where(mask, torch.linalg.vector_norm(error[..., 3:], dim=-1), 0.)
            values[f"phase:{index}:position_component_error_m"] = torch.where(mask, error[..., :3].abs().amax(-1), 0.)
            values[f"phase:{index}:rotation_component_error_rad"] = torch.where(mask, error[..., 3:].abs().amax(-1), 0.)
            # BEGIN keep the original grounded reference independent of the candidate.
            # Calling _phase_reference here would move the comparison target
            # with the endpoint or release pose and conceal that displacement.
            original_p, original_q = phase.reference.sample((progress-index).clamp(0, 1))
            original_error = _pose_error(position, quaternion, original_p, original_q)
            original_mask = (progress >= index-1e-7) & (progress <= index+1+1e-7)
            prefix = f"phase:{index}:original_reference"
            values[f"{prefix}_position_error_m"] = torch.where(
                original_mask, torch.linalg.vector_norm(original_error[..., :3], dim=-1), 0.)
            values[f"{prefix}_rotation_error_rad"] = torch.where(
                original_mask, torch.linalg.vector_norm(original_error[..., 3:], dim=-1), 0.)
            values[f"{prefix}_position_component_error_m"] = torch.where(
                original_mask, original_error[..., :3].abs().amax(-1), 0.)
            values[f"{prefix}_rotation_component_error_rad"] = torch.where(
                original_mask, original_error[..., 3:].abs().amax(-1), 0.)
            # END independent original-reference measurements.
        for index, event in enumerate(self.pose_events):
            p, r = event.errors(position, quaternion)
            mask = (progress-event.progress).abs() <= 1e-7
            values[f"function:{index}:position_error_m"] = torch.where(mask, p, 0.)
            values[f"function:{index}:rotation_error_rad"] = torch.where(mask, r, 0.)
        for index, event in enumerate(self.coordinate_events):
            unit = "m" if event.kind == "prismatic" else "rad"
            values[f"coordinate:{index}:error_{unit}"] = torch.where(
                (progress-event.progress).abs() <= 1e-7, event.error(position, quaternion), 0.)
        return values

    def project_residual(self, position, quaternion, progress, phase_index,
                         joint_position=None, joint_names=None):
        """True pose equalities for one fixed OMPL mode, never inequality centers."""
        import torch
        phase = self.phases[phase_index]
        if (phase.endpoint_family is not None or phase.frame_anchor is not None) and not hasattr(self, "_fixed_anchors"):
            raise ValueError("Coupled endpoint/frame projection requires one explicitly frozen whole-window proposal")
        pieces = []
        if phase.kind != "free":
            p, q = (self._phase_reference(phase_index, position, quaternion, progress)
                    if hasattr(self, "_fixed_anchors") else phase.reference.sample((progress-phase_index).clamp(0, 1)))
            error = _pose_error(position, quaternion, p, q)
            exact = np.isfinite(phase.pose_bounds[:, 0]) & (phase.pose_bounds[:, 0] == phase.pose_bounds[:, 1])
            pieces.append(error[..., exact] - _tensor(phase.pose_bounds[exact, 0], error))
        for coupling in self.couplings:
            if joint_position is None or joint_names is None:
                raise ValueError("Projection needs named joints for actuator coupling")
            columns = [tuple(joint_names).index(name) for name in coupling.joint_names]
            pieces.append(joint_position[..., columns] @ _tensor(coupling.matrix.T, joint_position)
                          - _tensor(coupling.target, joint_position))
        return torch.cat(pieces, -1) if pieces else position[..., :0]


def _path_from_rows(poses, source, *, model=None, coordinates=None, attachment=None):
    return PosePath(np.linspace(0, 1, len(poses)) if len(poses) > 1 else np.array([0.]),
                    poses, source, model, coordinates,
                    np.eye(4) if attachment is None else attachment)


def _model_from_diagnostics(value):
    if value.get("kind") == "free":
        return None
    return MotionModel(value["kind"], np.asarray(value["coordinates"], float),
                       np.asarray(value["reference_pose"], float), np.asarray(value["basis"], float),
                       None if value["axis_point"] is None else np.asarray(value["axis_point"], float),
                       value["position_tolerance"], value["rotation_tolerance"], value.get("diagnostics", {}))


def build_planning_problem(skills, *, joint_names, start_q, start_dq, tool_link,
                           reference_frame, base_pose, tool_transform,
                           numerical=NumericalTolerance(), couplings=()):
    """Compile the current skill and optional next into one conditional window.

    ``skills`` is an ordered list/tuple of one or two entries containing
    ``skill_id``, the compiled ``program``, its current bound ``scene``, and
    explicit ``barrier_can_stop`` permission. Every skill keeps its complete
    motion phases. Grounding inputs are copied. Next-skill manipulation uses
    object geometry plus the proposed attachment, never the actual-Holding
    wrapper with invented facts.
    A caller continues execution past a barrier only after actual feedback and
    rebinds remaining work if those conditions differ.
    """
    from .grasp import retreat_from_release, validate_timeline
    from .program import ground_program, ground_object_program, _object_reference, _intentions
    from .pipeline import _surface, _interior

    # BEGIN limit lookahead before grounding or predicting any skill state.
    if not isinstance(skills, (list, tuple)) or not 1 <= len(skills) <= 2:
        raise ValueError("Provide the current skill and at most one next skill")
    # END limit lookahead before grounding or predicting any skill state.
    phases, events, conditions, supports, interiors, pose_events, coordinate_events = [], [], [], [], [], [], []
    previous, skill_ids, tool_identity = None, set(), None
    predicted_objects, owner_initial_poses = {}, {}
    for spec in skills:
        skill_id = spec.get("skill_id")
        if not isinstance(skill_id, str) or not skill_id or skill_id in skill_ids:
            raise ValueError("Every window entry needs a unique skill instance ID")
        skill_ids.add(skill_id)
        program, scene = deepcopy(spec["program"]), deepcopy(spec["scene"])
        demo = program["demo"]
        frames, times = validate_timeline(demo["frame_indices"], demo["timestamps"])
        identity = (demo.get("tool_id"), demo.get("tool_frame"))
        if (any(not isinstance(value, str) or not value for value in identity)
                or identity != (scene.get("tool_id"), scene.get("tool_frame"))):
            raise ValueError("Every grounded successor requires the same demonstrated/current tool_id and tool_frame")
        if tool_identity is not None and identity != tool_identity:
            raise ValueError("A joint-planning window cannot silently change tools/TCP")
        tool_identity = identity
        if scene.get("reference_frame") != reference_frame:
            raise ValueError("Every grounding and collision query must use the same explicit reference frame")
        # BEGIN bind planning geometry to the first current owner observation.
        carrier_id = scene.get("carrier_id")
        if not isinstance(carrier_id, str) or not carrier_id:
            raise ValueError("Every robot-planned skill needs its physical carrier identity")
        if carrier_id not in owner_initial_poses:
            initial_key = "carrier_pose" if program["skill_kind"] == "grasp" else "initial_pose"
            owner_initial_poses[carrier_id] = _pose_array(scene[initial_key], "first current owner pose")
        # END bind planning geometry to the first current owner observation.
        source = {"skill_id": skill_id, "frame_indices": frames.tolist(),
                  "timestamps": times.tolist(), "kind": "current_grounding_and_original_program"}
        first_phase = len(phases)
        frame_progress = {}

        def append_interval(begin, end, reference, kind="pose", **kwargs):
            index = len(phases)
            phase = MotionPhase(f"{skill_id}:{begin}-{end}", reference, kind=kind,
                                metadata={"skill_id": skill_id, "carrier_id": scene.get("carrier_id"),
                                          "owner_initial_pose": owner_initial_poses[carrier_id].tolist(),
                                          "source_frames": frames[begin:end+1].tolist(), **kwargs.pop("metadata", {})},
                                **kwargs)
            phases.append(phase)
            for row in range(begin, end+1):
                frame_progress[int(frames[row])] = index+(row-begin)/max(1, end-begin)
            # A one-node phase still owns an endpoint, so its evidence and
            # verification barrier occur at that actual constant-pose end.
            if begin == end:
                frame_progress[int(frames[end])] = index+1.
            return index

        if program["skill_kind"] == "grasp":
            if previous is not None:
                scene["eef_pose"] = previous["eef_pose"].copy()
            predicted = predicted_objects.get(scene.get("carrier_id"))
            frame_anchor = None if predicted is None else predicted["frame_anchor"]
            if predicted is not None:
                scene["carrier_pose"] = predicted["pose"].copy()
            grounded = ground_program(program, scene)
            protected = int(np.flatnonzero(frames == grounded["protected_start_frame"])[0])
            eef = grounded["eef_poses"]
            if protected:
                append_interval(0, protected, _path_from_rows(eef[:protected+1], source), "free", frame_anchor=frame_anchor)
            elif grounded["connection_required"]:
                # No protected sample is displaced to manufacture an approach.
                # A new free connection ends exactly at the required entrance.
                entry = _path_from_rows(np.stack((scene["eef_pose"], eef[0])), source)
                phases.append(MotionPhase(f"{skill_id}:connection", entry, "free",
                                          metadata={"skill_id": skill_id, "no_source_frames": True,
                                                    "carrier_id": carrier_id,
                                                    "owner_initial_pose": owner_initial_poses[carrier_id].tolist()},
                                          frame_anchor=frame_anchor))
            append_interval(protected, len(frames)-1, _path_from_rows(eef[protected:], source),
                            metadata={"mode": "grasp", "candidate_attachment": grounded["candidate_T_carrier_eef"].tolist()},
                            frame_anchor=frame_anchor)
            if spec.get("barrier_can_stop") is not True:
                raise ValueError("Stable Holding verification requires an explicitly permitted hold boundary")
            carrier = scene.get("carrier_id")
            if not isinstance(carrier, str) or not carrier:
                raise ValueError("Joint grasp planning needs a current carrier identity")
            candidate = grounded["candidate_T_carrier_eef"].copy()
            previous = {"carrier_id": carrier, "attachment": candidate,
                        "object_pose": _pose_array(scene["carrier_pose"], "carrier_pose"),
                        "eef_pose": eef[-1].copy(), "condition": f"{skill_id}:stable_holding",
                        "attachment_condition_id": f"{skill_id}:stable_holding",
                        "variable_pose": frame_anchor is not None}
            predicted_objects[carrier] = {"pose": previous["object_pose"].copy(), "frame_anchor": frame_anchor}
            conditions.append({"kind": "candidate_attachment", "skill_id": skill_id,
                               "condition_id": f"{skill_id}:stable_holding",
                               "carrier_id": carrier, "T_carrier_eef": candidate.tolist(),
                               "requires_evidence": "stable_grasp_from_current_synchronized_evidence",
                               "progress": float(len(phases))})
            events.append(PlanningEvent("__stable_holding__", skill_id, float(len(phases)),
                                        int(frames[-1]), "verification", source,
                                        "stable_grasp_from_current_synchronized_evidence", True))
        elif program["skill_kind"] == "manipulation":
            commands = _intentions(demo["gripper_events"], frames)
            release_frame = demo.get("release_frame")
            if release_frame is None and any(event["command"] == "open" for event in commands):
                raise ValueError("An opening skill needs an explicit release boundary")
            if release_frame is not None and not any(
                    event["command"] == "open" and event["frame_index"] <= release_frame for event in commands):
                raise ValueError("A released skill needs an opening intention before its independent release boundary")
            carrier = scene.get("carrier_id")
            conditional = previous is not None
            moving_start = conditional and previous.get("variable_pose", False)
            if conditional:
                if previous["carrier_id"] != carrier or previous.get("attachment") is None:
                    raise ValueError("A successor manipulation needs the same still-held carrier or a new grasp")
                attachment = previous["attachment"].copy()
                attachment_condition_id = previous["attachment_condition_id"]
                scene["initial_pose"] = previous["object_pose"].copy()
                scene["eef_pose"] = previous["eef_pose"].copy()
                objects_result = ground_object_program(program, scene)
                objects = objects_result["poses"]
                last_held = len(objects)-1
                eef = np.empty((len(frames), 4, 4))
                eef[:last_held+1] = objects @ attachment
                if last_held < len(frames)-1:
                    # Reuse the same released-hand correspondence as the
                    # actual-Holding consumer, without fabricating its evidence.
                    source_objects = _poses_array(demo["T_reference_carrier"], "source object poses")
                    _, _, _, reference, first = _object_reference(
                        demo, scene, frames, source_objects, last_held, program["task"], scene["initial_pose"])
                    hand = reference[last_held:] @ first["carrier_binding"] @ _poses_array(demo["T_carrier_eef"], "source hand")[last_held:]
                    eef[last_held:] = retreat_from_release(hand, eef[last_held])
                grounded = {**objects_result, "eef_poses": eef,
                            "gripper_events": demo["gripper_events"], "release_frame": demo.get("release_frame")}
            else:
                grounded = ground_program(program, scene)
                objects, eef = grounded["object_poses"], grounded["eef_poses"]
                attachment = _pose_array(scene["holding"]["T_carrier_eef"], "actual attachment")
                attachment_condition_id = f"{skill_id}:actual_holding"
                conditions.append({"kind": "observed_attachment", "skill_id": skill_id,
                                   "condition_id": attachment_condition_id, "carrier_id": carrier,
                                   "T_carrier_eef": attachment.tolist(),
                                   "source": deepcopy(scene["holding"]), "progress": float(first_phase)})
                last_held = len(objects)-1
            conditions.append({"kind": "predicted_skill_boundary", "skill_id": skill_id,
                               "condition_id": f"{skill_id}:predicted_boundary",
                               "depends_on": previous["condition"] if conditional else attachment_condition_id,
                               "carrier_id": carrier, "initial_pose": scene["initial_pose"].tolist(),
                               "predicted_end_pose": objects[-1].tolist(),
                               "T_carrier_eef": attachment.tolist(), "progress": float(first_phase)})
            release = demo.get("release_frame")
            function_rows = sorted({int(np.flatnonzero(frames == event["frame_index"])[0])
                                    for event in demo["functional_events"]})
            knots = sorted({0, last_held, *function_rows})
            model_value = grounded["motion_checks"].get("model", {"kind": "free"})
            model = _model_from_diagnostics(model_value)
            on = program["task"]["relation"] == "on"
            if on and model is not None:
                raise ValueError("A support endpoint family cannot override a physical mechanism")
            function_frame = np.eye(4)
            if on:
                first_event = demo["functional_events"][0]
                binding = scene["functional_bindings"][first_event["event_id"]]
                function_frame = _pose_array(binding["carrier_mapping"], "carrier mapping") @ _pose_array(first_event["carrier_frame"], "material frame")
            elif model is None and program["task"]["relation"] != "tool_use":
                first_event = demo["functional_events"][0]
                function_frame = _pose_array(scene["functional_bindings"][first_event["event_id"]]["carrier_frame"], "material frame")
            intervals = list(zip(knots[:-1], knots[1:])) or [(0, 0)]
            for begin, end in intervals:
                body = objects[begin:end+1] @ function_frame
                path_attachment = _inverse(function_frame) @ attachment
                coords = None if model is None else model.coordinates[begin:end+1]
                path = _path_from_rows(body, source, model=model, coordinates=coords, attachment=path_attachment)
                family = ({"move_start": begin > 0 or moving_start,
                           "end": "variable" if end in function_rows and end > 0 else "unanchored"}
                          if on else None)
                if not on and moving_start and begin == 0 and model is None and program["task"]["relation"] != "tool_use":
                    family = {"move_start": True, "end": "fixed"}
                append_interval(begin, end, path, endpoint_family=family,
                                metadata={"mode": "held", "carrier_id": carrier,
                                          "attachment": attachment.tolist(), "conditional": conditional,
                                          "attachment_condition_id": attachment_condition_id})
            held_end = float(len(phases))
            if last_held < len(frames)-1:
                # The object ceases to follow the hand at release. Its ordinary
                # retreat geometry is an independent reference, not a payload.
                append_interval(last_held, len(frames)-1, _path_from_rows(eef[last_held:], source),
                                frame_anchor={"progress": held_end, "reference_pose": eef[last_held]},
                                metadata={"mode": "released_retreat", "carrier_id": carrier,
                                          "release_progress": held_end})
            for event in demo["functional_events"]:
                progress = frame_progress[int(event["frame_index"])]
                if on:
                    binding = scene["functional_bindings"][event["event_id"]]
                    face = binding["on_geometry"]
                    supports.append(SupportRequirement(progress, _inverse(attachment), scene["body_points"],
                                                       face["carrier_region_points"], face["carrier_outward_normal"],
                                                       _surface(scene["support"]), face["gap_tolerance"],
                                                       face["normal_tolerance"], scene.get("margin", 0.), binding["source"]))
                elif event.get("constraint_kind") != "joint_coordinate":
                    binding = scene["functional_bindings"][event["event_id"]]
                    material = _pose_array(binding["carrier_frame"], "current material function frame")
                    check = next(check for check in grounded["function_checks"] if check["event_id"] == event["event_id"])
                    pose_events.append(PoseEventRequirement(
                        progress, _inverse(attachment) @ material, np.asarray(check["goal_pose"]) @ material,
                        event["position_tolerance_m"], event["rotation_tolerance_rad"], event["source"]))
                else:
                    check = next(check for check in grounded["function_checks"] if check["event_id"] == event["event_id"])
                    coordinate_events.append(CoordinateEventRequirement(
                        progress, _inverse(attachment), np.asarray(check["goal_pose"]), model.axis,
                        model.kind, float(np.asarray(check["target_coordinates"])[0]),
                        event["coordinate_tolerance"], event["source"]))
                if program["task"]["relation"] == "inside" and "interior" in scene:
                    interior = _interior(scene["interior"])
                    interiors.append(InteriorRequirement(progress, progress, _inverse(attachment), scene["body_points"],
                                                         interior.halfspaces, scene.get("margin", 0.), interior.source))
                events.append(PlanningEvent(event["event_id"], skill_id, progress, int(event["frame_index"]),
                                            "function", event["source"]))
            if "entry_region" in scene:
                interior = _interior(scene["entry_region"])
                begin = frame_progress[int(frames[scene["entry_start_index"]])]
                interiors.append(InteriorRequirement(begin, held_end, _inverse(attachment), scene["body_points"],
                                                     interior.halfspaces, scene.get("margin", 0.), interior.source))
            if release is not None:
                if spec.get("barrier_can_stop") is not True:
                    raise ValueError("Release verification requires an explicitly permitted stop/hold boundary")
                events.append(PlanningEvent("__release_verified__", skill_id, held_end, int(release),
                                            "verification", source, "actual_release_from_current_feedback", True))
                conditions.append({"kind": "predicted_release", "carrier_id": carrier,
                                   "condition_id": f"{skill_id}:actual_release",
                                   "progress": held_end, "T_carrier_eef": attachment.tolist(),
                                   "nominal_pose": objects[-1].tolist(), "requires_evidence": "actual_release_from_current_feedback"})
            frame_anchor = ({"progress": held_end, "reference_pose": eef[last_held].copy()}
                            if release is not None else None)
            predicted_objects[carrier] = {"pose": objects[-1].copy(), "frame_anchor": frame_anchor}
            previous = {"carrier_id": carrier, "attachment": None if release is not None else attachment,
                        "object_pose": objects[-1].copy(), "eef_pose": eef[-1].copy(),
                        "attachment_condition_id": None if release is not None else attachment_condition_id,
                        "condition": f"{skill_id}:predicted_boundary", "variable_pose": on}
        else:
            raise ValueError("Only compiled grasp/manipulation programs define these local windows")
        for event in grounded["gripper_events"]:
            events.append(PlanningEvent(event["event_id"], skill_id, frame_progress[int(event["frame_index"])],
                                        int(event["frame_index"]), event["command"],
                                        {"original_event": deepcopy(event), **source}))
    # BEGIN exact collision-attachment spans shared by all held phases.
    # A skill boundary is not a release. Consecutive manipulation phases with
    # the same actual/candidate relation share one span until an evidenced
    # release boundary or the end of the current planning window.
    spans = []
    for index, phase in enumerate(phases):
        metadata = phase.metadata
        if metadata.get("mode") != "held":
            continue
        key = (metadata["carrier_id"], metadata["attachment_condition_id"])
        if spans and spans[-1]["key"] == key and spans[-1]["end"] == index:
            span = spans[-1]
            if not np.array_equal(span["attachment"], metadata["attachment"]):
                raise ValueError("A shared Holding condition cannot change its rigid attachment")
            span["end"] = index+1
            span["phases"].append(phase)
        else:
            spans.append({"key": key, "start": index, "end": index+1,
                          "attachment": metadata["attachment"], "phases": [phase]})
    for span in spans:
        releases = [item for item in conditions if item["kind"] == "predicted_release"
                    and item["carrier_id"] == span["key"][0] and item["progress"] == span["end"]]
        if len(releases) > 1 or (not releases and span["end"] != len(phases)):
            raise ValueError("An attachment can end only at its actual release boundary or the window end")
        release_id = releases[0]["condition_id"] if releases else None
        for phase in span["phases"]:
            phase.metadata["attachment_interval"] = [float(span["start"]), float(span["end"])]
            phase.metadata["release_condition_id"] = release_id
    # END exact collision-attachment spans shared by all held phases.
    return PlanningProblem(tuple(joint_names), start_q, start_dq, tool_link, reference_frame,
                           base_pose, tool_transform, tuple(phases), tuple(events), tuple(couplings),
                           numerical, tuple(conditions), tuple(supports), tuple(interiors),
                           tuple(pose_events), tuple(coordinate_events))
# END shared skill-process constraints for generation and verification.
