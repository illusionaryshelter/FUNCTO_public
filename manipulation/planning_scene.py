"""Declared scene entities, task features, observed free cells and hand commands.

Entity geometry is a fixed observed field in its local coordinate frame.
PosePath and Attachment supply its predicted rigid motion in the planning
frame. ContactRule retains task features in their declared coordinates.
"""

# BEGIN observed entity data replace sphere and completed-body interfaces.
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import NamedTuple
import numpy as np
import torch
from .planning_problem import PosePath, quaternion_conjugate, quaternion_multiply, quaternion_rotate, event_occurrence_key
from .optimizer_device_checks import constant_tensor

def _array(value, shape, label):
    result = np.asarray(value, dtype=float)
    if (result.shape != shape or not np.isfinite(result).all()):
        raise ValueError(f"{label} must have finite shape {shape}")
    return result.copy()


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must identify its actual source")
    return value


def _tensor(value, reference):
    # BEGIN graph-scoped constants use exactly the same as_tensor conversion.
    return constant_tensor(value, reference)


def _interval(progress, interval):
    return (progress >= interval[0]) & (progress <= interval[1])


def _valid_interval(value):
    result = _array(value, (2,), "progress interval")
    if result[0] < 0 or result[1] <= result[0]:
        raise ValueError("A progress interval must be ordered and nonnegative")
    return tuple(result)


def _world_points(points, position, quaternion):
    return (quaternion_rotate(quaternion.unsqueeze(-2), points)
            + position.unsqueeze(-2))


class ObservedSpace:
    """Immutable union of cells certified free by calibrated depth rays.

    ``known[z,y,x]`` identifies the closed voxel centred at ``origin + h*[x,y,z]``.
    This evidence is queried for moving scene entities. The robot view policy
    is recorded separately by OnlinePlanningScene.
    """
    def __init__(self, known, *, origin, voxel_size, observation_ids, reference_frame):
        mask = np.asarray(known)
        if mask.dtype != np.bool_ or mask.ndim != 3 or min(mask.shape) < 2 or mask.size > 8_000_000:
            raise ValueError("Observed free cells require a bounded three-dimensional boolean grid")
        self.origin = _array(origin, (3,), "observed grid origin")
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("Observed cell spacing must be finite positive metres")
        self.voxel_size = float(voxel_size)
        self.observation_ids = tuple(_text(row, "observation ID") for row in observation_ids)
        if not self.observation_ids:
            raise ValueError("Observed free cells require their view identities")
        self.reference_frame = _text(reference_frame, "observation reference frame")
        self._known = mask.copy()
        self._known.flags.writeable = False

    @property
    def known_free(self):
        result = self._known.copy()
        result.flags.writeable = False
        return result

    def clone_for_rollout(self):
        """Return the fixed evidence shared by all candidate trajectories."""
        return self


@dataclass
class Attachment:
    """Conditional rigid attachment over [start,end), never a Holding claim.

    The owner's external PosePath supplies its observed pose before attachment.
    After release the owner remains at the same rollout's actual release FK,
    preserving the optimization variables shared with subsequent constraints.
    An open command alone creates no release.
    """

    interval: tuple
    position_tool_owner: object
    quaternion_tool_owner: object
    condition_id: str
    release_condition_id: str | None

    def __post_init__(self):
        self.interval = _valid_interval(self.interval)
        self.position_tool_owner = _array(self.position_tool_owner, (3,), "attachment translation")
        self.quaternion_tool_owner = _array(self.quaternion_tool_owner, (4,), "attachment quaternion")
        if not np.isclose(np.linalg.norm(self.quaternion_tool_owner), 1., atol=1e-6):
            raise ValueError("Attachment quaternion must be normalized wxyz")
        _text(self.condition_id, "conditional attachment evidence ID")
        # None means Holding continues beyond the planned window. The scene/
        # problem binding below permits this only at the final window boundary.
        if self.release_condition_id is not None:
            _text(self.release_condition_id, "release verification condition ID")


@dataclass
class CollisionOwner:
    """Observed local entity and its declared motion in the planning frame."""
    owner_id: str
    query: object
    pose_path: PosePath
    source: str
    moving: bool = False
    attachments: tuple[Attachment, ...] = ()
    pose_interval: tuple = (0., 1.)

    def __post_init__(self):
        _text(self.owner_id, "owner identity")
        _text(self.source, "owner observation source")
        if self.query.owner_ids != frozenset((self.owner_id,)):
            raise ValueError("An entity field must identify exactly its owner")
        if type(self.pose_path) is not PosePath or self.pose_path.parent is not None:
            raise ValueError("Entity motion requires a serializable world-frame PosePath")
        self.pose_interval = _valid_interval(self.pose_interval)
        if type(self.moving) is not bool:
            raise TypeError("Entity motion must be explicitly declared")
        self.attachments = tuple(self.attachments)
        for index, relation in enumerate(self.attachments):
            if not isinstance(relation, Attachment):
                raise TypeError("Rigid motion relations must be Attachment records")
            if index and relation.interval[0] < self.attachments[index-1].interval[1]:
                raise ValueError("Attachment intervals must be ordered without overlap")
            if index and self.attachments[index-1].release_condition_id is None:
                raise ValueError("A new attachment requires the previous release declaration")
        if self.attachments:
            self.moving = True

    def pose(self, position, quaternion, progress, released_poses=()):
        predicted_p, predicted_q = self.predicted_pose(progress)
        if len(released_poses) != len(self.attachments):
            raise ValueError("Every attachment span needs its release state or explicit continued-Holding marker")
        for relation, release in zip(self.attachments, released_poses):
            held_p = position + quaternion_rotate(quaternion, _tensor(relation.position_tool_owner, position))
            held_q = quaternion_multiply(quaternion, _tensor(relation.quaternion_tool_owner, quaternion))
            # BEGIN a window endpoint is not a physical release.
            if relation.release_condition_id is None:
                held = progress >= relation.interval[0]
                predicted_p = torch.where(held[..., None], held_p, predicted_p)
                predicted_q = torch.where(held[..., None], held_q, predicted_q)
                continue
            # END continued Holding follows each current FK, including plateaus.
            held = (progress >= relation.interval[0]) & (progress < relation.interval[1])
            released = progress >= relation.interval[1]
            predicted_p = torch.where(released[..., None], release[0],
                                      torch.where(held[..., None], held_p, predicted_p))
            predicted_q = torch.where(released[..., None], release[1],
                                      torch.where(held[..., None], held_q, predicted_q))
        return predicted_p, predicted_q


    def predicted_pose(self, progress):
        start, end = self.pose_interval
        return self.pose_path.sample(((progress - start) / (end - start)).clamp(0, 1))




@dataclass
class ContactRule:
    """Declared source feature and finite target patch in their entity frames.

    Task points express allowed contact geometry, not an occupied body. A
    robot feature identifies its registered body explicitly. Owner feature
    points are entity-local. Robot feature points use TCP coordinates at zero
    finger displacement; feature_joint_translation maps the named prismatic
    coordinates into that same TCP chart. Gap and normal limits retain their
    task meaning independently of separation margins.
    """
    rule_id: str
    actor: str
    target_owner: str
    interval: tuple
    material_id: str
    feature_points: object
    feature_normal: object
    patch_origin: object
    patch_axes: object
    patch_half_size: object
    gap_bounds: tuple
    normal_tolerance_rad: float
    modeling_allowance_m: float
    geometry_error_m: float
    source: str
    required: bool = False
    feature_joint_translation: object = None
    numerical_position_tolerance_m: float = 1e-6
    numerical_angle_tolerance_rad: float = 1e-6
    feature_kind: str = "polygon"
    robot_body: str | None = None

    def __post_init__(self):
        for name in ("rule_id", "actor", "target_owner", "material_id", "source"):
            _text(getattr(self, name), name)
        self.interval = _valid_interval(self.interval)
        points = np.asarray(self.feature_points, float)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or self.feature_kind not in {"polygon", "edge"}
                or (self.feature_kind == "edge" and len(points) != 2)
                or (self.feature_kind == "polygon" and len(points) < 3)):
            raise ValueError("A finite task feature needs its original edge or polygon points")
        self.feature_points = points.copy()
        self.feature_normal = _array(self.feature_normal, (3,), "feature normal")
        if not np.isclose(np.linalg.norm(self.feature_normal), 1., atol=1e-6):
            raise ValueError("A task feature normal must have unit length")
        if self.feature_kind == "edge":
            delta = points[1]-points[0]
            length = np.linalg.norm(delta)
            if length == 0:
                raise ValueError("An edge needs distinct source endpoints")
            tangent = delta/length
            lateral = np.cross(self.feature_normal, tangent)
            lateral /= np.linalg.norm(lateral)
            self._feature_tangent = tangent
            self._feature_edge_normals = np.stack((tangent, -tangent, lateral, -lateral))
            self._feature_edge_offsets = np.array([-length, 0., 0., 0.])
        else:
            from scipy.spatial import ConvexHull
            basis = np.linalg.svd(points-points[0], full_matrices=True)[2][:2].T
            edges = ConvexHull((points-points[0]) @ basis).equations
            self._feature_edge_normals = edges[:, :2] @ basis.T
            self._feature_edge_offsets = edges[:, 2]
        self.patch_origin = _array(self.patch_origin, (3,), "patch origin")
        self.patch_axes = _array(self.patch_axes, (3, 3), "patch axes")
        self.patch_half_size = _array(self.patch_half_size, (2,), "patch half size")
        self.gap_bounds = tuple(_array(self.gap_bounds, (2,), "gap bounds"))
        if (not np.allclose(self.patch_axes.T @ self.patch_axes, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.patch_axes), 1., atol=1e-6)
                or np.any(self.patch_half_size <= 0) or self.gap_bounds[0] > self.gap_bounds[1]
                or not 0 <= self.normal_tolerance_rad < math.pi/2
                or not 0 <= self.modeling_allowance_m <= self.geometry_error_m):
            raise ValueError("Contact patch, task limits and model allowance must be explicit")
        if self.feature_joint_translation is not None:
            mapping = np.asarray(self.feature_joint_translation, float)
            if (mapping.ndim != 3 or mapping.shape[:2] != points.shape
                    or not np.isfinite(mapping).all()
                    or not np.allclose(mapping, mapping[:1], atol=1e-10, rtol=0)):
                raise ValueError("Prismatic features require one rigid translation map")
            self.feature_joint_translation = mapping.copy()
        if self.actor == "robot":
            _text(self.robot_body, "contact robot body")
        if (not np.isfinite([self.numerical_position_tolerance_m,
                             self.numerical_angle_tolerance_rad]).all()
                or min(self.numerical_position_tolerance_m,
                       self.numerical_angle_tolerance_rad) < 0):
            raise ValueError("Numerical contact tolerances must be finite and nonnegative")


def grasp_contact_rule_ids(condition, contacts):
    """Validate the finite rules selected by one explicit grasp activation.

    ``contacts`` contains serialized ContactRule records. The condition names
    their existing identities; it cannot change material geometry or enlarge
    owner/interval scope. Ordinary material rules without a grasp activation
    keep their independently declared semantics.
    """
    identities = condition.get("contact_rule_ids")
    if (not isinstance(identities, (list, tuple)) or not identities
            or any(not isinstance(value, str) or not value.strip() for value in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError("Finite grasp activation needs nonempty unique contact_rule_ids")
    if condition.get("kind") != "grasp_contact_activation":
        raise ValueError("Finite grasp rules require an explicit grasp contact activation")
    _text(condition.get("condition_id"), "finite grasp condition ID")
    _text(condition.get("carrier_id"), "finite grasp target owner")
    event = condition.get("activation_event", {})
    event_occurrence_key(event)
    arrival = condition.get("arrival_goal_id", ())
    if (event.get("kind") != "close" or len(arrival) != 2 or arrival[0] != "pose_events"
            or type(arrival[1]) is not int or arrival[1] < 0 or not condition.get("source")):
        raise ValueError("Finite grasp activation needs its original close, pose arrival and source")
    interval = _valid_interval(condition.get("interval", ()))
    rows = {row["rule_id"]: row for row in contacts}
    if len(rows) != len(contacts):
        raise ValueError("Finite contact rule identities must be unique")
    for identity in identities:
        rule = rows.get(identity)
        if (rule is None or rule.get("actor") != "robot"
                or rule.get("target_owner") != condition["carrier_id"]
                or tuple(rule.get("interval", ())) != interval):
            raise ValueError("Finite grasp rule must match the robot actor, target owner and exact activation interval")
    return tuple(identities)


def validate_grasp_contact_arrival(condition, requirement):
    """Require the exact owner/event provenance of the frozen arrival target.

    Equal progress and a valid collection index are insufficient: another
    owner's pose at that instant must never authorize these material rules.
    """
    source = requirement.source
    if (not isinstance(source, Mapping) or source.get("kind") != "frozen_perception_grasp_arrival"
            or source.get("carrier_id") != condition["carrier_id"]
            or source.get("activation_event") != condition["activation_event"]):
        raise ValueError("Finite grasp arrival must retain the exact target owner and original close source")


class GripperState(NamedTuple):
    """Coordinates/envelope and certainty shared by every collision consumer.

    ``uncertain`` identifies unmodelled future finger positions. A zero-width
    command reference is not evidence that this flag is false. ``half`` is the
    mechanical interval used only for conservative conditional diagnostics.
    """
    coordinates: torch.Tensor
    half: torch.Tensor
    controller_residual: torch.Tensor
    uncertain: torch.Tensor


class PrismaticGripper:
    """Bind native finger coordinates to a supported controller and geometry.

    Position tracking uses explicitly modeled solver coordinates. Timed
    actuator commands and optional nominal future geometry share the bound
    elapsed clock but remain separate objects. Binary control without a future
    model retains its mechanical envelope only as unresolved diagnostic
    evidence. A sampled width or command is never measured future contact.

    ``joint_bounds(progress)`` is the solver tracking contract. Independent
    ``finger_trajectory_factory`` supplies commands;
    ``nominal_finger_trajectory_factory(duration, progress_map, command_reference)``
    supplies declared future geometry from this binding's exact command.
    ``control_timestep_s`` is the explicit native device period shared by
    every command clock; it is not inferred from either trajectory factory.
    Without that geometry, only the actual initial fingers and invariant
    robot components contribute hard collision claims. Actual execution always
    checks the measured complete state. ``coupling_matrix`` and
    ``coupling_offset`` describe the native A q = b drive constraint.
    """

    def __init__(self, *, controller, joint_names, current_positions, joint_limits,
                 joint_bounds, coupling_matrix, coupling_offset,
                 commands, source, control_timestep_s, numerical_joint_tolerance=1e-5,
                 finger_trajectory_factory=None, nominal_finger_trajectory_factory=None):
        # BEGIN independent position-control intervals without solver fingers.
        if controller not in {"position_tracking", "timed_reference", "binary"}:
            raise ValueError("Controller must declare solver tracking, a timed reference or binary intentions")
        # END independent position-control intervals without solver fingers.
        self.controller = controller
        # BEGIN command quantization belongs to the device, not a factory type.
        if (isinstance(control_timestep_s, bool) or not np.isscalar(control_timestep_s)
                or not np.isfinite(control_timestep_s) or control_timestep_s <= 0):
            raise ValueError("Gripper commands require an explicit positive native control timestep")
        self.control_timestep_s = float(control_timestep_s)
        # END position tracking and independently commanded fingers share this clock.
        self.joint_names = tuple(_text(name, "native gripper joint") for name in joint_names)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Native gripper joint names must be unique")
        count = len(self.joint_names)
        self.current_positions = _array(current_positions, (count,), "current native gripper feedback")
        self.joint_limits = _array(joint_limits, (count, 2), "native gripper limits")
        if (np.any(self.joint_limits[:, 0] > self.joint_limits[:, 1])
                or np.any(self.current_positions < self.joint_limits[:, 0])
                or np.any(self.current_positions > self.joint_limits[:, 1])):
            raise ValueError("Current finger feedback must respect native joint limits")
        if ((controller == "position_tracking" and not callable(joint_bounds))
                or (controller != "position_tracking" and joint_bounds is not None)):
            raise TypeError("Only solver position tracking consumes progress-indexed joint bounds")
        if ((controller == "timed_reference") != callable(finger_trajectory_factory)):
            raise TypeError("Timed fingers require their explicit physical-clock trajectory factory")
        self.finger_trajectory_factory = finger_trajectory_factory
        self.finger_trajectory = None
        # BEGIN commands and a contact-conditioned future geometry are separate.
        if nominal_finger_trajectory_factory is not None and not callable(nominal_finger_trajectory_factory):
            raise TypeError("Nominal hand geometry requires an explicit trajectory factory")
        self.nominal_finger_trajectory_factory = nominal_finger_trajectory_factory
        self.nominal_finger_trajectory = None
        # END no position command is promoted to actual future contact width.
        self.joint_bounds = joint_bounds
        if not np.isfinite(numerical_joint_tolerance) or numerical_joint_tolerance < 0:
            raise ValueError("The numerical joint tolerance must be a finite nonnegative value")
        self.numerical_joint_tolerance = float(numerical_joint_tolerance)
        coupling = np.asarray(coupling_matrix, float)
        if coupling.ndim != 2 or coupling.shape[1] != count or not np.isfinite(coupling).all():
            raise ValueError("The actual native drive coupling must be supplied")
        self.coupling_matrix = coupling.copy()
        self.coupling_offset = _array(coupling_offset, (len(coupling),), "drive coupling offset")
        # Only independent actuator coordinates belong in the scene's OMPL
        # projection. The problem owns A q=b, so repeating those rows would
        # make the combined equality Jacobian rank deficient.
        rank = np.linalg.matrix_rank(coupling) if len(coupling) else 0
        self._drive_basis = (np.linalg.svd(coupling, full_matrices=True)[2][rank:]
                             if len(coupling) else np.eye(count))
        if len(coupling) and not np.allclose(coupling @ self.current_positions, self.coupling_offset, atol=1e-8):
            raise ValueError("Current feedback must satisfy the declared drive mapping")
        # BEGIN finger geometry uses full robot FK and its registered fields.
        # This object owns only command capability and the named joint curve.
        # END no calibrated collision surrogate is part of the hand controller.
        # BEGIN own each bound occurrence so later retry input cannot alter this scene.
        self.commands = tuple(deepcopy(commands))
        # BEGIN arm-only planning still binds measured fingers without issuing a command.
        # An explicitly empty event list is valid. validate_problem below
        # requires an exact match whenever an actual hand action is requested;
        # optional nominal geometry never invents an actuator event.
        # END source geometry and command occurrence identity remain separate.
        for event in self.commands:
            _text(event.get("event_id"), "gripper event ID")
            _text(event.get("skill_id"), "gripper skill instance ID")
            if event.get("command") not in {"open", "close", "hold"} or not np.isfinite(event.get("progress", np.nan)):
                raise ValueError("Gripper events need native command intentions and progress")
        identities = [event_occurrence_key(event) for event in self.commands]
        if len(set(identities)) != len(identities):
            raise ValueError("Gripper event occurrence identities must be unique")
        # END the same original event may recur only under an explicit new occurrence.
        self.source = _text(source, "gripper calibration and timing source")

    # BEGIN a separately controlled hand follows the same formal physical clock.
    def bind_clock(self, duration, progress_map):
        if self.controller == "position_tracking":
            return
        if self.controller == "timed_reference":
            trajectory = self.finger_trajectory_factory(duration, progress_map)
            if (tuple(trajectory.joint_names) != self.joint_names
                    or trajectory.duration != duration
                    or not np.array_equal(trajectory.sample(np.array([0.]))[0][0], self.current_positions)):
                raise ValueError("Finger reference must preserve named measured entrance and formal duration")
            self.finger_trajectory = trajectory
        if self.nominal_finger_trajectory_factory is not None:
            nominal = self.nominal_finger_trajectory_factory(duration, progress_map, self.finger_trajectory)
            # BEGIN future geometry must declare its own physical validity.
            from .gripper_trajectory import NominalFingerGeometry
            if (not isinstance(nominal, NominalFingerGeometry)
                    or tuple(nominal.joint_names) != self.joint_names or nominal.duration != duration
                    or not np.array_equal(nominal.sample(np.array([0.]))[0][0], self.current_positions)):
                raise ValueError("Nominal hand geometry must preserve measured entrance, names and bound clock")
            self.nominal_finger_trajectory = nominal
            # END no unscoped command reference can silently become geometry.

    def state(self, progress, joint_position, joint_names, *, trajectory_time=None, measured=False):
        expected = (*progress.shape, len(self.joint_names))
        limits = _tensor(self.joint_limits, progress)
        initial_positions = _tensor(self.current_positions, progress)
        certain = torch.zeros_like(progress, dtype=torch.bool)
        # BEGIN complete measured feedback always overrides every future model.
        if measured:
            if joint_position is None or not set(self.joint_names) <= set(joint_names or ()):
                raise ValueError("Measured hand geometry requires complete named finger coordinates")
            columns = [tuple(joint_names).index(name) for name in self.joint_names]
            coordinates = joint_position[..., constant_tensor(columns, joint_position, dtype=torch.long)]
            if coordinates.shape != expected:
                raise ValueError("Measured fingers must align with the queried states")
            tolerance = self.numerical_joint_tolerance
            violations = (torch.relu(limits[:, 0]-coordinates-tolerance).sum(-1)
                          + torch.relu(coordinates-limits[:, 1]-tolerance).sum(-1))
            if len(self.coupling_matrix):
                coupling = _tensor(self.coupling_matrix, progress)
                violations += torch.relu((coordinates @ coupling.T-_tensor(self.coupling_offset, progress)).abs()
                                         -tolerance*coupling.abs().sum(-1)).sum(-1)
            return GripperState(coordinates, torch.zeros_like(coordinates), violations, certain)
        # END missing measured coordinates never fall back to commands.
        # BEGIN a mixed batch keeps modeled and contact-dependent times separate.
        modeled = certain
        if self.controller in {"binary", "timed_reference"}:
            # BEGIN binary progress cannot identify the measured initial instant.
            # Neither a repeated zero-progress sample nor an isolated IK/search
            # query carries a time/velocity contract for the fingers. Preserve
            # the full continuous mechanical interval at every queried state;
            # current_positions remains the separately checked real feedback.
            lower = limits[:, 0].expand(expected)
            upper = limits[:, 1].expand(expected)
            if trajectory_time is not None:
                initial = trajectory_time == 0
                if trajectory_time.shape != progress.shape:
                    raise ValueError("Finger physical times must align with the query")
                lower = torch.where(initial[..., None], initial_positions, lower)
                upper = torch.where(initial[..., None], initial_positions, upper)
            uncertain = (torch.ones_like(progress, dtype=torch.bool) if trajectory_time is None
                         else trajectory_time > 0)
            if self.nominal_finger_trajectory is not None and trajectory_time is not None:
                model = self.nominal_finger_trajectory
                modeled = model.valid_mask_torch(trajectory_time)
                coordinates, _ = model.sample_torch(torch.where(
                    modeled, trajectory_time, torch.zeros_like(trajectory_time)))
                if coordinates.shape != expected:
                    raise ValueError("Named finger samples must align with the scene query")
                lower = torch.where(modeled[..., None], coordinates, lower)
                upper = torch.where(modeled[..., None], coordinates, upper)
                uncertain = uncertain & ~modeled
            # END time-independent binary envelope.
        else:
            lower, upper = self.joint_bounds(progress)
            uncertain = certain
        if lower.shape != expected or upper.shape != expected:
            raise ValueError("Reachable finger bounds must have shape [B,H,J]")
        tolerance = self.numerical_joint_tolerance
        violations = (torch.relu(lower - upper - tolerance).sum(-1)
                      + torch.relu(limits[:, 0] - lower - tolerance).sum(-1)
                      + torch.relu(upper - limits[:, 1] - tolerance).sum(-1))
        middle, half = (lower + upper) / 2, (upper - lower) / 2
        if self.controller == "position_tracking":
            if joint_position is None or joint_names is None:
                raise ValueError("Tracking fingers require actual named solver joints")
            indices = [tuple(joint_names).index(name) for name in self.joint_names]
            # BEGIN keep named-column indices at stable device addresses.
            coordinates = joint_position[..., constant_tensor(indices, joint_position, dtype=torch.long)]
            # END only index storage changes, never its order or selected values.
            # Tracking means a single executable command, not optimizer-selected width.
            violations = violations + torch.relu((upper - lower).abs() - tolerance).sum(-1)
            violations = violations + torch.relu((coordinates - middle).abs() - tolerance).sum(-1)
            half = torch.zeros_like(half)
        else:
            coordinates = middle
        initial = ((progress == 0) if self.controller == "position_tracking" else
                   (certain if trajectory_time is None else trajectory_time == 0)).to(progress.dtype)
        violations = violations + initial * (torch.relu(lower - initial_positions - tolerance).sum(-1)
                                             + torch.relu(initial_positions - upper - tolerance).sum(-1))
        if len(self.coupling_matrix) and self.controller == "position_tracking":
            coupling = _tensor(self.coupling_matrix, progress)
            coupling_tolerance = tolerance * coupling.abs().sum(-1)
            violations = violations + torch.relu((coordinates @ coupling.T
                                         - _tensor(self.coupling_offset, progress)).abs()
                                         - coupling_tolerance).sum(-1)
        elif len(self.coupling_matrix):
            coupling = _tensor(self.coupling_matrix, progress)
            violations += modeled.to(progress.dtype) * torch.relu(
                (coordinates @ coupling.T-_tensor(self.coupling_offset, progress)).abs()
                - tolerance*coupling.abs().sum(-1)).sum(-1)
        # END an unbound search query stays unknown even at progress zero.
        return GripperState(coordinates, half, violations, uncertain)

# END geometry queries reside in OnlinePlanningScene.
