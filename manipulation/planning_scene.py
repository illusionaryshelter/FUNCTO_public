"""Shared robot/owner identities and ordinary interaction-scene records.

All poses and spheres use the declared robot/world planning frame, metres, and
scalar-first quaternions. FK belongs to cuRobo: this module consumes its actual
robot spheres and tool poses without reconstructing a robot from task names.
An owner query contains geometry in that owner's local frame. A background
query contains only the explicitly separated static background.

The queries are sphere-proxy collision models, not complete contact physics or
continuous-curve certificates. Observation coverage, material contact features,
controller capabilities and conservative sphere coverage are explicit inputs.
"""

# BEGIN observed scene, moving-owner queries and bounded contact permissions.
from __future__ import annotations

from copy import copy, deepcopy
from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .planning_problem import (
    PosePath, quaternion_conjugate, quaternion_multiply, quaternion_rotate,
)
# BEGIN scene commands and planner events share exact occurrence identities.
from .planning_problem import event_occurrence_key
# END source names alone do not identify an explicitly repeated operation.
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
    # END eager queries keep their ordinary device-following tensor creation.


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


def _local_spheres(spheres, position, quaternion):
    points = quaternion_rotate(quaternion_conjugate(quaternion).unsqueeze(-2),
                               spheres[..., :3] - position.unsqueeze(-2))
    return torch.cat((points, spheres[..., 3:4]), dim=-1)


def _world_spheres(spheres, position, quaternion):
    local = _tensor(spheres, position)
    points = _world_points(local[..., :3], position, quaternion)
    radii = local[..., 3:4].expand(*points.shape[:-1], 1)
    return torch.cat((points, radii), dim=-1)


# BEGIN bounded batched hand-pair penalties with unchanged sum and derivatives.
# BEGIN size pair batches by live tensor elements instead of a tiny pair cap.
_SELF_PAIR_STATE_ELEMENTS = 1_048_576


def _self_pair_chunk_size(states, pairs):
    """Bound expansion while evaluating every declared pair exactly once.

    Larger chunks change floating sum grouping, not the nonnegative pair
    terms or their mathematical zero set. One pair is the irreducible chunk
    when the state batch itself exceeds the element target.
    """
    return min(pairs, max(1, _SELF_PAIR_STATE_ELEMENTS // states)) if pairs else 0
# END no pair is discarded and checkpointing still bounds backward scratch.


def _pair_penetration(spheres, pairs):
    """Sum the original positive linear penetrations of these explicit pairs.

    Native cuRobo self-collision differentiates a maximum over pairs; it cannot
    replace this sum without changing the optimizer objective. Torch's norm
    preserves its zero-distance derivative, without artificial clearance.
    """
    first = spheres.index_select(-2, pairs[:, 0])
    second = spheres.index_select(-2, pairs[:, 1])
    distance = torch.linalg.vector_norm(first[..., :3] - second[..., :3], dim=-1)
    return torch.relu(first[..., 3] + second[..., 3] - distance).sum(-1)


def _self_pair_sum(spheres, pair_indices):
    """Evaluate every pair in bounded chunks, recomputing saved intermediates.

    Non-reentrant checkpointing uses Torch's own derivatives while avoiding
    one retained [B,H,P,3] graph for every chunk. Inputs are immutable during
    native backward; pair indices are explicit arguments, not loop closures.
    All costs are nonnegative, so a zero sum means every declared pair passed.
    """
    total = spheres.new_zeros(spheres.shape[:2])
    if not len(pair_indices):
        return total
    # BEGIN keep the original integer pair order without repeated host copies.
    pairs = constant_tensor(pair_indices, spheres, dtype=torch.long)
    # END the declared pair order is preserved within memory-bounded chunks.
    step = _self_pair_chunk_size(math.prod(spheres.shape[:2]), len(pairs))
    gradients = torch.is_grad_enabled() and spheres.requires_grad
    for begin in range(0, len(pairs), step):
        selected = pairs[begin:begin+step]
        value = (checkpoint(_pair_penetration, spheres, selected,
                            use_reentrant=False, preserve_rng_state=False)
                 if gradients else _pair_penetration(spheres, selected))
        total = total + value
    return total
# END bounded batched hand-pair penalties with unchanged sum and derivatives.


class CuroboSphereQuery:
    """Use V2's actual differentiable query with bounded, private work buffers.

    ``scene_collision`` is a cuRobo ``SceneCollision``. It is constructed by
    the caller under its GPU budget; importing this module creates no device
    context. ``owner_ids`` identifies every physical owner in this partition,
    including owners fused into a background voxel field. It cannot omit an
    active object merely because the obstacle was given another mesh name.

    V2 returns nonnegative collision *cost*, not free-space signed distance.
    Activation zero therefore reports penetration of the supplied proxy.
    V2 saves its mutable gradient buffer for backward. The saved-tensor hook
    snapshots that buffer and the output is cloned: a later query cannot
    overwrite a still-live autograd graph. Each rollout also gets its own
    finite cache via ``clone_for_rollout``.
    """

    # BEGIN bound complete validation for the admitted robot geometry capacity.
    # Three million sphere queries fit the existing 64-MiB buffer at 20 bytes
    # per sphere plus scalar weights. This also admits 1536 validation times
    # at all three mode sides for the bounded 640-sphere robot capacity. The
    # planner separately bounds every rollout's retained/live allocations;
    # cumulative cache bytes and semantic slots remain checked below.
    def __init__(self, scene_collision, *, owner_ids, source,
                 max_query_spheres=3_000_000, max_buffers=64,
                 max_cache_bytes=64 * 1024 * 1024):
        # END this storage limit changes no geometry, sampling or query values.
        if not callable(getattr(scene_collision, "get_sphere_distance_raw", None)):
            raise TypeError("A cuRobo V2 SceneCollision query is required")
        self.scene_collision = scene_collision
        self.owner_ids = frozenset(_text(item, "owner_id") for item in owner_ids)
        self.source = _text(source, "collision geometry source")
        if (type(max_query_spheres) is not int or max_query_spheres <= 0
                or type(max_buffers) is not int or max_buffers <= 0
                or type(max_cache_bytes) is not int or max_cache_bytes <= 0):
            raise ValueError("Collision workspaces require explicit positive bounds")
        self.max_query_spheres, self.max_buffers = max_query_spheres, max_buffers
        self.max_cache_bytes = max_cache_bytes
        self._cache_bytes = 0
        self._buffers = {}

    def clone_for_rollout(self):
        result = copy(self)
        result._buffers = {}
        result._cache_bytes = 0
        return result

    def __call__(self, spheres, *, key):
        if spheres.ndim != 4 or spheres.shape[-1] != 4:
            raise ValueError("Collision spheres must have shape [B,H,S,4]")
        if spheres.dtype != torch.float32:
            raise ValueError("The installed cuRobo sphere kernel requires float32")
        if math.prod(spheres.shape[:-1]) > self.max_query_spheres:
            raise ValueError("Collision query exceeds the declared sphere budget")
        # Only this lazy import reaches the installed cuRobo implementation.
        from curobo._src.geom.collision.buffer_collision import CollisionBuffer

        shape = tuple(spheres.shape)
        # BEGIN one current shape per semantic query, with owned saved gradients.
        # A rollout can change its validation horizon. Old shape buffers have
        # no future consumer: every outstanding backward owns the clone below.
        # Replace them before allocating the new shape, rather than retaining
        # one GPU allocation for every historical horizon in this request.
        slot = (key, str(spheres.device))
        byte_count = math.prod(shape[:-1]) * 5 * 4 + 8
        previous = self._buffers.get(slot)
        previous_bytes = 0 if previous is None else math.prod(previous[0][:-1]) * 5 * 4 + 8
        if self._cache_bytes-previous_bytes+byte_count > self.max_cache_bytes:
            raise ValueError("Collision workspaces exceed the total query cache byte limit")
        if previous is not None and previous[0] != shape:
            del self._buffers[slot]
            self._cache_bytes -= previous_bytes
            del previous
        if slot not in self._buffers:
            if len(self._buffers) >= self.max_buffers:
                raise ValueError("Collision query exceeds its bounded workspace cache")
            config = self.scene_collision.device_cfg
            if not config.is_same_torch_device(spheres.device):
                raise ValueError("Query and cuRobo scene must share a device")
            self._buffers[slot] = (
                shape, CollisionBuffer.from_shape(spheres.shape, config),
                spheres.new_ones(1), spheres.new_zeros(1))
            self._cache_bytes += byte_count
        _, buffer, weight, activation = self._buffers[slot]
        # END shape replacement changes storage only, never native query values.
        with torch.autograd.graph.saved_tensors_hooks(lambda value: value.clone(), lambda value: value):
            value = self.scene_collision.get_sphere_distance_raw(
                spheres.contiguous(), buffer, weight, activation, return_loss=True)
        return value.clone()


# BEGIN observation evidence is diagnostic and owns no collision constraint.
class ObservedSpace:
    """Immutable observed/free masks, queried only for reporting known centres.

    The mask uses ZYX layout and origin is the XYZ centre of cell [0,0,0].
    Neither a known centre nor this report certifies the whole robot volume.
    Unknown cells never become observed free because a motion was accepted.
    """
    def __init__(self, known, *, origin, voxel_size, observation_ids, reference_frame,
                 occupancy_evidence=None, known_free=None):
        mask = np.asarray(known)
        if (mask.dtype != np.bool_ or mask.ndim != 3 or min(mask.shape) < 2
                or mask.size > 8_000_000):
            raise ValueError('Observation evidence requires a bounded boolean ZYX grid')
        self.origin = _array(origin, (3,), 'observation grid origin')
        self.voxel_size = float(voxel_size)
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError('Observation voxel size must be finite and positive')
        self.observation_ids = tuple(_text(value, 'observation ID') for value in observation_ids)
        if not self.observation_ids:
            raise ValueError('Observation evidence needs its source IDs')
        self.reference_frame = _text(reference_frame, 'observation frame')
        self.occupancy_evidence = deepcopy(occupancy_evidence or {'solids': [], 'owners': {}, 'robot': None})
        free = mask if known_free is None else np.asarray(known_free)
        if free.dtype != np.bool_ or free.shape != mask.shape or np.any(free & ~mask):
            raise ValueError('Free-space evidence must be a boolean subset of known cells')
        self._known, self._known_free = mask.copy(), free.copy()
        self._known.flags.writeable = self._known_free.flags.writeable = False
        self._shape_xyz = np.asarray(mask.shape[::-1])

    @property
    def known_free(self):
        result = self._known_free.copy()
        result.flags.writeable = False
        return result

    def diagnostics(self, points):
        points = np.asarray(points, dtype=float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError('Observation diagnostics need finite XYZ centres')
        indices = np.floor((points.reshape(-1, 3)-self.origin)/self.voxel_size+.5).astype(np.int64)
        inside = ((indices >= 0) & (indices < self._shape_xyz)).all(-1)
        known = np.zeros(len(indices), dtype=bool)
        free = np.zeros(len(indices), dtype=bool)
        x, y, z = indices[inside].T
        known[inside] = self._known[z, y, x]
        free[inside] = self._known_free[z, y, x]
        return {'sample_count': len(indices), 'known_centres': int(known.sum()),
                'known_free_centres': int(free.sum()), 'unknown_centres': int((~known).sum()),
                'status': 'unknown' if not known.all() else 'observed_centres',
                'scope': 'Point-centre observation metadata only; no whole-body coverage or collision proof'}

    def clone_for_rollout(self):
        result = copy(self)
        result.occupancy_evidence = deepcopy(self.occupancy_evidence)
        return result
# END observation evidence is diagnostic and owns no collision constraint.


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
    """One explicit geometry partition and its independent motion prediction.

    ``pose_path.sample(progress)`` returns Torch position and wxyz quaternion.
    Moving owners require a conservative local sphere cover for owner/world
    queries. ``geometry_complete=False`` retains partial observation honestly;
    it cannot certify the motion of an incompletely covered payload or part.
    """

    owner_id: str
    query: object
    pose_path: object
    source: str
    geometry_complete: bool
    local_spheres: object = None
    sphere_materials: tuple = ()
    moving: bool = False
    attachments: tuple[Attachment, ...] = ()
    sphere_cover_error_m: float = 0.
    pose_interval: tuple = (0., 1.)
    # BEGIN fixed robot installation policy, scoped only to robot/this-owner queries.
    robot_excluded_spheres: tuple = ()
    robot_exclusion_source: str | None = None
    robot_exclusion_geometry_source: str | None = None
    # BEGIN explicit aliases for native fixed installation solids already checked.
    native_geometry_names: tuple = ()
    # END never infer a native pair exemption from proxy sphere indices.
    # END exclusions never apply to background or another moving actor.

    def __post_init__(self):
        _text(self.owner_id, "owner ID")
        _text(self.source, "owner observation source")
        if self.query.owner_ids != frozenset((self.owner_id,)):
            raise ValueError("An owner query must contain precisely that physical owner")
        if not callable(getattr(self.pose_path, "sample", None)):
            raise TypeError("Owner motion needs a differentiable PosePath")
        # BEGIN only completely serializable native paths bind collision owners.
        # Owner stationarity checks and execution-binding serialization require
        # one world-frame path. They store PosePath's declared fields, not an
        # arbitrary subclass's continuous evaluation rule. In particular, a
        # TransferredPosePath contains boundary corrections that its sampled
        # poses cannot reconstruct. Reject it even when parent is None rather
        # than silently replacing its rotating offsets with straight chords.
        if type(self.pose_path) is not PosePath or self.pose_path.parent is not None:
            raise ValueError("Collision owners require a single world-frame path with native PosePath serialization; composed and derived paths are reference-only")
        # END ordinary world-frame owners retain their existing scene contract.
        self.pose_interval = _valid_interval(self.pose_interval)
        if type(self.geometry_complete) is not bool or type(self.moving) is not bool:
            raise ValueError("Geometry coverage and motion must be explicitly declared")
        self.attachments = tuple(self.attachments)
        self.native_geometry_names = tuple(self.native_geometry_names)
        if self.native_geometry_names and (self.moving or self.attachments or not self.geometry_complete):
            raise ValueError("Native installation identities require a complete fixed owner")
        if len(set(self.native_geometry_names)) != len(self.native_geometry_names):
            raise ValueError("Native installation geometry identities must be unique")
        for name in self.native_geometry_names:
            _text(name, "native installation geometry identity")
        for index, relation in enumerate(self.attachments):
            if not isinstance(relation, Attachment):
                raise TypeError("Owner attachments must be explicit ordered attachment spans")
            if index and relation.interval[0] < self.attachments[index - 1].interval[1]:
                raise ValueError("Attachment spans must be ordered and cannot overlap")
            if index and self.attachments[index - 1].release_condition_id is None:
                raise ValueError("Regrasp requires a real preceding release condition")
        if self.attachments:
            self.moving = True
        # BEGIN accept only explicit native installation policy on complete static owners.
        self.robot_excluded_spheres = tuple(self.robot_excluded_spheres)
        if (len(set(self.robot_excluded_spheres)) != len(self.robot_excluded_spheres)
                or any(type(index) is not int or index < 0 for index in self.robot_excluded_spheres)):
            raise ValueError("Native robot installation exclusions require unique nonnegative sphere indices")
        if self.robot_excluded_spheres:
            if self.moving or not self.geometry_complete:
                raise ValueError("Native installation exclusions require a complete fixed collision owner")
            _text(self.robot_exclusion_source, "native installation pair-policy source")
            _text(self.robot_exclusion_geometry_source, "installation policy robot geometry source")
            if (not isinstance(self.pose_path, PosePath) or self.pose_path.motion_model is not None
                    or not np.allclose(self.pose_path.poses, self.pose_path.poses[0], atol=1e-12, rtol=0)):
                raise ValueError("A fixed installation exclusion cannot follow a moving pose path")
        elif self.robot_exclusion_source is not None or self.robot_exclusion_geometry_source is not None:
            raise ValueError("Installation exclusion provenance requires its explicit nonempty sphere set")
        # END body-pair policy cannot grant a dynamic/contact-owner exemption.
        if self.local_spheres is not None:
            value = np.asarray(self.local_spheres, float)
            if (value.ndim != 2 or value.shape[1] != 4 or not len(value)
                    or not np.isfinite(value).all() or np.any(value[:, 3] <= 0)):
                raise ValueError("Owner spheres must be finite positive local [S,4] proxies")
            self.local_spheres = value.copy()
            if self.sphere_materials and len(self.sphere_materials) != len(value):
                raise ValueError("Owner sphere material identities must match its sphere cover")
        if self.moving and self.local_spheres is None:
            raise ValueError("A moving owner needs its complete conservative sphere cover")
        if not np.isfinite(self.sphere_cover_error_m) or self.sphere_cover_error_m < 0:
            raise ValueError("Sphere coverage error must be explicitly bounded")

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


# BEGIN interaction permission is a scoped attempt, not a contact success proof.
@dataclass(frozen=True)
class ContactRule:
    """Permit only named actor spheres against one owner during one interval.

    Sphere identities come from the existing robot/object binding. A rule
    neither certifies contact nor alters any other actor, owner or background.
    Precise material patches and intended task effects are not admission data.
    """
    rule_id: str
    actor: str
    target_owner: str
    interval: tuple
    sphere_indices: tuple
    source: str

    def __post_init__(self):
        for name in ('rule_id', 'actor', 'target_owner', 'source'):
            _text(getattr(self, name), name)
        if self.actor == self.target_owner:
            raise ValueError('Interaction actor and target must differ')
        indices = tuple(self.sphere_indices)
        if (not indices or len(set(indices)) != len(indices)
                or any(type(index) is not int or index < 0 for index in indices)):
            raise ValueError('Interaction permission requires explicit unique sphere indices')
        object.__setattr__(self, 'sphere_indices', indices)
        object.__setattr__(self, 'interval', _valid_interval(self.interval))
# END interaction permission is a scoped attempt, not a contact success proof.


# BEGIN bind finite grasp materials to their original arrival/close condition.
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
# END rule identities never grant an entire robot/owner collision exemption.


# BEGIN one typed state owns both future geometry and its uncertainty.
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
# END command timing cannot independently select a physical collision model.


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
    ``self_collision_pairs`` is the robot's explicit geometric pair policy,
    never object-contact permission.
    """

    def __init__(self, *, controller, joint_names, current_positions, joint_limits,
                 joint_bounds, coupling_matrix, coupling_offset, sphere_indices,
                 local_spheres, sphere_joint_translation, self_collision_pairs,
                 commands, source, control_timestep_s, numerical_joint_tolerance=1e-5,
                 finger_trajectory_factory=None, nominal_finger_trajectory_factory=None,
                 self_collision_geometry=()):
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
        self.sphere_indices = tuple(sphere_indices)
        if (not self.sphere_indices or len(set(self.sphere_indices)) != len(self.sphere_indices)
                or any(type(i) is not int or i < 0 for i in self.sphere_indices)):
            raise ValueError("Every gripper proxy needs its explicit robot sphere index")
        self.local_spheres = _array(local_spheres, (len(self.sphere_indices), 4), "gripper local spheres")
        if np.any(self.local_spheres[:, 3] <= 0):
            raise ValueError("Gripper proxy radii must be positive")
        self.sphere_joint_translation = _array(sphere_joint_translation,
                                              (len(self.sphere_indices), 3, count), "prismatic proxy Jacobian")
        self.self_collision_pairs = tuple(tuple(pair) for pair in self_collision_pairs)
        for pair in self.self_collision_pairs:
            if (len(pair) != 2 or pair[0] == pair[1] or any(type(i) is not int or i < 0 for i in pair)
                    or not set(pair) & set(self.sphere_indices)):
                raise ValueError("Envelope self-collision pairs must involve the gripper")
        # BEGIN reject duplicate declarations before batched summation.
        if len(set(self.self_collision_pairs)) != len(self.self_collision_pairs):
            raise ValueError("Self-collision pair declarations must be unique")
        self._self_pair_indices = np.asarray(self.self_collision_pairs, dtype=np.int64).reshape(-1, 2)
        # END reject duplicate declarations before batched summation.
        # BEGIN complete native prismatic solids may replace their proxy pairs.
        from .prismatic_self_collision import PreparedPrismaticSelfCollision
        self.self_collision_model = (PreparedPrismaticSelfCollision(
            self_collision_geometry, joint_names=self.joint_names,
            sphere_indices=self.sphere_indices, local_spheres=self.local_spheres,
            sphere_joint_translation=self.sphere_joint_translation,
            collision_pairs=self._self_pair_indices) if self_collision_geometry else None)
        # END no source solid means the existing sphere model remains explicit.
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
    # END nominal finger geometry never asserts that commands equal actual feedback.

    def spheres(self, robot_spheres, position, quaternion, coordinates, half):
        local = _tensor(self.local_spheres, position)
        mapping = _tensor(self.sphere_joint_translation, position)
        centres = local[:, :3] + torch.einsum("ncj,bhj->bhnc", mapping, coordinates)
        # Triangle inequality covers every point in the continuous joint box.
        displacement = torch.einsum("ncj,bhj->bhnc", mapping.abs(), half)
        radii = local[:, 3] + torch.linalg.vector_norm(displacement, dim=-1)
        transformed = torch.cat((_world_points(centres, position, quaternion), radii[..., None]), -1)
        result = robot_spheres.clone()
        # BEGIN preserve the original advanced-index update under graph capture.
        result[..., constant_tensor(list(self.sphere_indices), result, dtype=torch.long), :] = transformed
        # END no sphere values or replacement order are changed.
        return result


# BEGIN share CPU-only hand/contact validation with the RGB-D allocator.
# BEGIN only command identity and scoped sphere membership are structural gates.
def _validate_contact_and_gripper_inputs(owner_materials, robot_materials, contacts, gripper, has_gripper):
    """Validate interaction indices before mapping without geometry proof work."""
    if type(has_gripper) is not bool or (has_gripper and not isinstance(gripper, PrismaticGripper)):
        raise ValueError('A gripper requires current feedback and a bounded robot model')
    if not has_gripper and gripper is not None:
        raise ValueError('Gripper declaration and geometry disagree')
    if gripper is not None:
        indices = list(gripper.sphere_indices)+[i for pair in gripper.self_collision_pairs for i in pair]
        if indices and max(indices) >= len(robot_materials):
            raise ValueError('Gripper indices must belong to the declared robot spheres')
    contacts = tuple(contacts)
    if len({r.rule_id for r in contacts}) != len(contacts):
        raise ValueError('Interaction rule IDs must be unique')
    for rule in contacts:
        if not isinstance(rule, ContactRule) or rule.target_owner not in owner_materials:
            raise ValueError('Interaction rules must identify a current owner')
        materials = robot_materials if rule.actor == 'robot' else owner_materials.get(rule.actor)
        if not materials or max(rule.sphere_indices) >= len(materials):
            raise ValueError('Interaction sphere indices must belong to the named actor')
    return contacts
# END only command identity and scoped sphere membership are structural gates.
