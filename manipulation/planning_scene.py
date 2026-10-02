"""Observed, owner-preserving geometry for generation and final verification.

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
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .planning_problem import (
    PosePath, quaternion_conjugate, quaternion_multiply, quaternion_rotate, quaternion_rotvec,
)
# BEGIN scene commands and planner events share exact occurrence identities.
from .planning_problem import event_occurrence_key
# END source names alone do not identify an explicitly repeated operation.
from .motion import _inverse, _pose_array


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
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


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
def _self_pair_chunk_size(states, pairs):
    """Bound temporary pair expansion independently of a robot's sphere count."""
    return min(pairs, 512, max(1, 1_048_576 // states)) if pairs else 0


# BEGIN bound guidance scratch while avoiding thousands of tiny IK launches.
_COVERAGE_POINT_PLANE_ELEMENTS = 1_048_576


def _coverage_plane_chunk_size(states, points, planes):
    """Use the same finite point/plane scratch target at every batch size.

    One plane is the irreducible query. Small IK batches can evaluate whole
    hulls at once; larger trajectory batches divide the exact same planes.
    """
    if any(type(value) is not int or value <= 0 for value in (states, points, planes)):
        raise ValueError("Coverage chunks require positive state, point and plane counts")
    return min(planes, max(1, _COVERAGE_POINT_PLANE_ELEMENTS // (states*points)))


def _coverage_halfspace_max(points, halfspaces):
    """Preserve the convex value while saving indices rather than signed planes.

    max(dim) and amax have identical values. At a unique maximizing face their
    derivatives agree; at a tie max selects one valid face subgradient. Its
    maintained Torch backward saves argmax indices, avoiding a retained
    [states, points, planes] input when a body checkpoint is recomputed.
    """
    normals, offsets = halfspaces[:, :3], halfspaces[:, 3]
    signed = (points@normals.T+offsets)/torch.linalg.vector_norm(normals, dim=-1)
    return signed.max(dim=-1).values
# END chunking changes neither the sample set nor the convex inequalities.


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
    pairs = torch.as_tensor(pair_indices, dtype=torch.long, device=spheres.device)
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

    def __init__(self, scene_collision, *, owner_ids, source,
                 max_query_spheres=2_000_000, max_buffers=64,
                 max_cache_bytes=32 * 1024 * 1024):
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
        slot = (key, shape, str(spheres.device))
        if slot not in self._buffers:
            if len(self._buffers) >= self.max_buffers:
                raise ValueError("Collision query exceeds its bounded workspace cache")
            byte_count = math.prod(shape[:-1]) * 5 * 4 + 8
            if self._cache_bytes + byte_count > self.max_cache_bytes:
                raise ValueError("Collision workspaces exceed the total query cache byte limit")
            config = self.scene_collision.device_cfg
            if not config.is_same_torch_device(spheres.device):
                raise ValueError("Query and cuRobo scene must share a device")
            self._buffers[slot] = (
                CollisionBuffer.from_shape(spheres.shape, config),
                spheres.new_ones(1), spheres.new_zeros(1))
            self._cache_bytes += byte_count
        buffer, weight, activation = self._buffers[slot]
        with torch.autograd.graph.saved_tensors_hooks(lambda value: value.clone(), lambda value: value):
            value = self.scene_collision.get_sphere_distance_raw(
                spheres.contiguous(), buffer, weight, activation, return_loss=True)
        return value.clone()


class ObservedSpace:
    """Conservative whole-sphere coverage of a supplied observation grid.

    ``known[z,y,x]`` means the cell is covered by current observation evidence,
    not that its occupancy is free. Occupancy is checked by the collision
    queries. Hidden surfaces, invalid rays and space outside the grid remain
    unknown. ``origin`` is the centre of cell [0,0,0]. The caller must preserve
    this mask when building its TSDF/ESDF; an owner reconstruction alone cannot
    serve as a full-scene mask.

    SciPy computes distance to unknown cell centres with an unknown border.
    Subtracting a full voxel diagonal accounts conservatively for unknown cell
    volume and trilinear interpolation. It can reject narrow observed passages;
    it never fills unobserved space to make a result pass.
    """

    def __init__(self, known, *, origin, voxel_size, observation_ids, reference_frame,
                 occupancy_evidence=None, known_free=None):
        mask = np.asarray(known)
        if (mask.dtype != np.bool_ or mask.ndim != 3 or min(mask.shape) < 2
                or mask.size > 8_000_000):
            raise ValueError("An explicit bounded 3D boolean observation grid is required")
        self.origin = _array(origin, (3,), "observation grid origin")
        if not np.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("Observation voxel size must be positive metres")
        self.voxel_size = float(voxel_size)
        self.observation_ids = tuple(_text(item, "observation ID") for item in observation_ids)
        if not self.observation_ids:
            raise ValueError("Coverage needs current observation provenance")
        self.reference_frame = _text(reference_frame, "coverage reference frame")
        # BEGIN retain the physical bodies used to prove occupied interiors.
        # Known does not mean free. An owner used for this evidence must still
        # have complete collision geometry, and robot FK evidence must match
        # the actual start state when the scene is bound to a problem.
        evidence = ({"solids": [], "owners": {}, "robot": None}
                    if occupancy_evidence is None else occupancy_evidence)
        if (not isinstance(evidence, Mapping) or set(evidence) != {"solids", "owners", "robot"}
                or not isinstance(evidence["solids"], (list, tuple)) or len(evidence["solids"]) > 64
                or not isinstance(evidence["owners"], Mapping) or len(evidence["owners"]) > 32):
            raise ValueError("Occupied coverage requires bounded explicit solid/body evidence")
        owner_ids, has_robot = set(), False
        for record in evidence["solids"]:
            if (not isinstance(record, Mapping)
                    or record.get("body_kind") not in {"owner", "robot_link"}
                    or record.get("occupancy_kind") not in {"verified_physical_solid", "verified_inner_solid"}):
                raise ValueError("Occupied coverage must identify verified physical solids and their current bodies")
            identity = _text(record.get("body_id"), "occupied body identity")
            if record["body_kind"] == "owner":
                owner_ids.add(identity)
            else:
                has_robot = True
        if owner_ids != set(evidence["owners"]) or has_robot != (evidence["robot"] is not None):
            raise ValueError("Occupied coverage solid identities and current body evidence disagree")
        self.occupancy_evidence = deepcopy(evidence)
        # BEGIN preserve raw observed free cells separately from occupied evidence.
        # The solid-volume oracle must never reinterpret occupied interiors as
        # free space. Existing pure-observation masks already have this meaning.
        if known_free is None and evidence["solids"]:
            self._known_free = None
        else:
            free = mask if known_free is None else np.asarray(known_free)
            if free.dtype != np.bool_ or free.shape != mask.shape or np.any(free & ~mask):
                raise ValueError("Raw free evidence must be a boolean subset of the known grid")
            self._known_free = free.copy()
            self._known_free.flags.writeable = False
        # END no collision model or initial-state assumption changes this mask.
        # END retain the physical bodies used to prove occupied interiors.
        distance = distance_transform_edt(np.pad(mask, 1), sampling=voxel_size)[1:-1, 1:-1, 1:-1]
        distance = np.maximum(distance - math.sqrt(3) * voxel_size, 0)
        self._distance = torch.from_numpy(distance.astype(np.float32))[None, None]
        self._shape_xyz = np.array(mask.shape[::-1], float)
        # BEGIN retain exact observation masks for the execution binding.
        self._known = mask.copy()
        self._known.flags.writeable = False
        # END serialization does not reconstruct free cells from a distance grid.

    @property
    def known_free(self):
        """Return an owned read-only copy of the original observed free mask."""
        if self._known_free is None:
            raise ValueError("Occupied coverage requires separate raw known_free evidence")
        result = self._known_free.copy()
        result.flags.writeable = False
        return result

    # BEGIN candidate metadata is private; dense evidence is shared read-only.
    def clone_for_rollout(self):
        """Copy mutable metadata while retaining immutable private grid storage.

        The distance tensor and raw-free array have no mutating scene API.
        They are request evidence, not candidate state: callers must construct
        a new ObservedSpace for an observation update and must not modify its
        private grids during planning. Sharing them avoids one dense-grid
        allocation per optimizer rollout.
        """
        result = copy(self)
        result.origin = self.origin.copy()
        result._shape_xyz = self._shape_xyz.copy()
        result.occupancy_evidence = deepcopy(self.occupancy_evidence)
        return result
    # END rule/relationship edits cannot rewrite another candidate's metadata.

    def violation(self, spheres):
        centre = spheres[..., :3]
        coordinates = (centre - _tensor(self.origin, centre)) / self.voxel_size
        grid = 2 * coordinates / _tensor(self._shape_xyz - 1, centre) - 1
        values = F.grid_sample(self._distance.to(device=centre.device, dtype=centre.dtype),
                               grid.reshape(1, 1, 1, -1, 3), mode="bilinear",
                               padding_mode="zeros", align_corners=True)
        clearance = values.reshape(spheres.shape[:-1])
        radius = spheres[..., 3]
        return torch.where(radius > 0, torch.relu(radius - clearance), torch.zeros_like(radius))


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


@dataclass
class ContactRule:
    """A material feature / owner patch permission, never a link exemption.

    The material points form a planar convex feature; checking every vertex
    against a rectangular patch covers the full feature, not merely a sphere
    centre. For robot actors they are tool-local. For owner actors they are
    owner-local. Optional affine joint translations represent calibrated
    prismatic finger motion and use the scene's declared gripper coordinates.
    The patch normal points out of the target; actor normals oppose it.
    """

    rule_id: str
    actor: str
    target_owner: str
    interval: tuple
    sphere_indices: tuple
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
    minimum_thickness_m: float
    source: str
    required: bool = False
    feature_joint_translation: object = None
    numerical_position_tolerance_m: float = 1e-6
    numerical_angle_tolerance_rad: float = 1e-6

    def __post_init__(self):
        for name in ("rule_id", "actor", "target_owner", "material_id", "source"):
            _text(getattr(self, name), name)
        self.interval = _valid_interval(self.interval)
        if not self.sphere_indices or any(type(i) is not int or i < 0 for i in self.sphere_indices):
            raise ValueError("Contact permission needs explicit material sphere indices")
        self.sphere_indices = tuple(self.sphere_indices)
        points = np.asarray(self.feature_points, float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("A contact feature needs its complete planar polygon vertices")
        self.feature_points = points.copy()
        self.feature_normal = _array(self.feature_normal, (3,), "material normal")
        if not np.isclose(np.linalg.norm(self.feature_normal), 1., atol=1e-6):
            raise ValueError("Material normal must be unit length")
        if np.max(np.abs((points - points[0]) @ self.feature_normal)) > 1e-8:
            raise ValueError("Material contact vertices must lie in their declared plane")
        self.patch_origin = _array(self.patch_origin, (3,), "patch origin")
        self.patch_axes = _array(self.patch_axes, (3, 3), "patch axes")
        if (not np.allclose(self.patch_axes.T @ self.patch_axes, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.patch_axes), 1., atol=1e-6)):
            raise ValueError("Patch axes must be a proper orthonormal frame")
        self.patch_half_size = _array(self.patch_half_size, (2,), "patch half size")
        gap = _array(self.gap_bounds, (2,), "contact gap bounds")
        if np.any(self.patch_half_size <= 0) or gap[0] > gap[1]:
            raise ValueError("Contact patch and gap bounds must be explicit and ordered")
        self.gap_bounds = tuple(gap)
        if (not 0 <= self.normal_tolerance_rad < math.pi / 2
                or not 0 <= self.modeling_allowance_m <= self.geometry_error_m
                or self.minimum_thickness_m <= 0
                or self.modeling_allowance_m >= self.minimum_thickness_m / 2):
            raise ValueError("Contact allowance must be bounded by error and thin-part scale")
        if self.feature_joint_translation is not None:
            value = np.asarray(self.feature_joint_translation, float)
            if (value.ndim != 3 or value.shape[:2] != points.shape or not np.isfinite(value).all()):
                raise ValueError("Prismatic material offsets need shape [vertices,3,joints]")
            self.feature_joint_translation = value.copy()
        if (not np.isfinite(self.numerical_position_tolerance_m)
                or not np.isfinite(self.numerical_angle_tolerance_rad)
                or min(self.numerical_position_tolerance_m, self.numerical_angle_tolerance_rad) < 0):
            raise ValueError("Numerical contact tolerances must be separately declared finite values")


class PrismaticGripper:
    """Bind native finger coordinates to a supported controller and geometry.

    Position tracking uses explicitly modeled solver coordinates. Timed
    actuator commands and optional nominal future geometry share the bound
    elapsed clock but remain separate objects. Binary control without a future
    model retains its mechanical envelope only as unresolved diagnostic
    evidence. A sampled width or command is never measured future contact.

    ``joint_bounds(progress)`` is the solver tracking contract. Independent
    ``finger_trajectory_factory`` supplies commands;
    ``nominal_finger_trajectory_factory`` supplies declared future geometry.
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
        # BEGIN own each bound occurrence so later retry input cannot alter this scene.
        self.commands = tuple(deepcopy(commands))
        if not self.commands:
            raise ValueError("Gripper geometry must be bound to explicit controller events")
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
            nominal = self.nominal_finger_trajectory_factory(duration, progress_map)
            if (tuple(nominal.joint_names) != self.joint_names or nominal.duration != duration
                    or not np.array_equal(nominal.sample(np.array([0.]))[0][0], self.current_positions)):
                raise ValueError("Nominal hand geometry must preserve measured entrance, names and bound clock")
            self.nominal_finger_trajectory = nominal

    def state(self, progress, joint_position, joint_names, *, trajectory_time=None):
        expected = (*progress.shape, len(self.joint_names))
        limits = _tensor(self.joint_limits, progress)
        measured = _tensor(self.current_positions, progress)
        if self.controller == "timed_reference" or self.nominal_finger_trajectory is not None:
            if ((self.nominal_finger_trajectory is None and self.finger_trajectory is None)
                    or trajectory_time is None or trajectory_time.shape != progress.shape):
                raise ValueError("Timed finger geometry requires its bound trajectory and actual query time")
            model = self.nominal_finger_trajectory or self.finger_trajectory
            coordinates, _ = model.sample_torch(trajectory_time)
            if coordinates.shape != expected:
                raise ValueError("Named finger samples must align with the scene query")
            tolerance = self.numerical_joint_tolerance
            violations = (torch.relu(limits[:, 0]-coordinates-tolerance).sum(-1)
                          + torch.relu(coordinates-limits[:, 1]-tolerance).sum(-1))
            if len(self.coupling_matrix):
                coupling = _tensor(self.coupling_matrix, progress)
                violations = violations + torch.relu(
                    (coordinates @ coupling.T-_tensor(self.coupling_offset, progress)).abs()
                    - tolerance*coupling.abs().sum(-1)).sum(-1)
            return coordinates, torch.zeros_like(coordinates), violations
        if self.controller == "binary":
            # BEGIN binary progress cannot identify the measured initial instant.
            # Neither a repeated zero-progress sample nor an isolated IK/search
            # query carries a time/velocity contract for the fingers. Preserve
            # the full continuous mechanical interval at every queried state;
            # current_positions remains the separately checked real feedback.
            lower = limits[:, 0].expand(expected)
            upper = limits[:, 1].expand(expected)
            if trajectory_time is not None:
                initial = trajectory_time == 0
                lower = torch.where(initial[..., None], measured, lower)
                upper = torch.where(initial[..., None], measured, upper)
            # END time-independent binary envelope.
        else:
            lower, upper = self.joint_bounds(progress)
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
            coordinates = joint_position[..., indices]
            # Tracking means a single executable command, not optimizer-selected width.
            violations = violations + torch.relu((upper - lower).abs() - tolerance).sum(-1)
            violations = violations + torch.relu((coordinates - middle).abs() - tolerance).sum(-1)
            half = torch.zeros_like(half)
        else:
            coordinates = middle
        initial = (progress == 0).to(progress.dtype)
        violations = violations + initial * (torch.relu(lower - measured - tolerance).sum(-1)
                                             + torch.relu(measured - upper - tolerance).sum(-1))
        if len(self.coupling_matrix) and self.controller == "position_tracking":
            coupling = _tensor(self.coupling_matrix, progress)
            coupling_tolerance = tolerance * coupling.abs().sum(-1)
            violations = violations + torch.relu((coordinates @ coupling.T
                                         - _tensor(self.coupling_offset, progress)).abs()
                                         - coupling_tolerance).sum(-1)
        return coordinates, half, violations
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
        result[..., list(self.sphere_indices), :] = transformed
        return result


# BEGIN share CPU-only hand/contact validation with the RGB-D allocator.
def _validate_contact_and_gripper_inputs(owner_materials, robot_materials, contacts, gripper, has_gripper):
    """Validate geometry identities before observation mapping touches CUDA.

    ``owner_materials`` maps every explicit owner to its moving sphere-material
    tuple, or None for a static owner. It carries no collision query or device
    object, allowing the RGB-D adapter and the final scene to share one check.
    """
    if type(has_gripper) is not bool or (has_gripper and not isinstance(gripper, PrismaticGripper)):
        raise ValueError("A gripper requires current feedback, drive capability and bounded geometry")
    if not has_gripper and gripper is not None:
        raise ValueError("Gripper declaration and geometry disagree")
    if gripper is not None:
        indices = list(gripper.sphere_indices) + [i for pair in gripper.self_collision_pairs for i in pair]
        if max(indices) >= len(robot_materials):
            raise ValueError("Gripper indices must belong to the declared robot sphere model")
    contacts = tuple(contacts)
    if len({rule.rule_id for rule in contacts}) != len(contacts):
        raise ValueError("Contact rule IDs must be unique")
    for rule in contacts:
        if rule.target_owner not in owner_materials or rule.actor == rule.target_owner:
            raise ValueError("Contact permissions require distinct, explicit owner pairs")
        if rule.actor == "robot":
            materials = robot_materials
        elif rule.actor in owner_materials and owner_materials[rule.actor] is not None:
            materials = owner_materials[rule.actor]
        else:
            raise ValueError("Only robot or moving-owner proxies can request contact permission")
        if (not materials or max(rule.sphere_indices) >= len(materials)
                or any(materials[index] != rule.material_id for index in rule.sphere_indices)):
            raise ValueError("Permission must match dedicated material contact proxies, not whole links")
        if rule.feature_joint_translation is not None:
            if (rule.actor != "robot" or gripper is None
                    or rule.feature_joint_translation.shape[-1] != len(gripper.joint_names)):
                raise ValueError("Finger contact features require the same calibrated prismatic model")
        if rule.actor == "robot" and gripper is not None:
            moving = [gripper.sphere_indices.index(i) for i in rule.sphere_indices if i in gripper.sphere_indices]
            if (moving and np.any(gripper.sphere_joint_translation[moving])
                    and rule.feature_joint_translation is None):
                raise ValueError("A moving finger material feature needs its actual joint translation")
        # Separate permissions cannot accumulate modelling allowances.
        for other in contacts:
            if (other is not rule and other.actor == rule.actor and other.target_owner == rule.target_owner
                    and set(other.sphere_indices) & set(rule.sphere_indices)
                    and max(other.interval[0], rule.interval[0]) < min(other.interval[1], rule.interval[1])):
                raise ValueError("Overlapping material permissions would double-count allowance")
    return contacts
# END CPU-only hand/contact validation.


class PlanningScene:
    """One scene contract for optimizer rollout, search and final checks.

    ``robot_sphere_materials`` assigns a specific material contact feature or
    None to *every* robot sphere. Such spheres must cover only that feature;
    palm/finger noncontact volume needs separate unmodified proxies. Geometry
    calibration and coverage of those proxies belong to the supplied model.
    Missing gripper information is rejected rather than assuming open fingers.
    A robot without a gripper must explicitly pass ``has_gripper=False``.
    """

    def __init__(self, *, background, owners, coverage, reference_frame,
                 robot_sphere_materials, robot_geometry_source,
                 contacts=(), gripper=None, has_gripper=True, robot_coverage=None,
                 numerical_position_tolerance_m=1e-4, numerical_rotation_tolerance_rad=1e-3,
                 native_geometry_materials=None):
        self.background = background
        # BEGIN independent solids need explicit material identity, not sphere-index guesses.
        self.native_geometry_materials = dict(native_geometry_materials or {})
        # END identity is serialized with the exact scene binding.
        self.owners = tuple(owners)
        ids = [owner.owner_id for owner in self.owners]
        if len(ids) != len(set(ids)) or set(ids) & set(background.owner_ids):
            raise ValueError("Active/contact owners must be partitioned out of the background by identity")
        if not isinstance(coverage, ObservedSpace):
            raise ValueError("A current full-scene observation coverage grid is required")
        self.reference_frame = _text(reference_frame, "planning scene frame")
        if coverage.reference_frame != reference_frame:
            raise ValueError("Observation coverage and FK must use the same calibrated frame")
        self.coverage = coverage
        self.robot_coverage = robot_coverage
        if robot_coverage is not None and robot_coverage.reference_frame != reference_frame:
            raise ValueError("Native solid coverage must use the same calibrated reference frame")
        # BEGIN the guidance grid and the mandatory volume oracle are one input.
        if robot_coverage is not None:
            robot_coverage.validate_observation_grid(
                known_free=coverage.known_free, origin=coverage.origin,
                voxel_size=coverage.voxel_size, reference_frame=coverage.reference_frame)
        # END frame names alone cannot bind independent free-cell evidence.
        # BEGIN immutable, bounded optimization guidance for the solid oracle.
        self._coverage_geometry = (None if robot_coverage is None else robot_coverage.guidance_geometry())
        self._coverage_samples = ()
        if robot_coverage is not None:
            # These identical model samples are immutable across every native
            # objective and checkpoint recomputation; select them only once.
            samples = []
            for solid in self._coverage_geometry["solids"]:
                vertices = np.asarray(solid["vertices_frame"], float)
                faces = np.asarray(solid["faces"], dtype=int)
                samples.append(np.concatenate((
                    vertices[np.linspace(0, len(vertices)-1, min(16, len(vertices)), dtype=int)],
                    vertices[faces[np.linspace(0, len(faces)-1, min(16, len(faces)), dtype=int)]].mean(1))))
            self._coverage_samples = tuple(samples)
            free = coverage.known_free
            self._has_free_cells = bool(free.any())
            distance = distance_transform_edt(~np.pad(free, 1), sampling=coverage.voxel_size)[1:-1, 1:-1, 1:-1]
            self._free_distance = torch.from_numpy(distance.astype(np.float32))[None, None]
            self._free_mask = torch.from_numpy(free.astype(np.float32))[None, None]
        # END sparse points guide optimization; only full solid CSG accepts coverage.
        self.observation_report = None
        self.robot_sphere_materials = tuple(robot_sphere_materials)
        if not self.robot_sphere_materials:
            raise ValueError("The actual robot sphere model and material identities are required")
        self.robot_geometry_source = _text(robot_geometry_source, "robot sphere model source")
        # BEGIN bind installation masks to this exact robot sphere layout.
        for owner in self.owners:
            if owner.robot_excluded_spheres and (
                    owner.robot_exclusion_geometry_source != self.robot_geometry_source
                    or max(owner.robot_excluded_spheres) >= len(self.robot_sphere_materials)):
                raise ValueError("Installation policy must match this scene's complete robot sphere model")
        # END static owners remain ordinary obstacles to every other actor.
        # BEGIN bind occupied-space evidence to bodies still checked for collision.
        self._occupied_owner_ids = frozenset(coverage.occupancy_evidence["owners"])
        if not self._occupied_owner_ids <= set(ids):
            raise ValueError("Every owner supplying occupied coverage must remain an explicit collision owner")
        robot_evidence = coverage.occupancy_evidence["robot"]
        if robot_evidence is not None and (
                robot_evidence.get("robot_geometry_source") != self.robot_geometry_source
                or robot_evidence.get("robot_spheres") != len(self.robot_sphere_materials)):
            raise ValueError("Occupied robot evidence must use this scene's complete native collision model")
        self.required_robot_links = (() if robot_evidence is None
                                     else tuple(sorted(robot_evidence["link_poses_output"])))
        if robot_coverage is not None:
            self.required_robot_links = tuple(sorted(set(self.required_robot_links)
                                                       | set(robot_coverage.required_link_names)))
        # END bind occupied-space evidence to bodies still checked for collision.
        if (not np.isfinite(numerical_position_tolerance_m) or not np.isfinite(numerical_rotation_tolerance_rad)
                or min(numerical_position_tolerance_m, numerical_rotation_tolerance_rad) < 0):
            raise ValueError("Attachment numerical tolerances must be separately declared finite values")
        self.numerical_position_tolerance_m = float(numerical_position_tolerance_m)
        self.numerical_rotation_tolerance_rad = float(numerical_rotation_tolerance_rad)
        # BEGIN validate the same hand/material contract before and after mapping.
        self.contacts = _validate_contact_and_gripper_inputs(
            {owner.owner_id: owner.sphere_materials if owner.moving else None for owner in self.owners},
            self.robot_sphere_materials, contacts, gripper, has_gripper)
        self.gripper = gripper
        # END shared CPU-only hand/material validation.
        self._fixed_release_poses = {}
        anchors = [x for owner in self.owners for relation in owner.attachments for x in relation.interval]
        anchors.extend(x for rule in self.contacts for x in rule.interval)
        if gripper is not None:
            anchors.extend(float(event["progress"]) for event in gripper.commands)
        self.required_progress_anchors = tuple(sorted(set(anchors)))

    # BEGIN bind occupied robot evidence to the actual planner's native model.
    def validate_robot_geometry(self, current_spheres_output, link_poses_output):
        """Compare actual q0 native FK, even when model labels/counts match.

        The backend calls this after its bounded model allocation and before
        optimizing. Every recorded solid link transform and the complete
        current sphere layout must match in the planning reference frame.
        The tiny numerical comparison allowance is smaller than the occupied
        voxel builder's additional inward numerical margin.
        """
        evidence = self.coverage.occupancy_evidence["robot"]
        if evidence is None:
            return
        expected = _array(evidence["current_spheres_output"],
                          (len(self.robot_sphere_materials), 4), "occupied current robot spheres")
        actual = _array(current_spheres_output, expected.shape, "actual current robot spheres")
        if not np.allclose(actual, expected, atol=1e-7, rtol=0):
            raise ValueError("Occupied robot evidence and the actual native collision sphere model disagree")
        if not isinstance(link_poses_output, Mapping) or set(link_poses_output) != set(self.required_robot_links):
            raise ValueError("Every occupied robot link needs its actual native current FK transform")
        for name, saved in evidence["link_poses_output"].items():
            predicted = _pose_array(saved, "occupied current robot link pose")
            actual = _pose_array(link_poses_output[name], "actual current robot link pose")
            if not np.allclose(actual, predicted, atol=1e-7, rtol=0):
                raise ValueError("Occupied robot evidence and the actual native link FK disagree")
    # END bind occupied robot evidence to the actual planner's native model.

    # BEGIN bind one collision scene to the exact generated skill problem.
    def validate_problem(self, problem):
        """Reject independently edited payload, mechanism and hand contracts.

        This is a cheap CPU preflight, before native solver allocation. It
        checks identity and calibration; partial observed geometry remains an
        explicit unknown term during evaluation. It never turns a candidate
        attachment condition into verified Holding evidence.
        """
        if self.reference_frame != problem.reference_frame:
            raise ValueError("Problem and collision scene require the same reference frame")
        # BEGIN bind the frozen initial-volume assumption to this exact entrance.
        if self.robot_coverage is not None:
            evidence = self._coverage_geometry
            if not np.array_equal(evidence["T_reference_base"], problem.base_pose):
                raise ValueError("Native coverage and planning must share the exact calibrated base pose")
            named = dict(zip(self.robot_coverage.baseline_joint_names, self.robot_coverage.baseline_joint_positions))
            if any(name not in named or named[name] != value for name, value in zip(problem.joint_names, problem.start_q)):
                raise ValueError("Native coverage baseline must use the exact measured arm entrance")
            hand_names = () if self.gripper is None else self.gripper.joint_names
            if set(named) != set(problem.joint_names) | set(hand_names):
                raise ValueError("Native coverage requires the complete arm and explicitly controlled hand coordinates")
            if self.gripper is not None and any(named[name] != value for name, value in zip(hand_names, self.gripper.current_positions)):
                raise ValueError("Native coverage baseline must preserve the actual measured hand")
        # END model-relative V0 cannot be relabelled from another feedback epoch.
        by_id = {owner.owner_id: owner for owner in self.owners}
        background_ids = set(self.background.owner_ids)
        condition_rows = [row for row in problem.conditions if "condition_id" in row]
        conditions = {row["condition_id"]: row for row in condition_rows}
        if len(conditions) != len(condition_rows):
            raise ValueError("Problem attachment/release condition IDs must be unique")
        initial_poses, used_attachments = {}, set()

        def matrix(position, quaternion):
            result = np.eye(4)
            result[:3, 3] = np.asarray(position)
            result[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
            return result

        def same_pose(first, second, label):
            if not np.allclose(first, second, atol=1e-7, rtol=0):
                raise ValueError(f"Collision owner and problem {label} disagree")

        # BEGIN prevent stale or predicted poses from widening current coverage.
        for identity, evidence in self.coverage.occupancy_evidence["owners"].items():
            p, q = by_id[identity].predicted_pose(torch.tensor(0., dtype=torch.float64))
            same_pose(matrix(p.numpy(), q.numpy()),
                      _pose_array(evidence["T_output_owner"], "occupied current owner pose"),
                      "occupied-evidence current pose")
        robot_evidence = self.coverage.occupancy_evidence["robot"]
        if robot_evidence is not None:
            names = tuple(robot_evidence["joint_names"])
            if set(names) != set(problem.joint_names):
                raise ValueError("Occupied robot evidence requires the same actual feedback joint layout")
            values = np.asarray(robot_evidence["joint_position"], float)
            indices = [names.index(name) for name in problem.joint_names]
            # These are observations of the same snapshot, not approximate
            # target constraints. A later or edited q needs new FK coverage.
            if values.shape != (len(names),) or not np.array_equal(values[indices], problem.start_q):
                raise ValueError("Occupied robot evidence must match the exact current joint feedback")
            if not np.array_equal(np.asarray(robot_evidence["T_output_robot_base"]), problem.base_pose):
                raise ValueError("Occupied robot evidence must use the same current base calibration")
        # END prevent stale or predicted poses from widening current coverage.
        for index, phase in enumerate(problem.phases):
            meta = phase.metadata
            carrier = meta.get("carrier_id")
            if meta.get("mode") in {"grasp", "held", "released_retreat"} and not carrier:
                raise ValueError("A grasp/held/released phase requires its physical carrier owner")
            if carrier is None:
                continue
            if carrier not in by_id or carrier in background_ids:
                raise ValueError("Every problem carrier must be an explicit owner outside the background")
            owner = by_id[carrier]
            initial = _pose_array(meta.get("owner_initial_pose"), "owner_initial_pose")
            if carrier not in initial_poses:
                p, q = owner.predicted_pose(torch.tensor(0., dtype=torch.float64))
                same_pose(matrix(p.numpy(), q.numpy()), initial, "initial observed pose")
                initial_poses[carrier] = initial
                # Grounded task carriers have no independently predicted
                # external motion before their first attachment. Check every
                # interpolation knot, not only equal endpoint poses: moving
                # an obstacle away and back would otherwise evade collision.
                path = owner.pose_path
                if not isinstance(path, PosePath):
                    raise ValueError("A task carrier requires an inspectable current-scene PosePath")
                until = owner.attachments[0].interval[0] if owner.attachments else problem.phase_count
                begin, end = owner.pose_interval
                global_knots = begin+(end-begin)*path.progress
                keys = np.unique(np.r_[0., until, global_knots[(global_knots > 0) & (global_knots < until)]])
                local = ((keys-begin)/(end-begin)).clip(0, 1)
                for pose in path.sample_numpy(local):
                    same_pose(pose, initial, "unattached task-carrier pose")
                if path.motion_model is not None:
                    coordinates = np.stack([np.interp(local, path.progress, path.coordinates[:, d])
                                            for d in range(path.motion_model.dimension)], -1)
                    if not np.allclose(coordinates, coordinates[0], atol=1e-10, rtol=0):
                        raise ValueError("An unattached task carrier cannot have independent mechanism motion")
            else:
                same_pose(initial, initial_poses[carrier], "initial observation across phases")
            if meta.get("mode") != "held":
                continue
            required = {"attachment", "attachment_condition_id", "attachment_interval", "release_condition_id"}
            if not required <= meta.keys():
                raise ValueError("A held phase requires its complete conditional attachment contract")
            span = _valid_interval(meta["attachment_interval"])
            if span[0] > index or span[1] < index+1 or span[1] > problem.phase_count:
                raise ValueError("Attachment interval must cover every held phase within the window")
            matches = [(j, relation) for j, relation in enumerate(owner.attachments)
                       if relation.interval == span]
            if len(matches) != 1:
                raise ValueError("Every held phase requires one matching collision attachment span")
            attachment_index, relation = matches[0]
            used_attachments.add((carrier, attachment_index))
            inverse = _inverse(_pose_array(meta["attachment"], "T_owner_tool"))
            same_pose(matrix(relation.position_tool_owner, relation.quaternion_tool_owner), inverse,
                      "attachment transform")
            if (relation.condition_id != meta["attachment_condition_id"]
                    or relation.release_condition_id != meta["release_condition_id"]):
                raise ValueError("Collision attachment and problem condition IDs disagree")
            for condition_id, progress, release in ((relation.condition_id, span[0], False),
                                                     (relation.release_condition_id, span[1], True)):
                if condition_id is None:
                    if not release or progress != problem.phase_count:
                        raise ValueError("Holding without a release condition may end only at the window boundary")
                    if any(row.get("kind") == "predicted_release" and row.get("carrier_id") == carrier
                           and row.get("progress") == progress for row in problem.conditions):
                        raise ValueError("Continued Holding cannot omit the problem's real release condition")
                    continue
                declared = conditions.get(condition_id)
                if (declared is None or declared.get("carrier_id") != carrier
                        or declared.get("progress") != progress):
                    raise ValueError("Attachment/release conditions must identify the same owner and exact boundary")
                if release and declared.get("kind") != "predicted_release":
                    raise ValueError("A release condition must refer to the problem's explicit release event")
                if not release and declared.get("kind") not in {"candidate_attachment", "observed_attachment"}:
                    raise ValueError("An attachment must refer to actual or explicitly conditional Holding evidence")
            # During a held span CollisionOwner.pose uses this exact rollout's
            # FK and the checked rigid attachment. The problem constrains that
            # same FK with its mechanism eta; a second independently sampled
            # mechanism path here would duplicate state rather than bind it.
        declared_attachments = {(owner.owner_id, index) for owner in self.owners
                                for index, _ in enumerate(owner.attachments)}
        if used_attachments != declared_attachments:
            raise ValueError("Every collision attachment must belong to the problem's held phases")
        # BEGIN every geometry command belongs to this exact candidate occurrence.
        expected_commands = Counter((*event_occurrence_key(event), event.kind, event.progress)
                                    for event in problem.events if event.kind in {"open", "close", "hold"})
        actual_commands = Counter() if self.gripper is None else Counter(
            (*event_occurrence_key(event), event["command"], event["progress"])
            for event in self.gripper.commands)
        if actual_commands != expected_commands:
            raise ValueError("Problem and gripper must share exact skill/event/occurrence/command/progress identities")
        # END old command intent cannot accidentally authorize a repeated occurrence.
        hand = self.gripper
        # BEGIN bind optional occupied finger evidence to actual feedback.
        # A native model can lock binary fingers at its default open width,
        # while the observed hand is elsewhere in the reachable interval.
        # Such a model still supports the existing conservative sweep checks,
        # but cannot prove occupied volume at its unobserved default pose.
        if hand is not None and robot_evidence is not None:
            native_current = dict(zip(robot_evidence["joint_names"], robot_evidence["joint_position"]))
            native_current.update(robot_evidence["locked_joint_positions"])
            if (not set(hand.joint_names) <= set(native_current)
                    or not np.allclose([native_current[name] for name in hand.joint_names],
                                       hand.current_positions, atol=1e-7, rtol=0)):
                raise ValueError("Occupied gripper evidence must match actual current fingers, including native locked joints")
        # END bind optional occupied finger evidence to actual feedback.
        if hand is not None and hand.controller == "position_tracking":
            if not set(hand.joint_names) <= set(problem.joint_names):
                raise ValueError("Tracking gripper joints must be actual native problem joints")
            indices = [problem.joint_names.index(name) for name in hand.joint_names]
            if not np.allclose(problem.start_q[indices], hand.current_positions,
                               atol=hand.numerical_joint_tolerance, rtol=0):
                raise ValueError("Tracking gripper and problem must use the same current native feedback")
            required_rows = Counter(tuple(row)+tuple([target]) for row, target in
                                    zip(hand.coupling_matrix, hand.coupling_offset))
            actual_rows = Counter()
            for coupling in problem.couplings:
                columns = {name: index for index, name in enumerate(coupling.joint_names)}
                for row, target in zip(coupling.matrix, coupling.target):
                    if not any(name in columns and row[columns[name]] != 0 for name in hand.joint_names):
                        continue
                    if any(value != 0 and name not in hand.joint_names for name, value in zip(coupling.joint_names, row)):
                        raise ValueError("Tracking drive coupling may not introduce unrelated native joints")
                    actual_rows[tuple(row[columns[name]] if name in columns else 0.
                                      for name in hand.joint_names)+(target,)] += 1
            if actual_rows != required_rows:
                raise ValueError("Tracking gripper and problem must share the same native A q=b coupling")
    # END bind one collision scene to the exact generated skill problem.

    def clone_for_rollout(self):
        # BEGIN isolate candidate policy and relationships without GPU copies.
        # Native SceneCollision geometry is shared read-only by its query
        # clones. A world update must build a new request, never mutate this
        # shared field while any rollout/backward still consumes it. The
        # robot oracle is likewise shared so its cumulative request budget
        # cannot be restarted by making another candidate.
        result = copy(self)
        result.background = self.background.clone_for_rollout()
        result.owners = tuple(copy(owner) for owner in self.owners)
        for owner in result.owners:
            owner.query = owner.query.clone_for_rollout()
            owner.pose_path = deepcopy(owner.pose_path)
            owner.attachments = deepcopy(owner.attachments)
            owner.local_spheres = (None if owner.local_spheres is None else owner.local_spheres.copy())
        result.contacts = deepcopy(self.contacts)
        result.gripper = deepcopy(self.gripper)
        result.coverage = self.coverage.clone_for_rollout()
        result.observation_report = deepcopy(self.observation_report)
        result._coverage_geometry = deepcopy(self._coverage_geometry)
        result._coverage_samples = tuple(points.copy() for points in self._coverage_samples)
        # Tensor.clone keeps the candidate's release-state autograd lineage;
        # deepcopy cannot copy non-leaf tensors and detach would drop gradients.
        result._fixed_release_poses = {
            key: tuple(value.clone() for value in pose)
            for key, pose in self._fixed_release_poses.items()}
        return result
        # END private rule/anchor edits cannot contaminate parent or siblings.

    # BEGIN native solid guidance plus a mandatory independent volume verdict.
    def robot_coverage_guidance(self, robot_link_poses, progress):
        """Guide native body samples toward F or V0; this is not a certificate.

        At most 32 real mesh points per body enter a checkpointed query. The
        exact original convex solids and raw free voxels remain owned by the
        CPU oracle, which checks every final validation state separately.
        """
        if self.robot_coverage is None:
            return torch.zeros_like(progress)
        if robot_link_poses is None or set(robot_link_poses) != set(self.robot_coverage.required_link_names):
            raise ValueError("Native solid guidance requires every actual named body pose")
        solids = self._coverage_geometry["solids"]
        planes = [_tensor(row["baseline_halfspaces_world"], progress) for row in solids]
        distance_grid = self._free_distance.to(device=progress.device, dtype=progress.dtype)
        mask_grid = self._free_mask.to(device=progress.device, dtype=progress.dtype)
        origin = _tensor(self.coverage.origin, progress)
        shape = _tensor(self.coverage._shape_xyz, progress)

        def point_violation(points):
            coordinates = (points-origin)/self.coverage.voxel_size
            clipped = torch.maximum(torch.zeros_like(coordinates), torch.minimum(coordinates, shape-1))
            grid = 2*clipped/(shape-1)-1
            sample_grid = grid.reshape(1, 1, 1, -1, 3)
            distance = F.grid_sample(distance_grid, sample_grid, align_corners=True,
                                     mode="bilinear", padding_mode="border").reshape(points.shape[:-1])
            inside = F.grid_sample(mask_grid, sample_grid, align_corners=True,
                                   mode="nearest", padding_mode="zeros").reshape(points.shape[:-1]) > .5
            outside = torch.linalg.vector_norm(points-(origin+clipped*self.coverage.voxel_size), dim=-1)
            free_cost = torch.where(inside & (outside == 0), torch.zeros_like(distance), distance+outside)
            if not self._has_free_cells:
                free_cost = torch.full_like(free_cost, float("inf"))
            initial_cost = torch.full_like(free_cost, float("inf"))
            for halfspaces in planes:
                signed_max = torch.full_like(free_cost, -float("inf"))
                # Keep one bounded point/plane tensor live, choosing a larger
                # exact batch for small IK queries instead of launching a
                # Python/GPU loop for each 32-face fragment of the same hull.
                chunk = _coverage_plane_chunk_size(math.prod(points.shape[:-2]),
                                                    points.shape[-2], len(halfspaces))
                for begin in range(0, len(halfspaces), chunk):
                    signed_max = torch.maximum(signed_max,
                        _coverage_halfspace_max(points, halfspaces[begin:begin+chunk]))
                initial_cost = torch.minimum(initial_cost, torch.relu(signed_max))
            return torch.minimum(free_cost, initial_cost).amax(-1)

        result = torch.zeros_like(progress)
        for solid, points in zip(solids, self._coverage_samples):
            p, r = robot_link_poses[solid["frame"]]
            world = _world_points(_tensor(points, progress), p, r)
            value = (checkpoint(point_violation, world, use_reentrant=False, preserve_rng_state=False)
                     if torch.is_grad_enabled() and world.requires_grad else point_violation(world))
            result = torch.maximum(result, value)
        return result

    def full_robot_positions(self, joint_position, joint_names, trajectory_time):
        """Compose named arm and nominal finger coordinates without guessing."""
        names = tuple(joint_names)
        values = joint_position
        if self.gripper is not None and (self.gripper.controller == "timed_reference" or self.gripper.nominal_finger_trajectory is not None):
            hand = self.gripper
            if set(names) & set(hand.joint_names):
                raise ValueError("Independent finger coordinates must not duplicate active arm coordinates")
            model = hand.nominal_finger_trajectory or hand.finger_trajectory
            fingers, _ = model.sample_torch(trajectory_time)
            values = torch.cat((values, fingers), -1)
            names += hand.joint_names
        return names, values

    def check_robot_coverage(self, joint_position, joint_names, trajectory_time, *, entrance=False):
        """Check actual coordinates; nominate only exact initial representations.

        A native curve's first sample may differ from measured q0 even when
        its boundary conditions use q0. Such a sample remains an ordinary
        full-volume query; the entrance request never grants a tolerance band.
        """
        if self.robot_coverage is None:
            return []
        hand = self.gripper
        independent_unknown = (hand is not None and not set(hand.joint_names) <= set(joint_names)
                               and hand.nominal_finger_trajectory is None)
        if independent_unknown:
            names = tuple(joint_names) + hand.joint_names
            measured = _tensor(hand.current_positions, joint_position).expand(*joint_position.shape[:-1], len(hand.joint_names))
            positions = torch.cat((joint_position, measured), -1)
        else:
            names, positions = self.full_robot_positions(joint_position, joint_names, trajectory_time)
        values = positions.detach().cpu().numpy().reshape(-1, len(names))
        times = trajectory_time.detach().cpu().numpy().reshape(-1)
        reports = []
        for row, time_s in zip(values, times):
            if independent_unknown and time_s > 0:
                violations = {name: float(value) for name, value in zip(joint_names, row[:len(joint_names)])
                              if not self.robot_coverage._checker.joint_limits[name][0] <= value <= self.robot_coverage._checker.joint_limits[name][1]}
                reports.append({"passed": False, "attempt_allowed": not violations,
                                "failure": "native_joint_limits" if violations else "future_finger_geometry_unbound",
                                "joint_limit_violations": violations, "scope": "no guessed future finger volume", "trajectory_time": float(time_s)})
                continue
            query = row
            representation = None
            # BEGIN preserve unmatched native first samples as ordinary states.
            query_entrance = False
            if entrance:
                columns = [self.robot_coverage.baseline_joint_names.index(name) for name in names]
                baseline = self.robot_coverage.baseline_joint_positions[columns]
                rounded = baseline.astype(np.float32).astype(float)
                # A mixed arm-native/CPU-finger entrance is accepted only if
                # every original value is exactly one declared representation.
                # Rounding an arbitrary nearby value first would hide motion.
                if np.all((row == baseline) | (row == rounded)):
                    query = baseline if np.array_equal(row, baseline) else rounded
                    representation = "measured_float64" if np.array_equal(row, baseline) else "exact_named_native_float32_binding"
                    query_entrance = True
            report = self.robot_coverage.check_state(joint_names=names, joint_positions=query, entrance=query_entrance)
            # END no coordinate is rounded before deciding entrance identity.
            if representation is not None:
                report = {**report, "entrance_representation": representation}
            reports.append(report)
        return reports
    # END reference volume verification never asserts controller tracking.

    def estimated_workspace_bytes(self, batch, horizon, *, robot_coverage_guidance=True):
        """Bound geometry-query work only; solver/model storage is additional.

        Includes one live gradient snapshot per query, reusable query buffers,
        whole-sphere coverage interpolation and its grid. Backend consumers
        must sum their independent rollout allocations and their own robot,
        optimizer, FK and trajectory allocations before choosing GPU budgets.
        """
        if type(batch) is not int or type(horizon) is not int or min(batch, horizon) <= 0:
            raise ValueError("Workspace estimation needs positive batch/horizon sizes")
        actors = {"robot": len(self.robot_sphere_materials)}
        actors.update({owner.owner_id: len(owner.local_spheres) for owner in self.owners if owner.moving})
        count = sum(size * (1 + len(self.owners) - int(actor != "robot")) for actor, size in actors.items())
        query_bytes = batch * horizon * count * 40
        # BEGIN count only actors that still execute whole-sphere coverage.
        # Native robot-volume coverage replaces that one query in components;
        # moving payloads keep their original unknown-space checks. Retain the
        # conservative shared-grid allowance and the separate native guidance
        # storage below. No collision term or allowed volume changes here.
        coverage_spheres = sum(size for actor, size in actors.items()
                               if actor != "robot" or self.robot_coverage is None)
        coverage_bytes = (batch * horizon * coverage_spheres * 128
                          + (1 + len(actors)) * self.coverage._distance.numel() * 4)
        # END disabled robot sphere queries cannot consume per-state scratch.
        # BEGIN include checkpointed hand-pair scratch and its retained input.
        pair_bytes = 0
        if self.gripper is not None and self.gripper.self_collision_pairs:
            pairs = len(self.gripper.self_collision_pairs)
            chunk = _self_pair_chunk_size(batch*horizon, pairs)
            # 24 floats per expanded pair cover indexed balls, displacement,
            # norm/ReLU and backward scratch. Reserve sphere values plus
            # accumulated gradients and explicit int64 pair indices as well.
            # Checkpointing frees chunk scratch rather than retaining all P.
            pair_bytes = batch*horizon*(96*chunk+32*actors["robot"])+16*pairs
        native_coverage_bytes = 0
        if self.robot_coverage is not None and robot_coverage_guidance:
            solids = self._coverage_geometry["solids"]
            plane_counts = [len(row["baseline_halfspaces_world"]) for row in solids]
            planes = max(plane_counts)
            frames = len(self.robot_coverage.required_link_names)
            joints = len(self.robot_coverage.baseline_joint_names)
            # One checkpointed body query is live; all frame poses/Jacobians
            # and their gradients remain resident for its backward pass.
            states = batch*horizon
            # Use the target itself as an upper bound, including rounding of
            # the per-body chunk count. Twenty-point pads may choose a larger
            # chunk than 32-point bodies, so a single 32-point chunk formula
            # would slightly underestimate their live tensor at some sizes.
            plane_elements = max(states*32, min(_COVERAGE_POINT_PLANE_ELEMENTS, states*32*planes))
            # A body's checkpoint may restore its complete reduction chain.
            # Per chunk, reserve 32 bytes per point for int64 argmax indices
            # and retained max values/gradients, plus 32 per baseline body for
            # relu/min chains and 128 for free-grid point operations. Bound
            # every possible sample count (pads have 20, most bodies 32).
            chain_bytes = 0
            for points in range(1, 33):
                chunks = sum(math.ceil(count/_coverage_plane_chunk_size(states, points, count))
                             for count in plane_counts)
                chain_bytes = max(chain_bytes, states*points*(32*chunks+32*len(solids)+128))
            native_coverage_bytes = (plane_elements*8 + chain_bytes
                + states*(32*32*8 + frames*(32+12*joints)*4)
                + sum(plane_counts)*24)
            native_coverage_bytes += self._free_distance.numel()*8
        return int(query_bytes + coverage_bytes + pair_bytes + native_coverage_bytes)
        # END include checkpointed hand-pair scratch and its retained input.

    def project_residual(self, progress, joint_position, joint_names):
        """Independent executable finger-drive equalities for OMPL projection.

        Output has J-rank(A) rows for tracking and zero rows otherwise. The
        caller must include the same native A q=b coupling in its problem;
        duplicate coupling rows are intentionally not introduced here. These
        are exact equalities, with projection convergence handled separately
        from the native controller command or task's permissible motion.
        """
        if self.gripper is None or self.gripper.controller != "position_tracking":
            return joint_position[..., :0]
        hand = self.gripper
        indices = [tuple(joint_names).index(name) for name in hand.joint_names]
        lower, upper = hand.joint_bounds(progress)
        command = (lower + upper) / 2
        return ((joint_position[..., indices] - command)
                @ _tensor(hand._drive_basis.T, joint_position))

    @staticmethod
    def _release_pose(relation, position, quaternion, progress):
        """Gather exact release FK; a missing anchor is an explicit violation.

        The first occurrence is the physical release. Later samples on the
        same progress plateau may move the hand while the owner stays fixed.
        When absent, the gathered numeric value is not accepted as
        a release prediction: ``found`` is false and subsequent samples fail.
        Keeping arithmetic finite avoids poisoning an earlier, still-attached
        query's derivatives when it does not need a release state yet.
        """
        if relation.release_condition_id is None:
            raise ValueError("Continued Holding has no physical release anchor")
        # Match while progress is still in its input dtype. Converting a rounded
        # FP32 event (e.g. 17+1/3) to double before matching invents a discrepancy.
        matches = (progress - relation.interval[1]).abs() <= 1e-7
        index = matches.to(torch.int64).argmax(-1)
        p = torch.gather(position, 1, index[:, None, None].expand(-1, 1, 3))
        q = torch.gather(quaternion, 1, index[:, None, None].expand(-1, 1, 4))
        return (p + quaternion_rotate(q, _tensor(relation.position_tool_owner, p)),
                quaternion_multiply(q, _tensor(relation.quaternion_tool_owner, q))), matches.any(-1)

    # BEGIN selected search prefixes own their actual release history.
    def with_release_history(self, position, quaternion, progress):
        """Copy release FK from an already selected prefix into a child branch.

        Future releases remain variables. The first reached occurrence of a
        release owns its pose, including on a progress plateau; extending a
        branch cannot replace it with the later hand pose. Full optimization
        still uses the original scene, where releases remain shared decisions.
        """
        result = self.clone_for_rollout()
        for owner in self.owners:
            for index, relation in enumerate(owner.attachments):
                key = (owner.owner_id, index)
                if relation.release_condition_id is None or key in result._fixed_release_poses:
                    continue
                if not bool((progress >= relation.interval[1]-1e-7).any()):
                    continue
                pose, found = self._release_pose(relation, position, quaternion, progress)
                if not bool(found.all()):
                    raise ValueError("A selected search prefix must include its exact release anchor")
                result._fixed_release_poses[key] = pose
        return result
    # END neither a source reference nor a sibling branch supplies release FK.

    def _permission(self, rule, actor_spheres, actor_pose, target_pose, progress, coordinates, half):
        points = _tensor(rule.feature_points, progress)
        if rule.feature_joint_translation is not None:
            mapping = _tensor(rule.feature_joint_translation, progress)
            points = points + torch.einsum("vcj,bhj->bhvc", mapping, coordinates)
            # Material permission is not granted to an uncertain width. The
            # full conservative envelope continues to participate in collision.
            certain = half.amax(-1) == 0
        else:
            certain = torch.ones_like(progress, dtype=torch.bool)
        world = _world_points(points, *actor_pose)
        target_local = quaternion_rotate(quaternion_conjugate(target_pose[1]).unsqueeze(-2),
                                         world - target_pose[0].unsqueeze(-2))
        patch_points = ((target_local - _tensor(rule.patch_origin, progress))
                        @ _tensor(rule.patch_axes, progress))
        footprint = torch.relu(patch_points[..., :2].abs()
                               - _tensor(rule.patch_half_size, progress)).amax(dim=(-1, -2))
        # A large proxy must not borrow a small feature's permission while its
        # other side touches the same owner outside the observed patch. Requiring
        # the entire projected sphere disk inside the patch is conservative and
        # needs no unavailable per-obstacle witness from an accumulated query.
        proxies = _local_spheres(actor_spheres[..., list(rule.sphere_indices), :], *target_pose)
        proxy_points = ((proxies[..., :3] - _tensor(rule.patch_origin, progress))
                        @ _tensor(rule.patch_axes, progress))
        proxy_footprint = torch.relu(proxy_points[..., :2].abs() + proxies[..., 3:4]
                                     - _tensor(rule.patch_half_size, progress)).amax(dim=(-1, -2))
        footprint = torch.maximum(footprint, proxy_footprint)
        heights = patch_points[..., 2]
        gap = (torch.relu(rule.gap_bounds[0] - heights).amax(-1)
               + torch.relu(heights - rule.gap_bounds[1]).amax(-1))
        actor_normal = quaternion_rotate(actor_pose[1], _tensor(rule.feature_normal, progress))
        target_normal = quaternion_rotate(target_pose[1], _tensor(rule.patch_axes[:, 2], progress))
        angle = torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(actor_normal, target_normal), dim=-1),
                            -(actor_normal * target_normal).sum(-1))
        normal = torch.relu(angle - rule.normal_tolerance_rad - rule.numerical_angle_tolerance_rad)
        residual = (torch.relu(footprint - rule.numerical_position_tolerance_m)
                    + torch.relu(gap - rule.numerical_position_tolerance_m) + normal)
        active = _interval(progress, rule.interval)
        allowed = active & certain & (residual <= 0)
        return (allowed, residual * active.to(progress.dtype),
                (active & ~certain).to(progress.dtype))

    def components(self, robot_spheres, position, quaternion, progress,
                   joint_position=None, joint_names=None, *, trajectory_time=None):
        """Return nonnegative [B,H] terms; no term can cancel a violation.

        Required event anchors must be included by the caller. On a mode
        boundary the caller evaluates the same q with both adjacent progress
        modes. Queries are discrete; final curve subdivision belongs to the
        planner. Returned diagnostics stay tensors and retain FK gradients.
        """
        if (robot_spheres.ndim != 4 or robot_spheres.shape[-1] != 4
                or robot_spheres.shape[:2] != progress.shape
                or robot_spheres.shape[2] != len(self.robot_sphere_materials)
                or position.shape != (*progress.shape, 3)
                or quaternion.shape != (*progress.shape, 4)):
            raise ValueError("Scene needs aligned FK spheres, tool pose and progress [B,H]")
        costs = {}
        coordinates = half = None
        spheres = robot_spheres
        if self.gripper is not None:
            coordinates, half, costs["gripper/controller"] = self.gripper.state(
                progress, joint_position, joint_names, trajectory_time=trajectory_time)
            spheres = self.gripper.spheres(spheres, position, quaternion, coordinates, half)
            # BEGIN retain every declared pair without a per-pair Python graph.
            if self.gripper.self_collision_pairs:
                costs["gripper/self"] = _self_pair_sum(spheres, self.gripper._self_pair_indices)
            # END retain every declared pair without a per-pair Python graph.
        # BEGIN an unmodelled future width does not establish a physical collision.
        uncertain_hand = torch.zeros_like(progress, dtype=torch.bool)
        uncertain_indices = []
        if self.gripper is not None:
            hand = self.gripper
            if hand.controller in {"binary", "timed_reference"} and hand.nominal_finger_trajectory is None:
                uncertain_hand = (torch.ones_like(progress, dtype=torch.bool) if trajectory_time is None else trajectory_time > 0)
                uncertain_indices = [index for index, mapping in zip(hand.sphere_indices, hand.sphere_joint_translation)
                                     if np.any(mapping != 0)]
                if hand.self_collision_pairs:
                    affected = [pair for pair in hand.self_collision_pairs if set(pair) & set(uncertain_indices)]
                    unresolved = _self_pair_sum(spheres, np.asarray(affected, dtype=np.int64).reshape(-1, 2)) if affected else torch.zeros_like(progress)
                    costs["unknown/gripper/self"] = unresolved * uncertain_hand
                    costs["gripper/self"] = costs["gripper/self"] - costs["unknown/gripper/self"]
        def collision_terms(values, key, actor):
            if actor == "robot" and uncertain_indices:
                unresolved = values[..., uncertain_indices].sum(-1) * uncertain_hand
                costs["unknown/" + key] = unresolved
                costs[key] = values.sum(-1) - unresolved
            else:
                costs[key] = values.sum(-1)
        # END fixed palm/body conflicts remain known at every query.
        poses = {}
        release_states = {}
        for owner in self.owners:
            releases = []
            for index, relation in enumerate(owner.attachments):
                key = (owner.owner_id, index)
                if relation.release_condition_id is None:
                    released = None
                elif key in self._fixed_release_poses:
                    released = self._fixed_release_poses[key]
                else:
                    released, found = self._release_pose(relation, position, quaternion, progress)
                    costs[f"unknown/release_anchor/{owner.owner_id}/{index}"] = (
                        (progress >= relation.interval[1]) & ~found[:, None]).to(progress.dtype)
                releases.append(released)
            release_states[owner.owner_id] = releases
            poses[owner.owner_id] = owner.pose(position, quaternion, progress, releases)
        actors = {"robot": (spheres, (position, quaternion))}
        for owner in self.owners:
            if owner.moving:
                cover = owner.local_spheres.copy()
                cover[:, 3] += owner.sphere_cover_error_m
                actors[owner.owner_id] = (_world_spheres(cover, *poses[owner.owner_id]), poses[owner.owner_id])
            # BEGIN require collision completeness for occupied owner evidence.
            # Otherwise a static partial front surface could let a robot enter
            # its now-known interior through an unmodelled back surface. An
            # inner occupancy mesh is never substituted for an outer collision
            # model. This is diagnostic uncertainty, not a global denial gate.
            if not owner.geometry_complete and (owner.moving or owner.owner_id in self._occupied_owner_ids):
                costs[f"unknown/owner/{owner.owner_id}"] = torch.ones_like(progress)
            # END require collision completeness for occupied owner evidence.
            for index, relation in enumerate(owner.attachments):
                if index == 0:
                    predicted_p, predicted_q = owner.predicted_pose(progress)
                else:
                    # Regrasp begins from the previous predicted release, not
                    # from the original observation or an independent proposal.
                    predicted_p, predicted_q = release_states[owner.owner_id][index - 1]
                held_p = position + quaternion_rotate(quaternion, _tensor(relation.position_tool_owner, position))
                held_q = quaternion_multiply(quaternion, _tensor(relation.quaternion_tool_owner, quaternion))
                # Attachment must start at the observed owner. Release instead
                # uses actual anchor FK above; constraining it to a nominal
                # object pose would silently remove allowed support freedom.
                boundary = (progress - relation.interval[0]).abs() <= 1e-7
                translation = torch.linalg.vector_norm(held_p - predicted_p, dim=-1)
                rotation = torch.linalg.vector_norm(quaternion_rotvec(quaternion_multiply(
                    quaternion_conjugate(predicted_q), held_q)), dim=-1)
                error = (torch.relu(translation - self.numerical_position_tolerance_m)
                         + torch.relu(rotation - self.numerical_rotation_tolerance_rad))
                costs[f"attachment/start/{owner.owner_id}/{index}"] = error * boundary.to(progress.dtype)
        for actor, (actor_spheres, actor_pose) in actors.items():
            # A point objective cannot decide whole-volume coverage. Native
            # metrics retain all other scene terms; the planner must invoke
            # check_robot_coverage at entrance, search and final acceptance.
            if actor != "robot" or self.robot_coverage is None:
                costs[f"unknown/space/{actor}"] = self.coverage.violation(actor_spheres).amax(-1)
            collision_terms(self.background(actor_spheres, key=(actor, "background")),
                            f"collision/{actor}/background", actor)
            for owner in self.owners:
                if actor == owner.owner_id:
                    continue
                adjusted = actor_spheres
                for rule in self.contacts:
                    if rule.actor != actor or rule.target_owner != owner.owner_id:
                        continue
                    allowed, residual, uncertain = self._permission(
                        rule, actor_spheres, actor_pose, poses[owner.owner_id], progress, coordinates, half)
                    if rule.required:
                        unresolved = uncertain_hand if actor == "robot" and rule.feature_joint_translation is not None else torch.zeros_like(progress, dtype=torch.bool)
                        costs[f"contact/{rule.rule_id}"] = residual * ~unresolved
                        costs[f"unknown/contact/{rule.rule_id}"] = torch.maximum(uncertain, residual * unresolved)
                    indices = list(rule.sphere_indices)
                    radius = actor_spheres[..., indices, 3]
                    costs[f"contact/proxy/{rule.rule_id}"] = torch.relu(rule.modeling_allowance_m + 1e-9 - radius).sum(-1)
                    adjusted = adjusted.clone()
                    # BEGIN combine independent permissions without resetting or stacking.
                    # Each rule shrinks the original proxy at most once. A
                    # later inactive rule cannot undo an earlier active one,
                    # and adjacent intervals meeting at one anchor select the
                    # larger single allowance rather than adding both.
                    candidate = radius - allowed[..., None].to(radius.dtype) * rule.modeling_allowance_m
                    adjusted[..., indices, 3] = torch.minimum(adjusted[..., indices, 3], candidate)
                    # END preserve every active material permission independently.
                local = _local_spheres(adjusted, *poses[owner.owner_id])
                # BEGIN retain native welded/masked installation pairs without deleting geometry.
                # Applying the mask to per-sphere costs leaves the same owner
                # active for all other robot links and for a carried payload.
                values = owner.query(local, key=(actor, owner.owner_id))
                if actor == "robot" and owner.robot_excluded_spheres:
                    enabled = values.new_ones(values.shape[-1])
                    enabled[list(owner.robot_excluded_spheres)] = 0
                    values = values * enabled
                collision_terms(values, f"collision/{actor}/{owner.owner_id}", actor)
                # END the observed background and unknown-space queries are unchanged.
        return costs

    def evaluate(self, robot_spheres, position, quaternion, progress,
                 joint_position=None, joint_names=None, *, trajectory_time=None):
        terms = self.components(robot_spheres, position, quaternion, progress,
                                joint_position=joint_position, joint_names=joint_names,
                                trajectory_time=trajectory_time)
        # BEGIN allowed observation/model uncertainty stays diagnostic.
        # A missing release anchor is different: the requested pose query
        # cannot be formed and must still fail this candidate.
        known = [value for key, value in terms.items()
                 if not key.startswith("unknown/") or key.startswith("unknown/release_anchor/")]
        return torch.stack(known).sum(0) if known else torch.zeros_like(progress)
        # END known collision, required relation and physical limits stay hard.

    # BEGIN optimize toward an actual tracking command inside its numerical band.
    def optimizer_preferences(self, progress, joint_position, joint_names):
        """Squared tracking error for optimizer use only; no acceptance change.

        A tracking drive already specifies one executable command at every
        progress. Its hard numerical band remains in components(). This small
        objective keeps optimization attracted to that command after entering
        the feasible band, instead of stopping at a float32-sensitive boundary.
        Binary drives supply no command width and therefore no such objective.
        Material contact and functional pose freedoms are never centered here.
        """
        hand = self.gripper
        if hand is None or hand.controller != "position_tracking":
            return torch.zeros_like(progress)
        if joint_position is None or joint_names is None:
            raise ValueError("Tracking preferences require the actual named solver joints")
        lower, upper = hand.joint_bounds(progress)
        if lower.shape != (*progress.shape, len(hand.joint_names)) or upper.shape != lower.shape:
            raise ValueError("Tracking command shape must match progress and native fingers")
        indices = [tuple(joint_names).index(name) for name in hand.joint_names]
        command = (lower+upper)/2  # Same single-command contract as hand.state().
        return ((joint_position[..., indices]-command)**2).sum(-1)
    # END optimize toward an actual tracking command inside its numerical band.

    # BEGIN exact bounded scene inputs for independent execution review.
    def to_execution_binding(self):
        """Own the model/permission inputs; labels and counts cannot bind geometry.

        Unsupported geometry remains an explicit required-query gap. Observation
        coverage remains unresolved evidence and never supplies contact permission.
        Native cuboids and complete convex meshes retain their filled-volume
        semantics. We do not take the hull of an arbitrary nonconvex mesh.
        """
        def plain(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, Mapping):
                return {str(k): plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value

        def geometry(query):
            explicit = getattr(query, "independent_geometry", None)
            if explicit is not None:
                return deepcopy(explicit)
            native = getattr(query, "scene_collision", None)
            model = getattr(native, "scene_model", None)
            if model is None or isinstance(model, list):
                return {"solids": [], "query_gaps": ["No single independent native scene model"]}
            solids, gaps = [], []
            for kind in ("cuboid", "cylinder", "sphere", "capsule", "mesh", "voxel"):
                for shape in getattr(model, kind, ()) or ():
                    if len(solids) >= 4096:
                        gaps.append("Independent scene solid budget exceeded")
                        break
                    row = {"name": shape.name, "kind": kind, "pose": list(shape.pose), "source": query.source}
                    if kind == "cuboid":
                        row["dimensions"] = list(shape.dims)
                    elif kind in {"cylinder", "sphere"}:
                        row["dimensions"] = [shape.radius] + ([shape.height] if kind == "cylinder" else [])
                    elif kind == "mesh":
                        mesh = shape.get_trimesh_mesh(process=False, transform_with_pose=False)
                        if len(mesh.vertices) > 65536 or not mesh.is_watertight or not mesh.is_convex:
                            gaps.append(f"{shape.name}: requires complete convex solid adaptation")
                            continue
                        row["vertices"] = np.asarray(mesh.vertices).tolist()
                    else:
                        gaps.append(f"{shape.name}: {kind} requires an independent solid adapter")
                        continue
                    solids.append(row)
            return {"solids": solids, "query_gaps": gaps}

        owners = []
        for owner in self.owners:
            path = owner.pose_path
            owners.append({"owner_id": owner.owner_id, "source": owner.source,
                           "geometry_complete": owner.geometry_complete, "moving": owner.moving,
                           "geometry": geometry(owner.query),
                           "pose_path": plain({"progress": path.progress, "poses": path.poses,
                               "source": path.source, "attachment": path.attachment,
                               "coordinates": path.coordinates,
                               "motion_model": None if path.motion_model is None else asdict(path.motion_model)}),
                           "pose_interval": list(owner.pose_interval),
                           "attachments": plain([asdict(row) for row in owner.attachments]),
                           "local_spheres": plain(owner.local_spheres), "sphere_materials": list(owner.sphere_materials),
                           "sphere_cover_error_m": owner.sphere_cover_error_m,
                           "robot_excluded_spheres": list(owner.robot_excluded_spheres),
                           "robot_exclusion_source": owner.robot_exclusion_source,
                           "native_geometry_names": list(owner.native_geometry_names)})
        hand = self.gripper
        gripper = None if hand is None else plain({
            "controller": hand.controller, "joint_names": hand.joint_names,
            "control_timestep_s": hand.control_timestep_s,
            "current_positions": hand.current_positions, "joint_limits": hand.joint_limits,
            "coupling_matrix": hand.coupling_matrix, "coupling_offset": hand.coupling_offset,
            "commands": hand.commands, "source": hand.source,
            "sphere_indices": hand.sphere_indices, "local_spheres": hand.local_spheres,
            "sphere_joint_translation": hand.sphere_joint_translation,
            "nominal_model": (None if hand.nominal_finger_trajectory is None else
                {"times": hand.nominal_finger_trajectory.times,
                 "position": hand.nominal_finger_trajectory.position,
                 "velocity": hand.nominal_finger_trajectory.velocity}),
            "command_reference": (None if hand.finger_trajectory is None else
                {"times": hand.finger_trajectory.times, "position": hand.finger_trajectory.position,
                 "velocity": hand.finger_trajectory.velocity})})
        return {"reference_frame": self.reference_frame, "robot_geometry_source": self.robot_geometry_source,
                "native_geometry_materials": deepcopy(self.native_geometry_materials),
                "robot_sphere_materials": list(self.robot_sphere_materials), "gripper": gripper,
                "background": {"owner_ids": sorted(self.background.owner_ids), "source": self.background.source,
                               "geometry": geometry(self.background)},
                "owners": owners, "contacts": plain([asdict(rule) for rule in self.contacts]),
                "coverage": {"origin": self.coverage.origin.tolist(), "voxel_size": self.coverage.voxel_size,
                             "shape_zyx": list(self.coverage._known.shape),
                             "known_packbits": np.packbits(self.coverage._known).tolist(),
                             "free_packbits": (None if self.coverage._known_free is None else
                                               np.packbits(self.coverage._known_free).tolist()),
                             "observation_ids": list(self.coverage.observation_ids),
                             "occupancy_evidence": plain(self.coverage.occupancy_evidence)},
                "observation_report": plain(self.observation_report),
                "scope": "Supplied observed/native geometry only; hidden space remains unresolved"}
    # END exact inputs travel with the formal result instead of mutable GPU objects.

    def describe(self):
        """Static scope/evidence for results; never report candidate Holding as verified."""
        return {
            "reference_frame": self.reference_frame,
            "observation_ids": self.coverage.observation_ids,
            "observation_build": deepcopy(self.observation_report),
            "current_occupied_evidence": deepcopy(self.coverage.occupancy_evidence),
            "occupied_owner_ids": tuple(sorted(self._occupied_owner_ids)),
            "background_owner_ids": tuple(sorted(self.background.owner_ids)),
            "owner_geometry_complete": {owner.owner_id: owner.geometry_complete for owner in self.owners},
            # BEGIN report exact installation exclusions alongside ordinary owner geometry.
            "native_installation_exclusions": {
                owner.owner_id: {"robot_sphere_indices": owner.robot_excluded_spheres,
                                 "policy_source": owner.robot_exclusion_source,
                                 "robot_geometry_source": owner.robot_exclusion_geometry_source}
                for owner in self.owners if owner.robot_excluded_spheres},
            # END no owner-wide or background-wide exemption is implied.
            "conditional_attachments": {owner.owner_id: [relation.condition_id for relation in owner.attachments]
                                        for owner in self.owners if owner.attachments},
            "release_requires_evidence": {
                owner.owner_id: [relation.release_condition_id for relation in owner.attachments
                                 if relation.release_condition_id is not None]
                for owner in self.owners if any(relation.release_condition_id is not None for relation in owner.attachments)},
            "attachment_numerical_tolerances": {"position_m": self.numerical_position_tolerance_m,
                                                "rotation_rad": self.numerical_rotation_tolerance_rad},
            "gripper_controller": None if self.gripper is None else self.gripper.controller,
            "gripper_control_timestep_s": None if self.gripper is None else self.gripper.control_timestep_s,
            "gripper_timing_contract": (None if self.gripper is None else
                                        "time-independent diagnostic envelope; future width remains unresolved" if self.gripper.controller == "binary"
                                        else "same-time named command and separately declared nominal geometry; actual q/dq supervised"
                                        if self.gripper.controller == "timed_reference"
                                        else "native position tracking; final solver joint-rate limits apply"),
            "robot_new_volume_coverage": (None if self.robot_coverage is None else {
                "initial_state_assumption": "native solid V0 at measured full named q0 is a valid initial occupied volume",
                "diagnostic": "CPU full-volume V(q) minus (V0 union original observed free cells); deficits remain eligible uncertainty",
                "optimizer": "bounded native body points only; not a coverage certificate"}),
            "numerical_joint_tolerance": None if self.gripper is None else self.gripper.numerical_joint_tolerance,
            "gripper_projection_dimension": (0 if self.gripper is None or self.gripper.controller != "position_tracking"
                                             else len(self.gripper._drive_basis)),
            "gripper_coupling": (None if self.gripper is None else {
                "joint_names": self.gripper.joint_names,
                "matrix": self.gripper.coupling_matrix.tolist(),
                "target": self.gripper.coupling_offset.tolist(),
                "source": self.gripper.source}),
            "required_progress_anchors": self.required_progress_anchors,
            "scope": "observed-space, sphere-proxy discrete checks; not continuous or physical contact certification",
        }
# END observed scene, moving-owner queries and bounded contact permissions.
