"""Frozen approximate collision models for bounded motion attempts.

One native Mapper ESDF or analytic primitive represents each observed owner.
Pair-scoped interaction masks allow ordinary contact without proving task
success. Observation gaps remain diagnostics; no source-solid confirmation,
contact field bank or complete-volume coverage certificate is constructed.
"""
# BEGIN bounded motion scene replaces complete contact/coverage certification.
from copy import copy, deepcopy
from dataclasses import asdict
from itertools import product

import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation

from .collision_records import encode_array, decode_array, MAX_ARRAY_BYTES
from .planning_problem import (PosePath, quaternion_conjugate, quaternion_rotate,
    quaternion_multiply)
from .planning_scene import (Attachment, CollisionOwner, ContactRule, ObservedSpace,
    PrismaticGripper, _tensor, _local_spheres, _world_spheres, _self_pair_sum,
    _self_pair_chunk_size, grasp_contact_rule_ids, validate_grasp_contact_arrival,
    _validate_contact_and_gripper_inputs)
from .holding_contact import HoldingContact, validate_holding_materials, validate_holding_conditions
from .optimizer_device_checks import constant_tensor


def _plain(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


class FrozenESDF:
    """One immutable XYZ distance grid; positive means outside the model.

    This is a numerical motion model, not a proof of physical surface shape.
    Centres outside the finite grid have no collision sample. Their unknown
    observation state is reported separately and cannot be called free space.
    """
    def __init__(self, distance, *, origin, voxel_size, owner_ids, source):
        values = np.asarray(distance, dtype=np.float32)
        if (values.ndim != 3 or min(values.shape) < 2 or values.nbytes > MAX_ARRAY_BYTES
                or not np.isfinite(values).all()):
            raise ValueError('A bounded finite XYZ distance grid is required')
        self.origin = np.asarray(origin, dtype=float).copy()
        self.voxel_size = float(voxel_size)
        if (self.origin.shape != (3,) or not np.isfinite(self.origin).all()
                or not np.isfinite(self.voxel_size) or self.voxel_size <= 0):
            raise ValueError('Finite grid origin and positive voxel size are required')
        self.owner_ids, self.source = frozenset(owner_ids), str(source)
        if not self.owner_ids or not self.source.strip():
            raise ValueError('A frozen field needs explicit owner identities and provenance')
        self.distance = values.copy()
        self.distance.flags.writeable = False
        self.upper = self.origin + (np.asarray(values.shape)-1)*self.voxel_size
        self._device = {}

    @classmethod
    def from_native_voxel(cls, voxel, *, owner_ids, source):
        """Freeze Mapper's XYZ grid with its negative-inside distance convention.

        Mapper.get_voxel_grid places its pose at the grid centre. The first
        sample is half of (shape-1) voxel steps below that centre. Only identity
        voxel rotation is accepted because owner poses already own transforms.
        """
        pose = np.asarray(voxel.pose, dtype=float)
        values = voxel.feature_tensor.detach().cpu().numpy()
        if (pose.shape != (7,) or not np.isfinite(pose).all()
                or not np.array_equal(pose[3:], [1., 0., 0., 0.])):
            raise ValueError('Mapper field must use its owner frame and identity grid rotation')
        h = float(voxel.voxel_size)
        origin = pose[:3]-(np.asarray(values.shape)-1)*h/2
        return cls(values.astype(np.float32), origin=origin, voxel_size=h,
                   owner_ids=owner_ids, source=source)

    def clone_for_rollout(self):
        """Share read-only fields and lazy device tensors across solver seeds."""
        return copy(self)

    def signed_distance(self, points):
        key = (points.device, points.dtype)
        if key not in self._device:
            self._device[key] = torch.as_tensor(np.array(self.distance.transpose(2, 1, 0), copy=True),
                device=points.device, dtype=points.dtype)[None, None]
        index = (points-_tensor(self.origin, points))/self.voxel_size
        grid = 2*index/_tensor(np.asarray(self.distance.shape)-1, points)-1
        return F.grid_sample(self._device[key], grid.reshape(1, 1, 1, -1, 3),
            align_corners=True, padding_mode='border', mode='bilinear').reshape(points.shape[:-1])

    def supports_centres(self, spheres):
        return ((spheres[..., :3] >= _tensor(self.origin, spheres)) &
                (spheres[..., :3] <= _tensor(self.upper, spheres))).all(-1)

    def __call__(self, spheres, *, key=None):
        residual = torch.relu(spheres[..., 3]-self.signed_distance(spheres[..., :3]))
        return torch.where((spheres[..., 3] > 0) & self.supports_centres(spheres),
                           residual, torch.zeros_like(residual))

    def record(self):
        return {'kind': 'frozen_esdf_v1', 'distance': encode_array(self.distance),
                'origin': self.origin.tolist(), 'voxel_size': self.voxel_size,
                'owner_ids': sorted(self.owner_ids), 'source': self.source}

    @classmethod
    def from_record(cls, row):
        if row.get('kind') == 'frozen_primitives_v3':
            return FrozenPrimitives(primitives=row['primitives'], owner_ids=row['owner_ids'],
                                    source=row['source'])
        if row.get('kind') != 'frozen_esdf_v1':
            raise ValueError('Only the current ordinary frozen-field schema is supported')
        return cls(decode_array(row['distance']), origin=row['origin'], voxel_size=row['voxel_size'],
                   owner_ids=row['owner_ids'], source=row['source'])


class FrozenPrimitives(FrozenESDF):
    """Basic explicit solids with analytic distances and no voxel/contact cache.

    Cuboid dimensions are full lengths, box dimensions half lengths. Cylinder
    and capsule dimensions are radius and half-height; poses are xyz+wxyz.
    """
    def __init__(self, *, primitives, owner_ids, source):
        self.primitives=deepcopy(primitives)
        if not self.primitives or len(self.primitives)>64:
            raise ValueError('Explicit bounded primitive inventory required')
        self.owner_ids,self.source=frozenset(owner_ids),str(source)
        self._device={}
        self.distance=np.empty((0,),dtype=np.float32)
        lower,upper=[],[]
        corners=np.asarray(list(product((-1.,1.),repeat=3)))
        for row in self.primitives:
            kind=row['kind']
            dims=np.asarray(row['dimensions'],dtype=float)
            count=3 if kind in {'cuboid','box'} else 1 if kind=='sphere' else 2
            if kind not in {'cuboid','box','sphere','cylinder','capsule'} or dims.shape!=(count,) or not np.isfinite(dims).all() or np.any(dims<=0):
                raise ValueError('Supported primitive kind and positive metric dimensions required')
            pose=np.asarray(row.get('pose',[0.,0.,0.,1.,0.,0.,0.]),dtype=float)
            if pose.shape!=(7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]),1.,atol=1e-6):
                raise ValueError('Primitive pose must be finite xyz+wxyz')
            row['pose']=pose.tolist()
            extent=(dims/2 if kind=='cuboid' else dims if kind=='box' else np.repeat(dims[0],3) if kind=='sphere' else np.array([dims[0],dims[0],dims[1]+(dims[0] if kind=='capsule' else 0)]))
            points=pose[:3]+(corners*extent)@Rotation.from_quat(pose[[4,5,6,3]]).as_matrix().T
            lower.append(points.min(0)); upper.append(points.max(0))
        self.origin=np.min(lower,axis=0)
        self.upper=np.max(upper,axis=0)

    def signed_distance(self,points):
        values=[]
        for row in self.primitives:
            pose=_tensor(row['pose'],points)
            local=quaternion_rotate(quaternion_conjugate(pose[3:]),points-pose[:3])
            dims=_tensor(row['dimensions'],points)
            kind=row['kind']
            if kind in {'box','cuboid'}:
                delta=local.abs()-(dims/2 if kind=='cuboid' else dims)
                value=torch.linalg.vector_norm(torch.relu(delta),dim=-1)+delta.amax(-1).clamp_max(0)
            elif kind=='sphere':
                value=torch.linalg.vector_norm(local,dim=-1)-dims[0]
            elif kind=='capsule':
                delta=torch.stack((local[...,0],local[...,1],torch.relu(local[...,2].abs()-dims[1])),-1)
                value=torch.linalg.vector_norm(delta,dim=-1)-dims[0]
            else:
                delta=torch.stack((torch.linalg.vector_norm(local[...,:2],dim=-1)-dims[0],local[...,2].abs()-dims[1]),-1)
                value=torch.linalg.vector_norm(torch.relu(delta),dim=-1)+delta.amax(-1).clamp_max(0)
            values.append(value)
        return torch.stack(values).amin(0)

    def supports_centres(self, spheres):
        return torch.ones(spheres.shape[:-1], dtype=torch.bool, device=spheres.device)

    def record(self):
        return {'kind': 'frozen_primitives_v3', 'primitives': deepcopy(self.primitives),
                'owner_ids': sorted(self.owner_ids), 'source': self.source}


class OnlinePlanningScene:
    """Known model collisions plus bounded pair-scoped interaction permission."""
    def __init__(self, *, background, owners, coverage, reference_frame,
                 robot_sphere_materials, robot_geometry_source, contacts=(), gripper=None,
                 has_gripper=True, robot_collision_bodies=(), native_geometry_materials=None,
                 initial_support_relations=(), holding_contacts=()):
        if not isinstance(background, FrozenESDF) or any(not isinstance(o.query, FrozenESDF) for o in owners):
            raise ValueError('Production scenes require ordinary frozen fields or primitives')
        if not isinstance(coverage, ObservedSpace) or coverage.reference_frame != reference_frame:
            raise ValueError('Observation diagnostics must retain their calibrated frame')
        self.background, self.owners, self.coverage = background, tuple(owners), coverage
        self.reference_frame, self.robot_geometry_source = reference_frame, robot_geometry_source
        self.robot_sphere_materials, self.contacts = tuple(robot_sphere_materials), tuple(contacts)
        if (not self.robot_sphere_materials or len({o.owner_id for o in self.owners}) != len(self.owners)
                or len({r.rule_id for r in self.contacts}) != len(self.contacts)):
            raise ValueError('Scene owners, rules and robot sphere identities must be explicit')
        self.gripper, self.has_gripper = gripper, has_gripper
        _validate_contact_and_gripper_inputs(
            {o.owner_id: (o.sphere_materials or (None,)*len(o.local_spheres)) if o.moving else None for o in self.owners},
            self.robot_sphere_materials, self.contacts, gripper, has_gripper)
        self.robot_collision_bodies = tuple(deepcopy(robot_collision_bodies))
        self.native_geometry_materials = deepcopy(native_geometry_materials or {})
        self.holding_contacts = tuple(holding_contacts)
        validate_holding_materials(self.holding_contacts, self.owners, self.robot_sphere_materials,
                                  self.native_geometry_materials, self.robot_collision_bodies)
        if {r.rule_id for r in self.contacts} & {r.rule_id for r in self.holding_contacts}:
            raise ValueError('Holding and attempted interaction rules need distinct identities')
        self.robot_sphere_geometry, self.robot_self_collision_pairs = [], []
        self.native_self_collision_pairs = []
        self._prepare_self_collision_pairs()
        self.robot_self_collision_padding = [0.]*len(self.robot_sphere_materials)
        self.robot_model = {}
        self.required_robot_links = ()
        self.grasp_contact_activations = ()
        self.observation_report = {}
        self._fixed_release_poses = {}
        self.required_progress_anchors = tuple(sorted(
            {v for r in self.contacts for v in r.interval} |
            {v for o in self.owners for a in o.attachments for v in a.interval}))
        self._bind_initial_supports(initial_support_relations)

    def _bind_initial_supports(self, relations, records=None):
        from .initial_support import InitialSupportRelation, PreparedInitialSupport
        relations = tuple(relations)
        if (any(not isinstance(r, InitialSupportRelation) for r in relations)
                or len({r.relation_id for r in relations}) != len(relations)
                or len({(r.actor, r.target_owner) for r in relations}) != len(relations)):
            raise ValueError('Initial supports need distinct typed relations and owner pairs')
        owners = {o.owner_id: o for o in self.owners}
        if any(r.actor not in owners or r.target_owner not in owners for r in relations):
            raise ValueError('Initial support must identify present scene owners')
        if records is not None and set(records) != {r.relation_id for r in relations}:
            raise ValueError('Initial support binding must retain the same relations')
        self.initial_support_relations = relations
        self._initial_support_models = {r.relation_id: PreparedInitialSupport(
            r, owners[r.actor], owners[r.target_owner],
            record=None if records is None else records[r.relation_id]) for r in relations}
        self.required_progress_anchors = tuple(sorted(set(self.required_progress_anchors) |
            {v for r in relations for v in r.interval}))

    def bind_online_robot(self, *, robot_sphere_geometry, robot_self_collision_pairs,
                          robot_self_collision_padding=None, robot_geometry_source=None, robot_model=None):
        """Bind calibrated sphere FK and native self pairs once before solving."""
        if robot_geometry_source is not None and robot_geometry_source != self.robot_geometry_source:
            raise ValueError('Online sphere calibration source disagrees with scene')
        indices, radii = [], np.zeros(len(self.robot_sphere_materials))
        for row in robot_sphere_geometry:
            ids = row['sphere_indices']
            centres, sizes = np.asarray(row['centers_local']), np.asarray(row['radii'])
            if centres.shape != (len(ids), 3) or sizes.shape != (len(ids),) or not np.isfinite(centres).all() or not np.isfinite(sizes).all():
                raise ValueError('Each robot link needs its exact finite native sphere geometry')
            indices.extend(ids)
            radii[ids] = sizes
        if sorted(indices) != list(range(len(radii))):
            raise ValueError('Every native sphere must occur once in online FK binding')
        pairs = np.asarray(robot_self_collision_pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.size and (pairs.min() < 0 or pairs.max() >= len(radii) or np.any(pairs[:, 0] == pairs[:, 1])):
            raise ValueError('Native self pairs must index different bound spheres')
        padding = np.zeros(len(radii)) if robot_self_collision_padding is None else np.asarray(robot_self_collision_padding).reshape(-1)
        if padding.shape != radii.shape or not np.isfinite(padding).all():
            raise ValueError('Native self padding must align with every bound sphere')
        self.robot_sphere_geometry = deepcopy(robot_sphere_geometry)
        self.robot_self_collision_pairs = pairs.tolist()
        # BEGIN every backend consumes the same precise native-pair replacement.
        self.native_self_collision_pairs = (self.gripper.self_collision_model.filter_pairs(pairs, padding)
            if self.gripper is not None and self.gripper.self_collision_model is not None else pairs.tolist())
        # END no entire finger, link, or owner collision class is disabled.
        # BEGIN compile fixed pair identities before any state query.
        self._prepare_self_collision_pairs()
        # END a rebind replaces this scene's arrays and device-cache dictionary.
        self.robot_self_collision_padding = padding.tolist()
        self.robot_model = deepcopy(robot_model or {})
        self.required_robot_links = tuple(row['link_name'] for row in robot_sphere_geometry)

    def _prepare_self_collision_pairs(self):
        """Preserve pair order while separating pairs without moving fingers.

        Whether a finger is currently uncertain remains a state-dependent
        Torch selection. Only its possible motion and pair membership belong
        to this fixed robot binding. Replacing the dictionary, rather than
        clearing it, leaves existing rollout clones on their original model.
        """
        self._uncertain_sphere_indices = (() if self.gripper is None else tuple(
            index for index, mapping in zip(self.gripper.sphere_indices, self.gripper.sphere_joint_translation)
            if np.any(mapping != 0)))
        pairs = np.asarray(self.native_self_collision_pairs, dtype=np.int64).reshape(-1, 2)
        moving = np.zeros(len(self.robot_sphere_materials), dtype=bool)
        moving[list(self._uncertain_sphere_indices)] = True
        known = pairs[~moving[pairs].any(axis=1)]
        self._self_collision_pair_arrays = (pairs, known)
        for value in self._self_collision_pair_arrays:
            value.flags.writeable = False
        self._self_collision_pair_device = {}

    def _self_collision_pair_tensors(self, spheres):
        """Upload each immutable int64 partition once per binding and device."""
        key = spheres.device
        if key not in self._self_collision_pair_device:
            # Own writable staging memory instead of exposing a Torch alias
            # to the read-only source arrays. Integer indices have one dtype
            # regardless of the query's floating-point dtype.
            self._self_collision_pair_device[key] = tuple(torch.as_tensor(
                np.array(value, copy=True), device=key, dtype=torch.long)
                for value in self._self_collision_pair_arrays)
        return self._self_collision_pair_device[key]

    def clone_for_rollout(self):
        result = copy(self)
        result._fixed_release_poses = dict(self._fixed_release_poses)
        result.background=self.background.clone_for_rollout()
        result.owners=tuple(copy(owner) for owner in self.owners)
        for owner in result.owners:
            owner.query=owner.query.clone_for_rollout()
            owner.attachments=deepcopy(owner.attachments)
        result.contacts=tuple(deepcopy(rule) for rule in self.contacts)
        result.gripper = None if self.gripper is None else copy(self.gripper)
        if result.gripper is not None:
            result.gripper.commands=deepcopy(self.gripper.commands)
        return result

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
        # BEGIN use the same ordered integer columns without host transfer in replay.
        indices = constant_tensor(indices, joint_position, dtype=torch.long)
        return ((joint_position[..., indices]-command)**2).sum(-1)

    def validate_problem(self, problem, *, goal_identities=None):
        """Validate concrete task bindings; candidate attachment remains conditional."""
        if problem.reference_frame != self.reference_frame:
            raise ValueError('Online scene and problem must share the calibrated frame')
        records = [asdict(rule) for rule in self.contacts]
        pose_ids = list(range(len(problem.pose_events))) if goal_identities is None else list(goal_identities['pose_events'])
        activations = []
        for row in problem.conditions:
            if row.get('kind') != 'grasp_contact_activation':
                continue
            grasp_contact_rule_ids(row, records)
            original = row['arrival_goal_id'][1]
            if original in pose_ids:
                validate_grasp_contact_arrival(row, problem.pose_events[pose_ids.index(original)])
            activations.append(deepcopy(row))
        self.grasp_contact_activations = tuple(activations)
        conditions={row['condition_id']:row for row in problem.conditions if 'condition_id' in row}
        if len(conditions)!=sum('condition_id' in row for row in problem.conditions):
            raise ValueError('Online condition identities must be unique')
        condition_ids=set(conditions)
        # BEGIN Holding binds its current internal materials, not a prior task goal.
        validate_holding_conditions(self.holding_contacts, self.owners, problem.conditions)
        # END the attachment's transform below is current geometry consistency only.
        for owner in self.owners:
            for relation in owner.attachments:
                if relation.condition_id not in condition_ids or (relation.release_condition_id is not None and relation.release_condition_id not in condition_ids):
                    raise ValueError('Every online attachment must identify this problem condition')
                declaration=conditions[relation.condition_id]
                if declaration.get('kind') not in {'candidate_attachment','observed_attachment'} or declaration.get('carrier_id')!=owner.owner_id or declaration.get('progress')!=relation.interval[0]:
                    raise ValueError('Online attachment must preserve the owner and exact condition boundary')
                if relation.release_condition_id is not None:
                    release=conditions[relation.release_condition_id]
                    if release.get('kind')!='predicted_release' or release.get('carrier_id')!=owner.owner_id or release.get('progress')!=relation.interval[1]:
                        raise ValueError('Online release must preserve its exact owner, event and boundary')
        if self.gripper is not None:
            commands = {(row['skill_id'], row['event_id'], row.get('occurrence',0), row['command'], row['progress']) for row in self.gripper.commands}
            events = {(event.skill_id, event.event_id, event.occurrence, event.kind, event.progress) for event in problem.events if event.kind in {'open','close','hold'}}
            if commands != events:
                raise ValueError('Online hand commands must preserve exact problem event occurrences')

    def components(self, robot_spheres, position, quaternion, progress,
                   joint_position=None, joint_names=None, *, trajectory_time=None,
                   active_contact_rule_ids=None, contact_permission_progress=None,
                   owner_poses=None, condition_evidence=None):
        """Evaluate known model constraints once, without task-effect proofs.

        Contact intervals describe a permitted attempt. They never compare
        measured TCP with a frozen target or require a successful grasp. The
        optional actual-state arguments retain exact observation identities.
        """
        if (robot_spheres.shape[:2] != progress.shape
                or robot_spheres.shape[-2:] != (len(self.robot_sphere_materials), 4)):
            raise ValueError('Scene query requires aligned [batch,time,sphere,4] robot geometry')
        if active_contact_rule_ids is not None and not set(active_contact_rule_ids) <= {r.rule_id for r in self.contacts}:
            raise ValueError('Interaction rule IDs must belong to the current scene')
        permission_progress = progress if contact_permission_progress is None else contact_permission_progress
        if permission_progress.shape != progress.shape:
            raise ValueError('Interaction progress must align with queried robot states')
        terms, spheres, state = {}, robot_spheres, None
        uncertain_hand = torch.zeros_like(progress, dtype=torch.bool)
        if self.gripper is not None:
            state = self.gripper.state(progress, joint_position, joint_names,
                trajectory_time=trajectory_time, measured=active_contact_rule_ids is not None)
            spheres = self.gripper.spheres(spheres, position, quaternion, state.coordinates, state.half)
            terms['gripper/controller'] = state.controller_residual
            uncertain_hand = state.uncertain
        moving = torch.zeros(spheres.shape[-2], dtype=torch.bool, device=spheres.device)
        moving[list(self._uncertain_sphere_indices)] = True
        uncertain_spheres = uncertain_hand[..., None] & moving

        def add_collision(values, name, actor):
            # An unresolved future finger is not a measured collision. Its
            # uncertainty is reported by the gripper model; actual execution
            # uses joint feedback and the existing robot geometry checker.
            if actor == 'robot':
                values = torch.where(uncertain_spheres, 0., values)
            terms[name] = values.sum(-1)

        if self.native_self_collision_pairs:
            padded = spheres.clone()
            padded[..., 3] = padded[..., 3]+_tensor(self.robot_self_collision_padding, spheres)
            pairs, known_pairs = self._self_collision_pair_tensors(spheres)
            terms['collision/robot/self'] = torch.where(uncertain_hand,
                _self_pair_sum(padded, known_pairs), _self_pair_sum(padded, pairs))
        if state is not None and self.gripper.self_collision_model is not None:
            known, _ = self.gripper.self_collision_model.components(state)
            terms['collision/robot/self_prismatic'] = known
        poses = {}
        for owner in self.owners:
            if owner_poses is not None and owner.owner_id in owner_poses:
                matrix = np.asarray(owner_poses[owner.owner_id], dtype=float)
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise ValueError('Observed owner transform must be finite 4x4')
                poses[owner.owner_id] = (
                    _tensor(matrix[:3, 3], progress).expand(*progress.shape, 3),
                    _tensor(Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]], progress).expand(*progress.shape, 4))
                continue
            releases = []
            for index, relation in enumerate(owner.attachments):
                key = owner.owner_id, index
                if relation.release_condition_id is None:
                    releases.append(None)
                elif key in self._fixed_release_poses:
                    releases.append(self._fixed_release_poses[key])
                else:
                    pose, found = self._release_pose(relation, position, quaternion, progress)
                    releases.append(pose)
                    # This checks that a requested future model transform can
                    # be formed; it does not assert that the object released.
                    terms[f'unsupported/release_anchor/{owner.owner_id}/{index}'] = (
                        (progress >= relation.interval[1]) & ~found[:, None]).to(progress.dtype)
            poses[owner.owner_id] = owner.pose(position, quaternion, progress, releases)
        actors = {'robot': spheres}
        for owner in self.owners:
            if owner.moving:
                cover = owner.local_spheres.copy()
                cover[:, 3] += owner.sphere_cover_error_m
                actors[owner.owner_id] = _world_spheres(cover, *poses[owner.owner_id])
        for actor, body in actors.items():
            add_collision(self.background(body), f'collision/{actor}/background', actor)
            for owner in self.owners:
                if actor == owner.owner_id:
                    continue
                local = _local_spheres(body, *poses[owner.owner_id])
                values = owner.query(local)
                for rule in self.contacts:
                    if rule.actor != actor or rule.target_owner != owner.owner_id:
                        continue
                    clock = permission_progress if actor == 'robot' else progress
                    active = (clock >= rule.interval[0]) & (clock <= rule.interval[1])
                    if active_contact_rule_ids is not None:
                        active = active & (rule.rule_id in active_contact_rule_ids)
                    selected = torch.zeros(values.shape[-1], dtype=torch.bool, device=values.device)
                    selected[list(rule.sphere_indices)] = True
                    values = torch.where(active[..., None] & selected, 0., values)
                if actor == 'robot':
                    for contact in self.holding_contacts:
                        if contact.target_owner != owner.owner_id:
                            continue
                        relation, = [r for r in owner.attachments if r.condition_id == contact.condition_id]
                        active = (permission_progress >= contact.interval[0]) & (permission_progress <= contact.interval[1])
                        if relation.release_condition_id is not None:
                            active = active & (permission_progress < contact.interval[1])
                        if condition_evidence is not None:
                            active = active & (condition_evidence.get(contact.condition_id) is True)
                        selected = torch.zeros(values.shape[-1], dtype=torch.bool, device=values.device)
                        selected[list(contact.sphere_indices)] = True
                        values = torch.where(active[..., None] & selected, 0., values)
                for support in self._initial_support_models.values():
                    if support.relation.actor == actor and support.relation.target_owner == owner.owner_id:
                        values = support.collision_values(values, progress)
                if actor == 'robot' and owner.robot_excluded_spheres:
                    values = values.clone()
                    values[..., list(owner.robot_excluded_spheres)] = 0.
                add_collision(values, f'collision/{actor}/{owner.owner_id}', actor)
        return terms

    def evaluate(self, *args, **kwargs):
        components_out = kwargs.pop('components_out', None)
        terms = self.components(*args, **kwargs)
        if components_out is not None:
            components_out.update(terms)
        return torch.stack(tuple(terms.values())).sum(0)

    def estimated_workspace_allocation(self, batch, horizon):
        """Bound only the model tensors actually used by ordinary field queries.

        Source component trees, contact banks, native Jacobians and coverage
        proofs no longer exist in this request and must not reserve memory.
        Both host/device dtype caches are bounded; pair expansion is chunked.
        """
        fields = {id(q.distance): q for q in (self.background, *(o.query for o in self.owners))}
        cells = sum(q.distance.size for q in fields.values())
        source = sum(q.distance.nbytes for q in fields.values())
        source += self.coverage._known.nbytes
        if self.coverage._known_free is not None:
            source += self.coverage._known_free.nbytes
        source += sum(o.local_spheres.nbytes for o in self.owners if o.local_spheres is not None)
        pairs = max((len(v) for v in self._self_collision_pair_arrays), default=0)
        if not self.robot_sphere_geometry:
            n = len(self.robot_sphere_materials)
            pairs = n*(n-1)//2
        pair_bytes = pairs*2*8*2
        spheres = len(self.robot_sphere_materials)+sum(len(o.local_spheres) for o in self.owners if o.moving)
        states = int(batch)*int(horizon)
        live = states*spheres*512 + states*_self_pair_chunk_size(states, pairs)*24*8
        if self.gripper is not None and self.gripper.self_collision_model is not None:
            source += self.gripper.self_collision_model.storage_nbytes*4
            live += states*len(self.gripper.self_collision_model.axis_pairs)*48
        return {'cpu_shared': int(source+cells*12+pair_bytes*2),
                'gpu_shared': int(cells*12+pair_bytes),
                'cpu_live': int(live+max((q.distance.nbytes for q in fields.values()), default=0)),
                'gpu_live': int(live), 'gpu_retained': 0, 'gpu_graph_live': int(live)}

    def query_state(self, spheres, tool_pose, progress, *, joint_position=None, joint_names=None,
                    trajectory_time=0., active_contact_rule_ids=None, owner_poses=None, condition_evidence=None):
        """Return model acceptance and a separate honest observation-gap report."""
        tool = np.asarray(tool_pose, dtype=float)
        if tool.shape != (4, 4) or not np.isfinite(tool).all():
            raise ValueError('State query needs a finite tool transform')
        p = torch.as_tensor(tool[:3, 3], dtype=torch.float64)[None, None]
        q = torch.as_tensor(Rotation.from_matrix(tool[:3, :3]).as_quat()[[3, 0, 1, 2]], dtype=torch.float64)[None, None]
        step = torch.tensor([[float(progress)]], dtype=torch.float64)
        body = torch.as_tensor(spheres, dtype=torch.float64)[None, None]
        if not torch.isfinite(body).all() or not torch.isfinite(step).all():
            raise ValueError('State geometry and progress must be finite')
        joints = None if joint_position is None else torch.as_tensor(joint_position, dtype=torch.float64)[None, None]
        terms = self.components(body, p, q, step, joint_position=joints, joint_names=joint_names,
            trajectory_time=torch.tensor([[float(trajectory_time)]], dtype=torch.float64),
            active_contact_rule_ids=active_contact_rule_ids, owner_poses=owner_poses,
            condition_evidence=condition_evidence)
        values = {name: float(value.item()) for name, value in terms.items()}
        violations = {name: value for name, value in values.items() if not np.isfinite(value) or value > 0}
        observation = self.coverage.diagnostics(np.asarray(spheres)[..., :3])
        return {'passed': not violations, 'components': values,
                'violations': [{'kind': name.split('/')[0], 'name': name, 'residual': value}
                               for name, value in violations.items()],
                'status': 'COLLISION_OR_CONSTRAINT' if violations else 'PASS',
                'observation': observation,
                'scope': 'Known approximate model only; observation gaps are unresolved and task success is unverified'}

    def describe(self):
        return {'model': 'frozen_motion_scene_v4', 'reference_frame': self.reference_frame,
                'robot_geometry_source': self.robot_geometry_source,
                'owners': [o.owner_id for o in self.owners],
                'required_progress_anchors': self.required_progress_anchors,
                'observation_report': deepcopy(self.observation_report),
                'unknown_space': 'diagnostic_only',
                'scope': 'Ordinary model collision checks and explicit interaction pairs; no task-success or coverage certificate'}

    def to_execution_binding(self):
        owners = []
        for owner in self.owners:
            row = {name:_plain(getattr(owner,name)) for name in ('owner_id','source','geometry_complete','moving','local_spheres','sphere_materials','sphere_cover_error_m','pose_interval','robot_excluded_spheres','robot_exclusion_source','robot_exclusion_geometry_source','native_geometry_names')}
            path = owner.pose_path
            if path.motion_model is not None:
                raise ValueError('UNSUPPORTED_MODEL: online moving owners require one explicit rigid pose path')
            row['pose_path'] = {'progress':_plain(path.progress),'poses':_plain(path.poses),'source':path.source}
            row['attachments'] = _plain([asdict(a) for a in owner.attachments])
            row['query'] = owner.query.record()
            owners.append(row)
        hand = self.gripper
        hand_record = None
        if hand is not None:
            hand_record = {key:_plain(getattr(hand,key)) for key in ('controller','joint_names','current_positions','joint_limits','coupling_matrix','coupling_offset','sphere_indices','local_spheres','sphere_joint_translation','self_collision_pairs','commands','source','control_timestep_s','numerical_joint_tolerance')}
            hand_record['self_collision_geometry'] = ([] if hand.self_collision_model is None
                                                     else deepcopy(hand.self_collision_model.geometry))
            hand_record['prepared_self_collision'] = (None if hand.self_collision_model is None
                                                     else hand.self_collision_model.record())
            # BEGIN production bindings retain declared geometry and its expiry.
            hand_record['nominal_model'] = (None if hand.nominal_finger_trajectory is None
                                            else hand.nominal_finger_trajectory.describe())
            # END command targets remain separate from this optional geometry.
        return {'schema':'online_scene_v4','reference_frame':self.reference_frame,
            'robot_geometry_source':self.robot_geometry_source,'robot_model':deepcopy(self.robot_model),
            'robot_sphere_materials':list(self.robot_sphere_materials),
            'robot_sphere_geometry':deepcopy(self.robot_sphere_geometry),
            'robot_self_collision_pairs':deepcopy(self.robot_self_collision_pairs),
            'robot_self_collision_padding':list(self.robot_self_collision_padding),
            'background':self.background.record(),'owners':owners,'contacts':_plain([asdict(r) for r in self.contacts]),
            # BEGIN serialize Holding geometry identity separately from task contact laws.
            'holding_contacts':_plain([asdict(row) for row in self.holding_contacts]),
            # END restored requests cannot recover this relation from old grasp patches.
            'initial_support_relations':[row.record() for row in self.initial_support_relations],
            'initial_support_preparation':{name:model.record() for name,model in self._initial_support_models.items()},
            'gripper':hand_record,'has_gripper':self.has_gripper,
            'robot_collision_bodies':_plain(self.robot_collision_bodies),'native_geometry_materials':deepcopy(self.native_geometry_materials),
            'grasp_contact_activations':deepcopy(self.grasp_contact_activations),
            'coverage':{'known':encode_array(self.coverage._known),'known_free':encode_array(self.coverage.known_free),'origin':self.coverage.origin.tolist(),
                'voxel_size':self.coverage.voxel_size,'observation_ids':list(self.coverage.observation_ids)},
            'observation_report':deepcopy(self.observation_report)}

    binding_record = to_execution_binding

    @classmethod
    def from_binding(cls,row):
        if row.get('schema')!='online_scene_v4':
            raise ValueError('Online scene binding schema is required')
        owners=[]
        for record in row['owners']:
            record=deepcopy(record)
            record['query']=FrozenESDF.from_record(record['query'])
            record['pose_path']=PosePath(**record['pose_path'])
            record['attachments']=tuple(Attachment(**a) for a in record['attachments'])
            owners.append(CollisionOwner(**record))
        hand=None
        if row['gripper'] is not None:
            spec=deepcopy(row['gripper'])
            prepared = spec.pop('prepared_self_collision')
            # BEGIN restore only the explicitly archived physical validity.
            from .gripper_trajectory import NominalFingerGeometry
            nominal_record = spec.pop('nominal_model')
            # JSON represents an empty [0,J] coupling matrix as []; restore
            # its declared native joint dimension before ordinary validation.
            spec['coupling_matrix'] = np.asarray(spec['coupling_matrix'], dtype=float).reshape(-1, len(spec['joint_names']))
            # Runtime reads measured finger joints. No archived command curve
            # is promoted to future physical contact or guessed feedback.
            spec['controller']='binary'
            spec['joint_bounds']=None
            hand=PrismaticGripper(**spec)
            if nominal_record is not None:
                nominal = NominalFingerGeometry.from_mapping(nominal_record)
                if (nominal.joint_names != hand.joint_names
                        or not np.array_equal(nominal.position[0], hand.current_positions)):
                    raise ValueError('Saved nominal geometry changed the measured named entrance')
                hand.nominal_finger_trajectory = nominal
            # END actual execution still supplies measured complete finger joints.
            if (prepared is None) != (hand.self_collision_model is None):
                raise ValueError('Saved native hand geometry and prepared model must agree')
            if prepared is not None:
                hand.self_collision_model.validate_record(prepared)
        cover=row['coverage']
        known=decode_array(cover['known'])
        coverage=ObservedSpace(known,origin=cover['origin'],voxel_size=cover['voxel_size'],
            observation_ids=cover['observation_ids'],reference_frame=row['reference_frame'],
            known_free=decode_array(cover['known_free']))
        scene=cls(background=FrozenESDF.from_record(row['background']),owners=owners,coverage=coverage,
            reference_frame=row['reference_frame'],robot_sphere_materials=row['robot_sphere_materials'],
            robot_geometry_source=row['robot_geometry_source'],contacts=[ContactRule(**r) for r in row['contacts']],
            holding_contacts=[HoldingContact(**value) for value in row['holding_contacts']],
            gripper=hand,has_gripper=row['has_gripper'],
            robot_collision_bodies=row['robot_collision_bodies'],native_geometry_materials=row['native_geometry_materials'])
        # BEGIN restore the optional support capability with its frozen binding.
        from .initial_support import InitialSupportRelation
        initial = tuple(InitialSupportRelation(**value) for value in row['initial_support_relations'])
        scene._bind_initial_supports(initial, row['initial_support_preparation'])
        # END ordinary scenes legitimately carry an empty support inventory.
        scene.bind_online_robot(robot_sphere_geometry=row['robot_sphere_geometry'],
            robot_self_collision_pairs=row['robot_self_collision_pairs'],robot_self_collision_padding=row['robot_self_collision_padding'],
            robot_model=row['robot_model'])
        scene.grasp_contact_activations=tuple(deepcopy(row['grasp_contact_activations']))
        scene.observation_report=deepcopy(row['observation_report'])
        return scene

# END bounded motion scene replaces complete contact/coverage certification.
