"""Frozen entity and observed-field collision models for motion attempts.

Registered complete solids use one native distance model. A Mapper ESDF is
an explicitly approximate observed field and retains sphere queries. Local
interaction masks allow contact without proving task success. Observation
gaps remain diagnostics; no source-component proof chain is constructed.
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
    PrismaticGripper, _tensor, _local_spheres, _world_spheres,
    grasp_contact_rule_ids, validate_grasp_contact_arrival,
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
        # BEGIN complete native solids own self/world collision decisions.
        self.robot_sphere_geometry = []
        self.native_collision_model = None
        self.native_robot_model, self.native_joint_names, self.native_self_pairs = {}, (), []
        self._uncertain_sphere_indices = (() if gripper is None else tuple(
            index for index, mapping in zip(gripper.sphere_indices, gripper.sphere_joint_translation)
            if np.any(mapping != 0)))
        # END proxy spheres remain only for observed fields and component identity.
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
        # BEGIN an entrance sphere subset is not a complete native permission.
        for relation in relations:
            if isinstance(owners[relation.target_owner].query, FrozenPrimitives):
                raise ValueError(f'Initial support {relation.relation_id} requires an observed ESDF target; '
                                 'its entrance sphere subset does not define a complete native component permission')
        # END the existing observed-plane relation keeps its original sphere scope.
        if records is not None and set(records) != {r.relation_id for r in relations}:
            raise ValueError('Initial support binding must retain the same relations')
        self.initial_support_relations = relations
        self._initial_support_models = {r.relation_id: PreparedInitialSupport(
            r, owners[r.actor], owners[r.target_owner],
            record=None if records is None else records[r.relation_id]) for r in relations}
        self.required_progress_anchors = tuple(sorted(set(self.required_progress_anchors) |
            {v for r in relations for v in r.interval}))

    # BEGIN one native model binds source solids, policies and complete permissions.
    def bind_online_robot(self, *, robot_sphere_geometry, native_robot_model,
                          native_joint_names, native_self_pairs, robot_model,
                          robot_geometry_source=None):
        """Compile complete native geometry once; keep proxies for ESDF queries.

        Native self-pair policies come from the independent robot validator.
        No sphere collision policy survives as a second rejection criterion.
        Full source geometry and its URDF are saved with the scene, so a
        restored request constructs exactly this same entity-distance model.
        """
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
        self.robot_sphere_geometry = deepcopy(robot_sphere_geometry)
        self.robot_model = deepcopy(robot_model)
        self.native_robot_model = deepcopy(native_robot_model)
        self.native_joint_names = tuple(native_joint_names)
        self.native_self_pairs = deepcopy(native_self_pairs)
        self.required_robot_links = tuple(row['link_name'] for row in robot_sphere_geometry)
        from .native_collision import NativeCollisionModel
        native_owners = {owner.owner_id: owner.query.primitives for owner in self.owners
                         if isinstance(owner.query, FrozenPrimitives)}
        if isinstance(self.background, FrozenPrimitives):
            native_owners = {'background': self.background.primitives, **native_owners}
        aliases = {owner.owner_id: tuple(owner.native_geometry_names) for owner in self.owners
                   if owner.native_geometry_names}
        self.native_collision_model = NativeCollisionModel(
            robot_model=self.native_robot_model, joint_names=self.native_joint_names,
            T_reference_base=self.robot_model['T_reference_base'], self_pairs=self.native_self_pairs,
            owners=native_owners,
            moving_owners=tuple(owner.owner_id for owner in self.owners
                                if owner.moving and owner.owner_id in native_owners),
            owner_aliases=aliases)
        self._uncertain_sphere_indices = (() if self.gripper is None else tuple(
            index for index, mapping in zip(self.gripper.sphere_indices, self.gripper.sphere_joint_translation)
            if np.any(mapping != 0)))
        self._bind_native_permissions(native_owners)

    def _bind_native_permissions(self, native_owners):
        """Map sphere declarations to complete source components exactly once.

        A rule selecting only part of a component cannot exempt its complete
        solid. The error names that rule and component instead of silently
        widening the permission or reverting to the conservative proxy.
        """
        geometry = {row['name'] for row in self.native_robot_model['geometries']}
        # BEGIN a component map is consumed only by declared local permissions.
        # Complete source solids and self policies already define collision
        # queries without a sphere-to-material table. Validate the groups
        # actually supplied; complete_components below checks each selected
        # rule/exclusion instead of demanding an unused global partition.
        sphere_ids = set(range(len(self.robot_sphere_materials)))
        robot_components, seen_spheres = {}, set()
        for body in self.robot_collision_bodies:
            indices, names = frozenset(body['sphere_indices']), tuple(body['geometry_names'])
            if (not indices <= sphere_ids or len(indices) != len(body['sphere_indices'])
                    or seen_spheres & indices or len(set(names)) != len(names)
                    or any(name not in geometry or name in robot_components for name in names)):
                raise ValueError('Native robot component/sphere identity must be a unique registered partition')
            seen_spheres.update(indices)
            robot_components.update((name, indices) for name in names)
        # END absent mapping does not remove any native collision pair.
        components = {'robot': robot_components}
        for owner in self.owners:
            if owner.owner_id in native_owners and owner.moving:
                components[owner.owner_id] = {
                    row['native_binding']['geometry_name']: frozenset(row['native_binding']['sphere_indices'])
                    for row in owner.query.primitives if 'native_binding' in row}

        def complete_components(actor, indices, identity):
            selected, complete, covered = set(indices), set(), set()
            for name, members in components[actor].items():
                if selected & members:
                    if not members <= selected:
                        raise ValueError(f'Native permission {identity} selects only part of component {actor}/{name}')
                    complete.add(name)
                    covered.update(members)
            if covered != selected:
                raise ValueError(f'Native permission {identity} has spheres without a complete registered {actor} component')
            return complete

        pairs = self.native_collision_model.pairs
        def pair_indices(actor, target, names):
            return tuple(index for index, pair in enumerate(pairs)
                         if pair['first_owner'] == actor and pair['second_owner'] == target
                         and pair['first_geometry'] in names)

        self._native_rule_pairs = {}
        for rule in self.contacts:
            if rule.target_owner in native_owners and rule.actor in components:
                names = complete_components(rule.actor, rule.sphere_indices, rule.rule_id)
                self._native_rule_pairs[rule.rule_id] = pair_indices(rule.actor, rule.target_owner, names)
        self._native_holding_pairs = {}
        for contact in self.holding_contacts:
            if contact.target_owner in native_owners:
                names = complete_components('robot', contact.sphere_indices, contact.rule_id)
                self._native_holding_pairs[contact.rule_id] = pair_indices('robot', contact.target_owner, names)
        excluded = []
        for owner in self.owners:
            # BEGIN native aliases already transfer their robot pairs to self policy.
            # An old world exclusion with no remaining world pair has nothing
            # to mask. A compound owner with extra components still has pairs
            # and must supply the same complete component mapping as any rule.
            has_robot_pair = any(pair['first_owner'] == 'robot' and pair['second_owner'] == owner.owner_id
                                 for pair in pairs)
            if owner.owner_id in native_owners and owner.robot_excluded_spheres and has_robot_pair:
                names = complete_components('robot', owner.robot_excluded_spheres,
                                            f'robot exclusion for {owner.owner_id}')
                excluded.extend(pair_indices('robot', owner.owner_id, names))
            # END pair existence, not an owner name, controls this unused conversion.
        self._native_excluded_pairs = tuple(excluded)
        # Native motion dependencies belong to the same URDF as its FK;
        # TCP-relative sphere translations only own the separate ESDF mask.
        uncertain = set(self.native_collision_model.geometry_depending_on(
            () if self.gripper is None else self.gripper.joint_names))
        self._native_uncertain_pairs = tuple(index for index, pair in enumerate(pairs)
            if (pair['first_owner'] == 'robot' and pair['first_geometry'] in uncertain)
            or (pair['second_owner'] == 'robot' and pair['second_geometry'] in uncertain))
    # END permission and uncertainty identities never depend on a queried collision.

    # BEGIN native distances consume the same named joint and owner poses as ESDF.
    def _native_components(self, joint_position, joint_names, state, poses, progress,
                           uncertain_hand, active_rules, active_holding):
        model = self.native_collision_model
        if model is None:
            raise ValueError('Scene collision queries require a bound complete native robot model')
        named = {}
        if joint_position is not None:
            names = tuple(joint_names or ())
            if (len(set(names)) != len(names)
                    or joint_position.shape != (*progress.shape, len(names))):
                raise ValueError('Native collision requires aligned complete named robot coordinates')
            named.update((name, joint_position[..., index]) for index, name in enumerate(names))
        if state is not None:
            named.update((name, state.coordinates[..., index])
                         for index, name in enumerate(self.gripper.joint_names))
        if set(named) != set(model.joint_names):
            raise ValueError('Native collision cannot fill absent robot coordinates with defaults')
        coordinates = (torch.stack([named[name] for name in model.joint_names], -1)
                       if model.joint_names else progress.new_empty((*progress.shape, 0)))
        transforms = []
        for identity in model.owner_ids:
            if identity == 'background':
                transforms.append(torch.eye(4, device=progress.device, dtype=progress.dtype)
                                  .expand(*progress.shape, 4, 4))
                continue
            position, quaternion = poses[identity]
            # Rotating the basis vectors gives columns of world-from-owner;
            # concatenate without detach so attachment/release motion keeps
            # the same Torch derivative as the complete robot coordinates.
            basis = torch.eye(3, device=progress.device, dtype=progress.dtype)
            rotation = quaternion_rotate(quaternion[..., None, :], basis).transpose(-1, -2)
            top = torch.cat((rotation, position[..., None]), -1)
            bottom = progress.new_tensor([0., 0., 0., 1.]).expand(*progress.shape, 1, 4)
            transforms.append(torch.cat((top, bottom), -2))
        owner_matrices = (torch.stack(transforms, -3) if transforms
                          else progress.new_empty((*progress.shape, 0, 4, 4)))
        enabled = torch.ones((*progress.shape, len(model.pairs)), dtype=torch.bool, device=progress.device)
        enabled[..., list(self._native_excluded_pairs)] = False
        enabled[..., list(self._native_uncertain_pairs)] &= ~uncertain_hand[..., None]
        for identity, indices in self._native_rule_pairs.items():
            enabled[..., list(indices)] &= ~active_rules[identity][..., None]
        for identity, indices in self._native_holding_pairs.items():
            enabled[..., list(indices)] &= ~active_holding[identity][..., None]
        values = model(coordinates, owner_matrices, enabled)
        return {name: values[..., index] for index, name in enumerate(model.groups)}
    # END no native proxy residual or separate hand checker participates.

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
        # BEGIN complete solids and observed fields consume identical permission clocks.
        active_rules = {}
        for rule in self.contacts:
            clock = permission_progress if rule.actor == 'robot' else progress
            active = (clock >= rule.interval[0]) & (clock <= rule.interval[1])
            if active_contact_rule_ids is not None:
                active = active & (rule.rule_id in active_contact_rule_ids)
            active_rules[rule.rule_id] = active
        active_holding = {}
        owners = {owner.owner_id: owner for owner in self.owners}
        for contact in self.holding_contacts:
            relation, = [relation for relation in owners[contact.target_owner].attachments
                         if relation.condition_id == contact.condition_id]
            active = (permission_progress >= contact.interval[0]) & (permission_progress <= contact.interval[1])
            if relation.release_condition_id is not None:
                active &= permission_progress < contact.interval[1]
            if condition_evidence is not None:
                active &= condition_evidence.get(contact.condition_id) is True
            active_holding[contact.rule_id] = active
        terms.update(self._native_components(joint_position, joint_names, state, poses, progress,
                                            uncertain_hand, active_rules, active_holding))
        native_actors = {'robot', *self.native_collision_model.owner_ids}
        # END an exact pair has one residual; sphere overlap cannot add a second veto.
        actors = {'robot': spheres}
        for owner in self.owners:
            if owner.moving:
                cover = owner.local_spheres.copy()
                cover[:, 3] += owner.sphere_cover_error_m
                actors[owner.owner_id] = _world_spheres(cover, *poses[owner.owner_id])
        for actor, body in actors.items():
            if not (actor in native_actors and isinstance(self.background, FrozenPrimitives)):
                add_collision(self.background(body), f'collision/{actor}/background', actor)
            for owner in self.owners:
                if actor == owner.owner_id:
                    continue
                if actor in native_actors and isinstance(owner.query, FrozenPrimitives):
                    continue
                local = _local_spheres(body, *poses[owner.owner_id])
                values = owner.query(local)
                for rule in self.contacts:
                    if rule.actor != actor or rule.target_owner != owner.owner_id:
                        continue
                    selected = torch.zeros(values.shape[-1], dtype=torch.bool, device=values.device)
                    selected[list(rule.sphere_indices)] = True
                    values = torch.where(active_rules[rule.rule_id][..., None] & selected, 0., values)
                if actor == 'robot':
                    for contact in self.holding_contacts:
                        if contact.target_owner != owner.owner_id:
                            continue
                        selected = torch.zeros(values.shape[-1], dtype=torch.bool, device=values.device)
                        selected[list(contact.sphere_indices)] = True
                        values = torch.where(active_holding[contact.rule_id][..., None] & selected, 0., values)
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
        """Account for observed-field tensors and the native group Jacobians.

        Native derivatives are stored once per output group, not as a
        pair-by-owner tensor. The double-precision bound covers both CPU
        query arrays and their Torch backward copies for either input dtype.
        Source component trees and duplicate sphere self-pairs do not exist.
        """
        fields = {id(q.distance): q for q in (self.background, *(o.query for o in self.owners))}
        cells = sum(q.distance.size for q in fields.values())
        source = sum(q.distance.nbytes for q in fields.values())
        source += self.coverage._known.nbytes
        if self.coverage._known_free is not None:
            source += self.coverage._known_free.nbytes
        source += sum(o.local_spheres.nbytes for o in self.owners if o.local_spheres is not None)
        spheres = len(self.robot_sphere_materials)+sum(len(o.local_spheres) for o in self.owners if o.moving)
        states = int(batch)*int(horizon)
        live = states*spheres*512
        native_cpu, native_gpu = 0, 0
        if self.native_collision_model is not None:
            model = self.native_collision_model
            columns = len(model.joint_names)+16*len(model.owner_ids)
            native_cpu = states*(len(model.groups)*(1+columns)+columns)*8
            native_gpu = states*len(model.groups)*columns*8
            # Complete source vertices and fixed pair tables are immutable;
            # Coal/Pinocchio storage is owned by the same bound CPU model.
            source += sum(np.asarray(row.get('vertices_frame', row.get('dimensions', ()))).nbytes
                          for row in self.native_robot_model['geometries'])
            source += len(model.pairs)*4*8
        return {'cpu_shared': int(source+cells*12), 'gpu_shared': int(cells*12),
                'cpu_live': int(live+native_cpu+max((q.distance.nbytes for q in fields.values()), default=0)),
                'gpu_live': int(live+native_gpu), 'gpu_retained': 0,
                'gpu_graph_live': int(live+native_gpu)}

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
                'scope': 'Registered solid distances and approximate observed ESDF; observation gaps and task success remain unverified'}

    def describe(self):
        return {'model': 'frozen_motion_scene_v5', 'reference_frame': self.reference_frame,
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
            hand_record = {key:_plain(getattr(hand,key)) for key in ('controller','joint_names','current_positions','joint_limits','coupling_matrix','coupling_offset','sphere_indices','local_spheres','sphere_joint_translation','commands','source','control_timestep_s','numerical_joint_tolerance')}
            # BEGIN production bindings retain declared geometry and its expiry.
            hand_record['nominal_model'] = (None if hand.nominal_finger_trajectory is None
                                            else hand.nominal_finger_trajectory.describe())
            # END command targets remain separate from this optional geometry.
        return {'schema':'online_scene_v5','reference_frame':self.reference_frame,
            'robot_geometry_source':self.robot_geometry_source,'robot_model':deepcopy(self.robot_model),
            'robot_sphere_materials':list(self.robot_sphere_materials),
            'robot_sphere_geometry':deepcopy(self.robot_sphere_geometry),
            # BEGIN current native source/policy records restore the sole solid model.
            'native_robot_model':deepcopy(self.native_robot_model),
            'native_joint_names':list(self.native_joint_names),
            'native_self_pairs':deepcopy(self.native_self_pairs),
            # END no archived sphere rejection criterion survives restoration.
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
        if row.get('schema')!='online_scene_v5':
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
            native_robot_model=row['native_robot_model'], native_joint_names=row['native_joint_names'],
            native_self_pairs=row['native_self_pairs'],
            robot_model=row['robot_model'])
        scene.grasp_contact_activations=tuple(deepcopy(row['grasp_contact_activations']))
        scene.observation_report=deepcopy(row['observation_report'])
        return scene

# END bounded motion scene replaces complete contact/coverage certification.
