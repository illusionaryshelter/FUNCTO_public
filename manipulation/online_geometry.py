"""One frozen, finite geometry model for planning and measured-state queries.

The Mapper's signed distances use positive free space. Unknown samples never
become free evidence. Contact subtraction operates on bounded field cells and
requires the complete cell to fit a finite material permission; no owner/link
pair can be disabled. CPU execution reads the exact arrays used by Torch during
optimization. Source-mesh reconstruction belongs to the separate audit API.
"""
# BEGIN unified frozen online geometry; no source-solid or layer fallback.
from copy import copy, deepcopy
from dataclasses import asdict
from itertools import product
import math

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from scipy.spatial.transform import Rotation

from .collision_records import encode_array, decode_array, MAX_ARRAY_BYTES
from .planning_problem import PosePath, quaternion_conjugate, quaternion_rotate, event_occurrence_key
from .planning_scene import (Attachment, CollisionOwner, ContactRule, ObservedSpace,
    PlanningScene, PrismaticGripper, _tensor, _local_spheres, _world_spheres,
    _self_pair_sum, grasp_contact_rule_ids, validate_grasp_contact_arrival)


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
    """Immutable XYZ signed distance and source-validity arrays in one frame.

    ``observed_sign`` identifies actual TSDF sign support; ``site_valid`` only
    identifies a nearest observed surface. A distant positive sample is not a
    free-space observation. Field bounds are fixed for the entire request.
    """
    def __init__(self, distance, *, origin, voxel_size, site_valid, observed_sign,
                 owner_ids, source, contact_max_cells=32768):
        distance = np.asarray(distance, dtype=np.float32)
        valid, sign = np.asarray(site_valid), np.asarray(observed_sign)
        if (distance.ndim != 3 or min(distance.shape) < 2 or distance.nbytes > MAX_ARRAY_BYTES
                or not np.isfinite(distance).all() or valid.shape != distance.shape
                or sign.shape != distance.shape or valid.dtype != bool or sign.dtype != bool):
            raise ValueError('UNSUPPORTED_MODEL: bounded complete XYZ field and source masks required')
        self.origin = np.asarray(origin, dtype=float).copy()
        self.voxel_size = float(voxel_size)
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all() or not self.voxel_size > 0:
            raise ValueError('Finite field origin and positive voxel size required')
        self.distance, self.site_valid, self.observed_sign = distance.copy(), valid.copy(), sign.copy()
        for value in (self.distance, self.site_valid, self.observed_sign):
            value.flags.writeable = False
        self.owner_ids, self.source = frozenset(owner_ids), str(source)
        self.contact_max_cells = int(contact_max_cells)
        self._device = {}
        self._contact_regions = {}
        self._contact_surfaces = {}
        self._contact_unions = {}
        self.upper = self.origin+(np.asarray(distance.shape)-1)*self.voxel_size
        occupied=self.site_valid & ((np.abs(self.distance)<=self.voxel_size*.51) | ((self.distance<0)&self.observed_sign))
        if occupied.any():
            extents=[np.flatnonzero(occupied.any(axis=tuple(d for d in range(3) if d!=axis))) for axis in range(3)]
            self.geometry_lower=self.origin+np.array([value[0] for value in extents])*self.voxel_size-self.voxel_size/2
            self.geometry_upper=self.origin+np.array([value[-1] for value in extents])*self.voxel_size+self.voxel_size/2
        else:
            self.geometry_lower=self.geometry_upper=None

    def clone_for_rollout(self):
        result=copy(self)
        # Dense arrays are immutable; request-specific finite inventories own
        # their dictionaries so a reused rule ID cannot rewrite a sibling.
        result._contact_regions=dict(self._contact_regions)
        result._contact_surfaces=dict(self._contact_surfaces)
        result._contact_unions=dict(self._contact_unions)
        return result

    def _sample(self, value, points, name):
        key = (name, points.device, points.dtype)
        if key not in self._device:
            self._device[key] = torch.as_tensor(np.array(value.transpose(2, 1, 0), copy=True),
                device=points.device, dtype=points.dtype)[None, None]
        index = (points-_tensor(self.origin, points))/self.voxel_size
        grid = 2*index/_tensor(np.asarray(self.distance.shape)-1, points)-1
        return F.grid_sample(self._device[key], grid.reshape(1, 1, 1, -1, 3),
            align_corners=True, padding_mode='zeros', mode='bilinear').reshape(points.shape[:-1])

    def signed_distance(self, points):
        return self._sample(self.distance, points, 'distance')

    def supported(self, spheres):
        """Whole influence must fit fixed bounds; interpolation needs valid sites."""
        radius = spheres[..., 3:4].clamp_min(0)
        interior = ((spheres[..., :3]-radius >= _tensor(self.origin, spheres)) &
                    (spheres[..., :3]+radius <= _tensor(self.upper, spheres))).all(-1)
        # Site validity is not free-space evidence. Unknown within these
        # numerical bounds is handled by the explicit scene policy.
        return interior

    def supports_centres(self,spheres):
        return ((spheres[...,:3]>=_tensor(self.origin,spheres)) &
                (spheres[...,:3]<=_tensor(self.upper,spheres))).all(-1)

    def intersects_geometry(self,spheres):
        if self.geometry_lower is None:
            return torch.zeros(spheres.shape[:-1],dtype=torch.bool,device=spheres.device)
        return ((spheres[...,:3]+spheres[...,3:4]>=_tensor(self.geometry_lower,spheres)) &
                (spheres[...,:3]-spheres[...,3:4]<=_tensor(self.geometry_upper,spheres))).all(-1)

    def __call__(self, spheres, *, key=None):
        values = torch.relu(spheres[..., 3]-self.signed_distance(spheres[..., :3]))
        intersects=self.intersects_geometry(spheres)
        valid = self._sample(self.site_valid,spheres[..., :3],'site') > 0
        return torch.where((spheres[..., 3] > 0) & intersects & valid, values, torch.zeros_like(values))

    def prepare_contact(self, rule, radius):
        """Freeze a bounded local inventory around the complete permission patch.

        Inventory includes signed occupied cells and every source surface cell.
        A complete query sphere must fit this inventory's interior; otherwise
        its ordinary field collision remains in force and model support fails.
        """
        corners = np.asarray(list(product((-1., 1.), repeat=3)))
        widths = np.r_[rule.patch_half_size, rule.modeling_allowance_m]
        points = rule.patch_origin + (corners*widths) @ rule.patch_axes.T
        support = float(radius)+2*math.sqrt(3)*self.voxel_size
        lo = np.maximum(np.floor((points.min(0)-support-self.origin)/self.voxel_size).astype(int), 0)
        hi = np.minimum(np.ceil((points.max(0)+support-self.origin)/self.voxel_size).astype(int)+1, self.distance.shape)
        if np.any(hi <= lo):
            raise ValueError('UNSUPPORTED_MODEL: finite contact ROI lies outside the frozen field')
        slices = tuple(slice(a, b) for a, b in zip(lo, hi))
        values, valid = self.distance[slices], self.site_valid[slices]
        # ESDF zero sites retain the observed surface even when direct sign is
        # absent there. Strict negative volume requires observed TSDF sign.
        occupied = valid & ((np.abs(values) <= self.voxel_size*.51) |
                           ((values < 0) & self.observed_sign[slices]))
        # Count actual represented material cells, not free lattice samples.
        # The parent field already bounds this view's memory; only occupied
        # cells participate in per-state finite subtraction and its work cap.
        if int(np.count_nonzero(occupied))>self.contact_max_cells:
            raise ValueError('UNSUPPORTED_MODEL: finite contact material inventory exceeds frozen cell budget')
        indices = np.argwhere(occupied)+lo
        centres = self.origin+indices*self.voxel_size
        self._contact_unions.clear()
        self._contact_regions[rule.rule_id] = (centres, self.origin+lo*self.voxel_size,
            self.origin+(hi-1)*self.voxel_size)
        values_at_cells=self.distance[tuple(indices.T)] if len(indices) else np.empty(0)
        self._contact_surfaces[rule.rule_id]=centres[np.abs(values_at_cells)<=self.voxel_size*.51]

    def required_contact(self,rule,policy):
        """Numerical finite surface/permission intersection, using actual field sites.

        A patch declaration supplies no measured contact. Required contact has
        a witness only when an observed zero-surface cell lies inside every
        material, target-patch, gap and normal constraint at the queried FK.
        """
        normals,bounds=policy['normals'],policy['bounds']
        points=self._contact_surfaces[rule.rule_id]
        if not len(points):
            return torch.ones_like(policy['active'],dtype=normals.dtype)
        best=torch.full_like(bounds[...,0],float('inf'))
        for start in range(0,len(points),256):
            cells=_tensor(points[start:start+256],normals)
            excess=torch.einsum('...pc,vc->...vp',normals,cells)-bounds.unsqueeze(-2)
            best=torch.minimum(best,torch.relu(excess).amax(-1).amin(-1))
        return best

    def contact_query(self, spheres, policies):
        """Subtract only wholly permitted cells, then query every forbidden cell.

        A box support function tests all eight cell corners against each
        finite permission plane. The remaining cells use exact sphere/box
        distance. Overlapping rules form a union; no sphere-level waiver is
        possible. The inventory and native field are fixed before optimization.
        """
        ordinary = self(spheres)
        unsupported = torch.zeros_like(ordinary)
        # The finite policy has no effect before its nominal interval, with
        # uncertain future fingers, or without an actual runtime latch. These
        # modes are already explicit tensors; omit their entire cell workload.
        policies=[(rule,policy) for rule,policy in policies if bool(policy['active'].any())]
        if not policies:
            return ordinary,unsupported
        for index in sorted({i for rule, _ in policies for i in rule.sphere_indices}):
            selected = [(rule, policy) for rule, policy in policies if index in rule.sphere_indices]
            centre, radius = spheres[..., index, :3], spheres[..., index, 3]
            # Union inventories before removing permitted volume. Duplicate
            # cells are immaterial under max; a bounded concatenation avoids
            # a state-dependent nearest-neighbour or CPU query in the solve.
            arrays = [self._contact_regions[rule.rule_id] for rule, _ in selected]
            union_key=tuple(rule.rule_id for rule,_ in selected)
            if union_key not in self._contact_unions:
                self._contact_unions[union_key]=np.unique(np.concatenate([row[0] for row in arrays]),axis=0)
            cells=self._contact_unions[union_key]
            if len(cells) > self.contact_max_cells:
                raise ValueError('UNSUPPORTED_MODEL: combined finite contact cell budget exceeded')
            any_active = torch.zeros_like(radius, dtype=torch.bool)
            fits = torch.zeros_like(any_active)
            for (rule, policy), (_, lower, upper) in zip(selected, arrays):
                active = policy['active'] & policy['normal_valid']
                any_active |= active
                fits |= active & ((centre-radius[..., None] >= _tensor(lower, centre)) &
                                  (centre+radius[..., None] <= _tensor(upper, centre))).all(-1)
            maximum = torch.zeros_like(radius)
            # Each chunk bounds state x cell x plane scratch; no horizon-sized
            # full inventory or reconstructed target solid is allocated.
            chunk = max(1, min(256, 131072//max(1, radius.numel())))
            def cell_residual(centre,radius,points,*parameters):
                permitted=torch.zeros((*radius.shape,len(points)),dtype=torch.bool,device=centre.device)
                for offset in range(0,len(parameters),4):
                    normals,bounds,active,normal_valid=parameters[offset:offset+4]
                    projection=torch.einsum('...pc,vc->...vp',normals,points)
                    support=normals.abs().sum(-1)*self.voxel_size/2
                    inside=(projection+support.unsqueeze(-2)<=bounds.unsqueeze(-2)).all(-1)
                    permitted|=inside & active.unsqueeze(-1) & normal_valid.unsqueeze(-1)
                offset=(centre.unsqueeze(-2)-points).abs()-self.voxel_size/2
                distance=torch.linalg.vector_norm(torch.relu(offset),dim=-1)+offset.amax(-1).clamp_max(0)
                penetration=torch.relu(radius.unsqueeze(-1)-distance)
                return torch.where(permitted,0.,penetration).amax(-1)
            parameters=tuple(value for _,policy in selected for value in
                (policy['normals'],policy['bounds'],policy['active'],policy['normal_valid']))
            for start in range(0,len(cells),chunk):
                points=_tensor(cells[start:start+chunk],centre)
                # Recompute finite cell arithmetic in backward instead of
                # retaining state x cell intermediates for every pad sphere.
                value=(checkpoint(cell_residual,centre,radius,points,*parameters,
                    use_reentrant=False,preserve_rng_state=False) if torch.is_grad_enabled() and centre.requires_grad
                    else cell_residual(centre,radius,points,*parameters))
                maximum=torch.maximum(maximum,value)
            result = torch.where(fits, maximum, ordinary[..., index])
            ordinary = ordinary.clone()
            ordinary[..., index] = result
            unsupported[..., index] = (any_active & ~fits).to(radius.dtype)
        return ordinary, unsupported

    def record(self):
        return {'kind':'frozen_esdf_v1', 'distance':encode_array(self.distance),
            'site_valid':encode_array(self.site_valid), 'observed_sign':encode_array(self.observed_sign),
            'origin':self.origin.tolist(), 'voxel_size':self.voxel_size,
            'owner_ids':sorted(self.owner_ids), 'source':self.source, 'contact_max_cells':self.contact_max_cells}

    @classmethod
    def from_record(cls, row):
        if row.get('kind') == 'frozen_primitives_v1':
            return FrozenPrimitives(primitives=row['primitives'],owner_ids=row['owner_ids'],source=row['source'],contact_voxel_size_m=row['contact_voxel_size_m'])
        if row.get('kind') != 'frozen_esdf_v1':
            raise ValueError('UNSUPPORTED_MODEL: online binding requires its frozen field')
        return cls(decode_array(row['distance']), origin=row['origin'], voxel_size=row['voxel_size'],
            site_valid=decode_array(row['site_valid']), observed_sign=decode_array(row['observed_sign']),
            owner_ids=row['owner_ids'], source=row['source'], contact_max_cells=row['contact_max_cells'])


class FrozenPrimitives(FrozenESDF):
    """Explicit calibrated solids, with exact Torch distances and no source mesh.

    cuboid dimensions are full lengths; box dimensions are half lengths.
    Cylinder/capsule dimensions are radius and half height. Poses are xyz+wxyz.
    Contact discretization is explicit and required only for finite subtraction.
    """
    def __init__(self, *, primitives, owner_ids, source, contact_voxel_size_m=None):
        self.primitives=deepcopy(primitives)
        if not self.primitives or len(self.primitives)>64:
            raise ValueError('Explicit bounded primitive inventory required')
        self.owner_ids,self.source=frozenset(owner_ids),str(source)
        self.contact_voxel_size_m=contact_voxel_size_m
        self._contact_regions,self._device={},{}
        self._contact_surfaces={}
        self._contact_unions={}
        self.contact_max_cells=32768
        self.distance=np.empty((0,),dtype=np.float32)
        self.site_valid=self.observed_sign=np.empty((0,),dtype=bool)
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
        self.geometry_lower,self.geometry_upper=self.origin,self.upper
        self.voxel_size=float(contact_voxel_size_m or 0.)

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

    def supported(self,spheres):
        return torch.ones(spheres.shape[:-1],dtype=torch.bool,device=spheres.device)

    supports_centres=supported

    def _sample(self,value,points,name):
        if name=='distance':
            return self.signed_distance(points)
        return torch.ones(points.shape[:-1],dtype=points.dtype,device=points.device)

    def prepare_contact(self,rule,radius):
        if self.contact_voxel_size_m is None or not self.voxel_size>0:
            raise ValueError('UNSUPPORTED_MODEL: primitive contact needs explicit finite contact_voxel_size_m')
        corners=np.asarray(list(product((-1.,1.),repeat=3)))
        points=rule.patch_origin+(corners*np.r_[rule.patch_half_size,rule.modeling_allowance_m])@rule.patch_axes.T
        margin=radius+2*math.sqrt(3)*self.voxel_size
        lo=points.min(0)-margin
        hi=points.max(0)+margin
        shape=np.ceil((hi-lo)/self.voxel_size).astype(int)+1
        if math.prod(shape)>self.contact_max_cells:
            raise ValueError('UNSUPPORTED_MODEL: primitive finite contact ROI exceeds cell budget')
        indices=np.indices(shape).reshape(3,-1).T
        centres=lo+indices*self.voxel_size
        distance=self.signed_distance(torch.as_tensor(centres,dtype=torch.float64)).numpy()
        cells=centres[distance<=math.sqrt(3)*self.voxel_size/2]
        self._contact_unions.clear()
        self._contact_regions[rule.rule_id]=(cells,lo,lo+(shape-1)*self.voxel_size)
        self._contact_surfaces[rule.rule_id]=centres[np.abs(distance)<=math.sqrt(3)*self.voxel_size/2]

    def record(self):
        return {'kind':'frozen_primitives_v1','primitives':deepcopy(self.primitives),
            'owner_ids':sorted(self.owner_ids),'source':self.source,'contact_voxel_size_m':self.contact_voxel_size_m}


class OnlinePlanningScene:
    """Frozen owners, finite permissions, and explicitly bounded unknown policy."""
    def __init__(self, *, background, owners, coverage, reference_frame,
                 robot_sphere_materials, robot_geometry_source, contacts=(), gripper=None,
                 has_gripper=True, unknown_policy, robot_collision_bodies=(),
                 native_geometry_materials=None, robot_owner_permissions=(),
                 numerical_position_tolerance_m=1e-6, numerical_rotation_tolerance_rad=1e-6):
        if robot_owner_permissions:
            raise ValueError('Whole robot/owner permissions are not admitted online')
        if not isinstance(background, FrozenESDF) or any(not isinstance(o.query, FrozenESDF) for o in owners):
            raise ValueError('UNSUPPORTED_MODEL: production scenes require frozen online fields')
        self.background, self.owners, self.coverage = background, tuple(owners), coverage
        self.reference_frame, self.robot_geometry_source = reference_frame, robot_geometry_source
        self.robot_sphere_materials, self.contacts = tuple(robot_sphere_materials), tuple(contacts)
        if (not self.robot_sphere_materials or len({o.owner_id for o in self.owners})!=len(self.owners)
                or len({r.rule_id for r in self.contacts})!=len(self.contacts)):
            raise ValueError('Online owners, finite rules and robot materials require explicit unique identities')
        self.gripper, self.has_gripper = gripper, has_gripper
        self.robot_collision_bodies = tuple(deepcopy(robot_collision_bodies))
        self.native_geometry_materials = deepcopy(native_geometry_materials or {})
        if set(unknown_policy) != {'conditional_regions', 'outside_regions'} or unknown_policy['outside_regions'] != 'forbid':
            raise ValueError('Unknown policy needs bounded conditional_regions and outside_regions=forbid')
        self.unknown_policy = deepcopy(unknown_policy)
        for row in self.unknown_policy['conditional_regions']:
            lower, upper = np.asarray(row['lower_m']), np.asarray(row['upper_m'])
            if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
                    or np.any(lower >= upper) or not row.get('source')):
                raise ValueError('Conditional unknown regions need finite ordered bounds and provenance')
        self.robot_sphere_geometry, self.robot_self_collision_pairs = [], []
        self.robot_self_collision_padding = [0.]*len(self.robot_sphere_materials)
        self.robot_model = {}
        self.required_robot_links = ()
        self.robot_owner_permissions = ()
        self.grasp_contact_activations = ()
        self.observation_report = {}
        self._fixed_release_poses = {}
        self.numerical_position_tolerance_m = float(numerical_position_tolerance_m)
        self.numerical_rotation_tolerance_rad = float(numerical_rotation_tolerance_rad)
        self.required_progress_anchors = tuple(sorted({x for r in self.contacts for x in r.interval} |
            {x for owner in self.owners for a in owner.attachments for x in a.interval}))
        identities = {owner.owner_id for owner in self.owners}
        for rule in self.contacts:
            if not isinstance(rule, ContactRule) or rule.target_owner not in identities or rule.actor not in {'robot', *identities}:
                raise ValueError('Finite contact must identify an actual actor and target owner')
            materials = self.robot_sphere_materials if rule.actor == 'robot' else next(o.sphere_materials for o in self.owners if o.owner_id == rule.actor)
            if any(i >= len(materials) or materials[i] != rule.material_id for i in rule.sphere_indices):
                raise ValueError('Finite permission sphere indices must identify precisely the declared material')

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
        self.robot_self_collision_padding = padding.tolist()
        self.robot_model = deepcopy(robot_model or {})
        self.required_robot_links = tuple(row['link_name'] for row in robot_sphere_geometry)
        self._prepare_contacts(radii)

    def _prepare_contacts(self, robot_radii):
        owners = {owner.owner_id:owner for owner in self.owners}
        for rule in self.contacts:
            radii = robot_radii if rule.actor == 'robot' else owners[rule.actor].local_spheres[:, 3]
            owners[rule.target_owner].query.prepare_contact(rule, max(radii[list(rule.sphere_indices)]))

    def clone_for_rollout(self):
        result = copy(self)
        result._fixed_release_poses = dict(self._fixed_release_poses)
        result.background=self.background.clone_for_rollout()
        result.owners=tuple(copy(owner) for owner in self.owners)
        for owner in result.owners:
            owner.query=owner.query.clone_for_rollout()
            owner.attachments=deepcopy(owner.attachments)
        result.contacts=tuple(deepcopy(rule) for rule in self.contacts)
        result.unknown_policy=deepcopy(self.unknown_policy)
        result.gripper = None if self.gripper is None else copy(self.gripper)
        if result.gripper is not None:
            result.gripper.commands=deepcopy(self.gripper.commands)
        return result

    _release_pose = staticmethod(PlanningScene._release_pose)
    with_release_history = PlanningScene.with_release_history
    _permission = PlanningScene._permission
    optimizer_preferences = PlanningScene.optimizer_preferences
    project_residual = PlanningScene.project_residual

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
        by_rule={rule.rule_id:rule for rule in self.contacts}
        for row in problem.conditions:
            if row.get('kind')!='observed_attachment':
                continue
            ids=row.get('contact_rule_ids')
            carrier=row.get('carrier_id')
            relations=[a for owner in self.owners if owner.owner_id==carrier for a in owner.attachments if a.condition_id==row.get('condition_id')]
            if not isinstance(ids,(list,tuple)) or not ids or len(set(ids))!=len(ids) or len(relations)!=1:
                raise ValueError('Observed Holding requires explicit finite material IDs and its one current attachment')
            relation=relations[0]
            for identity in ids:
                rule=by_rule.get(identity)
                if rule is None or rule.actor!='robot' or rule.target_owner!=carrier or rule.interval!=relation.interval:
                    raise ValueError('Holding material rules must match the exact actor, carrier and attachment interval')
            observed=np.asarray(row.get('T_carrier_eef'),dtype=float)
            actual=np.eye(4)
            actual[:3,3]=relation.position_tool_owner
            actual[:3,:3]=Rotation.from_quat(np.asarray(relation.quaternion_tool_owner)[[1,2,3,0]]).as_matrix()
            if observed.shape!=(4,4) or not np.isfinite(observed).all() or not np.allclose(np.linalg.inv(observed),actual,atol=1e-7,rtol=0):
                raise ValueError('Holding finite permission requires the same current measured attachment transform')
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

    def _unknown(self, spheres):
        unresolved = self.coverage.violation(spheres)
        conditional = torch.zeros_like(unresolved, dtype=torch.bool)
        for row in self.unknown_policy['conditional_regions']:
            centre, radius = spheres[..., :3], spheres[..., 3:4].clamp_min(0)
            conditional |= ((centre-radius >= _tensor(row['lower_m'], centre)) &
                            (centre+radius <= _tensor(row['upper_m'], centre))).all(-1)
        return torch.where(conditional, 0., unresolved).amax(-1), torch.where(conditional, unresolved, 0.).amax(-1)

    def components(self, robot_spheres, position, quaternion, progress,
                   joint_position=None, joint_names=None, *, trajectory_time=None,
                   active_contact_rule_ids=None, contact_permission_progress=None,
                   owner_poses=None, condition_evidence=None):
        """Same nonnegative numerical constraints in solve, final check and runtime."""
        if robot_spheres.shape[:2] != progress.shape or robot_spheres.shape[-2:] != (len(self.robot_sphere_materials),4):
            raise ValueError('Online scene requires aligned [batch,time,sphere,4] native FK')
        if active_contact_rule_ids is not None and not set(active_contact_rule_ids)<={rule.rule_id for rule in self.contacts}:
            raise ValueError('Actual contact latches must identify this scene finite rules')
        permission_progress=progress if contact_permission_progress is None else contact_permission_progress
        if permission_progress.shape!=progress.shape:
            raise ValueError('Finite permission progress must align with the queried states')
        costs, spheres = {}, robot_spheres
        coordinates = half = None
        if self.gripper is not None:
            if active_contact_rule_ids is not None and (joint_position is None or not set(self.gripper.joint_names)<=set(joint_names or ())):
                raise ValueError('UNSUPPORTED_MODEL: actual hand query requires complete named measured fingers')
            if active_contact_rule_ids is not None and joint_position is not None and set(self.gripper.joint_names) <= set(joint_names or ()):
                columns = [tuple(joint_names).index(name) for name in self.gripper.joint_names]
                coordinates = joint_position[..., columns]
                half = torch.zeros_like(coordinates)
            else:
                coordinates, half, costs['gripper/controller'] = self.gripper.state(progress,
                    joint_position, joint_names, trajectory_time=trajectory_time)
            spheres = self.gripper.spheres(spheres, position, quaternion, coordinates, half)
        uncertain_indices = []
        uncertain_hand = torch.zeros_like(progress,dtype=torch.bool)
        if half is not None:
            uncertain_hand = half.amax(-1)>0
            uncertain_indices = [index for index,mapping in zip(self.gripper.sphere_indices,self.gripper.sphere_joint_translation) if np.any(mapping!=0)]
        def sum_collision(values,name,actor):
            if actor=='robot' and uncertain_indices:
                selected=torch.zeros(values.shape[-1],dtype=torch.bool,device=values.device)
                selected[uncertain_indices]=True
                uncertainty=selected & uncertain_hand.unsqueeze(-1)
                costs['conditional/'+name]=torch.where(uncertainty,values,0.).sum(-1)
                costs[name]=torch.where(uncertainty,0.,values).sum(-1)
            else:
                costs[name]=values.sum(-1)
        if self.robot_self_collision_pairs:
            padded = spheres.clone()
            padded[...,3] = padded[...,3]+_tensor(self.robot_self_collision_padding,spheres)
            pairs=np.asarray(self.robot_self_collision_pairs)
            all_self=_self_pair_sum(padded,pairs)
            known_pairs=np.asarray([pair for pair in pairs if not set(pair)&set(uncertain_indices)],dtype=np.int64).reshape(-1,2)
            known_self=_self_pair_sum(padded,known_pairs)
            costs['collision/robot/self']=torch.where(uncertain_hand,known_self,all_self)
            costs['conditional/collision/robot/self']=torch.where(uncertain_hand,(all_self-known_self).clamp_min(0),0.)
        poses = {}
        for owner in self.owners:
            if owner_poses is not None and owner.owner_id in owner_poses:
                matrix = np.asarray(owner_poses[owner.owner_id], dtype=float)
                p = _tensor(matrix[:3,3], progress).expand(*progress.shape,3)
                q = _tensor(Rotation.from_matrix(matrix[:3,:3]).as_quat()[[3,0,1,2]],progress).expand(*progress.shape,4)
                poses[owner.owner_id] = p,q
                continue
            releases = []
            for index, relation in enumerate(owner.attachments):
                key = owner.owner_id,index
                if relation.release_condition_id is None:
                    releases.append(None)
                elif key in self._fixed_release_poses:
                    releases.append(self._fixed_release_poses[key])
                else:
                    pose, found = self._release_pose(relation,position,quaternion,progress)
                    releases.append(pose)
                    costs[f'unsupported/release_anchor/{owner.owner_id}/{index}'] = ((progress>=relation.interval[1]) & ~found[:,None]).to(progress.dtype)
                if active_contact_rule_ids is not None:
                    evidence = condition_evidence or {}
                    active = (progress>=relation.interval[0]) & (progress<relation.interval[1])
                    if not evidence.get(relation.condition_id,False):
                        costs[f'unsupported/holding/{owner.owner_id}/{index}'] = active.to(progress.dtype)
            poses[owner.owner_id] = owner.pose(position,quaternion,progress,releases)
        actors = {'robot':(spheres,(position,quaternion))}
        for owner in self.owners:
            if owner.moving:
                if not owner.geometry_complete:
                    costs[f'unsupported/payload/{owner.owner_id}'] = torch.ones_like(progress)
                cover = owner.local_spheres.copy()
                cover[:,3] += owner.sphere_cover_error_m
                actors[owner.owner_id] = (_world_spheres(cover,*poses[owner.owner_id]),poses[owner.owner_id])
        for actor,(body,actor_pose) in actors.items():
            costs[f'unknown/forbidden/{actor}'], costs[f'conditional/unknown/{actor}'] = self._unknown(body)
            costs[f'scene_update/background/{actor}'] = ((body[...,3]>0) & ~self.background.supported(body)).to(progress.dtype).amax(-1)
            sum_collision(self.background(body),f'collision/{actor}/background',actor)
            for owner in self.owners:
                if owner.owner_id == actor:
                    continue
                local = _local_spheres(body,*poses[owner.owner_id])
                policies = []
                for rule in self.contacts:
                    if rule.actor != actor or rule.target_owner != owner.owner_id:
                        continue
                    if rule.rule_id not in owner.query._contact_regions:
                        raise ValueError('UNSUPPORTED_MODEL: robot sphere geometry must bind before finite contact queries')
                    policy, normal, uncertainty = self._permission(rule,body,actor_pose,poses[owner.owner_id],
                        permission_progress if actor=='robot' else progress,coordinates,half)
                    if active_contact_rule_ids is not None:
                        policy['active'] &= rule.rule_id in active_contact_rule_ids
                    policies.append((rule,policy))
                    costs[f'contact/normal/{rule.rule_id}'] = normal*policy['active']
                    costs[f'conditional/finger/{rule.rule_id}'] = uncertainty
                    if rule.required:
                        costs[f'contact/required/{rule.rule_id}'] = owner.query.required_contact(rule,policy)*policy['active']
                values, gap = owner.query.contact_query(local,policies) if policies else (owner.query(local),torch.zeros_like(local[...,3]))
                costs[f'unsupported/contact/{actor}/{owner.owner_id}'] = gap.amax(-1)
                if actor == 'robot' and owner.robot_excluded_spheres:
                    values = values.clone()
                    values[...,list(owner.robot_excluded_spheres)] = 0
                sum_collision(values,f'collision/{actor}/{owner.owner_id}',actor)
                # A small owner grid represents only its observed partition.
                # Outside it the global observation policy still applies. At
                # its boundary, spheres intersecting its represented AABB need
                # complete numerical support and cannot fall back to coarse.
                intersects=owner.query.intersects_geometry(local)
                costs[f'scene_update/{actor}/{owner.owner_id}'] = (intersects & ~owner.query.supports_centres(local) & (local[...,3]>0)).to(progress.dtype).amax(-1)
        return costs

    def evaluate(self,*args,**kwargs):
        terms = self.components(*args,**kwargs)
        hard = [value for name,value in terms.items() if not name.startswith('conditional/')]
        return torch.stack(hard).sum(0)

    def estimated_workspace_allocation(self,batch,horizon,robot_coverage_horizon=0):
        fields = [self.background]+[o.query for o in self.owners]
        shared = sum(q.distance.nbytes+q.site_valid.nbytes+q.observed_sign.nbytes for q in fields)+self.coverage._distance.numel()*4
        live = int(batch)*int(horizon)*max(1,len(self.robot_sphere_materials))*512
        # These Torch queries own no per-manager native collision buffers.
        # Autograd scratch is sequential live memory, not a retained copy for
        # every manager/line-search particle; dense immutable grids are shared.
        return dict(cpu_shared=shared,cpu_live=live,gpu_shared=shared*3,gpu_live=live,
                    gpu_retained=0,gpu_graph_live=live)

    def query_state(self,spheres,tool_pose,progress,*,joint_position=None,joint_names=None,
                    trajectory_time=0.,active_contact_rule_ids=frozenset(),owner_poses=None,condition_evidence=None):
        """Cheap actual-state query; fresh finite latches and evidence are explicit."""
        tool = np.asarray(tool_pose,dtype=float)
        p = torch.as_tensor(tool[:3,3],dtype=torch.float64)[None,None]
        q = torch.as_tensor(Rotation.from_matrix(tool[:3,:3]).as_quat()[[3,0,1,2]],dtype=torch.float64)[None,None]
        step = torch.tensor([[float(progress)]],dtype=torch.float64)
        joints = None if joint_position is None else torch.as_tensor(joint_position,dtype=torch.float64)[None,None]
        terms = self.components(torch.as_tensor(spheres,dtype=torch.float64)[None,None],p,q,step,
            joint_position=joints,joint_names=joint_names,trajectory_time=torch.tensor([[float(trajectory_time)]],dtype=torch.float64),
            active_contact_rule_ids=active_contact_rule_ids,owner_poses=owner_poses,condition_evidence=condition_evidence)
        values = {name:float(value.item()) for name,value in terms.items()}
        violations = {name:value for name,value in values.items() if not name.startswith('conditional/') and value>0}
        status = ('SCENE_UPDATE_REQUIRED' if any(k.startswith('scene_update/') for k in violations) else
                  'UNSUPPORTED_MODEL' if any(k.startswith('unsupported/') for k in violations) else
                  'COLLISION_OR_CONSTRAINT' if violations else 'PASS')
        return {'passed':not violations,'violations':[{'kind':name.split('/')[0],'name':name,'residual':value} for name,value in violations.items()],'components':values,'status':status,
            'conditional_unknown':{k:v for k,v in values.items() if k.startswith('conditional/') and v>0}}

    def describe(self):
        return {'model':'frozen_online_esdf_v1','reference_frame':self.reference_frame,
            'robot_geometry_source':self.robot_geometry_source,'owners':[o.owner_id for o in self.owners],
            'unknown_policy':deepcopy(self.unknown_policy),'required_progress_anchors':self.required_progress_anchors,
            'observation_report':deepcopy(self.observation_report),
            'scope':'fixed observed fields; finite material subtraction; conditional unknown is not free evidence'}

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
        return {'schema':'online_scene_v1','reference_frame':self.reference_frame,
            'robot_geometry_source':self.robot_geometry_source,'robot_model':deepcopy(self.robot_model),
            'numerical_position_tolerance_m':self.numerical_position_tolerance_m,
            'numerical_rotation_tolerance_rad':self.numerical_rotation_tolerance_rad,
            'robot_sphere_materials':list(self.robot_sphere_materials),
            'robot_sphere_geometry':deepcopy(self.robot_sphere_geometry),
            'robot_self_collision_pairs':deepcopy(self.robot_self_collision_pairs),
            'robot_self_collision_padding':list(self.robot_self_collision_padding),
            'background':self.background.record(),'owners':owners,'contacts':_plain([asdict(r) for r in self.contacts]),
            'gripper':hand_record,'has_gripper':self.has_gripper,'unknown_policy':deepcopy(self.unknown_policy),
            'robot_collision_bodies':_plain(self.robot_collision_bodies),'native_geometry_materials':deepcopy(self.native_geometry_materials),
            'grasp_contact_activations':deepcopy(self.grasp_contact_activations),
            'coverage':{'known_free':encode_array(self.coverage.known_free),'origin':self.coverage.origin.tolist(),
                'voxel_size':self.coverage.voxel_size,'observation_ids':list(self.coverage.observation_ids)},
            'observation_report':deepcopy(self.observation_report)}

    binding_record = to_execution_binding

    @classmethod
    def from_binding(cls,row):
        if row.get('schema')!='online_scene_v1':
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
            # Runtime reads measured finger joints. No archived command curve
            # is promoted to future physical contact or guessed feedback.
            spec['controller']='binary'
            spec['joint_bounds']=None
            hand=PrismaticGripper(**spec)
        cover=row['coverage']
        known=decode_array(cover['known_free'])
        coverage=ObservedSpace(known,origin=cover['origin'],voxel_size=cover['voxel_size'],
            observation_ids=cover['observation_ids'],reference_frame=row['reference_frame'])
        scene=cls(background=FrozenESDF.from_record(row['background']),owners=owners,coverage=coverage,
            reference_frame=row['reference_frame'],robot_sphere_materials=row['robot_sphere_materials'],
            robot_geometry_source=row['robot_geometry_source'],contacts=[ContactRule(**r) for r in row['contacts']],
            gripper=hand,has_gripper=row['has_gripper'],unknown_policy=row['unknown_policy'],
            robot_collision_bodies=row['robot_collision_bodies'],native_geometry_materials=row['native_geometry_materials'],
            numerical_position_tolerance_m=row['numerical_position_tolerance_m'],
            numerical_rotation_tolerance_rad=row['numerical_rotation_tolerance_rad'])
        scene.bind_online_robot(robot_sphere_geometry=row['robot_sphere_geometry'],
            robot_self_collision_pairs=row['robot_self_collision_pairs'],robot_self_collision_padding=row['robot_self_collision_padding'],
            robot_model=row['robot_model'])
        scene.grasp_contact_activations=tuple(deepcopy(row['grasp_contact_activations']))
        scene.observation_report=deepcopy(row['observation_report'])
        return scene
# END unified online geometry; optional audit code is never called here.
