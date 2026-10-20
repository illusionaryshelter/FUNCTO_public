"""One signed-distance model for registered robot and owner solids.

Pinocchio supplies robot transforms and point Jacobians; Coal supplies shape
distances and their witnesses. Ordinary AABB separation skips unnecessary
narrow-phase queries. No proxy-overlap verdict, historical distance, task
permission, observed-world reconstruction or second acceptance path lives here.
"""
# BEGIN registered solid distances and their same-query witness derivatives.
from dataclasses import dataclass

import coal
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
import torch

from .robot_geometry_check import _convex_solid, _rigid_pose


@dataclass
class _Solid:
    owner: str
    name: str
    shape: object
    local_pose: np.ndarray
    frame: int | None
    owner_index: int | None
    centre: np.ndarray
    half: np.ndarray


def _primitive(kind, dimensions):
    """Convert explicit primitive dimensions without tessellation or fitting.

    This internal convention uses full box/axial lengths. FrozenPrimitives'
    half-height and box half-length conventions are converted by its caller.
    """
    values = np.asarray(dimensions, dtype=float)
    count = {'box': 3, 'sphere': 1, 'cylinder': 2, 'capsule': 2}.get(kind)
    if count is None or values.shape != (count,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Native collision needs a supported primitive and positive metric dimensions')
    return {'box': coal.Box, 'sphere': coal.Sphere,
            'cylinder': coal.Cylinder, 'capsule': coal.Capsule}[kind](*values)


class NativeCollisionModel:
    """Compile geometry/pair identities once, evaluate only current states.

    ``q_full`` follows ``joint_names`` and includes the caller's current hand
    model. Owner transforms follow ``owner_ids``. The caller compiles its
    explicit interaction permissions and uncertain geometry into ``enabled``;
    this module has no Holding, contact-material or skill semantics.

    Group outputs aggregate directly, so autograd stores group derivatives
    instead of a pair-by-owner Jacobian. All public pair records are simple
    identities, never geometry evidence cached from a previous candidate.
    """
    def __init__(self, *, robot_model, joint_names, T_reference_base,
                 self_pairs, owners, moving_owners, owner_aliases=None, owner_frames=None):
        self.joint_names = tuple(joint_names)
        self.owner_ids = tuple(owners)
        owner_aliases = {} if owner_aliases is None else owner_aliases
        if ('robot' in owners or len(set(self.joint_names)) != len(self.joint_names)
                or not set(moving_owners) <= set(owners)
                or not set(owner_aliases) <= set(owners)
                or set(owner_aliases) & set(moving_owners)):
            raise ValueError('Native collision requires distinct robot/owner identities and named joints')
        self._base = _rigid_pose(T_reference_base)
        self._model = pin.buildModelFromXML(robot_model['urdf_xml'])
        self._data = self._model.createData()
        if set(self.joint_names) != set(self._model.names[1:]):
            raise ValueError('Native collision q must cover the complete registered robot joint model')
        joints = [self._model.joints[self._model.getJointId(name)] for name in self.joint_names]
        if any(joint.nq != 1 or joint.nv != 1 for joint in joints):
            raise ValueError('Native collision requires bounded scalar robot coordinates')
        self._q_columns = np.asarray([joint.idx_q for joint in joints], dtype=np.int64)
        self._v_columns = np.asarray([joint.idx_v for joint in joints], dtype=np.int64)
        self._solids = []
        by_owner = {'robot': []}

        def add(owner, name, shape, local_pose, *, frame=None, owner_index=None):
            if not isinstance(name, str) or not name or any(self._solids[i].name == name for i in by_owner[owner]):
                raise ValueError('Every native solid needs a distinct component identity in its owner')
            shape.computeLocalAABB()
            lower, upper = np.asarray(shape.aabb_local.min_), np.asarray(shape.aabb_local.max_)
            by_owner[owner].append(len(self._solids))
            self._solids.append(_Solid(owner, name, shape, local_pose, frame, owner_index,
                                       (lower+upper)/2, (upper-lower)/2))

        for row in robot_model['geometries']:
            frame = int(self._model.getFrameId(row['frame']))
            if frame >= self._model.nframes:
                raise ValueError(f"Native robot solid has unknown frame: {row['frame']}")
            if row['kind'] == 'convex':
                shape, local = _convex_solid(row['vertices_frame']), np.eye(4)
            else:
                shape = _primitive(row['kind'], row['dimensions'])
                local = _rigid_pose(row['T_frame_geometry'])
            add('robot', row['name'], shape, local, frame=frame)
        if not by_owner['robot']:
            raise ValueError('Native collision requires the registered robot solid inventory')
        for owner_index, (identity, primitives) in enumerate(owners.items()):
            by_owner[identity] = []
            for index, row in enumerate(primitives):
                kind, dimensions = row['kind'], np.asarray(row['dimensions'], dtype=float).copy()
                if kind == 'box':
                    dimensions *= 2
                elif kind in {'cylinder', 'capsule'}:
                    dimensions[1] *= 2
                pose = np.asarray(row.get('pose', [0., 0., 0., 1., 0., 0., 0.]), dtype=float)
                if (pose.shape != (7,) or not np.isfinite(pose).all()
                        or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-6, rtol=0)):
                    raise ValueError('Native owner primitive pose must use finite xyz/wxyz')
                local = np.eye(4)
                local[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                local[:3, 3] = pose[:3]
                name = (row['native_binding']['geometry_name'] if 'native_binding' in row
                        else f'primitive:{index}')
                # A complete one-component alias already gives that component
                # its identity. Compound owners must name components explicitly
                # so a nonalias obstacle is never removed with an alias.
                aliases = tuple(owner_aliases.get(identity, ()))
                if len(primitives) == len(aliases) == 1 and 'native_binding' not in row:
                    name = aliases[0]
                add(identity, name, _primitive('box' if kind == 'cuboid' else kind, dimensions),
                    local, owner_index=owner_index)
            if not by_owner[identity]:
                raise ValueError('Native owner collision requires explicit component geometry')

        pairs, first_indices, second_indices, group_indices, margins = [], [], [], [], []
        groups = ['collision/robot/self']

        def pair(first, second, group, margin):
            margin = float(margin)
            if not np.isfinite(margin) or margin < 0:
                raise ValueError('Native collision pair margins must be finite and nonnegative')
            if group not in groups:
                groups.append(group)
            a, b = self._solids[first], self._solids[second]
            pairs.append(dict(first_owner=a.owner, first_geometry=a.name,
                              second_owner=b.owner, second_geometry=b.name,
                              group=group, margin=margin))
            first_indices.append(first)
            second_indices.append(second)
            group_indices.append(groups.index(group))
            margins.append(margin)

        robot_indices = {self._solids[index].name: index for index in by_owner['robot']}
        alias_targets, alias_actors = set(), set()
        for identity, names in owner_aliases.items():
            components = {self._solids[index].name: index for index in by_owner[identity]}
            if (len(set(names)) != len(names) or not set(names) <= set(robot_indices)
                    or not set(names) <= set(components)):
                raise ValueError(f'Native aliases need explicit registered component identities: {identity}')
            alias_targets.update(components[name] for name in names)
            alias_actors.update(robot_indices[name] for name in names)
        seen = set()
        for row in self_pairs:
            first, second = row['first'], row['second']
            key = frozenset((first, second))
            if first not in robot_indices or second not in robot_indices or len(key) != 2 or key in seen:
                raise ValueError('Native self pairs must name distinct registered solids exactly once')
            seen.add(key)
            pair(robot_indices[first], robot_indices[second], groups[0], row['margin'])
        for identity in self.owner_ids:
            for first in by_owner['robot']:
                for second in by_owner[identity]:
                    if first in alias_actors or second in alias_targets:
                        continue
                    pair(first, second, f'collision/robot/{identity}', 0.)
        # Directed actor/target groups retain the scene's existing permission
        # semantics. If both owners move, each declared actor group is explicit.
        for actor in self.owner_ids:
            if actor not in moving_owners:
                continue
            for target in self.owner_ids:
                if actor == target:
                    continue
                for first in by_owner[actor]:
                    for second in by_owner[target]:
                        pair(first, second, f'collision/{actor}/{target}', 0.)
        self.pairs, self.groups = tuple(pairs), tuple(groups)
        self._first = np.asarray(first_indices, dtype=np.int64)
        self._second = np.asarray(second_indices, dtype=np.int64)
        self._group = np.asarray(group_indices, dtype=np.int64)
        self._margin = np.asarray(margins, dtype=float)
        self._request = coal.DistanceRequest()
        self._request.enable_signed_distance = True
        # BEGIN structural motion identities use the same registered FK tree.
        self.robot_geometry_frames = {row['name']: row['frame'] for row in robot_model['geometries']}
        self._frame_joint_dependencies = {}
        # BEGIN a bound owner names a registered frame, independently of any task.
        # Query source indices select which q sample owns that frame. External
        # world transforms remain a separate physical source, named by -1.
        owner_frames = {} if owner_frames is None else dict(owner_frames)
        if not set(owner_frames) <= set(self.owner_ids):
            raise ValueError('Native frame bindings must name registered owners')
        for frame in owner_frames.values():
            self.frame_joint_dependencies(frame)
        self.owner_frames = owner_frames
        self._bound_frames = tuple(dict.fromkeys(int(self._model.getFrameId(name))
                                                for name in owner_frames.values()))
        self._owner_frame_indices = tuple(-1 if owner not in owner_frames else
            self._bound_frames.index(int(self._model.getFrameId(owner_frames[owner])))
            for owner in self.owner_ids)
        self._frames = tuple(dict.fromkeys([solid.frame for solid in self._solids if solid.frame is not None]
                                          + list(self._bound_frames)))
        self._frame_slots = {frame: index for index, frame in enumerate(self._frames)}
        # END these are immutable frame identities, not cross-evaluation FK caches.

    def frame_joint_dependencies(self, frame_name):
        """Return a fixed named-coordinate mask for one registered URDF frame.

        This is ancestry, not a numerical Jacobian test. Shared ancestors can
        cancel only when two frames consume the same query-time coordinates.
        """
        if frame_name not in self._frame_joint_dependencies:
            frame = int(self._model.getFrameId(frame_name))
            if frame >= self._model.nframes:
                raise ValueError(f'Collision dependency has unknown native frame: {frame_name}')
            names = set()
            joint = int(self._model.frames[frame].parentJoint)
            while joint:
                names.add(str(self._model.names[joint]))
                joint = int(self._model.parents[joint])
            self._frame_joint_dependencies[frame_name] = np.asarray(
                [name in names for name in self.joint_names], dtype=bool)
        return self._frame_joint_dependencies[frame_name]
        # END only the frozen URDF establishes these dependencies.

    # BEGIN unknown actuator dependencies follow this exact native FK tree.
    def geometry_depending_on(self, joint_names):
        """Return robot solids moved by any of the named scalar coordinates.

        A TCP-relative proxy translation is not a complete URDF dependency:
        ancestor revolute or prismatic joints can move an otherwise fixed
        local sphere. Traverse each native solid's frame ancestry once when
        the scene compiles its future-hand uncertainty mask.
        """
        unknown = set(joint_names)
        if not unknown <= set(self.joint_names):
            raise ValueError('Unknown native geometry coordinates must belong to the robot model')
        result = []
        for solid in self._solids:
            if solid.owner != 'robot':
                continue
            joint = int(self._model.frames[solid.frame].parentJoint)
            while joint:
                if str(self._model.names[joint]) in unknown:
                    result.append(solid.name)
                    break
                joint = int(self._model.parents[joint])
        return tuple(result)
    # END masks select native geometry, without adding an estimated joint value.

    def __call__(self, q_full, owner_transforms, enabled, owner_sources):
        self._validate_query(q_full, owner_transforms, enabled, owner_sources)
        derivatives = torch.is_grad_enabled() and (q_full.requires_grad or owner_transforms.requires_grad)
        return _NativeDistance.apply(q_full, owner_transforms, enabled, None, owner_sources, self, derivatives)

    # BEGIN signed clearances retain fixed/variable slots at every query time.
    def signed(self, q_full, owner_transforms, enabled, variable, *, owner_sources):
        """Return true minimum clearance and validity for each group/two slots.

        Slot 0 contains structurally fixed pairs; slot 1 contains pairs whose
        relative transform depends on the candidate. The caller supplies that
        structural classification independently of current distance/gradient.
        Empty slots are zero placeholders with ``valid=False``, not contact.

        ``owner_sources[b,t,n] >= 0`` binds owner n to its registered frame at
        that sample in batch b; its transform is fixed frame-from-owner.
        Source -1 instead uses the supplied world-from-owner transform. Both
        robot and bound-owner FK consume the identical q_full values.
        """
        self._validate_query(q_full, owner_transforms, enabled, owner_sources)
        if variable.shape != enabled.shape or variable.dtype != torch.bool:
            raise ValueError('Signed collision requires one structural variable mask per pair/state')
        derivatives = torch.is_grad_enabled() and (q_full.requires_grad or owner_transforms.requires_grad)
        values = _NativeDistance.apply(q_full, owner_transforms, enabled, variable, owner_sources, self, derivatives)
        constant_group = torch.as_tensor(self._group, device=enabled.device, dtype=torch.long)
        valid = torch.stack([torch.stack([
            (enabled & (variable == slot) & (constant_group == index)).any(-1)
            for slot in (False, True)], -1) for index in range(len(self.groups))], -2) if len(self.pairs) else torch.zeros_like(values, dtype=torch.bool)
        return values, valid

    def _validate_query(self, q_full, owner_transforms, enabled, owner_sources):
        expected = (*q_full.shape[:2], len(self.owner_ids), 4, 4)
        if (q_full.ndim != 3 or q_full.shape[-1] != len(self.joint_names)
                or owner_transforms.shape != expected
                or owner_sources.shape != expected[:-2] or owner_sources.dtype != torch.long
                or enabled.shape != (*q_full.shape[:2], len(self.pairs))
                or enabled.dtype != torch.bool
                or q_full.device != owner_transforms.device or q_full.dtype != owner_transforms.dtype
                or q_full.device != owner_sources.device):
            raise ValueError('Native collision requires aligned q, owner transforms and enabled pair masks')
    # END execution residuals and planning clearances share the exact query below.

    def _evaluate(self, coordinates, owner_transforms, enabled, derivatives, owner_sources, variable=None):
        """One FK/query path for residual sums or exact signed group minima."""
        states, dof = coordinates.shape
        signed = variable is not None
        channels = len(self.groups)*(2 if signed else 1)
        values = np.zeros((states, channels))
        q_gradient = np.zeros((*values.shape, dof)) if derivatives else None
        owner_gradient = np.zeros((*values.shape, len(self.owner_ids), 4, 4)) if derivatives else None
        if not np.isfinite(coordinates).all() or not np.isfinite(owner_transforms).all():
            raise ValueError('Native collision requires finite current joint coordinates and owner transforms')
        base_rotation = self._base[:3, :3]
        # BEGIN one q64 FK batch supplies every robot solid and bound owner.
        # This evaluation-local table supports release references without a
        # second FK path or a dense output-time/input-time Jacobian.
        frame_poses = np.empty((states, len(self._frames), 4, 4))
        frame_jacobians = np.zeros((states, len(self._frames), 6, dof)) if derivatives else None
        for state in range(states):
            q = np.empty(self._model.nq)
            q[self._q_columns] = coordinates[state]
            if derivatives:
                # This API computes FK and the joint Jacobians together.
                pin.computeJointJacobians(self._model, self._data, q)
            else:
                pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacements(self._model, self._data)
            for slot, frame in enumerate(self._frames):
                frame_poses[state, slot] = self._base @ self._data.oMf[frame].homogeneous
                if derivatives and dof:
                    # Pinocchio squeezes the single-column NumPy result;
                    # retain its mathematical 6-by-nv layout for all robots.
                    jacobian = np.asarray(pin.getFrameJacobian(self._model, self._data, frame,
                        pin.LOCAL_WORLD_ALIGNED)).reshape(6, self._model.nv)[:, self._v_columns]
                    frame_jacobians[state, slot, :3] = base_rotation @ jacobian[:3]
                    frame_jacobians[state, slot, 3:] = base_rotation @ jacobian[3:]
        bound_slots = [self._frame_slots[frame] for frame in self._bound_frames]
        bound_poses = frame_poses[:, bound_slots]
        bound_jacobians = frame_jacobians[:, bound_slots] if derivatives else None
        owner_matrices = owner_transforms.copy()
        for owner, frame in enumerate(self._owner_frame_indices):
            selected = owner_sources[:, owner] >= 0
            if selected.any():
                if frame < 0:
                    raise ValueError('A sourced owner must declare its registered robot frame')
                owner_matrices[selected, owner] = (bound_poses[owner_sources[selected, owner], frame]
                                                    @ owner_transforms[selected, owner])
        # END external world poses are retained only for explicit source -1.
        for state in range(states):
            poses, lower, upper = [], [], []
            for solid in self._solids:
                parent = (frame_poses[state, self._frame_slots[solid.frame]]
                          if solid.frame is not None else owner_matrices[state, solid.owner_index])
                pose = parent @ solid.local_pose
                centre = pose[:3, :3] @ solid.centre + pose[:3, 3]
                half = np.abs(pose[:3, :3]) @ solid.half
                poses.append(pose)
                lower.append(centre-half)
                upper.append(centre+half)
            lower, upper = np.asarray(lower), np.asarray(upper)
            gap = np.maximum(lower[self._first]-upper[self._second],
                             lower[self._second]-upper[self._first])
            def accumulate(solid, witness, direction, group):
                if solid.frame is not None:
                    # A fixed robot has a 3-by-0 point Jacobian. There is no
                    # joint derivative to request from Pinocchio in that case.
                    if dof == 0:
                        return
                    slot = self._frame_slots[solid.frame]
                    jacobian = frame_jacobians[state, slot]
                    offset = witness-frame_poses[state, slot, :3, 3]
                    point_jacobian = jacobian[:3]+np.cross(jacobian[3:].T, offset).T
                    q_gradient[state, group] += direction @ point_jacobian
                else:
                    matrix = owner_matrices[state, solid.owner_index]
                    point_local = matrix[:3, :3].T @ (witness-matrix[:3, 3])
                    owner_gradient[state, group, solid.owner_index, :3] += np.outer(direction, np.r_[point_local, 1.])

            def measure(index):
                first, second = self._first[index], self._second[index]
                a, b = self._solids[first], self._solids[second]
                transform_a, transform_b = poses[first], poses[second]
                result = coal.DistanceResult()
                coal.distance(a.shape, coal.Transform3s(transform_a[:3, :3], transform_a[:3, 3]),
                              b.shape, coal.Transform3s(transform_b[:3, :3], transform_b[:3, 3]),
                              self._request, result)
                distance = float(result.min_distance)
                if not np.isfinite(distance):
                    raise RuntimeError('Coal returned a nonfinite registered-solid distance')
                return distance, result

            def derivative(index, result, group, sign):
                if derivatives:
                    a, b = self._solids[self._first[index]], self._solids[self._second[index]]
                    normal = np.asarray(result.normal)
                    witness_a, witness_b = np.asarray(result.getNearestPoint1()), np.asarray(result.getNearestPoint2())
                    if not np.isfinite(np.r_[normal, witness_a, witness_b]).all():
                        raise RuntimeError('Coal returned nonfinite same-query derivative witnesses')
                    # delta d=n.(delta pB-delta pA). sign=+1 for
                    # clearance d-margin, -1 for the execution residual.
                    accumulate(a, witness_a, -sign*normal, group)
                    accumulate(b, witness_b, sign*normal, group)

            if signed:
                # A disjoint-box distance is a lower bound for enclosed
                # solids. Overlap has no signed penetration lower bound and
                # therefore must reach Coal. Bounds only skip impossible
                # minima; every returned finite clearance is a Coal query.
                bounds = np.linalg.norm(np.maximum(gap, 0.), axis=-1)-self._margin
                bounds[np.all(gap <= 0., axis=-1)] = -np.inf
                channel = self._group*2+variable[state].astype(np.int64)
                for group in range(channels):
                    indices = np.flatnonzero(enabled[state] & (channel == group))
                    indices = indices[np.argsort(bounds[indices], kind='stable')]
                    best, witness = np.inf, None
                    for index in indices:
                        if bounds[index] > best:
                            break
                        distance, result = measure(index)
                        clearance = distance-self._margin[index]
                        if clearance < best:
                            best, witness = clearance, (index, result)
                    if witness is not None:
                        values[state, group] = best
                        derivative(*witness, group, 1.)
            else:
                # Preserve the existing execution/entrance zero set and
                # positive residual sum, including declared self margins.
                indices = np.flatnonzero(enabled[state] & (gap.max(-1) <= self._margin))
                for index in indices:
                    distance, result = measure(index)
                    residual = self._margin[index]-distance
                    if residual > 0:
                        group = self._group[index]
                        values[state, group] += residual
                        derivative(index, result, group, -1.)
        return values, q_gradient, owner_gradient, bound_poses, bound_jacobians


class _NativeDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_full, owner_transforms, enabled, variable, owner_sources, model, derivatives):
        batch, horizon, dof = q_full.shape
        slots = (len(model.groups),) if variable is None else (len(model.groups), 2)
        sources = owner_sources.detach().cpu().numpy()
        if np.any(sources < -1) or np.any(sources >= horizon):
            raise ValueError('Owner frame sources must be -1 or a query index in their own batch')
        flat_sources = np.where(sources >= 0, sources+np.arange(batch)[:, None, None]*horizon, -1)
        values, q_gradient, owner_gradient, frame_poses, frame_jacobians = model._evaluate(
            q_full.detach().cpu().double().numpy().reshape(batch*horizon, dof),
            owner_transforms.detach().cpu().double().numpy().reshape(batch*horizon, len(model.owner_ids), 4, 4),
            enabled.detach().cpu().numpy().reshape(batch*horizon, len(model.pairs)), derivatives,
            flat_sources.reshape(batch*horizon, len(model.owner_ids)),
            None if variable is None else variable.detach().cpu().numpy().reshape(batch*horizon, len(model.pairs)))
        ctx.channels = values.shape[-1]
        ctx.owner_frame_indices = model._owner_frame_indices
        if derivatives:
            ctx.save_for_backward(
                q_full.new_tensor(q_gradient.reshape(batch, horizon, ctx.channels, dof)),
                owner_transforms.new_tensor(owner_gradient.reshape(batch, horizon, ctx.channels, len(model.owner_ids), 4, 4)),
                owner_transforms, owner_sources,
                q_full.new_tensor(frame_poses.reshape(batch, horizon, len(model._bound_frames), 4, 4)),
                q_full.new_tensor(frame_jacobians.reshape(batch, horizon, len(model._bound_frames), 6, dof)))
        return q_full.new_tensor(values.reshape(batch, horizon, *slots))

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        q_gradient, owner_gradient, owner_transforms, owner_sources, frame_poses, frame_jacobians = ctx.saved_tensors
        gradient = gradient.reshape(*q_gradient.shape[:2], ctx.channels)
        q_gradient = torch.einsum('btg,btgd->btd', gradient, q_gradient)
        owner_gradient = torch.einsum('btg,btgnij->btnij', gradient, owner_gradient)
        # BEGIN one owner wrench scatters only to its declared source sample.
        # The existing per-group matrix derivative is enough; no T-by-T
        # Jacobian or second cuRobo pose derivative is retained.
        batch, horizon, dof = q_gradient.shape
        for owner, frame in enumerate(ctx.owner_frame_indices):
            if frame < 0:
                continue
            source = owner_sources[..., owner]
            bound, index = source >= 0, source.clamp_min(0)
            pose = frame_poses[:, :, frame].gather(1, index[..., None, None].expand(batch, horizon, 4, 4))
            jacobian = frame_jacobians[:, :, frame].gather(1, index[..., None, None].expand(batch, horizon, 6, dof))
            world = pose @ owner_transforms[..., owner, :, :]
            matrix_gradient = owner_gradient[..., owner, :, :]
            force = matrix_gradient[..., :3, 3]
            torque = torch.cross(world[..., :3, :3].transpose(-1, -2),
                matrix_gradient[..., :3, :3].transpose(-1, -2), dim=-1).sum(-2)
            torque = torque+torch.cross(world[..., :3, 3]-pose[..., :3, 3], force, dim=-1)
            contribution = torch.einsum('btk,btkd->btd', torch.cat((force, torque), -1), jacobian)
            q_gradient.scatter_add_(1, index[..., None].expand(batch, horizon, dof),
                                    torch.where(bound[..., None], contribution, 0.))
            owner_gradient[..., owner, :, :] = torch.where(bound[..., None, None],
                pose.transpose(-1, -2) @ matrix_gradient, matrix_gradient)
        # END source -1 keeps the original external world-matrix gradient.
        return q_gradient, owner_gradient, None, None, None, None, None
# END task permissions and observed fields remain owned by the ordinary scene.
