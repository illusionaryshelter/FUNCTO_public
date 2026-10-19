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
                 self_pairs, owners, moving_owners, owner_aliases=None):
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

    def __call__(self, q_full, owner_matrices, enabled):
        expected = (*q_full.shape[:2], len(self.owner_ids), 4, 4)
        if (q_full.ndim != 3 or q_full.shape[-1] != len(self.joint_names)
                or owner_matrices.shape != expected
                or enabled.shape != (*q_full.shape[:2], len(self.pairs))
                or enabled.dtype != torch.bool
                or q_full.device != owner_matrices.device or q_full.dtype != owner_matrices.dtype):
            raise ValueError('Native collision requires aligned q, owner transforms and enabled pair masks')
        derivatives = torch.is_grad_enabled() and (q_full.requires_grad or owner_matrices.requires_grad)
        return _NativeDistance.apply(q_full, owner_matrices, enabled, self, derivatives)

    def _evaluate(self, coordinates, owner_matrices, enabled, derivatives):
        """One FK per state, library distances only for enabled AABB candidates."""
        states, dof = coordinates.shape
        values = np.zeros((states, len(self.groups)))
        q_gradient = np.zeros((*values.shape, dof)) if derivatives else None
        owner_gradient = np.zeros((*values.shape, len(self.owner_ids), 4, 4)) if derivatives else None
        if not np.isfinite(coordinates).all() or not np.isfinite(owner_matrices).all():
            raise ValueError('Native collision requires finite current joint coordinates and owner transforms')
        base_rotation = self._base[:3, :3]
        for state in range(states):
            q = np.empty(self._model.nq)
            q[self._q_columns] = coordinates[state]
            if derivatives:
                # This API computes FK and the joint Jacobians together.
                pin.computeJointJacobians(self._model, self._data, q)
            else:
                pin.forwardKinematics(self._model, self._data, q)
            pin.updateFramePlacements(self._model, self._data)
            poses, lower, upper = [], [], []
            for solid in self._solids:
                parent = (self._base @ self._data.oMf[solid.frame].homogeneous
                          if solid.frame is not None else owner_matrices[state, solid.owner_index])
                pose = parent @ solid.local_pose
                centre = pose[:3, :3] @ solid.centre + pose[:3, 3]
                half = np.abs(pose[:3, :3]) @ solid.half
                poses.append(pose)
                lower.append(centre-half)
                upper.append(centre+half)
            lower, upper = np.asarray(lower), np.asarray(upper)
            axis_gap = np.maximum(lower[self._first]-upper[self._second],
                                  lower[self._second]-upper[self._first]).max(-1)
            candidates = np.flatnonzero(enabled[state] & (axis_gap <= self._margin))
            frame_jacobians = {}

            def accumulate(solid, witness, direction, group):
                if solid.frame is not None:
                    # A fixed robot has a 3-by-0 point Jacobian. There is no
                    # joint derivative to request from Pinocchio in that case.
                    if dof == 0:
                        return
                    if solid.frame not in frame_jacobians:
                        # BEGIN Pinocchio's NumPy binding squeezes a one-column matrix.
                        # Its mathematical Jacobian remains 6-by-nv for every
                        # scalar-joint count, including a one-coordinate robot.
                        frame_jacobians[solid.frame] = np.asarray(pin.getFrameJacobian(
                            self._model, self._data, solid.frame, pin.LOCAL_WORLD_ALIGNED)).reshape(6, self._model.nv).copy()
                        # END joint-count changes cannot change derivative rank.
                    jacobian = frame_jacobians[solid.frame]
                    point_base = base_rotation.T @ (witness-self._base[:3, 3])
                    offset = point_base-self._data.oMf[solid.frame].translation
                    point_jacobian = base_rotation @ (jacobian[:3]+np.cross(jacobian[3:].T, offset).T)
                    q_gradient[state, group] += direction @ point_jacobian[:, self._v_columns]
                else:
                    matrix = owner_matrices[state, solid.owner_index]
                    point_local = matrix[:3, :3].T @ (witness-matrix[:3, 3])
                    owner_gradient[state, group, solid.owner_index, :3] += np.outer(direction, np.r_[point_local, 1.])

            for index in candidates:
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
                residual = self._margin[index]-distance
                if residual <= 0:
                    continue
                group = self._group[index]
                values[state, group] += residual
                if derivatives:
                    normal = np.asarray(result.normal)
                    witness_a, witness_b = np.asarray(result.getNearestPoint1()), np.asarray(result.getNearestPoint2())
                    if not np.isfinite(np.r_[normal, witness_a, witness_b]).all():
                        raise RuntimeError('Coal returned nonfinite same-query derivative witnesses')
                    # rho=margin-d and delta d=n.(delta pB-delta pA).
                    accumulate(a, witness_a, normal, group)
                    accumulate(b, witness_b, -normal, group)
        return values, q_gradient, owner_gradient


class _NativeDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_full, owner_matrices, enabled, model, derivatives):
        batch, horizon, dof = q_full.shape
        values, q_gradient, owner_gradient = model._evaluate(
            q_full.detach().cpu().double().numpy().reshape(batch*horizon, dof),
            owner_matrices.detach().cpu().double().numpy().reshape(batch*horizon, len(model.owner_ids), 4, 4),
            enabled.detach().cpu().numpy().reshape(batch*horizon, len(model.pairs)), derivatives)
        if derivatives:
            ctx.save_for_backward(
                q_full.new_tensor(q_gradient.reshape(batch, horizon, len(model.groups), dof)),
                owner_matrices.new_tensor(owner_gradient.reshape(batch, horizon, len(model.groups), len(model.owner_ids), 4, 4)))
        return q_full.new_tensor(values.reshape(batch, horizon, len(model.groups)))

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        q_gradient, owner_gradient = ctx.saved_tensors
        return (torch.einsum('btg,btgd->btd', gradient, q_gradient),
                torch.einsum('btg,btgnij->btnij', gradient, owner_gradient), None, None, None)
# END task permissions and observed fields remain owned by the ordinary scene.
