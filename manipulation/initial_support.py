"""Finite current support, separate from future precise material placement.

The relation retains an explicitly declared observed support and the current
approximate rigid-owner model. It bounds the owner's greatest nominal depth
relative to that model's entrance; it is not a physical penetration estimate.
"""

# BEGIN stateless initial support uses existing native geometry and numerics.
from copy import deepcopy
from dataclasses import asdict, dataclass

import coal
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .collision_records import decode_array, encode_array
from .native_owner_geometry import bind_native_owner_geometry
from .planning_scene import _array, _text, _valid_interval
from .planning_problem import quaternion_rotate, quaternion_conjugate


@dataclass
class InitialSupportRelation:
    """Current actor/independent plane layer, with no future contact material.

    The target owner must be the separately observed support-plane layer, not
    an arbitrary compound obstacle relabelled as support. Its finite allowed
    material is established by entrance native intersection, not a rectangle
    copied from a future placement feature.
    """

    relation_id: str
    actor: str
    target_owner: str
    interval: tuple
    plane_origin: object
    plane_normal: object
    source: dict

    def __post_init__(self):
        for name in ('relation_id', 'actor', 'target_owner'):
            _text(getattr(self, name), name)
        if self.actor == 'robot' or self.actor == self.target_owner:
            raise ValueError('Initial support requires a distinct complete owner and observed support')
        self.interval = _valid_interval(self.interval)
        if self.interval[0] != 0:
            raise ValueError('Initial support must start at the current request entrance')
        self.plane_origin = _array(self.plane_origin, (3,), 'initial support plane origin')
        self.plane_normal = _array(self.plane_normal, (3,), 'initial support plane normal')
        if not np.isclose(np.linalg.norm(self.plane_normal), 1., atol=1e-6, rtol=0):
            raise ValueError('Initial support needs a unit outward plane normal')
        evidence = self.source
        if (not isinstance(evidence, dict)
                or evidence.get('kind') != 'observed_plane_with_declared_initial_support'
                or not isinstance(evidence.get('observation_ids'), (tuple, list))
                or not evidence['observation_ids']
                or any(not isinstance(value, str) or not value.strip() for value in evidence['observation_ids'])
                or any(not isinstance(evidence.get(key), str) or not evidence[key].strip()
                       for key in ('assumption', 'scope'))):
            raise ValueError('Initial support requires independent observed-plane provenance and a declared support assumption')
        self.source = deepcopy(evidence)

    def record(self):
        row = asdict(self)
        for key in ('plane_origin', 'plane_normal'):
            row[key] = row[key].tolist()
        row['interval'] = list(row['interval'])
        return row


def _initial_pose(owner):
    position, quaternion = owner.pose_path.sample(torch.tensor(0., dtype=torch.float64))
    pose = np.eye(4)
    pose[:3, 3] = position.detach().cpu().numpy()
    pose[:3, :3] = Rotation.from_quat(quaternion.detach().cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
    return pose


class PreparedInitialSupport:
    """One frozen source subset and a differentiable whole-owner depth bound.

    Only components of the independently declared support-plane layer that
    intersect an entrance native box are eligible. Other owners and components
    not initially touched remain obstacles. The observation's discretization
    need not fit an ideal zero-thickness plane or a future material rectangle.
    Frozen finite source identities are shared by proposal and confirmation.
    """

    def __init__(self, relation, actor, target, numerical_tolerance, *, record=None):
        from .online_geometry import FrozenESDF, FrozenPrimitives

        if (not isinstance(relation, InitialSupportRelation) or actor.owner_id != relation.actor
                or target.owner_id != relation.target_owner or not actor.moving
                or not actor.geometry_complete or target.moving
                or isinstance(target.query, FrozenPrimitives)
                or not isinstance(target.query, FrozenESDF)):
            raise ValueError('Initial support requires a complete moving owner and one static observed support field')
        if (not np.isfinite(numerical_tolerance) or numerical_tolerance < 0
                or not any(start <= relation.interval[0] and relation.interval[1] <= end
                           for start, end in (item.interval for item in actor.attachments))):
            raise ValueError('Initial support must belong to an existing entrance attachment interval')
        geometry, _, _ = bind_native_owner_geometry(actor)
        self.relation, self.target_query = relation, target.query
        self.numerical_tolerance = float(numerical_tolerance)
        self.actor_vertices = np.concatenate([vertices @ pose[:3, :3].T + pose[:3, 3]
                                             for _, vertices, pose in geometry.values()])
        self.actor_vertices.flags.writeable = False
        mapping = np.linalg.inv(_initial_pose(target)) @ _initial_pose(actor)
        entrance = self.actor_vertices @ mapping[:3, :3].T + mapping[:3, 3]
        self.initial_height = float(((entrance - relation.plane_origin) @ relation.plane_normal).min())
        self.baseline_height = min(0., self.initial_height)
        self._device = {}
        if record is None:
            ids = self._select_components(geometry, mapping)
            distance = (None if not len(ids) else target.query.source_geometry.distance_field(
                origin=target.query.origin, shape=target.query.distance.shape,
                voxel_size=target.query.voxel_size, excluded_component_ids=ids))
            self.kind = 'unchanged' if not len(ids) else ('empty' if distance is None else 'field')
        else:
            if (record.get('model') != 'initial_observed_support_v1'
                    or record.get('relation') != relation.record()
                    or record.get('initial_height_m') != self.initial_height
                    or record.get('baseline_height_m') != self.baseline_height
                    or record.get('numerical_position_tolerance_m') != self.numerical_tolerance):
                raise ValueError('Frozen initial support must retain its exact entrance and relation binding')
            ids = decode_array(record['selected_component_ids'])
            if (ids.ndim != 1 or ids.dtype != np.int64 or (len(ids) and
                    (ids[0] < 0 or ids[-1] >= target.query.source_geometry.component_count
                     or np.any(np.diff(ids) <= 0)))):
                raise ValueError('Frozen initial support component identities must be distinct source indices')
            # Check every proposed exemption against the entrance native geometry;
            # restored fields cannot smuggle a new source component into permission.
            checked = self._select_components(geometry, mapping, candidates=ids)
            if not np.array_equal(ids, checked):
                raise ValueError('Frozen initial support includes material outside its entrance support scope')
            self.kind = record['kind']
            if self.kind not in {'unchanged', 'empty', 'field'} or (self.kind == 'unchanged') != (len(ids) == 0):
                raise ValueError('Frozen initial support field kind disagrees with its permission inventory')
            if (self.kind == 'empty') != (len(ids) > 0 and len(ids) == target.query.source_geometry.component_count):
                raise ValueError('Frozen initial support cannot discard remaining source material')
            distance = decode_array(record['distance']) if self.kind == 'field' else None
            if distance is not None and (distance.dtype != np.float32 or distance.shape != target.query.distance.shape):
                raise ValueError('Frozen initial support must retain the target field layout')
        self.selected_component_ids = np.array(ids, dtype=np.int64, copy=True)
        self.selected_component_ids.flags.writeable = False
        self.query = (FrozenESDF(distance, origin=target.query.origin, voxel_size=target.query.voxel_size,
            source_geometry=target.query.source_geometry, owner_ids=target.query.owner_ids,
            source='Finite current support: ' + relation.relation_id) if self.kind == 'field' else None)

    def _select_components(self, geometry, mapping, *, candidates=None):
        source = self.target_query.source_geometry
        boxes = []
        for row, vertices, pose in geometry.values():
            transform = mapping @ pose
            points = vertices @ transform[:3, :3].T + transform[:3, 3]
            lengths = np.asarray(row['dimensions']) * (2 if row['kind'] == 'box' else 1)
            boxes.append((coal.Box(*lengths), coal.Transform3s(transform[:3, :3], transform[:3, 3]),
                          points.min(0), points.max(0)))
        request = coal.DistanceRequest()
        request.enable_signed_distance = True
        boundary = max(float(request.gjk_tolerance), float(request.epa_tolerance))
        if candidates is None:
            candidates = np.unique(np.concatenate([source.component_indices((lo-boundary, hi+boundary))
                                                   for _, _, lo, hi in boxes]))
        selected = []
        for index in candidates:
            _, _, vertices = source._vertices(int(index))
            lo, hi = vertices.min(0), vertices.max(0)
            native = None
            for shape, transform, lower, upper in boxes:
                if np.any(hi + boundary < lower) or np.any(upper + boundary < lo):
                    continue
                if native is None:
                    _, _, other, pose, _ = source.collision_component(int(index))
                    native = other, (coal.Transform3s(np.eye(3), pose) if pose.shape == (3,)
                                     else coal.Transform3s(pose[:3, :3], pose[:3, 3]))
                result = coal.DistanceResult()
                coal.distance(shape, transform, *native, request, result)
                distance = float(result.min_distance)
                if not np.isfinite(distance):
                    raise ValueError('Initial support native entrance query returned a nonfinite distance')
                if distance <= boundary:
                    selected.append(int(index))
                    break
        return np.asarray(selected, dtype=np.int64)

    def components(self, actor_pose, target_pose, progress):
        position, quaternion = actor_pose
        key = (position.device, position.dtype)
        if key not in self._device:
            self._device[key] = tuple(torch.tensor(value, device=position.device, dtype=position.dtype)
                                     for value in (self.actor_vertices, self.relation.plane_origin,
                                                   self.relation.plane_normal))
        vertices, origin, normal = self._device[key]
        world = quaternion_rotate(quaternion.unsqueeze(-2), vertices) + position.unsqueeze(-2)
        local = quaternion_rotate(quaternion_conjugate(target_pose[1]).unsqueeze(-2),
                                  world - target_pose[0].unsqueeze(-2))
        height = ((local - origin) * normal).sum(-1).amin(-1)
        # The relation has no pose lock, departure event, or persistent latch.
        active = (progress >= self.relation.interval[0]) & (progress < self.relation.interval[1])
        residual = torch.where(active, torch.relu(self.baseline_height-height-self.numerical_tolerance), 0.)
        return active, residual

    def collision_values(self, spheres, ordinary, active):
        values = (ordinary if self.kind == 'unchanged' else
                  torch.zeros_like(ordinary) if self.kind == 'empty' else self.query(spheres))
        return torch.where(active.unsqueeze(-1), values, ordinary)

    def record(self):
        return {'model': 'initial_observed_support_v1', 'relation': self.relation.record(),
                'initial_height_m': self.initial_height, 'baseline_height_m': self.baseline_height,
                'numerical_position_tolerance_m': self.numerical_tolerance,
                'selected_component_ids': encode_array(self.selected_component_ids), 'kind': self.kind,
                **({'distance': encode_array(self.query.distance)} if self.query is not None else {})}
# END this approximation never rewrites attachment poses or future support goals.
