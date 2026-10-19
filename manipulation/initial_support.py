"""Finite current support, separate from future precise material placement.

The relation retains an explicitly declared observed support and the current
approximate rigid-owner model. It permits only the actor spheres already contacting that support in the
ordinary entrance model; no exact contact or depth proof is imposed.
"""

# BEGIN stateless initial support uses existing native geometry and numerics.
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .planning_scene import _array, _text, _valid_interval


@dataclass
class InitialSupportRelation:
    """Current actor/independent plane layer, with no future contact material.

    The target owner must be the separately observed support-plane layer, not
    an arbitrary compound obstacle relabelled as support. Its finite allowed
    sphere subset is established by one entrance distance query, independently
    of the future placement target.
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
    """A fixed entrance-contact sphere subset for one declared support pair.

    One ordinary distance query at the current owner poses identifies the
    already contacting sphere indices. Subsequent states apply only this
    subset to this support owner during the declared interval. No source
    geometry is rebuilt and no precise height/contact-success constraint is
    inferred. This is a model permission, not measured physical penetration.
    """
    def __init__(self, relation, actor, target, *, record=None):
        if (not isinstance(relation, InitialSupportRelation)
                or actor.owner_id != relation.actor or target.owner_id != relation.target_owner
                or not actor.moving or actor.local_spheres is None or target.moving):
            raise ValueError('Current support requires a moving actor and one static target owner')
        if not any(a.interval[0] <= relation.interval[0] and relation.interval[1] <= a.interval[1]
                   for a in actor.attachments):
            raise ValueError('Current support must lie within the declared entrance attachment')
        self.relation = relation
        count = len(actor.local_spheres)
        if record is None:
            mapping = np.linalg.inv(_initial_pose(target)) @ _initial_pose(actor)
            local = np.array(actor.local_spheres, copy=True)
            local[:, :3] = local[:, :3] @ mapping[:3, :3].T + mapping[:3, 3]
            local[:, 3] += actor.sphere_cover_error_m
            with torch.no_grad():
                values = target.query(torch.as_tensor(local, dtype=torch.float64))
            if not torch.isfinite(values).all():
                raise ValueError('Entrance support query returned nonfinite model distances')
            selected = np.flatnonzero(values.cpu().numpy() > 0)
        else:
            if record.get('model') != 'entrance_sphere_support_v1' or record.get('relation') != relation.record():
                raise ValueError('Saved support must retain the same relation and sphere-subset schema')
            selected = np.asarray(record['sphere_indices'], dtype=np.int64)
        if (selected.ndim != 1 or len(set(selected.tolist())) != len(selected)
                or np.any(selected < 0) or np.any(selected >= count)):
            raise ValueError('Entrance support indices must belong to the actor sphere model')
        self.sphere_indices = tuple(int(value) for value in selected)

    def collision_values(self, ordinary, progress):
        active = (progress >= self.relation.interval[0]) & (progress < self.relation.interval[1])
        selected = torch.zeros(ordinary.shape[-1], dtype=torch.bool, device=ordinary.device)
        selected[list(self.sphere_indices)] = True
        return torch.where(active[..., None] & selected, 0., ordinary)

    def record(self):
        return {'model': 'entrance_sphere_support_v1', 'relation': self.relation.record(),
                'sphere_indices': list(self.sphere_indices)}
# END a support permission is local to its initial sphere subset and target.
