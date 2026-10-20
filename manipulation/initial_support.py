"""Declared initial support over retained raw observed occupied sets."""

# BEGIN initial support grants only entrance-touched source-cell material.
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .implicit_distance import (ObservedField, PairPermission, query_distance, world_bounds,
                               box_distance)
from .observed_collision import query_minimum_height
from .observed_mapping import FrozenObservedField
from .planning_scene import _array, _text, _valid_interval
from .planning_problem import quaternion_rotate, quaternion_conjugate


@dataclass
class InitialSupportRelation:
    """Declared current actor and observed support-plane layer.

    The target owner must be the separately observed support-plane layer, not
    an arbitrary compound obstacle relabelled as support. Its finite allowed
    material is established by entrance observed-field intersection, not a rectangle
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
            raise ValueError('Initial support requires a distinct observed owner and observed support')
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
    """Freeze entrance-touched cells and bound the whole observed actor height.

    A component is the nonpositive trilinear set within one fully supported
    source cell. Only actual entrance intersection selects it. Adjacent cells
    retain their ordinary separation requirement, including at permission
    boundaries. The relation never adds a distance halo to selected material.
    """

    def __init__(self, relation, actor, target, numerical_tolerance):
        if (actor.owner_id!=relation.actor or target.owner_id!=relation.target_owner
                or not actor.moving or target.moving
                or not isinstance(actor.query,FrozenObservedField)
                or not isinstance(target.query,FrozenObservedField)):
            raise ValueError('Initial support needs a moving observed actor and static observed target')
        if (not np.isfinite(numerical_tolerance) or numerical_tolerance<=0
                or not any(item.interval[0]<=relation.interval[0] and relation.interval[1]<=item.interval[1]
                           for item in actor.attachments)):
            raise ValueError('Initial support must use its entrance attachment and positive numerical tolerance')
        self.relation,self.actor_field,self.target_field=relation,actor.query,target.query
        self.numerical_tolerance=float(numerical_tolerance)
        mapping=np.linalg.inv(_initial_pose(target))@_initial_pose(actor)
        self.initial_height=query_minimum_height(actor.query,mapping,relation.plane_origin,
            relation.plane_normal,precision_m=self.numerical_tolerance/4)
        self.selected_cells=[]
        self.unresolved_cells=[]
        if actor.query.bounds is not None:
            actor_bounds=world_bounds(*actor.query.bounds,mapping)
            # Traverse the existing hierarchy to prune spatially unrelated
            # source cells. Bounding overlap schedules a query, never permission.
            pending=[] if target.query.root is None else [target.query.nodes[target.query.root]]
            while pending:
                node=pending.pop()
                if box_distance(actor_bounds,(node.lower,node.upper))>0:continue
                if node.children:
                    pending.extend(target.query.nodes[i] for i in node.children);continue
                values=target.query._cell_values[node.cell].reshape(2,2,2)
                cell=ObservedField(values,np.ones((2,2,2),bool),
                    origin=target.query.cell_origins[node.cell],voxel_size=target.query.voxel_size)
                result=query_distance(actor.query,mapping,cell,np.eye(4),separation_m=0.,
                                      precision_m=self.numerical_tolerance/4)
                if result['status']=='intersection':self.selected_cells.append(node.cell)
                elif result['status']=='unresolved':self.unresolved_cells.append(node.cell)
        self.selected_cells=tuple(sorted(self.selected_cells))
        self.unresolved_cells=tuple(sorted(self.unresolved_cells))

    def permission(self,target_matrix,*,additional=None):
        """Remove selected original cell IDs from this pair's second domain."""
        regions=[] if additional is None else list(additional.regions)
        excluded=set(self.selected_cells)
        if additional is not None:excluded.update(additional.excluded_second_cells)
        return PairPermission(regions,excluded_second_cells=excluded)

    def components(self,actor_pose,target_pose,progress,*,check_request=None):
        """Compare height intervals; expose unresolved evidence independently."""
        from .online_geometry import _matrices
        actor_matrices,target_matrices=_matrices(actor_pose),_matrices(target_pose)
        residuals,unresolved=[],[]
        initial=self.initial_height
        for index in np.ndindex(tuple(progress.shape)):
            zero=actor_pose[0][index].sum()*0+target_pose[0][index].sum()*0
            active=self.relation.interval[0]<=float(progress[index].detach())<self.relation.interval[1]
            value,uncertain=zero,zero
            if active:
                mapping=np.linalg.inv(target_matrices[index])@actor_matrices[index]
                height=query_minimum_height(self.actor_field,mapping,self.relation.plane_origin,
                    self.relation.plane_normal,precision_m=self.numerical_tolerance/4,check_request=check_request)
                if initial['status']=='missing_observation' or height['status']=='missing_observation':
                    uncertain=zero+1
                else:
                    baseline_lower=min(0.,initial['lower_m']);baseline_upper=min(0.,initial['upper_m'])
                    certain=height['lower_m']>=baseline_upper-self.numerical_tolerance
                    violation=height['upper_m']<baseline_lower-self.numerical_tolerance
                    if violation:
                        magnitude=baseline_lower-height['upper_m']-self.numerical_tolerance
                        local=progress.new_tensor(height['witness_local'])
                        world=quaternion_rotate(actor_pose[1][index],local)+actor_pose[0][index]
                        relative=quaternion_rotate(quaternion_conjugate(target_pose[1][index]),
                                                   world-target_pose[0][index])
                        direction=((relative-progress.new_tensor(self.relation.plane_origin))
                                   *progress.new_tensor(self.relation.plane_normal)).sum()
                        value=zero+magnitude-direction+direction.detach()
                    uncertain=zero+float(not(certain or violation))
            residuals.append(value);unresolved.append(uncertain)
        return {f'initial_support/{self.relation.relation_id}':torch.stack(residuals).reshape(progress.shape),
                f'unresolved/initial_support/{self.relation.relation_id}':torch.stack(unresolved).reshape(progress.shape)}

    def record(self):
        return dict(model='initial_observed_support_v2',relation=self.relation.record(),
            initial_height=self.initial_height,numerical_position_tolerance_m=self.numerical_tolerance,
            selected_cells=list(self.selected_cells),unresolved_cells=list(self.unresolved_cells),
            component_semantics='nonpositive set in each fully supported source trilinear cell')
# END the original occupied field remains shared by collision and height queries.
