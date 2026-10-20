"""Optimization directions and decisions from the same observed implicit sets."""

# BEGIN geometry decisions are independent of the differentiable local direction.
import numpy as np
import torch

from .implicit_distance import inverse_points, query_distance
from .native_field_distance import query_native_field
from .planning_problem import quaternion_rotate


def _world(point, pose):
    position, quaternion = pose
    local = torch.as_tensor(point, dtype=position.dtype, device=position.device)
    return position + quaternion_rotate(quaternion, local)


def pair_residual(first, first_pose, first_matrix, second, second_pose, second_matrix, *,
                  permission=None, check_request=None, precision_m=1e-4, zero=None):
    """Query one rigid pair and attach a local rigid-motion derivative.

    Confirmed penetration has classification value one; clear and unresolved
    queries have value zero. The value is not a metric penetration depth.
    A common-point penetration uses a local zero-section normal from the
    observed field. Native witnesses retain their interval bounds; their
    midpoint supplies only a local rigid-motion direction. The direction
    neither estimates penetration nor supplies candidate acceptance evidence.

    ``zero`` may be one scalar from a batch's differentiable zero baseline.
    A standalone call constructs that baseline from its two position tensors.
    """
    # BEGIN native models retain their geometry; observations retain their cells.
    from .implicit_distance import ObservedField
    native_first = not isinstance(first, ObservedField)
    query = query_native_field if native_first else query_distance
    result = query(
        first, first_matrix, second, second_matrix,
        precision_m=precision_m, permission=permission, check_request=check_request)
    # BEGIN scene batches may share their already differentiable zero baseline.
    if zero is None:
        zero = first_pose[0].sum() * 0 + second_pose[0].sum() * 0
    # END the supplied baseline changes no geometric evidence or local direction.
    status = result["status"]
    if status == "clear":
        return zero, result
    if status == "unresolved":
        # The scene reports uncertainty separately. It is not collision
        # evidence and receives no fabricated penetration or direction.
        return zero, result
    first_local = inverse_points(result["witness_first"], first_matrix)
    second_local = inverse_points(result["witness_second"], second_matrix)
    # The query established a common point in the represented sets. Keep its
    # local coordinate for a rigid derivative and seek the observed boundary.
    boundary = second.local_zero(second_local)
    if boundary is not None:
        _, target, normal = boundary
        normal = quaternion_rotate(second_pose[1], torch.as_tensor(
            normal, dtype=second_pose[0].dtype, device=second_pose[0].device))
        signed = ((_world(first_local, first_pose) - _world(target, second_pose)) * normal).sum()
    else:
        boundary = None if native_first else first.local_zero(first_local)
        if boundary is None:
            result["direction"] = "unavailable"
            return zero + 1., result
        _, target, normal = boundary
        normal = quaternion_rotate(first_pose[1], torch.as_tensor(
            normal, dtype=first_pose[0].dtype, device=first_pose[0].device))
        signed = ((_world(second_local, second_pose) - _world(target, first_pose)) * normal).sum()
    result["direction"] = "local_zero_section_normal"
    return zero + 1. - (signed - signed.detach()), result
    # END local directions never certify a native or observed point's membership.
# END the derivative never changes the classification, threshold or occupied sets.


def query_region_intersection(field, pose, normals, bounds, *, precision_m=1e-6, check_request=None):
    """Find a supported zero surface in the finite convex contact region.

    An original zero sample or a sign-changing segment inside one supported
    cell proves surface contact by continuity. Negative interior material is
    insufficient. Interval bounds prune cells without a zero; finite unresolved
    boxes remain distinct from missing contact.
    """
    from .implicit_distance import (PairPermission, world_bounds, transform_points,
        transform_point_error, box_corners, CORNERS, _EDGES, _Node)
    region=(np.asarray(normals,float),np.asarray(bounds,float))
    pending=[] if field.root is None else [field.nodes[field.root]]
    unresolved=False
    # Stored occupied bounds may be tightened to the nonpositive side. Restore
    # each full supported cell so the sign test sees both sides of its surface.
    expanded=set()
    while pending:
        if check_request is not None:check_request()
        node=pending.pop()
        if node.cell>=0 and node.cell not in expanded:
            expanded.add(node.cell)
            lower=field.cell_origins[node.cell]
            node=_Node(lower,lower+field.voxel_size,cell=node.cell)
        if PairPermission._classify(world_bounds(node.lower,node.upper,pose),region)=='outside':
            continue
        if node.children:
            pending.extend(field.nodes[i] for i in node.children)
            continue
        corners=box_corners(node.lower,node.upper)
        values,_,errors=field.cell_value(node.cell,corners)
        if np.min(values-errors)>0 or np.max(values+errors)<0:
            continue
        original=field.cell_origins[node.cell]+CORNERS*field.voxel_size
        zeros=original[field._cell_values[node.cell]==0]
        for local in zeros:
            world=transform_points(local,pose);error=transform_point_error(local,pose)
            if PairPermission._classify((world-error,world+error),region)=='inside':
                return {'status':'contact','witness':world.tolist(),'evidence':'original_zero_sample'}
        world=transform_points(corners,pose)
        candidates=[(node.lower+node.upper)/2]
        # Clip each cell edge against the task halfspaces. Candidate points
        # still need their arithmetic inclusion test below; clipping alone
        # cannot turn a near-boundary approximation into accepted contact.
        for a,b in _EDGES:
            start,end=world[a],world[b]
            slope=region[0]@(end-start);offset=region[1]-region[0]@start
            lo,hi=0.,1.
            for v,w in zip(slope,offset):
                if v>0:hi=min(hi,w/v)
                elif v<0:lo=max(lo,w/v)
                elif w<0:lo,hi=1.,0.;break
            if lo<=hi:
                candidates.extend(corners[a]+t*(corners[b]-corners[a]) for t in (lo,(lo+hi)/2,hi))
        candidates=np.asarray(candidates)
        world=transform_points(candidates,pose);error=transform_point_error(candidates,pose)
        inside=np.array([PairPermission._classify((point-e,point+e),region)=='inside'
                         for point,e in zip(world,error)])
        values,_,errors=field.cell_value(node.cell,candidates)
        negative=np.flatnonzero(inside&(values < -errors))
        positive=np.flatnonzero(inside&(values > errors))
        if len(negative) and len(positive):
            a,b=negative[0],positive[0]
            return {'status':'contact','segment':world[[a,b]].tolist(),
                    'evidence':'opposite_sign_supported_segment'}
        axis=int(np.argmax(node.upper-node.lower))
        mid=(node.lower[axis]+node.upper[axis])/2
        if np.linalg.norm(node.upper-node.lower)<=precision_m or mid in (node.lower[axis],node.upper[axis]):
            unresolved=True
        else:
            hi,lo=node.upper.copy(),node.lower.copy();hi[axis],lo[axis]=mid,mid
            pending.extend((_Node(node.lower,hi,cell=node.cell),_Node(lo,node.upper,cell=node.cell)))
    return {'status':'unresolved' if unresolved else 'missing_contact'}
# END task contact requires a zero surface, while collision uses occupied sets.


def query_minimum_height(field, pose, origin, normal, *, precision_m, check_request=None):
    """Bound the minimum plane height over the represented occupied set.

    Spatial boxes provide lower bounds; actual supported occupied points
    provide upper bounds and a rigid-motion derivative witness. The stopping
    precision bounds numerical uncertainty and never changes a task tolerance.
    """
    import heapq
    from itertools import count
    from .implicit_distance import (world_bounds, transform_points, transform_point_error, _GAMMA)
    if not np.isfinite(precision_m) or precision_m<=0:
        raise ValueError('A directional extremum requires positive finite precision')
    if field.root is None:
        return dict(status='missing_observation',lower_m=None,upper_m=None,witness_local=None)
    origin,normal=np.asarray(origin,float),np.asarray(normal,float)
    pending,serial=[],count()
    upper,witness=np.inf,None
    def push(node):
        lower_box,upper_box=world_bounds(node.lower,node.upper,pose)
        corner=np.where(normal>=0,lower_box,upper_box)
        terms=(corner-origin)*normal
        lower=float(terms.sum()-_GAMMA*(np.abs(terms).sum()+np.abs(origin*normal).sum()))
        if lower<=upper:heapq.heappush(pending,(lower,next(serial),node))
    push(field.nodes[field.root])
    final_lower=-np.inf
    visits=0
    while pending:
        lower,_,node=heapq.heappop(pending)
        visits+=1
        if check_request is not None:check_request()
        if upper-lower<=precision_m:
            final_lower=lower;break
        points=field.occupied_points(node)
        if len(points):
            world=transform_points(points,pose);errors=transform_point_error(points,pose)
            terms=(world-origin)*normal
            values=terms.sum(-1)+(errors*np.abs(normal)).sum(-1)
            values+=_GAMMA*(np.abs(terms).sum(-1)+np.abs(origin*normal).sum())
            index=int(np.argmin(values))
            if values[index]<upper:upper=float(values[index]);witness=points[index].copy()
        if upper-lower<=precision_m:
            # Other queued cells can still have a smaller lower bound.
            final_lower=min(lower,pending[0][0] if pending else lower)
            if upper-final_lower<=precision_m:break
        children=field.split(node)
        if not children:
            final_lower=min(lower,pending[0][0] if pending else lower)
            return dict(status='unresolved',lower_m=final_lower,upper_m=upper,
                        witness_local=None if witness is None else witness.tolist(),visited_boxes=visits)
        for child in children:push(child)
    else:
        final_lower=upper
    return dict(status='bounded',lower_m=final_lower,upper_m=upper,
                witness_local=None if witness is None else witness.tolist(),visited_boxes=visits)
# END directional extrema and pair distances share the original field domain.
