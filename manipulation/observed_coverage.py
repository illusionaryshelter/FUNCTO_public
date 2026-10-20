"""Coverage of represented occupied sets by measured or declared allowed space."""

# BEGIN observation policy remains separate from entity collision distance.
from itertools import product
import numpy as np
from .implicit_distance import world_bounds, transform_points, transform_point_error


def query_coverage(field, pose, coverage, regions, *, precision_m=1e-4, check_request=None):
    """Bound only represented occupancy; boxes can prove coverage, not occupancy.

    Unknown-space witnesses must be actual occupied points. An overlapping
    bounding box only schedules subdivision. Robot callers do not invoke
    this owner-policy query: robot field-of-view absence allows an attempt.
    """
    known = coverage._known
    origin, h = coverage.origin, coverage.voxel_size
    shape = np.asarray(known.shape[::-1])
    regions = [(np.asarray(row['lower_m']), np.asarray(row['upper_m'])) for row in regions]

    def primitive_covered(lower, upper):
        if any(np.all(lower >= lo) and np.all(upper <= hi) for lo, hi in regions):
            return True
        lo = np.floor((lower-origin)/h+.5).astype(int)
        hi = np.ceil((upper-origin)/h+.5).astype(int)-1
        # For a zero-width face, either incident closed voxel can cover it.
        # Union incident voxels first, then require every remaining cell;
        # alternating observations across the face are therefore preserved.
        incident=[]
        for axis in range(3):
            if lower[axis]==upper[axis]:
                first=int(np.ceil((lower[axis]-origin[axis])/h-.5))
                last=int(np.floor((upper[axis]-origin[axis])/h+.5))
                incident.append(np.arange(max(0,first),min(shape[axis]-1,last)+1))
            else:
                if lo[axis]<0 or hi[axis]>=shape[axis]:return False
                incident.append(np.arange(lo[axis],hi[axis]+1))
        if any(len(indices)==0 for indices in incident):return False
        values=known[np.ix_(incident[2],incident[1],incident[0])]
        for axis in reversed(range(3)):
            if lower[axis]==upper[axis]:values=values.any(axis=2-axis,keepdims=True)
        return bool(values.all())

    def covered(box):
        lower, upper = box
        if primitive_covered(lower, upper):
            return True
        # Split at declared-region faces. On each resulting rectangular
        # piece a region covers the whole piece or none of its interior;
        # the remaining piece can therefore use the observed-cell union.
        cuts = []
        for axis in range(3):
            internal = [value[axis] for region in regions for value in region
                        if lower[axis] < value[axis] < upper[axis]]
            cuts.append(np.unique([lower[axis], *internal, upper[axis]]))
        for indices in product(*(range(max(1, len(axis)-1)) for axis in cuts)):
            lo = np.asarray([axis[index] for axis, index in zip(cuts, indices)])
            hi = np.asarray([axis[min(index+1, len(axis)-1)] for axis, index in zip(cuts, indices)])
            if not primitive_covered(lo, hi):
                return False
        return True

    def outside(box):
        lower, upper = box
        if any(np.all(upper >= lo) and np.all(lower <= hi) for lo, hi in regions):
            return False
        # Any observed closed voxel touching this box defeats an outside proof.
        lo = np.maximum(0, np.ceil((lower-origin)/h-.5).astype(int))
        hi = np.minimum(shape-1, np.floor((upper-origin)/h+.5).astype(int))
        if np.any(lo > hi):
            return True
        return not bool(known[lo[2]:hi[2]+1, lo[1]:hi[1]+1, lo[0]:hi[0]+1].any())

    pending = [] if field.root is None else [field.nodes[field.root]]
    unresolved, visited = False, 0
    while pending:
        if check_request is not None:
            check_request()
        node = pending.pop()
        visited += 1
        box = world_bounds(node.lower, node.upper, pose)
        if covered(box):
            continue
        local = field.occupied_points(node)
        if len(local):
            points = transform_points(local, pose)
            errors = transform_point_error(local, pose)
            for point, error in zip(points, errors):
                if outside((point-error, point+error)):
                    return {'status': 'unknown_witness', 'witness': point.tolist(), 'visited_boxes': visited}
        if node.cell >= 0 and np.linalg.norm(node.upper-node.lower) <= precision_m:
            unresolved = True
        else:
            children = field.split(node)
            if children:
                pending.extend(children)
            else:
                unresolved = True
    return {'status': 'unresolved' if unresolved else 'covered', 'visited_boxes': visited}
# END allowed regions grant policy permission without rewriting measured coverage.
