"""Batched values of the existing supported trilinear field and native solid.

The field kernel implements ObservedField.sample's XYZ lookup, including
closed incident cells. It retains the original float32 samples and performs
coordinate and interpolation arithmetic in float64. Native convex queries
use Warp's maintained point-to-mesh API on the registered robot boundary.
"""

# BEGIN fixed query arrays use the installed Warp execution and mesh library.
import numpy as np
import torch
import warp as wp


@wp.func
def _field_value(point: wp.vec3d,
        values: wp.array(dtype=wp.float32, ndim=3),
        valid: wp.array(dtype=wp.bool, ndim=3), origin: wp.vec3d, h: wp.float64,
        ):
    zero, one = wp.float64(0.0), wp.float64(1.0)
    g = (point-origin)/h
    nx, ny, nz = valid.shape[0], valid.shape[1], valid.shape[2]
    if g[0] < 0.0 or g[1] < 0.0 or g[2] < 0.0 or g[0] > wp.float64(nx) or g[1] > wp.float64(ny) or g[2] > wp.float64(nz):
        return zero, False
    ix = wp.min(wp.int32(wp.floor(g[0])), nx-1)
    iy = wp.min(wp.int32(wp.floor(g[1])), ny-1)
    iz = wp.min(wp.int32(wp.floor(g[2])), nz-1)
    x, y, z = ix, iy, iz
    found = valid[x, y, z]
    # The floor cell has priority, followed by the same incident-cell order
    # as ObservedField.sample. Only an exact lattice face admits its neighbor.
    for corner in range(1, 8):
        dx, dy, dz = (corner >> 2) & 1, (corner >> 1) & 1, corner & 1
        cx, cy, cz = ix-dx, iy-dy, iz-dz
        eligible = ((dx == 0 or g[0] == wp.float64(ix)) and
                    (dy == 0 or g[1] == wp.float64(iy)) and
                    (dz == 0 or g[2] == wp.float64(iz)))
        if not found and eligible and cx >= 0 and cy >= 0 and cz >= 0:
            if valid[cx, cy, cz]:
                found = True
                x, y, z = cx, cy, cz
    scalar = wp.float64(0.0)
    if found:
        u = g-wp.vec3d(wp.float64(x), wp.float64(y), wp.float64(z))
        for corner in range(8):
            dx, dy, dz = (corner >> 2) & 1, (corner >> 1) & 1, corner & 1
            wx = u[0] if dx else one-u[0]
            wy = u[1] if dy else one-u[1]
            wz = u[2] if dz else one-u[2]
            scalar += wp.float64(values[x+dx, y+dy, z+dz])*wx*wy*wz
    return scalar, found


@wp.func
def _native_value(p: wp.vec3d, mesh: wp.uint64,
        kind: wp.int32, shape: wp.vec3d, lower: wp.vec3d, upper: wp.vec3d,
        unit: wp.float64, search_distance: wp.float32):
    zero = wp.float64(0.0)
    # The native AABB only removes distant queries. Its faces never supply
    # an occupied value or a contact normal for the registered entity.
    if p[0] < lower[0]-unit or p[1] < lower[1]-unit or p[2] < lower[2]-unit or p[0] > upper[0]+unit or p[1] > upper[1]+unit or p[2] > upper[2]+unit:
        return unit
    if kind == 0:
        query = wp.vec3(wp.float32(p[0]), wp.float32(p[1]), wp.float32(p[2]))
        hit = wp.mesh_query_point(mesh, query, search_distance)
        value = search_distance
        if hit.result:
            closest = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
            value = wp.length(query-closest)*hit.sign
        return wp.float64(value)
    elif kind == 1:
        d = wp.vec3d(wp.abs(p[0]), wp.abs(p[1]), wp.abs(p[2]))-shape
        outside = wp.vec3d(wp.max(d[0], zero), wp.max(d[1], zero), wp.max(d[2], zero))
        return wp.length(outside)+wp.min(wp.max(d[0], wp.max(d[1], d[2])), zero)
    else:
        radial = wp.sqrt(p[0]*p[0]+p[1]*p[1])-shape[0]
        axial = wp.abs(p[2])-shape[1]
        radial_out, axial_out = wp.max(radial, zero), wp.max(axial, zero)
        return wp.sqrt(radial_out*radial_out+axial_out*axial_out)+wp.min(wp.max(radial, axial), zero)


@wp.func
def _permitted(point: wp.vec3d, row: wp.int32,
        row_regions: wp.array(dtype=wp.int32), region_planes: wp.array(dtype=wp.int32),
        normals: wp.array(dtype=wp.vec3d), bounds: wp.array(dtype=wp.float64)):
    # Each region is the intersection of its two declared finite regions;
    # different regions form a union. Planes are expressed in the source frame.
    for region in range(row_regions[row], row_regions[row+1]):
        inside = bool(True)
        for plane in range(region_planes[region], region_planes[region+1]):
            if wp.dot(point, normals[plane]) > bounds[plane]:
                inside = False
        if inside:
            return True
    return False


@wp.func
def _transform(point: wp.vec3d, transform: wp.mat44d):
    value = transform*wp.vec4d(point[0], point[1], point[2], wp.float64(1.0))
    return wp.vec3d(value[0], value[1], value[2])


@wp.kernel
def _field_values(points: wp.array(dtype=wp.vec3d),
        values: wp.array(dtype=wp.float32, ndim=3), valid: wp.array(dtype=wp.bool, ndim=3),
        origin: wp.vec3d, h: wp.float64,
        result: wp.array(dtype=wp.float64), supported: wp.array(dtype=wp.bool)):
    index = wp.tid()
    value, support = _field_value(points[index], values, valid, origin, h)
    result[index] = value
    supported[index] = support


# BEGIN maintained Warp tile reductions avoid global per-point result arrays.
QUERY_BLOCK_SIZE = wp.constant(256)


@wp.func
def _store_tile(row: wp.int32, block: wp.int32, lane: wp.int32, column: wp.int32,
        value: wp.float64, queried: wp.int32, unknown: wp.int32,
        maxima: wp.array(dtype=wp.float64, ndim=2), winners: wp.array(dtype=wp.int32, ndim=2),
        counts: wp.array(dtype=wp.int32, ndim=2), unknowns: wp.array(dtype=wp.int32, ndim=2)):
    maximum = wp.tile_max(wp.tile(value))[0]
    choice = int(2147483647)
    if value == maximum:
        choice = column
    winner = wp.tile_min(wp.tile(choice))[0]
    count = wp.tile_sum(wp.tile(queried))[0]
    missing = wp.tile_sum(wp.tile(unknown))[0]
    if lane == 0:
        maxima[row, block] = maximum
        winners[row, block] = winner
        counts[row, block] = count
        unknowns[row, block] = missing


@wp.kernel
def _field_pair_tiles(block_width: wp.int32, points: wp.array(dtype=wp.vec3d), poses: wp.array(dtype=wp.mat44d),
        row_regions: wp.array(dtype=wp.int32), region_planes: wp.array(dtype=wp.int32),
        normals: wp.array(dtype=wp.vec3d), bounds: wp.array(dtype=wp.float64),
        values: wp.array(dtype=wp.float32, ndim=3), valid: wp.array(dtype=wp.bool, ndim=3),
        origin: wp.vec3d, h: wp.float64, scale: wp.float64,
        maxima: wp.array(dtype=wp.float64, ndim=2), winners: wp.array(dtype=wp.int32, ndim=2),
        counts: wp.array(dtype=wp.int32, ndim=2), unknowns: wp.array(dtype=wp.int32, ndim=2)):
    row, block, lane = wp.tid()
    column = block*block_width+lane
    residual, queried, unknown = -wp.float64(wp.inf), int(0), int(0)
    if column < points.shape[0]:
        point = points[column]
        if not _permitted(point, row, row_regions, region_planes, normals, bounds):
            queried = 1
            value, supported = _field_value(_transform(point, poses[row]), values, valid, origin, h)
            if supported:
                residual = -value/scale
            else:
                unknown = 1
    _store_tile(row, block, lane, column, residual, queried, unknown, maxima, winners, counts, unknowns)


@wp.kernel
def _native_pair_tiles(block_width: wp.int32, points: wp.array(dtype=wp.vec3d), poses: wp.array(dtype=wp.mat44d),
        row_regions: wp.array(dtype=wp.int32), region_planes: wp.array(dtype=wp.int32),
        normals: wp.array(dtype=wp.vec3d), bounds: wp.array(dtype=wp.float64),
        mesh: wp.uint64, kind: wp.int32, shape: wp.vec3d,
        lower: wp.vec3d, upper: wp.vec3d, scale: wp.float64, distance: wp.float32,
        maxima: wp.array(dtype=wp.float64, ndim=2), winners: wp.array(dtype=wp.int32, ndim=2),
        counts: wp.array(dtype=wp.int32, ndim=2), unknowns: wp.array(dtype=wp.int32, ndim=2)):
    row, block, lane = wp.tid()
    column = block*block_width+lane
    residual, queried = -wp.float64(wp.inf), int(0)
    if column < points.shape[0]:
        point = points[column]
        if not _permitted(point, row, row_regions, region_planes, normals, bounds):
            queried = 1
            value = _native_value(_transform(point, poses[row]), mesh, kind, shape, lower, upper, scale, distance)
            residual = -value/scale
    _store_tile(row, block, lane, column, residual, queried, 0, maxima, winners, counts, unknowns)
# END every original point contributes to the same maximum and support counters.


def _pair_inputs(points, source_poses, target_poses, permissions):
    """Compose small pose/plane arrays before querying every fixed local point.

    A permission plane n·x<=b in world coordinates becomes
    (R_source.T@n)·a<=b-n·t_source. Its paired-region and union grouping
    remains the original declaration. The target transform is T_target^-1
    T_source, so no world/target point tensors are allocated for the batch.
    """
    relative = np.broadcast_to(np.eye(4), source_poses.shape).copy()
    relative[:, :3, :3] = np.einsum('qji,qjk->qik', target_poses[:, :3, :3], source_poses[:, :3, :3])
    relative[:, :3, 3] = np.einsum('qji,qj->qi', target_poses[:, :3, :3],
        source_poses[:, :3, 3]-target_poses[:, :3, 3])
    row_regions, region_planes, normals, bounds = [0], [0], [], []
    for pose, permission in zip(source_poses, permissions):
        regions = () if permission is None else permission.regions
        for first, second in regions:
            n, b = np.vstack((first[0], second[0])), np.r_[first[1], second[1]]
            normals.extend(n @ pose[:3, :3])
            bounds.extend(b-n @ pose[:3, 3])
            region_planes.append(len(bounds))
        row_regions.append(len(region_planes)-1)
    def array(value, torch_dtype, warp_dtype):
        # BEGIN enqueue the small input copies on the same stream as the query.
        # The result owner performs the sole host wait after all pair groups.
        tensor = torch.as_tensor(np.asarray(value), dtype=torch_dtype).contiguous().to(
            points.device, non_blocking=True)
        # END a per-group blocking upload would serialize preceding kernels.
        return wp.from_torch(tensor, dtype=warp_dtype)
    return [wp.from_torch(points, dtype=wp.vec3d), array(relative, torch.float64, wp.mat44d),
        array(row_regions, torch.int32, wp.int32), array(region_planes, torch.int32, wp.int32),
        array(np.asarray(normals).reshape(-1, 3), torch.float64, wp.vec3d),
        array(bounds, torch.float64, wp.float64)]


def submit_pair_maxima(target, points, source_poses, target_poses, permissions, scale):
    """Enqueue finite maxima, winning indices and support counts on the device.

    Each fixed source coordinate is queried once with the original permission
    and target definitions. Block reductions retain first-index tie semantics.
    Only small block results enter the final Torch reduction. The caller owns
    the packed device result and combines groups before the host transfer.
    Unknown field locations contribute to their own count, not to a maximum.
    """
    import coal
    inputs = _pair_inputs(points, source_poses, target_poses, permissions)
    # Warp executes one SIMT lane on CPU; CUDA uses the declared 256-thread block.
    block_width = QUERY_BLOCK_SIZE if points.is_cuda else 1
    inputs.insert(0, block_width)
    shape = (len(source_poses), (len(points)+block_width-1)//block_width)
    maxima = torch.empty(shape, dtype=torch.float64, device=points.device)
    winners = torch.empty(shape, dtype=torch.int32, device=points.device)
    counts = torch.empty_like(winners)
    unknowns = torch.empty_like(winners)
    if isinstance(target, DeviceField):
        field = target.field
        kernel = _field_pair_tiles
        inputs += [target.values, target.valid, wp.vec3d(*field.origin), field.voxel_size, scale]
    else:
        geometry = target.geometry
        if target.mesh is not None:
            kind, dimensions, mesh, distance = 0, np.zeros(3), target.mesh.id, target.search_distance
        elif isinstance(geometry, coal.Box):
            kind, dimensions, mesh, distance = 1, geometry.halfSide, 0, 0.
        else:
            kind, dimensions, mesh, distance = 2, [geometry.radius, geometry.halfLength, 0.], 0, 0.
        kernel = _native_pair_tiles
        inputs += [mesh, kind, wp.vec3d(*dimensions), wp.vec3d(*target.bounds[0]),
            wp.vec3d(*target.bounds[1]), scale, distance]
    inputs += [wp.from_torch(array) for array in (maxima, winners, counts, unknowns)]
    wp.launch_tiled(kernel, dim=shape, block_dim=block_width, inputs=inputs, device=target.device,
        stream=wp.stream_from_torch(points.device) if points.is_cuda else None)
    maximum, blocks = maxima.max(dim=1)
    indices = winners.gather(1, blocks[:, None]).ravel()
    # BEGIN keep all group results resident until the point's common host wait.
    # Counts and indices fit exactly in float64, alongside the scalar maximum.
    return torch.stack((maximum, indices, counts.sum(1), unknowns.sum(1)), dim=1)
    # END no device-to-host copy occurs inside a submitted pair query.


class DeviceField:
    """Keep an original field's scalar samples and supported cells resident."""

    def __init__(self, field, device):
        self.device, self.field = device, field
        self.values = wp.array(field.values, dtype=wp.float32, device=device)
        self.valid = wp.array(field.valid_cells, dtype=wp.bool, device=device)

    def sample(self, points):
        """Return tensor values and support for an arbitrary leading layout."""
        layout = points.shape[:-1]
        points = points.reshape(-1, 3).contiguous()
        result = torch.empty(len(points), dtype=torch.float64, device=points.device)
        supported = torch.empty(len(points), dtype=torch.bool, device=points.device)
        wp.launch(_field_values, dim=len(points), inputs=[wp.from_torch(points, dtype=wp.vec3d),
            self.values, self.valid, wp.vec3d(*self.field.origin), self.field.voxel_size,
            wp.from_torch(result), wp.from_torch(supported)], device=self.device,
            stream=wp.stream_from_torch(points.device) if points.is_cuda else None)
        return result.reshape(layout), supported.reshape(layout)
# END original-field values and winning derivatives share the same coordinates.
