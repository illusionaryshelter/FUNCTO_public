"""Export validity from the exact snapshot Mapper's native ESDF sign source.

This module is loaded only while building a native map. It reuses the native
hash lookup and weight threshold; no alternate TSDF decoder or signed field
is introduced. Static history is excluded by the RGB-D snapshot contract.
"""
import math

import numpy as np
import torch
import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from .voxel_layers import MAX_FINE_CELLS


def _source_kernel(lookup):
    @wp.kernel
    def kernel(tsdf: BlockSparseTSDFWarp, site: wp.array(dtype=wp.int32),
               center: wp.array(dtype=wp.float32), size: wp.float32,
               minimum_weight: wp.float32, ny: wp.int32, nz: wp.int32,
               bits: wp.array(dtype=wp.uint8)):
        i = wp.tid()
        z = i % nz
        y = (i // nz) % ny
        x = i // (ny*nz)
        sdf = lookup(tsdf, x, y, z, center, size, minimum_weight)
        value = wp.int32(0)
        if site[i] >= 0:
            value = value | 1
        if sdf < 1e9:
            value = value | 2
            if sdf < 0.0:
                value = value | 4
        bits[i] = wp.uint8(value)
    return kernel


# BEGIN source sign provenance is limited to the original physical TSDF box.
def _mask_source_sign_bounds(bits, *, source_center, source_shape_xyz, source_voxel_size,
                             query_center, query_voxel_size):
    """Clear only sign-source bits outside the source grid, in place.

    The native lookup truncates negative continuous indices toward zero;
    an ESDF centre just below the TSDF box can therefore borrow voxel zero's
    sign. A larger distance-query grid must not turn that alias into an
    observation at the queried position. Its propagated nearest-site bit
    remains valid and is retained, as are all native distance values.

    XYZ source counts and centres describe physical voxel boxes. The exported
    ESDF shape/centre/spacing independently describe its actual sample lattice.
    Per-axis slices avoid allocating a full coordinate inventory or 3-D mask.
    """
    source_center = np.asarray(source_center, dtype=float)
    source_shape_xyz = np.asarray(source_shape_xyz)
    query_center = np.asarray(query_center, dtype=float)
    if (bits.ndim != 3 or bits.dtype != np.uint8 or min(bits.shape) < 1
            or source_center.shape != (3,) or query_center.shape != (3,)
            or source_shape_xyz.shape != (3,) or source_shape_xyz.dtype.kind not in 'iu'
            or np.any(source_shape_xyz <= 0)
            or not np.isfinite([*source_center, *query_center, source_voxel_size, query_voxel_size]).all()
            or source_voxel_size <= 0 or query_voxel_size <= 0):
        raise ValueError('Source-sign masking needs finite source/query grids and XYZ uint8 bits')
    half_extent = source_shape_xyz * float(source_voxel_size) / 2
    lower, upper = source_center-half_extent, source_center+half_extent
    for axis, count in enumerate(bits.shape):
        centres = query_center[axis]+(np.arange(count)-(count-1)/2)*float(query_voxel_size)
        # Native TSDF voxel indices include the lower boundary but exclude
        # the upper one. Clear both observed-sign and negative-sign flags;
        # nearest-site provenance is independent of local sign support.
        outside = (centres < lower[axis]) | (centres >= upper[axis])
        selection = [slice(None)] * 3
        selection[axis] = outside
        bits[tuple(selection)] &= np.uint8(0xff ^ (2 | 4))
    return bits
# END numerical query support does not add TSDF observations or free space.


def export_source_bits(mapper, config, voxel, *, max_cells=MAX_FINE_CELLS):
    """Return XYZ bits: nearest site (1), observed sign (2), negative sign (4)."""
    if config.enable_static:
        raise ValueError("Snapshot source validity does not admit static map history")
    shape = tuple(voxel.feature_tensor.shape)
    # BEGIN share the byte-admitted local inventory bound with its consumers.
    if len(shape) != 3 or type(max_cells) is not int or max_cells <= 0 or math.prod(shape) > max_cells:
        raise ValueError("Fine source export requires a bounded complete XYZ grid")
    # END a larger inventory never skips mapping's independent memory check.
    device = voxel.feature_tensor.device
    bits = torch.empty(math.prod(shape), dtype=torch.uint8, device=device)
    center = torch.as_tensor(voxel.pose[:3], dtype=torch.float32, device=device)
    # The current Mapper has no public site-index accessor. Its completed
    # integrator owns this exact field; export before it is released. The
    # same installed lookup supplies the ESDF kernel's dynamic sign branch.
    site = mapper.integrator._site_index.reshape(-1)
    if site.numel() != bits.numel():
        raise ValueError("Source indices must match the completed native field")
    wp.launch(_source_kernel(mapper.tsdf.kernels.lookup_combined_sdf_at_esdf_coords),
              dim=bits.numel(), inputs=[mapper.tsdf.get_warp_data(), wp.from_torch(site),
              wp.from_torch(center), float(voxel.voxel_size), float(config.minimum_tsdf_weight),
              shape[1], shape[2], wp.from_torch(bits)], device=str(device),
              stream=wp.stream_from_torch(torch.cuda.current_stream(device)))
    # BEGIN constrain native sign provenance using the actual two grid frames.
    exported = np.ascontiguousarray(bits.cpu().numpy().reshape(shape))
    return _mask_source_sign_bounds(exported,
        source_center=config.grid_center.detach().cpu().numpy(),
        source_shape_xyz=config.grid_shape[::-1],
        source_voxel_size=float(np.float32(config.voxel_size)),
        query_center=voxel.pose[:3], query_voxel_size=float(voxel.voxel_size))
    # END the original native nearest-site bits and distance output are unchanged.
