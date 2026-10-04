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


def export_source_bits(mapper, config, voxel):
    """Return XYZ bits: nearest site (1), observed sign (2), negative sign (4)."""
    if config.enable_static:
        raise ValueError("Snapshot source validity does not admit static map history")
    shape = tuple(voxel.feature_tensor.shape)
    if len(shape) != 3 or math.prod(shape) > 1_000_000:
        raise ValueError("Fine source export requires a bounded complete XYZ grid")
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
    return np.ascontiguousarray(bits.cpu().numpy().reshape(shape))
