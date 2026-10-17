"""Original NumPy camera arithmetic followed by raw full-depth CUDA support.

This target-builder operation consumes DenseProjector's existing device
observations. It deliberately returns positional support only: entity labels
and query-normal agreement do not restrict the complete-depth surface domain.
The candidate projector retains its distinct oriented/entity support interface.
"""
from __future__ import annotations

import numpy as np
import torch


def dot3_einsum(a, b):
    """Match the installed NumPy three-term einsum: (p0+p2)+p1.

    The original geometry uses einsum for dot products, whereas its squared
    norms use add.reduce in coordinate order. These are different FP64
    operations at a boundary. Separate eager Torch multiply/add operations
    prevent contraction into a fused multiply-add with a different rounding.
    """
    product = a*b
    return (product[..., 0]+product[..., 2])+product[..., 1]


def norm3_numpy(a):
    """Match NumPy's three-coordinate square/add.reduce/sqrt operation."""
    squared = a*a
    return torch.sqrt((squared[..., 0]+squared[..., 1])+squared[..., 2])


def camera_pixels_numpy(points, K, T, image_shape):
    """Preserve the original camera/UV operation, strides and 8192-row blocks.

    BLAS multiplication can round differently between NumPy and CUDA, and
    between GEMV/GEMM row counts. A one-ulp UV difference can change rint at a
    half pixel and select another measured plane. This explicit CPU boundary
    preserves the original arithmetic for every point; it is not a heuristic
    ambiguous-point test or a retry after a CUDA failure.
    """
    # BEGIN exact original camera/UV arithmetic, before all GPU support gates.
    height, width = image_shape
    pixels = np.full((len(points), 2), np.nan)
    front = np.empty(len(points), dtype=bool)
    for begin in range(0, len(points), 8192):
        stop = min(begin+8192, len(points))
        camera = (points[begin:stop]-T[:3, 3]) @ T[:3, :3]
        homogeneous = camera @ K.T
        finite = np.isfinite(homogeneous).all(1) & (homogeneous[:, 2] != 0)
        uv = np.full((stop-begin, 2), np.nan)
        uv[finite] = homogeneous[finite, :2]/homogeneous[finite, 2, None]
        rounded = np.rint(uv)
        numeric = 64*np.finfo(float).eps*max(height, width)
        uv = np.where(np.abs(uv-rounded) <= numeric, rounded, uv)
        pixels[begin:stop] = uv
        front[begin:stop] = finite & (camera[:, 2] > 0)
    return pixels, front
    # END the original per-call row order and block boundaries are unchanged.


def project_positions(projector, points, view, uv, front):
    """Run the original raw support predicates using a device surface owner.

    The caller supplies UV/front from the original NumPy camera operation.
    DenseProjector supplies pixel lookup, padded observations and sentinel
    conversion. Owner observations bypass local
    plane tests. Every enclosing corner is checked, including zero-weight
    corners; no feature interpolation or oriented-entity predicate is used.
    """
    lookup = projector.lookup[view]
    height, width = lookup.shape
    # BEGIN UV/front are authoritative CPU camera outputs, never recomputed.
    inside = front & (uv >= 0).all(1) & (uv[:, 0] <= width-1) & (uv[:, 1] <= height-1)
    # END all remaining pixel and physical-support work stays on the device.
    # Only the indexing operands are made safe. Original UV and support masks
    # retain missing, out-of-frame and invalid-projection behavior.
    safe_uv = torch.nan_to_num(uv, nan=0., posinf=0., neginf=0.)
    safe_uv = torch.stack((safe_uv[:, 0].clamp(0, width-1),
                          safe_uv[:, 1].clamp(0, height-1)), dim=1)
    nearest_xy = torch.round(safe_uv).to(torch.int64)
    nearest = lookup[nearest_xy[:, 1], nearest_xy[:, 0]]
    present = inside & (nearest >= 0)
    ids = torch.where(present, nearest, -1)
    near = projector.observation_ids(ids)
    reference = projector.observed[near]
    delta = points-reference
    exact = present & (torch.abs(delta) <= (
        64*np.finfo(np.float64).eps*torch.maximum(torch.ones_like(reference), reference.abs()))).all(1)
    low, high = torch.floor(safe_uv).to(torch.int64), torch.ceil(safe_uv).to(torch.int64)
    corners = torch.stack((low, torch.stack((high[:, 0], low[:, 1]), dim=1),
                           torch.stack((low[:, 0], high[:, 1]), dim=1), high), dim=1)
    corner_ids = lookup[corners[:, :, 1], corners[:, :, 0]]
    safe_corners = projector.observation_ids(corner_ids)
    active = present & ~exact & (corner_ids >= 0).all(1)
    active &= projector.valid[near] & projector.valid[safe_corners].all(1)
    normal = projector.normals[near]
    thickness = projector.factor*projector.spacing[near]
    corner_delta = projector.observed[safe_corners]-reference[:, None]
    same = (dot3_einsum(corner_delta, normal[:, None]).abs() <= thickness[:, None]).all(1)
    same &= (dot3_einsum(projector.normals[safe_corners], normal[:, None]) >= projector.agreement).all(1)
    in_slab = dot3_einsum(delta, normal).abs() <= thickness
    local = norm3_numpy(delta) <= projector.radius
    return uv, exact | (active & same & in_slab & local)
