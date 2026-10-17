"""Keep the reviewed image-grid Jacobi geometry on the CUDA device.

The numerical statements and 32768-row schedule match the source geometry module.
Only target-construction outputs are retained: the wanted normal is zero,
so its unused agreement/eligibility outputs need not be computed or downloaded.
Input validation and the cuSOLVER per-matrix info check are deliberately kept.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from . import cusolver_jacobi
from .image_geometry import _image_inputs, _window_ids


def prepare():
    """Resolve the existing extension before the caller's measured interval."""
    return cusolver_jacobi.load_extension()


def target_geometry(points, pixels, camera_origin, radius_m, config, *,
                    image_shape, xyz, xy, stats=None):
    """Return normal, spacing, target-validity and viewing cosine on CUDA.

    ``xyz`` and ``xy`` are caller-owned device copies of these same validated
    CPU observations. This private entry is used only by the dual-view target constructor;
    it neither uploads a second copy nor stores a tensor outside the caller.
    """
    started = perf_counter()
    points, pixels, _, origin, shape, _ = _image_inputs(
        points, pixels, np.zeros(3), camera_origin, image_shape)
    if not xyz.is_cuda or not xy.is_cuda or xyz.dtype != torch.float64 or xy.dtype != torch.int64:
        raise ValueError('Target image geometry requires FP64 CUDA coordinates and int64 pixels')
    if tuple(xyz.shape) != points.shape or tuple(xy.shape) != pixels.shape:
        raise ValueError('Device observations must align with the validated CPU input')
    n = len(points)
    if not n:
        return xyz.clone(), torch.empty(0, dtype=torch.float64, device=xyz.device), \
            torch.empty(0, dtype=torch.bool, device=xyz.device), \
            torch.empty(0, dtype=torch.float64, device=xyz.device)
    pixel_radius = int(config.normal_pixel_radius)
    window = (2*pixel_radius+1)**2
    count, batch_size = min(window, n), 32768
    explicit_bound = shape[0]*shape[1]*8 + n*128 + min(n, batch_size)*window*128
    if explicit_bound > 6*1024**3:
        raise MemoryError('Image map and bounded target geometry exceed the explicit buffer budget')
    device = xyz.device
    origin_gpu = torch.tensor(origin, dtype=torch.float64, device=device)
    index_map = torch.full((shape[0]*shape[1],), -1, dtype=torch.int64, device=device)
    index_map[xy[:, 1]*shape[1]+xy[:, 0]] = torch.arange(n, device=device)
    delta = torch.arange(-pixel_radius, pixel_radius+1, dtype=torch.int64, device=device)
    dy, dx = torch.meshgrid(delta, delta, indexing='ij')
    offsets = torch.stack((dx.flatten(), dy.flatten()), dim=1)
    normals = torch.empty_like(xyz)
    spacings = torch.empty(n, dtype=torch.float64, device=device)
    view_cosines = torch.empty_like(spacings)
    valid = torch.empty(n, dtype=torch.bool, device=device)
    tiny = np.finfo(float).tiny
    with torch.inference_mode():
        for begin in range(0, n, batch_size):
            end = min(begin+batch_size, n)
            query = xyz[begin:end]
            ids = _window_ids(index_map, xy[begin:end], offsets, shape, count)
            present = ids >= 0
            neighbors = xyz[torch.where(present, ids, n-1)]
            distances = torch.linalg.vector_norm(neighbors-query[:, None], dim=2)
            usable = present & (distances <= radius_m*(1+1e-12))
            sizes = usable.sum(1)
            means = (neighbors*usable[..., None]).sum(1)/sizes[:, None].clamp_min(1)
            centered = (neighbors-means[:, None])*usable[..., None]
            covariance = torch.einsum('nki,nkj->nij', centered, centered)
            eigenvalues, eigenvectors = cusolver_jacobi.eigh(covariance, UPLO='L')
            eigenvalues = eigenvalues.clamp_min(0)
            planar = (sizes >= config.min_neighborhood_points) & (
                eigenvalues[:, 1] > config.rank_relative_tolerance**2*eigenvalues[:, 2])
            planar &= eigenvalues[:, 2] > tiny
            ratio = torch.full((len(query),), float('inf'), dtype=torch.float64, device=device)
            ratio[planar] = torch.sqrt(eigenvalues[planar, 0]/eigenvalues[planar, 1])
            normal = eigenvectors[:, :, 0]
            view = origin_gpu-query
            view_length = torch.linalg.vector_norm(view, dim=1)
            signed_view = torch.einsum('ij,ij->i', normal, view)/view_length.clamp_min(tiny)
            normal = torch.where((signed_view < 0)[:, None], -normal, normal)
            view_cosine = signed_view.abs().clamp(0, 1)
            positive = torch.sort(torch.where(usable & (distances > 0), distances, float('inf')), dim=1).values
            length, middle = min(4, count), min(4, count)//2
            spacing = (positive[:, middle-1]+positive[:, middle])/2 if length%2 == 0 else positive[:, middle]
            spacing_valid = torch.isfinite(spacing) & (spacing > 0)
            spacing = torch.where(spacing_valid, spacing, 0.)
            # build_target_surface ignores the wanted-normal eligibility output.
            # Its own target-plane predicate is kept verbatim in device form.
            good = torch.isfinite(ratio) & (ratio <= config.max_planarity_ratio)
            good &= torch.isfinite(spacing) & (spacing > 0) & (view_cosine >= config.min_view_cosine)
            good &= view_length > 0
            normals[begin:end], spacings[begin:end] = normal, spacing
            view_cosines[begin:end], valid[begin:end] = view_cosine, good
    if stats is not None:
        torch.cuda.synchronize(device)
        stats.append(dict(point_count=n, image_shape=list(shape), numerical_batch=batch_size,
            seconds=perf_counter()-started, host_geometry_outputs=0,
            solver='existing_checked_cusolver_jacobi',
            solver_info_downloads=(n+batch_size-1)//batch_size))
    return normals, spacings, valid, view_cosines
