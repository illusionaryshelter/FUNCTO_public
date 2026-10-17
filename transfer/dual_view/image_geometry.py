"""CUDA local geometry on complete source depth with unique integer pixels.

The selected row-major neighborhood order and 32768-row Jacobi schedule are
fixed. Input validation, uploads, finite physical masks, solver checks and CPU
outputs remain inside each call. No failed CUDA computation is replaced.
"""
from time import perf_counter

from . import cusolver_jacobi

OUTPUT_NAMES = ('normals', 'ratios', 'agreements', 'view_cosines', 'spacings', 'eligible')


def _image_inputs(points, pixels, wanted_normal, camera_origin, image_shape):
    """Validate once on CPU; duplicate pixels cannot silently overwrite a point.

    A missing image shape uses the rectangle from (0,0) to the largest observed
    coordinate. Omitting outer unobserved rows/columns cannot remove a neighbor.
    This rule is confined to this image interface, not arbitrary point clouds.
    """
    import numpy as np
    points, pixels = np.asarray(points), np.asarray(pixels)
    wanted, origin = np.asarray(wanted_normal, dtype=np.float64), np.asarray(camera_origin, dtype=np.float64)
    if points.dtype != np.float64 or points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError('Image geometry requires FP64 points shaped [N,3]')
    if pixels.shape != (len(points), 2) or pixels.dtype.kind not in 'iu':
        raise ValueError('Image geometry requires point-aligned integer pixels [N,2]')
    if wanted.shape != (3,) or origin.shape != (3,) or not all(np.isfinite(a).all() for a in (points, wanted, origin)):
        raise ValueError('Geometric coordinates and direction inputs must be finite')
    if np.any(pixels < 0):
        raise ValueError('Image pixel coordinates must be nonnegative')
    if image_shape is None:
        shape = (int(pixels[:, 1].max())+1, int(pixels[:, 0].max())+1) if len(points) else (0, 0)
    else:
        shape = tuple(image_shape)
        if len(shape) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in shape):
            raise ValueError('image_shape must contain two integer dimensions')
        shape = tuple(map(int, shape))
    height, width = shape
    if min(shape) < 0 or (len(points) and min(shape) < 1):
        raise ValueError('Nonempty observed pixels require positive image dimensions')
    if height*width > np.iinfo(np.int64).max or height*width*8 > 6*1024**3:
        raise MemoryError('Pixel index map exceeds the image geometry memory budget')
    if len(points) and (np.any(pixels[:, 0] >= width) or np.any(pixels[:, 1] >= height)):
        raise ValueError('Observed pixels lie outside image_shape')
    pixels = pixels.astype(np.int64, copy=False)
    linear = pixels[:, 1]*width+pixels[:, 0]
    if np.unique(linear).size != len(points):
        raise ValueError('Image geometry requires one unique observation per pixel')
    return points, pixels, wanted, origin, shape, linear


def _window_ids(index_map, query_pixels, offsets, shape, count):
    """Return original point IDs or -1, with the original k=min(window,N).

    Every valid integer pixel occurs once in the window. For N smaller than the
    window capacity, compact present positions before retaining N slots; this
    cannot truncate a present point because the entire image has only N points.
    Missing positions remain explicit rather than acquiring clamped neighbors.
    """
    import torch
    height, width = shape
    x = query_pixels[:, None, 0]+offsets[None, :, 0]
    y = query_pixels[:, None, 1]+offsets[None, :, 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    ids = index_map[y.clamp(0, height-1)*width+x.clamp(0, width-1)]
    ids = torch.where(inside, ids, -1)
    if count < len(offsets):
        slots = torch.arange(len(offsets), device=ids.device)[None].expand_as(ids)
        positions = torch.sort(torch.where(ids >= 0, slots, len(offsets)), dim=1).values[:, :count]
        chosen = ids.gather(1, positions.clamp_max(len(offsets)-1))
        ids = torch.where(positions < len(offsets), chosen, -1)
    return ids


def image_surface_geometry(points, pixels, wanted_normal, camera_origin,
                           sampling_radius_m, config, *, image_shape=None,
                           stats=None):
    """Compute all six original fields and return actual device results to CPU.

    CPU validation, uploads, map construction, neighborhood gathering, FP64
    arithmetic and downloads are all inside this call. Only neighborhood order
    and numerical implementation change. The physical-radius mask, holes,
    degeneracy tests, camera-facing sign and even-length median rule remain.
    No device failure is replaced by a CPU answer.
    """
    import numpy as np
    import torch
    from transfer.core import TransferConfig

    device = torch.device('cuda')
    gpu_batch_size = 32768
    if not isinstance(config, TransferConfig):
        raise ValueError('config must be TransferConfig')
    # BEGIN fixed selected numerical schedule: changing this can alter FP64 output.
    # The runtime memory cap remains active for both buffers and solver workspace.
    # END no alternative solver or numerical batch is selected at runtime.
    if isinstance(sampling_radius_m, (bool, np.bool_)) or not np.isscalar(sampling_radius_m) or not np.isfinite(sampling_radius_m) or sampling_radius_m <= 0:
        raise ValueError('Physical sampling radius must be finite and positive')
    def synchronize():
        torch.cuda.synchronize(device)
    synchronize()
    wall = perf_counter()
    start = perf_counter()
    points, pixels, wanted, origin, shape, linear = _image_inputs(points, pixels, wanted_normal, camera_origin, image_shape)
    validation_seconds = perf_counter()-start
    if not len(points):
        result = (np.zeros_like(points), *(np.zeros(0) for _ in range(4)), np.zeros(0, bool))
        if stats is not None:
            stats.append(dict(point_count=0, image_shape=list(shape), validation_seconds=validation_seconds,
                              wall_seconds=perf_counter()-wall))
        return result
    # BEGIN bound parameter arithmetic before allocation without NumPy overflow.
    radius = int(config.normal_pixel_radius)
    # END Python integer arithmetic preserves the same ordinary window radius.
    window = (2*radius+1)**2
    count = min(window, len(points))
    batch = min(gpu_batch_size, len(points))
    # This checks explicit allocations before forming a potentially enormous
    # parameterized window. CUDA library workspace is separately capped by the
    # caller's allocator limit and observed, not guessed by this estimate.
    explicit_bound = shape[0]*shape[1]*8 + points.nbytes + pixels.nbytes + len(points)*57 + batch*window*128
    if explicit_bound > 6*1024**3:
        raise MemoryError('Image map and bounded window exceed the explicit buffer budget')

    start = perf_counter()
    xyz = torch.tensor(points, dtype=torch.float64, device=device)
    xy = torch.tensor(pixels, dtype=torch.int64, device=device)
    wanted_gpu = torch.tensor(wanted, dtype=torch.float64, device=device)
    origin_gpu = torch.tensor(origin, dtype=torch.float64, device=device)
    synchronize()
    upload_seconds = perf_counter()-start
    start = perf_counter()
    index_map = torch.full((shape[0]*shape[1],), -1, dtype=torch.int64, device=device)
    index_map[xy[:, 1]*shape[1]+xy[:, 0]] = torch.arange(len(points), device=device)
    delta = torch.arange(-radius, radius+1, dtype=torch.int64, device=device)
    dy, dx = torch.meshgrid(delta, delta, indexing='ij')
    offsets = torch.stack((dx.flatten(), dy.flatten()), dim=1)
    synchronize()
    map_seconds = perf_counter()-start
    start = perf_counter()
    normals = torch.empty_like(xyz)
    ratios = torch.empty(len(points), dtype=torch.float64, device=device)
    agreements, view_cosines, spacings = (torch.empty_like(ratios) for _ in range(3))
    eligible = torch.empty(len(points), dtype=torch.bool, device=device)
    tiny = np.finfo(float).tiny
    with torch.inference_mode():
        for begin in range(0, len(points), gpu_batch_size):
            end = min(begin+gpu_batch_size, len(points))
            query = xyz[begin:end]
            ids = _window_ids(index_map, xy[begin:end], offsets, shape, count)
            present = ids >= 0
            # Match the old missing-index sentinel's last-point substitution;
            # the independent present mask still removes every missing slot.
            neighbors = xyz[torch.where(present, ids, len(points)-1)]
            distances = torch.linalg.vector_norm(neighbors-query[:, None], dim=2)
            usable = present & (distances <= sampling_radius_m*(1+1e-12))
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
            agreement = (normal @ wanted_gpu).clamp(-1, 1)
            positive = torch.sort(torch.where(usable & (distances > 0), distances, float('inf')), dim=1).values
            length, middle = min(4, count), min(4, count)//2
            spacing = (positive[:, middle-1]+positive[:, middle])/2 if length%2 == 0 else positive[:, middle]
            spacing_valid = torch.isfinite(spacing) & (spacing > 0)
            spacing = torch.where(spacing_valid, spacing, 0.)
            good = planar & spacing_valid & (ratio <= config.max_planarity_ratio)
            good &= agreement >= config.normal_min_agreement
            good &= (view_cosine >= config.min_view_cosine) & (view_length > 0)
            normals[begin:end], ratios[begin:end] = normal, ratio
            agreements[begin:end], view_cosines[begin:end] = agreement, view_cosine
            spacings[begin:end], eligible[begin:end] = spacing, good
    synchronize()
    compute_seconds = perf_counter()-start
    start = perf_counter()
    result = tuple(t.cpu().numpy() for t in (normals, ratios, agreements, view_cosines, spacings, eligible))
    synchronize()
    download_seconds = perf_counter()-start
    if stats is not None:
        stats.append(dict(point_count=len(points), image_shape=list(shape), normal_pixel_radius=int(radius),
            neighbors_per_point=count, numerical_batch=int(gpu_batch_size),
            neighborhood_order='row_major_parameterized_window', validation_seconds=validation_seconds,
            upload_seconds=upload_seconds, map_seconds=map_seconds, compute_seconds=compute_seconds,
            download_seconds=download_seconds, wall_seconds=perf_counter()-wall,
            uploaded_input_bytes=int(points.nbytes+pixels.nbytes+wanted.nbytes+origin.nbytes),
            downloaded_output_bytes=sum(a.nbytes for a in result), index_map_bytes=shape[0]*shape[1]*8))
    return result


def surface_geometry(points, pixels, wanted_normal, camera_origin,
                     sampling_radius_m, config, *, stats=None):
    """Source-plane entry with the exact selected tight image boundary.

    The complete source-depth points already contain every observed pixel.
    Omitting empty outer image rows cannot remove a neighbor; max+1 retains
    the original selected map shape and all 32768-row numerical boundaries.
    """
    shape = ((int(pixels[:, 1].max())+1, int(pixels[:, 0].max())+1)
             if len(pixels) else (0, 0))
    return image_surface_geometry(points, pixels, wanted_normal, camera_origin,
        sampling_radius_m, config, image_shape=shape, stats=stats)
