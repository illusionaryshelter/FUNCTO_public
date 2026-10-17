"""Device-resident target construction with an explicit exact CPU search boundary.

The input conversion, SciPy mutual-nearest association and NumPy camera/UV
arithmetic remain explicit CPU work, never a fallback from CUDA failure. Local
image geometry, pair gates, full-depth support, element mapping and pixel
adjacency run on CUDA. The finished ordinary surface is exported to CPU.

The dictionary retains the dual-view surface contract. A joined CPU worker
performs exact mutual-nearest search during device geometry; every exit joins
that worker, and transient device ownership ends before return.
"""
from __future__ import annotations

from time import perf_counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import torch

from transfer.core import TransferConfig, _sampling_parameters
from transfer import multiview_geometry as original

from .candidate_batch import DenseProjector
from . import surface_geometry
from .surface_projection import camera_pixels_numpy, dot3_einsum, norm3_numpy, project_positions


def prepare():
    """Load the existing Jacobi binding before starting a measured build."""
    return surface_geometry.prepare()


class _DeviceProjector(DenseProjector):
    """Own device geometry and account for the explicit CPU camera boundary."""

    # BEGIN explicit camera/UV boundary accounting; no answer cache is retained.
    def __init__(self, surface, *, device='cuda'):
        # Original calibration arrays may have Fortran-derived slice strides.
        # Their GPU copies are contiguous, so downloading those copies would
        # change the BLAS input layout. Retain the actual CPU input references.
        if not all(isinstance(surface[key], np.ndarray)
                   for key in ('K', 'T_world_camera_cv')):
            raise TypeError('Target projector requires original NumPy calibration arrays')
        self.camera_K = surface['K']
        self.camera_T = surface['T_world_camera_cv']
        super().__init__(surface, device=device)
        self.camera_stats = dict(calls=0, point_count=0, device_to_host_bytes=0,
            host_to_device_bytes=0, device_to_host_seconds=0.,
            numpy_camera_seconds=0., host_to_device_seconds=0.)
    # END the statistics describe work in the enclosing measured build.

    def tensor(self, value, dtype):
        if isinstance(value, torch.Tensor):
            if value.device != torch.device(self.device):
                raise ValueError('One surface projector cannot mix CUDA devices')
            return value.to(dtype=dtype)
        return super().tensor(value, dtype)


def _positions(projector, points, view):
    """Return raw positional support, not the oriented/entity support field.

    The actual point rows are downloaded once; the original CPU calibration
    arrays retain their own strides. NumPy computes camera/UV in original
    8192-row blocks; only UV/front return to CUDA. This explicit boundary avoids
    relying on CUDA matrix products at pixel-rounding discontinuities.
    """
    # BEGIN original CPU camera arithmetic and complete transfer accounting.
    if points.is_cuda:
        torch.cuda.synchronize(points.device)
    started = perf_counter()
    points_cpu = points.cpu().numpy()
    K, T = projector.camera_K[view], projector.camera_T[view]
    report = projector.camera_stats
    report['calls'] += 1
    report['point_count'] += len(points)
    report['device_to_host_bytes'] += points_cpu.nbytes
    report['device_to_host_seconds'] += perf_counter()-started
    started = perf_counter()
    uv_cpu, front_cpu = camera_pixels_numpy(points_cpu, K, T, projector.lookup[view].shape)
    report['numpy_camera_seconds'] += perf_counter()-started
    started = perf_counter()
    uv = torch.as_tensor(uv_cpu, device=points.device)
    front = torch.as_tensor(front_cpu, device=points.device)
    if points.is_cuda:
        torch.cuda.synchronize(points.device)
    report['host_to_device_bytes'] += uv_cpu.nbytes+front_cpu.nbytes
    report['host_to_device_seconds'] += perf_counter()-started
    # END CUDA receives authoritative UV/front, not a different camera formula.
    support = torch.empty(len(points), dtype=torch.bool, device=points.device)
    for begin in range(0, len(points), 32768):
        end = min(begin+32768, len(points))
        # BEGIN preserve raw projection's three-term einsum boundary arithmetic.
        _, positional = project_positions(projector, points[begin:end], view,
            uv[begin:end], front[begin:end])
        support[begin:end] = positional
        # END no orientation/entity predicate is substituted for raw support.
    return uv, support


def _adjacency(surface, count, config):
    """Build the same eight-pixel-edge graph, coalesced and sorted on CUDA.

    Compatible edges are mapped through accepted binary pairs. This is graph
    construction, not connected-components computation or transitive welding.
    Encoding (row,column) into one int64 key preserves canonical CSR order.
    """
    xyz, normals = surface['observation_xyz'], surface['observation_normals']
    valid, spacing = surface['observation_valid'], surface['observation_spacing']
    mapping = surface['observation_to_element']
    device = xyz.device
    rows, columns = [], []
    for lookup in surface['pixel_to_observation']:
        height, width = lookup.shape
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            y0, y1 = max(0, -dy), min(height, height-dy)
            x0, x1 = max(0, -dx), min(width, width-dx)
            a = lookup[y0:y1, x0:x1].reshape(-1)
            b = lookup[y0+dy:y1+dy, x0+dx:x1+dx].reshape(-1)
            present = (a >= 0) & (b >= 0)
            a, b = a[present], b[present]
            delta = xyz[b]-xyz[a]
            thickness = config.surface_height_spacing_factor * torch.minimum(spacing[a], spacing[b])
            compatible = valid[a] & valid[b]
            # BEGIN distinguish original einsum (021) from norm reduction (012).
            compatible &= dot3_einsum(normals[a], normals[b]) >= config.normal_min_agreement
            compatible &= norm3_numpy(delta) <= surface['sampling_radius_m']
            compatible &= dot3_einsum(delta, normals[a]).abs() <= thickness
            compatible &= dot3_einsum(delta, normals[b]).abs() <= thickness
            # END a one-ulp boundary must not add or remove a graph edge.
            a, b = mapping[a[compatible]], mapping[b[compatible]]
            distinct = a != b
            rows.append(a[distinct]); columns.append(b[distinct])
    row, column = torch.cat(rows), torch.cat(columns)
    if count > 3037000499:
        raise MemoryError('Surface IDs cannot be encoded as bounded int64 edge keys')
    keys = torch.unique(torch.cat((row*count+column, column*count+row)), sorted=True)
    if count:
        row, column = keys//count, keys%count
        lengths = torch.bincount(row, minlength=count)
    else:
        column = torch.empty(0, dtype=torch.int64, device=device)
        lengths = torch.empty(0, dtype=torch.int64, device=device)
    indptr = torch.cat((torch.zeros(1, dtype=torch.int64, device=device), lengths.cumsum(0)))
    # SciPy's original small graph uses int32 indices/indptr. Convert on-device
    # before the export boundary so CSR packing does not scan/cast them again.
    index_dtype = torch.int32 if max(count, len(keys)) <= np.iinfo(np.int32).max else torch.int64
    return torch.ones(len(keys), dtype=torch.bool, device=device), column.to(index_dtype), indptr.to(index_dtype)


class _SearchTree:
    """Use six actual SciPy query workers within the eight-thread budget.

    The background coordinator waits for these six workers while the original
    main thread submits CUDA work: 6 + 1 + 1. Tree construction, query options,
    bounded-search arithmetic and unbounded tie resolution are unchanged.
    """

    def __init__(self, points):
        self.tree = cKDTree(points)

    def query(self, points, *args, **kwargs):
        kwargs['workers'] = 6
        return self.tree.query(points, *args, **kwargs)


def _declare_overlap(report):
    """Make concurrent CPU work and the remaining join wait unambiguous."""
    report['implementation'] = 'dual_view_surface'
    report['search_overlap'] = dict(query_workers=6, maximum_active_cpu_threads=8,
        submitted_to_thread=False, search_succeeded=False, cleanup_complete=False,
        submit_seconds=0., cpu_work_seconds=0., main_wait_seconds=0., cleanup_seconds=0.,
        timing_scope='CPU search overlaps main-thread upload/local geometry; CPU work and join wait must not be added to the complete wall clock')
    report['scipy_stage_timing_scope'] = (
        'stages_seconds.scipy_exact_mutual_nearest is remaining main-thread join wait and count handling; '
        'search_overlap.cpu_work_seconds is concurrent CPU work, including fresh trees')


def _search_task(xyz_cpu, offsets, sizes, radius, report, clock_origin):
    """Run the extracted original search without touching CUDA or input arrays."""
    started = perf_counter()
    report['cpu_started_at_seconds'] = started-clock_origin
    try:
        result = _compute_pairs(xyz_cpu, offsets, sizes, radius)
        report['search_succeeded'] = True
        return result
    finally:
        finished = perf_counter()
        report['cpu_work_seconds'] = finished-started
        report['cpu_finished_at_seconds'] = finished-clock_origin


@contextmanager
def _search_scope(xyz_cpu, offsets, sizes, radius, report, clock_origin):
    """Own the search thread through success, CPU failure and CUDA failure.

    An empty view takes the original no-search branch without creating a
    thread. Otherwise every exit joins the worker. A simultaneous main-thread
    and worker exception preserves the main exception and chains the worker
    exception as its cause; neither failure is silently replaced by an answer.
    """
    executor, future = None, None
    submitted = perf_counter()
    try:
        if np.all(sizes > 0):
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='surface-exact-nn')
            future = executor.submit(_search_task, xyz_cpu, offsets, sizes, radius, report, clock_origin)
            report['submitted_to_thread'] = True
        else:
            future = Future()
            future.set_result(_search_task(xyz_cpu, offsets, sizes, radius, report, clock_origin))
        report['submit_seconds'] = perf_counter()-submitted
        report['main_geometry_started_at_seconds'] = perf_counter()-clock_origin
        try:
            yield future
        except BaseException as main_error:
            try:
                future.result()
            except BaseException as worker_error:
                if worker_error is not main_error:
                    raise main_error from worker_error
            raise
        else:
            # Also observe a worker failure if a future caller exits without
            # consuming its result. The actual constructor consumes it once.
            future.result()
    finally:
        cleanup_started = perf_counter()
        if executor is not None:
            executor.shutdown(wait=True)
        report['cleanup_seconds'] = perf_counter()-cleanup_started
        report['cleanup_complete'] = True


def _wait_for_pairs(future, report, clock_origin):
    """Measure remaining wait, not the worker's already overlapping duration."""
    started = perf_counter()
    report['main_reached_join_at_seconds'] = started-clock_origin
    try:
        return future.result()
    finally:
        report['main_wait_seconds'] = perf_counter()-started
        # This interval measures CPU work concurrent with the main geometry
        # path, not CUDA device-busy time. GPU profiling is a separate task.
        report['cpu_work_during_main_geometry_interval_seconds'] = max(0.,
            min(report.get('cpu_finished_at_seconds', started-clock_origin), started-clock_origin)
            - max(report.get('cpu_started_at_seconds', started-clock_origin),
                  report['main_geometry_started_at_seconds']))


def _compute_pairs(xyz_cpu, offsets, sizes, sampling_radius_m):
    """Original physical-radius and SciPy tie decisions with fresh six-worker trees."""
    pair_a_cpu, pair_b_cpu = np.empty(0, np.int64), np.empty(0, np.int64)
    if np.all(sizes > 0):
        a, b = xyz_cpu[:offsets[1]], xyz_cpu[offsets[1]:]
        distance, nearest_b = original._nearest_within_radius(_SearchTree(b), a, sampling_radius_m)
        nearest_a = original._nearest_within_radius(_SearchTree(a), b, sampling_radius_m)[1]
        pair_a_cpu = np.flatnonzero((nearest_b < len(b)) & (distance <= sampling_radius_m))
        pair_a_cpu = pair_a_cpu[nearest_a[nearest_b[pair_a_cpu]] == pair_a_cpu]
        pair_b_cpu = offsets[1]+nearest_b[pair_a_cpu]
    return pair_a_cpu, pair_b_cpu


@torch.inference_mode()
def build_target_surface(views, sampling_radius_m=.03, config=None, stats=None):
    """Return every original CPU surface field after declared CUDA construction.

    Call prepare() once outside timing. The caller enforces the global 6 GiB /
    half-device limit and serializes GPU work. This function does not increase
    that limit, cache answers, inspect target truth, or retry on another backend.
    """
    started = perf_counter()
    # BEGIN distinguish GPU support work from the original CPU camera/UV step.
    report = dict(complete=False, implementation='dual_view_surface',
        cpu_stages=['original_input_conversion', 'scipy_exact_mutual_nearest',
                    'numpy_camera_projection', 'final_numpy_csr_export'],
        gpu_stages=['image_jacobi_geometry', 'pair_geometry_and_raw_support',
                    'element_mapping', 'observed_view_raw_support', 'pixel_adjacency'],
        projection_timing_scope='pair_geometry_and_raw_projection and observed_views include CPU camera/UV, transfers and GPU support; raw_projection_camera is an included subset, not additional time',
        numerical_batch=32768, camera_numerical_batch=8192,
        stages_seconds={}, local_geometry=[],
        cpu_search_fallback=False, device_buffers_returned=False)
    # END these work labels do not misclassify mixed wall-clock stages as GPU-only.
    _declare_overlap(report)
    if stats is not None:
        stats.clear()
        stats.update(report)
    stage = report['stages_seconds']

    def finish(name, begin, *, sync=True):
        if sync:
            torch.cuda.synchronize()
        stage[name] = perf_counter()-begin

    config = TransferConfig() if config is None else config
    if not isinstance(config, TransferConfig):
        raise ValueError('config must be TransferConfig')
    _sampling_parameters(sampling_radius_m, sampling_radius_m)
    views = list(views)
    if (len(views) != 2 or any(not isinstance(v.get('id'), str) or not v['id'] for v in views)
            or len({v['id'] for v in views}) != 2):
        raise ValueError('Require exactly two views with distinct nonempty string ids')
    views.sort(key=lambda view: view['id'])
    part = perf_counter()
    arrays = [original._view_arrays(view) for view in views]
    sizes = np.asarray([len(a[0]) for a in arrays], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(sizes)]
    xyz_cpu = np.concatenate([a[0] for a in arrays])
    pixels_cpu = np.concatenate([a[1] for a in arrays])
    entity_cpu = np.concatenate([a[2] for a in arrays])
    finish('original_input_conversion', part, sync=False)
    # The image normal workspace has a separate per-view guard. This bound
    # covers persistent observations, graph buffers and bounded projections.
    if len(xyz_cpu)*768 + sum(a[3].size for a in arrays)*8 > 6*1024**3:
        raise MemoryError('Target observation and graph payload exceeds the surface buffer budget')
    # BEGIN exact coordinate search overlaps the unchanged device construction.
    with _search_scope(xyz_cpu, offsets, sizes, sampling_radius_m,
                       report['search_overlap'], started) as search_future:
        part = perf_counter()
        device = torch.device('cuda', torch.cuda.current_device())
        xyz = torch.tensor(xyz_cpu, dtype=torch.float64, device=device)
        pixels = torch.tensor(pixels_cpu, dtype=torch.int64, device=device)
        entity = torch.tensor(entity_cpu, dtype=torch.bool, device=device)
        lookups = [torch.tensor(np.where(a[3] >= 0, a[3]+offsets[i], -1),
                               dtype=torch.int64, device=device) for i, a in enumerate(arrays)]
        finish('observation_upload', part)
        part = perf_counter()
        local = [surface_geometry.target_geometry(a[0], a[1], a[5][:3, 3],
            sampling_radius_m, config, image_shape=a[6].shape,
            xyz=xyz[offsets[i]:offsets[i+1]], xy=pixels[offsets[i]:offsets[i+1]],
            stats=report['local_geometry']) for i, a in enumerate(arrays)]
        normals, spacing, valid, cosine = (torch.cat([value[k] for value in local]) for k in range(4))
        del local
        observation_view = torch.cat([torch.full((int(size),), i, dtype=torch.int64, device=device)
                                      for i, size in enumerate(sizes)])
        surface = dict(observation_xyz=xyz, observation_pixels=pixels,
            observation_view_index=observation_view, observation_normals=normals,
            observation_spacing=spacing, observation_valid=valid,
            observation_entity_member=entity, pixel_to_observation=lookups,
            view_offsets=offsets, view_ids=[v['id'] for v in views],
            K=np.stack([a[4] for a in arrays]), T_world_camera_cv=np.stack([a[5] for a in arrays]),
            depth_m=[a[6] for a in arrays], sampling_radius_m=float(sampling_radius_m),
            surface_height_spacing_factor=float(config.surface_height_spacing_factor),
            normal_min_agreement=float(config.normal_min_agreement))
        finish('image_jacobi_geometry', part)
        part = perf_counter()
        pair_a_cpu, pair_b_cpu = _wait_for_pairs(search_future, report['search_overlap'], started)
        report['mutual_nearest_pair_count'] = len(pair_a_cpu)
        finish('scipy_exact_mutual_nearest', part, sync=False)
        part = perf_counter()
        pair_a = torch.tensor(pair_a_cpu, dtype=torch.int64, device=device)
        pair_b = torch.tensor(pair_b_cpu, dtype=torch.int64, device=device)
        projector = _DeviceProjector(surface, device=device)
        # BEGIN the original camera/UV boundary is visible in the stage record.
        report['raw_projection_camera'] = projector.camera_stats
        # END these CPU/transfer costs are already included in the enclosing stages.
        delta = xyz[pair_b]-xyz[pair_a]
        thickness = config.surface_height_spacing_factor * torch.minimum(spacing[pair_a], spacing[pair_b])
        good = valid[pair_a] & valid[pair_b]
        # BEGIN preserve the original three-dimensional einsum reduction order.
        good &= dot3_einsum(normals[pair_a], normals[pair_b]) >= config.normal_min_agreement
        good &= dot3_einsum(delta, normals[pair_a]).abs() <= thickness
        good &= dot3_einsum(delta, normals[pair_b]).abs() <= thickness
        # END pair geometry uses the same exact threshold tests as the CPU path.
        pair_a, pair_b = pair_a[good], pair_b[good]
        ab_uv, ab_support = _positions(projector, xyz[pair_a], 1)
        ba_uv, ba_support = _positions(projector, xyz[pair_b], 0)
        good = ab_support & ba_support
        for uv, other in ((ab_uv, pair_b), (ba_uv, pair_a)):
            xy = pixels[other]
            good &= (xy >= torch.floor(uv)).all(1) & (xy <= torch.ceil(uv)).all(1)
        pair_a, pair_b = pair_a[good], pair_b[good]
        report['accepted_pair_count'] = len(pair_a)
        finish('pair_geometry_and_raw_projection', part)
        part = perf_counter()
        parent = torch.arange(len(xyz), dtype=torch.int64, device=device)
        parent[pair_b] = pair_a
        roots, mapping = torch.unique(parent, sorted=True, return_inverse=True)
        representative = roots.clone()
        choose_b = cosine[pair_b] > cosine[pair_a]
        representative[mapping[pair_a[choose_b]]] = pair_b[choose_b]
        count = len(roots)
        members = torch.zeros((count, 2), dtype=torch.bool, device=device)
        members[mapping, observation_view] = True
        element_entity = torch.zeros(count, dtype=torch.int64, device=device)
        element_entity.scatter_reduce_(0, mapping, entity.to(torch.int64), reduce='amax', include_self=True)
        element_entity = element_entity.bool()
        surface.update(xyz=xyz[representative], normals=normals[representative],
            spacing=spacing[representative], valid=valid[representative], entity_member=element_entity,
            representative_observation=representative, observation_to_element=mapping, member_views=members)
        finish('element_mapping', part)
        part = perf_counter()
        observed = torch.stack([_positions(projector, surface['xyz'], view)[1]
                                for view in range(2)], dim=1)
        surface['observed_views'] = observed | members
        finish('observed_views', part)
        part = perf_counter()
        graph = _adjacency(surface, count, config)
        finish('pixel_adjacency', part)
        part = perf_counter()
        result = {name: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                  for name, value in surface.items()}
        result['pixel_to_observation'] = [lookup.cpu().numpy() for lookup in lookups]
        data, columns, indptr = (value.cpu().numpy() for value in graph)
        result['adjacency'] = csr_matrix((data, columns, indptr), shape=(count, count))
        result['diagnostics'] = dict(raw_observation_count=len(xyz), element_count=count,
            paired_element_count=len(pair_a), entity_element_count=int(result['entity_member'].sum()),
            valid_element_count=int(result['valid'].sum()), unpaired_is_physical_exclusivity=False,
            geometry_domain='complete_valid_depth', representative='actual_less_grazing_measurement')
        finish('final_numpy_csr_export', part)
    # END the worker is joined before final reporting and return.
    report.update(complete=True, observation_count=len(xyz), element_count=count,
        adjacency_nonzeros=len(data), total_seconds=perf_counter()-started,
        numerical_contract='FP64 same Jacobi window order; CPU original coordinates, nearest tie selection and 8192-row camera/UV arithmetic; all output differences must be checked')
    if stats is not None:
        stats.update(report)
    return result
