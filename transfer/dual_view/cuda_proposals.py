"""Exact finite proposal scheduling with CUDA rank-ordered correspondence growth.

The CPU keeps NumPy's original conditional random calls and original Sim(3)
fits. A CUDA block grows one fixed plan, visiting the complete declared prefix
in appearance-rank order. Stopping after the first successful rank tile is an
exact minimum reduction, never a top-k approximation. Numerically ambiguous
KDTree-ball boundary rows are explicitly certified by the original CPU query
before that growth resumes; no CUDA exception selects a different backend.
"""

from functools import lru_cache
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parent
LOAD_RECORD = {}


@lru_cache(maxsize=1)
def load_extension():
    """Compile/load the local extension explicitly outside experiment clocks.

Only the existing Torch/CUDA toolchain is used. Build outputs remain beside
this module, compilation is serial, and no package or system file is changed.
Importing this module neither compiles code nor initializes a CUDA device.
"""
    from torch.utils.cpp_extension import load
    started = perf_counter()
    build = ROOT / 'build-cuda-proposals'
    build.mkdir(exist_ok=True)
    os.environ['MAX_JOBS'] = '1'
    extension = load(name='recovery_cuda_proposals',
        sources=[str(ROOT/'cuda_proposals_bind.cpp'), str(ROOT/'cuda_proposals_kernel.cu')],
        extra_cflags=['-O2', '-g0', '-std=c++17'],
        extra_cuda_cflags=['-O2', '--fmad=false', '--prec-div=true', '--prec-sqrt=true'],
        build_directory=str(build), with_cuda=True, verbose=True)
    LOAD_RECORD.update(seconds=perf_counter()-started, max_jobs=1,
        build_directory=str(build), fast_math=False, fused_multiply_add=False)
    return extension


def make_plans(queries, region_rows, source_size, order, target, tree, diagnostics):
    """Consume exactly the original conditional RNG method calls on the CPU.

There are at most two fits per growth and FIT_BUDGET == 2*GROWTH_BUDGET.
Therefore no earlier fit result can prevent a later growth's random calls.
The inner growth never calls the RNG. Seed distances also stay on the CPU,
because their scale-domain comparisons decide whether permutation is called.
"""
    from . import proposals as original
    if original.FIT_BUDGET != 2*original.GROWTH_BUDGET:
        raise ValueError('CUDA scheduling requires the reviewed two-fits-per-growth budget')
    rng = np.random.default_rng(original.SEED)
    all_rows = np.arange(len(queries))
    distances = cdist(queries, queries)
    plans = []
    for attempt in range(original.GROWTH_BUDGET):
        diagnostics['growth_attempts'] += 1
        if not len(region_rows) or len(queries) < 6 or tree is None:
            continue
        first = int(region_rows[attempt % len(region_rows)])
        second = int(rng.choice(all_rows[all_rows != first]))
        # Preserve the original NumPy scalar argument type passed to integers;
        # convert prefixes to device int32 only after all RNG calls finish.
        prefixes = np.asarray([original.prefix_length(len(row), attempt) for row in order])
        if attempt == original.GROWTH_BUDGET-1:
            diagnostics['full_domain_prefix_reached'] = bool(
                np.array_equal(prefixes, [len(row) for row in order]))
        if not prefixes[first] or not prefixes[second]:
            continue
        columns = [int(order[first][rng.integers(prefixes[first])]),
                   int(order[second][rng.integers(prefixes[second])])]
        source_distance = float(distances[first, second])
        target_distance = float(np.linalg.norm(target[columns[0]]-target[columns[1]]))
        if columns[0] == columns[1] or min(source_distance, target_distance) <= 0:
            continue
        scale = target_distance/source_distance
        if not original.MINIMUM_SCALE <= scale <= original.MAXIMUM_SCALE:
            continue
        tolerance = original.COMPATIBILITY_RATIO*source_size*scale
        permutation = rng.permutation(all_rows[(all_rows != first) & (all_rows != second)])
        outer = np.nextafter(scale*distances[:, first]+2*tolerance, np.inf)
        plans.append(dict(attempt=attempt, rows=[first, second], columns=columns,
            prefixes=prefixes, permutation=permutation.astype(np.int32),
            scale=scale, tolerance=tolerance, outer=outer))
    return plans, distances


def ball_roundoff_bound(tree):
    """Conservative squared-distance error enclosure for KDTree pruning.

    SciPy's rectangle tracker updates min_distance by (new_axis-old_axis), or
    recomputes all three axes. A path has fewer than tree.size nodes. Counting
    32 operations/node plus 16 for the final point/radius expressions, and using
    eps (twice binary64 unit roundoff), gives a conservative gamma(k) enclosure.
    Its absolute scale is an upward-rounded squared root-box diameter, which
    bounds every exact nonnegative rectangle distance for a center in the tree.
    This deliberately uses node count instead of traversing the tree for depth.

An overflowed or unrepresentable enclosure is infinity: the affected selected
row is then explicitly evaluated by the original CPU tree, never guessed.
"""
    epsilon = np.finfo(np.float64).eps
    operations = 32*(int(tree.size)+1)+16
    product = operations*epsilon
    if product >= .5:
        return np.inf
    with np.errstate(over='ignore', invalid='ignore'):
        spans = np.nextafter(tree.maxes-tree.mins, np.inf)
        squares = np.nextafter(spans*spans, np.inf)
        squared = np.nextafter(squares[0]+squares[1], np.inf)
        squared = max(float(np.nextafter(squared+squares[2], np.inf)),
                      np.finfo(np.float64).tiny)
        gamma = np.nextafter(product/(1-product), np.inf)
        result = gamma*squared
    return float(np.nextafter(result, np.inf)) if np.isfinite(result) else np.inf


def certify_row(plan, state, target, tree, inverse, distances):
    """Resolve one ambiguous row with the original tree and norm expressions.

The device has not appended a point for this row. The CPU evaluates the entire
tree-selected prefix, so a changed winner or an empty result affects every
later row correctly. RNG and fitting are deliberately absent here.
    """
    count, position = int(state[0]), int(state[1])
    row = int(plan['permutation'][position])
    rows = state[4:4+count].tolist()
    columns = state[16:16+count].tolist()
    candidates = np.asarray(tree.query_ball_point(target[columns[0]],
        plan['outer'][row], workers=1), dtype=np.int64)
    ranks = inverse[row, candidates]
    candidates = candidates[(ranks >= 0) & (ranks < plan['prefixes'][row])]
    candidates = candidates[~np.isin(candidates, columns)]
    evaluations = 0
    for anchor_row, anchor_column in zip(rows, columns):
        if not len(candidates):
            break
        desired = plan['scale']*distances[row, anchor_row]
        distance = np.linalg.norm(target[candidates]-target[anchor_column], axis=1)
        evaluations += len(candidates)
        candidates = candidates[np.abs(distance-desired) <= 2*plan['tolerance']]
    if len(candidates):
        column = int(candidates[np.argmin(inverse[row, candidates])])
        state[4+count], state[16+count] = row, column
        state[0] += 1
    state[1] += 1
    state[2] = 2 if state[0] == 12 or state[1] == len(plan['permutation']) else 0
    return evaluations


def grow_cuda(plans, distances, target, order, inverse, tree, stats):
    """Upload this call's inputs and execute bounded-memory growth/row resumes."""
    import torch
    extension = load_extension()
    if not torch.cuda.is_available():
        raise RuntimeError('Explicit CUDA proposal backend requires an available CUDA device')
    groups, rows = len(plans), len(order)
    # State columns: count, next permutation position, status, reserved,
    # selected source rows [12], selected target columns [12]. Status 0 runs,
    # 1 requests CPU boundary certification, and 2 means completed.
    state = np.full((groups, 28), -1, dtype=np.int32)
    state[:, :4] = [2, 0, 0, 0]
    for index, plan in enumerate(plans):
        state[index, 4:6] = plan['rows']
        state[index, 16:18] = plan['columns']
    lengths = np.asarray([len(value) for value in order], dtype=np.int64)
    offsets = np.r_[np.int64(0), np.cumsum(lengths)]
    ranked = np.concatenate(order).astype(np.int32, copy=False)
    prefixes = np.stack([plan['prefixes'] for plan in plans]).astype(np.int32)
    permutations = np.stack([plan['permutation'] for plan in plans])
    scales = np.asarray([plan['scale'] for plan in plans])
    bands = np.asarray([2*plan['tolerance'] for plan in plans])
    # Keep the CPU nextafter and multiply order of the original search radius.
    with np.errstate(over='ignore'):
        outer_squared = np.stack([plan['outer']*plan['outer'] for plan in plans])
    uncertainty = np.full(groups, ball_roundoff_bound(tree))
    inputs = (np.ascontiguousarray(target, dtype=np.float64),
              np.ascontiguousarray(distances), ranked, offsets, prefixes,
              permutations, scales, bands, outer_squared, uncertainty)
    stats['uploaded_input_bytes'] = sum(array.nbytes for array in inputs)+state.nbytes
    # This bound is only this module's live tensors; the experiment driver also
    # enforces the device-wide half-memory cap before starting any operation.
    if stats['uploaded_input_bytes'] > 512*1024**2:
        raise MemoryError('CUDA proposal inputs exceed the explicit 512 MiB module budget')
    started = perf_counter()
    device_inputs = [torch.as_tensor(array, device='cuda') for array in inputs]
    device_state = torch.as_tensor(state, device='cuda')
    device_work = torch.zeros((groups, 3), device='cuda', dtype=torch.int64)
    torch.cuda.synchronize()
    stats['upload_seconds'] = perf_counter()-started
    stats.update(cuda_kernel_seconds=0., state_transfer_seconds=0.,
                 boundary_certification_seconds=0., boundary_certified_rows=0,
                 boundary_cpu_point_anchor_distances=0, kernel_launches=0)
    while True:
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        extension.grow(*device_inputs, device_state, device_work)
        end.record()
        end.synchronize()
        stats['cuda_kernel_seconds'] += begin.elapsed_time(end)/1000
        stats['kernel_launches'] += 1
        started = perf_counter()
        state = device_state.cpu().numpy().copy()
        stats['state_transfer_seconds'] += perf_counter()-started
        pending = np.flatnonzero(state[:, 2] == 1)
        if not len(pending):
            if not np.all(state[:, 2] == 2):
                raise RuntimeError('CUDA proposal growth returned an invalid completion state')
            break
        started = perf_counter()
        for index in pending:
            stats['boundary_cpu_point_anchor_distances'] += certify_row(
                plans[index], state[index], target, tree, inverse, distances)
        stats['boundary_certified_rows'] += len(pending)
        stats['boundary_certification_seconds'] += perf_counter()-started
        if stats['kernel_launches'] > rows:
            raise RuntimeError('Boundary certification did not advance the growth position')
        started = perf_counter()
        device_state.copy_(torch.from_numpy(state))
        torch.cuda.synchronize()
        stats['state_transfer_seconds'] += perf_counter()-started
    work = device_work.cpu().numpy().sum(axis=0)
    stats.update(gpu_rank_candidates=int(work[0]), gpu_point_anchor_distances=int(work[1]),
                 gpu_source_rows=int(work[2]), cuda_completed_growths=groups,
                 maximum_ball_roundoff_squared=float(np.max(uncertainty)))
    return state


def execute(queries, region_rows, source_size, ids, order, inverse, target, tree,
            diagnostics, retain, fit_sim3, valid_transform, stats):
    """Run plans and restore original attempt/6-point/12-point fitting order."""
    if stats is None:
        stats = {}
    else:
        stats.clear()
    started = perf_counter()
    plans, distances = make_plans(queries, region_rows, source_size, order,
                                 target, tree, diagnostics)
    stats.update(seed_plan_seconds=perf_counter()-started, legal_growth_plans=len(plans),
                 fit_backend='original_cpu_numpy', growth_backend='cuda_rank_prefix',
                 boundary_backend='original_cpu_tree_one_row')
    if not plans:
        return
    state = grow_cuda(plans, distances, target, order, inverse, tree, stats)
    started = perf_counter()
    for index, plan in enumerate(plans):
        count = int(state[index, 0])
        for fitted in (6, 12):
            if count < fitted:
                continue
            selected_rows = state[index, 4:4+fitted].tolist()
            columns = state[index, 16:16+fitted].tolist()
            diagnostics['proposal_fit_attempts'] += 1
            try:
                transform = fit_sim3(queries[selected_rows], target[columns])
            except (ValueError, np.linalg.LinAlgError):
                transform = None
            if transform is None or not valid_transform(transform):
                diagnostics['proposal_fit_failures'] += 1
            else:
                retain(transform, {'origin': f'unified_{fitted}_point',
                    'growth_attempt': plan['attempt'], 'query_rows': selected_rows,
                    'surface_ids': ids[columns].tolist(), 'pair_scale': plan['scale']})
    stats['cpu_fit_seconds'] = perf_counter()-started
