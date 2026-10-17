"""Fresh candidate evidence on the same GPU projection rows as the baseline.

Feature coordinates, fixed-reference pixel connectivity, interpolation and
empirical support are evaluated on CUDA. The original camera calculation for
full-sphere bounding boxes remains an explicit CPU boundary. It changes only
search bounds: every pixel is still tested by the original physical predicates.
The caller retains CPU SciPy assignment and the original diagnostics schema.
"""
from functools import lru_cache
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LOAD_RECORD = {}
STATE_EXTRA = ('opposite_query_normal', 'outside_entity_footprint',
    'appearance_pending', 'missing_entity_observation', 'missing_feature',
    'missing_crop', 'outside_feature_crop', 'outside_feature_grid',
    'partial_feature_support', 'no_feature_support', 'appearance_valid',
    'appearance_valid_orientation_unknown')


@lru_cache(maxsize=1)
def prepare():
    """Load the existing Torch/CUDA toolchain before any measured inference."""
    from torch.utils.cpp_extension import load
    started = perf_counter()
    build = HERE/'build-cuda-evidence'
    build.mkdir(exist_ok=True)
    os.environ['MAX_JOBS'] = '1'
    extension = load(name='recovery_cuda_evidence',
        sources=[str(HERE/'cuda_evidence_bind.cpp'), str(HERE/'cuda_evidence_kernel.cu')],
        extra_cflags=['-O2', '-g0', '-std=c++17'],
        extra_cuda_cflags=['-O2', '--fmad=false', '--prec-div=true', '--prec-sqrt=true'],
        build_directory=str(build), with_cuda=True, verbose=True)
    LOAD_RECORD.update(seconds=perf_counter()-started, max_jobs=1,
        build_directory=str(build), fast_math=False, fused_multiply_add=False)
    return extension


class CudaEvidence:
    """Own per-inference responses and references, sharing projection geometry.

    No prediction or sampled answer is retained between infer calls. Fixed
    response/reference uploads are limited to 512 MiB; connectivity uses at
    most 64 MiB of scratch and splits work by cells without truncating a box.
    """

    def __init__(self, appearance, domain, projector, stats):
        from transfer.multiview_geometry import STATE_NAMES
        from local_models.marco import RESOLUTION
        self.extension = prepare()
        self.surface, self.projector = appearance.surface, projector
        self.stats = stats
        self.names = np.asarray(tuple(STATE_NAMES)+STATE_EXTRA, dtype=object)
        self.resolution = RESOLUTION
        self.views = appearance.views
        references = domain['empirical_support'].references
        if len(references) != len(appearance.queries):
            raise ValueError('Evidence references must match frozen query identities')
        # Check the allocation size before converting any complete response or
        # joining references. The limit also bounds the host conversion copy.
        payload = sum(len(row) for row in references)*8+(len(references)+1)*8
        payload += len(self.surface['observation_pixels'])*16
        payload += len(self.surface['observation_to_element'])*8
        payload += sum(np.asarray(v['response']).size*8 for v in self.views if v['missing_reason'] is None)
        if payload > 512*1024**2:
            raise MemoryError('Candidate evidence uploads exceed the explicit 512 MiB budget')
        sizes = np.asarray([len(row) for row in references], dtype=np.int64)
        offsets = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
        flat = np.concatenate(references).astype(np.float64, copy=False)
        response_values = [np.empty((0, 2, 2)) if v['missing_reason'] else
                           np.asarray(v['response'], dtype=np.float64) for v in self.views]
        pixels = np.asarray(self.surface['observation_pixels'], dtype=np.int64)
        elements = np.asarray(self.surface['observation_to_element'], dtype=np.int64)
        self.pixels = projector.tensor(pixels, torch.int64)
        self.elements = projector.tensor(elements, torch.int64)
        self.references = projector.tensor(flat, torch.float64)
        self.offsets = projector.tensor(offsets, torch.int64)
        self.responses = [projector.tensor(value, torch.float64) for value in response_values]
        self.parameters = [projector.factor, projector.agreement, projector.radius]
        self.stats.update(upload_bytes=int(payload), full_sphere_cpu_bounds_seconds=0.,
                          full_sphere_query_count=0, connectivity_cells=0,
                          peak_connectivity_scratch_bytes=0, view_batches=0)

    def _geometry(self, view):
        p = self.projector
        return [p.observed, p.normals, p.spacing, p.valid, p.entity, self.pixels, p.lookup[view]]

    def _connect(self, integer, view, corners, eligible, rows, low, high, connected):
        """Allocate bounded scratch; cell counts never alter physical support."""
        if not rows.numel():
            return
        sizes = ((high-low+1).prod(dim=1)).cpu().numpy()
        max_cells = 8*1024**2  # Two int32 arrays: exactly 64 MiB.
        if np.any(sizes <= 0) or np.any(sizes > max_cells):
            raise MemoryError('One exact evidence pixel box exceeds the scratch budget')
        start = 0
        while start < len(sizes):
            cumulative = np.cumsum(sizes[start:], dtype=np.int64)
            count = int(np.searchsorted(cumulative, max_cells, side='right'))
            stop = start+count
            total = int(cumulative[count-1])
            offsets = self.projector.tensor(np.r_[0, cumulative[:count]], torch.int64)
            visited = torch.empty(total, dtype=torch.int32, device=self.projector.device)
            queue = torch.empty_like(visited)
            self.extension.connect([integer, *self._geometry(view), corners, eligible,
                rows[start:stop], low[start:stop], high[start:stop], offsets],
                [visited, queue, connected], self.parameters)
            self.stats['connectivity_cells'] += total
            self.stats['peak_connectivity_scratch_bytes'] = max(
                self.stats['peak_connectivity_scratch_bytes'], total*8)
            start = stop

    def sample_view(self, packed, query_rows, view, groups):
        """Sample internal rows validated before upload by the batch adapter.

        Each (start, stop) group is one original projection call. The adapter
        constructs rows as arange(Q) or -1, so no unchecked external query ID
        reaches a device response/reference lookup.
        """
        uv, integer, flags = [value.contiguous() for value in packed]
        n, device = len(uv), self.projector.device
        if query_rows.shape != (n,) or query_rows.dtype != torch.int64:
            raise ValueError('Evidence query rows must align with projection rows')
        state = torch.empty(n, dtype=torch.int64, device=device)
        features = torch.empty((n, 4, 2), dtype=torch.int64, device=device)
        corners = torch.empty_like(features)
        weights = torch.empty((n, 4), dtype=torch.float64, device=device)
        eligible = torch.empty((n, 4), dtype=torch.bool, device=device)
        connected = torch.zeros_like(eligible)
        low = torch.empty((n, 2), dtype=torch.int64, device=device)
        high = torch.empty_like(low)
        record, response = self.views[view], self.responses[view]
        missing = record['missing_reason']
        crop = [0, 0, 1, 1] if missing else list(map(int, record['crop_xyxy']))
        resize = (1., 1.) if missing else record['resize_scale']
        fh, fw = response.shape[-2:]
        missing_state = -1 if missing is None else int(np.flatnonzero(self.names == missing)[0])
        self.extension.plan([uv, integer, flags, query_rows, *self._geometry(view)],
            [state, features, corners, weights, eligible, low, high],
            [*self.parameters, float(resize[0]), float(resize[1]), float(self.resolution-1)],
            [*crop, fh, fw, missing_state])
        rows = torch.nonzero(eligible.any(dim=1), as_tuple=False).flatten()
        self._connect(integer, view, corners, eligible, rows, low[rows], high[rows], connected)
        unresolved = torch.nonzero((eligible & ~connected).any(dim=1), as_tuple=False).flatten()
        if unresolved.numel():
            # Keep the baseline camera matrix calls and outward rounding for
            # boxes that need the entire sphere. The final graph remains CUDA.
            from transfer.dual_view.features import _sphere_pixel_bounds
            started = perf_counter()
            indices = unresolved.cpu().numpy()
            refs = integer[unresolved, 0].cpu().numpy()
            full_low, full_high = np.empty((len(refs), 2), np.int64), np.empty((len(refs), 2), np.int64)
            for begin, end in groups:
                lo, hi = np.searchsorted(indices, [begin, end])
                if hi > lo:
                    # Preserve each original candidate's matrix call shape;
                    # a singleton must not be merged into a different BLAS path.
                    full_low[lo:hi], full_high[lo:hi] = _sphere_pixel_bounds(self.surface, view, refs[lo:hi])
            self.stats['full_sphere_cpu_bounds_seconds'] += perf_counter()-started
            self.stats['full_sphere_query_count'] += len(refs)
            self._connect(integer, view, corners, eligible, unresolved,
                self.projector.tensor(full_low, torch.int64),
                self.projector.tensor(full_high, torch.int64), connected)
        scores = torch.empty(n, dtype=torch.float64, device=device)
        utilities = torch.empty_like(scores)
        elements = torch.empty(n, dtype=torch.int64, device=device)
        valid = torch.empty(n, dtype=torch.bool, device=device)
        self.extension.finish([integer, flags, query_rows, state, features, corners,
            weights, connected, response, self.references, self.offsets, self.elements],
            [scores, utilities, elements, valid], [fh, fw, *crop])
        self.stats['view_batches'] += 1
        return scores, utilities, elements, valid, state

    def assemble(self, projections, views):
        """Reconstruct the unchanged CPU result and two-view capacity inputs."""
        raw = np.stack([view[0] for view in views], axis=1)
        utility = np.stack([view[1] for view in views], axis=1)
        elements = np.stack([view[2] for view in views], axis=1)
        valid = np.stack([view[3] for view in views], axis=1)
        states = np.stack([self.names[view[4]] for view in views], axis=1)
        known = valid.any(axis=1)
        scores = np.max(np.where(np.isfinite(raw), raw, -np.inf), axis=-1)
        scores = np.where(known, scores, np.nan)
        counts = []
        for column in range(2):
            names, numbers = np.unique(states[:, column], return_counts=True)
            counts.append(dict(zip(names.tolist(), numbers.tolist())))
        sampled = dict(scores=scores, known=known, view_scores=raw, view_valid=valid,
            view_state=states, geometry=projections,
            view_orientation_known=np.stack([p['query_normal_valid'] for p in projections], axis=1),
            reason_counts=dict(zip(self.surface['view_ids'], counts)))
        return sampled, utility, elements
