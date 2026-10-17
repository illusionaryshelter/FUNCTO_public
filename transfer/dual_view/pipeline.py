"""Selected dual-view inference from resident observed inputs and MARCO features.

Stages: full-depth surface, source geometry, native pairs, response maps, fixed
target ranking, TEASER++ and finite candidate inference. No file reads, model
calls or historical predictions occur inside ``run``. Use in a serial process.
"""
from time import perf_counter
import torch
from threadpoolctl import threadpool_limits
from ..experiment import feature_observation
from . import pairs, surface
from .geometry import prepare_source
from .inference import infer
from .io import appearance_field, native_transforms
from .proposals import rank_domain


class InsufficientSource(ValueError):
    """Observed source inputs cannot instantiate the declared planar model."""
    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__(';'.join(self.reasons))


def prepare():
    """Load installed native code and configure the selected serial runtime.

    Call before measuring online computation. Missing native packages, CUDA
    or toolchain components raise directly; no backend is substituted. Builds
    use the existing environment and one compiler worker.
    """
    import teaserpp_python
    from .cuda_proposals import load_extension
    from .cuda_evidence import prepare as prepare_evidence
    if not torch.cuda.is_available():
        raise RuntimeError('The selected dual-view implementation requires CUDA')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.cuda.get_device_properties(0)
    cap = min(6*1024**3, device.total_memory//2)
    torch.cuda.set_per_process_memory_fraction(cap/device.total_memory)
    pairs.prepare()
    surface.prepare()
    load_extension()
    prepare_evidence()
    return dict(gpu=device.name, gpu_total_bytes=device.total_memory,
                gpu_allocator_cap_bytes=cap, native_backend='teaserpp')


def run(loaded, feature_pairs, *, sampling_radius_m=.03, sampling_spacing_m=.005):
    """Compute all stages freshly, returning the result and its actual assets.

    ``loaded`` holds one source, two calibrated depth views and fixed query
    identities. ``feature_pairs`` maps each available view ID to source/target
    feature arrays. Inputs remain caller-owned and read-only. Intermediate
    outputs support persistence and strict validation; they are never taken
    from a previous prediction. ``prepare`` must have completed first.
    """
    stages, native_pairs, surface_work, work = {}, {}, {}, {}
    with threadpool_limits(limits=1, user_api='blas'):
        torch.set_num_threads(1)
        torch.cuda.synchronize()
        started = perf_counter()
        mark = perf_counter()
        target = surface.build_target_surface(loaded['views'],
            sampling_radius_m=sampling_radius_m, stats=surface_work)
        torch.cuda.synchronize()
        stages['target_surface'] = perf_counter()-mark
        mark = perf_counter()
        source = prepare_source(loaded['source'], loaded['source_depth'],
            query_indices=loaded['query_indices'], query_region=loaded['query_region'],
            sampling_radius_m=sampling_radius_m)
        stages['source_geometry'] = perf_counter()-mark
        if source['reasons']:
            raise InsufficientSource(source['reasons'])
        mark = perf_counter()
        torch.set_num_threads(8)
        try:
            for view in loaded['views']:
                if view['observation'] is None:
                    continue
                native_pairs[view['id']] = pairs.native_region_pairs(
                    feature_observation(loaded['source'], loaded['source_crop']),
                    feature_observation(view['observation'], view['crop_xyxy']),
                    loaded['source']['region_mask'], feature_pairs[view['id']])
        finally:
            torch.set_num_threads(1)
        stages['native_pairs'] = perf_counter()-mark
        mark = perf_counter()
        appearance = appearance_field(loaded, target, feature_pairs)
        stages['response_maps'] = perf_counter()-mark
        mark = perf_counter()
        domain = rank_domain(source, target, appearance)[4]
        stages['ranking'] = perf_counter()-mark
        mark = perf_counter()
        native, native_records = native_transforms(loaded, native_pairs)
        stages['native_registration'] = perf_counter()-mark
        mark = perf_counter()
        result = infer(source, target, appearance, initial_transforms=native,
            prepared_domain=domain, sampling_radius_m=sampling_radius_m,
            sampling_spacing_m=sampling_spacing_m, work_stats=work)
        torch.cuda.synchronize()
        stages['candidate_inference'] = perf_counter()-mark
        result['diagnostics'].update(scheme='dual_view_observed_partial',
            native_backend='teaserpp', native_proposals=native_records,
            native_seconds=stages['native_registration'], target_truth_used=False,
            input_missing_views=[v['id'] for v in loaded['views'] if v['observation'] is None])
        torch.cuda.synchronize()
        elapsed = perf_counter()-started
    return dict(result=result, surface=target, source=source, native_pairs=native_pairs,
        native_transforms=native, appearance=appearance, domain=domain,
        stages_seconds=stages, continuous_seconds=elapsed, surface_work=surface_work, work=work)
