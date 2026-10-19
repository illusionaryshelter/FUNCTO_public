"""Selected dual-view inference from resident observed inputs and MARCO features.

Both models share observed geometry and raw response preparation. Planar uses
its fixed ranking/TEASER path; surface uses bounded native geometric slots and
source-weighted absolute geometry and appearance support. No file reads, model
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
    """Observed source inputs cannot instantiate the explicitly declared model."""
    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__(';'.join(self.reasons))


# BEGIN explicit model runtime: surface registration is entirely CPU-native.
def prepare(region_geometry):
    """Load only the explicitly selected model's native numerical runtime.

    This preparation precedes online timing. It never chooses a different
    model after a runtime failure. Planar retains its CUDA/TEASER kernels;
    surface needs only the installed Open3D and NumPy/SciPy CPU operations.
    """
    torch.set_num_threads(1)
    if region_geometry == 'surface':
        from . import curved_proposals, curved_inference
        runtime = dict(native_backend='open3d', geometry_backend='numpy_scipy', cuda_used=False)
    elif region_geometry == 'planar':
        import teaserpp_python
        from .cuda_proposals import load_extension
        from .cuda_evidence import prepare as prepare_evidence
        from . import candidate_events, region_scope
        if not torch.cuda.is_available():
            raise RuntimeError('The selected dual-view implementation requires CUDA')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.cuda.get_device_properties(0)
        cap = min(6*1024**3, device.total_memory//2)
        torch.cuda.set_per_process_memory_fraction(cap/device.total_memory)
        pairs.prepare()
        surface.prepare()
        load_extension()
        prepare_evidence()
        runtime = dict(gpu=device.name, gpu_total_bytes=device.total_memory,
            gpu_allocator_cap_bytes=cap, native_backend='teaserpp', cuda_used=True)
    else:
        raise ValueError('region_geometry must be planar or surface')
    # Resolve BLAS discovery before timing; run still scopes/restores its limit.
    with threadpool_limits(limits=1, user_api='blas'):
        pass
    return runtime
# END the selected model has one runtime, without automatic fallback.


def run(loaded, feature_pairs, *, sampling_radius_m=.03, sampling_spacing_m=.005):
    """Compute all stages freshly, returning the result and its actual assets.

    ``loaded`` holds one source, two calibrated depth views and fixed query
    identities. ``feature_pairs`` maps each available view ID to source/target
    feature arrays. Inputs remain caller-owned and read-only. Intermediate
    outputs support persistence and strict validation; they are never taken
    from a previous prediction. ``prepare`` must have completed first.
    """
    # BEGIN explicit geometry dispatch. The planar functions and argument
    # values remain the original numerical path; curve failures never reroute.
    geometry = loaded['region_geometry']
    if geometry == 'planar':
        build_target, build_source, infer_region = surface.build_target_surface, prepare_source, infer
    elif geometry == 'surface':
        from .curved_geometry import build_target_surface as build_target
        from .curved_geometry import prepare_source as build_source
        from .curved_inference import infer as infer_region
    else:
        raise ValueError('region_geometry must be planar or surface')
    # END geometry dispatch; each model owns its appearance scoring contract.
    stages, native_pairs, surface_work, work = {}, {}, {}, {}
    with threadpool_limits(limits=1, user_api='blas'):
        torch.set_num_threads(1)
        if geometry == 'planar':
            torch.cuda.synchronize()
        started = perf_counter()
        mark = perf_counter()
        # BEGIN call the selected geometry implementation with unchanged units.
        target = build_target(loaded['views'],
            sampling_radius_m=sampling_radius_m, stats=surface_work)
        if geometry == 'planar':
            torch.cuda.synchronize()
        stages['target_surface'] = perf_counter()-mark
        mark = perf_counter()
        source = build_source(loaded['source'], loaded['source_depth'],
            query_indices=loaded['query_indices'], query_region=loaded['query_region'],
            sampling_radius_m=sampling_radius_m)
        # END no global-plane gate is reached by the surface path.
        stages['source_geometry'] = perf_counter()-mark
        if source['reasons']:
            raise InsufficientSource(source['reasons'])
        # BEGIN planar-only initializers; the surface proposer owns all slots.
        if geometry == 'planar':
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
        # END no extra surface candidates exist outside its bounded proposer.

        mark = perf_counter()
        appearance = appearance_field(loaded, target, feature_pairs)
        stages['response_maps'] = perf_counter()-mark
        # BEGIN each model prepares only the evidence its decision consumes.
        if geometry == 'planar':
            mark = perf_counter()
            domain = rank_domain(source, target, appearance)[4]
            stages['ranking'] = perf_counter()-mark
            mark = perf_counter()
            native, native_records = native_transforms(loaded, native_pairs)
            stages['native_registration'] = perf_counter()-mark
        mark = perf_counter()
        if geometry == 'surface':
            result = infer_region(source, target, appearance,
                sampling_radius_m=sampling_radius_m,
                sampling_spacing_m=sampling_spacing_m, work_stats=work)
        else:
            result = infer_region(source, target, appearance, initial_transforms=native,
                prepared_domain=domain, sampling_radius_m=sampling_radius_m,
                sampling_spacing_m=sampling_spacing_m, work_stats=work)
        # END the surface evaluator freezes its own per-view response references.
        if geometry == 'planar':
            torch.cuda.synchronize()
        stages['candidate_inference'] = perf_counter()-mark
        # BEGIN preserve each explicit model's result identity.
        result['diagnostics'].update(
            scheme=('dual_view_observed_surface' if geometry == 'surface' else 'dual_view_observed_partial'),
            native_backend='open3d' if geometry == 'surface' else 'teaserpp', target_truth_used=False,
            input_missing_views=[v['id'] for v in loaded['views'] if v['observation'] is None])
        if geometry == 'planar':
            result['diagnostics'].update(native_proposals=native_records,
                native_seconds=stages['native_registration'])
        # END original planar metadata values remain unchanged.
        if geometry == 'planar':
            torch.cuda.synchronize()
        elapsed = perf_counter()-started
    outputs = dict(result=result, surface=target, source=source, appearance=appearance,
        stages_seconds=stages, continuous_seconds=elapsed, surface_work=surface_work, work=work)
    if geometry == 'planar':
        outputs.update(native_pairs=native_pairs, native_transforms=native, domain=domain)
    return outputs
