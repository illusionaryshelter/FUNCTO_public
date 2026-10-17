"""Local geometry with fixed-domain, capacity-limited observed evidence.

Only demonstration geometry can veto a local region. Entity context continues
to contribute matching evidence and separate model diagnostics. A finite
winner supplies an estimated transform, never a physical pose certificate.
"""

from time import perf_counter
import numpy as np
from ..core import TransferConfig, _sampling_parameters
from ..sim3 import rank
from . import geometry
from .features import capacity_score
from .proposals import build_proposals, transform_key

# BEGIN local-region decision and observed-support objective revision.
SCHEME = 'dual_view_observed_partial'
DISPLAY_DIAGNOSTIC_LIMIT = 6


# BEGIN array-wise finite output conversion.
def _finite_list(value):
    """Convert real numeric arrays without scalar Python recursion.

    Current callers supply FP64 measured-response arrays. Each finite value
    becomes a Python float, including the original sign of zero; each NaN or
    infinity becomes None. No reduction, score arithmetic or sorting occurs.
    The scalar branch is retained because the private helper also accepts a
    scalar; it is the original scalar expression, not a fallback algorithm.
    """
    value = np.asarray(value)
    if value.ndim == 0:
        return float(value) if np.isfinite(value) else None
    # Preserve the original zero-dimensional float conversion even for real
    # extended precision and invalid complex/date input. astype(float) would
    # introduce overflow warnings/np.seterr failures for finite longdouble.
    finite = np.isfinite(value)
    result = np.full(value.shape, None, dtype=object)
    result[finite] = [float(np.asarray(item)) for item in value[finite]]
    return result.tolist()

# END exact float/null values and signed zeros.


def _support_summary(support):
    return {'support_count': int(support['support'].sum()),
        'conflict_count': int(support['conflict'].sum()),
        'unknown_count': int(support['unknown'].sum()),
        **{key: support[key].tolist() for key in
           ('support', 'conflict', 'view_support', 'view_conflict', 'view_states')},
        'state_names': support['state_names']}


def _region_summary(mode):
    return {'seed_index': mode['seed_index'], 'center': mode['center'].tolist(),
        'normal': mode['normal'].tolist(), 'sample_indices': mode['sample_indices'].tolist(),
        'diagnostics': mode['diagnostics']}



# BEGIN selected CUDA evidence with explicit per-inference work ownership.
def _infer(source_pack, surface, appearance, *, initial_transforms=(),
          prepared_domain=None, sampling_radius_m=.03, sampling_spacing_m=.005,
          region_scope, work_stats):
    """Evaluate every native, grown and fixed-partial-fit finite candidate.

    A fixed empirical response reference and fixed query denominator make
    removing an observation edge unable to improve the support objective.
    This fact does not establish descriptor identity or scale/rotation accuracy.
    """
    started = perf_counter()
    from .candidate_events import candidate_events, validate_appearance
    batch_stats = work_stats['batch']
    _sampling_parameters(sampling_radius_m, sampling_spacing_m)
    queries = np.asarray(source_pack['query_xyz'], dtype=float)
    regional = np.asarray(source_pack['query_region'], dtype=bool)
    indices = np.asarray(source_pack['query_indices'], dtype=np.int64)
    normals = np.asarray(source_pack['query_normals'], dtype=float)
    check_region = np.asarray(source_pack['check_region'])
    q = len(queries)
    if (queries.shape != (q, 3) or normals.shape != queries.shape
            or regional.shape != (q,) or indices.shape != (q,)
            or check_region.dtype != np.bool_
            or check_region.shape != (len(source_pack['check_xyz']),)):
        raise ValueError('Fixed source coordinates, roles, normals and check roles must align')
    sampling = dict(sampling_radius_m=sampling_radius_m, sampling_spacing_m=sampling_spacing_m)
    timing = {name: 0. for name in ('proposals', 'partial_refinement', 'candidate_evaluation',
        'query_sampling', 'independent_geometry', 'capacity_assignment',
        'region_construction', 'decision_and_display')}
    diagnostic = {'method': SCHEME, 'scheme': SCHEME, 'target_truth_used': False,
        'quality_warnings': ['finite_search_not_global', 'scale_rotation_not_certified'],
        'fixed_query_denominator': q, 'query_indices': indices.tolist(),
        'query_region': regional.tolist(), 'source': source_pack.get('diagnostics', {}),
        'query_count': q, 'region_query_count': int(regional.sum()),
        'context_query_count': int((~regional).sum()), 'view_ids': list(surface['view_ids']),
        'point_index_domain': 'surface.xyz',
        'score_definition': 'fixed_Q_capacity_one_positive_centered_target_domain_mid_cdf',
        'local_geometry_contract': 'region_queries_region_checks_center',
        'search_status': 'not_started', 'search_stage': 'input',
        'global_optimality_certified': False, 'region_identified': False,
        'finite_candidate_unique': False, 'transform_certified': False,
        'sampling_radius_m': sampling_radius_m, 'sampling_spacing_m': sampling_spacing_m,
        'sampling_scale_applied': False, 'hypotheses': [], 'refinement_records': [],
        'output_meaning': 'finite observed-evidence region with an estimated transform',
        'refit_attempts': 0, 'refit_retained_count': 0, 'refit_unavailable_count': 0,
        'display_diagnostic_limit': DISPLAY_DIAGNOSTIC_LIMIT,
        'display_limit_does_not_restrict_inference': True, 'timing_seconds': timing}
    result = {'scheme': SCHEME, 'status': 'not_started', 'accepted': False,
        'reasons': [], 'modes': [], 'diagnostics': diagnostic,
        'src_anchor_idx': np.empty(0, np.int64), 'tgt_anchor_idx': np.empty(0, np.int64),
        'sim3_scale': None, 'sim3_rotation': None, 'sim3_translation': None,
        'region_identified': False, 'finite_candidate_unique': False,
        'transform_certified': False, 'diagnostic_hypotheses': diagnostic['hypotheses']}
    warnings = diagnostic['quality_warnings']
    if not len(source_pack['check_xyz']):
        warnings.append('no_independent_source_check_points')
    if np.any(~np.asarray(source_pack['query_normal_valid'], dtype=bool)):
        warnings.append('some_source_orientations_unavailable')
    if q == 0 or not regional.any() or source_pack['reasons']:
        result.update(status='missing_input', reasons=list(source_pack['reasons'])
                      or ['no_fixed_source_region_queries'])
        diagnostic.update(decision_status=result['status'], source_model_applicable=False)
        timing['total'] = perf_counter()-started
        return result
    diagnostic['source_model_applicable'] = True
    validate_appearance(appearance, surface)
    hypotheses, seen = [], {}

    def evaluate(transform, origins, domain, prepared):
        key = transform_key(transform)
        if key in seen:
            hypotheses[seen[key]]['summary']['origins'].extend(origins)
            return
        begin = perf_counter()
        mapped, oriented = prepared['query_points'], prepared['query_normals']
        part = perf_counter()
        sampled = prepared['sampled']
        # Feature sampling already performed the exact original-depth checks.
        # Reusing them preserves oriented geometry even for RGB-only rows.
        support = geometry.combine_support(sampled['geometry'])
        timing['query_sampling'] += perf_counter()-part
        part = perf_counter()
        checks = geometry.combine_support(prepared['check_projection'])
        center, normal = prepared['center'], prepared['normal']
        center_support = geometry.combine_support(prepared['center_projection'])
        timing['independent_geometry'] += perf_counter()-part
        part = perf_counter()
        utilities, element_ids = prepared['utilities'], prepared['elements']
        evidence = capacity_score(utilities, element_ids)
        timing['capacity_assignment'] += perf_counter()-part
        local_rows = np.flatnonzero(regional & support['support'])
        # The local evidence gate precedes capacity assignment. A tied LAP
        # optimum may give a shared element to context; that arbitrary tie
        # cannot erase a genuinely available local image observation.
        local_evidence = bool(np.any(utilities[regional] > 0))
        rejections = [reason for violated, reason in (
            (np.any(support['conflict'] & regional), 'region_query_geometry_model_conflict'),
            (np.any(checks['conflict'] & check_region), 'region_check_geometry_model_conflict'),
            (bool(center_support['conflict'][0]), 'center_geometry_model_conflict'),
            (len(local_rows) < 6, 'fewer_than_six_oriented_region_supports'),
            (rank(queries[local_rows])[0] < 2, 'supported_region_rank_below_two'),
            (not local_evidence, 'no_positive_region_appearance')) if violated]
        part = perf_counter()
        mode, reason = geometry._region_from_supported_center(surface, center, normal, center_support,
            prepared_domain=domain, region_scope=region_scope, **sampling)
        timing['region_construction'] += perf_counter()-part
        ambiguous = reason == 'center_projection_ambiguous'
        if reason is not None and not ambiguous:
            rejections.append(reason)
        cq = int(np.count_nonzero(support['conflict'] & ~regional))
        cc = int(np.count_nonzero(checks['conflict'] & ~check_region))
        uq = int(np.count_nonzero(support['unknown'] & ~regional))
        uc = int(np.count_nonzero(checks['unknown'] & ~check_region))
        entity_state = 'observed_conflict' if cq or cc else 'partially_observed' if uq or uc else 'observed_consistent'
        known, matched = np.asarray(sampled['known'], bool), np.asarray(evidence['matched'], bool)
        summary = {'hypothesis_id': len(hypotheses), 'origins': list(origins),
            'transform': [float(transform[0]), transform[1].tolist(), transform[2].tolist()],
            'score': float(evidence['score']), 'known_count': int(known.sum()),
            'unknown_count': int((~known).sum()),
            'known_region_count': int(np.count_nonzero(known & regional)),
            'known_context_count': int(np.count_nonzero(known & ~regional)),
            'positive_region_edge_count': int(np.count_nonzero(np.any(utilities > 0, axis=1) & regional)),
            'positive_region_matched_count': int(np.count_nonzero(matched & regional)),
            'matched_count': int(evidence['matched_count']),
            'capacity_lost_count': int(evidence['capacity_lost_count']),
            'scores': _finite_list(sampled['scores']), 'known': known.tolist(),
            'view_scores': _finite_list(sampled['view_scores']), 'view_utilities': _finite_list(utilities),
            'row_utilities': evidence['row_utilities'].tolist(),
            'view_valid': np.asarray(sampled['view_valid']).tolist(),
            'appearance_view_states': np.asarray(sampled['view_state']).tolist(),
            'appearance_reason_counts': sampled['reason_counts'],
            'geometry': _support_summary(support), 'checks': _support_summary(checks),
            'center_geometry': _support_summary(center_support), 'region_geometry_count': len(local_rows),
            'entity_model_status': entity_state, 'context_query_conflict_count': cq,
            'context_check_conflict_count': cc, 'context_query_unknown_count': uq,
            'context_check_unknown_count': uc, 'paired_query_rows': np.flatnonzero(matched).tolist(),
            'paired_surface_indices': evidence['element_ids'][matched].tolist(),
            'paired_view_indices': evidence['selected_view'][matched].tolist(),
            'mapped_center': center.tolist(), 'mapped_normal': normal.tolist(),
            'admitted_to_H': not rejections, 'output_available': mode is not None,
            'center_projection_ambiguous': ambiguous, 'output_reason': reason,
            'rejections': rejections, 'potential_winner': False,
            'region': None if mode is None else _region_summary(mode)}
        seen[key] = len(hypotheses)
        hypotheses.append({'transform': transform, 'summary': summary, 'center': center, 'normal': normal})
        diagnostic['hypotheses'].append(summary)
        timing['candidate_evaluation'] += perf_counter()-begin

    def reconstruct(h, domain, kind):
        mode, reason = geometry.region_for_transform(surface, source_pack, h['transform'],
            prepared_domain=domain, region_scope=region_scope, **sampling)
        if reason is not None:
            raise RuntimeError('Frozen candidate region could not be reconstructed')
        mode.update(type=kind, hypothesis_index=h['summary']['hypothesis_id'], score=h['summary']['score'])
        return mode

    try:
        diagnostic.update(search_stage='target_domain_and_finite_proposals', search_status='running')
        part = perf_counter()
        proposals, proposal_report, domain = build_proposals(source_pack, surface, appearance,
            initial_transforms=initial_transforms, prepared_domain=prepared_domain,
            work_stats=work_stats['proposals'])
        timing['proposals'] = perf_counter()-part
        diagnostic['proposals'] = proposal_report
        diagnostic['appearance_reference'] = domain['empirical_support'].diagnostics
        diagnostic['search_stage'] = 'local_geometry_observed_score_and_partial_refits'
        # BEGIN commit the same parent/refit events after bounded CUDA work.
        for event in candidate_events(proposals, source_pack, surface, domain, seen,
                                      appearance, batch_stats, work_stats['evidence']):
            if event['kind'] == 'evaluate':
                evaluate(event['transform'], event['origins'], domain, event['prepared'])
            elif event['kind'] == 'refinement':
                timing['partial_refinement'] += event['seconds']
                diagnostic['refinement_records'].append(event['report'])
                diagnostic['refit_attempts'] += 1
                diagnostic['refit_retained_count'] += event['retained']
                diagnostic['refit_unavailable_count'] += int(not event['retained'])
            else:
                raise event['error']
        # END candidate scores, decisions and diagnostic order are unchanged.
        part = perf_counter()
        admitted = [h for h in hypotheses if h['summary']['admitted_to_H']]
        diagnostic.update(search_stage='observed_region_decision',
            evaluated_transform_count=len(hypotheses), hypotheses_admitted=len(admitted))
        if not admitted:
            # Geometry-only candidates with no positive local image evidence
            # are different from a domain containing no feasible local region.
            appearance_only = any(h['summary']['rejections'] == ['no_positive_region_appearance']
                                  for h in hypotheses)
            result.update(status='insufficient_appearance' if appearance_only
                          else 'rejected_no_supported_region',
                reasons=['no_positive_region_appearance'] if appearance_only
                        else ['no_locally_supported_hypothesis_in_declared_search'])
        else:
            best = max(h['summary']['score'] for h in admitted)
            winners = [h for h in admitted if h['summary']['score'] == best]
            for h in winners:
                h['summary']['potential_winner'] = True
            diagnostic.update(best_score=best,
                potential_winner_ids=[h['summary']['hypothesis_id'] for h in winners],
                potential_winner_count=len(winners), finite_candidate_unique=len(winners) == 1)
            primary = None
            if best <= 0:
                result.update(status='insufficient_appearance', reasons=['no_positive_observed_support'])
            elif any(h['summary']['center_projection_ambiguous'] for h in winners):
                result.update(status='rejected_ambiguous_region', reasons=['center_projection_ambiguous'])
            elif len(winners) == 1:
                primary = reconstruct(winners[0], domain, 'primary')
            else:
                primary, reason = geometry.consensus_region(surface,
                    (reconstruct(h, domain, 'tied_primary') for h in winners),
                    np.asarray([h['center'] for h in winners]),
                    np.asarray([h['normal'] for h in winners]), **sampling)
                if reason is not None:
                    result.update(status='rejected_ambiguous_region', reasons=[reason])
                else:
                    warnings.append('tied_estimates_common_region_only')
            if primary is not None:
                primary.update(type='primary', score=best,
                    hypothesis_indices=[h['summary']['hypothesis_id'] for h in winners])
                result.update(accepted=True, status='accepted_observed_region', modes=[primary],
                    region_identified=True, finite_candidate_unique=len(winners) == 1)
                diagnostic['region_identified'] = True
                if len(winners) == 1:
                    h, transform = winners[0], winners[0]['transform']
                    result.update(sim3_scale=transform[0], sim3_rotation=transform[1], sim3_translation=transform[2],
                        src_anchor_idx=indices[h['summary']['paired_query_rows']],
                        tgt_anchor_idx=np.asarray(h['summary']['paired_surface_indices'], np.int64))
                if any(h['summary']['entity_model_status'] != 'observed_consistent' for h in winners):
                    warnings.append('entity_transform_not_fully_supported')
                if any(h['summary']['positive_region_matched_count'] < 6 for h in winners):
                    warnings.append('limited_capacity_distinct_region_appearance')
                if any(h['summary']['unknown_count'] for h in winners):
                    warnings.append('selected_estimate_has_unknown_appearance')
                # The strongest distant/surface competitor is a diagnostic
                # witness, without a tuned confidence margin or probability.
                cfg = TransferConfig()
                slab = cfg.surface_height_spacing_factor * surface['spacing'][primary['seed_index']]
                witnessed, competitors = set(), []
                for h in sorted(admitted, key=lambda a: (-a['summary']['score'], a['summary']['hypothesis_id'])):
                    if h['summary']['potential_winner'] or not h['summary']['output_available']:
                        continue
                    candidate = h['summary']['region']
                    delta = np.asarray(candidate['center'])-primary['center']
                    height = float(delta @ primary['normal'])
                    other_plane = abs(height) > slab or float(np.asarray(candidate['normal']) @ primary['normal']) < cfg.normal_min_agreement
                    tangent = np.sqrt(max(float(delta @ delta)-height**2, 0.))
                    kind = 'different_surface' if other_plane else 'same_surface_remote_location' if tangent > 2*sampling_radius_m else None
                    if kind is None or kind in witnessed:
                        continue
                    witnessed.add(kind)
                    competitors.append({'hypothesis_id': h['summary']['hypothesis_id'], 'kind': kind,
                        'score': h['summary']['score'], 'score_gap': best-h['summary']['score'],
                        'is_confidence_bound': False})
                    result['modes'].append(reconstruct(h, domain, 'competitor_'+kind))
                diagnostic['competing_regions'] = competitors
                if competitors:
                    warnings.append('other_observed_regions_compete')
        if not result['accepted']:
            choices = sorted((h for h in hypotheses if h['summary']['output_available']),
                key=lambda h: (not h['summary']['admitted_to_H'], -h['summary']['score'], h['summary']['hypothesis_id']))
            result['modes'] = [reconstruct(h, domain, 'rejected_diagnostic')
                              for h in choices[:DISPLAY_DIAGNOSTIC_LIMIT]]
            diagnostic.update(diagnostic_regions_available=len(choices),
                              diagnostic_regions_displayed=len(result['modes']))
        timing['decision_and_display'] = perf_counter()-part
        diagnostic.update(search_status='finite_domain_complete', search_stage='complete')
    except (MemoryError, InterruptedError) as error:
        result.update(accepted=False, status='search_incomplete',
            reasons=['declared_finite_search_interrupted'], modes=[],
            region_identified=False, finite_candidate_unique=False,
            sim3_scale=None, sim3_rotation=None, sim3_translation=None)
        diagnostic.update(search_status='search_incomplete', region_identified=False,
            finite_candidate_unique=False, interruption=str(error))
    timing['total'] = perf_counter()-started
    diagnostic['decision_status'] = result['status']
    return result
# END selected evaluator; original decision, warnings and interruption semantics.
# END local-region observed-evidence decision; no retired latent-score path.


# BEGIN one synchronous inference owns all mutable region/projection work.
def infer(source_pack, surface, appearance, *, initial_transforms=(),
          prepared_domain=None, sampling_radius_m=.03, sampling_spacing_m=.005,
          work_stats=None):
    """Run the selected CUDA/CPU estimator, retaining explicit failure states.

    The optional work mapping receives counters only; it supplies no cached
    prediction. Region subsets are released before this call returns.
    """
    from .region_scope import RegionScope
    work = dict(proposals={}, batch=[], evidence={}, region={})
    scope = RegionScope(geometry.sample_region)
    try:
        return _infer(source_pack, surface, appearance,
            initial_transforms=initial_transforms, prepared_domain=prepared_domain,
            sampling_radius_m=sampling_radius_m, sampling_spacing_m=sampling_spacing_m,
            region_scope=scope, work_stats=work)
    finally:
        work['region'] = dict(scope.counts)
        scope.close()
        if work_stats is not None:
            work_stats.clear()
            work_stats.update(work)
# END no inference-local geometry or response survives in module globals.
