"""Finite point-correspondence inference with explicit curved observations.

Absolute-position and same-measurement appearance support rank the finite set,
while complete observed-entity depth can refute a candidate. The highest score
selects one estimate; missing sampling output cannot promote another estimate.
"""

from time import perf_counter

import numpy as np

from ..sim3 import apply_sim3, rank
from . import curved_geometry as geometry
from .curved_features import SurfaceResponseField
from .inference import _finite_list, _support_summary
from .proposals import COMPATIBILITY_RATIO, transform_key, valid_transform
from .curved_proposals import CANDIDATE_BUDGET, build_proposals
# BEGIN: surface-only structural evidence; the planar evaluator is unchanged.
from .curved_structure import entity_evidence
# BEGIN one source-weighted geometry/appearance objective, without pair votes.
from .curved_scoring import (prepare_scoring, geometry_correspondences,
                             assigned_appearance, score_candidate)
# END appearance and geometry share one capacity-constrained correspondence.
# END: independent observed-entity module.


def combine_support(projections):
    """Preserve a conflict even when the other calibrated camera supports it."""
    support = np.column_stack([p['supported'] for p in projections])
    conflict = np.column_stack([p['conflict'] for p in projections])
    known, contradicted = support.any(1), conflict.any(1)
    return dict(support=known, conflict=contradicted, unknown=~(known | contradicted),
        view_support=support, view_conflict=conflict,
        view_states=np.column_stack([p['states'] for p in projections]),
        state_names=list(geometry.STATE_NAMES))


def _correspondence_geometry(source, target, transform):
    """Condition diagnostics for independently assigned actual target points.

    Ranks alone cannot identify an optimal rotation for inconsistent matches.
    The cross-covariance singular values and reflected-SVD gap detect algebraic
    ambiguity; information eigenvalues describe local sensitivity only, not a
    calibrated pose probability. Empty/insufficient correspondences stay so.
    """
    result = dict(count=len(source), source_rank=rank(source)[0], target_rank=rank(target)[0],
                  covariance_rank=0, rotation_algebraically_identifiable=False)
    if len(source) < 3:
        return result
    x, y = source-source.mean(0), target-target.mean(0)
    covariance = y.T@x/len(x)
    u, singular, vt = np.linalg.svd(covariance)
    maximum = float(singular[0])
    covariance_rank = int(np.count_nonzero(singular > maximum*1e-6)) if maximum > 0 else 0
    correction = 1. if np.linalg.det(u@vt) >= 0 else -1.
    rotation_gap = singular[1]+correction*singular[2]
    identifiable = covariance_rank >= 2 and rotation_gap > maximum*1e-6
    eigen = np.linalg.eigvalsh(x.T@x)[::-1]
    information = transform[0]**2*(eigen.sum()-eigen)
    residual = np.linalg.norm(apply_sim3(source, *transform)-target, axis=1)
    result.update(covariance_rank=covariance_rank,
        covariance_singular_values=singular.tolist(),
        rotation_svd_gap=float(rotation_gap), rotation_algebraically_identifiable=bool(identifiable),
        centered_position_eigenvalues_m2=eigen.tolist(),
        rotation_information_eigenvalues_m2=information.tolist(),
        weakest_to_strongest_rotation_information=(float(information.min()/information.max())
                                                    if information.max() > 0 else None),
        residuals_m=residual.tolist(), rms_residual_m=float(np.sqrt(np.mean(residual**2))),
        maximum_residual_m=float(residual.max()), uncertainty_is_calibrated=False)
    return result


def _region_summary(mode):
    if mode is None:
        return None
    return dict(seed_index=mode['seed_index'], center=mode['center'].tolist(), normal=None,
        region_indices=np.flatnonzero(mode['mask']).tolist(),
        sample_indices=mode['sample_indices'].tolist(), diagnostics=dict(mode['diagnostics']))


def _pose_differences(ranked):
    """Report raw differences from the selected estimate, without a gate."""
    reference = ranked[0]['transform']
    records = []
    for item in ranked:
        scale, rotation, translation = item['transform']
        relative = rotation@reference[1].T
        angle = np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1., 1.)))
        records.append(dict(hypothesis_id=item['summary']['hypothesis_id'],
            rotation_difference_deg=float(angle), scale=float(scale),
            translation_difference_m=float(np.linalg.norm(translation-reference[2]))))
    return records


def _incomplete_result(result, error, stage):
    """Clear every actionable field when the declared search did not finish."""
    result.update(scheme='dual_view_observed_surface', region_geometry='surface',
        status='search_incomplete', accepted=False, modes=[],
        reasons=['declared_finite_search_interrupted'], region_identified=False,
        finite_candidate_unique=False, transform_certified=False,
        src_anchor_idx=np.empty(0, np.int64), tgt_anchor_idx=np.empty(0, np.int64),
        sim3_scale=None, sim3_rotation=None, sim3_translation=None)
    diagnostics = result.setdefault('diagnostics', {})
    diagnostics.update(search_status='search_incomplete', search_stage=stage,
        decision_status='search_incomplete', interruption=str(error),
        region_identified=False, finite_candidate_unique=False, transform_certified=False,
        ranked_hypothesis_ids=[], top_score_hypothesis_ids=[], selected_hypothesis_id=None)
    result['diagnostic_hypotheses'] = diagnostics.setdefault('hypotheses', [])
    for hypothesis in result['diagnostic_hypotheses']:
        hypothesis['selected'] = False
    return result


def evaluate_candidates(source, surface, appearance, proposals, *,
                        sampling_radius_m=.03, sampling_spacing_m=.005):
    """Evaluate each supplied pose once, without candidate/refit expansion.

    Geometric association, position scoring and full-entity refutation use
    actual observed points. Appearance uses fixed per-view reference domains.
    The fixed-denominator score ranks measured positive support; it is not a
    probability or a claim that missing evidence uniquely determines a pose.
    """
    if (source.get('region_geometry') != 'surface' or surface.get('region_geometry') != 'surface'
            or not isinstance(appearance, SurfaceResponseField) or appearance.surface is not surface):
        raise ValueError('Curved evaluation requires the explicit curved source, surface and response field')
    started = perf_counter()
    q = len(source['query_xyz'])
    regional = np.asarray(source['query_region'], bool)
    warnings = ['finite_search_not_global', 'scale_rotation_not_certified',
                'surface_normals_unavailable', 'observed_graph_not_continuous_topology_certificate']
    if not source['check_region'].any():
        warnings.append('no_independent_source_region_check_points')
    diagnostics = dict(method='dual_view_observed_surface', scheme='dual_view_observed_surface',
        region_geometry='surface', target_truth_used=False, source=source['diagnostics'],
        query_indices=source['query_indices'].tolist(), query_region=regional.tolist(),
        query_count=q, region_query_count=int(regional.sum()),
        context_query_count=int((~regional).sum()), quality_warnings=warnings,
        view_ids=list(surface['view_ids']), point_index_domain='surface.xyz',
        # BEGIN the score is changed, not just its final selection rule.
        score_definition='source_scope_weighted_absolute_geometry_and_empirical_appearance',
        appearance_score_definition='assigned_measurement_own_view_positive_centered_mid_cdf',
        raw_appearance_diagnostic_definition='per_view_native_dot_responses_and_maximum_for_display',
        # END raw maxima are diagnostic and never borrowed by the scorer.
        observation_model=surface['diagnostics']['observation_model'],
        local_geometry_contract='actual_region_and_complete_observed_entity_measurements',
        sampling_metric='observed_graph_path_length', sampling_radius_m=sampling_radius_m,
        sampling_spacing_m=sampling_spacing_m, sampling_scale_applied=False,
        global_optimality_certified=False, transform_certified=False,
        region_identified=False, finite_candidate_unique=False,
        proposal_scope='explicit_caller_supplied_candidates', hypotheses=[],
        search_status='running', search_stage='candidate_evaluation')
    result = dict(scheme='dual_view_observed_surface', region_geometry='surface', accepted=False,
        status='not_started', reasons=[], modes=[], diagnostics=diagnostics,
        src_anchor_idx=np.empty(0, np.int64), tgt_anchor_idx=np.empty(0, np.int64),
        sim3_scale=None, sim3_rotation=None, sim3_translation=None,
        region_identified=False, finite_candidate_unique=False, transform_certified=False,
        diagnostic_hypotheses=diagnostics['hypotheses'])
    if source['reasons']:
        result.update(status='missing_input', reasons=list(source['reasons']))
        diagnostics.update(search_status='not_started', decision_status=result['status'])
        return result
    # BEGIN freeze source-only domains once for this entire finite evaluation.
    scoring_context = prepare_scoring(source)
    records, seen = [], {}
    # END no candidate can change the source scopes or query weights.

    def evaluate(transform, origins):
        key = transform_key(transform)
        if key in seen:
            records[seen[key]]['summary']['origins'].extend(origins)
            return
        if not valid_transform(transform):
            raise ValueError('Candidate is outside the declared proper Sim(3) domain')
        # This is the existing finite correspondence-compatibility budget,
        # expressed in target metres. It is not a depth noise estimate and is
        # kept separate from pixel discretization and depth measurement error.
        registration_error = COMPATIBILITY_RATIO*source['size']*transform[0]
        mapped = apply_sim3(source['query_xyz'], *transform)
        sampled = appearance.sample(mapped, np.full_like(mapped, np.nan), np.arange(q),
            registration_error_m=registration_error,
            query_position_error_m=transform[0]*source['query_position_error_m'])
        support = combine_support(sampled['geometry'])
        check_points = apply_sim3(source['check_xyz'], *transform)
        checks = combine_support([geometry.project_measurements(surface, check_points, i,
            registration_error_m=registration_error,
            query_position_error_m=transform[0]*source['check_position_error_m']) for i in range(2)])
        # BEGIN one appearance-independent assignment drives every geometry use.
        # Region rows have lexicographic capacity priority, followed by total
        # matched count and actual residual. A second assignment could reuse a
        # target element and make admission disagree with the scored evidence.
        assigned = geometry_correspondences(surface, sampled['geometry'], regional)
        rows = np.flatnonzero(assigned['matched'])
        matched_views = assigned['selected_view'][rows]
        actual_ids = assigned['observation_indices'][rows]
        region_rows = np.flatnonzero(regional & assigned['matched'])
        region_views = assigned['selected_view'][region_rows]
        region_observations = assigned['observation_indices'][region_rows]
        correspondence = _correspondence_geometry(source['query_xyz'][region_rows],
            assigned['target_xyz'][region_rows], transform)
        # BEGIN only the assigned observation may contribute appearance.
        assigned_features = assigned_appearance(sampled, assigned, supports)
        scored = score_candidate(scoring_context, transform, assigned,
            assigned_features['utilities'], registration_error_m=registration_error)
        # END geometry does not depend on feature availability or response rank.
        # END unknown feature footprints cannot delete real geometric evidence.
        local_rows = np.flatnonzero(regional & support['support'])
        rejections = []
        if np.any(support['conflict'] & regional):
            rejections.append('region_query_depth_model_conflict')
        if np.any(checks['conflict'] & source['check_region']):
            rejections.append('region_check_depth_model_conflict')
        if len(local_rows) < 6:
            rejections.append('fewer_than_six_region_position_supports')
        if len(region_rows) < 6:
            rejections.append('fewer_than_six_capacity_distinct_region_correspondences')
        if min(correspondence['source_rank'], correspondence['target_rank']) < 2:
            rejections.append('supported_correspondence_rank_below_two')
        if not correspondence['rotation_algebraically_identifiable']:
            rejections.append('correspondence_rotation_not_identifiable')
        # BEGIN: entity refutation is part of identity, not a quality warning.
        if np.any(support['conflict'] & ~regional):
            rejections.append('context_query_depth_model_conflict')
        if np.any(checks['conflict'] & ~source['check_region']):
            rejections.append('context_check_depth_model_conflict')
        structure = dict(status='not_evaluated_after_query_or_check_rejection')
        if not rejections:
            structure = entity_evidence(
                source, surface, transform, registration_error)
            if structure['conflict_count']:
                rejections.append('observed_entity_depth_model_conflict')
        # A demonstration may cover the entire observed entity. Independent
        # region capacity/rank still apply; requiring extra outside-region
        # points would reject that valid input solely because of its annotation.
        # END unknown entity rows neither refute nor supply positive matches.
        mode, region_reason = geometry.region_for_transform(surface, source, transform,
            registration_error_m=registration_error, sampling_radius_m=sampling_radius_m,
            sampling_spacing_m=sampling_spacing_m)
        # BEGIN: output availability must not reorder identity candidates.
        if mode is not None and (int(mode['mask'].sum()) < 6 or rank(surface['xyz'][mode['mask']])[0] < 2):
            region_reason = region_reason or 'insufficient_observed_region_geometry'
        # END: the chosen estimate's sampling gate is applied after selection.
        context_conflict = int(np.count_nonzero(support['conflict'] & ~regional))
        context_check_conflict = int(np.count_nonzero(checks['conflict'] & ~source['check_region']))
        context_unknown = int(np.count_nonzero(support['unknown'] & ~regional))
        context_check_unknown = int(np.count_nonzero(checks['unknown'] & ~source['check_region']))
        # BEGIN full-entity diagnostics supersede the sparse context summary.
        # A conflict or unknown measurement outside the query/check samples is
        # still evidence about this entity; do not label that result consistent.
        if structure['status'] == 'evaluated':
            entity_state = ('observed_conflict' if structure['conflict_count'] else
                            'partially_observed' if structure['unknown_count'] else 'observed_consistent')
        else:
            entity_state = ('observed_conflict' if context_conflict or context_check_conflict else
                            'partially_observed' if context_unknown or context_check_unknown else 'observed_consistent')
        # END preserve the sparse summary only when complete geometry was skipped.
        hypothesis_id = len(records)
        summary = dict(hypothesis_id=hypothesis_id, origins=origins,
            transform=[float(transform[0]), transform[1].tolist(), transform[2].tolist()],
            # BEGIN expose every input and weighted contribution of the new score.
            score=scored['score'], score_components=scored,
            assigned_appearance_responses=_finite_list(assigned_features['raw_responses']),
            assigned_appearance_known_count=int(np.isfinite(assigned_features['utilities']).sum()),
            assigned_appearance_unknown_count=int((~np.isfinite(assigned_features['utilities'])).sum()),
            # END per-view raw diagnostics below have their original meaning.
            appearance_scores=_finite_list(sampled['scores']),
            appearance_known=sampled['known'].tolist(), appearance_known_count=int(sampled['known'].sum()),
            appearance_unknown_count=int((~sampled['known']).sum()),
            appearance_view_scores=_finite_list(sampled['view_scores']),
            appearance_view_valid=sampled['view_valid'].tolist(), appearance_view_states=sampled['view_state'].tolist(),
            appearance_reason_counts=sampled['reason_counts'],
            geometry_region_matched_count=len(region_rows),
            geometry_capacity_lost_count=assigned['capacity_lost_count'],
            geometry_matched_count=assigned['matched_count'],
            geometry_unassigned_count=assigned['unknown_count'],
            regional_assignment_query_rows=region_rows.tolist(),
            regional_assignment_observation_indices=region_observations.tolist(),
            regional_assignment_view_indices=region_views.tolist(),
            paired_query_rows=rows.tolist(), paired_surface_indices=assigned['element_ids'][rows].tolist(),
            paired_observation_indices=actual_ids.tolist(), paired_view_indices=matched_views.tolist(),
            correspondence_geometry=correspondence, registration_error_budget_m=float(registration_error),
            geometry=_support_summary(support), checks=_support_summary(checks),
            per_view_position_residuals_m=[_finite_list(p['residual_m']) for p in sampled['geometry']],
            per_view_association_bounds_m=[_finite_list(p['association_bound_m']) for p in sampled['geometry']],
            per_view_observation_indices=[p['observation_indices'].tolist() for p in sampled['geometry']],
            region_geometry_count=len(local_rows), entity_model_status=entity_state,
            context_query_conflict_count=context_conflict, context_check_conflict_count=context_check_conflict,
            context_query_unknown_count=context_unknown, context_check_unknown_count=context_check_unknown,
            orientation_state='unknown', mapped_center=apply_sim3(source['center'], *transform).tolist(),
            mapped_normal=None, center_used_as_contact_measurement=False,
            entity_structure=structure,
            admitted_to_H=not rejections, rejections=rejections,
            output_available=mode is not None and region_reason is None,
            output_reason=region_reason, selected=False, region=_region_summary(mode))
        records.append(dict(transform=transform, summary=summary, mode=mode))
        diagnostics['hypotheses'].append(summary)
        seen[key] = hypothesis_id

    try:
        # BEGIN build candidate-independent reference distributions once.
        diagnostics['search_stage'] = 'appearance_reference_preparation'
        supports, reference_records = appearance.prepare_support()
        diagnostics.update(appearance_reference=reference_records,
            source_query_weights=scoring_context['query_weights'].tolist(),
            source_weight_sum=float(scoring_context['query_weights'].sum()),
            source_support_scopes=scoring_context['scope_aliases'],
            source_weighting=scoring_context['weighting_definition'])
        diagnostics['search_stage'] = 'candidate_evaluation'
        # END interruption cannot publish a candidate with a partial reference.
        for proposal in proposals:
            evaluate(proposal['transform'], list(proposal['origins']))
        admitted = [r for r in records if r['summary']['admitted_to_H']]
        diagnostics.update(evaluated_transform_count=len(records), hypotheses_admitted=len(admitted))
        # BEGIN select by the weighted geometry/appearance score, without an
        # interval or cross-candidate agreement gate. Missing observations supply
        # no positive score, but remain unknown rather than depth conflicts.
        # Stable sorting preserves proposal order for exact numerical ties;
        # there is no task-specific tie key, near-tie threshold or fallback.
        if not admitted:
            result.update(status='rejected_no_supported_structure',
                          reasons=['no_structurally_supported_hypothesis_in_declared_search'])
        else:
            ranked = sorted(admitted, key=lambda r: -r['summary']['score'])
            selected = ranked[0]
            best = selected['summary']['score']
            top = [r['summary']['hypothesis_id'] for r in ranked if r['summary']['score'] == best]
            selected['summary']['selected'] = True
            # This flag describes the entire geometrically admitted pool,
            # not just the number of exact score maximizers. Neither meaning
            # would establish uniqueness outside this finite search.
            unique = len(admitted) == 1
            diagnostics.update(selected_score=best,
                ranked_hypothesis_ids=[r['summary']['hypothesis_id'] for r in ranked],
                top_score_hypothesis_ids=top, finite_candidate_unique=unique,
                finite_candidate_unique_scope='only_one_geometrically_admitted_finite_candidate',
                ranked_candidate_pose_differences=_pose_differences(ranked),
                selected_hypothesis_id=selected['summary']['hypothesis_id'],
                selection_rule='highest_weighted_evidence_score_stable_proposal_order')
            if len(top) > 1:
                warnings.append('exact_top_score_tie_resolved_by_proposal_order')
            result.update(region_identified=True, finite_candidate_unique=unique)
            diagnostics['region_identified'] = True
            primary = None
            if selected['summary']['output_available']:
                primary = dict(selected['mode'])
            else:
                result.update(status='identified_region_sampling_unavailable',
                    reasons=[selected['summary']['output_reason'] or 'observed_region_unavailable'])
            if primary is not None:
                primary.update(type='primary', score=best,
                    hypothesis_indices=[selected['summary']['hypothesis_id']],
                    hypothesis_index=selected['summary']['hypothesis_id'])
                result.update(accepted=True, status='accepted_observed_region', modes=[primary],
                    region_identified=True, finite_candidate_unique=unique)
                chosen, transform = selected['summary'], selected['transform']
                result.update(sim3_scale=transform[0], sim3_rotation=transform[1], sim3_translation=transform[2],
                    src_anchor_idx=source['query_indices'][chosen['paired_query_rows']],
                    tgt_anchor_idx=np.asarray(chosen['paired_surface_indices'], np.int64))
                if chosen['entity_structure']['unknown_count']:
                    warnings.append('entity_transform_not_fully_supported')
                if chosen['assigned_appearance_unknown_count']:
                    warnings.append('selected_estimate_has_unknown_appearance')
        # END the chosen score estimate alone supplies the actionable region.
        if not result['accepted']:
            available = sorted((r for r in records if r['mode'] is not None),
                key=lambda r: (not r['summary']['admitted_to_H'], -r['summary']['score'], r['summary']['hypothesis_id']))
            result['modes'] = [dict(r['mode'], type='rejected_diagnostic',
                hypothesis_index=r['summary']['hypothesis_id'], score=r['summary']['score']) for r in available[:6]]
        diagnostics.update(search_status='finite_domain_complete', search_stage='complete')
    except (MemoryError, InterruptedError) as error:
        _incomplete_result(result, error, diagnostics['search_stage'])
    diagnostics.update(decision_status=result['status'], total_seconds=perf_counter()-started)
    return result


# BEGIN one bounded native global-to-local surface candidate path.
def infer(source, surface, appearance, *, sampling_radius_m=.03,
          sampling_spacing_m=.005, work_stats=None):
    """Generate at most 32 terminal poses, then run the measured evaluator.

    The proposer replaces its own coarse estimate with its refined estimate
    in each slot. Neither native MARCO initializers nor partial-refit branches
    can add poses here. The evaluator freezes per-view appearance references
    and source weights once before scoring these same terminal candidates.
    """
    if source['reasons']:
        return evaluate_candidates(source, surface, appearance, (),
            sampling_radius_m=sampling_radius_m, sampling_spacing_m=sampling_spacing_m)
    try:
        proposals, report = build_proposals(source, surface)
    except (MemoryError, InterruptedError) as error:
        failed = dict(diagnostics=dict(region_geometry='surface',
            query_count=len(source['query_xyz']), source=source['diagnostics'],
            query_indices=source['query_indices'].tolist(),
            query_region=source['query_region'].tolist(), candidate_budget=CANDIDATE_BUDGET,
            quality_warnings=['declared_finite_search_interrupted']))
        return _incomplete_result(failed, error, 'proposal_generation')
    result = evaluate_candidates(source, surface, appearance, proposals,
        sampling_radius_m=sampling_radius_m, sampling_spacing_m=sampling_spacing_m)
    result['diagnostics'].update(proposals=report, candidate_budget=CANDIDATE_BUDGET,
        proposal_scope='bounded_native_global_to_local_terminal_slots')
    if work_stats is not None:
        work_stats.update(proposals=report, geometry_backend='numpy_scipy',
            observation_footprint_cache_entries=len(appearance._footprint_cache),
            observation_footprint_cache_bytes=appearance._footprint_cache_bytes)
    return result
# END the finite candidate budget includes every complete evaluated pose.
