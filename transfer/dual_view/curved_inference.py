"""Finite point-correspondence inference with explicit curved observations.

The shared appearance proposals and score are retained, but surface admission
also checks the complete observed entity. Identity, finite-pool disagreement,
and availability of the frozen sampling anchor have separate decisions.
"""

from time import perf_counter

import numpy as np

from ..sim3 import apply_sim3, rank
from . import curved_geometry as geometry
from .curved_features import SurfaceResponseField
from .features import capacity_score
from .inference import _finite_list, _support_summary
from .proposals import COMPATIBILITY_RATIO, build_proposals, transform_key, valid_transform
from .refinement import partial_refits
# BEGIN: surface-only structural evidence; the planar evaluator is unchanged.
from .curved_structure import entity_evidence, compare_transforms
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


def _pose_differences(winners):
    """Report every finite best candidate's rotation relative to the first."""
    reference = winners[0]['transform']
    records = []
    for item in winners:
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
        potential_winner_ids=[], potential_winner_count=0)
    result['diagnostic_hypotheses'] = diagnostics.setdefault('hypotheses', [])
    for hypothesis in result['diagnostic_hypotheses']:
        hypothesis['potential_winner'] = False
    return result


def evaluate_candidates(source, surface, appearance, proposals, domain, *,
                        sampling_radius_m=.03, sampling_spacing_m=.005):
    """Evaluate an explicit finite proposal list and its existing partial refits.

    This numerical boundary also supports a transparent native-candidate-only
    experiment. It does not claim to have generated proposals itself. ``infer``
    below supplies the complete shared 128-growth/256-fit proposal stream.
    """
    if (source.get('region_geometry') != 'surface' or surface.get('region_geometry') != 'surface'
            or not isinstance(appearance, SurfaceResponseField) or appearance.surface is not surface):
        raise ValueError('Curved evaluation requires the explicit curved source, surface and response field')
    binding = domain.get('input_binding', ())
    if len(binding) != 3 or not all(a is b for a, b in zip(binding, (source, surface, appearance))):
        raise ValueError('Prepared response domain belongs to different observed inputs')
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
        fixed_query_denominator=q, query_count=q, region_query_count=int(regional.sum()),
        context_query_count=int((~regional).sum()), quality_warnings=warnings,
        view_ids=list(surface['view_ids']), point_index_domain='surface.xyz',
        score_definition='fixed_Q_capacity_one_positive_centered_target_domain_mid_cdf',
        appearance_reference=domain['empirical_support'].diagnostics,
        observation_model=surface['diagnostics']['observation_model'],
        local_geometry_contract='actual_region_and_complete_observed_entity_measurements',
        sampling_metric='observed_graph_path_length', sampling_radius_m=sampling_radius_m,
        sampling_spacing_m=sampling_spacing_m, sampling_scale_applied=False,
        global_optimality_certified=False, transform_certified=False,
        region_identified=False, finite_candidate_unique=False,
        proposal_scope='explicit_caller_supplied_candidates', hypotheses=[], refinement_records=[],
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
    records, seen = [], {}

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
        utility = domain['empirical_support'].utility(sampled['view_scores'], np.arange(q)[:, None])
        elements = np.full((q, 2), -1, np.int64)
        for view in range(2):
            ids = sampled['geometry'][view]['observation_indices']
            usable = np.isfinite(utility[:, view]) & (ids >= 0)
            elements[usable, view] = surface['observation_to_element'][ids[usable]]
        evidence = capacity_score(utility, elements)
        rows = np.flatnonzero(evidence['matched'])
        matched_views = evidence['selected_view'][rows]
        actual_ids = np.array([sampled['geometry'][view]['observation_indices'][row]
                              for row, view in zip(rows, matched_views)], np.int64)
        # BEGIN: local admission uses a separate regional capacity assignment.
        # A tie with an entity-context row in the fixed-Q scoring assignment
        # must not remove a valid regional correspondence from the rank gate.
        regional_rows = np.flatnonzero(regional)
        region_evidence = capacity_score(utility[regional], elements[regional])
        region_rows = regional_rows[region_evidence['matched']]
        region_views = region_evidence['selected_view'][region_evidence['matched']]
        region_observations = np.array([sampled['geometry'][view]['observation_indices'][row]
            for row, view in zip(region_rows, region_views)], np.int64)
        correspondence = _correspondence_geometry(source['query_xyz'][region_rows],
            surface['observation_xyz'][region_observations], transform)
        # END: regional admission assignment; the full fixed-Q score is unchanged.
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
            structure, context_source, context_target = entity_evidence(
                source, surface, transform, registration_error)
            structure['correspondence_geometry'] = _correspondence_geometry(
                context_source, context_target, transform)
            if structure['conflict_count']:
                rejections.append('observed_entity_depth_model_conflict')
            if len(context_source) < 6:
                rejections.append('fewer_than_six_distinct_context_measurements')
            context_geometry = structure['correspondence_geometry']
            if min(context_geometry['source_rank'], context_geometry['target_rank']) < 2:
                rejections.append('context_correspondence_rank_below_two')
            if not context_geometry['rotation_algebraically_identifiable']:
                rejections.append('context_correspondence_rotation_not_identifiable')
        # The six-point rule is the existing minimum fitting evidence, now
        # also required outside the region. It is not an identity certificate.
        # END: unknown entity rows neither refute nor supply positive matches.
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
            score=float(evidence['score']), scores=_finite_list(sampled['scores']),
            known=sampled['known'].tolist(), known_count=int(sampled['known'].sum()),
            unknown_count=int((~sampled['known']).sum()),
            view_scores=_finite_list(sampled['view_scores']), view_utilities=_finite_list(utility),
            view_valid=sampled['view_valid'].tolist(), appearance_view_states=sampled['view_state'].tolist(),
            appearance_reason_counts=sampled['reason_counts'],
            row_utilities=evidence['row_utilities'].tolist(), matched_count=len(rows),
            positive_region_matched_count=len(region_rows), capacity_lost_count=evidence['capacity_lost_count'],
            regional_assignment_query_rows=region_rows.tolist(),
            regional_assignment_observation_indices=region_observations.tolist(),
            regional_assignment_view_indices=region_views.tolist(),
            paired_query_rows=rows.tolist(), paired_surface_indices=evidence['element_ids'][rows].tolist(),
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
            output_reason=region_reason, potential_winner=False, region=_region_summary(mode))
        records.append(dict(transform=transform, summary=summary, mode=mode))
        diagnostics['hypotheses'].append(summary)
        seen[key] = hypothesis_id

    try:
        for number, proposal in enumerate(proposals):
            proposal = dict(proposal, proposal_id=number)
            evaluate(proposal['transform'], list(proposal['origins']))
            refined, record = partial_refits(source, surface, domain, proposal)
            diagnostics['refinement_records'].append(record)
            for item in refined:
                evaluate(item['transform'], list(item['origins']))
        admitted = [r for r in records if r['summary']['admitted_to_H']]
        diagnostics.update(evaluated_transform_count=len(records), hypotheses_admitted=len(admitted))
        # BEGIN: all structurally admitted estimates remain plausible. An
        # uncalibrated appearance score orders representatives; even an exact
        # highest score cannot dismiss a spatially incompatible structure.
        if not admitted:
            result.update(status='rejected_no_supported_structure',
                          reasons=['no_structurally_supported_hypothesis_in_declared_search'])
        else:
            winners = sorted(admitted, key=lambda r: (-r['summary']['score'], r['summary']['hypothesis_id']))
            selected = winners[0]
            best = selected['summary']['score']
            for item in admitted:
                item['summary']['potential_winner'] = True
            ids = [r['summary']['hypothesis_id'] for r in winners]
            agreement = compare_transforms(source, winners)
            diagnostics.update(best_score=best, potential_winner_ids=ids,
                potential_winner_count=len(winners), finite_candidate_unique=len(winners) == 1,
                admitted_candidate_pose_differences=_pose_differences(winners),
                finite_structure_agreement=agreement,
                structure_hypothesis_ids=ids,
                selected_hypothesis_id=selected['summary']['hypothesis_id'],
                selection_rule='highest_fixed_Q_score_only_after_all_admitted_structures_agree')
            primary = None
            if not agreement['agree']:
                result.update(status='rejected_ambiguous_structure',
                    reasons=['admitted_entity_transforms_disagree_beyond_declared_budgets'])
            else:
                result.update(region_identified=True, finite_candidate_unique=len(winners) == 1)
                diagnostics['region_identified'] = True
                if selected['summary']['output_available']:
                    primary = dict(selected['mode'])
                    available = [r for r in winners if r['summary']['output_available']]
                    diagnostics['sampling_alternatives'] = [dict(
                        hypothesis_id=r['summary']['hypothesis_id'],
                        available=r['summary']['output_available'],
                        reason=r['summary']['output_reason'],
                        seed_index=None if r['mode'] is None else r['mode']['seed_index']) for r in winners]
                    # Spatial agreement is not graph connectivity. Another
                    # available estimate on a disconnected measured patch
                    # preserves identity but prevents an actionable sampling
                    # choice. Missing anchors supply no alternative support.
                    if any(not primary['mask'][r['mode']['seed_index']]
                           or not r['mode']['mask'][primary['seed_index']] for r in available):
                        primary = None
                        result.update(status='identified_region_sampling_ambiguous',
                            reasons=['consistent_structures_have_disconnected_observed_anchor_regions'])
                    elif len(available) < len(winners):
                        warnings.append('some_consistent_estimates_have_unobserved_sampling_anchors')
                else:
                    result.update(status='identified_region_sampling_unavailable',
                        reasons=[selected['summary']['output_reason'] or 'observed_region_unavailable'])
                if len(winners) > 1:
                    warnings.append('multiple_estimates_agree_within_declared_spatial_budget')
            if primary is not None:
                primary.update(type='primary', score=best,
                    hypothesis_indices=[selected['summary']['hypothesis_id']],
                    hypothesis_index=selected['summary']['hypothesis_id'])
                result.update(accepted=True, status='accepted_observed_region', modes=[primary],
                    region_identified=True, finite_candidate_unique=len(winners) == 1)
                chosen, transform = selected['summary'], selected['transform']
                result.update(sim3_scale=transform[0], sim3_rotation=transform[1], sim3_translation=transform[2],
                    src_anchor_idx=source['query_indices'][chosen['paired_query_rows']],
                    tgt_anchor_idx=np.asarray(chosen['paired_surface_indices'], np.int64))
                if any(r['summary']['entity_structure']['unknown_count'] for r in winners):
                    warnings.append('entity_transform_not_fully_supported')
                if any(r['summary']['unknown_count'] for r in winners):
                    warnings.append('selected_estimate_has_unknown_appearance')
        # END: an unavailable anchor never promotes another identity estimate.
        if not result['accepted']:
            available = sorted((r for r in records if r['mode'] is not None),
                key=lambda r: (not r['summary']['admitted_to_H'], -r['summary']['score'], r['summary']['hypothesis_id']))
            result['modes'] = [dict(r['mode'], type='rejected_diagnostic',
                hypothesis_index=r['summary']['hypothesis_id'], score=r['summary']['score']) for r in available[:6]]
        diagnostics.update(search_status='finite_domain_complete', search_stage='complete')
    except (MemoryError, InterruptedError) as error:
        _incomplete_result(result, error, 'candidate_evaluation')
    diagnostics.update(decision_status=result['status'], total_seconds=perf_counter()-started)
    return result


def infer(source, surface, appearance, *, initial_transforms=(), prepared_domain=None,
          sampling_radius_m=.03, sampling_spacing_m=.005, work_stats=None):
    """Use the same finite point-only CUDA proposal plan, then CPU geometry."""
    work = {}
    try:
        proposals, report, domain = build_proposals(source, surface, appearance,
            initial_transforms=initial_transforms, prepared_domain=prepared_domain, work_stats=work)
    except (MemoryError, InterruptedError) as error:
        if work_stats is not None:
            work_stats.update(proposals=work, geometry_backend='numpy_scipy')
        # Preserve fixed input identity for result writers even if allocation
        # fails before a proposal/domain exists; no hypothetical scores appear.
        failed = dict(diagnostics=dict(region_geometry='surface',
            fixed_query_denominator=len(source['query_xyz']), source=source['diagnostics'],
            query_indices=source['query_indices'].tolist(),
            query_region=source['query_region'].tolist(),
            quality_warnings=['declared_finite_search_interrupted']))
        return _incomplete_result(failed, error, 'proposal_generation')
    result = evaluate_candidates(source, surface, appearance, proposals, domain,
        sampling_radius_m=sampling_radius_m, sampling_spacing_m=sampling_spacing_m)
    result['diagnostics'].update(proposals=report,
        proposal_scope='shared_complete_128_growth_256_fit_plan_and_partial_refits')
    if work_stats is not None:
        work_stats.update(proposals=work, geometry_backend='numpy_scipy',
            observation_footprint_cache_entries=len(appearance._footprint_cache),
            observation_footprint_cache_bytes=appearance._footprint_cache_bytes)
    return result
