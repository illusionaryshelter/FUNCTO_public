"""Coordinate native evidence, coarse correspondence, neural map and task F.

``run_region_transfer`` prepares observations from an already loaded skill;
``run_arrays`` executes the identical numerical chain for prepared evidence.
Both publish all source identities, optimizer states/trials, full target
assignments, native observation support and the task's region frame F.

The output is a dictionary with numerical ``arrays``, native ``views``, input
provenance, evidence metadata and the individual computation reports.  Tool
poses, their motion anchors and complete reference generation consume this
result through separate public components.  A region similarity summarizes
region geometry; it never scales a robot or the metric source tool motion.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.spatial.distance import cdist

from .coarse import register_candidate_context
from .context import orientation_evidence, read_region_frame
from .evidence import prepare_evidence, query_arrays
from .refinement import NeuralRefinementFailure, refine_identity_mapping
from .response import query_views


def run_arrays(arrays, views, *, input_context, input_evidence,
               coarse_steps=40, refinement_steps=200, seed=0):
    """Run original coarse/CPU-float64 refinement on prepared evidence.

    Input arrays use the keys returned by ``prepare_evidence``.  Source rows
    and current element IDs stay in their supplied order.  Coarse registration
    retains 25 original seeds and uses ``coarse_steps`` fits per seed.  Neural
    refinement uses ``min(30, N-1)`` source neighbors, ``refinement_steps``
    L-BFGS blocks and the explicit random seed (original default zero).

    Native queries of coarse and refined predictions are diagnostics: unknown
    response support does not alter correspondence, fitting or frame output.
    Numerical refinement failure retains its partial arrays and report in a
    result with ``status='failed'``.  Input appearance missing from the actual
    target domain retains a ``missing`` result before either numerical fit.
    """
    arrays = dict(arrays)
    views = list(views)
    stage_seconds = {}
    result = dict(status='computing', stage='native_observed_evidence',
        arrays=arrays, views=views, input_context=dict(input_context),
        input_evidence=dict(input_evidence), grounding_stage_seconds=stage_seconds)
    source, target = arrays['source_points_world_m'], arrays['target_points_world_m']
    source_owner = arrays['source_owner_points_world_m']
    target_owner = arrays['target_owner_points_world_m']
    source_frame, raw = arrays['source_frame_world'], arrays['fused_raw_response']
    result.update(source_rows=len(source), target_elements=len(target), source_frame=source_frame)
    if not arrays['target_native_valid'].all() or not np.isfinite(raw).all():
        result.update(status='missing', reason='missing_observed_target_appearance')
        return result

    result['stage'] = 'candidate_center_coarse_registration'
    phase = perf_counter()
    coarse_arrays, coarse = register_candidate_context(source, target, raw,
        source_frame, source_owner, target_owner, steps=coarse_steps,
        channels=int(arrays['descriptor_channels']))
    stage_seconds['coarse_registration'] = perf_counter()-phase
    arrays.update(coarse_arrays)
    selected = coarse['selected_candidate']
    coarse_points = arrays['coarse_mapped_points_world_m'][selected, -1]
    coarse_rotation = arrays['coarse_rotations'][selected, -1]
    coarse_frame = read_region_frame(source, coarse_points, source_frame)
    result.update(coarse=coarse, coarse_frame=coarse_frame)
    arrays.update(coarse_selected_points_world_m=coarse_points,
                  coarse_frame_world=coarse_frame['T_world_region'])

    result['stage'] = 'identity_preserving_neural_refinement'
    phase = perf_counter()
    try:
        refined, refinement = refine_identity_mapping(source, target,
            arrays['appearance_cost'], source_owner, target_owner, coarse_points,
            arrays['source_row_ids'], arrays['target_element_ids'], source_frame,
            coarse_rotation, steps=refinement_steps,
            n_neighbors=min(30, len(source)-1), seed=seed)
    except NeuralRefinementFailure as error:
        stage_seconds['identity_refinement'] = perf_counter()-phase
        arrays.update(error.arrays)
        result.update(status='failed', stage='failed_neural_refinement',
                      reason=str(error), refinement=error.report)
        return result
    arrays.update(refined)
    stage_seconds['identity_refinement'] = perf_counter()-phase
    result['refinement'] = refinement
    mapped = arrays['mapped_points_world_m']
    matching = arrays['target_indices']
    arrays.update(matching_target_indices=matching,
        matching_target_element_ids=arrays['target_element_ids'][matching],
        display_nearest_target_indices=cdist(mapped, target).argmin(axis=1))

    support_summary = {}
    phase = perf_counter()
    for name, points in (('coarse_query', coarse_points), ('mapped_query', mapped)):
        query = query_views(points, views, source_row_ids=arrays['source_row_ids'],
                            **input_evidence['observation_model'])
        arrays.update(query_arrays(query, name))
        per_view = []
        for view in query['views']:
            reasons, counts = np.unique(view['reason'], return_counts=True)
            per_view.append(dict(view_id=view['view_id'],
                                 reason_counts=dict(zip(reasons.tolist(), counts.tolist()))))
        support_summary[name] = dict(points=len(points), observed=int(query['valid'].sum()),
            unknown=int(query['unknown'].sum()), per_view=per_view)
    stage_seconds['mapped_observation_queries'] = perf_counter()-phase
    result['mapped_observation_support'] = support_summary

    result['stage'] = 'mapped_region_and_joint_direction'
    phase = perf_counter()
    frame = read_region_frame(source, mapped, source_frame)
    evidence = orientation_evidence(source, mapped, source_owner, target_owner)
    result.update(status='completed', region_frame=frame, context=evidence,
        refinement_relation_max_matrix_difference=float(np.max(
            np.abs(arrays['rotation_relation']-evidence['rotation_wahba']))))
    arrays['region_frame_world'] = frame['T_world_region']
    stage_seconds['region_frame_and_direction'] = perf_counter()-phase
    return result


def run_region_transfer(skill, views, *, observation_model, context=None,
                        surface=None, sampling_radius_m=.03,
                        coarse_steps=40, refinement_steps=200, seed=0):
    """Prepare two named observations and compute one full region result.

    ``surface`` is an explicit replay input with its original element IDs;
    otherwise the selected cross-instance strategy constructs the observed
    surface exactly once.  ``context`` preserves the public entry's source,
    prerequisite and start-state identities without interpreting task names.
    """
    phase = perf_counter()
    arrays, prepared, provenance, metadata = prepare_evidence(skill, views,
        observation_model=observation_model, context=context, surface=surface,
        sampling_radius_m=sampling_radius_m)
    evidence_seconds = perf_counter()-phase
    result = run_arrays(arrays, prepared, input_context=provenance,
        input_evidence=metadata, coarse_steps=coarse_steps,
        refinement_steps=refinement_steps, seed=seed)
    result['grounding_stage_seconds']['prepare_observed_evidence'] = evidence_seconds
    return result
