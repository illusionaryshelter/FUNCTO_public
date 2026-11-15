"""Build native appearance evidence in one explicit observed element domain.

Inputs are a version-2 skill loaded by ``load_grounding_skill``, exactly two
ordered calibrated views, and the declared RGB-D observation model.  The
source region and frame remain legitimate skill representation.  Current
geometry comes from explicit owner observation banks and depth images, while
the selected reference banks define appearance support.  A caller may supply
a previously constructed owner surface to replay its exact element identities;
otherwise this component builds that surface once.

Outputs are the original numerical arrays, native per-view response records,
the caller's context, and evidence metadata.  Source positions use the source
capture world; current positions use the calibrated current world.  Distances
and optical depths are metres.  View counts normalize appearance only; each
target element contributes one geometric sample.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from transfer.dual_view.curved_geometry import ObservationModel, build_target_surface
from transfer.dual_view.features import prepare_view, source_descriptors

from .response import query_views


def query_arrays(query, prefix):
    """Flatten a completed native query without changing rows or unknowns.

    The point axis is shared across ``response``, the Boolean support arrays,
    and all per-view measurement/pixel interpolation provenance.  Unknown
    raw responses retain NaN in the numerical archive.
    """
    output = {prefix+'_'+key: query[key] for key in
        ('response', 'valid', 'unknown', 'world_points', 'view_response', 'view_valid', 'view_count')}
    for key in ('reason', 'projected_pixels', 'optical_depth_m', 'depth_interval_m',
                'depth_pixel_xy', 'depth_weights', 'observation_ids',
                'depth_footprint_supported', 'owner_supported', 'slope_supported',
                'feature_xy', 'feature_corner_xy', 'feature_weights'):
        output[prefix+'_view_'+key] = np.stack([entry[key] for entry in query['views']], axis=1)
    return output


def prepare_evidence(skill, views, *, observation_model, context=None,
                     surface=None, sampling_radius_m=.03):
    """Return ``arrays, prepared_views, context, metadata`` for region fitting.

    ``skill['source']`` supplies ``observation``, ``feature``, ``crop_xyxy``,
    ``T_world_body_at_image`` and ``L_body_region``.  Every view supplies its
    ``id``, ``depth_m``, ``K``, ``T_world_camera_cv``, ``observation``, native
    FP32 ``feature`` and the exact feature producer's ``crop_xyxy``.  These
    fields describe the selected reference.  ``owner_observation``,
    ``owner_mask``, ``owner_evidence`` and
    ``background_mask`` separately identify the declared physical owner and
    observed background.  Object-scope inputs can share one bank; a component
    reference keeps its own mask and crop while geometry uses its owner bank.

    ``owner_evidence`` contains ``owner_id``, ``scope``, ``source``,
    ``reference_material_owner_id`` and ``reference_material_source``.  Equal
    declared owner IDs admit the actual reference measurements as local
    material observations of that owner, including measurements outside an
    independent owner mask.  All original masks and surface memberships stay
    intact; exact element IDs form the owner's unique observed input domain.

    ``observation_model`` is the existing ``ObservationModel`` or its explicit
    field mapping.  A supplied surface contains the original observation to
    element association.  Its stored ``view_ids`` must match this view order:
    reordering only the camera inputs would change measurement identities.
    ``context`` records provenance and prerequisite/start state supplied by
    the public grounding entry; this numerical component does not infer them.
    """
    # BEGIN two real views and distinct reference/owner domains precede arrays.
    views = list(views)
    if (len(views) != 2 or any(not isinstance(view['id'], str) or not view['id'] for view in views)
            or len({view['id'] for view in views}) != 2):
        raise ValueError('Cross-instance evidence requires exactly two distinct nonempty view IDs')
    source = skill['source']
    bank = source['observation']
    source_ids = np.flatnonzero(bank['region_mask'])
    source_frame = source['T_world_body_at_image'] @ source['L_body_region']
    source_points = bank['xyz'][source_ids]
    source_pixels = bank['pixels'][source_ids]
    source_owner = bank['xyz'][bank['entity_member']]
    descriptors = source_descriptors(bank, source_ids, source['feature'], source['crop_xyxy'])
    model = (observation_model if isinstance(observation_model, ObservationModel)
             else ObservationModel(**observation_model))
    model_values = asdict(model)
    if surface is None:
        geometry_views = [dict(view, observation=view['owner_observation']) for view in views]
        surface = build_target_surface(geometry_views, sampling_radius_m=sampling_radius_m, model=model)
    if list(surface['view_ids']) != [view['id'] for view in views]:
        raise ValueError('Surface observation identities require the same ordered view IDs')

    observation_pixels = surface['observation_pixels']
    observation_views = surface['observation_view_index']
    # The same complete depth surface supplies both domains.  Reference
    # membership reads each camera's actual selected mask at original measured
    # pixels.  Owner membership remains the explicit geometry-bank membership;
    # neither mask is merged, clipped or substituted for the other.
    reference_member = np.zeros(len(observation_pixels), dtype=bool)
    local_material_member = np.zeros(len(observation_pixels), dtype=bool)
    for index, view in enumerate(views):
        ids = np.flatnonzero(observation_views == index)
        pixels = observation_pixels[ids]
        reference_member[ids] = view['observation']['object_mask'][pixels[:, 1], pixels[:, 0]]
        # BEGIN membership follows the explicit model declaration, not a name
        # or geometric proximity.  Local material keeps its original measured
        # identity even when the independent owner segmentation omitted it.
        evidence = view['owner_evidence']
        material_owner = evidence['reference_material_owner_id']
        if isinstance(material_owner, str) and material_owner and material_owner == evidence['owner_id']:
            local_material_member[ids] = reference_member[ids]
        # END an unassociated reference still retains its complete target rows.
    observation_ids = np.flatnonzero(reference_member)
    target_ids, observation_elements = np.unique(
        surface['observation_to_element'][observation_ids], return_inverse=True)
    target_points = surface['xyz'][target_ids]
    independent_owner_observations = np.flatnonzero(surface['observation_entity_member'])
    local_material_observations = np.flatnonzero(local_material_member)
    independent_owner_elements = np.unique(
        surface['observation_to_element'][independent_owner_observations])
    local_material_elements = np.unique(
        surface['observation_to_element'][local_material_observations])
    owner_observations = np.unique(np.r_[independent_owner_observations, local_material_observations])
    owner_element_ids, owner_observation_elements = np.unique(
        surface['observation_to_element'][owner_observations], return_inverse=True)
    target_owner = surface['xyz'][owner_element_ids]
    # END one surface retains exact original IDs in both explicit domains.
    prepared_views, feature_sources = [], []
    for index, view in enumerate(views):
        identifier = view['id']
        depth = np.asarray(view['depth_m'])
        reference_mask = np.asarray(view['observation']['object_mask'])
        crop = np.asarray(view['crop_xyxy'])
        lookup = np.full(depth.shape, -1, dtype=np.int64)
        view_observations = np.flatnonzero(observation_views == index)
        uv = observation_pixels[view_observations]
        lookup[uv[:, 1], uv[:, 0]] = view_observations
        feature = view['feature']
        prepared = prepare_view(descriptors, feature, crop_xyxy=crop,
                                observation=None, view_id=identifier)
        prepared_views.append(dict(id=identifier, K=np.asarray(view['K']),
            T_world_camera_cv=np.asarray(view['T_world_camera_cv']),
            depth_m=depth, owner_mask=reference_mask, observation_ids=lookup,
            response=prepared['response'], crop_xyxy=crop))
        feature_sources.append(dict(view_id=identifier,
            crop_xyxy=crop, feature_bytes=feature.nbytes,
            response_bytes=prepared['response'].nbytes))

    query = query_views(target_points, prepared_views, source_row_ids=source_ids,
                        **model_values)
    # The optimized target domain consists of actual reference measurements whose
    # native image crop supplies appearance.  A missing target input retains
    # its full query evidence; it is not replaced by a zero response or by a
    # duplicated camera.  Predicted maps may later have unsupported positions.
    arrays = dict(source_row_ids=source_ids, source_points_world_m=source_points,
        source_pixels=source_pixels, source_frame_world=source_frame,
        # BEGIN the declared owner observation domain keeps its own element order.
        source_owner_points_world_m=source_owner, target_owner_points_world_m=target_owner,
        target_owner_element_ids=owner_element_ids,
        independent_owner_element_ids=independent_owner_elements,
        independent_owner_observation_ids=independent_owner_observations,
        independent_owner_observation_element_ids=surface['observation_to_element'][independent_owner_observations],
        local_material_element_ids=local_material_elements,
        local_material_observation_ids=local_material_observations,
        local_material_observation_element_ids=surface['observation_to_element'][local_material_observations],
        local_material_observation_pixels=observation_pixels[local_material_observations],
        local_material_observation_view_indices=observation_views[local_material_observations],
        owner_observation_ids=owner_observations,
        owner_observation_element_indices=owner_observation_elements,
        owner_observation_pixels=observation_pixels[owner_observations],
        owner_observation_view_indices=observation_views[owner_observations],
        owner_element_observation_counts=np.bincount(owner_observation_elements, minlength=len(owner_element_ids)),
        # END grasp steering maps reference assignments by these exact IDs.
        target_element_ids=target_ids, target_points_world_m=target_points,
        observation_ids=observation_ids,
        observation_element_indices=observation_elements,
        observation_view_indices=observation_views[observation_ids],
        observation_pixels=observation_pixels[observation_ids],
        element_observation_counts=np.bincount(observation_elements, minlength=len(target_ids)),
        fused_raw_response=query['response'], descriptor_channels=np.asarray(descriptors.shape[1]),
        view_ids=np.asarray(surface['view_ids']))
    arrays.update(query_arrays(query, 'target_native'))
    metadata = dict(method='same_world_native_response_valid_view_mean',
        source_rows=len(source_ids), target_elements=len(target_ids),
        target_owner_elements=len(owner_element_ids),
        independent_owner_elements=len(independent_owner_elements),
        local_material_elements=len(local_material_elements),
        observation_rows=len(observation_ids), descriptor_channels=descriptors.shape[1],
        target_feature_sources=feature_sources,
        two_view_points=int(np.count_nonzero(query['view_count'] == 2)),
        observation_model=model_values,
        owner_evidence=[dict(view_id=view['id'], **view['owner_evidence']) for view in views],
        surface_geometry_scope='declared owner observation banks; reference masks retain their separate measured domain',
        target_measure='one element per recorded geometric association; appearance views do not add geometric mass',
        status=('completed' if query['valid'].all() and np.isfinite(query['response']).all()
                else 'missing_observed_target_appearance'))
    return arrays, prepared_views, dict(context or {}), metadata
