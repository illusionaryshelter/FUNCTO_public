"""Cross-instance grasp predictions with task F and tool-reference B separated.

All transforms map homogeneous column coordinates; positions are metres and
angles are radians. Native AnyGrasp rows first produce actual robot tool poses
E. Relative object rotations from UNOPose then change orientation at that same
E origin. Official robot geometry and observed-background TSDF guide the final
local swing. Every view hypothesis survives for independent complete planning.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from transfer.cross_instance.context import context_grasp_proposal, rank_candidates
from local_models.anygrasp import AnyGraspBackend, NATIVE_LABEL_ROTATIONS

# BEGIN complete native rows, original pose scoring and explicit robot labels.
ROTATION_NATIVE_GRASP = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])

def exact_native_rows(rows):
    """Group equal full native rows without rounding, NMS or score filtering.

    Every original field participates, including quality and object identity.
    First-occurrence order preserves the existing stable ranking winner. The
    inverse map reconstructs the original row order from the retained rows.
    """
    _, first, inverse, counts = np.unique(rows, axis=0, return_index=True,
        return_inverse=True, return_counts=True)
    order = np.argsort(first, kind='stable')
    sorted_to_stable = np.empty_like(order)
    sorted_to_stable[order] = np.arange(len(order))
    return first[order], sorted_to_stable[inverse], counts[order]

class NoGraspCandidates(RuntimeError):
    """A completed native prediction exposed no candidates to relation ranking."""

    def __init__(self, diagnostics):
        super().__init__("AnyGrasp returned no native grasp candidates")
        self.diagnostics = diagnostics

def generate_and_rank(points_camera_m, source_G, evidence, backend, seed=0, *,
                      T_world_camera, region_mask, approach_thresh):
    """Bind each distinct native row once in first-occurrence order.

    ``points_camera_m`` contains observed owner points in the declared CV
    camera frame, stored as float32. It is passed directly to the SDK, so the
    recorded input domain and actual quantized input share the same coordinates.
    Camera calibration determines the world binding. The SDK supplies
    each native prediction's score, insertion depth, height and opening.
    ``region_mask`` is a nonempty bool array in exactly this input-point order.
    It retains the original region identities in the supplied owner order.
    ``approach_thresh`` is radians.
    The existing source relation supplies the G+Z approach direction.

    The proxy collision filter is disabled: official robot geometry and observed
    TSDF queries remain separate consumers with explicit finger joint inputs.

    Native pi-periodic labels are bound to two distinct actual Panda poses.
    Each retains the original native label-class score. The complete prior E
    enters soft position/orientation ranking after those actual poses exist.
    The prior remains separate from all native model predictions.
    """
    started = perf_counter()
    camera = np.asarray(T_world_camera, dtype=float)
    points_camera = np.asarray(points_camera_m, dtype=np.float32)
    # BEGIN source-conditioned sampling uses the preserved grounding output.
    region_mask = np.asarray(region_mask, dtype=bool)
    proposal = context_grasp_proposal(evidence, source_G)
    approach_camera = camera[:3, :3].T @ proposal[:3, 2]
    controls = dict(dense_grasp=False, collision_detection=False,
        region_steering=region_mask, approach_steering=approach_camera,
        approach_thresh=float(approach_thresh))
    # END the SDK receives both controls before native candidate generation.
    # Reset this observation for every call: an SDK return before forward must
    # report a missing model-consumed cloud, not retain a previous inference.
    backend.prepared_points_camera = None
    backend.prepared_region_mask = None
    backend.sampled_seed_indices = None
    native = backend.predict(points_camera, controls, seed=seed)
    preprocessing = dict(input_points=len(points_camera),
        input_frame='declared_camera_cv', T_world_camera=camera.copy(),
        input_region_points=int(region_mask.sum()),
        model_forward_observed=backend.prepared_points_camera is not None,
        model_points=(None if backend.prepared_points_camera is None
                      else len(backend.prepared_points_camera)),
        model_cloud_source='exact network forward input after native preprocessing')
    prepared_world = (None if backend.prepared_points_camera is None else
        backend.prepared_points_camera @ camera[:3, :3].T + camera[:3, 3])
    count = 0 if native is None else len(native)
    diagnostics = dict(generated_count=count, retained_count=count,
        native_pool_scope='all rows exposed by get_grasp before any caller NMS or sorting',
        controls=controls, preprocessing=preprocessing,
        native_hidden_filter_rejections='not exposed by SDK',
        seed=int(seed), seconds=perf_counter()-started)
    if count == 0:
        # BEGIN completed-forward failures retain the existing sampling evidence.
        raise NoGraspCandidates(dict(diagnostics, model_sources=backend.model_sources,
            geometry_source=backend.geometry_source, prepared_points_world=prepared_world,
            model_region_mask=backend.prepared_region_mask,
            model_seed_indices=backend.sampled_seed_indices))
        # END pre-forward returns keep these same observations explicitly missing.

    # BEGIN report actual seed membership in the native voxel domain.
    seed_indices = backend.sampled_seed_indices
    model_mask = backend.prepared_region_mask
    sampling = dict(model_region_points=int(model_mask.sum()),
        seed_count=len(seed_indices), unique_seed_count=len(np.unique(seed_indices)),
        all_seeds_in_region=bool(model_mask[seed_indices].all()),
        source='native forward region_mask and stage2_seed_inds')
    diagnostics['steering'] = dict(
        prior_origin='context_grasp_proposal_from_preserved_region_evidence',
        T_world_grasp_prior=proposal,
        T_world_tool_prior=proposal @ np.linalg.inv(backend.T_tool_grasp),
        position_scale_m=float(evidence['target_rms_m']), approach_camera=approach_camera,
        approach_world=proposal[:3, 2], approach_thresh_rad=float(approach_thresh),
        region_point_indices=np.flatnonzero(region_mask), sampling=sampling)
    # END the original masks and sampling identities remain available for audit.
    # BEGIN expose pi-label classes as actual robot bindings, not cheap scores.
    # BEGIN compact only complete equal outputs before binding or scoring.
    native_count = count
    first_rows, raw_to_unique, duplicate_counts = exact_native_rows(
        native)
    unique_count = len(first_rows)
    label_rotations = NATIVE_LABEL_ROTATIONS.copy()
    label_count = len(label_rotations)
    native_indices = np.repeat(first_rows, label_count)
    label_indices = np.tile(np.arange(label_count), unique_count)
    # END original raw row IDs remain the published candidate identities.
    count = len(native_indices)
    native_rows = native[native_indices].copy()
    native_rotations = native[native_indices, 4:13].reshape(-1, 3, 3).copy()
    native_centers = native[native_indices, 13:16].copy()
    native_poses = np.repeat(np.eye(4)[None], count, axis=0)
    native_poses[:, :3, :3] = native_rotations
    native_poses[:, :3, 3] = native_centers
    widths, heights, depths = (native[native_indices, 1].copy(),
        native[native_indices, 2].copy(), native[native_indices, 3].copy())
    quality = native[native_indices, 0].copy()
    rotations_world_native = camera[:3, :3] @ native_rotations
    centers_world_native = native_centers @ camera[:3, :3].T + camera[:3, 3]
    tips_world = centers_world_native + depths[:, None]*rotations_world_native[:, :, 0]
    grasps = np.repeat(np.eye(4)[None], count, axis=0)
    grasps[:, :3, :3] = rotations_world_native @ ROTATION_NATIVE_GRASP @ label_rotations[label_indices]
    # Every actual label binding uses its own robot anchor transform. This
    # formula also covers fixed robot anchors with transverse components.
    grasps[:, :3, 3] = tips_world - grasps[:, :3, :3] @ backend.tip_anchor_grasp
    tools = grasps @ np.linalg.inv(backend.T_tool_grasp)
    joints = backend.joint_positions_for_widths(widths)
    identifiers = np.asarray([f'anygrasp_{row:06d}_label_{label}'
                              for row,label in zip(native_indices,label_indices)])
    sources = np.full(count, 'anygrasp')
    tool_prior = proposal @ np.linalg.inv(backend.T_tool_grasp)
    position_scale = float(evidence['target_rms_m'])
    ranking = rank_candidates(tools, quality, tool_prior, position_scale)
    source_anchor = np.asarray(source_G)[:3,:3].T @ (
        np.asarray(evidence['source_center']) - np.asarray(source_G)[:3,3])
    region_anchors = grasps[:,:3,:3] @ source_anchor + grasps[:,:3,3]
    records = [dict(candidate_id=str(identifiers[index]), source='anygrasp',
        native_row_index=int(native_indices[index]), native_label_index=int(label_indices[index]),
        native_grasp_row=native_rows[index],
        T_reference_grasp=grasps[index], T_reference_tool=tools[index],
        quality=float(quality[index]), width_m=float(widths[index]),
        height_m=float(heights[index]), depth_m=float(depths[index]),
        joint_positions_m=joints[index], native_tip_world_m=tips_world[index])
        for index in range(count)]
    diagnostics.update(native_generated_count=native_count, native_unique_count=unique_count,
        exact_duplicate_rows_removed=native_count-unique_count, bound_candidate_count=count,
        generated_count=count, retained_count=count,
        native_pool_scope='all distinct complete native rows, first raw identity, both explicit pi-label robot bindings')
    # END native scores describe a label class; robot geometry keeps each pose.
    generation = dict(native_candidate_count=native_count, native_unique_count=unique_count,
        exact_duplicate_rows_removed=native_count-unique_count,
        deduplication='exact_all_native_columns_stable_first_occurrence', total_count=count,
        retained_count=count, seed=int(seed),
        candidate_source='native_anygrasp', quality_scoring='native_prediction_score_per_pi_label_class',
        prior_origin='preserved_region_relation_approach_steering',
        seconds=perf_counter()-started)
    diagnostics['generation'] = generation
    return dict(grasps_world=grasps, tool_poses_world=tools, quality=quality,
        sources=sources, candidate_ids=identifiers, candidate_records=records,
        rejected_candidates=[], context_proposal=proposal,
        prior_origin='preserved_region_relation_approach_steering',
        pool_diagnostics=diagnostics, prepared_points_world=prepared_world,
        preprocessing=preprocessing, generation=generation, ranking=ranking,
        binding=backend.T_tool_grasp.copy(), model_sources=backend.model_sources,
        native_label_rotations=label_rotations, native_row_indices=native_indices,
        native_label_indices=label_indices, region_anchor_points_world=region_anchors,
        native_first_row_indices=first_rows, native_raw_to_unique=raw_to_unique,
        native_duplicate_counts=duplicate_counts,
        device=backend.device,
        geometry_source=backend.geometry_source,
        candidate_joint_positions_m=joints, candidate_widths_m=widths,
        candidate_depths_m=depths, candidate_heights_m=heights,
        native_grasp_rows=native_rows, native_poses_camera=native_poses,
        native_tip_positions_world_m=tips_world,
        model_input_points_camera_m=points_camera,
        steering_region_mask=region_mask, steering_prior_grasp_world=proposal,
        steering_approach_camera=approach_camera,
        steering_prior_tool_world=tool_prior, position_scale_m=position_scale,
        model_region_mask=model_mask, model_seed_indices=seed_indices)
# END complete native rows, original pose scoring and explicit robot labels.


# BEGIN source representation and raw observation identity at model boundaries.
def source_capture_grasp(skill, source_frame_world, *, T_world_eef):
    """Return the demonstrated tool at the source image's owner placement.

    H_g is the complete source functional-frame-from-tool motion. Sampling its
    declared interaction time and composing source F supplies the same nominal
    image-capture tool used by native grasp steering and orientation transfer.
    The source owner pose and C_D provide a separately recorded comparison.
    Both tracks are legitimate source skill representation.
    """
    from .representation import generate_reference, sample_poses

    representation, source = skill["representation"], skill["source"]
    interaction = float(representation["interaction_time"])
    # The shared generator owns interpolation and the original event clock.
    # Its identity-parent reference exposes the same H_g sample used by every
    # later B H_g reference; encoder FK supplies its actual entrance boundary.
    source_reference = generate_reference(representation, T_world_function=np.eye(4),
                                          T_world_eef=T_world_eef)
    index = int(np.flatnonzero(source_reference["eef_timestamps"] == interaction)[0])
    relative = source_reference["eef_poses"][index]
    capture_tool = np.asarray(source_frame_world) @ relative
    owner_relation = sample_poses(representation["C_D"], representation["timestamps"], [interaction])[0]
    recorded_capture_tool = np.asarray(source["T_world_body_at_image"]) @ owner_relation
    return relative, capture_tool, dict(
        T_source_capture_tool=capture_tool,
        T_source_capture_recorded_tool=recorded_capture_tool,
        T_source_capture_body=source["T_world_body_at_image"],
        source_image_time_s=float(source["timestamp_s"]),
        source_interaction_time_s=interaction,
        source="complete source skill F_s H_g(t_interaction) at image capture",
        nominal_capture_max_matrix_error=float(np.max(np.abs(recorded_capture_tool - capture_tool))))


def record_owner_model_input(arrays, T_world_camera, input_camera_view_id):
    """Map region elements into the independent owner input by exact identity.

    ``target_element_ids`` indexes the region-reference domain;
    ``target_owner_element_ids`` is paired with the owner point input in its
    original order. A mapped region row outside that owner evidence retains
    its element ID, source-row ID and unsupported state. Coordinates never
    establish this association, including for coincident measured points.
    """
    camera = np.asarray(T_world_camera, dtype=float)
    owner = arrays["target_owner_points_world_m"]
    # BEGIN reference and owner domains share element identity, not row order.
    owner_ids = np.asarray(arrays["target_owner_element_ids"])
    if owner_ids.ndim != 1 or len(owner_ids) != len(owner):
        raise ValueError("Owner element IDs must align one-to-one with owner input points")
    lookup = dict(zip(owner_ids.tolist(), range(len(owner_ids))))
    if len(lookup) != len(owner_ids):
        raise ValueError("Owner input requires one point per registered element ID")
    region_ids = np.asarray(arrays["target_element_ids"])[
        np.asarray(arrays["matching_target_indices"], dtype=np.intp)]
    source_ids = np.asarray(arrays["source_row_ids"])
    if region_ids.shape != source_ids.shape:
        raise ValueError("Mapped region element IDs must retain every aligned source-row ID")
    owner_rows = np.fromiter((lookup.get(identity, -1) for identity in region_ids.tolist()),
                            dtype=np.intp, count=len(region_ids))
    supported = owner_rows >= 0
    points_camera = ((owner - camera[:3, 3]) @ camera[:3, :3]).astype(np.float32)
    region_mask = np.zeros(len(owner), dtype=bool)
    region_mask[owner_rows[supported]] = True
    arrays.update(grasp_model_input_points_camera_m=points_camera,
        grasp_model_input_points_world_m=owner.copy(),
        grasp_model_input_element_ids=owner_ids.copy(),
        grasp_model_region_element_ids=region_ids.copy(),
        grasp_model_region_source_row_ids=source_ids.copy(),
        grasp_model_region_owner_indices=owner_rows,
        grasp_model_region_owner_supported=supported,
        grasp_model_region_unsupported_element_ids=region_ids[~supported].copy(),
        grasp_model_region_unsupported_source_row_ids=source_ids[~supported].copy())
    provenance = dict(input_scope="registered_owner_observations",
        input_point_count=len(owner), owner_point_count=len(owner),
        region_point_count=int(region_mask.sum()), input_camera_view_id=str(input_camera_view_id),
        element_order_source="target_owner_element_ids paired with target_owner_points_world_m",
        region_source_row_count=len(region_ids),
        supported_region_source_row_count=int(supported.sum()),
        unsupported_region_source_row_count=int((~supported).sum()),
        region_identity_mapping="exact registered element ID; unsupported owner row is -1")
    # END unsupported region observations remain explicit model-input evidence.
    return points_camera, region_mask, provenance


def transport_orientation(rotation_query_reference, rotation_world_reference_camera,
                          rotation_world_query_camera, rotation_source_tool):
    """Compose raw relative object rotation, camera calibration and source E."""
    from scipy.spatial.transform import Rotation

    camera_rotation = Rotation.from_matrix(rotation_query_reference).as_matrix()
    object_rotation = (np.asarray(rotation_world_query_camera) @ camera_rotation
                       @ np.asarray(rotation_world_reference_camera).T)
    return object_rotation @ np.asarray(rotation_source_tool)


def _generation_arrays(generated, selected_index):
    """Publish every native candidate and its original scores in array form."""
    fields = dict(
        candidate_widths_m="candidate_widths_m", candidate_depths_m="candidate_depths_m",
        candidate_heights_m="candidate_heights_m", candidate_finger_joint_positions_m="candidate_joint_positions_m",
        candidate_native_poses_camera="native_poses_camera", candidate_native_grasp_rows="native_grasp_rows",
        candidate_native_tip_positions_world_m="native_tip_positions_world_m",
        candidate_native_row_indices="native_row_indices", candidate_native_label_indices="native_label_indices",
        grasp_native_first_row_indices="native_first_row_indices", grasp_native_raw_to_unique="native_raw_to_unique",
        grasp_native_duplicate_counts="native_duplicate_counts", grasp_steering_region_mask="steering_region_mask",
        grasp_steering_prior_grasp_world="steering_prior_grasp_world",
        grasp_steering_approach_camera="steering_approach_camera", grasp_steering_prior_tool_world="steering_prior_tool_world",
        grasp_model_region_mask="model_region_mask", grasp_model_seed_indices="model_seed_indices",
        candidate_grasp_poses_world="grasps_world", candidate_tool_poses_world="tool_poses_world",
        candidate_sources="sources", candidate_ids="candidate_ids", candidate_model_quality="quality",
        candidate_region_anchor_points_world="region_anchor_points_world",
        grasp_model_prepared_points_world_m="prepared_points_world",
        grasp_native_label_rotations="native_label_rotations", T_tool_grasp="binding")
    arrays = {output: generated[source] for output, source in fields.items()}
    ranking = generated["ranking"]
    arrays.update(candidate_order=ranking["order"],
        candidate_orientation_errors_rad=ranking["orientation_error_rad"],
        candidate_position_errors_m=ranking["position_error_m"],
        grasp_prior_position_scale_m=np.asarray(generated["position_scale_m"]),
        selected_tool_pose_world=generated["tool_poses_world"][selected_index])
    arrays.update({"candidate_" + name: ranking[name] for name in
        ("costs_total", "costs_quality", "costs_orientation", "costs_position")})
    return arrays
# END source data, native scores and actual consumed arrays retain their roles.


# BEGIN complete cross-instance grasp chain before shared reference generation.
def run_cross_instance_grasp(skill, region_result, views, *, feedback, robot_fixture, owner_id,
                             mapping_settings, max_tilt_rad, field_margin_m,
                             approach_thresh_rad, checkpoint_path=None):
    """Return every final E/B hypothesis with native prediction and field evidence.

    ``skill`` is already loaded. ``region_result`` is the completed public
    cross-instance region result; its F and identity-preserving assignments
    supply the grasp prior and native region steering mask. ``views`` are the
    public calibrated views in that same order. Feedback is the named robot
    measurement at image capture. Robot fixture and mapping settings explicitly
    declare the official model and observed-background integration parameters.

    The original equal-weight native-quality/full-tool-rotation/position score
    selects one actual AnyGrasp E. Each real UNOPose batch row replaces only
    its rotation. Final two-axis swing preserves the same origin. All final
    view hypotheses remain available for complete independent planning; task F
    stays distinct from B = E @ inverse(H_g(t_interaction)). The caller uses
    the shared reference generator to compose B with the complete source H_g.
    """
    from local_models.unopose import predict_batch
    from .grounding_geometry_evidence import (
        background_model_masks, group_state_checks, prepare_grasp_scene, query_tool_candidates)
    from .grasp_pose_refinement import refine_pose
    from .tool_geometry import load_robosuite_panda_tool

    if skill["representation"]["kind"] != "grasp":
        raise ValueError("Cross-instance grasp prediction requires source H_g")
    if region_result["status"] != "completed":
        raise ValueError("Cross-instance grasp prediction requires completed region correspondence")
    started = perf_counter()
    arrays = dict(region_result["arrays"])
    context = region_result["input_context"]
    observations = list(views)
    if [view["id"] for view in observations] != arrays["view_ids"].tolist():
        raise ValueError("Grasp and region observations must retain the same ordered view identities")
    relative_goal, source_tool, source_provenance = source_capture_grasp(
        skill, arrays["source_frame_world"], T_world_eef=feedback["eef_pose"])
    geometry = load_robosuite_panda_tool(tool_id=feedback["tool_id"], tool_frame=feedback["tool_frame"])
    model_geometry = geometry.grasp_generation_geometry()
    source_G = source_tool @ model_geometry["T_tool_grasp"]
    source_provenance.update(T_source_capture_grasp=source_G, T_tool_grasp=model_geometry["T_tool_grasp"])
    weights = (Path(__file__).resolve().parents[2] / "anygrasp_sdk/grasp_detection/checkpoint_detection.tar"
               if checkpoint_path is None else Path(checkpoint_path))
    backend = AnyGraspBackend(checkpoint_path=weights,
        gripper_geometry=model_geometry, tool_geometry=geometry)
    points_camera, region_mask, model_input = record_owner_model_input(
        arrays, observations[0]["T_world_camera_cv"], observations[0]["id"])
    # BEGIN visible owner evidence keeps its declared scope and source.
    model_input["owner_evidence"] = [dict(view_id=view["id"], **view["owner_evidence"])
                                     for view in observations]
    # END a selected component does not enlarge an independently supplied owner.
    # BEGIN model-input identities remain available on the native empty-pool path.
    try:
        generated = generate_and_rank(points_camera, source_G, region_result["context"], backend,
            seed=0, T_world_camera=observations[0]["T_world_camera_cv"],
            region_mask=region_mask, approach_thresh=approach_thresh_rad)
    except NoGraspCandidates as error:
        error.diagnostics.update(model_input=model_input,
            model_input_arrays={name: value for name, value in arrays.items()
                                if name.startswith("grasp_model_")})
        raise
    # END the existing typed failure owns the actual consumed point-domain evidence.
    rank = generated["ranking"]
    selected_index = int(rank["selected_index"])
    candidate_id = str(generated["candidate_ids"][selected_index])
    original_tool = generated["tool_poses_world"][selected_index].copy()
    native_fingers = generated["candidate_joint_positions_m"][selected_index].copy()
    measured = dict(zip(feedback["joint_names"], feedback["start_q"]))
    entrance_fingers = np.asarray([measured[name] for name in robot_fixture["finger_joint_names"]])
    arrays.update(_generation_arrays(generated, selected_index),
        source_capture_tool_pose_world=source_tool, source_capture_grasp_pose_world=source_G)
    selection = dict(index=selected_index, candidate_id=candidate_id,
        T_world_tool=original_tool, T_world_grasp=generated["grasps_world"][selected_index],
        model_quality=generated["quality"][selected_index], ranking_cost=rank["costs_total"][selected_index],
        quality_cost=rank["costs_quality"][selected_index], orientation_cost=rank["costs_orientation"][selected_index],
        position_cost=rank["costs_position"][selected_index],
        orientation_error_rad=rank["orientation_error_rad"][selected_index],
        position_error_m=rank["position_error_m"][selected_index],
        native_row_index=int(generated["native_row_indices"][selected_index]),
        native_label_index=int(generated["native_label_indices"][selected_index]),
        width_m=generated["candidate_widths_m"][selected_index],
        depth_m=generated["candidate_depths_m"][selected_index],
        height_m=generated["candidate_heights_m"][selected_index],
        finger_joint_positions_m=native_fingers, status="predicted_not_physically_validated")
    generation_seconds = perf_counter() - started

    # Static owner/background fields use their original explicit mask scopes.
    # Their evidence remains separate from the final self-filtered background.
    phase = perf_counter()
    # BEGIN canonical owner-local measurements remain outside static background.
    static_backgrounds, _, static_background_admission = background_model_masks(
        observations, arrays["local_material_observation_pixels"],
        arrays["local_material_observation_view_indices"])
    # END raw masks remain available with their independent observation scope.
    static_arrays, static_reports = query_tool_candidates(observations, context, geometry,
        np.repeat(original_tool[None], 2, axis=0), [candidate_id, candidate_id],
        source_provenance["source_interaction_time_s"], region_result["input_evidence"]["observation_model"],
        dict(observed_owner=[view["owner_mask"] for view in observations],
             # BEGIN background endpoints are an explicit independent input.
             observed_background=[static_backgrounds[view["id"]] for view in observations]),
             # END unassigned observed pixels do not acquire static identity.
        finger_joint_positions_m=np.stack((entrance_fingers, native_fingers)),
        inspection_state_ids=["entrance_open", "native_width"])
    arrays.update(static_arrays)
    static_seconds = perf_counter() - phase

    source = skill["source"]
    bank = source["observation"]
    reference_observation = dict(rgb=np.asarray(source["image"].convert("RGB")),
        depth=source["depth_m"], intrinsics=bank["K"], mask=bank["object_mask"])
    queries = [dict(rgb=np.asarray(view["image"].convert("RGB")), depth=view["depth_m"],
        intrinsics=view["K"], mask=view["owner_mask"]) for view in observations]
    phase = perf_counter()
    predicted = predict_batch(reference_observation, queries, seed=1)
    orientation_rows, initial_poses = [], []
    for index, view in enumerate(observations):
        raw = np.eye(4)
        raw[:3, :3] = predicted["pred_R"][index]
        raw[:3, 3] = predicted["pred_t"][index]
        tool = original_tool.copy()
        tool[:3, :3] = transport_orientation(predicted["pred_R"][index],
            np.asarray(bank["T_world_camera_cv"])[:3, :3],
            np.asarray(view["T_world_camera_cv"])[:3, :3], source_tool[:3, :3])
        initial_poses.append(tool)
        orientation_rows.append(dict(id=view["id"], batch_index=index, raw_transform=raw,
            pose_score=float(predicted["pred_pose_score"][index]), T_world_tool=tool,
            anygrasp_candidate_id=candidate_id,
            # BEGIN document the actual owner mask consumed by this query row.
            owner_evidence=dict(view["owner_evidence"]),
            query_owner_mask_pixels=int(np.count_nonzero(view["owner_mask"])),
            # END the reference/component mask retains its separate region role.
            quality_role="native quality belongs to original AnyGrasp pose; modified orientation unmeasured"))
    arrays.update({"unopose_" + name: value for name, value in predicted.items()})
    arrays["unopose_tool_poses_world"] = np.stack(initial_poses)
    unopose_seconds = perf_counter() - phase

    phase = perf_counter()
    prepared = prepare_grasp_scene(observations, fixture=robot_fixture, capture_feedback=feedback,
        tool_geometry=geometry, entrance_finger_positions=entrance_fingers,
        native_finger_positions=native_fingers, T_tool_grasp=model_geometry["T_tool_grasp"],
        tool_poses_world=initial_poses, max_tilt_rad=max_tilt_rad,
        mapping_settings=mapping_settings, owner_id=owner_id,
        local_material_observation_pixels=arrays["local_material_observation_pixels"],
        local_material_observation_view_indices=arrays["local_material_observation_view_indices"])
    arrays.update({"tilt_" + name: value for name, value in prepared["arrays"].items()})
    scene_seconds = perf_counter() - phase
    phase = perf_counter()
    hypotheses, refinement_rows = [], []
    for source_row, initial in zip(orientation_rows, initial_poses):
        row = refine_pose(initial, prepared["parts"], prepared["field"], prepared["basis_tool"],
                          max_tilt_rad, field_margin_m)
        row.update(id=source_row["id"], source_pose_score=source_row["pose_score"],
            source_raw_transform=source_row["raw_transform"], anygrasp_candidate_id=candidate_id,
            before=group_state_checks(row["before_checks"], prepared["states"], row["guide_before"]),
            after=group_state_checks(row["after_checks"], prepared["states"], row["guide_after"]))
        refinement_rows.append(row)
        tool = row["T_world_tool_proposed"]
        hypotheses.append(dict(id=source_row["id"], T_world_function=arrays["region_frame_world"],
            T_world_tool=tool, tool_reference_anchor_world=tool @ np.linalg.inv(relative_goal),
            source=dict(view_id=source_row["id"], anygrasp_candidate_id=candidate_id,
                native_quality=float(generated["quality"][selected_index]),
                native_quality_pose=original_tool, modified_pose_quality="unmeasured",
                unopose_pose_score=source_row["pose_score"],
                refinement_status=row["status"], proposal_status=row["proposal_status"])))
    arrays.update(final_tool_poses_world=np.stack([row["T_world_tool"] for row in hypotheses]),
        final_tool_reference_anchors_world=np.stack([row["tool_reference_anchor_world"] for row in hypotheses]))
    report = dict(source_capture_grasp=source_provenance, grasp_selection=selection,
        grasp_generation=dict(model_input=model_input, preprocessing=generated["preprocessing"],
            generation=generated["generation"], model_sources=generated["model_sources"],
            geometry_source=generated["geometry_source"], candidate_records=generated["candidate_records"],
            rejected_candidates=generated["rejected_candidates"], pool_diagnostics=generated["pool_diagnostics"],
            prior_origin=generated["prior_origin"]),
        static_tool_observation_checks=static_reports[candidate_id],
        static_background_admission=static_background_admission,
        unopose=dict(rows=orientation_rows, batch_size=int(predicted["batch_size"]),
            model_forward_calls=int(predicted["model_forward_calls"])),
        refinement=dict(rows=refinement_rows, scene=prepared["metadata"]),
        source_events=skill["representation"]["events"], input_context=context,
        timings=dict(native_generation_seconds=generation_seconds, static_inspection_seconds=static_seconds,
            unopose_seconds=unopose_seconds, final_scene_seconds=scene_seconds,
            final_orientation_seconds=perf_counter() - phase, total_seconds=perf_counter() - started),
        physical_trajectory_status="missing_not_executed")
    return dict(status="completed", hypotheses=hypotheses, arrays=arrays, report=report)
# END E hypotheses feed the single public full-reference generator independently.
