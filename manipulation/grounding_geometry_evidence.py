"""Observed RGB-D fields and official robot query evidence for grasp grounding.

Task and background occupancy use weighted TSDF values with their original
observed interpolation support. The official fixed robot contributes registered
collision solids and FK at named capture coordinates. Static owner/background
inspection and final self-filtered-background tilt are separate field products:
their semantic masks and ray support have different computational meanings.
"""
from __future__ import annotations

from time import perf_counter

import coal
import numpy as np

from .implicit_distance import box_corners
from .native_field_distance import _native_bounds, _transform, query_native_field
from .observed_mapping import integrate_entity
from .planning_observation import TASK_VOXEL_SIZE_M, _partition_masks, _self_filtered_views
from .robot_geometry_check import _convex_solid
from .robot_self_filter import classify_depth
from .grounding_robot_geometry import build_tool_parts, robot_self_filter_shapes

# BEGIN original observed-solid diagnostic and allocation formulas.
def query_tool_candidates(views, context, geometry, tool_poses, candidate_ids,
                          interaction_time, observation_model, mask_scopes, *,
                          finger_joint_positions_m, inspection_state_ids):
    """Return one field per entity and candidate/state/entity query records.

    mask_scopes maps an observation-entity ID to one Boolean image mask per
    view. The fixed official robot model is reusable because its geometry and
    installed component relations persist across tasks. The caller supplies one
    named-order finger configuration and inspection-state ID per query row,
    evaluated by FK. The same candidate may therefore appear at several
    explicit openings without acquiring a new candidate identity. Task objects and
    the environment are represented solely by their observed RGB-D TSDFs.
    All supplied poses use one field per mask scope. Their official-shape
    vertex union bounds storage; changing the displayed candidate never
    changes the observed representation being compared. A one-pose input is
    the same operation used by the complete grounding prediction.
    """
    import torch
    from curobo.perception import MapperCfg

    tool_poses = np.asarray(tool_poses, dtype=float)
    # BEGIN the checked shape uses each candidate's explicit nominal joint state.
    states = [geometry.at(positions) for positions in finger_joint_positions_m]
    component_poses = [{row['geom_id']:row['T_tool_geom'] for row in state['geoms']}
                       for state in states]
    world_vertices = np.concatenate([
        part['vertices'] @ (tool_pose @ poses[part['geom_id']])[:3, :3].T
        +(tool_pose @ poses[part['geom_id']])[:3, 3]
        for tool_pose, poses in zip(tool_poses, component_poses)
        for part in geometry.collision_geometries])
    # END official shapes and explicit joints determine the geometry bounds.
    size = TASK_VOXEL_SIZE_M
    padding = 4*size+observation_model['depth_error_m']
    lower, upper = world_vertices.min(axis=0)-padding, world_vertices.max(axis=0)+padding
    height, width = views[0]['depth_m'].shape
    images = [np.asarray(view['image'].convert('RGB')) for view in views]
    arrays = {}
    # BEGIN inspection rows share fields while retaining their actual candidate IDs.
    reports = {candidate_id:{} for candidate_id in candidate_ids}
    for candidate_id, state_id in zip(candidate_ids, inspection_state_ids):
        reports[candidate_id][state_id] = {}
    # END state IDs distinguish geometry queries, not generated grasps.
    for identifier, masks in mask_scopes.items():
        config = MapperCfg(extent_meters_xyz=tuple(upper-lower), voxel_size=size,
            extent_esdf_meters_xyz=tuple(upper-lower), esdf_voxel_size=size,
            grid_center=torch.tensor((lower+upper)/2, dtype=torch.float32),
            truncation_distance=2*size, image_height=height, image_width=width,
            depth_minimum_distance=.05, depth_maximum_distance=5., device='cuda:0')
        observations = [dict(depth=view['depth_m'], mask=mask, image=image,
            K=view['K'], T_entity_camera=view['T_world_camera_cv'])
            for view, mask, image in zip(views, masks, images)]
        field, mapping = integrate_entity(observations, config=config, name=identifier,
            source=context['observation_source']+'; explicitly declared calibrated RGB-D mask scope',
            max_host_rss_bytes=12_000_000_000, max_device_used_bytes=10_000_000_000)
        prefix = 'static_'+identifier
        arrays.update({prefix+'_values':field.values, prefix+'_observed':field.observed,
            prefix+'_weights':field.weights, prefix+'_origin_world_m':field.origin,
            prefix+'_voxel_size_m':np.asarray(field.voxel_size)})
        for candidate_id, state_id, tool_pose, poses, state, fingers in zip(
                candidate_ids, inspection_state_ids, tool_poses, component_poses,
                states, finger_joint_positions_m):
            results = []
            for part in geometry.collision_geometries:
                pose = tool_pose @ poses[part['geom_id']]
                answer = query_native_field(_convex_solid(part['vertices']), pose, field, np.eye(4))
                results.append(dict(answer, geometry_id=part['name'], role=part['role'],
                                    T_world_geometry=pose))
            statuses = [row['status'] for row in results]
            status = ('intersection' if 'intersection' in statuses else
                      'clear_represented_set' if all(value=='clear' for value in statuses) else 'unresolved')
            # BEGIN each witness names the exact opening queried in the common field.
            reports[candidate_id][state_id][identifier] = dict(
                status=status, inspection_state_id=state_id, observation_entity_id=identifier,
                checked_candidate_id=candidate_id, checked_tool_pose_world=tool_pose,
                checked_source_time_s=float(interaction_time), checked_capture_time_s=context['capture_time_s'],
                geometry_results=results, observation_model=observation_model, mapping=mapping,
                lower_world_m=lower, upper_world_m=upper, field_array_prefix=prefix,
                common_field_candidate_count=len(set(candidate_ids)),
                common_field_query_count=len(tool_poses),
                joint_positions_m=fingers, signed_pad_gap_m=state['signed_gap_m'],
                geometry_source='fixed official robot FK at the explicitly named inspection joint state',
                query_scope='static official tool versus the shared represented occupied set of the declared observation mask',
                bounds_source='union of all declared official tool vertices; bounds allocate storage only',
                physical_trajectory_status='missing_not_executed')
            # END the same E can retain separate entrance and native-width evidence.
    return arrays, reports

def storage_domain(parts, tool_poses_world, max_tilt_rad, settings):
    """Allocate samples around the complete rotated official assembly.

    A point at distance rho from the TCP moves by at most
    ``2*rho*sin(max_tilt_rad/2)``.  The union of all initial hypotheses is
    enlarged by that bound and the TSDF interpolation/truncation sample band.
    These bounds allocate array storage; they are never collision geometry.
    """
    # Native meshes and boxes have their complete finite extreme vertices.
    # A cylinder's display polygon does not bound the analytic curved solid;
    # its maintained native support projections supply allocation corners.
    # These corners remain storage coordinates, never collision material.
    allocation = []
    for part in parts:
        if isinstance(part["geometry"], coal.Cylinder):
            bounds = _native_bounds(part["geometry"], _transform(part["T_tool_geometry"]))
            allocation.append(box_corners(*bounds))
        else:
            allocation.append(part["vertices_tool"])
    vertices = np.concatenate(allocation)
    rho = float(np.linalg.norm(vertices, axis=1).max())
    initial = []
    for value in tool_poses_world:
        pose = np.asarray(value, dtype=float)
        initial.append(vertices @ pose[:3, :3].T + pose[:3, 3])
    initial = np.concatenate(initial)
    rotation_pad = float(2 * rho * np.sin(max_tilt_rad / 2))
    spacing = float(settings["resolution_m"])
    storage_pad = 2 * spacing + float(settings["depth_error_bound_m"])
    low = np.floor((initial.min(0) - rotation_pad - storage_pad) / spacing) * spacing
    high = np.ceil((initial.max(0) + rotation_pad + storage_pad) / spacing) * spacing
    sample_count = np.rint((high - low) / spacing).astype(np.int64) + 1
    return dict(initial_union_lower_m=initial.min(0).tolist(),
        initial_union_upper_m=initial.max(0).tolist(),
        maximum_component_radius_m=rho, max_tilt_rad=float(max_tilt_rad),
        rotation_displacement_bound_m=rotation_pad,
        interpolation_truncation_storage_padding_m=storage_pad,
        requested_sample_lower_m=low.tolist(), requested_sample_upper_m=high.tolist(),
        center_m=((low + high) / 2).tolist(), extent_m=(sample_count * spacing).tolist(),
        role="finite TSDF sample allocation; no represented collision body")
# END original observed-solid diagnostic and allocation formulas.


# BEGIN canonical local-material measurements qualify model background endpoints.
def background_model_masks(views, local_material_observation_pixels,
                           local_material_observation_view_indices):
    """Exclude declared owner material by its exact original measurement pixels.

    The region evidence producer has already established owner membership and
    supplies the corresponding observation pixels/view indices. This consumer
    performs pixel indexing only. Original reference, owner and background
    masks remain separate inputs; derived masks describe model admission.
    """
    views = list(views)
    pixels = np.asarray(local_material_observation_pixels)
    view_indices = np.asarray(local_material_observation_view_indices)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.dtype.kind not in "iu"
            or view_indices.shape != (len(pixels),) or view_indices.dtype.kind not in "iu"
            or np.any(view_indices < 0) or np.any(view_indices >= len(views))):
        raise ValueError("Local-material pixels require aligned original integer view/xy identities")
    backgrounds, materials, reports = {}, {}, {}
    for index, view in enumerate(views):
        identity = view["id"]
        supplied = np.asarray(view["background_mask"])
        if supplied.dtype != np.bool_ or supplied.shape != np.asarray(view["depth_m"]).shape:
            raise ValueError("Background evidence must retain the original Boolean image shape")
        selected = pixels[view_indices == index]
        if (np.any(selected < 0) or np.any(selected[:, 0] >= supplied.shape[1])
                or np.any(selected[:, 1] >= supplied.shape[0])):
            raise ValueError("Local-material measurement identities must lie in their original image")
        material = np.zeros(supplied.shape, dtype=bool)
        material[selected[:, 1], selected[:, 0]] = True
        backgrounds[identity] = supplied & ~material
        materials[identity] = material
        reports[identity] = dict(
            scope="explicit_static_background_excluding_declared_owner_local_material",
            source="region_result.local_material_observation_pixels/view_indices; exact measurement indexing",
            owner_evidence=dict(view["owner_evidence"]),
            supplied_background_pixels=int(np.count_nonzero(supplied)),
            local_material_measurement_count=len(selected),
            local_material_pixels=int(np.count_nonzero(material)),
            excluded_local_material_pixels=int(np.count_nonzero(supplied & material)),
            model_background_candidate_pixels=int(np.count_nonzero(backgrounds[identity])))
    return backgrounds, materials, reports
# END the owner declaration is consumed from the canonical material domain once.


# BEGIN CPU observation preparation retains reference, owner and background scopes.
def prepare_grasp_observations(views, *, fixture, capture_frames, reference_frame,
                               mapping_settings, owner_id,
                               local_material_observation_pixels,
                               local_material_observation_view_indices):
    """Return normalized views, explicit partitions, reports and source arrays.

    The reference observation keeps the selected functional component. The
    independent owner observation and its original mask provide only the
    declared visible owner material. Static background endpoints come from
    background_mask; an empty mask preserves unassigned pixels as unknown.
    The same official robot FK/classifier and existing semantic boundary
    operation determine which measured endpoints enter each field.
    """
    views = list(views)
    model_backgrounds, local_materials, admission_reports = background_model_masks(
        views, local_material_observation_pixels, local_material_observation_view_indices)
    filter_geometries = robot_self_filter_shapes(fixture, capture_frames)
    observations, observation_reports, observation_arrays = {}, {}, {}
    for index, view in enumerate(views):
        name = str(view["id"])
        owner_mask = np.asarray(view["owner_mask"])
        background_mask = np.asarray(view["background_mask"])
        depth = np.asarray(view["depth_m"])
        image = np.asarray(view["image"].convert("RGB"))
        K = np.asarray(view["K"], dtype=float)
        transform = np.asarray(view["T_world_camera_cv"], dtype=float)
        phase = perf_counter()
        filtered = classify_depth(observed_depth=depth, K=K, T_world_camera=transform,
            geometries=filter_geometries,
            padding_scale=mapping_settings["self_filter_padding_scale"],
            padding_offset_m=mapping_settings["self_filter_padding_offset_m"],
            source=mapping_settings["self_filter_source"])
        observations[name] = dict(image=image, depth=depth, K=K,
            T_output_camera=transform, reference_frame=reference_frame,
            self_filter=filtered["self_filter"], owner_masks={owner_id: owner_mask},
            background_mask=model_backgrounds[name], background_candidate_mask=background_mask,
            local_material_mask=local_materials[name])
        observation_reports[name] = dict(
            semantic_mask_pixels=int(owner_mask.sum()),
            reference_mask_pixels=int(np.count_nonzero(view["observation"]["object_mask"])),
            owner_point_count=len(view["owner_observation"]["xyz"]),
            owner_evidence=dict(view["owner_evidence"]),
            background_candidate_pixels=int(np.count_nonzero(background_mask)),
            background_admission=admission_reports[name],
            raw_valid_depth_pixels=int(np.count_nonzero(np.isfinite(depth) & (depth > 0))),
            image_shape=list(depth.shape), self_filter_seconds=perf_counter() - phase,
            self_filter=filtered["provenance"])
        observation_arrays.update({f"view_{index}_raw_depth_m": depth,
            f"view_{index}_image": image, f"view_{index}_semantic_owner_mask": owner_mask,
            f"view_{index}_reference_mask": np.asarray(view["observation"]["object_mask"]),
            f"view_{index}_background_candidate_mask": background_mask,
            f"view_{index}_background_model_mask": model_backgrounds[name],
            f"view_{index}_local_material_mask": local_materials[name],
            f"view_{index}_self_filter_labels": filtered["self_filter"]["labels"],
            f"view_{index}_free_ray_depth_m": filtered["self_filter"]["free_ray_depth_m"],
            f"view_{index}_K": K, f"view_{index}_T_world_camera_cv": transform})
    normalized = _self_filtered_views(observations, reference_frame,
                                      mapping_settings["depth_range_m"])
    masks = _partition_masks(normalized, (owner_id,), build_background=True)
    for index, (name, row) in enumerate(normalized.items()):
        background = masks[name][None]
        observation_reports[name].update(
            admitted_owner_depth_pixels=int(np.count_nonzero(masks[name][owner_id] & (row["depth"] > 0))),
            admitted_background_depth_pixels=int(np.count_nonzero(background & (row["depth"] > 0))),
            semantic_background_pixels=int(np.count_nonzero(observations[name]["background_candidate_mask"])),
            admitted_local_material_depth_pixels=int(np.count_nonzero(
                row["local_material_mask"] & (row["self_filter"]["labels"] == 2))),
            unknown_semantic_boundary_pixels=int(np.count_nonzero(
                (row["self_filter"]["labels"] == 2)
                & row["background_mask"] & ~background)),
            unassigned_observed_pixels=int(np.count_nonzero(
                (row["self_filter"]["labels"] == 2) & (row["depth"] > 0)
                & ~row["owner_masks"][owner_id] & ~row["local_material_mask"]
                & ~row["background_mask"])))
        observation_arrays.update({f"view_{index}_mapper_depth_m": row["depth"],
            f"view_{index}_admitted_background_mask": background,
            f"view_{index}_admitted_owner_mask": masks[name][owner_id],
            f"view_{index}_admitted_local_material_mask": (
                row["local_material_mask"] & (row["self_filter"]["labels"] == 2))})
    return normalized, masks, observation_reports, observation_arrays
# END model evidence masks retain their original pixel identities.


# BEGIN one self-filtered background for all orientation hypotheses.
def prepare_grasp_scene(views, *, fixture, capture_feedback, tool_geometry,
                        entrance_finger_positions, native_finger_positions,
                        T_tool_grasp, tool_poses_world, max_tilt_rad,
                        mapping_settings, owner_id, local_material_observation_pixels,
                        local_material_observation_view_indices):
    """Build the final tilt assembly and observed background once.

    Views use the public grounding fields image, depth_m, K,
    T_world_camera_cv, observation, owner_observation, owner_mask, owner_evidence,
    background_mask and id. Capture feedback supplies named
    coordinates, robot-base calibration and TCP calibration at image time.
    Both explicit finger states use the fixture's finger-joint order. The
    complete official robot filters capture rays; only the exact TCP-fixed
    assembly and finger descendants participate in local candidate rotation.
    """
    started = perf_counter()
    parts, states, capture_frames, components, part_arrays, robot_report = build_tool_parts(
        fixture, capture_feedback, entrance_finger_positions,
        native_finger_positions, tool_geometry)
    basis_tool = np.asarray(T_tool_grasp)[:3, :2].copy()
    domain = storage_domain(parts, tool_poses_world, max_tilt_rad, mapping_settings)
    robot_seconds = perf_counter() - started
    filter_started = perf_counter()
    normalized, masks, observation_reports, observation_arrays = prepare_grasp_observations(
        views, fixture=fixture, capture_frames=capture_frames,
        reference_frame=capture_feedback["reference_frame"],
        mapping_settings=mapping_settings, owner_id=owner_id,
        local_material_observation_pixels=local_material_observation_pixels,
        local_material_observation_view_indices=local_material_observation_view_indices)
    filter_seconds = perf_counter() - filter_started

    import torch
    from curobo.perception import MapperCfg

    spacing = float(mapping_settings["resolution_m"])
    height, width = next(iter(normalized.values()))["depth"].shape
    phase = perf_counter()
    config = MapperCfg(extent_meters_xyz=tuple(domain["extent_m"]), voxel_size=spacing,
        extent_esdf_meters_xyz=tuple(domain["extent_m"]), esdf_voxel_size=spacing,
        grid_center=torch.tensor(domain["center_m"]), truncation_distance=2 * spacing,
        image_height=height, image_width=width,
        depth_minimum_distance=mapping_settings["depth_range_m"][0],
        depth_maximum_distance=mapping_settings["depth_range_m"][1], device="cuda:0")
    configuration_seconds = perf_counter() - phase
    phase = perf_counter()
    field, mapping = integrate_entity(
        (dict(image=row["image"], depth=row["depth"], mask=masks[name][None],
              K=row["K"], T_entity_camera=row["T_output_camera"])
         for name, row in normalized.items()),
        config=config, name="observed_background",
        source=f"provided_rgbd:{','.join(normalized)}; explicitly declared static-background masks",
        max_host_rss_bytes=12_000_000_000, max_device_used_bytes=10_000_000_000)
    integration_seconds = perf_counter() - phase
    arrays = dict(basis_tool=basis_tool, **part_arrays, **observation_arrays,
        background_values=field.values, background_observed=field.observed,
        background_weights=field.weights, background_origin_world_m=field.origin,
        background_voxel_size_m=np.asarray(field.voxel_size))
    metadata = dict(status="prepared", capture_feedback_source=capture_feedback["source"],
        capture_time_s=float(capture_feedback["timestamp_s"]),
        reference_frame=capture_feedback["reference_frame"],
        basis_tool=basis_tool, basis_source="official T_tool_grasp rotation columns 0 and 1",
        basis_semantics="official grasp transverse axes; approach is column 2",
        parts=components, states=states, robot=robot_report, storage_domain=domain,
        mapping_settings=mapping_settings, view_ids=list(normalized),
        observations=observation_reports, field=mapping,
        timings=dict(input_and_robot_seconds=robot_seconds,
            observation_preparation_seconds=filter_seconds,
            mapper_configuration_seconds=configuration_seconds,
            field_build_seconds=integration_seconds, total_prepare_seconds=perf_counter() - started))
    return dict(parts=parts, basis_tool=basis_tool, field=field,
                states=states, metadata=metadata, arrays=arrays)


def group_state_checks(checks, states, guide):
    """Share each rigid-component query under every declared finger state."""
    output = {}
    for identity in states:
        rows = [row for row in checks if row["state_id"] in ("tcp_fixed", identity)]
        statuses = [row["status"] for row in rows]
        status = ("intersection" if "intersection" in statuses else
                  "clear_represented_set" if all(value == "clear" for value in statuses) else "unresolved")
        output[identity] = dict(status=status, geometry_results=rows, **states[identity])
    return dict(states=output, guide_support=guide)
# END all orientation hypotheses share this same immutable observed field.
