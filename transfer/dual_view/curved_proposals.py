"""Bounded native global-to-local Sim(3) proposals from observed structure.

Four sensor-derived voxel resolutions and eight fixed seeds allocate exactly
32 terminal slots before fitting. A slot is one Open3D FPFH/RANSAC initializer
followed by one Open3D Sim(3) ICP update; only its final pose is a candidate.
Neither native fitting nor descriptor resolution is the final weighted
identity score. Actual RGB-D support, negative evidence, absolute geometry
and assigned-measurement appearance belong to ``curved_inference``.

No feature-response field, target annotation, previous pose or mesh is an
input. View-local estimated normals and voxel centroids are proposal data,
never measured support. Native nearest-neighbor fitness is explicitly not an
identity or visibility certificate. Missing descriptors consume their fixed
slots without manufacturing normals or switching to another fitting method.
"""

from time import perf_counter

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from ..sim3 import rank
from .proposals import COMPATIBILITY_RATIO, MINIMUM_SCALE, MAXIMUM_SCALE, valid_transform


VOXEL_MULTIPLIERS = (1., 2., 4., 8.)
SEEDS = tuple(range(8))
CANDIDATE_BUDGET = len(VOXEL_MULTIPLIERS)*len(SEEDS)
RANSAC_ITERATIONS = 100000
ICP_ITERATIONS = 100


def _cloud(points):
    """Retain the supplied coordinates in the library's native geometry."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def _descriptor(points, pixels, camera_position, voxel):
    """Describe one real view after rejecting unidentified surface normals.

    Open3D uses normal radius 2v, feature radius 5v and neighborhood caps
    30/100. A normal needs spatial and original-image rank at least two:
    points along one image line may span a plane through the camera, whose
    normal orientation is unidentifiable. Such rows are absent from the
    descriptor cloud. Voxel centroids and averaged normals are used only
    after this per-view validity check.
    """
    feature = o3d.pipelines.registration.Feature()
    feature.data = np.empty((33, 0))
    detail = dict(input_count=len(points), normal_known_count=0, voxel_point_count=0)
    if len(points) < 3:
        return _cloud(np.empty((0, 3))), feature, detail
    distances, neighbors = cKDTree(points).query(points, k=min(30, len(points)),
        distance_upper_bound=2*voxel, workers=1)
    known = np.asarray([min(rank(points[row[np.isfinite(distance)]])[0],
        rank(pixels[row[np.isfinite(distance)]])[0]) >= 2
        for row, distance in zip(neighbors, distances)])
    detail['normal_known_count'] = int(known.sum())
    if not known.any():
        return _cloud(np.empty((0, 3))), feature, detail
    actual = _cloud(points)
    actual.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel, max_nn=30),
        fast_normal_computation=False)
    actual.orient_normals_towards_camera_location(camera_position)
    reduced = actual.select_by_index(np.flatnonzero(known).tolist()).voxel_down_sample(voxel)
    reduced.normalize_normals()
    detail['voxel_point_count'] = len(reduced.points)
    feature = o3d.pipelines.registration.compute_fpfh_feature(reduced,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5*voxel, max_nn=100),
        list(range(len(reduced.points))))
    return reduced, feature, detail


def _transform(matrix):
    """Validate the complete native matrix without locking uniform scale."""
    matrix = np.asarray(matrix)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    if not np.allclose(matrix[3], [0., 0., 0., 1.], rtol=0., atol=1e-12):
        return None
    scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    if not MINIMUM_SCALE <= scale <= MAXIMUM_SCALE:
        return None
    transform = (scale, matrix[:3, :3]/scale, matrix[:3, 3].copy())
    return transform if valid_transform(transform) else None


def _fit_summary(result):
    """Keep native objectives and the coarse pose as process diagnostics."""
    return dict(matrix=np.asarray(result.transformation).tolist(), fitness=float(result.fitness),
        inlier_rmse_m=float(result.inlier_rmse), correspondence_count=len(result.correspondence_set))


def build_proposals(source, surface):
    """Return at most 32 final poses and explicit native-work diagnostics.

    The complete schedule is fixed independently of every fit and final score.
    Native RANSAC may perform up to 100000 internal trials per slot; ICP may
    perform up to 100 iterations. These are not represented as 32 SVD calls.
    No candidate/refit expansion, clipping of a larger pose pool, or fallback
    solver occurs. Runtime failures propagate to the caller; this function
    never labels a failed or interrupted native search as completed.
    """
    if source.get('region_geometry') != 'surface' or surface.get('region_geometry') != 'surface':
        raise ValueError('Structural proposals require prepared curved observations')
    started = perf_counter()
    target_views = []
    pitches = [source['entity_pixel_pitch_m']]
    for view in range(len(surface['view_ids'])):
        selected = surface['observation_entity_member'] & (surface['observation_view_index'] == view)
        target_views.append((surface['observation_xyz'][selected],
            surface['observation_pixels'][selected], surface['T_world_camera_cv'][view, :3, 3]))
        pitches.append(surface['observation_spacing'][selected])
    medians = [float(np.median(values)) for values in pitches if len(values)]
    base_voxel = max(medians) if medians else None
    source_actual = _cloud(source['entity_xyz'])
    target_actual = _cloud(surface['xyz'][surface['entity_member']])
    report = dict(method='view_local_voxel_fpfh_native_ransac_sim3_icp',
        terminal_candidate_budget=CANDIDATE_BUDGET, proposal_count=0, slot_count=CANDIDATE_BUDGET,
        global_fit_calls=0, local_fit_calls=0, appended_refit_candidates=0,
        ransac_max_iterations_per_slot=RANSAC_ITERATIONS, ransac_confidence=.999,
        ransac_sample_count=4, actual_native_iteration_count='not_exposed_by_Open3D_binding',
        aggregate_ransac_iteration_upper_bound=0, icp_max_iterations_per_slot=ICP_ITERATIONS,
        uniform_scale_domain=[MINIMUM_SCALE, MAXIMUM_SCALE],
        exact_edge_scale_domain=[MINIMUM_SCALE, MAXIMUM_SCALE],
        edge_filter_boundary_and_short_edge_noise_complete=False,
        hypothesis_coverage_exhaustive=False, feature_pairing='strict_mutual_no_fallback',
        base_voxel_m=base_voxel, voxel_multipliers=list(VOXEL_MULTIPLIERS), seeds=list(SEEDS),
        normal_radius_voxels=2., normal_max_neighbors=30,
        feature_radius_voxels=5., feature_max_neighbors=100,
        descriptor_neighborhoods='independent_real_views_before_target_descriptor_concatenation',
        normal_validity='both_spatial_and_image_neighborhood_rank_at_least_two',
        native_objective='one_way_nearest_neighbor_fitness_not_identity_or_visibility_evidence',
        target_truth_used=False, previous_prediction_used=False, initial_pose_supplied=False,
        slots=[], levels=[], timings=dict(descriptor_seconds=0., native_ransac_seconds=0.,
            native_icp_seconds=0.))
    proposals = []
    missing_geometry = (base_voxel is None or not np.isfinite(base_voxel) or base_voxel <= 0
        or len(source_actual.points) < 3 or len(target_actual.points) < 3)
    for level, multiplier in enumerate(VOXEL_MULTIPLIERS):
        if missing_geometry:
            report['slots'].extend(dict(slot_id=level*len(SEEDS)+seed, level=level, seed=seed,
                terminal_candidate=False, failure='insufficient_observed_entity_geometry') for seed in SEEDS)
            continue
        phase = perf_counter()
        voxel = base_voxel*multiplier
        source_cloud, source_feature, detail = _descriptor(source['proposal_xyz'],
            source['proposal_pixels'], source['source_camera_position_world'], voxel)
        targets = [_descriptor(*view, voxel) for view in target_views]
        target_cloud = o3d.geometry.PointCloud()
        for cloud, _, _ in targets:
            target_cloud += cloud
        target_feature = o3d.pipelines.registration.Feature()
        target_feature.data = np.concatenate([np.asarray(feature.data) for _, feature, _ in targets], axis=1)
        pairs = (o3d.pipelines.registration.correspondences_from_features(
            source_feature, target_feature, True, 0.)
            if len(source_cloud.points) and len(target_cloud.points) else o3d.utility.Vector2iVector())
        report['timings']['descriptor_seconds'] += perf_counter()-phase
        report['levels'].append(dict(level=level, voxel_m=voxel, source=detail,
            target_views=[detail for _, _, detail in targets], strict_mutual_pair_count=len(pairs)))
        for seed in SEEDS:
            slot = level*len(SEEDS)+seed
            attempt = dict(slot_id=slot, level=level, seed=seed, terminal_candidate=False)
            report['slots'].append(attempt)
            if len(pairs) < 4:
                attempt['failure'] = 'fewer_than_four_strict_mutual_feature_pairs'
                continue
            o3d.utility.random.seed(seed)
            phase = perf_counter()
            threshold = 1.5*voxel
            # The reciprocal scale domain [.2,5] permits exact nonzero edge
            # ratios (.2,5). This broad filter is not the rigid .9 checker;
            # its noisy short-edge and endpoint coverage is not guaranteed.
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                source_cloud, target_cloud, pairs, threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(True), 4,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold),
                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(MINIMUM_SCALE)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(RANSAC_ITERATIONS, .999))
            elapsed = perf_counter()-phase
            report['global_fit_calls'] += 1
            report['timings']['native_ransac_seconds'] += elapsed
            attempt.update(global_result=_fit_summary(result), global_seconds=elapsed)
            # Open3D returns an identity matrix with no correspondences when
            # every RANSAC sample fails. Identity is then a failure sentinel,
            # not an observed initializer for a separate ICP search.
            if not len(result.correspondence_set):
                attempt['failure'] = 'global_result_has_no_correspondence_consensus'
                continue
            initial = _transform(result.transformation)
            if initial is None:
                attempt['failure'] = 'global_result_outside_proper_Sim3_domain'
                continue
            phase = perf_counter()
            # ICP updates this slot from actual source entity measurements to
            # actual fused target representatives. Its fixed initial-scale
            # distance allowance is an algorithmic budget, not sensor noise.
            # The final evaluator independently recomputes its final-scale
            # budget and all measured support/conflict/unknown states.
            radius = COMPATIBILITY_RATIO*source['size']*initial[0]
            refined = o3d.pipelines.registration.registration_icp(source_actual, target_actual, radius,
                result.transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_ITERATIONS))
            elapsed = perf_counter()-phase
            report['local_fit_calls'] += 1
            report['timings']['native_icp_seconds'] += elapsed
            attempt.update(refined_result=_fit_summary(refined), icp_seconds=elapsed,
                icp_distance_budget_m=radius,
                initial_matrix_unchanged=bool(np.array_equal(result.transformation, refined.transformation)))
            final = _transform(refined.transformation)
            if final is None:
                attempt['failure'] = 'refined_result_outside_proper_Sim3_domain'
                continue
            attempt['terminal_candidate'] = True
            proposals.append(dict(transform=final, proposal_id=slot,
                origins=[dict(kind=report['method'], slot_id=slot, level=level, seed=seed)]))
    report.update(proposal_count=len(proposals), proposal_seconds=perf_counter()-started,
        aggregate_ransac_iteration_upper_bound=report['global_fit_calls']*RANSAC_ITERATIONS,
        proposal_search_complete=True, proposal_stop='declared_slot_plan_completed')
    report['timings']['total_proposal_seconds'] = report['proposal_seconds']
    return proposals, report
