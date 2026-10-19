"""Actual RGB-D measurements and bounded graph neighborhoods for curved regions.

This is the explicit ``surface`` geometry path. It never estimates a global
plane, fills missing depth, averages measurements, or manufactures normals.
The declared observation model bounds discrete pixel association and local
depth variation; its support is not a probability or an exact topology proof.
All source rows survive, including rows whose target evidence stays unknown.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from ..core import _indices, _mask, _sampling_parameters, fps_indices
from ..multiview_geometry import _view_arrays
from ..sim3 import _points, apply_sim3, rank, rms_size


@dataclass(frozen=True)
class ObservationModel:
    """Deterministic observation assumptions, separate from the output radius.

    ``depth_error_m`` bounds one optical-depth measurement in this model; it
    is not an inferred sensor standard deviation. ``max_depth_slope`` bounds
    depth change per transverse pixel distance. A slope of three admits steep
    smooth patches while cutting the existing 8/40-mm step examples at their
    1.5-mm pixel pitch. That example verifies a rule, not universal calibration.
    A caller with a different measurement model must declare it explicitly.
    Registration error is supplied separately for each candidate projection.
    """

    depth_error_m: float = 0.0001
    max_depth_slope: float = 3.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be a finite nonnegative model bound')
        if self.max_depth_slope == 0:
            raise ValueError('A curved observation model needs a positive depth slope bound')


STATE_NAMES = ('measurement_supported', 'behind_camera', 'outside_image',
               'invalid_depth', 'association_unresolved', 'occluded',
               'depth_model_conflict', 'outside_entity', 'invalid_projection',
               'pixel_association_ambiguous')
_STATE = {name: i for i, name in enumerate(STATE_NAMES)}


# BEGIN preserve complete measured source structure alongside local evidence.
def prepare_source(source, depth_m, *, query_indices, query_region,
                   sampling_radius_m=.03, model=None):
    """Freeze the existing queries/checks and actual source sampling anchors.

    Independent checks use the same source-only 32-point FPS per role as the
    planar implementation. Medoids are actual region observations nearest the
    mean; exact ties remain a set. Neither that mean nor those anchors changes
    the demonstration's functional frame F, which belongs to the caller.
    """
    _sampling_parameters(sampling_radius_m, sampling_radius_m)
    points = _points(source['xyz'], 'source.xyz')
    region = _mask(source['region_mask'], len(points))
    entity = _mask(source['entity_member'], len(points))
    ids = _indices(query_indices, len(points), 'query_indices').copy()
    regional = _mask(query_region, len(ids)).copy()
    if (not np.array_equal(regional, region[ids])
            or np.any(~regional & ~entity[ids])):
        raise ValueError('Query provenance disagrees with source region/entity')
    # Validate the complete calibrated depth against the supplied point bank;
    # no local-plane validity mask participates in this position check.
    full, _, _, lookup, intrinsic, camera_pose, depth = _view_arrays(dict(depth_m=depth_m,
        K=source['K'], T_world_camera_cv=source['T_world_camera_cv'], observation=source))
    pixels = np.asarray(source['pixels'])
    model = ObservationModel() if model is None else model
    if not isinstance(model, ObservationModel):
        raise ValueError('model must be an ObservationModel')
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(source['K']).T
    position_error = model.depth_error_m*np.linalg.norm(rays, axis=1)
    if np.any(lookup[pixels[:, 1], pixels[:, 0]] < 0):
        raise ValueError('Every source point must name an actual depth measurement')
    # BEGIN retain the complete observed entity for structural proposals.
    # Query/check FPS is an evaluation budget, not a replacement for the source
    # object's geometry. Keep original row identities and actual measurements;
    # estimated descriptor normals never become measured source directions.
    entity_ids = np.flatnonzero(entity)
    inverse_intrinsic = np.linalg.inv(intrinsic)
    transverse_ray_pitch = min(np.linalg.norm(inverse_intrinsic[:, 0]),
                               np.linalg.norm(inverse_intrinsic[:, 1]))
    entity_pixel_pitch = depth[pixels[entity_ids, 1], pixels[entity_ids, 0]]*transverse_ray_pitch
    # END complete observed source entity and calibrated descriptor scale.
    unused = entity.copy()
    unused[ids] = False
    checks = []
    for role in (region, ~region):
        group = np.flatnonzero(unused & role)
        checks.append(group[fps_indices(points[group], 32)] if len(group) else group)
    check_ids = np.concatenate(checks)
    region_ids = np.flatnonzero(region)
    center = points[region].mean(0) if len(region_ids) else np.full(3, np.nan)
    cost = np.sum((points[region]-center)**2, axis=1)
    anchors = region_ids[cost == cost.min()] if len(cost) else region_ids
    size = rms_size(points[entity]) if entity.any() else 0.
    reasons = []
    if len(region_ids) < 6:
        reasons.append('fewer_than_six_source_region_points')
    if rank(points[region])[0] < 2:
        reasons.append('source_region_rank_below_two')
    if not np.isfinite(size) or size <= np.finfo(float).tiny:
        reasons.append('degenerate_source_entity_geometry')
    diagnostics = dict(region_geometry='surface', query_count=len(ids),
        observation_model=asdict(model),
        position_error_definition='depth_error_times_calibrated_source_ray_norm',
        region_query_count=int(regional.sum()), context_query_count=int((~regional).sum()),
        # BEGIN query count is a measurement budget, not a uniform score weight.
        query_normal_valid_count=0,
        # END the scorer reports its separate normalized source weights.
        check_point_count=len(check_ids), check_region_count=int(region[check_ids].sum()),
        check_context_count=int((~region[check_ids]).sum()), check_normal_valid_count=0,
        source_region_points=len(region_ids), full_depth_point_count=len(full),
        source_entity_point_count=len(entity_ids),
        source_entity_rms_m=float(size), source_anchor_indices=anchors.tolist(),
        anchor_definition='squared_distance_medoid_set_exact_ties_retained',
        normal_domain='not_required_for_position', quality_warnings=[])
    # BEGIN explicit structural inputs are independent of sampling anchors/F.
    # BEGIN proposal descriptors retain every demonstrated point, including
    # legal region observations outside the selected entity segmentation.
    proposal_ids = np.flatnonzero(entity | region)
    # END this union is separate from full-entity refutation and query FPS.
    return dict(region_geometry='surface', query_indices=ids, query_region=regional,
        entity_indices=entity_ids, entity_xyz=points[entity_ids].copy(),
        entity_position_error_m=position_error[entity_ids], entity_region=region[entity_ids].copy(),
        entity_pixel_pitch_m=entity_pixel_pitch,
        source_camera_position_world=camera_pose[:3, 3].copy(),
        # BEGIN actual source points/pixels for view-local proposal normals.
        proposal_xyz=points[proposal_ids].copy(), proposal_pixels=pixels[proposal_ids].copy(),
        # END estimated proposal normals never enter measured geometry.
        query_xyz=points[ids].copy(), query_normals=np.full((len(ids), 3), np.nan),
        query_position_error_m=position_error[ids],
        query_normal_valid=np.zeros(len(ids), bool), check_indices=check_ids,
        check_xyz=points[check_ids].copy(), check_region=region[check_ids].copy(),
        check_position_error_m=position_error[check_ids],
        check_normals=np.full((len(check_ids), 3), np.nan),
        check_normal_valid=np.zeros(len(check_ids), bool), center=center,
        normal=None, anchor_indices=anchors, anchor_xyz=points[anchors].copy(),
        anchor_position_error_m=position_error[anchors],
        size=float(size), diagnostics=diagnostics, reasons=reasons)
    # END actual source geometry remains separate from observed target evidence.
# END measured source preparation for the explicit curved branch.


def _pixel_edges(xyz, pixels, entity, lookup, K, pose, depth, model):
    """Return only observed, slope-bounded pixel edges, including safe diagonals.

    The depth bound uses transverse ray spacing, not the 3-cm output radius.
    An edge mainly along a viewing ray must also satisfy the implied minimum
    angle atan(1/max_depth_slope), apart from the declared depth-error budget.
    Diagonals require all four orthogonal cell sides; a missing pixel or cut
    side therefore cannot be bypassed by a diagonal shortcut.
    """
    height, width = lookup.shape
    horizontal = np.zeros((height, max(width-1, 0)), bool)
    vertical = np.zeros((max(height-1, 0), width), bool)
    records, rejected = [], 0
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0, y1 = 0, height-dy
        x0, x1 = max(0, -dx), min(width, width-dx)
        a0 = lookup[y0:y1, x0:x1]
        b0 = lookup[y0+dy:y1+dy, x0+dx:x1+dx]
        present = (a0 >= 0) & (b0 >= 0)
        yy, xx = np.nonzero(present)
        a, b = a0[yy, xx], b0[yy, xx]
        if not len(a):
            continue
        pa, pb = pixels[a], pixels[b]
        za, zb = depth[pa[:, 1], pa[:, 0]], depth[pb[:, 1], pb[:, 0]]
        ray_delta = np.column_stack((pb-pa, np.zeros(len(a)))) @ np.linalg.inv(K).T
        transverse = np.minimum(za, zb)*np.linalg.norm(ray_delta, axis=1)
        difference = np.abs(zb-za)
        allowed = model.max_depth_slope*transverse + 2*model.depth_error_m
        good = difference <= allowed
        # Perspective changes the world edge length away from the optical
        # axis. Bound it by its transverse part plus the permitted depth part.
        rays = np.column_stack((pa, np.ones(len(a)))) @ np.linalg.inv(K).T
        length = np.linalg.norm(xyz[b]-xyz[a], axis=1)
        upper = np.maximum(za, zb)*np.linalg.norm(ray_delta, axis=1)
        upper += allowed*np.linalg.norm(rays, axis=1)
        good &= length <= np.nextafter(upper, np.inf)
        if dy == 0:
            horizontal[yy, xx] = good
        elif dx == 0:
            vertical[yy, xx] = good
        else:
            # Both diagonal orientations occupy the same four-pixel cell.
            left = xx+x0+min(0, dx)
            top = yy+y0
            good &= (horizontal[top, left] & horizontal[top+1, left]
                     & vertical[top, left] & vertical[top, left+1])
        # BEGIN: segmentation holes cannot be crossed by an entity diagonal.
        # Full-depth edges remain separate for subpixel visibility evidence.
        entity_good = entity[a] & entity[b]
        if dy and dx:
            cell = np.stack((lookup[top, left], lookup[top, left+1],
                             lookup[top+1, left], lookup[top+1, left+1]), axis=1)
            entity_good &= np.all((cell >= 0) & entity[np.maximum(cell, 0)], axis=1)
        # END: entity-only diagonal qualification.
        rejected += int((~good).sum())
        records.append((a[good], b[good], length[good], entity_good[good]))
    if not records:
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0),
                np.empty(0, bool), rejected)
    return (*(np.concatenate([record[i] for record in records]) for i in range(4)), rejected)


def _weighted_graph(a, b, weights, count):
    """Build a symmetric CSR graph without adding duplicate edge lengths."""
    keep = a != b
    a, b, weights = a[keep], b[keep], weights[keep]
    left, right = np.minimum(a, b), np.maximum(a, b)
    order = np.lexsort((right, left))
    left, right, weights = left[order], right[order], weights[order]
    first = np.r_[True, (left[1:] != left[:-1]) | (right[1:] != right[:-1])] if len(left) else np.zeros(0, bool)
    left, right, weights = left[first], right[first], weights[first]
    return coo_matrix((np.r_[weights, weights], (np.r_[left, right], np.r_[right, left])),
                      shape=(count, count)).tocsr()


def build_target_surface(views, *, sampling_radius_m=.03, model=None, stats=None):
    """Keep complete depths and build explicitly observed curved adjacency.

    Cross-view fusion uses reciprocal geometric association and mutual unique
    nearest measurements, never feature scores. Each element contains at most
    one point per view and retains camera A's actual coordinate. Unpaired
    measurements remain available and are not physical exclusivity evidence.
    """
    model = ObservationModel() if model is None else model
    if not isinstance(model, ObservationModel):
        raise ValueError('model must be an ObservationModel')
    _sampling_parameters(sampling_radius_m, sampling_radius_m)
    views = list(views)
    if (len(views) != 2 or any(not isinstance(v.get('id'), str) or not v['id'] for v in views)
            or len({v['id'] for v in views}) != 2):
        raise ValueError('Require exactly two ordered views with distinct nonempty identities')
    arrays = [_view_arrays(view) for view in views]
    sizes = np.asarray([len(a[0]) for a in arrays], np.int64)
    offsets = np.r_[0, np.cumsum(sizes)]
    xyz = np.concatenate([a[0] for a in arrays])
    pixels = np.concatenate([a[1] for a in arrays])
    entity = np.concatenate([a[2] for a in arrays])
    spacing, position_error, edges, cut = [], [], [], 0
    for index, (points, xy, membership, lookup, K, pose, depth) in enumerate(arrays):
        z = depth[xy[:, 1], xy[:, 0]]
        spacing.append(z * min(np.linalg.norm(np.linalg.inv(K)[:, 0]),
                               np.linalg.norm(np.linalg.inv(K)[:, 1])))
        rays = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(K).T
        position_error.append(model.depth_error_m*np.linalg.norm(rays, axis=1))
        a, b, weight, eligible, rejected = _pixel_edges(points, xy, membership, lookup, K, pose, depth, model)
        edges.append((a+offsets[index], b+offsets[index], weight, eligible))
        cut += rejected
    spacing = np.concatenate(spacing)
    a, b, weights, eligible = (np.concatenate([edge[i] for edge in edges]) for i in range(4))
    depth_graph = _weighted_graph(a, b, weights, len(xyz))
    a, b, weights = a[eligible], b[eligible], weights[eligible]
    observation_graph = _weighted_graph(a, b, weights, len(xyz))
    surface = dict(region_geometry='surface', observation_model=model,
        observation_xyz=xyz, observation_pixels=pixels,
        observation_view_index=np.repeat(np.arange(2), sizes),
        observation_normals=np.full_like(xyz, np.nan), observation_spacing=spacing,
        observation_position_error_m=np.concatenate(position_error),
        observation_valid=np.ones(len(xyz), bool), observation_entity_member=entity,
        pixel_to_observation=[np.where(item[3] >= 0, item[3]+offsets[i], -1)
                              for i, item in enumerate(arrays)],
        view_offsets=offsets, view_ids=[v['id'] for v in views],
        K=np.stack([a[4] for a in arrays]), T_world_camera_cv=np.stack([a[5] for a in arrays]),
        depth_m=[a[6] for a in arrays], sampling_radius_m=float(sampling_radius_m),
        observation_adjacency=observation_graph, depth_adjacency=depth_graph)
    pair_a, pair_b = np.empty(0, np.int64), np.empty(0, np.int64)
    # Pair only entity measurements. Complete nonentity depths still supply
    # visibility/conflict evidence but need no region graph fusion.
    ia = np.flatnonzero(entity[:offsets[1]])
    ib = offsets[1]+np.flatnonzero(entity[offsets[1]:])
    if len(ia) > 1 and len(ib) > 1:
        da, ab = cKDTree(xyz[ib]).query(xyz[ia], k=2, workers=1)
        db, ba = cKDTree(xyz[ia]).query(xyz[ib], k=2, workers=1)
        local = np.arange(len(ia))
        mutual = (ba[ab[:, 0], 0] == local) & (da[:, 0] < da[:, 1])
        mutual &= db[ab[:, 0], 0] < db[ab[:, 0], 1]
        pair_a, pair_b = ia[mutual], ib[ab[mutual, 0]]
        forward = project_measurements(surface, xyz[pair_a], 1,
            query_position_error_m=surface['observation_position_error_m'][pair_a])
        backward = project_measurements(surface, xyz[pair_b], 0,
            query_position_error_m=surface['observation_position_error_m'][pair_b])
        good = forward['supported'] & backward['supported']
        good &= (forward['observation_indices'] == pair_b) & (backward['observation_indices'] == pair_a)
        pair_a, pair_b = pair_a[good], pair_b[good]
    parent = np.arange(len(xyz))
    parent[pair_b] = pair_a
    representatives, mapping = np.unique(parent, return_inverse=True)
    member_views = np.zeros((len(representatives), 2), bool)
    member_views[mapping, surface['observation_view_index']] = True
    element_entity = np.zeros(len(representatives), bool)
    np.logical_or.at(element_entity, mapping, entity)
    # A fused representative is an actual point; recomputing every edge length
    # makes path distance >= Euclidean endpoint distance after fusion too.
    ma, mb = mapping[a], mapping[b]
    merged_xyz = xyz[representatives]
    merged_weights = np.linalg.norm(merged_xyz[ma]-merged_xyz[mb], axis=1)
    surface.update(xyz=merged_xyz, normals=np.full_like(merged_xyz, np.nan),
        valid=np.ones(len(representatives), bool), spacing=spacing[representatives],
        position_error_m=surface['observation_position_error_m'][representatives],
        entity_member=element_entity, representative_observation=representatives,
        observation_to_element=mapping, member_views=member_views,
        adjacency=_weighted_graph(ma, mb, merged_weights, len(representatives)))
    entity_ids = np.flatnonzero(element_entity)
    observed_ids = np.flatnonzero(entity)
    surface['_entity_ids'] = entity_ids
    surface['_entity_tree'] = cKDTree(merged_xyz[entity_ids]) if len(entity_ids) else None
    surface['_observation_entity_ids'] = observed_ids
    surface['_observation_entity_tree'] = cKDTree(xyz[observed_ids]) if len(observed_ids) else None
    surface['diagnostics'] = dict(region_geometry='surface', observation_model=asdict(model),
        raw_observation_count=len(xyz), element_count=len(representatives),
        entity_element_count=int(element_entity.sum()), paired_element_count=len(pair_a),
        retained_pixel_edges=len(a), rejected_pixel_edges=cut,
        connectivity='slope_bounded_original_pixel_edges_with_supported_cross_view_pairs',
        representative='actual_first_view_measurement',
        unpaired_is_physical_exclusivity=False, normals_required=False,
        continuous_topology_certified=False)
    if stats is not None:
        stats.update(surface['diagnostics'])
    return surface


def _conflict_footprint_continuous(surface, camera, pixels, errors, view_index):
    """Require an uncertain query's complete projected footprint on one depth patch.

    A 3-D error bound includes lateral displacement. A nominal integer ray
    hitting a background surface therefore cannot refute a query whose error
    ball can reach a foreground ray across a depth cut. The rectangle below
    conservatively encloses the perspective projection of that ball; all its
    original pixels and orthogonal edges must be present. This only qualifies
    negative evidence; it never searches that rectangle for a favorable match.
    """
    K = surface['K'][view_index]
    lookup = surface['pixel_to_observation'][view_index]
    graph = surface['depth_adjacency']
    result = np.zeros(len(camera), bool)
    for row, (point, pixel, error) in enumerate(zip(camera, pixels, errors)):
        if error == 0:
            result[row] = True  # The nominal floor/ceil cell is checked below.
            continue
        if point[2] <= error:
            continue
        axis = K[:2].copy()
        axis[:, 2] -= pixel
        margin = error*np.linalg.norm(axis, axis=1)/(point[2]-error)
        lower, upper = np.floor(pixel-margin).astype(int), np.ceil(pixel+margin).astype(int)
        if (np.any(lower < 0) or upper[0] >= lookup.shape[1] or upper[1] >= lookup.shape[0]):
            continue
        patch = lookup[lower[1]:upper[1]+1, lower[0]:upper[0]+1]
        if np.any(patch < 0):
            continue
        horizontal = np.asarray(graph[patch[:, :-1].ravel(), patch[:, 1:].ravel()]).ravel()
        vertical = np.asarray(graph[patch[:-1].ravel(), patch[1:].ravel()]).ravel()
        result[row] = np.all(horizontal > 0) and np.all(vertical > 0)
    return result


# BEGIN direct measured-point and continuous-pixel support remain distinct.
def project_measurements(surface, points, view_index, *, registration_error_m=0.,
                         query_position_error_m=0.):
    """Associate a query with the deterministic nearest original pixel only.

    No favorable point/feature search or continuous-depth interpolation occurs.
    Direct support uses only the registration/source/target measurement bounds;
    it needs no continuous pixel cell. Additional pixel-rounding allowance uses
    the declared depth-slope model and requires that cell's observed continuity.
    A nearer-than-observed query exceeding that model is a model conflict, not
    an unconditional free-space certificate. Behind-first-return queries stay
    unknown/occluded. Exact half-pixel association ties remain unknown.
    ``support_kind`` names the final position-support evidence (including an
    actual nonentity measurement); all unknown/conflicting rows remain ``none``.
    ``association_bound_m`` uses the strict direct bound whenever that evidence
    supports the position; it must not report a discarded pixel allowance.
    """
    points = _points(points, 'curved projection points')
    if view_index not in (0, 1):
        raise ValueError('view_index must be 0 or 1')
    if (isinstance(registration_error_m, (bool, np.bool_))
            or not np.isfinite(registration_error_m) or registration_error_m < 0):
        raise ValueError('registration_error_m must be a finite nonnegative bound')
    query_error = np.broadcast_to(np.asarray(query_position_error_m, float), (len(points),))
    if not np.isfinite(query_error).all() or np.any(query_error < 0):
        raise ValueError('query_position_error_m must provide nonnegative finite bounds')
    K, pose = surface['K'][view_index], surface['T_world_camera_cv'][view_index]
    lookup = surface['pixel_to_observation'][view_index]
    height, width = lookup.shape
    model = surface['observation_model']
    camera = (points-pose[:3, 3]) @ pose[:3, :3]
    homogeneous = camera @ K.T
    front = np.isfinite(homogeneous).all(1) & (camera[:, 2] > 0)
    pixels = np.full((len(points), 2), np.nan)
    pixels[front] = homogeneous[front, :2]/homogeneous[front, 2, None]
    numeric = 64*np.finfo(float).eps*max(height, width)
    rounded = np.rint(pixels)
    pixels = np.where(np.abs(pixels-rounded) <= numeric, rounded, pixels)
    state = np.full(len(points), _STATE['behind_camera'], np.int8)
    inside = front & np.all(pixels >= 0, axis=1) & np.all(pixels <= [width-1, height-1], axis=1)
    state[front & ~inside] = _STATE['outside_image']
    ids = np.full(len(points), -1, np.int64)
    residual = np.full(len(points), np.nan)
    allowance = np.full(len(points), np.nan)
    depth_residual = np.full(len(points), np.nan)
    # BEGIN distinguish actual-point support from a continuous-cell allowance.
    support_kind = np.full(len(points), 'none', dtype='<U22')
    direct_allowance = np.full(len(points), np.nan)
    # END support provenance does not change the measurement association.
    active = np.flatnonzero(inside)
    xy = np.rint(pixels[active]).astype(np.int64)
    near = lookup[xy[:, 1], xy[:, 0]]
    exists = near >= 0
    state[active[~exists]] = _STATE['invalid_depth']
    active, xy, near = active[exists], xy[exists], near[exists]
    ids[active] = near
    if len(active):
        target = surface['observation_xyz'][near]
        z = surface['depth_m'][view_index][xy[:, 1], xy[:, 0]]
        ray_offset = np.column_stack((pixels[active]-xy, np.zeros(len(active)))) @ np.linalg.inv(K).T
        transverse = z*np.linalg.norm(ray_offset, axis=1)
        ray_norm = np.linalg.norm(np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(K).T, axis=1)
        measurement = query_error[active]+model.depth_error_m*ray_norm
        discretization = transverse*(1+model.max_depth_slope*ray_norm)
        bound = registration_error_m+measurement+discretization
        distance = np.linalg.norm(points[active]-target, axis=1)
        delta_z = camera[active, 2]-z
        axial_bound = (registration_error_m+query_error[active]+model.depth_error_m
                       +model.max_depth_slope*transverse)
        # BEGIN a measured point can support a query without interpolating the
        # adjacent depth cell. These strict bounds contain no transverse-pixel
        # or slope allowance, so a cut/missing neighboring pixel supplies no
        # invented geometry. The axial condition also preserves the stronger
        # depth restriction for off-axis rays with a wider world error bound.
        direct_bound = registration_error_m+measurement
        direct_axial_bound = registration_error_m+query_error[active]+model.depth_error_m
        direct = ((distance <= np.nextafter(direct_bound, np.inf))
                  & (np.abs(delta_z) <= direct_axial_bound))
        direct_allowance[active] = direct_bound
        # END direct support uses this same deterministic nearest original pixel.
        conflict = delta_z < -axial_bound
        behind = delta_z > axial_bound
        supported = (distance <= np.nextafter(bound, np.inf)) & ~(conflict | behind)
        state[active] = _STATE['association_unresolved']
        state[active[behind]] = _STATE['occluded']
        state[active[conflict]] = _STATE['depth_model_conflict']
        state[active[supported]] = _STATE['measurement_supported']
        member = surface['observation_entity_member'][near]
        state[active[supported & ~member]] = _STATE['outside_entity']
        # BEGIN: refutation must survive lateral position uncertainty too.
        conflict_rows = np.flatnonzero(conflict)
        continuous_error = _conflict_footprint_continuous(surface,
            camera[active[conflict_rows]], pixels[active[conflict_rows]],
            query_error[active[conflict_rows]]+registration_error_m, view_index)
        state[active[conflict_rows[~continuous_error]]] = _STATE['association_unresolved']
        # END: an uncertain footprint across a depth cut supplies no conflict.
        ties = np.any(np.abs(np.abs(pixels[active]-xy)-.5) <= numeric, axis=1)
        # BEGIN: any support that borrows the slope/pixel-rounding allowance
        # needs complete connected depth observations. Direct measured-point
        # support does not borrow that allowance and survives a neighboring
        # cut. Negative evidence keeps the original complete-cell restriction.
        lower, upper = np.floor(pixels[active]).astype(int), np.ceil(pixels[active]).astype(int)
        cell = np.column_stack((lookup[lower[:, 1], lower[:, 0]], lookup[lower[:, 1], upper[:, 0]],
                                lookup[upper[:, 1], lower[:, 0]], lookup[upper[:, 1], upper[:, 0]]))
        continuous = (cell >= 0).all(1)
        safe = np.maximum(cell, 0)
        for left, right in ((0, 1), (0, 2), (1, 3), (2, 3)):
            connected = np.asarray(surface['depth_adjacency'][safe[:, left], safe[:, right]]).ravel() > 0
            continuous &= (safe[:, left] == safe[:, right]) | connected
        state[active[~continuous & ~direct]] = _STATE['association_unresolved']
        # END: subpixel depth evidence qualification.
        state[active[ties]] = _STATE['pixel_association_ambiguous']
        # BEGIN attach support provenance only after all final state gates.
        position_supported = ((state[active] == _STATE['measurement_supported'])
                              | (state[active] == _STATE['outside_entity']))
        support_kind[active[position_supported & direct]] = 'direct_measurement'
        support_kind[active[position_supported & ~direct]] = 'continuous_pixel_model'
        # END ties, missing observations and conflicts never acquire provenance.
        # BEGIN report the bound that actually supplies the position evidence.
        effective_bound = np.where(position_supported & direct, direct_bound, bound)
        residual[active], allowance[active], depth_residual[active] = distance, effective_bound, delta_z
        # END unsupported rows retain the diagnostic pixel-model bound only.
    supported = state == _STATE['measurement_supported']
    position = supported | (state == _STATE['outside_entity'])
    conflict = state == _STATE['depth_model_conflict']
    return dict(supported=supported, position_supported=position, conflict=conflict,
        unknown=~(supported | conflict), entity_supported=supported,
        pixels=pixels, observation_indices=ids, states=state, state_names=list(STATE_NAMES),
        residual_m=residual, association_bound_m=allowance, camera_z_residual_m=depth_residual,
        support_kind=support_kind, direct_association_bound_m=direct_allowance,
        orientation_state=np.full(len(points), 'unknown'),
        query_normal_valid=np.zeros(len(points), bool))
# END curved measurement support; planar projection is independent and unchanged.


def bounded_members(surface, seed, radius, *, view_index=None):
    """Dijkstra in a safe Euclidean envelope; unknown/missing edges stay absent.

    ``view_index`` selects raw measurements in that camera for feature support.
    Otherwise the graph contains fused actual target representatives. The
    returned IDs are sorted and have actual graph path distance <= radius.
    """
    observed = view_index is not None
    points = surface['observation_xyz' if observed else 'xyz']
    labels = surface['observation_entity_member' if observed else 'entity_member']
    if not 0 <= seed < len(points) or not labels[seed]:
        return np.empty(0, np.int64), np.empty(0)
    key = '_observation_entity_' if observed else '_entity_'
    tree, ids = surface[key+'tree'], surface[key+'ids']
    chosen = ids[np.asarray(tree.query_ball_point(points[seed], np.nextafter(radius, np.inf), workers=1), np.int64)]
    if observed:
        chosen = chosen[surface['observation_view_index'][chosen] == view_index]
    chosen.sort()
    graph = surface['observation_adjacency' if observed else 'adjacency'][chosen][:, chosen]
    location = int(np.searchsorted(chosen, seed))
    distance = dijkstra(graph, directed=False, indices=location, limit=np.nextafter(radius, np.inf))
    good = distance <= radius*(1+1e-12)
    return chosen[good], distance[good]


def sample_region(surface, seed, *, sampling_radius_m=.03, sampling_spacing_m=.005):
    """Return only actual target measurements in a fixed graph-radius region."""
    _sampling_parameters(sampling_radius_m, sampling_spacing_m)
    members, distance = bounded_members(surface, int(seed), sampling_radius_m)
    return sample_members(surface, members, seed, sampling_radius_m=sampling_radius_m,
                          sampling_spacing_m=sampling_spacing_m, distances=distance)


def sample_members(surface, members, seed, *, sampling_radius_m=.03,
                   sampling_spacing_m=.005, distances=None):
    """Resample actual points after a region intersection, retaining seed connectivity.

    Radius always means distance in the fixed observation graph, including for
    an intersection of seed balls. Connectivity is checked in the retained
    membership, but removing vertices must not redefine that metric or impose
    a new radius from an arbitrarily ordered first seed. FPS is always rerun.
    """
    _sampling_parameters(sampling_radius_m, sampling_spacing_m)
    members = np.unique(np.asarray(members, np.int64))
    if seed not in members:
        raise ValueError('The retained region must contain its actual sampling seed')
    if distances is None:
        ball, original_distances = bounded_members(surface, int(seed), sampling_radius_m)
        members, _, positions = np.intersect1d(members, ball, return_indices=True)
        distances = original_distances[positions]
    graph = surface['adjacency'][members][:, members]
    _, components = connected_components(graph, directed=False)
    keep = components == components[int(np.searchsorted(members, seed))]
    members, distances = members[keep], np.asarray(distances)[keep]
    if not len(members):
        raise ValueError('A curved sampling seed needs an observed entity measurement')
    points = surface['xyz'][members]
    seed_local = int(np.searchsorted(members, seed))
    selected = np.zeros(len(members), bool)
    chosen = [seed_local]
    selected[seed_local] = True
    nearest = np.sum((points-points[seed_local])**2, axis=1)
    nearest[selected] = -np.inf
    while not selected.all():
        farthest = int(np.argmax(nearest))
        if nearest[farthest] < sampling_spacing_m**2*(1-1e-12):
            break
        chosen.append(farthest)
        selected[farthest] = True
        nearest = np.minimum(nearest, np.sum((points-points[farthest])**2, axis=1))
        nearest[selected] = -np.inf
    mask = np.zeros(len(surface['xyz']), bool)
    mask[members] = True
    return dict(seed_index=int(seed), center=surface['xyz'][seed].copy(), normal=None,
        mask=mask, sample_indices=members[np.asarray(chosen, np.int64)],
        diagnostics=dict(region_geometry='surface', sampling_metric='observed_graph_path_length',
            sampling_radius_m=float(sampling_radius_m), sampling_spacing_m=float(sampling_spacing_m),
            sampling_scale_applied=False, neighborhood_point_count=len(members), sample_count=len(chosen),
            maximum_graph_distance_m=float(distances.max(initial=0)),
            continuous_coverage_certified=False, orientation_state='unknown'))


def region_for_transform(surface, source, transform, *, registration_error_m,
                         sampling_radius_m=.03, sampling_spacing_m=.005):
    """Propagate frozen medoid anchors without projecting/moving frame F.

    Distinct seed associations are retained. Seeds connected within their
    declared association uncertainty describe a finite observed common region;
    disconnected seeds remain ambiguous instead of choosing an index winner.
    """
    anchors = apply_sim3(source['anchor_xyz'], *transform)
    projections = [project_measurements(surface, anchors, i,
        registration_error_m=registration_error_m,
        query_position_error_m=transform[0]*source['anchor_position_error_m']) for i in range(2)]
    by_anchor = []
    for row in range(len(anchors)):
        observations = [p['observation_indices'][row] for p in projections if p['supported'][row]]
        if any(p['conflict'][row] for p in projections):
            return None, 'source_anchor_depth_model_conflict'
        if not observations:
            return None, 'source_anchor_observation_unavailable'
        by_anchor.append(np.unique(surface['observation_to_element'][observations]))
    seeds = np.unique(np.concatenate(by_anchor)) if by_anchor else np.empty(0, np.int64)
    if not len(seeds):
        return None, 'source_anchor_observation_unavailable'
    modes = [sample_region(surface, int(seed), sampling_radius_m=sampling_radius_m,
                           sampling_spacing_m=sampling_spacing_m) for seed in seeds]
    common = np.logical_and.reduce([mode['mask'] for mode in modes])
    alternatives = [dict(seed_index=mode['seed_index'],
        region_indices=np.flatnonzero(mode['mask']).tolist(),
        sample_indices=mode['sample_indices'].tolist()) for mode in modes]
    anchor_record = dict(source_anchor_indices=source['anchor_indices'].tolist(),
        target_seed_indices=seeds.tolist(), seed_region_count=len(modes),
        anchor_alternatives=alternatives,
        anchor_region_combination='common_observed_intersection' if len(modes) > 1 else 'single_anchor',
        functional_frame_origin_changed=False)
    # A tie among source anchors can select truly different surface patches.
    # Require every seed to lie in the common supported patch, not just one
    # shared fringe pixel. The complete alternatives remain in diagnostics.
    if not common[seeds].all():
        modes[0]['diagnostics'].update(anchor_record)
        return modes[0], 'source_anchor_region_ambiguous'
    # Keep the complete fixed-radius region for a single seed. Multiple seeds
    # retain their separate regions; their intersection is the common output
    # evidence, and no invented centroid replaces an actual measurement.
    output = modes[0]
    if len(modes) > 1:
        output = sample_members(surface, np.flatnonzero(common), int(seeds[0]),
            sampling_radius_m=sampling_radius_m, sampling_spacing_m=sampling_spacing_m)
    output['diagnostics'].update(anchor_record)
    if not output['mask'][seeds].all():
        return output, 'source_anchor_region_ambiguous'
    return output, None
