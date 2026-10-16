"""Raw MARCO responses for the independent two-view transfer scheme.

Every score belongs to one fixed source descriptor and one continuous world
location. Geometry and complete interpolation support determine availability;
missing evidence remains NaN. This module neither ranks responses nor decodes
each camera into a separate match. The existing single-view modules are only
read for their native source-coordinate sampler.
"""

import numpy as np
from scipy.ndimage import binary_propagation

from ..features import sample_descriptors
from .geometry import project_oriented


def _crop(bounds, image_shape):
    """Validate recorded integer crop metadata without silently truncating it."""
    values = np.asarray(bounds)
    if (values.shape != (4,) or values.dtype.kind not in "iu"
            or values.dtype.kind == "b"):
        raise ValueError("crop_xyxy must contain four integer bounds")
    left, top, right, bottom = map(int, values)
    height, width = image_shape
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Recorded feature crop exceeds its original image")
    return values.astype(np.int64, copy=False)


def _queries(value):
    values = np.asarray(value)
    if (values.ndim != 2 or min(values.shape) < 1 or values.dtype.kind != "f"
            or not np.isfinite(values).all()):
        raise ValueError("Source descriptors must be finite nonempty [Q,D] floats")
    return values


def source_descriptors(source, query_indices, source_feature, source_crop):
    """Read fixed source identities with MARCO's native F/840 convention.

    Only the frozen query pixels need to be inside this crop. Source evidence
    outside a supplied entity segmentation is still a legitimate demonstration
    query. An absent/outside descriptor is an input error, never a deleted row.
    """
    ids = np.asarray(query_indices)
    pixels = np.asarray(source["pixels"])
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids)
            or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or np.any(ids >= len(pixels))):
        raise ValueError("query_indices must be unique valid source indices")
    if pixels.shape != (len(source["xyz"]), 2) or not np.isfinite(pixels).all():
        raise ValueError("Source pixels must align with source xyz")
    bounds = _crop(source_crop, np.asarray(source["object_mask"]).shape)
    local = pixels[ids] - bounds[:2]
    size = bounds[2:] - bounds[:2]
    if np.any(local < 0) or np.any(local > size - 1):
        raise ValueError("A frozen source query lies outside its recorded feature crop")
    return sample_descriptors(source_feature, local, tuple(size))


def model_bounds(queries, gamma, beta):
    """Sharp real-arithmetic bound from the final affine LayerNorm.

    A normalized D-vector y has zero mean and norm at most sqrt(D). Writing
    a=q*gamma gives q.f=q.beta+(a-mean(a)).y, so Cauchy--Schwarz yields the
    returned interval. Convex bilinear interpolation and valid-view max retain
    it. Outward endpoint rounding is only CPU arithmetic care; it does not
    certify the previously executed GPU LayerNorm or semantic correctness.
    """
    q = np.asarray(_queries(queries), dtype=np.float64)
    gamma, beta = np.asarray(gamma, dtype=np.float64), np.asarray(beta, dtype=np.float64)
    if (gamma.shape != (q.shape[1],) or beta.shape != gamma.shape
            or not np.isfinite(gamma).all() or not np.isfinite(beta).all()):
        raise ValueError("LayerNorm affine parameters must be finite descriptor-sized vectors")
    product = q * gamma
    centered = product - product.mean(axis=1, keepdims=True)
    radius = np.sqrt(q.shape[1]) * np.linalg.norm(centered, axis=1)
    bias = q @ beta
    return np.nextafter(bias-radius, -np.inf), np.nextafter(bias+radius, np.inf)


def prepare_view(queries, feature, *, crop_xyxy, observation, view_id):
    """Materialize only [Q,Hf,Wf] raw responses, then allow the feature to die.

    Callers load one existing feature map at a time. For a missing view pass
    feature=None; its crop and observation may also be None. The returned
    record always retains the camera identity and fixed query descriptors.
    """
    q = _queries(queries)
    if q.dtype != np.float32:
        raise ValueError("Prepared native responses require the frozen FP32 source descriptors")
    record = {"id": view_id, "response": None, "crop_xyxy": crop_xyxy,
              "observation": observation, "query_descriptors": q.copy()}
    if feature is None:
        return record
    values = np.asarray(feature)
    if (values.dtype != np.float32 or values.ndim != 4 or values.shape[0] != 1
            or values.shape[1] != q.shape[1] or min(values.shape[-2:]) < 2):
        raise ValueError("Expected an FP32 [1,D,Hf,Wf] target MARCO map")
    # Chunking never makes a second full feature-map copy and bounds the
    # finite-value check. FP32 multiplication preserves native raw dot scores.
    response = np.empty((len(q), *values.shape[-2:]), dtype=np.float32)
    flat = values[0].reshape(q.shape[1], -1)
    for start in range(0, flat.shape[1], 4096):
        block = flat[:, start:start+4096]
        if not np.isfinite(block).all():
            raise ValueError("Target feature map contains non-finite descriptors")
        response.reshape(len(q), -1)[:, start:start+4096] = q @ block
    record["response"] = response
    return record


def interval_score(scores, known, lower, upper):
    """Average fixed source rows: measured scores or their unknown model bounds."""
    scores, known = np.asarray(scores, dtype=float), np.asarray(known)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (scores.ndim != 1 or not len(scores) or known.dtype != np.bool_
            or known.shape != scores.shape or lower.shape != scores.shape
            or upper.shape != scores.shape or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(lower > upper)
            or not np.isfinite(scores[known]).all()):
        raise ValueError("Scores, known mask and finite bounds must share fixed Q rows")
    if np.any(known & ((scores < lower) | (scores > upper))):
        raise ValueError("Measured raw score violates the declared LayerNorm model bound")
    return {"lower": float(np.where(known, scores, lower).mean()),
            "upper": float(np.where(known, scores, upper).mean()),
            "coverage": float(known.mean())}


def _sampling_corners(feature_xy, shape):
    """Return unrenormalized positive bilinear corners, never outside clamping."""
    height, width = shape
    points = np.asarray(feature_xy, dtype=float)
    inside = (np.isfinite(points).all(1) & np.all(points >= 0, axis=1)
              & np.all(points <= [width-1, height-1], axis=1))
    safe = np.where(inside[:, None], points, 0)
    low = np.floor(safe).astype(np.int64)
    fraction = safe-low
    corners, weights = [], []
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        corner = low + [dx, dy]
        weight = (fraction[:, 0] if dx else 1-fraction[:, 0])
        weight = weight * (fraction[:, 1] if dy else 1-fraction[:, 1])
        weights.append(np.where(inside, weight, 0))
        corners.append(corner)
    return inside, np.stack(corners, axis=1), np.stack(weights, axis=1)


def _connected_in_boxes(surface, view_index, references, corner_pixels, positive, low, high):
    """Find fixed-reference 8-neighbor paths inside supplied pixel bounds.

    These boxes accelerate an exact graph query; they never alter normal,
    slab, entity, or physical-radius eligibility. Batch size follows temporary
    grid size rather than imposing a geometric cutoff on any query.
    """
    count, result = len(references), np.zeros((len(references), 4), bool)
    lookup = surface["pixel_to_observation"][view_index]
    xyz, normals = surface["observation_xyz"], surface["observation_normals"]
    obs_pixels = surface["observation_pixels"]
    valid = surface["observation_valid"] & surface["observation_entity_member"]
    structure = np.zeros((3, 3, 3), bool)
    structure[1] = True
    start = 0
    while start < count:
        stop = min(start+4096, count)
        sizes = high[start:stop]-low[start:stop]+1
        cells = int(sizes[:, 0].max()) * int(sizes[:, 1].max())
        stop = min(stop, start+max(1, 262144//max(1, cells)))
        refs, corners, used = references[start:stop], corner_pixels[start:stop], positive[start:stop]
        center = obs_pixels[refs]
        lo, hi = low[start:stop], high[start:stop]
        sizes = hi-lo+1
        yy, xx = np.mgrid[:int(sizes[:, 1].max()), :int(sizes[:, 0].max())]
        px, py = lo[:, 0, None, None]+xx, lo[:, 1, None, None]+yy
        in_box = (xx < sizes[:, 0, None, None]) & (yy < sizes[:, 1, None, None])
        inside = in_box & (px >= 0) & (py >= 0) & (px < lookup.shape[1]) & (py < lookup.shape[0])
        ids = lookup[np.clip(py, 0, lookup.shape[0]-1), np.clip(px, 0, lookup.shape[1]-1)]
        safe_ids = np.maximum(ids, 0)
        delta = xyz[safe_ids] - xyz[refs, None, None]
        reference_normal = normals[refs]
        plane_height = np.einsum("nhwi,ni->nhw", delta, reference_normal)
        normal_dot = np.einsum("nhwi,ni->nhw", normals[safe_ids], reference_normal)
        thick = surface["surface_height_spacing_factor"] * surface["observation_spacing"][refs]
        eligible = inside & (ids >= 0) & valid[safe_ids]
        eligible &= np.abs(plane_height) <= thick[:, None, None]
        eligible &= normal_dot >= surface["normal_min_agreement"]
        eligible &= np.linalg.norm(delta, axis=-1) <= surface["sampling_radius_m"]
        seeds = np.zeros(eligible.shape, bool)
        rows = np.arange(len(refs))
        seeds[rows, center[:, 1]-lo[:, 1], center[:, 0]-lo[:, 0]] = True
        connected = binary_propagation(seeds & eligible, structure=structure, mask=eligible)
        for column in range(4):
            x = corners[:, column, 0]-lo[:, 0]
            y = corners[:, column, 1]-lo[:, 1]
            active = used[:, column]
            result[start:stop, column][active] = connected[rows[active], y[active], x[active]]
        start = stop
    return result


def _sphere_pixel_bounds(surface, view_index, references):
    """Conservatively project the fixed 3-D sphere's enclosing camera cube."""
    radius = float(surface["sampling_radius_m"])
    pose, intrinsic = surface["T_world_camera_cv"][view_index], surface["K"][view_index]
    center = (surface["observation_xyz"][references]-pose[:3, 3]) @ pose[:3, :3]
    signs = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    homogeneous = (center[:, None]+radius*signs) @ intrinsic.T
    positive_depth = (homogeneous[:, :, 2] > 0).all(axis=1)
    height, width = surface["pixel_to_observation"][view_index].shape
    low = np.zeros((len(references), 2), dtype=np.int64)
    high = np.broadcast_to([width-1, height-1], low.shape).copy()
    ids = np.flatnonzero(positive_depth)
    if len(ids):
        uv = homogeneous[ids, :, :2]/homogeneous[ids, :, 2, None]
        lower = np.floor(np.nextafter(uv.min(axis=1), -np.inf))
        upper = np.ceil(np.nextafter(uv.max(axis=1), np.inf))
        low[ids] = np.maximum(lower, 0).astype(np.int64)
        high[ids] = np.minimum(upper, [width-1, height-1]).astype(np.int64)
    # Include the original pixel even if projective arithmetic rounded it one
    # ulp differently; this changes no exact-radius or local-plane test.
    pixel = surface["observation_pixels"][references]
    return np.minimum(low, pixel), np.maximum(high, pixel)


def _footprint_support(surface, view_index, references, corner_pixels, positive):
    """Check complete feature-corner support in one fixed local surface patch.

    Each positive corner must have this view's entity observation, valid
    plane, reference-oriented normal, fixed slab and original-pixel connection
    within the reference's 3-cm sphere. Zero-weight corners are irrelevant.
    A small pixel rectangle can prove a connection quickly. If it cannot,
    search the entire fixed sphere's conservative pixel bounding box, so a
    legitimate path around a missing pixel is not falsely discarded. Paths
    never change reference or borrow another camera's surface graph.
    """
    count = len(references)
    result = np.zeros((count, 4), bool)
    if not count:
        return result
    lookup = surface["pixel_to_observation"][view_index]
    xyz, normal = surface["observation_xyz"], surface["observation_normals"]
    x, y = corner_pixels[:, :, 0], corner_pixels[:, :, 1]
    inside = (x >= 0) & (y >= 0) & (x < lookup.shape[1]) & (y < lookup.shape[0])
    ids = lookup[np.clip(y, 0, lookup.shape[0]-1), np.clip(x, 0, lookup.shape[1]-1)]
    safe = np.maximum(ids, 0)
    delta = xyz[safe]-xyz[references, None]
    height = np.einsum("nki,ni->nk", delta, normal[references])
    dot = np.einsum("nki,ni->nk", normal[safe], normal[references])
    thickness = surface["surface_height_spacing_factor"] * surface["observation_spacing"][references]
    eligible = positive & inside & (ids >= 0) & surface["observation_valid"][safe]
    eligible &= surface["observation_entity_member"][safe]
    eligible &= (np.abs(height) <= thickness[:, None]) & (dot >= surface["normal_min_agreement"])
    eligible &= np.linalg.norm(delta, axis=-1) <= surface["sampling_radius_m"]
    center = surface["observation_pixels"][references]
    extent = np.where(eligible[:, :, None], corner_pixels, center[:, None])
    low, high = np.minimum(center, extent.min(axis=1)), np.maximum(center, extent.max(axis=1))
    result = _connected_in_boxes(surface, view_index, references, corner_pixels, eligible, low, high)
    unresolved = np.flatnonzero(np.any(eligible & ~result, axis=1))
    if len(unresolved):
        low, high = _sphere_pixel_bounds(surface, view_index, references[unresolved])
        result[unresolved] = _connected_in_boxes(surface, view_index, references[unresolved],
            corner_pixels[unresolved], eligible[unresolved], low, high)
    return result


class ResponseField:
    """Two prepared raw response maps with geometry-qualified world sampling."""

    def __init__(self, surface, queries, views, gamma, beta):
        from local_models.marco import resize_scales

        self.surface, self.queries = surface, _queries(queries)
        self.lower, self.upper = model_bounds(self.queries, gamma, beta)
        views = list(views)
        if len(views) != 2 or [v["id"] for v in views] != list(surface["view_ids"]):
            raise ValueError("Prepared views must match the two canonical surface identities in order")
        self.views = []
        for index, view in enumerate(views):
            record = dict(view)
            q = record["query_descriptors"]
            if not np.array_equal(q, self.queries):
                raise ValueError("Response maps belong to different frozen source descriptors")
            reason = None
            if record.get("observation") is None:
                reason = "missing_entity_observation"
            elif record.get("response") is None:
                reason = "missing_feature"
            elif record.get("crop_xyxy") is None:
                reason = "missing_crop"
            if reason is None:
                response = np.asarray(record["response"])
                if (response.ndim != 3 or response.shape[0] != len(self.queries)
                        or min(response.shape[-2:]) < 2 or not np.isfinite(response).all()):
                    raise ValueError("Prepared response must be finite [Q,Hf,Wf]")
                crop = _crop(record["crop_xyxy"], surface["pixel_to_observation"][index].shape)
                record.update(response=response, crop_xyxy=crop,
                              resize_scale=resize_scales(crop[2:]-crop[:2]))
            record["missing_reason"] = reason
            self.views.append(record)

    def sample(self, points, normals, query_rows=None):
        """Return all Q-by-N or selected-row scores at exactly these world points.

        ``query_rows=None`` gives scores[Q,N], view_scores[Q,N,2]. Otherwise
        each point selects one frozen source row, giving scores[N] and
        view_scores[N,2]. view_valid[N,2] records geometric/interpolation
        availability independently of which source descriptor is queried.
        """
        from local_models.marco import RESOLUTION

        points, normals = np.asarray(points, dtype=float), np.asarray(normals, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape:
            raise ValueError("World points and oriented normals must have equal [N,3] shape")
        rows = None if query_rows is None else np.asarray(query_rows)
        if rows is not None and (rows.shape != (len(points),) or rows.dtype.kind not in "iu"
                                 or np.any(rows < 0) or np.any(rows >= len(self.queries))):
            raise ValueError("query_rows must name one frozen source descriptor per point")
        shape = (len(self.queries), len(points)) if rows is None else (len(points),)
        view_scores = np.full((*shape, 2), np.nan)
        view_valid = np.zeros((len(points), 2), bool)
        states, geometry, counts = [], [], []
        for index, view in enumerate(self.views):
            projection = project_oriented(self.surface, points, normals, index)
            geometry.append(projection)
            names = np.asarray(projection["state_names"], dtype=object)
            state = names[np.asarray(projection["states"], dtype=int)].copy()
            # The geometry interface checks the continuous original-pixel
            # entity footprint even when the feature/crop itself is absent.
            position = np.asarray(projection["position_supported"], dtype=bool)
            plane = np.asarray(projection["plane_valid"], dtype=bool)
            oriented = np.asarray(projection["oriented_supported"], dtype=bool)
            normal_known = np.asarray(projection["query_normal_valid"], dtype=bool)
            state[position & ~plane] = "invalid_local_plane"
            state[position & plane & ~normal_known] = "missing_query_normal"
            state[position & plane & normal_known & ~oriented] = "opposite_query_normal"
            state[position & plane & oriented & ~projection["entity_supported"]] = "outside_entity_footprint"
            valid = np.asarray(projection["supported"], dtype=bool).copy()
            state[valid] = "appearance_pending"
            if view["missing_reason"] is not None:
                state[valid] = view["missing_reason"]
                valid[:] = False
            elif np.any(valid):
                response, crop = view["response"], view["crop_xyxy"]
                local = projection["pixels"]-crop[:2]
                in_crop = (np.isfinite(local).all(axis=1) & np.all(local >= 0, axis=1)
                           & np.all(local <= crop[2:]-crop[:2]-1, axis=1))
                state[valid & ~in_crop] = "outside_feature_crop"
                valid &= in_crop
                # shape[:0:-1] is [Wf,Hf]; response coordinates use F-1.
                grid = local * view["resize_scale"] * (np.array(response.shape[:0:-1])-1) / (RESOLUTION-1)
                inside, corners, weights = _sampling_corners(grid, response.shape[-2:])
                state[valid & ~inside] = "outside_feature_grid"
                valid &= inside
                active = np.flatnonzero(valid)
                if len(active):
                    selected_corners, positive = corners[active], weights[active] > 0
                    pixel_corners = np.rint(selected_corners * (RESOLUTION-1)
                        / (np.array(response.shape[:0:-1])-1) / view["resize_scale"] + crop[:2]).astype(np.int64)
                    # Feature padding has no original image/crop observation.
                    corner_inside = (np.all(pixel_corners >= crop[:2], axis=-1)
                                     & np.all(pixel_corners < crop[2:], axis=-1))
                    connected = _footprint_support(self.surface, index,
                        projection["observation_indices"][active], pixel_corners, positive)
                    supported = connected & corner_inside
                    complete = np.all(~positive | supported, axis=1)
                    any_support = np.any(positive & supported, axis=1)
                    state[active[~complete & any_support]] = "partial_feature_support"
                    state[active[~complete & ~any_support]] = "no_feature_support"
                    valid[active[~complete]] = False
                    chosen = active[complete]
                    if len(chosen):
                        values = np.zeros((len(self.queries), len(chosen))) if rows is None else np.zeros(len(chosen))
                        for corner in range(4):
                            weight = weights[chosen, corner]
                            positive_here = weight > 0
                            ids = chosen[positive_here]
                            xy = corners[ids, corner]
                            if rows is None:
                                values[:, positive_here] += response[:, xy[:, 1], xy[:, 0]] * weight[positive_here]
                            else:
                                values[positive_here] += response[rows[ids], xy[:, 1], xy[:, 0]] * weight[positive_here]
                        if rows is None:
                            view_scores[:, chosen, index] = values
                        else:
                            view_scores[chosen, index] = values
                        state[chosen] = "appearance_valid"
            view_valid[:, index] = valid
            states.append(state)
            unique, number = np.unique(state, return_counts=True)
            counts.append(dict(zip(unique.tolist(), number.tolist())))
        known_points = view_valid.any(axis=1)
        known = np.broadcast_to(known_points, shape).copy()
        scores = np.max(np.where(np.isfinite(view_scores), view_scores, -np.inf), axis=-1)
        scores = np.where(known, scores, np.nan)
        low = self.lower[:, None] if rows is None else self.lower[rows]
        high = self.upper[:, None] if rows is None else self.upper[rows]
        if np.any(np.isfinite(view_scores) & ((view_scores < low[..., None])
                                             | (view_scores > high[..., None]))):
            raise ValueError("Measured raw score violates the declared LayerNorm model bound")
        return {"scores": scores, "known": known, "view_scores": view_scores,
                "view_valid": view_valid, "view_state": np.stack(states, axis=1),
                "reason_counts": dict(zip(self.surface["view_ids"], counts)),
                "geometry": geometry}
