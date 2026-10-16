"""Raw MARCO responses and observed support for two-view region transfer.

Every score belongs to one fixed source descriptor and one continuous world
location. Geometry and complete interpolation support determine availability;
missing evidence remains NaN. A frozen target-domain empirical reference turns
measured responses into nonnegative support; one-to-one measurement capacity
prevents repeated use of the same target element. This is an observed-support
objective, not a bound on unobserved raw responses or a match probability.
"""

import numpy as np
from scipy.ndimage import binary_propagation
from scipy.optimize import linear_sum_assignment

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


# BEGIN fixed-domain observed support replaces latent LayerNorm intervals.
class EmpiricalSupport:
    """Freeze each source row's measured target-domain response distribution.

    The reference domain must be constructed before evaluating transforms.
    Mid-CDF ranks handle ties symmetrically. Taking the positive part of the
    centered rank retains only support above the reference median. No latent
    score is imputed, and constant responses at their reference value carry
    no appearance support. Query rows with an empty reference stay unknown.
    """

    def __init__(self, reference_scores, known):
        scores, known = np.asarray(reference_scores), np.asarray(known)
        if (scores.ndim != 2 or not scores.shape[0] or scores.dtype.kind not in "fiu"
                or known.shape != scores.shape or known.dtype != np.bool_
                or not np.isfinite(scores[known]).all()):
            raise ValueError("Reference scores and Boolean availability must be aligned [Q,N]")
        self.references = tuple(np.sort(scores[row, mask])
                                for row, mask in enumerate(known))
        for reference in self.references:
            reference.flags.writeable = False
        self.diagnostics = {
            "definition": "positive_centered_mid_cdf_on_fixed_observed_target_domain",
            "query_count": len(self.references),
            "reference_counts": [len(row) for row in self.references],
            "constant_rows": [i for i, row in enumerate(self.references)
                              if len(row) and row[0] == row[-1]],
            "empty_rows": [i for i, row in enumerate(self.references) if not len(row)],
            "is_match_probability": False,
        }

    def utility(self, scores, query_rows):
        """Return support in [0,1], preserving unknown responses as NaN.

        ``query_rows`` broadcasts to ``scores.shape``; for a [Q,N] response
        matrix use ``arange(Q)[:,None]``. Nonfinite score sentinels and empty
        reference rows have no observed support value. Neither becomes a
        finite raw response. Search-sorted ranks equal the mean of the strict
        and weak empirical CDFs without constructing a response-size copy.
        """
        values, rows = np.asarray(scores, dtype=float), np.asarray(query_rows)
        if rows.dtype.kind not in "iu":
            raise ValueError("query_rows must contain integer source identities")
        try:
            rows = np.broadcast_to(rows, values.shape)
        except ValueError as error:
            raise ValueError("query_rows must broadcast to the response shape") from error
        if np.any(rows < 0) or np.any(rows >= len(self.references)):
            raise ValueError("query_rows contains an unknown source identity")
        output = np.full(values.shape, np.nan)
        if not values.size:
            return output

        def evaluate_row(reference, measured):
            """Vectorized lookup; no Python loop visits a target element."""
            left = np.searchsorted(reference, measured, side="left")
            right = np.searchsorted(reference, measured, side="right")
            return np.maximum((left + right) / len(reference) - 1., 0.)

        # The complete Q-by-N reference scan and the final Q-by-V scores use
        # one broadcast source identity per row. Keep these common paths O(QN)
        # and never expand the broadcast row IDs into a second Q-by-N array.
        if rows.ndim >= 2 and all(stride == 0 for stride in rows.strides[1:]):
            for first in range(len(values)):
                reference = self.references[int(rows[(first,) + (0,) * (rows.ndim - 1)])]
                if not len(reference):
                    continue
                finite = np.isfinite(values[first])
                output[first][finite] = evaluate_row(reference, values[first][finite])
            return output
        flat_values, flat_rows, flat_output = values.ravel(), rows.ravel(), output.ravel()
        finite = np.isfinite(flat_values)
        for row in np.unique(flat_rows[finite]):
            reference = self.references[int(row)]
            if not len(reference):
                continue
            selected = np.flatnonzero(finite & (flat_rows == row))
            flat_output[selected] = evaluate_row(reference, flat_values[selected])
        return output


def capacity_score(view_utilities, view_element_ids):
    """Maximize fixed-Q observed support with capacity one per surface element.

    Each row is one fixed source query, each column in the input is a camera.
    An edge uses that camera's own utility and actual measurement element ID.
    Duplicate camera observations of the same row/element keep their maximum
    utility. Positive real edges have cost 1-u; Q dummy columns have cost 1.
    Zero-support real edges are omitted so numerical ties with dummies cannot
    create apparent evidence or anchor correspondences.

    Hiding edges only shrinks the matching feasible set, so it cannot improve
    the score with frozen references and other edges held fixed. This does not
    prove a scale or rotation correct: distinct pixels can still be ambiguous.
    """
    utility, elements = np.asarray(view_utilities, dtype=float), np.asarray(view_element_ids)
    if (utility.ndim != 2 or not utility.shape[0] or not utility.shape[1]
            or elements.shape != utility.shape or elements.dtype.kind not in "iu"
            or np.any(elements < -1) or np.isinf(utility).any()
            or np.any(np.isfinite(utility) & ((utility < 0) | (utility > 1)))
            or np.any(np.isfinite(utility) & (elements < 0))):
        raise ValueError("Require [Q,V] utilities in [0,1]/NaN and valid measurement element IDs")
    positive = np.isfinite(utility) & (utility > 0) & (elements >= 0)
    domain = np.unique(elements[positive])
    q = len(utility)
    costs = np.full((q, len(domain) + q), np.inf)
    costs[:, len(domain):] = 1.
    edge_views = np.full((q, len(domain)), -1, dtype=np.int64)
    for row, view in zip(*np.nonzero(positive)):
        column = int(np.searchsorted(domain, elements[row, view]))
        cost = 1. - utility[row, view]
        # Stable camera order resolves equal utilities without changing J.
        if cost < costs[row, column]:
            costs[row, column], edge_views[row, column] = cost, view
    rows, columns = linear_sum_assignment(costs)
    real = columns < len(domain)
    matched_rows, matched_columns = rows[real], columns[real]
    row_utilities = np.zeros(q)
    selected_view = np.full(q, -1, dtype=np.int64)
    selected_elements = np.full(q, -1, dtype=np.int64)
    selected_view[matched_rows] = edge_views[matched_rows, matched_columns]
    selected_elements[matched_rows] = domain[matched_columns]
    row_utilities[matched_rows] = utility[matched_rows, selected_view[matched_rows]]
    matched = selected_view >= 0
    positive_row_count = int(positive.any(axis=1).sum())
    return {"score": float(row_utilities.sum() / q),
            "row_utilities": row_utilities, "matched": matched,
            "selected_view": selected_view, "element_ids": selected_elements,
            "matched_count": int(matched.sum()),
            "positive_row_count": positive_row_count,
            "capacity_lost_count": positive_row_count - int(matched.sum()),
            "fixed_query_denominator": q}
# END fixed-domain observed support and measurement capacity.


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

    # BEGIN response sampling no longer depends on latent model-score bounds.
    def __init__(self, surface, queries, views):
        from local_models.marco import resize_scales

        self.surface, self.queries = surface, _queries(queries)
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
    # END measured raw responses need no LayerNorm affine parameters.

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
        states, geometry, counts, orientation_known = [], [], [], []
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
            # BEGIN separate measured RGB availability from source orientation.
            # Missing source normals cannot certify oriented geometry, but a
            # real target plane with complete RGB support is still observable.
            # Known reversed normals remain disallowed; no direction is filled.
            state[position & plane & normal_known & ~oriented] = "opposite_query_normal"
            orientation_eligible = oriented | ~normal_known
            state[position & plane & orientation_eligible & ~projection["entity_supported"]] = "outside_entity_footprint"
            valid = position & plane & orientation_eligible & projection["entity_supported"]
            orientation_known.append(normal_known)
            # END RGB eligibility does not change the geometry support record.
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
                        # BEGIN preserve unknown source orientation explicitly.
                        state[chosen] = np.where(normal_known[chosen], "appearance_valid",
                                                 "appearance_valid_orientation_unknown")
                        # END appearance-only support is distinguishable.
            view_valid[:, index] = valid
            states.append(state)
            unique, number = np.unique(state, return_counts=True)
            counts.append(dict(zip(unique.tolist(), number.tolist())))
        known_points = view_valid.any(axis=1)
        known = np.broadcast_to(known_points, shape).copy()
        scores = np.max(np.where(np.isfinite(view_scores), view_scores, -np.inf), axis=-1)
        scores = np.where(known, scores, np.nan)
        # BEGIN return measured responses and orientation provenance directly.
        return {"scores": scores, "known": known, "view_scores": view_scores,
                "view_valid": view_valid, "view_state": np.stack(states, axis=1),
                "view_orientation_known": np.stack(orientation_known, axis=1),
                "reason_counts": dict(zip(self.surface["view_ids"], counts)),
                "geometry": geometry}
        # END missing raw scores remain NaN; the fixed-domain objective is separate.
