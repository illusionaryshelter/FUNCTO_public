"""Observed-point identities and independent-view surface association.

All coordinates are in metres and all pixels retain their original RGB-D
image coordinates. A candidate is an actually observed point, never a mesh
vertex, an averaged location, or a predicted hidden surface. Another view can
provide appearance evidence only when its local observed plane and the four
pixels enclosing the continuous projection support that same surface.

The inherited plane thickness is an observed surface association resolution,
not a sensor/calibration error bound. Unsupported views contribute no evidence;
even a front-of-surface conflict cannot delete an owner's real observation.
"""

import numpy as np
from scipy.ndimage import maximum_filter

from .core import TransferConfig, _pixels, _surface_geometry
from .segmentation import _calibration
from .sim3 import _points


# BEGIN independent multiview observation geometry; existing routes unchanged.
STATE_NAMES = (
    "owner_observed", "co_observed", "behind_camera", "outside_image",
    "invalid_depth", "outside_selected_mask", "invalid_local_plane",
    "corner_surface_mismatch", "outside_local_radius",
    "front_of_observed_surface", "behind_observed_surface", "invalid_projection",
)
_STATE = {name: index for index, name in enumerate(STATE_NAMES)}


def _target_arrays(target):
    """Validate the measured-image contract without reading feature or truth data.

    The caller supplies an observation made from this depth image. Checking its
    pixel membership here protects owner support from invalid depths and masks;
    no geometric distance tolerance is introduced to repair supplied points.
    """
    view_id = target["id"]
    if not isinstance(view_id, str) or not view_id:
        raise ValueError("Each target id must be a nonempty string")
    observation = target["observation"]
    xyz = _points(observation["xyz"], "target observation xyz")
    pixels = _pixels(observation["pixels"], len(xyz))
    K, T = _calibration(observation["K"], observation["T_world_camera_cv"])
    depth = np.asarray(target["depth_m"])
    mask = np.asarray(target["selected_mask"])
    if (depth.ndim != 2 or depth.dtype.kind not in "fiu"
            or min(depth.shape) < 1):
        raise ValueError("depth_m must be a nonempty numeric optical-depth image")
    if mask.dtype != np.bool_ or mask.shape != depth.shape:
        raise ValueError("selected_mask must be Boolean and aligned with full depth_m")
    object_mask = np.asarray(observation["object_mask"])
    if (object_mask.dtype != np.bool_ or object_mask.shape != mask.shape
            or not np.array_equal(object_mask, mask)):
        raise ValueError("selected_mask must equal the observation's original object_mask")
    height, width = depth.shape
    if (np.any(pixels < 0) or np.any(pixels[:, 0] >= width)
            or np.any(pixels[:, 1] >= height)):
        raise ValueError("Observation pixels must lie inside the original depth image")
    measured = depth[pixels[:, 1], pixels[:, 0]]
    if (not np.all(mask[pixels[:, 1], pixels[:, 0]])
            or not np.all(np.isfinite(measured) & (measured > 0))):
        raise ValueError("Every target observation must have selected-mask valid depth")
    return xyz, pixels, K, T, depth, mask


def build_candidate_bank(targets):
    """Union exactly two measured point sets, preserving every observation owner.

    Views use lexicographically sorted ids, observations retain their original
    per-view order, and candidates use NumPy's exact lexicographic XYZ union.
    There is no rounding, spatial weld radius, or coordinate averaging. Equal
    image pixels in different cameras remain different observation identities.
    Nearby but unequal world coordinates also remain different candidates.
    """
    targets = list(targets)
    if len(targets) != 2:
        raise ValueError("The independent fusion path requires exactly two target views")
    for target in targets:
        if not isinstance(target.get("id"), str) or not target["id"]:
            raise ValueError("Each target id must be a nonempty string")
    if len({target["id"] for target in targets}) != 2:
        raise ValueError("Target view ids must be unique")
    targets.sort(key=lambda target: target["id"])
    arrays = [_target_arrays(target) for target in targets]
    sizes = np.asarray([len(array[0]) for array in arrays], dtype=np.int64)
    observed = np.concatenate([array[0] for array in arrays], axis=0)
    unique, first, inverse = np.unique(observed, axis=0, return_index=True,
                                       return_inverse=True)
    return {
        "xyz": unique,
        "observation_xyz": observed,
        "observation_view_index": np.repeat(np.arange(2, dtype=np.int64), sizes),
        "observation_local_index": np.concatenate(
            [np.arange(size, dtype=np.int64) for size in sizes]),
        "observation_candidate_index": inverse.astype(np.int64, copy=False),
        "view_ids": [target["id"] for target in targets],
        "view_offsets": np.concatenate(([0], np.cumsum(sizes))).astype(np.int64),
        "candidate_first_observation_index": first.astype(np.int64, copy=False),
    }


def _association_planes(depth, mask, K, T, sampling_radius_m, config):
    """Fit inherited planes using complete full-depth neighborhoods around SAM.

    A square dilation by the existing pixel radius retains every full-depth
    neighbor that the existing plane estimator could request for a SAM pixel.
    The dilation is only a computation crop; it never enlarges SAM membership.
    No image from another camera enters this neighborhood or its pixel tree.
    """
    expanded = maximum_filter(mask, size=2 * config.normal_pixel_radius + 1,
                              mode="constant", cval=0)
    retained = expanded & np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero(retained)
    pixels = np.column_stack((x, y)).astype(np.int64)
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    points = (rays * depth[y, x, None]) @ T[:3, :3].T + T[:3, 3]
    lookup = np.full(depth.shape, -1, dtype=np.int64)
    lookup[y, x] = np.arange(len(points))
    if not len(points):
        return points, np.empty((0, 3)), np.empty(0), np.empty(0, bool), lookup

    # The original routine combines geometry validity with agreement to an
    # already-transformed source normal. Correspondence fusion has no fitted
    # transform yet. A zero wanted normal deliberately disables that unrelated
    # agreement gate; its returned final-valid mask MUST NOT be used here.
    # Finite ratios already encode the routine's minimum support and rank
    # tests. Reapply only its geometry, spacing and camera-facing gates below.
    normals, ratios, _, view_cosines, spacings, _ = _surface_geometry(
        points, pixels, np.zeros(3), T[:3, 3], sampling_radius_m, config)
    valid = np.isfinite(ratios) & (ratios <= config.max_planarity_ratio)
    valid &= np.isfinite(spacings) & (spacings > 0)
    valid &= view_cosines >= config.min_view_cosine
    valid &= np.linalg.norm(points - T[:3, 3], axis=1) > 0
    return points, normals, spacings, valid, lookup


def associate_view(bank, target, *, config=None, sampling_radius_m=.03):
    """Project the candidate bank and mark appearance support from one view.

    Owner candidates directly retain their known original pixel coordinates;
    this prevents round-trip arithmetic at an image boundary from losing an
    actual observation. Foreign candidates use continuous pinhole projection.
    Their floor/ceil corner pixels must all have selected-mask valid depth and
    lie in the nearest pixel's fixed local plane slab. The candidate must also
    lie in that slab and within the existing physical neighborhood radius.

    ``state`` indexes ``state_names``. A failed association is unknown to the
    fusion score, including front/behind-surface diagnostics. No state deletes
    a candidate from ``bank``. This is a discrete observed-surface model, not a
    continuous free-space, collision, or calibrated uncertainty certificate.
    """
    config = TransferConfig() if config is None else config
    if not isinstance(config, TransferConfig):
        raise ValueError("config must be a TransferConfig")
    if (isinstance(sampling_radius_m, (bool, np.bool_))
            or not np.isscalar(sampling_radius_m)
            or not np.isfinite(sampling_radius_m) or sampling_radius_m <= 0):
        raise ValueError("sampling_radius_m must be finite and positive")
    observed, observed_pixels, K, T, depth, mask = _target_arrays(target)
    if target["id"] not in bank["view_ids"]:
        raise ValueError("The target view does not own observations in this bank")
    view_index = bank["view_ids"].index(target["id"])
    start, end = bank["view_offsets"][view_index:view_index + 2]
    if not np.array_equal(bank["observation_xyz"][start:end], observed):
        raise ValueError("The target observation differs from the candidate bank")
    candidates = bank["xyz"]
    count = len(candidates)
    owner_candidates = bank["observation_candidate_index"][start:end]
    owner = np.zeros(count, dtype=bool)
    owner[owner_candidates] = True
    camera = (candidates - T[:3, 3]) @ T[:3, :3]
    projected = camera @ K.T
    pixels = np.full((count, 2), np.nan)
    finite_projection = np.isfinite(projected).all(axis=1) & (projected[:, 2] != 0)
    pixels[finite_projection] = projected[finite_projection, :2] / projected[finite_projection, 2, None]
    pixels[owner_candidates] = observed_pixels
    state = np.full(count, _STATE["invalid_projection"], dtype=np.int8)
    finite_projection &= np.isfinite(pixels).all(axis=1)
    state[finite_projection & (camera[:, 2] <= 0)] = _STATE["behind_camera"]
    height, width = depth.shape
    infront = finite_projection & (camera[:, 2] > 0)
    inframe = infront & (pixels[:, 0] >= 0) & (pixels[:, 0] <= width - 1)
    inframe &= (pixels[:, 1] >= 0) & (pixels[:, 1] <= height - 1)
    state[infront & ~inframe] = _STATE["outside_image"]
    state[owner] = _STATE["owner_observed"]
    active = np.flatnonzero(inframe & ~owner)
    fitted_points = 0
    fitted_valid = 0

    def retain(indices, passed, reason):
        """Record the first failed geometric condition without changing owners."""
        state[indices[~passed]] = _STATE[reason]
        return indices[passed]

    if len(active):
        # Exact integers duplicate corners intentionally: no neighboring ray
        # is needed to support a point measured at a pixel center. Subpixels
        # retain all four corners, so nearest-pixel rounding cannot cross an
        # unsupported image/depth/SAM boundary and extend its plane silently.
        low = np.floor(pixels[active]).astype(np.int64)
        high = np.ceil(pixels[active]).astype(np.int64)
        corners = np.stack((low, np.column_stack((high[:, 0], low[:, 1])),
                            np.column_stack((low[:, 0], high[:, 1])), high), axis=1)
        corner_depth = depth[corners[:, :, 1], corners[:, :, 0]]
        passed = np.all(np.isfinite(corner_depth) & (corner_depth > 0), axis=1)
        active = retain(active, passed, "invalid_depth")
        corners = corners[passed]
        passed = np.all(mask[corners[:, :, 1], corners[:, :, 0]], axis=1)
        active = retain(active, passed, "outside_selected_mask")
        corners = corners[passed]

        if len(active):
            points, normals, spacings, valid, lookup = _association_planes(
                depth, mask, K, T, sampling_radius_m, config)
            fitted_points, fitted_valid = len(points), int(valid.sum())
            nearest = np.rint(pixels[active]).astype(np.int64)
            plane_indices = lookup[nearest[:, 1], nearest[:, 0]]
            # All enclosing corners are valid SAM pixels, so the nearest
            # pixel must be present in the dilated full-depth computation bank.
            if np.any(plane_indices < 0):
                raise RuntimeError("Valid SAM pixels missing from the plane computation bank")
            passed = valid[plane_indices]
            active = retain(active, passed, "invalid_local_plane")
            corners, plane_indices = corners[passed], plane_indices[passed]
            if len(active):
                seed = points[plane_indices]
                normal = normals[plane_indices]
                thickness = config.surface_height_spacing_factor * spacings[plane_indices]
                corner_indices = lookup[corners[:, :, 1], corners[:, :, 0]]
                corner_heights = np.einsum("nki,ni->nk", points[corner_indices] - seed[:, None], normal)
                passed = np.all(np.abs(corner_heights) <= thickness[:, None], axis=1)
                active = retain(active, passed, "corner_surface_mismatch")
                seed, normal, thickness = seed[passed], normal[passed], thickness[passed]
                delta = candidates[active] - seed
                passed = np.linalg.norm(delta, axis=1) <= sampling_radius_m
                active = retain(active, passed, "outside_local_radius")
                delta, normal, thickness = delta[passed], normal[passed], thickness[passed]
                signed_height = np.einsum("ni,ni->n", delta, normal)
                same_surface = np.abs(signed_height) <= thickness
                state[active[same_surface]] = _STATE["co_observed"]
                # Normals face this camera. The sign is only a discrete plane
                # conflict diagnostic; neither branch removes an owner point.
                state[active[~same_surface & (signed_height > 0)]] = _STATE["front_of_observed_surface"]
                state[active[~same_surface & (signed_height <= 0)]] = _STATE["behind_observed_surface"]

    supported = owner | (state == _STATE["co_observed"])
    return {"view_id": target["id"], "pixels": pixels, "supported": supported,
            "owner": owner, "state": state, "state_names": list(STATE_NAMES),
            "diagnostics": {
                "candidate_count": count, "owner_count": int(owner.sum()),
                "co_observed_count": int(np.count_nonzero(state == _STATE["co_observed"])),
                "unsupported_count": int(np.count_nonzero(~supported)),
                "state_counts": {name: int(np.count_nonzero(state == i))
                                 for i, name in enumerate(STATE_NAMES)},
                "plane_computation_point_count": fitted_points,
                "valid_plane_count": fitted_valid,
                "surface_resolution_model": "existing_local_spacing_fixed_plane_slab",
                "sampling_radius_m": float(sampling_radius_m),
                "hard_free_space_veto": False}}
# END independent multiview observation geometry.
