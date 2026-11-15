"""Native MARCO responses at one shared set of calibrated world positions.

``views`` contain explicit ``id``, ``response`` [Q,Hf,Wf], ``crop_xyxy``,
``depth_m``, ``owner_mask``, ``observation_ids`` [H,W], ``K`` and
``T_world_camera_cv`` arrays. The response is the FP32 dot-product grid from
FUNCTO ``prepare_view``. Rows name fixed source identities; columns name the
caller's world points. This module neither changes those point identities nor
assigns geometric mass to their camera observations.

Geometry uses the recorded optical-depth error and transverse depth-slope
model. Native image interpolation has a separate footprint: descriptor image
context is not a claim that every feature-cell corner is one material surface.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from local_models.marco import RESOLUTION, resize_scales
from transfer.dual_view.features import _sampling_corners


# BEGIN native sampling and its existing PyTorch position derivative.
def sample_native_pixels(response, pixels, crop_xyxy):
    """Return [Q,N] native values, with a Torch graph for continuous pixels.

    Precondition: callers qualify image/crop/grid and observed geometry before
    treating these values as evidence. ``grid_sample`` padding is numerical
    behavior only. The public observed query below samples qualified points
    and leaves every unsupported response NaN.

    ``align_corners=True`` turns the target feature coordinate
    ``local * resize_scale * (F-1) / 839`` into the normalized coordinate below.
    The native dot-product grid remains unchanged; interpolation uses float64.
    """
    pixels = pixels.to(dtype=torch.float64)
    crop = np.asarray(crop_xyxy)
    offset = torch.as_tensor(crop[:2], dtype=pixels.dtype, device=pixels.device)
    scale = torch.as_tensor(resize_scales(crop[2:]-crop[:2]),
                            dtype=pixels.dtype, device=pixels.device)
    grid = 2*(pixels-offset)*scale/(RESOLUTION-1)-1
    values = torch.as_tensor(response, dtype=pixels.dtype, device=pixels.device)
    sampled = functional.grid_sample(values[None], grid[None, :, None, :],
        mode='bilinear', padding_mode='zeros', align_corners=True)
    return sampled[0, :, :, 0]


def sample_native_world(response, world, *, crop_xyxy, K, T_world_camera_cv):
    """Sample prequalified world points without detaching their Torch graph.

    Position derivatives describe the current valid interpolation piece.
    Owner labels, missing depth and depth discontinuities are discrete support
    conditions evaluated by ``query_view``; their boundaries have no derivative
    supplied by this function.
    """
    world = world.to(dtype=torch.float64)
    pose = torch.as_tensor(T_world_camera_cv, dtype=world.dtype, device=world.device)
    intrinsic = torch.as_tensor(K, dtype=world.dtype, device=world.device)
    camera = (world-pose[:3, 3]) @ pose[:3, :3]
    homogeneous = camera @ intrinsic.T
    pixels = homogeneous[:, :2]/homogeneous[:, 2, None]
    return sample_native_pixels(response, pixels, crop_xyxy)
# END original response coordinates and library differentiation.


def project_world(world, K, T_world_camera_cv):
    """Project world metres to original CV pixels and optical depth metres."""
    pose = np.asarray(T_world_camera_cv)
    camera = (np.asarray(world)-pose[:3, 3]) @ pose[:3, :3]
    homogeneous = camera @ np.asarray(K).T
    with np.errstate(divide='ignore', invalid='ignore'):
        pixels = homogeneous[:, :2]/homogeneous[:, 2, None]
    # Exact stored depth pixels can reproject an ulp away from their integer.
    # This only corrects floating point representation, not a physical radius.
    nearest = np.rint(pixels)
    roundoff = 64*np.finfo(float).eps*np.maximum(1., np.abs(pixels))
    pixels = np.where(np.abs(pixels-nearest) <= roundoff, nearest, pixels)
    return pixels, camera[:, 2]


# BEGIN one world point, explicit measured support, and per-view provenance.
def query_view(world, view, *, depth_error_m, max_depth_slope):
    """Qualify RGB-D support and sample this camera's native response grid.

    Positive image-pixel bilinear corners must have observed owner depth.
    Their optical-depth range, expanded by the recorded measurement error,
    must contain the query depth. Pairwise depth changes use the original
    transverse-ray slope bound. This is local observation compatibility, not
    a reconstructed surface, normal estimate or unique material association.
    Every requested point retains a state and original measurement identities.
    """
    world = np.asarray(world, dtype=float)
    pixels, optical_depth = project_world(world, view['K'], view['T_world_camera_cv'])
    depth = np.asarray(view['depth_m'])
    owner = np.asarray(view['owner_mask'])
    lookup = np.asarray(view['observation_ids'])
    image_inside, corners, weights = _sampling_corners(pixels, depth.shape)
    positive = weights > 0
    height, width = depth.shape
    px = np.clip(corners[:, :, 0], 0, width-1)
    py = np.clip(corners[:, :, 1], 0, height-1)
    corner_depth = depth[py, px]
    corner_observation = lookup[py, px]
    measured = np.isfinite(corner_depth) & (corner_depth > 0) & (corner_observation >= 0)
    depth_supported = np.all(~positive | measured, axis=1)
    owner_supported = np.all(~positive | owner[py, px], axis=1)
    depth_low = np.min(np.where(positive, corner_depth, np.inf), axis=1)-depth_error_m
    depth_high = np.max(np.where(positive, corner_depth, -np.inf), axis=1)+depth_error_m

    # This is the same optical-depth/transverse-ray convention as the saved
    # ObservationModel. No fitted normal, output radius or task label enters.
    inverse_intrinsic = np.linalg.inv(view['K'])
    slope_supported = np.ones(len(world), dtype=bool)
    for left in range(4):
        for right in range(left):
            delta_pixel = corners[:, left]-corners[:, right]
            delta_ray = np.column_stack((delta_pixel, np.zeros(len(world)))) @ inverse_intrinsic.T
            transverse = np.minimum(corner_depth[:, left], corner_depth[:, right]) * np.linalg.norm(delta_ray, axis=1)
            bound = max_depth_slope*transverse+2*depth_error_m
            active_pair = positive[:, left] & positive[:, right]
            slope_supported &= ~active_pair | (np.abs(corner_depth[:, left]-corner_depth[:, right]) <= bound)

    reason = np.full(len(world), 'observed', dtype='<U40')
    pending = np.ones(len(world), dtype=bool)

    def reject(condition, label):
        selected = pending & condition
        reason[selected] = label
        pending[selected] = False

    finite = np.isfinite(pixels).all(axis=1) & np.isfinite(optical_depth)
    reject(~finite, 'invalid_projection')
    reject(optical_depth <= 0, 'behind_camera')
    reject(~image_inside, 'outside_image')
    reject(~depth_supported, 'missing_depth_measurement')
    reject(~owner_supported, 'outside_owner')
    reject(~slope_supported, 'depth_discontinuity')
    reject(optical_depth < depth_low, 'depth_model_conflict')
    reject(optical_depth > depth_high, 'occluded')

    response = view['response']
    crop = np.asarray(view['crop_xyxy'])
    local = pixels-crop[:2]
    crop_inside = np.all(local >= 0, axis=1) & np.all(local <= crop[2:]-crop[:2]-1, axis=1)
    reject(~crop_inside, 'outside_feature_crop')
    feature_shape = response.shape[-2:]
    feature_xy = local*resize_scales(crop[2:]-crop[:2])*(np.array(feature_shape[::-1])-1)/(RESOLUTION-1)
    feature_inside, feature_corners, feature_weights = _sampling_corners(feature_xy, feature_shape)
    reject(~feature_inside, 'outside_feature_grid')

    values = np.full((len(response), len(world)), np.nan)
    active = np.flatnonzero(pending)
    if len(active):
        selected_pixels = torch.as_tensor(pixels[active], dtype=torch.float64)
        values[:, active] = sample_native_pixels(response, selected_pixels, crop).detach().numpy()
    observation_ids = np.where(positive & image_inside[:, None], corner_observation, -1)
    return dict(view_id=view['id'], response=values, valid=pending, unknown=~pending,
        reason=reason, projected_pixels=pixels, optical_depth_m=optical_depth,
        depth_interval_m=np.column_stack((depth_low, depth_high)),
        depth_pixel_xy=corners, depth_weights=weights, observation_ids=observation_ids,
        depth_footprint_supported=depth_supported, owner_supported=owner_supported,
        slope_supported=slope_supported, feature_xy=feature_xy,
        feature_corner_xy=feature_corners, feature_weights=feature_weights)


def query_views(world, views, *, source_row_ids, depth_error_m, max_depth_slope):
    """Return mean raw evidence for one point domain, retaining every view.

    View support counts normalize appearance evidence only. They do not create
    geometric samples, target capacity or a new owner statistic. No-view points
    remain unknown/NaN; adding the absent camera never creates zero evidence.
    """
    per_view = [query_view(world, view, depth_error_m=depth_error_m,
                          max_depth_slope=max_depth_slope) for view in views]
    valid = np.stack([entry['valid'] for entry in per_view], axis=1)
    values = np.stack([entry['response'] for entry in per_view], axis=-1)
    count = valid.sum(axis=1)
    total = np.where(valid[None], values, 0.).sum(axis=-1)
    fused = np.full(total.shape, np.nan)
    np.divide(total, count[None], out=fused, where=count[None] > 0)
    return dict(response=fused, valid=count > 0, unknown=count == 0,
        source_row_ids=np.asarray(source_row_ids), world_points=np.asarray(world),
        view_ids=np.asarray([entry['view_id'] for entry in per_view]),
        view_response=values, view_valid=valid, view_count=count, views=per_view)
# END missing evidence and measured camera sources remain explicit.
