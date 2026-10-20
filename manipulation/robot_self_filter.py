"""Robot identity and occlusion from registered observed link fields.

The maintained TSDF raycaster predicts image depths from the same saved
weighted observations. Its candidate hits are checked against the raw field's
interpolation support before they influence measurement admission.
"""

# BEGIN observed-field self filtering shares the registered robot model.
from collections.abc import Mapping
import numpy as np
import torch

from .robot_geometry_check import _name
from .implicit_distance import inverse_points
from .resource_admission import require_device_memory, require_host_memory

SELF = np.uint8(1)
ENVIRONMENT = np.uint8(2)
AMBIGUOUS_SHADOW = np.uint8(3)
_MAX_PIXELS = 2_000_000


def validate_self_filter(value, depth_m):
    """Validate the only production observation contract, without mutation.

    Unknown/SELF pixels contribute neither measured surfaces nor free rays.
    ENVIRONMENT rays retain exactly their original finite positive depth; an
    endpoint cannot be extended, guessed, or silently restored by a consumer.
    """
    depth = np.asarray(depth_m, dtype=float)
    if (depth.ndim != 2 or not depth.size or depth.size > _MAX_PIXELS
            or not isinstance(value, Mapping)
            or set(value) != {"labels", "free_ray_depth_m", "source"}):
        raise ValueError("Self filter needs bounded H,W depth and exactly labels/free_ray_depth_m/source")
    labels, free = np.asarray(value["labels"]), np.asarray(value["free_ray_depth_m"], dtype=float)
    if (labels.dtype != np.uint8 or labels.shape != depth.shape
            or not np.isin(labels, (SELF, ENVIRONMENT, AMBIGUOUS_SHADOW)).all()
            or free.shape != depth.shape or not np.isfinite(free).all() or np.any(free < 0)):
        raise ValueError("Self-filter labels must be aligned uint8 three-state values and free depths finite nonnegative")
    source = _name(value["source"], "self-filter source")
    environment = labels == ENVIRONMENT
    if (np.any(environment & (~np.isfinite(depth) | (depth <= 0)))
            or not np.array_equal(free[environment], depth[environment])
            or np.any(free[~environment] != 0)):
        raise ValueError("Only valid ENVIRONMENT pixels may retain their exact original free-ray depth")
    return {"labels": labels, "free_ray_depth_m": free, "source": source}



def classify_depth(*, observed_depth, K, T_world_camera, bodies, body_poses,
                   block_checkpoints, depth_error_m, source,
                   max_host_rss_bytes, max_device_used_bytes, device="cuda:0"):
    """Classify measured endpoints using complete named robot FK at capture.

    A strictly negative supported endpoint is SELF. A supported predicted hit
    before the endpoint, including its declared depth error, is ambiguous.
    Remaining valid pixels keep their original environment depth. Missing
    robot observations are reported without filling their hidden geometry.
    """
    from curobo.perception import Mapper, MapperCfg
    from curobo.types import DeviceCfg, Pose
    from .planning_observation import _mapper_memory_bound, _mapper_intrinsics
    depth = np.asarray(observed_depth, dtype=float)
    if depth.ndim != 2 or depth.size > _MAX_PIXELS or not np.isfinite(depth_error_m) or depth_error_m < 0:
        raise ValueError("Self filtering needs bounded measured depth and its nonnegative error")
    height, width = depth.shape
    y, x = np.indices(depth.shape)
    rays = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(np.asarray(K, float)).T
    valid = np.isfinite(depth) & (depth > 0)
    endpoints_camera = rays*np.where(valid, depth, 0.)[..., None]
    world_camera = np.asarray(T_world_camera, float)
    endpoints = endpoints_camera @ world_camera[:3, :3].T+world_camera[:3, 3]
    labels = np.where(valid, ENVIRONMENT, AMBIGUOUS_SHADOW).astype(np.uint8)
    nearest = np.full(depth.shape, np.inf)
    tensor = DeviceCfg(device)
    counts = {}
    for body in bodies:
        if body.field is None:
            counts[body.name] = {"status": "missing_observation"}
            continue
        field = body.field
        matrix = np.asarray(body_poses[body.name], float)
        values, supported, _, error = field.sample(inverse_points(endpoints, matrix))
        own = valid & supported & (values+error < 0)
        labels[own] = SELF
        extent = field.shape*field.voxel_size
        # Subtract one representable ULP before ceil to recover the integer
        # grid shape despite a floating product/division round-trip.
        extent = np.nextafter(extent, np.zeros(3))
        center = field.origin+(field.shape-1)*field.voxel_size/2
        config = MapperCfg(extent_meters_xyz=tuple(extent), voxel_size=field.voxel_size,
            extent_esdf_meters_xyz=tuple(extent), esdf_voxel_size=field.voxel_size,
            grid_center=torch.tensor(center, dtype=torch.float32), truncation_distance=2*field.voxel_size,
            image_height=height, image_width=width, depth_minimum_distance=.05,
            depth_maximum_distance=5., device=device)
        bound = _mapper_memory_bound(config)
        require_host_memory(bound, pending_existing_bytes=0, max_rss_bytes=max_host_rss_bytes)
        require_device_memory(torch, device, native_bytes=bound, max_device_used_bytes=max_device_used_bytes)
        mapper = Mapper.load_blocks(block_checkpoints[body.name], config)
        camera_local = np.linalg.inv(matrix) @ world_camera
        rendered, _, render_valid = mapper.render(tensor.to_device(_mapper_intrinsics(K)),
            Pose.from_matrix(tensor.to_device(camera_local)[None]), (height, width))
        rendered = rendered.detach().cpu().numpy().reshape(depth.shape)
        render_valid = render_valid.detach().cpu().numpy().reshape(depth.shape)
        candidates = render_valid & np.isfinite(rendered) & (rendered > 0)
        hits = rays*np.where(candidates, rendered, 0.)[..., None]
        hits = hits @ camera_local[:3, :3].T+camera_local[:3, 3]
        _, supported_hits, _, _ = field.sample(hits)
        accepted = candidates & supported_hits
        nearest = np.minimum(nearest, np.where(accepted, rendered, np.inf))
        counts[body.name] = {"status": "observed_partial", "self_pixels": int(own.sum()),
                             "supported_render_hits": int(accepted.sum())}
        del mapper
    shadow = valid & (nearest <= depth+depth_error_m) & (labels != SELF)
    labels[shadow] = AMBIGUOUS_SHADOW
    filtered = validate_self_filter({"labels": labels,
        "free_ray_depth_m": np.where(labels == ENVIRONMENT, depth, 0.), "source": source}, depth)
    return {"self_filter": filtered, "predicted_depth_m": np.where(np.isfinite(nearest), nearest, 0.),
            "provenance": {"model": "same_observed_robot_fields", "bodies": counts,
                           "depth_error_m": float(depth_error_m)}}
# END unknown raycast fill is never admitted as field support or collision geometry.
