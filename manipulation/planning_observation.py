"""Calibrated RGB-D partitions to fixed observed entity fields and free cells."""

# BEGIN one measured geometry representation for every scene entity.
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import math
from numbers import Real
import numpy as np
# BEGIN restore the existing one-pixel semantic boundary operation.
from scipy.ndimage import binary_dilation, minimum_filter
# END use the installed SciPy implementation without a new dependency.
import torch
from .motion import _inverse, _pose_array
from .observations import _intrinsics
from .planning_problem import PosePath
from .resource_admission import (DEFAULT_MAX_DEVICE_USED_BYTES, DEFAULT_MAX_HOST_RSS_BYTES,
    require_device_memory, require_host_memory)
from .planning_scene import CollisionOwner, ObservedSpace, _text

TASK_VOXEL_SIZE_M = .002
DEFAULT_ENVIRONMENT_VOXEL_SIZE_M = .01
MAX_FINE_CELLS = 4_000_000

def _views(views, reference_frame, depth_range_m):
    """Normalize an explicitly calibrated snapshot without filling bad depth."""
    _text(reference_frame, "RGB-D planning reference frame")
    limits = np.asarray(depth_range_m, float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or not 0 <= limits[0] < limits[1]:
        raise ValueError("Depth range must be explicit ordered finite metres")
    if not isinstance(views, Mapping) or not 0 < len(views) <= 16:
        raise ValueError("A current snapshot needs one to sixteen identified RGB-D views")
    result, pixels = {}, 0
    for identity, record in views.items():
        _text(identity, "current observation ID")
        if not isinstance(record, Mapping) or record.get("reference_frame") != reference_frame:
            raise ValueError("Every current view must declare the same calibrated planning frame")
        raw_depth = np.asarray(record["depth"])
        if raw_depth.ndim != 2 or min(raw_depth.shape) < 3:
            raise ValueError("Each registered metric depth image must be at least 3 by 3")
        pixels += raw_depth.size
        if pixels > 8_000_000:
            raise ValueError("Current RGB-D views exceed the bounded eight-million-pixel input")
        depth = np.asarray(raw_depth, float)
        matrix = _intrinsics(record["K"]).copy()
        # CameraObservation/Mapper use fx,fy,cx,cy. Silently passing skew or a
        # different projection model would make geometry disagree with rays.
        if abs(matrix[0, 1])+abs(matrix[1, 0]) > 1e-12:
            raise ValueError("The native mapper requires rectified zero-skew pinhole intrinsics")
        valid = np.isfinite(depth) & (depth >= limits[0]) & (depth <= limits[1]) & (depth > 0)
        result[identity] = {**record, "depth": np.where(valid, depth, 0.), "K": matrix,
                            "T_output_camera": _pose_array(record["T_output_camera"], "T_output_camera").copy()}
    return result


def _mapper_intrinsics(matrix):
    """Preserve input rays under the Mapper's half-integer pixel convention.

    Input calibration places pixel (u, v) at the integer centre used by
    backprojection and free-ray evidence. The native depth integrator emits
    allocation rays at (u+.5, v+.5) and casts positive projected coordinates
    to integers when reading depth. Adding .5 to both principal coordinates
    preserves the physical rays and nearest-centre depth selection. The
    caller's calibration is copied, so geometry outside this Mapper boundary
    keeps its original coordinates; RGB/feature kernels are not redefined.
    """
    native = _intrinsics(matrix).copy()
    native[:2, 2] += .5
    return native


def _rectangle_min_depth(depth, lower_xy, upper_xy):
    """Query inclusive, in-image integer rectangles with bounded image storage.

    For width w take a=2**floor(log2(w)), and similarly b for height h.
    The four a-by-b blocks anchored inside the rectangle at its corners cover
    it exactly: a<=w<2a and b<=h<2b. Their minimum is the rectangle minimum,
    without enlarging its footprint. SciPy owns the sliding-minimum operation;
    only one block-size image is retained at a time, with no persistent cache.
    The caller has already checked the full continuous projection's bounds.
    """
    result = np.empty(len(lower_xy), dtype=depth.dtype)
    sizes = upper_xy-lower_xy+1
    blocks = 2**np.floor(np.log2(sizes)).astype(np.int64)
    for block in np.unique(blocks, axis=0):
        selected = (blocks == block).all(1)
        width, height = map(int, block)
        # Negative half-size origin makes each value the minimum of the block
        # starting at that pixel and extending right/down, including size one.
        filtered = minimum_filter(depth, size=(height, width),
            origin=(-(height//2), -(width//2)), mode="constant", cval=0.)
        left, top = lower_xy[selected].T
        right, bottom = (upper_xy[selected]-block+1).T
        result[selected] = np.minimum(
            np.minimum(filtered[top, left], filtered[top, right]),
            np.minimum(filtered[bottom, left], filtered[bottom, right]))
        del filtered
    return result


def _known_free_voxels(views, *, ray_depths, origin, shape, voxel_size, depth_error_m):
    """Prove each complete voxel free under the supplied pixel-ray model.

    With all camera-z coordinates positive, the perspective projection of a
    convex combination of corners is a positive depth-weighted combination
    of their projections. Thus the eight projected corners bound the entire
    cube, including after camera rotation. The rectangle's minimum ray depth
    must exceed the furthest corner's camera-z by depth_error_m.
    ``ray_depths`` is mandatory: online callers pass self-filtered free-ray
    evidence; historical audit callers explicitly select original depth.
    Pixel quantization matches observations.backproject: continuous coordinates
    must lie in [0,W-1]x[0,H-1], then use nearest-pixel np.rint. Missing rays
    remain zero. Subpixel material and unprovided calibration error are not
    certified by this discrete sensor model.
    """
    shape = tuple(shape)
    origin = np.asarray(origin, float)
    if (len(shape) != 3 or any(type(n) is not int or n < 2 for n in shape)
            or math.prod(shape) > 8_000_000 or origin.shape != (3,) or not np.isfinite(origin).all()
            or not np.isfinite(voxel_size) or voxel_size <= 0
            or not np.isfinite(depth_error_m) or depth_error_m < 0):
        raise ValueError("Coverage needs a bounded z,y,x grid and explicit metric error allowance")
    if not isinstance(ray_depths, Mapping) or set(ray_depths) != set(views):
        raise ValueError("Free-ray depths must explicitly identify every calibrated view")
    for name, view in views.items():
        rays = np.asarray(ray_depths[name])
        if (rays.shape != view["depth"].shape or not np.isfinite(rays).all()
                or np.any(rays < 0) or np.any(rays > view["depth"])):
            raise ValueError("Free-ray depths must be aligned finite nonnegative measured prefixes")
    # BEGIN complete-cube projection replaces the expanded sphere/square test.
    cells, chunk = math.prod(shape), 65_536
    # Four float64 image buffers bound the streaming separable SciPy filter;
    # 256 float64 values per cell cover simultaneous eight-corner geometry,
    # propagated roundoff, projections, selections and rectangle queries.
    # Admission precedes the online Mapper's separate allocation boundary.
    largest_image = max(np.asarray(row['depth']).size for row in views.values())
    require_host_memory(int(cells+32*largest_image+2048*min(cells, chunk)),
        pending_existing_bytes=0, max_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES)
    known = np.zeros(cells, bool)
    offsets = (np.indices((2, 2, 2)).reshape(3, -1).T-.5)*voxel_size
    eps = np.finfo(np.float64).eps
    for name, view in views.items():
        depth = np.asarray(ray_depths[name], dtype=np.float64)
        matrix, transform = np.asarray(view['K'], float), np.asarray(view['T_output_camera'], float)
        height, width = depth.shape
        rotation, translation = transform[:3, :3], transform[:3, 3]
        for start in range(0, len(known), chunk):
            indices = np.arange(start, min(start+chunk, len(known)))
            indices = indices[~known[indices]]
            if not len(indices):
                continue
            z, y, x = np.unravel_index(indices, shape)
            lattice = np.column_stack((x, y, z))
            corners = origin+voxel_size*lattice[:, None, :]+offsets
            camera = (corners-translation)@rotation
            # Fixed operation-scale float64 protection includes cancellation
            # while forming world coordinates, not just the small final point.
            # This is numerical protection, not a configurable sensor margin.
            scale = np.abs(origin)+voxel_size*(np.abs(lattice)+.5)+np.abs(translation)
            camera_error = 64*eps*(scale@np.abs(rotation))
            front = ((camera[..., 2]-camera_error[:, None, 2] > 1e-9).all(1)
                     & np.isfinite(camera).all((1, 2)) & np.isfinite(camera_error).all(1))
            indices, camera, camera_error = indices[front], camera[front], camera_error[front]
            if not len(indices):
                continue
            numerator = camera@matrix[:2].T
            uv = numerator/camera[..., 2, None]
            numerator_error = (camera_error@np.abs(matrix[:2].T))[:, None, :]
            numerator_error = numerator_error+16*eps*(np.abs(camera)@np.abs(matrix[:2].T))
            z_error = camera_error[:, None, 2, None]
            uv_error = ((numerator_error+np.abs(uv)*z_error)/(camera[..., 2, None]-z_error)
                        +16*eps*(1+np.abs(uv)))
            lower = np.nextafter((uv-uv_error).min(1), -np.inf)
            upper = np.nextafter((uv+uv_error).max(1), np.inf)
            furthest = np.nextafter((camera[..., 2]+camera_error[:, None, 2]).max(1)+depth_error_m, np.inf)
            # Check the complete continuous range before quantization; a
            # negative subpixel coordinate rounding to zero is still outside.
            inside = (np.isfinite(lower).all(1) & np.isfinite(upper).all(1)
                      & (lower >= 0).all(1) & (upper <= [width-1, height-1]).all(1)
                      & np.isfinite(furthest))
            lower_xy = np.rint(lower[inside]).astype(np.int64)
            upper_xy = np.rint(upper[inside]).astype(np.int64)
            low = _rectangle_min_depth(depth, lower_xy, upper_xy)
            known[indices[inside][furthest[inside] < low]] = True
    # END no enlarged pixel footprint, clipping, ray restoration or old fallback.
    return known.reshape(shape)


def _self_filtered_views(views, reference_frame, depth_range_m):
    """Validate measured classification before applying mapper depth limits.

    A sensor-valid measurement can lie outside this mapper's configured range.
    Such a pixel becomes ambiguous for this map and contributes neither an
    endpoint nor a free ray. No caller arrays or original depths are changed.
    """
    from .robot_self_filter import AMBIGUOUS_SHADOW, validate_self_filter
    normalized = _views(views, reference_frame, depth_range_m)
    for name, original in views.items():
        if "self_filter" not in original:
            raise ValueError("Online RGB-D views require explicit three-state self_filter evidence")
        evidence = validate_self_filter(original["self_filter"], original["depth"])
        valid = normalized[name]["depth"] > 0
        labels = evidence["labels"].copy()
        labels[~valid] = AMBIGUOUS_SHADOW
        normalized[name]["self_filter"] = {
            "labels": labels,
            "free_ray_depth_m": np.where(valid, evidence["free_ray_depth_m"], 0.),
            "source": evidence["source"],
        }
    return normalized


def _partition_masks(views, owner_ids, *, build_background=True):
    # BEGIN preserve entity masks and protect their uncertain image boundaries.
    """Partition measured endpoints with one eight-neighbour boundary ring.

    Check owner/owner overlap on the supplied masks before self filtering;
    uncertainty is not permission to silently choose a semantic winner.
    Only entities being integrated appear in owner_ids. A supplied background
    mask selects the static endpoints of the original observation independently
    of retained moving entities. Without that mask, the new-owner complement
    defines the background candidates. Environment pixels on the nonbackground
    side seed one 3-by-3 dilation; adjacent background candidates are withheld
    from integration. They acquire neither an entity identity nor free-space
    evidence. SELF, shadow and invalid-depth pixels do not seed this ring.
    Entity masks and independently filtered free rays remain unchanged.
    Retained background bypasses this integration.
    """
    from .robot_self_filter import ENVIRONMENT, validate_self_filter
    masks, observed = {}, {identity: 0 for identity in owner_ids}
    for name, view in views.items():
        image, depth = np.asarray(view["image"]), view["depth"]
        if image.dtype != np.uint8 or image.shape != (*depth.shape, 3):
            raise ValueError("Current RGB must be an aligned uint8 H,W,3 image")
        supplied = view["owner_masks"]
        if not isinstance(supplied, Mapping) or set(supplied) != set(owner_ids):
            raise ValueError("Each view must explicitly include every selected owner mask, including empty masks")
        if "self_filter" not in view:
            raise ValueError("Online RGB-D views require explicit three-state self_filter evidence")
        evidence = validate_self_filter(view["self_filter"], depth)
        environment = evidence["labels"] == ENVIRONMENT
        occupied = np.zeros(depth.shape, bool)
        masks[name] = {}
        for identity in owner_ids:
            mask = np.asarray(supplied[identity])
            if mask.dtype != np.bool_ or mask.shape != depth.shape:
                raise ValueError("Visible owner masks must be aligned boolean images")
            if np.any(mask & occupied):
                raise ValueError("Visible owner pixel identities must be disjoint before self filtering")
            occupied |= mask
            masks[name][identity] = mask & environment
            observed[identity] += int(np.count_nonzero(masks[name][identity] & (depth > 0)))
        # BEGIN background admission is independent of retained entity visibility.
        if build_background:
            background = view.get("background_mask")
            if background is None:
                background = ~occupied
            else:
                background = np.asarray(background)
                if background.dtype != np.bool_ or background.shape != depth.shape:
                    raise ValueError("A supplied background partition must be an aligned boolean image")
                if np.any(background & occupied):
                    raise ValueError("Background and newly mapped entity partitions must be disjoint")
            # The explicit static partition also excludes retained entities.
            # Its complement supplies their original image boundary without
            # requiring those entities to be segmented or integrated again.
            boundary = environment & background & binary_dilation(
                environment & ~background, structure=np.ones((3, 3), bool), iterations=1)
            masks[name][None] = environment & background & ~boundary
            masks[name][None].setflags(write=False)
        # END retained entities do not acquire a new segmentation requirement.
    if any(count == 0 for count in observed.values()):
        raise ValueError("Every explicit collision owner needs current valid depth evidence after self filtering")
    return masks
    # END the boundary changes background admission only.


def _self_filter_report(views):
    """Record excluded semantic pixels without converting unknown to free."""
    from .robot_self_filter import SELF, ENVIRONMENT, AMBIGUOUS_SHADOW
    reports = {}
    for name, view in views.items():
        evidence = view["self_filter"]
        labels = evidence["labels"]
        reports[name] = {
            "source": evidence["source"],
            "self_pixels": int(np.count_nonzero(labels == SELF)),
            "environment_pixels": int(np.count_nonzero(labels == ENVIRONMENT)),
            "ambiguous_shadow_pixels": int(np.count_nonzero(labels == AMBIGUOUS_SHADOW)),
            "owners": {
                identity: {
                    "semantic_pixels": int(np.count_nonzero(mask)),
                    "excluded_self_pixels": int(np.count_nonzero(mask & (labels == SELF))),
                    "excluded_ambiguous_shadow_pixels": int(np.count_nonzero(mask & (labels == AMBIGUOUS_SHADOW))),
                    "environment_pixels": int(np.count_nonzero(mask & (labels == ENVIRONMENT))),
                } for identity, mask in view["owner_masks"].items()
            },
        }
    return reports


def _mapper_memory_bound(config):
    """Conservative native buffer budget from the installed MapperCfg fields.

    Includes the real 10,000-block floor at block_size=8, TSDF/RGB storage,
    hashes and per-block metadata, camera candidate/support buffers, dense
    ESDF+PBA storage, retained collision-field copies, and a 2x scratch margin.
    CUDA context, loaded models and allocator caches are additional process
    costs; this bound is checked before creating any mapper.
    """
    cells = math.prod(math.ceil(x/config.esdf_voxel_size) for x in config.extent_esdf_meters_xyz)
    pixels = config.image_height*config.image_width
    samples = math.ceil(2*config.truncation_distance/(config.block_size*config.voxel_size/1.42))+1
    blocks = config.max_blocks
    tsdf = blocks*(config.block_size**3*4+config.color_grid_size**3*8+128)
    hashes = config.hash_capacity*16
    camera = pixels*(16+samples*16)+blocks*(32+config.max_support_pixels_per_block_camera*8)
    esdf = cells*128
    return int(2*(tsdf+hashes+camera+esdf)+16*1024**2)


def scene_task_owner_ids(scene):
    """Read the manipulated body and functional reference from a bound scene.

    These are semantic identities already established by task grounding. An
    additional collision owner, including an initial support surface, is an
    environment owner unless it is also the current functional reference.
    No object name, dataset, geometry size or observed success selects a role.
    """
    if not isinstance(scene, Mapping):
        raise ValueError('Task ownership requires a bound scene mapping')
    carrier = _text(scene.get('carrier_id'), 'task carrier identity')
    reference = scene.get('reference_id')
    if reference is None:
        return (carrier,)
    reference = _text(reference, 'task functional reference identity')
    if reference == carrier:
        raise ValueError('Carrier and functional reference must be distinct owners')
    return (carrier, reference)


def observation_resolutions(owner_ids, *, task_owner_ids, environment_voxel_size):
    """Resolve the common 2 mm task / coarser environment mapping contract.

    Task identities select the shared task resolution. Reused entity fields
    preserve their original samples. Resource admission checks each new map
    at the requested spacing.
    """
    if isinstance(owner_ids, (str, bytes)) or isinstance(task_owner_ids, (str, bytes)):
        raise ValueError('Owner identities must be explicit collections')
    owners, task = tuple(owner_ids), tuple(task_owner_ids)
    for identity in (*owners, *task):
        _text(identity, 'observation owner identity')
    if (len(set(owners)) != len(owners) or len(set(task)) != len(task)
            or not set(task) <= set(owners)):
        raise ValueError('Task owners must be distinct registered observation owners')
    if (isinstance(environment_voxel_size, (bool, np.bool_))
            or not isinstance(environment_voxel_size, Real)
            or not np.isfinite(environment_voxel_size)
            or environment_voxel_size <= TASK_VOXEL_SIZE_M):
        raise ValueError('Environment grid spacing must be finite and coarser than 2 mm')
    return {identity: TASK_VOXEL_SIZE_M if identity in task else float(environment_voxel_size)
            for identity in owners}


def _owner_mapper_config(config, views, masks, owner_id, pose, voxel_size, depth_error_m):
    """Enclose every labelled depth sample in a finite owner-local grid.

    The background and raw free-space grid keep their own resolution. This
    tighter grid spends precision on the observed owner's surfaces without
    asserting unseen geometry or dropping a labelled surface to save memory.
    Padding covers truncation and complete sample cells; memory admission is
    still applied to all partitions together before any native allocation.
    """
    if (isinstance(voxel_size, (bool, np.bool_)) or not isinstance(voxel_size, Real)
            or not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("Owner field resolution must be explicit positive metres")
    voxel_size = float(voxel_size)
    clouds = []
    inverse = _inverse(pose)
    for name, view in views.items():
        rows, columns = np.nonzero(masks[name][owner_id] & (view["depth"] > 0))
        pixels = np.column_stack((columns, rows, np.ones(len(rows))))
        camera = (pixels @ np.linalg.inv(view["K"]).T) * view["depth"][rows, columns, None]
        transform = inverse @ view["T_output_camera"]
        clouds.append(camera @ transform[:3, :3].T + transform[:3, 3])
    points = np.concatenate(clouds)
    if not len(points):
        raise ValueError("A refined owner field needs current labelled valid depth samples")
    lower, upper = points.min(axis=0), points.max(axis=0)
    # Padding retains complete interpolation cells around measured endpoints.
    padding = 4*voxel_size+depth_error_m
    extent = tuple(upper-lower+2*padding)
    return replace(config, grid_center=torch.as_tensor((lower+upper)/2, dtype=torch.float32),
        extent_meters_xyz=extent, extent_esdf_meters_xyz=extent,
        voxel_size=voxel_size, esdf_voxel_size=voxel_size,
        truncation_distance=2*voxel_size, max_visible_blocks_per_integration=None)


def _fine_mapper_config(config, request):
    """Freeze a caller-evidenced small region in this partition's local frame.

    The request is generic metric geometry, not an object/task heuristic.
    Sensor allowance is supplied with its measurement evidence. Numerical
    support is separately computed from this Mapper's truncation and samples.
    """
    center = np.asarray(request["center_m"], float)
    extent = np.asarray(request["extent_m"], float)
    size = float(request["voxel_size_m"])
    margin = float(request["measurement_allowance_m"])
    if (center.shape != (3,) or extent.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(extent).all()
            or np.any(extent <= 0) or not np.isfinite([size, margin]).all()
            or not 0 < size < config.esdf_voxel_size or margin < 0):
        raise ValueError("Fine ROI requires finite local bounds, smaller spacing and a measured allowance")
    _text(request["measurement_evidence"], "fine measurement evidence")
    _text(request["source"], "fine region bounds source")
    # Python integer products cannot wrap an oversized array into a small
    # admitted count. Check before constructing native block-pool metadata.
    shape = tuple(math.ceil(float(value)/size) for value in extent)
    if min(shape) < 3 or math.prod(shape) > MAX_FINE_CELLS:
        raise ValueError("Each complete fine ROI must fit the four-million-sample bound")
    return replace(config, grid_center=torch.as_tensor(center, dtype=torch.float32),
        extent_meters_xyz=tuple(extent), extent_esdf_meters_xyz=tuple(extent),
        voxel_size=size, esdf_voxel_size=size, truncation_distance=2*size,
        max_visible_blocks_per_integration=None)


def _require_mapping_memory(estimated_bytes, device, *,
                            max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES,
                            max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES):
    """Admit only the fixed online Mapper and numerical-array working memory."""
    device=torch.device(device)
    if device.type!='cuda' or type(estimated_bytes) is not int or estimated_bytes<0:
        raise ValueError('Native snapshot mapping requires explicit CUDA and a nonnegative byte estimate')
    required=estimated_bytes+64*1024**2
    # BEGIN online fields are materialized eagerly and have no lazy reservations.
    # Previously allocated partitions are already included in measured RSS;
    # charge only this partition, without importing source-audit reservations.
    require_host_memory(required,pending_existing_bytes=0,max_rss_bytes=max_host_rss_bytes)
    # END retain the same measured host and whole-device resource limits.
    require_device_memory(torch,device,native_bytes=required,max_device_used_bytes=max_device_used_bytes)
    return required


def build_scene_from_rgbd(views, owner_frames, *, mapper_cfg, depth_error_m, task_owner_ids,
                         reference_frame, robot_bodies, robot_geometry_source,
                         contacts=(), gripper=None, has_gripper=True, unknown_policy,
                         max_mapping_bytes, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES,
                         max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES,
                         fine_regions=None, observed_owners=(), background=None, coverage=None,
                         initial_support_relations=(), holding_contacts=()):
    """Combine newly observed entities with existing fixed entity fields.

    owner_frames names only entities to integrate from the supplied masks and
    depth. observed_owners supplies existing fields and their declared motion;
    these entities require no image, mask or visible depth. Their samples and
    local support are passed unchanged to the planning scene.

    An existing background and coverage are reused in world coordinates. A new
    background uses each view's explicit background_mask when supplied, or the
    complement of newly mapped owners, after withholding the one-pixel
    eight-neighbour semantic boundary. The caller supplies a static partition
    when the observation also contains separately retained moving entities.
    The existing unmapped_pixels count includes the withheld boundary;
    independently measured free-ray coverage is unchanged.
    """
    from .observed_mapping import integrate_entity, FrozenObservedField
    from .online_geometry import OnlinePlanningScene
    from .robot_self_filter import ENVIRONMENT

    # BEGIN select geometry sources by the public input, independently of tasks.
    direct = {}
    for owner in observed_owners:
        if (not isinstance(owner, CollisionOwner) or not isinstance(owner.query, FrozenObservedField)
                or owner.owner_id in direct):
            raise ValueError("Existing entity models require unique registered identities")
        direct[owner.owner_id] = owner
    frames = {identity: (_pose_array(row["T_output_owner"], "observed owner frame"),
                        _text(row["source"], "owner source")) for identity, row in owner_frames.items()}
    if set(frames).intersection(direct):
        raise ValueError("Choose one geometry source for each entity: RGB-D or an existing model")
    all_ids = tuple((*frames, *direct))
    if not set(task_owner_ids) <= set(all_ids):
        raise ValueError("Every task entity needs an observed or existing model")
    if background is not None and not isinstance(background, FrozenObservedField):
        raise TypeError("An existing background must retain its raw observed field")
    if coverage is not None and (not isinstance(coverage, ObservedSpace)
                                 or coverage.reference_frame != reference_frame):
        raise ValueError("Existing free-ray coverage must use the planning frame")
    needs_mapping = bool(frames) or background is None
    normalized, masks, configs, resolutions = {}, {}, {}, {}
    if needs_mapping or coverage is None:
        from curobo.perception import MapperCfg
        if not isinstance(mapper_cfg, MapperCfg):
            raise TypeError("New RGB-D integration requires the installed MapperCfg")
        if (mapper_cfg.num_cameras != 1 or mapper_cfg.enable_static
                or mapper_cfg.decay_factor != 1 or mapper_cfg.frustum_decay_factor != 1):
            raise ValueError("Frozen snapshots use one nondecaying measured TSDF channel")
        if not 0 < max_mapping_bytes <= min(max_host_rss_bytes, max_device_used_bytes):
            raise ValueError("Mapping allocation must fit the request's host and device ceilings")
        normalized = _self_filtered_views(views, reference_frame,
            (mapper_cfg.depth_minimum_distance, mapper_cfg.depth_maximum_distance))
        if direct and background is None and any("background_mask" not in row for row in normalized.values()):
            raise ValueError("With retained entities, provide the static background or its original pixel partition")
        masks = _partition_masks(normalized, frames, build_background=background is None)
        resolutions = observation_resolutions(all_ids, task_owner_ids=task_owner_ids,
                                               environment_voxel_size=mapper_cfg.voxel_size)
        if background is None:
            configs[None] = mapper_cfg
        configs.update({identity: _owner_mapper_config(mapper_cfg, normalized, masks,
            identity, pose, resolutions[identity], depth_error_m)
            for identity, (pose, _) in frames.items()})
        for identity, request in ({} if fine_regions is None else fine_regions).items():
            configs[identity] = _fine_mapper_config(configs[identity], request)
    estimates = {identity: _mapper_memory_bound(config) for identity, config in configs.items()}
    if estimates and max(estimates.values()) > max_mapping_bytes:
        raise ValueError("One serial Mapper exceeds the declared mapping allocation")
    if coverage is None:
        shape_xyz = np.asarray(mapper_cfg.grid_shape[::-1], int)
        origin = mapper_cfg.grid_center.detach().cpu().numpy()-(shape_xyz-1)*mapper_cfg.voxel_size/2
        ray_depths = {name: row["self_filter"]["free_ray_depth_m"] for name, row in normalized.items()}
        known = _known_free_voxels(normalized, ray_depths=ray_depths, origin=origin,
            shape=tuple(map(int, shape_xyz[::-1])), voxel_size=mapper_cfg.voxel_size, depth_error_m=depth_error_m)
        coverage = ObservedSpace(known, origin=origin, voxel_size=mapper_cfg.voxel_size,
                                observation_ids=tuple(normalized), reference_frame=reference_frame)
    queries, reports = {}, {}
    for identity, config in configs.items():
        inverse = np.eye(4) if identity is None else _inverse(frames[identity][0])
        name = "observed_background" if identity is None else identity
        observations = ({"image": view["image"], "depth": view["depth"], "mask": masks[key][identity],
                         "K": view["K"], "T_entity_camera": inverse @ view["T_output_camera"]}
                        for key, view in normalized.items())
        query, report = integrate_entity(observations, config=config, name=name,
            source=f"provided_rgbd:{','.join(normalized)}", max_host_rss_bytes=max_host_rss_bytes,
            max_device_used_bytes=max_device_used_bytes)
        queries[identity], reports[name] = query, report
    owners = [CollisionOwner(identity, queries[identity],
        PosePath(np.array([0., 1.]), np.stack((pose, pose)), source), source)
        for identity, (pose, source) in frames.items()]
    owners.extend(direct.values())
    scene = OnlinePlanningScene(background=queries[None] if background is None else background,
        owners=owners, coverage=coverage, reference_frame=reference_frame, robot_bodies=robot_bodies,
        robot_geometry_source=robot_geometry_source, contacts=contacts, gripper=gripper,
        has_gripper=has_gripper, unknown_policy=unknown_policy,
        initial_support_relations=initial_support_relations, holding_contacts=holding_contacts)
    scene.observation_partition_masks = {name: {"background": row[None]}
                                        for name, row in masks.items() if None in row}
    endpoint_report = {}
    for name, view in normalized.items():
        environment_count = int(np.count_nonzero(view["self_filter"]["labels"] == ENVIRONMENT))
        owners_count = {identity: int(np.count_nonzero(masks[name][identity])) for identity in frames}
        background_count = int(np.count_nonzero(masks[name][None])) if None in masks[name] else 0
        endpoint_report[name] = {"environment_pixels": environment_count,
            "new_owner_pixels": owners_count, "new_background_pixels": background_count,
            "unmapped_pixels": environment_count-background_count-sum(owners_count.values())}
    known = coverage._known
    scene.observation_report = {"source": "explicit RGB-D partitions and existing entity fields", "partitions": reports,
        "geometry_model": "observed_trilinear_occupied_sets", "depth_error_m": depth_error_m,
        "reused_observed_owners": {identity: owner.query.summary() for identity, owner in direct.items()},
        "reused_background": background is not None,
        "owner_voxel_sizes_m": {**{identity: value for identity, value in resolutions.items() if identity in frames},
                                  **{identity: owner.query.voxel_size for identity, owner in direct.items()}},
        "maximum_serial_mapping_bytes": max(estimates.values(), default=0),
        "known_free_voxels": int(known.sum()), "coverage_voxels": int(known.size),
        "self_filter": _self_filter_report(normalized), "endpoint_partition": endpoint_report}
    return scene
    # END an existing entity has no observation or visibility prerequisite.

# END independent ray evidence supplies coverage.
