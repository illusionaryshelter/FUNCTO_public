"""Current calibrated RGB-D to owner-separated cuRobo planning geometry.

The input is a registered snapshot of the current scene: camera-z depth in
metres, native pinhole calibration, camera-to-planning transforms, and explicit
visible owner/robot masks. No demonstration mesh, simulator state, inferred
back surface, or bounding-box erasure enters this adapter. cuRobo's existing
Mapper builds the collision field; independent raw depth rays bound coverage.
Optional, separately verified physical solid interiors can supply current
occupied evidence. A closed collision proxy is not such physical evidence.
"""

# BEGIN current RGB-D observation bridge using the installed native mapper.
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import math
from numbers import Real

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.spatial.transform import Rotation
import torch

from .motion import _inverse, _pose_array
from .observations import _intrinsics
from .planning_problem import PosePath
from .planning_scene import (
    CollisionOwner, CuroboSphereQuery, ObservedSpace, PlanningScene, _text,
    _validate_contact_and_gripper_inputs,
)


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


def _known_free_voxels(views, *, origin, shape, voxel_size, depth_error_m):
    """Prove each complete voxel free under the supplied pixel-ray model.

    A ball of half the voxel diagonal encloses the cell. Its full projected
    footprint must lie on valid pixels, and its furthest camera-z point must
    precede the smallest measured depth in that footprint by depth_error_m.
    Projection radii are rounded upward to powers of two to bound cached
    SciPy minimum filters. No interpolated invalid depth or hidden back volume
    can become observed. Subpixel geometry below the sensor's pixel model is
    not certified; calibration/depth error belongs to the supplied evidence.
    """
    shape = tuple(shape)
    origin = np.asarray(origin, float)
    if (len(shape) != 3 or any(type(n) is not int or n < 2 for n in shape)
            or math.prod(shape) > 8_000_000 or origin.shape != (3,) or not np.isfinite(origin).all()
            or not np.isfinite(voxel_size) or voxel_size <= 0
            or not np.isfinite(depth_error_m) or depth_error_m < 0):
        raise ValueError("Coverage needs a bounded z,y,x grid and explicit metric error allowance")
    known = np.zeros(math.prod(shape), bool)
    radius = math.sqrt(3)*voxel_size/2
    for view in views.values():
        depth, matrix, transform = view["depth"], view["K"], view["T_output_camera"]
        height, width = depth.shape
        filtered = {}
        for start in range(0, len(known), 65_536):
            indices = np.arange(start, min(start+65_536, len(known)))
            indices = indices[~known[indices]]
            if not len(indices):
                continue
            z, y, x = np.unravel_index(indices, shape)
            points = origin+voxel_size*np.column_stack((x, y, z))
            camera = (points-transform[:3, 3]) @ transform[:3, :3]
            front = camera[:, 2] > radius+1e-9
            indices, camera = indices[front], camera[front]
            if not len(indices):
                continue
            uv = (camera @ matrix.T)[:, :2]/camera[:, 2, None]
            # |u(X+d)-u(X)| <= r*(f+|u-cx|)/(z-r). Taking the
            # larger image-axis radius gives a conservative square footprint.
            projected = radius*(np.diag(matrix)[:2]+np.abs(uv-matrix[:2, 2]))/(camera[:, 2, None]-radius)
            bound = np.ceil(projected.max(1)+1.)
            usable = (np.isfinite(uv).all(1) & np.isfinite(bound) & (bound <= min(height, width))
                      & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height))
            indices, camera, uv, bound = indices[usable], camera[usable], uv[usable], bound[usable]
            if not len(indices):
                continue
            pixels = np.rint(uv).astype(np.int64)
            # Nearest-pixel rounding at the last half pixel can leave the
            # image. Keep it unknown before indexing any filter result.
            inside = (pixels[:, 0] < width) & (pixels[:, 1] < height)
            indices, camera, pixels, bound = indices[inside], camera[inside], pixels[inside], bound[inside]
            bins = 2**np.ceil(np.log2(bound)).astype(np.int64)
            for extent in np.unique(bins):
                extent = int(extent)
                if extent not in filtered:
                    filtered[extent] = minimum_filter(depth, size=2*extent+1, mode="constant", cval=0.)
                selected = bins == extent
                low = filtered[extent][pixels[selected, 1], pixels[selected, 0]]
                proved = camera[selected, 2]+radius+depth_error_m < low
                known[indices[selected][proved]] = True
    return known.reshape(shape)


# BEGIN current physical-solid evidence, separate from collision outer proxies.
def _pose_evidence(record, views, reference_frame, label):
    """Require a current snapshot identity and a whole-body pose-error bound."""
    if (not isinstance(record, Mapping) or record.get("state") != "current_observed"
            or record.get("reference_frame") != reference_frame):
        raise ValueError(f"{label} needs current observed pose evidence in the planning frame")
    source = _text(record.get("source"), f"{label} current pose source")
    identities = record.get("observation_ids")
    if (not isinstance(identities, (list, tuple)) or not identities
            or len(set(identities)) != len(identities) or not set(identities) <= set(views)):
        raise ValueError(f"{label} must identify synchronized views in this current snapshot")
    error = record.get("max_point_error_m")
    if not isinstance(error, Real) or not np.isfinite(error) or error < 0:
        raise ValueError(f"{label} requires an explicit nonnegative whole-body pose error in metres")
    return {"source": source, "observation_ids": list(identities), "max_point_error_m": float(error)}


def _robot_solid_poses(current_robot, links, views, reference_frame):
    """Evaluate only the supplied measured state with a private native FK buffer.

    The already loaded cuRobo model must expose each requested link. No robot
    is loaded, no future trajectory is sampled, and no collision mesh is
    promoted to physical solid evidence. The point-error bound must include
    feedback, calibration and FK error over every supplied solid, including
    angular uncertainty at its furthest point.
    """
    evidence = _pose_evidence(current_robot, views, reference_frame, "Robot solid")
    from curobo.kinematics import Kinematics
    from curobo.types import JointState

    model = current_robot.get("kinematics")
    if not isinstance(model, Kinematics):
        raise TypeError("Robot solid poses require the already loaded native Kinematics model")
    supplied_names = current_robot.get("joint_names", ())
    raw_position = np.asarray(current_robot.get("joint_position"))
    if (not isinstance(supplied_names, (list, tuple)) or not 0 < len(supplied_names) <= 16
            or raw_position.shape != (len(supplied_names),)):
        raise ValueError("Robot solid feedback must fit the bounded sixteen-joint input before conversion")
    names, position = tuple(supplied_names), np.asarray(raw_position, float)
    if (names != tuple(model.joint_names) or not np.isfinite(position).all()
            or model.total_spheres > 512 or len(model.config.kinematics_config.link_name_to_idx_map) > 64
            or len(model.tool_frames) > 64 or not set(links) <= set(model.tool_frames)):
        raise ValueError("Robot solid FK needs the bounded actual joint layout and every requested link frame")
    limits = model.config.kinematics_config.joint_limits.position.detach().cpu().numpy()
    if np.any(position < limits[0]) or np.any(position > limits[1]):
        raise ValueError("Measured robot solid state must respect its native joint limits")
    base = _pose_array(current_robot.get("T_output_robot_base"), "T_output_robot_base")
    geometry_source = _text(current_robot.get("robot_geometry_source"), "actual robot collision-model source")
    locked = model.config.kinematics_config.lock_jointstate
    locked_positions = {}
    if locked is not None:
        if len(names)+len(locked.joint_names) > 16:
            raise ValueError("Current solid evidence must also bound the native locked-joint state")
        values = locked.position.detach().cpu().numpy().reshape(-1)
        if values.shape != (len(locked.joint_names),) or not np.isfinite(values).all():
            raise ValueError("Locked robot joints need one finite native value per named joint")
        locked_positions = {name: float(value) for name, value in zip(locked.joint_names, values)}
    # One pose per requested link, with at most 16 DOF / 64 links / 512
    # spheres, is small. Private buffers avoid modifying a caller's live FK
    # values or gradients. Retain the actual complete current sphere model so
    # the planner can reject a different model with the same names/counts.
    fk = Kinematics(model.config, compute_spheres=True, compute_com=False)
    with torch.no_grad():
        query = model.device_cfg.to_device(position)[None, None]
        state = fk.compute_kinematics(JointState.from_position(query, joint_names=list(names)))
        xyz = state.tool_poses.position[0, 0].detach().cpu().numpy().copy()
        wxyz = state.tool_poses.quaternion[0, 0].detach().cpu().numpy().copy()
        spheres = state.robot_spheres[0, 0].detach().cpu().numpy().astype(float)
    poses = {}
    for link in links:
        index = model.tool_frames.index(link)
        pose = np.eye(4)
        pose[:3, 3] = xyz[index]
        pose[:3, :3] = Rotation.from_quat(wxyz[index, [1, 2, 3, 0]]).as_matrix()
        poses[link] = base@pose
    spheres[:, :3] = spheres[:, :3]@base[:3, :3].T+base[:3, 3]
    evidence.update(joint_names=list(names), joint_position=position.tolist(),
                    locked_joint_positions=locked_positions,
                    T_output_robot_base=base.tolist(), robot_geometry_source=geometry_source,
                    robot_spheres=int(model.total_spheres), current_spheres_output=spheres.tolist(),
                    link_poses_output={name: pose.tolist() for name, pose in poses.items()})
    return poses, evidence


def _known_solid_evidence(known_solids, owner_frames, current_robot, views, reference_frame):
    """Normalize trustworthy inner occupancy without inventing physical solids.

    ``occupancy_kind`` is a caller-provided evidence claim, independently of
    mesh topology: verified_physical_solid includes actual holes/cavities;
    verified_inner_solid is wholly contained in physical occupied material.
    A watertight convex hull, collision sphere cover, AABB, or partial RGB-D
    reconstruction is not sufficient. Topological checks alone cannot prove
    this claim, so its source remains explicit in the returned evidence.
    """
    if known_solids is None:
        return (), {"solids": [], "owners": {}, "robot": None}
    if not isinstance(known_solids, Mapping) or len(known_solids) > 64:
        raise ValueError("Known physical solids must be a bounded mapping of at most 64 bodies")
    import trimesh

    records, face_count, vertex_count = [], 0, 0
    for identity, supplied in known_solids.items():
        _text(identity, "known solid identity")
        if (not isinstance(supplied, Mapping)
                or supplied.get("occupancy_kind") not in {"verified_physical_solid", "verified_inner_solid"}):
            raise ValueError("Known occupancy requires independently verified physical/inner solid evidence, not an outer collision proxy")
        source = _text(supplied.get("geometry_source"), "verified solid occupancy source")
        mesh = supplied.get("mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("Known solid evidence needs an explicit metric trimesh.Trimesh")
        face_count += len(mesh.faces)
        vertex_count += len(mesh.vertices)
        # Check counts before copies, topology caches, triangle trees or
        # proximity workspaces. Input meshes are already caller-owned memory.
        if (not 0 < len(mesh.faces) <= 50_000 or face_count > 100_000 or vertex_count > 100_000):
            raise ValueError("Known solid meshes exceed the bounded 100,000-face/vertex CPU geometry budget")
        if (not np.isfinite(mesh.vertices).all() or not mesh.is_volume):
            raise ValueError("A verified solid also needs a finite, closed, consistently oriented volume mesh")
        error = supplied.get("max_geometry_error_m")
        if not isinstance(error, Real) or not np.isfinite(error) or error < 0:
            raise ValueError("Known solids require an explicit nonnegative geometry-error bound in metres")
        kind, body = supplied.get("body_kind"), supplied.get("body_id")
        if kind not in {"owner", "robot_link"}:
            raise ValueError("Known occupancy must bind to a current owner or an actual robot link")
        _text(body, "known solid body identity")
        offset = _pose_array(supplied.get("T_body_solid"), "T_body_solid")
        # Only bounded metric geometry is required. Trimesh.copy() would also
        # duplicate caller-owned texture images that have no occupancy role.
        geometry = trimesh.Trimesh(vertices=np.asarray(mesh.vertices).copy(),
                                   faces=np.asarray(mesh.faces).copy(), process=False, validate=False)
        records.append({"solid_id": identity, "body_kind": kind, "body_id": body,
                        "occupancy_kind": supplied["occupancy_kind"], "geometry_source": source,
                        "max_geometry_error_m": float(error), "mesh": geometry, "T_body_solid": offset})
    owner_frames = {} if owner_frames is None else owner_frames
    if not isinstance(owner_frames, Mapping) or len(owner_frames) > 32:
        raise ValueError("Known owner solids require bounded current owner-frame evidence")
    owners, poses, errors = {}, {}, {}
    for body in {row["body_id"] for row in records if row["body_kind"] == "owner"}:
        if body not in owner_frames:
            raise ValueError("A known owner solid must belong to the current observed owner frames")
        frame = owner_frames[body]
        evidence = _pose_evidence(frame, views, reference_frame, f"Owner {body}")
        pose = _pose_array(frame.get("T_output_owner"), "T_output_owner")
        # Pose evidence must be tied to visible current owner pixels, not only
        # an arbitrary source string or an unseen proposed owner transform.
        visible = False
        for identity in evidence["observation_ids"]:
            view = views[identity]
            masks = view.get("owner_masks", {})
            if body not in masks:
                continue
            mask = np.asarray(masks[body])
            if mask.dtype != np.bool_ or mask.shape != view["depth"].shape:
                raise ValueError("Known owner pose evidence requires aligned boolean owner masks")
            visible |= bool(np.any(mask & (view["depth"] > 0)))
        if not visible:
            raise ValueError("Known owner solids require visible current depth for their pose evidence")
        evidence["T_output_owner"] = pose.tolist()
        owners[body], poses[("owner", body)], errors[("owner", body)] = evidence, pose, evidence["max_point_error_m"]
    robot = None
    links = {row["body_id"] for row in records if row["body_kind"] == "robot_link"}
    if links:
        link_poses, robot = _robot_solid_poses(current_robot, sorted(links), views, reference_frame)
        for link, pose in link_poses.items():
            poses[("robot_link", link)], errors[("robot_link", link)] = pose, robot["max_point_error_m"]
    summaries = []
    for row in records:
        body = (row["body_kind"], row["body_id"])
        row["T_output_solid"] = poses[body]@row["T_body_solid"]
        row["max_pose_error_m"] = errors[body]
        # Account for the longest body-to-solid lever arm when comparing
        # native link rotations. This numerical inward guard is independent
        # of the caller's physical geometry and pose uncertainty contracts.
        offset = row["T_body_solid"]
        body_extent = (np.abs(offset[:3, 3])
                       + np.abs(offset[:3, :3])@np.abs(row["mesh"].vertices).max(0))
        row["numerical_inward_margin_m"] = max(
            1e-6*(1.+float(np.linalg.norm(body_extent))),
            64*np.finfo(float).eps*max(1., float(np.abs(row["T_output_solid"]).max())))
        summaries.append({key: value for key, value in row.items() if key not in {"mesh", "T_body_solid", "T_output_solid"}})
        summaries[-1].update(T_body_solid=row["T_body_solid"].tolist(), T_output_solid=row["T_output_solid"].tolist())
    return records, {"solids": summaries, "owners": owners, "robot": robot}


def _add_known_solid_voxels(known, records, *, origin, voxel_size):
    """Union only complete cells proved inside a current physical solid.

    Positive signed distance must exceed the cell's circumsphere plus both
    evidence error bounds. Concavities and cavities stay outside. Transformed
    mesh bounds only prune proximity queries; no bounding-box cell is ever
    accepted without a native trimesh volume/distance query. Each call caps
    the worst-case point/triangle candidate product before allocating it.
    """
    if not records:
        return 0
    import trimesh

    added = 0
    free = known.copy()
    origin, shape_xyz = np.asarray(origin, float), np.asarray(known.shape[::-1])
    radius = math.sqrt(3)*voxel_size/2
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    for row in records:
        mesh, pose = row["mesh"], row["T_output_solid"]
        local_box = mesh.bounds[0]+corners*(mesh.bounds[1]-mesh.bounds[0])
        world_box = local_box@pose[:3, :3].T+pose[:3, 3]
        if not np.isfinite(world_box).all():
            raise ValueError("Known solid bounds must remain finite in the planning frame")
        # Clamp floating indices before integer conversion, including distant
        # current bodies that have no overlap with this observation grid.
        low_float = np.ceil((world_box.min(0)+radius-origin)/voxel_size)
        high_float = np.floor((world_box.max(0)-radius-origin)/voxel_size)
        if np.any(low_float > shape_xyz-1) or np.any(high_float < 0):
            continue
        low = np.maximum(low_float, 0).astype(np.int64)
        high = np.minimum(high_float, shape_xyz-1).astype(np.int64)
        if np.any(low > high):
            continue
        crop_shape = tuple((high-low+1)[::-1])
        # trimesh may consider every triangle for a pathological concave mesh.
        # Bound that worst case to 262,144 point/triangle pairs per call;
        # typical rtree pruning is cheaper. No unbounded point batch is used.
        chunk = max(1, min(1024, 262_144//len(mesh.faces)))
        # A separate one-micrometre numerical shrink (scaled by local extent)
        # exceeds the 1e-7 native model/FK comparison threshold. It only
        # removes coverage, and is not a physical/contact error allowance.
        required = (radius+row["max_geometry_error_m"]+row["max_pose_error_m"]
                    + row["numerical_inward_margin_m"])
        for start in range(0, math.prod(crop_shape), chunk):
            flat = np.arange(start, min(start+chunk, math.prod(crop_shape)))
            z, y, x = np.unravel_index(flat, crop_shape)
            x, y, z = x+low[0], y+low[1], z+low[2]
            points = origin+voxel_size*np.column_stack((x, y, z))
            local = (points-pose[:3, 3])@pose[:3, :3]
            signed = trimesh.proximity.signed_distance(mesh, local)
            contained = np.isfinite(signed) & (signed > required)
            # Both tests already include whole-cell and physical error bounds.
            # Their overlap is contradictory current evidence, not a region
            # that may quietly become known after a Boolean union. Previously
            # proved occupied cells may overlap another same-body inner solid.
            if np.any(contained & free[z, y, x]):
                raise ValueError("Current solid occupancy contradicts proved-free RGB-D cells")
            added += int(np.count_nonzero(contained & ~known[z, y, x]))
            known[z[contained], y[contained], x[contained]] = True
    return added
# END current physical-solid evidence, separate from collision outer proxies.


def observed_space_from_rgbd(views, *, origin, shape, voxel_size, depth_error_m,
                             reference_frame, depth_range_m, known_solids=None,
                             owner_frames=None, current_robot=None):
    """Build coverage from original depth, before owner/robot pixel removal.

    ``views`` maps observation IDs to depth, K, T_output_camera and
    reference_frame. ``origin`` is the centre of cell [0,0,0], ``shape`` is
    (z,y,x). Raw depth proves free ray prefixes. Optional ``known_solids`` maps
    IDs to metric trimesh meshes, occupancy_kind (verified_physical_solid or
    verified_inner_solid), geometry_source, max_geometry_error_m, body_kind
    (owner or robot_link), body_id, and T_body_solid. Closed outer collision
    proxies and partial RGB-D surfaces are not physical occupancy evidence.

    Owner frames used by solids additionally require state=current_observed,
    reference_frame, observation_ids, and max_point_error_m. Robot solids use
    current_robot with those same evidence fields plus source, kinematics (an
    already loaded native model), joint_names, joint_position,
    T_output_robot_base and robot_geometry_source. Pose error bounds cover all
    points of the supplied solids, including angular/calibration uncertainty.
    Only current poses contribute; candidate future poses cannot fill unknown.
    Owner occupancy requires complete collision geometry downstream. Unknown
    surface bands and air outside an evidenced solid remain unknown.
    """
    normalized = _views(views, reference_frame, depth_range_m)
    known = _known_free_voxels(normalized, origin=origin, shape=shape, voxel_size=voxel_size,
                               depth_error_m=depth_error_m)
    records, evidence = _known_solid_evidence(known_solids, owner_frames, current_robot, normalized, reference_frame)
    _add_known_solid_voxels(known, records, origin=origin, voxel_size=voxel_size)
    return ObservedSpace(known, origin=origin, voxel_size=voxel_size,
                         observation_ids=tuple(normalized), reference_frame=reference_frame,
                         occupancy_evidence=evidence)


def _partition_masks(views, owner_ids):
    """Preserve every visible pixel outside explicit disjoint owner masks."""
    masks, observed = {}, {identity: 0 for identity in owner_ids}
    for name, view in views.items():
        image, depth = np.asarray(view["image"]), view["depth"]
        if image.dtype != np.uint8 or image.shape != (*depth.shape, 3):
            raise ValueError("Current RGB must be an aligned uint8 H,W,3 image")
        supplied = view["owner_masks"]
        if not isinstance(supplied, Mapping) or set(supplied) != set(owner_ids):
            raise ValueError("Each view must explicitly include every selected owner mask, including empty masks")
        robot = np.asarray(view["robot_mask"])
        if robot.dtype != np.bool_ or robot.shape != depth.shape:
            raise ValueError("The robot mask must be an explicit aligned boolean image")
        occupied = robot.copy()
        masks[name] = {}
        for identity in owner_ids:
            mask = np.asarray(supplied[identity])
            if mask.dtype != np.bool_ or mask.shape != depth.shape:
                raise ValueError("Visible owner masks must be aligned boolean images")
            if np.any(mask & occupied):
                raise ValueError("Visible owner/robot pixel identities must be disjoint")
            masks[name][identity] = mask.copy()
            observed[identity] += int(np.count_nonzero(mask & (depth > 0)))
            occupied |= mask
        masks[name][None] = ~occupied
    if any(count == 0 for count in observed.values()):
        raise ValueError("Every explicit collision owner needs current valid depth evidence")
    return masks


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


# BEGIN enforce the local mapping allocation limit before native allocation.
def _require_mapping_memory(estimated_bytes, device):
    """Check live device space after CPU validation and before any Mapper.

    The per-partition estimate already includes mapper scratch and retained
    fields. Another fixed 64 MiB covers the small native collision-query
    objects and transient depth/calibration tensors built around those fields.
    Existing models, CUDA context and other applications must fit outside this
    request, so only currently free device memory may satisfy it.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Native RGB-D mapping requires an explicit CUDA device")
    required_bytes = estimated_bytes+64*1024**2
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < required_bytes:
        raise ValueError(f"Insufficient free device memory for RGB-D mapping: {required_bytes} bytes required")
    return required_bytes
# END local mapping resource guard.


def build_scene_from_rgbd(views, owner_frames, *, mapper_cfg, depth_error_m,
                          reference_frame, robot_sphere_materials, robot_geometry_source,
                          gripper=None, has_gripper=True, contacts=(),
                          known_solids=None, current_robot=None,
                          max_mapping_bytes=256*1024**2):
    """Fuse a current snapshot into a partial, explicitly partitioned scene.

    Each view additionally requires image, owner_masks and robot_mask. Empty
    boolean masks explicitly declare no visible pixels; missing masks are an
    error. ``owner_frames[id]`` supplies observed T_output_owner and its source,
    in the same fixed gauge used by grounding. ``mapper_cfg`` is the installed
    cuRobo MapperCfg with explicit TSDF and ESDF extents and one camera buffer;
    views are integrated sequentially. A rotated owner's local grid encloses
    the same world bounds. No geometry is removed by those grid bounds.

    RGB-D supplies partial observed surfaces, so all produced owners remain
    geometry_complete=False and initially static. A caller binding a payload
    must still supply its evidenced conservative sphere cover and attachment;
    this adapter never invents unseen geometry. Coverage near actual contact
    and inside a currently occluding robot may remain insufficient. Optional
    known_solids/current_robot follow observed_space_from_rgbd's independently
    verified occupied-interior contract. Their inner meshes never become an
    outer collision model or change geometry_complete. An owner supplying
    occupied evidence is unknown until its collision geometry is complete.
    """
    from curobo.perception import Mapper, MapperCfg
    from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
    from curobo._src.geom.types import SceneCfg
    from curobo.types import CameraObservation, DeviceCfg, Pose

    if not isinstance(mapper_cfg, MapperCfg):
        raise TypeError("Provide the installed cuRobo MapperCfg directly")
    if (mapper_cfg.extent_esdf_meters_xyz is None or mapper_cfg.num_cameras != 1
            or mapper_cfg.feature_dim != 0 or mapper_cfg.lidar_num_sensors != 0
            or mapper_cfg.enable_static or mapper_cfg.decay_factor != 1 or mapper_cfg.frustum_decay_factor != 1
            or mapper_cfg.edt_solver != "pba" or mapper_cfg.color_grid_size != 1):
        raise ValueError("Snapshot mapping needs explicit bounds, one camera, exact PBA, and no static/feature/decay channels")
    if type(max_mapping_bytes) is not int or not 0 < max_mapping_bytes <= 1024**3:
        raise ValueError("RGB-D mapping needs an explicit positive allocation budget no larger than one GiB")
    normalized = _views(views, reference_frame,
                        (mapper_cfg.depth_minimum_distance, mapper_cfg.depth_maximum_distance))
    if any(view["depth"].shape != (mapper_cfg.image_height, mapper_cfg.image_width) for view in normalized.values()):
        raise ValueError("Current image sizes must match the explicitly allocated native camera buffer")
    if not isinstance(owner_frames, Mapping) or len(owner_frames) > 32:
        raise ValueError("Selected owner frames must be an explicit bounded mapping")
    frames = {}
    for identity, frame in owner_frames.items():
        _text(identity, "owner identity")
        if identity == "observed_background":
            raise ValueError("observed_background is reserved for unowned visible scene geometry")
        frames[identity] = (_pose_array(frame["T_output_owner"], "T_output_owner"),
                            _text(frame["source"], "current owner-frame source"))
    # BEGIN reject CPU-checkable scene contracts before native device access.
    robot_sphere_materials = tuple(robot_sphere_materials)
    if not robot_sphere_materials:
        raise ValueError("The actual robot sphere model and material identities are required")
    _text(robot_geometry_source, "robot sphere model source")
    contacts = _validate_contact_and_gripper_inputs(
        dict.fromkeys(frames), robot_sphere_materials, contacts, gripper, has_gripper)
    torch_device = torch.device(mapper_cfg.device)
    if torch_device.type != "cuda":
        raise ValueError("Native RGB-D mapping requires an explicit CUDA device")
    # END reject CPU-checkable scene contracts before mapping/FK allocations.
    masks = _partition_masks(normalized, frames)
    center = mapper_cfg.grid_center.detach().cpu().numpy()
    configs = {None: mapper_cfg}
    world_extent = np.asarray(mapper_cfg.get_actual_extent())
    esdf_extent = np.ceil(np.asarray(mapper_cfg.extent_esdf_meters_xyz)/mapper_cfg.esdf_voxel_size)*mapper_cfg.esdf_voxel_size
    for identity, (pose, _) in frames.items():
        inverse = _inverse(pose)
        configs[identity] = replace(mapper_cfg,
            grid_center=torch.as_tensor(inverse[:3, :3]@center+inverse[:3, 3], dtype=torch.float32),
            extent_meters_xyz=tuple(np.abs(inverse[:3, :3])@world_extent),
            extent_esdf_meters_xyz=tuple(np.abs(inverse[:3, :3])@esdf_extent),
            max_visible_blocks_per_integration=None)
    estimates = {identity: _mapper_memory_bound(config) for identity, config in configs.items()}
    if sum(estimates.values()) > max_mapping_bytes:
        raise ValueError(f"Native RGB-D mapper buffers exceed the declared budget: {sum(estimates.values())} bytes")
    shape_xyz = tuple(math.ceil(x/mapper_cfg.esdf_voxel_size) for x in mapper_cfg.extent_esdf_meters_xyz)
    origin = center-(np.asarray(shape_xyz)-1)*mapper_cfg.esdf_voxel_size/2
    known = _known_free_voxels(normalized, origin=origin, shape=shape_xyz[::-1],
                               voxel_size=mapper_cfg.esdf_voxel_size, depth_error_m=depth_error_m)
    free_count = int(known.sum())
    records, evidence = _known_solid_evidence(known_solids, owner_frames, current_robot, normalized, reference_frame)
    if evidence["robot"] is not None and (
            evidence["robot"]["robot_geometry_source"] != robot_geometry_source
            or evidence["robot"]["robot_spheres"] != len(robot_sphere_materials)):
        raise ValueError("Occupied robot evidence must use this scene's complete native collision model")
    occupied_count = _add_known_solid_voxels(known, records, origin=origin, voxel_size=mapper_cfg.esdf_voxel_size)
    coverage = ObservedSpace(known, origin=origin, voxel_size=mapper_cfg.esdf_voxel_size,
                             observation_ids=tuple(normalized), reference_frame=reference_frame,
                             occupancy_evidence=evidence)
    _require_mapping_memory(sum(estimates.values()), mapper_cfg.device)
    config_device = DeviceCfg(device=mapper_cfg.device)
    queries, statistics = {}, {}
    for identity, config in configs.items():
        # A fresh mapper per partition preserves ownership at every observed
        # surface. Queries retain their native field tensors; summing every
        # partition's budget also covers delayed native scratch cleanup. No
        # clear_region or AABB subtraction is used.
        mapper = Mapper(config)
        inverse = np.eye(4) if identity is None else _inverse(frames[identity][0])
        for name, view in normalized.items():
            depth = np.where(masks[name][identity], view["depth"], 0.).astype(np.float32)
            observation = CameraObservation(
                name=name, depth_image=config_device.to_device(depth)[None],
                rgb_image=torch.as_tensor(np.asarray(view["image"]).copy(), dtype=torch.uint8,
                                          device=config_device.device)[None],
                intrinsics=config_device.to_device(view["K"])[None],
                pose=Pose.from_matrix(config_device.to_device(inverse@view["T_output_camera"])[None]),
                resolution=list(depth.shape), depth_to_meter=1.)
            mapper.integrate(camera_observation=observation)
        stats = mapper.get_stats(scan_pool=True)
        # Native stats omit the optional holes key when no blocks exist.
        # Use the invariant itself, which also covers an empty background.
        if (stats["allocation_failures"] != 0
                or stats["num_allocated"] != stats["active_blocks"]+stats["free_count"]):
            raise RuntimeError("Native observation mapping lost geometry through block-pool allocation failure")
        voxel = mapper.compute_esdf()
        partition_id = "observed_background" if identity is None else identity
        voxel.name = partition_id
        native = SceneCollision.from_config(SceneCollisionCfg(
            device_cfg=config_device, scene_model=SceneCfg(voxel=[voxel]), cache={"voxel": 1}))
        queries[identity] = CuroboSphereQuery(native, owner_ids=(partition_id,),
                                             source=f"current_rgbd:{','.join(normalized)}")
        statistics[partition_id] = {"allocated_blocks": stats["num_allocated"],
                                    "max_blocks": config.max_blocks,
                                    "tsdf_voxel_size_m": float(config.voxel_size),
                                    "esdf_voxel_size_m": float(config.esdf_voxel_size),
                                    "truncation_distance_m": float(config.truncation_distance),
                                    "estimated_buffers_bytes": estimates[identity]}
        del mapper, observation
    owners = [CollisionOwner(identity, queries[identity],
                              PosePath(np.array([0., 1.]), np.repeat(pose[None], 2, axis=0), source),
                              source, geometry_complete=False)
              for identity, (pose, source) in frames.items()]
    result = PlanningScene(background=queries[None], owners=owners, coverage=coverage,
                           reference_frame=reference_frame, robot_sphere_materials=robot_sphere_materials,
                           robot_geometry_source=robot_geometry_source, contacts=contacts,
                           gripper=gripper, has_gripper=has_gripper)
    result.observation_report = {"source": "current calibrated RGB-D and explicit current physical-solid evidence", "partitions": statistics,
                                 "known_free_voxels": free_count, "additional_known_occupied_voxels": occupied_count,
                                 "known_voxels": int(known.sum()), "coverage_voxels": int(known.size),
                                 "occupancy_evidence": deepcopy(evidence),
                                 "depth_error_m": float(depth_error_m),
                                 "depth_range_m": [float(mapper_cfg.depth_minimum_distance),
                                                   float(mapper_cfg.depth_maximum_distance)],
                                 "geometry_precision": "native discretized TSDF/ESDF proxy; resolution is not a contact-accuracy certificate",
                                 "coverage": "whole-cell free ray prefixes or verified current solid interiors; all other volume remains unknown"}
    return result
# END current RGB-D observation bridge using the installed native mapper.
