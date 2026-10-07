"""Current calibrated RGB-D to owner-separated cuRobo planning geometry.

The input is a registered snapshot of the current scene: camera-z depth in
metres, native pinhole calibration, camera-to-planning transforms, and explicit
visible owner masks and explicit three-state robot self-filter evidence.
No demonstration mesh, simulator state, inferred
back surface, or bounding-box erasure enters this adapter. cuRobo's existing
Mapper builds the collision field; independently admitted environment rays
bound online coverage. Explicitly named audit APIs retain historical raw rays.
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
# BEGIN mapping and planning share explicit request-wide resource ceilings.
from .resource_admission import (
    DEFAULT_MAX_DEVICE_USED_BYTES, DEFAULT_MAX_HOST_RSS_BYTES,
    MAX_DEVICE_USED_BYTES, MAX_HOST_RSS_BYTES,
    ResourceLimitError, require_device_memory, require_host_memory,
)
# END native Mapper allocations never borrow PyTorch's cached free blocks.
from .planning_scene import (
    CollisionOwner, CuroboSphereQuery, ObservedSpace, PlanningScene, _text,
    _validate_contact_and_gripper_inputs,
)
# BEGIN body-wide replacement by a separately observed fine field.
from .voxel_layers import MAX_FINE_CELLS, LayeredVoxelQuery, validate_body_groups
# END both native and independent consumers retain the same layer inventory.


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


# BEGIN callers explicitly select independent free-ray evidence.
def _known_free_voxels(views, *, ray_depths, origin, shape, voxel_size, depth_error_m):
    """Prove each complete voxel free under the supplied pixel-ray model.

    A ball of half the voxel diagonal encloses the cell. Its full projected
    footprint must lie on valid admitted rays, and its furthest camera-z point
    must precede the smallest ray depth in that footprint by depth_error_m.
    ``ray_depths`` is mandatory: online callers pass self-filtered free-ray
    evidence; historical audit callers explicitly select original depth.
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
    if not isinstance(ray_depths, Mapping) or set(ray_depths) != set(views):
        raise ValueError("Free-ray depths must explicitly identify every calibrated view")
    for name, view in views.items():
        rays = np.asarray(ray_depths[name])
        if (rays.shape != view["depth"].shape or not np.isfinite(rays).all()
                or np.any(rays < 0) or np.any(rays > view["depth"])):
            raise ValueError("Free-ray depths must be aligned finite nonnegative measured prefixes")
    known = np.zeros(math.prod(shape), bool)
    radius = math.sqrt(3)*voxel_size/2
    for name, view in views.items():
        depth, matrix, transform = np.asarray(ray_depths[name]), view["K"], view["T_output_camera"]
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
# END masked endpoints never implicitly become free-ray evidence.


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


# BEGIN explicitly isolated raw-depth coverage for historical source audits.
def observed_audit_space_from_rgbd(views, *, origin, shape, voxel_size, depth_error_m,
                             reference_frame, depth_range_m, known_solids=None,
                             owner_frames=None, current_robot=None):
    """Historical source-audit coverage from original, unfiltered depth.

    This function is not online collision coverage. The online scene builder
    requires three-state self-filter evidence and excludes entire uncertain
    rays; this audit intentionally replays the original raw-depth contract.

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
    known = _known_free_voxels(normalized, ray_depths={name: row["depth"] for name, row in normalized.items()}, origin=origin, shape=shape, voxel_size=voxel_size,
                               depth_error_m=depth_error_m)
    # BEGIN retain original ray evidence before adding current solid interiors.
    # K may grow by O below; the robot volume oracle must still consume F.
    known_free = known.copy()
    records, evidence = _known_solid_evidence(known_solids, owner_frames, current_robot, normalized, reference_frame)
    _add_known_solid_voxels(known, records, origin=origin, voxel_size=voxel_size)
    return ObservedSpace(known, origin=origin, voxel_size=voxel_size,
                         observation_ids=tuple(normalized), reference_frame=reference_frame,
                         occupancy_evidence=evidence, known_free=known_free)
    # END occupied cells never become observed-free cells.
# END online coverage instead consumes three-state free-ray evidence.


# BEGIN explicitly historical binary-mask audit; production uses three states.
def _audit_partition_masks(views, owner_ids):
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
# END this audit API cannot be selected by a missing production field.


# BEGIN production three-state evidence contract and semantic ownership.
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


def _partition_masks(views, owner_ids):
    """Keep only environmental endpoints, preserving semantic conflict checks.

    Check owner/owner overlap on the supplied masks before self filtering;
    uncertainty is not permission to silently choose a semantic winner.
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
        masks[name][None] = environment & ~occupied
    if any(count == 0 for count in observed.values()):
        raise ValueError("Every explicit collision owner needs current valid depth evidence after self filtering")
    return masks


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
# END occupied endpoint selection and free-ray evidence remain independent.


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


# BEGIN enforce request totals for each new native mapping allocation.
def _require_audit_mapping_memory(estimated_bytes, device, *,
                            max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES,
                            max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES):
    """Check live host/device space before one new Mapper allocation.

    The per-partition estimate already includes mapper scratch and retained
    fields. Another fixed 64 MiB covers the small native collision-query
    objects and transient depth/calibration tensors built around those fields.
    Earlier partitions are already represented by live RSS/device usage and
    must not be charged again. This admission reserves the native-buffer bound
    on the host as conservative working room for the CPU TSDF export; it is
    not a measurement of that export's Python-object peak. The external process
    supervisor must continue enforcing the same total limits during export.
    CUDA context, other applications and retained fields count toward total
    device use. Native Mapper/Warp allocations cannot use PyTorch's cache.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Native RGB-D mapping requires an explicit CUDA device")
    if type(estimated_bytes) is not int or estimated_bytes < 0:
        raise ValueError("Mapping estimate must be a nonnegative byte count")
    required_bytes = estimated_bytes+64*1024**2
    # BEGIN preserve unmaterialized Coal reservations during map refinement.
    # Lazy independent geometry may already promise host arrays that have not
    # entered RSS. Resident geometry is excluded by this shared inventory.
    from .tsdf_geometry import ObservedTsdfGeometry
    require_host_memory(required_bytes,
                        pending_existing_bytes=ObservedTsdfGeometry.pending_storage_total_nbytes(),
                        max_rss_bytes=max_host_rss_bytes)
    # END retained fields in RSS are still counted only once.
    require_device_memory(torch, device, native_bytes=required_bytes,
                          max_device_used_bytes=max_device_used_bytes)
    return required_bytes
# END each native allocation checks the same request-wide host/device totals.


# BEGIN reject native index layouts that disagree with the complete mapped field.
def _require_native_voxel_layout(voxel, native):
    """Admit the same XYZ field for native queries and independent geometry.

    VoxelData stores float32 ``dims / voxel_size`` parameters. The two native
    SDF entry points now recover these integral sizes with wp.round, fixing
    cuRobo #699. Compare that actual kernel layout with the completed mapper
    tensor before publishing either consumer. This is an admission check;
    it never rewrites metadata or repairs an inconsistent field.
    """
    storage = native.data.voxels
    layer = storage.get_idx(voxel.name)
    values = storage.params[0, layer, :3].detach().cpu()
    field_shape = tuple(int(value) for value in voxel.feature_tensor.shape)
    # BEGIN match the corrected Warp kernel's positive roundf conversion.
    if (len(field_shape) != 3 or not bool(torch.isfinite(values).all())
            or bool((values <= 0).any())):
        raise ValueError("Completed native ESDF needs a dense XYZ field and finite positive grid params")
    # Warp uses C roundf (half away from zero); Python/torch round instead use
    # ties to even. These are positive float32 grid counts, so floor(x + 0.5)
    # on each promoted scalar reproduces the native conversion, including ties.
    native_shape = tuple(math.floor(float(value)+.5) for value in values.tolist())
    # END no legacy truncation or metadata-writing fallback remains.
    declared_shape = tuple(int(round(float(value)/float(voxel.voxel_size))) for value in voxel.dims)
    if native_shape != field_shape or declared_shape != field_shape:
        raise ValueError(
            f"Unsupported native ESDF grid layout for {voxel.name}: "
            f"field_shape_xyz={field_shape}, native_rounded_shape_xyz={native_shape}, "
            f"native_params_xyz={values.tolist()}, declared_shape_xyz={declared_shape}, "
            f"dims_m={list(voxel.dims)}, voxel_size_m={float(voxel.voxel_size)}")
# END no rounding repair, reshape or alternate field enters either consumer.


# BEGIN independently bounded owner resolution on the same current RGB-D input.
def _owner_mapper_config(config, views, masks, owner_id, pose, voxel_size, depth_error_m, query_radius_m=0.):
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
    # BEGIN the frozen online map includes every registered query centre that
    # can reach the observed owner. This is pre-admission model support, not
    # a region enlarged after a failed optimizer candidate.
    padding = 4*voxel_size+depth_error_m+float(query_radius_m)
    # END audit callers retain their explicitly zero extra sphere support.
    extent = tuple(upper-lower+2*padding)
    return replace(config, grid_center=torch.as_tensor((lower+upper)/2, dtype=torch.float32),
        extent_meters_xyz=extent, extent_esdf_meters_xyz=extent,
        voxel_size=voxel_size, esdf_voxel_size=voxel_size,
        truncation_distance=2*voxel_size, max_visible_blocks_per_integration=None)
# END refined owner fields remain partial collision geometry.


# BEGIN explicit small-region allocation before any native mapping work.
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
# END no allocation is reduced after seeing collision values.


# BEGIN one immutable mapped partition, shared by initial build and refinement.
def _map_partition(normalized, masks, identity, config, inverse, depth_error_m, request=None):
    """Reconstruct one field from original partitioned observations, sequentially."""
    from curobo.perception import Mapper
    from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
    from curobo._src.geom.types import SceneCfg
    from curobo.types import CameraObservation, DeviceCfg, Pose
    config_device = DeviceCfg(device=config.device)
    region = None
    # A fresh mapper per partition preserves ownership at every observed
    # surface. Queries retain their native field tensors; summing every
    # partition's budget also covers delayed native scratch cleanup. No
    # clear_region or AABB subtraction is used.
    mapper = Mapper(config)
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
    # BEGIN validate native indexing before a query or solid can be published.
    _require_native_voxel_layout(voxel, native)
    # END a mismatched complete field remains unsupported, not executable.
    query = CuroboSphereQuery(native, owner_ids=(partition_id,),
                                         source=f"current_rgbd:{','.join(normalized)}")
    # BEGIN preserve observed TSDF geometry independently of ESDF optimizer costs.
    # The Mapper is still alive here: export its weighted distance samples,
    # not the ESDF seeds expanded into occupied cubes. The adapter uses one
    # affine tetrahedral field for both its zero set and observed negative
    # region, with physical sample-centre coordinates and no unknown fill.
    from .tsdf_geometry import build_tsdf_geometry
    query.independent_geometry = build_tsdf_geometry(
        mapper, name=partition_id, source=query.source)
    # END native sphere costs retain their original differentiable ESDF.
    # BEGIN export the actual fine sign source before releasing its Mapper.
    if request is not None:
        from .native_voxel_sources import export_source_bits
        shape = tuple(voxel.feature_tensor.shape)
        bits = export_source_bits(mapper, config, voxel)
        # Raw depth retains robot/other-owner occlusion. Only its observed
        # free prefix can support a background sign; no masked hole is free.
        local_views = {name: {**view, "T_output_camera": inverse@view["T_output_camera"]}
                       for name, view in normalized.items()}
        fine_origin = np.asarray(voxel.pose[:3])-(np.asarray(shape)-1)*voxel.voxel_size/2
        free = _known_free_voxels(local_views, ray_depths={name: row["depth"] for name, row in local_views.items()}, origin=fine_origin, shape=shape[::-1],
            voxel_size=voxel.voxel_size, depth_error_m=depth_error_m).transpose(2, 1, 0)
        # Truncation, a full grid diagonal and one interpolation step
        # are numerical map support, never the sensor-error margin.
        support = math.ceil((config.truncation_distance+math.sqrt(3)*config.voxel_size
                              +config.esdf_voxel_size)/config.esdf_voxel_size)*config.esdf_voxel_size
        region = {"pose": list(voxel.pose), "shape_xyz": list(shape),
            "voxel_size_m": float(voxel.voxel_size), "numerical_support_m": support,
            "measurement_allowance_m": float(request["measurement_allowance_m"]),
            "measurement_evidence": request["measurement_evidence"], "source": request["source"],
            "observation_ids": list(normalized), "source_bits_hex": bits.tobytes().hex(),
            "raw_free_packbits_hex": np.packbits(free).tobytes().hex(),
            "distance_scope": "local source-supported distance; no global nearest-surface certificate"}
    # END positive ESDF values never become observation evidence.
    statistics = {"allocated_blocks": stats["num_allocated"],
                                "max_blocks": config.max_blocks,
                                "tsdf_voxel_size_m": float(config.voxel_size),
                                "esdf_voxel_size_m": float(config.esdf_voxel_size),
                                "truncation_distance_m": float(config.truncation_distance),
                                # BEGIN record the finite field actually queried.
                                "grid_center_local_m": config.grid_center.cpu().tolist(),
                                "extent_esdf_meters_xyz": list(config.extent_esdf_meters_xyz),
                                # BEGIN disclose counts even when a finite inventory refuses admission.
                                "independent_field": query.independent_geometry["field_summary"],
                                # END a resource refusal cannot masquerade as an empty observed target.
                                # END voxel resolution is not a sensor-error certificate.
                                "estimated_buffers_bytes": _mapper_memory_bound(config)}
    del mapper, observation
    return query, statistics, region
# END no different mapping formula exists for a refined field.


# BEGIN request-owned original observations for finite background remapping.
class _FrozenBackgroundRefinement:
    """Retain read-only sensor inputs, never a previous solver or fine field."""

    # BEGIN refinements inherit the original request's explicit capacity limits.
    def __init__(self, normalized, masks, config, request, depth_error_m, *,
                 max_mapping_bytes, max_host_rss_bytes, max_device_used_bytes):
        self.views, self.masks = {}, {}
        for name, view in normalized.items():
            self.views[name] = {}
            for key in ("image", "depth", "K", "T_output_camera"):
                value = np.asarray(view[key]).copy()
                value.setflags(write=False)
                self.views[name][key] = value
            mask = masks[name][None].copy()
            mask.setflags(write=False)
            self.masks[name] = {None: mask}
        self.config = deepcopy(config)
        self.request = deepcopy(request)
        self.depth_error_m = float(depth_error_m)
        self.max_mapping_bytes = max_mapping_bytes
        self.max_host_rss_bytes = max_host_rss_bytes
        self.max_device_used_bytes = max_device_used_bytes
    # END old retained fields are charged only through measured live totals.

    @property
    def storage_bytes(self):
        """Count the one shared immutable CPU observation snapshot."""
        return sum(value.nbytes for view in self.views.values() for value in view.values())+sum(
            value[None].nbytes for value in self.masks.values())

    def refine(self, current, witnesses):
        """Build one enlarged local map without changing frozen evidence.

        The old ROI is retained to prevent repeated movement/shrinkage. All
        positive complete-body witnesses enter the requested union; capacity
        refusal is explicit rather than dropping an inconvenient body.
        """
        report = {"status": "unavailable", "active_witnesses": len(witnesses),
                  "max_fine_cells": MAX_FINE_CELLS,
                  "max_mapping_bytes": self.max_mapping_bytes,
                  "max_host_rss_bytes": self.max_host_rss_bytes,
                  "max_device_used_bytes": self.max_device_used_bytes}
        if not witnesses:
            report["status"] = "no_active_witnesses"
            return None, report
        if len(witnesses) > 32768:
            raise ValueError("Background refinement witnesses exceed the finite sample inventory")
        envelopes = np.asarray([row["envelope_world_m"] for row in witnesses], dtype=float)
        if (envelopes.shape != (len(witnesses), 2, 3) or not np.isfinite(envelopes).all()
                or np.any(envelopes[:, 0] > envelopes[:, 1])):
            raise ValueError("Refinement requires finite ordered complete-body envelopes")
        report.update(active_positive_spheres=sum(int(row["positive_spheres"]) for row in witnesses),
                      active_bodies=sorted({row["body_id"] for row in witnesses}),
                      coarse_cost=sum(float(row["coarse_cost"]) for row in witnesses))
        # BEGIN distinguish native geometry from the Mapper's Python size input.
        # Mapper rounds its spacing and center to float32, but computes shape
        # as ceil(Python extent/Python spacing). Re-ceiling N*native_spacing
        # with the original Python spacing can spuriously turn N into N+1.
        size = float(self.request["voxel_size_m"])
        native_size = current.region.size
        padding = current.region.support+6*native_size
        previous_shape = tuple(current.region.shape)
        previous_half = np.asarray(previous_shape)*native_size/2
        previous_lower = current.region.pose[:3]-previous_half
        previous_upper = current.region.pose[:3]+previous_half
        lower = np.minimum(envelopes[:, 0].min(0)-padding, previous_lower)
        upper = np.maximum(envelopes[:, 1].max(0)+padding, previous_upper)
        expanded = (lower < previous_lower) | (upper > previous_upper)
        same_bounds = not expanded.any()
        center = current.region.pose[:3].copy()
        shape = list(previous_shape)
        if not same_bounds:
            # Decide coverage using the actual center the native Mapper will
            # store. Unchanged axes keep their exact old center and integer
            # shape; changed axes cover both sides after center quantization.
            center[expanded] = ((lower[expanded]+upper[expanded])/2).astype(np.float32)
            for axis in np.flatnonzero(expanded):
                span = 2*max(center[axis]-lower[axis], upper[axis]-center[axis])
                count = max(previous_shape[axis], math.ceil(span/native_size))
                # Division can round down at an integer. Check the published
                # physical bounds directly and add one cell only if needed.
                half = count*native_size/2
                if center[axis]-half > lower[axis] or center[axis]+half < upper[axis]:
                    count += 1
                shape[axis] = count
        # The integer shape owns admission. Encoding the Mapper's extent one
        # binary64 step inward avoids a second ceil caused by multiplication
        # roundoff; published geometry is still exactly N*native_spacing.
        cells = math.prod(shape)
        report["requested_cells"] = cells
        if cells > MAX_FINE_CELLS:
            report["status"] = "local_cell_budget_refused"
            return None, report
        extent = np.nextafter(np.asarray(shape, dtype=float)*size, 0.)
        actual_extent = np.asarray(shape)*native_size
        if (tuple(math.ceil(float(value)/size) for value in extent) != tuple(shape)
                or np.any(center-actual_extent/2 > lower)
                or np.any(center+actual_extent/2 < upper)):
            raise RuntimeError("Refined native grid must retain its exact shape and contain every requested bound")
        # END neither approximate equality nor nominal spacing can hide growth.
        request = {**self.request, "center_m": center.tolist(), "extent_m": extent.tolist(),
                   "source": "planner active complete-body background collision witnesses; same frozen RGB-D"}
        report.update(center_m=center.tolist(), extent_m=extent.tolist(), voxel_size_m=size,
                      shape_xyz=shape, actual_voxel_size_m=native_size,
                      actual_extent_m=actual_extent.tolist())
        if same_bounds:
            # Existing observation fields remain immutable. A new witness can
            # require another supported overlap box without needing any new
            # Mapper allocation or changing a previously captured selector.
            fine, region = current.fine.clone_for_rollout(), current.region.record
            mapping = deepcopy(current.mapping_report)
            report.update(mapping_reused=True, estimated_mapping_bytes=0)
        else:
            # BEGIN only a real physical expansion needs a Mapper admission.
            config = _fine_mapper_config(self.config, request)
            estimate = _mapper_memory_bound(config)
            report["estimated_mapping_bytes"] = estimate
            if estimate > self.max_mapping_bytes:
                report["status"] = "mapping_byte_budget_refused"
                return None, report
            # END unchanged fields never allocate or reserve remapping buffers.
            try:
                _require_audit_mapping_memory(estimate, config.device,
                    max_host_rss_bytes=self.max_host_rss_bytes,
                    max_device_used_bytes=self.max_device_used_bytes)
            except ResourceLimitError as error:
                # This admission check allocates nothing. Its explicit live
                # resource refusal retains the still-valid original scene;
                # Mapper/runtime exceptions below remain actual failures.
                report.update(status="live_mapping_memory_refused", reason=str(error))
                return None, report
            fine, mapping, region = _map_partition(self.views, self.masks, None, config,
                                                   np.eye(4), self.depth_error_m, request)
            report["mapping_reused"] = False
        replacement = LayeredVoxelQuery(current.coarse.clone_for_rollout(), fine, region)
        # Re-prove old boxes in the new field before adding new witnesses. A
        # body already at weight one is absent from the unresolved witness
        # list; omitting these boxes would withdraw its existing fine guide.
        previous_boxes = current.region.guidance_boxes+current.region.pose[:3]
        # High-cost new witnesses retain deterministic tie order. The source
        # predicate is checked again for every old and new envelope.
        ordered = np.argsort([-float(row["coarse_cost"]) for row in witnesses], kind="stable")
        support_report = replacement.region.prepare_guidance_boxes(
            np.concatenate((previous_boxes, envelopes[ordered])))
        support_report["previous_cuboid_envelopes"] = len(previous_boxes)
        report.update(mapping=mapping, support=support_report)
        if not support_report["cuboids"]:
            report["status"] = "observation_support_unavailable"
            return None, report
        if same_bounds and np.array_equal(replacement.region.guidance_boxes, current.region.guidance_boxes):
            report["status"] = ("cuboid_budget_refused" if support_report["cuboid_capacity_reached"]
                and support_report["supported_uncovered_envelopes"] else "unchanged_source_supported_cuboids")
            return None, report
        replacement.refinement_source = self
        replacement.mapping_report = deepcopy(mapping)
        report["status"] = "refined"
        return replacement, report
# END only the newly mapped fine layer is allocated between solver lifetimes.


def build_audit_scene_from_rgbd(views, owner_frames, *, mapper_cfg, depth_error_m,
                          # BEGIN every owner explicitly selects its own metric grid.
                          owner_voxel_sizes_m,
                          reference_frame, robot_sphere_materials, robot_geometry_source,
                          gripper=None, has_gripper=True, contacts=(),
                          known_solids=None, current_robot=None,
                          max_mapping_bytes=256*1024**2,
                          max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES,
                          max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES,
                          native_geometry_materials=None,
                          # END no name heuristic or shared owner-spacing fallback.
                          # BEGIN explicit pose-arrival robot/owner pair permissions.
                          robot_owner_permissions=(),
                          fine_regions=None, robot_collision_bodies=()):
                          # END mapping preserves the caller's bound policy.
                          # END no owner sample is omitted by the refinement.
    """Fuse a current snapshot into a partial, explicitly partitioned scene.

    Each view additionally requires image, owner_masks and robot_mask. Empty
    boolean masks explicitly declare no visible pixels; missing masks are an
    error. ``owner_frames[id]`` supplies observed T_output_owner and its source,
    in the same fixed gauge used by grounding. ``mapper_cfg`` is the installed
    cuRobo MapperCfg with explicit TSDF and ESDF extents and one camera buffer;
    views are integrated sequentially. ``owner_voxel_sizes_m`` must map exactly
    every owner ID to positive finite metres; an empty owner set requires {}.
    Each rotated owner-local grid encloses every valid labelled depth sample.
    A larger spacing changes the discretized TSDF/ESDF surfaces and truncation
    band; it can lose thin surface detail, so it is not a storage-only change
    or a contact-accuracy certificate. Raw pixels, owner poses, unknown space
    and the background/free-space grid are unchanged by these choices.

    RGB-D supplies partial observed surfaces, so all produced owners remain
    geometry_complete=False and initially static. A caller binding a payload
    must still supply its evidenced conservative sphere cover and attachment;
    this adapter never invents unseen geometry. Coverage near actual contact
    and inside a currently occluding robot may remain insufficient. Optional
    known_solids/current_robot follow observed_audit_space_from_rgbd's independently
    verified occupied-interior contract. Their inner meshes never become an
    outer collision model or change geometry_complete. An owner supplying
    occupied evidence is unknown until its collision geometry is complete.
    """
    from curobo.perception import MapperCfg

    if not isinstance(mapper_cfg, MapperCfg):
        raise TypeError("Provide the installed cuRobo MapperCfg directly")
    if (mapper_cfg.extent_esdf_meters_xyz is None or mapper_cfg.num_cameras != 1
            or mapper_cfg.feature_dim != 0 or mapper_cfg.lidar_num_sensors != 0
            or mapper_cfg.enable_static or mapper_cfg.decay_factor != 1 or mapper_cfg.frustum_decay_factor != 1
            or mapper_cfg.edt_solver != "pba" or mapper_cfg.color_grid_size != 1):
        raise ValueError("Snapshot mapping needs explicit bounds, one camera, exact PBA, and no static/feature/decay channels")
    # BEGIN reject invalid total/module budgets before any native device access.
    if type(max_host_rss_bytes) is not int or not 0 < max_host_rss_bytes <= MAX_HOST_RSS_BYTES:
        raise ValueError("RGB-D mapping requires a positive host limit no larger than twelve GiB")
    if type(max_device_used_bytes) is not int or not 0 < max_device_used_bytes <= MAX_DEVICE_USED_BYTES:
        raise ValueError("RGB-D mapping requires a positive device limit no larger than eight GiB")
    if (type(max_mapping_bytes) is not int or max_mapping_bytes <= 0
            or max_mapping_bytes > min(max_host_rss_bytes, max_device_used_bytes)):
        raise ValueError("RGB-D mapping needs a positive allocation budget within both declared total limits")
    # END no independent one-GiB ceiling constrains an explicitly admitted map.
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
    # BEGIN exact owner keys prevent silent precision defaults or unused entries.
    if not isinstance(owner_voxel_sizes_m, Mapping) or set(owner_voxel_sizes_m) != set(frames):
        raise ValueError("Owner voxel sizes must map exactly every selected owner ID")
    sizes = {}
    for identity, value in owner_voxel_sizes_m.items():
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError("Each owner voxel size must be explicit finite positive metres")
        sizes[identity] = float(value)
    # END each owner uses its caller-declared resolution in its own local frame.
    # BEGIN reject CPU-checkable scene contracts before native device access.
    robot_sphere_materials = tuple(robot_sphere_materials)
    if not robot_sphere_materials:
        raise ValueError("The actual robot sphere model and material identities are required")
    _text(robot_geometry_source, "robot sphere model source")
    # BEGIN multi-resolution selection needs every physical body's full cover.
    fine_regions = {} if fine_regions is None else dict(fine_regions)
    if not set(fine_regions) <= {None, *frames}:
        raise ValueError("Fine regions must name existing owner partitions, or None for background")
    if fine_regions:
        robot_collision_bodies = validate_body_groups(robot_collision_bodies, len(robot_sphere_materials))
    # END material identities cannot substitute for body ownership.
    contacts = _validate_contact_and_gripper_inputs(
        dict.fromkeys(frames), robot_sphere_materials, contacts, gripper, has_gripper)
    torch_device = torch.device(mapper_cfg.device)
    if torch_device.type != "cuda":
        raise ValueError("Native RGB-D mapping requires an explicit CUDA device")
    # END reject CPU-checkable scene contracts before mapping/FK allocations.
    masks = _audit_partition_masks(normalized, frames)
    center = mapper_cfg.grid_center.detach().cpu().numpy()
    configs = {None: mapper_cfg}
    # BEGIN all selected owner fields use every labelled point at their own spacing.
    for identity, (pose, _) in frames.items():
        configs[identity] = _owner_mapper_config(mapper_cfg, normalized, masks, identity,
            pose, sizes[identity], depth_error_m)
    # END raw depth, masks, background bounds and owner transforms are preserved.
    # BEGIN each layer gets its own Mapper, with all allocations admitted together.
    jobs = {(identity, "coarse"): config for identity, config in configs.items()}
    jobs.update({(identity, "fine"): _fine_mapper_config(configs[identity], request)
                 for identity, request in fine_regions.items()})
    estimates = {key: _mapper_memory_bound(config) for key, config in jobs.items()}
    # END a fine field is reconstructed from the same pixels, never upsampled.
    if sum(estimates.values()) > max_mapping_bytes:
        raise ValueError(f"Native RGB-D mapper buffers exceed the declared budget: {sum(estimates.values())} bytes")
    # BEGIN preflight the CPU TSDF record contract before allocating a Mapper.
    # Independent collision evidence exports a dense float32 TSDF snapshot.
    # Its existing record-array bound is separate from map workspace capacity;
    # finding this unsupported shape after GPU mapping would waste the work.
    from .tsdf_geometry import _require_sample_array_budget
    for config in jobs.values():
        _require_sample_array_budget(tuple(config.grid_shape[::-1]))
    # END a larger module quota cannot silently widen the geometry format.
    shape_xyz = tuple(math.ceil(x/mapper_cfg.esdf_voxel_size) for x in mapper_cfg.extent_esdf_meters_xyz)
    origin = center-(np.asarray(shape_xyz)-1)*mapper_cfg.esdf_voxel_size/2
    known = _known_free_voxels(normalized, ray_depths={name: row["depth"] for name, row in normalized.items()}, origin=origin, shape=shape_xyz[::-1],
                               voxel_size=mapper_cfg.esdf_voxel_size, depth_error_m=depth_error_m)
    # BEGIN preserve the same raw F used by the CPU-only RGB-D builder.
    known_free = known.copy()
    free_count = int(known.sum())
    records, evidence = _known_solid_evidence(known_solids, owner_frames, current_robot, normalized, reference_frame)
    if evidence["robot"] is not None and (
            evidence["robot"]["robot_geometry_source"] != robot_geometry_source
            or evidence["robot"]["robot_spheres"] != len(robot_sphere_materials)):
        raise ValueError("Occupied robot evidence must use this scene's complete native collision model")
    occupied_count = _add_known_solid_voxels(known, records, origin=origin, voxel_size=mapper_cfg.esdf_voxel_size)
    coverage = ObservedSpace(known, origin=origin, voxel_size=mapper_cfg.esdf_voxel_size,
                             observation_ids=tuple(normalized), reference_frame=reference_frame,
                             occupancy_evidence=evidence, known_free=known_free)
    # END native mapping and occupied-solid evidence cannot widen F.
    queries, statistics, regions = {}, {}, {}
    for (identity, layer), config in jobs.items():
        # BEGIN admit only this new partition against the measured live total.
        # Previous partitions' retained fields and allocator caches are already
        # in live memory. Adding the whole batch again would charge them twice.
        _require_audit_mapping_memory(estimates[identity, layer], config.device,
            max_host_rss_bytes=max_host_rss_bytes,
            max_device_used_bytes=max_device_used_bytes)
        # END the module's aggregate CPU estimate remains an independent bound.
        inverse = np.eye(4) if identity is None else _inverse(frames[identity][0])
        query, report, region = _map_partition(normalized, masks, identity, config, inverse,
            depth_error_m, fine_regions[identity] if layer == "fine" else None)
        queries[identity, layer] = query
        statistics.setdefault("observed_background" if identity is None else identity, {})[layer] = report
        if region is not None:
            regions[identity] = region
    # BEGIN publish one selector-backed query per physical owner partition.
    queries = {identity: (LayeredVoxelQuery(queries[identity, "coarse"], queries[identity, "fine"], regions[identity])
                         if identity in regions else queries[identity, "coarse"])
               for identity in configs}
    # END coarse and fine costs are never fused as separate obstacles.
    # BEGIN snapshot remapping follows the background through scene copies.
    if None in fine_regions:
        queries[None].refinement_source = _FrozenBackgroundRefinement(
            normalized, masks, configs[None], fine_regions[None], depth_error_m,
            max_mapping_bytes=max_mapping_bytes, max_host_rss_bytes=max_host_rss_bytes,
            max_device_used_bytes=max_device_used_bytes)
        queries[None].mapping_report = deepcopy(statistics["observed_background"]["fine"])
    # END owner substitutions and installation additions preserve this query.
    owners = [CollisionOwner(identity, queries[identity],
                              PosePath(np.array([0., 1.]), np.repeat(pose[None], 2, axis=0), source),
                              source, geometry_complete=False)
              for identity, (pose, source) in frames.items()]
    result = PlanningScene(background=queries[None], owners=owners, coverage=coverage,
                           reference_frame=reference_frame, robot_sphere_materials=robot_sphere_materials,
                           robot_geometry_source=robot_geometry_source, contacts=contacts,
                           # BEGIN preserve the independently calibrated native material identities.
                           gripper=gripper, has_gripper=has_gripper,
                           native_geometry_materials=native_geometry_materials,
                           # BEGIN retain identical pair permissions through observation binding.
                           robot_owner_permissions=robot_owner_permissions,
                           robot_collision_bodies=robot_collision_bodies)
                           # END RGB-D mapping does not activate actual contact permission.
                           # END the mapping does not itself create a contact permission.
    result.observation_report = {"source": "current calibrated RGB-D and explicit current physical-solid evidence", "partitions": statistics,
                                 # BEGIN distinguish requested capacity from observed process peaks.
                                 "owner_voxel_sizes_m": sizes,
                                 "estimated_mapping_buffers_bytes": sum(estimates.values()),
                                 "max_mapping_bytes": max_mapping_bytes,
                                 "max_host_rss_bytes": max_host_rss_bytes,
                                 "max_device_used_bytes": max_device_used_bytes,
                                 # END each scene and its refinements share these limits.
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

# BEGIN production mapping creates only the fixed online numerical model.
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



def _map_online_partition(normalized, masks, identity, config, inverse):
    """Integrate one partition and export its exact finite native distance field."""
    from curobo.perception import Mapper
    from curobo.types import CameraObservation, DeviceCfg, Pose
    from .native_voxel_sources import export_source_bits
    from .online_geometry import FrozenESDF
    device = DeviceCfg(device=config.device)
    mapper = Mapper(config)
    for name, view in normalized.items():
        depth = np.where(masks[name][identity], view['depth'], 0.).astype(np.float32)
        observation = CameraObservation(name=name, depth_image=device.to_device(depth)[None],
            rgb_image=torch.as_tensor(np.asarray(view['image']).copy(), dtype=torch.uint8,device=device.device)[None],
            intrinsics=device.to_device(view['K'])[None],
            pose=Pose.from_matrix(device.to_device(inverse@view['T_output_camera'])[None]),
            resolution=list(depth.shape),depth_to_meter=1.)
        mapper.integrate(camera_observation=observation)
    stats=mapper.get_stats(scan_pool=True)
    if stats['allocation_failures'] or stats['num_allocated'] != stats['active_blocks']+stats['free_count']:
        raise ValueError('UNSUPPORTED_MODEL: native mapping allocation lost observed geometry')
    voxel=mapper.compute_esdf()
    from .collision_records import MAX_ARRAY_BYTES
    bits=export_source_bits(mapper,config,voxel,max_cells=MAX_ARRAY_BYTES//4)
    values=voxel.feature_tensor.detach().cpu().float().numpy()
    shape=np.asarray(values.shape)
    origin=np.asarray(voxel.pose[:3])-(shape-1)*float(voxel.voxel_size)/2
    name='observed_background' if identity is None else identity
    query=FrozenESDF(values,origin=origin,voxel_size=voxel.voxel_size,
        site_valid=(bits&1)!=0,observed_sign=(bits&2)!=0,owner_ids=(name,),
        source=f"current_rgbd:{','.join(normalized)}")
    return query,{'allocated_blocks':stats['num_allocated'],'shape_xyz':shape.tolist(),
        'voxel_size_m':float(voxel.voxel_size),'origin':origin.tolist(),
        'source_surface_cells':int(((bits&1)!=0).sum()),'observed_sign_cells':int(((bits&2)!=0).sum())}


def build_scene_from_rgbd(views,owner_frames,*,mapper_cfg,depth_error_m,owner_voxel_sizes_m,
                         reference_frame,robot_sphere_materials,robot_geometry_source,
                         unknown_policy,gripper=None,has_gripper=True,contacts=(),
                         max_mapping_bytes=256*1024**2,max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES,
                         max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES,
                         native_geometry_materials=None,robot_owner_permissions=(),
                         fine_regions=None,robot_collision_bodies=()):
    """Build one fixed-resolution field per owner from the current snapshot.

    Every view requires ``self_filter`` with uint8 ``labels``, independent
    ``free_ray_depth_m`` and a nonempty ``source``. Only ENVIRONMENT pixels
    contribute measured surface endpoints and free rays. SELF and
    AMBIGUOUS_SHADOW contribute neither; raw depth remains unchanged.

    ``fine_regions`` now selects a partition's sole field before mapping. It
    never creates a coarse layer, blends resolutions or expands after solving.
    The explicit unknown policy is separate from both field values and raw
    free-ray evidence. The audit builder remains explicitly named above.
    """
    from curobo.perception import MapperCfg
    from .online_geometry import OnlinePlanningScene
    if not isinstance(mapper_cfg,MapperCfg):
        raise TypeError('Provide the installed native MapperCfg')
    if (mapper_cfg.extent_esdf_meters_xyz is None or mapper_cfg.num_cameras!=1
            or mapper_cfg.enable_static or mapper_cfg.decay_factor!=1 or mapper_cfg.frustum_decay_factor!=1
            or mapper_cfg.edt_solver!='pba' or mapper_cfg.feature_dim!=0 or mapper_cfg.lidar_num_sensors!=0):
        raise ValueError('Current snapshot requires one explicit bounded nondecaying native map')
    if robot_owner_permissions:
        raise ValueError('Whole robot/owner pair permissions are not supported online')
    if (type(max_mapping_bytes) is not int or not 0<max_mapping_bytes<=min(max_host_rss_bytes,max_device_used_bytes)
            or not 0<max_host_rss_bytes<=MAX_HOST_RSS_BYTES or not 0<max_device_used_bytes<=MAX_DEVICE_USED_BYTES):
        raise ValueError('Mapping budgets must fit explicit host/device limits')
    # BEGIN online views require audited three-state evidence, never old robot_mask.
    normalized=_self_filtered_views(views,reference_frame,(mapper_cfg.depth_minimum_distance,mapper_cfg.depth_maximum_distance))
    # END depth limits cannot restore excluded endpoints or free rays.
    if any(view['depth'].shape!=(mapper_cfg.image_height,mapper_cfg.image_width) for view in normalized.values()):
        raise ValueError('Snapshot image dimensions must match the allocated camera buffer')
    if set(owner_frames)!=set(owner_voxel_sizes_m) or len(owner_frames)>32:
        raise ValueError('Every selected owner must have exactly one explicit resolution')
    frames={identity:(_pose_array(row['T_output_owner'],'current owner pose'),_text(row['source'],'owner source')) for identity,row in owner_frames.items()}
    masks=_partition_masks(normalized,frames)
    contacts=_validate_contact_and_gripper_inputs(dict.fromkeys(frames),tuple(robot_sphere_materials),contacts,gripper,has_gripper)
    configs={None:mapper_cfg}
    # BEGIN derive fixed owner interpolation support from the declared robot
    # sphere model before allocating any map or trying any candidate.
    radius=0.
    for body in robot_collision_bodies:
        values=np.asarray(body['local_spheres'],dtype=float)
        if values.ndim!=2 or values.shape[1]!=4 or not np.isfinite(values).all():
            raise ValueError('Owner map support needs the declared complete body sphere geometry')
        radius=max(radius,float(values[:,3].max(initial=0.)))
    # END no task identity or success/failure result chooses a map bound.
    for identity,(pose,_) in frames.items():
        configs[identity]=_owner_mapper_config(mapper_cfg,normalized,masks,identity,pose,
            owner_voxel_sizes_m[identity],depth_error_m,query_radius_m=radius)
    requests={} if fine_regions is None else dict(fine_regions)
    if not set(requests)<=set(configs):
        raise ValueError('Fixed regions must identify an existing physical owner or background')
    for identity,request in requests.items():
        configs[identity]=_fine_mapper_config(configs[identity],request)
    estimates={identity:_mapper_memory_bound(config) for identity,config in configs.items()}
    from .collision_records import MAX_ARRAY_BYTES
    for config in configs.values():
        shape=tuple(math.ceil(float(x)/config.esdf_voxel_size) for x in config.extent_esdf_meters_xyz)
        if min(shape)<2 or math.prod(shape)*4>MAX_ARRAY_BYTES:
            raise ValueError('UNSUPPORTED_MODEL: frozen distance field exceeds the 32 MiB binary record limit')
    if sum(estimates.values())>max_mapping_bytes:
        raise ValueError('UNSUPPORTED_MODEL: fixed scene exceeds declared mapping allocation')
    background_cfg=configs[None]
    shape=tuple(math.ceil(float(x)/background_cfg.esdf_voxel_size) for x in background_cfg.extent_esdf_meters_xyz)
    center=background_cfg.grid_center.detach().cpu().numpy()
    origin=center-(np.asarray(shape)-1)*background_cfg.esdf_voxel_size/2
    # BEGIN only ENVIRONMENT rays can prove free volume in the online scene.
    ray_depths={name: row['self_filter']['free_ray_depth_m'] for name,row in normalized.items()}
    known=_known_free_voxels(normalized,ray_depths=ray_depths,origin=origin,shape=shape[::-1],
        voxel_size=background_cfg.esdf_voxel_size,depth_error_m=depth_error_m)
    coverage=ObservedSpace(known,origin=origin,voxel_size=background_cfg.esdf_voxel_size,
        observation_ids=tuple(normalized),reference_frame=reference_frame)
    # END SELF and AMBIGUOUS_SHADOW stay unknown to coverage.
    queries,reports={},{}
    for identity,config in configs.items():
        _require_mapping_memory(estimates[identity],config.device,max_host_rss_bytes=max_host_rss_bytes,
            max_device_used_bytes=max_device_used_bytes)
        inverse=np.eye(4) if identity is None else _inverse(frames[identity][0])
        queries[identity],reports['observed_background' if identity is None else identity]=_map_online_partition(normalized,masks,identity,config,inverse)
    owners=[CollisionOwner(identity,queries[identity],PosePath(np.array([0.,1.]),np.stack((pose,pose)),source),
        source,geometry_complete=False) for identity,(pose,source) in frames.items()]
    scene=OnlinePlanningScene(background=queries[None],owners=owners,coverage=coverage,
        reference_frame=reference_frame,robot_sphere_materials=robot_sphere_materials,
        robot_geometry_source=robot_geometry_source,unknown_policy=unknown_policy,
        contacts=contacts,gripper=gripper,has_gripper=has_gripper,
        robot_collision_bodies=robot_collision_bodies,native_geometry_materials=native_geometry_materials)
    # BEGIN retain the exact self-filter and per-owner exclusion evidence.
    scene.observation_report={'source':'current calibrated owner-partitioned RGB-D','partitions':reports,
        'estimated_mapping_buffers_bytes':sum(estimates.values()),'depth_error_m':float(depth_error_m),
        'known_free_voxels':int(known.sum()),'coverage_voxels':int(known.size),
        # BEGIN describe the registered online contact model used by this scene.
        'geometry_model':'one frozen signed field per partition; prepared forbidden-region fields and registered native planar contact',
        # END source-cell interpretation is confined to preparation, not online queries.
        'unknown_policy':deepcopy(unknown_policy),
        'self_filter':_self_filter_report(normalized)}
    # END excluded surfaces remain independently auditable.
    return scene
# END the production builder never exports source solids or installs refinement.
