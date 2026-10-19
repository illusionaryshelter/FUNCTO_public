"""Current calibrated RGB-D to owner-separated cuRobo planning geometry.

The input is a registered snapshot of the current scene: camera-z depth in
metres, native pinhole calibration, camera-to-planning transforms, and explicit
visible owner masks and explicit three-state robot self-filter evidence.
No demonstration mesh, simulator state, inferred
back surface, or bounding-box erasure enters this adapter. cuRobo's existing
Mapper builds the collision field; independently admitted environment rays
supply observation diagnostics. Explicitly named audit APIs retain raw rays.
Optional, separately verified physical solid interiors can supply current
occupied evidence. A closed collision proxy is not such physical evidence.
"""

# BEGIN current RGB-D observation bridge using the installed native mapper.
from collections.abc import Mapping
from dataclasses import replace
import math
from numbers import Real

import numpy as np
# BEGIN image filtering is only for independent depth-footprint queries.
from scipy.ndimage import minimum_filter
# END semantic ownership does not use image morphology.
from scipy.spatial.transform import Rotation
import torch

from .motion import _inverse, _pose_array
from .observations import _intrinsics
from .planning_problem import PosePath
# BEGIN mapping and planning share explicit request-wide resource ceilings.
from .resource_admission import (
    DEFAULT_MAX_DEVICE_USED_BYTES, DEFAULT_MAX_HOST_RSS_BYTES,
    MAX_DEVICE_USED_BYTES, MAX_HOST_RSS_BYTES,
    require_device_memory, require_host_memory,
)
# END native Mapper allocations never borrow PyTorch's cached free blocks.
from .planning_scene import (
    CollisionOwner, ObservedSpace, _text,
    _validate_contact_and_gripper_inputs,
)
# BEGIN body-wide replacement by a separately observed fine field.
from .voxel_layers import MAX_FINE_CELLS
# END both native and independent consumers retain the same layer inventory.

# BEGIN fixed role-based observation resolution, independent of task names.
TASK_VOXEL_SIZE_M = .002
# The two recorded-input CLI entry points share this environment-grid default;
# task-owner resolution remains fixed separately by the role policy below.
DEFAULT_ENVIRONMENT_VOXEL_SIZE_M = .01
# END callers identify task owners explicitly before allocating native maps.


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


# BEGIN adapt calibrated integer pixel centres only at the native depth boundary.
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
# END this is coordinate conversion, not a depth shift or geometric allowance.


# BEGIN exact rectangle queries preserve the complete projected voxel footprint.
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
# END downward block sizes accelerate an exact query; they never expand rays.


# BEGIN callers explicitly select independent free-ray evidence.
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
    # BEGIN conserve every admitted endpoint under the supplied identities.
    """Partition current environment endpoints into owners and background.

    Check owner/owner overlap on the supplied masks before self filtering;
    uncertainty is not permission to silently choose a semantic winner.
    All admitted endpoints outside explicit owners remain observed obstacles;
    proximity to an owner neither grants its identity nor erases occupancy.
    Background stays fixed for this planning request, so the caller must
    completely partition visible surfaces of owners that will move. Measured
    free-ray prefixes and robot self-filter evidence remain independent.
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
        # The disjoint owner intersections and this complement exhaust the
        # accepted environment. SELF/shadow pixels never enter any partition.
        masks[name][None] = environment & ~occupied
        masks[name][None].setflags(write=False)
    if any(count == 0 for count in observed.values()):
        raise ValueError("Every explicit collision owner needs current valid depth evidence after self filtering")
    return masks
    # END input masks alone assign identity; no measured environment is erased.


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

    All task identities must be present, even for invisible or completely
    modelled bodies. A complete body avoids allocating a redundant RGB-D map.
    Only observed partitions use these field spacings. Resource
    admission may reject a request but cannot silently coarsen a task owner.
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
# END one library policy applies before any mapping or native allocation.


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



# BEGIN native Mapper ESDF is the one production observation distance model.
def _map_online_partition(normalized, masks, identity, config, inverse):
    """Integrate one owner and freeze its native ESDF without source solids."""
    from time import perf_counter
    from curobo.perception import Mapper
    from curobo.types import CameraObservation, DeviceCfg, Pose
    from .online_geometry import FrozenESDF
    device = DeviceCfg(device=config.device)
    mapper = Mapper(config)
    for name, view in normalized.items():
        depth = np.where(masks[name][identity], view['depth'], 0.).astype(np.float32)
        observation = CameraObservation(name=name, depth_image=device.to_device(depth)[None],
            rgb_image=torch.as_tensor(np.asarray(view['image']).copy(), dtype=torch.uint8, device=device.device)[None],
            intrinsics=device.to_device(_mapper_intrinsics(view['K']))[None],
            pose=Pose.from_matrix(device.to_device(inverse @ view['T_output_camera'])[None]),
            resolution=list(depth.shape), depth_to_meter=1.)
        mapper.integrate(camera_observation=observation)
    stats = mapper.get_stats(scan_pool=True)
    if stats['allocation_failures'] or stats['num_allocated'] != stats['active_blocks']+stats['free_count']:
        raise ValueError('Native mapping failed to retain the supplied observations')
    started = perf_counter()
    voxel = mapper.compute_esdf()
    name = 'observed_background' if identity is None else identity
    query = FrozenESDF.from_native_voxel(voxel, owner_ids=(name,),
                                        source=f"current_rgbd:{','.join(normalized)}")
    seconds = perf_counter()-started
    return query, {'allocated_blocks': stats['num_allocated'],
        'shape_xyz': list(query.distance.shape), 'voxel_size_m': query.voxel_size,
        'origin': query.origin.tolist(), 'esdf_preparation_seconds': seconds,
        'geometry_model': 'Native Mapper ESDF; approximate negative-inside distance, not physical contact truth'}
# END native Mapper ESDF is the one production observation distance model.


def build_scene_from_rgbd(views,owner_frames,*,mapper_cfg,depth_error_m,task_owner_ids,
                         reference_frame,robot_sphere_materials,robot_geometry_source,
                         gripper=None,has_gripper=True,contacts=(),
                         max_mapping_bytes=256*1024**2,max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES,
                         max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES,
                         native_geometry_materials=None,
                         fine_regions=None,robot_collision_bodies=(),complete_owners=(),
                         initial_support_relations=(), holding_contacts=()):
    """Build observed fields and admit current complete owners exactly once.

    Every view requires ``self_filter`` with uint8 ``labels``, independent
    ``free_ray_depth_m`` and a nonempty ``source``. Only ENVIRONMENT pixels
    contribute measured surface endpoints and free rays. SELF and
    AMBIGUOUS_SHADOW contribute neither; raw depth remains unchanged.
    # BEGIN explicit input identity must conserve observed occupancy.
    Every admitted endpoint enters its supplied owner or the background.
    Background is a world-fixed snapshot for this request, so visible surfaces
    of owners that will move must be completely assigned by the input masks.
    Unnamed endpoints retain occupancy and cannot borrow an owner's contact
    permission. Diagnostics expose the actual immutable background input and
    endpoint counts; older serialized geometry requires rebuilding.
    # END no segmentation heuristic changes endpoint or free-ray admission.

    ``fine_regions`` now selects a partition's sole field before mapping. It
    never creates a coarse layer, blends resolutions or expands after solving.
    Unknown observation evidence is reported separately from collision costs.
    No complete-source or full-body coverage proof is performed by this builder.

    ``complete_owners`` supplies existing CollisionOwner geometry in the same
    current owner frames. Their semantic masks still exclude those pixels
    from background mapping, and their measured rays still bound coverage.
    Only the remaining partitions allocate a Mapper. Task owners (the carrier
    and functional reference) use 2 mm grids; other owners use the coarser
    environment spacing from mapper_cfg. Task identities are required inputs,
    not names inferred from masks. Complete analytic owners use their supplied
    primitive geometry and require no secondary contact grid.
    # BEGIN initial support remains a separate current-state relation.
    ``initial_support_relations`` describes finite support already present at
    this request's entrance. It is passed to the same frozen scene together
    with the complete actor geometry, without copying a future placement's
    material identities, pose tolerances or required-contact conditions.
    # END no observation, free-space evidence or complete owner is replaced.
    """
    from curobo.perception import MapperCfg
    from .online_geometry import OnlinePlanningScene
    if not isinstance(mapper_cfg,MapperCfg):
        raise TypeError('Provide the installed native MapperCfg')
    if (mapper_cfg.extent_esdf_meters_xyz is None or mapper_cfg.num_cameras!=1
            or mapper_cfg.enable_static or mapper_cfg.decay_factor!=1 or mapper_cfg.frustum_decay_factor!=1
            or mapper_cfg.edt_solver!='pba' or mapper_cfg.feature_dim!=0 or mapper_cfg.lidar_num_sensors!=0):
        raise ValueError('Current snapshot requires one explicit bounded nondecaying native map')
    if (type(max_mapping_bytes) is not int or not 0<max_mapping_bytes<=min(max_host_rss_bytes,max_device_used_bytes)
            or not 0<max_host_rss_bytes<=MAX_HOST_RSS_BYTES or not 0<max_device_used_bytes<=MAX_DEVICE_USED_BYTES):
        raise ValueError('Mapping budgets must fit explicit host/device limits')
    # BEGIN online views require audited three-state evidence, never old robot_mask.
    normalized=_self_filtered_views(views,reference_frame,(mapper_cfg.depth_minimum_distance,mapper_cfg.depth_maximum_distance))
    # END depth limits cannot restore excluded endpoints or free rays.
    if any(view['depth'].shape!=(mapper_cfg.image_height,mapper_cfg.image_width) for view in normalized.values()):
        raise ValueError('Snapshot image dimensions must match the allocated camera buffer')
    # BEGIN the production path owns precision for every current request.
    if len(owner_frames)>32:
        raise ValueError('At most 32 observed owners are supported')
    task_owner_ids = tuple(task_owner_ids) if not isinstance(task_owner_ids, (str, bytes)) else task_owner_ids
    owner_voxel_sizes_m = observation_resolutions(owner_frames,
        task_owner_ids=task_owner_ids, environment_voxel_size=mapper_cfg.voxel_size)
    # END callers cannot downgrade a functional reference to save memory.
    frames={identity:(_pose_array(row['T_output_owner'],'current owner pose'),_text(row['source'],'owner source')) for identity,row in owner_frames.items()}
    # BEGIN consume only the complete owner/background endpoint partition.
    masks=_partition_masks(normalized,frames)
    # END all map inputs and later diagnostics share one partition decision.
    # BEGIN admit complete geometry before choosing observed map partitions.
    # The semantic owner inventory remains intact: skipping its Mapper must
    # neither turn its pixels into background nor assert its occluded space free.
    direct={}
    for owner in complete_owners:
        if not isinstance(owner,CollisionOwner) or not owner.geometry_complete:
            raise ValueError('Direct owners require complete CollisionOwner geometry')
        identity=owner.owner_id
        if identity not in frames or identity in direct:
            raise ValueError('Direct owners require unique identities in the current owner frames and masks')
        position,quaternion=owner.pose_path.sample(torch.tensor(0.,dtype=torch.float64))
        position=position.detach().cpu().numpy()
        quaternion=quaternion.detach().cpu().numpy()
        pose=frames[identity][0]
        if (position.shape!=(3,) or quaternion.shape!=(4,)
                or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
                or not np.isclose(np.linalg.norm(quaternion),1.,atol=1e-7,rtol=0)
                or not np.allclose(position,pose[:3,3],atol=1e-7,rtol=0)
                or not np.allclose(Rotation.from_quat(quaternion[[1,2,3,0]]).as_matrix(),pose[:3,:3],atol=1e-7,rtol=0)):
            raise ValueError('Direct owner initial pose must match its current observation frame')
        direct[identity]=owner
    owner_materials={identity:direct[identity].sphere_materials if identity in direct else None for identity in frames}
    contacts=_validate_contact_and_gripper_inputs(owner_materials,tuple(robot_sphere_materials),contacts,gripper,has_gripper)
    # END complete-owner material contacts enter the same final scene once.
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
        # BEGIN complete geometry needs no throwaway observed field.
        if identity in direct:
            continue
        # END its semantic partition remains in masks for every Mapper call.
        configs[identity]=_owner_mapper_config(mapper_cfg,normalized,masks,identity,pose,
            owner_voxel_sizes_m[identity],depth_error_m,query_radius_m=radius)
    requests={} if fine_regions is None else dict(fine_regions)
    if not set(requests)<=set(configs):
        raise ValueError('Fixed regions must identify an observed mapped owner or background')
    for identity,request in requests.items():
        configs[identity]=_fine_mapper_config(configs[identity],request)
        # BEGIN a bounded ROI cannot bypass the task/environment precision rule.
        spacing = configs[identity].voxel_size
        if identity in task_owner_ids:
            if not .001 <= spacing <= TASK_VOXEL_SIZE_M:
                raise ValueError('A task-owner ROI must retain 1–2 mm grid spacing')
        elif spacing <= TASK_VOXEL_SIZE_M:
            raise ValueError('An environment ROI must remain coarser than 2 mm')
        # END ROI bounds and measurement evidence retain their existing checks.
    # BEGIN observation masks share the unchanged native background layout.
    coverage_cfg = configs[None]
    # END no full-body query certificate expands the map.
    estimates={identity:_mapper_memory_bound(config) for identity,config in configs.items()}
    from .collision_records import MAX_ARRAY_BYTES
    for config in configs.values():
        shape=tuple(math.ceil(float(x)/config.esdf_voxel_size) for x in config.extent_esdf_meters_xyz)
        if min(shape)<2 or math.prod(shape)*4>MAX_ARRAY_BYTES:
            # BEGIN report the common codec limit rather than a stale local cap.
            raise ValueError(f'UNSUPPORTED_MODEL: frozen distance field exceeds the {MAX_ARRAY_BYTES // 1024**2} MiB binary record limit')
            # END allocation still also requires the unchanged total budgets.
    if sum(estimates.values())>max_mapping_bytes:
        # BEGIN expose the actual allocation inventory before touching CUDA.
        partitions={('observed_background' if identity is None else identity):value for identity,value in estimates.items()}
        raise ValueError(f'UNSUPPORTED_MODEL: fixed scene exceeds declared mapping allocation: '
            f'estimated_bytes={sum(estimates.values())}, limit_bytes={max_mapping_bytes}, '
            f'mapped_partition_bytes={partitions}, direct_complete_owners={list(direct)}')
        # END direct geometry is not charged as a discarded observation map.
    # BEGIN distance-query support cannot enlarge observed-free evidence.
    background_cfg=coverage_cfg
    # END all ray inputs and the original background coverage grid stay fixed.
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
        queries[identity], reports['observed_background' if identity is None else identity] = _map_online_partition(
            normalized, masks, identity, config, inverse)
    # BEGIN assemble the original complete objects and newly observed fields.
    owners=[direct[identity] if identity in direct else
        CollisionOwner(identity,queries[identity],PosePath(np.array([0.,1.]),np.stack((pose,pose)),source),
        source,geometry_complete=False) for identity,(pose,source) in frames.items()]
    # END no partial scene or post-mapping owner replacement is constructed.
    scene=OnlinePlanningScene(background=queries[None],owners=owners,coverage=coverage,
        reference_frame=reference_frame,robot_sphere_materials=robot_sphere_materials,
        robot_geometry_source=robot_geometry_source,
        contacts=contacts,gripper=gripper,has_gripper=has_gripper,
        # BEGIN bind explicit current support alongside ordinary material rules.
        initial_support_relations=tuple(initial_support_relations),
        # BEGIN current held materials are passed without former grasp constraints.
        holding_contacts=tuple(holding_contacts),
        # END the final scene validates the native robot identities and attachment.
        # END the online scene owns support geometry and acceptance semantics.
        robot_collision_bodies=robot_collision_bodies,native_geometry_materials=native_geometry_materials)
    # BEGIN report the exact background input and endpoint conservation.
    # Counts describe the current input partition, not permanent object
    # identity or proof of the Mapper's reconstructed surface completeness.
    from .robot_self_filter import ENVIRONMENT
    scene.observation_partition_masks={name:{'background':masks[name][None]}
        for name in normalized}
    endpoint_partition_report={}
    for name,view in normalized.items():
        environment_pixels=int(np.count_nonzero(view['self_filter']['labels']==ENVIRONMENT))
        owner_pixels={identity:int(np.count_nonzero(masks[name][identity])) for identity in frames}
        background_pixels=int(np.count_nonzero(masks[name][None]))
        endpoint_partition_report[name]={
            'policy':'Every admitted ENVIRONMENT endpoint belongs to its supplied owner or background; '
                     'owner masks and independent free-ray evidence are unchanged',
            'map_lifecycle':'Fresh Mapper per partition from current snapshot views; background is fixed '
                            'in world coordinates for this request, so moving owners require complete '
                            'visible input masks; old scene bindings retain their original geometry',
            'environment_pixels':environment_pixels,
            'owner_pixels':owner_pixels,
            'background_pixels':background_pixels,
            'all_environment_endpoints_assigned':background_pixels+sum(owner_pixels.values())==environment_pixels,
        }
    # END reported background masks are the ones submitted to mapping.
    # BEGIN retain the exact self-filter and per-owner exclusion evidence.
    scene.observation_report={'source':'current calibrated owner-partitioned RGB-D','partitions':reports,
        # BEGIN record the library decision together with actual allocated grids.
        'resolution_policy': {'task_owner_ids': list(task_owner_ids),
            'task_voxel_size_m': TASK_VOXEL_SIZE_M,
            'environment_voxel_size_m': float(mapper_cfg.voxel_size),
            'owner_voxel_sizes_m': {identity: (float(configs[identity].voxel_size)
                if identity in configs else owner_voxel_sizes_m[identity]) for identity in frames},
            'source': 'Library task-owner mapping policy; explicit grounded carrier/functional reference identities'},
        # END complete geometry and observed maps share the same role inventory.
        'estimated_mapping_buffers_bytes':sum(estimates.values()),'depth_error_m':float(depth_error_m),
        # BEGIN distinguish native/complete geometry from allocated RGB-D fields.
        'mapped_partition_estimates_bytes':{('observed_background' if identity is None else identity):value for identity,value in estimates.items()},
        'direct_complete_owners':{identity:{'source':owner.source,'query_type':type(owner.query).__name__,
            'mapping_allocated':False} for identity,owner in direct.items()},
        # END mapping estimates do not claim to measure total scene residency.
        'known_free_voxels':int(known.sum()),'coverage_voxels':int(known.size),
        # BEGIN retain the independent free-ray grid beside numerical support.
        'coverage_grid': {'origin_m': origin.tolist(), 'shape_xyz': list(shape),
            'voxel_size_m': float(background_cfg.esdf_voxel_size),
            'source': 'Unchanged pre-query-expansion background grid and ENVIRONMENT free rays'},
        # END an enlarged ESDF is never recorded as newly observed free space.
        # BEGIN describe the registered online contact model used by this scene.
        'geometry_model':'Native Mapper ESDF and explicit analytic owner models; ordinary interaction pairs, no contact-success proof',
        # END source-cell interpretation is confined to preparation, not online queries.
        'unknown_space':'diagnostic_only',
        'self_filter':_self_filter_report(normalized),
        # BEGIN explicit identities conserve occupied endpoints independently of rays.
        'endpoint_partition':endpoint_partition_report}
        # END the original self-filter labels and free-ray evidence remain unchanged.
    # END excluded surfaces remain independently auditable.
    return scene
# END no old source-world audit, Boolean solver or adaptive refinement is installed.
