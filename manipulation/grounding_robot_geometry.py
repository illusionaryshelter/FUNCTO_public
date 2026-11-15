"""Official fixed-robot geometry for grounding, planning and measured-image filtering.

The robot and its installed end effector are fixed across tasks. Their official
collision shapes and fixed component relations are therefore reused directly;
calibration, measured joint coordinates and forward kinematics place them in
the observation world. Task entities enter the geometry interface through
RGB-D TSDF fields, independently of this robot-only model adapter.

The fixture is an explicit official-robot manifest with robot_mjcf_path,
urdf_path, named joints, registered collision components and mechanical pair
permissions. No running simulation state is read by this module.
"""

# BEGIN shared official robot geometry, FK and native pair policy.
from itertools import combinations
from time import perf_counter

import coal
import mujoco
import pinocchio as pin
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation

from .robot_geometry_check import NativePrimitive, NativeSolid, PairQuery, RobotGeometryChecker
from .robot_validation import RobotStateValidator
from .robot_self_filter import RobotCollisionShape
from .tool_geometry import _collision_geometry
from .robot_geometry_check import _convex_solid
from .sampled_geometry_objective import MaterialSamples

def _pose(position, quaternion):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    value[:3, 3] = position
    return value

def _native_geom_pose(model, geom_id):
    """Use MuJoCo's actual homogeneous geom matrix, not SciPy normalization."""
    position, quaternion = model.geom_pos[geom_id], model.geom_quat[geom_id]
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("Native collision geom poses must be finite")
    result, rotation = np.eye(4), np.empty(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    result[:3, :3], result[:3, 3] = rotation.reshape(3, 3), position
    return result

def _native_ignored_pairs(model, collision_links):
    """Keep exactly the native welded/parent and collision-mask exclusions.

    A link pair is ignored only when all its native geom pairs are excluded.
    Coarse proxy overlap is not a reason to remove an enabled collision pair.
    """
    result = {}
    filter_parent = not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    for i, name in enumerate(collision_links):
        body = model.body(name).id
        first = np.flatnonzero(model.geom_bodyid == body)
        for other_name in collision_links[i+1:]:
            other = model.body(other_name).id
            second = np.flatnonzero(model.geom_bodyid == other)
            weld_a, weld_b = int(model.body_weldid[body]), int(model.body_weldid[other])
            parent_a = int(model.body_weldid[model.body_parentid[weld_a]])
            parent_b = int(model.body_weldid[model.body_parentid[weld_b]])
            welded = weld_a == weld_b
            parent = filter_parent and weld_a != 0 and weld_b != 0 and (parent_a == weld_b or parent_b == weld_a)
            masked = all(not ((model.geom_contype[a] & model.geom_conaffinity[b]) or
                              (model.geom_contype[b] & model.geom_conaffinity[a]))
                         for a in first for b in second)
            if welded or parent or masked:
                result.setdefault(name, []).append(other_name)
    return result

# BEGIN consume the reusable robot asset directly, without task-scene or sphere certificates.
def _known_shapes(fixture, model):
    """Read explicitly registered entities from the compiled robot-only asset.

    Mesh vertices and their fixed local transforms are the model's original
    collision representation. Boxes and cylinders stay analytic. The second
    result retains geometry-local coordinates for shadow padding about each
    entity's own origin.
    """
    shapes, local_records = [], []
    for row in (*fixture["collision_geometries"], *fixture["fixed_collision_geometries"]):
        name, frame = row["geom"], row["link"]
        index = model.geom(name).id
        if model.body(model.geom_bodyid[index]).name != frame:
            raise ValueError(f"Registered robot geometry is attached to a different frame: {name}")
        local = _native_geom_pose(model, index)
        source = f"{fixture['robot_mjcf_path']}::geom[{name}]"
        kind = model.geom_type[index]
        if kind in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
            size = model.geom_size[index]
            label = "box" if kind == mujoco.mjtGeom.mjGEOM_BOX else "cylinder"
            dimensions = tuple(2 * size) if label == "box" else (size[0], 2 * size[1])
            shapes.append(NativePrimitive(name, frame, label, dimensions, local, source))
            fields = dict(kind=label, dimensions=dimensions)
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            native = _collision_geometry(model, index, "registered_robot_component")
            vertices = native["vertices"]
            shapes.append(NativeSolid(name, frame,
                vertices @ local[:3, :3].T + local[:3, 3], source))
            fields = dict(kind="convex", vertices=vertices)
        else:
            raise ValueError(f"Unsupported registered native collision shape: {name}")
        local_records.append((frame, local, dict(name=name, source=source, **fields)))
    return shapes, local_records


def robot_self_filter_shapes(fixture, frame_poses):
    """Place the same asset's original components at image-capture joint FK."""
    model = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])
    _, records = _known_shapes(fixture, model)
    return tuple(RobotCollisionShape(
        T_world_geometry=np.asarray(frame_poses[frame]) @ local, **fields)
        for frame, local, fields in records)
# END only the explicit robot model is loaded; task assets never enter this adapter.


# BEGIN the recorded robot topology supplies the complete current tool component.
def tool_geometry_names(fixture):
    """Return registered geometry under the named finger mechanism's root.

    This source adapter uses the fixture's declared finger joints to identify
    their common parent assembly. The production scene receives only explicit
    registered geometry IDs; it has no finger-count or robot-name convention.
    """
    model = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])

    def ancestors(body):
        chain = []
        while body:
            chain.append(body)
            body = int(model.body_parentid[body])
        return chain

    chains = [ancestors(int(model.body_parentid[model.jnt_bodyid[model.joint(name).id]]))
              for name in fixture["finger_joint_names"]]
    root = next(body for body in chains[0] if all(body in chain for chain in chains[1:]))
    return tuple(row["geom"] for row in fixture["collision_geometries"]
                 if root in ancestors(int(model.geom_bodyid[model.geom(row["geom"]).id])))
# END membership depends on the reusable robot model, not on the current object.


def _native_pair_policies(model, geometry_names):
    """Retain native body filtering and each geom's exact bitmask decision.

    Link-level policy is already defined by the maintained fixture adapter.
    Here individual geom masks additionally preserve excluded pairs inside a
    partially enabled link pair. Sphere separation certificates are deliberately
    absent: their native solids still require an actual distance query.
    """
    # BEGIN bind enabled pairs only when native physics processes contacts.
    # Geometric distance queries remain valid with these flags disabled, but
    # that is not the native physical collision policy claimed by this adapter.
    disabled = int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    if int(model.opt.disableflags) & disabled:
        raise ValueError("Native contact and constraint processing must both be enabled for the execution policy")
    # END reject disabled contact physics without changing geometry queries.
    bodies = {name: int(model.geom_bodyid[model.geom(name).id]) for name in geometry_names}
    links = {name: model.body(body).name for name, body in bodies.items()}
    ignored = _native_ignored_pairs(model, list(dict.fromkeys(links.values())))
    result = []
    for first, second in combinations(geometry_names, 2):
        a, b = model.geom(first).id, model.geom(second).id
        link_a, link_b = links[first], links[second]
        masked = not ((model.geom_contype[a] & model.geom_conaffinity[b])
                      or (model.geom_contype[b] & model.geom_conaffinity[a]))
        body_excluded = (bodies[first] == bodies[second] or link_b in ignored.get(link_a, ())
                         or link_a in ignored.get(link_b, ()))
        reasons = []
        if body_excluded:
            reasons.append("native same/welded-body, enabled parent filter, or all-geom mask")
        if masked:
            reasons.append("native individual geom contype/conaffinity mask")
        result.append((first, second, "excluded" if reasons else "required",
                       "; ".join(reasons) if reasons else "native enabled geom pair"))
    return result


def build_robot_validator(fixture, *, reference_frame, T_reference_base, tool_frame, T_frame_tcp):
    """Build the mandatory CPU gate from the same complete source robot model.

    # BEGIN the robot pose channel is calibrated FK, not a native pose sensor.
    Calibration identity is explicit. The recording backend estimates the
    world-frame TCP from measured robot joints and the known fixed model, with
    an identity link-from-TCP transform. This independent model checks that FK
    convention; a caller cannot relabel the frame or silently add an offset.
    # END matching models do not establish independent physical TCP accuracy.
    """
    # BEGIN the selected native model supplies geometry and pair semantics directly.
    model = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])
    if model.nq > 64 or model.nbody > 128 or model.ngeom > 64:
        raise ValueError("Selected fixture exceeds the established bounded CPU robot-check budget")
    expected_tool_transform = _pose(fixture["tool_transform"]["position"],
                                    fixture["tool_transform"]["quaternion_wxyz"])
    if (tool_frame != fixture["tool_link"] or tool_frame != fixture["tcp_site"]
            or not np.array_equal(T_frame_tcp, expected_tool_transform)):
        raise ValueError("Independent validator must retain the verified native TCP identity and transform")
    shapes, _ = _known_shapes(fixture, model)
    geometry_names = tuple(shape.name for shape in shapes)
    policy = _native_pair_policies(model, geometry_names)
    # BEGIN use the same explicit mechanical contract as native planning queries.
    # Mechanical permissions are explicit declarations over registered IDs.
    # Enabled physical contacts remain enabled in the native robot model.
    permitted = {frozenset((row["first"], row["second"])): row
                 for row in fixture["mechanical_contact_pairs"]}
    pairs = []
    for first, second, identity, reason in policy:
        declaration = permitted.get(frozenset((first, second)))
        if declaration is not None:
            if identity != "required":
                raise ValueError("Mechanical contact must remain an enabled native physical pair")
            identity = "mechanical_contact"
            reason += f"; explicit robot mechanical permission: {declaration['source']}"
        pairs.append(PairQuery(first, second, identity,
            f"{fixture['robot_mjcf_path']}: {reason}"))
    # END query all native solids while separating legality from contact physics.
    checker = RobotGeometryChecker(fixture["urdf_path"], base_frame=fixture["base_link"], geometries=shapes)
    for name in fixture["joint_names"]:
        if not np.array_equal(checker.joint_limits[name], model.jnt_range[model.joint(name).id]):
            raise ValueError(f"Independent Pinocchio and native position limits disagree: {name}")
    # These tolerances compare two double-precision kinematic models of the
    # same source asset. They never relax a collision distance, hardware limit,
    # physical tracking band, or object-contact permission.
    return RobotStateValidator(checker, joint_names=fixture["joint_names"],
        arm_joint_names=fixture["arm_joint_names"], finger_joint_names=fixture["finger_joint_names"],
        pairs=pairs, named_velocity_limits=fixture["velocity_limits"],
        reference_frame=reference_frame, T_reference_base=T_reference_base,
        tool_frame=tool_frame, T_frame_tcp=T_frame_tcp,
        fk_translation_tolerance_m=1e-8, fk_rotation_tolerance_rad=1e-8,
        source=(f"Known robot model {fixture['robot_mjcf_path']}; "
                f"{len(fixture['collision_geometries'])} robot entities and "
                f"{len(fixture['fixed_collision_geometries'])} fixed installations"))
    # END calibration tolerances retain their original kinematic-only meaning.
# END shared official robot geometry, FK and native pair policy.


# BEGIN exact TCP-fixed components and explicit native finger states.
def _model_q(model, names, positions):
    """Place the named encoder coordinates in the fixed model's q ordering."""
    q = pin.neutral(model)
    for name, value in zip(names, positions):
        q[model.joints[model.getJointId(name)].idx_q] = value
    return q

def _frame_placements(model, data, q):
    """Return model-base FK for every registered frame at an explicit state."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf

def _depends_on_joint(model, joint, selected):
    """Test mechanism membership from the original kinematic parent chain."""
    while joint:
        if joint in selected:
            return True
        joint = int(model.parents[joint])
    return False

def _coal_geometry(kind, dimensions, vertices):
    """Rebuild the original analytic primitive or original native convex solid."""
    if kind == "box":
        return coal.Box(*dimensions)
    if kind == "cylinder":
        return coal.Cylinder(*dimensions)
    if kind == "convex":
        return _convex_solid(vertices)
    raise ValueError(f"Unsupported registered tilt component kind: {kind}")

def build_tool_parts(fixture, feedback, entrance_finger_positions, native_finger_positions, tool_geometry):
    """Bind TCP-fixed geometry and declared finger states to the official asset.

    A frame with the TCP's exact ``parentJoint`` has a constant joint-local
    placement, including any fixed intervening frames.  Such parts require no
    arm IK.  Finger descendants are placed by their declared two slide states;
    their model-base FK cancels when converted into the same TCP coordinates.
    Other movable arm links do not belong to this local rigid-motion model.

    The robot and end effector are the fixed official configuration reused
    across tasks.  Their shapes and fixed transforms therefore remain valid;
    calibrated encoders place them at capture, independently of task objects.
    """
    started = perf_counter()
    native = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])
    model = pin.buildModelFromUrdf(fixture["urdf_path"])
    data = model.createData()
    tcp_frame = model.frames[model.getFrameId(fixture["tool_link"])]
    tcp_parent = int(tcp_frame.parentJoint)
    finger_joints = {int(model.getJointId(name)) for name in fixture["finger_joint_names"]}
    initial_q = _model_q(model, feedback["joint_names"], feedback["start_q"])
    # T_frame_tcp is the declared fixed calibration from the named robot
    # frame to the TCP.  A shared parent joint fixes the frame relation, while
    # this independent offset determines the point about which tilt rotates.
    T_frame_tcp = np.asarray(feedback["tool_transform"], dtype=float)
    T_tcp_frame = np.linalg.inv(T_frame_tcp)
    capture_placements = _frame_placements(model, data, initial_q)
    frame_names = tuple(dict.fromkeys(row["link"] for row in
        (*fixture["collision_geometries"], *fixture["fixed_collision_geometries"])))
    world_base = np.asarray(feedback["base_pose"], dtype=float)
    capture_frames = {name: world_base @ capture_placements[model.getFrameId(name)].homogeneous
                      for name in frame_names}

    states = {
        "entrance_open": {
            "joint_positions_m": list(entrance_finger_positions),
            "source": "named image-capture finger encoders",
        },
        "native_width": {
            "joint_positions_m": np.asarray(native_finger_positions).tolist(),
            "source": "selected native candidate aperture in declared finger order",
        },
    }
    # This maintained official-tool API owns the signed inner-pad-face gap.
    # It evaluates the same declared slide coordinates and supplies metadata;
    # the Coal shapes queried below are the complete fixed robot fixture.
    jaw = tool_geometry
    for state in states.values():
        state["signed_pad_gap_m"] = float(jaw.at(state["joint_positions_m"])["signed_gap_m"])

    parts, cache_arrays, component_records = [], {}, []
    for declaration in fixture["collision_geometries"]:
        frame = model.frames[model.getFrameId(declaration["link"])]
        fixed = int(frame.parentJoint) == tcp_parent
        finger = _depends_on_joint(model, int(frame.parentJoint), finger_joints)
        if not fixed and not finger:
            continue
        geom_id = native.geom(declaration["geom"]).id
        native_transform = _native_geom_pose(native, geom_id)
        native_kind = native.geom_type[geom_id]
        if native_kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            # The cylinder is an analytic filled Coal entity.  Trimesh only
            # supplies actual on-cylinder coordinates for guide samples and
            # rendering; these triangles never become the collision model.
            kind = "cylinder"
            dimensions = [float(native.geom_size[geom_id, 0]),
                          float(2 * native.geom_size[geom_id, 1])]
            display = trimesh.creation.cylinder(radius=dimensions[0], height=dimensions[1])
            vertices, faces = np.asarray(display.vertices), np.asarray(display.faces)
            display_source = "analytic cylinder display/guide samples from trimesh.creation.cylinder"
        else:
            native_geometry = _collision_geometry(native, geom_id, "official_tcp_tilt_component")
            vertices, faces = native_geometry["vertices"], native_geometry["faces"]
            kind = "box" if native_kind == mujoco.mjtGeom.mjGEOM_BOX else "convex"
            dimensions = (2 * native.geom_size[geom_id]).tolist() if kind == "box" else []
            display_source = "original compiled native vertices and faces"
        geometry = _coal_geometry(kind, dimensions, vertices)
        material = MaterialSamples(geometry)
        guide_local = np.unique(np.concatenate((material.points, vertices)), axis=0)
        source = f"{fixture['robot_mjcf_path']}::geom[{declaration['geom']}]"
        bound_states = {"tcp_fixed": None} if fixed else states
        for state_id, state in bound_states.items():
            if fixed:
                # Both placements refer to the SAME parent joint.  This is an
                # exact fixed-assembly transform, independent of arm posture.
                tool_frame = tcp_frame.placement.inverse() * frame.placement
            else:
                q = initial_q.copy()
                for name, value in zip(fixture["finger_joint_names"], state["joint_positions_m"]):
                    q[model.joints[model.getJointId(name)].idx_q] = value
                placed = _frame_placements(model, data, q)
                tool_frame = (placed[model.getFrameId(fixture["tool_link"])].inverse()
                              * placed[model.getFrameId(declaration["link"])])
            transform = T_tcp_frame @ tool_frame.homogeneous @ native_transform
            vertices_tool = vertices @ transform[:3, :3].T + transform[:3, 3]
            guide_tool = guide_local @ transform[:3, :3].T + transform[:3, 3]
            identity = declaration["geom"] if fixed else f"{declaration['geom']}@{state_id}"
            part = dict(id=identity, geometry_id=declaration["geom"], state_id=state_id,
                        geometry=geometry, T_tool_geometry=transform,
                        guide_points_tool=guide_tool, vertices_tool=vertices_tool,
                        faces=faces, source=source)
            index = len(parts)
            parts.append(part)
            component_records.append(dict(id=identity, geometry_id=declaration["geom"],
                state_id=state_id, link=declaration["link"], geometry_kind=kind,
                dimensions=dimensions, source=source,
                display_coordinate_source=display_source,
                T_tool_geometry=transform.tolist(), vertices_tool=vertices_tool.tolist(),
                faces=faces.tolist(), guide_point_count=len(guide_tool),
                frame_parent_joint=int(frame.parentJoint), tcp_parent_joint=tcp_parent,
                transform_source=("shared-parentJoint Frame.placement" if fixed else
                                  "official model FK at explicit finger coordinates")))
            cache_arrays.update({f"part_{index}_vertices_geometry": vertices,
                f"part_{index}_vertices_tool": vertices_tool,
                f"part_{index}_guide_points_tool": guide_tool,
                f"part_{index}_faces": faces,
                f"part_{index}_T_tool_geometry": transform})
    return parts, states, capture_frames, component_records, cache_arrays, {
        "seconds": perf_counter() - started,
        "tcp_frame": fixture["tool_link"], "tcp_parent_joint": tcp_parent,
        "rigid_geometry_ids": [part["geometry_id"] for part in parts if part["state_id"] == "tcp_fixed"],
        "finger_geometry_ids": list(dict.fromkeys(part["geometry_id"] for part in parts
                                                  if part["state_id"] != "tcp_fixed")),
        "source": fixture["source"], "robot_mjcf_path": fixture["robot_mjcf_path"],
        "kinematics_urdf_path": fixture["urdf_path"],
        "T_frame_tcp": T_frame_tcp.tolist(),
        "T_frame_tcp_source": "capture feedback tool_transform; fixed official fixture tool_transform declaration",
        "fixture_tool_transform": fixture["tool_transform"],
        "guide_definition": ("original MaterialSamples points plus original native polytope vertices "
                             "or on-analytic-surface display/guide samples"),
        "capture_frame_pose_source": "named image-capture joint encoders, calibration and official FK",
    }
# END exact TCP-fixed components and explicit native finger states.
