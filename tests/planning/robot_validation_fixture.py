"""Bind the generic independent execution validator to a verified robot fixture.

Only this recorded-model adapter understands fixture manifests and MuJoCo
source policy. The production validator receives explicit solids, named model
limits and a complete pair contract; it contains no episode/task exceptions.
"""

# BEGIN source-verified robot geometry and policy adapter for execution.
from itertools import combinations
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.robot_geometry_check import NativePrimitive, NativeSolid, PairQuery, RobotGeometryChecker
from manipulation.robot_validation import RobotStateValidator
from manipulation.robot_self_filter import RobotCollisionShape
from manipulation.tool_geometry import _collision_geometry
from robot_fixture import _native_ignored_pairs, _native_geom_pose, _pose


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
# END source-verified robot geometry and policy adapter for execution.
