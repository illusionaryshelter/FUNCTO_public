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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.robot_geometry_check import PairQuery, RobotGeometryChecker
from manipulation.robot_validation import RobotStateValidator
from check_recorded_robot_geometry import _native_shapes
from robot_fixture import _native_ignored_pairs, _pose, validate_fixture_collision_contract


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

    Calibration identity is explicit. The current recording backend returns
    MuJoCo's world-frame native TCP site with identity link-from-TCP transform;
    a caller cannot relabel that site or silently apply another offset.
    """
    robot_config = yaml.safe_load(Path(fixture["robot_config_path"]).read_text())
    validated = validate_fixture_collision_contract(fixture, robot_config)
    model = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])
    if model.nq > 64 or model.nbody > 128 or model.ngeom > 64:
        raise ValueError("Selected fixture exceeds the established bounded CPU robot-check budget")
    expected_tool_transform = _pose(fixture["tool_transform"]["position"],
                                    fixture["tool_transform"]["quaternion_wxyz"])
    if (tool_frame != fixture["tool_link"] or tool_frame != fixture["tcp_site"]
            or not np.array_equal(T_frame_tcp, expected_tool_transform)):
        raise ValueError("Independent validator must retain the verified native TCP identity and transform")
    shapes, geometry = _native_shapes(fixture, model)
    geometry_names = tuple(geometry)
    policy = _native_pair_policies(model, geometry_names)
    # Source and exported models must make exactly the same per-geom policy
    # decision, including same-body pairs and individually masked geom pairs.
    source_model = mujoco.MjModel.from_xml_path(fixture["source_scene_xml"])
    if policy != _native_pair_policies(source_model, geometry_names):
        raise ValueError("Independent pair policy differs between original source and exported robot")
    # BEGIN use the same explicit mechanical contract as native planning queries.
    # Fixture validation above independently binds every declaration to both
    # source and exported geometry; nothing is inferred from geom names or
    # contact flags. Enabled physical contacts remain enabled in MuJoCo.
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
            f"{fixture['source_scene_xml']} and {fixture['robot_mjcf_path']}: {reason}; "
            "no sphere-certificate exclusion"))
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
        source=(f"Source-verified fixture {fixture['robot_mjcf_path']}; "
                f"{validated['complete_native_geometry_count']} articulated and "
                f"{validated['complete_fixed_installation_geometry_count']} fixed native solids"))
# END source-verified robot geometry and policy adapter for execution.
