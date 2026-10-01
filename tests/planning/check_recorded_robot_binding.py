"""Audit an explicitly mapped recorded robot against a planning URDF on CPU.

Only the named robot subtree, its scalar joint columns, and its named TCP site
are admitted. No object state, contact stream, evaluator, physics step, CUDA
context, or planner is used. Collision results are sampled geometric witnesses:
an uncovered point disproves sphere coverage; covered samples do not prove it.
"""

# BEGIN recorded robot binding audit with explicit names and CPU-only evidence.
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from yourdfpy import URDF
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.tool_geometry import _collision_geometry


def _pose(position, rotation):
    result = np.eye(4)
    result[:3, :3] = np.asarray(rotation).reshape(3, 3)
    result[:3, 3] = position
    return result


def _mapping(values, width):
    result = [value.split(":") for value in values]
    if any(len(row) != width or any(not item for item in row) for row in result):
        raise ValueError("Every explicit mapping must contain the declared colon-separated fields")
    if len({row[0] for row in result}) != len(result):
        raise ValueError("Source names in a mapping must be unique")
    return result


def _robot_model(scene_path, root_body, collision_names):
    """Compile only the named world child and explicitly named collision geoms.

    Saved MuJoCo XML has expanded defaults. Mesh assets are selected from the
    robot's kept geoms; all non-robot bodies, visuals and scene sensors are
    excluded before compilation. Explicit inertials preserve native FK bodies.
    """
    scene = ET.parse(scene_path).getroot()
    source = scene.find(f"worldbody/body[@name='{root_body}']")
    if source is None:
        raise ValueError("root-body must identify a direct child of worldbody")
    robot = deepcopy(source)
    for parent in robot.iter():
        for child in list(parent):
            if child.tag not in ("body", "joint", "inertial", "site", "geom"):
                parent.remove(child)
            elif child.tag == "geom" and child.get("name") not in collision_names:
                parent.remove(child)
            elif child.tag == "geom":
                child.attrib.pop("material", None)
    kept = {geom.get("name") for geom in robot.iter("geom")}
    if kept != set(collision_names):
        raise ValueError("Every collision geom must be inside the explicit robot subtree")
    needed_meshes = {geom.get("mesh") for geom in robot.iter("geom") if geom.get("mesh")}
    minimal = ET.Element("mujoco", model="robot_binding_cpu_audit")
    compiler = scene.find("compiler")
    if compiler is not None:
        minimal.append(deepcopy(compiler))
    if scene.find("default") is not None:
        minimal.append(deepcopy(scene.find("default")))
    assets = ET.SubElement(minimal, "asset")
    source_assets = scene.find("asset")
    for mesh in source_assets.findall("mesh"):
        if mesh.get("name") in needed_meshes:
            copied = deepcopy(mesh)
            mesh_path = Path(copied.attrib["file"])
            if not mesh_path.is_absolute():
                mesh_directory = compiler.get("meshdir", "") if compiler is not None else ""
                mesh_path = scene_path.parent / mesh_directory / mesh_path
            copied.set("file", str(mesh_path.resolve()))
            assets.append(copied)
    ET.SubElement(minimal, "worldbody").append(robot)
    return mujoco.MjModel.from_xml_string(ET.tostring(minimal, encoding="unicode"))


def _error_summary(transforms):
    values = np.asarray(transforms)
    reference = values[0]
    delta = np.linalg.inv(reference)[None] @ values
    return {
        "first_relative_transform": reference.tolist(),
        "maximum_translation_bias_m": float(np.linalg.norm(values[:, :3, 3], axis=1).max()),
        "maximum_rotation_bias_rad": float(Rotation.from_matrix(values[:, :3, :3]).magnitude().max()),
        "maximum_translation_variation_m": float(np.linalg.norm(delta[:, :3, 3], axis=1).max()),
        "maximum_rotation_variation_rad": float(Rotation.from_matrix(delta[:, :3, :3]).magnitude().max()),
    }


def _pose_error(transform):
    """Return position and angle magnitudes without conflating their units."""
    return {"translation_m": float(np.linalg.norm(transform[:3, 3])),
            "rotation_rad": float(Rotation.from_matrix(transform[:3, :3]).magnitude())}


def run(args):
    if not 2 <= args.sample_count <= 64:
        raise ValueError("The CPU audit supports 2 to 64 evenly spaced recorded samples")
    joint_map = _mapping(args.joint_map, 3)
    link_map = _mapping(args.link_map, 2)
    geom_map = _mapping(args.geom_map, 2)
    signs = np.array([float(row[2]) for row in joint_map])
    if not np.all(np.isin(signs, (-1., 1.))):
        raise ValueError("Explicit joint mapping signs must be +1 or -1")
    catalog = json.loads(args.catalog.read_text())
    catalog_joints = {row["name"]: row for row in catalog["joints"]
                      if row["name"] in {item[0] for item in joint_map}}
    site_record = next(row for row in catalog["sites"] if row["name"] == args.tcp_site)
    model = _robot_model(args.scene, args.root_body, [row[0] for row in geom_map])
    data = mujoco.MjData(model)
    q_addresses = np.array([model.joint(row[0]).qposadr[0] for row in joint_map])
    if len(q_addresses) != model.nq or len(set(q_addresses)) != model.nq:
        raise ValueError("Joint mappings must cover every scalar robot coordinate exactly once")
    for row in catalog_joints.values():
        if row["qpos_count"] != 1 or row["qvel_count"] != 1:
            raise ValueError("This explicit binding audit only supports scalar robot joints")
    urdf = URDF.load(args.urdf, build_scene_graph=True, load_meshes=False,
                     build_collision_scene_graph=False, load_collision_meshes=False)
    config = yaml.safe_load(args.robot_config.read_text())["robot_cfg"]["kinematics"]
    if config["base_link"] != args.urdf_base or args.tool_link not in config["tool_frames"]:
        raise ValueError("Explicit URDF base and tool must match the selected planner configuration")
    geometries = {name: _collision_geometry(model, model.geom(name).id, "audit")
                  for name, _ in geom_map}

    # HDF5 selection is deliberately column-specific. Reading full qpos rows
    # would unnecessarily bring object free joints into this robot-only audit.
    with h5py.File(args.episode, "r") as stream:
        total_frames = len(stream["states/qpos"])
        if any(len(stream[key]) != total_frames for key in
               ("states/qvel", "states/site_xpos", "states/site_xmat")):
            raise ValueError("Recorded robot q, dq and TCP streams must share exactly the same frame count")
        frames = np.unique(np.linspace(0, total_frames - 1, args.sample_count, dtype=int))
        q_recorded = np.column_stack([
            stream["states/qpos"][frames, catalog_joints[row[0]]["qpos_start"]]
            for row in joint_map])
        dq_recorded = np.column_stack([
            stream["states/qvel"][frames, catalog_joints[row[0]]["qvel_start"]]
            for row in joint_map])
        recorded_positions = stream["states/site_xpos"][frames, site_record["id"]]
        recorded_rotations = stream["states/site_xmat"][frames, site_record["id"]]

    # Deterministic scalar joint-range probes include both opening extremes.
    # They are hypothetical FK configurations, never simulator actions.
    native_limits = np.array([model.jnt_range[model.joint(row[0]).id] for row in joint_map])
    rng = np.random.default_rng(7341)
    probes = rng.uniform(native_limits[:, 0], native_limits[:, 1], size=(16, len(joint_map)))
    probes[0], probes[1] = native_limits[:, 0], native_limits[:, 1]
    samples = np.concatenate((q_recorded, probes))
    link_errors = {name: [] for name, _ in link_map}
    tcp_offsets, record_errors, corrected_record_errors, sample_errors = [], [], [], []
    coverage = {name: {"urdf_link": link, "vertices_per_sample": len(geometries[name]["vertices"]),
                       "maximum_own_link_sphere_gap_m": -float("inf"),
                       "maximum_all_robot_sphere_gap_m": -float("inf"),
                       "uncovered_vertex_samples": 0} for name, link in geom_map}
    base_id, site_id = model.body(args.base_body).id, model.site(args.tcp_site).id
    for index, joints in enumerate(samples):
        data.qpos[q_addresses] = joints
        mujoco.mj_kinematics(model, data)
        world_base = _pose(data.xpos[base_id], data.xmat[base_id])
        urdf.update_cfg(dict(zip((row[1] for row in joint_map), signs * joints)))
        world_tool = world_base @ urdf.get_transform(args.tool_link, frame_from=args.urdf_base)
        world_tcp = _pose(data.site_xpos[site_id], data.site_xmat[site_id])
        tcp_offsets.append(np.linalg.inv(world_tool) @ world_tcp)
        # This fixed transform is derived from the two source model definitions,
        # not fitted to the recorded TCP. All 33 sampled configurations check
        # its invariance. Correcting TCP metadata does not move collision geoms.
        corrected_world_tcp = world_tool @ tcp_offsets[0]
        sample = {"kind": "recorded" if index < len(frames) else "hypothetical_fk",
                  "frame_index": int(frames[index]) if index < len(frames) else None,
                  "native_q": joints.tolist(), "urdf_q": (signs * joints).tolist(),
                  "raw_link_errors": {},
                  "raw_urdf_tool_vs_native_tcp": _pose_error(tcp_offsets[-1]),
                  "rigid_tcp_corrected_vs_native_tcp": _pose_error(np.linalg.inv(world_tcp) @ corrected_world_tcp)}
        for actual, planned in link_map:
            world_planned = world_base @ urdf.get_transform(planned, frame_from=args.urdf_base)
            link_errors[actual].append(np.linalg.inv(world_planned)
                                       @ _pose(data.xpos[model.body(actual).id], data.xmat[model.body(actual).id]))
            sample["raw_link_errors"][actual] = _pose_error(link_errors[actual][-1])
        if index < len(frames):
            recorded_tcp = _pose(recorded_positions[index], recorded_rotations[index])
            record_errors.append(np.linalg.inv(recorded_tcp) @ world_tcp)
            corrected_record_errors.append(np.linalg.inv(recorded_tcp) @ corrected_world_tcp)
            sample.update(native_dq=dq_recorded[index].tolist(),
                          urdf_dq=(signs * dq_recorded[index]).tolist(),
                          native_tcp_vs_recorded_tcp=_pose_error(record_errors[-1]),
                          rigid_tcp_corrected_vs_recorded_tcp=_pose_error(corrected_record_errors[-1]))
        sample_errors.append(sample)

        spheres_by_link = {}
        for link, sphere_rows in config["collision_spheres"].items():
            local = np.array([sphere["center"] for sphere in sphere_rows])
            radii = np.array([sphere["radius"] for sphere in sphere_rows]) + config["collision_sphere_buffer"]
            transform = world_base @ urdf.get_transform(link, frame_from=args.urdf_base)
            spheres_by_link[link] = (local @ transform[:3, :3].T + transform[:3, 3], radii)
        all_centers = np.concatenate([values[0] for values in spheres_by_link.values()])
        all_radii = np.concatenate([values[1] for values in spheres_by_link.values()])
        for name, link in geom_map:
            geometry = geometries[name]
            geom_id = geometry["geom_id"]
            vertices = (geometry["vertices"] @ data.geom_xmat[geom_id].reshape(3, 3).T
                        + data.geom_xpos[geom_id])
            own_centers, own_radii = spheres_by_link[link]
            own_gap = (np.linalg.norm(vertices[:, None] - own_centers[None], axis=-1) - own_radii).min(axis=1)
            all_gap = (np.linalg.norm(vertices[:, None] - all_centers[None], axis=-1) - all_radii).min(axis=1)
            record = coverage[name]
            record["maximum_own_link_sphere_gap_m"] = max(record["maximum_own_link_sphere_gap_m"], float(own_gap.max()))
            if float(all_gap.max()) > record["maximum_all_robot_sphere_gap_m"]:
                witness_index = int(all_gap.argmax())
                record.update(maximum_all_robot_sphere_gap_m=float(all_gap[witness_index]),
                              witness_robot_q=joints.tolist(),
                              witness_point_world=vertices[witness_index].tolist())
            record["uncovered_vertex_samples"] += int(np.count_nonzero(all_gap > 1e-9))

    report = {
        "scope": "Robot-only CPU kinematic and sampled collision-geometry audit; no planning or execution certification",
        "inputs": {key: str(getattr(args, key).resolve()) for key in ("scene", "catalog", "episode", "urdf", "robot_config")},
        "reproduce_argv": [sys.executable, *sys.argv],
        "object_state_or_contact_values_read": False,
        "gpu_used": False,
        "robot_model_dimensions": {"nq": model.nq, "nv": model.nv, "nbody": model.nbody, "ngeom": model.ngeom},
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
        "recorded_frame_indices": frames.tolist(),
        "hypothetical_fk_probe_count": len(probes),
        "T_world_base": world_base.tolist(),
        "base_body": args.base_body,
        "urdf_base": args.urdf_base,
        "tcp_site": args.tcp_site,
        "tool_link": args.tool_link,
        "native_body_definitions": {
            name: {"parent": model.body(int(model.body_parentid[model.body(name).id])).name,
                   "position_parent_m": model.body_pos[model.body(name).id].tolist(),
                   "quaternion_parent_wxyz": model.body_quat[model.body(name).id].tolist()}
            for name in sorted({args.root_body, args.base_body, *[row[0] for row in link_map]})},
        "joint_binding": [
            {"recorded_name": row[0], "urdf_name": row[1], "urdf_q_equals_native_q_times": float(sign),
             "qpos_column": catalog_joints[row[0]]["qpos_start"],
             "qvel_column": catalog_joints[row[0]]["qvel_start"],
             "native_range": native_limits[index].tolist(),
             "urdf_range": [urdf.joint_map[row[1]].limit.lower, urdf.joint_map[row[1]].limit.upper],
             "urdf_joint_origin": urdf.joint_map[row[1]].origin.tolist(),
             "urdf_axis": list(urdf.joint_map[row[1]].axis),
             "first_recorded_q": float(q_recorded[0, index]),
             "first_recorded_dq": float(dq_recorded[0, index]),
             "sampled_recorded_q_range": [float(q_recorded[:, index].min()), float(q_recorded[:, index].max())],
             "sampled_recorded_limit_violation": float(max(
                 0., native_limits[index, 0] - q_recorded[:, index].min(),
                 q_recorded[:, index].max() - native_limits[index, 1]))}
            for index, (row, sign) in enumerate(zip(joint_map, signs))],
        "mapped_link_relative_transforms": {
            actual: {"urdf_link": planned, **_error_summary(link_errors[actual])} for actual, planned in link_map},
        "T_urdf_tool_recorded_tcp": _error_summary(tcp_offsets),
        "recorded_tcp_vs_robot_only_mujoco": _error_summary(record_errors),
        "rigid_tcp_corrected_vs_recorded_tcp": _error_summary(corrected_record_errors),
        "samples": sample_errors,
        "rigid_tcp_correction_scope": (
            "Model-derived tool-to-TCP transform, checked over every recorded and hypothetical sample. "
            "It aligns the TCP only; raw hand/finger link errors and sphere undercoverage remain."),
        "urdf_cpu_scope": "yourdfpy evaluates the raw URDF at explicitly mapped q; CUDA FK kernels and config lock enforcement are not tested",
        "measured_joint_limit_handling": "Recorded robot q and dq are retained without clamping, replacing or zeroing soft-limit excursions",
        "raw_planner_lock_joints": config["lock_joints"],
        "sphere_count": len(all_radii),
        "collision_sphere_surface_witnesses": coverage,
        "collision_coverage_certified": False,
        "collision_witness_interpretation": (
            "Positive gap means a native collision-geom vertex is outside every configured active robot sphere. "
            "Mesh vertices and box corners are sampled; nonpositive gaps alone do not establish surface/solid coverage. "
            "Self-collision buffers are excluded because they do not inflate robot-versus-scene geometry."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sample_count": len(samples),
                      "peak_process_rss_mib": report["peak_process_rss_mib"],
                      "tcp_mapping": report["T_urdf_tool_recorded_tcp"],
                      "recorded_tcp_check": report["recorded_tcp_vs_robot_only_mujoco"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "catalog", "episode", "urdf", "robot-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("root-body", "base-body", "urdf-base", "tcp-site", "tool-link"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--joint-map", action="append", required=True, help="NATIVE:URDF:SIGN")
    parser.add_argument("--link-map", action="append", required=True, help="NATIVE_BODY:URDF_LINK")
    parser.add_argument("--geom-map", action="append", required=True, help="NATIVE_COLLISION_GEOM:URDF_LINK")
    parser.add_argument("--sample-count", type=int, default=17)
    run(parser.parse_args())
# END recorded robot binding audit with explicit names and CPU-only evidence.
