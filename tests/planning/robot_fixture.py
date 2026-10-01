"""Export an explicitly named native robot into a cuRobo planning fixture.

The input is an expanded, fixed-base MuJoCo subtree with scalar joints and
box/convex-mesh collision solids. Native names, axes, signs and transforms are
retained. ``yourdfpy`` owns URDF serialization and independent FK; no planner
or simulator scene state is admitted to this CPU-only model export.
"""

# BEGIN native robot export and continuous-volume sphere-cover certificates.
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import mujoco
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import trimesh
from yourdfpy import URDF, Robot, Link, Joint, Limit, Inertial, Geometry, Mesh, Box, Collision
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.tool_geometry import _collision_geometry
from check_recorded_robot_binding import _robot_model


def _pose(position, quaternion):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    value[:3, 3] = position
    return value


def _pose_dict(transform):
    return {"position": transform[:3, 3].tolist(), "quaternion_wxyz":
            Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist()}


def _cover_cells(vertices, *, count=None, box_halfsize=None, subdivisions=None):
    """Partition the complete native solid, rather than sampled surfaces.

    Mesh partitions are exact intersections of convex-hull halfspaces with
    Voronoi halfspaces. Sobol points only choose sites; they are not coverage
    evidence. A ball enclosing every vertex encloses its entire convex cell.
    Box cells use an analytic Cartesian partition with the same certificate.
    """
    if box_halfsize is not None:
        bins = np.asarray(subdivisions, int)
        if bins.shape != (3,) or np.any(bins < 1):
            raise ValueError("Box cover requires three positive subdivision counts")
        step = 2 * box_halfsize / bins
        offsets = np.asarray(list(product((-.5, .5), repeat=3))) * step
        return [(-box_halfsize + (np.asarray(index) + .5) * step) + offsets
                for index in product(*(range(value) for value in bins))], None
    hull = ConvexHull(vertices)
    planes = hull.equations
    points = qmc.Sobol(3, scramble=False).random_base2(12)
    points = vertices.min(0) + points * np.ptp(vertices, axis=0)
    points = points[np.all(points @ planes[:, :3].T + planes[:, 3] < -1e-9, axis=1)]
    if count is None or not 1 <= count <= min(128, len(points)):
        raise ValueError("Mesh cover requires a bounded positive cell count")
    sites, _ = kmeans2(points, count, iter=15, minit="++", seed=10)
    cells = []
    for index, site in enumerate(sites):
        others = np.delete(sites, index, axis=0)
        normals = others - site
        offsets = (np.einsum("ij,ij->i", others, others) - site @ site) / -2
        distances = np.linalg.norm(normals, axis=1)
        if np.any(distances < 1e-12):
            raise ValueError("Degenerate Voronoi sites cannot certify a cover")
        cuts = np.column_stack((normals, offsets)) / distances[:, None]
        cells.append(HalfspaceIntersection(np.concatenate((planes, cuts)), site).intersections)
    return cells, sites


def _geom_cover(model, geom_id, specification, destination):
    geometry = _collision_geometry(model, geom_id, "native_collision_proxy")
    local = _pose(model.geom_pos[geom_id], model.geom_quat[geom_id])
    is_box = model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
    cells, sites = _cover_cells(geometry["vertices"],
        count=specification.get("cells"),
        box_halfsize=model.geom_size[geom_id] if is_box else None,
        subdivisions=specification.get("subdivisions"))
    spheres, proof_vertices, offsets, margins = [], [], [0], []
    for cell in cells:
        body_vertices = cell @ local[:3, :3].T + local[:3, 3]
        center, _ = trimesh.nsphere.minimum_nsphere(body_vertices)
        # cuRobo consumes float32. Recompute and round the radius outward AFTER
        # rounding the serialized center, retaining a 1 micrometre allowance.
        center = np.asarray(center, dtype=np.float32).astype(float)
        distances = np.linalg.norm(body_vertices - center, axis=1)
        radius = float(np.nextafter(np.float32(distances.max() + 1e-6), np.float32(np.inf)))
        spheres.append({"center": center.tolist(), "radius": radius})
        proof_vertices.extend(body_vertices)
        offsets.append(len(proof_vertices))
        margins.append(radius - distances.max())
    centers = np.asarray([sphere["center"] for sphere in spheres])
    radii = np.asarray([sphere["radius"] for sphere in spheres])
    geom_centers = (centers - local[:3, 3]) @ local[:3, :3]
    planes = geometry["halfspaces"]
    excess = geom_centers @ planes[:, :3].T + planes[:, 3] + radii[:, None]
    np.savez_compressed(destination, native_vertices=geometry["vertices"],
                        native_faces=geometry["faces"], native_halfspaces=planes,
                        T_body_geom=local, cell_vertices_body=np.asarray(proof_vertices),
                        cell_vertex_offsets=np.asarray(offsets), sphere_centers_body=centers,
                        sphere_radii=radii, cell_enclosure_margin=np.asarray(margins),
                        partition_sites_geom=np.empty((0, 3)) if sites is None else sites)
    record = {"geom": model.geom(geom_id).name, "link": model.body(model.geom_bodyid[geom_id]).name,
              "representation": geometry["collision_representation"], "T_body_geom": local.tolist(),
              "certificate_path": str(destination), "sphere_count": len(spheres),
              "minimum_cell_enclosure_margin_m": float(min(margins)),
              "maximum_support_plane_excess_m": float(excess.max()),
              "source_contype": int(model.geom_contype[geom_id]),
              "source_conaffinity": int(model.geom_conaffinity[geom_id]),
              "scope": "collision proxy only; not RGB-D occupied-solid evidence"}
    return spheres, record, geometry


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


def export_fixture(*, scene_path, catalog_path, episode_path, root_body, visual_root, tcp_site,
                   arm_joint_names, finger_joint_names, velocity_limits, effort_limits,
                   cover_specifications, initial_frame, output, binding_audit_path,
                   max_acceleration, max_jerk):
    """Build one exact native model and its explicitly budgeted collision cover.

    Velocity/effort limits and the initial recorded frame are explicit inputs;
    MJCF has no velocity-limit field. No coordinate is clipped or re-signed.
    Unsupported source features fail before artifacts can be used for planning.
    """
    scene_path, catalog_path, episode_path = [Path(value).resolve() for value in
                                            (scene_path, catalog_path, episode_path)]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = ET.parse(scene_path).getroot()
    if scene.find("default") is not None:
        raise ValueError("Provide expanded native XML without unresolved collision defaults")
    source_visual_root = next((node for node in scene.iter("body") if node.get("name") == visual_root), None)
    if source_visual_root is None:
        raise ValueError("The explicit arm/hand collision and visual root is absent")
    enabled_geoms = {node.get("name") for node in source_visual_root.iter("geom")
                     if int(node.get("contype", "1")) or int(node.get("conaffinity", "1"))}
    if enabled_geoms != set(cover_specifications):
        raise ValueError("Cover specification must include every native enabled geom of the explicit arm/hand root")
    if any(scene.find(tag) is not None and len(scene.find(tag)) for tag in ("contact", "equality", "tendon")):
        raise ValueError("Explicit contact/equality/tendon rules require a matching exporter")
    names = list(arm_joint_names) + list(finger_joint_names)
    if len(set(names)) != len(names) or set(velocity_limits) != set(names) or set(effort_limits) != set(names):
        raise ValueError("Native scalar joints need unique names and explicit velocity/effort limits")
    model = _robot_model(scene_path, root_body, list(cover_specifications))
    if model.nq != len(names) or set(names) != {model.joint(i).name for i in range(model.njnt)}:
        raise ValueError("Named arm/finger joints must cover the robot exactly")
    # Preserve options used by the native collision-pair filter. This model
    # contains the robot only and is never used as a substitute scene rollout.
    robot_mjcf = output / "robot_only.xml"
    mujoco.mj_saveLastXML(str(robot_mjcf), model)
    xml = ET.parse(robot_mjcf)
    if scene.find("option") is not None:
        old = xml.getroot().find("option")
        if old is not None:
            xml.getroot().remove(old)
        from copy import deepcopy
        xml.getroot().append(deepcopy(scene.find("option")))
        xml.write(robot_mjcf, encoding="unicode")
        model = mujoco.MjModel.from_xml_path(str(robot_mjcf))
    catalog = json.loads(catalog_path.read_text())
    catalog_joints = {joint["name"]: joint for joint in catalog["joints"]}
    binding = []
    for name in names:
        row = catalog_joints[name]
        if row["qpos_count"] != 1 or row["qvel_count"] != 1:
            raise ValueError("Every exported recorded joint must be scalar")
        binding.append({"recorded_name": name, "qpos_column": row["qpos_start"],
                        "qvel_column": row["qvel_start"]})
    with h5py.File(episode_path, "r") as stream:
        q0 = np.asarray([stream["states/qpos"][initial_frame, row["qpos_column"]] for row in binding])
        frames = np.unique(np.linspace(0, len(stream["states/qpos"])-1, 17, dtype=int))
        recorded = np.column_stack([stream["states/qpos"][frames, row["qpos_column"]] for row in binding])
    data = mujoco.MjData(model)
    links, joints, covers, records = [], [], {}, []
    mesh_directory = output / "collision_meshes"
    mesh_directory.mkdir(exist_ok=True)
    root_id = model.body(root_body).id
    for body in range(1, model.nbody):
        name = model.body(body).name
        inertial = Inertial(origin=_pose(model.body_ipos[body], model.body_iquat[body]),
                            mass=float(model.body_mass[body]), inertia=np.diag(model.body_inertia[body]))
        collisions = []
        for geom_id in np.flatnonzero(model.geom_bodyid == body):
            geom_name = model.geom(int(geom_id)).name
            spheres, record, geometry = _geom_cover(model, geom_id, cover_specifications[geom_name], output/f"{geom_name}.npz")
            covers.setdefault(name, []).extend(spheres)
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                shape = Geometry(box=Box(size=2 * model.geom_size[geom_id]))
            else:
                mesh_path = mesh_directory / f"{geom_name}.obj"
                trimesh.Trimesh(vertices=geometry["vertices"], faces=geometry["faces"], process=False).convex_hull.export(mesh_path)
                shape = Geometry(mesh=Mesh(filename=str(mesh_path)))
                record["collision_mesh_path"] = str(mesh_path)
            collisions.append(Collision(name=geom_name, origin=_pose(model.geom_pos[geom_id], model.geom_quat[geom_id]), geometry=shape))
            records.append(record)
        links.append(Link(name=name, inertial=inertial, collisions=collisions))
        if body == root_id:
            if model.body_jntnum[body] or model.body_parentid[body] != 0:
                raise ValueError("The exported root must be fixed and directly below world")
            continue
        if model.body_jntnum[body] > 1:
            raise ValueError("This exporter supports at most one scalar joint per native body")
        joint = Joint(name=f"{name}_fixed", type="fixed", parent=model.body(model.body_parentid[body]).name,
                      child=name, origin=_pose(model.body_pos[body], model.body_quat[body]))
        if model.body_jntnum[body]:
            joint_id = int(model.body_jntadr[body])
            native = model.joint(joint_id).name
            if np.any(model.jnt_pos[joint_id] != 0) or model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError("Only origin-centred scalar hinge/slide joints are supported")
            if not model.jnt_limited[joint_id] or min(velocity_limits[native], effort_limits[native]) <= 0:
                raise ValueError("Exported scalar joints require finite explicit native limits")
            joint.name, joint.type = native, "revolute" if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE else "prismatic"
            joint.axis = model.jnt_axis[joint_id].copy()
            joint.limit = Limit(lower=float(model.jnt_range[joint_id, 0]), upper=float(model.jnt_range[joint_id, 1]),
                                velocity=float(velocity_limits[native]), effort=float(effort_limits[native]))
        joints.append(joint)
    site_id = model.site(tcp_site).id
    links.append(Link(name=tcp_site))
    joints.append(Joint(name=f"{tcp_site}_fixed", type="fixed", parent=model.body(model.site_bodyid[site_id]).name,
                        child=tcp_site, origin=_pose(model.site_pos[site_id], model.site_quat[site_id])))
    urdf = URDF(Robot(name="native_recorded_robot", links=links, joints=joints), load_meshes=False)
    urdf_path = output / "robot.urdf"
    urdf.write_xml_file(str(urdf_path))
    # Reloading verifies the serialized RPY representation, not an in-memory
    # transform shared with the exporter. Samples include open/closed limits.
    reference = URDF.load(urdf_path, load_meshes=False)
    limits = np.asarray([model.jnt_range[model.joint(name).id] for name in names])
    if not np.isfinite(q0).all() or np.any(q0 < limits[:, 0]) or np.any(q0 > limits[:, 1]):
        raise ValueError("Initial measured robot feedback is outside native joint limits")
    probes = np.random.default_rng(7341).uniform(limits[:, 0], limits[:, 1], size=(16, len(names)))
    probes[0], probes[1] = limits[:, 0], limits[:, 1]
    max_position, max_rotation = 0., 0.
    for sample in np.concatenate((recorded, probes)):
        for name, value in zip(names, sample):
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_kinematics(model, data)
        base = _pose(data.xpos[root_id], data.xquat[root_id])
        reference.update_cfg(dict(zip(names, sample)))
        for body in range(1, model.nbody):
            predicted = base @ reference.get_transform(model.body(body).name, frame_from=root_body)
            actual = _pose(data.xpos[body], data.xquat[body])
            max_position = max(max_position, float(np.linalg.norm(predicted[:3, 3]-actual[:3, 3])))
            max_rotation = max(max_rotation, float(Rotation.from_matrix(predicted[:3, :3].T @ actual[:3, :3]).magnitude()))
        actual_tcp = np.eye(4)
        actual_tcp[:3, :3], actual_tcp[:3, 3] = data.site_xmat[site_id].reshape(3, 3), data.site_xpos[site_id]
        predicted_tcp = base @ reference.get_transform(tcp_site, frame_from=root_body)
        max_position = max(max_position, float(np.linalg.norm(predicted_tcp[:3, 3]-actual_tcp[:3, 3])))
        max_rotation = max(max_rotation, float(Rotation.from_matrix(predicted_tcp[:3, :3].T @ actual_tcp[:3, :3]).magnitude()))
    if max_position > 1e-9 or max_rotation > 1e-9:
        raise ValueError(f"Serialized URDF differs from native FK: {max_position} m, {max_rotation} rad")
    sphere_count = sum(map(len, covers.values()))
    if sphere_count > 480:
        raise ValueError("Local fixture exceeds the explicitly bounded 480-sphere budget")
    config = {"robot_cfg": {"load_dynamics": False, "kinematics": {
        "format_version": 2.0, "urdf_path": str(urdf_path), "asset_root_path": str(output),
        "base_link": root_body, "tool_frames": [tcp_site], "use_global_cumul": True,
        "collision_link_names": list(covers), "mesh_link_names": list(covers),
        "collision_spheres": covers, "collision_sphere_buffer": 0.,
        "self_collision_buffer": {name: 0. for name in covers},
        "self_collision_ignore": _native_ignored_pairs(model, list(covers)),
        "lock_joints": dict(zip(finger_joint_names, q0[len(arm_joint_names):].tolist())),
        "cspace": {"joint_names": names, "default_joint_position": q0.tolist(),
                   "cspace_distance_weight": [1.] * len(names), "null_space_weight": [1.] * len(names),
                   "max_acceleration": float(max_acceleration), "max_jerk": float(max_jerk)}}}}
    config_path = output / "robot.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    finger_roots = {int(model.jnt_bodyid[model.joint(name).id]) for name in finger_joint_names}
    finger_links = []
    for name in covers:
        ancestor = model.body(name).id
        while ancestor and ancestor not in finger_roots:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor in finger_roots:
            finger_links.append(name)
    visual_root_id = model.body(visual_root).id
    cursor = visual_root_id
    while cursor != root_id:
        if cursor == 0 or model.body_jntnum[cursor]:
            raise ValueError("The explicit visual root must be rigidly fixed below the model base")
        cursor = int(model.body_parentid[cursor])
    visual_transform = np.linalg.inv(base) @ _pose(data.xpos[visual_root_id], data.xquat[visual_root_id])
    manifest = {"source_scene_xml": str(scene_path), "catalog_path": str(catalog_path),
        "episode_path": str(episode_path), "binding_audit_path": str(Path(binding_audit_path).resolve()),
        "robot_mjcf_path": str(robot_mjcf), "urdf_path": str(urdf_path), "robot_config_path": str(config_path),
        "base_link": root_body, "base_pose": _pose_dict(base), "tool_link": tcp_site, "tcp_site": tcp_site,
        "robot_visual_root": visual_root, "T_base_visual_root": visual_transform.tolist(),
        "tool_transform": _pose_dict(np.eye(4)), "joint_names": names,
        "arm_joint_names": list(arm_joint_names), "finger_joint_names": list(finger_joint_names),
        "finger_link_names": finger_links,
        "planning_joint_names": list(arm_joint_names), "joint_binding": binding,
        "initial_frame": int(initial_frame), "initial_joint_positions": q0.tolist(),
        "velocity_limits": velocity_limits, "effort_limits": effort_limits,
        "collision_geometries": records, "sphere_count": sphere_count,
        "sphere_materials": [None] * sphere_count,
        "robot_geometry_source": "compiled native MuJoCo arm/hand collision geoms below the declared robot_visual_root; complete convex-cell enclosing sphere cover; fixed mounts remain observed background",
        "coverage_proof": "All native enabled convex solids below robot_visual_root partitioned into convex cells; each serialized float32 sphere encloses every cell vertex with at least 1e-6 m allowance",
        "coverage_limit": "Conservative collision proxies, not RGB-D occupied-solid evidence; support-plane excess is not a Euclidean Hausdorff bound",
        "fk_audit": {"recorded_frames": frames.tolist(), "hypothetical_probes": len(probes),
                     "maximum_translation_error_m": max_position, "maximum_rotation_error_rad": max_rotation,
                     "consumer": "yourdfpy reloaded serialized URDF; cuRobo native FK is a separate runner check"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Explicit exporter keyword arguments as JSON")
    args = parser.parse_args()
    manifest = export_fixture(**json.loads(args.spec.read_text()))
    print(json.dumps({"manifest": str(Path(json.loads(args.spec.read_text())["output"]).resolve()/"manifest.json"),
                      "sphere_count": manifest["sphere_count"], "fk_audit": manifest["fk_audit"]}, indent=2))


if __name__ == "__main__":
    main()
# END native robot export and continuous-volume sphere-cover certificates.
