"""Render isolated local-region transfer fixtures derived from original.

This experiment module owns scene construction and evaluator annotations only.
It does not import production perception or planning.  ``observation.npz``
contains observed RGB-D geometry and a declared oracle whole-entity mask;
``evaluation.npz`` alone contains support/region annotations and body geometry.
The three furniture types are fixture definitions, never inference branches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "FUNCTO_public"
ORIGINAL = PROJECT / "scene/original"
CABINET_ASSET = ROOT / "LIBERO/libero/libero/assets/articulated_objects/slide_cabinet.xml"
DEFAULT_OUTPUT = ROOT / "sim-version-plans/2026-10-13-local-region-transfer/data"
sys.path[:0] = [str(ROOT), str(PROJECT)]

# This normalized, off-centre rectangle is fixed before any model inference.
# alpha measures left-to-right position; beta measures front-to-back position.
# Its boundaries are fractions of each physical support's own two edge vectors.
REGION_BOUNDS = np.array([[0.28, 0.58], [0.28, 0.48]], dtype=float)
VIEWS = ("view_a", "view_b")
VARIANTS = {
    "source": {"scale": [1., 1., 1.], "translation": [0., 0., 0.], "yaw_deg": 0.},
    "pose": {"scale": [1., 1., 1.], "translation": [0.04, -0.025, 0.], "yaw_deg": 12.},
    "shape": {"scale": [1.18, 0.85, 1.], "translation": [0.04, -0.025, 0.], "yaw_deg": 12.},
}
FIXTURES = {
    "shelf": {"root": "wooden_shelf_1_main", "support_body": "wooden_shelf_1_main",
              "support_geom": "wooden_shelf_1_g0", "z_band": [.2205, .222],
              "joint": None, "opening": 0., "base_yaw_deg": 0.,
              "surface_meaning": "upper exterior surface; does not test internal-tier disambiguation"},
    "drawer": {"root": "white_cabinet_1_main", "support_body": "white_cabinet_1_cabinet_top",
               "support_geom": "white_cabinet_1_g10", "z_band": [.157, .160],
               "joint": "white_cabinet_1_top_level", "opening": -.14, "base_yaw_deg": 0.},
    "cabinet": {"root": "transfer_cabinet_main", "support_body": "transfer_cabinet_object",
                "support_geom": "transfer_cabinet_geom_000", "z_band": None,
                "joint": "transfer_cabinet_slidedoor_joint", "opening": .24, "base_yaw_deg": 30.},
}


def write_json(path: Path, value) -> None:
    """Keep experiment metadata human-readable and reject nonfinite numbers."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def numbers(value) -> str:
    """Serialize XML vectors without precision loss from abbreviated logging."""
    return " ".join(f"{float(x):.17g}" for x in np.asarray(value).ravel())


def absolute_assets(tree: ET.Element, directory: Path) -> None:
    """Resolve existing assets; never copy, modify, or download third-party data."""
    for element in tree.iter():
        if element.get("file") is not None:
            element.set("file", str((directory / element.get("file")).resolve(strict=True)))


def import_cabinet(tree: ET.Element) -> ET.Element:
    """Attach the existing sliding-door cabinet, including its native defaults."""
    asset = ET.parse(CABINET_ASSET).getroot()
    absolute_assets(asset, CABINET_ASSET.parent)
    # Prefix every named entity and every reference, so source scene names and
    # imported defaults cannot accidentally resolve to each other's resources.
    mapping = {e.get("name"): "transfer_cabinet_" + e.get("name")
               for e in asset.iter() if e.get("name")}
    classes = {e.get("class"): "transfer_cabinet_" + e.get("class")
               for e in asset.iter("default") if e.get("class")}
    for element in asset.iter():
        for key in ("name", "mesh", "material", "texture", "joint"):
            if element.get(key) in mapping:
                element.set(key, mapping[element.get(key)])
        for key in ("class", "childclass"):
            if element.get(key) in classes:
                element.set(key, classes[element.get(key)])
    tree.find("asset").extend(list(asset.find("asset")))
    defaults = tree.find("default")
    if defaults is None:
        defaults = ET.SubElement(tree, "default")
    defaults.extend(list(asset.find("default")))
    actor = asset.find("worldbody/body")
    actor.set("name", FIXTURES["cabinet"]["root"])
    # Asset floor is at wrapper z=.025; original table is at world z=.900.
    # The cabinet replaces the original drawer location without moving table,
    # robot, cameras, mug, or block.  All scenes remain frozen observations.
    actor.set("pos", "0.0379608 0.281288 0.875")
    for index, geom in enumerate(actor.iter("geom")):
        geom.set("name", f"transfer_cabinet_geom_{index:03d}")
        # The source asset omits type on collision boxes; making its intended
        # geometry explicit prevents MuJoCo's default sphere from changing it.
        if geom.get("class") == "transfer_cabinet_slide_collision" and geom.get("type") is None:
            geom.set("type", "box")
    tree.find("worldbody").append(actor)
    return actor


def primitive_affine_mesh(geom: ET.Element, asset: ET.Element, scale: np.ndarray) -> None:
    """Apply exact affine geometry to boxes and tessellated round primitives.

    An anisotropically scaled rotated box is not generally another box.  Use
    maintained trimesh primitives and transform their vertices, rather than
    silently approximating it by changing three half-sizes.  Cylinder handles
    use 64 sides; contact dynamics are not evaluated in this rendering study.
    """
    import trimesh

    kind = geom.get("type", "sphere")
    size = np.fromstring(geom.get("size", ""), sep=" ")
    if kind == "box":
        mesh = trimesh.creation.box(extents=2 * size)
    elif kind == "cylinder":
        mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=64)
    elif kind == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    else:
        raise ValueError(f"Unsupported fixture primitive {kind}; no shape approximation is permitted")
    if geom.get("quat") is not None:
        q = np.fromstring(geom.get("quat"), sep=" ")
        rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    elif geom.get("euler") is not None:
        rotation = Rotation.from_euler("xyz", np.fromstring(geom.get("euler"), sep=" ")).as_matrix()
    else:
        rotation = np.eye(3)
    vertices = (mesh.vertices @ rotation.T) * scale
    name = geom.get("name") + "_affine_mesh"
    ET.SubElement(asset, "mesh", name=name, vertex=numbers(vertices),
                  face=" ".join(str(int(i)) for i in mesh.faces.ravel()))
    for key in ("size", "quat", "euler"):
        geom.attrib.pop(key, None)
    geom.set("type", "mesh")
    geom.set("mesh", name)


def scale_actor(tree: ET.Element, actor: ET.Element, scale: np.ndarray) -> None:
    """Scale the actor's local geometry before applying its world pose.

    All fixture child bodies have identity orientation. Mesh geoms have no
    separate rotation, so file-vertex scaling is exact. The explicit checks
    reject a future incompatible asset instead of applying a wrong transform.
    """
    if np.array_equal(scale, np.ones(3)):
        return
    assets = tree.find("asset")
    meshes = {m.get("name"): m for m in assets.findall("mesh")}
    scaled_meshes = set()
    for body in actor.iter("body"):
        if body is actor:
            continue
        if body.get("quat") or body.get("euler"):
            raise ValueError("Fixture affine scaling requires unrotated nested body frames")
        if body.get("pos"):
            body.set("pos", numbers(np.fromstring(body.get("pos"), sep=" ") * scale))
    for element in actor.iter():
        if element.tag in ("geom", "site", "joint", "inertial") and element.get("pos"):
            element.set("pos", numbers(np.fromstring(element.get("pos"), sep=" ") * scale))
        if element.tag == "joint" and element.get("type") == "slide":
            axis = np.fromstring(element.get("axis", "0 0 1"), sep=" ")
            factor = float(np.linalg.norm(scale * axis) / np.linalg.norm(axis))
            element.set("axis", numbers(scale * axis))
            if element.get("range"):
                element.set("range", numbers(np.fromstring(element.get("range"), sep=" ") * factor))
        if element.tag != "geom":
            continue
        mesh_name = element.get("mesh")
        if mesh_name:
            if element.get("quat") or element.get("euler"):
                raise ValueError("Fixture mesh rotation must be incorporated before affine scaling")
            if mesh_name not in scaled_meshes:
                mesh = meshes[mesh_name]
                mesh.set("scale", numbers(np.fromstring(mesh.get("scale", "1 1 1"), sep=" ") * scale))
                scaled_meshes.add(mesh_name)
        else:
            primitive_affine_mesh(element, assets, scale)


def build_xml(kind: str, variant: str) -> tuple[ET.Element, dict]:
    """Copy original, enforce one furniture type, then change only that actor."""
    tree = ET.parse(ORIGINAL / "scene.xml").getroot()
    absolute_assets(tree, ORIGINAL)
    world = tree.find("worldbody")
    removed_bodies, removed_geoms = [], []
    for element in list(world):
        if element.tag == "geom" and element.get("mesh", "").startswith("kitchen_background"):
            removed_geoms.append({"mesh": element.get("mesh"), "position": element.get("pos")})
            world.remove(element)
        if element.tag == "body" and element.get("name") in ("wooden_shelf_1_main", "white_cabinet_1_main"):
            if element.get("name") != FIXTURES[kind]["root"]:
                removed_bodies.append(element.get("name"))
                world.remove(element)
    if kind == "cabinet":
        actor = import_cabinet(tree)
    else:
        actor = world.find(f"body[@name='{FIXTURES[kind]['root']}']")
    parameters = VARIANTS[variant]
    scale = np.asarray(parameters["scale"])
    scale_actor(tree, actor, scale)
    position = np.fromstring(actor.get("pos"), sep=" ") + parameters["translation"]
    actor.set("pos", numbers(position))
    q = Rotation.from_euler("z", FIXTURES[kind]["base_yaw_deg"] + parameters["yaw_deg"], degrees=True).as_quat()
    actor.set("quat", numbers(q[[3, 0, 1, 2]]))
    present = [k for k, f in FIXTURES.items() if world.find(f"body[@name='{f['root']}']") is not None]
    if present != [kind]:
        raise AssertionError(f"Furniture exclusivity failed: {present}")
    audit = {"source_scene": str(ORIGINAL / "scene.xml"), "kind": kind, "variant": variant,
             "parameters": parameters, "base_yaw_deg": FIXTURES[kind]["base_yaw_deg"],
             "furniture_types_present": present,
             "removed_bodies": removed_bodies, "removed_background_geoms": removed_geoms,
             "additional_asset": str(CABINET_ASSET) if kind == "cabinet" else None,
             "physics_steps": 0, "robot_and_cameras_unchanged": True,
             "segmentation_input": "oracle rendered whole-entity segmentation; no target local-region label",
             "shape_geometry": "affine mesh vertices; 64-sided cylinder handles; dynamics not evaluated"}
    return tree, audit


def restore_named_state(model, source_model, source_data):
    """Restore surviving joints by name, never reuse a different model's vector."""
    import mujoco

    data = mujoco.MjData(model)
    qwidth = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4,
              int(mujoco.mjtJoint.mjJNT_SLIDE): 1, int(mujoco.mjtJoint.mjJNT_HINGE): 1}
    vwidth = {int(mujoco.mjtJoint.mjJNT_FREE): 6, int(mujoco.mjtJoint.mjJNT_BALL): 3,
              int(mujoco.mjtJoint.mjJNT_SLIDE): 1, int(mujoco.mjtJoint.mjJNT_HINGE): 1}
    for j in range(model.njnt):
        name = model.joint(j).name
        old = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if old < 0:
            continue
        if model.jnt_type[j] != source_model.jnt_type[old]:
            raise ValueError(f"Joint type changed: {name}")
        for values, addresses, width in (("qpos", "jnt_qposadr", qwidth), ("qvel", "jnt_dofadr", vwidth)):
            n = width[int(model.jnt_type[j])]
            new_start, old_start = getattr(model, addresses)[j], getattr(source_model, addresses)[old]
            getattr(data, values)[new_start:new_start+n] = getattr(source_data, values)[old_start:old_start+n]
    for a in range(model.nu):
        old = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.actuator(a).name)
        if old >= 0:
            data.ctrl[a] = source_data.ctrl[old]
    for b in range(model.nbody):
        mocap = model.body_mocapid[b]
        if mocap < 0:
            continue
        old = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, model.body(b).name)
        if old >= 0 and source_model.body_mocapid[old] >= 0:
            source_mocap = source_model.body_mocapid[old]
            data.mocap_pos[mocap] = source_data.mocap_pos[source_mocap]
            data.mocap_quat[mocap] = source_data.mocap_quat[source_mocap]
    data.time = source_data.time
    return data


def annotated_mesh_support(model, data, fixture: dict) -> np.ndarray:
    """Read the two pre-annotated support triangles, exclusively for evaluation.

    Original visual meshes differ slightly from their collision boxes.  The
    z bands in FIXTURES identify the upper exterior shelf or open drawer floor;
    vertices preserve their actual slight tilt. Nothing is fit to a prediction.
    """
    geom = model.geom(fixture["support_geom"])
    mesh_id = int(geom.dataid[0])
    vertices = model.mesh_vert[model.mesh_vertadr[mesh_id]:model.mesh_vertadr[mesh_id]+model.mesh_vertnum[mesh_id]]
    vertices = vertices @ data.geom_xmat[geom.id].reshape(3, 3).T + data.geom_xpos[geom.id]
    body = model.body(fixture["support_body"]).id
    vertices = (vertices - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
    faces = model.mesh_face[model.mesh_faceadr[mesh_id]:model.mesh_faceadr[mesh_id]+model.mesh_facenum[mesh_id]]
    triangles = vertices[faces]
    area_vectors = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    area = np.linalg.norm(area_vectors, axis=1)
    normals = area_vectors / np.maximum(area[:, None], 1e-12)
    z = triangles.mean(axis=1)[:, 2]
    low, high = fixture["z_band"]
    selected = (z > low) & (z < high) & (normals[:, 2] > .99) & (area > 1e-6)
    if selected.sum() != 2:
        raise ValueError(f"Physical support annotation expected two triangles: {fixture}")
    corners = np.unique(np.round(triangles[selected].reshape(-1, 3), 8), axis=0)
    if len(corners) != 4:
        raise ValueError("Annotated support is not the expected physical quadrilateral")
    front = corners[np.argsort(corners[:, 1])[:2]]
    back = corners[np.argsort(corners[:, 1])[2:]]
    origin, right = front[np.argsort(front[:, 0])]
    left_back = back[np.argmin(back[:, 0])]
    # BEGIN independently verify the mesh really is a parallelogram: do not
    # silently replace a future trapezoid's fourth corner by an inferred one.
    right_back = back[np.argmax(back[:, 0])]
    if not np.allclose(right+left_back-origin, right_back, atol=2e-7, rtol=0):
        raise ValueError("Physical support annotation is not a parallelogram")
    # END physical fourth-corner verification.
    return np.stack([origin, right, right+left_back-origin, left_back])


def support_corners(source_model, source_data, kind: str) -> np.ndarray:
    """Provide one fixture annotation in the supporting body's local frame."""
    if kind != "cabinet":
        return annotated_mesh_support(source_model, source_data, FIXTURES[kind])
    # The native cabinet floor box is centred at (-.1125,0,-.09), with half-
    # sizes (.1115,.15,.01). Its left/back walls occlude the outermost strip.
    # The support rectangle covers the actual unobstructed interior top face.
    return np.array([[-.202, -.15, -.08], [-.001, -.15, -.08],
                     [-.001, .126, -.08], [-.202, .126, -.08]])


def evaluate_visible_points(xyz: np.ndarray, corners: np.ndarray) -> dict:
    """Label observed points against a physical support and frozen rectangle.

    The affine two-vector coordinates remain correct even if anisotropic shape
    change makes the mildly tilted original quadrilateral non-orthogonal.
    The 0.4 mm plane tolerance covers rendering quantization but excludes the
    shelf's lower face, about 1.7 mm away. This is an evaluation-only tolerance.
    """
    origin = corners[0]
    edges = np.stack([corners[1]-origin, corners[3]-origin])
    normal = np.cross(*edges)
    normal /= np.linalg.norm(normal)
    uv = (xyz-origin) @ np.linalg.pinv(edges)
    residual = np.abs((xyz-origin) @ normal)
    support = (residual <= .0004) & np.all((uv >= 0) & (uv <= 1), axis=1)
    region = support & np.all((uv >= REGION_BOUNDS[:, 0]) & (uv <= REGION_BOUNDS[:, 1]), axis=1)
    a, b = REGION_BOUNDS
    patch_uv = np.array([[a[0], b[0]], [a[1], b[0]], [a[1], b[1]], [a[0], b[1]]])
    return {"region_mask": region, "support_mask": support,
            "region_center": origin + REGION_BOUNDS.mean(axis=1) @ edges,
            "region_corners": origin + patch_uv @ edges, "support_corners": corners,
            "support_origin": origin, "support_u": edges[0]/np.linalg.norm(edges[0]),
            "support_v": edges[1]/np.linalg.norm(edges[1]), "support_normal": normal,
            "support_width": np.linalg.norm(edges[0]), "support_depth": np.linalg.norm(edges[1]),
            "normalized_region_bounds": REGION_BOUNDS, "point_support_uv": uv,
            "support_plane_tolerance_m": np.array(.0004)}


def descendant_geoms(model, root_name: str) -> np.ndarray:
    """Return a whole-entity segmentation set, without support/patch filtering."""
    root = model.body(root_name).id
    bodies = {root}
    for b in range(root+1, model.nbody):
        if int(model.body_parentid[b]) in bodies:
            bodies.add(b)
    return np.flatnonzero(np.isin(model.geom_bodyid, list(bodies)))


def patch_visibility(evaluation, pixels, depth, K, T) -> dict:
    """Measure area visibility independently of sample density or prediction.

    A uniform 41x41 grid in the physical region estimates visible area. A
    projected sample is visible only if its nearest rendered pixel belongs to
    the annotated support plane and its optical depth agrees within a two-
    pixel footprint. This excludes a shelf underside sharing the same geom ID.
    The centre is checked separately because visible corners do not prove an
    observable centre. These metrics never choose an algorithm candidate.
    """
    bounds = REGION_BOUNDS
    alpha = bounds[0, 0] + (np.arange(41)+.5)/41 * np.ptp(bounds[0])
    beta = bounds[1, 0] + (np.arange(41)+.5)/41 * np.ptp(bounds[1])
    aa, bb = np.meshgrid(alpha, beta)
    uv = np.column_stack([aa.ravel(), bb.ravel()])
    corners = evaluation["support_corners"]
    edges = np.stack([corners[1]-corners[0], corners[3]-corners[0]])
    world = np.vstack([corners[0] + uv @ edges, evaluation["region_center"]])
    camera = (world-T[:3, 3]) @ T[:3, :3]
    xy = camera @ K.T
    xy = np.rint(xy[:, :2]/xy[:, 2, None]).astype(int)
    inside = (camera[:, 2] > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < depth.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < depth.shape[0])
    support_image = np.zeros(depth.shape, bool)
    support_pixels = pixels[evaluation["support_mask"]]
    support_image[support_pixels[:, 1], support_pixels[:, 0]] = True
    visible = np.zeros(len(world), bool)
    x, y = xy[inside].T
    tolerance = 2 * camera[inside, 2] / min(K[0, 0], K[1, 1])
    visible[inside] = support_image[y, x] & (np.abs(depth[y, x]-camera[inside, 2]) <= tolerance)
    area = float(np.linalg.norm(np.cross(*edges)) * np.ptp(bounds[0]) * np.ptp(bounds[1]))
    return {"physical_region_area_m2": area, "region_visible_area_fraction": float(visible[:-1].mean()),
            "estimated_visible_region_area_m2": area * float(visible[:-1].mean()),
            "region_center_visible": bool(visible[-1]), "area_grid_samples": len(world)-1}


def capture_case(kind: str, variant: str, output: Path, source_model, source_data) -> dict:
    """Render both immutable original cameras and save observation/GT separately."""
    import mujoco
    from collect.camera import FixedCamera

    folder = output / kind / variant
    folder.mkdir(parents=True, exist_ok=False)
    tree, audit = build_xml(kind, variant)
    ET.indent(tree)
    ET.ElementTree(tree).write(folder / "scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = restore_named_state(model, source_model, source_data)
    fixture = FIXTURES[kind]
    scale = np.asarray(VARIANTS[variant]["scale"])
    if fixture["joint"] is not None:
        joint = model.joint(fixture["joint"])
        # All imported fixture slides are axis-aligned; preserve normalized
        # opening while scaling. A future oblique-axis asset must supply its
        # original axis instead of passing this restricted fixture calculation.
        if np.count_nonzero(np.abs(model.jnt_axis[joint.id]) > 1e-12) != 1:
            raise ValueError("Fixture opening scaling requires an axis-aligned slide")
        factor = float(np.linalg.norm(scale * model.jnt_axis[joint.id]))
        data.qpos[int(joint.qposadr[0])] = fixture["opening"] * factor
        data.qvel[int(joint.dofadr[0])] = 0.
    mujoco.mj_forward(model, data)
    # Assert preserved robot kinematics, not merely preserved XML spelling.
    for name in ("robot0_base", "robot0_link7", "gripper0_right_rightfinger", "gripper0_right_leftfinger"):
        old = mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_BODY, name)
        new = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if old >= 0:
            assert new >= 0 and np.allclose(data.xpos[new], source_data.xpos[old], atol=1e-12)
            assert np.allclose(data.xmat[new], source_data.xmat[old], atol=1e-12)
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    np.savez_compressed(folder / "state.npz", integration_state=state, state_spec=spec)
    support_body = model.body(fixture["support_body"]).id
    local = support_corners(source_model, source_data, kind) * scale
    corners = local @ data.xmat[support_body].reshape(3, 3).T + data.xpos[support_body]
    geom_ids = descendant_geoms(model, fixture["root"])
    surface_names = sorted(model.geom(int(g)).name for g in geom_ids)
    surface_lookup = np.full(model.ngeom, -1, dtype=np.int32)
    for index, name in enumerate(surface_names):
        surface_lookup[model.geom(name).id] = index
    # BEGIN evaluator-only oracle registration: all opening displacements are
    # scaled with the fixture, so every rigid owner's state obeys this affine
    # map to the unscaled root frame. Neither matrix enters observation.npz.
    actor_id = model.body(fixture["root"]).id
    world_to_canonical = np.eye(4)
    world_to_canonical[:3, :3] = np.diag(1 / scale) @ data.xmat[actor_id].reshape(3, 3).T
    world_to_canonical[:3, 3] = -world_to_canonical[:3, :3] @ data.xpos[actor_id]
    canonical_to_world = np.linalg.inv(world_to_canonical)
    # END evaluator-only oracle registration.
    audit.update(robot_base_position=data.xpos[model.body("robot0_base").id].tolist(),
                 region_bounds=REGION_BOUNDS.tolist(), support_annotation=fixture,
                 support_corners_world=corners.tolist(), views={},
                 world_to_canonical=world_to_canonical.tolist(),
                 canonical_to_world=canonical_to_world.tolist())
    for view in VIEWS:
        view_dir = folder / view
        view_dir.mkdir()
        camera_config = ORIGINAL / view / "camera.json"
        camera = FixedCamera(model, data, 640, 480, camera_name=None, camera_config=camera_config)
        try:
            frame = camera.capture(data)
            calibration, renderer = camera.calibration, camera.renderer_info
        finally:
            camera.close()
        original_calibration = json.loads((ORIGINAL / view / "calibration.json").read_text())
        for field in ("K", "T_world_camera_cv"):
            assert np.allclose(calibration[field], original_calibration[field], atol=1e-9, rtol=0)
        depth = frame["depth_m"]
        mask = np.isin(frame["seg_geom_id"], geom_ids) & np.isfinite(depth) & (depth > 0)
        yy, xx = np.nonzero(mask)
        pixels = np.column_stack([xx, yy]).astype(np.int32)
        K, T = np.asarray(calibration["K"]), np.asarray(calibration["T_world_camera_cv"])
        rays = np.column_stack([xx, yy, np.ones(len(xx))]) @ np.linalg.inv(K).T
        xyz = (rays * depth[yy, xx, None]) @ T[:3, :3].T + T[:3, 3]
        evaluation = evaluate_visible_points(xyz, corners)
        # BEGIN oracle-only surface correspondences, separate from model input.
        evaluation.update(canonical_xyz=xyz @ world_to_canonical[:3, :3].T + world_to_canonical[:3, 3],
                          world_to_canonical=world_to_canonical,
                          canonical_to_world=canonical_to_world,
                          surface_id=surface_lookup[frame["seg_geom_id"][yy, xx]],
                          surface_names=np.asarray(surface_names))
        # END oracle-only surface correspondences.
        visibility = patch_visibility(evaluation, pixels, depth, K, T)
        evaluation.update({k: np.array(v) for k, v in visibility.items()})
        Image.fromarray(frame["rgb"]).save(view_dir / "rgb.png")
        np.save(view_dir / "depth_m.npy", depth, allow_pickle=False)
        np.save(view_dir / "seg_geom_id.npy", frame["seg_geom_id"], allow_pickle=False)
        write_json(view_dir / "calibration.json", calibration)
        write_json(view_dir / "camera.json", json.loads(camera_config.read_text()))
        np.savez_compressed(view_dir / "observation.npz", xyz=xyz, pixels=pixels,
                            object_mask=mask, K=K, T_world_camera_cv=T)
        np.savez_compressed(view_dir / "evaluation.npz", **evaluation)
        overlay = frame["rgb"].copy()
        support_pixels = pixels[evaluation["support_mask"]]
        region_pixels = pixels[evaluation["region_mask"]]
        if len(support_pixels):
            ux, vy = support_pixels.T
            overlay[vy, ux] = (.45 * overlay[vy, ux] + .55 * np.array([40, 180, 255])).astype(np.uint8)
        if len(region_pixels):
            ux, vy = region_pixels.T
            overlay[vy, ux] = (.25 * overlay[vy, ux] + .75 * np.array([255, 60, 60])).astype(np.uint8)
        Image.fromarray(overlay).save(view_dir / "evaluation-overlay.png")
        audit["views"][view] = {"camera_source": str(camera_config), "calibration_unchanged": True,
            "entity_visible_points": len(xyz), "support_visible_points": int(evaluation["support_mask"].sum()),
            "region_visible_points": int(evaluation["region_mask"].sum()), "renderer": renderer, **visibility,
            "visible_region_centroid_error_m": float(np.linalg.norm(xyz[evaluation["region_mask"]].mean(0)-evaluation["region_center"]))
                if evaluation["region_mask"].any() else None}
        if view == "view_a" and evaluation["region_mask"].sum() < 12:
            raise ValueError(f"Observable-state prerequisite failed for {kind}/{variant}/{view}: fewer than 12 region pixels")
    write_json(folder / "audit.json", audit)
    print(json.dumps({"kind": kind, "variant": variant,
                      "visible_region_points": {v: audit["views"][v]["region_visible_points"] for v in VIEWS}}), flush=True)
    return audit


def generate(output: Path) -> None:
    """Create nine fresh cases; never overwrite prior observation evidence."""
    from scene import load_scene

    source_model, source_data = load_scene("original")
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for kind in FIXTURES:
        for variant in VARIANTS:
            cases.append(capture_case(kind, variant, output, source_model, source_data))
    overview = Image.new("RGB", (3*640, 3*515), "white")
    for row, kind in enumerate(FIXTURES):
        for col, variant in enumerate(VARIANTS):
            overlay = Image.open(output / kind / variant / "view_a/evaluation-overlay.png")
            overview.paste(overlay, (col*640, row*515+35))
            ImageDraw.Draw(overview).text((col*640+10, row*515+10), f"{kind} / {variant} / blue=support, red=region", fill="black")
    overview.save(output / "overview.png")
    write_json(output / "manifest.json", {"original_scene": str(ORIGINAL / "scene.xml"),
        "region_rule": "fixed normalized rectangle on independently annotated physical support",
        "normalized_region_bounds": REGION_BOUNDS.tolist(), "whole_entity_masks": "oracle",
        "inference_files": ["rgb.png", "depth_m.npy", "calibration.json", "observation.npz"],
        "evaluation_only_files": ["evaluation.npz", "seg_geom_id.npy", "scene.xml", "state.npz", "audit.json"],
        "pairs": [{"id": f"{kind}_{variant}", "group": kind, "variation": variant,
                   "source": f"{kind}/source/view_a", "target": f"{kind}/{variant}/view_a"}
                  for kind in FIXTURES for variant in ("pose", "shape")],
        "cases": cases})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    generate(parser.parse_args().out.resolve())
