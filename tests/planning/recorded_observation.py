"""Adapt a known native robot and synchronized RGB-D to three-state filtering.

The only model admitted to the compiler is the explicitly named robot subtree
of an expanded MuJoCo XML file, plus that subtree's assets. Robot joint feedback
and an explicit root pose determine its state. The observed scene segmentation
and object joint/state arrays are never inputs. ``FixedCamera`` supplies the
same pixel-centre projection and optical-depth convention as observation RGB-D.

``prepare_robot_model`` and ``classify_robot_observation`` run on CPU without
creating a GL renderer or CUDA context. Each native collision component remains
a distinct solid; the helper returns labels and provenance without editing raw
depth. Camera helpers also serve the separate simulation camera adapters.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# This explicit local experiment uses the two existing sibling checkouts. No
# dataset name or dated artifact path is part of the observation interface.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
from manipulation.motion import _pose_array
from manipulation.observations import _intrinsics


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an explicit nonempty string")
    return value


def camera_config_from_calibration(calibration: Mapping[str, Any]) -> dict[str, Any]:
    """Invert FixedCamera.calibration, including its half-pixel convention."""
    width, height = calibration["width"], calibration["height"]
    if (type(width) is not int or type(height) is not int or min(width, height) < 3
            or width * height > 2_000_000):
        raise ValueError("Robot mask rendering requires an explicit bounded image size")
    matrix = _intrinsics(calibration["K"])
    if abs(matrix[0, 1]) + abs(matrix[1, 0]) > 1e-12:
        raise ValueError("FixedCamera requires rectified zero-skew pinhole calibration")
    pose = _pose_array(calibration["T_world_camera_cv"], "current T_world_camera_cv")
    near, far = float(calibration["near_m"]), float(calibration["far_m"])
    if not np.isfinite([near, far]).all() or not 0 < near < far:
        raise ValueError("Current calibration must supply finite ordered clipping distances in metres")
    rotation_gl = pose[:3, :3] @ np.diag([1., -1., -1.])
    quaternion = Rotation.from_matrix(rotation_gl).as_quat()[[3, 0, 1, 2]]
    top = (matrix[1, 2] + 0.5) / matrix[1, 1]
    return {
        "position_world_m": pose[:3, 3].tolist(),
        "quaternion_world_camera_gl_wxyz": quaternion.tolist(),
        "frustum_center_over_near": float(((width - 1) / 2 - matrix[0, 2]) / matrix[0, 0]),
        "frustum_bottom_over_near": float(top - height / matrix[1, 1]),
        "frustum_top_over_near": float(top),
        "frustum_halfwidth_over_near": float(width / (2 * matrix[0, 0])),
    }


def _robot_assets(scene: ET.Element, robot: ET.Element, scene_xml: Path) -> ET.Element:
    """Copy only explicit mesh/material/texture dependencies of robot geoms.

    Expanded XML is required, so class/default resolution cannot import an
    unrelated object's assets or silently change a robot visual. Texture files
    are kept because material transparency is part of rendered visibility.
    """
    source = scene.find("asset")
    catalog = {} if source is None else {(node.tag, node.get("name")): node for node in source}
    requested = {(tag, geom.get(tag)) for geom in robot.iter("geom")
                 for tag in ("mesh", "material", "hfield") if geom.get(tag)}
    for tag, name in tuple(requested):
        if (tag, name) not in catalog:
            raise ValueError(f"Robot geom references an absent {tag}: {name}")
        if tag == "material":
            material = catalog[(tag, name)]
            if len(material):
                raise ValueError("The known robot material must use expanded direct texture attributes")
            if material.get("texture"):
                requested.add(("texture", material.get("texture")))
    compiler = scene.find("compiler")
    mesh_directory = "" if compiler is None else compiler.get("meshdir", "")
    texture_directory = "" if compiler is None else compiler.get("texturedir", "")
    assets = ET.Element("asset")
    for tag, name in sorted(requested):
        if (tag, name) not in catalog:
            raise ValueError(f"Known robot asset is absent: {tag} {name}")
        copied = deepcopy(catalog[(tag, name)])
        # MuJoCo textures may use one file or six cube-face files. Resolve all
        # such filenames before compiling a standalone XML string.
        for attribute, value in tuple(copied.attrib.items()):
            if attribute == "file" or attribute.startswith("file") and tag == "texture":
                path = Path(value)
                if not path.is_absolute():
                    directory = mesh_directory if tag == "mesh" else texture_directory if tag == "texture" else ""
                    path = scene_xml.parent / directory / path
                if not path.is_file():
                    raise ValueError(f"Known robot asset file does not exist: {path}")
                copied.set(attribute, str(path.resolve()))
        assets.append(copied)
    return assets


def prepare_robot_model(*, scene_xml: str | Path, robot_root: str,
                        joint_map: Sequence[tuple[str, str, float]],
                        joint_positions: Mapping[str, float],
                        T_world_robot_root: Any,
                        calibration: Mapping[str, Any]):
    """Compile and position only the known fixed-base robot, without rendering.

    Each mapping row is ``(feedback_name, model_joint_name, sign)`` with sign
    +1 or -1. The supplied mapping must cover every scalar robot joint exactly
    once. Measured positions are neither clipped nor replaced with defaults.
    The explicit base transform is the world pose of ``robot_root`` itself.
    # BEGIN complete native collision assembly for three-state filtering.
    Choose the complete calibrated robot root, including its fixed mounting
    assembly. The caller retains those fixed bodies as explicit static collision
    owners. Native collision solids determine model-relative point ownership;
    visual meshes do not determine removal and no rendering is performed.
    # END self filtering does not certify free or occupied space.
    """
    # BEGIN validate the shared camera contract without creating render state.
    camera_config_from_calibration(calibration)
    # END camera projection remains available to separate simulator adapters.
    scene_xml = Path(scene_xml).resolve()
    robot_root = _text(robot_root, "robot_root")
    base = _pose_array(T_world_robot_root, "T_world_robot_root")
    scene = ET.parse(scene_xml).getroot()
    worldbody = scene.find("worldbody")
    matches = [] if worldbody is None else [node for node in worldbody.iter("body")
                                           if node.get("name") == robot_root]
    if len(matches) != 1:
        raise ValueError("robot_root must identify precisely one known fixed-base robot subtree")
    source = matches[0]
    if scene.find("default") is not None or any(
            node.get("class") or node.get("childclass") or node.tag == "include" for node in source.iter()):
        raise ValueError("Provide expanded MuJoCo XML with resolved robot defaults and assets")
    if any(node.tag == "joint" for node in source):
        raise ValueError("The explicit robot root must be fixed; its pose is supplied separately")
    robot = deepcopy(source)
    for parent in robot.iter():
        for child in list(parent):
            if child.tag not in ("body", "joint", "inertial", "geom"):
                parent.remove(child)
    if len(list(robot.iter("body"))) > 64 or len(list(robot.iter("geom"))) > 256:
        raise ValueError("Known robot exceeds the bounded local observation model")
    for attribute in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
        robot.attrib.pop(attribute, None)
    robot.set("pos", " ".join(map(str, base[:3, 3])))
    robot.set("quat", " ".join(map(str, Rotation.from_matrix(base[:3, :3]).as_quat()[[3, 0, 1, 2]])))

    minimal = ET.Element("mujoco", model="known_robot_observation")
    compiler = deepcopy(scene.find("compiler"))
    if compiler is None:
        compiler = ET.Element("compiler")
    # All copied asset paths are absolute, so source path stripping would
    # incorrectly discard provenance and redirect those files.
    compiler.set("strippath", "false")
    minimal.append(compiler)
    # BEGIN collision-only preparation needs no visual or depth-buffer configuration.
    assets = _robot_assets(scene, robot, scene_xml)
    minimal.append(assets)
    ET.SubElement(minimal, "worldbody").append(robot)
    model = mujoco.MjModel.from_xml_string(ET.tostring(minimal, encoding="unicode"))
    data = mujoco.MjData(model)
    # END no renderer configuration is copied into the standalone robot model.

    rows = tuple(joint_map)
    if not 0 < len(rows) <= 16 or any(len(row) != 3 for row in rows):
        raise ValueError("joint_map must explicitly list one to sixteen scalar joint bindings")
    feedback_names = [row[0] for row in rows]
    model_names = [row[1] for row in rows]
    signs = np.asarray([row[2] for row in rows], dtype=float)
    if (len(set(feedback_names)) != len(rows) or len(set(model_names)) != len(rows)
            or not np.isin(signs, (-1., 1.)).all() or set(joint_positions) != set(feedback_names)):
        raise ValueError("Joint mapping must be one-to-one with explicit signed current feedback")
    for name in feedback_names + model_names:
        _text(name, "joint identity")
    model_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
                         for index in range(model.njnt)]
    if set(model_names) != set(model_joint_names) or model.nq != len(rows):
        raise ValueError("Current feedback must cover precisely all robot scalar joints")
    measured = np.asarray([joint_positions[name] for name in feedback_names], float)
    if measured.shape != (len(rows),) or not np.isfinite(measured).all():
        raise ValueError("Current robot joint positions must be finite scalars")
    for name, value in zip(model_names, signs * measured):
        joint = model.joint(name)
        if model.jnt_type[joint.id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError("The known robot observation model accepts scalar hinge/slide joints only")
        data.qpos[joint.qposadr[0]] = value
    mujoco.mj_kinematics(model, data)
    provenance = {
        "source_scene_xml": str(scene_xml), "robot_root": robot_root,
        "source_scope": "only named known-robot subtree and its explicit asset dependencies",
        "object_state_read": False, "observed_scene_segmentation_read": False,
        "joint_map": [list(row) for row in rows],
        "joint_positions_feedback": {name: float(value) for name, value in zip(feedback_names, measured)},
        "T_world_robot_root": base.tolist(),
        "body_count": model.nbody, "geom_count": model.ngeom,
        "body_names": [node.get("name") for node in robot.iter("body")],
        "cloned_assets": [{"kind": node.tag, "name": node.get("name"), "file": node.get("file")}
                          for node in assets],
        "geometry_selection": "all robot geoms with nonzero contype or conaffinity, independent of visual group",
        "fixed_installation_policy": "caller retains complete known assembly collision geometry as static owners",
    }
    return model, data, provenance


# BEGIN three-state native-collision observation adapter; no depth equality mask.
def classify_robot_observation(*, scene_xml: str | Path, robot_root: str,
                               joint_map: Sequence[tuple[str, str, float]],
                               joint_positions: Mapping[str, float],
                               T_world_robot_root: Any, calibration: Mapping[str, Any],
                               observed_depth: Any, padding_scale: float,
                               padding_offset_m: float, source: str) -> dict[str, Any]:
    """Bind complete measured robot feedback to separate native collision geoms.

    Compiled mesh vertices already contain asset scale and principal-axis
    recentering. Their compiled geom pose is applied once. Scaling each geom
    about its own compiled local origin preserves its pose and never scales a
    link's translation. Fixed installation bodies are included regardless of
    visual group; no object state, observed segmentation or renderer is read.
    """
    from manipulation.robot_self_filter import RobotCollisionShape, classify_depth
    from manipulation.tool_geometry import _collision_geometry

    depth = np.asarray(observed_depth, dtype=float)
    if depth.shape != (calibration["height"], calibration["width"]):
        raise ValueError("Observed metric depth must match the supplied calibrated image dimensions")
    source = _text(source, "self-filter source")
    model, data, provenance = prepare_robot_model(
        scene_xml=scene_xml, robot_root=robot_root, joint_map=joint_map,
        joint_positions=joint_positions, T_world_robot_root=T_world_robot_root,
        calibration=calibration)
    geometries, identities = [], []
    for index in range(model.ngeom):
        if not (model.geom_contype[index] or model.geom_conaffinity[index]):
            continue
        name = model.geom(index).name
        # Native identifiers must remain explicit in diagnostics; inventing
        # names would prevent an inventory check against collision ownership.
        _text(name, "native collision geom name")
        pose = np.eye(4)
        pose[:3, :3] = data.geom_xmat[index].reshape(3, 3)
        pose[:3, 3] = data.geom_xpos[index]
        kind = model.geom_type[index]
        arguments = {}
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            native = _collision_geometry(model, index, "known_robot_self_filter")
            shape_kind, arguments = "convex", {"vertices": native["vertices"]}
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            shape_kind, arguments = "box", {"dimensions": tuple(2 * model.geom_size[index])}
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shape_kind = "cylinder"
            arguments = {"dimensions": (float(model.geom_size[index, 0]),
                                         float(2 * model.geom_size[index, 1]))}
        else:
            raise ValueError(f"Unsupported native self-filter collision geom type for {name}: {kind}")
        geometry_source = f"{Path(scene_xml).resolve()}::geom[{name}]"
        geometries.append(RobotCollisionShape(name, shape_kind, pose, geometry_source, **arguments))
        identities.append({"name": name, "body": model.body(model.geom_bodyid[index]).name,
                           "kind": shape_kind, "source": geometry_source,
                           "contype": int(model.geom_contype[index]),
                           "conaffinity": int(model.geom_conaffinity[index])})
    classified = classify_depth(
        observed_depth=depth, K=calibration["K"],
        T_world_camera=calibration["T_world_camera_cv"], geometries=geometries,
        padding_scale=padding_scale, padding_offset_m=padding_offset_m, source=source)
    classified["provenance"].update(known_robot=provenance,
        collision_geometries=identities, collision_geom_count=len(identities),
        body_names=provenance["body_names"], robot_root=provenance["robot_root"],
        synchronization="caller-supplied complete measured robot feedback and calibrated depth from one snapshot")
    return classified
# END unknown/self labels never create environmental surfaces or free rays.
