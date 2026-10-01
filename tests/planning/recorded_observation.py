"""Derive visible robot pixels from a known robot and one current RGB-D view.

The only model admitted to the renderer is the explicitly named robot subtree
of an expanded MuJoCo XML file, plus that subtree's assets. Robot joint feedback
and an explicit root pose determine its state. The observed scene segmentation
and object joint/state arrays are never inputs. ``FixedCamera`` supplies the
same pixel-centre projection and optical-depth convention as observation RGB-D.

``prepare_robot_model`` and ``robot_mask_from_depth`` run on CPU. The public
``render_robot_mask`` creates a GL renderer and must be called serially by the
owning runner; importing this module never creates a renderer or a CUDA context.
The helper returns masks and provenance, leaving all observation depth intact.
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
    Choose the modeled arm/hand root, excluding stationary mounts absent from
    the planner's robot collision model. Such mounts remain observed obstacles.
    """
    camera_config = camera_config_from_calibration(calibration)
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
    visual = deepcopy(scene.find("visual"))
    if visual is None:
        visual = ET.Element("visual")
    global_options = visual.find("global")
    if global_options is None:
        global_options = ET.SubElement(visual, "global")
    global_options.set("offwidth", str(calibration["width"]))
    global_options.set("offheight", str(calibration["height"]))
    clipping = visual.find("map")
    if clipping is None:
        clipping = ET.SubElement(visual, "map")
    # Matching both clipping planes also matches the observed depth buffer's
    # precision; the robot-only model's automatically computed extent must not
    # change the camera projection/depth convention.
    clipping.set("znear", str(float(calibration["near_m"])))
    clipping.set("zfar", str(float(calibration["far_m"])))
    minimal.append(visual)
    ET.SubElement(minimal, "statistic", extent="1")
    assets = _robot_assets(scene, robot, scene_xml)
    minimal.append(assets)
    ET.SubElement(minimal, "worldbody").append(robot)
    model = mujoco.MjModel.from_xml_string(ET.tostring(minimal, encoding="unicode"))
    data = mujoco.MjData(model)

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
    mujoco.mj_camlight(model, data)
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
        "render_visibility": "existing FixedCamera groups: visual groups 1 and 2; collision group 0 hidden",
    }
    return model, data, camera_config, provenance


def robot_mask_from_depth(*, depth_m: Any, robot_depth_m: Any,
                          robot_projection: Any, depth_tolerance_m: float,
                          depth_tolerance_source: str) -> dict[str, Any]:
    """Keep only depth-consistent projected pixels; leave all other depth alone.

    FixedCamera samples the same calibrated pixel centres without MSAA. No
    dilation or erosion changes the observed support. A foreground occluder
    more than the declared tolerance ahead of the robot is never removed. Any
    other surface indistinguishable from the robot within that tolerance
    remains a stated identification limit; this is not an occupancy certificate.
    """
    observed, predicted = np.asarray(depth_m, float), np.asarray(robot_depth_m, float)
    projection = np.asarray(robot_projection)
    if (observed.ndim != 2 or observed.shape != predicted.shape or projection.shape != observed.shape
            or projection.dtype != np.bool_ or observed.size > 2_000_000):
        raise ValueError("Observed depth, robot depth, and boolean projection must share a bounded H,W grid")
    tolerance = float(depth_tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Depth tolerance must be explicit finite nonnegative metres")
    source = _text(depth_tolerance_source, "depth tolerance source")
    observed_valid = np.isfinite(observed) & (observed > 0)
    predicted_valid = np.isfinite(predicted) & (predicted > 0)
    if np.any(projection & ~predicted_valid):
        raise ValueError("The robot-only projection contains invalid rendered robot depth")
    comparable = projection & observed_valid
    residual = observed - predicted
    consistent = comparable & (np.abs(residual) <= tolerance)
    mask = consistent
    uncertain = projection & ~mask

    def statistics(values):
        # These diagnostics never alter the caller's fixed metre tolerance.
        if not values.size:
            return {"count": 0, "min": None, "median": None, "p95": None, "p99": None, "max": None}
        quantiles = np.percentile(values, [0, 50, 95, 99, 100])
        return {"count": int(values.size), **dict(zip(
            ("min", "median", "p95", "p99", "max"), map(float, quantiles)))}

    return {
        "robot_mask": mask, "unknown_boundary": uncertain,
        "report": {
            "depth_tolerance_m": tolerance, "depth_tolerance_source": source,
            "robot_projection_pixels": int(projection.sum()),
            "depth_consistent_pixels": int(consistent.sum()),
            "robot_mask_pixels": int(mask.sum()), "unknown_boundary_pixels": int(uncertain.sum()),
            "depth_inconsistent_pixels": int((comparable & ~consistent).sum()),
            "occluding_foreground_pixels": int((comparable & (residual < -tolerance)).sum()),
            "observation_behind_robot_pixels": int((comparable & (residual > tolerance)).sum()),
            "invalid_observed_depth_under_robot_pixels": int((projection & ~observed_valid).sum()),
            "comparable_depth_residual_m": statistics(residual[comparable]),
            "comparable_absolute_depth_residual_m": statistics(np.abs(residual[comparable])),
            "depth_consistent_absolute_residual_max_m": (
                float(np.abs(residual[consistent]).max()) if consistent.any() else None),
            "residual_definition": "observed optical depth minus robot-only optical depth, in metres",
            "mask_policy": "same calibrated pixel centre; no dilation or erosion",
            "input_depth_modified": False,
            "unknown_boundary_scope": "unremoved/unresolved robot identity; retain original depth as obstacle/coverage evidence",
            "limitation": "depth agreement cannot distinguish a coincident foreground surface within the stated tolerance; no occupancy or collision-completeness claim",
        },
    }


def render_robot_mask(*, scene_xml: str | Path, robot_root: str,
                      joint_map: Sequence[tuple[str, str, float]],
                      joint_positions: Mapping[str, float], T_world_robot_root: Any,
                      rgb: Any, depth_m: Any, calibration: Mapping[str, Any],
                      observation_id: str, feedback_source: str,
                      depth_tolerance_m: float, depth_tolerance_source: str) -> dict[str, Any]:
    """Render one synchronized known-robot view; the caller owns GPU scheduling."""
    observation_id = _text(observation_id, "synchronized observation_id")
    feedback_source = _text(feedback_source, "current robot feedback source")
    image, depth = np.asarray(rgb), np.asarray(depth_m, dtype=float)
    shape = (calibration["height"], calibration["width"])
    if image.dtype != np.uint8 or image.shape != (*shape, 3) or depth.shape != shape:
        raise ValueError("Current RGB and metric depth must match their explicit camera calibration")
    # Validate the public tolerance before any GL allocation.
    if not np.isfinite(float(depth_tolerance_m)) or float(depth_tolerance_m) < 0:
        raise ValueError("Depth tolerance must be finite nonnegative metres")
    _text(depth_tolerance_source, "depth tolerance source")
    model, data, config, provenance = prepare_robot_model(
        scene_xml=scene_xml, robot_root=robot_root, joint_map=joint_map,
        joint_positions=joint_positions, T_world_robot_root=T_world_robot_root,
        calibration=calibration)
    sys.path.insert(0, str(WORKSPACE_ROOT))
    from collect.camera import FixedCamera

    camera = FixedCamera(model, data, width=shape[1], height=shape[0],
                         camera_name=None, camera_config=config)
    try:
        rendered_calibration = camera.calibration
        for key in ("K", "T_world_camera_cv"):
            if not np.allclose(rendered_calibration[key], calibration[key], rtol=0, atol=1e-9):
                raise ValueError(f"The robot-only rendered {key} differs from the current camera")
        for key in ("near_m", "far_m"):
            if not np.isclose(rendered_calibration[key], calibration[key], rtol=1e-7, atol=1e-9):
                raise ValueError("Robot-only rendering changed the calibrated clipping planes")
        rendered = camera.capture(data)
    finally:
        camera.close()
    # These IDs come solely from the independently compiled known robot. They
    # contain no observed-scene target labels or object visibility information.
    projection = rendered["seg_geom_id"] >= 0
    result = robot_mask_from_depth(
        depth_m=depth, robot_depth_m=rendered["depth_m"], robot_projection=projection,
        depth_tolerance_m=depth_tolerance_m, depth_tolerance_source=depth_tolerance_source)
    result["robot_depth_m"] = rendered["depth_m"]
    result["report"].update(
        observation_id=observation_id, feedback_source=feedback_source,
        source=provenance, rendered_calibration=rendered_calibration,
        synchronization="caller-supplied robot feedback and RGB-D from one unchanged simulation/sensor snapshot")
    return result
