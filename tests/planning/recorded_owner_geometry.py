"""Bind a declared offline box body to a current planning carrier.

This test helper handles the existing box-only source bodies. It uses their
body-local dimensions and explicit current carrier mapping, never recorded
world poses or runtime object state. cuRobo owns collision queries; a bounded
analytic box partition supplies the moving owner's conservative sphere cover.
Observation masks, occupied-space evidence and material-contact permissions
remain independent inputs to the caller's current scene.
"""

# BEGIN explicit offline box-body geometry and current attachment binding.
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.motion import _pose_array


def load_catalog_body_boxes(catalog_path, *, body_id):
    """Read every enabled box of one explicit leaf rigid body from its catalog.

    The supported source bodies have no child bodies. Refusing that broader
    case keeps a partial subtree from being labelled a complete moving owner.
    Fixed-parent/world positions, joints and current simulator data are not
    used; the caller separately supplies the current source-to-owner mapping.
    """
    catalog_path = Path(catalog_path)
    if catalog_path.stat().st_size > 16 * 1024**2:
        raise ValueError("The offline catalog exceeds the bounded metadata input size")
    catalog = json.loads(catalog_path.read_text())
    body = next(item for item in catalog["bodies"] if item["id"] == body_id)
    if any(item["parent_id"] == body_id for item in catalog["bodies"]):
        raise ValueError("This explicit box helper requires a leaf rigid source body")
    geoms = [item for item in catalog["geoms"] if item["body_id"] == body_id
             and (item["contype"] or item["conaffinity"])]
    if not geoms or len(geoms) > 128 or any(item["type"] != "mjGEOM_BOX" for item in geoms):
        raise ValueError("The complete declared rigid body must consist of 1 to 128 native collision boxes")
    boxes = [{"name": item["name"], "source_geom_id": item["id"],
              "position": deepcopy(item["pos_body"]),
              "quaternion_wxyz": deepcopy(item["quat_body_wxyz"]),
              "halfsize": deepcopy(item["size"])} for item in geoms]
    source = {"catalog_path": str(catalog_path.resolve()), "source_body_id": body_id,
              "source_body_name": body["name"], "enabled_geom_ids": [item["id"] for item in geoms],
              "geometry_complete": True,
              "scope": "all enabled native collision boxes of the declared offline leaf rigid body",
              "fields_used": ["geom.body_id", "geom.contype", "geom.conaffinity", "geom.type",
                              "geom.pos_body", "geom.quat_body_wxyz", "geom.size"],
              "source_world_poses_used": False, "current_object_state_read": False}
    return boxes, source


def _box_pose(box):
    position, quaternion = np.asarray(box["position"], float), np.asarray(box["quaternion_wxyz"], float)
    halfsize = np.asarray(box["halfsize"], float)
    if (position.shape != (3,) or quaternion.shape != (4,) or halfsize.shape != (3,)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or not np.isfinite(halfsize).all() or np.any(halfsize <= 0)
            or not np.isclose(np.linalg.norm(quaternion), 1., atol=1e-10, rtol=0)):
        raise ValueError("Every native box needs finite local XYZ, unit wxyz, and positive half sizes")
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    pose[:3, 3] = position
    return pose, halfsize


def _problem_binding(problem, owner_id):
    """Copy the exact current owner pose and deduplicated held spans.

    The inverse attachment maps owner-local points into the tool frame.
    Release IDs and progress come from the current compiler's held metadata,
    never a source frame guessed to be the executable release time.
    """
    phases = [phase for phase in problem.phases if phase.metadata.get("carrier_id") == owner_id]
    if not phases:
        raise ValueError("The explicit geometry owner is absent from the current planning problem")
    initial = _pose_array(phases[0].metadata["owner_initial_pose"], "current initial owner pose")
    attachments = []
    for phase in phases:
        metadata = phase.metadata
        if not np.allclose(initial, metadata["owner_initial_pose"], atol=1e-10, rtol=0):
            raise ValueError("A carrier's initial collision frame must remain the same across its phases")
        if metadata.get("mode") != "held":
            continue
        inverse = np.linalg.inv(_pose_array(metadata["attachment"], "compiled owner-from-tool attachment"))
        relation = {"interval": list(metadata["attachment_interval"]),
                    "position_tool_owner": inverse[:3, 3].tolist(),
                    "quaternion_tool_owner": Rotation.from_matrix(inverse[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist(),
                    "condition_id": metadata["attachment_condition_id"],
                    "release_condition_id": metadata["release_condition_id"]}
        same_span = [item for item in attachments if item["interval"] == relation["interval"]]
        if same_span:
            if same_span[0] != relation:
                raise ValueError("Repeated held phases disagree on their collision attachment")
        else:
            attachments.append(relation)
    return initial, attachments


def prepare_box_owner(problem, owner_id, *, boxes, T_owner_source, geometry_source,
                      mapping_source, cell_edge_m, max_spheres):
    """Prepare serializable local cuboids, whole-volume spheres and attachments.

    ``boxes`` is the complete explicitly declared source body, not sparse
    observed points. A Cartesian partition covers each box's entire volume.
    Each serialized float32 ball encloses every corner of its assigned cell;
    convexity then covers the whole cell, including overlapping source boxes.
    No cup hull or inferred drawer cavity replaces this exact box union.
    """
    mapping = _pose_array(T_owner_source, "explicit current-owner-from-source-body")
    if (not mapping_source or not isinstance(geometry_source, dict)
            or geometry_source.get("geometry_complete") is not True
            or not boxes or len(boxes) > 128
            or {box["source_geom_id"] for box in boxes} != set(geometry_source["enabled_geom_ids"])):
        raise ValueError("A complete owner requires every declared source collision box and explicit mapping evidence")
    if (not np.isfinite(cell_edge_m) or cell_edge_m <= 0
            or type(max_spheres) is not int or not 1 <= max_spheres <= 2048):
        raise ValueError("Box coverage needs an explicit positive cell edge and at most 2048 spheres")
    names = [box["name"] for box in boxes]
    if len(set(names)) != len(names):
        raise ValueError("Source collision boxes require unique names")
    initial, attachments = _problem_binding(problem, owner_id)
    parsed = [(_box_pose(box), box) for box in boxes]
    bins = [np.maximum(1, np.ceil(2*size/cell_edge_m).astype(int)) for ((_, size), _) in parsed]
    if sum(int(np.prod(count)) for count in bins) > max_spheres:
        raise ValueError("The explicit box partition exceeds the supplied moving-owner sphere budget")
    corners = np.asarray(list(product((-.5, .5), repeat=3)))
    cuboids, spheres, proof = [], [], []
    for ((source_pose, halfsize), box), count in zip(parsed, bins):
        current_pose = mapping @ source_pose
        cuboids.append({"name": box["name"], "pose": [*current_pose[:3, 3],
                        *Rotation.from_matrix(current_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]],
                        "dims": (2*halfsize).tolist()})
        step, begin, margins = 2*halfsize/count, len(spheres), []
        for index in product(*(range(int(value)) for value in count)):
            center_box = -halfsize+(np.asarray(index)+.5)*step
            vertices = (center_box+corners*step) @ current_pose[:3, :3].T+current_pose[:3, 3]
            center = (center_box @ current_pose[:3, :3].T+current_pose[:3, 3]).astype(np.float32).astype(float)
            maximum = float(np.linalg.norm(vertices-center, axis=1).max())
            radius = float(np.nextafter(np.float32(maximum+1e-6), np.float32(np.inf)))
            spheres.append([*center, radius])
            margins.append(radius-maximum)
        proof.append({"source_geom_id": box["source_geom_id"], "box_name": box["name"],
                      "subdivisions": count.tolist(), "sphere_range": [begin, len(spheres)],
                      "minimum_cell_corner_enclosure_margin_m": min(margins)})
    return {"owner_id": owner_id, "cuboids": cuboids, "local_spheres": spheres,
            "initial_pose": initial.tolist(), "attachments": attachments,
            "geometry_complete": True, "sphere_cover_error_m": 0.,
            "sphere_materials": [None]*len(spheres),
            "source": (f"Offline native box body {geometry_source['source_body_name']} from "
                       f"{geometry_source['catalog_path']}; explicit current carrier mapping and compiled attachment"),
            "provenance": {"geometry_source": deepcopy(geometry_source),
                           "T_owner_source": mapping.tolist(), "mapping_source": deepcopy(mapping_source),
                           "cell_edge_m": float(cell_edge_m), "max_spheres": max_spheres,
                           "box_count": len(boxes), "sphere_count": len(spheres), "coverage_cells": proof,
                           "coverage_proof": "Every complete native box is tiled by Cartesian cells; each serialized ball encloses all eight vertices of its convex cell",
                           "occupied_space_evidence_supplied": False,
                           "contact_permissions_supplied": False,
                           "current_object_state_read": False}}


def create_box_owner(prepared, *, device_cfg):
    """Allocate the existing native cuboid query only in the caller's GPU stage."""
    # BEGIN preserve explicit completeness at the serialized input boundary.
    # Loading an external JSON cannot upgrade a partial or undeclared body.
    # The preparation proof must still identify every native source box.
    source = prepared.get("provenance", {}).get("geometry_source", {})
    if (prepared.get("geometry_complete") is not True
            or source.get("geometry_complete") is not True):
        raise ValueError("An owner query requires explicitly complete prepared and source geometry")
    # END preserve explicit completeness at the serialized input boundary.
    from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
    from curobo._src.geom.types import Cuboid, SceneCfg
    from manipulation.planning_problem import PosePath
    from manipulation.planning_scene import Attachment, CollisionOwner, CuroboSphereQuery

    cuboids = [Cuboid(**box) for box in prepared["cuboids"]]
    native = SceneCollision.from_config(SceneCollisionCfg(
        device_cfg=device_cfg, scene_model=SceneCfg(cuboid=cuboids), cache={"cuboid": len(cuboids)}))
    query = CuroboSphereQuery(native, owner_ids=(prepared["owner_id"],), source=prepared["source"])
    initial = np.asarray(prepared["initial_pose"], float)
    return CollisionOwner(
        prepared["owner_id"], query,
        PosePath(np.array([0., 1.]), np.stack((initial, initial)), prepared["source"]),
        prepared["source"], geometry_complete=True,
        local_spheres=prepared["local_spheres"], sphere_materials=tuple(prepared["sphere_materials"]),
        attachments=tuple(Attachment(**relation) for relation in prepared["attachments"]),
        sphere_cover_error_m=prepared["sphere_cover_error_m"])
# END explicit offline box-body geometry and current attachment binding.
