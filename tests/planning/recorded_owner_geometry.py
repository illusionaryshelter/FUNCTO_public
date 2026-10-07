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
from scipy.optimize import nnls
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


# BEGIN true native polygon/edge features retain their original dimension.
def _box_material_feature(row, pose, halfsize):
    """Build native vertices and supporting normals in the owner frame.

    Face/edge axis and signs use this individual box's local axes. The hint
    already uses owner-local axes, so it must be compared with transformed
    adjacent normals, never with box-local normals. The cone projection only
    chooses a supporting material normal; it never edits functional evidence.
    """
    common = {"material_id", "native_geometry", "feature_kind", "source"}
    kind = row["feature_kind"]
    if kind == "polygon":
        if set(row) != common | {"face_axis", "face_sign"}:
            raise ValueError("A polygon material requires one complete native box face")
        axis, sign = row["face_axis"], row["face_sign"]
        if (type(axis) is not int or axis not in (0, 1, 2)
                or type(sign) is not int or sign not in (-1, 1)):
            raise ValueError("Native face axis and sign must identify one actual box face")
        tangents = [index for index in range(3) if index != axis]
        vertices = np.zeros((4, 3))
        vertices[:, axis] = sign*halfsize[axis]
        vertices[:, tangents] = np.asarray(list(product((-1., 1.), repeat=2)))*halfsize[tangents]
        normal = pose[:3, :3] @ (np.eye(3)[axis]*sign)
        thickness = float(2*halfsize[axis])
        audit = {"native_face_axis": axis, "native_face_sign": sign,
                 "native_thickness_source": "Complete box thickness normal to its selected native face"}
    elif kind == "edge":
        if set(row) != common | {"edge_axis", "edge_signs", "normal_hint"}:
            raise ValueError("An edge material requires its native edge and owner-local normal hint")
        axis, signs = row["edge_axis"], row["edge_signs"]
        if (type(axis) is not int or axis not in (0, 1, 2)
                or not isinstance(signs, (list, tuple)) or len(signs) != 2
                or any(type(sign) is not int or sign not in (-1, 1) for sign in signs)):
            raise ValueError("Native edge axis and the other two axes' signs must identify one actual box edge")
        fixed = [index for index in range(3) if index != axis]
        vertices = np.zeros((2, 3))
        vertices[:, axis] = [-halfsize[axis], halfsize[axis]]
        vertices[:, fixed] = np.asarray(signs)*halfsize[fixed]
        adjacent_box = np.eye(3)[fixed]*np.asarray(signs)[:, None]
        adjacent = adjacent_box @ pose[:3, :3].T
        hint = np.asarray(row["normal_hint"], float)
        if (hint.shape != (3,) or not np.isfinite(hint).all()
                or not np.isclose(np.linalg.norm(hint), 1., atol=1e-10, rtol=0)):
            raise ValueError("An edge normal hint must be an explicit owner-local unit vector")
        coefficients, residual = nnls(adjacent.T, hint)
        projected = adjacent.T @ coefficients
        magnitude = float(np.linalg.norm(projected))
        if magnitude <= np.finfo(float).eps:
            raise ValueError("The edge normal hint has no nonzero support in its native outward normal cone")
        normal = projected/magnitude
        angle = float(np.arctan2(np.linalg.norm(np.cross(hint, normal)), hint @ normal))
        # An edge has no fabricated thickness. The unchanged whole solid's
        # thinner incident face provides the existing allowance safety scale.
        thickness = float(2*np.min(halfsize[fixed]))
        audit = {"native_edge_axis": axis, "native_fixed_axes": fixed,
            "native_edge_signs": list(signs), "normal_hint_owner": hint.tolist(),
            "adjacent_normals_owner": adjacent.tolist(),
            "normal_cone_nonnegative_coefficients": coefficients.tolist(),
            "normal_cone_projection_residual": float(residual),
            "normal_cone_projection_angle_rad": angle,
            "normal_hint_endpoint_height_difference_m": float(abs(
                ((vertices[1]-vertices[0]) @ pose[:3, :3].T) @ hint)),
            "native_thickness_source": "Minimum full box thickness along the two incident face normals; not an edge thickness"}
    else:
        raise ValueError("Native box materials support only polygon and edge features")
    points = vertices @ pose[:3, :3].T+pose[:3, 3]
    recovered = (points-pose[:3, 3]) @ pose[:3, :3]
    audit.update(feature_frame="owner_local", native_vertex_frame="box_local",
        native_vertices_box=vertices.tolist(), T_owner_box=pose.tolist(),
        native_vertex_roundtrip_max_error_m=float(np.linalg.norm(recovered-vertices, axis=1).max()),
        support_plane_max_residual_m=float(np.abs((points-points[0]) @ normal).max()))
    return points, normal, thickness, audit
# END edge endpoints are never padded into a planar area or assigned a width.


# BEGIN native material identities remain separate from finite target patches.
def bind_box_owner_materials(prepared):
    """Bind complete native polygons/edges and whole-box sphere identities.

    Each feature is exactly a complete face or edge of one declared cuboid.
    Functional-region points cannot replace its native vertices. Labels name
    the complete source solid for the independent checker; ContactRule still
    limits permission to the true feature/target-patch intersection.
    """
    source = prepared.get("provenance", {}).get("geometry_source", {})
    if (prepared.get("geometry_complete") is not True
            or source.get("geometry_complete") is not True):
        raise ValueError("Owner materials require explicitly complete native geometry")
    owner_id = prepared["owner_id"]
    if not isinstance(owner_id, str) or not owner_id.strip() or owner_id == "robot":
        raise ValueError("Owner material binding requires an explicit nonrobot owner ID")
    boxes = {row["name"]: row for row in prepared["cuboids"]}
    covers = {row["box_name"]: row for row in prepared["provenance"]["coverage_cells"]}
    spheres = np.asarray(prepared["local_spheres"], float)
    if (len(boxes) != len(prepared["cuboids"]) or not boxes or set(boxes) != set(covers)
            or len(covers) != len(prepared["provenance"]["coverage_cells"])
            or {row["source_geom_id"] for row in covers.values()} != set(source["enabled_geom_ids"])
            or spheres.ndim != 2 or spheres.shape[1] != 4 or not len(spheres)
            or not np.isfinite(spheres).all() or np.any(spheres[:, 3] <= 0)):
        raise ValueError("Native box materials need every complete source box and sphere cover")
    groups, owned = {}, []
    for name, cover in covers.items():
        span = cover["sphere_range"]
        if (len(span) != 2 or any(type(value) is not int for value in span)
                or not 0 <= span[0] < span[1] <= len(spheres)):
            raise ValueError("Native box material sphere ranges must be explicit and bounded")
        groups[name] = list(range(*span))
        owned.extend(groups[name])
    if sorted(owned) != list(range(len(spheres))):
        raise ValueError("Native source box ranges must partition the complete owner sphere cover")
    materials, native, labels = {}, {}, [None]*len(spheres)
    for row in prepared.get("material_features", []):
        identity, name = row["material_id"], row["native_geometry"]
        if (not isinstance(identity, str) or not identity.strip() or identity in materials
                or name not in boxes or owner_id+"/"+name in native
                or not isinstance(row["source"], str) or not row["source"].strip()):
            raise ValueError("Owner materials need distinct identities and explicit native box features")
        box = boxes[name]
        pose, halfsize = _box_pose({"position": box["pose"][:3],
            "quaternion_wxyz": box["pose"][3:], "halfsize": np.asarray(box["dims"])/2})
        points, normal, thickness, audit = _box_material_feature(row, pose, halfsize)
        selected = groups[name]
        balls = spheres[selected]
        materials[identity] = {"material_id": identity, "native_geometry": name,
            "feature_kind": row["feature_kind"],
            "feature_points": points.tolist(), "feature_normal": normal.tolist(),
            "feature_joint_translation": None, "sphere_indices": selected,
            "native_thickness_m": thickness, "feature_geometry_audit": audit,
            "normal_cover_excess_m": float(np.max((balls[:, :3]-points[0]) @ normal+balls[:, 3])),
            "source": row["source"]}
        native[owner_id+"/"+name] = identity
        for index in selected:
            labels[index] = identity
    return {"owner_id": owner_id, "materials": materials, "sphere_materials": labels,
            "native_geometry_materials": native}
# END a face identity grants no permission by itself.


def create_box_owner(prepared, *, contact_voxel_size_m):
    """Bind declared complete cuboids to the shared finite online model."""
    # BEGIN preserve explicit completeness at the serialized input boundary.
    # Loading an external JSON cannot upgrade a partial or undeclared body.
    # The preparation proof must still identify every native source box.
    source = prepared.get("provenance", {}).get("geometry_source", {})
    if (prepared.get("geometry_complete") is not True
            or source.get("geometry_complete") is not True):
        raise ValueError("An owner query requires explicitly complete prepared and source geometry")
    # END preserve explicit completeness at the serialized input boundary.
    # BEGIN complete object boxes use the same CPU/Torch online query.
    from manipulation.online_geometry import FrozenPrimitives
    from manipulation.planning_problem import PosePath
    from manipulation.planning_scene import Attachment, CollisionOwner

    query = FrozenPrimitives(primitives=[{"kind": "cuboid", "dimensions": row["dims"],
        "pose": row["pose"]} for row in prepared["cuboids"]],
        owner_ids=(prepared["owner_id"],), source=prepared["source"],
        contact_voxel_size_m=contact_voxel_size_m)
    # END no source solid or separate device collision store is reconstructed.
    initial = np.asarray(prepared["initial_pose"], float)
    return CollisionOwner(
        prepared["owner_id"], query,
        PosePath(np.array([0., 1.]), np.stack((initial, initial)), prepared["source"]),
        prepared["source"], geometry_complete=True,
        local_spheres=prepared["local_spheres"], sphere_materials=tuple(prepared["sphere_materials"]),
        attachments=tuple(Attachment(**relation) for relation in prepared["attachments"]),
        sphere_cover_error_m=prepared["sphere_cover_error_m"])
# END explicit offline box-body geometry and current attachment binding.
