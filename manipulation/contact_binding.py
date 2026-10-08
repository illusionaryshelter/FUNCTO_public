"""Bind named robot faces or native rigid-owner features to finite permissions.

Robot calibration uses only an isolated model and named robot feedback. Rigid
owners provide complete native polygons or edges in their own frames.
A feature declaration identifies material; it does not itself permit contact.
"""

# BEGIN explicit native material calibration and permission construction.
from __future__ import annotations

from itertools import product
from copy import deepcopy
from dataclasses import asdict

import mujoco
import numpy as np

from .planning_scene import ContactRule


# BEGIN bind explicit finite material identities to existing grasp obligations.
def bind_grasp_contact_rules(problem, contacts, activations):
    """Attach declared rule IDs atomically, preserving every goal and event.

    ``activations`` maps each compiled grasp condition ID to its finite rule
    IDs. Neither owner coincidence nor matching nominal time establishes this
    association implicitly. Scene validation subsequently checks the actual
    pose event and close occurrence against the bound robot/owner models.
    """
    from .planning_scene import grasp_contact_rule_ids

    conditions = [row for row in problem.conditions
                  if row.get("kind") == "grasp_contact_activation"]
    identities = [row["condition_id"] for row in conditions]
    if (not isinstance(activations, dict) or len(set(identities)) != len(identities)
            or set(activations) != set(identities)):
        raise ValueError("Finite grasp bindings must name every compiled grasp condition exactly once")
    records = [asdict(rule) for rule in contacts]
    replacements, used = {}, set()
    for condition in conditions:
        row = deepcopy(condition)
        row["contact_rule_ids"] = deepcopy(activations[row["condition_id"]])
        selected = grasp_contact_rule_ids(row, records)
        if used.intersection(selected):
            raise ValueError("A finite grasp rule cannot activate through multiple conditions")
        used.update(selected)
        replacements[row["condition_id"]] = row
    # Validate the complete association before publishing any mutation. Other
    # condition records, actual q/dq, reference motion and event IDs are intact.
    problem.conditions = tuple(replacements.get(row.get("condition_id"), row)
                               for row in problem.conditions)
# END a finite binding never creates whole-hand permissions or new evidence.


# BEGIN current Holding selects finite rules by explicit attachment identity.
def bind_holding_contact_rules(problem, contacts, activations):
    """Bind permissions to existing measured attachment obligations.

    This association does not create Holding evidence. Scene binding checks
    the attachment relation and intervals; execution requires current verified
    evidence before activating these material IDs in a new motion request.
    """
    conditions = [row for row in problem.conditions if row.get("kind") == "observed_attachment"]
    identities = [row["condition_id"] for row in conditions]
    if (not isinstance(activations, dict) or len(set(identities)) != len(identities)
            or set(activations) != set(identities)):
        raise ValueError("Finite Holding bindings must name every current attachment exactly once")
    rules = {rule.rule_id: rule for rule in contacts}
    replacements = {}
    for condition in conditions:
        selected = activations[condition["condition_id"]]
        if (not isinstance(selected, (list, tuple)) or not selected or len(set(selected)) != len(selected)
                or any(identity not in rules or rules[identity].actor != "robot"
                       or rules[identity].target_owner != condition["carrier_id"] for identity in selected)):
            raise ValueError("Holding permission requires distinct finite robot material rules for its carrier")
        replacements[condition["condition_id"]] = {**deepcopy(condition), "contact_rule_ids": list(selected)}
    problem.conditions = tuple(replacements.get(row.get("condition_id"), row) for row in problem.conditions)
# END binding identities never copy a previous close activation latch.


# BEGIN retain finite grasp patches through an explicit rigid owner-frame handoff.
def rebind_grasp_contacts_for_holding(previous_bindings, *, grasp_condition_id,
        old_owner_id, new_owner_id, T_new_owner_old_owner, problem, source):
    """Return declarations for a current attachment, without copying its evidence.

    Only rules named by the selected previous grasp activation are transferred.
    Robot material faces remain robot-local; target patch origins and axes are
    rigidly mapped into the new owner's frame. The new compiled attachment's
    phase interval determines permission lifetime. Callers still acquire the
    current Holding evidence through the ordinary execution contract.
    """
    from .motion import _pose_array

    if any(not isinstance(value, str) or not value.strip()
           for value in (grasp_condition_id, old_owner_id, new_owner_id, source)):
        raise ValueError("Finite Holding handoff requires explicit identities and transform provenance")
    transform = _pose_array(T_new_owner_old_owner, "new-owner from old-owner contact mapping")
    rotation, translation = transform[:3, :3], transform[:3, 3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0)
            or abs(np.linalg.det(rotation)-1.) > 1e-12):
        raise ValueError("Finite contact handoff must preserve a rigid owner-frame mapping")
    if not isinstance(previous_bindings, dict):
        raise ValueError("Finite Holding handoff requires the previous declared grasp bindings")
    selected = previous_bindings.get("grasp_activations", {}).get(grasp_condition_id)
    if (not isinstance(selected, (list, tuple)) or not selected
            or any(not isinstance(name, str) or not name for name in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Finite Holding handoff must select one explicit prior grasp activation")
    declarations = previous_bindings.get("contacts", ())
    identities = [row["rule_id"] for row in declarations]
    if len(set(identities)) != len(identities) or not set(selected) <= set(identities):
        raise ValueError("Selected prior grasp rules must have unique complete contact declarations")
    by_id = dict(zip(identities, declarations))
    conditions = [row for row in problem.conditions
                  if row.get("kind") == "observed_attachment" and row.get("carrier_id") == new_owner_id]
    if len(conditions) != 1:
        raise ValueError("Finite Holding handoff needs exactly one current attachment for the new owner")
    condition_id = conditions[0].get("condition_id")
    if not isinstance(condition_id, str) or not condition_id:
        raise ValueError("Current attachment must carry an explicit condition identity")
    spans = {tuple(phase.metadata["attachment_interval"]) for phase in problem.phases
             if phase.metadata.get("attachment_condition_id") == condition_id
             and phase.metadata.get("carrier_id") == new_owner_id
             and "attachment_interval" in phase.metadata}
    if len(spans) != 1:
        raise ValueError("Current attachment must retain one explicit compiled held interval")
    interval = np.asarray(next(iter(spans)), dtype=float)
    if (interval.shape != (2,) or not np.isfinite(interval).all()
            or interval[0] < 0 or interval[1] <= interval[0]):
        raise ValueError("Current attachment held interval must be finite and ordered")
    contacts, material_ids = [], set()
    for identity in selected:
        row = deepcopy(by_id[identity])
        if row.get("actor", "robot") != "robot" or row.get("target_owner") != old_owner_id:
            raise ValueError("Selected grasp contacts must be finite robot patches for the declared old owner")
        origin, axes = np.asarray(row["patch_origin"], float), np.asarray(row["patch_axes"], float)
        if (origin.shape != (3,) or axes.shape != (3, 3) or not np.isfinite(origin).all()
                or not np.isfinite(axes).all()
                or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-10, rtol=0)
                or abs(np.linalg.det(axes)-1.) > 1e-10):
            raise ValueError("Prior finite contact patch must retain a proper rigid frame")
        row.update(actor="robot", target_owner=new_owner_id, interval=interval.tolist(),
            patch_origin=(rotation @ origin+translation).tolist(),
            patch_axes=(rotation @ axes).tolist(), source=f"{row['source']}; current Holding frame: {source}")
        contacts.append(row)
        material_ids.add(row["material_id"])
    faces = previous_bindings.get("material_faces", ())
    face_ids = [row["material_id"] for row in faces]
    if len(set(face_ids)) != len(face_ids) or not material_ids <= set(face_ids):
        raise ValueError("Transferred finite robot patches require their exact native material face declarations")
    return {"material_faces": [deepcopy(row) for row in faces if row["material_id"] in material_ids],
        "contacts": contacts, "holding_activations": {condition_id: list(selected)}, "source": source}
# END only a new Holding condition activates these unchanged finite permissions.


def calibrate_box_material_features(model, *, tcp_site, joint_positions,
                                    finger_joint_names, material_faces):
    """Return complete box faces and verified affine finger translations.

    ``material_faces`` names a native box, a local face axis and its signed
    outward normal. Every scalar robot coordinate is supplied by name. The
    finite probes span each declared slide joint and check the simultaneous
    midpoint and measured state; no contact-conditioned finger width is
    inferred. The result is expressed at zero finger coordinates in TCP axes.
    """
    names = tuple(finger_joint_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Material calibration needs distinct named finger coordinates")
    native_names = [model.joint(i).name for i in range(model.njnt)]
    if set(joint_positions) != set(native_names):
        raise ValueError("Material calibration needs every isolated robot coordinate by name")
    data = mujoco.MjData(model)
    for name, position in joint_positions.items():
        joint = model.joint(name).id
        if (model.jnt_type[joint] not in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)
                or not np.isfinite(position)):
            raise ValueError("Material calibration supports finite scalar robot feedback")
        data.qpos[model.jnt_qposadr[joint]] = position
    site = model.site(tcp_site).id
    addresses, limits = [], []
    for name in names:
        joint = model.joint(name).id
        bound = np.asarray(model.jnt_range[joint], float)
        if (model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_SLIDE or not model.jnt_limited[joint]
                or not np.isfinite(bound).all() or bound[0] >= bound[1]
                or not bound[0] <= joint_positions[name] <= bound[1]):
            raise ValueError("Material motion needs bounded native slide joints and legal measured q")
        addresses.append(model.jnt_qposadr[joint])
        limits.append(bound)
    limits = np.asarray(limits)
    declarations = list(material_faces)
    ids = [row["material_id"] for row in declarations]
    geometries = [row["native_geometry"] for row in declarations]
    if (not declarations or len(set(ids)) != len(ids) or len(set(geometries)) != len(geometries)
            or any(not isinstance(value, str) or not value for value in ids+geometries)):
        raise ValueError("Material IDs and native geometry names must be explicit and unique")
    local_faces = []
    for row in declarations:
        geom = model.geom(row["native_geometry"]).id
        axis, sign = row["face_axis"], row["face_sign"]
        if (model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX or type(axis) is not int
                or axis not in (0, 1, 2) or sign not in (-1, 1)
                or not isinstance(row["source"], str) or not row["source"]):
            raise ValueError("Each material needs an explicit native box face and evidence source")
        tangents = [index for index in range(3) if index != axis]
        vertices = np.zeros((4, 3))
        vertices[:, axis] = sign*model.geom_size[geom, axis]
        vertices[:, tangents] = np.asarray(list(product((-1., 1.), repeat=2))) * model.geom_size[geom, tangents]
        normal = np.eye(3)[axis]*sign
        local_faces.append((geom, vertices, normal))

    def query(coordinates):
        data.qpos[addresses] = coordinates
        mujoco.mj_kinematics(model, data)
        tool_rotation = data.site_xmat[site].reshape(3, 3)
        tool_position = data.site_xpos[site]
        points, normals = [], []
        for geom, vertices, normal in local_faces:
            rotation = data.geom_xmat[geom].reshape(3, 3)
            world = vertices @ rotation.T + data.geom_xpos[geom]
            points.append((world-tool_position) @ tool_rotation)
            normals.append(tool_rotation.T @ rotation @ normal)
        return np.asarray(points), np.asarray(normals)

    lower, normals = query(limits[:, 0])
    mapping = np.empty((*lower.shape, len(names)))
    for index in range(len(names)):
        probe = limits[:, 0].copy()
        probe[index] = limits[index, 1]
        points, check_normals = query(probe)
        if not np.allclose(check_normals, normals, atol=1e-10, rtol=0):
            raise ValueError("Finger motion rotates the material relative to the supplied TCP")
        mapping[..., index] = (points-lower)/(limits[index, 1]-limits[index, 0])
    zero = lower-np.einsum("mvcj,j->mvc", mapping, limits[:, 0])
    error = 0.
    for probe in (limits.mean(axis=1), np.asarray([joint_positions[name] for name in names]), limits[:, 1]):
        points, check_normals = query(probe)
        error = max(error, float(np.max(np.abs(points-zero-np.einsum("mvcj,j->mvc", mapping, probe)))))
        if error > 1e-9 or not np.allclose(check_normals, normals, atol=1e-10, rtol=0):
            raise ValueError("Named material faces are not affine in the native finger coordinates")
    return {row["material_id"]: {
        "material_id": row["material_id"], "native_geometry": row["native_geometry"],
        # BEGIN robot calibration always derives a full native planar face.
        "feature_kind": "polygon",
        # END callers cannot relabel its dimensionality in a permission.
        "feature_points": zero[index].tolist(), "feature_normal": normals[index].tolist(),
        "feature_joint_translation": mapping[index].tolist(),
        "finger_joint_names": list(names), "native_thickness_m": float(2*model.geom_size[local_faces[index][0], row["face_axis"]]),
        # BEGIN a complete native box is exactly its face extruded inward.
        # This proof concerns the source solid, not its exterior sphere cover.
        # The fixture adapter adds verified runtime sphere identities later.
        "online_material": {"kind": "native_planar_extrusion_v1",
            "thickness_m": float(2*model.geom_size[local_faces[index][0], row["face_axis"]]),
            "native_geometry": row["native_geometry"],
            "source": f"{row['source']}; complete native box face and inward extrusion"},
        # END arbitrary polygons are not silently treated as complete solids.
        "affine_fk_max_error_m": error, "source": row["source"]}
        for index, row in enumerate(declarations)}


def build_material_contact_rules(material_binding, declarations):
    """Combine calibrated material with explicit owner-local permissions.

    The target patch and every physical allowance come from the caller's
    declared observation/model evidence. They are never inferred from the
    robot material, a collision failure, or a desired grasp outcome.
    """
    result = []
    feature_fields = ("sphere_indices", "feature_kind", "feature_points", "feature_normal", "feature_joint_translation")
    for declaration in declarations:
        row = dict(declaration)
        # BEGIN select the actor's exact native material without sharing identities.
        # Rigid-owner features are expressed in that owner's coordinates. Their
        # sphere indices refer to its own complete cover, never robot indices.
        actor = row.setdefault("actor", "robot")
        materials = (material_binding["materials"] if actor == "robot" else
                     material_binding.get("owner_materials", {}).get(actor, {}))
        if row["material_id"] not in materials:
            raise ValueError("Contact material must belong to its explicitly bound actor")
        material = materials[row["material_id"]]
        if any(name in row for name in (*feature_fields, "online_material")):
            raise ValueError("Material geometry must come from its native binding")
        # END caller declarations only set the finite target patch and allowance.
        for name in feature_fields:
            row[name] = material[name]
        row["sphere_indices"] = tuple(row["sphere_indices"])
        # BEGIN bind online native capability only through its actor's source proof.
        if "online_material" in material:
            model = deepcopy(material["online_material"])
            # Robot covers retain their calibrated certificate. Complete rigid
            # owners carry their independently declared Cartesian box-cell
            # proof; neither actor borrows the other's native geometry names.
            sphere_source = (material.get("cover_certificate_path") if actor == "robot"
                             else material.get("sphere_binding_source"))
            if not isinstance(sphere_source, str) or not sphere_source.strip():
                raise ValueError("Online native material requires its verified collision-sphere cover binding")
            model.update(sphere_indices=list(row["sphere_indices"]),
                         sphere_binding_source=sphere_source)
            row["online_material"] = model
        # END source-audit features without capability remain audit-only.
        if row["minimum_thickness_m"] > material["native_thickness_m"]:
            raise ValueError("Contact thin-part scale must also respect the native material thickness")
        row["source"] = f"{row['source']}; native material: {material['source']}"
        result.append(ContactRule(**row))
    return tuple(result)
# END material identity does not grant whole-link or whole-owner exemptions.
