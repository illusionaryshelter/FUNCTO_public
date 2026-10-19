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


# BEGIN bind only a material's existing sphere identity to an interaction pair.
def build_material_contact_rules(material_binding, declarations):
    """Resolve named calibrated materials into small ordinary-contact rules.

    Declarations name a current actor, target and stage. Material selection
    supplies sphere indices; no patch, gap, normal or success proof is input.
    Old precise-contact records fail explicitly rather than silently widening.
    """
    result = []
    fields = {'rule_id', 'actor', 'target_owner', 'interval', 'material_id', 'source'}
    for declaration in declarations:
        if not isinstance(declaration, dict) or set(declaration) != fields:
            raise ValueError('Interaction declarations require rule_id, actor, target_owner, interval, material_id and source')
        actor = declaration['actor']
        materials = (material_binding['materials'] if actor == 'robot' else
                     material_binding.get('owner_materials', {}).get(actor, {}))
        material = materials.get(declaration['material_id'])
        if material is None:
            raise ValueError('Interaction material must belong to its named actor')
        result.append(ContactRule(rule_id=declaration['rule_id'], actor=actor,
            target_owner=declaration['target_owner'], interval=declaration['interval'],
            sphere_indices=tuple(material['sphere_indices']),
            source=f"{declaration['source']}; calibrated material: {material['source']}"))
    return tuple(result)
# END bind only a material's existing sphere identity to an interaction pair.
