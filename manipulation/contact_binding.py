"""Bind preserved task features and named contact rules to current tasks."""
# BEGIN task declarations are independent of collision representation.
from copy import deepcopy
from dataclasses import asdict

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


def build_material_contact_rules(material_binding, declarations):
    """Combine source task features with explicit finite target permissions.

    Robot features name their observed body. Other features use their actor's
    local coordinates. The material record supplies feature geometry; each
    declaration supplies the target, active interval and task allowances.
    """
    feature_fields = ("feature_kind", "feature_points", "feature_normal",
                      "feature_joint_translation")
    result = []
    for declaration in declarations:
        row = deepcopy(declaration)
        actor = row["actor"]
        materials = (material_binding["materials"] if actor == "robot"
                     else material_binding["owner_materials"][actor])
        material = materials[row["material_id"]]
        if set(row).intersection((*feature_fields, "robot_body")):
            raise ValueError("Task feature coordinates and body identity belong to the named material")
        for name in feature_fields:
            row[name] = deepcopy(material[name])
        if actor == "robot":
            row["robot_body"] = material["robot_body"]
        row["source"] = f"{row['source']}; task feature: {material['source']}"
        result.append(ContactRule(**row))
    return tuple(result)
# END source features do not replace the entity's observed collision field.
