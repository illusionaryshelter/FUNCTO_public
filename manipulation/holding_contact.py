"""Internal material contacts of an explicitly declared attachment model."""
# BEGIN a declared attachment carries nominal geometry, not physical Holding proof.
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .planning_scene import _text, _valid_interval


@dataclass(frozen=True)
class HoldingContact:
    """One registered robot body assigned an internal attachment contact.

    The caller declares this internal contact in its attachment model. The
    same nominal attachment places the owner for external collision checks.
    This does not establish physical Holding. Only this exact material
    solid/owner pair is internal; other robot solids and other owners remain
    obstacles. Release ends the relationship. No historical contact normal,
    gap, target patch, arrival pose, or close command appears in this record.
    """
    rule_id: str
    condition_id: str
    target_owner: str
    material_id: str
    robot_body: str
    interval: tuple
    source: str

    def __post_init__(self):
        for name in ('rule_id', 'condition_id', 'target_owner', 'material_id', 'robot_body', 'source'):
            _text(getattr(self, name), name)
        object.__setattr__(self, 'interval', _valid_interval(self.interval))


def bind_holding_contacts(problem, material_binding, declarations, *, source):
    """Bind declared attachment/material associations atomically.

    The caller names the robot's contact materials. Their registered body identities come from the observation partition
    and the model declaration. The condition identifies that same nominal model;
    no measured Holding flag is required or inferred.
    """
    _text(source, 'Holding material source')
    conditions = [row for row in problem.conditions if row.get('kind') == 'declared_attachment']
    ids = [row['condition_id'] for row in conditions]
    if not isinstance(declarations, dict) or len(set(ids)) != len(ids) or set(declarations) != set(ids):
        raise ValueError('Holding contacts must name every declared attachment exactly once')
    materials = material_binding['materials']
    replacements, contacts = {}, []
    for condition in conditions:
        identity, owner = condition['condition_id'], condition['carrier_id']
        selected = declarations[identity]
        if (not isinstance(selected, (list, tuple)) or not selected or len(set(selected)) != len(selected)
                or any(name not in materials for name in selected)):
            raise ValueError('Holding contact requires distinct calibrated robot material IDs')
        spans = {tuple(phase.metadata['attachment_interval']) for phase in problem.phases
                 if phase.metadata.get('carrier_id') == owner
                 and phase.metadata.get('attachment_condition_id') == identity}
        if len(spans) != 1:
            raise ValueError('Holding materials require one current attachment interval')
        interval, = spans
        rule_ids = []
        for name in selected:
            material = materials[name]
            rule_id = f'{identity}:material:{name}'
            contacts.append(HoldingContact(rule_id, identity, owner, name,
                material['robot_body'], interval, source))
            rule_ids.append(rule_id)
        replacements[identity] = {**deepcopy(condition), 'contact_rule_ids': rule_ids}
    problem.conditions = tuple(replacements.get(row.get('condition_id'), row) for row in problem.conditions)
    return tuple(contacts)


def validate_holding_conditions(contacts, owners, conditions):
    """Check model identity, without comparing a goal or a physical object pose."""
    from scipy.spatial.transform import Rotation

    by_rule = {contact.rule_id: contact for contact in contacts}
    consumed = set()
    for condition in conditions:
        if condition.get('kind') != 'declared_attachment':
            continue
        ids, owner = condition.get('contact_rule_ids'), condition.get('carrier_id')
        relations = [relation for row in owners if row.owner_id == owner for relation in row.attachments
                     if relation.condition_id == condition.get('condition_id')]
        if (not isinstance(ids, (tuple, list)) or not ids or len(set(ids)) != len(ids)
                or consumed.intersection(ids) or len(relations) != 1):
            raise ValueError('Declared attachment requires its material contacts and one attachment')
        relation, = relations
        for identity in ids:
            contact = by_rule.get(identity)
            if (contact is None or contact.condition_id != condition['condition_id']
                    or contact.target_owner != owner or contact.interval != relation.interval):
                raise ValueError('Holding contacts must match the current condition, owner and interval')
        actual = np.eye(4)
        actual[:3,3] = relation.position_tool_owner
        actual[:3,:3] = Rotation.from_quat(np.asarray(relation.quaternion_tool_owner)[[1,2,3,0]]).as_matrix()
        declared = np.asarray(condition.get('T_carrier_eef'), dtype=float)
        if (declared.shape != (4,4) or not np.isfinite(declared).all()
                or not np.allclose(np.linalg.inv(declared), actual, atol=1e-7, rtol=0)):
            raise ValueError('Declared condition and collision model must use the same attachment')
        consumed.update(ids)
    if consumed != set(by_rule):
        raise ValueError('Every internal Holding contact must belong to a declared attachment')
# END internal nominal contacts do not prove real Holding or task arrival.
