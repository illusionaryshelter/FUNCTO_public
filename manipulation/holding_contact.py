"""Internal material contacts of a currently held assembly, not grasp goals."""
# BEGIN Holding carries current geometry and material identity, never an old pose target.
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .planning_scene import _text, _valid_interval


@dataclass(frozen=True)
class HoldingContact:
    """One registered robot material solid touching its attached owner.

    Holding establishes this internal contact. The current attachment places
    the owner for all external collision checks. Only this exact material
    solid/owner pair is internal; other robot solids and other owners remain
    obstacles. Release ends the relationship. No historical contact normal,
    gap, target patch, arrival pose, or close command appears in this record.
    """
    rule_id: str
    condition_id: str
    target_owner: str
    material_id: str
    native_geometry: str
    sphere_indices: tuple
    interval: tuple
    source: str

    def __post_init__(self):
        for name in ('rule_id', 'condition_id', 'target_owner', 'material_id', 'native_geometry', 'source'):
            _text(getattr(self, name), name)
        indices = tuple(self.sphere_indices)
        if (not indices or len(set(indices)) != len(indices)
                or any(type(index) is not int or index < 0 for index in indices)):
            raise ValueError('Holding contact requires explicit unique native material sphere indices')
        object.__setattr__(self, 'sphere_indices', indices)
        object.__setattr__(self, 'interval', _valid_interval(self.interval))


def bind_holding_contacts(problem, material_binding, declarations, *, source):
    """Bind declared current Holding/material associations atomically.

    The caller names the robot's contact materials. Their complete native
    solids and sphere ordering come from robot calibration, not task names or
    a previous grasp pose. The semantic condition remains the only dependency.
    """
    _text(source, 'Holding material source')
    conditions = [row for row in problem.conditions if row.get('kind') == 'observed_attachment']
    ids = [row['condition_id'] for row in conditions]
    if not isinstance(declarations, dict) or len(set(ids)) != len(ids) or set(declarations) != set(ids):
        raise ValueError('Holding contacts must name every current observed attachment exactly once')
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
                material['native_geometry'], tuple(material['sphere_indices']), interval, source))
            rule_ids.append(rule_id)
        replacements[identity] = {**deepcopy(condition), 'contact_rule_ids': rule_ids}
    problem.conditions = tuple(replacements.get(row.get('condition_id'), row) for row in problem.conditions)
    return tuple(contacts)


def validate_holding_materials(contacts, owners, sphere_materials, native_materials, collision_bodies):
    """Reject partial, relabelled or unrelated robot geometry before solving."""
    identities, pairs = set(), set()
    owner_ids = {owner.owner_id for owner in owners}
    geometry_indices, sphere_geometry = {}, {}
    for body in collision_bodies:
        # BEGIN robot body groups use the mapper's geometry_names contract.
        # native_geometry_names belongs to owner aliases, not robot bodies.
        names = tuple(body['geometry_names'])
        # END use the same robot grouping consumed by native confirmation.
        for index in body['sphere_indices']:
            sphere_geometry[index] = names
        for name in names:
            geometry_indices.setdefault(name, set()).update(body['sphere_indices'])
    for contact in contacts:
        if not isinstance(contact, HoldingContact):
            raise ValueError('Holding scene input requires typed current material contacts')
        pair = (contact.native_geometry, contact.target_owner, contact.interval)
        if (contact.rule_id in identities or pair in pairs or contact.target_owner not in owner_ids
                or native_materials.get(contact.native_geometry) != contact.material_id
                or set(contact.sphere_indices) != geometry_indices.get(contact.native_geometry)
                or any(index >= len(sphere_materials) or sphere_materials[index] != contact.material_id
                       or sphere_geometry.get(index) != (contact.native_geometry,) for index in contact.sphere_indices)):
            raise ValueError('Holding contact must bind exactly one complete registered robot material and owner')
        # BEGIN a declaration cannot create an attachment through material identity.
        relations = [relation for owner in owners if owner.owner_id == contact.target_owner
                     for relation in owner.attachments if relation.condition_id == contact.condition_id]
        if len(relations) != 1 or relations[0].interval != contact.interval:
            raise ValueError('Holding contact must retain its current attachment interval')
        # END release lifetime belongs to the actual attachment contract.
        identities.add(contact.rule_id)
        pairs.add(pair)


def validate_holding_conditions(contacts, owners, conditions):
    """Check the same current attachment, without comparing any earlier goal."""
    from scipy.spatial.transform import Rotation

    by_rule = {contact.rule_id: contact for contact in contacts}
    consumed = set()
    for condition in conditions:
        if condition.get('kind') != 'observed_attachment':
            continue
        ids, owner = condition.get('contact_rule_ids'), condition.get('carrier_id')
        relations = [relation for row in owners if row.owner_id == owner for relation in row.attachments
                     if relation.condition_id == condition.get('condition_id')]
        if (not isinstance(ids, (tuple, list)) or not ids or len(set(ids)) != len(ids)
                or consumed.intersection(ids) or len(relations) != 1):
            raise ValueError('Observed Holding requires its current material contacts and one attachment')
        relation, = relations
        for identity in ids:
            contact = by_rule.get(identity)
            if (contact is None or contact.condition_id != condition['condition_id']
                    or contact.target_owner != owner or contact.interval != relation.interval):
                raise ValueError('Holding contacts must match the current condition, owner and interval')
        actual = np.eye(4)
        actual[:3,3] = relation.position_tool_owner
        actual[:3,:3] = Rotation.from_quat(np.asarray(relation.quaternion_tool_owner)[[1,2,3,0]]).as_matrix()
        observed = np.asarray(condition.get('T_carrier_eef'), dtype=float)
        if (observed.shape != (4,4) or not np.isfinite(observed).all()
                or not np.allclose(np.linalg.inv(observed), actual, atol=1e-7, rtol=0)):
            raise ValueError('Holding condition and collision model must use the same current attachment')
        consumed.update(ids)
    if consumed != set(by_rule):
        raise ValueError('Every internal Holding contact must belong to a current observed attachment')
# END contact geometry is internal to Holding; task arrival goals remain independent.
