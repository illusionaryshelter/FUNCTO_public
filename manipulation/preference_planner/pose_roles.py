"""Explicit pose preferences carried by the existing event/source contract.

The original pose geometry, event identity and diagnostic thresholds remain
on each PoseEventRequirement.  This planner adds a declared optimization
role inside its serialized source record.  No role is inferred from source
provenance, a skill name, a target transform or an observed execution result.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields

import numpy as np

from manipulation.planning_problem import PoseEventRequirement


# BEGIN explicit pose roles reuse the original event's serialized data fields.
def pose_objective(event):
    """Read the sole role/scale declaration from a typed event's source.

    This also accepts an original PoseEventRequirement reconstructed from a
    formal binding: the declaration needs no additional dataclass field.
    Missing declarations are input errors, never an implicit required role.
    """
    return event.source["pose_objective"]


def pose_objective_scales(event):
    """Return explicit preference scales in metres or radians per component."""
    return pose_objective(event)["scales"]


def pose_objective_components(event, position, quaternion):
    """Return the original physical errors selected by the declared scales.

    Position and wxyz orientation describe the tool in the problem reference
    frame.  The inherited geometry composes the material frame, compares it
    with the target and resolves swing/twist in target coordinates.  Free yaw
    is absent from the selected components; all raw diagnostics remain
    available through the inherited error_components and measurements APIs.
    """
    values = event.error_components(position, quaternion)
    return {name: values[name] for name in pose_objective_scales(event)}


class ObjectivePoseEvent(PoseEventRequirement):
    """Change hard membership while retaining every original dataclass field.

    The base component set remains the physical diagnostic allowed set.
    ``prior`` uses that geometry only for preferences and measurements;
    ``required`` continues to expose the original hard components.
    """

    @property
    def component_tolerances(self):
        if pose_objective(self)["role"] == "prior":
            return {}
        return super().component_tolerances

    def evaluate(self, position, quaternion):
        # A prior has no hard residual.  Preserve batch/horizon, device,
        # dtype and a zero derivative instead of Python sum(empty)'s integer.
        if pose_objective(self)["role"] == "prior":
            return position[..., 0] * 0.
        return super().evaluate(position, quaternion)


def _objective_declaration(event, row):
    """Validate the declared data once at the complete-request boundary."""
    if set(row) != {"goal_id", "role", "scales", "scale_source"}:
        raise ValueError("Declare goal_id, role, scales and scale_source for every pose event")
    if row["role"] not in {"prior", "required"}:
        raise ValueError("A pose objective role must be prior or required")
    if not isinstance(row["scale_source"], str) or not row["scale_source"].strip():
        raise ValueError("Pose preference scales require their explicit numerical source")
    # Access the original physical component set even when rebinding an
    # ObjectivePoseEvent.  Zero-width yaw still needs a positive preference
    # scale; a full free revolution contributes no yaw component at all.
    components = PoseEventRequirement.component_tolerances.fget(event)
    scales = row["scales"]
    if not isinstance(scales, Mapping) or set(scales) != set(components):
        raise ValueError("Pose scales must name exactly the original position/swing/twist components")
    normalized = {}
    for name in components:
        value = scales[name]
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value <= 0):
            raise ValueError("Pose preference scales must be finite positive metric numbers")
        normalized[name] = float(value)
    return {"role": row["role"], "scales": normalized, "scale_source": row["scale_source"]}


def bind_pose_objectives(problem, declarations):
    """Copy a complete problem and bind all original pose-event identities.

    ``declarations`` is a sequence of records with an original
    ``['pose_events', index]`` goal_id, role, scales and scale_source.  Invoke
    this before request-ledger construction or motion-block filtering.  Every
    pose event is declared exactly once, including required events.  The
    returned object retains the original PlanningProblem type and complete
    skill representation; only pose event runtime classes/source records
    change.  Runtime role data then lives solely in those source records.
    """
    if not isinstance(declarations, (list, tuple)):
        raise ValueError("Pose objectives require an explicit declaration sequence")
    by_index = {}
    for row in declarations:
        if not isinstance(row, Mapping):
            raise ValueError("Each pose objective declaration must be a record")
        identity = row.get("goal_id")
        if (not isinstance(identity, (list, tuple)) or len(identity) != 2
                or identity[0] != "pose_events" or type(identity[1]) is not int
                or not 0 <= identity[1] < len(problem.pose_events)
                or identity[1] in by_index):
            raise ValueError("Pose objectives need unique original pose-event identities")
        by_index[identity[1]] = _objective_declaration(problem.pose_events[identity[1]], row)
    if set(by_index) != set(range(len(problem.pose_events))):
        raise ValueError("Every original pose event requires an explicit role and scales")

    bound = deepcopy(problem)
    events = []
    for index, event in enumerate(bound.pose_events):
        # Source is an existing opaque-data field. Structured provenance
        # retains every original key, including contact identity metadata.
        # A scalar provenance value is retained as a named original operand.
        source = (deepcopy(dict(event.source)) if isinstance(event.source, Mapping)
                  else {"original_source": deepcopy(event.source)})
        source["pose_objective"] = by_index[index]
        values = {field.name: getattr(event, field.name) for field in fields(PoseEventRequirement)}
        values["source"] = source
        events.append(ObjectivePoseEvent(**values))
    bound.pose_events = tuple(events)
    return bound


def prior_goal_ids(binding):
    """Derive result-only pose identities from the same formal bound roles.

    The binding's original indexes survive motion-block filtering; local row
    numbers never replace them.  Original evaluator reconstruction retains
    its metric thresholds, so measured T/F/U and errors remain available when
    these identities enter the existing diagnostic-goal aggregation API.
    """
    rows = binding["problem"]["pose_events"]
    indexes = binding["goal_identities"]["pose_events"]
    if len(rows) != len(indexes):
        raise ValueError("Bound pose roles require every original goal identity")
    return [["pose_events", index] for row, index in zip(rows, indexes)
            if row["source"]["pose_objective"]["role"] == "prior"]
# END role semantics add no target-pose execution gate or observation input.
