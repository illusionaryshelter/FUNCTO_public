"""Restore observed task owners and their declared rigid relationships."""

# BEGIN recorded inputs use the same geometry contract as production scenes.
import json
from pathlib import Path

import numpy as np

from manipulation.observed_mapping import FrozenObservedField


def load_observed_owner(path):
    """Restore the same pre-relationship field and the declared rigid motion."""
    from manipulation.planning_problem import PosePath
    from manipulation.planning_scene import Attachment, CollisionOwner
    row = json.loads(Path(path).read_text())
    field = FrozenObservedField.load(row['field_path'])
    pose = np.asarray(row['initial_pose'], float)
    return CollisionOwner(row['owner_id'], field,
        PosePath(np.array([0., 1.]), np.stack((pose, pose)), row['source']), row['source'],
        moving=bool(row['attachments']),
        attachments=tuple(Attachment(**relation) for relation in row['attachments']))
# END the fixture carries identity and policy, while fields carry observations.


def problem_owner_binding(problem, owner_id):
    """Copy the exact current owner pose and deduplicated held spans.

    The inverse attachment maps owner-local points into the tool frame.
    Release IDs and progress come from the current compiler's held metadata,
    never a source frame guessed to be the executable release time.
    """
    from manipulation.motion import _pose_array
    from scipy.spatial.transform import Rotation
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

