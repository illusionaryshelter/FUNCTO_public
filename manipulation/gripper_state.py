"""Bind measured finger coordinates to hand geometry and arm-only planning.

Position targets belong to the executor. This boundary consumes synchronized
robot q/dq; it neither observes an object nor proves physical holding.
"""

# BEGIN measured post-grasp hand configuration, independent of optimizer DOF.
from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from .motion import _pose_array


def _names(values, label):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{label} must be an explicit nonempty joint-name sequence")
    names = tuple(values)
    if (any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError(f"{label} must contain unique nonempty names")
    return names


def _source(value, label):
    if not ((isinstance(value, str) and value.strip())
            or (isinstance(value, Mapping) and value)):
        raise ValueError(f"{label} requires explicit evidence provenance")
    return deepcopy(value)


def build_post_grasp_robot_config(robot, feedback):
    """Return a new measured-hand robot config and an arm-only start state.

    ``feedback`` is one complete named robot snapshot: q/dq, arm_joint_names,
    finger_joint_names, feedback_id, timestamp_s, source, reference_frame,
    base_pose, tool_pose, tool_link and tool_transform. Commanded finger targets
    and actuator ctrl may be retained in that snapshot but are never geometry
    inputs. Finger coordinates need only their robot feedback: object poses,
    contact evidence and attachment assumptions belong to separate models.

    The original cspace layout, arm rest posture, collision geometry and pair
    policy are preserved. Only finger locks and their matching default entries
    are updated from actual q. Planning q/dq contains only the named arm joints;
    full measured state remains available separately. The result describes a
    current nominal held-hand configuration, not a bound on future finger
    motion or a complete environment/payload collision scene. New measurements
    or a release require a new binding, not reuse of a stale model.
    """
    if not all(isinstance(value, Mapping) for value in (robot, feedback)):
        raise TypeError("Robot and complete feedback must be mappings")
    names = _names(feedback["joint_names"], "Complete robot feedback")
    arm = _names(feedback["arm_joint_names"], "Active arm joints")
    fingers = _names(feedback["finger_joint_names"], "Measured finger joints")
    if set(arm) & set(fingers) or set(arm) | set(fingers) != set(names):
        raise ValueError("Arm and finger names must partition complete measured robot state")
    q, dq = np.asarray(feedback["q"], float), np.asarray(feedback["dq"], float)
    if (q.shape != (len(names),) or dq.shape != q.shape
            or not np.isfinite(q).all() or not np.isfinite(dq).all()):
        raise ValueError("Complete measured q/dq must be finite and aligned with joint names")
    identifier, frame = feedback["feedback_id"], feedback["reference_frame"]
    if any(not isinstance(value, str) or not value.strip() for value in (identifier, frame)):
        raise ValueError("Current feedback needs an explicit identity and reference frame")
    source = _source(feedback["source"], "Robot feedback")
    # BEGIN a robot-only hand model does not require an object proof chain.
    timestamp = float(feedback["timestamp_s"])
    if not np.isfinite(timestamp):
        raise ValueError("Robot feedback needs a finite acquisition timestamp")
    # END missing object observations cannot reject measured finger geometry.
    base = _pose_array(feedback["base_pose"], "measured robot base")
    tool = _pose_array(feedback["tool_pose"], "measured tool pose")
    tool_transform = _pose_array(feedback["tool_transform"], "native tool calibration")
    tool_link = feedback["tool_link"]
    if not isinstance(tool_link, str) or not tool_link.strip():
        raise ValueError("Measured feedback must identify its calibrated native tool link")

    result = deepcopy(robot)
    kinematics = result["robot_cfg"]["kinematics"]
    cspace = kinematics["cspace"]
    configured = _names(cspace["joint_names"], "Native cspace joints")
    if set(configured) != set(names) or tuple(name for name in configured if name not in fingers) != arm:
        raise ValueError("Native active joint order must remain exactly the supplied arm order")
    previous_locks = kinematics.get("lock_joints")
    if previous_locks is not None and (not isinstance(previous_locks, Mapping)
                                      or not set(previous_locks) <= set(fingers)):
        raise ValueError("Binding a measured hand cannot silently unlock another native joint")
    defaults = np.asarray(cspace["default_joint_position"], float).copy()
    if defaults.shape != (len(configured),) or not np.isfinite(defaults).all():
        raise ValueError("Native default joint positions must match their full cspace layout")
    measured = dict(zip(names, q))
    locked = {name: float(measured[name]) for name in fingers}
    for name, value in locked.items():
        defaults[configured.index(name)] = value
    kinematics["lock_joints"] = locked
    cspace["default_joint_position"] = defaults.tolist()
    indices = [names.index(name) for name in arm]
    active_feedback = {
        "joint_names": list(arm), "start_q": q[indices].tolist(),
        "start_dq": dq[indices].tolist(), "reference_frame": frame,
        "base_pose": base.tolist(), "eef_pose": tool.tolist(),
        "tool_link": tool_link, "tool_transform": tool_transform.tolist(),
        "source": source, "feedback_id": identifier, "timestamp_s": timestamp,
    }
    binding = {
        "state": "measured_post_grasp", "robot_feedback_id": identifier,
        "timestamp_s": timestamp, "reference_frame": frame,
        "finger_joint_names": list(fingers), "finger_positions": list(locked.values()),
        "finger_velocities": [float(dq[names.index(name)]) for name in fingers],
        "active_joint_names": list(arm),
        "scope": "Measured finger coordinates in the robot collision model; no object pose or holding claim",
    }
    return {"robot_config": result, "planning_feedback": active_feedback,
            "full_feedback": deepcopy(feedback), "hand_binding": binding}
# END measured post-grasp hand configuration, independent of optimizer DOF.
