"""Bind a confirmed grasp to measured hand geometry and arm-only planning.

Position targets belong to the executor. This boundary consumes synchronized
robot feedback and caller-verified holding evidence; it neither executes a
closure nor infers contact from a commanded width or a quiet robot.
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


def build_post_grasp_robot_config(robot, feedback, scene):
    """Return a new measured-hand robot config and an arm-only start state.

    ``feedback`` is one complete named robot snapshot: q/dq, arm_joint_names,
    finger_joint_names, feedback_id, timestamp_s, source, reference_frame,
    base_pose, tool_pose, tool_link and tool_transform. Commanded finger targets
    and actuator ctrl may be retained in that snapshot but are never geometry
    inputs. ``scene`` is the same-time current observation, with observation_ids,
    initial_pose (reference-from-carrier), eef_pose, and ``holding``. Holding
    must already be verified, name this robot_feedback_id, carrier_id, the same
    observation_ids and its own source, and supply the actual T_carrier_eef.
    This function validates that binding;
    it does not certify the caller's contact/holding measurements.

    The original cspace layout, arm rest posture, collision geometry and pair
    policy are preserved. Only finger locks and their matching default entries
    are updated from actual q. Planning q/dq contains only the named arm joints;
    full measured state remains available separately. The result describes a
    current nominal held-hand configuration, not a bound on future finger
    motion or a complete environment/payload collision scene. New measurements
    or a release require a new binding, not reuse of a stale model.
    """
    if not all(isinstance(value, Mapping) for value in (robot, feedback, scene)):
        raise TypeError("Robot, complete feedback and current scene must be mappings")
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
    times = np.asarray([feedback["timestamp_s"], scene["timestamp_s"]], float)
    if not np.isfinite(times).all() or abs(times[0]-times[1]) > 1e-9:
        raise ValueError("Holding observations and complete robot feedback must be synchronized")
    if scene.get("state") != "current_observed" or scene.get("reference_frame") != frame:
        raise ValueError("Post-grasp geometry needs current observations in the robot reference frame")
    observations = _names(scene["observation_ids"], "Current observation identities")
    carrier_id = scene["carrier_id"]
    if not isinstance(carrier_id, str) or not carrier_id.strip():
        raise ValueError("The current observed grasp carrier must have an explicit identity")
    holding = scene.get("holding")
    if (not isinstance(holding, Mapping) or holding.get("verified") is not True
            or holding.get("robot_feedback_id") != identifier
            or holding.get("carrier_id") != carrier_id):
        raise ValueError("Confirmed holding must identify this robot feedback and observed carrier")
    if set(_names(holding["observation_ids"], "Holding observation identities")) != set(observations):
        raise ValueError("Confirmed holding must use the same current observation identities")
    holding_source = _source(holding.get("source"), "Confirmed holding")
    base = _pose_array(feedback["base_pose"], "measured robot base")
    tool = _pose_array(feedback["tool_pose"], "measured tool pose")
    tool_transform = _pose_array(feedback["tool_transform"], "native tool calibration")
    tool_link = feedback["tool_link"]
    if not isinstance(tool_link, str) or not tool_link.strip():
        raise ValueError("Measured feedback must identify its calibrated native tool link")
    observed_tool = _pose_array(scene["eef_pose"], "observed current EEF")
    carrier = _pose_array(scene["initial_pose"], "observed current carrier")
    attachment = _pose_array(holding["T_carrier_eef"], "actual holding transform")
    if (not np.allclose(tool, observed_tool, atol=1e-8, rtol=0)
            or not np.allclose(carrier @ attachment, tool, atol=1e-6, rtol=0)):
        raise ValueError("Actual holding, current object observation and measured EEF disagree")

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
        "source": source, "feedback_id": identifier, "timestamp_s": float(times[0]),
    }
    binding = {
        "state": "measured_post_grasp", "robot_feedback_id": identifier,
        "timestamp_s": float(times[0]), "reference_frame": frame,
        "observation_ids": list(observations), "carrier_id": carrier_id,
        "holding_source": holding_source,
        "finger_joint_names": list(fingers), "finger_positions": list(locked.values()),
        "finger_velocities": [float(dq[names.index(name)]) for name in fingers],
        "T_carrier_eef": attachment.tolist(), "active_joint_names": list(arm),
        "scope": "Measured hand robot collision configuration; environment and payload still require current evidence",
    }
    return {"robot_config": result, "planning_feedback": active_feedback,
            "full_feedback": deepcopy(feedback), "hand_binding": binding}
# END measured post-grasp hand configuration, independent of optimizer DOF.
