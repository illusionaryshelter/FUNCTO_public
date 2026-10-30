"""Experimental recording and saved-curve adapters for production execution.

Only this adapter reads a recorded integration-state snapshot or saved curve.
The production backend binds the initialized simulator, and the production
executor owns all command timing, measured-state gates and native stepping.
"""

# BEGIN explicit recording adapters; no production execution implementation.
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import h5py
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manipulation.execution import ExecutionLimits
from manipulation.gripper_trajectory import FingerTrajectory
from manipulation.robosuite_execution import RobosuiteExecutionBackend

# BEGIN use the same complete formal archive bound as its writer.
from recorded_task_planning import MAX_CURVE_METADATA_BYTES
from manipulation.evidence_archive import read_evidence_json
# END ordinary input-document and per-array limits remain separate.

# BEGIN archive inspection is separate from execution authorization.
def load_curve_archive(path):
    """Inspect exact recorded arrays, including rejected diagnostic curves."""
    from manipulation.planning import JointTrajectory

    path = Path(path)
    # BEGIN bound the complete formal scene archive before its first read.
    # Exact ESDF bytes and validation evidence are larger than ordinary input
    # metadata. Preserve them intact under a separate explicit archive limit;
    # reject oversized input before allocating text or decoding JSON.
    metadata_path = path.with_suffix(".json")
    if metadata_path.stat().st_size > MAX_CURVE_METADATA_BYTES:
        raise ValueError(f"Trajectory metadata exceeds the bounded 512 MiB archive size: {metadata_path}")
    # BEGIN count and read each shared completed record once with this archive.
    metadata = read_evidence_json(metadata_path, max_bytes=MAX_CURVE_METADATA_BYTES)
    # END the restored binding retains immutable numerical evidence.
    # END the archive is parsed once, without geometry or evidence truncation.
    with np.load(path, allow_pickle=False) as arrays:
        if str(arrays["classification"].item()) != metadata["classification"]:
            raise ValueError("Saved trajectory arrays and metadata disagree on planner status")
        curve = JointTrajectory(tuple(arrays["joint_names"].tolist()), arrays["times"].copy(),
            arrays["position"].copy(), arrays["velocity"].copy(), arrays["progress_u"].copy(),
            arrays["progress_s"].copy(), tuple(metadata["events"]), tuple(metadata["conditions"]),
            metadata["residuals"],
            # BEGIN preserve verified prefix obligations before deriving the next barrier.
            satisfied_dependencies=tuple(metadata["satisfied_dependencies"]))
            # END reloaded blocks cannot reactivate an already consumed barrier.
        # BEGIN load the exact planner finger polynomial and cross-check its arrays.
        command_events = [row for row in curve.events if row["kind"] in ("open", "close", "hold")]
        if "finger_trajectory" not in metadata:
            raise ValueError("Saved trajectories must explicitly declare their finger_trajectory field")
        saved_fingers = metadata["finger_trajectory"]
        finger_arrays = {"finger_joint_names", "finger_times", "finger_position", "finger_velocity"}
        if saved_fingers is None:
            if command_events or finger_arrays & set(arrays.files):
                raise ValueError("Finger command events require the complete saved planning finger reference")
            curve.finger_trajectory = None
        else:
            if not finger_arrays <= set(arrays.files):
                raise ValueError("Saved finger reference requires all named polynomial arrays")
            fingers = FingerTrajectory.from_mapping(saved_fingers)
            if (fingers.duration != curve.duration or fingers.executable_until != curve.executable_until
                    or tuple(arrays["finger_joint_names"].tolist()) != fingers.joint_names
                    or not np.array_equal(arrays["finger_times"], fingers.times)
                    or not np.array_equal(arrays["finger_position"], fingers.position)
                    or not np.array_equal(arrays["finger_velocity"], fingers.velocity)):
                raise ValueError("Saved arm/finger arrays and metadata must describe the same original clock and polynomial")
            originals = {(row["skill_id"], row["event_id"]): row for row in command_events}
            if len(originals) != len(command_events) or set(originals) != {
                    (row["skill_id"], row["event_id"]) for row in fingers.events}:
                raise ValueError("Saved finger reference event identities differ from the original arm trajectory")
            for row in fingers.events:
                original = originals[(row["skill_id"], row["event_id"])]
                if any(key not in row or row[key] != value for key, value in original.items()):
                    raise ValueError("Saved finger events changed an original command, time or progress anchor")
            curve.finger_trajectory = fingers
        # END reload never generates a replacement finger trajectory.
    if curve.next_verification != metadata["next_verification"]:
        raise ValueError("The saved evidence barrier must be preserved exactly")
    return curve, metadata


def load_saved_trajectory(path):
    """Validate a formal source archive against its own original physical limits.

    Loading does not adopt a new controller or observation contract. A caller
    must explicitly bind that contract before the source can be executed.
    """
    from manipulation.planning import PlanningResult
    from manipulation.execution_binding import check_result

    curve, metadata = load_curve_archive(path)
    if metadata["classification"] != "planner_accepted":
        raise ValueError("Rejected diagnostic archives are not executable PlanningResults")
    # BEGIN source verification consumes the saved contract, never proposed limits.
    if not isinstance(metadata.get("execution_limits"), dict):
        raise ValueError("Saved execution limits must contain the original explicit physical contract")
    execution_limits = ExecutionLimits.from_mapping(metadata["execution_limits"])
    if (metadata["execution_limits"] != execution_limits.describe()
            or not isinstance(metadata.get("execution_binding"), dict)
            or metadata["execution_binding"].get("execution_limits") != metadata["execution_limits"]):
        raise ValueError("Saved execution limits do not retain their complete explicit contract")
    # END a later rebind independently checks every proposed physical field.
    # BEGIN retain validated archive status for callers without a second read.
    result = PlanningResult(True, curve, {"origin": str(Path(path).resolve()),
                            "archive_classification": metadata["classification"]},
                            execution_binding=metadata.get("execution_binding"))
    check_result(result, execution_limits=execution_limits)
    # END no classification is returned without the full formal result check.
    return result
# END independent preflight still runs for every restored formal result.


def restore_recorded_simulation(*, scene_path, catalog_path, episode_path, frame_index,
                                base_link, tcp_site, arm_joint_names, finger_joint_names,
                                arm_kp, arm_kd, velocity_limits, robot_mjcf_path):
    """Restore once, then return the production backend and setup provenance.

    Full world integration state is used only for this explicit setup. The
    returned backend never reloads a recording and does not expose object
    states through robot proprioception. Python controller history is absent
    from the recording and is not described as restored.
    """
    from robosuite.utils.binding_utils import MjSim

    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("The recorded setup frame must be an explicit nonnegative integer")
    catalog = json.loads(Path(catalog_path).read_text())
    state_spec = catalog["integration_state"]
    if state_spec["spec_name"] != "mjSTATE_INTEGRATION":
        raise ValueError("Only the recording's declared full integration-state format is supported")
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    if state_spec["spec_value"] != spec:
        raise ValueError("Recording and native MuJoCo integration-state definitions differ")
    model = mujoco.MjModel.from_xml_path(str(Path(scene_path).resolve()))
    sim = MjSim(model)
    data = sim.data._data
    expected_size = mujoco.mj_stateSize(model, spec)
    if expected_size != state_spec["size"]:
        raise ValueError("Recorded integration-state size differs from the selected model")
    with h5py.File(episode_path, "r") as stream:
        state = np.asarray(stream["states/integration_state"][frame_index], dtype=float)
        recorded_time = float(stream["states/sim_time"][frame_index])
    if state.shape != (expected_size,) or not np.isfinite(state).all():
        raise ValueError("Selected recorded integration state must be finite and complete")
    mujoco.mj_setState(model, data, state, spec)
    mujoco.mj_forward(model, data)
    if not np.isclose(data.time, recorded_time, atol=1e-10, rtol=0):
        raise ValueError("Restored integration time does not match the selected recorded frame")
    # BEGIN distinguish encoder-equivalent state and known robot FK from GT poses.
    source = (f"Named robot encoder-equivalent q/dq and actuator commands from {Path(scene_path).resolve()}; "
              "tool/base poses use known robot FK and the fixed calibrated base; "
              f"restored {Path(episode_path).resolve()} frame {frame_index}")
    # END the setup recording is not a runtime object-pose observation channel.
    # BEGIN official fixed-robot dynamics are independent of the live scene.
    from manipulation.robot_gravity import RobotGravity
    gravity = RobotGravity(mujoco.MjModel.from_xml_path(str(robot_mjcf_path)),
        tuple(arm_joint_names)+tuple(finger_joint_names), source=str(robot_mjcf_path))
    backend = RobosuiteExecutionBackend(sim, base_link=base_link, tcp_site=tcp_site,
        arm_joint_names=arm_joint_names, finger_joint_names=finger_joint_names,
        arm_kp=arm_kp, arm_kd=arm_kd, velocity_limits=velocity_limits, robot_gravity=gravity, source=source)
    # END only native setup owns the world integration state.
    report = {"scene_path": str(Path(scene_path).resolve()), "trial_id": backend.trial_id,
        "episode_path": str(Path(episode_path).resolve()), "frame_index": frame_index,
        "sim_time": recorded_time, "integration_state_spec": deepcopy(state_spec),
        "state_read_scope": "Full integration state is used only for explicit simulator setup; planning receives named robot feedback and RGB-D",
        "controller_binding": backend.describe(),
        "python_controller_history": "New synchronized q/dq/ddq arm reference and independent gripper goals under the declared controller policy; prior Python/composite IK history is not restored"}
    return backend, report
# END explicit recording adapters; no production execution implementation.
