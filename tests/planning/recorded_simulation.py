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

# BEGIN archive inspection is separate from execution authorization.
def load_curve_archive(path):
    """Inspect exact recorded arrays, including rejected diagnostic curves."""
    from manipulation.planning import JointTrajectory

    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as arrays:
        if str(arrays["classification"].item()) != metadata["classification"]:
            raise ValueError("Saved trajectory arrays and metadata disagree on planner status")
        curve = JointTrajectory(tuple(arrays["joint_names"].tolist()), arrays["times"].copy(),
            arrays["position"].copy(), arrays["velocity"].copy(), arrays["progress_u"].copy(),
            arrays["progress_s"].copy(), tuple(metadata["events"]), tuple(metadata["conditions"]),
            metadata["residuals"])
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


def load_saved_trajectory(path, *, execution_limits):
    """Restore the formal result; plain labels and diagnostics cannot execute."""
    from manipulation.planning import PlanningResult
    from manipulation.execution_binding import check_result

    curve, metadata = load_curve_archive(path)
    if metadata["classification"] != "planner_accepted":
        raise ValueError("Rejected diagnostic archives are not executable PlanningResults")
    if not isinstance(execution_limits, ExecutionLimits):
        raise ValueError("Loading an executable curve requires explicit ExecutionLimits")
    if metadata.get("execution_limits") != execution_limits.describe():
        raise ValueError("Saved curve and execution must contain identical physical execution limits")
    result = PlanningResult(True, curve, {"origin": str(Path(path).resolve())},
                            execution_binding=metadata.get("execution_binding"))
    check_result(result, execution_limits=execution_limits)
    return result
# END independent preflight still runs for every restored formal result.


def restore_recorded_simulation(*, scene_path, catalog_path, episode_path, frame_index,
                                base_link, tcp_site, arm_joint_names, finger_joint_names,
                                arm_kp, arm_kd, velocity_limits):
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
    source = (f"Named native robot feedback and actuator controls from {Path(scene_path).resolve()}; "
              f"restored {Path(episode_path).resolve()} frame {frame_index}")
    backend = RobosuiteExecutionBackend(sim, base_link=base_link, tcp_site=tcp_site,
        arm_joint_names=arm_joint_names, finger_joint_names=finger_joint_names,
        arm_kp=arm_kp, arm_kd=arm_kd, velocity_limits=velocity_limits, source=source)
    report = {"scene_path": str(Path(scene_path).resolve()), "trial_id": backend.trial_id,
        "episode_path": str(Path(episode_path).resolve()), "frame_index": frame_index,
        "sim_time": recorded_time, "integration_state_spec": deepcopy(state_spec),
        "state_read_scope": "Full integration state is used only for explicit simulator setup; planning receives named robot feedback and RGB-D",
        "controller_binding": backend.describe(),
        "python_controller_history": "New absolute joint-position and independent gripper goals; prior Python/composite IK history is not restored"}
    return backend, report
# END explicit recording adapters; no production execution implementation.
