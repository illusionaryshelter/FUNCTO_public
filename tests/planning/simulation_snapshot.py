"""Save one completed native boundary together with its control state.

The simulator owns ``mjSTATE_INTEGRATION``, including the physical scene. The
backend separately owns named integral acceleration, its native time and the
controller policy. Only that explicit backend record crosses the controller
restore interface; native task state remains inside the simulator adapter.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np


def save_snapshot(backend, path):
    """Write native integration state and committed controller state to NPZ.

    The backend's snapshot API requires a completed physical-step boundary:
    its integral has already consumed that interval, and no newly issued
    control or requested output rebase is awaiting a physical step. The
    caller owns this paused backend throughout the synchronous save. Native
    and controller state therefore describe the same instant without another
    clock or a second step ledger. The parent directory must already exist.
    """
    control_state = backend.snapshot_control_state()
    control_json = json.dumps(control_state, allow_nan=False, separators=(",", ":"))
    state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    native_state = np.empty(mujoco.mj_stateSize(backend.native_model, state_spec))
    mujoco.mj_getState(backend.native_model, backend.native_data, native_state, state_spec)
    path = Path(path)
    # Use the given filename exactly and preserve any prior experiment file.
    # Unicode JSON and numeric arrays require neither pickle nor Python objects.
    with path.open("xb") as stream:
        np.savez_compressed(stream, integration_state=native_state,
                            state_spec=np.asarray(state_spec, dtype=np.int64),
                            control_state_json=np.asarray(control_json))
    return path


def restore_native_snapshot(model, data, path):
    """Restore native state and return the corresponding backend state record.

    ``model`` must be the same explicit physical model used for the snapshot.
    The caller refreshes derived native quantities and constructs the backend,
    then passes this returned record to ``restore_control_state``. That method
    checks named coordinates, controller policy and native time before loading
    the integral. Restoration never derives an integral from saturated ctrl.
    """
    with np.load(path, allow_pickle=False) as archive:
        control_state = json.loads(archive["control_state_json"].item())
        state_spec = int(archive["state_spec"].item())
        mujoco.mj_setState(model, data, archive["integration_state"], state_spec)
    return control_state
