"""Canonical simulation scenes with the confirmed centered Franka base.

These are experiment fixtures, not task-specific manipulation policies. Both
cases load the same way; their XML and recorded initial state own the layout.
The loader never searches dated experiments or substitutes another scene.
"""

from pathlib import Path


def scene_directory(case: str) -> Path:
    """Locate an explicitly named permanent scene, independent of the CWD."""
    if case not in ("original", "generalization"):
        raise ValueError("Scene must be 'original' or 'generalization'")
    return Path(__file__).resolve().parent / case


def load_scene(case: str):
    """Return native MuJoCo model/data restored without taking a physics step.

    The full integration state includes object and robot state. It belongs to
    simulation setup only; perception must still consume rendered observations.
    Camera configurations live at ``<case>/<view>/camera.json``.
    """
    import mujoco
    import numpy as np

    directory = scene_directory(case)
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    data = mujoco.MjData(model)
    with np.load(directory / "initial-state.npz", allow_pickle=False) as archive:
        state = archive["integration_state"]
        spec = int(archive["state_spec"])
    if spec != int(mujoco.mjtState.mjSTATE_INTEGRATION):
        raise ValueError("Scene must declare the native complete integration state")
    if state.shape != (mujoco.mj_stateSize(model, spec),) or not np.isfinite(state).all():
        raise ValueError("Initial state is incompatible with the scene model")
    mujoco.mj_setState(model, data, state, spec)
    mujoco.mj_forward(model, data)
    return model, data
