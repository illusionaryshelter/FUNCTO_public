"""Prepare an explicitly chosen, centered single-furniture simulation fixture.

This command edits scene data, never perception or skill parameters. Source
layouts remain available as construction inputs; tests load the new directory.
The existing named-state adapter and calibrated camera retain robot state and
camera geometry when removing a furniture subtree changes model dimensions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from . import load_scene, scene_directory


def prepare_scene(source_case, case, *, keep_body, remove_body):
    """Remove one fixed root body and place the retained body's XY at table XY.

    Inputs name fixture bodies explicitly. They do not select an inference
    method, a semantic prompt, or any skill-dependent numerical parameter.
    All rendered modalities and feedback describe the same unstepped state.
    """
    import mujoco
    from collect.camera import FixedCamera
    from transfer.scenes import restore_named_state

    source = scene_directory(source_case)
    destination = Path(__file__).resolve().parent / case
    if destination.parent != source.parent or not case or destination.exists():
        raise ValueError("Choose a new direct child scene directory")
    model_old, data_old = load_scene(source_case)
    tree = ET.parse(source / "scene.xml")
    world = tree.find("worldbody")
    keep = world.find(f"body[@name='{keep_body}']")
    remove = world.find(f"body[@name='{remove_body}']")
    if keep is None or remove is None or keep is remove:
        raise ValueError("Retained and removed furniture must be distinct root bodies")
    if keep.find("joint") is not None or keep.find("freejoint") is not None:
        raise ValueError("This fixture placement requires a fixed furniture root")
    table_xy = data_old.xpos[model_old.body("table").id, :2].copy()
    position = np.fromstring(keep.get("pos", "0 0 0"), sep=" ")
    position[:2] = table_xy
    keep.set("pos", " ".join(format(value, ".17g") for value in position))
    world.remove(remove)
    destination.mkdir()
    ET.indent(tree)
    tree.write(destination / "scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(destination / "scene.xml"))
    data = restore_named_state(model, model_old, data_old)
    mujoco.mj_forward(model, data)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, remove_body) != -1:
        raise AssertionError("Removed furniture still exists in the compiled model")
    if not np.allclose(data.xpos[model.body(keep_body).id, :2], table_xy, atol=1e-12):
        raise AssertionError("Furniture root does not share the table center XY")
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    np.savez_compressed(destination / "initial-state.npz", integration_state=state, state_spec=spec)

    feedback = json.loads((source / "robot_feedback.json").read_text())
    joints = [model.joint(name) for name in feedback["joint_names"]]
    feedback["qpos"] = [float(data.qpos[joint.qposadr[0]]) for joint in joints]
    feedback["qvel"] = [float(data.qvel[joint.dofadr[0]]) for joint in joints]
    site = model.site(feedback["tool_frame"]).id
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = data.site_xmat[site].reshape(3, 3), data.site_xpos[site]
    feedback.update(eef_pose=pose.tolist(), sim_time_s=float(data.time),
                    source="Same unstepped scene state: named robot q/dq and native tool-site FK")
    (destination / "robot_feedback.json").write_text(json.dumps(feedback, indent=2) + "\n")
    report = dict(source_case=source_case, case=case, kept_body=keep_body,
                  removed_body=remove_body, furniture_root_position_m=position.tolist(),
                  table_center_xy_m=table_xy.tolist(), physics_steps=0,
                  state_transfer="Surviving named joint q/dq, actuator ctrl, mocap and time; new integration state",
                  sim_time_s=float(data.time), views={})
    for view in ("view_a", "view_b", "top"):
        folder = destination / view
        folder.mkdir()
        camera = FixedCamera(model, data, 640, 480, camera_name=None,
                             camera_config=source / view / "camera.json")
        try:
            frame = camera.capture(data)
            calibration = camera.calibration
        finally:
            camera.close()
        old_calibration = json.loads((source / view / "calibration.json").read_text())
        for field in ("K", "T_world_camera_cv"):
            if not np.allclose(calibration[field], old_calibration[field], atol=1e-9, rtol=0):
                raise AssertionError(f"Changed {view} camera {field}")
        Image.fromarray(frame["rgb"]).save(folder / "rgb.png")
        for field in ("depth_m", "seg_geom_id"):
            np.save(folder / (field + ".npy"), frame[field], allow_pickle=False)
        calibration["timestamp_s"] = float(data.time)
        (folder / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
        shutil.copyfile(source / view / "camera.json", folder / "camera.json")
        depth = frame["depth_m"]
        report["views"][view] = dict(valid_depth_pixels=int((np.isfinite(depth) & (depth > 0)).sum()),
                                    renderer=calibration["rendering_device"])
    (destination / "layout.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--keep-body", required=True)
    parser.add_argument("--remove-body", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_scene(args.source, args.case, keep_body=args.keep_body,
                                   remove_body=args.remove_body), indent=2), flush=True)


if __name__ == "__main__":
    main()
