"""Observe native finger position control with explicit targets and limits.

This bounded robot-only probe changes no physics parameters. Initial q is an
explicit setup input; subsequent steps write only actuator controls. Contact
forces and actual convergence are diagnostics, never evidence of a grasp.
"""

# BEGIN bounded native finger-control diagnostic with physical feedback.
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import mujoco
import numpy as np


def _vector(value, size, label, *, nonnegative=False):
    result = np.asarray(value, dtype=float)
    if (result.shape != (size,) or not np.isfinite(result).all()
            or (nonnegative and np.any(result < 0))):
        raise ValueError(f"{label} must supply one finite native value per named joint")
    return result


def convergence(times, positions, velocities, target, position_tolerance, velocity_tolerance, window_s):
    """Measure a consecutive actual-state window; a quiet blocked finger fails q.

    Entry measurements may begin a window at command application, but no
    interval before that command is counted. A previously reached target that
    was lost before the final sample does not pass the final predicate.
    """
    times = np.asarray(times, dtype=float)
    positions, velocities = np.asarray(positions, dtype=float), np.asarray(velocities, dtype=float)
    position_error = np.abs(positions-target)
    within = ((position_error <= position_tolerance).all(axis=1)
              & (np.abs(velocities) <= velocity_tolerance).all(axis=1))
    start, first_settled = None, None
    final_window = 0.
    for index, good in enumerate(within):
        if not good:
            start = None
            continue
        if start is None:
            start = index
        duration = float(times[index]-times[start])
        if first_settled is None and duration >= window_s-1e-10:
            first_settled = float(times[index])
        if index == len(times)-1:
            final_window = duration
    return {"first_settled_sim_time_s": first_settled,
            "settled_at_end": bool(within[-1] and final_window >= window_s-1e-10),
            "final_consecutive_window_s": final_window,
            "final_position_error": position_error[-1].tolist(),
            "maximum_position_error": position_error.max(axis=0).tolist(),
            "final_velocity": velocities[-1].tolist(),
            "position_tolerance": position_tolerance.tolist(),
            "velocity_tolerance": velocity_tolerance.tolist(), "required_window_s": window_s}


def run(args):
    from robosuite.controllers.parts.gripper.simple_grip import SimpleGripController
    from robosuite.utils.binding_utils import MjSim

    started = time.monotonic()
    names = tuple(args.joint_names)
    if (not 1 <= len(names) <= 8 or len(set(names)) != len(names)
            or any(not name.strip() for name in names)
            or not 1 <= args.steps_per_command <= 1000
            or not 1 <= args.max_contact_rows <= 100_000):
        raise ValueError("Declare unique scalar joints, at most 3000 steps and at most 100000 contact rows")
    initial = _vector(args.initial_q, len(names), "Explicit initial q")
    opened = _vector(args.open_target, len(names), "Open target")
    closed = _vector(args.close_target, len(names), "Close target")
    position_tolerance = _vector(args.position_tolerance, len(names), "Position tolerance", nonnegative=True)
    velocity_tolerance = _vector(args.velocity_tolerance, len(names), "Velocity tolerance", nonnegative=True)
    if not isinstance(args.limit_source, str) or not args.limit_source.strip():
        raise ValueError("Convergence thresholds require an explicit source")
    model = mujoco.MjModel.from_xml_path(str(args.native_asset.resolve()))
    if (model.nq != len(names) or model.nv != len(names) or model.njnt != len(names)
            or model.nu != len(names) or model.ngeom > 64 or model.nbody > 32):
        raise ValueError("This budget covers only the declared small robot-only finger asset")
    timestep = float(model.opt.timestep)
    if (not np.isfinite(timestep) or timestep <= 0 or not np.isfinite(args.settling_window_s)
            or not 0 < args.settling_window_s <= args.steps_per_command*timestep):
        raise ValueError("The explicit settling window must fit the finite native command duration")
    ids = np.asarray([model.joint(name).id for name in names])
    if (not np.isin(model.jnt_type[ids], [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]).all()
            or not model.jnt_limited[ids].all()):
        raise ValueError("The probe requires explicitly limited native scalar robot joints")
    actuator_ids = []
    for joint_id in ids:
        found = np.flatnonzero((model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                              & (model.actuator_trnid[:, 0] == joint_id))
        if len(found) != 1:
            raise ValueError("Each finger joint needs exactly one native joint actuator")
        actuator_ids.append(int(found[0]))
    actuator_ids = np.asarray(actuator_ids)
    actuator_names = [model.actuator(index).name for index in actuator_ids]
    if any(not isinstance(name, str) or not name for name in actuator_names):
        raise ValueError("Every recorded native finger actuator must have an explicit name")
    bounds, gains = model.actuator_ctrlrange[actuator_ids], model.actuator_gainprm[actuator_ids, 0]
    if (not model.actuator_ctrllimited[actuator_ids].all() or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] >= bounds[:, 1])
            or np.any(model.actuator_gear[actuator_ids, 0] != 1)
            or np.any(model.actuator_gear[actuator_ids, 1:] != 0)
            or np.any(model.actuator_gaintype[actuator_ids] != mujoco.mjtGain.mjGAIN_FIXED)
            or np.any(model.actuator_dyntype[actuator_ids] != mujoco.mjtDyn.mjDYN_NONE)
            or np.any(model.actuator_biastype[actuator_ids] != mujoco.mjtBias.mjBIAS_AFFINE)
            or np.any(gains <= 0) or np.any(model.actuator_biasprm[actuator_ids, 0] != 0)
            or np.any(model.actuator_biasprm[actuator_ids, 1] != -gains)):
        raise ValueError("The asset must retain its native unit-gear position actuators")
    if any(np.any(value < bounds[:, 0]) or np.any(value > bounds[:, 1]) for value in (initial, opened, closed)):
        raise ValueError("Initial control and all targets must lie inside native control ranges")
    joint_ranges = model.jnt_range[ids]
    if (not np.isfinite(joint_ranges).all() or np.any(initial < joint_ranges[:, 0])
            or np.any(initial > joint_ranges[:, 1])):
        raise ValueError("Explicit initial q must lie inside its native mechanical range")
    qids, vids = model.jnt_qposadr[ids].copy(), model.jnt_dofadr[ids].copy()
    sim, output = MjSim(model), args.output.resolve()
    data = sim.data._data
    # The only qpos write is this declared setup. Nonzero initial widths avoid
    # silently choosing an overlapping zero-width asset pose; they do not prove
    # absence of contact, which is measured and retained below even at setup.
    data.qpos[qids] = initial
    data.ctrl[actuator_ids] = initial
    mujoco.mj_forward(model, data)
    controller = SimpleGripController(sim,
        {"joints": ids.tolist(), "qpos": qids.tolist(), "qvel": vids.tolist(), "actuators": actuator_ids.tolist()},
        actuator_range=(bounds[:, 0], bounds[:, 1]), policy_freq=1/timestep, use_action_scaling=False)
    output.mkdir(parents=True, exist_ok=False)
    snapshots, controls, contacts, offsets = [], [], [], [0]
    report = {"status": "running", "native_asset": str(args.native_asset.resolve()), "joint_names": list(names),
        "initial_q": initial.tolist(), "initial_q_source": "explicit --initial-q argument; setup only",
        "open_target": opened.tolist(), "close_target": closed.tolist(),
        "convergence_limit_source": args.limit_source, "native_timestep_s": timestep,
        "convergence_limits": {"position_tolerance": position_tolerance.tolist(),
                               "velocity_tolerance": velocity_tolerance.tolist(),
                               "settling_window_s": args.settling_window_s},
        "steps_per_command": args.steps_per_command, "maximum_contact_rows": args.max_contact_rows,
        "actuator_ctrl_passthrough_exact": True, "contact_capture_complete": True, "phases": [],
        "grasp_confirmed": False, "holding_evidence_available": False,
        "gpu_used": False, "recorded_scene_or_episode_loaded": False,
        "evidence_scope": "Robot-only control, actual convergence and native self-contact diagnostics; no carrier or grasp claim",
        "contact_sampling": "initial setup and every completed native step after mj_forward; forces are recomputed at that sampled state",
        "contact_force_semantics": "MuJoCo contact-frame force/torque; force_valid requires an active solved constraint",
        "native_parameters": {
            "joint_ranges": joint_ranges.tolist(), "joint_type": ["slide" if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_SLIDE else "hinge" for index in ids],
            "joint_damping": model.dof_damping[vids].tolist(), "joint_armature": model.dof_armature[vids].tolist(),
            "joint_frictionloss": model.dof_frictionloss[vids].tolist(),
            "actuator_names": actuator_names,
            "actuator_ctrlrange": bounds.tolist(), "actuator_gainprm": model.actuator_gainprm[actuator_ids].tolist(),
            "actuator_biasprm": model.actuator_biasprm[actuator_ids].tolist(),
            "actuator_forcelimited": model.actuator_forcelimited[actuator_ids].astype(bool).tolist(),
            "actuator_forcerange": model.actuator_forcerange[actuator_ids].tolist(),
            "gravity": model.opt.gravity.tolist(), "integrator": int(model.opt.integrator),
            "solver": int(model.opt.solver), "solver_iterations": int(model.opt.iterations)},
        "geometries": [{"geom_id": index, "geom_name": model.geom(index).name,
                        "body_id": int(model.geom_bodyid[index]), "body_name": model.body(int(model.geom_bodyid[index])).name}
                       for index in range(model.ngeom)]}

    def capture():
        snapshot = {"time": float(data.time), "q": data.qpos[qids].copy(), "dq": data.qvel[vids].copy(),
            "ctrl": data.ctrl[actuator_ids].copy(), "actuator_force": data.actuator_force[actuator_ids].copy(),
            "qfrc_actuator": data.qfrc_actuator[vids].copy(), "qfrc_passive": data.qfrc_passive[vids].copy(),
            "qfrc_constraint": data.qfrc_constraint[vids].copy(), "contact_count": int(data.ncon),
            "contact_complete": offsets[-1]+data.ncon <= args.max_contact_rows}
        snapshots.append(snapshot)
        if not snapshot["contact_complete"]:
            # Preserve the final q/dq/forces but do not silently retain an
            # arbitrary subset of this state's contacts.
            offsets.append(offsets[-1])
            report["contact_capture_complete"] = False
            raise RuntimeError("Native contact rows exceed the explicitly bounded retention budget")
        rows = np.empty((data.ncon, 26), dtype=float)
        for index, contact in enumerate(data.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            force = np.zeros(6)
            active = contact.exclude == 0 and contact.efc_address >= 0
            if active:
                mujoco.mj_contactForce(model, data, index, force)
            rows[index] = np.r_[a, b, model.geom_bodyid[a], model.geom_bodyid[b],
                contact.exclude, contact.efc_address, contact.dist, contact.pos, contact.frame, force, active]
        contacts.append(rows)
        offsets.append(offsets[-1]+len(rows))
        if (not np.isfinite(rows).all() or not all(np.isfinite(snapshot[key]).all()
                for key in ("q", "dq", "actuator_force", "qfrc_actuator", "qfrc_passive", "qfrc_constraint"))):
            raise FloatingPointError("Native finger feedback/forces became nonfinite")

    try:
        capture()
        for kind, target in (("open", opened), ("close", closed), ("hold", closed)):
            begin, start_time = len(snapshots)-1, float(data.time)
            phase = {"kind": kind, "start_time_s": start_time, "position_target": target.tolist(), "status": "running"}
            report["phases"].append(phase)
            for _ in range(args.steps_per_command):
                if time.monotonic()-started > 30 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 1024**2:
                    raise RuntimeError("The probe exceeded its 30-second / 1-GiB observed budget")
                controller.set_goal(target.copy())
                ctrl = np.asarray(controller.run_controller())
                if not np.array_equal(ctrl, target):
                    report["actuator_ctrl_passthrough_exact"] = False
                    raise RuntimeError("Unscaled native controller altered an explicit position target")
                data.ctrl[actuator_ids] = ctrl
                controls.append(ctrl.copy())
                sim.step()
                mujoco.mj_forward(model, data)
                capture()
                if not np.isclose(data.time, len(controls)*timestep, atol=1e-10, rtol=0):
                    raise RuntimeError("Native time did not advance by its unchanged timestep")
            rows = snapshots[begin:]
            phase.update(status="completed", end_time_s=float(data.time), actual_q=rows[-1]["q"].tolist(),
                         actual_dq=rows[-1]["dq"].tolist(), convergence=convergence(
                             [row["time"] for row in rows], [row["q"] for row in rows], [row["dq"] for row in rows],
                             target, position_tolerance, velocity_tolerance, args.settling_window_s))
        report["status"] = "completed"
        report["command_tracking_passed"] = all(phase["convergence"]["settled_at_end"] for phase in report["phases"])
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}", command_tracking_passed=False)
        raise
    finally:
        # NPZ preserves exact raw values, including any nonfinite failure.
        # Summaries below avoid using them as successful convergence evidence.
        rows = np.concatenate(contacts) if contacts else np.empty((0, 26))
        q = np.asarray([row["q"] for row in snapshots]).reshape(-1, len(names))
        arrays = {key: np.asarray([row[key] for row in snapshots]) for key in
                  ("actuator_force", "qfrc_actuator", "qfrc_passive", "qfrc_constraint")}
        np.savez_compressed(output/"native-finger-trajectory.npz", joint_names=np.asarray(names),
            actuator_names=np.asarray(report["native_parameters"]["actuator_names"]),
            sim_time=np.asarray([row["time"] for row in snapshots]), actual_q=q,
            actual_dq=np.asarray([row["dq"] for row in snapshots]),
            actuator_ctrl=np.asarray(controls).reshape(-1, len(names)),
            control_sim_time=np.arange(len(controls))*timestep,
            sampled_ctrl=np.asarray([row["ctrl"] for row in snapshots]), **arrays,
            contact_count=np.asarray([row["contact_count"] for row in snapshots]),
            contact_sample_complete=np.asarray([row["contact_complete"] for row in snapshots]),
            contact_offsets=np.asarray(offsets, dtype=np.int64),
            contact_geom1=rows[:, 0].astype(np.int32), contact_geom2=rows[:, 1].astype(np.int32),
            contact_body1=rows[:, 2].astype(np.int32), contact_body2=rows[:, 3].astype(np.int32),
            contact_exclude=rows[:, 4].astype(np.int32), contact_efc_address=rows[:, 5].astype(np.int32),
            contact_distance_m=rows[:, 6], contact_position_world=rows[:, 7:10],
            contact_frame=rows[:, 10:19].reshape(-1, 3, 3), contact_force=rows[:, 19:25],
            contact_force_valid=rows[:, 25].astype(bool),
            geom_names=np.asarray([row["geom_name"] or "" for row in report["geometries"]]),
            geom_body_ids=np.asarray([row["body_id"] for row in report["geometries"]], dtype=np.int32),
            geom_body_names=np.asarray([row["body_name"] or "" for row in report["geometries"]]))
        violation = np.maximum(0., np.maximum(joint_ranges[:, 0]-q, q-joint_ranges[:, 1]))
        report.update(native_physics_steps=max(0, len(snapshots)-1), control_attempts=len(controls),
            retained_contact_rows=len(rows), maximum_native_contact_count=max((row["contact_count"] for row in snapshots), default=0),
            minimum_native_contact_distance_m=(float(rows[:, 6].min())
                if len(rows) and np.isfinite(rows[:, 6]).all() else None),
            maximum_native_joint_limit_violation=(violation.max(axis=0).tolist()
                if len(q) and np.isfinite(violation).all() else None),
            elapsed_seconds=time.monotonic()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-asset", type=Path, required=True)
    parser.add_argument("--joint-names", nargs="+", required=True)
    for field in ("initial-q", "open-target", "close-target", "position-tolerance", "velocity-tolerance"):
        parser.add_argument("--"+field, nargs="+", type=float, required=True)
    parser.add_argument("--settling-window-s", type=float, required=True)
    parser.add_argument("--limit-source", required=True)
    parser.add_argument("--steps-per-command", type=int, required=True)
    parser.add_argument("--max-contact-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))
# END bounded native finger-control diagnostic with physical feedback.
