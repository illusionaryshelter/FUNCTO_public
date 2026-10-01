"""Measure or supervise a bounded control reference in a complete mounted scene.

This is an explicitly constructed control experiment, not a skill candidate.
It preserves the restored measured entrance velocity and native controller /
contact parameters. A measurement does not invent execution tolerances; an
executor trial requires its separately supplied, explicitly scoped limits.
"""

# BEGIN complete-context control diagnostic using the production backend.
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.execution import ExecutionLimits, FixedFingerTargets, TrajectoryExecutor
from manipulation.planning import JointTrajectory
from recorded_simulation import restore_recorded_simulation
from robot_validation_fixture import build_robot_validator


def read_json(path):
    if Path(path).stat().st_size > 16 * 1024**2:
        raise ValueError("Control diagnostic metadata exceeds 16 MiB")
    return json.loads(Path(path).read_text())


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "preparing", "scope": "complete mounted control baseline; no task acceptance",
              "mode": "supervised_control_reference" if args.execution_limits else "measurement",
              "planner_called": False, "planner_accepted": False, "holding_certified": False,
              "geometry_scope": "independent actual robot entities, not full scene/continuous collision",
              "steps_completed": 0, "saturated_coordinates": 0, "physics_parameters_changed": False,
              "fixture": str(args.fixture.resolve()), "frame_index": args.frame_index,
              "controller_manifest": str(args.controller_manifest.resolve()),
              "controller_part": args.controller_part, "gpu_used": False}
    backend = None
    samples, controls, contact_rows, contact_offsets, geometry_rows = [], [], [], [0], []
    caught = None
    try:
        fixture = read_json(args.fixture)
        source = read_json(args.controller_manifest)
        part = source["environment"]["controller_config"]["body_parts"][args.controller_part]
        if (part["type"] != "JOINT_POSITION" or part["input_type"] != "absolute"
                or part["interpolation"] is not None or part["gripper"]["type"] != "GRIP"):
            raise ValueError("The source must explicitly use absolute joint position and uninterpolated GRIP")
        # A manifest from another recording must not supply plausible-looking
        # gains for the scene under test. Exact source identity is available.
        if args.controller_manifest.resolve().parent != Path(fixture["episode_path"]).resolve().parent:
            raise ValueError("Controller manifest and fixture must belong to the same recording")
        backend, restored = restore_recorded_simulation(
            scene_path=fixture["source_scene_xml"], catalog_path=fixture["catalog_path"],
            episode_path=fixture["episode_path"], frame_index=args.frame_index,
            base_link=fixture["base_link"], tcp_site=fixture["tcp_site"],
            arm_joint_names=fixture["arm_joint_names"], finger_joint_names=fixture["finger_joint_names"],
            arm_kp=part["kp"], arm_kd=part["kd"], velocity_limits=fixture["velocity_limits"])
        initial = backend.proprioception()
        narm = len(backend.arm_joint_names)
        delta = np.asarray(args.arm_delta, dtype=float)
        if delta.shape != (narm,) or not np.isfinite(delta).all():
            raise ValueError("arm-delta requires one finite displacement per named arm joint")
        duration = args.move_seconds + args.hold_seconds
        if (not np.isfinite([args.move_seconds, args.hold_seconds]).all()
                or args.move_seconds <= 0 or args.hold_seconds <= 0):
            raise ValueError("Move and final-hold durations must be explicit, finite and positive")
        count = int(np.ceil(duration / backend.timestep))
        if not 0 < count <= args.max_physics_steps <= 5000 or not 0 < args.max_contact_rows <= 100000:
            raise ValueError("This small diagnostic is limited to 5000 steps and 100000 contact rows")
        q0, dq0 = np.asarray(initial["q"][:narm]), np.asarray(initial["dq"][:narm])
        target = q0 + delta
        events, targets = (), ()
        fingers = backend.finger_control.copy()
        if args.finger_targets is not None:
            if args.finger_command is None:
                raise ValueError("Explicit finger targets require their declared command kind")
            fingers = np.asarray(args.finger_targets, dtype=float)
            if fingers.shape != (len(backend.finger_joint_names),) or not np.isfinite(fingers).all():
                raise ValueError("Finger targets must contain every independent coordinate")
            events = ({"skill_id": "control_baseline", "event_id": "explicit_finger_command",
                       "kind": args.finger_command, "time": 0., "progress": 0., "stop": False,
                       "requires_evidence": None},)
            targets = ({"skill_id": "control_baseline", "event_id": "explicit_finger_command",
                        "kind": args.finger_command, "positions": fingers.tolist()},)
        elif args.finger_command is not None:
            raise ValueError("A finger command requires its independent position targets")
        # The first derivative is measured, not silently replaced by zero.
        # This explicitly new diagnostic curve has a stationary final hold;
        # it does not trim, retime or modify any original skill trajectory.
        curve = JointTrajectory(tuple(backend.arm_joint_names),
            np.array([0., args.move_seconds, duration]), np.stack((q0, target, target)),
            np.stack((dq0, np.zeros(narm), np.zeros(narm))),
            np.array([0., args.move_seconds / duration, 1.]), np.array([0., 1., 1.]),
            events, (), {})
        validator = build_robot_validator(fixture, reference_frame=initial["reference_frame"],
            T_reference_base=initial["base_pose"], tool_frame=initial["tool_link"],
            T_frame_tcp=initial["tool_transform"])
        report.update(restore=restored, native_timestep_s=backend.timestep,
                      reference_duration_s=duration, initial_feedback=_json(initial),
                      arm_delta=delta.tolist(), finger_target=fingers.tolist(),
                      controller_configuration=part, independent_validator=validator.describe())
        np.savez(output / "control-reference.npz", joint_names=curve.joint_names,
                 times=curve.times, position=curve.position, velocity=curve.velocity,
                 progress_u=curve.progress_u, progress_s=curve.progress_s)
        model, data = backend.native_model, backend.native_data
        robot_root = model.body(fixture["base_link"]).id
        actuator_ids = [model.actuator(name).id for name in backend.actuator_names]
        dof_ids = [int(model.jnt_dofadr[model.joint(name).id]) for name in backend.joint_names]
        report["native_parameters"] = {
            "joint_names": list(backend.joint_names), "joint_ranges": backend.joint_ranges.tolist(),
            "velocity_limits": backend.velocity_limits,
            "damping": model.dof_damping[dof_ids].tolist(),
            "armature": model.dof_armature[dof_ids].tolist(),
            "frictionloss": model.dof_frictionloss[dof_ids].tolist(),
            "actuator_names": list(backend.actuator_names),
            "actuator_gainprm": model.actuator_gainprm[actuator_ids].tolist(),
            "actuator_biasprm": model.actuator_biasprm[actuator_ids].tolist(),
            "actuator_ctrlrange": model.actuator_ctrlrange[actuator_ids].tolist(),
            "actuator_forcerange": model.actuator_forcerange[actuator_ids].tolist(),
            "gravity": model.opt.gravity.tolist(), "integrator": int(model.opt.integrator),
            "solver": int(model.opt.solver), "solver_iterations": int(model.opt.iterations)}

        def robot_geom(geom):
            body = int(model.geom_bodyid[geom])
            while body and body != robot_root:
                body = int(model.body_parentid[body])
            return body == robot_root

        robot_geoms = np.array([robot_geom(i) for i in range(model.ngeom)])

        def capture(elapsed):
            feedback = backend.proprioception()
            q, dq, _ = curve.sample(min(elapsed, duration))
            tcp = backend.reference_tool_pose(q)
            # Preserve the completed physical sample before any evaluator can
            # reject nonfinite feedback; terminal evidence must not disappear.
            samples.append({"time": elapsed, "q": feedback["q"], "dq": feedback["dq"],
                            "reference_q": q, "reference_dq": dq, "tcp": feedback["tool_pose"],
                            "reference_tcp": tcp, "ctrl": backend.actuator_control.copy(),
                            "actuator_force": data.actuator_force[actuator_ids].copy(),
                            "qfrc_actuator": data.qfrc_actuator[dof_ids].copy(),
                            "qfrc_constraint": data.qfrc_constraint[dof_ids].copy()})
            if len(contact_rows) + data.ncon > args.max_contact_rows:
                raise RuntimeError("Contact budget exhausted; no contacts are silently truncated")
            for index in range(data.ncon):
                contact = data.contact[index]
                first, second = int(contact.geom1), int(contact.geom2)
                force = np.zeros(6)
                active = bool(contact.efc_address >= 0 and not contact.exclude)
                if active:
                    mujoco.mj_contactForce(model, data, index, force)
                category = int(robot_geoms[first]) + int(robot_geoms[second])
                contact_rows.append([elapsed, first, second, category, float(contact.dist),
                                     int(active), *contact.pos.tolist(), *force.tolist(),
                                     *contact.frame.tolist(), int(contact.efc_address), int(contact.exclude)])
            contact_offsets.append(len(contact_rows))
            try:
                state_check = validator.check_state(feedback, stage="control_probe", trajectory_time=elapsed)
            except Exception as error:
                geometry_rows.append({"passed": False, "stage": "control_probe", "trajectory_time": elapsed,
                                      "error": f"{type(error).__name__}: {error}"})
                raise
            geometry_rows.append(state_check)

        capture(0.)
        report["status"] = "running"
        if args.execution_limits:
            limits = ExecutionLimits.from_mapping(read_json(args.execution_limits))
            report["execution_limits"] = limits.describe()
            executor = TrajectoryExecutor(backend, geometry_checker=validator)
            execution = executor.execute(curve, evaluation_mode="diagnostic_control_reference",
                finger_targets=FixedFingerTargets(tuple(backend.finger_joint_names),
                    "Explicit bounded control experiment; no source skill event", targets),
                initial_feedback=initial, execution_limits=limits, output=output / "execution",
                max_physics_steps=args.max_physics_steps,
                start_position_tolerance=0., start_velocity_tolerance=0.,
                step_callback=lambda model, data, record: capture(record["elapsed"]))
            report["execution"] = execution
            report["steps_completed"] = execution["executed_steps"]
        else:
            # Measurement deliberately has no invented pass band. All actual
            # geometry outcomes are recorded; only the supervised branch above
            # claims that the production executor enforced a supplied contract.
            for index in range(count):
                elapsed = (index + 1) * backend.timestep
                q, _, _ = curve.sample(min(elapsed, duration))
                control = backend.apply_control(q, fingers)
                controls.append({"time": index * backend.timestep,
                                 "requested_arm_torque": control["requested_arm_torque"],
                                 "arm_torque_saturated": control["arm_torque_saturated"]})
                report["saturated_coordinates"] += control["saturated_coordinates"]
                backend.step()
                report["steps_completed"] = index + 1
                capture(elapsed)
                if not np.isfinite(np.concatenate((samples[-1]["q"], samples[-1]["dq"]))).all():
                    raise FloatingPointError("Nonfinite native state ends the bounded diagnostic")
                actual_q, actual_dq = samples[-1]["q"], samples[-1]["dq"]
                ranges = backend.joint_ranges
                speeds = np.array([backend.velocity_limits[name] for name in backend.joint_names])
                if (np.any(actual_q < ranges[:, 0]) or np.any(actual_q > ranges[:, 1])
                        or np.any(np.abs(actual_dq) > speeds)):
                    report["first_hardware_limit_violation"] = {"elapsed_s": elapsed,
                        "q": actual_q.tolist(), "dq": actual_dq.tolist()}
                    raise RuntimeError("Actual native joint position/velocity limit exceeded")
        report["status"] = "completed"
    except Exception as error:
        caught = error
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (output / "exception.txt").write_text(traceback.format_exc())
    finally:
        if samples:
            report["steps_completed"] = max(report["steps_completed"], len(samples) - 1)
            child_report = output / "execution" / "execution-report.json"
            if child_report.is_file():
                report["execution"] = read_json(child_report)
            arrays = {key: np.asarray([row[key] for row in samples]) for key in samples[0]}
            arrays.update({"control_" + key: np.asarray([row[key] for row in controls])
                           for key in ("time", "requested_arm_torque", "arm_torque_saturated")})
            arrays.update(contacts=np.asarray(contact_rows, dtype=float).reshape(-1, 26),
                          contact_offsets=np.asarray(contact_offsets, dtype=int))
            np.savez_compressed(output / "measurement.npz", **arrays)
            arm_count = arrays["reference_q"].shape[1]
            errors = arrays["q"][:, :arm_count] - arrays["reference_q"]
            speed_errors = arrays["dq"][:, :arm_count] - arrays["reference_dq"]
            finite_rotation = (np.isfinite(arrays["tcp"][:, :3, :3]).all(axis=(1, 2))
                               & np.isfinite(arrays["reference_tcp"][:, :3, :3]).all(axis=(1, 2)))
            rotation_error = np.full(len(samples), np.nan)
            if np.any(finite_rotation):
                rotation_error[finite_rotation] = Rotation.from_matrix(np.einsum("nij,nkj->nik",
                    arrays["tcp"][finite_rotation, :3, :3],
                    arrays["reference_tcp"][finite_rotation, :3, :3])).magnitude()
            report["measured_errors"] = {
                "maximum_arm_position_rad": np.max(np.abs(errors), axis=0).tolist(),
                "maximum_arm_velocity_rad_s": np.max(np.abs(speed_errors), axis=0).tolist(),
                "maximum_tcp_position_m": float(np.max(np.linalg.norm(
                    arrays["tcp"][:, :3, 3] - arrays["reference_tcp"][:, :3, 3], axis=1))),
                "maximum_tcp_rotation_rad": float(np.max(rotation_error)),
                "final_arm_position_rad": errors[-1].tolist(),
                "final_arm_velocity_rad_s": arrays["dq"][-1, :arm_count].tolist()}
            report["contacts"] = {"rows": len(contact_rows), "complete": len(contact_offsets) == len(samples) + 1,
                "columns": ["elapsed_s", "geom1", "geom2", "robot_geom_count", "distance_m",
                            "force_valid", "x", "y", "z", "fx", "fy", "fz", "tx", "ty", "tz",
                            *[f"frame_{i}" for i in range(9)], "efc_address", "exclude"],
                "force_semantics": "MuJoCo contact-frame force/torque after mj_forward; inactive rows have zero force and force_valid=false",
                "categories": {"0": "environment support", "1": "robot/environment", "2": "robot self"},
                "geometries": [backend.native_model.geom(i).name for i in range(backend.native_model.ngeom)]}
            report["geometry_capture_complete"] = len(geometry_rows) == len(samples)
            report["independent_actual_geometry_passed"] = (
                bool(geometry_rows) and report["geometry_capture_complete"]
                and all(row["passed"] for row in geometry_rows))
            (output / "geometry-checks.json").write_text(json.dumps(_json(geometry_rows), indent=2, allow_nan=False) + "\n")
        report.update(elapsed_wall_s=time.monotonic() - started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (output / "report.json").write_text(json.dumps(_json(report), indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: report[key] for key in ("status", "mode", "steps_completed", "elapsed_wall_s", "peak_rss_mib")}))
    if caught:
        raise caught


def _json(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "controller-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--controller-part", required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--arm-delta", nargs="+", type=float, required=True)
    parser.add_argument("--move-seconds", type=float, required=True)
    parser.add_argument("--hold-seconds", type=float, required=True)
    parser.add_argument("--max-physics-steps", type=int, required=True)
    parser.add_argument("--max-contact-rows", type=int, required=True)
    parser.add_argument("--finger-targets", nargs="+", type=float)
    parser.add_argument("--finger-command", choices=("open", "close", "hold"))
    parser.add_argument("--execution-limits", type=Path)
    run(parser.parse_args())
# END complete-context control diagnostic.
