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
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.execution import ExecutionLimits, FixedFingerTargets, TrajectoryExecutor
from manipulation.gripper_trajectory import FingerTrajectory
from manipulation.planning import JointTrajectory
from recorded_simulation import restore_recorded_simulation, load_saved_trajectory
from robot_validation_fixture import build_robot_validator


def read_json(path):
    if Path(path).stat().st_size > 16 * 1024**2:
        raise ValueError("Control diagnostic metadata exceeds 16 MiB")
    return json.loads(Path(path).read_text())


# BEGIN report native finger force-limit samples independently of arm clipping.
def summarize_finger_force_limits(times_s, forces, *, joint_names, actuator_names, forcelimited, forcerange):
    """Count observed enabled force-limit boundaries with exact comparisons.

    The native floating-point samples use <= lower or >= upper, without an
    added tolerance. Disabled force limits never count, even when their stored
    default range is [0, 0]. A boundary observation alone does not establish
    excess unclipped demand, a tracking failure or a calibrated control bound.
    Nonfinite force samples remain separately identified and cannot count as
    successful observations at a finite force limit.
    """
    names, actuators = tuple(joint_names), tuple(actuator_names)
    times = np.asarray(times_s, dtype=float)
    forces = np.asarray(forces, dtype=float)
    enabled = np.asarray(forcelimited, dtype=bool)
    bounds = np.asarray(forcerange, dtype=float)
    count = len(names)
    if (not count or len(actuators) != count or len(set(names)) != count or len(set(actuators)) != count
            or times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all()
            or np.any(np.diff(times) < 0) or forces.shape != (len(times), count)
            or enabled.shape != (count,) or bounds.shape != (count, 2)
            or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError("Finger force summaries require complete ordered samples and named native force limits")
    finite = np.isfinite(forces)
    at_limit = finite & enabled & ((forces <= bounds[:, 0]) | (forces >= bounds[:, 1]))
    first, last = [], []
    for column in range(count):
        indexes = np.flatnonzero(at_limit[:, column])
        first.append(float(times[indexes[0]]) if len(indexes) else None)
        last.append(float(times[indexes[-1]]) if len(indexes) else None)
    return {"joint_names": list(names), "actuator_names": list(actuators),
            "native_forcelimited": enabled.tolist(), "native_forcerange": bounds.tolist(),
            "sample_count": len(times), "at_limit_sample_count": at_limit.sum(axis=0).tolist(),
            "first_at_limit_elapsed_s": first, "last_at_limit_elapsed_s": last,
            "nonfinite_force_sample_count": (~finite).sum(axis=0).tolist(),
            "comparison_rule": "finite(force) and native forcelimited and (force <= lower or force >= upper); exact native floating-point comparison, no added tolerance",
            "sampling": "Initial restored state and every completed step after mj_forward; elapsed seconds are relative to the diagnostic entrance",
            "scope": "Observed native actuator force-limit boundaries only; not a tracking predicate, excess-demand proof or control calibration"}
# END native finger force observations never replace execution acceptance.


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "preparing", "scope": "complete mounted control baseline; no task acceptance",
              "mode": "supervised_control_reference" if args.execution_limits else "measurement",
              "planner_called": False, "planner_accepted": False, "holding_certified": False,
              "geometry_scope": "independent actual robot entities, not full scene/continuous collision",
              # BEGIN distinguish arm command clipping from finger force limits.
              "steps_completed": 0, "arm_control_saturated_coordinates": 0, "physics_parameters_changed": False,
              # END native finger limits receive a separate sampled summary.
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
        q0, dq0 = np.asarray(initial["q"][:narm]), np.asarray(initial["dq"][:narm])
        limits = ExecutionLimits.from_mapping(read_json(args.execution_limits)) if args.execution_limits else None
        events, targets = (), ()
        fingers = backend.finger_control.copy()
        if args.trajectory and (limits is None or args.move_seconds is not None or args.hold_seconds is not None
                                or args.finger_targets is not None or args.finger_command is not None):
            raise ValueError("A saved planned baseline requires its identical limits and cannot receive new timing/finger events")
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
        # BEGIN either preserve an accepted exported baseline or construct a
        # separately labelled diagnostic; there is no resampling or relabelling.
        if args.trajectory:
            # BEGIN check the archive's own contract before an explicit rebind.
            planning_result = load_saved_trajectory(args.trajectory)
            # END new physical limits are checked at device binding below.
            curve = planning_result.trajectory
            if curve.events or curve.conditions:
                raise ValueError("Use run_recorded_simulation for skills with events or evidence conditions")
            duration, delta = curve.duration, curve.position[-1] - curve.position[0]
            report.update(mode="saved_planned_baseline", planner_accepted=True,
                          trajectory_source=str(args.trajectory.resolve()),
                          planning_scope="Use the source planner's declared scene scope; no recorded-scene acceptance is implied")
        else:
            delta = np.asarray(args.arm_delta, dtype=float)
            if delta.shape != (narm,) or not np.isfinite(delta).all():
                raise ValueError("arm-delta requires one finite displacement per named arm joint")
            if (args.move_seconds is None or args.hold_seconds is None
                    or not np.isfinite([args.move_seconds, args.hold_seconds]).all()
                    or args.move_seconds <= 0 or args.hold_seconds <= 0):
                raise ValueError("Move and final-hold durations must be explicit, finite and positive")
            duration = args.move_seconds + args.hold_seconds
            target = q0 + delta
            # The entrance derivative is measured, not replaced by zero. This
            # new control reference does not modify an original skill curve.
            curve = JointTrajectory(tuple(backend.arm_joint_names),
                np.array([0., args.move_seconds, duration]), np.stack((q0, target, target)),
                np.stack((dq0, np.zeros(narm), np.zeros(narm))),
                np.array([0., args.move_seconds / duration, 1.]), np.array([0., 1., 1.]), events, (), {})
        # BEGIN new diagnostic commands use the same production C1 reference.
        # A saved planner archive is never regenerated here. Only an explicitly
        # requested new control experiment owns new goals and timing. Native
        # cspace acceleration bounds constrain this reference; they do not
        # declare calibrated actual actuator acceleration or tracking error.
        target_contract = FixedFingerTargets(tuple(backend.finger_joint_names),
            "Explicit bounded control experiment; no source skill event", targets)
        if not args.trajectory and events:
            robot_config = yaml.safe_load(Path(fixture["robot_config_path"]).read_text())
            cspace = robot_config["robot_cfg"]["kinematics"]["cspace"]
            configured_names = tuple(cspace["joint_names"])
            acceleration = np.asarray(cspace["max_acceleration"], dtype=float)
            if acceleration.ndim == 0:
                acceleration = np.full(len(configured_names), float(acceleration))
            if (acceleration.shape != (len(configured_names),) or not np.isfinite(acceleration).all()
                    or np.any(acceleration <= 0.) or len(set(configured_names)) != len(configured_names)):
                raise ValueError("Control references require the explicit native cspace acceleration bounds")
            selected_acceleration = acceleration[[configured_names.index(name) for name in backend.finger_joint_names]]
            velocity = np.array([backend.velocity_limits[name] for name in backend.finger_joint_names])
            curve.finger_trajectory = FingerTrajectory.build(
                joint_names=backend.finger_joint_names, initial_position=initial["q"][narm:],
                initial_velocity=initial["dq"][narm:],
                commands=target_contract.bind(curve, finger_joint_names=backend.finger_joint_names,
                                               ctrlrange=backend.finger_ctrlrange),
                duration=curve.duration, executable_until=curve.executable_until, timestep=backend.timestep,
                position_limits=backend.joint_ranges[narm:].T,
                velocity_limits=np.stack((-velocity, velocity)),
                acceleration_limits=np.stack((-selected_acceleration, selected_acceleration)),
                source=f"New explicitly requested mounted control diagnostic; native reference bounds from {fixture['robot_config_path']}; not a contact prediction",
                feedback_id=initial["feedback_id"], timestamp_s=initial["timestamp_s"])
        finger_reference = curve.finger_trajectory
        # END measurement and supervision sample this identical saved reference.
        count = int(np.ceil(duration / backend.timestep - 1e-10))
        if not 0 < count <= args.max_physics_steps <= 5000 or not 0 < args.max_contact_rows <= 100000:
            raise ValueError("This small diagnostic is limited to 5000 steps and 100000 contact rows")
        # END exact saved baseline / separately classified control reference.
        validator = build_robot_validator(fixture, reference_frame=initial["reference_frame"],
            T_reference_base=initial["base_pose"], tool_frame=initial["tool_link"],
            T_frame_tcp=initial["tool_transform"])
        report.update(restore=restored, native_timestep_s=backend.timestep,
                      reference_duration_s=duration, initial_feedback=_json(initial),
                      arm_delta=delta.tolist(), finger_target=fingers.tolist(),
                      finger_reference=None if finger_reference is None else finger_reference.describe(),
                      controller_configuration=part, independent_validator=validator.describe())
        # BEGIN retain the exact new diagnostic reference separately from physics.
        reference_arrays = dict(joint_names=curve.joint_names, times=curve.times, position=curve.position,
                                velocity=curve.velocity, progress_u=curve.progress_u, progress_s=curve.progress_s)
        if finger_reference is not None:
            reference_arrays.update(finger_joint_names=finger_reference.joint_names,
                finger_times=finger_reference.times, finger_position=finger_reference.position,
                finger_velocity=finger_reference.velocity)
        np.savez(output / "control-reference.npz", **reference_arrays)
        (output / "control-reference.json").write_text(json.dumps({
            "classification": "planner_accepted" if args.trajectory else "diagnostic_control_reference",
            "events": curve.events, "duration": curve.duration, "executable_until": curve.executable_until,
            "finger_trajectory": None if finger_reference is None else finger_reference.describe(),
            "scope": "Exact command reference; no actual motion or Holding assertion"}, indent=2, allow_nan=False) + "\n")
        # END no reference is reconstructed from measured execution outcomes.
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
            # BEGIN preserve whether each native force range is actually enabled.
            "actuator_forcelimited": model.actuator_forcelimited[actuator_ids].astype(bool).tolist(),
            "actuator_forcerange": model.actuator_forcerange[actuator_ids].tolist(),
            # END a disabled [0, 0] range must not imply force saturation.
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
            # BEGIN retain the complete same-time reference for diagnostics.
            q, dq, ddq, _ = curve.sample(min(elapsed, duration))
            # END simulator diagnostic acceleration is not an online sensor.
            reference_fingers = (finger_reference.sample(min(elapsed, duration)) if finger_reference is not None
                                 else (np.full(len(backend.finger_joint_names), np.nan),
                                       np.full(len(backend.finger_joint_names), np.nan)))
            tcp = backend.reference_tool_pose(q)
            # Preserve the completed physical sample before any evaluator can
            # reject nonfinite feedback; terminal evidence must not disappear.
            samples.append({"time": elapsed, "q": feedback["q"], "dq": feedback["dq"],
                            "reference_q": q, "reference_dq": dq, "reference_ddq": ddq,
                            # BEGIN this offline diagnostic owns simulator truth.
                            # Production feedback deliberately contains no qacc;
                            # this array is not an online control or guard input.
                            "qacc": data.qacc[dof_ids].copy(),
                            # END retain native diagnostics outside proprioception.
                            "tcp": feedback["tool_pose"],
                            "reference_finger_q": reference_fingers[0], "reference_finger_dq": reference_fingers[1],
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
            report["execution_limits"] = limits.describe()
            executor = TrajectoryExecutor(backend, geometry_checker=validator)
            if not args.trajectory:
                raise ValueError("Production execution requires a saved formal result; use the explicit diagnostic measurement path for control probes")
            # BEGIN authorize this device contract separately from source proof.
            from manipulation.execution_binding import rebind_execution_result
            planning_result = rebind_execution_result(planning_result,
                controller_policy=backend.describe()["controller_policy"],
                result_only_goal_ids=[], finger_targets=target_contract,
                execution_limits=limits,
                source="Explicit mounted-robot trajectory-controller probe binding")
            # END the executor still performs the complete independent preflight.
            execution = executor.execute(planning_result,
                finger_targets=target_contract,
                initial_feedback=initial, execution_limits=limits, output=output / "execution",
                max_physics_steps=args.max_physics_steps,
                start_position_tolerance=args.start_position_tolerance,
                start_velocity_tolerance=args.start_velocity_tolerance,
                step_callback=lambda model, data, record: capture(record["elapsed"]))
            report["execution"] = execution
            report["steps_completed"] = execution["executed_steps"]
        else:
            # Measurement deliberately has no invented pass band. All actual
            # geometry outcomes are recorded; only the supervised branch above
            # claims that the production executor enforced a supplied contract.
            for index in range(count):
                elapsed = (index + 1) * backend.timestep
                # BEGIN every device call receives one same-time q/dq/ddq tuple.
                q, dq, ddq, _ = curve.sample(min(elapsed, duration))
                # BEGIN diagnostic measurement uses the same synchronous command.
                finger_command = (finger_reference.sample(min(elapsed, duration))[0]
                                  if finger_reference is not None else fingers)
                control = backend.apply_control(q, dq, ddq, finger_command)
                # END diagnostic input never substitutes zero moving derivatives.
                # END actual finger q/dq still come only from native feedback.
                # BEGIN retain all reference, torque-component and clipping data.
                controls.append({"time": index * backend.timestep, **control})
                # END each demand remains paired with its exact control tick.
                # BEGIN aggregate only arm command clipping, with no legacy alias.
                report["arm_control_saturated_coordinates"] += control["arm_control_saturated_coordinates"]
                # END finger force-limit observations are summarized from samples.
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
                # BEGIN read the production executor's exact arm-only counter.
                report["arm_control_saturated_coordinates"] = report["execution"]["arm_control_saturated_coordinates"]
                # END all execution modes use the same explicit report name.
                report["control_evidence"] = str(output / "execution" / "physical-trajectory.npz")
            else:
                report["control_evidence"] = str(output / "measurement.npz")
            arrays = {key: np.asarray([row[key] for row in samples]) for key in samples[0]}
            # BEGIN supervised mode owns a separate production control archive.
            if controls:
                arrays.update({"control_" + key: np.asarray([row[key] for row in controls])
                               for key in controls[0]})
            # END measurement mode preserves every component returned by its backend.
            arrays.update(contacts=np.asarray(contact_rows, dtype=float).reshape(-1, 26),
                          contact_offsets=np.asarray(contact_offsets, dtype=int))
            np.savez_compressed(output / "measurement.npz", **arrays)
            arm_count = arrays["reference_q"].shape[1]
            # BEGIN report force-limit observations from actual native forces.
            finger_actuators = np.asarray(actuator_ids[arm_count:], dtype=int)
            report["finger_actuator_force_limits"] = summarize_finger_force_limits(
                arrays["time"], arrays["actuator_force"][:, arm_count:],
                joint_names=backend.finger_joint_names, actuator_names=backend.actuator_names[arm_count:],
                forcelimited=backend.native_model.actuator_forcelimited[finger_actuators],
                forcerange=backend.native_model.actuator_forcerange[finger_actuators])
            # END this statistic does not change control or any acceptance gate.
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
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--arm-delta", nargs="+", type=float)
    reference.add_argument("--trajectory", type=Path)
    parser.add_argument("--move-seconds", type=float)
    parser.add_argument("--hold-seconds", type=float)
    parser.add_argument("--max-physics-steps", type=int, required=True)
    parser.add_argument("--max-contact-rows", type=int, required=True)
    parser.add_argument("--finger-targets", nargs="+", type=float)
    parser.add_argument("--finger-command", choices=("open", "close", "hold"))
    parser.add_argument("--execution-limits", type=Path)
    parser.add_argument("--start-position-tolerance", type=float, default=0.)
    parser.add_argument("--start-velocity-tolerance", type=float, default=0.)
    run(parser.parse_args())
# END complete-context control diagnostic.
