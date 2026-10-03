"""Compare measured-position hold and an unchanged curve at controller takeover.

Both trials restore the same full integration state. The hold explicitly uses
measured q and zero reference dq/ddq; the moving trial uses the unchanged
curve's analytic q/dq/ddq. No actual velocity, restored native ctrl, physics
parameter or curve coefficient is reset to remove a transient. This bounded
diagnostic is not task acceptance or a Holding observation.
"""

# BEGIN separate initialization effects from reference-curve motion.
import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.execution import ExecutionLimits
from recorded_simulation import load_saved_trajectory, restore_recorded_simulation
from robot_validation_fixture import build_robot_validator


def run(args):
    fixture = json.loads(args.fixture.read_text())
    manifest = json.loads(args.controller_manifest.read_text())
    if args.controller_manifest.resolve().parent != Path(fixture["episode_path"]).resolve().parent:
        raise ValueError("Fixture and controller must belong to the same recorded episode")
    part = manifest["environment"]["controller_config"]["body_parts"][args.controller_part]
    limits = ExecutionLimits.from_mapping(json.loads(args.execution_limits.read_text()))
    # BEGIN source validation is separate from the current control diagnostic.
    from manipulation.execution_binding import check_result
    planning_result = load_saved_trajectory(args.trajectory)
    check_result(planning_result, execution_limits=limits)
    # END this probe does not promote its control measurements to task proof.
    curve = planning_result.trajectory
    if curve.events or curve.conditions:
        raise ValueError("Takeover comparison requires an event-free control reference")
    if not np.isfinite(args.seconds) or not 0 < args.seconds <= .1:
        raise ValueError("Takeover diagnostic is limited to 100 milliseconds per trial")
    # BEGIN endpoint clamping may cover one native tick, not extend the curve.
    if args.seconds > curve.duration:
        raise ValueError("Takeover duration cannot exceed the unchanged reference interval")
    # END no repeated terminal velocity/acceleration is synthesized after it.
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "controller takeover A/B; no task acceptance", "gpu_used": False,
              "gains_or_physics_changed": False, "trial": {}, "trajectory": str(args.trajectory.resolve()),
              "initial_target": "hold: measured q with zero reference dq/ddq; trajectory: one unchanged analytic curve sample"}
    for mode in ("hold", "trajectory"):
        backend, restored = restore_recorded_simulation(
            scene_path=fixture["source_scene_xml"], catalog_path=fixture["catalog_path"],
            episode_path=fixture["episode_path"], frame_index=args.frame_index,
            base_link=fixture["base_link"], tcp_site=fixture["tcp_site"],
            arm_joint_names=fixture["arm_joint_names"], finger_joint_names=fixture["finger_joint_names"],
            arm_kp=part["kp"], arm_kd=part["kd"], velocity_limits=fixture["velocity_limits"])
        initial = backend.proprioception()
        narm = len(backend.arm_joint_names)
        if tuple(curve.joint_names) != backend.arm_joint_names:
            raise ValueError("Reference must preserve the named arm layout")
        q0 = np.asarray(initial["q"][:narm]).copy()
        # BEGIN entrance validation retains the actual nonzero velocity.
        qref, dqref, _, _ = curve.sample(0.)
        if max(np.max(np.abs(qref-q0)), np.max(np.abs(dqref-initial["dq"][:narm]))) > 2e-6:
            raise ValueError("Reference and restored measured entrance disagree")
        report["controller_policy"] = backend.describe()["controller_policy"]
        # END there is no diagnostic-only mutation of controller state.
        validator = build_robot_validator(fixture, reference_frame=initial["reference_frame"],
            T_reference_base=initial["base_pose"], tool_frame=initial["tool_link"],
            T_frame_tcp=initial["tool_transform"])
        fingers = backend.finger_control.copy()
        count = int(np.ceil(args.seconds/backend.timestep))
        if count > 100:
            raise ValueError("Takeover comparison exceeds 100 native steps")
        # BEGIN simulator-only diagnostic data stays outside online feedback.
        # This offline controller experiment records native dynamics for later
        # comparison. Neither these arrays nor their differences enter control
        # targets, the independent robot guard, or an execution observation.
        model, data = backend.native_model, backend.native_data
        dof_ids = np.asarray([int(model.joint(name).dofadr[0]) for name in backend.joint_names])
        actuator_ids = np.asarray([model.actuator(name).id for name in backend.actuator_names])

        def native_diagnostics():
            return {"qacc": data.qacc[dof_ids].copy(),
                    "actuator_force": data.actuator_force[actuator_ids].copy()}

        actual, controls, geometry = [initial], [], []
        diagnostics = [native_diagnostics()]
        # END the production proprioception contract contains no dynamics GT.
        for index in range(count):
            # BEGIN decompose the actual unified controller demand.
            if mode == "hold":
                q, dq, ddq = q0, np.zeros(narm), np.zeros(narm)
            else:
                q, dq, ddq, _ = curve.sample(min((index+1)*backend.timestep, curve.duration))
            result = backend.apply_control(q, dq, ddq, fingers)
            position = result["arm_position_feedback_torque"]
            damping = result["arm_velocity_feedback_torque"]
            velocity_feedforward = result["arm_velocity_feedforward_torque"]
            acceleration_feedforward = result["arm_acceleration_feedforward_torque"]
            compensation = result["arm_bias_torque"]
            reconstructed = position+damping+velocity_feedforward+acceleration_feedforward+compensation
            if not np.allclose(reconstructed, result["requested_arm_torque"], atol=1e-12, rtol=1e-12):
                raise RuntimeError("Native torque decomposition does not match actual controller demand")
            controls.append({"goal": np.asarray(q).copy(), "reference_dq": np.asarray(dq).copy(),
                             "reference_ddq": np.asarray(ddq).copy(), "position_torque": position.copy(),
                             "damping_torque": damping.copy(), "compensation_torque": compensation,
                             "velocity_feedforward_torque": velocity_feedforward.copy(),
                             "acceleration_feedforward_torque": acceleration_feedforward.copy(),
                             "requested_torque": result["requested_arm_torque"].copy(),
                             "actuator_ctrl": result["actuator_ctrl"].copy(),
                             "arm_torque_saturated": result["arm_torque_saturated"].copy()})
            # END every component belongs to the same pre-step reference/state.
            backend.step()
            state = backend.proprioception()
            actual.append(state)
            # BEGIN capture offline diagnostic truth without augmenting feedback.
            diagnostics.append(native_diagnostics())
            # END the following guard consumes only the declared robot state.
            check = validator.check_state(state, stage=mode, trajectory_time=(index+1)*backend.timestep)
            geometry.append(check["passed"])
            if not check["passed"]:
                raise RuntimeError("Independent actual robot geometry rejected takeover trial")
        # BEGIN label the distinct sensor and simulator-diagnostic sources.
        arrays = {key: np.asarray([row[key] for row in actual]) for key in ("q", "dq")}
        arrays.update({key: np.asarray([row[key] for row in diagnostics])
                       for key in ("qacc", "actuator_force")})
        report["native_diagnostics_source"] = "Offline native qacc/actuator_force; absent from online feedback and control/guard inputs"
        # END numerical archive alignment does not make GT a device sensor.
        arrays.update({key: np.asarray([row[key] for row in controls]) for key in controls[0]})
        np.savez_compressed(args.output/f"{mode}.npz", **arrays,
            elapsed=np.arange(count+1)*backend.timestep, control_elapsed=np.arange(count)*backend.timestep,
            joint_names=backend.joint_names, actuator_names=backend.actuator_names)
        report["trial"][mode] = {"restore": restored, "steps": count, "geometry_passed": all(geometry),
            "initial_q": np.asarray(initial["q"]).tolist(), "initial_dq": np.asarray(initial["dq"]).tolist(),
            "first_goal_error": (arrays["goal"][0]-q0).tolist(),
            "first_position_torque": arrays["position_torque"][0].tolist(),
            "first_damping_torque": arrays["damping_torque"][0].tolist(),
            "first_compensation_torque": arrays["compensation_torque"][0].tolist(),
            "first_requested_torque": arrays["requested_torque"][0].tolist(),
            "first_actual_force_increment": (arrays["actuator_force"][1]-arrays["actuator_force"][0]).tolist(),
            "maximum_actual_arm_acceleration": np.max(np.abs(arrays["qacc"][:, :narm]), axis=0).tolist()}
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({mode: {key: value for key, value in row.items() if key.startswith("first_")}
                      for mode, row in report["trial"].items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "controller-manifest", "trajectory", "execution-limits", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--controller-part", required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=.05)
    run(parser.parse_args())
# END bounded native-controller takeover comparison.
