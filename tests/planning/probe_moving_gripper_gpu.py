"""Bounded source-backed robot planning while a tracking gripper closes in motion.

The robot FK, collision geometry, active finger coordinates, self-collision
policy, optimizer and exported cubic are actual cuRobo V2 components. The
world and position-tracking controller are explicitly synthetic contracts.
This experiment supplies no actual contact, attachment or Holding evidence.
Run serially with the other GPU probes; the Torch allocator is capped at 2 GiB.
"""

# BEGIN native moving-gripper integration experiment.
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
from scipy.interpolate import BSpline, PPoly
from scipy.spatial.transform import Rotation

# BEGIN resolve imports from this FUNCTO checkout, independent of its directory name.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# END project-local imports.

# BEGIN an explicit device clock belongs to this synthetic tracking fixture.
CONTROL_TIMESTEP_S = .002
# END this is declared input, not an inferred or factory-provided clock.


def tracking_command(knots, closed, opened, initial=.875, final=.375):
    """An explicit smooth native-coordinate command, parameterized by progress.

    The installed SciPy spline implementation supplies polynomial coefficients.
    Four initial constants and three final replicas give a realizable cubic
    with zero endpoint velocity/acceleration. Its active domain is [1,N+3],
    matching the planner's declared first/last progress pauses. This command
    is fixed before solving; it is not a width inferred from a binary close.
    """
    # BEGIN signed source coordinates share an explicitly declared opening fraction.
    closed, opened = np.asarray(closed, float), np.asarray(opened, float)
    controls = np.r_[np.full(4, initial),
                     np.linspace(initial, final, knots+1)[1:], np.full(3, final)]
    spline = BSpline(np.arange(len(controls)+4, dtype=float)-3., controls, 3)
    polynomial = PPoly.from_spline(spline)

    def bounds(progress):
        import torch
        coordinate = 1.+(knots+2)*progress.clamp(0, 1)
        breaks = torch.as_tensor(polynomial.x, device=progress.device, dtype=progress.dtype)
        coefficient = torch.as_tensor(polynomial.c, device=progress.device, dtype=progress.dtype)
        segment = torch.searchsorted(breaks, coordinate.contiguous(), right=True).clamp(1, len(breaks)-1)-1
        offset = coordinate-breaks[segment]
        value = ((coefficient[0, segment]*offset+coefficient[1, segment])*offset
                 + coefficient[2, segment])*offset+coefficient[3, segment]
        origin = torch.as_tensor(closed, device=progress.device, dtype=progress.dtype)
        span = torch.as_tensor(opened-closed, device=progress.device, dtype=progress.dtype)
        command = origin+value[..., None]*span
        return command, command

    return bounds
    # END no positive-coordinate or equal-width convention is assumed.


def run(output, *, fixture_path, native_knots=8, interpolation_buffer_size=512):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"scope": "complete source robot, synthetic known world and tracking controller",
              "production_environment_validation": False,
              "actual_holding_evidence": False, "status": "running"}
    try:
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, JointState
        from manipulation.online_geometry import FrozenPrimitives, OnlinePlanningScene
        from manipulation.planning import PlanningBudget, plan_trajectory
        from manipulation import planning as planner_module
        from manipulation.planning_problem import (
            JointCoupling, MotionPhase, NumericalTolerance, PlanningEvent, PlanningProblem,
            PosePath, PoseEventRequirement, PoseProcessRequirement)
        from manipulation.planning_scene import (
            ObservedSpace, PrismaticGripper)
        from probe_robot_fixture import load_probe_fixture, calibrate_native_hand
        from robot_fixture import create_fixed_installation_owners

        if not torch.cuda.is_available():
            raise RuntimeError("Explicit GPU access is required for this bounded experiment")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        device = DeviceCfg("cuda:0")
        # BEGIN native FK and Coal consume the same complete source fixture.
        fixture, robot, validator = load_probe_fixture(fixture_path, reference_frame="robot_base")
        robot["robot_cfg"]["kinematics"]["lock_joints"] = None
        # Native configuration loading mutates extra-link dictionaries into
        # LinkParams. Retain the declared YAML input for the planning request.
        kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(deepcopy(robot), device_cfg=device))
        names = tuple(kinematics.joint_names)
        finger_names = tuple(fixture["finger_joint_names"])
        finger_columns = [names.index(name) for name in finger_names]
        if names != tuple(fixture["joint_names"]) or len(finger_names) != 2:
            raise AssertionError("All source coordinates and both independent fingers must remain active")
        parameters = kinematics.config.kinematics_config
        limits = np.asarray([fixture["position_limits"][name] for name in finger_names], dtype=float)
        closed = limits[np.arange(2), np.abs(limits).argmin(1)]
        opened = limits[np.arange(2), np.abs(limits).argmax(1)]
        span = opened-closed
        if np.any(span == 0):
            raise ValueError("This closing experiment requires distinct open and closed source endpoints")
        initial = kinematics.default_joint_position.reshape(-1).clone()
        initial[finger_columns] = device.to_device(closed+.875*span)
        coupling = np.array([[1., -span[0]/span[1]]])
        coupling_offset = coupling @ closed
        tool_index = kinematics.tool_frames.index(fixture["tool_link"])
        frame_problem = PlanningProblem(names, initial.cpu().numpy(), np.zeros(len(names)),
            fixture["tool_link"], "robot_base", np.eye(4), validator.T_frame_tcp,
            (MotionPhase("hand_calibration", PosePath(np.array([0., 1.]),
                np.stack((np.eye(4), np.eye(4))), "explicit calibration frame")),),
            source_timestamps=np.array([0., 1.]), source_progress=np.array([0., 1.]))

        def fk(coordinates):
            """Copy native reusable FK buffers before another query can overwrite them."""
            coordinates = torch.as_tensor(coordinates, dtype=torch.float32, device=device.device)
            state = kinematics.compute_kinematics(JointState.from_position(
                coordinates[None].contiguous(), joint_names=list(names)))
            p, r = frame_problem.transform_tool_pose(state.tool_poses.position[:, :, tool_index],
                                                      state.tool_poses.quaternion[:, :, tool_index])
            return (frame_problem.transform_robot_spheres(state.robot_spheres)[0].detach().cpu().numpy().copy(),
                    p[0].detach().cpu().numpy().copy(), r[0].detach().cpu().numpy().copy())

        geometry = calibrate_native_hand(robot,
            {"joint_names": names, "start_q": initial.cpu().numpy(),
             "finger_joint_names": finger_names, "finger_link_names": fixture["finger_link_names"]},
            frame_problem, device=device, position_limits=fixture["position_limits"], fixture=fixture)
        sphere_indices, self_pairs = geometry["sphere_indices"], geometry["self_collision_pairs"]
        local_spheres, affine = geometry["local_spheres"], geometry["sphere_joint_translation"]
        installations, installation_report = create_fixed_installation_owners(fixture,
            device_cfg=device, T_output_base=np.eye(4),
            sphere_indices_by_link=geometry["sphere_indices_by_link"])
        report["native_hand"] = {
            "active_joint_names": list(names), "finger_joint_names": list(finger_names),
            "finger_sphere_indices": list(sphere_indices), "enabled_self_collision_pairs": len(self_pairs),
            "joint_limits_m": limits.tolist(), "affine_native_fk_max_error_m": geometry["affine_fk_max_error_m"],
            "drive_coupling_matrix": coupling.tolist(), "drive_coupling_offset": coupling_offset.tolist(),
            "drive_contract": "explicit synthetic equal fraction of the signed mechanical opening ranges",
            "fixed_installation_owners": installation_report,
        }
        # END native registered sphere pairs remain the online self-collision model.

        # BEGIN vary solver capacity while preserving the physical command.
        budget = PlanningBudget(ik_seeds=2, trajectory_seeds=1, knots=native_knots,
                                ik_iterations=80, trajectory_iterations=250,
                                interpolation_buffer_size=interpolation_buffer_size,
                                # BEGIN admit the complete declared native sphere inventory.
                                max_robot_spheres=kinematics.total_spheres,
                                # END allocator and solver workspace limits remain unchanged.
                                maximum_trajectory_solves=2, search_seconds=0.)
        # This command stays identical for N=8 and N=24 native control grids.
        # Increasing solver capacity must not silently alter the requested
        # finger trajectory in skill progress or its numerical tolerance.
        bounds = tracking_command(8, closed, opened)
        report["tracking_command"] = {"kind": "fixed piecewise cubic B-spline in skill progress",
                                      "control_knots": 8, "initial_finger_m": (closed+.875*span).tolist(),
                                      "final_finger_m": (closed+.375*span).tolist(), "numerical_joint_tolerance_m": 1e-5}
        # END native capacity is independent of the fixed tracking command.
        events = (
            PlanningEvent("close", "moving_close", 0., 0, "close", "synthetic tracking command", stop=False),
        )
        hand = PrismaticGripper(
            controller="position_tracking", joint_names=finger_names,
            # BEGIN native event consumption uses explicit device metadata.
            control_timestep_s=CONTROL_TIMESTEP_S,
            # END the continuous tracking relation remains a separate input.
            current_positions=initial[finger_columns].cpu().numpy(), joint_limits=limits,
            joint_bounds=bounds, coupling_matrix=coupling, coupling_offset=coupling_offset,
            sphere_indices=sphere_indices, local_spheres=local_spheres,
            sphere_joint_translation=affine, self_collision_pairs=self_pairs,
            commands=tuple({"event_id": event.event_id, "skill_id": event.skill_id, "occurrence": event.occurrence,
                            "command": event.kind, "progress": event.progress} for event in events),
            source="explicit synthetic position-tracking drive; geometry calibrated against native source FK",
            self_collision_geometry=geometry['self_collision_geometry'])
        # BEGIN fixed analytic distance field for the declared distant cuboid.
        background = FrozenPrimitives(primitives=[{"kind": "cuboid", "pose": [0., 0., -1., 1., 0., 0., 0.],
            "dimensions": [.1, .1, .1]}], owner_ids=("synthetic_far_obstacle",),
            source="fully specified synthetic distant cuboid")
        # END planning and execution serialize this same field.
        coverage = ObservedSpace(np.ones((81, 81, 81), bool), origin=[-2., -2., -2.], voxel_size=.05,
                                 observation_ids=("explicit synthetic known-space grid",), reference_frame="robot_base")
        scene = OnlinePlanningScene(unknown_policy={"outside_regions": "forbid", "conditional_regions": []}, background=background, owners=installations, coverage=coverage, reference_frame="robot_base",
                              robot_sphere_materials=(None,)*kinematics.total_spheres,
                              robot_geometry_source=fixture["robot_geometry_source"],
                              gripper=hand)

        # A separately declared fixture relation requires the tool to move
        # throughout closing. No evidence or stop barrier is inferred from
        # the end of this synthetic tracking window.
        source_progress = np.linspace(0., 1., 5)
        reference_q = initial.cpu().numpy()[None].repeat(len(source_progress), axis=0)
        reference_q[:, 0] += .05*source_progress
        reference_q[:, 1] -= .01*source_progress
        reference_q[:, finger_columns] = bounds(torch.tensor(source_progress))[0].numpy()
        _, reference_position, reference_quaternion = fk(reference_q)
        poses = np.repeat(np.eye(4)[None], len(source_progress), axis=0)
        poses[:, :3, 3] = reference_position
        poses[:, :3, :3] = Rotation.from_quat(reference_quaternion[:, [1, 2, 3, 0]]).as_matrix()
        # BEGIN true tool process and terminal goal are independent of the seed.
        reference = PosePath(source_progress, poses, "synthetic native FK reference")
        problem = PlanningProblem(
            names, initial.cpu().numpy(), np.zeros(len(names)), fixture["tool_link"], "robot_base", np.eye(4), validator.T_frame_tcp,
            (MotionPhase("move_while_closing_reference", reference),),
            events=events, couplings=(JointCoupling(finger_names, coupling, coupling_offset,
                                                   "explicit synthetic equal opening fraction drive contract"),),
            numerical=NumericalTolerance(position_m=.001, rotation_rad=.005),
            pose_events=(PoseEventRequirement(1., np.eye(4), poses[-1], .001, .005,
                "Explicit fixture terminal reach while tracking both fingers"),),
            pose_relations=(PoseProcessRequirement(0., 1., reference, np.zeros((6, 2)),
                "Explicit synthetic requirement to move the tool throughout the closing command"),),
            source_timestamps=2.*source_progress, source_progress=source_progress)
        # END an ordinary reference alone cannot make this experiment pass.
        report["control_timestep_s"] = CONTROL_TIMESTEP_S
        report["budget"] = asdict(budget)
        report["reference_tool_translation_m"] = float(np.linalg.norm(reference_position[-1]-reference_position[0]))
        report["commanded_width_m"] = ((reference_q[[0, -1]][:, finger_columns]-closed) @ np.sign(span)).tolist()
        original_export = planner_module._CuroboPlanner.export
        report["candidate_scene_components"] = []

        def measured_export(planner, native_result):
            """Inspect the one accepted export without changing its result."""
            curve, count = original_export(planner, native_result)
            times = np.linspace(0., curve.duration, 401)
            # BEGIN the joint sample exposes acceleration before progress.
            coordinates, _, _, progress = curve.sample(times)
            # END hand diagnostics use the unchanged shared progress clock.
            with torch.no_grad():
                native_q = device.to_device(coordinates)[None].contiguous()
                native_s = device.to_device(progress)[None]
                state = kinematics.compute_kinematics(JointState.from_position(native_q, joint_names=list(names)))
                tool_p, tool_r = problem.transform_tool_pose(state.tool_poses.position[:, :, tool_index],
                                                            state.tool_poses.quaternion[:, :, tool_index])
                terms = planner._verification_scene.components(problem.transform_robot_spheres(state.robot_spheres),
                    tool_p, tool_r, native_s, native_q, names, trajectory_time=device.to_device(times)[None])
                expected = bounds(native_s)[0]
                report["candidate_scene_components"].append({
                    "maximum_by_component": {key: float(value.max().cpu()) for key, value in terms.items()},
                    "tracking_max_error_m": float((native_q[..., finger_columns]-expected).abs().max().cpu()),
                    "coupling_max_error_m": float((native_q[..., finger_columns] @ device.to_device(coupling.T)-device.to_device(coupling_offset)).abs().max().cpu()),
                })
            return curve, count

        planner_module._CuroboPlanner.export = measured_export
        try:
            result = plan_trajectory(problem, scene, robot, budget=budget, robot_validator=validator)
        finally:
            planner_module._CuroboPlanner.export = original_export
        report.update(status="completed", success=result.success, diagnostics=result.diagnostics)
        if result.trajectory is not None:
            trajectory = result.trajectory
            sample_times = np.linspace(0., trajectory.duration, 401)
            # BEGIN evaluate all reference components at these exact times.
            q, dq, ddq, progress = trajectory.sample(sample_times)
            # END this probe does not finite-difference the saved curve.
            _, p, _ = fk(q)
            width = (q[:, finger_columns]-closed) @ np.sign(span)
            closing = np.diff(width) < -1e-6
            tool_steps = np.linalg.norm(np.diff(p, axis=0), axis=-1)
            expected = bounds(torch.tensor(progress))[0].numpy()
            tracking_error = float(np.abs(q[:, finger_columns]-expected).max())
            coupling_error = float(np.abs(q[:, finger_columns] @ coupling.T-coupling_offset).max())
            simultaneous_motion = float(tool_steps[closing].sum())
            report["exported_motion"] = {
                "duration_s": trajectory.duration, "width_drop_m": float(width[0]-width[-1]),
                "closing_intervals": int(closing.sum()),
                "moving_while_closing_intervals": int(np.count_nonzero(closing & (tool_steps > 1e-6))),
                "tool_translation_during_closing_m": simultaneous_motion,
                "finger_tracking_max_error_m": tracking_error, "drive_coupling_max_error_m": coupling_error,
                "executable_until_s": trajectory.executable_until,
                "next_verification": trajectory.next_verification,
                "events": list(trajectory.events), "conditions": list(trajectory.conditions),
                "residuals": trajectory.residuals,
            }
            np.savez(output / "trajectory.npz", times=trajectory.times, position=trajectory.position,
                     velocity=trajectory.velocity, progress_u=trajectory.progress_u, progress_s=trajectory.progress_s,
                     sample_times=sample_times, sample_tool_position=p, sample_width=width)
            if (width[0]-width[-1] < .5*np.abs(span).sum()-1e-5 or simultaneous_motion < .005
                    or tracking_error > hand.numerical_joint_tolerance+5e-7
                    or coupling_error > problem.numerical.joint+5e-7):
                raise AssertionError("The accepted export must actually track both fingers while the tool moves")
            # BEGIN completion of a synthetic reference cannot invent evidence.
            if (trajectory.next_verification is not None or trajectory.conditions
                    or [(event["skill_id"], event["event_id"]) for event in trajectory.events]
                       != [("moving_close", "close")]):
                raise AssertionError("The explicit close identity must survive without a fabricated evidence barrier")
            # END no terminal command is scheduled outside activation coverage.
        report.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (output / "exception.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--native-knots", type=int, default=8, choices=range(6, 25))
    parser.add_argument("--interpolation-buffer-size", type=int, default=512)
    options = parser.parse_args()
    run(options.output, fixture_path=options.fixture, native_knots=options.native_knots,
        interpolation_buffer_size=options.interpolation_buffer_size)
# END native moving-gripper integration experiment.
