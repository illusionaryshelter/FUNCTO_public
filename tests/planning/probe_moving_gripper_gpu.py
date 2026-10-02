"""Bounded native Franka planning while a tracking gripper closes in motion.

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


def tracking_command(knots, initial=.035, final=.015):
    """An explicit smooth native-coordinate command, parameterized by progress.

    The installed SciPy spline implementation supplies polynomial coefficients.
    Four initial constants and three final replicas give a realizable cubic
    with zero endpoint velocity/acceleration. Its active domain is [1,N+3],
    matching the planner's declared first/last progress pauses. This command
    is fixed before solving; it is not a width inferred from a binary close.
    """
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
        command = value[..., None].expand(*progress.shape, 2)
        return command, command

    return bounds


def run(output, *, native_knots=8, interpolation_buffer_size=512):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"scope": "native nine-DOF Franka, synthetic known world and tracking controller",
              "production_environment_validation": False,
              "actual_holding_evidence": False, "status": "running"}
    try:
        import torch
        from curobo.content import get_robot_configs_path
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, JointState
        from curobo._src.util.config_io import load_yaml, join_path
        from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
        from curobo._src.geom.types import Cuboid, SceneCfg
        from manipulation.planning import PlanningBudget, plan_trajectory
        from manipulation import planning as planner_module
        from manipulation.planning_problem import (
            JointCoupling, MotionPhase, NumericalTolerance, PlanningEvent, PlanningProblem,
            PosePath, PoseEventRequirement, PoseProcessRequirement)
        from manipulation.planning_scene import (
            CuroboSphereQuery, ObservedSpace, PlanningScene, PrismaticGripper)

        if not torch.cuda.is_available():
            raise RuntimeError("Explicit GPU access is required for this bounded experiment")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        device = DeviceCfg("cuda:0")
        robot = deepcopy(load_yaml(join_path(get_robot_configs_path(), "franka.yml")))
        robot["robot_cfg"]["kinematics"]["lock_joints"] = None
        # Native configuration loading mutates extra-link dictionaries into
        # LinkParams. Retain the declared YAML input for the planning request.
        kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(deepcopy(robot), device_cfg=device))
        names = tuple(kinematics.joint_names)
        finger_names = ("panda_finger_joint1", "panda_finger_joint2")
        finger_columns = [names.index(name) for name in finger_names]
        if len(names) != 9:
            raise AssertionError("The actual Franka finger joints must remain active solver coordinates")
        parameters = kinematics.config.kinematics_config
        sphere_indices = tuple(int(index) for name in ("panda_leftfinger", "panda_rightfinger")
                               for index in parameters.get_sphere_index_from_link_name(name).cpu().tolist())
        if not sphere_indices:
            raise AssertionError("The native finger sphere geometry must be present")
        native_pairs = kinematics.get_self_collision_config().collision_pairs.cpu().numpy()
        self_pairs = tuple(tuple(int(index) for index in pair) for pair in native_pairs
                           if set(pair) & set(sphere_indices))
        if not self_pairs:
            raise AssertionError("The actual model must retain enabled finger/robot self-collision pairs")
        limits = parameters.joint_limits.position[:, finger_columns].T.cpu().numpy()
        initial = kinematics.default_joint_position.reshape(-1).clone()
        # The synthetic measured hand begins inside its mechanical interval.
        # Starting exactly at the 0.04 m URDF limit makes native cubic roundoff
        # a limit violation before any action; no physical tolerance is changed.
        initial[finger_columns] = .035

        def fk(coordinates):
            """Copy native reusable FK buffers before another query can overwrite them."""
            coordinates = torch.as_tensor(coordinates, dtype=torch.float32, device=device.device)
            state = kinematics.compute_kinematics(JointState.from_position(
                coordinates[None].contiguous(), joint_names=list(names)))
            return (state.robot_spheres[0].detach().cpu().numpy().copy(),
                    state.tool_poses.position[0, :, 0].detach().cpu().numpy().copy(),
                    state.tool_poses.quaternion[0, :, 0].detach().cpu().numpy().copy())

        # Derive the local affine finger model from actual native FK, then
        # verify it at unseen arm poses and widths. No assumed sphere order,
        # hand frame, left/right sign or dataset-specific geometry enters it.
        calibration = initial.cpu().numpy()[None].repeat(3, axis=0)
        calibration[:, finger_columns] = 0.
        calibration[1, finger_columns[0]] = .01
        calibration[2, finger_columns[1]] = .01
        spheres, position, quaternion = fk(calibration)
        local = np.stack([Rotation.from_quat(q[[1, 2, 3, 0]]).inv().apply(row[:, :3]-p)
                          for row, p, q in zip(spheres, position, quaternion)])[:, sphere_indices]
        local_spheres = np.c_[local[0], spheres[0, sphere_indices, 3]]
        affine = np.stack(((local[1]-local[0])/.01, (local[2]-local[0])/.01), -1)
        check = initial.cpu().numpy()[None].repeat(3, axis=0)
        check[:, 0] += [0., .025, .05]
        check[:, finger_columns] = np.array([.04, .03, .02])[:, None]
        check_spheres, check_position, check_quaternion = fk(check)
        measured_local = np.stack([Rotation.from_quat(q[[1, 2, 3, 0]]).inv().apply(row[:, :3]-p)
                                   for row, p, q in zip(check_spheres, check_position, check_quaternion)])[:, sphere_indices]
        expected_local = local_spheres[None, :, :3]+np.einsum("ncj,hj->hnc", affine, check[:, finger_columns])
        geometry_error = float(np.abs(measured_local-expected_local).max())
        if geometry_error > 2e-6:
            raise AssertionError("The affine gripper proxy must reproduce the real native FK geometry")
        report["native_hand"] = {
            "active_joint_names": list(names), "finger_joint_names": list(finger_names),
            "finger_sphere_indices": list(sphere_indices), "enabled_self_collision_pairs": len(self_pairs),
            "joint_limits_m": limits.tolist(), "affine_native_fk_max_error_m": geometry_error,
            "drive_coupling": "q_finger1 - q_finger2 = 0 (explicit synthetic symmetric drive contract)",
            "urdf_axes": [[0, 1, 0], [0, -1, 0]],
        }

        # BEGIN vary solver capacity while preserving the physical command.
        budget = PlanningBudget(ik_seeds=2, trajectory_seeds=1, knots=native_knots,
                                ik_iterations=80, trajectory_iterations=250,
                                interpolation_buffer_size=interpolation_buffer_size,
                                maximum_trajectory_solves=2, search_seconds=0.)
        # This command stays identical for N=8 and N=24 native control grids.
        # Increasing solver capacity must not silently alter the requested
        # finger trajectory in skill progress or its numerical tolerance.
        bounds = tracking_command(8)
        report["tracking_command"] = {"kind": "fixed piecewise cubic B-spline in skill progress",
                                      "control_knots": 8, "initial_finger_m": .035,
                                      "final_finger_m": .015, "numerical_joint_tolerance_m": 1e-5}
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
            joint_bounds=bounds, coupling_matrix=[[1., -1.]], coupling_offset=[0.],
            sphere_indices=sphere_indices, local_spheres=local_spheres,
            sphere_joint_translation=affine, self_collision_pairs=self_pairs,
            commands=tuple({"event_id": event.event_id, "skill_id": event.skill_id,
                            "command": event.kind, "progress": event.progress} for event in events),
            source="explicit synthetic position-tracking drive; geometry calibrated against native Franka FK")
        native_scene = SceneCollision.from_config(SceneCollisionCfg(
            device_cfg=device, scene_model=SceneCfg(cuboid=[Cuboid(
                name="synthetic_far_obstacle", pose=[0., 0., -1., 1., 0., 0., 0.], dims=[.1, .1, .1])]),
            cache={"cuboid": 1}))
        background = CuroboSphereQuery(native_scene, owner_ids=("synthetic_far_obstacle",),
                                       source="fully specified synthetic distant cuboid")
        coverage = ObservedSpace(np.ones((81, 81, 81), bool), origin=[-2., -2., -2.], voxel_size=.05,
                                 observation_ids=("explicit synthetic known-space grid",), reference_frame="robot_base")
        scene = PlanningScene(background=background, owners=(), coverage=coverage, reference_frame="robot_base",
                              robot_sphere_materials=(None,)*kinematics.total_spheres,
                              robot_geometry_source="full native Franka sphere model with active fingers",
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
            names, initial.cpu().numpy(), np.zeros(len(names)), "panda_hand", "robot_base", np.eye(4), np.eye(4),
            (MotionPhase("move_while_closing_reference", reference),),
            events=events, couplings=(JointCoupling(finger_names, np.array([[1., -1.]]), np.array([0.]),
                                                   "explicit synthetic symmetric drive contract"),),
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
        report["commanded_width_m"] = [float(2*reference_q[0, finger_columns[0]]),
                                       float(2*reference_q[-1, finger_columns[0]])]
        original_export = planner_module._CuroboPlanner.export
        report["candidate_scene_components"] = []

        def measured_export(planner, native_result):
            """Measure actual exported candidates, including rejected ones."""
            curve, count = original_export(planner, native_result)
            times = np.linspace(0., curve.duration, 401)
            coordinates, _, progress = curve.sample(times)
            with torch.no_grad():
                native_q = device.to_device(coordinates)[None].contiguous()
                native_s = device.to_device(progress)[None]
                state = kinematics.compute_kinematics(JointState.from_position(native_q, joint_names=list(names)))
                terms = scene.components(state.robot_spheres, state.tool_poses.position[:, :, 0],
                                         state.tool_poses.quaternion[:, :, 0], native_s, native_q, names)
                expected = bounds(native_s)[0]
                report["candidate_scene_components"].append({
                    "maximum_by_component": {key: float(value.max().cpu()) for key, value in terms.items()},
                    "tracking_max_error_m": float((native_q[..., finger_columns]-expected).abs().max().cpu()),
                    "coupling_max_error_m": float((native_q[..., finger_columns[0]]-native_q[..., finger_columns[1]]).abs().max().cpu()),
                })
            return curve, count

        planner_module._CuroboPlanner.export = measured_export
        try:
            result = plan_trajectory(problem, scene, robot, budget=budget)
        finally:
            planner_module._CuroboPlanner.export = original_export
        report.update(status="completed", success=result.success, diagnostics=result.diagnostics)
        if result.trajectory is not None:
            trajectory = result.trajectory
            sample_times = np.linspace(0., trajectory.duration, 401)
            q, dq, progress = trajectory.sample(sample_times)
            _, p, _ = fk(q)
            width = q[:, finger_columns].sum(-1)
            closing = np.diff(width) < -1e-6
            tool_steps = np.linalg.norm(np.diff(p, axis=0), axis=-1)
            expected = bounds(torch.tensor(progress))[0].numpy()
            tracking_error = float(np.abs(q[:, finger_columns]-expected).max())
            coupling_error = float(np.abs(q[:, finger_columns[0]]-q[:, finger_columns[1]]).max())
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
            if (width[0]-width[-1] < .035 or simultaneous_motion < .005
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
    parser.add_argument("--native-knots", type=int, default=8, choices=range(6, 25))
    parser.add_argument("--interpolation-buffer-size", type=int, default=512)
    options = parser.parse_args()
    run(options.output, native_knots=options.native_knots,
        interpolation_buffer_size=options.interpolation_buffer_size)
# END native moving-gripper integration experiment.
