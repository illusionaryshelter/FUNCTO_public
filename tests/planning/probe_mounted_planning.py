"""Plan a small explicit free reach with the recorded robot and native mounts.

This control-chain experiment uses a declared synthetic known box and empty
background. It does not establish feasibility in the recorded RGB-D scene.
The arm planner treats each measured finger coordinate as a rigid-tool model
parameter; the real fingers remain independently actuated and require the
separate executor's actual-state supervision. No physics runs in this script.
"""

# BEGIN bounded mounted-robot planning experiment, using production contracts.
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.execution import ExecutionLimits
from manipulation.planning import PlanningBudget
from manipulation.planning_problem import MotionPhase, PlanningProblem, PosePath
from recorded_task_planning import _json_value, plan_and_record
from robot_fixture import create_fixed_installation_owners
from robot_validation_fixture import build_robot_validator


class DeclaredEmptyBackground:
    """Zero occupied background in this explicitly synthetic experiment only.

    Mounts are separate production CollisionOwners. A zero background query
    neither declares the recorded table/objects absent nor supplies observed
    free-space evidence; the experiment's finite box is labelled separately.
    """

    owner_ids = ()

    def __call__(self, spheres, *, key):
        return spheres[..., 0] * 0.

    def clone_for_rollout(self):
        return DeclaredEmptyBackground()


def read_json(path):
    if Path(path).stat().st_size > 16 * 1024**2:
        raise ValueError("This bounded experiment accepts metadata of at most 16 MiB")
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=_json_value, indent=2, allow_nan=False) + "\n")


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    scope = {
        "experiment": "synthetic mounted-robot free reach; control-chain connectivity only",
        "recorded_environment_validated": False,
        "rgbd_or_mapper_used": False,
        "physical_execution_tested": False,
        "continuous_collision_certified": False,
        "planner_finger_model": "independent measured coordinates locked as rigid-tool parameters",
        "physical_fingers_mechanically_locked": False,
        "actual_finger_supervision": "required separately at every execution step with the original native controller",
        "execution_limits_role": "explicit control experiment bands; not skill calibration",
    }
    report = {"status": "preparing", "scope": scope, "planning_called": False,
              "planning_success": False, "gpu_used": False,
              "fixture": str(args.fixture.resolve()), "feedback": str(args.feedback.resolve()),
              "execution_limits_path": str(args.execution_limits.resolve())}
    torch = None
    try:
        fixture, feedback = read_json(args.fixture), read_json(args.feedback)
        limits = ExecutionLimits.from_mapping(read_json(args.execution_limits))
        full_names = tuple(fixture["joint_names"])
        arm_names, finger_names = tuple(fixture["arm_joint_names"]), tuple(fixture["finger_joint_names"])
        # Preserve the recorded name order and measured derivatives. In
        # particular, actuator commands do not supply either finger position.
        if (tuple(feedback["joint_names"]) != full_names or full_names != arm_names + finger_names
                or tuple(feedback["arm_joint_names"]) != arm_names
                or tuple(feedback["finger_joint_names"]) != finger_names
                or feedback["status"] != "validated" or feedback["reference_frame"] != "world"
                or Path(feedback["provenance"]["episode_path"]).resolve()
                != Path(fixture["episode_path"]).resolve()):
            raise ValueError("Feedback must retain the same validated named robot and source recording")
        actual_q, actual_dq = np.asarray(feedback["start_q"], float), np.asarray(feedback["start_dq"], float)
        delta = np.asarray(args.arm_delta, float)
        if (actual_q.shape != (len(full_names),) or actual_dq.shape != actual_q.shape
                or not np.isfinite(actual_q).all() or not np.isfinite(actual_dq).all()
                or delta.shape != (len(arm_names),) or not np.isfinite(delta).all()
                or not 0 < np.max(np.abs(delta)) <= .05):
            raise ValueError("Provide complete measured q/dq and a nonzero arm delta of at most 0.05 rad per joint")

        validator = build_robot_validator(fixture, reference_frame=feedback["reference_frame"],
            T_reference_base=feedback["base_pose"], tool_frame=feedback["tool_link"],
            T_frame_tcp=feedback["tool_transform"])
        actual = {"joint_names": list(full_names), "arm_joint_names": list(arm_names),
                  "finger_joint_names": list(finger_names), "q": actual_q, "dq": actual_dq,
                  "reference_frame": feedback["reference_frame"], "base_pose": feedback["base_pose"],
                  "tool_link": feedback["tool_link"], "tool_transform": feedback["tool_transform"],
                  "tool_pose": feedback["eef_pose"]}
        initial_check = validator.check_state(actual, stage="measured_planning_entrance", trajectory_time=0.)
        report.update(independent_validator=validator.describe(), initial_actual_check=initial_check,
                      execution_limits=limits.describe(), measured_joint_names=list(full_names),
                      measured_q=actual_q.tolist(), measured_dq=actual_dq.tolist(),
                      source_feedback=feedback["source"], entry_frame=feedback["entry_frame"])
        if not initial_check["passed"]:
            raise ValueError("Independent source-model check rejected the actual measured entrance")
        binding = limits.bind(SimpleNamespace(joint_names=arm_names, events=()),
            joint_names=full_names, arm_joint_names=arm_names, finger_joint_names=finger_names,
            native_joint_ranges=[validator.checker.joint_limits[name] for name in full_names],
            named_velocity_limits=fixture["velocity_limits"])
        finger_bounds = np.asarray(binding["initial_fingers"]["position_bounds"])
        narm = len(arm_names)
        if (np.any(actual_q[narm:] < finger_bounds[:, 0]) or np.any(actual_q[narm:] > finger_bounds[:, 1])
                or np.any(np.abs(actual_dq[narm:]) > np.asarray(binding["initial_fingers"]["maximum_speed"]))):
            raise ValueError("Measured fingers violate the separately supplied execution experiment limits")

        # This is an explicitly hypothetical endpoint configuration, never a
        # replacement feedback snapshot. Pin supplies its reachable TCP target;
        # the normal optimizer remains free to choose the arm configuration.
        target_q = actual_q.copy()
        target_q[:narm] += delta
        target_check = validator.checker.query(joint_names=full_names, joint_positions=target_q,
            T_reference_base=feedback["base_pose"], pairs=validator.pairs, frames=[feedback["tool_link"]])
        report["hypothetical_goal"] = {"arm_delta": delta.tolist(), "full_named_configuration": target_check,
                                       "is_measured_feedback": False}
        if (target_check["joint_limit_violations"] or any(
                row["policy"] == "required" and row["relation"] != "separated" for row in target_check["pairs"])):
            raise ValueError("Independent robot/installation geometry rejected the hypothetical endpoint")
        goal_pose = (np.asarray(target_check["frame_poses"][feedback["tool_link"]])
                     @ np.asarray(feedback["tool_transform"]))
        problem = PlanningProblem(arm_names, actual_q[:narm], actual_dq[:narm],
            feedback["tool_link"], feedback["reference_frame"], feedback["base_pose"], feedback["tool_transform"],
            (MotionPhase("synthetic_free_reach", PosePath(np.array([0., 1.]),
                np.stack((feedback["eef_pose"], goal_pose)), "Pinocchio FK of explicit measured q plus arm delta"),
                kind="free"),))

        config_path = Path(fixture["robot_config_path"])
        if config_path.stat().st_size > 16 * 1024**2:
            raise ValueError("Robot configuration exceeds the bounded metadata budget")
        robot = yaml.safe_load(config_path.read_text())
        kin = robot["robot_cfg"]["kinematics"]
        sphere_count = sum(len(rows) for rows in kin["collision_spheres"].values())
        if (not 1 <= sphere_count <= 480 or sphere_count != fixture["sphere_count"]
                or len(fixture["sphere_materials"]) != sphere_count
                or tuple(kin["cspace"]["joint_names"]) != full_names):
            raise ValueError("This control experiment requires the complete source robot within its 480-sphere budget")
        kin["lock_joints"] = {name: float(actual_q[full_names.index(name)]) for name in finger_names}
        # The source cspace includes all joints; cuRobo removes explicitly
        # locked coordinates itself. Keep its named defaults/weights aligned.
        kin["cspace"]["default_joint_position"] = actual_q.tolist()
        write_json(output / "planning-input.json", {
            "scope": scope, "joint_names": arm_names, "start_q": problem.start_q,
            "start_dq": problem.start_dq, "target_tcp_pose": goal_pose,
            "base_pose": problem.base_pose, "tool_link": problem.tool_link,
            "tool_transform": problem.tool_transform, "rigid_finger_parameters": kin["lock_joints"],
            "robot_configuration": robot, "execution_limits": limits.describe()})

        # All source-model, named-state and endpoint checks above are CPU-only.
        # The allocator cap precedes every cuRobo model/query allocation. No
        # second process, package installation or GPU environment modification
        # is launched by this experiment.
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, JointState
        from manipulation.planning_scene import ObservedSpace, PlanningScene

        torch.set_num_threads(2)
        if not torch.cuda.is_available():
            raise RuntimeError("The explicit mounted planning experiment requires CUDA")
        allocator_bytes = 2 * 1024**3
        total = torch.cuda.get_device_properties(0).total_memory
        free, _ = torch.cuda.mem_get_info(0)
        if total < allocator_bytes or free < allocator_bytes:
            raise RuntimeError("Insufficient free device memory for the fixed two-GiB experiment budget")
        torch.cuda.set_per_process_memory_fraction(allocator_bytes / total, device=0)
        torch.cuda.reset_peak_memory_stats(0)
        report["gpu_used"] = True
        report["resource_limits"] = {"torch_allocator_bytes": allocator_bytes, "cpu_threads": 2,
                                     "max_robot_spheres": 480, "initial_free_device_bytes": free}
        device = DeviceCfg("cuda:0")
        native = Kinematics(KinematicsCfg.from_robot_yaml_file(deepcopy(robot), device_cfg=device))
        if tuple(native.joint_names) != arm_names or native.total_spheres != sphere_count:
            raise ValueError("Loaded native arm order or sphere count differs from the verified input")
        parameters = native.config.kinematics_config
        indices_by_link = {link: [int(index) for index in
            parameters.get_sphere_index_from_link_name(link).cpu().tolist()]
            for link in kin["collision_link_names"]}
        owners, owner_report = create_fixed_installation_owners(fixture, device_cfg=device,
            T_output_base=problem.base_pose, sphere_indices_by_link=indices_by_link)
        # The box is specified by this synthetic experiment, not inferred from
        # an RGB-D image or filled around a failed candidate. Its finite border
        # still uses the production whole-sphere unknown-space constraint.
        origin = problem.base_pose[:3, 3] - 2.
        coverage = ObservedSpace(np.ones((81, 81, 81), bool), origin=origin, voxel_size=.05,
            observation_ids=("explicit synthetic known box; no recorded observation evidence",),
            reference_frame=problem.reference_frame)
        scene = PlanningScene(background=DeclaredEmptyBackground(), owners=owners, coverage=coverage,
            reference_frame=problem.reference_frame, robot_sphere_materials=(None,) * sphere_count,
            robot_geometry_source=fixture["robot_geometry_source"], has_gripper=False)
        scene.observation_report = {"scope": scope, "synthetic_known_box_origin_m": origin.tolist(),
                                    "synthetic_known_box_shape_zyx": [81, 81, 81], "voxel_size_m": .05}
        with torch.no_grad():
            state = native.compute_kinematics(JointState.from_position(
                device.to_device(problem.start_q)[None, None].contiguous(), joint_names=list(arm_names)))
            tool_index = native.tool_frames.index(problem.tool_link)
            position, quaternion = problem.transform_tool_pose(
                state.tool_poses.position[:, :, tool_index], state.tool_poses.quaternion[:, :, tool_index])
            terms = scene.components(problem.transform_robot_spheres(state.robot_spheres), position, quaternion,
                device.to_device([[0.]]), device.to_device(problem.start_q)[None, None], arm_names)
            report["initial_scene_components"] = {name: float(value.max().cpu()) for name, value in terms.items()}
        report.update(fixed_installation_owners=owner_report, scene=scene.describe())
        # Queries own their buffers; this temporary FK model is no longer
        # needed once its actual sphere ownership has been bound to the owners.
        del native, state, parameters, position, quaternion, terms
        # This newly constructed short-reach problem owns its own duration;
        # a 0.05-s native step keeps the resulting physical probe bounded. No
        # saved skill timing or solver feasibility tolerance is changed.
        # Reserve 256 export slots, 384 validation points and 16 refinements
        # for this two-anchor experiment. At most 181 native export points
        # require 361 validation samples (each interval is subdivided twice),
        # plus the original anchors. Exceeding this budget still rejects.
        # The production estimator accounts for every independent scene copy:
        # 416 robot spheres, three mounts and the finite coverage grid require
        # about 447 MiB here, within both 512 MiB and the 2-GiB allocator cap.
        budget = PlanningBudget(knots=16, ik_seeds=2, trajectory_seeds=1,
            ik_iterations=80, trajectory_iterations=100, maximum_trajectory_dt=.05,
            interpolation_buffer_size=256, max_validation_points=384, max_refinement_points=16,
            maximum_trajectory_solves=1, search_seconds=0., max_robot_spheres=480,
            max_scene_workspace_bytes=512 * 1024**2)
        report["planning_called"] = True
        result = plan_and_record(problem, scene, robot, output / "planning", budget=budget,
            provenance={"scope": scope, "fixture": str(args.fixture.resolve()),
                        "feedback": str(args.feedback.resolve()), "entry_frame": feedback["entry_frame"],
                        "goal_source": "Pinocchio FK of measured arm q plus explicit CLI delta",
                        "arm_delta": delta.tolist(), "rigid_finger_parameters": kin["lock_joints"],
                        "fixed_installation_owners": owner_report}, execution_limits=limits.describe())
        report.update(status="completed", planning_success=bool(result.success),
                      accepted_trajectory=(str(output / "planning" / "trajectory.npz")
                                           if result.success and result.trajectory is not None else None))
        return bool(result.success)
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                      error=f"{type(error).__name__}: {error}")
        (output / "exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if torch is not None and report["gpu_used"]:
            report.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                          torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        write_json(output / "report.json", report)
        print(json.dumps({"status": report["status"], "planning_success": report["planning_success"],
                          "report": str(output / "report.json"),
                          "accepted_trajectory": report.get("accepted_trajectory")}, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--feedback", required=True, type=Path)
    parser.add_argument("--execution-limits", required=True, type=Path)
    parser.add_argument("--arm-delta", required=True, type=float, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    if not run(parser.parse_args()):
        raise SystemExit(1)
# END bounded mounted-robot planning experiment, using production contracts.
