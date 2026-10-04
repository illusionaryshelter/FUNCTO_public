"""Bounded formal planning with independent native contact result grading.

The explicit fixture has one arm axis and two distant free fingers. The native
planner, accepted binding, independent Coal validator, executor and MuJoCo all
consume the same model. Contact truth stays in the offline report and never
confirms a runtime effect or releases an evidence dependency.
Run GPU mode serially; --cpu-check validates the native fixture without CUDA.
"""

# BEGIN explicit finite contact integration fixture and formal production path.
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import combinations
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from contact_fixture import native_model_text

NAMES = ("arm_q", "left_q", "right_q")
PAD_WIDTH = .01
DT = .002


def build_native(output):
    """Declare geometry, actuator policies and actual feedback before planning."""
    from robosuite.utils.binding_utils import MjSim
    from manipulation.execution import ExecutionLimits, FixedFingerTargets
    from manipulation.robosuite_execution import RobosuiteExecutionBackend
    from manipulation.robot_geometry_check import NativePrimitive, PairQuery, RobotGeometryChecker
    from manipulation.robot_validation import RobotStateValidator

    xml, urdf_text = native_model_text(pad_width=PAD_WIDTH)
    (output / "robot.xml").write_text(xml)
    urdf = output / "robot.urdf"
    urdf.write_text(urdf_text)
    model = mujoco.MjModel.from_xml_string(xml)
    sim = MjSim(model)
    sim.data._data.qpos[:] = [0., .01, -.01]
    sim.data._data.ctrl[:] = [0., .01, -.01]
    mujoco.mj_forward(model, sim.data._data)
    backend = RobosuiteExecutionBackend(sim, base_link="base", tcp_site="tcp",
        arm_joint_names=NAMES[:1], finger_joint_names=NAMES[1:], arm_kp=500., arm_kd=20.,
        velocity_limits=dict.fromkeys(NAMES, 1.), source="explicit native press fixture actual feedback")
    checker = RobotGeometryChecker(urdf, base_frame="base", geometries=[
        NativePrimitive("pad", "arm", "box", (PAD_WIDTH, PAD_WIDTH, .1), np.eye(4), "explicit native thin pad"),
        NativePrimitive("left_pad", "left", "box", (.02, .02, .02), np.eye(4), "explicit native left finger"),
        NativePrimitive("right_pad", "right", "box", (.02, .02, .02), np.eye(4), "explicit native right finger")])
    validator = RobotStateValidator(checker, joint_names=NAMES, arm_joint_names=NAMES[:1],
        finger_joint_names=NAMES[1:], pairs=tuple(PairQuery(a, b, "required", "explicit separated fixture solids")
            for a, b in combinations(checker.geometry_names, 2)),
        named_velocity_limits=dict.fromkeys(NAMES, 1.), reference_frame="world", T_reference_base=np.eye(4),
        tool_frame="tcp", T_frame_tcp=np.eye(4), fk_translation_tolerance_m=1e-8,
        fk_rotation_tolerance_rad=1e-8, source="independent exact native fixture dimensions and complete pair policy")
    targets = FixedFingerTargets(NAMES[1:], "explicit unchanged free-finger actuator goals", (
        {"skill_id": "press_fixture", "event_id": "hold", "kind": "hold", "positions": [.01, -.01]},))
    interval = {"position_bounds": [[0., .04], [-.04, 0.]], "maximum_speed": [1., 1.]}
    limits = ExecutionLimits(source="explicit fixture 3 mm contact tracking band and native hardware limits",
        joint_names=NAMES, arm_joint_names=NAMES[:1], finger_joint_names=NAMES[1:],
        arm_position_error=[.003], arm_velocity_error=[.2], tcp_position_error_m=.003,
        tcp_rotation_error_rad=.001, stop_arm_velocity=[.01], stop_finger_velocity=[.01, .01],
        initial_fingers=interval, events=({"skill_id": "press_fixture", "event_id": "hold", **interval},),
        observation_policy=None)
    covers = {"arm": [{"center": [0., 0., 0.], "radius": float(np.linalg.norm([PAD_WIDTH/2, PAD_WIDTH/2, .05]))}],
              "left": [{"center": [0., 0., 0.], "radius": float(np.sqrt(3)*.01)}],
              "right": [{"center": [0., 0., 0.], "radius": float(np.sqrt(3)*.01)}]}
    robot = {"robot_cfg": {"load_dynamics": False, "kinematics": {
        "format_version": 2.0, "urdf_path": str(urdf.resolve()), "asset_root_path": str(output.resolve()),
        "base_link": "base", "tool_frames": ["tcp"], "use_global_cumul": True,
        "collision_link_names": list(covers), "collision_spheres": covers, "collision_sphere_buffer": 0.,
        "self_collision_buffer": dict.fromkeys(covers, 0.), "self_collision_ignore": {},
        "lock_joints": {"left_q": .01, "right_q": -.01},
        "cspace": {"joint_names": list(NAMES), "default_joint_position": [0., .01, -.01],
            "cspace_distance_weight": [1.]*3, "null_space_weight": [1.]*3,
            "max_acceleration": 1., "max_jerk": 50.}}}}
    return backend, validator, targets, limits, robot


def build_problem(backend):
    """Declare a terminal contact-band goal and independently graded contact effect."""
    from manipulation.planning_problem import MotionPhase, NumericalTolerance, PlanningEvent, PlanningProblem, PoseEventRequirement, PosePath
    initial = backend.proprioception()
    end = initial["tool_pose"].copy()
    end[2, 3] = -.001
    reference = PosePath(np.array([0., 1.]), np.stack((initial["tool_pose"], end)),
                         "explicit straight reference to the finite contact band")
    return PlanningProblem(NAMES[:1], initial["q"][:1], initial["dq"][:1], "tcp", "world",
        initial["base_pose"], np.eye(4), (MotionPhase("press_reference", reference),),
        events=(PlanningEvent("hold", "press_fixture", 0., 0, "hold", "explicit free-finger hold"),
                PlanningEvent("contact_effect", "press_fixture", 1., 1, "effect", "independently evaluate physical contact")),
        conditions=({"condition_id": "press_contact", "skill_id": "press_fixture", "event_id": "contact_effect",
                     "effect_requirement": "positive native pad/slab contact evaluated independently in the terminal contact band"},),
        pose_events=(PoseEventRequirement(1., np.eye(4), end, .0015, .001,
                     "explicit fixture contact-band goal: TCP z in [-.0025,.0005] m"),),
        numerical=NumericalTolerance(position_m=1e-4, rotation_rad=1e-4),
        source_timestamps=np.array([0., 1.]), source_progress=np.array([0., 1.]))


def build_scene(device, robot, backend, targets, problem):
    """Build native queries and derive every sphere identity from native FK metadata."""
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
    from curobo._src.geom.types import Cuboid, SceneCfg
    from manipulation.planning_problem import PosePath
    from manipulation.planning_scene import CollisionOwner, ContactRule, CuroboSphereQuery, ObservedSpace, PlanningScene, PrismaticGripper
    from run_recorded_skill import finger_trajectory_factory

    kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(deepcopy(robot), device_cfg=device))
    config = kinematics.config.kinematics_config
    sphere_ids = {name: tuple(int(i) for i in config.get_sphere_index_from_link_name(name).cpu().tolist())
                  for name in ("arm", "left", "right")}
    if any(len(indices) != 1 for indices in sphere_ids.values()) or kinematics.total_spheres != 3:
        raise AssertionError("The fixture must retain exactly its three declared outward sphere covers")
    finger_ids = sphere_ids["left"] + sphere_ids["right"]
    actual = backend.proprioception()
    factory_feedback = {**actual, "start_q": actual["q"], "start_dq": actual["dq"], "sim_time_s": actual["sim_time"]}
    factory = finger_trajectory_factory(problem, factory_feedback, targets, timestep=DT,
        ctrlrange=backend.finger_ctrlrange, position_limits={"left_q": [0., .04], "right_q": [-.04, 0.]},
        velocity_limits={"left_q": 1., "right_q": 1.}, acceleration_limits={"left_q": 1., "right_q": 1.})
    # The fingers remain remote from every contact. This is an explicit free
    # actuator nominal model, separately sourced from the same constant command.
    def free_finger_model(duration, progress_map):
        model = factory(duration, progress_map)
        return replace(model, source="explicit nominal free-finger constant-position model; actual remains independently measured")
    affine = np.zeros((2, 3, 2)); affine[0, 1, 0] = affine[1, 1, 1] = 1.
    hand = PrismaticGripper(controller="timed_reference", joint_names=NAMES[1:],
        current_positions=actual["q"][1:], joint_limits=[[0., .04], [-.04, 0.]], joint_bounds=None,
        coupling_matrix=np.empty((0, 2)), coupling_offset=[], sphere_indices=finger_ids,
        local_spheres=[[0., .15, .15, np.sqrt(3)*.01], [0., -.15, .15, np.sqrt(3)*.01]],
        sphere_joint_translation=affine, self_collision_pairs=tuple(combinations(range(3), 2)),
        commands=({"skill_id": "press_fixture", "event_id": "hold", "occurrence": 0, "command": "hold", "progress": 0.},),
        source="explicit independent free finger actuators", control_timestep_s=DT,
        finger_trajectory_factory=factory, nominal_finger_trajectory_factory=free_finger_model)
    def query(name, position, dimensions):
        native = SceneCollision.from_config(SceneCollisionCfg(device_cfg=device,
            scene_model=SceneCfg(cuboid=[Cuboid(name=name, pose=[*position, 1., 0., 0., 0.], dims=dimensions)]),
            cache={"cuboid": 1}))
        return CuroboSphereQuery(native, owner_ids=(name,), source="explicit complete native fixture solid")
    background = query("far_background", [0., 0., -1.], [.1, .1, .1])
    owner = CollisionOwner("target", query("target", [0., 0., -.05], [.6, .6, .1]),
        PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "explicit static target pose"),
        "explicit complete fixture slab", True)
    half = PAD_WIDTH/2
    rule = ContactRule("press", "robot", "target", (0., 1.), sphere_ids["arm"], "press_face",
        [[-half,-half,0.],[half,-half,0.],[half,half,0.],[-half,half,0.]], [0.,0.,-1.],
        [0.,0.,0.], np.eye(3), [.1,.1], (-.0025,.0025), .01, .003, .003, .1,
        "declared finite pressing face and target patch")
    materials = [None]*3
    materials[sphere_ids["arm"][0]] = "press_face"
    coverage = ObservedSpace(np.ones((41,41,41), bool), origin=[-.5,-.5,-.5], voxel_size=.025,
        observation_ids=("explicit complete known fixture volume",), reference_frame="world")
    return PlanningScene(background=background, owners=(owner,), coverage=coverage, reference_frame="world",
        robot_sphere_materials=materials, robot_geometry_source="three exact outward native box covers",
        contacts=(rule,), gripper=hand, native_geometry_materials={"pad": "press_face"})


# BEGIN native contact truth is an output-only sample, never an observer value.
def _record_contact_truth(model, data, record, *, pad_id, slab_id, goal, contact_log):
    """Append an independent result diagnostic and deliberately return no evidence."""
    from manipulation.execution import _actual_pose_goal

    force_peak = 0.
    for index in range(data.ncon):
        contact = data.contact[index]
        if ({int(contact.geom1), int(contact.geom2)} == {pad_id, slab_id}
                and contact.exclude == 0 and contact.efc_address >= 0):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            force_peak = max(force_peak, abs(float(force[0])))
    measured = _actual_pose_goal(goal, record["feedback"], None)
    contact_log.append({"timestamp_s": float(data.time), "elapsed_s": record["elapsed"],
        "curve_time_s": record["curve_time"], "normal_force_n": force_peak,
        "goal_measurement": measured, "value": "T" if force_peak > .01 and measured["value"] == "T" else "F",
        "scope": "Independent native truth only; no runtime observation or completion label is returned"})
# END the step callback cannot supply an effect to the executor.


def run(output, *, cpu_check=False):
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {"status": "running", "scope": "formal bounded synthetic contact fixture", "cpu_check": cpu_check,
              "dimensions": {"arm_dof": 1, "finger_dof": 2, "robot_spheres": 3, "scene_cuboids": 2,
                             "max_physics_steps": 1000, "max_hold_steps": 100}}
    try:
        backend, validator, targets, limits, robot = build_native(output)
        problem = build_problem(backend)
        initial = backend.proprioception()
        state = validator.checker.query(joint_names=NAMES, joint_positions=initial["q"],
            T_reference_base=initial["base_pose"], pairs=validator.pairs, frames=("tcp",))
        np.testing.assert_allclose(state["frame_poses"]["tcp"], initial["tool_pose"], atol=1e-10)
        report["initial_feedback"] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                      for key, value in initial.items()}
        if cpu_check:
            report["status"] = "cpu_fixture_checked"
            return report
        import torch
        from curobo.types import DeviceCfg
        from manipulation.planning import PlanningBudget, plan_trajectory
        from manipulation.execution import TrajectoryExecutor, pose_goal_schedule
        from manipulation.execution_binding import rebind_execution_result
        if not torch.cuda.is_available():
            raise RuntimeError("This formal native probe requires explicitly authorized CUDA access")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        device = DeviceCfg("cuda:0")
        scene = build_scene(device, robot, backend, targets, problem)
        budget = PlanningBudget(ik_seeds=2, trajectory_seeds=1, knots=6, ik_iterations=80,
            trajectory_iterations=200, maximum_trajectory_dt=.1, maximum_trajectory_solves=2,
            search_seconds=0., max_robot_spheres=64, max_validation_points=256)
        report["budget"] = asdict(budget)
        # BEGIN the exact three-box source model gates planning and execution.
        result = plan_trajectory(problem, scene, robot, robot_validator=validator, budget=budget)
        # END sphere covers remain optimizer geometry, not a source-model substitute.
        report["planning"] = {"success": result.success, "diagnostics": result.diagnostics}
        if result.request_ledger is not None:
            (output/"request-ledger.json").write_text(json.dumps(result.request_ledger, indent=2, allow_nan=False)+"\n")
        if not result.success:
            raise AssertionError("Formal planner did not accept this declared contact fixture")
        result = rebind_execution_result(result, controller_policy=backend.describe()["controller_policy"],
            result_only_goal_ids=[], finger_targets=targets, execution_limits=limits,
            source="Explicit synthetic contact probe device contract; original pose goal remains online")
        contact_log = []
        pad_id, slab_id = backend.native_model.geom("pad").id, backend.native_model.geom("target_slab").id
        def collect_result(model, data, record):
            _record_contact_truth(model, data, record, pad_id=pad_id, slab_id=slab_id,
                                  goal=problem.pose_events[0], contact_log=contact_log)
        execution = TrajectoryExecutor(backend, geometry_checker=validator).execute(result,
            finger_targets=targets, initial_feedback=initial, execution_limits=limits, output=output/"execution",
            step_callback=collect_result, max_physics_steps=1000, max_hold_steps=100)
        report["execution"] = {key: execution[key] for key in
            ("status","effect_result","effects","executed_steps","geometry_preflight","evaluated_events","completed_goal_ids")}
        report["contacts"] = contact_log
        _, due = pose_goal_schedule(result.trajectory, problem.pose_events[0].progress, DT)
        checkpoint = next((row for row in contact_log if row["elapsed_s"] >= due), None)
        report["independent_contact_result"] = {"value": "U" if checkpoint is None else checkpoint["value"],
            "device_due_time_s": due, "measurement": checkpoint,
            "runtime_effect_value": execution["effect_result"]}
        report["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated()/1024**2
        report["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved()/1024**2
        if (execution["status"] != "curve_finished" or execution["effect_result"] != "U"
                or any(row.get("condition_id") == "press_contact" and row.get("value") == "T"
                       for row in execution["effect_history"])):
            raise AssertionError("Independent truth cannot become online effect evidence or release a dependency")
        if checkpoint is None or checkpoint["value"] != "T":
            raise AssertionError("The original contact checkpoint lacks a passing independent native result")
        report["status"] = "independent_contact_result_observed"
        return report
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_wall_s=time.monotonic()-start,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), cpu_check=args.cpu_check)
# END the report preserves formal planning, physical feedback and effects separately.
