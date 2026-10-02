"""Bounded native FK/model binding for current occupied-space evidence.

The finger-frame solid and current feedback are explicitly synthetic evidence
fixtures. They do not assert that a real Panda collision mesh is physically
solid. No trajectory is optimized or executed. The checks exercise the real
native q0 FK, occupied-cell adapter, planner model binding, and rejection of
a changed same-name/same-count collision model. Run serially with GPU probes.
"""

# BEGIN bounded current-solid native model-binding experiment.
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

# BEGIN resolve imports from this FUNCTO checkout, independent of its directory name.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# END project-local imports.


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"scope": "native current FK/model binding with a declared synthetic finger-frame solid",
              "real_robot_occupied_geometry_verified": False,
              "production_environment_validation": False, "status": "running"}
    try:
        import torch
        import trimesh
        from curobo.content import get_robot_configs_path
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, JointState
        from curobo._src.util.config_io import load_yaml, join_path
        from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
        from curobo._src.geom.types import Cuboid, SceneCfg
        from manipulation.planning import PlanningBudget, _CuroboPlanner
        from manipulation.planning_observation import observed_space_from_rgbd
        from manipulation.planning_problem import MotionPhase, PlanningProblem, PosePath
        from manipulation.planning_scene import CuroboSphereQuery, PlanningScene

        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        device = DeviceCfg("cuda:0")
        robot = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
        # The planner itself requests panda_hand. The occupied body deliberately
        # uses another already-selected collision link to exercise extra-frame
        # validation without extending the robot tree or changing the IK tool.
        kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(
            deepcopy(robot), tool_frames=["panda_hand", "panda_leftfinger"], device_cfg=device))
        initial = kinematics.default_joint_position.reshape(-1).clone()
        state = kinematics.compute_kinematics(JointState.from_position(
            initial[None, None].contiguous(), joint_names=kinematics.joint_names))
        pose = np.eye(4)
        pose[:3, 3] = state.tool_poses.position[0, 0, 0].detach().cpu().numpy()
        pose[:3, :3] = Rotation.from_quat(
            state.tool_poses.quaternion[0, 0, 0].detach().cpu().numpy()[[1, 2, 3, 0]]).as_matrix()
        solid_center = state.tool_poses.position[0, 0, 1].detach().cpu().numpy().copy()
        view_id, source = "synthetic_synchronized_snapshot", "native fixed-tool Franka fixture"
        # Current synthetic foreground depth occludes the entire local body
        # region. Depth therefore cannot certify its hidden interior or air.
        foreground_depth = max(.01, float(solid_center[2]-.2))
        views = {view_id: {"depth": np.full((16, 16), foreground_depth),
                           "K": np.array([[10., 0., 7.5], [0., 10., 7.5], [0., 0., 1.]]),
                           "T_output_camera": np.eye(4), "reference_frame": "robot_base"}}
        current = {"state": "current_observed", "reference_frame": "robot_base", "observation_ids": [view_id],
                   "source": "explicit synthetic current feedback", "max_point_error_m": 0.,
                   "kinematics": kinematics, "joint_names": list(kinematics.joint_names),
                   "joint_position": initial.cpu().numpy(), "T_output_robot_base": np.eye(4),
                   "robot_geometry_source": source}
        solids = {"synthetic_finger_interior": {
            "mesh": trimesh.creation.box(extents=[.08, .08, .08]),
            "occupancy_kind": "verified_inner_solid",
            "geometry_source": "declared synthetic physical finger-frame cube; not actual Panda occupancy evidence",
            "max_geometry_error_m": 0., "body_kind": "robot_link", "body_id": "panda_leftfinger",
            "T_body_solid": np.eye(4)}}
        coverage = observed_space_from_rgbd(
            views, origin=solid_center-.15, shape=(31, 31, 31), voxel_size=.01,
            depth_error_m=0., reference_frame="robot_base", depth_range_m=(.005, 3.),
            known_solids=solids, current_robot=current)
        center = solid_center
        balls = torch.tensor([[[[*center, .005], [*(center+np.array([.09, 0., 0.])), .005]]]], dtype=torch.float32)
        coverage_cost = coverage.violation(balls)[0, 0].tolist()
        if coverage_cost[0] != 0 or coverage_cost[1] <= 0:
            raise AssertionError("Only the evidenced current solid interior may become known")
        native_scene = SceneCollision.from_config(SceneCollisionCfg(
            device_cfg=device, scene_model=SceneCfg(cuboid=[Cuboid(
                name="synthetic_far_obstacle", pose=[0., 0., -1., 1., 0., 0., 0.], dims=[.1, .1, .1])]),
            cache={"cuboid": 1}))
        scene = PlanningScene(
            background=CuroboSphereQuery(native_scene, owner_ids=("synthetic_far_obstacle",), source="synthetic cuboid"),
            owners=(), coverage=coverage, reference_frame="robot_base",
            robot_sphere_materials=(None,)*kinematics.total_spheres,
            robot_geometry_source=source, has_gripper=False)
        # BEGIN this construction-only geometry probe supplies a real source clock.
        # No solve is requested here; the constant path is initialization data
        # for checking exact current-robot occupancy binding, not a task goal.
        problem = PlanningProblem(tuple(kinematics.joint_names), initial.cpu().numpy(), np.zeros(len(initial)),
                                  "panda_hand", "robot_base", np.eye(4), np.eye(4),
                                  (MotionPhase("fixed_fixture", PosePath(np.array([0., 1.]), np.stack((pose, pose)),
                                                                           "synthetic current tool reference")),),
                                  source_timestamps=np.array([0., 1.]), source_progress=np.array([0., 1.]))
        # END no removed phase-kind argument is passed as metadata.
        # Native installed inner iterations are IK=20 and TrajOpt=25. No
        # solve is called, but configuration still requires valid multiples.
        budget = PlanningBudget(ik_seeds=2, trajectory_seeds=1, knots=6, ik_iterations=20,
                                trajectory_iterations=25, maximum_trajectory_solves=1, search_seconds=0.)
        planner = _CuroboPlanner(problem, scene, robot, budget, "cuda:0")
        planner.close()
        del planner
        changed_feedback = deepcopy(problem)
        changed_feedback.start_q[0] += .001
        feedback_rejection = None
        try:
            unexpected = _CuroboPlanner(changed_feedback, scene, robot, budget, "cuda:0")
        except ValueError as error:
            feedback_rejection = str(error)
        else:
            unexpected.close()
            raise AssertionError("Stale occupied evidence must reject changed actual joint feedback")
        if "exact current joint feedback" not in feedback_rejection:
            raise AssertionError(f"Changed feedback was rejected for an unrelated reason: {feedback_rejection}")
        changed = deepcopy(robot)
        spheres = changed["robot_cfg"]["kinematics"]["collision_spheres"]
        first_link = next(link for link, rows in spheres.items() if rows)
        spheres[first_link][0]["radius"] += .002
        rejected = None
        try:
            unexpected = _CuroboPlanner(problem, scene, changed, budget, "cuda:0")
        except ValueError as error:
            rejected = str(error)
        else:
            unexpected.close()
            raise AssertionError("A changed collision model with unchanged names/counts must be rejected")
        if "collision sphere model disagree" not in rejected:
            raise AssertionError(f"Changed model was rejected for an unrelated reason: {rejected}")
        evidence = coverage.occupancy_evidence["robot"]
        report.update(status="passed", current_solid_coverage_violation=coverage_cost,
                      actual_model_binding_passed=True, changed_same_count_model_rejection=rejected,
                      changed_current_feedback_rejection=feedback_rejection,
                      synthetic_foreground_depth_m=foreground_depth,
                      required_robot_links=list(scene.required_robot_links),
                      recorded_sphere_count=evidence["robot_spheres"],
                      recorded_locked_joint_positions=evidence["locked_joint_positions"],
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
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
    run(parser.parse_args().output)
# END bounded current-solid native model-binding experiment.
